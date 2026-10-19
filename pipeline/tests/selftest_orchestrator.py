"""Offline self-test for the P2.2 orchestrator (no device, no live match).

Sections:
  1. mapping solver: direct / mirrored / noisy / disagreement -> fresh fit
  2. grid_to_world bounds
  3. simulated match, direct orientation   (FakeBattle + FakeExecutor + FakePolicy)
  4. simulated match, mirrored orientation
  5. simulated match, no reference -> fresh-fit calibration
  6. policy timeout resilience (orchestrator skip path + PolicyClient stale drop)
  7. real serve_policy replay of battle_diffcheck.ndjson (real torch, ~2-3min)
  8. Mirror runtime contract (live-2026-10-17 blocker) against engine validation
  9. deploy clamp: enemy-half troop targets retargeted to the own bank

Run: FirstLight venv python -I pipeline/tests/selftest_orchestrator.py [--skip-slow]
"""
from __future__ import annotations

import json
import math
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "pipeline" / "tools"))

from calibrate_grid import KIND_MAP, fit_homography  # noqa: E402

from pipeline.act.orchestrator import (  # noqa: E402
    MatchSession, calibration_points, fit_fresh_mapping, grid_to_world,
    mirror_matrix, solve_reference_mapping,
)
from pipeline.adapt.firstlight_adapter import FirstLightAdapter  # noqa: E402
from pipeline.host.policy_client import PolicyClient, PolicyError  # noqa: E402
from pipeline.ingest.frames import BattleFrame, HandCard, ObjectFrame, PlayerFrame  # noqa: E402

CAPTURE = WORKSPACE / "pipeline" / "artifacts" / "captures" / "battle_diffcheck.ndjson"


def fake_layout() -> dict:
    return {
        "screen_width": 1080,
        "screen_height": 1920,
        "card_slots_px": {"0": [335, 1690], "1": [540, 1690],
                          "2": [745, 1690], "3": [950, 1690]},
        "sample_region": {"x0f": 0.10, "x1f": 0.90, "y0f": 0.42, "y1f": 0.73},
    }


def build_h_ref() -> np.ndarray:
    """Plausible direct world->screen map for a 1080x1920 client.

    Full-board quad in the sim's owner-0 client view (own half = world y small
    = screen bottom): tower lines (y 2500/29500) must project inside the
    screen and into the bottom/top halves, or the mapping sanity gate —
    which projects exactly those towers — rejects the reference.
    """
    world = [(2000, 2500), (16000, 2500), (2000, 29500), (16000, 29500)]
    screen = [(120, 1700), (960, 1700), (120, 150), (960, 150)]
    return np.asarray(fit_homography(world, screen), dtype=float)


def apply_inv(h_inv: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    w = h_inv @ np.array([point[0], point[1], 1.0])
    return float(w[0] / w[2]), float(w[1] / w[2])


def pick_test_cards(adapter: FirstLightAdapter):
    """Two 8-card decks of real roster ids plus their elixir costs."""
    troops, others = [], []
    for card_id, spec in sorted(adapter.card_specs.items()):
        kind = KIND_MAP.get(spec.kind)
        if kind not in ("troop", "building", "spell"):
            continue
        cost = float(spec.elixir_cost or 0.0)
        (troops if kind in ("troop", "building") else others).append((cost, card_id))
    troops.sort()
    my_deck = [c for _, c in troops[:8]]
    enemy_deck = [c for _, c in (troops[8:14] + others[:2])]
    costs = {c: float(adapter.card_specs[c].elixir_cost or 0.0) for c in set(my_deck + enemy_deck)}
    assert len(my_deck) == 8 and len(set(enemy_deck)) == 8, (my_deck, enemy_deck)
    return my_deck, enemy_deck, costs


# -- fake machinery ----------------------------------------------------------

class FakeBattle:
    """Tiny arena sim: frames, elixir, hand cycling, kind14->kind15 spawns."""

    TOWER_HP = 140000

    def __init__(self, h_true: np.ndarray, decks: dict, costs: dict, *, seed: int = 7):
        self.h_inv = np.linalg.inv(np.asarray(h_true, dtype=float))
        self.decks = {o: list(decks[o]) for o in (0, 1)}
        self.costs = costs
        self.rng = np.random.default_rng(seed)
        self.elixir = {0: 5.0, 1: 5.0}
        self.hands = {o: deque(self.decks[o][:4]) for o in (0, 1)}
        self.cycles = {o: deque(self.decks[o][4:]) for o in (0, 1)}
        self.towers = [
            {"id": 5_000_000 + i, "owner": o,
             "x": (9000 if pos == 0 else (4500 if pos == 1 else 13500)),
             "y": 3000 if o == 0 else 29000}
            for i, (o, pos) in enumerate([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)])
        ]
        self.objects: list[dict] = []
        self.pending: list[tuple[int, int, float, float]] = []
        self.next_id = 6_000_000
        self.seq = 0
        self.tick = 0
        self.selected: int | None = None
        self.plays: list[tuple[int, float, float]] = []  # (card_id, x, y) actor spawns

    # -- execution sink interface (mirrors ActionExecutor) --------------------

    def select_slot(self, slot: int) -> None:
        self.selected = int(slot)

    def on_tap(self, sx: float, sy: float, source: str = "probe") -> None:
        if self.selected is None:
            return
        wx, wy = apply_inv(self.h_inv, (sx, sy))
        card = self.hands[0][self.selected]
        if self.costs.get(card, 99.0) > self.elixir[0] + 1e-6:
            self.selected = None  # tap rejected, like the real client
            return
        self.elixir[0] -= self.costs[card]
        self.pending.append((0, card, wx + float(self.rng.uniform(-40, 40)),
                             wy + float(self.rng.uniform(-40, 40))))
        self.cycles[0].append(card)
        self.hands[0][self.selected] = self.cycles[0].popleft()
        self.plays.append((card, wx, wy, source))
        self.selected = None

    # -- simulation ------------------------------------------------------------

    def step(self) -> BattleFrame:
        self.tick += 2
        self.seq += 1
        for o in (0, 1):
            self.elixir[o] = min(10.0, self.elixir[o] + 1.0 / 28.0)
        if self.tick % 100 == 0:  # the AI opponent deploys every ~5s
            for slot, card in enumerate(self.hands[1]):
                if self.costs.get(card, 99.0) <= self.elixir[1]:
                    self.elixir[1] -= self.costs[card]
                    self.pending.append((1, card, float(self.rng.uniform(2000, 16000)), 15000.0))
                    self.cycles[1].append(card)
                    self.hands[1][slot] = self.cycles[1].popleft()
                    break
        for owner, card, x, y in self.pending:
            self.next_id += 1
            self.objects.append({"id": self.next_id, "owner": owner, "card": card,
                                 "x": x, "y": y, "age": 0})
        self.pending.clear()
        for obj in self.objects:
            obj["y"] += 100 if obj["owner"] == 0 else -100  # march across the bridge
        players = tuple(
            PlayerFrame(owner=o, elixir=self.elixir[o],
                        hand=tuple(HandCard(slot, self.decks[o].index(cid), cid)
                                   for slot, cid in enumerate(self.hands[o])),
                        cycle=tuple(self.cycles[o]), deck=tuple(self.decks[o]))
            for o in (0, 1))
        frame_objects = [
            ObjectFrame(object_id=t["id"], kind=13, owner=t["owner"], x=t["x"], y=t["y"],
                        tx=0, ty=0, card_id=0, level=1, hp=self.TOWER_HP, max_hp=self.TOWER_HP,
                        shield=0, max_shield=0)
            for t in self.towers
        ]
        for obj in self.objects:
            frame_objects.append(ObjectFrame(
                object_id=obj["id"], kind=14 if obj["age"] == 0 else 15, owner=obj["owner"],
                x=int(round(obj["x"])), y=int(round(obj["y"])), tx=0, ty=0,
                card_id=obj["card"], level=1, hp=1000, max_hp=1000, shield=0, max_shield=0))
            obj["age"] += 1
        return BattleFrame(seq=self.seq, t_ms=self.tick * 50, tick=self.tick,
                           players=players, objects=tuple(frame_objects))


class FakeExecutor:
    """Tap sink wired into FakeBattle (ActionExecutor's interface)."""

    def __init__(self, battle: FakeBattle) -> None:
        self.battle = battle
        self.calls: list[tuple] = []

    def play_card(self, slot, world_xy, cal) -> None:
        self.calls.append(("play", slot, world_xy))
        sx, sy = cal.world_to_screen(*world_xy)
        self.battle.select_slot(slot)
        self.battle.on_tap(sx, sy, source="decision")

    def tap_card_slot(self, slot, cal) -> None:
        self.calls.append(("slot", slot))
        self.battle.select_slot(slot)

    def tap_screen(self, x, y) -> None:
        self.calls.append(("screen", x, y))
        self.battle.on_tap(x, y)

    def close(self) -> None:
        pass


class FakePolicy:
    """Scripted stand-in for serve_policy: legal plays every Nth act."""

    def __init__(self, play_every: int = 3, fail_first_act: bool = False,
                 first_play_grid: tuple[int, int] | None = None) -> None:
        self.play_every = play_every
        self.fails_left = 1 if fail_first_act else 0
        self.first_play_grid = first_play_grid
        self.decided: list[tuple] = []  # (tick, slot, card_id, (col, row))
        self.acts = 0
        self._act_number = 0

    def request(self, payload: dict, *, timeout_s=None, expect_tick=None) -> dict:
        op = payload["op"]
        if op == "start":
            return {"ok": True, "checkpoint_id": "fake"}
        if op == "end":
            return {"ok": True}
        obs = payload["observation"]
        tick = int(obs["tick"])
        self.acts += 1
        if self.fails_left > 0:
            self.fails_left -= 1
            raise PolicyError("serve_policy timed out after 1.0s (fake)")
        self._act_number += 1
        if self.play_every and self._act_number % self.play_every == 0:
            picked = self._pick_action(obs)
            if picked is not None:
                slot, card_id, (col, row) = picked
                if self.first_play_grid is not None and not self.decided:
                    col, row = self.first_play_grid  # force an off-mask cell
                self.decided.append((tick, slot, card_id, (col, row)))
                return {"ok": True, "tick": tick, "inference_ms": 45.0,
                        "actions": [{"kind": "play_card", "hand_slot": slot,
                                     "card_id": card_id, "target_grid": [col, row],
                                     "target_kind": "grid"}]}
        return {"ok": True, "tick": tick, "inference_ms": 35.0, "actions": []}

    @staticmethod
    def _pick_action(obs: dict):
        mask = obs["action_mask"]
        for slot, legal in enumerate(mask["hand_slots"]):
            if not legal:
                continue
            entry = mask["placement_masks"][str(slot)]
            for row, cells in enumerate(entry["row_major"]):
                for col, ok in enumerate(cells):
                    if ok:
                        return slot, int(entry["card_id"]), (col, row)
        return None


def run_sim(label: str, h_true: np.ndarray, h_ref, expect_mode: str,
            adapter: FirstLightAdapter, *, play_every: int = 3,
            fail_first_act: bool = False) -> dict:
    my_deck, enemy_deck, costs = pick_test_cards(adapter)
    battle = FakeBattle(h_true, {0: my_deck, 1: enemy_deck}, costs)
    executor = FakeExecutor(battle)
    policy = FakePolicy(play_every=play_every, fail_first_act=fail_first_act)
    frames: queue.Queue = queue.Queue()

    def feed() -> None:
        for _ in range(1500):
            frames.put(battle.step())
            time.sleep(0.001)  # near-real cadence: the drain-latest consumer skips backlog
        frames.put(None)

    threading.Thread(target=feed, daemon=True).start()
    session = MatchSession(executor=executor, frames=frames, policy=policy,
                           layout=fake_layout(), my_deck=my_deck, reference_h=h_ref,
                           episode_tag=f"sim-{label}", battle_end_s=2.0,
                           log=lambda msg: print(f"    {msg}") if msg.startswith(
                               ("client", "mapping", "episode", "battle", "self-cal")) else None)
    summary = session.run()
    assert summary["outcome"] == "completed", (label, summary)
    assert summary["calibration"] == expect_mode, (label, summary)
    assert summary["decisions"] > 20, (label, summary)
    assert summary["plays"] >= 5, (label, summary)
    # placement accuracy for decision plays only (probes are separate taps):
    # tap->screen->world round trip must land on the policy's intended cell
    decision_plays = [p for p in battle.plays if p[3] == "decision"]
    assert len(decision_plays) == len(policy.decided), \
        (label, len(decision_plays), len(policy.decided))
    for (card, wx, wy, _), decided in zip(decision_plays, policy.decided):
        col, row = decided[3]
        err = math.hypot(wx - (col * 1000 + 500), wy - (row * 1000 + 500))
        assert card == decided[2] and err < 400, (label, decided, wx, wy, err)
    report = Path(summary["report"])
    lines = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == summary["decisions"], (label, len(lines), summary["decisions"])
    print(f"  [{label}] decisions={summary['decisions']} plays={summary['plays']} "
          f"cal={summary['calibration']} OK")
    return summary


# -- sections ------------------------------------------------------------------

def section_1_mapping_solver() -> None:
    h_ref = build_h_ref()
    pts = calibration_points(fake_layout())
    inv_ref = np.linalg.inv(h_ref)
    inv_mir = np.linalg.inv(mirror_matrix(1080) @ h_ref)

    h, mode, resid = solve_reference_mapping(h_ref, 1080,
                                             [(p, apply_inv(inv_ref, p)) for p in pts[:2]])
    assert mode == "direct" and h is not None and resid < 5, (mode, resid)

    h, mode, resid = solve_reference_mapping(h_ref, 1080,
                                             [(p, apply_inv(inv_mir, p)) for p in pts[:2]])
    assert mode == "mirrored" and h is not None and resid < 5, (mode, resid)

    rng = np.random.default_rng(11)
    noisy = [(p, (w[0] + float(rng.uniform(-40, 40)), w[1] + float(rng.uniform(-40, 40))))
             for p, w in ((p, apply_inv(inv_ref, p)) for p in pts[:2])]
    h, mode, resid = solve_reference_mapping(h_ref, 1080, noisy)
    assert mode == "direct" and h is not None and resid < 150, (mode, resid)

    # readbacks displaced 6000 world units (~280px) fit neither variant
    bad = [(p, (w[0] + 6000.0, w[1])) for p, w in ((p, apply_inv(inv_ref, p)) for p in pts[:2])]
    h, mode, resid = solve_reference_mapping(h_ref, 1080, bad)
    assert h is None, (mode, resid)

    probes6 = [(p, (w[0] + 6000.0, w[1])) for p, w in ((p, apply_inv(inv_ref, p)) for p in pts)]
    h2, resid2 = fit_fresh_mapping(probes6)
    assert h2 is not None and resid2 < 5, resid2
    back = [apply_inv(np.linalg.inv(h2), p) for p, _ in probes6]
    for (_, want), got in zip(probes6, back):
        assert math.hypot(want[0] - got[0], want[1] - got[1]) < 50, (want, got)
    assert fit_fresh_mapping(probes6[:3])[0] is None  # too few probes
    print("  [mapping-solver] direct/mirrored/noisy/disagreement/fresh-fit OK")


def section_2_grid_to_world() -> None:
    assert grid_to_world((0, 0)) == (500.0, 500.0)
    assert grid_to_world((17, 31)) == (17500.0, 31500.0)
    assert grid_to_world([2, 5]) == (2500.0, 5500.0)
    for bad in ((-1, 0), (18, 0), (0, 32), (0, -1)):
        try:
            grid_to_world(bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    print("  [grid_to_world] bounds OK")


def section_3_to_5_sims(adapter: FirstLightAdapter) -> None:
    h_ref = build_h_ref()
    run_sim("direct", h_ref, h_ref, "reference-direct", adapter)
    run_sim("mirrored", mirror_matrix(1080) @ h_ref, h_ref, "reference-mirrored", adapter)
    run_sim("fresh-fit", h_ref, None, "fresh-fit", adapter)


def section_6_timeout(adapter: FirstLightAdapter) -> None:
    # orchestrator: a failed act must be skipped, not fatal
    h_ref = build_h_ref()
    run_sim("act-timeout", h_ref, h_ref, "reference-direct", adapter,
            fail_first_act=True)

    # PolicyClient: stale-tick drop + EOF sentinel against a stub child process
    class StubPolicy(PolicyClient):
        def __init__(self) -> None:
            self.proc = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(120)"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            self._responses: queue.Queue = queue.Queue()
            self.default_timeout_s = 2.0
            self._closed = False

    policy = StubPolicy()
    try:
        started = time.perf_counter()
        try:
            policy.request({"op": "act"}, timeout_s=0.3)
            raise AssertionError("expected timeout")
        except PolicyError:
            pass
        assert time.perf_counter() - started < 2.0
        policy._responses.put({"ok": True, "tick": 100})            # stale leftover
        policy._responses.put({"ok": True, "tick": 105, "actions": ["fresh"]})
        response = policy.request({"op": "act"}, timeout_s=1.0, expect_tick=105)
        assert response["tick"] == 105 and response["actions"] == ["fresh"], response
        # a tickless {"ok": false} error response must surface, not be dropped
        # as stale (live 2026-10-17: this hid serve_policy errors for a
        # whole match — every act timed out while the engine sat idle)
        policy._responses.put({"ok": False, "error": "boom"})
        try:
            policy.request({"op": "act"}, timeout_s=1.0, expect_tick=105)
            raise AssertionError("expected engine error")
        except PolicyError as error:
            assert "boom" in str(error), error
        policy._responses.put(None)
        try:
            policy.request({"op": "act"}, timeout_s=1.0)
            raise AssertionError("expected closed-stream error")
        except PolicyError as error:
            assert "closed" in str(error), error
    finally:
        policy._closed = True
        policy.proc.terminate()
    print("  [timeout] act-skip sim + stale-drop/EOF OK")


def section_7_replay() -> None:
    import pipeline.act.orchestrator as orch

    assert CAPTURE.is_file(), f"capture missing: {CAPTURE}"
    code = orch.run_replay(CAPTURE)
    assert code == 0, f"replay failed with code {code}"
    print("  [replay] real serve_policy end-to-end OK")


def section_8_mirror_contract(adapter: FirstLightAdapter) -> None:
    """Mirror runtime contract: the live-2026-10-17 blocker.

    serve_policy rejects any Mirror play whose runtime contract is incomplete;
    the adapter must track the actor's last played card and mirror it into
    both the private hand metadata and the placement mask entry.
    """
    from native_runner.contracts import ObservationV1
    from native_runner.training.v4.native_actions import MIRROR_CARD_ID, mirror_play_runtime_contract

    mirror, played = MIRROR_CARD_ID, 26000010
    deck = (26000010, 26000030, 26000031, 26000084, 28000002, MIRROR_CARD_ID, 28000013, 28000016)
    costs = {cid: float(adapter.card_specs[cid].elixir_cost or 0.0) for cid in deck}
    towers = tuple(
        ObjectFrame(object_id=5_000_000 + i, kind=13, owner=i // 3,
                    x=(9000 if i % 3 == 0 else (4500 if i % 3 == 1 else 13500)),
                    y=3000 if i // 3 == 0 else 29000, tx=0, ty=0, card_id=0,
                    level=1, hp=140000, max_hp=140000, shield=0, max_shield=0)
        for i in range(6))

    def frame(tick: int, hand_ids: tuple[int, ...], cycle: tuple[int, ...]) -> BattleFrame:
        players = tuple(
            PlayerFrame(owner=o, elixir=10.0,
                        hand=tuple(HandCard(slot, deck.index(cid), cid)
                                   for slot, cid in enumerate(hand_ids)),
                        cycle=cycle, deck=deck)
            for o in (0, 1))
        return BattleFrame(seq=tick, t_ms=tick * 50, tick=tick, players=players, objects=towers)

    hand1 = (played, 26000030, mirror, 26000084)
    cycle1 = (28000002, 28000013, 28000016, 26000031)
    hand2 = (26000030, 26000084, mirror, 26000031)   # played card left the hand
    cycle2 = (28000002, 28000013, 28000016, played)

    obs1 = adapter.observation(frame(100, hand1, cycle1), episode_id="t-mirror",
                               decks={0: deck, 1: deck})
    slot = hand1.index(mirror)
    assert obs1["action_mask"]["hand_slots"][slot] is False, obs1["action_mask"]["reasons"]
    assert obs1["action_mask"]["reasons"]["slots"][str(slot)] == "mirror_target_unknown"

    obs2 = adapter.observation(frame(110, hand2, cycle2), episode_id="t-mirror",
                               decks={0: deck, 1: deck})
    slot = hand2.index(mirror)
    assert obs2["action_mask"]["hand_slots"][slot] is True, obs2["action_mask"]["reasons"]
    entry = obs2["action_mask"]["placement_masks"][str(slot)]
    assert entry["effective_card_id"] == played and entry["card_id"] == played, entry
    assert math.isclose(entry["effective_cost"], costs[played] + 1.0), entry

    # the engine's own validation, on the exact objects decode would pass
    player = ObservationV1.from_mapping(obs2).players[obs2["owner"]]
    runtime = player.metadata["hand_runtime_by_slot"][str(slot)]
    got = mirror_play_runtime_contract(player=player, hand_slot=slot, deck=deck,
                                       card_costs=costs, placement_entry=entry)
    assert got == (played, costs[played] + 1.0, 0), got
    assert runtime["hand_slot"] == slot and runtime["visible_card_id"] == mirror
    print(f"  [mirror-contract] target={played} cost={got[1]} form={got[2]} OK")


def section_9_clamp(adapter: FirstLightAdapter) -> None:
    """Deploy clamp: enemy-half troop targets retarget to the own bank."""
    cl = MatchSession.clamp_deploy_target
    assert cl("troop", 5, 23, 0) == (5, 15, True)
    assert cl("troop", 5, 10, 0) == (5, 10, False)
    assert cl("spell", 5, 23, 0) == (5, 23, False)   # spells are legal anywhere
    assert cl("hero", 3, 22, 0) == (3, 15, True)
    assert cl("building", 17, 31, 0) == (17, 15, True)
    assert cl(None, 5, 23, 0) == (5, 23, False)      # unknown kinds pass through
    assert cl("troop", 5, 3, 1) == (5, 16, True)     # owner 1: own bank is row 16
    assert cl("troop", 5, 20, 1) == (5, 20, False)

    # wiring: an enemy-half troop decision must physically land on the bank
    my_deck, enemy_deck, costs = pick_test_cards(adapter)
    battle = FakeBattle(build_h_ref(), {0: my_deck, 1: enemy_deck}, costs)
    executor = FakeExecutor(battle)
    policy = FakePolicy(play_every=2, first_play_grid=(7, 23))
    frames: queue.Queue = queue.Queue()

    def feed() -> None:
        for _ in range(600):
            frames.put(battle.step())
            time.sleep(0.001)
        frames.put(None)

    threading.Thread(target=feed, daemon=True).start()
    session = MatchSession(executor=executor, frames=frames, policy=policy,
                           layout=fake_layout(), my_deck=my_deck,
                           reference_h=build_h_ref(), episode_tag="sim-clamp",
                           battle_end_s=2.0, log=lambda msg: None)
    summary = session.run()
    decision_plays = [p for p in battle.plays if p[3] == "decision"]
    assert decision_plays, summary
    _, wx, wy, _ = decision_plays[0]
    assert abs(wx - 7500.0) < 400 and abs(wy - 15500.0) < 400, (wx, wy)
    print("  [deploy-clamp] enemy-half troop retargeted to own bank OK")


def main() -> int:
    skip_slow = "--skip-slow" in sys.argv
    print("1/9 mapping solver")
    section_1_mapping_solver()
    print("2/9 grid_to_world")
    section_2_grid_to_world()
    print("3-5/9 simulated matches (adapter load first)")
    adapter = FirstLightAdapter(actor_owner=0)
    section_3_to_5_sims(adapter)
    print("6/9 timeout resilience")
    section_6_timeout(adapter)
    print("8/9 mirror runtime contract")
    section_8_mirror_contract(adapter)
    print("9/9 deploy clamp")
    section_9_clamp(adapter)
    if skip_slow:
        print("7/9 replay SKIPPED (--skip-slow)")
    else:
        print("7/9 real-engine replay (~2-3min)")
        section_7_replay()
    print("ALL OFFLINE SELF-TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
