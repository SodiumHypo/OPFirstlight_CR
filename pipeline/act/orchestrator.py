"""P2.2 orchestrator: one live match driven end-to-end by the FirstLight policy.

actual_game capture -> adapter v4 observation -> serve_policy decision ->
world->screen mapping -> executor taps. Session routing per workspace
CLAUDE.md: capture and taps target actual_game; inference is the host-side
serve_policy subprocess over the General checkpoint (the ai_rea instance
hosts the offline engine APK and stays out of this loop).

Opening self-calibration: the world frame is fixed (game arena coordinates);
per match the client renders it direct or x-mirrored (P2.1 finding), so two
x-extreme probes disambiguate the stored reference homography's variants and
disagreement falls back to a fresh 4-6 probe DLT fit (calibrate_grid's proven
machinery). The calibration file from tools/calibrate_grid.py provides the
reference mapping and the card-slot geometry.

Run (from the workspace root, FirstLight venv python):
  python -I pipeline/act/orchestrator.py                      # live match
  python -I pipeline/act/orchestrator.py --replay capture.ndjson   # offline replay
  python -I pipeline/tests/selftest_orchestrator.py           # offline self-test
"""
from __future__ import annotations

import datetime as dt
import json
import queue
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT.parent))      # workspace root: pipeline.*
sys.path.insert(0, str(PIPELINE_ROOT / "tools"))   # tool siblings

from calibrate_grid import (  # noqa: E402
    MAX_RUNTIME_S, SPAWNABLE_ENTITY_KINDS, BattleStream, apply_h, fit_homography,
    match_alive, pick_slot, reprojection_residuals, resolve_client_owner,
    ransac_homography,
)
from capture_battle_stream import ADB, ensure_binary, find_game_pid  # noqa: E402
from instances import resolve_role  # noqa: E402
from pipeline.adb_tools import adb_serial_default  # noqa: E402

from pipeline.act.executor import CALIBRATION_DIR, ActionExecutor, Calibration  # noqa: E402
from pipeline.adapt.firstlight_adapter import (  # noqa: E402
    ENTITY_KIND_BY_CARD_KIND, TOWER_OBJECT_ID_MAX, TOWER_OBJECT_ID_MIN,
    FirstLightAdapter,
)
from pipeline.host.policy_client import PolicyClient, PolicyError  # noqa: E402

FIRST_DECISION_TICK = 90   # engine warmup; serve_policy rejects earlier act calls
# 3 ticks = 150ms — serve_policy does not validate the interval (the wait
# action's next_decision_ticks: 5 is advisory metadata only), so the cadence
# is ours to tune. User directive 2026-10-17: kiting plays need faster
# reactions than the trained 250ms turn. Off-distribution risk (the model was
# trained on 250ms steps) is accepted as a live experiment — revert to 5 if
# decision quality degrades.
DECISION_STRIDE = 3        # one act every 3 ticks (150ms at 50ms ticks)
GRID_COLS, GRID_ROWS = 18, 32
GRID_CELL = 1000.0         # world units per cell (18000x32000 over 18x32)
KING_TOWER_X = 9000.0      # board x center — the x-mirror axis for play-context diagnostics
CAL_TOL_PX = 150.0         # probe acceptance; correct variants reproject within ~tens of px
CAL_WALL_S = 72.0          # self-calibration wall budget before solving with what we have
PROBE_ELIXIR_MIN = 8.0     # probes burn real tempo: only spend from elixir surplus
PROBE_AWAIT_S = 1.5
BATTLE_END_S = 20.0        # stream quiet this long after frames flowed = battle over
START_TIMEOUT_S = 60.0     # tensorizer build + first start on CPU
DECISION_TIMEOUT_S = 2.0   # live acts run slower under MuMu load; stale-drop makes waits safe
REPORTS_DIR = PIPELINE_ROOT / "artifacts" / "reports"


def mirror_matrix(screen_width: float) -> np.ndarray:
    """Screen-space x flip: the arena renders mirrored from the other side."""
    return np.array([[-1.0, 0.0, float(screen_width)],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])


def grid_to_world(grid) -> tuple[float, float]:
    """Native action grid (col, row) -> arena world units (cell centre)."""
    col, row = int(grid[0]), int(grid[1])
    if not (0 <= col < GRID_COLS and 0 <= row < GRID_ROWS):
        raise ValueError(f"target_grid {tuple(grid)!r} outside the {GRID_COLS}x{GRID_ROWS} board")
    return col * GRID_CELL + GRID_CELL / 2, row * GRID_CELL + GRID_CELL / 2


def solve_reference_mapping(h_ref: np.ndarray, screen_width: float,
                            probes: list) -> tuple:
    """Pick direct vs mirrored reference homography from probe readbacks.

    probes: [(screen (u, v), world (x, y))]. Returns (H, mode, max_residual_px)
    for the acceptable variant, or (None, mode, residual) when neither fits.
    """
    candidates = {"direct": h_ref, "mirrored": mirror_matrix(screen_width) @ h_ref}
    scored: dict[str, float] = {}
    for mode, h in candidates.items():
        residuals = [float(np.hypot(*np.subtract(apply_h(h, wx, wy), (u, v))))
                     for (u, v), (wx, wy) in probes]
        scored[mode] = max((r if np.isfinite(r) else 1e9) for r in residuals)
    mode = min(scored, key=scored.get)
    if scored[mode] <= CAL_TOL_PX:
        return candidates[mode], mode, scored[mode]
    return None, mode, scored[mode]


def fit_fresh_mapping(probes: list) -> tuple:
    """Fresh DLT fit from probes (fallback when no reference variant fits).

    4+ probes fit directly; 6+ get RANSAC outlier safety. Returns
    (H, max_residual_px) or (None, residual) when unacceptable.
    """
    world = [w for _, w in probes]
    screen = [s for s, _ in probes]
    if len(probes) >= 6:
        h, inliers = ransac_homography(world, screen, 800, 100.0, 20261017)
        if len(inliers) < 4:
            return None, 1e9
        residuals = [r for i, r in enumerate(reprojection_residuals(h, world, screen))
                     if i in inliers]
    elif len(probes) >= 4:
        h = fit_homography(world, screen)
        residuals = reprojection_residuals(h, world, screen)
    else:
        return None, 1e9
    worst = max(residuals)
    return (h, worst) if worst <= CAL_TOL_PX else (None, worst)


def screen_dict(layout: dict) -> dict:
    """Layout stores flat screen_width/height; Calibration expects a dict."""
    return {"width": int(layout["screen_width"]), "height": int(layout["screen_height"])}


def calibration_points(layout: dict) -> list[tuple[float, float]]:
    """Own-half screen probe points, x-extremes first (mirror disambiguation).

    The y offsets are deliberately uneven (0.30/0.50/0.65): four points on one
    line plus two on a second would be a degenerate two-line configuration for
    the DLT fallback — distinct rows keep the fresh fit well-posed.
    """
    w, h = float(layout["screen_width"]), float(layout["screen_height"])
    r = layout["sample_region"]
    x0, x1 = w * r["x0f"], w * r["x1f"]
    y0, y1 = h * r["y0f"], h * r["y1f"]
    dx, dy = x1 - x0, y1 - y0
    return [
        (x0 + 0.12 * dx, y0 + 0.50 * dy),
        (x1 - 0.12 * dx, y0 + 0.50 * dy),
        (x0 + 0.12 * dx, y1 - 0.30 * dy),
        (x1 - 0.12 * dx, y1 - 0.30 * dy),
        (x0 + 0.45 * dx, y0 + 0.30 * dy),
        (x0 + 0.78 * dx, y0 + 0.65 * dy),
    ]


class MatchSession:
    """One battle: resolve seat -> self-calibrate -> per-tick policy decisions."""

    def __init__(self, *, executor, frames, policy, layout: dict, my_deck,
                 reference_h: np.ndarray | None, episode_tag: str,
                 actor_owner: int | None = None, injected_cal: Calibration | None = None,
                 battle_end_s: float = BATTLE_END_S, seat_wait_s: float = 60.0,
                 log=print, cancel_check=None) -> None:
        self.executor = executor
        self.frames = frames
        self.policy = policy
        self.layout = layout
        self.slots_px = layout["card_slots_px"]
        self.my_deck = tuple(my_deck)
        self.reference_h = reference_h
        self.episode_tag = episode_tag
        self.battle_end_s = battle_end_s
        self.seat_wait_s = seat_wait_s
        self.log = log
        self.cancel_check = cancel_check
        self._slot_cal = Calibration({"card_slots_px": self.slots_px,
                                      "world_to_screen": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                      "screen": screen_dict(layout)})

        self.actor_owner = actor_owner
        self.client_owner: int | None = actor_owner
        self.adapter: FirstLightAdapter | None = None
        if actor_owner is not None:
            self.adapter = FirstLightAdapter(actor_owner=actor_owner)
        self.cal = injected_cal
        self.decks: dict[int, tuple[int, ...]] | None = None
        self.latest = {"frame": None, "time": 0.0}
        self._seen: set[int] = set()
        self.episode_id = episode_tag
        self.episode_alive = False
        self.restarts = 0
        self.next_decision_tick = FIRST_DECISION_TICK
        self.calibration_mode = "injected" if injected_cal is not None else "none"
        self.decisions = 0
        self.errors = 0
        self.plays = 0
        self.latencies: list[float] = []
        self.act_wall_ms: list[float] = []  # full act round trip incl. IPC/JSON
        self.records: list[dict] = []
        # deploy ground truth + placement-context diagnostics (live 2026-10-17:
        # the user saw a fireball land on empty ground with the sensible cell
        # at the x-mirror, and one session where 13 building deploys silently
        # failed — spells deploy anywhere, buildings only on valid own-half
        # cells, so a subtly wrong mapping rejects buildings while spells
        # still "work"). Track whether each played card actually leaves the
        # hand, and what the enemy field looked like at the target cell vs
        # its x-mirror twin at decision time.
        self._pending_deploy: dict | None = None
        self.deploys_failed = 0
        self._last_play_ctx: dict | None = None

    # -- frame plumbing ----------------------------------------------------

    def _next_frame(self, timeout: float):
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def _absorb(self, frame) -> None:
        self.latest["frame"] = frame
        self.latest["time"] = time.perf_counter()
        self._seen.update(o.object_id for o in frame.objects)
        if self.adapter is None:
            self.client_owner = resolve_client_owner(frame, set(self.my_deck), self.client_owner)
            if self.client_owner is not None:
                self.adapter = FirstLightAdapter(actor_owner=self.client_owner)
                self.log(f"client owner={self.client_owner} — adapter ready")

    def _drain(self, seconds: float) -> None:
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            frame = self._next_frame(0.1)
            if frame is not None:
                self._absorb(frame)

    # -- opening self-calibration -------------------------------------------

    def _client_elixir(self) -> float:
        if self.latest["frame"] is None or self.client_owner is None:
            return 0.0
        player = next((p for p in self.latest["frame"].players
                       if p.owner == self.client_owner), None)
        return float(player.elixir) if player is not None else 0.0

    def _calibrate(self) -> bool:
        """Solve world->screen for this match via opening deploy probes.

        Probes burn real elixir in a live match, so they are constrained
        (user 2026-10-17): one troop-card probe on the reference path —
        an x-extreme tap disambiguates the stored reference's direct vs
        mirrored variant AND the readback pins this match's y offset.
        Spells are pure waste and misplaced buildings hurt the defense,
        so probes are troop-only. Probes wait for elixir surplus before
        spending (relaxed late in the wall budget so calibration never
        silently starves the whole match). A missing/broken reference
        still falls back to a fresh DLT fit from 4-6 probes.
        """
        points = calibration_points(self.layout)
        probes: list = []
        deadline = time.perf_counter() + CAL_WALL_S
        relaxed_at = deadline - CAL_WALL_S / 2  # past this, probe at any elixir
        probe_index = 0
        tries = 0
        while probe_index < len(points) and time.perf_counter() < deadline:
            if self._cancelled():
                return False
            if self.reference_h is not None and len(probes) >= 1:
                h, mode, resid = solve_reference_mapping(
                    self.reference_h, self.layout["screen_width"], probes)
                if h is not None and self._accept_mapping(h, f"reference-{mode}", resid):
                    return True
            if time.perf_counter() < relaxed_at and self._client_elixir() < PROBE_ELIXIR_MIN:
                self._drain(0.4)  # wait for elixir surplus: probes spend real tempo
                continue
            point = points[probe_index]
            choice = None
            if self.adapter is not None and self.client_owner is not None:
                frame = self.latest["frame"]
                choice = pick_slot(frame, self.adapter, Counter(), self.client_owner,
                                   {"troop"})
            if choice is None:  # wait out elixir; move on after ~6 dry seconds
                self._drain(0.4)
                tries += 1
                if tries >= 15:
                    tries = 0
                    probe_index += 1
                continue
            slot, _cost, card_id = choice
            snapshot = set(self._seen)
            self.executor.tap_card_slot(slot, self._slot_cal)
            time.sleep(0.04)
            self.executor.tap_screen(*point)
            readback = self._await_probe_spawn(snapshot, card_id)
            if readback is None:
                tries += 1
                if tries >= 2:
                    tries = 0
                    probe_index += 1
                continue
            tries = 0
            probe_index += 1
            probes.append((tuple(point), (float(readback[0]), float(readback[1]))))
            self.log(f"cal probe {len(probes)}: screen=({point[0]:.0f},{point[1]:.0f}) "
                     f"world=({readback[0]},{readback[1]})")
        if self.reference_h is not None and probes:
            h, mode, resid = solve_reference_mapping(
                self.reference_h, self.layout["screen_width"], probes)
            if h is not None and self._accept_mapping(h, f"reference-{mode}", resid):
                return True
        h, resid = fit_fresh_mapping(probes)
        if h is not None and self._accept_mapping(h, "fresh-fit", resid):
            return True
        self.log(f"self-calibration failed: probes={len(probes)} best residual {resid:.0f}px")
        return False

    def _await_probe_spawn(self, snapshot: set, card_id: int):
        deadline = time.perf_counter() + PROBE_AWAIT_S
        while time.perf_counter() < deadline:
            frame = self._next_frame(0.25)
            if frame is None:
                if not match_alive(self.latest):
                    return None
                continue
            self._absorb(frame)
            fresh = [o for o in frame.objects if o.object_id not in snapshot
                     and o.card_id == card_id and o.owner == self.client_owner
                     and not (TOWER_OBJECT_ID_MIN <= o.object_id <= TOWER_OBJECT_ID_MAX)]
            if fresh:
                return (int(fresh[0].x), int(fresh[0].y))
        return None

    def _mapping_sane(self, h) -> bool:
        """Orientation gate on a solved mapping.

        Two probes on one horizontal line cannot constrain the y axis, so a
        degenerate reference accept can pass the residual check while being
        flipped/scaled off-board. The client always renders its own half at
        the screen bottom: project both trios of standard tower positions and
        require own-half towers low, enemy towers high, everything on-screen.
        """
        m = np.asarray(h)

        def proj(x: float, y: float) -> tuple[float, float]:
            w = m[2][0] * x + m[2][1] * y + m[2][2]
            return ((m[0][0] * x + m[0][1] * y + m[0][2]) / w,
                    (m[1][0] * x + m[1][1] * y + m[1][2]) / w)

        width = float(self.layout["screen_width"])
        height = float(self.layout["screen_height"])
        own_y, enemy_y = ((29500.0, 2500.0) if self.client_owner == 1 else (2500.0, 29500.0))
        own = [proj(x, own_y) for x in (3500.0, 9000.0, 14500.0)]
        enemy = [proj(x, enemy_y) for x in (3500.0, 9000.0, 14500.0)]
        if any(not (0 <= u <= width and 0 <= v <= height) for u, v in own + enemy):
            return False
        return (all(v > height * 0.55 for _, v in own)
                and all(v < height * 0.45 for _, v in enemy))

    def _accept_mapping(self, h, mode: str, resid: float) -> bool:
        if not self._mapping_sane(h):
            self.log(f"mapping {mode} rejected: tower projections fail orientation sanity")
            return False
        self.calibration_mode = mode
        self.log(f"mapping solved: {mode} (probe residual max {resid:.0f}px)")
        self.cal = Calibration({
            "world_to_screen": [[float(v) for v in row] for row in np.asarray(h)],
            "card_slots_px": self.slots_px,
            "screen": screen_dict(self.layout),
        })
        return True

    # -- decision loop -------------------------------------------------------

    def _cancelled(self) -> bool:
        return self.cancel_check is not None and bool(self.cancel_check())

    def run(self) -> dict:
        started = time.perf_counter()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"live_{self.episode_tag}.jsonl"

        while self.adapter is None and time.perf_counter() - started < self.seat_wait_s:
            if self._cancelled():
                return self._finish(report_path, "cancelled")
            frame = self._next_frame(0.5)
            if frame is not None:
                self._absorb(frame)
        if self.adapter is None:
            return self._finish(report_path, "no client seat resolved (no battle frames)")

        if self.cal is None:
            if self._cancelled():
                return self._finish(report_path, "cancelled")
            if not self._calibrate():
                return self._finish(report_path, "self-calibration failed")

        outcome = "completed"
        with report_path.open("w", encoding="utf-8") as report:
            self._report = report
            while time.perf_counter() - started < MAX_RUNTIME_S:
                if self._cancelled():
                    outcome = "cancelled"
                    self.log("cancelled — stopping session")
                    break
                frame = self._next_frame(0.5)
                if frame is not None:
                    # Decisions act on the newest state: when the consumer
                    # falls behind (a slow act at 10fps capture), skip the
                    # backlog instead of compounding latency frame by frame —
                    # and let the battle-quiet timer see an empty queue.
                    newest = frame
                    while True:
                        try:
                            nxt = self.frames.get_nowait()
                        except queue.Empty:
                            break
                        if nxt is None:  # replay EOF sentinel
                            newest = None
                            break
                        newest = nxt
                    if newest is not None:
                        self._absorb(newest)
                        self._on_frame(newest)
                else:
                    quiet = time.perf_counter() - self.latest["time"] if self.latest["time"] else 0.0
                    if quiet > self.battle_end_s:
                        self.log("battle over (stream quiet past recovery window)")
                        break
                latest = self.latest["frame"]
                if latest is not None and latest.tick > 7200:
                    self.log("battle over (max ticks)")
                    break
        self._end_episode()
        return self._finish(report_path, outcome)

    def _on_frame(self, frame) -> None:
        if (self.adapter is None or self.cal is None or self.client_owner is None
                or len(frame.players) < 2):
            return
        # Decisions ride a 5-tick schedule instead of tick % stride == 0: the
        # collector can skip or duplicate ticks (drain-latest consumer, dup
        # drops), and a missed stride multiple then doubled the decision gap
        # to 10 ticks. Scheduling the next slot at the next stride multiple
        # self-heals — one decision per slot whether or not the exact mark
        # was observed. The slot is consumed on entry so skipped/failing
        # acts retry at stride pace, not every frame.
        if frame.tick < self.next_decision_tick:
            return
        self.next_decision_tick = (frame.tick // DECISION_STRIDE + 1) * DECISION_STRIDE
        self._check_pending_deploy(frame)
        self._last_play_ctx = None
        if self.decks is None:
            self.decks = self._seat_decks(frame)
            if self.decks is None:
                return
        if not self.episode_alive:
            # bump the episode id BEFORE building the observation: the start
            # request binds the observation's episode_id, and a later id change
            # makes every following act fail the episode_id match (live
            # 2026-10-17: 47 consecutive restarts, all dead on the first act)
            self.restarts += 1
            if self.restarts > 1:
                self.episode_id = f"{self.episode_tag}-r{self.restarts}"
        observation = self.adapter.observation(frame, episode_id=self.episode_id,
                                               decks=self.decks)
        if not self.episode_alive:
            self._start_episode(frame, observation)
            return
        act_started = time.perf_counter()
        try:
            response = self.policy.request({"op": "act", "observation": observation},
                                           timeout_s=DECISION_TIMEOUT_S,
                                           expect_tick=frame.tick)
        except PolicyError as error:
            # serve_policy kills its episode on any error and replies without a
            # tick; a mere timeout also leaves the episode state unknown, so
            # re-start the episode before the next act (start is idempotent).
            self.episode_alive = False
            self.log(f"decision skipped (tick {frame.tick}): {error}")
            return
        # act_ms - inference_ms = IPC + observation JSON serialization overhead
        self.act_wall_ms.append((time.perf_counter() - act_started) * 1000.0)
        self.decisions += 1
        if self.decisions % 20 == 0:
            elixir = float(next(p for p in frame.players
                                if p.owner == self.client_owner).elixir)
            self.log(f"decisions={self.decisions} plays={self.plays} "
                     f"deploys_failed={self.deploys_failed} elixir={elixir:.1f}")
        latency = float(response.get("inference_ms", 0.0))
        if latency:
            self.latencies.append(latency)
        actions = response.get("actions", [])
        executed = [self._execute(action, frame) for action in actions]
        self.plays += sum(1 for ok in executed if ok)
        record = {
            "tick": frame.tick,
            "elixir": float(next(p for p in frame.players
                                 if p.owner == self.client_owner).elixir),
            "hand": [c.card_id for c in sorted(
                next(p for p in frame.players if p.owner == self.client_owner).hand,
                key=lambda c: c.slot)],
            "inference_ms": latency,
            "act_ms": round(self.act_wall_ms[-1], 1) if self.act_wall_ms else None,
            "actions": actions,
            "executed": executed,
            "play_context": self._last_play_ctx,
        }
        self.records.append(record)
        self._report.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _start_episode(self, frame, observation: dict) -> None:
        try:
            response = self.policy.request({
                "op": "start",
                "episode": self.adapter.episode_config(self.decks[0], self.decks[1]),
                "observation": observation,
                "initial_elixir": self.adapter.initial_elixir(frame),
            }, timeout_s=START_TIMEOUT_S)
        except PolicyError as error:
            self.errors += 1
            self.log(f"episode start failed: {error}")
            return
        self.episode_alive = bool(response.get("ok"))
        if not self.episode_alive:
            self.errors += 1
        self.log(f"episode start#{self.restarts} tick={frame.tick}: ok={self.episode_alive}"
                 + ("" if self.episode_alive else f" {response}"))

    def _end_episode(self) -> None:
        if not self.episode_alive:
            return
        try:
            self.policy.request({"op": "end"}, timeout_s=5.0)
        except PolicyError:
            pass
        self.episode_alive = False

    def _seat_decks(self, frame) -> dict[int, tuple[int, ...]] | None:
        decks: dict[int, tuple[int, ...]] = {}
        for player in frame.players[:2]:
            deck = tuple(dict.fromkeys(int(c) for c in player.deck))
            if player.owner == self.client_owner and self.my_deck:
                deck = self.my_deck  # verified input; the frame copy is only a cross-check
            if len(deck) != 8:
                if player.owner == self.client_owner:
                    self.log(f"client seat deck unusable ({len(deck)} cards) — cannot start")
                    return None
                self.log(f"enemy seat deck missing ({len(deck)} cards) — mirroring my deck "
                         "(embedding-only input)")
                deck = self.my_deck
            decks[player.owner] = deck
        return decks

    def _check_pending_deploy(self, frame) -> None:
        """Verify the last played card actually left the hand.

        A tap chain that the game rejects (invalid building cell, wrong spot)
        leaves the card in hand — the play silently never happened. Confirmed
        = the card is absent from the client hand on a later frame.
        """
        if self._pending_deploy is None or self.client_owner is None:
            return
        player = next((p for p in frame.players if p.owner == self.client_owner), None)
        if player is None or not player.hand:
            return
        played = self._pending_deploy["card"]
        if all(c.card_id != played for c in player.hand):
            self._pending_deploy = None
        elif frame.tick - self._pending_deploy["tick"] > 60:
            self.deploys_failed += 1
            self.log(f"deploy FAILED: card {played} still in hand 60 ticks after the play")
            self._pending_deploy = None

    def _threat_context(self, frame, world: tuple[float, float]) -> dict:
        """Enemy field at the target cell vs its x-mirror twin, at decision time.

        Settles 'the AI plays mirrored' with data: if threats systematically
        sit at the twin cell while plays go to the empty one, the perspective
        chain is broken; if neither cell has threats, the model simply chose
        that cell.
        """
        radius = 3500.0

        def enemies_at(wx: float, wy: float) -> int:
            return sum(1 for o in frame.objects
                       if o.kind == 15 and o.hp > 0 and o.owner != self.client_owner
                       and abs(o.x - wx) <= radius and abs(o.y - wy) <= radius)

        twin = (2 * KING_TOWER_X - world[0], world[1])
        return {
            "target": [int(world[0]), int(world[1])],
            "mirror": [int(twin[0]), int(twin[1])],
            "enemies_at_target": enemies_at(*world),
            "enemies_at_mirror": enemies_at(*twin),
            "enemies_total": sum(1 for o in frame.objects
                                 if o.kind == 15 and o.hp > 0
                                 and o.owner != self.client_owner),
        }

    @staticmethod
    def clamp_deploy_target(card_kind: str | None, col: int, row: int,
                            client_owner: int) -> tuple[int, int, bool]:
        """Retarget enemy-half deployables onto the own-side bank (same column).

        The training engine allows enemy-half troop placements, so the policy
        aims there; the live client rejects deep taps outright and clamps
        others onto the bridge line (live 144857-m2: 2 of 6 plays stuck in
        hand, P2.1: y pinned ~17499). Row 15 (owner 0) / row 16 (owner 1) is
        the own bank — legal at every column — so the physical result is
        deterministic and keeps the lane intent. Spells are legal anywhere
        and pass through untouched.
        """
        if card_kind not in ("troop", "building", "hero"):
            return col, row, False
        if client_owner == 1:
            return (col, 16, True) if row <= 15 else (col, row, False)
        return (col, 15, True) if row >= 16 else (col, row, False)

    def _execute(self, action: dict, frame) -> bool:
        if action.get("kind") != "play_card":
            return False
        slot = action.get("hand_slot")
        grid = action.get("target_grid")
        if slot is None or grid is None:
            self.log(f"action missing slot/grid: {action}")
            return False
        player = next((p for p in frame.players if p.owner == self.client_owner), None)
        hand_card = next((c.card_id for c in player.hand if c.slot == int(slot)), None) \
            if player is not None else None
        if hand_card is not None and action.get("card_id") not in (None, hand_card):
            self.log(f"card mismatch at slot {slot}: action={action.get('card_id')} "
                     f"hand={hand_card} — executing by slot")
        card_id = int(hand_card if hand_card is not None else action.get("card_id") or 0)
        spec = self.adapter.card_specs.get(card_id) if card_id else None
        kind = ENTITY_KIND_BY_CARD_KIND.get(spec.kind) if spec is not None else None
        col, row, clamped = self.clamp_deploy_target(kind, int(grid[0]), int(grid[1]),
                                                     self.client_owner)
        try:
            world = grid_to_world([col, row])
        except ValueError as error:
            self.log(f"illegal action skipped: {error}")
            return False
        sx, sy = self.cal.world_to_screen(*world)
        self.executor.play_card(int(slot), world, self.cal)
        self._last_play_ctx = self._threat_context(frame, world)
        if clamped:
            self._last_play_ctx["intended_target"] = [int(grid[0]), int(grid[1])]
        self._pending_deploy = {"card": card_id, "tick": frame.tick}
        self.log(f"play: card={action.get('card_id')} slot={slot} grid=({col},{row})"
                 + (f" (clamped from ({grid[0]},{grid[1]}))" if clamped else "")
                 + f" world=({world[0]:.0f},{world[1]:.0f}) -> screen=({sx:.0f},{sy:.0f})"
                 f" | enemies@target={self._last_play_ctx['enemies_at_target']}"
                 f" @x-mirror={self._last_play_ctx['enemies_at_mirror']}"
                 f" total={self._last_play_ctx['enemies_total']}")
        return True

    # -- wrap-up ---------------------------------------------------------------

    def _finish(self, report_path: Path, outcome: str) -> dict:
        latencies = sorted(self.latencies)
        acts = sorted(self.act_wall_ms)

        def pct(seq: list[float], q: float) -> float | None:
            return seq[min(len(seq) - 1, int(len(seq) * q))] if seq else None

        summary = {
            "outcome": outcome,
            "calibration": self.calibration_mode,
            "decisions": self.decisions,
            "plays": self.plays,
            "deploys_failed": self.deploys_failed,
            "errors": self.errors,
            "restarts": self.restarts,
            "inference_ms": {
                "p50": latencies[len(latencies) // 2] if latencies else None,
                "p95": pct(latencies, 0.95),
                "max": latencies[-1] if latencies else None,
            },
            "act_ms": {
                "p50": pct(acts, 0.50),
                "p95": pct(acts, 0.95),
                "max": acts[-1] if acts else None,
            },
            "report": str(report_path),
        }
        self.log("summary: " + json.dumps(summary, ensure_ascii=False))
        return summary


def _load_layout(serial_tag: str) -> dict:
    layout_path = CALIBRATION_DIR / f"layout_{serial_tag}.json"
    if not layout_path.is_file():
        sys.exit(f"layout missing: {layout_path} — run tools/calibrate_grid.py once first")
    return json.loads(layout_path.read_text(encoding="utf-8"))


def _load_my_deck() -> tuple[int, ...]:
    deck_path = CALIBRATION_DIR / "my_deck.json"
    if not deck_path.is_file():
        sys.exit(f"my_deck missing: {deck_path}")
    return tuple(json.loads(deck_path.read_text(encoding="utf-8"))["card_ids"])


def run_live(*, deck: tuple[int, ...] | None = None, checkpoint: Path | None = None,
             continuous: bool = False, should_continue=None, log=print) -> int:
    """Live pipeline host on the actual_game instance.

    deck/checkpoint override the calibration-file deck and the default
    checkpoint. continuous=True keeps re-arming matches (auto-reload): after
    a battle goes quiet the session is replaced and the next battle's frames
    re-resolve the seat; stream/executor/policy live across matches (the
    engine binds one episode per start request and start is idempotent).
    should_continue is polled between matches in continuous mode; True means
    "keep running". MatchSession's cancel_check wants the inverse (True =
    cancel now), so the conversion happens once below.
    """
    inst = resolve_role("actual_game")
    if not inst.android_started:
        sys.exit(f"instance {inst.name} is not running")
    serial = inst.adb_serial
    subprocess.run([ADB, "connect", serial], check=True, capture_output=True, timeout=15)
    serial_tag = serial.replace(":", "_")
    layout = _load_layout(serial_tag)
    my_deck = tuple(deck) if deck else _load_my_deck()
    stored = Calibration.load(serial)
    reference_h = np.array(stored.data["world_to_screen"]) if stored is not None else None

    pid = find_game_pid(serial)
    if pid is None:
        sys.exit(f"game is not running on {inst.name}")
    ensure_binary(serial)
    # Previous runs leave their collector alive on-device (su detaches from the
    # local adb shell); a second ptrace attached to the game breaks the stream.
    subprocess.run([ADB, "-s", serial, "shell",
                    'su 0 sh -c "kill $(pidof cr_battle_stream) 2>/dev/null; true"'],
                   capture_output=True, timeout=15)
    log(f"instance={inst.name} serial={serial} pid={pid}")

    frames: queue.Queue = queue.Queue()
    stream = BattleStream(serial, pid, frames, log=log)
    stream.start()
    executor = ActionExecutor(serial)
    policy = PolicyClient(checkpoint=checkpoint) if checkpoint is not None else PolicyClient()
    summary: dict | None = None
    match_no = 0
    # GUI passes "still running" (True = keep going); MatchSession cancels on
    # True — invert once here (live 2026-10-17: passing should_continue
    # straight through cancelled every session at the stage-1 entry check,
    # so the GUI sat "idle" no matter when a battle started).
    cancel_check = (lambda: not should_continue()) if should_continue is not None else None
    try:
        while True:
            match_no += 1
            tag = f"{serial_tag}_{dt.datetime.now():%Y%m%d_%H%M%S}-m{match_no}"
            session = MatchSession(executor=executor, frames=frames, policy=policy,
                                   layout=layout, my_deck=my_deck, reference_h=reference_h,
                                   episode_tag=tag, seat_wait_s=300.0, log=log,
                                   cancel_check=cancel_check)
            summary = session.run()
            if not continuous or summary["outcome"] == "cancelled":
                break
            if summary["outcome"] != "completed":
                log("continuous mode — re-arming for the next battle")
            # settle: drop the dead battle's residual frames and give the
            # client a moment on the result screen before re-arming
            while True:
                try:
                    frames.get_nowait()
                except queue.Empty:
                    break
            if should_continue is not None and not should_continue():
                break
            time.sleep(8.0)
    finally:
        stream.stop()
        executor.close()
        policy.close()
    return 0 if summary is not None and summary["outcome"] == "completed" else 1


def run_replay(capture: Path) -> int:
    """Offline integration: recorded frames + real serve_policy + tap recorder.

    Captures may hold engine-side decks (diffcheck runs) rather than the live
    client's my_deck, so the client seat and deck are derived from the capture
    itself instead of deck matching.
    """
    from smoke_serve_policy import load_battle_frames

    frames_list = load_battle_frames(capture)
    client_owner = None
    for frame in frames_list:
        client_owner = next((p.owner for p in frame.players[:2] if len(set(p.deck)) == 8), None)
        if client_owner is not None:
            break
    if client_owner is None:
        sys.exit(f"no seat with a full 8-card deck in {capture}")
    my_deck = next(p.deck for frame in frames_list
                   for p in frame.players[:2]
                   if p.owner == client_owner and len(set(p.deck)) == 8)
    serial = adb_serial_default()
    layout = _load_layout(serial.replace(":", "_"))
    stored = Calibration.load(serial)
    injected = Calibration({
        "world_to_screen": stored.data["world_to_screen"] if stored else [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "card_slots_px": layout["card_slots_px"],
        "screen": screen_dict(layout),
    })

    class TapRecorder:
        def __init__(self) -> None:
            self.plays: list = []

        def play_card(self, slot, world_xy, cal) -> None:
            self.plays.append((slot, world_xy,
                               cal.world_to_screen(*world_xy) if world_xy else None))

        def tap_card_slot(self, slot, cal) -> None:
            self.plays.append((slot, None, None))

        def tap_screen(self, x, y) -> None:
            self.plays.append((None, (x, y), None))

        def close(self) -> None:
            pass

    frames: queue.Queue = queue.Queue()

    def feed() -> None:
        for frame in frames_list:
            frames.put(frame)
            time.sleep(0.05)  # ~2.5x realtime: the recurrent policy expects the
            while frames.qsize() > 8:  # ~10fps stream cadence it was trained on
                time.sleep(0.005)
        frames.put(None)

    threading.Thread(target=feed, daemon=True).start()
    policy = PolicyClient()
    session = MatchSession(executor=TapRecorder(), frames=frames, policy=policy,
                           layout=layout, my_deck=my_deck, reference_h=None,
                           episode_tag=f"replay_{capture.stem}",
                           actor_owner=client_owner,
                           injected_cal=injected, battle_end_s=2.0)
    try:
        summary = session.run()
    finally:
        policy.close()
    plays = sum(1 for play in session.executor.plays if play[1] is not None)
    print(f"replay taps: {plays} placements, {len(session.executor.plays)} raw taps")
    return 0 if summary["plays"] >= 1 else 1


def main() -> int:
    if "--replay" in sys.argv:
        return run_replay(Path(sys.argv[sys.argv.index("--replay") + 1]))
    return run_live()


if __name__ == "__main__":
    raise SystemExit(main())
