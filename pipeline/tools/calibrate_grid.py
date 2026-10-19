"""Self-calibrate the world<->screen mapping for the live client.

During a friendly match the tool deploys affordable cards at a spread of
screen positions, reads each spawn's true world position back from the battle
stream (first-frame position == tap point), fits a world->screen homography
(RANSAC over DLT pair sets — moving units and clamped casts pollute samples),
and verifies it with world-space probes measured end-to-end while the match
is still alive.

Prerequisite: layout_<serial>.json in artifacts/calibration/ with card-slot
screen positions and the arena sample region (analyze a screencap first; the
tool captures one if the layout is missing).

Run: FirstLight venv python -I tools/calibrate_grid.py [--role actual_game] [--with-enemy]
"""
from __future__ import annotations

import datetime as dt
import json
import math
import queue
import struct
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

from capture_battle_stream import (  # noqa: E402
    ADB, REMOTE_BIN, ensure_binary, find_game_pid,
)
from frames import FrameNormalizer  # noqa: E402
from instances import resolve_role  # noqa: E402

from pipeline.act.executor import CALIBRATION_DIR, ActionExecutor  # noqa: E402
from pipeline.adapt.firstlight_adapter import (  # noqa: E402
    ENTITY_KIND_BY_CARD_KIND, TOWER_OBJECT_ID_MAX, TOWER_OBJECT_ID_MIN, CardKind, FirstLightAdapter,
)

# The adapter's map covers entity forms only (spells never spawn entities);
# calibration also casts spells, whose kind12 zone centre is the cast point.
KIND_MAP = dict(ENTITY_KIND_BY_CARD_KIND)
KIND_MAP[CardKind.SPELL] = "spell"

AWAIT_SPAWN_S = 1.2
MAX_RUNTIME_S = 12 * 60
MIN_SAMPLES = 6
POINT_RETRIES = 2  # a skipped sample point is retried this many times (elixir misses)
SAMPLE_TARGET = 8  # own-half samples to aim for; saves match time for verification
SAMPLE_BUDGET_S = 85.0  # stop sampling at this age once MIN_SAMPLES exist — verification needs a live match too
CONTENT_FREEZE_S = 20.0  # static elixir+hand this long while sampling ⇒ stop trusting the stream
INLIER_PX = 100.0  # clean fits reproject ≤46px, pollution ≥800px — 100 separates
RANSAC_ITERS = 800
RANSAC_SEED = 20261017  # fixed so fits are reproducible from the same samples
STREAM_STALL_S = 12.0  # frames quiet this long ⇒ stream lost (supervisor reconnect takes ~6-8s)
STREAM_QUIET_S = 3.0  # frames quiet this long ⇒ supervisor re-attaches the collector
MAX_STREAM_RESTARTS = 5
# Game ticks run ~50ms; sample at half a tick so every tick is observed (2
# samples/tick, duplicates dropped by FrameNormalizer). At 100ms the collector
# advanced 2 ticks per frame and the tick%5 decision gate only hit every other
# match — decisions ran at ~500ms instead of the trained 250ms cadence.
CAPTURE_INTERVAL_MS = 25
VERIFY_ATTEMPTS = 4  # per verify point; slots rotate so any affordable real card deploys
# Spells are calibration-usable too: casting spawns a kind12 zone object whose
# centre is the cast point — readable back like a troop spawn. Without spells a
# hand that cycles into all-spells deadlocks the sampler (nobody plays to rotate).
SPAWNABLE_ENTITY_KINDS = {"troop", "building", "spell"}
ENEMY_KINDS = {"spell"}  # troops tapped on the enemy half get clamped to the bridge — spells only there


def png_size(png: bytes) -> tuple[int, int]:
    width, height = struct.unpack(">II", png[16:24])
    return int(width), int(height)


def screencap(serial: str, out: Path) -> tuple[int, int]:
    png = subprocess.run([ADB, "-s", serial, "exec-out", "screencap", "-p"],
                         check=True, capture_output=True, timeout=30).stdout
    out.write_bytes(png)
    return png_size(png)


def _normalize(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: centroid to origin, mean point distance sqrt(2).

    World coords are ~1e4 while screen coords are ~1e2; raw DLT rows are
    needlessly ill-conditioned across that span.
    """
    mean = pts.mean(axis=0)
    dist = float(np.hypot(*(pts - mean).T).mean())
    scale = np.sqrt(2.0) / max(dist, 1e-9)
    t = np.array([[scale, 0.0, -scale * mean[0]],
                  [0.0, scale, -scale * mean[1]],
                  [0.0, 0.0, 1.0]])
    return t, (pts - mean) * scale


def fit_homography(world: list[tuple[float, float]], screen: list[tuple[float, float]]) -> np.ndarray:
    """DLT fit of the homogeneous map world -> screen (3x3)."""
    tw, world_n = _normalize(np.asarray(world, dtype=float))
    ts, screen_n = _normalize(np.asarray(screen, dtype=float))
    rows = []
    for (x, y), (u, v) in zip(world_n, screen_n):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, vt = np.linalg.svd(np.array(rows))
    return np.linalg.inv(ts) @ vt[-1].reshape(3, 3) @ tw


def apply_h(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    with np.errstate(divide="ignore", invalid="ignore"):  # degenerate RANSAC candidates hit w=0
        w = h[2, 0] * x + h[2, 1] * y + h[2, 2]
        return (float((h[0, 0] * x + h[0, 1] * y + h[0, 2]) / w),
                float((h[1, 0] * x + h[1, 1] * y + h[1, 2]) / w))


def reprojection_residuals(h: np.ndarray, world, screen) -> list[float]:
    return [float(np.hypot(*np.subtract(apply_h(h, x, y), (u, v)))) for (x, y), (u, v) in zip(world, screen)]


def median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


def _non_collinear(pts: list[tuple[float, float]]) -> bool:
    """4 nearly-collinear points are degenerate for DLT — reject the pick."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-9)
    norm_area = 0.0
    for a, b, c in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        (x0, y0), (x1, y1), (x2, y2) = pts[a], pts[b], pts[c]
        norm_area = max(norm_area, abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) / (scale * scale))
    return norm_area > 1e-3


def ransac_homography(world: list, screen: list, iters: int, thresh_px: float,
                      seed: int) -> tuple[np.ndarray, list[int]]:
    """MSAC over 4-point DLT fits; returns (H, inlier indices into world).

    Iterative worst-trim fails once pollution reaches ~3 samples: the fit gets
    dragged until a clean grid point is the worst residual, so trimming keeps
    the bad ones (match 8). Minimal random sets survive that. Scored by MSAC
    (sum of truncated residuals) rather than inlier count — a model that
    absorbs an outlier by trading error onto every clean point must lose to
    the tight consensus, and a count cannot see that trade. Collinear picks
    are skipped (degenerate DLT can otherwise absorb outliers into a bogus
    consensus); the consensus is refit until it stops moving.
    """
    rng = np.random.default_rng(seed)
    n = len(world)
    best: list[int] = []
    best_score = float("inf")
    for _ in range(iters):
        pick = rng.choice(n, 4, replace=False)
        quad = [world[i] for i in pick]
        if not _non_collinear(quad):
            continue
        try:
            h = fit_homography(quad, [screen[i] for i in pick])
        except np.linalg.LinAlgError:
            continue
        residuals = reprojection_residuals(h, world, screen)
        inliers = [i for i, r in enumerate(residuals) if r <= thresh_px]
        score = sum(min(r, thresh_px) for r in residuals)
        if score < best_score:
            best_score, best = score, inliers
            if len(best) == n:
                break
    if len(best) < 4:
        return np.eye(3), best
    for _ in range(3):  # refit on the consensus until it stops moving
        h = fit_homography([world[i] for i in best], [screen[i] for i in best])
        residuals = reprojection_residuals(h, world, screen)
        refined = [i for i, r in enumerate(residuals) if r <= thresh_px]
        if len(refined) < 4 or refined == best:
            break
        best = refined
    return h, best


def main() -> int:
    role = "actual_game"
    if "--role" in sys.argv:
        role = sys.argv[sys.argv.index("--role") + 1]
    with_enemy = "--with-enemy" in sys.argv
    inst = resolve_role(role)
    if not inst.android_started:
        sys.exit(f"instance {inst.name} is not running")
    serial = inst.adb_serial
    subprocess.run([ADB, "connect", serial], check=True, capture_output=True, timeout=15)

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    serial_tag = serial.replace(":", "_")
    layout_path = CALIBRATION_DIR / f"layout_{serial_tag}.json"
    if not layout_path.is_file():
        shot = CALIBRATION_DIR / f"screencap_{serial_tag}.png"
        width, height = screencap(serial, shot)
        sys.exit(f"layout missing; screencap {width}x{height} saved to {shot} — analyze it and write {layout_path}")
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    slots_px = layout["card_slots_px"]

    pid = find_game_pid(serial)
    if pid is None:
        sys.exit(f"game is not running on {inst.name}")
    ensure_binary(serial)
    # Previous runs leave their collector alive on-device (su detaches from the
    # local adb shell); a second ptrace attached to the game breaks the stream.
    subprocess.run([ADB, "-s", serial, "shell",
                    'su 0 sh -c "kill $(pidof cr_battle_stream) 2>/dev/null; true"'],
                   capture_output=True, timeout=15)
    print(f"instance={inst.name} serial={serial} pid={pid}")

    adapter = FirstLightAdapter(actor_owner=0)
    executor = ActionExecutor(serial)

    frames: queue.Queue = queue.Queue()
    stream = BattleStream(serial, pid, frames)
    stream.start()

    samples, n_own = build_samples(layout, with_enemy)
    print(f"{len(samples)} screen sample points; deploy-readback calibration starts (do not play during this)")

    records: list[dict] = []
    seen: set[int] = set()
    slot_stats: Counter = Counter()
    latest = {"frame": None, "time": 0.0}
    mode = "seek"
    index = 0
    deadline = 0.0
    snapshot: set[int] = set()
    pending_slot = -1
    pending_screen: tuple[float, float] | None = None
    started = time.perf_counter()
    await_frames = 0
    last_wait_log = 0.0
    point_tries: dict[int, int] = {}
    pending_card_id = -1
    pending_is_enemy = False
    enemy_wait_start: float | None = None
    sampling_started: float | None = None
    content_sig = None
    content_changed_at = 0.0

    deck_path = CALIBRATION_DIR / "my_deck.json"
    my_deck = set(json.loads(deck_path.read_text(encoding="utf-8"))["card_ids"])
    client_owner: int | None = None
    print(f"my deck loaded ({len(my_deck)} cards); client = the hand drawn from it")

    while time.perf_counter() - started < MAX_RUNTIME_S:
        try:
            frame = frames.get(timeout=0.5)
        except queue.Empty:
            frame = None
        if mode == "seek":
            if len(records) >= SAMPLE_TARGET and index < n_own:
                print(f"sample target {SAMPLE_TARGET} reached — skipping remaining own-half points")
                index = n_own
            elif (sampling_started is not None
                    and time.perf_counter() - sampling_started > SAMPLE_BUDGET_S
                    and len(records) >= MIN_SAMPLES):
                print(f"sample budget {SAMPLE_BUDGET_S:.0f}s reached — fitting with {len(records)} samples")
                break  # verification needs a live match; don't spend it all on sampling
            if index >= len(samples):
                break  # all sample points consumed — go fit, don't wait for stream end
        if frame is not None:
            latest["frame"] = frame
            latest["time"] = time.perf_counter()
            seen.update(o.object_id for o in frame.objects)
            client_owner = resolve_client_owner(frame, my_deck, client_owner)

        if mode == "await":
            if frame is not None:
                await_frames += 1
            # Readback: a new object whose card id matches the played card;
            # fall back to any new non-tower object if the engine remaps ids.
            fresh = [o for o in (frame.objects if frame is not None else [])
                     if o.object_id not in snapshot and o.card_id == pending_card_id]
            if not fresh:
                fresh = [o for o in (frame.objects if frame is not None else [])
                         if o.object_id not in snapshot and o.card_id > 0
                         and not (TOWER_OBJECT_ID_MIN <= o.object_id <= TOWER_OBJECT_ID_MAX)]
            if fresh:
                unit = fresh[0]
                if pending_is_enemy and unit.kind != 12:
                    # Units tapped on the enemy half clamp to the bridge line;
                    # only a kind12 spell zone certifies an unclamped cast there.
                    print(f"  enemy point got kind={unit.kind} (unit, clamped) — discarding")
                    mode = "seek"
                    continue
                records.append({
                    "screen": [round(pending_screen[0]), round(pending_screen[1])],
                    "world": [int(unit.x), int(unit.y)],
                    "slot": pending_slot,
                    "card_id": pending_card_id,
                    "multi": len(fresh) > 1,
                    "kind": unit.kind,
                    "owner": unit.owner,
                    "object_id": int(unit.object_id),
                })
                slot_stats[pending_slot] += 1
                print(f"  sample {len(records)}: screen=({pending_screen[0]:.0f},{pending_screen[1]:.0f}) "
                      f"world=({unit.x},{unit.y}) slot={pending_slot} kind={unit.kind} owner={unit.owner}")
                mode = "seek"
            elif time.perf_counter() > deadline:
                tries = point_tries.get(index - 1, 0) + 1
                point_tries[index - 1] = tries
                if tries < POINT_RETRIES:
                    index -= 1  # likely an elixir miss — replay the same point
                    print(f"  point retry {tries}/{POINT_RETRIES}: screen={pending_screen} slot={pending_slot}")
                else:
                    kinds = Counter(f"kind{o.kind}/own{o.owner}" for o in
                                    (frame.objects if frame is not None else []) if o.object_id not in snapshot)
                    print(f"  sample skipped (no spawn): screen={pending_screen} slot={pending_slot} "
                          f"frames={await_frames} new_now={dict(kinds)}")
                mode = "seek"
            continue

        if index >= len(samples):
            continue
        point, is_enemy = samples[index]
        choice = pick_slot(latest["frame"], adapter, slot_stats, client_owner,
                           ENEMY_KINDS if is_enemy else SPAWNABLE_ENTITY_KINDS)
        if choice is None:
            if not match_alive(latest):
                print("  stream lost — stopping sampling")
                break
            # The stream can keep emitting while its snapshot content freezes
            # (matches 9-11): stop trusting it for sampling picks too.
            player_now = (next((p for p in latest["frame"].players if p.owner == client_owner), None)
                          if latest["frame"] else None)
            sig = ((round(float(player_now.elixir), 1), tuple(c.card_id for c in player_now.hand))
                   if player_now is not None else None)
            if sig != content_sig:
                content_sig = sig
                content_changed_at = time.perf_counter()
            elif records and time.perf_counter() - content_changed_at > CONTENT_FREEZE_S:
                print("  stream content frozen (game may still be live) — stopping sampling")
                break
            # Enemy points need a spell in hand; with no plays nobody cycles the
            # deck, so give up on the point instead of deadlocking.
            if is_enemy:
                if enemy_wait_start is None:
                    enemy_wait_start = time.perf_counter()
                elif time.perf_counter() - enemy_wait_start > 10.0:
                    enemy_wait_start = None
                    print(f"  enemy point skipped (no spell cycled in): screen={point}")
                    index += 1
                    continue
            else:
                enemy_wait_start = None
            now = time.perf_counter()
            if now - last_wait_log > 4.0:  # visibility: why are we waiting
                last_wait_log = now
                player = (next((p for p in latest["frame"].players if p.owner == client_owner), None)
                          if latest["frame"] else None)
                if player is None:
                    print(f"  waiting: client owner unresolved (owner={client_owner})")
                else:
                    hand = ", ".join(
                        f"s{c.slot}:{adapter.card_specs.get(c.card_id).name if adapter.card_specs.get(c.card_id) else c.card_id}"
                        for c in player.hand)
                    print(f"  waiting: elixir={player.elixir:.0f} hand=[{hand}]")
            continue  # wait for elixir / a deployable card
        slot, _cost, card_id = choice
        index += 1
        executor.tap_screen(*slots_px[str(slot)])
        time.sleep(0.04)
        executor.tap_screen(*point)
        mode = "await"
        if sampling_started is None:
            sampling_started = time.perf_counter()
        deadline = time.perf_counter() + AWAIT_SPAWN_S
        await_frames = 0
        snapshot = set(seen)
        pending_slot = slot
        pending_card_id = card_id
        pending_is_enemy = is_enemy
        pending_screen = point

    # Stream and executor stay up through fitting AND verification — matches
    # 9-12 "stream lost" was this cleanup running before the verify phase and
    # killing the collector we were about to verify with.
    if len(records) < MIN_SAMPLES:
        sys.exit(f"only {len(records)} usable samples (need {MIN_SAMPLES}); aborting without fit "
                 f"(slot successes: {dict(slot_stats)})")

    world = [tuple(r["world"]) for r in records]
    screen = [tuple(r["screen"]) for r in records]
    h, inliers = ransac_homography(world, screen, RANSAC_ITERS, INLIER_PX, RANSAC_SEED)
    if len(inliers) < MIN_SAMPLES:
        sys.exit(f"RANSAC kept only {len(inliers)}/{len(records)} inliers (need {MIN_SAMPLES}); aborting")
    for i, record in enumerate(records):
        record["inlier"] = i in inliers
    rejected = [r for r in records if not r["inlier"]]
    for r in rejected:
        print(f"  RANSAC outlier: slot={r['slot']} screen={r['screen']} world={r['world']} card={r['card_id']}")
    all_residuals = reprojection_residuals(h, world, screen)
    residuals = [all_residuals[i] for i in inliers]
    print(f"fit: RANSAC {len(inliers)}/{len(records)} inliers; residual "
          f"median={median(residuals):.1f}px max={max(residuals):.1f}px")

    # Probe points inside the fitted region (never extrapolate the homography);
    # client's owner slot varies per match, so hardcoding a half is wrong.
    xs = [r["world"][0] for r in records]
    ys = [r["world"][1] for r in records]
    wx0, wx1, wy0, wy1 = min(xs), max(xs), min(ys), max(ys)
    mx, my = (wx1 - wx0) * 0.15, (wy1 - wy0) * 0.15
    verify_points = [(wx0 + mx, wy0 + my), (wx1 - mx, wy0 + my),
                     ((wx0 + wx1) / 2, (wy0 + wy1) / 2), (wx0 + mx, wy1 - my)]

    verify = []
    stream_lost = False
    print(f"verification starts (client_owner={client_owner})")
    for point in verify_points:
        if not match_alive(latest):
            print("  stream lost before verification — skipping remaining probes")
            stream_lost = True
            break
        ok = verify_probe(executor, frames, latest, seen, h, point, slots_px, adapter, client_owner)
        verify.append(ok)
        if ok:
            print(f"  verify world={point} -> readback={ok['readback']} error={ok['error_world']:.0f} world units")
        else:
            print(f"  verify world={point} -> no spawn")
    ok_errors = [v["error_world"] for v in verify if v]

    if stream_lost and len(ok_errors) < 2:
        verdict = "CALIBRATION UNVERIFIED (stream lost)"
    elif len(ok_errors) >= 2 and max(ok_errors) < 500:
        verdict = "CALIBRATION OK"
    else:
        verdict = "CALIBRATION WEAK (verify errors)"
    payload = {
        "version": "pipeline-calibration.v1",
        "serial": serial,
        "client_owner": client_owner,
        "screen": {"width": int(layout["screen_width"]), "height": int(layout["screen_height"])},
        "card_slots_px": slots_px,
        "world_to_screen": [[float(v) for v in row] for row in h],
        "samples": records,
        "inliers": inliers,
        "rejected": [{"slot": r["slot"], "screen": r["screen"], "world": r["world"],
                      "card_id": r["card_id"]} for r in rejected],
        "ransac": {"iters": RANSAC_ITERS, "inlier_px": INLIER_PX, "seed": RANSAC_SEED},
        "reprojection_residual_px": {"median": median(residuals), "max": max(residuals)},
        "verify": verify,
        "slot_successes": dict(slot_stats),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
    }
    out_path = CALIBRATION_DIR / f"calibration_{serial_tag}.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{verdict} -> {out_path}")
    executor.close()
    stream.stop()
    return 0


class BattleStream:
    """ptrace collector supervisor.

    Matches 9-11: mid-match the collector's snapshot froze (frames kept
    arriving with a stale elixir/hand) and then output ceased — while the game
    itself went on (confirmed by the user watching match 11). Re-attach
    instead of going blind: kill any on-device residue, start a fresh
    collector, keep feeding the same frame queue. `seen` is owned by the
    caller and never reset, so re-reads of live objects can't pose as fresh
    spawns.
    """

    def __init__(self, serial: str, pid: int, frames: queue.Queue, log=None):
        self.serial = serial
        self.pid = pid
        self.frames = frames
        self.log = log if log is not None else print
        self.proc = None
        self.stopped = False
        self.restarts = 0
        self.last_frame_time = 0.0
        self.last_attempt = 0.0

    def start(self) -> None:
        self._spawn()
        threading.Thread(target=self._watchdog, daemon=True).start()

    def stop(self) -> None:
        self.stopped = True
        if self.proc is not None:
            self.proc.terminate()

    def _spawn(self) -> None:
        self.proc = subprocess.Popen(
            [ADB, "-s", self.serial, "shell",
             f'su 0 sh -c "{REMOTE_BIN} {self.pid} {CAPTURE_INTERVAL_MS}"'],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        normalizer = FrameNormalizer()  # per-attach: no state bleeds across attaches

        def read() -> None:
            for line in self.proc.stdout:  # type: ignore[union-attr]
                frame = normalizer.feed(line)
                if frame is not None:
                    self.last_frame_time = time.perf_counter()
                    self.frames.put(frame)

        threading.Thread(target=read, daemon=True).start()

    def _watchdog(self) -> None:
        while not self.stopped:
            time.sleep(1.0)
            if self.last_frame_time == 0.0:
                continue  # first attach never produced a frame — startup failure, not ours to loop on
            now = time.perf_counter()
            if now - self.last_frame_time > STREAM_QUIET_S and now - self.last_attempt > 5.0:
                self._reconnect()

    def _reconnect(self) -> None:
        if self.stopped or self.restarts >= MAX_STREAM_RESTARTS:
            return
        self.restarts += 1
        self.last_attempt = time.perf_counter()
        self.log(f"  [stream] went quiet — re-attaching collector ({self.restarts}/{MAX_STREAM_RESTARTS})")
        subprocess.run([ADB, "-s", self.serial, "shell",
                        'su 0 sh -c "kill $(pidof cr_battle_stream) 2>/dev/null; true"'],
                       capture_output=True, timeout=15)
        time.sleep(0.4)
        if not self.stopped:
            self._spawn()


def build_samples(layout: dict, with_enemy: bool) -> tuple[list[tuple[tuple[float, float], bool]], int]:
    """Own-half grid first in a wide-stride order (an early stop keeps spread),
    then the enemy half (spells only, opt-in via --with-enemy).

    Troops tapped on the enemy half clamp to the bridge line — a boundary value
    that silently poisons the fit — so enemy points need spells, and since the
    hand only cycles when cards are played they burn real match time. The
    own-half grid carries the fit; enemy rows add little (match 8).
    """
    width = float(layout["screen_width"])
    height = float(layout["screen_height"])
    cols = layout.get("sample_cols", 5)

    def grid_points(region: dict, rows: int) -> list[tuple[float, float]]:
        return [
            (
                width * (region["x0f"] + ((c + 0.5) / cols) * (region["x1f"] - region["x0f"])),
                height * (region["y0f"] + ((r + 0.5) / rows) * (region["y1f"] - region["y0f"])),
            )
            for r in range(rows)
            for c in range(cols)
        ]

    own = grid_points(layout["sample_region"], layout.get("sample_rows", 3))
    stride = 7  # visit points out of row-major order; coprime with the count
    while math.gcd(stride, len(own)) != 1:
        stride += 1
    points = [(own[(i * stride) % len(own)], False) for i in range(len(own))]
    n_own = len(points)
    if with_enemy and layout.get("enemy_sample_region"):
        points.extend((p, True) for p in grid_points(layout["enemy_sample_region"],
                                                     layout.get("enemy_rows", 1)))
    return points, n_own


def match_alive(latest: dict) -> bool:
    """Worth tapping: the stream still delivers frames (or the supervisor is
    re-attaching to resume them).

    Never gates on battle content: the stream can freeze its elixir/hand
    snapshot while the match goes on — matches 9-11 starved probes on stale
    readiness data while the user watched the game live. Taps are the ground
    truth; the spawn readback decides. Before the first frame there is
    nothing to judge yet — keep waiting.
    """
    if latest["frame"] is None:
        return True
    return time.perf_counter() - latest["time"] <= STREAM_STALL_S


def resolve_client_owner(frame, my_deck: set, cached: int | None) -> int | None:
    """The player whose hand comes from our pre-filled deck is the client.

    serve_policy's flow feeds decks externally anyway, so the deck is a known
    input — no need for probes. players[] order and the client's owner slot
    vary per match (verified owner0 and owner1), so never assume players[0].
    """
    if frame is None:
        return cached
    if cached is not None:
        p = next((p for p in frame.players if p.owner == cached), None)
        if p is not None and p.hand:
            return cached
    best, best_hits = None, 0
    for p in frame.players:
        hits = sum(1 for c in p.hand if c.card_id in my_deck)
        if p.hand and hits >= 3 and hits > best_hits:
            best, best_hits = p, hits
    return best.owner if best is not None else cached


def pick_slot(frame, adapter, slot_stats: Counter, client_owner: int,
              allowed: set = SPAWNABLE_ENTITY_KINDS) -> tuple[int, float, int] | None:
    """Affordable deployable hand card, preferring the least-validated slot."""
    if frame is None:
        return None
    player = next((p for p in frame.players if p.owner == client_owner), None)
    if player is None:
        return None
    elixir = float(player.elixir)
    options = []
    for card in player.hand:
        spec = adapter.card_specs.get(card.card_id)
        if spec is not None:
            kind = KIND_MAP.get(spec.kind)
            cost = float(spec.elixir_cost or 0.0)
        else:
            # Spell-only cards absent from FirstLight's roster (Clone/Rage/...):
            # treat as 3-cost spells; a wrong cost just burns a retry.
            kind, cost = "spell", 3.0
        if kind not in allowed:
            continue
        if cost <= elixir + 1e-6:
            options.append((int(card.slot), cost, int(card.card_id)))
    if not options:
        return None
    # Prefer static readbacks: spells/buildings don't move between tap and
    # first stream frame, fast troops do (their spawn point reads shifted).
    kind_rank = {"spell": 0, "building": 1, "troop": 2}

    def _kind_rank(card_id: int) -> int:
        spec = adapter.card_specs.get(card_id)
        kind = KIND_MAP.get(spec.kind) if spec else "spell"
        return kind_rank.get(kind, 3)

    options.sort(key=lambda item: (slot_stats.get(item[0], 0), _kind_rank(item[2]), item[1]))
    return options[0]


def verify_probe(executor, frames, latest, seen, h, world_point, slots_px, adapter,
                 client_owner: int | None) -> dict | None:
    """Deploy at a world point through the fitted map; measure end-to-end error.

    Readiness never gates a probe: the stream can freeze its elixir/hand
    snapshot while the match goes on (matches 9-11 starved probes on stale
    data the user watched live), and a tap costs nothing when the real hand
    can't pay. Attempts rotate slots so any affordable real card deploys; the
    spawn readback is the only verdict.
    """
    sx, sy = apply_h(h, *world_point)
    for attempt in range(VERIFY_ATTEMPTS):
        choice = pick_slot(latest["frame"], adapter, Counter(), client_owner) if attempt == 0 else None
        slot = choice[0] if choice else attempt % len(slots_px)
        snapshot = set(seen)
        label = f"card={choice[2]}" if choice else "data-blind"
        print(f"  verify probe attempt {attempt + 1}/{VERIFY_ATTEMPTS}: slot={slot} {label} "
              f"-> screen=({sx:.0f},{sy:.0f})")
        executor.tap_screen(*slots_px[str(slot)])
        time.sleep(0.04)
        executor.tap_screen(sx, sy)
        deadline = time.perf_counter() + AWAIT_SPAWN_S + 0.6
        while time.perf_counter() < deadline:
            try:
                frame = frames.get(timeout=0.3)
            except queue.Empty:
                continue
            latest["frame"] = frame
            latest["time"] = time.perf_counter()
            seen.update(o.object_id for o in frame.objects)
            # Our own spawn carries our owner slot; requiring it keeps enemy
            # deployments (active in overtime) out of the readback.
            fresh = [o for o in frame.objects if o.object_id not in snapshot and o.card_id > 0
                     and not (TOWER_OBJECT_ID_MIN <= o.object_id <= TOWER_OBJECT_ID_MAX)
                     and (client_owner is None or o.owner == client_owner)]
            if fresh:
                unit = fresh[0]
                error = float(np.hypot(unit.x - world_point[0], unit.y - world_point[1]))
                return {"world": list(world_point), "readback": [int(unit.x), int(unit.y)],
                        "error_world": error, "attempts": attempt + 1}
        if not match_alive(latest):
            break  # stream dead and restarts exhausted — further taps are unreadable
    return None


if __name__ == "__main__":
    raise SystemExit(main())
