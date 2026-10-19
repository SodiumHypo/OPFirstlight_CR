"""Summarize a cr_battle_stream capture file (P1 sanity gate).

Usage: python analyze_capture.py <capture.ndjson>
Reports: frames/ticks, player state coverage (elixir/hand/deck), object
integrity (hp coverage, kinds, ids), root stability, capture gaps.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingest"))
from frames import FrameNormalizer  # noqa: E402


def main() -> int:
    path = Path(sys.argv[1])
    normalizer = FrameNormalizer()
    frames = []
    first_t = last_t = None
    battle_windows = []  # (start_t, end_t)
    cur = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            fr = normalizer.feed(line)
            s = normalizer.stats
            if first_t is None:
                first_t = s.last_seq  # noqa: F841 (kept for clarity below)
            try:
                raw = json.loads(line)
                t = raw.get("t_ms")
                active = raw.get("battle_active")
            except json.JSONDecodeError:
                continue
            if last_t is None:
                first_t = t
            last_t = t
            if active and cur is None:
                cur = t
            if not active and cur is not None:
                battle_windows.append((cur, t))
                cur = None
            if fr is not None:
                frames.append(fr)
    if cur is not None:
        battle_windows.append((cur, last_t))

    print(f"file: {path.name}")
    dur_s = (last_t - first_t) / 1000.0 if last_t and first_t else 0
    print(f"capture span: {dur_s:.1f}s   stats: {normalizer.stats}")

    if not frames:
        print("NO battle frames captured (no advancing tick stream).")
        return 0

    ticks = [f.tick for f in frames]
    print(f"battle frames: {len(frames)}   ticks: {min(ticks)}..{max(ticks)} (span {max(ticks)-min(ticks)})")
    for start, end in battle_windows:
        print(f"  active window: {(end-start)/1000.0:.1f}s")

    # player state coverage
    for owner in (0, 1):
        pf = [f.players[owner] for f in frames if len(f.players) > owner]
        if not pf:
            print(f"player {owner}: NO DATA")
            continue
        el = [p.elixir for p in pf]
        deck_ok = sum(1 for p in pf if len(p.deck) == 8)
        hand_ok = sum(1 for p in pf if len(p.hand) == 4)
        cyc = sum(len(p.cycle) for p in pf) / len(pf)
        print(
            f"player {owner}: frames={len(pf)} elixir {min(el):.2f}..{max(el):.2f} "
            f"deck8={deck_ok}/{len(pf)} hand4={hand_ok}/{len(pf)} avg_cycle={cyc:.2f}"
        )
        decks = {p.deck for p in pf}
        if len(decks) == 1:
            print(f"  deck (stable): {sorted(next(iter(decks)))}")
        else:
            print(f"  deck VARIANTS: {len(decks)}")

    # object integrity
    n_obj_frames = sum(1 for f in frames if f.objects)
    kinds = Counter()
    ids = set()
    hp_ok = max_ok = both_coords = 0
    for f in frames:
        for o in f.objects:
            kinds[o.kind] += 1
            ids.add(o.object_id)
            if o.hp >= 0:
                hp_ok += 1
            if o.max_hp >= 0:
                max_ok += 1
            if (o.x, o.y) != (o.tx, o.ty):
                both_coords += 1
    total_objs = sum(kinds.values())
    print(
        f"objects: frames_with={n_obj_frames}/{len(frames)} unique_ids={len(ids)} "
        f"records={total_objs} hp_present={hp_ok} maxHp_present={max_ok} moving={both_coords}"
    )
    print(f"  kinds: {dict(kinds.most_common(8))}")

    # sample: one object trajectory
    if ids:
        sample_id = sorted(ids)[0]
        traj = [(f.tick, o.x, o.y) for f in frames for o in f.objects if o.object_id == sample_id][:8]
        print(f"  sample id={sample_id} first positions: {traj}")

    # tick cadence
    if len(frames) > 2:
        import statistics

        diffs = [b.tick - a.tick for a, b in zip(frames, frames[1:]) if b.tick > a.tick]
        if diffs:
            print(f"tick deltas: median={statistics.median(diffs)} p95={sorted(diffs)[int(len(diffs)*0.95)-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
