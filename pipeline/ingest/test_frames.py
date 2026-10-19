"""Self-test for pipeline/ingest/frames.py with synthetic frames (no device needed).

Run: python -I test_frames.py   (exits 0 on success, raises on failure)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from frames import FrameNormalizer, parse_frame_line


def frame_json(seq, tick, *, active=True, objects=None, hand=None, deck=None, elixir=33210):
    return json.dumps(
        {
            "event": "battle_frame",
            "seq": seq,
            "t_ms": 1000 + seq,
            "battle_active": active,
            "tick": tick if active else -1,
            "root_slot": "tracked" if active else "none",
            "players": (
                [
                    {
                        "owner": 0,
                        "acctHi": 111,
                        "acctLo": 222,
                        "elixirRaw": elixir,
                        "elixir": elixir / 10000.0,
                        "hand": hand if hand is not None else [{"slot": 0, "deckSlot": 0, "cardId": 26000003}],
                        "cycle": [26000009],
                        "deck": deck if deck is not None else [{"slot": 0, "cardId": 26000003}],
                    },
                    {
                        "owner": 1,
                        "acctHi": 333,
                        "acctLo": 444,
                        "elixirRaw": 50000,
                        "elixir": 5.0,
                        "hand": [],
                        "cycle": [],
                        "deck": [],
                    },
                ]
                if active
                else []
            ),
            "objects": objects or [],
            "read_us": 1234,
        }
    )


def main() -> None:
    # 1. valid frame parses
    f = parse_frame_line(frame_json(0, 120))
    assert f is not None and f.tick == 120 and len(f.players) == 2
    assert f.players[0].hand[0].card_id == 26000003
    assert abs(f.players[0].elixir - 3.321) < 1e-9

    # 2. inactive / fatal / garbage lines -> None or raise
    assert parse_frame_line(frame_json(1, -1, active=False)) is None
    assert parse_frame_line(json.dumps({"event": "fatal", "error": "x"})) is None
    try:
        parse_frame_line("not json")
        raise AssertionError("expected JSONDecodeError")
    except json.JSONDecodeError:
        pass

    # 3. normalizer: emits new ticks, drops duplicates, counts malformed/inactive
    n = FrameNormalizer()
    assert n.feed(frame_json(0, 100)) is not None
    assert n.feed(frame_json(1, 100)) is None  # duplicate tick
    assert n.feed(frame_json(2, 102)) is not None
    assert n.feed(frame_json(3, -1, active=False)) is None
    assert n.feed("garbage") is None
    s = n.stats
    assert (s.emitted, s.dropped_duplicate_tick, s.dropped_inactive, s.dropped_malformed) == (2, 1, 1, 1), s
    # last_seq tracks only the active-frame stream; standby/garbage lines don't advance it
    assert s.last_tick == 102 and s.last_seq == 2 and not s.gaps

    # 4. gap detection
    n2 = FrameNormalizer()
    n2.feed(frame_json(5, 10))
    n2.feed(frame_json(9, 12))
    assert n2.stats.gaps == [(5, 9)], n2.stats.gaps

    # 4b. tick regression (battle end -> lobby tick 0) emits and is counted
    n3 = FrameNormalizer()
    n3.feed(frame_json(0, 3600))
    assert n3.feed(frame_json(1, 0)) is not None  # regression: new cycle
    assert n3.feed(frame_json(2, 0)) is None  # frozen lobby: dedup
    assert n3.stats.tick_regressions == 1 and n3.stats.emitted == 2

    # 5. objects round-trip
    obj = {
        "id": 5613443, "kind": 12, "owner": 0, "x": 9324, "y": 11243,
        "tx": 1000, "ty": 1200, "cardId": 26000003, "level": 9,
        "hp": 1400, "maxHp": 1400, "shield": -1, "maxShield": -1,
    }
    f2 = parse_frame_line(frame_json(20, 300, objects=[obj]))
    assert f2 is not None and f2.objects[0].object_id == 5613443 and f2.objects[0].max_hp == 1400

    print("test_frames: ALL OK")


if __name__ == "__main__":
    main()
