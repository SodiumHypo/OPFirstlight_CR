"""Battle-frame ingestion: parse and normalize cr_battle_stream NDJSON frames.

Input contract (pipeline/native/cr_battle_stream.c):
  {"event":"battle_frame","seq":N,"t_ms":ms,"battle_active":bool,"tick":int,
   "root_slot":"tracked|none",
   "players":[{"owner":0,"acctHi":u,"acctLo":u,"elixirRaw":i,"elixir":f,
               "hand":[{"slot":i,"deckSlot":i,"cardId":i}...],
               "cycle":[cardId...],
               "deck":[{"slot":i,"cardId":i}...]}, ...],   # empty when inactive
   "objects":[{"id":u,"kind":i,"owner":i,"x":i,"y":i,"tx":i,"ty":i,
               "cardId":i,"level":i,"hp":i,"maxHp":i,"shield":i,"maxShield":i}...],
   "read_us":us}

Normalization rules:
  - malformed/foreign lines are dropped (returned as None) — never fatal
  - inactive frames collapse to a sentinel (battle_active=False, no players)
  - duplicate ticks within a battle are dropped (the engine tick is the
    authoritative clock; capture may sample faster than the game simulates)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HandCard:
    slot: int
    deck_slot: int
    card_id: int


@dataclass(frozen=True)
class PlayerFrame:
    owner: int
    elixir: float
    hand: tuple[HandCard, ...]
    cycle: tuple[int, ...]
    deck: tuple[int, ...]


@dataclass(frozen=True)
class ObjectFrame:
    object_id: int
    kind: int
    owner: int
    x: int
    y: int
    tx: int
    ty: int
    card_id: int
    level: int
    hp: int
    max_hp: int
    shield: int
    max_shield: int


@dataclass(frozen=True)
class BattleFrame:
    seq: int
    t_ms: int
    tick: int
    players: tuple[PlayerFrame, ...]
    objects: tuple[ObjectFrame, ...]


@dataclass
class FrameStats:
    seen: int = 0
    dropped_malformed: int = 0
    dropped_inactive: int = 0
    dropped_duplicate_tick: int = 0
    emitted: int = 0
    tick_regressions: int = 0  # tick went backwards: battle restart / post-battle lobby
    last_tick: int | None = None
    last_seq: int = -1
    gaps: list = field(default_factory=list)  # (prev_seq, seq) pairs with seq jumps


def parse_frame_line(line: str) -> BattleFrame | None:
    """Parse one NDJSON line.

    Raises json.JSONDecodeError on unparsable lines (caller classifies as
    malformed); returns None for valid JSON that is not an active battle frame
    (standby traffic, fatal reports, ...).
    """
    raw = json.loads(line)
    if not isinstance(raw, dict) or raw.get("event") != "battle_frame":
        return None
    if not raw.get("battle_active"):
        return None
    tick = raw["tick"]
    if not isinstance(tick, int) or tick < 0:
        return None
    players = tuple(
        PlayerFrame(
            owner=int(p["owner"]),
            elixir=float(p["elixir"]),
            hand=tuple(HandCard(int(c["slot"]), int(c["deckSlot"]), int(c["cardId"])) for c in p["hand"]),
            cycle=tuple(int(v) for v in p["cycle"]),
            deck=tuple(int(d["cardId"]) for d in p["deck"]),
        )
        for p in raw["players"]
    )
    objects = tuple(
        ObjectFrame(
            object_id=int(o["id"]),
            kind=int(o["kind"]),
            owner=int(o["owner"]),
            x=int(o["x"]),
            y=int(o["y"]),
            tx=int(o["tx"]),
            ty=int(o["ty"]),
            card_id=int(o["cardId"]),
            level=int(o["level"]),
            hp=int(o["hp"]),
            max_hp=int(o["maxHp"]),
            shield=int(o["shield"]),
            max_shield=int(o["maxShield"]),
        )
        for o in raw["objects"]
    )
    return BattleFrame(seq=int(raw["seq"]), t_ms=int(raw["t_ms"]), tick=tick, players=players, objects=objects)


class FrameNormalizer:
    """Stateful frame filter: drops inactive and duplicate-tick frames, tracks gaps."""

    def __init__(self) -> None:
        self.stats = FrameStats()

    def feed(self, line: str) -> BattleFrame | None:
        stats = self.stats
        stats.seen += 1
        try:
            frame = parse_frame_line(line)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            stats.dropped_malformed += 1
            return None
        if frame is None:
            # inactive frames are normal standby traffic
            stats.dropped_inactive += 1
            return None
        if frame.seq != stats.last_seq + 1 and stats.last_seq >= 0:
            stats.gaps.append((stats.last_seq, frame.seq))
        stats.last_seq = frame.seq
        if stats.last_tick is not None:
            if frame.tick == stats.last_tick:
                # frozen state (lobby or stalled reader): emit once, then drop
                stats.dropped_duplicate_tick += 1
                return None
            if frame.tick < stats.last_tick:
                # tick went backwards: a battle ended (or a new one started)
                stats.tick_regressions += 1
        stats.last_tick = frame.tick
        stats.last_tick = frame.tick
        stats.emitted += 1
        return frame
