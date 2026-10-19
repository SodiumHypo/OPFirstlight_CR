"""Probe FirstLight V4 vocabularies against a real battle capture.

Answers (P1 adapter prerequisites):
  1. do capture deck/hand card IDs resolve in the competitive card catalog?
  2. can entity card IDs serve as native_data_global_id (archetype resolution)?
  3. which object kinds/IDs look like towers (for TowerStateV1 mapping)?

Run from anywhere:
  FirstLight venv python -I probe_firstlight_vocab.py <capture.ndjson>
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

FIRSTLIGHT_ROOT = Path(__file__).resolve().parents[2] / "firstlight"
sys.path.insert(0, str(FIRSTLIGHT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingest"))
from frames import FrameNormalizer  # noqa: E402


def main() -> int:
    capture = Path(sys.argv[1])

    from native_runner.training.v4.factory import production_semantic_bundle

    bundle = production_semantic_bundle()
    cat = bundle.entity_archetype_catalog
    cards = bundle.card_catalog

    keys = list(cat.archetype_keys)
    print(f"card_specs: {len(bundle.card_specs)}   archetype keys: {len(keys)}   character_global_ids: {len(cat.character_global_ids)}")
    print(f"archetype key samples: {keys[:12]}")
    print(f"character_global_id samples: {list(cat.character_global_ids)[:8]}")

    normalizer = FrameNormalizer()
    deck_cards: set[int] = set()
    hand_cards: set[int] = set()
    kind_card: dict[int, Counter] = defaultdict(Counter)          # kind -> cardId -> count
    kind_ids: dict[int, set] = defaultdict(set)                    # kind -> unique object ids
    kind_pos: dict[int, Counter] = defaultdict(Counter)            # kind -> (cardId) -> mean pos sample
    frames = 0
    with capture.open("r", encoding="utf-8") as fh:
        for line in fh:
            frame = normalizer.feed(line)
            if frame is None:
                continue
            frames += 1
            for p in frame.players:
                deck_cards.update(p.deck)
                hand_cards.update(c.card_id for c in p.hand)
            for o in frame.objects:
                kind_card[o.kind][o.card_id] += 1
                kind_ids[o.kind].add(o.object_id)
    print(f"\nframes: {frames}")
    print(f"deck card ids: {sorted(deck_cards)}")
    print(f"hand card ids: {sorted(hand_cards)}")

    def card_vocab(cid: int) -> object:
        try:
            return cards.vocab_id(cid)
        except Exception as error:  # noqa: BLE001
            return f"ERR:{error}"

    def card_in_specs(cid: int) -> bool:
        return int(cid) in bundle.card_specs

    print("\n-- card catalog resolution (deck ∪ hand) --")
    for cid in sorted(deck_cards | hand_cards):
        print(f"  {cid}: card_vocab={card_vocab(cid)} in_card_specs={card_in_specs(cid)}")

    print("\n-- per-kind object stats --")
    for kind in sorted(kind_card):
        ids = kind_ids[kind]
        tops = kind_card[kind].most_common(10)
        print(f"  kind={kind}: unique_ids={len(ids)} records={sum(kind_card[kind].values())} top_cardIds={tops}")

    print("\n-- entity archetype resolution (cardId as native_data_global_id) --")
    candidates = sorted({cid for kind in kind_card for cid in kind_card[kind]} | deck_cards | hand_cards)
    for cid in candidates:
        vocab = cat.runtime_global_vocab_id(cid)
        print(f"  {cid}: runtime_global_vocab_id={vocab} {'<-- RESOLVED' if vocab != 1 else ''}")

    print("\n-- known special global ids sanity --")
    for gid in (159_000_000, 159_000_001, 159_000_002, 159_000_004):
        print(f"  tower troop {gid}: {cat.runtime_global_vocab_id(gid)}")

    # card_id -> spec.name -> character global id -> archetype resolution
    name_to_gid = {name: gid for gid, name in cat.character_global_ids}
    archetype_keys = set(keys)

    def card_to_gid(spec: object) -> int | None:
        for form in (str(spec.name), *(str(f) for f in spec.summoned_forms)):  # type: ignore[attr-defined]
            gid = name_to_gid.get(form)
            if gid is not None and f"form:{form}" in archetype_keys:
                return gid
        return None

    print("\n-- card -> character form resolution (all 122 competitive cards) --")
    resolved: dict[int, int] = {}
    unresolved_specs: list[tuple[int, str, str]] = []
    for cid, spec in sorted(bundle.card_specs.items()):
        gid = card_to_gid(spec)
        if gid is not None:
            resolved[cid] = gid
        else:
            unresolved_specs.append((cid, str(spec.name), str(spec.kind)))
    print(f"  resolved via name->gid: {len(resolved)}/{len(bundle.card_specs)}")
    for cid, name, kind in unresolved_specs[:40]:
        vocab = cat.form_vocab_id(name)
        print(f"  UNRESOLVED {cid} name={name!r} kind={kind} form_vocab={vocab}")
    bad = {cid: gid for cid, gid in resolved.items() if cat.runtime_global_vocab_id(gid) == 1}
    print(f"  resolved but runtime_global_vocab_id==UNKNOWN: {len(bad)}")

    print("\n-- capture entity cardIds -> character gid --")
    entity_cards = sorted({cid for kind, counter in kind_card.items() for cid in counter if cid > 0})
    for cid in entity_cards:
        gid = resolved.get(cid)
        status = f"gid={gid} vocab={cat.runtime_global_vocab_id(gid)}" if gid else "NO MAPPING"
        kinds_with = [k for k, c in kind_card.items() if cid in c]
        print(f"  {cid} (kinds={kinds_with}): {status}")

    print("\n-- kind 13 candidate towers: per-id mean position & presence --")
    pos_acc: dict[int, list[list[int]]] = defaultdict(lambda: [0, 0, 0])  # id -> [sum_x, sum_y, n]
    with capture.open("r", encoding="utf-8") as fh:
        for line in fh:
            frame = normalizer.feed(line)
            if frame is None:
                continue
            for o in frame.objects:
                if o.kind == 13:
                    acc = pos_acc[o.object_id]
                    acc[0] += o.x
                    acc[1] += o.y
                    acc[2] += 1
    for oid, (sx, sy, n) in sorted(pos_acc.items()):
        print(f"  tower id={oid}: mean=({sx / n:.0f}, {sy / n:.0f}) frames={n}")

    print("\n-- summoned_forms of unresolved capture cards --")
    for cid in (26000012, 26000049, 26000067):
        spec = bundle.card_specs.get(cid)
        if spec is not None:
            print(f"  {cid} name={spec.name!r} summoned_cards={spec.summoned_cards} summoned_forms={spec.summoned_forms}")

    print("\n-- 203000018 / fuzzy form names --")
    print(f"  name for gid 203000018: {name_to_gid.get(203000018)}")
    for pattern in ("Bat", "Golem", "Skeleton", "TowerPrincess", "Tower"):
        hits = [name for name in name_to_gid if pattern.lower() in name.lower()]
        print(f"  forms containing {pattern!r}: {hits[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
