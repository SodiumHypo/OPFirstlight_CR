"""BattleFrame -> FirstLight V4 FAIR observation adapter.

Produces wire-format dicts accepted by native_runner.training.v4.serve_policy:
  episode_config(deck0, deck1, seed)  -> EpisodeConfigV1 mapping
  observation(frame, ...)             -> ObservationV1 mapping
  initial_elixir(frame)               -> {"0": elixir, "1": elixir}

Mapping rules (probed in tools/probe_firstlight_vocab.py and a full-match
differential replay against real captures):
  - towers: six fixed object ids 5000000-5000005 at standard 1v1 positions.
    Princess towers stream as kind 13, but king towers stream as kind 12 with
    card_id<=0 (and princesses flip to kind 12 in some state phases), so tower
    rows merge both kinds by object id, kind-13 rows winning on overlap.
  - entities: kind 15 only (deployed units with a resolvable character global
    id). Kind 14 rows echo attacks/projectiles with real card ids and hp and
    must not become entities.
  - spell areas (kind 12 with a card id) and the unresolvable 203000018 unit
    are dropped (P1 degradation; P2 reads the true LogicData global id in the
    C collector).

Coordinate system: native world units (18000x32000) pass through unchanged;
the V4 tensorizer applies owner-perspective flips itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

FIRSTLIGHT_ROOT = Path(__file__).resolve().parents[2] / "firstlight"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]  # workspace root (holds firstlight/ and pipeline/)
for _root in (str(WORKSPACE_ROOT), str(FIRSTLIGHT_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

from native_runner.arena import (  # noqa: E402
    GRID_HEIGHT,
    GRID_WIDTH,
    card_placement_mask,
    standard_active_tower_footprints,
)
from native_runner.contracts import CardKind, content_hash  # noqa: E402
from native_runner.training.v4.factory import production_semantic_bundle  # noqa: E402
from native_runner.training.v4.native_actions import MIRROR_CARD_ID  # noqa: E402

from pipeline.ingest.frames import BattleFrame, HandCard, ObjectFrame, PlayerFrame  # noqa: E402

KING_TOWER_X = 9000
SIDE_TOWER_TROOP_ID = 159_000_000  # Tower Princess (Null's Royale default)
# Tower slot ids are fixed by the game (stable across independent captures);
# they are the only reliable discriminator between tower rows and spell areas.
TOWER_OBJECT_ID_MIN = 5_000_000
TOWER_OBJECT_ID_MAX = 5_000_005
MAX_BATTLE_TICKS = 7200  # native_runner.match_factory.NATIVE_MATCH_END_TICK
WARMUP_TICKS = 130

ENTITY_KIND_BY_CARD_KIND = {
    CardKind.TROOP: "troop",
    CardKind.BUILDING: "building",
    CardKind.HERO: "hero",
}

_RULESET_ID: str | None = None


def ruleset_id() -> str:
    """Stable content-addressed ruleset identity for this pipeline's rules."""

    global _RULESET_ID
    if _RULESET_ID is None:
        _RULESET_ID = content_hash(
            {
                "version": "pipeline-live-ruleset.v1",
                "game_mode": 72000006,
                "arena": 54000001,
                "map_id": "standard-1v1",
                "placement": "firstlight-arena-v4",
            }
        )
    return _RULESET_ID


def _pair(x: float, y: float) -> list[float]:
    return [float(x), float(y)]


class FirstLightAdapter:
    """One actor seat per instance; pure construction, no I/O."""

    def __init__(self, *, actor_owner: int) -> None:
        if actor_owner not in (0, 1):
            raise ValueError("actor_owner must be 0 or 1")
        self.actor_owner = actor_owner
        bundle = production_semantic_bundle()
        self.card_specs = bundle.card_specs
        self.entity_archetype_catalog = bundle.entity_archetype_catalog
        self._name_to_gid = {name: gid for gid, name in bundle.entity_archetype_catalog.character_global_ids}
        self._card_gid: dict[int, int | None] = {}
        self._placements: dict[tuple[int, int], object] = {}
        # Mirror runtime tracking (live 2026-10-17: serve_policy rejects any
        # Mirror play whose runtime contract is incomplete — "Mirror play
        # runtime contract is incomplete" — so the actor's last played card
        # must be recovered across frames and mirrored into both the private
        # hand metadata and the placement mask entry).
        self._prev_hand_ids: tuple[int, ...] | None = None
        self._last_played: tuple[int, float] | None = None  # (card_id, static cost)
        self.stats = {"towers_skipped": 0, "entities_dropped": 0}

    # -- vocab mapping -----------------------------------------------------

    def card_global_id(self, card_id: int) -> int | None:
        """Map a card ID to its deployed character LogicData global ID.

        Card IDs (26xxxxx) are not entity global IDs; the deployed character is
        resolved through the spec name, falling back to the first summoned
        form for multi-unit cards (Bats -> Bat, SkeletonArmy -> Skeleton, ...).
        """

        cached = self._card_gid.get(card_id, 0)
        if cached != 0:
            return cached
        gid: int | None = None
        spec = self.card_specs.get(card_id)
        if spec is not None:
            for form in (str(spec.name), *(str(item) for item in spec.summoned_forms)):
                candidate = self._name_to_gid.get(form)
                if candidate is not None:
                    gid = candidate
                    break
        self._card_gid[card_id] = gid
        return gid

    def placement_mask(self, card_id: int, owner: int):
        """Cached arena.card_placement_mask for one card/owner."""

        key = (card_id, owner)
        mask = self._placements.get(key)
        if mask is None:
            spec = self.card_specs[card_id]
            mask = card_placement_mask(
                spec,
                owner,
                ruleset_id=ruleset_id(),
                form="base",
                active_tower_footprints=standard_active_tower_footprints(owner),
            )
            self._placements[key] = mask
        return mask

    # -- episode assembly --------------------------------------------------

    def episode_config(self, deck0: tuple[int, ...], deck1: tuple[int, ...], *, seed: int = 0) -> dict:
        return {
            "version": "episode-config.v1",
            "ruleset_id": ruleset_id(),
            "deck0": list(deck0),
            "deck1": list(deck1),
            "seed": int(seed),
            "observation_tier": "fair",
            "game_mode": 72000006,
            "arena": 54000001,
            "map_id": "standard-1v1",
            "tick_ms": 50,
            "decision_hz": 20.0 / 5.0,
            "environment": {
                "version": "environment-config.v1",
                "warmup_ticks": WARMUP_TICKS,
                "entity_limit": 256,
                "max_battle_ticks": MAX_BATTLE_TICKS,
            },
        }

    def initial_elixir(self, frame: BattleFrame) -> dict[str, float]:
        return {str(p.owner): float(p.elixir) for p in frame.players[:2]}

    # -- observation assembly ---------------------------------------------

    def observation(self, frame: BattleFrame, *, episode_id: str, decks: dict[int, tuple[int, ...]]) -> dict:
        if len(frame.players) < 2:
            raise ValueError("battle frame is missing player rows")
        by_owner = {p.owner: p for p in frame.players[:2]}
        towers, tower_ids = self._towers(frame)
        entities = self._entities(frame)
        mirror_target = self._mirror_target(by_owner[self.actor_owner].hand)
        return {
            "version": "observation.v1",
            "tier": "fair",
            "tick": int(frame.tick),
            "owner": self.actor_owner,
            "time": {},
            "phase": "regular",
            "episode_id": episode_id,
            "players": [
                self._player(by_owner[0], decks[0], tower_ids.get(0, ()), mirror_target),
                self._player(by_owner[1], decks[1], tower_ids.get(1, ()), mirror_target),
            ],
            "towers": towers,
            "entities": entities,
            "events": [],
            "pending_actions": [],
            "action_mask": self._action_mask(by_owner[self.actor_owner], mirror_target),
            "metadata": {"source": "pipeline-capture", "capture_seq": int(frame.seq)},
        }

    def _mirror_target(self, hand: tuple[HandCard, ...]) -> tuple[int, float] | None:
        """Track the actor's last played card across observation calls.

        Returns (card_id, static_cost) of the card Mirror would copy, or None
        while the target is unknown/ambiguous — the Mirror hand slot is masked
        illegal in that case (nothing has been played yet, or two cards left
        the hand between observations so the played card is ambiguous).
        """

        ids = tuple(sorted(int(card.card_id) for card in hand))
        if self._prev_hand_ids is not None:
            played = [card_id for card_id in self._prev_hand_ids if card_id not in ids]
            if len(played) == 1:
                spec = self.card_specs.get(played[0])
                if spec is not None:
                    self._last_played = (played[0], float(spec.elixir_cost or 0.0))
            elif len(played) > 1:
                self._last_played = None  # ambiguous — disable Mirror rather than guess
        self._prev_hand_ids = ids
        if self._last_played is None or self._last_played[0] == MIRROR_CARD_ID:
            return None  # Mirror cannot copy itself
        return self._last_played

    def _hand_runtime_entry(self, card: HandCard, mirror_target: tuple[int, float] | None) -> dict:
        """One hand_runtime_by_slot entry, shaped for the V4 Mirror contract."""

        if int(card.card_id) == MIRROR_CARD_ID and mirror_target is not None:
            effective_id = int(mirror_target[0])
            effective_cost = float(mirror_target[1]) + 1.0
        else:
            spec = self.card_specs.get(card.card_id)
            effective_id = int(card.card_id)
            effective_cost = float(spec.elixir_cost or 0.0) if spec is not None else 0.0
        return {
            "hand_slot": int(card.slot),
            "visible_card_id": int(card.card_id),
            "effective_card_id": effective_id,
            "effective_cost": effective_cost,
            "form_code": 0,
        }

    def _player(self, player: PlayerFrame, deck: tuple[int, ...], tower_ids: tuple[int, ...],
                mirror_target: tuple[int, float] | None = None) -> dict:
        if player.owner == self.actor_owner:
            hand = sorted(player.hand, key=lambda card: card.slot)
            hand_cards = [card.card_id for card in hand]
            metadata = {
                "hand_slot_by_card": {str(card.card_id): int(card.slot) for card in hand},
                "hand_runtime_by_slot": {
                    str(card.slot): self._hand_runtime_entry(card, mirror_target) for card in hand
                },
            }
            return {
                "owner": int(player.owner),
                "crowns": 0,
                "tower_ids": list(tower_ids),
                "private_state_visible": True,
                "elixir_exact": float(player.elixir),
                "hand": hand_cards,
                "next_card": int(player.cycle[0]) if player.cycle else None,
                "deck": list(deck),
                "cycle": list(player.cycle),
                "revealed_cards": [],
                "metadata": metadata,
            }
        return {
            "owner": int(player.owner),
            "crowns": 0,
            "tower_ids": list(tower_ids),
            "private_state_visible": False,
            "revealed_cards": [],
        }

    def _towers(self, frame: BattleFrame) -> tuple[list[dict], dict[int, tuple[int, ...]]]:
        towers: list[dict] = []
        by_owner: dict[int, list[int]] = {0: [], 1: []}
        rows: dict[int, ObjectFrame] = {}
        for obj in frame.objects:  # kind 13 is the primary princess-tower form
            if obj.kind == 13 and obj.card_id <= 0 and TOWER_OBJECT_ID_MIN <= obj.object_id <= TOWER_OBJECT_ID_MAX:
                rows[obj.object_id] = obj
        for obj in frame.objects:  # kind 12 fills towers absent from kind 13
            if (obj.kind == 12 and obj.card_id <= 0 and obj.object_id not in rows
                    and TOWER_OBJECT_ID_MIN <= obj.object_id <= TOWER_OBJECT_ID_MAX):
                rows[obj.object_id] = obj
        for obj in rows.values():
            if obj.hp <= 0:
                continue
            if obj.x < KING_TOWER_X:
                tower_kind = "princess_left"
            elif obj.x > KING_TOWER_X:
                tower_kind = "princess_right"
            else:
                tower_kind = "king"
            owner = int(obj.owner) if obj.owner in (0, 1) else None
            if owner is None:
                self.stats["towers_skipped"] += 1
                continue
            max_hp = float(obj.max_hp)
            hp = float(min(obj.hp, obj.max_hp))
            entry = {
                "entity_id": int(obj.object_id),
                "owner": owner,
                "tower_kind": tower_kind,
                "position": _pair(obj.x, obj.y),
                "hitpoints": hp,
                "max_hitpoints": max_hp,
                "active": True,
            }
            if tower_kind != "king":
                entry["tower_troop_id"] = SIDE_TOWER_TROOP_ID
            towers.append(entry)
            by_owner[owner].append(int(obj.object_id))
        return towers, {owner: tuple(ids) for owner, ids in by_owner.items()}

    def _entities(self, frame: BattleFrame) -> list[dict]:
        entities: list[dict] = []
        for obj in frame.objects:
            if obj.kind != 15 or obj.card_id <= 0:
                continue  # kind 14 echoes attacks/projectiles; only 15 is deployed
            gid = self.card_global_id(int(obj.card_id))
            if gid is None or self.entity_archetype_catalog.runtime_global_vocab_id(gid) == 1:
                self.stats["entities_dropped"] += 1
                continue  # unresolvable objects (203000018, spell leftovers)
            if obj.hp <= 0:
                self.stats["entities_dropped"] += 1
                continue
            spec = self.card_specs[int(obj.card_id)]
            entity_kind = ENTITY_KIND_BY_CARD_KIND.get(spec.kind)
            if entity_kind is None:
                self.stats["entities_dropped"] += 1
                continue
            owner = int(obj.owner) if obj.owner in (0, 1) else None
            entry: dict = {
                "entity_id": int(obj.object_id),
                "owner": owner,
                "card_id": int(obj.card_id),
                "entity_kind": entity_kind,
                "position": _pair(obj.x, obj.y),
                "native_data_global_id": int(gid),
                "visible": True,
            }
            if obj.max_hp > 0:
                entry["hitpoints"] = float(obj.hp)
                entry["max_hitpoints"] = float(obj.max_hp)
            entities.append(entry)
        return entities

    def _action_mask(self, player: PlayerFrame, mirror_target: tuple[int, float] | None = None) -> dict:
        elixir = float(player.elixir)
        hand = sorted(player.hand, key=lambda card: card.slot)
        slots: list[bool] = []
        reasons: dict[str, str] = {}
        masks: dict[str, dict] = {}
        has_unaffordable = False
        for card in hand:
            slot = int(card.slot)
            spec = self.card_specs.get(card.card_id)
            if spec is None:
                reasons[str(slot)] = "card_spec_unavailable"
                slots.append(False)
                continue
            if int(card.card_id) == MIRROR_CARD_ID and mirror_target is None:
                reasons[str(slot)] = "mirror_target_unknown"
                slots.append(False)
                masks[str(slot)] = self._placement_entry(card, slot, spec, mirror_target)
                continue
            if int(card.card_id) == MIRROR_CARD_ID:
                cost = float(mirror_target[1]) + 1.0  # Mirror costs source cost + 1
            else:
                cost = float(spec.elixir_cost or 0.0)
            if cost > elixir + 1e-6:
                reasons[str(slot)] = "insufficient_elixir"
                has_unaffordable = True
                slots.append(False)
            else:
                reasons[str(slot)] = "legal"
                slots.append(True)
            masks[str(slot)] = self._placement_entry(card, slot, spec, mirror_target)
        return {
            "kinds": {
                "wait": True,
                "wait_until_affordable": has_unaffordable,
                "play_card": any(slots),
                "activate_ability": False,
            },
            "hand_slots": slots,
            "placement_masks": masks,
            "ability_sources": [],
            "target_entities": [],
            "reasons": {
                "decision_ready": True,
                "slots": reasons,
                "ability": {},
                "raw_elixir": elixir,
                "reserved_elixir": 0.0,
                "effective_elixir": elixir,
                "pending_action_count": 0,
            },
        }

    def _placement_entry(self, card: HandCard, slot: int, spec,
                         mirror_target: tuple[int, float] | None = None) -> dict:
        mask = self.placement_mask(int(card.card_id), self.actor_owner)
        # For Mirror the runtime contract requires the entry to carry the
        # copied card's identity and cost (source cost + 1) and card_id to
        # equal the effective id; other cards keep effective == visible.
        if int(card.card_id) == MIRROR_CARD_ID and mirror_target is not None:
            effective_id = int(mirror_target[0])
            effective_cost = float(mirror_target[1]) + 1.0
        else:
            effective_id = int(card.card_id)
            effective_cost = float(spec.elixir_cost or 0.0)
        entry = {
            "card_id": effective_id,
            "visible_card_id": int(card.card_id),
            "effective_card_id": effective_id,
            "native_effective_card_id": int(card.card_id),
            "effective_cost": effective_cost,
            "form_code": 0,
            "native_form_code": 0,
            "shape": [GRID_HEIGHT, GRID_WIDTH],
            "row_major": [list(row) for row in mask.rows],
            "placement_rule": mask.rule.value,
            "accuracy": mask.accuracy.value,
            "reasons": list(mask.reasons),
            "source_native_hand_slot": slot,
        }
        if mask.footprint_width_tiles is not None:
            entry["footprint_width_tiles"] = int(mask.footprint_width_tiles)
            entry["footprint_height_tiles"] = int(mask.footprint_height_tiles)
        if mask.model_subcell_offset is not None:
            entry["model_subcell_offset"] = [float(mask.model_subcell_offset[0]), float(mask.model_subcell_offset[1])]
        return entry
