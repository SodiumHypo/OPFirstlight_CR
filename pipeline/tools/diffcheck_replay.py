"""Differential verification: replay a full capture through the adapter and
serve_policy on the engine's decision grid (tick % 5 == 0), recording every
decision point plus match-state deltas for fidelity review.

Run: FirstLight venv python -I tools/diffcheck_replay.py <capture.ndjson>
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT.parent))      # workspace root: pipeline.*
sys.path.insert(0, str(PIPELINE_ROOT / "tools"))   # smoke_serve_policy

from smoke_serve_policy import FIRST_DECISION_TICK, ServePolicy, load_battle_frames  # noqa: E402

from pipeline.adapt.firstlight_adapter import FirstLightAdapter  # noqa: E402

TOWER_FIELDS = ("entity_id", "owner", "tower_kind")


def main() -> int:
    capture = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PIPELINE_ROOT / "artifacts" / "captures" / "battle_diffcheck.ndjson"
    )
    frames = load_battle_frames(capture)
    actor_owner = 0

    decks = {owner: frames[0].players[owner].deck for owner in (0, 1)}
    for owner, deck in decks.items():
        if len(deck) != 8 or len(set(deck)) != 8:
            raise ValueError(f"seat {owner} deck is not 8 unique cards: {deck}")
    print(f"deck0: {sorted(decks[0])}")
    print(f"deck1: {sorted(decks[1])}")

    adapter = FirstLightAdapter(actor_owner=actor_owner)
    base_episode_id = f"pipeline-diffcheck-{capture.stem}"
    report_path = PIPELINE_ROOT / "artifacts" / "reports" / f"diffcheck_{capture.stem}.jsonl"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    service = ServePolicy()
    decisions = errors = restarts = 0
    latencies: list[float] = []
    last_towers: dict[int, dict] = {}
    tower_deaths: list[dict] = []
    entity_seen: set[int] = set()
    spawns: list[dict] = []

    def send_start(frame, episode_id: str) -> dict:
        return service.request({
            "op": "start",
            "episode": adapter.episode_config(decks[0], decks[1]),
            "observation": adapter.observation(frame, episode_id=episode_id, decks=decks),
            "initial_elixir": adapter.initial_elixir(frame),
        })

    episode_alive = False
    episode_id = base_episode_id
    with report_path.open("w", encoding="utf-8") as report:
        def emit(record: dict) -> None:
            report.write(json.dumps(record, ensure_ascii=False) + "\n")

        for frame in frames:
            observation = adapter.observation(frame, episode_id=episode_id, decks=decks)

            # match-state deltas: tower destructions and fresh deployments
            current_ids = {tower["entity_id"] for tower in observation["towers"]}
            for tower_id, prior in list(last_towers.items()):
                if tower_id not in current_ids:
                    tower_deaths.append({
                        "tick": frame.tick,
                        **{key: prior[key] for key in TOWER_FIELDS},
                        "last_hp": prior["hitpoints"],
                    })
                    del last_towers[tower_id]
            for tower in observation["towers"]:
                last_towers[tower["entity_id"]] = tower
            for entity in observation["entities"]:
                if entity["entity_id"] not in entity_seen:
                    entity_seen.add(entity["entity_id"])
                    spawns.append({
                        "tick": frame.tick,
                        "card_id": entity["card_id"],
                        "owner": entity["owner"],
                        "position": entity["position"],
                    })

            if frame.tick < FIRST_DECISION_TICK or frame.tick % 5 != 0:
                continue

            if not episode_alive:
                restarts += 1
                episode_id = base_episode_id if restarts == 1 else f"{base_episode_id}-r{restarts}"
                response = send_start(frame, episode_id)
                print(f"start#{restarts} tick={frame.tick}: ok={response.get('ok')}")
                if not response.get("ok"):
                    raise RuntimeError(f"restart failed: {response}")
                episode_alive = True
                continue

            response = service.request({"op": "act", "observation": observation})
            if not response.get("ok"):
                errors += 1
                emit({"tick": frame.tick, "error": response})
                print(f"act ERROR tick={frame.tick}: {response}")
                episode_alive = False
                continue

            decisions += 1
            latency = float(response.get("inference_ms", 0.0))
            if latency:
                latencies.append(latency)
            actor = frame.players[actor_owner]
            emit({
                "tick": frame.tick,
                "elixir": float(actor.elixir),
                "hand": [card.card_id for card in sorted(actor.hand, key=lambda c: c.slot)],
                "towers": {
                    str(owner): [round(tower["hitpoints"]) for tower in sorted(
                        observation["towers"], key=lambda t: t["entity_id"]) if tower["owner"] == owner]
                    for owner in (0, 1)
                },
                "entities": len(observation["entities"]),
                "inference_ms": latency,
                "actions": response.get("actions", []),
            })

    service.close()
    print("\n=== diffcheck summary ===")
    print(f"frames={len(frames)} decisions={decisions} errors={errors} restarts={restarts}")
    print(f"adapter drops: {adapter.stats}")
    if latencies:
        latencies.sort()
        p50 = statistics.median(latencies)
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        print(f"inference ms: p50={p50:.0f} p95={p95:.0f} max={latencies[-1]:.0f}")
    print(f"tower deaths ({len(tower_deaths)}):")
    for death in tower_deaths:
        print(f"  tick={death['tick']} owner={death['owner']} {death['tower_kind']} (last_hp={death['last_hp']:.0f})")
    spawn_counts: dict[str, int] = {}
    for spawn in spawns:
        key = f"owner{spawn['owner']}" if spawn["owner"] in (0, 1) else "owner?"
        spawn_counts[key] = spawn_counts.get(key, 0) + 1
    print(f"deployed entities: {spawn_counts}")
    print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
