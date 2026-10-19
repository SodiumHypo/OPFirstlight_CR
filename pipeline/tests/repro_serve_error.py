"""Repro of the 2026-10-17 live incident against the real serve_policy.

serve_policy answers any act error with a tickless {"ok": false, "error": ...}
line and kills its episode; PolicyClient used to discard those lines as stale,
so every live act timed out while the engine sat idle at 0% CPU. This drives
the engine into its error path and asserts the error text now surfaces as
PolicyError (and that the follow-up act reports the dead episode).

Run: FirstLight venv python -I pipeline/tests/repro_serve_error.py
"""
from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "pipeline" / "tools"))

from smoke_serve_policy import load_battle_frames  # noqa: E402

from pipeline.act.orchestrator import FIRST_DECISION_TICK  # noqa: E402
from pipeline.adapt.firstlight_adapter import FirstLightAdapter  # noqa: E402
from pipeline.host.policy_client import PolicyClient, PolicyError  # noqa: E402

CAPTURE = WORKSPACE / "pipeline" / "artifacts" / "captures" / "battle_diffcheck.ndjson"


def main() -> int:
    frames = load_battle_frames(CAPTURE)
    client_owner = next(p.owner for frame in frames for p in frame.players[:2]
                        if len(set(p.deck)) == 8)
    frame = next(f for f in frames
                 if f.tick >= FIRST_DECISION_TICK and len(f.players) == 2)
    decks = {p.owner: p.deck for p in frame.players[:2]}

    adapter = FirstLightAdapter(actor_owner=client_owner)
    observation = adapter.observation(frame, episode_id="repro-err", decks=decks)
    obs_tick = int(observation["tick"])
    policy = PolicyClient()
    try:
        start = policy.request({
            "op": "start",
            "episode": adapter.episode_config(decks[0], decks[1]),
            "observation": observation,
            "initial_elixir": adapter.initial_elixir(frame),
        })
        assert start.get("ok"), start

        act = {"op": "act", "observation": observation}
        ok = policy.request(act, timeout_s=30.0, expect_tick=obs_tick)
        assert ok.get("ok") and ok.get("tick") == obs_tick, ok
        print(f"baseline act tick={obs_tick} ok (inference_ms={ok.get('inference_ms')})")

        # owner mismatch with private state left on the old-actor row -> the
        # engine's leak check fires; either way the tickless error must surface
        bad = dict(observation)
        bad["owner"] = 1 - int(observation["owner"])
        try:
            policy.request({"op": "act", "observation": bad},
                           timeout_s=30.0, expect_tick=obs_tick)
            raise AssertionError("expected an error response to surface")
        except PolicyError as error:
            assert "timed out" not in str(error), error
            print(f"error surfaced: {error}")

        # the engine killed its episode on that error: the next act must say so
        try:
            policy.request(act, timeout_s=30.0, expect_tick=obs_tick)
            raise AssertionError("expected the dead-episode error to surface")
        except PolicyError as error:
            assert "start an episode" in str(error), error
            print(f"dead-episode surfaced: {error}")

        # phase 2: the OTHER seat as actor — live matches can resolve the
        # client to owner 1, and every replay so far used owner 0
        other = 1 - client_owner
        adapter2 = FirstLightAdapter(actor_owner=other)
        obs2 = adapter2.observation(frame, episode_id="repro-err-other", decks=decks)
        start2 = policy.request({
            "op": "start",
            "episode": adapter2.episode_config(decks[0], decks[1]),
            "observation": obs2,
            "initial_elixir": adapter2.initial_elixir(frame),
        })
        assert start2.get("ok"), start2
        ok2 = policy.request({"op": "act", "observation": obs2},
                             timeout_s=30.0, expect_tick=int(obs2["tick"]))
        print(f"actor_owner={other} act ok (inference_ms={ok2.get('inference_ms')})")
    finally:
        policy.close()
    print("REPRO PASSED: engine errors now surface as PolicyError")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
