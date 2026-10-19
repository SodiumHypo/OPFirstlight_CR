"""End-to-end P1 smoke: capture file -> adapter -> serve_policy decisions.

Starts native_runner.training.v4.serve_policy (General checkpoint, CPU),
replays one recorded battle through pipeline.adapt.firstlight_adapter, and
verifies the policy returns legal actions on real frames.

Run: FirstLight venv python -I tools/smoke_serve_policy.py [capture.ndjson]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT.parent))  # workspace root so "pipeline.*" imports resolve

from pipeline.adapt.firstlight_adapter import FIRSTLIGHT_ROOT, FirstLightAdapter  # noqa: E402
from pipeline.ingest.frames import BattleFrame, FrameNormalizer  # noqa: E402

CHECKPOINT = FIRSTLIGHT_ROOT / "checkpoints" / "General" / "checkpoint-step-00000460.pt"
FIRST_DECISION_TICK = 90
MAX_DECISIONS = 6


class ServePolicy:
    """NDJSON client for one serve_policy subprocess."""

    def __init__(self) -> None:
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f"checkpoint missing: {CHECKPOINT}")
        self.proc = subprocess.Popen(
            [
                str(FIRSTLIGHT_ROOT / ".venv" / "Scripts" / "python.exe"),
                "-m",
                "native_runner.training.v4.serve_policy",
                "--checkpoint",
                str(CHECKPOINT),
                "--device",
                "cpu",
            ],
            cwd=FIRSTLIGHT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            encoding="utf-8",
        )

    def request(self, payload: dict) -> dict:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("serve_policy closed the stream (check stderr)")
        return json.loads(line)

    def close(self) -> None:
        try:
            self.request({"op": "end"})
        except Exception:  # noqa: BLE001
            pass
        self.proc.terminate()


def load_battle_frames(capture: Path) -> list[BattleFrame]:
    """Read the capture and keep the longest strictly-advancing battle run."""

    normalizer = FrameNormalizer()
    frames: list[BattleFrame] = []
    with capture.open("r", encoding="utf-8") as fh:
        for line in fh:
            frame = normalizer.feed(line)
            if frame is not None:
                frames.append(frame)
    best_start = best_end = best_gain = 0
    run_start = 0
    for index in range(1, len(frames) + 1):
        last = frames[index - 1].tick
        if index == len(frames) or frames[index].tick <= last:
            gain = last - frames[run_start].tick
            if gain > best_gain:
                best_gain, best_start, best_end = gain, run_start, index
            run_start = index
    window = frames[best_start:best_end]
    print(f"frames: {len(frames)} emitted, battle window [{best_start}:{best_end}] "
          f"ticks {window[0].tick}..{window[-1].tick}")
    return window


def main() -> int:
    capture = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        PIPELINE_ROOT / "artifacts" / "captures" / "battle_20261016_190553.ndjson"
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
    episode_id = f"pipeline-smoke-{capture.stem}"
    service = ServePolicy()
    decisions = 0
    try:
        response = service.request({
            "op": "start",
            "episode": adapter.episode_config(decks[0], decks[1]),
            "observation": adapter.observation(frames[0], episode_id=episode_id, decks=decks),
            "initial_elixir": adapter.initial_elixir(frames[0]),
        })
        print(f"start: ok={response.get('ok')} checkpoint={response.get('checkpoint_id', '')[:16]} "
              f"step={response.get('update_step')}")
        if not response.get("ok"):
            raise RuntimeError(f"start failed: {response}")

        for frame in frames[1:]:
            observation = adapter.observation(frame, episode_id=episode_id, decks=decks)
            if frame.tick < FIRST_DECISION_TICK:
                continue
            if decisions == 0 or (decisions < MAX_DECISIONS and frame.tick % 250 < 5):
                response = service.request({"op": "act", "observation": observation})
                if not response.get("ok"):
                    raise RuntimeError(f"act failed at tick {frame.tick}: {response}")
                actions = response.get("actions", [])
                print(f"act tick={frame.tick}: {len(actions)} action(s) "
                      f"inference={response.get('inference_ms', 0):.0f}ms")
                for action in actions:
                    print(f"    {json.dumps(action, sort_keys=True)}")
                decisions += 1
            if decisions >= MAX_DECISIONS:
                break
        print(f"SMOKE OK: {decisions} decisions from {len(frames)} battle frames")
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
