"""NDJSON client for the FirstLight serve_policy subprocess.

Protocol (native_runner.training.v4.serve_policy): one JSON request per stdin
line, one JSON response per stdout line. Inference runs on the host CPU over
the General checkpoint (p50=118ms p95=139ms, D6 budget 300ms).

The blocking readline moves to a reader thread so a stalled engine can't
freeze the match loop; act responses are correlated by their echoed "tick"
(serve_policy returns it verbatim), so a response that outlives its timeout
is dropped as stale instead of poisoning the next decision.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time

from pipeline.adapt.firstlight_adapter import FIRSTLIGHT_ROOT

CHECKPOINT = FIRSTLIGHT_ROOT / "checkpoints" / "General" / "checkpoint-step-00000460.pt"


class PolicyError(RuntimeError):
    """serve_policy unreachable, timed out, or closed the stream."""


class PolicyClient:
    """One serve_policy subprocess; process owns one episode at a time.

    serve_policy caps itself to one torch thread (torch.set_num_threads(1)
    in its main), so replay latency (p50=117ms) holds even while the live
    emulator burns CPU; the env caps below are only a belt-and-suspenders
    guard for library-level thread pools.
    """

    def __init__(self, checkpoint: object = CHECKPOINT, device: str = "cpu",
                 default_timeout_s: float = 5.0, threads: int = 8) -> None:
        self.default_timeout_s = default_timeout_s
        env = {**os.environ,
               "OMP_NUM_THREADS": str(threads),
               "MKL_NUM_THREADS": str(threads),
               "OPENBLAS_NUM_THREADS": str(threads),
               "NUMEXPR_NUM_THREADS": str(threads)}
        self.proc = subprocess.Popen(
            [
                str(FIRSTLIGHT_ROOT / ".venv" / "Scripts" / "python.exe"),
                "-m", "native_runner.training.v4.serve_policy",
                "--checkpoint", str(checkpoint),
                "--device", device,
            ],
            cwd=FIRSTLIGHT_ROOT, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8",
        )
        self._responses: queue.Queue = queue.Queue()
        self._closed = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            try:
                self._responses.put(json.loads(line))
            except json.JSONDecodeError:
                print(f"[policy_client] non-JSON engine line dropped: {line[:120]!r}",
                      file=sys.stderr)
                continue  # engine chatter on the stream — drop, never fatal
        self._responses.put(None)  # EOF sentinel

    def request(self, payload: dict, *, timeout_s: float | None = None,
                expect_tick: int | None = None) -> dict:
        """Send one request and wait for its response.

        expect_tick correlates act responses (serve_policy echoes the
        observation tick): responses for older ticks are stale leftovers from
        an earlier timeout and are dropped.

        An {"ok": false, "error": ...} response IS the answer to this request
        (serve_policy returns it without a tick and kills its episode), so it
        surfaces immediately as PolicyError instead of being silently
        discarded — live 2026-10-17: tickless error responses were dropped as
        stale, every act then timed out while the engine sat idle, and the
        episode death went unnoticed.
        """
        if self._closed:
            raise PolicyError("policy client already closed")
        assert self.proc.stdin is not None
        timeout_s = self.default_timeout_s if timeout_s is None else timeout_s
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        deadline = time.perf_counter() + timeout_s
        dropped = 0
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise PolicyError(f"serve_policy timed out after {timeout_s:.1f}s "
                                  f"({dropped} stale responses discarded)")
            try:
                response = self._responses.get(timeout=remaining)
            except queue.Empty:
                raise PolicyError(f"serve_policy timed out after {timeout_s:.1f}s "
                                  f"({dropped} stale responses discarded)") from None
            if response is None:
                raise PolicyError("serve_policy closed the stream (check its stderr)")
            if response.get("ok") is False or "error" in response:
                raise PolicyError(f"serve_policy error: {response.get('error') or response}")
            if expect_tick is None or response.get("tick") == expect_tick:
                return response
            dropped += 1
            print(f"[policy_client] dropped stale response (expect tick {expect_tick}): "
                  f"{json.dumps(response)[:200]}", file=sys.stderr)
            # stale response for an already-abandoned tick — drop and keep waiting

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write(json.dumps({"op": "end"}) + "\n")
            self.proc.stdin.flush()
        except (OSError, AssertionError):
            pass
        self.proc.terminate()
