"""Stream cr_battle_stream NDJSON from a MuMu instance into a local capture file.

Resolves the instance via MuMuManager (roles: actual_game / ai_rea), discovers
the game PID, pushes the helper binary if stale, then streams frames and prints
live normalization stats. Stop with Ctrl+C.

Usage:
  python capture_battle_stream.py                    # actual_game, until Ctrl+C
  python capture_battle_stream.py --duration 180     # auto-stop after 3 min
  python capture_battle_stream.py --role actual_game --out captures/x.ndjson
"""
from __future__ import annotations

import argparse
import datetime as dt
import signal
import subprocess
import sys
import time
from pathlib import Path

from pipeline.adb_tools import adb_path

ADB = adb_path()
PKG = "nullsroyale.rel.free"
REMOTE_DIR = "/data/local/tmp/crai"
REMOTE_BIN = f"{REMOTE_DIR}/cr_battle_stream"
LOCAL_BIN = Path(__file__).resolve().parent.parent / "native" / "build" / "cr_battle_stream"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "artifacts" / "captures"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "host"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ingest"))

from instances import resolve_role  # noqa: E402
from frames import FrameNormalizer  # noqa: E402


def adb(serial: str, *args: str, timeout: int = 30) -> str:
    return subprocess.run(
        [ADB, "-s", serial, "shell", *args], check=True, capture_output=True, text=True, timeout=timeout
    ).stdout


def find_game_pid(serial: str) -> int | None:
    out = adb(serial, "ps -A -o PID,NAME")
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == PKG:
            return int(parts[0])
    return None


def ensure_binary(serial: str) -> None:
    if not LOCAL_BIN.exists():
        sys.exit(f"helper binary missing: {LOCAL_BIN} (build it first)")
    remote_size = adb(serial, f"wc -c < {REMOTE_BIN} 2>/dev/null || echo 0").strip()
    local_size = str(LOCAL_BIN.stat().st_size)
    if remote_size != local_size:
        subprocess.run([ADB, "-s", serial, "push", str(LOCAL_BIN), REMOTE_BIN], check=True, capture_output=True)
        adb(serial, f"chmod 755 {REMOTE_BIN}")
        print(f"pushed helper ({local_size} bytes)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", default="actual_game")
    ap.add_argument("--interval", type=int, default=100, help="ms between frames")
    ap.add_argument("--duration", type=float, default=None, help="seconds; default until Ctrl+C")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    inst = resolve_role(args.role)
    if not inst.android_started:
        sys.exit(f"instance {inst.name} is not running")
    serial = inst.adb_serial
    print(f"instance={inst.name} serial={serial}")
    # The adb server does not auto-discover emulator endpoints; register first.
    subprocess.run([ADB, "connect", serial], check=True, capture_output=True, text=True, timeout=15)

    pid = find_game_pid(serial)
    if pid is None:
        sys.exit(f"{PKG} is not running on {inst.name}")
    print(f"game pid={pid}")

    ensure_binary(serial)

    out_path = args.out or DEFAULT_OUT / f"battle_{dt.datetime.now():%Y%m%d_%H%M%S}.ndjson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"capture -> {out_path}")

    normalizer = FrameNormalizer()
    cmd = f'su 0 sh -c "{REMOTE_BIN} {pid} {args.interval}"'
    started = time.perf_counter()
    stats_at = started
    with subprocess.Popen(
        [ADB, "-s", serial, "shell", cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    ) as proc, out_path.open("w", encoding="utf-8") as sink:
        stop = {"now": False}

        def on_sigint(_sig, _frm):
            stop["now"] = True

        signal.signal(signal.SIGINT, on_sigint)
        for line in proc.stdout:  # type: ignore[union-attr]
            sink.write(line)
            frame = normalizer.feed(line)
            now = time.perf_counter()
            if frame is not None and now - stats_at >= 2.0:
                stats_at = now
                s = normalizer.stats
                print(
                    f"[{now - started:6.1f}s] tick={frame.tick:5d} objects={len(frame.objects):3d} "
                    f"elixir={frame.players[0].elixir:.2f} emitted={s.emitted} "
                    f"dups={s.dropped_duplicate_tick} gaps={len(s.gaps)}"
                )
            if stop["now"] or (args.duration is not None and now - started >= args.duration):
                proc.terminate()
                break
    s = normalizer.stats
    print(
        f"done: seen={s.seen} emitted={s.emitted} inactive={s.dropped_inactive} "
        f"malformed={s.dropped_malformed} dups={s.dropped_duplicate_tick} gaps={len(s.gaps)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
