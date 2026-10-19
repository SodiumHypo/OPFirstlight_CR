"""Measure input-tap round-trip latency over adb (host -> device input injection).

Compares the two channels the action executor can use:
  - spawn:  one `adb shell input tap` process per action (simple, ~2 process spawns)
  - persist: one long-lived `adb shell` process; commands written to its stdin

Reports min/p50/p95/max/mean in milliseconds for each mode.
Usage: python tap_latency.py [--serial 127.0.0.1:5555] [--x 5 --y 960] [-n 10]
Coordinate defaults to the left screen edge (inert spot in most UIs).
"""
import argparse
import statistics
import subprocess
import time

from pipeline.adb_tools import adb_path, adb_serial_default

ADB_DEFAULT = adb_path()


def mode_spawn(adb, serial, x, y, n, warmup=2):
    for _ in range(warmup):
        subprocess.run([adb, "-s", serial, "shell", "true"], check=True, capture_output=True)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        subprocess.run([adb, "-s", serial, "shell", "input", "tap", str(x), str(y)], check=True, capture_output=True)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def mode_persistent(adb, serial, x, y, n, warmup=2):
    proc = subprocess.Popen(
        [adb, "-s", serial, "shell"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    try:
        for i in range(warmup):
            proc.stdin.write(f"true; echo __W{i}$?\n")
            proc.stdin.flush()
            while f"__W{i}" not in (proc.stdout.readline() or ""):
                pass
        times = []
        for i in range(n):
            t0 = time.perf_counter()
            proc.stdin.write(f"input tap {x} {y}; echo __D{i}$?\n")
            proc.stdin.flush()
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError("persistent shell closed unexpectedly")
                if f"__D{i}" in line:
                    break
            times.append((time.perf_counter() - t0) * 1000.0)
    finally:
        try:
            proc.stdin.write("exit\n")
            proc.stdin.flush()
        except OSError:
            pass
        proc.kill()
    return times


def report(name, ts):
    ts = sorted(ts)
    p95 = ts[max(0, int(len(ts) * 0.95) - 1)]
    print(
        f"{name}: n={len(ts)} min={ts[0]:.1f}ms p50={statistics.median(ts):.1f}ms "
        f"p95={p95:.1f}ms max={ts[-1]:.1f}ms mean={statistics.mean(ts):.1f}ms"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adb", default=ADB_DEFAULT)
    ap.add_argument("--serial", default=adb_serial_default())
    ap.add_argument("--x", type=int, default=5)
    ap.add_argument("--y", type=int, default=960)
    ap.add_argument("-n", type=int, default=10)
    args = ap.parse_args()
    print(f"target={args.serial} tap=({args.x},{args.y}) n={args.n}")
    report("spawn    ", mode_spawn(args.adb, args.serial, args.x, args.y, args.n))
    report("persist  ", mode_persistent(args.adb, args.serial, args.x, args.y, args.n))


if __name__ == "__main__":
    main()
