"""Screen-level input executor for the live client via one persistent adb shell.

The executor keeps a single long-lived `adb shell` process per device (measured
p50 ~9ms vs ~100ms+ for per-action spawns, see tools/tap_latency.py) and sends
synchronous tap commands with echo markers. World->screen placement and
card-slot geometry come from a calibration file produced by
tools/calibrate_grid.py; without it only raw screen taps are available.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pipeline.adb_tools import adb_path

ADB_DEFAULT = adb_path()
CALIBRATION_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "calibration"


def calibration_path(serial: str) -> Path:
    return CALIBRATION_DIR / f"calibration_{serial.replace(':', '_')}.json"


class Calibration:
    """World<->screen mapping plus card-slot geometry for one device."""

    def __init__(self, data: dict) -> None:
        self.data = data

    @classmethod
    def load(cls, serial: str) -> "Calibration | None":
        path = calibration_path(serial)
        if not path.is_file():
            return None
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        m = self.data["world_to_screen"]  # 3x3 row-major, homogeneous
        w = m[2][0] * x + m[2][1] * y + m[2][2]
        sx = (m[0][0] * x + m[0][1] * y + m[0][2]) / w
        sy = (m[1][0] * x + m[1][1] * y + m[1][2]) / w
        return sx, sy

    def card_slot_px(self, slot: int) -> tuple[float, float]:
        entry = self.data["card_slots_px"][str(slot)]
        return float(entry[0]), float(entry[1])

    @property
    def screen(self) -> dict:
        return self.data["screen"]


class ActionExecutor:
    """Synchronous tap commands over one persistent adb shell process."""

    def __init__(self, serial: str, adb: str = ADB_DEFAULT) -> None:
        self.serial = serial
        self._adb = adb
        # The adb server does not auto-discover emulator endpoints; register first.
        subprocess.run([adb, "connect", serial], check=True, capture_output=True, timeout=15)
        self.proc = subprocess.Popen(
            [adb, "-s", serial, "shell"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._seq = 0
        self._command("true")  # warmup: first command carries extra connect latency

    def _command(self, cmd: str) -> int:
        """Run one shell command, wait for its completion marker, return $?.

        The adb shell pipe can die mid-session (adb server restart, emulator
        hiccup); one transparent reconnect-and-retry keeps callers simple.
        """
        try:
            return self._command_once(cmd)
        except (OSError, RuntimeError):
            self._reconnect()
            return self._command_once(cmd)

    def _command_once(self, cmd: str) -> int:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        self._seq += 1
        marker = f"__D{self._seq}"
        self.proc.stdin.write(f"{cmd}; echo {marker}=$?\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("persistent shell closed unexpectedly")
            if marker in line:
                try:
                    return int(line.rsplit("=", 1)[1].strip())
                except ValueError:
                    return -1

    def _reconnect(self) -> None:
        try:
            self.close()
        except OSError:
            pass
        self.proc = subprocess.Popen(
            [self._adb, "-s", self.serial, "shell"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._seq = 0
        self._command_once("true")  # warmup

    # Zero-duration `input tap` is not recognized as a click by the battle
    # scene (verified live: 20/20 deploys no-op while UI overlays still react);
    # a same-coordinate swipe with a hold duration reads as a real press.
    TAP_HOLD_MS = 80

    def tap_screen(self, x: float, y: float) -> None:
        xi, yi = int(round(x)), int(round(y))
        status = self._command(f"input swipe {xi} {yi} {xi} {yi} {self.TAP_HOLD_MS}")
        if status != 0:  # 255 seen during adb hiccups — one retry after reconnect
            status = self._command(f"input swipe {xi} {yi} {xi} {yi} {self.TAP_HOLD_MS}")
        if status != 0:
            raise RuntimeError(f"input swipe tap failed with status {status}")

    def tap_world(self, x: float, y: float, cal: Calibration) -> None:
        sx, sy = cal.world_to_screen(x, y)
        self.tap_screen(sx, sy)

    # Drag-deploy: one gesture from the hand slot to the arena point — how a
    # human plays. One command (pipe ~9ms + input-tool start ~40ms + drag) vs
    # the old tap-slot / card-lift sleep / tap-world chain (~300ms): the kiting
    # plays need sub-200ms execution. 80ms matches the proven TAP_HOLD_MS press
    # duration; release point is the deploy point regardless of drag speed.
    # Deploy verification (deploys_failed) is the regression tripwire; the
    # two-tap path is tap_card_slot+tap_world.
    DRAG_MS = 80

    def play_card(self, slot: int, world_xy: tuple[float, float] | None, cal: Calibration) -> None:
        """Drag the hand card onto the arena point (one gesture)."""
        sx, sy = cal.card_slot_px(slot)
        if world_xy is None:
            self.tap_screen(sx, sy)
            return
        wx, wy = cal.world_to_screen(*world_xy)
        cmd = (f"input swipe {int(round(sx))} {int(round(sy))} "
               f"{int(round(wx))} {int(round(wy))} {self.DRAG_MS}")
        status = self._command(cmd)
        if status != 0:  # 255 seen during adb hiccups — one retry after reconnect
            status = self._command(cmd)
        if status != 0:
            raise RuntimeError(f"input swipe drag failed with status {status}")

    def tap_card_slot(self, slot: int, cal: Calibration) -> None:
        x, y = cal.card_slot_px(slot)
        self.tap_screen(x, y)

    def close(self) -> None:
        try:
            assert self.proc.stdin is not None
            self.proc.stdin.write("exit\n")
            self.proc.stdin.flush()
        except (OSError, AssertionError):
            pass
        self.proc.kill()
