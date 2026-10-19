"""Resolve MuMu emulator instances by role via MuMuManager (never hardcode ports).

Roles in the dual-instance topology:
  ai_rea      — FirstLight offline engine / AI inference host (network-isolated)
  actual_game — live Null's Royale client (online, never isolated)

Instance NAMES are the stable identifiers (ASCII, user-controlled); index and
adb port are read dynamically at runtime.

CLI:
  python instances.py                 # print all instances
  python instances.py --role actual_game   # print adb serial for a role
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass

from pipeline.adb_tools import mumu_manager_path

ROLE_TO_INSTANCE = {
    "ai_rea": "ai_rea",
    "actual_game": "actual_game",
}


@dataclass(frozen=True)
class Instance:
    name: str
    index: int
    adb_serial: str
    android_started: bool

    @property
    def role(self) -> str:
        for role, name in ROLE_TO_INSTANCE.items():
            if name == self.name:
                return role
        return "unknown"


def list_instances(manager: str | None = None) -> dict[str, Instance]:
    manager = manager or mumu_manager_path()
    raw = subprocess.run(
        [manager, "info", "-v", "all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    ).stdout
    data = json.loads(raw)
    out: dict[str, Instance] = {}
    for entry in data.values():
        # Stopped instances omit adb_host_ip/adb_port entirely; serial "" = down.
        host, port = entry.get("adb_host_ip"), entry.get("adb_port")
        out[str(entry["name"])] = Instance(
            name=str(entry["name"]),
            index=int(entry["index"]),
            adb_serial=f"{host}:{port}" if host and port else "",
            android_started=bool(entry["is_android_started"]),
        )
    return out


def resolve_role(role: str, manager: str | None = None) -> Instance:
    manager = manager or mumu_manager_path()
    instances = list_instances(manager)
    want = ROLE_TO_INSTANCE.get(role)
    if want is None:
        raise KeyError(f"unknown role {role!r}; known: {sorted(ROLE_TO_INSTANCE)}")
    if want not in instances:
        raise KeyError(f"instance {want!r} not found; running: {sorted(instances)}")
    return instances[want]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--role", help="print serial for this role instead of listing all")
    args = ap.parse_args()
    if args.role:
        inst = resolve_role(args.role)
        print(inst.adb_serial)
        return 0
    for inst in list_instances().values():
        state = "up" if inst.android_started else "down"
        print(f"{inst.index}  {inst.name:<14} {inst.role:<12} {inst.adb_serial}  {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
