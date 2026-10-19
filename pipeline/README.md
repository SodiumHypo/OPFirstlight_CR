# Online Pipeline

The online streaming pipeline built on top of the FirstLight CR engine:
it reads live match state from a running game client, adapts it back into
FirstLight's offline engine, lets the V4 policy decide, and executes the
decided actions back in the live game — in real time.

在线实时管线：从运行中的游戏客户端读取实时对局状态，适配回 FirstLight 离线引擎，
由 V4 策略模型决策，再把动作实时执行回真实对局。

## Architecture

```
state capture ──▶ state adapter ──▶ AI inference ──▶ action execution
 (adb + native     (live memory       (V4 policy       (adb gesture
  memory reader)    frames -> V4       served by        injection into
                    observations)      serve_policy)    the live client)
```

| Module | Role |
| --- | --- |
| `native/` | C memory-reader (`cr_battle_stream`) pushed to the device; streams battle state over stdout |
| `ingest/` | frame stream plumbing, supervision, reconnection |
| `adapt/` | live frames → FirstLight V4 observation contract (nested versioning, deck/mirror runtime contracts, entity/tower mapping) |
| `host/` | `policy_client` (NDJSON subprocess client for `serve_policy`), match-session GUI, multi-instance role resolution |
| `act/` | match orchestrator: start-of-match self-calibration, decision scheduling, action execution, deploy verification |
| `tools/` | calibration (`calibrate_grid.py`), diagnostics (`tap_latency.py`, `report_passivity.py`), vocabulary probes |
| `tests/` | offline self-test suite: adapter contracts, orchestrator sims (direct/mirrored/fresh-fit mapping), timeout resilience, real-engine replay |

Key design points:

- **Dual-instance topology**: AI inference runs against a network-isolated
  offline engine instance; state capture and action execution target the live
  client instance. Roles are resolved by instance name via MuMuManager.
- **Start-of-match self-calibration**: the world→screen mapping changes per
  match (mirror + translation); one constrained troop probe resolves it before
  the first real play.
- **Machine-independence**: tool paths come from `CR_ADB`, `CR_MUMU_MANAGER`,
  `CR_ADB_SERIAL` env vars (see `pipeline/adb_tools.py`); calibration artifacts
  are device-specific and generated per device by `tools/calibrate_grid.py`.
- **Offline self-test** (`pipeline/tests/selftest_orchestrator.py`): simulated
  matches plus a real `serve_policy` replay — no emulator required for CI-style
  verification.

## Status

Experimental. The pipeline runs end to end in live friendly matches; current
work focuses on execution reliability and policy passivity analysis.
