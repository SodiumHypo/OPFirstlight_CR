<h1 align="center">OPFirstlight_CR</h1>
<p align="center"><strong>An online, real-time pipeline on top of the FirstLight CR engine</strong></p>
<p align="center">Read live match state from a running game → adapt it into the offline engine → let the V4 policy decide → execute the actions back in the live game.</p>
<p align="center"><strong>English</strong> · <a href="README.zh-CN.md">简体中文</a></p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Status-experimental-orange.svg" alt="experimental">
</p>

---

## What this is

[FirstLight CR](#credits-and-relationship-to-firstlight-cr) is an offline
Clash Royale AI environment (native battle engine, V4 recurrent policy,
IL + PPO training). This repository keeps that engine as the base and adds
**`pipeline/`** — the piece that connects the trained policy to a **live,
running game client** instead of only simulated matches:

```
┌─────────────┐   frames    ┌──────────────┐  V4 obs  ┌──────────────┐
│ state       │ ──────────▶ │ state        │ ───────▶ │ AI inference │
│ capture     │             │ adapter      │          │ (serve_policy)│
└─────────────┘             └──────────────┘          └──────┬───────┘
     ▲                                                        │ actions
     │          ┌──────────────┐  gestures  ┌───────────────▼┐
     └────────── │ live game    │ ◀───────── │ action         │
       state     │ client (adb) │            │ execution      │
                 └──────────────┘            └────────────────┘
```

- **State capture** — a small C memory reader pushed to the device streams
  battle state (entities, towers, elixir, hand) as NDJSON frames over adb.
- **State adaptation** — live frames are mapped onto FirstLight's V4
  observation contract (nested protocol versioning, deck/mirror runtime
  contracts, entity tracking, tower identification).
- **AI inference** — the unmodified FirstLight `serve_policy` process decides
  on every adapted observation; the pipeline only schedules when to ask.
- **Action execution** — card deployments are injected as adb gestures, with
  match-time self-calibration (the world→screen mapping changes per match)
  and deploy verification.

Two emulator instances are used at runtime: the **offline engine** (policy
inference, network-isolated) and the **live client** (state capture + action
execution, online). Session routing keeps the two roles separate.

## Repository layout

| Path | Contents |
| --- | --- |
| [`pipeline/`](pipeline/README.md) | **The online pipeline** (this fork's addition): capture, adaptation, orchestration, execution, calibration tools, offline self-tests |
| `native_runner/` | FirstLight battle engine, V4 training/serving code, competitive card data |
| `tools/`, `docs/` | FirstLight build tools and upstream documentation |

## Quick start

Requirements: the FirstLight offline engine set up first — see the upstream
[installation guide](docs/getting_started.md) (MuMu Player instance, offline
APK, Python 3.12 venv with the pinned dependencies).

Tool paths are machine-independent; everything resolves from environment
variables (see [`pipeline/adb_tools.py`](pipeline/adb_tools.py)):

| Variable | Meaning |
| --- | --- |
| `CR_ADB` | full path to `adb.exe` (falls back to `adb` on PATH) |
| `CR_MUMU_MANAGER` | full path to `MuMuManager.exe` (instance role resolution) |
| `CR_ADB_SERIAL` | default adb serial for CLI tools |

Verify the whole stack **without any emulator** via the offline self-test
(adapter contracts, orchestrator simulations, and a real `serve_policy`
replay of a recorded capture):

```bash
python -c "import sys; sys.path.insert(0, '.'); \
  from pipeline.tests.selftest_orchestrator import main; raise SystemExit(main())"
```

The pipeline entry point is
[`pipeline/act/orchestrator.py`](pipeline/act/orchestrator.py); module-level
details live in the [pipeline guide](pipeline/README.md).

## Status

Experimental. End-to-end live friendly matches run today; current work
focuses on execution reliability (gesture injection success rate) and
analyzing policy passivity. Expect rough edges.

## Credits and relationship to FirstLight CR

- The battle engine, V4 policy/training code, card data, and documentation in
  `native_runner/`, `tools/`, and `docs/` are the **FirstLight CR** project,
  licensed under [Apache 2.0](LICENSE) — full credit to its original authors.
- `pipeline/` is new code written for this edition.
- **No pretrained checkpoints are included** — see
  [checkpoints/README.md](checkpoints/README.md) for how to train or register
  your own; the [training guide](docs/training.md) covers IL + PPO.

## Notice

**For learning and research purposes only.** Do not use it against any
official or commercial online service. Clash Royale is a trademark of
Supercell; this project targets a private-server environment and is not
affiliated with or endorsed by Supercell.
