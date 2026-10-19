<h1 align="center">OPFirstlight_CR</h1>
<p align="center"><strong>构建在 FirstLight CR 引擎之上的在线实时管线</strong></p>
<p align="center">从运行中的游戏读取实时对局状态 → 适配进离线引擎 → V4 策略模型决策 → 把动作实时执行回真实对局。</p>
<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong></p>
<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Python-3.12-3776AB.svg" alt="Python 3.12">
  <img src="https://img.shields.io/badge/Status-experimental-orange.svg" alt="experimental">
</p>

---

## 这是什么

[FirstLight CR](#致谢与和-firstlight-cr-的关系) 是一个离线皇室战争 AI 环境
（原生对战引擎、V4 循环策略、IL + PPO 训练）。本仓库以该引擎为基座，新增
**`pipeline/`**——把训练好的策略接到**运行中的真实游戏客户端**上，而不只
限于模拟对局：

```
┌─────────────┐   帧流     ┌──────────────┐  V4 观测  ┌───────────────┐
│ 状态采集     │ ─────────▶ │ 状态适配      │ ────────▶ │ AI 推理        │
│ (内存读取)   │            │              │           │ (serve_policy) │
└─────────────┘            └──────────────┘           └───────┬───────┘
     ▲                                                        │ 动作
     │           ┌──────────────┐   手势注入  ┌────────────────▼┐
     └────────── │ 真实游戏客户端 │ ◀───────── │ 动作执行         │
       状态       │   (adb)      │            │                 │
                 └──────────────┘            └─────────────────┘
```

- **状态采集**——推送到设备上的小型 C 内存读取器，把对局状态（实体、塔、
  圣水、手牌）以 NDJSON 帧流经 adb 输出。
- **状态适配**——把实时帧映射到 FirstLight 的 V4 观测合约（嵌套协议版本、
  卡组/Mirror 运行时合约、实体跟踪、塔识别）。
- **AI 推理**——未经修改的 FirstLight `serve_policy` 进程对每个适配后的
  观测做决策；管线只负责调度"何时去问"。
- **动作执行**——出牌以 adb 手势注入，配合开局自校准（世界→屏幕映射每局
  都会变）和部署验证。

运行时使用两个模拟器实例：**离线引擎**（策略推理，网络隔离）与**真实客户端**
（状态采集 + 动作执行，联网）。会话路由保证两个角色严格分离。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| [`pipeline/`](pipeline/README.md) | **在线管线**（本版本的增量）：采集、适配、编排、执行、标定工具、离线自测 |
| `native_runner/` | FirstLight 对战引擎、V4 训练/服务代码、卡牌数据 |
| `tools/`、`docs/` | FirstLight 构建工具与上游文档 |

## 快速开始

前置条件：先完成 FirstLight 离线引擎部署——见上游[安装指南](docs/getting_started.md)
（MuMu 模拟器实例、离线 APK、Python 3.12 venv 与锁定版本依赖）。

工具路径与机器无关，全部从环境变量解析（见
[`pipeline/adb_tools.py`](pipeline/adb_tools.py)）：

| 变量 | 含义 |
| --- | --- |
| `CR_ADB` | `adb.exe` 完整路径（回退到 PATH 中的 `adb`） |
| `CR_MUMU_MANAGER` | `MuMuManager.exe` 完整路径（实例角色解析） |
| `CR_ADB_SERIAL` | CLI 工具的默认 adb 串口 |

无需模拟器即可验证全栈（适配器合约、编排器仿真、真实 `serve_policy`
回放已录制的对局）：

```bash
python -c "import sys; sys.path.insert(0, '.'); \
  from pipeline.tests.selftest_orchestrator import main; raise SystemExit(main())"
```

管线入口是
[`pipeline/act/orchestrator.py`](pipeline/act/orchestrator.py)；模块级细节见
[管线文档](pipeline/README.md)。

## 状态

实验性质。目前可以在真实友谊赛中端到端运行；当前工作重点是执行可靠性
（手势注入成功率）与策略被动性分析。请预期有毛边。

## 致谢与和 FirstLight CR 的关系

- `native_runner/`、`tools/`、`docs/` 中的对战引擎、V4 策略/训练代码、
  卡牌数据与文档属于 **FirstLight CR** 项目，采用 [Apache 2.0](LICENSE)
  许可证——版权归原作者所有。
- `pipeline/` 是本版本新写的代码。
- **本仓库不附带预训练模型**——自行训练或注册模型见
  [模型说明](checkpoints/README.md)，IL + PPO 训练见[训练文档](docs/training.md)。

## 声明

**仅供学习与研究使用**，请勿用于任何官方或商业在线服务。Clash Royale 是
Supercell 的商标；本项目面向私服研究环境，与 Supercell 无关，也未获得其认可。
