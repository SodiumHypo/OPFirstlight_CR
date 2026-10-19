# Models

**This repository does not ship pretrained checkpoints.**

Trained models from the original project are intentionally excluded here. To obtain models:

- Train your own — see the [training guide](../docs/training.md) (imitation learning on the [IL_Replay](https://huggingface.co/datasets/VanguardX101/IL_Replay) dataset, then PPO self-play).
- After training, checkpoints are written under `runs/` or your configured output directory. Copy them into this folder and list them in `manifest.json` (schema `firstlight-model-release.v1`: each entry needs `path`, `sha256`, `bytes`).

# 模型

**本仓库不附带任何预训练模型。**

训练方法见[训练文档](../docs/training.md)。训练产出的 checkpoint 放入本目录后，在 `manifest.json` 中登记（`path`、`sha256`、`bytes`）即可被评测与策略服务加载。
