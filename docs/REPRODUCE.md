# 复现流程

本仓库保存了 2026-11-18 的实际实验记录。`scripts/` 是整理后的运行入口，保持原数值代码，只将固定工作目录改为配置，并增加独立的执行器；没有用该执行器重跑完整 GPU 实验。

## 1. 运行条件与目录

目标是 Linux 单 GPU；原始实测硬件为 RTX 3090 24GB。Python 训练环境为 3.11.18，仿真环境为 3.10.14。训练显存采样峰值约 22 GB，包含 JAX 90% 预分配。下载的基模、原始专家数据、完整训练状态和缓存需要另留数十 GB 磁盘空间。

进入仓库根目录，设置一个专用于本项目的工作目录：

```bash
export PI05_WORKDIR="$PWD/runtime"
```

未设置时也默认使用仓库的 `runtime/`。`.env.example` 只是示例，不会自动载入。不要指向已有实验记录目录；脚本会拒绝覆盖已有训练和评测记录。

## 2. 环境与上游源码

先准备 Git、[uv](https://docs.astral.sh/uv/)、可用的 NVIDIA 驱动及 EGL/OpenGL 运行库，然后运行：

```bash
bash scripts/bootstrap.sh
```

该脚本只拉取固定源码版本、安装两个 Python 环境和创建空工作目录，不启动模型训练。它不安装系统驱动或创建云机器。

| 环境 | 关键版本 |
|---|---|
| 训练/策略服务 | Python 3.11.18、JAX/JAXlib 0.5.3、Flax 0.10.2、Orbax 0.11.13 |
| 仿真 | Python 3.10.14、NumPy 1.26.4、MuJoCo 3.2.3、robosuite 1.4.1 |

`environment/*-freeze.original.txt` 是实际机器的完整快照，包含当时的 editable 路径。安装脚本使用移除这些路径后的 server 版本列表与仿真直接依赖；可编辑项目另按固定 commit 安装。此快照不是跨操作系统、CUDA 镜像的通用求解锁。新环境安装尚未在干净镜像上全量验证。

已有合适 Python 时，可通过 `PI05_SERVER_PYTHON`、`PI05_SIM_PYTHON` 指定解释器。源目录已存在但版本不符时，脚本会停止。

## 3. 基础权重

```bash
"$PI05_WORKDIR/server-env311/bin/python" scripts/download_base.py
```

按 `configs/base-checkpoint-manifest.json` 的 GCS generation 下载原始 `pi05_base`，所有文件最后进行 CRC32C 核验。分块下载可恢复；同大小缓存也会接受最后的完整校验。需要联网，不会下载官方 `pi05_libero` 微调权重。首次构建分词器时，上游 OpenPI 可能还会获取其缓存资源。

## 4. 专家数据与归一化

```bash
"$PI05_WORKDIR/server-env311/bin/python" scripts/pipeline.py prepare
```

仅下载 `configs/dataset-manifest.json` 中已经选定的固定 revision、50 条轨迹及元数据；不重新选样，不跟随数据集 latest。校验 SHA256、帧号、状态/动作维度、有限数值和轨迹划分，再只用 40 条训练轨迹计算归一化。

原始数据保存在 `$PI05_WORKDIR/experiments/pi05-object-lora-20261118/raw/`。使用自带逐轨迹 Parquet loader，复用官方输入输出变换，不是直接运行默认 LeRobot 全库 loader。

## 5. 第 0–500 步

```bash
"$PI05_WORKDIR/server-env311/bin/python" scripts/pipeline.py train500
"$PI05_WORKDIR/server-env311/bin/python" scripts/pipeline.py eval0
"$PI05_WORKDIR/server-env311/bin/python" scripts/pipeline.py eval500
"$PI05_WORKDIR/server-env311/bin/python" scripts/pipeline.py verify500
```

`train500` 真实训练 500 步，保存第 0 与第 500 步完整状态、指标和独立 LoRA。`eval0`、`eval500` 各启动已预热的策略服务，再运行 10 个任务，结束后关闭服务。`verify500` 在 CPU 上核对基模、LoRA 导出文件、参数树与形状。

训练限时 1,800 秒；每次仿真限时 600 秒，服务预热等待上限 360 秒；验证限时 300 秒。任一阶段失败会停止，不会自动增加训练次数或改配置。阶段的启动记录位于 `logs/portable-*.json`，旧记录不覆盖。

## 6. 第 500–2,000 步

```bash
"$PI05_WORKDIR/server-env311/bin/python" scripts/pipeline.py train2000
"$PI05_WORKDIR/server-env311/bin/python" scripts/pipeline.py eval2000
"$PI05_WORKDIR/server-env311/bin/python" scripts/pipeline.py verify2000
```

`train2000` 依赖前一阶段的**完整第 500 步检查点**、训练数据和训练指标。它恢复参数与优化器，核对采样序列后追加 1,500 步；仅有 NPZ 适配器无法恢复 AdamW 动量。

学习率沿用原来的 500 步退火计划，额外 1,500 步为 1e-5。其他数据、模型、归一化与评测设置保持不变。累计 2,000 步后停止，不会自行续到 5,000 步。

## 7. 查看新运行结果

实验目录分别为：

```text
$PI05_WORKDIR/experiments/pi05-object-lora-20261118/
$PI05_WORKDIR/experiments/pi05-object-lora-2000-20261118/
```

- `training-summary.json`：是否完成、固定样本损失、参数更新核验和检查点位置。
- `training-metrics.jsonl`：逐步损失、梯度、学习率、抽样帧编号及计时。
- `evaluation/{initial,trained}/libero_object/summary.json`：有效回合数、成功数及错误。
- `evaluation/.../videos/`：逐任务视频。
- `logs/portable-*.json`：命令、源码哈希、退出状态与耗时。

提交到 Git 的历史结果仍在 `records/`，不会被新运行修改。核验历史记录可运行 `python scripts/check_artifacts.py`；它不验证新训练是否已完成，也不声称重新执行了 GPU 实验。

## 只想看命令

```bash
python scripts/pipeline.py train500 --dry-run
python scripts/pipeline.py train2000 --dry-run
python scripts/pipeline.py eval2000 --dry-run
```

此模式不初始化工作目录，不下载、不训练，也不启用 GPU。

## 重新绘制已归档曲线

安装 NumPy 与 matplotlib 后运行：

```bash
python scripts/plot_records.py
```

仅读取 `records/`，默认将图写到 `runtime/figures/training-2000.png`，不会覆盖 README 中已归档的原图。
