# 模型、数据与实现

## 模型身份

本次基础权重为 OpenPI 的 `pi05_base`，通过其官方 GCS 对象清单及 generation 获取，并核验 CRC32C。**它已经是预训练模型**；本项目是在该权重上微调 LoRA，不是从随机初始化训练整个 VLA。

`pi05_libero` 是此前直接部署测评过的官方 LIBERO 微调权重，未参与本项目自己的 LoRA 初始化。Hugging Face 的 `lerobot/pi05_libero_base` 属于 LeRobot 发布入口，本项目没有使用该仓库权重，不能仅凭名称假定逐参数相同。

当前使用 JAX/OpenPI 路线。安装的固定 LeRobot 源码为 OpenPI 依赖之一，并不表示使用 LeRobot 的 PyTorch π0.5 训练器。

## 固定来源

| 项目 | revision |
|---|---|
| OpenPI | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| LIBERO | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| LeRobot | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |
| `physical-intelligence/libero` 数据 | `a4336d589d589045d1c56423ffdf3b88a0e19b1f` |

## 数据与适配

LIBERO 同时提供任务套件、仿真环境和专家示范数据。本项目使用其中 Object 的 10 个任务。每任务取 4 条训练、1 条验证轨迹，完整轨迹编号、源路径、长度、LFS SHA256 见 [数据清单](../configs/dataset-manifest.json)。整理后的 downloader 直接使用该清单，不再重新随机选样或依赖此前部署结果文件。

原始状态维度 8、动作维度 7；官方变换将动作和状态适配到模型最大维度 32。读取两路 RGB 图像、状态、任务文本与 horizon=10 的动作块；轨迹末尾重复最后动作，不跨越轨迹。数据中的动作已是 LIBERO 控制增量，不再额外做 delta 变换。

训练集逐源帧计算 mean/std/q01/q99，验证轨迹不参与拟合。[参考归一化](../configs/reference-norm-stats.json) 是实际训练的记录；重新准备数据时会重新计算，而非借用官方已微调策略的统计量。

## LoRA 与优化

- π0.5，action horizon=10，`discrete_state_input=False`。
- `gemma_2b_lora`（rank=16）和 `gemma_300m_lora`（rank=32）。
- 所有原始参数冻结，包括视觉与动作投影，只更新名称包含 `lora` 的矩阵。
- 49,987,584 个可训练参数，3,353,433,872 个冻结参数。
- AdamW，梯度裁剪 1.0，batch size=1，seed=42，关闭 EMA 与 WandB。
- 20 步 warmup，峰值 1e-4，500 步退火至 1e-5；续到 2,000 步时保持该下限。
- 冻结参数初始化时转为 bfloat16。训练前后哈希一致的比较对象是转换后的状态，不是磁盘 float32 权重的原始字节。

## 复现实现边界

模型、官方训练状态初始化、梯度更新与 Orbax 保存恢复接口来自固定 OpenPI。数据读取、固定轨迹划分、逐参数审计、日志和续训衔接是项目封装。仿真调用固定官方 evaluator，并增加初始状态选择、计时、动作和状态记录。

独立 NPZ 只含 LoRA 参数，不能单独推理。加载时必须组合对应基础权重、相同模型配置和训练归一化；继续优化还需完整训练检查点里的 AdamW 状态。
