# π0.5：从基础权重到自己的 LoRA 检查点，再到 LIBERO 仿真

## 实际完成的工作

在已有 RTX 3090 24GB 云服务器上，使用官方 `pi05_base` 和真实 LIBERO-Object 专家示范，完成了数据准备、训练集归一化、**500 次真实梯度更新**、第 0 步及第 500 步检查点保存，以及两个检查点各 10 个回合的闭环仿真。未租用新 GPU，未将日志上传到 WandB。

| 项目 | 结果 |
|---|---|
| 专家训练轨迹 | 40 条，10 个任务，每任务 4 条；共 5,934 帧 |
| 独立验证轨迹 | 10 条，每任务 1 条；共 1,452 帧 |
| 训练更新 | 500 步，batch size=1 |
| 可训练 LoRA 参数 | 49,987,584，约占总参数 1.47% |
| 冻结的原始参数 | 3,353,433,872，训练前后全部叶子 SHA256 一致 |
| 发生变化的 LoRA 参数张量 | 20 / 20 |
| 训练期间设备显存采样峰值 | 22,041 MiB，包含 JAX 90% 预分配 |
| 首次编译后梯度更新中位耗时 | 0.290 秒，不含数据读取 |
| 梯度训练阶段耗时 | 231.64 秒，含首次梯度编译，不含初始化、前后诊断和保存 |
| 训练前 Object 闭环测试 | 0/10 |
| 训练后 Object 闭环测试 | 0/10 |

这是训练与验证流程的小规模试跑，不是完整官方训练或已收敛策略。之前官方 LIBERO 微调权重的 10/10 来自另一套权重和运行案例；本报告的训练前后比较使用自己的第 0 步与第 500 步检查点，配置、归一化与仿真初始案例保持一致。

## 训练损失与验证诊断

| 固定样本诊断 | 训练前 | 训练后 |
|---|---:|---:|
| 训练轨迹中点，10 帧均值 | 0.088259 | 0.050647 |
| 验证轨迹中点，10 帧均值 | 0.091266 | 0.058177 |

固定验证样本损失相对变化为降低 36.26%。每个任务取预先选定验证轨迹的中点，训练前后采用相同 flow 噪声种子；这只是 10 帧诊断，**不是全部验证集损失，也不是仿真成功率**。没有依据验证成功率搜索或挑选检查点，报告预先指定的第 500 步。

逐步训练损失前 50 步均值为 0.075228，后 50 步均值为 0.040775。这些更新的样本和噪声不同，单步曲线不应被理解为单调下降的性能指标。图见同目录 `pi05-lora-training.png`。

## 数据与实现

专家数据来自 `physical-intelligence/libero`，固定 revision 为 `a4336d589d589045d1c56423ffdf3b88a0e19b1f`。按任务匹配 Object，按 `42 + task_index` 固定随机顺序，每任务选 4 条训练、1 条验证轨迹。50 个 Parquet 文件均核验源文件 LFS SHA256、帧编号、任务编号、形状和有限数值。轨迹级划分无交叉，动作块在轨迹末端重复最后动作，不跨轨迹。

状态与动作归一化只使用 40 条训练轨迹，每个源帧统计一次，计算均值、标准差及精确 q01/q99；没有使用官方微调权重附带的归一化统计量，也没有使用验证数据拟合。原始动作已是 LIBERO 控制增量，不额外做 delta 转换。

使用固定版本官方 OpenPI 的模型、LIBERO 变换、训练状态初始化、梯度更新和 Orbax 检查点函数。为明确控制小样本轨迹划分，数据读取采用逐轨迹 Parquet 读取器，复用官方输入变换，替代默认的全库 LeRobot loader；它是本项目封装，不是原封不动运行官方训练命令。

模型为 π0.5、action horizon=10、`discrete_state_input=False`，沿用官方 LIBERO 的模型设置。PaliGemma 使用 rank=16 的 LoRA，动作专家使用 rank=32 的 LoRA；**所有原始参数都冻结，包括视觉与动作投影**。训练采用 AdamW、梯度裁剪 1.0、20 步 warmup，学习率由 1e-4 退火至 1e-5，随机种子 42，关闭 EMA。

初始加载时冻结参数转为 bfloat16。原始权重不变的核对比较的是该初始化完成后的训练前、训练后参数，不能解释为与磁盘 float32 权重逐字节相同。共保留两份完整 Orbax 检查点，另导出独立 LoRA 参数 `lora-adapters-step500.npz`；该文件需搭配对应基础权重、模型配置和训练归一化使用，不能单独推理。

初始化阶段曾遇到一次 Flax 参数列表审计接口错误，在任何优化器更新前停止。修复仅涉及审计迭代接口，未更改数值协议；失败日志和原脚本保留在 `attempts/001-parameter-audit-api-error/`，决策记录见 `decisions.jsonl`。

首次闭环验证还遇到了一次首轮推理编译触发 WebSocket 心跳超时，产生 0 个有效模型调用、0 个有效评测回合。它被保留为基础设施错误，没有计作任务失败。修复方式是在模型服务开放连接前预热，之后将策略 RNG 重置为预先声明的 seed=0；仅重跑评测，未增加训练步数。失败记录保留在 `attempts/002-first-inference-keepalive-timeout/`，修复后的评测脚本另有 `evaluation-provenance.sha256`。

## 闭环验证与解释范围

训练前后各跑 Object 的 10 个任务，每任务 1 回合，环境 seed=7，初始状态索引=1，策略服务初始 seed=0，每次用 10 步 flow 求解，执行预测块前 5 步后重新获取观测，最多 280 个控制步。初始状态索引 1 在运行前声明，与此前索引 0 的演示评测区分；**没有审计这些状态是否与专家示范采集状态重叠**，因此不声称验证了对全新场景的泛化。

10 对案例的初始仿真状态、首次机器人状态和两路图像完全一致。模型和仿真程序均完整运行，没有程序错误；成功判定取自固定版本 LIBERO 任务完成条件。视频选择任务顺序中的第一个案例 `pick up the alphabet soup and place it in the basket`，与成败无关。配对视频左侧是第 0 步，右侧是第 500 步，按仿真帧对齐，提前结束的一侧保留末帧并标注。

本次没有进行全参数微调，也未复现官方 30,000 步、全量示范、多回合的完整实验。500 个 batch size=1 更新只覆盖很少的训练样本抽样；有损失下降也不能替代成功率验证。对科研实习项目，可以准确描述为“完成 π0.5 LoRA 训练与 LIBERO 闭环验证流程，并记录实际结果”。

## 文件和复现

云端工程：`/home/linux/Documents/pi05-libero-20261118`。

实验目录：`experiments/pi05-object-lora-20261118/`。

- `protocol.json`、`dataset-manifest.json`：执行前协议、数据版本与完整轨迹划分。
- `data-audit.json`、`assets/`：数据核验与训练集归一化。
- `scripts/lora_common.py`、`prepare_data.py`、`train_pilot.py`：配置、数据与训练。
- `scripts/serve_pilot.py`、`eval_pilot.py`：加载自己的检查点并闭环验证。
- `scripts/load_adapter.py`：将导出的 LoRA 参数与官方基础权重、自己的归一化重新组合；参数树重建核验见 `adapter-export-verification.json`。实际仿真使用完整检查点。
- `training-metrics.jsonl`、`fixed-losses.jsonl`：500 步指标与固定样本前后诊断。
- `frozen-*-sha256.json`、`adapters-*-sha256.json`：参数更新范围的证据。
- `checkpoints/pi05_object_lora_3090/pilot500/0/`、`500/`：完整初始和训练后检查点，保留在云端。
- `lora-adapters-step500.npz`：导出的 LoRA 参数，已纳入下载结果包。
- `evaluation/`：20 个原始回合视频、动作、状态与计时记录。

下载结果包包含脚本、协议、数据清单、统计量、训练指标、参数校验、LoRA 参数和 20 段原始视频及 1 段配对视频；排除约 1GB 的原始数据、基础权重、大体积完整检查点和编译缓存。原始数据可根据固定 revision 和清单重新下载。脚本记录当前云端的绝对路径，移到其他机器时需修改工程目录；运行脚本拒绝覆盖已有实验记录。

OpenPI 固定版本 `215abfb217dbac7d5f1273282331b9b1866c0479`，LIBERO 固定版本 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。服务端使用 Python 3.11.18 / JAX 0.5.3，仿真使用 Python 3.10.14 / MuJoCo 3.2.3 / robosuite 1.4.1。训练与模型服务均已结束，云端数据、环境和检查点保留。

## 逐任务闭环结果

| 任务 | 训练前 | 训练后 |
|---|---|---|
| pick up the alphabet soup and place it in the basket | 失败 | 失败 |
| pick up the cream cheese and place it in the basket | 失败 | 失败 |
| pick up the salad dressing and place it in the basket | 失败 | 失败 |
| pick up the bbq sauce and place it in the basket | 失败 | 失败 |
| pick up the ketchup and place it in the basket | 失败 | 失败 |
| pick up the tomato sauce and place it in the basket | 失败 | 失败 |
| pick up the butter and place it in the basket | 失败 | 失败 |
| pick up the milk and place it in the basket | 失败 | 失败 |
| pick up the chocolate pudding and place it in the basket | 失败 | 失败 |
| pick up the orange juice and place it in the basket | 失败 | 失败 |
