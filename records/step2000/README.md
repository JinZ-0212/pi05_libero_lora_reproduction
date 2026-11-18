# π0.5 LoRA：从 500 步续训到累计 2,000 步

第 2,000 步完成 0/10 个任务，本轮仍未出现任务成功。

## 训练与成功率

从自己上次保存的第 500 步完整检查点恢复参数、AdamW 优化器状态，额外执行 **1,500 次真实更新**，累计达到 **2,000 步**。没有重新从基础模型开始训练，没有使用官方 `pi05_libero` 微调权重，也没有新增数据。

| 指标 | 第 0 步（历史结果） | 第 500 步（历史结果） | 第 2,000 步（本轮） |
|---|---:|---:|---:|
| Object 任务完成数 | 0/10 | 0/10 | 0/10 |
| 固定验证轨迹中点损失，10 帧均值 | 0.091266 | 0.058177 | 0.055123 |
| 固定训练轨迹中点损失，10 帧均值 | 0.088259 | 0.050647 | 0.048670 |

第 0、500 步的数值来自已完成记录，本轮只新增第 2,000 步的 10 个仿真回合。固定验证样本损失从第 500 步到第 2,000 步相对变化 **-5.25%**。这只是预先固定的 10 帧诊断，不是全部验证集损失，不能替代闭环成功率。

## 续训是否真实衔接

- 恢复时全局步数为 500，两个优化器计数器均为 500，动量状态随完整检查点恢复。
- 恢复后的全部冻结参数及 LoRA 参数 SHA256 与第 500 步记录一致。
- 重新生成并逐项核对前 500 次数据采样，然后从第 501 次采样继续；训练 flow RNG 沿用相同种子并由全局步数折叠。
- 恢复后的固定训练、验证损失与上次结束时一致。
- 新日志恰好包含第 501–2,000 步，共 1,500 条更新；损失与梯度全部为有限数值，梯度均非零。
- 冻结参数保持不变，20/20 个 LoRA 参数张量相对第 500 步发生变化。
- 保存第 2,000 步完整检查点，仿真服务实际重新加载该检查点；导出的独立 LoRA 文件逐张量哈希验证通过，共 20 个张量。

## 固定配置与资源

数据仍为 LIBERO-Object 的 40 条训练轨迹（5,934 帧）和 10 条验证轨迹（1,452 帧），轨迹级划分不变；归一化仍只由训练集拟合。模型仍为官方 `pi05_base` 初始化的 π0.5，PaliGemma LoRA rank=16、动作专家 rank=32；所有原始参数冻结。batch size=1，seed=42，EMA 关闭。

**学习率保持原来的 500 步退火计划，在本轮续训期间固定于其下限 1e-5。** 没有重启 warmup 或提高学习率。因此该结果回答的是“沿用原计划继续更新”的效果，不等价于从头规划一个 2,000 步的余弦退火训练。

已有 RTX 3090 24GB 上，额外梯度训练阶段耗时 514.00 秒（含首步编译，不含恢复、诊断与保存），预热后单次梯度更新中位耗时 0.291 秒；设备显存采样峰值 22,027 MiB，包含 JAX 90% 预分配，不代表最低显存需求。没有租新机器，没有上传 WandB。

## 仿真协议与限制

10 个 Object 任务，每任务 1 回合，环境 seed=7、初始状态索引=1、策略初始 RNG seed=0，flow 求解 10 步，每执行 5 个动作重新获取观测，最多 280 个控制步。服务开放连接前先预热，然后重置策略 RNG。

第 500 与第 2,000 步的 10 对案例，其初始仿真状态、首次机器人状态、两路相机图像全部一致。新评测完整结束，完成 10 个有效回合，无程序错误。任务完成条件使用固定版本 LIBERO 官方判断。

同一组案例已经看过，因此本轮是探索性续训比较；没有审计这些初始状态与专家示范采集状态是否重叠。单个训练种子、每任务仅 1 回合，不能声称稳定成功率或新场景泛化。最终检查点预先指定为 2,000 步，没有按成功率挑选检查点。

视频取任务顺序中的第一个任务 `pick up the alphabet soup and place it in the basket`，与成败无关。配对视频左侧第 500 步、右侧第 2,000 步，按仿真帧对齐；提前结束的一侧保留末帧并标记。

## 文件与复现

云端工程 `/home/linux/Documents/pi05-libero-20261118`；本轮实验 `experiments/pi05-object-lora-2000-20261118/`。原始 500 步实验保留在 `experiments/pi05-object-lora-20261118/`。

- `protocol.json`、`provenance.sha256`：执行前协议及脚本快照。
- `scripts/train_continue.py`：从父实验第 500 步检查点恢复并续训；`scripts/lora_common.py` 复用父实验原始数据。
- `optimizer-restored-audit.json`、`sampler-resume.json`：优化器与数据采样续接核验。
- `training-metrics.jsonl`、`fixed-losses.jsonl`：新增 1,500 步训练记录与固定样本诊断。
- `reference/`：上次第 500 步评测及训练记录。
- `evaluation/trained/libero_object/`：本轮 10 个有效回合的视频、动作、状态与成功判定。
- `comparison/`：逐任务配对结果、摘要及第 500/2,000 步视频。
- `lora-adapters-step2000.npz`：本轮导出的 LoRA 参数，须配合对应基础模型、配置及训练归一化使用。
- `checkpoints/pi05_object_lora_3090/continuation2000/2000/`：完整检查点，仅保留在云端，不放入下载包。

下载包包含本轮 LoRA 参数、训练和仿真记录、协议、统计量、脚本、曲线与报告；不包含原始数据、基础权重、大体积完整检查点和编译缓存。脚本保留当前云端绝对路径，换机器需修改；恢复训练还依赖父实验检查点。

OpenPI 固定版本 `215abfb217dbac7d5f1273282331b9b1866c0479`，LIBERO 固定版本 `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`；数据 revision `a4336d589d589045d1c56423ffdf3b88a0e19b1f`。本轮在累计 2,000 步及一次 10 任务评测后停止。

## 逐任务结果

| 任务 | 第 500 步 | 第 2,000 步 |
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
