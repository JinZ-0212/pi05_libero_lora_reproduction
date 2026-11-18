# π0.5 在 LIBERO Object、Goal、Long 上的仿真复现

日期：2026-11-18。实际补跑了 **60 个回合**，基础版和官方 LIBERO 微调版各 30 个；六组评测全部完成，无程序错误。新增三类任务中，基础版合计 0/30，微调版合计 30/30。

## 结果

| 任务组 | 基础权重 `pi05_base` | 微调权重 `pi05_libero` | 基础版评测耗时 | 微调版评测耗时 |
|---|---:|---:|---:|---:|
| Spatial（此前已完成） | 0/10 | 9/10 | 176.97 秒 | 118.29 秒 |
| Object | 0/10 | 10/10 | 185.36 秒 | 116.40 秒 |
| Goal | 0/10 | 10/10 | 173.14 秒 | 76.50 秒 |
| Long（LIBERO-10） | 0/10 | 10/10 | 285.73 秒 | 152.16 秒 |

每类任务组共 10 个任务，每个任务仅评测 1 个回合。这里记录的是小规模部署与仿真复现，未复现论文完整统计成绩。Spatial 是此前已完成的两次运行，本轮没有重跑；基础版参考使用此前的通用模型默认配置那一组。LIBERO-90 本轮未运行。

## 使用的权重和配置

- 基础版：官方 `gs://openpi-assets/checkpoints/pi05_base`，`Pi0Config(pi05=True)`，action horizon=50，`discrete_state_input=True`。
- 微调版：官方 `gs://openpi-assets/checkpoints/pi05_libero`，官方 `pi05_libero` 配置，action horizon=10，`discrete_state_input=False`。
- 本轮两种模型都只有推理，训练步数和权重更新数均为 0。“微调版”指下载的权重此前已经过官方微调。
- 两组均使用官方 LIBERO 输入输出适配器与 `pi05_libero` 附带的 LIBERO 归一化统计量；没有重新拟合统计量。
- 每次推理执行 10 步 flow 求解，预测后执行前 5 步，再获取新观测并重新规划。
- 环境 seed=7，初始状态索引=0，初始化后先等待 10 步。Object、Goal、Long 的动作步数上限分别为 280、300、520，沿用固定版本官方评测器。
- 先跑基础版三组，再跑微调版三组；每种条件只加载一次模型，策略 RNG 初始 seed=0，在 Object→Goal→Long 之间连续推进，不在每个任务组重新重置。

两种条件的模型配置和权重同时不同，适合比较这两套具体部署方案；不能把结果差异完全归因于微调权重。基础版的跨机器人动作语义与 LIBERO 接口是否兼容仍待验证；使用了目标域统计量，也不能将其描述为严格的无目标数据零样本评测。没有审计基础模型全部预训练数据，不声称其训练时从未见过 LIBERO。

## 配对与动作记录核对

- 30 对案例的初始仿真状态完全一致：True。
- 30 对案例的首次策略输入机器人状态完全一致：True。
- 30 对案例的首次主视角、腕部图像哈希均一致：True。
- 对六组日志逐任务核对了返回动作块的前 5 步与实际执行动作，数值差异均为 0；初始 10 步为官方等待动作。

上述核对可以确认本次两组输入案例一致、实际执行动作符合评测器的动作块截取方式，但不能证明基础模型的动作语义与控制器匹配。

## 视频

保留全部 60 段原始回合视频，并生成三个配对演示视频和三张起始／中间／结束帧对比图。每个任务组选择评测器顺序中的第一个任务，选择与成功或失败无关：

- Object：`pick up the alphabet soup and place it in the basket`。
- Goal：`open the middle drawer of the cabinet`。
- Long（LIBERO-10）：`put both the alphabet soup and the tomato sauce in the basket`。

配对视频左侧为基础版，右侧为微调版，按仿真帧索引对齐；较早结束的一侧保留末帧，并标注 `ended, frame frozen`。播放帧率为 10 FPS，不代表推理耗时或真实机器人控制频率。

## 推理资源

| 任务组 | 条件 | 模型调用数 | 服务端推理中位数 | 服务端 P95 |
|---|---|---:|---:|---:|
| Object | 基础版 | 560 | 150.50 ms | 153.40 ms |
| Object | 微调版 | 283 | 147.82 ms | 152.58 ms |
| Goal | 基础版 | 600 | 150.20 ms | 152.69 ms |
| Goal | 微调版 | 214 | 147.97 ms | 153.38 ms |
| Long（LIBERO-10） | 基础版 | 1040 | 150.36 ms | 153.01 ms |
| Long（LIBERO-10） | 微调版 | 501 | 147.36 ms | 150.69 ms |

RTX 3090 24GB 上，基础版三组运行期间设备显存采样峰值为 18,062 MiB，微调版为 18,062 MiB。设备显存包含 JAX 70% 预分配，不代表最低显存需求。延迟排除各组前 5 次调用，统计单次服务端动作块推理；两组动作块长度不同。评测耗时包含仿真、推理、录像和记录，不含权重下载与模型服务加载，不是物理执行时长。

## 文件与复现入口

完整压缩包保留本轮协议、执行前文件哈希、固定代码版本、依赖清单、运行脚本、逐任务结果、60 段原始录像、3 段配对录像、状态与动作日志、推理计时、显存采样和实际使用的归一化统计量；不含大体积模型权重。`previous-spatial/` 另存此前 Spatial 的结果与计时记录。

云端目录：`/home/linux/Documents/pi05-libero-20261118/experiments/extended-libero-20261118/`。

- `protocol.json`：执行前保存的探索性复现协议，保持原记录。
- `base/`、`tuned/`：各条件的三类任务结果；`episodes.jsonl` 中程序错误与任务失败分开记录。
- `comparison/paired-results.csv`、`comparison/comparison.json`：完整配对结果和核对信息。
- `scripts/run_extended.sh`、`serve_extended.py`、`eval_extended.py`：运行入口、模型服务和官方评测器记录封装。
- `scripts/summarize_extended.py`：评测完成后的汇总与可视化。

固定版本：OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479`；LIBERO `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`；LeRobot `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`。服务端为 Python 3.11.18 / JAX 0.5.3，仿真端为 Python 3.10.14 / MuJoCo 3.2.3 / robosuite 1.4.1。

脚本为当前云端环境保存的复现入口，依赖现有权重目录与已安装环境。若复跑，需要新建实验目录并相应修改脚本中的实验名；运行脚本会拒绝覆盖历史记录。模型服务已停止，云端环境、权重与结果均保留。

## 全部新增任务结果

| 任务组 | 任务 | 基础版 | 微调版 |
|---|---|---|---|
| Object | pick up the alphabet soup and place it in the basket | 失败 | 成功 |
| Object | pick up the cream cheese and place it in the basket | 失败 | 成功 |
| Object | pick up the salad dressing and place it in the basket | 失败 | 成功 |
| Object | pick up the bbq sauce and place it in the basket | 失败 | 成功 |
| Object | pick up the ketchup and place it in the basket | 失败 | 成功 |
| Object | pick up the tomato sauce and place it in the basket | 失败 | 成功 |
| Object | pick up the butter and place it in the basket | 失败 | 成功 |
| Object | pick up the milk and place it in the basket | 失败 | 成功 |
| Object | pick up the chocolate pudding and place it in the basket | 失败 | 成功 |
| Object | pick up the orange juice and place it in the basket | 失败 | 成功 |
| Goal | open the middle drawer of the cabinet | 失败 | 成功 |
| Goal | put the bowl on the stove | 失败 | 成功 |
| Goal | put the wine bottle on top of the cabinet | 失败 | 成功 |
| Goal | open the top drawer and put the bowl inside | 失败 | 成功 |
| Goal | put the bowl on top of the cabinet | 失败 | 成功 |
| Goal | push the plate to the front of the stove | 失败 | 成功 |
| Goal | put the cream cheese in the bowl | 失败 | 成功 |
| Goal | turn on the stove | 失败 | 成功 |
| Goal | put the bowl on the plate | 失败 | 成功 |
| Goal | put the wine bottle on the rack | 失败 | 成功 |
| Long（LIBERO-10） | put both the alphabet soup and the tomato sauce in the basket | 失败 | 成功 |
| Long（LIBERO-10） | put both the cream cheese box and the butter in the basket | 失败 | 成功 |
| Long（LIBERO-10） | turn on the stove and put the moka pot on it | 失败 | 成功 |
| Long（LIBERO-10） | put the black bowl in the bottom drawer of the cabinet and close it | 失败 | 成功 |
| Long（LIBERO-10） | put the white mug on the left plate and put the yellow and white mug on the right plate | 失败 | 成功 |
| Long（LIBERO-10） | pick up the book and place it in the back compartment of the caddy | 失败 | 成功 |
| Long（LIBERO-10） | put the white mug on the plate and put the chocolate pudding to the right of the plate | 失败 | 成功 |
| Long（LIBERO-10） | put both the alphabet soup and the cream cheese box in the basket | 失败 | 成功 |
| Long（LIBERO-10） | put both moka pots on the stove | 失败 | 成功 |
| Long（LIBERO-10） | put the yellow and white mug in the microwave and close it | 失败 | 成功 |
