# π0.5 / LIBERO 云端运行记录

运行日期：2026-11-18，Asia/Shanghai。已实际完成模型加载和闭环仿真，没有训练模型。

## 结果

| 项目 | 实测结果 |
|---|---|
| GPU | RTX 3090，24GB |
| 系统 | Ubuntu 22.04 |
| 模型 | 官方 `gs://openpi-assets/checkpoints/pi05_libero` |
| 任务集 | LIBERO-Spatial，10 个任务 |
| 评测设置 | 每任务 1 回合，初始状态索引 0，环境 seed=7 |
| 动作生成与执行 | 每次生成 10×7 动作；执行前 5 步后重新观察；默认 10 步 flow integration |
| 完成情况 | 10 回合正常完成，9 成功、1 失败；无环境或通信错误 |
| 客户端评测耗时 | 118.29 秒，含第一轮推理编译及仿真初始化，不含安装和下载 |
| 模型调用次数 | 245 次 |
| 首次调用往返耗时 | 20.25 秒，包含首次 JIT 编译 |
| 稳态服务端推理耗时 | 中位数 145.09ms，P95 149.68ms |
| 稳态客户端往返耗时 | 中位数 146.28ms，P95 151.24ms |
| GPU 设备显存占用峰值 | 17,744MiB，约 17.33GiB |

计时排除了最初 5 次调用，剩余 240 次。上述计时是单客户端、batch=1 的诊断结果，不是专门的吞吐 benchmark，也不能直接解释为真实机器人控制频率。显存数值包含 JAX 70% 预分配、CUDA 与渲染占用，不代表模型最低显存需求。

这是一轮流程验证：每任务仅一个初始状态、单个环境随机种子。**9/10 不等于正式 benchmark 成功率，未复现论文完整评测，也未验证零样本泛化。** 权重已针对 LIBERO 微调。

## 失败回合

任务：`pick up the black bowl on the wooden cabinet and place it on the plate`。

该回合正常运行至步数上限，未满足环境成功条件，不是程序异常。从采样画面可见，碗从柜上移动到桌面/盘子附近，但最终没有完成目标放置；具体失效机制需要进一步检查完整轨迹。

完整视频保存在压缩包的 `results/videos/`。成功与失败示例也已单独下载到本地。

## 云端项目与环境

项目目录：`/home/linux/Documents/pi05-libero-20261118`

- 模型环境：`server-env311`，Python 3.11.18、JAX 0.5.3。
- 仿真环境：`sim-env`，Python 3.10.14；继承现有 torch2.4_cuda12.1 环境中的 PyTorch 2.4.1，额外依赖安装在项目自己的虚拟环境内。
- 仿真关键依赖：robosuite 1.4.1、MuJoCo 3.2.3、NumPy 1.26.4、Numba 0.59.1。完整依赖清单在 `results/server-packages.txt` 与 `results/sim-packages.txt`。
- 官方评测器用于任务执行；外层脚本增加逐回合记录、调用计时，并在任务切换时关闭旧渲染环境。
- 模型服务仅监听服务器本机 `127.0.0.1:8000`。
- 权重已逐文件通过 GCS CRC32C 校验；下载清单保存在 `pi05-manifest.json`。
- 安装中失败的 Python 3.12 尝试与下载校验修复记录均保留在 `logs/`。
- 本轮结束后模型服务和 GPU 监控进程已退出，显存恢复至约 15MiB；权重和环境保留，可继续实验。

源码版本：

| 仓库 | commit |
|---|---|
| Physical-Intelligence/openpi | `215abfb217dbac7d5f1273282331b9b1866c0479` |
| Lifelong-Robot-Learning/LIBERO | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` |
| huggingface/lerobot | `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` |

## 继续运行

以下命令在云服务器执行。终端一启动模型服务：

```bash
cd /home/linux/Documents/pi05-libero-20261118
OPENPI_DATA_HOME="$PWD/cache/openpi" \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
OMP_NUM_THREADS=4 \
server-env311/bin/python -u serve_local.py
```

终端二可用官方评测器运行新的评测。先用每任务 2 回合验证；要做默认每任务 50 回合，先根据实际耗时安排预算。使用新输出目录，避免覆盖本轮视频。

```bash
cd /home/linux/Documents/pi05-libero-20261118
MUJOCO_GL=egl \
PYOPENGL_PLATFORM=egl \
LIBERO_CONFIG_PATH="$PWD/config/libero" \
PYTHONPATH="$PWD/sources/libero" \
sim-env/bin/python openpi-pinned/examples/libero/main.py \
  --args.host 127.0.0.1 \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 2 \
  --args.replan-steps 5 \
  --args.seed 7 \
  --args.video-out-path "$PWD/results-next/videos"
```

注意：官方视频命名不含 episode 索引；同任务同成败的多个回合会覆盖视频。扩大评测前应先给视频文件名添加回合编号。本次每任务只有 1 回合，没有这个覆盖问题。

`run_once.sh` 为本轮固定配置执行脚本，已设置防覆盖检查，不能直接覆盖当前结果。正式比较策略时，应创建新的记录与输出目录，并保持配对初始状态一致。

## 文件索引

- `results/summary.json`：本轮结果和退出状态。
- `results/episodes.jsonl`：10 个回合的任务名、成功状态和异常分类。
- `results/inference.jsonl`：245 次推理计时与动作形状。
- `results/measurements.json`：计时与显存汇总。
- `results/gpu.csv`：每 2 秒一次的设备显存与 GPU 利用率采样。
- `results/provenance.sha256`：运行前协议与执行脚本哈希。
- `results/videos/`：全部 10 个回合的视频。
- `logs/evaluation.log`、`logs/model-server.log`：仿真与模型端原始日志。
- `protocol.json`：运行前的范围、预算和停止条件；实际状态以结果记录为准。

官方参考：[OpenPI](https://github.com/Physical-Intelligence/openpi)、[LIBERO 运行说明](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/examples/libero/README.md)。
