# LoRA 参数与完整检查点

本地整理时已另存两份独立权重，文件名、大小和 SHA256 见 [weights-manifest.json](weights-manifest.json)：

- `lora-adapters-step500.npz`
- `lora-adapters-step2000.npz`

每份约 177 MiB，超过 GitHub 普通 Git 单文件 100 MiB 限制，故不纳入代码仓库；`.gitignore` 默认排除 NPZ。两份文件与 manifest 统一放在 Release 附件中：

- [Release 页面：v0.1.0-pilot](https://github.com/12345dghdheb/pi05_libero_lora_reproduction/releases/tag/v0.1.0-pilot)
- [下载两份 LoRA 参数与 manifest（371 MB）](https://github.com/12345dghdheb/pi05_libero_lora_reproduction/releases/download/v0.1.0-pilot/pi05-libero-lora-weights.zip)

ZIP 的 SHA256 为 `5fe4710415c8e8c8a55969d5743375abb507b5a75176a56a6ced19fefbdc5cfc`，解压后的两份 NPZ 可分别对照本目录的 manifest 核验。该 Release 标记为预发布，保留本轮 0/10 的实际结果。

## 推理需要哪些文件

基础 `pi05_base` 参数、对应 NPZ、相同模型配置，以及本实验的训练归一化统计量。已有这些文件且按复现说明准备好环境时，可以使用相应阶段的 `load_adapter.py`。

例如加载第 2,000 步适配器：

```bash
export PI05_WORKDIR="$PWD/runtime"
export PYTHONPATH="$PWD/scripts/continuation2000:$PWD/scripts"
export OPENPI_DATA_HOME="$PI05_WORKDIR/cache/openpi"
"$PI05_WORKDIR/server-env311/bin/python" scripts/continuation2000/load_adapter.py \
  --base-params "$PI05_WORKDIR/assets/pi05_base/params" \
  --adapters /absolute/path/to/lora-adapters-step2000.npz \
  --norm-folder "$PWD/records/step2000/assets/pi05_object_lora_3090/local/libero_object_small"
```

这个附加加载器用于组合导出的参数；模型服务首次推理可能需要编译。历史仿真实际使用的是完整 Orbax 检查点，通过 `serve_pilot.py` 预热后服务。不要将参数树重建检查说成“已经用 NPZ 加载器重新跑过仿真”。

## 继续训练需要哪些文件

仅有 NPZ 不能恢复原优化器状态。精确续训还需要完整 Orbax 检查点、原数据清单、归一化、抽样与随机种子协议。完整的第 0、500 和 2,000 步状态保留在原云端工程，未放入普通 Git 或本权重附件。

模型、适配器使用适用上游模型许可；本仓库不重新分发基础模型或专家原始数据。
