# 整理与验证范围

## 原始记录

`records/step500` 和 `records/step2000` 从已校验的本地结果包提取，保留文件原始字节；大型 NPZ 移到单独权重附件。每轮 `SOURCE_ARCHIVE.json` 保存源 ZIP 的 SHA256，`records/SHA256SUMS` 校验提取后的文件。

原始脚本、报告和日志中保留历史云端绝对路径。整理时未修改这些记录，因此旧协议、失败尝试和来源快照仍可以对照。`step2000/reference` 是当时为续训复制的第 500 步参考记录，属于重复证据，不计为新增评测。

## 可移植入口

`scripts/pilot500`、`scripts/continuation2000` 中的数值代码与原始脚本一致，只将固定运行根目录改为 `project_paths.ROOT`。新的 bootstrap、数据下载、工作目录初始化和 Python 执行器独立存放。固定数据 downloader 改为直接消费已归档 manifest，避免重新取 latest 或依赖旧部署结果文件。

以下检查不使用 GPU：

1. 两个源 ZIP 和提取记录的完整性，仓库文件大小检查。
2. 500+1,500 步更新日志、有限损失和非零梯度、冻结参数哈希、LoRA 更新范围、续训计数与数据划分。
3. 30 个有效评测回合、结果分母、三组相同初始状态和首次观测；失败心跳尝试不计入有效回合。
4. 可移植数值脚本相对原始脚本只有路径改动，Python AST 与 bootstrap shell 语法检查。
5. 更换包含空格的工作目录后的初始化、幂等性、训练与评测 dry-run。
6. 在已有 Linux Python 环境以 CPU 读取真实一帧训练数据，并导入新的路径配置与 loader；不做优化器更新。

机器可读检查结果见 [packaging-validation.json](../packaging-validation.json)。GitHub workflow 尚未在线执行；完整新环境安装和整理后执行器的端到端 GPU 运行没有重跑。历史 GPU 结果仅来自 `records/` 中的原始执行。
