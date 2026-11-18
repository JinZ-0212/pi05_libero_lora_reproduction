# 此前的部署实验：仅供背景参考

这些记录来自自己的 LoRA 训练之前，直接加载官方基础权重和官方 LIBERO 微调权重进行推理。

- [首次 Spatial 部署](spatial-report.md)
- [基础模型两种接入配置](base-deployment-report.md)
- [Object、Goal、Long 部署比较](extended-deployment-report.md)
- [当时的配对结果 JSON](comparison.json)

当时官方 `pi05_libero` 在 Object、Goal、Long 各为 10/10，Spatial 为 9/10；基础版在所测案例中均为 0/10。**这些是另一套已训练权重的部署结果，不是本项目自己训练得到的成绩。**

该比较的两套模型配置同时不同，使用了官方 LIBERO 归一化，不能将差异完全归因于微调，也不能称为严格无目标数据的 zero-shot 评测。其仿真初始状态索引为 0；自己的 LoRA 实验为 1，不能混为同一组案例。

此处保留历史报告原文，其中提到的当时打包文件、云端绝对路径和外部输出不属于本仓库的可移植运行入口。主实验的完整有效视频与证据在仓库 `records/` 中。
