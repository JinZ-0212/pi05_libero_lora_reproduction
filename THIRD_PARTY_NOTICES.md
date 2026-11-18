# 上游来源与许可

本项目在固定版本的 OpenPI 及 LIBERO 上进行流程封装、数据审计、实验记录与结果分析。没有将原始 VLA 架构或官方训练方法描述为原创。

| 来源 | 作用 | 许可/说明 |
|---|---|---|
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | 模型、训练状态、梯度更新、保存恢复、策略与评测接口 | Apache-2.0；副本见 [LICENSE-openpi](third_party/LICENSE-openpi.txt) |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | 仿真任务、环境与成功条件 | MIT；副本见 [LICENSE-libero](third_party/LICENSE-libero.txt) |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | 固定 OpenPI 环境的依赖 | Apache-2.0；副本见 [LICENSE-lerobot](third_party/LICENSE-lerobot.txt) |
| [physical-intelligence/libero](https://huggingface.co/datasets/physical-intelligence/libero) | 固定专家示范数据来源 | 适用源数据集条款；本仓库只包含清单、统计量与实验记录，未包含原始轨迹文件 |
| OpenPI `pi05_base` | 微调基础权重 | 适用上游模型及其底座许可；基础权重未进入本仓库 |

运行脚本通过导入固定上游实现完成训练与评测。记录中包含当时的源码调用路径、版本和参数哈希；上游完整源码在重新运行时另行下载。

新增封装与文档尚未指定独立开源许可证；此整理过程没有替项目作者作出该许可选择，也没有改变上游代码、数据或模型的许可。
