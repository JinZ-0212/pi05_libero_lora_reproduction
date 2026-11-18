# 上传到 GitHub

本项目对应仓库：[12345dghdheb/pi05_libero_lora_reproduction](https://github.com/12345dghdheb/pi05_libero_lora_reproduction)。以下说明适用于首次发布或换机器操作，权重单独作为 Release 附件。

## 代码仓库

仓库名：`pi05_libero_lora_reproduction`。

推荐简介：

> π0.5 LoRA training, checkpoint resume, and LIBERO closed-loop evaluation on one RTX 3090, with auditable logs and negative results.

先在 GitHub 创建空仓库，再在此目录运行：

```bash
git init -b main
git config core.longpaths true
git add .
git commit -m "Add pi05 LoRA training and LIBERO reproduction records"
git remote add origin https://github.com/12345dghdheb/pi05_libero_lora_reproduction.git
git push -u origin main
```

提交作者信息使用你自己的 Git 配置。首次交付 ZIP 不包含 `.git`，解压后执行上面的初始化命令即可；若已有 `origin`，可先用 `git remote -v` 核对地址。Windows 下原始记录的路径较长，本地仓库已设置 `core.longpaths=true`；换电脑克隆时建议放在较短目录。

在仓库主页将 README 固定的实际结果保留。GitHub Actions 只做 Python 标准库核验与脚本语法检查，不会启动下载或训练；本地整理阶段的检查范围记录在 `packaging-validation.json`，在线运行状态以 GitHub Actions 页面为准。

## 权重附件

两份 NPZ 和 `weights-manifest.json` 已作为 `pi05-libero-lora-weights.zip` 放入 [v0.1.0-pilot Release](https://github.com/12345dghdheb/pi05_libero_lora_reproduction/releases/tag/v0.1.0-pilot)，避免把 177 MiB 的文件提交到普通 Git。下载与校验说明见 `models/README.md`。

完整基础模型、专家原始数据和大体积 Orbax 检查点均不在仓库内。原始云端环境和完整检查点没有删除。

## 项目描述

可用于简历的一句话：

> 基于 OpenPI 完成 π0.5 LoRA 微调与 LIBERO 闭环评测流程，覆盖专家数据核验、训练集归一化、检查点保存恢复和参数更新审计；在 RTX 3090 24GB 上完成累计 2,000 步训练并报告实际评测结果。

当前自己的策略在 10 个任务上仍为 0/10，不应写成已复现官方高成功率。新增代码的独立开源许可尚未指定，上游许可副本与来源见 `THIRD_PARTY_NOTICES.md`。
