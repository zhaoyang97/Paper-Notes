---
title: >-
  ICML2026 信号/通信论文汇总 · 2篇论文解读
description: >-
  2篇ICML2026的信号/通信方向论文解读，涵盖少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICML2026"
  - "信号/通信"
  - "论文解读"
  - "论文笔记"
  - "少样本学习"
item_list:
  - u: "joint_model_and_data_sparsification_via_the_marginal_likelihood/"
    t: "Joint Model and Data Sparsification via the Marginal Likelihood"
  - u: "meta-learning_structure-preserving_dynamics/"
    t: "Meta-learning Structure-Preserving Dynamics"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**🧪 ICML2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/signal_comm/index.md) · [🔬 ICLR2026 (7)](../../ICLR2026/signal_comm/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/signal_comm/index.md) · [🧠 NeurIPS2025 (5)](../../NeurIPS2025/signal_comm/index.md) · [📹 ICCV2025 (3)](../../ICCV2025/signal_comm/index.md) · [🧪 ICML2025 (3)](../../ICML2025/signal_comm/index.md)

**[Joint Model and Data Sparsification via the Marginal Likelihood](joint_model_and_data_sparsification_via_the_marginal_likelihood.md)**

:   JMDS 通过**最大化边缘似然**的统一目标同时实现模型和数据稀疏化——避免分阶段优化的次优性，在 CIFAR / ImageNet / WikiText 上以 5-10× 联合压缩比下保持优于独立稀疏化的性能。

**[Meta-learning Structure-Preserving Dynamics](meta-learning_structure-preserving_dynamics.md)**

:   把 modulation-based 元学习（hyper-network 把 latent code $\bm{z}^{(k)}$ 映射成层级调制参数）系统性地引入 Hamiltonian / GENERIC 神经网络，提出两种新颖调制——latent multi-rank (MR) 与 latent SVD-like 调制，让一个共享网络在不知道系统参数 $\bm{\mu}$ 的情况下少样本适配整族新参数实例，同时严格保持能量守恒 / 耗散结构。
