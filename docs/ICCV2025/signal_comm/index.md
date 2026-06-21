---
title: >-
  ICCV2025 信号/通信论文汇总 · 3篇论文解读
description: >-
  3篇ICCV2025的信号/通信方向论文解读，涵盖多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "信号/通信"
  - "论文解读"
  - "论文笔记"
  - "多模态"
item_list:
  - u: "boosting_multimodal_learning_via_disentangled_gradient_learning/"
    t: "Boosting Multimodal Learning via Disentangled Gradient Learning"
  - u: "generalizable_non-line-of-sight_imaging_with_learnable_physical_priors/"
    t: "Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors"
  - u: "rectifying_magnitude_neglect_in_linear_attention/"
    t: "Rectifying Magnitude Neglect in Linear Attention"
item_total: 3
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📡 信号/通信

**📹 ICCV2025** · **3** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (2)](../../CVPR2026/signal_comm/index.md) · [🔬 ICLR2026 (7)](../../ICLR2026/signal_comm/index.md) · [🧪 ICML2026 (2)](../../ICML2026/signal_comm/index.md) · [🤖 AAAI2026 (3)](../../AAAI2026/signal_comm/index.md) · [🧠 NeurIPS2025 (5)](../../NeurIPS2025/signal_comm/index.md) · [🧪 ICML2025 (3)](../../ICML2025/signal_comm/index.md)

**[Boosting Multimodal Learning via Disentangled Gradient Learning](boosting_multimodal_learning_via_disentangled_gradient_learning.md)**

:   本文揭示了多模态学习中模态编码器和融合模块之间的优化冲突——融合模块会抑制回传到各模态编码器的梯度，导致即使是优势模态也比单模态模型表现差，并提出解耦梯度学习（DGL）框架通过截断融合模块到编码器的梯度并用独立的单模态损失替代来解决此问题。

**[Generalizable Non-Line-of-Sight Imaging with Learnable Physical Priors](generalizable_non-line-of-sight_imaging_with_learnable_physical_priors.md)**

:   提出Learnable Path Compensation (LPC)和Adaptive Phasor Field (APF)两个模块，分别解决NLOS成像中辐射强度衰减的材质依赖性问题和不同信噪比条件下的频域去噪问题，仅在合成数据上训练即可在多种真实数据集上实现SOTA泛化性能。

**[Rectifying Magnitude Neglect in Linear Attention](rectifying_magnitude_neglect_in_linear_attention.md)**

:   揭示 Linear Attention 完全忽略 Query 幅值信息导致注意力分数分布与 Softmax Attention 显著偏离，提出 Magnitude-Aware Linear Attention (MALA)，通过引入缩放因子 β 和偏移项 γ 使线性注意力恢复幅值感知能力，在分类、检测、分割、NLP、语音、图像生成等任务上全面超越现有方法。
