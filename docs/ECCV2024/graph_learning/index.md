---
title: >-
  ECCV2024 图学习论文汇总 · 4篇论文解读
description: >-
  4篇ECCV2024的图学习方向论文解读，涵盖持续学习、扩散模型等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ECCV2024"
  - "图学习"
  - "论文解读"
  - "论文笔记"
  - "持续学习"
  - "扩散模型"
item_list:
  - u: "confidence_self-calibration_for_multi-label_class-incremental_learning/"
    t: "Confidence Self-Calibration for Multi-Label Class-Incremental Learning"
  - u: "gkgnet_group_k-nearest_neighbor_based_graph_convolutional_network_for_multi-labe/"
    t: "GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition"
  - u: "senc_handling_self-collision_in_neural_cloth_simulation/"
    t: "SENC: Handling Self-collision in Neural Cloth Simulation"
  - u: "synchronous_diffusion_for_unsupervised_smooth_non-rigid_3d_shape_matching/"
    t: "Synchronous Diffusion for Unsupervised Smooth Non-Rigid 3D Shape Matching"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**🎞️ ECCV2024** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/graph_learning/index.md) · [🔬 ICLR2026 (118)](../../ICLR2026/graph_learning/index.md) · [💬 ACL2026 (24)](../../ACL2026/graph_learning/index.md) · [🧪 ICML2026 (35)](../../ICML2026/graph_learning/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/graph_learning/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/graph_learning/index.md)

**[Confidence Self-Calibration for Multi-Label Class-Incremental Learning](confidence_self-calibration_for_multi-label_class-incremental_learning.md)**

:   针对多标签类增量学习(MLCIL)中部分标签导致的过度自信预测和假阳性错误问题，提出 Confidence Self-Calibration (CSC) 框架，通过类增量图卷积网络(CI-GCN)校准标签关系 + 最大熵正则化校准置信度，在 MS-COCO 和 VOC 上大幅超越 SOTA。

**[GKGNet: Group K-Nearest Neighbor Based Graph Convolutional Network for Multi-Label Image Recognition](gkgnet_group_k-nearest_neighbor_based_graph_convolutional_network_for_multi-labe.md)**

:   提出首个全图卷积多标签识别模型 GKGNet，通过 Group KNN 机制动态构建标签与图像区域间的图结构，在 MS-COCO 和 VOC2007 上以更低计算量取得 SOTA。

**[SENC: Handling Self-collision in Neural Cloth Simulation](senc_handling_self-collision_in_neural_cloth_simulation.md)**

:   提出 SENC，通过基于 Global Intersection Analysis (GIA) 的自碰撞损失和自碰撞感知图神经网络，首次在自监督神经布料模拟中有效解决布料自碰撞问题。

**[Synchronous Diffusion for Unsupervised Smooth Non-Rigid 3D Shape Matching](synchronous_diffusion_for_unsupervised_smooth_non-rigid_3d_shape_matching.md)**

:   提出同步扩散正则化方法用于无监督非刚性3D形状匹配，核心思想是"在两个形状上同步地扩散同一函数应产生一致输出"，通过这一简单而高效的正则化可以显著提升现有深度功能映射方法的匹配平滑性，在FAUST、SCAPE、TOPKIDS等多个数据集上达到SOTA。
