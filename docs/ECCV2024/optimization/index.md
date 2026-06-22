---
title: >-
  ECCV2024 优化/理论论文汇总 · 2篇论文解读
description: >-
  2篇ECCV2024的优化/理论方向论文解读，收录 Fine-Grained Scene Graph Gener、Handling the Non-smooth Challenge in Tensor SVD等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2024"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "fine-grained_scene_graph_generation_via_sample-level_bias_prediction/"
    t: "Fine-Grained Scene Graph Generation via Sample-Level Bias Prediction"
  - u: "handling_the_non-smooth_challenge_in_tensor_svd_a_multi-objective_tensor_recover/"
    t: "Handling the Non-smooth Challenge in Tensor SVD: A Multi-objective Tensor Recovery Framework"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**🎞️ ECCV2024** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/optimization/index.md) · [🔬 ICLR2026 (220)](../../ICLR2026/optimization/index.md) · [🧪 ICML2026 (88)](../../ICML2026/optimization/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/optimization/index.md) · [🧠 NeurIPS2025 (126)](../../NeurIPS2025/optimization/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/optimization/index.md)

**[Fine-Grained Scene Graph Generation via Sample-Level Bias Prediction](fine-grained_scene_graph_generation_via_sample-level_bias_prediction.md)**

:   提出样本级偏置预测方法 SBP，通过 Bias-Oriented GAN 利用物体对 union region 的上下文信息预测样本特异性纠偏向量，将粗粒度关系修正为细粒度关系，在 VG/GQA/VG-1800 上相比数据集级纠偏方法平均提升 5.6%/3.9%/3.2% 的 Average@K。

**[Handling the Non-smooth Challenge in Tensor SVD: A Multi-objective Tensor Recovery Framework](handling_the_non-smooth_challenge_in_tensor_svd_a_multi-objective_tensor_recover.md)**

:   提出基于可学习张量核范数的多目标张量恢复框架 (MOTC)，通过引入可学习酉矩阵替代固定变换来解决 t-SVD 方法在非光滑张量数据上的性能退化问题，并通过多目标优化有效利用张量各维度的低秩性。
