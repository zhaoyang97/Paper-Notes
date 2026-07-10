---
title: >-
  ECCV2026 优化/理论论文汇总 · 2篇论文解读
description: >-
  2篇ECCV2026的优化/理论方向论文解读，收录 RBE-Flow、SON-GOKU：图着色实现多任务学习的无冲突调度等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ECCV2026"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "rbe-flow_recurrent_bayesian_estimation_on_feature_manifolds_for_cross-modal_regi/"
    t: "RBE-Flow: Recurrent Bayesian Estimation on Feature Manifolds for Cross-Modal Registration"
  - u: "son_goku_graph_coloring_mtl/"
    t: "SON-GOKU：图着色实现多任务学习的无冲突调度"
item_total: 2
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**🎞️ ECCV2026** · **2** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/optimization/index.md) · [🔬 ICLR2026 (220)](../../ICLR2026/optimization/index.md) · [🧪 ICML2026 (88)](../../ICML2026/optimization/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/optimization/index.md) · [🧠 NeurIPS2025 (126)](../../NeurIPS2025/optimization/index.md) · [📹 ICCV2025 (7)](../../ICCV2025/optimization/index.md)

**[RBE-Flow: Recurrent Bayesian Estimation on Feature Manifolds for Cross-Modal Registration](rbe-flow_recurrent_bayesian_estimation_on_feature_manifolds_for_cross-modal_regi.md)**

:   将跨模态密集光流估计重新建模为特征流形上的闭环递归贝叶斯状态估计问题，通过 RMO（递归流形优化）生成带不确定性的流观测、UAPU（不确定性自适应概率更新）用 Sigma 点投影进行 MMSE 最优贝叶斯融合并将后验协方差反馈调节下一次优化，在 OSdataset、WHU-OPT-SAR 和 RoadScene 三个跨模态数据集上一致达到 SOTA，AEPE 相比此前最佳方法降低 45.4%。

**[SON-GOKU：图着色实现多任务学习的无冲突调度](son_goku_graph_coloring_mtl.md)**

:   SON-GOKU 通过实时测量多任务学习中各任务的梯度冲突，构建冲突图并利用贪心图着色算法将任务动态划分为低冲突组，每个训练步只激活一组任务，周期性重着色以适应梯度演化，在六个数据集上一致提升了多种 MTL 基线方法的效果。
