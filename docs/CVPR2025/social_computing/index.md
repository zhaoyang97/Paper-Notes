---
title: >-
  CVPR2025 社会计算论文汇总 · 5篇论文解读
description: >-
  5篇CVPR2025的社会计算方向论文解读，涵盖对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "社会计算"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
item_list:
  - u: "as_language_models_scale_low-order_linear_depth_dynamics_emerge/"
    t: "As Language Models Scale, Low-order Linear Depth Dynamics Emerge"
  - u: "classifier-guided_clip_distillation_for_unsupervised_multi-label_classification/"
    t: "Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification"
  - u: "classifier-to-bias_toward_unsupervised_automatic_bias_detection_for_visual_class/"
    t: "Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers"
  - u: "learning_from_neighbors_category_extrapolation_for_long-tail_learning/"
    t: "Learning from Neighbors: Category Extrapolation for Long-Tail Learning"
  - u: "project-probe-aggregate_efficient_fine-tuning_for_group_robustness/"
    t: "Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness"
item_total: 5
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**📷 CVPR2025** · **5** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/social_computing/index.md) · [🔬 ICLR2026 (17)](../../ICLR2026/social_computing/index.md) · [💬 ACL2026 (44)](../../ACL2026/social_computing/index.md) · [🧪 ICML2026 (9)](../../ICML2026/social_computing/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/social_computing/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/social_computing/index.md)

**[As Language Models Scale, Low-order Linear Depth Dynamics Emerge](as_language_models_scale_low-order_linear_depth_dynamics_emerge.md)**

:   将 Transformer 的深度方向视为离散时间动力系统，发现在给定上下文内可以用仅 32 维的线性状态空间代理模型高精度预测层间灵敏度曲线（Spearman 达 0.99），而且令人惊讶的是：**模型越大，低阶线性代理越准确**——这是一条新的 scaling law。

**[Classifier-guided CLIP Distillation for Unsupervised Multi-label Classification](classifier-guided_clip_distillation_for_unsupervised_multi-label_classification.md)**

:   提出 Classifier-guided CLIP Distillation（CCD），通过 CAM 引导的局部视图标签聚合和 CLIP 预测去偏两项核心技术，在完全无标注的条件下达到与全监督方法持平的多标签分类性能（VOC12 上 90.1% mAP）。

**[Classifier-to-Bias: Toward Unsupervised Automatic Bias Detection for Visual Classifiers](classifier-to-bias_toward_unsupervised_automatic_bias_detection_for_visual_class.md)**

:   提出 C2B（Classifier-to-Bias），首个仅依靠分类任务的文本描述（无需任何标注数据）即可自动发现预训练视觉分类器偏差的框架，通过 LLM 生成类特定偏差候选、生成检索标题收集图像数据集、最后计算偏差分数，在 CelebA 和 ImageNet-X 上超越需要监督的 SOTA 偏差检测方法。

**[Learning from Neighbors: Category Extrapolation for Long-Tail Learning](learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)**

:   发现更细粒度的类别划分天然减轻长尾不平衡的影响，提出用 LLM 发现与现有类别相关的细粒度辅助类 + 网络爬虫收集图像 + 邻近静默损失防止辅助类喧宾夺主，在 ImageNet-LT 上 Few 类提升 16 个百分点（41.4→57.4）。

**[Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness](project-probe-aggregate_efficient_fine-tuning_for_group_robustness.md)**

:   提出 PPA（Project-Probe-Aggregate）三步方法，通过投影去除类代理信息放大偏差、以组先验校正探测组标签、聚合组权重，仅需不到 0.01% 可训练参数即可在无组标注情况下提升基础模型的群组鲁棒性。
