---
title: >-
  ICCV2025 社会计算论文汇总 · 4篇论文解读
description: >-
  4篇ICCV2025的社会计算方向论文解读，涵盖布局/合成、少样本学习等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "社会计算"
  - "论文解读"
  - "论文笔记"
  - "布局/合成"
  - "少样本学习"
item_list:
  - u: "gradient_extrapolation_for_debiased_representation_learning/"
    t: "Gradient Extrapolation for Debiased Representation Learning"
  - u: "learning_visual_proxy_for_compositional_zero-shot_learning/"
    t: "Learning Visual Proxy for Compositional Zero-Shot Learning"
  - u: "no_more_sibling_rivalry_debiasing_human-object_interaction_detection/"
    t: "No More Sibling Rivalry: Debiasing Human-Object Interaction Detection"
  - u: "propvg_end-to-end_proposal-driven_visual_grounding_with_multi-granularity_discri/"
    t: "PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👥 社会计算

**📹 ICCV2025** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (3)](../../CVPR2026/social_computing/index.md) · [🔬 ICLR2026 (17)](../../ICLR2026/social_computing/index.md) · [💬 ACL2026 (44)](../../ACL2026/social_computing/index.md) · [🧪 ICML2026 (9)](../../ICML2026/social_computing/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/social_computing/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/social_computing/index.md)

**[Gradient Extrapolation for Debiased Representation Learning](gradient_extrapolation_for_debiased_representation_learning.md)**

:   提出 GERNE 方法，通过构建具有不同虚假相关程度的两个 batch 并对其梯度进行线性外推，引导模型学习去偏差表征，在已知和未知属性情况下均优于 SOTA。

**[Learning Visual Proxy for Compositional Zero-Shot Learning](learning_visual_proxy_for_compositional_zero-shot_learning.md)**

:   提出 Visual Proxy（视觉代理）概念，在 CZSL 任务中首次引入文本引导的视觉类中心，并通过跨模态联合学习（CMJL）协同优化文本原型与视觉代理，在四个 CZSL 基准上达到闭世界 SOTA。

**[No More Sibling Rivalry: Debiasing Human-Object Interaction Detection](no_more_sibling_rivalry_debiasing_human-object_interaction_detection.md)**

:   发现并系统分析了 HOI 检测中的"有毒兄弟"偏差问题——高度相似的 HOI 三元组在输入端和输出端相互干扰竞争，提出"对比后校准"（C2C）和"合并后拆分"（M2S）两种去偏学习目标，在 HICO-DET 上超越 baseline +9.18% mAP、超越前 SOTA +3.59%。

**[PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination](propvg_end-to-end_proposal-driven_visual_grounding_with_multi-granularity_discri.md)**

:   提出PropVG，首个无需预训练检测器的端到端proposal-based视觉定位框架，将视觉定位分解为前景proposal生成+基于对比学习的指代评分两阶段，并引入多粒度目标判别模块（MTD）融合物体级和语义级信息判断目标是否存在，在10个数据集上刷新SOTA且推理速度比传统proposal方法快4倍。
