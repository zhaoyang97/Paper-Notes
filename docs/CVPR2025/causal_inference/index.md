---
title: >-
  CVPR2025 因果推理论文汇总 · 4篇论文解读
description: >-
  4篇CVPR2025的因果推理方向论文解读，收录 Adventurer、Image Quality Assessment、Joint Scheduling of Causal Pro等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2025"
  - "因果推理"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "adventurer_optimizing_vision_mamba_architecture_designs_for_efficiency/"
    t: "Adventurer: Optimizing Vision Mamba Architecture Designs for Efficiency"
  - u: "image_quality_assessment_investigating_causal_perceptual_effects_with_abductive_/"
    t: "Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference"
  - u: "joint_scheduling_of_causal_prompts_and_tasks_for_multi-task_learning/"
    t: "Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning"
  - u: "towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif/"
    t: "FG-VCE: Towards Fine-Grained Interpretability — Counterfactual Explanations for Misclassification with Saliency Partition"
item_total: 4
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔗 因果推理

**📷 CVPR2025** · **4** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (4)](../../CVPR2026/causal_inference/index.md) · [🔬 ICLR2026 (63)](../../ICLR2026/causal_inference/index.md) · [💬 ACL2026 (7)](../../ACL2026/causal_inference/index.md) · [🧪 ICML2026 (19)](../../ICML2026/causal_inference/index.md) · [🤖 AAAI2026 (7)](../../AAAI2026/causal_inference/index.md) · [🧠 NeurIPS2025 (20)](../../NeurIPS2025/causal_inference/index.md)

**[Adventurer: Optimizing Vision Mamba Architecture Designs for Efficiency](adventurer_optimizing_vision_mamba_architecture_designs_for_efficiency.md)**

:   提出 Adventurer 系列视觉模型，通过"头部平均池化 token"和"层间翻转"两个简单设计将图像输入适配到单向因果扫描框架中，使 Mamba 架构在视觉任务上实现 4-6 倍于现有 Vision Mamba 的训练速度，同时保持与 ViT 相当甚至更优的精度。

**[Image Quality Assessment: Investigating Causal Perceptual Effects with Abductive Counterfactual Inference](image_quality_assessment_investigating_causal_perceptual_effects_with_abductive_.md)**

:   将全参考图像质量评估（FR-IQA）形式化为反事实推断问题，通过结构因果模型（SCM）区分深度特征中与感知质量因果相关的成分和噪声成分，实现无需训练、可跨骨干网络的鲁棒质量预测，在多个基准数据集上取得竞争性性能。

**[Joint Scheduling of Causal Prompts and Tasks for Multi-Task Learning](joint_scheduling_of_causal_prompts_and_tasks_for_multi-task_learning.md)**

:   提出 JSCPT（Joint Scheduling of Causal Prompts and Tasks）框架，首先设计多任务视觉语言提示（MTVLP）并通过因果干预消除提示中的虚假相关特征，然后通过自适应任务调度器根据训练过程中任务关系的动态变化调整学习顺序和权重，在多个多任务视觉识别基准上取得显著提升。

**[FG-VCE: Towards Fine-Grained Interpretability — Counterfactual Explanations for Misclassification with Saliency Partition](towards_fine-grained_interpretability_counterfactual_explanations_for_misclassif.md)**

:   提出 FG-VCE（Fine-Grained Visual Contrastive Explanation）框架，通过 Shapley 值计算特征点贡献度、显著性分区模块隔离局部特征、以及迭代反事实生成策略，首次实现了对象级和部件级的细粒度反事实解释，揭示模型误分类的具体原因——"哪些细粒度特征导致了错误"以及"哪些局部区域主导了预测改变"。
