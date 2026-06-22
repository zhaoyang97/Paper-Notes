---
title: >-
  AAAI2026 自监督/表示学习论文汇总 · 16篇论文解读
description: >-
  16篇AAAI2026的自监督/表示学习方向论文解读，涵盖对抗鲁棒、持续学习、对齐/RLHF等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "自监督/表示学习"
  - "论文解读"
  - "论文笔记"
  - "对抗鲁棒"
  - "持续学习"
  - "对齐/RLHF"
item_list:
  - u: "bce3s_binary_cross-entropy_based_tripartite_synergistic_learning_for_long-tailed/"
    t: "BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition"
  - u: "catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh/"
    t: "CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds"
  - u: "expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e/"
    t: "Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning"
  - u: "explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea/"
    t: "Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning"
  - u: "fedgrpo_privately_optimizing_foundation_models_with_group-relative_rewards_from_/"
    t: "FedGRPO: Privately Optimizing Foundation Models with Group-Relative Rewards from Domain Clients"
  - u: "finextrol_controllable_motion_generation_via_fine-grained_text/"
    t: "FineXtrol: Controllable Motion Generation via Fine-Grained Text"
  - u: "from_pretrain_to_pain_adversarial_vulnerability_of_video_foundation_models_witho/"
    t: "From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models without Finetuning"
  - u: "goal_geometrically_optimal_alignment_for_continual_generalized_category_discover/"
    t: "GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery"
  - u: "hilomix_robust_high-_and_low-frequency_graph_learning_framework_for_mixing_addre/"
    t: "HiLoMix: Robust High- and Low-Frequency Graph Learning Framework for Mixing Address Association"
  - u: "improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt/"
    t: "Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision"
  - u: "improving_sustainability_of_adversarial_examples_in_class-incremental_learning/"
    t: "Improving Sustainability of Adversarial Examples in Class-Incremental Learning"
  - u: "let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-/"
    t: "Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment"
  - u: "robust_tabular_foundation_models/"
    t: "Robust Tabular Foundation Models"
  - u: "self-supervised_inductive_logic_programming/"
    t: "Self-Supervised Inductive Logic Programming"
  - u: "spikingformer_a_key_foundation_model_for_spiking_neural_networks/"
    t: "Spikingformer: A Key Foundation Model for Spiking Neural Networks"
  - u: "towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al/"
    t: "Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space"
item_total: 16
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**🤖 AAAI2026** · **16** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (91)](../../CVPR2026/self_supervised/index.md) · [🔬 ICLR2026 (81)](../../ICLR2026/self_supervised/index.md) · [💬 ACL2026 (1)](../../ACL2026/self_supervised/index.md) · [🧪 ICML2026 (28)](../../ICML2026/self_supervised/index.md) · [🧠 NeurIPS2025 (34)](../../NeurIPS2025/self_supervised/index.md) · [📹 ICCV2025 (13)](../../ICCV2025/self_supervised/index.md)

🔥 **高频主题：** 对抗鲁棒 ×5 · 持续学习 ×3 · 对齐/RLHF ×3

**[BCE3S: Binary Cross-Entropy Based Tripartite Synergistic Learning for Long-tailed Recognition](bce3s_binary_cross-entropy_based_tripartite_synergistic_learning_for_long-tailed.md)**

:   提出 BCE3S，一种基于二元交叉熵（BCE）的三方协同学习框架，将 BCE 式联合学习、BCE 式对比学习和 BCE 式分类器均匀性学习集成在一起，通过 Sigmoid 解耦不同类别的度量来抑制长尾不平衡效应，在 CIFAR10/100-LT、ImageNet-LT 和 iNaturalist2018 上均取得 SOTA。

**[CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds](catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh.md)**

:   提出 CATFormer，一种基于脉冲视觉 Transformer 的无数据重放持续学习框架，通过上下文自适应的动态放电阈值实现任务特定的神经元兴奋性调节，在长达 100 个任务序列中不仅不遗忘反而准确率提升（"逆向遗忘"现象）。

**[Expandable and Differentiable Dual Memories with Orthogonal Regularization for Exemplar-free Continual Learning](expandable_and_differentiable_dual_memories_with_orthogonal_regularization_for_e.md)**

:   提出 **EDD（Expandable and Differentiable Dual Memory）**，一种**无需存储旧样本**的持续学习方法，通过**可微分的共享记忆和任务特定记忆**将数据分解为可复用的子特征，结合**记忆扩展-剪枝**和**正交正则化**机制，在 CIFAR-10/100 和 Tiny-ImageNet 上超越 14 种 SOTA 方法，最终准确率分别达到 55.13%、37.24% 和 30.11%。

**[Explanation-Preserving Augmentation for Semi-Supervised Graph Representation Learning](explanation-preserving_augmentation_for_semi-supervised_graph_representation_lea.md)**

:   提出EPA-GRL（Explanation-Preserving Augmentation），利用少量标签训练的GNN explainer识别图的语义子图（explanation subgraph），增强时只扰动非语义部分（marginal subgraph），实现语义保持的图增强，在6个benchmark上显著优于语义无关的随机增强方法。

**[FedGRPO: Privately Optimizing Foundation Models with Group-Relative Rewards from Domain Clients](fedgrpo_privately_optimizing_foundation_models_with_group-relative_rewards_from_.md)**

:   提出 FedGRPO，将大模型优化重新定义为基于奖励的评估过程，通过能力感知的专家选择和联邦组相对策略优化（仅传输标量奖励信号），实现了隐私保护且通信效率极高的联邦基础模型优化，在数学推理和问答任务上性能接近甚至超越集中式 GRPO。

**[FineXtrol: Controllable Motion Generation via Fine-Grained Text](finextrol_controllable_motion_generation_via_fine-grained_text.md)**

:   提出 FineXtrol 框架，利用带时间标注的细粒度身体部位文本描述作为控制信号，通过双分支 ControlNet 架构和层级对比学习增强文本编码器的区分能力，实现高效、用户友好且精确的可控人体动作生成，在 HumanML3D 上多身体部位控制性能显著优于现有方法。

**[From Pretrain to Pain: Adversarial Vulnerability of Video Foundation Models without Finetuning](from_pretrain_to_pain_adversarial_vulnerability_of_video_foundation_models_witho.md)**

:   提出 Transferable Video Attack (TVA)，仅利用开源视频基础模型（VFM）的嵌入空间即可生成对抗扰动，无需任何下游任务知识便能有效攻击24个视频任务上的下游模型和多模态LLM。

**[GOAL: Geometrically Optimal Alignment for Continual Generalized Category Discovery](goal_geometrically_optimal_alignment_for_continual_generalized_category_discover.md)**

:   基于 Neural Collapse 理论，使用固定等角紧框架（ETF）分类器替代动态分类器，通过监督对齐和置信度引导的无监督对齐实现持续泛化类别发现，在四个基准上遗忘率降低 16.1%、新类发现提升 3.2%。

**[HiLoMix: Robust High- and Low-Frequency Graph Learning Framework for Mixing Address Association](hilomix_robust_high-_and_low-frequency_graph_learning_framework_for_mixing_addre.md)**

:   提出 HiLoMix，一种针对混币地址关联任务的鲁棒图学习框架，通过异质属性混合交互图（HAMIG）、频率感知图对比学习和基于置信度的标签加权监督学习，分别解决图稀疏、标签稀缺和标签噪声三大挑战，在 F1、AUC、MRR 上分别超越次优基线 5.69%、7.34% 和 15.61%。

**[Improving Region Representation Learning from Urban Imagery with Noisy Long-Caption Supervision](improving_region_representation_learning_from_urban_imagery_with_noisy_long-capt.md)**

:   提出 UrbanLN 框架，通过长文本感知的位置编码插值策略和数据-模型双层噪声抑制机制，改善基于 LLM 生成描述的城市区域表征学习。

**[Improving Sustainability of Adversarial Examples in Class-Incremental Learning](improving_sustainability_of_adversarial_examples_in_class-incremental_learning.md)**

:   提出SAE框架解决类增量学习（CIL）中对抗样本因域漂移而失效的问题，通过语义校正模块（CLIP+CIL模型联合引导）和过滤增强模块（去除语义混淆样本），使对抗样本在类别数增长9倍后仍保持攻击效果，平均攻击成功率提升31.28%。

**[Let the Void Be Void: Robust Open-Set Semi-Supervised Learning via Selective Non-Alignment](let_the_void_be_void_robust_open-set_semi-supervised_learning_via_selective_non-.md)**

:   提出 SkipAlign 框架，在对比学习的传统 pull/push 操作之外引入第三种 "skip" 操作，对低置信度样本选择性跳过对齐（只做温和排斥），使 ID 类形成紧凑"星系"、OOD 样本自然散布于"星际虚空"，在未见过的 OOD 检测中平均 AUC 提升 +3.1，最高 +7.1。

**[Robust Tabular Foundation Models](robust_tabular_foundation_models.md)**

:   提出 RTFM——一种模型无关的对抗训练框架，通过在合成数据生成器的参数空间上做 min-max 优化（最大化 TFM 与传统树模型之间的"最优性差距"），仅用不到 10 万额外合成数据集就显著提升了 TabPFN V2 在多个表格基准上的表现。

**[Self-Supervised Inductive Logic Programming](self-supervised_inductive_logic_programming.md)**

:   提出自监督归纳逻辑编程（SS-ILP）新设定及 Poker 系统，仅从少量正标签样本和无标签样本出发，自动生成正负样本，配合最大化通用的二阶确定性范式（SONF）背景理论，在无负样本情况下学习含递归和谓词发明的逻辑程序。

**[Spikingformer: A Key Foundation Model for Spiking Neural Networks](spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)**

:   提出 Spikingformer，通过将 MS Residual 与 Self-Attention 以 spike-driven 方式结合，解决 Spikformer 中 SEW Residual 导致的非脉冲计算问题，同时保持全局建模能力。

**[Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)**

:   提出 L-HAKT 框架，首次将 LLM 双 Agent 与双曲几何结合用于知识追踪：教师 Agent 解析题目语义并构建层级知识图谱，学生 Agent 模拟个体学习行为生成合成数据，通过双曲空间对比对齐校准合成数据与真实数据的分布差异，在四个教育数据集上 AUC 最高达 80.29%，相比 GKT 基线在 EdNet 上 AUC 提升 13.03%。
