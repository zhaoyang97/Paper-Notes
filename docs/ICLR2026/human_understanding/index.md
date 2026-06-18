---
title: >-
  ICLR2026 人体理解论文汇总 · 8篇论文解读
description: >-
  8篇ICLR2026的人体理解方向论文解读，涵盖情感分析、人脸/视线、对抗鲁棒、LLM、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "人体理解"
  - "论文解读"
  - "论文笔记"
  - "情感分析"
  - "人脸/视线"
  - "对抗鲁棒"
  - "LLM"
  - "多模态"
item_list:
  - u: "bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour/"
    t: "BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis"
  - u: "cross-domain_policy_optimization_via_bellman_consistency_and_hybrid_critics/"
    t: "Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics"
  - u: "event-t2m_event-level_conditioning_for_complex_text-to-motion_synthesis/"
    t: "Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis"
  - u: "gaitsnippet_gait_recognition_beyond_unordered_sets_and_ordered_sequences/"
    t: "GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences"
  - u: "inverse_virtual_try-on_generating_multi-category_product-style_images_from_cloth/"
    t: "Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals"
  - u: "neurogaze-distill_brain-informed_distillation_and_depression-inspired_geometric_/"
    t: "NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition"
  - u: "personax_multimodal_datasets_with_llm-inferred_behavior_traits/"
    t: "PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits"
  - u: "quamo_quaternion_motion_kinematics/"
    t: "QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧑 人体理解

**🔬 ICLR2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (175)](../../CVPR2026/human_understanding/index.md) · [🧪 ICML2026 (4)](../../ICML2026/human_understanding/index.md) · [🤖 AAAI2026 (20)](../../AAAI2026/human_understanding/index.md) · [🧠 NeurIPS2025 (21)](../../NeurIPS2025/human_understanding/index.md) · [📹 ICCV2025 (41)](../../ICCV2025/human_understanding/index.md) · [🧪 ICML2025 (3)](../../ICML2025/human_understanding/index.md)

**[BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)**

:   提出首个面向视频中矛盾/犹豫（A/H）识别的多模态数据集 BAH，包含来自加拿大9省224名参与者的1,118段视频共8.26小时，由行为科学专家标注，并提供了帧级和视频级的基线实验结果。

**[Cross-Domain Policy Optimization via Bellman Consistency and Hybrid Critics](cross-domain_policy_optimization_via_bellman_consistency_and_hybrid_critics.md)**

:   提出 Q Avatar 框架，通过跨域 Bellman 一致性量化源域模型可迁移性，利用自适应无超参权重函数混合源域和目标域 Q 函数，实现在状态-动作空间不同的跨域 RL 中的可靠知识迁移，无论源域模型质量或域相似性如何都能保证不产生负迁移。

**[Event-T2M: Event-level Conditioning for Complex Text-to-Motion Synthesis](event-t2m_event-level_conditioning_for_complex_text-to-motion_synthesis.md)**

:   提出 Event-T2M 框架，将文本提示分解为事件级别的原子动作，结合 TMR 编码器和事件级交叉注意力（ECA）模块注入 Conformer 扩散模型，显著提升多事件复杂动作生成的质量和语义对齐。

**[GaitSnippet: Gait Recognition Beyond Unordered Sets and Ordered Sequences](gaitsnippet_gait_recognition_beyond_unordered_sets_and_ordered_sequences.md)**

:   提出 Snippet 范式：将步态轮廓序列组织为若干"片段"（snippet），每个 snippet 由一个连续区间内随机抽取的帧构成，兼顾短程时序上下文与长程时序依赖，在 Gait3D 上以 2D 卷积骨干达到 77.5% Rank-1，超越所有 3D 卷积方法。

**[Inverse Virtual Try-On: Generating Multi-Category Product-Style Images from Clothed Individuals](inverse_virtual_try-on_generating_multi-category_product-style_images_from_cloth.md)**

:   提出TEMU-VTOFF——面向虚拟脱衣(VTOFF)任务的Dual-DiT架构，通过特征提取器+服装生成器分工协作，结合多模态混合注意力(MHA)融合图像/文本/掩码信息消解视觉歧义，并设计DINOv2驱动的服装对齐器保留高频细节，在VITON-HD和Dress Code多品类场景均达到SOTA。

**[NeuroGaze-Distill: Brain-informed Distillation and Depression-Inspired Geometric Priors for Robust Facial Emotion Recognition](neurogaze-distill_brain-informed_distillation_and_depression-inspired_geometric_.md)**

:   提出 NeuroGaze-Distill 跨模态蒸馏框架：从 EEG 脑电训练的教师模型中提取静态 Valence-Arousal 原型，通过 Proto-KD 和抑郁症启发的几何先验（D-Geo）注入纯视觉学生模型，无需 EEG-人脸配对数据，提升表情识别的跨数据集鲁棒性。

**[PersonaX: Multimodal Datasets with LLM-Inferred Behavior Traits](personax_multimodal_datasets_with_llm-inferred_behavior_traits.md)**

:   构建了 PersonaX 多模态数据集（含 LLM 推断的 Big Five 行为特质、面部嵌入和传记元数据），并提出两层分析框架：结构化独立性检验 + 非结构化因果表示学习（带可识别性理论保证），揭示跨模态因果结构。

**[QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture](quamo_quaternion_motion_kinematics.md)**

:   QuaMo 提出基于四元数微分方程（QDE）的 3D 人体运动学捕捉方法，通过在四元数单位球面约束下求解运动学方程，并引入二阶加速度增强的 meta-PD 控制器，实现了无不连续性、低抖动的在线实时人体运动估计，在 Human3.6M 等多个数据集上超越 SOTA。
