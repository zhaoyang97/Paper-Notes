---
title: >-
  CVPR2025 VLMReasoning论文汇总 · 13篇论文解读
description: >-
  13篇CVPR2025的 VLM Reasoning 方向论文解读，涵盖推理、多模态、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2025"
  - "VLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "LLM"
item_list:
  - u: "beyond_final_answers_crystal_benchmark_for_transparent_multimodal_reasoning_eval/"
    t: "Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation"
  - u: "coarse_correspondences_boost_spatial-temporal_reasoning_in_multimodal_language_m/"
    t: "Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models"
  - u: "critic-v_vlm_critics_help_catch_vlm_errors_in_multimodal_reasoning/"
    t: "Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning"
  - u: "document_haystacks_vision-language_reasoning_over_piles_of_1000_documents/"
    t: "Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents"
  - u: "espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_/"
    t: "ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models"
  - u: "insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m/"
    t: "Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models"
  - u: "mm-condchain_a_programmatically_verified_benchmark_for_visually_grounded_deep_co/"
    t: "MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning"
  - u: "mv-math_evaluating_multimodal_math_reasoning_in_multi-visual_contexts/"
    t: "MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts"
  - u: "reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat/"
    t: "Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence"
  - u: "seqafford_sequential_3d_affordance_reasoning_via_multimodal_large_language_model/"
    t: "SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model"
  - u: "spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava/"
    t: "Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA"
  - u: "thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea/"
    t: "Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World"
  - u: "thinking_in_space_how_multimodal_large_language_models_see_remember_and_recall_s/"
    t: "Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces"
item_total: 13
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 VLM Reasoning

**📷 CVPR2025** · **13** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (150)](../../CVPR2026/vlm_reasoning/index.md) · [🔬 ICLR2026 (112)](../../ICLR2026/vlm_reasoning/index.md) · [💬 ACL2026 (32)](../../ACL2026/vlm_reasoning/index.md) · [🧪 ICML2026 (31)](../../ICML2026/vlm_reasoning/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/vlm_reasoning/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/vlm_reasoning/index.md)

🔥 **高频主题：** 推理 ×11 · 多模态 ×10 · LLM ×4

**[Beyond Final Answers: CRYSTAL Benchmark for Transparent Multimodal Reasoning Evaluation](beyond_final_answers_crystal_benchmark_for_transparent_multimodal_reasoning_eval.md)**

:   提出 CRYSTAL benchmark（6372 实例），通过 Match F1 和 Ordered Match F1 两个指标在中间推理步骤层面评估 MLLM，揭示了普遍的 cherry-picking 行为和推理顺序混乱问题，并提出 CPR-Curriculum 训练策略改善推理质量。

**[Coarse Correspondences Boost Spatial-Temporal Reasoning in Multimodal Language Models](coarse_correspondences_boost_spatial-temporal_reasoning_in_multimodal_language_m.md)**

:   本文提出Coarse Correspondences，一种轻量级的training-free视觉提示方法，通过在图像帧上叠加目标跟踪得到的粗粒度实例对应关系标记，显著增强MLLM的空间时序推理能力，在ScanQA上提升+20.5%、OpenEQA上+9.7%、EgoSchema上+6.0%和R2R导航上+11%。

**[Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning](critic-v_vlm_critics_help_catch_vlm_errors_in_multimodal_reasoning.md)**

:   本文提出Critic-V框架，将VLM推理过程解耦为Reasoner（推理器）和Critic（评价器），通过DPO训练的Critic模型提供自然语言反馈迭代优化推理路径，在8个基准上的5个超越GPT-4V，数学推理任务提升尤为显著（MathVista +11.8%）。

**[Document Haystacks: Vision-Language Reasoning Over Piles of 1000+ Documents](document_haystacks_vision-language_reasoning_over_piles_of_1000_documents.md)**

:   提出 DocHaystack 和 InfoHaystack 两个大规模文档检索基准（每个问题对应 1000+ 文档），以及 V-RAG——一个视觉中心的检索增强生成框架，在 Recall@1 上比最佳基线提升 9%-11%。

**[ESPIRE: A Diagnostic Benchmark for Embodied Spatial Reasoning of Vision-Language Models](espire_a_diagnostic_benchmark_for_embodied_spatial_reasoning_of_vision-language_.md)**

:   提出 Espire，一个基于仿真环境的具身空间推理诊断基准，将 VLM 评估分解为定位和执行两阶段，通过全生成式范式系统评估 VLM 在多种空间推理维度和粒度上的能力。

**[Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)**

:   Insight-V 提出一个包含数据生成 pipeline 和多智能体推理系统的视觉推理增强方案：通过渐进式生成+多粒度评估构建高质量长链推理数据，设计推理Agent和总结Agent协作解题，配合迭代DPO进一步提升推理质量，在7个视觉推理基准上实现平均7%的提升。

**[MM-CondChain: A Programmatically Verified Benchmark for Visually Grounded Deep Compositional Reasoning](mm-condchain_a_programmatically_verified_benchmark_for_visually_grounded_deep_co.md)**

:   MM-CondChain 是首个针对视觉基础深层组合推理的 MLLM 基准，通过可验证程序中间表示（VPIR）自动构建多层条件链和链式硬负样本，最强模型仅获 53.33 Path F1，揭示深层组合推理是根本挑战。

**[MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts](mv-math_evaluating_multimodal_math_reasoning_in_multi-visual_contexts.md)**

:   本文提出 MV-MATH 基准，包含 2,009 道高质量多图数学题（来自真实 K-12 场景），系统评估了 25 个多模态大模型在多图数学推理场景下的能力，发现所有模型远低于人类水平（最佳 Claude 仅 33.9%），揭示了多图数学推理仍是 MLLM 的重大挑战。

**[Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)**

:   提出 VAEX-Bench 基准，首次系统评估 MLLM 的"抽象时空推理"能力——不是从单帧提取信息，而是需要跨房间/跨时间整合观察来推断全局空间布局、跨场景计数等，发现所有 SOTA 模型（包括 GPT-5.2、Gemini-3 Pro）在抽象推理上表现远低于人类。

**[SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model](seqafford_sequential_3d_affordance_reasoning_via_multimodal_large_language_model.md)**

:   提出 Sequential 3D Affordance Reasoning 任务，构建180K指令-点云对基准，通过在3D MLLM中引入 `<SEG>` token 和多粒度语言-点云融合模块，从复杂人类指令中推理并分割出序列化的affordance区域。

**[Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)**

:   通过在 LLaVA 框架中系统替换图像编码器（CLIP/SigLIP/SigLIP2/AIMv2）和引入 2D-RoPE 位置编码，发现 VLM 的空间推理能力主要由编码器的训练目标决定，指望仅靠 2D 位置结构改善空间理解是不够的。

**[Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)**

:   本文提出 Dyn-Bench——首个系统评估多模态大模型（MLLMs）在物理4D世界中动态感知、追踪和推理能力的大规模基准，包含 1K 视频、7K VQA 对和 3K 动态目标定位对，发现现有模型无法同时在时空推理和动态定位上表现良好，并提出 Mask-Guided Fusion 和 ST-TCM 两种结构化增强方法显著提升表现。

**[Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](thinking_in_space_how_multimodal_large_language_models_see_remember_and_recall_s.md)**

:   本文提出 VSI-Bench，一个基于视频的视觉空间智能基准（5000+ QA对），系统评估了 MLLM 的空间推理能力，发现空间推理是主要瓶颈，传统语言推理技术（CoT等）无法提升性能，但显式生成认知地图可改善空间距离推理。
