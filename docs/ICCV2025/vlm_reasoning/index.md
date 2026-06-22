---
title: >-
  ICCV2025 VLMReasoning论文汇总 · 15篇论文解读
description: >-
  15篇ICCV2025的 VLM Reasoning 方向论文解读，涵盖推理、多模态、Agent、LLM、问答、个性化生成等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "ICCV2025"
  - "VLM Reasoning"
  - "论文解读"
  - "论文笔记"
  - "推理"
  - "多模态"
  - "Agent"
  - "LLM"
  - "问答"
  - "个性化生成"
item_list:
  - u: "boosting_mllm_reasoning_with_text-debiased_hint-grpo/"
    t: "Boosting MLLM Reasoning with Text-Debiased Hint-GRPO"
  - u: "chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning/"
    t: "ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning"
  - u: "dwim_towards_tool-aware_visual_reasoning_via_discrepancy-aware_workflow_generati/"
    t: "DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning"
  - u: "finmmr_make_financial_numerical_reasoning_more_multimodal_comprehensive_and_chal/"
    t: "FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging"
  - u: "from_easy_to_hard_the_mir_benchmark_for_progressive_interleaved_multi-image_reas/"
    t: "From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning"
  - u: "llava-cot_let_vision_language_models_reason_step-by-step/"
    t: "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step"
  - u: "mmat1m_a_large_reasoning_dataset_for_multimodal_agent_tuning/"
    t: "MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning"
  - u: "perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat/"
    t: "Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation"
  - u: "physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la/"
    t: "Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models"
  - u: "r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro/"
    t: "R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization"
  - u: "reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q/"
    t: "ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering"
  - u: "taming_the_untamed_graph-based_knowledge_retrieval_and_reasoning_for_mllms_to_co/"
    t: "Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown"
  - u: "toolvqa_a_dataset_for_multistep_reasoning_vqa_with_external/"
    t: "ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools"
  - u: "training-free_personalization_via_retrieval_and_reasoning_on_fingerprints/"
    t: "Training-Free Personalization via Retrieval and Reasoning on Fingerprints"
  - u: "understanding_museum_exhibits_using_vision-language_reasoning/"
    t: "Understanding Museum Exhibits using Vision-Language Reasoning"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🧠 VLM Reasoning

**📹 ICCV2025** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (150)](../../CVPR2026/vlm_reasoning/index.md) · [🔬 ICLR2026 (112)](../../ICLR2026/vlm_reasoning/index.md) · [💬 ACL2026 (32)](../../ACL2026/vlm_reasoning/index.md) · [🧪 ICML2026 (31)](../../ICML2026/vlm_reasoning/index.md) · [🤖 AAAI2026 (10)](../../AAAI2026/vlm_reasoning/index.md) · [🧠 NeurIPS2025 (30)](../../NeurIPS2025/vlm_reasoning/index.md)

🔥 **高频主题：** 推理 ×13 · 多模态 ×7

**[Boosting MLLM Reasoning with Text-Debiased Hint-GRPO](boosting_mllm_reasoning_with_text-debiased_hint-grpo.md)**

:   揭示GRPO在MLLM推理中的两大问题——低数据利用率（难题上所有输出均错误导致梯度无效）和文本偏差（模型忽视图像仅依赖文本推理），提出Hint-GRPO（自适应提供推理提示）和文本偏差校准（测试时增强图像条件）两套方案，在3个基座MLLM上的11个数据集上显著提升推理能力。

**[ChartPoint: Guiding MLLMs with Grounding Reflection for Chart Reasoning](chartpoint_guiding_mllms_with_grounding_reflection_for_chart_reasoning.md)**

:   提出PointCoT方法，将反思性视觉定位（bounding box）集成到图表推理的思维链中，使MLLM在每个推理步骤都能与图表视觉内容交互验证，并构建了包含19.2K高质量样本的ChartPoint-SFT-62k数据集，在ChartBench上实现+5.04%的提升。

**[DWIM: Towards Tool-aware Visual Reasoning via Discrepancy-aware Workflow Generation & Instruct-Masking Tuning](dwim_towards_tool-aware_visual_reasoning_via_discrepancy-aware_workflow_generati.md)**

:   本文提出 DWIM 框架，通过差异感知的工作流生成策略筛选高质量训练数据，以及指令掩码微调策略只克隆有效动作，使 LLM 在组合式视觉推理中具备工具感知能力，在多个 VR 基准上取得 SOTA。

**[FinMMR: Make Financial Numerical Reasoning More Multimodal, Comprehensive, and Challenging](finmmr_make_financial_numerical_reasoning_more_multimodal_comprehensive_and_chal.md)**

:   提出 FinMMR，一个双语（中英文）多模态金融数值推理基准，包含 4300 道题目和 8700 张图像，覆盖 14 个金融子领域，要求模型进行多步精确数值计算；评测了 15 个 SOTA MLLM，最好模型在 Hard 子集仅达 53% 准确率，揭示了当前 MLLM 在专业领域多模态推理中的核心瓶颈。

**[From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning](from_easy_to_hard_the_mir_benchmark_for_progressive_interleaved_multi-image_reas.md)**

:   提出 MIR 基准，包含 22,257 个多图像交错推理问答对及五阶段推理步骤，并设计渐进式课程学习策略，从"简单到困难"逐步提升 MLLM 的多图像交错推理能力。

**[LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](llava-cot_let_vision_language_models_reason_step-by-step.md)**

:   LLaVA-CoT 提出了一种让视觉语言模型自主进行多阶段结构化推理的方法——通过构建 LLaVA-CoT-100k 结构化推理标注数据集训练模型依次执行"总结→视觉解读→逻辑推理→结论生成"四个阶段，并提出阶段级回溯搜索（SWIRES）实现测试时缩放，使 11B 模型超越 Gemini-1.5-pro 和 GPT-4o-mini。

**[MMAT-1M: A Large Reasoning Dataset for Multimodal Agent Tuning](mmat1m_a_large_reasoning_dataset_for_multimodal_agent_tuning.md)**

:   提出首个百万规模的多模态agent调优数据集MMAT-1M，通过四阶段数据引擎（基础数据→推理轨迹生成→反思纠错→格式整合）为MLLM注入CoT推理、工具调用和反思能力，在InternVL2.5-8B上平均提升2.7%，RAG任务上提升8.8%。

**[Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)**

:   提出 Abstract Perspective Change (APC) 框架，通过利用视觉基础模型构建场景抽象表示并执行透视变换，使 VLM 能够从任意视角进行空间推理，在合成与真实图像基准上大幅优于现有 VLM 和微调模型。

**[Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)**

:   提出 Physics Context Builders (PCBs)，一种模块化框架，通过微调小型专用 VLM 从仿真数据中学习生成详细的物理场景描述，作为物理上下文增强大型基础 VLM（如 GPT-4o）的物理推理能力，无需修改大模型本身。

**[R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization](r1-vl_learning_to_reason_with_multimodal_large_language_models_via_step-wise_gro.md)**

:   提出 StepGRPO，一种新的在线强化学习框架，通过两种无需过程奖励模型的规则化步级推理奖励（StepRAR 步级推理准确性奖励 + StepRVR 步级推理有效性奖励），解决 MLLM 在 RL 训练中的稀疏奖励问题，使 MLLM 能够自主探索和改进推理能力。

**[ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)**

:   提出 ReasonVQA 数据集，通过低成本可扩展框架将结构化百科知识（Wikidata）与图像自动融合，生成 1/2/3 跳的多跳推理问题，包含 598K 图像和 4.2M 问题，显著挑战了现有 VQA 模型。

**[Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown](taming_the_untamed_graph-based_knowledge_retrieval_and_reasoning_for_mllms_to_co.md)**

:   以《怪物猎人：世界》为测试平台，构建了包含文本、图像、视频和复杂实体关系的多模态知识图谱(MH-MMKG)，设计了238个复杂查询和多智能体知识检索方法，揭示了当前MLLM在领域特定任务中的知识检索与推理能力不足。

**[ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](toolvqa_a_dataset_for_multistep_reasoning_vqa_with_external.md)**

:   提出ToolVQA，一个23K样本的多模态工具使用VQA数据集，通过ToolEngine数据生成pipeline（图像引导DFS + LCS示例匹配）从真实图像中构造隐式多步推理问题（平均2.78步），在该数据上微调LLaVA-7B后在5个OOD benchmark上超过GPT-3.5-Turbo，并揭示了当前LFM在参数预测和答案总结方面的瓶颈。

**[Training-Free Personalization via Retrieval and Reasoning on Fingerprints](training-free_personalization_via_retrieval_and_reasoning_on_fingerprints.md)**

:   提出R2P，首个免训练的VLM个性化方法，利用VLM自身的世界知识提取概念"指纹"属性，通过检索-推理范式和跨模态属性验证实现个人概念识别，无需任何微调或大规模预训练。

**[Understanding Museum Exhibits using Vision-Language Reasoning](understanding_museum_exhibits_using_vision-language_reasoning.md)**

:   构建了一个包含 6500 万张图片和 2 亿个问答对的大规模博物馆展品数据集 Museum-65，并通过在该数据集上微调 BLIP 和 LLaVA 证明：领域特定的大规模数据集显著优于零样本 SOTA VLM，微调后的 LLaVA 在展品标题和产地识别上分别达到 57% 和 70% 的准确率（vs. GPT-4o 的 22% 和 33%）。
