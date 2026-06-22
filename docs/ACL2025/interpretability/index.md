---
title: >-
  ACL2025 可解释性论文汇总 · 22篇论文解读
description: >-
  22篇ACL2025的可解释性方向论文解读，涵盖 LLM、推理、情感分析等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "可解释性"
  - "论文解读"
  - "论文笔记"
  - "LLM"
  - "推理"
  - "情感分析"
item_list:
  - u: "a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be/"
    t: "A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability"
  - u: "an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall/"
    t: "An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall"
  - u: "around_the_world_in_24_hours_probing_llm_knowledge_of_time_and_place/"
    t: "Around the World in 24 Hours: Probing LLM Knowledge of Time and Place"
  - u: "bias_attribution_in_filipino_language_models_extending_a_bias_interpretability_m/"
    t: "Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages"
  - u: "cleme2_gec_evaluation/"
    t: "CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction"
  - u: "degenerate_knowledge_neurons/"
    t: "Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models"
  - u: "expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana/"
    t: "EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations"
  - u: "irt_router_multi_llm/"
    t: "IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory"
  - u: "llama_see_llama_do_entrainment/"
    t: "Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs"
  - u: "mechanistic_interpretability_of_emotion_inference_in_large_language_models/"
    t: "Mechanistic Interpretability of Emotion Inference in Large Language Models"
  - u: "normalized_aopc_faithfulness_metrics/"
    t: "Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability"
  - u: "output_centric_interpretability/"
    t: "Enhancing Automated Interpretability with Output-Centric Feature Descriptions"
  - u: "position-aware_automatic_circuit_discovery/"
    t: "Position-aware Automatic Circuit Discovery"
  - u: "probing_subphonemes_in_morphology_models/"
    t: "Probing Subphonemes in Morphology Models"
  - u: "probing_the_geometry_of_truth_consistency_and_generalization_of_truth_directions/"
    t: "Probing the Geometry of Truth: Consistency and Generalization of Truth Directions"
  - u: "reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti/"
    t: "Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference"
  - u: "safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl/"
    t: "Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety"
  - u: "separating_tongue_from_thought_activation_patching_reveals_language-agnostic_con/"
    t: "Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers"
  - u: "shortcut_neuron_eval/"
    t: "Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis"
  - u: "the_anatomy_of_evidence_an_investigation_into_explainable_icd_coding/"
    t: "The Anatomy of Evidence: An Investigation Into Explainable ICD Coding"
  - u: "the_knowledge_microscope_features_as_better_analytical_lenses_than_neurons/"
    t: "The Knowledge Microscope: Features as Better Analytical Lenses than Neurons"
  - u: "towards_explainable_temporal_reasoning_in_large_language_models_a_structure-awar/"
    t: "Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework"
item_total: 22
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔬 可解释性

**💬 ACL2025** · **22** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (34)](../../CVPR2026/interpretability/index.md) · [🔬 ICLR2026 (195)](../../ICLR2026/interpretability/index.md) · [💬 ACL2026 (63)](../../ACL2026/interpretability/index.md) · [🧪 ICML2026 (91)](../../ICML2026/interpretability/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/interpretability/index.md) · [🧠 NeurIPS2025 (80)](../../NeurIPS2025/interpretability/index.md)

🔥 **高频主题：** LLM ×7 · 推理 ×3

**[A Dual-Perspective NLG Meta-Evaluation Framework with Automatic Benchmark and Better Interpretability](a_dual-perspective_nlg_meta-evaluation_framework_with_automatic_benchmark_and_be.md)**

:   提出一个双视角 NLG 元评估框架，将传统的人-指标相关性分解为全局视角（序数分类，判断粗粒度质量等级）和局部视角（相邻对比，区分细粒度质量差异），并通过自动化基准构建方法避免人工标注和数据污染，在 16 个 LLM 评估器上实验发现 Qwen-2.5-72B 全局最优、DeepSeek-V3 局部最优。

**[An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)**

:   本文系统性地比较了多种机理可解释性方法（因果追踪、激活修补、探针分析等）在定位和解释LLM事实回忆机制方面的表现，揭示了不同方法的一致性、分歧点和各自的适用场景。

**[Around the World in 24 Hours: Probing LLM Knowledge of Time and Place](around_the_world_in_24_hours_probing_llm_knowledge_of_time_and_place.md)**

:   本文提出 GeoTemp 数据集（320k 提示，覆盖 289 个城市和 37 个时区），首次评估 LLM 联合时间和空间推理的能力，发现模型能独立处理时间计算和地理知识，但在需要结合两者时性能急剧下降。

**[Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages](bias_attribution_in_filipino_language_models_extending_a_bias_interpretability_m.md)**

:   将信息论偏见归因分数指标扩展到黏着语（菲律宾语），通过对子词分数取均值来处理复杂词素结构，在 4 个多语言 PLM 上揭示菲律宾语模型的偏见由实体类主题词（人物/物品/关系）驱动，与英语中动作类主题词（犯罪/性行为）形成鲜明对比。

**[CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction](cleme2_gec_evaluation.md)**

:   本文提出 CLEME2.0，一种可解释的 GEC 参考评估指标，通过将编辑解耦为四类（正确纠正 TP、错误纠正 FPne、欠纠正 FN、过纠正 FPun）并结合编辑加权技术，在 GJG15 和 SEEDA 两个人工评判数据集上达到了与人工判断最高相关性的 SOTA 结果。

**[Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](degenerate_knowledge_neurons.md)**

:   本文从结构和功能双重角度重新定义了LLM中的退化知识神经元（DKN），提出神经拓扑聚类方法获取任意数量和结构的DKN，并通过34个实验揭示了DKN与LLM鲁棒性、可进化性和复杂性的内在关联。

**[EXPERT: An Explainable Image Captioning Evaluation Metric with Structured Explanations](expert_an_explainable_image_captioning_evaluation_metric_with_structured_explana.md)**

:   本文提出 EXPERT，一种基于 VLM 微调的无参考图像描述评估指标，通过构建大规模结构化解释数据集并设计两阶段评估模板，在多个基准数据集上达到 SOTA 的同时，提供基于流畅度、相关性、描述性三个维度的高质量结构化解释。

**[IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory](irt_router_multi_llm.md)**

:   IRT-Router 借鉴心理测量学的项目反应理论（IRT），将 LLM 视为"考生"、query 视为"考题"，学习多维能力向量和难度/区分度参数实现可解释的多 LLM 路由，在 OOD 场景下达 87%+ 准确率且成本仅为 GPT-4o 的 1/30。

**[Llama See, Llama Do: A Mechanistic Perspective on Contextual Entrainment and Distraction in LLMs](llama_see_llama_do_entrainment.md)**

:   本文发现并定义了"上下文夹带"(contextual entrainment)现象——LLM会对上下文中出现过的任意token赋予更高概率，并通过可微掩码方法定位了负责该现象的entrainment heads，关闭这些头后可显著抑制干扰效应。

**[Mechanistic Interpretability of Emotion Inference in Large Language Models](mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)**

:   通过 probing、activation patching 和 generation steering 三种机制可解释性技术，发现 LLM 的情感表征功能性地定位于中间层的 MHSA 单元，并基于认知评估理论（appraisal theory）证明这些表征具有心理学合理性，成功通过干预评估概念（如 self-agency、pleasantness）引导情感输出。

**[Normalized AOPC: Fixing Misleading Faithfulness Metrics for Feature Attribution Explainability](normalized_aopc_faithfulness_metrics.md)**

:   本文揭示了广泛使用的 AOPC（扰动曲线下面积）忠实度指标在跨模型比较时会产生误导性结论（因为不同模型的 AOPC 上下界差异巨大），提出 Normalized AOPC (NAOPC) 通过 min-max 归一化消除模型间的不可比性，实验表明归一化可以根本性地改变模型忠实度排名。

**[Enhancing Automated Interpretability with Output-Centric Feature Descriptions](output_centric_interpretability.md)**

:   提出基于输出的特征描述方法（VocabProj和TokenChange），弥补了现有自动化可解释性管线仅依赖输入激活样本的局限，结合输入-输出双视角的集成方法在两类评估中均取得最优表现。

**[Position-aware Automatic Circuit Discovery](position-aware_automatic_circuit_discovery.md)**

:   提出位置感知的边归因修补方法（PEAP）和数据集 Schema 机制，解决了自动电路发现中忽略位置信息导致的抵消效应和重要性高估问题，实现了更小且更忠实的电路发现。

**[Probing Subphonemes in Morphology Models](probing_subphonemes_in_morphology_models.md)**

:   本文提出了一种语言无关的探测方法，研究在形态学变形任务上训练的 Transformer 模型如何隐式习得音韵特征，发现局部特征（如末辅音清化）在音素嵌入中编码良好，而长距离依赖（如元音和谐）在编码器层的上下文化表示中更显著。

**[Probing the Geometry of Truth: Consistency and Generalization of Truth Directions](probing_the_geometry_of_truth_consistency_and_generalization_of_truth_directions.md)**

:   系统性研究LLM内部"真值方向"(truth direction)的一致性与泛化能力，发现只有能力较强的模型才稳定展现一致的真值方向，且基于简单原子陈述训练的真实性探针可泛化至逻辑变换、问答任务和上下文知识场景。

**[Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)**

:   用机械可解释性技术（激活补丁 + Logit Lens + 电路消融）发现语言模型中实现三段论推理的完整电路：三阶段机制——长归纳偏差→中间项抑制（h11.10）→传递项移动，该电路在符号输入上既充分又必要，可迁移到自然语言输入，且跨 GPT-2/Pythia/LLaMA/Qwen 四种架构存在兼容模式。

**[Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety](safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl.md)**

:   提出 Rational 框架，通过推理增强微调让 LLM 在回答前进行显式的安全推理（分析意图、伦理和潜在危害），而非依赖僵硬的拒绝启发式，在保持有用性的同时显著提升对推理层面对抗攻击的鲁棒性。

**[Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers](separating_tongue_from_thought_activation_patching_reveals_language-agnostic_con.md)**

:   通过激活修补实验，首次提供了因果性证据证明大语言模型内部存在与语言解耦的概念表示——模型先确定输出语言，再确定概念，并且跨语言平均的概念表示不仅不损害翻译能力，反而能提升翻译准确率。

**[Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](shortcut_neuron_eval.md)**

:   提出通过对比分析和因果分析定位污染模型中的"捷径神经元"（shortcut neurons），并通过 activation patching 抑制这些神经元，实现更可信的 LLM 评估，与 MixEval 的 Spearman 相关系数超过 0.95。

**[The Anatomy of Evidence: An Investigation Into Explainable ICD Coding](the_anatomy_of_evidence_an_investigation_into_explainable_icd_coding.md)**

:   本文对 MDACE 数据集和当前可解释 ICD 编码系统进行了深入的应用导向分析，揭示了人工标注证据与代码描述的重叠规律、证据在文档中的分布特征，并提出了新的匹配度量来评估模型解释的实用性。

**[The Knowledge Microscope: Features as Better Analytical Lenses than Neurons](the_knowledge_microscope_features_as_better_analytical_lenses_than_neurons.md)**

:   本文通过系统实验验证了 SAE（稀疏自编码器）分解出的特征（features）在知识表达影响力、可解释性、单义性（monosemanticity）三个维度上全面优于传统神经元（neurons）作为分析单元，并提出首个基于 feature 的模型编辑方法 FeatureEdit，在隐私知识擦除任务上大幅超越神经元方法。

**[Towards Explainable Temporal Reasoning in Large Language Models: A Structure-Aware Generative Framework](towards_explainable_temporal_reasoning_in_large_language_models_a_structure-awar.md)**

:   提出 GETER 框架，通过轻量级 Structure-Text Adapter 将时序知识图谱的结构信息注入 LLM，使模型在时序推理任务中既能给出准确预测又能生成可解释的推理说明。
