---
title: >-
  NeurIPS2025 幻觉检测论文汇总 · 17篇论文解读
description: >-
  17篇NeurIPS2025的幻觉检测方向论文解读，涵盖多模态、LLM、推理、对齐/RLHF、对抗鲁棒等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "幻觉检测"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "LLM"
  - "推理"
  - "对齐/RLHF"
  - "对抗鲁棒"
item_list:
  - u: "auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models/"
    t: "Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models"
  - u: "benfords_curse_tracing_digit_bias_to_numerical_hallucination_in_llms/"
    t: "Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs"
  - u: "beyond_token_probes_hallucination_detection_via_activation_tensors_with_act-vit/"
    t: "Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT"
  - u: "causalllava_causal_disentanglement_for_mitigating_hallucinat/"
    t: "Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models"
  - u: "generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf/"
    t: "Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers"
  - u: "generate_but_verify_reducing_hallucination_in_visionlanguage/"
    t: "Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling"
  - u: "glsim_detecting_object_hallucinations_in_lvlms_via_globalloc/"
    t: "GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity"
  - u: "hallucination_as_an_upper_bound_a_new_perspective_on_text-to-image_evaluation/"
    t: "Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation"
  - u: "intervene-all-paths_unified_mitigation_of_lvlm_hallucinations_across_alignment_f/"
    t: "Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats"
  - u: "mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen/"
    t: "Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization"
  - u: "reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la/"
    t: "Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models"
  - u: "robust_hallucination_detection_in_llms_via_adaptive_token_selection/"
    t: "Robust Hallucination Detection in LLMs via Adaptive Token Selection"
  - u: "seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio/"
    t: "SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations"
  - u: "seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m/"
    t: "Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models"
  - u: "systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations/"
    t: "Systematic Reward Gap Optimization for Mitigating VLM Hallucinations"
  - u: "teaming_llms_to_detect_and_mitigate_hallucinations/"
    t: "Teaming LLMs to Detect and Mitigate Hallucinations"
  - u: "when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_/"
    t: "When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 👻 幻觉检测

**🧠 NeurIPS2025** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (33)](../../CVPR2026/hallucination/index.md) · [🔬 ICLR2026 (40)](../../ICLR2026/hallucination/index.md) · [💬 ACL2026 (28)](../../ACL2026/hallucination/index.md) · [🧪 ICML2026 (21)](../../ICML2026/hallucination/index.md) · [🤖 AAAI2026 (15)](../../AAAI2026/hallucination/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/hallucination/index.md)

🔥 **高频主题：** 多模态 ×6 · LLM ×4 · 推理 ×3 · 对齐/RLHF ×2 · 对抗鲁棒 ×2

**[Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)**

:   系统性审计推理大模型（RLLM）中幻觉的产生与传播机制，发现长 CoT 中的反思（reflection）会通过元认知偏差放大幻觉而非纠正它，即使在幻觉源头进行干预也难以改变最终结果（chain disloyalty），揭示现有幻觉检测方法在多步推理场景下严重不足。

**[Benford's Curse: Tracing Digit Bias to Numerical Hallucination in LLMs](benfords_curse_tracing_digit_bias_to_numerical_hallucination_in_llms.md)**

:   本文发现 LLM 的数值幻觉根源于预训练语料中符合 Benford 定律的数字频率分布——数字 1 出现概率 ~30% 而数字 9 仅 ~5%，这种偏差被 FFN 后期层的特定"数字选择性神经元"内化，提出数字选择性分数（DSC）定位偏差神经元并通过剪枝 0.01% 的神经元修正 1.36-3.49% 的错误预测。

**[Beyond Token Probes: Hallucination Detection via Activation Tensors with ACT-ViT](beyond_token_probes_hallucination_detection_via_activation_tensors_with_act-vit.md)**

:   将LLM的全部隐层激活组织为"激活张量"（层×token×隐维度），类比图像用ViT处理，设计ACT-ViT架构支持跨LLM联合训练，在15个LLM-数据集组合上一致超越传统probing方法，并展现出对未见数据集和未见LLM的强零样本/少样本迁移能力。

**[Causal-LLaVA: Causal Disentanglement for Mitigating Hallucination in Multimodal Large Language Models](causalllava_causal_disentanglement_for_mitigating_hallucinat.md)**

:   揭示 MLLM 中物体幻觉的表示层根因——数据集共现偏差导致的语义纠缠，提出双路因果解纠缠框架（Causal-Driven Projector + Causal Intervention Module），通过后门调整在 projector 和最终 Transformer 层分离共现物体表示，使 MME-Perception 提升 22.6%。

**[Generalization or Hallucination? Understanding Out-of-Context Reasoning in Transformers](generalization_or_hallucination_understanding_out-of-context_reasoning_in_transf.md)**

:   本文论证 LLM 的泛化能力和幻觉产生源于同一机制——脱语境推理（OCR），并在单层注意力模型上理论证明：分解参数化 $(W_O, W_V)$ 因梯度下降的核范数隐式偏差而能执行 OCR，而合并参数化 $W_{OV}$ 因 Frobenius 范数偏差而不能，且 OCR 是样本高效的（仅需 $m_{\text{train}}>0$）。

**[Generate, but Verify: Reducing Hallucination in Vision-Language Models with Retrospective Resampling](generate_but_verify_reducing_hallucination_in_visionlanguage.md)**

:   提出REVERSE框架，首次将生成调整和事后验证统一到单个VLM中：通过1.3M半合成样本的幻觉感知训练+推理时回溯重采样，使VLM能在生成过程中自动检测并修正幻觉，在CHAIR-MSCOCO上降低12%、HaloQuest上提升34%。

**[GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)**

:   提出GLSim，一种无训练的LVLM物体幻觉检测方法，通过融合全局场景相似度（物体token与最后instruction token的余弦相似度）和局部视觉定位相似度（物体token与Visual Logit Lens定位的Top-K图像patch的余弦相似度），在MSCOCO上以83.7% AUROC超越SVAR 9%、Internal Confidence 10.8%。

**[Hallucination as an Upper Bound: A New Perspective on Text-to-Image Evaluation](hallucination_as_an_upper_bound_a_new_perspective_on_text-to-image_evaluation.md)**

:   提出将文本到图像（T2I）模型中的幻觉定义为**偏差驱动的偏离**，建立了包含属性、关系和物体三类幻觉的分类学，并论证幻觉评估作为提示对齐评估的"上界"，可揭示模型隐藏偏差。

**[Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats](intervene-all-paths_unified_mitigation_of_lvlm_hallucinations_across_alignment_f.md)**

:   提出 AllPath，一个基于 Transformer 因果架构的多路径幻觉干预框架，首次发现 LVLM 的幻觉不来自单一因果路径而是 image-to-input-text、image-to-output-text、text-to-text 三条路径的交互，并且模型会根据问答对齐格式自适应选择不同路径；通过为每条路径设计轻量级关键 head 识别方法并自适应干预，在 POPE、MCQ-POPE、CHAIR、MME 四个不同格式 benchmark 上一致降低幻觉。

**[Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)**

:   提出 SymMPO（对称多模态偏好优化），通过对比图像的对称配对偏好学习和偏好边际一致性正则化，解决了现有视觉增强型 DPO 方法中目标函数不严格和间接偏好监督两大局限，在五个幻觉评测基准上取得了一致的性能提升。

**[Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)**

:   揭示了RL训练的推理模型（如DeepSeek-R1）比非推理模型产生更多幻觉，从理论上分析了三个根因（高方差梯度、熵约束、伪局部最优），并提出FSPO算法通过步级事实性验证调整token级advantage，在减少幻觉的同时保持甚至提升推理能力。

**[Robust Hallucination Detection in LLMs via Adaptive Token Selection](robust_hallucination_detection_in_llms_via_adaptive_token_selection.md)**

:   HaMI 将幻觉检测建模为多示例学习（MIL）问题，将生成序列视为 token 实例的"bag"，通过联合优化 token 选择和幻觉检测来自适应地定位最具指示性的 token，在四个 QA 基准上以 AUROC 大幅超越所有现有方法（最高提升 11.9%）。

**[SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)**

:   提出 SECA（Semantically Equivalent and Coherent Attacks），通过保持语义等价和语义连贯性的现实主义提示修改来诱发 LLM 幻觉，在多选 QA 任务上实现更高攻击成功率且几乎无语义错误。

**[Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)**

:   针对多模态大模型在退化文档场景下的OCR幻觉问题，提出首个退化文档幻觉评测基准KIE-HVQA，并设计基于GRPO的多目标奖励强化学习框架，在7B参数模型上实现比GPT-4o高约28%的幻觉抑制准确率提升。

**[Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)**

:   提出 Topic-level Preference Rewriting（TPR），通过 topic 级别的细粒度语义控制系统性优化偏好数据中的 reward gap 配置，结合课程学习策略逐步提高负样本难度，在多个幻觉基准上实现约 93% 的幻觉减少。

**[Teaming LLMs to Detect and Mitigate Hallucinations](teaming_llms_to_detect_and_mitigate_hallucinations.md)**

:   将单模型一致性方法（Self-Consistency + Semantic Entropy）推广到多个异构 LLM 的"联盟"设置，通过聚合不同训练背景的模型响应来打破单模型一致性幻觉，在 15 个 LLM 组成的模型池中评估大量联盟组合，发现匹配的强模型联盟在 92% 的情况下超越最强单模型基线，同时推理成本更低。

**[When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)**

:   发现大多模态模型（LMMs）在场景文字识别中存在"语义幻觉"问题（将无语义文本误识为语义合理的词），分析发现注意力集中于文本区域的Transformer层更不易幻觉，据此提出训练无关的ZoomText+Grounded Layer Correction框架，在TextHalu-Bench上提升约4-5%，在ST-VQA上提升约4%。
