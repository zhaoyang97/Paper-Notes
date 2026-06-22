---
title: >-
  AAAI2026 医疗LLM论文汇总 · 12篇论文解读
description: >-
  12篇AAAI2026的医疗 LLM 方向论文解读，涵盖医学影像、LLM、RAG、生物分子等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "AAAI2026"
  - "医疗 LLM"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "LLM"
  - "RAG"
  - "生物分子"
item_list:
  - u: "a_principle-driven_adaptive_policy_for_group_cognitive_stimu/"
    t: "A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment"
  - u: "bica_effective_biomedical_dense_retrieval_with_citation-aware_hard_negatives/"
    t: "BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives"
  - u: "clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp/"
    t: "CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records"
  - u: "expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical/"
    t: "Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering"
  - u: "gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms/"
    t: "GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs"
  - u: "learning_cell-aware_hierarchical_multi-modal_representations/"
    t: "Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling"
  - u: "lucid_learning-enabled_uncertainty-aware_certification_of_stochastic_dynamical_s/"
    t: "LUCID: Learning-Enabled Uncertainty-Aware Certification of Stochastic Dynamical Systems"
  - u: "measuring_stability_beyond_accuracy_in_small_open-source_medical_large_language_/"
    t: "Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology"
  - u: "mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso/"
    t: "MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains"
  - u: "real-time_trust_verification_for_safe_agentic_actions_using_trustbench/"
    t: "Real-Time Trust Verification for Safe Agentic Actions Using TrustBench"
  - u: "shortagesim_simulating_drug_shortages_under_information_asymmetry/"
    t: "ShortageSim: Simulating Drug Shortages under Information Asymmetry"
  - u: "voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he/"
    t: "Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding"
item_total: 12
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🩺 医疗 LLM

**🤖 AAAI2026** · **12** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (1)](../../CVPR2026/medical_nlp/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/medical_nlp/index.md) · [💬 ACL2026 (47)](../../ACL2026/medical_nlp/index.md) · [🧪 ICML2026 (4)](../../ICML2026/medical_nlp/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/medical_nlp/index.md) · [🧪 ICML2025 (4)](../../ICML2025/medical_nlp/index.md)

🔥 **高频主题：** 医学影像 ×4 · LLM ×2 · RAG ×2 · 生物分子 ×2

**[A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)**

:   针对老年认知障碍患者的群体认知刺激治疗（CST）场景，提出GCSD系统：通过多说话人上下文控制、动态参与者状态建模（soft prompt）、认知刺激注意力损失和多维奖励策略优化四个模块，基于Qwen-2.5-3B微调，在500+小时真实粤语CST对话和1万+模拟对话上训练，BLEU-4达27.93超越GPT-4o等大模型，A/B测试胜率50% vs GPT-4o的39%。

**[BiCA: Effective Biomedical Dense Retrieval with Citation-Aware Hard Negatives](bica_effective_biomedical_dense_retrieval_with_citation-aware_hard_negatives.md)**

:   提出利用 PubMed 引文链构建多跳语义图并进行随机游走的 hard negative 挖掘方法，仅用 20k 训练样本和极少微调步数，即让 33M/110M 小模型在 BEIR 和 LoTTE 上超越数十亿参数的检索基线。

**[CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)**

:   提出 CliCARE 框架，将非结构化的纵向癌症电子病历（EHR）转化为时序知识图谱（TKG），并与临床指南知识图谱对齐融合，为 LLM 提供循证依据的临床决策支持，同时设计了与专家评估高度相关的 LLM-as-a-Judge 评估协议。

**[Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering](expert-guided_prompting_and_retrieval-augmented_generation_for_emergency_medical.md)**

:   构建首个EMS急救领域多选QA数据集EMSQA（24.3K题、10个临床主题、4个认证等级），提出Expert-CoT和ExpertRAG框架将领域专业属性注入LLM推理与检索，比标准RAG最高提升4.59%准确率。

**[GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)**

:   GEM 提出了一种生成式熵引导偏好建模方法，通过认知过滤（基于熵的 CoT 评分）和 SEGA 算法（自评估组优势策略优化），在仅 3000 个偏好对的低资源场景下实现高效的 LLM 对齐。

**[Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](learning_cell-aware_hierarchical_multi-modal_representations.md)**

:   本文提出 CHMR 框架，通过结构感知传播解决生物模态缺失问题，引入树状向量量化(Tree-VQ)建模分子-细胞-基因间的层次依赖关系，在9个基准728个任务上分类提升3.6%、回归提升17.2%，实现鲁棒的细胞感知分子表征学习。

**[LUCID: Learning-Enabled Uncertainty-Aware Certification of Stochastic Dynamical Systems](lucid_learning-enabled_uncertainty-aware_certification_of_stochastic_dynamical_s.md)**

:   本文提出 LUCID，首个可为黑盒随机动力系统提供量化安全保证的验证引擎，通过数据驱动的控制障碍证书方法、条件均值嵌入和有限傅里叶核展开，将半无限非凸优化问题重构为可处理的线性规划。

**[Measuring Stability Beyond Accuracy in Small Open-Source Medical Large Language Models for Pediatric Endocrinology](measuring_stability_beyond_accuracy_in_small_open-source_medical_large_language_.md)**

:   系统评估了6个小型开源医学LLM（<10B参数）在儿科内分泌领域的表现，揭示仅靠准确率不足以衡量模型可靠性：语义无关的提示微调导致模型输出显著变化（Stuart-Maxwell p<10⁻⁴），高一致性不等于正确，甚至CUDA版本差异也能引发统计显著的输出偏移。

**[MIRAGE: Scaling Test-Time Inference with Parallel Graph-Retrieval-Augmented Reasoning Chains](mirage_scaling_test-time_inference_with_parallel_graph-retrieval-augmented_reaso.md)**

:   提出MIRAGE框架，将传统的线性推理链扩展为并行多链推理范式，结合结构化医学知识图谱的自适应检索（邻域扩展和多跳遍历），通过跨链验证解决矛盾，在三个医学QA基准上持续优于GPT-4o、ToT和Search-o1等方法。

**[Real-Time Trust Verification for Safe Agentic Actions Using TrustBench](real-time_trust_verification_for_safe_agentic_actions_using_trustbench.md)**

:   提出TrustBench双模式框架：(1) 基准模式——结合传统指标和LLM-as-a-Judge评估8个信任维度，学习Agent置信度与实际正确率的校准映射；(2) 验证模式——在Agent制定行动后、执行前实时计算信任分数，阻止87%的有害行动，延迟低于200ms，通过领域插件（医疗/金融/QA）实现专业化验证。

**[ShortageSim: Simulating Drug Shortages under Information Asymmetry](shortagesim_simulating_drug_shortages_under_information_asymmetry.md)**

:   提出 ShortageSim，首个基于 **LLM 多智能体**的药品短缺模拟框架，建模 FDA 监管者、制造商和购买者在信息不对称下的战略决策，在历史短缺数据上实现对解决滞后时间 84% 的预测改善，为监管策略评估提供受控测试平台。

**[Voices, Faces, and Feelings: Multi-modal Emotion-Cognition Captioning for Mental Health Understanding](voices_faces_and_feelings_multi-modal_emotion-cognition_captioning_for_mental_he.md)**

:   提出情感-认知协同多模态描述（ECMC）任务和框架，通过双流BridgeNet从视频、音频、文本中提取情感和认知特征，利用LLaMA生成自然语言描述，为心理健康评估提供可解释的情感-认知画像，显著提升辅助诊断的准确性和可解释性。
