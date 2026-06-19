---
title: >-
  ICLR2026 医疗LLM论文汇总 · 15篇论文解读
description: >-
  15篇ICLR2026的医疗 LLM 方向论文解读，涵盖医学影像、对话系统、推理、问答、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICLR2026"
  - "医疗 LLM"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "对话系统"
  - "推理"
  - "问答"
  - "Agent"
item_list:
  - u: "atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue/"
    t: "ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue"
  - u: "biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases/"
    t: "BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases"
  - u: "can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare/"
    t: "Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?"
  - u: "cancer-myth_evaluating_large_language_models_on_patient_questions_with_false_pre/"
    t: "Cancer-Myth: Evaluating Large Language Models on Patient Questions with False Presuppositions"
  - u: "counselbench_llm_mental_health_qa/"
    t: "CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA"
  - u: "emr-agent_automating_cohort_and_feature_extraction_from_emr_databases/"
    t: "EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases"
  - u: "from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for/"
    t: "From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents"
  - u: "histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g/"
    t: "HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction"
  - u: "knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine/"
    t: "Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine"
  - u: "mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo/"
    t: "mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules"
  - u: "medagentgym_agentic_training_biomedical/"
    t: "MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science"
  - u: "medarabench_large-scale_arabic_medical_question_answering_dataset_and_benchmark/"
    t: "MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark"
  - u: "resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and/"
    t: "Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis"
  - u: "simpletom_exposing_the_gap_between_explicit_tom_inference_and_implicit_tom_appli/"
    t: "SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs"
  - u: "survhte-bench_a_benchmark_for_heterogeneous_treatment_effect_estimation_in_survi/"
    t: "SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis"
item_total: 15
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🩺 医疗 LLM

**🔬 ICLR2026** · **15** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (1)](../../CVPR2026/medical_nlp/index.md) · [🧪 ICML2026 (4)](../../ICML2026/medical_nlp/index.md) · [💬 ACL2026 (47)](../../ACL2026/medical_nlp/index.md) · [🤖 AAAI2026 (12)](../../AAAI2026/medical_nlp/index.md) · [🧠 NeurIPS2025 (17)](../../NeurIPS2025/medical_nlp/index.md) · [🧪 ICML2025 (4)](../../ICML2025/medical_nlp/index.md)

🔥 **高频主题：** 医学影像 ×4 · 对话系统 ×2 · 推理 ×2 · 问答 ×2 · Agent ×2

**[ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)**

:   提出 ATPO（自适应树策略优化）算法，将多轮医疗对话建模为层级马尔可夫决策过程（H-MDP），通过不确定性感知的自适应树扩展机制动态分配rollout预算，结合Bellman误差和动作值方差的复合不确定性度量来引导探索，在三个医学对话基准上以Qwen3-8B超越GPT-4o。

**[BiomedSQL: Text-to-SQL for Scientific Reasoning on Biomedical Knowledge Bases](biomedsql_text-to-sql_for_scientific_reasoning_on_biomedical_knowledge_bases.md)**

:   提出 BiomedSQL，首个专门评估 Text-to-SQL 系统在生物医学知识库上科学推理能力的基准，包含 68,000 个问题/SQL/答案三元组，揭示当前最强模型（GPT-o3-mini 62.6%）与领域专家（90%）之间仍有巨大差距。

**[Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)**

:   研究稀疏自编码器（SAE）能否揭示和缓解 LLM 在医疗场景中的种族偏见：发现 SAE 能识别出与种族相关的有害联想（如黑人与暴力），但在复杂临床任务中缓解偏见的效果有限（FLDD < 3%），远不如简单的提示策略（FLDD 8-15%）。

**[Cancer-Myth: Evaluating Large Language Models on Patient Questions with False Presuppositions](cancer-myth_evaluating_large_language_models_on_patient_questions_with_false_pre.md)**

:   本文构建了 Cancer-Myth——一个由肿瘤血液科医生核验、含 585 个"带错误前提"癌症患者问题的对抗数据集，发现包括 GPT-5、Gemini-2.5-Pro、Claude-4-Sonnet 在内的所有前沿 LLM 纠正错误前提的成功率都不超过 43%，且加防范性提示等缓解手段会在"无错误前提"问题上引发大量误纠正、并拖累其他医疗基准，揭示了医疗 LLM 在患者沟通安全上的关键缺口。

**[CounselBench: A Large-Scale Expert Evaluation and Adversarial Benchmarking of LLMs in Mental Health QA](counselbench_llm_mental_health_qa.md)**

:   联合100名持证心理健康专家构建CounselBench双组件基准——CounselBench-EVAL（2,000条六维度专家评估）和CounselBench-Adv（120个对抗性问题+1,080条响应标注），系统性揭示LLM在心理健康开放式问答中表面得分高但存在过度泛化、擅自医疗建议等安全隐患，同时证明LLM-as-Judge在安全关键领域严重不可靠。

**[EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases](emr-agent_automating_cohort_and_feature_extraction_from_emr_databases.md)**

:   提出EMR-AGENT，首个基于LLM Agent的电子病历（EMR）自动化预处理框架，通过动态SQL交互替代手工规则编写，实现跨数据库的队列选择、特征提取和代码映射，在MIMIC-III/eICU/SICdb上表现优异并具强泛化能力。

**[From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents](from_conversation_to_query_execution_benchmarking_user_and_tool_interactions_for.md)**

:   提出EHR-ChatQA基准，首次评估数据库Agent在电子病历场景中的端到端交互工作流（澄清模糊查询→解决术语不匹配→生成SQL→返回答案），发现最强模型(o4-mini)的Pass@5超90%但Pass∧5(全部成功)大幅下降(差距达60%)，暴露了安全关键领域的鲁棒性缺陷。

**[HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)**

:   本文提出 HistoPrism，一个高效的 Transformer 架构，通过交叉注意力注入癌症类型条件来从 H&E 病理图像预测泛癌基因表达，并提出基于 Hallmark/GO 通路的 Gene Pathway Coherence (GPC) 评估框架，在通路级别预测上大幅超越 STPath，尤其在低方差核心生物通路上优势显著。

**[Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine](knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine.md)**

:   提出 LEON（LLM-based Entropy-guided Optimization with kNowledgeable priors），一种数学原理严格的方法，将个性化医疗治疗方案设计建模为条件黑箱优化问题，通过熵约束和对抗性源批评模型引导 LLM 在不微调的情况下作为零样本优化器提出个性化治疗计划。

**[mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules](mclm_a_modular_chemical_language_model_that_generates_functional_and_makeable_mo.md)**

:   提出 mCLM（模块化化学语言模型），通过将分子表示为可合成构建模块的序列，使 LLM 能生成同时满足药理功能和自动化合成可行性的分子，在 430 种 FDA 批准药物上显著改善了药代动力学和毒性性质。

**[MedAgentGym: A Scalable Agentic Training Environment for Code-Centric Reasoning in Biomedical Data Science](medagentgym_agentic_training_biomedical.md)**

:   构建了首个统一的生物医学数据科学 Agent 训练环境 MedAgentGym，包含 72,413 个任务实例（覆盖 12 个真实场景、129 个类别），配备可执行沙盒和可验证 ground truth，系统基准评估 29 个 LLM 揭示商业/开源差距，并通过高效多线程轨迹采样 + 离线/在线 RL 训练出 Med-Copilot，分别获得 +43.02%/+45.28% 提升，达到与 GPT-4o 竞争的性能。

**[MedAraBench: Large-scale Arabic Medical Question Answering Dataset and Benchmark](medarabench_large-scale_arabic_medical_question_answering_dataset_and_benchmark.md)**

:   作者把阿拉伯语地区医学院的纸质考试题手工数字化、清洗成 24,883 道带专业科室和难度标注的医学多选题，构建出大规模阿拉伯语医疗 QA 基准 MedAraBench，并用专家评审 + LLM-as-a-judge 双重质检后，对 16 个开闭源大模型做零样本评测，发现即便最强的 GPT-o3 也只有 0.765 准确率，暴露出当前模型在阿拉伯语医疗推理上的明显短板。

**[Resp-Agent: An Agent-Based System for Multimodal Respiratory Sound Generation and Disease Diagnosis](resp-agent_an_agent-based_system_for_multimodal_respiratory_sound_generation_and.md)**

:   提出 Resp-Agent 闭环多智能体框架，通过主动对抗课程规划器（Thinker-A2CA）协调可控呼吸音生成器与多模态诊断器，在 229k 规模基准上实现生成↔诊断协同设计，大幅提升长尾类别诊断性能。

**[SimpleToM: Exposing the Gap between Explicit ToM Inference and Implicit ToM Application in LLMs](simpletom_exposing_the_gap_between_explicit_tom_inference_and_implicit_tom_appli.md)**

:   SimpleToM 揭示了 LLM 在 Theory of Mind 上的关键缺陷：前沿模型能准确推断他人心理状态（显式 ToM），但在将此知识应用于行为预测和行为判断时性能急剧下降（应用 ToM），暴露了"知道什么"与"如何使用所知"之间的重大鸿沟。

**[SurvHTE-Bench: A Benchmark for Heterogeneous Treatment Effect Estimation in Survival Analysis](survhte-bench_a_benchmark_for_heterogeneous_treatment_effect_estimation_in_survi.md)**

:   提出 SurvHTE-Bench，首个面向右删失生存数据的异质处理效应（HTE）估计综合基准，涵盖 40 个合成数据集、10 个半合成数据集和 2 个真实数据集，系统评估了 53 种估计方法在不同因果假设违反和删失水平下的表现，发现没有单一方法占主导地位，生存 meta-learner（特别是 S-Learner-Survival 和 Matching-Survival）在高删失和假设违反场景下表现最为稳健。
