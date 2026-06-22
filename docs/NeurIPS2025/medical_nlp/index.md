---
title: >-
  NeurIPS2025 医疗LLM论文汇总 · 17篇论文解读
description: >-
  17篇NeurIPS2025的医疗 LLM 方向论文解读，涵盖医学影像、LLM、推理、RAG、文本摘要、时序预测等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "NeurIPS2025"
  - "医疗 LLM"
  - "论文解读"
  - "论文笔记"
  - "医学影像"
  - "LLM"
  - "推理"
  - "RAG"
  - "文本摘要"
  - "时序预测"
item_list:
  - u: "ai_should_sense_better_not_just_scale_bigger_adaptive_sensin/"
    t: "AI Should Sense Better, Not Just Scale Bigger: Adaptive Sensing as a Paradigm Shift"
  - u: "cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r/"
    t: "CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research"
  - u: "cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning/"
    t: "CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning"
  - u: "demo_guide-rag_evidence-driven_corpus_curation_for_retrieval-augmented_generatio/"
    t: "Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID"
  - u: "document_summarization_with_conformal_importance_guarantees/"
    t: "Document Summarization with Conformal Importance Guarantees"
  - u: "faithful_summarization_of_consumer_health_queries_a_cross-lingual_framework_with/"
    t: "Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs"
  - u: "h-ddx_a_hierarchical_evaluation_framework_for_differential_diagnosis/"
    t: "H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis"
  - u: "healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt/"
    t: "HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring"
  - u: "large_language_models_as_medical_codes_selectors_a_benchmark_using_the_internati/"
    t: "Large Language Models as Medical Codes Selectors: A Benchmark Using the International Classification of Primary Care"
  - u: "llm-assisted_emergency_triage_benchmark_bridging_hospital-rich_and_mci-like_fiel/"
    t: "LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation"
  - u: "mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r/"
    t: "Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval"
  - u: "patientsim_a_persona-driven_simulator_for_realistic_doctor-patient_interactions/"
    t: "PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions"
  - u: "position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu/"
    t: "Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models"
  - u: "raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica/"
    t: "RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification"
  - u: "shallow_robustness_deep_vulnerabilities_multi-turn_evaluation_of_medical_llms/"
    t: "Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs"
  - u: "the_physical_basis_of_prediction_world_model_formation_in_neural_organoids_via_a/"
    t: "The Physical Basis of Prediction: World Model Formation in Neural Organoids via an LLM-Generated Curriculum"
  - u: "time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri/"
    t: "Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series"
item_total: 17
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🩺 医疗 LLM

**🧠 NeurIPS2025** · **17** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (1)](../../CVPR2026/medical_nlp/index.md) · [🔬 ICLR2026 (20)](../../ICLR2026/medical_nlp/index.md) · [💬 ACL2026 (47)](../../ACL2026/medical_nlp/index.md) · [🧪 ICML2026 (4)](../../ICML2026/medical_nlp/index.md) · [🤖 AAAI2026 (12)](../../AAAI2026/medical_nlp/index.md) · [🧪 ICML2025 (4)](../../ICML2025/medical_nlp/index.md)

🔥 **高频主题：** 医学影像 ×6 · LLM ×4 · 推理 ×2 · RAG ×2 · 文本摘要 ×2

**[AI Should Sense Better, Not Just Scale Bigger: Adaptive Sensing as a Paradigm Shift](ai_should_sense_better_not_just_scale_bigger_adaptive_sensin.md)**

:   这篇立场论文受生物感觉系统的启发，主张AI研究必须从单纯的"扩模型"范式转向"优化输入"——通过在传感器层面动态调整参数（曝光、增益、多模态配置等），使小模型（5M参数的EfficientNet-B0）在理想传感器适应下超越大模型（632M参数的OpenCLIP-H），并提出了从单次感知到闭环感知-运动耦合的渐进式形式化框架。

**[CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)**

:   提出 CGBench，一个基于 ClinGen 专家标注的临床遗传学 benchmark，从变异和基因策展角度评估 LLM 的科学文献推理能力，涵盖证据评分、证据验证和实验证据提取三个任务，发现推理模型在细粒度任务上表现最佳但在高层判断上不如非推理模型。

**[CureAgent: A Training-Free Executor-Analyst Framework for Clinical Reasoning](cureagent_a_training-free_executor-analyst_framework_for_clinical_reasoning.md)**

:   CureAgent 提出 Executor-Analyst 协作框架，将精确工具调用（TxAgent/Llama-8B 做 Executor）与高层临床推理（Gemini 2.5 做 Analyst）解耦，配合分层集成（Stratified Ensemble）的 Late Fusion 拓扑保留证据多样性，在 CURE-Bench 上达到 83.8% 准确率（无需端到端微调），揭示了上下文-性能悖论和动作空间维度灾难两个关键 scaling 发现。

**[Demo: Guide-RAG: Evidence-Driven Corpus Curation for Retrieval-Augmented Generation in Long COVID](demo_guide-rag_evidence-driven_corpus_curation_for_retrieval-augmented_generatio.md)**

:   系统评估了六种 RAG 语料库配置用于长新冠（Long COVID）临床问答，发现将临床指南与高质量系统综述结合的 GS-4 配置在 faithfulness、relevance 和 comprehensiveness 三维度上一致优于单指南和大规模文献库方案，并提出 Guide-RAG 框架和 LongCOVID-CQ 评估数据集。

**[Document Summarization with Conformal Importance Guarantees](document_summarization_with_conformal_importance_guarantees.md)**

:   首次将Conformal Prediction应用于文档摘要，通过校准句子重要性分数的阈值，为抽取式摘要提供用户可控的覆盖率($1-\alpha$)和召回率($\beta$)的严格统计保证，方法模型无关且仅需小规模校准集。

**[Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs](faithful_summarization_of_consumer_health_queries_a_cross-lingual_framework_with.md)**

:   提出结合 TextRank 抽取式句子选择和医学命名实体识别 (NER) 来引导 LLM 生成忠实医学摘要的框架，在英文 MeQSum 和孟加拉语 BanglaCHQ-Summ 数据集上通过微调 LLaMA-2-7B 实现质量和忠实性的一致提升，SummaC 达 0.57，人工评估 82% 摘要保留关键医学信息。

**[H-DDx: A Hierarchical Evaluation Framework for Differential Diagnosis](h-ddx_a_hierarchical_evaluation_framework_for_differential_diagnosis.md)**

:   H-DDx 提出基于 ICD-10 分类层级的鉴别诊断评估框架——将预测和真实诊断扩展到祖先节点后计算层级 F1（HDF1），奖励"临床相关的近似正确"而非仅精确匹配，评估 22 个 LLM 后发现领域特化模型（MediPhi）在 HDF1 上从第 20 名升至第 2 名（Top-5 指标完全遮蔽其优势）。

**[HealthSLM-Bench: Benchmarking Small Language Models for Mobile and Wearable Healthcare Monitoring](healthslm-bench_benchmarking_small_language_models_for_mobile_and_wearable_healt.md)**

:   首个系统评估小语言模型 (SLMs, 1-4B参数) 在移动与可穿戴健康监测任务上表现的基准，覆盖zero-shot/few-shot/指令微调三种范式，并在iPhone上验证了端侧部署的可行性。

**[Large Language Models as Medical Codes Selectors: A Benchmark Using the International Classification of Primary Care](large_language_models_as_medical_codes_selectors_a_benchmark_using_the_internati.md)**

:   构建了一个 extract-retrieve-select 框架的医学编码基准，在 33 个 LLM 上评估 ICPC-2 编码选择能力，发现 28 个模型 F1>0.8，证明 LLM 无需微调即可有效自动化初级保健编码。

**[LLM-Assisted Emergency Triage Benchmark: Bridging Hospital-Rich and MCI-Like Field Simulation](llm-assisted_emergency_triage_benchmark_bridging_hospital-rich_and_mci-like_fiel.md)**

:   基于MIMIC-IV-ED构建了一个开放的、LLM辅助策划的急诊分诊基准数据集，定义了医院丰富资源和大规模伤亡事件(MCI)模拟两种场景，提供基线模型和SHAP可解释性分析，推动分诊预测研究的可复现性和普及化。

**[Mind the Gap: Aligning Knowledge Bases with User Needs to Enhance Mental Health Retrieval](mind_the_gap_aligning_knowledge_bases_with_user_needs_to_enhance_mental_health_r.md)**

:   提出一种基于"需求差距"分析的知识库增强框架，通过叠加真实用户数据（论坛帖子）与现有心理健康资源库来识别内容空白，并用定向增强策略以最少的文档增量达到接近完整语料库的 RAG 检索质量。

**[PatientSim: A Persona-Driven Simulator for Realistic Doctor-Patient Interactions](patientsim_a_persona-driven_simulator_for_realistic_doctor-patient_interactions.md)**

:   提出PatientSim——基于真实MIMIC临床数据和四维人格轴（性格、语言能力、病史记忆水平、认知混乱程度）的LLM患者模拟器，生成37种独特人格组合，在8个LLM上评估事实准确性和人格一致性，由4名临床专家验证平均质量得分3.89/4。

**[Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)**

:   这篇立场论文系统综述了LLM在非结构化临床转录文本主题分析中的应用现状，发现评估方法高度碎片化，并提出以有效性(Validity)、可靠性(Reliability)、可解释性(Interpretability)三维度为核心的标准化评估框架。

**[RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)**

:   提出RAxSS框架，将检索增强机制引入随机稀疏采样(SSS)流水线，通过窗口内相似度加权聚合替代均匀平均，在保持变长医学时间序列分类性能的同时提供从"哪里"到"为什么"的可解释性证据链。

**[Shallow Robustness, Deep Vulnerabilities: Multi-Turn Evaluation of Medical LLMs](shallow_robustness_deep_vulnerabilities_multi-turn_evaluation_of_medical_llms.md)**

:   提出MedQA-Followup框架系统评估医学LLM的多轮鲁棒性，发现模型在单轮扰动下表现尚可（浅层鲁棒性），但在多轮追问中准确率可从91.2%暴跌至13.5%（深层脆弱性），且间接上下文操纵比直接错误建议更具破坏力。

**[The Physical Basis of Prediction: World Model Formation in Neural Organoids via an LLM-Generated Curriculum](the_physical_basis_of_prediction_world_model_formation_in_neural_organoids_via_a.md)**

:   本文提出在人类神经类器官（organoids）中研究世界模型形成的框架，设计了三个渐进式虚拟环境（条件回避、捕食者-猎物、Pong），并引入 LLM 自动生成实验方案的元学习方法，结合多尺度生物物理评估策略量化生物学习的物理基础。

**[Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)**

:   构建 Time-IMM 数据集——首个按因果机制分类不规则性的多模态多变量时序 benchmark（9 种不规则类型分为触发/约束/伪影三大类，9 个数据集），配套 IMM-TSF 预测库支持异步多模态融合，实验表明显式建模多模态在不规则时序上平均降低 MSE 6.71%，最高达 38.38%。
