---
title: >-
  AAAI2026 信息检索/RAG论文汇总 · 21篇论文解读
description: >-
  21篇AAAI2026的信息检索/RAG 方向论文解读，涵盖 RAG、推理、LLM、Agent、对话系统、问答等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "AAAI2026"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "RAG"
  - "推理"
  - "LLM"
  - "Agent"
  - "对话系统"
  - "问答"
item_list:
  - u: "as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan/"
    t: "\"As Eastern Powers, I Will Veto.\" : An Investigation of Nation-Level Bias of Large Language Models in International Relations"
  - u: "beyond_perplexity_let_the_reader_select_retrieval_summaries_via_spectrum_project/"
    t: "Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score"
  - u: "cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen/"
    t: "Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation"
  - u: "comlq_benchmarking_complex_logical_queries_in_information_retrieval/"
    t: "ComLQ: Benchmarking Complex Logical Queries in Information Retrieval"
  - u: "comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re/"
    t: "ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning"
  - u: "convmix_a_mixed-criteria_data_augmentation_framework_for_conversational_dense_re/"
    t: "ConvMix: A Mixed-Criteria Data Augmentation Framework for Conversational Dense Retrieval"
  - u: "do_retrieval_augmented_language_models_know_when_they_dont_know/"
    t: "Do Retrieval Augmented Language Models Know When They Don't Know?"
  - u: "exposing_the_cracks_vulnerabilities_of_retrieval-augmented_llm-based_machine_tra/"
    t: "Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-Based Machine Translation"
  - u: "magnitude_matters_a_superior_class_of_similarity_metrics_for_holistic_semantic_u/"
    t: "Magnitude Matters: A Superior Class of Similarity Metrics for Holistic Semantic Understanding"
  - u: "mem-pal_towards_memory-based_personalized_dialogue_assistants_for_long-term_user/"
    t: "Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction"
  - u: "n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l/"
    t: "N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs"
  - u: "opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect/"
    t: "OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval"
  - u: "precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es/"
    t: "PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation"
  - u: "prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning/"
    t: "PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning"
  - u: "ragfort_dual-path_defense_against_proprietary_knowledge_base_extraction_in_retri/"
    t: "RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation"
  - u: "reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop/"
    t: "REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering"
  - u: "refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr/"
    t: "ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting"
  - u: "rrra_resampling_and_reranking_through_a_retriever_adapter/"
    t: "RRRA: Resampling and Reranking through a Retriever Adapter"
  - u: "sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte/"
    t: "SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention"
  - u: "towards_inference-time_scaling_for_continuous_space_reasoning/"
    t: "Towards Inference-Time Scaling for Continuous Space Reasoning"
  - u: "when_small_models_are_right_for_wrong_reasons_process_verification_for_trustwort/"
    t: "When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents"
item_total: 21
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🤖 AAAI2026** · **21** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (81)](../../ICLR2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [🧪 ICML2026 (26)](../../ICML2026/information_retrieval/index.md) · [🧠 NeurIPS2025 (25)](../../NeurIPS2025/information_retrieval/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/information_retrieval/index.md) · [🧪 ICML2025 (6)](../../ICML2025/information_retrieval/index.md)

🔥 **高频主题：** RAG ×6 · 推理 ×4 · LLM ×3 · Agent ×3 · 对话系统 ×2

**["As Eastern Powers, I Will Veto." : An Investigation of Nation-Level Bias of Large Language Models in International Relations](as_eastern_powers_i_will_veto_an_investigation_of_nation-level_bias_of_large_lan.md)**

:   系统性地研究 LLM 在国际关系领域的国家级偏见，基于联合国安理会真实数据设计三种偏见测试（直接问答、关联测试、投票模拟），揭示偏见的多维性——随模型和评知上下文变化，并提出 RAG+Reflexion 去偏框架。

**[Beyond Perplexity: Let the Reader Select Retrieval Summaries via Spectrum Projection Score](beyond_perplexity_let_the_reader_select_retrieval_summaries_via_spectrum_project.md)**

:   提出 Spectrum Projection Score (SPS) 这一无需训练的指标，通过衡量摘要 token 嵌入与 reader LLM 主子空间的对齐程度来评估检索摘要质量，替代传统困惑度指标。结合 xCompress 推理时控制器，在 5 个 QA 数据集上显著优于基于困惑度的方法（HotpotQA EM +3.6）。

**[Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation](cog-rag_cognitive-inspired_dual-hypergraph_with_theme_alignment_retrieval-augmen.md)**

:   提出 Cog-RAG，用主题超图和实体超图构建双超图索引，模拟人类"自顶向下"的认知过程进行两阶段检索（先主题后细节），实现从全局语义到局部信息的对齐生成。

**[ComLQ: Benchmarking Complex Logical Queries in Information Retrieval](comlq_benchmarking_complex_logical_queries_in_information_retrieval.md)**

:   构建了首个面向复杂逻辑查询的信息检索基准 ComLQ（含合取、析取、否定等 14 种查询类型），并提出子图引导的 LLM 数据合成方法和否定一致性评估指标 LSNC，揭示现有检索器在逻辑推理尤其是否定建模上的严重不足。

**[ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning](comorag_a_cognitive-inspired_memory-organized_rag_for_stateful_long_narrative_re.md)**

:   受人脑前额叶皮层元认知调控机制启发，提出 ComoRAG 框架，通过动态记忆工作空间和迭代探测查询实现有状态的多步推理，在长篇叙事理解（200K+ tokens）任务上显著超越现有 RAG 方法。

**[ConvMix: A Mixed-Criteria Data Augmentation Framework for Conversational Dense Retrieval](convmix_a_mixed-criteria_data_augmentation_framework_for_conversational_dense_re.md)**

:   提出 ConvMix 混合准则数据增强框架，从查询和文档双方向用 LLM 进行可扩展的相关性标注增强，并通过聚类多样性选择和 Fisher 信息近分布监督筛选，系统性提升对话式稠密检索性能。

**[Do Retrieval Augmented Language Models Know When They Don't Know?](do_retrieval_augmented_language_models_know_when_they_dont_know.md)**

:   系统分析RAG模型的拒绝校准问题，发现RALM在检索文档全部不相关时过度拒绝率超过55%（即使模型内部知识足够回答），提出结合不确定性估计和拒绝感知微调的机制来平衡拒绝与回答质量。

**[Exposing the Cracks: Vulnerabilities of Retrieval-Augmented LLM-Based Machine Translation](exposing_the_cracks_vulnerabilities_of_retrieval-augmented_llm-based_machine_tra.md)**

:   开发受控噪声注入框架系统评估检索增强翻译（REAL-MT），引入Fidelity和CAR两个新指标，在10语言对×4种噪声类型上揭示模型即使面对矛盾上下文仍盲目采纳（CAR保持65-78%），大推理模型（LRM）反而更脆弱（会"合理化"错误上下文），且噪声鲁棒性与干净上下文利用率存在根本性trade-off。

**[Magnitude Matters: A Superior Class of Similarity Metrics for Holistic Semantic Understanding](magnitude_matters_a_superior_class_of_similarity_metrics_for_holistic_semantic_u.md)**

:   提出两种无参数、幅度感知的向量相似度度量——Overlap Similarity (OS) 和 Hyperbolic Tangent Similarity (HTS)，在 4 个句子嵌入模型和 8 个 NLP 基准上，对分类任务（释义、推理）的 MSE 显著低于 Cosine Similarity 和 Dot Product，且无需任何额外训练开销。

**[Mem-PAL: Towards Memory-based Personalized Dialogue Assistants for Long-term User-Agent Interaction](mem-pal_towards_memory-based_personalized_dialogue_assistants_for_long-term_user.md)**

:   提出H2Memory四层分层异构记忆结构（日志图/背景记忆/主题大纲/原则），通过PAL-Set数据集（100用户×8.4个月交互）验证，在需求重述和方案建议任务上将BLEU-1从13.59提升至26.67。

**[N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)**

:   提出 N2N-GQA——首个用于开放域混合表格-文本问答的零样本框架，核心思路是将检索到的嘈杂文档构建为动态证据图（文档为节点、TF-IDF共享词为边），通过图中心性剪枝识别"桥接文档"连接多跳推理链，在 OTT-QA 上比 Vanilla RAG 提升 +39.6 EM（从 8.0 到 48.8），零样本即接近微调系统 CORE (49.0 EM)。

**[OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)**

:   提出 OPERA 框架，通过 Goal Planning Module 和 Reason-Execute Module 的分层架构，结合专为多 agent 设计的 MAPGRPO 训练算法，大幅提升 reasoning-oriented multi-hop retrieval 性能。

**[PRECISE: Reducing the Bias of LLM Evaluations Using Prediction-Powered Ranking Estimation](precise_reducing_the_bias_of_llm_evaluations_using_prediction-powered_ranking_es.md)**

:   将Prediction-Powered Inference（PPI）框架扩展到子实例级别的排序指标（如Precision@K），通过仅30-100条人工标注+大量LLM评判结果获得无偏的排序指标估计，计算复杂度从 $O(2^{|C|})$ 降至 $O(2^K)$，在印度电商搜索场景中成功指导LLM查询改写系统上线。

**[PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)**

:   受双系统认知理论启发，提出PRIME多Agent推理框架——Quick Thinking Agent（System 1）快速生成直觉答案，Reflection Agent评估可信度，不确定时触发System 2的6个专门化Agent（规划/搜索/阅读/假设/整合/决策）进行深度知识检索推理，使开源LLaMA 3在医学/多跳QA上接近GPT-4o性能。

**[RAGFort: Dual-Path Defense Against Proprietary Knowledge Base Extraction in Retrieval-Augmented Generation](ragfort_dual-path_defense_against_proprietary_knowledge_base_extraction_in_retri.md)**

:   提出 RAGFort，首个系统性防御 RAG 知识库抽取攻击的双路径框架，通过对比重索引（inter-class）隔离主题间边界和约束级联生成（intra-class）抑制敏感内容输出，在安全性上将知识恢复率降低至无保护的 0.51×，同时保持回答质量。

**[REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)**

:   提出 REAP 双模块迭代框架，通过子任务规划器 (SP) 维护全局视角动态指导推理轨迹，事实提取器 (FE) 从检索内容中提取结构化事实和潜在线索，两者递归协作解决多跳问答。在 4 个基准上以 Llama-3.1-8B 显著超越所有基线（HotpotQA F1 68.0 vs 次优 63.4）。

**[ReFeed: Retrieval Feedback-Guided Dataset Construction for Style-Aware Query Rewriting](refeed_retrieval_feedback-guided_dataset_construction_for_style-aware_query_rewr.md)**

:   提出一个检索反馈驱动的数据集生成框架，通过识别检索失败case、LLM风格化改写、重检索验证三步闭环，自动构建高质量的风格感知查询改写数据集，为训练检索对齐的改写模型提供数据基础。

**[RRRA: Resampling and Reranking through a Retriever Adapter](rrra_resampling_and_reranking_through_a_retriever_adapter.md)**

:   提出RRRA框架，通过在Bi-Encoder上添加轻量级可学习适配器来建模每个候选文档的假阴性概率，并将其同时用于训练时的负样本重采样和推理时的重排序，在NQ/TQ/MS MARCO上持续超越SimANS/TriSampler等强基线。

**[SR-KI: Scalable and Real-Time Knowledge Integration into LLMs via Supervised Attention](sr-ki_scalable_and_real-time_knowledge_integration_into_llms_via_supervised_atte.md)**

:   提出SR-KI框架，通过两阶段训练（检索层定位 + 注意力监督损失）实现结构化知识库向LLM KV缓存的高效注入，在单块A100 40GB GPU上支持最多40K知识库条目的注入，且通过top-100压缩实现高达99.75%的压缩率，同时保持88%以上的平均Recall@10检索性能。

**[Towards Inference-Time Scaling for Continuous Space Reasoning](towards_inference-time_scaling_for_continuous_space_reasoning.md)**

:   首次系统研究离散文本推理中的inference-time scaling技术能否迁移到连续潜空间推理模型（COCONUT），发现dropout采样能生成多样推理路径（Pass@32达44.43%），但PRM/ORM仅带来不足2.3%提升，根因在于连续思维表示缺乏区分正误推理的几何归纳偏置。

**[When Small Models Are Right for Wrong Reasons: Process Verification for Trustworthy Agents](when_small_models_are_right_for_wrong_reasons_process_verification_for_trustwort.md)**

:   通过分析 10,734 条推理轨迹揭示小型语言模型（7-9B）存在严重的"答对但理由错"（RWR）现象——50-69% 的正确答案包含根本性推理缺陷；提出推理完整性评分（RIS）作为过程级指标，发现 RAG 能有效改善推理质量而元认知干预反而有害，并蒸馏出快速分类器（0.86 F1, 100× 加速）用于实时部署。
