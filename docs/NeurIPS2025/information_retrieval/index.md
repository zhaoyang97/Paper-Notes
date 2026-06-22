---
title: >-
  NeurIPS2025 信息检索/RAG论文汇总 · 25篇论文解读
description: >-
  25篇NeurIPS2025的信息检索/RAG 方向论文解读，涵盖 RAG、推理、LLM等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "NeurIPS2025"
  - "信息检索/RAG"
  - "论文解读"
  - "论文笔记"
  - "RAG"
  - "推理"
  - "LLM"
item_list:
  - u: "acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking/"
    t: "AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking"
  - u: "chain-of-retrieval_augmented_generation/"
    t: "Chain-of-Retrieval Augmented Generation (CoRAG)"
  - u: "cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa/"
    t: "Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers"
  - u: "deep_research_brings_deeper_harm/"
    t: "Deep Research Brings Deeper Harm"
  - u: "dice_discrete_interpretable_comparative_evaluation_with_probabilistic_scoring_fo/"
    t: "DICE: Discrete Interpretable Comparative Evaluation with Probabilistic Scoring for RAG"
  - u: "enginuity_building_an_open_multi-domain_dataset_of_complex_engineering_diagrams/"
    t: "Enginuity: Building an Open Multi-Domain Dataset of Complex Engineering Diagrams"
  - u: "hierarchical_retrieval_the_geometry_and_a_pretrain-finetune_recipe/"
    t: "Hierarchical Retrieval: The Geometry and a Pretrain-Finetune Recipe"
  - u: "hifi-rag_hierarchical_content_filtering_and_two-pass_generation_for_open-domain_/"
    t: "HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG"
  - u: "how_should_we_evaluate_data_deletion_in_graph-based_ann_indexes/"
    t: "How Should We Evaluate Data Deletion in Graph-Based ANN Indexes?"
  - u: "hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge/"
    t: "HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation"
  - u: "improving_consistency_in_retrieval-augmented_systems_with_group_similarity_rewar/"
    t: "Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards"
  - u: "is_prm_necessary_problem-solving_rl_implicitly_induces_prm_capability_in_llms/"
    t: "Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs"
  - u: "learning_task-agnostic_representations_through_multi-teacher_distillation/"
    t: "Learning Task-Agnostic Representations through Multi-Teacher Distillation"
  - u: "mir-bench_can_your_llm_recognize_complicated_patterns_via_many-shot_in-context_r/"
    t: "MIR-Bench: Can Your LLM Recognize Complicated Patterns via Many-Shot In-Context Reasoning?"
  - u: "mitra_an_ai_assistant_for_knowledge_retrieval_in_physics_collaborations/"
    t: "MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations"
  - u: "murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m/"
    t: "MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining"
  - u: "reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation/"
    t: "Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation"
  - u: "retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations/"
    t: "Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations"
  - u: "retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o/"
    t: "Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization"
  - u: "rmit-adms_at_the_mmu-rag_neurips_2025_competition/"
    t: "RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition"
  - u: "scaling_language-centric_omnimodal_representation_learning/"
    t: "Scaling Language-Centric Omnimodal Representation Learning"
  - u: "secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo/"
    t: "SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG"
  - u: "superclip_clip_with_simple_classification_supervision/"
    t: "SuperCLIP: CLIP with Simple Classification Supervision"
  - u: "symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r/"
    t: "SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning"
  - u: "think_straight_stop_smart_structured_reasoning_for_efficient_multi-hop_rag/"
    t: "Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG"
item_total: 25
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔍 信息检索/RAG

**🧠 NeurIPS2025** · **25** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (81)](../../ICLR2026/information_retrieval/index.md) · [💬 ACL2026 (73)](../../ACL2026/information_retrieval/index.md) · [🧪 ICML2026 (26)](../../ICML2026/information_retrieval/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/information_retrieval/index.md) · [📹 ICCV2025 (5)](../../ICCV2025/information_retrieval/index.md) · [🧪 ICML2025 (6)](../../ICML2025/information_retrieval/index.md)

🔥 **高频主题：** RAG ×12 · 推理 ×4 · LLM ×2

**[AcuRank: Uncertainty-Aware Adaptive Computation for Listwise Reranking](acurank_uncertainty-aware_adaptive_computation_for_listwise_reranking.md)**

:   利用贝叶斯TrueSkill模型维护文档相关性的概率分布，在每轮迭代中只对排名不确定的文档进行重排序，实现根据查询难度自适应调配计算量的重排框架，在多个基准上以更少调用次数超越固定计算基线。

**[Chain-of-Retrieval Augmented Generation (CoRAG)](chain-of-retrieval_augmented_generation.md)**

:   提出 CoRAG 框架，通过拒绝采样自动生成中间检索链（子查询→子答案），微调 LLM 学习迭代检索和推理，并支持多种测试时解码策略（贪心 / Best-of-N / 树搜索）灵活扩展计算量，在多跳 QA 上 EM 提升 26+ 点，KILT 基准 9/10 任务达到 SOTA。

**[Cooperative Retrieval-Augmented Generation for Question Answering: Mutual Information Exchange and Ranking by Contrasting Layers](cooperative_retrieval-augmented_generation_for_question_answering_mutual_informa.md)**

:   提出CoopRAG框架，通过问题展开、基于检索器层对比的重排、以及推理链补全，实现检索器与LLM的双向合作，在多跳QA上超越HippoRAG2 5.3%，单跳QA上提升35.2%。

**[Deep Research Brings Deeper Harm](deep_research_brings_deeper_harm.md)**

:   揭示 Deep Research (DR) 智能体的严重安全隐患——即使底层 LLM 能正确拒绝有害请求，部署为 DR 智能体后仍能生成详细专业的危险报告；提出 Plan Injection 和 Intent Hijack 两种针对性越狱方法，以及 DeepREJECT 评估指标，在 6 个 LLM 上验证了 DR 智能体系统性地削弱了对齐机制。

**[DICE: Discrete Interpretable Comparative Evaluation with Probabilistic Scoring for RAG](dice_discrete_interpretable_comparative_evaluation_with_probabilistic_scoring_fo.md)**

:   提出 DICE 框架，通过两阶段评估（证据耦合深度分析 + 概率化 {A,B,Tie} 打分）和瑞士赛制锦标赛实现 RAG 系统的可解释、鲁棒、高效评估，在中文金融 QA 数据集上达到 85.7% 人类专家一致率，远超 RAGAS（45.7%）。

**[Enginuity: Building an Open Multi-Domain Dataset of Complex Engineering Diagrams](enginuity_building_an_open_multi-domain_dataset_of_complex_engineering_diagrams.md)**

:   提出 Enginuity——首个面向 AI 自动解析工程图的大规模开放多领域数据集方案，计划构建 50K+ 带有层级组件关系、空间连接和语义角色标注的汽车工程图，通过四阶段人机协同标注管线实现高质量与低成本的平衡，并定义了从符号检测到数字孪生生成的完整任务体系，为多模态大模型理解工程图中的视觉-结构知识提供了首个系统性基准资源。

**[Hierarchical Retrieval: The Geometry and a Pretrain-Finetune Recipe](hierarchical_retrieval_the_geometry_and_a_pretrain-finetune_recipe.md)**

:   研究双编码器（Dual Encoder）在层次化检索（Hierarchical Retrieval）中的可行性，理论证明嵌入维度只需与层次深度线性、文档数对数增长即可求解，并发现"远距离丢失"现象后提出预训练-微调策略，在 WordNet 上将远距离召回率从 19% 提升至 76%。

**[HiFi-RAG: Hierarchical Content Filtering and Two-Pass Generation for Open-Domain RAG](hifi-rag_hierarchical_content_filtering_and_two-pass_generation_for_open-domain_.md)**

:   通过分离轻量级 Flash 模型的过滤能力与 Pro 模型的推理能力，构建多阶段管道（查询优化→分层过滤→两阶段生成→引文验证），在 MMU-RAGent 竞赛中实现 SOTA 性能。

**[How Should We Evaluate Data Deletion in Graph-Based ANN Indexes?](how_should_we_evaluate_data_deletion_in_graph-based_ann_indexes.md)**

:   针对图基ANN索引缺乏统一数据删除评估方法的问题，形式化定义了逻辑删除、物理删除和重建三种基准方法，提出面向实际部署的评估框架和指标体系，并基于实验分析提出Deletion Control算法在精度约束下动态切换删除策略。

**[HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)**

:   提出 HyperGraphRAG，首个基于超图 (hypergraph) 结构的 RAG 方法，通过超边 (hyperedge) 建模 n 元关系（n≥2），克服了现有图谱 RAG 方法受限于二元关系的瓶颈，在医学、农业、计算机科学和法律等领域的问答任务中全面超越 StandardRAG 和 GraphRAG 系列方法。

**[Improving Consistency in Retrieval-Augmented Systems with Group Similarity Rewards](improving_consistency_in_retrieval-augmented_systems_with_group_similarity_rewar.md)**

:   提出 Con-RAG 框架，通过 Paraphrased Set GRPO (PS-GRPO) 在语义等价查询的多次生成之间计算组相似度奖励，训练 RAG 系统的生成器在释义输入下产生信息一致的输出，无需显式真实标签监督即可同时提升一致性和准确性。

**[Is PRM Necessary? Problem-Solving RL Implicitly Induces PRM Capability in LLMs](is_prm_necessary_problem-solving_rl_implicitly_induces_prm_capability_in_llms.md)**

:   系统研究表明纯 RL 训练（无需显式 PRM 监督）能隐式诱导出强大的过程判断能力，且现有 PRM 在 DeepSeek-R1/QwQ-32B 等强推理模型上甚至不如简单多数投票有效；提出 Self-PRM 让模型用自身的内部奖励信号重排输出，一致性地优于外部 PRM。

**[Learning Task-Agnostic Representations through Multi-Teacher Distillation](learning_task-agnostic_representations_through_multi-teacher_distillation.md)**

:   提出基于互信息最大化的任务无关多教师蒸馏框架，通过高斯核估计教师嵌入的条件分布来训练学生模型，使其在不依赖任何下游任务标签的情况下学到高信息密度的通用表示，在文本、视觉和分子建模三个领域均取得了同体量最优性能。

**[MIR-Bench: Can Your LLM Recognize Complicated Patterns via Many-Shot In-Context Reasoning?](mir-bench_can_your_llm_recognize_complicated_patterns_via_many-shot_in-context_r.md)**

:   提出 MIR-Bench，首个大规模多样化的 many-shot 上下文推理基准，通过从编程题中自动生成输入输出对来测试 LLM 的模式识别能力，发现 LLM 在 many-shot 场景下存在注意力分散导致的性能饱和现象，且转导推理普遍优于归纳推理。

**[MITRA: An AI Assistant for Knowledge Retrieval in Physics Collaborations](mitra_an_ai_assistant_for_knowledge_retrieval_in_physics_collaborations.md)**

:   提出 MITRA，一个面向大型物理实验协作（如 CERN CMS）的本地化 RAG 系统，采用两层向量数据库架构（摘要库 + 全文库）和完全本地部署策略，在语义检索任务上显著优于传统关键词搜索（BM25），Precision@1 从 0.13 提升至 0.75。

**[MuRating: A High Quality Data Selecting Approach to Multilingual Large Language Model Pretraining](murating_a_high_quality_data_selecting_approach_to_multilingual_large_language_m.md)**

:   提出 MuRating，一个可扩展的多语言数据选择框架：先通过配对比较聚合多个英文数据质量评分器，再借助翻译将质量信号迁移到 17 种语言，训练出语言无关的多语言质量评估模型，在 1.2B 和 7B 规模 LLM 预训练中取得了持续的性能提升。

**[Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)**

:   提出 CalibRAG 框架，通过训练一个温度条件化的 forecasting function 来确保 RAG 辅助决策过程中的置信度校准，不仅改善校准质量还提升了准确率。

**[Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)**

:   针对无线电法规这一法律敏感的高风险领域，设计了专用 RAG 管道并构建了首个 ITU 无线电法规多选题评估集，检索准确率达 97%，在 GPT-4o 上实现 +11.9% 的问答准确率提升，远超直接将文档塞入 prompt 的方式。

**[Retrieval is Not Enough: Enhancing RAG Reasoning through Test-Time Critique and Optimization](retrieval_is_not_enough_enhancing_rag_reasoning_through_test-time_critique_and_o.md)**

:   提出 AlignRAG 框架，将 RAG 重新定义为"检索增强推理"，通过训练专用 Critic Language Model (CLM) 在测试时迭代批评和修正推理过程，解决推理与检索证据之间的错位问题，8B CLM 在 OOD 任务上超越 72B 标准 CLM。

**[RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition](rmit-adms_at_the_mmu-rag_neurips_2025_competition.md)**

:   提出Routing-to-RAG (R2RAG)系统，通过LLM查询分类器将简单查询路由到单轮Vanilla RAG、复杂查询路由到迭代式Vanilla Agent，全部基于Qwen3-4B（未量化）和Qwen3-Reranker-0.6B两个轻量模型在单块消费级GPU上运行，获NeurIPS 2025 MMU-RAG竞赛开源赛道Best Dynamic Evaluation奖。

**[Scaling Language-Centric Omnimodal Representation Learning](scaling_language-centric_omnimodal_representation_learning.md)**

:   提出 LCO-Emb 框架，发现多模态大模型（MLLM）在生成式预训练中已隐式建立跨模态对齐，仅需轻量级的纯文本对比学习微调即可激活全模态表示能力，并发现生成能力与表示性能正相关的 Generation-Representation Scaling Law (GRSL)。

**[SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG](secon-rag_a_two-stage_semantic_filtering_and_conflict-free_framework_for_trustwo.md)**

:   提出 SeCon-RAG 两阶段防御框架，第一阶段用聚类+语义图联合过滤毒化文档，第二阶段在推理时做冲突感知过滤，在5个LLM和3个QA数据集上全面超越现有RAG防御方法，在100%投毒率下仍保持高准确率和极低攻击成功率。

**[SuperCLIP: CLIP with Simple Classification Supervision](superclip_clip_with_simple_classification_supervision.md)**

:   在CLIP对比学习框架中引入一个超简单的分类损失（仅需添加一个轻量线性层，FLOPs增加仅0.077%），利用原始文本token的分类信号恢复CLIP未充分利用的细粒度文本监督，在零样本分类、图文检索和纯视觉任务上一致提升性能。

**[SymRTLO: Enhancing RTL Code Optimization with LLMs and Neuron-Inspired Symbolic Reasoning](symrtlo_enhancing_rtl_code_optimization_with_llms_and_neuron-inspired_symbolic_r.md)**

:   提出 SymRTLO，首个将LLM与符号推理集成的神经符号框架用于RTL代码优化，通过检索增强优化规则、AST模板引导代码生成和FSM符号系统，在功耗、性能和面积(PPA)上分别获得最高43.9%、62.5%和51.1%的提升。

**[Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG](think_straight_stop_smart_structured_reasoning_for_efficient_multi-hop_rag.md)**

:   提出 TSSS (Think Straight, Stop Smart) 框架，通过 (i) 基于模板的推理缓存重复前缀并锚定子查询到主问题，(ii) 基于检索器的确定性终止器在子查询重复时停止推理，在多跳 RAG 基准上实现 SOTA 准确率和竞争效率。
