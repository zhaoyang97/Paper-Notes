---
title: >-
  ACL2025 图学习论文汇总 · 24篇论文解读
description: >-
  24篇ACL2025的图学习方向论文解读，涵盖问答、推理、LLM、RAG、图神经网络、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2025"
  - "图学习"
  - "论文解读"
  - "论文笔记"
  - "问答"
  - "推理"
  - "LLM"
  - "RAG"
  - "图神经网络"
  - "多模态"
item_list:
  - u: "a_generative_adaptive_replay_continual_learning_model_for_temporal_knowledge_gra/"
    t: "A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning"
  - u: "a_mutual_information_perspective_on_knowledge_graph_embedding/"
    t: "A Mutual Information Perspective on Knowledge Graph Embedding"
  - u: "agent_steerable_search_for_knowledge_graph_question_answering/"
    t: "Agent Steerable Search for Knowledge Graph Question Answering"
  - u: "beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning/"
    t: "Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning"
  - u: "can_graph_neural_networks_learn_language/"
    t: "Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?"
  - u: "croppable_knowledge_graph_embedding/"
    t: "Croppable Knowledge Graph Embedding"
  - u: "cross-document_contextual_coreference_resolution_in_knowledge_graphs/"
    t: "Cross-Document Contextual Coreference Resolution in Knowledge Graphs"
  - u: "disentangled_multi-span_evolutionary_network_against_temporal_knowledge_graph_re/"
    t: "Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning"
  - u: "extending_complex_logical_queries_uncertain_knowledge_graphs/"
    t: "Extending Complex Logical Queries on Uncertain Knowledge Graphs"
  - u: "fast-and-frugal_text-graph_transformers_are_effective_link_predictors/"
    t: "Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors"
  - u: "fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_/"
    t: "FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering"
  - u: "graphcheck_breaking_long-term_text_barriers_with_extracted_knowledge_graph-power/"
    t: "GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking"
  - u: "graphnarrator/"
    t: "GraphNarrator: Generating Textual Explanations for Graph Neural Networks"
  - u: "kg_llm_trustworthy_qa/"
    t: "Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering"
  - u: "kg_rag_recommendation/"
    t: "Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)"
  - u: "m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_/"
    t: "M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations"
  - u: "mrakl_multilingual_retrieval-augmented_knowledge_graph_construction_for_low-reso/"
    t: "mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages"
  - u: "multimodal_transformers_are_hierarchical_modal-wise_heterogeneous_graphs/"
    t: "Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs"
  - u: "ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled/"
    t: "Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering"
  - u: "paper_2401_14640/"
    t: "Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs"
  - u: "predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings/"
    t: "Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings"
  - u: "rscf_relationsemantics_consistent_filter_for_entity/"
    t: "RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph"
  - u: "simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme/"
    t: "SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation"
  - u: "the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que/"
    t: "The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**💬 ACL2025** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/graph_learning/index.md) · [🔬 ICLR2026 (118)](../../ICLR2026/graph_learning/index.md) · [💬 ACL2026 (24)](../../ACL2026/graph_learning/index.md) · [🧪 ICML2026 (35)](../../ICML2026/graph_learning/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/graph_learning/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/graph_learning/index.md)

🔥 **高频主题：** 问答 ×6 · 推理 ×4 · LLM ×4 · RAG ×3 · 图神经网络 ×2

**[A Generative Adaptive Replay Continual Learning Model for Temporal Knowledge Graph Reasoning](a_generative_adaptive_replay_continual_learning_model_for_temporal_knowledge_gra.md)**

:   本文提出深度生成自适应重放（DGAR）方法，利用预训练扩散模型生成历史实体分布表示、通过增强历史分布与当前分布的共同特征来缓解分布冲突，并设计逐层自适应重放机制整合历史与当前知识，在时序知识图谱推理的持续学习场景中显著缓解了灾难性遗忘问题。

**[A Mutual Information Perspective on Knowledge Graph Embedding](a_mutual_information_perspective_on_knowledge_graph_embedding.md)**

:   本文提出基于互信息最大化的知识图谱嵌入（KGE）框架，通过最大化三元组不同组成部分之间的互信息来提升实体和关系的语义表示能力，在复杂关系模式（1-N、N-1等）上取得一致性能提升。

**[Agent Steerable Search for Knowledge Graph Question Answering](agent_steerable_search_for_knowledge_graph_question_answering.md)**

:   本文提出一种基于智能体的可控知识图谱搜索框架，让LLM Agent能够根据问题类型和推理需求动态调整图搜索策略（如搜索深度、方向、剪枝规则），实现对知识图谱问答过程的精细控制。

**[Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)**

:   提出 MERRY，一个统一处理 KG 内（零样本 KGC）和 KG 外（KGQA）推理任务的知识图谱基础模型，通过多视角条件消息传递 (CMP) 融合文本和结构信息，在 28 个数据集上超越现有方法。

**[Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?](can_graph_neural_networks_learn_language.md)**

:   本文提出Morpher，一种多模态提示学习范式，在极弱文本监督（仅几个token的标签名）下，通过同时学习图提示和文本提示将预训练GNN嵌入到LLM的语义空间中，实现跨任务、跨领域的图分类迁移以及首个CLIP风格的GNN零样本分类原型。

**[Croppable Knowledge Graph Embedding](croppable_knowledge_graph_embedding.md)**

:   提出 MED 框架训练"可裁剪"知识图谱嵌入——一次训练同时优化 64 个不同维度的子模型（共享嵌入前缀），通过互学习、进化改进和动态损失权重，各维度子模型直接裁剪使用即超越独立训练和蒸馏方法，训练速度快 10 倍。

**[Cross-Document Contextual Coreference Resolution in Knowledge Graphs](cross-document_contextual_coreference_resolution_in_knowledge_graphs.md)**

:   提出基于知识图谱的跨文档共指消解方法，通过动态链接机制将文本实体提及与知识图谱节点关联，结合上下文嵌入和图消息传递推理提升跨文档实体识别的精度和召回率，在多个基准数据集上超越传统方法。

**[Disentangled Multi-span Evolutionary Network against Temporal Knowledge Graph Reasoning](disentangled_multi-span_evolutionary_network_against_temporal_knowledge_graph_re.md)**

:   提出 DiMNet，通过多跨度演化策略和跨时间解耦机制，分离节点语义的活跃/稳定特征，显著提升时序知识图谱（TKG）外推推理性能，在四个基准数据集上取得 SOTA。

**[Extending Complex Logical Queries on Uncertain Knowledge Graphs](extending_complex_logical_queries_uncertain_knowledge_graphs.md)**

:   本文提出"软查询"形式化框架，将复杂逻辑查询扩展到不确定知识图谱（带置信度值），并设计 SRC 方法结合前向推理和后向校准来高效回答软查询，理论证明误差不会灾难性级联。

**[Fast-and-Frugal Text-Graph Transformers are Effective Link Predictors](fast-and-frugal_text-graph_transformers_are_effective_link_predictors.md)**

:   提出 Fast-and-Frugal Text-Graph (FnF-TG) Transformer，通过 Transformer 的自注意力机制统一编码文本描述和图结构（ego-graph），在归纳链接预测任务上以小 BERT 模型超越了使用大 BERT+MPNN 的 SOTA，同时首次扩展到完全归纳设置（关系也可归纳）。

**[FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)**

:   提出 FiDeLiS 框架，通过 Path-RAG 预选候选集缩小搜索空间 + 演绎验证beam search (DVBS) 逐步构建并验证推理路径，在无需训练的情况下提升 LLM 在知识图谱问答中的准确性和可解释性。

**[GraphCheck: Breaking Long-Term Text Barriers with Extracted Knowledge Graph-Powered Fact-Checking](graphcheck_breaking_long-term_text_barriers_with_extracted_knowledge_graph-power.md)**

:   GraphCheck 提出一种图增强的事实核查框架，利用 LLM 从文档和声明中提取知识图谱三元组，通过 GNN 编码图结构并作为 soft prompt 注入冻结的 LLM 验证器，在单次推理调用中实现细粒度事实核查，在7个基准上平均提升 7.1%，且在医学领域表现出强泛化能力。

**[GraphNarrator: Generating Textual Explanations for Graph Neural Networks](graphnarrator.md)**

:   GraphNarrator是首个为图神经网络生成自然语言解释的方法，通过显著性图语言化生成伪标签、信息论指标驱动的专家迭代自改进、以及知识蒸馏训练端到端解释器，实现了忠实、简洁且人类友好的GNN决策解释。

**[Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)**

:   提出开放域知识图谱问答基准 OKGQA 及其扰动变体 OKGQA-P，通过统一的图引导检索-生成框架系统性地验证了 KG 增强可以有效降低 LLM 幻觉率（FActScore 提升约 20 个百分点），子图检索在各类查询上表现最优且对 KG 噪声具有鲁棒性。

**[Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation (K-RagRec)](kg_rag_recommendation.md)**

:   提出K-RagRec框架，通过从知识图谱中检索多跳子图为LLM推荐系统提供结构化、可靠的外部知识，结合基于流行度的选择性检索策略和GNN编码器，有效缓解LLM推荐中的幻觉和知识缺失问题。

**[M3HG: Multimodal, Multi-scale, and Multi-type Node Heterogeneous Graph for Emotion Cause Triplet Extraction in Conversations](m3hg_multimodal_multi-scale_and_multi-type_node_heterogeneous_graph_for_emotion_.md)**

:   提出 M3HG 模型，通过构建多模态多类型节点异构图来显式建模对话中的情感与原因上下文，并在句间和句内两个尺度上融合语义信息，实现多模态对话中情感-原因三元组的端到端提取。同时构建了首个中文多场景 MECTEC 数据集 MECAD。

**[mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages](mrakl_multilingual_retrieval-augmented_knowledge_graph_construction_for_low-reso.md)**

:   将多语言知识图谱构建（mKGC）重新定义为 QA 任务，提出基于 RAG 的 mRAKL 系统，利用非结构化单语数据作为检索源来克服低资源语言中结构化数据匮乏的困难，在 Tigrinya 和 Amharic 两种低资源语言上显著超越已有方法。

**[Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs](multimodal_transformers_are_hierarchical_modal-wise_heterogeneous_graphs.md)**

:   从图论视角证明了多模态 Transformer（MulTs）本质上是层次化模态异质图（HMHG），并基于此提出 GsiT 模型，通过 Interlaced Mask 机制实现仅 1/3 参数的 All-Modal-In-One 融合，同时性能显著超越传统 MulTs。

**[Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)**

:   提出 ORT（Ontology-Guided Reverse Thinking），利用知识图谱本体结构从目标逆向构建标签推理路径，引导知识检索，显著提升 LLM 的知识图谱问答能力。

**[Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](paper_2401_14640.md)**

:   提出 CAQA 基准，利用知识图谱自动生成包含四类归因类别（支持、部分支持、矛盾、无关）与四种推理复杂度的大规模问答归因评估数据集（161K 样本），系统评测 25 种自动归因评估器，揭示"部分支持"识别与复杂推理场景为当前评估器的核心瓶颈。

**[Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings](predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)**

:   提出 CondKGCP——基于谓词条件的 conformal prediction 方法用于知识图谱嵌入的不确定性量化，通过合并相似谓词增大校准集+双重校准（score+rank）减小预测集大小，在保证谓词级条件覆盖率的同时输出更紧凑的答案集，在多个KGE基准上优于5个baseline。

**[RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph](rscf_relationsemantics_consistent_filter_for_entity.md)**

:   提出 RSCF 插件式 KGE 方法，通过共享仿射变换 + 根植实体变换 + 归一化三特征确保"语义相似的关系产生相似的实体变换"（关系语义一致性），在距离模型和张量分解模型上均显著超越 SOTA，并从理论和实验上验证了一致性保持率。

**[SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation](simgrag_leveraging_similar_subgraphs_for_knowledge_graphs_driven_retrieval-augme.md)**

:   提出 SimGRAG 方法，通过"查询→模式图→子图"两阶段对齐策略，利用 LLM 将查询转化为图模式，再用图语义距离（GSD）度量在知识图谱中高效检索语义最相似的子图，实现即插即用的 KG 驱动 RAG，在问答和事实验证任务上超越所有现有方法。

**[The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)**

:   本文系统性地诊断了小语言模型（SLM，0.5B–8B）在 Think-on-Graph 知识图谱问答框架中失效的根因——**探索阶段（exploration）而非推理阶段是性能瓶颈**，并证明用零样本、即插即用的轻量级段落检索模块（SentenceBERT/GTR，仅~110M参数）替代 SLM 进行 KG 遍历，即可在 CWQ 和 WebQSP 两个基准上带来一致且显著的 EM 提升。
