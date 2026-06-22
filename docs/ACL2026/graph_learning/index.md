---
title: >-
  ACL2026 图学习论文汇总 · 24篇论文解读
description: >-
  24篇ACL2026的图学习方向论文解读，涵盖 RAG、推理、Agent等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ACL2026"
  - "图学习"
  - "论文解读"
  - "论文笔记"
  - "RAG"
  - "推理"
  - "Agent"
item_list:
  - u: "agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning/"
    t: "AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning"
  - u: "ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning/"
    t: "ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning"
  - u: "autonomous_knowledge_graph_exploration_with_adaptive_breadth-depth_retrieval/"
    t: "Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval"
  - u: "autopkg_an_automated_framework_for_dynamic_e-commerce_product-attribute_knowledg/"
    t: "AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction"
  - u: "cog_controllable_graph_reasoning_via_relational_blueprints_and_failure-aware_ref/"
    t: "CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs"
  - u: "collaboration_of_fusion_and_independence_hypercomplex-driven_robust_multi-modal_/"
    t: "Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion"
  - u: "comparing_human_and_large_language_model_interpretation_of_implicit_information/"
    t: "Comparing Human and Large Language Model Interpretation of Implicit Information"
  - u: "compliancenlp_knowledge-graph-augmented_rag_for_multi-framework_regulatory_gap_d/"
    t: "ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection"
  - u: "craftqa_a_code-driven_adaptive_framework_for_complex_structured_data_reasoning/"
    t: "CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning"
  - u: "ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment/"
    t: "EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment"
  - u: "evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks/"
    t: "Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks"
  - u: "from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co/"
    t: "From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context"
  - u: "graph-based_alternatives_to_llms_for_human_simulation/"
    t: "Graph-Based Alternatives to LLMs for Human Simulation"
  - u: "gs-quant_granular_semantic_and_generative_structural_quantization_for_knowledge_/"
    t: "GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion"
  - u: "industryasseteqa_a_neurosymbolic_operational_intelligence_system_for_embodied_qu/"
    t: "IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance"
  - u: "legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega/"
    t: "LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning"
  - u: "llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp/"
    t: "LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs"
  - u: "logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval/"
    t: "LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval"
  - u: "megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation/"
    t: "MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation"
  - u: "overcoming_the_impedance_mismatch_a_theoretical_roadmap_for_fusing_foundation_mo/"
    t: "Overcoming the Impedance Mismatch: A Theoretical Roadmap for Fusing Foundation Models and Knowledge Graphs"
  - u: "stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug/"
    t: "STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation"
  - u: "tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation/"
    t: "TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation"
  - u: "what_makes_ai_research_replicable_executable_knowledge_graphs_as_scientific_know/"
    t: "What Makes AI Research Replicable? Executable Knowledge Graphs as Scientific Knowledge Representations"
  - u: "which_bird_does_not_have_wings_negative-constrained_kgqa_with_schema-guided_sema/"
    t: "Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement"
item_total: 24
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**💬 ACL2026** · **24** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/graph_learning/index.md) · [🔬 ICLR2026 (118)](../../ICLR2026/graph_learning/index.md) · [🧪 ICML2026 (35)](../../ICML2026/graph_learning/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/graph_learning/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/graph_learning/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/graph_learning/index.md)

🔥 **高频主题：** RAG ×5 · 推理 ×4 · Agent ×2

**[AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)**

:   提出 AgentGL，首个基于强化学习的智能体图学习（AGL）框架，让 LLM 智能体通过图原生搜索工具自主导航文本属性图（TAG），在节点分类和链接预测任务上分别实现最高 17.5% 和 28.4% 的绝对准确率提升。

**[ARK: Answer-Centric Retriever Tuning via KG-augmented Curriculum Learning](ark_answer-centric_retriever_tuning_via_kg-augmented_curriculum_learning.md)**

:   提出ARK框架，通过三维答案充分性评分（Forward+Backward+Retriever对齐）筛选正样本，利用LLM构建的知识图谱生成渐进难度的困难负样本进行课程对比学习，在10个数据集上平均提升14.5% F1。

**[Autonomous Knowledge Graph Exploration with Adaptive Breadth-Depth Retrieval](autonomous_knowledge_graph_exploration_with_adaptive_breadth-depth_retrieval.md)**

:   本文提出 ARK：一个 training-free 的知识图谱检索 agent，只暴露「全局词法搜索」和「单跳邻居展开」两个最小工具，让 LLM 自主在广度和深度之间切换，无需种子节点或固定跳数；在 STaRK 三图上把 Hit@1 平均推到 59.1%，最高比 training-free baseline 提升 31.4%，并可把策略 label-free 蒸馏进 Qwen3-8B。

**[AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction](autopkg_an_automated_framework_for_dynamic_e-commerce_product-attribute_knowledg.md)**

:   提出 AutoPKG，一个多智能体 LLM 框架，从多模态电商商品内容自动构建 Product-Attribute 知识图谱（PKG），通过类型归纳 Agent、属性键发现 Agent、属性值提取 Agent 和集中式 KGD 决策 Agent 实现动态本体的持续演化和规范化，在 Lazada 数据集上取得 0.953 WKE（类型）和 0.724 WKE（属性键），线上 A/B 测试推荐 GMV 提升 7.89%。

**[CoG: Controllable Graph Reasoning via Relational Blueprints and Failure-Aware Refinement over Knowledge Graphs](cog_controllable_graph_reasoning_via_relational_blueprints_and_failure-aware_ref.md)**

:   CoG 是一个 training-free 的 KGQA 框架，把 Kahneman 的 Dual-Process Theory 落到 KG 推理上：System 1 离线把训练集 SPARQL 蒸馏成"关系蓝图"模板库，在线作为软结构约束指导 candidate relation 的 rerank 与剪枝；System 2 在搜索停滞时触发证据条件反思和定向回溯，纠正前期错误决策；在 CWQ / WebQSP / GrailQA 三个多跳 KGQA 基准上同时拿到 SOTA 准确率（GPT-4 backbone CWQ 77.8、WebQSP 89.7、GrailQA 86.4）和最低成本（CWQ 比 PoG 少 13% token、少 12% call）。

**[Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion](collaboration_of_fusion_and_independence_hypercomplex-driven_robust_multi-modal_.md)**

:   M-Hyper 把多模态知识图谱实体编码为双四元数（biquaternion）的四个正交基，分别承载结构 / 视觉 / 文本三个独立模态以及一个融合模态，通过 Hamilton 乘积同时实现"模态独立保留"和"成对充分交互"，在 DB15K / MKG-W / MKG-Y 三个数据集上以最低显存、最短训练时间打败 18 个 baseline。

**[Comparing Human and Large Language Model Interpretation of Implicit Information](comparing_human_and_large_language_model_interpretation_of_implicit_information.md)**

:   本文提出隐含信息提取（IIE）任务和基于 LLM 的三阶段提取管道（信息提取→推理验证→时序分析），构建结构化知识图谱来表示文本的隐含含义，并通过众包人类判断对比发现 LLM 在社交丰富语境中比人类更保守，但在短事实语境中人类更保守。

**[ComplianceNLP: Knowledge-Graph-Augmented RAG for Multi-Framework Regulatory Gap Detection](compliancenlp_knowledge-graph-augmented_rag_for_multi-framework_regulatory_gap_d.md)**

:   ComplianceNLP 是一个端到端的金融监管合规系统，把 12,847 条 SEC / MiFID II / Basel III 法规构造成知识图谱来增强 RAG 检索，配合 LEGAL-BERT 的多任务义务抽取和门槛打分的差距分析，在 RegObligation / GapBench 上以 87.7 F1 击败 GPT-4o+RAG 3.5 个点，并通过领域知识蒸馏 + Medusa 推测解码实现 $2.8\times$ 推理加速；4 个月并行运行处理了 9,847 条更新，达到 96.0% 召回率和 3.1× 分析师效率提升。

**[CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning](craftqa_a_code-driven_adaptive_framework_for_complex_structured_data_reasoning.md)**

:   CRAFTQA 用 CodeSTEP 生成可执行的逐步 Python 推理代码，并在预定义操作不够时由 CRAFT 动态生成自定义函数，从而显著提升表格、知识图谱和时序知识图谱上的复杂结构化数据问答能力，GPT-4o 版本在复杂推理 Overall 上达到 76.6%。

**[EA-Agent: A Structured Multi-Step Reasoning Agent for Entity Alignment](ea-agent_a_structured_multi-step_reasoning_agent_for_entity_alignment.md)**

:   提出 EA-Agent，将实体对齐（EA）分解为结构化多步推理过程，通过工具池（三元组选择器+对齐工具+反思器）的规划和执行实现可解释的对齐决策，配合奖励引导的离线策略优化持续改进规划能力，在 DBP15K 上 Hits@1 提升高达 3.17%，同时减少冗余三元组带来的效率问题。

**[Evaluating LLMs on Large-Scale Graph Property Estimation via Random Walks](evaluating_llms_on_large-scale_graph_property_estimation_via_random_walks.md)**

:   现有 LLM 图推理 benchmark 只用 20–50 节点的小图、还要求全图可见；本文用「随机游走统计量」把最多 2.39M 节点的真实图压成 prompt，提出 EstGraph 评估 LLM 在节点/边数、社区数、图结构、影响节点 4 项估计任务上的表现，发现 LLM 在中等规模图上可达 < 20% 相对误差并能识别图结构。

**[From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)**

:   本文提出 Gspell，一个轻量级后验解释框架，通过将 GNN 节点嵌入投影到 LLM 嵌入空间并构建混合提示（软提示+文本），使 LLM 能够直接推理 GNN 内部表示并生成自然语言解释和解释子图，在文本属性图（TAG）上实现了忠实性与可解释性的良好平衡。

**[Graph-Based Alternatives to LLMs for Human Simulation](graph-based_alternatives_to_llms_for_human_simulation.md)**

:   本文提出 GEMS（Graph-basEd Models for Human Simulation），将封闭式人类行为模拟任务建模为异构图上的链接预测问题，在三个数据集和三种评估设定下匹配或超越强 LLM 基线方法，同时参数量减少 3 个数量级。

**[GS-Quant: Granular Semantic and Generative Structural Quantization for Knowledge Graph Completion](gs-quant_granular_semantic_and_generative_structural_quantization_for_knowledge_.md)**

:   GS-Quant 把 KG 实体量化成"由粗到细"的离散 code 序列——用层次聚类树约束 RQ-VAE 让浅层 code 编码全局类别（如 "Person"）、深层 code 编码细粒度属性（如 "Artist"），再用 GPT-style decoder 重构 entity + ancestor 强制 code 之间产生因果依赖，最后把这些 code 加到 LLM 词表里做 LoRA 微调，在 WN18RR / FB15k-237 上 Hits@1 比 SOTA SSQR 高 2.2-2.4 个点。

**[IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance](industryasseteqa_a_neurosymbolic_operational_intelligence_system_for_embodied_qu.md)**

:   本文把工业资产维护问答重新建模成"具身决策"任务，提出由 episode 化遥测、FMEA 知识图谱、参数化反事实风险模拟器、provenance 校验和安全门组成的神经符号系统 IndustryAssetEQA，在 4 个工业数据集上把结构有效性、反事实方向准确率、解释蕴含率分别提升最多 0.51 / 0.47 / 0.64，并把专家判定的严重过度断言从 28% 压到 2%。

**[LegalGraphRAG: Multi-Agent Graph Retrieval-Augmented Generation for Reliable Legal Reasoning](legalgraphrag_multi-agent_graph_retrieval-augmented_generation_for_reliable_lega.md)**

:   LegalGraphRAG 用事实图、 ontology 图和规则图构成层级法律图谱，再用 Researcher-Auditor-Adjudicator 三代理流程完成检索、验证和裁决，在法律判决生成中提升准确性与证据可追溯性。

**[LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs](llms_underperform_graph-based_parsers_on_supervised_relation_extraction_for_comp.md)**

:   本文在六个关系抽取数据集上对比四个 LLM（7B-70B）和一个轻量级图解析器（124M参数），发现当文档的关系图平均边数超过约 18 条时，图解析器持续且显著优于 LLM，在最复杂的 ERFGC 数据集上 F1 差距达 13.2 个点，揭示了 LLM 在复杂语言图结构抽取上的根本局限。

**[LogosKG: Hardware-Optimized Scalable and Interpretable Knowledge Graph Retrieval](logoskg_hardware-optimized_scalable_and_interpretable_knowledge_graph_retrieval.md)**

:   本文提出 LogosKG，一个硬件对齐的知识图谱检索框架，通过将图遍历转化为三元稀疏矩阵（SUB/OBJ/REL）的乘法运算，配合度感知图分区、跨图路由和按需缓存，在单设备上实现了对十亿边规模 KG 的可扩展、可解释高跳检索，并通过下游 KG-LLM 交互实验揭示了图拓扑结构对 LLM 诊断推理的影响。

**[MegaRAG: Multimodal Knowledge Graph-Based Retrieval Augmented Generation](megarag_multimodal_knowledge_graph-based_retrieval_augmented_generation.md)**

:   MegaRAG 利用 MLLM 在长文档每一页并行抽取实体-关系并合并为多模态知识图谱（MMKG），再用"子图引导"的二轮 refinement 补全跨模态、跨页关系，配合双路检索与两阶段答案生成显著优于 GraphRAG/LightRAG/VisRAG，在 SlideVQA(2k) 上准确率 64.85%（baseline 最高 27.66%）。

**[Overcoming the Impedance Mismatch: A Theoretical Roadmap for Fusing Foundation Models and Knowledge Graphs](overcoming_the_impedance_mismatch_a_theoretical_roadmap_for_fusing_foundation_mo.md)**

:   这是一篇纯理论的立场论文：作者把"基础模型（连续概率空间）与知识图谱（离散确定结构）难以真正融合"这一现象形式化为 **Impedance Mismatch（阻抗失配）**，用度量嵌入理论证明了从词法注入到注意力级整合的三类主流方案各自的数学失败上界，并给出一条"涌现—注入—编辑"的知识全生命周期理论路线图。

**[STEM: Structure-Tracing Evidence Mining for Knowledge Graphs-Driven Retrieval-Augmented Generation](stem_structure-tracing_evidence_mining_for_knowledge_graphs-driven_retrieval-aug.md)**

:   STEM 将知识图谱多跳问答从逐步路径搜索改写为“先生成查询结构图、再按结构追踪证据子图”的问题，通过语义到结构投影、Triple-GNN 全局引导和结构匹配检索，在 WebQSP 与 CWQ 上显著提升 KG-RAG 的答案准确率和证据覆盖率。

**[TagRAG: Tag-guided Hierarchical Knowledge Graph Retrieval-Augmented Generation](tagrag_tag-guided_hierarchical_knowledge_graph_retrieval-augmented_generation.md)**

:   TagRAG 用“对象标签 + 领域标签链”替代 GraphRAG 中昂贵的实体社区划分和全图摘要，在保持全局知识整合能力的同时显著降低构图与检索成本，并在 UltraDomain 四个领域上以小模型 Qwen3-4B 获得高于 NaiveRAG、GraphRAG、LightRAG 和 MiniRAG 的胜率。

**[What Makes AI Research Replicable? Executable Knowledge Graphs as Scientific Knowledge Representations](what_makes_ai_research_replicable_executable_knowledge_graphs_as_scientific_know.md)**

:   本文提出 Executable Knowledge Graphs (xKG)，把论文中的技术概念和可运行代码片段组织成 Paper- Technique- Code 三层图结构，作为可插拔知识库辅助科研复现 agent，在 PaperBench Code-Dev 上为不同 agent 带来最高 10.90 个百分点的复制得分提升。

**[Which bird does not have wings: Negative-constrained KGQA with Schema-guided Semantic Matching and Self-directed Refinement](which_bird_does_not_have_wings_negative-constrained_kgqa_with_schema-guided_sema.md)**

:   本文提出了否定约束知识图谱问答（NEST KGQA）新任务和 NestKGQA 数据集，设计了 Python 格式逻辑形式 PyLF 来清晰表达否定约束，并提出 CUCKOO 框架通过约束感知草稿生成、Schema 引导语义匹配和自导向细化三个模块，在 few-shot 设置下实现了多约束问题的高效精确回答。
