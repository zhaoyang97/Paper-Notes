<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**💬 ACL2025** · 共 **10** 篇

**[Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning](beyond_completion_a_foundation_model_for_general_knowledge_graph_reasoning.md)**

:   提出 MERRY，一个统一处理 KG 内（零样本 KGC）和 KG 外（KGQA）推理任务的知识图谱基础模型，通过多视角条件消息传递 (CMP) 融合文本和结构信息，在 28 个数据集上超越现有方法。

**[Morpher: Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?](can_graph_neural_networks_learn_language.md)**

:   提出 Morpher，首个图-文多模态 prompt learning 范式——在冻结 GNN 和 LLM 参数的前提下，同时学习图 prompt 和文本 prompt + 跨模态投影器，用极弱文本监督（仅类别名几个词）将 GNN 表征对齐到 LLM 语义空间，首次实现 GNN 的 CLIP 式零样本图分类。

**[Croppable Knowledge Graph Embedding](croppable_knowledge_graph_embedding.md)**

:   提出 MED 框架训练"可裁剪"知识图谱嵌入——一次训练同时优化 64 个不同维度的子模型（共享嵌入前缀），通过互学习、进化改进和动态损失权重，各维度子模型直接裁剪使用即超越独立训练和蒸馏方法，训练速度快 10 倍。

**[Extending Complex Logical Queries on Uncertain Knowledge Graphs](extending_complex_logical_queries_uncertain_knowledge_graphs.md)**

:   提出在不确定知识图谱（Uncertain KG）上进行软查询（Soft Query）的新问题设定 SQUK，结合必要性（necessity）和重要性（importance）扩展一阶逻辑查询语义，并设计带校准的神经符号推理方法 SRC，避免前向推理中的级联错误。

**[GraphNarrator: Generating Textual Explanations for Graph Neural Networks](graphnarrator.md)**

:   提出GraphNarrator——首个为图神经网络生成自然语言解释的方法，通过将显著性图解释"语言化"为文本段落、用Expert Iteration迭代优化伪标签质量、最终蒸馏到端到端解释器模型，在三个数据集上生成的解释在忠实度、简洁性和人类偏好上均优于GPT-4o零样本解释。

**[Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)**

:   提出开放域知识图谱问答基准 OKGQA 及其扰动变体 OKGQA-P，通过统一的图引导检索-生成框架系统性地验证了 KG 增强可以有效降低 LLM 幻觉率（FActScore 提升约 20 个百分点），子图检索在各类查询上表现最优且对 KG 噪声具有鲁棒性。

**[Knowledge Graph Retrieval-Augmented Generation for LLM-based Recommendation](kg_rag_recommendation.md)**

:   提出 K-RagRec 框架，将知识图谱（KG）中的结构化关系信息引入 LLM 推荐系统的 RAG 流程——从 KG 中检索高质量的结构化实体关系信息来增强推荐生成，解决纯文本 RAG 忽略结构关系和引入噪声的问题。

**[Can LLMs Evaluate Complex Attribution in QA? Automatic Benchmarking using Knowledge Graphs](paper_2401_14640.md)**

:   提出 CAQA 基准，利用知识图谱自动生成包含四类归因类别（支持、部分支持、矛盾、无关）和四种推理复杂度的大规模问答归因评估数据集（161K 样本），系统性地评测了 25 种自动归因评估器的能力。

**[Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings](predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)**

:   提出 CondKGCP——基于谓词条件的 conformal prediction 方法用于知识图谱嵌入的不确定性量化，通过合并相似谓词增大校准集+双重校准（score+rank）减小预测集大小，在保证谓词级条件覆盖率的同时输出更紧凑的答案集，在多个KGE基准上优于5个baseline。

**[RSCF: Relation-Semantics Consistent Filter for Entity Embedding of Knowledge Graph](rscf_relationsemantics_consistent_filter_for_entity.md)**

:   提出 RSCF 插件式 KGE 方法，通过共享仿射变换 + 根植实体变换 + 归一化三特征确保"语义相似的关系产生相似的实体变换"（关系语义一致性），在距离模型和张量分解模型上均显著超越 SOTA，并从理论和实验上验证了一致性保持率。
