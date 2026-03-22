<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**🧠 NeurIPS2025** · 共 **6** 篇

**[Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking](diagnosing_and_addressing_pitfalls_in_kg-rag_datasets_toward_more_reliable_bench.md)**

:   系统审计16个KGQA数据集发现平均事实正确率仅57%（WebQSP 52%，MetaQA 20%），提出KGQAGen框架——通过LLM引导的子图扩展+SPARQL自动验证构建高质量多跳QA数据集KGQAGen-10k（96.3%准确率），揭示KG-RAG的主要瓶颈在检索而非推理。

**[Geometric Imbalance In Semi-Supervised Node Classification](geometric_imbalance_in_semi-supervised_node_classification.md)**

:   在黎曼流形上形式化半监督节点分类中的几何不平衡问题，提出伪标签对齐框架，在9个基准上一致超越基线，特别在严重类别不平衡时效果显著。

**[GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)**

:   提出首个图基础模型驱动的检索增强生成框架 GFM-RAG，通过 query-dependent GNN 在知识图谱上进行单步多跳推理，仅 8M 参数即可在未见数据集上零样本泛化，在多跳QA检索任务上大幅超越 SOTA。

**[GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)**

:   提出首个图拓扑导向的 prompting 框架 GraphTOP，通过将 topology-oriented prompting 建模为边重连问题并用 Gumbel-Softmax 松弛到连续空间，在 5 个数据集 4 种预训练策略下超越 6 个基线方法。

**[Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)**

:   从理论上揭示了 GNN 消息传递中平滑性（smoothness）与泛化性（generalization）之间的两难困境，提出 IGNN 框架通过三个简约设计原则（分离邻域变换、感知聚合、邻域关系学习）缓解该困境，在 30 个基线中表现最优且具备跨同质/异质图的通用性。

**[Relieving the Over-Aggregating Effect in Graph Transformers](relieving_the_over-aggregating_effect_in_graph_transformers.md)**

:   发现了 Graph Transformer 中的 over-aggregating 现象——大量节点以近均匀注意力分数被聚合导致关键信息被稀释，提出 Wideformer 通过分割聚合+引导注意力来缓解，作为即插即用模块在 13 个数据集上一致提升骨干模型性能。
