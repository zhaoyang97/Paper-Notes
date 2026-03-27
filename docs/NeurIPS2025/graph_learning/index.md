<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**🧠 NeurIPS2025** · 共 **11** 篇

**[BLISS: Bandit Layer Importance Sampling Strategy for Efficient Training of Graph Neural Networks](bliss_bandit_layer_importance_sampling_strategy_for_efficient_training_of_graph_.md)**

:   将GNN层级采样建模为多臂赌博机问题，通过EXP3算法动态更新边的采样分布以追踪训练过程中节点重要性的变化，在Citeseer上比PLADIES提升2.3% F1，在Yelp上提升2.7%，特别适合异质性图上的GAT。

**[Deliberation on Priors: Trustworthy Reasoning of LLMs on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)**

:   提出Deliberation over Priors（DP）框架，通过渐进式知识蒸馏（SFT+KTO偏好优化）提升关系路径生成的忠实度，并通过约束引导的内省-回溯机制保障可靠性，在ComplexWebQuestions上H@1提升13%且LLM调用次数减少77%。

**[Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking](diagnosing_and_addressing_pitfalls_in_kg-rag_datasets_toward_more_reliable_bench.md)**

:   系统审计16个KGQA数据集发现平均事实正确率仅57%（WebQSP 52%，MetaQA 20%），提出KGQAGen框架——通过LLM引导的子图扩展+SPARQL自动验证构建高质量多跳QA数据集KGQAGen-10k（96.3%准确率），揭示KG-RAG的主要瓶颈在检索而非推理。

**[Disentangling Hyperedges through the Lens of Category Theory](disentangling_hyperedges_through_the_lens_of_category_theory.md)**

:   首次用范畴论分析超边解耦，导出"因子表示一致性"自然性标准（聚合后解耦 vs 解耦后聚合应一致），提出Natural-HNN模型在癌症分型任务上比最佳baseline提升4.7%（BRCA F1从75.7%到80.4%），并100%正确捕获功能通路上下文。

**[Geometric Imbalance In Semi-Supervised Node Classification](geometric_imbalance_in_semi-supervised_node_classification.md)**

:   在黎曼流形上形式化半监督节点分类中的几何不平衡问题，提出伪标签对齐框架，在9个基准上一致超越基线，特别在严重类别不平衡时效果显著。

**[GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)**

:   提出首个图基础模型驱动的检索增强生成框架 GFM-RAG，通过 query-dependent GNN 在知识图谱上进行单步多跳推理，仅 8M 参数即可在未见数据集上零样本泛化，在多跳QA检索任务上大幅超越 SOTA。

**[Graph Persistence goes Spectral](graph_persistence_goes_spectral.md)**

:   提出 SpectRe——将图拉普拉斯谱信息融入持续同调（PH）图的新拓扑描述符，证明其表达力严格强于 PH 和谱信息单独使用，建立了局部稳定性理论，在合成和真实数据集上提升 GNN 的图分类能力。

**[GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)**

:   提出首个图拓扑导向的 prompting 框架 GraphTOP，通过将 topology-oriented prompting 建模为边重连问题并用 Gumbel-Softmax 松弛到连续空间，在 5 个数据集 4 种预训练策略下超越 6 个基线方法。

**[Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)**

:   从理论上揭示了 GNN 消息传递中平滑性（smoothness）与泛化性（generalization）之间的两难困境，提出 IGNN 框架通过三个简约设计原则（分离邻域变换、感知聚合、邻域关系学习）缓解该困境，在 30 个基线中表现最优且具备跨同质/异质图的通用性。

**[Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)**

:   首次形式化时空图神经网络(STGNN)中的 over-squashing 问题，揭示了因果卷积中反直觉的"时间远处偏好"现象（最早时间步对最终表示影响最大），并证明 time-and-space 和 time-then-space 架构在信息瓶颈上等价，为使用计算高效的 TTS 架构提供理论支持。

**[Relieving the Over-Aggregating Effect in Graph Transformers](relieving_the_over-aggregating_effect_in_graph_transformers.md)**

:   发现了 Graph Transformer 中的 over-aggregating 现象——大量节点以近均匀注意力分数被聚合导致关键信息被稀释，提出 Wideformer 通过分割聚合+引导注意力来缓解，作为即插即用模块在 13 个数据集上一致提升骨干模型性能。
