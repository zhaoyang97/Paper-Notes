<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**🧠 NeurIPS2025** · 共 **18** 篇

**[BLISS: Bandit Layer Importance Sampling Strategy for Efficient Training of Graph Neural Networks](bliss_bandit_layer_importance_sampling_strategy_for_efficient_training_of_graph_.md)**

:   提出 BLISS，将 GNN 的层级邻居采样建模为多臂老虎机问题，用 EXP3 算法动态调整每条边的采样概率，根据邻居对节点表示的方差贡献作为奖励信号，在 GCN 和 GAT 上维持或超越全批次训练精度。

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

**[Graphfaas Serverless Gnn Inference For Burst-Resilient Real-Time Intrusion Detec](graphfaas_serverless_gnn_inference_for_burst-resilient_real-time_intrusion_detec.md)**

:   提出GraphFaaS，基于Serverless的GNN推理架构用于突发负载下的实时入侵检测：时间局部性图构建+频率过滤+贪心图分区实现延迟降低85%、变异系数降低64%同时保持准确率。

**[GraphTOP: Graph Topology-Oriented Prompting for Graph Neural Networks](graphtop_graph_topology-oriented_prompting_for_graph_neural_networks.md)**

:   提出首个图拓扑导向的 prompting 框架 GraphTOP，通过将 topology-oriented prompting 建模为边重连问题并用 Gumbel-Softmax 松弛到连续空间，在 5 个数据集 4 种预训练策略下超越 6 个基线方法。

**[Making Classic GNNs Strong Baselines Across Varying Homophily: A Smoothness-Generalization Perspective](making_classic_gnns_strong_baselines_across_varying_homophily_a_smoothness-gener.md)**

:   从理论上揭示了 GNN 消息传递中平滑性（smoothness）与泛化性（generalization）之间的两难困境，提出 IGNN 框架通过三个简约设计原则（分离邻域变换、感知聚合、邻域关系学习）缓解该困境，在 30 个基线中表现最优且具备跨同质/异质图的通用性。

**[Moscat: Mixture of Scope Experts at Test for Generalizing Deeper GNNs](mixture_of_scope_experts_at_test_generalizing_deeper_graph_neural_networks_with_.md)**

:   通过 PAC-Bayes 界证明 GNN 深度变化导致不同同质性子群间的泛化偏好漂移，提出 Moscat——后处理注意力门控模型，在测试时自适应组合不同深度的独立训练 GNN 专家。

**[OCN: Effectively Utilizing Higher-Order Common Neighbors for Better Link Prediction](ocn_effectively_utilizing_higher-order_common_neighbors_for_better_link_predicti.md)**

:   揭示高阶公共邻居（CN）在链接预测中的冗余和过平滑问题，提出正交化（Gram-Schmidt 去除阶间线性相关）+ 归一化（除以路径数，广义资源分配启发式）解决方案，在 7 个数据集上平均提升 HR@100 7.7%，DDI 数据集上提升 13.3%。

**[Over-squashing in Spatiotemporal Graph Neural Networks](over-squashing_in_spatiotemporal_graph_neural_networks.md)**

:   首次形式化时空图神经网络(STGNN)中的 over-squashing 问题，揭示了因果卷积中反直觉的"时间远处偏好"现象（最早时间步对最终表示影响最大），并证明 time-and-space 和 time-then-space 架构在信息瓶颈上等价，为使用计算高效的 TTS 架构提供理论支持。

**[PKD: Preference-driven Knowledge Distillation for Few-shot Node Classification](preference-driven_knowledge_distillation_for_few-shot_node_classification.md)**

:   PKD 框架协同 LLM 和多 GNN 教师做文本属性图少样本节点分类——GNN 偏好节点选择器（GNS）用 KL 散度不确定性选择需要 LLM 标注的节点，节点偏好 GNN 选择器（NGS）用 RL 为每个节点匹配最优 GNN 教师，在 9 个数据集上一致 SOTA（Cornell 87% vs 基线 59-82%）。

**[Relieving the Over-Aggregating Effect in Graph Transformers](relieving_the_over-aggregating_effect_in_graph_transformers.md)**

:   发现了 Graph Transformer 中的 over-aggregating 现象——大量节点以近均匀注意力分数被聚合导致关键信息被稀释，提出 Wideformer 通过分割聚合+引导注意力来缓解，作为即插即用模块在 13 个数据集上一致提升骨干模型性能。

**[Uniedit A Unified Knowledge Editing Benchmark For Large Language Models](uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)**

:   构建UniEdit——基于25个开放域知识的统一LLM知识编辑基准，提出邻域多跳链采样(NMCS)算法评估编辑的波纹效应。

**[Unifying Text Semantics and Graph Structures for Temporal Text-attributed Graphs with LLMs](unifying_text_semantics_and_graph_structures_for_temporal_text-attributed_graphs.md)**

:   提出 Cross 框架——用 LLM 在策略采样的时间点上动态总结节点邻域的语义演变（Temporal Reasoning Chain），然后通过语义-结构协同编码器双向融合文本语义和图结构时序信息，在时序链接预测上平均 MRR 提升 24.7%，工业数据（微信）上 AUC 提升 3.7%。

**[What Expressivity Theory Misses: Message Passing Complexity for GNNs](what_expressivity_theory_misses_message_passing_complexity_for_gnns.md)**

:   批判 GNN 的二值表达力理论无法解释实际性能差异，提出 MPC——基于概率性 lossyWL 的连续、任务特定复杂度度量，与准确率的 Spearman 相关性达 -1（传统 WLC 恒为零），成功解释了 GCN+虚拟节点为何在长程任务上优于更高表达力的高阶模型。
