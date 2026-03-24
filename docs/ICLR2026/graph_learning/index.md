<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**🔬 ICLR2026** · 共 **9** 篇

**[A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)**

:   从图 Ricci 曲率的几何视角证明随机 k-SAT 问题的二部图表示具有固有的负曲率，且曲率随问题难度增加而下降，建立了 GNN 过压缩 (oversquashing) 与 SAT 求解困难之间的理论联系，并通过测试时图重布线验证了该理论。

**[Are We Measuring Oversmoothing in Graph Neural Networks Correctly?](are_we_measuring_oversmoothing_in_graph_neural_networks_correctly.md)**

:   指出广泛使用的Dirichlet energy指标无法在实际场景中正确捕获GNN过平滑现象，提出以特征表征的数值秩/有效秩（effective rank）作为替代度量，实验表明Erank与准确率的平均相关性达0.91（vs Dirichlet energy的0.72），在OGB-Arxiv上Dirichlet energy甚至呈现错误的相关方向，并从理论上证明对广泛的GNN架构族其数值秩收敛到1（秩坍塌），重新定义过平滑为秩坍塌而非特征向量对齐。

**[Beyond Simple Graphs: Neural Multi-Objective Routing on Multigraphs](beyond_simple_graphs_neural_multi-objective_routing_on_multigraphs.md)**

:   首次提出针对多重图（multigraph）的神经组合优化路由方法 GMS，包含直接在多重图上边级自回归构造的 GMS-EB 和先学习剪枝再节点级路由的双头 GMS-DH 两个变体，在非对称多目标 TSP 和 CVRP 上实现了接近精确求解器 LKH 的性能且速度快数十倍。

**[Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)**

:   通过在合成关系知识图谱上从头训练 Transformer，发现适当正则化会使模型隐层涌现出双线性关系结构（bilinear relational structure），该结构不仅能克服逆向诅咒（reversal curse），还能实现编辑单个事实后逻辑一致地传播到相关事实。

**[Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization](embodied_agents_meet_personalization_investigating_challenges_and_solutions_thro.md)**

:   提出 Memento 评估框架，系统揭示 LLM 具身智能体在个性化辅助任务中的记忆利用瓶颈（信息过载、多记忆协调失败），并设计层次化知识图谱用户画像记忆模块显著改善性能。

**[GRAPHITE: Graph Homophily Booster — Reimagining the Role of Discrete Features in Heterophilic Graph Learning](graph_homophily_booster_reimagining_the_role_of_discrete_features_in_heterophili.md)**

:   提出 GRAPHITE，一种通过引入"特征节点"作为 hub 间接连接相似特征节点来**直接提升图同质性**的非学习图变换方法，首次从"改变图结构"而非"改变 GNN 架构"的角度解决异质图问题，在 Actor 等多个困难基准上显著超越 27 种 SOTA 方法。

**[Graph Tokenization for Bridging Graphs and Transformers](graph_tokenization_for_bridging_graphs_and_transformers.md)**

:   提出 GraphTokenizer 框架，将图通过可逆的频率引导序列化转换为符号序列，再用 BPE 学习图子结构词汇表，使标准 Transformer（如 BERT/GTE）无需任何架构修改即可直接处理图数据，在 14 个 benchmark 上达到 SOTA。

**[Pairwise is Not Enough: Hypergraph Neural Networks for Multi-Agent Pathfinding](pairwise_is_not_enough_hypergraph_neural_networks_for_multi-agent_pathfinding.md)**

:   提出 HMAGAT，用有向超图注意力网络替代 GNN 的成对消息传递来建模多智能体路径规划中的群体交互，仅用 1M 参数和 1% 训练数据即超越 85M 参数的 SOTA 模型。

**[Structurally Human, Semantically Biased: Detecting LLM-Generated References with Embeddings and GNNs](structurally_human_semantically_biased_detecting_llm-generated_references_with_e.md)**

:   发现 LLM 生成的参考文献在图拓扑结构上与人类引用高度相似（图结构分类器仅 0.61 准确率），但在语义嵌入上可被检测（GNN+嵌入达 93% 准确率），揭示了 LLM 的"结构模仿、语义偏向"特性。
