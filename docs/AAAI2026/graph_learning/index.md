<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**🤖 AAAI2026** · 共 **7** 篇

**[Adaptive Riemannian Graph Neural Networks](adaptive_riemannian_graph_neural_networks.md)**

:   提出 ARGNN 框架，为图上每个节点学习一个连续的、各向异性的对角黎曼度量张量，从而自适应地捕获图中不同区域（层级结构 vs 密集社区）的局部几何特性，统一并超越了固定曲率和离散混合曲率的几何 GNN 方法。

**[Are Graph Transformers Necessary? Efficient Long-Range Message Passing with Fractal Nodes in MPNNs](are_graph_transformers_necessary_efficient_long-range_messag.md)**

:   提出分形节点（Fractal Nodes）增强 MPNN 的长距离消息传递：通过 METIS 图划分生成子图级聚合节点，结合低通+高通滤波器（LPF+HPF）与可学习频率参数 $\omega$，使用 MLP-Mixer 实跨子图通信，在保持 $O(L(|V|+|E|))$ 线性复杂度的同时达到甚至超越图 Transformer 的性能，获 AAAI Oral。

**[Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation](assemble_your_crew_automatic_multi-agent_communication_topol.md)**

:   提出 ARG-Designer，将多 Agent 系统的拓扑设计重新定义为条件自回归图生成任务，从零开始逐步生成 Agent 节点和通信边（而非从模板图剪枝），在6个基准上达到 SOTA（平均 92.78%），同时 Token 消耗比 G-Designer 降低约 50%，且支持无需重训练的角色扩展。

**[Posterior Label Smoothing for Node Classification](posterior_label_smoothing_for_node_classification.md)**

:   提出PosteL（Posterior Label Smoothing），通过贝叶斯后验分布从邻域标签中推导soft label用于节点分类，自然适应同质图和异质图，在8种backbone×10个数据集的80个组合中76个取得精度提升。

**[Relink: Constructing Query-Driven Evidence Graph On-the-Fly for GraphRAG](relink_constructing_query-driven_evidence_graph_on-the-fly_for_graphrag.md)**

:   提出从"先构建再推理"到"边推理边构建"的GraphRAG范式转变，通过Relink框架动态构建查询特定的证据图——结合高精度KG骨架和高召回潜在关系池，用查询驱动的排序器统一评估、按需补全缺失路径并过滤干扰事实——在5个多跳QA基准上平均提升EM 5.4%和F1 5.2%。

**[RFKG-CoT: Relation-Driven Adaptive Hop-count Selection and Few-Shot Path Guidance for Knowledge-Aware QA](rfkg-cot_relation-driven_adaptive_hop-count_selection_and_few-shot_path_guidance.md)**

:   提出RFKG-CoT，通过关系驱动的自适应跳数选择（利用KG关系激活掩码动态调整推理步数）和Few-Shot路径引导（Question-Paths-Answer格式的in-context示例），在4个KGQA基准上显著提升LLM的知识图谱推理能力，GPT-4在WebQSP上达91.5%（+6.6pp），Llama2-7B提升幅度最大达+14.7pp。

**[S-DAG: A Subject-Based Directed Acyclic Graph for Multi-Agent Heterogeneous Reasoning](s-dag_a_subject-based_directed_acyclic_graph_for_multi-agent.md)**

:   提出 S-DAG，通过 GNN 从问题中识别相关学科及其依赖关系构建有向无环图，将学科节点匹配到最擅长的专家 LLM（14 个 7-13B 领域模型），按 DAG 拓扑顺序协作推理（支撑学科→主导学科），用小模型池超越 GPT-4o-mini（59.73 vs 58.52）且接近 72B 模型。
