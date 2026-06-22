---
title: >-
  CVPR2025 图学习论文汇总 · 7篇论文解读
description: >-
  7篇CVPR2025的图学习方向论文解读，收录 Coeff-Tuning、DVHGNN、Hypergraph Vision Transformers等。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想。
tags:
  - "CVPR2025"
  - "图学习"
  - "论文解读"
  - "论文笔记"
item_list:
  - u: "coeff-tuning_a_graph_filter_subspace_view_for_tuning_attention-based_large_model/"
    t: "Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models"
  - u: "dvhgnn_multi-scale_dilated_vision_hgnn_for_efficient_vision_recognition/"
    t: "DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition"
  - u: "hypergraph_vision_transformers_images_are_more_than_nodes_more_than_edges/"
    t: "Hypergraph Vision Transformers: Images are More than Nodes, More than Edges"
  - u: "knowledge_bridger_towards_training-free_missing_modality_completion/"
    t: "Knowledge Bridger: Towards Training-Free Missing Modality Completion"
  - u: "nn-former_rethinking_graph_structure_in_neural_architecture_representation/"
    t: "NN-Former: Rethinking Graph Structure in Neural Architecture Representation"
  - u: "unbiased_video_scene_graph_generation_via_visual_and_semantic_dual_debiasing/"
    t: "Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing"
  - u: "universal_scene_graph_generation/"
    t: "Universal Scene Graph Generation"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**📷 CVPR2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (8)](../../CVPR2026/graph_learning/index.md) · [🔬 ICLR2026 (118)](../../ICLR2026/graph_learning/index.md) · [💬 ACL2026 (24)](../../ACL2026/graph_learning/index.md) · [🧪 ICML2026 (35)](../../ICML2026/graph_learning/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/graph_learning/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/graph_learning/index.md)

**[Coeff-Tuning: A Graph Filter Subspace View for Tuning Attention-Based Large Models](coeff-tuning_a_graph_filter_subspace_view_for_tuning_attention-based_large_model.md)**

:   将多头注意力重新解释为图卷积滤波器子空间，通过学习一组极小的子空间组合系数（$H \times H$ 矩阵）来线性组合预训练的注意力图，突破 softmax 造成的凸包约束从而扩展特征空间，以几乎零参数量的代价即插即用地提升各种 PEFT 方法的性能。

**[DVHGNN: Multi-Scale Dilated Vision HGNN for Efficient Vision Recognition](dvhgnn_multi-scale_dilated_vision_hgnn_for_efficient_vision_recognition.md)**

:   提出 DVHGNN，一种利用多尺度膨胀超图捕获图像 patch 间高阶相关性的视觉骨干网络，通过聚类+膨胀超图构造 (DHGC) 获取多尺度超边、动态超图卷积实现自适应特征交换，在 ImageNet-1K 上以 30.2M 参数达到 83.1% top-1 准确率，超越 ViG-S 1.0% 和 ViHGNN-S 0.6%。

**[Hypergraph Vision Transformers: Images are More than Nodes, More than Edges](hypergraph_vision_transformers_images_are_more_than_nodes_more_than_edges.md)**

:   提出HgVT，将层次化二部超图结构嵌入ViT中，通过主图像patch顶点和虚拟顶点的分离处理、动态余弦邻接构建和超边通信池三层注意力机制，无需聚类即可捕获patch间高阶语义关系，在ImageNet-1K上HgVT-Ti以7.7M参数达到76.2%准确率（超ViHGNN-Ti 1.9%），并在图像检索中达到73.23% mAP@10。

**[Knowledge Bridger: Towards Training-Free Missing Modality Completion](knowledge_bridger_towards_training-free_missing_modality_completion.md)**

:   本文提出 Knowledge Bridger，一个免训练的缺失模态补全框架，通过利用大型多模态模型（LMM）自动挖掘多模态知识、构建知识图谱来指导缺失模态的生成与排序，在通用场景和医学OOD场景下均超越了现有方法。

**[NN-Former: Rethinking Graph Structure in Neural Architecture Representation](nn-former_rethinking_graph_structure_in_neural_architecture_representation.md)**

:   NN-Former 提出混合 GNN-Transformer 架构预测器，发现现有方法忽略了"兄弟节点"（共享父/子节点）的拓扑信息，通过 Adjacency-Sibling Multihead Attention (ASMA) 和 Bidirectional Graph Isomorphism FFN (BGIFFN) 在 NAS-Bench-101/201 上 Kendall's Tau 达 0.877/0.890，延迟预测 MAPE 降低 48-64%。

**[Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing](unbiased_video_scene_graph_generation_via_visual_and_semantic_dual_debiasing.md)**

:   提出 VISA 框架，从视觉（记忆引导序列建模 MGSM 降低特征方差）和语义（迭代关系生成器 IRG 引入层次上下文减少对偏置先验的依赖）双重角度对视频场景图生成进行去偏置，在 Action Genome 等数据集上大幅提升尾部类别性能。

**[Universal Scene Graph Generation](universal_scene_graph_generation.md)**

:   本文提出 Universal Scene Graph（USG）表示及其解析器 USG-Par，通过跨模态对象关联器和文本中心场景对比学习，从任意模态组合（图像、文本、视频、3D）输入中生成统一的场景图，同时刻画模态不变和模态特有的场景语义。
