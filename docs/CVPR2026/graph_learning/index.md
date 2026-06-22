---
title: >-
  CVPR2026 图学习论文汇总 · 8篇论文解读
description: >-
  8篇CVPR2026的图学习方向论文解读，涵盖多模态、压缩/编码、图神经网络、RAG、LLM、推理等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "CVPR2026"
  - "图学习"
  - "论文解读"
  - "论文笔记"
  - "多模态"
  - "压缩/编码"
  - "图神经网络"
  - "RAG"
  - "LLM"
  - "推理"
item_list:
  - u: "adaptive_learned_image_compression_with_graph_neural_networks/"
    t: "Adaptive Learned Image Compression with Graph Neural Networks"
  - u: "graph2eval_automatic_multimodal_task_generation_for_agents_via_knowledge_graphs/"
    t: "Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs"
  - u: "m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera/"
    t: "M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation"
  - u: "mario_multimodal_graph_reasoning_with_large_language_models/"
    t: "Mario: Multimodal Graph Reasoning with Large Language Models"
  - u: "mixture-of-experts_based_feature_decoupling_for_open_vocabulary_scene_graph_gene/"
    t: "Mixture-of-Experts based Feature Decoupling for Open Vocabulary Scene Graph Generation"
  - u: "r2g_multi_view_circuit_graph_benchmark_suite_from_rtl_to_gdsii/"
    t: "R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII"
  - u: "robo-sgg_exploiting_layout-oriented_normalization_and_restitution_can_improve_ro/"
    t: "Robo-SGG: Exploiting Layout-Oriented Normalization and Restitution Can Improve Robust Scene Graph Generation"
  - u: "viterbiplannet_injecting_procedural_knowledge_via_differentiable_viterbi_for_pla/"
    t: "ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning"
item_total: 8
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🕸️ 图学习

**📷 CVPR2026** · **8** 篇论文解读

📌 **同领域跨会议浏览：** [🔬 ICLR2026 (118)](../../ICLR2026/graph_learning/index.md) · [💬 ACL2026 (24)](../../ACL2026/graph_learning/index.md) · [🧪 ICML2026 (35)](../../ICML2026/graph_learning/index.md) · [🤖 AAAI2026 (37)](../../AAAI2026/graph_learning/index.md) · [🧠 NeurIPS2025 (54)](../../NeurIPS2025/graph_learning/index.md) · [📹 ICCV2025 (1)](../../ICCV2025/graph_learning/index.md)

🔥 **高频主题：** 多模态 ×3

**[Adaptive Learned Image Compression with Graph Neural Networks](adaptive_learned_image_compression_with_graph_neural_networks.md)**

:   GLIC 把学习图像压缩里的非线性变换从固定卷积或窗口注意力，改造成由图神经网络驱动的内容自适应连接：先用双尺度图决定“连到哪里”，再用复杂度感知机制决定“连多少”，从而更好地建模局部与远程冗余，在三个标准数据集上都显著超过传统编解码器和近期 LIC 强基线。

**[Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](graph2eval_automatic_multimodal_task_generation_for_agents_via_knowledge_graphs.md)**

:   提出 Graph2Eval，一个知识图谱驱动的 agent 评估任务自动生成框架——通过从文档/网页构建结构化知识图谱、子图采样、LLM 条件生成和多阶段过滤，自动产出语义一致（+20%）且可解（+17%）的多模态 agent 任务，构建了包含 1319 个任务的 Graph2Eval-Bench。

**[M3KG-RAG: Multi-hop Multimodal Knowledge Graph-enhanced Retrieval-Augmented Generation](m3kg_rag_multi_hop_multimodal_knowledge_graph_enhanced_retrieval_augmented_genera.md)**

:   提出M3KG-RAG，通过轻量多Agent流水线构建多跳多模态知识图谱（M3KG），并设计GRASP机制进行实体定位和选择性剪枝，仅保留查询相关且有助回答的知识，大幅提升MLLM的音视觉推理能力。

**[Mario: Multimodal Graph Reasoning with Large Language Models](mario_multimodal_graph_reasoning_with_large_language_models.md)**

:   提出 Mario，针对多模态图（MMG）上的 LLM 推理，通过图条件视觉语言模型（GVLM）实现拓扑感知的跨模态对齐，再用模态自适应提示路由器（MAPR）为每个节点选择最优模态配置，在节点分类和链接预测上达到 SOTA。

**[Mixture-of-Experts based Feature Decoupling for Open Vocabulary Scene Graph Generation](mixture-of-experts_based_feature_decoupling_for_open_vocabulary_scene_graph_gene.md)**

:   针对开放词表场景图生成（OVSGG）里"只套用现成 VLM 特征、缺乏判别性属性、物体与关系语义割裂"的痛点，本文提出 MoE-FD：用混合专家自适应地把物体/关系特征解耦成形状、纹理、空间等子属性，再用迭代跨注意力让节点与边互相精炼，最终在 Visual Genome 全开放词表设定下把新类 R@100 大幅刷高（OvD+R 新关系 R@20 比 ACC 高 4.24%）。

**[R2G: A Multi-View Circuit Graph Benchmark Suite from RTL to GDSII](r2g_multi_view_circuit_graph_benchmark_suite_from_rtl_to_gdsii.md)**

:   提出 R2G，首个标准化的多视图电路图基准套件，在 30 个 IP 核上提供 5 种阶段感知的图表示（具有信息对等性），系统研究发现图表示选择比 GNN 模型选择对性能影响更大。

**[Robo-SGG: Exploiting Layout-Oriented Normalization and Restitution Can Improve Robust Scene Graph Generation](robo-sgg_exploiting_layout-oriented_normalization_and_restitution_can_improve_ro.md)**

:   针对鲁棒场景图生成（在噪声/模糊/天气等损坏图像上推理）里"视觉特征发生域偏移导致性能暴跌"的痛点，本文提出即插即用的 Robo-SGG：用实例归一化抹掉损坏带来的域特异统计、再用布局感知注意力把全局结构特征找回来（NRM），并用门控融合自适应平衡视觉与坐标特征（LEE），插到现有 SGG 模型上即在 VG-C 上把 PredCls/SGCls/SGDet 的 mR@50 相对提升 6.3% / 11.1% / 8.0%。

**[ViterbiPlanNet: Injecting Procedural Knowledge via Differentiable Viterbi for Planning](viterbiplannet_injecting_procedural_knowledge_via_differentiable_viterbi_for_pla.md)**

:   将过程知识图（PKG）通过可微Viterbi层端到端嵌入规划模型，使神经网络只需学习发射概率而非记忆完整过程结构，在CrossTask/COIN/NIV上以仅5-7M参数（比扩散/LLM方法少1-3个数量级）达到SOTA成功率，并建立了统一的评估基准。
