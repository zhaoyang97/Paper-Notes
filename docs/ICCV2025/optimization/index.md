---
title: >-
  ICCV2025 优化/理论论文汇总 · 7篇论文解读
description: >-
  7篇ICCV2025的优化/理论方向论文解读，涵盖联邦学习、模型压缩、个性化生成、多模态等方向。覆盖该方向前沿研究进展与技术创新，每篇含一句话总结、核心思想、方法详解、实验结果与局限性分析，5分钟读懂一篇论文核心思想，助你快速跟进AI领域最新研究动态、学术前沿趋势与核心技术突破。
tags:
  - "ICCV2025"
  - "优化/理论"
  - "论文解读"
  - "论文笔记"
  - "联邦学习"
  - "模型压缩"
  - "个性化生成"
  - "多模态"
item_list:
  - u: "addressing_representation_collapse_in_vector_quantized_models_with_one_linear_la/"
    t: "Addressing Representation Collapse in Vector Quantized Models with One Linear Layer"
  - u: "class-wise_federated_averaging_for_efficient_personalization/"
    t: "Class-Wise Federated Averaging for Efficient Personalization"
  - u: "federated_continual_instruction_tuning/"
    t: "Federated Continual Instruction Tuning"
  - u: "federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data/"
    t: "Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data"
  - u: "learning_interpretable_queries_for_explainable_image_classification_with_informa/"
    t: "Learning Interpretable Queries for Explainable Image Classification with Information Pursuit"
  - u: "memory-efficient_4-bit_preconditioned_stochastic_optimization/"
    t: "Memory-Efficient 4-bit Preconditioned Stochastic Optimization"
  - u: "zeroth-order_fine-tuning_of_llms_in_random_subspaces/"
    t: "Zeroth-Order Fine-Tuning of LLMs in Random Subspaces"
item_total: 7
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 📐 优化/理论

**📹 ICCV2025** · **7** 篇论文解读

📌 **同领域跨会议浏览：** [📷 CVPR2026 (22)](../../CVPR2026/optimization/index.md) · [🔬 ICLR2026 (220)](../../ICLR2026/optimization/index.md) · [🧪 ICML2026 (88)](../../ICML2026/optimization/index.md) · [🤖 AAAI2026 (21)](../../AAAI2026/optimization/index.md) · [🧠 NeurIPS2025 (126)](../../NeurIPS2025/optimization/index.md) · [🧪 ICML2025 (61)](../../ICML2025/optimization/index.md)

🔥 **高频主题：** 联邦学习 ×3

**[Addressing Representation Collapse in Vector Quantized Models with One Linear Layer](addressing_representation_collapse_in_vector_quantized_models_with_one_linear_la.md)**

:   提出SimVQ方法，通过一个可学习的线性变换层对码本向量进行重参数化（$\bm{C}\bm{W}$），将码本的不相交优化转化为联合空间优化，从根本上解决VQ模型中的表示崩塌问题，实现接近100%的码本利用率。

**[Class-Wise Federated Averaging for Efficient Personalization](class-wise_federated_averaging_for_efficient_personalization.md)**

:   cwFedAvg 将 FedAvg 从"按客户端聚合"扩展为"按类别聚合"，为每个类别创建专属全局模型，再根据各客户端的类别分布加权组合成个性化模型，配合权重分布正则化（WDR）增强类别分布与权重范数的关联，在保持 FedAvg 通信开销的同时显著提升非 IID 场景下的个性化性能。

**[Federated Continual Instruction Tuning](federated_continual_instruction_tuning.md)**

:   首次提出联邦持续指令微调（FCIT）基准，涵盖 2 种场景、4 种设置和 12 个数据集，并设计 DISCO 框架通过动态知识组织（DKO）和子空间选择性激活（SSA）有效解决数据异构性和灾难性遗忘。

**[Federated Prompt-Tuning with Heterogeneous and Incomplete Multimodal Client Data](federated_prompt-tuning_with_heterogeneous_and_incomplete_multimodal_client_data.md)**

:   提出 FED-PRIME，一个面向多模态数据模态缺失场景的联邦 Prompt-Tuning 框架，通过 inter-client 和 intra-client 两组 prompt 分别捕获跨客户端可对齐的缺失模式和客户端内特有的缺失模式，并通过聚类-对齐机制进行服务端聚合，在多种缺失数据设置下大幅超越现有基线。

**[Learning Interpretable Queries for Explainable Image Classification with Information Pursuit](learning_interpretable_queries_for_explainable_image_classification_with_informa.md)**

:   在CLIP语义嵌入空间中将信息追踪（Information Pursuit）的查询字典参数化为可学习向量，通过交替优化算法学习任务充分的可解释查询字典，缩小了可解释分类器与黑盒分类器的性能差距。

**[Memory-Efficient 4-bit Preconditioned Stochastic Optimization](memory-efficient_4-bit_preconditioned_stochastic_optimization.md)**

:   提出基于 Cholesky 分解 + 误差反馈的 4-bit 量化方案，将 Shampoo 优化器的预条件矩阵压缩至 4-bit 精度，在大幅降低 GPU 显存的同时保持与 32-bit Shampoo 接近的训练性能，并给出了光滑与非光滑两种场景下的收敛性证明。

**[Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)**

:   提出 SubZero（random Subspace Zeroth-order），通过逐层低秩扰动在随机子空间中估计梯度，显著降低零阶优化的梯度方差和角度误差，以接近推理的内存开销实现 LLM 的高效微调。
