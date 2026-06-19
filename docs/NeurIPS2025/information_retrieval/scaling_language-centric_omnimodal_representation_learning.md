---
title: >-
  [论文解读] Scaling Language-Centric Omnimodal Representation Learning
description: >-
  [NeurIPS 2025][信息检索/RAG][多模态表示学习] 提出 LCO-Emb 框架，发现多模态大模型（MLLM）在生成式预训练中已隐式建立跨模态对齐，仅需轻量级的纯文本对比学习微调即可激活全模态表示能力，并发现生成能力与表示性能正相关的 Generation-Representation Scaling Law (GRSL)。
tags:
  - "NeurIPS 2025"
  - "信息检索/RAG"
  - "多模态表示学习"
  - "对比学习"
  - "MLLM嵌入"
  - "跨模态对齐"
  - "生成-表示缩放律"
---

# Scaling Language-Centric Omnimodal Representation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.11693](https://arxiv.org/abs/2510.11693)  
**代码**: [GitHub](https://github.com/LCO-Embedding/LCO-Embedding)  
**领域**: 信息检索  
**关键词**: 多模态表示学习, 对比学习, MLLM嵌入, 跨模态对齐, 生成-表示缩放律

## 一句话总结

提出 LCO-Emb 框架，发现多模态大模型（MLLM）在生成式预训练中已隐式建立跨模态对齐，仅需轻量级的纯文本对比学习微调即可激活全模态表示能力，并发现生成能力与表示性能正相关的 Generation-Representation Scaling Law (GRSL)。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：跨模态表示对齐是多模态AI的核心课题。传统方法（如CLIP）依赖大规模配对数据的对比学习来实现视觉-语言对齐，但在多语言检索、视觉文本理解、交错多模态编码等复杂任务上性能趋于饱和。

**核心痛点**：

**CLIP范式的瓶颈**：CLIP风格模型靠扩大模型、数据、batch size来提升，但对深层跨模态理解任务效果有限

**MLLM嵌入优势的黑箱问题**：近期基于MLLM的嵌入方法表现优于CLIP，但为什么更好缺乏深入分析

**多模态训练数据需求庞大**：现有最优方法（如GME）需要800万对多模态配对数据

**核心洞察**：MLLM在生成式预训练过程中，语言解码器学会了在共享表示空间中利用多模态信号来生成单模态输出，因此已经隐式地实现了跨模态对齐。对比学习只需要作为轻量的"激活"步骤，而非从头学习对齐。

## 方法详解

### 整体框架

LCO-Emb 的核心思路非常简洁：取出 MLLM 的语言解码器（LLM），用纯文本对比学习 + LoRA 微调，再插回原始 MLLM 架构中。冻结模态编码器和投影器，仅更新解码器的 LoRA 参数。

### 关键设计

1. **隐式跨模态对齐的发现与验证**

    - 通过各向异性（Anisotropy）分析证明：原始 MLLM 表示空间存在退化（高各向同性度），纯文本对比学习后，不仅文本嵌入变得各向同性，图像、音频、视频嵌入也同步改善
    - 通过核级相似性（Kernel-level Similarity）分析证明：纯文本微调后，图像与语言模态间的 kNN 重合度显著提升，且 7B 模型比 3B 有更强的跨模态核对齐

2. **Language-Centric 对比学习策略**

    - 仅使用文本配对数据（all-NLI 的 27.6 万三元组）进行 InfoNCE 对比学习
    - 采用 LoRA 而非全参微调，核心目的不是参数效率，而是最小化对预训练权重的扰动，保留已建立的跨模态对齐结构
    - 可选: 加入约 9.4 万条合成多模态配对数据做校准，共 37 万样本

3. **多模态变体与模型融合**

    - 分别在不同数据集上微调（all-NLI 侧重语义相似性，Scale-1M 侧重多语言和场景描述），通过 Model Soup 权重平均融合取各自优势
    - 支持 LLaVA-Next、Qwen2.5-VL、Qwen2.5-Omni 多种 MLLM 骨干

### 训练策略

- 优化器：AdamW + cosine schedule，峰值学习率 $4 \times 10^{-4}$
- Batch size: 768（纯文本），多模态按比例放大至 ~1052
- 训练 2 个 epoch，LoRA rank=64, alpha=16（纯文本）/ 128（多模态）
- 训练时间：纯文本仅需 ~4.7 GPU Hours (3B) / ~9.3 GPU Hours (7B)

## 实验关键数据

### 主实验（MIEB-Lite 51任务）

| 模型 | 数据量 | 检索(en) | 聚类 | 零样本分类 | 线性探查 | 组合性 | 文档理解 | vSTS(en) | 均分 |
|------|--------|----------|------|-----------|---------|--------|---------|---------|------|
| CLIP-ViT-bigG (2B) | - | 34.2 | 80.8 | 72.4 | 77.8 | 35.0 | 35.5 | 73.4 | 51.3 |
| GME (7B) | 8.0M | 37.9 | 69.6 | 55.5 | 68.7 | 52.2 | 86.1 | 81.8 | 64.5 |
| LCO-Emb-VL (7B, 文本) | 276k | 31.8 | 52.7 | 49.1 | 68.5 | 40.4 | 66.0 | 88.4 | 60.4 |
| LCO-Emb-Omni (7B, 多模态) | 370k | **36.4** | **80.0** | **68.5** | **74.1** | 50.1 | **75.4** | **86.2** | **68.8** |

### 消融实验（训练策略对比，Qwen2.5-VL-7B）

| 训练策略 | GPU小时 | 多语言检索 | vSTS(en) | 文档理解 | 线性探查 | 均分 |
|---------|---------|-----------|---------|---------|---------|------|
| CLIP-style CL (多模态800K) | ~550h | 18.24 | 73.92 | 44.89 | 38.93 | 50.02 |
| Full-Finetune (纯文本) | ~17.3h | 44.05 | 83.15 | 58.02 | 53.34 | 66.49 |
| **LoRA (纯文本)** | **~9.3h** | **56.64** | **85.05** | **67.49** | **53.91** | **71.98** |

### 关键发现

- **纯文本训练超过多模态CLIP训练**：LoRA纯文本微调仅需 CLIP-style 训练 1/60 的时间，却在均分上高出 22 分
- **LoRA 显著优于全参微调**：保留预训练对齐结构是核心，LoRA 的约束力比全参微调更好
- **多教师融合锦上添花**：仅加入 9.4 万多模态数据（总量的 25%）即可将均分从 60.4 提升到 67.6

## 亮点与洞察

- **最重要的洞察**：对比学习在 MLLM 上的角色不是"学习对齐"，而是"激活对齐"——MLLM 已经在生成预训练中隐式建立了跨模态对齐，对比学习只是把表示空间从各向异性退化中"唤醒"
- **GRSL (Generation-Representation Scaling Law)**：模型的生成能力越强，经对比微调后的表示性能上界越高。这一关系通过 PAC-Bayesian 框架给出了理论解释——生成损失 $\mathcal{L}_g(P)$ 决定了表示性能的上界
- **LoRA 的新视角**：LoRA 的核心价值不是参数效率，而是对预训练知识和跨模态对齐的最小扰动
- **极致的数据效率**：仅 27.6 万文本对即可超过使用 800 万多模态数据的 GME

## 局限与展望

- 表示能力受限于底层 MLLM 的生成能力——如果基座模型在某些模态上生成能力弱，对比微调也无法弥补
- 纯文本变体在聚类和零样本分类上仍落后于 CLIP-style 编码器模型
- GRSL 的验证目前主要在 Qwen 系列模型上，更广泛的模型族验证尚不充分
- 没有探索更高效的对比学习目标（如 hard negative mining）

## 相关工作与启发

- **vs CLIP/SigLIP**：CLIP需要大规模配对数据从头学习对齐，LCO-Emb利用MLLM预训练已有的对齐，数据需求降低20倍+
- **vs GME**：GME用800万多模态数据训练，LCO-Emb用37万样本（仅4.6%）就在MIEB上超过GME
- **vs E5-V**：同为MLLM嵌入方法，LCO-Emb在Sub18上平均高出21.69分，归因于更好的骨干和LoRA策略

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 对MLLM隐式对齐的发现和GRSL是全新的洞察，理论化程度高
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖51个任务、多骨干、多模态、完整消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验图表丰富，但理论部分略紧凑
- 价值: ⭐⭐⭐⭐⭐ 揭示了MLLM表示学习的根本规律，对未来嵌入模型设计有范式级影响

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Multi-Label Cluster Discrimination for Visual Representation Learning](../../ECCV2024/information_retrieval/multi-label_cluster_discrimination_for_visual_representation_learning.md)
- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[NeurIPS 2025\] Learning Task-Agnostic Representations through Multi-Teacher Distillation](learning_task-agnostic_representations_through_multi-teacher_distillation.md)
- [\[AAAI 2026\] Towards Inference-Time Scaling for Continuous Space Reasoning](../../AAAI2026/information_retrieval/towards_inference-time_scaling_for_continuous_space_reasoning.md)
- [\[ICLR 2026\] Revela: Dense Retriever Learning via Language Modeling](../../ICLR2026/information_retrieval/revela_dense_retriever_learning_via_language_modeling.md)

</div>

<!-- RELATED:END -->
