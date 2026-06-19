---
title: >-
  [论文解读] LangBridge: Interpreting Image as a Combination of Language Embeddings
description: >-
  [信息检索/RAG] LangBridge 通过将视觉特征显式分解为 LLM 词汇嵌入的线性组合，实现了可解释的视觉-语言对齐，并支持跨 LLM 的预训练无关适配器迁移。 主流大型视觉语言模型（LVLM）遵循 LLaVA 范式，使用 MLP 将视觉特征映射到 LLM 的文本嵌入空间。虽然这种方法有效…
tags:
  - "信息检索/RAG"
---

# LangBridge: Interpreting Image as a Combination of Language Embeddings

## 基本信息

- **会议**: ICCV 2025
- **arXiv**: 2503.19404
- **代码**: [项目主页](https://curryx-001.github.io/LangBridge.github.io/)
- **领域**: 信息检索
- **关键词**: 视觉语言对齐, 适配器迁移, 词汇嵌入投影, MLP分析, 跨模型复用

## 一句话总结

LangBridge 通过将视觉特征显式分解为 LLM 词汇嵌入的线性组合，实现了可解释的视觉-语言对齐，并支持跨 LLM 的预训练无关适配器迁移。

## 研究背景与动机

主流大型视觉语言模型（LVLM）遵循 LLaVA 范式，使用 MLP 将视觉特征映射到 LLM 的文本嵌入空间。虽然这种方法有效，但存在两个核心问题：

**MLP 对齐机制不透明**：MLP 如何弥合模态差距的底层机制尚不清楚，少有研究深入分析 MLP 的基本对齐机制

**适配器不可迁移**：由于输入维度不匹配和特征分布差异，切换 LLM 骨干网络时必须重新训练 MLP 适配器，计算成本高

作者首先对 MLP 适配器的工作原理进行了系统性研究，发现了两个关键结论：
- 视觉嵌入与语义相关的文本 token 具有强相关性（如苹果图像 patch 与 "Green Apple" 文本的高余弦相似度）
- MLP 的投影能力在训练过程中渐进式发展，逐步学会将视觉特征投影到对应文本嵌入子空间

## 方法详解

### 整体框架

LangBridge 的核心思想是**语言基向量投影（Language Basis Vector Projection）**：将视觉嵌入表示为 LLM 词汇嵌入的加权线性组合：

$$\mathbf{v} = \sum_{k=1}^{N} \beta_k \mathbf{t}_k$$

其中 $\beta_k$ 是词汇 token 上的概率分布。该方法包含三个阶段：

### 关键设计

**Stage 1: 视觉特征提取**

使用 Vision Transformer（CLIP-ViT-L/14@336px）从输入图像中提取 patch 级视觉特征：

$$\{v_i\}_{i=1}^{N} = \text{ViT}(\mathcal{I}), \quad v_i \in \mathbb{R}^{D}$$

**Stage 2: 概率计算**

通过两层 MLP 将视觉特征投影到 LLM 文本嵌入空间，再附加线性层生成词汇上的概率分布：

$$\mathbf{p} = \mathbf{W} \cdot \text{MLP}(\mathbf{v})$$

其中 $\mathbf{W} \in \mathbb{R}^{T \times D}$，$T$ 为词汇表大小。

**Stage 3: 文本嵌入线性组合**

使用概率分布作为系数，线性组合 LLM 的词汇嵌入：

$$\mathbf{v}_{\text{tokens}} = \sum_{i=1}^{T} p_i \mathbf{e}_i$$

### 词汇表选择策略

直接使用完整词汇嵌入矩阵计算量巨大（~1B 参数）。作者合并了 LLaMA 和 Qwen 的词汇表，仅保留两者共有的 token，再通过 ShareGPT4V 和 LLaVA-CC3M-Pretrain-595K 数据集统计频率，选取 Top-19,200 高频 token 作为精简词汇表。

### 适配器跨 LLM 复用

LangBridge 只学习视觉 patch 与词汇嵌入之间的线性组合关系（概率分布），而非直接维度映射。迁移时：

$$P = \text{LangBridge}_{\text{LLM}_1}(I) \in \mathbb{R}^{|V_{\text{shared}}|}$$
$$\text{Visiontoken}_{\text{LLM}_2} = P \cdot V_{\text{shared}}$$

概率分布 $P$ 可直接用于加权任意目标 LLM 的词汇嵌入，无需重新预训练。

## 实验关键数据

### 主实验：同架构迁移

在 Qwen2-0.5B 上预训练的 LangBridge 直接迁移到更大模型的结果：

| SFT-LLM | Connector | GQA | TextVQA | MME | MMBench | MMVeT | POPE | SciQA |
|---|---|---|---|---|---|---|---|---|
| Qwen2-7B | 7B-Pretrain-MLPs | 62.92 | 57.24 | 1938 | 72.7 | 35.5 | 87.8 | 79.44 |
| Qwen2-7B | 0.5B-Pretrain-LB | 63.03 | 57.25 | 1886 | 71.7 | 34.1 | 88.2 | 79.23 |
| Qwen2.5-14B | 14B-Pretrain-MLPs | 63.71 | 61.32 | 2038 | 78.2 | 37.7 | 88.1 | 85.59 |
| Qwen2.5-14B | 0.5B-SFT-LB | 63.92 | 62.02 | 1990 | 77.4 | 38.4 | 87.6 | 84.77 |

关键发现：0.5B 模型预训练的 LangBridge 迁移到 14B 模型后，TextVQA (+1.14%) 和 MMVeT (+1.86%) 甚至超过基线。

### 消融实验：词汇表大小

| Vocab Size | GQA | TextVQA | MME | MMBench | MMVeT | POPE | SciQA |
|---|---|---|---|---|---|---|---|
| 19,200 | 63.15 | 57.34 | 1904 | 71.0 | 31.6 | 88.3 | 79.25 |
| 25,600 | 63.13 | 57.58 | 1842 | 71.8 | 32.9 | 87.9 | 79.01 |
| 32,000 | 63.11 | 57.19 | 1832 | 72.7 | 33.2 | 88.6 | 79.11 |

19,200 词汇表在 MME 上显著优于更大词汇表（-6%~-8%），整体最优。

### 其他关键发现

- **跨架构迁移**：Qwen2-0.5B 预训练的 LangBridge 迁移到 LLaMA3-8B 后，MMVeT 提升 +9.68%，综合性能提升
- **正常设置对比**：LangBridge 在 LLaMA3-8B 上直接训练，GQA/TextVQA/MME/MMBench 均优于 MLP 基线
- **计算成本**：训练时间仅增加约 10%（4.273 vs 3.876 s/iter），且可完全省略跨 LLM 预训练

## 亮点与洞察

1. **深刻的机理分析**：通过可视化分析揭示 MLP 适配器渐进式学习将视觉特征投影到文本嵌入子空间的过程，为后续设计提供了理论依据
2. **优雅的设计思路**：将"隐式投影"转化为"显式线性组合"，使视觉-语言对齐过程可解释
3. **实用的迁移能力**：在小模型上预训练一次，即可迁移到多个大模型，大幅降低多模型部署成本
4. **共享词汇表策略**：通过频率统计选择跨模型共享词汇，以最小参数量实现跨架构兼容

## 局限性

- 词汇表选择依赖于特定训练数据集的频率统计，可能不适用于其他数据分布
- 当前仅在 LLaVA 系框架上验证，未扩展到 Flamingo、BLIP-2 等其他架构
- 19,200 的词汇表大小可能无法覆盖所有细粒度视觉概念
- 在部分 benchmark 上（如 MME）迁移后仍有 2-4% 的性能下降

## 相关工作与启发

- **Ovis** 使用视觉嵌入表实现结构化对齐，但不支持跨 LLM 迁移
- **LLaVA** 系列的 MLP 适配器虽然简单有效，但缺乏可解释性
- 本文的"词汇嵌入作为基向量"思路可能启发更多基于 LLM 内在表征进行跨模态对齐的研究

## 评分

⭐⭐⭐⭐ — 机理分析深刻，方法设计优雅，跨 LLM 迁移实用价值高，但在某些 benchmark 上迁移性能仍有损失。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Your Language Model Secretly Contains Personality Subnetworks](../../ICLR2026/information_retrieval/your_language_model_secretly_contains_personality_subnetworks.md)
- [\[ACL 2025\] Re-identification of De-identified Documents with Autoregressive Infilling](../../ACL2025/information_retrieval/reidentification_deidentified.md)
- [\[ACL 2025\] Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Space](../../ACL2025/information_retrieval/psycholinguistic_visual_semantic.md)
- [\[CVPR 2025\] Advancing Myopia To Holism: Fully Contrastive Language-Image Pre-training](../../CVPR2025/information_retrieval/advancing_myopia_to_holism_fully_contrastive_language-image_pre-training.md)
- [\[ACL 2025\] Enhancing Lexicon-Based Text Embeddings with Large Language Models](../../ACL2025/information_retrieval/enhancing_lexicon-based_text_embeddings_with_large_language_models.md)

</div>

<!-- RELATED:END -->
