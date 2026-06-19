---
title: >-
  [论文解读] Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning
description: >-
  [CVPR 2025][计算生物][全切片图像表征学习] 提出 Cobra，一种无监督的基础模型无关 (FM-agnostic) 全切片图像 (WSI) 级别表征学习框架：将来自多个预训练 patch 级基础模型的嵌入作为特征空间增广，通过 Mamba-2 编码器和对比学习训练 slide 编码器，仅用 3048 张 WSI 预训练即在 15 个下游任务上平均 AUC 超过现有 slide 编码器至少 +4.4%。
tags:
  - "CVPR 2025"
  - "计算生物"
  - "全切片图像表征学习"
  - "自监督学习"
  - "对比学习"
  - "基础模型无关"
  - "Mamba-2"
---

# Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning

**会议**: CVPR 2025  
**arXiv**: [2411.13623](https://arxiv.org/abs/2411.13623)  
**代码**: [github](https://github.com/KatherLab/COBRA)  
**领域**: 医学图像  
**关键词**: 全切片图像表征学习, 自监督学习, 对比学习, 基础模型无关, Mamba-2

## 一句话总结

提出 Cobra，一种无监督的基础模型无关 (FM-agnostic) 全切片图像 (WSI) 级别表征学习框架：将来自多个预训练 patch 级基础模型的嵌入作为特征空间增广，通过 Mamba-2 编码器和对比学习训练 slide 编码器，仅用 3048 张 WSI 预训练即在 15 个下游任务上平均 AUC 超过现有 slide 编码器至少 +4.4%。

## 研究背景与动机

计算病理学中，全切片图像 (WSI) 可达 $150,000 \times 150,000$ 像素，无法直接用 ViT 处理。主流方法是将 WSI 切割为小 patch，用预训练基础模型 (FM) 提取 patch 嵌入，然后通过 MIL 聚合为 slide 级别预测。但 MIL 是**有监督的、任务特定的**。

无监督 slide 表征学习试图生成任务无关的 slide 嵌入，但面临核心挑战：**如何为 slide 级别学习生成有效的数据增广？** 传统图像增广对现代 FM 几乎无效（FM 已对这些变换不变），多染色/多模态方法受数据可用性限制。

Cobra 的核心洞察：**不同的patch级FM本身就构成了特征空间增广**——同一slide经不同FM编码产生不同但语义一致的patch嵌入序列，再结合不同放大倍率的嵌入，可直接在特征空间进行对比学习，无需任何像素级增广。

## 方法详解

### 整体框架

Cobra 在预处理阶段用 4 个预训练 FM（CTransPath、UNI、Virchow2、H-Optimus-0）在 3 个放大倍率（0.5、1.14、2 MPP）下提取 patch 嵌入。slide 编码器由三部分组成：(1) Embedding MLP 将不同维度的 FM 嵌入映射到共享空间；(2) 两层 Mamba-2 (SSD) 编码 patch 序列；(3) Multi-head Gated Attention 聚合为单一 slide 向量。使用 MoCo 风格对比损失训练。

### 关键设计

**1. Feature Space Augmentation via Multiple FMs — 多FM特征空间增广**

- **功能**: 无需像素级增广即可生成用于对比学习的正样本对
- **核心思路**: 同一患者的 WSI 经不同 FM（$fe_n \in \{CTP, UNI, V2, H0\}$）和不同放大倍率提取 patch 嵌入，这些嵌入构成同一 slide 的不同"视图"。查询 $q$ 和正键 $k^+$ 来自同一患者但不同 FM/放大倍率的嵌入
- **设计动机**: 不同 FM 有不同的架构、预训练数据和训练目标，因此捕获互补的形态学特征。不同放大倍率提供多尺度上下文。这种在特征空间（而非像素空间）的增广对 FM 不变性免疫，是比传统增广更有效的对比学习正样本生成策略

**2. Mamba-2 + Multi-head Gated Attention 架构**

- **功能**: 高效编码长序列 patch 嵌入并聚合为 slide 级别向量
- **核心思路**: 架构为 $z = f_A(f_S(f_E(H^{fe_n})))$。嵌入模块 $f_E$ 用 MLP+SiLU 将不同维度映射到共享 $d$ 维空间；状态空间模块 $f_S$ 用两层 Mamba-2 SSD 加残差连接编码序列；聚合模块 $f_A$ 用 $M$ 头门控注意力计算加权平均 $z = \sum_k a_k \cdot H_{S,k}$，其中 $a_k$ 通过 tanh-sigmoid 门控机制计算
- **设计动机**: Mamba-2 比 Transformer 在长序列上更高效，适合 WSI 的数千甚至数万 patch。门控注意力聚合比简单平均更能关注诊断相关区域

**3. 灵活的推理模式（Single-FM / Multi-FM / Unseen-FM）**

- **功能**: 推理时可使用训练时见过或未见过的任意 FM
- **核心思路**: Single-FM 模式：用编码嵌入 $H_S$ 计算注意力权重但用原始 patch 嵌入 $H^{fe_n}$ 做加权平均（Eq. 6）。Multi-FM 模式：多个 FM 的编码嵌入取平均后编码（Eq. 8）。Unseen-FM 模式：对训练时未见的 FM，仍通过嵌入模块将其映射到共享空间
- **设计动机**: 嵌入模块学到的映射具有泛化性，使 Cobra 能将**训练时未见的新 FM** 也转化为更好的 slide 级特征提取器。这对不断涌现的新 FM 非常有价值

### 损失函数 / 训练策略

- **损失函数**: InfoNCE 对比损失 $\mathcal{L}_q = -\log \frac{\psi(q, k^+)}{\sum_i \psi(q, k_i)}$，$\psi(x_1, x_2) = \exp(\text{sim}(x_1, x_2)/\tau)$
- **MoCo 风格训练**: 键编码器通过动量更新 $\theta_k \leftarrow m\theta_k + (1-m)\theta_q$
- **预训练数据**: 仅 3048 张来自 TCGA 的 WSI（跨 4 种组织类型），远少于 GigaPath 的 171K 或 PRISM 的 587K
- **模型规模**: 仅 15M 参数

## 实验关键数据

### 主实验

15 个下游分类任务（TCGA 训练，CPTAC 外部验证），平均 AUC：

| Slide Encoder | 预训练数据 | 参数量 | 平均 AUC |
|---------------|-----------|--------|----------|
| Mean CTransPath | - | - | 62.1 |
| Mean Virchow2 | - | - | 73.8 |
| GigaPath-SE | 171K WSI | 86M | 71.5 |
| CHIEF | 60K WSI | 1M | - |
| MADELEINE | 21K WSI | 5M | - |
| **Cobra (V2)** | **3K WSI** | **15M** | **78.2** |

### 消融实验

Cobra 各组件贡献（推理模式对比）：

| 推理模式 | 平均 AUC | 说明 |
|----------|----------|------|
| Mean patch embedding (V2) | 73.8 | 无 slide encoder |
| Cobra Single-FM (V2) | **78.2** | 使用原始嵌入做加权平均 |
| Cobra Multi-FM (4个) | 77.5 | 融合所有训练 FM |
| Cobra + unseen FM | 有提升 | 对未见 FM 也有效 |

### 关键发现

1. **极高的数据效率**：仅 3K WSI 预训练即超过用 171K WSI 的 GigaPath-SE，平均 AUC 78.2 vs 71.5
2. **FM 无关性**：Cobra 能将训练时未见的 FM（如新发布的 FM）也转化为更好的 slide 编码器
3. **Single-FM 推理优于 Multi-FM**：使用原始 patch 嵌入做加权平均（Eq. 6）优于使用编码后嵌入（Eq. 4），因为保留了 FM 特有的信息
4. **低放大倍率也能工作**：在计算效率与性能之间有良好的折中

## 亮点与洞察

1. **"FM 本身就是增广"的思路极其简洁优雅**：避免了传统 SSL 中设计增广策略的难题
2. **仅 3K WSI + 15M 参数**就达到 SOTA，数据效率令人惊讶，对资源受限机构非常友好
3. **对未见 FM 的泛化能力**使 Cobra 具有前瞻性价值——新 FM 不断涌现时无需重新训练

## 局限与展望

1. 嵌入模块对未见 FM 的映射质量取决于嵌入维度匹配和特征空间相似性
2. 仅在 4 种组织类型上预训练，对罕见肿瘤类型的泛化性未验证
3. Mamba-2 的序列建模假设 patch 有固定顺序，可能不最优
4. 可探索更多 FM 组合和自适应的 FM 选择策略

## 相关工作与启发

- **GigaPath**: 用 masked autoencoder 训练 slide 编码器，需 171K WSI。Cobra 用对比学习和多 FM 增广更高效
- **MADELEINE**: 利用多模态 (H&E + IHC) 对比学习，受数据限制。Cobra 仅需单模态
- **PRISM**: 用 587K WSI + 文本/基因等多模态训练。Cobra 在单模态和极少数据下超越
- **MambaMIL**: 将 Mamba 应用于 MIL，启发了 Cobra 的架构设计

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 多FM作为特征空间增广的思路原创且优雅
- **实验充分度**: ⭐⭐⭐⭐⭐ — 15个下游任务、外部验证、多种基线、消融全面
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，数学表述严谨
- **价值**: ⭐⭐⭐⭐⭐ — 对计算病理学领域有重要意义，极高的数据效率和FM无关性具有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Learning the Interaction Prior for Protein-Protein Interaction Prediction: A Model-Agnostic Approach](../../ICML2026/computational_biology/learning_the_interaction_prior_for_protein-protein_interaction_prediction_a_mode.md)
- [\[CVPR 2026\] CARE: A Molecular-Guided Foundation Model with Adaptive Region Modeling for Whole Slide Image Analysis](../../CVPR2026/computational_biology/care_a_molecular-guided_foundation_model_with_adaptive_region_modeling_for_whole.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](../../ICML2025/computational_biology/stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[ICML 2025\] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics](../../ICML2025/computational_biology/global_context-aware_representation_learning_for_spatially_resolved_transcriptom.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](../../NeurIPS2025/computational_biology/iterative_foundation_model_fine-tuning_on_multiple_rewards.md)

</div>

<!-- RELATED:END -->
