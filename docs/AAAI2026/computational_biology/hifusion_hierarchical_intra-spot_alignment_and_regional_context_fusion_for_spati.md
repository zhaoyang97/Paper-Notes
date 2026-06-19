---
title: >-
  [论文解读] HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology
description: >-
  [AAAI 2026][计算生物][空间转录组学] 提出 HiFusion 框架，通过层次化 spot 内建模（HISM）和上下文感知跨尺度融合（CCF）两个互补模块，从 H&E 染色全切片图像中准确预测空间基因表达，在两个基准数据集的 2D 切片交叉验证和 3D 样本特异性评估中均达到 SOTA。
tags:
  - "AAAI 2026"
  - "计算生物"
  - "空间转录组学"
  - "基因表达预测"
  - "多尺度特征融合"
  - "组织病理学"
  - "交叉注意力"
---

# HiFusion: Hierarchical Intra-Spot Alignment and Regional Context Fusion for Spatial Gene Expression Prediction from Histopathology

**会议**: AAAI 2026  
**arXiv**: [2511.12969](https://arxiv.org/abs/2511.12969)  
**代码**: [GitHub](https://github.com/Advanced-AI-in-Medicine-and-Physics-Lab/HiFusion)  
**领域**: 医学图像 / 空间转录组学  
**关键词**: 空间转录组学, 基因表达预测, 多尺度特征融合, 组织病理学, 交叉注意力

## 一句话总结

提出 HiFusion 框架，通过层次化 spot 内建模（HISM）和上下文感知跨尺度融合（CCF）两个互补模块，从 H&E 染色全切片图像中准确预测空间基因表达，在两个基准数据集的 2D 切片交叉验证和 3D 样本特异性评估中均达到 SOTA。

## 研究背景与动机

- **空间转录组学（ST）** 能够在保留空间定位的同时实现全基因组表达谱分析，但临床推广受限于高昂成本、专用设备和有限的可扩展性
- **H&E 染色 WSI** 在临床病理中常规获取，成本低廉，蕴含与基因表达密切关联的丰富形态学特征（如 HER2 阳性乳腺癌中 ERBB2 过表达与特定形态表型的关联）
- **现有方法的局限**：
    - 大多数方法将每个 spot 视为同质区域，忽略了 spot 内的层次结构（一个 55-100μm 的 spot 内包含多种细胞类型、核纹理、亚细胞模式）
    - 区域上下文信息仅作为辅助输入，未显式建模 spot 与周围组织的语义关联
    - TRIPLEX、ASIGN 等方法使用大尺寸区域 patch（超过 1000×1000 像素），但大感受野可能引入形态学噪声
- **核心问题**：如何同时捕获 spot 内细粒度形态学异质性和周围组织的生物学相关上下文

## 方法详解

### 整体框架

HiFusion 是一个双分支框架，包含两个关键组件：

1. **HISM（Hierarchical Intra-Spot Modeling）**：层次化 spot 内建模与对齐
2. **CCF（Context-Aware Cross-Scale Fusion）**：上下文感知跨尺度融合

输入为 spot 图像 $X^S \in \mathbb{R}^{n \times H_S \times W_S \times 3}$ 和区域邻域 patch $X^N \in \mathbb{R}^{n \times H_N \times W_N \times 3}$，目标是学习映射 $\phi: \{X^S, X^N\} \rightarrow Y$，预测基因表达向量 $Y \in \mathbb{R}^{n \times m}$。

### HISM 模块：层次化 Spot 内建模

核心思想是通过多分辨率子 patch 分解捕获 spot 内形态学异质性：

- **Level-0**：原始 spot 图像（224×224），通过共享编码器 $f_\theta$ 提取全局特征图 $\mathbf{F}_0^S \in \mathbb{R}^{d \times h \times w}$
- **Level-1**：将 spot 分解为 $p \times p$（2×2）个不重叠子 patch，捕获亚组织/区域结构
- **Level-2**：将 spot 分解为 $q \times q$（7×7）个不重叠子 patch，捕获细胞/亚细胞级信息
- 所有子 patch 通过**共享编码器**（ResNet-18）提取特征，按原始空间位置重建特征图
- 若重建分辨率与 Level-0 不匹配，使用双线性插值对齐

**特征对齐损失**：确保跨尺度语义一致性

$$\mathcal{L}_{\text{align}} = \sum_{s=1}^{2} \|\tilde{\mathbf{F}}_s^S - \mathbf{F}_0^S\|_1$$

利用 CNN 的平移不变性，鼓励细粒度特征保持全局语义一致。

### CCF 模块：上下文感知跨尺度融合

- **区域编码**：邻域 patch 通过轻量级编码器（ResNet-10）+ 全局平均池化，得到区域表示 $\mathbf{Q}_i^N \in \mathbb{R}^{1 \times d}$
- **可学习加权融合**：三个尺度的 spot 特征通过 softmax 归一化的可学习权重自适应融合

$$\mathbf{F}_{\text{fused}}^S = \sum_{s=0}^{2} \omega_s \cdot \mathbf{F}_s^S, \quad \omega_s = \frac{\exp(\alpha_s)}{\sum_{j=0}^{2} \exp(\alpha_j)}$$

- **跨注意力融合**：区域表示作为 Query，融合后的多尺度 spot 特征（经自适应平均池化为 $k^2$ 个 token）作为 Key 和 Value
- **残差连接 + 预测头**：

$$\hat{\mathbf{y}}_i = \text{FC}(\text{LayerNorm}(\mathbf{Q}_i^N + \phi_{ca}(\mathbf{Q}_i^N, \mathbf{K}_i^S, \mathbf{V}_i^S)))$$

该设计使模型能选择性关注生物学相关的上下文信息，同时抑制空间噪声。

### 损失函数

总损失由三部分组成：

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{main}} + \mathcal{L}_{\text{aux}}}_{\mathcal{L}_{\text{reg}}} + \lambda \mathcal{L}_{\text{align}}$$

- $\mathcal{L}_{\text{main}}$：MSE 回归损失
- $\mathcal{L}_{\text{aux}}$：多尺度辅助监督，各层级特征独立预测基因表达
- $\mathcal{L}_{\text{align}}$：跨尺度特征对齐正则化（$\lambda=1$）

## 实验

### 数据集

| 数据集 | 描述 | 样本数 | spot 数 |
|--------|------|--------|---------|
| HER2 | HER2 阳性乳腺肿瘤 | 36 WSI（8 patients） | 13,620 |
| ST-Data | 乳腺癌（ST-Net） | 16 samples | 41,544 |

预测 top-250 高表达基因，基因表达值经 spot-wise 归一化 + 对数变换。

### 主实验结果

| 方法 | HER2-2D MSE↓ | HER2-2D PCC↑ | HER2-3D MSE↓ | HER2-3D PCC↑ | ST-2D MSE↓ | ST-2D PCC↑ | ST-3D MSE↓ | ST-3D PCC↑ |
|------|-------------|-------------|-------------|-------------|-----------|-----------|-----------|-----------|
| ST-Net | 0.6523 | 0.4621 | 0.5323 | 0.7042 | 0.5798 | 0.5304 | 0.4939 | 0.7443 |
| TRIPLEX | 0.5715 | 0.4750 | 0.2899 | 0.7471 | 0.5389 | 0.5387 | 0.2857 | 0.7780 |
| ASIGN-2D | 0.5830 | 0.4601 | 0.3116 | 0.7316 | 0.5449 | 0.5373 | 0.2822 | 0.7741 |
| **HiFusion** | **0.5459** | **0.4961** | **0.2846** | **0.7492** | **0.5095** | **0.5613** | **0.2711** | **0.7838** |

- 2D 评估：HiFusion 在 HER2 上 MSE 低于 TRIPLEX 2.1-2.6%，PCC 高 2%+
- 3D 评估：相比 ST-Net 改善 22-25%；3D 内样本学习策略优于 ASIGN-3D 的复杂 3D 对齐方案

### 消融实验

**HISM 分解层级**：
- 1×1（无分解）已接近第二好 baseline（TRIPLEX）
- 1×1 + 2×2 + 7×7 组合最优，互补的空间粒度：全局组织 → 亚区域结构 → 细胞级

**特征对齐损失**：
- 加入对齐损失在 HER2 上 MSE 降低约 2%，PCC 提升超过 2%

**CCF 模块**：
- Spot token 数量：2×2（4 个 token）最优，更多 token 引入噪声
- 邻域 patch 大小：2 倍 spot 尺寸（448×448）最优，更大区域引入无关组织信号

### 癌症标志基因可视化

- ERBB2：HiFusion MAE=0.711, PCC=0.518 vs ASIGN MAE=1.074, PCC=-0.035
- KRT19：HiFusion MAE=0.446, PCC=0.230（最优）
- CD74：HiFusion MAE=0.584, PCC=0.357（最优）
- 视觉上 HiFusion 最准确定位高表达区域

## 亮点与洞察

1. **层次化 spot 内建模**思路新颖，利用多分辨率分解 + 共享编码器，计算高效且语义一致
2. **适度上下文**优于大范围上下文：448（2×spot）优于更大的 1120，说明过大感受野引入噪声
3. **3D 内样本学习**策略反直觉但有效：仅用单层训练、预测邻近层，优于复杂的跨样本 3D 策略
4. **设计简洁实用**：ResNet-18/10 骨干，单卡 RTX 4090 可训练，可扩展性强
5. 跨尺度特征对齐损失是重要的正则化手段，确保多分辨率特征语义一致

## 局限性

- 仅在乳腺癌数据集（HER2 和 ST-Data）上验证，未涵盖其他组织类型或疾病
- 编码器采用固定的 ResNet 架构，未探索基于 foundation model（如 UNI、CONCH）的预训练编码器
- 区域上下文仅使用单个邻域 patch，未考虑更灵活的空间图建模（如 GNN）
- 仅预测 top-250 基因，全基因组预测仍待探索

## 相关工作

- **ST-Net**（He et al. 2020）：DenseNet 独立 spot 预测，开创性工作
- **HisToGene**（Pang et al. 2021）：ViT 建模长程依赖
- **Hist2ST**（Zeng et al. 2022）：ConvMixer + GNN 邻域建模
- **EGN/BLEEP**：图像相似性检索/对比学习，但对染色变异敏感
- **TRIPLEX**（Chung et al. 2024）：三分支架构（spot/邻域/全局），CVPR 2024
- **ASIGN**（Zhu et al. 2025）：3D 组织切面对齐 + 图模型，当前 SOTA

## 评分 ⭐⭐⭐⭐

方法设计清晰合理，层次化分解 + 跨注意力融合的组合有效解决了多尺度建模问题。实验充分，消融完整，癌症标志基因可视化增强了临床可解释性。但数据集多样性有限，未探索更强的预训练编码器，泛化性有待进一步验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] From Spots to Pixels: Dense Spatial Gene Expression Prediction from Histology Images](../../CVPR2026/computational_biology/from_spots_to_pixels_dense_spatial_gene_expression_prediction_from_histology_ima.md)
- [\[CVPR 2026\] HyperST: Hierarchical Hyperbolic Learning for Spatial Transcriptomics Prediction](../../CVPR2026/computational_biology/hyperst_hierarchical_hyperbolic_learning_for_spatial_transcriptomics_prediction.md)
- [\[AAAI 2026\] SpaCRD: Multimodal Deep Fusion of Histology and Spatial Transcriptomics for Cancer Region Detection](spacrd_multimodal_deep_fusion_of_histology_and_spatial_transcriptomics_for_cance.md)
- [\[CVPR 2026\] Multi-View Hierarchical Alignment Learning for Spatial Transcriptomics](../../CVPR2026/computational_biology/multi-view_hierarchical_alignment_learning_for_spatial_transcriptomics.md)
- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](../../CVPR2026/computational_biology/cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)

</div>

<!-- RELATED:END -->
