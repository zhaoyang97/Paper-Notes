---
title: >-
  [论文解读] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing
description: >-
  [AAAI 2026 Oral][语义分割][半监督学习] 提出 S5 框架，首次将半监督语义分割扩展为遥感基础模型（RSFM）的预训练范式，通过构建百万级 RS4P-1M 数据集和 MoE 多数据集微调策略，在多个遥感分割与检测基准上达到 SOTA。 遥感语义分割依赖像素级标注，获取成本极高。半监督语义分割（S4）通过伪标…
tags:
  - "AAAI 2026 Oral"
  - "语义分割"
  - "半监督学习"
  - "remote sensing"
  - "foundation model"
  - "图像分割"
  - "mixture of experts"
---

# S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing

**会议**: AAAI 2026 Oral  
**arXiv**: [2508.12409](https://arxiv.org/abs/2508.12409)  
**代码**: [S5](https://github.com/lianglyu/S5)  
**领域**: 遥感图像分割  
**关键词**: semi-supervised learning, remote sensing, foundation model, semantic segmentation, mixture of experts

## 一句话总结

提出 S5 框架，首次将半监督语义分割扩展为遥感基础模型（RSFM）的预训练范式，通过构建百万级 RS4P-1M 数据集和 MoE 多数据集微调策略，在多个遥感分割与检测基准上达到 SOTA。

## 研究背景与动机

遥感语义分割依赖像素级标注，获取成本极高。半监督语义分割（S4）通过伪标签和一致性正则化利用无标注数据，但现有 S4 研究局限于小规模数据集和模型（如仅在 iSAID 上切分训练集），无法充分利用海量地球观测数据。

与此同时，遥感基础模型（RSFMs）虽然在自监督（MAE、对比学习）和有监督预训练上取得进步，但有监督预训练受限于标注规模，而自监督预训练与下游分割任务存在 gap。核心矛盾在于：**能否将 S4 扩展到百万级无标注遥感图像上，作为 RSFM 的预训练策略？**

本文的核心 idea 是将 S4 从"数据集内部的训练策略"升级为"大规模预训练范式（S4P）"，并通过数据筛选保证伪标签质量，通过 MoE 微调实现多数据集高效适配。

## 方法详解

### 整体框架

S5 包含三个阶段：(1) 数据集构建——通过熵过滤和多样性扩展策略从 MillionAID 等数据集中筛选百万级无标注图像，构建 RS4P-1M；(2) S4 预训练（S4P）——以 iSAID 为有标注数据、RS4P-1M 为无标注数据，使用 FixMatch 进行半监督预训练；(3) MoE 多数据集微调（MoE-MDF）——在共享 backbone 中嵌入 FFN-MoE 模块，用共享专家和数据集特定专家联合微调多个下游基准。

### 关键设计

**1. 熵过滤 + 多样性扩展的数据筛选策略**

直接使用所有无标注数据会因分布不匹配导致伪标签质量下降。S5 先在 iSAID 上训练初始分割模型（ViT-H + UperNet），对无标注图像进行推理，计算像素级平均熵：

$$E(x) = -\frac{1}{H \times W} \sum_{i=1}^{H \times W} \sum_{k=1}^{K} P^k(x^i) \log P^k(x^i)$$

低熵样本优先选择（伪标签更可靠），但仅选低熵会导致语义冗余。因此进一步用 K-Means 对有标注图像特征聚类为 $M$ 个簇，将无标注图像分配到最近簇，按配额选择：

$$B_m^u = B^u \cdot \frac{N_m^l}{B^l}$$

每簇达到配额后停止选择，保证语义多样性。最终构建包含 100 万图像的 RS4P-1M 数据集。

**2. 基于 FixMatch 的 S4 预训练**

采用弱-强增强一致性正则化。对无标注图像生成弱增强（随机缩放、裁剪、翻转）和强增强（CutMix、颜色抖动、高斯模糊）两个视图，总损失为：

$$\mathcal{L} = \mathcal{L}_s + \lambda \mathcal{L}_{u_s}$$

其中监督损失 $\mathcal{L}_s$ 为标准交叉熵，无监督损失仅对高置信度伪标签（$\max(p_j^{u_w}) \geq \tau$）计算：

$$\mathcal{L}_{u_s} = \frac{1}{B_u} \sum_{j=1}^{B_u} \mathbb{1}(\max(p_j^{u_w}) \geq \tau) \mathcal{L}_{ce}(\hat{y}_j, p_j^{u_s})$$

所有 RSFM 均以 MAE 预训练权重初始化，S4P 在此基础上进一步增强表征能力。

**3. MoE 多数据集微调（MoE-MDF）**

传统"一个数据集一个模型"导致参数冗余。MoE-MDF 在 ViT 的 FFN 中引入分支结构：共享专家学习通用特征，数据集特定专家学习领域特定特征。具体地，中间特征 $F_{\text{FFN}}$ 经过两个并行线性层：

$$F^{\text{shared}} = \text{Linear}_{\text{shared}}^{D \to (1-\alpha)C}(F^{\text{FFN}})$$
$$F^{\text{specific}} = \text{Linear}_{\text{specific}}^{D \to \alpha C}(F^{\text{FFN}})$$

最终拼接为输出 $F^{\text{out}} = \text{Concat}(F^{\text{shared}}, F^{\text{specific}})$。$\alpha$ 控制共享与特定的比例，推理时不增加额外延迟。

### 损失函数 / 训练策略

- 预训练阶段：FixMatch 的监督 + 无监督联合损失，置信度阈值 $\tau$ 过滤低质伪标签
- 微调阶段：各数据集特定 decoder + 共享 MoE backbone，共享专家用所有数据更新，特定专家仅用对应数据集更新

## 实验关键数据

### 主实验

| 方法 | Backbone | 检测参数(M) | DIOR-R mAP | DOTA-v2 mAP | 分割参数(M) | Vaihingen mIoU | Potsdam mF1 | LoveDA mIoU | OpenEarthMap mIoU |
|------|----------|------------|-----------|------------|------------|---------------|-------------|-------------|-------------------|
| RVSA | ViT-B+RVSA | 222.4 | 68.06 | 55.22 | 412.8 | 78.49 | 91.58 | 52.44 | 66.63 |
| GFM | Swin-B | 208.2 | 67.67 | 59.15 | 387.6 | 79.61 | 91.85 | 54.98 | 67.78 |
| MTP | ViT-L+RVSA | 669.2 | 74.54 | 58.41 | 1309.6 | 80.62 | 92.47 | 54.16 | 69.04 |
| SelectiveMAE | ViT-L | 669.2 | 71.75 | 57.84 | 1309.6 | 80.45 | 92.78 | 54.31 | 69.30 |
| BillionFM | ViT-G | 1993.9 | 73.62 | 58.69 | - | - | 92.58 | 54.40 | - |
| **S5** | **ViT-B** | **138.3** | **72.95** | **57.20** | **160.4** | **79.85** | **92.40** | **54.02** | **68.65** |
| **S5** | **ViT-L** | **377.8** | **75.21** | **59.71** | **435.0** | **80.72** | **92.78** | **55.67** | **69.66** |
| **S5** | **ViT-H** | **730.0** | **75.30** | **59.89** | **824.5** | **80.85** | **92.97** | **55.65** | **70.02** |

S5-ViT-L 在多数据集设置下仅需 435M 参数（Scale-MAE/SelectiveMAE 的 1/3），且性能全面超越。

### 消融实验

**预训练数据集对比**（ViT-B backbone）：

| 无标注数据 | 图像数 | iSAID Val | Vaihingen | LoveDA | DIOR-R |
|-----------|--------|----------|-----------|--------|--------|
| 无（MAE baseline） | - | 65.93 | 78.27 | 52.47 | 68.02 |
| SAMRS | 100k | 67.59 | 79.61 | 53.66 | 69.13 |
| MillionAID-random | 100k | 66.32 | 79.49 | 53.20 | 69.02 |
| MillionAID*（筛选后） | 100k | **67.66** | **79.77** | **53.81** | **69.65** |

筛选策略构建的 MillionAID* 在所有任务上优于随机采样和 SAMRS。

**微调策略对比**（MAE+S4P, ViT-B）：

| 微调策略 | 参数(M) | Vaihingen | Potsdam | OpenEarthMap | LoveDA | 平均 |
|---------|---------|-----------|---------|-------------|--------|------|
| SDF（单数据集） | 412.8 | 79.93 | 92.24 | 67.35 | 54.51 | 73.51 |
| MDF（多数据集） | 132.1 | 79.82 | 92.25 | 68.41 | 54.53 | 73.75 |
| MoE-MDF (α=1/4) | 160.4 | **79.85** | **92.40** | **68.80** | 54.57 | **74.15** |
| MoE-MDF (α=1/2) | 188.7 | 79.84 | 92.39 | 68.66 | **54.64** | 73.88 |

MoE-MDF 在参数量仅为 SDF 的 1/3 时，平均精度提升 0.64 个点。α=1/4 为最优比例。

### 关键发现

- S4P 作为 MAE 之后的二次预训练，在所有下游任务上均有显著提升
- 数据筛选策略有效：筛选后 100k 图像优于随机 100k 且可扩展到 1M
- 模型规模和数据规模双重扩展均有效：ViT-B→ViT-H、100k→1M 均持续提升
- MoE-MDF 以极少参数开销实现多数据集统一部署

## 亮点与洞察

- **新范式**：将半监督学习从"训练技巧"升级为"基础模型预训练策略"，概念上有突破性
- **实用性强**：RS4P-1M 数据集构建不依赖额外标注（如 SAMRS 依赖 SAM 生成 mask），可扩展性好
- **参数效率突出**：MoE-MDF 在多数据集场景下参数量仅为传统方法的 1/3~1/4

## 局限与展望

- 预训练仅使用 FixMatch 作为 S4 方法，未探索更新的半监督方法（如 UniMatchV2）带来的增益
- MoE 设计较简单（仅共享+特定两个分支），未引入路由机制或更复杂的专家选择
- 数据筛选需要先在 iSAID 上训练模型，对初始模型质量有依赖

## 相关工作与启发

- **vs MTP**: MTP 使用有监督多任务预训练，S5 使用半监督预训练，不需要额外标注但效果更好（ViT-L: 75.21 vs 74.54 mAP on DIOR-R）
- **vs SAMRS**: SAMRS 依赖 SAM 生成 mask 构建标注，规模受限；S5 直接利用无标注数据，更易扩展
- **vs SelectiveMAE**: 两者都关注数据选择，但 SelectiveMAE 用于自监督预训练，S5 用于半监督预训练

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 S4 扩展为预训练范式是新思路，RS4P-1M 数据集构建方法实用
- 实验充分度: ⭐⭐⭐⭐ 6 个基准、多尺度模型、详细消融，实验全面
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，图表丰富，框架图直观
- 价值: ⭐⭐⭐⭐ 为遥感基础模型提供了新的预训练范式，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] From Softmax to Dirichlet: Evidential Learning for Semi-supervised Semantic Segmentation](../../CVPR2026/segmentation/from_softmax_to_dirichlet_evidential_learning_for_semi-supervised_semantic_segme.md)
- [\[AAAI 2026\] LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention](lwganet_addressing_spatial_and_channel_redundancy_in_remote_sensing_visual_tasks.md)
- [\[CVPR 2026\] ReSAM: Refine, Requery, and Reinforce: Self-Prompting Point-Supervised Segmentation for Remote Sensing Images](../../CVPR2026/segmentation/resam_refine_requery_and_reinforce_self-prompting_point-supervised_segmentation_.md)
- [\[ECCV 2024\] LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation](../../ECCV2024/segmentation/lass3d_language-assisted_semi-supervised_3d_semantic_segmentation_with_progressi.md)
- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)

</div>

<!-- RELATED:END -->
