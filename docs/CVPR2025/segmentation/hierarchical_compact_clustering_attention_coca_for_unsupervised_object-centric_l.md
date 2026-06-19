---
title: >-
  [论文解读] Hierarchical Compact Clustering Attention (COCA) for Unsupervised Object-Centric Learning
description: >-
  [CVPR 2025][语义分割][无监督目标发现] COCA-Net 提出基于物理紧凑性（compactness）的层级聚类注意力层，通过自底向上的层级合并策略发现物体中心，解决了 Slot Attention 在初始化敏感性、slot 数量预设和背景分割等方面的固有缺陷，在六个无监督物体发现数据集上达到 SOTA。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "无监督目标发现"
  - "目标中心学习"
  - "层级聚类"
  - "紧凑性"
  - "注意力机制"
---

# Hierarchical Compact Clustering Attention (COCA) for Unsupervised Object-Centric Learning

**会议**: CVPR 2025  
**arXiv**: [2505.02071](https://arxiv.org/abs/2505.02071)  
**代码**: 无  
**领域**: 分割 / 无监督学习  
**关键词**: 无监督目标发现, 目标中心学习, 层级聚类, 紧凑性, Slot Attention

## 一句话总结

COCA-Net 提出基于物理紧凑性（compactness）的层级聚类注意力层，通过自底向上的层级合并策略发现物体中心，解决了 Slot Attention 在初始化敏感性、slot 数量预设和背景分割等方面的固有缺陷，在六个无监督物体发现数据集上达到 SOTA。

## 研究背景与动机

**领域现状**：目标中心学习（OCL）旨在以无监督方式将场景分解为独立物体表示（slot），Slot Attention（SA）及其变体是主流方法。SA 通过类似软 K-Means 的迭代注意力机制将像素分配给 slot。

**现有痛点**：(1) **初始化敏感**：SA 从同一潜在分布初始化所有 slot，导致路由问题（routing problem），多个 slot 绑定到同一物体；(2) **slot 数量需预设**：预定义数量不当会导致过分割或欠分割；(3) **背景分割差**：SA 难以有效处理散布的、非紧凑的背景区域。

**核心矛盾**：SA 继承了 K-Means 的根本局限—对初始化敏感、需预定簇数、假设簇形状和大小，这些限制在复杂场景中表现尤其严重。

**切入角度**：层级凝聚聚类（HAC）天然具有输出簇数灵活、对噪声鲁棒、可捕获层级关系等优点，但在 OCL 领域尚未被充分挖掘。物理学中的紧凑性度量可作为空间归纳偏置区分前景（紧凑）和背景（分散）。

**核心 idea**：将物理紧凑性度量引入层级凝聚聚类框架，构建无需预设 slot 数量的自底向上物体发现网络。

## 方法详解

### 整体框架

COCA-Net 采用编码器-解码器架构：编码器由多个 COCA 层级联组成，自底向上逐层聚类像素特征到物体 slot；解码器使用 Spatial Broadcast Decoder（SBD）从 slot 重建图像。训练目标为像素重建的 MSE 损失。

### 关键设计

1. **紧凑性评分 (Compactness Scoring)**:

    - 功能：为每个候选物体 mask 计算紧凑性分数，作为空间归纳偏置引导聚类
    - 核心思路：基于转动惯量（Moment of Inertia, MI）的质量归一化紧凑性度量。对每个节点 $i$ 的亲和力 mask $\boldsymbol{\Lambda}_i$ 计算 $\mathcal{C}^i(\boldsymbol{\Lambda}_i) = \frac{I^\mu(\Theta_\mu)}{I^\mu(\boldsymbol{\Lambda})}$，即该 mask 的 MI 与同面积圆的 MI 之比。圆（最紧凑形状）得分为 1，分散形状得分接近 0
    - 设计动机：前景物体通常具有紧凑凸形，背景元素往往分散带孔洞。MI 紧凑性度量具有可加性、尺度不变性，且当以形状质心为轴时 MI 最小→紧凑性最大，自然定位物体中心

2. **顺序物体中心发现 (Sequential Object Centroid Discovery)**:

    - 功能：在每个窗口内迭代发现并分离物体簇，无需预设簇数
    - 核心思路：基于 Stick-Breaking Clustering (SBC) 变体——初始化全 1 的 scope 张量 $\mathbf{Z}$，每次迭代选择紧凑性最高的节点 $\omega_m$ 作为锚点，其亲和力 mask 经 scope 遮蔽后成为输出簇 $\Pi_m$，再更新 scope 排除已聚类节点。紧凑性评分仅需在聚类循环前计算一次
    - 设计动机：利用 MI 紧凑性的性质——物体质心处紧凑性最高——自然分离不同物体中心。所有亲和力 mask 的紧凑性并行计算，比 Sequential Slot 方法高效

3. **层级池化与跳跃连接 (Hierarchical Pool, Aggregate & Skip Connect)**:

    - 功能：跨层级传递特征和聚类结构，构建物体表示的层级树
    - 核心思路：每个 COCA 层将输入节点划分为不重叠窗口，在窗口内并行执行聚类和池化。层间通过合并相邻两层的聚类 mask 实现跳跃连接，使特征在每隔一层间直接传递。最终层的聚类构成物体 slot，整个层级聚类结构形成 dendrogram 树
    - 设计动机：非重叠窗口划分使聚类可并行执行；跳跃连接促进深层网络的无监督特征学习；dendrogram 可同时从编码器端评估分割质量

### 损失函数 / 训练策略

- 唯一优化目标为像素重建的 MSE 损失，从零开始无监督训练
- 像素特征编码器为简单的逐点处理骨干网络（每像素独立编码外观+位置信息）
- 为层级中每个像素维护 5 个物理属性：面积、质量、密度、转动惯量、均值位置
- 所有方法统一使用 SBD 解码器，slot 数量设为数据集中物体最大数量

## 实验关键数据

### 主实验

6 个数据集上的解码器端前景物体分割（ARI/mSC，3 种子均值±标准差）：

| 数据集 | 方法 | ARI↑ | mSC↑ |
|-------|------|------|------|
| Multi-dSprites | BOQSA | 0.91±0.01 | 0.89±0.01 |
| Multi-dSprites | **COCA-Net** | **0.93±0.01** | **0.91±0.01** |
| ObjectsRoom | INVSA | 0.88±0.00 | 0.80±0.01 |
| ObjectsRoom | **COCA-Net** | **0.88±0.00** | 0.82±0.01 |
| ShapeStacks | BOQSA | 0.83±0.09 | 0.80±0.09 |
| ShapeStacks | **COCA-Net** | **0.91±0.01** | **0.85±0.01** |
| CLEVR6 | INVSA | 0.96±0.01 | 0.87±0.03 |
| CLEVR6 | **COCA-Net** | **0.98±0.00** | **0.92±0.01** |

### 含背景分割关键数据

| 数据集 | 方法 | ARI↑ (含BG) | mSC↑ (含BG) |
|-------|------|------------|------------|
| ObjectsRoom | INVSA | 0.66±0.12 | 0.68±0.07 |
| ObjectsRoom | **COCA-Net** | **0.95±0.01** | **0.87±0.02** |
| Multi-dSprites | BOQSA | 0.34±0.06 | 0.56±0.02 |
| Multi-dSprites | **COCA-Net** | **0.98±0.00** | **0.96±0.00** |

### 编码器端评估

COCA-Net 编码器端分割同样远超竞争方法，如 ShapeStacks 上 ENC-FG ARI 0.82 vs BOQSA 的 0.49（提升约 67%）。

### 关键发现

- COCA-Net 在几乎所有数据集和指标上优于或持平 SOTA，且方差极小（鲁棒性强）
- 含背景分割中表现尤其突出，在 ObjectsRoom 上比第二名提升约 30% ARI
- 编码器端即可产出高质量分割 mask，表明 COCA-Net 编码器可独立作为目标中心特征提取器
- 唯一弱点是 ShapeStacks 含背景 ARI（0.31），因该数据集将所有背景归为单一 GT mask，而 COCA-Net 会合理分割多个背景区域

## 亮点与洞察

1. **物理直觉与深度学习的优雅结合**：紧凑性是一个直觉清晰的物理量，将其作为可微分操作嵌入网络，比纯数据驱动方法更具可解释性
2. **从 K-Means 到 HAC 的范式转变**：跳出 Slot Attention 系列的 K-Means 框架，用层级凝聚聚类避免了初始化敏感、簇数预设等根本问题
3. **编码器端高质量分割**：现有方法几乎只评估解码器端 mask，COCA-Net 编码器的 dendrogram 本身即提供高质量结果，为下游任务提供特征提取器的可能

## 局限与展望

- 目前仅在合成数据集上验证，尚未展示在真实世界复杂场景的表现
- 层级结构的超参数（窗口大小、层数等）需要针对数据集调整
- 紧凑性假设对凹形或环形物体可能不成立
- 可探索将 COCA 编码器与更强的解码器（如扩散模型）结合

## 相关工作与启发

- **Slot Attention**：本文的直接对比对象，COCA 解决了其 routing problem
- **GENv2 / BOQSA**：Sequential Slot 模型和改进型 SA，均为本文主要 baseline
- **TokenCut**：基于图聚类的无监督目标发现，使用预训练骨干
- 启发：紧凑性这一物理归纳偏置可能对其他需要空间先验的视觉任务有参考价值

## 评分

⭐⭐⭐⭐ — 在无监督目标中心学习领域提出了新颖且有效的范式转变，物理直觉与网络设计结合优雅，实验充分但缺乏真实场景验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Scene-Centric Unsupervised Panoptic Segmentation](scene-centric_unsupervised_panoptic_segmentation.md)
- [\[CVPR 2025\] Style-Editor: Text-driven Object-Centric Style Editing](style-editor_text-driven_object-centric_style_editing.md)
- [\[ICCV 2025\] Ensemble Foreground Management for Unsupervised Object Discovery](../../ICCV2025/segmentation/ensemble_foreground_management_for_unsupervised_object_discovery.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] Revisiting Audio-Visual Segmentation with Vision-Centric Transformer](revisiting_audio-visual_segmentation_with_vision-centric_transformer.md)

</div>

<!-- RELATED:END -->
