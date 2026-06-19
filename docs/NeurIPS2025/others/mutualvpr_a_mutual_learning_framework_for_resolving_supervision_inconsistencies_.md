---
title: >-
  [论文解读] MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering
description: >-
  [NeurIPS 2025][Visual Place Recognition] 提出 MutualVPR 互学习框架，通过特征驱动的自适应 K-means 聚类动态分配场景类别标签，解决分类式 VPR 方法中由视角变化和遮挡导致的监督不一致问题。 视觉位置识别（VPR）：要求从查询图像匹配出已访问位置，是自动驾驶和机器人长…
tags:
  - "NeurIPS 2025"
  - "Visual Place Recognition"
  - "自适应聚类"
  - "互学习"
  - "监督一致性"
  - "DINOv2"
---

# MutualVPR: A Mutual Learning Framework for Resolving Supervision Inconsistencies via Adaptive Clustering

**会议**: NeurIPS 2025  
**arXiv**: [2412.09199](https://arxiv.org/abs/2412.09199)  
**代码**: [有](https://github.com/Gucci233/MutualVPR)  
**领域**: 视觉定位 / 图像检索  
**关键词**: Visual Place Recognition, 自适应聚类, 互学习, 监督一致性, DINOv2

## 一句话总结

提出 MutualVPR 互学习框架，通过特征驱动的自适应 K-means 聚类动态分配场景类别标签，解决分类式 VPR 方法中由视角变化和遮挡导致的监督不一致问题。

## 研究背景与动机

**视觉位置识别（VPR）** 要求从查询图像匹配出已访问位置，是自动驾驶和机器人长期定位的核心组件。

**分类式 VPR** 方法（如 CosPlace、EigenPlaces）将环境按地理坐标划分为网格，每个网格一个类别标签，使用分类损失训练。这避免了对比学习中昂贵的难样本挖掘。但存在严重的**监督不一致问题**：

**视角变化 a)**：同一位置不同朝向拍摄的图像视觉内容差异巨大，但被分配同一标签

**视角变化 b)**：视觉相似但朝向标签不同的图像被分到不同类

**遮挡问题**：建筑物、车辆等遮挡导致面向同一参考点的图像内容差异大（EigenPlaces 的假设失效）

**CosPlace** 依赖人工标注朝向标签划分类别 → 不保证类内视觉一致性

**EigenPlaces** 用 SVD 分解假设面向同一参考点的图像相似 → 在有遮挡的城市环境中失效

**核心洞察**：有效的 VPR 监督应反映**语义相似性**，而非仅依赖空间邻近或固定朝向假设。

## 方法详解

### 整体框架

MutualVPR 包含**特征提取**和**自适应聚类**两个交替优化的阶段（图3）：

- **阶段1**：更新后的聚类标签监督特征学习
- **阶段2**：改进后的特征指导重新聚类

两者共同演化（co-evolution），逐步消除监督不一致。

### 关键设计

1. **特征编码器**：

    - 基于 DINOv2 ViT-B/14 骨干，冻结预训练参数
    - 引入 **MulConv Adapter**：瓶颈结构 + 三个并行卷积分支（不同感受野），提取多尺度特征
    - GeM 池化 + 全连接层降维至 512 维描述子
    - Adapter 通过可学习缩放因子 $s$ 控制贡献：$z_l = \text{MLP}(\text{LN}(z_l')) + s \cdot \text{Adapter}(\text{LN}(z_l')) + z_l'$

2. **自适应聚类的互学习**：

    - **初始化**：按 UTM 坐标划分粗粒度位置网格 $x = \{\lfloor east/M \rfloor, \lfloor north/M \rfloor\}$
    - **网格内细分**：在每个 UTM 区域内，基于学习到的描述子进行迭代 K-means 聚类
    - 类别定义：$C = \{e_i, n_j, h \mid e_i, n_j \in x, h \in K\}$，$K$ 控制视角方向的划分粒度
    - **核心**：特征编码器和聚类过程**迭代互更新**，避免早期错误分配的误差累积
    - 分类损失采用 **LMCL（Large Margin Cosine Loss）**

3. **多角度裁剪策略**：

    - 从全景图按不同起始角度（0° 和 30°）每隔 60° 裁剪，生成训练数据
    - 增加语义连续性，相邻裁剪图像自然共享语义内容

### 训练策略

- 图像尺寸 504×504，描述子维度 512，网格大小 $M=10$，聚类数 $K=3$
- 特征提取器学习率 1e-5，分类器学习率 1e-2，Adam 优化器 + 余弦退火
- 训练 50 个 epoch，每个 epoch 10,000 次迭代
- 将所有 UTM 类分为8组轮替训练，每个 epoch 随机选择 1/5 进行聚类

## 实验关键数据

### 主实验——多基准对比（表1）

| 方法 | 维度 | 训练集 | MSLS-val R@1 | Pitts30k R@1 | Tokyo24/7 R@1 | SF-XL-test R@1 |
|------|------|--------|------------|------------|-------------|--------------|
| CosPlace | 512 | SF-XL | 84.4 | 89.6 | 76.5 | 64.8 |
| EigenPlaces | 512 | SF-XL | 88.1 | 92.3 | 84.8 | **83.8** |
| MixVPR | 4096 | GSV-Cities | 87.1 | 91.6 | 87.0 | 69.2 |
| SALAD+CM | 512+32 | MSLS+GSV | **90.4** | 90.9 | 92.8 | 78.4 |
| BoQ | 512 | GSV-Cities | 88.4 | **93.1** | 91.9 | 79.6 |
| **MutualVPR** | **512** | **SF-XL** | 89.2 | 90.9 | 92.4 | 80.8 |

### 遮挡鲁棒性实验——SF-XL-Occlusion（表2）

| 方法 | R@1 | R@5 | R@10 | R@20 |
|------|-----|-----|------|------|
| CosPlace | 32.9 | 43.4 | 46.1 | 48.7 |
| EigenPlaces | 36.8 | 51.8 | 56.6 | 59.2 |
| MixVPR | 30.3 | 35.5 | 38.2 | 44.7 |
| SALAD+CM | 40.8 | 53.7 | 58.3 | 61.3 |
| **MutualVPR** | **47.4** | **65.8** | **71.1** | **73.7** |

MutualVPR 在遮挡场景下 R@1 **领先第二名**（CricaVPR1/SALAD+CM 40.8%）**6.6个百分点**。

### 消融——聚类数 $K$ 的影响（表4）

| 骨干 | K | Tokyo24/7 R@1 | SF-XL-test R@1 |
|------|---|-------------|--------------|
| DINOv2 | 1（无聚类） | 80.6 | 61.1 |
| DINOv2 | 3 | **91.1** | **77.0** |
| DINOv2 | 6 | 86.0 | 70.9 |
| ResNet50 | 1 | 68.3 | 52.1 |
| ResNet50 | 6 | **82.9** | **74.5** |

### 关键发现

- 不使用聚类（$K=1$）性能最差，证明视角方向分类的有效性
- DINOv2 骨干最优 $K=3$，ResNet50 最优 $K=6$——最优聚类数取决于骨干和数据
- 自适应聚类 vs 固定朝向标签：即使不使用多角度裁剪，MutualVPR（DINOv2, 0°）R@1=91.1 vs CosPlace（DINOv2, 0°）R@1=90.2
- 遮挡场景的大幅领先源于两个机制：**自适应监督纠正**（迭代纠正初始错误分配）和**类内语义紧凑性**（语义相似视图在特征空间中更近）

## 亮点与洞察

- **问题分析深入**：通过 t-SNE 可视化清晰展示了三类监督不一致的具体表现
- **简洁有效**：仅用 K-means 聚类和特征学习的交替迭代就解决了分类式 VPR 的核心问题
- **无需朝向标签**：完全依赖视觉语义自组织，比 CosPlace 的人工标签和 EigenPlaces 的 SVD 假设更灵活
- 仅用 SF-XL 训练，就在多个基准上与用 GSV-Cities（更大更干净）训练的对比学习方法相当

## 局限与展望

- 聚类数 $K$ 为固定超参数，未能随数据特征自适应调整
- 互学习的收敛性缺乏理论保证
- 仅验证了二维图像（全景裁剪），未尝试 3D 点云等其他模态
- 最优 $K$ 对骨干网络敏感，缺乏自动选择策略

## 相关工作与启发

- **对比学习 VPR**（NetVLAD、MixVPR、CricaVPR）依赖精心构建的训练数据避免监督不一致
- **分类式 VPR**（CosPlace、EigenPlaces）高效但受固定标签限制
- 互学习/自训练范式在半监督学习中广泛使用，本文将其巧妙应用于 VPR 的标签细化
- 启发：其他依赖粗粒度标签监督的检索任务可借鉴类似的自适应聚类策略

## 评分

- 新颖性: ⭐⭐⭐⭐ — 互学习框架和自适应聚类思路简洁优雅
- 实验充分度: ⭐⭐⭐⭐ — 多基准验证充分，遮挡实验有说服力，消融完整
- 写作质量: ⭐⭐⭐⭐ — 问题分析清晰，可视化直观
- 价值: ⭐⭐⭐⭐ — 有效解决分类式 VPR 的痛点，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Improving Accuracy and Calibration via Differentiated Deep Mutual Learning](../../CVPR2025/others/improving_accuracy_and_calibration_via_differentiated_deep_mutual_learning.md)
- [\[CVPR 2025\] Instance-wise Supervision-level Optimization in Active Learning](../../CVPR2025/others/instance-wise_supervision-level_optimization_in_active_learning.md)
- [\[ACL 2025\] Adaptive Feature-based Low Rank Plus Sparse Decomposition for Subspace Clustering](../../ACL2025/others/adaptive_feature-based_low_rank_plus_sparse_decomposition_for_subspace_clusterin.md)
- [\[NeurIPS 2025\] Coresets for Clustering Under Stochastic Noise](coresets_for_clustering_under_stochastic_noise.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)

</div>

<!-- RELATED:END -->
