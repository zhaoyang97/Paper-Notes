---
title: >-
  [论文解读] A Flag Decomposition for Hierarchical Datasets
description: >-
  [CVPR 2025][图像恢复][Flag流形] 本文提出Flag Decomposition (FD)，一种将层次结构数据分解为保持层级关系的flag流形表示（Stiefel坐标）的算法，在去噪、聚类和少样本学习任务中展示了相比SVD等标准方法的优势。 领域现状：层次结构广泛存在于数据中——分类体系、光谱层级、特征层级等…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "Flag流形"
  - "层次数据"
  - "矩阵分解"
  - "去噪"
  - "少样本学习"
---

# A Flag Decomposition for Hierarchical Datasets

**会议**: CVPR 2025  
**arXiv**: [2502.07782](https://arxiv.org/abs/2502.07782)  
**代码**: [https://github.com/nmank/FD](https://github.com/nmank/FD)  
**领域**: 图像恢复 / 子空间学习  
**关键词**: Flag流形, 层次数据, 矩阵分解, 去噪, 少样本学习

## 一句话总结
本文提出Flag Decomposition (FD)，一种将层次结构数据分解为保持层级关系的flag流形表示（Stiefel坐标）的算法，在去噪、聚类和少样本学习任务中展示了相比SVD等标准方法的优势。

## 研究背景与动机

**领域现状**：层次结构广泛存在于数据中——分类体系、光谱层级、特征层级等。Flag流形编码嵌套子空间序列，已在运动平均、子空间聚类、降维等任务中展示了潜力。

**现有痛点**：现有flag流形应用大多限于使用SVD等标准矩阵分解方法来提取flag，没有专门用于分解层次数据并保持其层级结构的通用算法。标准降维方法（如PCA）会展平层级结构，丢失本质的部分-整体关系信息。

**核心矛盾**：数据中的层级关系编码了重要的结构信息（如粗到细的邻域、光谱带的组织方式），但标准线性代数工具将所有维度等同对待，无法保持这些关系。

**本文目标**：设计一个general算法，将带有层级索引标注的数据矩阵分解为保持层级的flag表示，并展示其在多种应用中的优势。

**切入角度**：利用flag流形的数学结构——嵌套子空间序列 $\mathcal{V}_1 \subset \mathcal{V}_2 \subset \cdots \subset \mathcal{V}_k$ 天然对应层级数据的列索引层级。

**核心 idea**：给定数据矩阵 $\mathbf{D}$ 及其列的层级分组 $\mathcal{A}_1 \subset \mathcal{A}_2 \subset \cdots \subset \mathcal{A}_k$，将其分解为 $\mathbf{D}\mathbf{P} = \mathbf{Q}\mathbf{R}$，其中 $\mathbf{Q}$ 是Stiefel坐标表示的flag，$\mathbf{R}$ 是块上三角矩阵，$\mathbf{P}$ 是排列矩阵。这种分解使得 $[[\mathbf{Q}]]$ 的嵌套子空间层级精确对应数据的列层级。

## 方法详解

### 整体框架
输入：数据矩阵 $\mathbf{D} \in \mathbb{R}^{n \times p}$ + 列层级 $\mathcal{A}_1 \subset \cdots \subset \mathcal{A}_k$。算法按层级分组将列重排，对每一层逐步执行正交化（去除已有子空间的投影后做QR分解），得到保持层级的flag坐标 $[\![\mathbf{Q}]\!]$。应用：去噪（在flag上投影）、聚类（用flag距离）、少样本学习（flag作为原型）。

### 关键设计

1. **层级保持的逐层正交化**:

    - 功能：将数据矩阵分解为保持嵌套子空间的flag坐标
    - 核心思路：第 $i$ 层的新列 $\mathbf{B}_i$ 先投影到前 $i-1$ 层子空间的正交补空间，再对投影后的结果做QR分解得到 $\mathbf{Q}_i$。最终 $\mathbf{Q} = [\mathbf{Q}_1 | \mathbf{Q}_2 | \cdots | \mathbf{Q}_k]$ 保证了 $[\mathbf{Q}_1] \subset [\mathbf{Q}_1, \mathbf{Q}_2] \subset \cdots$
    - 设计动机：标准QR分解不考虑列的层级分组，FD通过分组投影顺序确保层级结构被保持

2. **Flag距离用于鲁棒聚类**:

    - 功能：利用flag流形上的几何距离改善子空间聚类
    - 核心思路：flag弦距离（chordal distance）同时考虑嵌套子空间各层的对齐程度，比Grassmannian距离（只看整体子空间）提供更细粒度的相似度度量。对含噪声/异常值的数据，flag表示天然更鲁棒
    - 设计动机：层级信息是额外约束，利用它可以更好区分噪声和信号

3. **Flag原型用于少样本学习**:

    - 功能：用flag替代单一原型向量作为类别表示
    - 核心思路：对每个类别的少量样本特征构建flag原型（利用特征层级），分类时计算测试样本到各flag原型的距离。flag原型包含比单一向量更丰富的类别子空间结构
    - 设计动机：少样本学习中样本量小，保留特征的层级结构可以提供更有信息量的类别表示

### 损失函数 / 训练策略
不适用于训练——FD是确定性矩阵分解算法。在少样本学习中，使用flag弦距离作为最近邻分类的度量。

## 实验关键数据

### 主实验

| 应用 | 方法 | 效果 |
|---|---|---|
| 高光谱去噪 | SVD重建 | 基线 |
| 高光谱去噪 | **FD重建** | **更好（保留光谱层级）** |
| 子空间聚类（含噪） | Grassmannian距离 | 噪声敏感 |
| 子空间聚类（含噪） | **Flag距离** | **更鲁棒** |
| 少样本分类 | 单向量原型 | 基线 |
| 少样本分类 | **Flag原型** | **更高准确率** |

### 消融实验

| 配置 | 效果 | 说明 |
|---|---|---|
| SVD（无层级感知） | 基线 | 丢失层级结构 |
| FD（保持层级） | 更优 | 层级信息有价值 |
| Flag距离 vs Grassmannian距离 | Flag更优 | 多层对齐更有区分度 |

### 关键发现
- 在含噪数据上，FD的聚类性能显著优于SVD——层级约束提供了对噪声的天然鲁棒性
- 高光谱去噪中保留光谱层级比直接截断SVD奇异值更合理
- 少样本学习中flag原型一致优于标准原型网络
- 算法计算开销与QR分解相当，可用于大规模数据

## 亮点与洞察
- **flag流形的应用推广**：将抽象的微分几何概念"落地"为有实用价值的数据分析工具。FD算法简洁实用
- **层级结构的数学形式化**：用嵌套子空间严格表达"部分到整体"关系，比临时性的层级处理更有理论保证
- **广泛适用性**：同一个分解框架适用于去噪、聚类、少样本学习三种不同任务

## 局限与展望
- 需要预先知道数据的层级结构（列的分组方式）——无法自动发现层级
- 假设层级对应线性子空间的嵌套，非线性层级关系无法直接处理
- 目前仅在中小规模数据上验证，大规模深度学习场景的效果有待测试
- 可探索与深度学习特征提取器结合，自动构建特征层级

## 相关工作与启发
- **vs SVD/PCA**: 不保持层级结构；FD在有层级的数据上严格优于
- **vs Grassmannian方法**: 仅考虑单一子空间；flag考虑嵌套子空间序列，信息更丰富
- flag流形视角可迁移到任何具有天然层级的数据分析任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个通用的层级保持flag分解算法
- 实验充分度: ⭐⭐⭐ 涵盖三种应用但每种深度有限
- 写作质量: ⭐⭐⭐⭐ 数学严谨，直觉示例清晰
- 价值: ⭐⭐⭐⭐ 提供了处理层次数据的新数学工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Luminance-Aware Statistical Quantization: Unsupervised Hierarchical Learning for Illumination Enhancement](../../NeurIPS2025/image_restoration/luminance-aware_statistical_quantization_unsupervised_hierarchical_learning_for_.md)
- [\[ICLR 2026\] ProtoTS: Learning Hierarchical Prototypes for Explainable Time Series Forecasting](../../ICLR2026/image_restoration/protots_learning_hierarchical_prototypes_for_explainable_time_series_forecasting.md)
- [\[NeurIPS 2025\] scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy](../../NeurIPS2025/image_restoration/scsplit_bringing_severity_cognizance_to_image_decomposition_in_fluorescence_micr.md)
- [\[NeurIPS 2025\] Learning Cocoercive Conservative Denoisers via Helmholtz Decomposition for Poisson Inverse Problems](../../NeurIPS2025/image_restoration/learning_cocoercive_conservative_denoisers_via_helmholtz_decomposition_for_poiss.md)
- [\[CVPR 2026\] HiDRA: Hierarchical Degradation Representation and Adaptation with Generative Priors for Enhancing Infrared Vision](../../CVPR2026/image_restoration/hidra_hierarchical_degradation_representation_and_adaptation_with_generative_pri.md)

</div>

<!-- RELATED:END -->
