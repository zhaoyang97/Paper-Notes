---
title: >-
  [论文解读] Angular Constraint Embedding via SpherePair Loss for Constrained Clustering
description: >-
  [NeurIPS 2025][自监督学习][约束聚类] 本文提出SpherePair损失函数，通过在角度空间（而非欧几里得空间）进行成对约束嵌入学习，实现了不依赖锚点(anchor)、不需要预知聚类数的深度约束聚类方法，并提供了严格的理论保证来确定最优超参数。 领域现状：深度约束聚类(DCC)通过整合领域知识（成对约束：同簇…
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "约束聚类"
  - "角度嵌入"
  - "SpherePair损失"
  - "深度约束聚类"
  - "球面表示"
---

# Angular Constraint Embedding via SpherePair Loss for Constrained Clustering

**会议**: NeurIPS 2025  
**arXiv**: [2510.06907](https://arxiv.org/abs/2510.06907)  
**代码**: 有（论文中提及代码仓库）  
**领域**: 聚类 / 表示学习  
**关键词**: 约束聚类, 角度嵌入, SpherePair损失, 深度约束聚类, 球面表示

## 一句话总结

本文提出SpherePair损失函数，通过在角度空间（而非欧几里得空间）进行成对约束嵌入学习，实现了不依赖锚点(anchor)、不需要预知聚类数的深度约束聚类方法，并提供了严格的理论保证来确定最优超参数。

## 研究背景与动机

**领域现状**：深度约束聚类(DCC)通过整合领域知识（成对约束：同簇/异簇）来改进无监督聚类。现有方法主要分两类：(1) 端到端DCC，引入锚点将聚类建模为伪分类任务；(2) 深度约束嵌入，学习编码约束信息的表示后再聚类。

**现有痛点**：端到端方法（如VanillaDCC、VolMaxDCC、CIDEC等）依赖锚点将局部成对关系间接推断为全局聚类分配，但锚点与真实聚类中心可能错位，且必须预知聚类数K。深度约束嵌入方法（如AutoEmbedder、CPAC）在欧几里得空间学习表示，但由于欧几里得距离范围为 $[0, +\infty)$，需要手动设置margin，且正负对之间的距离控制容易冲突。

**核心矛盾**：端到端方法的锚点机制与嵌入方法的欧几里得距离都存在根本缺陷——前者需要全局信息但只有局部约束，后者的无界距离空间导致超参数敏感且约束可能冲突。

**本文目标** (1) 如何在不使用锚点的情况下公平地编码正负约束？ (2) 如何消除手动调参（margin、嵌入维度）的需求？ (3) 如何在不知道K的情况下推断聚类数？

**切入角度**：作者观察到角度空间（cosine similarity）天然有界 $[0, \pi]$，利用球面上的几何性质可以保证等距聚类分布且不产生约束冲突。

**核心 idea**：在角度空间而非欧几里得空间进行约束嵌入学习，利用球面几何的封闭性消除约束冲突，并通过理论推导自动确定最优超参数。

## 方法详解

### 整体框架

输入是数据集和成对约束（正对：同簇，负对：异簇），通过深度自编码器学习角度嵌入空间，使同簇样本角度接近、异簇样本角度分离到负区(negative zone)。学完嵌入后，对归一化的球面表示 $\mathcal{Z}_{\text{sphere}}$ 应用K-means等标准聚类算法完成聚类，无需端到端联合训练。

### 关键设计

1. **SpherePair损失函数**:

    - 功能：在角度空间编码成对约束，驱动正对角度趋近0、负对角度趋近负区边界 $\pi/\omega$
    - 核心思路：对每对约束 $(a_i, b_i, y_i)$，计算嵌入向量间的角度 $\theta$，用余弦映射将角度归一化到相似度 $\text{Sim} \in [0,1]$，然后用逻辑回归损失 $\mathcal{L}_{\text{ang}}$ 优化。正对直接用 $\cos(\theta)$，负对用 $\cos(\min(\omega\theta, \pi))$，其中 $\omega$ 是角度因子，控制负区大小
    - 设计动机：角度距离有界 $[0, \pi]$，避免了欧几里得距离无界导致的归一化问题和margin敏感性。负区机制 $\pi/\omega$ 确保不同簇保持足够分离

2. **理论确定的最优超参数**:

    - 功能：从理论上确定角度因子 $\omega$ 和嵌入维度 $D$ 的最优值，消除手动调参
    - 核心思路：Theorem 4.4证明当 $D \geq K$ 时，$\omega \geq \pi/\arccos(-1/(K-1))$ 均有效；Corollary 4.5进一步证明当 $D$ 足够大时，$\omega = 2$ 是最优选择（对任意K普遍有效）。此时嵌入中K个簇在(K-1)维子空间形成正则单纯形
    - 设计动机：理论保证了SpherePair在各种数据集上无需针对不同K调整 $\omega$，只需固定 $\omega=2$；$D \geq K$ 的要求在实际中很容易满足，即使K未知

3. **基于PCA的聚类数推断**:

    - 功能：在K未知时，从学好的球面表示中推断聚类数
    - 核心思路：基于Theorem 4.6的角度不变性——当PCA投影维度 $d \geq K-1$ 时，跨簇角度保持不变。计算负约束涉及的样本在不同维度子空间投影中的最小簇间角度 $\delta_d$，找到序列 $\{\bar{\delta}_d\}$ 的平台起点 $d^* = K-1$，即可推断 $\hat{K} = d^* + 1$
    - 设计动机：利用球面表示的几何结构直接推断K，无需像端到端方法那样重新训练或做繁琐的后验聚类验证

### 损失函数 / 训练策略

总体损失为 $\mathcal{L} = \mathcal{L}_{\text{ang}} + \lambda \mathcal{L}_{\text{recon}}$，其中重建损失 $\mathcal{L}_{\text{recon}}$ 防止退化表示。嵌入在解码前归一化以保持角度性质。默认 $\lambda = 0.02$，$\omega = 2$（理论确定），$\rho = 0.05$。使用Adam优化器，预训练堆叠去噪自编码器进行初始化。

## 实验关键数据

### 主实验

在8个基准上与7种DCC基线对比（ACC/NMI/ARI，约束数1k/5k/10k）：

| 数据集 | 约束数 | 指标 | SpherePair | 次优基线 | 提升 |
|--------|--------|------|-----------|---------|------|
| CIFAR100-20 | 10k | ACC | 62.8/62.6 | 54.6 (VanillaDCC) | +8.2 |
| CIFAR10 | 10k | ACC | 90.5/89.9 | 90.1 (CIDEC) | +0.4 |
| FashionMNIST | 10k | ACC | 84.8/83.6 | 81.5 (SDEC) | +3.3 |
| ImageNet10 | 1k | ACC | 92.1/91.7 | 94.3 (SDEC) | 接近SOTA |

### 消融实验

| 配置 | 关键特性 | 说明 |
|------|---------|------|
| SpherePair (full) | $\omega=2$, $D=10/20$ | 完整模型，所有数据集最优 |
| SpherePair† (无预训练) | 随机初始化 | 仍优于多数预训练基线，证明角度嵌入学习本身的强大 |
| AutoEmbedder | 欧几里得距离+手动margin | 表现最差的嵌入方法之一，需大量调参 |
| 不平衡约束(IMB1/IMB2) | 约束分布偏斜 | SpherePair在IMB条件下性能稳定，端到端方法严重退化 |

### 关键发现

- **SpherePair在所有数据集和所有约束规模上均取得最优或近最优结果**，特别是在低约束量(1k)和不平衡约束条件下优势更明显
- **泛化能力强**：训练集和测试集性能差距极小（通常<1%），说明学到的表示能很好地泛化到未见样本
- **聚类数推断准确**：PCA平台检测方法在多数数据集上能准确推断K值
- **无预训练的SpherePair†仍优于大多数预训练基线**，证明角度约束嵌入的方法论优势不依赖于初始化

## 亮点与洞察

- **从欧几里得到角度空间的范式转变**：最核心的洞察是角度空间的有界性 $[0,\pi]$ 天然解决了欧几里得空间的margin敏感性和约束冲突问题。这种思路可以迁移到其他需要度量学习的任务，如半监督学习、少样本学习中的原型网络
- **理论驱动的超参数消除**：通过严格数学推导将 $\omega$ 固定为2、$D \geq K$ 即可，这在深度学习研究中少见。从正则单纯形几何出发推导出的条件，消除了传统方法中最头疼的调参环节
- **将表示学习与聚类彻底解耦**：学完嵌入后直接用K-means，且cosine和欧几里得距离在归一化特征上等价，实践中极其便利。这种思路对大规模数据集的可扩展性有重要意义

## 局限与展望

- **理论保证依赖约束质量**：如果负约束未覆盖所有真实簇，聚类数推断会失败。实际中约束的采样分布是否合理是一个实践挑战
- **全连接网络架构限制**：为公平比较，实验使用了简单的全连接编码器。在更复杂的数据（如高分辨率图像）上是否需要更强的backbone尚未验证
- **假设数据有明确的簇结构**：对于连续分布或层次结构的数据，等距球面嵌入的假设可能不成立
- **约束获取成本未讨论**：论文假设约束可用，但在实际场景中获取高质量成对约束本身可能是瓶颈

## 相关工作与启发

- **vs VanillaDCC/VolMaxDCC**: 这些端到端方法通过锚点连接表示和聚类分配，但锚点错位会传播误差。SpherePair完全去除锚点，用角度嵌入后K-means的两阶段方式避免了这个问题
- **vs AutoEmbedder**: 同样是无锚点的约束嵌入，但AutoEmbedder在欧几里得空间需要手动设置margin。SpherePair用角度空间的有界性替代了margin，$\omega$ 由理论确定
- **vs 监督角度学习（ArcFace等）**: 人脸识别中的角度损失依赖类标签和类中心原型。SpherePair首次将角度学习应用于只有成对约束的弱监督场景，不需要类中心

## 评分

- 新颖性: ⭐⭐⭐⭐ 角度空间约束嵌入是该领域首创，理论基础扎实
- 实验充分度: ⭐⭐⭐⭐⭐ 8个数据集、7种基线、多种约束条件和超参数鲁棒性分析非常全面
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但数学符号密集，可读性对非专业读者有挑战
- 价值: ⭐⭐⭐⭐ 提供了约束聚类的新范式，理论贡献和实用性兼具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ReSA: Clustering Properties of Self-Supervised Learning](../../ICML2025/self_supervised/clustering_properties_of_self-supervised_learning.md)
- [\[CVPR 2026\] Reframing Long-Tailed Learning via Loss Landscape Geometry](../../CVPR2026/self_supervised/reframing_long-tailed_learning_via_loss_landscape_geometry.md)
- [\[ICML 2026\] Provable Accuracy Collapse in Embedding-Based Representations under Dimensionality Mismatch](../../ICML2026/self_supervised/provable_accuracy_collapse_in_embedding-based_representations_under_dimensionali.md)
- [\[CVPR 2026\] Franca: Nested Matryoshka Clustering for Scalable Visual Representation Learning](../../CVPR2026/self_supervised/franca_nested_matryoshka_clustering_for_scalable_visual_representation_learning.md)
- [\[ICLR 2026\] Architecture-Agnostic Test-Time Adaptation via Backprop-Free Embedding Alignment](../../ICLR2026/self_supervised/architecture-agnostic_test-time_adaptation_via_backprop-free_embedding_alignment.md)

</div>

<!-- RELATED:END -->
