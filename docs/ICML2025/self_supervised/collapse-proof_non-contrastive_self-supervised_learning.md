---
title: >-
  [论文解读] Collapse-Proof Non-Contrastive Self-Supervised Learning
description: >-
  [ICML 2025][自监督学习][非对比自监督学习] 提出 FALCON 方法，基于超维计算 (hyperdimensional computing) 原理设计投影器和损失函数，理论证明可同时避免四种已知训练失败模式（表示崩塌、维度崩塌、聚类崩塌、簇内崩塌），并使表征自然具备去相关和聚类特性。 自监督学习 (SSL) 虽…
tags:
  - "ICML 2025"
  - "自监督学习"
  - "非对比自监督学习"
  - "崩塌避免"
  - "超维计算"
  - "特征去相关"
  - "聚类表示"
---

# Collapse-Proof Non-Contrastive Self-Supervised Learning

**会议**: ICML 2025

**arXiv**: [2410.04959](https://arxiv.org/abs/2410.04959)

**作者**: Emanuele Sansone, Tim Lebailly, Tinne Tuytelaars (KU Leuven)

**领域**: 自监督学习

**关键词**: 非对比自监督学习, 崩塌避免, 超维计算, 特征去相关, 聚类表示

## 一句话总结

提出 FALCON 方法，基于超维计算 (hyperdimensional computing) 原理设计投影器和损失函数，理论证明可同时避免四种已知训练失败模式（表示崩塌、维度崩塌、聚类崩塌、簇内崩塌），并使表征自然具备去相关和聚类特性。

## 研究背景与动机

自监督学习 (SSL) 虽在无标签数据上取得巨大成功，但训练过程中存在多种 **失败模式 (failure modes)**，限制了方法的可靠性和广泛应用：

**表示崩塌 (Representation Collapse)**: 所有输入的表示坍缩到同一常向量

**维度崩塌 (Dimensional Collapse)**: 嵌入仅占据向量空间的低维子空间

**聚类崩塌 (Cluster Collapse)**: 数据点仅被分配到可用原型的子集

**簇内崩塌 (Intracluster Collapse)**: 同一聚类内样本的表示差异趋近于零

现有方法（如动量编码器、停止梯度、非对称投影头等）都是启发式策略，无法从理论上保证避免所有崩塌。**特征去相关方法**（如 Barlow Twins）和**聚类方法**（如 SwAV）分别处理部分崩塌，但尚无统一解决方案。

本文目标：找到一组**充分条件**，保证同时避免所有四种崩塌，并据此设计简洁的投影器和损失函数。

## 方法详解

### 整体框架

FALCON (FAiLure-proof non-CONtrastive SSL) 方法包含以下流程：

1. 对无标签数据生成增强对 $(X, X')$
2. 编码器 $g: \mathbb{R}^d \to \mathbb{R}^f$ 提取表示 $Z = g(X)$
3. FALCON 投影器生成嵌入和概率分配
4. 使用 FALCON 损失函数训练

### 投影器设计

投影器执行两步操作：

$$\mathbf{H} = \sqrt{f/n} \cdot \text{L2-norm}(\text{BN}(\text{Linear}(\mathbf{Z})))$$

$$\mathbf{P} = \text{Softmax}(\mathbf{H}\mathbf{W}/\tau)$$

其中关键设计：
- **字典矩阵** $\mathbf{W} \in \{-1, 1\}^{f \times c}$：元素独立采样自 Rademacher 分布
- **温度参数** $\tau = \sqrt{f/n} \cdot \log\frac{1 - \epsilon(c-1)}{\epsilon}$
- 字典大小 $c \gg f$（超完备字典）

这一设计的核心创新来自**超维计算**：随机 Rademacher 向量在高维空间中近似正交：

$$\mathbb{E}_W\{w_j^T w_{j'}\} = \begin{cases} 1 & j = j' \\ 0 & j \neq j' \end{cases}, \quad \text{Var}_W\{\cos(w_j, w_{j'})\} = \frac{1}{f}$$

这使得 $c$ 可远大于 $f$，突破正交向量数量 $\leq f$ 的限制。

### 损失函数

$$\mathcal{L}_{\text{FALCON}}(\mathcal{D}) = -\frac{\beta}{n}\sum_{i=1}^n \sum_{j=1}^c p_{ij} \log p'_{ij} - \sum_{j=1}^c q_j \log \frac{1}{n}\sum_{i=1}^n p_{ij}$$

- **第一项（不变性损失）**: 鼓励增强对产生一致分配，可分解为熵项 + KL散度项
- **第二项（先验匹配损失）**: 强制分配分布匹配先验 $\mathbf{q}$（均匀分布），防止聚类崩塌
- $\beta > 0$ 平衡两项

### 关键理论保证

**定理（嵌入）**: 在最优解处，每个嵌入与字典中恰好一个码字对齐：

$$\forall i \in [n], \exists! j \in [c] \text{ s.t. } \mathbf{h}_i = \alpha_{ij}\mathbf{w}_j + (\alpha_{ij} - \frac{1}{\sqrt{n}})\sum_{k \neq j}\mathbf{w}_k$$

**推论1（完美对齐）**: 当 $c \to \infty$，$\mathbf{h}_i = \frac{1}{\sqrt{n}} \mathbf{w}_j$

**推论2（对角协方差）**: $\mathbf{H}^T\mathbf{H} = \mathbf{I}$，即嵌入特征完全去相关

**推论3（块对角邻接）**: 邻接矩阵 $\mathbf{H}\mathbf{H}^T$ 为块对角矩阵，等大小块 $n/c$，即天然聚类

## 实验关键数据

### 主实验：下游任务泛化性能（Table 1）

| 方法 | SVHN NMI | CIFAR-10 NMI | CIFAR-100 NMI | SVHN Acc. | CIFAR-10 Acc. | CIFAR-100 Acc. |
|------|----------|-------------|--------------|-----------|--------------|----------------|
| Barlow Twins | 0.06 | 0.05 | 0.10 | 0.76 | 0.65 | 0.28 |
| SwAV | 0.03 | 0.29 | 0.12 | 0.45 | 0.56 | 0.10 |
| Self-Classifier | 0.07 | 0.28 | 0.26 | 0.58 | 0.59 | 0.15 |
| GEDI | 0.07 | 0.29 | 0.25 | 0.58 | 0.64 | 0.38 |
| **FALCON (c=16384)** | **0.31** | **0.35** | **0.58** | **0.78** | **0.68** | **0.41** |

### ImageNet-100 线性探测结果（Table 2，ViT-small，100 epochs）

| 方法 | c=100 | c=500 | c=1K | c=5K | c=10K | c=50K | c=100K | c=200K | c=300K | c=500K |
|------|-------|-------|------|------|-------|-------|--------|--------|--------|--------|
| DINO Top-1 | 64.1% | 65.8% | 65.1% | 65.7% | 66.6% | 67.5% | 67.7% | 68.3% | 67.0% | 67.2% |
| **FALCON Top-1** | **64.9%** | **68.4%** | **70.2%** | **72.2%** | **72.9%** | **73.9%** | **72.2%** | **73.9%** | **73.6%** | **74.0%** |
| DINO Top-5 | 87.0% | 88.6% | 88.7% | 87.7% | 89.2% | 89.2% | 89.4% | 89.8% | 89.2% | 89.9% |
| **FALCON Top-5** | **87.7%** | **89.9%** | **91.2%** | **91.7%** | **92.3%** | **92.6%** | **92.6%** | **92.9%** | **92.8%** | **92.9%** |

### 关键发现

1. **字典大小的单调增益**: 聚类和分类性能随字典大小 $c$ 单调提升，FALCON 是唯一能系统性利用大字典的方法
2. **崩塌完全避免**: 协方差矩阵随 $c$ 增大趋向对角，邻接矩阵块大小缩小，奇异值分布更均匀
3. **超越 DINO**: 在 ImageNet-100 上，FALCON 显著优于 DINO 且设计更简洁（无需停止梯度、教师中心化、EMA 等）

## 亮点与洞察

1. **首次将超维计算与 SSL 结合**：利用高维随机向量的准正交性突破字典大小限制，是极有创意的跨领域联接
2. **理论驱动的设计**：每个组件（Rademacher 字典、L2 归一化、大字典）都有明确的理论动机，而非凭经验调参
3. **统一框架**：同时实现特征去相关（Barlow Twins 族）和聚类（SwAV 族）的优势，理论证明两者可共存
4. **极简设计**：无需动量编码器、停止梯度、Sinkhorn 聚类层等常见 SSL 技巧，训练管线大幅简化

## 局限性

1. **实验规模有限**: 主实验使用 ResNet-8 在 SVHN/CIFAR 上验证，ImageNet-100 使用小 ViT；未在 ImageNet-1K 或更大数据上验证
2. **仅考虑均匀先验**: 理论分析限于 $q_j = 1/c$ 的情况，非均匀先验（现实数据中更常见）未深入探讨
3. **训练不稳定性**: 在 ImageNet-100 上出现训练不稳定（部分码字初期未使用），需将 KL 匹配改为反向 KL
4. **骨干网络容量假设**: 主要理论结果假设骨干网络具有无限容量，有限容量下仅有部分保证

## 相关工作与启发

- **Barlow Twins** (Zbontar et al., 2021): 通过交叉协方差对角化避免维度崩塌，但不处理聚类崩塌
- **SwAV** (Caron et al., 2020): Sinkhorn 聚类层处理聚类崩塌，但需额外启发式
- **DINO** (Caron et al., 2021): 需要不对称设计（停止梯度、教师中心化、EMA），FALCON 在更简单设计下超越
- **VICReg** (Bardes et al., 2022): 方差-不变性-协方差正则化
- 本文的**超维计算视角**可能启发更多 SSL 与高维随机结构的交叉研究

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 5 | 超维计算与 SSL 的首次结合，统一去相关和聚类 |
| 理论深度 | 5 | 完整的理论分析，四种崩塌均有证明保证 |
| 实验充分性 | 3 | 中小规模数据验证充分，缺乏大规模实验 |
| 写作质量 | 4 | 结构清晰，理论推导严谨 |
| 实用价值 | 4 | 简化 SSL 训练管线，但需大规模验证 |
| **综合** | **4.2** | 理论贡献突出的方法论文，实验规模待扩展 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICML 2025\] Generalization Analysis for Supervised Contrastive Representation Learning under Non-IID Settings](generalization_analysis_for_supervised_contrastive_representation_learning_under.md)
- [\[ICML 2025\] ReSA: Clustering Properties of Self-Supervised Learning](clustering_properties_of_self-supervised_learning.md)
- [\[ICML 2025\] Discovering Global False Negatives On the Fly for Self-supervised Contrastive Learning](discovering_global_false_negatives_on_the_fly_for_self-supervised_contrastive_le.md)
- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](../../ICLR2026/self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)

</div>

<!-- RELATED:END -->
