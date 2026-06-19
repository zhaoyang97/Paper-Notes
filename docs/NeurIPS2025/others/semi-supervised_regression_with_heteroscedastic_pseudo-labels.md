---
title: >-
  [论文解读] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels
description: >-
  [NeurIPS 2025][半监督回归] 提出基于异方差建模的不确定性感知伪标签框架，通过双层优化动态校准每个伪标签的不确定性，避免错误伪标签对回归模型的负面影响，在多个 SSR 基准上取得 SOTA。 半监督回归（SSR）面临的核心困难与分类不同：回归的输出是连续值，无法像分类那样通过置信度阈值筛选伪标签的可靠性…
tags:
  - "NeurIPS 2025"
  - "半监督回归"
  - "伪标签"
  - "异方差不确定性"
  - "双层优化"
  - "不确定性估计"
---

# Semi-Supervised Regression with Heteroscedastic Pseudo-Labels

**会议**: NeurIPS 2025  
**arXiv**: [2510.15266](https://arxiv.org/abs/2510.15266)  
**代码**: [GitHub](https://github.com/sxq/Heteroscedastic-Pseudo-Labels)  
**领域**: 半监督学习 / 回归  
**关键词**: 半监督回归, 伪标签, 异方差不确定性, 双层优化, 不确定性估计

## 一句话总结

提出基于异方差建模的不确定性感知伪标签框架，通过双层优化动态校准每个伪标签的不确定性，避免错误伪标签对回归模型的负面影响，在多个 SSR 基准上取得 SOTA。

## 研究背景与动机

半监督回归（SSR）面临的核心困难与分类不同：回归的输出是连续值，无法像分类那样通过置信度阈值筛选伪标签的可靠性。现有 SSR 方法主要依赖一致性正则化（如 TNNR 的循环一致性、UCVME 的不确定性一致性、RankUp 的排序一致性），但仅靠一致性约束不足以处理伪标签中的噪声。

作者在 UTKFace 数据集上可视化了 UCVME 生成的伪标签分布，发现即使施加了不确定性一致性约束，伪标签的方差仍然很大，尤其对于样本量较少的年龄段。

关键洞察是：**伪标签的噪声是异方差的（heteroscedastic）**，即不同样本的伪标签错误程度不同，且与输入特征相关。因此需要一种能动态评估和调节每个伪标签可靠性的机制。

然而，简单地引入一个辅助网络联合端到端训练来估计不确定性存在根本缺陷：模型无法区分"困难但正确的样本"（模型预测与正确伪标签差距大）和"容易但错误的样本"（模型预测接近真值但与错误伪标签差距大），两种情况都会导致高不确定性，但前者不应被抑制。

## 方法详解

### 整体框架

本方法包含两个网络：
- **回归网络** $f_\theta$：主模型，预测回归值
- **不确定性学习器** $g_\phi$：轻量 MLP，为每个伪标签动态估计不确定性

两者通过**双层优化**框架训练：内层更新回归网络（使用标注数据 + 不确定性加权的伪标签数据），外层更新不确定性学习器（在另一批标注数据上评估回归网络的泛化性能）。

### 关键设计

#### 1. **异方差伪标签建模**

将每个未标注样本 $x_j^u$ 的伪标签建模为异方差高斯：

$$\hat{y}_j = f_\theta(x_j^u) + \epsilon_j, \quad \epsilon_j \sim \mathcal{N}(0, \sigma_j^2)$$

对应的负对数似然损失为：

$$\mathcal{L}_u = \sum_{x_j^u \in \mathcal{B}_u} \frac{1}{\sigma_j^2} (\hat{y}_j - f_\theta(x_j^u))^2 + \sum_{x_j^u \in \mathcal{B}_u} \log(\sigma_j^2)$$

当 $\sigma_j^2 = 1$ 时退化为标准 MSE。设计动机：当伪标签不准确时，模型可以通过增大 $\sigma_j^2$ 来降低该伪标签的权重，同时 $\log(\sigma_j^2)$ 项防止所有不确定性趋向无穷大。

#### 2. **双层优化框架**

不确定性学习器 $g_\phi$ 输出对数方差 $z_j = \log \sigma_j^2 = g_\phi(r_j, \hat{y}_j)$，其中 $r_j$ 是回归模型对强增强输入的预测，$\hat{y}_j$ 是对弱增强输入的伪标签（借鉴 FixMatch）。

**内层优化**（更新 $\theta$）：

$$\theta^*(\phi) = \arg\min_\theta \mathcal{L}^{inner} := \mathcal{L}_l(\theta) + \lambda \mathcal{L}_u(\theta, \phi)$$

其中 $\mathcal{L}_l = \sum (y_i - f_\theta(x_i^l))^2$，$\mathcal{L}_u$ 使用不确定性加权。

**外层优化**（更新 $\phi$）：

$$\phi^* = \arg\min_\phi \mathcal{L}^{outer} := \sum_{x_k^l \in \hat{\mathcal{B}}_l} (y_k - f_{\theta^*(\phi)}(x_k^l))^2$$

外层在**另一批标注数据**上评估更新后的回归模型，确保不确定性估计有利于泛化。

设计动机：通过外层目标"审视"内层的效果，如果 $g_\phi$ 给错误伪标签分配了低不确定性导致内层学偏，外层损失会增大并修正 $g_\phi$。

#### 3. **高效近似与训练**

完整的双层优化需要对整个网络做二阶导数展开，成本高昂。本文的关键近似是 **仅对回归头（单层全连接层）做梯度展开**，因为 $\phi$ 主要通过回归头影响损失。实际额外开销仅 ~9ms/iter 和 ~17MB 显存。

### 损失函数 / 训练策略

每次迭代三步：
1. 从标注集采样 $\mathcal{B}_l$、无标注集采样 $\mathcal{B}_u$、再采样一批标注集 $\hat{\mathcal{B}}_l$
2. 计算 $\mathcal{L}^{inner}$，用梯度下降更新 $\theta$
3. 计算 $\mathcal{L}^{outer}$，用梯度下降更新 $\phi$

理论分析（Theorem 1）表明，优化 $\phi$ 等价于最大化内层梯度与外层梯度的对齐：

$$\min_\phi -\langle \nabla_\theta \mathcal{L}^{inner}(\theta, \phi), \nabla_\theta \mathcal{L}^{outer}(\theta) \rangle$$

即确保带伪标签训练的梯度方向与纯标注数据的梯度方向一致。

## 实验关键数据

### 主实验

在三个数据集（UTKFace、IMDB-WIKI、STS-B）上，分别使用 5%、10%、20% 标注比例。

| 数据集 | 标注比例 | 指标 | 本文 | UCVME | RankUp | 提升 |
|---|---|---|---|---|---|---|
| UTKFace | 5% | MAE↓ | **5.639** | 5.862 | 5.719 | vs RankUp -1.4% |
| UTKFace | 5% | R²↑ | **0.523** | 0.495 | 0.495 | vs RankUp +5.7% |
| IMDB-WIKI | 5% | MAE↓ | **9.177** | 9.730 | 10.251 | vs Mean Teacher -3.3% |
| IMDB-WIKI | 5% | R²↑ | **0.664** | 0.633 | 0.599 | vs UCVME +4.9% |
| IMDB-WIKI | 20% | MAE↓ | **8.166** | 8.309 | 8.216 | 接近 Fully-Sup (7.974) |
| STS-B | 5% | MSE↓ | **1.540** | 1.713 | 1.844 | vs SSDKL -4.1% |
| STS-B | 5% | R²↑ | **0.270** | 0.188 | 0.126 | vs SSDKL +13.0% |

在标注稀缺（5%）时优势最突出；随着标注增多，与 UCVME 和 RankUp 的差距缩小但仍保持竞争力。

### 消融实验

| 配置 | γ=5% MAE↓ | γ=5% R²↑ | γ=10% MAE↓ | 说明 |
|---|---|---|---|---|
| Baseline (无 UL, 无 BLO) | 9.512 | 0.651 | 8.864 | 固定 σ²=1 的标准伪标签 |
| Baseline + UL (无 BLO) | 9.914 | 0.630 | 9.562 | 联合训练反而变差！ |
| **Baseline + UL + BLO (完整)** | **9.177** | **0.664** | **8.539** | 双层优化是关键 |

### 关键发现

1. **联合训练不确定性学习器反而有害**：没有双层优化时，$g_\phi$ 和 $f_\theta$ 联合训练会导致不确定性估计不准确，抑制了"困难但正确"样本的学习。
2. **不确定性与预测误差正相关**：可视化显示 $g_\phi$ 估计的 $\sigma^2$ 随样本预测误差增大而增大，符合预期。
3. **计算开销极低**：每次迭代仅增加 9ms 和 17MB 显存，远低于 UCVME（257ms, 10057MB）和 SimRegMatch（548ms, 7419MB）。
4. **亚组分析**：在年龄稀缺的高龄段，本文方法的伪标签精度显著优于 UCVME。

## 亮点与洞察

- 将分类中的伪标签范式适配到回归，通过异方差建模优雅地解决了连续值伪标签的可靠性问题。
- 双层优化的设计避免了直接联合训练的退化问题，用外层的标注数据作为"裁判"来校准不确定性。
- 仅展开回归头梯度的近似策略巧妙地在理论优雅性和计算效率之间取得平衡。
- 理论上证明了双层优化等价于内外梯度对齐，提供了清晰的直觉解释。

## 局限与展望

- 假设标注数据和未标注数据可以同时访问，可能不适用于隐私敏感场景。
- 未考虑标注数据中的系统性偏差（如人口学不平衡），可能通过伪标签放大。
- 不确定性学习器的输入仅使用预测值和伪标签，未充分利用特征空间信息。
- 高标注比例下优势缩小，说明方法主要在极端低标注场景发挥作用。

## 相关工作与启发

- **UCVME**：使用共享编码器+双头预测均值和方差，但不确定性一致性约束不足以保证准确校准。
- **FixMatch**：弱增强生成伪标签、强增强训练模型的策略被本文借鉴。
- **Meta-learning / DARTS 风格的双层优化**：本文的框架直接借鉴了 MAML 和 DARTS 的交替更新策略。
- 启发：在噪声标签学习中，用双层优化动态调节样本权重/不确定性是一个通用且强力的策略。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 异方差伪标签 + 双层优化的组合在 SSR 中是新的，但各组件已有先例
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个数据集三个标注比例、多个基线、消融、可视化、计算成本分析俱全
- **写作质量**: ⭐⭐⭐⭐ 动机叙述清晰，双层优化的推导严谨，理论与实验互相呼应
- **价值**: ⭐⭐⭐⭐ 为 SSR 提供了简洁高效的解决方案，低标注场景优势明显

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[AAAI 2026\] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking](../../AAAI2026/others/semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](../../ICML2025/others/regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)
- [\[NeurIPS 2025\] Regression Trees Know Calculus](regression_trees_know_calculus.md)
- [\[NeurIPS 2025\] Semi-infinite Nonconvex Constrained Min-Max Optimization](semi-infinite_nonconvex_constrained_min-max_optimization.md)

</div>

<!-- RELATED:END -->
