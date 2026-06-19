---
title: >-
  [论文解读] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning
description: >-
  [AAAI 2026][半监督学习] 提出 SC-SSL 框架，通过引入扩展分类器：进行解耦采样控制来缓解特征级不平衡，并利用线性层偏置项：作为优化偏差向量在推理时直接校准 logits，在多种数据分布设定下达到 SOTA。 1. 领域现状：半监督学习（SSL）通过伪标签和一致性正则化利用无标签数据，但现实数据往往呈长尾分布…
tags:
  - "AAAI 2026"
  - "半监督学习"
  - "类别不平衡"
  - "采样控制"
  - "伪标签"
  - "校准"
---

# Sampling Control for Imbalanced Calibration in Semi-Supervised Learning

**会议**: AAAI 2026  
**arXiv**: [2511.18773](https://arxiv.org/abs/2511.18773)  
**代码**: [https://github.com/Sheldon04/SC-SSL](https://github.com/Sheldon04/SC-SSL)  
**领域**: 半监督学习 / 类别不平衡  
**关键词**: 半监督学习, 类别不平衡, 采样控制, 伪标签, 校准

## 一句话总结

提出 SC-SSL 框架，通过引入**扩展分类器**进行解耦采样控制来缓解特征级不平衡，并利用线性层**偏置项**作为优化偏差向量在推理时直接校准 logits，在多种数据分布设定下达到 SOTA。

## 研究背景与动机

1. **领域现状**：半监督学习（SSL）通过伪标签和一致性正则化利用无标签数据，但现实数据往往呈长尾分布，导致伪标签偏向头部类别（类别不平衡半监督学习，CISSL）。
2. **现有痛点**：现有方法（ACR、CPE、SimPro）通过估计无标签数据的类分布来调整 logits，但存在两个问题：(a) 粗粒度处理，将数据不平衡与不同类别学习难度导致的偏差混为一谈；(b) 保守策略导致大量无标签数据未被有效利用——只保留少量高质量伪标签以避免确认偏差。
3. **核心矛盾**：数据不平衡和优化不平衡是两个独立的偏差来源，但现有方法将它们耦合处理。在双分类器设定下，输出分类器和原始分类器都无法有效调整非头部类的采样概率。
4. **本文目标**：从更细粒度的角度——特征级和 logits 级——分别抑制模型偏差。
5. **切入角度**：基于自训练的展开-分离假设（expansion-separation assumption），即使噪声伪标签也能通过一致性正则化传播有效监督信号，关键是控制非头部类的采样概率。
6. **核心 idea**：引入第三个扩展分类器专门提升非头部类的采样概率以平衡特征学习，推理时用偏置项差异隔离并校正优化偏差。

## 方法详解

### 整体框架

SC-SSL 在 FixMatch 基础上增加两个额外分类器：**输出分类器** $F_b$（平衡采样用于推理）和**扩展分类器** $F_e$（过采样非头部类以平衡特征学习），三者共享骨干网络 $B$。训练阶段通过理论分析确定采样控制的关键变量（$\gamma_u$、$\Delta p$、$\rho$），动态调整不同分布下的采样概率。推理阶段分析线性层的偏置项模式，用优化偏差向量 $\mathbf{b}_{opt}$ 直接校准 logits。

### 关键设计

1. **扩展分类器与采样控制（训练阶段）**

    - 功能：通过专门为非头部类设计的分类器提升其采样概率，平衡特征空间中的梯度贡献。
    - 核心思路：基于简化的二分类分析（定理 0.1），伪标签的采样概率主要受三个因素影响：数据不平衡因子 $\gamma_u$、logit 调整量 $\Delta p$、置信度阈值 $\rho$。输出分类器和原始分类器都无法大幅调整 $\Delta p$ 或 $\rho$（前者需保证分类准确性，后者会违反分离假设）。因此引入扩展分类器 $F_e$，其 $\tau_e=4$（大于输出分类器的 $\tau_b=2$），并根据展开因子 $c$ 初始化更低的非头部类阈值 $\rho_e^0(\text{non-head})$，同时通过动态调整 $\rho^t(k) = \rho^{t-1}(k) - \alpha \cdot \mathbb{I}(\mathbf{b}_{opt}(k) > \nu)$ 适应训练过程中的偏差变化。
    - 设计动机：展开假设保证即使少量自信预测也能传播标签信息，但前提是非头部类有足够的采样。现有双分类器设定无法有效增加非头部类采样，需要第三个分类器专门负责。

2. **偏置项分析与推理校准**

    - 功能：在推理阶段直接校正优化导致的偏差。
    - 核心思路：观察发现线性层偏置项 $\mathbf{b}$ 编码了两种偏差：数据分布偏差和优化偏差。随机采样下头部类偏置高（数据偏差），扩展采样下尾部类偏置高（过校正）。输出分类器使用平衡采样训练，其偏置项排除了数据偏差，仅保留优化偏差。因此定义 $\mathbf{b}_{opt}$ 为优化偏差向量，推理时 $\tilde{F}(B(x)) = F_b(B(x)) - \mathbf{b}_{opt} = W_b(B(x))$，直接去除偏置实现无偏预测。
    - 设计动机：权重矩阵 $\mathbf{W}$ 需要与特征向量交互，难以解耦分析；而偏置项是一个干净的代理，可以直接隔离优化偏差。

3. **分布估计先验**

    - 功能：在训练前利用 $\mathbf{b}_{opt}$ 近似估计无标签数据分布，初始化伪标签采样策略。
    - 核心思路：经过几轮估计训练后，用校准后的输出在无标签数据上推断各类数量 $N^e$，通过 KL 散度匹配预定义分布 $o^* = \arg\min_o D_{KL}(N^e, Q^{(o)})$，据此确定展开因子 $c$ 和初始阈值。
    - 设计动机：不需要假设无标签数据分布，但利用估计先验可以更好地初始化采样策略。

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{basic} + \mathcal{L}_{sup}^b(\tau_b, F_b) + \mathcal{L}_{con}^b(\rho_b, F_b) + \mathcal{L}_{sup}^e(\tau_e, F_e) + \mathcal{L}_{con}^e(\rho_e, F_e)$，其中 $\mathcal{L}_{basic}$ 是基础 SSL 损失，$\tau_b=2$, $\tau_e=4$。展开因子 $c$ 根据估计分布取 4-6，$\rho_{max}=0.95$，步长 $\alpha=0.005$，阈值 $\nu=1.0$。

## 实验关键数据

### 主实验

在 CIFAR10-LT 和 CIFAR100-LT 上的测试准确率：

| 方法 | CIFAR10 Consist (100-100) | CIFAR10 Inverse (100-100) | CIFAR100 Consist (15-15) | CIFAR100 Inverse (15-15) |
|------|--------------------------|--------------------------|-------------------------|-------------------------|
| FixMatch+LA | 81.49 | 80.68 | 58.56 | 58.21 |
| w/ ACR | 84.10 | 89.46 | 60.34 | 61.79 |
| w/ CPE | 84.46 | 87.10 | 59.83 | 60.83 |
| **w/ SC-SSL** | **86.53** | **89.97** | **60.65** | **62.99** |

ImageNet-127 上的结果：

| 方法 | 32×32 | 64×64 |
|------|-------|-------|
| SimPro | 59.4 | 67.2 |
| **SC-SSL** | **62.3** | **69.4** |

### 消融实验

| 配置 | CIFAR10 Consist | CIFAR10 Inverse | 说明 |
|------|----------------|----------------|------|
| 2 类别划分 | 83.89 | 86.02 | 最优的头/非头划分 |
| 3 类别划分 | 83.54 | 85.98 | 更细粒度划分无明显提升 |
| 4 类别划分 | 83.50 | 86.15 | 2分即足够 |

### 关键发现

- SC-SSL 在所有分布设定（一致、反向、均匀、高斯、未知）下都优于或持平 SOTA。
- 偏置项校准对推理精度提升显著——不同采样策略下偏置项的模式清晰可辨。
- STL10-LT 在未知分布设定下提升尤为明显（79.26% vs 76.94%），表明方法对分布不确定性的鲁棒性。
- 简单的 2 类别划分（头/非头）已经足够，更细粒度划分不带来额外增益。

## 亮点与洞察

- **偏置项作为优化偏差的代理变量**：这个观察非常巧妙——通过对比不同采样策略下的偏置项模式，干净地分离数据偏差和优化偏差。这个思路可以迁移到任何涉及不平衡训练的分类任务。
- **理论驱动的采样控制**：通过简化二分类模型分析清楚地识别了三个关键控制变量，为采样策略设计提供了理论依据。
- **展开-分离假设的实际应用**：将自训练理论的展开因子与采样阈值初始化关联，使理论分析直接指导超参设置。

## 局限与展望

- 展开因子 $c$ 的设定依赖预定义锚点分布，对分布假设有一定限制。
- 仅在分类任务上验证，检测、分割等任务的适用性未探讨。
- 三分类器设计增加了训练计算量和超参数。
- 偏置项校准假设特征不平衡已被缓解，对特征不平衡缓解不充分时效果可能下降。

## 相关工作与启发

- **vs ACR**：ACR 通过预定义分布锚点调整一致性正则化，SC-SSL 则通过扩展分类器主动控制采样概率，从根源解决特征偏差。
- **vs SimPro**：SimPro 用概率模型建模任意分布，但仍依赖 logit 调整；SC-SSL 通过偏置项直接校准，更简洁有效。
- **vs ABC**：ABC 引入额外分类器防止偏差但未控制采样概率，SC-SSL 的扩展分类器有明确的采样控制目标。

## 评分

- 新颖性: ⭐⭐⭐⭐ 偏置项分析和三分类器解耦设计有新意
- 实验充分度: ⭐⭐⭐⭐ 多数据集多分布设定全面覆盖
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，动机论证充分
- 价值: ⭐⭐⭐⭐ 对 CISSL 领域有实际贡献，但应用场景相对窄

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Semi-Supervised High Dynamic Range Image Reconstructing via Bi-Level Uncertain Area Masking](semi-supervised_high_dynamic_range_image_reconstructing_via_bi-level_uncertain_a.md)
- [\[ECCV 2024\] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](../../ECCV2024/others/rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)
- [\[NeurIPS 2025\] Semi-Supervised Regression with Heteroscedastic Pseudo-Labels](../../NeurIPS2025/others/semi-supervised_regression_with_heteroscedastic_pseudo-labels.md)
- [\[AAAI 2026\] Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States](enhancing_control_policy_smoothness_by_aligning_actions_with_predictions_from_pr.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)

</div>

<!-- RELATED:END -->
