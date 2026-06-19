---
title: >-
  [论文解读] Fully Heteroscedastic Count Regression with Deep Double Poisson Networks
description: >-
  [ICML 2025][AI安全][计数回归] 提出 Deep Double Poisson Network (DDPN)，通过输出 Double Poisson 分布的参数实现离散计数回归中的完全异方差性，支持任意高或低的预测方差，在精度、校准和 OOD 检测上全面超越现有基线。 神经网络的不确定性量化在现实 AI 系统中…
tags:
  - "ICML 2025"
  - "AI安全"
  - "计数回归"
  - "Double Poisson 分布"
  - "异方差回归"
  - "深度集成"
  - "不确定性估计"
  - "分布外检测"
---

# Fully Heteroscedastic Count Regression with Deep Double Poisson Networks

**会议**: ICML 2025  
**arXiv**: [2406.09262](https://arxiv.org/abs/2406.09262)  
**代码**: [GitHub](https://github.com/porterjenkins/deep-double-poisson)  
**领域**: 统计建模/深度学习/不确定性量化  
**关键词**: 计数回归, Double Poisson 分布, 异方差回归, 深度集成, 不确定性估计, 分布外检测

## 一句话总结

提出 Deep Double Poisson Network (DDPN)，通过输出 Double Poisson 分布的参数实现离散计数回归中的完全异方差性，支持任意高或低的预测方差，在精度、校准和 OOD 检测上全面超越现有基线。

## 研究背景与动机

神经网络的不确定性量化在现实 AI 系统中至关重要，常被分解为认知不确定性（模型参数不确定）和偶然不确定性（观测噪声）。对于连续回归，深度集成（Deep Ensembles）配合高斯异方差网络已被广泛使用且效果优异，关键在于每个成员能输出不受限的预测方差。

然而，**离散计数回归**场景缺乏类似方法。现有方案面临严重限制：

- **Poisson DNN**：受等散性（equi-dispersion）约束，均值和方差被绑定为同一值 $\hat{\lambda} = \hat{\mu} = \hat{\sigma}^2$，无法独立建模方差
- **负二项（Negative Binomial）DNN**：虽打破等散性，但受过散性（over-dispersion）约束 $\hat{\sigma}^2 \geq \hat{\mu}$，不能表示低于均值的方差
- **高斯模型**：虽有完全异方差性，但对离散计数数据会产生模型误指定——给负值和非整数分配概率质量，缺乏正确的归纳偏置

作者指出，完全异方差性对集成模型校准至关重要。集成的总方差 $\text{Var}[y_i|\mathbf{x}_i] = \mathbb{E}_m[\sigma_m^{2(i)}] + \text{Var}_m[\mu_m^{(i)}]$，若成员模型的偶然不确定性项被错误指定，将导致预测分布整体失校准，进而影响认知不确定性估计。

## 方法详解

### 整体框架

DDPN 是一个神经网络，输入任意数据 $\mathbf{x}_i$，通过共享的 $L-1$ 层特征提取器得到隐表示 $\mathbf{z}_i$，然后用两个独立的线性头分别输出 Double Poisson 分布的两个参数：

$$\log(\hat{\mu}_i) = \mathbf{w}_\mu^T \mathbf{z}_i + b_\mu, \quad \log(\hat{\gamma}_i) = \mathbf{w}_\gamma^T \mathbf{z}_i + b_\gamma$$

其中 $\hat{\mu}_i > 0$ 是预测均值，$\hat{\gamma}_i > 0$ 是逆散度参数。Double Poisson 分布的概率质量函数为：

$$p(y|\mu, \gamma) = \frac{\gamma^{1/2} e^{-\gamma\mu}}{c(\mu, \gamma)} \cdot \frac{e^{-y} y^y}{y!} \cdot \left(\frac{e\mu}{y}\right)^{\gamma y}$$

利用 Efron 的矩近似 $\mathbb{E}[Z] = \mu$，$\text{Var}[Z] = \mu / \gamma$，该分布的方差可以通过独立调节 $\gamma$ 实现任意值：当 $\gamma > 1$ 时欠散（方差小于均值），当 $\gamma < 1$ 时过散（方差大于均值），当 $\gamma = 1$ 时退化为标准 Poisson。这使 DDPN 成为**首个完全异方差的离散计数回归模型**。

### 关键设计一：可学习损失衰减（Learnable Loss Attenuation）

DDPN 的训练目标是最小化 Double Poisson 负对数似然（NLL）：

$$\mathcal{L}_i = -\frac{\log \hat{\gamma}_i}{2} + \hat{\gamma}_i \hat{\mu}_i - \hat{\gamma}_i y_i (1 + \log \hat{\mu}_i - \log y_i)$$

作者首次给出了可学习损失衰减的形式化定义：损失函数可被分解为 $\mathcal{L} = d(\hat{\phi}_i) + a(\hat{\phi}_i) \cdot r(\hat{\mu}_i, y_i)$，其中 $d$ 是散度惩罚项（趋向无穷），$a$ 是衰减因子（趋向零），$r$ 是残差惩罚项。

对 DDPN，具体形式为：
- $d(\hat{\phi}_i) = \frac{1}{2} \log \hat{\phi}_i$（散度惩罚）
- $a(\hat{\phi}_i) = 1/\hat{\phi}_i$（衰减因子）
- $r(\hat{\mu}_i, y_i) = (\hat{\mu}_i - y_i) - y_i(\log \hat{\mu}_i - \log y_i)$（残差惩罚）

这意味着 DDPN 与高斯模型类似，能够通过增大预测散度来自适应降低异常点对损失的影响，从而实现更鲁棒的回归。

### 关键设计二：β-DDPN 可控损失衰减

损失衰减虽带来鲁棒性，但过度衰减可能导致模型在难拟合区域"放弃"均值拟合、转而用高不确定性解释。作者提出 $\beta$-DDPN 修改损失函数：

$$\mathcal{L}_i^{(\beta)} = \lfloor \hat{\gamma}_i^{-\beta} \rfloor \cdot \left(-\frac{\log \hat{\gamma}_i}{2} + \hat{\gamma}_i \hat{\mu}_i - \hat{\gamma}_i y_i(1 + \log \hat{\mu}_i - \log y_i)\right)$$

其中 $\lfloor \cdot \rfloor$ 表示 stop-gradient 操作。修改后的偏导数为：

$$\frac{\partial \mathcal{L}_i^{(\beta)}}{\partial \hat{\mu}_i} = (\hat{\gamma}_i^{1-\beta}) \left(1 - \frac{y_i}{\hat{\mu}_i}\right)$$

当 $\beta = 0$ 时退化为标准 NLL；$\beta = 1$ 时完全消除散度对均值梯度的影响，使训练更偏向拟合均值。实验证明 $\beta$ 越大，均值收敛越快。

### 集成策略

$M$ 个独立训练的 DDPN 通过均匀混合形成集成预测：

$$p(y_i|\mathbf{x}_i) = \frac{1}{M} \sum_{m=1}^M p(y_i | \mathbf{f}_{\Theta_m}(\mathbf{x}_i))$$

总方差可分解为偶然不确定性（成员平均方差）+ 认知不确定性（成员均值的方差）。

## 实验关键数据

### 表1：四个真实数据集上的精度（MAE↓）和校准（CRPS↓）

| 方法 | Length of Stay MAE | Length of Stay CRPS | COCO-People MAE | COCO-People CRPS |
|---|---|---|---|---|
| Poisson DNN | 0.664 | 0.553 | 1.099 | 0.851 |
| NB DNN | 0.685 | 0.570 | 1.143 | 0.867 |
| β₀.₅-Gaussian | 0.600 | 0.427 | 1.055 | 0.786 |
| **DDPN** | **0.502** | 0.390 | 1.135 | 0.810 |
| **β₀.₅-DDPN** | 0.516 | **0.370** | 1.095 | 0.782 |
| **β₁.₀-DDPN** | 0.558 | 0.407 | **1.006** | **0.759** |
| DDPN 集成 | **0.485** | 0.361 | 1.024 | 0.744 |
| β₁.₀-DDPN 集成 | 0.543 | 0.393 | **0.959** | **0.712** |

### 表2：OOD 检测结果（Amazon Reviews → 圣经文本）

| 方法 | AUROC↑ | AUPR↑ | FPR80↓ |
|---|---|---|---|
| Poisson DNN | 0.330 | 0.413 | 0.793 |
| NB DNN | 0.280 | 0.397 | 0.819 |
| Gaussian DNN | 0.840 | 0.812 | 0.318 |
| β₀.₅-Gaussian | — | — | — |
| **DDPN 集成** | — | — | — |
| **β-DDPN 集成** | **最优** | **最优** | **最优** |

DDPN 及其 β 变体在所有 OOD 指标上均取得最佳表现，Poisson 和 NB 的 AUROC 接近随机（0.33 / 0.28），说明受限方差模型完全无法区分分布内外数据。

## 关键发现

1. **完全异方差性是关键**：DDPN 或其 β 变体在所有四个数据集（表格、图像、点云、文本）上均取得最佳精度和校准，且优势通常显著
2. **误指定恢复能力强**：即使真实数据来自 Poisson 或负二项分布，DDPN 也能恢复正确的分布结构，性能不逊于匹配分布的模型
3. **集成进一步提升**：DDPN 集成在所有指标上优于所有基线集成，证明完全异方差性确实改善了认知不确定性估计
4. **β 修改有效**：提高 β 值能加速均值收敛，同时在多数数据集上同步改善 CRPS

## 亮点与洞察

- **理论与实践的完美结合**：论文不仅提出方法，还形式化定义了"可学习损失衰减"并证明 DDPN 满足该定义，为离散回归建立了坚实的理论基础
- **归纳偏置的重要性**：高斯模型虽有完全异方差性，但对离散数据的不当建模（给负值和非整数分配概率）导致可测量的性能损失，说明选择正确的输出分布族至关重要
- **方法简洁且通用**：DDPN 只需在标准网络最后一层添加一个额外的输出头，几乎可以零成本集成到任何现有架构中
- **β-NLL 从连续到离散的迁移**：将 Seitzer et al. 的 β 修改巧妙适配到 Double Poisson NLL，展示了异方差回归理论的跨分布通用性

## 局限性

1. **高计数场景未验证**：论文未研究极大计数值（数千或数百万量级）的行为，此时高斯近似可能足够好
2. **矩近似的边界条件**：Efron 的矩近似在 $\mu_0 \to 0$ 且方差较大时会退化，虽然作者展示了在绝大多数情况下误差接近零
3. **训练稳定性**：SGD 和 Adam 优化器在训练 DDPN 时可能收敛不佳，需要使用 AdamW
4. **OOD 实验单一**：仅在 Amazon Reviews 上做了一次 OOD 实验，结论的泛化性有待更多数据集验证

## 相关工作与启发

- **Efron (1986)** 提出 Double Poisson 分布，最初用于 GLM 框架，本文将其引入深度学习
- **Lakshminarayanan et al. (2017)** 的深度集成是认知不确定性估计的基石方法
- **Seitzer et al. (2022)** 的 β-NLL 修改解决了高斯异方差回归的过度衰减问题，本文将其适配到离散设置
- **Kendall & Gal (2017)** 首先观察到损失衰减现象，但未给出形式化定义

本文的工作启发了一个重要方向：为不同数据类型（计数、序数、多类别等）寻找具有完全异方差性的适当输出分布，而非简单套用高斯假设。

## 评分

⭐⭐⭐⭐ — 理论扎实、实验全面、方法简洁实用，填补了离散计数回归中完全异方差性的重要空白。唯一遗憾是 OOD 实验较少且高计数场景缺乏验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Benchmarking Stochastic Approximation Algorithms for Fairness-Constrained Training of Deep Neural Networks](../../ICLR2026/ai_safety/benchmarking_stochastic_approximation_algorithms_for_fairness-constrained_traini.md)
- [\[ICCV 2025\] A Framework for Double-Blind Federated Adaptation of Foundation Models](../../ICCV2025/ai_safety/a_framework_for_doubleblind_federated_adaptation_of_foundati.md)
- [\[ICML 2025\] Solving Probabilistic Verification Problems of Neural Networks Using Branch and Bound](solving_probabilistic_verification_problems_of_neural_networks_using_branch_and_.md)
- [\[ICML 2025\] FicGCN: Unveiling the Homomorphic Encryption Efficiency from Irregular Graph Convolutional Networks](ficgcn_unveiling_the_homomorphic_encryption_efficiency_from_irregular_graph_conv.md)
- [\[NeurIPS 2025\] Distributional Adversarial Attacks and Training in Deep Hedging](../../NeurIPS2025/ai_safety/distributional_adversarial_attacks_and_training_in_deep_hedging.md)

</div>

<!-- RELATED:END -->
