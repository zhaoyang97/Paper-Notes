---
title: >-
  [论文解读] Rao-Blackwellised Reparameterisation Gradients
description: >-
  [NeurIPS 2025][优化/理论][Rao-Blackwell] 提出 R2-G2 估计器作为重参数化梯度的 Rao-Blackwell 化版本，证明 Bayesian MLP 的局部重参数化是其特例，并将低方差梯度的优势推广到一系列概率模型。 现有痛点 现有痛点：领域现状：潜在高斯变量在概率机器学习中被广泛使用…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "Rao-Blackwell"
  - "重参数化"
  - "梯度估计"
  - "变分推断"
  - "贝叶斯MLP"
---

# Rao-Blackwellised Reparameterisation Gradients

**会议**: NeurIPS 2025

**arXiv**: [2506.07687](https://arxiv.org/abs/2506.07687)

**代码**: 无

**领域**: 优化 / 概率机器学习

**关键词**: Rao-Blackwell, 重参数化, 梯度估计, 变分推断, 贝叶斯MLP

## 一句话总结

提出 R2-G2 估计器作为重参数化梯度的 Rao-Blackwell 化版本，证明 Bayesian MLP 的局部重参数化是其特例，并将低方差梯度的优势推广到一系列概率模型。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：潜在高斯变量在概率机器学习中被广泛使用，梯度估计器是支撑梯度优化的核心机制。重参数化技巧（reparameterisation trick）因其简单且低方差的特性成为变分推断的默认选择。

核心挑战：

**方差仍有改进空间**: 重参数化梯度虽优于 REINFORCE，但在复杂模型中方差仍然较高

**Rao-Blackwell 化机会**: 利用条件期望降低方差是经典统计思想，但如何系统应用于重参数化梯度尚不清楚

**局部重参数化的理论理解不足**: 局部重参数化（Kingma et al.）在 Bayesian MLP 中表现出色，但其与 Rao-Blackwell 化的联系未被揭示

## 方法详解

### 整体框架

R2-G2 (Rao-Blackwellised Reparameterisation Gradient) 通过对重参数化梯度中的部分随机变量取条件期望，系统地降低梯度估计的方差。

### 关键设计

**1. 重参数化梯度的分解**

对变分目标 $\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z)}[f(z)]$，重参数化梯度为：
$$\nabla_\phi \mathcal{L} = \mathbb{E}_{\epsilon \sim p(\epsilon)}[\nabla_\phi f(T_\phi(\epsilon))]$$

当模型含多个潜变量 $z = (z_1, z_2, \ldots, z_L)$ 时,可以对部分变量的噪声取条件期望。

**2. R2-G2 估计器**

对某层 $l$ 的梯度，对该层以外的噪声变量取条件期望：
$$\hat{g}^{\text{R2-G2}}_l = \mathbb{E}_{\epsilon_{\backslash l}}[\nabla_{\phi_l} f(T_\phi(\epsilon)) | \epsilon_l]$$

关键结果: 由 Rao-Blackwell 定理保证 $\text{Var}(\hat{g}^{\text{R2-G2}}) \leq \text{Var}(\hat{g}^{\text{reparam}})$

**3. 与局部重参数化的统一**

- 证明: 对 Bayesian MLP，局部重参数化梯度恰好是 R2-G2 的一个特例
- 这一联系首次被揭示,统一了两个看似不同的方法
- 推论: 可以将局部重参数化的优势推广到非 MLP 架构

### 损失函数 / 训练策略

- ELBO 目标: $\mathcal{L} = \mathbb{E}_q[\log p(x|z)] - \text{KL}(q(z) || p(z))$
- 使用 R2-G2 估计 ELBO 的梯度
- 初始训练阶段使用 R2-G2 可以更快收敛，后期可切换回标准重参数化

## 实验关键数据

### 主实验

Bayesian Neural Network 变分推断 (测试对数似然):

| 方法 | Boston Housing | Concrete | Wine Quality | Protein |
|------|---------------|----------|-------------|---------|
| 标准重参数化 | -2.72 | -3.18 | -0.98 | -2.85 |
| 局部重参数化 | -2.65 | -3.09 | -0.94 | -2.79 |
| R2-G2 (Ours) | **-2.62** | **-3.05** | **-0.92** | **-2.76** |

VAE 训练 ELBO (MNIST, 中间潜变量层):

| 方法 | ELBO (final) | 梯度方差 (log) | 收敛轮次 |
|------|-------------|--------------|---------|
| 标准重参数化 | -86.5 | -3.2 | 150 |
| R2-G2 (Ours) | **-85.8** | **-4.5** | **110** |

### 消融实验

R2-G2 在不同模型深度下的方差降低比率:

| 模型深度(层数) | 梯度方差降低 | ELBO 提升 |
|-------------|-----------|----------|
| 2层 | 1.5x | +0.2 |
| 4层 | 3.2x | +0.5 |
| 8层 | 7.8x | +0.9 |
| 16层 | 15.1x | +1.4 |

### 关键发现

1. R2-G2 的方差降低随模型深度**线性增长**——模型越深,优势越大
2. 初始训练阶段使用 R2-G2 的收益最大, 因为此时梯度方差对优化影响最大
3. 在多层概率模型中（如深层VAE），R2-G2 的优势尤为显著
4. 局部重参数化 = R2-G2 + Bayesian MLP 的特化，为理解前者提供了新视角

## 亮点与洞察

- **理论优雅**: Rao-Blackwell 化是降低方差的最自然方式,且保证了改进
- **统一现有方法**: 将局部重参数化纳入统一框架
- **可扩展性**: 方法可推广到任何使用重参数化技巧的概率模型

## 局限与展望

1. 条件期望的解析计算仅适用于高斯等特定分布族
2. 在某些模型结构中,R2-G2 的额外计算开销可能抵消方差降低的收益
3. 大规模深度模型（如 GPT 级别的贝叶斯训练）的验证不足
4. 与其他方差降低技术（如 Control Variates）的组合效果未探索

## 相关工作与启发

- **重参数化技巧** (Kingma & Welling, 2014): VAE 的核心技术
- **局部重参数化** (Kingma et al., 2015): Bayesian MLP 的高效梯度估计
- **Rao-Blackwell 定理**: 统计学中方差降低的基本工具

## 评分

- ⭐ 创新性: 7/10 — 理论联系虽优雅,但 Rao-Blackwell 化思想本身已有先例
- ⭐ 实用性: 7/10 — 对概率模型从业者有价值,但应用范围相对狭窄
- ⭐ 写作质量: 9/10 — 理论推导精准,与已有工作的关系阐述清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients](efficient_federated_learning_against_byzantine_attacks_and_data_heterogeneity_vi.md)
- [\[ICML 2026\] A2SG: Adaptive and Asymmetric Surrogate Gradients for Training Deep Spiking Neural Networks](../../ICML2026/optimization/a2sgadaptive_and_asymmetric_surrogate_gradients_for_training_deep_spiking_neural.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)

</div>

<!-- RELATED:END -->
