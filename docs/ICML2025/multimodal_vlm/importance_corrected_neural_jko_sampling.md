---
title: >-
  [论文解读] Importance Corrected Neural JKO Sampling
description: >-
  [ICML 2025][多模态VLM][Neural JKO] 提出 Importance Corrected Neural JKO Sampling (Neural JKO IC)，将连续归一化流（CNF）的局部 JKO 步与基于重要性权重的拒绝重采样步交替使用，克服 Wasserstein 梯度流在多模态分布上的局部最优问题，同时保持独立同分布采样和密度可评估性。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "Neural JKO"
  - "Rejection Sampling"
  - "Continuous Normalizing Flows"
  - "Importance Sampling"
  - "多模态分布"
---

# Importance Corrected Neural JKO Sampling

**会议**: ICML 2025  
**arXiv**: [2407.20444](https://arxiv.org/abs/2407.20444)  
**代码**: [github.com/johertrich/neural_JKO_ic](https://github.com/johertrich/neural_JKO_ic)  
**领域**: 采样方法, Wasserstein 梯度流, 归一化流  
**关键词**: Neural JKO, Rejection Sampling, Continuous Normalizing Flows, Importance Sampling, 多模态分布

## 一句话总结

提出 Importance Corrected Neural JKO Sampling (Neural JKO IC)，将连续归一化流（CNF）的局部 JKO 步与基于重要性权重的拒绝重采样步交替使用，克服 Wasserstein 梯度流在多模态分布上的局部最优问题，同时保持独立同分布采样和密度可评估性。

## 研究背景与动机

从未归一化概率密度函数中采样是机器学习的核心问题。给定可积函数 $g: \mathbb{R}^d \to \mathbb{R}_{>0}$，目标是从 $q(x) = g(x)/Z_g$ 采样，其中归一化常数 $Z_g$ 未知。

传统方法分为两类：
- **MCMC 方法**（如 Langevin、HMC）：基于局部变换，**无法正确分配多模态分布的质量**
- **生成模型**（如归一化流、扩散模型）：使用反向 KL 散度，但该目标函数在目标分布非 log-concave 时是非凸的，**容易发生 mode collapse**

Neural JKO 通过正则化速度场将 CNF 训练与 JKO 方案关联，使其更稳定。但底层 Wasserstein 梯度流仍受非凸性困扰，在多模态目标上收敛缓慢或陷入次优解。

## 方法详解

### 整体框架

Neural JKO IC 交替执行两种步骤：
1. **Neural JKO 步骤**：通过 CNF 局部调整样本位置
2. **重要性拒绝步骤**：基于重要性权重非局部修正推断分布

### Neural JKO 方案

将 JKO 方案的 Wasserstein 近端算子用神经 ODE 参数化：

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mu_\tau^k}\left[-\log(g(z_\theta(x,\tau))) - \ell_\theta(x,\tau) + \omega_\theta(x,\tau)\right]$$

其中 $(z_\theta, \ell_\theta, \omega_\theta)$ 满足 ODE 系统：
- $\dot{z}_\theta = v_\theta(z_\theta, t)$（位置演化）
- $\dot{\ell}_\theta = \text{trace}(\nabla v_\theta)$（密度变化）
- $\dot{\omega}_\theta = \|v_\theta\|^2$（正则化项）

### 理论保证

**定理 3.3**：证明了 JKO 方案的速度场序列 $v_{\tau_l}$ 在 $\tau_l \to 0$ 时**强收敛**到 Wasserstein 梯度流的速度场。

### 重要性拒绝步骤

给定当前近似分布 $\mu$（密度 $p$）和目标分布 $\nu$（密度 $q$），对样本 $X \sim \mu$ 计算接受概率：

$$\alpha(X) = \min\left\{1, \frac{g(X)}{c \cdot f(X)}\right\}$$

以概率 $\alpha(X)$ 保留样本，否则从 $\mu$ 重新采样替换。

**定理 4.2 的关键结论**：
1. 拒绝步骤后的密度 $\tilde{p}(x) = p(x)(\alpha(x) + 1 - \mathbb{E}[\alpha(X)])$，**可显式计算**
2. $\text{KL}(\tilde{\mu}, \nu) \leq \text{KL}(\mu, \nu)$，即**每步 KL 散度单调递减**

超参数 $c$ 通过二分搜索确定，使得约 $r = 20\%$ 的样本被重采样。

### 密度可评估性

与 MCMC 方法不同，Neural JKO IC 在每步都能追踪样本的密度值，支持：
- 独立同分布采样
- 密度评估

## 实验关键数据

### 主实验：Energy Distance

| 分布 | MALA | HMC | DDS | CRAFT | Neural JKO | **Neural JKO IC** |
|------|------|-----|-----|-------|---------|-----------------|
| Mustache | 4.6e-2 | 1.7e-2 | 6.9e-2 | 9.2e-2 | 1.8e-2 | **2.9e-3** |
| shifted 8 Modes | 5.3e-3 | 4.1e-5 | 1.2e-2 | 5.2e-2 | 1.3e-1 | **1.2e-5** |

### 关键发现

- 在高维多模态目标上（最高 $d = 1600$），Neural JKO IC 在几乎所有测试分布上**显著优于**现有方法
- 纯 Neural JKO 在 shifted 8 Modes 上完全失败（energy distance 0.13），加入重要性修正后下降 4 个数量级至 1.2e-5
- 拒绝步骤有效修正了梯度流导致的模态权重错误分配

## 亮点与洞察

1. **局部与非局部结合**：CNF 做局部调整，拒绝采样做非局部修正，两者互补
2. **密度可追踪**：不同于 MCMC 生成非独立样本且密度未知，Neural JKO IC 生成 iid 样本且密度已知
3. **理论严谨**：证明了速度场的强收敛和 KL 单调递减
4. **自生成 proposal**：拒绝步骤的 proposal 分布由模型自身生成，避免了经典拒绝采样对 proposal 的敏感性

## 局限性

- 高维情况下重要性权重可能高度不平衡，拒绝率上升
- 需要能评估未归一化密度 $g(x)$，不适用于纯数据驱动场景
- 每次 JKO 步需训练一个新的 CNF，总训练开销较大
- 理论收敛需要目标函数满足 $\lambda$-凸性等条件

## 相关工作

- MCMC（MALA、HMC）
- Sequential Monte Carlo (SMC)
- 神经 ODE / 连续归一化流 (CNF)
- OT-flow（正则化速度场）
- Stochastic Normalizing Flows
- Stein Variational Gradient Descent (SVGD)

## 评分

⭐⭐⭐⭐⭐ — 理论贡献扎实（速度场强收敛证明），实验全面且在多模态高维目标上显著领先。方法设计优雅，将梯度流与拒绝采样的优势统一在一个框架中。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Evading Data Provenance in Deep Neural Networks](../../ICCV2025/multimodal_vlm/evading_data_provenance_in_deep_neural_networks.md)
- [\[ICML 2026\] Conditional Diffusion Sampling](../../ICML2026/multimodal_vlm/conditional_diffusion_sampling.md)
- [\[ICML 2026\] Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics](../../ICML2026/multimodal_vlm/dimension-free_multimodal_sampling_via_preconditioned_annealed_langevin_dynamics.md)
- [\[NeurIPS 2025\] Hierarchical Self-Attention: Generalizing Neural Attention Mechanics to Multi-Scale Problems](../../NeurIPS2025/multimodal_vlm/hierarchical_self-attention_generalizing_neural_attention_mechanics_to_multi-sca.md)
- [\[NeurIPS 2025\] CoIDO: Efficient Data Selection for Visual Instruction Tuning via Coupled Importance-Diversity Optimization](../../NeurIPS2025/multimodal_vlm/coido_efficient_data_selection_for_visual_instruction_tuning_via_coupled_importa.md)

</div>

<!-- RELATED:END -->
