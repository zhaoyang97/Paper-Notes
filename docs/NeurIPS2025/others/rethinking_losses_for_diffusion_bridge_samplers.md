---
title: >-
  [论文解读] Rethinking Losses for Diffusion Bridge Samplers
description: >-
  [NeurIPS 2025][扩散桥采样器] 本文揭示了扩散桥采样器中流行的 Log Variance (LV) 损失存在的理论缺陷——不满足数据处理不等式且梯度与 rKL 不等价——并提出用 log-derivative trick 计算 rKL 梯度（rKL-LD），在多个基准上一致性超越 LV 损失，且训练更加稳定、对超参数不敏感。
tags:
  - "NeurIPS 2025"
  - "扩散桥采样器"
  - "损失函数"
  - "反向KL散度"
  - "Log Variance损失"
  - "可学习扩散系数"
---

# Rethinking Losses for Diffusion Bridge Samplers

**会议**: NeurIPS 2025  
**arXiv**: [2506.10982](https://arxiv.org/abs/2506.10982)  
**代码**: [GitHub](https://github.com/sanokows/RethinkingLossesForDiffusionBridgeSamplers)  
**领域**: LLM评测  
**关键词**: 扩散桥采样器, 损失函数, 反向KL散度, Log Variance损失, 可学习扩散系数

## 一句话总结

本文揭示了扩散桥采样器中流行的 Log Variance (LV) 损失存在的理论缺陷——不满足数据处理不等式且梯度与 rKL 不等价——并提出用 log-derivative trick 计算 rKL 梯度（rKL-LD），在多个基准上一致性超越 LV 损失，且训练更加稳定、对超参数不敏感。

## 研究背景与动机

从未归一化分布中采样是计算物理、化学和贝叶斯推断中的基本问题。扩散桥采样器（如 DBS 和 CMCD）通过学习前向和反向扩散过程之间的传输路径来实现采样，是当前该领域的 SOTA 方法。

现有训练损失主要有两种：

**rKL-R（重参数化技巧计算 rKL）**: 在多步扩散中容易出现梯度消失/爆炸问题，实际表现不佳

**LV 损失（Log Variance）**: 无需反向传播期望，被广泛认为优于 rKL-R

然而，作者发现了一个被忽视的关键问题：LV 损失与 rKL 损失的梯度等价性**仅在只学习反向扩散过程参数时成立**。当涉及扩散桥（前向过程也有可学参数）或学习扩散系数时，这种等价性不再成立。更重要的是，LV 损失不满足数据处理不等式（DPI），这意味着它作为潜变量模型的训练目标缺乏理论基础。

这一发现促使作者重新审视 rKL 损失，转而采用 log-derivative trick 来计算其梯度（rKL-LD），既避免了重参数化技巧的梯度问题，又保持了 rKL 的理论优势。

## 方法详解

### 整体框架

核心思路是将扩散桥采样器的训练损失从 LV 替换为 rKL-LD，同时引入可学习的扩散系数来自适应调节采样过程中的随机性。方法适用于两种主流扩散桥架构：DBS 和 CMCD。

### 关键设计

1. **rKL-LD 梯度估计器**: 使用 log-derivative trick（也称 REINFORCE/score function trick）来计算 rKL 散度关于模型参数的梯度。对于反向过程参数 $\alpha$，梯度为：

$$\nabla_\alpha^{LD} D_{KL}(q_{\alpha,\nu} \| p_{\phi,\nu}) = \mathbb{E}_{X_{0:T} \sim q_{\alpha,\nu}} \left[ \left(\log \frac{q_{\alpha,\nu}(X_{0:T})}{p_{\phi,\nu}(X_{0:T})} - b \right) \nabla_\alpha \log q_{\alpha,\nu}(X_{0:T}) \right]$$

   其中 $b$ 是控制变量，用于降低方差。对于前向过程参数 $\phi$，梯度简化为 $-\mathbb{E}[\nabla_\phi \log p_{\phi,\nu}]$。关键的设计动机在于：(a) 避免重参数化技巧的梯度消失/爆炸问题；(b) 保持 rKL 基于数据处理不等式的理论保证。

2. **LV 损失的理论分析与梯度差异揭示**: 作者推导了 LV 损失关于共享参数 $\nu$ 的梯度，发现它等价于 Jeffrey 距离的梯度（在最优点处），而非 rKL 的梯度。具体地，对于共享参数：

$$\nabla_\nu D_{LV}^{q_{\alpha,\nu}^*}(q_{\alpha,\nu}, p_{\phi,\nu}) = \mathbb{E}\left[\left(\log \frac{q_{\alpha,\nu}}{p_{\phi,\nu}} - b\right) \nabla_\nu \log \frac{q_{\alpha,\nu}}{p_{\phi,\nu}}\right]$$

   这与 rKL-LD 的梯度不同，且作者通过反例证明 LV 损失不满足数据处理不等式 $D_f(\pi_0(X_0) \| q_{\alpha,\nu}(X_0)) \leq D_f(p_{\phi,\nu}(X_{0:T}) \| q_{\alpha,\nu}(X_{0:T}))$，从而质疑 LV 作为扩散桥训练目标的合理性。

3. **可学习扩散系数**: 将 SDE 的扩散系数 $\sigma_{\text{diff}} \in \mathbb{R}^N$ 设为可学参数（每个维度独立学习），自适应调节 exploration-exploitation 权衡。较大的系数增加随机性以更好覆盖多模态分布，较小的系数减少噪声以提高精度。关键发现是可学习扩散系数在 rKL-LD 下一致性提升性能，而在 LV 损失下常导致训练不稳定甚至发散。

### 损失函数 / 训练策略

训练损失为 rKL 散度，使用 log-derivative trick 计算梯度并结合控制变量降低方差。对 DBS，前向和反向过程使用独立的神经网络；对 CMCD，使用共享控制函数 $u_\gamma$ 加上可学习的插值函数 $\eta(t)$。训练 40000 轮，batch size 2000，128 步扩散，并对学习率、初始 $\sigma_{\text{diff}}$ 和先验方差进行网格搜索。

## 实验关键数据

### 主实验（贝叶斯学习基准）

| 方法 | Seeds (26d) | Sonar (61d) | Credit (25d) | Brownian (32d) | LGCP (1600d) |
|------|-------------|-------------|--------------|----------------|--------------|
| CMCD: LV ☆ | -74.13 | -109.53 | -504.91 | -0.05 | 460.84 |
| CMCD: LV (学σ) | -73.53 | -109.66 | **-628.39†** | **-6.05†** | **447.74†** |
| CMCD: rKL-LD ☆ | -74.10 | -109.25 | -504.88 | 0.36 | 466.73 |
| CMCD: rKL-LD (学σ) | **-73.45** | **-108.83** | **-504.58** | **1.06** | 465.80 |
| DBS: LV ☆ | -74.12 | -110.66 | -506.21 | -9.39† | 460.48 |
| DBS: rKL-LD (学σ) | **-73.50** | **-108.88** | **-504.71** | **0.85** | **469.89** |

*ELBO (↑越高越好)；☆ = 不学习扩散系数；† = 训练发散*

### 合成目标基准

| 方法 | GMM-40 Sinkhorn(↓) | GMM-40 ELBO(↑) | MoS-10 Sinkhorn(↓) | MoS-10 ELBO(↑) |
|------|---------------------|-----------------|---------------------|-----------------|
| CMCD: LV ☆ | 2559.20 | -37.37 | 1263.78 | -52.52 |
| CMCD: rKL-LD (学σ) | **2301.16** | **-21.94** | **915.52** | -34.93 |
| DBS: LV ☆ | 2073.09 | -35.45 | 1220.27 | -57.49 |
| DBS: rKL-LD (学σ) | 2133.50 | **-30.44** | **1051.34** | **-43.66** |

### 关键发现

1. **rKL-LD 一致性优于 LV**: 在贝叶斯任务上，CMCD+rKL-LD 在 5 个中的 4 个任务上显著优于 LV；DBS+rKL-LD 在 5 个中的 4 个任务上显著优于 LV
2. **LV + 可学习扩散系数 = 灾难**: LV 损失下学习 $\sigma_{\text{diff}}$ 在 CMCD 的 3/5 和 DBS 的 4/5 任务上导致训练发散
3. **rKL-LD + 可学习扩散系数 = 双赢**: rKL-LD 下学习 $\sigma_{\text{diff}}$ 从不导致性能恶化，且常带来显著提升
4. **超参数鲁棒性**: rKL-LD 对初始 $\sigma_{\text{diff}}$ 值不敏感，不同初始化均收敛到相近的最优解

## 亮点与洞察

- 理论贡献扎实：通过反例证明 LV 不满足 DPI，并系统分析了三种参数类型（$\alpha$, $\phi$, $\nu$）下 LV 与 rKL-LD 梯度的差异
- 发现不同维度的最优扩散系数显著不同（图1中间），说明维度级别的自适应噪声调节是有意义的
- 从实践角度解决了扩散桥社区一个长期困扰：为什么 LV 经常需要精心调参且训练不稳定

## 局限与展望

- rKL-LD 仍然因 rKL 的 mode-seeking 特性而存在 mode collapse 风险（在超参不当时）
- 当前只考虑了时间不变的扩散系数，时间依赖的 $\sigma_{\text{diff}}(t)$ 是未来方向
- 未在结合 off-policy buffer + MCMC 更新的设置下与 LV 做全面比较

## 相关工作与启发

- 与 GFlowNet 中的 Trajectory Balance 损失（本质就是 LV 损失）有直接联系，该分析对 GFlowNet 社区也有启示
- 离散域中的扩散采样器已成功使用 rKL-LD（如组合优化和自旋晶格统计物理），本文将其推广到连续域扩散桥
- 启发：损失函数选择需要考虑参数共享结构，当参数在前向/反向过程间共享时需特别注意

## 评分

- 新颖性: ⭐⭐⭐⭐ 理论发现有意义（LV 不满足 DPI），但 rKL-LD 本身不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖两种架构（CMCD/DBS）、三种损失（rKL-R/LV/rKL-LD）、贝叶斯和合成目标，消融研究充分
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，但符号较多需要反复对照
- 价值: ⭐⭐⭐⭐⭐ 扩散桥采样器社区的重要实践指导，直接可用的改进方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Rethinking PCA Through Duality](rethinking_pca_through_duality.md)
- [\[NeurIPS 2025\] Adjoint Schrödinger Bridge Sampler](adjoint_schrödinger_bridge_sampler.md)
- [\[ICML 2025\] Gradient Aligned Regression via Pairwise Losses](../../ICML2025/others/gradient_aligned_regression_via_pairwise_losses.md)
- [\[NeurIPS 2025\] Modeling Cell Dynamics and Interactions with Unbalanced Mean Field Schrödinger Bridge](modeling_cell_dynamics_and_interactions_with_unbalanced_mean_field_schrödinger_b.md)
- [\[ICML 2025\] Rethinking Aleatoric and Epistemic Uncertainty](../../ICML2025/others/rethinking_aleatoric_and_epistemic_uncertainty.md)

</div>

<!-- RELATED:END -->
