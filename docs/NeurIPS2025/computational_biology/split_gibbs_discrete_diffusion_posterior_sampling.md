---
title: >-
  [论文解读] Split Gibbs Discrete Diffusion Posterior Sampling
description: >-
  [NeurIPS 2025][计算生物][离散扩散模型] 提出 SGDD（Split Gibbs Discrete Diffusion），一种基于分裂 Gibbs 采样原理的即插即用离散扩散后验采样算法，通过引入辅助变量和基于 Hamming 距离的正则化势函数，将后验采样分解为似然采样步和先验采样步交替进行，在 DNA 序列设计、离散图像逆问题和音乐填充等任务上大幅超越基线。
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "离散扩散模型"
  - "后验采样"
  - "Split Gibbs Sampling"
  - "逆问题"
  - "DNA序列设计"
---

# Split Gibbs Discrete Diffusion Posterior Sampling

**会议**: NeurIPS 2025  
**arXiv**: [2503.01161](https://arxiv.org/abs/2503.01161)  
**代码**: [GitHub](https://github.com/chuwd19/Split-Gibbs-Discrete-Diffusion-Posterior-Sampling)  
**领域**: 计算生物  
**关键词**: 离散扩散模型, 后验采样, Split Gibbs Sampling, 逆问题, DNA序列设计

## 一句话总结

提出 SGDD（Split Gibbs Discrete Diffusion），一种基于分裂 Gibbs 采样原理的即插即用离散扩散后验采样算法，通过引入辅助变量和基于 Hamming 距离的正则化势函数，将后验采样分解为似然采样步和先验采样步交替进行，在 DNA 序列设计、离散图像逆问题和音乐填充等任务上大幅超越基线。

## 研究背景与动机

**领域现状**：扩散模型后验采样方法在连续空间取得了显著进展（如 DPS、SMC、变分方法等），广泛应用于图像恢复和科学逆问题。然而这些方法依赖于似然函数 $\nabla_{\mathbf{x}_t} \log p(\mathbf{y}|\mathbf{x}_t)$ 的梯度信息，无法直接推广到离散状态空间。

**核心痛点**：在离散状态空间中，$\log p(\mathbf{y}|\mathbf{x})$ 仅在有限支撑上有定义，不存在有意义的梯度。现有的离散扩散后验采样方法（如 SVDD-PM）需要近似价值函数，类似强化学习中的技巧，难以调优且在复杂引导信号下性能受限。

**本文切入角度**：分裂 Gibbs 采样（Split Gibbs Sampling）框架天然不依赖似然函数的梯度。通过精心设计正则化势函数使先验步对应离散扩散模型的部分去噪过程，实现了离散扩散模型的即插即用后验采样。

## 方法详解

### 整体框架

目标：从后验分布 $p(\mathbf{x}|\mathbf{y}) \propto p(\mathbf{y}|\mathbf{x})p(\mathbf{x})$ 中采样，其中 $p(\mathbf{x})$ 由预训练离散扩散模型建模。

核心思路：引入辅助变量 $\mathbf{z}$，构造增广分布 $\pi(\mathbf{x}, \mathbf{z}; \eta) \propto \exp(-f(\mathbf{z};\mathbf{y}) - g(\mathbf{x}) - D(\mathbf{x}, \mathbf{z}; \eta))$。当正则化参数 $\eta \to 0$ 时，两个边缘分布 $\pi^X(\mathbf{x}; \eta)$ 和 $\pi^Z(\mathbf{z}; \eta)$ 都收敛到目标后验。通过 Gibbs 采样交替进行似然步和先验步。

### 关键设计

1. **基于 Hamming 距离的正则化势函数（核心创新）**：为了使先验步能够精确对应离散扩散模型的去噪过程，设计了特殊的势函数：
    $D(\mathbf{x}, \mathbf{z}; \eta) = d(\mathbf{x}, \mathbf{z}) \log \frac{1+(N-1)e^{-\eta}}{(N-1)(1-e^{-\eta})}$
   其中 $d(\mathbf{x}, \mathbf{z})$ 是 Hamming 距离。设计动机：当 $\eta \to 0^+$ 时，$D \to \infty$（除非 $\mathbf{x}=\mathbf{z}$），保证收敛；同时使先验采样步 $\mathbf{x}^{(k+1)} \sim \pi(\mathbf{x}, \mathbf{z}=\mathbf{z}^{(k)}; \eta) \propto p_0(\mathbf{x})(\tilde\beta/(1-\tilde\beta))^{d(\mathbf{z}^{(k)}, \mathbf{x})}$ 精确等价于离散扩散模型从噪声水平 $\sigma_t = \eta$ 对 $\mathbf{z}^{(k)}$ 的条件去噪。

2. **似然采样步**：每次迭代的似然步从 $\pi(\mathbf{x}=\mathbf{x}^{(k)}, \mathbf{z}; \eta) \propto \exp(-f(\mathbf{z};\mathbf{y}) - D(\mathbf{x}^{(k)}, \mathbf{z}; \eta))$ 中采样。由于非归一化概率密度可直接计算，可使用 Metropolis-Hastings 等 MCMC 方法高效采样。关键优势：不需要似然函数的梯度，天然适配离散空间。

3. **退火调度**：使用 $\{\eta_k\}_{k=0}^{K-1}$ 从大到小退火，加速马尔可夫链混合时间。大 $\eta$ 允许更多探索，小 $\eta$ 逐步逼近后验。

### 收敛性保证（Theorem 1）

SGDD 的收敛分析利用广义 Fisher 信息，在考虑不完美 score 函数和 CTMC 离散化误差的情况下，证明了平均相对 Fisher 散度以 $O(1/K)$ 的速率收敛：
$$\frac{1}{K}\sum_{k=0}^{K-1} \text{avg Fisher} \leq \frac{2\text{KL}(\pi_0\|\mu_0)}{Kt^*} + \frac{4M\epsilon}{c} + \frac{2MLt^*}{cH}$$
这比现有方法（如 SVDD-PM 依赖代理价值函数和无限粒子假设）的理论保证更强。

## 实验关键数据

### 主实验1：合成数据后验采样精度

| 维度 D | 方法 | Hellinger ↓ | TV ↓ |
|--------|------|------------|------|
| D=2 | SVDD-PM (M=100) | 0.275 | 0.292 |
| D=2 | SMC | 0.238 | 0.248 |
| D=2 | DPS | 0.182 | 0.176 |
| D=2 | **SGDD** | **0.149** | **0.125** |
| D=10 | SVDD-PM (M=100) | 0.448 | 0.494 |
| D=10 | DPS | 0.410 | 0.453 |
| D=10 | **SGDD** | **0.334** | **0.365** |

### 主实验2：DNA 增强子序列设计（奖励引导生成）

| 方法 | Pred-Activity (median) ↑ | Pred-Activity (avg) ↑ | ATAC Acc% ↑ | Log-likelihood ↑ |
|------|--------------------------|----------------------|-------------|-------------------|
| SVDD-PM | 5.41 | 5.08 | 49.9 | -241 |
| DRAKES w/ KL | 5.61 | 5.24 | 92.5 | -264 |
| CG | 2.90 | 2.76 | 0.0 | -265 |
| **SGDD (β=50)** | **9.14** | **8.96** | **93.0** | -261 |

SGDD 的 median 活性比之前 SOTA（DRAKES）高 **42%**。

### 主实验3：离散图像逆问题（MNIST XOR/AND）

| 任务 | 方法 | PSNR ↑ | Accuracy% ↑ |
|------|------|--------|-------------|
| XOR | SVDD-PM | 11.81 | 51.4 |
| XOR | SMC | 10.05 | 27.8 |
| XOR | **SGDD** | **20.17** | **91.2** |
| AND | SVDD-PM | 10.04 | 33.7 |
| AND | **SGDD** | **17.25** | **79.4** |

SGDD 在 XOR 任务上 PSNR 提升 **8.36 dB**。

### 消融实验

| 配置 | Hellinger (D=5) | 说明 |
|------|----------------|------|
| SGDD 完整版 | 0.214 | 含退火调度 |
| 固定 η（无退火） | ~0.35 | 退火至关重要 |
| H=5 Euler步 | ~0.30 | 需要足够步数 |
| H=20 Euler步 | 0.214 | 默认设置 |

### 关键发现

- 随着维度增加（D=2→10），所有方法的精度都下降，但 SGDD 下降最慢
- 当观测较稀疏时（如大面积遮挡），SGDD 能从多个可信模式采样（如数字1、4、7、9），展示了良好的后验多样性
- 引导强度 β 控制了奖励与先验之间的权衡：β 越大活性越高但 log-likelihood 下降

## 亮点与洞察

- Split Gibbs 框架避免了梯度计算，是离散后验采样最自然的选择
- 将正则化势函数巧妙设计为 Hamming 距离的函数，使先验步精确等价于扩散去噪，无需任何近似
- 理论保证比现有方法更强——不依赖代理价值函数且考虑了离散化误差
- 算法实现简洁，仅需现成的离散扩散模型和似然函数，真正即插即用

## 局限与展望

- 似然采样步使用 MCMC，在高维离散空间中混合速度可能较慢
- 退火调度 $\{\eta_k\}$ 的选择目前依赖经验，缺乏自适应策略
- 离散扩散模型本身的预训练质量直接影响后验采样质量
- 在更大规模的离散生成任务（如蛋白质序列设计）上尚未验证

## 相关工作与启发

- 与连续空间的 Split Gibbs 方法（PnP-DM, DiffPIR）类似，但将 $\ell_2$ 正则化推广到 Hamming 距离正则化
- 与 SVDD-PM（基于价值函数近似的方法）和 SMC 方法形成互补，SGDD 不需要价值函数训练
- 可能启发将 Split Gibbs 框架扩展到其他离散结构（如图结构、组合优化）

## 评分

- **新颖性**: ⭐⭐⭐⭐ Hamming 距离势函数设计精巧，但 Split Gibbs 框架本身已有先例
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖合成数据、DNA、图像、音乐四大场景，基线全面
- **写作质量**: ⭐⭐⭐⭐ 动机清晰、理论推导完整，但符号较多
- **价值**: ⭐⭐⭐⭐⭐ 为离散扩散后验采样提供了统一且高效的解决方案，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Constrained Discrete Diffusion](constrained_discrete_diffusion.md)
- [\[NeurIPS 2025\] Why Masking Diffusion Works: Condition on the Jump Schedule for Improved Discrete Diffusion](why_masking_diffusion_works_condition_on_the_jump_schedule_for_improved_discrete.md)
- [\[NeurIPS 2025\] Remasking Discrete Diffusion Models with Inference-Time Scaling](remasking_discrete_diffusion_models_with_inference-time_scaling.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)
- [\[ICML 2026\] Stein Diffusion Guidance: Training-Free Posterior Correction for Sampling Beyond High-Density Regions](../../ICML2026/computational_biology/stein_diffusion_guidance_training-free_posterior_correction_for_sampling_beyond_.md)

</div>

<!-- RELATED:END -->
