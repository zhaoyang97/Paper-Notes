---
title: >-
  [论文解读] MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control
description: >-
  [NeurIPS 2025][优化/理论][离散扩散模型] 提出 Masked Diffusion Neural Sampler (MDNS)，基于连续时间马尔可夫链（CTMC）的随机最优控制理论，通过对齐路径测度来训练离散神经采样器，在状态空间基数高达 $10^{122}$ 的 Ising/Potts 模型上准确采样，大幅超越现有学习型基线。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "离散扩散模型"
  - "神经采样器"
  - "随机最优控制"
  - "连续时间马尔可夫链"
  - "Ising模型"
  - "Potts模型"
---

# MDNS: Masked Diffusion Neural Sampler via Stochastic Optimal Control

**会议**: NeurIPS 2025  
**arXiv**: [2508.10684](https://arxiv.org/abs/2508.10684)  
**作者**: Yuchen Zhu, Wei Guo, Jaemoo Choi (Georgia Tech), Guan-Horng Liu (FAIR at Meta), Yongxin Chen, Molei Tao (Georgia Tech)  
**代码**: [github.com/yuchen-zhu-zyc/MDNS](https://github.com/yuchen-zhu-zyc/MDNS)  
**领域**: 优化  
**关键词**: 离散扩散模型, 神经采样器, 随机最优控制, 连续时间马尔可夫链, Ising模型, Potts模型  

## 一句话总结

提出 Masked Diffusion Neural Sampler (MDNS)，基于连续时间马尔可夫链（CTMC）的随机最优控制理论，通过对齐路径测度来训练离散神经采样器，在状态空间基数高达 $10^{122}$ 的 Ising/Potts 模型上准确采样，大幅超越现有学习型基线。

## 研究背景与动机

### 问题背景
从未归一化目标分布 $\pi(x) = \frac{1}{Z} e^{-U(x)}$ 中采样是统计物理、贝叶斯推断、组合优化等领域的基本问题。传统 MCMC 方法（Langevin MC、Metropolis-Hastings、Glauber dynamics）在高维多模态分布上收敛缓慢。近年来，连续空间的扩散模型神经采样器（如 DIS、NETS）取得显著进展，但**离散状态空间**的扩散采样方法仍未被充分研究。

### 已有工作的不足
- **LEAPS** [HAJ25]：通过 Gumbel softmax 技巧将 CTMC 轨迹松弛为连续概率向量来保持可微性，但引入有偏梯度估计和数值不稳定性，即使在低维也无法收敛到正确目标分布
- **传统 MCMC**：在大状态空间和低温（多模态）分布上混合时间指数增长
- 离散扩散模型主要用于生成建模（文本、蛋白质），但几乎未被用于从已知能量函数的分布中采样

### 核心动机
设计一套基于随机最优控制的训练框架，克服 CTMC 轨迹不连续带来的优化困难，实现高维离散分布的高效准确采样。

## 方法详解

### 问题建模
- **目标**：给定势函数 $U$，从 $\pi(x) = \frac{1}{Z} e^{-U(x)}$ 在离散状态空间 $\mathcal{X}_0 = \{1,\ldots,N\}^D$ 上采样
- **方法**：学习 CTMC 的生成器 $Q^u$，驱动初始分布 $p_{\text{init}}$ 在终止时间 $T$ 达到 $\pi$
- 通过匹配受控路径测度 $\mathbb{P}^u$ 与最优路径测度 $\mathbb{P}^*$ 来学习 $Q^u$，等价于最小化 KL 散度 $\text{KL}(\mathbb{P}^u \| \mathbb{P}^*)$

### 随机最优控制公式
将采样问题形式化为 CTMC 上的随机最优控制（SOC）问题：

$$\min_u \mathbb{E}_{X \sim \mathbb{P}^u} \left[ \int_0^T \sum_{y \neq X_t} \left( Q_t^u \log \frac{Q_t^u}{Q_t^0} - Q_t^u + Q_t^0 \right)(X_t, y) \, dt - r(X_T) \right]$$

其中 $r = -U - \log p_{\text{base}}$。最优生成器为参考生成器的乘性扰动：$Q_t^*(x,y) = Q_t^0(x,y) \exp(V_t(y) - V_t(x))$，$V_t$ 为值函数。

### Masked Diffusion 参考过程
选择参考路径测度 $\mathbb{P}^0$ 为 masked discrete diffusion 的生成过程：
- 初始分布：全 mask 序列 $p_{\text{mask}}$
- 终止分布：均匀分布 $p_{\text{unif}}$
- 参考生成器：$Q_t^0(x, x^{d \leftarrow n}) = \frac{\gamma(t)}{N} \mathbf{1}_{x^d = \mathbf{M}}$

**关键性质**（Lemma 2）：此参考路径测度是无记忆的（memoryless），保证 SOC 问题解的存在唯一性。

**最优生成器结构**（Lemma 3）：$Q_t^*(x, x^{d \leftarrow n}) = \gamma(t) \Pr_{X \sim \pi}(X^d = n | X^{\text{UM}} = x^{\text{UM}}) \mathbf{1}_{x^d = \mathbf{M}}$，即 score 网络 $s_\theta$ 只需预测目标分布 $\pi$ 关于未 mask 位置的条件边际分布。

### 四种学习目标
由于 CTMC 轨迹为纯跳过程，目标函数对参数 $\theta$ 不可微。本文提出四种无需可微性的学习目标：

1. **RERF (Relative-entropy with REINFORCE)**：利用 REINFORCE 技巧得到 KL 散度梯度的无偏估计
   $$\mathcal{F}_{\text{RERF}} = \mathbb{E}_{X \sim \mathbb{P}^{\bar{u}}} W^{\bar{u}}(X) W^u(X)$$

2. **LV (Log-variance)**：最小化 RN 导数对数的方差
   $$\mathcal{F}_{\text{LV}} = \text{Var}_{X \sim \mathbb{P}^{\bar{u}}} W^u(X)$$

3. **CE (Cross-entropy)**：反向 KL 散度，对 $\mathbb{P}^u$ 凸，优化景观良性
   $$\mathcal{F}_{\text{CE}} = \mathbb{E}_{X \sim \mathbb{P}^{\bar{u}}} \frac{1}{Z} e^{W^{\bar{u}}(X)} W^u(X)$$

4. **WDCE (Weighted Denoising Cross-entropy)**：核心创新——将采样轨迹终端 $X_T$ 作为重要性加权样本，对其重新 mask 后计算 denoising cross-entropy loss，避免沿整条轨迹反向传播
   $$\mathcal{F}_{\text{WDCE}} = \mathbb{E}_{X \sim \mathbb{P}^{\bar{u}}} \left[ \frac{e^{W^{\bar{u}}(X)}}{Z} \mathbb{E}_\lambda \left[ w(\lambda) \mathbb{E}_{\mu_\lambda(\tilde{x}|X_T)} \sum_{d: \tilde{x}^d = \mathbf{M}} -\log s_\theta(\tilde{x})_{d, X_T^d} \right] \right]$$

WDCE 的优势：每次 score 模型调用的全部输出都被利用（而非仅用一个元素），且通过 replay buffer 和 $R$ 次重采样进一步摊销 RN 导数的 $O(D)$ 计算开销。

### 理论保证
- **采样保证**（Proposition 1）：路径测度的 KL 散度上界直接控制采样分布的 KL 散度
- **归一化常数估计**（Proposition 2）：$\hat{Z} = e^{W^u(X)}$ 是 $Z$ 的无偏估计；训练至 $\text{KL} \leq \varepsilon^2/2$ 即可保证 $|\hat{Z}/Z - 1| \leq \varepsilon$ 以至少 $3/4$ 概率成立

## 实验关键数据

### 实验1：4x4 Ising 模型——学习目标对比

$J=1, h=0.1, \beta_{\text{high}}=0.28$，状态空间 $|\mathcal{X}_0| = 2^{16} \approx 65536$。各目标训练 1000 步，batch size 256。

| 学习目标 | ESS ↑ | TV ↓ | KL ↓ | $\chi^2$ ↓ | $\widehat{\text{KL}}(\mathbb{P}^u\|\mathbb{P}^*)$ ↓ | $|\log\hat{Z}|$ 误差 ↓ |
|---------|-------|------|------|-------|---------|---------|
| $\mathcal{F}_{\text{RERF}}$ | 0.9621 | 0.0799 | 0.0380 | 0.0845 | 0.0188 | **3e-5** |
| $\mathcal{F}_{\text{LV}}$ | **0.9713** | **0.0748** | **0.0348** | **0.0714** | **0.0141** | 4.6e-4 |
| $\mathcal{F}_{\text{CE}}$ | 0.9513 | 0.0833 | 0.0393 | 0.0903 | 0.0248 | 9.9e-4 |
| $\mathcal{F}_{\text{WDCE}}$ | 0.9644 | 0.0799 | 0.0382 | 0.0868 | 0.0177 | 3.0e-4 |
| Baseline (MH) | / | 0.0667 | 0.0325 | 0.0628 | / | / |

在除 $\log\hat{Z}$ 外的所有指标上，排名为 LV > WDCE > RERF > CE。四种目标均能学到接近真实分布的采样器。

### 实验2：16x16 Ising/Potts 模型——高维扩展

状态空间 Ising: $2^{256}$（$\approx 10^{77}$），Potts ($q=3$): $3^{256}$（$\approx 10^{122}$）。使用 WDCE 训练。

**Ising 模型**（$J=1, h=0$）：

| 温度 | 方法 | 磁化强度误差 ↓ | 2点关联误差 ↓ | ESS ↑ |
|------|------|-------------|------------|-------|
| $\beta_{\text{low}}=0.6$ | **MDNS** | **9.9e-3** | 2.4e-3 | **0.981** |
| | LEAPS | 2.4e-2 | 5.8e-1 | 0.261 |
| | MH | 1.9e-2 | **7.7e-4** | / |
| $\beta_{\text{critical}}=0.4407$ | **MDNS** | **3.7e-3** | **2.0e-3** | **0.933** |
| | LEAPS | 7.4e-3 | 1.6e-1 | 0.384 |
| | MH | 4.6e-3 | 2.5e-3 | / |
| $\beta_{\text{high}}=0.28$ | **MDNS** | 8.5e-3 | **1.0e-3** | 0.962 |
| | LEAPS | 7.4e-3 | 1.6e-3 | **0.987** |
| | MH | **6.1e-3** | 1.1e-3 | / |

**Potts 模型**（$q=3, J=1$）：

| 温度 | 方法 | 磁化强度误差 ↓ | 2点关联误差 ↓ | ESS ↑ |
|------|------|-------------|------------|-------|
| $\beta_{\text{low}}=1.2$ | **MDNS** | **1.3e-3** | **8.8e-5** | **0.933** |
| | LEAPS | 2.9e-1 | 2.5e-1 | 0.012 |
| | MH | 7.4e-1 | 5.6e-1 | / |
| $\beta_{\text{critical}}=1.005$ | **MDNS** | **4.3e-3** | **2.9e-3** | **0.875** |
| | LEAPS | 2.7e-1 | 2.0e-1 | 0.004 |
| | MH | 5.2e-1 | 3.5e-1 | / |
| $\beta_{\text{high}}=0.5$ | **MDNS** | **2.2e-3** | **5.8e-4** | 0.983 |
| | LEAPS | 2.9e-3 | 1.2e-3 | **0.991** |
| | MH | 3.5e-2 | 1.6e-2 | / |

在低温/临界温度下 MDNS 相比 LEAPS 优势极为显著（ESS 0.933 vs 0.012），MH 在 Potts 模型上连续运行 20+ 小时仍无法混合。

## 亮点

- **理论-算法统一**：将离散采样严格建模为 CTMC 随机最优控制问题，最优生成器结构与 masked diffusion 的 score 函数自然对应，理论优雅
- **四种无偏/低方差学习目标**：完全规避 Gumbel softmax 的有偏梯度问题，WDCE 通过重要性加权 + 重采样实现高维可扩展训练
- **极端高维验证**：在状态空间基数 $10^{122}$ 的 Potts 模型上成功采样，ESS 高达 0.93，LEAPS 仅 0.01，MH 完全失败
- **warm-up 策略**：先在高温（简单分布）训练再迁移到低温（多模态分布），有效帮助模型定位模式
- **归一化常数估计**：副产品地提供 $Z$ 的无偏估计，有严格概率保证

## 局限与展望

- **仅验证了统计物理模型**：Ising 和 Potts 模型具有规则格点结构，在图结构分布或组合优化问题上的表现未知
- **warm-up 策略缺乏系统化研究**：当前实验中 warm-up 温度和步数人工设定，未自动化
- **计算开销**：虽然 WDCE 比 CE/LV 高效，但仍需要生成完整 CTMC 轨迹来获取重要性权重
- **与预训练模型的结合**：论文提出可用于微调预训练离散扩散模型（给定奖励函数），但未实际验证
- **理论分析不完整**：关于 masked diffusion 插值路径优于几何退火路径的猜想仅停留在推测阶段
- **仅支持有限离散空间**：框架限于 $\{1,\ldots,N\}^D$，不适用于连续或无穷可数状态空间

## 与相关工作的对比

- **LEAPS [HAJ25]**：使用几何退火 $\pi_\eta \propto e^{-\eta U}$ 和 Gumbel softmax 松弛，梯度有偏，在低温/大状态空间失败；MDNS 使用 masked diffusion 插值 + 无偏目标，全面优于 LEAPS
- **DIS/PIS [ZC22, VGD23]**：连续空间的 SDE 神经采样器，MDNS 是其离散空间对应物，核心思想类似（路径测度匹配），但需全新的技术手段处理离散跳跃过程
- **NETS [AVE25]**：连续空间非平衡输运采样器，同样基于 SOC 理论，MDNS 将该理论扩展至离散 CTMC
- **Masked Diffusion Models [LME24, Ou+25]**：用于生成建模（从数据中学习），MDNS 将其反向用于从已知能量函数采样
- **MCMC 方法**：MH 在 Potts 模型上 20 小时无法混合，MDNS 100k 步训练即可准确采样

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次将 masked diffusion + SOC 框架用于离散分布采样，四种学习目标的设计系统完整
- 实验充分度: ⭐⭐⭐⭐ — Ising/Potts 多温度多尺度实验详尽，消融研究全面，但缺乏物理模型之外的应用验证
- 写作质量: ⭐⭐⭐⭐⭐ — 理论推导严谨，从 SOC 到 masked diffusion 的连接清晰自然，算法伪代码完整
- 价值: ⭐⭐⭐⭐⭐ — 为离散空间采样提供了强大且可扩展的新范式，在超高维问题上展现出突破性性能

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)
- [\[ICLR 2026\] A Schrödinger Eigenfunction Method for Long-Horizon Stochastic Optimal Control](../../ICLR2026/optimization/a_schrödinger_eigenfunction_method_for_long-horizon_stochastic_optimal_control.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)

</div>

<!-- RELATED:END -->
