---
title: >-
  [论文解读] Enhancing Parallelism in Decentralized Stochastic Convex Optimization
description: >-
  [ICML2025][优化/理论][去中心化SGD] 提出 Decentralized Anytime SGD (DAT-SGD)，通过在渐变平均查询点上计算梯度来缓解共识距离偏差，将去中心化随机凸优化的并行度上界从 $\mathcal{O}(\rho^{1/2} N^{1/4})$ 提升至 $\mathcal{O}(\rho \sqrt{N})$，在高连通拓扑下首次匹配中心化学习的速率。
tags:
  - "ICML2025"
  - "优化/理论"
  - "去中心化SGD"
  - "随机凸优化"
  - "并行度上界"
  - "gossip通信"
  - "Anytime SGD"
---

# Enhancing Parallelism in Decentralized Stochastic Convex Optimization

**会议**: ICML2025  
**arXiv**: [2506.00961](https://arxiv.org/abs/2506.00961)  
**代码**: 无  
**领域**: 去中心化优化 / 分布式学习  
**关键词**: 去中心化SGD, 随机凸优化, 并行度上界, gossip通信, Anytime SGD

## 一句话总结

提出 Decentralized Anytime SGD (DAT-SGD)，通过在渐变平均查询点上计算梯度来缓解共识距离偏差，将去中心化随机凸优化的并行度上界从 $\mathcal{O}(\rho^{1/2} N^{1/4})$ 提升至 $\mathcal{O}(\rho \sqrt{N})$，在高连通拓扑下首次匹配中心化学习的速率。

## 研究背景与动机

**分布式学习的并行度瓶颈**：分布式学习通过多台机器并行处理数据来加速训练，但增加机器数 $M$ 超过临界点后会降低收敛效率。这一问题在去中心化系统中尤为严重——节点间通过 gossip 协议通信，稀疏拓扑放大了性能退化。

**中心化与去中心化的差距**：中心化 Minibatch SGD (Dekel et al., 2012) 允许使用至多 $\mathcal{O}(\sqrt{N})$ 台机器而不损失统计效率。然而，已有去中心化方法（D-SGD, Koloskova et al., 2020）的并行度上界仅为 $\mathcal{O}(\rho^{1/2} N^{1/4})$，即使在近完全图拓扑（$\rho \approx 1$）下仍存在明显差距。

**根本原因——共识距离**：去中心化训练中各节点模型不同步，导致梯度估计相对全局共识产生偏差。D-SGD 中共识距离 $\Xi_t = \frac{1}{M}\sum_i \|w_t^i - \bar{w}_t\|^2$ 的界为 $\mathcal{O}(\eta^2/\rho)$，是限制并行度的核心瓶颈。

**核心问题**：去中心化方法是否在理想通信条件下也无法充分利用并行度？本文给出否定答案。

## 方法详解

### 框架：Anytime SGD → 去中心化扩展

**Anytime SGD 回顾**（Cutkosky, 2019）：与标准 SGD 在当前迭代 $w_t$ 上计算梯度不同，Anytime SGD 在过去迭代的加权平均 $x_t$ 上计算梯度。给定非负权重 $\{\alpha_t\}$，更新规则为：

$$w_{t+1} = w_t - \eta \alpha_t g_t$$

$$x_{t+1} = \frac{\alpha_{1:t-1}}{\alpha_{1:t}} x_t + \frac{\alpha_t}{\alpha_{1:t}} w_{t+1}$$

其中 $\alpha_{1:t} = \sum_{\tau=1}^t \alpha_\tau$，$g_t$ 是在 $x_t$（而非 $w_t$）处的随机梯度。关键洞察：平均查询点 $x_t$ 变化比迭代 $w_t$ 更缓慢，天然适合控制去中心化中的共识距离。

### DAT-SGD 算法

将 Anytime SGD 扩展到去中心化设置，每轮每台机器 $i$ 执行：

1. **采样与梯度计算**：$g_t^i = \nabla f_i(x_t^i, z_t^i)$（在查询点 $x_t^i$ 而非 $w_t^i$ 处）
2. **局部更新**：
    - $w_{t+1/2}^i = w_t^i - \eta \alpha_t g_t^i$
    - $x_{t+1/2}^i = \frac{\alpha_{1:t-1}}{\alpha_{1:t}} x_t^i + \frac{\alpha_t}{\alpha_{1:t}} w_{t+1/2}^i$
3. **Gossip 通信**（同时交换 $w$ 和 $x$）：
    - $w_{t+1}^i = \sum_j w_{t+1/2}^j P_{ij}$
    - $x_{t+1}^i = \sum_j x_{t+1/2}^j P_{ij}$

其中 $P$ 为 gossip 矩阵（对称双随机矩阵），$\rho = 1 - |\lambda_2| \in (0,1]$ 为谱间隙。

### 关键设计：共识距离的紧致控制

D-SGD 中共识距离递推为 $\Xi_{t+1} \leq (1-\rho/2)\Xi_t + C \cdot G^2\eta^2/\rho$，解出 $\Xi_t \leq \mathcal{O}(\eta^2/\rho)$。

DAT-SGD 引入查询点共识距离 $\Gamma_t = \frac{1}{M}\sum_i \|x_t^i - \bar{x}_t\|^2$，其递推为：

$$\Gamma_{t+1} \leq \left(1-\frac{\rho}{2}\right)\Gamma_t + \frac{C}{\rho t^2}(\Xi_t + G^2\eta^2)$$

由于加权平均的 $1/t^2$ 衰减因子，查询点共识距离受到更强的收缩控制。代入 $\Xi_t$ 的界后：

$$\Gamma_{t+1} \leq \left(1-\frac{\rho}{2}\right)\Gamma_t + \frac{2C \cdot G^2\eta^2}{\rho^2 t^2}$$

相比 D-SGD 中的 $\mathcal{O}(\eta^2/\rho)$ 界，DAT-SGD 的查询点共识距离显著更小，从而放松了对学习率 $\eta$ 的约束。

### 权重选择与学习率

采用线性权重 $\alpha_t = t$，学习率：

$$\eta = \min\left\{\frac{1}{24LT}, \frac{\rho^2}{K}, \frac{D_1\sqrt{M}}{\sqrt{3}\sigma T^{3/2}}, \sqrt{\frac{D_1}{2K\tilde{\sigma}}} \frac{\rho}{T}\right\}$$

其中 $D_1 = \|w_1 - x^*\|$，$K^2 = 5120L^2$，$\tilde{\sigma}^2 = 2\sigma^2 + \zeta^2$。

## 理论结果

### 主定理（Theorem 4.1）

在凸光滑函数、有界噪声方差 $\sigma^2$ 和有界异质性 $\zeta^2$ 假设下：

$$\mathbb{E}[f(\bar{x}_T) - f^*] \leq \mathcal{O}\left(\frac{\sigma D_1}{\sqrt{MT}} + \frac{D_1^{3/2}\sqrt{L\tilde{\sigma}}}{\rho T} + \frac{LD_1^2}{T}\right)$$

| 指标 | D-SGD (Koloskova+2020) | DAT-SGD (本文) |
|------|----------------------|---------------|
| 收敛率 | $\mathcal{O}\left(\frac{\sigma^2}{M\epsilon^2}+\frac{\sigma\sqrt{\rho}+\zeta}{\rho\epsilon^{3/2}}+\frac{1}{\rho\epsilon}\right)$ | $\mathcal{O}\left(\frac{\sigma^2}{M\epsilon^2}+\frac{\sqrt{\sigma}+\sqrt{\zeta}}{\rho\epsilon}+\frac{1}{\epsilon}\right)$ |
| 并行度上界 | $\mathcal{O}(\rho^{1/2} N^{1/4})$ | $\mathcal{O}(\rho\sqrt{N})$ |

### 不同拓扑下的并行度对比

| 拓扑 | $1/\rho$ | D-SGD | DAT-SGD |
|------|----------|-------|---------|
| Ring | $\mathcal{O}(M^2)$ | $\mathcal{O}(N^{1/8})$ | $\mathcal{O}(N^{1/6})$ |
| Torus | $\mathcal{O}(M)$ | $\mathcal{O}(N^{1/6})$ | $\mathcal{O}(N^{1/4})$ |
| 近完全图 | $\approx 1$ | $\mathcal{O}(\rho^{1/2} N^{1/4})$ | $\mathcal{O}(\rho\sqrt{N})$ |

### 关键发现

- **近完全图**：DAT-SGD 达到 $\mathcal{O}(\rho\sqrt{N})$，当 $\rho = \Omega(1)$ 时恢复中心化 $\mathcal{O}(\sqrt{N})$ 的并行度，**首次消除去中心化与中心化的差距**
- **一般拓扑**：并行度也全面提升（Ring 从 $N^{1/8}$ 到 $N^{1/6}$，Torus 从 $N^{1/6}$ 到 $N^{1/4}$）
- **瞬态迭代复杂度**：$\mathcal{O}(M/\rho^2)$，比 D-SGD 改善 $M^2$ 倍
- **局部迭代收敛**（Corollary 4.2）：每台机器的局部迭代也收敛，额外项 $M\tilde{\sigma}D_1/\rho^2 T^2$ 不影响并行度上界

## 亮点与洞察

- **简洁优雅**：算法修改极小——仅将梯度查询点从当前迭代替换为加权平均，却获得并行度的根本性提升
- **理论突破**：在高连通拓扑下首次匹配中心化学习速率，回答了"去中心化是否本质受限"这一开放问题
- **通用性**：Anytime SGD 框架此前已用于异步和局部训练，本文将其优势延伸至去中心化设置
- **直觉清晰**：共识距离的改善源于查询点变化速度更慢（$1/t^2$ 递推因子），打破了 D-SGD 的瓶颈

## 局限与展望

- **仅限凸设置**：分析在 SCO 框架下，未扩展到非凸优化（实际深度学习的主要场景）
- **通信开销翻倍**：需同时交换 $w_t^i$ 和 $x_t^i$，通信量是 D-SGD 的两倍
- **无实验验证**：全文为理论贡献，缺少实际深度学习任务上的验证
- **异质性假设较强**：采用全局有界异质性 $\zeta^2$，未放松到仅在最优点处有界的更弱假设
- **下界缺失**：并行度上界 $\mathcal{O}(\rho\sqrt{N})$ 是否紧致（是否存在匹配下界）尚不清楚
- **固定拓扑**：分析假设固定通信图，未覆盖时变拓扑或随机 gossip

## 相关工作与启发

- **D-SGD** (Lian+2017, Koloskova+2020)：标准去中心化SGD，本文的基线方法
- **梯度追踪** (Koloskova+2021)：消除数据异质性影响但并行度仍受限
- **Anytime SGD** (Cutkosky+2019)：本文核心工具，渐变查询点框架
- **异步训练** (Aviv+2021)：Anytime 框架在异步设置的应用
- **Local SGD** (Dahan & Levy, 2024)：Anytime 框架在局部训练的应用
- **去中心化动量** (He+2022)：非凸设置用动量替换SGD，并行度 $\mathcal{O}((\rho\sqrt{N})^{2/3})$

## 评分 ⭐⭐⭐⭐

理论贡献扎实，在去中心化凸优化并行度这一经典问题上取得显著突破，算法设计简洁、分析深刻。但局限于凸设置且无实验，实际影响力需要后续非凸扩展来验证。

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](../../NeurIPS2025/optimization/isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[ICML 2025\] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization](a_near-optimal_single-loop_stochastic_algorithm_for_convex_finite-sum_coupled_co.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[NeurIPS 2025\] Problem-Parameter-Free Decentralized Bilevel Optimization](../../NeurIPS2025/optimization/problem-parameter-free_decentralized_bilevel_optimization.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)

</div>

<!-- RELATED:END -->
