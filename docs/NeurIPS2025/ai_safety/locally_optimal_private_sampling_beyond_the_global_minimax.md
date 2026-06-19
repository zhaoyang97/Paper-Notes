---
title: >-
  [论文解读] Locally Optimal Private Sampling: Beyond the Global Minimax
description: >-
  [NeurIPS 2025][AI安全][本地差分隐私] 在本地差分隐私（LDP）下的采样问题中，提出局部minimax：框架，利用公共数据分布 $P_0$ 定义的邻域约束，推导出闭式最优采样器，在理论和实验上均一致优于全局minimax采样器：。 领域现状： 本地差分隐私（LDP）允许用户在设备上随机化数据后再共享…
tags:
  - "NeurIPS 2025"
  - "AI安全"
  - "本地差分隐私"
  - "隐私采样"
  - "minimax最优"
  - "f-散度"
  - "公共数据"
---

# Locally Optimal Private Sampling: Beyond the Global Minimax

**会议**: NeurIPS 2025  
**arXiv**: [2510.09485](https://arxiv.org/abs/2510.09485)  
**代码**: [GitHub](https://github.com/hradghoukasian/private_sampling)  
**领域**: AI安全  
**关键词**: 本地差分隐私, 隐私采样, minimax最优, f-散度, 公共数据

## 一句话总结

在本地差分隐私（LDP）下的采样问题中，提出**局部minimax**框架，利用公共数据分布 $P_0$ 定义的邻域约束，推导出闭式最优采样器，在理论和实验上均**一致优于全局minimax采样器**。

## 研究背景与动机

**领域现状**: 本地差分隐私（LDP）允许用户在设备上随机化数据后再共享，无需信任中央服务器，被Google、Apple、Microsoft广泛部署。隐私采样（private sampling）是从分布 $P$ 中生成满足LDP约束的单个样本的任务。

**现有痛点**: Park et al. (NeurIPS'24) 建立了全局minimax框架，找到最坏情况下最优的采样器。但全局minimax是**过于悲观的**——它假设对手可以选择任意分布，而现实中数据分布不会是最坏情况（退化分布）。

**核心矛盾**: 全局minimax最优采样器为所有输入分布使用同一变换，无法利用"数据分布接近某个已知参考分布"这一实际先验信息，导致隐私-效用权衡次优。

**本文目标**: 建立**局部minimax**框架——当我们知道数据分布 $P$ 在某个已知公共分布 $P_0$ 附近时，能否设计更好的隐私采样器？

**切入角度**: 用 $E_\gamma$-散度（hockey-stick divergence）定义分布邻域 $N_\gamma(P_0) = \{P: E_\gamma(P\|P_0) = E_\gamma(P_0\|P) = 0\}$，在此约束下求解minimax最优。关键发现：**局部minimax风险等于将分布类限制到邻域后的全局minimax风险**。

**核心 idea**: 将全局minimax框架从pure LDP推广到functional LDP，然后证明局部minimax最优采样器就是将全局最优采样器限制到 $P_0$ 邻域，进而推导出不依赖 $f$-散度选择的闭式解。

## 方法详解

### 整体框架

两层理论构建：
1. **全局minimax under functional LDP**: 推广Park et al.的结果到一般的functional LDP（包含approximate DP和Gaussian LDP）
2. **局部minimax**: 利用全局结果推导邻域约束下的最优采样器

### 关键设计

#### 1. 全局minimax最优采样器（Theorem 3.4）

- **功能**: 对分布类 $\tilde{\mathcal{P}}_{c_1,c_2,h}$ 和任意 $f$-散度，找到满足 $g$-FLDP 的minimax最优采样器
- **核心公式**: 
$$q_g^\star(P)(x) = \lambda^\star p(x) + (1 - \lambda^\star) h(x)$$
  其中 $\lambda^\star_{c_1,c_2,g} = \inf_{\beta \geq 0} \frac{e^\beta + \frac{c_2 - c_1}{1-c_1}(1 + g^*(-e^\beta)) - 1}{(1-c_1)e^\beta + c_2 - 1}$
- **物理含义**: 以概率 $\lambda^\star$ 从原分布 $P$ 采样，以 $1-\lambda^\star$ 从参考分布 $h$ 采样——完美的隐私-效用混合

#### 2. 局部minimax最优采样器 — functional LDP版（Theorem 4.1）

- **功能**: 在 $N_\gamma(P_0)$ 邻域上求解minimax最优采样器
- **核心思路**: 证明局部minimax风险等于设定 $c_1 = 1/\gamma$, $c_2 = \gamma$, $h = p_0$ 时的全局minimax风险
- **公式**: 直接将全局结果代入 $c_1, c_2, h$ 即得局部解
- **设计动机**: 邻域 $N_\gamma(P_0)$ 约束了密度比 $p(x)/p_0(x) \in [1/\gamma, \gamma]$，这自然对应全局框架中的分布类参数

#### 3. 局部minimax最优采样器 — pure LDP版（Theorem 5.1）

- **功能**: 在pure $\varepsilon$-LDP下给出更强的逐点最优采样器
- **核心公式**:
$$q(x) = \text{clip}\left(\frac{1}{r_P} p(x); \frac{\gamma+1}{\gamma+e^\varepsilon} p_0(x), \frac{(\gamma+1)e^\varepsilon}{\gamma+e^\varepsilon} p_0(x)\right)$$
- **关键优势**: 这是**非线性采样器**，通过clipping操作逐点适应每个输入分布 $P$，比线性采样器（Theorem 4.1）严格更优（Proposition 5.2）
- **设计动机**: 线性采样器对所有分布使用统一变换，而非线性采样器可以instance-optimal地最小化每个具体 $P$ 的 $f$-散度

### 理论性质

- **f-散度无关**: 最优采样器的结构不依赖于具体选择哪个 $f$-散度（KL、TV、Hellinger等）
- **逐点优越性**: $D_f(P \| \mathbf{Q}^\star_\varepsilon(P)) \leq D_f(P \| \mathbf{Q}^\star_{g_\varepsilon}(P))$ 对所有 $P \in N_\gamma(P_0)$
- **涵盖多种隐私定义**: pure LDP, approximate LDP, Gaussian LDP均为functional LDP的特例

## 实验关键数据

### 离散空间（$k=20$, uniform参考）

| $\varepsilon$ | 度量 | 全局最优 | 局部最优 |
|-----------|------|---------|---------|
| 0.1 | KL | ~0.18 | ~0.065 |
| 0.5 | KL | ~0.14 | ~0.035 |
| 1.0 | KL | ~0.10 | ~0.018 |
| 2.0 | KL | ~0.04 | ~0.004 |
| 0.1 | TV | ~0.14 | ~0.09 |
| 1.0 | TV | ~0.08 | ~0.04 |

局部采样器在所有隐私级别和所有散度度量下一致优于全局采样器。

### 连续空间（1D Laplace混合, 100个客户端分布）

| $\varepsilon$ | 度量 | 全局worst-case | 局部worst-case |
|-----------|------|---------|---------|
| 0.1 | KL | ~0.18 | ~0.04 |
| 0.5 | KL | ~0.12 | ~0.02 |
| 1.0 | KL | ~0.07 | ~0.008 |
| 2.0 | KL | ~0.025 | ~0.002 |
| 0.1 | Hellinger² | ~0.06 | ~0.02 |
| 1.0 | Hellinger² | ~0.028 | ~0.003 |

### 关键发现

1. **局部采样器一致且显著优于全局**: 在所有设置下，worst-case $f$-散度降低2-10倍
2. **低隐私预算（小$\varepsilon$）下优势更大**: $\varepsilon=0.1$时KL散度从0.18降到0.04
3. **Pure LDP非线性采样器优于functional LDP线性采样器**: 逐点clipping带来额外增益
4. **Gaussian LDP下同样有效**: 实验详见附录

## 亮点与洞察

1. **优雅的理论还原**: 局部minimax → 特殊参数化的全局minimax，一步到位
2. **f-散度无关性**: 极其强大的理论保证——无论用什么散度度量效用都是最优的
3. **Clipping即最优**: pure LDP下的非线性采样器本质上就是一个clipping操作，简洁且直觉
4. **公共数据的充分利用**: 与Zamanlooy et al.不同，这里公共数据**确实降低了minimax风险**
5. **从全局最悲观到局部现实**: 理论上证明了利用先验信息可以大幅改善隐私-效用权衡

## 局限与展望

1. **仅在合成数据上验证**: 缺乏高维真实数据集的实验
2. **计算复杂度**: clipping采样器在高维时计算成本高（涉及归一化常数 $r_P$）
3. **邻域参数 $\gamma$ 需要选择**: 实际中如何确定 $P_0$ 和 $\gamma$ 是开放问题
4. **单样本设定**: 仅生成一个样本的场景，多样本扩展未涉及
5. **整数约束 $\gamma \in \mathbb{N}$**: 技术性限制，虽然可通过放大克服
6. **$E_\gamma$-散度邻域可推广至一般 $f$-散度邻域**: 留作未来工作

## 相关工作与启发

- **与Park et al. (NeurIPS'24)的关系**: 直接在其全局框架上构建，关键创新是引入邻域约束
- **与Zamanlooy et al.的区别**: 他们要求采样器保留公共先验（硬约束），而本文将公共数据作为邻域中心（软约束）——后者更灵活且确实降低风险
- **Feldman et al.**: 使用相同的 $E_\gamma$ 邻域定义，但在密度估计任务而非采样
- **启发点**: 局部minimax在其他DP问题中的应用（分布估计、学习、生成）

## 评分

⭐⭐⭐⭐ (4/5)
- 理论贡献优美精炼，从全局到局部的还原令人印象深刻
- 但缺乏真实数据实验和高维可扩展性验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Beyond Last-Click: An Optimal Mechanism for Ad Attribution](beyond_last-click_an_optimal_mechanism_for_ad_attribution.md)
- [\[NeurIPS 2025\] Nearly-Linear Time Private Hypothesis Selection with the Optimal Approximation Factor](nearly-linear_time_private_hypothesis_selection_with_the_optimal_approximation_f.md)
- [\[NeurIPS 2025\] Differentially Private Bilevel Optimization: Efficient Algorithms with Near-Optimal Rates](differentially_private_bilevel_optimization_efficient_algorithms_with_near-optim.md)
- [\[NeurIPS 2025\] Private Continual Counting of Unbounded Streams](private_continual_counting_of_unbounded_streams.md)
- [\[NeurIPS 2025\] Optimal Adjustment Sets for Nonparametric Estimation of Weighted Controlled Direct Effect](optimal_adjustment_sets_for_nonparametric_estimation_of_weighted_controlled_dire.md)

</div>

<!-- RELATED:END -->
