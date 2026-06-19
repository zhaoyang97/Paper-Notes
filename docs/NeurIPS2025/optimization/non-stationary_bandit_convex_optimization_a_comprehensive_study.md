---
title: >-
  [论文解读] Non-Stationary Bandit Convex Optimization: A Comprehensive Study
description: >-
  [NeurIPS 2025][优化/理论][Bandit凸优化] 系统研究了非平稳环境下的Bandit凸优化问题，提出两个算法（TEWA-SE和cExO），统一建立了三种非平稳度量（切换数S、总变差Δ、路径长度P）下的遗憾上下界，多个设定下达到极小极大最优。 Bandit凸优化（BCO）是一类基础的序贯决策问题：学习者从连续…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "Bandit凸优化"
  - "非平稳环境"
  - "动态遗憾"
  - "自适应遗憾"
  - "极小极大最优"
---

# Non-Stationary Bandit Convex Optimization: A Comprehensive Study

**会议**: NeurIPS 2025  
**arXiv**: [2506.02980](https://arxiv.org/abs/2506.02980)  
**代码**: 无  
**领域**: 优化  
**关键词**: Bandit凸优化, 非平稳环境, 动态遗憾, 自适应遗憾, 极小极大最优

## 一句话总结

系统研究了非平稳环境下的Bandit凸优化问题，提出两个算法（TEWA-SE和cExO），统一建立了三种非平稳度量（切换数S、总变差Δ、路径长度P）下的遗憾上下界，多个设定下达到极小极大最优。

## 研究背景与动机

Bandit凸优化（BCO）是一类基础的序贯决策问题：学习者从连续域中选择动作，仅能观察到一个查询点上的（带噪声的）损失函数值，无法获得梯度信息。在非平稳环境中，最优动作随时间变化，需要设计算法最小化"动态遗憾"（与时变比较器序列的差距）。

已有研究分散地处理不同的非平稳度量：有的关注损失函数总变差 $\Delta$，有的关注路径长度 $P$。本文的目标是**系统统一**这些结果，建立完整的BCO非平稳遗憾界图景，覆盖凸和强凸两种函数类。

## 方法详解

### 整体框架

建立了遗憾概念之间的转化关系链：自适应遗憾 $R^{\text{ada}}$ → 切换遗憾 $R^{\text{swi}}$ → 动态遗憾 $R^{\text{dyn}}$ 和路径长度遗憾 $R^{\text{path}}$。因此只需分析自适应遗憾，即可系统推导出所有度量下的界。

### 关键设计

1. **TEWA-SE（多项式时间算法）**: 基于"sleeping experts"框架设计。核心思想是并行运行多个学习率不同的"专家"算法，用倾斜指数加权平均聚合。每个专家在其生命周期内运行投影在线梯度下降。创新点在于：(a) 使用单点梯度估计 $\mathbf{g}_t = (d/h) y_t \boldsymbol{\zeta}_t$ 替代精确梯度，$h$ 为平滑参数，$\boldsymbol{\zeta}_t$ 是球面均匀采样；(b) 设计强凸代理损失 $\ell_t^\eta(\mathbf{x}) = -\eta \mathbf{g}_t^\top(\mathbf{x}_t - \mathbf{x}) + \eta^2 G^2 \|\mathbf{x}_t - \mathbf{x}\|^2$，利用曲率信息从 $\mathcal{O}(\sqrt{|I|})$ 降低到 $\mathcal{O}(\log |I|)$；(c) 使用几何覆盖方案调度专家，指数网格分配学习率，保证每轮仅 $\mathcal{O}(\log T)$ 个活跃专家。

2. **cExO（改进凸损失的界）**: 针对一般凸函数，TEWA-SE仅得到次优 $T^{3/4}$ 速率。cExO基于指数权重在离散化动作空间上运行，关键修改是在"裁剪单纯形" $\tilde{\Delta} = \Delta(\mathcal{C}) \cap [\gamma, 1]^{|\mathcal{C}|}$ 上运行镜像下降，防止算法过度集中于单一动作，从而更好地适应环境变化。虽非多项式时间可计算，但在维度 $d$ 的依赖上提供了更紧的界。

3. **极小极大下界**: 对强凸函数，通过构造硬实例函数类证明 $\Omega(d\sqrt{ST} \wedge d^{2/3}\Delta^{1/3}T^{2/3})$ 下界，与TEWA-SE上界匹配。对路径长度遗憾也建立了 $\Omega((d^2 P)^{2/5} T^{3/5})$ 下界，改进了已有的 $d\sqrt{PT}$ 下界。

### 损失函数 / 训练策略

学习者每轮选择动作 $\mathbf{z}_t \in \Theta$，观测 $y_t = f_t(\mathbf{z}_t) + \xi_t$，目标最小化动态遗憾。平滑参数 $h = \min(\sqrt{d} \mathsf{B}^{-1/4}, r)$ 控制梯度估计精度与偏差的tradeoff。并通过Bandit-over-Bandit框架实现参数无关版本。
强凸情况下TEWA-SE的自适应遗憾为 $\mathcal{O}(d\sqrt{\mathsf{B}}/\alpha)$，不需要知道强凸参数 $\alpha$。

## 实验关键数据

### 主实验

本文为纯理论贡献，无实验数据。主要结果通过定理和推论呈现。

| 设定 | 度量 | TEWA-SE | cExO | 下界 |
|------|------|---------|------|------|
| 强凸+已知S | $R^{\text{swi}}$ | $d\sqrt{ST}$ (最优) | — | $d\sqrt{ST}$ |
| 强凸+已知Δ | $R^{\text{dyn}}$ | $d^{2/3}\Delta^{1/3}T^{2/3}$ (最优) | — | $d^{2/3}\Delta^{1/3}T^{2/3}$ |
| 凸+已知S | $R^{\text{swi}}$ | $\sqrt{d}S^{1/4}T^{3/4}$ | $d^{5/2}\sqrt{ST}$ (最优) | $\sqrt{ST}$ |
| 凸+已知P | $R^{\text{path}}$ | $d^{2/5}P^{1/5}T^{4/5}$ | $d^{5/3}P^{1/3}T^{2/3}$ (新) | $(d^2 P)^{2/5}T^{3/5}$ |

### 消融实验

| 配置 | 结果 | 说明 |
|------|------|------|
| 线性代理损失 vs 强凸代理损失 | $\mathcal{O}(\sqrt{|I|})$ vs $\mathcal{O}(\log |I|})$ | 强凸代理是利用曲率的关键 |
| 定义域裁剪 vs 无裁剪 | 自适应遗憾 vs 静态遗憾 | 裁剪将静态遗憾转化为自适应遗憾 |
| 已知参数 vs BoB框架 | 额外 $d^{1/2}T^{3/4}$ 开销 | 参数无关版本代价可接受 |

### 关键发现

- 强凸情况下TEWA-SE在 $d, T, S, \Delta$ 四个参数上同时达到极小极大最优
- 凸函数的BCO与OCO之间的 $T^{3/4}$ vs $T^{1/2}$ gap反映了零阶vs一阶信息的基本分离
- 路径长度遗憾的下界改进（$T^{3/5}$ vs $T^{1/2}$）利用了 $P = o(T)$ 假设

## 亮点与洞察

- 建立了BCO非平稳遗憾的完整理论图景并首次证明多个设定下的最优性
- 遗憾概念之间的转化关系链（Proposition 1）提供了优雅的统一框架，从自适应遗憾出发推导出所有其他遗憾
- 代理损失设计 $\eta^2 G^2 \|\mathbf{x}_t - \mathbf{x}\|^2$ 巧妙绕过了OCO中 $\mathbb{E}[\|\mathbf{g}_t\|]$ 和 $\mathbb{E}[\|\mathbf{g}_t\|^2]$ 之间条件的限制
- 将sleeping experts与梯度裁剪结合到bandit设定，技术含量很高
- TEWA-SE无需知道强凸参数α即可自适应利用曲率，继承了MetaGrad的优雅性质
- Proposition 1中从path-length到switching regret的新转化使用了简单的几何论证

## 局限与展望

- cExO不是多项式时间算法，凸函数的高效最优算法仍是开放问题
- 维度 $d$ 的依赖在凸情况下不紧（cExO为 $d^{5/2}$），实际问题中维度可能很高
- 路径长度遗憾的最优速率（特别是 $d$ 的依赖）尚未确定，上下界之间存在gap
- 参数无关版本的Bandit-over-Bandit框架引入额外 $T^{5/6}$ 或 $T^{3/4}$ 开销，能否通过更巧妙的方法消除有待研究
- 纯理论工作，缺少数值实验验证
- 噪声假设为sub-Gaussian，对heavy-tail噪声的推广是重要方向
- 一点函数值估计的trade-off（平滑参数 $h$ 的选择）在实际中难以精确调优

## 相关工作与启发

- 继承自MetaGrad系列工作的tilted exponential weighting思想，适配到bandit设定
- cExO的裁剪技巧来源于MAB文献中的非平稳处理方法
- 为非平稳BCO研究提供了系统的基线，后续工作可关注缩小凸函数情况的gap
- Flaxman et al.的一点梯度估计框架在BCO中的核心地位：本文的所有上界最终依赖于这种估计
- 与OCO文献的清晰对比：一阶信息可获得 $\sqrt{\mathsf{B}}$ 自适应遗憾，零阶仅能获得 $\mathsf{B}^{3/4}$
- 提出的path-length下界 $(d^2 P)^{2/5} T^{3/5}$ 改进了已有 $d\sqrt{PT}$ 下界，利用了 $P = o(T)$ 的关键假设
- MAB文献中的非平稳技术（change-detection、discounting、sliding window）不能直接推广到连续域
- 开放问题：凸函数的多项式时间最优算法，以及路径长度遗憾的精确最优速率

## 评分

- 新颖性: ⭐⭐⭐⭐ 建立了BCO非平稳问题的首个系统性理论框架
- 实验充分度: ⭐⭐ 纯理论工作，无实验验证
- 写作质量: ⭐⭐⭐⭐⭐ 论述清晰，遗憾概念关系图直观，表格总结精炼
- 价值: ⭐⭐⭐⭐ 为非平稳Bandit凸优化奠定了坚实的理论基础和完整的最优性图景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICML 2025\] Right Now, Wrong Then: Non-Stationary Direct Preference Optimization under Preference Drift](../../ICML2025/optimization/right_now_wrong_then_non-stationary_direct_preference_optimization_under_prefere.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)
- [\[NeurIPS 2025\] Beyond Õ(√T) Constraint Violation for Online Convex Optimization with Adversarial Constraints](beyond_tildeosqrtt_constraint_violation_for_online_convex_optimization_with_adve.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)

</div>

<!-- RELATED:END -->
