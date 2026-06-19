---
title: >-
  [论文解读] Plasticity as the Mirror of Empowerment
description: >-
  [NEURIPS2025][可塑性] 本文提出广义有向信息（GDI）：作为度量智能体可塑性（plasticity）的信息论工具，揭示可塑性是赋权（empowerment）的"镜像"——两者使用相同度量、仅方向相反，并证明了两者之间存在严格的张力约束（tension bound）。 智能体的两个基本能力：任何智能体都具备两个核…
tags:
  - "NEURIPS2025"
  - "可塑性"
  - "赋权"
  - "信息论"
  - "广义有向信息"
  - "智能体设计"
---

# Plasticity as the Mirror of Empowerment

**会议**: NEURIPS2025  
**arXiv**: [2505.10361](https://arxiv.org/abs/2505.10361)  
**代码**: 无  
**领域**: 其他  
**关键词**: 可塑性, 赋权, 信息论, 广义有向信息, 智能体设计  

## 一句话总结
本文提出**广义有向信息（GDI）**作为度量智能体可塑性（plasticity）的信息论工具，揭示可塑性是赋权（empowerment）的"镜像"——两者使用相同度量、仅方向相反，并证明了两者之间存在严格的张力约束（tension bound）。

## 研究背景与动机

**智能体的两个基本能力**：任何智能体都具备两个核心能力——被观测所塑造的能力（可塑性）和影响未来观测的能力（赋权）。缺乏任一能力的系统难以被视为真正的智能体。

**赋权已有成熟定义**：Klyubin et al. (2005) 首次将赋权定义为智能体对未来可观测状态的控制量度，此后被广泛用于内在动机、安全对齐、技能发现等领域。

**可塑性缺乏统一形式化**：虽然"可塑性"在神经科学（突触可塑性）、生物学（环境响应性）和机器学习（可塑性丧失问题）中均有讨论，但各领域定义零散，缺乏与赋权对等的通用数学定义。

**有向信息的局限**：Massey (1990) 的有向信息要求两个序列等长且从时间起点开始，无法灵活衡量任意时间窗口上的信息流。

**连续学习中的可塑性丧失**：Dohare et al. (2021, 2024) 和 Lyle et al. (2023) 等工作表明神经网络在持续学习中易丧失可塑性，但缺乏统一的理论框架来量化这一现象。

**核心目标**：构建一个通用的、以智能体为中心的可塑性度量，使其与赋权处于同等理论地位，并揭示两者的内在联系。

## 方法详解

### 整体框架

本文基于最小假设的智能体-环境交互模型：智能体 λ 和环境 e 共享一个接口 (A, O)，通过离散时间步交替发送动作和观测信号。核心思路是通过扩展有向信息来同时定义可塑性和赋权，从而揭示它们的对称结构。

### 关键设计

**模块1：广义有向信息（GDI）**

传统有向信息 I(X₁:n → Y₁:n) = Σᵢ I(X₁:ᵢ; Yᵢ | Y₁:ᵢ₋₁) 仅适用于等长序列且从时间起点开始。GDI 将其推广到任意区间 [a:b] 和 [c:d]：

$$\mathbb{I}(X_{a:b} \rightarrow Y_{c:d}) = \sum_{i=\max(a,c)}^{d} \mathbb{I}(X_{a:\min(b,i)}; Y_i \mid X_{1:a-1}, Y_{1:i-1})$$

GDI 严格推广了有向信息（当 a=c=1, b=d=n 时退化为有向信息），同时满足时间一致性（未来序列对过去的 GDI 为零）、区间可加性、和**守恒律**（Theorem 3.5）：

$$\mathbb{I}(X_{a:b}; Y_{c:d} \mid X_{1:a-1}, Y_{1:c-1}) = \mathbb{I}(X_{a:b} \rightarrow Y_{c:d}) + \mathbb{I}(Y_{c:d} \hookrightarrow X_{a:b})$$

**模块2：可塑性的形式化定义**

基于 GDI，对任意智能体 λ、环境集合 E 和时间区间 [a:b]→[c:d]，定义可塑性为：

$$\mathfrak{P}_{a:b}^{c:d}(\lambda, \mathcal{E}) = \max_{e \in \mathcal{E}} \mathbb{I}(O_{a:b} \rightarrow A_{c:d})$$

直觉上：观测序列 O 对动作序列 A 的影响越大，智能体越"可塑"。该定义满足一系列理想性质：

| 性质 | 说明 |
|------|------|
| 非负性 | P(λ, E) ≥ 0 |
| 零可塑性的充要条件 | 当且仅当观测不影响动作时为零 |
| 确定性智能体也可有可塑性 | 确定性策略也能被观测塑造 |
| 环境集合单调性 | E_small ⊆ E_big → P(λ, E_small) ≤ P(λ, E_big) |
| 零可塑性示例 | 开环智能体、常数智能体、仅依赖历史长度的智能体 |

**模块3：可塑性-赋权的镜像关系**

GDI 同样可以推广赋权的定义为 E(Λ, e) = max_λ I(A_{a:b} → O_{c:d})。由于智能体和环境在数学上是对称的（交换 A 和 O 即可互换），Proposition 4.6 证明：

$$\mathfrak{E}(\lambda) = \mathfrak{P}(e), \quad \mathfrak{P}(\lambda) = \mathfrak{E}(e)$$

即：**智能体的赋权等于环境的可塑性，智能体的可塑性等于环境的赋权**。

### 核心理论结果：张力定理（Theorem 4.8）

对于任何智能体-环境对 (λ, e) 和时间区间 [a:b], [c:d]，令 m = min{(b-a+1)log|O|, (d-c+1)log|A|}，则：

$$\mathfrak{E}_{a:b}^{c:d}(\lambda, e) + \mathfrak{P}_{c:d}^{a:b}(\lambda, e) \leq m$$

该上界是**紧的**：存在极端情况使得一方达到 m 而另一方为零。这意味着智能体无法在同一时间窗口内同时最大化赋权和可塑性——提高对环境的控制必然降低被环境塑造的能力。

## 实验关键数据

### 主实验：Q-learning 在两臂 Bernoulli 赌博机上的可塑性与赋权

实验使用蒙特卡洛估计器在 [1:3]→[2:5] 区间上估计可塑性和赋权。

**实验1：ε-greedy 探索参数对可塑性的影响**

| ε 值 | 可塑性趋势 | 解释 |
|------|-----------|------|
| ε = 0（纯贪心） | 最高 | 动作完全由观测驱动的 Q 值决定 |
| ε = 0.5 | 中等 | 一半动作随机、一半由观测驱动 |
| ε = 1（纯随机） | 零 | 动作与观测完全无关 |

**实验2：初始 Q 值（乐观/悲观）对可塑性与赋权的影响**

| 初始 Q 值 | 可塑性 | 赋权 | 两者之和 |
|-----------|--------|------|----------|
| Q₀ = -1（悲观） | 较高 | 较低 | < m |
| Q₀ = 0（中性） | 中等 | 中等 | < m |
| Q₀ = 1（乐观） | 较低 | 最高 | < m |

关键发现：(1) 赋权在该设定中普遍高于可塑性；(2) 乐观初始化提升赋权（更多探索→更多控制）；(3) 张力上界 m 在所有实验中均得到验证，两者之和从未超过 m。

### 思想实验：走廊环境

在一个包含 n+1 个房间的走廊中，每个房间有一个开关和灯。从左到右，智能体对灯的控制从 0/n 增加到 n/n。最左侧房间（无控制）最大化可塑性，最右侧房间（完全控制）最大化赋权，中间房间平滑插值——直观展示了 Pareto 前沿上的张力。

## 亮点与洞察

1. **概念突破**：首次为可塑性提供通用、数学精确的定义，使其与赋权处于对等地位
2. **优雅的镜像结构**：可塑性和赋权使用同一度量、仅方向相反，揭示了智能体-环境交互的深层对称性
3. **GDI 的独立价值**：广义有向信息严格推广了 Massey 的有向信息，保留所有性质同时支持任意时间窗口，对信息论和因果推断领域有独立贡献
4. **张力定理的实践意义**：定量揭示了"适应性 vs 控制力"的不可兼得，为智能体设计提供了新的约束视角
5. **极简假设**：理论仅要求离散接口和有限集合，不假设 MDP、马尔可夫性或任何特定学习算法

## 局限性

1. **纯理论贡献**：实验仅为简单的两臂赌博机验证，缺乏大规模或实际 RL 环境中的实证
2. **确定性环境下可塑性为零**：对确定性环境，定义给出零可塑性，这虽有理论解释但与直觉可能冲突
3. **未涉及目标/奖励**：整个框架不包含目标导向行为的讨论，限制了对实际 RL 算法的直接指导
4. **GDI 计算复杂度**：蒙特卡洛估计 GDI 的计算成本随状态空间指数增长，大规模应用不现实
5. **暂未连接到可塑性丧失**：虽然动机涉及连续学习中的可塑性丧失，但未给出具体的连接或解决方案

## 相关工作与启发

| 方向 | 代表工作 | 本文关联 |
|------|---------|---------|
| 赋权 | Klyubin et al. (2005), Capdepuy (2011) | 将赋权与可塑性统一到 GDI 框架 |
| 可塑性丧失 | Dohare et al. (2024), Lyle et al. (2023) | 提供可塑性的通用定义，可为这些工作提供理论基础 |
| 稳定性-可塑性困境 | Carpenter & Grossberg (1988) | 张力定理为该困境的形式化提供了新视角 |
| 有向信息 | Massey (1990), Massey & Massey (2005) | GDI 严格推广了有向信息并扩展了守恒律 |
| 通用智能体 | Hutter (2004), Abel et al. (2023) | 采用相同的最小假设智能体-环境交互框架 |
| 内在动机 | Mohamed & Rezende (2015) | 可塑性可作为赋权之外的另一种内在驱动信号 |

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ — 首次将可塑性形式化并揭示其与赋权的镜像关系，概念贡献卓越
- 实验充分度: ⭐⭐ — 仅有简单赌博机实验，缺乏大规模实证验证
- 写作质量: ⭐⭐⭐⭐⭐ — 论证层层递进，从公理化要求到定义到定理到实验，结构极为清晰
- 价值: ⭐⭐⭐⭐ — 理论框架优美且有深远影响潜力，但离实际应用还有距离

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Meta-learning three-factor plasticity rules for structured credit assignment with sparse feedback](meta-learning_three-factor_plasticity_rules_for_structured_credit_assignment_wit.md)
- [\[ICLR 2026\] FIRE: Frobenius-Isometry Reinitialization for Balancing the Stability-Plasticity Tradeoff](../../ICLR2026/others/fire_frobenius_isometry_reinitialization.md)
- [\[NeurIPS 2025\] A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)
- [\[NeurIPS 2025\] Structure-Aware Spectral Sparsification via Uniform Edge Sampling](structure-aware_spectral_sparsification_via_uniform_edge_sampling.md)
- [\[NeurIPS 2025\] Evolutionary Prediction Games](evolutionary_prediction_games.md)

</div>

<!-- RELATED:END -->
