---
title: >-
  [论文解读] Enforcing Axioms for AI Alignment under Loss-Based Rules
description: >-
  [ICLR 2026][LLM对齐][RLHF] 在线性社会选择框架下，基于损失的奖励模型（包括多项式奖励）无法保证 Pareto 最优性（PO），但当训练数据均匀覆盖嵌入空间时可在极限中恢复 PO——为宪法风格对齐提供了可证明的数据设计方案。 领域现状：RLHF 及其变体（Constitutional AI、NLHF）是当…
tags:
  - "ICLR 2026"
  - "LLM对齐"
  - "RLHF"
  - "Pareto最优性"
  - "社会选择"
  - "Constitutional AI"
  - "奖励模型公理"
  - "数据设计"
---

# Enforcing Axioms for AI Alignment under Loss-Based Rules

**会议**: ICLR 2026  
**代码**: 无  
**领域**: LLM Alignment / 社会选择理论  
**关键词**: RLHF, Pareto最优性, 社会选择, Constitutional AI, 奖励模型公理, 数据设计

## 一句话总结

在线性社会选择框架下，基于损失的奖励模型（包括多项式奖励）无法保证 Pareto 最优性（PO），但当训练数据均匀覆盖嵌入空间时可在极限中恢复 PO——为宪法风格对齐提供了可证明的数据设计方案。

## 研究背景与动机

**领域现状**：RLHF 及其变体（Constitutional AI、NLHF）是当前主流大模型对齐范式，核心步骤是在二元偏好数据上最小化损失以训练奖励模型，再以此指导策略优化。Constitutional AI 进一步引入少量"原则"（如 HHH：helpfulness / honesty / harmlessness）作为对比判断的引导器，使原则扮演"选民"的角色。

**现有痛点**：Ge et al. (2024) 在线性社会选择模型中证明了一个令人意外的负面结论——最优线性奖励模型可能违背 Pareto 最优性（PO）：即使所有原则都认为响应 A 优于 B，训练出的奖励函数仍可能给 B 更高分数。这与对齐目标直接矛盾，但其是否可通过更强的奖励函数类、更合理的数据分布来修复尚不清楚。

**核心矛盾**：现有分析固定在有限候选集上的最坏情况视角，而实际训练关键在于模型对新数据的泛化能力与数据分布的选择——两者都不在经典社会选择框架的讨论范围内。

**本文目标**：从三个方向探索 PO 违背的鲁棒性与修复路径：（1）扩大奖励函数类（多项式奖励）；（2）将公理推广到嵌入空间全局（泛化视角）；（3）通过数据设计恢复公理保证。

**核心 idea**：PO 违背的根本原因是损失优化中隐式的"范数约束"——某些方向的比较更多/更长，优化时倾向于优先满足这些方向；当数据均匀覆盖嵌入球面 $S^{d-1}$ 时，该偏置消失，PO 在极限中可被保证。

## 方法详解

### 整体框架

论文在 Ge et al. (2024) 的线性社会选择模型基础上构建理论分析。$n$ 个"原则"作为选民，各持线性效用方向 $v_i \in \mathbb{R}^d$，通过候选响应的嵌入向量上的成对二元比较产生偏好数据；基于损失的投票规则输出最小化总损失 $L(\theta)$ 的奖励函数 $r_\theta(x) = \langle \theta, x \rangle$。论文沿三条主线逐步推进：先证明多项式奖励依然违背 PO（负面结论），再证明均匀数据可恢复 PO（正面结论），并分析为什么 PMC 更难保证。

### 关键设计

**1. PO 违背的最简直觉——隐式范数约束**

Ge et al. 的反例复杂（6 个候选），本文给出最小化直觉：单选民 $v = (\varepsilon, 1)$，三候选 $a=(1,0), b=(0,0), c=(-\delta,\delta)$（$\delta \ll 1$）。在单位范数约束 $\|\theta\|=1$ 下，Bradley-Terry 损失 $\ell_{BT}(x) = \log(1 + e^x)$ 被 $(a,b)$ 和 $(a,c)$ 的比较项主导（因为这两对方向更"长"），导致最优 $\theta$ 接近 $x$ 轴，使 $\langle\theta, b\rangle > \langle\theta, c\rangle$，而选民也偏好 $b \succ c$，但 $\langle\theta, c\rangle$ 相较 $\langle\theta, b\rangle$ 被错误排序在该特定情形下引发违背。核心机制："不同方向的比较对损失的贡献量不同，长度/数量优势方向劫持了有限的范数'预算'。"

**2. 多项式奖励仍违背 PO——Theorem 4.1**

自然猜想：更丰富的奖励类（bounded-degree 多项式）可以绕过线性的限制。本文证明这不成立。构造：$m+1 = d(d+1)+2$ 个候选点，两个（加权）选民 $v_1=(1,0)$ 和 $v_2=(0,1)$，仅在 $c_0 \succ c_1$ 上全体同意（PO 要求点）。通过将候选分布在 $d$ 条斜率为 $-2$ 的直线 $L_j$ 上，使得退化实例（$c_0=c_1$）的唯一最优多项式恰好是违背 PO 的 $p^*(x,y) = -x-y$；再用 **Berge 极大定理**（上半连续性）证明对足够小的 $\delta > 0$，非退化实例的最优多项式仍满足 $p(c_1) > p(c_0)$，与所有选民的 PO 要求相悖。该定理对所有满足严格凸、下有界、$\ell'(0)>0$ 的损失函数成立，包括 Bradley-Terry 损失。

**3. 均匀数据恢复 PO——Theorem 5.1**

将分析从有限候选集推广到连续嵌入空间。定义"理想化"均匀数据设置：损失为对超球面 $S^{d-1}$ 上的积分
$$L(\theta) := \sum_{i=1}^n \int_{x \in S^{d-1},\, \langle v_i, x\rangle \geq 0} \ell(-\langle\theta, x\rangle)\, dx$$
即每个选民对其偏好方向半球上的所有单位向量都提供比较。在此设置下，对任意满足 PO 所需方向 $x$（即 $\langle v_i, x\rangle > 0, \forall i$），若 $\theta$ 满足 $\langle\theta, x\rangle \leq 0$，则可以构造 $\theta' = (1+\delta)\theta + \varepsilon x$ 使得 $\|\theta'\|_2 = \|\theta\|_2$ 且 $L(\theta') < L(\theta)$（借助每个 $L_i$ 单调随 $\langle\theta, v_i\rangle$ 递减的性质），从而 $\theta$ 不是最优解。结论：**至少两个不同选民时，任意最优 $\theta^*$ 均满足 PO**。数据均匀性消除了方向偏置，恢复了社会选择公理。

**4. PMC 在均匀数据下仍失败——Theorem 5.2**

配对多数一致性（PMC）要求：若严格多数选民偏好方向 $x$，奖励也须满足。在均匀数据设置下，当两个选民 $v_1, v_2$ 分别占 $p > 1/2$ 和 $1-p < 1/2$ 比例时，PMC 要求输出 $v_1$；而损失最小化却倾向于在 $v_1, v_2$ 之间插值——这是一种连续性/平滑性偏置，与 PMC 的离散跳变要求根本不兼容。PMC 在实践中本就值得质疑：细微多数（如 $51\%$）是否应完全忽略另一方？本文对此持开放态度。

## 实验关键数据

本文为纯理论工作，核心贡献为定理与证明，无数值实验。

### 主要定理汇总

| 定理 | 结论 | 条件 |
|------|------|------|
| Theorem 4.1 | 多项式奖励仍违背 PO 和 PMC | 严格凸损失，$\ell'(0)>0$，有限候选集 |
| Theorem 5.1 | 均匀数据下 PO 可被恢复 | 凸非递减损失，$\ell'(0)>0$，$\geq 2$ 选民 |
| Theorem 5.2 | 均匀数据下 PMC 仍失败 | 严格凸损失，$\ell'(0)>0$，$\geq 2$ 选民 |

### 关键发现

- PO 违背不随奖励函数类的增大而消失——根本原因在于数据分布，而非函数类的表达能力。
- 均匀数据是修复 PO 的充分条件；实践中可通过 PCA 分析嵌入差异的方向分布来评估距离均匀的程度。
- PMC 目前在有限和无限数据两种设置下均无法被损失规则保证，其对实际对齐的必要性值得重新审视——细微多数是否应压倒少数意见本身就是价值观问题。

- Bradley-Terry 损失的最优解等价于 Borda 排名（已知结论），与本文框架形成一致。

## 亮点与洞察

- 把 RLHF 奖励模型违背对齐公理的根因从"线性类太弱"精准定位到"数据方向分布不均"，分析角度新颖。
- Berge 极大定理的运用优雅地将退化实例（理想情况）的唯一性结论延拓至扰动实例，是整个负面定理的核心技巧。
- 给出了具备可证明保证的实践配方：通过数据集设计（平衡比较方向覆盖）而非改变训练 pipeline 来恢复 PO——对 Constitutional AI 的工程实现有直接指导意义。
- 将社会选择公理从"固定候选集"推广到"嵌入空间全局"，为 reward generalization 的理论分析建立了新基础。

## 局限与展望

- 嵌入空间线性表示的假设（Linear Representation Hypothesis）是简化假设，真实 LLM 的特征空间是否满足尚需验证。
- 均匀数据设置是理想化的连续极限，有限样本下收敛速度的显式采样复杂度界尚未建立。
- 目前只考虑固定嵌入（frozen embedding），端到端微调改变嵌入分布的情形未被涵盖。
- PMC 在当前框架下是否有替代放松版本值得进一步研究；基于重加权的数据-查询策略能否更有效地以定向方式恢复 PO 也是开放问题。

## 相关工作与启发

- **vs Ge et al. (2024)**：后者发现线性奖励违背 PO 并用组合社会选择规则修复，本文证明扩展到多项式奖励也无法修复，但通过数据设计而非替换 pipeline 实现了修复——两条路线互补。
- **vs Christiano et al. (2017) / RLHF**：RLHF 的 Bradley-Terry 最优解对应 Borda 排名（Anderson et al., 2009）；本文在此基础上揭示 Borda 规则在线性社会选择框架中的公理缺陷。
- **vs Constitutional AI (Bai et al., 2022b)**：本文直接为 Constitutional AI 的原则聚合提供理论依据——原则充当线性方向选民，均匀数据覆盖策略就是宪法风格对齐的可证明数据方案。
- **vs Procaccia et al. (2025)**：后者通过重加权 BT 损失实现 clone 鲁棒性，思路与本文数据重加权方向一致，但关注的公理不同。
- **vs Nash Learning from Human Feedback (Munos et al., 2024)**：NLHF 在社会选择中等价于 von Neumann winner / Fishburn 最大彩票规则，本文框架可进一步探索这类博弈论对齐方案是否满足 PO。
- **vs 多目标对齐 (Kim et al., 2025)**：比例对齐（proportional alignment）追求跨群体公平，与本文的 PO / PMC 是互补的公理视角，合并考虑可构建更完整的对齐公理体系。

## 评分

- 新颖性: ⭐⭐⭐⭐ 问题切入点新颖——将 PO 违背归因于数据方向分布而非函数类，并给出可操作的数据设计方案，显著超越了先验工作的纯负面结论。
- 实验充分度: ⭐⭐⭐ 纯理论工作，定理严谨完备，但缺乏实证验证（如在真实 LLM 嵌入上验证均匀数据假设的近似程度）。
- 写作质量: ⭐⭐⭐⭐ 结构清晰，负面→正面结论的递进叙事流畅；最小化直觉示例对理解 PO 违背机制帮助显著。
- 价值: ⭐⭐⭐⭐ 为宪法风格对齐的数据设计提供了理论基础，"平衡数据覆盖恢复 PO"的结论对工业界有直接参考价值。

> **适读人群**：对 RLHF 理论基础、社会选择与 AI 对齐交叉方向感兴趣的研究者；Constitutional AI 工程实践中关注数据质量保证的从业者。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Humanline: Online Alignment as Perceptual Loss](humanline_online_alignment_as_perceptual_loss.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [\[ICLR 2026\] Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)
- [\[ICML 2025\] Challenges and Future Directions of Data-Centric AI Alignment](../../ICML2025/llm_alignment/challenges_and_future_directions_of_data-centric_ai_alignment.md)
- [\[ICLR 2026\] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)

</div>

<!-- RELATED:END -->
