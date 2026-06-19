---
title: >-
  [论文解读] Multiple-Policy Evaluation via Density Estimation
description: >-
  [ICML 2025][multiple-policy evaluation] 提出 CAESAR 算法，通过两阶段方法（粗估计访问分布 + 最优采样分布下的密度比估计）同时评估 K 个策略，实现非渐近、实例依赖的样本复杂度，核心技术是"粗估计"——仅需 $O(1/\epsilon)$ 样本即可获得常数倍精度的分布近似。
tags:
  - "ICML 2025"
  - "multiple-policy evaluation"
  - "density estimation"
  - "importance sampling"
  - "coarse estimation"
  - "sample complexity"
---

# Multiple-Policy Evaluation via Density Estimation

**会议**: ICML 2025  
**arXiv**: [2404.00195](https://arxiv.org/abs/2404.00195)  
**代码**: 无  
**领域**: 其他  
**关键词**: multiple-policy evaluation, density estimation, importance sampling, coarse estimation, sample complexity

## 一句话总结

提出 CAESAR 算法，通过两阶段方法（粗估计访问分布 + 最优采样分布下的密度比估计）同时评估 K 个策略，实现非渐近、实例依赖的样本复杂度，核心技术是"粗估计"——仅需 $O(1/\epsilon)$ 样本即可获得常数倍精度的分布近似。

## 研究背景与动机

**领域现状**：策略评估是强化学习的基础问题，在策略迭代、策略梯度等方法中作为核心子步骤。在实际应用中，常需同时评估用不同超参训练的 K 个策略以选择最优策略。单策略离线评估已有丰富方法，包括重要性采样、模型法和双重鲁棒估计器。

**现有痛点**：最朴素的做法是独立运行 K 次单策略评估，但样本复杂度线性于 K，完全忽略了策略间的相似性。Dann et al. (2023) 提出在线方法利用轨迹合成 (trajectory synthesis) 来复用策略重叠部分的样本，但需要构建生成模型作为中间步骤，实际中可能不可行。

**核心矛盾**：多策略评估的挑战在于如何设计一个共享的数据收集策略，使得一份数据集能高效支撑所有 K 个策略的评估。这要求采样分布同时覆盖所有目标策略的访问分布——而不是为每个策略单独采样。

**本文目标** 设计最优的数据收集（行为）分布，并用离线评估技术从共享数据中同时估计所有 K 个策略的值。

**切入角度**：作者观察到，如果能以常数倍精度（即"粗估计"）近似目标策略的访问分布，就足以计算出接近最优的采样分布。而粗估计只需 $O(1/\epsilon)$ 样本——远低于精确估计的 $O(1/\epsilon^2)$。

**核心 idea**：用低成本的粗估计指导最优采样，再在最优采样下用 DualDICE 风格的密度比估计完成精确评估。

## 方法详解

### 整体框架

CAESAR (Coarse and Adaptive EStimation with Approximate Reweighing) 分为两个阶段。输入是 K 个目标策略 $\pi_1, \ldots, \pi_K$ 和一个表格 MDP 环境。Phase I 用少量样本获取访问分布的粗估计；Phase II 基于粗估计构建最优采样分布，从中采集共享数据集，再用 IDES 算法估计所有策略的密度比和期望回报。

### 关键设计

1. **粗估计 (Coarse Estimation)**:

    - 功能：以常数倍精度近似每个策略的访问分布 $d_h^{\pi_k}(s,a)$
    - 核心思路：对每个目标策略收集 $O(1/\epsilon)$ 条轨迹，利用简单的浓度不等式得到乘法精度的分布估计。具体地，粗估计 $\hat{d}$ 满足 $c_1 d(s,a) \leq \hat{d}(s,a) \leq c_2 d(s,a)$，其中 $c_1, c_2$ 是常数。关键在于乘法精度只需线性（而非二次）样本量
    - 设计动机：Phase I 的样本成本是低阶项，可视为"免费"的预处理。粗估计的灵活性在于常数乘子可以是任意值，使得公式更简洁优雅

2. **最优采样分布构建**:

    - 功能：根据粗估计计算一个混合行为分布 $\mu^*$，使得在所有 K 个策略上的最大密度比最小化
    - 核心思路：最优采样问题可形式化为 $\min_\mu \max_{k \in [K]} \sum_{s,a} (d_h^{\pi_k}(s,a))^2 / \mu_h(s,a)$。由于粗估计的乘法精度，用 $\hat{d}$ 替代 $d$ 后得到的近似最优解与真实最优解相差有界（不超过常数倍）
    - 设计动机：最优采样分布有效利用策略间的访问重叠——当多个策略在某些状态-动作对上有高访问频率时，采样分布自然集中在这些区域

3. **IDES (Importance Density EStimation)**:

    - 功能：从最优采样分布的数据中估计每个策略的边际重要性密度比 $w_h^{\pi_k}(s,a) = d_h^{\pi_k}(s,a) / \mu_h(s,a)$
    - 核心思路：受 DualDICE 启发，设计逐步 (step-wise) 二次损失函数。对每个时间步 $h$，通过最小化 $L_h(w) = \mathbb{E}_\mu[(w_h(s,a) - T_h w_{h+1}(s,a))^2]$ 来估计密度比，其中 $T_h$ 是与转移核相关的算子。使用随机梯度下降求解，配合 Median of Means 估计器将期望结果转化为高概率结果
    - 设计动机：直接估计边际密度比避免了标准重要性采样中指数级别的方差（与视野 H 成指数关系）。相比 DualDICE 的原始版本（针对无限视野），IDES 非平凡地扩展到有限视野 MDP

### 损失函数

IDES 的核心损失为逐步二次目标：$L_h(w) = \mathbb{E}_\mu[(w_h(s,a) - r_h(s,a) \cdot w_h(s,a) - \gamma \sum_{s'} P_h(s'|s,a) \sum_{a'} \pi(a'|s') w_{h+1}(s',a'))^2]$。该损失在每个时间步 $h$ 独立优化，配合强凸性和光滑性保证 SGD 的收敛速率。

## 实验关键数据

### 样本复杂度理论对比

| 方法 | 样本复杂度 | 利用策略相似性 | 非渐近 | 已知上界 |
|------|----------|-------------|--------|---------|
| 朴素 K 次评估 | $O(K H^3 / \epsilon^2)$ | 否 | 是 | 是 |
| Dann et al. (2023) | 依赖轨迹合成 | 是（在线） | 是 | 否 |
| Yin & Wang (2020) | $O(H^3/\epsilon^2 \cdot \text{覆盖项})$ | 否 | 否（渐近） | 否 |
| **CAESAR** | $O(H^4/\epsilon^2 \cdot \sum_h \max_k \sum_{s,a} d^2/\mu^*)$ | **是（离线）** | **是** | **是** |

### 各组件贡献分析

| 组件 | 样本复杂度贡献 | 说明 |
|------|-------------|------|
| Phase I 粗估计 | $O(1/\epsilon)$（低阶） | 线性而非二次，可忽略 |
| 最优分布计算 | 基于粗估计，无额外采样 | 使用凸优化求解 |
| Phase II 精确评估 | $O(H^4/\epsilon^2 \cdot \text{实例项})$（主项） | 占总成本绝大部分 |
| MARCH（确定性策略） | 多项式于 $S, A$ | 尽管确定性策略数指数级 |

### 关键发现

1. 粗估计成本 $O(1/\epsilon)$ 是精确估计 $O(1/\epsilon^2)$ 的低阶项——使两阶段方法的前期成本可忽略
2. 最优采样分布有效利用策略间的访问重叠，当 K 个策略高度相似时样本复杂度接近单策略评估
3. CAESAR 的已知上界优势意味着可以在运行前预估所需样本量——而 Yin & Wang 的结果只能事后验证
4. β-距离可用于 MARCH 算法，对指数级数量的确定性策略实现多项式复杂度的粗估计

## 亮点与洞察

1. **粗估计概念简洁但威力巨大**：乘法精度的近似只需线性样本——这一技术完全通用，可在其他需要两阶段估计的问题中复用
2. **探索-利用分离设计优雅**：Phase I（探索获取粗信息）和 Phase II（利用精确评估）职责清晰
3. **IDES 的独立贡献价值**：作为 DualDICE 到有限视野 MDP 的严格扩展，可脱离多策略评估独立使用
4. **已知上界的实际意义**：不依赖未知真实分布的上界使得算法可以自信地预测样本需求

## 局限性

- 仅适用于表格 MDP，函数逼近场景的扩展是重要开放问题
- $H^4$ 的视野依赖仍然较大，与下界之间是否有 gap 未知
- 理论导向的工作，缺乏大规模实证实验验证算法的实际表现
- 计算最优采样分布需要已知 MDP 结构，限制了对未知环境的直接应用

## 相关工作与启发

- DualDICE (Nachum et al., 2019) 的边际密度比估计思想被扩展到有限视野多策略场景
- 与 Amortila et al. (2024) 的探索目标有联系，但粗估计使得优化更简单
- 粗估计技术有广泛适用性——任何需要先粗略探索再精确利用的两阶段问题都可借鉴
- Zhang & Zanette (2023) 和 Li et al. (2023) 也使用了乘法精度估计，但本文给出了更干净、更通用的理论框架

## 评分

⭐⭐⭐⭐ 理论工作扎实，粗估计技术和 IDES 算法有独立贡献价值。局限在于表格 MDP 假设和缺乏实证。整体是离线多策略评估理论的重要推进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Minimum Variance Path Principle for Accurate and Stable Score-Based Density Ratio Estimation](../../ICLR2026/learning_theory/a_minimum_variance_path_principle_for_accurate_and_stable_score-based_density_ra.md)
- [\[ICML 2025\] On Fine-Grained Distinct Element Estimation](on_fine-grained_distinct_element_estimation.md)
- [\[ICML 2025\] Learning-Augmented Algorithms for MTS with Bandit Access to Multiple Predictors](learning-augmented_algorithms_for_mts_with_bandit_access_to_multiple_predictors.md)
- [\[ICML 2026\] Cutting LLM Evaluation Costs with SySRs: A Bandit Algorithm that Provably Exploits Model Similarity](../../ICML2026/learning_theory/cutting_llm_evaluation_costs_with_sysrs_a_bandit_algorithm_that_provably_exploit.md)
- [\[ICML 2026\] Provably Data-driven Multiple Hyper-parameter Tuning with Structured Loss Function](../../ICML2026/learning_theory/provably_data-driven_multiple_hyper-parameter_tuning_with_structured_loss_functi.md)

</div>

<!-- RELATED:END -->
