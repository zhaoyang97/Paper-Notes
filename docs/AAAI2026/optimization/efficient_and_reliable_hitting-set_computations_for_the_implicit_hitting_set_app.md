---
title: >-
  [论文解读] Efficient and Reliable Hitting-Set Computations for the Implicit Hitting Set Approach
description: >-
  [AAAI2026][优化/理论][隐式击中集] 针对隐式击中集框架中击中集组件依赖商用IP求解器带来的数值不稳定问题，提出基于伪布尔推理和随机局部搜索的替代方案及混合策略，实现了首个可认证的IHS计算并在1786个基准实例上展示了效率与可靠性的有效权衡。 领域现状：隐式击中集（IHS）方法是一种求解NP-hard组合优化问…
tags:
  - "AAAI2026"
  - "优化/理论"
  - "隐式击中集"
  - "伪布尔优化"
  - "可认证计算"
  - "随机局部搜索"
  - "IP求解器"
---

# Efficient and Reliable Hitting-Set Computations for the Implicit Hitting Set Approach

**会议**: AAAI2026  
**arXiv**: [2508.07015](https://arxiv.org/abs/2508.07015)  
**代码**: [https://bitbucket.org/coreo-group/pbhs](https://bitbucket.org/coreo-group/pbhs)  
**领域**: 组合优化 / 形式化验证  
**关键词**: [隐式击中集, 伪布尔优化, 可认证计算, 随机局部搜索, IP求解器]

## 一句话总结

针对隐式击中集框架中击中集组件依赖商用IP求解器带来的数值不稳定问题，提出基于伪布尔推理和随机局部搜索的替代方案及混合策略，实现了首个可认证的IHS计算并在1786个基准实例上展示了效率与可靠性的有效权衡。

## 研究背景与动机

**领域现状** 隐式击中集（IHS）方法是一种求解NP-hard组合优化问题的通用框架，已在MaxSAT、约束优化、伪布尔优化（PBO）等多个领域取得了成功的实际应用。IHS迭代交替调用决策预言机提取不一致性来源（核），以及优化器计算这些核上的击中集。

**现有痛点** 击中集优化器几乎标准地通过整数规划（IP）求解器实现。然而商用IP求解器默认使用浮点运算，可能因数值不稳定产生错误解。实验中确实发现Gurobi在4个实例上共9次错误声称最优性，精确SCIP也出现4次错误结果。

**核心矛盾** 效率与可靠性之间存在根本性权衡：浮点IP求解器最快但不可靠，精确IP求解器可靠但实例解决数减少超过15%且内存翻倍。更严重的是，现有IHS框架无法对计算结果提供独立可验证的正确性证书。

**本文目标** 如何在保持合理性能的同时，为IHS计算提供可认证的正确性保证。

**切入角度** 利用冲突驱动的伪布尔求解器的近期发展，探索基于PB推理的击中集计算替代方案，并与IP求解和随机局部搜索进行混合组合。

**核心 idea** 通过PB推理、SLS和IP求解的巧妙组合，实现首个可认证IHS计算框架，在效率损失可控的前提下获得可验证的正确性保证。

## 方法详解

### 整体框架

PBO-IHS算法在核集合 $\mathcal{K}$ 和目标函数 $O$ 之间迭代：每轮先调用 Solve-HS 在当前核上计算最优（或次优）解 $\gamma$，获取下界；再调用 Extract-Cores 对原始公式 $F$ 在 $\gamma$ 的假设下提取新核。当上界等于下界时终止。本文重点改进 Solve-HS 组件，提出统一的算法框架（Algorithm 2）允许灵活组合IP、PB优化和SLS三种方法。

### 关键设计

1. **多种PB优化算法实例化Solve-HS**:

    - 功能：用冲突驱动的伪布尔优化替代IP求解器来计算核集合上的最优解
    - 核心思路：提出三种PB优化变体——解改进搜索（SIS, 维护上界 $ub_c$，每轮求解 $\mathcal{K} \wedge (O < ub_c)$）、核引导搜索（CG, 维护重构目标 $O^R$ 并通过核驱动搜索）和核增强搜索（CB, CG+SIS混合）。为支持增量调用，SIS引入具体化约束 $r_{ub_c} \Rightarrow (\gamma(O) < ub_c)$，避免每次重建求解器
    - 设计动机：PB求解器基于精确算术推理，不存在浮点数值问题，且天然支持生成VeriPB格式的正确性证书

2. **混合策略与SLS集成**:

    - 功能：将随机局部搜索（SLS）和证明生成PB求解器与IP求解器组合，最小化对证明生成器的调用次数
    - 核心思路：提出三种混合策略——OptLB（仅在下界等于上界时调用PB求解器验证）、AllLB（每次下界更新都验证）和ForceLB（区分是否要求最优的调用）。SLS通过改造NuPBO实现增量调用，并引入多样化机制（两阶段搜索：修复前解+随机重启），以 $\text{argmax}_i \cos(z_t, z_i)$ 的方式保证解的多样性
    - 设计动机：非最优调用不需要证明，通过OptLB策略只在关键时刻调用开销较大的PB求解器，将认证开销降至最低

### 损失函数 / 训练策略

本文属于组合优化求解器设计，不涉及传统意义上的损失函数。核心优化目标为伪布尔优化问题 $(F, O)$：最小化 $O = \sum_i w_i \ell_i + lb$，约束为公式 $F$ 中的所有PB约束 $\sum_i a_i \ell_i \geq A$。证明生成基于VeriPB格式，使用切割平面操作和强化规则，支持引入具体化变量。

## 实验关键数据

### 主实验

| 配置 | 解决实例数 | 平均内存(GB) |
|------|-----------|-------------|
| Gurobi (浮点IP) | 951 | 0.66 |
| SCIP (浮点IP) | 913 | 1.16 |
| Exact SCIP (精确IP) | 762 | 2.10 |
| Core Guided (PB) | 687 | 0.63 |
| SIS + CB (PB) | 716 | 2.62 |
| OptLB混合 (Gurobi+SIS+CB) | 746 | 1.46 |

### 消融实验

| 配置 | 实例数变化 | 内存变化(GB) | 独占实例数 |
|------|----------|-------------|-----------|
| Gurobi + SLS | +17 | -0.02 | 31 |
| Exact SCIP + SLS | +44 | -0.48 | 58 |
| SIS+CB + SLS | +19 | -0.53 | 53 |
| OptLB + CG + SLS | +11 | -0.02 | 21 |

### 关键发现

- Gurobi在4个实例上9次错误声称最优性，Exact SCIP也有4次错误，证实了浮点数值不稳定的现实风险
- 证明日志的中位运行时开销仅10.7%，90百分位开销为100%
- SLS集成对Exact SCIP提升最大（+44实例），同时内存减少近20%
- OptLB混合策略是效率与可认证性的最佳平衡点，总共746实例中741个可生成证明

## 亮点与洞察

- 首次为IHS范式实现完整的可认证计算，填补了VeriPB在该领域的空白
- 巧妙的分层证明策略（OptLB）将认证开销降到最低，仅在确定最优解时才调用证明生成器
- 发现SLS与精确求解器的互补性特别强：大量实例只被其中一种方法解决
- 开发了通用PB证明日志API，两个求解器共享同一证明接口，预期可广泛复用

## 局限与展望

- PB求解器单独使用时仍明显落后于商用IP求解器（687-716 vs 951实例）
- VeriPB证明检查时间中位数是求解时间的588%，检查器效率是瓶颈
- 仅在PBO领域评估，尚未扩展到MaxSAT等其他IHS实例化场景
- SLS多样化策略仍存在局部最优陷阱，改进SLS启发式有进一步提升空间

## 相关工作与启发

本文建立在MaxSAT的IHS方法、VeriPB证明格式、NuPBO局部搜索等多条研究线的交汇处。对形式化验证社区的启发在于：通过混合策略可以在几乎不损失效率的情况下获得可认证性保证，这种"信任但验证"的范式值得在更多求解器设计中推广。对PBO竞赛中的认证赛道也有直接的推动意义。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次实现可认证IHS计算，混合策略设计思路新颖
- 实验充分度: ⭐⭐⭐⭐ 1786个基准实例，多种配置系统对比
- 写作质量: ⭐⭐⭐⭐ 技术细节清晰，算法描述规范
- 价值: ⭐⭐⭐⭐ 对可信AI和可验证计算领域有重要的工程和理论价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Set Smoothness Unlocks Clarke Hyper-stationarity in Bilevel Optimization](../../NeurIPS2025/optimization/set_smoothness_unlocks_clarke_hyper-stationarity_in_bilevel_optimization.md)
- [\[ICML 2026\] Diversity-Driven Offline Multi-Objective Optimization via Nested Pareto Set Learning](../../ICML2026/optimization/diversity-driven_offline_multi-objective_optimization_via_nested_pareto_set_lear.md)
- [\[AAAI 2026\] ECPv2: Fast, Efficient, and Scalable Global Optimization of Lipschitz Functions](ecpv2_fast_efficient_and_scalable_global_optimization_of_lipschitz_functions.md)
- [\[ICML 2026\] On the Interaction of Batch Noise, Adaptivity, and Compression, under $(L_0,L_1)$-Smoothness: An SDE Approach](../../ICML2026/optimization/on_the_interaction_of_batch_noise_adaptivity_and_compression_under_l_0l_1-smooth.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)

</div>

<!-- RELATED:END -->
