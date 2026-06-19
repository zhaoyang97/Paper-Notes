---
title: >-
  [论文解读] Constrained and Robust Policy Synthesis with Satisfiability-Modulo-Probabilistic-Model-Checking
description: >-
  [AAAI 2026][强化学习][马尔可夫决策过程] 本文提出首个能在任意结构约束下高效计算鲁棒策略的框架，通过将 SAT 求解器与概率模型检测算法紧密集成，实现对有限马尔可夫决策过程（MDP）的约束策略合成和鲁棒策略合成，在数百个 benchmark 上验证了可行性和竞争力。 领域现状：计算给定有限 MDP 的最优奖励策…
tags:
  - "AAAI 2026"
  - "强化学习"
  - "马尔可夫决策过程"
  - "鲁棒策略"
  - "约束满足"
  - "概率模型检测"
  - "SAT求解"
---

# Constrained and Robust Policy Synthesis with Satisfiability-Modulo-Probabilistic-Model-Checking

**会议**: AAAI 2026  
**arXiv**: [2511.08078](https://arxiv.org/abs/2511.08078)  
**代码**: 无  
**领域**: 强化学习 / 形式化验证  
**关键词**: 马尔可夫决策过程, 鲁棒策略, 约束满足, 概率模型检测, SAT求解

## 一句话总结

本文提出首个能在任意结构约束下高效计算鲁棒策略的框架，通过将 SAT 求解器与概率模型检测算法紧密集成，实现对有限马尔可夫决策过程（MDP）的约束策略合成和鲁棒策略合成，在数百个 benchmark 上验证了可行性和竞争力。

## 研究背景与动机

**领域现状**：计算给定有限 MDP 的最优奖励策略是规划、控制器合成和验证等应用的基础。标准方法（如值迭代、策略迭代）可以在多项式时间内找到最优策略。然而，实际应用中对策略有更多要求：需要策略在 MDP 参数扰动下仍然表现良好（鲁棒性），以及策略的表示形式或实现成本需满足额外的结构约束。

**现有痛点**：（1）鲁棒策略合成需要考虑 MDP 参数的不确定性集合，将单一优化问题扩展为对整个不确定性集合的 min-max 问题，计算复杂度显著增加；（2）结构约束（如策略的内存大小限制、确定性要求、状态聚合约束等）使得问题从连续优化变为组合优化；（3）现有方法通常只能处理特定类型的约束或鲁棒性需求，缺乏统一框架。

**核心矛盾**：灵活性与效率之间的矛盾——允许表达任意约束的框架通常计算效率低下，而高效方法通常只适用于特定问题片段。

**本文目标**：设计一个同时具备灵活性（支持一阶理论中的任意结构约束）和效率（紧密集成 SAT 求解与概率模型检测）的策略合成框架。

**切入角度**：作者借鉴 SMT（Satisfiability Modulo Theories）的思想，将约束策略合成问题分解为：SAT 求解器处理组合约束部分，概率模型检测算法处理 MDP 分析部分，两者通过冲突驱动学习紧密交互。

**核心 idea**：构建 Satisfiability-Modulo-Probabilistic-Model-Checking（SM-PMC）框架，将 SAT 求解器的组合搜索能力与概率模型检测的 MDP 分析能力融合，实现灵活且高效的约束/鲁棒策略合成。

## 方法详解

### 整体框架

SM-PMC 框架的输入是一个或一组 MDP、优化目标（如最大化期望奖励）以及一阶理论表达的结构约束。框架通过 SAT 求解器生成候选策略（满足结构约束），通过概率模型检测器评估候选策略在 MDP 上的性能，然后利用评估结果生成冲突子句反馈给 SAT 求解器，迭代优化直至找到最优可行策略或证明不可行。

### 关键设计

1. **一阶约束表达（First-Order Theory Encoding）**:

    - 功能：以统一的形式化语言表达各种结构约束。
    - 核心思路：将 MDP 的状态-动作空间编码为命题变量，策略的结构约束（如确定性、有限内存、成本预算、状态聚合等）编码为一阶逻辑公式。这使得用户可以灵活地指定任意类型的约束，而不需要为每种约束设计专门的算法。
    - 设计动机：现有方法（如 MILP、凸优化）只能处理特定形式的约束。一阶理论具有强大的表达力，可以统一处理各种约束类型。

2. **SAT-PMC 紧密集成**:

    - 功能：在组合搜索和概率分析之间实现高效的信息传递。
    - 核心思路：SAT 求解器负责在满足约束的策略空间中搜索，生成候选策略；PMC（Probabilistic Model Checker）负责在给定策略下分析 MDP 的概率性质（如期望奖励、到达概率等）。当 PMC 发现当前候选策略不满足性能要求时，会生成一个解释性冲突子句（lemma），为 SAT 求解器提供剪枝信息，避免搜索相似的无效区域。
    - 设计动机：与朴素的枚举+验证方法相比，紧密集成通过冲突驱动学习实现了指数级的搜索空间缩减，是效率的关键来源。

3. **鲁棒策略合成扩展**:

    - 功能：在 MDP 参数不确定性下找到性能有保证的策略。
    - 核心思路：将鲁棒性需求编码为对一组 MDP（不确定性集合）的约束——策略需要在集合中所有 MDP 上都满足性能要求。框架将多个 MDP 的 PMC 查询整合到同一个 SAT 搜索过程中，通过共享冲突信息加速求解。
    - 设计动机：鲁棒策略合成的朴素方法是对每个可能的 MDP 分别求解再取交集，计算成本极高。SM-PMC 的集成方式使得不同 MDP 的分析结果可以互相提供剪枝信息。

### 损失函数 / 训练策略

不涉及神经网络训练。目标函数为 MDP 的期望奖励或到达概率最大化，约束条件通过一阶理论表达。优化通过 SAT 求解器的分支-约束框架迭代求解。

## 实验关键数据

### 主实验

在数百个 benchmark 上进行评估，覆盖约束策略合成和鲁棒策略合成两大类问题。

| 任务类型 | 指标 | SM-PMC | 对比方法 | 说明 |
|----------|------|--------|----------|------|
| 约束策略合成 | 求解时间 | 竞争力强 | MILP等 | 在通用约束上更灵活 |
| 约束策略合成 | 求解率 | 高 | 专用方法 | 对任意约束类型均可处理 |
| 鲁棒策略合成 | 求解时间 | 竞争力强 | SOTA方法 | 与特定片段的最优方法相当 |
| 鲁棒策略合成 | 策略质量 | 最优/近优 | 启发式方法 | 提供精确解而非近似解 |

### 消融实验

| 配置 | 性能 | 说明 |
|------|------|------|
| 完整SM-PMC | 最快 | SAT+PMC紧密集成 |
| 无冲突学习 | 慢数量级 | 缺少PMC反馈的剪枝 |
| 枚举+验证 | 不可扩展 | 朴素方法在大状态空间上失败 |
| 仅MILP | 部分约束 | 只能处理线性约束，灵活性不足 |

### 关键发现

- 冲突驱动学习是效率的关键——从 PMC 到 SAT 的冲突反馈使搜索空间缩减了数量级。
- 框架对约束类型高度灵活，可以处理之前需要专用算法的各种约束。
- 在鲁棒策略合成上，与专门为此设计的 SOTA 方法具有竞争力，同时还支持额外的结构约束。
- 框架的可扩展性主要受限于底层 MDP 的状态空间大小，对中等规模 MDP 效果良好。

## 亮点与洞察

- **统一框架的灵活性**是本文最大优势。将多种策略合成变体（约束、鲁棒、约束+鲁棒）统一到一个框架中，避免了为每种变体设计专门算法的工程负担。
- **SAT 与 PMC 的融合**是跨领域方法论创新——将形式化方法（SAT/SMT）社区的技术引入规划和强化学习领域，打破了学科壁垒。
- 实际应用场景丰裕——控制器对安全性和鲁棒性有严格要求的机器人系统、自动驾驶规划等领域都可以直接受益。

## 局限与展望

- 框架针对有限状态 MDP，不直接适用于连续状态空间或大规模 MDP。
- 可扩展性受限于 SAT 求解器和 PMC 的固有复杂度，对非常大的状态空间可能不实用。
- 未与深度强化学习的鲁棒方法进行对比，两类方法适用的问题规模不同。
- 可以探索与神经网络策略参数化的结合，利用 SM-PMC 进行验证而非合成。

## 相关工作与启发

- **vs MILP 方法**: MILP 方法高效但只能处理线性约束，SM-PMC 支持任意一阶约束但可能在线性约束子问题上不如 MILP 快。
- **vs Robust MDP 方法**: 传统 Robust MDP 方法假设矩形不确定性集合，SM-PMC 可处理更一般的不确定性描述。
- **vs 深度RL鲁棒方法**: 深度 RL 方法适用于大规模连续问题但只能提供近似解，SM-PMC 适用于中等规模离散问题但提供精确解。

## 评分

- 新颖性: ⭐⭐⭐⭐ SAT与PMC的融合框架是创新的方法论贡献
- 实验充分度: ⭐⭐⭐⭐ 数百个benchmark覆盖多种问题变体
- 写作质量: ⭐⭐⭐⭐ 形式化表述严谨，问题定义清晰
- 价值: ⭐⭐⭐⭐ 为安全关键系统的策略合成提供了实用工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Boolean Satisfiability via Imitation Learning](../../ICLR2026/reinforcement_learning/boolean_satisfiability_via_imitation_learning.md)
- [\[ICML 2026\] Beyond Scalar Rewards: Dense Feedback for LLM Policy Synthesis in Sequential Social Dilemmas](../../ICML2026/reinforcement_learning/beyond_scalar_rewards_dense_feedback_for_llm_policy_synthesis_in_sequential_soci.md)
- [\[ICLR 2026\] ROMI: Model-based Offline RL via Robust Value-Aware Model Learning with Implicitly Differentiable Adaptive Weighting](../../ICLR2026/reinforcement_learning/model-based_offline_rl_via_robust_value-aware_model_learning_with_implicitly_dif.md)
- [\[ACL 2026\] Bridging SFT and RL: Dynamic Policy Optimization for Robust Reasoning](../../ACL2026/reinforcement_learning/bridging_sft_and_rl_dynamic_policy_optimization_for_robust_reasoning.md)
- [\[AAAI 2026\] G-UBS: Towards Robust Understanding of Implicit Feedback via Group-Aware User Behavior Simulation](g-ubs_towards_robust_understanding_of_implicit_feedback_via_group-aware_user_beh.md)

</div>

<!-- RELATED:END -->
