---
title: >-
  [论文解读] Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities
description: >-
  [NeurIPS 2025][LLM评测][automated theorem proving] 提出 Ineq-Comp 基准，通过对简单不等式种子问题施加人类直觉可轻松处理的组合变换（变量复制、代数重写），揭示当前 LLM 形式化定理证明器在组合推理上的根本性缺陷——即使 DeepSeek-Prover-V2-7B 也有 20%+ 的性能下降。
tags:
  - "NeurIPS 2025"
  - "LLM评测"
  - "automated theorem proving"
  - "compositional reasoning"
  - "formal verification"
  - "Lean 4"
  - "mathematical inequalities"
  - "benchmark"
---

# Ineq-Comp: Benchmarking Human-Intuitive Compositional Reasoning in Automated Theorem Proving on Inequalities

**会议**: NeurIPS 2025  
**arXiv**: [2505.12680](https://arxiv.org/abs/2505.12680)  
**代码**: [GitHub](https://github.com/haoyuzhao123/LeanIneqComp)  
**领域**: LLM评测  
**关键词**: automated theorem proving, compositional reasoning, formal verification, Lean 4, mathematical inequalities, benchmark

## 一句话总结

提出 Ineq-Comp 基准，通过对简单不等式种子问题施加人类直觉可轻松处理的组合变换（变量复制、代数重写），揭示当前 LLM 形式化定理证明器在组合推理上的根本性缺陷——即使 DeepSeek-Prover-V2-7B 也有 20%+ 的性能下降。

## 研究背景与动机

近年来基于 LLM 的形式化定理证明取得了巨大进展：AlphaProof、DeepSeek-Prover-V2 等系统在 MiniF2F 等基准上不断刷新记录。然而，现有基准存在以下问题：

**MiniF2F** 规模小且难度跨度极大，从入门到 IMO 级别混杂

**ProofNet / PutnamBench** 聚焦高难度问题，当前开源模型仅能解决少数
3. 这些基准都**未显式测试组合推理能力**——即将已知的简单推理步骤组合形成更复杂论证的能力
4. 存在**数据污染**风险

作者的核心洞察：数学推理的本质是"站在巨人肩膀上"——复杂论证由简单且已理解的步骤链接而成。如果一个证明器能证明种子问题，那么对该问题进行简单的组合变换后，人类几乎不需要额外推理即可解决的变体，证明器也应该能够处理。

## 方法详解

### 整体框架

Ineq-Comp 基准包含三个子集：

- **Ineq-Simp**：150 个问题，由 75 个种子问题通过两类变换生成
- **Ineq-Mix**：基于规则的自动生成框架，可无限扩展
- **Ineq-Real**：50 个来自真实竞赛的不等式问题

### 关键设计

**种子问题（75个）**：

| 技术类别 | 数量 | 说明 |
|---------|------|------|
| AM-GM 不等式 | 25 | 算术-几何均值不等式 |
| Cauchy-Schwarz | 25 | 柯西-施瓦茨不等式 |
| 其他（Jensen/Schur/SOS/归纳）| 25 | Jensen(10) + Schur(5) + SOS(5) + 归纳(5) |

每个种子问题均附带经验证的 Lean 4 证明，由具有 IMO/CMO 经验的人员编写和审核。

**Type I 变换（变量复制 + 组合）**：

给定种子不等式 $f(X) \geq g(X)$，其中 $X = (x_1, \ldots, x_n)$，条件 $X \in \mathcal{C}$：

- 用新变量 $Y = (y_1, \ldots, y_n)$ 复制同一不等式
- 若 $g(X) \geq 0$：乘法组合 $f(X) \cdot f(Y) \geq g(X) \cdot g(Y)$
- 若 $g(X)$ 不保证非负：加法组合 $f(X) + f(Y) \geq g(X) + g(Y)$

这类变换对人类而言极其简单——只需将问题分解为两个相同部分。

**Type II 变换（代数变换）**：

对种子问题的变量施加变换 $T: \mathbb{R}^n \to \mathbb{R}^n$（如平方、开方），证明在条件 $T(X) \in \mathcal{C}$ 下 $f(T(X)) \geq g(T(X))$ 成立。

**Ineq-Mix 自动生成框架**：

三类规则可自由组合：
1. **组合变换**：通过加法/乘法/max/min 组合不等式
2. **变量级变换**：用代数表达式替换变量
3. **问题级变换**：对两端施加单调函数（如 $\exp$, $\log$）

### 损失函数

本文是基准测试论文，不涉及训练损失函数。评估指标使用标准的 pass@N 准确率：问题被视为"已解决"当且仅当 N 次生成中至少一次产生了正确证明。

## 实验关键数据

### 主实验

在 Ineq-Simp 上的 pass@3200 准确率（%）：

| 模型 | AM-GM Seed | AM-GM Type I | AM-GM Type II | Cauchy Seed | Cauchy Type I | Cauchy Type II |
|------|-----------|-------------|--------------|------------|--------------|---------------|
| DeepSeek-Prover-V2-7B | **96.0** | **84.0** | **76.0** | **76.0** | **52.0** | **48.0** |
| Kimina-7B | 80.0 | 44.0 | 64.0 | 68.0 | 52.0 | 36.0 |
| STP | 64.0 | 44.0 | 40.0 | 44.0 | 24.0 | 28.0 |
| Goedel-Prover-SFT | 60.0 | 4.0 | 24.0 | 40.0 | 32.0 | 28.0 |
| DeepSeek-Prover-V1.5-RL | 60.0 | 4.0 | 24.0 | 24.0 | 12.0 | 12.0 |

**关键发现**：除 DeepSeek-Prover-V2-7B 外，所有模型从种子到 Type I 的性能下降极为严重——Goedel-Prover 从 60% 暴跌至 4%。

**模型规模扩展实验**（pass@32）：

| 模型 | Seed | Type I | Type II |
|------|------|--------|---------|
| DeepSeek-Prover-V2-7B | 66.2 | 47.0 | 42.1 |
| DeepSeek-Prover-V2-671B | 88.0 | 68.3 | 70.7 |
| Goedel-Prover-V2-8B | 69.0 | 35.0 | 49.0 |
| Goedel-Prover-V2-32B | 90.0 | 68.0 | 83.0 |

虽然扩大模型规模提升了绝对性能，但种子与变体之间的差距依然存在。

### 消融实验

**In-Context Learning（提供种子证明）**：

| 模型 | AM-GM Type I | AM-GM Type II | Cauchy Type I | Cauchy Type II |
|------|-------------|--------------|--------------|---------------|
| Qwen2.5-Coder-32B (128) | 28.0 | 40.0 | 48.0 | 20.0 |
| DeepSeek-R1-32B w/ thinking (128) | 16.0 | 44.0 | 84.0 | 40.0 |
| GPT-4o (16) | 12.0 | 40.0 | 56.0 | 16.0 |
| Claude 3.7 Sonnet (16) | 36.0 | 28.0 | 32.0 | 20.0 |

即使直接提供种子问题的完整证明，模型仍难以泛化到变体——**组合推理的缺陷根源不在于缺乏知识，而在于缺乏组合泛化能力**。

**微调实验**：

在 AM-GM 种子问题上生成合成数据（约 8k 可编译样本）微调 Goedel-Prover-SFT：
- **In-distribution (AM-GM Type I)**：准确率显著提升至 56%
- **Out-of-distribution**：性能几乎无提升，接近微调前水平
- MiniF2F 上的性能未退化，排除了灾难性遗忘的可能

### 关键发现

1. **战术单一性**：几乎所有模型严重依赖 `nlinarith` + `sq_nonneg`（平方和方法），很少尝试直接应用 AM-GM、Cauchy 等经典不等式
2. **自然语言 vs 形式化的鸿沟**：模型在自然语言注释中经常正确提到 AM-GM 等策略，但在 Lean 代码中放弃高级策略，转而使用暴力代数操作
3. **O3 模型对比**：OpenAI O3 在自然语言模式下能轻松分解并求解这些组合问题，证明瓶颈在于形式化推理系统而非数学内容本身
4. **Ineq-Mix 比 Ineq-Real 更难**：自动生成的组合问题对模型来说比真实世界竞赛题更具挑战性，这一反直觉结果突显了组合推理的核心困难

## 亮点与洞察

1. **benchmark 设计精妙**：自底向上的构造方式使得难度可控、可扩展，且能精准定位组合推理这一特定能力的缺陷
2. **揭示了一个深层问题**：当前形式化证明器不是"不够聪明"，而是缺乏将已有知识组合应用的能力——这与人类数学推理的核心能力形成鲜明对比
3. **O3 + 形式化的差距**：自然语言推理能轻松处理的组合问题，在形式化设定中却成为主要瓶颈，暗示形式化训练数据中缺乏组合推理模式
4. **微调无法弥补**：这不是简单的数据不足问题，适度规模的 SFT 只能学习特定模式而无法泛化

## 局限性

1. 仅聚焦于不等式领域，组合推理在代数、几何、数论等其他数学领域的表现尚未探索
2. 种子问题难度限于入门竞赛级别，对更高难度问题的适用性未验证
3. 由于代码不公开，未评估 HunyuanProver、BFS-Prover 等更新的树搜索方法
4. Ineq-Mix 的随机生成可能引入一些形式上有效但数学意义不大的问题

## 相关工作与启发

- **MiniF2F / ProofNet / PutnamBench**：现有基准侧重整体难度而非特定推理能力的诊断
- **AlphaProof / DeepSeek-Prover-V2**：最强的形式化证明系统，但在组合推理上仍有明显弱点
- **AIPS / 不等式求解器**：面向更高难度，而本文从简单问题出发暴露了不同维度的失败
- **启发**：未来证明器可能需要显式的"分解-组合"推理模块，或者在 RL 训练中引入组合推理的课程学习

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 从组合推理角度审视形式化定理证明是全新视角
- **技术深度**: ⭐⭐⭐⭐ — 基准设计严谨，变换定义清晰，消融实验全面
- **实验充分性**: ⭐⭐⭐⭐⭐ — 涵盖 6+ 种证明器、多种预算、ICL/微调/规模扩展等多角度验证
- **实用价值**: ⭐⭐⭐⭐ — 为形式化定理证明社区提供了可持续扩展的诊断工具
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表直观，动机阐述有力
- **综合评分**: ⭐⭐⭐⭐ (8/10)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Are LLM Belief Updates Consistent with Bayes' Theorem?](../../ICML2025/llm_evaluation/are_llm_belief_updates_consistent_with_bayes_theorem.md)
- [\[ACL 2025\] PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations](../../ACL2025/llm_evaluation/patch_psychometrics-assisted_benchmarking_of_large_language_models_against_human.md)
- [\[ICCV 2025\] HiERO: Understanding the Hierarchy of Human Behavior Enhances Reasoning on Egocentric Videos](../../ICCV2025/llm_evaluation/hiero_understanding_the_hierarchy_of_human_behavior_enhances_reasoning_on_egocen.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[ACL 2025\] FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging](../../ACL2025/llm_evaluation/financereasoning_benchmarking_financial_numerical_reasoning_more.md)

</div>

<!-- RELATED:END -->
