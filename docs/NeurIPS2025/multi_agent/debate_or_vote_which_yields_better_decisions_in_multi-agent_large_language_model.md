---
title: >-
  [论文解读] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?
description: >-
  [NeurIPS 2025 Spotlight][多智能体][多智能体辩论] 通过理论和实验证明，多智能体辩论（MAD）的性能提升主要来自多数投票（ensembling）而非辩论本身——辩论过程构成 martingale（期望不变），即辩论不系统性地提升正确率，并基于此理论提出通过偏向正确信号来改进 MAD。
tags:
  - "NeurIPS 2025 Spotlight"
  - "多智能体"
  - "多智能体辩论"
  - "多数投票"
  - "Martingale"
  - "贝叶斯信念更新"
  - "LLM集成"
---

# Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2508.17536](https://arxiv.org/abs/2508.17536)  
**代码**: [https://github.com/deeplearning-wisc/debate-or-vote](https://github.com/deeplearning-wisc/debate-or-vote)  
**领域**: LLM Agent  
**关键词**: 多智能体辩论, 多数投票, Martingale, 贝叶斯信念更新, LLM集成

## 一句话总结
通过理论和实验证明，多智能体辩论（MAD）的性能提升主要来自多数投票（ensembling）而非辩论本身——辩论过程构成 martingale（期望不变），即辩论不系统性地提升正确率，并基于此理论提出通过偏向正确信号来改进 MAD。

## 研究背景与动机

**领域现状**：Multi-Agent Debate (MAD) 已成为提升 LLM 推理的流行范式，多个 LLM 通过结构化讨论协作解题，近年出现了各种变体（去中心化、稀疏、中心化等拓扑结构、角色分配等）。

**现有痛点**：MAD 变得越来越复杂，但其有效性的根源始终不清楚——到底是"多个 agent 的集成效应"还是"agent 间交互讨论"带来了性能提升？

**核心矛盾**：如果大部分性能提升来自简单集成（majority voting），那 MAD 中额外的辩论、通信、架构设计等复杂性就不值得。

**本文目标** 严格分离 MAD 中"集成"和"辩论"的贡献，从理论上解释为什么辩论本身不能系统性提升性能。

**切入角度**：将 MAD 形式化为贝叶斯信念更新过程，用 Dirichlet-Compound-Multinomial (DCM) 模型刻画 agent 行为，证明辩论构成 martingale。

**核心 idea**：辩论在期望意义上不改善正确率（martingale），大部分性能来自多数投票的集成效应。

## 方法详解

### 整体框架
将 MAD 分解为两个组件：(1) Multi-Agent（多 agent 集成 → majority voting）和 (2) Debate（agent 间迭代交流 → 信念更新）。通过比较"只投票不辩论" vs "辩论后投票"来隔离各自贡献。

### 关键设计

1. **DCM 生成模型**:

    - 功能：形式化 agent 的回答生成过程
    - 核心思路：每个 agent 在第 $t$ 轮持有信念向量 $\boldsymbol{\alpha}_{i,t}$（Dirichlet 参数），先采样 $\boldsymbol{\theta}_{i,t} \sim \text{Dirichlet}(\boldsymbol{\alpha}_{i,t})$，再从 $\text{Categorical}(\boldsymbol{\theta}_{i,t})$ 生成回答。边际概率 $P(y_{i,t}=k) = \alpha_{i,t}^{(k)} / \sum_j \alpha_{i,t}^{(j)}$
    - 设计动机：DCM 自然捕捉了 LLM 的内部不确定性（Dirichlet prior）和输出随机性（采样），且具有贝叶斯共轭性便于分析

2. **Majority Voting 成功概率定理 (Theorem 1)**:

    - 核心结论：即使正确答案仅略胜于其他选项（$\theta_1 \ll 1/2$），随着 agent 数 $N$ 增加，投票正确概率趋近 1
    - 下界：$\mathbb{P}(y_{mv}=1) \geq 1 - \exp(-N(\Delta/\sqrt{K} - 1/\sqrt{N})^2)$
    - 关键洞察：投票本身就有"放大效应"，不需要辩论

3. **Martingale 定理 (Theorem 2 - 核心贡献)**:

    - 功能：证明辩论过程中 agent 对正确答案的信念构成 martingale
    - 核心结论：$\mathbb{E}[p_{i,t} | \boldsymbol{\alpha}_{t-1}] = p_{i,t-1}$，即辩论不改变对正确答案的期望信念
    - 前提条件：邻居平均信念等于自身信念（同质 agent + 全连接时自然满足）
    - 深刻含义：辩论的每一轮都是随机游走——有时 agent 被纠正（好），有时被误导（坏），**期望上互相抵消**。这解释了为什么 MAD 往往不显著优于 majority voting

4. **MAD-oracle 改进策略**:

    - 功能：打破 martingale 的零均值漂移，偏向正确信号
    - 核心思路：一旦 agent 生成正确答案，就"锁定"其状态不再被后续辩论影响（oracle 方式，需要 ground truth）
    - 实用变体：MAD-confidence——高置信度的回答不易被改变
    - 设计动机：Martingale 意味着需要人为引入"正确方向的漂移"才能让辩论有用

## 实验关键数据

### 主实验 (Qwen2.5-7B-Instruct, 5 agents)

| 方法 | Arithmetics | GSM8K | MMLU PM | MMLU FL | HellaSwag | CSQA | HH-RLHF | Avg |
|------|------------|-------|---------|---------|-----------|------|---------|-----|
| Single-Agent | 0.814 | 0.871 | 0.787 | 0.491 | 0.788 | 0.815 | 0.477 | 0.721 |
| Decentralized MAD (T=2) | 0.760 | 0.887 | 0.805 | 0.556 | 0.803 | 0.857 | 0.497 | 0.738 |
| Decentralized MAD (T=5) | 0.670 | 0.833 | 0.805 | 0.476 | 0.800 | 0.843 | 0.507 | 0.705 |
| **Majority Voting** | **0.990** | **0.940** | 0.794 | **0.540** | 0.803 | 0.830 | 0.487 | **0.769** |

### 消融实验

| 观察 | 发现 |
|------|------|
| Agent 数 1→5 | 性能持续提升，主要来自集成效应 |
| Debate 轮数 T=2→5 | 部分场景性能反而下降（尤其 Arithmetics） |
| Centralized MAD | 大幅劣于 Majority Voting（single judge 瓶颈） |
| Martingale 验证 | agent 平均准确率在辩论轮次间基本持平（实验验证理论预测） |

### 关键发现
- **Majority Voting 在大多数场景下与 MAD 持平甚至更好**——特别是 Arithmetics（0.99 vs 0.67-0.84）
- 辩论轮数增加不一定有益，T=5 时部分 benchmark 性能显著下降（过度辩论导致正确 agent 被误导）
- Centralized MAD 表现最差——中心节点成为性能瓶颈
- MAD-oracle 大幅提升（验证了"偏向正确信号"策略的上界）
- 扩展到 32B 模型和异质 agent 设置，结论一致

## 亮点与洞察
- **Martingale 理论框架**：将 MAD 的辩论过程形式化为 martingale，这是对 MAD 有效性机制的首个严格理论分析。优美且有解释力——辩论的"零漂移"性质完美解释了为什么更多轮辩论不等于更好结果
- **"简单即强大"的反直觉结论**：在多 agent LLM 系统中，简单的投票策略就能获得大部分收益，复杂的辩论架构可能是过度工程
- **理论指导设计**：从 martingale 理论出发，提出"偏向正确信号"的改进方向，从理论分析直接产出可操作的设计原则

## 局限与展望
- DCM 模型对 LLM 行为的近似可能不完美——真实 LLM 的信念更新不完全符合贝叶斯共轭
- 实验主要用 7B/8B 模型，更大模型的辩论动态可能不同（但 32B 实验显示结论一致）
- MAD-oracle 需要 ground truth，实用变体（confidence-based）的效果有限
- 仅考虑同质 agent 设置（所有 agent 用同一模型），虽然也做了异质实验但不是重点
- 开放式生成任务上 martingale 理论的适用性需要进一步验证

## 相关工作与启发
- **vs BCCS**：BCCS 试图用信念校准改进共识，本文从理论上说明辩论本身不改善期望——两文结论互补
- **vs DyLAN**：DyLAN 动态选 agent 做辩论，但本文暗示如果底层是 martingale，动态选择的收益可能有限
- **vs Self-Consistency (Wang et al.)**：Self-Consistency 就是 majority voting on single agent，本文将此推广到多 agent 场景并给出理论支撑

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ Martingale 分析框架是全新的理论贡献，反直觉结论有很高学术价值
- 实验充分度: ⭐⭐⭐⭐ 7 个 benchmark、多种 MAD 变体、多种模型规模，消融充分
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，实验与理论呼应紧密
- 价值: ⭐⭐⭐⭐⭐ 对 multi-agent LLM 系统的设计有根本性指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](large_language_models_miss_the_multi-agent_mark.md)
- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](../../ACL2026/multi_agent/agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](../../AAAI2026/multi_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[ICLR 2026\] Cache-to-Cache: Direct Semantic Communication Between Large Language Models](../../ICLR2026/multi_agent/cache-to-cache_direct_semantic_communication_between_large_language_models.md)
- [\[ICML 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](../../ICML2025/multi_agent/theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)

</div>

<!-- RELATED:END -->
