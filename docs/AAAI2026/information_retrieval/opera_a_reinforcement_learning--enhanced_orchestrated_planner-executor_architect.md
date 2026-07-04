---
title: >-
  [论文解读] OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval
description: >-
  [AAAI2026][信息检索/RAG][RAG] 提出 OPERA 框架，通过 Goal Planning Module 和 Reason-Execute Module 的分层架构，结合专为多 agent 设计的 MAPGRPO 训练算法，大幅提升 reasoning-oriented multi-hop retrieval 性能。
tags:
  - "AAAI2026"
  - "信息检索/RAG"
  - "RAG"
  - "multi-hop retrieval"
  - "reinforcement-learning"
  - "GRPO"
  - "multi-agent"
---

# OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval

**会议**: AAAI2026  
**arXiv**: [2508.16438](https://arxiv.org/abs/2508.16438)  
**代码**: [Ameame1/OPERA](https://github.com/Ameame1/OPERA)  
**领域**: 信息检索  
**关键词**: RAG, multi-hop retrieval, reinforcement-learning, GRPO, multi-agent

## 一句话总结
提出 OPERA 框架，通过 Goal Planning Module 和 Reason-Execute Module 的分层架构，结合专为多 agent 设计的 MAPGRPO 训练算法，大幅提升 reasoning-oriented multi-hop retrieval 性能。

## 背景与动机

### 领域现状

**领域现状**：现有 RAG 系统在复杂 multi-hop 问题上表现不佳，主要瓶颈在于 retrieval 与 reasoning 的弱耦合

### 现有痛点

**现有痛点**：静态 plan（如 PlanRAG）无法动态适应检索过程中的新情况

### 核心矛盾

**核心矛盾**：单一 LLM 同时负责 planning 和 execution，导致可靠性受限

### 解决思路

**解决思路**：已有 RL 方法（如 BGM）仅优化 retriever-LLM 间隙，未实现 agent 级别的精细 credit assignment

### 解决思路

**本文目标**：如何在 RAG 框架中实现 reasoning 与 retrieval 的深度耦合，使系统能够针对复杂 multi-hop 问题进行有效的计划分解、自适应检索和精准过滤？

## 方法详解

### 整体框架
OPERA 解耦为两层：
1. **Goal Planning Module (GPM)**: Plan Agent 将复杂问题分解为子目标 $\mathcal{P}=\{p_1,\dots,p_m\}$，子目标间通过 placeholder 建立依赖
2. **Reason-Execute Module (REM)**: Analysis-Answer Agent 进行信息充分性判断 $\phi \in \{0,1\}$ 和答案抽取；Rewrite Agent 在信息不足时重写 query 以改善后续检索
3. **Trajectory Memory Component (TMC)**: 记录全部操作轨迹，增强可解释性

### 关键设计：MAPGRPO
基于 GRPO 扩展为 Multi-Agents Progressive Group Relative Policy Optimization：
- 三个 agent 按顺序训练，每个 agent 使用异构 reward function
- Plan Agent reward: $r_{\text{plan}} = \lambda_1 f_{\text{logic}} + \lambda_2 f_{\text{struct}} + \lambda_3 f_{\text{exec}}$
- Analysis-Answer Agent reward: $r_{\text{ana}} = \alpha \cdot \mathbb{I}[\phi=\phi^*] + \beta \cdot \text{EM}(a_i,a_i^*) + \gamma \cdot f_{\text{format}}$
- Rewrite Agent reward: $r_{\text{rew}} = \omega_1 \sqrt{\text{NDCG@}k} + \omega_2 f_{\text{format}}$
- **High-Score Sample Selection**: 每个 group 中注入一个预打分的高质量样本 $c_{\text{best}}$，缓解 reward sparsity，降低 policy gradient variance

## 实验关键数据


### 主实验

| 方法 | HotpotQA EM | 2WikiMHQA EM | Musique EM |
|------|------------|-------------|------------|
| Adaptive-RAG (SFT) | 45.7% | 30.1% | 24.3% |
| BGM (RL) | 41.5% | 44.3% | 19.6% |
| **OPERA (MAPGRPO)** | **57.3%** | **60.2%** | **39.7%** |

- Musique 上相对最佳 baseline 提升 63.4%
- Ablation: 移除 Plan Agent → EM 从 39.7% 降至 17.1%（低于无训练 CoT 的 21.2%）
- Out-of-domain: 在 NQ（单跳）上 MAPGRPO 达 36.6% EM，SFT 反而降至 19.5%

## 亮点与洞察
- 架构设计 > 训练方法：移除 Plan Agent 后性能崩塌，证明分层架构是核心贡献
- MAPGRPO 的 expert injection 策略显著降低 policy gradient variance
- 框架在 out-of-domain 任务（单跳 QA）也能自适应，RL 训练不会过拟合固定模式

## 局限与展望
- Rewrite Agent 训练不稳定（条件激活导致 reward 稀疏）
- 理论仅保证局部收敛，Musique EM 仍 < 40%
- 推理延迟较高，Analysis-Answer Agent latency 波动大
- 暂未验证在更大参数量 LLM（>7B）上的效果
- 对模糊分解和长推理链仍有困难

## 相关工作与启发

| 维度 | OPERA | Adaptive-RAG | ReAct | BGM |
|------|-------|-------------|-------|-----|
| Planning | 动态子目标 | 按复杂度路由 | 无显式 plan | 无 |
| Retrieval | Rewrite Agent 自适应 | 固定策略 | 推理-行动循环 | RL 桥接 |
| 训练 | MAPGRPO (role-specific) | SFT | 无训练 | GRPO |
| 可解释性 | TMC 轨迹记录 | 无 | 部分 | 无 |

## 相关工作与启发
- 多 agent 分层架构 + 角色专用 reward 是一种有前景的 RAG 设计范式
- MAPGRPO 的 sequential training + high-score sample injection 可推广到其他多 agent RL 场景
- TMC 的设计思路可用于提升 AI 系统的审计与调试能力
- "架构贡献 > 训练贡献" 的结论值得在其他复杂系统设计中参考
- Rewrite Agent 的条件激活机制体现了计算资源按需分配的思想

## 评分
- 新颖性: ⭐⭐⭐⭐ — 多 agent 架构 + MAPGRPO 训练方法具有原创性
- 实验充分度: ⭐⭐⭐⭐ — 三个主流 benchmark + ablation + OOD 实验
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，理论分析较完整
- 价值: ⭐⭐⭐⭐ — 对复杂 RAG 系统设计有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](../../ACL2026/information_retrieval/agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](../../ACL2026/information_retrieval/learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[AAAI 2026\] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](../../ACL2026/information_retrieval/chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](../../ACL2026/information_retrieval/language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)

</div>

<!-- RELATED:END -->
