---
title: >-
  [论文解读] KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search
description: >-
  [ICML2025][LLM Agent][知识库问答] 提出 KBQA-o1，将 ReAct Agent 与蒙特卡洛树搜索（MCTS）结合，通过策略模型和奖励模型驱动的启发式搜索实现知识库问答，在低资源设置下以 Llama-3.1-8B 将 GrailQA F1 从 48.5%（GPT-3.5-turbo SOTA）提升至 78.5%。
tags:
  - "ICML2025"
  - "LLM Agent"
  - "知识库问答"
  - "KBQA"
  - "蒙特卡洛树搜索"
  - "MCTS"
  - "ReAct Agent"
  - "低资源"
  - "逻辑形式生成"
---

# KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search

**会议**: ICML2025  
**arXiv**: [2501.18922](https://arxiv.org/abs/2501.18922)  
**代码**: [LHRLAB/KBQA-o1](https://github.com/LHRLAB/KBQA-o1)  
**领域**: LLM Agent  
**关键词**: 知识库问答, KBQA, 蒙特卡洛树搜索, MCTS, ReAct Agent, 低资源, 逻辑形式生成

## 一句话总结
提出 KBQA-o1，将 ReAct Agent 与蒙特卡洛树搜索（MCTS）结合，通过策略模型和奖励模型驱动的启发式搜索实现知识库问答，在低资源设置下以 Llama-3.1-8B 将 GrailQA F1 从 48.5%（GPT-3.5-turbo SOTA）提升至 78.5%。

## 研究背景与动机
知识库问答（KBQA）旨在利用 Freebase、Wikidata 等大规模结构化知识库回答自然语言问题。现有 LLM-based 方法面临三大挑战：

**KB 环境感知不足**：端到端方法直接生成逻辑形式，难以感知 KB 中的实体和关系，导致生成不合法的查询

**局部最优 vs 搜索空间爆炸**：CoT 方法易陷入局部最优，ToT 方法虽扩展搜索空间但面临指数级增长

**高度依赖标注数据**：训练开源 LLM 需要大量高质量人工标注，但对大规模 KB 标注成本极高

KBQA-o1 的核心思路：将 KBQA 建模为 Agent 在 KB 环境中的探索过程，用 MCTS 平衡探索效率和搜索质量，并通过自动标注进行增量微调减少对人工标注的依赖。

## 方法详解

### 整体框架
KBQA-o1 包含三个核心组件：Agent 初始化、MCTS 启发式探索、增量微调。

### 1. ReAct-based Agent 设计
- 设计了 8 种**原子查询工具**（Extract_entity, Find_relation, Merge, Order, Compare, Time_constraint, Count, Finish）
- Agent 状态空间 $\mathcal{H}$：每步由 Thought-Action-Observation 三元组更新
- 探索空间动态依赖当前状态和 KB 环境，工具参数从 KB 候选集中选择

### 2. MCTS 启发式搜索
- **策略模型 $\pi_{\text{policy}}$**：以当前状态为输入，预测到终态的完整探索序列，用 SFT 训练
- **奖励模型 $\pi_{\text{reward}}$**：以问题为输入评估生成的逻辑形式质量，用 SFT 训练
- **评分函数**：$R_\pi(y|x) = \beta + \alpha \log \pi(y|x)$，其中 $\beta=100$ 为满分

MCTS 四阶段：
- **Selection**：UCT 算法平衡探索和利用，选择子节点
- **Expansion**：策略模型通过 beam search 生成候选，SimCSE 与 KB 可执行选项匹配，策略模型打分取 top-d
- **Simulation**：选最优子节点向前模拟至终态
- **Back-propagation**：奖励模型评估完整轨迹，Q 值逐层回传

### 3. 增量微调
- 用少量标注数据初始化策略/奖励模型
- MCTS 探索未标注问题生成自动标注数据
- 奖励模型过滤低质量标注，增量微调提升两个模型

### 损失函数
- 策略模型：$\mathcal{L}_{\text{SFT}}(\pi_{\text{policy}}, \mathcal{D}_a) = -\mathbb{E}[\sum_{t=1}^l \log \pi_{\text{policy}}(\sum_{i=t}^l \mathbf{e}_i | \mathbf{h}_{t-1})]$
- 奖励模型：$\mathcal{L}_{\text{SFT}}(\pi_{\text{reward}}, \mathcal{D}_a) = -\mathbb{E}[\log \pi_{\text{reward}}(F_{\mathbf{h}_l} | \mathcal{Q})]$

## 实验关键数据

### GrailQA（40-shot 低资源设置）

| 方法 | LLM | I.I.D F1 | Comp. F1 | Zero-shot F1 | Overall F1 |
|------|-----|----------|----------|--------------|------------|
| KB-BINDER | GPT-3.5-turbo | 43.3 | 36.6 | 44.0 | 42.2 |
| ARG-KBQA | GPT-3.5-turbo | 51.5 | 41.8 | 52.1 | 48.5 |
| **KBQA-o1** | Llama-3.1-8B | **85.5** | **77.6** | **76.1** | **78.5** |
| KBQA-o1 | Qwen2.5-72B | 87.4 | 83.0 | 81.9 | 82.1 |
| 全监督 TIARA | - | 91.2 | 74.8 | 80.7 | 81.9 |

### WebQSP（100-shot）& GraphQ（100-shot）

| 数据集 | 方法 | F1 |
|--------|------|-----|
| WebQSP | ARG-KBQA (GPT-3.5) | 58.8 |
| WebQSP | KBQA-o1 (Llama-3.3-70B) | **67.0** |
| GraphQ | KBQA-o1 (Llama-3.3-70B) | **35.1** |

### 关键发现
- 8B 模型 KBQA-o1 **大幅超越** GPT-3.5-turbo 的 SOTA 方法（78.5% vs 48.5%），提升 30 个点
- 在 Compositional 和 Zero-shot 等困难场景提升尤为显著
- 支持 Llama-3、Qwen2.5、Gemma-2 等多种开源模型，具有 plug-and-play 特性
- MCTS agent 探索和增量微调均有显著消融贡献

## 亮点与洞察
1. **将 KBQA 建模为 Agent+MCTS 问题**是本文的核心创新，将 AlphaGo 的思路迁移到知识库问答
2. **原子查询工具设计**精巧，8 种工具覆盖了 KBQA 的所有逻辑形式构造需求
3. **SimCSE 匹配**巧妙地将模型生成与 KB 环境选项对齐，解决了 KB 感知不足的问题
4. **自动标注 + 增量微调**大幅降低标注依赖，40-shot 即可接近甚至超越全监督方法
5. 多模型通用的 plug-and-play 设计增强了方法的实用性

## 局限与展望
1. MCTS 搜索带来的推理开销较大（N 次 rollout），在线部署延迟较高
2. 仅在 Freebase 数据集上验证，对 Wikidata 等其他 KB 的泛化性未充分讨论
3. 策略和奖励模型需要分别微调，训练管线较复杂
4. 自动标注数据的质量依赖奖励模型的泛化能力，可能存在误差累积

## 相关工作与启发
- **RAP (Hao et al., 2023)**：首次将 MCTS 与 LLM 结合用于推理，KBQA-o1 将其拓展到 KB 环境
- **ReAct (Yao et al., 2023b)**：Agent 框架的基础
- **KB-BINDER / KB-Coder**：先前低资源 KBQA 方法，基于 ICL 的 GPT-3.5
- 启示：MCTS 驱动的 Agent 探索可推广到其他结构化知识查询任务

## 评分
- 新颖性: ⭐⭐⭐⭐ （Agent+MCTS 在 KBQA 的首次应用，框架设计完整）
- 实验充分度: ⭐⭐⭐⭐ （三个数据集、多模型、消融实验完备）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，理论和实验相辅相成）
- 价值: ⭐⭐⭐⭐⭐ （低资源 KBQA 的突破性工作，开源可用）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](../../ICLR2026/llm_agent/tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)
- [\[ACL 2026\] LiTS: A Modular Framework for LLM Tree Search](../../ACL2026/llm_agent/lits_a_modular_framework_for_llm_tree_search.md)
- [\[ACL 2025\] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](../../ACL2025/llm_agent/multi_agent_dialect_bias_privacy_qa.md)
- [\[ICML 2026\] Agentic Monte Carlo: Simulating Reinforcement Learning for Black-Box Agents](../../ICML2026/llm_agent/agentic_monte_carlo_simulating_reinforcement_learning_for_black-box_agents.md)
- [\[ICML 2025\] GuardAgent: Safeguard LLM Agents via Knowledge-Enabled Reasoning](guardagent_safeguard_llm_agents_by_a_guard_agent_via_knowledge-enabled_reasoning.md)

</div>

<!-- RELATED:END -->
