---
title: >-
  [论文解读] Voting or Consensus? Decision-Making in Multi-Agent Debate
description: >-
  [多智能体] 系统性对比了多智能体辩论中 7 种决策协议（投票 vs 共识），发现共识协议在知识任务上提升 2.8%、投票协议在推理任务上提升 13.2%，并提出 AAD 和 CI 两种增强答案多样性的新方法，分别带来 3.3% 和 7.4% 的性能提升。 - 核心问题： 多智能体辩论（Multi-Agent Debate…
tags:
  - "多智能体"
---

# Voting or Consensus? Decision-Making in Multi-Agent Debate

- **会议**: ACL 2025
- **arXiv**: [2502.19130](https://arxiv.org/abs/2502.19130)
- **代码**: [GitHub](https://github.com/lkaesberg/decision-protocols)
- **领域**: 其他
- **关键词**: 多智能体辩论, 决策协议, 投票, 共识, 答案多样性, AAD, CI

## 一句话总结

系统性对比了多智能体辩论中 7 种决策协议（投票 vs 共识），发现共识协议在知识任务上提升 2.8%、投票协议在推理任务上提升 13.2%，并提出 AAD 和 CI 两种增强答案多样性的新方法，分别带来 3.3% 和 7.4% 的性能提升。

## 研究背景与动机

- **核心问题**: 多智能体辩论（Multi-Agent Debate, MAD）的成功高度依赖参数选择，其中**决策协议**（decision protocol）——即多个 agent 如何从讨论中收敛到最终答案——对结果影响巨大，但现有研究将其作为固定变量而非可优化的关键因素。
- **现有方法局限**:
    - **缺乏系统对比**: Exchange-of-Thought（Yin et al. 2023）仅使用共识方法，Yang et al. 2024 仅关注投票协议，ReConcile（Chen et al. 2023）混合两者但未单独分析各协议的贡献，导致无法回答"对于特定任务类型，哪种决策协议最优"这一基本问题。
    - **参数混淆**: 先前研究在实验中同时改变多个参数（决策协议 + 讨论轮数 + agent 数量 + 响应生成器），无法孤立决策协议本身对性能的影响，实验可比较性差。
    - **任务适应性未量化**: 直觉上知识任务和推理任务可能需要不同决策策略，但这一假设未被定量验证，现有方法对所有任务类型不加区分地使用同一种协议。
- **本文动机**: 通过严格的**单变量控制实验**——仅改变决策协议这一个因素，在 3 个知识任务 + 3 个推理任务上系统评估 4 种投票协议和 3 种共识协议的效果差异，并提出促进答案多样性的新方法。

## 方法详解

### 整体框架

基于 Llama 3（8B / 70B）搭建多智能体辩论系统：自动生成 3 个专家角色（persona），在多轮讨论后通过指定的决策协议达成最终答案。整个框架由三个核心组件构成：**讨论范式**（定义 agent 间通信结构和轮次规则）、**决策协议**（定义何时终止讨论以及如何选出最终方案）、**响应生成器**（定义 agent 回复风格，如中性、批判性或仅推理）。每个 agent 每轮生成一条回复，仅保留最近两轮的消息以控制上下文长度。

### 关键设计

1. **7 种决策协议的统一评估框架**: 实现了 3 种共识协议（多数共识 >50%、超级多数共识 >66%、全体一致 100%）和 4 种投票协议（简单投票——每人一票取最多；排名投票——按排名加权；批准投票——每人可投多票；累积投票——分配 25 分）。共识协议要求 agent 在讨论中逐步趋同直至达到协议阈值，投票协议则在 3 轮讨论后由所有 agent 从候选方案中投票选出最终答案。关键区别在于：共识是"协商—收敛"过程，投票是"探索—选择"过程。

2. **All-Agents Drafting (AAD)**: 针对默认设置中后续 agent 被第一个 agent 答案偏置的问题，AAD 强制所有 agent 在第一轮**独立生成**各自的初始方案，不可见其他 agent 的输出。从第二轮起恢复正常讨论。这确保了初始答案池具有多样性，避免群体思维（groupthink）。AAD 兼容所有 7 种决策协议。

3. **Collective Improvement (CI)**: 在 AAD 独立起草基础上进一步限制通信——取消 agent 之间的直接消息交换，每轮结束后各 agent 只能看到上一轮的**解决方案集合**（而非讨论历史），独立地对已有方案进行改进或提出新方案。CI 专为投票协议设计（因为不允许 agent 之间建立渐进共识），通过抑制过度交流来维持答案多样性，使投票池在整个讨论过程中保持丰富。

## 实验

### 基准数据集

| 数据集 | 任务类型 | 具体内容 | 样本量 |
|--------|---------|---------|-------|
| MMLU | 知识 | 广泛主题多选题测试 | 子集采样 |
| MMLU-Pro | 知识 | 领域专精难题多选 | 子集采样 |
| GPQA | 知识 | 专家级问答（难以网搜） | 子集采样 |
| SQuAD 2.0 | 推理 | 阅读理解（含不可答问题） | 子集采样 |
| StrategyQA | 推理 | 多步推理是非题 | 子集采样 |
| MuSR | 推理 | 长文本谋杀案推理 | 子集采样 |

### 主实验：决策协议对比（Llama 3 8B，3 次运行平均 ± std）

| 决策协议类别 | MMLU | MMLU-Pro | GPQA | SQuAD 2.0 | StrategyQA | MuSR |
|-------------|------|----------|------|-----------|------------|------|
| **投票协议均值** | 较低 | 较低 | 较低 | **+13.1%** | **+0.2%** | **+26.4%** |
| **共识协议均值** | **+2.3%** | **+4.9%** | **+1.3%** | 较低 | 较低 | 较低 |
| CoT Baseline | 低于 MAD | 低于 MAD | 低于 MAD | 低于 MAD | 低于 MAD | 低于 MAD |

**核心发现**: 共识协议在全部 3 个知识任务上一致优于投票（平均 +2.8%），投票协议在全部 3 个推理任务上显著优于共识（平均 +13.2%），且均优于 CoT 单 agent 基线。共识平均 1.42 轮达成决策，投票需 3.38 轮。批准投票因 agent 过度顺从而在 59% 的情况下无法决策。

### 缩放分析（StrategyQA，简单投票协议）

| 缩放维度 | 变化范围 | 效果趋势 | 解读 |
|---------|---------|---------|------|
| 增加 agent 数量 | 1 → 10 | 准确率线性上升 ↑ | 类似 self-consistency 多采样，更大知识库 |
| 增加讨论轮数 | 1 → 10 | 准确率线性下降 ↓ | 问题漂移（problem drift）导致偏离原始任务 |
| 挑战轮次（提供讨论历史） | 额外 +1 轮 | 挑战率下降 10%，无正面效果 | agent 倾向于认同已有讨论，无法有效自我修正 |

### 答案多样性实验（StrategyQA）

| 方法 | 答案余弦相似度 | 平均准确率 | vs Baseline |
|------|--------------|-----------|-------------|
| Baseline | 0.888 | 58.3% | — |
| AAD | 0.870 | 62.8% | **+3.3%** |
| CI | 0.845 | 65.7% | **+7.4%** |
| Critical Response | 0.843 | 59.4% | +1.1% |
| Reasoning Response | 0.916 | 51.9% | -6.4% |

答案多样性（余弦相似度更低）与任务准确率正相关。CI 在相似度最低（0.845）时获得最高准确率（65.7%）。但直接通过提示风格（批判性/仅推理）改变多样性效果不稳定，甚至有害。

### 关键发现

1. **任务类型决定最优决策协议**: 知识任务用共识（多 agent 交叉验证减少事实错误），推理任务用投票（允许并行探索多条推理路径后择优）。
2. **增 agent 优于增轮数**: 扩 agent 数如同 self-consistency 多采样，效果线性提升；增加讨论轮数反而因 problem drift 导致性能下降，挑战了"更多讨论 = 更好结果"的直觉。
3. **结构化通信比提示工程更有效**: AAD/CI 通过改变通信结构（而非改变提示语气）来提升多样性，效果稳定；批判性和推理限制型提示反而可能降低讨论质量。
4. **共识决策效率更高**: 共识协议仅需 1.42 轮（投票 3.38 轮），在知识任务上以更低计算成本获得更高性能。
5. **批准投票在 LLM agent 中失效**: agent 的过度顺从倾向导致批准投票 59% 无法达成决策，揭示了人类决策协议直接迁移至 LLM 系统的局限性。

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|------------|------|
| 新颖性 | 6 | 首次系统对比投票 vs 共识并提出 AAD/CI，但核心思想（独立采样、限制通信）非全新 |
| 实验充分性 | 8 | 6 个数据集 × 7 种协议 × 多种消融，控制变量严谨，3 次重复实验 |
| 实用价值 | 8 | 给出了明确的协议选择指南和可复现的实践建议，开源代码和数据 |
| 表达清晰度 | 8 | 结构清晰，图表丰富，实验设计和结论有明确对应关系 |

## 亮点

- 首次系统性地在知识/推理两类任务上对比 7 种决策协议，建立了清晰的任务—协议选择矩阵
- 严格的单变量控制实验设计：每次仅改变决策协议，消除了参数混淆问题
- AAD 和 CI 方法简洁优雅——不修改模型、不改变提示内容，仅调整通信结构即获得显著提升
- 发现 agent 缩放（增加数量）比讨论缩放（增加轮数）更有效，为多智能体系统资源分配提供指导
- 量化揭示了答案多样性与任务性能之间的正相关关系（cosine similarity vs accuracy）
- 揭示了 LLM agent 的过度顺从问题在批准投票中的极端表现（59% 无法决策），为协议设计敲响警钟

## 局限性

- 实验仅使用 Llama 3（8B / 70B），未验证在 GPT-4、Claude 等闭源模型上的泛化性
- 多智能体辩论的计算开销大（共识约 5×、投票约 10× 于 CoT 基线），性能提升与资源消耗之间的 ROI 需审慎评估
- 受计算限制使用数据集子集采样（95% 置信水平），虽有 3 次重复但仍存在一定统计波动
- 仅关注决策协议这一维度，未深入探讨 persona 设计、提示工程与决策协议之间的交互效应
- agent 的过度顺从（sycophancy）倾向是根本性限制，AAD 和 CI 只是缓解而未根治
- 未考虑异构 agent（不同模型组成的团队）的决策动态，现实部署中异构团队可能更常见

## 相关工作

- **多智能体辩论**: Du et al. 2023（Improving Factuality & Reasoning）、Exchange-of-Thought（Yin et al. 2023，共识方法）、ReConcile（Chen et al. 2023，混合投票+共识）、Liang et al. 2024（鼓励发散思维）
- **LLM Agent 增强**: 自一致性（Wang et al. 2023，多路径采样投票）、CoT 推理（Wei et al. 2022）、persona-based prompting（Jiang et al. 2024）、Self-Refine（Madaan et al. 2023）
- **决策理论与投票机制**: 社会选择理论（List 2022）、共识 vs 投票（Jones 1994）、Yang et al. 2024（多投票协议 LLM 对比）
- **MALLM 框架**: Becker et al. 2025 提出多智能体 LLM 协作框架

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次控制变量系统化比较，方法论贡献扎实
- **实用性**: ⭐⭐⭐⭐⭐ — 给出了明确的任务-协议选择指南，实践价值高
- **严谨性**: ⭐⭐⭐⭐ — 多次重复实验+标准差报告，但数据子集有限
- **综合**: ⭐⭐⭐⭐

<!-- Systematic comparison of voting vs consensus decision protocols in multi-agent LLM debates -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives](../../ACL2026/multi_agent/social_dynamics_as_critical_vulnerabilities_that_undermine_objective_decision-ma.md)
- [\[ACL 2025\] CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate](cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate.md)
- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](../../NeurIPS2025/multi_agent/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)
- [\[ACL 2026\] ConSensus: Multi-Agent Collaboration for Multimodal Sensing](../../ACL2026/multi_agent/consensus_multi-agent_collaboration_for_multimodal_sensing.md)
- [\[ACL 2026\] OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](../../ACL2026/multi_agent/oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)

</div>

<!-- RELATED:END -->
