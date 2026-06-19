---
title: >-
  [论文解读] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration
description: >-
  [ACL 2025][LLM 其他][多Agent系统] 本文提出 AgentDropout，通过在多轮讨论中动态消除冗余 Agent（节点剪枝）和冗余通信边（边剪枝），在降低 21.6% prompt token 消耗的同时提升了 1.14 分的任务性能。 1. 领域现状：基于 LLM 的多 Agent 系统（MAS）通过…
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "多Agent系统"
  - "通信拓扑优化"
  - "节点剪枝"
  - "边剪枝"
  - "Token效率"
---

# AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration

**会议**: ACL 2025  
**arXiv**: [2503.18891](https://arxiv.org/abs/2503.18891)  
**代码**: [https://github.com/wangzx1219/AgentDropout](https://github.com/wangzx1219/AgentDropout)  
**领域**: LLM/NLP  
**关键词**: 多Agent系统, 通信拓扑优化, 节点剪枝, 边剪枝, Token效率

## 一句话总结

本文提出 AgentDropout，通过在多轮讨论中动态消除冗余 Agent（节点剪枝）和冗余通信边（边剪枝），在降低 21.6% prompt token 消耗的同时提升了 1.14 分的任务性能。

## 研究背景与动机

1. **领域现状**：基于 LLM 的多 Agent 系统（MAS）通过模拟人类协作来解决复杂任务，但通信开销巨大且效率低下。
2. **现有痛点**：现有方法（如 AgentPrune）只修剪通信边但不修剪 Agent 节点，且在所有通信轮次中使用相同的剪枝策略，忽略了不同讨论阶段可能需要不同角色组合。
3. **核心矛盾**：MAS 中的冗余既来自不必要的信息交换（边冗余），也来自不必要的参与者（节点冗余）。现有方法只解决了前者。
4. **本文目标**：同时识别和消除冗余 Agent 和冗余通信，且在不同讨论轮次中允许不同的角色组合。
5. **切入角度**：借鉴管理学理论——高效团队会根据任务需求动态调整成员角色和职责。
6. **核心 idea**：将 MAS 建模为通信图，通过可训练的邻接矩阵权重来学习哪些节点和边在哪些轮次中是重要的，然后分别进行 Node Dropout 和 Edge Dropout。

## 方法详解

### 整体框架

AgentDropout 分两步优化通信拓扑：(1) Node Dropout——训练轮内邻接矩阵权重，计算节点度数，在每轮讨论中移除贡献最小的节点；(2) Edge Dropout——在节点剪枝后的图上进一步训练边权重并剪枝低权重边。两步都使用策略梯度（REINFORCE）来优化不可微的效用函数。

### 关键设计

1. **Node Dropout**:

    - 功能：在不同讨论轮次中动态移除贡献最小的 Agent
    - 核心思路：初始化所有边权重为0.5，使用策略梯度优化轮内邻接矩阵 $\tilde{\mathcal{A}}_{intra}$。优化后，对每一轮的邻接矩阵计算每个节点的加权入度+出度之和，移除度数最小的节点（即 TopkNodes 函数保留高度数节点，移除低度数节点）。使用 dropout rate $\alpha$ 控制移除比例。
    - 设计动机：不同讨论阶段需要不同专家参与——例如初始阶段需要广泛的信息收集，后期需要专注的决策者。动态角色分配比固定团队更高效。

2. **Edge Dropout**:

    - 功能：精细化修剪节点间的冗余通信路径
    - 核心思路：在节点剪枝后的图上，继续用策略梯度同时优化轮内和轮间邻接矩阵，然后将低于阈值的边权重置零。使用 DAGSample 算法确保采样的通信图是有向无环图。
    - 设计动机：即使在减少了参与者后，剩余 Agent 之间仍可能存在不必要的信息交换。边剪枝进一步减少冗余通信。

3. **策略梯度优化**:

    - 功能：在不可微的效用函数下优化图拓扑
    - 核心思路：由于任务性能评估通常依赖外部 API（不可微），使用 REINFORCE 估计梯度：$\nabla \approx \frac{1}{M}\sum_{m=1}^{M}\mu(\mathcal{G}_m)\nabla\log(p(\mathcal{G}_m))$，其中采样 $M$ 个图计算概率加权的性能。
    - 设计动机：MAS 的效用函数通常是黑盒的，策略梯度是唯一可行的优化方式。

### 损失函数 / 训练策略

- 目标函数：最大化采样通信图上的任务性能期望
- 使用少量数据（约几十个样本）迭代更新邻接矩阵权重
- 推理时使用固定的优化后拓扑

## 实验关键数据

### 主实验

| 方法 | MMLU | GSM8K | HumanEval | Prompt Token↓ | Completion Token↓ |
|------|------|-------|-----------|---------------|-------------------|
| Vanilla MAS | 基准 | 基准 | 基准 | 100% | 100% |
| AgentPrune | 提升 | 提升 | 提升 | -12% | -8% |
| **AgentDropout** | **+1.14平均** | **提升** | **提升** | **-21.6%** | **-18.4%** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 仅Node Dropout | 提升 | 单独有效 |
| 仅Edge Dropout | 提升 | 单独有效 |
| Node + Edge | 最优 | 互补提升 |
| 不同dropout率 | α=0.3~0.5最优 | 过高会丢失关键信息 |

### 关键发现

- Node Dropout 和 Edge Dropout 互补——前者减少参与者数量，后者优化剩余参与者的通信
- 在更大更强的 LLM 上，MAS 的性能仍可通过交互优化进一步提升
- AgentDropout 展现了良好的跨域可迁移性和结构鲁棒性

## 亮点与洞察

- **动态角色分配**的理念直接来自管理学，将组织行为学的洞察应用到 AI 系统设计中，很有启发性。
- Node Dropout 使不同讨论轮次拥有不同的团队组合，这比固定团队更灵活，可以迁移到任何多Agent场景。
- 用极少数据（几十个样本）就能优化通信拓扑，实用性很强。

## 局限与展望

- 策略梯度优化需要多次运行 MAS 来估计梯度，前期的优化成本不低
- 目前 dropout 率是全局超参数，按轮次自适应调整可能更优
- 仅在推理/数学/代码任务上测试，创意类任务的效果未知

## 相关工作与启发

- **vs AgentPrune**: 只做边剪枝且策略不随轮次变化，AgentDropout 增加了节点剪枝和动态调整
- **vs 标准 MAS**: 全连接通信浪费大量 token，AgentDropout 证明稀疏通信反而能提升性能

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 Dropout 概念从神经网络扩展到多Agent图拓扑，角度新颖
- 实验充分度: ⭐⭐⭐⭐ 多任务测试，消融和可迁移性分析全面
- 写作质量: ⭐⭐⭐⭐ 图示清晰，与人类团队的类比直观
- 价值: ⭐⭐⭐⭐ 实用性强，代码开源，对多Agent系统优化有实际指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Many Heads Are Better Than One: Improved Scientific Idea Generation by A LLM-Based Multi-Agent System](virsci_multi_agent_idea_gen.md)
- [\[ACL 2025\] MasRouter: Learning to Route LLMs for Multi-Agent Systems](masrouter_learning_to_route_llms_for_multi-agent_systems.md)
- [\[ACL 2025\] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents](axis_efficient_human-agent-computer_interaction_with_api-first_llm-based_agents.md)
- [\[ACL 2025\] Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning](graph_counselor_multiagent_graphrag.md)
- [\[ACL 2025\] Dynamic Parallel Tree Search for Efficient LLM Reasoning](dynamic_parallel_tree_search_for_efficient_llm_reasoning.md)

</div>

<!-- RELATED:END -->
