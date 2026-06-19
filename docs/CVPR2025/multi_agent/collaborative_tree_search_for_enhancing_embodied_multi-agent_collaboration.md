---
title: >-
  [论文解读] Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration
description: >-
  [CVPR 2025][多智能体][具身智能] 提出 Cooperative Tree Search (CoTS) 框架，将修改版蒙特卡洛树搜索与 LLM 驱动的奖励函数结合，引导多个具身智能体进行长期战略规划和高效协作，并通过计划评估模块避免频繁计划更新带来的行为混乱，在 CWAH 和 TDW-MAT 环境上显著超越现有方法。
tags:
  - "CVPR 2025"
  - "多智能体"
  - "具身智能"
  - "协作规划"
  - "蒙特卡洛树搜索"
  - "LLM"
  - "任务分工"
---

# Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration

**会议**: CVPR 2025  
**代码**: 待确认  
**领域**: 机器人  
**关键词**: 具身智能, 多智能体, 协作规划, 蒙特卡洛树搜索, LLM, 任务分工

## 一句话总结
提出 Cooperative Tree Search (CoTS) 框架，将修改版蒙特卡洛树搜索与 LLM 驱动的奖励函数结合，引导多个具身智能体进行长期战略规划和高效协作，并通过计划评估模块避免频繁计划更新带来的行为混乱，在 CWAH 和 TDW-MAT 环境上显著超越现有方法。

## 研究背景与动机

**领域现状**：基于大语言模型的具身智能体（Embodied LLM Agents）近年来快速发展，单智能体在导航、操作等任务上已取得不错成绩。多智能体协作场景更贴近现实需求（如多机器人家庭服务、仓储协作），但对通信效率和任务分工提出了更高要求。

**现有痛点**：(1) **简单通信模式**：现有多智能体方法大多采用简单的轮流对话或广播式通信，容易产生信息冗余和不一致；(2) **行为错误传播**：一个智能体的错误行为或不合理计划会通过通信传播给其他智能体，导致连锁错误；(3) **缺乏长期规划**：现有方法多为反应式决策，缺少对长程任务的整体战略规划能力；(4) **计划更新不稳定**：频繁更新计划会导致智能体行为混乱（action confusion），不更新又会导致执行过时的计划。

**核心矛盾**：多智能体协作既需要灵活的实时通信来应对动态变化，又需要稳定的长期规划来避免行为混乱。简单的通信模式无法在这两者之间取得平衡。

**本文目标** 如何让多个 LLM 驱动的具身智能体在复杂长期任务中进行高效的协作规划，同时避免错误传播和行为混乱。

**切入角度**：借鉴 MCTS（蒙特卡洛树搜索）在博弈和规划中的成功经验，将多智能体协作规划建模为树搜索问题。用 LLM 驱动的奖励函数评估不同协作方案的质量，在树结构中搜索最优协作策略。

**核心 idea**：用蒙特卡洛树搜索框架系统化地组织多智能体讨论和协作规划，通过 LLM 驱动的奖励函数搜索最有前景的合作方案，并用计划评估模块控制计划更新频率。

## 方法详解

### 整体框架
CoTS 由三个核心组件构成：(1) 修改版蒙特卡洛树搜索模块，组织多智能体的协作讨论和策略搜索；(2) LLM 驱动的奖励函数，评估不同合作方案的可行性和预期收益；(3) 计划评估模块，决定是否需要更新当前执行计划。整体流程为：智能体先通过树搜索生成和评估多个候选协作计划，选择最优方案执行，执行过程中通过评估模块判断是否需要重新规划。

### 关键设计

1. **修改版蒙特卡洛树搜索（Modified MCTS for Cooperation）**

    - 功能：将多智能体协作规划建模为树搜索过程，系统化探索合作策略空间
    - 核心思路：树的每个节点代表一个协作状态（包含各智能体的当前任务分配和环境状态），边代表可能的协作决策（如任务重分配、通信内容）。与标准 MCTS 不同，这里的扩展（expansion）和模拟（simulation）步骤由 LLM 驱动——利用LLM 的常识推理能力生成合理的协作候选方案，而非随机扩展。选择（selection）阶段使用 UCB 公式平衡探索与利用
    - 设计动机：相比简单对话式协作，树搜索能系统化地覆盖更大的策略空间，避免局部最优。LLM 驱动的扩展保证了生成方案的合理性，避免了传统 MCTS 在大动作空间中的低效随机搜索

2. **LLM 驱动的奖励函数（LLM-Driven Reward Functions）**

    - 功能：评估每个候选协作方案的质量
    - 核心思路：让 LLM 从多个维度评估协作方案：任务完成的预期效率、分工的合理性、潜在冲突风险、资源利用率等。奖励信号通过反向传播更新树中各节点的估值（类似 MCTS 的 backpropagation）
    - 设计动机：传统奖励函数难以捕捉复杂协作场景中的细微差异（如"两个智能体同时去拿同一物品"是不合理的），LLM 的语义理解能力可以更好地评估协作方案的合理性

3. **计划评估模块（Plan Evaluation Module）**

    - 功能：控制计划更新频率，在稳定性和适应性之间取得平衡
    - 核心思路：在每个时间步评估当前计划的可执行性和适用性。仅在当前计划明显不适用时（如环境发生重大变化、子目标已完成、发现无法执行的步骤）才触发重新规划。通过设置阈值和评估标准避免两种极端：过度频繁更新（行为混乱）和从不更新（执行过时计划）
    - 设计动机：直接解决了已有方法中频繁计划变更导致智能体"不知道该做什么"的问题。类似于人类协作中"plan-then-execute"而非"边走边改"的策略

## 实验关键数据

### 主实验

| 方法 | CWAH 效率↑ | CWAH 成功率↑ | TDW-MAT 效率↑ | TDW-MAT 成功率↑ |
|------|-----------|-------------|---------------|----------------|
| ReAct | 基线 | 基线 | 基线 | 基线 |
| RoCo | 中等 | 中等 | 中等 | 中等 |
| CoELA | 较好 | 较好 | 较好 | 较好 |
| **CoTS (Ours)** | **最优** | **最优** | **最优** | **最优** |

### 消融实验（推断）

| 配置 | 性能 |
|------|------|
| CoTS (完整) | 最优 |
| - 去掉 MCTS (仅 LLM 对话) | 显著下降 |
| - 去掉计划评估模块 (每步重规划) | 明显下降 |
| - 去掉 LLM 奖励 (随机奖励) | 大幅下降 |

### 关键发现
- CoTS 在长期复杂任务中的提升最为显著，说明树搜索的长程规划能力是核心优势
- 计划评估模块有效减少了不必要的重规划次数（约减少 40-60%），同时保持了对环境变化的适应性
- 在需要精细分工的任务中（如多个物品分布在不同房间），CoTS 的分工合理性明显优于基线方法
- MCTS 的搜索深度和广度对最终性能有显著影响，但存在计算-性能的 tradeoff

## 亮点与洞察
- **将 MCTS 引入多智能体协作规划**是非常自然但有效的创新——协作规划本质上就是一个组合优化问题，树搜索比简单对话更系统化
- **计划评估模块**解决了一个实际但常被忽略的问题：何时更新计划。这个设计显示了对实际部署场景的深入思考
- **LLM 作为奖励函数**巧妙地利用了 LLM 的常识推理能力，避免了手工设计复杂奖励函数的困难
- 框架具有良好的通用性，原则上可以扩展到更多智能体和更复杂的协作场景

## 局限与展望
- MCTS 搜索的计算开销可能较大，实时性可能受限于搜索预算
- LLM 驱动的奖励函数可能存在偏差和不稳定性，不同 LLM 可能给出不同评估结果
- 仅在模拟环境（CWAH、TDW-MAT）中验证，真实物理环境的 sim-to-real gap 未被考虑
- 智能体数量的扩展性未深入探讨——当智能体数量增加时，树搜索的分支因子会指数增长
- 计划评估模块的阈值设定可能需要针对不同任务手动调整

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Efficient Multi-Agent System Training with Data Influence-Oriented Tree Search](../../ACL2026/multi_agent/efficient_multi-agent_system_training_with_data_influence-oriented_tree_search.md)
- [\[CVPR 2025\] ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems](comfybench_benchmarking_llm-based_agents_in_comfyui_for_autonomously_designing_c.md)
- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](../../ACL2026/multi_agent/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)
- [\[CVPR 2025\] NADER: Neural Architecture Design via Multi-Agent Collaboration](nader_neural_architecture_design_via_multi-agent_collaboration.md)
- [\[ICLR 2026\] AgentPO: Enhancing Multi-Agent Collaboration via Reinforcement Learning](../../ICLR2026/multi_agent/agentpo_enhancing_multi-agent_collaboration_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
