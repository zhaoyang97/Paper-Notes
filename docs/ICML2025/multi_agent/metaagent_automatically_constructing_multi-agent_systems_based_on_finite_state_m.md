---
title: >-
  [论文解读] MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines
description: >-
  [ICML 2025][多智能体][Multi-Agent System] 提出 MetaAgent，一个基于有限状态机（FSM）的框架，给定任务描述即可自动设计多智能体系统，无需外部训练数据，支持工具调用和状态回溯，在文本任务、ML 任务和软件开发任务上超越现有自动设计方法并逼近人工设计系统性能。
tags:
  - "ICML 2025"
  - "多智能体"
  - "Multi-Agent System"
  - "Finite State Machine"
  - "LLM Agent"
  - "自动化设计"
  - "工具集成"
---

# MetaAgent: Automatically Constructing Multi-Agent Systems Based on Finite State Machines

**会议**: ICML 2025  
**arXiv**: [2507.22606](https://arxiv.org/abs/2507.22606)  
**代码**: [SaFoLab-WISC/MetaAgent](https://github.com/SaFoLab-WISC/MetaAgent/)  
**领域**: 优化  
**关键词**: Multi-Agent System, Finite State Machine, LLM Agent, 自动化设计, 工具集成

## 一句话总结

提出 MetaAgent，一个基于有限状态机（FSM）的框架，给定任务描述即可自动设计多智能体系统，无需外部训练数据，支持工具调用和状态回溯，在文本任务、ML 任务和软件开发任务上超越现有自动设计方法并逼近人工设计系统性能。

## 研究背景与动机

现有多智能体系统面临两大核心问题：

**人工设计成本高**：MetaGPT、ChatDev 等系统需要大量人力实现复杂代码库，且只能解决特定场景任务，泛化能力有限。

**已有自动设计方法存在明显短板**：
   - SPP、AutoAgents、EvoAgent 为每个具体案例单独设计系统，缺乏泛化性
   - SPP 不支持工具使用
   - ADAS、Symbolic Learning 依赖大量外部数据和迭代训练步骤
   - 所有现有方法均采用线性/辩论/协调器等刚性通信结构，缺乏状态回溯能力，遇到错误时难以修正前序步骤

| 特性 | MetaGPT | AutoAgents | SPP | EvoAgent | ADAS | Symbolic | **MetaAgent** |
|------|---------|------------|-----|----------|------|----------|---------------|
| 自动设计 | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | **✓** |
| 泛化能力 | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | **✓** |
| 工具支持 | ✗ | ✓ | ✗ | ✓ | ✗ | ✓ | **✓** |
| 回溯能力 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓** |
| 无需外部数据 | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | **✓** |

MetaAgent 是唯一同时满足全部五项关键特性的框架。

## 方法详解

### 整体框架

MetaAgent 将多智能体系统建模为有限状态机 $\mathcal{M} = (\Sigma, S, s_0, F, \delta)$：

- $\Sigma$：输入字母表，即任务域中的具体案例集合
- $S$：有限状态集合
- $s_0 \in S$：初始状态
- $F \subseteq S$：终止状态集合
- $\delta$：状态转移函数

每个状态包含四个核心组件：
1. **Task-Solving Agent**：负责执行当前子任务的智能体
2. **State Instruction**：自然语言指令，描述该状态下需要完成的子任务
3. **Condition Verifier**：检查输出是否满足状态转移条件
4. **Listeners**：接收当前状态输出的下游智能体

框架分为两个阶段：**构建阶段**（自动生成 FSM）和**部署阶段**（FSM 控制执行）。

### 关键设计

#### 1. Agent 设计（Construction Phase - Step 1）

Designer LLM 接收任务描述后：
- 生成全面的任务分析和系统目标
- 设计最精简但有效的 Agent 集合（降低成本）
- 为每个 Agent 输出结构化 JSON 配置：名称、系统提示词、分配的工具（代码解释器、搜索引擎等）

#### 2. 状态与转移条件设计（Construction Phase - Step 2）

Designer LLM 以"远见规划者"角色构建 FSM：
- 预判任务执行过程中可能遇到的各种场景（不同类型输入、中间结果差异、多样化输出）
- 为每个状态定义：State Instruction、Assigned Agent、Condition Verifier、Listeners
- 转移条件采用**自然语言定义**，由 LLM 作为 Condition Verifier 判断是否满足

**FSM 的通用性**：其他多智能体通信结构都是 FSM 的约束特例：
- 线性结构 = 每状态仅一个转移函数的 FSM（无回溯、无验证器）
- 去中心化辩论 = 仅从末尾回溯到开头的 FSM
- 协调器模式 = 共享验证器的 FSM

#### 3. FSM 优化算法（Optimization Phase）

初始设计的 FSM 常存在冗余状态和过长的信息传递链。优化算法：
- 遍历 FSM 中每一对状态
- 使用 LLM 判断两个状态是否可合并
- 消除冗余状态，缩短执行链路
- **无需外部数据**，**无需迭代训练**

#### 4. 部署与执行（Deployment Phase）

从初始状态 $s_0$ 开始：
1. 用户查询 + 当前状态指令 → Task-Solving Agent 执行
2. Agent 输出 → Condition Verifier 检查
3. 满足转移条件 → 转移到下一状态（**可回溯到已出现过的状态**）
4. 转移前将输出保存为 Listeners 的 memory
5. 到达终止状态 $F$ 或超过最大转移次数则停止

**状态回溯**是核心优势：例如在软件开发任务中，产品经理 Agent 发现信息有误，FSM 可回溯到信息收集状态重新执行。

### 损失函数 / 训练策略

MetaAgent 不依赖传统的损失函数或梯度训练。其"优化"是通过 LLM 驱动的状态合并算法实现的：
- 输入：初始构建的 FSM（可能包含冗余状态）
- 操作：对每对状态 $(s_i, s_j)$，LLM 判断功能是否重叠或可合并
- 输出：精简后的 FSM
- 核心思想：减少状态数量 → 缩短执行链路 → 降低错误传播 → 提升鲁棒性

## 实验关键数据

### 主实验

实验覆盖三类任务：文本推理任务、机器学习任务、软件开发任务。

| 任务类型 | 指标 | MetaAgent | 之前SOTA（自动） | 之前SOTA（人工） | 提升 |
|----------|------|-----------|-----------------|-----------------|------|
| Text-Based (Creative Writing + GPQA) | 准确率 | SOTA | 之前 prompt-based SOTA | — | +9% |
| ML Bench | 平均表现 | 97% of best | 其他自动/人工方法 | 最佳人工系统 | 超越所有其他框架 |
| Software Development | Checkpoint 通过数 | +50% | — | 人工设计系统 | +50% |

### 消融实验

| 配置 | 关键指标变化 | 说明 |
|------|-------------|------|
| 去除工具使用 | 性能下降 | Agent 无法与外部环境交互，能力范围受限 |
| 去除 FSM 优化 | 性能下降 | 冗余状态导致信息传递链过长，错误累积 |
| 去除状态回溯 | 性能下降 | 遇到错误时无法修正前序步骤，等效退化为线性结构 |

三项消融均在上述所有任务类型上观察到一致的性能下降，验证了工具集成、FSM 优化和回溯机制的必要性。

### 关键发现

1. **自动设计可逼近人工设计**：在 ML Bench 上达到最佳人工系统 97% 的性能，在软件开发上甚至超越人工系统 50%
2. **FSM 结构优于刚性通信结构**：自定义的条件验证器 + 不受约束的状态转移 提供了最大灵活性
3. **状态回溯是关键差异化因素**：MetaAgent 是唯一具备回溯能力的自动设计框架
4. **无需外部数据的优化是可行的**：LLM 驱动的状态合并算法有效且高效

## 亮点与洞察

1. **FSM 作为统一范式**：将多智能体通信结构统一到 FSM 框架下，揭示了线性/辩论/协调器都是 FSM 的约束特例，这是一个优雅的理论贡献
2. **自然语言转移条件**：用 LLM 而非硬编码字符串匹配来判断状态转移，大幅提升了对复杂场景的适应能力
3. **"元设计"思想**：用一个 Designer LLM 来设计整个多智能体系统，实现了 meta-level 的自动化
4. **轻量优化**：不需要 ADAS/Symbolic Learning 那样的大量迭代和外部数据，降低了部署门槛
5. **工具集成**：代码解释器 + 搜索引擎的组合使 Agent 能处理实际任务

## 局限与展望

1. **依赖 Designer LLM 的能力**：FSM 设计质量取决于 LLM 的规划能力，弱模型可能生成低质量的 FSM
2. **状态合并的粒度控制**：当前合并策略是两两遍历，时间复杂度为 $O(|S|^2)$，状态数多时效率下降
3. **自然语言条件的鲁棒性**：LLM 作为 Condition Verifier 可能在边界情况下判断不准确
4. **缺乏在线自适应学习**：FSM 一旦构建完成即固定，无法根据实际运行反馈动态调整结构
5. **评估任务范围有限**：主要在文本/ML/软件开发三类任务上验证，更多领域（如多模态、科研任务）需进一步验证

## 相关工作与启发

- **MetaGPT / ChatDev**：人工设计的软件开发多智能体系统，功能强但泛化性有限
- **ADAS / Symbolic Learning**：基于自迭代的自动设计方法，但依赖外部数据
- **SPP**：基于 prompt 的自动方法，为每个案例单独设计，不支持工具
- **CodeAct**：将代码作为 Agent 动作的思路对 MetaAgent 的工具集成有启发
- **FSM 在 Agent 中的应用**：之前的工作（AgentLite、StateFlow）用硬编码 FSM 控制 Agent，MetaAgent 实现了 FSM 的自动化设计

## 评分

- **新颖性**: ★★★★☆ — FSM 统一范式和自动化设计是清晰的创新点
- **技术深度**: ★★★☆☆ — 方法本质是 prompt engineering + LLM 判断，缺乏深度技术贡献
- **实验充分性**: ★★★★☆ — 覆盖多类任务，有消融实验，但缺少 LLM backbone 对比
- **实用价值**: ★★★★☆ — 降低了多智能体系统的构建门槛，有开源代码
- **写作质量**: ★★★★☆ — 问题定义清晰，动机充分，表述清楚

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](../../NeurIPS2025/multi_agent/maszero_designing_multiagent_systems_with_zero_supervision.md)
- [\[ACL 2025\] Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](../../ACL2025/multi_agent/beyond_frameworks_multi_agent_collaboration.md)
- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/multi_agent/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [\[ICML 2026\] Securing Multi-Agent Systems Against Corruptions via Node Contribution Backpropagation](../../ICML2026/multi_agent/securing_multi-agent_systems_against_corruptions_via_node_contribution_backpropa.md)
- [\[ICLR 2026\] Stochastic Self-Organization in Multi-Agent Systems](../../ICLR2026/multi_agent/stochastic_self-organization_in_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
