---
title: >-
  [论文解读] MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents
description: >-
  [ACL 2025][LLM Agent][multi-agent] 提出 MultiAgentBench 基准测试和 MARBLE 框架，系统评估 LLM 多智能体系统在协作与竞争场景中的表现，包含 6 种交互场景（研究、Minecraft、数据库、编程、讨价还价、狼人杀），引入基于里程碑的 KPI 指标和协调评分，发现 gpt-4o-mini 整体任务分最高、图结构协调协议在研究场景中表现最佳、认知规划可提升里程碑达成率 3%。
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "multi-agent"
  - "benchmark"
  - "collaboration"
  - "competition"
  - "coordination protocol"
  - "emergent behavior"
---

# MultiAgentBench: Evaluating the Collaboration and Competition of LLM Agents

## 基本信息

**会议**: ACL 2025  
**代码**: [MultiagentBench/MARBLE](https://github.com/MultiagentBench/MARBLE)  
**机构**: University of Illinois Urbana-Champaign  
**领域**: LLM Agent / 多智能体系统  
**关键词**: multi-agent, benchmark, collaboration, competition, coordination protocol, emergent behavior  

## 一句话总结

提出 MultiAgentBench 基准测试和 MARBLE 框架，系统评估 LLM 多智能体系统在协作与竞争场景中的表现，包含 6 种交互场景（研究、Minecraft、数据库、编程、讨价还价、狼人杀），引入基于里程碑的 KPI 指标和协调评分，发现 gpt-4o-mini 整体任务分最高、图结构协调协议在研究场景中表现最佳、认知规划可提升里程碑达成率 3%。

## 研究背景与动机

- **单智能体评估的局限**：现有基准如 AgentBench、GAIA、ToolBench 等主要评估单个 LLM 智能体的推理和生成能力，忽略了多智能体交互中的协调和竞争动态
- **多智能体系统的兴起**：LLM 多智能体系统在软件开发（MetaGPT、ChatDev）、科学研究、游戏等领域展现出强大潜力，但缺乏系统的评估标准
- **评估维度不足**：不仅需要评估任务完成度，还需衡量协调质量、沟通效率和规划能力
- **核心目标**：构建涵盖协作和竞争场景的全面多智能体评估框架

## 方法详解

### 整体框架：MARBLE (Multi-agent cooRdination Backbone with LLM Engine)

MARBLE 由四个核心模块组成：

#### 1. Agent Graph Module

将智能体关系建模为图 $G = (\mathcal{A}, E)$：
- $\mathcal{A} = \{a_1, a_2, \dots, a_n\}$：智能体集合
- 每条边 $(a_i, r, a_j)$ 表示关系类型：collaborates（协作）、supervises（监督）、negotiates（谈判）
- 通信和协调仅在有显式关系的智能体间进行

#### 2. Cognitive Module（认知模块）

- 维护每个智能体的内部状态：角色设定、智能体间关系、推理策略
- 融合心智理论（Theory of Mind）和社会智能
- 支持 CoT、ReACT 等推理策略
- 模拟人类基于社交线索持续更新心理模型的过程

#### 3. Coordination Engine（协调引擎）

支持四种协调协议：

| 协议 | 类型 | 特点 |
|------|------|------|
| **Star** | 中心化 | 单一规划者分配任务，强监督但可扩展性有限 |
| **Tree** | 中心化 | 层次结构，顶层规划者委派给下级规划者 |
| **Graph-Mesh** | 去中心化 | 智能体直接通信，并发规划和分布式决策 |
| **Chain** | 去中心化 | 顺序传递决策，适合有依赖关系的任务 |

#### 4. Planner 设计

四种规划策略：

1. **Vanilla Prompting**：直接零样本指令生成计划
2. **Chain-of-Thought (CoT)**：逐步推理，输入任务+智能体档案+历史
3. **Group Discussion**：多智能体共享见解和约束，协作审议
4. **Cognitive Self-Evolving Planning**：生成预期结果，存入记忆，与实际表现对比，迭代改进（类似 Reflexion）

### 基准任务设计

#### 共同目标场景（协作）

| 场景 | 描述 | 规模 |
|------|------|------|
| **Research** | 多agent合作撰写研究提案 | 100 个测试用例 |
| **Minecraft** | 协作建造结构 | 100 个测试用例 |
| **Database** | 5个agent诊断不同根因 | 100 个测试用例 |
| **Coding** | 集体编程和模块开发 | 100 个测试用例 |

#### 对立目标场景（竞争）

| 场景 | 描述 |
|------|------|
| **Werewolf（狼人杀）** | 两组对抗，涉及欺骗策略 |
| **Bargaining（讨价还价）** | 资源谈判，最大化个人收益 |

### 评估指标

#### 任务完成指标

- **KPI**：基于里程碑的关键绩效指标，$\text{KPI}_{\text{overall}} = \frac{1}{NM}\sum_{j=1}^N n_j$
- **Task Score (TS)**：最终输出质量评分（LLM 评分或规则评估）

#### 协调指标

- **Communication Score ($C_{\text{score}}$)**：基于 LLM 评估沟通质量（5分制）
- **Planning Score ($P_{\text{score}}$)**：评估任务组织、角色维护和策略调整（5分制）
- **Coordination Score (CS)**：上述两项的平均值

## 实验

### 实验设置

- **模型**：Meta-Llama-3.1-8B、Meta-Llama-3.1-70B、Meta-Llama-3.3-70B、GPT-3.5-turbo、GPT-4o-mini
- **参数**：max_token_num=1024，temperature=0.7，top_p=1.0
- **迭代**：研究任务 5 轮，Minecraft 20 轮，通信最大 5 轮
- **默认协议**：Graph-Mesh

### 主实验一：模型性能对比（表1）

| 模型 | Research TS | Minecraft TS | Database TS | Coding TS | Bargaining TS | Werewolf TS |
|------|-----------|-------------|-----------|---------|-------------|-----------|
| Llama-3.1-8B | 80.87 | 6.12 | 34.00 | 59.90 | 72.81 | 12.64 |
| Llama-3.1-70B | 80.80 | 0.21 | 53.00 | 62.10 | 72.13 | 19.82 |
| Llama-3.3-70B | 80.00 | 9.15 | 28.50 | 56.60 | 73.15 | 36.33 |
| GPT-3.5-turbo | 70.20 | 5.05 | 45.00 | 55.50 | 71.67 | 15.69 |
| **GPT-4o-mini** | **84.13** | **33.60** | 45.00 | **65.10** | **74.47** | 14.06 |

**关键发现**：
1. **GPT-4o-mini 任务分最高**：在 Research、Minecraft、Coding、Bargaining 中领先
2. **协调分≠任务分**：Llama-3.1-70B 在 Minecraft 协调分高达 75.00，但任务分仅 0.21
3. **能力因任务而异**：Llama-3.3-70B 在 Werewolf 中 TS 最高（36.33），但在其他任务中不突出

### 主实验二：协调协议对比

- **Graph 协议**在研究场景中表现最佳（最高任务分和规划效率）
- **Tree 协议**表现最差（高 token 消耗，最低任务和协调分）
- **Star 和 Graph** 任务分相近

### 规划策略对比

- **Cognitive Evolving Planning** 协调分最优，任务分与 CoT 相当
- **Group Discussion** 反而表现最差——过大的规划组反而阻碍效率（类似现实中的大型组织问题）

### 消融实验

**迭代次数**：
- 任务分和协调分在 1→7 轮间上升，10 轮时骤降，20 轮时任务分恢复但协调分不变
- 过多迭代可能导致协调退化（通信开销或指令冲突）

**智能体数量**：
- KPI 随智能体数量增加而降低（更复杂的协调）
- 协调分在 1→3 个智能体时显著提升，之后趋于平缓
- 任务分增长更为渐进

### 涌现行为分析

三种关键涌现模式：
1. **策略性信息共享**：智能体选择性披露关键信息（如狼人杀中预言家隐瞒检验结果）
2. **信任极化协作**：角色身份驱动协作分裂，过度怀疑的村民可能攻击己方
3. **角色驱动策略迭代**：角色（如预言家、女巫）在游戏过程中调整策略

## 亮点与洞察

1. **首个系统性多智能体评估框架**：覆盖协作+竞争、6 种场景、多种协调协议，填补了评估空白
2. **里程碑KPI创新**：超越简单的任务成功/失败二分，追踪中间进展和个体贡献
3. **反直觉发现**：Group Discussion 最差、协调分高不等于任务分高、更多智能体不一定更好
4. **涌现行为的发现**：LLM 智能体在信息不对称和角色冲突下展现出类似人类的社会行为模式
5. **模型能力仍是关键**：协调改善无法弥补基础能力的不足

## 局限性

1. **场景覆盖有限**：未涵盖开放世界、任务导向对话等复杂场景
2. **模型覆盖不全**：未包含 DeepSeek 等最新模型
3. **消融不够深入**：未充分研究记忆机制（长期/短期/共享记忆）和不同工作流方法
4. **竞争机制简单**：未涵盖多方谈判、重复博弈、随机因素等复杂情况
5. **评估依赖 LLM**：KPI 里程碑检测和协调评分依赖 LLM 判断，可能存在偏差

## 相关工作

- **多智能体系统**：MetaGPT (Hong et al., 2024)、AgentVerse (Chen et al., 2023)、ChatDev (Li et al., 2023)
- **多智能体协作**：认知扩展 (Zhuge et al.)、群体扩展 (Qian et al., 2024)
- **游戏中的智能体**：GameNGen (Valevski et al., 2024)、CUISINEWORLD (Gong et al., 2023)、Voyager (Wang et al., 2023)
- **单智能体基准**：AgentBench (Liu et al., 2023)、GAIA (Mialon et al., 2023)

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**：首个系统性多智能体评估基准，设计全面（+1）
- **实验多样性**：6 种场景 × 5 种模型 × 4 种协调协议 × 4 种规划策略（+0.5）
- **洞察深度**：涌现行为分析和反直觉发现增加了学术贡献（+0.5）
- **实用价值**：开源框架可供社区使用（+0.5）
- **扣分**：评估依赖 LLM 判断的可靠性存疑、模型覆盖不够广、某些任务设计偏简单（-1）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [\[ACL 2025\] The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs](the_behavior_gap_evaluating_zero-shot_llm_agents_in_complex_task-oriented_dialog.md)
- [\[NeurIPS 2025\] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](../../NeurIPS2025/llm_agent/trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [\[ACL 2025\] Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration](dpt_agent_dual_process.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)

</div>

<!-- RELATED:END -->
