---
title: >-
  [论文解读] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems
description: >-
  [ACL2026][多智能体][失败归因] TraceElephant 主张多智能体失败归因应在开发者真实可见的完整执行轨迹下评测，并提供 220 条失败 trace、责任 agent 与关键失败 step 标注，证明 full observability 能将 step-level 归因从输出-only 的 16% 提升到 28%-30% 以上。
tags:
  - "ACL2026"
  - "多智能体"
  - "失败归因"
  - "多智能体系统"
  - "执行轨迹"
  - "可观测性"
  - "调试基准"
---

# Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems

**会议**: ACL2026  
**arXiv**: [2604.22708](https://arxiv.org/abs/2604.22708)  
**代码**: https://github.com/TraceElephant/TraceElephant  
**领域**: LLM评测 / 多智能体调试  
**关键词**: 失败归因、多智能体系统、执行轨迹、可观测性、调试基准

## 一句话总结
TraceElephant 主张多智能体失败归因应在开发者真实可见的完整执行轨迹下评测，并提供 220 条失败 trace、责任 agent 与关键失败 step 标注，证明 full observability 能将 step-level 归因从输出-only 的 16% 提升到 28%-30% 以上。

## 研究背景与动机
**领域现状**：LLM-based multi-agent systems 已经广泛用于网页检索、软件工程、复杂任务分解和工具调用。但这些系统由多个 agent、prompt、工具和环境交互组成，失败时很难定位责任组件和最早出错步骤。

**现有痛点**：已有失败归因 benchmark 主要使用部分可观测 trace，只记录 agent 输出，不记录输入上下文、系统 prompt、agent 配置、工具日志和环境状态。这样的设置适合黑盒观察，但不符合开发者调试场景，因为开发者通常能访问完整执行日志。

**核心矛盾**：失败归因的目标是提供可修复的诊断，而可修复诊断依赖“这个 agent 当时看到了什么、被如何配置、调用了什么工具、环境返回了什么”。只看输出序列容易把上游信息缺失和下游 agent 自身错误混在一起。

**本文目标**：构建一个 developer-facing failure attribution benchmark，既提供完整执行轨迹，也提供可复现实验环境，使研究者能评估静态 trace 分析、动态 replay 和 counterfactual probing 对归因准确率的影响。

**切入角度**：论文用“seeing the whole elephant”比喻完整可观测性。作者先分析 Who&When 的 184 个失败案例，发现至少 21% 在 output-only logs 下无法可靠归因；随后重新设计 benchmark schema，让输入、输出、工具日志、系统架构都成为一等信息。

**核心 idea**：失败归因不应只问“日志里哪个输出看起来错了”，而应问“在完整执行叙事中，哪个组件在什么输入条件下做出的哪个决策让失败不可避免”。

## 方法详解
TraceElephant 是一个 benchmark，而非新的归因模型。它的贡献在于定义任务、收集完整轨迹、设计标注协议，并系统比较不同 observability 和归因技术。评测单位既可以是真正的多 agent，也可以是单 agent scaffold 中具有独立决策职责的功能组件。

### 整体框架
每个样本由一个失败任务的完整执行 trace、可运行系统环境、责任 agent/component 标签和 decisive failure step 标签组成。step-level attribution 要找到失败变得不可避免的最早步骤；agent-level attribution 要找到该步骤中主要负责的组件。

数据来自三类代表性系统：Captain-Agent、Magentic-One 和 SWE-Agent。它们分别覆盖动态团队组装、中心化多 agent 编排和软件工程单 agent scaffold。任务来源包括 GAIA、AssistantBench 与 SWE-Bench，共采集 380 条 trace，其中 220 条失败 trace 被用于归因标注。

### 关键设计
**1. 完整执行轨迹 schema：把开发者调试真正要看的输入、配置、工具日志都记下来，而不只记输出**

已有失败归因 benchmark 大多只保留 agent 的输出序列，于是「上游传错信息」和「当前组件自己推理错」常被混在一起，根因无从分辨。TraceElephant 重新设计 trace schema，让输入和元数据成为一等信息：trace 级别记录 task_id、task_instruction、system_name、agent_configuration 和 system_architecture；每个 step 同时记录输入字段（step_id、agent_id、agent_name、input_context）和输出字段（output_content、tool_logs），其中 tool_logs 进一步保存工具名、输入参数、输出和执行状态。有了完整上下文，归因方法才能判断一步出错到底是因为这个 agent 收到的信息本就不完整、prompt 构造有误、还是工具返回异常——也就是把「上游传错」和「本地推理错」区分开。

**2. 多轮专家失败归因标注：为每条失败 trace 同时标出责任组件和关键失败步骤**

失败归因不是简单分类，尤其 step-level 要判定「失败从哪一步开始变得不可逆」，单人一次标注很难可靠。TraceElephant 采用多轮专家标注：专家先各自独立给出 agent-level 和 step-level 标签，再对分歧案例联合讨论达成共识。第一轮标注的一致性（Krippendorff's alpha）在 agent-level 为 0.72、step-level 为 0.64——agent 归因相对一致，step 归因明显更难，这个差距本身就量化了「定位最早不可逆失败点」的细粒度难度。

**3. 静态与动态归因评测：既考被动读日志，也考主动重跑验证假设**

开发者真实调试很少停留在读日志，常常会重跑系统、改输入、验证猜想，benchmark 若只给静态日志就脱离了实战。TraceElephant 设两档配置：静态配置只提供完整 trace，方法包括 All-at-Once、Binary Search、Step-by-Step 和 Static Agentic；动态配置额外提供可 replay 的运行环境，Dynamic Agentic 先用静态 agentic 方法提出候选归因，再从候选失败点重跑做 counterfactual check 来验证。把主动调试能力纳入评测，就把基准从被动的日志分类推向了交互式诊断。

### 损失函数 / 训练策略
本文没有训练新模型，评测指标是 agent-level accuracy 和 step-level accuracy。默认静态 agentic 方法基于 mini-SWE-agent 导航 trace 信息；动态方法在静态候选基础上执行 replay 和 counterfactual probing。实验还区分 w/ ground truth 与 w/o ground truth 场景，模拟是否提供正确答案或测试通过信号。

## 实验关键数据

### 主实验
TraceElephant 覆盖三类系统和三类任务来源，共 380 条执行轨迹，其中 220 条失败轨迹被标注。

| System | Task Source | # Traces | # Failed |
|--------|-------------|----------|----------|
| Captain-Agent | GAIA | 126 | 73 |
| Captain-Agent | AssistantBench | 21 | 12 |
| Magentic-One | GAIA | 119 | 74 |
| Magentic-One | AssistantBench | 30 | 17 |
| SWE-Agent | SWE-Bench | 84 | 44 |
| Total | All | 380 | 220 |

不同归因技术的整体平均表现显示，Dynamic Agentic 最好，Static Agentic 是最强静态方法。

| 配置 | Ground Truth | Agent Acc | Step Acc |
|------|--------------|-----------|----------|
| All-at-Once | w/ | 62.2 | 28.1 |
| Binary Search | w/ | 38.9 | 12.9 |
| Step-by-Step | w/ | 60.9 | 16.7 |
| Static Agentic | w/ | 65.9 | 30.3 |
| Dynamic Agentic | w/ | 66.7 | 33.3 |
| All-at-Once | w/o | 58.0 | 22.7 |
| Static Agentic | w/o | 59.1 | 26.1 |
| Dynamic Agentic | w/o | 60.6 | 27.6 |

### 消融实验
可观测性消融直接验证了“输入和元数据不可省”。当退化到 output-only，即 w/o metadata & input，结果与 Who&When 的设置接近，step-level 准确率显著下降。

| Observability 配置 | All-at-Once Agent | All-at-Once Step | Static Agentic Agent | Static Agentic Step |
|--------------------|-------------------|------------------|----------------------|---------------------|
| Full trace | 0.62 | 0.28 | 0.66 | 0.30 |
| w/o metadata | 0.55 | 0.21 | 0.57 | 0.23 |
| w/o input | 0.54 | 0.18 | 0.56 | 0.19 |
| w/o metadata & input | 0.51 | 0.16 | 0.54 | 0.17 |

### 关键发现
- 完整 trace 对 step-level 归因尤其关键。All-at-Once 从 full trace 的 0.28 降到 output-only 的 0.16，Static Agentic 从 0.30 降到 0.17。
- 动态 replay 对 step-level 有额外提升：w/ ground truth 时，Dynamic Agentic step accuracy 33.3%，Static Agentic 为 30.3%，约提升 10%。agent-level 提升较小，因为责任组件通常已可从静态角色和交互结构中推断。
- ground truth 不可用时所有方法下降，说明正确答案或测试信号能显著帮助细粒度归因；但 agentic 方法下降更小，表明主动检索和验证可以部分补偿参考信号缺失。
- 外部环境交互和具体操作 agent 是主要失败来源，几乎超过 50%；planner/orchestrator 也贡献 18%-29% 的失败，常见原因是错误分解、错误分派或协调逻辑传播。

## 亮点与洞察
- 论文最重要的观点是把失败归因放回“开发者调试”语境。只看输出对学术 benchmark 简洁，但对真实修复不够，因为修复需要知道输入、配置和工具上下文。
- TraceElephant 的 schema 很实用，尤其是 input_context 和 tool_logs。很多 agent bug 都发生在 prompt 构造、信息传递和工具返回格式中，这些内容如果不记录，归因会天然模糊。
- 动态配置把 benchmark 从静态数据集变成可实验环境。这点很关键，因为未来归因方法可以执行假设检验、反事实重跑和自动 fault injection，而不只是让 LLM 读长日志。
- 结果说明 step-level attribution 仍然很难。即使 full trace + Dynamic Agentic，也只有 33.3%，这为专门的因果追踪、图推理和结构化 trace summarization 留出了很大空间。

## 局限与展望
- 作者承认 benchmark 只覆盖三个 MAS，虽然三者设计范式多样，但仍不能代表所有未来 agent 架构，尤其是更多异步、长期记忆或真实用户交互系统。
- 本文强调 developer-facing full observability，因此不覆盖完全黑盒平台调试场景。对于只拿到外部输出日志的第三方分析者，TraceElephant 的设定可能过于理想。
- 标注任务本身存在主观性。Krippendorff's alpha 中 step-level 只有 0.64，说明“最早失败点”在复杂 trace 中并不总是唯一明确。
- 当前 Static Agentic 只使用基本工具查看 step I/O。未来可以把 agent 交互图、工具调用图和时间依赖显式建模，或者训练专用小模型来做高效归因。

## 相关工作与启发
- **vs Who&When**: Who&When 主要提供 output-only 部分可观测 trace，适合黑盒归因；TraceElephant 提供完整输入、输出、工具、配置和可运行环境，更贴近开发者调试。
- **vs ECHO / AgenTracer / GraphTracer / FAMAS**: 这些方法关注如何做 agent failure attribution，TraceElephant 更偏 benchmark 基础设施，为这些方法提供更完整、更可复现的评测场景。
- **vs 传统 delta debugging / statistical debugging**: 传统软件调试假设状态离散、执行可追踪、组件更确定；LLM-based MAS 的自然语言状态和非确定性输出使归因更依赖完整上下文和反事实验证。
- **启发**: 未来 agent 平台应默认记录结构化 execution trace，包括输入 prompt、可见历史、工具日志和环境状态。没有可观测性，再强的自动归因方法也会被信息缺失限制。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 任务本身已有相关基准，但 full observability + replayable environment 的开发者视角很有新意。
- 实验充分度: ⭐⭐⭐⭐☆ 系统、任务和配置较丰富，可观测性消融清楚；但系统数量仍有限。
- 写作质量: ⭐⭐⭐⭐☆ 动机和数据 schema 讲得清楚，实验结论有实际调试含义。
- 价值: ⭐⭐⭐⭐⭐ 对多智能体系统可观测性、调试工具和失败归因研究都有基础设施价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation](diversity_collapse_in_multi-agent_llm_systems_structural_coupling_and_collective.md)
- [\[ACL 2026\] OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)
- [\[ACL 2026\] Conjunctive Prompt Attacks in Multi-Agent LLM Systems](conjunctive_prompt_attacks_in_multi-agent_llm_systems.md)
- [\[ACL 2026\] CIA: Inferring the Communication Topology from LLM-based Multi-Agent Systems](cia_inferring_the_communication_topology_from_llm-based_multi-agent_systems.md)
- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](../../ICML2026/multi_agent/maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)

</div>

<!-- RELATED:END -->
