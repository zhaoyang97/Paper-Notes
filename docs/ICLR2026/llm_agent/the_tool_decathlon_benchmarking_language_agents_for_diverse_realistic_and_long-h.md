---
title: >-
  [论文解读] The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution
description: >-
  [ICLR 2026][LLM Agent][language agent] 提出 Toolathlon，一个覆盖 32 个软件应用、604 个工具和 108 个任务的语言 Agent 基准，强调真实多样的环境状态和长程多步交互（平均约 20 轮工具调用），最强模型 Claude-4.5-Sonnet 仅达 38.6% 成功率。
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "language agent"
  - "benchmark"
  - "MCP"
  - "tool calling"
  - "long-horizon"
  - "多应用交互"
  - "执行式评估"
---

# The Tool Decathlon: Benchmarking Language Agents for Diverse, Realistic, and Long-Horizon Task Execution

**会议**: ICLR 2026  
**arXiv**: [2510.25726](https://arxiv.org/abs/2510.25726)  
**代码**: 待确认 (基于 MCP 服务器)  
**领域**: LLM Agent / Benchmark / 工具使用  
**关键词**: language agent, benchmark, MCP, tool calling, long-horizon, 多应用交互, 执行式评估

## 一句话总结
提出 Toolathlon，一个覆盖 32 个软件应用、604 个工具和 108 个任务的语言 Agent 基准，强调真实多样的环境状态和长程多步交互（平均约 20 轮工具调用），最强模型 Claude-4.5-Sonnet 仅达 38.6% 成功率。

## 研究背景与动机
**领域现状**：语言 Agent 需要在真实世界中完成跨应用、多步骤的复杂工作流，如管理日历邮件联动、监控数据库生成报告等。这要求 Agent 具备工具发现、多轮推理、状态追踪和跨系统协调的综合能力。

**现有痛点**：现有 Agent 基准存在三大不足——(1) 应用领域狭窄（多聚焦于单一工具或 API），(2) 任务过于简化（一两步调用即可完成），(3) 环境状态不真实（使用空白或极简的初始状态，而非真实软件中的复杂数据），无法充分评估 Agent 的实际部署能力。

**核心矛盾**：研究者需要可靠评估 Agent 的真实世界表现，但现有基准的"玩具级"任务与真实工作流之间存在巨大鸿沟，导致基准得分与实际能力脱节——Agent 在简单基准上表现良好，但在真实场景中频繁失败。

**MCP 的机遇**：Model Context Protocol (MCP) 的出现为构建标准化工具接口提供了基础设施。Toolathlon 基于高质量的 MCP 服务器构建工具层，部分由团队自行实现或修订，确保接口质量和一致性。

**评估可靠性**：现有基准多依赖 LLM 判断或字符串匹配来评估 Agent 输出，误差大且不可复现。需要基于程序执行的严格验证机制，确保每个任务的完成与否有确定性判定。

**核心 idea**：构建首个同时满足"多样应用覆盖 + 真实环境状态 + 长程复杂任务 + 执行式验证"四个维度的 Agent 综合基准。

## 方法详解

### 整体框架
Toolathlon 是一个语言 Agent 基准，把 32 个真实软件应用通过 MCP 服务器统一暴露成 604 个工具，让 Agent 在带有真实初始数据的环境里完成 108 个长程任务。整个基准由四块拼成：多样的应用工具层、真实的环境状态、平均约 20 轮调用的复杂任务，以及基于程序执行的确定性验证。

### 关键设计

这四个设计点不是流水线的先后阶段，而是基准在四个维度上同时"加难度"：工具层让选工具变难、环境状态让起点变脏、任务设计让链路变长、执行式评估让判分变严。

**1. 应用与工具层：用 MCP 把"工具发现"本身做成挑战**

单工具基准里 Agent 不需要挑工具，但真实部署中"先在一堆工具里找对的"本身就是难点。Toolathlon 横跨 32 个软件应用，从日常办公（Google Calendar、Notion、Gmail）一直到专业工具（WooCommerce 电商、Kubernetes 容器编排、BigQuery 数据分析），合计暴露 604 个工具/API。工具层统一架在 MCP（Model Context Protocol）服务器上，团队对部分现成的 MCP 服务器做了修订或重新实现，以保证接口质量和跨应用的行为一致性。如此一来，Agent 每一步都要在数百个工具里定位正确的工具组合，而不是面对一两个预先选好的 API——这把"工具检索"从送分题变回真实障碍。

**2. 真实环境状态：让 Agent 在已有的脏数据里干活**

很多基准虽然工具可用，但初始环境是空白的，Agent 只需从零创建，难度被人为拉低。Toolathlon 反过来为每个任务注入来自真实软件的初始状态——Canvas 课程里有数十名学生的真实选课与成绩数据，财务表格里有现成的账目，电商系统里有历史订单。这样 Agent 必须先在已有的复杂数据中检索、筛选、关联，再动手修改，更贴近一个新员工接手既有系统时的处境，而非在干净沙盒里做演示；很多失败正是出在"没看清已有状态就贸然写入"。

**3. 任务设计与长程复杂度：用约 20 轮调用维持多步规划压力**

短任务一两步就结束，无法暴露规划和上下文维护的弱点。Toolathlon 的 108 个任务全部手动编写或从真实工作流提炼，每个都要求 Agent 跨多个应用协作，平均需要约 20 轮工具调用才能完成，属于典型的长程（long-horizon）任务。任务类型覆盖信息检索聚合、跨系统数据同步、条件判断与流程执行、报告生成等，要求 Agent 在几十步交互中持续追踪状态、按条件分支、并从中途失误里恢复——任何一步走偏都会在后续累积放大，这正是短任务测不出来的能力。

**4. 执行式评估：靠任务后的状态检查给出确定性判定**

用 LLM 判分或字符串匹配会把噪声引进评估本身，结果难以复现。Toolathlon 给每个任务配一个专用评估脚本，直接检查任务结束后的系统状态变化——数据库记录、文件内容、API 状态是否如预期改变——据此严格判定是否完成。整套评估是确定性的，同时覆盖正确性（是否达成目标）和完整性（是否遗漏步骤）两个维度，部分任务还设有中间检查点来定位 Agent 在哪一步掉链子。相比"读 Agent 的最终回复来打分"，这种"看世界状态变没变"的判分既客观又可复现。

## 实验关键数据

### 主实验：模型成功率对比

| 模型 | 成功率(%) | 平均工具调用轮数 | 类型 | 说明 |
|------|----------|---------------|------|------|
| Claude-4.5-Sonnet | **38.6** | 20.2 | 闭源 | 最强模型 |
| GPT-4o / GPT-5 系列 | 25-35 (推测) | ~20 | 闭源 | 中等表现 |
| DeepSeek-V3.2-Exp | **20.1** | ~20 | 开源最强 | 开源权重最佳 |
| 其他开源模型 | <20 | 变化大 | 开源 | 普遍不足 |

### 不同应用类别的表现差异

| 应用类别 | 典型应用 | 模型表现趋势 | 难点分析 |
|---------|---------|------------|---------|
| 日常办公 | Calendar, Gmail | 相对较好 | API 结构化程度高 |
| 项目管理 | Notion, Canvas | 中等 | 需理解复杂数据结构 |
| 开发运维 | Kubernetes, Git | 较差 | 要求专业领域知识 |
| 数据分析 | BigQuery, Sheets | 较差 | 需要多步数据处理 |
| 电子商务 | WooCommerce | 较差 | 业务逻辑复杂 |

### 关键发现
- 即使最强的 Claude-4.5-Sonnet 也只达到 38.6% 的成功率，说明当前 Agent 在处理真实、长程、多应用任务时仍有巨大不足。
- 开源模型和闭源模型之间存在约 18 个百分点的差距（38.6% vs 20.1%），工具使用能力仍是开源模型的明显短板。
- 任务失败的主要原因包括：工具选择错误、长上下文中丢失关键信息、跨应用数据格式适配失败、缺乏错误恢复能力。
- 专业领域工具（如 Kubernetes、BigQuery）的成功率显著低于日常办公工具，说明领域知识是当前 Agent 的重要瓶颈。
- MCP 服务器的标准化接口降低了工具集成的复杂度，但 Agent 在"理解工具能力边界"方面仍有欠缺。

## 亮点与洞察
- **"真实环境状态"的设计哲学**：Toolathlon 不仅提供工具接口，还提供复杂的真实数据环境。这迫使 Agent 学会"在混乱中工作"——类似人类新员工面对已有数据和系统时的挑战。这比空白环境测试更能反映实际部署中的困难。
- **MCP 作为基础设施**：以 MCP 服务器为工具层的设计具有前瞻性——随着 MCP 生态系统的增长，Toolathlon 可以自然扩展到更多应用和工具。
- **长程任务的评估价值**：平均 20 轮调用的任务长度有效测试了 Agent 的规划、上下文维护和错误恢复能力，这些在短任务中无法体现。

## 局限与展望
- **任务规模有限**：108 个任务虽然手动高质量构建，但数量相对较少，可能不足以评估模型在更广泛场景中的泛化能力。
- **静态环境**：任务环境在评估开始时固定，不涉及动态变化的环境（如实时通知、并发用户操作），这在真实世界中很常见。
- **人工标注成本高**：每个任务需要手动编写评估脚本，扩展到更多任务/应用的成本较高。
- **缺少交互式场景**：所有任务都是单向的——Agent 执行任务，没有用户在中途提供反馈或修改需求的场景。
- **模型调用成本**：20 轮平均调用意味着大量的 API 开销，这限制了对更多模型进行全面评估的可行性。
- **地理与语言局限**：当前应用和任务以英语为主，面向北美/全球化的软件工具，对非英语环境和本地化应用的覆盖不足。
- **安全性维度缺失**：基准主要评估任务完成率，未系统评估 Agent 在工具使用中的安全性（如数据泄露、权限越界、误操作风险）。

## 相关工作与对比
- **vs AgentBench (Liu et al., 2023)**：AgentBench 覆盖了操作系统、数据库、Web 等场景，但任务长度和环境状态复杂度远不如 Toolathlon。Toolathlon 在应用多样性和环境真实性上是显著升级。
- **vs ToolBench (Qin et al., 2024)**：ToolBench 提供了大规模的 API 集合和自动化任务生成，但任务多为单步或少步调用，环境状态简单。Toolathlon 强调长程多步和真实状态。
- **vs SWE-bench (Jimenez et al., 2024)**：SWE-bench 聚焦软件工程单一领域（代码修复），Toolathlon 则覆盖 32 个不同领域的应用，测试 Agent 的通用工具使用能力。
- **vs GAIA (Mialon et al., 2023)**：GAIA 测试通用 AI 助手的网络搜索和推理能力，任务较短。Toolathlon 的 MCP 框架使其更接近真实的应用集成场景。
- **vs τ-bench (Yao et al., 2024)**：τ-bench 同样关注工具使用，但任务设计更偏向于推理能力验证。Toolathlon 更强调应用覆盖的广度和环境状态的真实性。
- **vs AgentDojo (Debenedetti et al., 2024)**：AgentDojo 侧重安全性（prompt injection 防御），Toolathlon 侧重功能性（任务完成率），二者评估视角互补。

## 评分
- 新颖性: ⭐⭐⭐⭐ MCP + 真实环境状态 + 长程任务的组合是 Agent 评估的新范式
- 实验充分度: ⭐⭐⭐⭐⭐ SOTA 模型全面测试，32 应用 × 604 工具 × 108 任务
- 写作质量: ⭐⭐⭐⭐ 基准描述清晰，但部分细节较密集
- 价值: ⭐⭐⭐⭐⭐ 填补了 Agent 评估中"真实性"的关键空缺，有望成为社区标准基准

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LongHorizonUI: A Unified Framework for Robust Long-Horizon Task Automation of GUI Agent](longhorizonui_a_unified_framework_for_robust_long-horizon_task_automation_of_gui.md)
- [\[CVPR 2026\] WebGym: Scaling Training Environments for Long-Horizon Visual Web Agents with Realistic Tasks](../../CVPR2026/llm_agent/webgym_scaling_training_environments_for_long-horizon_visual_web_agents_with_rea.md)
- [\[ICLR 2026\] MEM1: Learning to Synergize Memory and Reasoning for Efficient Long-Horizon Agents](mem1_learning_to_synergize_memory_and_reasoning_for_efficient_long-horizon_agent.md)
- [\[ICLR 2026\] Benchmarking LLM Tool-Use in the Wild](benchmarking_llm_tool-use_in_the_wild.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)

</div>

<!-- RELATED:END -->
