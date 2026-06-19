---
title: >-
  [论文解读] Large Language Models Miss the Multi-Agent Mark
description: >-
  [NeurIPS 2025][多智能体][多智能体系统] Position paper 通过调研 1400+ 篇论文，系统论证当前 MAS LLMs 在四个维度偏离传统 MAS 基础理论——LLM 缺乏原生社会行为、环境设计以 LLM 为中心、缺少异步协调和标准通信协议、涌现行为缺乏量化，指出该领域有忽视 40 年 MAS 成果而重新发明轮子的风险。
tags:
  - "NeurIPS 2025"
  - "多智能体"
  - "多智能体系统"
  - "Position Paper"
  - "社会智能"
  - "异步通信"
  - "涌现行为"
---

# Large Language Models Miss the Multi-Agent Mark

**会议**: NeurIPS 2025  
**arXiv**: [2505.21298](https://arxiv.org/abs/2505.21298)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 多智能体系统, Position Paper, 社会智能, 异步通信, 涌现行为

## 一句话总结

Position paper 通过调研 1400+ 篇论文，系统论证当前 MAS LLMs 在四个维度偏离传统 MAS 基础理论——LLM 缺乏原生社会行为、环境设计以 LLM 为中心、缺少异步协调和标准通信协议、涌现行为缺乏量化，指出该领域有忽视 40 年 MAS 成果而重新发明轮子的风险。

## 研究背景与动机

**领域现状**：MAS LLMs（基于 LLM 的多智能体系统）近年爆发式增长，用于软件工程、多机器人规划、数据分析、科学推理、社会模拟等场景。框架如 AutoGen、MetaAgent、CAMEL 等层出不穷。

**现有痛点**：该领域大量挪用了多智能体系统（MAS）的术语——"agent"、"协作"、"涌现"，却未真正参与 MAS 的基础理论。这导致了一系列根本性问题：LLMs 被单独预训练来回答用户请求，从未被训练与其他 agent 交互；环境设计完全围绕 LLM 构建，忽略其固有局限（幻觉、非确定性、长期记忆缺失）；通信依赖自然语言这一昂贵且含糊的媒介；涌现行为的声称多基于观察描述而非量化度量。

**核心矛盾**：MAS 自 1980-90 年代以来已有 40 多年的理论和实践积累，但当前 MAS LLMs 的建设者在框架设计中几乎不参考这些成果。比如 Google、Anthropic、Microsoft、OpenAI 发布的 agent 框架仅引用 ML 研究，对数十年的 MAS 文献视而不见。

**本文目标** 系统分析 MAS LLMs 与传统 MAS 的差距，识别被忽略的核心问题，并为每个问题提出有建设性的研究方向。

**切入角度**：以 MAS 理论为标尺，从社会智能、环境设计、协调通信、涌现行为四个维度审视当前实践。作者团队横跨牛津、KCL、Sussex 的 CS 和 MAS 研究者，具备深厚的传统 MAS 背景。

**核心 idea**：当前号称"多智能体"的 LLM 系统大多不具备真正的多智能体特性，社区应该回归 MAS 基础理论避免重复造轮子。

## 方法详解

### 整体框架

本文采用批判性调研分析的方法论，对约 112 篇 MAS LLMs 基准/评估论文、1400+ 篇 MAS LLMs 论文、60+ 篇涌现行为论文进行系统调研，分四个维度展开分析并提出研究方向。

### 关键设计

1. **论点一：LLM 缺乏原生社会行为**：

    - 核心论证：MAS 中的智能 agent 需要三种能力——反应性（reactivity）、主动性（proactiveness）和社会性（social ability）。LLM 具备前两者，但社会性完全靠 prompt 注入或 orchestrator 强加，而非训练获得
    - 关键证据：Cemri et al. 发现 37% 的 MAS LLMs 失败来自 agent 间对齐和协调问题；LLM 在 Theory of Mind 基准上表现差；多数 MAS LLMs 实际上退化为 ensemble（多数投票）而非真正协作
    - 研究方向：在预训练阶段就融入多 agent 合作/竞争场景，利用 TextGrad 等基于文本反馈的训练方法让 agent 在交互中学习社会行为

2. **论点二：环境设计以 LLM 为中心**：

    - 核心论证：传统 MAS 环境设计不假设 agent 架构，但当前 MAS LLMs 假设所有 agent 都是 LLM + 自然语言通信。这忽略了 LLM 的三大固有缺陷：非确定性（温度设为 0 也不完全确定）、幻觉（偏离预设身份/角色）、长期记忆缺失
    - 关键数据：调研 112 篇 MAS LLMs 论文发现大多数运作在部分可观察、确定性假设、离散时间、文本表示的环境中
    - 典型案例：CAMEL 中两个 LLM 会不自觉交换角色、陷入无限消息循环；MetaAgent 中 LLM 会产生能力幻觉、偏离预设身份
    - 研究方向：设计多模态环境减少自然语言中介；用结构化格式替代自由文本；集成形式化规划器或神经符号方法

3. **论点三：缺少异步协调和标准通信**：

    - 核心论证：异步性是真正 MAS 的核心特征，但调研 1400+ 篇 MAS LLMs 论文仅发现 22 篇显式涉及异步交互。自然语言通信昂贵且含糊，KQML、FIPA ACL 等已有的结构化 agent 通信标准被完全忽视
    - AutoGen 案例：虽然支持异步 API，但开发者必须为每个动作和事件手动定义异步调用，用同步语言做异步编程极易引入 bug
    - 研究方向：框架应默认异步、同步为例外；借鉴 Petri 网等并发模型分析 MAS LLMs 的可达性和有界性；建立类似 Google A2A 的标准 agent 通信协议

### 论点四：涌现行为缺乏量化

调研 60+ 篇声称研究涌现行为的论文，发现绝大多数仅做定性观察而无量化指标。例如 Generative Agents（Stanford 小镇模拟）中涌现概念从未被形式化定义，结果主要是"让系统跑一段时间然后观察有趣行为"。Project Sid（Minecraft 文明模拟）声称 LLM 可以实现 AI 文明里程碑，但证据纯粹是描述性的。研究方向：建立可证伪的涌现行为定义，区分弱涌现（可从底层推导）和强涌现（需要新假设），借鉴经济学和系统理论中的成熟定义。

## 实验关键数据

### 调研统计

| 调研维度 | 数据量 | 关键发现 |
|---------|--------|--------|
| 环境特征分析 | 112 篇 MAS LLMs 论文 | 大多使用部分可观察+确定性假设+文本环境 |
| 异步性调研 | 1400+ 篇 | 仅 22 篇显式支持异步交互 |
| 涌现行为论文 | 60+ 篇 | 极少数定义了可测量的量化指标 |
| 失败分析 | 引用 Cemri et al. | 37% 失败来自 agent 间协调问题 |

### 关键发现

- **MAS LLMs 的"多智能体"名不副实**：多数系统本质上是由 orchestrator 控制的 LLM 流水线或 ensemble
- **自然语言通信的不可扩展性**：自然语言的歧义性和高计算成本使得大规模 agent 网络通信不可持续
- **涌现 vs 幻觉的混淆**：当前缺乏区分"真正涌现行为"和"LLM 幻觉/巧合输出"的方法论

## 亮点与洞察

- **数据驱动的批判视角**：不只是观点文章，而是基于 1400+ 篇论文的系统调研，数据说服力强
- **桥接两个社区**：将 MAS 社区 40 年积累的理论（KQML、BDI 架构、并发系统理论）与 LLM Agent 社区连接，指出了大量可复用的理论工具
- **每个批判都伴随建设性方向**：不只说问题，还给出了具体可行的研究路径。比如"用 Petri 网建模 MAS LLMs 的状态转换"就是一个很好的具体切入点

## 局限与展望

- **立场偏向传统 MAS**：可能低估了 LLM 在某些场景中"无需严格 MAS 理论"就能有效运作的实践价值
- **缺乏实证对比**：未构建一个"遵循 MAS 原则的 LLM 系统" vs "现有系统"的实验对比来验证其论点
- **对工业实践的影响有限**：工业界更关心"能用就行"，而非理论纯粹性
- **涌现行为的定义难题**：虽然批评了现有研究缺乏量化，但自身也未给出一个可操作的涌现指标定义

## 相关工作与启发

- **vs CAMEL/AutoGen/MetaAgent**：本文批判性分析了这些框架的 MAS 不足之处，但也承认它们在实际应用中的工程价值
- **vs 传统 MAS 教科书（Wooldridge）**：本文大量引用 Wooldridge 的经典定义作为理论基准
- **值得注意的新方向**：Google A2A 协议、MCP（Model Context Protocol）等正在尝试标准化 agent 间通信，与本文的研究方向呼应

## 评分

- 新颖性: ⭐⭐⭐⭐ 视角独特，系统性地从传统 MAS 理论审视 LLM Agent 生态
- 实验充分度: ⭐⭐⭐ 调研数据量大但缺乏实证实验验证论点
- 写作质量: ⭐⭐⭐⭐⭐ 论证结构清晰，每个论点都有数据支撑和替代观点回应
- 价值: ⭐⭐⭐⭐ 对于 LLM Agent 研究者是重要的警示和理论补充

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)
- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](../../ACL2026/multi_agent/agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](../../AAAI2026/multi_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[ICLR 2026\] Cache-to-Cache: Direct Semantic Communication Between Large Language Models](../../ICLR2026/multi_agent/cache-to-cache_direct_semantic_communication_between_large_language_models.md)
- [\[AAAI 2026\] AgentODRL: A Large Language Model-based Multi-agent System for ODRL Generation](../../AAAI2026/multi_agent/agentodrl_a_large_language_model-based_multi-agent_system_fo.md)

</div>

<!-- RELATED:END -->
