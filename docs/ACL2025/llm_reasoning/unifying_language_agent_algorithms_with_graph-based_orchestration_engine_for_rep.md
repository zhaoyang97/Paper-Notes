---
title: >-
  [论文解读] Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research
description: >-
  [ACL 2025][LLM推理][语言代理框架] 提出 AGORA 框架，通过 DAG 图编排引擎将 CoT、ReAct、ToT、RAP 等 10 种主流 Agent 推理算法统一为可插拔的 Operator 模块，在数学推理和多模态任务上系统比较后发现：简单的 CoT 方法在准确率和成本效益上往往优于复杂算法，而一句提示语改动就能带来 90% 的性能飞跃。
tags:
  - "ACL 2025"
  - "LLM推理"
  - "语言代理框架"
  - "图编排引擎"
  - "Agent推理算法"
  - "标准化评测"
  - "模块化架构"
---

# Unifying Language Agent Algorithms with Graph-based Orchestration Engine for Reproducible Agent Research

**会议**: ACL 2025  
**arXiv**: [2505.24354](https://arxiv.org/abs/2505.24354)  
**代码**: 有 ([https://github.com/om-ai-lab/OmAgent](https://github.com/om-ai-lab/OmAgent))  
**领域**: 其他  
**关键词**: 语言代理框架, 图编排引擎, Agent推理算法, 标准化评测, 模块化架构

## 一句话总结

提出 AGORA 框架，通过 DAG 图编排引擎将 CoT、ReAct、ToT、RAP 等 10 种主流 Agent 推理算法统一为可插拔的 Operator 模块，在数学推理和多模态任务上系统比较后发现：简单的 CoT 方法在准确率和成本效益上往往优于复杂算法，而一句提示语改动就能带来 90% 的性能飞跃。

## 研究背景与动机

**领域现状**：LLM 驱动的语言代理（Language Agent）正快速渗透到各应用领域，Gartner 预测 2025 年将有 33% 的组织部署 LLM 应用。围绕 Agent 开发已涌现出 LangChain、AutoGPT、AgentVerse 等通用框架，以及 ChemCrow（化学）、OS-Copilot（操作系统）等垂直框架；评测方面则有 AgentBench、WebArena 等基准。

**现有痛点**：尽管框架众多，每种 Agent 推理算法（CoT、ReAct、ToT、RAP 等）都有各自独立的实现和评测设置，导致三个严重问题：（1）每接入一种新算法都需要大量定制工程，模块不可复用；（2）不同框架的组件接口不兼容，无法在同一环境中公平比较算法；（3）已有的 Agent Leaderboard（如 Galileo）主要评估 LLM 的工具调用和 API 交互能力，没有同时衡量推理算法本身的有效性。

**核心矛盾**：Agent 算法越来越复杂（从 CoT 到 ToT 到 RAP 再到多模态搜索），但缺乏控制变量的统一评测平台，无法回答"复杂算法是否真的比简单算法好？"这个关键问题。研究者在选择推理策略时缺少数据支撑，往往凭直觉选择更复杂的方法。

**本文目标** （1）如何用统一架构实现并管理多种异构 Agent 算法？（2）如何在控制变量的条件下公平比较这些算法在不同 LLM 上的表现？（3）不同规模模型应该搭配什么样的推理策略？

**切入角度**：作者观察到所有 Agent 算法本质上都可以抽象为"节点 → 边 → 图"的工作流模式——每个推理步骤是节点，步骤间的依赖关系是边，整个算法是一张有向无环图。这意味着用一个通用的图编排引擎就能统一表达所有算法。

**核心 idea**：用 DAG 图编排引擎将 10 种 Agent 推理算法统一为可插拔模块，在同一评测框架下系统揭示"简单即有效"的实践法则。

## 方法详解

### 整体框架

AGORA（Agent Graph-based Orchestration for Reasoning and Assessment）构建于 OmAgent 框架之上，将 Agent 开发分为三层：底层是基于 DAG 的图编排引擎，负责任务调度和执行；中间层是模块化的算法 Operator 库，包含 10 种推理算法的标准化实现；顶层是多用途的客户端接口，支持实时交互、批量评测和命令行调试。输入是用户查询或评测数据集，经过算法选择和工作流编排后，输出推理结果及评测分数。

### 关键设计

1. **DAG 图编排引擎（Graph-based Workflow Orchestration Engine）**:

    - 功能：以有向无环图统一表达所有 Agent 工作流，支持任务的自动调度、异步执行和可视化调试
    - 核心思路：将每个 Agent 工作流建模为 DAG，其中节点分为两类——**简单任务节点**（开发者定义的自定义逻辑，如 LLM 调用、工具执行）和**逻辑任务节点**（内置的控制流原语，如 if-else 分支和 while 循环）。基于 Netflix Conductor 库实现，自动处理节点间的数据流传递和依赖解析。支持异步分布式执行，可并行运行无依赖的任务节点，同时提供工作流的可视化界面追踪每个节点的输入输出状态
    - 设计动机：不同 Agent 算法的控制流差异很大（CoT 是线性链、ToT 是树搜索、GoT 是子图聚合），传统的硬编码 pipeline 无法适配。DAG 抽象可以用统一的方式表达任意拓扑结构，从而让不同算法共享同一个执行引擎，大幅减少重复工程

2. **模块化 Operator 算法库**:

    - 功能：将 10 种主流推理算法封装为标准化、可插拔的 Operator 模块
    - 核心思路：每个 Operator 包含清晰的输入/输出接口规范，内部可调用 LLM 推理、工具执行、记忆读写等公共服务。目前实现了 CoT（零样本/少样本）、SC-CoT（多路径投票）、ToT（BFS/DFS 树搜索）、ReAct（思考-行动-观察循环）、PoT（程序生成+执行）、DnC（分治递归）、GoT（子任务图聚合）、RAP（MCTS 规划搜索）以及两种多模态算法 V*（LLM 引导视觉搜索）和 ZoomEye（树结构缩放探索）。此外对 ReAct 进行了增强——受 Reflexion 启发将 Think 和 Action 拆分为两次独立模型调用形成 ReAct-Pro，对 PoT 将短答和选择题流程合并为"程序执行器 + 答案提取器"两阶段管线，对 GoT 从特定任务（排序、关键词计数）扩展为通用任务处理
    - 设计动机：过去每种算法都有自己独立的代码库，组件不可复用，也无法在同一条件下公平比较。标准化 Operator 接口使得切换推理策略只需更改配置而非重写代码，同时保证评测时除算法本身外其他变量完全一致

3. **多场景客户端评测接口**:

    - 功能：提供三种即插即用的交互/评测客户端，覆盖定性研究到大规模定量评估的完整需求
    - 核心思路：**WebPageClient** 提供 Web 聊天界面用于实时定性测试；**ProgrammaticClient** 读取预定义 JSON 测试文件进行批量自动化评测，支持输出日志和分数统计；**DefaultClient** 提供轻量命令行界面用于开发调试。三种客户端通过统一配置文件切换，共享同一套 Agent 工作流和算法实例。评测框架定义了四个核心指标：准确率（Accuracy）、成本（Cost，美元计）、Token 消耗量、通过率（Pass Rate，有效预测占比）
    - 设计动机：Agent 研究同时需要定性分析（观察行为模式）和定量评估（跑 benchmark），而现有框架通常只支持其中一种。统一的客户端体系让研究者无需改动 Agent 代码就能在不同评测模式间切换

### 实验设置与超参

框架不涉及模型训练。LLM 默认温度为 0。关键算法参数：SC-CoT 温度为 1、采样路径数为 5；ToT 使用 BFS 搜索、b=1、最大深度和最大步数均为 6、评估次数为 3；ReAct-Pro 最大步数为 10。多模态方面，ZoomEye 设最小 patch 为 384、深度限制 5、置信阈值 0~0.4；V* 最大搜索步数限制为 10（避免高分辨率图像搜索时间过长）。

## 实验关键数据

### 主实验 — 数学推理（GSM8K / MATH-500 / AQuA）

| 模型 | 代理算法 | GSM8K (Acc) | AQuA (Acc) | 特点 |
|------|---------|-------------|------------|------|
| Doubao-lite-32k | CoT | 89.31% | — | 成本仅 $0.0558 |
| GPT-4o | CoT | 最优 | 最优 | 但 Agent 框架提升有限 |
| Qwen2.5-72B | CoT | ≈GPT-4o | — | 开源模型超越 GPT-4o |
| Llama-3.3-70B | CoT | ≈GPT-4o | — | 70B 开源可匹敌商业模型 |
| deepseek-r1-1.5B | CoT | — | — | 仅 1.5B 参数超过 InternLM-7B |
| GPT-3.5 | ReAct | 38.13% | 34.25% | 基线 |
| GPT-3.5 | ReAct-Pro | 74.91% | 64.57% | +96% / +88.5% |

### 消融/对比 — ReAct vs ReAct-Pro 改进拆解

| 改动 | AQuA Acc | 相对基线提升 | 说明 |
|------|----------|-------------|------|
| ReAct 基线 | 34.25% | — | Think+Action 合并为一次调用 |
| +拆分 Think/Action | 40.16% | +17.3% | 两次独立调用，各司其职 |
| +加一句提示语 | 64.57% | +88.5% | "You can take as many steps as needed" |

### 主实验 — 多模态推理（MME-RealWorld 2K-4K）

| Agent 算法 | VLM | Score | Pass Rate | Total Tokens |
|------------|-----|-------|-----------|--------------|
| ZoomEye | Qwen2.5-VL-72B | 51.56 | 99.81% | 78.1M |
| ZoomEye | Qwen2.5-VL-7B | 48.06 | 96.50% | 95.9M |
| IO (直接推理) | Qwen2.5-VL-72B | 44.47 | 100% | 6.2M |
| ZoomEye | InternVL2.5-8B | 43.42 | 99.34% | 155.9M |
| IO | Qwen2.5-VL-7B | 42.86 | 100% | 6.2M |
| ZoomEye | Llava-v1.5-7B | 31.60 | 98.86% | 114.4M |
| V* | seal_vqa + vsm | 15.14 | 72.37% | — |

### 关键发现

- **简单算法更稳健**：CoT 在大多数模型上准确率最高、token 最少。ToT 的思考生成和状态评估并未降低推理难度，反而将 token 消耗增加 5-10 倍而准确率提升不超过 1-2%。PoT 依赖代码生成质量，在小模型上反而拖累性能。根本原因在于简单方法减少了"误差累积"——多步推理每步都可能引入错误，单链推理显著降低错误传播风险。
- **提示语是最大杠杆**：ReAct-Pro 仅添加一句"You can take as many steps as needed"就使 AQuA 准确率从 34.25% 飞升到 64.57%，GSM8K 从 38.13% 到 74.91%。这句话根本性改变了模型的行为模式，说明 prompt 设计的边际收益远超算法复杂度的提升。
- **开源模型追平商业模型**：70B 级别开源模型（Llama-3.3-70B、Qwen2.5-72B）在数学推理上已超过 GPT-4o；deepseek-r1-1.5B 以 1.5B 参数超越 InternLM-7B，推理模型在小规模下优势显著。
- **多模态 Agent 大幅提升小模型**：ZoomEye 使 Qwen2.5-VL-7B（48.06）超过了 72B 直接推理（44.47），说明 Agent 工作流能有效弥补小模型能力不足，但代价是 token 消耗增加约 15 倍。
- **V* 泛化性不足**：V* 的 pass rate 仅 72.37%，因其依赖特定训练的 seal_vqa/vsm 模型，在高分辨率图像上搜索时间极长须限制步数，导致目标定位频繁失败。

## 亮点与洞察

- **统一 benchmark 的系统性价值**：在同一框架下控制变量比较 10 种算法 × 多种 LLM 的组合，这种"横向拉齐"的实验设计是本文最大贡献，打破了各 Agent 算法之间"各自为政"的评测困境，为后续研究提供了方法论参考。
- **"简单即有效"的实践法则**：Score vs Cost 分析清晰显示最优性价比集中在 CoT 配置上，这一结论对工业部署具有直接指导意义——不应盲目堆砌算法复杂度，而应从 CoT 出发按需引入复杂性。
- **Prompt engineering 的杠杆效应**：ReAct-Pro 的案例是教科书级示范——通过拆分调用和添加一句话指令就实现了接近翻倍的性能提升，成本几乎为零。这一思路可迁移到任何 Agent 系统的提示优化中。

## 局限与展望

- 评测仅覆盖数学推理和多模态问答两类任务，缺少工具使用、网页交互、代码生成等更复杂的真实世界场景；结论的可推广性有待验证。
- GoT、RAP、DnC 因 token 消耗过高被排除在成本分析之外，无法完整评估这些复杂方法在资源充足时的潜在优势。
- 缺乏对推理延迟/时间的系统测量，成本分析只看了 token 和美元，未考虑时间维度。
- 未探索根据任务特性自适应选择推理策略的方法；未来可结合 meta-learning 或 router 机制动态选择最优算法。
- V* 的评估使用了专用训练模型而非通用 VLM，与其他算法的比较并不完全公平。

## 相关工作与启发

- **vs LangChain/AutoGPT**：它们提供通用 Agent 开发基础设施但不关注推理算法的比较和优化；AGORA 的核心差异在于将算法本身作为可对比的模块对象，而非仅提供开发工具。
- **vs AgentBench/WebArena**：这些 benchmark 侧重于评估 Agent 在特定环境中的端到端表现；AGORA 更关注推理算法本身的对比，两者互补。
- **vs Agent Leaderboard (Galileo)**：其评估的是 LLM 工具调用能力；AGORA 同时评估模型能力和推理策略的交互效应，维度更全面。

## 评分

- **新颖性**: ⭐⭐⭐ — 各组件算法均已有先例，贡献在于统一整合和系统性比较的实验设计
- **实验充分度**: ⭐⭐⭐⭐ — 10 种算法 × 多种模型 × 3 个数学基准 + 1 个多模态基准，发现有洞察力
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，关键发现提炼精准，实践指导性强
- **价值**: ⭐⭐⭐⭐ — 为 Agent 开发者提供了"先 CoT 后复杂"的选型依据和可复用的评测框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Enhancing Video-LLM Reasoning via Agent-of-Thoughts Distillation](../../CVPR2025/llm_reasoning/enhancing_video-llm_reasoning_via_agent-of-thoughts_distillation.md)
- [\[ICLR 2026\] AgentMath: Empowering Mathematical Reasoning for Large Language Models via Tool-Augmented Agent](../../ICLR2026/llm_reasoning/agentmath_empowering_mathematical_reasoning_for_large_language_models_via_tool-a.md)
- [\[ICLR 2026\] Achieving Olympia-Level Geometry Large Language Model Agent via Complexity Boosting Reinforcement Learning](../../ICLR2026/llm_reasoning/achieving_olympia-level_geometry_large_language_model_agent_via_complexity_boost.md)
- [\[ACL 2025\] Beyond the Answer: Advancing Multi-Hop QA with Fine-Grained Graph Reasoning and Evaluation](beyond_the_answer_advancing_multi-hop_qa_with_fine-grained_graph_reasoning_and_e.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](../../NeurIPS2025/llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)

</div>

<!-- RELATED:END -->
