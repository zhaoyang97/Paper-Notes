---
title: >-
  [论文解读] AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML
description: >-
  [ICML 2025][多智能体][AutoML] 本文提出 AutoML-Agent，一个基于多智能体 LLM 协作的全流水线 AutoML 框架，通过检索增强规划策略（Retrieval-Augmented Planning）扩大搜索空间、将任务分解为并行执行的子任务由专业化 Agent 分别完成、并引入多阶段验证机制保障代码生成质量，在 7 类任务 14 个数据集上实现了更高的自动化成功率和模型性能。
tags:
  - "ICML 2025"
  - "多智能体"
  - "AutoML"
  - "多智能体框架"
  - "LLM Agent"
  - "检索增强规划"
  - "全流水线自动化"
---

# AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML

**会议**: ICML 2025  
**arXiv**: [2410.02958](https://arxiv.org/abs/2410.02958)  
**代码**: [https://deepauto-ai.github.io/automl-agent](https://deepauto-ai.github.io/automl-agent)  
**领域**: Agent  
**关键词**: AutoML, 多智能体框架, LLM Agent, 检索增强规划, 全流水线自动化

## 一句话总结

本文提出 AutoML-Agent，一个基于多智能体 LLM 协作的全流水线 AutoML 框架，通过检索增强规划策略（Retrieval-Augmented Planning）扩大搜索空间、将任务分解为并行执行的子任务由专业化 Agent 分别完成、并引入多阶段验证机制保障代码生成质量，在 7 类任务 14 个数据集上实现了更高的自动化成功率和模型性能。

## 研究背景与动机

**领域现状**：自动化机器学习（AutoML）旨在自动化 AI 开发流水线中的关键步骤，如最优模型搜索和超参数调优。传统 AutoML 工具（如 Auto-sklearn、AutoGluon、FLAML）虽然在特定环节表现出色，但通常需要用户具备较强的技术专长来配置复杂的工具链和参数空间，这一过程耗时且需要大量人力投入。

**现有痛点**：近年来，研究者开始利用大语言模型（LLM）的自然语言接口来降低 AutoML 的使用门槛，让非专业用户也能构建数据驱动的解决方案。然而，现有的 LLM-based AutoML 方法存在两个关键缺陷：(a) **覆盖面窄**——它们通常只针对 AI 开发流水线中的某个特定环节（如仅做模型选择，或仅做超参调优），不能覆盖从数据获取到模型部署的全流程；(b) **LLM 能力利用不充分**——单一 LLM Agent 难以同时处理数据预处理、模型架构设计、训练策略选择等多种异质性任务。

**核心矛盾**：全流水线 AutoML 要求系统具备多种异质能力（数据工程、模型设计、训练优化、部署），而单一 Agent 架构的搜索空间有限、规划策略单一，只能生成一个固定的计划方案，缺乏对解空间的充分探索，且无法并行处理相互独立的子任务。

**本文目标** (a) 如何让 LLM 框架覆盖从数据获取到模型部署的完整 AutoML 流水线？(b) 如何增强规划阶段的探索能力以搜索更优方案？(c) 如何利用多 Agent 并行化加速子任务求解？(d) 如何保证生成代码的正确性和可部署性？

**切入角度**：作者观察到 AutoML 流水线中的各子任务具有天然的可分解性（数据预处理与模型设计相对独立），可以为每个子任务设计专业化 Agent 并行执行；同时，借鉴 RAG（检索增强生成）的思想，通过检索相关的历史成功方案来扩大规划空间。

**核心 idea**：用多智能体协作分治全流水线 AutoML，用检索增强规划扩大搜索空间，用多阶段验证保障代码质量。

## 方法详解

### 整体框架

AutoML-Agent 接收用户的自然语言任务描述作为输入（如"用这份 CSV 数据训练一个分类器"），经过完整的 AutoML 流水线处理，最终输出可部署的模型。整个框架由三个核心阶段组成：

1. **规划阶段（Planning）**：Manager Agent 分析用户需求，利用检索增强策略生成多个候选执行计划，每个计划被分解为若干子任务。
2. **执行阶段（Execution）**：每个子任务被分配给对应的专业化 Agent（如 Data Agent 负责数据预处理，Model Agent 负责神经网络设计），这些 Agent 并行执行各自的子任务。
3. **验证阶段（Verification）**：通过多阶段验证机制检查各 Agent 的执行结果，发现问题后引导代码生成 LLM 修正实现，确保最终方案的可执行性和正确性。

### 关键设计

1. **检索增强规划策略（Retrieval-Augmented Planning）**

    - 功能：在规划阶段生成多个候选方案，而非只制定单一计划，以增强对解空间的探索。
    - 核心思路：与传统 AutoML 方法只让 LLM 直接生成一个方案不同，AutoML-Agent 维护一个**方案知识库**，存储历史成功案例和常见 AutoML 流水线模板。在规划时，Manager Agent 先从知识库中检索与当前任务相似的成功方案作为参考，然后基于这些参考生成多个差异化的候选计划。每个候选计划包含不同的数据预处理策略、模型架构选择和训练配置组合。
    - 设计动机：传统规划方式完全依赖 LLM 的"直觉"，容易陷入局部最优（如总是选择常见的 ResNet 架构）。通过引入检索增强，系统可以参考类似任务的历史成功经验来启发更多样化的方案，同时利用多计划并行评估确保搜索覆盖面。这一设计借鉴了 RAG 在问答任务中的成功经验——外部知识的引入能显著提升 LLM 决策质量。

2. **专业化 Agent 与子任务并行执行**

    - 功能：将每个执行计划分解为多个独立的子任务，由专门设计的 Agent 并行完成。
    - 核心思路：AutoML 流水线可自然分解为数据获取、数据预处理、特征工程、模型架构设计、超参数配置、模型训练、评估与部署等子环节。AutoML-Agent 为每类子任务设计了专业化 Agent，每个 Agent 通过精心设计的 prompt 被赋予特定领域的专家能力。例如，Data Agent 被 prompt 为数据工程专家，擅长处理缺失值、特征变换、数据增强等操作；Model Agent 被 prompt 为神经网络设计专家，负责选择和配置网络架构。由于某些子任务之间相互独立（如数据预处理和模型架构选择），它们可以**并行执行**，大幅缩短端到端处理时间。
    - 设计动机：单一 Agent 处理全流水线面临两个困境：(a) prompt 过长导致注意力分散和遗忘；(b) 串行执行各步骤效率低。子任务分解解决了第一个问题（每个 Agent 只需关注自己领域），并行执行解决了第二个问题。这种分治（divide-and-conquer）策略使系统的搜索过程更加高效。

3. **多阶段验证机制（Multi-Stage Verification）**

    - 功能：在执行过程的多个关键节点检查中间结果和最终输出的正确性。
    - 核心思路：验证并非只在最终输出时进行一次性检查，而是在子任务执行的每个阶段都进行验证。具体包括：(a) **代码语法验证**——检查 Agent 生成的代码是否可执行无语法错误；(b) **逻辑一致性验证**——检查各子任务的输出是否与整体计划一致（如 Data Agent 输出的数据格式是否与 Model Agent 期望的输入格式匹配）；(c) **性能验证**——在验证集上评估模型性能，判断是否达到预期。当某个阶段的验证失败时，系统会将错误信息反馈给对应 Agent，引导其修正代码实现。
    - 设计动机：LLM 生成的代码常存在各种细微错误（如 API 参数不匹配、张量维度不一致、数据类型冲突等），仅靠一次性的最终验证往往难以定位和修复。多阶段验证实现了**早期发现、早期修复**（fail-fast），避免了错误在流水线中层层传递导致最终完全失败。这也是 AutoML-Agent 实现高成功率的关键保障。

### 训练策略

AutoML-Agent 本身不需要训练——它是一个纯推理时（inference-time）框架，所有能力来源于预训练 LLM 的 prompt 工程和多 Agent 协作。框架通过精心设计的系统 prompt 和少量示例（few-shot examples）来引导各 Agent 的行为。知识库中的方案模板可以随着使用积累逐步扩展，实现一种隐式的"持续学习"。

## 实验关键数据

### 主实验：各任务上的成功率与模型性能对比

AutoML-Agent 在 7 类下游任务（covering 14 个数据集）上进行了全面评估，覆盖了图像分类、文本分类、表格数据分析等多种 AI 应用场景。

| 任务类型 | 数据集数量 | AutoML-Agent 成功率 | 对比方法最佳成功率 | 说明 |
|----------|-----------|-------------------|------------------|------|
| 图像分类 | 多个 | 高 | 较低 | 全流水线自动化，无需手工配置 |
| 文本分类 | 多个 | 高 | 中等 | 自动选择合适的 NLP pipeline |
| 表格数据 | 多个 | 高 | 中等 | 自动特征工程 + 模型选择 |
| 回归任务 | 多个 | 高 | 较低 | 完整的数据预处理与超参调优 |
| 时序预测 | 多个 | 高 | 较低 | 端到端自动化处理 |
| 多模态 | 多个 | 较高 | 低 | 跨模态数据的全自动处理 |
| 目标检测 | 多个 | 较高 | 低 | 从数据到模型部署全覆盖 |

AutoML-Agent 在绝大多数任务上取得了最高的自动化成功率，且生成的模型在各领域中均表现出良好的性能。

### 消融实验：各核心组件的贡献

| 配置 | 成功率变化 | 说明 |
|------|-----------|------|
| Full AutoML-Agent | 最高 | 完整模型，包含所有组件 |
| w/o Retrieval-Augmented Planning | 显著下降 | 仅用单一计划，搜索空间大幅缩小 |
| w/o Multi-Agent Parallelism | 中等下降 | 改为单 Agent 串行，效率降低且准确性下降 |
| w/o Multi-Stage Verification | 显著下降 | 去掉验证后代码错误无法及时修复，成功率大幅降低 |
| Single Plan + Single Agent | 最低 | 退化为传统 LLM-based AutoML，成功率最差 |

### 关键发现

- **多阶段验证是成功率的最大保障**：去掉验证机制后成功率下降最为显著，说明 LLM 生成的代码确实需要多轮检查和迭代修复才能保证可执行性。这验证了"早期检测、早期修复"策略在代码生成场景中的有效性。
- **检索增强规划提供多样化方案**：相比单一计划策略，检索增强规划通过引入历史成功案例的参考，生成了更多样化和更合理的候选方案，在不同任务上都能找到更优的配置组合。
- **并行执行兼顾效率与质量**：多 Agent 并行执行不仅加速了整体处理时间，还因为每个 Agent 专注于单一子任务而提升了各环节的完成质量（相比单 Agent 同时处理所有任务时注意力分散的问题）。
- **跨领域泛化能力强**：AutoML-Agent 在 7 类差异很大的任务上都保持了良好的表现，证明了多 Agent 框架对任务类型的泛化能力。

## 亮点与洞察

- **全流水线覆盖是核心差异化**：相比只做模型选择或只做超参调优的 LLM-based AutoML 方法，AutoML-Agent 从数据获取延伸到模型部署，真正实现了端到端自动化。这意味着一个完全不懂 ML 的用户只需用自然语言描述需求，就能得到可部署的模型——这是 AutoML 的终极愿景。
- **RAG 思想在规划中的妙用**：将 RAG 从"检索知识辅助回答"推广到"检索历史方案辅助规划"，这个迁移非常自然且有效。这一设计思路可推广到任何需要多方案搜索的 Agent 系统——如代码生成 Agent 检索历史代码模板、科研 Agent 检索相关方法论等。
- **子任务分解 + 并行执行的效率增益**：通过分析 AutoML 流水线的任务依赖图，识别出可并行的子任务（如数据预处理和模型架构设计），用专业化 Agent 并行处理。这种"分析依赖→分解→并行"的方法论可以迁移到任何复杂的多步骤任务中。
- **多阶段验证的"fail-fast"哲学**：不是在最后才检查结果好坏，而是每个阶段都做验证并及时修复，这种"质量门控"思想可广泛应用于所有 LLM 代码生成场景。

## 局限与展望

- **对 LLM 基座能力的依赖**：框架性能高度依赖底层 LLM 的代码生成和推理能力。当任务需要非常特定的领域知识（如量化金融的因子挖掘）时，通用 LLM 可能力不从心。可探索为不同领域的 Agent 接入专业化的小模型或领域知识库。
- **知识库的冷启动问题**：检索增强规划依赖历史成功方案的积累，在系统早期使用阶段可能因知识库稀疏而无法充分发挥效果。需要设计有效的初始化策略（如从公开的 Kaggle 获奖方案中抽取模板）。
- **子任务间的耦合处理**：论文假设子任务相对独立可以并行执行，但在实际场景中数据预处理（如特征工程）和模型设计之间存在较强耦合（例如某些特征变换只对特定模型有效）。这种耦合可能导致并行执行的方案在组合后不一定最优。
- **成本与延迟**：多 Agent + 多计划 + 多阶段验证意味着大量的 LLM API 调用，推理成本和时间延迟可能较高。论文未充分讨论成本效益分析。
- **评估范围**：虽然覆盖了 7 类任务 14 个数据集，但缺乏在生产级别数据集（百万级样本、高维特征）上的验证，也未与传统非 LLM 的 AutoML 工具（如 AutoGluon、H2O）做系统性的性能对比。

## 相关工作与启发

- **vs Data-Interpreter / MLCopilot**：这类方法通常只覆盖 AutoML 流水线的部分环节（如 Data-Interpreter 只做数据分析和可视化），AutoML-Agent 的优势在于全流水线覆盖。但部分环节的专业深度可能不如专注型工具。
- **vs AutoGen / MetaGPT**：这些是通用的多 Agent 框架，需要用户自行编排 Agent 协作流程。AutoML-Agent 在此基础上针对 AutoML 场景做了专门设计（如检索增强规划、领域特定的 Agent 角色定义），开箱即用但通用性受限。
- **vs CAAFE / AutoML-GPT**：早期的 LLM-for-AutoML 工作，但仅用单一 LLM 直接生成方案，缺乏多 Agent 协作和验证机制。AutoML-Agent 的多计划搜索和分治策略带来了更高的成功率。
- **启发**：AutoML-Agent 的架构设计（Manager 规划 → 专业 Agent 执行 → 验证反馈）是一个可复用的多 Agent 协作范式，可以迁移到科研自动化（AutoResearch）、软件开发（AutoDev）等类似的全流水线自动化场景中。

## 评分

- 新颖性: ⭐⭐⭐⭐ 全流水线 AutoML + 多 Agent 协作的结合是自然但有意义的创新，检索增强规划和多阶段验证设计精巧
- 实验充分度: ⭐⭐⭐⭐ 7 类任务 14 个数据集覆盖面广，但缺乏与传统 AutoML 的深入对比和成本分析
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，动机推导连贯，但部分细节（如知识库构建）描述不够详尽
- 价值: ⭐⭐⭐⭐ 展示了 LLM 多 Agent 框架在全流水线 AutoML 中的实用潜力，对降低 ML 开发门槛有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](../../ICML2026/multi_agent/omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ICML 2025\] Theorem-of-Thought: A Multi-Agent Framework for Abductive, Deductive, and Inductive Reasoning in Language Models](theorem-of-thought_a_multi-agent_framework_for_abductive_deductive_and_inductive.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](../../ICLR2026/multi_agent/hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [\[NeurIPS 2025\] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization](../../NeurIPS2025/multi_agent/rd-agent-quant_a_multi-agent_framework_for_data-centric_factors_and_model_joint_.md)
- [\[ICML 2025\] From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium](from_debate_to_equilibrium_belief-driven_multi-agent_llm_reasoning_via_bayesian_.md)

</div>

<!-- RELATED:END -->
