---
title: >-
  [论文解读] Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model
description: >-
  [ACL 2025][LLM Agent][对话Agent] 本文提出CoALM（Conversational Agentic Language Model），通过构建融合多轮ReAct推理和复杂API调用的多任务训练数据CoALM-IT，训练出同时擅长传统任务型对话（TOD）和语言Agent（LA）工具调用的统一模型，在MultiWOZ、BFCL V3和API-Bank三个基准上超越GPT-4o等专用模型。
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "对话Agent"
  - "工具调用"
  - "多轮对话"
  - "统一模型"
  - "任务型对话"
---

# Can a Single Model Master Both Multi-turn Conversations and Tool Use? CoALM: A Unified Conversational Agentic Language Model

**会议**: ACL 2025  
**链接**: [ACL Anthology](https://aclanthology.org/2025.acl-long.605/)
**代码**: 无  
**领域**: LLM Agent / 对话系统 / 工具使用  
**关键词**: 对话Agent, 工具调用, 多轮对话, 统一模型, 任务型对话

## 一句话总结

本文提出CoALM（Conversational Agentic Language Model），通过构建融合多轮ReAct推理和复杂API调用的多任务训练数据CoALM-IT，训练出同时擅长传统任务型对话（TOD）和语言Agent（LA）工具调用的统一模型，在MultiWOZ、BFCL V3和API-Bank三个基准上超越GPT-4o等专用模型。

## 研究背景与动机

**领域现状**：当前AI对话助手需要两种核心能力：(1) 传统任务型对话（TOD）能力——维护多轮对话中的用户意图、管理对话状态、操作有限的API集合（如餐厅预订系统）；(2) 语言Agent（LA）能力——使用开放集合的工具/API，执行复杂的多步推理任务。

**现有痛点**：TOD系统和LA系统各有专长但互有短板：TOD系统擅长多轮对话管理（维持上下文、处理用户意图变更），但只能操作预定义的有限API集合，接入新服务需要重新收集训练数据；LA系统擅长灵活的工具调用和推理，但在多轮对话中无法有效维持用户意图，容易"忘记"之前的对话上下文。

**核心矛盾**：多轮对话管理和灵活工具使用是构建优秀对话Agent的两个必要条件，但现有方法针对其中一种能力优化时往往损害另一种。两类能力的训练数据格式和目标不同，简单混合训练效果不佳。

**本文目标**：(1) 量化分析两类方法在跨领域评测中的差距；(2) 设计统一的训练方案使单个模型同时具备TOD和LA能力。

**切入角度**：通过精心构建的多任务训练数据集（CoALM-IT），将多轮ReAct推理与API调用交织在对话流程中，使模型在训练时就接触到两种能力的融合场景。

**核心 idea**：用统一的数据格式将TOD的对话状态管理和LA的ReAct推理融合在一起，训练单一模型同时掌握两种能力。

## 方法详解

### 整体框架

CoALM的核心是构建CoALM-IT训练数据集，然后在其上微调LLaMA系列模型（8B/70B/405B）。输入是多轮对话历史+可用工具描述，输出是融合了推理、工具调用和自然语言回复的多步响应。

### 关键设计

1. **CoALM-IT多任务训练数据构建**:

    - 功能：创建统一格式的混合训练数据，覆盖TOD和LA两种能力
    - 核心思路：从现有的TOD数据集（MultiWOZ等）和LA数据集（ToolBench等）出发，将它们转化为统一的对话格式。关键创新在于将TOD中的对话状态更新操作转化为类似API调用的形式，并引入ReAct风格的思考-行动-观察循环。这样TOD数据和LA数据在格式上统一，模型可以从两者中学习互补的能力。同时通过数据增强为TOD场景引入更多样的API描述,为LA场景引入更丰富的多轮对话上下文
    - 设计动机：数据格式的统一是实现能力统一的基础——如果TOD和LA使用完全不同的格式训练，模型在推理时无法灵活切换

2. **多轮ReAct推理与对话管理融合**:

    - 功能：在多轮对话中嵌入ReAct推理流程
    - 核心思路：在每个对话轮次中，模型首先进行Thought（思考当前用户意图和对话历史），然后决定Action（调用API还是直接回复），如果调用API则等待Observation（API返回结果），最后生成Response（基于推理和API结果的自然语言回复）。这种格式自然地将TOD的对话状态管理（通过Thought跟踪意图变化）和LA的工具调用（通过Action执行API）统一起来
    - 设计动机：ReAct的显式推理步骤使模型能够在每轮对话中"反思"上下文变化，比端到端生成更好地维持对话连贯性

3. **多尺度模型训练策略**:

    - 功能：在不同规模的LLaMA模型上验证方法的可扩展性
    - 核心思路：分别训练CoALM-8B、CoALM-70B和CoALM-405B，使用相同的CoALM-IT数据集但不同的训练超参数。在三个互补的基准上评估：MultiWOZ 2.4（TOD，测试对话管理）、BFCL V3（LA，测试函数调用）、API-Bank（LA，测试多步API使用）
    - 设计动机：验证统一训练方案在不同模型规模下的有效性，同时探索规模效应

### 训练策略

使用标准的指令微调（instruction tuning）流程，在CoALM-IT上对LLaMA模型进行全参数微调。训练损失为标准的下一个token预测。

## 实验关键数据

### 主实验

| 基准 | 指标 | CoALM-8B | CoALM-70B | CoALM-405B | GPT-4o | 专用SOTA |
|------|------|----------|-----------|------------|--------|---------|
| MultiWOZ 2.4 | Joint Goal Acc | 高 | 更高 | 最优 | 较低 | TOD专用模型 |
| BFCL V3 | Function Call Acc | 高 | 更高 | 最优 | 基线 | LA专用模型 |
| API-Bank | API Call Success | 高 | 更高 | 最优 | 基线 | LA专用模型 |

### 消融实验

| 配置 | MultiWOZ | BFCL V3 | API-Bank | 说明 |
|------|----------|---------|----------|------|
| CoALM-IT（全数据） | 最优 | 最优 | 最优 | 完整模型 |
| 仅TOD数据训练 | 好 | 差 | 差 | 缺失LA能力 |
| 仅LA数据训练 | 差 | 好 | 好 | 缺失TOD能力 |
| 简单混合（无格式统一） | 中等 | 中等 | 中等 | 格式不统一影响效果 |
| 不含ReAct | 下降明显 | 下降 | 下降 | 推理步骤缺失 |

### 关键发现
- 单一CoALM模型在TOD和LA两个领域都超越了领域专用的最好方法，证明了统一模型的可行性
- CoALM甚至超越了GPT-4o，说明针对性训练数据的质量比模型规模更重要
- 数据集的格式统一（而非简单混合）是关键——同样的数据不做格式统一时效果大幅下降
- 规模效应明显：8B→70B→405B的提升在三个基准上一致

## 亮点与洞察
- **"打通TOD与LA"的研究视角**非常有价值——现有研究将两者割裂，但真实的对话助手需要同时具备这两种能力。CoALM首次系统性地验证了统一模型的可行性
- **数据工程的力量**再次被证明——CoALM-IT的设计（格式统一+ReAct融合）是方法成功的核心，比模型架构创新更重要
- 该方法为构建实用的对话Agent提供了清晰的技术路线：高质量的统一训练数据 + 指令微调

## 局限与展望
- CoALM-IT的构建过程需要大量人工设计和质量控制，可扩展性有限
- 仅在英语场景下评估，多语言对话Agent的能力尚不清楚
- 未考虑安全性和幻觉控制——工具调用错误在实际部署中可能造成严重后果
- 未来可以探索在统一框架中加入更多能力维度（如知识问答、代码执行等）

## 相关工作与启发
- **vs ToolLLM**: ToolLLM专注于工具使用能力的微调，不涉及多轮对话管理；CoALM在工具使用上表现相当的同时还具备TOD能力
- **vs SOLOIST/SimpleTOD**: 传统TOD微调方法只能处理预定义API，CoALM通过统一格式实现了开放API处理
- **vs ReAct**: ReAct提出了思考-行动-观察的推理框架，CoALM将其嵌入多轮对话场景，是ReAct在对话Agent中的自然扩展

## 评分
- 新颖性: ⭐⭐⭐⭐ 打通TOD与LA的统一建模思路有开创性，但技术手段（格式统一+SFT）较为直接
- 实验充分度: ⭐⭐⭐⭐⭐ 三个互补基准+多规模模型+详细消融，非常充分
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，分析有说服力
- 价值: ⭐⭐⭐⭐⭐ 为构建统一对话Agent提供了标杆

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] T1: A Tool-Oriented Conversational Dataset for Multi-Turn Agentic Planning](../../NeurIPS2025/llm_agent/t1_a_tool-oriented_conversational_dataset_for_multi-turn_agentic_planning.md)
- [\[ACL 2025\] GUICourse: From General Vision Language Model to Versatile GUI Agent](guicourse_from_general_vision_language_model_to_versatile_gui_agent.md)
- [\[ACL 2025\] Bel Esprit: Multi-Agent Framework for Building AI Model Pipelines](bel_esprit_multi-agent_framework_for_building_ai_model_pipelines.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](toolhop_multi_hop_tool_use.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)

</div>

<!-- RELATED:END -->
