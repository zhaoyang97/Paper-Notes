---
title: >-
  [论文解读] ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems
description: >-
  [CVPR 2025][多智能体][LLM Agent] ComfyBench 提出了首个评估LLM Agent在ComfyUI中自主设计协作AI系统能力的综合性Benchmark（200个任务、3205个节点文档、20个课程工作流），并提出ComfyAgent框架通过代码化工作流表示和多Agent协作，达到了与o1-preview相当的解决率，但在创意任务上仅解决15%，揭示了LLM Agent在自主系统设计上的巨大差距。
tags:
  - "CVPR 2025"
  - "多智能体"
  - "LLM Agent"
  - "工作流生成"
  - "ComfyUI"
  - "协作AI系统"
  - "Benchmark"
---

# ComfyBench: Benchmarking LLM-based Agents in ComfyUI for Autonomously Designing Collaborative AI Systems

**会议**: CVPR 2025  
**arXiv**: [2409.01392](https://arxiv.org/abs/2409.01392)  
**代码**: [https://github.com/xxyQwQ/ComfyBench](https://github.com/xxyQwQ/ComfyBench)  
**领域**: LLM评测  
**关键词**: LLM Agent, 工作流生成, ComfyUI, 协作AI系统, Benchmark

## 一句话总结

ComfyBench 提出了首个评估LLM Agent在ComfyUI中自主设计协作AI系统能力的综合性Benchmark（200个任务、3205个节点文档、20个课程工作流），并提出ComfyAgent框架通过代码化工作流表示和多Agent协作，达到了与o1-preview相当的解决率，但在创意任务上仅解决15%，揭示了LLM Agent在自主系统设计上的巨大差距。

## 研究背景与动机

**领域现状**：以往AI研究主要集中在开发单一庞大模型（monolithic models）来最大化特定任务的智能，但另一种思路是用LLM Agent来**自主设计协作式AI系统**——即多个AI模型组合起来的pipeline/workflow。

**现有痛点**：(1) 缺乏评估LLM Agent设计协作AI系统能力的标准化Benchmark；(2) 即使用强大的LLM作为Agent，也难以理解和组合复杂的节点系统（如ComfyUI有数千个不同功能的节点）；(3) 现有Agent框架缺乏从已有workflow中学习的机制。

**核心矛盾**：ComfyUI式的可视化工作流系统虽然灵活强大，但其节点图的组合空间极为庞大——3205个节点的排列组合和参数配置构成了一个复杂的设计空间，远超单个LLM的上下文理解能力。

**本文目标** (1) 构建评估Agent工作流设计能力的Benchmark；(2) 开发能有效利用节点文档和已有工作流来生成新workflow的Agent框架。

**切入角度**：将工作流转化为代码表示（而非可视化节点图），使LLM能更好地理解和生成workflow；采用多Agent协作，让不同Agent负责文档检索、流程学习和代码生成。

**核心 idea**：用代码表示工作流使LLM可以理解和生成，用多Agent协作从已有workflow中学习并设计新的协作AI系统。

## 方法详解

### 整体框架

ComfyBench包含两个部分：(1) **Benchmark**——200个多样化任务指令（涵盖文生图、图编辑、风格迁移、超分辨率、物体移除等各种图像生成挑战），3205个ComfyUI节点的详细文档，以及20个参考工作流供Agent学习；(2) **ComfyAgent**——一个多Agent框架，能自主读取节点文档和参考workflow，为给定任务生成新的workflow代码，该代码可被解释器反向转换为ComfyUI工作流并执行。

### 关键设计

1. **代码化工作流表示（Code-based Workflow Representation）**:

    - 功能：将ComfyUI的可视化节点图转换为LLM可理解的Python代码格式
    - 核心思路：每个ComfyUI workflow由JSON描述（节点ID、类型、参数、连线），将其转换为等价的Python代码，包含节点实例化和连接关系。代码可被解释器反向转换为JSON workflow并在ComfyUI中执行
    - 设计动机：LLM天然理解代码，代码比JSON节点描述更紧凑、更有结构性，便于LLM生成和理解

2. **多Agent协作系统**:

    - 功能：分工协作完成从任务理解到workflow生成的全流程
    - 核心思路：多个Agent分别负责不同职责——文档检索Agent从3205个节点文档中检索相关节点信息，课程学习Agent从20个参考workflow中找到最相关的已有方案作为示例，代码生成Agent基于检索到的节点文档和参考workflow生成新的workflow代码
    - 设计动机：将复杂的系统设计任务分解为可管理的子任务，利用Agent间的协作弥补单个LLM的能力局限

3. **双指标评估体系**:

    - 功能：全面评估Agent的工作流设计能力
    - 核心思路：**Pass Rate**（通过率）——生成的workflow能否被ComfyUI正确执行（无语法/连接错误）；**Resolve Rate**（解决率）——执行结果是否满足任务要求（由人工或自动评估判断）
    - 设计动机：通过率衡量Agent对节点系统的理解程度，解决率衡量Agent对任务语义的理解和创造性设计能力

### 损失函数 / 训练策略

无需训练——ComfyAgent是推理时的Agent框架，基于LLM的in-context learning和工具调用能力。

## 实验关键数据

### 主实验

| Agent | 通过率 (Pass Rate) | 解决率 (Resolve Rate) |
|-------|-------------------|---------------------|
| 基础LLM Agent | 低 | 低 |
| **ComfyAgent** | **与o1-preview相当** | **与o1-preview相当** |
| o1-preview | 基准 | 基准 |

- ComfyAgent显著超越其他Agent方法
- 创意任务上仅解决15%

### 消融实验

| 配置 | 说明 |
|------|------|
| ComfyBench完整设置 | 200个任务，3205节点，20参考workflow |
| 不同任务类别 | 创意任务vs标准任务差异巨大（15% vs 更高） |

### 关键发现

- **创意任务是核心瓶颈**：ComfyAgent仅能解决15%的创意任务，说明LLM Agent在需要组合创新的场景中仍有巨大提升空间
- **代码化表示显著优于JSON表示**：LLM更容易理解和生成Python代码形式的workflow
- **从已有workflow学习至关重要**：课程学习机制通过提供参考方案极大提高了生成质量
- **与o1-preview对齐**：ComfyAgent（基于开源LLM）达到了o1-preview的水平，显示了Agent框架设计的重要性

## 亮点与洞察

- **问题设定的前瞻性**：从"让AI变聪明"转向"让AI设计AI系统"，代表了一种范式转变。用LLM Agent替代人类进行AI系统组装，是AGI的一个重要方向
- **代码作为通用接口**的思路很有启发——将各种复杂系统的操作统一为代码生成问题，是LLM Agent通用化的关键
- **Benchmark设计实用**：200个真实的图像生成任务、3205个真实节点文档，直接在ComfyUI生态中评估，结果可执行可验证

## 局限与展望

- 创意任务仅15%解决率，说明Agent在需要全新组合（而非模仿已有方案）时严重不足
- Benchmark限于ComfyUI的图像生成领域，未覆盖视频、音频等其他协作系统
- 评估中resolve rate的判定标准可能存在主观性
- 与o1-preview对比时，缺乏更详细的per-category分析

## 相关工作与启发

- **vs 通用Agent（如AutoGPT、MetaGPT）**: 通用Agent关注通用任务分解，ComfyAgent专注于工作流设计，在这一特定场景更有效
- **vs 代码生成（如Copilot）**: 代码生成关注单一程序，ComfyAgent生成的是协作系统的"蓝图"，需要理解多组件间的交互

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个评估Agent设计协作AI系统的Benchmark，问题设定有开创性
- 实验充分度: ⭐⭐⭐ 核心指标报告充分，但缺少详细的per-model/per-category数据
- 写作质量: ⭐⭐⭐⭐ 问题motivation清晰，系统设计逻辑完整
- 价值: ⭐⭐⭐⭐ Benchmark + Agent框架的组合对LLM Agent研究具有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](../../NeurIPS2025/multi_agent/maszero_designing_multiagent_systems_with_zero_supervision.md)
- [\[CVPR 2025\] Collaborative Tree Search for Enhancing Embodied Multi-Agent Collaboration](collaborative_tree_search_for_enhancing_embodied_multi-agent_collaboration.md)
- [\[AAAI 2026\] COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis](../../AAAI2026/multi_agent/coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)
- [\[ICLR 2026\] When Agents "Misremember" Collectively: Exploring the Mandela Effect in LLM-based Multi-Agent Systems](../../ICLR2026/multi_agent/when_agents_misremember_collectively_exploring_the_mandela_effect_in_llm-based_m.md)
- [\[NeurIPS 2025\] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks](../../NeurIPS2025/multi_agent/medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)

</div>

<!-- RELATED:END -->
