---
title: >-
  [论文解读] Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors
description: >-
  [AAAI 2026][LLM Agent][工具调用] 提出VGCO框架，利用LLM作为分层编辑器，通过验证引导的方式迭代优化工具文档和知识库上下文，显著提升大规模工具调用场景下的检索召回、工具选择和参数填充准确率。 1. 领域现状：工具调用(Tool Calling)已成为扩展LLM能力的关键机制，使模型能调用外部API…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "工具调用"
  - "上下文优化"
  - "LLM编辑器"
  - "知识库检索"
  - "文档优化"
---

# Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors

**会议**: AAAI 2026  
**arXiv**: [2512.13860](https://arxiv.org/abs/2512.13860)  
**代码**: 无  
**领域**: 其他  
**关键词**: 工具调用, 上下文优化, LLM编辑器, 知识库检索, 文档优化

## 一句话总结

提出VGCO框架，利用LLM作为分层编辑器，通过验证引导的方式迭代优化工具文档和知识库上下文，显著提升大规模工具调用场景下的检索召回、工具选择和参数填充准确率。

## 研究背景与动机

1. **领域现状**：工具调用(Tool Calling)已成为扩展LLM能力的关键机制，使模型能调用外部API/工具来获取实时信息或执行复杂操作。当前方法分为两大类：基于微调的方法（如xLAMs、ToolBench等）和免调优方法（依赖上下文学习）。

2. **现有痛点**：工具文档和知识库内容通常是为人类编写的，与LLM的理解需求不匹配。在工业级场景中，企业管理着数百个API工具，功能重叠、边界模糊的问题尤为严重，导致工具调用频繁失败。

3. **核心矛盾**：现有上下文优化框架（如DSPy、SAMMO、Promptim）存在三大局限——无状态操作（不关注上下文当前状态）、缺乏结构化动作空间（仅依赖token级调整）、缺少领域特定评估机制。

4. **本文目标**：如何系统性地优化工具相关文档和知识库上下文，使其更好地对齐LLM的工具调用需求，尤其是在大规模工业场景下。

5. **切入角度**：作者观察到，许多工具调用失败并非源于模型推理能力不足，而是上下文信息不完整、不一致或结构不当。

6. **核心 idea**：将工具上下文分层为检索层、工具层和参数层，利用验证信号引导LLM编辑器在每个层级进行有针对性的迭代优化。

## 方法详解

### 整体框架

VGCO分为两个阶段运行：(1) **评估阶段**：收集结构化信号，识别和诊断上下文失败案例；(2) **优化阶段**：通过分层LLM编辑生成修订后的工具文档和上下文，并通过离线验证后集成到生产系统中。整体pipeline为"用户查询→验证数据集评估→识别错误→LLM编辑器修复→验证改进→更新知识库"的闭环。

### 关键设计

1. **分层LLM编辑器 (Hierarchical LLMs-as-Editors)**:
    - 功能：将工具上下文优化分解为检索层($D_r$)、工具层($D_t$)、参数层($D_p$)三个层级，分别由对应的编辑器$\mathcal{M}_r, \mathcal{M}_t, \mathcal{M}_p$负责优化。
    - 核心思路：优化自顶向下进行——先优化检索内容以提升候选工具的召回率，再优化工具描述以提升选择准确率，最后优化参数schema以提升参数填充正确性。每层编辑器根据失败案例进行修改，只有当修改后的评估指标$eval(\mathcal{M}(\tilde{D},X),Y)$优于修改前时才保留变更。
    - 设计动机：工具使用的上下文天然具有层级结构，现有方法将文档视为整体单元忽略了层级间的依赖关系，分层处理能隔离编辑范围、减少意外副作用。

2. **验证引导的编辑指令 (Guided Instruction Editor)**:
    - 功能：为每个层级的编辑器提供结构化指导，克服现有优化器的无状态、无约束问题。
    - 核心思路：每个编辑器的行为由引导系统提示控制，包含三个关键要素——**状态空间**（追踪先前编辑及其效果，防止重复退化）、**动作空间**（定义任务特定的编辑操作如添加、删除、统一术语等）、**奖励信号**（利用ground truth逐query验证错误并提供反馈）。动作空间按层级定制：检索层编辑器优化查询-工具对齐，工具层编辑器增强语义精度，参数层编辑器细化schema定义。
    - 设计动机：直接解决了现有优化器的三大局限，通过结构化约束将编辑控制在高语义层级操作上，避免了token级调整引入的边界问题。

3. **迭代验证与更新机制**:
    - 功能：确保每次编辑都是可验证的、可增量的改进。
    - 核心思路：在每次迭代中，推理模型$\mathcal{M}$在当前文档上运行验证集，收集失败案例$E_c$。编辑器对每个失败案例生成修改建议，仅当修改后全局评估指标提升时才接受更新。整个过程持续到预定义的性能阈值。
    - 设计动机：不同于依赖模型自生成反馈的self-refinement方法，VGCO利用显式验证信号，确保改进是可解释的、鲁棒的。

### 损失函数 / 训练策略

VGCO本身是一个免调优框架，不涉及传统意义上的损失函数。其"训练"是通过LLM编辑器的迭代优化实现的。论文同时探讨了编辑器的后训练方案：可通过SFT（最大化正确编辑的似然）、DPO（利用正负编辑样本对的偏好学习）或GRPO（组内相对偏好优化）来进一步提升编辑器性能。

## 实验关键数据

### 主实验

在xLAM数据集（100个最常用工具）上，以Claude Sonnet 3.5为推理模型的结果：

| 编辑方法 | 检索召回 | 工具选择 | 参数填充 | 最终准确率 |
|----------|---------|---------|---------|-----------|
| Raw | 78.98 | 71.5 | 57.5 | 32.5 |
| ReAct | 91.64 | 75.0 | 56.0 | 38.5 |
| DRAFT | 95.62 | 78.0 | 62.0 | 46.2 |
| **VGCO (Claude-4)** | **97.48** | **80.0** | **65.0** | **50.7** |
| **VGCO (Claude-4.5)** | **98.30** | **84.0** | **74.0** | **61.0** |

在BFCL数据集上（Claude Sonnet 4编辑器 + Claude Sonnet 3.7推理）：

| 编辑方法 | 工具选择 | 参数填充 |
|----------|---------|---------|
| Raw | 69.1 | 55.8 |
| DRAFT | 92.1 | 88.3 |
| **VGCO** | **96.9** | **94.7** |

### 消融实验

| 配置 | 工具选择 | 参数填充 | 说明 |
|------|---------|---------|------|
| Full Guided Instruction | 96.94 | 94.73 | 完整模型 |
| w/o Common Issues | 82.25 | 81.43 | 去掉错误类型列表 |
| w/o ICL Examples | 74.89 | 65.43 | 去掉上下文学习示例，影响最大 |
| w/o Requirements | 91.18 | 89.35 | 去掉编辑约束条件 |
| w/o Analysis Task | 87.32 | 85.41 | 去掉结构化分析任务 |

### 关键发现

- ICL示例对性能影响最大，去掉后工具选择和参数填充分别下降22%和29%，说明上下文示例是对齐编辑器行为的核心要素
- 迭代优化的最大收益出现在前两轮（如Claude Sonnet 3.5 + Claude Sonnet 4.5编辑器从0.599提升到0.801），之后逐渐收敛
- 更强的编辑器模型（Claude Sonnet 4.5）在所有推理模型上都能产生最好且最稳定的改进
- VGCO在工具选择和参数填充这两个需要细粒度上下文推理的子任务上优势最为显著

## 亮点与洞察

- **分层解耦思路**很巧妙：将文档优化分解为检索/选择/参数三层，每层独立优化再级联传播，既降低了优化复杂度，又保证了层间一致性。这种思路可以迁移到任何涉及层级决策的系统优化中。
- **验证引导 vs 自我反思**：不依赖模型自身反馈而使用外部验证信号来引导优化，这比self-refinement更可靠。核心洞察是"工具调用的失败大多不是推理问题而是上下文问题"。
- **上下文工程视角**：将问题框架从"prompt优化"提升到"上下文工程"层面，不仅优化prompt还优化检索内容、工具描述和参数schema，覆盖了整个工具调用pipeline。

## 局限与展望

- 目前仅验证了单轮(single-turn)工具调用场景，多轮对话中的上下文优化是重要的未来方向
- 编辑器依赖高质量验证数据集，当验证数据覆盖不全时优化效果可能有限
- 在工具数量继续扩大（超过数百个）时的可扩展性尚需进一步验证
- 领域迁移能力（domain shift）和低资源场景下的鲁棒性需要额外验证

## 相关工作与启发

- **vs DRAFT**: DRAFT也通过LLM试错交互来改进工具文档，但它不分解工具调用子任务，每次迭代编辑完整任务。VGCO通过分层编辑实现更精细的控制和更高效的优化。
- **vs DSPy/SAMMO**: 这些通用prompt优化框架在工业级场景下效果有限，因为它们是无状态的、缺乏结构化动作空间。VGCO针对工具调用场景设计了专门的状态感知和动作约束机制。
- **vs ReAct**: ReAct通过推理-行动交替来改善工具使用，但不修改工具文档本身。VGCO从源头改善文档质量，与ReAct是互补关系。

## 评分

- 新颖性: ⭐⭐⭐⭐ 分层编辑器+验证引导的思路在工具调用优化中有不错的创新性，但整体框架偏工程
- 实验充分度: ⭐⭐⭐⭐ 多模型、多数据集、多基线对比完整，消融实验清晰
- 写作质量: ⭐⭐⭐⭐ 问题动机阐述清晰，框架描述系统，但公式较多略显冗长
- 价值: ⭐⭐⭐⭐ 对工业级工具调用场景有很强的实用价值，但学术新颖性一般

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](../../ACL2026/llm_agent/higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)
- [\[AAAI 2026\] Loss-Guided Auxiliary Agents for Overcoming Mode Collapse in GFlowNets](loss-guided_auxiliary_agents_for_overcoming_mode_collapse_in_gflownets.md)
- [\[ACL 2025\] Play2Prompt: Zero-shot Tool Instruction Optimization for LLM Agents via Tool Play](../../ACL2025/llm_agent/play2prompt_zero-shot_tool_instruction_optimization_for_llm_agents_via_tool_play.md)
- [\[AAAI 2026\] LLMTM: Benchmarking and Optimizing LLMs for Temporal Motif Analysis in Dynamic Graphs](llmtm_benchmarking_and_optimizing_llms_for_temporal_motif_analysis_in_dynamic_gr.md)

</div>

<!-- RELATED:END -->
