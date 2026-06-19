---
title: >-
  [论文解读] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions
description: >-
  [ACL 2025][函数调用] CONFETTI 提出了一个面向多轮对话场景的函数调用（function-calling）评测基准，包含 109 段人类模拟对话、313 个用户轮次和 86 个 API，通过 off-policy turn-level 评估和 dialog act 标注系统性地测试 LLM 在复杂对话场景下的工具调用能力，发现即便最强模型（Nova Pro）也仅达 40% 左右，链式调用更是普遍短板。
tags:
  - "ACL 2025"
  - "函数调用"
  - "对话评测基准"
  - "多轮交互"
  - "工具使用"
  - "LLM评估"
---

# CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions

**会议**: ACL 2025  
**arXiv**: [2506.01859](https://arxiv.org/abs/2506.01859)  
**代码**: 无  
**领域**: 其他  
**关键词**: 函数调用, 对话评测基准, 多轮交互, 工具使用, LLM评估

## 一句话总结

CONFETTI 提出了一个面向多轮对话场景的函数调用（function-calling）评测基准，包含 109 段人类模拟对话、313 个用户轮次和 86 个 API，通过 off-policy turn-level 评估和 dialog act 标注系统性地测试 LLM 在复杂对话场景下的工具调用能力，发现即便最强模型（Nova Pro）也仅达 40% 左右，链式调用更是普遍短板。

## 研究背景与动机

**领域现状**：随着 LLM 作为智能助手的应用日益广泛，函数调用（function-calling）已成为 LLM 与外部工具和 API 交互的核心能力。现有的 function-calling 评测基准（如 ToolBench、API-Bench、BFCL 等）主要关注单轮调用或简单的多轮场景，评测的重点在于"能否正确调用单个 API"。

**现有痛点**：真实用户与 LLM 的交互远比单轮调用复杂——用户会追问、纠正目标、切换话题、提出模糊需求甚至隐含意图。现有基准在以下方面存在明显不足：（1）**缺乏真实的对话复杂性**，不涉及 goal correction、goal switching 等场景；（2）**评估粒度过粗**，通常只看最终是否调用正确，忽略了中间的对话管理质量；（3）**API 规模有限**，不能测试模型在大量候选 API 中的选择能力。

**核心矛盾**：function-calling 的评测需要同时考虑"工具选择的准确性"和"对话管理的合理性"，但现有基准几乎都只关注前者，忽略了后者。而在多轮对话中，一个优秀的 agent 不仅要调对 API，还要能正确理解模糊需求、处理目标切换、进行合理的澄清追问。

**本文目标**：构建一个涵盖多种对话复杂性维度的 function-calling 基准，同时评估工具调用准确性和对话响应质量，揭示当前 LLM 在真实多轮工具使用场景中的真实水平。

**切入角度**：作者从真实对话交互的复杂性出发，人工设计了覆盖多种会话难点的测试场景，并引入 dialog act 标注来评估模型在非函数调用轮次中的响应质量。

**核心 idea**：通过精心设计的人类模拟对话 + off-policy turn-level 评估 + dialog act 分析，构建一个全面刻画 LLM 对话式 function-calling 能力的多维评测框架。

## 方法详解

### 整体框架

CONFETTI 的构建流程包括三个阶段：（1）**对话设计与收集**——由标注者模拟用户与 agent 的多轮交互，刻意设计各种复杂场景；（2）**标注体系建立**——为每个轮次标注期望的函数调用和 dialog act；（3）**评估框架**——给定对话历史，让待评估 LLM 在每个 turn 独立作答（off-policy），然后与参考标注进行多维度对比。

### 关键设计

1. **多维对话复杂性覆盖**:

    - 功能：确保基准能测试到 LLM 在真实场景中会遇到的各种对话难点
    - 核心思路：定义了多种对话复杂性类型，包括：follow-ups（追问，在前一轮结果基础上继续提问）、goal correction（用户修正之前的目标，如"不，我改成明天的航班"）、goal switching（完全切换到新话题/新需求）、ambiguous goals（模糊需求，需要 agent 主动澄清）、implicit goals（未明确说出但可以推断的意图）。109 段对话系统性地覆盖这些维度
    - 设计动机：真实用户不会像评测数据那样"配合性"地提出清晰需求，对话中充满了修正、切换和模糊表达。只有覆盖这些复杂性才能真正测出模型的实际能力

2. **Off-Policy Turn-Level 评估**:

    - 功能：在每个对话轮次独立评估模型，避免误差累积影响评测公平性
    - 核心思路：给模型提供标准的对话历史（而非模型自己之前的输出）作为上下文，在每个 user turn 让模型独立生成响应。这样做可以确保不同模型在相同的上下文下被评估，消除了"on-policy 评估中前序错误导致后续全错"的问题。同时支持评估链式函数调用（chained function-calls），即单个 turn 内需要连续调用多个 API 的场景
    - 设计动机：on-policy（让模型自己完成整段对话）评估虽然更贴近真实场景，但会因为中间某个turn出错导致后续评估失效。off-policy 方式可以独立评估每个 turn 的能力，获得更细粒度、更公平的结果

3. **Dialog Act 标注系统**:

    - 功能：评估模型在非函数调用轮次中的对话管理质量
    - 核心思路：为每个轮次的标准响应标注 dialog act（如 inform、request、confirm、clarify 等），评估模型不仅要"做对事"（调用正确API），还要"说对话"（生成合适的对话行为）。例如面对模糊需求时，模型应该生成 clarification 而不是直接猜测调用；在需要确认时应该 confirm 而不是直接执行
    - 设计动机：function-calling 不止是 API 调用——一个好的对话 agent 需要在调用之前做适当的信息收集、确认和澄清。Dialog act 评估弥补了现有基准只关注函数调用正确性的短板

### 损失函数 / 训练策略

CONFETTI 是一个评测基准而非训练方法，不涉及损失函数或训练策略。评估指标方面，主要使用函数调用的 precision/recall/F1（检查函数名和参数的匹配度）以及 dialog act 的准确率。对于链式调用，还评估调用序列的完整性和顺序正确性。

## 实验关键数据

### 主实验

论文评估了多个 SOTA LLM 在 CONFETTI 上的表现，按综合 F1 分数排名：

| 模型 | Overall F1 (%) | 函数名准确率 | 参数准确率 | 链式调用成功率 |
|------|---------------|------------|-----------|-------------|
| Nova Pro | 40.01 | 较高 | 中等 | 低 |
| Claude Sonnet v3.5 | 35.46 | 高 | 中等 | 低 |
| Llama 3.1 405B | 33.19 | 中等偏高 | 中等 | 低 |
| Command-R-Plus | 31.18 | 中等 | 中等 | 低 |
| Mistral-Large-2407 | 30.07 | 中等 | 中等偏低 | 很低 |

### 消融/分析实验

不同维度下的性能变化分析：

| 分析维度 | 现象 | 说明 |
|---------|------|------|
| API 数量增加 (5→20+) | 部分模型性能急剧下降 | 大量候选 API 时选择困难加剧 |
| 对话长度增加 | 部分模型性能稳定，部分明显退化 | 暴露了长上下文处理的差异 |
| 链式函数调用 | 所有模型表现严重不佳 | 最普遍的短板，单 turn 多次调用极具挑战 |
| Goal switching 场景 | 显著难于 follow-ups | 切换目标需要更强的上下文理解 |
| Ambiguous goals | 多数模型倾向直接猜测而非澄清 | 对话管理能力普遍不足 |

### 关键发现

- **链式函数调用是所有模型的共同短板**：单轮内需要连续调用多个 API 时，即便是最强模型也表现不佳。这表明当前  LLM 在规划多步操作方面仍有明显缺陷
- **API 数量是关键瓶颈**：当可用 API 超过 20 个时，很多模型的准确率显著下降，说明模型在大规模 API 选择时存在"选择困难"。但 Nova Pro 和 Claude 在此方面表现相对稳健
- **长对话并非普遍问题**：部分模型（如 Claude、Nova Pro）能较好地处理长对话，而另一些模型则随上下文增长明显退化，反映了不同模型在长上下文能力上的差异
- **最优模型也仅 40%**：此结果说明对话式 function-calling 仍然是一个远未解决的挑战，目前的 SOTA 距离实际可用还有很大差距
- **Dialog act 分析揭示了"对话管理盲区"**：模型更擅长执行明确指令，但在需要主动澄清、确认或引导对话方向时表现不佳

## 亮点与洞察

- **对话复杂性维度的系统化设计是最大亮点**：不同于简单的 API 调用测试，CONFETTI 明确定义了 follow-ups、goal correction、goal switching 等维度，为 function-calling 评测提供了更贴近真实场景的框架。这套维度分类法可以指导未来工具调用相关数据集的构建
- **Off-policy turn-level 评估方法很实用**：解决了多轮评估中误差累积的公平性问题，使得不同模型可以在完全相同的上下文下被比较。这种评估范式值得推广到其他多轮交互评测场景
- **链式调用的低分是重要的社区信号**：揭示了当前 LLM function-calling 能力的真正瓶颈不在单次调用，而在多步规划和连续执行，为后续研究指明了方向

## 局限与展望

- **数据规模较小**：109 段对话、313 个 turn 的规模限制了统计显著性和对长尾场景的覆盖
- **API 覆盖面有限**：86 个 API 虽然不少，但相比真实世界中工具生态的多样性还远远不够
- **缺少训练数据配套**：CONFETTI 仅是评测基准，没有提供对应的训练数据或改进方法
- **评估仍以参考匹配为主**：对函数调用的评估主要是与参考标注对比，但在多轮对话中可能存在多种合理的调用方案，基于参考匹配可能低估了某些合理的替代方案
- **模型版本时效性**：评估的模型（如 Claude Sonnet v3.5、Llama 3.1 等）已有更新版本，结果可能不反映最新水平

## 相关工作与启发

- **vs BFCL (Berkeley Function-Calling Leaderboard)**: BFCL 主要关注单轮函数调用的准确性，CONFETTI 则强调多轮对话中的复杂交互场景。两者互补——BFCL 测基础能力，CONFETTI 测应用能力
- **vs ToolBench**: ToolBench 规模更大但自动生成质量较低，CONFETTI 规模虽小但人工精心设计、标注质量高。CONFETTI 的对话复杂性维度设计更系统
- **vs API-Bench**: API-Bench 侧重API选择准确性，缺少对话管理维度的评估。CONFETTI 的 dialog act 标注弥补了这一空白
- **启发**：CONFETTI 的评估框架（复杂性维度 + turn-level off-policy + dialog act）可以作为构建更大规模对话式 agent 评测基准的蓝图

## 评分

- 新颖性: ⭐⭐⭐⭐ 多维对话复杂性设计和 dialog act 评估在 function-calling benchmark 中属首创，思路新颖
- 实验充分度: ⭐⭐⭐⭐ 评估了多种 SOTA 模型且分析维度丰富，但数据规模偏小影响统计力度
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义到位，评测结果分析有深度
- 价值: ⭐⭐⭐⭐ 填补了对话式 function-calling 评测的空白，链式调用的发现对社区很有价值，但缺少改进方案略减实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Spotting Out-of-Character Behavior: Atomic-Level Evaluation of Persona Fidelity in Open-Ended Generation](spotting_out-of-character_behavior_atomic-level_evaluation_of_persona_fidelity_i.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ACL 2025\] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction](visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)
- [\[ACL 2025\] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs](a_practical_approach_for_building_production-grade_conversational_agents_with_wo.md)
- [\[ICML 2025\] Regression for the Mean: Auto-Evaluation and Inference with Few Labels through Post-hoc Regression](../../ICML2025/others/regression_for_the_mean_auto-evaluation_and_inference_with_few_labels_through_po.md)

</div>

<!-- RELATED:END -->
