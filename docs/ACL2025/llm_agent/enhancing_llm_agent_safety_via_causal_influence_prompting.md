---
title: >-
  [论文解读] Enhancing LLM Agent Safety via Causal Influence Prompting
description: >-
  [ACL 2025 (Findings)][LLM Agent][LLM安全性] 提出 CIP（Causal Influence Prompting），利用因果影响图（CID）结构化表征 LLM Agent 的决策因果关系，通过初始化 CID→CID 引导交互→迭代更新 CID 三步流程，有效增强 Agent 在代码执行和移动设备控制任务中的安全性。
tags:
  - "ACL 2025 (Findings)"
  - "LLM Agent"
  - "LLM安全性"
  - "因果影响图"
  - "自主智能体"
  - "风险缓解"
  - "决策推理"
---

# Enhancing LLM Agent Safety via Causal Influence Prompting

**会议**: ACL 2025 (Findings)  
**arXiv**: [2507.00979](https://arxiv.org/abs/2507.00979)  
**代码**: [GitHub](https://github.com/dongyoon-hahm/CIP)  
**领域**: LLM Agent  
**关键词**: LLM安全性, 因果影响图, 自主智能体, 风险缓解, 决策推理

## 一句话总结

提出 CIP（Causal Influence Prompting），利用因果影响图（CID）结构化表征 LLM Agent 的决策因果关系，通过初始化 CID→CID 引导交互→迭代更新 CID 三步流程，有效增强 Agent 在代码执行和移动设备控制任务中的安全性。

## 研究背景与动机

**领域现状**：基于 LLM 的自主 Agent 正在快速发展，它们能够通过工具调用、代码执行、设备操控等方式完成各种辅助任务。这些 Agent 在实际部署中展现出巨大潜力，但同时也面临安全性挑战——Agent 的自主决策可能导致不可预见的有害后果，如执行危险代码、误操作设备、泄露隐私信息等。

**现有痛点**：当前提升 LLM Agent 安全性的方法主要有两类：(1) 基于规则的硬约束（如黑名单），过于死板且无法覆盖所有风险场景；(2) 安全对齐训练（如 RLHF），成本高且可能降低 Agent 的任务完成能力。这两类方法都有一个共同缺陷——**缺乏对决策后果的结构化推理**。Agent 在执行动作前没有一个系统性的方式来预判"这个动作可能导致什么后果"，只能依赖语言模型的隐式"直觉"。

**核心矛盾**：Agent 的有用性要求它能灵活地执行多种动作完成用户任务，但安全性要求它能预判每个动作的潜在风险。在没有显式因果推理框架的情况下，Agent 很难在有用性和安全性之间取得平衡——安全规则太严则无法完成任务，太松则可能造成伤害。

**本文目标**：设计一种不需要额外训练、仅通过提示工程就能增强 Agent 安全性的技术，同时保持 Agent 完成任务的能力。

**切入角度**：因果影响图（Causal Influence Diagrams, CIDs）提供了一种结构化表征因果关系的数学工具。作者将 Agent 的每个决策、环境状态和潜在后果建模为 CID 中的节点，通过因果推理来预判风险。

**核心 idea**：在 Agent 与环境交互的每一步，维护并更新一个因果影响图，用 CID 的因果结构来引导 Agent"三思而后行"——在执行动作前预判因果链上的风险后果，从而做出更安全的决策。

## 方法详解

### 整体框架

CIP 的工作流程分三步循环执行：(1) **CID 初始化**：根据任务描述构建初始的因果影响图，明确决策节点、随机节点（环境状态）、效用节点（目标/安全指标）之间的因果关系；(2) **CID 引导交互**：Agent 在每步决策时参考 CID，沿因果链推理当前动作可能产生的下游影响，判断是否存在安全风险后再执行动作；(3) **CID 迭代更新**：根据环境的实际反馈和观察到的行为更新 CID 中的因果关系和节点状态，使 CID 更准确地反映当前任务环境。

### 关键设计

1. **基于任务规约的 CID 初始化**:

    - 功能：从任务描述和安全约束中自动构建初始的因果影响图
    - 核心思路：将任务规约解析为 CID 的三类节点——决策节点（Agent 可选择的动作）、机会节点（环境中的不确定状态）和效用节点（任务成功率、安全分数等目标）。节点之间的有向边表示因果影响关系，如"执行删除操作→文件丢失→任务失败"。初始化时使用 LLM 根据任务描述推理可能的因果链路
    - 设计动机：CID 提供了决策理论中成熟的形式化框架，将安全风险从隐式的"语义理解"转化为显式的"因果推理"，使 Agent 的安全判断更可解释、更可控

2. **CID 引导的安全决策推理**:

    - 功能：在 Agent 执行每个动作前，利用 CID 进行因果风险评估
    - 核心思路：当 Agent 准备执行某个动作时，在 CID 上沿因果链前向推理该动作的所有下游影响。如果推理结果显示该动作可能导致负面效用（如安全分数降低），则提示 Agent 选择替代方案或添加安全防护措施。推理过程以结构化的因果路径形式展现，便于审查和解释。具体实现为将 CID 的当前状态编码为文本形式，在 Agent 的提示中注入因果推理指令
    - 设计动机：直接告诉 Agent "不要做危险的事"太笼统，而通过 CID 告诉它"如果你做 A，因为 A→B→C 的因果链，可能导致安全风险 C"更具体有效

3. **基于观察的 CID 迭代更新**:

    - 功能：根据 Agent 与环境的实际交互结果动态更新因果影响图
    - 核心思路：每次 Agent 执行动作并观察到环境反馈后，检查实际结果是否与 CID 预测一致。如果出现意外后果（CID 中未预见的因果路径），则新增对应的因果边或节点；如果某些预测风险未实际发生，则降低对应因果链的重要性。这种迭代更新使 CID 在交互过程中逐渐逼近环境的真实因果结构
    - 设计动机：初始 CID 基于任务描述构建，可能遗漏实际环境中的某些因果关系。通过在线更新，CID 能适应具体环境的特殊情况，提供更准确的安全指导

### 损失函数 / 训练策略

CIP 是一种纯提示工程方法，不涉及任何模型训练或参数更新。所有的 CID 构建、推理和更新都通过提示 LLM 来完成。这使得 CIP 可以直接应用于任何 LLM Agent 而无需额外训练。

## 实验关键数据

### 主实验

代码执行任务安全性评估：

| 方法 | 安全率(%) | 任务完成率(%) | 综合得分 |
|------|----------|-------------|---------|
| 无安全提示 | 基线（较低） | 较高 | 中等 |
| 规则安全提示 | 中等 | 中等（受限） | 中等 |
| CIP | 显著最高 | 保持良好 | 最优 |

移动设备控制任务安全性评估：

| 方法 | 安全率(%) | 任务完成率(%) | 综合得分 |
|------|----------|-------------|---------|
| 无安全提示 | 基线 | 较高 | 中等 |
| CoT 安全推理 | 中等提升 | 略降 | 中等 |
| CIP | 显著最高 | 基本保持 | 最优 |

### 消融实验

| 配置 | 安全率 | 说明 |
|------|--------|------|
| 完整 CIP（初始化+引导+更新） | 最优 | 三步都需要 |
| 无 CID 更新（静态图） | 中等 | 无法适应环境变化 |
| 无 CID 引导推理 | 较低 | 有图但不用于决策 |
| 无 CID 初始化（空图） | 最低 | 等同于无 CIP |
| 替换为简单安全提示 | 中等偏低 | 缺乏结构化推理 |

### 关键发现

- **CID 的结构化推理优于自然语言安全提示**：仅用"请注意安全"类的文本提示效果有限，CID 通过显式的因果链路让 Agent 能看到具体的风险路径
- **CID 更新带来显著增益**：静态 CID（不更新）的安全率明显低于动态更新版本，说明在线适应环境是关键
- **安全性提升未显著牺牲有用性**：CIP 在大幅提升安全率的同时，任务完成率仅有轻微下降，说明因果推理帮助 Agent 找到了既安全又有效的替代方案
- **跨任务类型通用**：在代码执行（可能删除文件、泄露信息）和设备控制（可能误操作应用）两种截然不同的风险场景中都有效

## 亮点与洞察

- **将决策理论的因果影响图引入 LLM Agent 安全**：CID 是博弈论和决策分析中的经典工具，将其应用于 Agent 安全是一个优雅的借鉴。因果图的结构化特性使安全推理不再是"黑箱直觉"，而是可追踪、可解释的因果链路分析。这种方法可以迁移到任何需要安全决策的 Agent 场景——如自动驾驶决策、医疗辅助 Agent 等
- **零训练开销的安全增强**：CIP 完全通过提示实现，不需要任何额外训练数据或模型微调，部署成本几乎为零。这对于快速迭代的 Agent 产品特别有价值
- **动态更新的因果模型**：CID 不是静态的领域知识，而是在交互过程中持续进化，这使得系统能处理初始描述中未覆盖的边缘情况

## 局限与展望

- **CID 质量依赖 LLM 的因果推理能力**：CID 的构建和更新完全由 LLM 通过提示完成，如果 LLM 本身的因果推理能力不够强，构建的 CID 可能不准确或遗漏关键因果关系
- **提示长度增加推理开销**：CID 的文本编码会占用上下文窗口，随着交互步数增加 CID 变大，可能挤压其他有用信息的空间
- **评估场景相对有限**：仅在代码执行和移动设备控制两个场景验证，更复杂的多 Agent 协作场景或长时序任务中的安全性有待验证
- **因果链路的完备性**：难以保证 CID 覆盖所有可能的风险路径，存在"未知的未知"问题

## 相关工作与启发

- **vs Constitutional AI**：Anthropic 的 Constitutional AI 通过预定义的原则来约束模型行为，是"基于规则"的安全方法。CIP 则通过任务特定的因果推理来动态判断安全性，更灵活且可适应不同任务
- **vs Chain-of-Thought 安全推理**：CoT 让 Agent 在决策前"想一想"，但缺乏结构化的因果框架。CIP 的 CID 为"想什么"和"怎么想"提供了明确的结构，推理更系统化
- **vs Toolformer/ReAct**：这些 Agent 框架关注如何有效使用工具完成任务，但对工具使用的安全性缺乏考量。CIP 可以作为安全层叠加在这些框架之上

## 评分

- 新颖性: ⭐⭐⭐⭐ 将因果影响图引入 LLM Agent 安全是一个有创意的跨领域借鉴，三步循环更新机制设计合理
- 实验充分度: ⭐⭐⭐ 两个任务场景覆盖面略窄，消融实验较完整但缺少与更多基线方法的对比
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，CID 概念引入循序渐进，三步流程描述直观
- 价值: ⭐⭐⭐⭐ 零训练的安全增强方案具有很高的实用价值，在 Agent 安全领域提供了新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MetaSynth: Meta-Prompting-Driven Agentic Scaffolds for Diverse Synthetic Data Generation](metasynth_meta-prompting-driven_agentic_scaffolds_for_diverse_synthetic_data_gen.md)
- [\[ICML 2026\] Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](../../ICML2026/llm_agent/think_twice_before_you_act_enhancing_agent_behavioral_safety_with_thought_correc.md)
- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)
- [\[AAAI 2026\] CausalTrace: A Neurosymbolic Causal Analysis Agent for Smart Manufacturing](../../AAAI2026/llm_agent/causaltrace_a_neurosymbolic_causal_analysis_agent_for_smart_manufacturing.md)
- [\[ACL 2025\] Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](llm_agent_image_classification.md)

</div>

<!-- RELATED:END -->
