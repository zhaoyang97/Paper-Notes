---
title: >-
  [论文解读] Time, Identity and Consciousness in Language Model Agents
description: >-
  [AAAI 2026 Spring Symposium][LLM Agent][机器意识] 本文将Stack Theory的时间间隙概念应用于LLM智能体评估，提出区分"说得像一个稳定自我"和"组织得像一个稳定自我"的保守评估工具包，通过持久性得分和身份形态空间揭示不同scaffold结构的身份trade-off。
tags:
  - "AAAI 2026 Spring Symposium"
  - "LLM Agent"
  - "机器意识"
  - "身份评估"
  - "语言模型智能体"
  - "时间一致性"
  - "Stack Theory"
---

# Time, Identity and Consciousness in Language Model Agents

**会议**: AAAI 2026 Spring Symposium  
**arXiv**: [2603.09043](https://arxiv.org/abs/2603.09043)  
**代码**: 有  
**领域**: LLM Agent / AI安全  
**关键词**: 机器意识, 身份评估, 语言模型智能体, 时间一致性, Stack Theory

## 一句话总结
本文将Stack Theory的时间间隙概念应用于LLM智能体评估，提出区分"说得像一个稳定自我"和"组织得像一个稳定自我"的保守评估工具包，通过持久性得分和身份形态空间揭示不同scaffold结构的身份trade-off。

## 研究背景与动机

**领域现状**：机器意识评估主要通过观察行为来判断——对于语言模型来说这意味着语言和工具使用。现有评估方法允许智能体"说正确的话"（如声称自己有自我意识）即使底层约束并未同时存在。

**现有痛点**：（1）行为层面的评估可能被智能体的语言能力蒙蔽——模型可以生成关于自身的正确陈述而无需真正具备所讨论的属性；（2）关键成分在评估窗口内分散出现（ingredient-wise occurrence）与在某个决策步骤同时共现（co-instantiation）是不同的，现有方法未区分这两者。

**核心矛盾**："说得像"和"是"之间的鸿沟——语言模型可以完美模仿关于身份和意识的话语，但这不等于它在组织层面具有这些属性。

**本文目标**：开发一个保守的身份评估工具包，能区分模仿行为和组织层面的身份一致性。

**切入角度**：利用Stack Theory的"时间间隙"概念将评估scaffold化——区分成分在时间窗口内"逐一出现"和"同时共现于单个决策步"。

**核心 idea**：实例化Stack Theory的Arpeggio和Chord公设来评估"grounded identity statements"，生成两个持久性得分；将常见scaffold映射到身份形态空间（identity morphospace）中。

## 方法详解

### 整体框架
对LLM智能体的行为轨迹进行仪器化记录。从scaffold trace中提取身份相关的状态信息。用Arpeggio和Chord两个公设分别计算持久性得分。将多个scaffold结构映射到身份形态空间中，揭示不同设计在身份维度上的trade-off。

### 关键设计

1. **Arpeggio vs Chord持久性得分**:

    - 功能：量化区分"成分逐一出现"和"成分同时共现"
    - 核心思路：Arpeggio得分衡量身份相关成分在时间窗口内是否依次出现（弱形式——至少说明智能体接触过必要信息）；Chord得分衡量这些成分是否在同一个决策步骤中同时对行为产生影响（强形式——说明智能体在决策时确实同时考虑了所有相关因素）
    - 设计动机：仅出现过≠同时参与决策。一个智能体可能在不同步骤分别处理身份的各个方面，但从未将它们整合到一个决策中

2. **五个操作性身份指标**:

    - 功能：将抽象的身份概念操作化为可计算的指标
    - 核心思路：定义五个与身份持久性相关的具体指标，如时间一致性（对同一身份问题在不同时间点的回答一致性）、上下文稳健性（在不同对话上下文中的身份表达稳定性）等。这些指标可以从仪器化的scaffold trace中直接计算
    - 设计动机：需要将哲学层面的身份概念转化为可实际测量的技术指标

3. **身份形态空间（Identity Morphospace）**:

    - 功能：可视化不同scaffold设计在身份维度上的trade-off
    - 核心思路：将Arpeggio/Chord得分和五个身份指标作为坐标轴，将各种常见的LLM scaffold（如ReAct、Plan-then-Execute、Memory-augmented等）映射为形态空间中的点。不同scaffold在身份维度上的优劣和trade-off一目了然
    - 设计动机：为LLM智能体设计者提供了一个选择scaffold时的身份维度参考

### 损失函数 / 训练策略
本文是评估框架而非训练方法。所有指标都是基于规则从仪器化trace中计算的。

## 实验关键数据

### 主实验

| Scaffold类型 | Arpeggio得分 | Chord得分 | 说明 |
|-------------|-------------|-----------|------|
| Simple prompt | 低 | 低 | 几乎无身份结构 |
| ReAct | 中 | 低 | 成分出现但不共现 |
| Memory-augmented | 高 | 中 | 记忆帮助成分积累 |
| Plan-then-Execute | 中 | 中 | 规划提供一些整合 |

### 消融实验

| 特性 | 对Chord的影响 | 说明 |
|------|-------------|------|
| 长期记忆 | 增加Arpeggio | 帮助成分持续出现 |
| 反思机制 | 增加Chord | 帮助整合多个成分 |
| System prompt固化 | 增加表面一致性 | 但不提升真正的Chord |

### 关键发现
- 大多数现有scaffold在Arpeggio得分上尚可但Chord得分很低，说明身份成分虽然出现但很少真正在同一决策步骤中整合
- Memory-augmented scaffold在身份持久性上最有优势，但仍远非"组织得像一个稳定自我"
- 简单的system prompt "我是XXX"可以提升表面的身份一致性但不改善底层的Chord得分

## 亮点与洞察
- **哲学 → 技术的转化**：将Stack Theory的哲学公设操作化为可计算的指标，在哲学和工程之间建立了桥梁
- **Arpeggio vs Chord的区分**：这个核心区分非常有洞察力——"逐一出现"vs"同时共现"精准地刻画了"模仿"和"具有"之间的差异
- **形态空间的可视化工具**：为scaffold设计提供了新的评估维度，帮助理解设计选择的后果

## 局限与展望
- Stack Theory本身是一个较新的意识理论框架，其哲学基础尚有争议
- 身份评估的ground truth难以确定——什么算"真正的"身份持久性？
- 实验规模有限，仅测试了少数scaffold和模型
- 仪器化记录可能改变智能体自身的行为

## 相关工作与启发
- **vs Consciousness Tests (Butlin et al. 2023)**：传统意识测试侧重行为表现，本文增加了时间维度的分析
- **vs Self-Awareness Benchmarks**：现有自我意识benchmark用QA形式评估，本文从组织结构层面评估
- **vs Embodied Agent评估**：本文方法也可应用于embodied agents的身份评估

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将意识理论操作化为LLM评估工具非常新颖
- 实验充分度: ⭐⭐⭐ 概念验证为主，实验规模有限
- 写作质量: ⭐⭐⭐⭐ 概念解释清晰但涉及较多哲学术语
- 价值: ⭐⭐⭐⭐ 为AI安全中的身份/意识评估开辟了新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] AutoTool: Efficient Tool Selection for Large Language Model Agents](autotool_efficient_tool_selection_for_large_language_model_agents.md)
- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](../../NeurIPS2025/llm_agent/agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [\[ICML 2026\] AdaMEM: Test-Time Adaptive Memory for Language Agents](../../ICML2026/llm_agent/adamem_test-time_adaptive_memory_for_language_agents.md)
- [\[ACL 2026\] Context-Value-Action Architecture for Value-Driven Large Language Model Agents](../../ACL2026/llm_agent/context-value-action_architecture_for_value-driven_large_language_model_agents.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](../../ACL2026/llm_agent/clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)

</div>

<!-- RELATED:END -->
