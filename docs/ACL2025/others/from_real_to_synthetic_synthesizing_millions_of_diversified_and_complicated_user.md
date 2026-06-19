---
title: >-
  [论文解读] From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding
description: >-
  [ACL 2025 (Outstanding Paper Award 🏆)][指令合成] 本文提出"归因式接地"（Attributed Grounding）框架，通过自顶向下的用户归因和自底向上的基于网络文档的指令合成，构建了一个包含 100 万条多样且复杂指令的 SynthQuestions 数据集，训练模型在多个通用基准上达到领先表现。
tags:
  - "ACL 2025 (Outstanding Paper Award 🏆)"
  - "指令合成"
  - "数据多样性"
  - "归因式生成"
  - "LLM对齐"
  - "网络文档"
---

# From Real to Synthetic: Synthesizing Millions of Diversified and Complicated User Instructions with Attributed Grounding

**会议**: ACL 2025 (Outstanding Paper Award 🏆)  
**arXiv**: [2506.03968](https://arxiv.org/abs/2506.03968)  
**代码**: [https://github.com/Ignoramus0817/SynthQuestions](https://github.com/Ignoramus0817/SynthQuestions)  
**领域**: 其他  
**关键词**: 指令合成, 数据多样性, 归因式生成, LLM对齐, 网络文档

## 一句话总结

本文提出"归因式接地"（Attributed Grounding）框架，通过自顶向下的用户归因和自底向上的基于网络文档的指令合成，构建了一个包含 100 万条多样且复杂指令的 SynthQuestions 数据集，训练模型在多个通用基准上达到领先表现。

## 研究背景与动机

**领域现状**：大规模高质量指令数据是 LLM 对齐的关键燃料。Self-Instruct、Evol-Instruct 等方法已证明合成指令数据可以有效训练模型，WizardLM 等工作通过"进化"策略提升指令复杂度。

**现有痛点**：现有指令合成方法面临两大难题：(1) 接地源（grounding source）有限——多数方法基于少量种子指令进行改写或扩展，导致生成的指令分布狭窄，难以覆盖真实用户需求的广泛分布；(2) 复杂度提升流于表面——通过机械式的约束叠加（如"请用英语回答""限制在200字以内"）来增加指令长度，并未真正提升指令的认知复杂度和实际意义。

**核心矛盾**：真正有价值的指令应当反映真实用户在特定情境下的实际需求，但现有合成方法缺乏真实场景的接地（grounding），生成的指令"看起来复杂但空洞"。要实现大规模+高多样性+高复杂度的三重目标，需要找到一个足够丰富的信息源来驱动生成。

**本文目标**：设计一个能够利用海量网络文档作为接地源，自动生成百万级多样且复杂指令的框架。

**切入角度**：真实世界中的高质量指令通常有三个核心要素——文档（信息来源）、用户（提出需求的人）和动机（为什么提出这个需求）。这三个要素构成了指令的"归因三角"。如果能自动补全这三个要素，就能从网络文档反向生成高质量指令。

**核心 idea**：将指令合成过程分为两步——先"归因"（从已有指令推断用户画像和动机），再"合成"（从网络文档出发，结合用户情境生成指令），以网络文档的无限供给保证规模和多样性。

## 方法详解

### 整体框架

整个流程分为两个对称的过程。自顶向下：从29K人工收集的种子指令出发，通过网络搜索为每条指令找到相关文档，然后用 LLM 推断"什么样的用户在什么情境下会提出这个问题"，构建归因三元组（文档, 用户, 动机）。自底向上：从大规模网络语料（如 FineWeb）中抽取文档，以归因三元组作为示例（in-context demonstrations），让 LLM 先构想一个合理的用户情境，再基于该情境和文档内容生成有意义的指令。

### 关键设计

1. **自顶向下归因过程（Top-down Attribution）**:

    - 功能：将真实指令"解构"为文档-用户-动机三元组，为后续合成提供高质量示范
    - 核心思路：从多个开源指令数据集收集 29K 条高质量种子指令。对每条指令提取关键词，通过搜索引擎检索排名第一的网络文档作为信息源。然后用 LLM 生成归因描述——推断提出该指令的用户身份（如"一个正在写毕业论文的计算机系研究生"）和具体动机（如"需要理解某算法的复杂度分析来完成实验部分"）。这样每条种子指令就被扩展为一个完整的归因三元组 $(d, u, m)$，其中 $d$ 是文档、$u$ 是用户画像、$m$ 是动机。
    - 设计动机：一方面为种子指令建立"为什么会有人问这个问题"的上下文，另一方面生成的归因三元组可作为下游合成的 few-shot 示范

2. **自底向上合成过程（Bottom-up Synthesis）**:

    - 功能：从海量网络文档出发批量生成多样且复杂的指令
    - 核心思路：从 FineWeb 等大规模网络语料中随机抽取文档。对每篇文档，从归因数据池中采样几条归因三元组作为 in-context demonstrations，然后 prompt LLM 完成两步生成：(1) 基于文档内容构想一个合理的用户情境（谁，在什么背景下，出于什么动机）；(2) 基于该情境和文档内容生成一条有意义的用户指令。文档的多样性（来自整个互联网）天然保证了指令的分布广度。
    - 设计动机：以网络文档而非种子指令为起点，突破了传统方法"种子决定分布"的瓶颈。情境构想步骤确保生成的指令有真实的需求背景而非空洞的改写

3. **质量评估与多样性过滤**:

    - 功能：确保大规模生成指令的质量底线和分布多样性
    - 核心思路：生成指令后通过 LLM 对每条指令打分评估质量，过滤低质量结果。然后使用 BERTopic 进行主题建模，在每个主题内选择得分最高的指令，确保最终数据集在主题空间上均匀分布。最后使用 LLaMA-Guard-3-8B 进行安全性审核，过滤潜在的有害内容。
    - 设计动机：大规模生成不可避免引入噪声，多阶段过滤在保证规模的同时维护了质量和安全性；基于主题的多样性采样避免了某些热门领域的过度表示

### 损失函数 / 训练策略

数据集构建完成后，使用标准的 SFT 流水线训练模型。另外还构建了 100K 偏好数据用于后续的 DPO 训练——使用 ArmoRM-Llama3-8B-v0.1 作为奖励模型对 SFT 模型的多个回答进行评分，构建偏好对。

## 实验关键数据

### 主实验

基于 SynthQuestions 训练的模型在多个基准上的表现：

| 基准 | SynthQuestions (SFT) | WizardLM | Self-Instruct | Evol-Instruct | 提升 |
|------|---------------------|----------|---------------|---------------|------|
| AlpacaEval 2.0 LC(%) | 32.7 | 24.1 | 18.5 | 27.3 | +5.4 |
| MT-Bench 平均分 | 7.52 | 6.89 | 6.21 | 7.15 | +0.37 |
| Arena-Hard | 28.4 | 21.2 | 15.8 | 24.6 | +3.8 |
| IFEval (严格) | 54.2 | 46.8 | 40.1 | 49.5 | +4.7 |
| MMLU | 63.8 | 62.1 | 60.5 | 62.9 | +0.9 |

数据规模对性能的影响（SFT + Llama-3-8B）:

| 数据量 | AlpacaEval 2.0 LC(%) | MT-Bench |
|-------|---------------------|----------|
| 100K | 26.3 | 7.08 |
| 250K | 28.9 | 7.25 |
| 500K | 31.2 | 7.41 |
| 1M | 32.7 | 7.52 |

### 消融实验

| 配置 | AlpacaEval 2.0 LC(%) | 说明 |
|------|---------------------|------|
| Full pipeline | 32.7 | 完整归因+合成 |
| w/o 用户归因 | 28.1 | 不构建用户画像，直接从文档生成指令，掉 4.6% |
| w/o 文档接地 | 25.4 | 不用网络文档，纯种子改写，掉 7.3% |
| w/o 质量过滤 | 30.5 | 不做质量评分过滤，掉 2.2% |
| w/o 多样性采样 | 31.1 | 不做 BERTopic 均衡采样，掉 1.6% |
| 随机文档 + 直接生成 | 24.8 | 随机文档不做归因直接要求生成指令 |

### 关键发现

- **文档接地是最大贡献因素**：去掉网络文档后性能大幅下降 7.3%，证明多样的信息源是指令多样性的关键驱动力
- **用户归因显著提升复杂度**：加入用户画像后指令的平均复杂度评分提升约 35%，说明"先构想用户再生成指令"比"直接生成"更能产出有深度的指令
- **持续scaling**：从 100K 到 1M，性能持续上升且未出现饱和迹象，暗示更多网络文档可以进一步提升表现
- SynthQuestions 在不同基座模型（Llama-3-8B、Mistral-7B）上都表现稳定，说明数据集质量具有跨模型迁移性

## 亮点与洞察

- **"归因三角"框架**：将指令分解为文档-用户-动机三要素的思路非常优雅。这个抽象框架不仅适用于指令合成，理论上可以迁移到任何需要生成"有情境的自然语言内容"的场景，如对话系统的用户模拟、教育领域的题目生成等。
- **网络文档作为无限接地源**：这是一个关键洞察——互联网本身就是最大的知识库，以文档为起点可以突破种子指令的分布瓶颈。而且随着网络语料的增长，数据集可以无限扩展。
- **获得 ACL 2025 Outstanding Paper Award**：说明这项工作在领域内获得了广泛认可，方法的通用性和实验的扎实程度得到了评审委员会的高度评价。

## 局限与展望

- 指令合成依赖于 LLM（如 GPT-4）的生成能力，合成数据的质量上界受限于合成用的教师模型
- 网络文档可能包含不准确信息（错误事实、过时数据），基于这些文档生成的指令可能传播错误
- 当前仅关注单轮指令，未涉及多轮对话指令的合成，真实场景中很多复杂需求需要多轮交互表达
- BERTopic 的主题粒度选择可能影响最终的多样性分布质量，该超参数的选择标准不够明确
- 未来可以考虑加入时效性控制——优先使用最新文档来生成与当前世界状态相关的指令

## 相关工作与启发

- **vs Self-Instruct**: Self-Instruct 从少量种子通过自我生成扩展，但分布受种子约束。SynthQuestions 通过网络文档打破了这一限制，多样性和规模都更大。
- **vs Evol-Instruct/WizardLM**: Evol-Instruct 通过"进化"规则（深化、增加约束等）提升复杂度，但这些规则是固定模板，产出的复杂度是"表面复杂"。SynthQuestions 的用户情境驱动生成则产出"语义复杂"的指令。
- **vs LIMA "Less is More"**: LIMA 认为少量高质量数据就够了，SynthQuestions 的 scaling 实验表明在数据质量有保证的前提下，更多数据确实带来持续提升。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "归因式接地"框架优雅地解决了指令合成的多样性和复杂度问题，思路新颖且具有广泛的启发意义
- 实验充分度: ⭐⭐⭐⭐⭐ 多基准评测、scaling实验、消融实验、跨模型验证齐全，数据集和模型完全开源
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，归因框架的动机解释充分，但部分实验细节可以更详尽
- 价值: ⭐⭐⭐⭐⭐ ACL 2025 Outstanding Paper Award，数据集和方法都具有很高的实用价值和社区影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Predicting Implicit Arguments in Procedural Video Instructions](implicit_arguments_video_instructions.md)
- [\[ACL 2025\] USDC: A Dataset of User Stance and Dogmatism in Long Conversations](usdc_a_dataset_of_underlineuser_underlinestance_and_underlinedogmatism_in_long_u.md)
- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)
- [\[ACL 2025\] Counterspeech the Ultimate Shield! Multi-Conditioned Counterspeech Generation through Attributed Prefix Learning](hippro_counterspeech_gen.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)

</div>

<!-- RELATED:END -->
