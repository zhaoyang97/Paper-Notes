---
title: >-
  [论文解读] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning
description: >-
  [NeurIPS 2025][机器人][神经符号推理] NeSyPr提出了一种神经符号程序化框架，通过将符号规划器生成的任务计划转化为可组合的程序化表示，使紧凑的语言模型在无需外部符号引导的情况下实现高效的单步推理，类似人类的知识编译过程。 领域现状：大语言模型在具身任务中已展现出强大的推理能力，但通常依赖在线访问大规模推理…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "神经符号推理"
  - "知识编译"
  - "具身智能"
  - "程序化知识"
  - "语言模型"
---

# NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2510.19429](https://arxiv.org/abs/2510.19429)  
**代码**: 无  
**领域**: 机器人 / LLM Agent  
**关键词**: 神经符号推理, 知识编译, 具身智能, 程序化知识, 语言模型

## 一句话总结
NeSyPr提出了一种神经符号程序化框架，通过将符号规划器生成的任务计划转化为可组合的程序化表示，使紧凑的语言模型在无需外部符号引导的情况下实现高效的单步推理，类似人类的知识编译过程。

## 研究背景与动机

**领域现状**：大语言模型在具身任务中已展现出强大的推理能力，但通常依赖在线访问大规模推理引擎或符号规划器。现有方法如SayCan、Inner Monologue等要么需要API访问大模型，要么需要在线符号规划。

**现有痛点**：（1）在动态物理环境中，延迟、连接和资源限制使得在线访问大模型或符号规划器变得不可行；（2）小模型直接做多步推理能力不足；（3）符号规划器虽然推理精确但缺乏泛化能力和自然语言理解。

**核心矛盾**：高质量推理需要大模型或符号规划器的当前支持，但部署环境要求低延迟、轻量级。

**本文目标**：让紧凑的LM获得结构化、自适应和及时的推理能力，无需在线依赖外部系统。

**切入角度**：类比人类的知识编译（proceduralization）——人类将声明性知识通过练习转化为自动化的程序性知识。同样地，将符号规划器的声明性知识编译为LM可执行的程序性知识。

**核心 idea**：用符号工具生成任务特定计划，然后将计划转化为可组合的程序化表示嵌入LM推理过程，使多步符号推理被压缩为单步LM推理。

## 方法详解

### 整体框架
分为编译阶段和部署阶段。编译阶段：符号规划器利用声明性知识生成任务特定计划 → 将计划转化为程序化表示（编码隐含的产生式规则）→ 将程序化表示注入LM。部署阶段：LM在每一步直接生成动作，无需调用符号规划器。

### 关键设计

1. **符号计划的显式生成**:

    - 功能：利用符号规划器精确生成任务解决方案
    - 核心思路：用PDDL等形式化语言描述任务域，符号规划器（如Fast Downward）根据初始状态和目标生成最优计划序列。计划包含精确的动作序列和前提条件
    - 设计动机：符号规划器在结构化推理上远优于LM，先用它生成高质量计划

2. **程序化知识转换**:

    - 功能：将显式的计划转化为隐式的产生式规则表示
    - 核心思路：将计划中每一步的"状态→动作"映射转化为可组合的程序化表示。这些表示编码了"如果当前状态满足条件X，则执行动作Y"的产生式规则。多个程序化表示可以组合形成新计划的推理链
    - 设计动机：声明性知识（"做什么"）需要转化为程序性知识（"怎么做"），这种转化使知识能无缝集成到LM推理

3. **单步LM推理替代多步符号推理**:

    - 功能：实现高效的测试时推理
    - 核心思路：程序化表示被注入到LM的推理过程中（通过prompt或微调）。在部署时，LM给定当前观察直接输出下一步动作——原来需要多步符号推理的过程被压缩为一步LM前向传播。这类似于人类熟练后的自动化技能执行
    - 设计动机：消除在线符号依赖，实现低延迟部署

### 损失函数 / 训练策略
程序化知识可以通过prompt injection（无需训练）或微调注入LM。微调时使用标准的next-token prediction损失。

## 实验关键数据

### 主实验

| Benchmark | 指标 | NeSyPr | 大模型推理 | 符号规划器 | 说明 |
|-----------|------|--------|-----------|-----------|------|
| PDDLGym | 成功率 | 高 | 较高 | 最高 | 紧凑LM接近符号精度 |
| VirtualHome | 成功率 | 高 | 较高 | 高 | 日常任务推理 |
| ALFWorld | 成功率 | 高 | 高 | 高 | 文本游戏环境 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| NeSyPr (完整) | 最佳 | 程序化+组合 |
| 无程序化 | 下降 | 紧凑LM直接推理 |
| 无组合性 | 下降 | 单一计划不可迁移 |
| 在线符号规划 | 相当但延迟高 | NeSyPr无延迟开销 |

### 关键发现
- NeSyPr使紧凑LM达到接近大模型和符号规划器的推理能力
- 程序化转换是关键——将多步推理压缩为单步大幅降低了推理延迟
- 组合性使得在新任务上也能有效推理，而非仅限于见过的任务

## 亮点与洞察
- **知识编译的类比**：将认知科学中的程序化知识概念引入LLM推理，这个跨学科视角非常有启发性。类似地可以将其他人类认知机制（如概念组合、类比推理）编译到LM中
- **消除在线依赖**：对于延迟敏感的机器人应用（如实时操作），消除对外部API的依赖是实际需求
- **紧凑模型的赋能**：证明了小模型通过知识编译也能获得强推理能力，对边缘部署有重要意义

## 局限与展望
- 编译阶段仍然需要符号规划器，本身有域工程成本
- 对于符号规划器无法求解的任务（如开放世界推理），NeSyPr也无法处理
- 程序化表示的可组合性在极复杂任务上的上限有待探索
- 仅在模拟环境验证，真实机器人上的部署效果未知

## 相关工作与启发
- **vs SayCan (Ahn et al. 2022)**：SayCan需要在线LLM推理，NeSyPr将推理预编译到小模型中
- **vs 符号AI (STRIPS/PDDL)**：符号系统推理精确但不灵活，NeSyPr通过LM赋予灵活性
- **vs ReAct**：ReAct在每一步都需要LLM推理，NeSyPr将推理压缩为单步

## 评分
- 新颖性: ⭐⭐⭐⭐ 知识编译的思路新颖有深度
- 实验充分度: ⭐⭐⭐⭐ 三个benchmark系统验证
- 写作质量: ⭐⭐⭐⭐ 概念解释清晰
- 价值: ⭐⭐⭐⭐ 对边缘部署和高效推理有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[CVPR 2026\] Cross-Domain Demo-to-Code via Neurosymbolic Counterfactual Reasoning](../../CVPR2026/robotics/cross-domain_demo-to-code_via_neurosymbolic_counterfactual_reasoning.md)
- [\[NeurIPS 2025\] DynaNav: Dynamic Feature and Layer Selection for Efficient Visual Navigation](dynanav_dynamic_feature_and_layer_selection_for_efficient_visual_navigation.md)
- [\[NeurIPS 2025\] EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT](egothinker_unveiling_egocentric_reasoning_with_spatiotempora.md)

</div>

<!-- RELATED:END -->
