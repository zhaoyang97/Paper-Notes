---
title: >-
  [论文解读] MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis
description: >-
  [ACL 2025 (main)][LLM Agent][鉴别诊断] 提出 MEDDxAgent 框架，通过中央编排器 DDxDriver 协调病史采集模拟器、知识检索智能体和诊断策略智能体三个模块进行迭代式鉴别诊断（DDx），在交互式诊断场景下实现超过 10% 的准确率提升，同时提供完整的推理可解释性。
tags:
  - "ACL 2025 (main)"
  - "LLM Agent"
  - "鉴别诊断"
  - "多模块智能体"
  - "迭代学习"
  - "可解释推理"
  - "医学LLM"
---

# MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis

**会议**: ACL 2025 (main)  
**arXiv**: [2502.19175](https://arxiv.org/abs/2502.19175)  
**代码**: [https://github.com/nec-research/meddxagent](https://github.com/nec-research/meddxagent)  
**领域**: Agent / 医学诊断  
**关键词**: 鉴别诊断、多模块智能体、迭代学习、可解释推理、医学LLM

## 一句话总结

提出 MEDDxAgent 框架，通过中央编排器 DDxDriver 协调病史采集模拟器、知识检索智能体和诊断策略智能体三个模块进行迭代式鉴别诊断（DDx），在交互式诊断场景下实现超过 10% 的准确率提升，同时提供完整的推理可解释性。

## 研究背景与动机

**领域现状**：鉴别诊断（DDx）是临床决策的核心环节，医生需要根据症状、病史和医学知识迭代地缩小可能的疾病范围。近年来 LLM 在辅助诊断方面展现了潜力，但现有方案距离真实临床场景仍有较大差距。

**现有痛点**：已有方法存在五个关键限制：(1) 仅在单一数据集上评估，泛化性不足；(2) 孤立地优化单个诊断组件（如仅优化诊断策略），缺乏多阶段整合；(3) 假设一开始就能获取完整的患者档案，这在现实中不成立——医生通常从年龄、性别、主诉等有限信息开始；(4) 缺乏迭代学习机制，无法通过多轮交互更新诊断结论；(5) 过度依赖医学QA基准，无法反映真实DDx任务的复杂性。

**核心矛盾**：真实的鉴别诊断是一个渐进式的信息收集和推理过程，而现有方法将其简化为一次性的"给定完整信息→输出诊断"范式，丧失了诊断过程中最关键的迭代探索能力。

**本文目标**：设计一个模块化、可解释的多智能体DDx框架，支持在患者信息逐步获取的交互式场景下进行迭代诊断。

**切入角度**：借鉴真实临床流程的三阶段结构（病史采集、知识检索、诊断策略），将其建模为三个可组合的智能体模块，通过 ReAct 范式的编排器统一协调。

**核心 idea**：将鉴别诊断建模为编排器驱动的多智能体迭代交互过程，其中每个模块独立可替换、整体协同提升诊断精度和可解释性。

## 方法详解

### 整体框架

MEDDxAgent 采用中心化架构：DDxDriver 编排器作为核心枢纽，管理三个功能模块——病史采集模拟器（History Taking Simulator）、知识检索智能体（Knowledge Retrieval Agent）和诊断策略智能体（Diagnosis Strategy Agent）。所有模块仅与 DDxDriver 通信，由后者负责信息维护、调度和迭代控制。整个系统遵循 ReAct（thought-action-observation）范式运行。

### 关键设计

1. **病史采集模拟器（History Taking Simulator）**:

    - 功能：模拟医患交互过程，在患者信息不完整时渐进地收集诊断所需数据
    - 核心思路：使用两个 LLM 分别扮演患者和医生。患者 LLM 持有完整档案，医生 LLM 从初始档案和 DDxDriver 提供的对话目标出发提问。交互持续直到目标达成或达到最大轮数限制，对话历史返回给 DDxDriver
    - 设计动机：真实场景下医生不可能一开始就拥有所有信息，必须通过追问来收集，这一模块使系统能处理信息不完整的现实诊断场景

2. **知识检索智能体与诊断策略智能体**:

    - 功能：知识检索从外部数据库（Wikipedia、PubMed）获取医学知识辅助诊断；诊断策略基于当前信息生成和排序候选诊断列表
    - 核心思路：知识检索智能体从 DDxDriver 构建的搜索查询中提取关键医学概念，在外部数据库中检索并生成证据摘要。诊断策略智能体支持三种模式——零样本、固定少样本和基于嵌入相似度的动态少样本，并可与 CoT 推理结合，利用 BioClinicalBERT 或 BGE 嵌入进行患者相似度匹配
    - 设计动机：外部知识对罕见病诊断至关重要（LLM 内部知识可能过时或不足）；动态少样本策略通过检索最相似的病例指导诊断，比固定样本更灵活

3. **DDxDriver 编排器与迭代学习机制**:

    - 功能：作为统一协调中心，管理患者档案、调度模块执行、记录推理轨迹、控制停止条件
    - 核心思路：支持两种迭代模式——固定迭代（按顺序循环执行三个模块直到预设轮数）和动态迭代（DDxDriver 根据当前观察自主决定下一步调用哪个模块）。每轮迭代后更新患者档案和候选诊断排名
    - 设计动机：固定迭代确保结构化的信息收集，动态迭代则允许系统灵活应对特殊情况（如发现罕见病需额外检索知识），两者互补

### 损失函数 / 训练策略

MEDDxAgent 是一个无需训练的推理框架，不涉及模型参数更新。所有模块通过 prompt 工程和 ReAct 范式协调工作，核心开销在推理时的多轮 LLM 调用。

## 实验关键数据

### 主实验（非交互式设置，完整患者档案）

| 数据集 | 方法 | GTPA@1 | GTPA@5 | Avg Rank |
|--------|------|--------|--------|----------|
| DDxPlus | Few-shot (CoT, Dyn_BAII) | 0.97 | 1.00 | 1.03 |
| iCraft-MD | Few-shot (CoT, Dyn_BERT) | 0.64 | 0.73 | 3.68 |
| RareBench | Few-shot (CoT, Dyn_BAII) | 0.82 | 0.88 | 2.11 |

### 交互式诊断对比（GPT-4o，无完整档案）

| 方法 | DDxPlus GTPA@1 | iCraft-MD GTPA@1 | RareBench GTPA@1 |
|------|----------------|-------------------|------------------|
| KR (n=0, 无交互) | 0.18 | 0.15 | 0.07 |
| DS (n=5) | 0.72 | 0.40 | 0.50 |
| MEDDx (iter=1, n=5) | 0.74 | 0.52 | 0.51 |
| MEDDx (iter=3, n=15) | **0.86** | **0.54** | 0.50 |
| Llama3.1-70B MEDDx (iter=2) | 0.71 | 0.37 | **0.48** |
| Llama3.1-8B MEDDx (iter=2) | 0.56 | 0.14 | 0.09 |

### 关键发现

- 无交互信息时（n=0），诊断准确率骤降（RareBench 的 GTPA@1 从 0.45 降到 0.07），验证了此前方法假设完整档案的不合理性
- 病史采集问答轮数增加到 10-15 次后收益趋于饱和，存在信息收集和诊断效率的最佳平衡点
- MEDDxAgent 在 DDxPlus 上甚至超越了使用完整档案的零样本基线（0.86 vs 0.69），说明迭代搜集+多模块协作可以弥补信息不足
- 固定迭代一致优于动态迭代，因为较小模型往往过度偏爱病史采集模块而忽视知识检索，导致冗余问诊
- 模型规模效应明显：GPT-4o > Llama3.1-70B >> Llama3.1-8B，小模型在 iCraft-MD 和 RareBench 上甚至无法有效利用迭代

## 亮点与洞察

- **模块化设计的可扩展性**：每个模块可独立替换和升级，这种"乐高式"架构使得框架能方便地适配新的数据集和 LLM，是医学 AI 系统工程化的良好范式
- **可解释性的实用价值**：DDxDriver 记录所有中间推理步骤和模块调用决策，这在医学场景中尤为关键——医生需要理解 AI 的诊断逻辑才能信任其建议
- **交互式评估范式的推动**：论文揭示了"完整档案假设"的虚假乐观，首次系统化地评估交互式 DDx，这对社区有重要的 benchmark 贡献

## 局限与展望

- 固定迭代优于动态迭代暴露了当前 LLM 在长周期规划和智能调度方面的不足，未来可引入强化学习来优化调度策略
- 仅评估了英文数据集和特定疾病类别，对多语言和更广泛疾病谱的泛化性尚未验证
- 推理成本较高——每个患者需要多轮 LLM 调用，实际部署需要考虑延迟和成本优化
- 模拟器中患者和医生均由 LLM 扮演，可能无法完全模拟真实医患交互的复杂性

## 相关工作与启发

- **vs AMIE (Tu et al., 2024)**: AMIE 也做交互式诊断但聚焦展示 LLM 单体能力，MEDDxAgent 则将问题分解为多模块协同，模块化程度更高
- **vs iCRAFT-MD (Li et al., 2024)**: iCRAFT-MD 提供了皮肤病数据集和交互评估框架，MEDDxAgent 在此基础上进一步整合了多模块和迭代学习
- 这篇论文对医学 AI Agent 的模块化设计思路值得参考——将复杂任务分解为可独立优化的子系统

## 评分

- 新颖性: ⭐⭐⭐⭐ 交互式DDx+多模块Agent的组合有新意，但各模块设计较常规
- 实验充分度: ⭐⭐⭐⭐⭐ 三数据集、三规模模型、多种设置的系统性评估非常扎实
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但篇幅较长、部分内容有重复
- 价值: ⭐⭐⭐⭐ 对医学AI Agent领域有实质性推动，揭示了交互式评估的重要性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration](mam_modular_multi-agent_framework_for_multi-modal_medical_diagnosis_via_role-spe.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2026\] LiTS: A Modular Framework for LLM Tree Search](../../ACL2026/llm_agent/lits_a_modular_framework_for_llm_tree_search.md)
- [\[ACL 2025\] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)
- [\[ICML 2025\] xChemAgents: Agentic AI for Explainable Quantum Chemistry](../../ICML2025/llm_agent/xchemagents_agentic_ai_for_explainable_quantum_chemistry.md)

</div>

<!-- RELATED:END -->
