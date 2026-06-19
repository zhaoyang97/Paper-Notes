---
title: >-
  [论文解读] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering
description: >-
  [AAAI 2026][信息检索/RAG][多跳问答] 提出 REAP 双模块迭代框架，通过子任务规划器 (SP) 维护全局视角动态指导推理轨迹，事实提取器 (FE) 从检索内容中提取结构化事实和潜在线索，两者递归协作解决多跳问答。在 4 个基准上以 Llama-3.1-8B 显著超越所有基线（HotpotQA F1 68.0 vs 次优 63.4）。
tags:
  - "AAAI 2026"
  - "信息检索/RAG"
  - "多跳问答"
  - "检索增强生成"
  - "自适应规划"
  - "事实提取"
  - "多任务微调"
---

# REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering

**会议**: AAAI 2026  
**arXiv**: [2511.09966](https://arxiv.org/abs/2511.09966)  
**代码**: [https://github.com/Deus-Glen/REAP](https://github.com/Deus-Glen/REAP)  
**领域**: NLP理解 / RAG  
**关键词**: 多跳问答, 检索增强生成, 自适应规划, 事实提取, 多任务微调

## 一句话总结
提出 REAP 双模块迭代框架，通过子任务规划器 (SP) 维护全局视角动态指导推理轨迹，事实提取器 (FE) 从检索内容中提取结构化事实和潜在线索，两者递归协作解决多跳问答。在 4 个基准上以 Llama-3.1-8B 显著超越所有基线（HotpotQA F1 68.0 vs 次优 63.4）。

## 研究背景与动机

**领域现状**：RAG 通过引入外部知识缓解 LLM 幻觉，但多跳问答 (MHQA) 需要跨多文档整合信息。现有迭代 RAG 方法（IRCoT、Iter-RetGen 等）采用多轮检索+逐步推理，部分方法引入 MCTS 等搜索算法寻找最优推理轨迹。

**现有痛点**：(1) 缺乏全局规划——增量分解复杂查询为子查询可能陷入局部推理死局；(2) 检索内容利用不足——模型倾向仅提取直接答案，忽略对最终解答关键的潜在线索；(3) 引入评分器/决策器增加系统复杂度和降低可解释性。

**核心矛盾**：线性管道式推理无法应对多跳场景中的推理失败和路径纠偏——当某一步推理产生错误或不完整信息时，缺乏机制检测并修正整体推理方向。

**本文目标** 将多跳推理从线性管道重构为动态状态驱动循环，在保持全局视角的同时实现错误检测和轨迹纠偏。

**切入角度**：显式维护结构化子任务计划和事实列表，将规划（SP）与事实采集（FE）功能解耦但紧密协调。

**核心 idea**：SP 提供全局规划视角指导推理方向，FE 提供高保真结构化事实，两者递归反馈形成自纠正推理循环。

## 方法详解

### 整体框架
给定复杂查询 $Q$，Decomposer 首先生成结构化任务计划 $\mathcal{P}_0 = \{(id_i, q_i, deps_i)\}$。然后进入 SP-FE 迭代循环：SP 分析当前状态决定下一步动作，FE 执行检索和推理产生结构化事实，事实反馈给 SP 更新计划。循环直到计划完全解决，最后 Synthesizer 生成最终答案。

### 关键设计

1. **子任务规划器 (Sub-task Planner, SP)**:

    - 功能：维护全局任务计划，动态指导推理轨迹
    - 核心思路：根据 FE 返回事实的满足级别 $l_t$ 分派到两个子模块：
        - **Plan Updater**（处理理想情况，$l_t = \text{DirectAnswer}$）：做确定性规则更新——(a) 事实替换：用新获得的具体实体替换待处理子任务中的抽象占位符；(b) 计划分叉：当子任务产生多个有效子答案时，将后续依赖子任务复制为并行分支
        - **Re-Planner**（处理非理想情况，$l_t = \text{PartialClue/Failed}$）：(a) 务实充分性评估：先判断部分信息是否功能上足以满足后续推理需求（若足够则视为已解决，避免完美主义搜索循环）；(b) 范围化计划修复：区分局部问题（微调子查询）和系统性缺陷（剪枝无效分支+注入新子任务序列）
    - 设计动机：将规划按难度分流——简单更新用轻量规则处理，复杂异常用重量级推理处理，兼顾效率和鲁棒性

2. **事实提取器 (Fact Extractor, FE)**:

    - 功能：从检索文档中提取高保真结构化事实和潜在线索
    - 核心思路：对每个子查询 $q_t$，检索 top-5 文档后，LLM 生成结构化事实元组 $f_t = (s_t, e_t, r_t, l_t)$——核心陈述 $s_t$、文本证据 $e_t \subseteq D_t$、推理过程 $r_t$（CoT解释）、满足级别 $l_t$（DirectAnswer/PartialClue/Failed）。关键点：FE 以历史事实 $\mathcal{F}_{t-1}$ 为条件，实现跨步骤的共指消解和关系识别
    - 设计动机：(1) 不仅提取直接答案还识别潜在线索，避免遗漏关键信息；(2) 结构化元组确保可追溯性；(3) 满足级别为 SP 的决策提供信号

3. **多任务微调**:

    - 功能：通过联合训练提升数据稀缺的 Re-Planner 性能
    - 核心思路：Decomposer、Plan Updater、Re-Planner 三个模块共享"基于现有信息生成/修改结构化任务计划"的功能共性。将三者的数据集合并训练同一模型 $M_\phi$：$\min_\phi \sum_{task} \lambda_{task} \mathbb{E}[\mathcal{L}_{task}(M_\phi(x), y)]$。从数据丰富的任务（分解、更新）向数据稀缺的任务（重规划）迁移知识
    - 设计动机：Re-Planner 因调用频率低导致训练数据稀缺，单独训练效果差

### 损失函数 / 训练策略
- 多任务微调：各任务权重 $\lambda = 1$，标准语言模型损失
- 使用 GPT-4 在 HotpotQA + 2WikiMultihopQA 上运行 REAP 收集 7000 样本，过滤后得 5556 训练样本
- 推理用 Llama-3.1-8B-Instruct，检索用 e5-large-v2，top-5，最多 5 轮迭代

## 实验关键数据

### 主实验

| 方法 | HotpotQA F1 | 2Wiki F1 | MuSiQue‡ F1 | Bamboogle‡ F1 |
|------|------------|---------|------------|--------------|
| Standard RAG | 48.6 | 38.5 | 13.5 | 30.9 |
| IRCoT | 51.4 | 36.5 | 18.6 | 30.1 |
| R1-Searcher | 63.4 | 69.4 | 33.8 | 58.0 |
| **REAP** | **68.0** | **79.6** | **38.3** | **65.2** |

### 消融实验

| 配置 | HotpotQA F1 | 2Wiki F1 | MuSiQue F1 | Bamboogle F1 |
|------|------------|---------|------------|-------------|
| w/o Replan | 64.9 | 78.6 | 34.2 | 61.6 |
| w/o Verify | 65.1 | 78.0 | 34.8 | 60.8 |
| w/o Clue | 64.6 | 76.5 | 35.2 | 62.7 |
| **REAP (full)** | **68.0** | **79.6** | **38.3** | **65.2** |

### 关键发现
- **Re-Planner 贡献最大**：去掉后 HotpotQA 降 3.1%、MuSiQue 降 4.1%、Bamboogle 降 3.6%，尤其在复杂多跳数据集上影响更大
- **2Wiki 上提升最显著**：F1 从次优的 69.4 (R1-Searcher) 提升到 79.6 (+10.2)，说明结构化规划对需要跨文档推理的任务效果突出
- **泛化能力强**：仅在 HotpotQA + 2Wiki 上训练，在域外的 MuSiQue 和 Bamboogle 上同样最优
- **多任务微调效果显著**：联合训练 (ft-all) 显著优于分别训练 (ft-separate)，特别是 Re-Planner 组件
- REAP 用 8B 模型即超越 70B 基线的多数方法

## 亮点与洞察
- **"务实充分性评估"是精妙设计**：不追求每个子任务都完美完成，而是评估部分信息是否"够用"就继续推进。这避免了在困难子任务上无限循环，同时不丢失关键信息。可迁移到任何多步决策系统
- **结构化事实元组的设计**：将答案、证据、推理过程和满足级别打包为四元组，使得 SP 可以基于满足级别做分流决策，实现了模块间的松耦合高内聚
- **规划/执行分离但紧密协调**：SP 不执行任何检索或推理，FE 不做任何规划决策，但两者通过结构化接口紧密互动。这种分离大幅提高了可解释性和调试性

## 局限与展望
- 训练数据依赖 GPT-4 生成的 5556 样本，数据质量受 GPT-4 能力限制
- 最大迭代轮数固定为 5，对需要更多跳的极复杂查询可能不够
- 仅在英文 Wikipedia 语料上测试，对多语言和专业领域知识库的效果未验证
- Re-Planner 的"范围化修复"策略（局部 vs 系统性）的判断标准可能需要更精细的设计
- 当检索器本身质量差时（top-5 中无相关文档），框架的纠偏能力有限

## 相关工作与启发
- **vs R1-Searcher**: R1-Searcher 用强化学习学习检索和生成策略，REAP 通过结构化规划+事实提取超越它（HotpotQA F1 68.0 vs 63.4），且 REAP 更可解释
- **vs IRCoT**: IRCoT 将 CoT 与多轮检索结合，但缺乏全局规划和错误纠偏。REAP 在所有基准上大幅超越
- **vs SearChain**: SearChain 也做问题分解+多轮检索，但未显式维护全局状态和事实列表。REAP 在 2Wiki 上超 33 个 F1 点

## 评分
- 新颖性: ⭐⭐⭐⭐ SP-FE 双模块递归反馈+务实充分性评估+事实满足级别分流都是有价值的设计
- 实验充分度: ⭐⭐⭐⭐⭐ 4个数据集（含2个域外）、10个基线、详细消融和多任务微调分析
- 写作质量: ⭐⭐⭐⭐ 形式化表述清晰，框架图直观
- 价值: ⭐⭐⭐⭐⭐ 多跳 RAG 的实用改进，8B 模型超越 70B 基线，强泛化能力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FrugalRAG: Less is More in RL Finetuning for Multi-hop Question Answering](../../ICLR2026/information_retrieval/frugalrag_less_is_more_in_rl_finetuning_for_multi-hop_question_answering.md)
- [\[AAAI 2026\] N2N-GQA: Noise-to-Narrative for Graph-Based Table-Text Question Answering Using LLMs](n2n-gqa_noise-to-narrative_for_graph-based_table-text_question_answering_using_l.md)
- [\[AAAI 2026\] PRIME: Planning and Retrieval-Integrated Memory for Enhanced Reasoning](prime_planning_and_retrieval-integrated_memory_for_enhanced_reasoning.md)
- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](../../ACL2025/information_retrieval/mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](../../ACL2026/information_retrieval/chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)

</div>

<!-- RELATED:END -->
