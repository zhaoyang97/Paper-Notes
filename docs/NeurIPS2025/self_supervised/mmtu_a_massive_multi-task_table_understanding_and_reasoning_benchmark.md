---
title: >-
  [论文解读] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark
description: >-
  [NeurIPS 2025][自监督学习][表格理解] 构建了一个包含 28,136 道问题、覆盖 25 种真实表格任务的大规模基准 MMTU，系统评估 LLM 在专业级表格理解、推理和操作方面的能力，发现即使是 GPT-5 等前沿推理模型也仅得分约 69.6%。 表格数据在电子表格、数据库、计算笔记本等真实应用中至关重要…
tags:
  - "NeurIPS 2025"
  - "自监督学习"
  - "表格理解"
  - "benchmark"
  - "LLM评估"
  - "多任务推理"
  - "结构化数据"
---

# MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark

**会议**: NeurIPS 2025  
**arXiv**: [2506.05587](https://arxiv.org/abs/2506.05587)  
**代码**: [有](https://github.com/MMTU-Benchmark/MMTU)  
**领域**: 机器人  
**关键词**: 表格理解, benchmark, LLM评估, 多任务推理, 结构化数据

## 一句话总结

构建了一个包含 28,136 道问题、覆盖 25 种真实表格任务的大规模基准 MMTU，系统评估 LLM 在专业级表格理解、推理和操作方面的能力，发现即使是 GPT-5 等前沿推理模型也仅得分约 69.6%。

## 研究背景与动机

表格数据在电子表格、数据库、计算笔记本等真实应用中至关重要，传统上需要数据工程师、DBA 等专业人员操作。虽然 LLM 在表格任务上已展现出很好的潜力，但现有评测存在严重局限：

**首先，任务覆盖面过窄**——现有表格基准主要聚焦于 NL-to-SQL 和 Table-QA 两类任务，忽略了数十年计算机科学研究中积累的大量其他专业表格任务（如表格转换、实体匹配、数据清洗、列关系发现等）。

**其次，现有基准规模有限**——例如电子表格基准通常只有几百个测试用例，且绑定特定文件格式。相比 MMLU（15,908题）和 MMMU（11,550题）等 NLP 基准，表格领域缺乏可比规模的综合评测。

**第三，已有 NLP 基准易饱和**——如 GSM8k、HumanEval 等已被前沿模型基本解决，需要更具挑战性的新基准来持续推动模型进步。

MMTU 旨在填补这一空白，构建一个涵盖专业级表格任务的大规模、多样化、具有挑战性的基准。

## 方法详解

### 整体框架

MMTU 并非提出新模型，而是一个精心策划的基准测试。它的设计流程包含：文献调研→任务筛选→数据标准化→质量检查→专家验证五个步骤。

### 关键设计

1. **25 种任务、7 大类别的全面覆盖**: 从数据管理 (SIGMOD/VLDB)、编程语言 (PLDI/POPL)、Web 数据 (WWW/WSDM) 等社区过去二十年的研究中系统调研，最终选择 25 种面向用户的、可客观评估的、基于真实数据的任务。七大类别包括：Table Transform（表格变换）、Table Matching（表格匹配）、Data Cleaning（数据清洗）、Table Join（表格连接）、Column Transform（列变换）、Column Relationship（列关系）、Table Understanding（表格理解）、NL-to-Code、Table QA、KB Mapping。其中大部分任务此前从未用于评估基础模型。

2. **标准化三元组格式**: 所有 28,136 道问题统一为 <Instruction, Input-Table(s), Ground-truth> 格式，便于不同 LLM 的即插即用评测。问题涉及 61,763 张真实表格（包括 Web 表格 74.9%、电子表格 7.4%、关系表格 17.7%）。26.1% 的问题需要编码（SQL/Python Pandas/公式）。

3. **多层质量保障**: (a) 使用 o4-mini 进行歧义性和正确性检查，移除 8% 的问题；(b) LLM 筛查隐私和安全风险；(c) 单一数据集贡献上限 1000 题保证多样性；(d) 每个任务抽样 20 题由领域专家人工验证。

4. **灵活评估框架**: 不同于 MMLU 等多选题基准，MMTU 采用开放式结构化答案，支持执行式评估（SQL/Python 代码执行）和结构化输出评估（如无序 JSON 列表比较），更贴近真实场景。

### 损失函数 / 训练策略

不适用（基准测试论文）。

## 实验关键数据

### 主实验

**模型整体表现 (Table 3, 精选)**:

| 模型类型 | 模型 | MMTU 得分 | 每题成本($) |
|----------|------|-----------|------------|
| Reasoning | GPT-5 | 0.696 | 0.01727 |
| Reasoning | o3 | 0.691 | 0.01539 |
| Reasoning | GPT-5-mini | 0.667 | 0.00276 |
| Reasoning | Gemini-2.5-pro | 0.665 | 0.00790 |
| Reasoning | DeepSeek-R1 | 0.579 | 0.00167 |
| Reasoning | Qwen3-32B | 0.506 | 0.00017 |
| Chat | GPT-5-Chat | 0.577 | 0.00534 |
| Chat | DeepSeek-V3 | 0.555 | 0.00095 |
| Chat | GPT-4o | 0.507 | 0.01019 |
| Chat | Llama-3.3-70B | 0.454 | 0.00150 |

### 消融/分析实验

**推理 vs 聊天模型差异**:

| 对比维度 | 推理模型 (最优) | 聊天模型 (最优) | 差距 |
|----------|----------------|----------------|------|
| 最优得分 | 69.6% (GPT-5) | 57.7% (GPT-5-Chat) | +11.9pp |
| 性价比最优 | Qwen3-32B (0.506, $0.00017) | - | - |

### 关键发现

1. **推理模型显著优于聊天模型**：最优推理模型比最优聊天模型高出 10+ 百分点，说明 MMTU 任务本质需要编码和逻辑推理能力
2. **前沿模型仍有巨大提升空间**：即使 GPT-5 也仅 69.6%，距离人类专家水平差距明显
3. **前沿模型对表格序列化格式不敏感**：Markdown/CSV/JSON/HTML 格式影响不大，反映模型理解多样数据格式的能力提升
4. **LLM 仍然难以处理长表格上下文**：大量行/列的表格中，需要跨单元格整体推理（尤其列方向）的复杂任务仍然困难
5. **表格级扰动（行/列洗牌）会降低性能**：即使这些变化应该是语义不变的，暴露了模型表格理解的鲁棒性不足
6. **更新更大的模型显著优于旧模型/小模型**：表明基础模型在表格能力上确实在快速进步

## 亮点与洞察

- **填补重要空白**：首次将数据管理/PL 社区数十年的专业表格任务引入 LLM 评测体系，25 个任务中大多数从未用于评估基础模型
- **规模可比 MMLU**：28,136 题的规模使其成为表格领域最全面的基准
- **注重公平性**：从质量检查到专家验证的多层保障，以及隐私安全审查
- **发现推理能力是瓶颈**：推理模型大幅优于聊天模型，说明简单的模式匹配不够

## 局限与展望

- 仅包含可客观评估的任务，排除了表格摘要、表格增强等主观/创造性任务
- 仅从现有研究文献抽样，遗漏了实践中重要但文献不充分的任务（如多轮表格操作）
- 当前仅使用文本输入，未考虑多模态（如视觉表格）输入
- 对各任务的难度分布不均，部分任务可能已被较好解决

## 相关工作与启发

- 可类比 MMLU 对 NLP 领域的推动作用，MMTU 有望成为表格 AI 领域的核心评测
- 与 SWE-bench 等编码基准互补，覆盖了数据处理层面的编码需求
- 对电子表格 Copilot、数据库 Copilot 等产品方向有直接的评测价值
- 未来可扩展为多语言版本或加入多模态表格输入

## 评分

- 新颖性: ⭐⭐⭐⭐ (任务覆盖面是最大创新，单项任务设计不算新)
- 实验充分度: ⭐⭐⭐⭐⭐ (大量模型对比+多维分析)
- 写作质量: ⭐⭐⭐⭐⭐ (组织清晰，图表丰富，数据详实)
- 价值: ⭐⭐⭐⭐⭐ (填补关键空白，对领域发展有长期推动意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Contrastive Representations for Temporal Reasoning](contrastive_representations_for_temporal_reasoning.md)
- [\[ICML 2025\] MTL-UE: Learning to Learn Nothing for Multi-Task Learning](../../ICML2025/self_supervised/mtl-ue_learning_to_learn_nothing_for_multi-task_learning.md)
- [\[NeurIPS 2025\] Understanding Ice Crystal Habit Diversity with Self-Supervised Learning](understanding_ice_crystal_habit_diversity_with_self-supervised_learning.md)
- [\[NeurIPS 2025\] CleverBirds: A Multiple-Choice Benchmark for Fine-grained Human Knowledge Tracing](cleverbirds_a_multiple-choice_benchmark_for_fine-grained_human_knowledge_tracing.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)

</div>

<!-- RELATED:END -->
