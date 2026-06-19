---
title: >-
  [论文解读] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables
description: >-
  [NeurIPS 2025][多模态VLM][长上下文理解] 提出 NeedleInATable (NIAT) 基准，将表格中每个单元格视为"针"，评估 LLM 对长结构化表格的细粒度感知能力，揭示现有模型在复杂下游任务上的高分可能依赖数据捷径而非真正的表格理解。 领域现状： 长上下文 LLM 发展迅速…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "长上下文理解"
  - "结构化表格"
  - "LLM评估基准"
  - "表格感知"
  - "数据合成"
---

# NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables

**会议**: NeurIPS 2025  
**arXiv**: [2504.06560](https://arxiv.org/abs/2504.06560)  
**代码**: [GitHub](https://github.com/wlr737/NeedleInATable)  
**领域**: Multimodal / VLM / 表格理解  
**关键词**: 长上下文理解, 结构化表格, LLM评估基准, 表格感知, 数据合成

## 一句话总结

提出 NeedleInATable (NIAT) 基准，将表格中每个单元格视为"针"，评估 LLM 对长结构化表格的细粒度感知能力，揭示现有模型在复杂下游任务上的高分可能依赖数据捷径而非真正的表格理解。

## 研究背景与动机

**领域现状**: 长上下文 LLM 发展迅速，各种基准如 Needle-in-a-Haystack 已用于评估非结构化文本的长上下文处理能力，但结构化表格场景被严重忽视。

**现有痛点**: 现有表格基准（如 WTQ、TabFact）主要考察高层推理能力，忽略了模型对单个表格单元格的基础细粒度感知——而这恰恰是可靠表格应用的基石。

**核心矛盾**: 模型在复杂表格推理任务上表现不错，但这可能源于数据集特有的相关性或捷径（shortcuts），而非真正理解表格的二维结构。

**本文目标**: 构建一个评估 LLM 对长表格中每个单元格感知能力的基准，并验证提升该能力是否能反哺下游表格任务。

**切入角度**: 将表格类比为"干草堆"，将单个单元格类比为"针"，设计 Cell-Locating 和 Cell-Lookup 两类原子任务。

**核心idea**: 如果模型连最基础的单元格定位都做不好，那其在复杂任务上的好成绩就值得怀疑；增强 NIAT 能力可以从根本上提升表格理解。

## 方法详解

### 整体框架

NIAT 基准包含 750 张表格和 287K 测试样本，覆盖三种表格结构（平坦表、水平表、层级表）、三种格式（Markdown、HTML、图像），以及多种表格尺寸。

### 关键设计

1. **Cell-Locating 任务**: 给定行列索引，要求模型提取对应单元格内容。→ 评估模型对基本二维表格结构的理解。→ 通过预定义模板自动构造查询，无需 GPT-4o 参与。→ 与 Cell-Lookup 形成互补，测试不同层面的能力。

2. **Cell-Lookup 任务**: 给定简单查找问题（答案为特定单元格，无需聚合运算），要求模型检索目标单元格。→ 评估模型利用语义线索进行行列交叉检索的能力。→ 使用 GPT-4o ICL 生成查找问题，并通过 self-consistency 过滤无效问题。→ 结果表明 LLM 擅长此类语义匹配但不擅长结构定位。

3. **strong2weak 数据合成方法**: 利用 GPT-4o 在训练集表格上生成 NIAT 查询和链式思维（CoT）推理响应，用于微调弱模型。→ 因为直接用短答案微调会导致过拟合捷径。→ 设计 6 种更难的查找子任务（如 cell-retrieval 需在全表搜索），提高数据多样性。→ 仅 12K 合成数据即可显著提升 NIAT 和下游任务性能。

### 训练策略

- 在 Llama3.1-8B-Instruct 和 Qwen2.5-7B-Instruct 上微调
- 使用 GPT-4o 合成的 CoT 推理过程作为目标响应
- 6K Cell-Locating + 6K Cell-Lookup = 共 12K 训练样本

## 实验关键数据

### 主实验

| 模型 | Cell-Locating Avg | Cell-Lookup Avg | Overall |
|------|:-:|:-:|:-:|
| Llama3.1-8B-Instruct | 6.16 | 65.74 | 35.95 |
| Qwen2.5-7B-Instruct | 9.46 | 47.60 | 28.53 |
| TableGPT2 | 8.84 | 73.87 | 41.36 |
| GPT-4o | 26.00 | 68.30 | 47.15 |
| DeepSeek-R1 | 65.91 | 80.99 | 73.45 |
| Qwen3-30B-A3B | 16.49 | 78.62 | 47.55 |

### 下游任务提升（合成数据微调后）

| 模型 | WTQ | TabFact | HiTab | TABMWP | Avg |
|------|:-:|:-:|:-:|:-:|:-:|
| Qwen2.5-7B 原始 | 52.90 | 70.00 | 30.50 | 54.42 | 51.96 |
| Qwen2.5-7B + NIAT | 60.28 | 61.28 | **62.28** | 72.39 | **64.06** |
| Llama3.1-8B 原始 | 49.90 | 62.80 | 26.10 | 54.78 | 48.40 |
| Llama3.1-8B + NIAT | **67.43** | **78.57** | **49.41** | 66.15 | **65.39** |

### 消融实验

| 微调数据类型 | WTQ | TabFact | HiTab | TABMWP |
|------|:-:|:-:|:-:|:-:|
| Cell-Locating + Cell-Lookup | **67.43** | **78.57** | **49.41** | 66.15 |
| 仅 Cell-Locating | 67.33 | 67.45 | 33.44 | **70.50** |
| 仅 Cell-Lookup | 59.00 | 53.50 | 35.00 | 69.44 |
| 4个下游数据集直接微调 | 64.78 | 61.35 | 53.76 | 67.15 |

### 关键发现

- **Lost-in-the-Middle-Table 现象**: 所有 LLM（包括 GPT-4o）对表格首行和末行的感知明显优于中间行，随表格尺寸增大性能急剧下降。
- **Cell-Locating vs Cell-Lookup 差距巨大**: 模型在 Cell-Lookup 上远好于 Cell-Locating，说明它们依赖语义共现而非真正理解表格结构。
- **注意力模式分析**: LLM 展现 Multi-Slash（关注同列）和 Local-Triangle（关注同行表头）两种注意力模式。
- **仅 12K NIAT 数据微调即超越使用百万级数据的 TableGPT2**: 验证了基础感知能力的重要性。

## 亮点与洞察

- 提出了一个简洁而有力的视角：用最基础的单元格定位/查找来检验 LLM 的表格理解是否"真实"。
- Lost-in-the-Middle-Table 现象是对 Lost-in-the-Middle 的表格版延伸，非常有启发性。
- strong2weak 数据合成思路高效，12K 数据即可带来显著提升。
- 揭示了现有表格基准可能存在的数据泄露和捷径问题。

## 局限与展望

- NIAT 基准目前仅覆盖英文表格，缺少多语言评估。
- Cell-Locating 任务较为简单，可以设计更复杂的结构理解任务（如跨表关联）。
- 合成数据方法依赖 GPT-4o，成本较高，可探索更廉价的数据生成方式。
- 仅评估了零样本设置，few-shot 和 fine-tuning 对比不够充分。
- 对超长表格（>120K token）的评估有限。

## 相关工作与启发

- Needle-in-a-Haystack 系列（RULER、InfiniBench、LongBench）聚焦非结构化文本，NIAT 是其结构化表格版本。
- TableGPT2、StructLLM 等表格 LLM 通过大规模表格指令微调提升能力，但 NIAT 揭示其基础感知仍有不足。
- DeepSeek-R1 的 test-time scaling 在 NIAT 上表现最好，暗示推理链对表格结构理解很有帮助。

## 评分

- 新颖性: ⭐⭐⭐⭐ 视角新颖，从最基础的单元格感知切入检验表格理解的"真实性"
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖大量开源/闭源模型，多种表格结构和格式，分析详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，论述逻辑连贯
- 价值: ⭐⭐⭐⭐ 对表格理解领域有重要的基准贡献和方法论启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models](hope_hybrid_of_position_embedding_for_long_context_visionlan.md)
- [\[NeurIPS 2025\] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)
- [\[NeurIPS 2025\] VITRIX-CLIPIN: Enhancing Fine-Grained Visual Understanding in CLIP via Instruction Editing Data and Long Captions](vitrix-clipin_enhancing_fine-grained_visual_understanding_in_clip_via_instructio.md)
- [\[NeurIPS 2025\] Context Informs Pragmatic Interpretation in Vision-Language Models](context_informs_pragmatic_interpretation_in_vision-language_models.md)
- [\[NeurIPS 2025\] FlySearch: Exploring how vision-language models explore](flysearch_exploring_how_vision-language_models_explore.md)

</div>

<!-- RELATED:END -->
