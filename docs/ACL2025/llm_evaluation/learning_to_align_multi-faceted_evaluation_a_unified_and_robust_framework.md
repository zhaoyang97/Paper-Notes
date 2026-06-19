---
title: >-
  [论文解读] Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework
description: >-
  [ACL 2025][LLM评测][LLM evaluation] 提出 ARJudge 评估框架，通过微调 Analyzer 自适应生成评估标准并执行文本+代码双驱动分析，配合无需微调的 Refiner 综合判断，在多个评估基准上超越现有微调评估器，尤其在指令遵循评估上通过代码驱动分析提升高达 11.1%。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "LLM evaluation"
  - "multi-faceted analysis"
  - "code-driven evaluation"
  - "fine-tuned evaluator"
  - "instruction following"
---

# Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework

**会议**: ACL 2025  
**arXiv**: [2502.18874](https://arxiv.org/abs/2502.18874)  
**代码**: 无  
**领域**: 其他  
**关键词**: LLM evaluation, multi-faceted analysis, code-driven evaluation, fine-tuned evaluator, instruction following

## 一句话总结

提出 ARJudge 评估框架，通过微调 Analyzer 自适应生成评估标准并执行文本+代码双驱动分析，配合无需微调的 Refiner 综合判断，在多个评估基准上超越现有微调评估器，尤其在指令遵循评估上通过代码驱动分析提升高达 11.1%。

## 研究背景与动机

用 LLM 来评估 LLM 的输出质量已成为重要范式，但现有的微调评估器面临两个核心问题：

**预定义标准的适应性不足**
   - 现有方法（如 Auto-J、Prometheus）使用固定的通用评估标准（如"简洁性""逻辑结构"）
   - 这些标准无法覆盖创意写作、专业领域等需要特定评估维度的任务
   - 面对未见过的新指令时泛化能力差

**客观约束评估不稳定**
   - LLM 在评估量化要求（如字数限制）和结构约束（如格式要求）时表现不可靠
   - 连基本的文本属性（如计数单词数量）都难以准确判断
   - 纯文本分析在客观验证方面有天然局限

作者认为，要构建鲁棒的评估器需要两个能力：**自适应生成评估标准**（评什么）+ **多面向分析**（怎么评），特别是结合代码工具来处理客观要求。

## 方法详解

### 整体框架

ARJudge 由两个组件构成：
- **Analyzer**（微调）：自适应生成评估标准，执行文本分析和代码驱动分析
- **Refiner**（无微调）：综合 Analyzer 的多面向分析结果，进行最终判断

训练数据来自一个精心构建的 **Composite Analysis Corpus**（复合分析语料库）。

### 关键设计

1. **复合分析语料库构建**

    - **评估标准生成**：两类问题
        - Type 1（文本分析用）：给定指令 + 3 个样例回复，用 LLM 生成 3 个评估问题
        - Type 2（代码分析用）：通过 self-instruct 为指令添加客观约束（如字数限制），生成对应评估问题
    - **文本分析收集**：给 LLM 指令、两个回复和评估问题，要求进行比较分析，过滤与人工标注矛盾的样本
    - **代码驱动分析开发**：
        - 用 Claude-3.5-Sonnet 为客观评估问题生成 Python 验证函数
        - 函数输入为回复文本，输出为验证结果
        - 双重过滤：执行测试 + 反向验证（让 LLM 解释代码目的并检查与原问题的一致性）
    - 设计动机：将"评什么"和"怎么评"有机结合到一个训练流程中

2. **Analyzer 训练**

    - 基于 Qwen2.5-7B-Instruct 模型微调
    - 训练样本约 25K，包含三类任务：评估问题生成、文本分析、代码生成
    - 使用不同 prompt 模板区分问题生成和回复分析任务
    - 文本/代码分析共享同一 prompt 模板，通过起始提示词触发不同模式
    - 设计动机：多任务训练使模型学会自适应选择分析方式

3. **Refiner 综合判断**

    - 使用同一 Qwen2.5-7B-Instruct 模型的零样本推理
    - 输入为 Analyzer 的所有分析结果
    - 要求 Refiner 重新审视指令要求，综合判断哪个回复更好
    - 设计动机：保留通用模型的宏观评估能力，弥补 Analyzer 可能的偏见

### 损失函数 / 训练策略

- 使用标准的监督微调损失（next token prediction）
- 贪心解码（temperature=0）生成分析和判断
- 仅微调 Analyzer，Refiner 使用相同模型但不做微调

## 实验关键数据

### 主实验（成对比较准确率）

| 模型 | JudgeLM | PandaLM | Auto-J | MTBench | LLMBar | 平均 |
|------|---------|---------|--------|---------|--------|------|
| GPT-4o | 81.8 | 83.1 | 78.6 | 78.8 | 79.8 | 80.4 |
| Claude-3.5-Sonnet | 82.9 | 86.4 | 78.2 | 80.8 | 83.4 | 82.3 |
| Auto-J-13B | 77.9 | 77.2 | 79.7 | 75.0 | 27.8 | 67.5 |
| Prometheus2-7B | 76.5 | 76.3 | 75.1 | 74.3 | 41.5 | 68.7 |
| **ARJudge** | **81.0** | **82.4** | **78.5** | **78.3** | **68.2** | **77.7** |

ARJudge 在所有微调评估器中最优，在 LLMBar 上比最佳微调基线 Prometheus2 提升 26.7%。

### 消融实验

| 设置 | JudgeLM | PandaLM | Auto-J | MTBench | LLMBar |
|------|---------|---------|--------|---------|--------|
| Qwen2.5-7B（基线） | 80.0 | 80.7 | 73.8 | 75.2 | 52.6 |
| **ARJudge** | 81.0 | 82.4 | 78.5 | 78.3 | 68.2 |
| -w/o 微调 | 73.1 | 75.6 | 68.7 | 70.0 | 62.5 |
| -w/o 微调&多面向 | 74.7 | 72.2 | 65.6 | 67.8 | 63.7 |
| -w/o Refine | 81.7 | 82.8 | 79.6 | 79.1 | 63.7 |

微调为最关键组件（去掉后全面下降），Refiner 对挑战性样本（LLMBar）帮助显著。

### 关键发现

1. **代码驱动分析有效**：在 IFEval 上代码分析的一致性远超纯文本分析，生成代码执行成功率 100%
2. **多轮分析的效果**：增加分析轮次在常规测试集上有正向效果，但在 LLMBar 等挑战集上可能放大不确定性
3. **Refinement 的双面性**：微调 Analyzer 下 Refiner 保持性能，但无微调 Analyzer 下 Refiner 反而降低性能（自我纠正失败）
4. **显著超越骨干模型**：对比 Qwen2.5-7B 基础水平，ARJudge 平均提升 15.6%
5. **LLMBar 泛化能力**：对抗性设计的 LLMBar 上表现接近 DeepSeek-V3 (80.4 vs 68.2)

## 亮点与洞察

- **"评什么+怎么评"的统一框架**：不再预定义标准，而是让模型自适应产生评估维度
- **代码作为评估工具**：将代码执行引入评估流程，解决了 LLM 在客观判断上的不可靠性
- **Analyzer-Refiner 分工设计**：微调的专家分析 + 未微调的通用综合，兼顾深度和广度
- **反向验证机制**：代码生成后通过"解释→核查"两步确保代码与评估目标一致

## 局限与展望

1. 仅限于成对比较评估，不支持对单个回复的评分
2. 工具使用仅限 Python 代码，未考虑搜索引擎、知识库等其他验证工具
3. Refiner 依赖 LLM 自身推理能力，若基础能力不足则改进有限
4. 训练数据构建依赖 GPT-4o 和 Claude，存在级联偏差
5. 代码生成局限于客观约束验证，无法处理主观质量评估

## 相关工作与启发

- 相比 Auto-J 的预定义标准 + 文本分析，ARJudge 增加了自适应标准生成和代码分析两个维度
- 相比 Prometheus 的固定评估模板，ARJudge 的多面向分析更灵活
- 启发：可将代码驱动分析扩展到更多场景（如事实核查接入搜索引擎、数学推理接入计算器）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 代码驱动的客观评估分析是新颖的视角，Analyzer-Refiner 架构有层次感
- **实验充分度**: ⭐⭐⭐⭐ — 5 个测试集、多种基线、消融实验和 case study 全面
- **写作质量**: ⭐⭐⭐⭐ — 语料库构建流程清晰，图表丰富，但部分内容较冗长
- **价值**: ⭐⭐⭐⭐ — 为构建更可靠的 LLM 评估器提供了实用框架，代码分析思路有推广价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MARS: Benchmarking the Metaphysical Reasoning Abilities of Language Models with a Multi-task Evaluation Dataset](mars_benchmarking_the_metaphysical_reasoning_abilities_of_language_models_with_a.md)
- [\[ACL 2025\] EducationQ: Evaluating LLMs' Teaching Capabilities Through Multi-Agent Dialogue Framework](educationq_evaluating_llms_teaching_capabilities_through_multi-agent_dialogue_fr.md)
- [\[NeurIPS 2025\] Words That Unite The World: A Unified Framework for Deciphering Central Bank Communications Globally](../../NeurIPS2025/llm_evaluation/words_that_unite_the_world_a_unified_framework_for_deciphering_central_bank_comm.md)
- [\[ACL 2025\] READoc: A Unified Benchmark for Realistic Document Structured Extraction](readoc_a_unified_benchmark_for_realistic_document_structured_extraction.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)

</div>

<!-- RELATED:END -->
