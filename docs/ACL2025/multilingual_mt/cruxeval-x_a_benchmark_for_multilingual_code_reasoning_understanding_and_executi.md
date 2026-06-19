---
title: >-
  [论文解读] CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution
description: >-
  [ACL 2025][多语言/翻译][代码推理] 提出 CruxEval-X，一个覆盖 19 种编程语言的多语言代码推理基准，通过全自动的测试引导翻译流水线从 Python 版 CruxEval 扩展而来，包含 12,660 个题目和 19K 测试用例，对 24 个 LLM 的评估揭示了编程语言间的相关性以及单语言训练模型的跨语言泛化能力。
tags:
  - "ACL 2025"
  - "多语言/翻译"
  - "代码推理"
  - "多语言基准"
  - "输入推理"
  - "输出推理"
  - "跨语言泛化"
---

# CruxEval-X: A Benchmark for Multilingual Code Reasoning, Understanding and Execution

**会议**: ACL 2025  
**arXiv**: [2408.13001](https://arxiv.org/abs/2408.13001)  
**代码**: 有（数据集公开）  
**领域**: 代码智能 / 多语言评估  
**关键词**: 代码推理, 多语言基准, 输入推理, 输出推理, 跨语言泛化

## 一句话总结

提出 CruxEval-X，一个覆盖 19 种编程语言的多语言代码推理基准，通过全自动的测试引导翻译流水线从 Python 版 CruxEval 扩展而来，包含 12,660 个题目和 19K 测试用例，对 24 个 LLM 的评估揭示了编程语言间的相关性以及单语言训练模型的跨语言泛化能力。

## 研究背景与动机

现有代码基准存在两大偏差，严重限制了对 LLM 编码能力的全面评估：

**编程语言偏差**：超过 95% 的代码生成基准由 Python 主导。Java、C/C++ 等流行语言的 LLM 能力几乎未被系统评估。简单地将编程语言从 Python 换为 C++ 就可能导致正确推理变为错误（如类型系统差异导致的运行时行为不同）。

**编码任务偏差**：大多数基准聚焦于代码生成（自然语言→程序），而代码推理（给定程序，推理输入或输出）这一同样重要的能力被严重忽视。

**构建多语言基准的困难**：
   - 人工标注成本高昂（如 McEval 花费 $12,000）
   - 自动翻译效果差（ChatGPT 平均仅 64% 翻译成功率）
   - 竞赛平台数据存在数据泄漏问题

CruxEval-X 正是为填补这些空白而设计，从已有的 Python 代码推理数据集 CruxEval 自动扩展到 19 种编程语言。

## 方法详解

### 整体框架

构建流水线分为三个主要步骤：
1. 函数签名翻译（类型映射）
2. 测试用例翻译（规则增强）
3. 迭代生成与修复（多 LLM 协作）

### 关键设计

#### 1. 函数签名翻译（Step I）

- **功能**：将 Python 函数签名翻译为其他语言的等价签名
- **核心思路**：由于 Python 不需要显式类型注解，先从测试用例推断变量类型，再使用类型映射规则转换
- **设计动机**：类型信息是跨语言翻译的关键，Python 的动态类型与 C++ 的静态类型之间存在根本差异
- 例如：`def f(s1:str, s2:str) -> str` → `std::string f(std::string s1, std::string s2)`
- 800 个 Python 题目全部可完成签名翻译

#### 2. 测试用例翻译（Step II）

- **功能**：将 Python 测试用例翻译为其他语言的等价形式
- **核心思路**：采用并增强 MultiPL-E 的映射规则
- **两项改进**：
    - 增强结构化类型处理（如为 C# 的 Dict 添加相等函数）
    - 将复杂类型泛化为更通用的类型（如 `List[Union(str, int)]` → `List(str)`，前提是不改变功能）
- **设计动机**：原有规则对混合类型支持不足，需要针对性增强

#### 3. 迭代生成与修复（Step III）

- **功能**：利用多个 LLM 迭代翻译 Python 代码到目标语言
- **核心思路**：
    - **多轮生成**：先用 GPT-3.5-Turbo 生成初始翻译，再用 DeepseekCoder-33B-Instruct 迭代处理
    - **自动修复**：每个 LLM 生成后，将错误代码及错误信息反馈给 LLM 进行修正
    - **早停机制**：如果连续 k 轮新增正确数量低于阈值 δ，则停止
    - **基于重叠的多轮修复**：对已被 ≥15 种语言正确翻译但剩余语言失败的题目，使用 GPT-4o 进行最后的修复
- **设计动机**：单一 LLM 的翻译成功率有限，多模型协作+迭代修复可显著提升覆盖率

最终从 800 个 Python 题目中产出 500 个跨19种语言对齐的高质量题目，另有 38 个通过人工微调补全。

### 评估任务设计

两类推理任务：
- **输出推理（Output Reasoning）**：给定代码和输入，预测输出
- **输入推理（Input Reasoning）**：给定代码和输出，预测输入
- 评估指标：Pass@1（贪心解码，temperature=0）

## 实验关键数据

### 主实验：24 个 LLM 在 19 种语言上的 Pass@1

**输入推理（部分）**：

| 模型 | 参数量 | Python | C++ | Java | JavaScript | Rust | Racket | 平均 |
|------|--------|--------|-----|------|-----------|------|--------|------|
| GPT-4o | - | 70.6 | 64.6 | 69.8 | 73.2 | 73.6 | 67.4 | ~71 |
| DeepseekCoder-V2 | 236B | 64.0 | 57.0 | 64.8 | 67.0 | 63.6 | 58.0 | ~63 |
| GPT-4o-mini | - | 59.6 | 52.2 | 57.2 | 59.6 | 61.2 | 51.2 | ~58 |
| CodeLlama-Instruct | 34B | 51.2 | 48.4 | 44.4 | 52.6 | 48.6 | 42.4 | ~48 |
| phi-1.5 | 1.3B | 25.8 | 16.0 | 26.8 | 9.8 | 25.2 | 6.6 | ~19 |
| phi-1 | 1.3B | 11.8 | 7.0 | 2.8 | 17.0 | 5.4 | 11.2 | ~11 |

**输出推理（部分）**：

| 模型 | Python | C++ | Java | JavaScript | Rust | Racket | 平均 |
|------|--------|-----|------|-----------|------|--------|------|
| GPT-4o | 75.4 | 74.8 | 73.2 | 77.6 | 74.4 | 70.8 | ~74 |
| DeepseekCoder-V2 | 66.8 | 66.2 | 67.6 | 65.4 | 65.8 | 62.2 | ~65 |
| phi-1.5 | 25.6 | 26.0 | 15.8 | 23.0 | 22.0 | 16.8 | ~22 |

### 消融实验：跨语言泛化

phi-1（仅在 Python 上训练）的表现：

| 度量 | Python | 其他18种语言平均 |
|------|--------|----------------|
| 输入推理 | 11.8% | ~10.7% |
| 输出推理 | 22.4% | ~15.1% |
| 语法正确率 | 97.0% | 49.1% |

phi-1.5（Python+自然语言增强）：

| 度量 | Python | 其他18种语言平均 |
|------|--------|----------------|
| 输入推理 | 25.8% | ~19.0% |
| 输出推理 | 25.6% | ~21.7% |
| 语法正确率 | 98.7% | 72.0% |

**关键发现**：仅在 Python 上训练的模型仍能在其他语言上达到可观的推理性能，表明 LLM 具备显著的跨语言泛化能力。

### 编程语言相关性

| 语言对 | 余弦相似度 |
|-------|----------|
| JavaScript - TypeScript | 0.87-0.91（最高） |
| Racket - 其他语言 | 最低 |
| 平均（所有语言对） | 0.7+ |

输出推理的语言间相关性略高于输入推理（0.79 vs 0.75）。

### 关键发现

1. **CruxEval-X 对所有 LLM 都有挑战性**：即便 GPT-4o 也只能达到约 70-75% 的 Pass@1
2. **输入推理和输出推理能力相当**：在多种语言上，两种推理任务的表现相似
3. **单语言模型的跨语言泛化**：phi-1（仅 Python 训练）在未见过的语言上仍能获得 16-26% 的输出推理成功率
4. **自然语言能力促进代码推理**：phi-1 到 phi-1.5 的改进主要来自自然语言数据增强，但代码推理也大幅提升
5. **语法结构相似性决定泛化程度**：PHP 与 Python 结构相似，泛化效果好；Racket 语法独特，泛化最差
6. **输入/输出长度负相关**：较长的输入/输出字符串使推理更困难

## 亮点与洞察

- **全自动构建流水线**：测试引导+多 LLM 迭代生成修复，成本远低于人工标注，质量有保障
- **填补代码推理多语言评估空白**：首个覆盖 19 种语言的代码推理基准
- **跨语言泛化发现**：单语言训练模型在未见语言上的表现揭示了 LLM 内部的语言无关推理能力
- **实用的基准构建方法论**：可推广到其他以 Python 为基础的数据集的多语言扩展

## 局限与展望

1. **模型生成数据的潜在偏差**：虽然实验表明不影响公平性，但理论上仍存在风险
2. **翻译质量非 100%**：800 题最终仅 500 题完全对齐，部分语言适配存在困难
3. **无法保留语言特有特性**：翻译自 Python 的代码无法体现目标语言的特有范式和最佳实践
4. **500 个对齐题目**：虽够区分 LLM 差异，但对深入分析特定语言仍显不足
5. **未评估最新大模型**：如 GPT-4o 之后的模型和 Claude 系列

## 相关工作与启发

- 扩展了 CruxEval（Gu et al., 2024）的单语言局限
- 与 McEval（人工标注多语言基准）互补，提供了自动化替代方案
- 跨语言泛化发现与自然语言处理中的多语言迁移学习相呼应
- 为衡量 LLM"理解代码"vs"模式匹配"提供了新角度

## 评分

- **新颖性**: ⭐⭐⭐⭐ (首个19语言代码推理基准，自动构建流水线设计精巧)
- **实验充分度**: ⭐⭐⭐⭐⭐ (24个模型×19种语言，分析维度丰富：泛化、相关性、关键因素)
- **写作质量**: ⭐⭐⭐⭐ (结构清晰，案例分析生动，但部分表格过大影响阅读)
- **价值**: ⭐⭐⭐⭐ (填补重要空白，方法论可复用，发现有启发性)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] EXECUTE: A Multilingual Benchmark for LLM Token Understanding](execute_a_multilingual_benchmark_for_llm_token_understanding.md)
- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2025\] Are Rules Meant to be Broken? Understanding Multilingual Moral Reasoning as a Computational Pipeline with UniMoral](are_rules_meant_to_be_broken_understanding_multilingual_moral_reasoning_as_a_com.md)
- [\[ACL 2025\] M3FinMeeting: A Multilingual, Multi-Sector, and Multi-Task Financial Meeting Understanding Evaluation Dataset](m3finmeeting_a_multilingual_multi-sector_and_multi-task_financial_meeting_unders.md)
- [\[ACL 2025\] KnowCoder-X: Boosting Multilingual Information Extraction via Code](knowcoder-x_boosting_multilingual_information_extraction_via_code.md)

</div>

<!-- RELATED:END -->
