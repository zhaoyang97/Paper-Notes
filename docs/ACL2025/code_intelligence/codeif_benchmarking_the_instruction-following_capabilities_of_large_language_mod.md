---
title: >-
  [论文解读] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation
description: >-
  [ACL 2025][代码智能][代码生成] 提出 CodeIF，第一个系统性评估 LLM 在代码生成中指令遵循能力的基准，含 8 大类 50 个细粒度约束指令、4 种新评估指标，并对 35 个 SOTA 模型进行全面评估。 领域现状：LLM 在代码生成领域取得显著进展，但其理解和执行复杂指令的能力仍是关键挑战…
tags:
  - "ACL 2025"
  - "代码智能"
  - "代码生成"
  - "指令遵循"
  - "评估基准"
  - "约束满足"
  - "LLM评估"
---

# CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation

**会议**: ACL 2025  
**arXiv**: [2502.19166](https://arxiv.org/abs/2502.19166)  
**代码**: [https://github.com/lin-rany/codeIF](https://github.com/lin-rany/codeIF)  
**领域**: 代码智能  
**关键词**: 代码生成, 指令遵循, 评估基准, 约束满足, LLM评估

## 一句话总结

提出 CodeIF，第一个系统性评估 LLM 在代码生成中指令遵循能力的基准，含 8 大类 50 个细粒度约束指令、4 种新评估指标，并对 35 个 SOTA 模型进行全面评估。

## 研究背景与动机

**领域现状**：LLM 在代码生成领域取得显著进展，但其理解和执行复杂指令的能力仍是关键挑战。现有评估框架（如 HumanEval、MBPP）主要关注代码的功能正确性，缺乏对指令遵循维度的系统评估。

**现有痛点**：(1) 现有代码基准不考量模型是否遵循了全局格式、命名规范、结构控制等约束；(2) 缺少面向多约束问题的评估指标，无法区分模型在不同约束类型上的表现差异；(3) 指令遵循评估领域（如 FollowBench、InfoBench）虽然丰富，但均未针对代码生成场景设计。

**核心矛盾**：代码生成实际中需要同时满足功能需求和编码规范约束，但现有评估只关心能否"跑通"，忽略了"写得好不好"和"是否遵守约束"。

**本文目标**：构建一个多语言、多难度、多约束维度的代码生成指令遵循基准，并提出可量化模型遵循能力的评估指标。

**切入角度**：从编码约束出发，将约束拆解为不可再分的原子指令（Atomic Instructions），并建模指令间的依赖关系，实现二值化客观评估。

**核心 idea**：将代码指令遵循分解为 8 类 50 种原子约束，配合 4 种层次化评估指标，系统刻画 LLM 在约束满足、逻辑连贯性和依赖处理上的能力。

## 方法详解

### 整体框架

CodeIF 的构建分为三步：(1) 构建约束指令体系（8 大类 50 细粒度子指令）；(2) 基于 McEval 和 FullStackBench 等基准，结合 GPT-4 自动生成约束指令列表，覆盖 Java/Python/Go/C++ 四种语言；(3) 利用 LLM 建模原子约束之间的依赖关系。

### 关键设计

1. **约束指令体系**：8 个类别覆盖不同抽象层次——Global（全局规范）、Structural Control（控制流/数据结构）、Variable（命名/初始化）、Interface（接口约束）、Function（函数约束）、Class（类约束）、File（文件/库约束）、Combination（跨维度组合）。每个原子指令设计为二值评判（Yes/No），避免主观判断。

2. **多语言多难度设计**：数据集覆盖 Java（353）、Python（348）、C++（269）、Go（230），分为 Easy（平均 11.99 条指令/任务）和 Hard（平均 13.80 条指令/任务）两个难度，共 1200 个任务。

3. **指令依赖建模**：使用 LLM 自动标注原子约束间的前置依赖关系，例如"创建函数体"依赖于"命名函数"和"定义参数类型"。这种依赖结构用于支持更严格的评估指标 RSR。

4. **自动化生成流程**：利用 GPT-4 基于 20 个详细示例自动生成任务级约束指令列表，经人工审核确保质量。

### 评估指标

本文提出四个互补的评估指标，对于含 $m$ 个问题、每个问题 $n_i$ 个约束的数据集：

1. **CSR (Completely Satisfaction Rate)**：$\text{CSR} = \frac{1}{m}\sum_{i=1}^{m}\prod_{j=1}^{n_i} r_{i,j}$，所有约束全部满足才计分，最严格指标。

2. **SSR (Soft Satisfaction Rate)**：$\text{SSR} = \frac{1}{m}\sum_{i=1}^{m}\frac{\sum_{j=1}^{n_i} r_{i,j}}{n_i}$，计算每题平均约束满足比例，更柔和。

3. **RSR (Rigorous Satisfaction Rate)**：在 SSR 基础上引入依赖关系，约束 $j$ 被满足的前提是其所有前置依赖 $D_{i,j}$ 也被满足：$\text{RSR} = \frac{1}{m}\sum_{i=1}^{m}\frac{\sum_{j=1}^{n_i}[r_{i,j}\cdot\prod_{k\in D_{i,j}}r_{i,k}]}{n_i}$。

4. **CCSR (Consistent Continuity Satisfaction Rate)**：$\text{CCSR} = \frac{1}{m}\sum_{i=1}^{m}\frac{L_i}{n_i}$，其中 $L_i$ 为最长连续满足约束序列长度，衡量持续遵循能力。

## 实验关键数据

### 主实验

| 模型 | CSR(Full) | SSR(Full) | RSR(Full) | CCSR(Full) |
|------|-----------|-----------|-----------|------------|
| DeepSeek-V3 | **0.414** | **0.821** | **0.764** | **0.712** |
| Claude-3-5-Sonnet | 0.444 | 0.727 | 0.692 | 0.652 |
| GPT-4O | 0.383 | 0.748 | 0.689 | 0.650 |
| Gemini-Exp-1206 | 0.357 | 0.744 | 0.685 | 0.636 |
| Qwen2.5-Coder-32B | 0.365 | 0.736 | 0.679 | 0.634 |
| Llama-3.3-70B | 0.307 | 0.698 | 0.632 | 0.589 |
| Llama-3.2-1B | 0.034 | 0.218 | 0.182 | 0.152 |

### 消融实验

| 难度 | GPT-4O CSR | DeepSeek-V3 CSR | Claude CSR |
|------|-----------|----------------|-----------|
| Easy | 0.441 | 0.468 | 0.525 |
| Hard | 0.325 | 0.359 | 0.362 |

Hard 任务上最好的 CSR 仅为 0.362（Claude），说明严格约束满足极具挑战性。

### 关键发现

- **DeepSeek-V3 综合最强**：在 SSR/RSR/CCSR 三项全面领先，尤其在组合约束任务上达到 0.831 的 SSR。
- **Claude-3-5-Sonnet CSR 最高**：在 Java 上 CSR 达 0.504，但 Python 上 SSR 仅 0.703。
- **模型规模正相关但非绝对**：Qwen2.5 系列展示了明显的规模效应，但 Llama3 系列则不一致。
- **闭源 > 开源**：GPT-4O、Claude 在复杂约束上显著优于同规模开源模型。
- **C++ 是最难语言**：复杂模板元编程导致各模型表现最差。
- **常见偏离类型**：命名规范（如要求 PascalCase 但输出 snake_case）、忽视禁止性指令（被要求不用 if 但仍使用）。

## 亮点与洞察

- 首个针对代码生成中指令遵循的专用基准，填补了重要空白
- 4 指标体系层次化很精巧：CSR 看全满足、SSR 看平均、RSR 看依赖链、CCSR 看连续性
- 原子指令的二值化设计避免了主观评判，GPT-4 自动评估与人类标注的 Pearson 相关达 0.87
- 指令依赖建模是独到之处，RSR 指标可捕捉级联失败

## 局限与展望

1. **语言覆盖不足**：仅含 Java/Python/Go/C++，缺少 JavaScript、Rust、Swift 等流行语言
2. **静态评估为主**：重点在代码结构和命名规范，忽略运行时行为、性能、调试能力
3. **指标权重均一**：所有约束等权处理，但实际中语法正确性远比命名规范重要
4. **评估自动化依赖 GPT-4**：二值判断依赖 GPT-4-1106-Preview，成本较高且可能引入偏差

## 相关工作与启发

- **代码基准**：HumanEval、MBPP 关注功能正确性，EvalPlus 增强测试用例，BigCodeBench 扩展函数调用——均不涉及指令遵循
- **指令遵循基准**：InfoBench、FollowBench、CFBench 评估文本生成中的指令遵循——本文将此扩展到代码域
- **启发**：可将约束遵循能力与代码功能正确性联合评估；约束依赖图可用于诊断 LLM 的推理链断裂点

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首个代码指令遵循基准，定位精准
- **实验充分度**: ⭐⭐⭐⭐⭐ — 35 个模型、4 语言、2 难度、多维分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富
- **价值**: ⭐⭐⭐⭐ — 为代码生成评估提供新维度，对模型改进有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Tree-of-Evolution: Tree-Structured Instruction Evolution for Code Generation in Large Language Models](tree_of_evolution_code_gen.md)
- [\[ACL 2025\] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation](dynacode_a_dynamic_complexity-aware_code_benchmark_for_evaluating_large_language.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)
- [\[ACL 2025\] LongCodeU: Benchmarking Long-Context Language Models on Long Code Understanding](benchmarking_long-context_language_models_on_long_code_understanding.md)

</div>

<!-- RELATED:END -->
