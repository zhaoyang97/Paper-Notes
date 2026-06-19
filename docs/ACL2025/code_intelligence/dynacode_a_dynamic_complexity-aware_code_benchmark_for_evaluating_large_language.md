---
title: >-
  [论文解读] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation
description: >-
  [ACL 2025][代码智能][代码生成] 提出 DynaCode，一个动态复杂度感知的代码生成基准，通过将代码问题按圈复杂度分类并用调用图（Call Graph）组合嵌套，生成约 1.89 亿个唯一问题，有效缓解数据污染并系统评估 LLM 在不同复杂度下的代码生成能力。 领域现状：HumanEval、MBPP 等静态基准…
tags:
  - "ACL 2025"
  - "代码智能"
  - "代码生成"
  - "动态评估"
  - "数据污染"
  - "复杂度感知"
  - "调用图"
---

# DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation

**会议**: ACL 2025  
**arXiv**: [2503.10452](https://arxiv.org/abs/2503.10452)  
**代码**: [https://github.com/HWH-2000/DynaCode](https://github.com/HWH-2000/DynaCode)  
**领域**: 代码智能  
**关键词**: 代码生成, 动态评估, 数据污染, 复杂度感知, 调用图

## 一句话总结

提出 DynaCode，一个动态复杂度感知的代码生成基准，通过将代码问题按圈复杂度分类并用调用图（Call Graph）组合嵌套，生成约 1.89 亿个唯一问题，有效缓解数据污染并系统评估 LLM 在不同复杂度下的代码生成能力。

## 研究背景与动机

**领域现状**：HumanEval、MBPP 等静态基准是评估 LLM 代码生成的标准工具。EvalPlus、BigCodeBench、CRUXEval 等改进了测试用例和评估粒度，但核心局限未变。

**现有痛点**：(1) **数据污染**——静态基准规模小且公开，模型可能在训练时记忆了测试用例，如 Meta-Llama-3-8B-Instruct 和 Phi-2 已被报告存在数据污染；(2) **复杂度不可控**——现有基准缺乏系统的复杂度控制，用行数或时间复杂度衡量过于粗糙，无法捕捉深层嵌套和复杂执行依赖。

**核心矛盾**：需要一个既能抵抗记忆化又能细粒度区分模型能力的基准，而静态小规模数据集天然做不到这两点。

**本文目标**：构建一个动态、大规模、复杂度可控的代码评估框架，能自动生成海量唯一问题并按复杂度分级评估。

**切入角度**：将独立代码问题视为"节点"，通过调用图（有向无环图）将多个问题组合成嵌套代码问题，同时从代码复杂度和图复杂度两个维度构建复杂度矩阵。

**核心 idea**：用圈复杂度分类代码问题 + 调用图结构组合嵌套 = 动态生成 1.89 亿唯一问题的二维复杂度感知基准。

## 方法详解

### 整体框架

DynaCode 构建分为四步：(1) 收集代码问题并按圈复杂度分类（Code Complexity Classification）；(2) 设计 16 种调用图结构（Call Graph Construction）；(3) 组合代码复杂度和图复杂度构建二维复杂度矩阵（Complexity-aware Metrics）；(4) 按调用图将问题组合成嵌套问题并生成测试用例（Benchmark Generation）。

### 关键设计

1. **代码复杂度分类**：使用圈复杂度（Cyclomatic Complexity）$\nu = E - N + 2P$ （$E$ 为控制流图边数、$N$ 为节点数、$P$ 为连通分量数）对 ground truth 代码进行分析。通过 Radon 工具计算，将问题分为 4 个 Unit：

    - Unit 1：简单顺序/基本逻辑
    - Unit 2：含分支和简单循环
    - Unit 3：多分支嵌套
    - Unit 4：复杂递归和深层控制流

2. **调用图构建**：定义有向无环图 $G = (V, E)$，最多 5 个节点，根节点唯一。共 16 种不同结构，从简单线性调用到复杂多分支依赖。图复杂度度量为：
    $\mathcal{M}(G) = L_{\max}(G) \times B(G) \times |E|$
   其中 $L_{\max}$ 为最长路径、$B$ 为分支数、$|E|$ 为边数。按阈值分为 4 个 Level。

3. **二维复杂度矩阵**：$\mathcal{C} = \{c_{\xi,\eta} \mid 1 \leq \xi \leq n, 1 \leq \eta \leq m\}$，其中 $\xi$ 对应代码复杂度 Unit，$\eta$ 对应调用图 Level，提供细粒度的能力评估框架。

4. **动态生成与抗污染**：(1) 通过 Monkeytype 自动生成输入输出类型注释，按类型匹配将问题嵌套组合；(2) 持续从 LeetCode 等来源引入新问题更新基准；(3) 生成约 **1.89 亿** 唯一嵌套代码问题。

### 测试用例生成

从调用图根节点注入输入值，批量执行嵌套代码，过滤掉执行出错的"坏生成"，保留有效嵌套代码和测试用例。

## 实验关键数据

### 主实验（Pass@1）

| 模型 | MBPP | MBPP+ | DynaCode Avg | Unit 1 | Unit 2 | Unit 3 | Unit 4 |
|------|------|-------|-------------|--------|--------|--------|--------|
| GPT-4o | 87.6 | 72.2 | **55.4** | 74.4 | 48.7 | 56.2 | 42.3 |
| DeepSeek-V3 | 87.6 | 73.0 | 52.1 | 65.9 | 41.6 | 53.6 | 47.3 |
| Qwen2.5-Coder-32B | 90.5 | 77.0 | 43.2 | 59.3 | 33.6 | 44.1 | 36.0 |
| Llama-3.1-405B | 88.4 | 73.0 | 41.0 | 49.7 | 40.0 | 47.6 | 26.9 |
| GPT-3.5-Turbo | 82.5 | 69.7 | 29.3 | 34.9 | 30.5 | 25.6 | 26.1 |
| Llama-3.1-8B | 68.3 | 55.6 | 9.9 | 14.1 | 9.7 | 8.4 | 7.4 |

### 数据污染验证（微调实验）

| 模型 | 微调前 MBPP+ | 微调后 MBPP+ | 微调前 DynaCode | 微调后 DynaCode |
|------|------------|------------|---------------|---------------|
| GPT-3.5-Turbo | 69.7 | 88.6 (+18.9) | 32.6 | 36.0 (+3.4) |
| Llama-3.1-8B | 55.6 | 98.1 (+42.5) | 10.6 | 23.6 (+13.0) |

微调在 MBPP+ 上大幅提升（记忆化），但在 DynaCode 上提升很小，证明 DynaCode 有效抗数据污染。

### 错误分析（GPT-3.5-Turbo）

| 能力 | Unit 1 | Unit 2 | Unit 3 | Unit 4 |
|------|--------|--------|--------|--------|
| 问题理解 | 64.1% | 79.9% | 88.2% | 88.8% |
| 代码模式生成 | 6.6% | 0.4% | 0.2% | 1.5% |
| 上下文管理 | 29.4% | 19.7% | 11.7% | 9.4% |

### 关键发现

- **平均性能下降 16.8%~45.7%**：相比 MBPP+，所有模型在 DynaCode 上均大幅下降，GPT-3.5 下降 40.4 个百分点
- **LLM 擅长顺序执行**：在线性调用图 {G1-G4, G8} 上表现最好，多分支结构 {G9-G16} 上显著下降
- **复杂度越高理解越差**：问题理解错误从 Unit 1 的 64.1% 上升到 Unit 4 的 88.8%，说明复杂嵌套使模型连题意都理解不了
- **75 题即够稳定**：敏感性分析显示 75 个以上动态生成问题即可产生稳定评估结果
- **GPT-4o 抗复杂度能力最强**：在各 Level 上表现最稳健，但高复杂度图上仍明显下降

## 亮点与洞察

- 调用图组合的设计非常巧妙：既实现了动态化（海量组合）、又引入了结构复杂度维度
- 二维复杂度矩阵（代码复杂度 × 图复杂度）提供了前所未有的细粒度评估视角
- 数据污染验证实验设计严谨：微调对比 + 仅用 unit function 微调的对照实验
- 错误类型分析深入：揭示了随复杂度增长，错误从"上下文管理"转向"问题理解"的规律
- 1.89 亿问题的规模使得模型记忆化几乎不可能

## 局限与展望

1. **调用图规模有限**：最多 5 个节点，未来高性能 LLM 可能学会处理这种模式，需扩展更大规模图
2. **仅支持 Python**：Code 问题来自 MBPP+，只有 Python 语言
3. **圈复杂度不够全面**：未考虑递归深度、数据依赖复杂度等维度
4. **组合可能引入伪题**：自动类型匹配和嵌套组合可能产生语义不自然的问题

## 相关工作与启发

- **动态评估**：DyVal（图生成）、NPHardEval（NP 难问题）、DyVal2（LLM agent 变换）——主要面向推理，本文首次面向代码
- **代码基准**：HumanEval → EvalPlus → BigCodeBench → SWE-Bench，复杂度递增但均为静态
- **启发**：调用图思路可扩展到更大规模（10+ 节点）、引入递归调用图；可与 SWE-Bench 类真实场景结合

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 动态 + 复杂度感知的双重创新，调用图组合设计精巧
- **实验充分度**: ⭐⭐⭐⭐ — 12 模型、4 Unit × 4 Level、数据污染验证、错误分析
- **写作质量**: ⭐⭐⭐⭐ — 公式清晰，图示丰富
- **价值**: ⭐⭐⭐⭐⭐ — 解决数据污染+复杂度评估两大痛点，实用性极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] CodeIF: Benchmarking the Instruction-Following Capabilities of Large Language Models for Code Generation](codeif_benchmarking_the_instruction-following_capabilities_of_large_language_mod.md)
- [\[ACL 2025\] Personality-Guided Code Generation Using Large Language Models](personality_guided_code_gen.md)
- [\[ACL 2025\] TeXpert: A Multi-Level Benchmark for Evaluating LaTeX Code Generation by LLMs](texpert_a_multi-level_benchmark_for_evaluating_latex_code_generation_by_llms.md)
- [\[ACL 2025\] CodeReviewQA: The Code Review Comprehension Assessment for Large Language Models](codereviewqa_the_code_review_comprehension_assessment_for_large_language_models.md)
- [\[ACL 2025\] FEA-Bench: A Benchmark for Evaluating Repository-Level Code Generation for Feature Implementation](feabench_repo_code_gen.md)

</div>

<!-- RELATED:END -->
