---
title: >-
  [论文解读] Putnam-AXIOM: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs
description: >-
  [ICML2025][LLM推理][数学推理基准] 提出 Putnam-AXIOM —— 522 道大学级 Putnam 竞赛数学题 + 100 道程序化功能变体，揭示 LLM 数学推理中的记忆依赖，并引入 Teacher-Forced Accuracy (TFA) 作为超越最终答案的推理质量评估指标。
tags:
  - "ICML2025"
  - "LLM推理"
  - "数学推理基准"
  - "数据污染"
  - "功能变体"
  - "Teacher-Forced Accuracy"
  - "Putnam竞赛"
---

# Putnam-AXIOM: A Functional & Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs

**会议**: ICML2025  
**arXiv**: [2508.08292](https://arxiv.org/abs/2508.08292)  
**代码**: [brando90/putnam-axiom](https://github.com/brando90/putnam-axiom)  
**领域**: LLM评测 / 数学推理  
**关键词**: 数学推理基准, 数据污染, 功能变体, Teacher-Forced Accuracy, Putnam竞赛

## 一句话总结

提出 Putnam-AXIOM —— 522 道大学级 Putnam 竞赛数学题 + 100 道程序化功能变体，揭示 LLM 数学推理中的记忆依赖，并引入 Teacher-Forced Accuracy (TFA) 作为超越最终答案的推理质量评估指标。

## 研究背景与动机

**现有基准饱和**：GSM8K 已被 GPT-4 刷至 97.1%，MATH 数据集也达到 87.92%，无法区分前沿模型的推理能力差异。

**数据污染严重**：MATH、AGIEval、OlympiadBench 等基准的题目大量存在于互联网上，很可能已被纳入预训练数据，模型通过"记忆答案"而非"推理"获得高分。

**评估维度单一**：当前主流的"boxed answer"评估只看最终答案是否正确，完全忽略了推理过程的质量——模型可能靠猜测或错误推导碰巧答对。

**已有难题基准的不足**：ARB、OlympiadBench 包含大量无法自动评分的证明题，需要昂贵的人工评估；PutnamBench 侧重形式化定理证明（Lean/Isabelle/Coq），应用门槛高。

核心动机：需要一个**难度足够高、抗污染、可自动评分、且能评估推理过程**的数学推理基准。

## 方法详解

### 3.1 Putnam-AXIOM Original 数据集

- **来源**：1938–2023 年间的 William Lowell Putnam 数学竞赛题
- **规模**：522 道题，覆盖 11 个数学领域（几何、代数、三角、微积分、线性代数、组合、概率、数论、复数、微分方程、分析）
- **难度分级**：保留原始考试 ID（A/B 场次 + 1-6 编号，6 最难）
- **Modified Boxing**：对 221 道（42.3%）原本需要证明或多答案的题目，增加一个简单的计算步骤，使其产生唯一的 boxed 答案，从而可自动评分。例如：原题要求找出满足条件的所有 $n$ 值，修改为"求前 $k$ 个满足条件的 $n$ 值之和"
- **评分机制**：使用等价函数将 TeX 转为 SymPy 对象，判断数值等价性（例如 0.5 = 1/2 = \frac{1}{2}）

### 3.2 Putnam-AXIOM Variation 数据集

为对抗数据污染，对 100 道题创建程序化功能变体，分为两类：

| 变体类型 | 数量 | 操作 | 答案变化 |
|---------|------|------|---------|
| Variable Change | 63 | 替换变量名（如 $x \to w$, $y \to v$） | 不变 |
| Constant Change | 37 | 修改数值常数 + 替换变量名 | 改变 |

关键设计：每个变体通过 Python 脚本生成，可产生**无限多**等难度的新实例。评估时随机采样 5 个 snapshot，取均值和 95% 置信区间。

### 3.3 Teacher-Forced Accuracy (TFA)

TFA 直接评估模型对参考解答推理过程的"理解"程度。给定问题 $q$ 和 ground truth 解答 token 序列 $s_1, s_2, \dots, s_N$，令 $\hat{s}_i$ 为 teacher forcing 下模型的预测：

$$\text{TFA} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\hat{s}_i = s_i]$$

辅助指标还包括：

- **TFCE (Teacher-Forced Cross Entropy)**：$\text{TFCE} = -\frac{1}{N} \sum_{i=1}^{N} \log \mathbb{P}(\hat{s}_i = s_i \mid q, s_1, \dots, s_{i-1})$
- **Perplexity**：$\exp(\text{TFCE})$
- **BPC (Bits Per Character)**：按字符而非 token 归一化的交叉熵

TFA 的优势：无需额外标注数据，无需训练奖励模型，仅需一次前向传播；与最终答案准确率相关，但能惩罚"碰巧答对但推理错误"的情况。

### 3.4 数据污染模拟实验

使用 LoRA 微调模型模拟污染场景：

| 阶段 | Original 准确率 | Variation 准确率 |
|------|----------------|-----------------|
| 微调前 | 23% | 12% |
| 微调后 | **80%** | 33% |

微调后原题准确率飙升至 80%，但变体仅提升至 33% → 模型在"记忆"原题答案，而非学习推理能力。

## 实验设置与主要结果

### 实验设置

- 使用 LM Harness 评估框架，标准化 prompt 模板（要求逐步推理 + boxed 答案）
- 评测 18 个模型：7 个 Base、7 个 Instruct/RL、4 个闭源
- Variation 评估：5 个随机 snapshot 取均值，计算 95% CI

### Original 数据集结果（Table 1 精选）

| 模型 | 正确/总计 | 准确率 | TFA |
|------|----------|--------|-----|
| o1-preview | 219/522 | **41.94%** | - |
| GPT-4o | 101/522 | 19.35% | - |
| Claude-3.5 Sonnet | 83/522 | 15.96% | - |
| Qwen2-Math-7B-Instruct | 60/522 | 11.49% | 0.758 |
| GPT-4 | 59/522 | 11.30% | - |
| NuminaMath-7B-Base | 54/522 | 10.34% | 0.742 |
| Gemma-2B-Instruct | 5/522 | 0.95% | 0.634 |

最强模型 o1-preview 也只有不到 42%，大部分模型低于 10%，证明基准难度充足。

### Variation 数据集结果（准确率下降）

| 模型 | Original（100题） | Variation | 相对降幅 |
|------|-------------------|-----------|---------|
| o1-preview | 46.8% → | 下降 19.6pp | 46.8% 相对降 |
| DeepSeek-R1-Qwen-32B | - | - | **37.5%** 最大降幅 |
| GPT-4o | - | - | 36% 降幅 |

10 个模型的 Original 与 Variation 置信区间不重叠 → 差异具有统计显著性，表明模型在原题上的表现被记忆人为抬高。

### TFA 指标分析

- TFA 在 MATH 数据集上与 boxed accuracy 高度相关，是有效的推理过程代理指标
- ROSCOE 的 18 个指标在不同模型间可比性差，多数基于嵌入的指标无法有效区分推理质量
- TFA 仅需前向传播，无需额外模型或标注，成本极低

### 错误分析

o1-preview 的解答虽然大体沿正确逻辑路径推进，但**缺乏数学严谨性**——频繁使用未证明的声明来推进证明，人工阅卷下这些解答只能得到很少的分数。这揭示了 LLM 在形式推理上的根本不足。

## 局限与展望

1. **TFA 依赖参考解答**：模型可能用完全不同但正确的解法，TFA 会低估其能力；对于微调为特定风格（如代码生成）的模型尤其不公平
2. **变体覆盖有限**：仅 100/522 题有功能变体（19.2%），部分题目因缺乏可变常数或答案不可 box 化而无法生成变体
3. **Modified Boxing 可能改变题目性质**：42.3% 的题目被修改以适应自动评分，这一转换过程可能引入偏差
4. **缺少最新模型**：评测时间点限制，未包含 GPT-4o-mini、Claude-3.5-Sonnet-V2、DeepSeek-V3 等更新模型
5. **仅评估英文**：Putnam 竞赛本身为英文，未考虑多语言数学推理
6. **闭源模型无法计算 TFA**：缺乏 log probabilities 访问权限，限制了 TFA 的适用范围

## 相关工作与启发

- **MATH / GSM8K**：经典但已饱和的数学基准，Putnam-AXIOM 从难度和抗污染两方面超越
- **FrontierMath (Srivastava et al., 2024)**：首创功能变体思路，本文将其从高中级扩展到大学竞赛级
- **PutnamBench**：同源竞赛但侧重形式化定理证明（Lean/Isabelle/Coq），互补而非竞争
- **ROSCOE**：18 种推理指标套件，但跨模型可比性差；TFA 提供了更简洁实用的替代
- **过程监督 (PRM)**：需要大量步骤级标注和额外奖励模型，TFA 为零训练成本的轻量替代

## 个人点评

**优点**：
- 问题定位精准：同时解决基准饱和 + 数据污染两大痛点
- 功能变体设计巧妙：程序化生成无限新实例，原理简单但效果显著（19.6pp 下降清楚说明记忆依赖）
- TFA 指标实用：一次前向传播、无需标注、可自动计算，为推理过程评估提供了低成本方案
- 实验设计严谨：LoRA 微调实验精准模拟污染场景，5 snapshot + CI 的评估方法统计可靠

**不足**：
- 变体仅覆盖 19.2% 题目，说服力有限
- 论文分类放在 model_compression 下显然不合适，实际属于 LLM 评测/数学推理领域
- 缺少与 FrontierMath、Minerva、MathOdyssey 等同期难题基准的直接对比
- TFA 与正确率的相关性仅在 MATH 数据集上验证，未在 Putnam-AXIOM 本身上验证

## 评分

- 新颖性: ⭐⭐⭐⭐ — 功能变体 + TFA 的组合有创新价值，但各自单独看并非全新思路
- 实验充分度: ⭐⭐⭐⭐ — 18 个模型 + 微调污染模拟 + 统计检验，较为全面；缺最新模型和跨基准对比
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机明确，图表信息量大
- 价值: ⭐⭐⭐⭐ — 对 LLM 数学推理评估领域有实际贡献，GitHub 代码已开源，可复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] TimE: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios](../../NeurIPS2025/llm_reasoning/time_a_multilevel_benchmark_for_temporal_reasoning_of_llms_i.md)
- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](../../NeurIPS2025/llm_reasoning/realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[NeurIPS 2025\] CoRe: Benchmarking LLMs' Code Reasoning Capabilities through Static Analysis Tasks](../../NeurIPS2025/llm_reasoning/core_benchmarking_llms_code_reasoning_capabilities_through_static_analysis_tasks.md)
- [\[ACL 2025\] Enhancing Mathematical Reasoning in LLMs by Stepwise Correction](../../ACL2025/llm_reasoning/enhancing_mathematical_reasoning_in_llms_by_stepwise_correction.md)
- [\[ICML 2026\] FloorplanQA: A Benchmark for Spatial Reasoning in LLMs Using Structured Representations](../../ICML2026/llm_reasoning/floorplanqa_a_benchmark_for_spatial_reasoning_in_llms_using_structured_represent.md)

</div>

<!-- RELATED:END -->
