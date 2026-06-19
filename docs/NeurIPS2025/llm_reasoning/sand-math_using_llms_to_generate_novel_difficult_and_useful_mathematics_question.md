---
title: >-
  [论文解读] SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers
description: >-
  [NeurIPS 2025][LLM推理][数学推理] 提出 SAND-Math，一个无需种子数据集的全自动合成数学问题生成管线，通过 Difficulty Hiking 系统性提升题目难度，仅 500 道增强 LIMO 基线即可在 AIME25 上提升 4.39pp。 领域现状： DeepSeek-R1、o3、Gemini…
tags:
  - "NeurIPS 2025"
  - "LLM推理"
  - "数学推理"
  - "合成数据"
  - "Difficulty Hiking"
  - "数据质量"
  - "后训练"
---

# SAND-Math: Using LLMs to Generate Novel, Difficult and Useful Mathematics Questions and Answers

**会议**: NeurIPS 2025  
**arXiv**: [2507.20527](https://arxiv.org/abs/2507.20527)  
**代码**: [HuggingFace Dataset](https://huggingface.co/datasets/amd/SAND-MATH)  
**领域**: 音频/语音 (LLM数学推理)  
**关键词**: 数学推理, 合成数据, Difficulty Hiking, 数据质量, 后训练

## 一句话总结

提出 SAND-Math，一个无需种子数据集的全自动合成数学问题生成管线，通过 Difficulty Hiking 系统性提升题目难度，仅 500 道增强 LIMO 基线即可在 AIME25 上提升 4.39pp。

## 研究背景与动机

**领域现状**: DeepSeek-R1、o3、Gemini 2.5 Pro 等前沿推理模型在数学基准上表现卓越，但训练数据和方法未公开。LIMO 和 S1 的研究表明推理能力更依赖数据质量（尤其是高难度问题）而非规模。

**现有痛点**: (a) NuminaMath、OpenR1 等依赖人工策展的竞赛题，劳动密集且有限；(b) KPDDS、MetaMathQA、WizardMath 等合成方法主要 remix 现有 GSM8K/MATH 训练集，难以超越种子数据难度；(c) MATH2 需要人类专家参与，无法规模化。

**核心矛盾**: 高质量高难度数学训练数据的供给严重不足，而现有合成方法被种子数据难度天花板限制。

**本文目标**: 构建完全自动化的管线，从零生成高难度数学问题，并确保正确性、新颖性和难度递增。

**切入角度**: 利用 SOTA LLM 的"元认知"能力——它们能隐式建模高难度数学问题的特征并生成新问题。

**核心 idea**: LLM 自身可以从最小提示出发生成高难度数学题，再通过 Difficulty Hiking（跨领域概念融合）进一步提升难度。

## 方法详解

### 整体框架

五阶段管线：生成 → 正确性过滤 → 去重去污染 → 难度过滤与评分 → 新颖性过滤，加上可选的 Difficulty Hiking。

### 关键设计

#### 1. 问题与解答生成
- 教师模型 $\mathcal{M}_{\text{teacher}}$（DeepSeek-R1）从经验优化的 prompt 直接生成问题 $q_i$ 和 $k=2$ 个独立解答
- 初始得到 $\mathcal{D}_1 = 23{,}437$ 道题

#### 2. 自一致性过滤（Self-Consistency）
- 仅保留 $k$ 个解答一致的问题：$a'_{i1} = a'_{i2} = \cdots = a'_{ik}$
- 保留 17,578 道（~74%）

#### 3. 去重 + 去污染
- **语义去重**: semhash 框架，0.99 相似度阈值，去除 1,293 条（7.3%）
- **去污染**: 检索模型找 top-5 候选 + 判断模型语义验证，仅移除 4 道题
- 剩余 16,281 道

#### 4. 难度过滤 + 评分
- **性能过滤**: 求解模型 $\mathcal{M}_{\text{solver}}$（Qwen2.5-32B）答错的题目才保留 → 9,211 道（56.6%）
- **难度评分**: 判断模型 $\mathcal{M}_{\text{judge}}$（Llama-3.3-70B）打 1-10 分（参考 AoPS 的 AIME 题目标准）

#### 5. 新颖性过滤
- 网络搜索 + 语义相似度（gte-Qwen2-7B 嵌入），阈值 $\tau=0.85$
- 去除 4%，最终得到 **8,842 道 SAND-Math 题目**

#### 6. Difficulty Hiking（核心创新）
- 重新提示教师模型改写问题，输入包括：原题、难度评分、同分支定理 + 跨领域概念的强制融合
- 单次迭代：难度均值从 5.02 → 5.98
- 将中低难度题（4.0-5.0）转化为高难度题（6.0-8.0）

### 损失函数/训练策略

- 学生模型：Qwen2.5-32B-Instruct
- 全参数 SFT，LLaMA-Factory 框架
- 学习率 5e-6，10 epochs，cosine scheduler
- DeepSpeed ZeRO-3, 8× AMD MI300X GPU
- 评估：pass@1 (n=16, temp=0.7) for AIME/AMC, greedy for MATH500

## 实验关键数据

### 主实验 — 增强效果对比

| 训练数据 | 样本量 | AIME25 | AIME24 | AMC | MATH500 | 平均 |
|---------|-------|--------|--------|-----|---------|------|
| LIMO 基线 | 817 | 44.50 | 56.30 | 91.41 | 93.80 | 71.50 |
| LIMO + SAND-Math | 817+500 | **48.89** | **57.92** | **92.50** | **94.00** | **73.32** |
| LIMO + openr1_math | 817+500 | 47.71 | 56.04 | 92.50 | 93.80 | 72.51 |
| LIMO + MetamathQA | 817+500 | 31.04 | 46.25 | 47.24 | 56.40 | 45.23 |
| LIMO + OpenMathInstruct | 817+500 | 18.13 | 38.96 | 64.53 | 72.40 | 48.50 |

SAND-Math 在 AIME25 上超越次佳合成数据集 MetamathQA **17.85pp**。

### 消融实验 — Difficulty Hiking 效果

| 数据集 | AIME25 | AIME24 | AMC24 | MATH500 | 平均 |
|-------|--------|--------|-------|---------|------|
| LIMO + Base (1500) | 46.38 | 59.09 | 92.71 | 93.6 | 72.94 |
| LIMO + DH (1500) | **49.23** | **60.55** | **93.17** | **94.6** | **74.39** |
| LIMO + DH_w_LF (1500) | 49.23 | 60.83 | 93.28 | 93.0 | 74.08 |

Difficulty Hiking 提升均值 72.94 → 74.39（+1.45pp），长推理链也有帮助。

### 关键发现

1. SAND-Math 独立微调 (69.10) 已接近人工策展的 openr1_math (70.27)
2. 增强场景下 SAND-Math 优于所有数据集（+0.81pp 超 openr1_math）
3. 难度分布集中在 6-8（远高于其他合成数据集的 3-5）
4. 管线总体产出率 ~35%（from 23K → 8.8K）

## 亮点与洞察

- **无种子数据**: 完全从零生成，不依赖 GSM8K/MATH 等现有训练集
- **Difficulty Hiking 的优雅设计**: 利用 LLM 的元认知能力，通过跨领域概念融合系统性提升复杂度
- **极高的样本效率**: 仅 500 道 SAND-Math 题目就能显著增强强基线
- **完整的质量保障管线**: 自一致性 → 去重 → 去污染 → 难度过滤 → 新颖性过滤，每一步都有量化验证

## 局限与展望

1. 输出质量上限受教师模型能力限制（DeepSeek-R1 产出率 41.9%，GPT-OSS 120B 可达 74.0%）
2. 仅用 500 样本做概念验证，大规模后训练待探索
3. Difficulty Hiking 仅展示单次迭代，多次迭代效果未知
4. 局限于数学领域，泛化到科学/代码推理待验证

## 相关工作与启发

- **LIMO** (Ye et al., 2025): 少即是多的推理数据哲学
- **S1** (Muennighoff et al., 2025): 简单的测试时缩放
- **MetaMathQA** (Yu et al., 2023): 基于种子数据的数学问题 bootstrap
- **启发**: 高质量 > 大规模，元认知能力可被系统性利用于数据生成

## 评分

⭐⭐⭐⭐ (4/5)
方法简洁有效，Difficulty Hiking 思路新颖，但实验规模偏小（仅 500 样本验证），且对教师模型的依赖性强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](realmath_a_continuous_benchmark_for_evaluating_language_models_on_research-level.md)
- [\[ICML 2025\] MARGE: Improving Math Reasoning for LLMs with Guided Exploration](../../ICML2025/llm_reasoning/marge_improving_math_reasoning_for_llms_with_guided_exploration.md)
- [\[AAAI 2026\] ARCHE: A Novel Task to Evaluate LLMs on Latent Reasoning Chain Extraction](../../AAAI2026/llm_reasoning/arche_a_novel_task_to_evaluate_llms_on_latent_reasoning_chai.md)
- [\[ICLR 2026\] From Abstract to Contextual: What LLMs Still Cannot Do in Mathematics](../../ICLR2026/llm_reasoning/from_abstract_to_contextual_what_llms_still_cannot_do_in_math_word_problem_solvi.md)
- [\[ICLR 2026\] DAG-Math: Graph-of-Thought Guided Mathematical Reasoning in LLMs](../../ICLR2026/llm_reasoning/dag-math_graph-of-thought_guided_mathematical_reasoning_in_llms.md)

</div>

<!-- RELATED:END -->
