---
title: >-
  [论文解读] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models
description: >-
  [NeurIPS 2025][社会计算][数据归因] DATE-LM构建了首个面向LLM的统一数据归因评估基准，通过训练数据选择、毒性过滤和事实归因三大应用驱动任务系统评估多种归因方法，发现无单一方法全面占优且简单基线在某些场景可媲美归因方法。 领域现状：数据归因方法量化训练数据对模型输出的影响，在数据集策展、模型可解释性和…
tags:
  - "NeurIPS 2025"
  - "社会计算"
  - "数据归因"
  - "benchmark"
  - "LLM"
  - "数据选择"
  - "事实溯源"
---

# DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2507.09424](https://arxiv.org/abs/2507.09424)  
**代码**: [GitHub](https://github.com/DataAttributionEval/DATE-LM)  
**领域**: 社会计算 / 数据归因  
**关键词**: 数据归因, benchmark, LLM, 数据选择, 事实溯源

## 一句话总结
DATE-LM构建了首个面向LLM的统一数据归因评估基准，通过训练数据选择、毒性过滤和事实归因三大应用驱动任务系统评估多种归因方法，发现无单一方法全面占优且简单基线在某些场景可媲美归因方法。

## 研究背景与动机

**领域现状** 数据归因方法量化训练数据对模型输出的影响，在数据集策展、模型可解释性和数据定价等方面日益重要，已有LESS、MATES等针对LLM的方法涌现。

**现有痛点** LLM训练栈复杂导致方法间难以公平比较；基于重训练的评估协议计算成本巨大；现有工作缺乏全面的应用驱动评估，且常忽略非归因基线的对比。

**核心矛盾** 数据归因方法在LLM规模下的真实效用不明确——它们在不同应用中的性价比如何？是否值得投入高计算成本？

**本文目标** 提供统一、可扩展的基准平台，使研究者在多种LLM架构和任务上进行公平、可复现的数据归因方法对比。

**切入角度** 设计模块化三阶段评估管线（归因评分→子集选择→任务评估），配合预训练检查点和公开排行榜降低评估门槛。

**核心 idea** 通过统一管线和应用驱动的任务设计，实现首次大规模、系统性的LLM数据归因方法对比。

## 方法详解

### 整体框架
DATE-LM包含三阶段统一管线：(1) 归因评分——使用方法 $\tau$ 对训练集 $\mathcal{D}$ 中每个样本相对参考集 $\mathcal{D}_{ref}$ 打分，得到 $\mathcal{S} = \tau(\mathcal{D}, \mathcal{D}_{ref}, \theta)$；(2) 子集选择——基于 $\mathcal{S}$ 通过top-k或概率采样选取子集 $\mathcal{D}_s$；(3) 任务评估——在 $\mathcal{D}_s$ 上运行下游任务并计算指标。管线对 $\tau$ 不可知，新方法只需提供评分即可接入。

### 关键设计

1. **训练数据选择任务**:
    - 功能：评估归因方法选出高质量训练数据以提升LLM能力
    - 核心思路：预训练设定使用FineWeb+LAMBADA，通过Gumbel-top-k采样保证多样性，在SciQ/ARC/BoolQ等7个任务上评估；微调设定使用Tulu 3+MMLU/GSM8K/BBH。采用短训练（200步WSD调度）代替完整预训练降低成本
    - 设计动机：统一使用Gumbel-top-k隔离归因评分函数本身的效果，避免不同选择策略（聚类、多臂赌博等）混淆比较

2. **毒性/偏见过滤任务**:
    - 功能：评估归因方法从训练集中检测有害样本的能力
    - 核心思路：构造 $\mathcal{D} = \mathcal{D}_{benign} \cup \mathcal{D}_{unsafe}$（10K良性+<100有害），以AUPRC评估。创新引入**异构过滤**——在良性数据中加入安全对齐的"干扰"样本（如拒绝有害提问的回答），使得有害数据更难区分
    - 设计动机：同构过滤过于简单，异构设定更贴近实际安全对齐场景，是更严格但更有意义的测试

3. **事实归因任务**:
    - 功能：追溯LLM输出事实到训练中的支撑证据
    - 核心思路：基于ROME数据集，使用Recall@50和MRR评估。关键引入**反事实设定**——用相关但错误的实体（如Microsoft→Google）替换支撑证据中的实体，打破词汇重叠，迫使方法捕捉语义贡献
    - 设计动机：先前基准严重偏向词汇重叠检测，BM25轻易得到高分，反事实设定使评估更公平

## 实验关键数据

### 主实验——预训练数据选择（1B模型, 30K步）

| 方法 | 7任务平均 | FLOPS |
|------|----------|-------|
| Random | 49.83 | 1× |
| BM25 | 50.26 | 1× |
| Grad Sim | 50.26 | 11× |
| MATES | 50.13 | 1.13× |
| EDU | **50.63** | 1.07× |

### 毒性过滤（平均AUPRC）

| 方法 | 同构 | 异构 |
|------|------|------|
| WildGuard（基线） | **0.827** | **0.817** |
| LESS | 0.704 | 0.515 |
| Grad Sim | 0.584 | 0.466 |

### 消融实验——Gumbel温度敏感性

| 温度 | 效果 |
|------|------|
| 过低（0.1） | 多样性不足，部分方法低于Random |
| 适中（0.5-1.0） | 多数方法达最优 |
| 过高（2.0） | 随机性过强，退化为均匀采样 |

### 关键发现
- 无单一归因方法在所有任务上占优
- 简单非归因基线（EDU、WildGuard）多任务上匹配甚至超越归因方法
- 异构过滤下归因方法性能暴跌（如LESS从0.704降至0.515）
- 评估设计选择（Gumbel温度、参考集组合）对结果影响巨大

## 亮点与洞察
- 首个带公开排行榜的LLM数据归因统一基准，含完整评估管线和预训练检查点
- 反事实事实归因和异构毒性过滤揭示了先前评估的系统偏差
- "高成本归因方法 vs 低成本基线"的性价比分析具有重要实践意义

## 局限与展望
- 评估仅限1B-8B模型，更大规模上结论可能不同
- 短训练代理可能无法反映长期训练效果
- 微调评估仅考虑单任务设定，多任务场景待探索

## 相关工作与启发
- **vs TRAK/LDS**: DATE-LM用应用驱动评估替代昂贵的LOO评估，更贴近实际但评估角度不同
- **vs LESS**: LESS在基准上表现中等偏上而非全面最优，说明需要更多元的评估视角
- **vs Quanda**: 面向通用ML的框架，DATE-LM专门面向LLM并提供检查点降低门槛

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个综合LLM数据归因基准，反事实设定和异构过滤有创新
- 实验充分度: ⭐⭐⭐⭐⭐ 8种方法、3大任务、多模型规模，严谨全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰动机充分
- 价值: ⭐⭐⭐⭐ 提供急需的标准化评估工具和重要实验洞察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Uncovering Strategic Egoism Behaviors in Large Language Models](uncovering_strategic_egoism_behaviors_in_large_language_models.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](active_slice_discovery_in_large_language_models.md)
- [\[ICML 2025\] OR-Bench: An Over-Refusal Benchmark for Large Language Models](../../ICML2025/social_computing/or-bench_an_over-refusal_benchmark_for_large_language_models.md)
- [\[ACL 2026\] SMARTER: A Data-efficient Framework to Improve Toxicity Detection with Explanation via Self-augmenting Large Language Models](../../ACL2026/social_computing/smarter_a_data-efficient_framework_to_improve_toxicity_detection_with_explanatio.md)
- [\[ACL 2025\] Exploring Gender Bias in Large Language Models: An In-depth Dive into the German Language](../../ACL2025/social_computing/exploring_gender_bias_in_large_language_models_an_in-depth_dive_into_the_german_.md)

</div>

<!-- RELATED:END -->
