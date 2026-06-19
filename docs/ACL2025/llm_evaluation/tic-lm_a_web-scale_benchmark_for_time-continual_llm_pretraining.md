---
title: >-
  [论文解读] TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining
description: >-
  [ACL 2025][LLM评测][持续预训练] 本文提出TiC-LM，一个基于114个月Common Crawl数据（2.9T tokens）的大规模时间持续学习基准，通过150+实验系统评估了优化器、数据回放和正则化方法在持续预训练场景下的表现，发现自回归学习率调度结合固定比例数据回放可以在仅2.6倍计算量下接近从头训练的性能。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "持续预训练"
  - "时间分布漂移"
  - "数据回放"
  - "Common Crawl"
  - "遗忘问题"
---

# TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining

**会议**: ACL 2025  
**arXiv**: [2504.02107](https://arxiv.org/abs/2504.02107)  
**代码**: [GitHub](https://github.com/apple/ml-tic-lm)  
**领域**: LLM评测  
**关键词**: 持续预训练, 时间分布漂移, 数据回放, Common Crawl, 遗忘问题

## 一句话总结

本文提出TiC-LM，一个基于114个月Common Crawl数据（2.9T tokens）的大规模时间持续学习基准，通过150+实验系统评估了优化器、数据回放和正则化方法在持续预训练场景下的表现，发现自回归学习率调度结合固定比例数据回放可以在仅2.6倍计算量下接近从头训练的性能。

## 研究背景与动机

**领域现状**：大语言模型（LLM）通常在大量历史网络数据上从头训练，但训练数据存在知识截止问题，模型在新数据上的性能会随时间衰减。同时从头重新训练LLM的计算成本极其高昂。

**现有痛点**：现有持续语言建模的研究存在严重的规模和范围局限：（1）多数先前工作只在单一领域（如Wikipedia、新闻、Twitter）上训练和评估；（2）更近期的大规模研究虽然在通用网络数据上进行，但不关注时间分布漂移且训练轮次不超过3轮；（3）缺少跨多个时间步和领域的系统性评估。

**核心矛盾**：实际LLM训练使用通用网络数据并需要在多种任务上表现良好，但现有基准无法模拟这一场景。需要一个规模足够大、时间跨度足够长的基准来真正研究LLM持续预训练的有效策略。

**本文目标** (1) 持续预训练能否以更低成本匹配定期从头训练；(2) 在通用网络数据上持续训练时遗忘是否是核心挑战；(3) 遗忘的影响是否因领域而异。

**切入角度**：受TiC-CLIP启发，构建一个基于完整Common Crawl时间序列的超大规模基准，将月度CC dump作为自然的时间分布漂移单元。

**核心 idea**：通过构建覆盖114个月、2.9T tokens的时间层化网络语料基准，系统研究LLM持续预训练中不同策略在学习新知识与保留旧知识之间的权衡。

## 方法详解

### 整体框架

TiC-LM模拟一个设置：每个月的CC dump逐一揭示，LLM首先在初始月份数据上预训练，然后每月在固定token预算内持续更新（可选回放旧数据）。评估覆盖通用CC数据和特定领域（Wikipedia、StackExchange、代码文档）的时间层化评测。

### 关键设计

1. **TiC-CommonCrawl (TiC-CC) 数据集**:

    - 功能：提供大规模时间层化的训练和评估数据
    - 核心思路：收集2013年5月至2024年7月的114个CC月度dump，基于DataComp-LM管线进行处理：使用resiliparse提取纯文本，应用RefinedWeb的启发式过滤器，执行月份内（而非跨月份）的模糊去重。不使用DCLM-Baseline的分类器过滤器以保持因果性。总计29T tokens，实验使用2.9T的子集。
    - 设计动机：保持因果性（不用未来数据处理过去数据），保留自然的时间分布漂移，比先前基准提供100倍以上的tokens和10倍以上的时间步

2. **多维度时间层化评估体系**:

    - 功能：全面评估持续训练模型在不同时间和领域上的表现
    - 核心思路：设计四类评估：(1) TiC-CC：通用CC数据的月度held-out perplexity（含Wiki子集和News子集）；(2) TiC-Wikipedia：基于完整Wikipedia dump的10年跨度评估，使用专有名词perplexity捕捉事实变化；(3) TiC-StackExchange：8-170个月跨度的QA answer perplexity；(4) TiC-CodeDocs：NumPy和PyTorch官方文档16个版本的perplexity。另外还有DCLM Core的22个静态下游任务。
    - 设计动机：区分ID/Backward/Forward性能，捕捉遗忘和适应新数据的权衡，不同领域对回放的需求可能不同

3. **持续学习方法系统评估**:

    - 功能：在统一框架下比较三大类持续学习策略
    - 核心思路：(1) **优化器方法**：Cyclic Cosine（月内余弦衰减）、Cyclic Cosine + AR（跨轮次自回归衰减最大学习率）、Rsqrt（倒数平方根无限调度）、Schedule-Free（迭代平均优化器）；(2) **数据回放**：当前月份分配比例 $\alpha_t$ 的token，剩余均匀分配给历史月份，尝试 $\alpha_t=1/t$（等比例）和 $\alpha_t=1/2$（固定半数）；(3) **正则化**：LwF（KL散度惩罚输出差异）和EWC（Fisher信息矩阵加权约束参数更新）。
    - 设计动机：全面覆盖持续学习领域的主要方法范式，在统一的大规模设置下公平比较

### 损失函数 / 训练策略

标准的自回归语言建模loss（下一个token预测的交叉熵）。实验在1B和3B参数规模上进行，训练token量为220B和440B。Oracle基线每两年从头训练一次（共使用1.16T tokens），用作持续训练方法的性能上界。使用regret矩阵（每个checkpoint在每个月评估数据上的相对perplexity）来可视化ID/Backward trade-off。

## 实验关键数据

### 主实验

| 方法 (3B, 440B tokens) | TiC-CC Bwd↓ | TiC-CC ID↓ | TiC-CC-Wiki Bwd↓ | TiC-CC-News Bwd↓ |
|------------------------|-------------|------------|-------------------|-------------------|
| Cyclic Cosine | 0.082 | -0.011 | 0.029 | 0.071 |
| Cyclic Cosine + AR | 0.058 | -0.002 | 0.014 | 0.044 |
| Replay (α=1/2) | 0.007 | 0.007 | 0.010 | 0.005 |
| Replay (α=1/2) + AR | **-0.002** | 0.016 | **-0.001** | **-0.009** |
| Oracle Series (1.16T) | -0.003 | 0.035 | 0.004 | -0.008 |

*数值为相对Oracle-2024-07的log-perplexity，越低越好，负值表示超越Oracle*

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| α=1/t vs α=1/2 | 1/t 的Bwd最低但ID最差 | 等比例回放在114个月后当前月份数据比例过低 |
| 有无AR调度 | AR统一降低Bwd，略升ID | 跨轮次衰减学习率有助于减少遗忘 |
| LwF / EWC | 几乎无改善 | 正则化方法在大规模web数据持续训练中效果有限 |
| 1B vs 3B | 趋势一致 | 较大模型受益略显著 |
| 220B vs 440B | 440B整体更优 | 更多token预算改善了所有配置 |

### 关键发现

- **Replay (α=1/2) + AR可以在2.6倍更少的计算量下接近Oracle系列的性能**，是所有持续训练方法中最均衡的选择
- 遗忘在通用CC数据上确实是严峻挑战：不用回放的方法在旧月份数据上performance显著下降
- 回放的最优比例因领域而异：通用web数据需要大量回放来防止遗忘，但在StackOverflow和PyTorch等快速演化的领域，回放旧数据反而会损害性能
- 正则化方法（LwF/EWC）在大规模web数据持续训练中几乎无效，这与它们在小规模持续学习中的有效性形成对比
- 更强的检索器排在前面的不相关段落更具干扰性

## 亮点与洞察

- 基准规模空前：2.9T tokens、114个时间步，比先前持续学习基准大100倍以上
- 发现域依赖性：遗忘的影响和回放的需求高度依赖于领域的演化速度，这对实际部署有重要指导意义
- 实验全面系统：150+实验覆盖了多种方法、规模和评估维度
- 数据因果性设计：严格保证每个月的处理只使用当月及之前的数据，避免信息泄露

## 局限与展望

- 未探索指令微调后的持续学习效果
- 仅使用英语数据，未考虑多语言场景
- 仅在1B和3B规模实验，更大模型（7B+）的持续训练策略可能不同
- 未考虑数据过期和删除的成本（隐私法规等实际约束）
- 跨月份去重的影响仅初步探索，可能影响遗忘评估的准确性

## 相关工作与启发

- TiC-CLIP (Garg et al., 2024) 是视觉领域时间持续学习的类似工作，本文将其理念扩展到LLM预训练
- DataComp-LM (Li et al., 2024a) 提供了数据处理管线的基础
- 与Gupta et al. (2023)、Ibrahim et al. (2024)等web数据持续训练工作不同，本文聚焦时间分布漂移和长期持续更新
- 回放策略的effective性确认对实际LLM更新部署（如月度/季度更新）有直接参考价值

## 评分

- **新颖性**: 8/10 — 基准设计填补了重要空白，但方法本身都是已有的
- **技术深度**: 7/10 — 系统性评估为主，方法创新有限
- **实验充分性**: 9/10 — 150+实验，多规模多领域多方法全面比较
- **写作质量**: 8/10 — 结构清晰，图表设计信息密度高
- **应用价值**: 9/10 — 对LLM持续更新的工程实践有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] WebWalker: Benchmarking LLMs in Web Traversal](webwalker_benchmarking_llms_in_web_traversal.md)
- [\[ACL 2025\] HellaSwag-Pro: A Large-Scale Bilingual Benchmark for Evaluating the Robustness of LLMs in Commonsense Reasoning](hellaswag-pro_a_large-scale_bilingual_benchmark_for_evaluating_the_robustness_of.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ICML 2025\] DataDecide: How to Predict Best Pretraining Data with Small Experiments](../../ICML2025/llm_evaluation/datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)
- [\[ACL 2025\] Benchmarking Uncertainty Quantification Methods for Large Language Models with LM-Polygraph](benchmarking_uncertainty_quantification_methods_for_large_language_models_with_l.md)

</div>

<!-- RELATED:END -->
