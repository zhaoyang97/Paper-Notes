---
title: >-
  [论文解读] Self Iterative Label Refinement via Robust Unlabeled Learning
description: >-
  [NeurIPS 2025][计算生物][自我精炼] 提出一种迭代式管道方法，利用鲁棒的无标签-无标签（UU）学习框架来精炼LLM生成的伪标签，仅需极少人工标注即可在分类和生成式安全对齐任务中超越GPT-4o和DeepSeek-R1的自我精炼方法。 大语言模型（LLM）在各种下游任务上表现出色，但进一步提升其能力通常需要大量…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "自我精炼"
  - "伪标签"
  - "UU学习"
  - "LLM对齐"
  - "弱监督分类"
---

# Self Iterative Label Refinement via Robust Unlabeled Learning

**会议**: NeurIPS 2025  
**arXiv**: [2502.12565](https://arxiv.org/abs/2502.12565)  
**代码**: [GitHub](https://github.com/HikaruAsano/self-iterative-label-refinement)  
**领域**: 对齐/RLHF  
**关键词**: 自我精炼, 伪标签, UU学习, LLM对齐, 弱监督分类

## 一句话总结

提出一种迭代式管道方法，利用鲁棒的无标签-无标签（UU）学习框架来精炼LLM生成的伪标签，仅需极少人工标注即可在分类和生成式安全对齐任务中超越GPT-4o和DeepSeek-R1的自我精炼方法。

## 研究背景与动机

大语言模型（LLM）在各种下游任务上表现出色，但进一步提升其能力通常需要大量高质量的人工标注反馈（如RLHF）。虽然RLAIF试图用模型自身生成的信号替代人工标注来降低成本，但模型在自我评估时存在固有偏差，尤其在模型内部知识不足的领域中，自我精炼方法往往无法带来提升甚至导致性能退化。

作者观察到两个关键点：（1）在现代社会中，收集大量无标注数据非常容易，但获取高质量标注数据代价高昂；（2）现有的LLM自我精炼方法本质上依赖模型自身的内部知识来生成反馈和修正，当知识不足时就会失败。因此，他们提出将精炼过程从LLM的内部知识中解耦出来，转而利用数据驱动的特征，通过UU学习框架来迭代去噪和改善伪标签。

## 方法详解

### 整体框架

整个管道分为三步循环迭代：（1）LLM初始标注：用LLM对无标注语料生成初始伪标签；（2）鲁棒UU学习：将语料分为伪正类和伪负类子集，用鲁棒UU学习训练分类器；（3）重新标注：训练好的分类器重新标注整个数据集，更新伪正类和伪负类集合，进入下一轮迭代。

### 关键设计

1. **无标签-无标签（UU）学习框架**: 核心思想是利用两个具有不同正类先验比例的无标注数据集来训练分类器。给定两个无标注语料 $\widetilde{\mathcal{C}}_p$ 和 $\widetilde{\mathcal{C}}_n$，其正类比例分别为 $\theta_p$ 和 $\theta_n$（$\theta_p > \theta_n$），UU学习的风险函数为：

    $R_{\text{uu}}(g) = aR_{\tilde{p}}^+(g) - bR_{\tilde{p}}^-(g) - cR_{\tilde{n}}^+(g) + dR_{\tilde{n}}^-(g)$

   其中系数 $a, b, c, d$ 由 $\pi_+, \theta_p, \theta_n$ 计算得出。当 $\theta_p=1, \theta_n=0$ 时，退化为标准有监督学习。这意味着通过LLM伪标签划分的两个子集天然满足UU学习的条件——伪正类集中正类比例更高。

2. **鲁棒UU学习（Robust UU Learning）**: 原始UU风险函数包含负项（如 $-bR_{\tilde{p}}^-(g)$），容易导致过拟合。鲁棒版本引入广义Leaky ReLU函数 $f$ 来调节负风险：

    $R_{\text{ruu}}(g) = f(aR_{\tilde{p}}^+(g) - cR_{\tilde{n}}^+(g)) + f(dR_{\tilde{n}}^-(g) - bR_{\tilde{p}}^-(g))$

   其中 $f(x) = x$ 当 $x > 0$，$f(x) = \lambda x$ 当 $x < 0$（$\lambda < 0$）。这保留正风险值，同时将负风险转换为正值，有效减少过拟合。

3. **迭代重标注与收敛**: 每轮训练完分类器 $g^{(t)}$ 后，对整个数据集重新标注：$\tilde{y}_i^{(t)} = \text{sign}(g^{(t)}(x_i))$，形成更新后的伪正类和伪负类集合。理想情况下，随着迭代进行，伪正类集中的真正正类比例趋近1，伪负类集趋近0，最终等价于标准有监督学习。

### 损失函数 / 训练策略

- 分类器在Transformer最终隐状态上接一个仿射层输出一维分数
- 使用QLoRA进行4-bit量化微调，AdamW优化器（学习率 $1.0 \times 10^{-4}$，batch size 16，3 epochs）
- $\lambda$ 固定为 -0.001
- 类先验估计：oracle直接使用真实值；few-labeled仅用50或100个标注样本估计 $\hat{\theta}_p$ 和 $\hat{\theta}_n$

## 实验关键数据

### 主实验（简单任务 RQ1）

| 数据集 | 模型 | 初始LLM准确率 | Ours (50-labeled) 最终 | 提升 |
|--------|------|--------------|----------------------|------|
| Fake News | Meta-Llama-3-8B | ~0.75 | ~0.90 | +15% |
| Saroco | Llama-3.2-1B | 0.576 | ~0.80 | +22% |
| Safety | 多模型 | ~0.65 | ~0.85 | +20% |

在简单任务上，PIE和CCP在Saroco和Safety上完全失败，而本方法在所有任务上均稳定提升。仅50个标注样本即可快速收敛至Oracle上界。

### 困难任务（RQ2）

| 数据集 | 方法 | 最终准确率 | 与初始相比 |
|--------|------|-----------|-----------|
| Corona Sentiment | GPT-4o自我精炼 | 略有提升 | 微弱增益 |
| Corona Sentiment | DeepSeek-R1 | 停滞 | 无增益 |
| Corona Sentiment | Ours (Oracle) | 稳步提升 | 显著提升 |
| Green Patent | GPT-4o自我精炼 | 性能退化 | 负增益 |
| Green Patent | Ours (Oracle) | 稳步提升 | 显著提升 |
| Protein Structure | GPT-4o自我精炼 | 性能退化 | 负增益 |
| Protein Structure | Ours (Oracle) | 稳步提升 | 显著提升 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| PN (标准监督) | 低准确率 | 伪标签噪声严重时完全失败 |
| UU (非鲁棒) | 有提升但低于Ours | 负风险项导致过拟合 |
| Ours (Oracle) | 最高 | 理论上界 |
| Ours (50-labeled) | 接近Oracle | 仅需50个标注样本 |

### 关键发现

- 即使GPT-4o和DeepSeek-R1的自我精炼完全失败的困难任务上，本方法仍能稳步提升
- 50个标注样本的估计版本与Oracle几乎等效，展现了UU学习对类先验估计误差的鲁棒性
- 在安全对齐（RQ3）中，用精炼后的分类器作为奖励模型进行RLAIF，生成的回答安全性显著提升，而Vanilla RLAIF甚至不如SFT基线

## 亮点与洞察

- 将UU学习这一经典弱监督学习技术引入LLM自我精炼，思路简洁而有效
- 仅需极少人工标注（50个样本），无需外部工具或知识库
- 方法与LLM的内部知识解耦，依靠数据驱动特征提升，适用于LLM知识不足的专业领域
- 在分类和生成式对齐任务上均有效，展现了良好的通用性

## 局限与展望

- 当初始伪标签噪声极高、两个子集的类先验几乎相同时，效果有限
- 性能受初始标签质量之外的其他因素影响（如任务本身的可分类性），未来可探索建模实例级别的可分类性
- 当前仅验证了二分类场景，多分类扩展有待探索
- 可考虑融入辅助信息（如推理链、检索上下文）进一步提升分类精度

## 相关工作与启发

- UU学习源自弱监督学习领域，本文创新性地将其与LLM伪标签精炼结合
- 与PIE（渐进式置信度标签选择）和CCP（对比式半监督学习）相比，本方法在LLM生成的噪声标签上更加鲁棒
- 可启发将更多经典弱监督/半监督技术应用于LLM自我改进场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ UU学习+LLM伪标签精炼的结合新颖，但方法本身改动较小
- **实验充分度**: ⭐⭐⭐⭐⭐ 6个数据集覆盖简单/困难任务，含分类和生成式对齐，对比充分
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数学推导完整，研究问题驱动的实验设计
- **价值**: ⭐⭐⭐⭐ 为LLM在低资源和专业领域的自我改进提供了实用方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood](prospero_active_learning_for_robust_protein_design_beyond_wild-type_neighborhood.md)
- [\[NeurIPS 2025\] Evaluating Multiple Models Using Labeled and Unlabeled Data](evaluating_multiple_models_using_labeled_and_unlabeled_data.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[NeurIPS 2025\] Post Hoc Regression Refinement via Pairwise Rankings](post_hoc_regression_refinement_via_pairwise_rankings.md)
- [\[ICML 2025\] scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data](../../ICML2025/computational_biology/scssl-bench_benchmarking_self-supervised_learning_for_single-cell_data.md)

</div>

<!-- RELATED:END -->
