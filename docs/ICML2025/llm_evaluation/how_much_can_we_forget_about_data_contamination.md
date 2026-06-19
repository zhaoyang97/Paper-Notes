---
title: >-
  [论文解读] How Much Can We Forget about Data Contamination?
description: >-
  [ICML 2025][LLM评测][数据污染] 通过受控实验系统量化数据污染对 LLM benchmark 评估的影响，发现在超过 Chinchilla 最优五倍以上的训练数据量下，即使 144 次重复的污染数据也能被完全遗忘；进一步证明权重衰减是遗忘的关键机制，并据此推断 Llama 3 405B 等大型模型已遗忘训练早期的数据。
tags:
  - "ICML 2025"
  - "LLM评测"
  - "数据污染"
  - "遗忘"
  - "benchmark过拟合"
  - "Chinchilla缩放"
  - "权重衰减"
  - "AdamW"
  - "LLM评估"
---

# How Much Can We Forget about Data Contamination?

**会议**: ICML 2025  
**arXiv**: [2410.03249](https://arxiv.org/abs/2410.03249)  
**代码**: [GitHub](https://github.com/tml-tuebingen/forgetting-contamination/)  
**领域**: LLM评测  
**关键词**: 数据污染, 遗忘, benchmark过拟合, Chinchilla缩放, 权重衰减, AdamW, LLM评估

## 一句话总结

通过受控实验系统量化数据污染对 LLM benchmark 评估的影响，发现在超过 Chinchilla 最优五倍以上的训练数据量下，即使 144 次重复的污染数据也能被完全遗忘；进一步证明权重衰减是遗忘的关键机制，并据此推断 Llama 3 405B 等大型模型已遗忘训练早期的数据。

## 研究背景与动机

机器学习的核心原则之一是模型不应在测试集上训练，但大规模语言模型的训练数据往往从互联网抓取，不可避免地包含 benchmark 评估数据——即数据污染。GPT-3、Llama 3 等模型的报告中均发现了训练数据与 benchmark 的重叠。

当前对数据污染的理解存在关键空白：

**小规模污染是否必然导致评估失效**仍不清楚——现代模型训练超过百万步梯度更新，训练过程中某一步使用了污染数据不一定影响最终评估。

**记忆化研究**表明样本需要被多次重复才能被记住，**知识获取研究**也发现事实需要多次释义才能被学习。
3. 训练数据规模已远超 Chinchilla 最优（如 Llama 3 70B 使用了超过 10 倍 Chinchilla 的数据），但污染文献几乎未考虑这种数据富余对污染影响的稀释效应。

本文的核心问题：**在什么条件下，数据污染确实会使 benchmark 评估失效？**

## 方法详解

### 整体框架

采用受控实验设计：从头训练语言模型（最大 1.6B 参数），在训练数据中显式插入 benchmark 问题，沿三个维度进行缩放实验：

- **模型参数量**：124M → 350M → 774M → 1.6B
- **训练 token 数**：1× → 15× Chinchilla
- **污染重复次数**：4 → 12 → 32 → 144 次

训练数据为 FineWeb-Edu 100BT，混合使用七个 benchmark（ARC-Easy, SocialIQA, WinoGrande, PiQA, BoolQ, MMLU, HellaSwag）。

### 污染插入方式

- 将 benchmark 问题分为多个子集（2000-10000 题）
- 保留 10000 题的 holdout 集从不加入训练数据
- 其他子集以不同重复次数（4/12/32/144）随机插入训练数据
- 采用精确污染（exact contamination）——训练数据与评估数据完全相同
- 对 benchmark 中的近重复问题进行过滤（基于 Levenshtein 距离）

### 遗忘实验设计

在 15× Chinchilla 训练中，将污染集中在第 1-2 个 Chinchilla 之间，然后观察后续训练中交叉熵损失差异的衰减。

关键变体：
- 持续在新数据上训练 vs 在固定 100M token 上重复训练
- 污染数据在训练早期/中期/晚期/均匀分布的影响

### 权重衰减与遗忘的理论分析

AdamW 优化器的参数更新可分解为权重衰减更新和梯度更新。通过迭代展开，最终模型权重可表示为：

$$\theta_T = w_0^T \theta_0 - \sum_{t=1}^T w_t^T \gamma_t \hat{g}_t$$

其中累积权重衰减为：

$$w_{t_1}^{t_2} = \prod_{i=t_1+1}^{t_2} (1 - \gamma_i \lambda)$$

**命题 1**：当 $T \geq \frac{\log(1/\epsilon)}{\lambda \gamma_{\text{avg}}}$ 时，$w_{t_1}^{t_2} \leq \epsilon$。

含义：经过足够多的梯度步数后，早期梯度更新对最终模型权重的贡献趋近于零。

## 实验关键数据

### 三维缩放实验

Chinchilla 最优模型的绝对准确率（holdout vs 污染）：

| 模型 | Holdout | 4× | 12× | 32× | 144× |
|------|---------|-----|------|------|------|
| 124M | 42.22 | 48.14 | 56.92 | 80.70 | 96.45 |
| 350M | 44.72 | 55.69 | 69.90 | 89.20 | 95.50 |
| 774M | 49.16 | 64.76 | 81.30 | 92.95 | 96.05 |
| 1.6B | 52.06 | 67.61 | 82.32 | 91.85 | 95.40 |

关键发现：
- 过拟合随**参数量增加**而增大（124M→1.6B，4×污染的准确率差从 6→15 个百分点）
- 过拟合随**训练 token 数增加**而减小——15× Chinchilla 下，12 次重复的污染影响完全消失
- 过拟合随**重复次数增加**而增大

### 遗忘实验

- 集中在第 1-2 Chinchilla 插入 144 次污染后，继续训练 5 个 Chinchilla 的新数据，污染效应**完全消失**
- 关键条件：必须在**新数据流**上继续训练；如果在固定数据集上多轮训练，遗忘会稳定在非零水平
- 均匀分布在整个训练过程中的污染反而比集中在末尾的更难遗忘（间隔重复效应）

### OLMo-7B 验证

- 从 OLMo-7B 中间 checkpoint 插入 4 次全 benchmark 污染，平均准确率提升 17 个百分点
- 继续预训练 13% 剩余训练时间后，WinoGrande 和 ARC-Easy 的污染效应已不显著，HellaSwag 和 PiQA 约剩 2 个百分点
- 1B 和 7B 模型的遗忘曲线在按参数比（5.9×）缩放后高度吻合——遗忘具有缩放行为

### 权重衰减实验

四种不同权重衰减参数（{50, 5, 1, 0.1}）下：
- 权重衰减越大，遗忘越快（x 轴范围从 120 步到 62500 步）
- **经验遗忘始终快于累积权重衰减**——权重衰减是遗忘的上界

### 大规模训练推断

通过仅分析学习率调度和权重衰减参数计算累积权重衰减：
- **OLMo-7B**：前 40% 训练数据的梯度贡献已衰减至零
- **Llama 3 405B**：前 10% 训练数据可能已被遗忘

## 亮点与洞察

1. **实践意义重大**：对于超过 5× Chinchilla 训练的现代 LLM，小规模污染可能对评估影响微乎其微——这为 benchmark 评估的可靠性提供了新视角。

2. **间隔重复效应**：均匀分布在训练过程中的污染比集中在末尾的更难遗忘，这与人类学习中的"间隔重复效应"异曲同工。

3. **理论与实验桥接**：通过 AdamW 的累积权重衰减提供了一个无需实际重训练即可估计遗忘程度的理论工具。

4. **新数据是遗忘的关键**：多 epoch 训练（重复使用相同数据）不会导致完全遗忘，只有持续注入新数据才能实现。

## 局限性

- 实验针对的是 benchmark 问题，结论不一定适用于隐私场景（随机字符串或可识别信息的遗忘可能不同）
- 小模型的实验结论通过 OLMo-7B 验证，但更大规模模型仅通过累积权重衰减推断
- 未考虑指令微调阶段的污染影响
- 仅评估了精确污染，未研究释义污染或部分污染的情况

## 相关工作

- **数据污染检测**：Brown 2020 (n-gram 匹配)、Oren 2024 (推断式检测)、Jiang 2024 (受控污染实验)
- **遗忘研究**：Tirumala 2022（指数慢速遗忘）、Jagielski 2023（经验遗忘）、Toneva 2019（不可遗忘样本）
- **数据归因**：Koh & Liang 2017 (影响函数)、Grosse 2023

## 评分

⭐⭐⭐⭐ — 研究问题重要且实验设计严谨，三维缩放分析清晰全面，理论工具（累积权重衰减）实用。对 LLM 评估社区有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] DataDecide: How to Predict Best Pretraining Data with Small Experiments](datadecide_how_to_predict_best_pretraining_data_with_small_experiments.md)
- [\[ACL 2025\] AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge](../../ACL2025/llm_evaluation/antileakbench_preventing_data_contamination_by_automatically_constructing_benchm.md)
- [\[ICML 2025\] Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation](leveraging_online_olympiad-level_math_problems_for_llms_training_and_contaminati.md)
- [\[ACL 2025\] MMLU-CF: A Contamination-free Multi-task Language Understanding Benchmark](../../ACL2025/llm_evaluation/mmlu-cf_a_contamination-free_multi-task_language_understanding_benchmark.md)
- [\[ICML 2025\] The Best of Both Worlds: Bridging Quality and Diversity in Data Selection with Bipartite Graph](the_best_of_both_worlds_bridging_quality_and_diversity_in_data_selection_with_bi.md)

</div>

<!-- RELATED:END -->
