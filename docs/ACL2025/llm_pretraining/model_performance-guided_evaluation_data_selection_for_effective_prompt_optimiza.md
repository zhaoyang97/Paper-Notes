---
title: >-
  [论文解读] Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization
description: >-
  [ACL 2025][预训练][提示优化] 提出 IPOMP——一种两阶段评估数据选择方法，第一阶段通过语义聚类和边界分析选取多样化样本，第二阶段利用提示优化过程中的实时模型性能迭代替换冗余样本，在 BIG-bench 和 LIAR 上将提示优化效果提升 1.6%-3.1%，稳定性提升 50%+，额外开销不到 1%。
tags:
  - "ACL 2025"
  - "预训练"
  - "提示优化"
  - "核心集选择"
  - "评估数据选取"
  - "实时模型性能"
  - "语义聚类"
---

# Model Performance-Guided Evaluation Data Selection for Effective Prompt Optimization

**会议**: ACL 2025  
**arXiv**: [2505.10736](https://arxiv.org/abs/2505.10736)  
**代码**: 无  
**领域**: LLM预训练  
**关键词**: 提示优化, 核心集选择, 评估数据选取, 实时模型性能, 语义聚类

## 一句话总结

提出 IPOMP——一种两阶段评估数据选择方法，第一阶段通过语义聚类和边界分析选取多样化样本，第二阶段利用提示优化过程中的实时模型性能迭代替换冗余样本，在 BIG-bench 和 LIAR 上将提示优化效果提升 1.6%-3.1%，稳定性提升 50%+，额外开销不到 1%。

## 研究背景与动机

提示优化（Prompt Optimization）是提升 LLM 任务表现的关键步骤，但自动化提示优化面临一个被忽视的核心问题：**评估数据的选择**。

现有痛点：

**全量评估不现实**：在整个训练集上评估每个候选提示成本太高

**随机采样不可靠**：随机选取的子集很可能不具代表性，导致评估不稳定、优化的提示不够好

**现有核心集方法不适用**：
   - 语义聚类对高度相似的任务样本效果差
   - 模型性能方法需预先收集评估数据，成本高且迁移差
   - 新/私有数据集无历史性能数据可用

这是首个针对提示优化场景提出评估数据选择方法的工作。

## 方法详解

### 整体框架

IPOMP 是嵌入提示优化迭代过程的两阶段方法：

- **Stage 1（Diverse Sample Selection）**：语义聚类 + 边界样本选取
- **Stage 2（Real-time Performance-Guided Refinement）**：每次迭代动态替换冗余样本

### 关键设计

1. **Stage 1: 多样化样本选取**：

    - Sentence-BERT 编码所有训练样本，K-means 聚类（$k=5$）
    - 按比例从每个簇采样 $\alpha N$ 个样本（语义代表性）
    - 在语义空间中找最远距离样本对，取 $(1-\alpha)N$ 个（边界多样性）
    - HNSW 加速边界点检测，避免 $O(dN^2)$ 全量距离计算
    - 设计动机：聚类保证代表性，边界样本弥补极端情况的覆盖

2. **Stage 2: 实时性能引导迭代精炼**：

    - 核心观察：约 20% 样本在不同候选提示上的表现高相关（>0.9），即大量冗余
    - 每次迭代记录性能矩阵：样本 × (输出标签数 × 候选提示数)，使用 logits
    - 层级聚类将高相关样本分组，随机选 $\beta$ 比例冗余样本
    - 用语义空间中最不相似的训练样本替换
    - 设计动机：利用优化中"免费"的模型反馈，零额外推理开销识别冗余

3. **与现有性能引导方法的关键区别**：

    - Anchor-Point 需额外预热阶段（~200秒 + API 成本）
    - Prediction-based 需在其他 LLM 上训练评估器
    - IPOMP 直接利用优化过程的副产品，零额外推理成本

### 关键超参数

- 评估集大小 $N=20$，聚类数 $K=5$
- 聚类 vs 边界比例 $\alpha=0.5$
- 相关性阈值 $CT=0.9$，替换率 $\beta=0.5$

## 实验关键数据

### 主实验（Accuracy ± SD，跨三种优化方法的平均）

| 方法 | GPT-3.5 BIG-bench | GPT-4o-mini BIG-bench | GPT-3.5 LIAR | GPT-4o-mini LIAR |
|------|:-:|:-:|:-:|:-:|
| Random | 0.719±0.035 | 0.704±0.041 | 0.742±0.041 | 0.748±0.048 |
| Clustering | 0.725±0.029 | 0.725±0.029 | 0.752±0.036 | 0.788±0.038 |
| Boundary | 0.727±0.040 | 0.723±0.038 | 0.770±0.035 | 0.797±0.047 |
| Anchor-Point | 0.745±0.028 | 0.756±0.027 | 0.801±0.027 | 0.807±0.024 |
| Prediction-based | 0.725±0.038 | 0.705±0.044 | 0.746±0.039 | 0.750±0.043 |
| **IPOMP** | **0.757±0.012** | **0.778±0.011** | **0.820±0.012** | **0.833±0.012** |

### 消融实验

| 变体 | GPT-3.5 BB Acc | GPT-4o-mini BB Acc |
|------|:-:|:-:|
| IPOMP (完整) | 0.757±0.012 | 0.778±0.011 |
| Stage 1 only | 0.733±0.034 | 0.743±0.012 |
| Random + Stage 2 | 0.737±0.021 | 0.738±0.012 |

去掉 Stage 2：Acc 降 2.4%，SD 增 2.83 倍。用随机替换 Stage 1：Acc 降 2%。

### 时间开销（BIG-bench，GPT-3.5，秒）

| 组成 | APO | APE | EVOPROMPT | 平均 |
|------|-----|-----|-----------|------|
| Stage 1 | 0.45 | 0.37 | 0.51 | 0.45 |
| Stage 2 | 2.74 | 2.31 | 3.43 | 2.83 |
| Anchor-Point 预热 | 200.36 | 205.32 | 207.85 | 204.51 |

IPOMP 额外开销 < 1%，Anchor-Point 预热增加 ~50% 时间。

### 关键发现

1. 所有核心集方法都优于 Random，证明评估数据选择对提示优化很重要
2. IPOMP 对比次优 Anchor-Point：Acc +1.6%~3.1%，SD 降低 50%+
3. Stage 2 可作为通用插件：为 Random/Boundary/Clustering 带来 2.3%/1.1%/1.5% 提升
4. 样本量 20 是最优平衡点
5. 一轮精炼即将高相关冗余从 19% 降至 10%

## 亮点与洞察

1. **问题定义新颖**：首次在提示优化中研究评估数据选择问题
2. **实时性能利用的优雅设计**：巧妙复用优化过程已有的模型反馈
3. **通用性**：Stage 2 可即插即用增强任何数据选择方法
4. **稳定性提升**：SD 降低 50%+ 意味着生产环境中的可靠性大幅提升

## 局限与展望

1. **模型覆盖有限**：仅 GPT-3.5 和 GPT-4o-mini
2. **任务偏向分类**：生成任务上效果未知
3. **Logits 依赖**：对无法获取 logits 的 API 适用性受限
4. **提示优化技术**：仅覆盖 APE、APO、EVOPROMPT

## 相关工作与启发

- 提示优化三大类：非定向（APE/EvoPrompt）、定向（APO）
- 核心集选择从几何方法到性能方法的演进
- 核心启发：提示优化中"评什么"和"怎么优化"同等重要

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次定义并解决提示优化的评估数据选择问题
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集、多模型、多优化技术、详细消融
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，方法描述系统
- **价值**: ⭐⭐⭐⭐ — Stage 2 的通用插件实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] DavIR: Data Selection via Implicit Reward for Large Language Models](davir_data_selection_via_implicit_reward_for_large_language_models.md)
- [\[ICLR 2026\] Predicting Training Re-evaluation Curves Enables Effective Data Curriculums](../../ICLR2026/llm_pretraining/predicting_training_re-evaluation_curves_enables_effective_data_curriculums_for_.md)
- [\[ICML 2025\] LLM Data Selection and Utilization via Dynamic Bi-level Optimization](../../ICML2025/llm_pretraining/llm_data_selection_and_utilization_via_dynamic_bi-level_optimization.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] AutoDS: Autonomous Data Selection with Zero-shot Generative Classifiers for Mathematical Texts](autonomous_data_selection_with_zero-shot_generative_classifiers_for_mathematical.md)

</div>

<!-- RELATED:END -->
