---
title: >-
  [论文解读] A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior
description: >-
  [ACL 2025][眼动追踪] 本文提出基于标记时空点过程（marked spatio-temporal point process）的阅读行为统一概率模型，同时建模注视何时发生、落在哪里、持续多久，避免传统聚合测量的信息损失，发现 surprisal 对细粒度眼动的预测贡献极其有限。 领域现状：计算心理语言学中…
tags:
  - "ACL 2025"
  - "眼动追踪"
  - "Hawkes过程"
  - "时空点过程"
  - "surprisal theory"
  - "阅读行为建模"
---

# A Spatio-Temporal Point Process for Fine-Grained Modeling of Reading Behavior

**会议**: ACL 2025  
**arXiv**: [2506.19999](https://arxiv.org/abs/2506.19999)  
**代码**: [GitHub](https://github.com/rycolab/spatio-temporal-reading)  
**领域**: 计算心理语言学 / NLP理解  
**关键词**: 眼动追踪, Hawkes过程, 时空点过程, surprisal theory, 阅读行为建模

## 一句话总结

本文提出基于标记时空点过程（marked spatio-temporal point process）的阅读行为统一概率模型，同时建模注视何时发生、落在哪里、持续多久，避免传统聚合测量的信息损失，发现 surprisal 对细粒度眼动的预测贡献极其有限。

## 研究背景与动机

**领域现状**：计算心理语言学中，眼动追踪实验是研究人类语言理解的核心范式。标准做法是将原始 scanpath 数据聚合为词级阅读时间指标（如 first fixation duration、gaze duration、total fixation duration），然后用线性混合效应模型进行分析。

**现有痛点**：聚合过程会丢失大量信息。从时间维度看，将多次注视合并为单一测量会混淆不同认知过程（如首次注视与回归注视对应不同机制）；从空间维度看，聚合依赖预定义的兴趣区域（通常按词划分），丢弃了注视在词内具体位置的信息，也阻碍了对更小语言单元（音节、语素）的研究。

**核心矛盾**：聚合策略的选择直接影响理论验证结论。例如 surprisal theory 预测上下文可预测性对 gaze duration 影响更强，但实验反而发现对 total fixation time 效果更大。这种反直觉的结果可能是聚合方式造成的伪影，而非真实的认知效应，但基于聚合数据无法确定因果。

**本文目标** (1) 如何在不损失信息的前提下，统一建模注视的时间、空间和持续时间三个维度？(2) 对原始 scanpath 进行建模时，surprisal 等预测因子的效应大小是否与聚合数据一致？

**切入角度**：作者将阅读过程形式化为时空标记点过程——注视和扫视的交替序列天然适合点过程框架。利用 Hawkes 过程的自激性质可以捕捉先前注视对后续注视在时空上的影响。

**核心 idea**：用 Hawkes 过程建模扫视的时空分布、对数正态分布建模注视持续时间，构成统一的标记时空点过程，直接对原始 scanpath 建模。

## 方法详解

### 整体框架

输入是一个完整的 scanpath $\mathcal{T} = \{(t_n, \mathbf{s}_n, d_n)\}_{n=1}^N$，其中 $t_n$ 是注视开始时间，$\mathbf{s}_n$ 是屏幕上的空间坐标，$d_n$ 是注视持续时间。模型包含两个组件：(1) 时空 Hawkes 过程建模注视的发生时间和落点位置（when & where），(2) 对数正态分布建模注视持续时间（how long）。生成 scanpath 时迭代采样：先采样下一次注视的时间和位置，再采样其持续时间，更新历史后重复直至达到时间上限。

### 关键设计

1. **时空 Hawkes 过程建模扫视规划**:

    - 功能：建模下一次注视何时发生、落在屏幕何处
    - 核心思路：强度函数 $\lambda(t_n, \mathbf{s}_n; \mathcal{H}_{n-1}) = \nu + \sum_{m=1}^{n-1} \phi_m(t_n - t_m - \delta(n,m)) \psi_m(\mathbf{s}_n)$，其中 $\nu$ 是基础强度，$\phi_m$ 是指数衰减时间核（控制历史注视的时间影响力），$\psi_m$ 是以历史注视位置为中心的二维高斯空间密度。时间核的激发强度和衰减率通过注视级预测因子的线性组合参数化，允许不同条件下的自激行为不同。空间密度支持三种中心变换：恒等变换（baseline）、仿射变换（学习左到右阅读方向的常数位移）、带预测因子的完整变换
    - 设计动机：Hawkes 过程的自激结构天然契合阅读行为——前一次注视"激发"后续注视在其附近发生，多个历史注视的贡献叠加形成多模态分布，恰好对应了人类阅读中前进扫视、回归和重新注视等多种行为

2. **对数正态卷积模型建模注视持续时间**:

    - 功能：预测每次注视持续多长时间
    - 核心思路：注视持续时间服从对数正态分布 $g(d_n | \mathcal{H}_{n-1}, t_n)$，其对数均值 $\xi_n(t_n)$ 通过卷积方式整合历史预测因子的溢出效应。具体地，$\xi_n^c(t_n) = \mathbf{x}_n^\top \mathbf{w} + \sum_{k \in K} w'_k \sum_{m=1}^{n-1} x_{mk} \gamma(t_n - t_m | \alpha_k, \beta_k, \theta_k)$，其中 $\gamma$ 是 shifted gamma 分布核函数，捕捉先前认知处理对后续注视时长的延迟影响
    - 设计动机：心理语言学的溢出效应（spillover effect）表明处理某个词的认知负担会延续到后续几个词的阅读时间。传统做法用 Markov 假设只看前 $l$ 个注视，而卷积方法理论上可以捕捉无限远的历史影响

3. **Reader-Specific Effects (RSE) 模型**:

    - 功能：为不同读者学习个体特异性的时空参数
    - 核心思路：将预测因子向量扩展为 $\mathbf{x}_m = \mathbf{1} \oplus \mathbf{u}_m$，其中 $\mathbf{u}_m$ 是读者的 one-hot 编码向量，使时间核的激发强度/衰减率和空间密度中心都包含全局效应+读者特异效应
    - 设计动机：不同读者有不同的阅读风格（扫视幅度、回归频率等），传统模型将所有读者视为同一"平均读者"会丢失这种变异性

### 损失函数 / 训练策略

整体优化目标是最大化 scanpath 的对数似然，分为扫视模型和持续时间模型两部分分别优化。使用 SGD + Nesterov 动量训练，80/10/10 数据划分，早停策略（patience=5）。对超参数进行网格搜索（batch size、学习率、权重衰减）。采用 warm-starting 策略：先训简单模型，用其参数初始化复杂模型。

## 实验关键数据

### 主实验：扫视规划模型对比

| 模型 | 每注视对数似然增益 (nats) | 相对提升 |
|------|------------------------|---------|
| Poisson baseline | 0 (基线) | — |
| Last-fixation baseline | ~0.6 | — |
| Standard Hawkes | ~0.8 | — |
| CSS (常数空间偏移) | ~1.5 | — |
| RSE (读者特异效应) | **2.44** | ~1047% 似然提升 |

### 预测因子对扫视的边际贡献

| 预测因子 | 在RSE基础上的相对增益 | 说明 |
|---------|-------------------|------|
| 词长 (length) | ~4% | 贡献最大 |
| 词级 surprisal | <2% | 效果微弱 |
| 字符级 surprisal | <2% | 效果微弱 |
| Unigram surprisal | <2% | 效果微弱 |

### 关键发现

- **RSE 模型学到了约 10.61 个字符的全局右移偏移**，完美对应英语从左到右的阅读方向。读者特异效应和方向性空间偏移是最重要的建模因素。
- **surprisal 等语言预测因子对扫视规划的贡献极为有限**（<2% 相对增益），远小于时空依赖和读者效应的贡献。这挑战了 surprisal theory 在细粒度眼动层面的解释力。
- **卷积模型与 Markov 模型表现几乎相同**，表明过去注视对当前持续时间的影响可能是有界的，不需要无限历史。
- **对原始 scanpath 建模时，surprisal 效应比聚合数据上小一个数量级**，暗示文献中基于聚合数据报告的效应大小可能被人为放大。

## 亮点与洞察

- **从简单到复杂的递增式建模设计非常清晰**：从 Poisson → Last-fixation → Hawkes → CSS → RSE → +predictors，每一步的增益都可量化，让读者精确理解每个组件的贡献。这种递增消融的实验设计值得学习。
- **质疑 surprisal theory 在细粒度层面的适用性**：在聚合数据上显著的效应在原始数据上几乎消失，这提示了一个重要的方法论警示——理论验证的结论可能高度依赖数据预处理方式。
- **时空 Hawkes 过程用于认知建模**是一个精巧的类比迁移。Hawkes 过程原本用于地震余震、犯罪传播等领域，将其自激特性映射到阅读中"前一注视激发后续注视"非常自然。

## 局限性

- **仅使用 MECO 英语数据集**（46 个读者、12 篇短文），未验证在其他语言（特别是从右到左的语言）上的泛化性。
- **空间核假设各向同性方差**，可能无法充分捕捉眼动的方向性偏差（水平方向的扫视本应比垂直方向更频繁）。
- **未建模行间跳转**：作者尝试加入右边距距离特征但训练不稳定。对于多行文本阅读，行末到行首的跳转是重要行为。
- **字符级和词级 surprisal 使用不同语言模型**（GPT-2 vs mGPT），性能差异不能完全归因于粒度差异。

## 相关工作与启发

- **vs E-Z Reader / SWIFT**：这些是经典的认知眼动控制模型，但不是数据驱动的，且不捕捉语言处理对注视的影响。本文的点过程框架更灵活，但不提供认知机制解释。
- **vs ScanDL (Bolliger et al., 2023)**：用扩散模型生成合成 scanpath，但面向下游 NLP 任务增强而非理论验证。两种方法互补——ScanDL 可以用生成质量评估点过程模型。
- **vs 标准 surprisal 分析 (Smith & Levy, Wilcox et al.)**：本文直接挑战了这些工作的结论可靠性，指出聚合策略可能系统性地放大效应大小。

## 评分

- 新颖性: ⭐⭐⭐⭐ 将时空Hawkes过程引入阅读行为建模，概念自然且新颖，但点过程本身不新
- 实验充分度: ⭐⭐⭐⭐⭐ 层层递进的消融设计极其清晰，从基线到完整模型每一步都有量化贡献
- 写作质量: ⭐⭐⭐⭐⭐ 数学推导严谨，记号统一，图示直观
- 价值: ⭐⭐⭐⭐ 对心理语言学方法论有重要警示作用，但对 NLP 社区的直接实用价值有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] FRACTAL: Fine-Grained Scoring from Aggregate Text Labels](fractal_fine-grained_scoring_from_aggregate_text_labels.md)
- [\[ACL 2025\] Guidelines for Fine-grained Sentence-level Arabic Readability Annotation](guidelines_for_fine-grained_sentence-level_arabic_readability_annotation.md)
- [\[ACL 2025\] Barec: A Large and Balanced Corpus for Fine-grained Arabic Readability Assessment](a_large_and_balanced_corpus_for_fine-grained_arabic_readability_assessment.md)
- [\[ACL 2025\] Decoding Reading Goals from Eye Movements](decoding_reading_goals_from_eye_movements.md)

</div>

<!-- RELATED:END -->
