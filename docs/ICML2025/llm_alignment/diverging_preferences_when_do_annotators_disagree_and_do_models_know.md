---
title: >-
  [论文解读] Diverging Preferences: When do Annotators Disagree and do Models Know?
description: >-
  [ICML 2025][LLM对齐][偏好分歧] 本文系统分析了 RLHF 偏好数据集中标注者分歧的原因（建立了包含 10 个类别的分类法），发现超过 75% 的分歧源于个人偏好而非标注噪声，提出了分布式奖励模型（Mean-Var Reward Model）来有效区分分歧偏好与高一致偏好，并揭示了 LLM-as-Judge 评估方法在分歧情况下的系统性偏见。
tags:
  - "ICML 2025"
  - "LLM对齐"
  - "偏好分歧"
  - "奖励建模"
  - "多元化对齐"
  - "LLM-as-Judge"
  - "分布式奖励模型"
---

# Diverging Preferences: When do Annotators Disagree and do Models Know?

**会议**: ICML 2025  
**arXiv**: [2410.14632](https://arxiv.org/abs/2410.14632)  
**代码**: 无  
**领域**: LLM对齐/RLHF  
**关键词**: 偏好分歧, 奖励建模, 多元化对齐, LLM-as-Judge, 分布式奖励模型

## 一句话总结
本文系统分析了 RLHF 偏好数据集中标注者分歧的原因（建立了包含 10 个类别的分类法），发现超过 75% 的分歧源于个人偏好而非标注噪声，提出了分布式奖励模型（Mean-Var Reward Model）来有效区分分歧偏好与高一致偏好，并揭示了 LLM-as-Judge 评估方法在分歧情况下的系统性偏见。

## 研究背景与动机
**领域现状**：RLHF 已成为对齐 LLM 的标准方法，但在偏好数据收集中，标注者之间的分歧是普遍现象——MultiPref 中 39%、HelpSteer2 中 24% 的样本存在标注者分歧。

**现有痛点**：当前的奖励建模流程（如 Bradley-Terry）将标注者分歧视为简单噪声，通过多数投票聚合标签。这种做法忽视了分歧往往反映了不同用户视角的合理差异。

**核心矛盾**：标准奖励模型对分歧偏好和高一致偏好预测出相似的奖励差距，无法区分两者，导致 RLHF 训练的 LLM 只学会迎合一种用户视角，违背多元化对齐的目标。

**本文目标**：（1）理解标注者分歧的根本原因；（2）设计能识别分歧偏好的奖励模型；（3）发现并缓解 LLM-as-Judge 评估中的偏见。

**切入角度**：从数据分析入手，释放 MultiPref 和 HelpSteer2 的个体标注信息，建立分歧原因的分类法，然后基于此分析推动奖励建模和评估方法的改进。

**核心 idea**：将奖励建模为分布（而非单一标量），既能预测偏好方向，也能捕捉标注者之间的分歧程度。

## 方法详解

### 整体框架
方法分为三部分：（1）分歧原因分析与分类法构建；（2）分布式奖励模型设计与训练；（3）LLM-as-Judge 偏见分析与缓解。

### 关键设计

1. **分歧原因分类法**:

    - 功能：将偏好分歧划分为 4 大类、10 个子类
    - 核心思路：通过人工分析 200 个分歧样本，归纳分歧来源
    - 四大类：
        - **任务类**（Task Underspecification, 20-22%）：prompt 欠明确，多种合理解读
        - **响应风格类**（Verbosity 38-44%, Format 20-32%, Complexity 10%, Aesthetic Taste 14-22%）：个人偏好差异
        - **拒绝类**（Comply vs. Refuse 5%, Refuse vs. Refuse 20%）：安全判断分歧
        - **错误类**（Hallucinations & Errors 14-24%）：事实错误的判定差异
    - 设计动机：**超过 75% 的分歧**源于任务欠明确和响应风格等合理偏好差异，而非标注员犯错
    - 标注一致性：Cohen's κ = 0.58-0.59, Krippendorff's α = 0.62-0.68

2. **Mean-Var 分布式奖励模型 (KL)**:

    - 功能：将每个响应的奖励建模为正态分布 $r_A \sim \mathcal{N}(\mu_A, \sigma_A^2)$，同时预测均值和方差
    - 核心思路：均值反映整体偏好方向，方差捕捉标注者间的分歧程度
    - 关键公式：$r_A - r_B \sim \mathcal{N}(\mu_A - \mu_B, \sigma_A^2 + \sigma_B^2 - 2\rho\sigma_A\sigma_B)$
    - 其中 $\rho$ 建模两个响应之间的相关性（基于 tie 频率）
    - 训练损失：使用 KL 散度损失，将 $r_A - r_B$ 的值映射到标注偏好标签的概率分布上
    - 偏好映射区间：tie 对应 $(-0.5, 0.5)$，slight prefer 对应 $[0.5, 1.5)$，significant prefer 对应 $[1.5, \infty)$
    - 分歧识别：$|\mu_A - \mu_B| - \lambda(\sigma_A + \sigma_B)$，当方差大、均值差小时识别为分歧
    - 与 Bradley-Terry 的区别：后者只输出标量奖励，无法捕捉不确定性

3. **LLM-as-Judge 偏见分析与缓解**:

    - 功能：分析 LLM-as-Judge 在分歧偏好上的行为，提出过滤方法
    - 核心发现：LLM-as-Judge 在分歧偏好样本上选出"赢家"的比例（73.8%）与高一致偏好（73.1%）几乎相同
    - 偏见类型：偏好详细格式化输出、偏好合规而非拒绝
    - 缓解方案：使用分布式奖励模型识别分歧样本，从评估基准中移除

### 损失函数 / 训练策略
- Mean-Var (KL)：使用 KL 散度损失训练，将 $r_A - r_B$ 映射到 5 个偏好类别（显著优于 A / 略优于 A / 平局 / 略优于 B / 显著优于 B）
- 对比基线 Mean-Var (NLL, Independent)：使用负对数似然损失且假设独立，效果较差
- Classification (KL)：基于 Likert-5 分数的 5 分类器，预测分数分布
- 所有模型基于 Llama-3-8B-Instruct 训练

## 实验关键数据

### 分布式 vs 标量奖励模型

| 奖励模型 | MultiPref Pref Acc | MultiPref Div AUROC | HS2 Pref Acc | HS2 Div AUROC |
|---------|-------------------|-------------------|-------------|--------------|
| Skywork (27B) | 0.651 | 0.494 | — | — |
| Nemotron (70B) | 0.638 | 0.400 | — | — |
| Bradley-Terry (Agg) | 0.663 | 0.458 | 0.683 | 0.482 |
| Bradley-Terry (All) | 0.648 | 0.438 | 0.678 | 0.489 |
| **Mean-Var (KL, ours)** | **0.664** | **0.615** | **0.684** | **0.582** |
| Classification (KL) | — | — | 0.659 | **0.648** |

### LLM-as-Judge 在不同偏好类型的表现

| 偏好类型 | MultiPref 选出赢家% | HelpSteer2 选出赢家% |
|---------|-------------------|-------------------|
| 高一致偏好 | 73.1% | 64.6% |
| 高一致平局 | 42.6% | 51.9% |
| 分歧偏好（全部）| 73.8% | 57.3% |
| 分歧偏好（显著）| 76.0% | 65.0% |

### 关键发现
- 标准 Bradley-Terry 奖励模型的 Diverging ID AUROC 接近随机（~0.5），无法区分分歧和一致偏好
- Mean-Var (KL) 在保持偏好准确率的同时，将 Div AUROC 从 0.46 提升到 0.62（+0.16）
- LLM-as-Judge 在分歧偏好上表现出与高一致偏好几乎相同的"决断性"，系统性偏向某种风格
- Verbosity（冗长度）是最大的分歧来源（38-44%），而非之前假设的标注噪声

## 亮点与洞察
- **数据洞察深刻**：首次系统性地从真实偏好数据集中分析分歧原因，建立了有实证基础的分歧分类法
- 揭示了一个被广泛忽视的问题：奖励模型对分歧样本和一致样本"一视同仁"，无法支持多元化对齐
- 分布式奖励模型是一个优雅的解决方案：一个模型同时完成偏好预测和分歧识别两个任务
- 对 LLM-as-Judge 偏见的发现对当前流行的评估方法（如 Arena-Hard）具有重要的警示意义

## 局限与展望
- 仅在两个英语数据集上验证，跨语言和跨文化的分歧模式可能不同
- 分布式奖励模型目前未直接集成到 RLHF 训练流程中，如何将分歧信息融入策略优化是开放问题
- 分歧分类法的粒度可能不够——例如 Verbosity 和 Format 可能存在重叠
- LLM-as-Judge 偏见的缓解方法依赖于奖励模型，存在循环依赖风险

## 相关工作与启发
- 与多元化对齐（Pluralistic Alignment）研究方向密切相关，为该领域提供了实证数据支撑
- 分布式奖励建模的思想可推广到其他存在标注不确定性的场景（如安全评估）
- 分歧分类法对偏好数据集的构建和质量控制具有直接指导意义
- 对 reward hacking 问题提供了新的分析视角：一些"对齐失败"可能只是模型学到了特定标注者子群的偏好

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Do We Really Need Curated Malicious Data for Safety Alignment in Multi-Modal LLMs?](../../CVPR2025/llm_alignment/do_we_really_need_curated_malicious_data_for_safety_alignment_in_multi-modal_lar.md)
- [\[AAAI 2026\] When Human Preferences Flip: An Instance-Dependent Robust Loss for RLHF](../../AAAI2026/llm_alignment/when_human_preferences_flip_an_instance-dependent_robust_loss_for_rlhf.md)
- [\[NeurIPS 2025\] Capturing Individual Human Preferences with Reward Features](../../NeurIPS2025/llm_alignment/capturing_individual_human_preferences_with_reward_features.md)
- [\[NeurIPS 2025\] Ask a Strong LLM Judge when Your Reward Model is Uncertain](../../NeurIPS2025/llm_alignment/ask_a_strong_llm_judge_when_your_reward_model_is_uncertain.md)
- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)

</div>

<!-- RELATED:END -->
