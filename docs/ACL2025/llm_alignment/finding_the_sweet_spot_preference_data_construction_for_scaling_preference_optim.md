---
title: >-
  [论文解读] Finding the Sweet Spot: Preference Data Construction for Scaling Preference Optimization
description: >-
  [ACL2025][LLM对齐][DPO] 发现传统的 DPO 偏好数据构建策略（max-min）在增加采样量时性能反而下降，通过基于奖励分布的系统性探索发现 rejected 响应应选在 μ−2σ 而非最小值，据此提出了一种随采样量增加而持续提升的偏好数据构建方法。 DPO（Direct Preference Optimi…
tags:
  - "ACL2025"
  - "LLM对齐"
  - "DPO"
  - "preference optimization"
  - "preference pair construction"
  - "reward distribution"
  - "scaling"
  - "on-policy sampling"
---

# Finding the Sweet Spot: Preference Data Construction for Scaling Preference Optimization

**会议**: ACL2025  
**arXiv**: [2502.16825](https://arxiv.org/abs/2502.16825)  
**代码**: [XYaoooo/DPO_Pair](https://github.com/XYaoooo/DPO_Pair)  
**领域**: LLM Alignment  
**关键词**: DPO, preference optimization, preference pair construction, reward distribution, scaling, on-policy sampling  

## 一句话总结

发现传统的 DPO 偏好数据构建策略（max-min）在增加采样量时性能反而下降，通过基于奖励分布的系统性探索发现 rejected 响应应选在 μ−2σ 而非最小值，据此提出了一种随采样量增加而持续提升的偏好数据构建方法。

## 研究背景与动机

DPO（Direct Preference Optimization）是当前最流行的 LLM 对齐方法之一，通过偏好对（chosen-rejected）直接优化策略模型，避免了 RLHF 中训练奖励模型和 PPO 的复杂性。

当前常用的偏好数据构建流程是：
1. 对每个 prompt 从策略模型采样 n 个 on-policy 响应
2. 用奖励模型评分
3. 选奖励最高的作为 chosen，最低的作为 rejected

直觉上，增加采样数量 n 应该能获得更高质量的 chosen 和更差的 rejected，从而提升对齐效果。**然而实验显示这并非如此——增加 n 后性能反而下降或波动**。

这一反直觉的发现促使作者深入研究偏好对构建的最优策略，特别是 rejected 响应的选择位置对 DPO 训练效果的关键影响。

## 方法详解

### 1. 问题发现：Max-Min 策略失效

在 Llama-3-8B、Llama-3-8B-Instruct、Mistral-7B-v0.1、Mistral-7B-Instruct-v0.2 四个模型上，使用 UltraFeedback 数据集的 prompts，将采样数从 5 扩展到 200。AlpacaEval 2 评测结果：
- Llama Base: 性能波动不稳定
- Llama Instruct: 性能随 n 增加而显著下降
- Mistral Base: 类似下降趋势
- Mistral Instruct: 先微升后下降

### 2. 基于奖励分布的偏好对构建

**核心思路**：不再按排名划分样本，而是基于每个 prompt 的奖励分布（近似正态）来选择偏好对。

对每个 prompt 的 n 个响应奖励，近似为 $N(\mu_i, \sigma_i^2)$，在分布的关键位置选取样本：

**7 个代表性采样点**：
$$\{min, \mu-2\sigma, \mu-\sigma, \mu, \mu+\sigma, \mu+2\sigma, max\}$$

构建所有 $C_7^2 = 21$ 种偏好对组合，训练 21 × 4 = 84 个策略模型，系统性评估每种组合的效果。

### 3. 关键发现

通过 84 个模型在 AlpacaEval 2 上的评测，得出以下结论：

**发现 1：Rejected 响应的最优位置是 μ−2σ，而非 min**

这是最重要的发现。传统选最小奖励的 rejected 样本实际上过于极端，可能导致训练产生"捷径学习"——模型仅学会避免极端差的输出，而非学会高质量与中等质量的区分。

**发现 2：Chosen 响应越好越好（在 rejected 合理的前提下）**

当 rejected 固定在 μ−2σ 时，chosen 从 μ 到 μ+σ 到 μ+2σ 到 max，性能持续提升。(μ+2σ, μ−2σ) 组合在多数设置下最优。例如 Llama-3-8B-Instruct 在此组合下达到 LC win rate 48.18%，比 max-min 策略高约 3 个百分点。

**发现 3：奖励间距过小则效果很差**

(μ+2σ, μ+σ) 这种小间距偏好对仅达到 34.63% LC win rate，因为模型难以学习微小的质量差异。

**发现 4：DPO 训练是鲁棒的**

所有 21 种偏好对都不会导致性能低于 SFT 基线，说明 DPO 对不同偏好对具有基本鲁棒性。

### 4. 可扩展的偏好数据构建策略

基于上述发现，提出简单实用的策略：

- **Rejected 选择**：从 5 个随机样本中选奖励最低的（作为 μ−2σ 的有效近似）
- **Chosen 选择**：从全部 n 个样本中选奖励最高的

这样当 n 增加时，chosen 质量自然提升，而 rejected 保持在 μ−2σ 附近（因为始终从 5 个样本中选最低），实现性能的持续改善。

### 损失函数分析

DPO 目标函数：

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \Big[\log \sigma(r(x,y_w) - r(x,y_l))\Big]$$

其中 $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$

不同偏好对导致不同的训练动态：
- (max, min)：训练 loss 最低但可能**过拟合**
- (max, μ−2σ)：loss 适中，泛化性能最好
- (max, μ+2σ)：loss 停滞不下降，模型**欠拟合**

## 实验

### 实验设置

- **模型**：Llama-3-8B, Llama-3-8B-Instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.2
- **数据**：UltraChat-200k (SFT), UltraFeedback (DPO prompts)
- **奖励模型**：ArmoRM (主实验), Skywork (验证泛化性)
- **采样**：温度 0.8，每 prompt 5-400 样本，使用 vLLM 加速
- **评测**：AlpacaEval 2 (LC win rate + win rate), Arena-Hard

### 主实验：21 种偏好对的全面探索

在 200 样本/prompt 设置下，84 个模型的热力图（Figure 4）清晰显示：
- 每列中（固定 chosen），rejected 在 μ−2σ 时性能最好
- 每行中（固定 rejected），chosen 越高越好
- 对角线附近（小间距）性能最差

### 扩展实验

**更多采样点**：将奖励位置扩展到 μ±3σ 和 μ±4σ，发现：
- μ+3σ/μ+4σ 与 max 无显著差异
- μ−3σ/μ−4σ 与 min 无显著差异
- 说明 μ±2σ 已足够覆盖有意义的奖励空间

**扩展到 400 样本/prompt**：在 Llama-3-8B-Instruct 上结论保持一致。

### 可扩展策略的效果

| n (采样数) | 传统 max-min LC | 传统 max-min WR | 本文方法 LC | 本文方法 WR |
|-----------|--------|--------|--------|--------|
| 5 | ~45% | ~47% | ~45% | ~47% |
| 50 | ~43% | ~44% | ~47% | ~49% |
| 100 | ~42% | ~43% | ~48% | ~50% |
| 200 | ~41% | ~41% | ~49% | ~50% |

传统方法随 n 增加而下降，本文方法持续提升。

### 与先前工作对比

| 数据(方法) | #Sample | AE LC | AE WR | AH WR |
|-----------|---------|-------|-------|-------|
| Baseline* (SimPO) | 5 | 53.7 | 47.5 | 36.5 |
| Baseline* (DPO) | 5 | 48.2 | 47.5 | 35.2 |
| Baseline† (DPO) | 400 | 42.0 | 42.0 | 34.5 |
| **Ours (DPO)** | **400** | **49.1** | **50.2** | **37.3** |

在 400 个样本下，本文方法的 DPO 超越了 Meng et al. (2024) 的 5 样本 DPO baseline，甚至在 WR 和 AH 上超越了其 SimPO baseline。

### 跨奖励模型验证

使用 Skywork 奖励模型替换 ArmoRM，在 Llama-3-8B-Instruct 上验证，趋势一致——性能随 n 增加而提升后趋于平稳。

### 学术基准评测

在 ARC、HellaSwag、TruthfulQA、GSM8K 上无性能下降，说明对齐改进不以牺牲通用能力为代价。

## 亮点与洞察

1. **反直觉发现的价值**：max-min 策略随采样增加反而变差，这一发现对整个 DPO 社区具有重要警示意义——更多采样并不自动等于更好的训练数据
2. **统计分布视角**：将奖励空间按正态分布的关键位置进行系统化划分，提供了可解释且可复现的分析框架。μ−2σ 的有效性可从训练动态角度理解：既提供足够的对比信号，又避免过于极端的样本导致过拟合
3. **实用性极高**：最终策略极其简单——从 5 个样本选最差作 rejected，从全部样本选最好作 chosen。无需额外模型或复杂计算
4. **过拟合的精确诊断**：通过 loss 曲线分析揭示了 max-min 的失败机制：训练 loss 过快下降导致过拟合，而 μ−2σ 提供了"甜蜜点"般的对比难度

## 局限性

1. **依赖强奖励模型**：方法有效性取决于奖励模型的质量，低质量奖励模型可能导致次优结果
2. **计算成本**：生成大量样本（如 200-400/prompt）需要显著的推理计算资源
3. **仅测试 DPO**：未探索 SimPO、IPO、KTO 等其他偏好优化方法是否有同样的发现
4. **正态分布假设**：奖励分布并非总是正态分布，偏态或多模态分布下 μ±nσ 的切分可能不够合理

## 相关工作

- **DPO 及变体**：Rafailov et al. (2023) 的 DPO, Meng et al. (2024) 的 SimPO, Ethayarajh et al. (2024) 的 KTO
- **偏好数据构建**：Dong et al. (2023) 的自生成偏好数据, Kim et al. (2025) 的偏好对研究
- **RLHF**：Ouyang et al. (2022) 的 InstructGPT, Schulman et al. (2017) 的 PPO
- **推理扩展**：Brown et al. (2024) 的 repeated sampling, Snell et al. (2024) 的 inference scaling

## 评分

⭐⭐⭐⭐⭐ (5/5)

选题精准、实验充分（84 个模型的系统分析）、发现具有直接实用价值。揭示了 DPO 社区长期忽视的偏好对构建问题，提出的策略既简单又有效。尤其是在当前 on-policy DPO 训练广泛使用的背景下，这一工作具有很高的实践指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Preference Optimization by Estimating the Ratio of the Data Distribution](../../NeurIPS2025/llm_alignment/preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ACL 2025\] AutoMixAlign: Adaptive Data Mixing for Multi-Task Preference Optimization in LLMs](automixalign_adaptive_data_mixing.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)
- [\[ACL 2025\] Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)

</div>

<!-- RELATED:END -->
