---
title: >-
  [论文解读] DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO
description: >-
  [NeurIPS 2025][LLM对齐][视频推理] 提出DeepVideo-R1，将GRPO重新表述为回归优势值的Reg-GRPO（消除clipping/min等保护机制），同时通过难度感知数据增强缓解优势值消失问题，在视频推理任务上相比标准GRPO提升高达10.1个百分点。 领域现状 领域现状：基于RL的后训练（如GR…
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "视频推理"
  - "强化学习微调"
  - "GRPO"
  - "回归目标"
  - "难度感知增强"
---

# DeepVideo-R1: Video Reinforcement Fine-Tuning via Difficulty-aware Regressive GRPO

**会议**: NeurIPS 2025  
**arXiv**: [2506.07464](https://arxiv.org/abs/2506.07464)  
**代码**: [GitHub](https://github.com/mlvlab/DeepVideoR1)  
**领域**: LLM对齐 / 视频大语言模型  
**关键词**: 视频推理, 强化学习微调, GRPO, 回归目标, 难度感知增强

## 一句话总结

提出DeepVideo-R1，将GRPO重新表述为回归优势值的Reg-GRPO（消除clipping/min等保护机制），同时通过难度感知数据增强缓解优势值消失问题，在视频推理任务上相比标准GRPO提升高达10.1个百分点。

## 研究背景与动机

### 领域现状

**领域现状**：基于RL的后训练（如GRPO）可有效增强LLM推理能力，但在视频大语言模型（VideoLLM）中的应用仍不充分

### 现有痛点

**现有痛点**：GRPO应用于VideoLLM面临两个关键问题：

### 核心矛盾

**核心矛盾**：保护机制依赖**：PPO风格的clipping和min操作在策略偏离过大时产生零梯度，阻碍探索和收敛

### 解决思路

**解决思路**：优势值消失**：样本过易或过难时组内奖励相同，优势值为零，训练信号丢失

### 补充说明

**补充说明**：视频推理涉及复杂时空语义理解，这两个问题在视频任务中尤为突出

### 补充说明

**补充说明**：已有工作主要关注设计奖励函数，对GRPO算法本身的改进相対不足

## 方法详解

### 整体框架

DeepVideo-R1包含两个关键创新：（1）Reg-GRPO将GRPO目标改为直接回归组相对优势值，无需裁剪和min等保护机制；（2）难度感知数据增强根据样本难度动态调整输入，确保多样化的奖励信号。

### 关键设计

1. **回归式GRPO（Reg-GRPO）**:
    - 功能：将RL目标从PPO风格优化转为直接回归优势值
    - 核心思路：利用KL约束RL目标闭式解的重参数化，定义预测优势 $\hat{A}_\theta^{(i)} = \frac{\rho(\mathbf{x}, \mathbf{y}^{(i)}) - \mu_\rho}{\sigma_\rho}$，其中 $\rho = \log \frac{\pi_\theta}{\pi_{\theta_{old}}}$，最小化与目标优势的MSE
    - 设计动机：回归损失天然没有clipping截断问题，且归一化自然消除配分函数 $Z(\mathbf{x})$

2. **难度感知数据增强**:
    - 功能：根据样本难度动态调整视频-文本输入
    - 核心思路：用回放缓冲区的平均奖励作参照，计算难度 $\Delta_\mathcal{R}(\mathbf{x})$
    - 设计动机：适中难度样本产生最多样的奖励分布，保证有效梯度

3. **双向难度调节**:
    - **降低难度**（困难样本）：从成功推理轨迹中提取部分推理线索注入提示，强度按难度自适应缩放
    - **增加难度**（简单样本）：对视频帧添加高斯噪声或遮蔽，强度与容易程度成正比

### 损失函数 / 训练策略

$$\mathcal{L}_{\text{Reg-GRPO}}(\theta) = \mathbb{E}\left[(\hat{A}^{(i)} - \hat{A}_\theta^{(i)})^2 - \beta \mathbb{D}_{KL}[\pi_\theta || \pi_{ref}]\right]$$

- KL散度约束防止策略过度偏离参考模型
- 回放缓冲区存储最近 $W$ 步数据用于动态难度基准计算

## 实验关键数据

### 主实验（表格）

SEED-Bench-R1验证集和LongVideoBench表现：

| 方法 | SEED-Bench-R1 (Acc) | LongVideoBench |
|------|---------------------|----------------|
| Qwen2.5-VL-7B (SFT) | 55.4 | 57.3 |
| + GRPO | 55.8 | 54.1 |
| + Reg-GRPO | 63.2 | 59.4 |
| + **DeepVideo-R1** | **65.9** | **60.7** |

相比GRPO提升10.1分（SEED-Bench-R1）。

### 消融实验

- **Reg-GRPO vs GRPO**：所有基准一致优于GRPO，收敛更快
- **难度增强贡献**：在Reg-GRPO基础上额外提升2.3分
- **降难 vs 增难**：单独使用均有效，联合效果最佳
- **零优势值比例**：难度增强从约30%降至约10%

### 关键发现

- 回归目标梯度更稳定，无clipping截断导致的零梯度区域
- 难度感知增强有效解决vanishing advantage的根本原因——奖励方差为零
- 在ID和OOD任务上均有一致提升，表明增强的是泛化能力

## 亮点与洞察

- Reg-GRPO推导简洁：从RL闭式解出发，配分函数在组归一化中自然消除
- 难度感知增强是curriculum learning的一种RL-native实现
- 从成功路径提取推理线索作为降难手段，是有趣的自我引导策略
- 方法不限于视频领域，适用于任何使用GRPO的场景

## 局限与展望

- 仅在7B规模模型上验证，更大规模的缩放效果未知
- 推理线索提取需额外生成步骤，增加数据准备成本
- 与DPO等其他对齐方法的对比缺失
- 回放缓冲区窗口大小的敏感性分析不够深入

## 相关工作与启发

- Reg-GRPO与REBEL（直接奖励回归）思路相近，但在组相对设置下更自然
- 与NoisyRollout互补：NoisyRollout改进探索多样性，本文改进优化目标
- 难度感知思想可与VideoChat-R1、TimeZero等VideoLLM RL工作结合

## 评分

- ⭐⭐⭐⭐ — RL算法改进理论清晰、效果显著，难度增强策略实用，但规模验证有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay](improving_data_efficiency_for_llm_reinforcement_fine-tuning_through_difficulty-t.md)
- [\[NeurIPS 2025\] Mechanism Design for LLM Fine-tuning with Multiple Reward Models](mechanism_design_for_llm_fine-tuning_with_multiple_reward_models.md)
- [\[NeurIPS 2025\] Attack via Overfitting: 10-shot Benign Fine-tuning to Jailbreak LLMs](attack_via_overfitting_10-shot_benign_fine-tuning_to_jailbreak_llms.md)
- [\[ACL 2026\] Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO](../../ACL2026/llm_alignment/mitigating_selection_bias_in_large_language_models_via_permutation-aware_grpo.md)
- [\[NeurIPS 2025\] Strategyproof Reinforcement Learning from Human Feedback](strategyproof_reinforcement_learning_from_human_feedback.md)

</div>

<!-- RELATED:END -->
