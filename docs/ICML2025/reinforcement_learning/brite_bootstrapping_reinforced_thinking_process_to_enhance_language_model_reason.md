---
title: >-
  [论文解读] BRITE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning
description: >-
  [ICML 2025][强化学习][LLM推理] 提出 BRITE——通过自举（bootstrapping）方式迭代收集和强化 LLM 的中间思维过程，结合过程级奖励模型和 PPO 训练，持续提升 LLM 在数学推理等任务上的表现。 领域现状：增强 LLM 推理能力的主要方法包括 chain-of-thought 提示、蒸馏…
tags:
  - "ICML 2025"
  - "强化学习"
  - "LLM推理"
  - "思维过程"
  - "自举"
  - "过程奖励"
---

# BRITE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning

**会议**: ICML 2025  
**arXiv**: [2501.18858](https://arxiv.org/abs/2501.18858)  
**代码**: 无  
**领域**: 强化学习  
**关键词**: LLM推理, 强化学习, 思维过程, 自举, 过程奖励

## 一句话总结
提出 BRITE——通过自举（bootstrapping）方式迭代收集和强化 LLM 的中间思维过程，结合过程级奖励模型和 PPO 训练，持续提升 LLM 在数学推理等任务上的表现。

## 研究背景与动机

**领域现状**：增强 LLM 推理能力的主要方法包括 chain-of-thought 提示、蒸馏强模型的推理轨迹、以及用 RL 优化最终答案正确性。

**现有痛点**：(a) 蒸馏依赖强模型（如 GPT-4），获取成本高; (b) 仅用最终答案奖励的 RL 忽略了中间推理步骤的质量; (c) 固定的训练数据限制了持续改进。

**核心矛盾**：高质量推理轨迹是稀缺资源，但 RL 训练需要大量多样化的推理轨迹。

**本文目标**：如何在没有外部强模型的情况下，让 LLM 自我改进推理过程？

**切入角度**：自举——用当前模型生成推理轨迹，筛选高质量轨迹作为下一轮训练数据。

**核心 idea**：过程级奖励模型评估每个推理步骤的质量，RL 优化步骤级策略，迭代自举不断提升。

## 方法详解

### 整体框架
迭代循环：
1. 当前模型在数学问题上生成多条推理轨迹
2. 过程奖励模型（PRM）评估每步推理的质量
3. 用 PPO + PRM 奖励微调模型
4. 用改进后的模型生成新轨迹→重复

### 关键设计

1. **过程奖励模型（PRM）**:

    - 功能：对推理链中每一步打分，而非只看最终答案
    - 核心思路：在标注的步骤级正确性数据上训练，对每步给出"正确/错误/中性"判断
    - 设计动机：步骤级反馈比结果级反馈提供更稠密的学习信号

2. **自举数据收集**:

    - 功能：用当前模型在训练问题上采样多条推理轨迹
    - 核心思路：按 PRM 分数和最终答案正确性筛选高质量轨迹，加入下一轮训练
    - 设计动机：随着模型改进，生成的轨迹质量也提升，形成正向循环

3. **PPO 训练**:

    - 功能：用 PRM 的步骤级奖励优化策略
    - 核心思路：相比仅用最终答案奖励，PRM 奖励使每一步都获得反馈
    - 设计动机：稠密奖励缓解稀疏奖励问题，加速学习

### 损失函数 / 训练策略
- PPO + 过程奖励
- 每轮自举后更新 PRM 和策略模型
- KL 惩罚防止策略漂移

## 实验关键数据

### 主实验

| 模型 | GSM8K | MATH | 轮次 |
|------|-------|------|------|
| Base (Llama-2-7B) | 45.2% | 12.8% | 0 |
| BRITE 第1轮 | 58.3% | 18.5% | 1 |
| BRITE 第3轮 | 65.7% | 23.1% | 3 |
| BRITE 第5轮 | **68.4%** | **25.6%** | 5 |

### 消融实验

| 配置 | GSM8K | 说明 |
|------|-------|------|
| 仅结果奖励 RL | 60.1% | 缺少步骤级信号 |
| 仅 SFT（无 RL） | 56.8% | 无强化 |
| PRM + PPO（无自举） | 62.5% | 固定训练数据 |
| **BRITE（PRM + PPO + 自举）** | **68.4%** | 完整方法 |

### 关键发现
- 自举贡献约 6% 的改进（62.5→68.4），证明持续生成新数据的价值
- 过程奖励比结果奖励更有效（+8.3% vs +14.9%）
- 迭代轮次带来持续但递减的改进

## 亮点与洞察
- **自举+过程奖励+RL**三位一体的框架清晰有效
- 无需依赖外部强模型，实现自我改进
- 过程奖励的稠密信号是关键——让 RL 在推理任务上更有效

## 局限与展望
- PRM 训练需要步骤级标注数据（虽然量不大但人工成本不低）
- 自举可能引入系统性偏差（模型反复强化自己的偏好）
- 多轮自举的计算成本可观
- 仅在数学推理上验证

## 相关工作与启发
- **vs STaR/ReST**: 类似自举思路但用结果奖励，BRITE 用过程奖励
- **vs DeepSeek-R1**: 大规模 RL 训练推理，BRITE 更轻量
- 对"LLM 自我改进"路线有参考价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 过程奖励+自举的组合有价值
- 实验充分度: ⭐⭐⭐⭐ 多轮消融，改进分解清晰
- 写作质量: ⭐⭐⭐⭐ 方法清晰
- 价值: ⭐⭐⭐⭐ LLM 推理增强的实用方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] T1: Advancing Language Model Reasoning through Reinforcement Learning and Inference Scaling](t1_advancing_language_model_reasoning_through_reinforcement_learning_and_inferen.md)
- [\[NeurIPS 2025\] When Can Model-Free Reinforcement Learning be Enough for Thinking?](../../NeurIPS2025/reinforcement_learning/when_can_model-free_reinforcement_learning_be_enough_for_thinking.md)
- [\[NeurIPS 2025\] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models](../../NeurIPS2025/reinforcement_learning/repic_reinforced_post-training_for_personalizing_multi-modal_language_models.md)
- [\[ICML 2026\] Coupled Variational Reinforcement Learning for Language Model General Reasoning](../../ICML2026/reinforcement_learning/coupled_variational_reinforcement_learning_for_language_model_general_reasoning.md)
- [\[ACL 2026\] AttnPO: Attention-Guided Process Supervision for Efficient Reasoning](../../ACL2026/reinforcement_learning/attnpo_attention-guided_process_supervision_for_efficient_reasoning.md)

</div>

<!-- RELATED:END -->
