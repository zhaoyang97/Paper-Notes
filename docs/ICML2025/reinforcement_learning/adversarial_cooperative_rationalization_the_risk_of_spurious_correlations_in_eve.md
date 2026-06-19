---
title: >-
  [论文解读] Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets
description: >-
  [ICML 2025][强化学习][自解释模型] 揭示协作理据化框架（RNP）中的隐蔽缺陷——即使在干净数据集上，生成器的采样偏差也会引入理据与标签间的虚假相关，提出对抗检测+指令干预方法，在文本和图分类上显著超越现有方法。 领域现状 领域现状：协作理据化（RNP）是主流的模型无关自解释框架——生成器选择输入中最信息的子集作…
tags:
  - "ICML 2025"
  - "强化学习"
  - "自解释模型"
  - "理据化"
  - "虚假相关"
  - "对抗攻击"
  - "采样偏差"
---

# Adversarial Cooperative Rationalization: The Risk of Spurious Correlations in Even Clean Datasets

**会议**: ICML 2025  
**arXiv**: [2505.02118](https://arxiv.org/abs/2505.02118)  
**代码**: [https://github.com/jugechengzi/Rationalization-A2I](https://github.com/jugechengzi/Rationalization-A2I)  
**领域**: 强化学习/可解释性  
**关键词**: 自解释模型, 理据化, 虚假相关, 对抗攻击, 采样偏差

## 一句话总结
揭示协作理据化框架（RNP）中的隐蔽缺陷——即使在干净数据集上，生成器的采样偏差也会引入理据与标签间的虚假相关，提出对抗检测+指令干预方法，在文本和图分类上显著超越现有方法。

## 研究背景与动机

### 领域现状

**领域现状**：协作理据化（RNP）是主流的模型无关自解释框架——生成器选择输入中最信息的子集作为理据，预测器基于理据进行预测，两者协作训练最大化准确率。

**现有痛点**：作者发现一个反直觉的现象——即使去掉"最大化准确率"目标让生成器随机选择，预测器仍能达到很高准确率。说明预测器可能利用了理据选择过程引入的虚假相关。

**核心矛盾**：生成器的采样过程改变了数据分布——$P(Y|Z)$ 在采样后可能不等于 $P(Y|Z|g)$，即使原始数据中 $Y \perp Z$。

**本文目标**：检测和消除理据化过程引入的虚假相关。

**切入角度**：通过对抗攻击暴露虚假相关→用指令机制阻止预测器学习这些虚假模式。

**核心 idea**：对抗性检查+指令干预的两阶段方法。

## 方法详解

### 整体框架
1. 分析：证明生成器的采样偏差如何引入虚假相关
2. 攻击：设计对抗方法暴露这些相关
3. 防御：引入"指令"机制防止预测器利用虚假相关

### 关键设计

1. **采样偏差分析**:

    - 功能：理论分析生成器采样如何改变数据分布
    - 核心思路：$Y \perp T$ 在原始数据集中不意味着 $Y \perp T$ 在采样后的 $(Z,Y)$ 对中
    - 设计动机：解释了为什么随机理据也能达到高准确率

2. **对抗检测+指令防御**:

    - 功能：(a) 对抗攻击识别哪些模式是虚假相关; (b) 指令机制告诉预测器忽略这些模式
    - 核心思路：用对抗生成器专门构造利用虚假相关的理据→分析哪些特征被利用→在正常训练中加入指令排除这些特征
    - 设计动机：先"知道敌人是谁"再"告诉模型不要跟随"

### 损失函数 / 训练策略
- 标准理据化损失 + 指令正则化项
- 适用于 GRU、BERT、GCN

## 实验关键数据

### 主实验
6 个文本 + 2 个图分类数据集：

| 方法 | 文本F1 (平均) | 图F1 (平均) |
|------|-------------|------------|
| RNP (原始) | 72.3% | 68.5% |
| A2I (本文) | **81.7%** | **78.2%** |
| LLaMA-3.1-8B | 79.5% | - |

### 关键发现
- 在所有数据集上均显著超越现有理据化方法
- 甚至在部分任务上超越 LLaMA-3.1-8B
- 虚假相关在所有理据化方法中普遍存在

## 亮点与洞察
- **"干净数据集也有虚假相关"**的发现颠覆了常识——采样过程本身就是偏差来源
- 对抗+指令的两阶段框架简洁有效

## 局限与展望
- 对抗检测增加训练复杂度
- 指令设计需要领域知识

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 揭示了被忽视的根本问题
- 实验充分度: ⭐⭐⭐⭐ 文本+图，多架构
- 写作质量: ⭐⭐⭐⭐ 分析深入
- 价值: ⭐⭐⭐⭐ 对可解释AI研究有重要警示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Enhancing Cooperative Multi-Agent Reinforcement Learning with State Modelling and Adversarial Exploration](enhancing_cooperative_multi-agent_reinforcement_learning_with_state_modelling_an.md)
- [\[NeurIPS 2025\] Learning to Clean: Reinforcement Learning for Noisy Label Correction](../../NeurIPS2025/reinforcement_learning/learning_to_clean_reinforcement_learning_for_noisy_label_correction.md)
- [\[NeurIPS 2025\] Risk-Averse Total-Reward Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/risk-averse_total-reward_reinforcement_learning.md)
- [\[NeurIPS 2025\] Risk-Averse Constrained Reinforcement Learning with Optimized Certainty Equivalents](../../NeurIPS2025/reinforcement_learning/risk-averse_constrained_reinforcement_learning_with_optimized_certainty_equivale.md)
- [\[ICML 2025\] Learning to Incentivize in Repeated Principal-Agent Problems with Adversarial Agent Arrivals](learning_to_incentivize_in_repeated_principal-agent_problems_with_adversarial_ag.md)

</div>

<!-- RELATED:END -->
