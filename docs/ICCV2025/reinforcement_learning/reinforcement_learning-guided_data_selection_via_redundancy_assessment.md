---
title: >-
  [论文解读] RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment
description: >-
  [ICCV 2025][强化学习][数据选择] 提出 RL-Selector，引入 ε-sample cover 概念量化样本冗余度，将数据选择建模为强化学习过程，通过轻量 A2C 策略网络自适应优化选择策略，在多个基准数据集上以更少数据达到接近甚至超越全量训练的泛化性能。 深度学习模型的成功高度依赖大规模数据集…
tags:
  - "ICCV 2025"
  - "强化学习"
  - "数据选择"
  - "数据冗余"
  - "ε-sample cover"
  - "核心集"
  - "A2C"
  - "训练效率"
---

# RL-Selector: Reinforcement Learning-Guided Data Selection via Redundancy Assessment

**会议**: ICCV 2025  
**arXiv**: [2506.21037](https://arxiv.org/abs/2506.21037)  
**代码**: 待确认  
**领域**: 强化学习 / 数据选择  
**关键词**: 数据选择, 数据冗余, 强化学习, ε-sample cover, 核心集, A2C, 训练效率

## 一句话总结

提出 RL-Selector，引入 ε-sample cover 概念量化样本冗余度，将数据选择建模为强化学习过程，通过轻量 A2C 策略网络自适应优化选择策略，在多个基准数据集上以更少数据达到接近甚至超越全量训练的泛化性能。

## 研究背景与动机

深度学习模型的成功高度依赖大规模数据集，但大规模数据带来了高昂的计算和存储开销。同时，现实数据集往往包含大量冗余样本，不仅浪费资源，还可能导致过拟合。**数据选择（Data Selection）** 旨在训练前识别最具代表性的样本子集，以更少数据实现相当的模型性能。

现有方法存在三大局限：

**重要性评分法的 "group effect" 问题**：基于手工评分（如 EL2N、Forgetting score）逐样本打分，忽视所选子集的整体效果——高分和低分样本的组合可能显著影响模型表现

**忽视训练动态**：多数方法使用收敛后的代理模型进行选择，偏好后期困难样本，而非整个训练过程中真正有价值的样本

**缺乏跨比例灵活性**：按特定选择比例选出的核心集难以迁移到其他比例，需重新完整选择

为此本文提出将数据选择建模为 RL 问题，通过策略学习捕获训练动态和样本间相互关系。

## 方法详解

### 整体框架

RL-Selector 框架包含两个模型：

- **目标模型 $f_\theta$**（如 ResNet-18）：提取特征并估计样本冗余度，动态更新以捕获训练动态，选择完成后不再使用
- **RL 策略模型（A2C）**：轻量级 Advantage Actor-Critic，actor 和 critic 各仅 3 层线性网络，输出每个样本的保留/剔除决策

### 关键设计一：ε-sample cover 概念

**定义**：若样本 $\mathbf{x}_i$ 和 $\mathbf{x}_j$ 属于同一类别且特征距离 $\|\tilde{\mathbf{x}}_i - \tilde{\mathbf{x}}_j\| \leq \epsilon$，则称 $\mathbf{x}_i$ 被 $\mathbf{x}_j$ ε-覆盖。

三个关键理论命题：

- **Proposition 1**：互相 ε-覆盖的样本模型输出差异 $\leq \mathcal{O}(\epsilon)$
- **Proposition 2**：互相 ε-覆盖的样本损失差异趋于零
- **Proposition 3**：互相 ε-覆盖的样本对模型权重的梯度更新几乎相同

核心结论：**高度 ε-覆盖的样本是冗余的，移除它们对模型泛化影响极小**。

### 关键设计二：RL 驱动的选择策略

将数据选择建模为 MDP $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, T)$：

- **状态 $s$**：目标模型倒数第二层输出的特征图，随训练动态变化
- **动作 $a$**：样本级选择分数 $\pi \in \mathbb{R}^N$，训练中连续值，最终以 0.5 阈值二值化

**奖励函数两部分：**

1. **选择比例对齐奖励 $r_1$**：惩罚当前比例偏离目标比例，通过归一化保证对称
2. **冗余度奖励 $r_2$**：鼓励剔除高 ε-覆盖度的样本。每个样本的覆盖度 $E_c(\mathbf{x}_{ki}) = \sum_j D_{kij}$（同类特征距离之和），越小越冗余

总奖励 $r = r_1 + r_2$。奖励 $r_2 = E_c(\mathbf{x}) \odot \pi(\mathbf{x})$。

### 策略优化

采用 A2C 算法：

- Actor 损失：$\mathcal{L}(\theta_a) = -\log \pi_{\theta_a}(a|s) A(s,a)$
- Critic 损失：$\mathcal{L}(\theta_c) = \mathbb{E}[(A(s,a))^2]$
- 选择结果约束在目标比例 ±1% 范围内

### 迁移微调

支持从已有选择策略出发仅用 15 epoch 微调适配新比例，大幅降低选择成本。

### 泛化分析

通过影响函数（Influence Function）证明：ε-覆盖的被剔除样本与保留样本对模型参数和测试损失影响相近，理论保证数据选择不损害泛化。

## 实验关键数据

### 主实验概况

在所有基准数据集（CIFAR-100、Tiny-ImageNet、ImageNet-1k）和所有选择比例上 **一致超越** 10 个 SOTA 基线（Random、EL2N、MoSo、GraNd、Glister、Herding、CG-Score、Forgetting、Moderate-DS、Self-sup. prototypes）。大规模 ImageNet-1k 上优势尤为显著。

### 跨架构泛化表格 (CIFAR-10, ResNet-18 选择 → ResNet-50 训练)

| 方法 | 60% | 70% | 80% | 90% | 100% |
|------|-----|-----|-----|-----|------|
| EL2N | 90.32 | 90.97 | 91.61 | 91.75 | 92.34 |
| MoSo | 90.73 | 91.13 | 91.50 | 92.23 | 92.34 |
| Moderate-DS | 90.42 | 90.84 | 90.91 | 91.88 | 92.34 |
| Self-sup. Proto | 90.11 | 90.85 | 91.82 | 91.98 | 92.34 |
| **RL-Selector** | **91.79** | **92.06** | **91.93** | **92.79** | 92.34 |

关键发现：90% 选择比例下达到 **92.79%**，超越全量训练 92.34%。

### 消融实验关键结论

| 实验设置 | 关键发现 |
|----------|----------|
| ε-sample cover vs 其他度量 | ε-sample cover 更好量化冗余度 |
| RL 策略 vs 静态选择 | RL 动态优化显著优于静态 |
| A2C vs 其他 RL 算法 | A2C 跨比例稳定性最好 |
| 迁移微调 (15 epochs) | 性能仅轻微下降，效率大幅提升 |

### 分布外泛化

ImageNet-1k 上选出子集训练的模型在 ImageNet-A/R/Hard 等分布外测试集上**优于全量训练模型**，说明去冗余反而提升鲁棒泛化。

## 亮点与洞察

1. **理论驱动的冗余度量**：ε-sample cover 从特征空间距离定义冗余，三个命题证明冗余样本在输出、损失和梯度三个层面的可替代性
2. **RL 建模的合理性**：数据选择需考虑样本组合效应和训练动态，RL 框架恰好匹配
3. **90% 选择超越全量训练**：CIFAR-10 和 ImageNet-1k 均观察到此现象，有力证明去冗余能提升泛化
4. **轻量 RL 模块**：A2C 的 actor/critic 仅 3 层线性网络，开销低
5. **跨架构泛化**：用 ResNet-18 选择的子集在 ResNet-50、ViT、Swin 上均表现良好

## 局限性

1. RL 选择需完整训练目标模型，超大数据集（如 LAION-5B）选择成本仍高
2. ε-sample cover 依赖欧式距离，对特征提取器质量有强依赖
3. 仅在分类任务验证，未扩展至检测、分割、生成等
4. 理论分析基于线性化 ReLU 网络简化假设
5. 迁移微调在低选择比例下性能下降明显

## 相关工作

- **重要性评分法**：EL2N、Forgetting Score、GraNd、Memorization——逐样本打分，忽视 group effect
- **数据分布法**：Moderate-DS（中位数选择）、CCS（分布覆盖）、D2 pruning（图采样）
- **优化法**：Glister（双层优化）、Self-supervised prototypes
- **强化学习**：A2C 在效率和稳定性上优于 DQN、PPO

## 评分

| 维度 | 分数 (1-10) |
|------|-------------|
| 新颖性 | 7 |
| 理论深度 | 7 |
| 实验充分性 | 8 |
| 实用价值 | 7 |
| 写作质量 | 7 |
| **总评** | **7** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] LearnAlign: Data Selection for LLM Reinforcement Learning with Improved Gradient Alignment](../../ACL2026/reinforcement_learning/learnalign_data_selection_for_llm_reinforcement_learning_with_improved_gradient_.md)
- [\[ICML 2026\] Single-Rollout Hidden-State Dynamics for Training-Free RLVR Data Selection](../../ICML2026/reinforcement_learning/single-rollout_hidden-state_dynamics_for_training-free_rlvr_data_selection.md)
- [\[ICCV 2025\] mDP3: A Training-free Approach for List-wise Frame Selection in Video-LLMs](mdp3_a_training-free_approach_for_list-wise_frame_selection_in_video-llms.md)
- [\[ICML 2025\] Zero-Shot Generalization of Vision-Based RL Without Data Augmentation](../../ICML2025/reinforcement_learning/zero-shot_generalization_of_vision-based_rl_without_data_augmentation.md)
- [\[ICCV 2025\] Progressor: A Perceptually Guided Reward Estimator with Self-Supervised Online Refinement](progressor_a_perceptually_guided_reward_estimator_with_self-supervised_online_re.md)

</div>

<!-- RELATED:END -->
