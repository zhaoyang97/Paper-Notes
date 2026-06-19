---
title: >-
  [论文解读] Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning
description: >-
  [ICCV 2025][视频生成][视觉强化学习] 提出DisWM框架，通过从"干扰视频"中预训练解纠缠表示，然后通过离线到在线的潜空间蒸馏将语义知识迁移到下游世界模型，提升视觉强化学习在环境变化下的样本效率和鲁棒性。 核心矛盾 核心矛盾：领域现状：视觉强化学习（VRL）在实际场景中面临严峻挑战：环境的复杂性、易变性和视觉干…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "视觉强化学习"
  - "世界模型"
  - "解纠缠表示"
  - "知识迁移"
  - "潜空间蒸馏"
---

# Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning

**会议**: ICCV 2025  
**arXiv**: [2503.08751](https://arxiv.org/abs/2503.08751)  
**代码**: [https://qiwang067.github.io/diswm](https://qiwang067.github.io/diswm)  
**领域**: 视频生成  
**关键词**: 视觉强化学习, 世界模型, 解纠缠表示, 知识迁移, 潜空间蒸馏

## 一句话总结

提出DisWM框架，通过从"干扰视频"中预训练解纠缠表示，然后通过离线到在线的潜空间蒸馏将语义知识迁移到下游世界模型，提升视觉强化学习在环境变化下的样本效率和鲁棒性。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：视觉强化学习（VRL）在实际场景中面临严峻挑战：环境的复杂性、易变性和视觉干扰会导致性能严重下降。即使微小的环境变化（如光照条件改变）也会导致像素级的巨大偏移，使训练好的策略失效。

现有解纠缠表示学习（DRL）方法虽然有望提升VRL的可解释性和鲁棒性，但存在关键局限：它们通常从零开始学习，缺乏对世界的先验知识，需要大量环境交互才能学到理想的行为。

本文的核心思想是：利用容易获得的"干扰视频"（包含视觉干扰的视频）提取语义先验知识，并将这种解纠缠能力通过潜空间蒸馏迁移到下游控制任务中。关键的是，预训练视频和下游任务可以来自不同的域（如DMC和MuJoCo），在视觉外观、物理动力学、动作空间和奖励函数上都可以不同。

## 方法详解

### 整体框架

DisWM包含三个阶段：（1）在干扰视频上离线预训练无动作的视频预测模型以提取解纠缠特征；（2）通过潜空间蒸馏将语义知识迁移到在线世界模型；（3）在线微调世界模型，结合动作和奖励信息进一步增强解纠缠表示。

### 关键设计

1. **解纠缠表示预训练（Disentangled Representation Pretraining）**: 在无动作的干扰视频上训练一个基于$\beta$-VAE的视频预测模型，包含三个组件：后验学习器（将观测$o_t$编码为潜状态$z_t$）、先验模块（基于历史状态预测未来潜状态）和解码器（从$z_t$重建$\hat{o}_t$）。训练损失包含三部分：图像重建损失、无动作KL散度损失（保证先验-后验一致性）、解纠缠损失$\beta_2 \text{KL}[q_{\phi'}(\mathbf{z}_t|o_t) \| p(\mathbf{z}_t)]$——通过将后验分布推向标准正态分布$\mathcal{N}(\mathbf{0}, I)$来增强潜空间的正交性和解纠缠性。设计动机：干扰视频中丰富的视觉变化自然提供了解纠缠学习所需的因子变化，为下游任务注入"世界知识"。

2. **离线到在线潜空间蒸馏（Offline-to-Online Latent Distillation）**: 简单的预训练-微调范式在大域差异下会导致预训练的解纠缠信息被覆盖。因此通过KL散度蒸馏将预训练模型的解纠缠潜变量$\mathbf{z}_{disen}$的能力迁移到世界模型的$\mathbf{z}_{task}$中：$\mathcal{L}_{distill} = \text{KL}(\mathbf{z}_{disen} \| \mathbf{z}_{task}) = \sum \mathbf{z}_{disen} \cdot \log(\frac{\mathbf{z}_{disen}}{\mathbf{z}_{task}})$。蒸馏权重$\eta$在适应阶段从0.1逐步衰减到0.01，实现从重度依赖预训练知识到逐步自主学习的平滑过渡。

3. **解纠缠世界模型适应（Disentangled World Model Adaptation）**: 构建基于DreamerV2的完整世界模型$\mathcal{M}_\phi$，包含循环转移函数$h_t = f_\phi(h_{t-1}, z_{t-1}, a_{t-1})$、后验/先验状态、重建、奖励预测和折扣因子预测。总训练损失在标准世界模型损失基础上增加了解纠缠正则化和蒸馏损失两项：$\mathcal{L}(\phi) = \text{重建} + \text{奖励预测} + \text{折扣预测} + \alpha\text{KL散度} + \beta\text{KL}[q_\phi(\mathbf{z}_t|o_t) \| p(\mathbf{z}_t)] + \eta\mathcal{L}_{distill}$。关键洞察：在线微调阶段引入动作和奖励丰富了数据多样性，反过来增强了解纠缠表示学习——形成正向循环。

### 损失函数 / 训练策略

预训练阶段使用100万帧干扰视频数据集（通过DreamerV2与带视觉颜色干扰的环境交互生成）。在线微调限制在$1\times10^6$环境步。潜变量维度$\mathbf{z}_{disen}$和$\mathbf{z}_{task}$均设为20。视觉观测resize至$64\times64$。训练在单卡RTX 3090上约需16小时、55GB显存。在训练中点会更换颜色方案，模拟变化的干扰。

## 实验关键数据

### 主实验

| 任务 | DisWM | DreamerV2 | APV | TED | DV2 Finetune | 说明 |
|------|-------|-----------|-----|-----|--------------|------|
| Cheetah Run→Walker Walk | 最优 | 较差 | 中等 | 中等 | 次优 | 跨域迁移 |
| Reacher Easy→Cheetah Run | 最优 | 较差 | 中等偏下 | 中等 | 次优 | 跨域迁移 |
| Cheetah Run→Hopper Stand | 最优 | 较差 | 低 | 中等 | 次优 | DMC内迁移 |
| Finger Spin→Reacher Easy | 最优 | 较差 | 低 | 中等 | 次优 | DMC内迁移 |
| Finger Spin→Cartpole Swingup | 最优 | 较差 | 中等 | 中等 | 次优 | DMC内迁移 |
| Reacher Easy→Pusher (MuJoCo) | 最优 | 较差 | 低 | 中等 | 中等 | 跨域（DMC→MuJoCo） |

DisWM在所有6个任务上一致达到最优或接近最优的样本效率和最终性能。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| DisWM完整 | 最优episode return | 基线 |
| w/o 潜空间蒸馏 | 早期训练性能下降 | 蒸馏在训练早期提供关键的知识迁移 |
| w/o 解纠缠约束（预训练+微调） | 显著性能下降 | DRL训练和解纠缠表示对学习效率至关重要 |
| β太小（解纠缠不足） | 学到纠缠表示 | 无法有效处理环境变化 |
| β太大（过度解纠缠） | 重建质量下降 | 影响环境建模精度 |
| η太低 | 知识迁移不足 | 下游agent获取先验知识不够 |
| η太高 | 过拟合预训练模型 | 不利于下游任务自适应学习 |
| 不同预训练视频源 | 均优于无预训练DreamerV2 | 框架对预训练域选择鲁棒 |

### 关键发现

- DisWM即使在预训练域和下游域差距极大时（DMC→MuJoCo，动力学/动作空间/奖励均不同）仍能有效迁移语义知识
- 使用任何DMC任务的干扰视频预训练都能提升Cartpole Swingup的性能，说明框架对预训练域选择具有鲁棒性
- β-VAE的遍历可视化清楚展示了预训练模型成功学到了颜色、位置等独立因子
- MuJoCo Pusher的微调阶段可视化展示了世界模型对臂颜色、物体位置等属性的细粒度解纠缠

## 亮点与洞察

- 将VRL的环境变化鲁棒性问题框架化为域迁移学习问题，视角新颖
- "干扰视频"的利用很巧妙：视觉干扰恰好为解纠缠学习提供了天然的因子变化
- 离线到在线蒸馏设计避免了直接微调导致的知识遗忘，且渐进式权重衰减策略实用
- 在线微调阶段动作/奖励信息反哺解纠缠学习的正向循环是一个有趣的发现

## 局限与展望

- 在更复杂的环境变化（如时变背景视频干扰）中解纠缠学习仍面临挑战
- 预训练数据集（100万帧）的生成本身需要与环境交互，可考虑使用真实世界的无标注视频
- 当前仅在连续控制任务上验证，可扩展到离散动作空间或更复杂的机器人操作任务
- 解纠缠的质量依赖于β的调节，缺乏自适应机制

## 相关工作与启发

- 与APV、IPV等视频预训练世界模型方法对话，DisWM通过解纠缠约束提供了更强的环境变化鲁棒性
- 解纠缠表示+世界模型的组合为可解释VRL提供了有前景的方向
- 潜空间蒸馏的跨域迁移思路可推广到sim-to-real等更实际的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ 干扰视频→解纠缠预训练→世界模型蒸馏的pipeline设计新颖
- 实验充分度: ⭐⭐⭐⭐ 六个任务、完整消融、灵敏度分析、跨域验证
- 写作质量: ⭐⭐⭐⭐ 框架清晰，可视化丰富
- 价值: ⭐⭐⭐⭐ 为环境变化下的VRL提供了实用的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] DisMo: Disentangled Motion Representations for Open-World Motion Transfer](../../NeurIPS2025/video_generation/dismo_disentangled_motion_representations_for_openworld_moti.md)
- [\[NeurIPS 2025\] RLGF: Reinforcement Learning with Geometric Feedback for Autonomous Driving Video Generation](../../NeurIPS2025/video_generation/rlgf_reinforcement_learning_with_geometric_feedback_for_autonomous_driving_video.md)
- [\[ICCV 2025\] Long-Context State-Space Video World Models](long-context_state-space_video_world_models.md)
- [\[ICCV 2025\] NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors](normalcrafter_learning_temporally_consistent_normals_from_video_diffusion_priors.md)
- [\[CVPR 2026\] NS-Diff: Fluid Navier-Stokes Guided Video Diffusion via Reinforcement Learning](../../CVPR2026/video_generation/ns-diff_fluid_navier-stokes_guided_video_diffusion_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
