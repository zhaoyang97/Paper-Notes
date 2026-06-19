---
title: >-
  [论文解读] RLVR-World: Training World Models with Reinforcement Learning
description: >-
  [NeurIPS 2025][图像生成][世界模型] 提出 RLVR-World 框架，将强化学习可验证奖励（RLVR）范式拓展到世界模型训练，通过将目标度量（如预测准确率、感知质量）作为可验证奖励直接优化，在语言和视频两类世界模型上取得显著提升。 世界模型旨在预测环境状态在动作干预下的转移，是模型驱动规划和强化学习的核心组…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "世界模型"
  - "RLVR"
  - "GRPO"
  - "视频预测"
  - "自回归生成"
---

# RLVR-World: Training World Models with Reinforcement Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.13934](https://arxiv.org/abs/2505.13934)  
**代码**: [GitHub](https://thuml.github.io/RLVR-World)  
**领域**: 世界模型 / 强化学习后训练  
**关键词**: 世界模型, RLVR, GRPO, 视频预测, 自回归生成

## 一句话总结

提出 RLVR-World 框架，将强化学习可验证奖励（RLVR）范式拓展到世界模型训练，通过将目标度量（如预测准确率、感知质量）作为可验证奖励直接优化，在语言和视频两类世界模型上取得显著提升。

## 研究背景与动机

世界模型旨在预测环境状态在动作干预下的转移，是模型驱动规划和强化学习的核心组件。当前世界模型普遍使用最大似然估计（MLE）作为训练目标（例如语言模型的 next-token prediction、扩散模型的变分下界优化），但这类代理目标与世界模型的真实使命——**状态转移预测精度或感知质量**——存在本质偏差。

具体来说，MLE 的问题体现在三个层面：

**目标错位**：似然目标与下游评价指标（如准确率、LPIPS）不直接对齐，会导致重复生成、幻觉等退化现象

**非端到端优化**：基于离散 tokenizer 的自回归架构无法直接优化像素级指标

**多步误差累积**：teacher-forcing 训练忽略了多步预测中误差传播的影响

受 DeepSeek-R1 等推理模型通过 RLVR 成功提升数学和代码能力的启发，作者提出将 RLVR 范式推广到世界模型：用规则化的可验证奖励替代学习式奖励模型，直接优化预测指标。

## 方法详解

### 整体框架

RLVR-World 将不同模态的世界模型统一到自回归序列建模框架下。核心思路分三步：
1. 将当前状态和动作编码为"问题" token 序列 $q(s,a)$，将下一状态编码为"回答" token 序列 $o(s')$
2. 先用 MLE 预训练世界模型
3. 使用 RLVR 后训练，以预测指标作为可验证奖励微调

### 关键设计

1. **统一序列建模**：无论是文本、视频还是传感器数据，都通过模态特定的 tokenization 转化为 token 序列。语言用 BPE，图像/视频用离散视觉 tokenizer（iVideoGPT 的压缩 tokenizer），低维连续值用均匀分箱。这种统一使得 RLVR 可以跨模态通用。

2. **预测指标作为可验证奖励**：给定输入 $q(s,a)$，模型生成一组样本 $\{o_i\}_{i=1}^G$，解码出预测状态 $\hat{s}_i'$，通过与真值 $s'$ 比较计算奖励：
    $R_i = \text{sign}(D) \cdot D(\hat{s}_i', s')$
   其中 $\text{sign}(D) = -1$ 表示越低越好的指标（如 MSE、LPIPS），$\text{sign}(D) = 1$ 反之。这种设计的核心优势在于奖励是完全可验证的、无需人工标注。

3. **GRPO 优化算法**：采用群体相对策略优化（GRPO），不需要独立的价值函数。给定问题 $q$，采样一组回答 $\{o_i\}_{i=1}^G$，组内归一化计算优势：
    $\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$
   配合裁剪目标和 KL 散度惩罚进行策略更新。

### 损失函数 / 训练策略

- **预训练阶段**：标准 MLE 目标 $\mathcal{J}_{\text{MLE}}(\theta) = \sum_{t} \log p_\theta(o_t(s') | q(s,a), o_{<t}(s'))$
- **RLVR 后训练**：GRPO 目标函数包含裁剪比率项和 KL 正则化
- **语言世界模型**：先 SFT 再 RLVR，使用二元准确率奖励或任务特定奖励
- **视频世界模型**：奖励定义为 L1 损失 + LPIPS 的负值之和：$R = -\sum_\tau [L_1(\hat{s}_\tau, s_\tau) + \text{LPIPS}(\hat{s}_\tau, s_\tau)]$

## 实验关键数据

### 主实验

**文本游戏状态预测（ByteSized32）**

| 模型 | Unchanged Acc | Changed Acc | Overall Acc |
|------|:---:|:---:|:---:|
| Base (1.5B) | 11.98% | 0.08% | 7.11% |
| SFT | 38.88% | 24.21% | 32.87% |
| **RLVR-World (binary)** | 73.57% | 33.14% | 57.01% |
| **RLVR-World (task-specific)** | **83.66%** | **33.80%** | **63.24%** |
| GPT-4 | 73.90% | 51.60% | 64.76% |

**视频世界模型：RT-1 多步预测**

| 模型 | Repetition Rate↓ | MSE↓ | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| Base | 48.6% | 0.659 | 23.1 | 80.9 | 14.8 |
| Base (w/ rep. rejection) | 0.0% | 0.593 | 23.3 | 81.0 | 14.4 |
| **RLVR-World** | **9.9%** | **0.486** | **24.1** | **82.4** | **13.4** |
| 相对提升 Δ | +79.6% | +26.1% | +4.5% | +1.9% | +9.2% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 不同指标作为奖励 | 各指标最佳对应自身 | 用 LPIPS 训练在 LPIPS 上最好，MSE 训练在 MSE 上最好 |
| GRPO group size=2→16 | 收敛速度和最终性能持续提升 | 增大组大小提供更好的探索空间 |
| 加入重复惩罚奖励 | 重复率 0%，LPIPS=13.7 | 可在消除重复的同时保持预测质量 |
| test-time scaling | RLVR 单次采样 > Base best-of-5 | 但 N=100 时 Base 追上 RLVR |

### 关键发现

- RLVR 仅需数百步梯度更新即可获得显著提升，而 MLE 需数十万步
- RLVR 有效缓解了视频世界模型的重复帧问题（重复率从 48.6% 降至 9.9%）
- 强化后的世界模型在下游的 model-predictive control（网页导航）和策略评估（机器人操作）中都带来性能提升

## 亮点与洞察

- **RLVR 作为通用后训练范式**的理念非常有前瞻性：不仅适用于推理模型，可以推广到所有有可验证度量的生成模型
- 将"世界模型"和"推理模型"做了巧妙类比：两者都需要从代理目标转向任务对齐的直接优化
- 用 iVideoGPT 的压缩 tokenizer 解决视频序列长度爆炸问题，使 GRPO 在视频模态上可行
- Real2Sim 策略评估实验展示了实际应用价值

## 局限与展望

- RLVR 训练通常几百步就收敛，性能天花板受限于基础模型能力
- test-time scaling 存在上限：N 增大时 base model 可以追上 RLVR
- 当前视频世界模型在特定数据集上训练，尚未验证 OOD 泛化能力
- 奖励设计依赖传统视觉度量（MSE/LPIPS），未融入物理规则或时序一致性约束

## 相关工作与启发

- 与 DeepSeek-R1 的 RLVR 思路一脉相承，但拓展到了生成模型领域
- 与 DPO/RLHF 用于扩散模型微调的工作互补，RLVR 的优势在于不需要学习奖励模型
- 为未来通用世界模型（如 Cosmos）的后训练提供了可行范式

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将 RLVR 从推理模型推广到世界模型，跨语言和视频两种模态验证，思路新颖且有影响力
- **实验充分度**: ⭐⭐⭐⭐ 涵盖文本游戏、网页导航、机器人操作等多种场景，消融充分，但缺少更大规模基础模型的实验
- **写作质量**: ⭐⭐⭐⭐⭐ 论文结构清晰，动机 articulation 精准，图表设计优秀
- **价值**: ⭐⭐⭐⭐⭐ 提出的通用范式可广泛应用于各类生成模型的后训练优化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[ICCV 2025\] Aether: Geometric-Aware Unified World Modeling](../../ICCV2025/image_generation/aether_geometric-aware_unified_world_modeling.md)
- [\[ECCV 2024\] Controlling the World by Sleight of Hand](../../ECCV2024/image_generation/controlling_the_world_by_sleight_of_hand.md)
- [\[NeurIPS 2025\] Towards Robust Zero-Shot Reinforcement Learning](towards_robust_zero-shot_reinforcement_learning.md)
- [\[CVPR 2025\] MirrorVerse: Pushing Diffusion Models to Realistically Reflect the World](../../CVPR2025/image_generation/mirrorverse_pushing_diffusion_models_to_realistically_reflect_the_world.md)

</div>

<!-- RELATED:END -->
