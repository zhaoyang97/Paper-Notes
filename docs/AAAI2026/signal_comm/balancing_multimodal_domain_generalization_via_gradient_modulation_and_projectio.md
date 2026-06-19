---
title: >-
  [论文解读] Balancing Multimodal Domain Generalization via Gradient Modulation and Projection
description: >-
  [AAAI2026 Oral][信号/通信][多模态] 提出 Gradient Modulation Projection (GMP) 策略，通过解耦分类与域不变梯度的调制（IGDM）以及冲突自适应梯度投影（CAGP），解决多模态域泛化中模态间优化不平衡和任务间梯度冲突问题，在多个基准上达到 SOTA。
tags:
  - "AAAI2026 Oral"
  - "信号/通信"
  - "多模态"
  - "Gradient Modulation"
  - "Gradient Projection"
  - "Optimization Imbalance"
---

# Balancing Multimodal Domain Generalization via Gradient Modulation and Projection

**会议**: AAAI2026 Oral  
**arXiv**: [2603.14175](https://arxiv.org/abs/2603.14175)  
**代码**: 待确认  
**领域**: 视频理解  
**关键词**: Multimodal Domain Generalization, Gradient Modulation, Gradient Projection, Optimization Imbalance  

## 一句话总结

提出 Gradient Modulation Projection (GMP) 策略，通过解耦分类与域不变梯度的调制（IGDM）以及冲突自适应梯度投影（CAGP），解决多模态域泛化中模态间优化不平衡和任务间梯度冲突问题，在多个基准上达到 SOTA。

## 背景与动机

多模态域泛化（MMDG）旨在利用视频、音频等多模态信息的互补优势，让模型能泛化到训练时未见过的域。在真实应用场景（如跨环境动作识别、视听事件检测）中，测试数据往往来自不同设备、不同环境，有效的域泛化至关重要。

然而，多模态学习中普遍存在**优化不平衡**问题：不同模态在训练过程中收敛速度不同，导致梯度贡献不均——某些模态主导了学习过程，而其他模态则被抑制。实验表明（Table 1），联合训练中各单模态分支的表现远不如独立训练的对应模型，说明现有 MMDG 训练策略未能充分利用各模态的能力。

更关键的是，现有平衡策略（如 OGM-GE、Grad-Blending）仅基于**源域分类性能**来调节各模态的梯度贡献。这忽略了一个核心洞察：在源域上分类能力强的模态，未必能学到良好的域不变特征，因此在目标域上的泛化效果可能很差。Table 1 实验清楚地展示了这一点——传统方法在源域上提升明显，但在目标域上增益甚微。

## 核心问题

本文识别并解决了 MMDG 中的两类不平衡问题：

1. **模态间不平衡（Inter-Modality Imbalance）**：不同模态的梯度幅度差异持续存在，导致强模态主导优化，弱模态长期欠优化。传统方法仅依据分类梯度大小来平衡，忽略了域不变性目标，可能压制对跨域泛化至关重要的模态。

2. **任务间冲突（Inter-Task Conflicts）**：分类损失梯度 $g_c^m$ 和域对抗损失梯度 $g_d^m$ 经常指向相反方向（余弦相似度为负），形成梯度冲突。这种冲突在不同模态中程度不同（如视频模态冲突严重、音频模态冲突较小），统一的冲突解决策略无法适应这种模态特异性差异。

## 方法详解

### 整体框架：GMP

GMP 包含两个核心组件——IGDM 和 CAGP，分别处理模态间不平衡和任务间冲突。

### 组件一：Inter-Modality Gradient Decoupled Modulation (IGDM)

IGDM 的核心思路是**解耦调制**——将分类梯度和域不变梯度分别调制，而非统一缩放。具体步骤如下：

**Step 1: 计算双置信度指标**

- **语义置信度（Semantic Confidence）** $q_i^m$：度量模态 $m$ 对样本 $i$ 的分类确定性，取自分类器的 softmax 输出中真实类别的概率。
- **域置信度（Domain Confidence）** $c_i^m$：度量模态 $m$ 对域判别的确定性，取自域判别器输出中真实域标签的概率。域置信度越低，表示该模态学到的特征越具域不变性。

**Step 2: 计算差异比**

在每个 mini-batch 上，计算两个模态之间的比值：

- $\rho_t^m$：语义置信度比，$\rho_t^m > 1$ 表示模态 $m$ 在分类上更强。
- $\sigma_t^m$：域置信度比，$\sigma_t^m > 1$ 表示模态 $m$ 在域不变性上更强。

**Step 3: 解耦调制系数**

- 分类梯度调制系数 $k_t^m = 1 - \tanh(\alpha_k \cdot \rho_t^m)$（当 $\rho_t^m > 1$ 时），用于抑制分类过强模态的分类梯度。
- 域梯度调制系数 $p_t^m = 1 - \tanh(\alpha_p \cdot \sigma_t^m)$（当 $\sigma_t^m > 1$ 时），用于抑制域不变性过强模态的域梯度。

这样，分类梯度和域梯度被**独立**调制，而非用同一个系数统一缩放，实现了更精细的控制。

### 组件二：Conflict-Adaptive Gradient Projection (CAGP)

CAGP 在调制后的梯度上处理任务间冲突，核心设计有三点：

1. **冲突感知**：仅在 $\hat{g}_c^m \cdot \hat{g}_d^m < 0$ 时才触发投影，否则保持原始梯度。
2. **模态特异**：对每个模态独立判断和投影。
3. **弱任务保护**：利用相对任务强度比 $\Gamma_t^m = \rho_t^m / \sigma_t^m$ 判断哪个任务更强。当 $\Gamma_t^m > 1$（分类更强）时，将分类梯度投影到域梯度的正交方向；当 $\Gamma_t^m < 1$（域不变性更强）时，将域梯度投影到分类梯度的正交方向。始终保留弱任务梯度的完整方向，仅移除强任务梯度中的冲突分量。

## 实验关键数据

### 基准数据集

- **EPIC-Kitchens**：厨房动作识别数据集，视频+音频
- **HAC**：视听数据集，视频+音频

### 与现有梯度策略比较（Table 2）

| 方法 | EPIC-Kitchens | HAC |
|------|:---:|:---:|
| Base（拼接融合） | 55.06 | 61.86 |
| OGM-GE | 55.71 | 62.83 |
| Grad-Blending | 55.49 | 62.66 |
| **GMP（本文）** | **57.36** | **64.91** |

GMP 在 EPIC-Kitchens 上超过最优基线 +1.65%，HAC 上超过 +2.08%。

### 与 MMDG 方法集成（Table 3）

GMP 作为即插即用模块集成到 RNA-Net、MOOSA、SimMMDG、CMRF 后均有提升，其中 SimMMDG+GMP 在两个数据集上分别达到 62.03% 和 69.11%。

### 单模态泛化提升

视频分支从 48.86% 提升到 52.33%（+3.47%），音频分支从 34.15% 提升到 35.88%（+1.73%）。传统联合训练导致视频分支比独立训练低 6.12%，GMP 将这一差距缩小到 2.65%。

### 消融实验（Table 4）

- IGDM 单独使用：EPIC 55.98%，HAC 63.05%
- CAGP 单独使用：EPIC 55.34%，HAC 63.41%
- 两者结合（Full）：EPIC 57.36%，HAC 64.91%，证明两个组件互补
- 统一调制替换解耦调制：性能下降至 54.97%/62.50%
- 去掉任一置信度（$k_t^m$ 或 $p_t^m$）：性能均明显下降
- 固定投影方向或使用 PCGrad：均不如自适应的 CAGP

## 亮点

- **首次从优化视角分析 MMDG**，指出传统平衡策略在 MMDG 场景下失效的根本原因（仅关注分类忽略泛化）
- **解耦调制设计精巧**：用语义置信度和域置信度分别调制两类梯度，比统一调制更细粒度
- **弱任务保护的梯度投影**符合直觉——在两个目标冲突时，优先保全较弱目标的学习进度
- **即插即用的通用性**：GMP 可无缝集成到多种现有 MMDG 方法中，均带来增益
- 消融实验充分，理论分析清晰，t-SNE 可视化直观展示了效果

## 局限与展望

- 仅在视频+音频两模态上验证，缺少对更多模态（如文本、IMU、深度图等）的验证
- 超参数 $\alpha_k$、$\alpha_p$ 需在 [0,1] 上调参，在不同数据集间的敏感性需关注
- 域判别器的质量直接影响域置信度的可靠性，若域判别器训练不稳定可能导致 IGDM 效果下降
- 数据集规模相对较小（EPIC-Kitchens、HAC），在大规模数据集上的表现有待验证
- 梯度投影操作增加了额外的计算开销，虽然论文未讨论效率影响

## 与相关工作的对比

| 对比维度 | 传统方法（OGM-GE 等） | GMP（本文） |
|:--|:--|:--|
| 平衡依据 | 仅基于分类性能 | 同时考虑分类和域不变性 |
| 梯度调制 | 统一调制 | 解耦调制（分类/域梯度分别调制） |
| 冲突处理 | 无 / PCGrad 统一处理 | 自适应投影 + 弱任务保护 |
| 目标域表现 | 提升有限（+0.43%~+0.65%） | 显著提升（+2.30%） |
| 通用性 | 独立方法 | 可作为插件集成到现有 MMDG 方法 |

与 SimMMDG、MOOSA 等注重架构/表示的 MMDG 方法正交互补，GMP 从优化层面提供增益。

## 启发与关联

- **多目标优化视角在多模态学习中的应用**是一个有潜力的方向，本文的解耦思路可推广到其他多任务多模态场景
- 弱任务保护的投影思路与 PCGrad 等多任务学习方法相关，但针对 MMDG 场景的任务强度自适应是新颖的
- 域置信度这一指标的设计思路，可以启发其他需要衡量域不变性质量的工作
- 对于 video understanding 领域，如何在多模态联合训练中避免模态坍塌/抑制，本文提供了一种梯度层面的解决范式

## 评分
- 新颖性: ⭐⭐⭐⭐ (首次从优化视角切入 MMDG，解耦调制 + 自适应投影设计新颖)
- 实验充分度: ⭐⭐⭐⭐ (消融完整、多基线比较、集成实验、可视化，但数据集较少)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，动机论证充分)
- 价值: ⭐⭐⭐⭐ (即插即用的通用性强，方向有意义)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)
- [\[CVPR 2026\] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](../../CVPR2026/signal_comm/clay_conditional_visual_similarity.md)
- [\[ICML 2025\] Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization](../../ICML2025/signal_comm/fourier_position_embedding_enhancing_attentions_periodic_extension_for_length_ge.md)
- [\[CVPR 2025\] ABC-Former: Auxiliary Bimodal Cross-domain Transformer with Interactive Channel Attention](../../CVPR2025/signal_comm/abc-former_auxiliary_bimodal_cross-domain_transformer_with_interactive_channel_a.md)

</div>

<!-- RELATED:END -->
