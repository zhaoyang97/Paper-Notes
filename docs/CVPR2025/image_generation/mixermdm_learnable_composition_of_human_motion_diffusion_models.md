---
title: >-
  [论文解读] MixerMDM: Learnable Composition of Human Motion Diffusion Models
description: >-
  [CVPR 2025][图像生成][扩散模型] 提出 MixerMDM，首个可学习的运动扩散模型组合技术，通过 Transformer-based Mixer 模块预测动态混合权重，以对抗训练方式学习如何融合个体运动和交互运动扩散模型，实现细粒度可控的人-人交互运动生成。 领域现状： 文本驱动的人体运动生成已取得显著进展…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "扩散模型"
  - "model composition"
  - "adversarial training"
  - "dynamic mixing"
  - "human-human interaction"
---

# MixerMDM: Learnable Composition of Human Motion Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2504.01019](https://arxiv.org/abs/2504.01019)  
**代码**: [项目主页](https://pabloruizponce.com/papers/MixerMDM)  
**领域**: image_generation (motion generation)  
**关键词**: human motion diffusion, model composition, adversarial training, dynamic mixing, human-human interaction

## 一句话总结

提出 MixerMDM，首个可学习的运动扩散模型组合技术，通过 Transformer-based Mixer 模块预测动态混合权重，以对抗训练方式学习如何融合个体运动和交互运动扩散模型，实现细粒度可控的人-人交互运动生成。

## 研究背景与动机

**领域现状**: 文本驱动的人体运动生成已取得显著进展，但高质量运动数据稀缺且格式不统一。现有方法倾向于在专门数据集上训练特化模型（如个体运动模型、交互运动模型），各自有其优势领域。

**核心矛盾**: 如何组合不同专化模型的生成能力？个体运动模型擅长个人内部动作（intrapersonal）的多样性和精确控制，交互模型擅长人际间（interpersonal）的全局轨迹和朝向。现有模型组合方法（DiffusionBlending、DualMDM）依赖手动设定的固定或预定义权重调度器，无法针对不同运动和条件动态调整混合策略。

**本文切入点**: 设计一个可学习的 Mixer 模块，能根据当前生成的运动内容、文本条件和去噪步骤动态预测混合权重，以对抗训练方式保持各预训练模型的核心特性。

## 方法详解

### 整体框架

MixerMDM 的 pipeline：在去噪过程的每个时间步 $t$：
1. 两个预训练模型 $\mathcal{M}^a$（交互模型）和 $\mathcal{M}^b$（个体模型）分别生成运动 $x_t^a$ 和 $x_t^b$
2. Mixer 模块接收两个运动和各自条件，预测混合权重 $w_t$
3. 通过混合公式融合两个运动得到 $x_t^m$
4. $x_t^m$ 作为下一步去噪的输入反馈给两个模型

对于个体+交互混合场景，还需要 centering（规范化起始位置）和 alignment（恢复全局轨迹）函数来匹配不同模型的输入格式。

### 关键设计

#### 1. Mixer 模块（核心架构）

基于 Transformer encoder 的轻量级模块（21M 参数，远小于预训练模型的 300M+）：

**输入**: 两个预训练模型的运动输出 $x_t^a, x_t^b$，各自条件 $c^a, c^b$，当前时间步 $t$

**处理**: 4 层多头注意力（latent dim=512, 8 heads），将输入编码为高维表示

**输出**: MLP 解码为混合权重 $w_t$，提供 4 种粒度变体：
- **Global [G]**: 一个全局标量
- **Temporal [T]**: 每帧一个权重
- **Spatial [S]**: 每个关节一个权重
- **Spatio-Temporal [ST]**: 每帧每关节一个权重（最精细）

混合公式：$x_t^m = x_t^a + w_t \cdot (x_t^b - x_t^a)$

#### 2. 对抗训练策略

由于混合运动没有 ground truth，设计基于 GAN 的对抗训练：

- **每个预训练模型一个判别器**: $\mathcal{D}^a$ 和 $\mathcal{D}^b$
- **正样本**: 各预训练模型自身的输出（$x_t^a$ 对 $\mathcal{D}^a$，$x_t^b$ 对 $\mathcal{D}^b$）
- **负样本**: MixerMDM 生成的混合运动 $x_t^m$
- **目标**: Mixer（生成器）让混合运动同时欺骗两个判别器，从而保持两个模型的核心特性

#### 3. 模块化设计

Mixer 直接使用预训练模型的最终输出进行混合，不依赖模型内部特征，因此可以无缝替换底层预训练模型（只要在同一数据集上训练过）。实验证明，将最差组合的预训练模型配合最佳 Mixer 权重，Overall Alignment 可提升 37%。

### 损失函数

**生成器损失**:
$$\mathcal{L}_{adv}^G = -\mathcal{D}^a(x_t^m) - \mathcal{D}^b(x_t^m) + L1$$

**判别器损失（hinge loss）**:
$$\mathcal{L}_{adv}^D = \min(0, -1-\mathcal{D}^a(x_t^m)) + \min(0, -1-\mathcal{D}^b(x_t^m)) + \min(0, -1+\mathcal{D}^a(x_t^a)) + \min(0, -1+\mathcal{D}^b(x_t^b)) + L1$$

$L1$ 正则项惩罚两个判别器损失的差距过大。训练 300 epochs，batch size=128，lr=$10^{-5}$，单卡 4090 约 36 小时。

## 实验关键数据

### 主实验表（模型组合对比）

| 方法 | 交互模型 | 个体模型 | Overall Alignment ↑ | Adaptability |
|------|---------|---------|---------------------|-------------|
| DiffusionBlending | in2IN | in2IN | 0.221 | — |
| DualMDM | in2IN | in2IN | 0.217 | — |
| **MixerMDM [ST]** | **in2IN** | **in2IN** | **0.335** | 0.112 |
| Finetuned baseline | — | — | 0.289 | — |

### 消融表（不同混合粒度）

| 类型 | $\mathcal{M}^a$=in2IN, $\mathcal{M}^b$=in2IN | Adaptability |
|------|------|-------------|
| Global [G] | 0.317 | 0.004 |
| Temporal [T] | 0.245 | 0.002 |
| Spatial [S] | 0.310 | 0.015 |
| **Spatio-Temporal [ST]** | **0.335** | 0.112 |

### 用户研究

| 方法 | Interaction Avg Rank ↓ | 1st Ranked ↑ | Individual Avg Rank ↓ | 1st Ranked ↑ |
|------|----------------------|-------------|----------------------|-------------|
| DiffusionBlending | 2.531 | 4.57% | 2.446 | 10.57% |
| DualMDM | 2.286 | 10.29% | 2.051 | 24.57% |
| **MixerMDM** | **1.182** | **85.14%** | **1.309** | **74.86%** |

### 关键发现

1. **ST 粒度最优**: Spatio-Temporal 混合权重在 Alignment 和 Adaptability 之间达到最佳平衡
2. **对抗训练有效**: 混合权重曲线与手动预设有显著差异，学习到的策略更优
3. **模块化迁移**: 最佳 Mixer 权重迁移到较差模型组合可带来 37% 的 Overall Alignment 提升
4. **去噪早期偏个体、晚期偏交互**: 所有学到的权重曲线都呈现此规律，验证了先前研究的假设

## 亮点与洞察

1. **首个可学习的运动扩散模型组合**: 相比手动设定的固定或预定义权重，动态学习混合策略是质的飞跃
2. **巧妙利用 GAN 范式**: 在没有混合运动 GT 的情况下，将各模型输出作为伪真值进行对抗训练
3. **模块化特性**: Mixer 不依赖模型内部表示，可零成本替换底层模型，具有很强的实用价值
4. **评估体系贡献**: 提出 Alignment 和 Adaptability 两个新指标填补了该任务定量评估的空白

## 局限性

1. **推理开销**: 虽然 Mixer 比预训练模型小约 10 倍，但每个去噪步都需计算混合权重，增加推理时间
2. **对抗训练不稳定**: GAN 训练的固有问题——超参敏感、训练不稳定
3. **数据表示限制**: 要求被混合的模型预测统一的数据表示格式，不同表示格式的模型需要重新训练
4. **当前仅验证了 2 模型组合**: 未探索 3 个及以上模型的多路混合

## 相关工作与启发

- **DiffusionBlending / DualMDM**: 前序工作使用固定/预定义权重，MixerMDM 将其推广为可学习的动态版本
- **MDM / InterGen / in2IN**: 作为底层预训练模型被组合使用
- **MultiDiffusion (图像域)**: 扩散模型路径融合的思想，MixerMDM 将其引入运动生成
- **启发**: 可学习的模型组合是一种通用范式，可推广到其他扩散模型应用（如图像/视频/音频的多模型融合）

## 评分 ⭐

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐ |
| 工程实用性 | ⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Nonisotropic Gaussian Diffusion for Realistic 3D Human Motion Prediction](nonisotropic_gaussian_diffusion_for_realistic_3d_human_motion_prediction.md)
- [\[CVPR 2025\] InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](interedit_navigating_text-guided_multi-human_3d_motion_editing.md)
- [\[CVPR 2025\] Move-in-2D: 2D-Conditioned Human Motion Generation](move-in-2d_2d-conditioned_human_motion_generation.md)
- [\[ECCV 2024\] Realistic Human Motion Generation with Cross-Diffusion Models](../../ECCV2024/image_generation/realistic_human_motion_generation_with_cross-diffusion_models.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)

</div>

<!-- RELATED:END -->
