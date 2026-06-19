---
title: >-
  [论文解读] Not All Parameters Matter: Masking Diffusion Models for Enhancing Generation Ability
description: >-
  [CVPR 2025][图像生成][参数掩码] MaskUNet 发现扩散模型中"将某些 U-Net 参数置零反而能提升生成质量"这一反直觉现象，提出基于时间步和样本内容的可学习二值掩码动态选择参数，COCO 2014 FID 从 12.85 降至 11.72（+8.8%），T2I-CompBench 颜色绑定从 0.375 提升至 0.699。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "参数掩码"
  - "扩散模型"
  - "Gumbel-Sigmoid"
  - "语义绑定"
  - "即插即用"
---

# Not All Parameters Matter: Masking Diffusion Models for Enhancing Generation Ability

**会议**: CVPR 2025  
**arXiv**: [2505.03097](https://arxiv.org/abs/2505.03097)  
**代码**: [https://gudaochangsheng.github.io/MaskUnet-Page/](https://gudaochangsheng.github.io/MaskUnet-Page/)  
**领域**: 图像生成  
**关键词**: 参数掩码、扩散模型、Gumbel-Sigmoid、语义绑定、即插即用

## 一句话总结

MaskUNet 发现扩散模型中"将某些 U-Net 参数置零反而能提升生成质量"这一反直觉现象，提出基于时间步和样本内容的可学习二值掩码动态选择参数，COCO 2014 FID 从 12.85 降至 11.72（+8.8%），T2I-CompBench 颜色绑定从 0.375 提升至 0.699。

## 研究背景与动机

1. **领域现状**：Stable Diffusion 在文本-图像生成中广泛使用。现有微调方法（LoRA、全量微调）通过修改参数权重提升效果，但可能损害预训练模型的泛化性。
2. **现有痛点**：(1) 全量微调容易过拟合且破坏预训练知识；(2) LoRA 等参数高效方法效果有限；(3) 扩散模型的不同时间步和不同内容可能需要不同的参数子集——一刀切的参数更新不够灵活。
3. **核心矛盾**：预训练模型的参数包含了大量通用知识，但也包含对特定生成任务有害的"噪声参数"——如何在不学习新知识的情况下，仅通过"选择性遗忘"来提升生成质量？
4. **本文目标**：通过学习哪些参数应该被激活、哪些应该被置零来提升生成质量。
5. **切入角度**：观察到随机置零 U-Net 的部分参数（即使是大权重参数）有时反而提升了生成效果——说明参数选择本身就是一种优化手段。
6. **核心 idea**：基于时间步嵌入和样本全局特征的 Gumbel-Sigmoid 二值掩码。

## 方法详解

### 整体框架

输入（噪声潜码 $z$ + 时间步 $t$）→ MLP 生成掩码特征 $z' = \text{FC}(t_{emb}) + \text{GAP}(z)$ → 4 层 MLP → Gumbel-Sigmoid 生成二值掩码 $m$ → U-Net 权重逐元素掩码 $\hat{w} = m' \odot w$ → 正常扩散去噪。

### 关键设计

1. **时间步+样本依赖的掩码生成**

    - 功能：根据当前去噪步骤和图像内容动态选择参数
    - 核心思路：时间步嵌入 $t_{emb}$ 通过 FC 层 + 潜码的全局平均池化 $\text{GAP}(z)$ 融合后经 4 层 MLP 生成 logits，Gumbel-Sigmoid 二值化
    - 设计动机：消融显示去掉时间步 FID 从 21.88 升至 22.30，去掉样本升至 22.14——两者都提供了有用的条件信号

2. **Gumbel-Sigmoid 二值掩码**

    - 功能：在训练时保持可微分，推理时生成硬二值决策
    - 核心思路：$m = \sigma((\hat z + g) / \tau)$，其中 $g$ 为 Gumbel 噪声，$\tau$ 为温度参数。训练时 $\tau > 0$ 使梯度流通，推理时 $\tau \to 0$ 得到 0/1 掩码
    - 设计动机：直接用 argmax 不可微，Gumbel 技巧是标准的解决方案

3. **免训练版本（基于奖励模型）**

    - 功能：不需要额外训练，直接用奖励模型指导掩码优化
    - 核心思路：$\mathcal{L}_{reward} = \sum_i \omega_i \Psi_i(x_0', c)$，使用 ImageReward + HPSv2 作为优化目标
    - 设计动机：训练版需要数据集微调，免训练版更灵活但效果稍弱

### 损失函数 / 训练策略

训练版：标准扩散去噪损失。免训练版：奖励模型损失。仅 MLP 掩码生成器的参数可训练。

## 实验关键数据

### 主实验

| 方法 | COCO 2014 FID↓ | COCO 2017 FID↓ | T2I Color↑ | GenEval Overall↑ |
|------|---------------|---------------|-----------|-----------------|
| SD 1.5 | 12.85 | 23.39 | 0.375 | 0.39 |
| LoRA | 12.82 | 23.18 | - | - |
| Full FT | 14.06 | 24.45 | - | - |
| SynGen | - | - | 0.629 | 0.43 |
| **SynGen+MaskUNet** | - | - | **0.699** | **0.50** |
| **MaskUNet** | **11.72** | **21.88** | - | - |

### 消融实验

| 条件 | COCO 2017 FID↓ | 说明 |
|------|---------------|------|
| 完整（时间步+样本） | **21.88** | 最优 |
| w/o 时间步 | 22.30 | 时间步信息重要 |
| w/o 样本 | 22.14 | 样本信息也重要 |
| SD 1.5 基线 | 23.39 | - |

### 关键发现

- 全量微调 FID 反而升高（12.85→14.06），证明修改参数权值的危险性——MaskUNet 仅选择参数避免了这个问题
- 与 SynGen 叠加后颜色绑定从 0.629 升至 0.699（+11%），说明 MaskUNet 与现有方法正交互补
- 在 DreamBooth、Textual Inversion、Text2Video-Zero 等下游任务上也有效

## 亮点与洞察

- **"减法大于加法"的发现**：选择性置零参数比修改参数更有效——挑战了"微调=改参数"的默认假设
- **即插即用+可叠加**：不修改模型权重，可以与任何现有方法（SynGen、DreamBooth 等）叠加
- **反直觉的核心洞察**：U-Net 的大参数可能是有害的——这为理解扩散模型参数提供了新视角

## 局限与展望

- 不能学习新知识——只能选择现有参数，对全新概念无效
- 推理时有掩码生成的额外开销
- 仅验证了 U-Net 架构，DiT 等新架构未测试
- 后续可结合 LoRA 实现"选择+修改"的双重优化

## 相关工作与启发

- **vs LoRA**: LoRA 修改参数子空间，MaskUNet 选择参数子集——两者正交
- **vs SynGen**: SynGen 修改注意力机制，MaskUNet 修改参数选择——叠加后效果更好

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "参数选择替代参数修改"是全新视角
- 实验充分度: ⭐⭐⭐⭐⭐ FID+T2I-CompBench+GenEval+多下游任务+消融
- 写作质量: ⭐⭐⭐⭐ 洞察清晰
- 价值: ⭐⭐⭐⭐⭐ 对扩散模型优化范式的重要补充

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Enhancing Creative Generation on Stable Diffusion-based Models](enhancing_creative_generation_on_stable_diffusion-based_models.md)
- [\[CVPR 2025\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)
- [\[CVPR 2025\] Not Just Text: Uncovering Vision Modality Typographic Threats in Image Generation Models](not_just_text_uncovering_vision_modality_typographic_threats_in_image_generation.md)
- [\[CVPR 2025\] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)
- [\[CVPR 2025\] Enhancing Facial Privacy Protection via Weakening Diffusion Purification](enhancing_facial_privacy_protection_via_weakening_diffusion_purification.md)

</div>

<!-- RELATED:END -->
