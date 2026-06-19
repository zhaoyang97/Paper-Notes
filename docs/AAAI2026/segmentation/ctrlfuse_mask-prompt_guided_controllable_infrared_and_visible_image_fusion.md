---
title: >-
  [论文解读] CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion
description: >-
  [AAAI2026][语义分割][infrared-visible image fusion] 提出 CtrlFuse，通过 mask prompt 引导 SAM 微调，实现红外-可见光图像的交互式可控融合，在融合质量和下游分割/检测任务上同时取得提升。 红外-可见光图像融合旨在结合两种模态的互补信息，为智能无人系统提供全天…
tags:
  - "AAAI2026"
  - "语义分割"
  - "infrared-visible image fusion"
  - "controllable fusion"
  - "提示学习"
  - "SAM"
  - "图像分割"
---

# CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion

**会议**: AAAI2026  
**arXiv**: [2601.08619](https://arxiv.org/abs/2601.08619)  
**代码**: [Sevryy/CtrlFuse](https://github.com/Sevryy/CtrlFuse)  
**领域**: 图像分割  
**关键词**: infrared-visible image fusion, controllable fusion, mask prompt, SAM, semantic segmentation

## 一句话总结

提出 CtrlFuse，通过 mask prompt 引导 SAM 微调，实现红外-可见光图像的交互式可控融合，在融合质量和下游分割/检测任务上同时取得提升。

## 背景与动机

红外-可见光图像融合旨在结合两种模态的互补信息，为智能无人系统提供全天候感知能力。可见光图像提供丰富颜色和高分辨率，但弱光条件下性能下降；红外图像可补偿暗光不足但缺乏纹理信息。

现有方法存在两个核心缺陷：

1. **像素级融合方法**只关注融合图像与源图像的像素一致性，忽略了融合图像对下游感知任务的适配性
2. **任务驱动融合方法**通过级联检测/分割模型隐式学习固定语义类别，无法根据不同应用需求动态控制对特定目标的关注

例如，现有方法虽然在训练中学习了目标语义，但在实际车辆分割场景中仍然表现不佳。这表明需要一种语义可控的多模态融合架构，能够根据不同语义需求进行动态可控融合。

## 核心问题

如何构建一个交互式可控的多模态图像融合框架，使用户能够通过 mask prompt 动态指定感兴趣的语义目标，同时实现融合质量与下游任务性能的相互促进？

## 方法详解

### 整体架构

CtrlFuse 包含四个核心组件：

- **多模态骨干编码器-解码器**：分别提取红外特征 $F_{ir}$ 和可见光特征 $F_{vis}$，拼接后通过解码器生成参考图像 $I_{ref}$
- **Reference Prompt Encoder (RPE)**：在 mask 引导下动态编码任务相关的语义 prompt
- **Prompt-Semantic Fusion Module (PSFM)**：将语义 prompt 显式注入融合特征
- **冻结的 SAM**：提供强大的语义感知基础能力

### Reference Prompt Encoder

以红外分支为例：

1. 将 mask prompt 与红外特征 $F_{ir}$ 做 Hadamard 积后平均池化，得到目标特征 $F_t$
2. 将 $F_{ir}$ 和 $F_{ref}$ 分别与 $F_t$ 拼接后卷积，生成 support 特征 $F_{supp}$ 和 query 特征 $F_{qry}$
3. 使用可学习 queries $Q \in \mathbb{R}^{N \times C}$（$N=40$），通过交叉注意力从 $F_{supp}$ 提取类别相关信息得到 $Q'$
4. $Q'$ 再与 $F_{qry}$ 交叉注意力生成参考 prompt $P'$，经冻结的 SAM Prompt Encoder 生成最终 prompt embedding $P$

$$Q' = \text{SelfAttn}_1(\text{CrossAttn}_1(Q, F_{supp}))$$
$$P' = \text{SelfAttn}_2(\text{CrossAttn}_2(Q', F_{qry}))$$

### Prompt-Semantic Fusion Module

1. 对编码特征 $F$ 下采样后展平为序列 $F_{seq}$
2. $F_{seq}$ 与 prompt embedding $P$ 通过交叉注意力机制融合
3. 恢复空间维度并上采样，与 SAM 分割 mask $M$ 逐元素相乘，得到类别增强特征 $F^p$

$$F^p = M \cdot \text{Up}(\text{View}(\text{CrossAttn}(F_{seq}, P)))$$

最终融合特征由初步融合特征 $F_{ref}$ 与红外和可见光的 prompt 特征 $F_{ir}^p$、$F_{vis}^p$ 逐元素相加得到，输入解码器生成最终融合图像。

### 训练策略

端到端训练，同时优化融合损失 $\mathcal{L}_{fusion}$ 和分割损失 $\mathcal{L}_{seg}$。

## 实验关键数据

### 融合质量（三个数据集）

| 数据集 | 最优指标 | 具体表现 |
|--------|---------|---------|
| FMB | PSNR/Q_abf/N_abf 最优 | PSNR=63.292, Q_abf=0.719 |
| DroneVehicle | MSE/PSNR/SCD 最优 | PSNR=60.317, SCD=1.552 |
| MSRS | PSNR/N_abf 最优 | PSNR=64.75, N_abf=0.018 |

### 语义分割（MSRS）

- mIoU=0.7963，8种方法中最优
- 在 Car、Curve、Guardrail、Color Tone 四个类别上最优

### 目标检测（DroneVehicle）

- AP@\[0.5:0.95\] 总体=0.525，最优
- car 类别 AP=0.651，bus 类别 AP=0.521，均为最优

### 消融实验（MSRS）

| 配置 | SSIM | SCD | 结论 |
|------|------|-----|------|
| w/o Prompt | 0.933 | 1.635 | prompt 对结构保持至关重要 |
| w/o Seg | 0.939 | 1.636 | 分割分支有助于融合质量 |
| w/o Vis | 0.915 | 1.681 | 可见光分支不可或缺 |
| w/o Ir | 0.938 | 1.622 | 红外分支不可或缺 |
| Exchange SQ | 0.924 | 1.659 | 原始 support/query 设计更优 |
| **完整模型** | **0.969** | **1.726** | 各组件协同提升 |

## 亮点

1. **交互式可控融合**：首次在红外-可见光融合中引入 mask prompt 实现交互式动态融合，用户可以指定关注目标
2. **融合-分割协同增强**：联合优化使融合质量和分割性能相互促进，微调后的 SAM 分支甚至超越原始 SAM 模型的分割效果
3. **对 prompt 质量鲁棒**：即使 mask 不完整或质量较低（只标注部分目标），融合结果仍能有效突出目标区域
4. **通用 prompt 来源**：可直接使用 Grounded-SAM 从文本生成 mask prompt，无需标注数据即可在新数据集上实现可控融合

## 局限与展望

1. **依赖 mask prompt 输入**：需要额外的 mask 作为引导，增加了使用复杂度；自动化 prompt 生成管线（如文本到 mask）的质量直接影响最终效果
2. **冻结 SAM 的瓶颈**：SAM image encoder 和 mask decoder 均冻结，对红外模态的适应能力有限，可考虑轻量级 adapter 微调
3. **分类定位有误差**：论文被归类为 3d_vision，实际属于 image fusion / multimodal perception 领域
4. **仅支持灰度融合输出**：最终融合图像为单通道 $I_{\mathcal{F}} \in \mathbb{R}^{1 \times H \times W}$，丢弃了可见光的颜色信息
5. **计算开销未详述**：使用 SAM 作为辅助网络的推理速度和显存占用未充分讨论

## 与相关工作的对比

| 方法 | 特点 | 局限 |
|------|------|------|
| SeAFusion | 分割驱动，联合优化 | 固定语义类别，不可控 |
| PSFusion | 高层视觉任务驱动 | 隐式语义学习，无交互 |
| SDCFusion | 分割驱动+深度分解 | 仍然受限于预定义类别 |
| LDFusion | CLIP 文本引导 | 文本语义粗糙，难以精细控制 |
| **CtrlFuse** | **mask prompt + SAM 微调** | **显式语义注入，交互可控** |

CtrlFuse 与现有任务驱动方法的本质区别在于：从"固定类别隐式语义学习"转变为"mask prompt 引导的显式可控语义注入"，借助 SAM 的强大零样本泛化能力实现对任意语义目标的动态融合。

## 启发与关联

1. **Prompt Tuning 范式的迁移**：将 NLP/视觉大模型中的 prompt tuning 思想引入底层图像融合任务，这种"基础模型 + prompt"的范式可推广到其他底层视觉任务（去噪、超分等）
2. **任务协同优化**：融合和分割的相互促进表明，多任务联合训练中任务间的正向迁移效应值得深入挖掘
3. **可控性作为评价维度**：除传统像素指标外，可控性应成为多模态融合方法的重要评价维度

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次在红外-可见光融合中实现基于 mask prompt 的交互可控融合
- 实验充分度: ⭐⭐⭐⭐ — 三个数据集，融合/分割/检测三类任务，消融完整
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，方法描述详细，图表丰富
- 价值: ⭐⭐⭐⭐ — 为多模态融合引入可控范式，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] HCLFuse: Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws](../../NeurIPS2025/segmentation/revisiting_generative_infrared_and_visible_image_fusion_based_on_human_cognitive.md)
- [\[CVPR 2026\] PR-MaGIC: Prompt Refinement Via Mask Decoder Gradient Flow For In-Context Segmentation](../../CVPR2026/segmentation/pr-magic_prompt_refinement_via_mask_decoder_gradient_flow_for_in-context_segment.md)
- [\[AAAI 2026\] Text-guided Controllable Diffusion for Realistic Camouflage Images Generation](text-guided_controllable_diffusion_for_realistic_camouflage_images_generation.md)
- [\[AAAI 2026\] Empowering DINO Representations for Underwater Instance Segmentation via Aligner and Prompter](empowering_dino_representations_for_underwater_instance_segmentation_via_aligner.md)
- [\[CVPR 2025\] Task-driven Image Fusion with Learnable Fusion Loss](../../CVPR2025/segmentation/task-driven_image_fusion_with_learnable_fusion_loss.md)

</div>

<!-- RELATED:END -->
