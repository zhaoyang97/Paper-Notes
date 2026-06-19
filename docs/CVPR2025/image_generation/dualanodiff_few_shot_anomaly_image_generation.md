---
title: >-
  [论文解读] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation
description: >-
  [CVPR 2025][图像生成][异常图像生成] 提出 DualAnoDiff，利用双关联扩散模型同时生成整体异常图像和对应异常部分，解决了少样本场景下异常图像生成中多样性不足、融合不自然和掩码不对齐的问题，在下游异常检测任务中达到 SOTA。 领域现状：工业异常检测面临异常样本稀缺的问题，现有方法要么是无监督仅用正常样本…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "异常图像生成"
  - "少样本生成"
  - "双扩散模型"
  - "工业缺陷检测"
  - "图像-掩码对"
---

# DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation

**会议**: CVPR 2025  
**arXiv**: [2408.13509](https://arxiv.org/abs/2408.13509)  
**代码**: [https://github.com/yinyjin/DualAnoDiff](https://github.com/yinyjin/DualAnoDiff)  
**领域**: 图像生成 / 异常检测  
**关键词**: 异常图像生成, 少样本生成, 双扩散模型, 工业缺陷检测, 图像-掩码对

## 一句话总结

提出 DualAnoDiff，利用双关联扩散模型同时生成整体异常图像和对应异常部分，解决了少样本场景下异常图像生成中多样性不足、融合不自然和掩码不对齐的问题，在下游异常检测任务中达到 SOTA。

## 研究背景与动机

**领域现状**：工业异常检测面临异常样本稀缺的问题，现有方法要么是无监督仅用正常样本，要么是半监督用少量异常数据，但在异常定位和分类上表现有限。

**现有痛点**：现有异常生成方法分两类——模型无关方法（如 Cut-Paste）生成不够逼真；生成模型方法（如 AnomalyDiffusion）只关注异常部分，导致异常与原图融合不自然，且单独生成的掩码可能出现在背景上。

**核心矛盾**：异常图像和对应掩码需要高度对齐，但现有方法将整体图像生成和异常部分生成分开处理，缺乏显式的对齐约束。

**本文目标**：设计一种能同时生成整体异常图像和对应异常部分的方法，确保两者一致性和掩码精确对齐。

**切入角度**：受 LayerDiffusion 启发，将异常图像分解为整体图像和异常部分两个层次，用两个关联的扩散模型分别生成。

**核心 idea**：用双关联扩散模型（Dual-Interrelated Diffusion）同时生成整体异常图像和对应异常区域，通过自注意力交互模块保持两者一致性。

## 方法详解

### 整体框架

输入少量异常图像-掩码对，模型包含两个基于预训练 Stable Diffusion 的分支（通过 LoRA 微调）：全局分支 SD 生成完整异常图像，异常分支 SD* 生成仅含异常区域的图像。两分支共享时间步，通过自注意力交互模块（SAIM）在每个注意力块后交换信息。

### 关键设计

1. **双关联扩散模型（Dual-Interrelated Diffusion）**:

    - 功能：同时生成整体异常图像和对应的异常部分
    - 核心思路：在预训练 SD 上添加两个 LoRA 分别微调为全局分支和异常分支，使用嵌套提示（如"a vfx with sks"和"sks"），两分支共享相同时间步同步去噪
    - 设计动机：通过同时生成整体和局部，实现异常与原图的自然融合，并确保掩码与异常高度对齐

2. **自注意力交互模块（SAIM）**:

    - 功能：在双分支之间交换位置、细节和语义信息
    - 核心思路：将两分支的中间特征 $\varphi_i(z)$ 和 $\varphi_i(z')$ 拼接后重排为"bw 2 c"格式，进行共享自注意力计算，再分离回各分支并加残差连接
    - 设计动机：确保生成的整体异常图像和局部异常图像在空间位置和语义上保持一致

3. **背景补偿模块（BCM）**:

    - 功能：保持生成图像中背景的准确性和物体形状的完整性
    - 核心思路：先用 U2-Net 分割出背景图像 $I_b$，将其作为额外条件送入全局分支，提取背景的 Key 和 Value 通过自适应融合 MLP（带可学习缩放因子 $\gamma$）注入全局分支的自注意力层
    - 设计动机：解决少样本场景下物体变形、背景混淆等问题，让模型更专注于物体本身的生成

### 损失函数 / 训练策略

总损失为两个分支的标准扩散损失之和：全局分支预测整体图像的噪声，异常分支预测异常部分的噪声，两者都受对应提示引导。文本编码器 $\tau_\theta$ 为可训练的，掩码通过 SAM 等分割算法从生成的异常部分中获取。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文 | AnomalyDiffusion | 提升 |
|--------|------|------|-------------------|------|
| MVTec AD (pixel) | AUROC | 99.1% | - | SOTA |
| MVTec AD (pixel) | AP | 84.5% | - | SOTA |
| MVTec AD 平均 | IS | 最高 | 次高 | 多类别领先 |
| MVTec AD 平均 | IC-L | 最高 | 次高 | 多类别领先 |

### 消融实验

验证了双分支结构、SAIM、BCM 各组件的贡献：移除任一组件都会导致生成质量和下游任务性能下降。

### 关键发现

- 双分支结构能同时保证生成的多样性和掩码的精确性
- BCM 对背景简单但物体复杂的类别（如 bottle、pill）改善尤为明显
- 嵌套提示的设计让模型能正确分离物体属性和异常属性

## 亮点与洞察

- 将异常图像生成问题转化为同时生成整体和局部的双流问题，思路新颖
- 仅用两个 LoRA 即可扩展单一扩散模型为双关联模型，参数高效
- 从生成的异常部分直接获取掩码，避免了掩码单独生成的不对齐问题

## 局限与展望

- 依赖 U2-Net 进行前景分割，对分割质量有一定要求
- 训练数据极少（每类平均 8 张），对某些复杂纹理类别效果仍有提升空间
- 生成的异常类型受限于训练样本中已有的异常类型

## 相关工作与启发

- AnomalyDiffusion 是最直接的前驱工作，但只关注异常部分生成
- LayerDiffusion 的层分解思路启发了本文的双分支设计
- 可以考虑将此方法扩展到其他需要图像-标注对齐的少样本生成任务

## 评分

- 新颖性：8/10 — 双关联扩散的设计思路独特
- 技术深度：7/10 — 各模块设计合理但相对直观
- 实验充分度：8/10 — 在多个下游任务上验证了有效性
- 写作质量：7/10 — 结构清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](dual_diffusion_unified_generation_understanding.md)
- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)
- [\[CVPR 2025\] TurboFill: Adapting Few-Step Text-to-Image Model for Fast Image Inpainting](turbofill_adapting_few-step_text-to-image_model_for_fast_image_inpainting.md)
- [\[CVPR 2025\] Diffusion Self-Distillation for Zero-Shot Customized Image Generation](diffusion_self-distillation_for_zero-shot_customized_image_generation.md)
- [\[CVPR 2025\] Enhancing Image Aesthetics with Dual-Conditioned Diffusion Models Guided by Multimodal Perception](enhancing_image_aesthetics_with_dual-conditioned_diffusion_models_guided_by_mult.md)

</div>

<!-- RELATED:END -->
