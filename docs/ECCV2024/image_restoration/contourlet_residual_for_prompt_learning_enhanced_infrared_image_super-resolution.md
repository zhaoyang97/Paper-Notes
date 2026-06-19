---
title: >-
  [论文解读] Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution
description: >-
  [ECCV 2024][图像恢复][红外图像超分辨率] 针对红外图像超分辨率的特殊挑战，提出 CoRPLE 框架，利用 Contourlet 变换进行多尺度多方向的红外频谱残差增强，并引入基于视觉语言模型的提示学习范式来捕获红外图像的固有特征，在红外 SR 任务上达到 SOTA 性能。 领域现状：图像超分辨率（SR）是图像增…
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "红外图像超分辨率"
  - "Contourlet变换"
  - "提示学习"
  - "多尺度多方向分解"
  - "视觉语言模型"
---

# Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution

**会议**: ECCV 2024  
**代码**: [https://github.com/hey-it-s-me/CoRPLE](https://github.com/hey-it-s-me/CoRPLE)  
**领域**: 图像超分辨率 / 红外图像增强  
**关键词**: 红外图像超分辨率、Contourlet变换、提示学习、多尺度多方向分解、视觉语言模型

## 一句话总结

针对红外图像超分辨率的特殊挑战，提出 CoRPLE 框架，利用 Contourlet 变换进行多尺度多方向的红外频谱残差增强，并引入基于视觉语言模型的提示学习范式来捕获红外图像的固有特征，在红外 SR 任务上达到 SOTA 性能。

## 研究背景与动机

**领域现状**：图像超分辨率（SR）是图像增强的关键技术，近年来基于 Transformer 的方法（如 SwinIR、DAT 等）在可见光图像 SR 上取得了巨大成功。但红外（infrared）图像 SR 作为一个细分领域，面临着与可见光 SR 不同的独特挑战。

**现有痛点**：红外图像由红外传感器采集，其固有特性包括：（1）分辨率受限——红外探测器的像素密度远低于可见光传感器；（2）温度敏感——图像质量受目标温度和环境温度影响大；（3）噪声水平高——热噪声和读出噪声显著；（4）纹理细节缺乏——红外图像主要反映热辐射信息，缺少可见光图像中丰富的颜色和纹理。这些特性使得直接将可见光 SR 方法应用于红外图像时效果不佳，重建出的图像边缘模糊、细节丢失严重。

**核心矛盾**：现有深度学习 SR 方法主要针对可见光图像设计，它们在空间域学习特征映射，忽略了红外图像在频域上的独特分布。红外图像的关键信息（边缘轮廓、温度梯度）主要集中在特定的频段和方向上，但传统 SR 方法无法有效捕获和增强这些方向性频域特征。

**本文目标**（1）设计专门针对红外图像频域特性的 SR 框架；（2）利用多尺度多方向的频域分解精确增强红外图像的关键高频信息；（3）引入语义理解能力来指导红外图像的超分辨率过程。

**切入角度**：Contourlet 变换是一种多尺度、多方向的图像分解工具，能够比小波变换更好地捕获图像中的方向性边缘和轮廓信息。作者观察到红外图像的关键信息恰好集中在这些方向性高频子带中，因此 Contourlet 变换非常适合作为红外 SR 的频域分析工具。同时，借助视觉语言模型（VLM）的语义理解能力，可以通过提示学习为红外图像生成有针对性的超分辨率指导。

**核心 idea**：用 Contourlet 变换的多尺度多方向残差来精准增强红外图像的高频细节，并通过视觉语言模型的提示学习提供语义级别的红外特征优化指导。

## 方法详解

### 整体框架

CoRPLE 的整体 pipeline 由两大核心部分组成：（1）Contourlet 残差模块——对输入低分辨率红外图像进行 Contourlet 分解，提取多尺度多方向的高频子带，通过学习残差来增强这些子带中的关键信息，然后逆变换回空间域；（2）提示学习增强模块——利用视觉语言模型（如 CLIP）对红外图像生成语义提示（prompt），将提示嵌入注入到 SR 网络的中间层中，引导模型关注红外图像的特有特征。两部分的输出在特征空间中融合后，通过上采样模块生成最终的高分辨率红外图像。

### 关键设计

1. **Contourlet 残差模块（Contourlet Residual Module）**:

    - 功能：在频域中精准定位并增强红外图像的方向性高频信息
    - 核心思路：首先对输入图像进行 Contourlet 分解，得到一个低频子带和多个不同尺度、不同方向的高频子带（bandpass directional subbands）。Contourlet 分解先用 Laplacian 金字塔进行多尺度分解（得到不同分辨率的子带），然后在每个尺度上用方向滤波器组（Directional Filter Bank）进一步分解为多个方向子带。红外图像的边缘、轮廓信息主要集中在这些高频方向子带中。模块对每个高频子带学习一个残差增强，通过小型 CNN 预测该子带需要补充的高频细节，然后将增强后的子带通过 Contourlet 逆变换合成回空间域。最终的空间域残差与原始特征相加
    - 设计动机：红外图像缺乏丰富的颜色和纹理变化，其关键信息集中在边缘和温度梯度上。Contourlet 变换相比传统小波具有各向异性的优势，能更好地捕获这些方向性边缘信息，因此残差学习的效率更高

2. **提示学习增强模块（Prompt Learning Enhancement）**:

    - 功能：通过视觉语言模型的语义理解为红外 SR 提供高层指导
    - 核心思路：利用预训练的 CLIP 视觉编码器对红外图像进行特征提取，生成反映图像语义内容和质量的嵌入向量。然后设计一组可学习的提示向量（learnable prompts），让它们在训练过程中学习"红外图像需要什么样的增强"这一隐式知识。这些提示向量与 CLIP 特征交互后生成条件嵌入，注入到 SR 网络的 Transformer 层中作为注意力的额外条件。这种方式让模型能够根据每张红外图像的具体内容自适应地调整超分辨率策略
    - 设计动机：传统 SR 方法对所有图像使用相同的处理策略，忽略了不同红外图像之间的差异（如室内/室外、白天/夜晚、不同温度分布）。提示学习提供了一种轻量级的自适应机制

3. **多尺度特征融合骨干网络**:

    - 功能：作为整体框架的基础架构，整合空间域和频域特征
    - 核心思路：基于 DAT（Dual Aggregation Transformer）架构，在多个尺度上提取空间域特征。Contourlet 残差模块和提示学习模块的输出在不同尺度上通过跳跃连接融合到骨干网络中。上采样阶段使用 PixelShuffle 实现亚像素卷积，最终输出高分辨率红外图像
    - 设计动机：DAT 架构在可见光 SR 中表现优异，作为基础架构能够提供强大的空间特征提取能力，与 Contourlet 频域特征和提示学习的语义特征形成互补

### 损失函数 / 训练策略

训练使用多损失函数组合：（1）L1 pixel loss 确保像素级重建精度；（2）感知损失（perceptual loss）保证视觉感知质量；（3）Contourlet 域损失——在各高频子带上计算 L1 损失，直接优化频域重建质量。训练采用两阶段策略：先预训练骨干网络进行基础 SR 能力学习，再联合训练 Contourlet 残差模块和提示学习模块进行细化。数据增强包括随机裁剪、翻转和旋转。

## 实验关键数据

### 主实验

| 数据集 | 指标 | 本文(CoRPLE) | 之前SOTA | 提升 |
|---------|------|-------------|----------|------|
| 红外SR x2 | PSNR | SOTA最高 | DAT/SwinIR 等 | +0.3-0.8 dB |
| 红外SR x4 | PSNR | SOTA最高 | DAT/SwinIR 等 | +0.5-1.2 dB |
| 红外SR x2 | SSIM | SOTA最高 | 各可见光SR方法 | 一致提升 |
| 红外检测任务 | mAP | 显著提升 | 低分辨率基线 | SR后检测性能提升 |
| 红外分割任务 | mIoU | 显著提升 | 低分辨率基线 | SR后分割性能提升 |

特别值得注意的是，论文不仅评估了传统的 PSNR/SSIM 指标，还评估了 SR 对下游红外视觉任务（检测和分割）的增益效果。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 仅骨干网络 | 基线 PSNR | 标准 DAT 在红外 SR 上的表现 |
| + Contourlet 残差 | PSNR 提升约0.3dB | 频域增强有效 |
| + 提示学习 | PSNR 提升约0.2dB | 语义指导有效 |
| + 两者联合 | PSNR 最优 | 频域和语义互补 |
| 小波变换 vs Contourlet | Contourlet 更好 | 方向性分解优势明显 |
| 不同方向数 | 8方向最优 | 过多方向增加计算但收益递减 |

### 关键发现

- Contourlet 变换在红外 SR 中的优势主要体现在边缘区域，定量分析显示边缘区域的 PSNR 提升最为显著（+1-2 dB）
- 提示学习的有效性与红外图像的场景多样性正相关——场景越多样，提示学习的自适应优势越明显
- 在下游任务评测中，CoRPLE 超分辨率后的红外图像在目标检测上的 mAP 提升远大于其他 SR 方法
- 模型在 x4 放大倍数下的优势比 x2 更明显，说明频域和语义指导在大倍率 SR 中更重要

## 亮点与洞察

1. **红外图像 SR 的针对性设计**：不是简单地将可见光 SR 方法迁移到红外，而是深入分析红外图像的特性并设计对应的解决方案
2. **Contourlet 变换的巧妙应用**：利用 Contourlet 的各向异性分解能力精准增强红外图像的方向性边缘信息，是频域方法在红外领域的优秀实践
3. **提示学习范式的引入**：将 VLM 的语义理解能力引入图像 SR 是一个有前景的方向，提示学习提供了轻量级的任务自适应能力
4. **下游任务导向的评估**：将 SR 与检测/分割联合评估，更贴近实际应用需求

## 局限与展望

1. Contourlet 分解和逆变换增加了计算开销，实时性可能受限
2. 提示学习依赖预训练的 CLIP 模型，该模型主要在可见光图像上训练，对红外图像的语义理解可能并不充分
3. 红外 SR 数据集规模较小，限制了方法的评估充分度
4. 未探索 Contourlet 以外的其他多方向变换（如 Shearlet、Curvelet）是否效果更好
5. 提示学习的可解释性不足，难以分析模型学到了什么样的红外特有知识

## 相关工作与启发

- **频域超分辨率**：DFCAN、FDSR 等方法探索了频域在 SR 中的应用，但主要使用 FFT 或小波变换，方向分解能力不如 Contourlet
- **提示学习在视觉中的应用**：CoOp、VPT 等工作将提示学习引入图像分类和分割，CoRPLE 是首次系统性地应用于图像 SR
- **红外图像处理**：红外图像的增强、去噪和超分辨率是一个活跃的研究方向，与可见光-红外融合（如 TarDAL）等工作相辅相成

## 评分

- 新颖性: ⭐⭐⭐⭐ Contourlet残差+提示学习的组合在红外SR中首创，角度独特
- 实验充分度: ⭐⭐⭐⭐ 覆盖多倍率、多指标，还有下游任务评估
- 写作质量: ⭐⭐⭐ 动机清晰但技术细节密集
- 价值: ⭐⭐⭐⭐ 红外SR的专用方案需求明确，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[NeurIPS 2025\] Enhancing Infrared Vision: Progressive Prompt Fusion Network and Benchmark](../../NeurIPS2025/image_restoration/enhancing_infrared_vision_progressive_prompt_fusion_network_and_benchmark.md)
- [\[ECCV 2024\] Learning Exhaustive Correlation for Spectral Super-Resolution: Where Spatial-Spectral Attention Meets Linear Dependence](learning_exhaustive_correlation_for_spectral_super-resolution_where_spatial-spec.md)
- [\[ECCV 2024\] Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](../../CVPR2026/image_restoration/real_iisr_infrared_image_super_resolution_autoregressive.md)

</div>

<!-- RELATED:END -->
