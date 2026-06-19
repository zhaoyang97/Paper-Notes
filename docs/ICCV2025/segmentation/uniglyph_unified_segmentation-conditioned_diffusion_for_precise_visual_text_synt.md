---
title: >-
  [论文解读] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis
description: >-
  [ICCV 2025][语义分割][视觉文本生成] 提出 UniGlyph，一种以分割掩码为统一条件信号的视觉文本生成框架，通过自适应字形条件（AGC）和字形区域损失（GRL）替代传统的渲染字形条件，实现单一 ControlNet 架构下中英文文字图像生成的 SOTA，尤其在小字体和复杂排版场景大幅领先。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "视觉文本生成"
  - "分割掩码条件"
  - "扩散模型"
  - "中英文字形"
  - "小字体生成"
  - "ControlNet"
---

# UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis

**会议**: ICCV 2025  
**arXiv**: [2507.00992](https://arxiv.org/abs/2507.00992)  
**代码**: 未公开  
**领域**: 图像分割  
**关键词**: 视觉文本生成, 分割掩码条件, 扩散模型, 中英文字形, 小字体生成, ControlNet

## 一句话总结

提出 UniGlyph，一种以分割掩码为统一条件信号的视觉文本生成框架，通过自适应字形条件（AGC）和字形区域损失（GRL）替代传统的渲染字形条件，实现单一 ControlNet 架构下中英文文字图像生成的 SOTA，尤其在小字体和复杂排版场景大幅领先。

## 研究背景与动机

文本到图像生成中精确渲染视觉文字（字形 glyph）仍是核心未解难题，现有方法面临字符边缘模糊、语义不一致、字体/颜色控制不足等问题。当前主流的 ControlNet 方案（如 AnyText、GlyphDraw2）使用预渲染的字形图像作为条件，但存在根本性缺陷：

**信息退化问题**：预渲染字形仅保留字形形状和位置，丢失了原始字体特征和颜色信息。这种不完整的条件信号迫使模型在训练中学习从合成字形（通常默认字体）到真实排版变化的隐式映射。为补偿信息丢失，现有方法不得不引入辅助模块：
- AnyText：文本嵌入替换模块（位置编码）
- GlyphDraw2：风格引导分支（字体/颜色控制）

这导致了**多分支架构膨胀**：计算复杂度增加、模型可复用性降低、优化冲突——在生成小字体或风格化字形时表现尤为明显。

本文的核心洞察：**分割掩码天然保留了所有字形属性**——形状、位置、字体风格和颜色——不需要任何辅助控制模块。基于此，UniGlyph 用像素级视觉文字分割掩码替代渲染字形图像作为统一条件输入。

## 方法详解

### 整体框架

UniGlyph 包含三个核心组件：
1. **双语文本分割模型**：基于 Hi-SAM (SAM-TS) 微调，从图像中提取像素级文本分割掩码
2. **Flow Matching 扩散模型 + DiT ControlNet**：基于 FLUX.1-dev，用分割掩码作为 ControlNet 条件
3. **LayoutTransformer（可选）**：推理时自动生成文本布局和风格信息

### 关键设计一：自适应字形条件（Adaptive Glyph Condition, AGC）

分割掩码直接使用存在问题：黑色文字在黑色背景上会融合。同时分割模型对小字体不准确。因此设计自适应策略：

用 PP-OCRv4 获取每个字形区域 $R_i$ 的边界框，计算单字平均面积 $A_{\text{avg},i} = A_i / N_i$，设阈值 $T = 4900$ 像素：

$$G_i = \begin{cases} \text{Canny}(M_{\text{seg}}) + M_{\text{seg}} \odot I, & \text{if } A_{\text{avg},i} > T \\ (M_{\text{pos}} \odot I_i)^{\text{blur}}, & \text{if } A_{\text{avg},i} \leq T \end{cases}$$

- **大字形区域**：使用分割掩码提取的原始颜色字形 + Canny 边缘增强（解决黑字融合问题）
- **小字形区域**：退化为位置掩码裁剪的原图区域 + 高斯模糊边界（避免不准确分割的干扰）

最终条件：$G = \bigcup_i G_i$

### 关键设计二：Flow Matching 扩散模型

基于 Flow Matching 框架，学习连续时间速度场 $\mathbf{v}^*(z_t, t)$。图像 $I$ 和字形条件 $G$ 通过 VAE 编码为潜在表示 $z_0, z_g$，ControlNet 产生字形特征 $z_s = C(z_g, c_{te}, t)$。Flow matching 损失：

$$L_{\text{fm}} = \mathbb{E}_{z_0, z_s, c_{te}, t}\left[\|\mathbf{v}_\theta(z_t, z_s, c_{te}, t) - \mathbf{v}^*(z_t, t)\|_2^2\right]$$

### 关键设计三：字形区域损失（Glyph Region Loss）

利用分割掩码在像素空间对字形区域施加额外的 MSE 损失，等效于对字形区域赋予更高的损失权重：

掩码选择同样基于自适应策略：

$$M_{\text{gr}} = \begin{cases} M_{\text{seg}}, & \text{if } A_{\text{avg},i} > T \\ M_{\text{pos}}, & \text{if } A_{\text{avg},i} \leq T \end{cases}$$

字形区域损失：

$$L_{\text{gr}} = \mathbb{E}_{\mathbf{x}_0, \hat{\mathbf{x}}_0}\left[\|M_{\text{gr}} \odot (\hat{\mathbf{x}}_0 - \mathbf{x}_0)\|_2^2\right]$$

总损失：$L = L_{\text{fm}} + \lambda \cdot L_{\text{gr}}$，其中 $\lambda = 1$，前 10 万步禁用 $L_{\text{gr}}$。

### LLM 布局预测

微调开源 LLM 将用户提示转换为结构化布局：`<rewritten prompt, texts, bboxes, fonts, colors>`。预定义字体和颜色集合映射为特殊 token，仅用 1000 条海报数据微调。

## 实验

### 主实验（AnyText-benchmark）

| 方法 | 中文 Sen.Acc | 中文 NED | 英文 Sen.Acc | 英文 NED |
|------|-------------|----------|-------------|----------|
| AnyText-V1.1 | 0.6923 | 0.8423 | 0.6564 | 0.8685 |
| GlyphDraw2 | 0.7350 | 0.8451 | 0.7369 | 0.8921 |
| AnyText2 | 0.7130 | 0.8516 | 0.8096 | 0.9184 |
| CharGen | 0.7499 | 0.8609 | 0.8096 | 0.9205 |
| **UniGlyph** | **0.8267** | **0.8976** | **0.9018** | **0.9582** |

UniGlyph 中文准确率超 AnyText2 +11.4%，英文超 CharGen +9.2%。

### 小字体生成（MiniText-benchmark）

| 方法 | Sen.Acc | NED | ClipScore |
|------|---------|-----|-----------|
| SD3 | 0.0000 | 0.0005 | 0.7990 |
| AnyText-V1.1 | 0.0138 | 0.4680 | 0.8098 |
| GlyphDraw2 | 0.0100 | 0.4508 | 0.8146 |
| Glyph-ByT5 | 0.3881 | 0.8268 | 0.8594 |
| **UniGlyph** | **0.7925** | **0.9537** | 0.8124 |

UniGlyph 在小字体上的准确率是次优方法（Glyph-ByT5）的 **2 倍以上**。

### 消融实验

$\lambda$ 值的影响：

| $\lambda$ | Sen.Acc | NED | ClipScore |
|-----------|---------|-----|-----------|
| 0 | 0.8179 | 0.8952 | 0.7868 |
| 0.1 | 0.8166 | 0.8945 | 0.7871 |
| 1 | **0.8188** | **0.8958** | **0.7896** |
| 4 | 0.8158 | 0.8949 | 0.7870 |

AGC 的必要性：

| 方法 | Sen.Acc | NED | ClipScore |
|------|---------|-----|-----------|
| w/o AGC | 0.7724 | 0.9348 | 0.8064 |
| w/o Gaussian Blur | 0.7851 | 0.9508 | 0.7963 |
| UniGlyph | **0.7849** | **0.9507** | **0.8097** |

关键发现：
1. 字形区域损失（$\lambda > 0$）相比无损失（$\lambda = 0$）提升了准确率和图像质量（ClipScore）
2. 自适应混合策略显著提升小字体生成准确率（0.7724→0.7851），高斯模糊进一步恢复 ClipScore
3. 训练数据量仅需 7.36M（远低于 AnyText 的 30M 和 TextDiffuser 数千万），证明方法的高样本效率

## 亮点与洞察

1. **范式转变**：用分割掩码替代渲染字形，从信息保留的角度根本性地解决了条件信号退化问题
2. **架构简化**：单一 ControlNet 替代多分支架构，消除了位置编码/风格引导等辅助模块
3. **自适应设计**：根据字形区域大小在精确分割和粗略位置之间切换，优雅处理了分割模型对小字体的不足
4. **数据集贡献**：GlyphMM-3M（3M+ 高分辨率双语图像）和 MiniText-benchmark 填补了社区空白

## 局限性

1. 分割模型对极小字体的分割仍不精确，需要 fallback 到位置掩码
2. 推理时需要先运行文本分割模型，增加了推理流程的复杂度
3. 由于资源限制仅使用了数据集的子集训练，潜力尚未充分发挥
4. 字形区域损失需要在训练中从潜在空间重建图像到像素空间，降低训练速度

## 相关工作

- **文本渲染方法**：AnyText、GlyphDraw2 基于 ControlNet + 渲染字形条件；Glyph-ByT5 用字符级编码器
- **可控生成**：ControlNet、T2I-Adapter、IP-Adapter 提供多种控制信号
- **文本分割**：Hi-SAM 基于 SAM 的层次化文本分割模型
- **LLM 布局生成**：LayoutGPT、TextDiffuser-2 利用 LLM 生成布局边界框

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 分割掩码统一条件的范式转变具有开创性
- **技术质量**: ⭐⭐⭐⭐ — 自适应策略设计精巧，但消融实验在缩小分辨率上进行
- **实用性**: ⭐⭐⭐⭐⭐ — 中英文支持、小字体能力强、架构简洁
- **写作质量**: ⭐⭐⭐⭐ — 问题分析清晰，实验设置完善

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[NeurIPS 2025\] HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios](../../NeurIPS2025/segmentation/hopadiff_holistic-partial_aware_fourier_conditioned_diffusion_for_referring_huma.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)
- [\[NeurIPS 2025\] FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis](../../NeurIPS2025/segmentation/fast_foreground-aware_diffusion_with_accelerated_sampling_trajectory_for_segment.md)
- [\[ICCV 2025\] Learning Precise Affordances from Egocentric Videos for Robotic Manipulation](learning_precise_affordances_from_egocentric_videos_for_robotic_manipulation.md)

</div>

<!-- RELATED:END -->
