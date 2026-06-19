---
title: >-
  [论文解读] Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots
description: >-
  [ICML 2025][图像生成][自回归模型] 提出 Hi-MAR，在掩码自回归图像生成中引入低分辨率 token 作为中间枢纽，建立从粗到细的层次化生成流程，并用 Diffusion Transformer Head 增强 token 间依赖建模，在 ImageNet 上以更少计算量显著超越 MAR（FID 提升 0.38）。
tags:
  - "ICML 2025"
  - "图像生成"
  - "自回归模型"
  - "层次化生成"
  - "掩码自回归"
  - "全局上下文"
  - "Transformer"
---

# Hierarchical Masked Autoregressive Models with Low-Resolution Token Pivots

**会议**: ICML 2025  
**arXiv**: [2505.20288](https://arxiv.org/abs/2505.20288)  
**代码**: [https://github.com/HiDream-ai/himar](https://github.com/HiDream-ai/himar)  
**领域**: 扩散模型 / 图像生成  
**关键词**: 自回归模型, 层次化生成, 掩码自回归, 全局上下文, Diffusion Transformer Head

## 一句话总结
提出 Hi-MAR，在掩码自回归图像生成中引入低分辨率 token 作为中间枢纽，建立从粗到细的层次化生成流程，并用 Diffusion Transformer Head 增强 token 间依赖建模，在 ImageNet 上以更少计算量显著超越 MAR（FID 提升 0.38）。

## 研究背景与动机
**领域现状**：自回归（AR）模型在视觉生成中逐渐崛起，以 MAR 为代表的掩码自回归模型通过连续值 token + diffusion loss 避免了离散化带来的信息损失。

**现有痛点**：MAR 等模型仅在单一尺度的稠密 token 序列上执行自回归建模，缺乏全局上下文信息，特别是对早期 token 的预测不利。此外，MAR 使用 MLP-based diffusion head 独立处理每个 token，忽略了 token 之间的空间依赖关系，可能产生异常亮点等伪影。

**核心矛盾**：单尺度自回归将全局结构构建和局部细节精修混为一体，不符合人类"先全局后局部"的感知习惯。

**本文目标** (a) 如何在自回归建模中引入全局结构信息？(b) 如何在 diffusion head 中建模 token 间的相互依赖？

**切入角度**：先用少量低分辨率 token 捕获全局结构，再以此为条件引导高分辨率稠密 token 的生成。

**核心 idea**：用低分辨率 token 做"枢纽"的分层掩码自回归 + 用 Transformer 替代 MLP 的 diffusion head。

## 方法详解

### 整体框架
Hi-MAR 是两阶段层次化掩码自回归模型。输入图像同时编码为低分辨率（128×128）和高分辨率（256×256）的连续 token 序列。第一阶段在低分辨率 token 上做掩码自回归建模，输出条件 token（而非直接的视觉 token）以反映全局结构；第二阶段将这些条件 token 与高分辨率掩码 token 拼接，送入同一个 Transformer 进行精细生成。

### 关键设计

1. **层次化掩码自回归 Transformer（Hi-MAR Transformer）**:

    - 功能：分两阶段建模，先粗后细
    - 核心思路：第一阶段用双向注意力在低分辨率 token 上做 MAR，输出条件 token $Z^s$；第二阶段将 $Z^s$ 与高分辨率掩码 token 拼接，再次经过 Transformer 生成稠密条件 token
    - 设计动机：直接用低分辨率视觉 token（而非条件 token）来引导会导致训练-推理不一致——训练时用 ground truth 低分辨率 token，推理时用预测的（含噪声的）低分辨率 token。用 Transformer 输出的条件 token 代替视觉 token 可缓解此问题

2. **Scale-aware Transformer Block**:

    - 功能：让共享 Transformer 感知当前处理的是哪个尺度
    - 核心思路：用正弦嵌入编码 scale 信息，通过 MLP 生成 scale vector $v$，再用 adaLN-Zero 操作调制 LayerNorm 的缩放/偏移参数及残差连接的缩放参数：$z_{a} = z^i + \gamma_1 \cdot \text{Attention}(\alpha_1 \cdot \text{LN}(z^i) + \beta_1)$
    - 设计动机：共享 Transformer 同时处理两个尺度的 token，不加 scale 引导会导致模糊

3. **Diffusion Transformer Head**:

    - 功能：替代 MLP-based diffusion head，在 masked token 预测时建模所有 token 之间的依赖
    - 核心思路：在第二阶段使用带自注意力的 Transformer blocks 作为 diffusion head，输入是所有（masked + unmasked）条件 token 经 adaLN 调制后的表示，而非仅 masked token 的 MLP 独立处理
    - 设计动机：MLP head 独立处理每个 token，丢失了图像的全局空间结构信息，Transformer head 通过自注意力捕获 token 间交互

### 损失函数 / 训练策略
- 第一阶段 masking ratio 在 $[0.7, 1.0]$ 随机采样（同 MAR）
- 第二阶段使用 MaskGIT 的 cosine masking 策略
- 两阶段都使用标准 diffusion denoising loss：$\mathcal{L}(z_i, x_i) = \mathbb{E}_{\epsilon,t}[\|\epsilon - \epsilon_\theta(x_i^t|t, z_i)\|^2]$
- 推理时第一阶段 32 步，第二阶段仅 4 步（因有 Transformer head 更强的建模能力 + 全局结构已由第一阶段提供）

## 实验关键数据

### 主实验

| 数据集 | 模型 | FID (w/ CFG) ↓ | IS ↑ | Precision | Recall |
|--------|------|----------|------|-----------|--------|
| ImageNet 256 | MAR-B | 2.31 | 281.7 | 0.82 | 0.57 |
| ImageNet 256 | **Hi-MAR-B** | **1.93** | **293.0** | 0.81 | 0.59 |
| ImageNet 256 | MAR-H | 1.55 | 303.7 | 0.81 | 0.62 |
| ImageNet 256 | **Hi-MAR-H** | **1.52** | **322.78** | 0.80 | 0.63 |
| MS-COCO 256 | MAR | 6.36 | - | - | - |
| MS-COCO 256 | **Hi-MAR-S** | **4.77** | - | - | - |

### 消融实验

| 配置 | FID ↓ | 说明 |
|------|-------|------|
| MAR-B baseline | 2.31 | 基础单尺度 MAR |
| + 层次化 (视觉 token 引导) | 2.28 | 训练推理不一致，几乎无提升 |
| + 层次化 (条件 token 引导) | 2.07 | 显著提升 0.24 |
| + Diff Transformer Head (第二阶段) | 1.98 | 再降 0.09 |
| + Scale vector (完整 Hi-MAR) | **1.93** | 最终最优 |

### 关键发现
- 用条件 token 替代视觉 token 引导是最关键的设计，贡献了 0.24 FID 提升
- Diffusion Transformer Head 仅在第二阶段有效；在第一阶段替换 MLP head 无显著收益
- Hi-MAR 推理速度更快：第二阶段仅需 4 步就可达近饱和质量，总计算量仅为 MAR 的 54%

## 亮点与洞察
- **层次化生成的巧妙解耦**：先生成全局结构（低分辨率 token），再精修细节（稠密 token），既符合人类感知也降低了计算量
- **训练-推理一致性设计**：用 Transformer 输出的条件 token 而非直接的视觉 token 来引导第二阶段，巧妙回避了 ground truth / prediction 不一致的问题
- **可迁移思路**：Diffusion Transformer Head 的设计（用自注意力替代 MLP 来建模 token 间依赖）可以迁移到其他需要 per-token prediction 的任务

## 局限与展望
- 仅验证了两级层次结构，更多级（如 3-4 级）的效果未探索
- 低分辨率 token 的分辨率选择（128 vs 64 vs 32）的影响未深入分析
- 文本到图像生成仅在 MS-COCO 上验证，缺少大规模 T2I 数据集（如 LAION）的实验

## 评分
- 新颖性: ⭐⭐⭐⭐ 层次化 MAR + Transformer diffusion head 组合创新
- 实验充分度: ⭐⭐⭐⭐ ImageNet + MS-COCO + 充分消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观
- 价值: ⭐⭐⭐⭐ 自回归图像生成的有效改进方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](../../ICCV2025/image_generation/lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[CVPR 2025\] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](../../CVPR2025/image_generation/hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)
- [\[NeurIPS 2025\] Conditional Panoramic Image Generation via Masked Autoregressive Modeling](../../NeurIPS2025/image_generation/conditional_panoramic_image_generation_via_masked_autoregres.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](../../ICCV2025/image_generation/dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)

</div>

<!-- RELATED:END -->
