---
title: "Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment"
authors: "Fatemeh Behrad, Farzad Khorasani, Amirhossein Kazerouni, Reza Vahidimajd, Mohammad Hajizadeh, Luc Van Gool"
venue: "CVPR 2025"
date: 2025-04-03
tags: [image-aesthetics, vit, token-reduction, multi-scale, efficient-inference]
arxiv: "2504.02522"
---

# Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment

**作者**: Fatemeh Behrad, Farzad Khorasani 等  
**机构**: KU Leuven  
**会议**: CVPR 2025  

## 研究背景与动机

图像美学评估（Image Aesthetic Assessment, IAA）旨在自动判断图像的美学质量，广泛应用于照片推荐、自动裁剪、图像增强等场景。Vision Transformer (ViT) 在 IAA 任务上表现优异，但面临以下核心问题：

**固定分辨率的局限**：ViT 通常将输入图像 resize 为固定尺寸（如 224×224），这会破坏图像的原始构图和长宽比，而这些恰恰是美学评估的关键因素

**高分辨率的计算代价**：直接处理高分辨率图像会导致 token 数量急剧增加，注意力计算复杂度为 $O(n^2)$，实际部署困难

**构图信息丢失**：传统的 resize 和 center crop 会改变主体位置关系和前景/背景比例，导致模型无法学习到真正的构图美学

**多尺度信息缺失**：美学评估需要同时关注全局布局（构图、色彩分布）和局部细节（纹理、锐度），单一尺度的 ViT 难以兼顾

Charm 的核心思想是在保留构图（Composition）、高分辨率（High-resolution）、长宽比（Aspect Ratio）和多尺度（Multi-scale）信息的同时，大幅减少 token 数量。

## 方法详解

### CHARM 核心设计

Charm 代表四个关键属性的首字母缩写：
- **C**omposition：保留原始构图
- **H**igh-resolution：保留高分辨率细节
- **A**spect **R**atio：保留原始长宽比
- **M**ulti-scale：融合多尺度信息

### 双尺度 Token 策略

Charm 采用两层级的 token 化方案：

**粗粒度 Token（Coarse Tokens）**：
- 使用扩大的 patch 尺寸 $p' = p \times n$，其中 $p$ 为原始 patch 大小，$n$ 为缩放因子
- 覆盖整张图像，保持全局构图和长宽比信息
- Token 数量 = $\frac{H \times W}{(p \times n)^2}$

**细粒度 Token（Fine Tokens）**：
- 使用原始 patch 尺寸 $p$，仅在选定的重要区域提取
- 通过注意力分数或内容显著性选择最重要的 patch 位置
- 提供高分辨率局部细节

### 尺度嵌入（Scale Embedding）

为了让模型区分粗粒度和细粒度 token，引入可学习的尺度嵌入：

$$e_{scale} = \begin{cases} e_{coarse} & \text{if token from coarse scale} \\ e_{fine} & \text{if token from fine scale} \end{cases}$$

最终的 token 嵌入：$e_{token} = e_{patch} + e_{pos} + e_{scale}$

### Token 减少效率

| 配置 | Token 数量 | 相对原始 |
|------|-----------|---------|
| 原始 ViT (224×224) | 196 | 100% |
| 原始 ViT (448×448) | 784 | 400% |
| Charm (448×448) | ~180 | ~23% (对高分辨率) |
| Token 减少率 | - | **77.7%** |

### 训练策略

1. 使用预训练 ViT（如 DINOv2 或 CLIP）初始化
2. 冻结主干网络，仅微调尺度嵌入和分类头
3. 动态批处理：由于不同图像的长宽比不同，Token 数量可能不同，使用 padding + attention mask 处理

## 实验结果

### AVA 数据集

| 方法 | SRCC ↑ | PLCC ↑ | ACC ↑ |
|------|--------|--------|-------|
| NIMA (Inception) | 0.636 | 0.642 | 0.815 |
| MUSIQ | 0.726 | 0.738 | 0.832 |
| VILA | 0.738 | 0.745 | 0.841 |
| Baseline ViT | 0.741 | 0.745 | - |
| **Charm (ours)** | **0.773** | **0.779** | - |
| 提升 | +3.2% | **+4.5%** | - |

### TAD66k 数据集

| 方法 | SRCC ↑ | PLCC ↑ | ACC ↑ |
|------|--------|--------|-------|
| TANet | 0.432 | 0.441 | 0.692 |
| VILA | 0.521 | 0.534 | 0.743 |
| Baseline ViT | 0.538 | 0.547 | 0.752 |
| **Charm (ours)** | **0.612** | **0.625** | **0.794** |
| 提升 | +7.4% | +7.8% | **+14.8%** (vs TANet) |

### 消融实验

| 组件 | AVA PLCC | TAD66k ACC |
|------|----------|-----------|
| 仅粗粒度 | 0.753 | 0.768 |
| 仅细粒度 | 0.742 | 0.759 |
| 粗+细 (无尺度嵌入) | 0.769 | 0.783 |
| **Charm 完整** | **0.779** | **0.794** |

## 核心创新点

1. **CHARM 设计哲学**：系统性地保留四个美学关键属性（构图/高分辨率/长宽比/多尺度）
2. **双尺度 Token 策略**：避免了高分辨率带来的计算爆炸，同时保留全局和局部信息
3. **77.7% Token 减少**：在大幅降低计算量的同时提升美学评估精度
4. **即插即用**：可直接应用于任意预训练 ViT，无需修改 Transformer 架构

## 局限性

- 细粒度 patch 的选择策略目前依赖启发式方法，可能不是最优
- 对长宽比极端的图像（如全景图），padding 开销较大
- 未探索在视频美学评估中的扩展

## 相关工作

- MUSIQ: 多尺度图像质量评估 Transformer
- VILA: 视觉-语言对齐的美学评估
- TANet: 主题感知的美学网络
- Token 剪枝/合并：DynamicViT, ToMe 等

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Expert Pyramid Tuning: Efficient Parameter Fine-Tuning for Expertise-Driven Task Allocation](expert_pyramid_tuning_efficient_parameter_fine-tuning_for_expertise-driven_task_.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](../../ACL2025/model_compression/state_offset_tuning_ssm_peft.md)
- [\[ACL 2025\] C3A: Parameter-Efficient Fine-Tuning via Circular Convolution](../../ACL2025/model_compression/parameter-efficient_fine-tuning_via_circular_convolution.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](../../ICML2025/model_compression/parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ICCV 2025\] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models](../../ICCV2025/model_compression/vit-linearizer_distilling_quadratic_knowledge_into_linear-time_vision_models.md)

</div>

<!-- RELATED:END -->
