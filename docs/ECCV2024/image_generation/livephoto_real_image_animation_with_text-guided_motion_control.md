---
title: >-
  [论文解读] LivePhoto: Real Image Animation with Text-guided Motion Control
description: >-
  [ECCV 2024][图像生成] 提出 LivePhoto 图像动画框架，通过运动强度估计模块和文本重加权模块解决文本到运动映射的歧义性，实现基于真实图像和文本描述的高质量视频生成，且用户可额外控制运动强度。 文本引导的视频生成虽取得进展，但现有方法存在一个被忽视的问题：文本仅能控制视频的空间内容，却难以控制时间维度上的运…
tags:
  - "ECCV 2024"
  - "图像生成"
---

# LivePhoto: Real Image Animation with Text-guided Motion Control

**会议**: ECCV 2024  
**arXiv**: [2312.02928](https://arxiv.org/abs/2312.02928)  
**领域**: 图像生成

## 一句话总结

提出 LivePhoto 图像动画框架，通过运动强度估计模块和文本重加权模块解决文本到运动映射的歧义性，实现基于真实图像和文本描述的高质量视频生成，且用户可额外控制运动强度。

## 研究背景与动机

文本引导的视频生成虽取得进展，但现有方法存在一个被忽视的问题：**文本仅能控制视频的空间内容，却难以控制时间维度上的运动**。用户输入图像和运动描述（如"摇头"、"镜头推进"），生成的视频往往几乎静止或运动过于剧烈。

本文分析了两个核心原因：
1. 文本无法充分描述运动速度和幅度（如"摇头"缺少速度信息），导致同一文本对应多种运动强度，优化存在歧义
2. 文本同时包含"内容描述"和"运动描述"，当内容描述与参考图像冲突时，整个文本的影响被抑制，运动描述也被连带削弱

## 方法详解

### 整体框架

LivePhoto 基于冻结的 Stable Diffusion v1.5 构建，输入包括参考图像、文本和运动强度。核心组件：
- 参考图像潜表示与噪声拼接作为 UNet 输入（含帧嵌入和强度嵌入，共 10 通道）
- 内容编码器（DINOv2）提取图像 patch token，通过交叉注意力注入
- 可训练的运动模块（Motion Module）捕获帧间时序关系
- 文本重加权模块调整文本嵌入权重

### 关键设计

**运动强度估计**：使用 SSIM 衡量相邻帧的结构相似度来参数化运动强度：

$$\mathbf{I}(\mathbf{X}^n) = \frac{1}{n} \sum_{i=0}^{n-2} \text{SSIM}(\mathbf{x}^i, \mathbf{x}^{i+1})$$

将训练数据的运动强度划分为 1-10 共 10 个等级，转化为 1 通道嵌入图拼接到 UNet 输入中。用户推理时默认使用等级 5，也可自由调节。

**文本重加权**：在 CLIP 文本编码器之后添加 3 层 Transformer 编码器和线性投影层，预测每个 token 的权重（0-1 范围），再与对应文本嵌入相乘。模块自动学会强调运动相关词（如"waving"），抑制可能与参考图像冲突的内容描述词。

**先验反转**：推理时将参考图像的 DDIM 反转噪声加入初始噪声，提供外观先验：

$$\tilde{\mathbf{z}}_T^n = \alpha^n \cdot \text{Inv}(\mathbf{r}_0) + (1-\alpha^n) \cdot \mathbf{z}_T^n$$

### 损失函数

简单的 MSE 噪声预测损失，训练时以 0.5 概率丢弃文本实现 classifier-free guidance。

## 实验关键数据

### 主实验

用户研究评分（5 分制），与 VideoComposer、Pikalabs、GEN-2 对比：

| 方法 | 图像一致性 ↑ | 文本一致性 ↑ | 内容质量 ↑ | 运动质量 ↑ |
|------|:-:|:-:|:-:|:-:|
| VideoComposer | 2.8 | 3.5 | 3.6 | 3.6 |
| Pikalabs | 3.9 | 2.7 | **4.6** | 3.1 |
| GEN-2 | 3.7 | 2.5 | **4.8** | 3.3 |
| **LivePhoto** | 3.6 | **4.7** | 3.7 | **3.9** |

LivePhoto 在文本一致性和运动质量上显著领先，验证了运动控制能力。

### 消融实验

图像内容引导模块的逐步消融（WebVID 验证集）：

| 方法 | DINO Score ↑ | CLIP Score ↑ |
|------|:-:|:-:|
| 仅参考潜表示 | 82.3 | 91.7 |
| + 内容编码器 | 85.9 | 93.2 |
| + 先验反转 | **90.8** | **95.2** |

新模块的消融：

| 方法 | DINO Score ↑ | CLIP Score ↑ |
|------|:-:|:-:|
| LivePhoto（完整） | **90.8** | **95.2** |
| 去掉运动强度引导 | 90.3 | 94.8 |
| 去掉文本重加权 | 90.1 | 93.9 |

### 关键发现

- 运动强度引导解决了"静止或过度运动"的二元困境，提供连续可控性
- 文本重加权有效区分运动描述和内容描述，避免内容冲突削弱运动控制
- 模型可跨域泛化：动物、人类、卡通、自然风景均有效
- 可生成"凭空创造"内容（如向空杯中倒水、模拟闪电）

## 亮点与洞察

- **问题分析精准**：对文本无法控制运动的根因（歧义性和内容冲突）分析透彻
- **运动强度参数化**：SSIM 量化运动强度并离散化为 10 级的设计简洁实用
- **文本重加权可解释**：模块自动学会强调动词和运动相关词
- **学术方法对标商业产品**：在文本一致性上显著超越 GEN-2 和 Pikalabs

## 局限性

- 基于 SD-1.5 实现，输出分辨率仅 256×256
- 训练数据为 WebVID，泛化到特定领域可能需要额外适配
- 运动强度过高（等级 10）可能产生运动模糊

## 评分

⭐⭐⭐⭐ 问题分析深入，运动强度和文本重加权设计巧妙，效果在文本控制维度显著领先

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Local Action-Guided Motion Diffusion Model for Text-to-Motion Generation](local_action-guided_motion_diffusion_model_for_text-to-motion_generation.md)
- [\[ECCV 2024\] MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model](motionlcm_real-time_controllable_motion_generation_via_latent_consistency_model.md)
- [\[ECCV 2024\] ReNoise: Real Image Inversion Through Iterative Noising](renoise_real_image_inversion_through_iterative_noising.md)
- [\[ECCV 2024\] M2D2M: Multi-Motion Generation from Text with Discrete Diffusion Models](m2d2m_multi-motion_generation_from_text_with_discrete_diffusion_models.md)
- [\[ECCV 2024\] Text2Place: Affordance-aware Text Guided Human Placement](text2place_affordance-aware_text_guided_human_placement.md)

</div>

<!-- RELATED:END -->
