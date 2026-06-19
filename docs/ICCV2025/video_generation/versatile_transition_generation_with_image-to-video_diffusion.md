---
title: >-
  [论文解读] Versatile Transition Generation with Image-to-Video Diffusion
description: >-
  [ICCV 2025][视频生成][视频过渡生成] 本文提出VTG统一过渡视频生成框架，基于图像到视频扩散模型，通过插值初始化（噪声SLERP+LoRA插值+文本SLERP）、双向运动微调和DINOv2表征对齐正则化，在物体变形、运动预测、概念融合、场景过渡四类任务上实现平滑高保真过渡。 1. 领域现状： 过渡视频生成包括物…
tags:
  - "ICCV 2025"
  - "视频生成"
  - "视频过渡生成"
  - "图像变形"
  - "双向运动预测"
  - "LoRA插值"
  - "表征对齐正则化"
---

# Versatile Transition Generation with Image-to-Video Diffusion

**会议**: ICCV 2025  
**arXiv**: [2508.01698](https://arxiv.org/abs/2508.01698)  
**代码**: [项目主页](https://mwxely.github.io/projects/yang2025vtg/index)  
**领域**: 视频生成  
**关键词**: 视频过渡生成, 图像变形, 双向运动预测, LoRA插值, 表征对齐正则化

## 一句话总结
本文提出VTG统一过渡视频生成框架，基于图像到视频扩散模型，通过插值初始化（噪声SLERP+LoRA插值+文本SLERP）、双向运动微调和DINOv2表征对齐正则化，在物体变形、运动预测、概念融合、场景过渡四类任务上实现平滑高保真过渡。

## 研究背景与动机

1. **领域现状**: 过渡视频生成包括物体变形（DiffMorpher）、视频帧插值（RIFE等）和场景过渡（SEINE），但各方法针对特定任务，缺乏统一框架。
2. **现有痛点**: (1)图像变形方法（DiffMorpher等）生成不连续的静态图像而非时序连贯帧；(2)视频帧插值在内容差异大时产生非自然过渡；(3)现有框架要么只做形态变形+运动预测，要么只做场景过渡，没有统一方案。
3. **核心矛盾**: 高质量过渡需同时满足语义相似性、输入保真度、帧间平滑性和文本对齐四个标准。图像到视频扩散模型随机初始化潜变量导致帧间"闪烁"；仅支持前向运动预测导致正反输入不对称。
4. **本文目标**: 能否设计一个通用过渡生成器同时处理物体变形、概念融合、运动预测和场景过渡？
5. **切入角度**: 在I2V扩散模型基础上引入三个互补设计：插值初始化（处理大内容差异）、双向运动（消除方向不对称）、表征对齐（增强保真度）。
6. **核心 idea**: 通过球面插值噪声+LoRA融合+文本SLERP统一四类过渡任务，双向运动微调消除方向偏差。

## 方法详解

### 整体框架
基于DynamiCrafter预训练的I2V扩散模型。给定首帧$x^1$和末帧$x^N$及对应文本，VTG分推理和训练两部分：推理时通过DDIM反转获取两端点latent噪声并SLERP插值；训练时仅微调时序注意力层的value/output矩阵和MLP投影器（150个高质量视频）。

### 关键设计

1. **插值初始化（Interpolation-based Initialization）**:
    - 功能: 缓解随机噪声导致的突变，保持物体身份，处理大内容差异
    - 核心思路: 三重插值——(1)**噪声SLERP**: 对两端点DDIM反转得到$z_{t1}$和$z_{tN}$，使用球面线性插值$z_{tn} = \frac{\sin((1-\lambda)\phi)}{\sin\phi}z_{t1} + \frac{\sin(\lambda\phi)}{\sin\phi}z_{tN}$关联中间帧噪声，仅在早期去噪步注入。(2)**LoRA插值**: 对两端点分别训练LoRA $\Delta\theta_1, \Delta\theta_N$（仅200步），线性插值$\Delta\theta = (1-\lambda_{LoRA})\Delta\theta_1 + \lambda_{LoRA}\Delta\theta_N$融合语义。(3)**帧感知文本SLERP**: 对两端文本嵌入$c_1, c_N$做SLERP，$c_{\lambda} = \text{SLERP}(c_1, c_N, \lambda_{text})$实现逐帧文本条件过渡。
    - 设计动机: 线性插值在高斯latent中产生不太可能的范数，SLERP保持欧几里得范数和分布内采样。LoRA捕获图像扩散模型中缺失的高层语义。文本SLERP解决单一caption无法描述中间帧混合含义的问题。

2. **双向运动预测（Bidirectional Motion Prediction）**:
    - 功能: 消除I2V扩散模型正反输入顺序导致的质量不对称
    - 核心思路: 将时序自注意力图旋转180度实现注意力关系反转，同时反转噪声latent的时序维度。正向U-Net和反向U-Net分别预测前向和后向运动噪声。后向预测结果再次反转后与前向融合: $\epsilon_t = (1-\lambda_{BMP})\epsilon_{t,i} + \lambda_{BMP}\epsilon'_{t,N-i}$（$\lambda_{BMP}=0.5$）。仅微调时序注意力层的value和output矩阵。损失: $\mathcal{L}_{BMP} = \|\text{flip}(\epsilon_t) - \epsilon_{\theta_{w,o}}(z_{t'}, c, t, A'_{i,j})\|_2^2$。
    - 设计动机: I2V模型偏向与首帧相似（条件图像泄漏），且仅预训练前向运动。双向融合确保一致的运动路径。

3. **表征对齐正则化（Representation Alignment Regularization）**:
    - 功能: 增强生成过渡帧的保真度，减少模糊
    - 核心思路: 将中间扩散latent分帧patchify后通过可训练MLP投影对齐DINOv2特征。逐patch计算余弦相似度: $\mathcal{L}_{RAR} = -\sum_{n=1}^{N}\mathbb{E}[\frac{1}{P}\sum_{p=1}^{P}\text{sim}(y_*^{[p]}, y_\phi(h_t)^{[p]})]$。推理时丢弃DINOv2编码器和MLP。
    - 设计动机: 扩散latent固有地缺乏高频语义，DINOv2特征包含丰富的自监督语义信息。训练时蒸馏DINOv2特征到扩散过程中，推理零开销。

### 损失函数 / 训练策略
仅150个高质量视频轻量微调。BMP微调时序注意力V/O矩阵；RAR训练MLP投影器。AdamW优化，学习率1e-5，4张A100约20K迭代。LoRA训练每对输入仅需200步约85秒。DDIM采样50步。

## 实验关键数据

### 主实验

| 方法 | MorphBench FID↓ | MorphBench PPL↓ | TC-Bench TCR↑ | Smoothness↑ |
|--------|------|------|----------|------|
| DiffMorpher | 70.49 | 18.19 | 41.82 | — |
| SEINE | 82.03 | 47.72 | — | — |
| DynamiCrafter | 87.32 | 42.09 | — | — |
| TVG | 86.92 | 35.18 | — | — |
| **VTG (本文)** | **67.39** | **22.80** | **最优** | **最优** |

### 消融实验

| 配置 | FID↓ | PPL↓ | 说明 |
|------|------|------|------|
| Full VTG | 最优 | 最优 | 完整模型 |
| w/o Noise SLERP | 上升 | 上升 | 中间帧随机突变 |
| w/o LoRA Interpolation | 上升 | — | 语义融合不足 |
| w/o Text SLERP | 上升 | — | 无帧级文本条件 |
| w/o BMP | 上升 | 上升 | 正反方向不对称 |
| w/o RAR | 上升 | — | 细节模糊 |

### 关键发现
- 物体变形任务中VTG显著优于DiffMorpher（FID 67.39 vs 70.49），因为DiffMorpher缺少时序建模
- 概念融合中生成了合理的中间语义（如狮子色和狮子大小的卡车），而基线出现突变
- 双向运动权重$\lambda_{BMP}=0.5$即可有效消除方向偏差
- RAR在高频纹理场景（自行车辐条、织物纹理）中提升最为显著

## 亮点与洞察
- 四类过渡任务的统一定义和统一框架：物体变形/概念融合/运动预测/场景过渡
- 三重插值策略（噪声+LoRA+文本）逻辑互补：噪声层面的结构+LoRA层面的语义+文本层面的条件
- TransitBench基准数据集的构建：200对首末帧，首次为概念融合和场景过渡提供标准评估
- 推理时RAR零开销：DINOv2仅用于训练正则化

## 局限与展望
- LoRA训练需要为每对输入85秒，批量生成时开销可观
- 仅150个训练视频，运动多样性受限
- 基于DynamiCrafter（UNet架构），可迁移到更新的DiT架构
- TransitBench规模较小（200对），可大幅扩展

## 相关工作与启发
- **vs DiffMorpher**: 基于图像扩散的变形缺乏时序建模，产生不连续帧序列
- **vs SEINE**: 用随机掩码条件层做场景过渡但概念融合效果差
- **vs Generative Inbetweening**: 正反向噪声融合但忽略身份保持和大内容差异

## 评分
- 新颖性: ⭐⭐⭐⭐ 统一四类过渡任务的框架设计+三重插值策略
- 实验充分度: ⭐⭐⭐⭐ 四类任务各有基准+TransitBench新基准
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法组件逻辑连贯
- 价值: ⭐⭐⭐⭐ 过渡生成的统一范式，对视频编辑和电影制作有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Multi-identity Human Image Animation with Structural Video Diffusion](multi-identity_human_image_animation_with_structural_video_diffusion.md)
- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)
- [\[CVPR 2026\] Transition Matching Distillation for Fast Video Generation](../../CVPR2026/video_generation/transition_matching_distillation_for_fast_video_generation.md)
- [\[ICCV 2025\] TIP-I2V: A Million-Scale Real Text and Image Prompt Dataset for Image-to-Video Generation](tip-i2v_a_million-scale_real_text_and_image_prompt_dataset_for_image-to-video_ge.md)
- [\[ICCV 2025\] STiV: Scalable Text and Image Conditioned Video Generation](stiv_scalable_text_and_image_conditioned_video_generation.md)

</div>

<!-- RELATED:END -->
