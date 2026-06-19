---
title: >-
  [论文解读] ZIM: Zero-Shot Image Matting for Anything
description: >-
  [ICCV 2025][语义分割][图像抠图] 提出ZIM——一种零样本图像抠图模型，通过标签转换器将SA1B分割标签转为精细抠图标签构建SA1B-Matte数据集，并设计层次像素解码器和提示感知遮罩注意力机制，在保持零样本泛化能力的同时实现微观级精细抠图。 SAM在零样本分割领域取得突破，但其生成的mask缺乏精细边缘质量…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "图像抠图"
  - "零样本"
  - "SAM"
  - "标签转换"
  - "层次解码器"
---

# ZIM: Zero-Shot Image Matting for Anything

**会议**: ICCV 2025  
**arXiv**: [2411.00626](https://arxiv.org/abs/2411.00626)  
**代码**: [https://naver-ai.github.io/ZIM](https://naver-ai.github.io/ZIM)  
**领域**: 图像分割  
**关键词**: 图像抠图, 零样本, SAM, 标签转换, 层次解码器

## 一句话总结

提出ZIM——一种零样本图像抠图模型，通过标签转换器将SA1B分割标签转为精细抠图标签构建SA1B-Matte数据集，并设计层次像素解码器和提示感知遮罩注意力机制，在保持零样本泛化能力的同时实现微观级精细抠图。

## 研究背景与动机

SAM在零样本分割领域取得突破，但其生成的mask缺乏精细边缘质量（如头发丝、树枝等细节）。现有将SAM扩展到抠图的方法（Matte-Any、Matting-Any）依赖公开抠图数据集微调，但这些数据集**仅包含宏观级标签**（如整个人像），导致微调后模型在微观级粒度上丧失零样本能力——给定微观级提示时仍输出宏观级结果（灾难性遗忘）。

核心矛盾：
- **SAM**：零样本能力强，但mask粗糙、存在棋盘格伪影
- **现有抠图模型**：精细度高，但零样本泛化差
- **数据瓶颈**：大规模微观级抠图标注成本极高

本文的关键洞察是：可以通过**标签转换器**将SA1B中海量的微观级分割标签自动转换为抠图标签，无需人工标注即可获得大规模训练数据。

## 方法详解

### 整体框架

ZIM基于SAM架构，包含四个组件：(1) 图像编码器（ViT-B, stride 16）；(2) 提示编码器；(3) Transformer解码器（token-to-image和image-to-token交叉注意力）；(4) **改进的层次像素解码器**。两个核心贡献分别对应**数据构建**和**网络架构**。

### 关键设计

1. **标签转换器（Label Converter）**：基于MGMatting + Hiera-base-plus骨干网络。输入图像和分割标签，输出精细抠图标签。训练数据来自6个公开抠图数据集（共20,591自然图像+118,749合成图像），通过阈值化、降分辨率、高斯模糊、膨胀/腐蚀等操作从matte标签反推粗糙分割标签作为输入。

   两个关键策略解决训练难题：
    - **空间泛化增强（SGA）**：随机裁剪分割标签和对应matte标签中的相同区域，迫使转换器适应不完整/异常的输入模式，提升对微观级分割标签的泛化能力。
    - **选择性转换学习（STL）**：非所有物体都需要精细抠图（如汽车、桌子）。从ADE20K收集粗粒度物体mask（187,063个），训练时其ground-truth matte等于原始分割标签（即不做转换），让模型学会**选择性**地仅对需要精细处理的物体进行转换。

   训练损失：$L = L_{l1} + \lambda L_{grad}$，其中 $L_{grad}$ 为梯度损失。

2. **层次像素解码器（Hierarchical Pixel Decoder）**：SAM原始解码器仅含两层转置卷积（stride 4），易产生棋盘格伪影。新解码器采用多分辨率特征金字塔设计，输入图像生成stride 2/4/8三个层级的特征图，图像嵌入依次上采样并与对应分辨率特征拼接。最终输出stride 2的高分辨率特征图。**仅增加10ms推理时间**。

3. **提示感知遮罩注意力（Prompt-Aware Masked Attention）**：

    - 框提示：生成二值注意力mask $\mathcal{M}^b \in \{0, -\infty\}$，强制模型聚焦框内区域
    - 点提示：生成基于2D高斯分布的软注意力mask $\mathcal{M}^p \in [0,1]$（标准差 $\sigma=21$）
    - 仅应用于token-to-image交叉注意力层（实验表明应用于image-to-token会干扰全局特征捕获）

### 损失函数 / 训练策略

- SA1B-Matte数据集：转换SA1B（约2.2M标签，1%子集）用于训练
- 训练损失同标签转换器：$L = L_{l1} + \lambda L_{grad}$（$\lambda=10$）
- AdamW优化器，lr=1e-5，cosine衰减，500K迭代
- 从SAM预训练权重微调

## 实验关键数据

### 主实验（MicroMat-3K测试集，框提示）

| 方法 | 提示 | Fine-grained SAD↓ | Fine-grained MSE↓ | Fine-grained Grad↓ | Coarse SAD↓ | Coarse MSE↓ |
|------|------|------|------|------|------|------|
| SAM | box | 36.086 | 11.057 | 14.867 | 3.516 | 1.044 |
| HQ-SAM | box | 124.262 | 42.457 | 13.673 | 8.458 | 2.733 |
| Matte-Any | box | 34.661 | 9.746 | 7.021 | 6.950 | 1.983 |
| Matting-Any | box | 246.214 | 68.372 | 19.185 | 109.639 | 23.780 |
| **ZIM (ours)** | **box** | **9.961** | **1.893** | **4.813** | **1.860** | **0.448** |

*ZIM在所有指标上大幅领先，MSE较SAM降低83%，较Matte-Any降低81%。*

### 消融实验（ZIM组件分析，框提示）

| 注意力 | 解码器 | Fine-grained SAD↓ | Fine MSE↓ | Fine Grad↓ | Coarse SAD↓ | Coarse MSE↓ |
|--------|--------|------|------|------|------|------|
| ✗ | ✗ | 13.623 | 2.718 | 6.516 | 2.071 | 0.474 |
| ✓ | ✗ | 13.198 | 2.504 | 6.445 | 2.049 | 0.471 |
| ✗ | ✓ | 11.074 | 2.094 | 5.401 | 2.069 | 0.487 |
| **✓** | **✓** | **9.961** | **1.893** | **4.813** | **1.860** | **0.448** |

*两个组件互补：层次解码器主要降低Grad误差（减少伪影），注意力机制进一步提升整体精度。*

### 关键发现

- **下游可迁移性**：用ZIM替换SAM在Matte-Any、HQ-SAM、Inpainting Anything、医学图像分割、3D分割等多个下游框架中均获得显著提升
- **训练数据影响**：用公开抠图数据训练的ZIM在MicroMat-3K上MSE从1.893暴增到38.332，验证了微观级数据的必要性
- **域偏移问题**：ZIM在传统抠图测试集（全身人像）上框提示性能较差（SAM固有的提示歧义问题），但通过密集多点提示可超越所有现有方法

## 亮点与洞察

- **数据工程创新**：标签转换器将分割标签→抠图标签的思路简洁有效，SGA和STL两个策略设计精巧
- **轻量改进大收益**：层次解码器仅增加10ms延迟，却解决了SAM长期存在的棋盘格问题
- **MicroMat-3K测试集**：3000张高质量微观级抠图标注，填补了零样本抠图评估空白
- **实用性强**：支持点/框/文本/涂鸦多种提示模式

## 局限与展望

- 继承SAM在模糊提示下的歧义问题（整体/部分歧义）
- 无法处理透明物体（如玻璃、火焰）的透明度预测
- 标签转换器的质量上限取决于源抠图数据集的覆盖范围
- 未探索更大backbone（ViT-H）的潜力

## 相关工作与启发

- 标签转换思路可推广到其他密集预测任务（如将粗标注升级为精细标注）
- 选择性转换学习（STL）思想适用于任何需要区分"需要/不需要细化"的场景
- 层次解码器设计可作为SAM系列模型的即插即用改进

## 评分

- 新颖性: ⭐⭐⭐⭐ (标签转换+数据集构建思路新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (23个数据集零样本评估+多下游任务+详尽消融)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图表丰富)
- 价值: ⭐⭐⭐⭐⭐ (实际应用价值极高，填补零样本抠图空白)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Robust 3D Shape Reconstruction in Zero-Shot from a Single Image in the Wild](../../CVPR2025/segmentation/robust_3d_shape_reconstruction_in_zero-shot_from_a_single_image_in_the_wild.md)
- [\[CVPR 2026\] DSS: Discover, Segment, and Select for Zero-shot Camouflaged Object Segmentation](../../CVPR2026/segmentation/discover_segment_and_select_a_progressive_mechanism_for_zero-shot_camouflaged_ob.md)
- [\[ECCV 2024\] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](../../ECCV2024/segmentation/efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)
- [\[ICCV 2025\] Object-level Correlation for Few-Shot Segmentation](object-level_correlation_for_few-shot_segmentation.md)
- [\[ICCV 2025\] MOVE: Motion-Guided Few-Shot Video Object Segmentation](move_motion-guided_few-shot_video_object_segmentation.md)

</div>

<!-- RELATED:END -->
