---
title: >-
  [论文解读] Rethinking Image Super-Resolution from Training Data Perspectives
description: >-
  [ECCV 2024][图像恢复][超分辨率] 从训练数据角度重新思考图像超分辨率，提出自动化数据评估流水线构建 DiverSeg 数据集（低分辨率但高质量、目标多样的图像），证明在该数据集上训练的 SR 模型可以超越使用高分辨率数据集（DF2K、LSDIR）训练的模型。 图像超分辨率（SR）领域在过去十年取得了巨大进展…
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "超分辨率"
  - "训练数据"
  - "数据集构建"
  - "图像质量评估"
  - "目标多样性"
---

# Rethinking Image Super-Resolution from Training Data Perspectives

**会议**: ECCV 2024  
**arXiv**: [2409.00768](https://arxiv.org/abs/2409.00768)  
**代码**: [https://github.com/gohtanii/DiverSeg-dataset](https://github.com/gohtanii/DiverSeg-dataset)  
**领域**: 图像恢复  
**关键词**: 超分辨率, 训练数据, 数据集构建, 图像质量评估, 目标多样性

## 一句话总结

从训练数据角度重新思考图像超分辨率，提出自动化数据评估流水线构建 DiverSeg 数据集（低分辨率但高质量、目标多样的图像），证明在该数据集上训练的 SR 模型可以超越使用高分辨率数据集（DF2K、LSDIR）训练的模型。

## 研究背景与动机

图像超分辨率（SR）领域在过去十年取得了巨大进展，但主要集中在网络架构的改进上。训练数据方面，传统依赖 DIV2K、Flickr2K 等高分辨率数据集（合称 DF2K），近年来 LSDIR 进一步扩展到 84,991 张高分辨率图像。

现有数据集构建的两个核心标准：

**分辨率与质量**：要求 HD/2K/4K 高分辨率，手动排除压缩伪影

**多样性**：包含多种场景、光照、纹理

**核心矛盾**：收集无压缩的高分辨率图像既困难又昂贵，导致数据集难以大规模扩展。ImageNet 有 128 万张图像但包含低分辨率和 JPEG 压缩图像。

**关键问题**：训练数据到底需要什么条件？高分辨率是否真的必要？

**核心发现**：三个因素正向影响 SR 性能：(i) 低压缩伪影、(ii) 图像内高多样性（更多目标）、(iii) 大规模数据集。低分辨率但满足这些条件的图像甚至优于高分辨率数据。

## 方法详解

### 整体框架

提出自动化图像评估流水线，从大规模低分辨率数据集（ImageNet、Places365、PASS）中筛选构建 SR 训练数据集 DiverSeg。流水线包含两个步骤：**源选择（Source Selection）** 和 **基于目标的过滤（Object-based Filtering）**。

### 关键设计

1. **源选择 — 基于 Blockiness 分布的质量估计**：

    - 核心思路：通过估计数据集的 JPEG 压缩质量来筛选高质量数据源
    - 使用 **blockiness 度量** 计算每张图像的块效应值 $B(x)$，通过核密度估计得到数据集级别的 blockiness 分布 $p_{X,q}(b)$
    - Blockiness 通过子带 DCT 系数的变化来量化：
    $B(x) = \sum_{i=1}^{P}\sum_{j=1}^{P}\left|\frac{\bar{V}_{crop}(i,j) - \bar{V}(i,j)}{\bar{V}(i,j)}\right|$
    - 将目标数据集的分布与参考数据集（DF2K）在不同质量级别下的基准分布通过 KL 散度进行比较，估计质量：
    $\hat{q}_X = \sum_{q \in S} q \frac{\exp(-D_{KL}(p_{X,1.0} || p_{Z,q}))}{\sum_{q' \in S} \exp(-D_{KL}(p_{X,1.0} || p_{Z,q'}))}$
    - 结果：ImageNet 质量 95.5%，Places365 质量 75.0%（被过滤掉），PASS 质量 99.8%
    - 设计动机：传统方法需要人工逐张检查图像质量，而本方法通过统计分布自动估计整个数据集的质量，无需人工评估

2. **基于目标的过滤 — 图像多样性筛选**：

    - 核心假设：包含更多目标区域的图像对 SR 训练更有效
    - 两种过滤方法：
        - **基于分割的过滤**：使用 SAM (ViT-H) 计算分割 mask 数量 $R(x)$，阈值 $\theta = 100$，从 ImageNet 中筛选出 260K 图像
        - **基于检测的过滤**：使用 Detic (ViT-B) 计算检测到的目标数 $R(x)$，阈值 $\theta = 18$，同样得到 260K 图像
    - 设计动机：手动质量评估时，评估者倾向于关注细节丰富的图像，这间接排除了目标数少的图像。本方法将这一隐含偏好显式化

3. **DiverSeg 数据集**：

    - DiverSeg-I：从 ImageNet 筛选的 259K 图像
    - DiverSeg-P：从 PASS 筛选的 267K 图像
    - DiverSeg-IP：两者联合的 527K 图像
    - 特点：低分辨率（平均 233K 像素 vs DF2K 的 2.8M），但高质量（低 blockiness）、高多样性（平均 146 个分割 mask vs DF2K 的 103）

### 损失函数 / 训练策略

按照各 SR 模型原始论文的训练设置进行训练（MSRResNet、EDSR、RCAN、SwinIR、HAT），仅替换训练数据集。使用标准的 L1 或 L2 损失。关键在于验证数据集质量的影响，而非改变训练策略。

## 实验关键数据

### 主实验

×4 SR 性能对比（PSNR/SSIM，5 个基准数据集）：

| 模型 | 训练数据 | Set5 | BSD100 | Urban100 | Manga109 |
|------|---------|------|--------|----------|----------|
| SwinIR | DF2K | 32.92/0.9044 | 27.92/0.7489 | 27.45/0.8254 | 32.03/0.9260 |
| SwinIR | LSDIR | 32.86/0.9036 | 27.92/0.7492 | 27.79/0.8331 | 31.98/0.9262 |
| SwinIR | **DiverSeg-I** | **32.97/0.9053** | **27.98/0.7508** | **27.83/0.8336** | **32.34/0.9283** |
| HAT | DF2K | 33.03/0.9056 | 27.99/0.7514 | 27.93/0.8365 | 32.44/0.9292 |
| HAT | LSDIR | 32.93/0.9053 | 28.01/0.7525 | 28.45/0.8469 | 32.57/0.9306 |
| HAT | **DiverSeg-I** | **33.15/0.9071** | **28.07/0.7542** | 28.51/0.8477 | **32.90/0.9325** |
| RCAN | DF2K | 32.50/0.8990 | 27.75/0.7421 | 26.73/0.8058 | 31.17/0.9165 |
| RCAN | **DiverSeg-I** | **32.70/0.9012** | **27.81/0.7443** | **27.03/0.8116** | **31.58/0.9210** |

### 消融实验

| 配置 | 关键指标(Urban100 PSNR) | 说明 |
|------|---------|------|
| ImageNet全量(1.28M) | 较低 | 含大量低质量压缩图像 |
| ImageNet过滤(260K) | 提升 | 去除低质量后性能改善 |
| DiverSeg-I(260K, θ=100) | **最优** | 高质量+高多样性双重过滤 |
| Places365 | 最差 | 质量仅75%，大量压缩伪影 |
| PASS(1.44M) | 好 | 质量99.8%，但多样性不足 |
| DiverSeg-P(267K) | 优于DF2K | 从PASS过滤后多样性提升 |
| 阈值θ=0(无过滤) | 基线 | 与全量数据相比 |
| 阈值θ=100 | 最优 | 目标多样性过滤的甜点 |

### 关键发现

- **高分辨率非必需**：低分辨率（~233K pixels）但高质量的数据集可以超越高分辨率数据集（DF2K ~2.8M pixels）
- **压缩伪影有害**：Places365 的低质量（75%）导致 SR 性能最差，验证了压缩伪影对 SR 训练的负面影响
- **目标多样性重要**：图像内更多目标→更多纹理和边缘→更好的 SR 性能
- **scale效应**：在同等质量下，更多图像通常带来更好性能
- **DiverSeg-I 在所有 5 个 SR 模型上均超越 DF2K**，且对 CNN 和 Transformer 架构均有效

## 亮点与洞察

1. **颠覆传统认知**：证明了高分辨率图像不是 SR 训练的必要条件，这极大地降低了构建 SR 数据集的门槛
2. **自动化流水线**：完全自动化的数据集构建流程，消除了耗时的人工质量评估
3. **Blockiness 质量估计**：巧妙地利用 KL 散度比较 blockiness 分布来估计数据集质量，无需逐张分析
4. **普适性强**：方法适用于所有测试的 SR 模型（3 个 CNN + 2 个 Transformer），不依赖特定架构
5. **实际意义大**：未来可以轻松从任何大规模图像数据集中自动筛选 SR 训练数据

## 局限与展望

- 目标过滤的阈值（θ=100/18）是手动设定的，未提供自动选择策略
- 仅在 ×4 SR 上验证，未覆盖 ×2、×8 等其他倍率
- SAM 和 Detic 模型本身的计算开销较大，处理百万级数据集需要可观资源
- 未分析图像语义类别的分布对 SR 的影响（如自然场景 vs 城市场景）
- 未考虑与 data augmentation 策略的交互效应

## 相关工作与启发

- 对 ImageNet 预训练在 SR 中的作用提供了更深入的理解（HAT 用 ImageNet 预训练，但可能不是最优的 SR 数据）
- blockiness 质量估计方法可推广到其他低级视觉任务的数据集构建中
- "目标多样性"这一发现可能暗示 SR 模型需要的是丰富的局部纹理模式而非全局高分辨率

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks](overcoming_distribution_mismatch_in_quantizing_image_super-resolution_networks.md)
- [\[ICCV 2025\] Outlier-Aware Post-Training Quantization for Image Super-Resolution](../../ICCV2025/image_restoration/outlier-aware_post-training_quantization_for_image_super-resolution.md)
- [\[ECCV 2024\] Accelerating Image Super-Resolution Networks with Pixel-Level Classification](accelerating_image_super-resolution_networks_with_pixel-level_classification.md)
- [\[ECCV 2024\] Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)
- [\[ECCV 2024\] Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution](pairwise_distance_distillation_for_unsupervised_real-world_image_super-resolutio.md)

</div>

<!-- RELATED:END -->
