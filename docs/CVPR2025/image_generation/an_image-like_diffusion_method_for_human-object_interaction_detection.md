---
title: "HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection"
description: "提出HOI Image表示和多项式扩散方法用于人-物交互检测，HICO-DET full mAP达47.71，通过Slice Patchification和先验初始化大幅超越标准扩散"
tags:
  - CVPR2025
  - 图像生成
  - 扩散模型
  - HOI检测
  - 图像生成
---

# HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection

**会议**: CVPR 2025  
**机构**: Lancaster University  
**关键词**: 人物交互检测、扩散模型、多项式扩散、HOI Image  

## 研究背景与动机

人-物交互检测（Human-Object Interaction Detection, HOI Detection）是场景理解的核心任务，旨在从图像中检测 <人, 交互动作, 物体> 三元组。例如，"一个人在骑自行车"需要识别人、自行车以及"骑"这个交互关系。

传统HOI检测方法（如QPIC、CDN）通常基于Transformer解码器，使用一组可学习的query来预测HOI三元组。这种方法的问题是：query数量固定，难以处理交互数量变化较大的场景；且一次性预测缺乏迭代优化的能力。

近年来，扩散模型在生成任务中展现了卓越的迭代去噪能力。能否将扩散模型应用于HOI检测？现有尝试（如DiffHOI）直接在bounding box坐标上做扩散，但效果有限，因为：
1. HOI的核心不在于精确的框坐标，而在于**谁和谁在做什么**
2. 连续高斯扩散对类别型输出（如交互类型）不自然
3. 缺少利用检测器先验的机制

HOI-IDiff的核心创新在于：**将HOI三元组重新编码为一种"图像"，然后在这个图像上应用专门设计的多项式扩散。**

## 方法详解

### 核心创新1：HOI Image构造

将每个场景的所有HOI关系编码为一张 $H \times W \times 2$ 的概率图像：

$$I_{\text{HOI}}[h, w, :] = v_{\text{obj}}(h) \otimes m_{\text{int}}(w)$$

其中：
- $H$ = 场景中的人-物对数量
- $W$ = 交互类别数量（HICO-DET为117类）
- 通道0：物体类别概率 $v_{\text{obj}} \in \Delta^{|\mathcal{O}|}$（单纯形上的概率分布）
- 通道1：交互类型概率 $m_{\text{int}} \in \{0, 1\}^{|\mathcal{A}|}$（多标签二进制指示）

**直觉理解**：HOI Image的每一行对应一个人-物对，每一列对应一种交互，像素值表示该交互发生的概率。这种表示将结构化预测转换为了图像生成问题。

### 核心创新2：多项式扩散

标准高斯扩散对连续数据添加高斯噪声，但HOI Image的每个像素是概率值（和为1），高斯噪声会破坏这一约束。

**多项式扩散**的前向过程：

$$q(x_t | x_{t-1}) = \text{Cat}(x_t; (1 - \beta_t) x_{t-1} + \beta_t / K)$$

其中 $K$ 是类别数。关键差异：
- 系数是 $(1-\beta_k)$ 而不是 $\sqrt{1-\beta_k}$
- 噪声项是均匀分布 $1/K$ 而不是高斯分布
- 始终保持概率和为1

| 特性 | 高斯扩散 | 多项式扩散 |
|------|---------|----------|
| 数据类型 | 连续值 | 概率分布 |
| 噪声类型 | 高斯 $\mathcal{N}(0,1)$ | 均匀 $1/K$ |
| 前向系数 | $\sqrt{1-\beta_t}$ | $(1-\beta_t)$ |
| 概率约束 | 无 | 始终满足 $\sum=1$ |
| 终态 | $\mathcal{N}(0,I)$ | 均匀分布 |

### 核心创新3：Slice Patchification

传统ViT将图像切分为局部patch（如16×16），但HOI Image的语义结构不同——每一行是完整的人-物对信息，每一列是完整的交互类型信息。局部patch会破坏这种行-列语义。

**Slice Patchification**提出切片式分块：
- **水平切片**：$H$ 个宽为 $W$ 的行向量（每个切片是一个完整的人-物对）
- **垂直切片**：$W$ 个高为 $H$ 的列向量（每个切片是一个完整的交互类型）

两组切片分别送入Transformer处理后融合。这保证了行内和列内的全局依赖，同时通过交叉注意力建立行-列之间的关系。

### 核心创新4：检测器先验初始化

标准扩散从纯噪声开始去噪，但HOI检测可以利用目标检测器（如DETR）的输出作为先验：

$$x_T = (1 - \alpha) \cdot \text{Uniform} + \alpha \cdot \text{DetectorPrior}$$

检测器先验提供了初始的人-物配对猜测，大幅减少了去噪步数。

## 实验结果

### HICO-DET

| 方法 | Full mAP | Rare mAP | Non-Rare mAP |
|------|---------|---------|-------------|
| QPIC | 29.07 | 21.85 | 31.23 |
| CDN | 32.07 | 27.19 | 33.53 |
| GEN-VLKT | 33.75 | 29.25 | 35.10 |
| HOICLIP | 34.69 | 31.12 | 35.74 |
| 标准扩散 baseline | 42.50 | 40.12 | 43.21 |
| **HOI-IDiff** | **47.71** | **48.36** | **47.52** |

### V-COCO

| 方法 | Scenario 1 | Scenario 2 |
|------|-----------|-----------|
| QPIC | 58.8 | 61.0 |
| CDN | 63.9 | 65.9 |
| HOICLIP | 66.2 | 68.5 |
| **HOI-IDiff** | **73.4** | **76.1** |

### 消融实验

| 配置 | HICO-DET Full mAP |
|------|------------------|
| 标准高斯扩散 | 42.50 |
| + 多项式扩散 | 44.23 |
| + Slice Patchification | 45.89 |
| + 检测器先验 | 46.84 |
| + 全部优化 | **47.71** |

从42.50到47.71的逐步提升验证了每个组件的贡献。

## 方法分析

### 为什么Slice Patchification有效？

传统patch打断了HOI Image的行-列语义结构。例如，一个16×16的patch包含了16个人-物对的部分交互信息——既不完整表示任何一个人-物对，也不完整表示任何一种交互。Slice则保证了语义单元的完整性。

### 为什么多项式扩散比高斯扩散好？

HOI Image的像素是概率分布，高斯噪声会产生负值和非归一化值，需要额外的归一化步骤。多项式扩散在整个过程中保持概率约束，生成的中间结果都是有效的概率分布。

## 局限性

- HOI Image的大小随场景中人-物对数量变化，批处理需要padding
- 多项式扩散的去噪步数仍然较多（通常100步）
- 在密集交互场景（>50个人-物对）上的效率有待优化

## 总结

HOI-IDiff通过将HOI检测重新定义为"概率图像生成"问题，巧妙地利用了扩散模型的迭代优化能力。多项式扩散、Slice Patchification和检测器先验三大创新协同工作，在HICO-DET和V-COCO上均达到了新的SOTA。这种"将结构化预测转化为图像生成"的思路具有广泛的启发意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](../../ICCV2025/image_generation/scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](../../CVPR2026/image_generation/vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[CVPR 2025\] Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](mitigating_memorization_in_text-to-image_diffusion_via_region-aware_prompt_augme.md)
- [\[CVPR 2025\] Re-HOLD: Video Hand Object Interaction Reenactment via Adaptive Layout-instructed Diffusion Model](re-hold_video_hand_object_interaction_reenactment_via_adaptive_layout-instructed.md)

</div>

<!-- RELATED:END -->
