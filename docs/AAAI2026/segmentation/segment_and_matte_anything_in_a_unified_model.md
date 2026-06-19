---
title: >-
  [论文解读] Segment and Matte Anything in a Unified Model (SAMA)
description: >-
  [AAAI 2026][语义分割][SAM扩展] 提出SAMA——一种SAM的轻量级扩展框架，通过多视图局部编码器(MVLE)捕获细粒度局部特征、局部化适配器(Local-Adapter)将局部细节注入解码过程，以及双任务预测头，仅增加1.8%参数即可在统一模型中同时实现高质量交互式分割和Alpha Matting，在DIS-5K和多个Matting基准上达到SOTA。
tags:
  - "AAAI 2026"
  - "语义分割"
  - "SAM扩展"
  - "统一分割与抠图"
  - "多视图局部编码器"
  - "局部化适配器"
  - "交互式分割"
  - "Alpha Matting"
---

# Segment and Matte Anything in a Unified Model (SAMA)

**会议**: AAAI 2026  
**arXiv**: [2601.12147](https://arxiv.org/abs/2601.12147)  
**作者**: Zezhong Fan, Xiaohan Li, Topojoy Biswas, Kaushiki Nag, Kannan Achan  
**领域**: 图像分割 / 图像抠图  
**关键词**: SAM扩展, 统一分割与抠图, 多视图局部编码器, 局部化适配器, 交互式分割, Alpha Matting  

## 一句话总结
提出SAMA——一种SAM的轻量级扩展框架，通过多视图局部编码器(MVLE)捕获细粒度局部特征、局部化适配器(Local-Adapter)将局部细节注入解码过程，以及双任务预测头，仅增加1.8%参数即可在统一模型中同时实现高质量交互式分割和Alpha Matting，在DIS-5K和多个Matting基准上达到SOTA。

## 研究背景与动机

精确的目标分割是计算机视觉核心任务，涵盖语义/实例分割（为每个像素分配类别）和自然图像抠图（生成连续Alpha Matte以捕获半透明边界，如头发、玻璃等）。SAM在分割领域具有里程碑意义，经过10亿+mask训练后展现出优秀的零样本泛化能力。然而，SAM的原始mask常常缺乏紧贴目标的边界和亚像素精度。

现有改进方案如HQ-SAM、DIS-SAM、Pi-SAM等虽然提升了分割质量，但存在两个关键挑战：

**细粒度感知不足**：SAM作为交互式分割模型，难以捕获目标物体的精细结构

**高分辨率细节集成困难**：在解码过程中融合高分辨率细节的同时保持零样本泛化能力非常困难

另一方面，交互式抠图（通过稀疏用户引导估计精确Alpha Matte）虽然能实现优秀的边界细节，但缺乏目标级推理能力。近期研究发现**分割与抠图之间存在强相关性**：分割提供全局目标线索，抠图提供局部边界精度。利用这种协同效应构建统一模型的潜力尚未被充分挖掘。

## 方法详解

### 整体架构
SAMA在冻结SAM参数的基础上，引入三个轻量级组件：
- **多视图局部编码器(MVLE)**：从多个局部视图提取高分辨率特征
- **局部化适配器(Local-Adapter)**：将局部特征注入SAM解码过程
- **双任务预测头**：分别用于分割和抠图的输出

整体流程：将输入图像视为全局视图，同时裁剪为4个非重叠局部patch作为局部视图。全局和局部视图分别通过SAM编码器得到特征图，随后通过MVLE融合、Local-Adapter精化、预测头输出。

### 多视图局部编码器 (MVLE)
SAM仅使用单一全局表示，限制了模型捕获细粒度视觉细节的能力。MVLE的设计灵感来自人类视觉系统——区分远景全局上下文和近景局部细节：

1. 将输入图像均匀裁剪为4个非重叠局部patch
2. 每个patch上采样回原始分辨率，通过同一编码器提取高分辨率局部特征图
3. 对全局特征进行多尺度平均池化（接受域为4/8/16），获得多尺度上下文表示
4. 在每个空间区域内，以局部特征为Query、全局池化特征为Key/Value进行交叉注意力对齐

### 局部化适配器 (Local-Adapter)
Local-Adapter负责将高分辨率局部特征注入SAM解码器，包含三个步骤：

1. **第一层交叉注意力**：将MVLE输出的局部特征与编码器早期层特征通过残差连接融合作为Key/Value，解码器输出作为Query，实现局部-全局信息集成
2. **第二层交叉注意力**：交换Query/Key-Value角色（受GLIP和GroundingDINO启发），实现双向交互，使适配器同时具备全局和局部感知能力
3. **置信度图融合**：通过1×1卷积+Sigmoid生成置信度图C，与交叉注意力输出逐元素相乘后加回全局特征。该机制保护SAM的零样本泛化能力，防止过拟合和灾难性遗忘

### 双任务预测头
- 引入两个可学习的SAMA token（分割token和抠图token），取代SAM原始输出token
- 两个轻量级任务专用预测头分别处理分割和抠图，通过插值上采样 + 卷积层（BN + GeLU）重建细节
- 实现同时生成高分辨率分割mask和Alpha Matte

### 训练策略
- **数据**：分割任务使用DIS-5K和ThinObject-5K（高质量标注）；抠图任务使用AIM和AIM-500
- **冻结SAM骨干**，仅训练新增模块；训练分割时冻结抠图头，反之亦然
- **分割损失**：$\mathcal{L}_{seg} = \mathcal{L}_{BCE} + \mathcal{L}_{IoU} + \mathcal{L}_{SSIM}$
- **抠图损失**：$\mathcal{L}_{matting} = \mathcal{L}_{l_1} + \mathcal{L}_{SSIM} + \mathcal{L}_{Grad} + \mathcal{L}_{Laplacian}$

## 实验结果

### 实验一：DIS-5K分割基准（Table 1）

在极细粒度分割数据集DIS-5K上与SAM系列及专用分割模型对比：

| 方法 | DIS-VD $F_\beta^{max}$↑ | DIS-VD MAE↓ | DIS-VD $S_\alpha$↑ | DIS-TE(All) $F_\beta^{max}$↑ | DIS-TE(All) MAE↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| SAM | 0.835 | 0.069 | 0.808 | 0.773 | 0.096 |
| HQ-SAM | 0.851 | 0.045 | 0.848 | 0.859 | 0.045 |
| Pi-SAM | 0.883 | 0.035 | 0.889 | 0.893 | 0.033 |
| DIS-SAM | 0.920 | 0.031 | 0.909 | 0.917 | 0.029 |
| BiRefNet | 0.891 | 0.038 | 0.898 | 0.896 | 0.035 |
| **SAMA** | **0.942** | **0.021** | **0.930** | **0.926** | **0.026** |

**发现**：SAMA在所有SAM-based模型中全面领先，在DIS-VD上$F_\beta^{max}$达到0.942，MAE降至0.021。即使与专门为DIS任务训练的BiRefNet相比，SAMA同样具有竞争力。

### 实验二：Matting基准（Table 2）

在Composition-1K和Distinction-646上与trimap-based和trimap-free方法对比：

| 方法 | 类型 | Comp-1K SAD↓ | Comp-1K MSE↓ | Dist-646 SAD↓ | Dist-646 MSE↓ |
|------|:---:|:---:|:---:|:---:|:---:|
| VITMatte | trimap-based | 21.5 | 3.3 | 21.22 | 2.1 |
| MODNet | trimap-free | 47.1 | 12.3 | 41.7 | 9.0 |
| MFC-Net | trimap-free | 35.6 | 8.7 | 34.5 | 7.8 |
| **SAMA** | **trimap-free** | **22.8** | **2.9** | **22.4** | **2.2** |

**发现**：SAMA在trimap-free方法中达到SOTA，大幅领先MODNet和MFC-Net。值得注意的是，不依赖trimap输入的SAMA与最好的trimap-based方法VITMatte性能接近（Comp-1K MSE甚至更低：2.9 vs 3.3），展现了强大的泛化能力。

### 消融实验（Table 3-5）
- **MVLE+Local-Adapter**：两者缺一不可。在DIS-VD上，基线$F_\beta^{max}$为0.872；仅加MVLE为0.882，仅加L-A为0.893；两者结合达0.942，提升8%
- **多任务学习**：联合训练优于单独训练。联合训练使抠图SAD从62.70降至25.69（RefMatte-RW100），matting数据提供的边界细节反过来提升了分割精度

## 亮点与创新
- **统一框架首创**：首个基于SAM的同时执行交互式分割和抠图的模型，仅增加1.8%参数
- **MVLE多视图策略**：将输入裁剪为局部patch再上采样，模拟人眼近远景差异化处理，有效增强细粒度感知
- **置信度图保护机制**：Local-Adapter通过置信度门控融合局部信息，巧妙平衡了精度提升与零样本泛化的矛盾
- **任务互补增益**：实验验证分割与抠图联合训练相互促进——分割提供全局语义，抠图提供边界精度

## 局限性
- 论文仅在图像上实验，未扩展到视频分割/抠图场景
- 多视图裁剪+分别编码会增加推理延迟（虽然论文称开销边际，但4倍编码器前向传播不可忽视）
- 训练数据有限（DIS-5K和AIM），未验证在更大规模数据上的表现
- 论文未讨论与SAM2/SAM3等后续版本的兼容性

## 相关工作
- **交互式分割**：SAM、HQ-SAM、Pi-SAM、DIS-SAM等SAM变体改进分割精度或扩展功能
- **图像抠图**：trimap-based方法(DIM, VITMatte)和trimap-free方法(MODNet, MAM, MatAny)
- **分割-抠图统一**：先前工作发现两个任务在结构上高度相关(Wang & Cohen 2005; Zheng et al. 2024)，但缺乏统一建模

## 总结评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 综合推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[AAAI 2026\] Segment Anything Across Shots: A Method and Benchmark](segment_anything_across_shots_a_method_and_benchmark.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [\[CVPR 2026\] Attack for Defense: Adversarial Agents for Point Prompt Optimization Empowering Segment Anything Model](../../CVPR2026/segmentation/attack_for_defense_adversarial_agents_for_point_prompt_optimization_empowering_s.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)

</div>

<!-- RELATED:END -->
