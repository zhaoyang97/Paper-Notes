---
title: >-
  [论文解读] Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models
description: >-
  [CVPR 2025][语义分割][概念瓶颈模型] 提出SALF-CBM，将任意视觉网络转化为空间感知的概念瓶颈模型，通过CLIP视觉提示生成空间化概念图，同时提供"在哪里"（热力图）和"是什么"（概念）的双重解释，在ImageNet上甚至超越原始backbone精度。 深度神经网络虽然达到了人类水平的性能…
tags:
  - "CVPR 2025"
  - "语义分割"
  - "概念瓶颈模型"
  - "空间可解释性"
  - "视觉提示"
  - "零样本分割"
  - "模型调试"
---

# Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models

**会议**: CVPR 2025  
**arXiv**: [2502.20134](https://arxiv.org/abs/2502.20134)  
**代码**: [https://itaybenou.github.io/show-and-tell/](https://itaybenou.github.io/show-and-tell/)  
**领域**: 分割/可解释AI  
**关键词**: 概念瓶颈模型, 空间可解释性, 视觉提示, 零样本分割, 模型调试

## 一句话总结

提出SALF-CBM，将任意视觉网络转化为空间感知的概念瓶颈模型，通过CLIP视觉提示生成空间化概念图，同时提供"在哪里"（热力图）和"是什么"（概念）的双重解释，在ImageNet上甚至超越原始backbone精度。

## 研究背景与动机

深度神经网络虽然达到了人类水平的性能，但缺乏像人类一样解释决策的能力——即同时说明"看到了什么"和"在哪里看到的"。

现有可解释AI方法存在二选一的局限：

1. **归因方法（热力图）**：如GradCAM、LRP等，能显示空间关注区域，但缺乏语义描述。高亮区域可能是模糊的，不知道模型"看到了什么概念"
2. **概念瓶颈模型（CBM）**：将特征投影到可解释的概念空间，但现有CBM都是全局性的——只能说"在整张图里看到了羽毛"，无法指出在哪里
3. **精度损失**：现有CBM的瓶颈层通常导致分类精度下降，限制了实际应用

核心问题：能否构建一个统一框架，同时提供空间定位和概念级解释，且不损失甚至提升分类精度？

## 方法详解

### 整体框架

给定预训练backbone，SALF-CBM通过四步转化：(1) GPT自动生成任务相关概念列表；(2) 用CLIP视觉提示计算空间化概念相似度矩阵 $P$；(3) 训练空间感知概念瓶颈层将features投影到概念图；(4) 在池化后的概念激活上训练稀疏分类层。

### 关键设计

**设计一：基于视觉提示的局部概念相似度**

- **功能**：为每张训练图像的每个空间位置计算与各概念的相似度
- **核心思路**：在图像上建立 $\tilde{H} \times \tilde{W}$ 的均匀网格，在每个网格位置画红色圆圈生成增强图像 $x_n^{(h,w)}$，用CLIP计算该增强图像与每个概念文本的余弦相似度：$P[n,m,h,w] = \frac{I_n^{(h,w)} \cdot T_m}{\|I_n^{(h,w)}\| \|T_m\|}$
- **设计动机**：CLIP的视觉提示特性（红色圆圈可引导CLIP关注特定区域）被巧妙利用来获取局部语义，无需任何额外标注。预计算一次即可

**设计二：空间感知概念瓶颈层**

- **功能**：将backbone的黑盒特征图投影到可解释的概念图空间
- **核心思路**：保留backbone特征的空间信息（不做全局池化），用单个 $1 \times 1$ 卷积层（$M$ 个输出通道）将特征图映射为概念图 $c(x) \in \mathbb{R}^{M \times \tilde{H} \times \tilde{W}}$，用立方余弦相似度损失与预计算的相似度矩阵 $P$ 对齐
- **设计动机**：传统CBM对特征做全局池化后投影，丢失了空间信息。保留空间维度使得概念图天然具有定位能力，且参数量与非空间CBM相同

**设计三：交互式模型探索与调试**

- **功能**：支持用户主动查询和干预模型决策
- **核心思路**：(a) "解释任何区域"：用户指定ROI（点/框/掩码），聚合该区域内的概念激活，展示top-k概念；(b) "局部干预"：用户可以在特定区域增强或抑制某个概念的激活（$c(x)[m] \leftarrow c(x)[m] + \beta I$），观察预测如何变化
- **设计动机**：ante-hoc可解释性的核心优势是允许干预。空间化概念图使局部干预成为可能，支持反事实分析和模型调试

### 损失函数

瓶颈层训练使用立方余弦相似度损失：

$$\mathcal{L}_{CBL} = -\sum_{m=1}^{M} \sum_{h,w} sim(q[m,h,w], p[m,h,w])$$

分类层使用交叉熵损失+弹性网络正则化（L1+L2）确保稀疏性。

## 实验关键数据

### 分类精度对比

| 方法 | 稀疏 | CUB-200 | Places365 | ImageNet |
|------|------|---------|-----------|----------|
| Standard Backbone | Yes | 75.96% | 38.46% | 74.35% |
| P-CBM | Yes | 59.60% | N/A | N/A |
| LF-CBM | Yes | 74.31% | 43.68% | 71.95% |
| **SALF-CBM** | **Yes** | **74.35%** | **46.73%** | **75.32%** |
| Standard Backbone | No | 76.70% | 48.56% | 76.13% |
| **SALF-CBM** | **No** | 76.21% | **49.38%** | **76.26%** |

### 零样本分割（ImageNet-Segmentation）

| 方法 | Pixel Acc.↑ | mIoU↑ | mAP↑ |
|------|------------|-------|------|
| GradCAM | 71.34% | 53.34% | 83.88% |
| FullGrad | 73.04% | 55.78% | 88.35% |
| **SALF-CBM** | **76.94%** | **58.30%** | 85.31% |

### 关键发现

1. SALF-CBM在ImageNet上以75.32%/76.26%（稀疏/非稀疏）超越原始backbone（74.35%/76.13%），证明空间化概念表示不仅不损失精度，还能提升
2. 零样本分割pixel accuracy比最佳归因方法（FullGrad）高+3.9%，mIoU高+2.52%
3. 稀疏与非稀疏SALF-CBM差异小于1%（ImageNet），说明少量概念即可捕获决策信息
4. 概念瓶颈层不引入额外可学习参数（相比非空间CBM），空间信息是"免费的"

## 亮点与洞察

1. **"Show and Tell"的统一框架**：首次在单一模型中同时提供空间和概念级解释，两者自然耦合
2. **CLIP视觉提示的创意应用**：用红色圆圈引导CLIP关注局部区域，无需分割模型即可获取空间化概念标注
3. **精度提升而非下降**：空间概念瓶颈层竟然能提升分类精度，打破了"可解释性需要牺牲精度"的传统认知

## 局限与展望

1. 视觉提示计算（红圆圈+CLIP推理）的预计算成本较高，尤其在高分辨率网格下
2. 概念列表依赖GPT生成，质量取决于提示设计和类别覆盖
3. 红色圆圈作为视觉提示可能在特定图像中引入artifact（如图像本身有红色物体）
4. 可以探索更高效的局部概念计算方式，如SAM+CLIP的组合

## 相关工作与启发

- **LF-CBM**：无标注概念瓶颈模型的先驱，本文在其基础上加入空间维度
- **CLIP + Visual Prompting**：视觉提示让CLIP关注局部区域的能力在此被创造性利用
- **CRAFT**：基于矩阵分解的概念归因方法，也提供空间定位但是后验的
- 启发：空间化概念表示可以作为下游Dense Prediction任务的中间表示

## 评分

⭐⭐⭐⭐ — 优雅地统一了空间归因和概念解释两大XAI范式，且实现了"可解释性提升精度"的理想结果。CLIP视觉提示的利用非常巧妙。局限在于预计算成本和对CLIP能力的依赖。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Deep Nets with Subsampling Layers Unwittingly Discard Useful Activations at Test-Time](../../ECCV2024/segmentation/deep_nets_with_subsampling_layers_unwittingly_discard_useful_activations_at_test.md)
- [\[CVPR 2025\] Comparative Evaluation of Traditional Methods and Deep Learning for Brain Glioma Imaging](comparative_evaluation_of_traditional_methods_and_deep_learning_for_brain_glioma.md)
- [\[CVPR 2026\] Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation](../../CVPR2026/segmentation/concept-aware_lora_for_domain-aligned_segmentation_dataset_generation.md)
- [\[CVPR 2025\] A Distractor-Aware Memory for Visual Object Tracking with SAM2](a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)
- [\[CVPR 2025\] EditAR: Unified Conditional Generation with Autoregressive Models](editar_unified_conditional_generation_with_autoregressive_models.md)

</div>

<!-- RELATED:END -->
