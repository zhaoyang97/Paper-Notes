---
title: >-
  [论文解读] ReMamber: Referring Image Segmentation with Mamba Twister
description: >-
  [ECCV 2024][语义分割][指称图像分割] 本文首次将 Mamba 架构引入指称图像分割（RIS）任务，提出 Mamba Twister 模块通过通道扫描和空间扫描的"扭转"机制实现高效的视觉-语言特征融合，在 RefCOCO/RefCOCO+/G-Ref 三个基准上取得了超越 Transformer 方法的竞争性结果，同时保持线性计算复杂度。
tags:
  - "ECCV 2024"
  - "语义分割"
  - "指称图像分割"
  - "Mamba"
  - "多模态融合"
  - "状态空间模型"
  - "视觉-语言交互"
---

# ReMamber: Referring Image Segmentation with Mamba Twister

**会议**: ECCV 2024  
**arXiv**: [2403.17839](https://arxiv.org/abs/2403.17839)  
**代码**: [https://github.com/yyh-rain-song/ReMamber](https://github.com/yyh-rain-song/ReMamber)  
**领域**: 图像分割 / 多模态VLM  
**关键词**: 指称图像分割, Mamba, 多模态融合, 状态空间模型, 视觉-语言交互

## 一句话总结

本文首次将 Mamba 架构引入指称图像分割（RIS）任务，提出 Mamba Twister 模块通过通道扫描和空间扫描的"扭转"机制实现高效的视觉-语言特征融合，在 RefCOCO/RefCOCO+/G-Ref 三个基准上取得了超越 Transformer 方法的竞争性结果，同时保持线性计算复杂度。

## 研究背景与动机

**领域现状**：指称图像分割（RIS）需要根据自然语言描述定位并分割图像中的特定目标，是多模态理解的核心任务。目前主流方法基于 Transformer 架构，如 LAVT、CRIS、CGFormer 等，通过注意力机制建模视觉与语言之间的交互关系。

**现有痛点**：Transformer 的注意力机制具有二次方的计算和内存复杂度，在处理大尺寸图像和长文本描述时资源消耗严重。这在捕获长距离视觉-语言依赖关系时尤为明显，限制了模型在资源受限场景中的部署。

**核心矛盾**：Mamba 作为新兴的状态空间模型（SSM）提供了线性复杂度的替代方案，但直接将 Mamba 应用于多模态交互面临根本性挑战——Mamba 的扫描操作在不同通道的 token 之间交互不足，无法有效融合来自不同模态的信息。简单地将文本 token 拼接在图像 token 前面（In-context Conditioning）会导致文本信息在长图像序列处理过程中被稀释。

**本文目标** (1) 如何在 Mamba 架构中实现有效的多模态特征融合；(2) 如何克服 Mamba 通道间交互不足的固有缺陷；(3) 如何在保持线性复杂度的同时达到 Transformer 级别的分割精度。

**切入角度**：作者观察到 Mamba 中信息主要沿空间维度流动，通道间几乎独立。因此提出将多模态特征沿通道维度排列成"混合特征立方体"，然后通过沿通道和空间两个维度交替扫描来"扭转"这个立方体，强制不同模态的特征在扫描过程中交织融合。

**核心 idea**：通过构造视觉-文本-交互三路特征的混合立方体，并用 Channel Scan + Spatial Scan 的扭转机制替代 Transformer 的注意力融合，实现线性复杂度的多模态特征交互。

## 方法详解

### 整体框架

ReMamber 的输入是一张图像和一段文本描述，输出是对应目标的分割掩码。整体架构由多个 Mamba Twister Block 堆叠而成（4 个 block，VSS Layer 数量配置为 2-2-15-2），每个 block 包含若干 Visual State Space (VSS) Layer 和一个 Twisting Layer。VSS Layer 负责提取空间视觉特征，Twisting Layer 负责注入文本条件并实现多模态融合。各 block 输出的中间特征被送入解码器生成最终分割掩码。

### 关键设计

1. **Visual State Space (VSS) Layer**:

    - 功能：在空间维度上处理 2D 图像特征
    - 核心思路：由于 SSM 原本设计用于处理 1D 因果序列数据，直接应用于 2D 图像效果不佳。VSS Layer 采用 VMamba 提出的 Cross-Scan-Module（CSM），将图像 patch 展开为序列后沿四个方向扫描，确保所有像素的信息在特征变换过程中得到整合。相当于用 Mamba 替代了 ViT 中的自注意力层
    - 设计动机：保持 Mamba 的线性复杂度优势，同时适配 2D 图像的非因果特性

2. **混合特征立方体构造 (Hybrid Feature Cube)**:

    - 功能：显式构建图像与文本之间的细粒度对应关系
    - 核心思路：分别计算全局交互和局部交互。全局交互将文本序列池化为全局向量 $\mathbf{F}_t^{CLS}$ 并扩展到图像特征大小；局部交互通过矩阵乘法 $\mathbf{F}_c = \mathbf{F}_i \mathbf{W}_i \cdot (\mathbf{F}_t \mathbf{W}_t)^T$ 计算每个图像 patch 与每个文本 token 的相关性，再经卷积映射。最后将视觉特征、全局文本特征、局部交互特征沿通道拼接形成混合立方体 $\mathbf{F}_{cube} \in \mathbb{R}^{h \times w \times (C_i + C_t + C_c)}$
    - 设计动机：通过全局+局部两种交互确保每个视觉 token 同时感知文本的整体语义和细粒度词汇关联，避免单一表示的信息损失

3. **Twisting 扭转机制 (Channel Scan + Spatial Scan)**:

    - 功能：在混合特征立方体上促进模态内和模态间的信息交流
    - 核心思路：先进行 Channel Scan——将混合特征立方体沿通道维度视为有序序列，用 1D SSM 扫描促进跨通道（即跨模态）的信息融合；再进行 Spatial Scan——用 VSS Layer 沿空间维度进行 2D 扫描，在每个模态内部传播融合后的信息。公式为 $\mathbf{F}_{out} = \text{SSM}_{spatial}(\text{SSM}_{channel}(\mathbf{F}_{cube}))$。PCA 可视化显示 Channel Scan 将不同模态特征聚拢到文本分布附近，Spatial Scan 再将融合后的特征重新散开
    - 设计动机：解决 Mamba 通道间交互不足的核心缺陷。通过两步扫描的"扭转"，不同模态的信息在通道维度和空间维度上交替交织，实现深度融合

### 损失函数 / 训练策略

采用端到端训练，使用简单的卷积解码器。损失函数为标准的分割损失（BCE + Dice loss）。使用 ImageNet 预训练权重初始化 VMamba 部分，输入分辨率在 SOTA 对比实验中设为 480。

## 实验关键数据

### 主实验

| 数据集 | 指标(oIoU) | ReMamber | LAVT (Swin-B) | CRIS (CLIP-R101) | 提升 vs LAVT |
|--------|-----------|----------|---------------|------------------|-------------|
| RefCOCO val | oIoU | 74.54 | 72.73 | 70.47 | +1.81 |
| RefCOCO testA | oIoU | 76.74 | 75.82 | 73.18 | +0.92 |
| RefCOCO testB | oIoU | 70.89 | 68.79 | 66.10 | +2.10 |
| RefCOCO+ val | oIoU | 65.00 | 62.14 | 62.27 | +2.86 |
| RefCOCO+ testA | oIoU | 70.78 | 68.38 | 68.08 | +2.40 |
| G-Ref val | oIoU | 63.9 | 61.24 | 59.87 | +2.66 |

### 消融实验

| 融合方式 | RefCOCO val mIoU | RefCOCO+ val mIoU | G-Ref val mIoU |
|---------|-----------------|-------------------|---------------|
| Attention-based | 65.3 | 54.0 | 50.5 |
| In-Context | 69.1 | 58.4 | 54.8 |
| Norm Adaptation | 70.2 | 60.3 | 59.3 |
| Mamba Twister | **71.6** | **61.6** | **61.1** |

| 扫描配置 | RefCOCO val mIoU | 说明 |
|---------|-----------------|------|
| Channel Scan only | 62.3 | 仅通道扫描，严重掉点 |
| Spatial Scan only | 70.0 | 仅空间扫描，接近完整模型 |
| Parallel | 71.0 | 两者并行相加 |
| Channel→Spatial | **71.6** | 最优配置 |

### 关键发现

- Spatial Scan 对性能贡献最大，单独去掉 Channel Scan 后模型仍有较好表现，但去掉 Spatial Scan 后严重下降
- Attention-based Conditioning 在 Mamba 架构中表现最差，说明交叉注意力与 Mamba 的序列依赖特性存在根本冲突
- 全局和局部交互缺一不可，同时去掉全局特征时 RefCOCO val mIoU 从 71.6 降到 69.9
- 推理速度和训练内存均优于同等规模的 LAVT，尤其在高分辨率（1024）时优势明显

## 亮点与洞察

- **Twisting机制的巧妙设计**：通过将不同模态特征排列在通道维度上，再用两次不同维度的SSM扫描实现"扭转"融合，这种设计既简洁又有效，避免了在Mamba中引入注意力机制的额外开销。本质上是利用SSM的序列建模能力来做跨模态通信
- **Cross-Attention与Mamba不兼容的洞察**：实验揭示注意力机制与Mamba的序列依赖特性存在根本性矛盾，这对后续Mamba多模态研究有重要指导意义
- **扭转思路可迁移**：混合特征立方体 + 多维扫描的融合范式可以推广到其他多模态/多尺度特征融合任务，如视频理解中的时空融合、多传感器融合等

## 局限与展望

- 解码器结构简单，仅使用几层卷积，缺乏精细的多尺度特征聚合能力
- 文本编码器的选择和预训练方式未做深入探索，可能限制了语言理解的上界
- Channel Scan 的序列顺序是固定的通道排列，是否有更优的排序策略值得研究
- 未与大规模视觉-语言预训练模型（如 SAM、SEEM）进行对比

## 相关工作与启发

- **vs LAVT**: LAVT 使用 Swin Transformer + 语言感知融合模块，本文用 Mamba + Twisting 替代，在所有数据集上取得更优结果，且计算效率更高
- **vs CRIS**: CRIS 使用 CLIP 视觉编码器 + 文本到像素对比学习，本文不依赖 CLIP 预训练，使用 ImageNet 预训练的 VMamba 即可超越
- **vs CGFormer**: 基于 Transformer query 的框架，将分割视为 proposal 级分类问题，而本文是密集预测范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将Mamba用于RIS并提出有效的多模态融合方案，但整体框架设计相对直觉
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集全面对比，四种融合方式深入分析，消融实验充分，还有分布可视化和注意力图分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述合理，图示直观
- 价值: ⭐⭐⭐⭐ 为Mamba在多模态任务中的应用开拓了新方向，提供了多种融合设计的比较分析

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](../../NeurIPS2025/segmentation/safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)
- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](../../ICCV2025/segmentation/latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](../../AAAI2026/segmentation/rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)
- [\[ICML 2025\] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [\[ECCV 2024\] Segmentation-Guided Layer-Wise Image Vectorization with Gradient Fills](segmentation-guided_layer-wise_image_vectorization_with_gradient_fills.md)

</div>

<!-- RELATED:END -->
