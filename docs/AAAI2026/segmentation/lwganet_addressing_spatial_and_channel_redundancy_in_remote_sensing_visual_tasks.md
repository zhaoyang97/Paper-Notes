---
title: >-
  [论文解读] LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention
description: >-
  [AAAI 2026 Oral][语义分割][lightweight backbone] 针对遥感图像中的空间冗余（大面积均质背景）和通道冗余（极端尺度变化导致单一特征空间低效）问题，提出 LWGANet 轻量化骨干，通过 Top-K 稀疏全局特征交互（TGFI）和异构分组注意力（LWGA）模块实现高效多尺度特征表示，在 12 个数据集 4 类遥感任务上达到 SOTA。
tags:
  - "AAAI 2026 Oral"
  - "语义分割"
  - "lightweight backbone"
  - "remote sensing"
  - "注意力机制"
  - "spatial redundancy"
  - "channel redundancy"
---

# LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention

**会议**: AAAI 2026 Oral  
**arXiv**: [2501.10040](https://arxiv.org/abs/2501.10040)  
**代码**: [GitHub](https://github.com/AeroVILab-AHU/LWGANet)  
**领域**: 图像分割  
**关键词**: lightweight backbone, remote sensing, grouped attention, spatial redundancy, channel redundancy

## 一句话总结
针对遥感图像中的空间冗余（大面积均质背景）和通道冗余（极端尺度变化导致单一特征空间低效）问题，提出 LWGANet 轻量化骨干，通过 Top-K 稀疏全局特征交互（TGFI）和异构分组注意力（LWGA）模块实现高效多尺度特征表示，在 12 个数据集 4 类遥感任务上达到 SOTA。

## 背景与动机

### 现有痛点

**现有痛点**：**领域现状**：遥感图像的深度学习分析面临两种固有冗余：(1) **空间冗余**——显著前景目标稀疏分布在大面积均质背景（道路、农田、海洋）中，密集计算将大量资源浪费在低语义价值的背景区域；(2) **通道冗余**——遥感图像中存在极端尺度变化，单一统一特征表示难以同时高效捕获细粒度纹理和宏观空间上下文，导致大量通道对当前尺度目标无关。

现有轻量化骨干（如 MobileNetV2、EfficientFormerV2）主要针对自然图像设计，采用均质分组策略（如 depthwise separable convolution），在所有通道分组上施加相同操作，无法有效应对遥感数据的通道冗余。卷积模型（FasterNet）局部表示强但缺乏全局上下文，Transformer 模型（EfficientFormerV2）全局建模强但抑制高频空间信息。

### 解决思路

**本文目标**：如何设计一个同时解决遥感图像空间冗余和通道冗余的轻量化骨干网络，在精度和效率之间取得最优平衡？

## 方法详解

### 整体框架
LWGANet 采用四阶段层次化架构，空间分辨率依次下降 4/8/16/32 倍。提供 L0/L1/L2 三个变体（stem 通道数 32/64/96），Block 数量 $[N_1, N_2, N_3, N_4] = [1,2,4,2]$（L0/L1）或 $[1,4,4,2]$（L2）。每个 LWGA Block 包含 LWGA 模块 + Channel MLP + 残差连接。

### 关键设计

**LWGA 模块（异构分组注意力）**  
将通道均分为 4 个非重叠路径 $\{\mathbf{X_1}, \mathbf{X_2}, \mathbf{X_3}, \mathbf{X_4}\}$，每组 $C/4$ 通道，分别路由到针对不同尺度优化的专用算子：
- **GPA（Gate Point Attention）**：$1{\times}1$ 卷积扩展到 $C$ 通道再恢复，sigmoid 生成逐点注意力，捕获细粒度细节
- **RLA（Regular Local Attention）**：$3{\times}3$ 卷积 + BN + 激活，利用卷积归纳偏置捕获局部纹理
- **SMA（Sparse Medium-range Attention）**：TGFI 降采样至 $H/3 \times W/3$，十字星形注意力（窗口 $L=11$）建模中程依赖，再插值恢复
- **SGA（Sparse Global Attention）**：Stage 1-2 用 $5{\times}5$ 分组卷积 + $7{\times}7$ 空洞卷积近似全局注意力；Stage 3 用 TGFI + 标准 4-head self-attention；Stage 4 直接在全特征图上做 dense self-attention

四路输出 concat 融合为 $\mathbf{Y} \in \mathbb{R}^{H \times W \times C}$。

**TGFI 模块（Top-K 全局特征交互）**  
解决空间冗余：(1) 将特征图划分为非重叠区域，从每个区域选取最显著的 token（Top-K），保存空间坐标 $\mathcal{P}_{loc}$；(2) 仅在稀疏采样的 token 集上进行交互；(3) 将增强的表示通过 $\mathcal{P}_{loc}$ 恢复到原始位置。

## 实验关键数据

**场景分类（NWPU/AID/UCM）**：


### 主实验

| 方法 | Params(M) | FLOPs(G) | NWPU Top-1 | AID | UCM |
|------|-----------|----------|------------|-----|-----|
| MobileNet V2 1.0x | 2.28 | 0.319 | 95.06 | 93.65 | 97.14 |
| EfficientFormerV2 S0 | 3.36 | 0.396 | 94.52 | 93.80 | 97.14 |
| **LWGANet-L0** | **1.72** | **0.186** | **95.49** | **94.60** | **98.57** |
| MobileViT S | 5.03 | 1.75 | 95.19 | 95.25 | 97.14 |
| **LWGANet-L2** | 13.0 | 1.87 | **96.17** | **95.45** | **98.57** |

**旋转目标检测（DOTA 1.0/1.5/DIOR-R）**：LWGANet-L2 以 29.2M 参数在三个数据集均值上达到 73.49 mAP，超过 PKINet-S（72.30）和 DecoupleNet-D2（72.09）。

**语义分割**：UAVid 上 mIoU 69.1（+1.3 vs ResNet18），LoveDA 上 mIoU 53.6（+1.2 vs UnetFormer）。

**变化检测**：A2Net-LWGANet-L0 以 2.91M 参数在 LEVIR/WHU/CDD/SYSU 四个数据集上均值 IoU 83.49，优于 A2Net（82.86）。

**推理速度**：LWGANet-L0 在 GPU 上 13,234 FPS，CPU 上 80 FPS，远超 EfficientFormerV2（1,299 GPU FPS）。

## 亮点与洞察
- 明确识别并同时解决遥感图像的空间冗余和通道冗余两个核心瓶颈
- 异构分组策略：不同通道组使用不同尺度算子，比均质分组更高效
- TGFI 的稀疏采样机制将全局建模复杂度从密集降为稀疏
- SGA 的渐进式策略（卷积近似→稀疏注意力→密集注意力）优雅适配不同阶段
- 12 个数据集 4 类任务的全面验证，覆盖分类/检测/分割/变化检测

## 局限与展望
- Top-K 选择策略基于激活值最大值，可能遗漏某些低对比度但重要的前景区域
- 通道四等分的分组策略较为固定，可探索自适应或 NAS 搜索的分组比例
- 仅在 ImageNet-1K 预训练，未探索遥感专用预训练的效果
- 未在更大分辨率（如 4K 遥感影像）上测试实际部署性能

## 相关工作与启发
与 MobileNetV2 等均质轻量化模型相比，LWGANet 通过异构分组显著提升了多尺度表示能力。与 FasterNet（纯 CNN）相比，在 NWPU 上高出 2.19% 且速度接近。与 EfficientFormerV2（混合架构）相比，精度和速度均大幅领先。本文的核心差异在于首次从"数据冗余"角度出发设计遥感专用轻量化骨干。

## 相关工作与启发
- 异构分组注意力的思路可推广到其他存在多尺度特征需求的任务（医学影像、工业检测）
- TGFI 的稀疏交互思想与 sparse attention、token pruning 等效率优化方向相关
- 渐进式全局建模策略（从近似到精确）可借鉴用于其他层次化架构设计

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[CVPR 2026\] ReAttnCLIP: Training-Free Open-Vocabulary Remote Sensing Image Segmentation via Re-defined Attention in CLIP](../../CVPR2026/segmentation/reattnclip_training-free_open-vocabulary_remote_sensing_image_segmentation_via_r.md)
- [\[AAAI 2026\] RSVG-ZeroOV: Exploring a Training-Free Framework for Zero-Shot Open-Vocabulary Visual Grounding in Remote Sensing Images](rsvg-zeroov_exploring_a_training-free_framework_for_zero-shot_open-vocabulary_vi.md)
- [\[AAAI 2026\] RS2-SAM2: Customized SAM2 for Referring Remote Sensing Image Segmentation](rs2-sam2_customized_sam2_for_referring_remote_sensing_image_segmentation.md)
- [\[CVPR 2026\] HySeg: Learning Generative Priors for Structure-Aware Remote Sensing Segmentation](../../CVPR2026/segmentation/hyseg_learning_generative_priors_for_structure-aware_remote_sensing_segmentation.md)

</div>

<!-- RELATED:END -->
