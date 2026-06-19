---
title: >-
  [论文解读] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba
description: >-
  [ICCV 2025][语义分割][轻量级视觉骨干] 提出 TinyViM，一种基于频率解耦的轻量级卷积-Mamba 混合视觉骨干，通过拉普拉斯混合器将低频分量输入 Mamba 建模全局上下文、高频分量用深度卷积增强，配合频率斜坡 Inception 结构逐层调节频率配比，在分类/检测/分割任务上以 2-3 倍吞吐量超越现有 Mamba 模型。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "轻量级视觉骨干"
  - "Mamba"
  - "频率解耦"
  - "拉普拉斯金字塔"
  - "高低频分离"
---

# TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba

**会议**: ICCV 2025  
**arXiv**: [2411.17473](https://arxiv.org/abs/2411.17473)  
**代码**: [GitHub](https://github.com/xwmaxwma/TinyViM)  
**领域**: 图像分割  
**关键词**: 轻量级视觉骨干, Mamba, 频率解耦, 拉普拉斯金字塔, 高低频分离

## 一句话总结

提出 TinyViM，一种基于频率解耦的轻量级卷积-Mamba 混合视觉骨干，通过拉普拉斯混合器将低频分量输入 Mamba 建模全局上下文、高频分量用深度卷积增强，配合频率斜坡 Inception 结构逐层调节频率配比，在分类/检测/分割任务上以 2-3 倍吞吐量超越现有 Mamba 模型。

## 研究背景与动机

Mamba 凭借线性复杂度的全局建模能力在视觉领域备受关注，ViM、VMamba 等将其应用于图像分类并取得不错效果。然而，现有轻量级 Mamba 骨干（如 EfficientVMamba）在性能和效率上**无法与基于卷积或 Transformer 的轻量级方法竞争**。

作者通过频谱分析发现了一个关键现象：**在卷积-Mamba 混合架构下，Mamba 主要建模低频信息，同时抑制高频信息**。具体表现为：
- Mamba 块处理后，频谱中心的低频分量被增强
- 边缘和纹理等高频细节被削弱

这一观察引发两个推论：
1. 将所有频率分量都输入 Mamba 块是**低效**的——高频信息对 Mamba 而言是无用负担
2. 全阶段统一处理低频信息会**退化高频分量**，损害细粒度识别能力

核心动机：既然 Mamba 天然偏好低频，不如**显式解耦高低频**，只让 Mamba 处理低频（分辨率更小，计算更少），高频用高效卷积增强。

## 方法详解

### 整体框架

TinyViM 采用四阶段多尺度设计。每个阶段包含 Local Block（重参数化 3×3 卷积 + FFN）和 TinyViM Block（Laplace Mixer + FFN）。阶段间用 Patch Embedding 进行下采样和通道扩展。

### 关键设计一：频率解耦的定量验证

构建使用卷积和标准 Mamba（SS2D）的 baseline，并对比不同输入方式：

| 输入变体 | GMACs | 吞吐量 | Top-1 (%) |
|---------|:-----:|:-----:|:---------:|
| Baseline（全频率） | 0.96 | 1673 | 79.1 |
| 仅低频 | 0.93 | 2574 | 79.0 |
| 仅高频 | 0.96 | 1377 | 78.6 |
| 高+低并行 | 0.97 | 1509 | 79.1 |

仅低频输入的精度几乎无损（79.0 vs 79.1），但吞吐量提升 1.5 倍。这验证了频率解耦策略的有效性。

### 关键设计二：拉普拉斯混合器（Laplace Mixer）

对输入特征 $X \in \mathbb{R}^{H \times W \times D}$，沿通道维度按比例 $\alpha$ 分为低频输入 $X_l$ 和高频输入 $X_h$：

**低频分支**：通过拉普拉斯金字塔分解：
$$X_{ll} = \text{Pool}(X_l), \quad X_{lh} = X_l - \text{Upsample}(X_{ll})$$

低频分量 $X_{ll}$ 的分辨率为原始的 $\frac{1}{2}$，输入 SS2D（VMamba 的 2D 选择性扫描）获取全局上下文：
$$\hat{X}_{ll} = \text{SS2D}(X_{ll})$$

**高频分支**：将 $X_{lh}$（低频分支的高频残差）和 $X_h$ 拼接，用重参数化 3×3 深度卷积增强：
$$\hat{X}_{hh} = \text{Rep}_3(X_{hh})$$

最后元素级相加对应通道的高低频特征，经 1×1 卷积融合。

### 关键设计三：频率斜坡 Inception

基于两个认知：(1) 深层网络存在特征冗余；(2) 浅层需更多高频细节，深层需更多全局信息。

提出逐阶段调节分区系数 $\alpha$：浅层分配更多通道给高频分支（$\alpha$ 小），深层分配更多给低频分支（$\alpha$ 大）：

| 阶段 | $\alpha_1$ | $\alpha_2$ | $\alpha_3$ | $\alpha_4$ | Top-1 |
|------|:---:|:---:|:---:|:---:|:---:|
| 均等分配 | 0.5 | 0.5 | 0.5 | 0.5 | 79.0 |
| **斜坡分配** | **0.25** | **0.5** | **0.5** | **0.75** | **79.2** |

### 损失函数

标准分类损失（交叉熵），下游任务使用对应框架的默认损失。

## 实验

### ImageNet-1K 分类

| 模型 | 类型 | Param | GMACs | 吞吐量(im/s) | Top-1 (%) |
|------|------|-------|-------|:----------:|:---------:|
| SwiftFormer-S | CNN+ViT | 6.1M | 1.0 | 2626 | 78.5 |
| EfficientVMamba-T | Mamba | 6.1M | 1.0 | 1396 | 76.5 |
| **TinyViM-S** | **CNN+Mamba** | **5.6M** | **0.9** | **2563** | **79.2** |
| MobileOne-S4 | CNN | 14.8M | 3.0 | 1223 | 79.4 |
| EfficientVMamba-S | Mamba | 11M | 1.3 | 674 | 78.7 |
| **TinyViM-B** | **CNN+Mamba** | **11M** | **1.5** | **1851** | **81.2** |
| VMamba-T | Mamba | 30M | 4.9 | 383 | 82.6 |
| EfficientVMamba-B | Mamba | 33M | 4.0 | 580 | 81.8 |
| **TinyViM-L** | **CNN+Mamba** | **31.7M** | **4.7** | **843** | **83.3** |

TinyViM-S 以 5.6M 参数达到 79.2%，超过 EfficientVMamba-T 2.7%，吞吐量高 1.8 倍。

### COCO 检测与实例分割（Mask R-CNN）

| 骨干 | 吞吐量 | AP^box | AP^mask |
|------|:-----:|:------:|:-------:|
| EfficientVMamba-S | 104 | 39.3 | 36.7 |
| SwiftFormer-L1 | 174 | 41.2 | 38.1 |
| **TinyViM-B** | **180** | **42.3** | **38.7** |
| FastViT-SA24 | 93 | 42.0 | 38.0 |
| EfficientVMamba-B | 104 | 43.4 | 39.5 |
| **TinyViM-L** | **119** | **44.5** | **40.3** |

TinyViM-B 在 AP^box 上超过 SwiftFormer-L1 +1.1，吞吐量也更高。

### ADE20K 语义分割（Semantic FPN）

| 骨干 | mIoU |
|------|:----:|
| EfficientFormer-L1 | 38.9 |
| TinyViM-S | 38.9 |
| SwiftFormer-L1 | 41.1 |
| **TinyViM-B** | **41.9** |
| PoolFormer-S36 | 42.0 |
| **TinyViM-L** | **44.1** |

### 消融实验

**拉普拉斯卷积核大小**：

| kernel | Top-1 | 吞吐量 |
|--------|:-----:|:-----:|
| 3 | 79.0 | 2510 |
| 5 (axial) | 79.0 | 2598 |
| **7 (axial)** | **79.2** | **2563** |
| 9 (axial) | 79.2 | 2479 |

7×7 轴向卷积在精度和效率之间取得最佳平衡。

### 关键发现

1. 频率解耦是 Mamba 轻量化的关键——只用低频做隐状态传递不损精度但大幅提速
2. 频率斜坡 Inception 的渐进式频率分配优于均等分配，符合"浅层高频、深层低频"的认知
3. 重参数化对小模型无增益（TinyViM-S ±0），对大模型有帮助（TinyViM-L +0.2）
4. ERF 可视化显示 TinyViM 的感受野显著大于 MobileOne 和 SwiftFormer

## 亮点与洞察

- **频率视角的独特切入**：首次从频域分析 Mamba 的行为偏好，并据此设计专门架构
- **极致效率**：TinyViM-S 以 5.6M 参数/0.9 GMACs 达到 79.2%，在超轻量级赛道极具竞争力
- **吞吐量优势突出**：是同类 Mamba 模型的 2-3 倍，真正适用于实时部署

## 局限性

- 拉普拉斯金字塔分解增加了实现复杂度，非标准算子在部分硬件上可能无法充分优化
- 频率解耦假设 Mamba 总是偏好低频，在不同预训练设置下是否成立需进一步验证
- 仅在 V100 上测试吞吐量，在移动端/边缘设备的实际延迟未报告

## 相关工作

- 高效视觉骨干：MobileNet 系列、EfficientFormer、SwiftFormer、FastViT 等
- Vision Mamba：ViM、VMamba、EfficientVMamba、QuadMamba 等
- 频率分析：此前有工作分析 CNN/Transformer 的频率偏好，本文首次分析 Mamba

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 频率解耦 + Mamba 的组合思路新颖，分析驱动设计
- **技术深度**: ⭐⭐⭐⭐ — 频谱分析→量化验证→架构设计的逻辑链完整
- **实验**: ⭐⭐⭐⭐ — 分类/检测/分割全覆盖，消融细致
- **写作**: ⭐⭐⭐⭐ — 动机论证充分，图表清晰

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MambaVision: A Hybrid Mamba-Transformer Vision Backbone](../../CVPR2025/segmentation/mambavision_a_hybrid_mamba-transformer_vision_backbone.md)
- [\[ICCV 2025\] VSSD: Vision Mamba with Non-Causal State Space Duality](vssd_vision_mamba_with_non-causal_state_space_duality.md)
- [\[CVPR 2025\] MambaOut: Do We Really Need Mamba for Vision?](../../CVPR2025/segmentation/mambaout_do_we_really_need_mamba_for_vision.md)
- [\[ICML 2025\] QMamba: On First Exploration of Vision Mamba for Image Quality Assessment](../../ICML2025/segmentation/qmamba_on_first_exploration_of_vision_mamba_for_image_quality_assessment.md)
- [\[ICCV 2025\] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy](deris_decoupling_perception_and_cognition_for_enhanced_referring_image_segmentat.md)

</div>

<!-- RELATED:END -->
