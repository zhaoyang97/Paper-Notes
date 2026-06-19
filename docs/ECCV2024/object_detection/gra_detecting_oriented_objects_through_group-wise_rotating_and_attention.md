---
title: >-
  [论文解读] GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention
description: >-
  [ECCV2024][目标检测][Group-wise Rotating] 提出轻量级的 Group-wise Rotating and Attention (GRA) 模块，通过将卷积核分组旋转并施加分组空间注意力，在参数量减少近 50% 的同时超越了此前 SOTA 方法 ARC，在 DOTA-v2.0 上取得新的最优性能。
tags:
  - "ECCV2024"
  - "目标检测"
  - "Group-wise Rotating"
  - "注意力机制"
  - "Dynamic Neural Networks"
---

# GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention

**会议**: ECCV2024  
**arXiv**: [2403.11127](https://arxiv.org/abs/2403.11127)  
**代码**: [wangjiangshan0725/GRA](https://github.com/wangjiangshan0725/GRA)  
**领域**: 目标检测  
**关键词**: Oriented Object Detection, Group-wise Rotating, Spatial Attention, Dynamic Neural Networks  
**机构**: 清华大学, DAMO Academy (阿里巴巴), ModelTC

## 一句话总结

提出轻量级的 Group-wise Rotating and Attention (GRA) 模块，通过将卷积核分组旋转并施加分组空间注意力，在参数量减少近 50% 的同时超越了此前 SOTA 方法 ARC，在 DOTA-v2.0 上取得新的最优性能。

## 背景与动机

旋转目标检测（Oriented Object Detection）旨在用旋转边界框定位和识别多种朝向的物体，广泛应用于遥感、自动驾驶、文字识别等场景。近年来的研究已从多角度（包围框表示、损失函数、网络架构、标签分配策略）推进该领域，但最新趋势聚焦于开发方向感知的检测骨干网络。

此前 SOTA 方法 Adaptive Rotated Convolution (ARC) 使用 $m$ 个独立卷积核，每个核旋转不同角度后分别提取特征，再通过加权求和聚合输出。该方法虽然提升了性能，但带来两个关键问题：

1. **参数量过大**：使用 $m$ 个卷积核导致参数量增长 $m$ 倍。标准 ResNet-50 约 23.5M 参数，集成 ARC ($m=4$) 后膨胀至 57.2M，对存储资源受限的遥感设备等部署场景形成严重挑战。
2. **特征不精确**：不同角度卷积核提取的特征通过加权求和混合，会将目标特征与噪声耦合。实验发现，某个旋转角度的卷积核主要捕获与其角度对齐的物体特征，对其他朝向物体产生不良噪声，加权求和后导致大量低置信度检测结果。

## 核心问题

如何在旋转目标检测骨干网络中**同时实现模型有效性和参数高效性**？具体而言：

- 如何在不复制多个完整卷积核的情况下捕获多种旋转方向的细粒度特征？
- 如何避免不同朝向特征混合带来的噪声干扰？

## 方法详解

### 整体框架

GRA 模块由两个核心组件组成：**Group-wise Rotating**（分组旋转）和 **Group-wise Attention**（分组注意力），用于替换骨干网络中 ResNet 后三个 stage 的 $3 \times 3$ 卷积。

### 1. Group-wise Rotating

该组件包含三个步骤：

**角度预测 (Angle Generator)**：轻量网络从输入特征图 $\boldsymbol{x} \in \mathbb{R}^{C_{\text{in}} \times H_{\text{in}} \times W_{\text{in}}}$ 中预测 $n$ 个旋转角度和缩放因子。具体流程为：深度可分离卷积 → ReLU → LayerNorm → 全局池化 → 两个线性层（各输出 $n$ 维），分别得到 $\{\theta_j\}$ 和 $\{\lambda_j\}$。

**分组 (Grouping)**：将卷积核 $\boldsymbol{W} \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times k \times k}$ 沿 $C_{\text{out}}$ 维度均匀分成 $n$ 组，每组包含 $C_{\text{out}}/n$ 个子核。

**旋转 (Rotating)**：每组核以对应预测角度 $\theta_j$ 旋转，并乘以缩放因子 $\lambda_j$：

$$\widetilde{\boldsymbol{W}}_j = \{\lambda_j \times \text{Rotate}(\boldsymbol{w}_{j,l}, \theta_j)\}$$

旋转通过双线性插值实现。所有旋转后的分组核拼接后与输入做标准卷积，得到输出特征 $\boldsymbol{y}$。

**与分组卷积的区别**：分组卷积沿 $C_{\text{in}}$ 维度分组输入特征，各组独立卷积；而 GRA 沿 $C_{\text{out}}$ 维度分组卷积核用于不同角度旋转，卷积本身仍是标准卷积。

### 2. Group-wise Attention

卷积输出 $\boldsymbol{y}$ 自然分为 $n$ 组，每组 $\boldsymbol{y}_j \in \mathbb{R}^{C_{\text{out}}/n \times H_{\text{out}} \times W_{\text{out}}}$ 主要捕获接近角度 $\theta_j$ 的物体特征，但对其他朝向物体存在噪声。分组注意力机制的处理流程：

1. 对每组特征分别做 Max Pooling 和 Avg Pooling，拼接得到 $\boldsymbol{S}_j \in \mathbb{R}^{2 \times H_{\text{out}} \times W_{\text{out}}}$
2. 经卷积层 $F$ 调整通道后接 Sigmoid 得到注意力图 $\widetilde{\boldsymbol{S}}_j \in \mathbb{R}^{1 \times H_{\text{out}} \times W_{\text{out}}}$
3. 逐元素乘以原始特征组：$\widetilde{\boldsymbol{y}}_j = \boldsymbol{y}_j \odot \widetilde{\boldsymbol{S}}_j$

这一机制增强各组中与对应旋转角度对齐的目标区域，同时抑制不相关区域的噪声。

### 设计优势

- **更细粒度的角度建模**：$n$ 可设较大值（实验中 $n=32$），预测更多角度，参数增加极少
- **即插即用**：可无缝嵌入任何卷积网络
- **可复用预训练权重**：仅需加载标准 ResNet 预训练权重，冻结 ResNet 参数只训练 GRA 模块即可

## 实验关键数据

### DOTA-v1.0（单尺度训练测试，12 epochs）

以 Oriented R-CNN + ResNet-50 为例：

| 骨干网络 | 参数量 (M) | mAP (%) |
|---------|-----------|---------|
| R50 (baseline) | 41.37 | 75.81 |
| R50_ARC | 75.06 | 77.35 |
| **R50_GRA** | **41.65 (↓43%)** | **77.63** |

GRA 在 6 种检测器上均一致超越 ARC，参数量减少 43%–46%。

### DOTA-v2.0（含大量小目标）

| 方法 | mAP (%) |
|------|---------|
| Oriented R-CNN + R50_ARC | 55.91 |
| Oriented R-CNN + R50_GRA | 56.63 |
| **R50_GRA (40 epochs)** | **57.95 (SOTA)** |

### HRSC2016

Oriented R-CNN + R50_GRA: mAP 72.59%，优于 ARC 的 72.39%。

### 消融实验

**分组数量** ($n$)：$n$ 从 2 增加到 32，mAP 从 76.82% 提升至 77.63%，参数仅增加 0.23M，FLOPs 增加 1.2G。

**各组件贡献**（Oriented R-CNN, DOTA-v1.0）：

| Group-wise Rotating | 缩放因子 $\lambda$ | Group-wise Attention | mAP (%) |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | ✗ | 75.81 |
| ✓ | ✗ | ✗ | 76.73 |
| ✓ | ✓ | ✗ | 77.25 |
| ✓ | ✓ | ✓ | **77.63** |

### 预训练策略

加载公开 ResNet 预训练权重、仅训练 GRA 模块可达 77.39% mAP，接近从零预训练的 77.64%，无需额外 ImageNet 训练资源。

## 亮点

1. **参数效率极高**：通过分组旋转代替多核复制，参数量降低近 50%，性能反超 ARC
2. **分组注意力去噪**：精准解决了 ARC 加权求和引入噪声的痛点，避免不同朝向特征的互相干扰
3. **灵活性强**：即插即用模块，兼容多种单阶段和两阶段检测器；支持复用公开预训练权重，降低训练成本
4. **分析细致**：通过置信度分布可视化清晰揭示了 ARC 加权求和导致特征退化的根因

## 局限与展望

1. **核大小限制**：目前仅替换 $3 \times 3$ 卷积核，在更大核（如 $7 \times 7$）上的效果未验证
2. **仅限 ResNet 架构**：未在 ConvNeXt、ViT 等现代架构上测试，通用性有待验证
3. **sample-wise 角度预测**：角度由全局池化后的全连接层预测，属于样本级（非空间级），对单张图像内多朝向物体的细粒度处理仍有空间
4. **旋转角度冗余**：当 $n$ 过大时，各组预测相似角度，存在饱和现象

## 与相关工作的对比

| 方法 | 核心思路 | 参数效率 | 特征质量 |
|------|---------|---------|---------|
| **ARC** | $m$ 个完整核各自旋转后加权求和 | 差（$m$ 倍参数） | 加权求和引入噪声 |
| **ReDet** | 旋转等变操作（群论） | 中等 | 保持旋转等变性 |
| **LSKNet** | 自适应选择核大小的空间注意力 | 中等 | 未建模旋转信息 |
| **GRA (本文)** | 单核分组旋转 + 分组空间注意力 | **高**（仅 +0.28M） | 分组注意力去噪 |

## 启发与关联

1. **分组策略的高效性**：将"多核独立操作"转化为"单核分组操作"是一种通用的参数压缩思路，可推广到其他动态网络场景
2. **特征去噪的精细化**：按功能分组后分别施加注意力，比全局注意力更有针对性，对多任务学习场景也有借鉴意义
3. **与旋转等变网络的互补性**：GRA 是数据驱动的动态旋转，ReDet 是结构性等变，两者是否可结合值得探索

## 评分
- 新颖性: ⭐⭐⭐⭐ — 分组旋转 + 分组注意力的组合设计简洁高效，解决了 ARC 的两大痛点
- 实验充分度: ⭐⭐⭐⭐ — 多数据集、多检测器、多消融实验全面，可视化分析有说服力
- 写作质量: ⭐⭐⭐⭐ — 问题分析清晰，motivation 图表设计精良
- 价值: ⭐⭐⭐⭐ — 参数减半性能提升，实用性强，但架构泛化性有待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ElasticFormer: Detecting Objects in HRW Shots via Elastic Computing Vision Transformer](../../CVPR2026/object_detection/elasticformer_detecting_objects_in_hrw_shots_via_elastic_computing_vision_transf.md)
- [\[CVPR 2025\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](../../CVPR2025/object_detection/show_dont_tell_detecting_novel_objects_by_watching_human_videos.md)
- [\[CVPR 2026\] Detecting Unknown Objects via Energy-Based Separation for Open World Object Detection](../../CVPR2026/object_detection/detecting_unknown_objects_via_energy-based_separation.md)
- [\[ECCV 2024\] Visible and Clear: Finding Tiny Objects in Difference Map](visible_and_clear_finding_tiny_objects_in_difference_map.md)
- [\[ECCV 2024\] Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation](projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)

</div>

<!-- RELATED:END -->
