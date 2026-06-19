---
title: >-
  [论文解读] Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking
description: >-
  [CVPR 2025][视频理解][UAV跟踪] 发现轻量级 ViT 跟踪器中深层存在显著冗余（特征饱和），提出相似度引导的层自适应方法 SGLATrack，动态禁用冗余层并仅保留一个最优层，在 GPU 上实现 225 FPS 的实时 UAV 跟踪。 UAV 跟踪对推理效率要求极高，因为无人机的算力和电力资源有限…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "UAV跟踪"
  - "ViT加速"
  - "层冗余"
  - "动态层选择"
  - "实时推理"
---

# Similarity-Guided Layer-Adaptive Vision Transformer for UAV Tracking

**会议**: CVPR 2025  
**arXiv**: [2503.06625](https://arxiv.org/abs/2503.06625)  
**代码**: [GitHub](https://github.com/GXNU-ZhongLab/SGLATrack)  
**领域**: 视频理解  
**关键词**: UAV跟踪, ViT加速, 层冗余, 动态层选择, 实时推理

## 一句话总结

发现轻量级 ViT 跟踪器中深层存在显著冗余（特征饱和），提出相似度引导的层自适应方法 SGLATrack，动态禁用冗余层并仅保留一个最优层，在 GPU 上实现 225 FPS 的实时 UAV 跟踪。

## 研究背景与动机

UAV 跟踪对推理效率要求极高，因为无人机的算力和电力资源有限。现有 ViT 跟踪器面临以下挑战：

1. **完整 ViT 过于笨重**：Mixformer、OSTrack 等方法虽然精度高，但推理速度远不能满足 UAV 实时需求
2. **现有加速方法有局限**：Aba-ViTrack 通过 token 裁剪加速，但引入了非结构化访问开销；AVTrack 在每层附加分类器决定是否执行，但估计输入复杂度困难且分类器冗余
3. **层冗余未被充分探索**：轻量级 ViT 中的层冗余问题尚未被系统研究

本文首次系统分析了轻量级 ViT 跟踪器中的层冗余现象：搜索特征在浅层变化显著，但在某一层达到饱和后，后续层的特征变化很小，对最终预测影响有限。

## 方法详解

### 整体框架

SGLATrack 采用 one-stream 架构，输入模板图像 $\mathbf{Z} \in \mathbb{R}^{3 \times 128 \times 128}$ 和搜索图像 $\mathbf{S} \in \mathbb{R}^{3 \times 256 \times 256}$，通过 patch embedding 拼接后送入 ViT。在饱和层 $l^*$ 处，选择模块决定保留哪一个后续层 $l^* + k$，其余层全部禁用。最终搜索特征 $\mathbf{X}_s^{l^*+k}$ 送入预测头输出目标框。

### 关键设计

**1. 基于特征饱和的层冗余分析**

- **功能**：确定 ViT 中哪些层是冗余的，可以安全禁用
- **核心思路**：通过逐层计算余弦相似度 $\text{Cos}(\mathbf{X}_s^i, \mathbf{X}_s^{i-1})$ 并分析各层输出的 AUC 变化，发现深层特征变化越来越小（余弦相似度接近 1），AUC 增长趋于平缓。设定饱和层 $l^* = 6$（12 层 ViT 中）
- **设计动机**：浅层细节对跟踪更有价值，深层语义信息相对冗余。这一分析为后续的层裁剪提供了理论依据

**2. 选择模块与相似度引导的层选择**

- **功能**：动态选择饱和层之后保留哪一个层，以最小化性能损失
- **核心思路**：选择模块 $\mathcal{M}$ 是一个 3 层 MLP（隐藏维度 160），输入饱和层特征的第一维 $\mathbf{z} = \mathbf{e}_1^T \mathbf{X}^{l^*} \in \mathbb{R}^N$，输出各后续层被选中的概率 $\hat{\mathbf{y}} = \sigma(\mathcal{M}(\mathbf{z})) \in \mathbb{R}^K$。选择概率最高的层保留，其余禁用
- **设计动机**：固定保留某一层无法适应所有场景。不同跟踪场景需要不同层的表征，因此需要动态选择

**3. 层级相似度损失 (Layer-wise Similarity Loss)**

- **功能**：优化选择模块，使其学会选择与饱和层特征最相似的后续层
- **核心思路**：$\mathcal{L}_{sim} = \frac{1}{K} \sum_{k=1}^{K} |\hat{y}^k - y^k|$，其中期望概率 $y^k = 1$ 当该层与饱和层余弦相似度最大，否则 $y^k = 0$。这引导模型选择能最大化保持目标注意力聚焦的层
- **设计动机**：如果饱和层已聚焦目标，与其最相似的后续层也最可能保持这种聚焦，因此高相似度有利于一致的注意力

### 损失函数

总损失由分类损失、回归损失和相似度损失组成：

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_{iou} \mathcal{L}_{iou} + \lambda_{L1} \mathcal{L}_{L1} + \gamma \mathcal{L}_{sim}$$

其中 $\lambda_{iou} = 2$，$\lambda_{L1} = 5$，$\gamma = 0.2$。分类使用 Focal loss，回归使用 $L_1$ + GIoU loss。

## 实验关键数据

### 主实验：五个 UAV 数据集平均性能与速度

| 方法 | Avg. AUC(%) | Avg. P(%) | GPU FPS | CPU FPS |
|------|-----------|---------|---------|---------|
| TCTrack (CVPR'22) | 58.7 | 77.8 | 135.8 | - |
| HCAT (ECCV'22) | 62.1 | 80.4 | 110.1 | - |
| AVTrack-DeiT (ICML'24) | 63.7 | 81.9 | 197.3 | - |
| **SGLATrack-DeiT*** | **64.7** | **82.6** | **224.7** | **74.8** |
| **SGLATrack-EVA** | 63.7 | 81.9 | **236.9** | **77.2** |

### 消融实验：层自适应的效果

| 变体 | LA | UAV123 AUC | UAVTrack112 AUC | FPS | Params(M) | FLOPs(G) |
|------|-----|-----------|----------------|-----|-----------|----------|
| SGLATrack-DeiT* | ✗ | 67.1 | 67.8 | 175.5 | 7.98 | 2.39 |
| SGLATrack-DeiT* | ✓ | 66.9 | 67.5 | **224.7** | **5.81** | **1.68** |
| SGLATrack-EVA | ✗ | 65.3 | 67.1 | 185.4 | 5.76 | 1.73 |
| SGLATrack-EVA | ✓ | 65.1 | 66.9 | **236.9** | **4.15** | **1.20** |

### 饱和层位置选择

| $l^*$ | UAV123 AUC | UAVTrack112 AUC | FPS |
|-------|-----------|----------------|-----|
| 5 | 66.1 | 66.4 | 239.6 |
| **6** | **66.9** | **67.5** | **224.7** |
| 7 | 66.9 | 67.7 | 211.3 |

### 关键发现

- 层自适应仅造成约 0.2-0.3% AUC 下降，但速度提升约 28%（175→225 FPS）
- 参数量减少 ~27%（7.98M→5.81M），FLOPs 减少 ~30%
- 动态选择最相似层（#2 Maximizing）比固定层（#1）效果好 1.5% AUC
- CPU 速度也达到 ~75 FPS，超越部分 DCF 方法

## 亮点与洞察

1. **层冗余分析本身就是贡献**：首次系统证明轻量级 ViT 跟踪器中存在显著层冗余，浅层特征对跟踪更关键
2. **相比 AVTrack 更高效**：只在饱和层处做一次选择决策，无需每层都附加分类器
3. **通用性强**：方法成功应用于 ViT-tiny、DeiT-tiny、EVA-tiny 三种不同骨干

## 局限与展望

- 饱和层 $l^*$ 作为超参数需要预先确定，不同模型可能需要不同设置
- 只保留一个后续层可能不是最优，某些场景下多层组合可能更好
- 未来可扩展到更大规模 ViT 和其他视觉任务

## 相关工作与启发

- **OSTrack**：one-stream 跟踪框架的基础
- **AVTrack**：动态层激活的先驱，但每层附加分类器冗余
- **DynamicViT/AViT**：token 裁剪方法，但引入非结构化访问开销

## 评分

⭐⭐⭐⭐ — 分析充分、方法简洁、效果实用。在几乎不损失精度的情况下实现了显著加速，对 UAV 部署有直接价值。层冗余分析为 ViT 加速提供了新视角。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MUST: The First Dataset and Unified Framework for Multispectral UAV Single Object Tracking](must_the_first_dataset_and_unified_framework_for_multispectral_uav_single_object.md)
- [\[CVPR 2026\] Rethinking Occlusion Modeling for UAV Tracking](../../CVPR2026/video_understanding/rethinking_occlusion_modeling_for_uav_tracking.md)
- [\[CVPR 2025\] MambaVLT: Time-Evolving Multimodal State Space Model for Vision-Language Tracking](mambavlt_time-evolving_multimodal_state_space_model_for_vision-language_tracking.md)
- [\[ICCV 2025\] General Compression Framework for Efficient Transformer Object Tracking](../../ICCV2025/video_understanding/general_compression_framework_for_efficient_transformer_object_tracking.md)
- [\[CVPR 2025\] Context-Enhanced Memory-Refined Transformer for Online Action Detection](context-enhanced_memory-refined_transformer_for_online_action_detection.md)

</div>

<!-- RELATED:END -->
