---
title: >-
  [论文解读] Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction
description: >-
  [AAAI 2026 Oral][自动驾驶][协同感知] 提出首个使用稀疏3D语义高斯基元作为协同感知通信介质的纯视觉语义占据预测框架，通过ROI裁剪+刚性变换传输高斯+邻域融合模块抑制噪声冗余，在mIoU上比单车提升+8.42，比baseline协同方法提升+3.28。 - 领域现状：协同感知通过V2X通信扩展单车感知范围…
tags:
  - "AAAI 2026 Oral"
  - "自动驾驶"
  - "协同感知"
  - "3D高斯溅射"
  - "语义占据"
  - "V2X通信"
  - "纯视觉"
---

# Vision-Only Gaussian Splatting for Collaborative Semantic Occupancy Prediction

**会议**: AAAI 2026 Oral  
**arXiv**: [2508.10936](https://arxiv.org/abs/2508.10936)  
**代码**: [GitHub](https://github.com/ChengChen2020/VOGS-CP)  
**领域**: 自动驾驶 / 协同感知  
**关键词**: 协同感知, 3D高斯溅射, 语义占据, V2X通信, 纯视觉  

## 一句话总结

提出首个使用稀疏3D语义高斯基元作为协同感知通信介质的纯视觉语义占据预测框架，通过ROI裁剪+刚性变换传输高斯+邻域融合模块抑制噪声冗余，在mIoU上比单车提升+8.42，比baseline协同方法提升+3.28。

## 研究背景与动机

- **领域现状**：协同感知通过V2X通信扩展单车感知范围。3D语义占据预测比BEV和3D检测提供更细粒度的场景理解。
- **现有痛点**：现有协同占据方法(CoHFF)使用三平面特征需要深度监督、多阶段训练、跨agent对齐复杂；密集体素特征通信代价高。
- **核心矛盾**：精细3D表示需要大量数据传输 vs V2X通信带宽有限。
- **本文目标**：设计通信高效、端到端训练的协同语义占据预测方案。
- **切入角度**：3D高斯是稀疏的、可刚性变换的、同时编码几何和语义的表示。
- **核心 idea**：用3D高斯基元替代体素/平面特征作为V2X通信介质，天然支持刚性对齐和稀疏传输。

## 方法详解

### 整体框架

单车：Image-to-Gaussian模块（随机初始化高斯→多尺度图像特征引导细化）→Gaussian-to-voxel溅射→占据预测。协同：Gaussian包装（刚性变换+ROI裁剪）→传输→跨agent高斯融合→溅射。

### 关键设计

**设计1：高斯基元作为通信介质**
- **功能**：将3D高斯(均值+尺度+旋转+不透明度+语义)作为V2X消息。
- **核心思路**：高斯在刚性变换下封闭（均值旋转+平移，旋转四元数相乘，尺度/不透明度/语义不变），ROI裁剪只传输ego感兴趣区域内的高斯。
- **设计动机**：比体素特征更稀疏、比平面特征更保留3D结构、对齐只需简单刚性变换。

**设计2：跨Agent高斯融合模块**
- **功能**：融合来自多个agent的高斯基元，抑制噪声和冗余。
- **核心思路**：邻域条件proposal→跨邻域池化→与ego高斯属性混合更新。不同于GaussianFormer的单agent细化，专门处理跨agent不一致性。
- **设计动机**：不同agent的高斯可能冗余或冲突（遮挡导致噪声），需要学习融合。

**设计3：端到端单阶段训练**
- **功能**：整个流程端到端训练，无需单独的深度估计或多阶段schedule。
- **核心思路**：基于GaussianFormer的Image-to-Gaussian+Gaussian-to-voxel，加上融合模块一起优化。
- **设计动机**：CoHFF需要两阶段训练和独立深度网络，增加了复杂性。

### 损失函数/训练策略

标准语义占据损失（CE + Lovász loss），单阶段端到端训练。

## 实验关键数据

### 主实验

| 方法 | IoU↑ | mIoU↑ |
|------|------|-------|
| 单车GSFormer | 67.76 | 29.20 |
| CoHFF(协同) | 50.46 | 34.16 |
| Zero-Shot堆叠 | 67.88 | 30.54 |
| Naive融合 | 70.10 | 36.02 |
| **Learned融合** | **72.87** | **37.44** |

### 消融实验

| 通信量 | mIoU |
|--------|------|
| 100%高斯 | 37.44 |
| 34.6%高斯 | 36.06 (+1.9 vs单车) |

### 关键发现

1. 即使零样本堆叠（无联合训练），高斯就能改善协同感知，验证了显式表示的优势。
2. 仅34.6%通信量仍能超过单车+1.9 mIoU，通信效率极高。
3. Learned融合比Naive融合改善+1.4 mIoU，邻域融合模块有效。

## 亮点与洞察

1. 首次将3D高斯溅射引入协同感知，开创性方向。
2. 高斯的刚性变换封闭性使跨agent对齐变得trivial。
3. 稀疏+显式+可解释的表示比隐式特征更适合通信约束场景。

## 局限与展望

1. 实验仅在仿真数据上验证，未在真实V2X数据集测试。
2. 高斯初始化仍为随机，可探索更好的初始化策略。
3. 未考虑通信延迟和异步问题。

## 相关工作与启发

- GaussianFormer将高斯用于单车占据预测，本文将其推广到协同场景。
- CoHFF是首个协同占据框架但依赖三平面+深度监督，本文大幅简化。
- 启发：选择合适的表示可以同时解决通信效率和对齐难度问题。

## 评分

| 维度 | 评分 |
|------|------|
| 创新性 | ★★★★★ |
| 实用性 | ★★★★☆ |
| 实验充分性 | ★★★☆☆ |
| 写作清晰度 | ★★★★☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](../../ICCV2025/autonomous_driving/gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] Semantic Causality-Aware Vision-Based 3D Occupancy Prediction](../../ICCV2025/autonomous_driving/semantic_causality-aware_vision-based_3d_occupancy_prediction.md)
- [\[ECCV 2024\] GaussianFormer: Scene as Gaussians for Vision-Based 3D Semantic Occupancy Prediction](../../ECCV2024/autonomous_driving/gaussianformer_scene_as_gaussians_for_vision-based_3d_semantic_occupancy_predict.md)
- [\[CVPR 2025\] OccMamba: Semantic Occupancy Prediction with State Space Models](../../CVPR2025/autonomous_driving/occmamba_semantic_occupancy_prediction_with_state_space_models.md)
- [\[CVPR 2026\] Generalizing Visual Geometry Priors to Sparse Gaussian Occupancy Prediction](../../CVPR2026/autonomous_driving/generalizing_visual_geometry_priors_to_sparse_gaussian_occupancy_prediction.md)

</div>

<!-- RELATED:END -->
