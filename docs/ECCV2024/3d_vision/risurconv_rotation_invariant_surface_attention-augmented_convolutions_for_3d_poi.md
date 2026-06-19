---
title: >-
  [论文解读] RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation
description: >-
  [ECCV 2024][3D视觉][点云分类] 提出 RISurConv，通过构建局部三角表面并提取高表达力旋转不变表面属性（RISP），结合注意力增强卷积，实现首次在精度上超越非旋转不变方法的旋转不变点云分析网络。 核心矛盾 核心矛盾：领域现状：3D点云深度学习大多关注平移和点排列不变性，旋转不变性被较少研究…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "点云分类"
  - "旋转不变性"
  - "自注意力"
  - "表面属性"
  - "3D分割"
---

# RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation

**会议**: ECCV 2024  
**arXiv**: [2408.06110](https://arxiv.org/abs/2408.06110)  
**代码**: [有](https://github.com/cszyzhang/RISurConv)  
**领域**: 3D视觉  
**关键词**: 点云分类, 旋转不变性, 自注意力, 表面属性, 3D分割

## 一句话总结

提出 RISurConv，通过构建局部三角表面并提取高表达力旋转不变表面属性（RISP），结合注意力增强卷积，实现首次在精度上超越非旋转不变方法的旋转不变点云分析网络。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：3D点云深度学习大多关注平移和点排列不变性，旋转不变性被较少研究。现有旋转不变方法（如RIConv、RIConv++）通过手工设计特征保证旋转不变性，但精度远低于非旋转不变方法（如PointTransformer v2）。主要原因是生成旋转不变特征时丢失了全局信息，且LRF/LRA不够稳定。本文目标是缩小甚至消除旋转不变方法与非旋转不变方法之间的精度差距。

## 方法详解

### 整体框架

1. 对每个参考点构建K近邻局部点集
2. 每个邻居构建两个三角表面，提取14维旋转不变表面属性（RISP）
3. RISP经MLP嵌入后通过两层自注意力（SA）层精炼特征
4. 五层RISurConv + Transformer编码器 + 全连接层输出分类/分割结果

### 关键设计

**旋转不变表面属性（RISP）**：对每个邻居点 x_i，选取两个相邻邻居 x_{i-1} 和 x_{i+1} 构建两个三角面。提取14维特征包括：距离L₀、5个欧氏空间角度（三角内角和二面角）、8个切空间角度（法向量与边的夹角）。RISP完整描述了双三角及其关系，具有数学完备性。

**RISurConv算子**：包含两个自注意力模块——SA1在邻域K个点间进行特征精炼，SA2在N个代表点间进行全局特征精炼。两者协同提升特征表达力。

### 损失函数

分类使用交叉熵损失，分割使用标准分割损失。

## 实验关键数据

### 主实验

**ModelNet40分类精度（Overall Accuracy %）**：

| 方法 | 旋转不变 | z/z | SO3/SO3 | z/SO3 | Std. |
|------|----------|-----|---------|-------|------|
| PointNet++ | ✗ | 89.3 | 85.0 | 28.6 | 33.8 |
| Pt Transformer v2 | ✗ | 94.2 | 88.3 | 51.8 | 23.0 |
| RIConv++ | ✓ | 91.3 | 91.3 | 91.3 | 0.0 |
| **RISurConv** | **✓** | **96.0** | **96.0** | **96.0** | **0.0** |

**ScanObjectNN真实场景分类（PB_T50_RS）**：

| 方法 | z/z | SO3/SO3 | z/SO3 |
|------|-----|---------|-------|
| RIConv++ | 80.3 | 80.3 | 80.3 |
| **RISurConv** | **93.1** | **93.1** | **93.1** |

**ShapeNet部件分割（mIoU %）**：

| 方法 | SO3/SO3 | z/SO3 |
|------|---------|-------|
| RIConv++ (xyz+nor) | 80.5 | 80.5 |
| **RISurConv (xyz+nor)** | **81.5** | **81.5** |

### 消融实验

| 消融项 | 精度 |
|--------|------|
| 完整模型 (A) | 96.0 |
| 去除L₀ (B) | 95.5 |
| 仅切空间角 (C) | 90.9 |
| 仅L₀+ϕ (D) | 88.2 |
| 去除SA1+SA2+TE (E) | 92.8 |
| 去除Transformer Encoder (D) | 94.3 |

### 关键发现

- 首个旋转不变方法在ModelNet40上超越所有非旋转不变方法（96.0% vs PT v2的94.2%）
- 在ScanObjectNN上超越RIConv++ 12.8个百分点
- 角度特征比距离特征更重要，切空间和欧氏空间角度互补
- 自注意力模块使特征分布更均匀

## 亮点与洞察

1. **首次超越非旋转不变方法**：证明旋转不变特征不必以牺牲精度为代价
2. **局部三角表面构建**：比逐点操作更好地捕捉局部几何结构
3. **RISP的完备性**：14维特征完全描述双三角结构，增加更多特征不再提升性能
4. 自注意力使权重重分配，提升特征有效性

## 局限与展望

- 参数量较大（14M vs RIConv++的0.4M），推理速度有一定下降
- 法向量估计质量影响分类精度（w/o normal比w/ normal低0.4%）
- 在细粒度分类上仍有提升空间

## 相关工作与启发

- RIConv仅考虑局部特征导致精度下降
- GCAConv使用LRF，但LRF不够稳定
- 启发：从点的操作升级到表面操作是提升3D特征表达力的有效途径

## 评分

- 创新性：★★★★★ 局部三角表面 + RISP特征设计精妙，首次打破旋转不变精度天花板
- 实用性：★★★★☆ 旋转不变性对机器人和自动驾驶场景非常重要
- 实验质量：★★★★★ 多个数据集全面验证，消融实验详尽

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Enhancing Rotation-Invariant 3D Learning with Global Pose Awareness and Attention Mechanisms](../../AAAI2026/3d_vision/enhancing_rotation-invariant_3d_learning_with_global_pose_awareness_and_attentio.md)
- [\[ECCV 2024\] TPA3D: Triplane Attention for Fast Text-to-3D Generation](tpa3d_triplane_attention_for_fast_text-to-3d_generation.md)
- [\[ECCV 2024\] Dual-level Adaptive Self-Labeling for Novel Class Discovery in Point Cloud Segmentation](dual-level_adaptive_self-labeling_for_novel_class_discovery_in_point_cloud_segme.md)
- [\[AAAI 2026\] Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning](../../AAAI2026/3d_vision/hierarchical_direction_perception_via_atomic_dot-product_operators_for_rotation-.md)
- [\[CVPR 2026\] 4D Local Modeling Toward Dynamic Global Perception for Ambiguity-free Rotation-Invariant Point Cloud Analysis](../../CVPR2026/3d_vision/4d_local_modeling_toward_dynamic_global_perception_for_ambiguity-free_rotation-i.md)

</div>

<!-- RELATED:END -->
