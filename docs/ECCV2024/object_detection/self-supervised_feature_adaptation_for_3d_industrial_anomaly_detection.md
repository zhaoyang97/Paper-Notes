---
title: >-
  [论文解读] Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection
description: >-
  [ECCV 2024][目标检测][3D异常检测] 提出 LSFA（Local-to-global Self-supervised Feature Adaptation）框架，通过模态内特征紧凑性优化（IFC）和跨模态局部到全局一致性对齐（CLC）两个自监督策略对预训练特征进行任务导向适配，在 MVTec-3D AD 上取得 97.1% I-AUROC，超越 SOTA +3.4%。
tags:
  - "ECCV 2024"
  - "目标检测"
  - "3D异常检测"
  - "多模态特征适配"
  - "自监督学习"
  - "记忆库"
  - "工业检测"
---

# Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection

**会议**: ECCV 2024  
**arXiv**: [2401.03145](https://arxiv.org/abs/2401.03145)  
**代码**: 未公开  
**领域**: 人体理解  
**关键词**: 3D异常检测, 多模态特征适配, 自监督学习, 记忆库, 工业检测

## 一句话总结

提出 LSFA（Local-to-global Self-supervised Feature Adaptation）框架，通过模态内特征紧凑性优化（IFC）和跨模态局部到全局一致性对齐（CLC）两个自监督策略对预训练特征进行任务导向适配，在 MVTec-3D AD 上取得 97.1% I-AUROC，超越 SOTA +3.4%。

## 研究背景与动机

工业异常检测通常作为无监督任务进行，只使用正常样本训练模型。现有 2D 异常检测方法已取得很好效果，但仅依赖 RGB 数据不足以识别细微的几何表面异常。因此需要同时利用 RGB 图像和 3D 点云进行多模态异常检测。

现有多模态方法（如 PatchCore+FPFH、M3DM）直接使用在 ImageNet 上预训练的模型构建特征数据库，但存在两个核心问题：

**域偏差问题**：预训练知识与工业场景存在较大差距，导致模型容易将异常区域误判为正常（假阴性）

**纹理复杂性问题**：对于纹理复杂的类别（如 cookie），模型难以识别细微的异常模式

本文的核心洞察是：**预训练特征不能直接使用，需要针对异常检测任务进行自监督适配**，从模态内紧凑性和跨模态一致性两个维度提升特征质量。

## 方法详解

### 整体框架

LSFA 以 RGB 图像 $I \in \mathbb{R}^{H \times W \times 3}$ 和点云 $P \in \mathbb{R}^{N \times 3}$ 作为输入。分别用预训练的 ViT-B/8（DINO 预训练）提取 RGB 特征，用 PointMAE 提取 3D 特征。在预训练特征提取器之后接入轻量化的 Transformer encoder 层作为 adaptor $\Psi_I$ / $\Psi_P$，通过 IFC 和 CLC 两个自监督目标优化 adaptor 参数。推理阶段仅使用适配后的局部特征构建 PatchCore 记忆库进行异常评分。

### 关键设计

1. **跨模态局部到全局一致性对齐（CLC）**：解决两个模态特征未对齐导致融合效果差的问题

    - **特征投影**：通过距离加权插值将 3D 点特征重映射到 2D patch 空间，使两个模态共享相同的 patch 数目 $N_m$，自然建立局部对应关系
    - **局部对齐（LA）**：对同一位置的 RGB 和点云 patch 特征计算对比损失，最大化相同位置跨模态特征相似度，最小化不同位置的相似度：
    $\mathcal{L}_{LA} = -\log \frac{\exp(\langle F_{I_i}^{\prime j}, F_{P_i}^{\prime j}\rangle)}{\sum_{t,k} \exp(\langle F_{I_i}^{\prime t}, F_{P_i}^{\prime k}\rangle)}$
    - **全局对齐（GA）**：对局部特征用 k-means 聚类得到实例级全局特征 $G_{I_i}$/$G_{P_i}$，在 batch 内跨样本进行对比对齐：
    $\mathcal{L}_{GA} = -\log \frac{\exp(\langle G_{I_i}^{\prime}, G_{P_i}^{\prime}\rangle)}{\sum_{t,x} \exp(\langle G_{I_t}^{\prime}, G_{P_x}^{\prime}\rangle)}$
    - 设计动机：局部对齐保证空间细粒度一致性，全局对齐保证对象级结构信息交互，两者互补

2. **模态内特征紧凑性优化（IFC）**：解决预训练特征中正常/异常特征分布难以区分的问题

    - **局部紧凑性（LC）**：为每个模态维护动态更新的 patch 级记忆库 $M_I^L$，最小化当前 patch 特征与记忆库最近邻的距离：
    $\mathcal{L}_{LC} = \sum_{i=1}^{N_b} \sum_{j=1}^{N_m} \min_{Q \in M_I^L} \|F_{I_i}^j - Q\|_2$
    - **全局紧凑性（GC）**：维护原型级记忆库 $M_I^G$，以类似方式优化全局特征的紧凑性
    - 记忆库采用 FIFO 队列机制动态更新，每次迭代将当前 batch 特征入队，最老特征出队
    - 设计动机：缩小正常特征的分布范围，使异常特征更容易被检测为离群点

3. **推理流程**：适配完成后仅使用局部特征（丢弃全局特征），分别对 RGB 和 3D 模态用 PatchCore 计算异常分数，最终取两个模态的平均值

### 损失函数 / 训练策略

总损失为两部分加权和：

$$\mathcal{L}_{LSFA} = \mathcal{L}_{IFC} + \lambda \mathcal{L}_{CLC}$$

其中 $\mathcal{L}_{IFC} = \mathcal{L}_{LC} + \mathcal{L}_{GC}$，$\mathcal{L}_{CLC} = \mathcal{L}_{LA} + \mathcal{L}_{GA}$

- 优化器：AdamW，学习率 $2 \times 10^{-3}$，cosine warm-up
- Batch size：8
- RGB 特征提取器：ViT-B/8（DINO 预训练），768维输出池化到 56×56
- 3D 特征提取器：Point Transformer（ShapeNet 预训练），3/7/11 层输出拼接

## 实验关键数据

### 主实验

**MVTec-3D AD I-AUROC（RGB+3D 多模态）**

| 方法 | Bagel | Cookie | Dowel | Tire | Mean |
|------|-------|--------|-------|------|------|
| PatchCore+FPFH | 0.918 | 0.883 | 0.932 | 0.886 | 0.865 |
| AST | 0.983 | 0.971 | 0.932 | 0.797 | 0.937 |
| M3DM | 0.994 | 0.972 | 0.942 | 0.850 | 0.945 |
| **LSFA** | **1.000** | **0.989** | **0.961** | **0.951** | **0.971** |

**MVTec-3D AD AUPRO（RGB+3D 多模态）**

| 方法 | Bagel | Cookie | Mean |
|------|-------|--------|------|
| PatchCore+FPFH | 0.976 | 0.973 | 0.959 |
| M3DM | 0.970 | 0.950 | 0.964 |
| **LSFA** | **0.986** | 0.946 | **0.968** |

仅 3D 模态下 LSFA Mean I-AUROC 达到 0.921，超越 M3DM 的 0.874 达 +4.7%。

### 消融实验

**IFC 和 CLC 组件消融**

| IFC | CLC | I-AUROC | AUPRO | P-AUROC |
|-----|-----|---------|-------|---------|
| ✗ | ✗ | 0.929 | 0.953 | 0.987 |
| ✗ | ✓ | 0.957 | 0.963 | 0.990 |
| ✓ | ✗ | 0.959 | 0.964 | 0.992 |
| ✓ | ✓ | **0.971** | **0.968** | **0.993** |

**CLC 内部损失消融**

| $\mathcal{L}_{GA}$ | $\mathcal{L}_{LA}$ | I-AUROC | AUPRO |
|---|---|---------|-------|
| ✗ | ✗ | 0.929 | 0.953 |
| ✗ | ✓ | 0.949 | 0.961 |
| ✓ | ✗ | 0.952 | 0.961 |
| ✓ | ✓ | **0.959** | **0.964** |

### 关键发现

- IFC 和 CLC 各自贡献约 +2.8%/+3.0% I-AUROC，组合后进一步提升到 +4.2%
- 局部和全局两个层次的对齐/紧凑性都是必要的，单独使用效果不如组合
- 在 Eyecandies 数据集上也取得最佳结果（RGB I-AUROC 87.5%，AUPRO 97.8%）
- 对 cable gland 和 tire 这类挑战性类别提升尤为显著

## 亮点与洞察

1. **问题分析到位**：清晰指出直接使用预训练特征的两个缺陷（过度估计正常性 + 复杂纹理下漏检），并提出针对性解决方案
2. **自监督适配思路**：不需要任何异常标注，完全利用正常样本间的模态内/跨模态自监督信号进行特征微调
3. **Local-to-global 设计哲学**：无论是跨模态对齐还是模态内紧凑性，都从局部和全局两个粒度进行优化，确保多尺度信息的充分利用
4. **推理零开销**：与 M3DM 相比，LSFA 在推理阶段不引入额外模块，仅更换了特征提取器后的 adaptor 权重

## 局限与展望

1. adaptor 结构仅使用单层 Transformer encoder，可能在更复杂场景下表达能力不足
2. 记忆库大小需要手动调节，不同类别的最优值可能不同
3. 仅在两个相对小规模的 3D 异常检测数据集上评估，更大规模场景下的泛化性未知
4. 两个模态异常分数的简单平均可能不是最优的融合策略

## 相关工作与启发

- **PatchCore**：基于记忆库的特征嵌入方法，是本文的推理基础
- **M3DM**：多模态 3D 异常检测的前作，本文在其基础上增加了模态内优化和多粒度对齐
- **CFA**：提出 coupled-hypersphere 微调框架适配特征，思路与本文类似但仅用于 2D
- **启发**：特征适配的思想可推广到其他使用预训练特征的下游任务，local-to-global 的自监督对齐策略具有通用性

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将自监督特征适配引入 3D 异常检测，IFC+CLC 的 local-to-global 设计合理且有效
- **实验充分度**: ⭐⭐⭐⭐⭐ 多数据集、多模态、多指标的全面评估，消融实验细致
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，图示直观
- **价值**: ⭐⭐⭐⭐ 在 3D 工业异常检测领域达到新 SOTA，方法简洁实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection](../../AAAI2026/object_detection/casl_curvature-augmented_self-supervised_learning_for_3d_anomaly_detection.md)
- [\[ECCV 2024\] WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)
- [\[ECCV 2024\] Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)
- [\[ECCV 2024\] DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning](dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning.md)
- [\[CVPR 2025\] Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](../../CVPR2025/object_detection/large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)

</div>

<!-- RELATED:END -->
