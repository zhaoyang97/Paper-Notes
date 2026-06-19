---
title: >-
  [论文解读] Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation
description: >-
  [CVPR 2025][3D视觉][点云实例分割] Relation3D 通过自适应超点聚合模块（ASAM）、对比学习引导的超点精炼（CLSR）和关系感知自注意力（RSA）三个组件增强了 Transformer-based 3D 实例分割中场景特征内部关系和 query 间关系的建模，在 ScanNetV2/ScanNet++/ScanNet200/S3DIS 上取得 SOTA。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "点云实例分割"
  - "关系建模"
  - "对比学习"
  - "自适应超点聚合"
  - "Transformer"
---

# Relation3D: Enhancing Relation Modeling for Point Cloud Instance Segmentation

**会议**: CVPR 2025  
**arXiv**: [2506.17891](https://arxiv.org/abs/2506.17891)  
**代码**: [GitHub](https://github.com/)  
**领域**: 3D视觉 / 点云实例分割  
**关键词**: 点云实例分割, 关系建模, 对比学习, 自适应超点聚合, Transformer

## 一句话总结

Relation3D 通过自适应超点聚合模块（ASAM）、对比学习引导的超点精炼（CLSR）和关系感知自注意力（RSA）三个组件增强了 Transformer-based 3D 实例分割中场景特征内部关系和 query 间关系的建模，在 ScanNetV2/ScanNet++/ScanNet200/S3DIS 上取得 SOTA。

## 研究背景与动机

**领域现状**：3D 点云实例分割旨在预测场景中每个物体实例的二值前景 mask 和语义标签。当前主流方法基于 Transformer 编码器-解码器框架，使用 instance query 通过 mask attention 与场景特征交互来生成实例 mask。代表性工作包括 SPFormer、Mask3D、QueryFormer 和 Maft。

**现有痛点**：现有 Transformer-based 方法主要通过 mask attention 建模场景特征与 query 特征之间的"外部关系"，但忽略了两类"内部关系"：(1) 场景特征（超点）之间的关系——同一实例内超点特征一致性不足、不同实例间区分度不够；(2) query 特征之间的关系——传统 self-attention 仅隐式计算相似度，缺乏空间和几何关系的显式建模。

**核心矛盾**：超点特征通过 pooling 聚合，但同一超点内点特征差异大（变异值 1.86），pooling 引入不合适特征并模糊有区分力的特征。同时，position embedding 通常不准确（SPFormer 的可学习位置编码缺乏具体空间含义，Mask3D/Maft 的位置编码与实际 mask 位置存在偏差），使得 self-attention 的空间关系建模不充分。

**本文目标**：(1) 如何有效建模场景特征之间的关系？(2) 如何更好地建模 query 之间的关系？

**切入角度**：从特征关系建模的角度切入——对场景特征，用自适应权重聚合替代 pooling 并用对比学习引导更新方向；对 query 特征，将显式的位置和几何关系作为 bias 嵌入 self-attention。

**核心 idea**：通过三个互补的关系建模模块（ASAM + CLSR + RSA），从场景特征和 query 特征两个层面增强 Transformer 解码器的关系建模能力。

## 方法详解

### 整体框架

输入点云（含位置、颜色、法线）经 Sparse UNet 提取点级特征 $F \in \mathbb{R}^{N \times C}$，通过 ASAM 聚合为超点级特征 $F_{\text{super}} \in \mathbb{R}^{M \times C}$。初始化 $K$ 个 instance query $Q \in \mathbb{R}^{K \times C}$，送入 Transformer 解码器迭代更新。解码器中包含 RSA（增强 query 间关系）、mask attention（query-场景外部关系）、以及每隔 $r$ 层执行的 CLSR（用对比学习引导超点特征更新）。

### 关键设计

1. **自适应超点聚合模块 (ASAM)**:

    - 功能：将点级特征自适应地聚合为超点级特征，强调有区分力的点、抑制不合适特征
    - 核心思路：对点级特征 $F$ 分别做 max-pooling 和 mean-pooling 得到 $F_{\max}$ 和 $F_{\text{mean}}$，计算它们与原始点特征的差值，通过两个独立 MLP 预测每个点的权重 $\mathcal{W}_{\max} = \text{MLP}_1(F_{\max} - F)$，在每个超点内做 softmax 归一化后加权聚合。最终将两条路径的结果 concat 并通过 MLP 降维。整个过程可用 point-wise MLP 和 torch-scatter 并行化
    - 设计动机：直接 pooling 在超点内点特征差异大时会引入噪声。通过与 pooling 统计量的差异自适应分配权重，能让有意义的、有区分力的点特征获得更高权重

2. **对比学习引导的超点精炼模块 (CLSR)**:

    - 功能：在解码器中利用 query 特征反向更新超点特征，并通过对比学习约束更新方向
    - 核心思路：采用双路径结构，超点特征作为 $\mathcal{Q}$、query 特征作为 $\mathcal{K}$ 和 $\mathcal{V}$ 进行 cross-attention（与常规相反）。基于实例标注构建超点关系矩阵 $R_{\text{super}}^{\text{GT}}$，计算归一化超点特征的余弦相似度矩阵 $\mathcal{S}$，用 BCE loss 约束 $L_{\text{cont}} = \text{BCE}(\frac{\mathcal{S}+1}{2}, R_{\text{super}}^{\text{GT}})$。每隔 $r=3$ 层执行一次精炼以控制计算开销
    - 设计动机：mask attention 只建模 query 到场景的单向关系，双路径设计让信息双向流动加快收敛。对比损失显式引导同实例超点特征趋近、不同实例超点特征远离

3. **关系感知自注意力 (RSA)**:

    - 功能：在 self-attention 中融入 query 间显式的位置和几何关系
    - 核心思路：首先计算每个 query 的 mask 对应的 3D bounding box（中心 $x,y,z$ 和尺度 $l,w,h$），然后计算两两 query 间的位置相对关系（坐标差/尺度的 log）和几何相对关系（尺度比的 log），得到 6 维关系编码 $\mathfrak{T} \in \mathbb{R}^{K \times K \times 6}$。经 sin-cos 位置编码升维后通过线性变换得到 $R_q \in \mathbb{R}^{K \times K \times \mathcal{H}}$，作为 bias 加入注意力分数：$\text{RSA}(Q) = \text{Softmax}(\frac{\mathcal{QK}^T}{\sqrt{C}} + R_q)\mathcal{V}$
    - 设计动机：传统 position embedding 与实际 mask 位置不匹配，导致空间关系建模不准确。直接用 mask 对应的 bbox 计算显式关系并嵌入注意力权重，将隐式关系建模与显式空间几何关系有效结合

### 损失函数 / 训练策略

总损失 $L_{all} = \lambda_1 L_{ce} + \lambda_2 L_{bce} + \lambda_3 L_{dice} + \lambda_4 L_{center} + \lambda_5 L_{score} + \lambda_6 L_{cont}$，前五项继承自 Maft，新增对比损失 $L_{cont}$（权重 $\lambda_6=1$）。在 ASAM 之后和每次 CLSR 之后都计算对比损失。单卡 RTX4090 训练 512 epochs，AdamW 优化器，最大学习率 0.0002，体素化尺寸 0.02m，$K=400$（ScanNet++/ScanNet200 用 500）。

## 实验关键数据

### 主实验

**ScanNetV2 验证集 / 测试集**：

| 方法 | val mAP | val AP50 | val AP25 | test mAP | test AP50 | test AP25 |
|------|---------|---------|---------|----------|----------|----------|
| Maft | 58.4 | 75.9 | 84.5 | 57.8 | 77.4 | - |
| SPFormer | 56.3 | 73.9 | 82.9 | 54.9 | 77.0 | 85.1 |
| **Relation3D** | **62.5** | **80.2** | **87.0** | **62.2** | **81.6** | **90.1** |

**ScanNet++ 验证集/测试集**：mAP 23.1→**28.2** (+5.1), 测试集 20.9→**24.2** (+3.3)

### 消融实验

对比损失 $L_{cont}$ 的逐阶段效果（越低越好）：

| 阶段 | Maft baseline | ASAM 后 | CLSR 第二次 | CLSR 第三次 |
|------|:---:|:---:|:---:|:---:|
| $L_{cont}$ | 1.057 | 0.7255 | 0.5841 | **0.5739** |

### 关键发现

- ASAM 比标准 pooling 增强了超点特征的区分度，CLSR 进一步逐阶段降低对比损失
- RSA 融入显式空间几何关系后 self-attention 的关系建模更有效
- 相比基线 Maft，在 ScanNetV2 val 上 mAP +4.1、AP50 +4.3、AP25 +2.5
- T-SNE 可视化清晰展示了同实例特征的聚集和不同实例特征的分离

## 亮点与洞察

- 精准定位了 Transformer-based 3D 实例分割的两类内部关系建模不足的问题
- 对比学习不是直接作用于 query 而是作用于超点特征，引导场景表征质量提升
- RSA 受 2D 目标检测（Relation-DETR）启发，首次将关系先验引入 3D 实例分割
- 所有改进不增加推理计算量（训练时的 CLSR 开销可控，RSA 开销很小）

## 局限与展望

- 方法建立在固定的超点预分割之上，超点质量直接影响后续表现
- 对比学习需要 GT 实例标注构建关系矩阵，无法用于无监督场景
- 在类别数极多的 ScanNet200 上提升更大，说明关系建模对复杂场景益处更明显
- 未讨论在户外点云（如自动驾驶）上的适用性

## 相关工作与启发

- **Relation-DETR** 的关系先验 bias 策略被成功迁移到 3D 场景
- **对比学习** 作为约束特征空间结构的辅助损失在 3D 领域有很好效果
- 双路径（query↔superpoint）的思路可推广到其他需要双向信息流的 Transformer 任务

## 评分

- **新颖性**: 7/10 — 个别模块有新意但整体是已知技术的组合迁移
- **实验充分度**: 9/10 — 四个数据集，消融详细，可视化丰富
- **写作质量**: 8/10 — 问题定义清晰，动机推导有说服力
- **价值**: 7/10 — 实验结果 solid，但方法的通用性有待验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Sketchy Bounding-Box Supervision for 3D Instance Segmentation](sketchy_bounding-box_supervision_for_3d_instance_segmentation.md)
- [\[CVPR 2025\] P-SLCR: Unsupervised Point Cloud Semantic Segmentation via Prototypes Structure Learning and Consistent Reasoning](p-slcr_unsupervised_point_cloud_semantic_segmentation_via_prototypes_structure_l.md)
- [\[CVPR 2025\] Any3DIS: Class-Agnostic 3D Instance Segmentation by 2D Mask Tracking](any3dis_class-agnostic_3d_instance_segmentation_by_2d_mask_tracking.md)
- [\[ICCV 2025\] JointDiT: Enhancing RGB-Depth Joint Modeling with Diffusion Transformers](../../ICCV2025/3d_vision/jointdit_enhancing_rgb-depth_joint_modeling_with_diffusion_transformers.md)
- [\[CVPR 2025\] PMA: Towards Parameter-Efficient Point Cloud Understanding via Point Mamba Adapter](pma_towards_parameter-efficient_point_cloud_understanding_via_point_mamba_adapte.md)

</div>

<!-- RELATED:END -->
