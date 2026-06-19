---
title: >-
  [论文解读] MoST: Efficient Monarch Sparse Tuning for 3D Representation Learning
description: >-
  [CVPR 2025][3D视觉][PEFT] 提出首个基于重参数化的3D PEFT方法MoST，设计Point Monarch结构化矩阵（在Monarch基础上加入KNN局部特征平滑），仅调3.6%参数在多个3D benchmark上超越全量微调。 领域现状： 3D点云预训练-微调范式（Point-MAE、ReCon、Po…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "PEFT"
  - "点云"
  - "Monarch matrix"
  - "reparameterization"
  - "3D representation learning"
  - "K-Rectify"
---

# MoST: Efficient Monarch Sparse Tuning for 3D Representation Learning

**会议**: CVPR 2025  
**arXiv**: [2503.18368](https://arxiv.org/abs/2503.18368)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: PEFT, point cloud, Monarch matrix, reparameterization, 3D representation learning, K-Rectify

## 一句话总结

提出首个基于重参数化的3D PEFT方法MoST，设计Point Monarch结构化矩阵（在Monarch基础上加入KNN局部特征平滑），仅调3.6%参数在多个3D benchmark上超越全量微调。

## 研究背景与动机

**领域现状**: 3D点云预训练-微调范式（Point-MAE、ReCon、PointGPT等）需要对整个模型进行全量微调，计算和内存开销大。参数高效微调(PEFT)在NLP/2D视觉领域已成熟，但3D点云领域仍处于探索阶段。

**现有方案的不足**:
1. **Adapter/Prompt方法**（IDPT、DAPT、PPT、PointGST）: 引入额外推理开销，且专为Transformer设计，无法泛化到Mamba/U-Net等架构
2. **LoRA等重参数化方法**: 不引入推理开销，但低秩假设捕获全局信息、忽视局部几何特征，在3D PEFT中表现差
3. Monarch矩阵虽比LoRA表达力更强，但同样无法捕获点云的局部空间结构

**核心洞察**: 通过实验发现KNN邻域内特征的L2距离与分类准确率高度相关——距离越低（局部特征越平滑），性能越好。LoRA和Monarch的局部距离高，Point Monarch则最低。

## 方法详解

### 整体框架

MoST在训练时将dense更新权重矩阵重参数化为稀疏的Point Monarch结构化矩阵，推理时合并回原始权重，因此**零推理开销**。适用于任意包含dense层的backbone。

### 关键设计

#### 模块一：Point Monarch结构化矩阵

在标准Monarch矩阵 $M = PLP^\top R$ 基础上，加入两个K-Rectify操作：
$$\text{Point Monarch} = K \cdot PLP^\top R \cdot K$$

其中 $L, R$ 是 $b$ 个块的对角分块矩阵（每块 $d/b \times d/b$），参数量 $2d^2/b \ll d^2$。$P$ 是行主序到列主序的置换（channel shuffle）。**K** 是KNN局部token线性变换，捕获点云空间局部特征。

#### 模块二：K-Rectify（参数无关的局部特征平滑）

K-Rectify通过三步实现无参数的局部信息交换：
1. **KNN grouping**: 在3D坐标空间找每个patch中心的K近邻
2. **IDW插值**: 对邻域特征用逆距离加权（Inverse Distance Weighting）计算新中心特征
3. **残差修正**: $Kx = x + \lambda x_{new}$，$\lambda$ 为超参数

矩阵形式: $K = I + \lambda(A \odot D)$，其中 $A$ 是KNN邻接矩阵，$D$ 是归一化逆距离矩阵。K本身也是稀疏的，无可学习参数。

#### 模块三：多层特征融合策略

参数无关的backbone-header对齐策略：在backbone各层输出特征上进行融合，避免"知识瓶颈"，增强预训练知识向下游任务头的传递。

### 损失函数

使用下游任务标准损失（分类用交叉熵，分割用交叉熵+Dice loss），不引入额外正则化。关键在于训练时权重更新矩阵的结构约束（Point Monarch稀疏性），而非损失函数创新。

## 实验关键数据

### 主实验表

**多backbone PEFT对比** (ScanObjectNN PB_50_RS / ModelNet40 acc.%):

| 方法 | 3D? | 无推理开销? | Point-MAE | ReCon | Mamba3D | PointGPT |
|------|-----|-----------|-----------|-------|---------|----------|
| Full FT | - | - | 85.18/93.8 | 90.01/92.5 | 92.05/94.7 | 93.4/94.1 |
| LoRA | ✗ | ✓ | 82.76/92.50 | 85.70/92.87 | 87.16/92.42 | 91.92/92.95 |
| DAPT | ✓ | ✗ | 88.27/92.99 | 89.31/93.27 | 88.55/92.87 | 93.02/94.2 |
| PointGST | ✓ | ✗ | 89.3/93.5 | 89.49/93.6 | 89.97/93.72 | 94.83/94.8 |
| **MoST (b=8)** | **✓** | **✓** | **92.92/94.77** | **93.55/95.06** | **93.30/95.18** | **97.50/96.23** |

关键亮点：MoST在Point-MAE上超越Full FT 7.74%，在PointGPT上达到97.50%!

### 消融表

**block size b的影响** (Point-MAE, PB_50_RS/MN40):

| b | 参数量(M) | PB_50_RS | ModelNet40 |
|---|----------|----------|------------|
| 32 | 0.8 | 91.95 | 94.04 |
| 16 | 1.3 | 92.71 | 94.49 |
| 8 | 2.3 | 92.92 | 94.77 |

**各分量贡献** (K-Rectify消融):
- 标准Monarch（无K）性能介于LoRA和MoST之间
- 只加前置K或后置K均有提升，前后都加（完整MoST）效果最佳
- 局部特征距离排序: MoST < Full FT < Monarch < LoRA，与性能排序一致

### 关键发现

1. **重参数化 > Adapter/Prompt**: 在所有backbone上，MoST均超越adapter类方法，且无推理开销
2. **局部特征平滑是3D PEFT的关键**: KNN邻域特征L2距离与性能高度相关
3. **泛化性极强**: Transformer、Mamba、层次架构三类backbone均适用
4. **可与其他矩阵分解组合**: 将L/R进一步Low-rank或Kronecker分解可进一步压缩参数
5. **超越全量微调**: Point-MAE (+7.74%), I2P-MAE (+3.16%), 几乎所有分类任务上超越Full FT

## 亮点与洞察

1. **首个3D重参数化PEFT**: 填补了重要空白——3D领域之前仅有adapter/prompt类方法
2. **简洁的物理直觉**: 点云局部特征平滑→更好表征→更高性能，实验验证了这一假设
3. **K-Rectify的精巧设计**: 无参数、利用3D坐标几何信息、与batch matmul兼容、保持Monarch的块对角结构
4. **实用价值**: 97.5% acc on ScanObjectNN相当惊人，且仅需3.6%参数

## 局限性

1. K-Rectify需要显式3D坐标（点云xyz），不能直接用于没有空间坐标的模态
2. KNN搜索在大规模点云上有计算开销（虽然论文认为线性变换部分开销更大）
3. 场景级分割（S3DIS）提升幅度相对object-level任务较小
4. K-Rectify的超参数λ和KNN-K需要针对不同任务调整

## 相关工作与启发

- **Monarch矩阵** (Dao et al.): 高效结构化矩阵，MoST将其扩展到不规则3D点云
- **LoRA**: 低秩重参数化PEFT的代表，但忽视局部特征→在3D PEFT中不如MoST
- **IDPT/DAPT/PointGST**: 3D adapter/prompt方法，有推理开销且限制backbone类型
- **启发**: 类似的"几何感知稀疏矩阵"思路可推广到graph neural networks和mesh processing的PEFT

## 评分

⭐⭐⭐⭐ — 研究动机清晰（局部特征→结构化矩阵），设计简洁有效（Point Monarch = Monarch + K-Rectify），实验扎实（5个backbone×多任务），97.5%在ScanObjectNN上极有竞争力；学术新颖度和简洁度均佳。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Point-SRA: Self-Representation Alignment for 3D Representation Learning](../../AAAI2026/3d_vision/point-sra_self-representation_alignment_for_3d_representation_learning.md)
- [\[CVPR 2025\] Text-Guided Sparse Voxel Pruning for Efficient 3D Visual Grounding](text-guided_sparse_voxel_pruning_for_efficient_3d_visual_grounding.md)
- [\[CVPR 2025\] Learning Class Prototypes for Unified Sparse-Supervised 3D Object Detection](learning_class_prototypes_for_unified_sparse-supervised_3d_object_detection.md)
- [\[NeurIPS 2025\] On Geometry-Enhanced Parameter-Efficient Fine-Tuning for 3D Scene Segmentation](../../NeurIPS2025/3d_vision/on_geometry-enhanced_parameter-efficient_fine-tuning_for_3d_scene_segmentation.md)
- [\[ICCV 2025\] SpatialSplat: Efficient Semantic 3D from Sparse Unposed Images](../../ICCV2025/3d_vision/spatialsplat_efficient_semantic_3d_from_sparse_unposed_images.md)

</div>

<!-- RELATED:END -->
