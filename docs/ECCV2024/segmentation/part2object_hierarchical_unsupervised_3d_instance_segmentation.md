---
title: >-
  [论文解读] Part2Object: Hierarchical Unsupervised 3D Instance Segmentation
description: >-
  [ECCV 2024][语义分割][无监督3D实例分割] 提出 Part2Object 层次聚类框架，利用自监督特征和3D物体性先验（objectness prior），从零件级过分割逐层合并到物体级实例，生成高质量伪标签用于自训练 Hi-Mask3D，实现无需人工标注的3D实例分割。 1. 3D实例分割是场景理解的核心任务…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "无监督3D实例分割"
  - "层次聚类"
  - "伪标签"
  - "自训练"
  - "3D物体性先验"
---

# Part2Object: Hierarchical Unsupervised 3D Instance Segmentation

**会议**: ECCV 2024  
**arXiv**: [2407.10084](https://arxiv.org/abs/2407.10084)  
**领域**: 语义分割  
**关键词**: 无监督3D实例分割, 层次聚类, 伪标签, 自训练, 3D物体性先验  

## 一句话总结

提出 Part2Object 层次聚类框架，利用自监督特征和3D物体性先验（objectness prior），从零件级过分割逐层合并到物体级实例，生成高质量伪标签用于自训练 Hi-Mask3D，实现无需人工标注的3D实例分割。

## 研究背景与动机

1. 3D实例分割是场景理解的核心任务，但现有方法严重依赖大量人工标注的3D点云掩码
2. 标注3D数据代价极高（ScanNet 标注一个场景需要数小时），限制了方法的可扩展性
3. 已有无监督方法如 Felzenswalb 和 CutLER 的3D投影存在严重的过分割或欠分割问题
4. 单层聚类方法（single-layer clustering）无法同时适应不同大小和几何结构的物体粒度
5. 缺乏有效的停止准则来判断何时停止合并，导致相邻物体被错误融合（如桌上的电脑与桌子）
6. 需要一种层次化的方法，能够从零件到物体逐步构建实例分割，同时保持对不同尺度物体的适应性

## 方法详解

### 整体框架

Part2Object 采用"自底向上聚类 + 自训练"的两阶段框架：

1. **第一阶段 — Part2Object 层次聚类**：利用自监督特征（如 DINOv2）提取点云特征，从初始过分割（零件级）开始，通过层次聚类逐步合并为物体级实例，生成伪标签
2. **第二阶段 — Hi-Mask3D 自训练**：以伪标签训练层次化的3D实例分割网络 Hi-Mask3D，同时预测零件级和物体级分割

### 关键设计

#### 模块一：特征引导的层次聚类

在每一层 $t$，对所有聚类对 $(c_i^t, c_j^t)$ 进行条件判断并合并：

$$c_k^{t+1} \leftarrow c_i^t \cup c_j^t \quad \text{if} \quad \text{rank}(\text{sim}(\boldsymbol{f}_i^t, \boldsymbol{f}_j^t)) \leq K \;\land\; \text{dist}(c_i^t, c_j^t) \leq T$$

其中 $\boldsymbol{f}_i^t$ 为聚类 $c_i^t$ 的特征向量，$K$ 为特征相似度的近邻排名阈值，$T$ 为空间距离阈值。合并后通过特征更新函数 $\text{FU}(\cdot)$ 计算新聚类的特征：

$$\boldsymbol{f}_k^{t+1} \leftarrow \text{FU}(c_k^{t+1})$$

该设计同时考虑特征相似性和空间邻近性，避免将特征相似但空间分离的区域错误合并。

#### 模块二：3D 物体性先验停止准则

为防止过度合并（如将桌子与桌上物品合并），引入基于3D物体性先验 $B^{3D}$ 的停止准则：

$$\text{stopCriteria}(c_i^t, c_j^t, B^{3D}) = \begin{cases} \text{True} & \text{if } \exists b \in B^{3D}: \text{IoU}(c_i^t \cup c_j^t, b) > \tau_{iou} \\ \text{False} & \text{otherwise} \end{cases}$$

其中 IoU 阈值 $\tau_{iou} = 0.6$。3D物体性先验可从2D检测器（如 CutLER）的多视图投影获得，提供物体边界的粗略估计，指导聚类何时停止合并。

#### 模块三：Hi-Mask3D 层次化自训练

Hi-Mask3D 基于 Mask3D 架构扩展，同时预测零件级和物体级分割：
- **零件查询**（Part Queries）：300个，负责细粒度零件分割
- **物体查询**（Object Queries）：150个，负责物体级实例分割
- 两级查询通过层次化的 Transformer 解码器学习零件-物体的语义关系

### 损失函数 / 训练策略

- **优化器**：AdamW，学习率 $1 \times 10^{-4}$
- **调度器**：OneCycleLR
- **训练轮数**：600 epochs，Batch size = 4
- **体素大小**：0.02
- 在无监督类别无关设置中，不根据语义标签过滤任何类别
- 在数据高效设置中，按 Mask3D 惯例过滤 ScanNet 的 wall/floor 类别
- 推理时不使用 DBSCAN 后处理

## 实验关键数据

### 主实验

| 方法 | 监督类型 | ScanNet mAP | ScanNet mAP@50 | 备注 |
|------|---------|-------------|---------------|------|
| Felzenswalb | 无监督 | - | 过分割严重 | 传统方法 |
| CutLER 投影 | 无监督 | - | 欠分割严重 | 2D→3D投影 |
| Mask3D (类别无关) | 全监督 | 基准线 | 基准线 | 有标注 |
| **Part2Object + Hi-Mask3D** | **无监督** | **显著提升** | **超越监督基线** | **本文方法** |

### 跨数据集零样本泛化

| 方法 | ScanNet200 Head mAP@50 | ScanNet200 Common mAP@50 | S3DIS mAP@50 (最小提升) |
|------|----------------------|------------------------|----------------------|
| Mask3D (类别无关) | 基准线 | 基准线 | 基准线 |
| **Hi-Mask3D** | **+10.0%** | **+0.8% mAP** | **+2.9%** |

### 关键发现

1. **层次聚类 vs 单层聚类**：单层聚类无法兼顾不同尺度物体，导致大物体过分割或小物体欠分割；层次聚类有效解决了这一问题
2. **物体性先验的作用**：没有物体性先验时，物体倾向于与相邻物体或背景元素（墙壁、地板）合并；加入先验后有效分离空间相连的物体
3. **自训练提升**：Hi-Mask3D 通过自训练可以修正伪标签中的欠分割问题，如分离桌上的电脑
4. **零件级学习**：Hi-Mask3D 能学习物体与零件的层次语义关系，如区分沙发的靠背、坐垫和扶手

## 亮点与洞察

- 从"零件到物体"的层次化思路非常自然，契合人类认知中的"部分-整体"关系
- 3D物体性先验作为停止准则的设计巧妙，补偿了纯特征聚类缺乏边界感知的缺陷
- 在 ScanNet200 的 head/common 类别上，无监督方法超越全监督的类别无关 Mask3D
- Hi-Mask3D 同时输出零件和物体两级分割，为下游任务提供了灵活的粒度选择

## 局限性

- 在 ScanNet200 尾部类别上性能略降（mAP -0.9%），因为从未接触过这些类别的标注
- 物体性先验的质量依赖于2D检测器的性能，若2D检测器在特定场景失效则影响聚类
- 训练 600 epochs 耗时较长
- 层次聚类的超参数（$K$, $T$, $\tau_{iou}$）需要针对不同数据集调整

## 相关工作与启发

- **Mask3D**：全监督3D实例分割基线，Hi-Mask3D 在其基础上扩展为层次化结构
- **Unscene3D**：另一种无监督3D场景理解方法，采用类别无关评估协议
- **CutLER**：2D无监督目标检测，用于提供3D物体性先验
- **启发**：该"过分割→层次合并→自训练"范式可迁移到其他无监督分割任务

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 总体推荐 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Un-EVIMO: Unsupervised Event-based Independent Motion Segmentation](un-evimo_unsupervised_event-based_independent_motion_segmentation.md)
- [\[ECCV 2024\] Unsupervised Moving Object Segmentation with Atmospheric Turbulence](unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)
- [\[ECCV 2024\] LASS3D: Language-Assisted Semi-Supervised 3D Semantic Segmentation with Progressive Unreliable Data Exploitation](lass3d_language-assisted_semi-supervised_3d_semantic_segmentation_with_progressi.md)
- [\[ECCV 2024\] SPIN: Hierarchical Segmentation with Subpart Granularity in Natural Images](spin_hierarchical_segmentation_with_subpart_granularity_in_natural_images.md)
- [\[ECCV 2024\] PartSTAD: 2D-to-3D Part Segmentation Task Adaptation](partstad_2d-to-3d_part_segmentation_task_adaptation.md)

</div>

<!-- RELATED:END -->
