---
title: >-
  [论文解读] Self-Organizing Visual Prototypes for Non-Parametric Representation Learning
description: >-
  [ICML 2025][目标检测][自监督学习] 提出 Self-Organizing Prototypes (SOP) 策略，用多个语义相似的支持嵌入（support embeddings）替代传统 SSL 中单一原型来表示特征空间的局部区域，并引入非参数化 MIM 任务，在检索、检测、分割等下游任务上取得 SOTA 表现。
tags:
  - "ICML 2025"
  - "目标检测"
  - "自监督学习"
  - "非参数化原型"
  - "支持嵌入"
  - "表征学习"
  - "掩码图像建模"
  - "Transformer"
---

# Self-Organizing Visual Prototypes for Non-Parametric Representation Learning

**会议**: ICML 2025  
**arXiv**: [2505.21533](https://arxiv.org/abs/2505.21533)  
**领域**: 目标检测  
**关键词**: 自监督学习, 非参数化原型, 支持嵌入, 表征学习, 掩码图像建模, Vision Transformer

## 一句话总结

提出 Self-Organizing Prototypes (SOP) 策略，用多个语义相似的支持嵌入（support embeddings）替代传统 SSL 中单一原型来表示特征空间的局部区域，并引入非参数化 MIM 任务，在检索、检测、分割等下游任务上取得 SOTA 表现。

## 研究背景与动机

现有原型式自监督学习（如 DINO、iBOT）依赖大量可学习原型来表示数据的隐含聚类。然而存在两个核心问题：

**过聚类问题**：为了覆盖特征空间，需要 $K \gg C$ 个原型（$C$ 为真实类别数），导致每个原型对应的样本过少，学到的特征倾向简单化

**欠表示问题**：单一原型难以编码一个聚类区域的所有关键特征，导致视图与原型之间的交互不够强，无法有效拉动表征

**正则化依赖**：现有方法需要 centering、sharpening、Sinkhorn-Knopp 等技巧避免坍缩，去掉这些正则化训练就会失败

作者假设：通过在局部区域引入多个语义相似的支持嵌入来增强原型的特征集，可以更好地表示特征空间并提升训练稳定性。

## 方法详解

### 整体框架

SOP 框架包含两个预训练任务：
- **全局级 SOP 损失** $\mathcal{L}_{\text{[CLS]}}$：基于 [CLS] token 的聚类分配预测
- **局部级 SOP-MIM 损失** $\mathcal{L}_{\text{patch}}$：基于 patch token 的掩码图像建模

最终损失为两者的线性组合：$\mathcal{L}_{\text{SOP}} = \lambda_1 \mathcal{L}_{\text{[CLS]}} + \lambda_2 \mathcal{L}_{\text{patch}}$，默认 $\lambda_1 = \lambda_2 = 1$。

### 核心设计：Self-Organizing Prototypes

**支持嵌入选择机制**：
1. 维护两个记忆库 $\mathbf{E}^C \in \mathbb{R}^{N_C \times d}$（[CLS] 级） 和 $\mathbf{E}^P \in \mathbb{R}^{N_p \times d}$（patch 级）
2. 从记忆库随机采样 $K$ 个锚点 $\mathbf{A} = \{a_i\}_{i=0}^K$
3. 对每个锚点用球面 $k$-NN 搜索 $k$ 个最近邻作为支持嵌入（SEs）
4. 每个 SOP 由 1 个锚点 + $k$ 个 SEs 组成，形成有向无环图结构

**软贡献权重**：SEs 对 SOP 的贡献通过与锚点的余弦相似度加权：

$$P^{\text{[CLS]}}(\mathbf{u}) = \sigma(\langle \mathbf{u}, \mathbf{D}^T \rangle) \mathbf{Y}$$

其中 $\mathbf{Y}$ 编码各 SE 的软贡献权重，$\mathbf{D}$ 包含所有 SOPs 的嵌入。

### SOP-MIM 任务

对掩码 patch token 进行重建，但使用非参数化局部 SOPs 替代可学习的离散 tokenizer：

$$\mathcal{L}_{\text{patch}} = -\sum_{l=1}^{L} m_l P^{\text{patch}}(\mathbf{z}_l^1)^T \log(P^{\text{patch}}(\hat{\mathbf{z}}_l^1))$$

### 损失函数

全局级损失（非参数化版本）：

$$\mathcal{L}_{\text{[CLS]}} = -\sum_{\mathbf{x} \sim \mathbf{X}} P^{\text{[CLS]}}(\mathbf{z}_0^1)^T \log(P^{\text{[CLS]}}(\mathbf{z}_0^2))$$

默认配置：$K=4096$ 个锚点，$k=8$ 个 SEs，记忆库大小 $N_C=65536$，$N_p=8192$，特征维度 $d=256$。

## 实验关键数据

### 主实验：ImageNet 线性评估

| 方法 | 架构 | k-NN | 线性探针 | 1% 微调 | 10% 微调 | 100% 微调 |
|------|------|------|---------|---------|---------|-----------|
| DINO | ViT-S/16 | - | - | - | - | - |
| iBOT | ViT-S/16 | - | - | - | - | - |
| **SOP** | **ViT-S/16** | - | - | - | - | - |
| iBOT | ViT-B/16 | - | - | - | - | - |
| **SOP** | **ViT-B/16** | - | - | - | - | - |
| iBOT | ViT-L/16 | - | - | - | - | - |
| **SOP** | **ViT-L/16** | **79.2** | - | - | - | - |

**关键发现**：ViT-L 上 SOP k-NN 达到 79.2%，比 iBOT 提升 +1.2%，与 I-JEPA ViT-H（79.3%）相当但参数量仅一半。

### 目标检测与分割

| 方法 | COCO AP^b | COCO AP^m | ADE20K Lin. mIoU | ADE20K UPerNet mIoU |
|------|-----------|-----------|-------------------|---------------------|
| Supervised | 49.8 | 43.2 | 35.4 | 46.6 |
| DINO | 50.1 | 43.4 | 34.5 | 46.8 |
| iBOT | 51.2 | 44.2 | 38.3 | 50.0 |
| **SOP** | **51.4** | **44.3** | **38.7** | **50.6** |

**关键发现**：SOP 在 COCO 检测上 AP^b +0.2，ADE20K 语义分割 mIoU +0.6，显著超过监督基线 +5.0 mIoU。

### 消融实验：支持嵌入数量

| SEs 数量 (CLS) | 1 | 2 | 4 | **8** | 16 |
|----------------|---|---|---|-------|-----|
| k-NN | - | - | - | **70.0** | - |

**关键发现**：全局任务在 8 个 SEs 时效果最佳；局部 MIM 任务使用单个 SE 即可。

### 图像检索 (Oxford & Paris)

在 Hard 分割上，SOP 相比 iBOT 的 mAP 提升高达 **+3.2**。鲁棒性评估中，SOP 在 ImageNet-9 的 7 个背景变换中 6 个取得最优。

## 亮点与洞察

1. **非参数化可行性**：证明不需要学习原型参数，仅用记忆库中的数据特征即可完成高质量 SSL 预训练
2. **动态区域覆盖**：SOPs 通过随机采样避免固定锚点导致的训练坍缩（Table C.7 证实固定锚点立即坍缩）
3. **缓解过聚类**：仅需 1024 个 SOPs 即可达到 iBOT 使用 8192 原型的效果（Table 10），差距仅 0.5%
4. **随模型扩展性能增益放大**：从 ViT-S 到 ViT-L，SOP 与竞争方法的差距逐步放大

## 局限性

1. 表格数据缺失具体数值（作者在 LaTeX 中使用了条件格式，实际数值需查 PDF）
2. 计算开销与 iBOT 相似（193.5h vs 193.4h），未实现效率提升
3. 仅在 ImageNet-1M 上预训练，缺乏更大规模数据集的验证
4. 未探索与 DINOv2 等使用更多数据+正则化技巧方法的结合

## 相关工作与启发

- **与 iBOT 的关系**：SOP 可视为 iBOT 的非参数化替代，消除了可学习原型和 Sinkhorn-Knopp 正则化的需要
- **与 NNCLR 的差异**：NNCLR 使用最近邻作为正样本优化 InfoNCE，SOP 使用多个近邻作为投票器优化区域级匹配
- **启发**：支持嵌入的概念可推广到其他需要表征空间局部结构的任务（如少样本学习、开放词汇检测）

## 评分

- **创新性**: ★★★★☆ — 非参数化 SOP + 支持嵌入投票机制是自监督学习的有趣范式转换
- **实用性**: ★★★★☆ — 无需额外正则化、不依赖过聚类，实用价值高
- **实验完整性**: ★★★★★ — 涵盖检索/检测/分割/迁移/鲁棒性/消融，极为全面
- **写作质量**: ★★★★☆ — 结构清晰，公式严谨

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Understanding the Emergence of Multimodal Representation Alignment](understanding_the_emergence_of_multimodal_representation_alignment.md)
- [\[NeurIPS 2025\] Automated Detection of Visual Attribute Reliance with a Self-Reflective Agent](../../NeurIPS2025/object_detection/automated_detection_of_visual_attribute_reliance_with_a_self-reflective_agent.md)
- [\[AAAI 2026\] CASL: Curvature-Augmented Self-supervised Learning for 3D Anomaly Detection](../../AAAI2026/object_detection/casl_curvature-augmented_self-supervised_learning_for_3d_anomaly_detection.md)
- [\[NeurIPS 2025\] DETree: DEtecting Human-AI Collaborative Texts via Tree-Structured Hierarchical Representation Learning](../../NeurIPS2025/object_detection/detree_detecting_human-ai_collaborative_texts_via_tree-structured_hierarchical_r.md)
- [\[ICLR 2026\] PAANO: Patch-Based Representation Learning for Time-Series Anomaly Detection](../../ICLR2026/object_detection/paano_patch-based_representation_learning_for_time-series_anomaly_detection.md)

</div>

<!-- RELATED:END -->
