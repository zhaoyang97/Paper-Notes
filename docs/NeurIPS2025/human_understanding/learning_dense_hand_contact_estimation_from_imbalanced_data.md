---
title: >-
  [论文解读] Learning Dense Hand Contact Estimation from Imbalanced Data
description: >-
  [NeurIPS 2025][人体理解][手部接触估计] 提出 HACO 框架，通过平衡接触采样（BCS）解决类别不平衡和顶点级类别平衡损失（VCB Loss）解决空间不平衡，首次在 14 个数据集（65.5 万图像）上训练稠密手部接触估计模型，在多种交互场景下达到 SOTA。 手部接触估计对理解人手交互至关重要…
tags:
  - "NeurIPS 2025"
  - "人体理解"
  - "手部接触估计"
  - "数据不平衡"
  - "类别平衡损失"
  - "大规模训练"
  - "ViT"
---

# Learning Dense Hand Contact Estimation from Imbalanced Data

**会议**: NeurIPS 2025  
**arXiv**: [2505.11152](https://arxiv.org/abs/2505.11152)  
**代码**: [有](https://github.com/dqj5182/HACO_RELEASE)  
**领域**: 其他  
**关键词**: 手部接触估计, 数据不平衡, 类别平衡损失, 大规模训练, ViT

## 一句话总结

提出 HACO 框架，通过平衡接触采样（BCS）解决类别不平衡和顶点级类别平衡损失（VCB Loss）解决空间不平衡，首次在 14 个数据集（65.5 万图像）上训练稠密手部接触估计模型，在多种交互场景下达到 SOTA。

## 研究背景与动机

手部接触估计对理解人手交互至关重要，近年来手部交互数据集覆盖了物体、另一只手、场景和人体等多种交互类型。然而，从这些数据集中有效学习稠密手部接触估计面临两个核心挑战：

**类别不平衡**：手部接触数据集中大部分区域不在接触状态。DexYCB 的非接触:接触比为 2.7:1，InterHand2.6M 高达 19.5:1，Decaf 为 21.7:1

**空间不平衡**：手部接触高度集中在指尖区域（因为运动捕捉数据集中的动作多涉及指尖的精确控制），导致模型难以泛化到手掌、手背等其他区域的接触

作者的核心洞察：这两种不平衡需要不同策略分别处理——类别不平衡需要采样策略，空间不平衡需要损失函数设计。

## 方法详解

### 整体框架

HACO 基于 ViT 骨干（HaMeR 预训练权重）：

1. **图像编码**：RGB 图像 → Patch Embedding → ViT → 图像特征 $\mathbf{F} \in \mathbb{R}^{1280 \times 16 \times 12}$
2. **接触解码**：Contact Token 作为查询，通过自注意力 + 交叉注意力 Transformer 与图像特征交互
3. **输出**：线性层 + Contact Initialization（可学习嵌入，类似残差连接）+ Sigmoid → 778 个 MANO 顶点的接触概率
4. **多尺度监督**：通过回归矩阵将 778 维映射到 336、84、21 维的粗粒度表示

### 关键设计

#### 1. 平衡接触采样（BCS）

**目标**：缓解接触/非接触类别不平衡。

**接触平衡分数**定义：

$$s_i = \frac{1}{V}(\mathbf{c}_i^\top (1 - \bar{\mathbf{c}}) - \mathbf{c}_i^\top \bar{\mathbf{c}})$$

其中 $\bar{\mathbf{c}} = \frac{1}{N}\sum_{i=1}^N \mathbf{c}_i$ 是数据集级别的平均接触概率。高分值表示接触模式偏离数据集均值更大。

**非线性分箱**：使用对数间隔将样本分为 $K$ 个 bin（曲率参数 $\beta = 5$），对高接触分数区域提供更细的分辨率：

$$\tau_k = s_{\min} + (s_{\max} - s_{\min}) \cdot \frac{\log(1 + \beta \cdot x_k)}{\log(1 + \beta)}$$

最后对各 bin 进行分层重采样，确保每个 bin 包含相同数量的样本。

#### 2. 顶点级类别平衡损失（VCB Loss）

**动机**：标准 CB Loss 对整个手只有两个权重（接触/非接触），无法区分指尖（接触频繁）和手背（接触稀少）的差异。

**核心改进**：为每个顶点 $v$ 计算独立的权重：

$$\alpha_{y_v} = \frac{1}{E_n^{(y_v)}} = \frac{1 - \beta}{1 - \beta^{n_{y_v}}}$$

其中 $n_{y_v}$ 是顶点 $v$ 在数据集中出现类别 $y$ 的次数。

最终 VCB Loss：

$$\mathcal{L}_{\text{VCB}} = \frac{1}{|V|} \sum_{v \in V} \alpha_{y_v} \ell_{\text{BCE}}(y_v, p_v)$$

**渐进式加权策略**：训练初期仅使用全局 CB Loss，VCB 分量权重随 epoch 线性增加，在最后一个 epoch 达到最大值，实现从全局到顶点自适应监督的平滑过渡。

#### 3. 辅助损失

- **正则化损失**：预测接触与数据集均值的 L1 距离，防止过度偏离
- **平滑损失**：鼓励预测少量大接触区域而非多个碎片化区域

### 损失函数 / 训练策略

总损失 = VCB Loss (权重 1.0) + 正则化损失 (权重 0.1) + 平滑损失 (权重 1.0)

训练配置：
- 骨干：HaMeR 预训练 ViT
- 优化器：AdamW，lr = $10^{-5}$，batch size 24
- 学习率衰减：第 5 和第 10 epoch 乘 0.9
- 训练 10 epochs，单卡 NVIDIA A6000
- 数据增强：随机缩放、裁剪、旋转 + 低分辨率、噪声、模糊增强

## 实验关键数据

### 主实验（MOW 数据集，手-物体接触）

| 方法 | Precision ↑ | Recall ↑ | F1-Score ↑ |
|------|-----------|----------|-----------|
| POSA | 0.134 | 0.128 | 0.101 |
| BSTRO | 0.204 | 0.126 | 0.112 |
| DECO | 0.246 | 0.235 | 0.197 |
| **HACO** | **0.525** | **0.607** | **0.522** |

HACO 的 F1 分数是 DECO 的 **2.65 倍**。

### 下游任务验证

| 接触方法 | 任务 | 关键指标 |
|---------|------|---------|
| DeepContact (3D输入) | 抓取优化 | F1=0.612, MPJPE=37.155 |
| **HACO (图像输入)** | 抓取优化 | **F1=0.666**, **MPJPE=36.520** |
| EasyHOI 原始 | 手物重建 | MPVPE=21.254 |
| **EasyHOI + HACO** | 手物重建 | **MPVPE=21.093** |

### 消融实验

| 策略 | Precision ↑ | Recall ↑ | F1-Score ↑ |
|------|-----------|----------|-----------|
| w/o BCS | 0.520 | 0.542 | 0.481 |
| **w/ BCS** | **0.525** | **0.607** | **0.522** |

| 损失函数 | Precision ↑ | Recall ↑ | F1-Score ↑ |
|---------|-----------|----------|-----------|
| CE Loss | 0.530 | 0.294 | 0.348 |
| Focal Loss | 0.518 | 0.387 | 0.409 |
| CB Loss | 0.484 | 0.534 | 0.465 |
| **VCB Loss** | **0.525** | **0.607** | **0.522** |

### 关键发现

1. **BCS 主要提升 Recall**：+12.0%（0.542→0.607），说明采样策略成功让模型学到更多接触样本
2. **VCB Loss 全面优于其他不平衡处理方法**：包括 Focal Loss、CB Loss、Asymmetric Loss 等 8 种方法，F1 从最佳 0.465（CB）提升到 0.522
3. **数据多样性至关重要**：每种交互类型（HO/HH/HS/HB）都有独特贡献，只有完整组合才能在所有测试集上达到最佳
4. **仅图像输入超越 3D 输入**：HACO 仅用 RGB 图像就在 3D 抓取优化中超越了使用完整 3D 网格的 DeepContact

## 亮点与洞察

1. **首个大规模手部接触估计模型**：整合 14 个数据集，覆盖手-物体/手-手/手-场景/手-人体 4 类交互
2. **将 CB Loss 从类别级推广到顶点级**：VCB Loss 是一个简单但有效的创新，思路可迁移到其他空间不平衡问题
3. **渐进式训练策略**：从全局平衡到局部平衡的过渡设计简洁有效
4. **Contact Initialization**：可学习的接触初始化嵌入类似残差连接，稳定训练

## 局限与展望

1. 仅预测右手接触，未处理左/右手对称问题
2. 评估时跳过完全无接触的手（Recall/F1 对纯非接触样本未定义），可能高估性能
3. 14 个数据集的接触标注方式可能不完全一致，可能引入噪声
4. 未探索时序信息（视频中的接触时序一致性）
5. 多尺度监督的 4 个分辨率级别是固定的，未自适应选择

## 相关工作与启发

- **DECO**：通过众包标注扩展接触数据，但未处理不平衡问题
- **BSTRO**：基于 Transformer 直接从视觉输入预测人体-场景接触，但数据规模有限
- **ContactOpt**：用接触优化手物体姿态，但接触来自 3D 几何而非视觉
- **启发**：VCB Loss 的顶点级加权思想可推广到其他密集预测任务的空间不平衡问题（如语义分割中的边缘/小物体）

## 评分

- 新颖性: ⭐⭐⭐ — BCS 和 VCB Loss 是对已有方法的合理扩展，创新程度适中
- 实验充分度: ⭐⭐⭐⭐⭐ — 14 数据集大规模训练 + 多场景测试 + 下游任务验证 + 全面消融
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，数据分析详实
- 价值: ⭐⭐⭐⭐ — 提供了首个通用手部接触估计模型和实用的不平衡处理方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Dynamic Reconstruction of Hand-Object Interaction with Distributed Force-aware Contact Representation](../../ICCV2025/human_understanding/dynamic_reconstruction_of_hand-object_interaction_with_distributed_force-aware_c.md)
- [\[CVPR 2025\] UniHOPE: A Unified Approach for Hand-Only and Hand-Object Pose Estimation](../../CVPR2025/human_understanding/unihope_a_unified_approach_for_hand-only_and_hand-object_pose_estimation.md)
- [\[AAAI 2026\] VPHO: Joint Visual-Physical Cue Learning and Aggregation for Hand-Object Pose Estimation](../../AAAI2026/human_understanding/vpho_joint_visual-physical_cue_learning_and_aggregation_for_hand-object_pose_est.md)
- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](../../ICCV2025/human_understanding/posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[ICCV 2025\] HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation](../../ICCV2025/human_understanding/hcceposebf_predicting_front_back_surfaces_to_construct_ultra-dense_2d-3d_corresp.md)

</div>

<!-- RELATED:END -->
