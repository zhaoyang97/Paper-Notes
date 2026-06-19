---
title: >-
  [论文解读] Joint Self-Supervised Video Alignment and Action Segmentation
description: >-
  [ICCV 2025][语义分割][视频对齐] 提出 VAOT/VASOT 框架，基于融合 Gromov-Wasserstein 最优传输和结构先验，首次将自监督视频对齐和动作分割统一到单一模型中，视频对齐性能优于现有方法，动作分割也达到 SOTA。 视频对齐：（帧到帧匹配）和动作分割：（帧到动作标签分配）都需要对视频进行细…
tags:
  - "ICCV 2025"
  - "语义分割"
  - "视频对齐"
  - "动作分割"
  - "最优传输"
  - "自监督学习"
  - "Gromov-Wasserstein"
---

# Joint Self-Supervised Video Alignment and Action Segmentation

**会议**: ICCV 2025  
**arXiv**: [2503.16832](https://arxiv.org/abs/2503.16832)  
**代码**: [https://retrocausal.ai/research/](https://retrocausal.ai/research/)  
**领域**: 图像分割  
**关键词**: 视频对齐, 动作分割, 最优传输, 自监督学习, Gromov-Wasserstein

## 一句话总结

提出 VAOT/VASOT 框架，基于融合 Gromov-Wasserstein 最优传输和结构先验，首次将自监督视频对齐和动作分割统一到单一模型中，视频对齐性能优于现有方法，动作分割也达到 SOTA。

## 研究背景与动机

**视频对齐**（帧到帧匹配）和**动作分割**（帧到动作标签分配）都需要对视频进行细粒度时序理解，但这两个任务此前从未被联合研究。

现有问题：

**视频对齐**: VAVA 使用标准 Kantorovich 最优传输 + 最优性先验，但难以平衡多个损失且不能处理重复动作

**动作分割**: TOT、UFSA 等方法在顺序变化、不平衡分割和重复动作场景下性能下降

**ASOT** 用融合 GW 最优传输解决了动作分割问题，但未涉及视频对齐

核心观察：两个任务都需要细粒度时序理解，多任务学习可以共享表示并互相促进。特别是，视频对齐可以显著提升动作分割性能。

## 方法详解

### 整体框架

提出两个方法：
- **VAOT**（单任务）: 基于融合 GW 最优传输的自监督视频对齐
- **VASOT**（多任务）: 统一最优传输框架，联合视频对齐 + 动作分割

### 关键设计

1. **Video Alignment Optimal Transport (VAOT)**:

    - 基于融合 Gromov-Wasserstein (FGW) 最优传输：
    $\mathcal{F}_{FGW} = (1-\alpha)\mathcal{F}_{KOT}(\mathbf{C}, \mathbf{T}) + \alpha \mathcal{F}_{GW}(\mathbf{C}^x, \mathbf{C}^y, \mathbf{T})$
    - **视觉线索** (KOT): 代价矩阵 $\mathbf{C}_{ij} = 1 - \frac{\mathbf{x}_i^\top \mathbf{y}_j}{\|\mathbf{x}_i\| \|\mathbf{y}_j\|}$ 衡量帧间视觉相似性
    - **结构先验** (GW): 通过 $\mathbf{C}^x$ 和 $\mathbf{C}^y$ 定义时序一致性约束，惩罚将时序邻近帧映射到时序远离帧的配对
    - 结构先验设计精巧：在半径 $r$ 内的邻近帧配对到远离帧时产生 $1/r$ 的代价
    - 能自然处理顺序变化和重复动作

2. **高效数值求解**:

    - 添加熵正则化 $-\epsilon H(\mathbf{T})$，通过投影镜面下降求解
    - 利用 $\mathbf{C}^x$ 和 $\mathbf{C}^y$ 的稀疏结构，每次迭代 $O(NM)$ 复杂度
    - 通常 25 次迭代内收敛，可在 GPU 上高效训练

3. **背景/冗余帧处理**:

    - 在 X 和 Y 中各添加一个虚拟帧
    - 如果某帧与所有对端帧的匹配概率都低于阈值 $\zeta$，则匹配到虚拟帧
    - 虚拟帧及其关联帧不参与损失计算

4. **VASOT - 联合多任务框架**:

    - 将 VAOT（视频对齐）和 ASOT（动作分割）整合到统一框架
    - 视频对齐进行帧到帧匹配 $(X \leftrightarrow Y)$，动作分割进行帧到动作匹配 $(X \leftrightarrow A, Y \leftrightarrow A)$
    - 共享帧编码器参数 $\theta$ 和动作嵌入 $\mathbf{A}$

### 损失函数 / 训练策略

**VAOT 损失**: 交叉熵损失对齐归一化相似度 $\mathbf{P}$ 和伪标签 $\mathbf{T}^*$

$$\mathcal{L} = -\sum_{i=1}^{N}\sum_{j=1}^{M} \mathbf{T}_{ij}^* \log \mathbf{P}_{ij}$$

**VASOT 联合损失**:

$$\mathcal{L}_{joint} = w_{align}\mathcal{L}_{xy} + w_{seg}(\mathcal{L}_{xa} + \mathcal{L}_{ya})$$

- $w_{align} = w_{seg} = 1$ 时两个任务都能获得好结果
- 不对 $\mathbf{T}^*$ 反向传播梯度
- 伪标签用增强的代价矩阵 $\tilde{\mathbf{C}} = \mathbf{C} + \rho\mathbf{R}$ 计算，$\mathbf{R}$ 为时序先验
- 动作嵌入 $\mathbf{A}$ 通过 K-Means 初始化
- 视频对齐用 ResNet-50 编码器，动作分割用 2 层 MLP 编码器

## 实验关键数据

### 主实验 (表格)

**视频对齐结果** (IKEA ASM):

| 方法 | Acc@0.1 | Acc@0.5 | Acc@1.0 | AP@5 | AP@10 | AP@15 |
|------|---------|---------|---------|------|-------|-------|
| TCC | 22.70 | 25.04 | 25.63 | 18.03 | 17.53 | 17.20 |
| VAVA | 29.12 | 29.95 | 29.10 | 26.42 | 25.73 | 25.80 |
| **VAOT** | **33.73** | **36.42** | **38.64** | **31.49** | **31.92** | **32.01** |

**动作分割结果**:

| 方法 | Breakfast MoF/F1/mIoU | 50 Salads (Eval) MoF/F1/mIoU | Desktop MoF/F1/mIoU |
|------|----------------------|------------------------------|---------------------|
| ASOT | 56.1/38.3/18.6 | 59.3/53.6/30.1 | 70.4/68.0/45.9 |
| **VASOT** | **57.5/39.0/18.8** | **60.6/57.4/34.5** | **70.9/75.1/49.3** |

### 消融实验 (表格)

**设计选择消融 (IKEA ASM)**:

| 变体 | Acc@0.1 | Acc@0.5 | Acc@1.0 | AP@5 | AP@10 | AP@15 |
|------|---------|---------|---------|------|-------|-------|
| w/o 结构先验 | 30.29 | 35.52 | 37.81 | 27.54 | 27.33 | 27.15 |
| w/o 时序先验 | 17.84 | 17.84 | 17.84 | 15.63 | 15.64 | 15.56 |
| w/o 平衡约束 | 17.84 | 20.71 | 25.24 | 15.49 | 15.69 | 15.78 |
| w/o 虚拟帧 | 30.16 | 34.49 | 36.10 | 29.57 | 29.24 | 28.87 |
| **All** | **33.73** | **36.42** | **38.64** | **31.49** | **31.92** | **32.01** |

### 关键发现

- **多任务关系的不对称性**: 动作分割对视频对齐帮助很小，但视频对齐显著提升动作分割。这可能因为视频对齐是更细粒度的帧到帧任务，为动作分割提供了更好的表示
- 时序先验 $\mathbf{R}$ 是最关键的组件，去除后性能断崖式下降
- 平衡约束优于不平衡约束（因为视频对齐的帧数远多于动作类别数，天然更平衡）
- VAOT 对超参数 $r$ 和 $\alpha$ 较为鲁棒，Acc@1.0 和 Progress 在大范围内稳定
- 在 in-the-wild 数据集（IKEA ASM）上对 VAVA 的提升最为显著

## 亮点与洞察

- 首次将视频对齐和动作分割统一到单一最优传输框架，理论优雅且实用
- FGW 结构先验设计巧妙，自然地在单一框架中处理顺序变化、背景帧和重复动作
- 多任务学习的不对称互惠发现令人启发：细粒度任务（对齐）可以提升粗粒度任务（分割），反之则不然
- 虚拟帧的简单设计有效处理了实际视频中的背景/冗余帧

## 局限与展望

- 动作类别数 K 需要预先设定为 GT 值，限制了完全无监督的应用
- 多任务中 $w_{align}$ 和 $w_{seg}$ 仅用简单等权，更高级的多任务权重学习策略可进一步提升
- ResNet-50 编码器可能限制了视频表示能力，可考虑视频基础模型
- 仅处理单一活动内的对齐和分割，跨活动场景未涉及

## 相关工作与启发

- 从 ASOT 的动作分割最优传输扩展到视频对齐，展示了优雅的方法迁移
- 与 VAVA 的对比证明：FGW + 结构先验 > Kantorovich + 最优性先验
- 深度监督、复杂多任务权重学习、联合关键点匹配与聚类等是有前景的后续方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次统一视频对齐和动作分割，FGW 用于视频对齐的适配有创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 7 个数据集，全面的消融和超参数敏感性分析
- **写作质量**: ⭐⭐⭐⭐ 公式推导严谨，整体结构清晰
- **价值**: ⭐⭐⭐⭐ 理论贡献扎实，多任务互惠的发现对社区有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CLOT: Closed Loop Optimal Transport for Unsupervised Action Segmentation](clot_closed_loop_optimal_transport_for_unsupervised_action_segmentation.md)
- [\[NeurIPS 2025\] Exploring Structural Degradation in Dense Representations for Self-supervised Learning](../../NeurIPS2025/segmentation/exploring_structural_degradation_in_dense_representations_for_self-supervised_le.md)
- [\[CVPR 2025\] Soft Self-Labeling and Potts Relaxations for Weakly-Supervised Segmentation](../../CVPR2025/segmentation/soft_self-labeling_and_potts_relaxations_for_weakly-supervised_segmentation.md)
- [\[ICCV 2025\] Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss](prompt_guidance_and_human_proximal_perception_for_hot_prediction_with_regional_j.md)
- [\[ICCV 2025\] Skeleton Motion Words for Unsupervised Skeleton-Based Temporal Action Segmentation](skeleton_motion_words_for_unsupervised_skeleton-based_temporal_action_segmentati.md)

</div>

<!-- RELATED:END -->
