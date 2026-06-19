---
title: >-
  [论文解读] Dual-level Adaptive Self-Labeling for Novel Class Discovery in Point Cloud Segmentation
description: >-
  [ECCV 2024][3D视觉][新类发现] 提出双层自适应自标注方法，通过半松弛最优传输处理类别不平衡问题，并结合区域级表示增强点级分类器的学习，在点云分割中实现高效的新类发现。 领域现状：Novel Class Discovery (NCD) 旨在利用已知类别的语义知识来发现和分割新类别。在点云分割中…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "新类发现"
  - "点云分割"
  - "自标注学习"
  - "最优传输"
  - "不平衡学习"
---

# Dual-level Adaptive Self-Labeling for Novel Class Discovery in Point Cloud Segmentation

**会议**: ECCV 2024  
**arXiv**: [2407.12489](https://arxiv.org/abs/2407.12489)  
**代码**: [https://github.com/RikkiXu/NCD_PC](https://github.com/RikkiXu/NCD_PC)  
**领域**: 3D视觉  
**关键词**: 新类发现, 点云分割, 自标注学习, 最优传输, 不平衡学习

## 一句话总结

提出双层自适应自标注方法，通过半松弛最优传输处理类别不平衡问题，并结合区域级表示增强点级分类器的学习，在点云分割中实现高效的新类发现。

## 研究背景与动机

**领域现状**：Novel Class Discovery (NCD) 旨在利用已知类别的语义知识来发现和分割新类别。在点云分割中，现有方法(如NOPS)使用在线逐点聚类方法，通过简化的等类别大小约束来避免退化解。

**现有痛点**：
   - **类别不平衡**：点云数据中新类别固有的分布严重不平衡，等类别大小约束与实际分布不符，强制均匀约束导致生成的伪标签不能反映真实分布
   - **缺乏空间上下文**：逐点聚类忽略了物体的丰富空间上下文信息，导致语义分割的表示不够表达
   - **NOPS的bi-level优化策略**计算耗时且引入了额外超参数

**核心矛盾**：均匀分布约束在训练后期过强，导致伪标签趋向均匀而非反映真实的不平衡分布；点级学习缺少空间上下文，区域内的点本应有一致语义但被独立处理

**本文目标** 在点云分割的NCD任务中，生成适应不平衡类别分布的高质量伪标签，同时利用区域一致性提升分割质量

**切入角度**：将最优传输的等式约束松弛为带KL散度的惩罚项，自适应调节约束强度；利用DBSCAN聚类引入区域级表示

**核心 idea**：通过半松弛最优传输自适应生成不平衡伪标签，结合点级和区域级双层表示学习，互补提升新类分割性能

## 方法详解

### 整体框架

模型包含编码器 $f_\theta$、分类器 $h = [h^s, h^u]$（分别对应已知类和新类）。训练时，已知类使用交叉熵损失监督，新类通过双层自标注学习：(1) 点级自标注使用半松弛OT生成伪标签；(2) 区域级自标注先用DBSCAN将点云聚类为区域再生成区域伪标签。两个层级共享原型/分类器参数。自适应正则化动态调节KL约束强度 $\gamma$。

### 关键设计

1. **半松弛最优传输的不平衡自标注(Imbalanced Self-Labeling, ISL)**:

    - **功能**：为新类生成反映真实不平衡分布的伪标签
    - **核心思路**：将标准OT中的两个等式约束松弛为一个等式+一个KL惩罚：
    $\min_{\mathbf{Q}} \langle \mathbf{Q}, \mathbf{C} \rangle_F + \gamma \text{KL}(\mathbf{Q}^\top \mathbf{1}_M, \boldsymbol{\mu})$
    $\text{s.t. } \mathbf{Q} \in \{\mathbf{Q} \in \mathbb{R}^{M \times N} | \mathbf{Q}\mathbf{1}_N = \boldsymbol{\nu}\}$
    - 其中 $\mathbf{C} = -\log \mathbf{P}$ 为代价矩阵(基于模型预测)，$\boldsymbol{\mu}$ 为类别边际先验(均匀分布)，$\boldsymbol{\nu}$ 为样本边际分布
    - 行约束保持等式（确保每个点分配到恰好一个类），列约束松弛为KL惩罚（允许类别大小偏离均匀分布）
    - 通过引入熵约束 $-\epsilon \mathcal{H}(\mathbf{Q})$，可用高效的scaling算法求解：
    $\epsilon \langle \mathbf{Q}, \log \frac{\mathbf{Q}}{e^{-\mathbf{C}/\epsilon}} \rangle_F + \gamma \text{KL}(\mathbf{Q}^\top \mathbf{1}_M, \boldsymbol{\mu})$
    - **设计动机**：标准OT强制均匀分布不适合不平衡数据；与NOPS的bi-level优化相比，直接使用scaling算法更快更简洁

2. **自适应正则化(Adaptive Regularization, AR)**:

    - **功能**：在训练过程中动态调节KL约束的强度 $\gamma$
    - **核心思路**：监控伪标签分布 $\frac{1}{M}\mathbf{Q}^\top \mathbf{1}_M$ 与均匀分布 $\boldsymbol{\nu}$ 之间的KL距离。当KL距离持续 $T$ 个迭代低于阈值 $\rho$ 时，减小 $\gamma$：
    $\gamma_{t+1} = \lambda \cdot \gamma_t, \quad \text{if } \text{KL}(\frac{1}{M}\mathbf{Q}^\top \mathbf{1}_M, \boldsymbol{\nu}) \leq \rho \text{ for } T \text{ iters}$
    - 其中 $\lambda < 1$ 为衰减系数
    - **设计动机**：固定 $\gamma$ 要么过分约束(分布太均匀)要么过于松弛(退化解)。训练初期需要较强约束避免退化，后期需要放松让模型学习真实分布

3. **区域级表示学习(Region-level Learning)**:

    - **功能**：利用点云的空间上下文信息，在区域级别进行自标注学习
    - **核心思路**：
        - 使用DBSCAN算法将每个场景的点云聚类为若干连续区域 $\{r_k\}_{k=0}^K$
        - 对同一区域内的点特征进行平均池化得到区域表示：$\mathbf{z}_r = \text{AvgPool}(\mathbf{z}_p | r_k \text{ is same})$
        - 区域表示使用与点级相同的分类器(原型共享)进行预测
        - 对区域级预测同样使用半松弛OT生成区域伪标签 $\mathbf{Q}_r^u$
    - DBSCAN参数：$\epsilon_{\text{dbscan}} = 0.5$（确保95%点参与区域学习），min-samples = 2
    - **设计动机**：同一区域的点大概率属于同一类别，区域级表示减少了点级噪声，提供更鲁棒的语义信号；共享原型确保两个层级学习方向一致

### 损失函数 / 训练策略

总体损失：
$$\mathcal{L} = \mathcal{L}_s + \alpha \mathcal{L}_u^p + \beta \mathcal{L}_u^r$$

- $\mathcal{L}_s = -\frac{1}{N}\sum_{i=1}^N y_i^s \log p_i^s$：已知类交叉熵损失
- $\mathcal{L}_u^p = \frac{1}{M}\langle \mathbf{Q}_p^u, -\log \mathbf{P}_r^u \rangle_F$：点级自标注损失（伪标签来自点级OT，预测来自区域级—交叉监督）
- $\mathcal{L}_u^r = \frac{1}{K}\langle \mathbf{Q}_r^u, -\log \mathbf{P}_r^u \rangle_F$：区域级自标注损失
- 使用数据增强创建两个视角进行变换不变性学习
- 优化器：AdamW，初始学习率1e-3，cosine衰减到1e-5

## 实验关键数据

### 主实验（SemanticPOSS，4个split平均mIoU）

| 方法 | Novel mIoU | Known mIoU | All mIoU |
|------|-----------|-----------|---------|
| NOPS | 24.0 | 38.3 | 35.4 |
| NOPS* (优化训练设置) | 24.5 | 47.5 | 41.2 |
| **Ours** | **29.2** | **47.7** | **43.7** |
| Full (全监督上界) | - | - | 48.5 |

### 主实验（SemanticKITTI，4个split平均mIoU）

| 方法 | Novel mIoU | Known mIoU | All mIoU |
|------|-----------|-----------|---------|
| NOPS | 22.9 | 42.4 | 37.8 |
| NOPS* | 21.4 | 47.8 | 40.6 |
| **Ours** | **26.8** | **47.2** | **42.4** |
| Full (全监督上界) | - | - | 47.9 |

### 消融实验（SemanticPOSS Split 0）

| ISL | AR | Region | Building | Car | Ground | Plants | mIoU |
|-----|-----|--------|----------|-----|--------|--------|------|
| ✗ | ✗ | ✗ | 16.1 | 4.2 | 54.9 | 37.9 | 28.3 |
| ✓ | ✗ | ✗ | 57.4 | 32.1 | 18.9 | 37.2 | 34.1 |
| ✓ | ✓ | ✗ | 70.8 | 34.7 | 16.8 | 57.9 | 40.7 |
| ✓ | ✓ | ✓ | **74.6** | **41.4** | **22.5** | **66.4** | **45.7** |

### Head/Medium/Tail类别分析（SemanticPOSS）

| 方法 | Head | Medium | Tail | 说明 |
|------|------|--------|------|------|
| NOPS | 37.5 | 21.9 | 4.4 | 尾部类极差 |
| **Ours** | **45.0** | **30.8** | **11.2** | 全面提升 |
| 提升 | +7.5% | +8.9% | +6.8% | 尾部提升最显著 |

### 关键发现
- 自适应正则化使伪标签分布不再被迫趋向均匀，能够反映数据的真实不平衡特性
- 区域级学习在共享原型时带来~10%提升，不共享原型反而降低性能——两个层级需要统一的学习方向
- DBSCAN的epsilon参数并非极敏感：0.3-0.7范围内都能获得良好结果，0.7最优(mIoU 53.5)
- 比NOPS的bi-level优化策略在计算速度上有明显优势，scaling算法更适合大规模问题

## 亮点与洞察
- **半松弛OT**的formulation比NOPS更优雅：将等式约束替换为KL惩罚，既避免退化又保留灵活性，数学上更简洁
- **自适应$\gamma$调节**通过简单的衰减策略实现了训练动态的自我调节——初期强约束防退化，后期弱约束学真实分布
- **双层表示+共享原型**的设计让点级和区域级形成互补：区域级提供平滑信号指导大方向，点级提供精细粒度分割
- 通过数据增强创建两个视角进行交叉监督（点级伪标签+区域级预测），增加了标签和预测之间的信息互补

## 局限与展望
- 尾部类的分割精度虽然有提升但仍然较低（11.2%），说明长尾不平衡问题仍极具挑战
- DBSCAN对epsilon参数存在一定依赖，过大导致区域太粗放（混合不同类别），过小导致上下文信息不足
- 新类数量需要预先指定（$|C^u|$已知），实际场景中可能不知道有多少新类
- 区域级表示使用简单的平均池化，可以探索更复杂的聚合方式（如attention池化）

## 相关工作与启发
- **vs NOPS**: NOPS使用等式约束的标准OT+multi-head+overclustering，计算开销大且均匀约束不适合不平衡数据；本文用半松弛OT+自适应正则化，更简洁高效
- **vs Zhang et al.**: Zhang等人的bi-level优化需要交替优化辅助变量 $\mathbf{w}$ 和伪标签 $\mathbf{Q}$，引入额外超参数且计算慢；本文直接用scaling算法求解半松弛OT，速度更快
- **vs GrowSP**: GrowSP用于无监督点云预训练中的区域分割，本文借鉴了区域化的空间先验思想，但针对NCD任务设计了双层自标注学习

## 评分
- 新颖性: ⭐⭐⭐⭐ 半松弛OT+自适应正则化的组合新颖，双层设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 两个数据集+4个split+详细消融+参数敏感性分析+head/medium/tail分析，非常充分
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，实验分析详尽，附录提供了丰富的补充材料
- 价值: ⭐⭐⭐⭐ 解决了点云NCD中的核心不平衡问题，方法通用性强可扩展到其他NCD场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Geometric-Aware Hypergraph Reasoning for Novel Class Discovery in Point Cloud Segmentation](../../CVPR2026/3d_vision/geometric-aware_hypergraph_reasoning_for_novel_class_discovery_in_point_cloud_se.md)
- [\[NeurIPS 2025\] Novel Class Discovery for Point Cloud Segmentation via Joint Learning of Causal Representation and Reasoning](../../NeurIPS2025/3d_vision/novel_class_discovery_for_point_cloud_segmentation_via_joint_learning_of_causal_.md)
- [\[ECCV 2024\] P2P-Bridge: Diffusion Bridges for 3D Point Cloud Denoising](p2p-bridge_diffusion_bridges_for_3d_point_cloud_denoising.md)
- [\[ECCV 2024\] AEDNet: Adaptive Embedding and Multiview-Aware Disentanglement for Point Cloud Completion](aednet_adaptive_embedding_and_multiview-aware_disentanglement_for_point_cloud_co.md)
- [\[ECCV 2024\] RISurConv: Rotation Invariant Surface Attention-Augmented Convolutions for 3D Point Cloud Classification and Segmentation](risurconv_rotation_invariant_surface_attention-augmented_convolutions_for_3d_poi.md)

</div>

<!-- RELATED:END -->
