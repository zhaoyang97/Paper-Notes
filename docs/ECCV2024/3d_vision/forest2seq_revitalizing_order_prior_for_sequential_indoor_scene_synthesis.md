---
title: >-
  [论文解读] Forest2Seq: Revitalizing Order Prior for Sequential Indoor Scene Synthesis
description: >-
  [ECCV 2024][3D视觉][室内场景合成] 提出Forest2Seq框架，通过将无序的室内场景物体组织为层次化的场景树/森林结构，用广度优先遍历导出有意义的排列顺序作为先验知识，配合Transformer自回归解码器显著提升室内场景合成质量。 领域现状： 自动化室内场景合成已从手工先验约束优化发展到基于深度学习的方法…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "室内场景合成"
  - "自回归生成"
  - "序列顺序"
  - "场景树/森林"
  - "Transformer"
---

# Forest2Seq: Revitalizing Order Prior for Sequential Indoor Scene Synthesis

**会议**: ECCV 2024  
**arXiv**: [2407.05388](https://arxiv.org/abs/2407.05388)  
**代码**: 无公开代码  
**领域**: 3D视觉  
**关键词**: 室内场景合成, 自回归生成, 序列顺序, 场景树/森林, Transformer

## 一句话总结

提出Forest2Seq框架，通过将无序的室内场景物体组织为层次化的场景树/森林结构，用广度优先遍历导出有意义的排列顺序作为先验知识，配合Transformer自回归解码器显著提升室内场景合成质量。

## 研究背景与动机

**领域现状**: 自动化室内场景合成已从手工先验约束优化发展到基于深度学习的方法，包括自回归Transformer模型（SceneFormer、ATISS、COFS）、基于图的方法（DiffuScene）等。

**现有痛点**: 当前自回归模型将场景视为无序集合或使用随机顺序，缺乏对物体之间语义关系和层次结构的理解。ATISS通过置换不变性回避顺序问题，COFS用masked language model假设布局无序——但这些方法忽略了顺序中蕴含的有价值信息。

**核心矛盾**: 场景物体的放置有天然的层次逻辑（先放大家具再放小配件），但现有方法的随机或频率排序无法捕捉这种结构。无序导致物体间交叉、位置不合理等问题。

**本文目标**: 为自回归场景生成寻找更好的物体排列顺序先验，使模型能按照"先主后次"的直觉空间推理原则生成场景。

**切入角度**: 将场景解析为层次化树结构（功能区→主要家具→附属物品），通过聚类算法自动挖掘隐式层次，用广度优先遍历线性化为有序序列。

**核心 idea**: 顺序（order）是自回归室内场景生成中被忽视的重要先验，通过场景森林结构化排列可显著提升生成质量。

## 方法详解

### 整体框架

Forest2Seq包含两个核心模块：(1) **序列构建**——将无序场景物体集合 $\mathcal{O}=\{o_1, ..., o_n\}$ 通过Modified Euclidean Distance Clustering (MEDC)聚类解析为场景树/森林，再用BFS遍历得到有序序列 $\mathcal{S}=\pi(\mathcal{O})$；(2) **序列生成**——用decoder-only causal Transformer配合去噪策略自回归生成场景物体序列。

### 关键设计

1. **场景树构建（Scene Tree）**:

    - **功能**: 将场景中的物体按功能区域组织为层次化树结构。
    - **核心思路**: 每个物体表示为有向包围盒 $o_i = (c_i, t_i, b_i, r_i)$（类别、位置、尺寸、旋转）。使用Modified Euclidean Distance Clustering (MEDC)定义物体间距离矩阵：
    $m_{ij} = d_{ij} + \lambda \cdot (1 - \text{GIoU}(\bar{o}_i, \bar{o}_j))$
   其中 $d_{ij}$ 为中心点欧氏距离，$\text{GIoU} \in [-1, 1]$ 评估包围盒重叠程度，$\lambda=0.02$。通过DBSCAN算法（eps=0.15, min\_samples=2）将物体聚类为多个功能区。每个簇中最大物体为根节点（主要家具），其余为子节点（附属物品）。
    - **设计动机**: 符合直觉——先放沙发再放茶几，先放床再放床头柜。树结构编码了"主从"关系。

2. **场景森林（Scene Forest）**:

    - **功能**: 处理灵活物体（如柜子）可能属于多个功能区的歧义性。
    - **核心思路**: 对于DBSCAN聚类中的离群点（如可放在任何区域的柜子），将其与每个可能的父节点关联，生成树的集合（森林）。训练时从森林中随机选取一棵树进行BFS序列化。
    - **设计动机**: 单棵树强制离群物体归属某一个区域，引入虚假偏差。森林表示允许同一场景有多种合法排列，自然实现数据增强。

3. **广度优先遍历（BFS Linearization）**:

    - **功能**: 将场景树/森林转换为线性序列供Transformer处理。
    - **核心思路**: 对场景树进行BFS遍历：先生成所有根节点（各功能区主要家具），再逐层生成子节点。同层兄弟节点随机打乱以消除人为排序偏差。序列记为 $\mathcal{S}_F = \pi_F(\mathcal{O})$。
    - **设计动机**: BFS保证先放主要家具、后放附属物品的层次顺序。实验表明BFS优于DFS，因为BFS生成的序列集合内部一致性更高（hamming距离1.87 vs DFS的4.01）。

4. **Transformer解码器 + 去噪策略**:

    - **功能**: 自回归生成物体序列。
    - **核心思路**: 框架包含4个组件：
        - **Layout Encoder**: 小型ViT将二值布局掩码 $s_0 \in \mathbb{R}^{64 \times 64}$ 编码为起始token $x_0 \in \mathbb{R}^{512}$
        - **Object Encoder**: 将物体属性编码为token $x_i = [\lambda(c_i); \psi(t_i); \psi(b_i); \psi(r_i)] \in \mathbb{R}^{512}$
        - **Causal Transformer**: $\hat{x}_i = f_\theta(x_{<i}; x_0)$，使用masked自注意力和绝对位置编码
        - **Attribute Extractor**: 输出混合逻辑分布 $p(h) = \sum_{j=1}^{K} \alpha_j \text{Logistic}(\mu_j, \sigma_j)$（连续属性），softmax分布（类别属性）
    - **去噪策略**: 训练时5%概率将物体token替换为[MASK]，5%概率用随机类别替换真实类别，减少过拟合和误差传播。

### 损失函数 / 训练策略

- **损失函数**: 负对数似然：$\mathcal{L}_\theta = -\sum_{i=1}^{N} \log p_\theta(s_i | s_{<i})$，即各token条件概率的交叉熵之和
- 混合逻辑分布使用 $K=10$ 个组件
- **优化器**: AdamW，学习率1e-4，无warmup/decay
- **训练**: batch size 128, 1000 epochs, dropout 0.1, 每10 epoch验证选最优模型
- **数据增强**: 0°-360°随机旋转 + 场景森林的随机树选择
- **预训练迁移**: 小规模房型（library/living/dining）使用bedroom预训练初始化

## 实验关键数据

### 主实验（场景合成）

| 方法 | Bedroom KL↓ | Bedroom FID↓ | Living KL↓ | Living FID↓ | Dining KL↓ | Dining FID↓ | Library KL↓ | Library FID↓ |
|------|------------|-------------|-----------|-----------|-----------|-----------|-----------|-----------|
| FastSynth | 6.4 | 88.1 | 17.6 | 66.6 | 51.8 | 58.9 | 43.1 | 86.6 |
| SceneFormer | 5.2 | 90.6 | 31.3 | 68.1 | 36.8 | 60.1 | 23.2 | 89.1 |
| ATISS | 8.6 | 73.0 | 14.1 | 43.3 | 15.6 | 47.6 | 10.1 | 75.3 |
| COFS | 5.0 | 73.2 | 8.1 | 35.9 | 9.3 | 43.1 | 6.7 | 75.7 |
| DiffuScene | 5.1 | 69.0 | 8.3 | 38.2 | 7.9 | 45.8 | — | — |
| **Forest2Seq** | **4.2** | **67.9** | **5.9** | **35.2** | **5.5** | **40.2** | **5.2** | **69.1** |

### 消融实验（Living Room）

| 排列方式 | 多样性 | 不一致性 | KL↓ | FID↓ | CAS(%) |
|---------|-------|---------|-----|------|--------|
| Random(single) | 1 | 0 | 20.0 | 49.4 | 83.7 |
| Fixed(频率) | 1 | 0 | 17.9 | 49.8 | 80.1 |
| Tree+BFS | 1 | 0 | 7.90 | 36.1 | 68.1 |
| Random(multiple) | ∞ | 9.54 | 13.1 | 43.3 | 76.4 |
| Forest+DFS | 2.83 | 4.01 | 9.40 | 40.5 | 71.7 |
| **Forest+BFS** | **2.83** | **1.87** | **5.90** | **35.2** | **68.0** |

### 关键发现

- **顺序先验至关重要**: Random→Tree+BFS，KL从20.0降至7.90，FID从49.4降至36.1，证明有意义的排列顺序对场景生成质量有巨大影响
- **森林优于单树**: Tree→Forest进一步将KL从7.90降至5.90，森林表示有效处理灵活物体并提供数据增强
- **BFS优于DFS**: BFS序列集内部一致性高（hamming距离1.87 vs 4.01），主要家具放置更准确
- **位置编码很重要**: 无位置编码时KL=11.1，绝对位置编码KL=5.9，说明模型确实在利用顺序信息
- **模型极其紧凑**: 仅9.99MB参数，是COFS(19.4MB)的51%、DiffuScene(74.1MB)的13%
- **注意力可视化**: 森林排序下注意力集中在关键前序物体上，随机排序下注意力分散均匀——直接证明排序先验影响了模型内部表示
- **用户研究**: Living room 52%偏好率，Dining room 58%偏好率，远超所有baseline

## 亮点与洞察

- **重新发现顺序的价值**: 当ATISS等工作努力追求置换不变性时，本文发现引入正确顺序反而更有效——这是一个有价值的逆向思考
- **场景"语法"的发现**: 场景有类似语言的"语法"——主语(主家具)→谓语(空间关系)→宾语(附属物品)，BFS遍历自然编码了这种"语法"
- **森林表示的优雅性**: 用多棵树的集合处理灵活物体的归属歧义，既合理又自然实现数据增强
- **简单有效**: 核心方法不需要复杂的网络设计，仅通过改变输入顺序就带来巨大提升，说明数据表示的重要性
- **极小参数量**: 48%参数缩减 + 同等推理速度 + 更好质量 = 效率与效果的双赢

## 局限与展望

- 未将门窗作为额外条件，可能导致家具挡住窗户
- 缺乏物体间空间约束，偶有重叠现象
- 对非常规户型（L型、不规则形状）生成效果有限，受限于训练数据多样性
- 场景解析依赖DBSCAN超参数（eps=0.15），不同房型可能需要不同设置
- 未来可探索可学习的排序模块，端到端联合优化排序和生成

## 相关工作与启发

- **vs ATISS**: 随机打乱顺序+无位置编码实现置换不变性。Forest2Seq证明这种设计抛弃了有价值的排序信息，KL从14.1降到5.9。
- **vs COFS**: 用BART双向编码+自回归解码。Forest2Seq用更简单的decoder-only架构+更好的排序先验取得更好结果，参数量还少一半。
- **vs DiffuScene**: 全连接图+DDPM去噪。Forest2Seq用树/森林结构化表示+自回归生成，效果更好且推理快200×（0.16s vs 34.9s）。
- **vs Set2Seq**: 经典集合→序列工作，但使用学习的排序而非结构化排序。本文方法物理含义更明确。

## 评分

- 新颖性: ⭐⭐⭐⭐ 核心idea（场景树/森林排序）直觉且有效，但技术上相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ 4种房型×6种baseline + 6种排列对比消融 + 注意力可视化 + 用户研究 + 下游任务，极其充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，Figure 1-6解释力强，消融实验说服力十足
- 价值: ⭐⭐⭐⭐ "顺序先验很重要"的发现对自回归生成有普遍启发意义，方法简洁可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MegaScenes: Scene-Level View Synthesis at Scale](megascenes_scene-level_view_synthesis_at_scale.md)
- [\[CVPR 2025\] NeRFPrior: Learning Neural Radiance Field as a Prior for Indoor Scene Reconstruction](../../CVPR2025/3d_vision/nerfprior_learning_neural_radiance_field_as_a_prior_for_indoor_scene_reconstruct.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Analysis-by-Synthesis Transformer for Single-View 3D Reconstruction](analysis-by-synthesis_transformer_for_single-view_3d_reconstruction.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)

</div>

<!-- RELATED:END -->
