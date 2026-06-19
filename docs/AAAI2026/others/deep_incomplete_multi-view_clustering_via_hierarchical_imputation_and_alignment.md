---
title: >-
  [论文解读] Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment
description: >-
  [AAAI 2026][不完整多视图聚类] 提出 DIMVC-HIA，一个集成层次化填充与双重对齐的深度不完整多视图聚类框架，先填充缺失聚类分配再填充缺失特征，在高缺失率（70%）下仍保持稳健性能。 多视图聚类（MVC）通过整合多源异构数据提供比单视图更丰富的洞察。然而，现实场景中"所有视图完整可观测"的假设通常不成立：——…
tags:
  - "AAAI 2026"
  - "不完整多视图聚类"
  - "层次化填充"
  - "能量模型"
  - "对比对齐"
  - "缺失视图"
---

# Deep Incomplete Multi-View Clustering via Hierarchical Imputation and Alignment

**会议**: AAAI 2026  
**arXiv**: [2601.09051](https://arxiv.org/abs/2601.09051)  
**代码**: [有](https://github.com/YMBest/DIMVC-HIA)  
**领域**: LLM评测  
**关键词**: 不完整多视图聚类, 层次化填充, 能量模型, 对比对齐, 缺失视图

## 一句话总结

提出 DIMVC-HIA，一个集成层次化填充与双重对齐的深度不完整多视图聚类框架，先填充缺失聚类分配再填充缺失特征，在高缺失率（70%）下仍保持稳健性能。

## 研究背景与动机

多视图聚类（MVC）通过整合多源异构数据提供比单视图更丰富的洞察。然而，**现实场景中"所有视图完整可观测"的假设通常不成立**——传感器故障、数据损坏、传输错误等原因导致视图缺失普遍存在。

现有不完整多视图聚类（IMVC）方法分为两类：

**基于填充的方法**：重建缺失视图，但面临**错误传播**问题——糟糕的填充扭曲结构模式，反过来降低后续填充质量，形成恶性循环

**无需填充的方法**：直接从可用视图学习共享潜空间，但在高缺失率下面临实例未对齐、信息不平衡和表示不确定性等问题

核心挑战：如何在不引入偏差的情况下准确填充缺失视图，同时保持跨视图语义一致性和簇内紧凑性？

## 方法详解

### 整体框架

DIMVC-HIA 包含四个关键组件：

1. **View-specific Autoencoders**：每个视图独立的编码器-解码器 + 共享聚类预测器
2. **层次化填充模块**：先填充聚类分配 → 再填充潜在特征（由粗到细）
3. **能量基语义对齐模块**：基于 EBM 增强簇内紧凑性
4. **对比分配对齐模块**：增强跨视图一致性和聚类置信度

### 关键设计

**1. 视图特征学习与预测**

每个视图 $v$ 使用独立的 AutoEncoder 提取潜在特征：

$$H_v = E_v(X_v; \phi_v^e), \quad \hat{X}_v = D_v(H_v; \phi_v^d)$$

潜在特征通过**共享**聚类预测器映射为软聚类分配：$Q_v(i) = F(H_v(i); \vartheta) \in \mathbb{R}^K$。

**2. 层次化填充——关键创新**

**第一层：聚类分配填充（由粗到细的核心）**

核心洞察：软聚类分配空间编码了比原始特征更丰富的语义信息，更直接反映潜在聚类结构。因此先在分配空间跨视图传递信息。

步骤：
- 对每对共观测视图 $(v, v')$，提取共观测样本的软聚类分配
- 计算跨视图相似度矩阵：$S_{v,v'} = Q_v^{v,v'} (Q_{v'}^{v',v})^\top$
- 使用**标签感知对比相似度**（排除假负样本——同簇但不同索引的样本）计算视图间语义对齐得分
- 按相似度降序排列参考视图列表 $\mathcal{R}_v$
- 对每个缺失样本选择最语义对齐的可用视图进行分配填充

$$Q_v^*(i) = \begin{cases} Q_v(i), & \text{if } G(i,v) = 1 \\ Q_{\pi_v^i}(i), & \text{otherwise} \end{cases}$$

**第二层：潜在特征填充**

基于已填充的聚类分配，使用簇内统计进行特征重建：
- 确定缺失样本最可能的簇标签：$\hat{y}_v(i) = \arg\max_k Q_v^*(i, k)$
- 计算簇原型（该簇所有可用样本的平均潜在特征）
- 用簇原型作为缺失特征的填充值

$$H_v^*(i) = \begin{cases} H_v(i), & \text{if } G(i,v) = 1 \\ \mathcal{C}_v(\hat{y}_v(i)), & \text{otherwise} \end{cases}$$

**3. 能量基语义对齐（EBM）**

为每个簇定义一个视图共享的能量函数 $\mathcal{E}_{\theta_k}: \mathbb{R}^d \to \mathbb{R}^+$，低能量表示与簇的强兼容性。

找到每个簇中能量最低的"锚点"（最可靠特征），然后让所有同簇特征的能量向锚点靠拢：

$$\mathcal{L}_\text{EBM}^k = \frac{1}{|\mathcal{H}_k|} \sum_{\mathbf{h} \in \mathcal{H}_k} |\mathcal{E}_{\theta_k}(\mathbf{h}) - \varepsilon_\text{min}^k|$$

与传统的中心距离正则化不同，EBM 允许灵活塑造连续能量景观。

**4. 对比分配对齐（CAA）**

包含两个子目标：

- **对比对齐损失**：拉近同一样本在不同视图的聚类分配分布
- **熵正则化**：促进平衡的聚类分配，防止退化（所有样本分到同一簇）

$$\mathcal{L}_\text{CAA} = \frac{1}{2}\sum_v \sum_{v' \neq v} [\text{sim}(v,v') \cdot \mathcal{L}_\text{ca}^{v,v'} + \mathcal{L}_\text{reg}^{v,v'}]$$

其中相似度 $\text{sim}(v, v')$ 作为自适应权重，语义对齐度高的视图对获得更大的对齐权重。

### 损失函数 / 训练策略

总目标函数：$\mathcal{L} = \mathcal{L}_\text{REC} + \alpha \cdot \mathcal{L}_\text{EBM} + \beta \cdot \mathcal{L}_\text{CAA}$

- 先独立预训练 100 epochs AutoEncoder（仅 $\mathcal{L}_\text{REC}$）
- 再联合微调 200 epochs（完整损失）
- $\alpha = 0.1$，$\beta = 0.01$（对所有数据集固定）
- lr = 0.0001，NVIDIA RTX 3080

## 实验关键数据

### 主实验

**表 1：不同缺失率下的聚类性能（ACC/NMI/PUR）**

| 数据集 | 方法 | η=0.1 ACC | η=0.3 ACC | η=0.5 ACC | η=0.7 ACC |
|--------|------|-----------|-----------|-----------|-----------|
| BDGP | DSIMVC | 98.00 | 96.08 | 93.56 | 91.12 |
| BDGP | **DIMVC-HIA** | **98.40** | **96.25** | **95.16** | **92.32** |
| MNIST-USPS | DCG | 99.05 | 97.48 | 96.09 | 92.58 |
| MNIST-USPS | **DIMVC-HIA** | **99.10** | **97.54** | **96.48** | **93.66** |
| Fashion | ProImp | 96.26 | 93.48 | 91.01 | 86.74 |
| Fashion | **DIMVC-HIA** | **98.84** | **97.16** | **96.51** | **95.27** |
| Handwritten | GIMVC | 92.14 | 93.58 | 90.73 | 86.10 |
| Handwritten | **DIMVC-HIA** | **96.85** | **96.35** | **95.15** | **94.05** |

**表 2：高缺失率 η=0.7 下各数据集最佳对比**

| 数据集 | 最佳 Baseline | Baseline ACC | DIMVC-HIA ACC | 提升 |
|--------|-------------|-------------|---------------|------|
| BDGP | PMIMC | 91.72 | 92.32 | +0.60 |
| MNIST-USPS | ProImp | 93.42 | 93.66 | +0.24 |
| Fashion-MNIST | ProImp | 86.74 | **95.27** | **+8.53** |
| Handwritten | GIMVC | 86.10 | **94.05** | **+7.95** |

### 消融实验

在 Fashion-MNIST（η=0.5）上移除各组件的影响：

- 移除 $\mathcal{L}_\text{CAA}$：性能下降**最大**，证明对比分配对齐是最关键组件
- 移除 $\mathcal{L}_\text{EBM}$：性能明显下降，确认能量对齐的重要性
- 移除 $\mathcal{L}_\text{REC}$：性能下降，说明重建损失对稳定训练不可或缺

超参数敏感性分析：$\alpha \in [0.01, 0.10]$，$\beta \in [0.01, 0.05]$ 范围内性能稳定，无显著波动。

### 关键发现

- **高缺失率下优势更明显**：在 η=0.7 时 Fashion-MNIST 上超出 ProImp 8.53%，说明层次化填充策略在严重缺失场景下更有效
- **极低精度波动**：BDGP 上从 η=0.1 到 η=0.7 精度仅从 98.40 降至 92.32（约 6% 变化），而 DSIMVC 降了近 7%
- **收敛稳定**：损失曲线在前 25 epochs 快速下降后平稳收敛

## 亮点与洞察

1. **层次化填充的巧妙设计**：先填充语义层面的聚类分配，再用已填充的分配信息指导特征级填充，由粗到细，避免了直接特征填充的噪声问题
2. **标签感知对比相似度**：排除假负样本（同簇不同索引）的设计提高了跨视图语义匹配的质量
3. **EBM 替代距离正则化**：使用能量景观而非简单的中心距离约束，提供了更灵活的簇内紧凑性建模
4. **桥接填充与无填充范式**：综合了两类方法的优点——既做填充又通过对齐减轻填充误差

## 局限与展望

- 簇原型填充是非参数的，可能导致所有缺失样本在特征空间中坍缩到簇中心附近，缺乏多样性
- 仅在 4 个标准 benchmark 上实验，数据规模较小（最大 10,000 样本）
- 未讨论视图数量增加时的可扩展性（最多 6 个视图）
- EBM 为每个簇维护独立的能量函数，当簇数 K 很大时可能增加计算负担
- 缺失模式假设为随机缺失（MCAR），未考虑非随机缺失（MNAR）场景

## 相关工作与启发

- 与 DSIMVC、ProImp 等 SOTA 方法对比，DIMVC-HIA 的层次化填充策略在高缺失率下优势明显
- EBM 在聚类中的应用尚属新颖，可迁移至其他无监督学习任务
- 标签感知对比学习的假负样本排除策略值得在其他对比学习场景中借鉴

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 层次化先分配后特征的填充策略新颖，EBM 在聚类中的应用有创新
- **技术深度**: ⭐⭐⭐⭐ — 四个组件设计完整，数学推导清晰
- **实验充分性**: ⭐⭐⭐ — 4 个数据集 4 种缺失率，但数据规模偏小
- **实用价值**: ⭐⭐⭐ — 在多模态数据融合场景有实际意义，但计算开销可能限制大规模应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](../../NeurIPS2025/others/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)
- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[CVPR 2026\] Plug-and-Play Incomplete Multi-View Clustering via Janus-Faced Affinity Learning with Topology Harmonization](../../CVPR2026/others/plug-and-play_incomplete_multi-view_clustering_via_janus-faced_affinity_learning.md)
- [\[AAAI 2026\] DECOR: Deep Embedding Clustering with Orientation Robustness](decor_deep_embedding_clustering_with_orientation_robustness.md)
- [\[CVPR 2026\] EXOTIC: External Vision-driven Incomplete Multi-view Classification](../../CVPR2026/others/exotic_external_vision-driven_incomplete_multi-view_classification.md)

</div>

<!-- RELATED:END -->
