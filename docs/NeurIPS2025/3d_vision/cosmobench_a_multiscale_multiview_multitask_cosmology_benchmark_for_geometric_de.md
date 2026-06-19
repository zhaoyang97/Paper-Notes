---
title: >-
  [论文解读] CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning
description: >-
  [NeurIPS 2025][3D视觉][geometric deep learning] 提出 CosmoBench——目前最大的宇宙学几何深度学习基准，包含 3.4 万点云和 2.5 万有向树，覆盖多尺度、多视角、多任务，并揭示简单线性模型有时能超越大型 GNN。 宇宙学模拟数据丰富但缺乏统一基准：宇宙学模拟产生的点云和…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "geometric deep learning"
  - "cosmology benchmark"
  - "点云"
  - "图神经网络"
  - "merger tree"
---

# CosmoBench: A Multiscale, Multiview, Multitask Cosmology Benchmark for Geometric Deep Learning

**会议**: NeurIPS 2025  
**arXiv**: [2507.03707](https://arxiv.org/abs/2507.03707)  
**代码**: [GitHub](https://github.com/TeresaHuang/CosmoBench)  
**领域**: 3D视觉  
**关键词**: geometric deep learning, cosmology benchmark, point cloud, graph neural network, merger tree

## 一句话总结

提出 CosmoBench——目前最大的宇宙学几何深度学习基准，包含 3.4 万点云和 2.5 万有向树，覆盖多尺度、多视角、多任务，并揭示简单线性模型有时能超越大型 GNN。

## 研究背景与动机

**宇宙学模拟数据丰富但缺乏统一基准**：宇宙学模拟产生的点云和 merger tree 数据极为丰富，但社区缺少像 ShapeNet/ModelNet 那样统一的基准来系统评估机器学习方法，导致方法间难以公平比较。

**几何深度学习在宇宙学中的突破远未到来**：GDL 已在计算机视觉、结构生物学、气候科学等领域取得显著进展，但宇宙学尚未观察到类似突破，需要大规模基准来催化方法创新。

**现有基准规模与多样性不足**：Balla et al. 提出的基准仅包含 3,560 个 Quijote 点云，在数据规模、物理尺度、数据模态和任务多样性上均有限。

**从模拟到观测的鸿沟需要弥合**：推断宇宙学参数、预测星系速度、重建 merger tree 等任务具有直接的科学价值（如推断宇宙膨胀率、补偿硬件限制等），但现有方法尚不成熟。

**简单方法 vs 复杂模型的权衡未被充分探索**：社区倾向于使用越来越复杂的深度学习模型，但在宇宙学任务上，简单物理驱动方法可能同样有效甚至更优，需要系统比较。

**多尺度行为差异待揭示**：宇宙学中的线性（大尺度）和非线性（小尺度）物理行为差异显著，不同方法在不同尺度上的表现规律尚不清楚。

## 方法详解

### 整体框架

CosmoBench 从三大宇宙学模拟套件（Quijote、CAMELS-SAM、CAMELS）中策划数据，构建包含 **点云数据集**（3 个，共 34,752 个点云）和 **有向树数据集**（CS-Trees，24,996 棵树）的多尺度、多视角基准。涵盖四类任务：从点云预测宇宙学参数（图级回归）、从位置预测星系/暗物质晕速度（节点回归）、从 merger tree 预测宇宙学参数（图级回归）、重建细粒度 merger tree（节点分类/图超分辨率）。对每类任务提供物理基线、线性模型基线和深度学习基线。

### 关键设计一：多尺度点云数据集构建

- **功能**：从三个模拟套件提取不同空间尺度的点云数据集。
- **核心思路**：Quijote（1000 cMpc/h，大尺度线性域）含 32,752 个暗物质晕点云；CAMELS-SAM（100 cMpc/h，中间非线性尺度）含 1,000 个星系点云并附带 merger tree；CAMELS（25 cMpc/h，深度非线性尺度）含 1,000 个流体力学模拟星系点云。每个点云包含 3D 位置和速度，标注对应的宇宙学参数 $(\Omega_m, \sigma_8)$。
- **设计动机**：覆盖从线性到深度非线性的完整物理尺度谱，使研究者能系统分析不同方法在不同物理制度下的行为差异。

### 关键设计二：基于不变量特征的 GNN 消息传递

- **功能**：设计保持 E(3) 不变性的图神经网络，同时支持高阶消息传递。
- **核心思路**：构建半径图后，边特征使用归一化距离 $d_{ij}/R_c$ 和两个点积不变量；通过 Delaunay 三角剖分识别边邻居，利用欧几里得和 Hausdorff 距离提取节点-节点、节点-边、边-边的 E(3) 不变特征 $\text{Inv}(\cdot,\cdot)$；消息传递同时在节点和边嵌入上进行，使用可学习的非线性更新函数。
- **设计动机**：宇宙学点云具有平移和反射对称性，GNN 必须尊重这些物理对称性；高阶消息传递（边-边交互）旨在捕获超越两点相关函数的高阶聚类信息。

### 关键设计三：线性最小二乘基线（LLS）

- **功能**：用仅 49 个参数的线性模型作为强基线，从成对距离统计量预测宇宙学参数。
- **核心思路**：对每个点云，在 12 个不同截断半径 $R_c$ 下计算点对距离分布的均值、标准差和 $(1/3, 2/3)$ 分位数，得到 48 个特征；用贪心策略在验证集上选择截断半径，然后用带偏置的最小二乘拟合预测目标参数。
- **设计动机**：物理启发的简单模型可作为复杂方法的"sanity check"——如果 GNN 无法显著超越仅用 49 个参数的线性模型，说明深度模型可能未有效利用高阶信息，也为社区提供了极低计算成本的参考基线。

### 关键设计四：Merger Tree 数据集与超分辨率任务

- **功能**：从 CAMELS-SAM 提取有向 merger tree 并设计时间超分辨率任务。
- **核心思路**：选取根节点质量大于 $10^{13} M_\odot/h$ 的树，修剪小质量子树以去除信息泄露风险，每个模拟选 25 棵树共得到 24,996 棵。超分辨率任务通过遮蔽偶数时间步将树粗化，然后在每个合并节点添加虚拟节点，训练分类器判断被遮蔽的合并节点是否确实存在。
- **设计动机**：Merger tree 记录了暗物质晕的形成历史，是点云之外的重要数据模态；时间超分辨率任务模拟了因存储限制导致的时间分辨率不足问题，直接服务于即将到来的 Euclid/LSST 等大型巡天项目。

## 损失函数与训练

- 点云宇宙学参数预测和速度预测均使用 MSE 损失，评估指标为决定系数 $R^2$，不确定性由测试集 bootstrap 标准差给出
- Merger tree 节点分类使用二元交叉熵损失，评估指标为准确率
- GNN 和 DeepSets 使用 Adam 优化器训练，数据集按 60/20/20 划分训练/验证/测试集

## 实验关键数据

### 表1：点云宇宙学参数预测 ($R^2$ ↑)

| 方法 | Quijote $\Omega_m$ | Quijote $\sigma_8$ | 参数量 | CAMELS-SAM $\Omega_m$ | CAMELS-SAM $\sigma_8$ | CAMELS $\Omega_m$ | CAMELS $\sigma_8$ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2PCF+MLP | 0.85 | 0.84 | 11K | 0.73 | 0.82 | 0.84 | 0.30 |
| LLS (49参数) | 0.83 | 0.80 | 49 | 0.77 | 0.82 | 0.78 | 0.28 |
| GNN | 0.80 | 0.77 | 671K | 0.75 | 0.83 | 0.78 | 0.24 |
| GNN (w/o edgeMP) | 0.80 | 0.79 | 128K | 0.72 | 0.84 | 0.80 | 0.27 |

**关键发现**：仅 49 个参数的 LLS 模型性能与拥有数十万参数的 GNN 相当甚至更优；去除边-边消息传递对 GNN 性能无显著影响；$\sigma_8$ 在小体积 CAMELS 数据集上预测严重退化。

### 表2：点云速度预测 ($R^2$ ↑)

| 方法 | Quijote v | CAMELS-SAM v | CAMELS v |
|------|:---:|:---:|:---:|
| 线性理论 (oracle*) | 0.377 | 0.237 | 0.297 |
| LLS (60参数) | **0.435** | 0.211 | 0.249 |
| GNN (126K参数) | 0.410 | **0.287** | 0.253 |

**关键发现**：大尺度上 LLS 超越 GNN 和线性理论 oracle；GNN 在中等尺度（CAMELS-SAM）最优；ML 方法无需宇宙学参数先验知识即可超越需要先验的线性理论。

## 亮点

1. **规模空前的宇宙学 ML 基准**：34K 点云 + 25K merger tree，来自超 4100 万核时的模拟，覆盖三个空间尺度
2. **"少即多"的重要发现**：49 个参数的线性模型可以匹敌甚至超越数十万参数的 GNN，揭示当前深度模型在宇宙学任务上的瓶颈
3. **多模态多任务的统一接口**：提供 PyTorch 统一接口，涵盖点云和有向树两种数据模态、四类任务
4. **从模拟到观测的初步桥梁**：引入 redshift space 速度预测变体，向真实观测场景迈进

## 局限性

1. CAMELS 和 CAMELS-SAM 数据集仅各 1,000 个样本，对数据驱动方法仍显不足
2. 仅考虑 $\Omega_m$ 和 $\sigma_8$ 两个宇宙学参数（Quijote 额外含 3 个），未覆盖暗能量等重要参数
3. 当前 GNN 在大多数任务上未显著超越简单基线，说明基准的上界还远未被探索
4. Merger tree 超分辨率任务仅使用 200 棵最大的树，样本量有限
5. Redshift space 处理较为简化（仅沿 z 轴），距真实巡天观测仍有差距

## 相关工作

- **宇宙学中的 GDL**：已有工作将 GNN 用于从星系分布推断宇宙学参数（Villanueva-Domingo et al., Makinen et al.），但现成方法在仅用位置信息的任务上表现不佳
- **点云基准**：CV 领域有 ShapeNet、ModelNet，生物学有 AlphaFold DB、MoleculeNet，但宇宙学缺乏统一基准；Balla et al. 的先驱工作规模较小（3,560 个点云）
- **图基准**：OGB、TU Datasets 等主要覆盖生物和社交网络，宇宙学图数据未被纳入；CosmoBench 响应了 Dwivedi et al. 改进图基准的倡议

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首个大规模宇宙学几何学习基准，多尺度多模态设计独特
- 实验充分度: ⭐⭐⭐⭐ — 提供物理、线性、深度学习三类基线，消融实验充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，物理背景交代充分
- 价值: ⭐⭐⭐⭐ — 填补了宇宙学 ML 基准的空白，发现了简单模型的竞争力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Copresheaf Topological Neural Networks: A Generalized Deep Learning Framework](copresheaf_topological_neural_networks_a_generalized_deep_learning_framework.md)
- [\[CVPR 2025\] A2Z-10M+: Geometric Deep Learning with A-to-Z BRep Annotations for AI-Assisted CAD Modeling and Reverse Engineering](../../CVPR2025/3d_vision/a2z-10m_geometric_deep_learning_with_a-to-z_brep_annotations_for_ai-assisted_cad.md)
- [\[ICCV 2025\] SiM3D: Single-Instance Multiview Multimodal and Multisetup 3D Anomaly Detection Benchmark](../../ICCV2025/3d_vision/sim3d_single-instance_multiview_multimodal_and_multisetup_3d_anomaly_detection_b.md)
- [\[ICCV 2025\] EgoM2P: Egocentric Multimodal Multitask Pretraining](../../ICCV2025/3d_vision/egom2p_egocentric_multimodal_multitask_pretraining.md)
- [\[NeurIPS 2025\] Learning Generalizable Shape Completion with SIM(3) Equivariance](learning_generalizable_shape_completion_with_sim3_equivariance.md)

</div>

<!-- RELATED:END -->
