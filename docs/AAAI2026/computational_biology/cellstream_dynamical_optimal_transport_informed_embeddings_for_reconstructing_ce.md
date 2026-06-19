---
title: >-
  [论文解读] CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data
description: >-
  [AAAI 2026 Oral][计算生物][单细胞RNA测序] 提出 CellStream，一种将自编码器与非平衡动态最优传输（unbalanced dynamical OT）联合学习的深度学习框架，从离散时间点的单细胞快照数据中同时学习低维嵌入和连续细胞动态轨迹，在时间一致性和速度一致性上显著优于现有方法。
tags:
  - "AAAI 2026 Oral"
  - "计算生物"
  - "单细胞RNA测序"
  - "最优传输"
  - "细胞轨迹推断"
  - "降维嵌入"
  - "自编码器"
---

# CellStream: Dynamical Optimal Transport Informed Embeddings for Reconstructing Cellular Trajectories from Snapshots Data

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.13786](https://arxiv.org/abs/2511.13786)  
**代码**: [github.com/PQ-Zhang/CellStream](https://github.com/PQ-Zhang/CellStream)  
**领域**: 计算生物  
**关键词**: 单细胞RNA测序, 最优传输, 细胞轨迹推断, 降维嵌入, 自编码器

## 一句话总结

提出 CellStream，一种将自编码器与非平衡动态最优传输（unbalanced dynamical OT）联合学习的深度学习框架，从离散时间点的单细胞快照数据中同时学习低维嵌入和连续细胞动态轨迹，在时间一致性和速度一致性上显著优于现有方法。

## 研究背景与动机

### 问题定义

时间分辨的单细胞 RNA 测序（scRNA-seq）数据可以在离散时间点捕获单细胞级别的基因表达谱。但由于测序会破坏细胞，获得的只是**稀疏、静态的快照**，而非连续轨迹。核心挑战在于：如何从这些噪声快照中重建连续的细胞分化动态？

### 现有方法的不足

**几何嵌入方法**（PCA、t-SNE、UMAP、Diffusion Maps）：专注保留拓扑/几何关系，但**忽略时间结构**，不同时间点的细胞群可能在嵌入空间中重叠

**RNA 速度方法**（CellPath、VeloViz、Ocelli）：依赖 RNA 速度估计，在低 unspliced counts 数据集上不可靠，且不显式考虑跨时间点分析

**深度生成模型**（scVI、GeneFormer、TarDis）：基于 VAE/Transformer，在动态建模的可解释性方面存在挑战

**动态 OT 方法**（TIGON、CytoBridge）：需要**预计算嵌入**作为输入（如 PCA/UMAP），嵌入构建与轨迹推断**解耦**，无法利用时间结构

### 核心动机

**嵌入构建和细胞动态推断应该联合学习**——嵌入空间应显式编码时间信息，而动态学习可以反过来指导更好的嵌入构建。

## 方法详解

### 整体框架

CellStream 包含三个可学习组件：
1. **自编码器** $f_\theta^{enc}, f_\theta^{dec}$：将高维基因表达映射到低维嵌入空间并重建
2. **速度场网络** $\mathbf{v}_\phi$：在嵌入空间中建模细胞运动方向
3. **生长项网络** $g_\psi$：建模细胞增殖/凋亡导致的质量变化

通过 Block Coordinate Descent（BCD）策略交替优化自编码器和动态组件。

### 关键设计

#### 1. **非平衡动态最优传输（Unbalanced Dynamical OT）**

标准动态 OT 假设质量守恒，不适用于存在细胞增殖/凋亡的生物过程。CellStream 引入生长项 $g(t, \mathbf{x})$ 修正连续性方程：

$$\partial_t \rho + \nabla_\mathbf{x} \cdot (\mathbf{v} \rho) = g \rho$$

使用 Wasserstein-Fisher-Rao（WFR）距离同时约束传输和生长的代价：

$$\mathcal{L}_{\text{WFR}} = \int_0^T \int_{\mathbb{R}^d} (\|\mathbf{v}(t,\mathbf{x})\|^2 + \alpha g^2(t,\mathbf{x})) \rho(t,\mathbf{x}) \, d\mathbf{x} \, dt$$

其中 $\alpha=1$ 控制传输与生长代价的相对权重。

#### 2. **嵌入空间中的联合优化**

核心优化目标：

$$\min_{f^{enc}, f^{dec}, \mathbf{v}, g} \mathcal{L}_{AE}(f^{enc}, f^{dec}) + \lambda \mathcal{L}_{\text{WFR}}^{emb}(f^{enc}, \mathbf{v}, g)$$

通过 Lagrangian-Eulerian 等价性，将 WFR 损失从 PDE 形式转换为粒子形式，避免求解高维 PDE：

$$\mathcal{L}_{\text{WFR}}^{emb, par} = \sum_{j=1}^{N_0} \int_0^T (\|\mathbf{v}_\phi(t, \hat{\mathbf{z}}^j(t))\|^2 + \alpha g_\psi^2(t, \hat{\mathbf{z}}^j(t))) \hat{w}^j(t) \, dt$$

粒子轨迹通过 Neural ODE 求解器高效计算。

#### 3. **数据匹配损失**

由于非平衡设定，数据匹配损失由质量损失和 OT 损失组成：

$$\mathcal{L}_{Match} = \lambda_{Mass} \mathcal{L}_{Mass} + \lambda_{OT} \mathcal{L}_{OT}$$

- **质量损失**：$\mathcal{L}_{Mass} = \sum_i |\sum_j \hat{w}_i^j - N_i/N_0|$，确保预测的细胞群大小与观测一致
- **OT 损失**：$\mathcal{L}_{OT} = \sum_i \mathcal{W}_2(\hat{\mathbf{w}}_i / \|\hat{\mathbf{w}}_i\|_1, \mathbf{w}_i / \|\mathbf{w}_i\|_1)$，使用归一化后的 Wasserstein-2 距离衡量分布差异

### 损失函数 / 训练策略

**总损失**：$\mathcal{L} = \lambda_{AE} \mathcal{L}_{AE} + \lambda_{WFR} \mathcal{L}_{WFR}^{emb} + \lambda_{Match} \mathcal{L}_{Match}$

超参数设置：$\lambda_{AE}=10, \lambda_{WFR}=1, \lambda_{Match}=5, \lambda_{OT}=1, \lambda_{Mass}=1$

**训练策略**：
1. **PCA 初始化**自编码器
2. **BCD 交替优化**：先固定动态组件优化自编码器（同时考虑 $\mathcal{L}_{AE}$ 和动态损失），再固定自编码器优化 $\mathbf{v}_\phi, g_\psi$
3. 使用 Adam 优化器

**网络架构**：
- 自编码器：3 层隐藏层，维度 10，ReLU 激活，输出维度 2
- 速度网络：4 层隐藏层，维度 10，Tanh 激活
- 生长网络：3 层隐藏层，维度 10，Tanh 激活

## 实验关键数据

### 主实验

| 数据集 | 指标 | CellStream | VeloViz | MIOFlow | TIGON+PCA | TIGON+UMAP | TIGON+DiffMap |
|--------|------|------------|---------|---------|-----------|------------|---------------|
| EMT | VC | **0.97** | 0.88 | 0.96 | 0.66 | 0.70 | 0.68 |
| EMT | TC | **0.99** | 0.70 | 0.77 | 0.59 | 0.94 | 0.57 |
| iPSC | VC | 0.97 | 0.41 | **0.98** | 0.32 | 0.74 | 0.78 |
| iPSC | TC | 0.91 | 0.92 | 0.92 | 0.94 | **0.95** | 0.84 |
| MOSTA | VC | **0.98** | 0.83 | 0.42 | 0.85 | 0.43 | 0.98 |
| MOSTA | TC | **0.99** | 0.89 | 0.92 | 0.99 | 0.99 | 0.99 |

（VC = 速度一致性，TC = 时间一致性；高值更好）

### 消融实验

| 配置 | 关键影响 | 说明 |
|------|---------|------|
| AE 与动态解耦 | VA/TC 下降 | 联合学习嵌入和动态是必要的 |
| 去除生长项 | 质量不匹配 | 非平衡 OT 对建模增殖/凋亡至关重要 |
| 不同噪声水平 | CellStream 鲁棒 | 在高噪声下仍保持高 VA 和 TC |
| 超参数 α 敏感性 | 适度敏感 | 结果对 α 相对稳定 |

### 关键发现

1. **模拟数据上的噪声鲁棒性**：在逐步增加噪声的 5 组模拟数据上，CellStream 在 VA 和 TC 上保持稳定高分，而 Diffusion Maps 在高噪声下 TC 急剧下降
2. **EMT 数据集**：CellStream 嵌入清晰展示了上皮-间充质转化的时间结构，不同时间点细胞群有序排列；VeloViz 和 MIOFlow 的时间点则严重重叠
3. **iPSC 分叉事件**：CellStream 成功揭示了干细胞向中胚层和内胚层的分叉分化，而 MIOFlow 和 TIGON+PCA 未能完全解析
4. **空间转录组（MOSTA）**：CellStream 正确捕获了小鼠器官发生中的细胞群动态增长和漂移

## 亮点与洞察

1. **联合学习范式**：首次将嵌入学习与动态 OT 推断统一在一个端到端框架中，避免了预计算嵌入的信息损失
2. **动态反馈嵌入**：在训练自编码器时，WFR 损失提供的"实时轨迹反馈"使嵌入空间主动编码时间结构，而非事后分析
3. **粒子形式的 WFR 计算**：巧妙利用 Euler-Lagrange 等价性，避免了直接求解高维 PDE，使方法可扩展
4. 提出了两个新的评估指标（VC 和 TC），填补了缺乏 ground truth 时评估嵌入质量的空白

## 局限与展望

1. **解码器重建精度有限**：作者承认当前解码器架构不足以将动态高精度地投射回原始基因表达空间
2. **嵌入维度固定为 2**：虽然便于可视化，但限制了表达能力，更复杂的生物过程可能需要更高维嵌入
3. **计算效率**：Neural ODE 求解器的计算代价可能限制对大规模数据集的应用
4. 未探索与基因调控网络的结合
5. 缺乏对多组学数据和细胞间通讯的支持

## 相关工作与启发

- **TIGON**（Sha et al., 2024）是最相关的工作，但将嵌入和动态推断解耦
- **MIOFlow**（Huguet et al., 2022）使用测地线自编码器学习低维流形，但不处理非平衡情况
- **CytoBridge**（Zhang et al., 2025）将 RUOT 与 Mean-Field Schrödinger Bridge 结合，但同样依赖预计算嵌入
- CellStream 的联合学习思想可能启发其他需要同时学习表征和动态的领域

## 评分

- 新颖性: ⭐⭐⭐⭐ （联合嵌入+动态 OT 学习是新颖的问题定义）
- 实验充分度: ⭐⭐⭐⭐ （覆盖模拟和多个真实数据集，有消融和噪声鲁棒性分析）
- 写作质量: ⭐⭐⭐⭐⭐ （数学推导严谨，框架描述清晰）
- 价值: ⭐⭐⭐⭐ （对单细胞生物学的轨迹推断具有实用价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Variational Regularized Unbalanced Optimal Transport: Single Network, Least Action](../../NeurIPS2025/computational_biology/variational_regularized_unbalanced_optimal_transport_single_network_least_action.md)
- [\[CVPR 2026\] Cell-Type Prototype-Informed Neural Network for Gene Expression Estimation from Pathology Images](../../CVPR2026/computational_biology/cell-type_prototype-informed_neural_network_for_gene_expression_estimation_from_.md)
- [\[ICML 2026\] From Feasible to Practical: Pareto-Optimal Synthesis Planning](../../ICML2026/computational_biology/from_feasible_to_practical_pareto-optimal_synthesis_planning.md)
- [\[CVPR 2026\] CryoHype: Reconstructing a Thousand Cryo-EM Structures with Transformer-Based Hypernetworks](../../CVPR2026/computational_biology/cryohype_reconstructing_a_thousand_cryo-em_structures_with_transformer-based_hyp.md)
- [\[ICLR 2026\] Controllable Sequence Editing for Biological and Clinical Trajectories](../../ICLR2026/computational_biology/controllable_sequence_editing_for_biological_and_clinical_trajectories.md)

</div>

<!-- RELATED:END -->
