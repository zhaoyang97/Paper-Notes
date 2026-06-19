---
title: >-
  [论文解读] Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations
description: >-
  [ICML2025][图学习][图神经网络] 将消息传递建模为双曲偏微分方程组，证明节点特征的解空间由拉普拉斯矩阵的特征向量张成，从而将拓扑结构信息内嵌到节点表示中，并通过多项式近似建立与谱 GNN 的桥梁以增强其性能。 传统 GNN 通过消息传递（Message Passing）学习图的拓扑特征，本质上是将节点特征经傅里叶…
tags:
  - "ICML2025"
  - "图学习"
  - "图神经网络"
  - "双曲偏微分方程"
  - "谱图卷积"
  - "多项式滤波器"
  - "消息传递"
---

# Hyperbolic-PDE GNN: Spectral Graph Neural Networks in the Perspective of A System of Hyperbolic Partial Differential Equations

**会议**: ICML2025  
**arXiv**: [2505.23014](https://arxiv.org/abs/2505.23014)  
**代码**: [GitHub](https://github.com/YueAWu/Hyperbolic-GNN)  
**领域**: 图学习  
**关键词**: 图神经网络, 双曲偏微分方程, 谱图卷积, 多项式滤波器, 消息传递

## 一句话总结

将消息传递建模为双曲偏微分方程组，证明节点特征的解空间由拉普拉斯矩阵的特征向量张成，从而将拓扑结构信息内嵌到节点表示中，并通过多项式近似建立与谱 GNN 的桥梁以增强其性能。

## 研究背景与动机

传统 GNN 通过消息传递（Message Passing）学习图的拓扑特征，本质上是将节点特征经傅里叶变换到谱域做图卷积后再逆变换回空间域。然而问题在于：**逆变换回的空间域与图拓扑无关**——节点特征向量的基底是欧几里得空间的单位向量 $\mathbf{e}_1, \mathbf{e}_2, \dots, \mathbf{e}_d$，这些基底不携带任何拓扑信息，因此难以保证学到的特征真正编码了图结构。

现有的微分方程建模方法（如 GRAND、GraphCON）提供了可解释的方式将节点嵌入到具有特定性质的空间中。本文进一步提出：能否找到一种动力系统，使得其解空间的基底天然描述图的拓扑结构？

## 方法详解

### 1. 图上的双曲 PDE 系统

将连续域的双曲 PDE 离散化到图上。对于节点 $v_i$ 在任意维度 $l$ 上，定义图上的梯度为邻居特征差，散度为所有邻居特征差之和：

$$\nabla x_{il} := x_{il} - x_{jl}, \quad \nabla \cdot \nabla x_{il} = \sum_{v_j \in \mathcal{N}(v_i)} (x_{il} - x_{jl})$$

组合所有节点、所有维度的方程，得到矩阵形式的**双曲 PDE 系统**：

$$\frac{\partial^2 \mathbf{X}}{\partial t^2} = a^2 \hat{\mathbf{L}} \mathbf{X}$$

其中 $\hat{\mathbf{L}}$ 是拉普拉斯矩阵（取不同归一化形式），$a$ 是传播速度系数。

### 2. 解空间的推导（核心理论）

**定理 3.1**：将二阶 PDE 通过变量替换转化为一阶常系数齐次线性 ODE 系统，证明存在由基解矩阵确定的解空间。

**定理 3.3**（关键）：系数矩阵 $\mathbf{C} = \text{diag}(\mathbf{I}, a^2\hat{\mathbf{L}})$ 的特征值为 $\lambda'_1 = \cdots = \lambda'_n = 1$ 和 $\lambda'_{n+k} = a^2 \hat{\lambda}_k$，对应特征向量来自 $\hat{\mathbf{L}}$ 的特征向量 $\hat{\mathbf{u}}_k$。因此：

$$\boldsymbol{\Phi}(t) = [e^{\lambda'_1 t}\mathbf{u}'_1, \dots, e^{\lambda'_{2n} t}\mathbf{u}'_{2n}]$$

**含义**：节点特征在此范式下自动展开为拉普拉斯特征向量基的线性组合，信息沿特征向量方向传播，天然编码拓扑结构。

### 3. 多项式近似

直接求解需要特征分解（$O(n^3)$ 复杂度），且仅靠 $a$ 调节缺乏灵活性。引入多项式近似解空间：

$$\frac{\partial^2 \mathbf{X}}{\partial t^2} \approx \sum_{k=0}^{K} \theta_{tk} p_k(\mathbf{L}) \mathbf{X}$$

以 Chebyshev 多项式为例：$\frac{\partial^2 \mathbf{X}}{\partial t^2} = \sum_{k=0}^{K-1} T_k(\mathbf{L} - \mathbf{I}) \mathbf{X} \mathbf{W}_k$

### 4. 前向 Euler 数值求解

用前向 Euler 法离散化时间，初始化 $\mathbf{X}(t_0) = \phi_0(\mathbf{X})$，$\dot{\mathbf{X}}(t_0) = \phi_1(\mathbf{X})$，得到节点特征的迭代公式：

$$\mathbf{X}(t_{m+1}) = (2\mathbf{I} + \tau^2 P(\mathbf{L}, t_m)) \mathbf{X}(t_m) - \mathbf{X}(t_{m-1})$$

这一公式可直接嵌套任意多项式谱 GNN（ChebNet、BernNet、JacobiConv 等），形成 **Hyperbolic-PDE 增强范式**。

## 实验关键数据

### 节点分类（Table 4: 与谱 GNN baseline 对比）

| 方法 | Cora | CiteSeer | PubMed | Actor |
|------|------|----------|--------|-------|
| GCN | 87.14 | 79.86 | 86.74 | 33.23 |
| ChebNet | 86.67 | 79.11 | 87.95 | 37.61 |
| BernNet | 88.95 | 80.09 | 88.48 | 41.79 |
| JacobiConv | 88.98 | 80.78 | 89.62 | 41.17 |
| NFGNN | 89.82 | 80.56 | 89.89 | 40.62 |
| UniFilter | 89.49 | 81.39 | **91.44** | 40.84 |
| **Ours** | **90.82** | **81.88** | 91.36 | 42.03 |

### 增强已有谱 GNN 的效果（Table 5 精选）

| 基线 → 增强后 | Cora | PubMed | Texas | Cornell |
|--------------|------|--------|-------|---------|
| SGC → Hyp-SGC | 85.48→**86.22** | 85.36→**88.25** | 81.31→**93.61** | 72.62→**91.28** |
| APPNP → Hyp-APPNP | 88.14→**90.07** | 88.12→**90.79** | 90.98→91.31 | **91.81**→87.45 |
| GPR → Hyp-GPR | 88.57→**90.82** | 88.46→**91.36** | 92.95→**93.28** | 91.37→**92.77** |
| ChebNet → Hyp-Cheb | 86.67→**89.38** | 87.95→**90.50** | 86.22→**93.93** | 83.93→**91.06** |

在异质图（Texas、Cornell）上提升尤为显著：SGC 在 Cornell 上提升 **+18.66%**，ChebNet 在 Texas 上提升 **+7.71%**。

## 亮点与洞察

1. **理论优雅**：从双曲 PDE 出发严格证明解空间由拉普拉斯特征向量张成，为"消息传递为何能捕获拓扑"提供了数学基础
2. **即插即用范式**：不是一个独立模型，而是一个可增强任意谱 GNN 的通用框架，实验已验证对 SGC/APPNP/GPR/ChebNet/BernNet 等均有效
3. **异质图上优势明显**：传统谱 GNN 在异质图上表现差，双曲 PDE 范式可带来 5-18% 的大幅提升
4. **物理直觉**：双曲 PDE 描述波动传播，信息沿特征向量方向以有限速度传播，比热方程（扩散）模型更符合消息传递的局部性

## 局限与展望

1. **同质图增益有限**：在 Cora/CiteSeer 等同质图上提升 1-2%，与异质图上的大幅提升形成对比
2. **部分增强失效**：Hyperbolic-BernNet 在某些数据集上不如原始 BernNet（如 Cora 88.52→88.34），说明并非所有多项式基底都与双曲 PDE 范式兼容
3. **超参数依赖**：传播速度 $a$、时间步长 $\tau$、时间步数 $m$ 等引入额外超参数
4. **大规模图未验证**：最大数据集 DeezerEurope 仅 2.8 万节点，未在百万级图上测试可扩展性
5. **仅限节点分类任务**：未验证在链接预测、图分类等其他任务上的效果

## 相关工作与启发

- **微分方程视角的 GNN**：GRAND（扩散方程）、GraphCON（振荡方程），本文用双曲 PDE 提供新视角
- **谱图滤波器**：ChebNet → BernNet → JacobiConv → ChebNetII，本文提供统一增强框架
- **启发**：是否可以用其他类型 PDE（如椭圆型、抛物型）构建不同性质的解空间？不同 PDE 类型适配什么样的图任务？

## 评分
- 新颖性: ⭐⭐⭐⭐ (双曲 PDE 建模消息传递并严格证明解空间结构，理论新颖)
- 实验充分度: ⭐⭐⭐⭐ (10 个数据集 + 多个基线增强，但缺少大规模图和其他任务)
- 写作质量: ⭐⭐⭐⭐ (理论推导清晰，物理直觉好，但符号较多)
- 价值: ⭐⭐⭐⭐ (即插即用框架有实际价值，但同质图增益有限)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] On Measuring Long-Range Interactions in Graph Neural Networks](on_measuring_long-range_interactions_in_graph_neural_networks.md)
- [\[AAAI 2026\] Hyperbolic Continuous Structural Entropy for Hierarchical Clustering](../../AAAI2026/graph_learning/hyperbolic_continuous_structural_entropy_for_hierarchical_clustering.md)
- [\[ICML 2025\] LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification](llm_enhancers_for_gnns_an_analysis_from_the_perspective_of_causal_mechanism_iden.md)
- [\[ICML 2025\] Open Your Eyes: Vision Enhances Message Passing Neural Networks in Link Prediction](open_your_eyes_vision_enhances_message_passing_neural_networks_in_link_predictio.md)
- [\[ICLR 2026\] A Geometric Perspective on the Difficulties of Learning GNN-based SAT Solvers](../../ICLR2026/graph_learning/a_geometric_perspective_on_the_difficulties_of_learning_gnn-based_sat_solvers.md)

</div>

<!-- RELATED:END -->
