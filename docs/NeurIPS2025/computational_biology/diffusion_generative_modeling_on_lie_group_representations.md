---
title: >-
  [论文解读] Diffusion Generative Modeling on Lie Group Representations
description: >-
  [NeurIPS 2025 Spotlight][计算生物][李群表示] 提出在李群表示空间：（而非李群本身）上构建扩散过程的新理论框架，通过广义分数匹配将非阿贝尔李群的弯曲动力学映射到欧几里得空间中，实现无模拟训练的李群扩散模型，并证明标准分数匹配是其平移群的特例。 - 问题：许多科学数据（分子构象、旋转、刚体变换）天然分…
tags:
  - "NeurIPS 2025 Spotlight"
  - "计算生物"
  - "李群表示"
  - "广义分数匹配"
  - "随机微分方程"
  - "分子构象生成"
  - "流形扩散"
---

# Diffusion Generative Modeling on Lie Group Representations

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2502.02513](https://arxiv.org/abs/2502.02513)  
**代码**: 无  
**领域**: 计算生物  
**关键词**: 李群表示, 广义分数匹配, 随机微分方程, 分子构象生成, 流形扩散

## 一句话总结

提出在李群**表示空间**（而非李群本身）上构建扩散过程的新理论框架，通过广义分数匹配将非阿贝尔李群的弯曲动力学映射到欧几里得空间中，实现无模拟训练的李群扩散模型，并证明标准分数匹配是其平移群的特例。

## 研究背景与动机

- **问题**：许多科学数据（分子构象、旋转、刚体变换）天然分布在弯曲流形上（如 SO(3), SE(3)），而非欧几里得空间
- **现有困难**：流形上的扩散算法面临两大难题——(1) 一般流形上参数化向量场尚未解决；(2) Langevin 更新需投影回流形以保持几何结构
- **实际困境**：即使数据对称性明确（如蛋白质的扭转角空间），AlphaFold3 仍在笛卡尔坐标上做扩散，因为流形扩散性能增益不明显
- **核心问题**：能否在保留欧几里得空间平坦性优势的同时，利用李群的对称结构？本文的答案是——在**表示空间** $\text{Im}(\rho_X) \subseteq GL(X)$ 上构建扩散

## 方法详解

### 整体框架

不在李群 $G$ 上做扩散，也不在数据空间 $X$ 上做普通扩散，而是利用 $G$ 对 $X$ 的作用表示 $\rho_X: G \to GL(X)$：

1. 将数据点 $\mathbf{x} \in X$ 通过流坐标 $\boldsymbol{\tau}$ 参数化（即找到李群变换参数）
2. 在流坐标空间（欧几里得，可用高斯）上定义前向扩散
3. 通过李群作用将流坐标映射回数据空间
4. 训练网络预测**广义分数**（沿基本向量场方向的对数密度导数）
5. 反向 SDE 由广义分数引导，在数据空间 $X$ 中生成样本

### 关键设计

#### 1. 广义分数匹配 (Generalized Score Matching)

将标准分数函数 $\nabla \log p(\mathbf{x})$ 替换为沿李代数基本向量场方向的导数：

$$\mathcal{L} = \mathbf{\Pi}(\mathbf{x})^\top \nabla$$

其中 $\mathbf{\Pi}(\mathbf{x})$ 是基本向量场矩阵。广义 Fisher 散度为：

$$D_{\mathcal{L}}(p||q_\theta) = \int_X p(\mathbf{x})|\mathcal{L}\log p(\mathbf{x}) - \mathbf{s}_\theta(\mathbf{x})|^2 d\mathbf{x}$$

#### 2. 三个充分条件

要使广义分数匹配可用于 Langevin 动力学，需满足：

| 条件 | 含义 | 数学要求 |
|------|------|----------|
| 完备性 | $\mathbf{\Pi}$ 保留密度的全部信息 | dim($G/G_\mathbf{x}$) $\geq$ dim($X$) 几乎处处 |
| 齐性 | 任意两点可通过 $G$ 变换连接 | $X$ 是 $G$ 的齐性空间 |
| 交换性 | 各方向的 Langevin 更新互相独立 | $[\mathcal{L}_A, \mathcal{L}_B]f(\mathbf{x}) = 0$ |

关键洞察：**非阿贝尔群也可满足交换性条件**——李代数中不对易的元素，其在 $X$ 上诱导的流可以对易（如 $\mathfrak{so}(3)$ 在某些表示下）。

#### 3. 核心定理：配对 SDE

**定理 3.1**：存在一对 SDE，前向过程精确可解，反向过程由广义分数引导：

**前向 SDE**：
$$d\mathbf{x} = \left[\beta(t)\mathbf{\Pi}(\mathbf{x})\mathbf{f}(\mathbf{x}) + \frac{\gamma(t)^2}{2}\rho_X(\Omega)\right]dt + \gamma(t)\mathbf{\Pi}(\mathbf{x})d\mathbf{W}$$

**精确解**：$\mathbf{x}(t) = \prod_{i=1}^n e^{\tau_i(t)A_i} \mathbf{x}(0)$，其中 $\boldsymbol{\tau}(t)$ 服从简单 SDE

**反向 SDE** 包含三个额外项：
- **二次 Casimir 元素** $\rho_X(\Omega) = \sum_i A_i^2$：补偿流坐标曲率导致的轨道偏离
- **散度修正** $\mathbf{\Pi}\nabla^\top \cdot \mathbf{\Pi}$：修正非常系数 SDE 的概率流
- **广义分数** $\mathbf{\Pi}\mathcal{L}\log p_t$：引导采样

#### 4. 标准分数匹配是特例

当 $G = T(n)$（平移群），$\mathbf{\Pi} = I$，Casimir 和散度项消失，退化为标准 DDPM。

#### 5. SO(2) × R⁺ 示例

取 $G = SO(2) \times \mathbb{R}_+$ 作用于 $X = \mathbb{R}^2$：
- 基本向量场 $\mathbf{\Pi}(\mathbf{x}) = \begin{pmatrix} x & -y \\ y & x \end{pmatrix}$
- SDE 沿径向（缩放）和角度（旋转）分解
- 渐近分布 $p_T$ 为 $(e^{\eta_r}\cos\eta_\theta, e^{\eta_r}\sin\eta_\theta)$，可通过采样两个高斯变量得到

### 损失函数 / 训练策略

- 标准去噪分数匹配目标，但目标是广义分数而非标准分数：

$$\mathbb{E}_t\left\{w(t)\mathbb{E}_{\mathbf{x}(0)}\mathbb{E}_{\mathbf{x}(t)}\left[|\mathbf{s}_\theta(\mathbf{x}(t),t) - \mathcal{L}\log p_t(\mathbf{x}(t)|\mathbf{x}(0))|^2\right]\right\}$$

- 因为条件广义分数在高斯假设下有闭式解 $-\Sigma(t)^{-1}\boldsymbol{\eta}_t$，训练简化为预测噪声（类似 DDPM）
- 使用方差保持调度器，训练/采样算法见 Algorithm 1/2

## 实验关键数据

### 主实验

#### 2D/3D/4D 分布生成

- 使用 $G = SO(d) \times \mathbb{R}_+$ 在 $X = \mathbb{R}^d$ 上
- 成功对混合高斯、环面、莫比乌斯带等进行建模
- 利用对称性可将学习降维（如径向对称分布仅需学习径向分数）

#### 旋转 MNIST 桥接

| 模型 | 平均准确率 ↑ | 平均 FID ↓ |
|------|-------------|-----------|
| **GSM (本文)** | **0.96 ± 0.02** | **85.8 ± 15.7** |
| BBDM | 0.80 ± 0.10 | 133.4 ± 19.0 |

- 仅学习 1 维分数（旋转角），大幅简化问题
- BBDM 在完整像素空间操作，可能生成错误数字

#### QM9 分子构象生成

- $G = (SO(3) \times \mathbb{R}_+)^N$ 作用于 $X = \mathbb{R}^{3N}$
- 生成构象的 UFF 能量与真实构象接近，甚至略低
- 李群扩散 $\Delta_\theta = -0.2159$ vs 标准扩散 $\Delta_\gamma = -0.2144$

#### CrossDocked2020 分子对接

| 方法 | RMSD (Å) ↓ |
|------|-----------|
| **GSM (本文, SE(3))** | **2.91 ± 1.0** |
| RSGM (黎曼扩散) | 5.6 ± 1.2 |
| BBDM (欧几里得) | 2.92 ± 1.57 |

### 消融实验

- 2D 混合高斯的 W2 距离对比中，李群扩散与标准扩散在能力上等价（可学习任意分布）
- 但对称选择匹配数据结构时，有效维度降低，学习效率提升

### 关键发现

1. 适当选择李群可**降低有效维度**（旋转 MNIST 从 784d 降到 1d）
2. 可在**非平凡分布间桥接**（BBDM 做不到的）
3. SE(3) 引导的分子对接显著优于黎曼扩散方法
4. 框架可自然扩展到 flow matching（附录 F）

## 亮点与洞察

- **理论统一性**：标准扩散（平移群）、黎曼扩散（群上扩散）、本文方法（表示空间扩散）是同一框架的不同实例
- **Casimir 元素的几何直觉**：补偿曲率导致的轨道偏离，如 SO(2) 中切向运动会使点偏离圆轨道，Casimir 将其拉回
- **维度压缩**：当数据结构与群对称匹配时，分数学习的有效维度可大幅降低
- **simulation-free 训练**：前向 SDE 有精确解，避免了非阿贝尔群上扩散的数值模拟困难

## 局限与展望

1. 交换性条件限制了可用的群-空间组合，需要仔细选择
2. 分子对接实验规模较小（与 DiffDock 等未直接对比）
3. 高维群（如 SO(n) for large n）的计算效率待验证
4. 当前仅验证了 SO(2), SO(3), SE(3) 等常见群，更复杂的对称群有待探索
5. 非齐性空间的推广（限制在轨道内生成）有理论但缺乏实验

## 相关工作与启发

- **De Bortoli et al. (2022)**：黎曼流形上的扩散，与本文在李群情况下动力学形式相似，但需数值模拟
- **Corso et al. (2023) DiffDock**：SE(3) 扩散用于分子对接，直接在群上操作
- **Kim et al. (2022)**：用双射映射将非线性问题线性化，与本文的表示空间思想有联系
- **启发**：选择正确的对称群可以将复杂问题降维到简单问题，这一思想可推广到更多科学计算场景

## 评分

| 维度 | 分数 | 评价 |
|------|------|------|
| 新颖性 | ★★★★★ | 全新理论框架，统一了多种扩散范式，数学优美 |
| 技术深度 | ★★★★★ | 严谨的数学推导，证明了新类型 SDE 的精确解 |
| 实验充分性 | ★★★☆☆ | 实验规模偏小，与 SOTA 方法对比不足 |
| 实用价值 | ★★★★☆ | 分子构象/对接应用有潜力，但工程实现复杂 |
| 写作质量 | ★★★★☆ | 数学严谨但可读性有挑战，需较强的微分几何基础 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](../../ICML2025/computational_biology/geometric_generative_modeling_with_noise-conditioned_graph_networks.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)

</div>

<!-- RELATED:END -->
