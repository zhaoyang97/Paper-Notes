---
title: >-
  [论文解读] Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling
description: >-
  [ICML 2025][自监督学习][高斯过程] 提出两种基于最近邻的高斯过程先验近似方法（HPA 和 SPA），将近邻驱动的稀疏性引入 GPVAE 的潜空间推断，在保留关键潜变量依赖的同时实现可扩展的 mini-batch 训练，避免了对大量诱导点或受限核函数的依赖。 变分自编码器 (VAE) 在表示学习和生成建模中取得了…
tags:
  - "ICML 2025"
  - "自监督学习"
  - "高斯过程"
  - "变分自编码器"
  - "最近邻近似"
  - "结构化潜变量"
  - "可扩展推断"
---

# Neighbour-Driven Gaussian Process Variational Autoencoders for Scalable Structured Latent Modelling

**会议**: ICML 2025  
**arXiv**: [2505.16481](https://arxiv.org/abs/2505.16481)  
**代码**: 无  
**领域**: 自监督学习  
**关键词**: 高斯过程, 变分自编码器, 最近邻近似, 结构化潜变量, 可扩展推断

## 一句话总结

提出两种基于最近邻的高斯过程先验近似方法（HPA 和 SPA），将近邻驱动的稀疏性引入 GPVAE 的潜空间推断，在保留关键潜变量依赖的同时实现可扩展的 mini-batch 训练，避免了对大量诱导点或受限核函数的依赖。

## 研究背景与动机

变分自编码器 (VAE) 在表示学习和生成建模中取得了巨大成功，但标准 VAE 假设潜变量服从完全分解的高斯先验，无法捕获序列、空间等结构化数据中潜变量之间的相关性。高斯过程变分自编码器 (GPVAE) 通过将 GP 先验替代独立高斯先验来建模潜变量间的结构化依赖，但直接使用 GP 带来 $\mathcal{O}(N^3)$ 的计算瓶颈。

现有的可扩展 GPVAE 方案主要有两类缺陷：

**受限核假设**: 部分方法（如 MGPVAE）仅支持特定的 Matérn 核或低秩核，限制了表达能力。

**诱导点方法**: SVGPVAE 等方法使用少量伪点近似后验，但在数据变化快时需要大量诱导点，且优化诱导点位置本身存在困难。

**采样方法**: 全贝叶斯方法（如 SGPBAE）虽然校准好，但采样耗时长。

本文的核心洞察是：在许多结构化数据集中（如视频帧的时间邻近性、空间区域的局部模式），聚焦于少量最近邻即可捕获大部分核心相关结构。这一思路与地理学第一定律一致，也是 NNGP 方法的理论基础。

## 方法详解

### 整体框架

本文沿用标准 GPVAE 架构：编码器 $q_\phi(\mathbf{Z}|\mathbf{Y})$ 产生潜变量的均值和方差，解码器 $p_\theta(\mathbf{Y}|\mathbf{Z})$ 从潜变量重建观测，GP 先验 $p_\psi(\mathbf{Z}|\mathbf{X})$ 在潜变量上施加结构化依赖。每个潜变量通道 $l$ 使用独立的核函数 $k_\psi^l$，充分利用 GP 的表达能力。

训练目标是最大化 ELBO：

$$\mathcal{L} = \mathbb{E}_{q_\phi(\mathbf{Z}|\mathbf{Y})}[\log p_\theta(\mathbf{Y}|\mathbf{Z})] - \text{KL}[q_\phi(\mathbf{Z}|\mathbf{Y}) \| p_\psi(\mathbf{Z}|\mathbf{X})]$$

核心问题在于 KL 项涉及稠密的 $N \times N$ 协方差矩阵 $\mathbf{K_{XX}}$，无法分解为 mini-batch。本文提出两种近邻驱动的近似来解决这一瓶颈。

### 关键设计

#### 1. 层次先验近似 (HPA - Hierarchical Prior Approximation)

HPA 引入辅助二值随机向量 $\mathbf{w} \in \{0,1\}^N$ 来指示潜变量的选择，通过"关闭"非邻居间的交互来构造稀疏协方差结构：

- **层次先验**: $p(\mathbf{Z}|\mathbf{w}) = \mathcal{N}(\mathbf{Z}|\mathbf{0}, \mathbf{D_w}\mathbf{K_{XX}}\mathbf{D_w})$，其中 $\mathbf{D_w} = \text{diag}(\mathbf{w})$
- **变分分布**: $q(\mathbf{Z}|\mathbf{w}) = \mathcal{N}(\mathbf{Z}|\mathbf{D_w}\mu(\mathbf{Y}), \mathbf{D_w}\sigma^2(\mathbf{Y})\mathbf{D_w})$
- **近邻采样策略**: 对 mini-batch 中的每个点 $\mathbf{x}_i$，在整个数据集 $\mathbf{X}$ 中找 top-$H$ 个最近邻，索引记为 $n(i)$

HPA 的 mini-batch ELBO 为：

$$\mathcal{L}_\text{HPA} \approx \frac{N}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \left\{ \mathbb{E}_{q(\mathbf{z}_i|\mathbf{y}_i)}[\log p(\mathbf{y}_i|\mathbf{z}_i)] - \frac{1}{N}\text{KL}[q(\mathbf{Z}_{n(i)}) \| p(\mathbf{Z}_{n(i)})] \right\}$$

KL 项分解为若干 $H \times H$ 低维协方差矩阵的运算，当 $H=N$ 时恢复原始全批次 ELBO。

#### 2. 稀疏精度近似 (SPA - Sparse Precision Approximation)

SPA 基于 Vecchia 近似，通过概率链式法则将 GP 联合分布分解为条件分布的乘积，并施加条件独立性：

- **精确分解**: $p(\mathbf{Z}) = p(\mathbf{z}_1)\prod_{j=2}^N p(\mathbf{z}_j|\mathbf{z}_{1:j-1})$
- **近邻近似**: $p(\mathbf{Z}) \approx p(\mathbf{z}_1)\prod_{j=2}^N p(\mathbf{z}_j|\mathbf{z}_{n(j)})$

其中 $n(j)$ 是 $\mathbf{x}_j$ 在前序点 $\{\mathbf{x}_h\}_{h=1}^{j-1}$ 中的 $H$ 个最近邻。这等价于对先验精度矩阵 $\mathbf{K_{XX}}^{-1}$ 进行稀疏 Cholesky 分解。

SPA 的 mini-batch ELBO 为：

$$\mathcal{L}_\text{SPA} \approx \frac{N}{|\mathcal{I}|}\sum_{i \in \mathcal{I}} \mathbb{E}_{q(\mathbf{z}_i|\mathbf{y}_i)}[\log p(\mathbf{y}_i|\mathbf{z}_i)] - \frac{N}{|\mathcal{J}|}\sum_{j \in \mathcal{J}} \mathbb{E}_{q(\mathbf{Z}_{n(j)})}\text{KL}[q(\mathbf{z}_j) \| p(\mathbf{z}_j|\mathbf{Z}_{n(j)})]$$

当 $H=N$ 恢复完整 ELBO，$H=0$ 退化为标准 VAE。

#### 3. HPA 与 SPA 的互补关系

| 特性 | HPA | SPA |
|------|-----|-----|
| 稀疏化目标 | 协方差矩阵 | 精度矩阵 |
| 稀疏机制 | 通过层次选择变量关闭非邻居交互 | 将联合分布链式分解为近邻条件分布 |
| 邻居选取范围 | 全局 $\mathbf{X}$ 中 top-$H$ | 前序点中 top-$H$ |
| 理论根源 | 层次 NNGP (Tran et al., 2021) | Vecchia 近似 (Vecchia, 1988) |
| 退化条件 | $H=N$ → 全批次 ELBO | $H=0$ → 标准 VAE |

#### 4. 预测后验

对新输入 $\mathbf{x}_*$，预测仅需考虑其在 $\mathbf{X}$ 中的 $H$ 个最近邻：

$$q(\mathbf{z}_*|\mathbf{Y}) = \int p(\mathbf{z}_*|\mathbf{Z}_{n(*)}) q(\mathbf{Z}_{n(*)}|\mathbf{Y}_{n(*)}) d\mathbf{Z}_{n(*)}$$

预测后验是高斯分布，可高效采样进行蒙特卡洛估计。

### 损失函数 / 训练策略

- **训练目标**: 最大化近邻近似的 ELBO（HPA 或 SPA 形式）
- **参数联合优化**: 编码器 $\phi$、解码器 $\theta$、核参数 $\psi$ 通过 mini-batch SGD 联合学习
- **核函数灵活性**: 支持任意核函数（RBF、Matérn 等），无需受限于特定核假设
- **最近邻预计算**: 使用 Faiss 在 GPU 上加速最近邻搜索
- **计算复杂度**: 最近邻搜索 $\mathcal{O}(HN)$，KL 项的 Cholesky 分解 $\mathcal{O}(LN_bH^3)$，其中 $N_b$ 为 batch size，$L$ 为潜空间维度，$H$ 为近邻数

## 实验关键数据

### 主实验

论文在三类任务上进行了实验：表示学习、数据插补、条件生成。

| 数据集 | 任务 | 指标 | GPVAE-HPA/SPA | SVGPVAE (诱导点) | 提升 |
|--------|------|------|---------------|-----------------|------|
| Moving Ball | 表示学习 | RMSE | 最优（H=10） | 需要更多诱导点 | 更低重建误差 |
| 时序数据 | 数据插补 | RMSE / NLL | 优于其他 GPVAE | 中等 | 预测精度+训练速度 |
| 空间数据 | 条件生成 | Log-likelihood | 竞争性表现 | 受诱导点数量限制 | 灵活核+更快收敛 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| $H=0$ (无近邻) | 退化为标准 VAE | SPA 退化验证，无法捕获结构化依赖 |
| $H=10$ | 接近最优 | 少量近邻即可捕获核心相关结构 |
| $H=N$ (全部) | 恢复全批次 ELBO | 计算代价 $\mathcal{O}(N^3)$，无法扩展 |
| HPA vs SPA | 性能相近 | HPA 稀疏协方差，SPA 稀疏精度，互补 |
| RBF vs Matérn 核 | 均支持 | 核函数选择灵活，不受限于特定核 |

### 关键发现

1. **少量近邻即高效**: $H=10$ 左右即可达到接近全批次 GP 的性能，体现了数据的局部相关性原理
2. **优于诱导点方法**: 在相同精度下，本文方法比 SVGPVAE 所需的等效参数更少，训练更快
3. **核函数灵活**: 不再受限于低秩核或 Matérn 核，可自由选择 RBF、周期核等
4. **可扩展性强**: 复杂度从 $\mathcal{O}(N^3)$ 降至 $\mathcal{O}(LN_bH^3)$，适用于大规模数据集

## 亮点与洞察

1. **巧妙的问题转化**: 将 NNGP 从观测空间迁移到 VAE 的潜空间，将每个数据点视为自己的"诱导变量"，用近邻关系替代全局依赖
2. **两种互补的稀疏化策略**: HPA 从协方差角度、SPA 从精度矩阵角度分别提供稀疏近似，提供了多元选择
3. **地理学第一定律的深刻应用**: "近的事物更相关" 这一直觉在潜空间中同样成立，为 GP 的局部近似提供了合理的理论支撑
4. **工程实用性**: 使用 Faiss 加速近邻搜索、支持任意核函数、标准编码器-解码器架构，降低了使用门槛

## 局限与展望

1. **近邻数 $H$ 需要调参**: 虽然实验显示 $H=10$ 较通用，但最优 $H$ 可能随数据集变化，缺乏自适应选择机制
2. **辅助信息依赖**: 需要明确的辅助输入 $\mathbf{X}$（如时间戳、坐标）来定义近邻关系，对缺乏自然序的数据适用性有限
3. **前序排序敏感性**: SPA 依赖数据的排序（链式分解的顺序），不同排序可能影响近似质量
4. **潜在扩展方向**:
    - 自适应近邻数选择（如根据核的 lengthscale 动态调整 $H$）
    - 结合诱导点和近邻方法的混合策略
    - 扩展到非欧几里得空间（如图结构数据）的近邻定义

## 相关工作与启发

- **GPVAE 系列**: Casale et al. (2018) 首先提出 GPVAE 但受限于低秩核；Fortuin et al. (2020) 用于时间序列插补但仅适用于短序列
- **SVGPVAE** (Jazbec et al., 2021): 基于诱导点的可扩展方案，诱导点优化困难是主要瓶颈
- **MGPVAE** (Zhu et al., 2023): 利用 Matérn 核的状态空间表示实现 Kalman 滤波推断，但核选择受限
- **NNGP** (Datta et al., 2016; Wu et al., 2022): 在地统计学中广泛使用，证明在大规模任务中可优于标准诱导点方法
- **启发**: 局部性原理是 GP 可扩展化的核心，将其迁移到深度生成模型的潜空间是一个值得进一步挖掘的方向

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 将 NNGP 思想引入 GPVAE 潜空间是新颖的迁移，两种互补近似设计精巧 |
| 理论性 | 4 | 推导严谨，HPA/SPA 均有清晰的退化和恢复条件 |
| 实用性 | 4 | 支持任意核、Faiss 加速、标准架构，工程友好 |
| 写作质量 | 4 | 逻辑清晰，符号统一，动机阐述充分 |
| 综合评分 | 4 | 在 GPVAE 可扩展性这一重要问题上提供了实用且优雅的解决方案 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] AdaWorld: Learning Adaptable World Models with Latent Actions](adaworld_learning_adaptable_world_models_with_latent_actions.md)
- [\[CVPR 2026\] An Optimal Transport-driven Approach for Cultivating Latent Space in Online Incremental Learning](../../CVPR2026/self_supervised/an_optimal_transport_driven_approach_for_cultivating_latent_space_in_online_incr.md)
- [\[CVPR 2025\] ScaleLSD: Scalable Deep Line Segment Detection Streamlined](../../CVPR2025/self_supervised/scalelsd_scalable_deep_line_segment_detection_streamlined.md)
- [\[ICLR 2026\] InfoNCE Induces Gaussian Distribution](../../ICLR2026/self_supervised/infonce_induces_gaussian_distribution.md)
- [\[NeurIPS 2025\] Curiosity-driven RL for Symbolic Equation Solving](../../NeurIPS2025/self_supervised/curiosity-driven_rl_for_symbolic_equation_solving.md)

</div>

<!-- RELATED:END -->
