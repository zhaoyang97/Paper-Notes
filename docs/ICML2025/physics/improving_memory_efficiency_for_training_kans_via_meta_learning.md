---
title: >-
  [论文解读] Improving Memory Efficiency for Training KANs via Meta Learning
description: >-
  [ICML2025][物理/科学计算][Kolmogorov-Arnold Networks] 提出 MetaKANs，用一个小型元学习器（meta-learner）生成 KAN 中所有可学习激活函数的参数，将可训练参数量从 KAN 的 $(G+k+1)$ 倍压缩到接近 MLP 水平（约 1/3 到 1/9），同时保持甚至提升性能。
tags:
  - "ICML2025"
  - "物理/科学计算"
  - "Kolmogorov-Arnold Networks"
  - "Meta Learning"
  - "parameter efficiency"
  - "HyperNetwork"
  - "可学习激活函数"
---

# Improving Memory Efficiency for Training KANs via Meta Learning

**会议**: ICML2025  
**arXiv**: [2506.07549](https://arxiv.org/abs/2506.07549)  
**代码**: [GitHub](https://github.com/Murphyzc/MetaKAN)  
**领域**: KAN训练 / 元学习 / 内存效率  
**关键词**: Kolmogorov-Arnold Networks, Meta Learning, parameter efficiency, HyperNetwork, 可学习激活函数

## 一句话总结

提出 MetaKANs，用一个小型元学习器（meta-learner）生成 KAN 中所有可学习激活函数的参数，将可训练参数量从 KAN 的 $(G+k+1)$ 倍压缩到接近 MLP 水平（约 1/3 到 1/9），同时保持甚至提升性能。

## 研究背景与动机

**KAN 的参数膨胀问题**：KAN（Kolmogorov-Arnold Networks）用可学习的 B-spline 参数化的单变量函数替代传统 MLP 中的固定激活函数，带来了更好的可解释性和函数逼近能力。然而，每个激活函数都需要 $G+k+1$ 个可训练系数（$G$ 为网格点数，$k$ 为 spline 阶数），导致同结构下 KAN 的参数量是 MLP 的 $(G+k+1)$ 倍。例如 $G=5, k=3$ 时参数量就是 MLP 的 9 倍，严重限制了 KAN 在大规模任务上的可扩展性。

**现有 KAN 变体的局限**：ChebyshevKAN、WavKAN、FastKAN 等变体虽然改进了基函数的选择和计算效率，但可学习激活函数带来的参数膨胀问题依然存在。

**核心观察**：KAN 中所有激活函数属于同一个函数族 $\mathcal{F}$，它们共享相同的参数生成规则。学习这些参数可以看作是一个多任务学习问题——每个激活函数是一个"任务"，所有任务的权重生成服从一个共同规则。这为用一个小网络统一生成权重提供了理论动机。

## 方法详解

### KAN 的参数结构回顾

标准 KAN 中每个激活函数参数化为：

$$\phi(t; \mathbf{w}) = w_b \cdot \text{SiLU}(t) + \sum_{i=1}^{G+k} c_i B_i(t) = \mathbf{w}^\top \mathbf{B}(t)$$

其中 $\mathbf{w} = [w_b, c_1, \ldots, c_{G+k}]^\top \in \mathbb{R}^{G+k+1}$。

对于结构为 $[n_0, n_1, \ldots, n_L]$ 的 KAN，总参数量为：

$$|\mathcal{W}| = \sum_{l=0}^{L-1}(n_l \times n_{l+1}) \times (G+k+1)$$

### MetaKAN 框架

**核心思想**：用一个小型 MLP（meta-learner）$M_\theta$ 从可学习的 prompt $z \in \mathbb{R}$ 生成激活函数的权重：

$$M_\theta: \mathbb{R} \to \mathbb{R}^{G+k+1}, \quad z \mapsto \mathbf{w}$$

每个激活函数被分配一个标量 prompt $z_\alpha^{(l)}$ 作为唯一标识符，meta-learner（两层 MLP，隐藏维度 $d_{\text{hidden}}$）学习从 prompt 到权重的映射规则。激活函数变为：

$$\phi(t; z, \theta) = M_\theta(z)^\top \mathbf{B}(t)$$

**参数量对比**：

| 模型 | 参数量 |
|------|--------|
| MLP | $\sum_{l}(n_l \times n_{l+1})$ |
| KAN | $\sum_{l}(n_l \times n_{l+1}) \times (G+k+1)$ |
| MetaKAN | $\sum_{l}(n_l \times n_{l+1}) + C \times (d_{\text{hidden}}+1) \times (G+k+1)$ |

其中 $C$ 是 meta-learner 的数量（浅层网络 $C=1$）。当网络足够大时，MetaKAN 的参数量趋近于 MLP。

### 深层 KAN 的扩展：层聚类策略

不同层的权重生成规则存在差异（输入分布不同），单一 meta-learner 难以应对。解决方案：

1. 用 K-Means 根据各层输出通道数将 $L$ 层聚类为 $C$ 个簇 $\{L_1, \ldots, L_C\}$
2. 每个簇分配独立的 meta-learner $M_{\theta_{(c)}}$
3. 同簇内的层共享一个 meta-learner

$$\mathbf{w}_\alpha^{(l)} = M_{\theta_{(c)}}(z_\alpha^{(l)}), \quad \forall l \in L_c$$

### 训练目标

端到端优化 prompt $\mathcal{Z}$ 和 meta-learner 参数 $\Theta$：

$$\mathcal{Z}^\star, \Theta^\star = \arg\min_{\mathcal{Z}, \Theta} \mathbb{E}_{\mathbf{x}} \left[ \ell(\text{MetaKAN}(\mathbf{x}; \mathcal{Z}, \Theta), f(\mathbf{x})) \right]$$

回归任务用 MSE，分类任务用交叉熵。

### 模型无关性

MetaKAN 框架可直接应用于各种 KAN 变体，已验证：MetaFastKAN（基于 RBF）、MetaWavKAN（基于小波）、MetaConvKAN（卷积版）、MetaKALNConv、MetaKAGNConv。参数压缩比约为 $1/\text{dim}(\mathbf{w})$。

## 实验关键数据

### 符号回归（Feynman 数据集，G=5）

| 函数 | KAN MSE | KAN #Param | MetaKAN MSE | MetaKAN #Param |
|------|---------|------------|-------------|----------------|
| I.6.20a | 5.94e-4 | 58 | **3.88e-4** | 218 |
| I.8.4 | 1.30e-2 | 979 | **3.76e-3** | 461 |
| I.9.18 | 2.39e-3 | 781 | **1.70e-3** | 993 |
| I.12.5 | 1.32e-3 | 57 | **1.16e-4** | 30 |
| I.15.3x | 1.57e-2 | 223 | **5.38e-3** | 80 |

在 G=5 的设定下，MetaKAN 在 18 个函数中有 16 个超越 KAN。

### 图像分类（ConvKAN，4层）

| 模型 | MNIST Acc | #Param | CIFAR-10 Acc | #Param |
|------|-----------|--------|--------------|--------|
| KANConv | 98.43 | 3.49M | 41.92 | 3.49M |
| MetaKANConv | 96.03 | **391K** | **45.97** | **393K** |
| FastKANConv | 99.36 | 3.49M | 68.12 | 3.49M |
| MetaFastKANConv | 98.54 | **392K** | 66.69 | **392K** |
| KAGNConv | 99.15 | 1.94M | 72.08 | 1.94M |
| MetaKAGNConv | **99.21** | **391K** | — | — |

MetaKAGNConv 在 MNIST 上以约 1/5 参数量反超 KAGNConv。MetaKANConv 在 CIFAR-10 上以约 1/9 参数量超越 KANConv（45.97 vs 41.92）。

### 参数效率总结

- 参数压缩比：约 **1/3 到 1/9**（取决于 $G+k+1$ 和网络规模）
- 在大多数 benchmark 上，MetaKAN 以更少参数达到相当或更优的性能
- 在 PDE 求解任务上同样有效

## 亮点与洞察

1. **理论动机清晰**：从 KAN 的工作机制出发，将权重学习理解为多任务学习问题，meta-learner 学习共享的权重生成规则，比暴力优化每个激活函数参数更高效
2. **模型无关**：框架可无缝应用于 KAN/FastKAN/WavKAN/ConvKAN/KALN/KAGN 等变体，通用性强
3. **参数接近 MLP**：当网络规模足够大时，MetaKAN 参数量趋近 MLP——这从根本上解决了 KAN 原论文中提到的与 MLP 之间的训练成本差距
4. **Prompt 设计巧妙**：每个激活函数用一个标量 prompt 标识，简单有效，类似 LLM 中的 in-context learning 思想
5. **层聚类策略**：避免了为每层分配独立 meta-learner 的参数开销，用 K-Means 聚类平衡表达能力和效率

## 局限与展望

1. **小网络可能不省参数**：当 KAN 很小（总激活函数数量少）时，meta-learner 的固定开销可能导致 MetaKAN 参数量反而更多（如 Table 2 中部分小结构）
2. **推理时无速度提升**：MetaKAN 仍需在前向传播时通过 meta-learner 生成权重再计算，推理延迟可能不降反升
3. **meta-learner 结构固定**：仅考虑了两层 MLP 作为meta-learner，未探索更复杂或自适应的架构
4. **prompt 维度为标量**：每个激活函数仅用 1 维 prompt 标识，对于非常深的网络可能表达能力不足（虽然消融实验显示增大维度效果有限）
5. **大规模视觉任务验证不足**：图像实验仅在 MNIST/CIFAR 级别验证，未在 ImageNet 等大规模数据集上测试

## 相关工作与启发

- **HyperNetworks**（Ha et al., 2017）：用辅助网络生成主网络权重的先驱工作，但采用启发式策略；MetaKAN 基于 KAN 的工作机制设计，更有针对性
- **KAN 原论文**（Liu et al., 2024）：明确指出 KAN 训练成本高的问题，MetaKAN 从参数效率角度提供了解决方案
- **Meta-learning 范式**：将每个激活函数的权重学习视为一个"任务"，共享 meta-learner 学习跨任务的权重生成规则

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将 KAN 权重学习理解为多任务学习并用 meta-learner 生成权重的视角新颖
- 实验充分度: ⭐⭐⭐⭐ — 符号回归/PDE/分类多任务验证，多种 KAN 变体覆盖，消融充分
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，参数分析透彻
- 价值: ⭐⭐⭐⭐ — 实用性强，有效降低 KAN 训练门槛，促进 KAN 在更大规模问题上的应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Neural Emulator Superiority: When Machine Learning for PDEs Surpasses its Training Data](../../NeurIPS2025/physics/neural_emulator_superiority_when_machine_learning_for_pdes_surpasses_its_trainin.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](../../NeurIPS2025/physics/enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)
- [\[ICML 2025\] PAC Learning with Improvements](pac_learning_with_improvements.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/physics/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/physics/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)

</div>

<!-- RELATED:END -->
