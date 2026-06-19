---
title: >-
  [论文解读] Geometric Generative Modeling with Noise-Conditioned Graph Networks
description: >-
  [ICML2025][计算生物][Noise-Conditioned Graph Networks] 提出 Noise-Conditioned Graph Networks (NCGNs)，使 GNN 架构根据噪声级别动态调整消息传递的范围和图分辨率：高噪声时用远程连接+低分辨率，低噪声时用局部连接+高分辨率，在 3D 点云、空间转录组和图像生成中均超越固定架构基线。
tags:
  - "ICML2025"
  - "计算生物"
  - "Noise-Conditioned Graph Networks"
  - "扩散模型"
  - "Flow Matching"
  - "动态消息传递"
  - "图粗粒化"
  - "3D 点云生成"
---

# Geometric Generative Modeling with Noise-Conditioned Graph Networks

**会议**: ICML2025  
**arXiv**: [2507.09391](https://arxiv.org/abs/2507.09391)  
**代码**: [GitHub](https://github.com/peterpaohuang/ncgn)  
**领域**: 计算生物  
**关键词**: Noise-Conditioned Graph Networks, 扩散模型, Flow Matching, 动态消息传递, 图粗粒化, 3D 点云生成

## 一句话总结

提出 Noise-Conditioned Graph Networks (NCGNs)，使 GNN 架构根据噪声级别动态调整消息传递的范围和图分辨率：高噪声时用远程连接+低分辨率，低噪声时用局部连接+高分辨率，在 3D 点云、空间转录组和图像生成中均超越固定架构基线。

## 研究背景与动机

### 领域现状

**领域现状**：几何图生成的重要性**：3D 点云、分子结构、空间基因组学等都涉及带空间信息的图结构生成。

### 现有痛点

**现有痛点**：现有 GNN 的静态局限**：流式生成模型中的 GNN 在整个去噪过程中使用固定的 kNN 或半径图，忽略了噪声级别对图信号的影响。

### 核心矛盾

**核心矛盾**：信息论动机**：理论分析表明，噪声增大时恢复信号需要更远的邻居信息，且图可以用更低分辨率表示。

## 方法详解

### 理论基础

- **Lemma 3.1 (互信息公式)**：给出了节点原始特征 $x_1^{(i)}$ 与半径 $r$ 内噪声特征聚合 $Y_t^{(i,r)}$ 之间互信息的解析表达式。
- **Theorem 3.2 (范围递增)**：当 SNR 下降时，存在更大半径 $r_2 > r_1$ 使得 $I(x_1^{(i)}, Y_{c_2}^{(r_2)}) > I(x_1^{(i)}, Y_{c_2}^{(r_1)})$。
- **Proposition 3.3 (位置噪声)**：噪声增大时，原本相近节点的期望距离增大，因此需要更大的消息传递半径。

### Dynamic Message Passing (DMP)

给定边界条件 $(r_0, s_0)$ 和 $(r_1, s_1)$ 以及自适应调度函数 $f$：

$$r_t, s_t = f(t, r_0, r_1, s_0, s_1)$$

约束条件：
- **单调性**：$t' < t$（更高噪声）$\Rightarrow r' \geq r, s' \leq s$
- **边界一致**：$f(0) = (r_0, s_0)$，$f(1) = (r_1, s_1)$

每步操作：
1. **图粗粒化**：将 $N$ 个节点聚合为 $s_t$ 个超节点（体素聚类/均值池化）。
2. **连接构建**：按 $r_t$ 构建 kNN/半径图。
3. **消息传递**：在粗粒化图上做 GCN/GAT。
4. **反粗粒化**：将信息映射回原始节点。

### 复杂度

当 $r_t \cdot s_t = r_1 N$ 时，消息传递在整个生成过程中保持线性时间复杂度。

## 实验关键数据

### 主实验 1：3D 点云生成 (ModelNet40)

| 方法 | GCN $\mathcal{W}_2$ (×10⁻²) | GAT $\mathcal{W}_2$ (×10⁻²) |
|---|---|---|
| Random | 8.624 | 8.624 |
| KNN | 5.882 | 5.598 |
| Long-Short Range | 4.315 | 4.741 |
| **DMP** | **4.215** | **4.263** |

平均提升 16.15%。

### 主实验 2：空间转录组学

- DMP 在 GCN 上全面优于 kNN 和全连接基线，GAT 上可比。

### 主实验 3：图像生成 (DiT-DMP)

| | DiT | DiT-DMP |
|---|---|---|
| FID↓ | 84.051 | **63.983** |
| IS↑ | 16.735 | **24.681** |
| Precision↑ | 0.296 | **0.446** |

仅修改两行代码（FlexiViT 动态 patch + 邻域注意力）即可显著提升。

### 消融：调度函数选择

- 指数调度最佳 ($\mathcal{W}_2 = 4.263 \times 10^{-2}$)，线性和 ReLU 也优于基线，对数调度不及基线。

## 亮点与洞察

1. **信息论视角的理论支撑**：SNR 下降→最优半径增大→图分辨率可降低，提供了动态架构的理论依据。
2. **简单强大的实现**：DMP 可以通过最小代码修改集成到现有模型（如 DiT）。
3. **线性复杂度**：通过粗粒化平衡连接范围的增大，避免了全连接的二次复杂度。
4. **注意力权重的实证**：训练后的 GAT 注意力分布确实随噪声级别变化，验证了理论预测。

## 局限与展望

- 调度函数 $f$ 是预定义的，未学习最优调度。
- 仅调整连接范围和分辨率，未探索其他可调整维度（层数、宽度、消息传递类型）。
- 粗粒化策略（体素聚类）可能不是所有场景的最优选择。
- Theorem 3.2 的理论假设（相关结构）在实践中可能不严格成立。

## 相关工作与启发

- **Flow Matching (Lipman et al., 2022; Tong et al., 2023)**：DMP 在此框架内作为模块化插件。
- **DiT (Peebles & Xie, 2023)**：图像生成 SOTA，本文的 DiT-DMP 展示了与现有模型的快速集成。
- **Torsional Diffusion (Jing et al., 2022)**：使用固定半径图的代表，本文的改进目标。
- **启发**：动态架构的思想可推广到分子生成、图像超分辨率等更多流式生成任务。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](graph_generative_pre-trained_transformer.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](../../NeurIPS2025/computational_biology/generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](../../NeurIPS2025/computational_biology/towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[NeurIPS 2025\] Random Search Neural Networks for Efficient and Expressive Graph Learning](../../NeurIPS2025/computational_biology/random_search_neural_networks_for_efficient_and_expressive_graph_learning.md)

</div>

<!-- RELATED:END -->
