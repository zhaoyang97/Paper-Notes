---
title: >-
  [论文解读] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings
description: >-
  [NeurIPS 2025][计算生物][蛋白质构象生成] 提出 LD-FPG：框架，使用 Chebyshev 图神经网络将蛋白质全原子 MD 轨迹编码到低维潜在空间，再用 DDPM 在该空间中生成新的构象集合体（ensemble），首次实现了包含侧链所有重原子的蛋白质构象生成。 领域现状：蛋白质功能依赖于不同构象状态之间的…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "蛋白质构象生成"
  - "潜在扩散"
  - "图神经网络"
  - "全原子建模"
  - "GPCR"
---

# Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings

**会议**: NeurIPS 2025  
**arXiv**: [2506.17064](https://arxiv.org/abs/2506.17064)  
**代码**: 有（开源）  
**领域**: 计算生物  
**关键词**: 蛋白质构象生成, 潜在扩散, 图神经网络, 全原子建模, GPCR  
**arXiv**: [2506.17064](https://arxiv.org/abs/2506.17064)  
**代码**: 无  
**领域**: 医学图像  

## 一句话总结

提出 **LD-FPG** 框架，使用 Chebyshev 图神经网络将蛋白质全原子 MD 轨迹编码到低维潜在空间，再用 DDPM 在该空间中生成新的构象集合体（ensemble），首次实现了包含侧链所有重原子的蛋白质构象生成。

## 研究背景与动机

**领域现状**：蛋白质功能依赖于不同构象状态之间的动态转换。AlphaFold2等方法主要预测单一静态构象，无法捕捉功能性构象多样性。

**现有痛点**：现有生成模型要么只生成骨架（无侧链），要么只能产生粗粒度表示，要么局限于从头设计而非特定蛋白质的构象采样。关键的侧链重排往往决定了分子识别和催化机制。

**核心矛盾**：生成全原子（包括每个侧链重原子）的构象集合的需求 vs 已有方法的能力缺口——尤其是对于 GPCR 这类在膜环境中具有复杂动力学的蛋白质。

**本文目标**：从已有的分子动力学（MD）模拟数据中学习并生成特定蛋白质（如多巴胺 D2 受体）的高保真全原子构象集合。

**切入角度**：不模拟新的MD轨迹，而是学习MD数据的潜在表示——将构象建模为相对于参考结构的变形（deformation）。

**核心 idea**：用 ChebNet 编码、池化压缩、DDPM采样、条件解码的四阶段流水线，在紧凑的潜在空间中生成全原子构象。

## 方法详解

### 整体框架（图1）

1. **ChebNet编码**：将MD帧的全原子坐标编码为逐原子潜在嵌入 $Z^{(t)} \in \mathbb{R}^{N \times d_z}$
2. **池化压缩**：将高维 $Z^{(t)}$ 池化为紧凑的潜在向量 $\mathbf{h}_0$（约60-1100维）
3. **DDPM生成**：在池化后的潜在空间中训练DDPM，生成 $\mathbf{h}_0^{\text{gen}}$
4. **条件解码**：以参考结构的潜在表示 $Z_{\text{ref}}$ 为条件，从 $\mathbf{h}_0^{\text{gen}}$ 解码回全原子坐标

### 关键设计 1：ChebNet 多跳编码

- **功能**：将每帧的Kabsch对齐后的原子坐标映射到潜在嵌入。
- **核心思路**：使用4层 Chebyshev 图卷积（$K=4$ 阶多项式）：
$$H^{(l+1)} = \sigma\left(\sum_{k=0}^{K-1} \Theta_k^{(l)} T_k(\tilde{L}) H^{(l)}\right)$$
  构建 $k$-NN 图（$k=4$），每层后接 BatchNorm，最终输出 $L_2$ 归一化。
- **设计动机**：光谱图卷积能捕获多跳原子间关系，不依赖全局注意力即可编码局部几何。
- **条件机制**：使用冻结的预训练编码器生成参考结构嵌入 $C = Z_{\text{ref}}$，比直接以原始坐标为条件效果更好。

### 关键设计 2：三种池化策略

| 策略 | 描述 | $d_z$ | 潜在维度 |
|---|---|---|---|
| **Blind pooling** | 全局自适应平均池化所有N个原子 | 16 | ~100 |
| **Sequential pooling** | 先解码骨架，再以骨架信息为条件解码侧链 | 8 | ~100 |
| **Residue pooling** | 按残基分别池化，每个残基独立描述 | 4 | $N_{\text{res}} \times d_p \approx 1100$ |

- **设计动机**：高维 $Z^{(t)}$（D2R: 最高35K维）直接输入DDPM不可行，必须压缩。$d_p > 200-300$ 妨碍扩散训练，$d_p < 50$ 损失重构质量。

### 关键设计 3：扩散与损失函数

- **DDPM损失**：$\mathcal{L}_{\text{diffusion}}(\theta) = \mathbb{E}_{t,\mathbf{h}_0,\epsilon}[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}\mathbf{h}_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2]$
- **解码器损失**：Blind和Residue用 $\mathcal{L}_{\text{coord}}$（坐标MSE），Sequential分阶段用 $\mathcal{L}_{\text{BB}}$ 和 $\mathcal{L}_{\text{SC}}$
- **可选的二面角微调**：$\mathcal{L}_{\text{Dec}} = w_{\text{base}}\mathcal{L}_{\text{coord}} + \lambda_{\text{mse}}\mathcal{L}_{\text{mse\_dih}} + \lambda_{\text{div}}\mathcal{L}_{\text{div\_dih}}$

## 实验关键数据

### 解码器重构性能（Table 1）

| Decoder | lDDT$_{\text{All}}$ ↑ | lDDT$_{\text{BB}}$ ↑ | $\sum$JSD$_{\text{bb}}$ ↓ | $\sum$JSD$_{\text{sc}}$ ↓ | MSE$_{\text{sc}}$ ↓ |
|---|---|---|---|---|---|
| Blind (dz=16) | 0.714 | 0.792 | 0.0032 | 0.0290 | 0.3934 |
| Sequential (dz=8) | **0.718** | **0.800** | **0.0026** | 0.0192 | 0.5130 |
| Residue (dz=4) | 0.704 | 0.777 | 0.0078 | **0.0125** | **0.2257** |
| Ground Truth (MD) Ref | 0.698 | 0.779 | - | - | - |

### 扩散生成性能（Table 2）

| Model | lDDT$_{\text{All}}$ ↑ | $\sum$JSD$_{\text{bb}}$ ↓ | $\sum$JSD$_{\text{sc}}$ ↓ | Avg. Clashes ↓ |
|---|---|---|---|---|
| Blind pooling | **0.719** | 0.006582 | 0.04185 | 1350.5 |
| Sequential pooling | 0.712 | **0.0029** | 0.02895 | 1220.5 |
| Residue pooling | 0.688 | 0.0117 | **0.0224** | **1145.6** |
| MD reference | ~0.698 | - | - | ~1023 |

### 消融实验

- **ChebNet 编码保真度**：dz=16时重构 MSE$_{\text{bb}}$=0.0008，JSD~0.00016，建立了保真度上界
- **二面角微调**：对Blind策略仅微弱改善JSD，同时略微降低lDDT
- **BioEmu对比**：通用MD模型BioEmu生成的A100分布（mean=-17.19）与D2R-MD参考（mean≈-47.5）严重偏离

### 关键发现

- **三种池化策略各有所长**：Blind优于全局保真度，Sequential在骨架几何上最佳，Residue在侧链旋转异构体和碰撞数量上最优
- **Residue pooling** 虽然全局骨架指标稍逊，但在A100构象景观覆盖上最完整（多epoch采样后），这得益于其更大的有效潜在空间（~1.1K维）
- 生成结构的碰撞数（1145-1350）仍高于MD参考（~1023），这是当前主要局限

## 亮点与洞察

1. **首次全原子构象生成**：据作者所知，这是第一个专门为全原子蛋白质构象集合生成设计的潜在扩散框架
2. **参考结构变形建模**：将生成重定义为相对于参考结构的变形学习，大大简化了生成任务
3. **Residue pooling 的设计直觉**：按残基分池的策略让每个残基有独立的变形描述符，这与蛋白质化学的基本单位（氨基酸残基）自然对应
4. **对比 BioEmu 的分析有说服力**：通用模型在特定膜蛋白功能态采样上严重不足，凸显了系统特异性方法的必要性

## 局限与展望

1. **碰撞问题**：生成结构的原子碰撞数显著高于MD参考，需要引入轻量级能量代理或物理约束
2. **Residue pooling 依赖多epoch采样**：需要聚合不同DDPM训练阶段的样本才能获得完整的构象多样性
3. **单一系统验证**：仅在D2R（一种GPCR）上验证，泛化到其他蛋白质体系需要进一步工作
4. **无等变性**：ChebNet不天然保证SE(3)等变性，虽然Kabsch对齐缓解了这一问题
5. **池化信息损失**：从35K维到~100维的激进压缩不可避免地损失了细节，更大的 $d_p$ 值需要更多训练数据

## 相关工作与启发

- **与 AlphaFlow/ESMFlow 的区别**：后者通过扰动静态预测来采样，LD-FPG直接从MD数据学习系统特异性的构象分布
- **与 LatentDiff 的互补**：LatentDiff生成新的蛋白质骨架折叠，LD-FPG生成已知蛋白的构象集合
- **对药物发现的意义**：GPCR是约50%上市药物的靶点，准确的构象景观建模可加速变构位点发现和药物设计

## 评分

⭐⭐⭐⭐ (4/5)

填补了全原子蛋白质构象生成的重要空白。框架设计合理，三种池化策略的系统比较提供了有价值的设计指导。主要不足是碰撞问题、单系统验证和多epoch采样需求。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Diffusion Generative Modeling on Lie Group Representations](diffusion_generative_modeling_on_lie_group_representations.md)
- [\[NeurIPS 2025\] Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning](generative_distribution_embeddings_lifting_autoencoders_to_the_space_of_distribu.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](../../ICML2025/computational_biology/geometric_generative_modeling_with_noise-conditioned_graph_networks.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)

</div>

<!-- RELATED:END -->
