---
title: >-
  [论文解读] Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning
description: >-
  [NeurIPS 2025][计算生物][分布嵌入] 提出生成分布嵌入（GDE），将自编码器提升到分布空间——编码器作用于样本集合，解码器替换为条件生成模型，学习分布级别的表示，并在6个计算生物学任务上验证有效性。 领域现状：现代科学（特别是计算生物学）越来越需要跨尺度推理——分析单位不是单个数据点（如一个细胞）…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "分布嵌入"
  - "自编码器"
  - "Wasserstein空间"
  - "多尺度表示"
  - "计算生物学"
---

# Generative Distribution Embeddings: Lifting Autoencoders to the Space of Distributions for Multiscale Representation Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.18150](https://arxiv.org/abs/2505.18150)  
**代码**: 有 (GitHub)  
**领域**: 计算生物  
**关键词**: 分布嵌入, 自编码器, Wasserstein空间, 多尺度表示, 计算生物学

## 一句话总结

提出生成分布嵌入（GDE），将自编码器提升到分布空间——编码器作用于样本集合，解码器替换为条件生成模型，学习分布级别的表示，并在6个计算生物学任务上验证有效性。

## 研究背景与动机

**领域现状**：现代科学（特别是计算生物学）越来越需要跨尺度推理——分析单位不是单个数据点（如一个细胞），而是数据点所属的分布（如患者的所有细胞数据）。核方法、Wasserstein 空间方法、变分自编码器等各有局限。

**现有痛点**：
   - 传统编码器处理单个数据点，丢失了**群体级别**的信号
   - 核均值嵌入（KME）非参数但不生成
   - Wasserstein Wormhole 等方法限制于固定数量点的采样
   - 现有方法缺乏从分布嵌入反向生成（解码回分布）的能力

**核心矛盾**：层次化数据（患者→细胞→基因表达）中，单元级别的噪声很大（如分子欠采样），但需要的是分布级别的信号。

**本文目标**：构建一个通用框架，能学习分布的压缩表示，并能从表示中重新采样该分布。

**切入角度**：将自编码器概念提升——编码器接收样本集合（经验分布），解码器是条件生成模型（给定嵌入向量采样）。关键约束是编码器必须具有"分布不变性"。

**核心 idea**：通过分布不变编码器 + 条件生成模型的组合，将任何自编码器框架提升到分布空间，学习等价于预测充分统计量的表示。

## 方法详解

### 整体框架

GDE 由两部分组成：
- **编码器 $\mathcal{E}$**：将样本集合 $S_{i,m} = \{x_{ij}\}_{j=1}^m$ 映射到潜在表示 $z_i$
- **条件生成器 $\mathcal{G}$**：给定 $z_i$ 生成新样本，使得 $\mathcal{G}(\mathcal{E}(S_{i,m})) \xrightarrow{m \to \infty} P_i$

**训练算法**：对每个集合 $S_{i,m_i}$，子采样 $\tilde{S}_{i,m}$，计算 $z_i = \mathcal{E}(\tilde{S}_{i,m})$，用生成器损失 $\ell(\tilde{S}_{i,m}, \mathcal{G}(z_i))$ 反向传播。

### 关键设计

#### 1. 分布不变性

编码器必须满足两个条件：
- **置换不变性**：样本顺序不影响嵌入
- **比例不变性**：将每个样本复制 $K$ 次不改变嵌入

这确保编码器仅依赖于经验分布 $P_{i,m} = \frac{1}{m}\sum_{j=1}^m \delta_{x_{ij}}$。

**理论保证**：
- 分布不变编码器可捕获任意分布属性
- 非分布不变架构可能虚假编码与分布无关的噪声特征
- 分布不变性 + Hadamard 可微性 → 嵌入的中心极限定理：$\sqrt{m}(\mathcal{E}(S_{i,m}) - \phi(P_i)) \xrightarrow{d} \mathcal{N}(0, \Sigma_{\phi,i})$

**实现**：mean pooling 和 M/Z 估计量满足分布不变性；sum pooling 不满足

#### 2. 条件生成器的灵活性

任何条件生成模型都可用于 GDE：
- VAE（如 CVAE）
- 去噪扩散模型（DDPM）
- Sinkhorn 生成模型
- 切片 Wasserstein 模型
- 自回归序列模型（如 ProGen2, HyenaDNA）

#### 3. 从标签到分布的泛化

当数据不自然形成层次结构时，通过标签空间构造分布：
- **离散标签**：按标签分组
- **连续标签**：高斯核加权采样
- **噪声标签**：似然加权
- 统一为从标签先验 $Q^{(\mathcal{Y})}$ 采样的通用框架

### 理论性质

#### 预测充分统计量

GDE 学到的表示近似预测充分统计量——条件于该表示可预测新样本，同时边际化采样噪声。实验验证：Poisson 分布上 GDE 估计器的 MSE 优于 Rao-Blackwell 估计器（$n=10$ 时 3.12e-3 vs 3.79e-3）。

#### Wasserstein 几何

- 潜在空间 $L_2$ 距离与 $W_2$ 距离高度相关（高斯分布 $\rho = 0.96$，GMM $\rho = 0.76$）
- 潜在空间线性插值近似最优传输测地线
- 先验 $Q$ 不均匀时，几何发生自适应扭曲

## 实验关键数据

### 合成数据基准

| 模型 | Normal | GMM | MNIST | FMNIST |
|------|--------|-----|-------|--------|
| KME + DDPM | 0.04 | 2.17 | 80.46 | 111.01 |
| $W_2$ Wormhole | 0.20 | 2.88 | 263.29 | 320.18 |
| **GDE** | **0.02** | **1.82** | **63.79** | **102.21** |

### 应用1：患者级表示（6.3M单核RNA-seq）

| 指标 | Supervised | Semi-supervised GDE |
|------|-----------|-------------------|
| Accuracy | 0.8791 | **0.8887** |
| ROC AUC | 0.4872 | **0.5131** |
| F1 Score | 0.1293 | **0.1479** |

### 应用2：克隆群体建模（谱系追踪scRNA-seq）

GDE + CVAE 超越 Wasserstein Wormhole 超过 **2 bits** 的互信息

### 应用3：转录组扰动预测

| 方法 | $R^2$↑ | MSE↓ |
|------|--------|------|
| Mean (直接回归) | 0.378 | 1.855 |
| scVI | 0.421 | 1.551 |
| **GDE** | **0.458** | **1.501** |

### 应用4：单细胞图像表型预测

- 5072个基因扰动，2000万+单细胞图像
- 零样本预测 held-out 扰动的核信号强度：$R^2 = 0.7055$，MSE = 0.00068

### 应用5：酵母启动子设计（3400万序列）

GDE 嵌入空间恢复了表达量分位数的平滑梯度，重建的转录因子结合位点（TFBS）基序分布与真实数据高度一致

### 应用6：病毒蛋白时空建模（SARS-CoV2，100万序列）

- 时间预测 MAE：GDE **1.83±0.01** 月 vs ESM baseline 2.24±0.01 月
- 国家分类准确率：GDE 0.28 vs ESM 0.25 vs majority 0.21

### 关键发现

1. **mean-pooled deep sets + DDPM** 在 30 种编码器-生成器组合中表现最佳
2. **GDE 在所有合成基准上超越 KME 和 Wasserstein Wormhole**
3. **半监督 GDE 优于纯监督模型**，利用无标签数据的分布结构
4. **GDE 潜在空间天然具有 Wasserstein 几何**，与 OT 测地线对齐

## 亮点与洞察

- **概念优雅**：将"自编码分布"提炼为分布不变编码器 + 条件生成器的极简框架
- **理论深度**：连接预测充分统计量、信息几何、Wasserstein 空间三大理论
- **通用性极强**：同一框架跨越 DNA 序列、蛋白质序列、基因表达、显微图像四大数据域
- **中心极限定理保证**：推理时可以使用所有样本（数百万级），嵌入稳定收敛
- **先验感知的几何**：潜在空间几何随元分布 $Q$ 自适应调整，对高密度区域分配更高分辨率

## 局限与展望

1. **集合构造需要领域知识**：如何分组样本（元分布先验 $Q$ 的选择）依赖领域经验
2. **编码器梯度传播**：通过生成器传递梯度到编码器存在工程挑战
3. **大集合规模的扩展**：编码数百万样本虽有 CLT 保证但实际计算仍需优化
4. **理论假设可交换性**：不适用于集合内非 i.i.d. 的样本
5. **Wasserstein 等距的机制性证据不足**：目前仅有经验观察，缺乏形式化证明

## 相关工作与启发

- 泛化了核均值嵌入（KME）和 Wasserstein Wormhole 为 GDE 的特例
- 与 Meta Flow Matching、Fisher-Rao 流模型互补
- **核心启发**：任何条件生成模型都可以"免费"升级为分布表示学习器——只需配合分布不变编码器

## 评分

⭐⭐⭐⭐⭐ (5/5)

**理由**：概念创新性极高（将自编码器提升到分布空间），理论-实验双强（CLT+充分统计量+Wasserstein几何 × 6个大规模生物应用），通用性极强，实验规模令人印象深刻（6M细胞、20M图像、34M序列）。是分布级表示学习领域的标杆性工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] Towards Multiscale Graph-based Protein Learning with Geometric Secondary Structural Motifs](towards_multiscale_graph-based_protein_learning_with_geometric_secondary_structu.md)
- [\[NeurIPS 2025\] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)
- [\[CVPR 2025\] Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning](../../CVPR2025/computational_biology/unsupervised_foundation_model-agnostic_slide-level_representation_learning.md)
- [\[ICML 2025\] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics](../../ICML2025/computational_biology/global_context-aware_representation_learning_for_spatially_resolved_transcriptom.md)

</div>

<!-- RELATED:END -->
