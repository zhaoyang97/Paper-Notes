---
title: >-
  [论文解读] Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction
description: >-
  [NeurIPS 2025][图像生成][图扩散模型] 提出 GeoMancer 框架，通过黎曼 GyroKernel 自编码器替代数值不稳定的指数映射，将多层级图特征解耦到任务特定的积流形上，并引入流形约束扩散和自引导生成策略，在分子生成、节点分类和图回归等任务上统一建模并取得 SOTA 性能。 图扩散模型近年来在图结构数…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "图扩散模型"
  - "黎曼流形"
  - "积流形"
  - "分子生成"
  - "几何自编码器"
---

# Toward a Unified Geometry Understanding: Riemannian Diffusion Framework for Graph Generation and Prediction

**会议**: NeurIPS 2025  
**arXiv**: [2510.04522](https://arxiv.org/abs/2510.04522)  
**代码**: [GitHub](https://github.com/RingBDStack/GeoMancer)  
**领域**: 扩散模型 / 图生成 / 几何学习  
**关键词**: 图扩散模型, 黎曼流形, 积流形, 分子生成, 几何自编码器

## 一句话总结

提出 GeoMancer 框架，通过黎曼 GyroKernel 自编码器替代数值不稳定的指数映射，将多层级图特征解耦到任务特定的积流形上，并引入流形约束扩散和自引导生成策略，在分子生成、节点分类和图回归等任务上统一建模并取得 SOTA 性能。

## 研究背景与动机

图扩散模型近年来在图结构数据的学习和生成上取得了显著进展，可以通过潜在空间扩散将生成和预测任务统一建模——将预测任务重新表述为条件生成问题。然而，现有方法将节点、边、图级别特征嵌入到同一个欧几里得潜在空间中，忽视了图数据固有的非欧几里得性质。

作者通过 t-SNE 可视化发现：不同层级的潜在表示在共享的欧几里得空间中发生了纠缠，尽管它们具有**不同的内在几何属性**（曲率异质性）。例如，层级结构适合双曲空间建模，密集连接图更适合球面空间。将这些几何属性不同的特征强行混在同一空间中，无法充分释放其几何潜力。

构建理想的黎曼扩散框架面临两大挑战：（1）**编码阶段的数值不稳定性**：使用指数映射将特征投射到积流形上容易导致数值爆炸，难以优化；（2）**扩散生成阶段的流形偏移**：生成过程中模型倾向于偏离原始数据流形，且无条件生成任务缺乏引导信息来纠正这种偏移。

## 方法详解

### 整体框架

GeoMancer 包含三部分：（1）黎曼 GyroKernel 自编码器——使用广义 Fourier 变换替代指数映射实现等距不变特征映射，将多层级特征编码到各自的积流形上；（2）流形约束扩散——在几何潜在空间中进行经典扩散，并引入 CFG++ 实现流形约束条件生成；（3）自引导策略——对无条件生成任务通过聚类生成伪标签，统一为条件生成框架。

### 关键设计

1. **黎曼 GyroKernel 映射**：用 Bochner 定理构造等距不变核函数，替代数值不稳定的指数/对数映射。在曲率为 $\kappa$ 的陀螺向量球 $\mathbb{G}^n_\kappa$ 中，广义 Fourier 特征为 $\text{gF}^\kappa_{\omega,b,\lambda}(x) = A_{\omega,x}\cos(\lambda\langle\omega,x\rangle_\kappa + b)$，其中 $\langle\omega,x\rangle_\kappa = \log\frac{1+\kappa\|x\|^2}{\|x-\omega\|^2}$ 是陀螺向量球中的符号距离。初始化不同曲率 $\kappa_i$ 的积流形表示 $V_\kappa$，通过 $\bar{V}_\kappa = \phi_{\text{gF}}(V_\kappa)$ 获得等距不变的欧几里得特征。核心优势是既保留了黎曼空间的几何属性，又能直接在欧几里得空间中操作。

2. **多层级编码-解码架构**：编码器使用图 Transformer (GrIT) 同时处理节点和边特征，通过注意力机制捕获关系：$z^{l+1}_{e_{ij}} = \sigma(\rho((Qz^l_{x_i}, Kz^l_{x_j}) \odot E_w z^l_{e_{ij}}) + E_b z^l_{e_{ij}})$。编码后将几何先验赋予嵌入 $\bar{Z} = Z\bar{V}_\kappa$，使每个维度捕获不同的几何信息。解码器将复杂积流形拆解为简单子流形 $\mathcal{M} \to \mathcal{M}_1 \times \cdots \times \mathcal{M}_m$，为每个任务层级选择最适合的几何表示。

3. **自引导策略**：对无条件图生成缺少标签引导的问题，利用图级表示 $Z_G$ 的 k-means 聚类生成伪标签 $C$，将无条件生成重新表述为条件生成 $P(G|C)$。在采样阶段随机选择 $C$。这使所有图任务（无条件生成、条件生成、预测）统一为条件生成框架。

4. **流形约束条件生成**：采用 CFG++ 方法实现流形约束采样，生成过程为：$\tilde{Z_0} = (Z_t - \sqrt{1-\bar{\alpha}_t}\tilde{\epsilon}_\theta(Z_t, \tau(y))) / \sqrt{\bar{\alpha}_t}$，其中 $\tilde{\epsilon}_\theta$ 结合条件和无条件噪声预测：$\tilde{\epsilon}_\theta(\hat{Z_t}, \tau(y)) = (1-\lambda)\epsilon_\theta(\hat{Z_t}, \tau(y)) - \lambda\epsilon_\theta(\hat{Z_t})$。这确保生成的数据保持在干净数据流形上。

### 损失函数 / 训练策略

训练目标包括任务损失和正则化约束：$\mathcal{L} = \mathcal{L}_{tgt} + \mathcal{L}_{reg}$。任务损失根据任务类型选择交叉熵（分类/生成）或 MSE（回归）。正则化损失为 KL 散度 $D_{\text{KL}}(q(Z|(X,E)) \| \mathcal{N}(0,I))$，防止潜在空间高方差。扩散训练使用标准噪声预测策略。

## 实验关键数据

### 主实验

QM9 无条件分子生成：

| 模型 | 有效性(%)↑ | 唯一性(%)↑ | FCD↓ | NSPDK↓ | 新颖性(%)↑ |
|------|:--------:|:--------:|:----:|:------:|:--------:|
| DiGress | 99.01 | 96.34 | 0.25 | 0.0003 | 35.46 |
| GruM | 99.69 | 96.90 | 0.11 | 0.0002 | 24.15 |
| LGD | 98.46 | 97.53 | 0.32 | 0.0004 | 56.35 |
| **GeoMancer** | **100.00** | 95.74 | **0.09** | **0.0002** | **90.43** |

节点分类准确率：

| 模型 | Photo | Physics | Pubmed | Cora | Citeseer |
|------|:-----:|:-------:|:------:|:----:|:--------:|
| NAGphormer | 95.49 | 97.34 | 91.76 | 82.13 | 71.40 |
| LGD | 96.94 | 98.55 | 92.88 | 82.81 | 72.40 |
| **GeoMancer** | **97.05** | **98.78** | **93.10** | **83.50** | **72.60** |

### 消融实验

| 配置 | 有效性(%)↑ | FCD↓ | NSPDK↓ | 新颖性(%)↑ | 说明 |
|------|:--------:|:----:|:------:|:--------:|------|
| w/o self-guidance | 98.99 | 0.12 | 0.0003 | 54.06 | 伪标签引导对新颖性贡献显著 |
| w/o CFG++ | 100.00 | 0.09 | 0.0010 | 76.43 | 流形约束改善分布拟合(NSPDK) |
| w/o Riemannian | 100.00 | 0.25 | 0.0004 | 90.26 | 黎曼几何显著提升 FCD |
| **GeoMancer (完整)** | **100.00** | **0.09** | **0.0002** | **90.43** | 三者互补 |

### 关键发现

- GeoMancer 在 QM9 上实现 100% 有效性的同时将新颖性从 56.35%（LGD）大幅提升至 90.43%
- 即使不使用 3D 信息也能实现高质量的条件分子生成
- 自引导机制是新颖性提升的关键（+36%），黎曼几何对分布拟合质量提升最大
- 流形可视化显示不同任务层级确实偏好不同曲率的子流形
- 图回归任务（ZINC12k）上超越了所有传统回归模型和图 Transformer

## 亮点与洞察

- 统一框架设计极具野心：一个模型同时处理无条件生成、条件生成、节点分类和图回归四类任务
- GyroKernel 替代指数映射的设计巧妙：保留了黎曼空间的几何属性同时避免了数值问题，使整个扩散过程可以在欧几里得空间中进行
- 自引导策略简单有效：利用潜在空间已有的几何信息生成伪标签，将所有任务统一为条件生成

## 局限与展望

- 积流形的组件数量和曲率选择是超参数，当前通过可学习曲率部分解决但缺乏理论指导
- 分子生成仅在 QM9（≤9重原子）上验证，对更大分子的扩展性有待检验
- 节点分类的提升幅度相对较小，可能因为该任务本身几何复杂度较低
- 各组件（GyroKernel、积流形解耦、CFG++）来自不同已有工作的组合，核心原创贡献的边界感稍弱
- 未与最新的 Riemannian Diffusion 方法（如 HypDiff）在图生成上直接对比

## 相关工作与启发

- 与 LGD 的关系：在同一潜在图扩散框架上增加了几何意识，思路延续但改进显著
- GyroKernel 方法借鉴自 MotifRGC 和 HyLA，但推广到了积流形和多层级图任务
- CFG++ 流形约束采样来自图像生成领域，被巧妙地应用到图生成中
- 可启发将几何感知扩散框架应用于蛋白质设计、材料科学等更复杂的图结构数据

## 评分

- **新颖性**: ⭐⭐⭐⭐ 几何感知图扩散框架的统一视角新颖，但组件多为已有技术组合  
- **实验充分度**: ⭐⭐⭐⭐ 任务覆盖面广（生成+分类+回归），消融完整  
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，但符号较多需要一定数学背景  
- **价值**: ⭐⭐⭐⭐ 为图数据的几何理解提供了统一框架，对分子建模领域有实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Riemannian Consistency Model](riemannian_consistency_model.md)
- [\[NeurIPS 2025\] Co-Reinforcement Learning for Unified Multimodal Understanding and Generation](coreinforcement_learning_for_unified_multimodal_understandin.md)
- [\[CVPR 2025\] Dual Diffusion for Unified Image Generation and Understanding](../../CVPR2025/image_generation/dual_diffusion_for_unified_image_generation_and_understanding.md)
- [\[NeurIPS 2025\] LLM Meets Diffusion: A Hybrid Framework for Crystal Material Generation](llm_meets_diffusion_a_hybrid_framework_for_crystal_material_generation.md)
- [\[NeurIPS 2025\] Exploring Variational Graph Autoencoders for Distribution Grid Data Generation](exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)

</div>

<!-- RELATED:END -->
