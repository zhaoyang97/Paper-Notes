---
title: >-
  [论文解读] Geometric Representation Condition Improves Equivariant Molecule Generation
description: >-
  [ICML 2025 (Spotlight)][计算生物][分子生成] GeoRCG 提出两阶段分子生成框架——先生成低维的几何表示(informative representation)，再以此为条件生成完整分子，在条件分子生成任务上平均提升 50%，同时可将扩散步数从 1000 减少到 100。
tags:
  - "ICML 2025 (Spotlight)"
  - "计算生物"
  - "分子生成"
  - "等变扩散"
  - "几何表示条件"
  - "两阶段生成"
  - "条件分子生成"
---

# Geometric Representation Condition Improves Equivariant Molecule Generation

**会议**: ICML 2025 (Spotlight)  
**arXiv**: [2410.03655](https://arxiv.org/abs/2410.03655)  
**代码**: [https://github.com/GraphPKU/GeoRCG](https://github.com/GraphPKU/GeoRCG)  
**领域**: 计算生物  
**关键词**: 分子生成, 等变扩散, 几何表示条件, 两阶段生成, 条件分子生成

## 一句话总结
GeoRCG 提出两阶段分子生成框架——先生成低维的几何表示(informative representation)，再以此为条件生成完整分子，在条件分子生成任务上平均提升 50%，同时可将扩散步数从 1000 减少到 100。

## 研究背景与动机

**领域现状**：3D 分子生成是药物设计的关键环节。等变扩散模型（如 EDM、GeoLDM）已成为该领域主流方法，能生成 SE(3)-等变的 3D 分子结构。

**现有痛点**：现有模型在无条件生成时表现尚可，但在条件生成（给定特定属性生成分子）时表现明显不足。这是因为条件信号（如 HOMO-LUMO gap）难以有效引导高维 3D 结构的生成过程。

**核心矛盾**：直接用标量属性条件化 3D 分子扩散过程，条件信号太弱（一个数值 vs 几十到上百个原子的 3D 坐标），引导效果有限。

**本文目标**：如何在等变分子生成中有效注入条件信息，特别是在条件分子生成任务中大幅提升性能。

**切入角度**：引入一个信息量丰富的中间"几何表示"作为桥梁——先生成表示，再以表示为条件生成分子。

**核心 idea**：用易于生成的语义丰富几何表示为高难度的 3D 分子生成提供目标导向的引导。

## 方法详解

### 整体框架

- **第一阶段**：在低维空间生成几何表示 $\mathbf{r}$（包含分子的关键结构和属性信息）
- **第二阶段**：以 $\mathbf{r}$ 为条件，用等变扩散模型生成完整的 3D 分子 $(\mathbf{x}, \mathbf{h})$
- 两阶段均可独立训练和推理

### 关键设计

1. **几何表示设计**:

    - 从预训练的等变 GNN 提取分子的几何表示
    - 表示编码了原子类型、空间排布、局部几何特征等信息
    - 比原始 3D 坐标维度低，但携带关键语义信息
    - **设计动机**：条件信号需要既信息丰富又易于生成——太简单（标量属性）无法有效引导，太复杂（完整分子）等于没有简化

2. **两阶段解耦生成**:

    - 第一阶段独立训练一个生成 $\mathbf{r}$ 的扩散模型（可以是条件或无条件的）
    - 第二阶段训练条件扩散：$p_\theta(\mathbf{x}, \mathbf{h} | \mathbf{r})$
    - 理论保证：两阶段的组合分布可以逼近真实分子分布
    - **设计动机**：分解复杂生成问题为两个较简单的子问题

3. **等变性保证**:

    - 几何表示 $\mathbf{r}$ 本身满足 SE(3)-等变性
    - 条件扩散过程保持等变性
    - 组合后的整体生成仍然是等变的
    - **设计动机**：物理分子必须尊重旋转/平移对称性

### 损失函数 / 训练策略

- 表示提取：预训练等变 GNN 的潜层输出
- 第一阶段：标准扩散损失 on $\mathbf{r}$
- 第二阶段：条件等变扩散损失 $\|\epsilon - \epsilon_\theta(\mathbf{x}_t, \mathbf{h}_t, t, \mathbf{r})\|^2$
- 基础生成器可选 EDM 或 SemlaFlow

## 实验关键数据

### 主实验

| 数据集 | 任务 | GeoRCG | 基线方法 | 提升 |
|--------|------|--------|---------|------|
| QM9 | 无条件生成 | 显著改善 | EDM/GeoLDM | 质量明显提升 |
| GEOM-DRUG | 无条件生成 | 显著改善 | EDM | 更好 |
| QM9 | 条件生成 (α) | SOTA | EDM/cond | **~50% 平均提升** |
| QM9 | 条件生成 (gap) | SOTA | EDM/cond | **~50% 平均提升** |
| QM9 | 条件生成 (μ) | SOTA | EDM/cond | **~50% 平均提升** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 无几何表示条件 | 基线 | 标准单阶段生成 |
| 随机噪声作为条件 | 接近基线 | 说明需信息丰富的表示 |
| 标量属性条件 | 小幅改善 | 信息太少 |
| 几何表示条件 (1000步) | 最优 | 完整配置 |
| 几何表示条件 (**100步**) | **接近 1000步** | 步数减少 10x |

### 关键发现

- **条件生成提升巨大**：平均 50% 的性能改善，这是通过更好的条件化方式实现的
- **步数大幅减少**：几何表示引导下，100 步即可接近 1000 步的质量
- **通用框架**：EDM 和 SemlaFlow 两种基础生成器都从 GeoRCG 获益
- 表示的语义丰富性是关键——随机表示无效

## 亮点与洞察

1. **理论支撑**：证明了两阶段分解生成可以逼近真实分布
2. **简单而有效**：方法概念简单但效果惊人（50% 提升）
3. **双重收益**：同时提升质量和减少采样步数
4. **Spotlight 论文**：ICML 2025 以 Spotlight 接收，说明社区认可

## 局限与展望

1. 需要预训练的 GNN 来提取表示，引入额外依赖
2. 表示设计可能需要领域知识来选择最优的 GNN 和层
3. 两阶段推理总时间需评估（虽然第二阶段步数减少）
4. 在更大的分子（如蛋白质）上的扩展性未验证

## 相关工作与启发

- EDM (Equivariant Diffusion Model) 和 GeoLDM 是主要基线
- Classifier-guided diffusion 的思想在这里被推广为"表示引导"
- 启发：中间表示条件化可能是提升所有条件扩散模型的通用策略

## 评分
- 新颖性: ⭐⭐⭐⭐ 两阶段表示条件化的思路简洁有效
- 实验充分度: ⭐⭐⭐⭐⭐ QM9+GEOM-DRUG，多属性条件，步数消融
- 写作质量: ⭐⭐⭐⭐ 清晰，理论+实验平衡好
- 价值: ⭐⭐⭐⭐⭐ 50% 提升 + Spotlight = 高影响力工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Scalable Non-Equivariant 3D Molecule Generation via Rotational Alignment](scalable_non-equivariant_3d_molecule_generation_via_rotational_alignment.md)
- [\[ICML 2025\] Neural Graph Matching Improves Retrieval Augmented Generation in Molecular Machine Learning](neural_graph_matching_improves_retrieval_augmented_generation_in_molecular_machi.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](../../NeurIPS2025/computational_biology/unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[NeurIPS 2025\] Manipulating 3D Molecules in a Fixed-Dimensional E(3)-Equivariant Latent Space](../../NeurIPS2025/computational_biology/manipulating_3d_molecules_in_a_fixed-dimensional_e3-equivariant_latent_space.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](geometric_generative_modeling_with_noise-conditioned_graph_networks.md)

</div>

<!-- RELATED:END -->
