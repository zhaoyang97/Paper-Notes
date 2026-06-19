---
title: >-
  [论文解读] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design
description: >-
  [ICML2025][计算生物][统一分子生成] 提出 UniMoMo，首个统一小分子、肽和抗体三类分子的 3D binder 设计框架，使用“块图”作为统一表示、迭代全原子自编码器压缩潜空间、E(3)-等变扩散模型生成，在三个基准上超越领域特定模型。 领域现状 领域现状：三类分子各有优势：小分子适合口服（吸收好）、肽擅长细…
tags:
  - "ICML2025"
  - "计算生物"
  - "统一分子生成"
  - "小分子"
  - "肽"
  - "抗体"
  - "几何潜空间扩散"
  - "自编码器"
  - "构建块表示"
---

# UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design

**会议**: ICML2025  
**arXiv**: [2503.19300](https://arxiv.org/abs/2503.19300)  
**代码**: 无  
**领域**: 生物分子设计 / 药物发现  
**关键词**: 统一分子生成, 小分子, 肽, 抗体, 几何潜空间扩散, 自编码器, 构建块表示

## 一句话总结

提出 UniMoMo，首个统一小分子、肽和抗体三类分子的 3D binder 设计框架，使用“块图”作为统一表示、迭代全原子自编码器压缩潜空间、E(3)-等变扩散模型生成，在三个基准上超越领域特定模型。

## 研究背景与动机

### 领域现状

**领域现状**：三类分子各有优势**：小分子适合口服（吸收好）、肽擅长细胞内靶向（穿透能力）、抗体高特异性治疗重疾。

### 现有痛点

**现有痛点**：现有方法的片面性**：各领域都有专门的生成模型，无法跨域共享数据和知识。

### 核心矛盾

**核心矛盾**：统一表示的挑战**：小分子是功能基团组合，肽/抗体是氨基酸线性排列，纯原子图忽略层次化先验且计算成本高。

### 解决思路

**解决思路**：跨域可迁移性**：分子-蛋白质相互作用的物理化学规则是通用的，统一建模可利用更大更多样的数据。

## 方法详解

### 统一表示：块图 (Graph of Blocks)

- 每个 Block 对应一个标准氨基酸或一个分子片段（由 Principal Subgraph 算法提取）。
- 保留全原子几何和层次化结构。
- 二进制 prompt 控制生成类型（1=仅氨基酸，0=允许分子片段）。
- 词汇表 $\mathbb{V}$ 包含 20 种标准氨基酸 + 提取的常见分子片段（如苯环、吲哚等）。
- 分子图 $\mathcal{G} = (\mathcal{V}, \mathcal{E})$，其中 $\mathcal{V}$ 包含每个块的原子类型和坐标，$\mathcal{E}$ 包含块内和块间化学键。

### 迭代全原子自编码器 (VAE)

- **编码器**：将每个 Block 压缩为潜表示 $(\bm{z}_i \in \mathbb{R}^8, \vec{\bm{z}}_i \in \mathbb{R}^3)$，KL 正则化。
- **解码器两阶段**：
    - 第一步：预测 Block 类型 $s_i \in \mathbb{V}$，查表获取原子类型和键内化学键。
    - 第二步：迭代式结构模块，类似轻量 Flow Matching，10 步由高斯采样初始化逼近真实原子坐标。
- **键间化学键预测**：基于空间邻近 (3.5Å) 的原子对预测键类型。

### 几何潜空间扩散模型

- DDPM 在潜空间 $[\bm{z}_i, \vec{\bm{z}}_i]$ 上运行，避免全原子反复迭代。
- 去噪网络用等变 Transformer (GeoTF)，保证 E(3) 等变性。
- 训练目标：
  $$\mathcal{L}_{\text{LDM}} = \mathbb{E}_{t}\left[\frac{\sum_i \|\epsilon_i - \epsilon_\theta(\mathcal{Z}_x^t, \mathcal{Z}_y, t)[i]\|^2}{|\mathcal{Z}_x^t|}\right]$$

## 实验关键数据

### 肽设计

| 模型 | AAR↑ | C-RMSD↓ | L-RMSD↓ | ΔG↓ | IMP↑ |
|---|---|---|---|---|---|
| RFDiffusion | 34.68% | 4.69 | 1.88 | -13.47 | 5.38% |
| PepFlow | 35.47% | 2.87 | 1.79 | -21.71 | 15.22% |
| PepGLAD | **38.62%** | 2.74 | 1.60 | -23.12 | 18.28% |
| **UniMoMo (single)** | 37.59% | **2.48** | **1.48** | **-28.72** | - |

### 抗体和小分子

- 抗体 CDR-H3 设计：UniMoMo 在 AAR 和 RMSD 上均与 MEAN/DiffAb 竞争或超越。
- 小分子 SBDD：在 Vina Score、QED、SA 等指标上与 TargetDiff、DecompDiff 竞争。

### 消融实验

- **多域联合训练 vs 单域训练**：统一模型在多数指标上优于单域模型，验证了跨域可迁移性。
- **混合分子类型生成**：对同一靶点可生成不同类型的 binder。

## 亮点与洞察

1. **首个统一分子生成框架**：用块图表示弥合小分子与生物大分子的表示差异。
2. **潜空间扩散的效率**：在块级潜空间做扩散，复杂度大幅低于全原子扩散。
3. **迭代解码器**：轻量 flow matching 式解码具体原子坐标，平衡了精度与效率。
4. **跨域迁移实证**：联合训练碮实地提升了各域性能。5. **训练技巧**：编码器输出坐标加随机噪声增强解码器鲁棒性；teacher forcing 以 50% 概率使用 GT 键间化学键。
6. **距离损失设计**：仅在 $t < 0.25$ 时施加原子对距离约束，避免早期混沌阶段的无效约束。
7. **推理两轮精炼**：生成后额外做一轮编码-解码循环，利用预测的键间化学键进一步优化结构。

### 训练数据规模

- 肽：~8K 肽-蛋白质复合物
- 抗体：~5K 抗原-抗体结构
- 小分子：CrossDocked2020 数据集
- 统一训练利用了约 13K+ 跨域样本

## 局限与展望

- 未经实验室湿实验验证，仅依赖计算指标（如 Vina Score）。
- 片段词汇表的大小和质量影响生成多样性，次优的分解可能导致无法生成某些结构。
- 仅处理单链 binder，未扩展到多链复合物或共价修饰。
- 解码器中键间键预测与坐标的一致性可能存在矛盾。
- 潜空间维度 $d=8$ 的选择未充分消融。
- 对柔性强的分子（如长链小分子）效果未知。

## 相关工作与启发

- **PepGLAD (Kong et al., 2024b)**：本文的肽设计基线，同样使用几何潜扩散。
- **GET (Kong et al., 2024a)**：统一生物分子表示先驱，但仅用于相互作用预测。
- **AlphaFold 3**：跨分子类型结构预测的里程碑，开启了统一建模思路。
- **启发**：统一表示 + 跨域训练的范式可推广到核酸、糖类等更多分子类型。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Uncertainty-Aware Multi-Objective Reinforcement Learning-Guided Diffusion Models for 3D De Novo Molecular Design](../../NeurIPS2025/computational_biology/uncertainty-aware_multi-objective_reinforcement_learning-guided_diffusion_models.md)
- [\[NeurIPS 2025\] Towards Unified and Lossless Latent Space for 3D Molecular Latent Diffusion Modeling](../../NeurIPS2025/computational_biology/towards_unified_and_lossless_latent_space_for_3d_molecular_latent_diffusion_mode.md)
- [\[NeurIPS 2025\] Pharmacophore-Guided Generative Design of Novel Drug-Like Molecules](../../NeurIPS2025/computational_biology/pharmacophore-guided_generative_design_of_novel_drug-like_molecules.md)
- [\[ICML 2025\] Geometric Generative Modeling with Noise-Conditioned Graph Networks](geometric_generative_modeling_with_noise-conditioned_graph_networks.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](../../NeurIPS2025/computational_biology/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)

</div>

<!-- RELATED:END -->
