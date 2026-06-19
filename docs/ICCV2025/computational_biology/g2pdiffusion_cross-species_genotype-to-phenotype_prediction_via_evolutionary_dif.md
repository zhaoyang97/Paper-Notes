---
title: >-
  [论文解读] G2PDiffusion: Cross-Species Genotype-to-Phenotype Prediction via Evolutionary Diffusion
description: >-
  [ICCV 2025][计算生物][genotype-to-phenotype] 提出G2PDiffusion，首个基于扩散模型的跨物种基因型到表型预测框架，通过进化信号（多序列比对MSA和环境上下文）条件化生成形态学图像，实现从DNA序列预测物种外观。 理解基因如何与环境因素交互决定表型是基因工程的核心挑战…
tags:
  - "ICCV 2025"
  - "计算生物"
  - "genotype-to-phenotype"
  - "扩散模型"
  - "multiple sequence alignment"
  - "cross-species"
  - "evolutionary biology"
---

# G2PDiffusion: Cross-Species Genotype-to-Phenotype Prediction via Evolutionary Diffusion

**会议**: ICCV 2025  
**arXiv**: [2502.04684](https://arxiv.org/abs/2502.04684)  
**代码**: 无  
**领域**: 计算生物  
**关键词**: genotype-to-phenotype, diffusion model, multiple sequence alignment, cross-species, evolutionary biology

## 一句话总结

提出G2PDiffusion，首个基于扩散模型的跨物种基因型到表型预测框架，通过进化信号（多序列比对MSA和环境上下文）条件化生成形态学图像，实现从DNA序列预测物种外观。

## 研究背景与动机

理解基因如何与环境因素交互决定表型是基因工程的核心挑战，对作物育种、保护生物学和个性化医疗有重大意义。当前基因型到表型预测模型面临两大瓶颈：

**局限于单一物种**：GWAS和QTL等传统方法依赖特定物种的标注数据，无法跨物种泛化

**标注成本高昂**：表型特征位于高维空间，需要专业设备和大量人力标注，跨物种研究成本更为巨大

作者的核心创新视角：**将图像作为形态学表型的代理**，将基因型到表型预测问题重新定义为条件图像生成任务。通过从数百万跨物种的DNA-图像对中学习，实现高效且可扩展的跨物种预测。这一视角突破了传统方法将表型建模为数值回归或分类的局限。

## 方法详解

### 整体框架

G2PDiffusion包含三个核心组件：(1) MSA检索引擎从外部数据库检索同源序列，识别进化保守和变异区域；(2) 环境感知MSA条件编码器，整合遗传信息和环境因素（经纬度）学习基因型-环境（GxE）交互表征；(3) 动态表型对齐模块，在扩散去噪过程中增强基因型-表型一致性。基于标准条件扩散模型生成256×256形态学图像。

### 关键设计

1. **MSA检索引擎 (Multiple Sequence Alignments Retrieval Engine)**: 使用MMseqs2工具从外部序列数据库中快速检索top-m条与查询DNA序列进化相似度最高的同源序列：$\mathcal{D}(G_q, m) = \text{top}_m(\{G_i\}, \text{MMseqs}(\cdot, G_q))$。检索结果通过列对比生成进化保守性向量$V = [v_1, ..., v_l]$，其中$v_i \in \{0,1\}$指示该位置是否保守。DNA序列使用k-mer分词（k=3）。设计动机：MSA揭示进化约束和功能保守区域，帮助区分哪些遗传变异真正影响表型。

2. **环境感知MSA条件编码器 (Environment-Aware MSA Conditioner)**: 设计双注意力机制处理MSA矩阵。进化感知行注意力使用进化向量生成的门控权重$\mathbf{w}_v$调制序列内注意力：$H^{row}_{i,:} = \text{Softmax}\left(\frac{QK^\top \odot \mathbf{w}_v}{\sqrt{d}}\right)V$。环境感知列注意力将经纬度映射到球坐标系$(x,y,z) = (\cos\beta\cos\lambda, \cos\beta\sin\lambda, \sin\beta)$后经MLP生成环境权重$\mathbf{w}_e$调制跨序列注意力。行列注意力输出经LayerNorm合并：$H^{MSA} = \text{LayerNorm}(H^{row} + H^{col})$。设计动机：行注意力捕捉进化约束，列注意力注入环境选择压力的影响。

3. **动态表型对齐采样 (Dynamic Alignment Sampling)**: 训练一个对齐模型$g_\phi(X_t, t)$将扩散过程中的含噪图像嵌入对齐到DNA嵌入。使用对比学习损失：$\mathcal{L}_{align} = -\log\frac{\exp[g_\phi(X_t,t) \cdot C^+]}{\sum_{j=1}^{B}\exp[g_\phi(X_t,t) \cdot C_j]}$。在采样时通过梯度引导将对齐信号注入去噪过程。设计动机：相比直接使用CLIP损失引导，动态对齐能更好适应扩散过程中的噪声特性。

### 损失函数 / 训练策略

- 扩散模型训练使用标准去噪目标：$L_{DM} = \mathbb{E}_{\epsilon,t}[\|\epsilon - \epsilon_\theta(X_t, t, C)\|_2^2]$
- 对齐模型使用对比学习损失$\mathcal{L}_{align}$
- 训练在8卡A100上进行，Adam优化器，学习率1e-5，batch size 128，训练100k步，余弦退火
- 采样时引导强度$w$控制对齐引导的强度

## 实验关键数据

### 主实验

**CLIBDScore和成功率 (BIOSCAN-5M Seen Set)**：

| 方法 | CLIBDScore Top-1 | Top-5 | 成功率 Top-1 | 成功率 Top-5 | 成功率 Top-100 |
|------|-----------------|-------|-------------|-------------|---------------|
| Random | 0.005 | - | 4.4% | - | - |
| DF-GAN | 0.054 | 0.154 | 5.6% | 18.7% | 52.6% |
| Stable Diffusion | 0.100 | 0.219 | 11.5% | 36.6% | 74.8% |
| ControlNet | 0.107 | 0.228 | 12.4% | 39.1% | 77.0% |
| **G2PDiffusion** | **0.182** | **0.302** | **31.7%** | **65.8%** | **94.0%** |

**PES (表型嵌入相似度)**：

| 方法 | Top-1 | Top-5 | Top-10 | Top-50 | Top-100 |
|------|-------|-------|--------|--------|---------|
| DF-GAN | 0.021 | 0.134 | 0.167 | 0.276 | 0.301 |
| Stable Diffusion | 0.062 | 0.207 | 0.240 | 0.349 | 0.389 |
| ControlNet | 0.061 | 0.212 | 0.254 | 0.359 | 0.403 |
| **G2PDiffusion** | **0.152** | **0.291** | **0.346** | **0.478** | **0.511** |

### 消融实验

**环境感知MSA条件编码器和动态对齐的效果**：

| 配置 | CLIBDScore Top-1/5 | 成功率 Top-1/5 | PES Top-1/5 |
|------|-------------------|---------------|-------------|
| Baseline (DNABERT) | 0.100/0.219 | 11.5%/26.6% | 0.062/0.187 |
| + Conditioner | 0.125/0.235 | 16.7%/28.2% | 0.098/0.254 |
| + Alignment | 0.167/0.289 | 27.1%/51.2% | 0.137/0.268 |
| **+ Both (完整)** | **0.182/0.302** | **31.7%/65.8%** | **0.152/0.291** |

**MSA检索数量m的影响**：

| m值 | CLIBDScore Top-1 | 成功率 Top-5 | PES Top-1 |
|-----|-----------------|-------------|-----------|
| 0 (无MSA) | 0.178 | 53.1% | 0.151 |
| 1 | **0.193** | **65.8%** | 0.143 |
| 2 (默认) | 0.182 | 65.8% | **0.152** |
| 3 | 0.166 | 58.9% | 0.128 |

### 关键发现

- G2PDiffusion在所有指标上大幅超越基线，Top-5成功率65.8% vs ControlNet的39.1%
- 环境感知条件编码器和动态对齐各自贡献显著，组合效果最佳（成功率从11.5%→31.7%）
- MSA检索m=1或2效果最好，过多检索引入噪声序列反而降低性能
- **未见物种的泛化能力**：在未见物种上Top-100成功率达80.3%，证明跨物种泛化能力
- 生成模型可以从同一基因型生成不同视角的图像，展现对3D结构的理解

## 亮点与洞察

1. **问题重定义的范式创新**：将基因型→表型预测从数值回归重定义为条件图像生成，一举解决跨物种泛化和高维表型建模两大难题
2. **进化信号的巧妙利用**：行/列双注意力分别建模序列内进化约束和跨序列环境选择压力，在数学上优雅且生物学上合理
3. **评估指标体系构建**：提出CLIBDScore、成功率和PES三个互补指标，从DNA-图像对齐、统计阈值和表型嵌入空间三个角度评估
4. **球坐标系环境编码**：将经纬度映射到球面坐标而非直接使用平面坐标，正确反映地球表面的几何特性

## 局限与展望

1. 仅在昆虫标本数据集(BIOSCAN-5M)上验证，未扩展到植物、哺乳动物等其他类群
2. 环境因素仅使用经纬度，缺乏温度、湿度、海拔等更丰富的生态变量
3. DNA barcode通常只涵盖部分基因（如COI），可能无法捕捉所有影响表型的遗传变异
4. Top-1成功率仅31.7%，距实用仍有差距，需要多次采样取最佳
5. 缺少对生成图像的精细表型特征（如翅脉、刚毛）的定量评估
6. 可探索将表型变异与具体遗传位点关联的可解释性分析

## 相关工作与启发

- **BIOSCAN-5M**: 最大的多模态昆虫标本数据集，提供DNA-图像-分类-地理配对数据
- **CLIBD**: CLIP式的DNA-图像对比学习模型，本文用其构建CLIBDScore评估指标
- **Stable Diffusion/ControlNet**: 条件图像生成的代表方法，作为扩散基线
- **AlphaFold**: 蛋白质结构预测的范式转变，本文在基因型-表型层面追求类似突破

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个基因型到表型的扩散模型，问题定义和方法设计均具开创性
- **实验充分度**: ⭐⭐⭐⭐ 指标体系完备，消融研究充分，包含未见物种泛化测试
- **写作质量**: ⭐⭐⭐⭐ 生物学动机清晰，但方法部分LaTeX公式较多影响可读性
- **价值**: ⭐⭐⭐⭐ 开辟AI辅助基因组分析的新方向，虽然距应用仍有距离但前景广阔

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Deciphering Genotype-Phenotype Mechanisms from High-Content Profiling via Knowledge-Guided Multi-modal Graph Learning](../../CVPR2026/computational_biology/deciphering_genotype-phenotype_mechanisms_from_high-content_profiling_via_knowle.md)
- [\[NeurIPS 2025\] Inferring Stochastic Dynamics with Growth from Cross-Sectional Data](../../NeurIPS2025/computational_biology/inferring_stochastic_dynamics_with_growth_from_cross-sectional_data.md)
- [\[ICLR 2026\] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering](../../ICLR2026/computational_biology/evoflows_evolutionary_edit-based_flow-matching_for_protein_engineering.md)
- [\[ICML 2026\] Protein Circuit Tracing via Cross-layer Transcoders](../../ICML2026/computational_biology/protein_circuit_tracing_via_cross-layer_transcoders.md)
- [\[NeurIPS 2025\] UniSite: The First Cross-Structure Dataset and Learning Framework for End-to-End Ligand Binding Site Detection](../../NeurIPS2025/computational_biology/unisite_the_first_cross-structure_dataset_and_learning_framework_for_end-to-end_.md)

</div>

<!-- RELATED:END -->
