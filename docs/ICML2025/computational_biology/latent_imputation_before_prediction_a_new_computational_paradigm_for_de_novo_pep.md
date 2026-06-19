---
title: >-
  [论文解读] Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing
description: >-
  [ICML 2025][计算生物][De Novo Peptide Sequencing] LIPNovo 提出在肽段预测前，通过隐空间补全（latent imputation）来弥补质谱中碎片缺失信息的新范式，利用可学习peak queries和二部匹配补全理论peak隐表示，在三个基准上大幅超越 CasaNovo 等 SOTA（氨基酸精度提升 5.6%-20%）。
tags:
  - "ICML 2025"
  - "计算生物"
  - "De Novo Peptide Sequencing"
  - "Missing Fragmentation"
  - "Latent Imputation"
  - "Bipartite Matching"
  - "Mass Spectrometry"
---

# Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing

**会议**: ICML 2025  
**arXiv**: [2505.17524](https://arxiv.org/abs/2505.17524)  
**代码**: [https://github.com/usr922/LIPNovo](https://github.com/usr922/LIPNovo)  
**领域**: 生物信息学 / 蛋白质组学  
**关键词**: De Novo Peptide Sequencing, Missing Fragmentation, Latent Imputation, Bipartite Matching, Mass Spectrometry

## 一句话总结
LIPNovo 提出在肽段预测前，通过隐空间补全（latent imputation）来弥补质谱中碎片缺失信息的新范式，利用可学习peak queries和二部匹配补全理论peak隐表示，在三个基准上大幅超越 CasaNovo 等 SOTA（氨基酸精度提升 5.6%-20%）。

## 研究背景与动机
**领域现状**：从头肽段测序使用 encoder-decoder 架构将质谱编码为 latent 后自回归预测氨基酸序列。

**现有痛点**：质谱中碎片信息缺失严重（因碎片效率不足和仪器限制），导致序列预测信息不足。

**核心矛盾**：现有方法直接从不完整谱图预测，缺失碎片比例越高性能衰降越明显。

**切入角度**：在预测前在隐空间补全缺失的理论碎片谱，类似 DETR 的集合预测框架。

**核心idea**：训练时用已知肽段序列生成理论谱作为监督，推理时补全模块在无理论谱的情况下也能工作。

## 方法详解

### 整体框架
观测谱 → Peak Encoder → 隐表示 $z$ → Imputation Module (Transformer Decoder + 可学习 queries) → 补全的理论peak隐表示 → 拼接到原始 $z$ → Peptide Decoder → 氨基酸序列。

### 关键设计

1. **隐空间补全（Latent Space Imputation）**:

    - 计算理论谱 $x'$（所有 b- 和 y-ion 的 m/z 值），用同一编码器编码得 $z'$
    - 补全模块 $\Phi_\theta$ 以观测谱隐表示 $z$ 为输入，生成固定大小 $M$ 个预测
    - 训练目标：使预测逼近理论谱隐表示 $z'$

2. **二部匹配训练（Bipartite Matching）**:

    - 受 DETR 启发，用 Hungarian 算法寻找最优匹配 $\hat{\sigma}$
    - 匹配代价结合 MSE 距离和置信概率：$\mathcal{L}_{\text{Match}} = \|...\|^2 + 1 - p_{\sigma(j)}$
    - 补全损失结合匹配MSE和二分类CE

3. **推理时丢弃理论谱分支**:

    - 训练时需要真实标签生成理论谱；推理时仅用补全模块的预测
    - 高置信度补全结果拼接到原始谱表示中

### 损失函数 / 训练策略
总损失 = 补全损失 $\mathcal{L}_{\text{Imputation}}$ + 肽段预测交叉熵。

## 实验关键数据

### 主实验

| 数据集 | 指标 | LIPNovo | CasaNovo | 提升 |
|--------|------|---------|----------|------|
| Nine-species | AA Prec | 0.848 | 0.792 | +5.6% |
| Seven-species | AA Prec | 0.652 | 0.452 | +20.0% |
| HC-PT | AA Prec | 0.699 | 0.587 | +11.2% |

### 消融实验

| 配置 | Nine-species AA Prec | 说明 |
|------|---------------------|------|
| Full LIPNovo | 0.848 | 完整模型 |
| w/o Imputation | 0.792 | 退化为CasaNovo |
| w/o 高置信度过滤 | 0.831 | 直接拼接所有预测 |
| w/o 二部匹配 | 0.815 | 使用顺序匹配 |

### 关键发现
- 碎片缺失比例越高，LIPNovo 优势越明显
- 补全质量与肽段测序性能正相关，验证了补全的有效性
- 即使在低缺失率下也有提升，说明补全提供了互补信息

## 亮点与洞察
- "先补全再预测"范式简洁有效，可迁移到其他缺失数据场景
- 利用 DETR 的集合预测框架解决可变长度目标问题
- 训练时需要理论谱作为监督但推理时不需要，设计巧妙

## 局限与展望
- 理论谱计算假设电荷为+1且强度统一，可能不够精确
- 固定大小 $M$ 的预测集合可能浪费或不足
- 对翻译后修饰(PTM)的处理有待加强

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 补全范式新颖且有效
- 实验充分度: ⭐⭐⭐⭐ 三个基准+消融
- 写作质量: ⭐⭐⭐⭐ 流程清晰
- 价值: ⭐⭐⭐⭐⭐ 对蛋白质组学有重要推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[ICLR 2026\] A New Paradigm for Genome-wide DNA Methylation Prediction Without Methylation Input](../../ICLR2026/computational_biology/a_new_paradigm_for_genome-wide_dna_methylation_prediction_without_methylation_in.md)
- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](../../NeurIPS2025/computational_biology/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[ICML 2025\] Protein Structure Tokenization: Benchmarking and New Recipe](protein_structure_tokenization_benchmarking_and_new_recipe.md)

</div>

<!-- RELATED:END -->
