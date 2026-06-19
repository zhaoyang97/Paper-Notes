---
title: >-
  [论文解读] Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines
description: >-
  [ICCV 2025][计算生物][显微图像表征学习] 提出将外部生物知识（蛋白质互作图谱+单细胞基础模型的转录组特征）整合到显微图像预训练中，显式解耦扰动特异性和细胞系特异性表征，提升模型在未见细胞系上的扰动筛查泛化能力。 荧光显微镜图像分析在药物发现中至关重要，但将预训练模型迁移到新的（de novo）细胞系时面临重大挑…
tags:
  - "ICCV 2025"
  - "计算生物"
  - "显微图像表征学习"
  - "扰动筛查"
  - "生物知识图谱"
  - "单细胞基础模型"
  - "de novo细胞系"
---

# Integrating Biological Knowledge for Robust Microscopy Image Profiling on De Novo Cell Lines

**会议**: ICCV 2025  
**arXiv**: [2507.10737](https://arxiv.org/abs/2507.10737)  
**代码**: [https://github.com/The-Real-JerryChen/BioMicroscopyProfiler](https://github.com/The-Real-JerryChen/BioMicroscopyProfiler)  
**领域**: 计算生物  
**关键词**: 显微图像表征学习, 扰动筛查, 生物知识图谱, 单细胞基础模型, de novo细胞系

## 一句话总结

提出将外部生物知识（蛋白质互作图谱+单细胞基础模型的转录组特征）整合到显微图像预训练中，显式解耦扰动特异性和细胞系特异性表征，提升模型在未见细胞系上的扰动筛查泛化能力。

## 研究背景与动机

荧光显微镜图像分析在药物发现中至关重要，但将预训练模型迁移到新的（de novo）细胞系时面临重大挑战：

**细胞系间的异质性**：不同细胞系在形态（细胞形状、大小、细胞核/质比）和生物学（基因表达谱、信号通路活性）上存在巨大差异。即使施加相同的基因扰动，不同细胞系的表型响应也大不相同

**虚假特征依赖**：由于训练集中细胞系数量有限，模型可能学习到与细胞系相关但非因果性的虚假特征，在新细胞系上性能急剧下降

**现有预训练策略局限**：弱监督学习（WSL）和自监督方法（MAE、DINO）虽然在已知扰动上表现良好，但未专门设计来处理de novo细胞系场景

## 方法详解

### 整体框架

在现有预训练方法（WSL、SimCLR、BYOL、MoCo v3、MAE）基础上，增加两个生物知识引导组件：(1) 基于蛋白质互作网络构建扰动关系图，通过图正则化损失引导模型学习扰动特异性表征；(2) 利用单细胞基础模型编码RNA-seq数据得到细胞系特异性表征，以token形式融入ViT。

### 关键设计

1. **扰动关系图构建与正则化**：

    - 构建扰动关系图 $\mathcal{G} = (\mathcal{V}, \mathcal{E}, W, \psi)$，节点为基因/化学扰动，边权重来自三个生物数据库：STRING（蛋白质互作得分，置信度阈值200）、Hetionet（二值连接，通过random walk得到概率矩阵）、以及基于原始图像特征的基因-基因相似度
    - **图拉普拉斯正则化**：$\mathcal{L}_{lap} = \text{tr}(F^\top(D-W)F)$，强制高权重连接的扰动具有相似特征表征
    - **图节点对比学习**：$\mathcal{L}_{con} = \frac{1}{|\mathcal{N}(i)|}\sum_{v_j \in \mathcal{N}(i)} \log \frac{\exp(\text{sim}(f_{v_i}, f_{v_j})/\tau)}{\sum_{k \in \mathcal{V}} \exp(\text{sim}(f_{v_i}, f_{v_k})/\tau)}$，拉近生物学相关扰动、推远无关扰动的特征
    - **理论保证**（Proposition 3.1）：最小化图正则化损失隐式最小化同一扰动类别内的类内距离

2. **细胞系特异性表征学习**：

    - 从GSE portal收集各细胞系的RNA-seq数据，选取 $k$ 个高变异基因
    - 使用单细胞基础模型（scGPT或scVI），在细胞类型注释任务上fine-tune后提取紧凑的细胞系嵌入：$h_c = \text{scFM}(E_c)$
    - 细胞系嵌入对训练集和测试集的细胞系都可获取（因为RNA-seq数据独立于显微图像），从而可迁移到未见细胞系

3. **扰动-细胞系信息融合**：

    - 借鉴视觉-语言模型的范式，将细胞系特征转为 $m$ 个转录组token $T^c = [t_1^c, ..., t_m^c]$
    - 作为前缀拼接到图像patch token $T^p$ 前：$z = \text{ViT}([T^c, T^p])$
    - 最终取图像token $z_{m+1}, ..., z_{m+n}$ 的平均作为视觉表征用于下游分类

### 损失函数 / 训练策略

总目标 = 标准预训练损失（取决于基线方法） + 图正则化损失（$\mathcal{L}_{lap}$ 或 $\mathcal{L}_{con}$）

评测设置：
- 在RxRx1的HUVEC、RPE、HepG2三个细胞系上预训练
- 在U2OS（RxRx1）上one-shot fine-tune评测
- 在HRCE和VERO（RxRx19a，5通道，COVID-19数据集）上few-shot fine-tune评测
- ViT-S/16作为视觉编码器，8×A100 GPU训练

## 实验关键数据

### 主实验 (表格)

De novo细胞系上的扰动筛查性能（Top-1 / Top-5 准确率 %）：

| 预训练方法 | 配置 | U2OS Top-1 | U2OS Top-5 | HRCE Top-1 | HRCE Top-5 | VERO Top-1 | VERO Top-5 |
|------------|------|------------|------------|------------|------------|------------|------------|
| 无预训练 | baseline | 0.09 | 0.47 | 0.07 | 0.33 | 3.21 | 16.02 |
| WSL | baseline | 4.12 | 8.84 | 3.57 | 8.73 | 34.11 | 72.26 |
| WSL | **Ours** | **4.79** | **9.60** | **4.24** | **9.78** | **38.95** | **75.89** |
| SimCLR | baseline | 4.22 | 8.57 | 3.68 | 9.05 | 32.82 | 72.90 |
| SimCLR | **Ours** | **4.59** | **9.07** | **3.99** | **9.21** | **38.71** | **75.00** |
| MoCo v3 | baseline | 2.18 | 5.80 | 2.20 | 5.97 | 20.73 | 53.31 |
| MoCo v3 | **Ours** | **2.56** | **6.36** | **2.53** | **6.86** | **25.56** | **62.26** |
| MAE | baseline | 1.83 | 5.32 | 1.24 | 3.95 | 23.06 | 58.23 |
| MAE | **Ours** | **2.10** | **5.83** | **1.79** | **5.17** | **24.60** | **63.31** |

### 消融实验 (表格)

U2OS细胞系上细胞系特异性（CS）和扰动特异性（PS）表征的消融：

| CS | PS | WSL Top-1 | WSL Top-5 | SimCLR Top-1 | BYOL Top-1 | MoCo Top-1 | MAE Top-1 |
|----|-----|-----------|-----------|--------------|------------|------------|-----------|
| ✗ | ✗ | 4.12 | 8.84 | 4.22 | 3.61 | 2.18 | 1.83 |
| ✓ | ✗ | 4.64 | 9.48 | 4.39 | 3.71 | 2.45 | 1.95 |
| ✗ | ✓ | 4.25 | 9.09 | 4.48 | 3.97 | 2.33 | 1.98 |
| **✓** | **✓** | **4.79** | **9.60** | **4.59** | **3.80** | **2.56** | **2.10** |

scFM维度消融（WSL预训练，U2OS评测）：
- scVI: dim=128时最优，dim=512时性能下降
- scGPT: dim=256时最优，但相比scVI无显著优势
- 原始基因表达数据略优于baseline但不如scFM嵌入

### 关键发现

- 整合生物知识在所有5种预训练方法上都带来一致的提升，证明了方法的通用性
- U2OS上平均Top-1提升约20%（相对提升），VERO细胞系上提升更显著（WSL: 34.11→38.95）
- 扰动特异性表征（图正则化）在SimCLR、BYOL、MAE上的提升比细胞系特异性表征更稳定
- BYOL上添加细胞系特异性表征反而降低性能（3.61→3.71），但扰动关系图始终有正向效果
- 当训练/测试数据来自不同batch时（split condition 2），图正则化的效果更显著，验证了Proposition 3.1
- 图对比损失权重1e-5比1e-2效果更好，说明正则化不宜过强以免干扰主任务
- WSL在所有预训练方法中表现最好，因为其直接优化扰动分类目标

## 亮点与洞察

- **将de novo细胞系泛化作为显式研究问题**是一个有价值的贡献。现有方法通常在同一细胞系内评测，忽视了跨细胞系的迁移能力
- **解耦扰动特异性和细胞系特异性信息的思路新颖**：通过外部知识引导一个、通过scFM引导另一个，使两者显式分离
- **理论保证**（Proposition 3.1）为方法提供了数学基础——图正则化隐式减小类内距离
- 利用RNA-seq数据为未见细胞系提供特征表示，巧妙绕过了"测试时没有图像"的问题

## 局限与展望

- 绝对性能仍然很低：U2OS上最好也只有4.79% Top-1准确率（1138类分类），说明de novo细胞系泛化仍然极具挑战
- 预训练集仅有3个细胞系，限制了细胞系特异性表征的多样性，更大规模数据可能带来更好效果
- RxRx19a和RxRx1的成像协议不同（5通道 vs 6通道），增加了跨数据集迁移的难度，这是物理限制而非方法局限
- 图正则化的权重是需要调整的超参数，在不同split条件下表现不一
- 未探索更大的ViT模型（如ViT-B/L），受限于数据集规模

## 相关工作与启发

- Kraus et al.和RxRx系列数据集是该领域的重要基础
- 将蛋白质互作网络作为先验知识引入视觉模型的思路可推广到其他生物信息学+计算机视觉交叉问题
- 单细胞基础模型（scGPT/scVI）作为跨模态桥接的使用是该文的创新点，可启发更多模态融合方案
- 未来可结合Cell Painting等更丰富的表型标注和化学结构信息，进一步增强模型

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次系统性地引入生物知识图谱和单细胞基础模型增强显微图像预训练
- **实验充分度**: ⭐⭐⭐⭐ 5种预训练基线×3个de novo细胞系，消融实验细致（CS/PS分离、图损失分析、scFM维度）
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰、方法动机充分、理论分析严谨
- **价值**: ⭐⭐⭐⭐ 为药物发现中的跨细胞系泛化提供了有前景的方向，尽管绝对性能仍有较大提升空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] De novo generation of functional terpene synthases using TpsGPT](../../NeurIPS2025/computational_biology/de_novo_generation_of_functional_terpene_synthases_using_tpsgpt.md)
- [\[ICML 2025\] UniMoMo: Unified Generative Modeling of 3D Molecules for De Novo Binder Design](../../ICML2025/computational_biology/unimomo_unified_generative_modeling_of_3d_molecules_for_de_novo_binder_design.md)
- [\[ICML 2025\] Latent Imputation before Prediction: A New Computational Paradigm for De Novo Peptide Sequencing](../../ICML2025/computational_biology/latent_imputation_before_prediction_a_new_computational_paradigm_for_de_novo_pep.md)
- [\[ICML 2025\] PepTune: De Novo Generation of Therapeutic Peptides with Multi-Objective-Guided Discrete Diffusion](../../ICML2025/computational_biology/peptune_de_novo_generation_of_therapeutic_peptides_with_multi-objective-guided_d.md)
- [\[NeurIPS 2025\] One Small Step with Fingerprints, One Giant Leap for De Novo Molecule Generation from Mass Spectra](../../NeurIPS2025/computational_biology/one_small_step_with_fingerprints_one_giant_leap_for_de_novo_molecule_generation_.md)

</div>

<!-- RELATED:END -->
