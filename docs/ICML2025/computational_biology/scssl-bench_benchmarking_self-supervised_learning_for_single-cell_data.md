---
title: >-
  [论文解读] scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data
description: >-
  [ICML2025 Spotlight][计算生物][自监督学习] 提出 scSSL-Bench，一个系统性 benchmark，在 9 个单细胞数据集上评估 19 种自监督学习方法在批次校正、细胞类型注释和缺失模态预测三个下游任务上的表现，揭示了通用 SSL 方法与领域专用方法之间的任务特异性权衡。
tags:
  - "ICML2025 Spotlight"
  - "计算生物"
  - "自监督学习"
  - "single-cell genomics"
  - "benchmark"
  - "batch correction"
  - "对比学习"
---

# scSSL-Bench: Benchmarking Self-Supervised Learning for Single-Cell Data

**会议**: ICML2025 Spotlight  
**arXiv**: [2506.10031](https://arxiv.org/abs/2506.10031)  
**代码**: [BoevaLab/scSSL-Bench](https://github.com/BoevaLab/scSSL-Bench)  
**领域**: 计算生物  
**关键词**: self-supervised learning, single-cell genomics, benchmark, batch correction, contrastive learning

## 一句话总结

提出 scSSL-Bench，一个系统性 benchmark，在 9 个单细胞数据集上评估 19 种自监督学习方法在批次校正、细胞类型注释和缺失模态预测三个下游任务上的表现，揭示了通用 SSL 方法与领域专用方法之间的任务特异性权衡。

## 研究背景与动机

**问题场景**：单细胞 RNA 测序（scRNA-seq）和多组学测序技术可以在单细胞分辨率下刻画细胞异质性，但产生的数据面临两个核心挑战：(1) 数据维度极高（数万基因 × 数十万细胞），(2) 不同实验批次引入的系统性技术偏差（batch effects）会掩盖真实的生物信号。

**SSL 在单细胞数据中的应用现状**：自监督学习在 CV 和 NLP 领域取得了巨大成功（SimCLR、MoCo、BYOL 等），已有多项工作将其迁移到单细胞数据（CLEAR、CLAIRE、Concerto 等）。然而，现有研究缺乏：

- 通用 SSL 方法与单细胞专用方法的系统性对比
- 超参数、数据增强策略、正则化技术在单细胞场景下的消融研究
- 单细胞基础模型（scGPT、Geneformer）与对比学习方法的公平比较

**三个核心研究问题**：

- **RQ1**：单细胞专用 SSL 方法是否一定优于通用 SSL 方法？单模态 vs. 多模态数据上表现有何差异？
- **RQ2**：超参数（嵌入维度、投影维度）与增强策略如何影响通用 SSL 在单细胞数据上的性能？
- **RQ3**：为图像数据设计的 domain-specific batch normalization（DSBN）和多模态集成技术是否对单细胞数据同样有益？

## 方法详解

### Benchmark 整体设计

scSSL-Bench 的流水线为：输入 cell-by-gene 计数矩阵 → 选择 19 种方法之一进行训练（对比方法使用数据增强创建正负样本对） → 在三个下游任务上评估学得的表征。

### 评估的 19 种方法（四大类）

1. **通用 SSL（7 种）**：SimCLR、MoCo、SimSiam、NNCLR、BYOL、VICReg、BarlowTwins — 均从 CV 领域迁移，使用共享权重编码器将两个增强视图编码后通过投影器进行对比学习
2. **单细胞对比方法（4 种）**：CLEAR（InfoNCE + 高斯噪声/掩码/交叉增强）、CLAIRE（基于 MNN 的增强 + MoCo 架构）、Concerto（教师-学生蒸馏 + dropout 增强）、scCLIP（类 CLIP 多模态对比）
3. **单细胞生成方法（7 种）**：scVI（VAE + 零膨胀负二项分布）、totalVI（多模态 VAE）、scGPT / Geneformer / scBERT（基础模型）、scButterfly（双 VAE）、scTEL（Transformer+LSTM）
4. **基线（2 种）**：SCDC、PCA

### 数据集（9 个）

- **单模态（7 个）**：PBMC、Pancreas、Immune Cell Atlas、MCA、HIC、Lung、Tabula Sapiens（均为 scRNA-seq）
- **多模态（2 个）**：PBMC-M、BMMC（CITE-seq，包含 RNA + 蛋白质/ADT）

### 下游任务与评估指标

**批次校正**：使用 scIB 工具评估，综合得分为

$$Total = 0.6 \times Bio + 0.4 \times Batch$$

其中 Bio 衡量细胞嵌入与真实细胞类型的一致性，Batch 衡量批次效应去除程度。

**细胞类型注释**：将数据集划分为 reference（训练集）和 query（测试集，包含最多 3 个未见批次），用 KNN 分类器在 reference 嵌入上训练后对 query 进行标注，指标为 macro-F1 和 accuracy。

**缺失模态预测**：在多模态数据上，给定 query 的 RNA 表达预测其蛋白质表达，通过 kNN 探测取最近邻平均值，指标为 Pearson 相关系数。

### 增强策略评估

来自 CLEAR 的四种增强（各 50% 概率施加）：

- **Masking**：随机将 20% 基因置零
- **Gaussian Noise**：对 80% 基因加均值 0、标准差 0.2 的高斯噪声
- **InnerSwap**：在同一细胞内交换 10% 基因的表达值
- **CrossOver**：与另一随机细胞交叉变异 25% 基因表达值

来自 CLAIRE 的邻域增强：基于互最近邻（MNN）或 batch-balanced KNN（BBKNN）图构建正样本对。

### 其他消融设计

- **Domain-Specific Batch Normalization (DSBN)**：为每个实验批次分配独立的 batch normalization 层
- **投影器是否保留**：推理阶段是否保留投影头（projector）
- **嵌入维度与投影维度**：系统评估不同维度设置对性能的影响

## 实验关键数据

### 批次校正（Table 1，5 个数据集）

| 方法 | PBMC-M Total | BMMC Total | PBMC Total | Pancreas Total | Immune Total |
|------|:---:|:---:|:---:|:---:|:---:|
| SimCLR | 0.700 | 0.767 | 0.447 | 0.721 | 0.635 |
| VICReg | 0.651 | **0.761** | 0.490 | **0.733** | 0.644 |
| BYOL | **0.754** | 0.722 | 0.379 | 0.610 | 0.479 |
| CLAIRE | — | — | **0.774** | 0.732 | 0.539 |
| scVI | — | — | 见论文 | 见论文 | 见论文 |
| scGPT (finetuned) | — | — | 0.770 | 0.662 | **0.781** |
| scGPT (zero-shot) | — | — | 0.451 | 0.351 | 0.435 |
| Geneformer (finetuned) | — | — | 0.199 | 0.177 | 0.114 |
| PCA (baseline) | 见论文 | 见论文 | 见论文 | 见论文 | 见论文 |

**关键发现**：

- **单模态批次校正**：专用方法 scVI、CLAIRE 和微调后的 scGPT 表现最优
- **多模态批次校正**：通用 SSL（SimCLR、VICReg、BYOL）反而显著优于领域专用方法，说明当前缺乏有效的多模态单细胞集成框架
- scGPT zero-shot 远不如 finetuned 版本，Geneformer finetuned 表现很差（Total < 0.2）
- Concerto 在大多数单模态数据集上 Bio 得分极低（PBMC 仅 0.055）

### 细胞类型注释

- VICReg 和 SimCLR 在细胞类型注释任务上全面优于所有单细胞专用方法
- 通用 SSL 方法在此任务上的优势稳定且显著

### 缺失模态预测

- 在多模态数据的缺失模态预测中，通用 SSL 方法同样优于专用方法

### 增强策略消融

- **随机掩码（Masking）是最有效的增强技术**，在所有任务上均优于领域特定增强（CrossOver、InnerSwap）
- MNN 和 BBKNN 邻域增强对批次校正有一定帮助，但不如简单的 masking 通用

### 设计决策消融（RQ3）

- **DSBN 无益**：domain-specific batch normalization 不改善甚至可能损害单细胞数据的表现
- **投影器应丢弃**：推理阶段保留投影头不提升性能，与 CV 领域的经验一致
- **嵌入维度**：中等到较大的嵌入维度（如 128-256）一致地带来更好的结果

## 亮点与洞察

1. **"通用优于专用"的反直觉发现**：在细胞类型注释、多模态集成和缺失模态预测上，从 CV 直接迁移的通用 SSL 方法（VICReg、SimCLR）竟然全面击败了专门为单细胞设计的方法，这对领域专用方法的设计理念提出了挑战
2. **Masking 的普适优势**：简单的随机掩码增强超越了所有精心设计的生物学先验增强，暗示单细胞数据的增强策略可能不需要过多领域知识
3. **基础模型的局限**：scGPT 仅在微调后才有竞争力，Geneformer 即便微调也表现极差，说明当前单细胞基础模型在批次校正场景下还不成熟
4. **多模态集成的空白**：论文明确指出当前缺乏有效的单细胞多模态集成框架，这是一个重要的未来研究方向
5. **标准化评估平台的贡献**：19 种方法 × 9 个数据集 × 3 个任务的系统评估，提供了社区急需的公平比较基础

## 局限与展望

1. **多模态数据集仅 2 个**：PBMC-M 和 BMMC 都是 CITE-seq 数据，缺少 10x Multiome（RNA+ATAC）等其他多模态技术的评估，结论的泛化性有限
2. **增强策略搜索不够深入**：仅评估了 CLEAR 和 CLAIRE 提出的增强，未探索更多可能性（如基于 GAN 的增强、mixup 等）
3. **缺乏计算效率分析**：19 种方法的训练时间、内存占用、可扩展性未系统报告，实际选择方法时缺少成本维度
4. **生成方法使用了标签信息**：scVI 等生成方法在训练时利用了 batch/cell type 标注，与纯自监督方法的对比不完全公平（论文有提及但未深入讨论）
5. **下游任务局限**：未评估基因调控网络推断、轨迹推断、扰动预测等其他重要的单细胞分析任务
6. **超参数调优范围**：通用 SSL 方法使用了统一的超参数设置，可能未充分发挥某些方法的潜力

## 相关工作与启发

- **Richter et al., 2024**：前期工作比较了 MAE、BYOL 和 BarlowTwins 在单细胞上的表现，但缺乏与专用方法的对比 — 本文填补了这一空白
- **scIB (Luecken et al., 2022)**：单细胞集成 benchmark 框架，本文的评估指标体系建立在其基础上
- **CLIP (Radford et al., 2021)**：scCLIP 借鉴了 CLIP 的多模态对比思想，但在单细胞场景下表现不如通用 SSL 方法
- **启发**：通用方法在新领域的强劲表现提示我们，领域专用设计不一定总是有益的；简单但有效的增强（masking）可能是最佳起点

## 评分

- 新颖性: ⭐⭐⭐ — 方法层面无新提出，核心贡献是系统性 benchmark 和实验发现
- 实验充分度: ⭐⭐⭐⭐⭐ — 19 方法 × 9 数据集 × 3 任务 + 大量消融实验，非常全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，实验设计合理，结论有数据支撑
- 价值: ⭐⭐⭐⭐ — 为单细胞 SSL 社区提供了标准化评估平台和可操作的实践建议，揭示了多个重要的反直觉发现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Gene Incremental Learning for Single-Cell Transcriptomics](../../AAAI2026/computational_biology/gene_incremental_learning_for_single-cell_transcriptomics.md)
- [\[ICML 2026\] Learning the Neighborhood: Contrast-Free Multimodal Self-Supervised Molecular Graph Pretraining](../../ICML2026/computational_biology/learning_the_neighborhood_contrast-free_multimodal_self-supervised_molecular_gra.md)
- [\[NeurIPS 2025\] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](../../NeurIPS2025/computational_biology/scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)
- [\[ICML 2025\] DeepSeq: High-Throughput Single-Cell RNA Sequencing Data Labeling via Web Search-Augmented Agentic Generative AI Foundation Models](deepseq_high-throughput_single-cell_rna_sequencing_data_labeling_via_web_search-.md)
- [\[ICML 2025\] Protein Structure Tokenization: Benchmarking and New Recipe](protein_structure_tokenization_benchmarking_and_new_recipe.md)

</div>

<!-- RELATED:END -->
