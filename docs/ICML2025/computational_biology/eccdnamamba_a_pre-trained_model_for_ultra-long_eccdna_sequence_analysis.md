---
title: >-
  [论文解读] eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis
description: >-
  [ICML 2025][计算生物][染色体外环状DNA] eccDNAMamba 是首个面向环状DNA的双向状态空间编码器，结合BPE分词、环状数据增强和SpanBERT式预训练，在保持线性时间复杂度的同时支持高达200Kbp的超长eccDNA序列建模，在癌症分类和真实eccDNA识别任务上显著超越DNABERT-2、HyenaDNA和Caduceus。
tags:
  - "ICML 2025"
  - "计算生物"
  - "染色体外环状DNA"
  - "状态空间模型"
  - "Mamba-2"
  - "双向编码器"
  - "基因组预训练"
  - "超长序列建模"
  - "BPE分词"
---

# eccDNAMamba: A Pre-Trained Model for Ultra-Long eccDNA Sequence Analysis

**会议**: ICML 2025  
**arXiv**: [2506.18940](https://arxiv.org/abs/2506.18940)  
**代码**: [https://github.com/zzq1zh/GenAI-Lab](https://github.com/zzq1zh/GenAI-Lab)  
**领域**: 基因组学 / 计算生物学  
**关键词**: 染色体外环状DNA, 状态空间模型, Mamba-2, 双向编码器, 基因组预训练, 超长序列建模, BPE分词

## 一句话总结

eccDNAMamba 是首个面向环状DNA的双向状态空间编码器，结合BPE分词、环状数据增强和SpanBERT式预训练，在保持线性时间复杂度的同时支持高达200Kbp的超长eccDNA序列建模，在癌症分类和真实eccDNA识别任务上显著超越DNABERT-2、HyenaDNA和Caduceus。

## 研究背景与动机

染色体外环状DNA（eccDNA）是广泛存在于多种生物体中的功能性重要元素，在癌症基因组中尤为突出。eccDNA通常携带癌基因或远端调控元件，对肿瘤进化、治疗抗性和肿瘤内异质性产生重要影响。

然而，分析eccDNA全长序列面临两大核心挑战：

**环状拓扑的建模困难**：将环状序列在任意断点处线性化会产生人工边界，可能破坏具有生物学意义的头尾交互。

**超长序列的计算瓶颈**：许多eccDNA超过10,000bp，标准Transformer的二次复杂度使其无法处理如此长度的序列。

现有基因组基础模型存在明显局限：
- **HyenaDNA**：利用隐式卷积处理长序列，但为单向模型且不感知环状拓扑。
- **DNABERT-2**：使用BPE分词捕获序列motif，但保留线性输入假设和标准Transformer层。
- **Caduceus**：提供线性可扩展性，但基于单核苷酸分辨率，在超长eccDNA下游应用中扩展性受限。

核心动机：设计一个专为环状DNA定制的高效预训练模型，兼顾双向上下文建模、环状结构保持和线性时间复杂度。

## 方法详解

### 整体框架

eccDNAMamba的架构包含以下核心组件：

1. **BPE分词器**：对DNA序列进行字节对编码，将高频核苷酸模式编码为token
2. **环状数据增强**：将序列首部64个token追加到尾部以保持头尾依赖关系
3. **双向Mamba-2编码器**：前向和反向两个Mamba-2实例并行处理，通过共享MLP融合
4. **SpanMLM预训练目标**：掩码连续span进行重建

### 关键设计

#### 1. Byte-Pair Encoding (BPE) 分词

DNA序列为非结构化数据，采用BPE自适应合并高频相邻子串：

$(a^*, b^*) = \arg\max_{(a,b)} \text{freq}_{C_t}(a,b)$

BPE识别高频核苷酸模式并编码为token，使模型直接在motif级结构上操作，显著减少序列长度。预训练语料经BPE编码后，平均每个token对应5.16个碱基对。

#### 2. 环状数据增强

eccDNA为环状分子，但标准线性表示忽略了头尾依赖关系。将序列前$s=64$个token追加到序列末尾：

$\tilde{x} = (\texttt{[CLS]}, x_1, x_2, \ldots, x_L, x_1, x_2, \ldots, x_s)$

这使得模型能够学习前缀与尾部之间的长程依赖关系。

#### 3. 双向Mamba-2编码

Mamba-2原本为decoder-only结构，仅支持单向信息流。eccDNAMamba修改为双向感知：

- **前向编码**：$\vec{\mathbf{h}} = \overrightarrow{\text{Mamba}}(\overrightarrow{\tilde{\mathbf{x}}})$
- **反向编码**：$\overleftarrow{\mathbf{h}} = \overleftarrow{\text{Mamba}}(\overleftarrow{\tilde{\mathbf{x}}})$

两个方向的输出对齐到前向顺序后，通过共享MLP聚合生成最终嵌入。[CLS] token的嵌入用于下游分类任务。

#### 4. Span Masking 预训练

采用SpanBERT策略：随机选择连续span（平均约3个连续token），总计掩码15%的序列。其中80%替换为[MASK]，10%随机替换，10%不变。预训练损失为span-masked交叉熵：

$\mathcal{L}_{\text{SpanMLM}} = -\sum_{i \in \mathcal{M}} \log p_\theta(x_i \mid x_{\backslash \mathcal{M}})$

#### 5. Padding处理

Mamba-2不原生支持padding。采用两种策略：(1) 将padding token的嵌入归零并阻止训练更新；(2) 应用Transformer式注意力掩码抑制padding影响，并在每层将padding位置的隐藏状态和残差重置为零。

### 预训练设置

- 数据：120K条eccDNA序列（约1亿token），来自CircleBase（601K条人类eccDNA）和eccDNA Atlas（630K条动物eccDNA），过滤>10Kbp序列后剩余1,087,886条
- 模型：基于mamba2-130m配置初始化
- 训练：3个epoch，学习率$5 \times 10^{-4}$，线性warmup 6%步数，AdamW优化器，BF16混合精度，3×NVIDIA L40S

## 实验关键数据

### 主实验：癌症vs健康eccDNA分类

| 任务 | 模型 | F1 | Accuracy | Precision | Recall |
|------|------|-----|----------|-----------|--------|
| <10Kbp | **eccDNAMamba** | **0.8242** | **0.8242** | 0.8242 | **0.8242** |
| <10Kbp | DNABERT-2 | 0.8187 | 0.8187 | 0.8187 | 0.8187 |
| <10Kbp | HyenaDNA | 0.8105 | 0.8104 | 0.8105 | 0.8105 |
| <10Kbp | Caduceus | 0.8216 | 0.8220 | **0.8248** | 0.8220 |
| 10-200Kbp | **eccDNAMamba** | **0.8147** | **0.8175** | **0.8377** | **0.8174** |
| 10-200Kbp | DNABERT-2 | 0.5702 | 0.5725 | 0.5740 | 0.5725 |
| 10-200Kbp | HyenaDNA | 0.7261 | 0.7350 | 0.7699 | 0.7350 |
| 10-200Kbp | Caduceus | 0.7102 | 0.7125 | 0.7192 | 0.7125 |

关键发现：在超长序列（10-200Kbp）上，eccDNAMamba优势更加显著。DNABERT-2因截断为10Kbp而严重退化（F1仅0.57），而eccDNAMamba无截断处理，保持0.81+的稳定性能。

### 真实eccDNA vs 伪环状DNA

| 模型 | F1 | Accuracy | Precision | Recall |
|------|-----|----------|-----------|--------|
| **eccDNAMamba** | **0.7401** | **0.7407** | **0.7428** | **0.7407** |
| DeepCircle (fine-tuned) | 0.6712 | 0.6742 | 0.6808 | 0.6742 |
| DeepCircle (zero-shot) | 0.6363 | 0.6532 | 0.6883 | 0.6532 |

eccDNAMamba在F1上超过DeepCircle 6.9个百分点，证明真实eccDNA确实编码了可学习的非随机序列模式。

### 生物学分析

- 通过MEME-Suite STREME工作流分析，发现模型的癌症预测决策由**CG-rich序列motif**驱动（锌指转录因子结合位点特征）
- Tomtom比对CIS-BP 2.0数据库返回568个转录因子匹配，其中218个属于C2H2锌指家族
- **ZNF24**和**ZNF263**排名前列，这些基因与细胞增殖控制和癌基因调控相关
- 假阴性（FN）样本展示AT-rich模式，提示可能存在替代调控逻辑

## 亮点与洞察

1. **首个环状DNA专用预训练模型**：eccDNAMamba是首个为环状基因组设计的双向状态空间编码器，填补了领域空白
2. **极端数据效率**：仅用1亿token预训练即超越了使用350亿token（Caduceus）和2620亿token（DNABERT-2）的基础模型
3. **无性能退化的长序列扩展**：唯一在所有指标上处理超长序列时保持鲁棒的模型
4. **可解释的生物学发现**：CG-rich C2H2锌指motif作为主导序列特征，为癌症eccDNA的功能理解提供了新线索

## 局限性

1. 模型对CG-rich motif的依赖可能导致盲区（AT-rich假阴性）
2. 预训练仅基于序列信息，未整合调控注释等外部数据
3. 下游任务限于分类，尚未探索eccDNA来源预测等任务
4. 环状增强策略的token数（64）可能需要针对不同长度范围调优

## 相关工作

- **基因组基础模型**：DNABERT-2、HyenaDNA、Caduceus、Nucleotide Transformer
- **eccDNA建模工具**：eccDNA-Pipe（检测）、DeepCircle（CNN分类器）
- **状态空间模型**：Mamba、Mamba-2

## 评分

⭐⭐⭐⭐ — 首创性强，在环状DNA建模这一未被充分探索的领域提出了专用解决方案。实验 solid，尤其是超长序列上的优势令人印象深刻。生物学分析增加了工作的深度。但下游任务种类有限，且CG-rich偏差可能限制泛化能力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Graph Generative Pre-trained Transformer (G2PT)](graph_generative_pre-trained_transformer.md)
- [\[ICLR 2026\] A Joint Diffusion Model with Pre-Trained Priors for RNA Sequence-Structure Co-Design](../../ICLR2026/computational_biology/a_joint_diffusion_model_with_pre-trained_priors_for_rna_sequence-structure_co-de.md)
- [\[AAAI 2026\] TrinityDNA: A Bio-Inspired Foundational Model for Efficient Long-Sequence DNA Modeling](../../AAAI2026/computational_biology/trinitydna_a_bio-inspired_foundational_model_for_efficient_long-sequence_dna_mod.md)
- [\[CVPR 2026\] HINGE: Adapting a Pre-trained Single-Cell Foundation Model to Spatial Gene Expression Generation from Histology Images](../../CVPR2026/computational_biology/adapting_a_pre-trained_single-cell_foundation_model_to_spatial_gene_expression_g.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](../../NeurIPS2025/computational_biology/scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)

</div>

<!-- RELATED:END -->
