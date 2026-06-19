---
title: >-
  [论文解读] MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging
description: >-
  [AAAI2026 Oral][计算生物][DNA foundation model] 提出 MergeDNA，通过可微分 Token Merging 实现上下文感知的动态 DNA tokenization，结合层次化 autoencoder 和自适应 masked token modeling 预训练，380M 参数超越 1.3B GENERator。
tags:
  - "AAAI2026 Oral"
  - "计算生物"
  - "DNA foundation model"
  - "token merging"
  - "dynamic tokenization"
  - "genome modeling"
  - "masked language modeling"
---

# MergeDNA: Context-aware Genome Modeling with Dynamic Tokenization through Token Merging

**会议**: AAAI2026 Oral  
**arXiv**: [2511.14806](https://arxiv.org/abs/2511.14806)  
**代码**: 无  
**领域**: 计算生物  
**关键词**: DNA foundation model, token merging, dynamic tokenization, genome modeling, masked language modeling

## 一句话总结
提出 MergeDNA，通过可微分 Token Merging 实现上下文感知的动态 DNA tokenization，结合层次化 autoencoder 和自适应 masked token modeling 预训练，380M 参数超越 1.3B GENERator。

## 研究背景与动机

### 领域现状

**领域现状**：DNA foundation model 领域快速发展，DNABERT-2 使用 BPE tokenizer，HyenaDNA/Caduceus 使用 SSM 处理长序列，VQDNA 引入可学习的 VQ tokenizer，GENERator 扩展到 1.3B 参数规模。这些方法在 tokenization、长序列建模和预训练目标三个维度上各自优化，但缺乏统一框架。

### 现有痛点

**现有痛点**：(1) DNA 序列的信息密度分布极不均匀（仅约 2% 是编码序列 CDS，大量为非编码区域），但现有 tokenizer（固定 k-mer / BPE）对所有区域一视同仁；(2) DNA 没有天然的"词"边界——有意义的单元可能是 3 bases（codon）、6-10 bases（转录因子结合位点）或更长，固定粒度的 tokenization 必然丢失信息；(3) DNA 序列极长（数万到数百万 bases），需要同时捕获短程 motif 和长程依赖。

### 核心矛盾

**核心矛盾**：信息密集区域需要细粒度 token 保留细节，重复/低信息密度区域应合并以节省计算和扩大感受野。固定粒度的 tokenization 无法同时满足这两个需求——fine-grained 全局 token 计算代价大，coarse-grained 全局 token 丢失编码区域细节。

### 解决思路

**本文目标**：设计一个端到端可学习的基因组建模框架，同时解决动态 tokenization 和信息密度自适应预训练。**切入角度**：将 ViT 领域的 Token Merging（ToMe）思路迁移到 DNA 序列，通过可微分的 merge 操作自动学习上下文感知的 token 粒度。**核心idea**：用 local-window token merging 实现动态压缩，配合自适应 masked token modeling（按信息密度加权 mask 概率），在一个统一框架中解决 tokenization + 建模 + 预训练目标三个问题。

## 方法详解

### 整体框架
MergeDNA 采用层次化 autoencoder 架构，包含四个模块：(1) Local Encoder 作为可学习 tokenizer，通过多层 local-window self-attention + 可微分 token merging 将 bases 合并为变长 tokens；(2) Latent Encoder 用全注意力 Transformer 捕获全局依赖；(3) Latent Decoder 对称映射回 token 空间；(4) Local Decoder 通过 token unmerging 恢复原始长度并重建。

### 关键设计

1. **Local-window Token Merging**:

    - 功能：在 Local Encoder 中实现上下文感知的动态 tokenization
    - 核心思路：每层先做 local-window self-attention（窗口大小 16），然后用 lightweight grouping embedding 计算相邻 token 的相似度，选取 top-$r_l$ 对进行 soft merging（加权平均保证可微分）。多层堆叠后逐步压缩序列长度至 $L \approx N/2$。source matrix $\mathcal{S} \in \{0,1\}^{L \times N}$ 记录 merge 关系用于 unmerge 恢复
    - 设计动机：soft merging 保证端到端可微分训练，local window 限制 merge 范围在相邻 bases 间（符合 DNA 局部语义连续性），多层渐进压缩比一步压缩更稳定

2. **自适应 Masked Token Modeling (AMTM)**:

    - 功能：按信息密度加权的预训练目标
    - 核心思路：利用 Latent Encoder 的 global token merging 结果识别重要 token（merge group 大小反映信息密度），按重要性采样 mask $K$ 个 token 进行预测。masking 概率与 merge group 大小成反比——越重要（越难被合并）的 token 越可能被 mask
    - 设计动机：信息密集区域（如 CDS、转录因子结合位点）被 merge 的概率低，因此它们的 merge group 更小。AMTM 确保预训练重点关注这些高信息密度区域

3. **Merged Token Reconstruction (MTR)**:

    - 功能：端到端重建损失，驱动 tokenizer 学习有意义的 merge 策略
    - 核心思路：重建损失 $\mathcal{L}_{MTR} = -\frac{1}{N}\sum_{i=1}^{N}\log P(\hat{X}_i | X_i; \theta)$，训练时对压缩率做高斯采样（$L \in [0.4N, 0.6N]$），使模型对不同压缩率鲁棒
    - 设计动机：压缩率随机化是一种数据增强策略，防止模型过拟合到单一压缩比

### 损失函数
总损失：$\mathcal{L}_{total} = \mathcal{L}_{MTR}(\theta) + \lambda \mathcal{L}_{MTR}(\theta \setminus \{\phi\}) + \mathcal{L}_{AMTM}(\theta)$，其中 $\lambda = 0.25$，第二项固定 tokenizer 参数仅更新 decoder。

## 实验关键数据

### 主实验

在 GUE Benchmark（8 tasks）和 NT Benchmark（18 tasks）上评测。

| 方法 | Params | Enhancers (3 tasks) | Species (2 tasks) | Regulatory (3 tasks) | Avg (8 tasks) |
|------|--------|:---:|:---:|:---:|:---:|
| NT-500M | 500M | 84.56% | 96.64% | 89.05% | 89.26% |
| GENERator | 1.3B | 84.87% | 96.95% | 90.30% | 90.71% |
| **MergeDNA** | **380M** | **85.11%** | 96.84% | **90.66%** | **90.87%** |

NT Benchmark（18 tasks）：MergeDNA 平均 MCC 78.39%，超越 MxDNA (78.14%) 和所有其他 baseline。

### 消融实验

| 配置 | Avg MCC (8 tasks) | 说明 |
|------|:---:|------|
| Full MergeDNA | 90.87% | 完整模型 |
| w/o AMTM | 89.91% | 去自适应 mask，掉 0.96% |
| w/o Token Merging | 89.12% | 固定 tokenization，掉 1.75% |
| 固定压缩率 (50%) | 90.23% | 去随机化，掉 0.64% |

### 关键发现
- 380M 参数下超越 1.3B 的 GENERator，证明动态 tokenization 的参数效率优势
- Token merging 贡献最大（去除后掉 1.75%），说明上下文感知的动态 tokenization 确实优于固定策略
- 在 Splice Site 任务上尤其突出（Donor: 98.93%，Acceptor: 98.67%），说明 merge 策略能自适应识别剪接位点的边界信息
- 跨模态迁移：在 RNA 和 protein 下游任务上也表现出良好泛化

## 亮点与洞察
- **统一三个维度**：首次将动态 tokenization、长序列建模和自适应预训练目标整合在一个端到端可学习框架中
- **信息密度自适应**：tokenizer 自动对编码区分配更细粒度 token、对重复区域合并，与 DNA 的生物学特性高度吻合
- **参数高效**：380M 超越 1.3B 模型，说明 "聪明的 tokenization" 比暴力堆参数更有效
- Token merging 从 ViT 迁移到 DNA 的成功案例，启发性地表明 merge 策略可扩展到任何长序列模态（音频、时序信号）

## 局限与展望
- 预训练序列长度仅 4096，对真实基因组级别（数百万 bases）仍显不足
- Token merging 的 local window 固定为 16，可能限制对更长 motif 的发现
- 缺少与 Evo2 等超大规模模型的直接对比
- 下游任务主要是分类，缺少生成任务（如序列设计）的验证

## 相关工作与启发
- **vs DNABERT-2 (BPE tokenizer)**: 固定 BPE 不考虑上下文和信息密度；MergeDNA 的动态 tokenizer 平均提升 3.5%+
- **vs VQDNA (VQ tokenizer)**: 同为可学习 tokenizer，但 VQDNA 是离散 VQ，MergeDNA 是连续 soft merge，梯度传播更顺畅
- **vs HyenaDNA/Caduceus (SSM)**: MergeDNA 用层次化 Transformer 替代 SSM，在效率和性能间取得更好平衡
- 自适应 masking 策略（按信息密度调整 mask 概率）可作为通用预训练技巧推广

## 评分
- 新颖性: ⭐⭐⭐⭐ Token merging 在 DNA 上的首次系统应用，框架设计完整
- 实验充分度: ⭐⭐⭐⭐ 双 benchmark + 消融 + 跨模态迁移，但缺少超大规模对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 为 DNA foundation model 的 tokenization 提供新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Apo2Mol: 3D Molecule Generation via Dynamic Pocket-Aware Diffusion Models](apo2mol_3d_molecule_generation_via_dynamic_pocket-aware_diff.md)
- [\[ICML 2025\] Global Context-aware Representation Learning for Spatially Resolved Transcriptomics](../../ICML2025/computational_biology/global_context-aware_representation_learning_for_spatially_resolved_transcriptom.md)
- [\[ICLR 2026\] AntigenLM: Structure-Aware DNA Language Modeling for Influenza](../../ICLR2026/computational_biology/antigenlm_structure-aware_dna_language_modeling_for_influenza.md)
- [\[ICML 2026\] Rethinking Genomic Modeling Through Optical Character Recognition](../../ICML2026/computational_biology/rethinking_genomic_modeling_through_optical_character_recognition.md)
- [\[ICML 2026\] DNAChunker: Learnable Tokenization for DNA Language Models](../../ICML2026/computational_biology/dnachunker_learnable_tokenization_for_dna_language_models.md)

</div>

<!-- RELATED:END -->
