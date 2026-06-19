---
title: >-
  [论文解读] JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model
description: >-
  [NeurIPS 2025][计算生物][DNA基础模型] 提出JanusDNA，首个双向DNA基础模型，结合Mamba-Attention-MoE混合架构和Janus Modeling训练范式，以自回归的训练效率实现双向理解，在多个基因组基准上达到SOTA。 领域现状：大语言模型正被应用于DNA序列建模…
tags:
  - "NeurIPS 2025"
  - "计算生物"
  - "DNA基础模型"
  - "双向建模"
  - "注意力机制"
  - "Mixture-of-Experts"
  - "基因组学"
---

# JanusDNA: A Powerful Bi-directional Hybrid DNA Foundation Model

**会议**: NeurIPS 2025  
**arXiv**: [2505.17257](https://arxiv.org/abs/2505.17257)  
**代码**: [GitHub](https://github.com/Qihao-Duan/JanusDNA)  
**领域**: 计算生物  
**关键词**: DNA基础模型, 双向建模, Mamba-Attention, Mixture-of-Experts, 基因组学

## 一句话总结

提出JanusDNA，首个双向DNA基础模型，结合Mamba-Attention-MoE混合架构和Janus Modeling训练范式，以自回归的训练效率实现双向理解，在多个基因组基准上达到SOTA。

## 研究背景与动机

**领域现状**：大语言模型正被应用于DNA序列建模，但直接迁移面临独特挑战——需要处理超长序列（>10k碱基对）的长程依赖且需要双向理解。

**现有痛点**：
   - **序列长度与分辨率矛盾**：注意力机制难以处理长序列，k-mer分词扩大窗口但牺牲分辨率（丢失SNP信息）
   - **单向理解**：解码器模型（HyenaDNA, Evo）仅支持单向，而DNA许多调控元件（如双向启动子）需要双向
   - **训练低效**：MLM（BERT式）仅15%token参与损失计算，对长序列训练效率极低

**核心矛盾**：双向理解能力（MLM）与训练效率（自回归）之间的权衡。

**本文目标**：构建一个高效的双向DNA基础模型，兼具长序列处理能力和训练效率。

**切入角度**：设计新的预训练范式（Janus Modeling）让所有token都参与损失计算（如自回归），同时保持双向理解（如MLM）。

**核心 idea**：通过双向独立编码+精心设计的注意力掩码融合，实现全token损失计算的双向预训练。

## 方法详解

### 整体框架

JanusDNA包含三个核心组件：(1) **Janus Modeling**——高效双向预训练方法；(2) **Mamba-Attention-MoE混合架构**；(3) **反向互补（RC）处理策略**。正向和反向序列分别通过独立的Mamba+MoE栈编码，再通过FlexAttention融合，实现无信息泄露的双向预测。

### 关键设计

1. **Janus Modeling（双向高效训练）**：

    - 功能：让每个token基于完整双向上下文被预测，且所有token参与损失
    - 为什么：MLM仅15%token计算损失效率低；自回归效率高但单向
    - 怎么做：
        - 前向编码：$H_t^F = \text{ForwardEncoder}(x_1, ..., x_t)$
        - 后向编码：$H_t^B = \text{BackwardEncoder}(x_T, ..., x_t)$
        - 双向融合：通过精心设计的注意力掩码 $\mathcal{M}_{ij}$ 确保预测 $x_t$ 时仅使用 $H_k^F (k<t)$ 和 $H_j^B (j>t)$
    - 训练目标：$\mathcal{L}_{bidirectional} = -\sum_{t=1}^{T} \log P(x_t | x_1,...,x_{t-1}, x_{t+1},...,x_T)$
    - 区别：比MLM快约2倍（稀疏掩码），学习效率显著更高

2. **混合架构（Mamba-Attention-MoE）**：

    - 功能：结合SSM的长序列效率、注意力的全局理解和MoE的稀疏扩容
    - 为什么：纯注意力无法处理百万级碱基对，纯SSM缺乏全局融合
    - 怎么做：
        - Mamba层高效编码局部上下文
        - MoE层按比例替代FFN层扩大模型容量（稀疏激活）
        - FlexAttention层实现双向融合
    - MoE辅助损失：$\mathcal{L}_{total} = \alpha \cdot N \cdot \sum_{i=1}^N f_i \cdot P_i$ 确保专家均衡使用
    - 区别：可在单张80GB GPU上处理100万碱基对

3. **反向互补（RC）处理**：

    - 功能：并行处理DNA正链和反向互补链
    - 为什么：DNA双链结构包含等价信息，非回文基序需要同时识别正反两种形式
    - 怎么做：正链和RC链分别输入同一模型，输出表示池化后合并

4. **注意力掩码设计（FlexAttention Mask）**：

    - 功能：控制2T长度输入序列中的注意力信息流
    - 为什么：必须防止位置t处预测时泄露自身信息
    - 怎么做：四条规则控制前向段内、后向段内、前向-后向交叉注意力

### 损失函数 / 训练策略

- 主损失：双向预测损失 $\mathcal{L}_{bidirectional}$（所有token参与）
- MoE辅助损失：确保专家负载均衡
- 预训练数据：人类参考基因组HG38，单核苷酸分辨率分词
- 上下文长度131,072（可扩展至1M）

## 实验关键数据

### 主实验

**Genomic Benchmark (8个任务, Top-1 Accuracy, 5-fold CV)**：

| 模型 | 激活参数 | Mouse Enhancers | Coding vs Inter. | Human Regulatory | Human NonTATA |
|------|---------|-----------------|-------------------|------------------|---------------|
| HyenaDNA | 436k | 0.780 | 0.904 | 0.869 | 0.944 |
| Caduceus-PS | 470k | **0.793** | 0.910 | 0.873 | 0.945 |
| **JanusDNA** | 426k | 0.770 | 0.912 | **0.877** | **0.957** |

**Nucleotide Transformer Benchmark (18个任务) - 选取关键组蛋白标记**：

| 模型 | 激活参数 | H3 | H3k14ac | H3k36me3 | H3k4me3 |
|------|---------|-----|---------|----------|---------|
| Enformer | 252M | 0.719 | 0.288 | 0.344 | 0.158 |
| NT-v2 | 500M | 0.784 | 0.551 | 0.625 | 0.410 |
| Caduceus-PH | 1.9M | 0.815 | 0.631 | 0.601 | 0.544 |
| **JanusDNA** | **2M** | **0.835** | **0.729** | **0.702** | **0.688** |

**DNALongBench eQTL任务 (AUROC, 序列长度450k)**：

| 模型 | Artery Tibial | Muscle Skeletal | Nerve Tibial | Whole Blood |
|------|--------------|-----------------|--------------|-------------|
| Enformer(252M) | 0.741 | 0.621 | 0.683 | 0.689 |
| Caduceus-PH(7.7M) | 0.690 | 0.789 | 0.842 | 0.769 |
| **JanusDNA(7.7M)** | **0.852** | **0.864** | **0.914** | **0.821** |

### 消融实验

**Janus Modeling vs Masked Modeling效率对比**（10k步训练，last-token预测准确率）：
- Janus Modeling在所有隐藏维度（32/64/128）上显著优于Masked Modeling
- Janus训练速度：约27分钟/1000步，约为Masked Modeling的2倍快
- 隐藏维度128时，Janus在5k步达到的精度，Masked需要10k步

### 关键发现

- JanusDNA在18个NT任务中12个达到SOTA，超越250倍参数量的模型
- 在eQTL长程任务上显著超越专家模型Enformer
- Janus Modeling比MLM训练效率提升约2倍
- 单张80GB GPU处理100万碱基对，实用性强
- MoE层有效扩大模型容量而不显著增加计算成本

## 亮点与洞察

- **"Janus双面神"的绝妙比喻**：两个方向的独立编码+融合，完美对应双链DNA的生物学本质
- **打破参数量-性能的线性关系**：2M参数超越500M+模型
- **训练范式创新**：同时解决MLM效率低和自回归单向性两个根本问题
- **FlexAttention掩码设计精巧**：在2T长度输入上实现无信息泄露的全token双向预测

## 局限与展望

- 仅在人类参考基因组上预训练，缺乏跨物种和基因组变异数据
- 未集成表观遗传信息（染色质可及性、组蛋白修饰等多模态数据）
- 长序列的计算资源需求仍然较高
- 未来可探索CTCF介导的染色质环等功能特征的建模

## 相关工作与启发

- 与Caduceus的双向SSM思路不同：Caduceus通过双向Mamba实现，JanusDNA通过Janus Modeling+融合注意力实现
- Mamba + Attention混合架构的趋势在NLP（Jamba等）和基因组学中同时出现
- MoE稀疏扩容策略对超长序列模型特别有价值
- 单核苷酸分辨率对SNP研究至关重要

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ Janus Modeling训练范式极具创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 35个任务、三大基准、完整消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表精美
- 价值: ⭐⭐⭐⭐⭐ 为DNA基础模型确立新范式，实际影响深远

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] SPACE: Your Genomic Profile Predictor is a Powerful DNA Foundation Model](../../ICML2025/computational_biology/space_your_genomic_profile_predictor_is_a_powerful_dna_foundation_model.md)
- [\[NeurIPS 2025\] Iterative Foundation Model Fine-Tuning on Multiple Rewards](iterative_foundation_model_fine-tuning_on_multiple_rewards.md)
- [\[ICML 2025\] SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics](../../ICML2025/computational_biology/stofm_a_multi-scale_foundation_model_for_spatial_transcriptomics.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)
- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)

</div>

<!-- RELATED:END -->
