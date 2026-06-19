---
title: >-
  [论文解读] Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning
description: >-
  [ICML 2025][预训练][data mixing] 提出 Chameleon 框架，利用 kernel ridge leverage scores（KRLS）在代理模型的嵌入空间中量化各训练域的重要性，以仅 DoReMi 1/10 的计算成本达到同等或更优的数据混合效果，且支持新域引入时无需重训代理模型、统一处理预训练和微调场景。
tags:
  - "ICML 2025"
  - "预训练"
  - "data mixing"
  - "domain reweighting"
  - "kernel ridge leverage scores"
  - "pretraining"
  - "finetuning"
---

# Chameleon: A Flexible Data-mixing Framework for Language Model Pretraining and Finetuning

**会议**: ICML 2025  
**arXiv**: [2505.24844](https://arxiv.org/abs/2505.24844)  
**代码**: [GitHub - Chameleon](https://github.com/LIONS-EPFL/Chameleon)  
**领域**: LLM预训练 / 数据混合  
**关键词**: data mixing, domain reweighting, kernel ridge leverage scores, pretraining, finetuning

## 一句话总结

提出 Chameleon 框架，利用 kernel ridge leverage scores（KRLS）在代理模型的嵌入空间中量化各训练域的重要性，以仅 DoReMi 1/10 的计算成本达到同等或更优的数据混合效果，且支持新域引入时无需重训代理模型、统一处理预训练和微调场景。

## 研究背景与动机

**领域现状**：预训练 LLM 的数据来源多样（网页文本、学术论文、代码、书籍等），不同域的混合比例显著影响模型泛化性能。近年来，域重加权方法（domain reweighting）成为数据混合的核心研究方向，DoReMi 和 DoGE 是代表性工作，都使用小型代理模型来推导域权重再训练大模型。

**现有痛点**：DoReMi 需要训练两个模型（参考模型+代理模型），用 Group DRO 优化域权重；DoGE 跟踪域特定梯度，计算成本高。两种方法的核心问题是：(1) 计算昂贵——代理模型训练本身就相当于一次完整的 LM 训练；(2) 域变化敏感——当数据集新增或删除域时，必须从头重训代理模型；(3) 不支持微调——这些方法只为预训练设计，微调场景的域重加权是空白。

**核心矛盾**：要获得好的域权重就需要代理模型迭代优化，但迭代优化本身就是主要的计算瓶颈。同时，域权重与代理模型的训练过程耦合，导致无法在域变化时复用已有计算。

**本文目标**：设计一种将域权重计算与代理模型训练解耦的方法，满足三个目标——(1) 低计算成本获得高质量域权重；(2) 新域引入时免重训；(3) 统一处理预训练和微调。

**切入角度**：从数据本身的内在特性出发（data-centric），而非从模型优化过程出发。核心观察：如果我们已经有了代理模型，可以直接从数据的嵌入表示中提取域间关系，无需通过训练过程间接推导。

**核心 idea**：用代理模型嵌入空间中的 kernel ridge leverage scores 直接量化各域的独特性与代表性，预训练时用逆 KRLS（强调共性域），微调时用 KRLS（强调独特域）。

## 方法详解

### 整体框架

Chameleon 的 pipeline：(1) 用均匀权重训练一个小型代理模型；(2) 提取各域数据的中间层嵌入，对每个域取均值得到域嵌入向量 $x_i$；(3) 构建域亲和矩阵 $\Omega_\mathcal{D} = XX^\top$，其中 $X = [x_1, ..., x_k]^\top$；(4) 计算 KRLS 得分 $S_\lambda(D_i) = [\Omega_\mathcal{D}(\Omega_\mathcal{D} + k\lambda I)^{-1}]_{ii}$；(5) 预训练用逆分数 $\alpha^{PT} \propto \text{softmax}(S_\lambda^{-1})$，微调用正分数 $\alpha^{FT} \propto \text{softmax}(S_\lambda)$。

### 关键设计

1. **域嵌入与亲和矩阵**:

    - 功能：将每个训练域表示为一个向量，并量化域间的语义关系
    - 核心思路：对域 $D_i$ 中的数据，用代理模型的第 $L$ 层隐藏状态取均值得到域嵌入 $x_i = \frac{1}{|D_i|}\sum_{a \in D_i} h_{\theta_p}^{(L)}(a)$。域亲和矩阵 $\Omega_\mathcal{D} = XX^\top$ 的每个元素 $\Omega_{ij} = x_i^\top x_j$ 衡量域 $i$ 和域 $j$ 在嵌入空间中的相似度。UMAP 可视化显示，语义相似的域（如 CC 和 C4）在嵌入空间中聚集，独特域（如 ArXiv、Github）则分离
    - 设计动机：域嵌入是最自然的域粒度表示——比样本级嵌入计算量小几个量级（只需一次前向传播+均值），同时保留了域间关系的核心信息

2. **Kernel Ridge Leverage Scores（KRLS）量化域独特性**:

    - 功能：为每个域计算一个标量分数，表示该域在整体嵌入空间中有多"独特"
    - 核心思路：KRLS 定义为 $S_\lambda(D_i) = [\Omega_\mathcal{D}(\Omega_\mathcal{D} + k\lambda I)^{-1}]_{ii}$，即核岭回归 hat matrix 的对角元素。高 KRLS 的域在嵌入空间中占据独特位置，不能被其他域线性组合近似；低 KRLS 的域有高度冗余性，可被其他域充分表示。正则化参数 $\lambda$ 控制独特性判定的严格程度
    - 设计动机：KRLS 在统计学中有坚实的理论基础——其逆与 Christoffel 函数成正比，后者精确刻画数据分布在特征空间中的局部密度。这为域权重提供了理论支撑而非启发式规则

3. **预训练用逆 KRLS、微调用 KRLS 的对偶策略**:

    - 功能：根据训练阶段的不同目标自适应调整域权重方向
    - 核心思路：预训练目标是学习通用知识——应该多采样那些被广泛共享的、高密度区域的域（低 KRLS = 高代表性），所以用逆 KRLS 作为权重 $\alpha^{PT} = \text{softmax}(S_\lambda^{-1})$。微调目标是学习域特定知识——应该多采样那些独特的、差异化的域（高 KRLS = 高独特性），所以直接用 KRLS 作为权重 $\alpha^{FT} = \text{softmax}(S_\lambda)$
    - 设计动机：预训练和微调对同一数据分布有截然不同的需求，但 DoReMi/DoGE 只为预训练设计。KRLS 的对偶性（正/逆分别衡量独特性/代表性）自然地给出了统一框架

### 损失函数 / 训练策略

代理模型用标准语言模型交叉熵损失训练，均匀域权重。KRLS 计算不涉及额外训练，仅需矩阵运算，复杂度 $O(k^3)$（$k$ 为域数量，通常 <20）。整个流程不对代理模型的训练过程做任何修改。

## 实验关键数据

### 主实验表格（SlimPajama，684M 参数，预训练域困惑度）

| 域 | Uniform | DoReMi | DoGE | **Chameleon** | RegMix |
|----|---------|--------|------|-------------|--------|
| ArXiv | 8.16 | 9.16 | 9.07 | **8.31** | 11.35 |
| Book | 42.55 | 46.48 | 40.30 | **39.23** | 41.52 |
| CC | 45.26 | **40.62** | 38.99 | 40.11 | 37.32 |
| C4 | 49.00 | 43.92 | 40.65 | 42.59 | 43.85 |
| Github | **3.99** | 4.10 | 4.09 | 4.20 | 4.99 |
| StackExchange | 7.99 | 8.35 | **7.39** | 7.94 | 10.63 |
| Wikipedia | **12.42** | 10.78 | 15.74 | 13.90 | 20.88 |
| **平均 PPL↓** | 24.20 | 23.34 | 22.32 | **22.31** | 24.36 |
| **计算成本 (FLOPs)** | 0 | 1.34×10¹⁸(10×) | 6.68×10¹⁷(5×) | **1.36×10¹⁷(1×)** | 1.20×10¹⁸(9×) |

### 下游推理任务准确率（13 个 benchmark 平均）

| 方法 | ARC-E | COPA | HellaSwag | Lambada | PiQA | WinoGrande | **平均↑** |
|------|-------|------|-----------|---------|------|------------|---------|
| Uniform | 36.8 | 55.7 | 26.5 | 13.5 | 59.2 | 50.5 | 37.9 |
| DoReMi | 37.6 | 59.3 | 27.0 | 13.6 | 59.5 | 51.3 | 38.4 |
| DoGE | 38.0 | 62.3 | 27.2 | 14.7 | 60.0 | 52.0 | 39.4 |
| **Chameleon** | 37.8 | 61.9 | 27.1 | **15.1** | **60.5** | **52.1** | **39.6** |
| RegMix* | 39.1 | 63.0 | 27.0 | 16.5 | 57.6 | 50.9 | 39.3 |

*RegMix 利用了下游任务信息，其他方法均不使用。

### 关键发现

- **计算成本降低 5-10 倍**：Chameleon 仅需 1.36×10¹⁷ FLOPs（约 DoReMi 的 1/10），却在平均 PPL 上达到最优（22.31 vs DoGE 22.32）
- **下游推理准确率最高**：在 13 个 benchmark 的平均上（39.6%），Chameleon 超过了不使用下游信息的所有方法，甚至略超利用下游信息的 RegMix（39.3%）
- **域变化免重训**：新增域时只需计算新域的嵌入并更新亲和矩阵，无需重训代理模型。实验显示即使域数量翻倍，Chameleon 在 1% 的重训成本下仍优于需要完全重训的 baseline
- **微调场景一致改善**：在微调域重加权实验中，KRLS 权重在所有域上均改善了测试困惑度，验证了对偶策略的有效性

## 亮点与洞察

- **KRLS 在数据混合中的应用**是核心方法论创新——将统计学中成熟的 leverage scores 工具引入 LLM 训练，为域权重提供了有理论支撑的计算方法，而非启发式搜索
- **"无需代理模型训练信号"是对 DoReMi 的根本性简化**：域权重完全由数据的嵌入几何关系决定，不依赖训练过程中的损失、梯度等动态信号
- **预训练/微调的对偶策略**巧妙利用了 KRLS 正/逆分数的互补含义：逆分数→代表性→通用知识→预训练；正分数→独特性→专项知识→微调

## 局限性

- 嵌入层 $L$ 的选择对域嵌入质量有影响，目前通过实验选取中间层，缺乏理论指导
- 仅使用线性核（$\kappa(x_i, x_j) = x_i^\top x_j$），虽然模型本身提供非线性，但不排除非线性核能捕捉更复杂的域间关系
- 在超大规模域数量（$k > 100$）下的可扩展性未验证，虽然 $O(k^3)$ 复杂度在 $k < 20$ 时可忽略
- 域嵌入取均值可能丢失域内分布的多样性信息——如果一个域包含多个子簇，均值可能不具代表性

## 相关工作与启发

- **vs DoReMi**：DoReMi 用 Group DRO 在代理训练中动态调整域权重，域权重在训练过程中波动大；Chameleon 一次性从嵌入几何中计算固定权重，稳定且计算高效
- **vs DoGE**：DoGE 跟踪域特定梯度来捕捉域对训练的贡献，计算成本高；Chameleon 用 KRLS 从数据表示直接度量域独特性，无需梯度计算
- **vs Data Pruning 方法（如 DSIR）**：数据剪枝在样本级别工作，选择与目标分布匹配的样本；Chameleon 在域级别工作，调整域的采样概率。两者互补，可联合使用

## 评分

- 新颖性: ⭐⭐⭐⭐ KRLS 在数据混合中的首次应用，理论动机清晰
- 实验充分度: ⭐⭐⭐⭐ 预训练/微调/域变化三种场景全覆盖，与主流方法对比充分
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，方法推导连贯，图表信息量大
- 价值: ⭐⭐⭐⭐⭐ 将数据混合的计算成本降低一个量级，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] How to Synthesize Text Data without Model Collapse?](how_to_synthesize_text_data_without_model_collapse.md)
- [\[ICML 2025\] Metadata Conditioning Accelerates Language Model Pre-training](metadata_conditioning_accelerates_language_model_pre-training.md)
- [\[ACL 2025\] Data Caricatures: On the Representation of African American Language in Pretraining Corpora](../../ACL2025/llm_pretraining/data_caricatures_on_the_representation_of_african_american_language_in_pretraini.md)
- [\[ICML 2026\] AC-ODM: Actor–Critic Online Data Mixing for Sample-Efficient LLM Pretraining](../../ICML2026/llm_pretraining/ac-odm_actor--critic_online_data_mixing_for_sample-efficient_llm_pretraining.md)
- [\[NeurIPS 2025\] Language Model Behavioral Phases are Consistent Across Architecture, Training Data, and Scale](../../NeurIPS2025/llm_pretraining/language_model_behavioral_phases_are_consistent_across_archi.md)

</div>

<!-- RELATED:END -->
