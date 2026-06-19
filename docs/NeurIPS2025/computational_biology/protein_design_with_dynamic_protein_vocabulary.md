---
title: >-
  [论文解读] Protein Design with Dynamic Protein Vocabulary
description: >-
  [NeurIPS 2025 Spotlight][计算生物][蛋白质设计] 提出 ProDVa 方法，将天然蛋白质片段作为"动态词汇"引入生成式蛋白质设计，通过文本编码器+蛋白质语言模型+片段编码器的三组件架构，利用不到 0.04% 的训练数据即可设计出功能对齐且结构可折叠的蛋白质序列，在 pLDDT>70 比例上超越 SOTA 模型 Pinal 达 7.38%。
tags:
  - "NeurIPS 2025 Spotlight"
  - "计算生物"
  - "蛋白质设计"
  - "动态词汇表"
  - "片段检索"
  - "可折叠性"
  - "功能对齐"
---

# Protein Design with Dynamic Protein Vocabulary

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.18966](https://arxiv.org/abs/2505.18966)  
**代码**: [GitHub](https://github.com/sornkL/ProDVa)  
**领域**: 计算生物  
**关键词**: 蛋白质设计, 动态词汇表, 片段检索, 可折叠性, 功能对齐

## 一句话总结

提出 ProDVa 方法，将天然蛋白质片段作为"动态词汇"引入生成式蛋白质设计，通过文本编码器+蛋白质语言模型+片段编码器的三组件架构，利用不到 0.04% 的训练数据即可设计出功能对齐且结构可折叠的蛋白质序列，在 pLDDT>70 比例上超越 SOTA 模型 Pinal 达 7.38%。

## 研究背景与动机

蛋白质设计是生物技术的核心挑战，目标是在庞大的序列空间中设计具有特定功能的新蛋白质。近年来深度生成模型（如 ProteinDT、Pinal、PAAG）已经能够基于文本描述进行功能导向的蛋白质设计，但面临一个根本问题：**设计出的蛋白质难以折叠成稳定的三维结构**。

经典的蛋白质设计方法（如合理设计、定向进化）之所以成功，是因为它们利用了天然蛋白质的已知结构作为设计基础。这启发了一个自然的问题：**能否通过利用天然蛋白质片段（如 motif、功能位点等）来增强生成模型的可折叠性？**

作者进行了一个关键的实验验证：即使**随机地**将天然蛋白质片段插入序列中（Random+ 方法），也能显著扩展生成蛋白质的分布多样性并提升可折叠性。在 UMAP 可视化中，Random+ 生成的蛋白质比纯 Random 更接近天然蛋白质的分布。这个实验发现奠定了整个方法的基础——片段本身携带了结构先验。

现有方法的核心不足：
- ProteinDT、PAAG 等在 pLDDT 和 PAE 指标上表现很差（接近随机）
- Pinal 虽然表现较好但需要 17.6 亿训练对，且缺乏开源训练脚本
- ESM3 虽然 PPL 最低但序列重复模式严重，可折叠性差

## 方法详解

### 整体框架

ProDVa 由三个核心组件构成：文本语言模型 (TextLM)、蛋白质语言模型 (PLM) 和片段编码器 (FE)。训练时使用 InterPro 数据库自动获取蛋白质序列的功能注释（片段类型+描述），推理时根据输入文本动态检索相关片段作为候选词汇。

### 关键设计

1. **动态蛋白质词汇表**：将蛋白质序列分解为氨基酸 token 和天然蛋白质片段两种单元。静态词汇 $V_{\text{tokens}}$ 使用 BPE 分词器分割氨基酸序列，动态词汇来自 InterPro 标注的片段（Domain、Family、Active Site 等 8 类）。片段通过 Fragment Encoder 映射到与 PLM 相同的嵌入空间。关键创新在于生成时每一步可以选择输出一个 token **或**一个完整片段：

$$p(x_i = k | \mathbf{H}_{\text{pre}}) = \frac{\exp(\mathbf{H}_{\text{pre}} \mathbf{W}_{\text{out}}^{(k)})}{\sum_{k' \in V_{\text{tokens}} \cup S} \exp(\mathbf{H}_{\text{pre}} \mathbf{W}_{\text{out}}^{(k')})}$$

2. **功能注释学习**：引入两个辅助损失来充分利用片段的功能注释：

    - **类型损失 $\mathcal{L}_{\text{TYPE}}$**：对片段进行类型分类（8 类），使用加权交叉熵处理类别不平衡
    - **描述损失 $\mathcal{L}_{\text{DESC}}$**：使用 InfoNCE 对比学习损失对齐片段表示和描述文本表示（正样本为配对的片段-描述，负样本为同批次中其他对）

$$\mathcal{L}_{\text{DESC}} = -\frac{1}{\sum_i |S_i|} \sum_i \sum_j \log \frac{\exp(\text{sim}(\mathbf{u}_{ij}, \mathbf{v}_{ij})/\tau)}{\sum_k \sum_l \exp(\text{sim}(\mathbf{u}_{ij}, \mathbf{v}_{kl})/\tau)}$$

3. **推理时片段检索**：给定输入的功能描述文本 $t$，使用 PubMedBERT 嵌入检索 Top-K 个最相似的文本-蛋白质对（默认 K=16），用 InterPro 提取这 K 个蛋白质的片段作为候选集。使用 Top-k 采样解码增强多样性。

### 损失函数 / 训练策略

总训练目标：$\mathcal{L} = \mathcal{L}_{\text{NTP}} + \alpha \mathcal{L}_{\text{TYPE}} + \beta \mathcal{L}_{\text{DESC}}$

- TextLM 用 GPT-2 初始化，PLM 和 FE 用 ProtGPT2 初始化
- 分类头和描述投影层仅训练时使用，推理时丢弃
- 检索后端使用 Faiss 支持的 txtai 框架

## 实验关键数据

### 主实验 1：CAMEO 子集（功能关键词生成）

| 模型 | #训练对 | pLDDT↑ | %>70↑ | PAE↓ | %<10↑ | ProTrek↑ | Keyword Recovery↑ |
|------|--------|--------|-------|------|-------|---------|-------------------|
| Random+(E) | - | 62.38 | 32.65% | 17.23 | 9.28% | 3.29% | 0.00% |
| ProteinDT | 541K | 38.70 | 0.20% | 26.25 | 0.00% | 7.43% | 0.05% |
| Pinal | 1.76B | 66.50 | 47.21% | 14.57 | 33.53% | 14.57% | 30.46% |
| ESM3 | 539M | 59.79 | 31.49% | 17.40 | 21.37% | 3.76% | 5.49% |
| **ProDVa** | **392K** | **75.88** | **77.00%** | **6.39** | **83.88%** | 14.43% | 30.34% |

### 主实验 2：Mol-Instructions（自然语言描述生成）

| 模型 | pLDDT↑ | %>70↑ | PAE↓ | %<10↑ | ProTrek↑ | EvoLlama↑ |
|------|--------|-------|------|-------|---------|----------|
| Pinal | 75.25 | 68.97% | 10.96 | 58.44% | 17.50% | 53.42% |
| Chroma | 59.18 | 20.17% | 15.03 | 28.62% | 2.10% | 40.10% |
| **ProDVa** | **76.86** | **76.35%** | **8.66** | **68.06%** | **17.40%** | 51.10% |

### 消融实验（Vanilla Multimodal Baseline 对比）

| 配置 | pLDDT | %>70 | PAE | %<10 | ProTrek Score | 说明 |
|------|-------|------|-----|------|--------------|------|
| Vanilla (无片段) | ~72 | ~63% | ~11 | ~58% | ~10% | 仅用GPT-2+ProtGPT2 |
| ProDVa | 76.86 | 76.35% | 8.66 | 68.06% | 17.40% | 完整模型 |

ProDVa 在 pLDDT 提升 4.63%，PAE 降低 2.71%，pLDDT>70 比例提升 13.66%，证实了片段和功能注释的重要性。

### 关键发现

- 即使随机插入片段（Random+）也能改善可折叠性，验证了片段作为结构先验的有效性
- ProDVa 仅用 0.02%~0.04% 的 Pinal 训练数据即超越其可折叠性指标
- Top-K 检索参数分析显示 K=16 为最优，过多检索反而降低功能对齐性
- ProDVa 同样适用于无条件蛋白质生成，超越 Pinal 22.76%（pLDDT>70）

## 亮点与洞察

- 核心 insight 极为简洁有力：天然蛋白质片段本身就携带折叠先验，利用它们作为生成单元比逐氨基酸生成更高效
- 动态词汇的概念巧妙地将 NLP 中的 copy/retrieval 机制迁移到蛋白质设计
- 数据效率极高——0.04% 训练数据超越 SOTA，具有很强的实用价值

## 局限与展望

- 依赖 InterPro 数据库进行片段标注，对新发现蛋白质的覆盖可能不足
- 语言对齐指标略低于 Pinal（尤其 EvoLlama Score），文本理解能力仍有提升空间
- 未包含湿实验验证，可折叠性评估完全依赖 ESMFold 预测
- K 值选择需要仔细调参，不同任务可能需要不同最优值

## 相关工作与启发

- 与 Pinal 的对比最为核心：Pinal 先生成结构再设计序列（structure-then-sequence），ProDVa 直接在序列空间操作但融入结构片段先验
- 动态词汇的思路可以推广到其他生物序列设计（如 RNA、多肽）
- 片段检索的范式类似 RAG，有潜力结合更大规模的蛋白质结构数据库

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 动态词汇+片段先验的 idea 新颖且验证充分
- 实验充分度: ⭐⭐⭐⭐ 两个benchmark+多baseline+消融+无条件生成，全面
- 写作质量: ⭐⭐⭐⭐ 动机引入自然，图表清晰
- 价值: ⭐⭐⭐⭐ 数据效率高，方法通用性好，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](../../ACL2025/computational_biology/concept_bottleneck_language_models_for_protein_design.md)
- [\[NeurIPS 2025\] PROSPERO: Active Learning for Robust Protein Design Beyond Wild-Type Neighborhood](prospero_active_learning_for_robust_protein_design_beyond_wild-type_neighborhood.md)
- [\[ICML 2025\] Flexibility-conditioned Protein Structure Design with Flow Matching](../../ICML2025/computational_biology/flexibility-conditioned_protein_structure_design_with_flow_matching.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](../../ICML2025/computational_biology/elucidating_the_design_space_of_multimodal_protein_language_models.md)
- [\[ACL 2026\] ProtoCycle: Reflective Tool-Augmented Planning for Text-Guided Protein Design](../../ACL2026/computational_biology/protocycle_reflective_tool-augmented_planning_for_text-guided_protein_design.md)

</div>

<!-- RELATED:END -->
