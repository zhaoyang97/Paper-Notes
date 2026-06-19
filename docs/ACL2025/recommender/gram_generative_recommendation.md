---
title: >-
  [论文解读] GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion
description: >-
  [ACL 2025][推荐系统][生成式推荐] 提出 GRAM 生成式推荐框架，通过**语义到词汇翻译**将隐式物品层次/协同关系编码到 LLM 词汇空间，并用**多粒度迟融合**独立编码不同粒度提示再在解码端融合，在四个基准上 Recall@5 提升 11.5–16.0%、NDCG@5 提升 5.3–13.6%。
tags:
  - "ACL 2025"
  - "推荐系统"
  - "生成式推荐"
  - "多粒度融合"
  - "语义翻译"
  - "LLM推荐"
  - "协同过滤"
---

# GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion

**会议**: ACL 2025  
**arXiv**: [2506.01673](https://arxiv.org/abs/2506.01673)  
**代码**: [https://github.com/skleee/GRAM](https://github.com/skleee/GRAM)  
**领域**: 推荐系统  
**关键词**: 生成式推荐, 多粒度融合, 语义翻译, LLM推荐, 协同过滤  

## 一句话总结

提出 GRAM 生成式推荐框架，通过**语义到词汇翻译**将隐式物品层次/协同关系编码到 LLM 词汇空间，并用**多粒度迟融合**独立编码不同粒度提示再在解码端融合，在四个基准上 Recall@5 提升 11.5–16.0%、NDCG@5 提升 5.3–13.6%。

## 研究背景与动机

**领域现状**: 生成式推荐（Generative Recommendation）将推荐建模为 text-to-text 生成任务，利用 LLM 的预训练知识直接生成物品标识符。P5、TIGER、LC-Rec、IDGenRec 等方法已在多项基准上验证了该范式的有效性。

**已有方案的不足**: (a) **隐式物品关系缺失**——LLM 擅长理解文本语义，但物品间的层次分类关系（如"护肤品→面霜→保湿面霜"）和协同过滤关系（行为共现模式）难以直接用自然语言表达，现有方法要么完全忽略，要么使用 RQ-VAE 等量化方案引入 OOV token，与语言语义脱节；(b) **长文本信息瓶颈**——物品包含标题、品牌、描述、类别等丰富元数据，但将历史序列中所有物品信息拼接为单一提示会导致序列超长（Amazon Beauty 平均 1440 tokens），Transformer 的二次复杂度使其不可扩展，截断/关键词提取则不可避免地丢失信息。

**核心矛盾**: 推荐任务需要两类本质不同的知识——**协同过滤信号**（用户行为交互模式）和**语义信息**（物品文本描述），现有 LLM 框架缺乏将两者统一表示和高效利用的机制。

**本文切入点**: 提出"语义→词汇翻译"将推荐信号编码为 LLM 原生词汇 token（而非外部嵌入），并通过"多粒度迟融合"将不同信息源独立编码、解码时才聚合，兼顾信息完整性与计算效率。

**核心 idea**: 语义翻译编码物品关系 + 多粒度迟融合保留丰富语义 ＝ 更强的生成式推荐。

## 方法详解

### 整体框架

GRAM 基于 T5 encoder-decoder 架构，包含两个核心组件协同工作：**(1) 语义到词汇翻译（Semantic-to-Lexical Translation）**——训练前的预处理阶段，将物品的层次分类关系和协同过滤关系编码为 LLM 词汇空间内的文本 ID；**(2) 多粒度迟融合（Multi-granular Late Fusion）**——推理阶段，为粗粒度用户提示和细粒度物品提示使用独立编码器分别编码，在解码器 cross-attention 中才进行融合，避免早期拼接的信息损失和效率问题。

### 关键设计

1. **语义到词汇翻译（Semantic-to-Lexical Translation）**

    - **层次语义索引**: 对物品嵌入进行层次化 k-means 聚类（递归分裂至簇大小 < c 或达到最大深度 l），每个聚类用 TF-IDF 选出最具代表性的词汇 token 命名，生成形如 "soap-mild-mango" 的层次文本 ID。共享前缀的物品语义相关，自回归解码时从粗到细逐步生成
    - **协同语义文本化**: 利用预训练 SASRec 模型获取物品嵌入，对每个物品找到 top-k 最相似物品，将其层次 ID 拼接为额外属性 "similar items: soap-essence-argan, shampoo-essence-argan, …"，以文本形式注入协同过滤信号
    - **设计动机**: 使用 LLM 已有词汇而非 OOV token，保留预训练知识；层次结构确保语义一致的推荐

2. **多粒度迟融合（Multi-granular Late Fusion）**

    - **粗粒度用户提示**: 将用户历史序列的物品 ID（逆序排列避免截断近期项）拼接为 "What would the user purchase after {ID sequence}?"，捕获整体偏好和序列依赖
    - **细粒度物品提示**: 为历史中每个物品构建独立提示，包含 ID、协同语义、标题、品牌、描述等全部属性，保留详细信息
    - **迟融合解码**: 用 T5 encoder 分别编码各提示，添加位置嵌入后拼接为统一 key-value 矩阵，在解码器 cross-attention 中融合。信息链接（在物品提示中保留 ID）桥接粗/细粒度信息
    - **设计动机**: 早期融合（拼接输入文本）导致 O(n²) 复杂度和注意力稀释；迟融合让每个信息源被充分编码，仅在解码时交互

3. **训练与推理策略**

    - **训练**: 标准序列到序列交叉熵损失 + teacher forcing，目标是生成下一个物品的层次文本 ID
    - **推理**: 离线阶段预计算所有物品提示编码；在线阶段仅编码用户提示 + constrained beam search（前缀树确保生成合法 ID），beam size = 50
    - **设计动机**: 两阶段推理将细粒度物品编码的计算离线化，大幅降低在线延迟

## 实验关键数据

### 主实验（四个基准数据集 × 14个基线）

| 方法 | Beauty R@5 | Beauty N@5 | Toys R@5 | Toys N@5 | Sports R@5 | Sports N@5 | Yelp R@5 | Yelp N@5 |
|------|-----------|-----------|---------|---------|------------|------------|---------|---------|
| FDSA (best trad.) | 0.0570 | 0.0412 | 0.0619 | 0.0455 | 0.0283 | 0.0201 | 0.0331 | 0.0218 |
| LC-Rec | 0.0503 | 0.0352 | 0.0543 | 0.0385 | 0.0259 | 0.0175 | 0.0341 | 0.0235 |
| IDGenRec | 0.0463 | 0.0328 | 0.0462 | 0.0323 | 0.0273 | 0.0186 | 0.0310 | 0.0219 |
| **GRAM** | **0.0641** | **0.0451** | **0.0718** | **0.0516** | **0.0375** | **0.0256** | **0.0476** | **0.0326** |
| **提升** | +12.4% | +9.5% | +16.0% | +13.6% | +11.5% | +5.3% | +12.3% | +8.1% |

### 消融实验（Beauty / Toys，R@5 / N@5）

| 配置 | Beauty R@5 | Beauty N@5 | Toys R@5 | Toys N@5 |
|------|-----------|-----------|---------|---------|
| GRAM (full) | 0.0641 | 0.0451 | 0.0718 | 0.0516 |
| w/o 层次ID | 0.0605 | 0.0438 | 0.0630 | 0.0466 |
| w/o 协同语义 | 0.0567 | 0.0396 | 0.0589 | 0.0406 |
| w/o 用户提示 | 0.0634 | 0.0443 | 0.0709 | 0.0510 |
| w/o 物品提示 | 0.0582 | 0.0404 | 0.0574 | 0.0397 |
| w/o 信息链接 | 0.0628 | 0.0441 | 0.0702 | 0.0507 |
| w/o 位置嵌入 | 0.0563 | 0.0395 | 0.0665 | 0.0465 |

### ID 类型对比

| ID 类型 | Beauty R@5 | Beauty N@5 | Toys R@5 | Toys N@5 |
|---------|-----------|-----------|---------|---------|
| Hierarchical ID | **0.0641** | **0.0451** | **0.0718** | **0.0516** |
| RQ-VAE ID | 0.0605 | 0.0432 | 0.0662 | 0.0477 |
| Keyword ID | 0.0605 | 0.0438 | 0.0630 | 0.0466 |
| Category ID | 0.0512 | 0.0367 | 0.0465 | 0.0350 |
| Title ID | 0.0478 | 0.0342 | 0.0564 | 0.0412 |

### 关键发现

- **协同语义贡献最大**: 去掉协同过滤属性后 N@5 下降 10.8–27.2%，说明 LLM 确实缺乏行为共现信息
- **物品提示 > 用户提示**: 细粒度物品提示移除后 N@5 下降高达 30.1%，而用户提示移除仅下降 1.2%，细节信息对推荐更关键
- **尾部物品增益显著**: 对低频（尾部 80%）物品，GRAM 相比最佳基线 R@5 提升 42.6%、N@5 提升 47.8%——语义翻译有效弥补数据稀疏
- **层次 ID 优于 RQ-VAE**: 使用 LLM 原生词汇比 OOV token 更好利用预训练知识，N@5 提升 8.2%

## 亮点与洞察

- **"翻译"而非"嵌入"**——将推荐信号转化为 LLM 已知词汇 token，完全避免 OOV token 的语义鸿沟，是比 RQ-VAE/codebook 更优雅的方案
- **迟融合的工程智慧**——独立编码各物品再在解码端融合，既保留完整信息又避免超长序列的二次复杂度，且物品编码可离线预计算
- **两个创新正交互补**——语义翻译解决"编码什么信息"，迟融合解决"怎么高效利用"，缺一不可
- **对尾部物品的巨大改善**（R@5 +42.6%）揭示了生成式推荐在长尾场景的潜力

## 局限与展望

- 层次聚类依赖文本嵌入质量，对嵌入模型的选择敏感
- 仅在 T5-small 上验证，更大规模 LLM 上的效果未知
- 协同过滤模型（SASRec）需要预训练，增加 pipeline 复杂度
- 全部为离线评估，缺少在线 A/B 测试验证

## 评分

- **新颖性**: ⭐⭐⭐⭐ 语义翻译将推荐信号嵌入 LLM 词汇空间的思路新颖且自然，多粒度迟融合是对 FiD 在推荐领域的巧妙迁移
- **实验充分度**: ⭐⭐⭐⭐⭐ 四数据集 + 14个基线（6传统+8生成式）+ 完整消融 + ID类型对比 + 头/尾分析 + 超参敏感性
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰，Figure 1–5 的可视化有助理解，方法推导完整
- **价值**: ⭐⭐⭐⭐ 两个组件可独立复用，对生成式推荐领域有实质性推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](../../AAAI2026/recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[AAAI 2026\] From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization](../../AAAI2026/recommender/from_ids_to_semantics_a_generative_framework_for_cross-domain_recommendation_wit.md)
- [\[ACL 2026\] HSUGA: LLM-Enhanced Recommendation with Hierarchical Semantic Understanding and Group-Aware Alignment](../../ACL2026/recommender/hsuga_llm-enhanced_recommendation_with_hierarchical_semantic_understanding_and_g.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](../../AAAI2026/recommender/inductive_generative_recommendation_via_retrieval-based_speculation.md)
- [\[AAAI 2026\] When Top-ranked Recommendations Fail: Modeling Multi-Granular Negative Feedback for Explainable and Robust Video Recommendation](../../AAAI2026/recommender/when_top-ranked_recommendations_fail_modeling_multi-granular_negative_feedback_f.md)

</div>

<!-- RELATED:END -->
