---
title: >-
  [论文解读] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation
description: >-
  [语义分割] 提出 ActionPiece，首个上下文感知的动作序列分词器，将用户行为序列建模为"特征集合的序列"，通过类 BPE 的合并策略在集合内部和相邻集合之间发现高频特征模式，使同一动作在不同上下文中被分词为不同 token，显著提升生成式推荐性能。 生成式推荐（Generative Recommendation…
tags:
  - "语义分割"
---

# ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation

## 一句话总结

提出 ActionPiece，首个上下文感知的动作序列分词器，将用户行为序列建模为"特征集合的序列"，通过类 BPE 的合并策略在集合内部和相邻集合之间发现高频特征模式，使同一动作在不同上下文中被分词为不同 token，显著提升生成式推荐性能。

## 研究背景与动机

生成式推荐（Generative Recommendation, GR）是一种新兴范式：将用户的历史交互动作分词为离散 token，然后用自回归模型生成 token 序列作为推荐结果。这种方式共享紧凑词表，在可扩展性和内存效率上有优势。

现有 GR 方法（TIGER、P5-CID、VQ-Rec、HSTU 等）的核心问题在于**分词时完全独立处理每个动作**——同一物品无论出现在什么序列中都会被映射为相同的 token 模式。这忽略了一个重要事实：同一动作在不同上下文中可能承载不同含义。例如购买同一件T恤，在运动装备购买序列和日常休闲序列中的语义是不同的。

作者类比 NLP 中分词技术的发展：早期词级分词（context-independent）→ 现代子词分词如 BPE（context-aware），指出推荐系统的动作分词仍停留在"词级"阶段，需要向上下文感知演进。但推荐场景与文本有本质差异——文本是一维字符序列，而物品的特征构成无序集合，需要专门设计适配"集合序列"的分词算法。

## 方法详解

### 整体框架

ActionPiece 的流程分为三步：

1. **词表构建**：在训练语料（用户动作序列语料库）上，从初始词表（每个唯一特征对应一个 token）出发，迭代地合并高频共现 token 对，直到达到目标词表大小 $Q$
2. **分词/分段**：利用集合排列正则化（Set Permutation Regularization, SPR），将动作序列的每个特征集合随机排列后展平，用传统 BPE 分段方法切分，产生多种语义等价的 token 序列
3. **模型训练与推理**：用 Transformer encoder-decoder 自回归生成下一个动作的 token 序列；训练时每个 epoch 用不同排列做数据增强，推理时用多个排列做集成

### 关键设计一：加权共现计数（Weighted Co-occurrence Counting）

与标准 BPE 不同，ActionPiece 处理的是"集合的序列"而非一维序列。token 对有两种共现类型：(1) 同一集合内的两个 token；(2) 相邻集合中的两个 token。为了统一处理，作者将集合随机排列展平为一维序列，计算两个 token 相邻的期望概率作为权重。

对于同一集合 $\mathcal{A}_i$ 中的两个 token $c_1, c_2$：

$$P(c_1, c_2) = \frac{2}{|\mathcal{A}_i|}$$

对于相邻集合 $\mathcal{A}_i$ 和 $\mathcal{A}_{i+1}$ 中的两个 token $c_1 \in \mathcal{A}_i, c_3 \in \mathcal{A}_{i+1}$：

$$P(c_1, c_3) = \frac{1}{|\mathcal{A}_i| \times |\mathcal{A}_{i+1}|}$$

这种加权方式使得集合较小时内部 token 对的权重更高（因为特征间关联更紧密），而跨集合的权重随集合大小增大而衰减。通过遍历语料库累加所有出现位置的权重，得到每个 token 对的总共现分数，选择最高分的 token 对进行合并。

### 关键设计二：中间节点与语料更新机制

合并 token 对时，如果两个 token 来自同一集合，直接替换即可。但如果来自两个相邻集合，产生的新 token 跨越了动作边界。作者引入**中间节点（Intermediate Node）**解决这一问题：

- 用双向链表维护每个动作序列，每个节点代表一个 token 集合
- 合并跨集合 token 时，在两个"动作节点"之间插入一个中间节点存放新 token
- 中间节点最多包含一个 token，计算共现时视为大小为 1 的集合
- 合并动作节点与中间节点的 token 时，新 token 留在中间节点中

这种设计保证了任意两个动作节点之间最多一个中间节点，第四种 token 类型（跨动作特征组合）得以自然表达。

### 关键设计三：集合排列正则化（SPR）

朴素分段策略（按合并优先级贪心匹配）会导致词表中大量 token 未被使用（利用率仅 56.89%）。SPR 的核心思想：

- 训练时，每个 epoch 对每个动作的特征集合生成随机排列，展平后用标准 BPE 分段
- 不同排列产生语义等价但 token 不同的序列，天然构成数据增强
- 推理时，生成 $q$ 个不同排列的分段结果，分别通过模型得到排序列表，平均得分做集成

SPR 将 token 利用率从 56.89% 提升到 87.01%+，并且训练效率几乎不受影响（排列操作在 CPU 上异步完成）。

### 高效实现

朴素实现的时间复杂度为 $O(QNLm^2)$。通过构建倒排索引（token 对 → 包含它们的链表位置）和全局堆（lazy-update 策略），将复杂度降至 $O(\log Q \log H \cdot NLm^2)$，其中 $H = O(NLm)$ 为堆的最大规模。

## 实验关键数据

### 主实验：与基线方法对比（Amazon Reviews 数据集）

| 数据集 | 指标 | SASRec | VQ-Rec | TIGER | HSTU | SPM-SID | **ActionPiece** | 提升 |
|--------|------|--------|--------|-------|------|---------|----------------|------|
| Sports | R@5 | 0.0233 | 0.0181 | 0.0264 | 0.0258 | 0.0280 | **0.0316** | +12.86% |
| Sports | N@10 | 0.0192 | 0.0154 | 0.0225 | 0.0215 | 0.0234 | **0.0264** | +12.82% |
| Beauty | R@5 | 0.0387 | 0.0434 | 0.0454 | 0.0469 | 0.0475 | **0.0511** | +7.58% |
| Beauty | N@10 | 0.0318 | 0.0372 | 0.0384 | 0.0389 | 0.0399 | **0.0424** | +6.00% |
| CDs | R@5 | 0.0351 | 0.0314 | 0.0492 | 0.0417 | 0.0509 | **0.0544** | +6.88% |
| CDs | N@10 | 0.0263 | 0.0264 | 0.0411 | 0.0346 | 0.0424 | **0.0451** | +6.37% |

### 消融实验（NDCG@10）

| 变体 | Sports | Beauty | CDs |
|------|--------|--------|-----|
| w/o tokenization（直接用原始特征） | 0.0215 | 0.0389 | 0.0346 |
| w/o context-aware（仅集合内合并） | 0.0258 | 0.0416 | 0.0429 |
| w/o weighted counting（等权计数） | 0.0257 | 0.0412 | 0.0435 |
| SPR 仅用于推理 | 0.0192 | 0.0316 | 0.0329 |
| SPR 仅用于训练 | 0.0244 | 0.0387 | 0.0422 |
| TIGER + SPR（无 ActionPiece 词表） | 0.0202 | 0.0330 | 0.0351 |
| **ActionPiece (40k)** | **0.0264** | **0.0424** | **0.0451** |

## 关键发现

1. **上下文感知是核心**：去掉跨集合合并（w/o context-aware）后性能显著下降，证明跨动作特征模式的捕获是关键贡献
2. **SPR 不可单独使用**：将 SPR 应用于 TIGER（上下文无关分词）反而降低性能，因为它破坏了 RQ-VAE 语义 ID 的内部顺序但未引入新的语义信息
3. **词表大小的 trade-off**：增大词表可提升性能并缩短 token 序列长度，但也增加参数量；TIGER 简单增大词表并不能获得提升，说明问题不在词表大小而在分词质量
4. **推理集成的边际效益递减**：集成数 $q$ 从 1 增到 5 时性能提升明显，5 到 7 时提升放缓

## 亮点与洞察

- **优雅的类比迁移**：将 NLP 中 BPE 的成功经验迁移到推荐系统，但不是简单套用——针对"集合序列"这一独特数据结构做了深入适配（加权计数、中间节点、SPR），体现了对问题本质的理解
- **一石二鸟的 SPR**：集合排列正则化同时解决了 token 利用率低和推理集成两个问题，设计简洁高效
- **跨动作 token 的洞察**：合并后的 token 可以横跨相邻动作的特征，这在概念上等价于捕获用户行为的"转移模式"（如从T恤到袜子的品牌一致性），为分词赋予了序列级语义
- **Google DeepMind 出品，代码开源**，工程实现有高效的倒排索引和 lazy-update 堆，考虑了实际部署的效率

## 局限性

1. **仅在 Amazon Reviews 小规模数据上验证**：三个数据集最大也只有 75k 用户、64k 物品，未在工业级大规模场景验证
2. **特征选择依赖人工**：需要预定义物品的离散特征集合，对连续特征需要离散化，特征工程的质量直接影响分词效果
3. **推理开销随集成数线性增长**：虽然可并行，但 $q$ 次前向传播的计算成本仍然较高
4. **未与 LLM-based 推荐方法对比**：如 TALLRec 等基于大语言模型的推荐方法未纳入比较

## 相关工作与启发

- **TIGER**（Rajput et al., 2023）：用 RQ-VAE 生成层次化语义 ID，是主要对比基线；ActionPiece 的核心区别在于上下文感知
- **SPM-SID**（Singh et al., 2024）：将 SentencePiece 应用于语义 ID 模式的合并，最强基线但仍是上下文无关的
- **HSTU**（Zhai et al., 2024）：直接用原始特征作为 token，不做分词；ActionPiece 的 w/o tokenization 消融验证了分词的必要性
- **BPE for NLP**（Sennrich et al., 2016）：ActionPiece 的直接灵感来源，但从一维序列扩展到集合序列是非平凡的

**启发**：这种"从 NLP 分词技术到结构化序列分词"的迁移思路可推广到其他领域——音频建模（帧特征集合序列）、时间序列预测（多变量集合序列）、序列决策等。上下文感知的 token 化思想也可能启发其他领域的表示学习。

## 评分

- 新颖性：⭐⭐⭐⭐ — 首个上下文感知动作分词器，类比清晰但技术适配非平凡
- 技术深度：⭐⭐⭐⭐ — 加权计数、中间节点、SPR 三个设计环环相扣，高效实现有工程深度
- 实验充分性：⭐⭐⭐⭐ — 消融全面，分析深入（token 利用率、集成效果、case study），但数据集规模偏小
- 实用价值：⭐⭐⭐⭐ — 代码开源，方法通用，但工业级验证缺失
- 总体推荐：⭐⭐⭐⭐ — 将 NLP 分词技术优雅迁移到推荐系统的高质量工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Pixel-Level Reasoning Segmentation via Multi-turn Conversations](../../ACL2025/segmentation/pixel-level_reasoning_segmentation_via_multi-turn_conversations.md)
- [\[ICML 2025\] Balanced Learning for Domain Adaptive Semantic Segmentation](balanced_learning_for_domain_adaptive_semantic_segmentation.md)
- [\[ICML 2025\] IT³: Idempotent Test-Time Training](it3_idempotent_test-time_training.md)
- [\[ICML 2025\] Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery](alberta_wells_dataset_pinpointing_oil_and_gas_wells_from_satellite_imagery.md)
- [\[ICML 2025\] Adapter Naturally Serves as Decoupler for Cross-Domain Few-Shot Semantic Segmentation](adapter_naturally_serves_as_decoupler_for_cross-domain_few-shot_semantic_segment.md)

</div>

<!-- RELATED:END -->
