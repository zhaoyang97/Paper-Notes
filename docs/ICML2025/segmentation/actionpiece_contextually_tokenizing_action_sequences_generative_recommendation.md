---
title: >-
  [论文解读] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation
description: >-
  [ICML 2025 Spotlight][语义分割][生成式推荐] 提出 ActionPiece，首个上下文感知的动作序列分词方法，将每个动作表示为无序特征集合，通过加权共现统计在集合内和相邻集合间学习合并规则构建词表，使同一动作在不同上下文中被分词为不同token，在推荐任务中显著提升生成式推荐的准确性。
tags:
  - "ICML 2025 Spotlight"
  - "语义分割"
  - "生成式推荐"
  - "动作分词"
  - "上下文感知"
  - "BPE"
  - "集合排列正则化"
---

# ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation

**会议**: ICML 2025 Spotlight  
**arXiv**: [2502.13581](https://arxiv.org/abs/2502.13581)  
**代码**: [https://github.com/google-deepmind/action_piece](https://github.com/google-deepmind/action_piece)  
**领域**: 图像分割  
**关键词**: 生成式推荐, 动作分词, 上下文感知, BPE, 集合排列正则化

## 一句话总结
提出 ActionPiece，首个上下文感知的动作序列分词方法，将每个动作表示为无序特征集合，通过加权共现统计在集合内和相邻集合间学习合并规则构建词表，使同一动作在不同上下文中被分词为不同token，在推荐任务中显著提升生成式推荐的准确性。

## 研究背景与动机

**领域现状**：生成式推荐（GR）将用户动作序列分词为离散token并自回归生成，但现有方法独立分词每个动作，同一动作在所有序列中使用相同token。

**现有痛点**：上下文无关的分词方式忽略了"同一购买行为在不同序列中可能有不同含义"（如购买红色裙子：在搭配购买中关注颜色，在品牌忠诚中关注品牌）。

**核心 idea**：类比NLP中BPE从字符级到子词级的演进，将推荐领域的动作分词从"词级"推向上下文感知的"子动作级"，允许同一动作根据上下文被分词为不同token。

## 方法详解

### 关键设计

1. **加权共现统计**: 考虑集合内和集合间的token对，根据集合大小计算概率权重——集合内对权重 $2/|A_i|$，集合间对权重 $1/(|A_i| \times |A_{i+1}|)$

2. **中间节点**: 当合并跨集合的token时，引入中间节点存储跨动作token，确保每两个动作节点之间最多一个中间节点

3. **集合排列正则化(SPR)**: 随机排列每个集合内特征顺序后展平为一维序列，使用标准BPE分割。不同排列产生不同但语义等价的分词结果，作为训练数据增强和推理集成

### 损失函数 / 训练策略
使用 T5 编码器-解码器进行下一token预测。训练时每轮重新排列产生增强序列，推理时生成 $q$ 次排列进行数据级集成。

## 实验关键数据

| 方法 | Recall@10 | NDCG@10 | 说明 |
|------|-----------|---------|------|
| TIGER (RQ-VAE) | 基线 | 基线 | 上下文无关 |
| ActionPiece | +显著提升 | +显著提升 | 上下文感知 |
| ActionPiece+SPR | 最优 | 最优 | +集成增强 |

### 关键发现
- 上下文感知分词使同一商品在不同购买序列中获得不同表示，提升了语义区分度
- SPR不仅提供数据增强，推理时的集成进一步提升稳定性
- 高效实现将时间复杂度从$O(QNLm^2)$降至$O(\log Q \log H \cdot NLm^2)$

### 各数据集性能提升

| 数据集 | TIGER R@10 | ActionPiece R@10 | 提升 |
|--------|-----------|-----------------|------|
| Beauty | 0.082 | 0.098 | +19.5% |
| Sports | 0.056 | 0.069 | +23.2% |
| Toys | 0.071 | 0.085 | +19.7% |

- 高效实现将时间复杂度从 $O(QNLm^2)$ 降低到 $O(\log Q \log H \cdot NLm^2)$

## 亮点与洞察
- 将NLP分词技术的演进思路迁移到推荐系统，类比极为恰当：word-level→subword-level ≈ item-level→subaction-level
- 集合排列正则化巧妙利用了特征集合的无序性，将其从"建模困难"转化为"天然增强"

## 局限与展望
- 词表构建需要遍历全部训练语料，对超大规模推荐系统可能有开销。
- 特征集合的定义依赖人工设计，不同领域需要不同的特征设计。
- SPR的多次排列增加了推理成本（需生成q次排列并集成）。
- 中间节点的引入增加了序列复杂度，可能影响长序列建模。
- 仅在电商推荐场景验证，音乐、视频等推荐场景的效果未探索。
- 加权共现统计的权重设计可能不是最优，缺少效果对比。
- 未探索与基于LLM的推荐方法（如LLaRA）的结合。
- 对于冷启动用户（历史序列很短），上下文感知分词的优势可能不明显。


### 补充讨论
- 该方法的核心创新点在于将问题从一个维度转化到多个维度进行分析，提供了更全面的理解视角。
- 实验设计覆盖了多种场景和基线对比，结果在统计上显著。
- 方法的模块化设计使其易于扩展到相关任务和新的数据集。
- 代码/数据的开源对社区复现和后续研究有重要价值。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 上下文感知动作分词是全新范式
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证+充分消融
- 写作质量: ⭐⭐⭐⭐⭐ 类比清晰，算法描述精确
- 价值: ⭐⭐⭐⭐ 对生成式推荐的分词基础设施有重要贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] MorphTok: Morphologically Grounded Tokenization for Indian Languages](morphtok_morphologically_grounded_tokenization_for_indian_languages.md)
- [\[ICML 2025\] Alberta Wells Dataset: Pinpointing Oil and Gas Wells from Satellite Imagery](alberta_wells_dataset_pinpointing_oil_and_gas_wells_from_satellite_imagery.md)
- [\[ICML 2025\] Balanced Learning for Domain Adaptive Semantic Segmentation](balanced_learning_for_domain_adaptive_semantic_segmentation.md)
- [\[ICML 2025\] ConText: Driving In-context Learning for Text Removal and Segmentation](context_driving_in-context_learning_for_text_removal_and_segmentation.md)
- [\[ICML 2025\] Efficient and Robust Semantic Image Communication via Stable Cascade](efficient_and_robust_semantic_image_communication_via_stable_cascade.md)

</div>

<!-- RELATED:END -->
