# CART: A Generative Cross-Modal Retrieval Framework with Coarse-To-Fine Semantic Modeling

## 基本信息

- **会议**: ACL 2025
- **arXiv**: [2406.17507](https://arxiv.org/abs/2406.17507)
- **代码**: 未公开
- **领域**: 信息检索
- **关键词**: 生成式检索, 跨模态检索, 语义标识符, K-Means, RQ-VAE, 粗到细
- **一句话总结**: 提出首个支持文本到图像/音频/视频的生成式跨模态检索框架 CART，通过 K-Means + RQ-VAE 构建粗到细语义标识符，结合特征融合策略，在检索性能和效率之间实现优异平衡。

## 研究背景与动机

跨模态检索旨在通过不同模态数据的交互搜索语义相关的实例。现有方法主要有两类：

**单塔模型**（如 BLIP-2, InternVL-G）：查询和候选之间做细粒度交互（如交叉注意力），检索精度高但延迟巨大，不适用于大规模检索

**双塔模型**（如 CLIP, CLAP）：将不同模态映射到联合嵌入空间计算相似度，效率更高但精度受模态鸿沟限制

**生成式检索**是一种新兴范式——为每个候选项分配标识符，将检索问题转化为序列到序列生成问题。其优势在于：
- 检索速度与数据集大小无关
- 无需维护统一嵌入空间
- 利用生成模型的强大能力提升性能

**核心挑战**：将生成式检索从文档检索扩展到跨模态检索面临三个问题：
1. 多模态数据缺乏可直接作为标识符的文本（不像文档有标题/关键词）
2. 低层视觉/听觉信息构建的标识符与自然语言查询存在语义鸿沟
3. 生成式检索缺乏查询与候选之间的显式交互过程

## 方法详解

### 整体框架

CART（Cross-modal Autoregressive Retrieval Transformer）包含三个模块：
1. **语义标识符生成**: 为每个候选项构建分层语义标识符
2. **标题增强**: 为多模态数据生成文本描述作为查询
3. **特征融合**: 在编码器-解码器架构中融合多层特征

### 关键设计一：粗到细语义标识符生成

标识符由三部分组成：**粗粒度token + 细粒度token + 唯一性token**

**粗粒度 Token (Coarse Token)**:
- 使用 ImageBind 编码所有候选项的嵌入
- 对嵌入执行 K-Means 聚类
- 聚类编号作为标识符的**第一个token**
- 直觉：第一个token至关重要，如果预测错误后续生成无意义；K-Means能捕获全局语义分类

**细粒度 Token (Fine Token)**:
- 计算原始嵌入与K-Means聚类中心的残差（突出细微差异）
- 使用 RQ-VAE（残差向量量化变分自编码器）对残差进行多层量化
- RQ-VAE 包含M个独立码本，递归量化残差：$v_m = \arg\min_k \|r_{m-1} - e_m^k\|$
- 每层量化捕获不同粒度的特征差异
- 训练损失包含重建损失和commitment损失

**唯一性 Token (Unique Token)**:
- 维护前缀数据库检测标识符冲突
- 对冲突的标识符追加计数器值，确保每个候选项拥有唯一标识符

最终标识符格式：$(k, v_1, v_2, \cdots, v_M, u)$

### 关键设计二：标题增强 (Caption Enhancement)

- 使用预训练多模态模型为每个候选项生成文本描述
- 将描述作为额外的查询，与标识符配对进行训练
- 有效弥合多模态标识符与自然语言查询之间的语义鸿沟

### 关键设计三：粗到细特征融合

使用标准编码器-解码器架构。编码器各层捕获不同层次的语义表示，设计两分支融合策略：

**粗融合 (Coarse Fusion)**:
$$Z = W[E_1, E_2, \ldots, E_S] + b$$
将所有编码器层输出拼接后通过融合层，再与解码器输入做交叉注意力，并通过 sigmoid 自门控进行后处理。

**细融合 (Fine Fusion)**:
$$L(Y, E(q)) = \sum_{i=1}^{S} \alpha_i \odot \mathcal{C}(Y, E_i)$$
类似 MoE 思路，将每个编码器层视为一个"专家"，各层输出独立与解码器交互，通过可学习权重调节各层贡献。

最终将粗融合和细融合的输出相加作为下一解码器层的输入。

### 损失函数

- 标准交叉熵损失：最大化正确标识符生成概率
- **双向KL散度损失**（R-Drop）：两次前向传播（不同dropout）的输出分布保持一致，防止过拟合
$$\mathcal{L}(\theta) = \sum_{(q,d)} (\log p(d|E(q), \theta) + \omega \mathcal{L}_{KL})$$

### 推理
- 使用**约束波束搜索**：利用前缀数据库构建前缀树，限制模型只生成有效标识符

## 实验

### 数据集
- 文本-图像：Flickr30K, MS-COCO
- 文本-音频：Clotho, AudioCaps
- 文本-视频：MSR-VTT, MSVD

### 主实验结果

**vs 单塔模型（Table 1）**:

| 方法 | Flickr30K R@1 | Flickr30K R@10 | 吞吐量 |
|------|-------------|---------------|--------|
| BLIP-2 | 89.7 | 98.9 | 1.68/s |
| InternVL-G | 85.0 | 98.6 | 2.03/s |
| **CART** | **81.8** | **98.4** | **105.8/s** |

CART 在Recall接近单塔模型（R@10仅差0.5），但**吞吐量提升63倍**。

**vs 双塔模型（Table 2 精选）**:

| 任务 | 方法 | R@1 | R@5 | R@10 |
|------|------|-----|-----|------|
| 文本-图像 (Flickr) | ImageBind | 74.9 | 93.0 | 96.1 |
| | **CART** | **81.8** | **96.1** | **98.4** |
| 文本-音频 (Clotho) | ONE-PEACE | 22.4 | 49.0 | 62.7 |
| | **CART** | **46.4** | **70.6** | **76.0** |
| 文本-视频 (MSR-VTT) | Cap4Video | 49.3 | 74.3 | 83.8 |
| | **CART** | **52.6** | **75.4** | **84.2** |

**在音频检索上优势最大**，R@1 提升超过100%。

**vs 生成式检索模型（Table 3）**:
CART 大幅超越 GRACE（使用预定义标识符），在 Flickr30K R@1 上 81.78 vs 68.4（Atomic ID）。

### 消融实验（Table 4, Flickr30K）

| 设置 | R@1 | R@10 | MRR@10 |
|------|-----|------|--------|
| w/o consistency loss | 81.64 | 98.04 | 87.85 |
| w/o fusion strategy | 75.54 | 96.72 | 83.11 |
| w/o K-Means | 79.50 | 97.52 | 86.12 |
| w/o RQ-VAE | 76.22 | 96.16 | 83.31 |
| **CART (完整)** | **81.78** | **98.38** | **88.04** |

- **融合策略**影响最大（去掉后 R@1 下降6.2%），说明多层特征交互至关重要
- 去掉 RQ-VAE（仅用层次K-Means）损失也显著，层次K-Means会丢失簇间语义信息
- K-Means 提供的先验知识对第一个token的准确预测有重要贡献

### 效率分析
- 随候选数增加，CLIP/CLAP 吞吐量持续下降（需逐个计算相似度）
- CART 在 CPU/GPU 上吞吐量保持稳定，候选数无关（标识符已编码进模型参数）
- 1M候选项+100并发查询场景下优势极其显著

## 亮点与洞察

1. **首个全面支持文本-图像/音频/视频的生成式跨模态检索框架**
2. **K-Means + RQ-VAE 互补设计**: K-Means 提供全局语义分类，RQ-VAE 捕获细微差异
3. **效率与性能的优秀平衡**: 性能接近甚至超越双塔模型，效率远超单塔模型
4. **音频检索突破**: 在 Clotho 和 AudioCaps 上大幅超越所有基线
5. **唯一性token的工程智慧**: 简单的前缀数据库方案优雅地解决了标识符冲突问题

## 局限性

1. **未在超大规模数据集上验证**: 实验数据集规模有限（Flickr30K仅31K图片）
2. **模型更新成本**: 新增候选项需要重新生成标识符并微调模型
3. **标识符质量依赖 ImageBind**: 嵌入质量直接影响标识符语义质量
4. **仅支持文本查询**: 未探索图-图、音-文等其他跨模态方向
5. **训练需4张V100**: 相比双塔模型的对比学习，训练范式更复杂

## 相关工作

- **跨模态检索**: CLIP, BLIP-2, InternVL, ImageBind, LanguageBind
- **生成式检索**: DSI, NCI, GENRE, SEAL, GRACE
- **向量量化**: RQ-VAE, VQ-VAE, SoundStream
- **信息检索**: BM25, Dense Retrieval, Meshed-Memory Transformer

## 评分 ⭐⭐⭐⭐

- **创新性**: ⭐⭐⭐⭐ — 首次将生成式检索全面扩展到跨模态场景，标识符设计新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三种模态、六个数据集、三种检索范式的全面对比
- **实用性**: ⭐⭐⭐⭐ — 在大规模检索场景下有显著效率优势
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] POGA: Paraphrased and Oppositional Graph Alignment for Fine-Grained Cross-Modal Retrieval](../../CVPR2026/information_retrieval/poga_paraphrased_and_oppositional_graph_alignment_for_fine-grained_cross-modal_r.md)
- [\[ACL 2025\] Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval](maximal_matching_matters_preventing_representation_collapse_for_robust_cross-mod.md)
- [\[CVPR 2025\] NeighborRetr: Balancing Hub Centrality in Cross-Modal Retrieval](../../CVPR2025/information_retrieval/neighborretr_balancing_hub_centrality_in_cross-modal_retrieval.md)
- [\[CVPR 2025\] GENIUS: A Generative Framework for Universal Multimodal Search](../../CVPR2025/information_retrieval/genius_a_generative_framework_for_universal_multimodal_search.md)
- [\[ICML 2025\] FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems](../../ICML2025/information_retrieval/fedrag_a_framework_for_fine-tuning_retrieval-augmented_generation_systems.md)

</div>

<!-- RELATED:END -->
