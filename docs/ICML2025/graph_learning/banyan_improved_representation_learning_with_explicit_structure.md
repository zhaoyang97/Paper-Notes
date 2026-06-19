---
title: >-
  [论文解读] Banyan: Improved Representation Learning with Explicit Structure
description: >-
  [ICML 2025][图学习][递归神经网络] Banyan 通过纠缠层次树结构：和对角化消息传递：两大创新，仅用 14 个非嵌入参数就在语义文本相似度任务上超越了大规模 Transformer 模型，为低资源语言的语义表示学习提供了高效可行的替代方案。 语义表示是 RAG、问答、摘要等众多 NLP 应用的基础…
tags:
  - "ICML 2025"
  - "图学习"
  - "递归神经网络"
  - "层次结构学习"
  - "语义表示"
  - "对角化消息传递"
  - "纠缠树"
---

# Banyan: Improved Representation Learning with Explicit Structure

**会议**: ICML 2025  
**arXiv**: [2407.17771](https://arxiv.org/abs/2407.17771)  
**代码**: [github.com/exlab-research/Banyan](https://github.com/exlab-research/Banyan)  
**领域**: 图学习  
**关键词**: 递归神经网络, 层次结构学习, 语义表示, 对角化消息传递, 纠缠树

## 一句话总结

Banyan 通过**纠缠层次树结构**和**对角化消息传递**两大创新，仅用 14 个非嵌入参数就在语义文本相似度任务上超越了大规模 Transformer 模型，为低资源语言的语义表示学习提供了高效可行的替代方案。

## 研究背景与动机

语义表示是 RAG、问答、摘要等众多 NLP 应用的基础。当前主流方法依赖大规模 Transformer，需要海量数据和算力来训练。这在低资源语言场景中面临严峻挑战——数据不足、算力受限，使得 scaling 方案不可行。

从语言学和认知科学出发，**组合语义原则**（compositional semantics）认为：理解整体语义 = 理解部分语义 + 理解组合规则。这一原则天然具有高效性——通过系统的合成规则处理新表达，而非逐一存储含义。

前驱工作 Self-StrAE 已初步验证了结构化学习的潜力，但与大规模预训练 Transformer 相比仍有差距。本文提出 Banyan 模型，着力解决两个核心问题：

**上下文缺失**：Self-StrAE 中每个句子独立构建树，同一子结构（如 "ate doughnuts"）在不同上下文（"Lisa ate..." vs "Homer ate..."）中无法共享信息

**参数冗余**：原始线性层消息传递函数参数量与嵌入维度 U 的平方成正比，效率低下

## 方法详解

### 整体框架

Banyan 本质上是一个**递归神经网络（RvNN）**，同时学习结构和表示。核心流程：

1. **上行组合**（Upward Compose）：从叶节点（token 嵌入）出发，基于余弦相似度贪心合并最相似的相邻节点对，逐层自底向上构建树结构，直到产生根节点嵌入
2. **纠缠建图**（Entangling）：将 batch 内所有句子的独立树结构合并为一个共享图——相同子结构的节点被统一为单个节点，避免重复
3. **下行分解**（Downward Decompose）：从根节点自顶向下分解，每个节点的下行嵌入通过平均其在不同上下文中的多个分解结果来融合全局语境
4. **重建/预测**：在叶节点处重建原始 token，通过交叉熵损失训练

### 关键设计

#### 纠缠树（Entangled Trees）

这是 Banyan 区别于 Self-StrAE 的核心创新。Self-StrAE 为每个句子独立构建二叉树，树之间互不关联。Banyan 则维护一个**全局前沿**（global frontier），将 batch 内所有句子的叶节点放入同一个前沿进行合并。

关键步骤（Algorithm 1）：

- 初始化时将所有句子的 token 放入全局前沿 $\mathcal{A}$
- 每次从前沿中找到余弦相似度最高的相邻节点对 $(e_{i^\star}, e_{i^\star+1})$
- 通过组合函数生成父节点 $e_p$
- **关键**：通过节点身份 $s_n$（span 标识）检查该组合是否已存在于图中，若已存在则复用而非新建
- 在前沿中**所有出现位置**同时替换该节点对为父节点
- 持续迭代直至所有句子归并完成

这样做的三重好处：

- **上下文聚合**：同一 span 在不同上下文中的下行嵌入通过取平均来融合：$\underline{e} = \frac{1}{|\mathcal{Y}|}\sum_y \underline{e}^y$
- **消除假负样本**：对比学习中，Self-StrAE 中同一 span 的不同实例会被错误地当作负样本互相推离；纠缠树通过节点去重自然避免了这个问题
- **内存高效**：去重后节点数大幅减少，显著降低内存占用

#### 对角化消息传递（Diagonalised Message Passing）

第二个创新是将组合/分解函数从线性层简化为对角操作。原始 Self-StrAE 使用全连接线性层：

$$C_\Phi(\bar{e}_i, \bar{e}_{i+1}) = \text{hcat}(\bar{e}_i, \bar{e}_{i+1})\Phi + \phi, \quad \Phi \in \mathbb{R}^{2U \times U}$$

Banyan 改用**逐元素缩放**（element-wise scaling）：

$$C(\bar{e}_i, \bar{e}_{i+1}) = (\bar{e}_i \cdot \sigma(\Phi_l) + \bar{e}_{i+1} \cdot \sigma(\Phi_r)) + \phi$$

$$D(\underline{e}_i) = (\underline{e}_i \cdot \sigma(\Theta_l) + \theta_l, \; \underline{e}_i \cdot \sigma(\Theta_r) + \theta_r)$$

其中 $\Phi_l, \Phi_r, \phi, \Theta_l, \Theta_r, \theta_l, \theta_r \in \mathbb{R}^U$，$\sigma$ 为 sigmoid 函数。

设计动机和优势：

- **灵感来源**：借鉴了线性 RNN 复兴中对角化技术（Mamba、RWKV 等），引入沿结构深度的衰减记忆
- **参数缩减**：参数量从 $O(U^2)$ 降至 $O(U)$，减少了 $U$ 倍
- **压缩序保持**：sigmoid 非线性确保数值稳定且执行记忆衰减——上行组合时表示幅度增大（两子节点之和），下行分解时幅度减小回到叶节点水平
- **空间一致性**：编码器和解码器的嵌入被约束在同一空间中，使得摊余学习（amortisation）成为成功重建的必要条件

#### 嵌入维度配置

Self-StrAE 将 256 维嵌入视为 $16 \times 16$ 的方阵（$K=16, U=16$），组合/分解函数独立作用于 $K$ 个通道。Banyan 将配置改为 $K=128, U=2$——**最大化独立通道数**，同时将每通道维度降至极低。这使得非嵌入参数总数仅为 **14 个**（$7 \times U = 7 \times 2$）。

### 损失函数 / 训练策略

**训练目标**：由于对角化函数约束编码/解码嵌入处于同一空间，Banyan 可以直接使用交叉熵损失进行训练，而无需 Self-StrAE 中的对比损失：

$$\mathcal{L}_{\text{CE}}(\mathbf{w}, \hat{\mathbf{w}}) = -\frac{1}{N}\sum_{n=1}^N w_n \cdot \log \hat{w}_n$$

**训练数据**：仅使用约 1000 万 token 的英文 Wikipedia 子集（模拟低资源设置），远少于 RoBERTa 使用的 1 亿 token。

**批次纠缠估计**：理想情况下应在全部数据上构建纠缠树，但数据量指数增长使之不可行。Banyan 在每个 batch 上独立构建纠缠树作为估计，该估计是无偏的——batch 内的平均是总体平均的无偏估计。

## 实验关键数据

### 主实验

**英文句子级语义相似度（Spearman ρ × 100）**：

| 模型 | STS-12 | STS-13 | STS-14 | STS-15 | STS-B | SICK-R | 平均 Score |
|------|--------|--------|--------|--------|-------|--------|-----------|
| Self-StrAE | 31.98 | 53.88 | 37.73 | 55.23 | 39.53 | 51.78 | 46.59 |
| GloVe+stopword rm | 39.00 | 41.61 | 39.31 | 51.06 | 48.40 | 52.80 | 44.96 |
| Sent2Vec | 38.14 | 51.37 | 48.64 | 67.28 | 53.39 | 59.67 | 53.28 |
| RoBERTa+SimCSE | 50.63 | 62.23 | 54.17 | 68.77 | 53.53 | 56.87 | 59.02 |
| **Banyan** | **51.38** | **69.60** | **63.20** | **73.08** | **61.90** | **55.23** | **62.97** |

**英文检索和分类任务**：

| 模型 | Quora NDCG@10 | Arguana NDCG@10 | SST-2 Acc | MRPC F1 |
|------|--------------|----------------|-----------|---------|
| Self-StrAE | 40.02 | 15.48 | 74.67 | 80.34 |
| RoBERTa+SimCSE | 59.30 | 21.84 | 75.97 | 80.83 |
| **Banyan** | **65.71** | **28.28** | 75.96 | 79.48 |

### 消融实验

| 配置 | 说明 | 效果 |
|------|------|------|
| Self-StrAE 基线 | 独立树 + 全连接层 + 对比损失 | Score 46.59 |
| + 纠缠树 | 替换为全局纠缠图结构 | 显著提升（消除假负样本、共享上下文） |
| + 对角化消息传递 | 对角缩放替代线性层 | 进一步提升（参数从 1072 降至 14） |
| + CE 损失替换对比损失 | 利用空间一致性切换目标 | 训练更稳定 |
| K=128, U=2 配置 | 最大化通道数、最小化通道维度 | 最佳性能 |

**模型规模对比**：

| 模型 | 非嵌入参数量 |
|------|------------|
| **Banyan** | **14** |
| Self-StrAE | 1,072 |
| RoBERTa (M) | ~10M |
| MiniLM-L12 | ~21M |
| XLM-R | ~85M |
| Llama 3.1 | ~8B |
| Mistral Nemo | ~12B |

### 关键发现

1. **极致参数效率**：14 个非嵌入参数超越了拥有数百万乃至数十亿参数的 Transformer，在 STS 任务上 Banyan（62.97）> RoBERTa+SimCSE（59.02），参数量差距超过 5 个数量级
2. **跨层次语义迁移**：与 GloVe/Sent2Vec 不同，结构化模型能将词级语义无缝迁移到句子级——体现了组合语义的真正优势
3. **检索任务优势显著**：Quora 检索 Banyan NDCG@10 达 65.71，超越 SimCSE RoBERTa 的 59.30，说明结构化表示在 RAG 关键应用中具有实际价值
4. **多语言低资源场景表现突出**：在 9 种语言的 SemRel 评测中，Banyan 平均得分 60.01，超越 XLM-R（46.61）、Llama 3.1 8B（53.06）等大模型，在 Hausa 等极低资源语言上优势尤为明显（49.68 vs 4.1）

## 亮点与洞察

- **反直觉的极简主义**：在深度学习追求更大规模的时代，Banyan 用 14 个参数击败了数十亿参数的模型，证明了正确的归纳偏置（inductive bias）比蛮力 scaling 更有价值
- **纠缠树的优雅设计**：通过批次级共享图结构，一举三得——融合上下文、消除假负样本、降低内存占用
- **对角化的深远影响**：借鉴线性 RNN 的思路用于树结构，不仅减参还提性能，暗示"沿结构的记忆衰减"是一个普适的好归纳偏置
- **名字的隐喻**：模型取名 Banyan（榕树），因为榕树有多个根（对应纠缠图的多根结构）且节点可在不同分支间共享——兼具诗意和技术贴切性

## 局限与展望

1. **词级相似度弱项**：在 SimLex-999 上表现较差（14.65），因为模型需要同时建模 similarity 和 relatedness，而 SimLex 严格排除 relatedness
2. **分类任务无优势**：SST-2 和 MRPC 上 Banyan 与 baseline 持平，说明分类可能更依赖参数化的上层分类器而非表示质量本身
3. **贪心合并策略**：基于余弦相似度的贪心合并可能产生次优树结构，未来可探索更全局的结构优化方法
4. **batch 纠缠的近似性**：虽然无偏，但 batch 级纠缠树仅是全数据纠缠的近似，batch 越大效果理应越好，但受限于内存
5. **仅限 NLP**：当前仅在文本语义相似度任务上验证，未探索视觉、多模态等领域的适用性

## 相关工作与启发

- **Self-StrAE**（Opper et al., 2023）：Banyan 的直接前驱，提供了基于表示相似度的自动结构归纳方法
- **线性 RNN 复兴**（Mamba, RWKV 等）：对角化记忆衰减思想的来源，证明了简化不意味着性能下降
- **SimCSE**（Gao et al., 2021）：对比学习增强表示质量的代表，Banyan 在不使用对比学习的情况下超越了它
- **SemRel 数据集**（Ousidhoum et al., 2024）：为低资源语言评测提供了标准化测试集
- **启发**：该工作启示我们在图学习和结构化表示领域，精心设计的归纳偏置可能比 scaling 更有效；纠缠图的思路也可以推广到其他需要跨实例共享结构的场景

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|-----------|------|
| 新颖性 | 9 | 纠缠树和对角化消息传递都是原创且优雅的设计 |
| 技术深度 | 8 | 理论分析扎实，无偏估计论证清晰 |
| 实验充分度 | 8 | 英文多维评测 + 9 种语言跨语言验证，消融完整 |
| 实用价值 | 8 | 低资源语言场景直接可用，14 参数极易部署 |
| 写作质量 | 8 | 结构清晰，图示直观，Banyan 命名有巧思 |
| **综合** | **8.2** | 一篇令人眼前一亮的工作，用极简方案挑战 scaling law |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] NN-Former: Rethinking Graph Structure in Neural Architecture Representation](../../CVPR2025/graph_learning/nn-former_rethinking_graph_structure_in_neural_architecture_representation.md)
- [\[ICML 2025\] TopInG: Topologically Interpretable Graph Learning via Persistent Rationale Filtration](toping_topologically_interpretable_graph_learning_via_persistent_rationale_filtr.md)
- [\[NeurIPS 2025\] SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs](../../NeurIPS2025/graph_learning/sstag_structure-aware_self-supervised_learning_method_for_text-attributed_graphs.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](../../ICML2026/graph_learning/t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](../../ICLR2026/graph_learning/towards_improved_sentence_representations_using_token_graphs.md)

</div>

<!-- RELATED:END -->
