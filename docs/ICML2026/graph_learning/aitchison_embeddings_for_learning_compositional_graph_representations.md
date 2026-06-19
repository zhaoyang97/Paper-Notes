---
title: >-
  [论文解读] Aitchison Embeddings for Learning Compositional Graph Representations
description: >-
  [ICML2026][图学习][Aitchison 几何] 本文提出 AICoG，将节点表示为 simplex 上的潜在原型混合，并用 Aitchison 几何与 ILR 等距坐标学习图嵌入，在保持与欧氏 latent distance model 同等表达力的同时，让节点角色相似性具有基于相对比例 trade-off 的内生解释。
tags:
  - "ICML2026"
  - "图学习"
  - "Aitchison 几何"
  - "图表示学习"
  - "组合数据"
  - "ILR 坐标"
  - "可解释嵌入"
---

# Aitchison Embeddings for Learning Compositional Graph Representations

**会议**: ICML2026  
**arXiv**: [2605.00716](https://arxiv.org/abs/2605.00716)  
**代码**: https://github.com/Nicknakis/AICoG  
**领域**: 图学习 / 可解释表示学习  
**关键词**: Aitchison 几何, 图表示学习, 组合数据, ILR 坐标, 可解释嵌入  

## 一句话总结
本文提出 AICoG，将节点表示为 simplex 上的潜在原型混合，并用 Aitchison 几何与 ILR 等距坐标学习图嵌入，在保持与欧氏 latent distance model 同等表达力的同时，让节点角色相似性具有基于相对比例 trade-off 的内生解释。

## 研究背景与动机

**领域现状**：图表示学习通常把节点映射到欧氏向量空间，用随机游走、矩阵分解、GNN 或 latent distance model 保持结构邻近性。这些方法在链接预测和节点分类上有效，但嵌入维度往往缺少语义，距离和方向很难直接解释。

**现有痛点**：很多网络并不只是“相邻节点相似”，而是存在连续、重叠的结构角色。节点可能同时具有多种 latent archetype 的比例，例如社交网络中的桥接者、内容生产者、社区核心等混合角色。传统 mixed-membership 模型可以表达角色混合，但通常假设角色是离散、可识别、坐标轴对齐的；普通欧氏嵌入虽灵活，却无法说明某个方向对应什么相对角色变化。

**核心矛盾**：图嵌入需要既有预测性能，又能解释节点为何相似。欧氏空间有表达力但语义弱，离散角色模型有解释性但过于刚性；连续重叠角色更像“多个原型比例的相对权衡”，而不是单个坐标值。

**本文目标**：作者希望构建一种图嵌入框架，把节点角色显式建模为 simplex 上的 composition，用适合组合数据的 Aitchison 几何定义距离，使相似性天然对应原型比例之间的 log-ratio trade-off。

**切入角度**：组合数据的核心是“比例有意义，绝对尺度无意义”。Aitchison 几何正是处理这种相对信息的标准工具；ILR transformation 又能把 simplex 等距映射到无约束欧氏空间，从而兼顾几何语义和优化便利性。

**核心 idea**：把每个节点表示为潜在 archetype 的组成比例 $\mathbf{z}_i$，用 ILR 坐标 $\mathbf{x}_i=\operatorname{ILR}(\mathbf{z}_i)$ 保持 Aitchison 距离，再用 latent distance likelihood 学习图结构。

## 方法详解

### 整体框架

AICoG 从一个无向简单图 $\mathcal{G}=(V,E)$ 出发，为每个节点学习一个 $K$ 维 composition $\mathbf{z}_i\in\Delta^{K-1}$。这里每个维度不是普通欧氏坐标，而是一个潜在原型因子的相对贡献；所有分量为正且和为 1。节点角色不是某个单一原型，而是在 simplex 内部的连续混合。

为了避免直接在受约束 simplex 上优化，方法使用 isometric log-ratio transformation。给定 contrast space 的正交基 $\mathbf{V}$，ILR 坐标为 $\mathbf{x}_i=\log(\mathbf{z}_i)^\top\mathbf{V}\in\mathbb{R}^{K-1}$。Aitchison 距离 $d_A(\mathbf{z}_i,\mathbf{z}_j)$ 等于 ILR 空间里的欧氏距离 $\|\mathbf{x}_i-\mathbf{x}_j\|_2$，因此可以在欧氏坐标中做标准优化，但解释仍留在组合比例的 log-ratio 语义中。

图结构通过 Bernoulli latent distance model 学习。对节点对 $(i,j)$，模型定义 log-odds $\eta_{ij}=-\|\mathbf{x}_i-\mathbf{x}_j\|_2+\gamma_i+\gamma_j$，其中 $\gamma_i$ 捕获节点度异质性。训练目标是最大化所有边/非边的 Bernoulli log-likelihood；为避免 $O(N^2)$ 全对计算，非边项通过均匀 subsampling 近似，使每次迭代复杂度降到 $O(|E|)$。

### 关键设计

**1. Simplex 节点角色与 Aitchison 几何：用相对比例而非绝对坐标定义相似**

普通欧氏嵌入和离散角色模型各有短板：前者的坐标轴没有固有语义，后者（如 MMSBM）又假设角色是离散、可识别、坐标轴对齐的，遇到"一个节点同时混合多种连续重叠角色"就力不从心。AICoG 把节点 $i$ 的角色直接表示为 simplex 上的组成 $\mathbf{z}_i=(z_{i1},\dots,z_{iK})\in\Delta^{K-1}$，每一维是某个潜在原型（archetype）的相对贡献、各分量为正且和为 1——simplex 顶点对应被单一原型主导的纯角色、内部点对应混合角色。关键是用 Aitchison 几何而非欧氏距离来比较这些组成：度数或活跃度这类绝对规模往往只是干扰因素，两个节点可能交互量差很多却有相同的相对交互模式，因此 Aitchison 几何只看比例（log-ratio）、对整体缩放不敏感，两个节点相似当且仅当它们在各原型间的 log-ratio 权衡相似。这样"相似"就天然落在"相对角色分布"而非"绝对交互量"上，贴合连续重叠角色的语义。

**2. ILR 等距坐标与可学习 basis：把 simplex 优化变成无约束欧氏优化**

直接在受约束的 simplex 上做梯度优化很别扭，而直接解释每个 simplex 分量又会退回"坐标轴=角色"的老问题。等距对数比变换（isometric log-ratio, ILR）一举化解这对两难：给定 contrast space 的正交基 $\mathbf{V}$，把组成映射为坐标 $\mathbf{x}_i=\log(\mathbf{z}_i)^\top\mathbf{V}\in\mathbb{R}^{K-1}$，并保证 Aitchison 距离恰好等于 ILR 坐标的欧氏距离 $d_A(\mathbf{z}_i,\mathbf{z}_j)=\|\mathbf{x}_i-\mathbf{x}_j\|_2$。于是模型可以在无约束的 $\mathbb{R}^{K-1}$ 里用标准梯度法优化，几何语义却仍留在组合比例里。又因为任意两个合法 ILR basis 只差一个正交变换、距离和 likelihood 都不变，可解释性是表示空间本身的性质而非某套坐标的性质：论文既用域无关的固定 Helmert basis，也用与嵌入联合训练的 learned basis（并可配 varimax 旋转得到更稀疏的 balance）。每个 balance 都是一组原型对另一组原型的 log-ratio 对比，比单看某一维坐标更契合连续角色空间。

**3. 子组合一致性：有语义地移除原型组件**

欧氏嵌入的维度没有固有语义，随手丢掉几维很难说清丢的是什么；而组合数据的组件本身就是原型比例，"移除一部分原型"是有明确几何含义的合法操作。AICoG 利用 Aitchison 几何的子组合一致性（subcompositional coherence）：选定原型子集 $S$ 后，对相应分量重新归一化（re-closure）得到子组合 $\mathbf{z}_i^{(S)}$，论文证明（Lemma 3.1）其 ILR 距离恰好等于原 ILR 差向量在对应子空间上的正交投影范数。这意味着无需重新训练，就能直接对训练好的模型移除若干原型组、重归一化后评估节点分类性能的保留率，从而探查"哪些原型组真正影响预测"。比起对训练好的黑盒做 post-hoc 归因，这种解释在建模时就内生在表示空间里。

### 损失函数 / 训练策略

节点 composition 通过无约束 logits $\tilde{\mathbf{z}}_i$ 参数化，并用 row-wise softmax 得到 $\mathbf{z}_i$。边概率由 $\eta_{ij}=-\|\mathbf{x}_i-\mathbf{x}_j\|_2+\gamma_i+\gamma_j$ 进入 logistic Bernoulli likelihood，完整 log-likelihood 为 $\sum_{i<j}[Y_{ij}\eta_{ij}-\log(1+\exp(\eta_{ij}))]$。作者证明，ILR-compositional latent distance model 与 $\mathbb{R}^{K-1}$ 中普通 Euclidean latent distance model 可表示的边概率矩阵集合相同，因此组合约束不牺牲表达力。

实验中 AICoG 使用 Adam 优化 Bernoulli negative log-likelihood，学习率 $10^{-2}$，训练 5000 次迭代。维度 $D=K-1$，评估 $D\in\{8,16,32,64\}$。数据集包括 Cora、Citeseer、LastFM、DBLP、AstroPh、GrQc、HepTh；比较对象包括 Node2Vec、Role2Vec、NetMF、MMSBM、MNMF、SLIM-Raa、HM-LDM 和 Simplex-Euclidean。

## 实验关键数据

### 主实验

| 任务 / 数据集 | 维度 | 强 baseline | AICoG (HB) | AICoG (LB) | 主要结论 |
|--------------|------|-------------|------------|------------|----------|
| Link prediction AstroPh AUC-ROC | 64 | SLIM-Raa 0.969 | 0.976 | 0.976 | AICoG 达到最优 |
| Link prediction GrQc AUC-ROC | 64 | SLIM-Raa 0.949 | 0.961 | 0.961 | 大幅超过传统 mixed-membership |
| Link prediction HepTh AUC-ROC | 64 | SLIM-Raa 0.920 | 0.929 | 0.928 | Aitchison 几何稳定领先 |
| Link prediction Cora AUC-ROC | 64 | HM-LDM 0.806 | 0.851 | 0.852 | 组合几何对 citation graph 有明显收益 |
| Node classification Cora Micro-F1 | 64 | Node2Vec 0.814 / HM-LDM 0.814 | 0.831 | 0.833 | 可解释模型不牺牲分类性能 |
| Node classification LastFM Micro-F1 | 64 | Node2Vec 0.865 | 0.870 | 0.870 | 与最强欧氏 baseline 持平略优 |

### 消融实验

| 分析项 | 设置 | 关键指标 | 说明 |
|------|------|---------|------|
| Aitchison vs simplex 欧氏 | Simplex-Euclidean | Cora AUC-ROC 64 维仅 0.709，而 AICoG 约 0.851 | 关键不是 simplex 约束本身，而是组合数据的 Aitchison 几何 |
| 合成 membership recovery | AICoG vs MMSBM | ILR-continuous: $\ell_1$ 0.900 vs 1.452，cosine 0.645 vs 0.432，JS 0.154 vs 0.356 | AICoG 更能恢复连续/内部 membership |
| membership interiority | Cora | AICoG entropy 1.064、near-corner 5.55%；MMSBM entropy 0.191、near-corner 78.95% | AICoG 学到的角色更重叠、更 interior，且 label-informative |
| 单 balance 解释 | Cora learned ILR basis | 单个 balance 约 0.40 probe accuracy，ANOVA $F\approx319$，MI $\approx0.44$ | 一个 log-ratio 对比就能捕获部分标签结构 |
| 子组合评估 | Cora 64 维，随机移除组件 | AICoG 在 aggressive compression 下 retention 最强 | re-closure 后的组件限制保留了有语义的几何结构 |

### 关键发现
- AICoG 在链接预测上非常强，尤其 Cora、GrQc、HepTh 等数据集上，固定 Helmert basis 和 learned basis 的结果几乎一致，支持 ILR basis 正交不变性的论点。
- Simplex-Euclidean 明显掉点，说明把节点放到 simplex 上还不够；如果仍用普通欧氏距离比较比例，无法捕获 compositional data 的相对语义。
- 节点分类中，纯欧氏方法仍很强，但 AICoG 在 Cora 和 LastFM 上达到或超过 Node2Vec/Role2Vec，证明可解释几何不必以明显预测性能为代价。
- 合成实验显示 MMSBM 容易把 membership 推向 near-discrete 角点，而 AICoG 更适合连续、重叠的角色结构。

## 亮点与洞察
- 最大亮点是把图角色解释从“某个坐标轴是什么”转成“多个原型之间的相对 trade-off 是什么”。这避免了 mixed-membership 模型必须识别离散角色的限制。
- ILR 的使用很巧妙：它让模型优化时看起来像普通欧氏 latent distance model，但所有距离都可回译成 Aitchison 空间中的 log-ratio 差异。
- Expressive equivalence 的理论保证降低了采用组合几何的顾虑。方法不是用解释性换表达力，而是在同等 latent distance 表达力下改变几何语义。
- 子组合分析提供了一种比 post-hoc attribution 更自然的解释方式。移除某组 archetype、re-close、看预测保持多少，这个操作本身在组合数据理论里是合法的。

## 局限与展望
- AICoG 最适合节点角色本来具有 compositional 语义的图；如果图结构主要由局部同质性或非比例因素决定，未必比普通欧氏嵌入更准。
- 论文主要在 featureless graph 和无监督表示学习设置下评估，没有与现代带属性 GNN 或端到端监督图模型直接竞争。
- 训练协议假设图连通或由大连通分量主导。作者也指出，对许多小连通分量或断开图的扩展仍是未来方向。
- ILR basis 虽然不影响距离，但影响人类如何阅读 balance；learned basis 和 varimax rotation 提供帮助，但如何把 archetype 与领域知识自动对齐仍未解决。
- 似然模型仍是基于 pairwise distance 的 Bernoulli graph model，复杂关系如有向边、异质边、多关系图和动态演化图还需要扩展。

## 相关工作与启发
- **vs Node2Vec / DeepWalk**: 这些方法通过随机游走学习欧氏嵌入，预测强但维度语义弱；AICoG 的距离直接解释为原型比例 trade-off。
- **vs Role2Vec / GraphWave**: 它们关注结构角色，但通常仍输出普通向量；AICoG 将角色建模为连续组合，并用 Aitchison 几何定义相似性。
- **vs MMSBM / mixed-membership SBM**: MMSBM 提供角色 membership，但更偏离散和坐标轴解释；AICoG 允许 continuous interior compositions，并通过 geometry 而非唯一坐标解释角色。
- **vs SLIM-Raa / HM-LDM**: 这些 simplex latent distance baseline 表达力较强，部分数据集可接近 AICoG；本文优势在于用 Aitchison/ILR 给 simplex 表示加入更原则化的组合语义。
- **vs post-hoc graph explainability**: 常见 GNN explainer 解释特定预测或子图；AICoG 把可解释性嵌入表示空间本身，使距离、balance 和 subcomposition 都可解释。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 将 Aitchison 几何系统引入图角色嵌入很有辨识度，理论上还证明不损失 latent distance 表达力。
- 实验充分度: ⭐⭐⭐⭐☆ 链接预测、节点分类、合成恢复、interiority、basis 可视化和 subcomposition 分析都较完整；缺少与属性 GNN 的直接比较。
- 写作质量: ⭐⭐⭐⭐☆ 方法动机清晰，几何解释扎实；部分数学细节对非 compositional data 读者门槛较高。
- 价值: ⭐⭐⭐⭐☆ 对可解释图表示学习很有启发，尤其适合角色连续重叠、比例语义自然存在的网络分析场景。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](../../ICLR2026/graph_learning/towards_improved_sentence_representations_using_token_graphs.md)
- [\[ICLR 2026\] CORDS - Continuous Representations of Discrete Structures](../../ICLR2026/graph_learning/cords_-_continuous_representations_of_discrete_structures.md)
- [\[ACL 2025\] Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings](../../ACL2025/graph_learning/predicate-conditional_conformalized_answer_sets_for_knowledge_graph_embeddings.md)
- [\[ICML 2026\] T-GINEE: A Tensor-Based Multilayer Graph Representation Learning](t-ginee_a_tensor-based_multilayer_graph_representation_learning.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](../../ACL2026/graph_learning/agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
