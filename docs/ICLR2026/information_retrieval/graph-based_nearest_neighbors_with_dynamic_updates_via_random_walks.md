---
title: >-
  [论文解读] Graph-based Nearest Neighbors with Dynamic Updates via Random Walks
description: >-
  [ICLR 2026][信息检索/RAG][HNSW] 本文为 HNSW 图索引提出一个基于随机游走的全新理论框架，并据此设计出 SPatch：删除算法——删点后在其邻域上"成团再稀疏化"，在召回、查询速度、删除耗时、内存占用四个指标上同时取得良好折中。 领域现状：HNSW 是当前最主流的图式 ANN 索引…
tags:
  - "ICLR 2026"
  - "信息检索/RAG"
  - "HNSW"
  - "向量删除"
  - "随机游走"
  - "图稀疏化"
  - "击中时间"
  - "RAG"
---

# Graph-based Nearest Neighbors with Dynamic Updates via Random Walks

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=l97Kacqdfk](https://openreview.net/forum?id=l97Kacqdfk)  
**代码**: 待确认  
**领域**: 信息检索 / 近似最近邻搜索 (ANN)  
**关键词**: HNSW, 向量删除, 随机游走, 图稀疏化, 击中时间, RAG  

## 一句话总结
本文为 HNSW 图索引提出一个基于随机游走的全新理论框架，并据此设计出 **SPatch** 删除算法——删点后在其邻域上"成团再稀疏化"，在召回、查询速度、删除耗时、内存占用四个指标上同时取得良好折中。

## 研究背景与动机
**领域现状**：HNSW 是当前最主流的图式 ANN 索引，靠多层"长程边导航 + 短程边精搜"实现高吞吐高召回，是 RAG 检索的核心组件。它天生支持插入，却没有删除能力。

**现有痛点**：现实数据高度动态（广告下架、过季商品下架、旧版本商品淘汰），删除是刚需，但已有方案各有硬伤——

- **墓碑 (Tombstone)**：标记不删，召回最好但内存永不释放、删除越多查询越慢（多走很多步才到目标）；
- **不打补丁 (No patch)**：直接删点不修图，快但召回崩坏（尤其 mass deletion）；
- **局部重连 (Local reconnect)**：只在 $N(p)$ 内补边，召回略好于不打补丁但仍偏低；
- **FreshDiskANN / 全局重连**：用 2-hop 邻域或重插入提召回，但删除非局部、耗时长、难并行。

**核心矛盾**：召回、查询速度、删除耗时、内存这四个指标此前无法兼得，且大多数方案缺乏理论保证。

**本文目标**：在统一的理论框架下，给出一个四项指标都不拉胯、且有可证明保证的删除算法。

**核心 idea**（🎯 **随机游走孪生框架 + 击中时间保持的稀疏化**）：把 HNSW 的"贪心走到最近邻"软化成"按距离 softmax 概率随机游走"，从而把图视为加权图、把删除视为图稀疏化问题；删点后用 star-mesh 变换在邻域成团以精确保持游走概率，再按边权稀疏化，使删除前后的击中时间（到达目标的期望步数）近似不变。

## 方法详解

### 整体框架
SPatch 的关键洞察是给 HNSW 找一个"孪生"的随机游走解释：把贪心搜索替换为按 $\exp(-r^2\|q-v\|^2)$ 的 softmax 概率游走（称 softmax walk），于是 HNSW 图变成一张以查询 $q$ 为参数的加权图。在这个视角下，删除一个点 $p$ 等价于"消去 $p$ 这个节点而尽量保持游走统计量不变"——先用 star-mesh 变换把 $p$ 的邻域补成完全图（精确保持游走概率），再按边权采样稀疏化（近似保持击中时间），最后落地成确定性的 top-t 选边算法。

```mermaid
flowchart LR
    A[贪心搜索 HNSW] -->|softmax 软化| B[加权图上的随机游走<br/>w=exp -r²·dist²]
    B --> C[删点 p]
    C --> D[star-mesh 变换<br/>邻域成团 精确保持游走概率]
    D --> E[按边权稀疏化<br/>近似保持击中时间]
    E --> F[确定性落地<br/>取 top-t 重边]
```

### 关键设计

**1. Softmax walk：把贪心搜索翻译成随机游走** — 原始 HNSW 在邻域里确定性地走向 $\arg\min_{v\in N(u)}\|q-v\|$，本文把它"软化"为以概率 $\frac{\exp(-r^2\|q-v\|^2)}{\sum_{c\in N(u)}\exp(-r^2\|q-c\|^2)}$ 走向邻居 $v$。当 $r$ 足够大时，高斯核给近点指数级更高的概率，softmax walk 与硬贪心几乎等价（实验证实），但代价是整张图被重新解读为加权图：边 $\{u,v\}$ 看作两条有向边，权重 $w(u,v)=\exp(-r^2\|q-v\|^2)$ 只依赖终点和查询、与源点无关。这一步是后续全部理论的地基——它让"图搜索"变成"加权图上的随机游走"，从而能调用稀疏化与击中时间这套成熟工具。

**2. Star-mesh 变换：删点时精确保持游走概率** — 理论上删点的"最优"策略（定理 3.1）是删 $p$ 后在 $N(p)$ 上成团，能让搜索路径完全等价，但成团会让子图过密、每步距离计算暴涨。本文在随机游走模型下证明了更精细的版本（定理 4.1）：删 $p$ 后对所有 $u,v\in N(p)$ 重设新权重 $w'(u,v)=w(u,v)+\frac{w(u,p)\cdot w(p,v)}{\deg(p)}$，则新图上 $u\to v$ 的转移概率恰好等于旧图上"$u\to p\to v$ 或 $u\to v$"的概率。这正是电网理论里的 star-mesh 变换（源自 Schur 补 / 高斯消元），相当于把流经 $p$ 的概率质量精确地"焊接"回邻居之间。

**3. 按边权稀疏化 + 击中时间保证** — 成团后图变密，需要稀疏化。本文不走经典的有效电阻谱稀疏化，而用更简单的"按边权（即 $\sqrt{W}B$ 的平方行范数）采样"：采 $s=200\varepsilon^{-2}$ 条边即可让稀疏后的 Laplacian 满足 $\|\tilde C^\top\tilde C-L\|_F\le\varepsilon\cdot\mathrm{tr}[W]$（定理 4.3）。关键贡献是把这个 Frobenius 范数误差翻译成对**击中时间**（随机游走从 $u$ 到 $v$ 的期望步数）的乘性-加性误差界（定理 4.5）：删除前后到达目标的期望步数近似不变。直觉上，按边权采样天然让"近点簇"保持良好连通；而簇间的弱连接由 HNSW 上层负责，所以只在底层做稀疏化不会破坏可达性。推论 4.6 进一步说明在"单簇"和"多小簇"两种常见结构下，所需采样边数仅与 $|N(p)|$ 线性相关。

**4. 从随机采样到确定性 top-t 选边** — 既然 $r$ 足够大时采到 top-t 之外边的概率指数级小，就可以把"按权采 $t$ 条边"直接替换成"取权重最大的 top-t 条边"，避免无放回采样的开销。落地时还做了两点工程化：因 FAISS 的图是有向的，对每个出邻居 $u$ 从入邻居里选 $t=\alpha\cdot\lceil\frac{|L|+|R|}{|R|}\rceil$ 个使 $w'(u,v)$ 最大的 $v$；且保留删点前已有的边、只补充大权新边，轻度加密而不显著增边。全部实验都用这个确定性版本。

## 实验关键数据

**设置**：FAISS 的 `HNSWFlatIndex`（度数 $m=32$），抽取为 networkx 无向图做删除。4 个 1M 规模数据集：SIFT(128d)、GIST(960d)、MS MARCO 的 MPNet(768d) 与 MiniLM(384d)。Mass deletion 设定：累计删除 80% 的点，每删 0.8% 就用固定查询集评测一次。指标：top-10 召回、每查询距离计算次数（吞吐代理）、总删除耗时、底层边数（内存代理）。

### 主实验（删除算法四指标对比，定性总结自 Figure 2 与 Table 1）

| 方法 | 召回 | 查询速度 | 删除耗时 | 空间 |
|------|------|----------|----------|------|
| Tombstone | ✓ 最佳 | ✗ 慢(2.5–3× SPatch 的距离计算) | ✓ | ✗ 不释放 |
| No patch | ✗ 崩坏 | ✓ | ✓ | ✓ |
| Local reconnect | ✗ 偏低 | ✓ | ✓ | ✓ |
| FreshDiskANN (2-hop) | ✓ | ✓ | ✗ 慢 | ✓ |
| Global reconnect | ✓ | ✓ | ✗ 慢 | ✓ |
| **SPatch (本文)** | ✓ | ✓ | ✓ | ✓ |

SPatch 是唯一四项全部良好的方法：召回仅略低于墓碑/周期重建，但查询速度远快于墓碑、内存随删除量下降，删除速度也远快于 FreshDiskANN 和周期重建。

### 消融 / 验证实验（Softmax walk vs. 贪心搜索，Table 2）

| 数据集 | Softmax 召回 | Greedy 召回 | Softmax 距离计算 | Greedy 距离计算 |
|--------|--------------|-------------|------------------|------------------|
| MPNet | 89.98% | 91.60% | 1562 | 1722 |
| SIFT | 91.13% | 91.95% | 1049 | 1109 |
| GIST | 71.38% | 72.46% | 1606 | 1653 |
| MiniLM | 89.27% | 90.09% | 1708 | 1802 |

### 关键发现
- Softmax walk 与贪心搜索在召回、吞吐上高度一致（召回差 <1.7%，距离计算 softmax 反而更少），证明随机游走"孪生模型"是 HNSW 的合理理论代理。
- 击中时间虽不直接等价于"命中正确最近邻"（softmax walk 会停在局部极小），但实验显示它是设计稀疏化算法的好代理。
- 内存代理（底层边数）确实随真实内存占用同步下降（Appendix Figure 7）。

## 亮点与洞察
- **理论与实践的优雅桥接**：把工业界纯启发式的 HNSW 删除问题，提升到"加权图随机游走 + 图稀疏化"的可证明框架，star-mesh 变换 + 击中时间界给出了"为什么这样删"的解释。
- **softmax walk 的孪生视角本身有独立价值**：它给 HNSW 提供了一个可微/可分析的概率化解读，未来或可用于其他 HNSW 理论分析。
- **指标全面性**：跳出"只优化召回"的窠臼，把删除耗时、内存释放也纳入同一框架并实证四项全赢。

## 局限与展望
- **稀疏化不保证连通性**：按边权采样可能丢弃簇间弱边，本文靠"只在底层稀疏化、簇间交给上层"绕开，但这是一个假设而非保证；查询若严重漂移向被删簇时风险未知。
- **击中时间 ≠ 召回**：理论保的是击中时间，softmax walk 会停在局部极小，所以理论与最终召回之间仍有 gap，只能靠实验佐证击中时间是好代理。
- **场景偏向 mass deletion**：主打大批量删除且不插入，插入/删除混合的真实在线负载、增量更新场景尚未充分评测。
- **实现链路较重**：从 FAISS 抽图转 networkx 做删除，工程上偏研究原型，生产级集成与并行化还需打磨。

## 相关工作与启发
- **图式 ANN 与 HNSW 理论**：Malkov & Yashunin (HNSW)、DiskANN、以及一系列对贪心搜索可证明性的分析（Prokhorenkova & Shekhovtsov、Indyk & Xu 的 α-shortcut reachability 等），本文在其上补齐了"删除"的理论拼图。
- **HNSW 删除算法谱系**：FreshDiskANN(Singh 等)、global/local reconnect(Xu 等)，本文统一归纳为"删点 + 邻域打补丁"，并指出其缺乏理论保证、mass deletion 召回/速度双输。
- **图稀疏化与随机游走**：谱稀疏化（有效电阻采样，Spielman & Srivastava）、Laplacian 求解器；本文改用更简单的行范数采样（Drineas & Kannan）换取实现效率，保证较弱但够用。这条"用稀疏化做动态图维护"的思路对其他动态图数据结构（增量图神经网络索引、动态社交图）也有启发。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个为 HNSW 删除提供随机游走理论框架并落地成实用算法，star-mesh + 击中时间保持的组合很巧。
- **实验充分度**: ⭐⭐⭐⭐ 4 个 1M 数据集、四项指标、mass deletion 全程曲线 + softmax/greedy 对照，扎实；但混合负载与生产级测试欠缺。
- **写作质量**: ⭐⭐⭐⭐ 动机—理论—落地—实验逻辑清晰，定理与算法对应明确，理论部分稍密集。
- **价值**: ⭐⭐⭐⭐⭐ 直击 RAG/向量库的真实刚需（动态删除），既有理论又有四项全赢的工程折中，落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Flow of Spans: Generalizing Language Models to Dynamic Span-Vocabulary via GFlowNets](flow_of_spans_generalizing_language_models_to_dynamic_span-vocabulary_via_gflown.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)
- [\[ICLR 2026\] Query-Aware Flow Diffusion for Graph-Based RAG with Retrieval Guarantees](query-aware_flow_diffusion_for_graph-based_rag_with_retrieval_guarantees.md)
- [\[ICLR 2026\] LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora](linearrag_linear_graph_retrieval_augmented_generation_on_large-scale_corpora.md)
- [\[ICLR 2026\] Youtu-GraphRAG: Vertically Unified Agents for Graph Retrieval-Augmented Complex Reasoning](youtu-graphrag_vertically_unified_agents_for_graph_retrieval-augmented_complex_r.md)

</div>

<!-- RELATED:END -->
