---
title: >-
  [论文解读] Forest-Based Graph Learning for Semi-Supervised Node Classification
description: >-
  [ICLR 2026][图学习][生成树] 把图上的消息传递重新诠释为「在若干棵生成树（森林）上的传输」，用同配性引导采样高质量树 + 线性时间的树聚合器，在 O(n+m) 复杂度下实现全局感受野，半监督节点分类上同时打败深层 GNN 和图 Transformer。 领域现状：图神经网络要捕获长程知识…
tags:
  - "ICLR 2026"
  - "图学习"
  - "生成树"
  - "森林"
  - "长程信息传播"
  - "同配性"
  - "线性复杂度"
  - "Transformer"
---

# Forest-Based Graph Learning for Semi-Supervised Node Classification

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5asbtzIVpS](https://openreview.net/forum?id=5asbtzIVpS)  
**代码**: [https://anonymous.4open.science/r/FGL/](https://anonymous.4open.science/r/FGL/)  
**领域**: 图学习 / 半监督节点分类  
**关键词**: 生成树, 森林, 长程信息传播, 同配性, 线性复杂度, 图 Transformer  

## 一句话总结
把图上的消息传递重新诠释为「在若干棵生成树（森林）上的传输」，用同配性引导采样高质量树 + 线性时间的树聚合器，在 O(n+m) 复杂度下实现全局感受野，半监督节点分类上同时打败深层 GNN 和图 Transformer。

## 研究背景与动机
**领域现状**：图神经网络要捕获长程知识，主流有两条路——深层局部模型（堆叠多层 GNN，如 GCNII）靠层数扩大感受野，浅层全局模型（图 Transformer，如 Graphormer/DIFFormer）靠 1~2 层全局注意力一次性覆盖所有节点对。

**现有痛点**：两条路都被复杂度卡死。深层 GNN 层数多、串行不可并行、易过平滑；图 Transformer 是二次复杂度 $O(n^2)$，大图直接 OOM。后续用稀疏化（自适应选择、图重连）压复杂度的工作，又要牺牲全局覆盖、依赖复杂的选择策略，可能丢掉重要的节点交互。

**核心矛盾**：作者把它抽象成一个公式——**总开销 = 单结构开销 × 结构数量**。局部原语（一阶邻域、短随机游走）单结构便宜但要海量结构才能覆盖远距离；全局算子结构少但单结构开销爆炸。两个因子按下葫芦浮起瓢，是现有范式的固有缺陷。

**本文目标**：找一种**同时压住两个因子**的中间层级结构，打破「性价比 vs 全局感受野」的死结。

**核心 idea**：**[生成树即最稀疏的全局结构]** 生成树是连接所有节点的最小子图，在结构数量受限时它是「最简单却能全局覆盖」的结构；单棵树知识不全，就用**一片森林（多棵互补的生成树）**。把图学习重新表述为「在森林上做信息传输」，并用**同配性引导采样**保证树的质量、用**线性时间树聚合器**保证效率。

## 方法详解

### 整体框架
FGL（Forest-based Graph Learning）由四步串成一条流水线：**预处理**增广原图保证连通并提升同配率 → **树采样器**从一个偏向高同配树的分布里采出 $N_T$ 棵生成树 → **树聚合器**在每棵树上线性时间地传播全局消息 → **树融合器**把多棵树的结果与局部信息融合成最终嵌入，喂给线性分类器。

```mermaid
flowchart LR
    G[原始图 G] --> P[预处理<br/>伪标签+kNN增广]
    P --> Ghat[增广图 Ĝ<br/>连通且更同配]
    Ghat --> HE[同配性估计器<br/>局部注意力打边分]
    HE --> S[树采样器<br/>Wilson 算法采 N_T 棵树]
    S --> A[树聚合器<br/>两递推 O n 传播]
    A --> F[树融合器<br/>RowNorm+Mean+局部残差]
    F --> H[最终嵌入 H''<br/>线性预测]
```

### 关键设计

**1. 同配性引导的树采样器：让森林天然偏向「同类相连」的树。** 既然目标是节点分类，好树的标准就是**高同配率**（树上的边尽量连同类节点）+ **多样性**（树之间别太重叠，否则森林退化成单棵树）。作者把树的得分定义成边得分之积，从而在图上定义树分布 $P_{\hat G}(T)=\prod_{e\in T}s(e)\big/\sum_{T\subseteq\hat G}\prod_{e\in T}s(e)$，只要给同配边高分、异配边低分，分布就自动偏向高同配树。边得分由一个**局部注意力同配性估计器**给出：$\alpha_{i\to j}=\mathrm{softmax}_j(Q_iK_j^\top/\sqrt c)$，在一阶邻域内用伪标签 $Y'$ 监督训练，边得分取 $s(e)=(\alpha_{i\to j}+\alpha_{j\to i})/2$。最后用带权 **Wilson 算法**以近 $O(n)$ 每棵树的代价独立采出 $N_T$ 棵树。理论上（Theorem 2）作者证明：随着分数比 $\Delta=p/q$（同配边分 / 异配边分）增大，采样分布的期望同配率单调上升并趋向由图的同配连通分量数决定的上界——**估计器越准，诱导出的树分布质量越高**，给整套采样策略提供了可证明的支撑。

**2. 线性时间树聚合器：用两个递推把二次的节点对交互压成线性。** 核心观察是：树上相邻两点 $u,v$ 的全局聚合消息**只在一条边的方向上有差异**。因此只要消息聚合器 $f_{\mathrm{Agg}}$ 满足「可组合 / 可拆解」两个性质——存在算子 $\mathcal M_+/\mathcal M_-$ 使得合并后能还原（Property I/II），就能用两个递推完成全树传播（Theorem 1）：自底向上算每个节点子树的聚合消息 $S_u=f_{\mathrm{Agg}}(\{S_v\}_{v\in \mathrm{Child}(u)}\cup\{g(H_u)\})$（Recursion I），再自顶向下用 $\mathcal M_-$ 减掉自身贡献、$\mathcal M_+$ 加上父节点信息得到 $H'_v=\mathcal M_+(S_v,\mathcal M_-(H'_{\mathrm{Fa}(v)},S_v))$（Recursion II）。实现上取加权和/加权差作为算子的线性变体：

$$S_u=\sum_{v\in\mathrm{Child}(u)}(\alpha_{v\to u}W_A)S_v+W_B H_u,\quad H'_v=S_v+\alpha_{\mathrm{Fa}(v)\to v}W_A\big(H'_{\mathrm{Fa}(v)}-\alpha_{v\to\mathrm{Fa}(v)}W_A S_v\big)$$

注意力权重 $\alpha$ 复用同配性估计器，强化同配边、弱化异配边。这一设计的妙处在于：它**用线性运行时实现了等价于二次节点对交互的全局传播**，且既能在树间并行、也能在单树内并行（把树重构成「矮而宽」、以质心为根可支持更多线程）。该框架还兼容线性注意力、线性 RNN、状态空间模型（SSM）等多种聚合器，通用性强。

**3. 树融合器：补回局部知识 + 多树取平均稳住表示。** 单棵生成树太稀疏会丢局部结构，于是先从输入特征算一份局部信息 $H=(\beta_1\hat A_{\hat G}+\beta_2\alpha+(1-\beta_1-\beta_2)I)^{K_L}XW_H$（$K_L\le2$ 的低阶传播）。对 $N_T$ 棵树各跑一遍树聚合器得到 $H'^{(k)}$，先对每行做 L2 归一化做数值稳定，再取均值得全局信息 $H'=\mathrm{Mean}\{\mathrm{RowNorm}(H'^{(k)})\}$；最后用残差系数 $\gamma$ 平衡全局与局部：$H''=(1-\gamma)H'+\gamma H$，作为最终节点嵌入。整套训练每个 epoch 仅 $O((n+m)Kd)$ 时间与空间，且可并行。

## 实验关键数据

### 主实验表格
9 个数据集（同配：Cora/Citeseer/Pubmed/ArXiv；异配：Actor/Cornell/Texas/Wisconsin/Flickr）半监督节点分类准确率（%），对比 26 个基线，**Ours 平均排名 1.22**：

| Method | 类别 | Cora | Citeseer | Pubmed | Actor | Cornell | Texas | Wisconsin | ArXiv | Flickr | Avg.Rank |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GCNII | DeepGNN | 85.34 | 73.24 | 79.88 | 34.64 | 74.61 | 69.19 | 70.31 | 51.91 | 41.79 | 8.78 |
| SGFormer | GT | 82.38 | 71.82 | 80.64 | 37.80 | 68.65 | 78.92 | 80.00 | 45.73 | 40.13 | 7.22 |
| DIFFormer | GT | 83.32 | **74.46** | 78.16 | 34.51 | 60.00 | 68.11 | 63.92 | 53.60 | 44.25 | 10.56 |
| GraphMamba | Mamba | 54.36 | 58.98 | 70.90 | 36.05 | 74.05 | 77.29 | 80.39 | 33.59 | 42.30 | 13.89 |
| **Ours** | **Forest** | **85.46** | 74.42 | **81.00** | **39.88** | **83.24** | **91.89** | **86.27** | **56.47** | **47.22** | **1.22** |

相对增益：对 GT/DIFFormer/GCN/GCNII 平均准确率分别提升 16.2% / 16.1% / 24.5% / 11.9%；Wisconsin 上对它们分别拉开 20.2% / 35.0% / 50.7% / 22.7%。异配图（Cornell/Texas/Wisconsin）优势尤其夸张。

### 消融实验表格

| No. | 变体 | Cora | Pubmed | Actor | Texas | Wisconsin | Flickr |
|---|---|---|---|---|---|---|---|
| (1) | w.o. 全局子模块 | 80.00 | 76.13 | 34.73 | 82.88 | 83.92 | 39.63 |
| (2) | w.o. 局部子模块 | 82.18 | 77.48 | 35.08 | 69.93 | 75.49 | 32.17 |
| (3) | 均匀采样树 | 83.63 | 78.45 | 36.13 | 82.58 | 84.80 | 42.77 |
| (4) | 单棵同配引导树 | 83.73 | 78.55 | 36.32 | 84.83 | 85.29 | 42.96 |
| (5) | **FGL-完整** | **85.46** | **81.00** | **39.88** | **91.89** | **86.27** | **47.22** |

(4) vs (3)：单棵同配引导树就胜过多棵随机树 → 同配引导采样是关键；(5) vs (4)：森林一致优于单树 → 多树捕获互补拓扑知识；(1)(2) 掉点 → 全局/局部子模块都不可少。

### 关键发现
- **效率（sec/epoch）**：ArXiv 上 Ours 仅 0.246s，ANS-GT 24.5s、GOAT 58.8s；对 DIFFormer/GCNII 也有 2~5 倍加速，多数图比基线更快，且多个 GT 在 ArXiv/Flickr 上直接 OOM。
- **树数量**：6~10 棵树是各数据集的最优区间，先升后稳/降，说明少量树就能覆盖图的本质结构，更多树边际收益递减并有冗余风险，印证「结构数量少 → 低总开销」。
- **估计器精度**：随同配边平均得分 $p$ 提高，准确率单调上升，验证 Theorem 2 的「估计器越准 → 树分布越好」。

## 亮点与洞察
- **一个公式戳破范式困境**：把所有图学习范式抽象成「单结构开销 × 结构数量」，再用生成树这个「最稀疏的全局结构」同时压住两项，视角干净有力。
- **理论与工程闭环**：Theorem 2 证明「估计器准度 → 树分布同配率」的渐近关系，Theorem 1 给出线性传播的两递推，估计器精度实验直接验证理论，少见的「能证、能跑、能验」三位一体。
- **聚合器的通用性**：树聚合器只要求 $f_{\mathrm{Agg}}$ 满足组合/拆解性质，因此能即插即用线性注意力、线性 RNN、SSM 等，框架可扩展性强。
- **同配/异配通吃**：在异配图（Texas 91.89、Wisconsin 86.27）上的优势远超同配图，说明同配引导采样确实能在「难图」上挑出有用的结构路径。

## 局限与展望
- **依赖伪标签与同配性估计器的质量**：预处理用伪标签做 kNN 增广、采样靠估计器打分，若伪标签噪声大或估计器在极端异配/标签极稀缺下失准，树分布质量会被拖累（理论上界也受图本身同配连通分量数限制）。
- **超参较多**：$N_T$、$\beta_1,\beta_2$、$K_L$、$\gamma$、kNN 的 $k$ 都需调，跨数据集泛化的鲁棒性有待更大规模验证。
- **任务范围**：目前只在半监督节点分类上验证，图分类、链路预测、异常检测等任务上森林范式是否同样有效是自然的延伸方向。
- **采样开销与可复现性**：Wilson 算法虽近线性，但每棵树都要采，超大规模图（十亿级边）上的端到端开销与方差仍需考察。

## 相关工作与启发
- **深层局部模型**（GCNII、DropEdge、各类 norm）：靠堆层 + 抗过平滑技巧扩感受野，本文指出其串行不可并行是固有瓶颈。
- **浅层全局模型 / 图 Transformer**（Graphormer、SAN、NodeFormer、Exphormer、SGFormer、DIFFormer）：直接建模全局节点对交互但二次复杂度，稀疏化版本牺牲结构偏置。
- **生成树 / 随机游走类**：本文把「生成树是最小连通子图」的经典图论事实用作长程传播的载体，并借 Wilson 算法做均匀/带权生成树采样，是把组合数学工具引入 GNN 范式的好例子。
- **启发**：当一个领域陷入「A 与 B 不可兼得」的 trade-off 时，与其在两端做近似，不如回到开销的第一性原理找一个能同时控制两个因子的中间结构——FGL 的「森林」就是这种思路的范本。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 「图学习 = 森林上的传输」是真正的新范式，配套的同配引导采样 + 线性树聚合器自成体系，不是增量改进。
- **实验充分度**: ⭐⭐⭐⭐ 26 个基线 × 9 数据集 + 效率/消融/超参/估计器精度全覆盖，10 次初始化报均值；略缺更大图（十亿边）与节点分类外任务的验证。
- **写作质量**: ⭐⭐⭐⭐ 「总开销公式」的动机推导清晰，理论—方法—实验闭环；公式密度高，部分实现细节挪到附录，初读需对照图 2/图 3。
- **价值**: ⭐⭐⭐⭐⭐ 同时在性能和效率上胜过深层 GNN 与图 Transformer，且范式可扩展到多种聚合器与任务，对长程图学习有方法论级别的启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Geometric Imbalance in Semi-Supervised Node Classification](../../NeurIPS2025/graph_learning/geometric_imbalance_in_semi-supervised_node_classification.md)
- [\[ICLR 2026\] Learning Posterior Predictive Distributions for Node Classification from Synthetic Graph Priors](learning_posterior_predictive_distributions_for_node_classification_from_synthet.md)
- [\[ICLR 2026\] Topological Anomaly Quantification for Semi-Supervised Graph Anomaly Detection](topological_anomaly_quantification_for_semi-supervised_graph_anomaly_detection.md)
- [\[ICLR 2026\] Low-Rank Few-Shot Node Classification by Node-Level Graph Diffusion](low-rank_few-shot_node_classification_by_node-level_graph_diffusion.md)
- [\[ICLR 2026\] HarmonyGNNs: Harmonizing Heterophily and Homophily in GNNs via Self-Supervised Node Encoding](harmonygnns_harmonizing_heterophily_and_homophily_in_gnns_via_self-supervised_no.md)

</div>

<!-- RELATED:END -->
