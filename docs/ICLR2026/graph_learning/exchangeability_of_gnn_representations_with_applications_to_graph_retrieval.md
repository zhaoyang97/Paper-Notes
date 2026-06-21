---
title: >-
  [论文解读] Exchangeability of GNN Representations with Applications to Graph Retrieval
description: >-
  [ICLR 2026][图学习][可交换性 (Exchangeability)] 本文发现并证明了一个全新的概率对称性——经标准训练的 GNN 节点嵌入在维度轴上是**可交换随机变量**，并利用这一性质把高维传输相似度近似成一维排序后的欧氏相似度，从而首次为非对称图相似度设计出统一的 LSH 检索框架 GraphHash。
tags:
  - "ICLR 2026"
  - "图学习"
  - "可交换性 (Exchangeability)"
  - "图神经网络"
  - "局部敏感哈希 (LSH)"
  - "最优传输相似度"
  - "图检索"
---

# Exchangeability of GNN Representations with Applications to Graph Retrieval

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=HQcCd0laFq](https://openreview.net/forum?id=HQcCd0laFq)  
**代码**: [https://rebrand.ly/graphhash](https://rebrand.ly/graphhash)  
**领域**: 图学习 / 图检索 / 概率对称性  
**关键词**: 可交换性 (Exchangeability), GNN 表示, 局部敏感哈希 (LSH), 最优传输相似度, 图检索  

## 一句话总结
本文发现并证明了一个全新的概率对称性——经标准训练的 GNN 节点嵌入在维度轴上是**可交换随机变量**，并利用这一性质把高维传输相似度近似成一维排序后的欧氏相似度，从而首次为非对称图相似度设计出统一的 LSH 检索框架 GraphHash。

## 研究背景与动机
- **领域现状**：神经图检索的目标是从大规模语料图库 $\mathcal{C}=\{G_c\}$ 中找出与查询图 $G_q$ 最相关的 top-b 个图。近年来工作已确认，基于**最优传输 (transportation)** 的节点嵌入相似度（如子图匹配的 hinge 距离、图编辑距离 GED）在准确率上显著优于单向量池化和图核方法。
- **现有痛点**：传输相似度虽准，却**极不适合索引与检索**。高维 $D>1$ 下计算传输距离的精确算法是 $O(n^3)$，近似的 Sinkhorn 迭代也要 $O(n^2)$；更棘手的是，子图匹配、GED 等图相似度天生**非对称**，而现有 LSH（L1 网格投影、L2 线投影）只支持对称欧氏距离，导致海量图库无法做亚线性时间检索。
- **核心矛盾**：检索要"快"就得用对称欧氏度量 + 可哈希结构，要"准"就得用非对称的高维传输度量——两者长期不可兼得。
- **本文目标**：找到一座桥梁，把昂贵的高维非对称传输相似度，无损（高概率）地降维成可哈希的一维欧氏相似度，从而同时拿到准确率与亚线性检索速度。
- **核心 idea**：**「可交换性」(Exchangeability)**。作者证明：在标准 iid 初始化 + 置换不变损失 + 常见优化器（SGD/Adam）下，训练后的 GNN 节点嵌入矩阵 $X\in\mathbb{R}^{n\times D}$ 的各列 $X[:,1],\dots,X[:,D]$ 是可交换随机变量，即 $p(X)=p(X\pi)$。这意味着每个嵌入维度同分布，于是**任意单个维度都能给出整体传输相似度的浓度估计**，把 $D$ 维问题降到 1 维排序匹配。

## 方法详解

### 整体框架
本文是"先证一个性质、再用这个性质"的两段式结构：第一段在理论上确立 GNN 嵌入的可交换性（§3），第二段把它转化为图检索的实用 LSH 算法 GraphHash（§4）。证明走"参数置换变换 $\Gamma_\pi$ → 梯度等变 → 参数密度不变 → 嵌入可交换"四步链条；应用则走"用同分布把传输相似度降到一维排序相似度 → 用傅里叶特征把排序相似度写成内积 → 用随机超平面哈希"三步落地。

```mermaid
flowchart TD
    A[iid 初始化参数 θ₀<br/>p(θ₀)=p(Γπ θ₀)] --> B[Lemma 2: 构造置换诱导变换 Γπ<br/>Xπ = GNN_Γπ(θ)(G)]
    B --> C[Lemma 3: 损失置换不变 ⇒ 梯度等变]
    C --> D[Lemma 4: 训练轨迹中参数密度不变<br/>p(θt)=p(Γπ θt)]
    D --> E[Theorem 5: 嵌入可交换<br/>p(X)=p(Xπ)]
    E --> F[Prop 7: 单维相似度浓度<br/>sim/D ≈ simd]
    F --> G[Prop 9: 傅里叶特征<br/>simd ∝ cos(T̂q,d, T̂c,d)]
    G --> H[随机超平面哈希<br/>GraphHash 索引/查询]
```

### 关键设计

**1. 用置换诱导变换 $\Gamma_\pi$ 把"嵌入可交换"归约为"参数对称"。** 要证明嵌入各维可交换，直接分析非线性训练动力学很难。作者的巧思是把维度置换 $\pi$ "搬到"参数空间：对两层 MLP $x=\sigma(\sigma(\text{feat}\,\Psi)\Theta)$，只要令 $\Gamma_\pi(\theta):=\theta\,\mathrm{Diag}(I,\pi)$ 就能让输出从 $x$ 变成 $x\pi$。对 GNN，由于消息传递把置换在层间纠缠在一起，构造更复杂，但 Lemma 2 证明对 gated GNN、GIN、GAT、GCN 乃至图 Transformer 都存在这样一个双射 $\Gamma_\pi$，且其雅可比行列式 $|\det(\partial\Gamma_\pi(\theta)/\partial\theta)|=1$（保测度）。这把"嵌入维度可交换"等价转化为"参数密度在 $\Gamma_\pi$ 下不变"。

**2. 用"梯度等变 + 初始 iid"把对称性从初始化传播到任意训练步。** 训练起点 $t=0$ 时参数 iid，天然有 $p(\theta_0)=p(\Gamma_\pi(\theta_0))$。关键问题是训练会不会破坏它。由于损失对嵌入维度置换不变（这点在图检索里自动成立，因为相似度只依赖传输方案），损失景观在 $\Gamma_\pi$ 下对称，于是梯度等变（Lemma 3）：置换初始参数等价于置换整条训练轨迹，$\theta_0\mapsto\Gamma_\pi(\theta_0)\Rightarrow\theta_t\mapsto\Gamma_\pi(\theta_t)$。结合保测度性质，Lemma 4 得 $p(\theta_t)=p(\Gamma_\pi(\theta_t))$ 对所有 $t\ge0$ 成立，最终 Theorem 5 推出 $p(X)=p(X\pi)$。一个直接推论是：跨随机种子平均的期望嵌入矩阵 $\mathbb{E}[X]$ 坍缩为**秩一矩阵**——这与 GNN 过平滑、Transformer 秩坍缩等现象遥相呼应。

**3. 用同分布把高维传输相似度降成一维排序后的欧氏相似度。** 作者把多种图相似度统一成一个凸、可分解、可非对称的代价函数 $\rho(x)=\sum_d\rho(x[d])$：$\rho(\cdot)=[\cdot]_+$ 即子图匹配 hinge 距离，$\rho(\cdot)=e_\ominus[\cdot]_++e_\oplus[-\cdot]_+$ 即带非对称增删代价的 GED。整体传输相似度为
$$\mathrm{sim}(G_c,G_q)=\max_{P\in\mathcal{P}_n}\sum_{u,u'}\sum_{d\in[D]} s\big(x^{(q)}(u)[d]-x^{(c)}(u')[d]\big)\cdot P[u,u'].$$
由 Proposition 6（联合可交换）可知各维同分布，于是任取单维 $d$ 的相似度 $\mathrm{sim}_d$ 就是整体的无偏缩放估计。而单维传输问题退化为标量匹配——配合 $s$ 凹性，可化简为对**排序后向量**的相似度：$\mathrm{sim}_d(G_c,G_q)=s\big(\mathrm{SORT}(X^{(q)}[:,d])-\mathrm{SORT}(X^{(c)}[:,d])\big)$。Proposition 7 给出浓度保证：取 $D>\frac{1}{\epsilon^2\delta}$，则 $\Pr(|\mathrm{sim}/D-\mathrm{sim}_d|\le\epsilon)\ge1-\beta_0\delta$。

**4. GraphHash：用傅里叶特征 + 随机超平面把非对称相似度变成可哈希内积。** 一维排序相似度 $\mathrm{sim}_d=s(\cdot)$ 形似平移核 $\kappa(x-x')$ 但因非对称不是真核，无法直接套 Rahimi-Recht 随机傅里叶特征。作者推广 Roy et al. (2023)，Proposition 9 证明存在 $\mathbb{R}^4$ 傅里叶特征向量使 $\mathrm{sim}_d=\sum_u\int F_{q,d}(\iota\omega_u)^\top F_{c,d}(\iota\omega_u)\,d\omega_u$；以 $p(\omega)\propto|S(\iota\omega)|$（评分函数傅里叶变换的幅值）采样 $M$ 个频率得有限维向量 $\hat T_{\bullet,d}\in\mathbb{R}^{4nM}$，并满足 $\mathrm{sim}_d\propto\cos(\hat T_{q,d},\hat T_{c,d})$。最后用随机超平面 $W[r,t]\sim\mathcal{N}(0,1)$ 得哈希码 $h^{(d)}(G_c)=\mathrm{sign}(W\hat T_{c,d})$，构成对原始非对称传输相似度的**有效 (asymmetric) LSH**——这是首个支持子图匹配、一般代价 GED 等多种非对称图相似度的统一 LSH 框架。

## 实验关键数据

### 主实验（检索性能，MAP vs. 检索图数）
- 数据集：TUDatasets 中 ptc-fr / ptc-fm / cox2 / ptc-mr，每个 500 查询图 + 100,000 语料图。
- 两类非对称监督信号：子图匹配 (SM, 用 VF2) 与 GED（非对称代价 $e_\oplus=1$ 插入、$e_\ominus=2$ 删除）。
- 对比 5 个 ANN 基线：FourierHashNet、Random Hyperplanes (RH)、IVF (FAISS/ColBERT 风格)、DiskANN (HNSW)、Random。

| 方法 | SM 任务 | GED 任务 | 备注 |
|------|---------|----------|------|
| **GraphHash (本文)** | **全面最优** | **全面最优** | 唯一覆盖完整选择度谱 |
| FourierHashNet | 次优，但 SM 上 MAP 在 ptc-fr/ptc-mr 低于 50% 穷举上限 | 次优 | 单向量池化，选择度受限 |
| RH | 比 random 还差（对称余弦不适配 containment） | 偶尔追平 GraphHash 但方差大 | 难调参 |
| IVF / DiskANN | 差 | 差 | 多向量但依赖 L2/cosine 对称度量 |
| Random | 显著最低 | 显著最低 | 下界参照 |

### 可交换性验证实验
- 在 cox2 子集（1,024 查询-语料对）上独立训练 **5,000 个** GNN（$D=10$，Adam，20 epoch，ranking loss）。
- **边际分布**：固定节点 $v$ 的 $X^{(c)}[v,d]$ 在 $d=1,5,10$ 上的经验密度，从初始化到 epoch 20 始终重合 → 各维同分布。
- **MMD 直接检验**：采样 100 个置换计算 $\mathrm{MMD}^2$，cox2 上 GED 为 $-3.89\times10^{-5}\pm2.69\times10^{-5}$、SM 为 $-1.18\times10^{-6}\pm3.28\times10^{-5}$，极小 → $p_X$ 与 $p_{X\pi}$ 几乎相同。
- **秩一验证**：均值嵌入 $\mathbb{E}[X]$ 的最大奇异值占比 $\sigma_1^2/\sum_i\sigma_i^2$ 随初始化数增加收敛到 1 → 期望嵌入矩阵秩为 1。
- **训练全程稳定**：从初始化到 epoch 8、epoch 20，密度图与 MMD 检验都未见对称性破坏，说明该性质对训练阶段鲁棒。

### 关键发现
1. 可交换性不仅初始化时成立，**经过非凸损失反向传播后依然保持**——这是论文最反直觉、也最关键的实证结论。
2. 哈希位数 $\dim_h\approx10$ 时 AUC 取得最优 MAP-检索量权衡，再增大收益递减。
3. 多向量索引 (IVF/DiskANN) 在非对称监督下反而失效，印证"对称度量 ≠ 图相似度"的核心痛点。
4. RH 在 SM 任务上甚至差于随机检索，说明对池化向量做余弦哈希完全不适配子图包含 (containment) 这类非对称查询。
5. GraphHash 是唯一能覆盖完整选择度谱（从高召回到高精度）的方法，FourierHashNet 在 ptc-fr/ptc-mr 上 MAP 早早封顶。

## 亮点与洞察
- **概率对称性 vs. 代数对称性**：以往权重空间对称（Hecht-Nielsen 的置换不变、模型合并等）都是脱离训练的代数性质，本文首次刻画训练中**自然涌现**的概率对称性，把"随机初始化 + 等变梯度"作为对称性的来源，理论视角新颖。
- **覆盖面广**：可交换性对 gated GNN、GIN、GAT、GCN、图 Transformer 等广谱架构，以及 SGD/Adam 等常见优化器、Kaiming/Xavier 等标准初始化都成立，不是个别模型的巧合。
- **一个性质打通理论与系统**：可交换性既能解释 GNN/Transformer 的秩坍缩（独立的理论价值），又能直接落地为亚线性检索算法，理论-应用闭环漂亮。
- **首个非对称图相似度的 LSH**：把子图匹配、一般代价 GED 等异质度量统一进一个 LSH 框架，填补了"非对称传输相似度无法哈希"的空白。
- **降维即降复杂度**：从 $O(n^3)$/$O(n^2)$ 的高维传输求解，降到一维排序（$O(n\log n)$）+ 哈希查桶，可扩展性实质提升。
- **保测度构造的优雅**：$\Gamma_\pi$ 的雅可比行列式恒为 1，意味着置换变换不改变参数空间的概率测度，这是把"初始对称"无损传递到"训练后对称"的数学关键。

## 局限与展望
- **依赖置换不变损失**：核心证明建立在损失对嵌入维度置换不变之上；虽然作者在附录指出可放宽到"联合变换不变"（中间表示置换 + 后续层参数置换的组合），但对完全任意损失的普适性仍需更多验证。
- **秩一坍缩的双刃剑**：期望嵌入秩一既是可交换性的证据，也提示单次训练模型的表达可能受概率对称性约束，论文未深入探讨它与表示能力/过平滑的定量关系。
- **数据规模偏小**：可交换性验证用 cox2 小子集 + 5000 次重训，检索实验数据集均为分子图 (TUDatasets)，在大规模异构图、社交网络等场景的有效性有待检验。
- **近似误差与 $D$ 的权衡**：Proposition 7 要求 $D>\frac{1}{\epsilon^2\delta}$ 才能保证浓度，低维嵌入下近似质量、以及频率采样数 $M$ 对哈希精度的影响可进一步刻画。

## 相关工作与启发
- **权重空间对称性**：Hecht-Nielsen (1990) 的 MLP 置换不变、Godfrey et al. (2022) 对不同激活的对称刻画、模型合并 (Ainsworth et al., 2022; Peña et al., 2022) 等——本文把这条线从"代数/静态"推进到"概率/训练涌现"。
- **秩坍缩现象**：与 Transformer 输出秩坍缩 (Dong et al., 2023)、状态空间模型 (Joseph et al., 2025)、GNN 过平滑 (Roth et al., 2024) 互为印证，可交换性提供了统一的概率解释视角。
- **图检索与传输相似度**：Roy et al. (2022) 的子图匹配 hinge 距离、Jain et al. (2024)/Zhuo et al. (2022) 的 GED，以及 FourierHashNet (Roy et al., 2023) 的 hinge-LSH——本文在 Roy et al. (2023) 基础上推广到任意非对称相似度。
- **随机傅里叶特征**：Rahimi & Recht (2007) 用有限维傅里叶特征近似平移核 $\kappa(x-x')$，本文将其从对称核推广到非对称评分函数 $s(\cdot)$，是 GraphHash 可哈希化的技术基石。
- **启发**：可交换性提示"训练涌现的概率对称性"可能是一类通用工具——凡是损失对某种内部置换不变的网络，都可能存在类似的同分布结构，可用于降维、压缩、近似检索等下游任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ — 首次发现并严格证明 GNN 嵌入训练涌现的可交换性，并把它转化为检索算法，理论与应用双重原创。
- **实验充分度**: ⭐⭐⭐⭐ — 5000 次重训验证可交换性 + 四数据集两类非对称监督的检索对比扎实；但数据集偏分子图、规模偏小。
- **写作质量**: ⭐⭐⭐⭐ — 四步证明链条清晰、理论到应用过渡自然；部分公式与傅里叶特征推导较密，需对 LSH/最优传输有背景。
- **价值**: ⭐⭐⭐⭐⭐ — 既给出 GNN/Transformer 秩坍缩的统一概率解释，又交付首个支持非对称图相似度的可扩展 LSH 框架，理论意义与实用价值兼具。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] GNN-as-Judge: Unleashing the Power of LLMs for Graph Learning with GNN Feedback](gnn-as-judge_unleashing_the_power_of_llms_for_graph_learning_with_gnn_feedback.md)
- [\[ICLR 2026\] On The Expressive Power of GNN Derivatives](on_the_expressive_power_of_gnn_derivatives.md)
- [\[ICLR 2026\] Towards Improved Sentence Representations using Token Graphs](towards_improved_sentence_representations_using_token_graphs.md)
- [\[ICLR 2026\] Directed Semi-Simplicial Learning with Applications to Brain Activity Decoding](directed_semi-simplicial_learning_with_applications_to_brain_activity_decoding.md)
- [\[ICLR 2026\] CORDS - Continuous Representations of Discrete Structures](cords_-_continuous_representations_of_discrete_structures.md)

</div>

<!-- RELATED:END -->
