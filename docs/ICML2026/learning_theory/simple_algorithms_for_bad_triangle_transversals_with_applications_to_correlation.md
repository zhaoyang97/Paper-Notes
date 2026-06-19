---
title: >-
  [论文解读] Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering
description: >-
  [ICML 2026 Spotlight][算法理论 / 近似算法 / 图聚类][坏三角形覆盖] 本文为有符号图上的"坏三角形覆盖"问题（Bad Triangle Transversal, BTT）给出两个仅需单次解 LP 的简洁 2-近似算法，证明在完全图上 BTT 与 Correlation Clustering、MinSTC、Cluster Deletion 同时具有 $\tfrac{2137}{2136}$ 的 NP-难逼近下界，并构造了一种新的 pivot 流程把任意可行 BTT 覆盖转化为最多 $\tfrac{3}{2}|F|$ 错误的聚类，从而把 BTT 与 CC 最优值的差距从 2 收紧到 $3/2$。
tags:
  - "ICML 2026 Spotlight"
  - "算法理论 / 近似算法 / 图聚类"
  - "坏三角形覆盖"
  - "Correlation Clustering"
  - "近似算法"
  - "LP rounding"
  - "难度证明"
---

# Simple Algorithms for Bad Triangle Transversals with Applications to Correlation Clustering

**会议**: ICML 2026 Spotlight  
**arXiv**: [2602.04463](https://arxiv.org/abs/2602.04463)  
**代码**: 无  
**领域**: 算法理论 / 近似算法 / 图聚类  
**关键词**: 坏三角形覆盖, Correlation Clustering, 近似算法, LP rounding, 难度证明  

## 一句话总结
本文为有符号图上的"坏三角形覆盖"问题（Bad Triangle Transversal, BTT）给出两个仅需单次解 LP 的简洁 2-近似算法，证明在完全图上 BTT 与 Correlation Clustering、MinSTC、Cluster Deletion 同时具有 $\tfrac{2137}{2136}$ 的 NP-难逼近下界，并构造了一种新的 pivot 流程把任意可行 BTT 覆盖转化为最多 $\tfrac{3}{2}|F|$ 错误的聚类，从而把 BTT 与 CC 最优值的差距从 2 收紧到 $3/2$。

## 研究背景与动机
**领域现状**：有符号图 $G=(V, E^+, E^-)$ 在社交网络、Ising 模型、聚类等场景广泛使用。"坏三角形"指恰有一条负边的三角形——它是结构平衡理论的最小不平衡单元。Correlation Clustering (CC) 要求把节点划分到簇中，使簇间正边 + 簇内负边总数最小；Bansal 等人最早就用"打包不相交坏三角形"得到 CC 的常数近似，后续 Ailon 的 pivot 也是 3-近似的标杆。

**现有痛点**：(1) BTT 是 CC 的下界，因此在 LP 难解的大规模图上，研究者转而求 BTT 的快速覆盖再"转换"为 CC 聚类；但已知 BTT 算法要么是 3-近似（取一组边不相交坏三角形的所有边），要么沿用 Krivelevich (1995) 在无符号三角形覆盖上的 2-近似——后者需要反复求解 $\mathcal{O}(m)$ 次 LP，时间瓶颈 $\widetilde{\mathcal{O}}(m^{\alpha+1})$，在大图上完全不实用。(2) BTT 在完全图上是否能严格优于 2-近似、是否存在硬度下界，长期是空白。(3) 把 BTT 覆盖转化为 CC 聚类的现有 MatchFlipPivot (Veldt, 2022) 只能保证 $\text{OPT}_{CC} \le 2\,\text{OPT}_\Delta$，进而 CC 近似比从 $\alpha$ 变成 $2\alpha$，损失太多。

**核心矛盾**：BTT 在 3-uniform hypergraph 上等价于带"二部"约束的顶点覆盖（每条 hyperedge 恰含一个负边节点），既不是普通 VC 也不是 $k$-partite VC，Lovász 1975 的随机化 VC 算法不能直接套；而要得到优于 3 的近似，又只能借助 LP 松弛 $\text{LP}_\Delta$ 的结构信息。

**本文目标**：(1) 给出能直接对 $\text{LP}_\Delta$ 解做单次 rounding、不需要反复重解的 2-近似；(2) 在完全图上证明 BTT/CC/MinSTC/CD 的统一硬度下界；(3) 改进 cover-to-cluster 的转换比，从 2 收紧到 $3/2$，进而把 CC 近似比从 $6$ 改进到 $3+\epsilon$。

**切入角度**：作者注意到 $\text{LP}_\Delta$ 的约束结构具备"每条 bad triangle 恰含 1 条负边"的二部性。如果给"正边"和"负边"设定不同的 rounding 阈值——正边用 $x_e \ge 1/2$，负边用 $x_e > 0$（或对随机阈值 $r$ 取 $x_e > 1-r$ vs $x_e \ge r/2$）——就能利用这条二部约束在单轮 rounding 内得到 2-近似。

**核心 idea**：用"非对称阈值的单轮 LP rounding"换掉 Krivelevich 的"低速迭代 rounding"；用"分类型 budget 攻击 + 改良 pivot 概率"换掉 Veldt 的"翻转边再 pivot"。

## 方法详解

### 整体框架
论文围绕 BTT 给出四块互相衔接的贡献：先是两个**只解一次 LP** 的 2-近似覆盖算法（Algorithm 2 确定性、Algorithm 3 随机化，分别取代 Krivelevich 1995 的迭代重解），再是一个把覆盖转成聚类的**改良 pivot**（Algorithm 4，把 cover-to-cluster 比从 2 收紧到 $3/2$），最后是一套**完全图上的统一硬度下界**（用一个 2SAT gadget 同时打 BTT/CC/MinSTC/CD）。前两块解决「怎么算得又快又准」，第三块解决「算出的覆盖怎么变成 CC 聚类、损失多少」，第四块界定「理论极限在哪」。下面分这四个核心设计展开。

### 关键设计

**1. 简单确定性 2-近似（Algorithm 2）：一次 LP rounding 取代 Krivelevich 的迭代重解**

已有的 2-近似（Krivelevich 1995）要反复求解 $\mathcal{O}(m)$ 次 LP，时间瓶颈 $\widetilde{\mathcal{O}}(m^{\alpha+1})$，大图上根本跑不动。本文的关键观察是 BTT 在 3-uniform hypergraph 上有"每条 bad triangle 恰含一条负边"的二部结构，于是正负边可以各按一套阈值一轮搞定：解一次 $\text{LP}_\Delta$ 拿到分数解 $\{x_e\}$，直接输出 $E^-_{>0}\cup E^+_{\ge 1/2}$——所有取值非零的负边加所有取值 $\ge 1/2$ 的正边。正确性靠互补松弛：对偶 PackingLP 只在 $x_e>0$ 的边上有紧约束，配合二部性可证总边数 $\le 2\cdot\text{LP}_\Delta\le 2\cdot\text{OPT}_\Delta$。因为只解一次 LP，时间被 LP 求解器主导（$\widetilde{\mathcal{O}}(m^\alpha)$），比 Krivelevich 快了约 $m$ 倍。

**2. 随机化 2-近似（Algorithm 3）：对加权与近似 LP 解都鲁棒，配现代组合求解器跑到 $m^{3/2}$**

确定性版虽快，但解 LP 本身在大图上仍是瓶颈，且不支持加权与"只有近似最优解"的现实设定。Algorithm 3 用一个随机阈值搞定这两点：抽 $r\in[0,1]$，取 $\{e\in E^+:x_e\ge r/2\}\cup\{e\in E^-:x_e>1-r\}$ 作覆盖。证明用积分技巧——覆盖任意坏三角形 $t=\{e_1,e_2,e_3\}$ 的概率与 $\sum_{e\in t}x_e\ge 1$ 直接挂钩，期望边数恰为 $2\sum_e x_e$。它还能干净去随机化（Remark 3.4）：把正负边按 $x_e$ 排成两条扫描线，所有 $r$ 区间只产生 $\mathcal{O}(|E|)$ 个候选解，取最小者即可，额外只花 $\mathcal{O}(|E|\log n)$。最大的加分项在 Remark 3.5：当 $\{x_e\}$ 只是 $(1+\epsilon)$-近似最优时仍给 $(2+2\epsilon)$-近似，于是能接上 Cao et al. 2024 的 $\widetilde{\mathcal{O}}(\epsilon^{-7}m^{3/2})$ 组合 LP 求解器，把完全图上的整体复杂度压到 $\widetilde{\mathcal{O}}(\epsilon^{-7}m^{3/2})$，几乎追平"找一组极大不相交坏三角形"的下界时间。

**3. 改良 pivot：把 cover-to-cluster 比从 2 收紧到 $3/2$（Algorithm 4 / Theorem 5.1）**

前两个算法只解决「找一个小覆盖 $F$」，但实际要的是 CC 聚类——还得把 $F$ 转成簇。Veldt 2022 的 MatchFlipPivot 走「翻转 $F$ 中所有边的符号、再在辅助图上 pivot」的路子，只能保证 $\text{OPT}_{CC}\le 2\,\text{OPT}_\Delta$，损失整整一倍。本文不翻边，而是把 Ailon pivot 的确定性吸入规则改成**依赖覆盖 $F$ 的概率规则**：当被考察的边 $uv\in F$ 时，正边以概率 $1/4$、负边以概率 $3/4$ 吸入当前 pivot 的簇；$uv\notin F$ 则沿用 Ailon 的确定性规则（正边吸入、负边不吸）。分析的巧思在于换了一套「代价分摊」——给每条 $uv\in F$ 分配单位预算 $b(uv)=1$、其余为 0，把 pivot 产生的错误记到预算而非 LP 值上；只要证任意三元组 $\{u,v,w\}$ 都满足「错误之概率和 / 预算之概率和 $\le\tfrac32$」（完全图按三角形中正边条数分四类逐一验证），按 Ailon 框架立刻得到期望 $\tfrac32|F|$ 个错误，即 Theorem 1.4 的 $\text{OPT}_{CC}\le\tfrac32\text{OPT}_\Delta$。把它接到 Theorem 1.1 的 $(2+\epsilon)$ 覆盖上，CC 近似比就从 Veldt 的 6 一步降到 $3+\epsilon$，且时间复杂度同档。

**4. 完全图统一硬度下界：一套 gadget 打四个问题（Theorem 1.2 / 4.6-4.7）**

在算法把上界推到 2 之后，自然要问「下界在哪、还能不能更好」。本文给出完全图上的第一个**显式常数硬度**：用一个六角形 gadget + 子句边，从 Chlebík & Chlebíková 的 Minimum 2CNF Deletion (MD) 做 gap-preserving 归约——每个变量配 12 节点六角形（含 6 个 crown），子句把 crown 接到 clause node，使 BTT 最优解恰对应 MD 最优解；MD 的 $2\delta n$ vs $3\delta n$ gap 翻译成 BTT 的 $(11+2\delta)n$ vs $(11+3\delta)n$，取 $\delta=1/194$ 即得「NP-hard 难以 $<\tfrac{2137}{2136}$ 逼近」。真正的概念性贡献在于**复用**：同一构造下 MinSTC+/CC/CD 的最优值都等于 $\text{OPT}_\Delta(G)$，所以一次归约同时给出四个问题的同一下界——此前 MinSTC+ 没有任何显式下界、CC 也只有更弱的随机化下界。再配上 Lemma 4.1（$\text{LP}_\Delta$ 的 integrality gap 在完全图上仍 $\ge 2$），本文同时圈出了「基于 $\text{LP}_\Delta$ 无法突破 2-近似」的硬边界。

### 损失函数 / 训练策略
本文是组合优化与近似算法论文，不涉及训练循环；所有 2-近似都是关于 $\text{LP}_\Delta$ 解的 rounding 过程，所有 $\tfrac{3}{2}$-近似是基于 Algorithm 4 的随机 pivot；分析依赖标准的 LP 互补松弛、积分概率论证以及 Ailon 框架下的"三元组分摊"。

## 实验关键数据

### 主实验
本论文是纯理论工作，没有实验代码。下表汇总各算法在 BTT 上的近似比与时间复杂度（$m$ 为边数，$\alpha \ge 2$ 为 LP 求解的矩阵乘法常数）：

| 算法 | 近似比 | 解 LP 次数 | 时间复杂度 | 工作 |
|------|--------|-----------|-----------|------|
| 标准 3-近似（极大边不相交坏三角形） | 3 | 0 | $\mathcal{O}(m^{3/2})$（完全图） | 多篇 |
| Krivelevich 1995 (Algorithm 1) | 2 | $\mathcal{O}(m)$ | $\widetilde{\mathcal{O}}(m^{\alpha+1})$ | Krivelevich 1995 |
| Algorithm 2（确定性） | 2 | 1 | $\widetilde{\mathcal{O}}(m^\alpha)$ | **本文** |
| Algorithm 3（随机化） | 2 (期望) | 1 (可近似 LP) | $\widetilde{\mathcal{O}}(\epsilon^{-7} m^{3/2})$（完全图，Theorem 1.1） | **本文** |
| Algorithm 3 + 去随机化 | 2 | 1 | + $\mathcal{O}(|E|\log n)$ | **本文** |

### 下界与转换比

| 结果 | 结论 | 适用范围 | 类型 |
|------|------|---------|------|
| Theorem 4.2 | BTT 与 Vertex Cover 同等难逼近，UGC-hard $\ge 2$ | 一般图 | 难度 |
| Theorem 4.6 | BTT 在完全图上 NP-hard 难以 $< \tfrac{2137}{2136}$ 逼近 | 完全图 | 难度 |
| Theorem 4.7 | 同一 gadget 给出 CC / MinSTC / CD 的 $\tfrac{2137}{2136}$ 下界 | 完全图 | 难度 |
| Theorem 5.1 / 1.4 | $\text{OPT}_{CC} \le \tfrac{3}{2}\,\text{OPT}_\Delta$（改进 Veldt 2022 的 2） | 完全图 | 转换 |
| Lemma 4.1 | $\text{LP}_\Delta$ 的 integrality gap $\ge 2$（即便在完全图上） | 完全图 | 紧性 |

### 关键发现
- **单次 LP rounding 足以达到 2-近似**：Algorithm 2/3 都证明了 BTT 的二部结构允许在一轮内完成 rounding，使 Krivelevich 的 $m^{\alpha+1}$ 复杂度可改进为 $m^\alpha$（或在完全图上的 $m^{3/2}$）。
- **2-近似已是一般图上的极限**：Theorem 4.2 与 Khot–Regev 2008 结合，意味着除非 UGC 失效，无法突破 2-近似——这把研究重心转向"在完全图上能否做到 $< 2$"的开放问题。
- **同一归约打四个问题**：用 Chlebík 2CNF gadget 一次性给出 BTT/CC/MinSTC/CD 的 $\tfrac{2137}{2136}$ 下界，是本文的概念性贡献；之前 MinSTC 没有任何显式下界，CC 也只有 Cao 等 2024 的 $24/23$ 随机化下界——本文给出的是更强的确定性归约。
- **$\tfrac{3}{2}$ pivot 直接撬动 CC 近似比**：结合 Theorem 1.1 与 Theorem 1.4，BTT 的 $(2+\epsilon)$ 近似转换得到 CC 的 $(3+\epsilon)$ 近似，比 Veldt 2022 的 MatchFlipPivot 的 6-近似显著改善，且时间复杂度同档。

## 亮点与洞察
- **"二部 hypergraph VC"视角**：作者把 BTT 重新解释为一种特殊的 3-uniform hypergraph VC（每条 hyperedge 恰含 1 个负节点），并利用这条结构信息为正负边分别设置 rounding 阈值——这种"按边类型分桶"的思路可迁移到任何含有"标签 / 颜色"约束的覆盖问题。
- **"两条线一起扫描" 去随机化**：Remark 3.4 把 Algorithm 3 的连续随机变量 $r$ 离散化为正负边按 LP 值排序的两条扫描线，只检查 $\mathcal{O}(|E|)$ 个候选阈值——这是把 LP rounding 类算法去随机化的非常干净的范式。
- **预算-pivot 分析框架**：把每条 cover 边设为单位预算，再证明任意三元组的"错误 vs 预算"比 $\le \tfrac{3}{2}$，借助 Ailon 框架直接得 cover-to-cluster 比；这种"换一种代价分摊"的小修改让 Veldt 的 2-bound 一步退到 $3/2$，提示在 CC 系列问题里"如何选 pivot 概率"还有挖掘空间。
- **gadget 复用打多问题下界**：用同一个六角形 + 子句结构同时打 BTT/CC/MinSTC/CD 的硬度，让"完全图上四个看似不同的聚类/编辑问题在硬度上是同质的"这一观察显式化。
- **加权与近似 LP 友好**：Algorithm 3 同时支持加权 BTT 和"只有 $(1+\epsilon)$-近似 LP 解"，这两个属性让它能直接接入 LambdaSTC、temporal MinSTC+ 等实际带权问题，无需重证。

## 局限与展望
- 文中所有"快速性"都是 LP 求解之后的渐近复杂度，对常数与具体 LP 求解器选择不敏感，但工程实现上 $\widetilde{\mathcal{O}}(\epsilon^{-7} m^{3/2})$ 的 $\epsilon^{-7}$ 因子对小 $\epsilon$ 仍然吃力。
- 在一般图上 2 是最优近似（UGC-hard），所以本文工作的"提升空间"完全锁在完全图；作者提出的关键开放问题 $\text{OPT}_{CC} = \text{OPT}_\Delta$ 既无证明也无反例。
- Algorithm 4 的 $\tfrac{3}{2}$ 比依赖随机 pivot；如果能给出 $(1+\epsilon)$ 比的多项式时间转换，BTT 的 $(2+\epsilon)$ 近似将直接翻译成 CC 的 $(2+2\epsilon)$ 近似，逼近 LP integrality gap 下界——这是该方向的下一站。
- 没有跑任何实证实验：在真实社交网络/生物网络数据上 Algorithm 2/3 vs MatchFlipPivot 谁更快、聚类质量如何，仍是空白。

## 相关工作与启发
- **vs Krivelevich (1995)**：同样 2-近似但需迭代解 $\mathcal{O}(m)$ 个 LP；本文 Algorithm 2/3 都只解一次 LP，并显式利用了"恰含一条负边"的有符号结构，从而比无符号三角形覆盖的 Krivelevich 算法更简洁、更快。
- **vs Veldt (2022) MatchFlipPivot**：Veldt 用"翻转覆盖边再 pivot"得到 $\text{OPT}_{CC} \le 2\,\text{OPT}_\Delta$；本文用预算-pivot 分析直接拿到 $\tfrac{3}{2}$，把 CC 近似比从 6 改善到 $3+\epsilon$。
- **vs Cao et al. (2024b)**：他们给出完全图上 $\text{LP}_\Delta$ 的 $(1+\epsilon)$ 组合近似求解和 $2.4$-近似 CC 算法；本文 Algorithm 3 与 Cao 的求解器无缝拼接（Remark 3.5），得到 $(2+\epsilon)$ BTT、$(3+\epsilon)$ CC。
- **vs Chawla et al. (2015)**：Chawla 给出 CC 在完全图上的 2.06-近似但需要解 $\text{LP}_{\text{CC}}$（$\Theta(n^3)$ 约束）；本文方向相反——通过更轻量的 $\text{LP}_\Delta$ 换取更高的常数因子但更好的可扩展性。
- **vs Cohen-Addad et al. (2022)**：他们突破了 CC 的 2-近似（后续到 1.485-近似）；本文给出了 BTT 在完全图上"$< 2$-近似不可能基于 $\text{LP}_\Delta$"的紧性结果（Lemma 4.1 integrality gap $\ge 2$），意味着若想沿 CC 路线继续突破，必须脱离 $\text{LP}_\Delta$ 框架。
- **vs Charikar et al. (2005)**：Charikar 用 hexagram 类似的 gadget 证明 CC 在完全图上是 APX-hard；本文给出了第一个**显式常数下界** $\tfrac{2137}{2136}$，并把它一次扩到 MinSTC+ / CD。
- **vs Bansal et al. (2002) 与 Ailon et al. (2008)**：早期 CC 算法都用"打包边不相交坏三角形"作 fractional dual 或直接 pivot；本文沿用 Ailon 的 pivot 框架但改预算和概率，把 cover→cluster 与 CC 近似比同时改进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Improved Approximation Algorithms for Chromatic and Pseudometric-Weighted Correlation Clustering](../../NeurIPS2025/learning_theory/improved_approximation_algorithms_for_chromatic_and_pseudometric-weighted_correl.md)
- [\[NeurIPS 2025\] Learning-Augmented Streaming Algorithms for Correlation Clustering](../../NeurIPS2025/learning_theory/learning-augmented_streaming_algorithms_for_correlation_clustering.md)
- [\[ICML 2026\] Quantum Algorithms for Triangle Cut Sparsification](quantum_algorithms_for_triangle_cut_sparsification.md)
- [\[ICML 2026\] Estimating Correlation Clustering Cost in Node-Arrival Stream](estimating_correlation_clustering_cost_in_node-arrival_stream.md)
- [\[ICML 2025\] Sparse-Pivot: Dynamic Correlation Clustering for Node Insertions](../../ICML2025/learning_theory/sparse-pivot_dynamic_correlation_clustering_for_node_insertions.md)

</div>

<!-- RELATED:END -->
