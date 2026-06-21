---
title: >-
  [论文解读] Welfarist Formulations for Diverse Similarity Search
description: >-
  [ICLR 2026][信息检索/RAG][近邻搜索] 本文把"检索结果的属性多样性"建模成数理经济学里的**福利函数最大化**问题——把每个属性当成一个 agent，用 **Nash 社会福利（几何均值）**取代标准近邻搜索的相似度求和，从而在"相关性"和"多样性"之间做**随查询自适应**的权衡，并给出可套在任意 ANN 之上、带可证明近似保证的高效算法。
tags:
  - "ICLR 2026"
  - "信息检索/RAG"
  - "近邻搜索"
  - "多样性"
  - "Nash 社会福利"
  - "ANN"
  - "RAG"
---

# Welfarist Formulations for Diverse Similarity Search

**会议**: ICLR 2026  
**论文**: [OpenReview](https://openreview.net/forum?id=welfarist_formulations_for_diverse_similarity_search)（⚠️ 链接以原文为准）  
**代码**: 无  
**领域**: 信息检索 / 近邻搜索  
**关键词**: 近邻搜索, 多样性, Nash 社会福利, ANN, RAG

## 一句话总结
本文把"检索结果的属性多样性"建模成数理经济学里的**福利函数最大化**问题——把每个属性当成一个 agent，用 **Nash 社会福利（几何均值）**取代标准近邻搜索的相似度求和，从而在"相关性"和"多样性"之间做**随查询自适应**的权衡，并给出可套在任意 ANN 之上、带可证明近似保证的高效算法。

## 研究背景与动机
**领域现状**：近邻搜索（Nearest Neighbor Search, NNS）是 web 搜索、推荐系统、以及近年 RAG 的底层原语：给定查询向量 $q$ 和向量库 $P$，找出 $k$ 个与 $q$ 最相似的向量，目标是 $\arg\max_{S\subseteq P,|S|=k}\sum_{v\in S}\sigma(q,v)$。高维大规模下精确求解太贵，于是有了近似近邻（ANN）这一整套（LSH、k-d 树、聚类、图索引 DiskANN/HNSW 等）。

**现有痛点**：纯相关性会让结果高度冗余——同一个商家的广告刷屏、同一个域名霸榜、同一种颜色的衬衫占满一屏。所以"多样性"和"相关性"一样重要（Google 限制单域名结果、微软在广告里加多样性约束都是实例）。要量化多样性，自然的做法是给每个向量打上**属性**（颜色、卖家、类目……），让结果在属性上分散。

**核心矛盾**：此前唯一的多样性方案（Anand et al. 2025）是**硬约束**——规定每个属性 $\ell$ 至多贡献 $k'$ 个结果，然后在此约束下最大化相关性。但 $k'$ 是一个**固定的拍脑袋配额**：它无法随查询意图自适应。查"blue shirt"时，"blue"这个颜色约束反而会把合法的蓝衬衫挡在外面、牺牲相关性；查"shirts"时又希望颜色尽量分散。同一个 $k'$ 不可能同时伺候这两类查询。更糟的是，把硬约束推广到**多属性**（一个向量同时有多个属性）时，连"是否存在满足约束的 $k$ 个向量"都是 NP-hard。

**本文目标**：要一个既保相关、又促多样，且**能随查询自动调节**、还能让从业者**参数化控制权衡**的统一框架，并配上能落地（套在现成 ANN 上）、有理论保证的算法。

**切入角度**：作者借来数理经济学的**集体福利（collective welfare）**理论。把 $c$ 个属性看作 $c$ 个 agent，每个 agent 的"效用"是被选中向量里属于该属性的那部分相似度之和；用**福利函数**而非简单求和来聚合这些效用。

**核心 idea**：用 **Nash 社会福利（几何均值）替代算术求和**作为检索目标——几何均值天然奖励"各属性效用都不为零"，因此在不设硬配额的情况下自动促成多样性，且当某属性确实主导查询意图时又允许全选该属性，从而做到查询自适应。

## 方法详解

### 整体框架
本文是一个**目标函数 + 算法**的工作，没有神经网络 pipeline，核心是把检索目标从"相似度求和"换成"福利函数"，再为新目标设计带证明的贪心算法。整体可以拆成三层：

1. **建模层（NaNNS）**：每个属性 $\ell\in[c]$ 是一个 agent，子集 $S$ 给它的效用 $u_\ell(S)=\sum_{v\in S\cap D_\ell}\sigma(q,v)$（$D_\ell$ 是带属性 $\ell$ 的向量集）。检索目标变成最大化 Nash 社会福利 $\mathrm{NSW}(S)=\big(\prod_{\ell=1}^c(u_\ell(S)+\eta)\big)^{1/c}$。
2. **推广层（p-NNS）**：用 $p$-均值 $M_p$ 统一一族目标——$p=1$ 退回标准 NNS（只要相关性），$p\to 0$ 是 NSW，$p\to-\infty$ 是平均主义（极致多样）。一个旋钮连续调权衡。
3. **算法层**：单属性下给出**精确最优**的贪心算法 Nash-ANN，能把任意 ANN 当子程序调用、并继承其近似比；多属性下证明 NP-hard，并给出对 $\log\mathrm{NSW}$ 的 $(1-1/e)$ 近似算法。

由于本文是纯算法/理论改进（换目标函数 + 贪心选择），不涉及多阶段神经网络流水线，这里用文字 + 公式讲清，不画 Mermaid 框架图。

### 关键设计

**1. NaNNS：把属性当 agent、用几何均值替代求和**

这是全文的建模基石，直接针对"硬配额无法自适应"的痛点。定义每个属性 agent 的效用为该属性被选向量的累计相似度 $u_\ell(S)=\sum_{v\in S\cap D_\ell}\sigma(q,v)$，整体目标取这 $c$ 个效用（加平滑常数 $\eta>0$ 防零）的几何均值：

$$\mathrm{NaNNS}:\quad \arg\max_{S\subseteq P,\,|S|=k}\ \log\mathrm{NSW}(S)=\frac{1}{c}\sum_{\ell\in[c]}\log\big(u_\ell(S)+\eta\big).$$

为什么有效：几何均值满足福利经济学的多条公平公理（对称性、无关 agent 独立性、尺度不变性、Pigou–Dalton 原则），并由 AM–GM 不等式夹在平均主义（$\min_\ell u_\ell$）和功利主义（算术均值）之间——因此它"既不让任何属性饿死、也不强行均分"。关键是它的边际收益是**递减的**：当某属性 $u_\ell$ 已经很大时，再往它加一个向量带来的 $\log$ 增量很小，于是模型自动把名额让给效用还低的属性，**多样性是优化目标自带的副产品，而不是外挂的硬约束**。同时若某属性天然主导查询（"blue shirt"里全是蓝），其他属性效用本就接近 0，几何均值不会强迫去选无关颜色，于是相关性也被保住——这正是硬配额做不到的查询自适应。注意：若把几何均值换回算术均值，单属性下就**精确退化**回标准 NNS，说明 NSW 是 NNS 的严格福利化推广。

**2. p-mean 推广：一个旋钮连续调"相关↔多样"**

NaNNS 给的是一个固定平衡点，但不同任务需要的多样化强度不同。作者用广义 $p$-均值把目标参数化：对 $p\in(-\infty,1]$，

$$M_p(u_1,\dots,u_c)=\Big(\tfrac{1}{c}\textstyle\sum_{\ell=1}^c u_\ell^{\,p}\Big)^{1/p},\qquad \mathrm{p\text{-}NNS}:\ \max_{|S|=k} M_p\big(u_1(S),\dots,u_c(S)\big).$$

$p=1$ 是功利主义（算术均值，即纯相关性、放弃多样性），$p\to0$ 是 NSW，$p\to-\infty$ 是平均主义（min，极致多样但几乎不看相关性）。这给了从业者一个**单调旋钮**：$p$ 越小越偏多样、越大越偏相关，可按数据集甚至按查询临时调，而不必像硬约束那样去猜一个离散的整数配额 $k'$。实验里 $p\in\{-10,-1,-0.5,0,0.5,1\}$ 平滑地刻画了整条 Pareto 前沿。

**3. Nash-ANN：可套在任意 ANN 之上的精确贪心算法**

有了新目标还得能高效解。单属性设定下（每个向量恰好一个属性，$D_\ell$ 互不相交），作者给出 Algorithm 1：**预处理**阶段对每个属性 $\ell$ 单独跑一次近邻搜索，取该属性内对 $q$ 最相似的 $k$ 个候选 $\widehat D_\ell$（这一步可以是精确近邻，也可以直接调任意标准 ANN）；**贪心选择**阶段维护每个属性已选的累计相似度 $w_\ell$，每轮选边际 $\log\mathrm{NSW}$ 增益最大的下一个向量：

$$a=\arg\max_{\ell\in[c]}\Big[\log\big(w_\ell+\eta+\sigma(q,v^\ell_{(k_\ell+1)})\big)-\log(w_\ell+\eta)\Big],$$

迭代 $k$ 次。由于 $\log$ 的凹性保证了每个属性内"先选更相似的、边际递减"，可以证明这个贪心**就是 NaNNS 的最优解**（Theorem 1，关键引理是把 ALG 和 OPT 的属性计数逐步对齐到完全一致），复杂度 $O(kc)+\sum_\ell \mathrm{ENN}(D_\ell,q)$。最实用的一点是 **Corollary 2**：若预处理用的是 $\alpha$-近似 ANN（而非精确近邻），Nash-ANN 输出的 NSW 仍达到最优的 $\alpha$ 倍——也就是说它**继承底层 ANN 的近似比和亚线性时间**，可以直接架在 DiskANN/HNSW 这类工业索引之上，而不用重建索引。$p$-NNS 也有对应的精确高效算法（Algorithm 3）。

**4. 多属性的 NP-hard 与 (1−1/e) 近似**

当一个向量可带多个属性（$|atb(v)|\ge1$，各 $D_\ell$ 相交）时，结构被破坏：作者证明 NaNNS 变成 **NP-hard**（Theorem 3，由最大独立集归约）——这同时解释了为何硬约束方案在多属性下连可行性判定都困难。好消息是 $\log\mathrm{NSW}(S)=\frac1c\sum_\ell\log(u_\ell(S)+\eta)$ 关于所选集合是**单调子模**的，于是标准的边际增益贪心（Algorithm 2，Multi Nash-ANN）给出 $\log\mathrm{NSW}(\mathrm{ALG})\ge(1-\tfrac1e)\log\mathrm{NSW}(\mathrm{OPT})$ 的多项式时间近似（Theorem 4，$1-1/e\approx0.63$）。实际中先用一次 ANN 拉回 $L=10000$ 个候选，再在候选集内跑这个贪心。

### 损失函数 / 训练策略
本文无模型训练，"目标函数"即 $\log\mathrm{NSW}$ / $M_p$；唯一超参是平滑常数 $\eta>0$（保证 NSW 非零）与权衡旋钮 $p$。算法均为确定性贪心，无梯度优化。

## 实验关键数据

### 主实验
在 4 个数据集上对比 ANN（纯相关）、Div-ANN（硬约束，Anand et al. 2025，不同 $k'$）、Nash-ANN（本文）。相关性用**近似比** $\big(\sum_{v\in A}\sigma\big)/\big(\sum_{v\in O}\sigma\big)\in[0,1]$（$O$ 为真·top-$k$），多样性用**熵** $\sum_\ell -p_\ell\log p_\ell$（$p_\ell=|S\cap D_\ell|/|S|$，越高越多样，上限 $\log k$）。

| 数据集 | 规模 / 维度 | 属性 | 关键结论（$k=50$，单属性） |
|--------|------------|------|------|
| Amazon | 92K / 768 | 商品颜色 | Nash-ANN 近似比≈1，熵≈Div-ANN($k'{=}1$)，且 Pareto **支配** Div-ANN($k'{=}2$) |
| Deep1b-(Clus) | 9.99M / 96 | 合成 | Nash-ANN Pareto **支配** Div-ANN($k'{=}5$) |
| Sift1m-(Clus) | 1M / 128 | 合成(多属性 $c{=}80$,4 类) | Multi Nash-ANN 优于 Multi Div-ANN 的相关↔多样前沿 |
| ArXiv | 200K / 1536 | 年份、论文类目 | 趋势一致（见附录 F.3） |

定性上（Figure 1，Amazon，$k=9$）：查"shirts"时 Nash 方法选出多种颜色的衬衫，查"blue shirt"时则收敛到蓝色——**同一目标无需改参数即可随查询切换**，正是硬配额做不到的。

### 消融 / 参数分析
通过扫 $p$ 验证 $p$-mean 的旋钮作用（Figure 2 底行）：

| 配置 | 相关性（近似比） | 多样性（熵） | 说明 |
|------|------|------|------|
| $p=1$ | 最高（≈1） | 最低 | 退化为标准 NNS，纯相关 |
| $p=0$（Nash） | 高 | 高 | 自适应平衡点 |
| $p=-1,-0.5$ | 居中 | 更高 | 偏多样 |
| $p=-10$ | 最低 | 最高（趋向 min/平均主义） | 几乎只看多样性 |

随 $p$ 从 1 降到 $-10$，结果在相关↔多样的 Pareto 前沿上**平滑滑动**，单/多属性设定下趋势一致。

### 关键发现
- **NSW 自带多样性**：不设任何硬配额，几何均值的边际递减就把名额匀给弱势属性，且查询主导某属性时不牺牲相关性——这是相对硬约束最核心的优势。
- **可继承底层 ANN**：Nash-ANN 把 ANN 当黑盒子程序，$\alpha$-近似输入 → $\alpha$-近似输出，无需重建工业索引即可部署；另有更快的启发式（先拉大候选集再选）进一步提速（附录 F.5）。
- **多属性是真·难点**：硬约束在多属性下连可行性都是 NP-hard，而 NSW 借子模性拿到 $(1-1/e)$ 近似，回避了这道墙。

## 亮点与洞察
- **跨学科借力**：把福利经济学的 Nash 社会福利搬进检索，多样性不再是"加约束"而是"换目标"，一个公式同时拿到公平性公理和相关性——视角很漂亮且可迁移到推荐排序、广告分配等任何"相关 vs 公平/多样"的场景。
- **几何均值的边际递减 = 免费的多样性**：这是最"啊哈"的点——无需任何额外参数，凹 $\log$ 的边际收益递减天然实现了自适应配额。
- **算法与索引解耦**：Corollary 2 那种"近似比可乘性继承"让方法即插即用，工程落地成本极低；这种"包一层福利贪心"的范式可复用到带子模目标的其他检索任务。

## 局限与展望
- **属性需预先标注且离散**：方法依赖每个向量已有清晰的属性标签（颜色/卖家/类目），对没有显式属性或属性连续的场景不直接适用。
- **多属性只有近似**：单属性精确最优，但多属性 NP-hard，只能 $(1-1/e)$ 近似，且需先用 ANN 拉一个较大候选集（$L=10^4$），候选集大小会影响最终质量。
- **$\eta$ 与 $p$ 的选择**：虽然 $p$ 比硬配额 $k'$ 更连续好调，但"该取多少 $p$"在真实部署里仍需按任务调参；平滑常数 $\eta$ 对结果的敏感性文中讨论有限。⚠️ 具体数值以原文附录为准。
- 改进方向：把属性多样性推广到连续/层次属性、与重排序模型联合学习 $p$、或在多属性下寻找更紧的近似。

## 相关工作与启发
- **vs Div-ANN（Anand et al. 2025，硬约束）**：他们用"每属性至多 $k'$ 个"的硬配额最大化相关性，本文用 NSW/$p$-mean 软目标。区别在于硬约束固定且离散、无法随查询自适应、多属性下 NP-hard 到连可行性都难；本文软目标查询自适应、单属性可精确最优、多属性有可证明近似，且常常 Pareto 支配 Div-ANN。
- **vs 标准 ANN（DiskANN/HNSW 等）**：标准 ANN 只优化相关性、零多样性。本文不替换它们，而是把它们当子程序，在其上套一层福利贪心——继承其速度与近似比，额外换来多样性。
- **vs MMR 等经典多样化检索**：传统最大边际相关（MMR）靠"相似度−冗余惩罚"启发式，缺乏公平公理与近似保证；本文从福利经济学公理出发，目标可解释、算法有理论担保。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 Nash 社会福利引入近邻搜索、统一相关与多样，视角新且自洽
- 实验充分度: ⭐⭐⭐⭐ 四数据集 + 单/多属性 + $p$ 扫描齐全，但多为权衡曲线、缺端到端 RAG 下游验证
- 写作质量: ⭐⭐⭐⭐ 动机清晰、定理与算法陈述严谨，部分证明与算法细节进了附录
- 价值: ⭐⭐⭐⭐⭐ 即插即用、可证明、解决硬约束自适应难题，对检索/推荐/广告有直接落地价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LotusFilter: Fast Diverse Nearest Neighbor Search via a Learned Cutoff Table](../../CVPR2025/information_retrieval/lotusfilter_fast_diverse_nearest_neighbor_search_via_a_learned_cutoff_table.md)
- [\[ICLR 2026\] ELViS: Efficient Visual Similarity from Local Descriptors that Generalizes Across Domains](elvis_efficient_visual_similarity_from_local_descriptors_that_generalizes_across.md)
- [\[ICLR 2026\] Hybrid Deep Searcher: Scalable Parallel and Sequential Search Reasoning](hybrid_deep_searcher_scalable_parallel_and_sequential_search_reasoning.md)
- [\[ICLR 2026\] Lean Finder: Semantic Search for Mathlib That Understands User Intents](lean_finder_semantic_search_for_mathlib_that_understands_user_intents.md)
- [\[ICLR 2026\] Demystifying Deep Search: A Holistic Evaluation with Hint-free Multi-Hop Questions and Factorised Metrics](demystifying_deep_search_a_holistic_evaluation_with_hint-free_multi-hop_question.md)

</div>

<!-- RELATED:END -->
