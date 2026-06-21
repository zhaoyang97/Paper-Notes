---
title: >-
  [论文解读] A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport
description: >-
  [ICLR2026][优化/理论][最优传输] 本文给出第一个对**所有** $p\in[1,\infty]$（含 $p=\infty$）都成立的、真正平方时间的常数因子近似算法：在任意度量空间上，用 $O(n^2+(n^{3/2}\varepsilon^{-1}\log n\log\Delta)^{1+o(1)}\log U)$ 时间算出一个 $(4+\varepsilon)$-近似的 $W_p$ 最优传输方案，把此前 $O(\log n)$ 的近似比一举压成常数。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "最优传输"
  - "近似算法"
  - "$W_p$ 距离"
  - "有向 spanner"
  - "双色最近对"
---

# A Scalable Constant-Factor Approximation Algorithm for $W_p$ Optimal Transport

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=RPQKJxrEPs](https://openreview.net/forum?id=RPQKJxrEPs)  
**代码**: https://anonymous.4open.science/r/anon_matching-2B3B/ （匿名仓库，proof-of-concept Python 实现）  
**领域**: optimization  
**关键词**: 最优传输, 近似算法, $W_p$ 距离, 有向 spanner, 双色最近对

## 一句话总结
本文给出第一个对**所有** $p\in[1,\infty]$（含 $p=\infty$）都成立的、真正平方时间的常数因子近似算法：在任意度量空间上，用 $O(n^2+(n^{3/2}\varepsilon^{-1}\log n\log\Delta)^{1+o(1)}\log U)$ 时间算出一个 $(4+\varepsilon)$-近似的 $W_p$ 最优传输方案，把此前 $O(\log n)$ 的近似比一举压成常数。

## 研究背景与动机
**领域现状**：给定两个支撑在有限点集 $A,B\subseteq X$ 上的离散分布 $\mu,\nu$，把质量 $\delta$ 从 $a$ 搬到 $b$ 的代价是 $\delta\, d(a,b)^p$，$W_p$ 最优传输就是要找搬运总代价最小的方案 $\sigma$，其代价 $w_p(\sigma)=\big(\sum_{a,b}\sigma(a,b)\,d(a,b)^p\big)^{1/p}$；当 $p\to\infty$ 时退化为 $w_\infty(\sigma)=\max_{\sigma(a,b)>0}d(a,b)$。不同 $p$ 各有用武之地：$W_1$ 度量总位移（词移距离 WMD），$W_2$ 有单调性与平移不变性，$W_\infty$ 近年用于拓扑数据分析和神经网络拓扑层的收敛性分析。

**现有痛点**：精确解可以写成最小费用流（MCF），用 Chen et al. (2022) 的内点法在 $n^{2+o(1)}$ 时间求解，但算法极其复杂、不可实用——连"稠密图里是否存在完美匹配能不能在平方时间判定"这种更简单的问题都还是公开难题。于是研究转向近似。但现有近似都有硬伤：Charikar (2002) 把度量嵌进层次良分树（HST）得到 $O(\log n)$-近似；Sinkhorn 熵正则（Cuturi 2013）只给**加性**误差 $\varepsilon\cdot\mathrm{diam}$、且运行时间随 $p$ 恶化（$W_1$ 是 $n^2/\varepsilon^{O(1)}$，到 $W_p$ 变 $n^2/\varepsilon^{O(p)}$），更**完全无法处理 $p=\infty$**；最接近本文的 Lahn et al. (2025) 在任意度量下给出 $O(\log n)$-相对近似，时间 $O(n^2\log U\log\Delta\log n)$，但近似比仍是对数级。

**核心矛盾**：近似比、运行时间、$p$ 的取值范围三者难以兼得。要么近似比好但只在欧氏/低维成立（无法推广到 $p\ge2$），要么通用但近似比是 $O(\log n)$，要么干脆碰不了 $p=\infty$。根子在于：当 $p>1$ 时，传输代价 $d(\cdot,\cdot)^p$ **不再是度量**（违反三角不等式），所以经典的无向度量 spanner / 距离 oracle 直接失效。

**本文目标**：在**任意度量**、**全 $p\in[1,\infty]$**、**真正平方时间**这三个约束下，把近似比从 $O(\log n)$ 降到常数。

**切入角度**：作者发现，与其去近似那个不是度量的 $d^p$，不如先在子平方空间里高精度近似底层度量 $d$ 本身，再把这个近似"抬升"到 $d^p$ 上。这需要一种新的、能容忍 $p$ 次幂放大的近似结构。

**核心 idea**：用 Bourgain 多尺度采样造一个**两层聚类**，导出一个 $(4+\varepsilon)$-近似的代理距离 $d_C$；再把它编织成一个**有向 spanner** 来近似 $d^p$（有向是为了绕开 $d^p$ 非度量的障碍）；最后所有算法只通过"双色最近对（BCP）"原语访问数据、绝不显式计算两两距离，从而做到既快又可实现。

## 方法详解

### 整体框架
方法分三层、最终落到两套算法。第一层是**距离近似**：在点集 $P=A\cup B$ 上构造一个两层聚类 $\mathcal{C}$，并定义代理距离 $d_C(x,y)$，证明它把真实度量夹在 $d\le d_C\le(4+\varepsilon)d$ 之间，而存储只需约 $n^{3/2}$ 而非 $O(n^2)$。第二层是**结构化**：把聚类编织成一个有向图（spanner），让 $B$ 到 $A$ 的最短路长度近似 $d^p$，把"近似 $d$"抬升成"近似 $d^p$"。第三层是**求解**：在这个稀疏有向图上跑算法——要么用 Chen et al. (2022) 的 MCF 内点法（理论最优，证明定理 1.1，但不可实现），要么用一套只依赖 BCP 原语的组合算法（可实现，证明定理 1.2/1.3）。

整条链路的关键约束是 $\Delta$（点集 spread，即最大与最小非零距离之比）与 $U$（概率最大最小之比），运行时间里到处出现 $\log\Delta,\log U$。最终结论：MCF 版给出 $(4+\varepsilon)$-近似、$O(n^2+(n^{3/2}\varepsilon^{-1}\log n\log\Delta)^{1+o(1)}\log U)$ 时间，对全 $p$（含 $\infty$）成立；查询模型下预处理一次 $X$ 后，对任意支撑在 $X$ 上的分布对可在子平方时间回答 $O(k)$-近似查询。作者还配了条件下界：$p=\infty$ 时若能在 $O(n^2)$ 内做到相对因子优于 $2$，就能在平方时间解任意二部图的完美匹配——这说明 $(4+\varepsilon)$ 这个常数已经卡在难度边界附近。

### 关键设计

**1. 两层聚类 + 代理距离：用 $n^{3/2}$ 空间夹出 $(4+\varepsilon)$ 度量近似**

痛点是：要平方时间就不能存 $O(n^2)$ 的两两距离矩阵，但又要能随时取到足够准的距离。作者借 Bourgain 多尺度采样的思路构造聚类。设 $P_0=P$，把每个点以概率 $n^{-1/2}$ 独立采样进 $P_1$，期望 $|P_1|=\sqrt{n}$。取半径几何序列 $r_i=(1+\tfrac{\varepsilon}{4})^i$，$0\le i\le t=\lceil\log_{1+\varepsilon/4}\Delta\rceil$。生成两类簇：(i) 对每个未采样点 $q\in P_0\setminus P_1$，在它的 Voronoi 集 $V(q,P_1)=\{y:d(y,q)<d(y,P_1)\}$ 内按半径切出 $C_q[i]=\{x\in V(q,P_1):d(x,q)\le r_i\}$；(ii) 对每个采样点 $q\in P_1$，直接按半径切出 $C_q[i]=\{x\in P_0:d(x,q)\le r_i\}$。

代理距离的定义极简洁：$d_C(x,y)=2r_i$，其中 $i$ 是**同时包含 $x,y$ 的最小索引的簇**。引理 2.2 证 $d(x,y)\le d_C(x,y)\le(4+\varepsilon)d(x,y)$，证明思路是分情形：若 $x$ 被 $P_1$ 中某点 $a$ "隔开"（存在 $a\in P_1$ 使 $d(x,a)<d(x,y)$），由三角不等式 $d(y,a)\le 2d(x,y)$，于是存在 type-(ii) 簇 $C_a[i]$ 同时含 $x,y$ 且 $2r_i\le 2(1+\tfrac{\varepsilon}{4})d(y,a)\le 4(1+\tfrac{\varepsilon}{4})d(x,y)$；若没被隔开则 $x\in V(y,P_1)$，type-(i) 簇 $C_y[i]$ 给出 $2r_i\le2(1+\tfrac{\varepsilon}{4})d(x,y)$。关键的代价控制来自**度数引理 2.1**：每个点参与的簇数 $\deg_{\mathcal{C}}(p)=O(\sqrt{n}\,\varepsilon^{-1}\log n\log\Delta)$ 以 $\ge1-1/n$ 概率成立（type-(ii) 由 Chernoff 控住 $|P_1|\le2\sqrt n$；type-(i) 靠"$p$ 只可能落进比它到最近 $P_1$ 点更近的那些 Voronoi 集"，把数目归约成一个几何随机变量 $s$，$\Pr[s>2\sqrt n\log n]\le n^{-2}$）。因此总空间 $O(n^{3/2}\varepsilon^{-1}\log\Delta)$，构造时间 $O(n^2)$。这一步是全文的地基：在子平方空间里换来常数因子的距离近似。

**2. 有向 spanner：绕开 $d^p$ 非度量，把度量近似抬升到代价近似**

当 $p>1$，$d^p$ 不满足三角不等式，经典无向 spanner 不能用。作者的做法是给每条边定向。对每个簇 $C$（索引 $i$）新建两个 Steiner 顶点 $a_C,b_C$，加三类边：$a_C\to b_C$ 权 $(2r_i)^p$；对 $a\in A\cap C$ 加零权边 $a\to a_C$；对 $b\in B\cap C$ 加零权边 $b_C\to b$。这张图未必强连通，但保证从任意 $b\in B$ 到任意 $a\in A$ 都有路。引理 2.3 证：图上最短路距离满足 $d^p(a,b)\le d_{G,p}(a,b)\le(4+\varepsilon)^p d^p(a,b)$——也就是把引理 2.2 的 $(4+\varepsilon)$ 原样按 $p$ 次幂抬上去。和经典 spanner 的两点不同正是它快又简单的原因：一是在多个尺度插入精心选择的 **Steiner 点**，大幅简化结构、把构造时间降到 $O(n^2+kn^{1+1/k}\log\Delta)$（经典 spanner 要超平方预处理）；二是**定向**，专门用来表达 $d^p$ 的非对称传输方向。图的规模 $|V|=O(n\varepsilon^{-1}\log\Delta)$、$|E|=O(n^{3/2}\varepsilon^{-1}\log n\log\Delta)$，稀疏到可以直接喂给 MCF：加超级源汇、按 $\mu(a),\nu(b)$ 设容量，跑 Chen et al. (2022) 得 $(|E|)^{1+o(1)}\log U$ 时间的最小费用流，再沿正流路径拆出传输方案 $\sigma$，即定理 1.1。$p=\infty$ 时改为对 $O(\varepsilon^{-1}\log\Delta)$ 个半径值做**二分 + 一系列最大流**：在删去所有代价 $>2r_i$ 边的子图 $G_i$ 上验流是否满，找到最小可行 $i^\*$ 即得 $W_\infty$ 方案——这正是本文能首次把平方时间算法延伸到 $p=\infty$ 的关键。

**3. BCP 原语 + 组合算法：永不显式算距离，只问"加权双色最近对"**

MCF 内点法虽是理论最优却无法实现。作者另造一套可实现的组合算法，核心是把"两两距离"这个昂贵操作全部封进一个**动态加权双色最近对（BCP）** 原语：定义加权距离 $d_w(a,b)=d_C^p(a,b)-w(a)-w(b)$，要在 $A,B$ 的插入/删除下维护 $\mathrm{BCP}_w(A,B)=\arg\min d_w$。数据结构正好复用同一套聚类：每个簇里把 $A\cap C$、$B\cap C$ 各放一个按权值排序的 max-heap，取堆顶 $a_C,b_C$ 算 $\phi_C=(2r_i)^p-w(a_C)-w(b_C)$，再用全局 min-heap $H$ 以 $\phi_C$ 为键，则 $H$ 的根恰是全局 BCP（引理 2.4）。插入/删除一个点只需更新它所在的 $\deg_{\mathcal{C}}(q)$ 个簇，耗时 $O(\deg_{\mathcal{C}}(q)\log n)$。有了 BCP，就能把 Gabow–Tarjan 费用缩放 / 匈牙利搜索整套搬上来：匈牙利搜索本质是用 slack 当边长的 Dijkstra，每次取 $\arg\min(s(a',b')+\ell_{b'})$ 恰好就是一次加权 BCP 查询（附录 D）。OT 需要 $O(n\log U)$ 次匈牙利搜索，每次 $O(\sum_q\deg_{\mathcal{C}}(q)\log n)=O(kn^{1+1/k}\log^2 n)$，得定理 1.2 的 $O(n^{2+1/k}\log^2 n\log\Delta\log U)$；对等质量的 $W_p$-**匹配**问题，搜索次数可降到 $O(\sqrt n)$，得定理 1.3 的 $O(n^2\varepsilon^{-3/2}\log^2 n\log\Delta)$。整套算法**完全不显式计算任何 $d(a,b)$**，只反复调 BCP——这是它在高维（如 MNIST 784 维，两两距离极贵）依然可扩展的根本原因。

## 实验关键数据

### 主实验
proof-of-concept Python 实现，单机 8 核 Apple M1 / 16GB。数据为单位立方体（≤10 维）上的均匀分布与截断正态，外加 MNIST（784 维）。对比对象是 Lahn et al. (2025) 的 HST 代理距离（$O(\log n)$ 失真），两者都用同一套原始-对偶框架，唯一区别就是代理距离。

| 对比项（$p=2$, 均匀分布） | 本文 cluster-dist | Lahn 2025 HST-dist | 结论 |
|--------|------|----------|------|
| 代理距离最大失真 | 始终 $\le(4+\varepsilon)$ | 明显更大 | 本文有理论上界且更紧 |
| 代理距离平均失真 | 接近 $2$ | 明显更大 | 实测远好于最坏界 |
| 最优匹配代价之比 | 基准 | $>4.5\times$ 更大 | cluster-dist 误差小一个量级 |

### 近似比与效率

| 配置 | 观测值 | 说明 |
|------|---------|------|
| 近似比（$p\in\{1,2,3,4,5,\infty\}$，均匀/正态/MNIST） | 典型 $1.5\sim2$ | 始终远在 $(4+\varepsilon)$ 之内，且对 $p$ 稳定 |
| 平均度数（$d\le10$） | 紧贴引理 2.1 理论界 | 两层聚类空间高效、跨设置稳定 |
| BCP 查询次数（主导运行时间） | 按预测规模增长、对 $p$ 几乎不变 | 配合 $\tilde O(n^{1/2})$ 每查询，整体经验上呈平方时间 |

### 关键发现
- **最坏界很松**：理论保证 $(4+\varepsilon)$，但实测近似比常在 $1.5\sim2$，平均失真也接近 $2$；说明 worst-case 分析悲观，工程上效果好得多。
- **对 $p$ 不敏感**：近似比和 BCP 查询次数在 $p=1$ 到 $p=\infty$ 之间几乎不变，验证了"全 $p$ 统一处理"不是纸面承诺。
- **代理距离质量是关键变量**：在完全相同的求解框架下，仅把 HST-dist 换成 cluster-dist，最优匹配代价就小 $4.5$ 倍以上——直接量化了"换一个更准的代理距离"带来的收益。

## 亮点与洞察
- **"先近似度量，再抬升到 $d^p$"这一步换序很巧**：直接近似非度量的 $d^p$ 会被经典 spanner 工具拒之门外；作者退一步先用聚类高精度近似 $d$（引理 2.2），再用有向 spanner 把 $(4+\varepsilon)$ 按 $p$ 次幂整体抬上去（引理 2.3），一招解决"全 $p$ 通用"。
- **定向是绕开非度量障碍的关键**：$d^p$ 不满足三角不等式，但只要保证 $B\to A$ 有向可达、并在边权里编码 $(2r_i)^p$，最短路就能逼近 $d^p$，不需要强连通——这是把度量 spanner 推广到非度量代价的可复用思路。
- **把"昂贵的距离计算"全部封进 BCP 原语**：算法只通过双色最近对访问数据、绝不显式算 $d(a,b)$，使得高维（MNIST 784 维）下依然可扩展，这对任何"距离贵但只需最近对"的几何算法都有借鉴价值。
- **首个能做 $p=\infty$ 的平方时间方法**：通过对半径二分 + 一串最大流处理 $W_\infty$，填补了 Sinkhorn 等加性方法完全无法触及的空白。
- **条件下界自证其紧**：定理 1.4 表明 $p=\infty$ 时把相对因子从 $2$ 再往下压（在 $O(n^2)$ 内）会解决"任意二部图平方时间完美匹配"这一长期公开问题，说明本文 $(4+\varepsilon)$ 已逼近难度天花板。

## 局限与展望
- **常数因子仍有距离**：$(4+\varepsilon)$ 是常数但并非 $(1+\varepsilon)$；下界只挡住了 $p=\infty$ 优于 $2$ 的情形，$2\sim4$ 之间是否可改进、有限 $p$ 能否做到更小常数，仍未解决。
- **理论最优算法不可实现**：定理 1.1 依赖 Chen et al. (2022) 的内点法 MCF，本身没有实用实现；真正能跑的是定理 1.2/1.3 的组合算法，但它们的时间里带 $1/k$ 次幂与多个 $\log$ 因子，常数与对数开销在中等规模上未必比成熟 Sinkhorn 实现更快——论文只给了 proof-of-concept Python 验证，缺与优化实现的端到端墙钟对比。
- **对 spread $\Delta$ 和概率比 $U$ 的依赖**：运行时间含 $\log\Delta\cdot\log U$，当点集尺度跨度极大或概率极不均衡时这些因子会膨胀；对连续分布需先离散化，本文不直接覆盖。
- **随机化与高概率保证**：度数界、整体复杂度都是以 $\ge1-1/n$ 概率成立的 Las Vegas 结果，最坏情形下单点度数仍可能 $\Theta(n)$。

## 相关工作与启发
- **vs Lahn et al. (2025)**：同为任意度量、原始-对偶框架，但他们用 HST 代理距离只有 $O(\log n)$ 失真且不覆盖 $p=\infty$；本文换成聚类代理距离把近似比降到常数 $(4+\varepsilon)$、且统一处理全 $p$，实测代理距离质量高 $4.5\times$ 以上。
- **vs Sinkhorn / 熵正则（Cuturi 2013 及后续）**：Sinkhorn 给加性误差 $\varepsilon\cdot\mathrm{diam}$、随 $p$ 时间恶化到 $n^2/\varepsilon^{O(p)}$、且完全无法处理 $p=\infty$；本文给相对近似、对 $p$ 几乎不敏感、能算 $W_\infty$。
- **vs 欧氏专用 $(1+\varepsilon)$ 近似（Agarwal et al. 2022/2024、Andoni–Zhang 2023 等）**：那些方法近似比更好但绑定欧氏/固定维度且难以推广到 $p\ge2$；本文牺牲到常数因子，换来任意度量 + 全 $p$ 的通用性。
- **vs 经典 spanner / 距离 oracle（Thorup–Zwick、Baswana–Sen 等）**：经典构造只近似度量本身、需超平方预处理、且无法表达非度量的 $d^p$；本文用多尺度 Steiner 点 + 定向，既降低预处理到子平方、又支持动态 BCP 查询，是对这一工具链的针对性改造。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个全 $p$（含 $\infty$）平方时间常数因子近似，"先近似度量再抬升 + 有向 spanner + BCP 原语"的组合很原创
- 实验充分度: ⭐⭐⭐ 验证了理论界和代理距离优势，但只有 proof-of-concept Python，缺与优化实现的墙钟对比
- 写作质量: ⭐⭐⭐⭐ 定理陈述清晰、证明思路（引理 2.1/2.2/2.3）层层递进，理论部分组织得当
- 价值: ⭐⭐⭐⭐ 把 OT 近似比从对数降到常数并打通 $p=\infty$，对 WMD 等共享支撑的查询场景尤其有用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Memory-Efficient Hierarchical Algorithm for Large-scale Optimal Transport Problems](a_memory-efficient_hierarchical_algorithm_for_large-scale_optimal_transport_prob.md)
- [\[ICLR 2026\] HOTA: Hamiltonian Framework for Optimal Transport Advection](hota_hamiltonian_framework_for_optimal_transport_advection.md)
- [\[ICLR 2026\] Neural Hamilton–Jacobi Characteristic Flows for Optimal Transport](neural_hamilton--jacobi_characteristic_flows_for_optimal_transport.md)
- [\[ICLR 2026\] Neural Optimal Transport Meets Multivariate Conformal Prediction](neural_optimal_transport_meets_multivariate_conformal_prediction.md)
- [\[ICLR 2026\] Elastic Optimal Transport: Theory, Application, and Empirical Evaluation](elastic_optimal_transport_theory_application_and_empirical_evaluation.md)

</div>

<!-- RELATED:END -->
