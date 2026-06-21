---
title: >-
  [论文解读] Sublinear Spectral Clustering Oracle with Little Memory
description: >-
  [ICLR 2026][学习理论][谱聚类预言机] 本文为可聚类图设计了第一个"小内存"亚线性谱聚类预言机：用一种分批估计随机游走碰撞概率的新子程序，把构造数据结构所需的空间从过去铁打的 $\Omega(\sqrt{n})$ 一路压到可低于 $n^{0.01}$，换来一条 $S\cdot T=\tilde{O}(n)$ 的空间-时间权衡曲线，并证明这条曲线在一类自然方法下几乎最优。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "亚线性算法"
  - "谱聚类预言机"
  - "空间-时间权衡"
  - "随机游走"
  - "碰撞概率"
---

# Sublinear Spectral Clustering Oracle with Little Memory

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0GpolO2auw](https://openreview.net/forum?id=0GpolO2auw)  
**领域**: 学习理论 / 亚线性算法  
**关键词**: 谱聚类预言机, 亚线性算法, 空间-时间权衡, 随机游走, 碰撞概率

## 一句话总结
本文为可聚类图设计了第一个"小内存"亚线性谱聚类预言机：用一种分批估计随机游走碰撞概率的新子程序，把构造数据结构所需的空间从过去铁打的 $\Omega(\sqrt{n})$ 一路压到可低于 $n^{0.01}$，换来一条 $S\cdot T=\tilde{O}(n)$ 的空间-时间权衡曲线，并证明这条曲线在一类自然方法下几乎最优。

## 研究背景与动机

**领域现状**：图聚类是把顶点划分成"内部稠密、外部稀疏"的若干社区。在大图上，人们不想读完整张图再聚类，于是研究**亚线性谱聚类预言机**（Peng 2020；Gluch et al. 2021；Shen & Peng 2023）：给定对邻接表的查询访问，先用亚线性时间构造一个紧凑数据结构 $D$，之后对任意顶点 $x$ 只需亚线性时间就能回答 $\textsc{WhichCluster}(G,x)$——它属于哪个簇，而无需付出全局 $\Omega(n)$ 的代价。

**现有痛点**：所有已有的亚线性谱聚类预言机，构造 $D$ 时**至少需要 $\Omega(\sqrt{n})$ 的空间**。Peng (2020) 用 $\tilde{\Theta}(\sqrt{n})$ 空间；Gluch et al. (2021) 与 Shen & Peng (2023) 用 $\Omega(n^{1-\delta})$（$\delta\le \tfrac12$），仍 $\ge \sqrt{n}$。对真正的万亿边大图、流式模型（单遍亚线性空间）、云平台（存储与传输昂贵）以及 GPU/TPU（算力强但片上内存有限）这些场景，$\sqrt{n}$ 的工作内存就是瓶颈：内存装不下、频繁访问主存的开销主导一切。

**核心矛盾**：这个 $\sqrt{n}$ 空间下界**并非来自聚类过程本身，而是来自其中的点积估计步骤**。谱聚类的关键原语是估计谱嵌入的点积 $\langle f_x,f_y\rangle$，它可归约为估计两条随机游走分布的碰撞概率 $\langle M^t\mathbf{1}_x,\,M^t\mathbf{1}_y\rangle$（$M$ 是惰性随机游走转移矩阵）。旧做法从每个顶点出发跑 $R\approx\sqrt{n}$ 条独立游走、把终点全存下来构造经验分布再求点积——空间 $O(R)$、时间 $O(Rt)$ 死死绑在一起，为保精度 $R$ 必须 $\ge\sqrt{n}$，于是空间也降不下来。

**本文目标**：(1) 打破 $\sqrt{n}$ 空间壁垒，做出空间远小于 $\sqrt{n}$（如 $n^{0.01}$）仍保持亚线性查询时间的预言机；(2) 刻画空间 $S$ 与查询时间 $T$ 之间的权衡前沿，并证明其最优性。

**切入角度**：既然瓶颈在"空间与游走条数 $R$ 被绑死"，那能不能**解绑**——保住总游走条数 $R$ 不变（精度不掉），却只用远小于 $R$ 的内存？分布检验领域（Canonne & Yang 2024 的均匀性检验）里的分批技巧正好提供了灵感。

**核心 idea**：用**分批**估计碰撞概率代替"一次性存下所有游走终点"——把 $R$ 条游走切成 $B=R/M$ 个批次，每批只跑 $M$ 条、只暂存这 $M$ 个终点算出本批点积，再对所有批求平均，空间因此降到 $O(M)$，得到权衡 $M\cdot R\approx n$，即 $S\cdot T=\tilde{O}(n)$。

## 方法详解

### 整体框架

输入是一张 $d$-正则、$(k,\varphi,\varepsilon)$-可聚类图 $G$（每个簇内部电导 $\phi_{in}(C_i)\ge\varphi$、外部电导 $\phi_{out}(C_i,V)\le\varepsilon$、各簇大小同阶），只能通过邻接表查询访问。输出是一个数据结构 $D$，支持对任意顶点的 $\textsc{WhichCluster}$ 查询，且整体误分类比例小。

整条管线是自底向上的四层嵌套，关键是**只换掉最底层的"碰撞概率估计"原语**，上层聚类框架几乎照搬 Shen & Peng (2023)：

- **最底层**：`EstRWDot`（Alg. 1）用分批技巧在 $\tilde{O}(M)$ 空间内估计一对顶点的碰撞概率 $\langle M^t\mathbf{1}_x,M^t\mathbf{1}_y\rangle$；`EstColliProb`（Alg. 2）把它推广到一组采样顶点之间，输出一个 Gram 矩阵。
- **中间层**：点积预言机 `InitOracle`（Alg. 3）+ `QueryDot`（Alg. 4），把 Gram 矩阵做特征分解得到紧凑矩阵 $\Psi$，从而把碰撞概率"翻译"成谱嵌入点积 $\langle f_x,f_y\rangle$ 的估计，且空间-时间满足同一条权衡。
- **上层**：聚类预言机用点积预言机在采样地标顶点之间建一张相似度图 $H$，$H$ 的连通分量就是簇；查询时把 $x$ 和地标比一遍即可定簇。
- **副产物**：把这套碰撞概率估计直接用来检测随机游走算子的谱隙，得到区分"1 簇 vs 2 簇"的亚线性算法，并配上一个匹配的空间-时间下界。

下面四个关键设计自底向上，分别对应这四层。

### 关键设计

**1. 分批碰撞概率估计：用 $O(M)$ 内存撬动 $R$ 条游走**

这是打破 $\sqrt{n}$ 壁垒的真正发动机，直接针对"空间被游走条数绑死"这个痛点。旧做法把 $R\approx\sqrt{n}$ 条游走的终点一次性全存下来构造经验分布，空间被迫 $\ge\sqrt{n}$。`EstRWDot`（Alg. 1）改成分批：取批数 $B=R/M$，对 $b=1,\dots,B$ 各从 $x$、$y$ 出发跑 $M$ 条长度 $t$ 的惰性随机游走，用本批终点频率构造经验分布 $\hat p_x,\hat p_y$，算出本批点积 $Z_b=\langle\hat p_x,\hat p_y\rangle$，最后取 $Z=\frac1B\sum_b Z_b$。任一时刻只需存住一个批次的 $M$ 个终点，空间降到 $O(M\log n)$ 比特，时间仍是 $O(Rt)$——总游走条数 $R$ 没变，精度因此不掉。

形式保证（Lemma 3.1）：取 $t\ge \tfrac{20\log n}{\varphi^2}$、$1\le M\le O\!\big(\tfrac{n^{1/2-20\varepsilon/\varphi^2}}{k}\big)$，只要

$$R\ \ge\ \frac{c\,k^2\,n^{-1+40\varepsilon/\varphi^2}}{\sigma_{err}^2\,M},$$

则以 $\ge 0.99$ 概率有 $|Z-\langle M^t\mathbf{1}_x,M^t\mathbf{1}_y\rangle|\le\sigma_{err}$。其有效之处在于：精度只由总条数 $R$ 决定，而内存只由批大小 $M$ 决定，两者被**解绑**；于是把 $M$ 选得远小于 $\sqrt{n}$，就得到权衡 $M\cdot R\approx n$。`EstColliProb`（Alg. 2）把这一估计推广到一组采样顶点 $I_S$（$|I_S|=s$），对每一对 $(s_i,s_j)$ 调 `EstRWDot`，重复 $O(\log n)$ 次取逐元素中位数（median trick）得到稳健的 Gram 矩阵 $G\approx(M^tS)^T(M^tS)$，空间 $\tilde{O}(M\cdot s^2)$。该分批思想借鉴 Canonne & Yang (2024) 的均匀性检验，但本文是首个把它用于图上随机游走的严格分析。

**2. 小内存点积预言机：从碰撞概率反解谱嵌入点积**

有了碰撞概率，还要把它变成谱聚类真正需要的谱嵌入点积 $\langle f_x,f_y\rangle$（$f_x=U_{[k]}^T\mathbf{1}_x$ 是前 $k$ 个特征向量给出的嵌入）。`InitOracle`（Alg. 3）先用 `EstColliProb` 在 $s$ 个均匀随机采样的地标顶点上算出 Gram 矩阵 $G$，对 $\tfrac{n}{s}G$ 做特征分解 $\tfrac{n}{s}G=\hat W\hat\Sigma\hat W^T$，构造紧凑矩阵

$$\Psi\ =\ \tfrac{n}{s}\,\hat W_{[k]}\,\hat\Sigma_{[k]}^{-2}\,\hat W_{[k]}^T\ \in\ \mathbb{R}^{s\times s},$$

它就是预处理阶段唯一要长期保存的对象（亚线性大小 $n^{O(\varepsilon/\varphi^2)}\cdot\mathrm{polylog}$）。查询时 `QueryDot`（Alg. 4）对 $x,y$ 各算出它们到 $s$ 个地标的碰撞概率向量 $\alpha_x,\alpha_y$（同样取 $O(\log n)$ 次中位数），输出 $\langle f_x,f_y\rangle_{apx}=\alpha_x^T\Psi\,\alpha_y$。Theorem 3.2 保证 $|\langle f_x,f_y\rangle_{apx}-\langle f_x,f_y\rangle|\le \xi/n$，且初始化与查询的空间-时间为

$$S\ =\ \big(\tfrac{k}{\xi}\big)^{O(1)} n^{O(\varepsilon/\varphi^2)} M\,\mathrm{polylog}(n),\qquad T\ =\ \big(\tfrac{k}{\xi}\big)^{O(1)}\frac{n^{1+O(\varepsilon/\varphi^2)}}{M}\cdot\frac{1}{\varphi^2}\,\mathrm{polylog}(n).$$

可见乘积 $S\cdot T=\tilde{O}(n^{1+O(\varepsilon/\varphi^2)})$ 不依赖 $M$——$M$ 只是沿权衡曲线滑动的旋钮。相比 Gluch et al. (2021) 的点积预言机至少 $\tilde\Omega(\sqrt{n})$ 空间，这里可低至远小于 $\sqrt{n}$，壁垒由此打破。注意为保证亚线性查询时间，需 $M\ge n^{c\varepsilon/\varphi^2}$（否则 $n^{1+O(\varepsilon/\varphi^2)}/M$ 不再亚线性）。

**3. 相似度图聚类预言机：把新点积原语即插即用**

最上层的聚类逻辑几乎原样复用 Shen & Peng (2023)，只是把它们的点积预言机替换成本文的小内存版本——这正体现"瓶颈在点积、不在聚类"的判断。其依据是一条结构性质：在 $(k,\varphi,\varepsilon)$-可聚类图中，对大多数顶点，若 $x,y$ 同簇则 $\langle f_x,f_y\rangle\approx\tfrac{k}{n}$，否则 $\approx 0$。于是预处理阶段（`ConstructOracle`）采样 $s=\tfrac{k\log k}{\gamma}$ 个顶点构成集合 $S$，对每一对 $u,v\in S$ 用 `QueryDot` 估 $\langle f_u,f_v\rangle_{apx}$，值大就在初始为空的相似度图 $H=(S,\varnothing)$ 上连一条边；$H$ 的连通分量即对应各簇。查询 $\textsc{WhichCluster}(G,x)$ 时，`Search` 把 $x$ 和 $O(\tfrac{k\log k}{\gamma})$ 个地标比一遍，依 $H$ 的连通分量定簇。整个 $D$（含 $\Psi$、相似度图 $H$ 等）用 $O\!\big(n^{O(\varepsilon/\varphi^2)} M\,\mathrm{poly}(\tfrac{k\log n}{\gamma})\big)$ 比特空间构造，每次查询 $O\!\big(n^{1+O(\varepsilon/\varphi^2)}\tfrac{1}{M}\,\mathrm{poly}(\tfrac{k\log n}{\gamma\varphi})\big)$ 时间。关键是：引入空间约束**只影响空间与查询时间，不影响其余保证**——电导隙与误分类率与原方法完全一致，分别为 $O(\log k\cdot\varepsilon)|C_i|$（对应 Gluch 路线，Item 1）或 $O(\mathrm{poly}(k)\cdot\varepsilon^{1/3})|C_i|$（对应 Shen & Peng 路线，Item 2）。

**4. 区分 1 簇 vs 2 簇及匹配下界：证明权衡几乎最紧**

作为主结果的推论，本文给出区分"单簇 $\varphi$-扩张图"与"两个等大不相交 $\varphi$-扩张图之并"的亚线性算法（Alg. 5），并配一个匹配下界，从而说明上面的空间-时间权衡本质上无法再改进。算法思路是把任务归约为检测 $M^t$ 的谱隙：取 $t=O(\log n/\varphi^2)$，由 Cheeger 不等式，1 簇时 $v_2(M)\le 1-\varphi^2/4$ 故 $v_2(M^t)\le n^{-10}$，2 簇时 $v_2(M^t)$ 恒为 $1$——两种情形谱上判然有别。算法不显式存 $M^t$，而是用分批技巧建一个小代理矩阵 $G\approx(M^tS)^T(M^tS)\in\mathbb{R}^{O(\log n)\times O(\log n)}$，直接对它特征分解，若 $\big(v_2(\tfrac{n}{s}G)\big)^2<0.6$ 判为 1 簇。该算法用 $\tilde{O}(M)$ 空间、$\tilde{O}(n/M)$ 时间（Theorem 1.2）。

下界（Theorem 1.3）则证明：任何仅通过随机游走预言机求解该问题、错误率 $\le 1/3$ 的算法必满足 $S\cdot T\ge\Omega(n)$。证明把图上随机游走归约到**空间受限的分布检验**——区分"全集上均匀分布"与"两半各自均匀"。其核心是一个新的归约：构造成对的困难实例，并用**归纳耦合论证**保证两种情形下随机游走历史在全变差距离上始终难以区分，从而每次观测在空间约束 $S$ 下最多提供 $O(S/n)$ 的判别信息，于是 $T\cdot O(S/n)=\Omega(1)$ 给出 $S\cdot T=\Omega(n)$。由于一条随机游走查询可用 $O(\log n)$ 次邻接表查询模拟，上界与下界在 $\mathrm{polylog}$ 因子内吻合，说明基于碰撞概率估计这一类方法的权衡几乎最紧。

### 损失函数 / 训练策略
本文为纯理论/算法工作，不涉及训练与损失函数；核心"超参"是权衡参数 $M$（批大小），它单调换取空间与查询时间，以及游走长度 $t=20\log n/\varphi^2$ 和中位数重复次数 $O(\log n)$。

## 实验关键数据

实验用随机块模型（SBM）合成图验证空间-时间权衡，仅作理论佐证（服务器 Intel Xeon Platinum 8562Y / 768 GB RAM，每个数据取 5 次独立运行平均）。主图为 $n=3000,\ k=3,\ p=0.07$（簇内边概率），$q=0.002$（簇间边概率），三个簇各 1000 顶点。对比对象是 Shen & Peng (2023) 的原始预言机与本文的小内存变体（两者采样顶点数、游走长度、中位数重复次数都相同，只差在空间-时间参数上）。

### 主实验（空间效率）

以 10400 词为 $1\times$ 基线，记录构造 $D$ 实际占用的"词数"作为空间 $S$ 的代理：

| 预言机 | 空间(词数) | × 基线 | 构造成功率 | 准确率 |
|--------|-----------|--------|-----------|--------|
| 本文 | 9900 | 0.95× | 1 | 0.9833 |
| 本文 | 10100 | 0.97× | 1 | 0.9900 |
| 本文 | 10400 | 1× | 1 | 0.9907 |
| 之前 | 34840 | 3.35× | 0 | 0（构造失败） |
| 之前 | 43888 | 4.22× | 0.6 | 0.9860 |
| 之前 | 44383 | 4.27× | 1 | 0.9997 |
| 之前 | 61223 | 5.89× | 1 | 1.0000 |

本文预言机在约 $1\times$ 空间下即达 0.98+ 准确率并稳定构造成功；旧方法即便给到 $3.35\times$ 基线空间仍**构造失败**（成功率 0），要达到可比准确率需 $4.27\times$ 空间——即本文把构造 $D$ 的空间需求压低了约 4 倍而不损精度。

### 消融实验（空间-时间权衡）

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 改变权衡参数 $M$ | 准确率 0.9833 ∼ 1 | 各参数设置下准确率均稳定保持高位 |
| $S$ vs $T$（Figure 1） | $S\cdot T\approx\tilde{O}(n^{1+O(\varepsilon)})$ | 以"每次查询的随机游走总条数"作 $T$ 的代理，$S$ 与 $T$ 成反比 |

### 关键发现
- **空间换时间的旋钮真实存在**：Figure 1 显示内存随查询时间增大而减小、反之亦然，曲线与理论 $S\cdot T=\tilde{O}(n^{1+O(\varepsilon)})$ 吻合，且全程准确率保持 0.9833∼1。
- **旧方法的"硬下限"被直观验证**：旧预言机在空间不足时不是准确率缓降，而是直接构造失败（相似度图 $H$ 的连通分量过多或过少导致 $D$ 建不出来），这正对应 $\Omega(\sqrt{n})$ 的空间壁垒。
- 未实验 $\log(k)$-电导隙那一版（Item 1），因其构造 $D$ 的运行时间含 $2^{\mathrm{poly}(k)}$ 因子不实用。

## 亮点与洞察
- **定位瓶颈比解决瓶颈更关键**：作者识别出 $\sqrt{n}$ 空间下界根本不在聚类逻辑、而在点积/碰撞概率估计这一底层原语，于是只需替换一个子程序、上层框架原样复用，就拿下整套改进——这种"找到真正承重墙再动它"的思路极具复用价值。
- **分批解绑"精度↔内存"是可迁移的通用手法**：把 $R$ 条样本切成 $R/M$ 批、只暂存一批、对批求平均，使内存只由批大小 $M$ 决定而精度只由总数 $R$ 决定。这一招源自分布检验，本文首次严格用于图上随机游走，几乎可移植到任何"靠大量蒙特卡洛样本估期望"的亚线性算法。
- **上界配匹配下界**：通过把随机游走归约到空间受限分布检验、用归纳耦合控制全变差距离，证明 $S\cdot T=\Omega(n)$，让"权衡几乎最紧"不只是经验观察而有理论背书。

## 局限与展望
- **依赖较强的可聚类假设**：要求图是 $(k,\varphi,\varepsilon)$-可聚类、簇大小平衡、外部电导 $\varepsilon$ 很小（$\varepsilon/\varphi^2\le 10^{-5}$ 量级），现实图未必满足，重叠社区或层次结构超出适用范围。
- **常数与 polylog 偏大**：空间含 $(k/\xi)^{O(1)}$、$n^{O(\varepsilon/\varphi^2)}$、$\log^4 n$ 等因子，理论上"远小于 $\sqrt{n}$"在中等规模 $n$ 上的实际收益可能被常数吞掉；实验也只到 $n=3000$ 的合成 SBM 图，未在真实大图上验证。
- **下界仅覆盖随机游走类方法**：$S\cdot T=\Omega(n)$ 是针对"只用随机游走/碰撞概率估计"的自然方法证的，是否存在跳出这一框架、突破该权衡的算法仍开放。
- **改进方向**：放松电导隙到更大 $\varepsilon$、把分批技巧推广到有向/带权图与层次聚类预言机、在真实万亿边图上工程化验证。

## 相关工作与启发
- **vs Peng (2020)**：第一个鲁棒亚线性谱聚类预言机，空间与查询时间均 $\tilde{O}_\varphi(\sqrt{n}\cdot\mathrm{poly}(k/\varepsilon))$，误分类 $O(kn\sqrt{\varepsilon})$；本文把空间可压到远小于 $\sqrt{n}$ 并给出可调权衡。
- **vs Gluch et al. (2021)**：点积预言机用 $\Omega(n^{1-\delta})$ 空间、误分类 $O(\log k\cdot\varepsilon)|C_i|$；本文 Item 1 复用其聚类框架但换上小内存点积原语，误分类不变而空间-时间满足 $S\cdot T=\tilde{O}(n^{1+O(\varepsilon)})$。
- **vs Shen & Peng (2023)**：误分类 $O(\mathrm{poly}(k)\cdot\varepsilon^{1/3})|C_i|$、空间 $\Omega(n^{1-\delta})$；本文 Item 2 直接以其聚类预言机为骨架、仅替换点积预言机，是实验主对比对象。
- **vs Canonne & Yang (2024)**：均匀性检验中用分批降低内存；本文借其思想但首次严格分析图上随机游走，并把它嵌入谱聚类全流程。
- **vs 空间-时间权衡谱系（Raz 2017 奇偶学习；Diakonikolas et al. 2019 分布检验下界）**：本文把这类"内存受限下需要多少观测"的视角引入亚线性谱聚类，并贡献了随机游走→空间受限分布检验的新归约。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次打破谱聚类预言机的 $\sqrt{n}$ 空间壁垒，给出可调权衡并配匹配下界。
- 实验充分度: ⭐⭐⭐ 仅合成 SBM 图、规模偏小，作为理论佐证够用但不充分。
- 写作质量: ⭐⭐⭐⭐ 技术综述清晰，瓶颈定位与归约思路讲得透彻。
- 价值: ⭐⭐⭐⭐ 为内存受限/大图聚类提供了原理性新工具，分批解绑手法可迁移。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Dynamical properties of dense associative memory](dynamical_properties_of_dense_associative_memory.md)
- [\[ICLR 2026\] Oracle-Efficient Hybrid Online Learning with Constrained Adversaries](oracle-efficient_hybrid_learning_with_constrained_adversaries.md)
- [\[ICLR 2026\] On the Spectral Differences Between NTK and CNTK and Their Implications for Point Cloud Recognition](on_the_spectral_differences_between_ntk_and_cntk_and_their_implications_for_poin.md)
- [\[ICLR 2026\] Memory-Statistics Tradeoff in Continual Learning with Structural Regularization](memory-statistics_tradeoff_in_continual_learning_with_structural_regularization.md)
- [\[ICLR 2026\] A Biologically Plausible Dense Associative Memory with Exponential Capacity](a_biologically_plausible_dense_associative_memory_with_exponential_capacity.md)

</div>

<!-- RELATED:END -->
