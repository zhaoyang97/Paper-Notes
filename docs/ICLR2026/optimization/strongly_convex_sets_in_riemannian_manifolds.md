---
title: >-
  [论文解读] Strongly Convex Sets in Riemannian Manifolds
description: >-
  [ICLR 2026][优化/理论][强凸集] 这篇论文首次系统地把"集合的强凸性"从欧氏（Hilbert）空间推广到黎曼流形，给出三种互不等价的强凸集定义、它们之间的蕴含关系、一套构造性识别工具（光滑强凸函数的下水平集是强凸集），并证明了在（近似）黎曼 scaling 不等式下黎曼 Frank-Wolfe（RFW）算法可以**线性收敛**。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "强凸集"
  - "黎曼流形"
  - "scaling 不等式"
  - "Frank-Wolfe"
  - "线性收敛"
---

# Strongly Convex Sets in Riemannian Manifolds

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=4sDszSYKP6](https://openreview.net/forum?id=4sDszSYKP6)  
**领域**: 优化理论 / 黎曼优化  
**关键词**: 强凸集, 黎曼流形, scaling 不等式, Frank-Wolfe, 线性收敛

## 一句话总结
这篇论文首次系统地把"集合的强凸性"从欧氏（Hilbert）空间推广到黎曼流形，给出三种互不等价的强凸集定义、它们之间的蕴含关系、一套构造性识别工具（光滑强凸函数的下水平集是强凸集），并证明了在（近似）黎曼 scaling 不等式下黎曼 Frank-Wolfe（RFW）算法可以**线性收敛**。

## 研究背景与动机

**领域现状**：在欧氏/Hilbert 空间里，强凸性（strong convexity）是凸优化的核心结构，无论作用在范数、函数还是**集合**上，都能带来更快的收敛、更紧的泛化界和更强的集中不等式。特别地，当约束集合是强凸集时，Frank-Wolfe（FW）这类无投影方法能从 $O(1/T)$ 提速到 $O(1/T^2)$，甚至在最优点落在集合外时直接线性收敛。集合强凸性在 Hilbert 空间里有多个**等价**刻画（球内含定义、scaling 不等式等），早已被研究透彻。

**现有痛点**：现代机器学习越来越多地把问题放在非欧氏空间——最优传输的 Wasserstein 几何、流形上的分布、SPD 矩阵数据、双曲嵌入、机器人运动技能等都依赖黎曼优化。但"函数的强凸性"虽然能较平滑地推广到测地空间，"**集合的强凸性**"在欧氏空间之外却没有一个公认定义。CAT(0)/非正曲率空间里有若干替代提法（Aleksandrov 1957 及后续），但它们彼此**不等价**、推论各异，且都不能直接给出现代优化算法所需的结构性质。

**核心矛盾**：欧氏空间里集合强凸的多个定义之所以等价，靠的是直线、范数、法锥这些"平直"工具；一旦换到弯曲流形，"直线"变成测地线、"加法"要走指数映射、平行移动会引入曲率扭曲，原本等价的定义就分裂开来，而且没有哪一个能同时兼顾"几何直观"和"可被算法利用"。

**本文目标**：在唯一测地（uniquely geodesic）集合上，(1) 给出几个能在欧氏退化时塌缩回经典定义的强凸集定义；(2) 厘清它们之间的蕴含关系；(3) 提供易于验证的识别工具；(4) 证明这些结构能转化为算法收敛加速。

**切入角度**：作者把欧氏定义里的三个要素逐一"黎曼化"——把直线段 $(1-t)x+ty$ 换成测地线 $\gamma(t)$，把"加一个球"换成在切空间 $T_{\gamma(t)}M$ 上加球再用指数映射推回流形，把法锥 scaling 不等式换成对数映射版本，从而得到一族定义；再用"双指数映射"的残差项来量化欧氏与黎曼之间的偏差。

**核心 idea**：用测地线、指数/对数映射和切空间重写欧氏强凸集的三种刻画，得到 **Geodesic / Riemannian / Double Geodesic** 三种黎曼强凸集，配上**黎曼 scaling 不等式（RSI）及其近似版**作为"算法友好"的桥梁，把集合的几何结构直接翻译成 RFW 的线性收敛。

## 方法详解

### 整体框架

本文是一篇纯理论论文，没有数据流水线，整条逻辑链是"**定义 → 关系 → 识别 → 算法**"四步：

先回顾欧氏空间里集合 $\alpha$-强凸的两个经典等价定义（Proposition 1.1）：(a) 对集合内任意两点 $x,y$ 连成的线段上任一点，都能塞进一个半径 $\propto \alpha t(1-t)\|x-y\|^2$ 的球仍在集合内；(b) scaling 不等式 $\langle w; v-x\rangle \ge \alpha\|w\|\|v-x\|^2$（$v$ 是 $w$ 方向上的支撑点）。本文工作建立在一个标准假设上——集合 $C$ 紧致且唯一测地凸（Assumption 1.2，保证对数映射 $\mathrm{Exp}_x^{-1}$ 处处良定义），外加一个距离等价假设（Assumption 1.3，允许用不同于黎曼距离的 $d$，常数 $\ell_M, L_M$ 刻画偏差）。

在此之上，第 2 节把欧氏定义"黎曼化"成三种强凸集（详见关键设计 1）；第 2 节末用 RSI 把几何性质接到算法分析（关键设计 2）；第 3 节引入双指数映射与 Approximate RSI，弥合"双测地强凸"与 RSI 之间的缺口；第 4 节给出构造性判据——光滑强凸函数的下水平集是强凸集（关键设计 3）；第 5 节落地到算法，证明 RFW 在（近似）RSI 下线性收敛（关键设计 4），并用球面与 SPD 上的数值实验验证。

三种定义按"强到弱"排成一条蕴含链（Eq. 7）：

$$\text{Riemannian str. cvx} \xRightarrow{\text{Hadamard}} \text{Geodesic str. cvx} \xLeftrightarrow{\text{A.1.3}} \text{Double Geodesic str. cvx} \Rightarrow \text{Approx. RSI},$$

同时 Riemannian str. cvx $\Rightarrow$ RSI $\Rightarrow$ Approx. RSI。Riemannian 强凸是最强、最受限的（要求对数像是欧氏强凸集），Approximate RSI 最弱、但已足够撑起算法收敛。

### 关键设计

**1. 三种黎曼强凸集定义：把欧氏的"线段+加球"按几何结构的使用程度逐级抬升**

欧氏定义之所以在流形上分裂，是因为"塞进一个球"既可以用流形的度量（测地球）来理解，也可以用切空间结构（指数映射推回）来理解，二者在弯曲空间里不再一致。作者据此给出三个层次：

- **Geodesic strong convexity（Definition 2.1）**：只用流形的测地度量结构，不碰切空间。要求对任意 $x,y\in C$ 连成的测地线 $\gamma$ 与 $t\in[0,1]$，度量球 $\{z\in M\mid d(\gamma(t),z)\le \alpha t(1-t)d^2(x,y)\}\subseteq C$。这是最贴近"度量空间"的提法。
- **Riemannian strong convexity（Definition 2.2）**：最强的一档，借助指数映射。要求对每个 $x\in C$，对数像 $\mathrm{Exp}_x^{-1}(C)\subseteq T_xM$ 在切空间里按欧氏定义 (1) 是 $\alpha$-强凸集。直观上，只有当流形曲率有界、$C$ 不太大、对数映射只轻微扭曲 $C$ 时才成立。
- **Double Geodesic strong convexity（Definition 2.3）**：用切空间+指数映射，但更贴近欧氏代数形式。对测地线 $\gamma$ 上任一点 $\gamma(t)$，在其切空间 $T_{\gamma(t)}M$ 里取半径 $\alpha t(1-t)d^2(x,y)$ 的球，球内每个 $z$ 经 $\mathrm{Exp}_{\gamma(t)}(z)$ 推回都落在 $C$ 内。"double"指这个点由两条测地线生成——一条连 $x,y$，一条从 $\gamma(t)$ 出发。

三者之间的关系（Prop. 2.5–2.7）是本设计的关键产出：Riemannian 强凸在 Cartan-Hadamard（完备非正曲率）流形上蕴含 Geodesic 强凸（常数缩放为 $\alpha\ell_M/L_M^2$）；Geodesic 与 Double Geodesic 在 Assumption 1.3 下**相互蕴含**（常数差一个 $\ell_M$ 或 $1/L_M$ 因子），这一点很值得注意，因为前者只依赖度量结构、后者依赖流形结构，它们在温和条件下殊途同归。当 $M$ 退化为欧氏空间时，三者全部塌缩回 (1)。

**2. 黎曼 scaling 不等式（RSI）与 Approximate RSI：把"算法友好"的支撑刻画搬到流形上**

欧氏空间里 scaling 不等式（Prop. 1.1(b)）是分析算法收敛最顺手的强凸刻画，但它在流形上不再与全局强凸**等价**。作者定义 RSI（Definition 2.4）作为欧氏 (2) 的黎曼对应：对所有 $x\in C$、$w\in T_xM$，

$$\langle w; \mathrm{Exp}_x^{-1}(v)\rangle_x \ge \alpha\|w\|_x\,\|\mathrm{Exp}_x^{-1}(v)\|_x^2,\quad v\in\arg\max_{z\in C}\langle w;\mathrm{Exp}_x^{-1}(z)\rangle_x.$$

它把欧氏内积/范数换成切空间内积、把 $v-x$ 换成对数映射 $\mathrm{Exp}_x^{-1}(v)$。Prop. 2.5 证明 Riemannian 强凸 $\Rightarrow$ RSI，于是几何结构被翻译成算法可用的不等式。

但（双）测地强凸与 RSI 之间没有显然的直接联系，缺口出在"两条测地线拼起来"不等于"切空间里两个向量相加"。作者用**双指数映射**（Definition 3.1）$\mathrm{Exp}_x(u,v):=\mathrm{Exp}_{\mathrm{Exp}_x(u)}(\Gamma^{\mathrm{Exp}_x(u)}_x v)$（先沿 $u$ 走、再平行移动 $v$ 后沿其走）刻画这一拼接，并把它写成"指数映射算子" $h_x(u,v)=u+v+R_x(u,v)$，残差 $R_x$ 衡量与欧氏相加的偏差（常曲率空间如球面、罗氏空间有显式 Taylor 展开）。据此定义 **Approximate RSI**（Definition 3.2）：在 RSI 右端加一个残差项 $\langle w; r(x)\rangle_x$。Prop. 3.3 证明 Double Geodesic 强凸 $\Rightarrow$ Approximate RSI，残差 $r(x)$ 由 $R_x$ 在特定方向取值给出。当 $x,v$ 靠近时残差相对主项 $\alpha\|w\|_x d(x,v)^2$ 可忽略，这正是后面算法局部线性收敛的关键。

**3. 下水平集判据：把"难以直接验证"的强凸性归约为"函数光滑+强凸"**

直接按定义验证一个集合是否强凸往往很难。作者把欧氏的经典结论——$L$-光滑、$\mu$-强凸函数的下水平集 $Q_s=\{x\mid f(x)-f^*\le s\}$ 本身是强凸集（Journée 等 2010）——推广到流形（Theorem 4.2）。具体地，若 $f$ 在满足 Assumption 1.2 的 $C$ 上是真、测地 $L$-光滑、$\mu$-强凸，$x^*$ 满足 $\nabla f(x^*)=0$，且 $Q_s$ 测地严格凸，则 $Q_s$ 是测地强凸集，常数

$$\alpha = \frac{\mu}{2\,(2sL\max\{\ell_M^{-2};1\})^{1/2}}.$$

证明依赖一条黎曼光滑性引理（Lemma 4.1）：$\|\nabla f(x)\|_x\le\sqrt{2L(f(x)-f(x^*))}$，它来自流形上函数对偶性的工具。这个判据是"构造性"的——给了一条把强凸函数变成强凸集的可操作路径。论文给出两个具体例子：单位球面上 $f(x)=d^2(x_0,x)$ 在 $s<(\pi/2)^2$ 时的下水平集，以及 SPD 矩阵流形（Cartan-Hadamard）上平方距离的下水平集，都是测地强凸集。此外 Theorem D.4 还给出另一条独立判据：有界曲率流形上半径足够小的测地球是 Riemannian 强凸集（涵盖球面、双曲空间、Grassmann/Stiefel 流形、特殊正交群、SPD 锥）。

**4. RFW 在（近似）RSI 下的线性收敛：几何结构兑现为算法加速**

前三步建立的结构最终要回答"有什么用"。黎曼 Frank-Wolfe（RFW，Algorithm 1）在每步求线性极小化预言机 $v_t=\arg\min_{v\in C}\langle\nabla f(x_t);\mathrm{Exp}_{x_t}^{-1}(v)\rangle$，再沿测地线短步更新。已有工作（Weber & Sra 2023）证明它在凸光滑下 $O(1/T)$、函数强凸且最优点在内部时线性收敛——但这些都是"函数侧"的强凸。本文补上"**集合侧**"强凸带来的加速：

- **Theorem 5.1（精确 RSI）**：若 $C$ 满足 RSI，且 $f$ 的无约束最优点落在 $C$ 外（存在 $c>0$ 使 $\min_{x\in C}\|\nabla f(x)\|_x>c$），则 RFW 全局线性收敛，$f(x_{t+1})-f(x^*)\le (f(x_t)-f(x^*))\cdot\max\{1/2,\,1-\alpha c/(2L)\}$。这把欧氏的 Demyanov-Rubinov（1970）结论搬到了黎曼设定。
- **Theorem 5.2（近似 RSI）**：当只满足 Approximate RSI（残差满足 $\|R_x(u,w)\|_x\le C\max\{\|u\|^2\|w\|;\|w\|^2\|u\|\}$）时，经过一段由 $O(1/T)$ 全局率保证的 burn-in，使迭代点足够靠近（$d(x_t,v_t)^2$ 小于某曲率相关阈值）后，RFW 转为**局部线性收敛**，速率同上式。

关键之处在于：常数无需事先知道，一旦条件满足 RFW 自动切换到线性率。当集合是 Riemannian 强凸时残差界 (12) 自动成立，精确与近似 RSI 同时满足；而在常曲率空间（球面、双曲）上即便精确 RSI 失效，近似 RSI 仍可成立——这把可用范围扩到了 CAT(0) 之外。

## 实验关键数据

本文是理论论文，实验为"验证理论速率"的数值算例，没有 benchmark 对比表。核心是两个最小化算例，配合一张三定义对照表。

### 三种强凸集定义对照（Table 1 概括）

| 定义 | 是否用切空间 | 核心工具 | 含义 | 典型例子 |
|------|------|------|------|------|
| Geodesic str. cvx | 否 | 测地度量 | 测地点周围存在测地球含于集合 | 同 Double Geodesic |
| Double Geodesic str. cvx | 是 | Exp + 平行移动 | 切空间球经指数映射含于集合 | 光滑强凸函数的下水平集 |
| Riemannian str. cvx | 是 | Log 映射 | 集合的对数像是欧氏强凸集 | 有界曲率流形上的小测地球 |

### 数值算例

| 算例 | 流形 / 设置 | 观察到的现象 | 对应理论 |
|------|------|------|------|
| 球面二次最小化 | $S^{n-1}$，$n=500$，$f(x)=\|A(x-x^\star)\|^2/2$，$A\in\mathbb{R}^{250\times500}$，约束 $d(x,x_c)\le R<\pi/2$ | 收敛曲线呈现两段：先 $O(t^{-1})$ 全局率，后**局部线性**率，两段清晰可辨 | Theorem 5.2 |
| 有限温度自由能最小化 | SPD 流形上测地球约束，最小化 $F(X)=E_\mu(X)-TS(X)$（电子结构密度矩阵） | 线性化能量+精确熵的信赖域子问题在测地球（强凸集）上按理论速率收敛 | Theorem 4.2 + 5.x |

### 关键发现
- **两段收敛曲线**是核心证据：RFW 在 burn-in 阶段走全局 $O(1/T)$ 率，迭代点进入"$x_t,v_t$ 足够近"的区域后自动切换到局部线性率，与 Theorem 5.2 的"近似 RSI + burn-in"预测完全吻合。
- **最优点在边界**是触发线性率的前提：实验中刻意取 $R=0.9\,\mathrm{dist}(x^\star,x_c)$，保证解落在约束边界上、$\arg\min\|\nabla f\|>c$，这正对应"无约束最优点在集合外"的假设。
- **实际可落地的场景**：球面约束二次问题对应带层级标签的神经网络训练（"每个类别的分类器落在以其超类分类器为中心的球面上"）；SPD 算例对应电子结构计算中的有限温度自由能最小化，说明该理论不只是抽象推广。

## 亮点与洞察
- **把"等价"在流形上"解耦"成层级**：欧氏空间里多个集合强凸定义等价，作者敏锐地指出这种等价依赖平直结构，弯曲后会分裂，于是不强行只留一个，而是给出三档定义并厘清蕴含关系——这种"承认分裂、再排序"的处理比硬凑一个统一定义更诚实也更有用。
- **双指数映射的残差是点睛之笔**：用 $h_x(u,v)=u+v+R_x(u,v)$ 把"两条测地线拼接 ≠ 切空间向量相加"这件事量化成一个可控残差，从而把弱化的 Approximate RSI 与算法收敛接上，是连接几何与算法的关键桥。
- **常数自适应**：RFW 无需知道 $\alpha,c,C$ 等常数，满足条件即自动切换到线性率，这让理论结果在实践中不需要繁琐调参就能兑现。
- **可迁移**：作者明确指出强凸集结构在欧氏侧被利用的所有场景（广义幂法、在线学习、strongification 技巧）都有望搬到黎曼侧，给出了清晰的后续方向。

## 局限与展望
- **强假设**：全程要求集合紧致且唯一测地凸（Assumption 1.2），这排除了正曲率流形上较大的集合（指数映射不再单射）；Riemannian 强凸更进一步要求曲率有界、集合足够小。
- **常数依赖曲率与尺度**：下水平集判据的强凸常数 $\alpha\propto \mu/\sqrt{sL}$ 随集合变大（$s$ 增大）而退化，近似 RSI 的 burn-in 阈值也依赖曲率上界 $K$，实际收益对几何尺寸敏感。
- **实验偏验证性**：数值算例旨在"印证理论速率"，规模与对比都有限，缺少与其它黎曼优化方法（如黎曼梯度/信赖域）在真实任务上的端到端比较，工程价值还需更多验证。
- **Approximate RSI 的残差界仅在常曲率/对称空间有显式形式**：一般流形上残差展开 (Gavrilov 2007 的级数) 是否好用、$C$ 常数多大，仍是开放问题。

## 相关工作与启发
- **vs CAT(0) / Aleksandrov（1957）等度量空间提法**：CAT(0) 文献从"环境空间非正曲率"全局推出凸性；本文是其**集合级细化**，分析局部曲率与指数映射如何共同诱导特定子集的强凸性，在非正曲率时能恢复度量凸性，又能延伸到 CAT(0) 之外。
- **vs Paris（2020/2021）在线学习里的测地凸**：他们用测地凸但不量化集合曲率、也不给算法强凸条件；本文补上了"集合曲率刻画 + 算法可用条件"。
- **vs proximal smoothness（Davis 等 2025）**：二者结构上"对偶"——proximal smoothness 要求在所有外向切方向上一致的**上界**，RSI 只在受限切子集的 LMO 最大化点处给出单侧**下界**，分别服务不同算法机制。
- **vs Weber & Sra（2023）的 RFW 分析**：他们证的是"**函数**强凸"下的 RFW 率；本文补上"**集合**强凸"带来的加速（最优点在外时线性、强凸函数时 $O(1/T^2)$ 的黎曼对应），与之互补。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个系统研究黎曼流形上集合强凸性的工作，填补了非欧氏设定下的概念空白。
- 实验充分度: ⭐⭐⭐ 理论论文，数值算例能印证速率但规模有限、缺横向比较。
- 写作质量: ⭐⭐⭐⭐ 逻辑链"定义→关系→识别→算法"清晰，定理与例子搭配得当，唯证明全在附录。
- 价值: ⭐⭐⭐⭐ 为黎曼优化提供了可被算法利用的结构基础，后续 strongification 类工作有明确迁移路径。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds](riemannian_zeroth-order_gradient_estimation_with_structure-preserving_metrics_fo.md)
- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](../../NeurIPS2025/optimization/an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[ICLR 2026\] Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference](adaptive_gradient_descent_on_riemannian_manifolds_and_its_applications_to_gaussi.md)
- [\[AAAI 2026\] GHOST: Solving the Traveling Salesman Problem on Graphs of Convex Sets](../../AAAI2026/optimization/ghost_solving_the_traveling_salesman_problem_on_graphs_of_convex_sets.md)
- [\[ICLR 2026\] Riemannian Optimization on Relaxed Indicator Matrix Manifold](riemannian_optimization_on_relaxed_indicator_matrix_manifold.md)

</div>

<!-- RELATED:END -->
