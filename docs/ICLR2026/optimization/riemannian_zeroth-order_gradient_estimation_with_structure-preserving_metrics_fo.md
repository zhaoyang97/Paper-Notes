---
title: >-
  [论文解读] Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds
description: >-
  [ICLR 2026][优化/理论][黎曼零阶优化] 针对黎曼流形上度量"测地不完备"导致指数映射可能把扰动点送出流形、零阶估计器失效的问题，本文构造了一个保持原驻点结构、又测地完备的共形等价度量 $g'$，并在纯内蕴（不依赖嵌入）视角下给出两点对称零阶估计器的均方误差上界（揭示其与流形曲率的关系），配合无偏的拒绝采样，最终把黎曼零阶 SGD 的最优收敛复杂度从欧氏度量推广到一般黎曼度量。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "黎曼零阶优化"
  - "测地完备性"
  - "结构保持度量"
  - "截面曲率"
  - "拒绝采样"
---

# Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=yEfKDCbtv1](https://openreview.net/forum?id=yEfKDCbtv1)  
**领域**: 优化理论 / 黎曼优化 / 零阶优化  
**关键词**: 黎曼零阶优化, 测地完备性, 结构保持度量, 截面曲率, 拒绝采样

## 一句话总结
针对黎曼流形上度量"测地不完备"导致指数映射可能把扰动点送出流形、零阶估计器失效的问题，本文构造了一个保持原驻点结构、又测地完备的共形等价度量 $g'$，并在纯内蕴（不依赖嵌入）视角下给出两点对称零阶估计器的均方误差上界（揭示其与流形曲率的关系），配合无偏的拒绝采样，最终把黎曼零阶 SGD 的最优收敛复杂度从欧氏度量推广到一般黎曼度量。

## 研究背景与动机

**领域现状**：当目标函数包含黑盒模块或不可微外部求解器（如 PDE solver）时，无法拿到显式梯度，只能用零阶方法——通过函数值的差分来估计梯度方向。在黎曼流形上，经典的两点对称零阶估计器为

$$\hat\nabla f(p) = \frac{f(\exp_p(\mu v)) - f(\exp_p(-\mu v))}{2\mu}\, v,$$

其中 $v$ 是切空间 $T_pM$ 上的随机方向，$\mu>0$ 是扰动步长，$\exp_p$ 是把切向量沿测地线送回流形的指数映射（实践中常用 retraction 一阶近似）。

**现有痛点**：流形通常被当作欧氏空间 $\mathbb{R}^n$ 的子流形，继承一个欧氏度量 $g_E$ 来简化数值计算。但 $g_E$ **不一定测地完备**——指数映射 $\exp_p$ 不保证在整个切空间 $T_pM$ 上有定义。于是随机采到的方向 $v$ 一旦太长，$\exp_p(\mu v)$ 就可能落到流形外、变成未定义，零阶估计直接崩。论文给出的三个真实例子都落在这一类：mesh 优化（顶点不能压到三角形边界上）、灌溉喷头布局（喷头不能放到田边界外）、协方差矩阵估计（正定矩阵集 $\mathcal{S}^{++}_d$ 是开锥）。

**核心矛盾**：理论上可以先换一个测地完备的度量（Nomizu–Ozeki 定理保证任意无边光滑流形都存在这样的度量），再用 Nash 嵌入定理找一个等价的完备欧氏度量，从而直接套用已有的完备情形收敛分析。但 Nash 嵌入的构造性证明在数值上极其困难，对实际优化算法不可行。完备性与可计算性之间存在这道鸿沟。

**本文目标**：在给定度量 $g$ 可能测地不完备的前提下，设计一个仍能找到 $g$ 意义下驻点的黎曼零阶优化算法。

**切入角度**：与其去构造一个新的"环境欧氏空间"，不如换一个测地完备但**仍保持原驻点**的度量 $g'$，并把整套梯度估计与收敛分析改写成只依赖流形自身几何（内蕴）、不依赖任何嵌入的形式。

**核心 idea**：用一个共形等价、测地完备、且保持 $\epsilon$-驻点的"结构保持度量" $g'=h g$ 替换原度量，再在内蕴框架下分析零阶估计的误差并证明收敛——既绕开了不完备性，又能把驻点结论搬回原度量 $g$。

## 方法详解

### 整体框架
论文要解决的是：黑盒目标 $\min_{p\in M} f(p)=\mathbb{E}_\xi[f(p;\xi)]$，流形 $(M,g)$ 的度量 $g$（典型是继承的欧氏度量）测地不完备，导致零阶扰动可能出界。整条思路分三步串起来：

第一步，**换度量**——证明对任意 $g$ 都能构造一个结构保持度量 $g'=h(p)\,g$（$h$ 是正光滑函数），它测地完备（扰动永远落在指数映射定义域内）、共形等价（驻点集不变）、并保持 $\epsilon$-驻点。第二步，**内蕴估计**——在新度量 $g'$（不再是原环境空间的欧氏度量）下，重新分析两点对称估计器的均方误差，得到一个只含流形曲率、不含任何嵌入信息的上界，并据此证明 SGD 收敛。第三步，**无偏采样**——由于估计器要求方向 $v$ 在 $g$-单位球面上均匀分布，而朴素的"高斯归一化"在非欧度量下有系统偏差，论文用拒绝采样给出严格均匀的无偏采样。最后用推论把 $g'$ 下的 $\epsilon$-驻点在适当条件下搬回原度量 $g_E$，匹配完备情形的最优复杂度。这是一篇纯优化理论工作，没有多模块流水线，故不配框架图。

### 关键设计

**1. 结构保持度量：换一个测地完备又不丢驻点的共形度量**

直接面对的痛点是：原度量 $g$ 测地不完备，扰动 $\mu v$ 可能把点送出指数映射的定义域。论文定义（Definition 2.5）的结构保持度量 $g'$ 要同时满足三条：(a) **测地完备**——存在 $\rho>0$，使任意 $p$ 处的指数映射定义域都包含半径 $\rho$ 的球 $B_p(\rho)=\{v: \|v\|_g\le\rho\}$，从而只要取扰动步长 $\mu<\rho$，扰动点必在定义域内；(b) **共形等价**——存在正光滑函数 $h$ 使 $g'_p(v,w)=h(p)\,g_p(v,w)$，共形缩放不改变梯度为零的位置，故驻点集完全保留；(c) **$\epsilon$-驻点保持**——任意 $g$ 下的 $\epsilon$-驻点（即 $\sqrt{g_p(\nabla f,\nabla f)}<\epsilon$）也是 $g'$ 下的 $\epsilon$-驻点。

Theorem 2.6 证明这样的 $g'$ 对任意 $g$ 都存在：构造沿用 Nomizu–Ozeki(1961) 的经典做法，但**额外调整共形系数 $h$** 以满足第 (c) 条 $\epsilon$-驻点保持。直觉上（论文 Figure 1）：在这个度量下，从 $p$ 出发的测地线无论方向多长都不会离开概率单纯形，于是零阶扰动永远合法。需要强调的是反向不成立——$g'$ 下的 $\epsilon$-驻点未必是 $g$ 下的 $\epsilon$-驻点，这个不对称留到收敛分析里处理。

**2. 内蕴零阶梯度估计与曲率敏感的 MSE 上界：把误差归因到流形曲率**

换了 $g'$ 之后冒出新麻烦：$g'$ 一般不再是原环境欧氏空间继承来的欧氏度量，已有依赖"嵌入"的零阶分析全部失效。论文的回应是彻底走内蕴路线——不去找新的环境空间，只用流形自身结构分析估计器。在测地完备的一般度量 $g$ 下，对单位向量 $v$ 取对称估计器（式 3），Theorem 2.7 给出其均方误差上界：

$$\mathbb{E}_{v\sim\mathrm{Unif}(S^{d-1})}\Big[\big\|\hat\nabla f(p;v)-\tfrac{1}{d}\nabla f(p)\big\|_p^2\Big]\le \frac{1+\mu^2\kappa^2}{d}\,\|\nabla f(p)\|_p^2 + O(\mu^2),$$

其中 $v$ 均匀采自 $g$ 诱导的单位球面 $S^{d-1}\subset T_pM$，$\kappa$ 是流形 $(M,g')$ 截面曲率绝对值的一致上界（Assumption 2.4），并要求三阶/四阶导有界（Assumption 2.3）。这个界的关键意义在于：曲率项 $\kappa$ 直接量化了局部几何对估计器方差的影响；当 $\kappa=0$（平坦）时上界退化为欧氏空间下零阶估计的经典方差表达。换言之，**度量选得越平，估计越准**——这给"该选哪个共形度量"提供了理论指针。基于此，Theorem 2.9 进一步证明在一般黎曼度量下 SGD 收敛。

**3. $g$-单位球面上的无偏拒绝采样：消除非欧度量下的采样偏差**

估计器要求方向 $v$ 在 $g$-单位球面上**均匀**分布。由于 $g$ 在切空间上定义了双线性型 $g_p(u,v)=u^\top A v$（$A\succ0$，$A_{ij}=g_p(\partial_{x_i},\partial_{x_j})$），$g$-单位球面等价于椭球 $C=\{v\in\mathbb{R}^d: v^\top A v=1\}$。欧氏空间里"采高斯再归一化"很简单，但在非欧度量下这套 rescaling 会让样本**系统性地堆到椭球短轴附近**（论文 Figure 2），引入有偏估计——logistic 目标下甚至直接发散。

论文改用拒绝采样（Algorithm 1）：先对 $A$ 做特征分解 $A=Q\Lambda Q^\top$，令 $L=Q\Lambda^{-1/2}$、$\lambda_{\max}=\max\mathrm{diag}(\Lambda)$；每轮采 $z\sim\mathcal{N}(0,I_d)$、归一化 $s=z/\|z\|$、得候选 $v=Ls$，再采 $u\sim U(0,1)$，当 $u<\sqrt{v^\top A^2 v}/\lambda_{\max}$ 时接受。Proposition 2.8 证明输出严格服从椭球 $C$ 上的均匀分布。这个无偏采样是前面 MSE 上界成立（其中假设 $v$ 均匀）的前提，也是实验里稳定收敛的关键。

**4. 把 $g'$ 下的驻点搬回原度量：匹配完备情形的最优复杂度**

Theorem 2.9 给出在完备度量 $g$ 下 SGD 的收敛率 $\min_t\|\nabla f(p_t)\|_{p_t}^2\lesssim\sqrt{d/T}$（取 $\eta\lesssim\sqrt{d/T}$、$\mu\lesssim \tfrac{1}{d^2}\sqrt{d/T}$）。但实际起点是不完备的欧氏度量 $g_E$，我们构造 $g=h g_E$ 后在 $g$ 下收敛，得到的却是 $g$ 意义下的 $\epsilon$-驻点。Corollary 2.10 给出搬回 $g_E$ 的充分条件：(a) $g_E$ 本身测地完备；或 (b) $g_E$ 下的 $\epsilon$-驻点集 $K=\{p:\|\nabla_{g_E}f(p)\|_{p,g_E}\le\epsilon\}$ 紧致。任一条件下共形系数 $h$ 有一致上界，于是 $g$ 下的 $\epsilon$-驻点（差一个常数缩放）也是 $g_E$ 下的 $\epsilon$-驻点，最终复杂度为 $T\le O(d/\epsilon^4)$——把已知欧氏度量下黎曼零阶 SGD 的最优复杂度推广到了一般黎曼度量这一更广的类。即使两个条件都不满足，Theorem 2.9 仍保证收敛，只是复杂度可能更差。

## 实验关键数据

实验目的是验证理论：采样偏差如何影响收敛、曲率如何影响估计精度，以及在真实测地不完备任务（mesh 优化）上的有效性。

### 主实验

| 实验 | 对比项 | 关键现象 | 对应理论 |
|------|--------|----------|----------|
| 合成·采样偏差 | 拒绝采样 vs rescaling（16 次平均） | 拒绝采样稳定更优；rescaling 在 logistic 目标下同超参直接发散 | Prop. 2.8（无偏） |
| 合成·曲率影响 | 4 个共形等价、不同曲率的度量（50,000 次试验测 MSE） | 截面曲率 $K(p_0)$ 越小，零阶估计 MSE 越低 | Thm. 2.7（$\kappa$ 项） |
| 真实·mesh 优化 | 结构保持 vs 无约束/Reversion/Soft Projection（20,000 步） | 结构保持法全程稳定降误差、最终 MSE 最低 | Thm. 2.6 + 整体框架 |

合成实验用的目标：采样偏差实验取二次型 $f_{\text{quadratic}}$ 与 $f_{\text{logistic}}$（配非欧度量 $g_A(u,v)=u^\top A v$）；曲率实验在概率单纯形上取 KL 目标 $f_{\text{KL}}(p)=\mathrm{KL}(p\|q)$ 与欧氏目标 $f_{\text{Euclidean}}(p)=\tfrac12\|p-q\|^2$（$q=\tfrac1d\mathbf{1}_d$ 为单纯形质心）。

### 消融实验（mesh 优化的 baseline 对比）

任务是优化 $20\times20$ 粗网格节点位置去逼近 $200\times200$ 细网格上 Helmholtz 方程 $\nabla^2 f=-k^2 f$（$k=10$）的真解，节点用其六邻居的凸组合表示（落在概率单纯形上，测地不完备）。

| 方法 | 处理越界的方式 | 表现 |
|------|----------------|------|
| Unconstrained | 不处理 | 频繁违反网格有效性，约第 16,000 步剧烈震荡 |
| Reversion | 越界则退回原位 | 防住非法更新但约 8,000 步后停滞 |
| Soft Projection | 沿方向逐步缩小 $\mu$ 直到合法 | 稳定但推进缓慢，14,000 步后几无改善 |
| 结构保持（本文） | 扭曲黎曼结构使扰动永不出界 | 全程持续降误差、最终 MSE 最低且无失稳 |

### 关键发现
- 采样无偏是"地基"：rescaling 的偏差不止掉点，在 logistic 目标上会让同样超参的训练直接发散，说明非欧度量下采样方案的选择是收敛与否的分水岭，而非锦上添花。
- MSE 上界中的曲率项 $\kappa$ 被实验直接验证——四个共形度量里曲率越小 MSE 越低，理论与经验一致，也反过来支持"把流形选平"这个度量设计指针。
- 处理越界的朴素手段（退回/软投影）要么停滞要么龟速，本质是"事后补救"；结构保持度量把合法性写进几何本身（事前保证），因此既保可行性又保持收敛效率。

## 亮点与洞察
- 把"测地不完备"这个常被忽视、却在 mesh/正定矩阵/布局这类真实问题里普遍存在的坑摆到台面，并给出可计算的解法（构造 $g'$ 而非走不可行的 Nash 嵌入），问题选得准、解法落地。
- MSE 上界 $\tfrac{1+\mu^2\kappa^2}{d}\|\nabla f\|^2$ 把零阶估计误差干净地归因到流形截面曲率，$\kappa=0$ 退化回欧氏经典结论——这条"曲率—方差"联系既漂亮又给出可操作的度量选择准则。
- "非欧单位球面均匀采样"被还原成椭球 $v^\top A v=1$ 上的拒绝采样，简单且可证无偏；这个采样子程序本身可复用到任何需要在一般黎曼度量下采方向的零阶/随机算法。
- 共形等价保驻点 + 调 $h$ 保 $\epsilon$-驻点，是把"换度量"做成"无损"的关键技巧：换地基但不动解集，再用 Corollary 2.10 把结论搬回去。

## 局限与展望
- $g'$ 下的 $\epsilon$-驻点搬回原度量 $g_E$ 需要额外条件（$g_E$ 完备或 $\epsilon$-驻点集紧致），否则复杂度可能更差——"扁平化没有免费午餐"，估计精度与优化动态之间要权衡。
- 分析依赖一串较强假设：Hessian 有界（$L$-光滑）、三/四阶导有界、截面曲率全局有界、retraction 正则性（Assumption 2.2 的 $C_{\text{Ret}}$）；现实黑盒目标未必满足。
- 实验规模偏小（合成问题 + $20\times20$ 网格），偏向验证理论，缺少高维大规模黑盒任务上的实证；拒绝采样的接受率随维度/$A$ 条件数如何变化也未给出经验数据。
- 结构保持度量的构造保证存在性，但具体 $h$ 的选择空间很大，如何在"足够平（小 $\kappa$）"与"搬回 $g_E$ 的常数缩放可控"之间挑最优度量，仍是开放问题。

## 相关工作与启发
- **vs 标准黎曼零阶优化（Li et al. 2023a;b 等）**: 他们普遍假设测地完备 / 依赖欧氏嵌入来定义估计器；本文不要求完备、走纯内蕴分析，把适用范围从"嵌入欧氏度量的流形"扩到"一般黎曼度量的流形"。
- **vs Nomizu–Ozeki + Nash 嵌入路线**: 理论上可先转完备度量再嵌入欧氏空间套已有分析，但 Nash 构造数值上不可行；本文沿用 Nomizu–Ozeki 构造、改共形系数 $h$ 以额外保 $\epsilon$-驻点，避开了嵌入。
- **vs 欧氏零阶估计（Nesterov & Spokoiny 2017）**: 本文 MSE 上界在 $\kappa=0$ 时退化为其经典结论，可视为把欧氏零阶方差分析推广到带曲率的黎曼情形。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统处理测地不完备下的黎曼零阶优化，结构保持度量 + 内蕴曲率敏感上界都是新贡献
- 实验充分度: ⭐⭐⭐ 实验扣理论扣得紧，但规模偏小、偏验证性，缺大规模黑盒任务
- 写作质量: ⭐⭐⭐⭐ 问题动机—构造—分析—搬回的逻辑链清晰，定理与假设交代完整
- 价值: ⭐⭐⭐⭐ 为 mesh/正定矩阵/黑盒布局等不完备流形上的零阶优化提供了可证明可计算的工具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Strongly Convex Sets in Riemannian Manifolds](strongly_convex_sets_in_riemannian_manifolds.md)
- [\[ICLR 2026\] Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference](adaptive_gradient_descent_on_riemannian_manifolds_and_its_applications_to_gaussi.md)
- [\[ICLR 2026\] Scalable Second-Order Riemannian Optimization for K-means Clustering](scalable_second-order_riemannian_optimization_for_k-means_clustering.md)
- [\[ICLR 2026\] Unbiased Gradient Estimation for Event Binning via Functional Backpropagation](unbiased_gradient_estimation_for_event_binning_via_functional_backpropagation.md)
- [\[ICLR 2026\] FZOO: Fast Zeroth-Order Optimizer for Fine-Tuning Large Language Models towards Adam-Scale Speed](fzoo_fast_zeroth-order_optimizer_for_finetuning_large_language_models_towards_ad.md)

</div>

<!-- RELATED:END -->
