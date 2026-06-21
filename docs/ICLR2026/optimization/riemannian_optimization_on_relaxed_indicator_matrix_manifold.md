---
title: >-
  [论文解读] Riemannian Optimization on Relaxed Indicator Matrix Manifold
description: >-
  [ICLR 2026][优化/理论][指示矩阵] 本文提出一种新的指示矩阵松弛——把列和约束从"等于某个固定值"放宽到"落在区间 $(l,u)$ 内"，证明这个松弛集构成一个 $(n-1)c$ 维的嵌入子流形（RIM 流形），并配套给出一整套黎曼优化工具箱，使原本在双随机流形上需 $O(n^3)$ 的梯度/Hessian 计算降到 $O(n)$，在百万级变量的图像去噪与 Ratio Cut 聚类上比双随机流形方法快 70–200 倍且结果更优。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "指示矩阵"
  - "黎曼优化"
  - "流形"
  - "双随机矩阵"
  - "Ratio Cut"
---

# Riemannian Optimization on Relaxed Indicator Matrix Manifold

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ERJd7dMN6U](https://openreview.net/forum?id=ERJd7dMN6U)  
**代码**: 见论文 Appendix H  
**领域**: 优化理论 / 流形优化 / 聚类  
**关键词**: 指示矩阵, 黎曼优化, 流形, 双随机矩阵, Ratio Cut

## 一句话总结
本文提出一种新的指示矩阵松弛——把列和约束从"等于某个固定值"放宽到"落在区间 $(l,u)$ 内"，证明这个松弛集构成一个 $(n-1)c$ 维的嵌入子流形（RIM 流形），并配套给出一整套黎曼优化工具箱，使原本在双随机流形上需 $O(n^3)$ 的梯度/Hessian 计算降到 $O(n)$，在百万级变量的图像去噪与 Ratio Cut 聚类上比双随机流形方法快 70–200 倍且结果更优。

## 研究背景与动机
**领域现状**：指示矩阵 $F\in\mathrm{Ind}_{n\times c}=\{X\mid X_{ij}\in\{0,1\},\,X\mathbf{1}_c=\mathbf{1}_n\}$ 是聚类、分类等机器学习任务的核心对象，但直接优化它是一个 0-1 规划，属于 NP-hard 问题。实践中普遍的做法是把这个离散约束**松弛**成某个连续流形，再在流形上做优化。历史上有三种主流松弛：Stiefel 流形 $\{X\mid X^TX=I\}$（谱聚类的基础）、单随机流形 $\{X\mid X\mathbf{1}_c=\mathbf{1}_n,X>0\}$（Fuzzy K-means）、以及最新的双随机流形 $\{X\mid X\mathbf{1}_c=\mathbf{1}_n,X^T\mathbf{1}_n=r,X>0\}$。

**现有痛点**：这三种松弛各有硬伤。Stiefel 流形优化要 $O(n^2c)$，且只能对 $\mathrm{tr}(F^TLF)$ 形式给解析最优解，聚类时还得 K-means 后处理，难以扩展到大规模。单随机流形对列和不加任何约束，会导致空簇或类别极度不平衡，无法把"每类样本数"这种先验喂进模型。双随机流形把列和**死死钉**在先验分布 $r$ 上——可现实里未知数据集的真实类分布几乎不可能提前知道，约束过严；更要命的是它的黎曼梯度涉及 $(I-FF^T)^\dagger$ 这种 $n\times n$ 伪逆，优化复杂度高达 $O(n^3)$，百万变量下几乎跑不动。

**核心矛盾**：松弛要同时满足两个相互拉扯的诉求——既要**能灵活注入类别先验**（单随机做不到），又要**计算上可扩展**（双随机做不到）。约束越紧（双随机）越能用先验，但计算越贵且对未知分布越脆弱；约束越松（单随机）越廉价，但丢掉了类平衡信息。

**本文目标**：构造一种松弛，使先验信息的注入量可调（从"完全不知道"到"精确知道列和"连续过渡），同时把核心运算复杂度压到线性。

**切入角度**：作者观察到，列和不必是一个**点**约束，可以是一个**区间**约束——把 $X^T\mathbf{1}_n=r$ 放宽为 $l<X^T\mathbf{1}_n<u$。区间宽窄正好对应先验强弱：先验多就取紧区间，先验少就取宽区间，极端情况 $l=u$ 退化为双随机、$l<0,u>n$ 退化为单随机。这个集合恰好是欧氏空间的嵌入子流形，于是可以套黎曼优化的全套机器。

**核心 idea**：用"区间型列和约束"代替"点型列和约束"，得到一个统一、可调、且黎曼运算只需 $O(n)$ 的松弛指示矩阵流形（RIM manifold）。

## 方法详解

### 整体框架
方法可以看成一条"定义流形 → 装备黎曼结构 → 给出收缩映射 → 落地到具体问题"的链条。输入是某个待优化的指示矩阵目标函数 $H(F)$（如图割损失），输出是 RIM 流形上的（近似）最优解 $F^*$。

第一步，定义松弛集合
$$M=\{X\mid X\mathbf{1}_c=\mathbf{1}_n,\ l<X^T\mathbf{1}_n<u,\ X>0\}$$
并证明它是 $(n-1)c$ 维的嵌入子流形（Lemma 1）。第二步，给 $M$ 装上由欧氏内积限制得到的黎曼度量，从而把切空间、黎曼梯度、黎曼联络、黎曼 Hessian 全部用"减去逐列均值投影"这一个统一操作写出来（Lemma 2–4），这是复杂度从 $O(n^3)$ 掉到 $O(n)$ 的根本原因。第三步，给出三种把切向量映射回流形的收缩（Retraction）方法——Dykstra 投影（能得到测地线）、对偶梯度上升、Sinkhorn 变体——并对比它们的效率。第四步，把这套工具箱应用到 Max Cut / Ratio Cut 等图割问题，给出欧氏梯度/Hessian 闭式表达，并证明 Lipschitz 连续与收敛性。基于工具箱还实现了黎曼梯度下降（RIMRGD）、黎曼共轭梯度（RIMRCG）、黎曼信赖域（RIMRTR）三种求解器。

整个方法是"理论工具箱"而非多模块串行 pipeline，核心价值在每一层的闭式推导，因此下面用公式逐一展开关键设计。

### 关键设计

**1. 区间型松弛与 RIM 流形：用可调区间统一三种已有松弛**

针对"列和点约束过严、且对未知分布脆弱"这一痛点，本文把列和约束 $X^T\mathbf{1}_n=r$ 替换为区间约束 $l<X^T\mathbf{1}_n<u$，得到 $M=\{X\mid X\mathbf{1}_c=\mathbf{1}_n,\ l<X^T\mathbf{1}_n<u,\ X>0\}$。这个设计的巧妙在于"一个参数化框架吞掉三种旧松弛"：当 $l<0,u>n$ 时区间约束失效，$M$ 退化为单随机流形；当 $u=l=r$ 时区间收缩成点，$M$ 退化为双随机流形。于是先验的注入是**连续可调**的——知道得多就把 $(l,u)$ 收紧、甚至设 $l=u=$ 真实分布；一无所知就放宽到几乎无约束。

作者进一步证明（Lemma 1）这个集合是欧氏空间的嵌入子流形，维数 $\dim M=(n-1)c$。"是流形"这件事至关重要：它意味着可以名正言顺地调用黎曼几何工具（切空间、梯度、联络、Hessian、测地线），而不是停留在一个普通的凸约束集合上做投影。这是后面所有线性复杂度结论的地基。

**2. 由欧氏内积限制得到的黎曼工具箱：把梯度/Hessian 降到 $O(n)$**

双随机流形之所以慢，是因为它的黎曼度量（Fisher 信息度量）导致梯度里出现 $(I-FF^T)^\dagger$ 这种 $n\times n$ 伪逆，单算一次梯度就要 $O(n^3)$。本文反其道而行：**不用 Fisher 度量，直接把欧氏内积 $\langle U,V\rangle=\sum_{ij}U_{ij}V_{ij}$ 限制到流形上**。这一选择让所有黎曼量都坍缩成"原始欧氏量减去逐列均值"这种廉价操作。

具体地，切空间为 $T_XM=\{U\mid U\mathbf{1}_c=\mathbf{0}\}$（Lemma 2）；黎曼梯度只是欧氏梯度去掉行均值分量：
$$\mathrm{grad}\,H(F)=\mathrm{Grad}\,H(F)-\tfrac{1}{c}\,\mathrm{Grad}\,H(F)\mathbf{1}_c\mathbf{1}_c^T$$
黎曼联络 $\nabla_VU=\bar\nabla_VU-\tfrac{1}{c}\bar\nabla_VU\,\mathbf{1}_c\mathbf{1}_c^T$（Lemma 3，选取使 Hessian 自伴随的唯一兼容联络），黎曼 Hessian 同样是 $\mathrm{hess}\,H[V]=\mathrm{Hess}\,H[V]-\tfrac{1}{c}\mathrm{Hess}\,H[V]\mathbf{1}_c\mathbf{1}_c^T$（Lemma 4）。这些式子里只有"逐列求和、除以 $c$、复制回 $c$ 列"的操作，共 $2nc$ 次加法、$n$ 次除法，因此黎曼梯度与黎曼 Hessian 都只需 $O(n)$（见复杂度对比表），相比双随机流形的 $O(n^3)$ 直接快了 $O(n^2)$ 倍。这是全文的工程内核——理论上"换一个度量"就抹掉了立方项。

**3. 三种收缩映射：在投影、对偶与 Sinkhorn 之间权衡测地线与速度**

有了切空间还需要"收缩"把切向量 $tV$ 映回流形 $M$。本文给出三条路线。其一是 **Dykstra 投影**：注意到 $M=\Omega_1\cap\Omega_2\cap\Omega_3$（行和约束、列和下界、列和上界三个闭凸集的交），通过交替投影到三个集合求 $\arg\min_{F\in M}\|F-(X+tV)\|_F^2$。其中行和投影为 $\mathrm{Proj}_{\Omega_1}(X)=(X_{ij}+\eta_i)_+$，列和界投影是分段的（满足约束时不动，越界时平移修正，见式 7）。本文进一步证明（Theorem 1）这个正交投影收缩在 $t$ 充分小时满足 $R_X(0)=X$、$\frac{d}{dt}R_X(tV)|_{t=0}=V$ 且 $\frac{D}{dt}R_X'(tV)|_{t=0}=0$，即它给出的是**测地线**——为优化提供更好的收敛保证。

其二是**对偶梯度上升**（Theorem 3），把投影写成关于拉格朗日乘子 $\nu,\omega,\rho$ 的对偶问题 $\max_{\omega,\rho\ge0}\frac12\|\max(0,X+tV-\nu\mathbf{1}_c^T-\mathbf{1}_n\omega^T+\mathbf{1}_n\rho^T)\|_F^2-\langle\nu,\mathbf{1}_n\rangle-\langle\omega,u\rangle+\langle\rho,l\rangle$，对乘子做梯度上升后再代回 $\max(0,\cdot)$ 得到收缩点。其三是 **Sinkhorn 变体收缩**（Theorem 4），用两个对角缩放矩阵把 $X\odot\exp(tV\oslash X)$ 推到流形上，等价于求解带熵正则（系数为 1）的双边界最优传输问题——但作者坦言熵正则系数的选取缺乏充分依据。实验表明：矩阵小时 Sinkhorn 快，规模一大 **Dykstra 显著占优且产出测地线**，因此推荐大规模场景用 Dykstra。三选一的设计让用户能按规模与对测地线性质的需求取舍。

**4. 图割落地与收敛性保证：把工具箱接到 Max Cut / Ratio Cut 上**

工具箱要有用就得能接真实问题。本文以图割为例：Max Cut 损失 $H_m(F)=-\mathrm{tr}(F^TSF)$、Ratio Cut 损失 $H_r(F)=\mathrm{tr}(F^TLF(F^TF)^{-1})$，把原本 $F\in\mathrm{Ind}_{n\times c}$ 的约束松弛到 RIM 流形上求解。借助设计 2 的统一公式，Max Cut 的黎曼梯度直接是 $\mathrm{grad}\,H_m(F)=-SF+\tfrac1cSF\mathbf{1}_c\mathbf{1}_c^T$，Hessian 也一行写出。Ratio Cut 的欧氏梯度（Theorem 5）为 $2\big(LF(F^TF)^{-1}-F(F^TF)^{-1}(F^TLF)(F^TF)^{-1}\big)$；虽含 $(F^TF)^{-1}$ 求逆，但 $F^TF\in\mathbb{R}^{c\times c}$ 而 $c\ll n$，求逆只要 $O(c^3)$，不破坏整体线性复杂度。

更关键的是收敛性兜底：本文证明（Theorem 6）任意形如 $\mathrm{tr}((F^TLF)(F^TWF)^{-1})$ 的图割问题，其欧氏梯度谱范数有界 $\|\mathrm{Grad}\,H(F)\|_{\mathbb{S}}\le 2(\tfrac{\|L\|_{\mathbb{S}}\sqrt n}{\alpha}+\tfrac{\|W\|_{\mathbb{S}}\|L\|_{\mathbb{S}}n^{3/2}}{\alpha^2})$，其中 $\alpha=\sigma_{\min}(W)\cdot\tfrac{l^2}{n}$，从而 $H$ 是 Lipschitz 连续的；并进一步给出 Lipschitz 光滑性与收敛速率分析（Appendix A.11）。这让"在 RIM 流形上跑 Ratio Cut"不只是经验上快，而是有严格收敛保证。

### 损失函数 / 训练策略
本文不是训练神经网络，而是直接在 RIM 流形上最小化目标函数。基于工具箱实现了三种黎曼求解器：黎曼梯度下降（RIMRGD）、黎曼共轭梯度（RIMRCG）、黎曼信赖域（RIMRTR）。对比双随机流形方法时，作者刻意保持线搜索方法与停止准则完全一致，唯一差异就是流形工具箱本身，以隔离出"流形选择"带来的增益。

## 实验关键数据

### 主实验
**Ratio Cut 优化的时间与损失（$l=u$，节选自 Table 5）**：RIM 流形上的求解器在时间上大幅领先双随机流形方法，损失也更低。

| 数据集 | DSRGD 时间/损失 | RIMRCG 时间/损失 | RIMRGD 时间/损失 |
|--------|------|------|------|
| COIL20 | 8.978 / 28.17 | **0.685** / 27.46 | 1.145 / **24.83** |
| JAFFE | 2.224 / 30.06 | **0.119** / 29.92 | 0.149 / 29.56 |
| USPS20 | 9.238 / 25.52 | **0.735** / 19.91 | 5.257 / **16.46** |
| PalmData25 | 43.39 / 737.1 | 18.77 / 516.3 | 9.506 / **456.0** |

**凸优化（范数逼近，RTR 求解，节选 Table 3）**：RIM 流形在时间和最终损失上同时碾压双随机流形，损失低了好几个数量级。

| 规模 $n$ ($c{=}100$) | RIM Cost | RIM Time | DSM Cost | DSM Time |
|------|------|------|------|------|
| 5000 | 1.31E-19 | 0.990 | 1.78E-09 | 91.40 |
| 10000 | 1.26E-17 | 1.721 | 2.83E-09 | 241.4 |

作者称在百万级变量、$l=u$ 时，RIM 流形比双随机流形方法快 **70–200 倍**且损失更低。

### 消融实验
**三种收缩方法的执行时间对比（$l=u$，节选 Table 2，单位秒）**：随规模增大 Dykstra 优势显著。

| 配置 | $n{=}1000,c{=}1000$ | $n{=}10000,c{=}1000$ | 说明 |
|------|------|------|------|
| Sinkhorn | 0.082 | 3.614 | 小矩阵最快，大矩阵变慢 |
| Dual（对偶） | 0.178 | 56.56 | 大规模最慢 |
| Dykstra | **0.067** | **0.789** | 大规模最快，且产出测地线 |

**去噪指标（RIM vs DSM，节选 Table 4）**：在多种噪声/正则组合下 RIM 全面优于双随机。

| 指标 | RIM (0.3,0.3) | DSM (0.3,0.3) |
|------|------|------|
| PSNR↑ | **19.27** | 18.33 |
| SSIM↑ | **0.502** | 0.412 |
| LPIPS↓ | **0.561** | 0.671 |

聚类上（Table 6，ACC/NMI/ARI 三指标、8 个真实数据集、对比 10+ 算法），在 RIM 流形上做 Ratio Cut 的聚类精度整体超过最新聚类算法。

### 关键发现
- **度量选择是提速关键**：把 Fisher 度量换成欧氏内积限制，黎曼梯度/Hessian 从 $O(n^3)$ 降到 $O(n)$，这是 70–200 倍加速的根因，而非单纯实现优化。
- **收缩方法随规模换挡**：小矩阵 Sinkhorn 快，大矩阵 Dykstra 既快又给测地线，因此作者明确推荐大规模用 Dykstra。
- **更松的约束反而结果更好**：RIM 在去噪图像上比双随机更平滑、噪点更少（29.77s/损失 1.05e5 对 85.33s/损失 1.17e5），说明放宽列和约束不仅省时间，还能逃离双随机过严约束带来的劣解。

## 亮点与洞察
- **"区间代替点"是极简却统一的 idea**：仅把一个等式约束放宽成区间，就同时统一了单随机、双随机两种已有松弛，并让先验注入量连续可调——一个参数 $(l,u)$ 把"先验强弱"和"约束松紧"对齐起来，非常优雅。
- **换度量抹掉立方项**：双随机的 $O(n^3)$ 来自 Fisher 度量诱导的伪逆；本文直接限制欧氏内积，使所有黎曼运算坍缩成"减逐列均值"，这个"选对度量比选对算法更省事"的洞察可迁移到其他带行/列和约束的流形优化。
- **Dykstra 投影即测地线**：证明正交投影收缩在小步长下恰为测地线，把"廉价的交替投影"和"有收敛保证的测地线"两件好事合一，是工具箱里最漂亮的一笔。
- 思路可迁移：任何带"行和固定 + 列和区间 + 非负"约束的问题（最优传输、软分配、平衡聚类）都能直接套这套 $O(n)$ 工具箱。

## 局限与展望
- 作者承认：作为 NP-hard 指示矩阵问题的松弛，本文只能证明 RIM 流形在各方面优于已有松弛，**无法给出松弛最优解与原始离散最优解之间的精确界**。
- RIM 流形用"投影到闭集"作为收缩，当问题严格要求开集（$X>0$ 而非 $X\ge0$）时需要引入一个小修正项 $\varepsilon$，这是理论上的瑕疵。
- 自己发现的局限：实验集中在图割/去噪/聚类，对分类等其他指示矩阵任务以及非图割型非凸目标的普适性还需更多验证；$(l,u)$ 该如何根据数据自适应选取也未给出系统方案。

## 相关工作与启发
- **vs 双随机流形方法（DSRGD/DSRCG）**：他们把列和钉死在先验分布上、黎曼梯度需 $O(n^3)$ 伪逆；本文放宽成区间、用欧氏度量限制把同类运算降到 $O(n)$，在 $l=u$ 时还能直接当双随机的快速替代，实测快 70–200 倍且损失更低。
- **vs 单随机流形（Fuzzy K-means 类）**：他们不约束列和、易产生空簇/类不平衡；本文通过 $(l,u)$ 区间把类别先验喂进模型，聚类指标更优。
- **vs Stiefel 流形（谱聚类）**：他们 $O(n^2c)$ 且只对 $\mathrm{tr}(F^TLF)$ 给解析解、还需 K-means 后处理；本文直接在松弛流形上优化 Ratio Cut 并带收敛证明，避免谱分解的尺度瓶颈。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ "区间松弛 + 欧氏度量限制"统一三种已有流形并把复杂度从 $O(n^3)$ 降到 $O(n)$，是有真正理论分量的新构造。
- 实验充分度: ⭐⭐⭐⭐ 覆盖凸/非凸、去噪/图割/聚类、百万级变量与多套对比算法，但主要集中在图割系任务。
- 写作质量: ⭐⭐⭐⭐ 引理-定理链条清晰，复杂度对比与收敛证明完整；部分推导较密、对读者背景要求高。
- 价值: ⭐⭐⭐⭐⭐ 给一大类带行/列和约束的流形优化提供了线性复杂度的通用工具箱，落地价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Landing with the Score: Riemannian Optimization through Denoising](landing_with_the_score_riemannian_optimization_through_denoising.md)
- [\[ICLR 2026\] Scalable Second-Order Riemannian Optimization for K-means Clustering](scalable_second-order_riemannian_optimization_for_k-means_clustering.md)
- [\[ICLR 2026\] FedMC: Federated Manifold Calibration](fedmc_federated_manifold_calibration.md)
- [\[ICLR 2026\] Strongly Convex Sets in Riemannian Manifolds](strongly_convex_sets_in_riemannian_manifolds.md)
- [\[ICML 2026\] RMNP: Row-Momentum Normalized Preconditioning for Scalable Matrix-Based Optimization](../../ICML2026/optimization/rmnp_row-momentum_normalized_preconditioning_for_scalable_matrix-based_optimizat.md)

</div>

<!-- RELATED:END -->
