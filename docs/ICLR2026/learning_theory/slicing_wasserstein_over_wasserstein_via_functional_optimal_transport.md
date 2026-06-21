---
title: >-
  [论文解读] Slicing Wasserstein over Wasserstein via Functional Optimal Transport
description: >-
  [ICLR 2026][最优传输][Wasserstein over Wasserstein] 本文提出**双重切片 Wasserstein（DSW）**距离，用「球面域切片 + 分位数函数的 $L^2$ 高斯过程切片」两层切片来高效逼近代价高昂的 Wasserstein over Wasserstein（WoW）距离，并证明在离散元测度上 DSW 的最小化与 WoW 的最小化等价，避开了已有切片方法对高阶矩的数值不稳定依赖，在数据集、形状、图像比较上都能作为 WoW 的可扩展替代。
tags:
  - "ICLR 2026"
  - "最优传输"
  - "学习理论"
  - "Wasserstein over Wasserstein"
  - "切片 Wasserstein"
  - "元测度"
  - "高斯过程"
---

# Slicing Wasserstein over Wasserstein via Functional Optimal Transport

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=l3KtyVZde3](https://openreview.net/forum?id=l3KtyVZde3)  
**代码**: https://github.com/MoePien/slicing_wasserstein_over_wasserstein  
**领域**: 最优传输 / 学习理论  
**关键词**: 最优传输, Wasserstein over Wasserstein, 切片 Wasserstein, 元测度, 高斯过程

## 一句话总结
本文提出**双重切片 Wasserstein（DSW）**距离，用「球面域切片 + 分位数函数的 $L^2$ 高斯过程切片」两层切片来高效逼近代价高昂的 Wasserstein over Wasserstein（WoW）距离，并证明在离散元测度上 DSW 的最小化与 WoW 的最小化等价，避开了已有切片方法对高阶矩的数值不稳定依赖，在数据集、形状、图像比较上都能作为 WoW 的可扩展替代。

## 研究背景与动机

**领域现状**：最优传输（OT）给出的 Wasserstein 距离能在任意 Polish 空间上度量概率测度之间的几何差异。由于 $\mathcal{P}_2(X)$ 本身又是一个完备可分度量空间，可以在「测度的测度」（meta-measure，元测度）上再定义一层 Wasserstein 距离，得到 **Wasserstein over Wasserstein（WoW）** 距离 $\mathcal{W}(\cdot,\cdot;\mathcal{P}_2(X))$。它在比较图像分布、点云分布、带标签数据集（OTDD）等「非欧几里得对象的分布」时特别自然——例如两张图像直接用欧氏距离往往失效，而把图像表示成 patch 分布、再用 WoW 比较则对小扰动稳健。

**现有痛点**：WoW 极其昂贵。若元测度支撑在 $N$ 个测度上、每个测度又有 $n$ 个支撑点，仅计算所需的距离矩阵就已是 $O(N^2 n^2 \log n)$ 复杂度，因为每次外层传输都要反复求解内层 Wasserstein。为了加速，已有切片（sliced）方法要么假设元测度是高斯混合等**参数化**形式，要么走 s-OTDD 的**矩方法**路线——后者只在有限阶矩存在时才良定义，且实际实现因数值不稳定只能用前几阶矩（原始实现仅到第 5 阶），高维下精度受限。

**核心矛盾**：切片 Wasserstein 的威力来自「把高维传输降成可解析计算的 1d 传输」，但元测度的「底空间」本身是无穷维的 Wasserstein 空间 $\mathcal{P}_2(\mathbb{R})$，既没有 Banach 空间结构、也没有合适的分位数映射可直接套用经典切片；而绕道用矩展开来描述这个无穷维对象，又会撞上高阶矩的数值不稳定。

**本文目标**：构造一个对一般元测度都良定义、可计算、且数值稳定的切片距离，使其能作为 WoW 的「即插即用」替代，并在理论上保证替代的合理性（最小化等价）。

**切入角度**：作者抓住一个经典而干净的等距事实——**1d Wasserstein 空间 $(\mathcal{P}_2(\mathbb{R}),\mathcal{W})$ 与分位数函数在 $L^2([0,1])$ 中的像是等距同构的**。于是「1d 测度之间的 Wasserstein 距离」就变成「$L^2$ 函数之间的欧氏距离」，从而可以在函数空间里做线性投影来切片，彻底回避矩展开。

**核心 idea**：先把元测度的底域用经典球面切片降到 1d，再借助分位数等距把 1d 元测度搬进函数空间 $L^2([0,1])$、用高斯过程参数化的 $L^2$ 投影做第二层切片——「双重切片」既保留了切片的高效，又用函数投影代替了不稳定的高阶矩。

## 方法详解

### 整体框架

设输入是两个经验元测度 $\mu,\nu\in\mathcal{P}_e(\mathcal{P}_e(\mathbb{R}^d))$，即「分布的分布」：$\mu=\frac{1}{N}\sum_{i=1}^N\delta_{\mu_i}$，其中每个 $\mu_i=\frac{1}{n_i}\sum_k\delta_{x_{i,k}}$ 是 $\mathbb{R}^d$ 上的经验测度。目标是高效算出一个能替代 WoW 的距离。

整条计算管线是两层切片的串联：

1. **第一层——域切片（外层球面切片）**。在欧氏底域 $\mathbb{R}^d$ 上取方向 $\theta\in S^{d-1}$，用线性投影 $\pi_\theta(x)=\langle\theta,x\rangle$ 把每个底层测度 $\mu_i$ 推到实轴上，得到一个 1d 元测度 $\pi_{\theta,\sharp}\mu\in\mathcal{P}_2(\mathcal{P}_2(\mathbb{R}))$。这一步把「$\mathbb{R}^d$ 上分布的分布」降成「$\mathbb{R}$ 上分布的分布」。

2. **第二层——分位数切片（内层函数切片）**。对 1d 元测度，利用分位数等距 $q:\mathcal{P}_2(\mathbb{R})\to L^2([0,1]),\ \mu\mapsto Q_\mu$ 把每个 1d 底层测度搬成 $L^2([0,1])$ 里的一个分位数函数；于是 1d 元测度变成「$L^2$ 函数上的分布」。再用高斯过程采样得到的方向 $g\in L^2([0,1])$ 做内积投影 $\pi_g$，把这些函数投到实轴，得到普通的 1d 测度，最后算解析的 1d Wasserstein 距离。这一层叫 **SQW（sliced quantile WoW）**。

3. **合成与蒙特卡洛估计**。把上述两层投影方向（球面方向 $\theta$ 与高斯过程样本路径 $g$）同时采样、用蒙特卡洛求平均，得到 **double-sliced Wasserstein（DSW）** 距离。整条管线只需排序求分位数 + 一维积分（数值积分求内积）+ 一维 Wasserstein，全部解析或近解析，避免了任何高阶矩。

这套管线的理论骨架是先在「任意 Banach 空间」上把切片 Wasserstein 推广出来（关键设计 1），SQW 与 DSW 都是它在 $L^2([0,1])$ 与元测度上的特例；最后用一条等价定理（关键设计 4）保证「最小化 DSW」与「最小化 WoW」在离散元测度上一致，从而 DSW 是合理的替代而非随意的近似。

### 关键设计

**1. Banach 空间上的 ξ-切片 Wasserstein：用参考测度代替不存在的无穷维球面均匀分布**

经典切片 Wasserstein 在 $\mathbb{R}^d$ 上对方向 $\theta$ 按球面 $S^{d-1}$ 上的均匀分布积分。但要把切片搬到无穷维函数空间，立刻遇到障碍：**无穷维球面上不存在均匀概率测度**。作者的解法是放弃「在球面上找特定测度」，转而在对偶空间 $U^*$ 上任取一个参考测度 $\xi\in\mathcal{P}_2(U^*)$，定义投影 $\pi_v(u)=\langle v,u\rangle=v(u)$，并给出 $\xi$-切片 Wasserstein 距离

$$\mathrm{SW}(\mu,\nu;\xi):=\Big(\int_{U^*}\mathcal{W}^2(\pi_{v,\sharp}\mu,\pi_{v,\sharp}\nu;\mathbb{R})\,d\xi(v)\Big)^{1/2}.$$

只要 $\xi$ 的支撑覆盖所有方向（定理 3.1 给出充分条件 $\mathrm{supp}\,\xi\cap\mathrm{span}\,v\notin\{\emptyset,\{0\}\}$），它就是一个真正的度量而非仅伪度量。这一抽象层是全文的地基：它让我们可以**只用「容易采样的参考测度」就在任意可分 Banach 空间上做切片**，而不必构造球面上的特定分布。作者还证明两点稳健性——若 $\xi_1,\xi_2$ 互相绝对连续且 Radon–Nikodym 导数有界，则两者诱导的切片距离度量等价（命题 3.2）；在有限维欧氏情形，取 $\xi$ 等价于标准高斯时，$\xi$-切片距离与经典切片 Wasserstein 强等价（命题 3.3）。

**2. SQW：用 1d Wasserstein 等距 + 高斯过程把 1d 元测度搬进 $L^2$ 再切片**

要切「1d 测度的分布」这个无穷维对象，本文的杠杆是分位数等距：对 $\mu\in\mathcal{P}_2(\mathbb{R})$，其分位数函数 $Q_\mu(s)=\inf\{x\in\mathbb{R}\mid\mu((-\infty,x])\ge s\}$，使得

$$\mathcal{W}(\mu,\nu;\mathbb{R})=\Big(\int_0^1 |Q_\mu(s)-Q_\nu(s)|^2\,ds\Big)^{1/2},$$

即映射 $q:\mu\mapsto Q_\mu$ 是 $\mathcal{P}_2(\mathbb{R})\to L^2([0,1])$ 的**等距嵌入**。于是把元测度 $\mu\in\mathcal{P}_2(\mathcal{P}_2(\mathbb{R}))$ 推到 $q_\sharp\mu\in\mathcal{P}_2(L^2([0,1]))$，WoW 距离就完全等于 $L^2$ 上元测度间的 Wasserstein 距离。把设计 1 套到 $U=L^2([0,1])$，固定参考测度 $\xi$，就得到 **sliced quantile WoW**

$$\mathrm{SQW}(\mu,\nu;\xi):=\mathrm{SW}(q_\sharp\mu,q_\sharp\nu;\xi).$$

为了让参考测度「易采样且全支撑」，作者选**高斯测度**，并利用 $L^2([0,1])$ 上高斯测度与高斯过程的一一对应，取协方差核 $k_\sigma(t,s)=\exp(-|t-s|^2/2\sigma^2)$ 对应的高斯过程 $G$。该核光滑保证样本路径几乎处处光滑、核的普适性（universal）保证对应高斯测度全支撑，从而由推论 3.3.1，SQW 是 $\mathcal{P}_2(\mathcal{P}_2(\mathbb{R}))$ 上的度量。这正是对 s-OTDD 矩方法的替代：**分位数函数等距是「无损」描述 1d 测度的方式，不像截断到前几阶矩那样既有信息损失又数值不稳定**。

**3. DSW：球面域切片 + 分位数切片的双重切片，把多维元测度降到可解析的 1d**

SQW 只能处理底域为 $\mathbb{R}$ 的 1d 元测度；而实际数据（图像 patch、点云）的底域是 $\mathbb{R}^d$，多维 Wasserstein 空间 $\mathcal{P}_2(\mathbb{R}^d)$ 既没有 Banach 结构、也没有合适的分位数映射，无法直接切。作者的做法是**先切底域**：用 $\pi_\theta:\mathcal{P}_2(\mathbb{R}^d)\to\mathcal{P}_2(\mathbb{R}),\ \mu\mapsto\pi_{\theta,\sharp}\mu$ 把多维元测度降成 1d 元测度，再对降维后的对象套 SQW，按球面均匀分布对 $\theta$ 积分，得到 **double-sliced WoW**

$$\mathrm{DSW}(\mu,\nu;\xi):=\Big(\int_{S^{d-1}}\mathrm{SQW}^2(\pi_{\theta,\sharp}\mu,\pi_{\theta,\sharp}\nu;\xi)\,dS^{d-1}(\theta)\Big)^{1/2}.$$

数值上，外层球面积分与内层 $\xi$ 积分**同时**用蒙特卡洛逼近：采样方向 $\theta_s$ 与高斯过程样本路径 $g_s$，分位数函数由排序支撑点得到分段常值表示，函数内积用带权 $w_r$ 的求积（trapezoidal）近似 $\widehat{\langle q(\pi_{\theta,\sharp}\mu_i),g\rangle}=\sum_{r=1}^R w_r\,q(\pi_{\theta,\sharp}\mu_i)(t_r)\,g(t_r)$，最终
$$\widehat{\mathrm{DSW}}(\mu,\nu):=\Big(\tfrac{1}{S}\sum_{s=1}^S \mathcal{W}^2(\widehat{\pi_{g_s,\sharp}q_\sharp\pi_{\theta_s,\sharp}\mu},\,\widehat{\pi_{g_s,\sharp}q_\sharp\pi_{\theta_s,\sharp}\nu};\mathbb{R})\Big)^{1/2}.$$
全程只有排序、一维求积、一维 Wasserstein，因此远比逐对求 WoW 便宜。

**4. WoW 等价定理：保证 DSW 不是随意近似，而是最小化意义下的合法替代**

切片距离的价值取决于「它和原距离是否同进退」。作者证明（定理 4.1）：对正的高斯参考 $\xi\in\mathcal{P}_2(L^2([0,1]))$，DSW 在离散经验元测度集 $\mathcal{P}_e(\mathcal{P}_e(\mathbb{R}^d))$ 上是一个**度量**；而且对固定大小 $N$、$n_i\equiv\tilde n$ 的经验元测度，紧支撑底域 $X\subset\mathbb{R}^d$ 上有收敛等价

$$\mathrm{DSW}(\mu_n,\mu;\xi)\to 0\iff \mathrm{SW}(\mu_n,\mu;\xi)\to 0\iff \mathcal{W}(\mu_n,\mu;\mathbb{R}^d)\to 0\quad(n\to\infty).$$

证明思路是构造一个传输计划利用 WoW 本身的度量性质，再借高斯过程的离散化与 $\mathcal{P}_e^N(\mathcal{P}_e^{\tilde n}(X))$ 的紧性把三者串起来。这条「拓扑度量等价」正是 DSW 作为 WoW 替代的合法性凭证：**最小化 DSW 等价于最小化 WoW**，因此凡是把 WoW 当训练损失或检索度量的下游任务，都可以直接换成便宜得多的 DSW。

### 损失函数 / 训练策略
本文不是训练某个网络，而是提出一族距离度量；无专门训练目标。关键超参为：球面投影数 $S$、$L^2$ 求积网格点数 $R$、高斯核带宽 $\sigma$。实验显示在 MNIST-2000 上对这三者都不敏感（精度只在小范围波动），增大 $S$ 主要降低方差、把与 WoW 的相关性从 0.99 推到 1.0。

## 实验关键数据

### 主实验

**形状分类（KNN，越高越好）**：把形状建模成度量测度空间、用局部距离分布表示成 $\mathcal{P}_e(\mathcal{P}_e(\mathbb{R}))$，比较 SQW 与 TLB（即精确 1d WoW）、STLB、Anchor Energy（AE）、Gromov–Wasserstein（GW）。

| 距离 | 2D shapes Acc(%) | Animals Acc(%) | FAUST-1000 Acc(%) | MNIST-2000 Acc(%) | FAUST-1000 时间(ms) |
|--------|--------|--------|--------|--------|--------|
| SQW (本文) | 99.5±1.2 | 99.1±1.3 | 42.7±5.9 | 84.8±4.7 | 13.8±15.1 |
| TLB(精确1d WoW) | 100.0±0.3 | 100.0±0.0 | 40.2±6.0 | 88.7±4.5 | 60.1±9.6 |
| STLB | 99.5±1.2 | 99.3±1.8 | 39.4±5.6 | 84.1±5.0 | 14.0±14.9 |
| AE | 99.7±0.9 | 97.8±1.8 | 41.8±5.3 | 88.1±4.5 | 25.2±12.0 |
| GW | 99.7±0.6 | 100.0±0.0 | 33.0±5.3 | —超时— | 1048.2±357.3 |

SQW 在保持与精确 WoW（TLB）相当精度的同时，在大规模 FAUST-1000 / MNIST-2000 上有明显的运行时优势，尤其相对 GW（在 MNIST-2000 上直接超时）。

**与 OTDD / s-OTDD 的相关性**：把带标签数据集表示成元测度，以「精确 OTDD」为基准，比较 DSW 与 s-OTDD 跟 OTDD 的相关系数（100 次数据划分，MNIST / FashionMNIST / CIFAR-10）。结论是 **DSW 与 OTDD 的相关性在三个数据集上都强于 s-OTDD**，说明分位数切片比矩切片更忠实地复刻了 OTDD。

### 消融与分析实验

| 分析 | 关键设置 | 发现 |
|------|---------|------|
| 参数敏感性 | 变 $S\in\{10^2,10^3\}$、$R\in\{10,10^2\}$、$\sigma$ | MNIST-2000 精度仅在 82.6%–85.0% 间微动，对三参数都鲁棒 |
| SQW vs 精确 WoW(TLB) | MNIST-2000 散点 + 相关 | Pearson/Spearman ≥0.99；增大 $S$ 把相关性推到 1.0，$\sigma$ 几乎不影响线性关系 |
| 点云生成评测 | ModelNet-10，对比 OT-NNA / WoW / DSW | DSW 与 WoW 行为一致（都能抓 mode collapse、随噪声单调上升），且 $M\!\ge\!N$ 时不像 OT-NNA 出现反常上升；$M=N=10,\,m=n=500$ 时 DSW≈0.25s，WoW≈4.5s，OT-NNA≈8.5s |
| 图像 patch 分布 | 64×64 Perlin 纹理，patch=8 | patch-based DSW 与欧氏 Wasserstein 都在「真参数」处取最小，但 DSW 对 lacunarity/persistence 变化更敏感、判别力更强；patch-WoW≈40s 而 DSW≈1s |

### 关键发现
- **数值稳定性优势**：相比 s-OTDD 受限于前几阶矩，DSW 用分位数函数等距「无损」描述 1d 测度，因此对 $S,R,\sigma$ 都不敏感、且与 OTDD 相关性更高。
- **代价大幅下降**：在点云、图像 patch 这类高分辨率/大批量场景，DSW 比 WoW 与 OT-NNA 快一个数量级以上（0.25s vs 4.5s/8.5s；1s vs 40s），且仍是「无界的真度量」，能区分 OT-NNA 难以区分的差异。
- **行为一致**：DSW 在 mode collapse、噪声、分辨率三类扰动下都与 WoW 同向变化，印证了等价定理的实际意义。

## 亮点与洞察
- **用「等距搬运」代替「矩展开」**：分位数映射 $q$ 把 1d Wasserstein 变成 $L^2$ 欧氏距离这一经典事实，被本文用作切无穷维元测度的杠杆——这是最让人「啊哈」的一步，既无损又把数值不稳定的高阶矩彻底踢出管线。
- **「任意 Banach 空间切片」是可复用的抽象**：用「任取参考测度 $\xi$」绕开「无穷维球面无均匀分布」的障碍，这套 $\xi$-切片框架本身可迁移到任何无穷维生成模型、函数型数据的切片传输上，不限于本文的元测度。
- **双重切片的解耦**：把「多维」交给球面切片、把「测度的测度」交给分位数 + 高斯过程切片，两层各司其职，使最终计算落到纯一维 Wasserstein，工程上极易实现（排序 + 求积）。
- **理论与实用闭环**：等价定理把「便宜的近似」升级成「最小化意义下的合法替代」，让 DSW 能放心地替换任何以 WoW/OTDD 为损失或度量的下游任务。

## 局限与展望
- **等价性限定在离散经验元测度**：定理 4.1 的度量性与收敛等价是针对 $\mathcal{P}_e(\mathcal{P}_e(\mathbb{R}^d))$（且 $N,\tilde n$ 固定、紧支撑底域）证明的，连续元测度上的拓扑性质仍待分析。
- **尚未对齐完整 OTDD**：当前为可比性把标签度量设为零，未真正利用标签空间 $Y$；作者提出未来可用 hybrid slicing 把 DSW 扩到 $\mathcal{P}_2(Y\times\mathcal{P}_2(\mathbb{R}^d))$，或像 s-OTDD 那样引入卷积投影。
- **样本复杂度未知**：切片 Wasserstein 在高维通常比 Wasserstein 需要更少样本，而 WoW 逼近元测度需要大量样本；DSW 是否继承更好的样本复杂度仍是开放问题。
- **高斯过程核的选择**：实现固定用 RBF 核，核族/带宽对不同模态数据的影响只做了有限敏感性分析，更系统的核设计未探讨。

## 相关工作与启发
- **vs s-OTDD（Nguyen et al., 2025）**：同样走分层切片，但 s-OTDD 用**矩方法**把 1d 元测度投影到前几阶矩，只在有限阶矩存在时良定义、且数值不稳定限制到前 5 阶；本文用**分位数函数等距 + 高斯过程 $L^2$ 投影**无损描述 1d 测度，避开高阶矩、与 OTDD 相关性更强。
- **vs SWBDG / SWB1DG（Bonet et al., 2025c，并行工作）**：它们用 Busemann 函数的水平集作为元测度空间中仿射超平面的推广来直接切 $\mathcal{P}_2(\mathcal{P}_2(\mathbb{R}^d))$，依赖高斯近似与 Busemann 函数闭式；本文是「域切片 + 函数切片」的双重切片路线，不依赖高斯近似。
- **vs 精确 WoW / OTDD（Bonet et al., 2025b；Alvarez-Melis & Fusi, 2020）**：WoW/OTDD 是本文要替代的「真值」，但 $O(N^2 n^2\log n)$ 级代价难以扩展；DSW 在保持行为一致的前提下把计算降到一维。
- **vs Han (2023) 的 Hilbert 空间切片**：本文在其思想上推广到一般 Banach 空间，且**不构造球面上的特定测度**，而是允许任取易采样的参考测度，工程上更友好。
- **可迁移启发**：「等距嵌入 + 函数空间切片」的范式可用于任何能等距搬进 $L^2$ 的对象（如 1d 分布族、函数型数据），把昂贵的传输换成函数内积投影。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用分位数等距 + Banach 空间切片绕开高阶矩，是对 sliced WoW 的干净且通用的新框架。
- 实验充分度: ⭐⭐⭐⭐ 覆盖形状、数据集、点云、图像四类任务并给出相关性/敏感性分析，但多为相关性与代价对比，缺端到端下游训练。
- 写作质量: ⭐⭐⭐⭐⭐ 理论层层递进（Banach → SQW → DSW → 等价定理），定义与定理交代清晰。
- 价值: ⭐⭐⭐⭐⭐ 为任何以 WoW/OTDD 为度量的任务提供了快一个数量级、数值稳定且有等价保证的替代。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem](revisiting_tree-sliced_wasserstein_distance_through_the_lens_of_the_fermatweber_.md)
- [\[ICLR 2026\] Test-Time Verification via Optimal Transport: Coverage, ROC, & Sub-Optimality](test-time_verification_via_optimal_transport_coverage_roc_sub-optimality.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)
- [\[ICLR 2026\] A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers](a_statistical_learning_perspective_on_semi-dual_adversarial_neural_optimal_trans.md)
- [\[ICLR 2026\] Curse of Slicing: Why Sliced Mutual Information is a Deceptive Measure of Statistical Dependence](curse_of_slicing_why_sliced_mutual_information_is_a_deceptive_measure_of_statist.md)

</div>

<!-- RELATED:END -->
