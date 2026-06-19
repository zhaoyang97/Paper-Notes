---
title: >-
  [论文解读] Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation
description: >-
  [ICML 2026][3D视觉][最优传输] 本文提出 CDOT（Convex Distance Operator Transport），通过把每个度量空间的距离矩阵和耦合一起"算子化"，用 $\|D_X T_\pi - T_\pi D_Y\|_{\mathrm{HS}}^2$ 替代 FGW 中那个非凸的成对距离差平方，从而首次得到一个**对耦合 $\pi$ 严格凸**、同时仍然是合法伪度量、并具备有限样本风险界的异构空间对齐框架。
tags:
  - "ICML 2026"
  - "3D视觉"
  - "最优传输"
  - "Gromov–Wasserstein"
  - "距离算子"
  - "凸优化"
  - "Frank–Wolfe"
---

# Convex Distance Operator Transport: A Convex and Geometry-Preserving Formulation

**会议**: ICML 2026  
**arXiv**: [2606.02047](https://arxiv.org/abs/2606.02047)  
**代码**: 暂无  
**领域**: 最优传输 / 凸优化 / 度量测度空间  
**关键词**: 最优传输, Gromov–Wasserstein, 距离算子, 凸优化, Frank–Wolfe

## 一句话总结
本文提出 CDOT（Convex Distance Operator Transport），通过把每个度量空间的距离矩阵和耦合一起"算子化"，用 $\|D_X T_\pi - T_\pi D_Y\|_{\mathrm{HS}}^2$ 替代 FGW 中那个非凸的成对距离差平方，从而首次得到一个**对耦合 $\pi$ 严格凸**、同时仍然是合法伪度量、并具备有限样本风险界的异构空间对齐框架。

## 研究背景与动机

**领域现状**：跨异构域比较概率分布的事实标准是 Gromov–Wasserstein（GW）及其加入节点特征的版本 Fused GW（FGW）：它们用 $|d_\mathcal{X}(X,X') - d_\mathcal{Y}(Y,Y')|^2$ 这样的成对距离差去衡量结构错位，再叠加特征对齐项 $\mathbb{E}_\pi[\|f_\mathcal{X}(X)-f_\mathcal{Y}(Y)\|^2]$。这套框架被用在图分类、脑连接组比对、shape matching 等需要跨域比较的任务上。

**现有痛点**：FGW/GW 的结构项里出现了 $\pi \otimes \pi$ 这种**张量积形式**，是 $\pi$ 的二次型但 Hessian 不定，因而**目标非凸**。结果就是：(i) Frank–Wolfe / 投影梯度只能保证收敛到局部稳定点；(ii) 对节点基数不同、局部几何抖动的图非常敏感——稍微多几个节点或边权扰动，逐对距离比对就崩；(iii) 几乎所有非凸 GW 变体都拿不到针对"算法实际输出的耦合"的统计一致性和有限样本界，只能证明经验风险与总体风险的距离。表 1 把 GW、FGW、Entropic GW、Sliced GW、Low-rank GW、GW-SDP、IsoRank、COPT 摆在一起，发现"在 mm 空间上同时具备伪度量、凸性、一致性、有限样本界"这一栏几乎全是 ✗。

**核心矛盾**：保留 GW 那种**逐对距离**比较，就能拿到漂亮的伪度量性质，但也直接把非凸性焊死在目标里；想凸化（如熵正则、SDP 松弛、切片），又会破坏度量性或丢掉显式传输方案。本文作者要同时拿到三件事：**凸 + 伪度量 + 显式耦合**。

**本文目标**：(1) 给出一个对 $\pi$ 严格凸、仍能感知几何结构的对齐目标；(2) 证明它在 attributed compact mm 空间上诱导一个合法伪度量；(3) 给出能在多项式时间内全局收敛的算法，并配上有限样本风险分解。

**切入角度**：把"距离"从矩阵升格成算子——定义距离算子 $(D_{\mathbb{P}_X} f)(x) = \int d_\mathcal{X}(x,x') f(x') \mathbb{P}_X(dx')$，把"耦合"升格成条件期望算子 $(T_\pi g)(x) = \int g(y) \pi(dy|x)$。在算子层面，"对齐结构"自然变成 $D_X T_\pi$ 与 $T_\pi D_Y$ 是否**交换**（intertwine），用 Hilbert–Schmidt 范数差衡量。这种"算子交换误差"是 $T_\pi$ 的线性函数 → 平方 HS 范数对 $\pi$ 二次且半正定 → 直接凸。

**核心 idea**：用"$\pi$ 看每个 $x$ 到 $Y$ 的条件平均距离"减去"$\pi$ 看每个 $y$ 到 $X$ 的条件平均距离"作为结构错位度量，**把 GW 的逐边比较升级成聚合距离剖面比较**——既消除了非凸的 $\pi\otimes\pi$ 张量积，又对节点数差异和局部噪声更鲁棒（图 1 中两个不同节点数的环图在 CDOT 下结构等价、在 FGW 下不等价）。

## 方法详解

### 整体框架

CDOT 要解决的核心问题是：怎样在不破坏凸性的前提下，衡量两个异构度量空间的结构错位。FGW 的做法是逐对地比较距离差 $|d_\mathcal{X}-d_\mathcal{Y}|^2$，几何含义清楚却把非凸的 $\pi\otimes\pi$ 焊死在目标里；CDOT 换了一个视角——先把"距离"和"耦合"都升格成算子，再让"结构对齐"变成两个算子是否交换的问题。

具体来说，输入是两个 attributed compact mm 空间 $\mathfrak{X} = (\mathcal{X}, d_\mathcal{X}, \mathbb{P}_X, f_\mathcal{X})$、$\mathfrak{Y} = (\mathcal{Y}, d_\mathcal{Y}, \mathbb{P}_Y, f_\mathcal{Y})$，要输出一个传输方案 $\pi \in \Pi(\mathbb{P}_X, \mathbb{P}_Y)$。第一步把两侧的距离写成距离算子 $D_{\mathbb{P}_X}$、$D_{\mathbb{P}_Y}$，把待求耦合写成条件期望算子 $T_\pi$；于是 CDOT 目标可以写成特征对齐项加上一个"算子交换误差"的结构项 $\mathcal{L}_\alpha(\pi) = (1-\alpha)\,\mathbb{E}_\pi[c_f(X,Y)] + \tfrac{\alpha}{2}\|D_{\mathbb{P}_X} T_\pi - T_\pi D_{\mathbb{P}_Y}\|_{\mathrm{HS}}^2$，其中 $c_f(x,y)=\|f_\mathcal{X}(x)-f_\mathcal{Y}(y)\|_2^2$。落到有限样本时，距离算子用归一化距离矩阵 $d_\mathcal{X}(X_i,X_j)/n_X$ 离散化、耦合用 $n\pi$，结构项变成 $\|D_{\hat{\mathbb{P}}_X}\pi - \pi D_{\hat{\mathbb{P}}_Y}\|_F^2$，整个问题就是传输多面体上的一个凸二次规划（QP）。求解时用 projection-free 的 Frank–Wolfe，每步只解一个线性最小化（标准 LP，可直接调 OT solver）；若下游需要硬匹配，再用 Hungarian 算法把软耦合投影到置换矩阵 $\hat P = \arg\max_{P\in\mathcal{P}_n} \mathrm{Tr}(P^\top \hat\pi)$。

### 关键设计

**1. 算子层结构正则项：用对 $\pi$ 凸的"算子交换误差"替代 FGW 非凸的逐对距离差**

FGW 结构项里 $\pi\otimes\pi$ 的双线性形式是非凸性的根源，CDOT 把它换成正则项 $\mathcal{R}(\pi) = \|D_{\mathbb{P}_X} T_\pi - T_\pi D_{\mathbb{P}_Y}\|_{\mathrm{HS}}^2$。关键观察是：通过积分变换可证 $\mathcal{R}(\pi) = \iint \Gamma_\pi(x,y)^2\,\mathbb{P}_X(dx)\mathbb{P}_Y(dy)$，其中 $\Gamma_\pi(x,y) = \mathbb{E}_\pi[d_\mathcal{X}(x,X)\mid Y=y] - \mathbb{E}_\pi[d_\mathcal{Y}(y,Y)\mid X=x]$。换句话说，CDOT 比较的不是逐对距离，而是"$x$ 在 $\mathcal{X}$ 中的条件平均距离剖面"与"$y$ 在 $\mathcal{Y}$ 中的条件平均距离剖面"之差——把"逐对距离"聚合成了"期望距离"。

这一改写之所以有效，是因为 $T_\pi$ 对 $\pi$ 是线性算子，HS 范数的平方对 $T_\pi$（进而对 $\pi$）就是二次半正定，叠加线性的特征项后整体 $\mathcal{L}_\alpha$ 严格凸（Theorem 3.4）；与此同时它仍保留几何含义——刻画的是节点与其周围结构的关系，因此目标的平方根 $d_{\mathrm{CT}}^{(\alpha)}$ 在 attributed compact mm 空间上仍构成合法伪度量（Theorem 3.5）。聚合期望相比逐对距离还天然对节点基数差异鲁棒：相似几何、不同节点数的两个空间在 CDOT 下可被识别为等价（图 1）。

**2. Dispersion gap 分解：精确量化 CDOT 相对 GW 丢掉了什么**

为了解释 GW 为什么非凸、CDOT 又凸在哪里，本文在同一坐标系下做了一个干净的拆解。定义 dispersion $\mathcal{V}(\pi) = \iint \big(\mathrm{Var}_\pi[d_\mathcal{X}(x,X)\mid Y=y] + \mathrm{Var}_\pi[d_\mathcal{Y}(y,Y)\mid X=x]\big)\,\mathbb{P}_X(dx)\mathbb{P}_Y(dy)$，它捕捉的是耦合诱导出的"距离的条件方差"。Theorem 3.7 严格证明 GW 的结构代价正好等于 CDOT 结构代价加上这个 dispersion：

$$\mathcal{R}_{\mathrm{GW},2}(\pi) - \mathcal{R}(\pi) = \mathcal{V}(\pi).$$

也就是说，GW 等于"凸结构项 + 凹 dispersion 惩罚"。这给非凸性一个几何解释：在 $(\mathcal{R},\mathcal{V})$ 平面上，GW 的等高线是斜线 $x+y=c$，会和非凸的"飞镖型"可行域相切于局部最优；CDOT 的等高线则是竖线 $x=c$，沿水平方向直奔全局最优（图 2）。这同时解释了一个经验现象——GW 偏好近乎确定性的耦合（为了压小 dispersion），而 CDOT 不惩罚 dispersion，倾向于更扩散的软耦合，所以需要后接 LAP 才能拿到硬匹配。

**3. Frank–Wolfe + glued measure：把算法实际输出的耦合接到总体最优上**

以往 GW 类方法即便能证经验风险逼近总体风险，也无法保证"算法返回的那个 $\hat\pi$ 本身"逼近总体最优；CDOT 的凸性让这件事可以分两步控制。第一步，离散 CDOT 是凸 QP，标准 FW（步长 $\gamma_t=2/(t+2)$）以 $O(1/T)$ 全局收敛，把优化误差压住。第二步，用 glued measure $\Phi_n(\hat\pi)(dx,dy) = \int Q_X(dx\mid \hat x) Q_Y(dy\mid \hat y) \hat\pi(d\hat x, d\hat y)$（$Q_X,Q_Y$ 分别是 $\mathbb{P}_X\leftrightarrow\hat{\mathbb{P}}_X$、$\mathbb{P}_Y\leftrightarrow\hat{\mathbb{P}}_Y$ 的最优耦合）把离散解抬升回总体空间，从而把统计误差用 $W_1$ 控住。两步合起来，Theorem 5.6 给出针对算法输出的有限样本界

$$\big|\mathcal{L}_\alpha(\Phi_n(\hat\pi)) - \min \mathcal{L}_\alpha\big| \le \tfrac{32\alpha n_{\min}}{T+3} + C\,\big(W_1^{d_\mathcal{X}}(\mathbb{P}_X,\hat{\mathbb{P}}_X) + W_1^{d_\mathcal{Y}}(\mathbb{P}_Y,\hat{\mathbb{P}}_Y)\big),$$

其中第一项是优化误差、第二项是统计误差。Corollary 5.7 进一步在 $n_{\min}/T_n\to 0$ 下给出几乎必然的风险一致性——这是 GW 家族此前在 mm 空间上拿不到的针对算法输出的保证。

### 损失函数 / 训练策略

经验目标为 $\hat{\mathcal{L}}_\alpha(\pi) = (1-\alpha)\langle C_f, \pi\rangle_F + \tfrac{\alpha}{2}\, n_X n_Y \|D_{\hat{\mathbb{P}}_X}\pi - \pi D_{\hat{\mathbb{P}}_Y}\|_F^2$，融合权重默认取 $\alpha=0.5$（合成与真实数据通用），迭代数 $T \in \{50,100,200\}$ 已足够把优化误差压到可忽略。每步复杂度 $\mathcal{O}(n^3)$（与 FGW 同阶），论文给出的 lazy gradient FW 变体利用二次结构把梯度做增量化更新、压低常数因子；相比 $\mathcal{O}(n^6)$ 的 GW-SDP 整整低三个量级。需要硬匹配的应用在收敛后再叠一层 Hungarian。

## 实验关键数据

### 主实验

合成 2D 聚类点云（$N=4n$，重复 100 次，报告 MSE），把 CDOT 与 FGW、Entropic FGW、IsoRank、Spectral、COPT 同台比较：

| $n$ | CDOT | FGW | EFGW | IsoRank | Spectral | COPT |
|----|------|------|------|---------|----------|------|
| 100 | **0.0077** | 0.0146 | 0.0098 | 0.0141 | 1.3447 | 0.6664 |
| 300 | **0.0027** | 0.0055 | 0.0038 | 0.0053 | 1.3276 | 0.6670 |
| 500 | **0.0016** | 0.0034 | 0.0025 | 0.0033 | 1.3373 | 0.6670 |

CDOT 在所有样本量下都是最低 MSE，且 MSE 随 $n$ 增大单调下降——直接经验验证了 Theorem 5.6 的统计一致性。Spectral / COPT 因为不用特征信息，几乎学不到正确对齐。

OASIS-3 脑连接组节点对齐准确率（100 个被试两两匹配）：

| 方法 | Diffusion 距离 | Geodesic 距离 | Topology |
|------|----------------|---------------|----------|
| CDOT | **0.6136** | 0.4640 | – |
| FGW  | 0.1853 | **0.5375** | – |
| EFGW | 0.4097 | 0.4583 | – |
| IsoRank | – | – | **0.4055** |
| Spectral / COPT | – | – | 0.0737 / 0.0253 |

CDOT 在 diffusion 距离下大幅领先（0.61 vs FGW 0.19）；geodesic 上 FGW 稍好，原因是 FGW 直接惩罚成对距离失真，更吃这种"对单条最短路敏感"的几何信息；而 diffusion 把所有路径平均化，反而拉低了 GW 需要的"逐对差异对比度"，CDOT 的算子聚合则能把这种全局信号利用起来。

图分类基准（节点数 17–40）：

| 数据集 | CDOT | FGW | GW | COPT |
|--------|------|-----|----|----- |
| MUTAG  | **0.862** | 0.825 | 0.718 | 0.633 |
| IMDB-B | **0.642** | 0.602 | – | 0.637 |
| PROTEINS | **0.755** | 0.736 | 0.661 | 0.695 |
| NCI1   | **0.748** | 0.730 | 0.571 | 0.599 |
| ENZYMES| **0.513** | 0.445 | 0.238 | 0.235 |

5/5 数据集 CDOT 取胜，相对 FGW 的提升在 1.7–6.8 个百分点之间，对 GW/COPT 差距更大。

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|----------|------|
| $\alpha$ 扫描 | $\alpha=0.5$ 附近最稳定 | 特征项与结构项需要平衡，结构项太重时会偏向纯几何 |
| 迭代数 $T=50/100/200$ | $T=200$ 已足够 | 与 Theorem 5.6 的 $O(1/T)$ 速率一致 |
| 距离归一化 | 用最大值归一化后跨数据集稳定 | 算子尺度对 HS 范数敏感 |
| 软耦合 vs LAP 后处理 | 软耦合自带扩散性 | dispersion gap 解释：CDOT 不惩罚 dispersion，需 LAP 才能拿硬匹配 |
| 距离选择（diffusion vs geodesic） | CDOT 偏好 diffusion，FGW 偏好 geodesic | 印证 CDOT 用聚合距离剖面、FGW 用逐对差异的本质区别 |

### 关键发现

- **凸性是真的有用**：在合成数据上 CDOT 系统性优于 FGW，且 EFGW（用熵正则把 GW 平滑成"近凸"）和 CDOT 接近，从反面印证非凸性是 FGW 的主要瓶颈。
- **dispersion gap 不只是理论玩具**：脑连接组里 diffusion vs geodesic 的反向胜负，恰好对应"距离方差大/小"的两种几何，CDOT 因为不惩罚 dispersion 在高方差几何上表现更好。
- **算法实际输出本身就有一致性**：脑连接组、图分类等真实数据上 CDOT 都能稳定取胜，且方差通常更小（IMDB-B 上 std 0.04 vs FGW 0.07），暗示 FW 在凸景观上比 FGW 的局部最优陷阱稳得多。

## 亮点与洞察

- **"算子化几何"是把 GW 凸化的真正钥匙**：把距离矩阵和耦合都升格成算子，"结构对齐"自然变成两个算子的交换误差，从而把双线性 $\pi\otimes\pi$ 化掉。这种"提升到算子层再用线性谱论"的思路在 kernel mean embedding、HSIC 里早有先例，但用来给 GW 凸化是新的，背后的关键是 $T_\pi$ 是 $\pi$ 的线性算子。
- **dispersion gap 是少有的"非凸性来源精确分解"**：能写出 $\mathcal{R}_{\mathrm{GW}} = \mathcal{R}_{\mathrm{CDOT}} + \mathcal{V}$（凸 + 凹），并配上 $(\mathcal{R},\mathcal{V})$ 平面的等高线几何示意（图 2），让"GW 局部最优陷阱"从经验观察升级为结构性解释，可以迁移到其他用 $\pi\otimes\pi$ 的目标（如 sliced GW、low-rank GW）做类似分析。
- **glued measure 是连接"离散解"与"总体最优"的标准动作**：用 $\Phi_n$ 把 $\hat\pi$ 升回总体空间，再用 $W_1$ 控统计误差——这个 trick 在其他对算法输出做一致性证明的场景（如 entropic OT 估计、neural OT）都能复用。

## 局限与展望

- 作者承认：(i) 零判别性还差一步——$d_{\mathrm{CT}}^{(\alpha)} = 0$ 不蕴含 mm 空间等同，只蕴含一种"分数式结构等价"，给出完整的零集刻画留作未来工作；(ii) 统计率在 $\mathbb{R}^d$ 上是 $O(n^{-1/d})$，在无穷维设定下不带额外结构假设没有多项式率。
- 自己发现：(i) 实际跑出来的耦合扩散性强，几乎所有需要硬匹配的应用都得叠 Hungarian，软硬之间的近似缺口没有系统量化；(ii) 单步 $\mathcal{O}(n^3)$ 仍然是 $n$ 上千的图任务的实际瓶颈，lazy FW 只是常数加速，没有给 sliced/low-rank 类的次三次方变体；(iii) Assumption 5.5（最优耦合的条件分布在 $W_1$ 下 Lipschitz）相当强，在离散图任务上是否成立需逐例验证。
- 改进思路：把 dispersion 拆解作为"诊断器"指导设计带有可控凹项的"半凸 CDOT"，在保留 CDOT 全局最优的同时部分恢复 GW 的硬匹配偏好；与 entropic regularization 或 sliced 思路结合做 $\mathcal{O}(n^2)$ 近似；把算子层框架拓展到非欧度量（树度量、Wasserstein-on-Wasserstein）。

## 相关工作与启发

- **vs FGW (Vayer et al., 2020)**：FGW 用 $\mathbb{E}_{\pi\otimes\pi}|d_\mathcal{X}-d_\mathcal{Y}|^2$ 做逐对距离对齐，目标对 $\pi$ 非凸；CDOT 用算子交换误差对齐，目标对 $\pi$ 凸；同样 $\mathcal{O}(n^3)$ 复杂度，但 CDOT 多了伪度量 + 一致性 + 有限样本界这三件大礼包。
- **vs Entropic GW (Peyré et al., 2016)**：熵正则把 GW 平滑成"近凸"，但破坏了原始目标的伪度量性、依赖 $\varepsilon$ 调参；CDOT 在**原始未正则**形式就是凸的，不需要熵桥接。
- **vs GW-SDP (Chen et al., 2024)**：通过 SDP 松弛拿到凸目标，但代价是 $\mathcal{O}(n^6)$ 计算复杂度且不保有伪度量；CDOT 把复杂度压回 $\mathcal{O}(n^3)$ 且仍是伪度量。
- **vs Sliced GW / Sliced FGW**：切片方法用一维投影避开非凸性，但要么不输出显式耦合（只能算 discrepancy），要么牺牲度量性；CDOT 同时保留了"显式传输方案 + 度量性 + 凸性"三件套。
- **启发**：把"几何信息提升到算子，再在 RKHS / $L^2$ 算子层操作"的范式非常适合移植到 graph kernel、scenegraph alignment、跨模态 OT 等需要"既要凸又要几何感知"的问题；dispersion gap 的拆解可作为未来"非凸 OT 目标凸化"研究的通用工具。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个真正凸的几何感知 OT 目标，dispersion gap 给出 GW 非凸性的精确分解，思路新颖且解释力强。
- 实验充分度: ⭐⭐⭐⭐ 合成 + 脑连接组 + 5 个图分类数据集覆盖了三类典型异构对齐场景，但缺更大规模图（>1k 节点）和 3D shape 等高频应用。
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、表 1 对比信息密度极高，凸性 / 伪度量 / dispersion gap / 风险界一脉相承。
- 价值: ⭐⭐⭐⭐ 给 GW 家族补上了"凸 + 一致性 + 有限样本界"的关键空缺，理论价值显著；实践上能直接替换 FGW 用在图对齐 / 脑连接组分析上。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Learning Convex Decomposition via Feature Fields](../../CVPR2026/3d_vision/learning_convex_decomposition_via_feature_fields.md)
- [\[ICML 2026\] AvAtar: Learning to Align via Active Optimal Transport](avatar_learning_to_align_via_active_optimal_transport.md)
- [\[ICML 2026\] Streaming Sliced Optimal Transport](streaming_sliced_optimal_transport.md)
- [\[ECCV 2024\] Differentiable Convex Polyhedra Optimization from Multi-view Images](../../ECCV2024/3d_vision/differentiable_convex_polyhedra_optimization_from_multi-view_images.md)
- [\[CVPR 2025\] 3D Convex Splatting: Radiance Field Rendering with 3D Smooth Convexes](../../CVPR2025/3d_vision/3d_convex_splatting_radiance_field_rendering_with_3d_smooth_convexes.md)

</div>

<!-- RELATED:END -->
