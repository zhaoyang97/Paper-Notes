---
title: >-
  [论文解读] Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference
description: >-
  [ICLR2026][优化/理论][黎曼优化] 本文提出 RAdaGD——一族无需线搜索的黎曼流形自适应梯度下降方法，通过在线估计局部光滑常数自动调步长，在"局部测地光滑 + 广义测地凸"的弱假设下取得非遍历收敛率 $f(x_k)-f(x^\star)\le O(1/k)$，并据此给出高斯变分推断在目标对数密度**不满足全局 L-光滑**时的首个收敛保证。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "黎曼优化"
  - "自适应步长"
  - "收敛率"
  - "高斯变分推断"
  - "Bures-Wasserstein 几何"
---

# Adaptive gradient descent on Riemannian manifolds and its applications to Gaussian variational inference

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=2TTQjRkgFn](https://openreview.net/forum?id=2TTQjRkgFn)  
**代码**: https://github.com/wldyddl5510/RAdaGD  
**领域**: optimization  
**关键词**: 黎曼优化, 自适应步长, 收敛率, 高斯变分推断, Bures-Wasserstein 几何

## 一句话总结
本文提出 RAdaGD——一族无需线搜索的黎曼流形自适应梯度下降方法，通过在线估计局部光滑常数自动调步长，在"局部测地光滑 + 广义测地凸"的弱假设下取得非遍历收敛率 $f(x_k)-f(x^\star)\le O(1/k)$，并据此给出高斯变分推断在目标对数密度**不满足全局 L-光滑**时的首个收敛保证。

## 研究背景与动机
**领域现状**：黎曼优化是把欧氏空间的优化推广到弯曲流形（如正定矩阵流形、双曲空间、高斯测度空间）的框架，在机器学习、计算机视觉、统计推断中应用广泛。求解 $\min_{x\in N} f(x)$ 的标准方法是黎曼梯度下降（RGD）：$x_{k+1}=\exp_{x_k}(-s_k\,\mathrm{Grad}\,f(x_k))$，其中 $\exp$ 是指数映射、$\mathrm{Grad}$ 是黎曼梯度。当 $f$ 是测地 $L$-光滑且测地凸时，RGD 取固定步长 $s_k=1/L$ 即可达到 $O(1/k)$ 收敛率。

**现有痛点**：固定步长 $1/L$ 的 RGD 有两个硬伤——（i）"测地 $L$-光滑 + 测地凸"本身就是很强的假设，很多自然函数（比如非欧 Hadamard 流形上的平方距离函数 $x\mapsto d^2(x,p)$）压根不满足全局测地 $L$-光滑；（ii）即便满足，也需要事先知道光滑常数 $L$，而 $L$ 在实际问题里往往未知或难估。

**核心矛盾**：欧氏空间里近年已经涌现了一批"自适应"算法（Malitsky–Mishchenko、Suh–Ma 等），能不靠线搜索、不需要预知 $L$ 就自动调步长。但把这套自适应机制搬到黎曼流形上，会被流形**曲率**这一额外维度卡住：曲率会让"co-coercivity"（梯度共强制性）等关键不等式多出一个与曲率相关的因子 $\zeta$，欧氏分析无法直接平移。因此黎曼版自适应方法此前基本是空白。

**本文目标**：（1）造一族无需线搜索、不需预知 $L$ 的黎曼自适应梯度下降；（2）把它放在比 $L$-光滑更宽松的假设下分析；（3）落到一个有现实意义的应用——高斯变分推断（GVI），给出在目标非 $L$-光滑时的收敛保证。

**切入角度**：作者注意到 Suh–Ma (2025) 的欧氏自适应方法核心是"用相邻两步估出局部光滑常数 $L_{k+1}$、再据此调步长"。这套思路本质只依赖**局部**信息，因此天然适合放宽到"局部测地光滑"——一个比全局 $L$-光滑弱得多、且**所有 $C^2$ 函数都满足**的条件。难点在于把曲率因子 $\zeta$ 干净地塞进 Lyapunov 分析里。

**核心 idea**：用一个在线估计的局部光滑常数 $L_{k+1}$ 替代固定的 $1/L$ 步长，再引入第三个辅助序列 $\tilde B_k$ 专门吸收曲率项 $\zeta_k$，从而把欧氏自适应方法的 $O(1/k)$ 保证迁移到黎曼流形，并据此攻下 GVI 的非光滑情形。

## 方法详解

### 整体框架
RAdaGD（Riemannian Adaptive Gradient Descent）整体上仍是 RGD 的迭代骨架 $x_{k+1}=\exp_{x_k}(-s_k\,\mathrm{Grad}\,f(x_k))$，唯一也是全部的创新都在"步长 $s_k$ 怎么定"上。它不做线搜索（线搜索在流形上要反复算指数映射，很贵），而是每步用刚走完的这一步的信息**反推**出局部光滑常数的一个代理 $L_{k+1}$，再通过三个标量序列 $A_k,B_k,\tilde B_k$ 把 $L_{k+1}$、上一步步长 $s_k$ 和曲率因子 $\zeta_k$ 揉成下一步步长 $s_{k+1}$。

输入是初始点 $x_0$、初始步长 $s_0$ 和三个序列 $\{A_k\},\{B_k\},\{\tilde B_k\}$；输出是收敛到极小点的迭代序列 $\{x_k\}$。整篇论文的逻辑链条是：先把"放宽的假设"（局部测地光滑 + 广义测地凸）讲清并证明它确实更宽（定理 3.3：所有 $C^2$ 函数都局部测地光滑）→ 给出自适应步长规则（算法 1）→ 用 Lyapunov 函数证 $O(1/k)$ 收敛（定理 4.1、4.5）→ 分曲率已知/未知/无界三种情形给配套推论 → 落到 Bures-Wasserstein 流形上的 GVI（推论 5.4，BWAdaGVI）。这是一篇以"算法 + 收敛证明"为主体的理论论文，不存在多模块 pipeline，故不配框架图。

### 关键设计

**1. 弱化假设：局部测地光滑 + 广义测地凸，替换全局 L-光滑**

传统黎曼分析要求 $f$ 全局测地 $L$-光滑，即梯度沿测地线平行移动后的差被 $L\cdot d(x,y)$ 一致控制。本文把"一致"换成"在每个紧集上各有一个常数"：$f$ 称为**局部测地光滑**，若对每个紧集 $K\subseteq M$ 存在 $L_K$ 使 $\|\Gamma_x^y\,\mathrm{Grad}\,f(x)-\mathrm{Grad}\,f(y)\|_y\le L_K\,d(x,y),\ \forall x,y\in K$（$\Gamma_x^y$ 是平行移动）。这个放宽的价值由定理 3.3 撑起：**完备黎曼流形上任意 $C^2$ 函数都局部测地光滑**——这在欧氏空间几乎显然，但作者指出黎曼版此前没有正式陈述，而它正是后续所有应用的"入场券"。前面提到的平方距离函数 $d^2(x,p)$ 在非欧 Hadamard 流形上不是全局 $L$-光滑，却因为是 $C^2$ 而局部测地光滑，正好落进新假设里。

凸性一侧用的是**广义测地凸**（沿广义测地线的凸性，源自 Wasserstein 空间）：$f(y)\ge f(x)+\langle\Gamma_x^z\,\mathrm{Grad}\,f(x),\,\log_z y-\log_z x\rangle_z$ 对某个基点 $z$ 成立。取 $z=x$ 就退化成普通测地凸，所以它比测地凸更强；但它在采样、变分推断等场景里很自然，且是把 RGD 分析做干净所需要的灵活性。两个假设合起来，作者证出黎曼版的**局部 co-coercivity**（命题 3.4），这是把欧氏自适应分析搬过来的关键引理。

**2. 无线搜索的自适应步长规则：在线估局部光滑常数**

算法每步先正常走一步得到 $x_{k+1}$，然后用这一步的实际"梯度变化 vs 函数值变化"反推一个局部光滑代理：

$$L_{k+1}=\frac{-\tfrac12\,\big\|\Gamma_{x_{k+1}}^{x_k}\mathrm{Grad}\,f(x_{k+1})-\mathrm{Grad}\,f(x_k)\big\|_{x_k}^2}{f(x_{k+1})-f(x_k)+s_k\big\langle\Gamma_{x_{k+1}}^{x_k}\mathrm{Grad}\,f(x_{k+1}),\,\mathrm{Grad}\,f(x_k)\big\rangle_{x_k}}.$$

直觉上 $L_{k+1}$ 就是用相邻两步、把曲率影响（通过平行移动 $\Gamma$）纠正后估出的"当前局部有多陡"。拿到 $L_{k+1}$ 后，步长更新分两路：先算两个比例因子

$$r_k^s=\min\Big\{\tfrac{A_{k-1}+1}{A_k},\,\tfrac{A_k}{\tilde B_{k+1}}\Big\},\qquad r_k^L=\begin{cases}\tfrac{A_k}{\tilde B_{k+1}} & s_k<1/L_{k+1}\\[4pt]\big(\tfrac{A_k}{B_k}+\tfrac{\tilde B_{k+1}}{A_k}-1\big) & s_k\ge 1/L_{k+1}\end{cases}$$

再取 $s_{k+1}=\min\{r_k^s\,s_k,\ r_k^L/L_{k+1}\}$。两个 $\min$ 的设计保证步长既不会涨太快（被 $r_k^s s_k$ 压住、维持稳定），又始终被局部光滑代理 $1/L_{k+1}$ 牵引（贴合当前曲率与陡峭程度）。整个过程只需算梯度、指数映射、平行移动这些 RGD 本就要算的几何量，**不做任何额外的线搜索**——相比线搜索动辄要试探多个候选步长、每次都算一遍指数映射，这里每步只多两次度量求值和一次平行移动，渐近时间复杂度与 RGD 同阶。

**3. 三序列 $A_k,B_k,\tilde B_k$ 与曲率因子 $\zeta$ 的吸收**

RAdaGD 之所以是"一族"而非单个算法，是因为步长规则被三个标量序列 $A_k,B_k,\tilde B_k$ 参数化，不同选择给出不同实例。它们不是随便取的，而是被一组耦合不等式（论文式 (10)）约束：$\tfrac{A_k+1}{A_{k+1}}\ge1,\ \tfrac{A_k}{\tilde B_{k+1}}\ge1,\ \big(\tfrac{A_k}{B_k}+\tfrac{\tilde B_{k+1}}{A_k}-1\big)\ge r$。满足这些不等式，就能让一个精心设计的 Lyapunov 函数

$$V_k=s_{k+1}A_k(f(x_k)-f^\star)+\tfrac12 s_k^2 B_k\|\mathrm{Grad}\,f(x_k)\|^2+\tfrac12\|\log_{x_{k+1}}x^\star\|^2$$

单调不增（命题 4.2），进而推出收敛。这里最关键的创新是第三个序列 $\tilde B_k$：它是相对 Suh–Ma 欧氏版**新增**的，专门用来吸收曲率因子 $\zeta_k=\zeta(d(x_k,x^\star))$。$\zeta$ 由流形截面曲率下界 $K_{\min}$ 定义——非负曲率时 $\zeta=1$，负曲率时 $\zeta(\rho)=\rho\sqrt{-K_{\min}}\coth(\rho\sqrt{-K_{\min}})>1$，刻画"测地三角形比欧氏更胖"带来的额外松弛。$\tilde B_k$ 需满足 $\tilde B_{k+1}\ge B_{k+1}+\zeta_{k+1}$，相当于在 Lyapunov 预算里给曲率项留出专门的额度。

取 $A_k=\alpha(k+1)+1+\bar\zeta,\ B_k=\alpha(k+1),\ \tilde B_k=\alpha(k+1)+\bar\zeta$（$\alpha\in(0,1]$，$\bar\zeta=\sup_x\zeta$）就得到定理 4.1 的代表性实例，达到非遍历收敛率 $f(x_k)-f^\star\le O(1/k)$，与固定步长 RGD 同阶，且梯度范数平方以 $O(1/k^2)$ 收敛。作者还按曲率信息分三档给推论：$\bar\zeta$ 有界且**已知**（推论 4.6，最简单）；$\bar\zeta$ **无界**但初始距离 $d(x_0,x^\star)$ 有上界（推论 4.7，靠归纳论证逐步控住 $\zeta_k$）；$\bar\zeta$ 有界但**未知**（推论 4.8，只要让 $\tilde B_k-B_k\to\infty$，存在某个 $k_0$ 后不等式 (12) 自动成立）。三档覆盖了从正曲率到强负曲率、从曲率已知到完全未知的实际场景。代价是：和非自适应 RGD 比，收敛率多了个 $O(\bar\zeta)$ 常数因子——正曲率时无害（$\bar\zeta=1$），强负曲率时可能变大，作者坦言"这是自适应的固有代价还是分析的瑕疵"仍是开放问题。

**4. 落地高斯变分推断：BWAdaGVI**

核心应用是高斯变分推断（GVI）：在所有非奇异高斯里找一个在 KL 意义下最逼近目标分布 $\pi\propto e^{-V}$ 的高斯，即 $\min_{\mu\in\mathrm{BW}(\mathbb R^n)}F(\mu)=\mathbb E_{X\sim\mu}[V(X)]+\mathbb E_{X\sim\mu}[\log\mu(X)]$，$F$ 等于 $D_{\mathrm{KL}}(\mu\|\pi)$ 加常数。高斯空间 $\mathrm{BW}(\mathbb R^n)$ 可用均值-协方差参数化为 $\mathbb R^n\times\mathrm{SPD}(n)$，在 **Bures-Wasserstein 几何**下是一个非负曲率的乘积黎曼流形，故 $\zeta=1$。把 RAdaGD 套上去（目标函数取 $f=F$）天然合适，但需要验证 $F$ 满足本文假设：作者证出当势函数 $V$ 凸（即 $\pi$ 对数凹）且满足一个温和增长条件 $|V(X)|\le C(1+\|X\|^p)\exp(a\|X\|^\beta),\ \beta\in[0,2)$ 时，$F$ 既广义测地凸（引理 5.2）又局部测地光滑（命题 5.3）。后者非平凡，因为 $\mathrm{BW}(\mathbb R^n)$ 测地不完备，不能直接套定理 3.3，作者另证了它。

由于测地不完备，直接搬推论 4.6 不行，需对算法做小改造——加一步对步长的额外裁剪 $s_{k+1}\leftarrow\min\{s_{k+1},\ (1-\delta)/\max_i\lambda_i(\mathbb E_{X\sim\mu_{k+1}}[\nabla^2 V(X)]-\Sigma_{k+1}^{-1})\}$，并要求迭代协方差的最小特征值一致有下界 $\lambda_{\min}(\Sigma_k)\ge\epsilon$，由此得到 **BWAdaGVI**（推论 5.4），保住与推论 4.6 相同的收敛保证。这一步的意义在于：以往 GVI 算法（Lambert 2022、Diao 2023 等）的收敛证明都依赖 $V$ 全局 $L$-光滑，而很多实际模型（如贝叶斯泊松回归，$V(\theta)=\sum_i(\exp(X_i^T\theta)-Y_iX_i^T\theta)$ 因指数依赖 $\theta$ 而非全局 $L$-光滑）不满足。BWAdaGVI 用"凸 + 温和增长"替换 $L$-光滑，是据作者所知**第一个**在非 $L$-光滑目标下给 GVI 提供可证收敛的算法。

### 损失函数 / 训练策略
本文是优化算法本身，没有训练目标；"目标函数"即被优化的 $f$（GVI 里是 KL 散度 $F$）。实践中作者推荐的序列选择（引理 5.1）是 $A_k=2(k+2)^{0.1}+2,\ B_k=2(k+2)^{0.1},\ \tilde B_k=2(k+2)^{0.1}+1$，满足非负曲率下推论 4.6 的条件，给出 $O(1/k^{0.1})$ 的（较保守但实测更快）速率——作者指出定理 4.1 的理论参数虽然 sound 但步长偏保守，故实验另选这套。

## 实验关键数据

### 主实验
论文实验聚焦在一个能凸显"非 $L$-光滑"价值的代表任务：贝叶斯泊松回归的 GVI（$\ell=50$ 个样本、$n=25$ 维，$X_i\sim N(0,I_n),\ Y_i\sim\mathrm{Poisson}(\exp(\theta^TX_i))$），对比对象是 SOTA 的 FBGVI（Diao et al., 2023）。因为泊松回归的 $V$ 不全局 $L$-光滑，FBGVI 在此任务**没有理论保证**，而 RAdaGD/BWAdaGVI 有。

| 任务 | 指标 | 本文 (BWAdaGVI) | 对比 (FBGVI) | 结论 |
|------|------|------|----------|------|
| 泊松回归 GVI | $F(\mu_k)-F_{\min}$（步长 $s=1,1/2,1/4$） | 更低、收敛更快 | 较慢 | 各步长下均优于 FBGVI（图 1 左） |
| 泊松回归 GVI | $\lambda_{\min}(\Sigma_k)$ 随迭代 | 始终远离 0 | — | 经验上验证推论 5.4 的特征值下界假设温和（图 1 右） |

### 消融实验
本文是理论论文，无传统意义上的模块消融；与"消融"最接近的是**曲率依赖性分析**——在 $f$ 测地 $L$-光滑、$\alpha=1$ 的简化设定下显式比较自适应 vs 非自适应的收敛常数。

| 设定 | 收敛率上界（首项常数） | 说明 |
|------|---------|------|
| RAdaGD（本文，定理 4.1） | $\sim\frac{3+\bar\zeta}{2(k+2+\bar\zeta)}(\cdots)Ld(x_0,x^\star)^2$，含额外 $O(\bar\zeta)$ 因子 | 自适应、无需预知 $L$ |
| 经典固定步长 RGD（Zhang–Sra 2016） | $\frac{\bar\zeta L}{2(k-1+\bar\zeta)}d(x_0,x^\star)^2$ | 需预知 $L$ |

### 关键发现
- **曲率因子 $\zeta$ 是自适应的"成本中心"**：非负曲率时 $\bar\zeta=1$，RAdaGD 与固定步长 RGD 几乎同阶（只多一个常数）；强负曲率时多出的 $O(\bar\zeta)$ 因子可能变大。这个因子到底是自适应的固有代价还是分析瑕疵，是论文留下的开放问题。
- **非 $L$-光滑才是真正的卖点**：在泊松回归这种全局 $L$-光滑失效的任务上，RAdaGD 不仅有理论保证，实测还稳定快过依赖 $L$-光滑假设的 FBGVI——说明放宽假设不是"牺牲性能换通用性"，而是两头都拿到。
- **特征值下界假设温和**：BWAdaGVI 理论上需要 $\lambda_{\min}(\Sigma_k)\ge\epsilon$，图 1 右经验显示迭代协方差的最小特征值始终远离零，假设在实践中几乎不构成限制。

## 亮点与洞察
- **"在线估局部 $L$"这一招最巧**：把固定步长 $1/L$ 换成用相邻两步反推的 $1/L_{k+1}$，一举解决了"$L$ 未知"和"全局 $L$-光滑太强"两个问题——既不用线搜索，又把适用函数类从全局 $L$-光滑扩到所有 $C^2$ 函数。这种"用一阶信息在线标定二阶常数"的思路可迁移到很多需要预知光滑常数的算法。
- **第三序列 $\tilde B_k$ 是干净处理曲率的关键**：把欧氏方法搬到流形最大的障碍是曲率项 $\zeta$，作者没有粗暴放进步长，而是新设一个序列在 Lyapunov 预算里专门给曲率留额度（$\tilde B_{k+1}\ge B_{k+1}+\zeta_{k+1}$）。这种"为新增的耦合项单设一个吸收变量"的设计模式很值得借鉴。
- **定理 3.3（$C^2\Rightarrow$ 局部测地光滑）虽小却是支点**：一个在欧氏看似显然、黎曼却没人正式写过的命题，撑起了整套"弱假设"的合法性，也直接成全了 GVI 的非光滑保证。提醒做理论时，把"显然但没人证过的桥"补上往往打开新应用。
- **三档曲率推论覆盖现实**：按 $\bar\zeta$ 已知/无界/未知分别给配套保证，把"理论上漂亮"做成"各种曲率与先验信息条件下都能用"，工程友好度高。

## 局限与展望
- **广义测地凸仍是较强假设**：作者承认它比普通测地凸强；虽然在采样/变分推断里自然，但限制了适用函数类。
- **$O(\bar\zeta)$ 因子在强负曲率下可能放大**：自适应相比非自适应 RGD 多付的曲率常数，在强负曲率流形上可能不小，且其"固有 vs 分析瑕疵"未解。
- **BWAdaGVI 依赖额外技术假设**：需要协方差最小特征值一致下界 $\lambda_{\min}(\Sigma_k)\ge\epsilon$，以及 BW-梯度可解析或可随机近似；目前理论未覆盖随机近似版本。
- **实验规模偏小、任务单一**：主实验只有一个 $\ell=50,n=25$ 的泊松回归 GVI 算例，更广的几何与任务放在附录，正文证据较薄。
- **展望**：把欧氏自适应加速（Nesterov 型）搬到流形、做近端变体、发展带可控误差的随机版本、引入 retraction / 非精确平行移动以贴近实际计算，都是自然的下一步。

## 相关工作与启发
- **vs 固定步长 RGD（Zhang–Sra 2016）**：后者需预知 $L$ 且要求全局测地 $L$-光滑；本文在线估 $L_{k+1}$、只需局部测地光滑，适用函数类大幅扩张，代价是收敛常数多 $O(\bar\zeta)$ 因子（非负曲率下可忽略）。
- **vs Suh–Ma (2025) 欧氏自适应方法**：本文的直接灵感来源；区别在于步长规则与分析都要处理曲率项 $\zeta$，为此新增第三序列 $\tilde B_k$，并分曲率已知/无界/未知三档给保证——这部分是从欧氏到黎曼"高度非平凡"的延伸。
- **vs FBGVI / 既有 GVI 算法（Lambert 2022、Diao 2023、Kim 2023）**：它们的收敛证明都要求目标对数密度 $V$ 全局 $L$-光滑；本文用"凸 + 温和增长"替换，给出非 $L$-光滑情形下 GVI 的首个可证收敛保证，且在泊松回归上实测更优。
- **启发**：本文示范了"先补一个看似显然但没人证过的基础命题（$C^2\Rightarrow$ 局部光滑），再据此放宽全套假设、打开新应用"的理论推进范式；以及"为跨设定新增的耦合项单设吸收序列"的分析技巧，二者都可迁移到其他从欧氏向流形（或向更弱假设）推广的优化工作。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个达到 $O(1/k)$ 非遍历率的黎曼自适应方法，且给出 GVI 非 $L$-光滑情形的首个收敛保证。
- 实验充分度: ⭐⭐⭐ 理论扎实但正文实验仅一个小规模泊松回归算例，更多任务压在附录。
- 写作质量: ⭐⭐⭐⭐ 假设—算法—收敛—应用的逻辑链清晰，曲率三档处理交代得当；符号偏密、对非黎曼背景读者门槛较高。
- 价值: ⭐⭐⭐⭐ 把欧氏自适应优化干净地搬上流形，并直接惠及变分推断这一高频应用，理论与应用两端都有落点。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] An Adaptive Algorithm for Bilevel Optimization on Riemannian Manifolds](../../NeurIPS2025/optimization/an_adaptive_algorithm_for_bilevel_optimization_on_riemannian_manifolds.md)
- [\[ICLR 2026\] Strongly Convex Sets in Riemannian Manifolds](strongly_convex_sets_in_riemannian_manifolds.md)
- [\[ICLR 2026\] Riemannian Zeroth-Order Gradient Estimation with Structure-Preserving Metrics for Geodesically Incomplete Manifolds](riemannian_zeroth-order_gradient_estimation_with_structure-preserving_metrics_fo.md)
- [\[ICLR 2026\] Corner Gradient Descent](corner_gradient_descent.md)
- [\[NeurIPS 2025\] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies](../../NeurIPS2025/optimization/natural_gradient_descent_for_improving_variational_inference_based_classificatio.md)

</div>

<!-- RELATED:END -->
