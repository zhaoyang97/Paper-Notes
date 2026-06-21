---
title: >-
  [论文解读] Physics-informed learning under mixing: How physical knowledge speeds up learning
description: >-
  [ICLR 2026][学习理论][物理信息正则化] 本文为"带物理信息正则化的经验风险最小化"在**非独立同分布（依赖/混合）数据**下推导出超额风险的高概率界与期望界，证明只要物理先验与真值"对齐"（$\|D(f^\star)\|_{L^2}^2\simeq 0$），学习率就能从慢的 Sobolev 极小极大率 $O(T^{-d})$ 一路加速到与 i.i.d. 最优率一致的 $O(1/T)$，且**不会因为数据依赖而损失有效样本量**。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "统计学习"
  - "物理信息正则化"
  - "依赖数据"
  - "超额风险界"
  - "Sobolev 极小极大率"
  - "鞅偏移复杂度"
---

# Physics-informed learning under mixing: How physical knowledge speeds up learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=IvLVPbeoRx](https://openreview.net/forum?id=IvLVPbeoRx)  
**代码**: 待确认  
**领域**: 学习理论 / 统计学习  
**关键词**: 物理信息正则化, 依赖数据, 超额风险界, Sobolev 极小极大率, 鞅偏移复杂度

## 一句话总结
本文为"带物理信息正则化的经验风险最小化"在**非独立同分布（依赖/混合）数据**下推导出超额风险的高概率界与期望界，证明只要物理先验与真值"对齐"（$\|D(f^\star)\|_{L^2}^2\simeq 0$），学习率就能从慢的 Sobolev 极小极大率 $O(T^{-d})$ 一路加速到与 i.i.d. 最优率一致的 $O(1/T)$，且**不会因为数据依赖而损失有效样本量**。

## 研究背景与动机
**领域现状**：物理信息机器学习（physics-informed ML）把已知的物理规律——典型地是一个偏微分方程（PDE）——注入到学习算法中，常见做法是在损失里加一个"物理一致性"正则项，或把它当作约束。大量经验工作（PINN 一脉）表明这样能显著提升数据效率、泛化与可解释性，对下游的安全控制尤其重要。

**现有痛点**：物理先验"经验上有用"是公认的，但**理论上"到底有多有用、为什么有用"几乎没人说清**。已有的统计学习分析几乎都假设数据是 i.i.d.，而真实场景里数据往往来自一条动力学轨迹 $X_{t+1}=f^\star(X_t)+W_t$——前后样本强相关。

**核心矛盾**：处理依赖数据的经典手法是"分块"（blocking）——把长度为 $T$ 的轨迹切成长度 $k$ 的块、让相邻块近似独立。但这会把有效样本量从 $T$ 缩水到 $T/k$，于是收敛率被拖慢，而且 Nagaraj et al. (2020) 证明这种"样本缩水"在最坏情况下是**不可避免**的。一句话：依赖数据天然导致更慢的率，而物理先验能不能把它救回来，没人量化过。

**本文目标**：在统计学习框架下，对带物理信息正则的（正则化）ERM，推导**依赖于复杂度**的超额风险界——即界的大小随 $\|D(f^\star)\|_{L^2}$（先验与真值的偏差）缩放——并刻画这个界对应的收敛率。

**切入角度**：作者把 Mendelson 的 small-ball 方法、以及 Ziemann & Tu (2022) 对非线性依赖数据的局部化分析，扩展到"带椭圆 PDE 算子正则"的加权、向量值 Sobolev 空间上。关键直觉是：物理正则不只是缩小假设空间，它能让分析里的"复杂度项"摆脱对 mixing（混合系数）的依赖。

**核心 idea**：用一个编码 PDE 的椭圆算子 $D$ 构造正则项 $\Psi(f)=\|D(f)\|_{L^2}^2$，并证明在"知识对齐"下，依赖数据带来的代价会被**整体转移到一次性的 burn-in 时间**里，从而超额风险的主项恢复成 i.i.d. 的 $O(1/T)$。

## 方法详解

### 整体框架
这是一篇纯理论论文，没有算法 pipeline，"方法"指的是一条**证明链**：从一个带物理正则的优化问题出发，把"难以直接分析的超额风险"逐步换成"可控的复杂度量"，最后优化各个自由参数得到收敛率。整体要解决的问题是——在数据依赖时，超额风险 $\|\hat f-f^\star\|_{L^2}^2$ 的收敛率到底是多少、物理先验如何改变它。

学习问题设定如下：数据是一条长度 $T$ 的轨迹 $\mathcal D=\{X_t,Y_t\}_{t=0}^{T-1}$，由非线性动力系统 $Y_t=X_{t+1}=f^\star(X_t)+W_t$ 生成，其中 $W_t$ 是关于轨迹 σ-代数滤波的鞅差序列，且条件次高斯（方差代理 $\sigma_W^2$）。估计量由正则化 ERM 给出：

$$\hat f=\arg\min_{f\in F}\ \frac{1}{T}\sum_{t=0}^{T-1}\|Y_t-f(X_t)\|_2^2+\lambda_T\,\Psi(f),$$

假设空间 $F$ 取为加权、向量值 Sobolev 空间 $H^s$ 中半径 $\rho_f$ 的球（$s\ge 2d_X$），并假设 $f^\star\in F$（否则会多出一个与数据无关的确定性偏置项，可解耦处理）。

证明链分四步推进，与下面的「关键设计」一一对应：① 用椭圆 PDE 算子 $D$ 把物理先验写成正则项 $\Psi$；② 用单边集中不等式把超额风险（$L^2$ 范数）上界化为其经验版本，转移到一个临界球面 $\partial B(r)$ 上；③ 把经验版本进一步用**有效假设类的鞅偏移复杂度** $M_T[F^\rho]$ 控住，并借助"自归一化"让这个复杂度**不依赖 mixing**；④ 引入 S-persistence 与 $(C,\alpha)$-超压缩性来界定"下等距事件"的概率，从而把数据依赖的代价挤进 burn-in 时间。最后通过优化临界半径 $r$、有效类半径 $\rho$ 和正则参数 $\lambda_T$，得到既有"慢的 Sobolev 项"又有"快的 i.i.d. 项"的界。

### 关键设计

**1. 物理先验编码为椭圆算子正则：让"近似满足 PDE"成为可分析的惩罚项**

痛点是"物理先验有用"无法被分析工具捕捉。作者把物理知识形式化为：$f^\star$ 近似满足一个由线性微分算子 $D:H^s\to L^2$ 诱导的 PDE，算子按分量写成 $[D(f)]_i=\sum_{|\alpha|\le s}p_{i,\alpha}D^\alpha f_i$，系数 $p_{i,\alpha}$ 有界。正则项就是这个算子残差的 $L^2$ 范数平方

$$\Psi(f)=\|D(f)\|_{L^2(\Omega^T,P_X;\mathbb R^{d_Y})}^2,$$

并定义**知识对齐（knowledge alignment）** 为 $\|D(f^\star)\|_{L^2}^2\simeq 0$——即真值"几乎严格"满足该 PDE。这里有两个关键技术选择：一是要求 $D$ 是**椭圆**算子（Assumption 4：$\sum_{|\alpha|=s}p_{i,\alpha}\xi^\alpha\ne 0$），它是 Laplace/Poisson 算子的推广，保证正则项能给假设类带来足够的"刚性"以压低复杂度；二是 $\Psi$ 被证明是 Lecué & Mendelson 意义下的 *2-proper* 正则子，这使得后续的复杂度依赖型分析得以套用。换句话说，正则项不再只是"加个惩罚让解平滑"，而是把"真值落在 $D$ 的近似零空间里"这条具体物理信息，翻译成了能进入熵数/复杂度估计的结构。

**2. 单边集中 + 下等距事件：把超额风险换成它的经验版本**

直接分析 $L^2$ 超额风险很难，因为底层概率测度未知。作者沿用 Mendelson (2014) 的 small-ball 思路，构造一个高概率事件，使得对某参数 $\theta$ 有

$$\|f-f^\star\|_{L^2}^2\le\frac{\theta}{T}\sum_{t=0}^{T-1}\|f(X_t)-f^\star(X_t)\|_2^2.$$

这条**单边集中不等式**把"理论范数"压到"经验范数"上，于是分析对象从不可观测的 $L^2$ 风险变成可处理的经验风险。它的失败集合被定义为**下等距事件（lower isometry event）** $A_r$，其概率由临界半径 $r$ 控制——$r$ 越大，不等式越容易成立。在依赖数据下，i.i.d. 版的 small-ball 条件不再直接可用，作者改用 $\alpha=2$ 的 $(C,\alpha)$-超压缩性在临界球面 $\partial B(r)=\{f\in F:\|f-f^\star\|_{L^2}^2=r^2\}$ 上重建这个"小球"性质。这一步是整条证明的支点：它把"估计有多准"的问题，转成"经验损失在临界球面上有多大"的问题。

**3. 鞅偏移复杂度 + 自归一化：让复杂度项摆脱对 mixing 的依赖**

经验风险这一侧需要一个能反映"有效假设类有多复杂"的量。作者用**鞅偏移复杂度（martingale offset complexity）**：对有效假设类 $F^\rho_\star$（已平移到以 $f^\star$ 为中心）中的每个 $f$，

$$\frac{1}{T}\sum_{t=0}^{T-1}\|f(X_t)\|_2^2\ \le\ \sup_{f\in F^\rho_\star}\frac{1}{T}\sum_{t=0}^{T-1}\Big(4\langle W_t,f(X_t)\rangle_2-\|f(X_t)\|_2^2\Big)\ \overset{.}{=}\ M_T[F^\rho_\star].$$

精妙之处在右边的 $-\|f(X_t)\|_2^2$ 这个"偏移（offset）"项：它带来**自归一化效应**，恰好抵消噪声内积 $\langle W_t,f(X_t)\rangle_2$ 的波动（思路来自 Liang et al. 2015）。正是这个抵消，使得鞅偏移复杂度**不依赖于数据的 mixing 程度**——也就是说，复杂度本身不会因为样本相关而变大。这与"分块法把复杂度/有效样本量直接打折"形成鲜明对比，是本文能拿到快率的核心机制。有效假设类 $F^\rho=\{f\in F:\Psi(f-f^\star)\le\rho\}$ 由正则项诱导，物理先验越强（$\rho$ 越小），这个类越小、复杂度越低。

**4. S-persistence + 超压缩性：把数据依赖的代价转移进 burn-in 时间**

剩下要量化的是"下等距事件失败的概率"，它直接关联数据的依赖结构。作者用两个对依赖数据的弱正则性假设来处理：**S-persistence**（Assumption 6，一条单边指数不等式，$S\in[1,\infty)$ 越大表示 $X_t$ 越依赖过去，对大量 Markov 链与 $\phi$-mixing 过程成立且 $S$ 可取常数）和 $(C,\alpha)$-**超压缩性**。把期望超额风险按 $A_r$ 及其补集分解，借助 S-persistence、超压缩性和有界性 $\|f\|_{L^\infty}\le B$，就能把下等距事件的概率压成对 $T$ 指数衰减的项 $\sim B^2 N_\infty(\partial B(r),r/\sqrt\theta)\exp(-8T/(\theta^2 C_r S))$。这一步的意义是：**数据依赖（$S$）只出现在指数项里，也就是只影响"需要多少样本 burn-in"，而不污染超额风险的主项**。于是最终的界形如

$$\mathbb E\big[\|\hat f-f^\star\|_{L^2}^2\big]\le C_{\text{slow}}\frac{\Psi(f^\star)^{d'/2}}{T^{d}}+C_{\text{fast}}\frac{\sigma_W^2}{T},$$

其中 $d=\frac{2s}{2s+d_X}$ 是 Sobolev 极小极大率、$d'=\frac{2d_X}{2s+d_X}$。当知识对齐 $\Psi(f^\star)\simeq 0$ 时，慢的 Sobolev 项被湮灭，只剩快的 $O(1/T)$——这就是"物理知识加速学习"的精确表述。代价仅仅是更长的 burn-in 时间 $T$（其下界显式依赖 $S$、$r$、$d_X$、$s$）。

### 损失函数 / 训练策略
优化目标即上面的正则化 ERM（公式 3.3）：数据拟合的平方损失 + $\lambda_T\Psi(f)$。理论给出了让界成立的**正则参数下界**，例如期望界要求 $\lambda_T\ge\frac{4(C_I+C_{II})(\sigma_W^2)^d}{3T\,\Psi(f^\star)^{1-d'/2}}$，以及有效类半径 $\rho\ge 10\,\Psi(f^\star)$；高概率界对 $\lambda_T$ 有类似但更细的下界（含 $\log(1/\delta)$ 项）。这些条件本质上是在"正则强度足够压住复杂度"与"不过度正则引入偏差"之间取一个由样本量 $T$ 决定的平衡点。

## 实验关键数据
本文是纯理论工作，**没有数值实验**，其"关键数据"是与已有率的理论对比，以及主定理给出的率。

### 主结果：收敛率对比

| 工作 | 假设类 | 数据 | 正则化 | 收敛率 |
|------|--------|------|--------|--------|
| Nussbaum (2006) | $L^2$ Sobolev | i.i.d. | 无 | $\sigma_W^2\,T^{-2s/(2s+1)}$ |
| Farahmand & Szepesvári (2012) | 一般 Sobolev | 非 i.i.d. | 无 | $T^{-d}\log T$ |
| Lecué & Mendelson (2017) | 一般 | i.i.d. | proper 正则 | 复杂度依赖型 |
| Ziemann & Tu (2022) | 一般 | 非 i.i.d. | 无 | 由熵数 $q=d_X/s$ 决定 |
| **本文** | 加权向量值 Sobolev | **非 i.i.d.（混合）** | 物理信息正则 | $C_{\text{slow}}\Psi(f^\star)^{d'/2}T^{-d}+C_{\text{fast}}\sigma_W^2T^{-1}$ |

### 两条主定理（期望/高概率）

| 设定 | 率（超额风险） | 对齐后（$\Psi(f^\star)\simeq 0$） |
|------|----------------|----------------------------------|
| 期望界（Thm 5.2） | $C_{\text{slow}}\dfrac{\Psi(f^\star)^{d'/2}}{T^{d}}+C_{\text{fast}}\dfrac{\sigma_W^2}{T}$ | $O(1/T)$（i.i.d. 最优率） |
| 高概率界（Thm 5.1，$\ge 1-6\delta$） | $C_{\text{slow}}\dfrac{\max\{\Psi(f^\star)^{d'/4},\Psi(f^\star)^{d'/2}\}}{T^{d}}+C_{\text{fast}}\dfrac{\sigma_W^2\log(1/\delta)}{T}$ | $O(\log(1/\delta)/T)$ |

### 关键发现
- **数据依赖被"挪走"了**：依赖参数 $S$ 只进入 burn-in 时间条件（对 $T$ 的下界），不进入超额风险的渐近主项——这正是相对"分块法必然样本缩水"的本质突破。
- **加速来自对齐而非正则本身**：只有当 $\Psi(f^\star)\simeq 0$（先验与真值一致）时慢项才消失；若先验有偏（$\Psi(f^\star)$ 大），慢的 Sobolev 项仍在，率退回 $T^{-d}$。物理先验"对不对"直接决定能否加速。
- **首次在概率意义上给出复杂度依赖界**：相较最相近的 Douměche et al. (2024)（i.i.d. + 仅期望界），本文同时覆盖非 i.i.d. 数据与高概率界。

## 亮点与洞察
- **自归一化是"打败 mixing"的关键招**：鞅偏移复杂度里的 $-\|f(X_t)\|_2^2$ 偏移项让噪声波动自我抵消，使复杂度量与混合系数脱钩——这是把依赖数据的代价从"率"挪到"burn-in"的技术枢纽，值得借鉴到其他依赖数据的泛化分析。
- **"复杂度依赖型界"把先验质量量化进了率**：界显式地随 $\|D(f^\star)\|_{L^2}$ 缩放，等于给出了一把尺子——物理先验越准，率越快、burn-in 越短，把"物理知识有多有用"从定性说法变成可计算的指数。
- **加权、向量值 Sobolev + 椭圆性的处理可迁移**：为支撑分析，作者把标量 Sobolev 嵌入定理、超压缩性、熵数估计等都推广到加权向量值空间并处理算子非平凡零空间/边界条件，这套工具对其他"PDE 约束 + 学习"的理论分析是现成的脚手架。

## 局限性 / 可改进方向
- **加速以更长 burn-in 为代价**：依赖越强（$S$ 越大）、维度越高（$d_X$ 大、$s$ 相对小），所需最小样本量越大，定理只在 $T$ 充分大后才生效；小样本/强依赖场景下加速可能看不到。
- **强假设链**：椭圆性、$s\ge 2d_X$、密度上下有界（Assumption 1）、$f^\star\in F$、$S$ 为常数等假设较多；若 $f^\star\notin F$ 会多出一个未在本文分析的确定性偏置项。
- **完全没有数值验证**：所有结论都是渐近率与常数依赖，缺乏哪怕一个 toy PDE 系统上的实验来展示 burn-in 实际有多长、对齐程度如何影响有限样本表现。
- **改进思路**：给出 $C_{\text{slow}}/C_{\text{fast}}$ 与 burn-in 的可计算上界并做实证标定；放宽对齐为"部分对齐"并刻画率的连续过渡；推广到非椭圆或时变算子、以及 $S$ 随 $T$ 增长的更强依赖场景。

## 相关工作与启发
- **vs 分块法（Yu 1994; Sancetta 2021 等）**：分块把轨迹切成近独立块来套 i.i.d. 工具，但有效样本量缩水导致率次优（Nagaraj et al. 2020 证明最坏情况不可避免）。本文不分块，靠单边集中 + 自归一化把依赖代价转入 burn-in，主项率不打折。
- **vs Ziemann & Tu (2022)**：本文建立在其对非线性依赖数据的 small-ball/偏移复杂度局部化分析之上，但**不是简单套用**——物理正则带来新困难：要刻画有效假设类的熵数（椭圆性、算子非平凡零空间、边界条件）、确定轨迹超压缩性、并在加权向量值 Sobolev 空间上工作。
- **vs Douměche et al. (2024)**：精神最接近（都推导复杂度依赖率），但对方处理 i.i.d. 数据且只给期望界；本文核心推进是**非 i.i.d. 数据 + 同时给出期望与高概率界**。
- **vs 谱方法路线（Caponnetto & De Vito 2007 等）**：统计学习导出率有两条主线——RKHS 积分算子谱分析 vs 经验过程 + small-ball；本文属于后者，把 small-ball 方法适配到依赖数据并叠加物理正则。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在依赖/混合数据下量化物理先验对学习率的加速，并同时给出概率与期望的复杂度依赖界。
- 实验充分度: ⭐⭐⭐ 纯理论、证明完整且有率对比表，但完全没有数值验证 burn-in 与有限样本行为。
- 写作质量: ⭐⭐⭐⭐ 设定、假设与证明链条清晰，非形式化定理先给直觉；符号偏重，对非理论读者门槛较高。
- 价值: ⭐⭐⭐⭐⭐ 把"物理知识有多有用"从经验说法变成可计算的率与 burn-in，为安全学习控制等下游提供了坚实理论支撑。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Preventing Model Collapse Under Overparametrization: Optimal Mixing Ratios for Interpolation Learning and Ridge Regression](preventing_model_collapse_under_overparametrization_optimal_mixing_ratios_for_in.md)
- [\[ICLR 2026\] Learning under Quantization for High-Dimensional Linear Regression](learning_under_quantization_for_high-dimensional_linear_regression.md)
- [\[ICLR 2026\] How Reinforcement Learning after Next-Token Prediction Facilitates Learning](how_reinforcement_learning_after_next-token_prediction_facilitates_learning.md)
- [\[ICLR 2026\] How hard is learning to cut? Trade-offs and sample complexity](how_hard_is_learning_to_cut_trade-offs_and_sample_complexity.md)
- [\[ICLR 2026\] The Lie of the Average: How Class Incremental Learning Evaluation Deceives You?](the_lie_of_the_average_how_class_incremental_learning_evaluation_deceives_you.md)

</div>

<!-- RELATED:END -->
