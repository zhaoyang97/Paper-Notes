---
title: >-
  [论文解读] Prediction with Expert Advice under Local Differential Privacy
description: >-
  [ICLR 2026][在线学习][专家建议预测] 本文研究本地差分隐私（LDP）约束下的「专家建议预测」在线学习问题：先指出经典随机游走算法 RW-FTPL 天然满足 LDP，再设计两个改进——RW-AdaBatch 利用「换手次数少」这一性质做自适应分批，在数据越「简单」时隐私放大越强且几乎不损失 utility；RW-Meta 用共享噪声的方式在「不额外消耗隐私预算」的前提下，从一组本身就是学习算法的数据依赖专家中做私有选择，在真实 COVID-19 医院数据上比经典基线和 SOTA 中心化 DP 算法高出 1.5–3 倍。
tags:
  - "ICLR 2026"
  - "在线学习"
  - "学习理论"
  - "差分隐私"
  - "专家建议预测"
  - "本地差分隐私"
  - "随机游走"
  - "隐私放大"
  - "动态环境"
---

# Prediction with Expert Advice under Local Differential Privacy

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=B9H2705C7c](https://openreview.net/forum?id=B9H2705C7c)  
**代码**: https://github.com/ben-jacobsen/dp-online-learning  
**领域**: 在线学习 / 学习理论 / 差分隐私  
**关键词**: 专家建议预测、本地差分隐私、随机游走、隐私放大、动态环境

## 一句话总结
本文研究本地差分隐私（LDP）约束下的「专家建议预测」在线学习问题：先指出经典随机游走算法 RW-FTPL 天然满足 LDP，再设计两个改进——RW-AdaBatch 利用「换手次数少」这一性质做自适应分批，在数据越「简单」时隐私放大越强且几乎不损失 utility；RW-Meta 用共享噪声的方式在「不额外消耗隐私预算」的前提下，从一组本身就是学习算法的数据依赖专家中做私有选择，在真实 COVID-19 医院数据上比经典基线和 SOTA 中心化 DP 算法高出 1.5–3 倍。

## 研究背景与动机

**领域现状**：专家建议预测（Prediction with Expert Advice）是在线学习的经典框架——每一步玩家收到 $n$ 个专家的建议，必须决定听谁的，随后环境揭示一个增益向量 $g_t \in [0,1]^n$，玩家拿到所选动作对应的回报，目标是最大化整局累计回报、即最小化对「事后最优固定动作」的静态 regret。当这些专家建议涉及人类行为（医疗诊断、人口流动、用电、公共卫生）时，纯粹追求精度的算法可能通过输出泄露敏感数据，因此需要差分隐私。

**现有痛点**：以往工作几乎都走「中心化 DP（CDP）」路线，即假设有一个可信中心化 curator 拿到全部原始数据后再加噪。但可信 curator 在高敏感场景里恰恰难以建立。本地 DP（LDP）——在数据产生处就给每个增益向量加噪、算法只能看到加噪后的序列——更适合这类场景，且实现简单（只需在数据源采样独立随机数，不需要安全聚合那套多轮交互和密码学协议）。然而 LDP 下的专家算法几乎是一片空白，已知的相关工作只有一篇（且面向联邦+随机对手，技术路线不同）。

**核心矛盾**：第一，LDP 把噪声加在每个数据点上、而非加在分析结果上，噪声规模天然更大，直接用会比非私有基线差一个 $\sqrt{n}/\mu$ 的因子——怎么在不破坏隐私的前提下把这种「逐点噪声」用得更省？第二，非私有文献里「专家」常常本身就是会随数据自适应的学习算法，但在私有场景下，「私有地选出最好的专家」和「真的去执行这个专家的建议」是两回事——后者若专家本身不满足 DP 就会泄露隐私，于是以往工作只敢用与数据无关的「静态专家」，这在动态环境里是严重的局限。

**切入角度**：作者注意到他们要 build 的经典算法 RW-FTPL（Devroye et al., 2013）有一个被忽视的性质——它设计之初就是为了「换手次数少」（很少改变预测）。这种「有限换手」意味着算法对邻近数据点的排列几乎不变，而「排列不变」正是 shuffle 模型做隐私放大的核心。作者把这条线索同时用到了两个方向：用它做自适应分批（RW-AdaBatch），以及用它构造一套共享噪声机制让多个数据依赖专家「藏在同一份噪声后面」（RW-Meta）。

**核心 idea**：把 RW-FTPL 的「有限换手 / 排列不变」从一个 utility 性质重新解读为隐私资源——分批时让数据点找到「人群」来隐藏（放大隐私），选专家时让所有专家复用同一份加噪增益向量（后处理不变性 ⇒ 零额外隐私成本）。

## 方法详解

### 整体框架

全文围绕一个经典算法 **RW-FTPL** 展开，并向两个方向改进。RW-FTPL 的做法是：开局采样一条高斯随机游走噪声 $z_0,\dots,z_T \stackrel{iid}{\sim} \mathcal{N}(0,\eta^2 I_n)$，每一步选择 $x_t=\arg\max_i (G_{t-1}+S_{t-1})_i$，其中 $G_t=\sum_{s=0}^t g_s$ 是累计真实增益、$S_t=\sum_{s=0}^t z_s$ 是累计噪声（一条对称随机游走，算法名由此而来）。换一个视角，定义加噪增益 $\tilde g_t := g_t + z_t$，则 RW-FTPL 不过是对这些加噪向量的后处理；而每个 $\tilde g_t$ 本身满足本地高斯 DP，隐私参数 $\mu=\Delta/\eta$，其中敏感度 $\Delta=\max_{g,g'}\lVert g-g'\rVert_2\le\sqrt{n}$。它的期望静态 regret 至多 $(\eta+\tfrac{2}{\eta})\sqrt{2T\log n}$。

在此之上：**RW-AdaBatch**（静态环境）保持 RW-FTPL 的输出基本不变，但通过「在保证预测大概率不变时把数据自适应地批在一起」，让落在大批次里的点享受更强的隐私；**RW-Meta**（动态环境）把「专家」升级成本身会学习的算法（learner），并让所有 learner 复用同一份加噪增益向量，从而在零额外隐私成本下做私有的专家选择与执行。三者的隐私/regret 对照如下（$T$ 步、$n$ 专家、$m$ learner）：

| 算法 | 隐私保证 | 渐近 regret | regret 定义 |
|------|----------|-------------|-------------|
| Agarwal & Singh (2017) | $(\varepsilon,\delta)$-CDP | $\frac{\sqrt{T\log n}+\sqrt{n\log n}\,\log^{1.5}T}{\varepsilon}$ | 事后最优静态动作 |
| RW-FTPL (Devroye et al., 2013) | $\mu$-GLDP | $\frac{\sqrt{Tn\log n}}{\mu}$ | 事后最优静态动作 |
| **RW-AdaBatch**（参数 $\alpha>0$） | $\mu$-GLDP；放大后 GCDP | $\frac{(1+\alpha)\sqrt{Tn\log n}}{\mu}$ | 事后最优静态动作 |
| **RW-Meta**（$m$ learner） | $\mu$-GLDP | $\frac{\sqrt{Tnm\log m}}{\mu}$ | 事后最优 learner |

### 关键设计

**1. RW-FTPL 天然满足 LDP：把「随机游走扰动」重写成对加噪增益的后处理**

第一个贡献其实是一个观察：作为基础的经典算法 RW-FTPL 不需要任何修改就已经是 LDP 算法。关键在换一个视角看它的决策规则——把每步注入的随机游走增量 $z_t$ 直接并进数据，得到加噪增益 $\tilde g_t=g_t+z_t$，那么 $\arg\max_i(G_{t-1}+S_{t-1})_i$ 就等价于对一串 $\tilde g_t$ 的后处理。由于每个 $\tilde g_t$ 是把高斯噪声加在增益向量上、噪声规模按敏感度 $\Delta\le\sqrt n$ 标定，单点就满足 $\mu$-高斯 LDP，$\mu=\Delta/\eta$。再由 DP 的后处理不变性，整条算法满足 $\mu$-GLDP。这一步把「为了少换手而设计的非私有算法」直接接进了隐私框架，也为后两个算法提供了同一根隐私骨架——它们都不另外加噪，只是更聪明地复用这同一份噪声。

**2. RW-AdaBatch：把「有限换手」变成自适应分批，做出随数据自动变强的隐私放大**

RW-FTPL 的设计初衷是「很少改变预测」，直觉上这意味着它对邻近数据点的排列几乎不变。RW-AdaBatch 把这种「几乎不变」做成「在许多数据依赖区间上严格排列不变」：只要能高概率保证接下来若干步预测不会变，就把这些步的数据批在一起（buffer 累积、延迟更新 $\tilde G$）。落在大批次里的点因此有了一个「人群」可藏，隐私被放大——类比 shuffle 模型的 local-to-central 放大。其妙处在于放大强度随数据「简单程度」自动增强：当某步存在明显的最优专家（gap 非零）时，预测更难被翻转，批次更大、隐私更强。

放大有多强、要付出多少 utility，作者用一条高斯随机游走定理刻画。**Theorem 1** 给出：若起点 $v$ 的 gap 为 $k$，则 $B$ 步随机游走中 leader 发生改变的概率至多
$$2\Phi(-\sqrt2\,\beta)+2\sqrt{\pi}\,\varphi(-\beta)\bigl(\Phi(\beta)-\Phi(-\beta)\bigr),\quad \beta=\frac{k}{\eta\sqrt{2B}}-\sqrt{\log(2n-2)}.$$
**Corollary 1** 的 `ComputeDelay` 子程序用它（配根查找）选出最大批次 $B_t$，使「未来 $t$ 步内换预测」的概率被 $\alpha\sqrt{\log n/(t+B_t)}$ 上界控制；**Corollary 2** 由此证明 RW-AdaBatch 的期望 regret 至多 $(1+\tfrac{\alpha}{2})(\eta+\tfrac{2}{\eta})\sqrt{2T\log n}$——只比 RW-FTPL 多一个可由超参 $\alpha$ 任意调小的乘性因子，几乎零 utility 成本。隐私侧，落在大小为 $B$ 的批次里的点享受 ex-post 的中心化 DP $\mu/\sqrt B$；ex-ante 放大量没有闭式，作者构造一个「随机意义上比真实批次更小」的代理分布来给下界（**Theorem 2**：gap 的随机变量在 $G_{t-1}=0$ 即所有均值相等时随机最小，对应最坏情况），并写出该最坏 gap 的 CDF/PDF：
$$F_k(\varepsilon)=1-n\!\int_{-\infty}^{\infty}\!\varphi(x)\Phi(x-\varepsilon)^{n-1}\,dx,\qquad f_k(\varepsilon)=n(n-1)\!\int_{-\infty}^{\infty}\!\varphi(x)\varphi(x-\varepsilon)\Phi(x-\varepsilon)^{n-2}\,dx.$$
再经 Lemma 1（随机更小的批次分布 ⇒ 更小的 tradeoff 函数）与 Lemma 2（tradeoff 函数的联合凹性，化成高斯混合）即可数值计算放大后的隐私曲线。

**3. RW-Meta：用去相关的共享噪声，从数据依赖专家中零额外隐私成本地私有选择**

动态环境里，光选出最好的 learner 没用，还得真的执行它的建议；若 learner 本身依赖数据而不满足 DP，执行这一步就会泄露隐私。朴素做法是把隐私预算分给 $m$ 个 learner，但这会让每个 learner 的误差随 $\sqrt m$ 增长。RW-Meta 的破解之道是让所有 learner **共享同一份**加噪增益向量 $\tilde g$——既然每个 learner 只读这份已经满足 LDP 的噪声数据，整套流程仍只是 $\tilde g$ 的后处理，由后处理不变性整体满足 $\mu$-GLDP，且每个 learner 的误差关于 $m$ 是常数，远好于朴素基线。

具体地，$\langle x_{t,i},\tilde g_t\rangle$ 是 learner $i$ 在 $t$ 步增益的无偏估计，累计后 $\sum_s X_t\tilde g_t\sim\mathcal N(G^{(m)}_t,\Sigma_t)$，$\Sigma_t=\eta^2\sum_s X_tX_t^\top$——麻烦在协方差矩阵依赖数据。作者引入一个**去相关步**消除它：令
$$\Sigma_t^*=\Sigma_t-\tfrac{1}{m^2}(\mathbf 1^\top\Sigma_t\mathbf 1)\mathbf 1\mathbf 1^\top,$$
每步采样 $y_t\sim\mathcal N\!\bigl(0,\max(2t,\lambda_{\max}(\Sigma^*_{t-1}))I-\Sigma^*_{t-1}\bigr)$ 并选 $j_t=\arg\max_j(\tilde G^{(m)}_{t-1}+y_t)_j$。由于算法对协方差 $\mathbf 1\mathbf 1^\top$ 的可加噪声不变，去相关后可归约到「缩放单位阵协方差」的情形，得到 **Theorem 3**：相对最优 learner 的期望 regret 至多
$$\Bigl[\max\!\bigl(\sqrt2,\ \eta\,\lambda_{\max}(\Sigma_T^*/(\eta^2T))^{1/2}\bigr)+\sqrt2\Bigr]\sqrt{2T\log m}=O\!\Bigl(\tfrac{\sqrt{Tnm\log m}}{\mu}\Bigr).$$
界里的主导项是某个与专家经验协方差相关矩阵的最大特征值，可理解为「专家间独立程度 / 学习任务难度」：当所有 learner 每步都建议不同动作时 $\lambda_{\max}(\Sigma_T^*/(\eta^2T))=1$，得到更紧的 $O(\tfrac1\mu\sqrt{Tn\log m})$；最坏情形是 learner 分成 $m/2$ 大小的团伙。值得注意的是这里的 regret 是相对最优 learner（可能远好于最优静态动作），而 RW-FTPL 恰是 RW-Meta 在 $m=n$、所有 learner 建议互不相同且不随时间变化时的特例。

## 实验关键数据

### 主实验

在美国 DHHS 的 COVID-19 医院周报数据上评测 RW-Meta：每个州预测「哪些医院每周报告的 COVID 患者密度最高」。用 13 个 learner（窗口 8/16/32/64 × 弱/中/强正则的滚动高斯回归，外加基础 RW-FTPL），在新墨西哥 / 宾州 / 加州三州（保留 24 / 146 / 293 家医院、共 148 周）上、四个隐私级别（$\mu=\infty,1,0.5,0.25$）各跑 100 次取平均。对比对象是 SOTA 的中心化 DP 专家算法 Agarwal & Singh (2017)（两种参数：Min Noise 取满足 $\mu$-GCDP 的最小噪声，Min Regret 取优化最坏 regret 界的噪声）。下表为平均增益（gain，越高越好）节选：

| 州 / 隐私 | RW-Meta (LDP) | Agarwal Min Noise (CDP) | Agarwal Min Regret (CDP) | 最优线性模型 |
|-----------|---------------|--------------------------|---------------------------|--------------|
| NM, $\mu=1$ | **36.87** | 22.83 | 13.41 | 40.69 |
| NM, $\mu=0.25$ | **22.46** | 15.58 | 13.38 | 25.85 |
| PA, $\mu=1$ | **40.32** | 23.55 | 15.38 | 44.30 |
| CA, $\mu=1$ | **55.62** | 25.09 | 16.91 | 61.16 |
| CA, $\mu=0.25$ | **33.34** | 17.93 | 16.84 | 38.43 |

RW-Meta 在所有设置下都优于其余算法：相对最强的中心化 DP 基线（Min Noise）领先约 1.5–3 倍（如 CA $\mu=1$：$55.62/25.09\approx2.2\times$，NM $\mu=1$：$36.87/22.83\approx1.6\times$），且本身只用更弱的 LDP（无需可信 curator）。

### 消融 / 分析

| 配置 | 关键现象 | 说明 |
|------|----------|------|
| RW-Meta vs 数据无关专家 | 在低/中隐私下取得**负静态 regret** | 数据无关专家无论噪声多小都几乎做不到 |
| RW-Meta vs 最优线性模型 | 平均约 **90%** | 但「哪个线性模型最优」随州/隐私级别剧烈变化 |
| 最优 learner 的漂移 | NM 偏好窗口 64，PA 高隐私下偏好窗口 8+强正则 | 印证 meta-learning 价值：无需事先押注单一超参 |
| RW-AdaBatch 隐私放大 | 解析界对 Monte Carlo 仿真「合理但非完美」紧 | 松弛主要在中等 FPR 区间，低 FPR/小 $\delta$ 处很紧 |

### 关键发现
- **数据依赖专家是胜负手**：RW-Meta 能取得负静态 regret，意味着它超过了任何单一固定动作；这正是「专家本身会随数据学习」带来的，数据无关专家从原理上做不到。
- **LDP 反而赢过 CDP**：在低维区间（$T\gtrsim n/\varepsilon^2$）里，靠「动态环境 + 零额外隐私成本的专家选择」带来的收益，盖过了 LDP 比 CDP 多付的噪声代价。
- **RW-AdaBatch 的放大是「免费」的**：utility 只多一个可调小的 $1+\alpha/2$ 因子（实验用 $\alpha=0.01$），而隐私在「简单数据」上自动变强，低 FPR 区间界很紧。

## 亮点与洞察
- **把 utility 性质回收成隐私资源**：RW-FTPL 当年「少换手」是为了低 regret，本文却把同一性质读成「排列不变 ⇒ 可放大隐私 / 可共享噪声」，一鱼两吃，思路非常漂亮。
- **共享噪声 = 零成本多专家选择**：让所有 learner 复用同一份加噪增益，靠 DP 后处理不变性绕开「分预算导致误差 $\sqrt m$ 增长」的两难，是把 DP 性质用到极致的范例。
- **隐私放大随数据难度自适应**：「数据越简单（最优专家越明显）→ 隐私越强」这一点和 shuffle 模型精神一致，但发生在 online、本地、数据依赖的设定下，并配了可数值计算的紧界。
- **去相关技巧可迁移**：用 $\Sigma^*=\Sigma-\tfrac1{m^2}(\mathbf 1^\top\Sigma\mathbf 1)\mathbf 1\mathbf 1^\top$ 把数据依赖协方差归约到缩放单位阵，这种「利用算法对某子空间噪声的不变性来简化分析」的手法，在其他带结构噪声的私有在线算法里也可能用得上。

## 局限与展望
- **utility 分析限定 oblivious 对手**：regret 界假设增益向量可任意但不依赖所加噪声的实现值（LDP 保证对自适应对手仍成立，但 utility 不一定）。
- **RW-AdaBatch 的 ex-ante 放大无闭式**：只能数值计算，且解析界在中等 FPR 区间偏松（作者归因于对 gap 收缩速率的保守估计），对很大批次的概率有低估。
- **RW-Meta 依赖 learner 多样性**：界由专家经验协方差的最大特征值主导，若 learner 高度相关（分成大团伙）则收益打折；实验也显示「最优 learner」高度不稳定，需要预先准备足够多样的候选。
- **评测任务较单一**：只在 COVID-19 医院密度预测一个真实任务上验证 RW-Meta；与中心化 DP 的对比作者也强调是「刻画 local vs central 代价」而非证明谁绝对更优。可进一步在更多隐私关键任务、以及与同期 Saha et al. (2025) 的动态环境算法上做对比。

## 相关工作与启发
- **vs RW-FTPL (Devroye et al., 2013)**：本文的基础。RW-FTPL 为低 regret + 少换手设计、未考虑隐私；本文指出它天然 $\mu$-GLDP，并把「少换手」升级为分批放大（RW-AdaBatch）与共享噪声选专家（RW-Meta），RW-FTPL 是 RW-Meta 的特例。
- **vs Agarwal & Singh (2017)**：SOTA 的中心化 DP 专家算法，是本文主要实验基线。它需要可信 curator、面向静态环境；RW-Meta 用更弱的 LDP + 数据依赖专家，在低维区间反而领先 1.5–3 倍。
- **vs Asi et al. (2023)**：同样利用经典在线算法的有限换手来构造私有专家算法，但聚焦高维区间（local 与 central DP 差距最大处）且是中心化模型；RW-AdaBatch 属于这条「隐私 ↔ 有限换手」研究线的本地化贡献。
- **vs Saha et al. (2025)**（同期）：同样为动态环境设计私有专家算法，但通过构造指数级候选专家集喂给 Asi et al. (2023) 的高维 CDP 算法，运行时随 $T$ 指数增长；RW-Meta 对一小组精心选择的候选定义 regret，运行时随 $T$ 线性，是定性不同的 regret 概念。
- **vs Gao et al. (2024)**：目前已知唯一的 LDP 专家算法，但面向联邦上下文与随机对手，技术结果与本文差异显著。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统研究 LDP 下专家建议预测，「有限换手→隐私放大」「共享噪声→零成本多专家」两个角度都很原创。
- 实验充分度: ⭐⭐⭐⭐ 真实 COVID 数据 + 多州多隐私级别 + 仿真验证放大界，但真实任务只有一个、缺与同期动态算法的直接对比。
- 写作质量: ⭐⭐⭐⭐ 理论脉络清晰、定理与算法伪代码完整；公式密集，部分放大界依赖附录证明。
- 价值: ⭐⭐⭐⭐⭐ 为高敏感场景（医疗/公共卫生）提供了无需可信 curator 的实用在线学习算法，且证明 LDP 在合适区间能反超 CDP。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Asymptotic Optimality of the High-Dimensional Gaussian Mechanism and Improved Low-Dimensional Mechanisms for Differential Privacy](../../ICML2026/learning_theory/asymptotic_optimality_of_the_high-dimensional_gaussian_mechanism_and_improved_lo.md)
- [\[ICLR 2026\] Why We Need New Benchmarks for Local Intrinsic Dimension Estimation](why_we_need_new_benchmarks_for_local_intrinsic_dimension_estimation.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)
- [\[ICLR 2026\] Distribution-informed Online Conformal Prediction](distribution-informed_online_conformal_prediction.md)

</div>

<!-- RELATED:END -->
