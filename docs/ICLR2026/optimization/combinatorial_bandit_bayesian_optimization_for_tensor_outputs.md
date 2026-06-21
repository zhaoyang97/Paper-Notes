---
title: >-
  [论文解读] Combinatorial Bandit Bayesian Optimization for Tensor Outputs
description: >-
  [ICLR2026][优化/理论][张量输出] 针对"输出是一整块多模态张量"的昂贵黑箱系统，本文提出张量输出贝叶斯优化框架 TOBO（用一种能捕捉跨模态相关性的张量高斯过程 TOGP 当代理模型 + UCB 采样），并进一步把它扩展到"只有张量中一部分元素计入目标"的组合老虎机贝叶斯优化（CBBO）设定，设计 CMAB-UCB2 准则联合选输入点和子集，两者都给出了 $\tilde{O}(\sqrt{T})$ 的次线性 regret 证明。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "张量输出"
  - "贝叶斯优化"
  - "组合老虎机"
  - "张量高斯过程"
  - "regret 界"
---

# Combinatorial Bandit Bayesian Optimization for Tensor Outputs

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=BgX4s1d2d0](https://openreview.net/forum?id=BgX4s1d2d0)  
**代码**: 待确认  
**领域**: 贝叶斯优化 / 组合多臂老虎机 / 高斯过程 / 张量学习 / 学习理论  
**关键词**: 张量输出、贝叶斯优化、组合老虎机、张量高斯过程、regret 界

## 一句话总结
针对"输出是一整块多模态张量"的昂贵黑箱系统，本文提出张量输出贝叶斯优化框架 TOBO（用一种能捕捉跨模态相关性的张量高斯过程 TOGP 当代理模型 + UCB 采样），并进一步把它扩展到"只有张量中一部分元素计入目标"的组合老虎机贝叶斯优化（CBBO）设定，设计 CMAB-UCB2 准则联合选输入点和子集，两者都给出了 $\tilde{O}(\sqrt{T})$ 的次线性 regret 证明。

## 研究背景与动机

**领域现状**：贝叶斯优化（Bayesian Optimization, BO）是优化"昂贵、黑箱、求一次值代价很高"目标函数的主流手段，广泛用于超参搜索、实验设计、机器人控制。它的标准套路是：用已观测数据拟合一个高斯过程（GP）代理模型，再优化一个采集函数（acquisition function，如 UCB）在"探索"和"利用"间权衡，逐步挑下一个查询点。绝大多数 BO 处理标量输出，近年也被扩展到多输出（multi-output BO, MOBO）。

**现有痛点**：但据作者所知，没有任何工作处理过**张量值输出**——即系统对一个输入 $x$ 的响应是一整块 $m$ 模态张量 $f(x)\in\mathbb{R}^{t_1\times\cdots\times t_m}$。最直接的办法是把张量拉直（vectorize）成向量再套 MOBO，但拉直会**抹掉张量天然的跨模态结构相关性**（mode-wise dependency），而这种相关在时空过程等复杂系统里恰恰至关重要。

**核心矛盾**：已有的张量输出 GP（HOGP）为了算得动，普遍采用**可分离（separable）协方差**——联合协方差被写成输入核与各模态核的 Kronecker 乘积。可分离的代价是：它假设"跨模态的相关结构不随输入变化"，这个假设在复杂系统里经常被违背，导致预测精度差、后验推断数值不稳，最终 BO 表现退化。

**本文目标**：(1) 造一个既灵活、又可扩展、还能捕捉"输入相关（input-dependent）的张量内相关"的 GP 代理；(2) 为张量输出 BO 设计采集函数和序贯查询策略并给出 regret 保证；(3) 进一步处理更实际的设定——只有张量里 $k<T$ 个元素真正计入目标，需要联合选"查询点 + 元素子集"。

**切入角度**：把"线性共区域化模型"（Linear Model of Coregionalization, LMC）从向量值输出**推广到张量值输出**，用一组基核的混合让跨元素相关性可以随输入变化，从而摆脱可分离假设；把"只选部分元素"自然建模成组合多臂老虎机（CMAB）——每个张量元素是一个 base arm，每轮选一个大小为 $k$ 的 super-arm。

**核心 idea**：用"LMC 张量核 + UCB"做张量输出 BO（TOBO），再把"GP 代理 + 组合老虎机 UCB"耦合成 CMAB-UCB2 解决子集选择（TOCBBO），并对两者都证出次线性 regret。

## 方法详解

### 整体框架

把问题分两层。第一层是 **TOBO（张量输出 BO）**：目标函数 $f:\mathcal{X}\to\mathcal{Y}$ 输入 $x\in\mathcal{X}\subset\mathbb{R}^d$、输出是 $m$ 模态张量。因为没法直接最大化一整块张量，先用一个有界线性算子 $\mathcal{L}_f$ 把张量标量化，求

$$x^\star=\arg\max_{x\in\mathcal{X}}\mathcal{L}_f f(x).$$

TOBO 的循环是经典 BO 三步：用张量输出高斯过程（TOGP）拟合 $f$ → 算后验均值/方差 → 用 UCB 采集函数挑下一个 $x_{n+1}$ → 观测带噪输出 $y_i=f(x_i)+\varepsilon_i$ → 回到第一步。

第二层是 **TOCBBO（组合老虎机 BO）**：现在只有张量里 $k$ 个元素计入目标，用二值指示向量 $\lambda\in\{0,1\}^T,\ \mathbf{1}^\top\lambda=k$ 表示选哪些（等价于选一个大小 $k$ 的 super-arm $S$），目标变成联合优化

$$(x^\star,\lambda^\star)=\arg\max_{x\in\mathcal{X},\lambda\in\Lambda}\mathcal{H}_f\,\tilde f(x,\lambda),\quad \tilde f(x,\lambda)=e(\lambda)\,\mathrm{vec}(f(x)).$$

直接在 $x$ 和 $\lambda$ 上联合搜索不可行（从 $T$ 个元素里选 $k$ 个有 $\binom{T}{k}$ 种），所以用 CMAB-UCB2 把它**解耦成两步**：先固定当前最优 super-arm 选 $x$，再固定 $x$ 用组合老虎机 UCB 选 $\lambda$。下面四个关键设计依次撑起这两层。

### 关键设计

**1. LMC 张量输出核：让跨元素相关随输入变化，摆脱可分离假设**

这是全文地基，针对的就是"可分离协方差假设跨模态相关不随输入变"这个痛点。作者把向量值的 LMC 推广到张量：令 $\mathrm{vec}(f(x))=\sum_{\ell=1}^{m}\sum_{j=1}^{r_\ell} a_{\ell j}\,u_{\ell j}(x)$，其中 loading 向量 $a_{\ell j}=\mathrm{vec}(A_{\ell j})$ 由张量 core $A_{\ell j}\in\mathbb{R}^{t_1\times\cdots\times t_m}$ 参数化，$\{u_{\ell j}\}$ 是一组独立的标量 GP、协方差为基核 $k_{\ell j}(x,x')$。由此得到**非可分离核**（Definition 1）：

$$K(x,x')=\sum_{l=1}^{m}\sum_{j=1}^{t_l}\mathrm{vec}(A_l)\,\mathrm{vec}(A_l)^\top k_{lj}(x,x').$$

关键在于：跨元素相关由一组基核 $\{k_{lj}\}$ 的**混合**支配，所以相关结构能随输入 $x$ 变化——这正是可分离核做不到的。当所有分量共享同一个基核、卷积退化时，它就坍缩回**可分离核**（Definition 2）$K(x,x')=\mathrm{vec}(A)\,\mathrm{vec}(A)^\top k(x,x')$。作者证明（Proposition 1）两类核都对称半正定，因此 TOGP 先验良定义。这套构造比已有张量核更一般：已有张量核可被看成"某种低秩张量分解诱导的特例"，而本文用共区域化结构能编码更广的张量约束。为控参数量，core 张量可再加 CP / Tensor-Train 低秩分解；为应对大 $nT$，用 Nyström 低秩近似把 $(K_n+\eta I_{nT})^{-1}$ 的求逆降到 $O(nTn_\ell^2+n_\ell^3)$。

**2. TOGP 代理 + UCB 采集：把张量后验不确定性接进序贯查询**

有了核就能写出 TOGP 后验。给 $n$ 个观测，新点 $x$ 处 $\mathrm{vec}(f(x))$ 的后验仍是高斯，均值

$$\hat\mu_n(x)=\mu+K_n^\top(x)(K_n+\eta I_{nT})^{-1}(Y_n-\mathbf{1}_n\otimes\mu),$$

协方差 $\hat K_n(x,x)=\sigma^2[K(x,x)-K_n^\top(x)(K_n+\eta I_{nT})^{-1}K_n(x)]$。超参用极大似然估计。采集函数把标量化目标 $\mathcal{L}_f$ 套到后验上：

$$\alpha_{\text{UCB}}(x\mid\mathcal{D}_n)=\mathcal{L}_f\hat\mu_n(x)+\beta_n\,\hat K_n(x,x)^{1/2},$$

下一个查询点 $x_{n+1}=\arg\max_x\alpha_{\text{UCB}}$。这一步的价值在于：$\hat K_n(x,x)$ 是**张量后验的整体不确定性**，UCB 因此会朝"张量预测最不确定的方向"探索，而不是逐元素独立地看不确定性——这正好把设计 1 捕捉到的跨元素相关用上了。

**3. 部分观测 TOGP（PTOGP）：让 GP 能在"只看到 k 个元素"时仍做后验**

进入 CBBO 后，每轮只观测 super-arm $S_i$ 对应的 $k$ 个元素，其余未观测。直接用 TOGP 不行，因为它假设整块张量都看得到。作者用二值选择矩阵 $e(\lambda)\in\{0,1\}^{k\times T}$ 把 TOGP 投影到被选子集上，得到部分观测 TOGP：先验 $\tilde f(x,\lambda)\sim\mathcal{PTOGP}(e(\lambda)\mu,\ \tau^2 e(\lambda)K(x,x')e(\lambda')^\top)$。对新 pair $(x,\lambda)$ 的后验仍是 $\mathbb{R}^k$ 上高斯，均值

$$\tilde\mu_n(x,\lambda)=e(\lambda)\mu+\sigma^2 e(\lambda)K_n^\top(x)E_n^\top\tilde\Sigma_n^{-1}(\mathrm{vec}(\tilde Y_n)-E_n(\mathbf 1_n\otimes\mu)),$$

其中 $E_n$ 是按各轮 super-arm 拼出的块对角选择矩阵、$\tilde\Sigma_n=\sigma^2 E_nK_nE_n^\top+\tau^2 I_{nk}$。关键意义：因为协方差核（设计 1）本身编码了跨元素相关，**已观测的 $k$ 个元素能反过来推断未观测元素的后验**，这让"在没全看到的情况下评估任意候选 super-arm"成为可能——CMAB-UCB2 第二步正靠它。

**4. CMAB-UCB2：把"联合选输入+子集"解耦成两步序贯 UCB**

直接在连续 $x$ 和组合 $\lambda$ 上联合优化是 $\binom{T}{k}$ 级组合爆炸。CMAB-UCB2 把它拆开：第 $n+1$ 轮，先取目前最优 pair $(x_n^\star,\lambda_n^\star)$，**固定 super-arm 为 $\lambda_n^\star$ 选输入**

$$x_{n+1}=\arg\max_{x\in\mathcal{X}}\mathcal{H}_f\tilde\mu_n(x,\lambda_n^\star)+\tilde\beta_n\|\tilde K_n(x,x;\lambda_n^\star,\lambda_n^\star)\|^{1/2};$$

再**固定 $x_{n+1}$ 选 super-arm**，给每个候选 super-arm 建 UCB 并取最大

$$\lambda_{n+1}=\arg\max_{\lambda\in\Lambda}\mathcal{H}_f\tilde\mu_n(x_{n+1},\lambda)+\tilde\rho_n\|\tilde K_n(x_{n+1},x_{n+1};\lambda,\lambda)\|^{1/2}.$$

两步都是 UCB 形式，但作用对象不同：第一步是连续输入空间的 BO，第二步是离散子集空间的组合老虎机。这种解耦把不可解的联合搜索降成两个可解子问题，且和已有 BBO（如 CoCaBO 把每个类别变量当独立老虎机）本质不同——本文是**联合选多个可能相关的 arm**组成 super-arm，而非逐变量独立选。

### 损失函数 / 训练策略

没有可训练网络，"训练"指 TOGP/PTOGP 的超参 $\Theta=\{\theta,a,\sigma^2,\tau^2\}$ 用极大似然估计；core 张量的秩用交叉验证按 MAE 选（$r^\star=\arg\min_r \mathrm{MAE}(r)$）。理论侧给出两条 regret 保证：在 $f$ 服从 TOGP 先验、标量化算子 Lipschitz 的假设下，**TOBO**（Theorem 1）以高概率满足

$$R_N\le L\sqrt{C_1 N\gamma_N(K,\eta)}\,\beta_N+\tfrac{\pi^2}{6},$$

其中 $\gamma_N$ 是最大信息增益；**TOCBBO**（Theorem 2）的 regret 多出 super-arm 选择那一项 $\sqrt{C_4 TN\tilde\gamma_N}\,\tilde\rho_N$。结合可分离核下 $\gamma_n=O(T(\log n)^{d+1})$（高斯核），两者都达到 $O(\sqrt{T\log T})$ 的次线性 regret——这是首个针对张量值输出的贝叶斯 regret 分析。

## 实验关键数据

评测分合成数据和真实案例两块，统一对比把张量拉直 + MOGP 当代理的基线：sMTGP、MLGP、MVGP，每个再配 UCB / 随机两种采样；并用"TOGP + 随机采样"（TOGP-RS）做消融。

### 主实验：合成数据预测与优化

三种合成设置（不同模态数 $m$、不同张量尺寸），CBBO 任务取 $k=T/6$。先看代理模型的预测精度（NLL / MAE，越低越好）：

| 设置 | 指标 | TOGP（本文） | sMTGP | MLGP | MVGP |
|------|------|------|------|------|------|
| (1) | NLL | **503.0** | 749.4 | 707937.1 | 11152.2 |
| (1) | MAE | **0.1571** | 0.1684 | 0.9428 | 0.6746 |
| (2) | NLL | **-18.1** | -5.0 | 7066.9 | 46.54 |
| (3) | MAE | **0.1372** | 0.1501 | 1.1670 | 1.0000 |

TOGP 在三种设置都拿到最低 NLL / MAE。MLGP 最差（其拟合协方差在该配置下退化为奇异）；sMTGP 因为还建模了张量各模态、好于完全忽略张量结构的 MVGP。

再看优化性能（$\text{MSE}_x$ / $\text{MAE}_y$ 越低越好，Acc 是 CBBO 选对子集的准确率）：

| 任务 | 方法 | 设置(1) MSE$_x$ | 设置(1) Acc | 设置(3) MSE$_x$ | 设置(3) Acc |
|------|------|------|------|------|------|
| BO | TOBO | **0.0000** | – | **0.0001** | – |
| BO | sMTGP-UCB | 0.0001 | – | 0.0048 | – |
| CBBO | TOCBBO | **0.0023** | **1.00** | **0.0021** | **1.00** |
| CBBO | sMTGP-UCB | 0.1832 | 0.67 | 0.0075 | 0.86 |
| CBBO | MVGP-UCB | 0.0032 | 1.00 | 0.0151 | 0.71 |

TOBO / TOCBBO 在三种设置都拿最小 $\text{MSE}_x$、$\text{MAE}_y$，且 TOCBBO 的子集选择准确率在三种设置都达到 1.00。逐轮 log 瞬时 regret 曲线（Figure 1）显示 TOBO / TOCBBO 始终最低，和理论上 UCB 比随机采样更优一致。

### 消融实验

| 配置 | 说明 | 现象 |
|------|------|------|
| TOBO / TOCBBO（Full） | TOGP + UCB | 各设置最优 |
| TOGP-RS | TOGP + 随机采样（去掉 UCB） | $\text{MSE}_x$ 明显变差，如设置(1) BO 从 0.0000 升到 0.0893 |
| sMTGP-UCB / MVGP-UCB | 换掉 TOGP 代理 | 预测与优化均退化，印证 TOGP 核的作用 |

两条消融轴说明：UCB 采集（vs 随机）和 TOGP 代理（vs 拉直 MOGP）各自都有贡献，且二者的优劣排序在预测表和优化表之间一致。

### 关键发现
- **TOGP 代理是性能根因**：去掉它（换 sMTGP/MVGP）后预测和优化同步退化；UCB 与随机采样的差距说明采集函数同样不可省。
- **可分离 vs 非可分离按需选**：作者在设置 (1)(2) 用可分离核、设置 (3) 用非可分离核来平衡灵活性与算力，说明非可分离核虽更强但有计算代价。
- **真实数据同样成立**：在 REEN 数据集上 TOGP 取得最优 NLL（15.67 vs sMTGP 33.32 / MLGP 88.27 / MVGP 48.72）与 MAE（0.0883 最低），四个真实案例（CHEM / MAT / PRINT / REEN）上 TOBO / TOCBBO 的优化曲线都领先。

## 亮点与洞察
- **把 LMC 从向量推广到张量**是最漂亮的一步：用 core 张量 + 基核混合，一举既保留张量结构、又让跨元素相关随输入变化，还能把已有可分离张量核统一成自己的特例。
- **PTOGP 让"部分观测也能推全张量"**：因为核里编码了跨元素相关，已看到的 $k$ 个元素能反推没看到的元素后验——这是组合老虎机那一步能评估任意候选 super-arm 的前提，设计上环环相扣。
- **CMAB-UCB2 的解耦很务实**：把 $\binom{T}{k}\times$ 连续空间的不可解联合优化拆成"先选点、再选子集"两个 UCB 子问题，既绕开组合爆炸又保住了 regret 证明。
- **首个张量输出 BO 的 regret 分析**：理论上把 BO 的 $O(\sqrt{T})$ 经典结论延伸到张量值输出和组合老虎机设定，填了一块此前空白；"先证可分离核下信息增益 $\gamma_n$ 的阶、再代回 regret 模板"的思路可迁移到其他结构化输出。

## 局限与展望
- **可分离 vs 非可分离的取舍靠人工**：非可分离核更灵活但更贵，文中按设置手动选用哪种，缺少自动判定机制。
- **组合规模仍受限**：CMAB-UCB2 第二步要对候选 super-arm 建 UCB，当 $T$、$k$ 很大时枚举/搜索候选集合的代价仍可能成为瓶颈，文中实验 $k=T/6$ 且张量尺寸不大。
- **实验规模偏小**：合成与真实数据的张量维度都不算高（如 REEN 为 $10\times2$），更高阶、更大模态张量上的可扩展性（Nyström 近似之外）有待检验。
- **依赖 GP 先验假设**：regret 保证建立在"$f$ 真服从 TOGP 先验 + 标量化算子 Lipschitz"上，先验失配时的鲁棒性未讨论。

## 相关工作与启发
- **vs 高阶 GP（HOGP, 如 Belyaev 2015 / Kia 2018 / Zhe 2019）**：他们用可分离/Kronecker 结构核，跨模态相关不随输入变；本文用 LMC 张量核做到非可分离、输入相关，且把他们的核统一成低秩分解诱导的特例。
- **vs 多输出 BO（MOBO，如 Chowdhury & Gopalan 2021 / Belakaria 2019）**：MOBO 面向向量值多输出、且基于熵/超体积的方法在目标数增大时计算不可行；本文是首个面向张量值输出并给出 regret 的 BO 框架。
- **vs 老虎机 BO（BBO，如 CoCaBO / Nguyen 2020）**：BBO 把每个类别变量当独立老虎机、用独立 GP 建模，无法捕捉张量内相关、也只逐变量选单 arm；本文的 CBBO 要联合选多个相关 arm 组成 super-arm，BBO 是其特例。
- **启发**：当输出本身有强结构（张量/图/序列）时，与其拉直套通用 multi-output 方法，不如把结构直接编码进核/代理模型；"结构化核 → 信息增益阶 → regret 界"这条链条值得在其他结构化黑箱优化里复用。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个张量输出 BO + 组合老虎机 BO 框架，并给出对应 regret 分析，填补明确空白。
- 实验充分度: ⭐⭐⭐⭐ 合成三设置 + 四个真实案例，预测/优化/消融齐全，但张量规模偏小、可扩展性验证有限。
- 写作质量: ⭐⭐⭐⭐ 理论推导完整、对比表清晰；符号较密集，部分定义在正文重复。
- 价值: ⭐⭐⭐⭐ 为结构化输出黑箱优化提供了可迁移的核构造 + 解耦 + regret 分析范式。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] BoGrape: Bayesian optimization over graphs with shortest-path encoded](bogrape_bayesian_optimization_over_graphs_with_shortest-path_encoded.md)
- [\[ICLR 2026\] Adaptive Acquisition Selection for Bayesian Optimization with Large Language Models](adaptive_acquisition_selection_for_bayesian_optimization_with_large_language_mod.md)
- [\[ICLR 2026\] Symmetry-Aware Bayesian Optimization via Max Kernels](symmetry-aware_bayesian_optimization_via_max_kernels.md)
- [\[ICLR 2026\] Contextual Causal Bayesian Optimisation](contextual_causal_bayesian_optimisation.md)
- [\[ICLR 2026\] NExCO: Native Solution Expansion for Diffusion-based Combinatorial Optimization](nexco_native_solution_expansion_for_diffusion-based_combinatorial_optimization.md)

</div>

<!-- RELATED:END -->
