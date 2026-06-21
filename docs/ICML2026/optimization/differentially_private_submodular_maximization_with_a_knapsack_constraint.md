---
title: >-
  [论文解读] Differentially Private Submodular Maximization with a Knapsack Constraint
description: >-
  [ICML 2026][优化/理论][子模最大化] 本文给出背包约束下子模最大化（SMK）的差分隐私算法：单调目标下达到最优的 $(1-1/e)$ 近似且把附加误差从「多项式依赖 $n$」改进到「polylog 依赖 $n$」、查询复杂度从指数级降到多项式级，并首次为非单调目标给出有可证保证（$1/4$ 近似）的差分隐私算法。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "子模最大化"
  - "背包约束"
  - "差分隐私"
  - "近似算法"
  - "组合优化"
---

# Differentially Private Submodular Maximization with a Knapsack Constraint

**会议**: ICML 2026  
**arXiv**: [2606.14951](https://arxiv.org/abs/2606.14951)  
**代码**: 待确认  
**领域**: 优化 / 差分隐私  
**关键词**: 子模最大化, 背包约束, 差分隐私, 近似算法, 组合优化

## 一句话总结
本文给出背包约束下子模最大化（SMK）的差分隐私算法：单调目标下达到最优的 $(1-1/e)$ 近似且把附加误差从「多项式依赖 $n$」改进到「polylog 依赖 $n$」、查询复杂度从指数级降到多项式级，并首次为非单调目标给出有可证保证（$1/4$ 近似）的差分隐私算法。

## 研究背景与动机

**领域现状**：子模最大化刻画「边际收益递减」，是离散优化的基石，广泛用于特征选择、推荐、影响力最大化、样本选择、可解释规则学习等机器学习任务。其中一个最经典的变体是**背包约束子模最大化（SMK）**：每个元素 $v$ 有代价 $c(v)$，要在预算 $\sum_{v\in S}c(v)\le B$ 下最大化子模函数 $f(S)$。当这些任务涉及敏感个人数据（如医疗特征选择），就需要在保证强效用的同时给出严格的**差分隐私（DP）**保证。

**现有痛点**：DP 子模最大化此前主要研究**基数约束**（cardinality）和**拟阵约束**（matroid），背包约束据作者所知只有 Sadeghi & Fazel (2021) 一篇处理过，且存在三个明显短板：① 它依赖对子模函数**多线性扩展的精确求值**，在 value-oracle 模型下最坏要 $2^{|N|}$ 次查询（指数级）；② 只支持**单调**函数，把一大类带多样性/正则惩罚的非单调应用排除在外；③ 其效用保证里的附加误差对**地面集规模 $n$ 是多项式依赖**、且随**最小元素代价 $c_{\min}$** 缩放（$\Delta/(c_{\min}\varepsilon)$），都偏松。

**核心矛盾**：把组合型 SMK 算法私有化，会撞上两个基数/拟阵约束里不存在的难点——其一，背包下贪心选的是**密度分数** $f(u\mid S)/c(u)$，不同元素代价不同导致**分数敏感度差异很大**，直接套指数机制会让误差正比于最坏敏感度 $\Delta\cdot B/c_{\min}\ge \Delta k$，凭空多出一阶 $k$ 依赖；其二，达到最优近似的 Two-Guess-Greedy 需要 $O(n^2)$ 次枚举迭代，用标准（advanced）组合定理摊隐私预算会让误差随 $n$ 多项式增长，够不到 SOTA 的 polylog 依赖。

**本文目标**：在单调与非单调 SMK 上，同时改进**近似比、附加误差、查询复杂度**三项，弥合与基数/拟阵约束设置的差距。

**核心 idea**：用「细粒度敏感度的私有选择机制 + 把并行枚举结构当作 DP 子程序集合做选择」两件武器，分别破解上述两个难点——前者让误差只依赖 $\Delta f$ 而非最坏密度敏感度，后者绕开标准组合惩罚换来 polylog 依赖。

## 方法详解

### 整体框架
论文针对三种设置给出三个算法，共享同一套私有化思路。**单调**情形以 Feldman 等 (2023) 的 **Two-Guess-Greedy**（先枚举猜出最优解中的两个元素、再在剩余实例上跑密度贪心，是目前最实用的紧近似算法）为骨架做私有化，得到最优 $(1-1/e)$ 近似（Algorithm 2）；另给一个只跑单次密度贪心变体的更快算法，换来 $1/2$ 近似（Algorithm 7）。**非单调**情形沿用「密度贪心 + 以固定概率随机丢弃已选元素」的范式（Han 等 2021）做私有化，给出首个有保证的 DP 算法、$1/4$ 期望近似（Algorithm 3）。三者都把贪心的「选密度最大元素」这一步替换成带噪的私有选择，并通过两项技术控制误差与隐私预算。

理解骨架要先理解非私有的 Two-Guess-Greedy 为何能达到 $(1-1/e)$：单纯按密度贪心在背包下没有常数近似保证（反例容易构造），Sviridenko (2004) 通过「先猜出 OPT 里的三个元素、把它们强制放进解、再在残余实例上跑密度贪心」修复了这一点，Feldman 等把「猜三个」降到「猜两个」，于是要枚举所有 $O(n^2)$ 个元素对、对每对各跑一遍密度贪心、最后取最好的——正是这个「枚举对 + 取最优」的并行结构，被本文重新利用成隐私上的优势。私有化时，每个「(枚举对, 密度贪心)」执行体被看作一个 DP 子程序，贪心内部每步的元素选择则换成带噪私有选择。

### 关键设计

**1. 用 GRNM 处理「密度分数敏感度不齐」，让误差只依赖 $\Delta f$**

背包贪心每步要选密度 $f(u\mid S)/c(u)$ 最大的元素，但元素代价 $c(u)$ 各异，使得不同候选的密度分数敏感度差别极大。标准的指数机制 / Report-Noisy-Max（RNM）误差正比于**全体候选的最大敏感度** $\Delta=\max_u\Delta_{q(u;\cdot)}$，在这里就是 $\Delta f\cdot B/c_{\min}$；由于 $B/c_{\min}\ge k$（$k$ 为可行解最大基数），这会平白引入一阶 $k$。作者改用 Raskhodnikova & Smith (2016) 的**广义私有选择 GRNM**：它对「精心构造的辅助分数」施加 RNM，**按每个候选自身的敏感度**给出细粒度效用界——以 $1-\beta$ 概率，输出 $u^*$ 满足 $q(u^*)\ge \max_u\{q(u)-(2\Delta_{q(u;\cdot)}/\varepsilon)\log(|C|/\beta)\}$，从而对高敏感度候选施加适当惩罚、而不让所有候选都背最坏敏感度的锅。最终附加误差正比于 $\Delta f$（而非 $\Delta f\cdot B/c_{\min}$），与基数约束设置持平。

**2. 把 Two-Guess-Greedy 的并行枚举当作「DP 子程序集合」做选择，把组合惩罚降到 polylog**

Two-Guess-Greedy 的 $O(n^2)$ 次枚举步是隐私上的「组合瓶颈」：每一步都消耗隐私预算，用 advanced composition 摊下来误差随 $n$ 多项式增长。作者的关键观察是——这 $O(n^2)$ 个枚举步彼此**高度并行**，天然契合「从一组 DP 子程序里选最好输出」的框架。每个子程序是密度贪心的一次私有化执行。基于 Cohen 等 (2023) 的**广义选择框架**（其隐私损失**与子程序总数无关**），作者**随机化每个枚举步被调用的次数**，从而绕过标准组合惩罚，把附加误差对 $n$ 的依赖压到 **polylogarithmic**。这是相对 Sadeghi & Fazel 多项式依赖的根本性改进，也是单调结果（Theorem 4.1）成立的核心。

**3. 非单调情形「不因负边际收益停机 + 用 RNM 取最优观测解」，并控制选择次数**

非单调 SMK 的私有化额外有三个障碍，作者逐一化解。① **选择次数不定带来的组合瓶颈**：因为已选元素会被随机丢弃，算法可能做多达 $n$ 次选择，标准组合会让误差随 $\sqrt{n}$ 涨；作者证明所需私有选择次数被一族**独立几何随机变量之和随机控制**、集中在 $O(k)$ 附近，从而以高概率不超隐私预算。② **对负边际收益的鲁棒性**：非私有分析（Han 等 2021）依赖只选「严格正边际增益」的元素来保证进展，但 DP 加噪会让负边际贡献的元素被误选；作者改为**不因「无正增益元素」停机、而是继续加入可行元素**，最后用 RNM 选出**观测到的近似最优解**，把分析聚焦到「仍存在足够大增益的早期迭代」，通过给 Han 等的分析加一个有界 slack，证明早期高概率选到大贡献元素，从而即便噪声掩盖了边际符号也能拿到 $1/4$ 近似。③ **密度分数敏感度不齐**：同单调情形，继续用 GRNM 解决。

### 一个例子：隐私预算下的医疗特征选择
以论文开篇的 Example 1.1 走一遍能让抽象机制落地。给定一个敏感医疗数据集 $\{(x_i,y_i)\}_{i=1}^n$，每个个体有特征向量 $x_i$ 与是否患病的标签 $y_i$，每个特征 $j$（如某项血检）有采集代价 $c_j$，要在预算 $\sum_{j\in S}c_j\le B$ 下选一组特征 $S$ 使预测尽量准。把目标建成「所选特征与标签之间互信息」这类单调子模函数 $f_D$，它依赖敏感数据 $D$、敏感度为 $\Delta f$。直接跑非私有 Two-Guess-Greedy 会泄露个体信息：某个人是否患病可能改变某次贪心选择，从而被反推。本文算法的私有化做法是：枚举所有特征对作为「猜测」、对每对在残余特征上跑密度贪心，每步不再选「密度严格最大」的血检项，而是用 GRNM 在带噪分数下挑选——便宜但敏感度高的特征会被适当压一压，避免某个高敏感度特征因一两个人的数据而被确定性选入；最后用 Cohen 框架从所有枚举分支里私有地选出整体最好的特征集。这样既满足 $(\varepsilon,\delta)$-DP，又把附加误差控制在 $\tilde{O}(\Delta f\cdot k^{1.5}/\varepsilon)$ 量级、与基数约束设置相当。若把目标换成带多样性惩罚的版本（鼓励选互不冗余的特征），$f_D$ 就变成非单调，此时改用 Algorithm 3：容忍噪声偶尔选入负增益特征、最后用 RNM 兜底取最优观测解，仍拿到 $1/4$ 期望近似。

### 损失函数 / 训练策略
本文是纯理论工作，无训练。隐私分析建立在标准工具上：差分隐私定义 $\Pr[A(D)\in S]\le e^\varepsilon\Pr[A(D')\in S]+\delta$；Laplace 机制（返回 $q(D)+\text{Lap}(\Delta q/\varepsilon)$ 为 $\varepsilon$-DP）；基础/高级组合定理（$k$ 折 $\varepsilon_0$-DP 组合后为 $k\varepsilon_0$-DP 或 $(\sqrt{2k\log(1/\delta)}\varepsilon_0+k\varepsilon_0(e^{\varepsilon_0}-1),\delta)$-DP）；以及指数噪声的 Report-Noisy-Max（RNM）与其广义版 GRNM、Cohen 等的广义选择框架。子模敏感度定义为 $\Delta f=\max_{S\subseteq N, D\sim D'}|f_D(S)-f_{D'}(S)|$。

## 实验关键数据

> 纯理论论文，「实验」即理论保证的对比。下表汇总三个主结果与此前唯一的背包 DP 工作（$n$=地面集规模，$k$=可行解最大基数，$L=\max_u f(u)$，$\beta$=失败概率，预算归一 $B=1$ 故 $1/c_{\min}\ge k$）。

### 主结果对比（$(\varepsilon,\delta)$-DP，$\Delta$-敏感目标）

| 算法 | 单调? | 近似比 | 附加误差（量级） | 查询复杂度 |
|------|-------|--------|------------------|------------|
| Algorithm 2（本文 i） | 是 | $1-1/e$（最优） | $\frac{\Delta k^{1.5}}{\varepsilon}\sqrt{\log\frac{n}{\delta\beta}\log\frac{n}{\beta}}$ | $O(\beta^{-1}n^3 k)$ |
| Algorithm 7（本文 ii） | 是 | $1/2$ | $\frac{\Delta k^{1.5}}{\varepsilon}\sqrt{\log\frac1\delta\log\frac n\beta}$ | $O(nk)$ |
| Algorithm 3（本文 iii） | 否 | $1/4$（期望） | $\frac{\Delta k}{\varepsilon}\sqrt{(k+\log\frac1\delta)\log\frac1\delta\log\frac n\beta}$ | $O(nk)$ |
| Sadeghi & Fazel 2021 | 是 | $1-1/e$ | $\frac{\Delta}{c_{\min}\varepsilon}\sqrt{\frac1\beta\log\frac1\delta\, n\log\frac n\beta}+\frac{\beta L}{c_{\min}^2}$ | $O(2^n)$ |

### 关键改进对照

| 维度 | 本文（单调 Alg.2） | Sadeghi & Fazel 2021 | 改进点 |
|------|--------------------|----------------------|--------|
| 对 $n$ 的误差依赖 | polylog($n$) | 多项式（$\sqrt{n}$ 级） | 靠 Cohen 选择框架绕开组合惩罚 |
| 误差与代价的关系 | 随可行解最大尺寸 $k$ 缩放 | 随 $1/c_{\min}$ 缩放（更松） | 靠 GRNM 细粒度敏感度 |
| 查询复杂度 | 多项式 $O(\beta^{-1}n^3k)$ | 指数 $O(2^n)$（需精确多线性扩展） | 改用组合型贪心骨架 |
| 适用范围 | 单调 + 非单调 | 仅单调 | 首个非单调 DP-SMK |

### 关键发现
- **下界视角**：基数约束下 $(\varepsilon,\delta)$-DP 的期望附加误差下界为 $\Omega(kc\log(\varepsilon/\delta)/\varepsilon)$（基数是背包的特例，故对本文设置也成立）；本文误差比它多出 $\tilde{O}(\sqrt{k})$，但下界是对 1-decomposable 目标推的，而本文处理更一般的有界敏感度子模类，且对 $k$ 的依赖已与基数约束下多项式时间算法的最优结果持平。
- **近似比已触最优/SOTA**：单调 $(1-1/e)$ 是信息论最优；非单调 $1/4$ 匹配现有最实用的组合型非私有保证。
- **三难点各有专属武器**：敏感度不齐→GRNM；组合瓶颈→随机化枚举次数 + Cohen 选择框架；负边际收益→不停机 + RNM 取最优观测解。
- **效用-效率可调**：同为单调目标，Algorithm 2 用 $O(\beta^{-1}n^3k)$ 查询换最优 $(1-1/e)$，Algorithm 7 用 $O(nk)$ 查询换 $1/2$，给了大规模实例一个「掉一档近似比、省两阶查询」的折中选项。

## 亮点与洞察
- **「并行枚举即子程序集合」是最漂亮的一招**：把 Two-Guess-Greedy 的 $O(n^2)$ 枚举重新解读成「从一堆 DP 子程序里做选择」，从而调用 Cohen 等隐私损失与子程序数无关的框架——这把「猜测/枚举型最优近似算法」私有化的通用路径点亮了，可迁移到其他需要枚举 OPT 元素的私有组合优化。
- **细粒度敏感度选择对付「代价异质」很关键**：背包相对基数的本质难点就是密度分数敏感度不齐，用 GRNM 按候选自身敏感度计费、对高敏感度候选惩罚，是把误差从 $\Delta k$ 降回 $\Delta$ 的钥匙。
- **非单调下「容忍负增益、事后用 RNM 兜底」**：与其要求每步严格正增益（DP 噪声下无法保证），不如放任继续选、最后私有地挑最优观测解，把分析重心移到早期迭代——这是处理「噪声掩盖符号」的巧解。

## 局限与展望
- **附加误差仍比下界高 $\tilde{O}(\sqrt{k})$**：单调附加误差含 $k^{1.5}$，与 $\Omega(k)$ 级下界尚有 $\sqrt{k}$ 间隙，是否可消尚未解决。
- **单调最优近似的查询复杂度偏高**：Algorithm 2 为 $O(\beta^{-1}n^3k)$，对大规模实例计算代价不小（论文也因此补了更快的 $1/2$ 近似 $O(nk)$ 版作折中）。
- **纯理论、无实证**：未在 Example 1.1 那类医疗特征选择等真实任务上给出经验效用/运行时间，实际表现待验证。
- **依赖有界敏感度假设**：采用 bounded-sensitivity $\Delta$ 子模类，对敏感度无界或难界定的目标不直接适用。
- **隐私模型为中心化 DP**：分析建立在可信服务器持有原始数据 $D$ 的中心化差分隐私模型上，未涉及本地 DP（LDP）或 shuffle 模型，分布式/不可信服务器场景下能否保持同等效用是开放问题。

## 相关工作与启发
- **vs Sadeghi & Fazel (2021)**：此前唯一的 DP 背包子模工作，依赖精确多线性扩展（最坏 $2^n$ 查询）、仅单调、误差对 $n$ 多项式且随 $1/c_{\min}$ 缩放；本文用组合贪心骨架换来多项式查询、polylog($n$) 误差、覆盖非单调。
- **vs Mitrovic et al. (2017)**：在**基数**约束下给出 $(1-1/e)$ 近似、$\tilde{O}(\Delta k^{1.5}/\varepsilon)$ 误差；本文把同等量级的误差与近似搬到更难的**背包**约束，证明背包并不比基数贵多少（除 $\sqrt k$ 间隙外）。
- **vs Gupta et al. (2010) / Chaturvedi et al. (2023)**：分别在 ∆-decomposable 目标与流式设置下研究 DP 子模并给出下界；本文处理更一般的有界敏感度类，并把它们的下界拿来界定自身误差的最优性。
- **vs 非私有 Two-Guess-Greedy（Feldman et al. 2023）/ 密度贪心丢弃（Han et al. 2021）**：本文以它们为骨架，核心贡献是「如何在不牺牲近似比的前提下把它们私有化」。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个非单调 DP-SMK，且把「并行枚举→子程序选择」的私有化路径点亮，单调误差与查询双重改进。
- 实验充分度: ⭐⭐⭐⭐ 纯理论，保证全面且与下界/SOTA 对齐，但无任何经验验证。
- 写作质量: ⭐⭐⭐⭐ 三难点—三武器结构清晰，Table 1 对比一目了然；细节重在附录。
- 价值: ⭐⭐⭐⭐ 补齐 DP 子模优化在背包约束与非单调目标上的关键空白，理论意义明确。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] A General Framework for Dynamic Consistent Submodular Maximization](a_general_framework_for_dynamic_consistent_submodular_maximization.md)
- [\[ICML 2026\] Budget-Feasible Mechanisms for Submodular Welfare Maximization in Procurement Auctions](budget-feasible_mechanisms_for_submodular_welfare_maximization_in_procurement_au.md)
- [\[NeurIPS 2025\] Online Two-Stage Submodular Maximization](../../NeurIPS2025/optimization/online_two-stage_submodular_maximization.md)
- [\[NeurIPS 2025\] A Unified Approach to Submodular Maximization Under Noise](../../NeurIPS2025/optimization/a_unified_approach_to_submodular_maximization_under_noise.md)
- [\[ICLR 2026\] Submodular Function Minimization with Dueling Oracle](../../ICLR2026/optimization/submodular_function_minimization_with_dueling_oracle.md)

</div>

<!-- RELATED:END -->
