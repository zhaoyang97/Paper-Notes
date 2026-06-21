---
title: >-
  [论文解读] Efficient and Sharp Off-Policy Learning under Unobserved Confounding
description: >-
  [ICLR 2026][因果推理][未观测混淆] 本文为"存在未观测混淆"的个性化离线策略学习推导了价值函数锐界的闭式表达 + 半参数有效估计量：，把原本不稳定的 minimax 优化化简为一次普通最小化，并证明最小化该估计量即可得到最优的混淆鲁棒策略。 - 领域现状：离线策略学习（off-policy learning）希…
tags:
  - "ICLR 2026"
  - "因果推理"
  - "未观测混淆"
  - "边际敏感性模型(MSM)"
  - "半参数效率"
  - "锐界(sharp bound)"
  - "鲁棒策略学习"
---

# Efficient and Sharp Off-Policy Learning under Unobserved Confounding

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=7nTKiJLkWS](https://openreview.net/forum?id=7nTKiJLkWS)  
**代码**: [https://github.com/konstantinhess/Efficient_sharp_policy_learning](https://github.com/konstantinhess/Efficient_sharp_policy_learning)  
**领域**: 因果推断 / 离线策略学习 / 敏感性分析  
**关键词**: 未观测混淆, 边际敏感性模型(MSM), 半参数效率, 锐界(sharp bound), 鲁棒策略学习  

## 一句话总结
本文为"存在未观测混淆"的个性化离线策略学习推导了价值函数锐界的**闭式表达 + 半参数有效估计量**，把原本不稳定的 minimax 优化化简为一次普通最小化，并证明最小化该估计量即可得到最优的混淆鲁棒策略。

## 研究背景与动机
- **领域现状**：离线策略学习（off-policy learning）希望从观测数据里学出"给定协变量该如何分配处理"的最优策略，标准做法（DM/IPW/双重稳健 DR）都建立在**无混淆假设**（unconfoundedness）之上——即观测协变量 $X$ 捕获了所有同时影响处理选择和结果的因素。
- **现有痛点**：现实中无混淆假设几乎总被违反。例如病人种族会影响其能否获得某种治疗，但病历里通常不记录种族。一旦存在未观测混淆 $U$，价值函数 $V(\pi)$ **无法点识别**，标准方法给出有偏估计，甚至学出"有害"的策略。
- **核心矛盾**：唯一处理该任务的已有方法 Kallus & Zhou (2018a/2021) 用边际敏感性模型(MSM)做混淆鲁棒学习，但有两个硬伤：(i) 必须数值求解基于 IPW 结果的 **minimax 优化**，不稳定；(ii) **不具半参数效率**，方差大、有限样本表现差，且其界并非锐界。
- **本文目标**：在 MSM 框架下，给价值函数的最坏情形界（worst-case bound）求出闭式解，并构造一个半参数有效、且证明能导出最优鲁棒策略的估计量。
- **核心 idea**：**先把"内层 sup"解析掉再优化**——通过 MSM 把 $Q(a,x)$ 的锐界写成闭式（分位数加权形式），从而最坏情形价值 $V^{+,*}(\pi)=\sup_{\tilde p\in\mathcal P(\Gamma)}V(\pi)$ 直接有显式表达，minimax 退化为对 $\pi$ 的单纯最小化；再用一步偏差校正（基于有效影响函数）让该界的估计达到最低方差。

## 方法详解

### 整体框架
方法分四步串成一条链：① 用 MSM 约束真实倾向得分与名义倾向得分之比（混淆强度由 $\Gamma\ge 1$ 刻画）；② 推导潜在结果条件均值锐界 $Q^{\pm,*}(a,x)$ 的闭式分位数加权表达，进而得到价值函数锐界 $V^{\pm,*}(\pi)$ 的闭式；③ 对该界求**有效影响函数**，构造一步偏差校正估计量 $\hat V^{+,*}(\pi)$，达到半参数效率；④ 把策略类参数化为神经网络 $\pi_\theta$，对 $\hat V^{+,*}(\pi_\theta)$ 做梯度下降（含样本分裂/交叉拟合）即得鲁棒策略。

```mermaid
flowchart LR
    A[观测数据 X,A,Y] --> B[MSM 约束<br/>Γ⁻¹≤倾向比≤Γ]
    B --> C[Q±,* 锐界闭式<br/>分位数加权]
    C --> D[价值函数锐界<br/>V+,*=∫Σ Q+,*·π]
    D --> E[有效影响函数 EIF<br/>一步偏差校正]
    E --> F[半参数有效估计量<br/>V̂+,*]
    F --> G[对 π_θ 梯度下降<br/>得鲁棒策略 π*]
```

### 关键设计

**1. 把 minimax 的内层 sup 解析成闭式锐界：** 这是全文的"杠杆点"。原始目标是 $\pi^*=\arg\min_{\pi}\sup_{\tilde p\in\mathcal P(\Gamma)}V(\pi)$，内层对所有与 MSM 兼容的分布取上确界，正是 Kallus & Zhou 必须数值 minimax 的根源。作者证明（Prop. 4.1）锐上界可逐点拆解：$V^{\pm,*}(\pi)=\int_{\mathcal X}\sum_a Q^{\pm,*}(a,x)\,\pi(a\mid x)\,dp(x)$，其中 $Q^{\pm,*}$ 又有闭式 $Q^{\pm,*}(a,x)=c^{\mp}(a,x)\mu^{\pm}(a,x)+c^{\pm}(a,x)\bar\mu^{\pm}(a,x)$。这里 $c^{\pm}(a,x)=b^{\pm}e(a,x)+\Gamma^{\pm1}$、$b^{\pm}=1-\Gamma^{\pm1}$，而 $\mu^{\pm},\bar\mu^{\pm}$ 是用条件分位数 $F^{-1}_{x,a}(\alpha^{\pm})$（$\alpha^+=\Gamma/(1+\Gamma)$）切割后对结果 $Y$ 的截断条件期望。直觉上，最坏情形等价于"在 MSM 允许的范围内，把概率质量尽量堆到使结果更差的那一侧分位数"，因此界由分位数阈值决定。这样一来 $\sup$ 被显式算出，外层只剩 $\arg\min_\pi V^{+,*}(\pi)$ 这一个普通最小化，彻底绕开不稳定的 IPW minimax。

**2. 基于有效影响函数的一步偏差校正估计量：** 锐界 $V^{+,*}$ 依赖一组冗杂的 nuisance 函数 $\eta=\{e(a,x),F^{-1}_{a,x}(\alpha^{\pm}),\mu^{\pm},\bar\mu^{\pm}\}$，若直接把估计的 $\hat\eta$ 插回去（朴素 plug-in），nuisance 的估计误差会带来一阶偏差。作者推导出该锐界对应的有效影响函数（EIF，非平凡，因为要支持**离散多处理**而非仅二元处理，影响函数与已有 CATE 工作完全不同），据此做一步偏差校正：$\hat V^{+,*}(\pi)=\mathbb P_n\{\text{plug-in 项}-\widehat{\text{一阶偏差}}\}$（式(15)，含对分位数指示 $\hat\Delta^+$、截断期望 $\hat\mu^+,\hat{\bar\mu}^+$ 等校正项）。Theorem 4.3 证明：在 $\mathbb E[|Y|^2]<\infty$ 且密度在分位数邻域有界等温和条件下，该估计量**半参数有效**，即在所有无偏估计中方差最低。

**3. 学习保证：最小化估计界 ⇒ 最优鲁棒策略：** 仅有"界估得准"不够，还要保证"最小化这个界真的能学到好策略"。作者用策略类的 Rademacher 复杂度 $R_n(\Pi)$ 给出泛化界（Theorem 4.4）：在 $|Y|\le C_y$、$C_v=2C_y(1+\Gamma^{-1}+\Gamma)$ 的设定下，以至少 $1-\delta$ 概率，对所有 $\pi\in\Pi$ 同时成立 $V(\pi)\le \hat V^{+,*}(\pi)+2C_v\big(R_n(\Pi)+\tfrac52\sqrt{\tfrac{1}{2n}\log\tfrac2\delta}\big)$。含义是：估计的锐上界高概率地真正上界住未知的真实价值，于是样本足够时最小化 $\hat V^{+,*}$ 也就最小化了 $V$，导出最优 $\pi^*$。

**4. 推广到带基线的策略改进（policy improvement）：** 医疗里常已有"标准疗法"作为基线策略 $\pi_0$，此时关心的是相对改进的遗憾 $R_{\pi_0}(\pi)=V(\pi)-V(\pi_0)$（负值即改进）。作者证明上述结果直接平移：给出遗憾上界的闭式 $R^+_{\pi_0}(\pi)=\int\sum_a\big(Q^{+,*}(a,x)\pi(a\mid x)-Q^{-,*}(a,x)\pi_0(a\mid x)\big)dp(x)$（Cor. 4.5）、对应的半参数有效估计量（Cor. 4.6），以及**改进保证**（Cor. 4.7）——只要经验估计 $\hat R^+_{\pi_0}(\pi)$ 为负，就高概率保证 $\pi$ 优于基线、不会引入危害，这在高风险医疗场景尤为关键。

## 实验关键数据

### 主实验：随混淆强度变化（合成数据，遗憾越低越好）
数据生成过程取自 Kallus et al. (2019)，二元处理；同步改变 DGP 里的真实混淆 $\Gamma^*$ 与估计器里的敏感性参数 $\Gamma$，报告相对随机策略的遗憾。

| 方法 | $\Gamma^*{=}2$ | $\Gamma^*{=}6$ | $\Gamma^*{=}10$ | $\Gamma^*{=}14$ | $\Gamma^*{=}16$ |
|---|---|---|---|---|---|
| 标准 IPW | −1.31 | −0.09 | −0.06 | −0.05 | −0.03 |
| 标准 DR | −1.30 | −0.18 | −0.07 | −0.05 | −0.04 |
| Kallus & Zhou (2018a/2021) | −1.21 | −0.40 | −0.16 | −0.10 | −0.08 |
| **本文 Efficient+Sharp** | **−1.12** | **−0.89** | **−0.64** | **−0.50** | **−0.30** |

混淆越强差距越大：标准方法 $\Gamma^*>1$ 后几乎失效；唯一可比基线 Kallus & Zhou 也迅速退化；本文在大 $\Gamma^*$ 下相对增益最高可达约 **4 倍**。

### 消融与稳健性实验

| 实验 | 设置 | 关键结果 |
|---|---|---|
| 敏感性参数误设 (Fig. 3) | DGP 固定 $\Gamma^*{=}7$，估计器 $\Gamma$ 从 1 扫到 100 | 即便完全误设、甚至 $\Gamma{=}100$（近乎无假设），本文仍显著优于有偏 DR；Kallus & Zhou 在 $\Gamma$ 偏大时迅速退回基线 |
| 半参数效率 (Fig. 4) | 本文有效估计量 vs 锐界的朴素 plug-in | 有效估计量在低样本下遗憾更低，且随样本量增大增益更明显（印证最低方差性质） |
| 真实医疗数据 (Fig. 5) | International Stroke Trial（4 种处理：阿司匹林/肝素/二者/无），人为剔除部分病人并删去舒张压制造混淆，目标延长生存天数 TD | 本文在 $\Gamma{=}24$ 处最优，整体治疗策略最佳且对 $\Gamma$ 稳健；仅在极小 $\Gamma$ 时退化（此时不防混淆、且 nuisance 更复杂） |

### 关键发现
- 标准 DM/IPW/DR 在有混淆时系统性失效——这是假设违背而非调参问题。
- 闭式锐界 + 有效估计的组合，使方法对敏感性参数误设具有强鲁棒性，这是 minimax 基线做不到的（后者会"退回基线"）。
- 方法天然支持**离散多处理**（如中风试验的 4 种治疗），突破了多数敏感性分析工作只能处理二元处理的限制。

## 亮点与洞察
- **"先解析内层、再优化外层"的范式**：把一个不稳定的 minimax 问题在 MSM 结构下化简为闭式上界 + 单纯最小化，是本文最优雅之处，也是性能与稳定性提升的根源。
- **半参数效率落到策略学习**：以往锐界 + EIF 主要用于 CATE 估计，本文首次把"锐界的半参数有效估计"完整搬到策略学习，并因支持离散处理而需要全新的影响函数推导。
- **可验证的安全保证**：Cor. 4.7 提供"经验遗憾界为负即保证不劣于标准疗法"的可检验条件，对医疗等高风险决策极具实用价值。

## 局限与展望
- **依赖 MSM 与 $\Gamma$ 的正确设定**：方法本质是"在给定混淆强度上界内做最坏情形优化"，$\Gamma$ 仍需领域知识或数据驱动启发式来定；虽对误设鲁棒，但 $\Gamma$ 严重偏小会失去保护。
- **nuisance 估计更重**：需估计倾向得分、条件分位数、截断条件期望等多组 nuisance，比标准 DR 复杂，低混淆/小样本时这一开销可能不划算。
- **静态单步设定**：当前聚焦单步、离散处理的策略学习，连续处理与序贯/MDP 场景（动态策略）留作未来；后者影响函数会进一步不同。
- **MSM 框架的固有局限**：MSM 只约束倾向得分比，无法刻画所有形式的混淆结构，换用其他敏感性模型时锐界形式需重新推导。

## 相关工作与启发
- **无混淆下的离线策略学习**：DM (Qian & Murphy 2011)、IPW (Swaminathan & Joachims 2015)、DR (Athey & Wager 2021; Dudik et al. 2011)——本文把 DR 的"有效影响函数"思想推广到了有混淆的锐界场景。
- **混淆鲁棒策略学习**：Kallus & Zhou (2018a/2021) 是唯一直接对标的前作，本文在"锐界 + 半参数效率 + 闭式"三点上全面超越。
- **因果敏感性分析**：MSM (Tan 2006) 及 CATE 锐界系列 (Dorn & Guo 2022; Frauen et al. 2023)——本文复用其 $Q^{\pm,*}$ 分位数加权分解，但把目标从 CATE 估计转为策略价值优化，并扩展到离散多处理。
- **启发**：对"内层带 sup/inf 的鲁棒优化"，若约束集有良好结构（如 MSM 的比值约束），优先尝试把内层解析成闭式，往往比直接 minimax 更稳更高效；同时"锐界 + EIF 偏差校正"是一套可迁移到许多部分识别问题的方法论模板。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 首个在 MSM 下给价值函数锐界做半参数有效估计、并证明导出最优鲁棒策略的工作，闭式化 minimax 的思路干净有力。
- **实验充分度**: ⭐⭐⭐⭐ 合成数据（变混淆/误设/效率）+ 真实中风试验多角度验证，且覆盖多处理；规模偏小、缺更大真实数据集与更多任务。
- **写作质量**: ⭐⭐⭐⭐⭐ 动机—理论—算法—保证—实验层层递进，定理与直觉解释配合到位，对比表清晰。
- **价值**: ⭐⭐⭐⭐⭐ 直击医疗/公共政策等高风险决策中"无混淆假设不成立"的核心痛点，并给出可验证的安全保证，实用与理论意义俱强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Causal Imitation Learning under Expert-Observable and Expert-Unobservable Confounding](causal_imitation_learning_under_expert-observable_and_expert-unobservable_confou.md)
- [\[ICLR 2026\] Function Induction and Task Generalization: An Interpretability Study with Off-by-One Addition](function_induction_and_task_generalization_an_interpretability_study_with_off-by.md)
- [\[ICLR 2026\] Efficient Ensemble Conditional Independence Test Framework for Causal Discovery](efficient_ensemble_conditional_independence_test_framework_for_causal_discovery.md)
- [\[ICLR 2026\] Privacy-Protected Causal Survival Analysis Under Distribution Shift](privacy-protected_causal_survival_analysis_under_distribution_shift.md)
- [\[ICLR 2026\] Learning Dynamic Causal Graphs Under Parametric Uncertainty via Polynomial Chaos Expansions](learning_dynamic_causal_graphs_under_parametric_uncertainty_via_polynomial_chaos.md)

</div>

<!-- RELATED:END -->
