---
title: >-
  [论文解读] Non-Asymptotic Analysis of (Sticky) Track-and-Stop
description: >-
  [ICLR 2026][学习理论][pure exploration] 本文为纯探索领域两大经典算法 Track-and-Stop（TAS）与 Sticky Track-and-Stop（S-TAS）首次给出了**非渐近（有限置信度）样本复杂度上界**，填补了它们"只在 $\delta\to0$ 时被证明最优、却不知有限 $\delta$ 下表现如何"的理论空白。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "纯探索"
  - "多臂老虎机"
  - "pure exploration"
  - "best-arm identification"
  - "Track-and-Stop"
  - "Sticky Track-and-Stop"
  - "有限置信度"
  - "样本复杂度"
  - "非渐近分析"
---

# Non-Asymptotic Analysis of (Sticky) Track-and-Stop

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=vebqP5aioj](https://openreview.net/forum?id=vebqP5aioj)  
**代码**: 待确认  
**领域**: 学习理论 / 纯探索 / 多臂老虎机  
**关键词**: pure exploration, best-arm identification, Track-and-Stop, Sticky Track-and-Stop, 有限置信度, 样本复杂度, 非渐近分析  

## 一句话总结
本文为纯探索领域两大经典算法 Track-and-Stop（TAS）与 Sticky Track-and-Stop（S-TAS）首次给出了**非渐近（有限置信度）样本复杂度上界**，填补了它们"只在 $\delta\to0$ 时被证明最优、却不知有限 $\delta$ 下表现如何"的理论空白。

## 研究背景与动机
- **领域现状**：在纯探索（pure exploration）问题中，统计学家顺序地从 $K$ 个未知分布（臂）采样，要在错误率不超过风险参数 $\delta$ 的前提下，用尽量少的样本回答关于环境的问题（如找出最优臂）。Garivier & Kaufmann (2016) 提出的 **Track-and-Stop（TAS）** 是奠基性方法，当答案映射 $i^\star(\mu)$ 单值时（如唯一最优臂），它在 $\delta\to0$ 时**渐近最优**；Degenne & Koolen (2019) 又提出 **Sticky TAS（S-TAS）**，把渐近最优性推广到答案多值的设定（如 $\epsilon$-最优臂识别，多个臂都算正确答案）。
- **现有痛点**：两者的最优性证明都是**渐近性质的**——只刻画了 $\delta\to0$ 的极限行为，对真实使用中的有限 $\delta$（finite-confidence）区间毫无保证。多篇工作（Barrier 2023 等）指出根源在于：数据稀少时经验均值尚未集中，TAS 的采样规则会剧烈震荡，难以分析；S-TAS 更复杂，问题更严重。
- **核心矛盾**：为绕开这个不稳定性，已有工作要么给 TAS 加置信区间（Degenne et al. 2019，但要解一个无高效 oracle 的 max-max-min 难题），要么把采样规则"偏置"向均匀探索（Barrier et al. 2022，但牺牲了经验性能）。可经验上**原版 TAS 一直最快**，于是问题变成：能不能**不改算法**，直接给原版 TAS / S-TAS 证出有限置信度保证？
- **本文目标**：回答"能否刻画 TAS 与 S-TAS 的非渐近保证"，且要求结论对**任意结构化问题与任意答案对应关系**都成立。
- **核心 idea**：**[绕开收敛性论证，改用"信息累积"视角]** 不去追问采样比例 $N(t)/t$ 收敛到哪里（这正是原渐近证明的核心，但脆弱），而是在一串"好事件" $E_t$ 下，**把停止规则用算法自己的采样规则下界化**，再借助一个推广到 $M$ 之外的 oracle 权重表达式注入最优速率 $T^\star(\mu)^{-1}$，从而直接证出有限 $\delta$ 上界。

## 方法详解

### 整体框架
本文不是新算法，而是一套**非渐近证明框架**。两个主定理（TAS 见 Theorem 1，S-TAS 见 Theorem 2）形式一致：期望停止时间被界为 $\mathbb{E}_\mu[\tau_\delta]\le 2eK+10K^4+T_0(\delta)$，其中 $T_0(\delta)$ 是满足 $\beta_{t,\delta}\le tT^\star(\mu)^{-1}-g(t)$ 的最小时刻，刻画"算法以多快速度积累信息越过停止阈值 $\beta_{t,\delta}$"。整套论证沿"好事件归约 → 停止规则下界化 → 注入最优速率"三步推进。

```mermaid
flowchart LR
  A[好事件 E_t<br/>经验均值集中] --> B[归约:<br/>E[τ_δ] ≤ T̄ + 2eK]
  B --> C[下界化停止规则<br/>≈ 采样规则求和]
  C --> D[注入推广 oracle 权重 ω⋆<br/>引出 T⋆μ⁻¹]
  D --> E[T_0δ 有限<br/>且 δ→0 时恢复渐近最优]
```

### 关键设计

**1. 好事件归约：把期望停止时间拆成"坏事件概率 + 确定性停止时刻"。** 分析的起点是定义一族好事件 $E_t=\{\forall s\in[\lceil\sqrt t\rceil,t]:\sum_k N_k(s)d(\hat\mu_k(s),\mu_k)\le 8K\log s\}$，即在不太早的时刻经验均值都足够接近真值。它有两条互补性质：一是坏事件概率可和，$\sum_{t\ge3}\mathbb{P}_\mu(E_t^c)\le 2eK$（贡献了上界里那个 $2eK$）；二是存在阈值 $\bar T$ 使 $t\ge\bar T$ 时 $E_t$ 蕴含停止，即 $E_t\subseteq\{\tau_\delta\le t\}$。两者一拼即得 $\mathbb{E}_\mu[\tau_\delta]\le\bar T+2eK$，把"求期望停止时间"转化为"找一个确定性的 $\bar T$"。这一招借鉴自 Degenne et al. (2019)、Jourdan & Degenne (2023) 的有限置信度分析，与原版依赖 $N(t)/t$ 收敛的渐近思路彻底不同。

**2. 停止规则 ↔ 采样规则的下界对接：证明的技术核心。** 难点在于：算法**停止**靠的是 $\max_i\inf_\lambda\sum_k N_k(t)d(\hat\mu_k(t),\lambda_k)\ge\beta_{t,\delta}$，而它**采样**用的是每一步的 $\inf_{\lambda\in\neg i_s}\sum_k\omega_k(s)d(\tilde\mu_k(s),\lambda_k)$，两者并不直接相等。本文的关键观察是：在 $E_t$ 下由 C-Tracking 保证 $N(t)\approx\sum_{s}\omega(s)$、且 $d(\hat\mu_k,\cdot)\approx d(\mu_k,\cdot)$，于是停止规则可被算法自己每一步采样目标之和**下界化**（仅差一个 $\widetilde O(\sqrt t)$ 的次线性项）。对接成功后，TAS 单值的情形可直接用"若 $i_s\ne i^\star(\mu)$ 则 $\mu\in\neg i_s$"完成放缩，这一步显式依赖答案单值，也解释了为何 TAS 在多答案下会失效。

**3. 推广 oracle 权重到 $M$ 之外：让最优速率得以注入。** 经典文献里最优速率写作 $T^\star(\mu)^{-1}=\sup_\omega\inf_{\lambda\in\neg i^\star(\mu)}\sum_k\omega_k d(\mu_k,\lambda_k)$（式(1)），但 $i^\star$ 只对 $\mu\in M$ 有定义，而算法的经验估计 $\hat\mu(t)$ 可能落在 $M$ 外。本文改用等价改写 $T^\star(\mu)^{-1}=\sup_\omega\max_{i\in I}\inf_{\lambda\in\neg i}\sum_k\omega_k d(\mu_k,\lambda_k)$（式(2)/(4)），把 oracle 权重的定义自然延拓到任意模型。正是这一改写让分析能在中间步骤引入任意 $\omega^\star\in\omega^\star(\mu)$，最终把累积信息量化为 $(t-\sqrt t-1)T^\star(\mu)^{-1}-\widetilde O(t^{3/4})$，使 $T_0(\delta)$ 有限并在 $\delta\to0$ 时精确恢复 $T^\star(\mu)\log(1/\delta)$ 的渐近最优率。两个温和假设（$\sigma^2$ 次高斯、参数有界 $M\subset[\mu_{\min},\mu_{\max}]$）配合一个对 $\hat\mu$ 的投影 $\tilde\mu$，仅用于让 KL 散度 $d(\tilde\mu_k(s),\cdot)$ 在 $t$ 很小时"良好行为"；论文还说明如何去掉投影，代价仅多一项 $T_M$。

**4. S-TAS 专属的"答案集坍缩时刻" $T_\mu$：处理多答案的额外难点。** 在多答案设定下，S-TAS 早期对所选答案 $i_s$ 几乎没有控制（只靠 $I$ 上一个预设全序），无法像 TAS 那样直接从 $\neg\bar\imath$ 切到 $\neg i_s$。本文借助 $\mu\mapsto i_F(\mu)$ 的**上半连续性**证明（Lemma 4）：存在时刻 $T_\mu$，当 $t\ge T_\mu$ 时在 $E_t$ 下置信域 $C_s$ 内所有候选模型 $\mu'$ 都满足 $i_F(\mu')\subseteq i_F(\mu)\cup(I\setminus i^\star(\mu))$——即坏答案被排除。$T_\mu$ 由"$\mu$ 到答案翻转边界的距离" $\epsilon_\mu$ 决定（在 $\epsilon$-best-arm 中可显式取 $\epsilon_\mu=\Delta_\mu/2$）。于是 Theorem 2 的 $T_0(\delta)$ 只比 TAS 多出一项问题相关常数 $T_\mu T^\star(\mu)^{-1}$。值得注意的是，该证明**不依赖** $N(t)/t$ 收敛到何处、也不用 $\omega^\star(\mu,\neg\bar\imath)$ 的凸性（这两者恰是 Degenne & Koolen 2019 渐近证明的支柱），全程只在"信息累积"的语言里推进。

## 实验关键数据

本文是**纯理论工作，无实验**。其"关键结果"以两个样本复杂度定理的形式给出，下面用表格梳理。

### 主结果：两个非渐近上界

| 算法 | 适用设定 | $\mathbb{E}_\mu[\tau_\delta]$ 上界 | $g(t)$ 量级 |
|------|----------|-----------------------------------|-------------|
| TAS (Thm 1) | 单答案 $\lvert i_F(\mu)\rvert=1$ | $2eK+10K^4+T_0(\delta)$，$T_0(\delta)=\inf\{t:\beta_{t,\delta}\le tT^\star(\mu)^{-1}-g(t)\}$ | $64\sigma DLK^2\log K\sqrt{t}\log^2 t+16\sigma D\sqrt{K}t^{3/4}\log t$ |
| S-TAS (Thm 2) | 任意多答案 | $2eK+10K^4+T_0(\delta)$，$T_0(\delta)=\inf\{t:\beta_{t,\delta}\le (t-T_\mu)T^\star(\mu)^{-1}-g(t)\}$ | $80\sigma DLK^2\log K\sqrt{t}\log^2 t+32\sigma D\sqrt{K}t^{3/4}\log t$ |

### 量级对比

| 维度 | 结论 |
|------|------|
| 主项随 $\delta$ | $T_0(\delta)\sim T^\star(\mu)\log(1/\delta)$（差多对数因子与常数），$\delta\to0$ 时**精确恢复渐近最优** |
| $\delta$-无关项 | $2eK$（坏事件概率和）+ $10K^4$（技术步骤要求 $t\ge10K^4$），均为分析产物 |
| 次线性余项 | $\beta_{t,\delta}+g(t)$ 以 $O(\log(1/\delta)+t^{3/4})$ 次线性增长，被右端 $tT^\star(\mu)^{-1}$ 的线性项追上，保证 $T_0(\delta)$ 有限 |
| 去投影代价 | 不用投影即得原版 Garivier & Kaufmann (2016) 的 TAS，上界仅多一个问题相关项 $T_M$ |

### 关键发现
- TAS 在中等 $\delta$ 下经验上一直很快（前人多次基准测试），本文从理论侧补全了"它确实享有有限置信度保证"的解释。
- S-TAS 是文献中已知**唯一**能最优求解任意多答案纯探索问题的算法，本文给出该一般类问题的**首个**有限置信度分析。
- 单答案时 S-TAS 的 $T_\mu$ 依赖可去掉，回到与 TAS 同形（仅相差常数倍），但两者证明因采样规则不同而本质不同。

## 亮点与洞察
- **"不改算法、只改证明"**：避开了 Degenne et al. (2019) 加置信区间（带来 max-max-min 难题）和 Barrier et al. (2022) 偏置采样（牺牲性能）的两条改算法路线，第一次直接为原版 TAS 给出有限置信度保证。
- **方法论换轨**：用"信息累积"代替"采样比例收敛"作为分析对象，绕开了多答案下 $N(t)/t$ 收敛到凸包而非 oracle 权重的根本障碍，这是 TAS 在多答案下失效、却能为 S-TAS 证出非渐近界的关键分水岭。
- **一个小改写撬动全局**：把 $T^\star(\mu)^{-1}$ 从式(1) 改写成对答案取 $\max$ 的式(2)，看似平凡，却让 oracle 权重能定义在 $M$ 之外、从而在经验估计落到 $M$ 外时仍可推进证明。

## 局限与展望
- **上界含 $10K^4$ 与 $T_\mu$ 等可能宽松的项**：作者明言 $2eK$、$10K^4$ 是分析产物，$T_0(\delta)$ 也只是隐式定义（附录另给显式上界），这些项在臂数 $K$ 大或 $T_\mu$ 大时未必紧，离匹配下界仍有距离。
- **依赖两个假设**：次高斯指数族 + 参数有界，虽温和但排除了重尾等更一般分布；投影步骤虽可去除但要付 $T_M$ 的代价。
- **无下界配对**：本文只给上界，未证明这些非渐近上界在 $K$、$\Delta_\mu$ 等问题量上是否信息论意义下最优，留作开放问题。
- **未做数值验证**：纯理论，没有实验展示有限 $\delta$ 下上界的实际紧度，读者难以直观感受常数大小。

## 相关工作与启发
- **单答案 + TAS 谱系**：Garivier & Kaufmann (2016) 奠基，后被推广到结构化（Lipschitz、单峰）等设定（Wang et al. 2021；Poiani et al. 2024；Kaufmann & Koolen 2021），但最优性一直停在渐近层面。
- **绕不稳定性的两条改算法路线**：Degenne et al. (2019) 的乐观 TAS、Barrier et al. (2022) 的偏置采样，是与本文最接近的有限置信度工作——本文的贡献正是不走这两条路，直接分析原版。
- **多答案 + S-TAS**：Degenne & Koolen (2019) 给出 S-TAS 及其渐近最优性，并证明 $\mu\mapsto i_F(\mu)$ 的上半连续性——后者成为本文 $T_\mu$ 论证的支点。先前多答案的有限置信度结果都局限于 $\epsilon$-best-arm 等子类（Even-Dar 2002；Jourdan et al. 2023 等），本文第一次覆盖任意多答案。
- **启发**：当渐近证明依赖某个"收敛到哪"的脆弱论证时，换成"累积了多少信息"的下界化视角，可能既更稳健又能延伸到更一般的设定——这一思路或可迁移到其他基于 lower-bound game 的顺序决策算法的有限样本分析。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 解决了文献中两个公开问题（原版 TAS 与一般多答案 S-TAS 的有限置信度分析），且方法论上从"收敛"换轨到"信息累积"，是干净的概念创新。
- **实验充分度**: ⭐⭐⭐ — 纯理论工作无实验，但定理形式完整、量级讨论清楚；扣分在于缺数值验证来体现上界常数的实际紧度。
- **写作质量**: ⭐⭐⭐⭐ — 背景铺垫到位，两条证明给了详尽 proof sketch 并点明 TAS 与 S-TAS 的关键差异，逻辑链条清晰。
- **价值**: ⭐⭐⭐⭐ — 为两个被广泛使用、却只有渐近保证的基础算法提供了首个有限置信度理论支撑，对纯探索理论社区有扎实的奠基价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Online Inventory Optimization in Non-Stationary Environment](online_inventory_optimization_in_non-stationary_environment.md)
- [\[ICLR 2026\] Stop Guessing: Choosing the Optimization-Consistent Uncertainty Measurement for Evidential Deep Learning](stop_guessing_choosing_the_optimization-consistent_uncertainty_measurement_for_e.md)
- [\[ICLR 2026\] Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization](sharp_asymptotic_theory_for_q-learning_with_textttld2z_learning_rate_and_its_gen.md)
- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)
- [\[ICLR 2026\] On Smoothness Bounds for Non-Clairvoyant Scheduling with Predictions](on_smoothness_bounds_for_non-clairvoyant_scheduling_with_predictions.md)

</div>

<!-- RELATED:END -->
