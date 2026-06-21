---
title: >-
  [论文解读] Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis
description: >-
  [ICLR2026][学习理论][Best-of-N] 本文把推理时对齐里最常用的 Best-of-N（BoN）放进"软化"框架 Soft Best-of-N（SBoN）来分析，给出 SBoN/BoN 相对参考策略的 KL 散度上界、相对最优策略的遗憾（regret）上下界，并证明：当代理奖励模型质量差、发生过优化时，存在一个有限的逆温度 $\beta$ 使 SBoN 的遗憾界比 BoN 更紧，从而缓解 reward hacking。
tags:
  - "ICLR2026"
  - "学习理论"
  - "LLM 对齐"
  - "Best-of-N"
  - "推理时对齐"
  - "KL 散度"
  - "遗憾界"
  - "奖励过优化"
---

# Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=tCv1D3M7Lb](https://openreview.net/forum?id=tCv1D3M7Lb)  
**代码**: 待确认  
**领域**: 学习理论 / LLM 对齐  
**关键词**: Best-of-N、推理时对齐、KL 散度、遗憾界、奖励过优化

## 一句话总结
本文把推理时对齐里最常用的 Best-of-N（BoN）放进"软化"框架 Soft Best-of-N（SBoN）来分析，给出 SBoN/BoN 相对参考策略的 KL 散度上界、相对最优策略的遗憾（regret）上下界，并证明：当代理奖励模型质量差、发生过优化时，存在一个有限的逆温度 $\beta$ 使 SBoN 的遗憾界比 BoN 更紧，从而缓解 reward hacking。

## 研究背景与动机
**领域现状**：大模型对齐的后处理方法很多——RLHF、DPO、SLiC、controlled decoding，以及推理时不训练模型、直接在采样阶段选答案的 Best-of-N（BoN）。这些方法本质上都在近似同一个问题的解：KL 正则化的奖励最大化 $\max_\pi \mathbb{E}_{y\sim\pi}[r(y,x)] - \frac{1}{\beta}\mathrm{KL}(\pi\|\pi_{\mathrm{ref}})$，其最优解是一个"倾斜分布"（tilted policy）$\pi_{\beta,r}(y|x)\propto \pi_{\mathrm{ref}}(y|x)\exp(\beta r(y,x))$，在奖励高和贴近参考模型之间折中。

**现有痛点**：已有理论几乎都在**理想假设**下分析 BoN——假设手里就是真奖励 $r^\star$，没有代理误差。Beirami 等人据此证明 $\mathrm{KL}(\pi_{\mathrm{BoN}}\|\pi_{\mathrm{ref}})\le \log N - 1 + \tfrac1N$，并说明 BoN 在 reward-vs-KL 曲线上几乎最优。但现实里 BoN 选答案靠的是**学出来的代理奖励** $\hat r$，它只是真奖励的近似。一旦 $\hat r$ 和 $r^\star$ 有偏差，BoN 这种"硬选最高分"的贪婪规则就会**过优化（overoptimization）**代理奖励、选到真实质量更低的答案，也就是 reward hacking。

**核心矛盾**：BoN 取 N 个候选里代理分最高的那一个（$\beta\to\infty$ 的极限），完全相信 $\hat r$ 的排序。当 $\hat r$ 不可靠时，越使劲选（N 越大、越贪婪）反而越糟——proxy 的微小排序错误被放大。已有遗憾分析（Huang et al. 2025）给出的界随奖励估计误差的 $L_\infty$ 范数增长，且在误差消失时不一定收敛干净。

**本文目标**：在**有代理误差**的现实设定下，量化两件事如何影响对齐质量：(a) 对齐策略与参考策略的 KL 散度；(b) 遗憾（regret），即最优策略与对齐策略的真实奖励之差。并回答：什么时候该"软化"BoN、软多少。

**切入角度**：作者不直接分析硬 BoN，而是借助它的光滑版本 **Soft Best-of-N（SBoN）**——把"取最高分"换成"按代理分做 softmax 采样"，引入一个可调的逆温度 $\beta$。$\beta\to\infty$ 退回 BoN，$\beta\to0$ 退回从参考模型直接采样。有了这个连续旋钮，就能把 BoN 当成 SBoN 的一个端点，统一推导随 $N$、$\beta$、奖励质量变化的标度行为。

**核心 idea**：用"软化镜头"看 BoN——证明在过优化场景下，适度软化（有限 $\beta$）能在"KL 折中收益"和"代理估计误差"之间取得平衡，使 SBoN 的遗憾界严格优于硬 BoN。

## 方法详解

### 整体框架
这是一篇纯理论分析论文，没有要训练的模型，"方法"就是一套围绕 SBoN 的分析框架：先定义采样策略和评价指标，再推 KL 散度界、再推遗憾上下界，最后用界的结构解释"何时该软化"。

整体链路是：① 固定一个 prompt $x$，从参考策略 $\pi_{\mathrm{ref}}$ 独立采 $N$ 个候选 $Y_{1:N}$；② SBoN 不再硬取最高代理分，而是按 softmax 概率 $P_Z(i)\propto\exp(\beta\hat r(Y_i,x))$ 从 $N$ 个候选里抽一个返回，闭式策略为 $\pi^{\mathrm{SBoN}}_{\hat r}(y|x)=\pi_{\mathrm{ref}}(y|x)\exp(\beta\hat r(y,x))/Z_{N,\beta}$；③ 为了刻画"代理奖励有多不准"，定义**倾斜误差** $\varepsilon_{\beta,r}$ 和**覆盖度** $C_{\beta,r}$ 两个量；④ 用这两个量加 $N,\beta$ 写出 KL 散度上界（Lemma 4.1/4.2）和遗憾上下界（Theorem 5.2/5.6），其中 BoN 的界都通过 $\beta\to\infty$ 的极限从 SBoN 的界导出；⑤ 比较 SBoN 与 BoN 的界，刻画出存在最优 $\beta$、SBoN 胜出的过优化区间。

理解全篇的两条主线索是 KL 散度（衡量对齐策略偏离参考多远，越大越能拿高奖励但也越容易过优化）和遗憾（衡量离真实最优还差多少），所有定理都在量化"代理误差 + N + $\beta$"如何同时拉扯这两条线。

### 关键设计

**1. Soft Best-of-N：把"硬取最高分"换成可调温度的 softmax 采样**

BoN 的根本问题是它对代理奖励"全信"——确定性地取 $i^\star=\arg\max_i\hat r(Y_i,x)$，proxy 排序错一点就被照单全收。作者从 KL 正则化奖励最大化的角度重新推导选择规则：在 $N$ 个候选索引的单纯形 $\Delta_N$ 上最大化期望代理奖励 $\mathbb{E}_Z[\hat r(Y_Z,x)]$，并加一个熵正则 $\frac1\beta H(P_Z)$。唯一解就是 softmax 分布

$$P_Z(i)=\frac{\exp(\beta\hat r(Y_i,x))}{\sum_{j=1}^N\exp(\beta\hat r(Y_j,x))},$$

从中采样 $Z$ 并返回 $Y_Z$，即 SBoN。它的闭式策略（Mayrink Verdun et al. 2025，Lemma 1）是 $\pi^{\mathrm{SBoN}}_{\hat r}(y|x)=\pi_{\mathrm{ref}}(y|x)\exp(\beta\hat r(y,x))/Z_{N,\beta}$，形式上正是**有限样本版的倾斜最优策略** $\pi_{\beta,\hat r}\propto\pi_{\mathrm{ref}}\exp(\beta\hat r)$。逆温度 $\beta$ 是一个连续旋钮：$\beta\to\infty$ 退回硬 BoN，$\beta\to-\infty$ 退回 worst-of-N，$\beta\to0$ 退回均匀采样（等价于直接从 $\pi_{\mathrm{ref}}$ 采）。这一步的价值在于把离散的"选最高"问题嵌进连续族，BoN 成了端点，后面所有 BoN 的界都能通过取 $\beta\to\infty$ 极限从 SBoN 顺手导出。

**2. 倾斜误差与覆盖度：两个刻画"代理有多坏、参考有多稀"的标量**

要谈过优化，必须先有一个量把"代理奖励 $\hat r$ 偏离真奖励 $r^\star$ 有多远"压成一个可进定理的数。作者定义**倾斜误差**

$$\varepsilon_{\beta,r}(x):=\frac1\beta\log\Big(\mathbb{E}_{Y\sim\pi_{\mathrm{ref}}}\big[e^{\beta(r^\star(Y,x)-\hat r(Y,x))^2}\big]\Big),$$

它把平方估计误差按倾斜权重平均：$\beta=0$ 时退化为均方误差 MSE，$\beta\to\infty$ 时变成估计误差的上确界范数平方 $\|r^\star-\hat r\|_\infty^2$。它取值在 $[0,1]$、关于 $\beta$ 单调递增，并约定 $\varepsilon_{\beta,r}>0$ 即进入"奖励错配 / 过优化"区。关键之处是这里用的是**校准后**的奖励（calibrated：值域 $[0,1]$、且在参考模型下服从 $\mathrm{Unif}(0,1)$），所以只要 proxy 是真奖励的严格单调变换、排序不变，$\varepsilon$ 就为 0、不发生过优化——这点用原始未校准奖励是抓不到的。

配套定义**覆盖度** $C_{\beta,r}(x):=\sum_y \pi_{\beta,r}^2(y|x)/\pi_{\mathrm{ref}}(y|x)$（即倾斜策略对参考的 $\chi^2$ 散度 +1），其极限 $C_{\infty,r}(x)=1/\sum_i\pi_{\mathrm{ref}}(y^{\max}_{i,r}(x)|x)$ 衡量"参考模型多难生成最优答案"：参考越少吐出最优响应，$C_\infty$ 越大，对齐越吃力。$\varepsilon$（代理质量）和 $C_\infty$（参考质量）就是后面所有界里的两个核心旋钮。

**3. KL 散度上界：随 $N,\beta$ 增长的标度，以及代理误差带来的额外代价**

对齐拿到多少真实奖励增益，受 KL 散度约束——由 Pinsker 不等式，$\mathbb{E}_{\pi^{\mathrm{SBoN}}_{r^\star}}[r^\star]\le 0.5+\sqrt{\tfrac12\mathrm{KL}(\pi^{\mathrm{SBoN}}_{r^\star}\|\pi_{\mathrm{ref}})}$，即真实奖励相对参考的提升不超过 KL 的平方根，所以想拿高奖励就需要"敢偏离"、需要大 KL。作者先在真奖励下给出

$$\mathrm{KL}\big(\pi^{\mathrm{SBoN}}_{r^\star}(y|x)\,\|\,\pi_{\mathrm{ref}}(y|x)\big)\le \log\!\Big(\frac{N}{1+(N-1)\exp(-\beta)}\Big)\quad(\text{Lemma 4.1}),$$

取 $\beta\to\infty$ 即得 $\mathrm{KL}(\pi_{\mathrm{BoN}}\|\pi_{\mathrm{ref}})\le\log N$。这个界比 Beirami 的 $\log N-1+\tfrac1N$ 松，但胜在对任意 $\beta$ 都成立、在 $\beta=0$ 处紧，且界随 $\beta$ 增大而增大——印证"温度越高越敢偏离"。更关键的是 Lemma 4.2 量化了"用代理换真奖励"的代价：

$$\mathrm{KL}\big(\pi^{\mathrm{SBoN}}_{r^\star}\,\|\,\pi^{\mathrm{SBoN}}_{\hat r}\big)\le \frac{N\beta\sqrt{\varepsilon_{\beta,r}(x)}}{1+(N-1)\exp(-\beta)}\Big(\frac{N\exp(2\beta)}{(N-1)^2}+1\Big),$$

当 $\varepsilon=0$（无过优化）时这个界为 0、两策略重合。Remark 4.4 由此点出**核心张力**：固定 $N$ 时，Lemma 4.1 要 $\beta$ 大（换更好的 KL 折中、更高奖励），Lemma 4.2 却要 $\beta$ 小（让代理对真策略的估计更准），两者打架，于是存在一个最优 $\beta$ 平衡二者——这正是 SBoN 可能胜过 BoN 的来源。

**4. 遗憾上下界：用界的结构证明"过优化时该软化、不过优化时该硬选"**

最终目标是遗憾 $\Delta J_{r^\star}=J_{r^\star}(\pi^\star_{r^\star})-J_{r^\star}(\pi^{\mathrm{SBoN}}_{\hat r})$，即真实最优与对齐策略的真奖励差。Theorem 5.2 给出 SBoN 的遗憾上界

$$\Delta J_{r^\star}\le \sqrt{\varepsilon_{\beta,r}(x)}\Big(\sqrt{C_{\infty,\hat r}}+\sqrt{C_{\infty,r^\star}}\Big)+2\sqrt{\tfrac12\log\!\Big(1+\tfrac{C_{\infty,\hat r}-1}{N}\Big)}+\frac{\log C_{\infty,r^\star}(x)}{\beta},$$

三项分别对应代理误差、有限样本统计误差、软化引入的偏置（$\propto 1/\beta$）。BoN 的界（Proposition 5.3）由 $\beta\to\infty$ 极限导出：$\sqrt{\varepsilon_\beta}$ 变成 $L_\infty$ 误差 $\sqrt{\varepsilon_\infty}$，$1/\beta$ 项消失。对比 Huang et al. 2025，本文的界在误差消失（$\varepsilon_\infty=0$）或 $N$ 增大时**保持有限**，且基于校准奖励。Theorem 5.6 / Proposition 5.7 进一步在 Margin 假设 $\gamma(x)=1-\sup_{y\notin Y^\star}r^\star(y,x)\in(0,1)$ 下给出**下界**，说明界不是空的。

最后 Remark 5.8/5.9 把"何时软化"讲透：定义 $g(\beta)=\beta(\sqrt{\varepsilon_\infty}-\sqrt{\varepsilon_\beta})$，因 $g(0)=g(\infty)=0$ 且 $g\ge0$，必存在最大值点 $\beta^\star\in(0,\infty)$。**过优化时**（$\varepsilon>0$），若 $\frac{\log C_{\infty,r^\star}}{\sqrt{C_{\infty,\hat r}}+\sqrt{C_{\infty,r^\star}}}\le g(\beta^\star)$，则 SBoN 在 $\beta^\star$ 处的遗憾界严格紧于 BoN——该软化。**不过优化时**（$\varepsilon=0$），SBoN 的界比 BoN 多了 $\log C/\beta$ 这一只增不减的项，反而 BoN（$\beta\to\infty$）更优——该硬选、该让 $N$、$\beta$ 都尽量大。

## 实验关键数据

### 主实验
实验是为印证理论而非刷榜：用 Olmo-2 1B 作生成器、Attaq 数据集（含有害 prompt）测无害性（harmlessness，越高越好），以 LLM-as-a-Judge 当真奖励 $r^\star$，对每个 RM 采 256 个响应做经验校准（算分位数）。两组对照——强代理奖励 ArmoRM 8B（接近真奖励）vs 弱代理奖励 Beaver 7B。

| 设定 | 代理奖励模型 | 现象 | 与理论对应 |
|------|------------|------|-----------|
| 强 RM | ArmoRM 8B | 无害性随 $N$ 增大而单调上升，硬 BoN 表现好 | $\varepsilon\approx0$，不过优化，BoN 占优（Remark 5.9） |
| 弱 RM | Beaver 7B | 大 $N$ 时 BoN 因 reward hacking 退化，SBoN 用合适 $\beta$ 稳住/反超 | $\varepsilon>0$，过优化，存在 $\beta^\star$ 使 SBoN 胜（Remark 4.4 / 5.8） |

### 消融实验
理论论文的"消融"主要是数值实验扫超参，验证界的形状：

| 扫的变量 | 观察 | 说明 |
|---------|------|------|
| 逆温度 $\beta$（弱 RM） | 中等 $\beta$ 的无害性曲线高于 $\beta\to\infty$（BoN）和 $\beta\to0$ | 印证存在有限最优 $\beta^\star$ |
| 样本数 $N$ | KL 上界随 $N$ 增长，与 Lemma 4.1 的 $\log\frac{N}{1+(N-1)e^{-\beta}}$ 一致 | App. I.2 数值验证 KL 界形状 |
| RM 质量（强→中→弱） | RM 越弱，BoN 越早过优化、软化收益越大 | $\varepsilon_{\beta,r}$ 越大、软化越有用 |

### 关键发现
- **过优化是软化收益的前提**：只有当代理奖励质量差（$\varepsilon>0$）时，SBoN 才比 BoN 好；代理够准时反而该用 BoN。这与理论里 $g(\beta)$ 的判据完全吻合。
- **存在有限最优温度**：弱 RM 下无害性关于 $\beta$ 非单调，太硬（BoN）会 reward hacking、太软（接近 $\pi_{\mathrm{ref}}$）没对齐效果，中间有甜点。
- **奖励校准是分析能成立的关键**：用校准奖励才能让"proxy 是真奖励单调变换 ⇒ $\varepsilon=0$ ⇒ 不过优化"这条逻辑成立，原始未校准奖励抓不到这层。

## 亮点与洞察
- **把 BoN 嵌进连续族再分析**：通过 SBoN 的逆温度 $\beta$ 把硬 BoN 当成 $\beta\to\infty$ 端点，所有 BoN 的界都靠取极限从 SBoN 顺手得到，一套分析覆盖两个算法——这种"先软化再取极限"的套路可迁移到其他贪婪选择型推理算法。
- **倾斜误差这个指标设计得很巧**：$\varepsilon_{\beta,r}$ 用同一个 $\beta$ 把 MSE（$\beta=0$）和 $L_\infty$ 误差（$\beta\to\infty$）连成一条谱，正好对应 SBoN 从软到硬的过渡，让"代理误差"和"采样温度"在同一尺度下对话。
- **判据可操作**：Remark 5.8 给出 $\frac{\log C_{\infty,r^\star}}{\sqrt{C_{\infty,\hat r}}+\sqrt{C_{\infty,r^\star}}}\le g(\beta^\star)$ 这种显式条件，原则上能根据"代理多差、参考多稀"判断该不该软化、软多少，对实践有指导意义。

## 局限与展望
- 作者承认自己给 BoN 的 KL 上界 $\log N$ 比 Beirami 等人的 $\log N-1+\tfrac1N$ **更松**，未来需要推一个对 BoN 渐近紧的界。
- 分析里**倾斜误差和 SBoN 策略共用同一个 $\beta$**，作者指出解耦这两个逆温度是有前景的方向（误差刻画的温度和采样温度不必绑定）。
- 假设较强：奖励**校准**（值域 $[0,1]$ 且参考下均匀）、Achievable maximum reward（最优响应处奖励 $=1$）、prompt/响应集有限、Margin 假设 $\gamma(x)\in(0,1)$。实际 RM 未必满足校准，校准本身要采 256 个样本估分位数，成本不低。
- 实验规模小（1B 生成器、单数据集、无害性单指标），是为验证理论而设，离"指导大规模对齐生产"还有距离。

## 相关工作与启发
- **vs Beirami et al. 2024 / Mroueh 2024**：他们在真奖励、双射假设下给出更紧的 BoN KL 界 $\log N-1+\tfrac1N$；本文放宽到**有代理误差**的现实设定，代价是 BoN 的 KL 界变松（$\log N$），但换来对任意 $\beta$ 通用、并能分析过优化。
- **vs Yang et al. 2024**：他们证明 BoN 渐近等价于 KL 约束 RL 的解，但前提是能拿到最优/真奖励；本文正是去掉这个理想假设、研究 proxy 与真奖励差距如何通过遗憾和 KL 体现。
- **vs Huang et al. 2025**：同样研究 BoN 遗憾，但他们的界随奖励估计误差的 $L_\infty$ 范数增长、用未校准奖励；本文的界在误差消失或 $N\to\infty$ 时保持有限，且基于校准奖励，能干净刻画"无过优化时遗憾以 $O(\exp(-N))$ 衰减"。
- **vs Mayrink Verdun et al. 2025**：他们提出 SBoN 并分析其向倾斜最优策略的收敛速率；本文借用其 SBoN 定义，但聚焦**过优化场景下**的 KL 与遗憾界，这是已有文献基本空白的部分。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在有代理误差/过优化设定下统一给出 SBoN 与 BoN 的 KL 与遗憾上下界，并刻画 SBoN 胜出区间。
- 实验充分度: ⭐⭐⭐ 实验只为印证理论，规模与指标偏小，但强/弱 RM 对照清晰、与理论预测一致。
- 写作质量: ⭐⭐⭐⭐ 概念（倾斜误差、覆盖度）定义清楚，定理与极限关系组织得当，便于追索。
- 价值: ⭐⭐⭐⭐ 为"何时该软化 BoN、软多少"提供了可操作的理论判据，对推理时对齐有指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Sharp KL Convergence Analysis for Diffusion Models under Minimal Assumptions](a_sharp_kl_convergence_analysis_for_diffusion_models_under_minimal_assumptions.md)
- [\[ICLR 2026\] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem](revisiting_tree-sliced_wasserstein_distance_through_the_lens_of_the_fermatweber_.md)
- [\[ICLR 2026\] Towards a Sharp Analysis of Offline Policy Learning for f-Divergence-Regularized Contextual Bandits](towards_a_sharp_analysis_of_offline_policy_learning_for_f-divergence-regularized.md)
- [\[ICLR 2026\] On the Interpolation Effect of Score Smoothing in Diffusion Models](on_the_interpolation_effect_of_score_smoothing_in_diffusion_models.md)
- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)

</div>

<!-- RELATED:END -->
