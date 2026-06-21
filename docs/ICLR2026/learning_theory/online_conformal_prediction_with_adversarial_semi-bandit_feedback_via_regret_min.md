---
title: >-
  [论文解读] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization
description: >-
  [ICLR 2026][学习理论][在线 conformal prediction] 本文把"只在真实标签落进预测集时才能看到标签"的在线 conformal prediction 问题，重构成一个对抗多臂老虎机问题（每个阈值候选当作一只臂），通过设计专属损失函数把 regret 与 miscoverage 率显式挂钩，并改造 EXP3.P 得到 OCP-Unlock+ 算法，在不依赖 i.i.d. 假设的对抗数据流下首次给出长程覆盖保证。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "在线学习"
  - "Conformal Prediction"
  - "在线 conformal prediction"
  - "对抗半老虎机反馈"
  - "EXP3.P"
  - "regret 最小化"
  - "长程覆盖保证"
---

# Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=RMWcdp5IUy](https://openreview.net/forum?id=RMWcdp5IUy)  
**代码**: 待确认  
**领域**: 学习理论 / 在线学习 / Conformal Prediction  
**关键词**: 在线 conformal prediction, 对抗半老虎机反馈, EXP3.P, regret 最小化, 长程覆盖保证

## 一句话总结
本文把"只在真实标签落进预测集时才能看到标签"的在线 conformal prediction 问题，重构成一个对抗多臂老虎机问题（每个阈值候选当作一只臂），通过设计专属损失函数把 regret 与 miscoverage 率显式挂钩，并改造 EXP3.P 得到 OCP-Unlock+ 算法，在不依赖 i.i.d. 假设的对抗数据流下首次给出长程覆盖保证。

## 研究背景与动机
**领域现状**：conformal prediction 是一种模型无关的不确定性量化方法，它为每个输入构造一个"预测集"，并保证该集合以用户指定的概率包含真实标签（即 coverage guarantee）。在线版本（online conformal prediction）处理数据点逐个到来的场景，即使在对抗数据流下也能给出长程覆盖保证：经过足够多步后，经验覆盖率会逼近目标水平 $1-\alpha$。

**现有痛点**：几乎所有现有在线 conformal prediction 方法都默认**全反馈（full feedback）**——每一步都能看到真实标签 $y_t$。它们要么用 $y_t$ 估计分位数，要么用 $y_t$ 评估多个候选预测集上的 miscoverage 损失，本质上都离不开真实标签。但现实里标签往往拿不到：人在回路（human-in-the-loop）系统中，只有当预测集"框住"了正确答案、值得人去核对时，标签才被揭示。

**核心矛盾**：Ge et al. (2025) 提出过一个**半老虎机反馈（semi-bandit feedback）**设定——真实标签 $y_t$ 只在它落进所选预测集时才揭示——但他们的覆盖保证只在 i.i.d. 数据流下成立。也就是说：要么你假设全反馈、要么你假设 i.i.d.，没人能同时处理"反馈受限"+"对抗数据流"这两个最贴近现实的难点。

**本文目标**：在**对抗半老虎机反馈**下设计算法，使 miscoverage 率 $\mathrm{MC}(T)=\frac1T\sum_t m_t(\pi_t)$ 满足 $\mathrm{MC}(T)\le\alpha+\varepsilon(T)$ 且 $\varepsilon(T)\to0$，同时控制预测集平均大小 $\mathrm{Ineff}(T)=\frac1T\sum_t|\hat C_{\pi_t}(x_t)|$（否则永远输出整个标签空间 $\mathcal Y$ 就能平凡地达到覆盖）。

**切入角度**：作者观察到，在线 conformal prediction 的核心决策就是选一个阈值 $\pi\in[0,1]$ 来参数化预测集 $\hat C_\pi(x)=\{\tilde y: f_t(x,\tilde y)\ge\pi\}$。如果把连续阈值空间离散化、每个候选阈值当成一只"臂"，整个问题就变成了对抗多臂老虎机——而老虎机领域早有 EXP3.P 这类在自适应对手下仍有高概率 sublinear regret 的成熟算法。

**核心 idea**：用"对抗老虎机 + regret 最小化"代替"分位数估计"来做在线 conformal prediction，并通过一个量身定制的损失把 regret 翻译成 coverage 保证。

## 方法详解

### 整体框架
本文要解决的是：在对抗、且标签只在被预测集框住时才可见的数据流上，逐步构造预测集并保证长程覆盖。整体思路是一条"问题翻译 + 算法改造"的链路：先把阈值空间 $[0,1]$ 离散成 $K$ 个候选 $\Pi$，把每个阈值候选当作老虎机的一只臂（**问题重构**）；再为 conformal prediction 设计一个由 miscoverage 项和 inefficiency 项组成的损失函数 $\ell_t(\pi)$（**损失设计**），它让学习者既被罚漏覆盖、又被罚预测集过大；然后证明一条 learner-agnostic 的桥梁引理（**regret→coverage**），表明任何对该损失取得 sublinear regret 的学习者都能达到目标覆盖；最后改造 EXP3.P，利用半老虎机反馈下的额外信息和阈值单调性，得到收敛更快的 **OCP-Unlock+** 算法。

每一步都对应原文 Section 4 的一个子节，四个关键设计与框架严格同序对应。

### 关键设计

**1. 把在线 conformal prediction 重构成对抗多臂老虎机：把"选阈值"变成"拉臂"**

针对的痛点是：连续阈值空间无法直接套用现成的老虎机算法。作者把 $[0,1]$ 均匀离散成 $K=|\Pi|$ 个候选阈值，每个 $\pi\in\Pi$ 对应一个阈值参数化预测集 $\hat C_\pi$，于是"每步选一个阈值"就等价于"每步从 $K$ 只臂里拉一只"。学习者表现用相对最优臂（事后诸葛）的 regret 度量：

$$\mathrm{Reg}(T):=\sum_{t=1}^{T}\ell_t(\pi_t)-\min_{\pi\in\Pi}\sum_{t=1}^{T}\ell_t(\pi).$$

这个重构和典型对抗老虎机有个关键区别（原文 Table 1）：典型老虎机里对手直接控制损失向量 $\ell_t$，而这里**对手控制的是样本 $(x_t,y_t)$、损失函数 $\ell_t:\Pi\to[\ell_{\min},\ell_{\max}]$ 由学习者自己设计**。这正是下一个设计能成立的前提——损失是可设计的，才能把它做成有利于覆盖控制的形状。同时作者考虑的是**自适应对手**：它可以根据学习者过去的动作来选下一个样本，这正是 EXP3.P 能处理而普通 EXP3 处理不了的情形。

**2. 为 conformal prediction 量身定制的损失函数：把"覆盖优先"写进损失形状**

只看 miscoverage 反馈是不够的——一个永远输出 $\pi_t=0$（即整个 $\mathcal Y$）的懒惰学习者能拿到 $\mathrm{MC}(T)=0$，但预测集毫无信息量。所以损失必须同时罚"漏覆盖"和"集合太大"：

$$\ell_t(\pi;\alpha):=d_t(\pi;\alpha)+a_t(\pi),$$

其中 miscoverage 项基于反馈 $m_t(\pi)=\mathbb 1(y_t\notin\hat C_\pi(x_t))$ 设计为

$$d_t(\pi;\alpha)=m_t(\pi)+\mathbb 1(m_t(\pi)=0)\,(\alpha+\alpha(1-\alpha))+\mathbb 1(m_t(\pi)=1)\,(-\alpha(1-\alpha)).$$

这里 $g(\alpha)>0$、$h(\alpha)<0$ 以一种 $\alpha$-自适应的方式调节损失：覆盖成功（$m_t=0$）时加一点正项、覆盖失败（$m_t=1$）时减一点，使得 $d_t$ 在覆盖成功的阈值集 $\Pi_t^*=\{\pi:m_t(\pi)=0\}$ 上始终不大于失败集，且二者差值 $(1+h(\alpha))-g(\alpha)$ 随 $\alpha$ 增大趋于 0——也就是 $\alpha$ 越小越优先保覆盖。inefficiency 项 $a_t(\pi)$ 则可取任意惩罚大集合的形式（原文 Eq. 7 给了一个 $[0,1]$ 内、阶为 $o(T)^{-1}$ 的具体设计，用常数 $c$ 平衡覆盖与集合大小的 trade-off）。这种"覆盖优先"的损失形状是后面所有覆盖保证的根基。

**3. 从 regret 到 coverage 的桥梁：让任意 sublinear-regret 学习者都能保覆盖**

最小化 regret 并不天然等于控制 miscoverage 率 $\mathrm{MC}(T)$，这是本文必须打通的一环。作者证明了一条 learner-agnostic 的引理（Lemma 1），借鉴了 selective generation 里用 regret 界控制 FDR 的思路（Lee et al., 2025）：在 $a_t(\pi)$ 取 $o(T)^{-1}$ 阶的设计下，对任意对损失 Eq. 5 产生 regret $\mathrm{Reg}(T)$ 的学习者，

$$\mathrm{MC}(T)-\alpha\le\frac1T\mathrm{Reg}(T)+C_{\mathrm{MC}}(T),$$

其中修正项 $C_{\mathrm{MC}}(T)$ 是 $o(T)^{-1}$ 阶的常数项（实验显示其中的 $C_{\mathrm{gap}}(T)$ 随 $T$ 增大而减小）。这条桥梁的意义是：只要学习者 regret 是 sublinear 的（$\mathrm{Reg}(T)/T\to0$），长程覆盖就至少能达到 $1-\alpha$。于是问题彻底归约为"找一个在自适应对手下 sublinear regret 的老虎机算法"——EXP3.P 正是现成答案。

**4. OCP-Unlock+：用阈值单调性"解锁"额外反馈，加速逼近目标覆盖**

直接套 EXP3.P（作者称作 OCP-Bandit）能拿到 $1-\delta$ 高概率 sublinear regret，但它浪费了半老虎机的信息。关键观察是：当 $m_t(\pi_t)=0$（覆盖成功）时真实标签 $y_t$ 被揭示，于是 $f_t(x_t,y_t)$ 可算，**所有** $\pi\in\Pi$ 的 $m_t(\pi)$ 都能评估（full unlocking）；即便 $m_t(\pi_t)=1$（失败），由于 $m_t(\pi)$ 关于 $\pi$ 单调非减，凡 $\pi\ge\pi_t$ 必有 $m_t(\pi)=1$，于是也能评估一个子集（partial unlocking）。作者把这个可评估子集定义为**解锁集**：

$$\Pi_t(\pi_t):=\begin{cases}\Pi & \text{if } m_t(\pi_t)=0\\ \{\pi\in\Pi:\pi\ge\pi_t\} & \text{if } m_t(\pi_t)=1.\end{cases}$$

据此设计了一个随可用反馈量自适应的有偏增益估计器 $\tilde g_t(\pi\mid\Pi_t(\pi_t))$，对 $m_t(\pi_t)=0$ 和 $=1$ 两种情形分别构造（原文 Eq. 11 的 (A)、(B) 两支）。其中沿用了"覆盖优先"逻辑：对 $\Pi_t^*$ 内的阈值给更大的重要性权重、对更大的预测集削弱探索项 $\beta$ 的权重。当 $m_t(\pi_t)=1$ 时，无法计算的那部分增益用一个**伪增益（pseudo-gain）** $\tilde g_t(\pi)$ 近似（基于 $\tilde\ell_t(\pi)=1-\alpha(1-\alpha)-\frac{c\alpha}{1+\alpha(1-2\alpha)\pi}o(T)^{-1}$），并刻意构造成 $\tilde g_t(\pi_1)\ge g_t(\pi_2)$（$\pi_1$ 在解锁集外、$\pi_2$ 在内），把单调性写进估计器。最终 OCP-Unlock+ 在 $\beta=\sqrt{\frac{\ln K}{KT}}$、$\gamma=1.05\sqrt{\frac{K\ln K}{T}}$、$\eta=0.95\sqrt{\frac{\ln K}{KT}}$ 的设置下，给出高概率覆盖偏差界（Theorem 1）：

$$\mathrm{MC}(T)-\alpha\le\ell_{\mathrm{diff}}\!\left(\sqrt{\tfrac{C\ln K}{T}}+4.15\sqrt{\tfrac{K\ln K}{T}}+\sqrt{\tfrac{K}{T\ln K}\ln(\delta^{-1})}+2o(T)^{-1}\right)+C_{\mathrm{MC}}(T),$$

其中 $\ell_{\mathrm{diff}}=\ell_{\max}-\ell_{\min}$。这就是 Lemma 1 的桥梁套上 OCP-Unlock+ 的 sublinear regret 后的结果——解锁额外反馈不改变 regret 的渐近阶，但显著加快了实际逼近目标覆盖的速度。

### 损失函数 / 训练策略
核心训练目标即上文损失 $\ell_t(\pi)=d_t(\pi)+a_t(\pi)$；策略更新沿用 EXP3.P 的指数权重 + 均匀混合形式：$p_t(\pi)\propto(1-\gamma)\frac{\exp(\eta\tilde G_{t-1}(\pi))}{\sum_{\tilde\pi}\exp(\eta\tilde G_{t-1}(\tilde\pi))}+\gamma\frac1K$，其中累积增益 $\tilde G_t(\pi)=\tilde G_{t-1}(\pi)+\tilde g_t(\pi)$ 用解锁集上的有偏估计器累加。超参 $\beta,\gamma,\eta$ 按 Theorem 1 取值。

## 实验关键数据

### 主实验
在分类（ImageNet）与回归（UCI Airfoil Self-Noise）两类任务上，各自考察 i.i.d. 与 non-i.i.d. 两种设定，每个配置在 50 次独立运行上取平均。对比对象：SPS（Ge et al., 2025，半老虎机但只对 i.i.d. 有保证）、MVP（Bastani et al., 2022，**全反馈 oracle 基线**）。

| 任务 / 设定 | 配置 | 长程 miscoverage MC | inefficiency Ineff |
|------|------|------|------|
| ImageNet（$\alpha=0.15,K=200,T=5\times10^4$）i.i.d. | MVP (oracle, 全反馈) | 逼近目标 $\alpha$ | 最小 |
| 同上 i.i.d. | SPS | $\le\alpha$（任意时刻覆盖，i.i.d. 下成立） | 较大 |
| 同上 i.i.d. | **OCP-Unlock+** | 逼近目标 $\alpha$ | 居中 |
| ImageNet non-i.i.d.（scoring 函数随时间演化） | SPS | 持续下降、难以适应 | 波动 |
| 同上 non-i.i.d. | **OCP-Unlock+** | 逼近目标 $\alpha$ | 波动但受控 |
| Airfoil 回归（$\alpha=0.1,K=20$）协变量漂移 | SPS | 变得不可靠 | — |
| 同上 协变量漂移 | **OCP-Unlock+** | 保守（略低于目标） | 偏大 |

核心结论：在 non-i.i.d. / 协变量漂移下，**SPS 的覆盖保证崩掉**（MC 持续偏离目标），而 OCP-Unlock+ 始终把 MC 拉回目标附近，表现可与拥有全反馈的 oracle MVP 相提并论——这正是本文相对 SPS 的核心增益。

### 消融实验
原文附录给出多项分析（Appendix I–L）：

| 配置 | 关注点 | 说明 |
|------|--------|------|
| 变化离散化级别 $K$ | 臂数对覆盖/集合大小的影响 | Experiment 3，$K$ 影响逼近速度与 inefficiency |
| OCP-Bandit / OCP-Unlock / OCP-Unlock+ | 反馈利用程度逐级递增 | Experiment 4，越充分利用 unlocking 信息，逼近目标覆盖越快 |
| 变化目标 $\alpha$ | 算法对目标覆盖水平的敏感性 | Experiment 5 |
| 分析 $C_{\mathrm{MC}}(T)$ | 经验验证修正项随 $T$ 衰减 | Experiment 6，$C_{\mathrm{gap}}(T)$ 随 $T$ 增大而减小 |

### 关键发现
- **解锁额外反馈是逼近速度的关键**：从 OCP-Bandit → OCP-Unlock → OCP-Unlock+，对半老虎机额外信息（full / partial unlocking）的利用越充分，达到目标覆盖所需步数越少，而 regret 的渐近阶不变。
- **对抗鲁棒性是相对 SPS 的真正分水岭**：i.i.d. 下三方都还行，但一旦进入 non-i.i.d./协变量漂移，依赖 i.i.d. 假设的 SPS 直接失效，OCP-Unlock+ 仍稳。
- **代价是偏保守**：OCP-Unlock+ 的 inefficiency 通常大于全反馈 oracle MVP（尤其漂移下集合偏大），这是反馈受限下"宁可框大一点也要保覆盖"的合理代价。

## 亮点与洞察
- **问题翻译的漂亮之处**：把"连续阈值上的在线 conformal prediction"翻译成"离散臂的对抗老虎机"，一举接通了 EXP3.P 几十年的 sublinear-regret 理论积累，这种"换个领域语言就解锁整套工具"的思路非常可迁移。
- **损失可设计 = 杠杆点**：与典型老虎机由对手定损失不同，这里损失由学习者设计，作者据此把"覆盖优先"直接刻进损失和增益估计器的形状，这是把抽象 regret 界落地成具体覆盖保证的关键杠杆。
- **单调性是免费的额外反馈**：阈值化预测集的 miscoverage 关于 $\pi$ 单调，这个结构性事实让"失败那一步"也能解锁一整个子集的反馈（partial unlocking），几乎零成本地提升样本效率——凡是带单调结构的在线决策问题都值得借鉴。
- **regret→目标指标的归约范式**：Lemma 1 把 regret 界翻译成 miscoverage 界，和 selective generation 里 regret→FDR 控制是同一套路，说明"先证 regret、再做一步问题专属归约"是给在线方法挂目标保证的通用配方。

## 局限与展望
- **作者承认的局限**：算法是 context-free 的，不利用输入侧的额外上下文信息；inefficiency 相对全反馈 oracle 偏大、漂移下偏保守。
- **离散化的代价**：把 $[0,1]$ 离散成 $K$ 个臂，覆盖偏差界含 $\sqrt{K\ln K/T}$ 项，$K$ 太大拖慢逼近、太小则阈值粒度不够——$K$ 的选择是一个需要权衡的超参，理论上没有自适应取法。
- **自己发现的局限**：实验只在 ImageNet 分类 + 单个 UCI 回归数据集上验证，对抗 non-i.i.d. 用的是人工构造的 scoring 漂移 / 协变量漂移，真实对抗强度与现实人在回路场景的差距未充分检验。
- **改进思路**：引入上下文（contextual bandit）让阈值策略随输入自适应；探索 $K$ 的自适应离散化或连续臂老虎机以去掉 $\sqrt K$ 项；把伪增益的近似误差纳入界内做更紧的分析。

## 相关工作与启发
- **vs SPS (Ge et al., 2025)**：同为半老虎机反馈，但 SPS 的覆盖保证依赖 i.i.d. 假设，对抗/漂移下失效；本文是首个在**对抗**半老虎机反馈下给出长程覆盖保证的工作，靠的是 regret-最小化框架而非分位数估计。
- **vs MVP (Bastani et al., 2022)**：MVP 是**全反馈** oracle，每步能评估所有 $\pi$ 的 miscoverage，因而 inefficiency 最小；本文在反馈严重受限下逼近其覆盖表现，代价是集合略大。
- **vs Gibbs & Candès / Angelopoulos et al. 等在线 OGD 系**：这些方法基于在线梯度下降做分位数/quantile tracking，且基本默认全反馈；本文走的是离散臂 + 指数权重的老虎机路线，天然适配"标签按需揭示"的偏反馈结构。
- **vs Wang & Qiao (2024) 的 bandit 反馈**：那是更严格的 bandit 反馈（仅预测标签等于真值时才见标签），本文的 semi-bandit（落进集合即见标签）信息更多，也因此能解锁单调性带来的额外反馈。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个对抗半老虎机反馈下的在线 conformal prediction，问题设定 + regret→coverage 归约都是新的。
- 实验充分度: ⭐⭐⭐⭐ 分类/回归 × i.i.d./non-i.i.d. 四象限 + 多项消融，但数据集偏少、对抗设定较人工。
- 写作质量: ⭐⭐⭐⭐ 理论链路清晰、Table 1 的问题对照很到位，但记号密集、关键估计器细节需翻附录。
- 价值: ⭐⭐⭐⭐⭐ 把覆盖保证推广到"标签按需揭示 + 对抗数据流"的现实人在回路场景，实用意义强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Distribution-informed Online Conformal Prediction](distribution-informed_online_conformal_prediction.md)
- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICLR 2026\] Singleton-Optimized Conformal Prediction](singleton-optimized_conformal_prediction.md)
- [\[ICLR 2026\] Bandit Learning in Matching Markets Robust to Adversarial Corruptions](bandit_learning_in_matching_markets_robust_to_adversarial_corruptions.md)
- [\[ICLR 2026\] Conformal Prediction for Long-Tailed Classification](conformal_prediction_for_long-tailed_classification.md)

</div>

<!-- RELATED:END -->
