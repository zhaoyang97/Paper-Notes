---
title: >-
  [论文解读] Lipschitz Bandits with Stochastic Delayed Feedback
description: >-
  [ICLR 2026][在线学习][Lipschitz bandit] 首次系统研究连续臂空间 Lipschitz bandit 在随机延迟反馈下的学习问题，针对有界延迟提出 Delayed Zooming 算法（通过 lazy update 机制保持 $\Delta(x) \leq 6r_t(x)$ 的子最优 gap 界），针对无界延迟提出 DLPP 分阶段剪枝策略（遗憾与延迟分位数 $Q(p)$ 挂钩），并建立实例相关下界证明 DLPP 近最优。
tags:
  - "ICLR 2026"
  - "在线学习"
  - "Bandit 算法"
  - "Lipschitz bandit"
  - "延迟反馈"
  - "zooming 算法"
  - "分阶段剪枝"
  - "遗憾下界"
  - "分位数"
---

# Lipschitz Bandits with Stochastic Delayed Feedback

**会议**: ICLR 2026  
**arXiv**: [2510.00309](https://arxiv.org/abs/2510.00309)  
**代码**: 无  
**领域**: 在线学习 / Bandit 算法  
**关键词**: Lipschitz bandit, 延迟反馈, zooming 算法, 分阶段剪枝, 遗憾下界, 分位数

## 一句话总结

首次系统研究连续臂空间 Lipschitz bandit 在随机延迟反馈下的学习问题，针对有界延迟提出 Delayed Zooming 算法（通过 lazy update 机制保持 $\Delta(x) \leq 6r_t(x)$ 的子最优 gap 界），针对无界延迟提出 DLPP 分阶段剪枝策略（遗憾与延迟分位数 $Q(p)$ 挂钩），并建立实例相关下界证明 DLPP 近最优。

## 研究背景与动机

**领域现状**：Lipschitz bandit 将经典 MAB 扩展到连续度量空间 $(\mathcal{A}, \mathcal{D})$，奖励函数 $\mu$ 满足 1-Lipschitz 条件。经典 Zooming 算法通过自适应离散化在无延迟设定下达到最优遗憾率 $\tilde{O}(T^{(d_z+1)/(d_z+2)})$，其中 $d_z$ 为 zooming 维度。延迟反馈在 MAB、线性 bandit、kernel bandit 中已有广泛研究，但 Lipschitz bandit 的延迟问题完全空白。

**现有痛点**：(1) 连续臂空间 + 延迟反馈产生双重复杂性——每个采样点代表一个邻域区域，其估计依赖延迟的观测；(2) Zooming 算法核心分析依赖"置信半径仅在拉臂时变化"——延迟下此性质被打破，因为延迟奖励到达时未拉臂的置信半径也会缩小；(3) 有限臂延迟方法（如 Delayed-UCB1）分析不涉及连续空间的覆盖论证，无法直接推广。

**核心矛盾**：如何在反馈延迟且可能永远缺失的情况下，仍然保持对连续度量空间的高效自适应探索？

**本文目标** 设计在有界/无界随机延迟下均能达到近最优遗憾率的 Lipschitz bandit 算法，并证明其最优性。

**切入角度**：将问题按延迟支撑分为两个子问题——有界延迟用"zooming + lazy update"，无界延迟用"分阶段累积可靠反馈 + 淘汰"。

**核心 idea**：通过 lazy update 控制置信半径缩小速度（有界延迟），或通过分阶段 round-robin 采样和 $Q(p)$ 分位数联系拉臂/观测次数（无界延迟），恢复无延迟最优遗憾率。

## 方法详解

### 整体框架

这篇论文要解决的问题是：在连续臂空间上做 Lipschitz bandit，但奖励不再即时返回、而要等一个随机延迟才到手，甚至可能永远不到。问题形式化为三元组 $(\mathcal{A}, \mathcal{D}, \mu)$——$\mathcal{A}$ 是紧致倍增度量空间，$\mathcal{D}$ 为度量，$\mu: \mathcal{A} \to [0,1]$ 是 1-Lipschitz 未知奖励函数。每轮 $t$ 选臂 $x_t$，生成奖励 $y_t = \mu(x_t) + \epsilon_t$（$\epsilon_t$ 为 sub-Gaussian 噪声），但这条奖励要在随机延迟 $\tau_t \sim f_\tau$（独立于臂和奖励）之后才被观测到；目标是压低累积遗憾 $R(T) = \sum_{t=1}^T (\mu^* - \mu(x_t))$。

整篇工作按延迟「严不严重」切成两条算法线、再配一个下界封口，三块正好对应下面三个关键设计。延迟有界（$\tau_t \leq \tau_{\max}$）时，沿用经典 Zooming 的自适应离散化骨架，但用 **Delayed Zooming** 的 lazy update 机制把被延迟反馈拖快的置信半径重新锁住，恢复无延迟时的最优遗憾率。延迟无界、甚至有反馈永久缺失时，实时维护置信半径已不可行，于是换成 **DLPP** 的「分阶段攒够可靠反馈再剪枝」策略，把遗憾和延迟的分位数 $Q(p)$ 而非最坏情况挂钩。最后用一个匹配的**实例相关下界**证明 DLPP 已近最优。

### 关键设计

**1. Delayed Zooming：用 lazy update 锁住置信半径的缩小速度（有界延迟 $\tau_t \leq \tau_{\max}$）**

经典 Zooming 分析有一条命门：置信半径只在「拉到这条臂」时才变化，所以它和拉臂次数严格同步。可一旦有延迟，先前拉过的臂的奖励会在以后某一轮才到达，导致一条**当前根本没被拉**的臂的置信半径也在悄悄缩小，整套覆盖-激活论证就垮了。Delayed Zooming 的对策分两步。第一步是把置信半径的计算从拉臂次数 $n_t(x)$ 换成实际观测次数 $v_t(x)$：

$$r_t(x) = \sqrt{\frac{4\log T + 2\log(2/\delta)}{1 + v_t(x)}}$$

这样半径只反映真正到手的信息。第二步是引入 **lazy update 机制**：为每个活跃臂维护一个缓存队列 $Q[x]$，记 $s$ 为这条臂上次被拉的时刻，一旦累积观测会让 $v_t(x)+1 > 4 v_s(x)$，就把后续到达的反馈先压进缓存、暂不更新半径，直到下次真正拉这条臂时再一并消化。把观测增速卡在「不超过上次的 4 倍」这条线上，就能保证 $r_t(x) \geq \tfrac{1}{2} r_s(x)$——半径在两次拉臂之间最多缩一半。代价是子最优 gap 界从无延迟的 $\Delta(x) \leq 3r_t(x)$ 松到 $\Delta(x) \leq 6r_t(x)$（常数翻倍），但覆盖论证得以恢复，最终遗憾界为 $\tilde{O}\big(T^{\frac{d_z+1}{d_z+2}} + \tau_{\max} T^{\frac{d_z}{d_z+2}}\big)$，延迟只贡献一个加性项。

**2. DLPP：分阶段攒够可靠反馈再剪枝（无界延迟，含反馈永久缺失 $\tau = \infty$）**

当延迟可能无界甚至永远不到达时，再去维护实时的置信半径已无意义——你永远不知道某个反馈会不会来。DLPP 干脆放弃实时更新，改成「攒够统计量再决策」的分阶段淘汰式探索。第 $m$ 阶段维护一组半径 $r_m = 2^{-m}$ 的覆盖球 $\mathcal{B}_m$，对每个球做均匀 round-robin 采样，直到这个球累积到

$$v_m = \frac{2\log T + \log(2/\delta)}{2r_m^2}$$

个观测才停。阶段结束时按经验均值剪枝：凡是 $\hat\mu_m^* - \hat\mu_m(B) \geq 8r_m$ 的区域（明显落后于当前最佳球）一律淘汰，幸存区域再细分进入下一阶段。难点在于：采样是按拉臂次数走的，但决策依赖延迟后才到的观测次数，两者并不同步。DLPP 用 Chernoff 不等式把它们的概率联系起来——

$$\Pr\!\Big(v_{t+Q(p)}(B) \leq \tfrac{p}{2}\, n_t(B)\Big) \leq \exp\!\Big(-\tfrac{p}{8}\, n_t(B)\Big)$$

也就是说，只要再多等 $Q(p)$ 轮，已拉的臂里至少有 $p/2$ 的比例会变成可用观测，于是遗憾自然地和**延迟分位数** $Q(p)$（$p$ 比例的反馈在 $Q(p)$ 轮内到达）挂上钩。即使部分反馈彻底缺失，算法依然能工作，遗憾界为

$$R(T) \lesssim \min_{p \in (0,1]} \left\{ \frac{1}{p} T^{\frac{d_z+1}{d_z+2}} \Big(c\log\tfrac{T}{\delta}\Big)^{\frac{1}{d_z+2}} + Q(p) \right\}$$

对 $p$ 取 min 让界自动适配延迟分布的「中心质量」而非最坏情况。

**3. 实例相关下界：证明 DLPP 几乎无法再改进**

最后论文给出与上界形式匹配的下界，说明 DLPP 已近最优（至对数因子）。构造的关键是一族特殊延迟分布：延迟以概率 $p$ 取固定值 $\tau_0$、否则取 $\infty$（永久缺失）。在这族分布上用 Bernoulli 采样耦合，把延迟 Lipschitz bandit 归约到一个无延迟版本——以概率 $p$ 模拟出能即时看到反馈的算法，从而把无延迟的遗憾下界搬过来。最终得到

$$R(T) \gtrsim \frac{T^{(d_z+1)/(d_z+2)}(c\log T)^{1/(d_z+2)}}{p\log T} - \frac{1}{p} + \bar\Delta \cdot Q(p)$$

其中 $\bar\Delta = \int_\mathcal{A} \Delta(x) \big/ \int_\mathcal{A} 1$ 是平均子最优 gap。下界的最后一项 $\bar\Delta \cdot Q(p)$ 有清晰的来源：前 $Q(p)$ 轮完全收不到任何反馈，这段「盲飞」期的遗憾不可避免，恰好和上界里的 $Q(p)$ 项对上。

### 损失函数 / 训练策略

两个算法均为理论算法，无需梯度训练。关键参数选择均由理论推导确定：Delayed Zooming 的 lazy update 阈值 $4v_s(x)$、DLPP 每阶段所需观测数 $v_m$ 与剪枝阈值 $8r_m$。遗憾率的最优化由选择 $\rho = (\log T / T)^{1/(d_z+2)}$（有界延迟）或 $M = \frac{\log(T/(c\log T))}{d_z+2}$（DLPP）得到。

## 实验关键数据

### 主实验（三种奖励函数 × 两种延迟分布，$T=60000$，30 次独立试验）

| 奖励函数 | 算法 | 无延迟 | 均匀 $\mathbb{E}[\tau]=20$ | 均匀 $\mathbb{E}[\tau]=50$ | 几何 $\mathbb{E}[\tau]=20$ | 几何 $\mathbb{E}[\tau]=50$ |
|----------|------|:------:|:------:|:------:|:------:|:------:|
| 三角函数(1D) | Delayed Zooming | 138.97 | 154.55 | 171.07 | 159.30 | 152.98 |
| 三角函数(1D) | DLPP | 304.60 | 314.87 | 326.71 | 312.44 | 325.74 |
| 正弦函数(1D) | Delayed Zooming | 130.64 | 137.31 | 148.69 | 132.88 | 144.08 |
| 正弦函数(1D) | DLPP | 178.05 | 195.35 | 209.97 | 186.28 | 208.80 |
| 二维函数(2D) | Delayed Zooming | 1445.86 | 1843.05 | 1858.45 | 1463.38 | 1828.15 |
| 二维函数(2D) | DLPP | **1120.64** | **1159.85** | **1136.46** | **1120.63** | **1142.55** |

### 理论结果对比

| 设定 | 算法 | 遗憾上界 | 退化检验 |
|------|------|----------|----------|
| 有界延迟 $\tau_{\max}$ | Delayed Zooming | $\tilde{O}(T^{(d_z+1)/(d_z+2)} + \tau_{\max} T^{d_z/(d_z+2)})$ | $\tau_{\max}=0$ 退化为经典 Lipschitz bandit |
| 有界延迟 $d_z=0$ | Delayed Zooming | $O(\sqrt{cT\log T} + c\tau_{\max})$ | 退化为有限臂 MAB with delay |
| 无界延迟 | DLPP | $\min_p \{ \frac{1}{p} T^{(d_z+1)/(d_z+2)} (\cdot)^{1/(d_z+2)} + Q(p) \}$ | $p=0.5$ 时依赖中位数 $\tau_{\text{med}}$ |
| 下界 | — | $\frac{R}{p\log T} - \frac{1}{p} + \bar\Delta \cdot Q(p)$ | 与 DLPP 上界至对数因子匹配 |

### 关键发现

- 两种算法在有界/无界延迟下均保持次线性遗憾，延迟引起的额外遗憾仅为加性项
- 1D 场景 Delayed Zooming 遗憾更低，2D 场景 DLPP 剪枝+离散化策略更优（2D 无延迟 DLPP 已优于 Zooming）
- Delayed Zooming 在几何分布（无界）延迟实验中也工作良好，但理论保证仅限有界延迟——这是一个 open problem
- DLPP 遗憾曲线呈分段线性——源于阶段内均匀采样的阶段性遗憾累积

## 亮点与洞察

- **Lazy Update 的技术贡献**：看似简单的"用 $v_t$ 替代 $n_t$ + 缓存"，但分析极其非平凡。分析证明 $\Delta(x) \leq 6r_t(x)$ 的常数因子翻倍是精确的——当 $v_t(x)+1 > 4v_s(x)$ 时 $r_t(x)$ 恰好可能降至 $r_s(x)/2$ 以下，必须停止更新
- **分位数刻画的优雅性**：DLPP 遗憾界通过 $\min_{p \in (0,1]}$ 优化分位数，适应延迟分布的"中心质量"而非最坏情况——取 $p=0.5$ 得中位数依赖，取更小 $p$ 可处理重尾
- **上下界匹配**：下界构造使用 Bernoulli 耦合技巧——以概率 $p$ 模拟无延迟算法，证明延迟遗憾至少为 $\Omega(R/(p\log T))$，加上前 $Q(p)$ 轮无信息的 $\bar\Delta \cdot Q(p)$ 项，展示 DLPP 几乎不可改进

## 局限与展望

- Delayed Zooming 理论保证仅限有界延迟，扩展到无界延迟是论文明确指出的 open problem
- DLPP 需要覆盖 oracle，高维空间覆盖计算可能代价高昂
- 实验仅在 1D 和 2D 合成函数上验证（三角/正弦/二维双峰），缺乏更现实的应用场景
- 两种算法均需知晓时间窗口 $T$，未讨论 anytime 版本
- 未讨论对抗性延迟（adversarial delay）场景

## 相关工作与启发

- **vs Delayed-UCB1 (Joulani et al., 2013)**: 有限臂延迟 MAB，分析不涉及覆盖-激活机制，直接推广不可行
- **vs BLiN (Feng et al., 2022)**: 首个 batched Lipschitz bandit，但依赖即时/批量反馈且限于 $[0,1]^d$ 空间和轴对齐 cube 划分
- **vs Lancewicki et al. (2021)**: 无界延迟 MAB 通过分位数函数刻画遗憾，本文将此推广到连续空间并证明了 DLPP 的匹配下界

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次研究 Lipschitz bandit + 延迟反馈的交叉问题，但核心技术路线基于已有方法的组合扩展
- 实验充分度: ⭐⭐⭐⭐ 理论完整（上界+匹配下界），实验验证充分但场景偏合成
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，两种算法层次分明，定理陈述简洁优美
- 价值: ⭐⭐⭐⭐ 对 bandit 理论有扎实的基础贡献，填补了重要的理论空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Bi-Lipschitz Autoencoder With Injectivity Guarantee](bi-lipschitz_autoencoder_with_injectivity_guarantee.md)
- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](variance-dependent_regret_lower_bounds_for_contextual_bandits.md)
- [\[ICLR 2026\] Online Conformal Prediction with Adversarial Semi-bandit Feedback via Regret Minimization](online_conformal_prediction_with_adversarial_semi-bandit_feedback_via_regret_min.md)
- [\[ICLR 2026\] Data-to-Energy Stochastic Dynamics](data-to-energy_stochastic_dynamics.md)

</div>

<!-- RELATED:END -->
