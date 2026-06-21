---
title: >-
  [论文解读] Primal-Dual Policy Optimization for Linear CMDPs with Adversarial Losses
description: >-
  [ICLR 2026][强化学习][线性 CMDP] 本文针对损失对抗、代价随机的有限时段线性 CMDP，提出首个能同时保证亚线性 regret 与约束违反（均为 $\tilde{O}(K^{3/4})$）的原始-对偶策略优化算法，核心是用一类全新的「加权 LogSumExp softmax 策略」配合周期性策略混合与正则化对偶更新，把策略类的覆盖数和对偶变量同时压住。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "线性 CMDP"
  - "对抗损失"
  - "原始-对偶"
  - "策略优化"
  - "覆盖数"
---

# Primal-Dual Policy Optimization for Linear CMDPs with Adversarial Losses

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=baP3Dw7bWO](https://openreview.net/forum?id=baP3Dw7bWO)  
**代码**: 未公开  
**领域**: 强化学习 / 安全 RL / 约束 MDP 理论  
**关键词**: 线性 CMDP, 对抗损失, 原始-对偶, 策略优化, 覆盖数

## 一句话总结
本文针对损失对抗、代价随机的有限时段线性 CMDP，提出首个能同时保证亚线性 regret 与约束违反（均为 $\tilde{O}(K^{3/4})$）的原始-对偶策略优化算法，核心是用一类全新的「加权 LogSumExp softmax 策略」配合周期性策略混合与正则化对偶更新，把策略类的覆盖数和对偶变量同时压住。

## 研究背景与动机

**领域现状**：安全强化学习常用约束 MDP（CMDP）建模——智能体在最小化累积损失的同时，要保证累积代价不超预算。为了应付大状态空间，近年引入了线性函数逼近的「线性 CMDP」，假设转移核、损失、代价都对特征 $\phi(s,a)\in\mathbb{R}^d$ 线性，于是算法复杂度和 regret 都不再依赖状态数 $|S|$。

**现有痛点**：已有的线性 CMDP 工作（Ghosh et al. 2022/2024、Kitamura et al. 2025）全都假设**随机环境**——损失和代价要么固定、要么从固定分布采样。但自动驾驶里损失会因突发路况、恶劣天气剧烈跳变，服务机器人里损失随用户偏好、任务切换波动，这些场景里「损失信号固定」的假设过于理想。另一边，能处理对抗损失的工作（Qiu et al. 2020 等）又都局限在**表格（tabular）设定**，状态空间有限且小，无法扩展到大状态空间。

**核心矛盾**：把标准的原始-对偶策略优化搬到对抗线性 CMDP 时，会撞上一个技术死结——为了把对偶变量约束在一个紧致范围内（这是 regret 与违反分析的前提），常用手段是在优化策略前对策略做一次「扰动/混合」（policy mixing）。可这个加性的混合操作会**破坏策略优化的递归结构**，使得最终策略不再是普通 softmax，导致值函数类的**覆盖数（covering number）无法用现成工具分析**。覆盖数控制不住，高概率好事件就证不出来。

**本文目标**：设计一个原始-对偶策略优化算法，在「损失对抗（全信息反馈）+ 代价随机（bandit 反馈）」的有限时段线性 CMDP 上，同时拿到亚线性的 regret 和约束违反界。

**切入角度**：作者正面分析混合操作诱导出的新策略类，并意识到「混合频率」是一根同时牵动覆盖数和对偶变量的杠杆——混合越频繁，覆盖数越大（坏）但对偶变量越好控；混合越稀疏则相反。于是只要找到一个让二者都可控的混合节奏，再配一个能在稀疏混合下做漂移分析的对偶更新，问题就能解开。

**核心 idea**：用「加权 LogSumExp softmax 策略 + 周期性混合 + 正则化对偶更新」三件套，把覆盖数和对偶变量这一对此消彼长的量同时压到亚线性所需的量级。

## 方法详解

### 整体框架

算法在有限时段 CMDP 上按 episode $k=1,\dots,K$ 在线运行。每个 episode 开头对手任意选定损失 $f^k$（不向智能体公开其结构、只在交互中给全信息反馈），代价 $g^k$ 则从固定分布 i.i.d. 采样、只给 bandit 反馈。算法维护一个**原始变量（策略 $\hat\pi^k$）**和一个**对偶变量 $Y_k$**（对应约束 $V^\pi_{g,1}(s_1)\le b$ 的拉格朗日乘子），通过原始-对偶镜像下降交替更新。整段流程可拆成四个组件：epoch 初始化（按设计矩阵行列式翻倍触发，重置策略为均匀策略、对偶变量归零，并做**特征收缩**）→ 执行策略并用最小二乘估计 $Q$ 函数 → 带周期性混合的策略优化 → 正则化对偶更新。

理论贡献集中在后两个组件以及对它们的分析：混合操作让策略写成式 (1) 的加权 LogSumExp softmax，需要全新的覆盖数论证；周期性混合调节混合频率以平衡覆盖数与对偶变量；正则化对偶更新在稀疏混合下仍能做漂移分析、把 $Y_k$ 压到 $\tilde O(H^2/\gamma)$。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["episode k 开始<br/>对手选损失 f^k"] --> B["Epoch 初始化<br/>行列式翻倍则重置<br/>+ 特征收缩"]
    B --> C["执行策略 π̂^k<br/>最小二乘估 Q_f, Q_g"]
    C -->|每 K^{3/4} 步混合一次| D["1. 加权 LogSumExp softmax<br/>+ 2. 周期性策略混合"]
    D --> E["3. 正则化对偶更新<br/>Y_{k+1}"]
    E -->|进入 k+1| A
```

### 关键设计

**1. 加权 LogSumExp softmax 策略：让「混合 + 优化」诱导出的策略类有可分析的覆盖数**

把策略混合 $\hat\pi^{k-1}\leftarrow(1-\theta)\hat\pi^{k-1}+\theta\pi_{\text{unif}}$ 和策略优化 $\hat\pi^k\propto\hat\pi^{k-1}\exp(-\alpha\hat Q^{k-1})$ 串起来后，加性的混合打断了乘性优化的递归，最终策略不再是单一 softmax，而是

$$\hat\pi^k\propto\sum_{i=1}^{k}\zeta_i\exp\Big(-\alpha\sum_{j=k-i}^{k-1}\hat Q^j\Big),$$

即对不同长度的 $Q$ 估计部分和 $\sum_j\hat Q^j$ 分配不同权重 $\zeta_i$。作者称之为加权 LogSumExp softmax（因为它等价于对加权 LogSumExp 再取 softmax）。它比无混合时的 $\hat\pi^k\propto\exp(-\alpha\sum_{j}\hat Q^j)$ 更具表达力，但也更难分析：权重 $\{\zeta_i\}$ 不只依赖混合参数 $\theta$，还依赖 $Q$ 估计，不同策略的权重各不相同，**无法直接套用 LogSumExp 的标准 Lipschitz 性质**。作者的关键技术是证明一条关于混合次数 $n$ 的递归 $\|\hat\pi^n_1-\hat\pi^n_2\|_1\le c\|\hat\pi^{n-1}_1-\hat\pi^{n-1}_2\|_1+\|P^n_1-P^n_2\|_2$，反复展开后用参数差之和控制策略差，从而建立 Lipschitz 性，最终得到对数覆盖数 $\log N_\epsilon=\tilde O(n^2 d^2)$（$n$ 为最大混合次数，$d$ 为特征维度）。为简化策略表达、便于覆盖数论证，算法还借用 Cassel & Rosenberg (2024) 的**特征收缩** $\bar\phi=\phi\cdot\sigma(-\beta_w\|\phi\|_{(\Lambda)^{-1}}+\log K)$，用它代替对 $Q$ 估计的截断，避免 $Q$ 无界膨胀。

**2. 周期性策略混合：把混合频率当旋钮，平衡覆盖数与对偶变量**

由式 (2)，覆盖数随混合次数平方增长——若每个 episode 都混合，$n$ 可达 $K$，$\log N_\epsilon$ 涨到 $\tilde O(K^2 d^2)$，太大而无法得到亚线性界；可若混合太少，对偶变量又压不住（违反分析需要它）。覆盖数与对偶变量之间存在由混合频率主导的内在权衡。作者的解法是每隔 $K^{B}$ 个 episode 才混合一次（$B\in(0,1)$ 为周期参数），于是混合步数至多 $K^{1-B}$，代入 $\log N_\epsilon=\tilde O(n^2d^2)$ 即得可控的覆盖数。实现上（算法取 $K^{3/4}$ 为混合周期）：仅当 $k-k_e\equiv 0\bmod K^{3/4}$ 时执行 $\tilde\pi^k\leftarrow(1-\theta)\hat\pi^k+\theta\pi_{\text{unif}}$，其余 episode 直接 $\tilde\pi^k\leftarrow\hat\pi^k$。这把「混合带来的好处（控对偶）」和「混合带来的坏处（涨覆盖数）」解耦到一个可调频率上。

**3. 正则化对偶更新：在稀疏混合下仍能做漂移分析，把对偶变量压成紧界**

光有周期性混合还不够——稀疏混合让旧有的（每 episode 都混合的）漂移分析失效。作者给对偶更新加一项正则：

$$Y_{k+1}\leftarrow\Big[Y_k+\eta\big(\hat V^k_{g,1}(s_1)-b\big)+\underbrace{(-c_1 Y_k-c_2)}_{\text{正则项}}\Big]_+,$$

其中 $Y_k+\eta(\hat V^k_{g,1}(s_1)-b)$ 是标准的对偶在线梯度上升步，正则项 $-c_1Y_k-c_2$ 把对偶变量往 0 拉、保持紧致。正则项不是随手加的：漂移分析通常需要一个由决策变量与梯度构成的内积项 $\langle x_{t+1}-x_t,\nabla_t\rangle$，在策略优化里它对应 $\mathbb{E}_P[\langle\hat\pi^{k+1}-\hat\pi^k,\hat Q^k\rangle]$，但转移核 $P$ 未知、这一项无法直接放进算法；作者对它取下界，得到的恰好就是正则项，对应系数 $c_1=4\alpha\eta H^3$、$c_2=4\alpha\eta H^3+4\theta\eta H^2$。配合周期性混合，作者对混合 episode 处的对偶子序列 $\{Z_n\}$（$Z_n=Y_{k_e+nK^B}$）先做漂移分析，再外推到所有 $k$，最终证得 $Y_k=\tilde O(H^2/\gamma)$。与 Ghosh et al. (2022) 直接把对偶变量硬截断在 $2H/\gamma$ 不同，正则化的好处是它依赖损失代价固定的假设，因而能从随机推广到对抗设定；与 Qiu et al. (2020) 基于占用测度的对偶更新相比，期望代价对占用测度线性、对策略却非线性，故占用测度的漂移分析无法照搬到策略优化，这正凸显本文设计的难度与价值。

### 损失函数 / 训练策略

算法本质是带约束的在线镜像下降：原始侧用 KL 散度作为 Bregman 距离对策略做镜像下降步 $\hat\pi^{k+1}\propto\tilde\pi^k\exp(-\alpha(\hat Q^k_f+Y_k\hat Q^k_g))$，对偶侧用上面的正则化梯度上升。关键超参（$K$ 为 episode 数）：步长 $\alpha=H^{-1}K^{-3/4}$、对偶步长 $\eta=H^{-2}K^{-3/4}$、混合参数 $\theta=K^{-1}$、混合周期 $K^{3/4}$、乐观奖励参数 $\beta_b=\tilde O(K^{1/4}dH)$（其中 $K^{1/4}$ 来自覆盖数论证）。计算复杂度 $O(d^3HK+d^2|A|HK^2)$，与 $|S|$ 无关。

## 实验关键数据

本文是理论工作，数值实验只为验证理论趋势，没有横向 benchmark 表。实验在 Ghosh et al. (2022) 的 job-scheduling CMDP 上改造出对抗损失，跑 10 个随机种子、每个 $K=10^5$ 个 episode。

### 主结果（理论界）

| 指标 | 本文界 | 对比 | 意义 |
|------|--------|------|------|
| Regret$(K)$ | $\tilde O\big(\sqrt{d^3H^4}K^{3/4}+dH^3K^{3/4}+d^3H^4K^{1/2}+\tfrac{H^6}{\gamma^2}K^{1/4}\big)$ | 此前对抗线性 CMDP 无亚线性结果 | 首个亚线性 regret |
| Violation$(K)$ | $\tilde O\big(\tfrac{dH^5}{\gamma}K^{3/4}+\sqrt{d^3H^4}K^{3/4}+d^3H^4K^{1/2}\big)$ | 同上 | 首个亚线性约束违反 |
| 对偶变量 $Y_k$ | $\tilde O(H^2/\gamma)$ | Ghosh 截断在 $2H/\gamma$ | 经正则+漂移分析得紧界 |
| 覆盖数 $\log N_\epsilon$ | $\tilde O(n^2d^2)$ | softmax 仅需 Lipschitz | 加权 LogSumExp 的新论证 |

### 数值验证

| 观测 | 现象 | 结论 |
|------|------|------|
| Regret 曲线 | 随 $K$ 亚线性增长 | 与 $\tilde O(K^{3/4})$ 一致 |
| Violation 曲线 | 早期快速上升、最终收敛到 0 | 约束最终被满足 |

### 关键发现
- 两项界的主导项都是 $K^{3/4}$，比表格对抗 CMDP 常见的 $\sqrt K$ 差一截，差距正源于覆盖数论证引入的 $K^{1/4}$（体现在 $\beta_b$）——这是为换取大状态空间能力付出的代价。
- 周期性混合是把覆盖数从 $\tilde O(K^2d^2)$ 拉回可控量级的关键开关，缺它则无法亚线性。
- 约束违反曲线「先涨后收敛到 0」直观印证了对偶变量被正则项拉回 0 的机制。

## 亮点与洞察
- 把「混合频率」识别为同时牵动覆盖数与对偶变量的单一杠杆，是整篇论文的题眼——一个旋钮调两个看似无关的量，思路非常干净。
- 加权 LogSumExp softmax 的覆盖数递归 $\|\hat\pi^n_1-\hat\pi^n_2\|_1\le c\|\hat\pi^{n-1}_1-\hat\pi^{n-1}_2\|_1+\dots$ 是可迁移的技术：凡是「混合/扰动打断了 softmax 递归」的场景，都可借这条按混合次数归纳的 Lipschitz 思路。
- 正则项不是 ad-hoc 加的，而是对一个含未知转移核的内积项取下界自然得到，这种「从想要的分析倒推算法形式」的做法很优雅。

## 局限与展望
- regret/violation 停在 $\tilde O(K^{3/4})$，作者明确把能否做到 $\tilde O(\sqrt K)$ 列为开放问题。
- 代价是 bandit 反馈但**损失要求全信息反馈**，作者也把「损失也只给 bandit 反馈时能否设计样本高效算法」列为未来工作。
- 假设条件偏强：转移核/损失/代价均线性、需 Slater 条件、状态空间虽可大但仍设为有限；界对 $H$、$1/\gamma$ 的多项式依赖也较高（如 violation 含 $H^5/\gamma$）。
- 实验仅在单个改造的 job-scheduling 环境上、且只验证趋势，缺与其它对抗/约束 RL 算法的实测对比。

## 相关工作与启发
- **vs Ghosh et al. (2022)（随机线性 CMDP）**：都用原始-对偶 + 线性逼近，但对方损失代价固定、对偶变量靠硬截断在 $2H/\gamma$；本文损失对抗、用正则化对偶更新，分析不依赖损失固定，故能推广到对抗设定。
- **vs Qiu et al. (2020)（对抗表格 CMDP）**：对方用占用测度作原始变量、状态空间有限且小；本文用策略作原始变量并加线性逼近以处理大状态空间。期望代价对占用测度线性、对策略非线性，故占用测度的对偶漂移分析无法照搬，本文的策略侧设计填补了这一空白。
- **vs Cassel & Rosenberg (2024)（对抗线性无约束 MDP）**：借用其特征收缩技术来简化策略表达、控住 $Q$ 估计；本文的新意在于把它嵌入带约束的原始-对偶框架并配合周期性混合。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个对抗线性 CMDP 亚线性算法，加权 LogSumExp softmax 的覆盖数论证是全新技术。
- 实验充分度: ⭐⭐⭐ 理论为主，数值实验仅单环境验证趋势，无横向对比。
- 写作质量: ⭐⭐⭐⭐ 把三个技术贡献与其动机讲得清楚，挑战—技术对应明确。
- 价值: ⭐⭐⭐⭐ 为大状态空间的对抗安全 RL 打开理论缺口，技术可迁移到混合打断递归的其它设定。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Tension Between Optimality and Adversarial Robustness in Policy Optimization](on_the_tension_between_optimality_and_adversarial_robustness_in_policy_optimizat.md)
- [\[ICLR 2026\] SHAPO: Sharpness-Aware Policy Optimization for Safe Exploration](shapo_sharpness-aware_policy_optimization_for_safe_exploration.md)
- [\[ICLR 2026\] DuPO: Enabling Reliable Self-Verification via Dual Preference Optimization](dupo_enabling_reliable_self-verification_via_dual_preference_optimization.md)
- [\[NeurIPS 2025\] Global Convergence for Average Reward Constrained MDPs with Primal-Dual Actor-Critic](../../NeurIPS2025/reinforcement_learning/global_convergence_for_average_reward_constrained_mdps_with_primal-dual_actor_cr.md)
- [\[ICLR 2026\] A Unifying View of Coverage in Linear Off-Policy Evaluation](a_unifying_view_of_coverage_in_linear_off-policy_evaluation.md)

</div>

<!-- RELATED:END -->
