---
title: >-
  [论文解读] MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation
description: >-
  [ICML 2026][测试时自适应 / PAC-Bayes 理论 / 不确定性量化][PAC-Bayes 界] 论文为 test-time adaptation 提供了第一份"目标风险 ≤ 源经验风险 + KL 复杂度 + MMD 分布偏移项"的 PAC-Bayes 上界，并把 MMD-球解读为 Walley 意义下的 credal set，从而用"上下风险区间"自然分离 aleatoric 与 epistemic 不确定性，给出"何时应当 adapt、何时该 abstain"的可计算判据。
tags:
  - "ICML 2026"
  - "测试时自适应 / PAC-Bayes 理论 / 不确定性量化"
  - "PAC-Bayes 界"
  - "MMD"
  - "credal set"
  - "测试时自适应"
  - "认知不确定性"
---

# MMD-Balls as Credal Sets: A PAC-Bayesian Framework for Epistemic Uncertainty in Test-Time Adaptation

**会议**: ICML 2026  
**arXiv**: [2605.21783](https://arxiv.org/abs/2605.21783)  
**代码**: 论文未公开  
**领域**: 测试时自适应 / PAC-Bayes 理论 / 不确定性量化  
**关键词**: PAC-Bayes 界, MMD, credal set, 测试时自适应, 认知不确定性

## 一句话总结
论文为 test-time adaptation 提供了第一份"目标风险 ≤ 源经验风险 + KL 复杂度 + MMD 分布偏移项"的 PAC-Bayes 上界，并把 MMD-球解读为 Walley 意义下的 credal set，从而用"上下风险区间"自然分离 aleatoric 与 epistemic 不确定性，给出"何时应当 adapt、何时该 abstain"的可计算判据。

## 研究背景与动机

**领域现状**：以 TENT、EATA、SAR、MEMO 为代表的 test-time adaptation (TTA) 方法在分布漂移下显著提升精度，做法基本都是用测试批次的统计量去微调 BN/参数。

**现有痛点**：所有这些 TTA 方法都没有任何形式上的保证——既不告诉你"偏移到什么程度时模型还能信"，也不告诉你"何时根本不该 adapt"。在自动驾驶、医学影像等安全关键场景里，模型可能正在静默退化，但 TTA 自己浑然不觉。

**核心矛盾**：现有的 predictive uncertainty 工具（贝叶斯网、ensemble、conformal）把 aleatoric（数据本身的噪声）和 epistemic（对分布的未知）混在一起报，而 TTA 缺的恰恰是"分布层面的认知不确定性"。同时，过去 PAC-Bayes 用于 domain adaptation 时依赖 NP-hard 的 $\mathcal{H}$-divergence（Germain 2013），实际不可计算。

**本文目标**：能否给出一个 (i) 由可计算量 MMD 主导、(ii) 提供有限样本上界、(iii) 自然分离 epistemic 与 aleatoric、(iv) 能告诉你何时该 adapt 的统一理论框架。

**切入角度**：作者注意到 MMD-ball $\mathcal{C}_\varepsilon(P_s)=\{Q:\mathrm{MMD}(P_s,Q)\le \varepsilon\}$ 在数学上恰恰满足 Walley credal set 的全部要求——它就是"在分辨率 $\varepsilon$ 下与源分布不可区分的所有分布构成的集合"。

**核心 idea**：用 RKHS-Lipschitz 假设把"目标分布与源分布的损失差"上界化为 $L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t)$，然后把这一项接到经典 PAC-Bayes 界上；再把 $\varepsilon$ 替换 MMD 取 sup，就得到 credal set 上的最坏风险界，自然给出 lower/upper risk 区间。

## 方法详解

### 整体框架
框架不是"算法"而是"理论 + 决策准则"，可以拆成 4 个相互衔接的部件：

1. **PAC-Bayes 上界（Theorem 1）**：在 covariate shift + RKHS-Lipschitz loss 假设下，把目标风险 $R_{P_t}(\rho)$ 拆成"源经验风险 + KL 复杂度 + MMD 偏移惩罚"三项。
2. **有限样本版本（Theorem 3）**：用无偏 MMD 估计量 $\widehat{\mathrm{MMD}}_u$ 替换种群 MMD，再用 Sutherland/Tolstikhin 的次高斯集中给出闭式宽度 $\varepsilon_{m,n}(\delta)$。
3. **Credal set 几何（Definition 5 + Proposition 7 + Corollary 9）**：把 MMD-球看成 credal set，给出 worst-case 风险 $\overline{R}_\varepsilon(\rho)$ 和最佳风险 $\underline{R}_\varepsilon(\rho)$，imprecision 宽度 $\overline{R}_\varepsilon-\underline{R}_\varepsilon$ 直接量化 epistemic 不确定性。
4. **测地保持（Proposition 10 + Corollary 11）**：在 RKHS 几何下证明源-目标邻域的测地距离差被 $\sqrt{2\gamma}\,C_W\,\mathrm{MMD}(P_s,P_t)$ 控制，给"kernel-guided adaptation 保护稀有类"提供理论解释。

四件套合起来构成"分布层面的 epistemic intelligence"：监控 MMD → 算 credal 区间 → 触发 adapt / abstain。

### 关键设计

**1. PAC-Bayes + MMD 偏移惩罚：把"分布偏移量"显式写进泛化上界（Theorem 1 / 3）**

TTA 方法之所以没保证，根源在于过去能上界化"源-目标风险差"的工具要么不可算（$\mathcal{H}$-divergence 是 NP-hard），要么不带有限样本版本。本文换了一把更顺手的尺子：MMD。在 Assumption 1（条件期望损失落在 RKHS 里且范数有界，$L(w,\cdot)\in\mathcal{H}$、$\|L(w,\cdot)\|_\mathcal{H}\le L_\mathcal{H}$）下，用 reproducing property 加一步 Cauchy-Schwarz 就能把风险差压成一个 MMD 项 $|R_{P_t}(\rho)-R_{P_s}(\rho)|\le L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t)$，再把它叠到经典 McAllester PAC-Bayes 界上，得到 TTA 场景下第一份显式带分布偏移惩罚的上界：

$$R_{P_t}(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{\frac{\mathrm{KL}(\rho\|\pi)+\log(2\sqrt{n}/\delta)}{2n}}+L_\mathcal{H}\cdot \mathrm{MMD}(P_s,P_t).$$

Theorem 3 进一步把不可观测的种群 MMD 换成无偏估计 $\widehat{\mathrm{MMD}}_u$，再补一个集中宽度 $\varepsilon_{m,n}=\sqrt{2\log(2/\alpha)/\min(m,n)}$，整条界就全部可计算，复杂度 $O((m+n)^2)$，速率 $O(1/\sqrt{n})$ 达 minimax 最优。这条界的好处不只是"能算"：MMD 项是线性增长，所以偏移变大时界是平滑变松而不是指数崩塌——这正是 epistemic uncertainty 该有的行为，模型在偏移加剧时只是"越来越不确定"而非"突然失效"。

**2. MMD-球当成 credal set：把点估计升级为 lower-upper 风险区间，干净分离 aleatoric 与 epistemic**

光有一条上界还不够，安全场景真正想知道的是"不确定性有多大、其中多少来自数据噪声、多少来自分布未知"。本文的关键观察是：MMD-球 $\mathcal{C}_\varepsilon(P_s)=\{Q:\mathrm{MMD}(P_s,Q)\le \varepsilon\}$ 在数学上恰好就是一个 credal set——在分辨率 $\varepsilon$ 下与源分布不可区分的全部分布。由特征核的线性性可证它凸且弱闭（Lemma 6），于是可以对整个集合取最坏风险：

$$\sup_{Q\in\mathcal{C}_\varepsilon(P_s)}R_Q(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{\frac{\mathrm{KL}+\log}{2n}}+L_\mathcal{H}\varepsilon,$$

inf 方向用 Germain 的 PAC-Bayes 下界对称地推出最佳风险 $\underline{R}_\varepsilon$。两端一减，imprecision 宽度 $\overline{R}_\varepsilon-\underline{R}_\varepsilon\le 2\sqrt{(\mathrm{KL}+\log)/2n}+2L_\mathcal{H}\varepsilon$ 把两种不确定性物理地拆开了：前一项随源样本 $n$ 衰减，是"看的数据还不够多"的估计不确定性（aleatoric 侧）；后一项随 $\varepsilon$ 线性增长，是"目标分布到底偏了多远"的分布不确定性（epistemic 侧）。这是第一次把 Walley 的行为派 imprecise probability 与 PAC-Bayes 接通，并给出可操作的 epistemic-vs-aleatoric 分解，正面回应了 Hüllermeier & Waegeman 2021 对 ML 不确定性度量混为一谈的批评。

**3. RKHS 测地保持：从几何上解释为什么 MMD-bounded adaptation 比 entropy 最小化更能保护稀有类**

实践里 kernel-guided adaptation 似乎比 TENT 那种 entropy 最小化更不容易抹掉少数类，但一直停在直觉。本文把它升级成几何定理。在 Assumption 2（encoder 可因子化为有界线性层乘 RKHS 特征图 $f_\theta=W\cdot \phi_\theta$、$\|W\|_{op}\le C_W$）下，对 RBF 核做局部线性化 $d_k(x,y)=\sqrt{2\gamma}\|f_\theta(x)-f_\theta(y)\|+O(\bar\epsilon^2)$，再走一步反三角不等式，就得到源-目标邻域的测地距离漂移同样被 MMD 控制：

$$\big|\mathbb{E}_{y\sim P_s}[d_k(x_i,y)]-\mathbb{E}_{y\sim P_t}[d_k(x_i,y)]\big|\le \sqrt{2\gamma}\,C_W\,\mathrm{MMD}(P_s,P_t)+O(\bar\epsilon^2).$$

关键在于这个 bound 与类别频率无关——它不区分"这块区域有多少样本"。所以稀有类那种"小但结构紧凑"的局部几何天然被保护住；反观 entropy 最小化会把低密度区误判成高熵区然后压平，恰恰把稀有类抹掉。这一步让 epistemic 控制不只约束风险数值，还约束了表示几何本身。

### 损失函数 / 训练策略
论文不提新训练算法，所有结果都建在两个关键假设上：(1) Assumption 1 要求 $L(w,\cdot)$ 落在 RKHS 中且有界范数（对 softmax + RBF 核可借核普适性近似成立）；(2) Assumption 2 要求 encoder 可分解为有界线性 + RKHS 特征图（在 NTK regime、显式 MMD 正则、谱归一化下近似成立）。作者在第 8 节讨论了把 (1) 放宽为 $\mathbb{E}_{w\sim\rho}[\|L(w,\cdot)\|_\mathcal{H}]\le L_\mathcal{H}$ 的方向。

## 实验关键数据

### 主实验
论文是理论工作，**没有完整的实验表**——所有"数据"以定理形式给出。这里把它们整理成对照表以便阅读。

| 不等式 | 主要项 | 说明 |
|---|---|---|
| Theorem 1（种群 MMD） | $R_{P_t}(\rho)\le \hat{R}_{P_s}(\rho)+\sqrt{(\mathrm{KL}+\log(2\sqrt n/\delta))/(2n)}+L_\mathcal{H}\,\mathrm{MMD}(P_s,P_t)$ | 给出 TTA 第一份显式 MMD 偏移惩罚的 PAC-Bayes 界 |
| Theorem 3（有限样本） | 上式 + $L_\mathcal{H}\,(\widehat{\mathrm{MMD}}_u+\varepsilon_{m,n}(\delta/2))$，$\varepsilon_{m,n}=\sqrt{2\log(2/\alpha)/\min(m,n)}$ | 全部可计算，$O((m+n)^2)$ |
| Proposition 7（worst-case） | $\sup_{Q\in\mathcal{C}_\varepsilon(P_s)} R_Q(\rho)\le \hat R_{P_s}(\rho)+\sqrt{\cdots}+L_\mathcal{H}\varepsilon$ | 整个 credal set 上的最坏风险有界 |
| Corollary 9（区间宽度） | $\overline{R}_\varepsilon-\underline{R}_\varepsilon\le 2\sqrt{\cdots/2n}+2L_\mathcal{H}\varepsilon$ | epistemic 不确定性 = 估计项 + 偏移项 |
| Proposition 10（几何） | $|\mathbb{E}_{P_s}[d_k]-\mathbb{E}_{P_t}[d_k]|\le \sqrt{2\gamma}C_W\,\mathrm{MMD}+O(\bar\epsilon^2)$ | MMD 控制 RKHS 测地距离漂移 |

### 消融实验

| 假设 / 设置 | 影响 | 说明 |
|---|---|---|
| 关闭 RKHS-Lipschitz（Assumption 1） | Theorem 1 失效 | 偏移项不再有线性 MMD 上界 |
| 用 posterior mean ρ 代替分布 ρ | KL 项退化 | PAC-Bayes 复杂度项变为 0，但失去 prior 正则化 |
| 把 MMD 换成 $\mathcal{H}$-divergence | 回到 Germain 2013 | 上界仍成立但不可计算 |
| 关闭 covariate shift 假设 | $L(w,x)$ 在源/目标不再一致 | 偏移项需要联合 $(x,y)$ 的 MMD，需扩展核 |
| 核非 characteristic | $\varepsilon=0$ 不再蕴含 $Q=P_s$ | credal set 退化 |

### 关键发现
- 当 $\mathrm{MMD}(P_s,P_t)\to 0$，界精确恢复经典 PAC-Bayes，没有引入冗余；分布偏移大时，界"平滑变松"而非"突然失效"，这正是 epistemic uncertainty 应有的性质。
- 决策准则：给定容忍 $r_{\max}$，当 $\underline{R}_\varepsilon(\rho)>r_{\max}$ 时连最好的目标分布都不可接受——应当 abstain；当 $\overline{R}_\varepsilon(\rho)<r_{\max}$ 时即使最坏分布也安全——无需 adapt；只有当 $\overline{R}_\varepsilon>r_{\max}>\underline{R}_\varepsilon$ 时 adapt 才有意义。这把"adapt 与否"从经验问题转成可计算的判定。
- $\varepsilon$ 可以用 MMD 两样本检验的渐近 null 分布在水平 $\alpha$ 处校准：拒绝 $H_0:P_t=P_s$ 等价于 $\widehat{\mathrm{MMD}}_u>\varepsilon_\alpha$，因此 credal set 宽度直接对应"对零假设的证据强度"。

## 亮点与洞察
- 把 Walley 1991 年的 imprecise probability 与 PAC-Bayes generalization 在 MMD-球这个具体几何对象上接通，是少见的"60 年代决策理论 × 90 年代统计学习 × 2010 年代核方法"的三方融合，且关键步骤只用 reproducing property + Cauchy-Schwarz，证明短得不可思议。
- "$\varepsilon$ 增大时界线性而不是指数变松"是非常宝贵的结构性质，说明 MMD 是分布偏移的"良性几何量"——这一点对未来设计 distribution-shift-aware loss / OOD detector 都有指导意义。
- 测地保持引理给"为什么 entropy minimization 会杀稀有类"提供形式化解释：entropy 在低密度区天然鼓励压平，而 MMD 控制下的 adaptation 与类别频率无关，于是稀有类被几何上保护下来。

## 局限与展望
- Assumption 1 对深网 + softmax loss 只是"informal 由 RBF 普适性支持"，并未给出严格构造。作者建议放宽为期望意义下的 $\mathbb{E}_{w\sim\rho}[\|L(w,\cdot)\|_\mathcal{H}]\le L_\mathcal{H}$，但仍是开放问题。
- Assumption 2 对一般 ResNet / ViT 只在 NTK 极限或显式 MMD 正则下近似成立，针对具体架构的紧界仍待研究。
- MMD 收敛率 $O(1/\sqrt n)$ 虽 minimax 最优，但实际可能松；adaptive kernel selection / bandwidth tuning 能给出更紧的数据相关速率。
- 整篇论文**没有实证实验**：所有定理都没有在真实 TTA benchmark（CIFAR-10-C、ImageNet-C 等）上验证 bound 的紧度与决策准则的有效性，这是直接可挖的下一步。
- 与 conformal prediction 的联动方案 $\alpha(\varepsilon)=\alpha_0+g(\varepsilon)$ 只是 Appendix F 的草图，需要严格校准。

## 相关工作与启发
- **vs Germain et al. 2013（PAC-Bayes domain adaptation）**：他们用 $\mathcal{H}$-divergence，NP-hard 且无 finite-sample 版本；本文用 MMD，$O((m+n)^2)$ 可算且自带集中不等式。
- **vs TENT / EATA / SAR（TTA 算法）**：这些方法靠经验技巧（entropy 最小化、sharpness 正则、memory bank）提点，没有任何上界；本文给出何时该 adapt 的形式化判据，与它们正交可叠加。
- **vs 贝叶斯神经网 / Ensemble**：predictive uncertainty 把 aleatoric 与 epistemic 混在一起报；本文用 credal set 宽度显式分离两者。
- **vs Conformal prediction**：conformal 给个体级别的 prediction set 覆盖保证，但不约束总体风险；本文给分布级别的风险区间，两者可组合为"用 $\varepsilon$ 自适应调 conformal 覆盖率"。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一次把 PAC-Bayes、kernel mean embedding、Walley credal set 三者打通到 TTA 场景，组合本身就是新结构。
- 实验充分度: ⭐⭐ 全篇没有任何实证验证，是纯理论 short paper（疑似 EIML workshop track）。
- 写作质量: ⭐⭐⭐⭐ 定理-证明草图-备注的结构清晰，符号克制，关键证明只用 reproducing property + Cauchy-Schwarz，对读者非常友好。
- 价值: ⭐⭐⭐⭐ 给"何时该 adapt / 何时该 abstain"提供了可计算判据；如果配上实验和校准，可直接服务于安全关键 ML 部署。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] On the Learnability of Test-Time Adaptation: A Recovery Complexity Perspective](on_the_learnability_of_test-time_adaptation_a_recovery_complexity_perspective.md)
- [\[ICLR 2026\] Epistemic Uncertainty Quantification To Improve Decisions From Black-Box Models](../../ICLR2026/learning_theory/epistemic_uncertainty_quantification_to_improve_decisions_from_black-box_models.md)
- [\[ICLR 2026\] Efficient Credal Prediction through Decalibration](../../ICLR2026/learning_theory/efficient_credal_prediction_through_decalibration.md)
- [\[ICLR 2026\] Expressive Power of Implicit Models: Rich Equilibria and Test-Time Scaling](../../ICLR2026/learning_theory/expressive_power_of_implicit_models_rich_equilibria_and_test-time_scaling.md)
- [\[ICML 2026\] On Regret Bounds of Thompson Sampling for Bayesian Optimization](on_regret_bounds_of_thompson_sampling_for_bayesian_optimization.md)

</div>

<!-- RELATED:END -->
