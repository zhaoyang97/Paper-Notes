---
title: >-
  [论文解读] Conformal Robustness Control: A New Strategy for Robust Decision
description: >-
  [ICLR 2026][优化/理论][保形预测] 针对"用保形预测做鲁棒决策"中"覆盖约束过保守"的痛点，本文提出 Conformal Robustness Control（CRC），把预测集的构造**直接放到显式鲁棒性约束下优化**（而非要求覆盖率），用光滑代理 + 拉格朗日交替梯度求解，并给出非渐近理论保证与测试时有限样本校准，在组合投资、股票、电池储能等任务上拿到更低的风险证书和决策损失，同时把鲁棒性精准卡在目标水平。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "保形预测"
  - "条件鲁棒优化"
  - "风险敏感决策"
  - "预测集优化"
  - "有限样本保证"
---

# Conformal Robustness Control: A New Strategy for Robust Decision

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bt4Ahpemmi](https://openreview.net/forum?id=bt4Ahpemmi)  
**代码**: 待确认  
**领域**: 鲁棒优化 / 保形预测 / 决策理论  
**关键词**: 保形预测, 条件鲁棒优化, 风险敏感决策, 预测集优化, 有限样本保证

## 一句话总结
针对"用保形预测做鲁棒决策"中"覆盖约束过保守"的痛点，本文提出 Conformal Robustness Control（CRC），把预测集的构造**直接放到显式鲁棒性约束下优化**（而非要求覆盖率），用光滑代理 + 拉格朗日交替梯度求解，并给出非渐近理论保证与测试时有限样本校准，在组合投资、股票、电池储能等任务上拿到更低的风险证书和决策损失，同时把鲁棒性精准卡在目标水平。

## 研究背景与动机
**领域现状**：在组合投资、医疗诊断、交通规划这类风险敏感场景里，决策者要在结果 $Y$ 未知时选一个决策 $z(X)$，希望决策损失 $\phi(Y, z(X))$ 不超过某个风险证书 $r(X)$ 的概率足够高，即满足 $(1-\alpha)$ 级鲁棒性约束 $P\{\phi(Y, z(X)) \le r(X)\} \ge 1-\alpha$，同时尽量把 $r(X)$ 压低以提高效率。条件鲁棒优化（Conditional Robust Optimization, CRO）是当前主流框架：先构造一个预测集 $U(X)$，再解 minmax 问题 $z_U(X) := \arg\min_{z\in\mathcal Z}\max_{y\in U(X)} \phi(y,z)$，对应风险证书 $r_U(X) := \max_{y\in U(X)}\phi(y, z_U(X))$。

**现有痛点**：为了满足鲁棒性目标，主流做法（Johnstone & Cox 2021；Sun et al. 2023）用**保形预测**把 $U(X)$ 构造成覆盖率 $P\{Y\in U(X)\} \ge 1-\alpha$ 的预测集，再代入 minmax。问题在于——覆盖是鲁棒性的**充分条件，但不是必要条件**。强行让预测集"包住真值"，往往比"保证决策不翻车"要求得更多，于是预测集偏大、风险证书偏高、决策过度保守。论文 Figure 1 给了一个直观例子：CRO 在名义 90% 鲁棒水平下实际把鲁棒性顶到了 98%、覆盖率 90%，风险证书 1.93；而真正只需要 90% 鲁棒性时，覆盖率只要 56% 就够，风险证书可降到 1.25。

**核心矛盾**：覆盖约束 $\Rightarrow$ 鲁棒性约束是单向蕴含，中间那一截"多出来的覆盖"全部转化成了不必要的保守和效率损失。

**本文目标**：把约束从"覆盖率"换成"鲁棒性"本身——直接在 $P\{\phi(Y,z_U(X))\le r_U(X)\}\ge 1-\alpha$ 约束下最小化期望风险证书 $E[r_U(X)]$，并解决两个随之而来的问题：(1) 这个约束含指示函数、不可微，怎么优化；(2) 怎么给出有限样本下的鲁棒性保证。

**切入角度**：既然真正要满足的是鲁棒性约束，那就别绕道覆盖，把预测集本身当成优化变量、在鲁棒性约束下端到端地学。

**核心 idea**：用"显式鲁棒性约束"替代"覆盖约束"来构造预测集（Conformal Robustness Control），在不牺牲鲁棒性的前提下显著提高决策效率。

## 方法详解

### 整体框架
CRC 的目标是求解风险规避决策策略优化（RA-DPO）问题：
$$\min_{z(\cdot), r(\cdot)} E[r(X)] \quad \text{s.t.} \quad P\{\phi(Y, z(X)) \le r(X)\} \ge 1-\alpha.$$
直接在任意函数形式的 $z(\cdot)$、$r(\cdot)$ 上优化很难，CRO 框架的做法是引入预测集 $U(\cdot)$，让决策和风险证书都由它派生，于是问题转写为只对预测集优化：
$$\min_{U(\cdot)} E[r_U(X)] \quad \text{s.t.} \quad P\{\phi(Y, z_U(X)) \le r_U(X)\} \ge 1-\alpha. \tag{4}$$
关键的一步是论文证明了这一转写**不损失最优性**（Theorem 3.1）：RA-DPO 的最优期望风险证书与问题 (4) 的最优值相等。这意味着"只优化预测集"和"在全部决策/证书函数上优化"是等价的，给后续把预测集参数化提供了理论底气。

随后把预测集参数化为 $U_\theta(\cdot)$（如 box、椭球），问题变成对参数 $\theta$ 的约束优化；再用样本平均近似目标和约束，得到经验版本，用光滑代理把不可微的约束变成可梯度优化的形式，交替梯度下降求解（Algorithm 1）。训练完得到的预测集只有渐近鲁棒性，论文再加一道**测试时校准**（Cal-CRC, Algorithm 2），借全保形预测把单个测试点的鲁棒性提升到**有限样本**精确成立。整条流水线就是：参数化预测集 → 鲁棒性约束下经验优化（含理论保证）→ 测试时保形校准。

### 关键设计

**1. 用鲁棒性约束替代覆盖约束：从源头去保守化**

这是全文的立足点，针对的正是"覆盖是充分非必要条件"带来的过度保守。传统 CRO（RA-CPO）在覆盖约束 $P\{Y\in U(X)\}\ge 1-\alpha$ 下优化期望风险证书；Kiyani et al. (2025) 给出过覆盖约束下的最优预测集闭式解，但它依赖最小化 VaR 函数，在连续决策空间 $\mathcal Z$ 下一般不可解（VaR 问题难处理）。CRC 把约束直接换成鲁棒性约束（问题 (4) 中的 $P\{\phi(Y, z_U(X))\le r_U(X)\}\ge 1-\alpha$）。因为鲁棒性约束比覆盖约束**更松**（覆盖 $\Rightarrow$ 鲁棒，反之不然），可行域更大，最优风险证书自然更低——论文在 Appendix B.3 进一步证明在同样参数化下问题 (5) 给出的风险证书严格不高于覆盖约束版本。直觉上：覆盖要求"真值落进集合"，鲁棒只要求"决策损失被证书罩住"，后者允许预测集更小、决策更激进，效率因此提升。

**2. 预测集参数化 + 经验优化：把"学预测集"落成一个可解的优化问题**

为了在连续决策空间下可解，CRC 不直接优化抽象的 $U(\cdot)$，而是参数化为 $U_\theta(\cdot)$。回归场景下用两类常见形状：box 集 $U_\theta(x)=\{y: h^{lo}_\theta(x)\le y\le h^{hi}_\theta(x)\}$（逐分量上下界），和椭球集 $U_\theta(x)=\{y: (y-\mu_\theta(x))^\top \Sigma_\theta^{-1}(x)(y-\mu_\theta(x))\le 1\}$（能刻画分量间相关性），附录还给了多面体集的例子。给定 i.i.d. 标注数据 $D_n=\{(X_i,Y_i)\}_{i=1}^n$，把问题 (5) 的期望和概率都用样本平均替换，得到经验问题：
$$\hat\theta = \arg\min_{\theta} \frac1n\sum_{i=1}^n r_\theta(X_i) \quad \text{s.t.} \quad \frac1n\sum_{i=1}^n \mathbf 1\{\phi(Y_i, z_\theta(X_i))\le r_\theta(X_i)\}\ge 1-\alpha. \tag{6}$$
这一步把"在鲁棒性约束下学预测集"具体化成一个标准的经验约束优化，正是"CRC"这个名字（Conformal Robustness Control，强调显式鲁棒性约束、区别于覆盖控制）的来源。

**3. 光滑代理 + 拉格朗日交替梯度，并配非渐近理论保证**

问题 (6) 的约束含指示函数 $\mathbf 1\{\cdot\}$，关于 $\theta$ 不光滑、不能直接梯度优化。CRC 取拉格朗日 $L(\lambda;\theta)=f(\theta)+\lambda g(\theta)$，其中 $f(\theta)=\frac1n\sum_i r_\theta(X_i)$（若 CRO 子问题能转成凸规划，则可微，梯度用隐式微分工具算），$g(\theta)=1-\alpha-\frac1n\sum_i \mathbf 1\{\phi(Y_i,z_\theta(X_i))\le r_\theta(X_i)\}$。把指示函数换成高斯误差函数光滑代理 $\tilde{\mathbf 1}\{a\le b\}=\frac12(1+\mathrm{erf}(\frac{b-a}{\sqrt2\sigma}))$（$\sigma$ 控制光滑度），得到光滑约束 $\tilde g(\theta)$，于是 $\min_\theta\max_{\lambda\ge0}\tilde L(\lambda;\theta)$ 可用交替梯度求解：对 $\theta$ 做几步梯度下降，对 $\lambda$ 做投影梯度上升 $\lambda\leftarrow\max\{0,\lambda+\eta\tilde g(\theta)\}$（Algorithm 1）。理论上，论文用覆盖数（covering number）刻画参数类复杂度，给出两条非渐近结论：Theorem 3.2（鲁棒性 gap）保证 $P\{\phi(Y,z_{\hat\theta}(X))\le r_{\hat\theta}(X)\mid D_n\}\ge 1-\alpha-\Delta_n$，其中 $\Delta_n=5\sqrt{\frac{\log(2N(\Theta,\|\cdot\|_\infty,n^{-1}))+\log n}{2n}}+\frac{4(L_\phi L_z+L_r)\rho_0}{n}$；Theorem 3.3（风险证书最优性）保证 $\hat\theta$ 的期望风险证书与放松到 $1-\alpha+\Delta_n$ 水平的最优解之差以同阶速率收敛。对 $d$ 维参数空间，覆盖数 $\approx n^d$，两者都以 $O(\sqrt{d\log n/n})$ 收敛到 0。

**4. Cal-CRC：测试时有限样本鲁棒校准**

Theorem 3.2 只给渐近鲁棒性，但实际部署要对**具体测试点** $X_{n+1}$ 给出有限样本保证。CRC 把标注数据切成训练集 $D_{train}$ 和校准集 $D_{cal}$，先在 $D_{train}$ 上跑 Algorithm 1 得到 $U_{\hat\theta_0}(\cdot)$，再用全保形预测（full conformal）在 $D_{cal}$ 和 $X_{n+1}$ 上校准。校准不重训全部参数 $\theta$（太贵且没必要），而是只调一个半径参数 $t\in\mathbb R^+$ 来控制集合大小：box 用 $U_{\theta,t}(x)=\{y: h^{lo}_\theta(x)-t\le y\le h^{hi}_\theta(x)+t\}$，椭球用 $U_{\theta,t}(x)=\{y:(y-\mu_\theta(x))^\top\Sigma_\theta^{-1}(x)(y-\mu_\theta(x))\le t\}$，二者都构成嵌套集族（$t_1\le t_2 \Rightarrow U_{\theta,t_1}\subseteq U_{\theta,t_2}$）。对每个候选标签 $y$，用增广校准集算出满足经验鲁棒率 $\ge 1-\alpha$ 的最小阈值 $\hat t^y$（式 (7)），据此判定 $y$ 是否进入校准预测集 $U_{Cal}(X_{n+1})$，最后在该集合上做 minmax 得决策（Algorithm 2）。Theorem 4.1 证明：只要标注数据与测试数据 i.i.d.（可交换性），就有有限样本鲁棒性 $P\{\phi(Y_{n+1}, z_{U_{Cal}}(X_{n+1}))\le r_{U_{Cal}}(X_{n+1})\}\ge 1-\alpha$，与经典保形预测同样只依赖可交换性。遍历全部 $y$ 的开销可用离散化技巧规避。

### 损失函数 / 训练策略
训练目标即式 (6) 的经验约束优化，实现上用光滑拉格朗日 $\tilde L(\lambda;\theta)=f(\theta)+\lambda\tilde g(\theta)$ 做 $\theta$-梯度下降 / $\lambda$-投影梯度上升的交替更新（Algorithm 1）。需满足两组温和正则条件：Condition 3.1（$\phi$ 关于 $z$ Lipschitz、$z_\theta$ 与 $r_\theta$ 关于 $\theta$ Lipschitz 且 $r_\theta$ 有界），Condition 3.2（$V_\theta(X,Y)=\phi(Y,z_\theta(X))-r_\theta(X)$ 的密度一致有界）——前者在 CRO 子问题可转为光滑凸规划时由 KKT 条件与隐函数定理保证，后者是保形预测文献中常见的浓度假设。测试时叠加 Cal-CRC（Algorithm 2）做样本切分 + 保形校准。

## 实验关键数据

### 主实验
任务：(i) 合成组合投资优化；(ii) 真实美股组合优化；(iii) 电池储能控制。基线：CRO（带保形预测集）、E2E（端到端最小化期望风险证书）；CRC 应用到椭球集记为 CRC-E、box 集记为 CRC-B（CRO/E2E 同理）。评价指标：风险证书（$r_U(X)$ 均值）、决策损失（$\phi(Y,z_U(X))$ 均值）、鲁棒性（损失 $\le$ 证书的样本占比）、覆盖率。

US 股票问题（损失 $\phi(y,z)=-y^\top z$，每次随机选 15 只股票，多次重复）：

| 方法 | 风险证书 ($\alpha{=}0.1$) | 决策损失 | 鲁棒性(%) | 风险证书 ($\alpha{=}0.2$) | 决策损失 | 鲁棒性(%) |
|------|------|------|------|------|------|------|
| **CRC-B** | **1.160** | -0.055 | 90.9 | **0.731** | -0.059 | 80.6 |
| CRO-B | 3.794 | -0.051 | 99.9 | 3.017 | -0.054 | 99.5 |
| E2E-B | 2.129 | -0.046 | 96.7 | 1.512 | -0.041 | 92.7 |
| **CRC-E** | **1.028** | -0.077 | 90.8 | **0.701** | -0.075 | 80.6 |
| CRO-E | 6.345 | -0.069 | 99.9 | 6.195 | -0.046 | 99.8 |
| E2E-E | 4.995 | -0.071 | 98.6 | 4.503 | -0.070 | 96.4 |

CRC 在风险证书和决策损失上全面占优，且鲁棒性精准卡在名义目标 $1-\alpha$ 附近（如 $\alpha=0.1$ 时约 90.8%–90.9%）；CRO/E2E 则把鲁棒性顶到 96%–99.9%，是典型的过度保守——多出来的鲁棒性以更高风险证书（CRO-E 高达 6.345，是 CRC-E 的 6 倍多）为代价。

### 消融实验
论文未采用传统"逐模块去除"式消融，而是通过**变名义水平 $\alpha$** 和**变样本量 $n$** 的对照来验证方法成色（合成数据，Figure 2/3）：

| 对照设置 | 关键观察 | 说明 |
|------|---------|------|
| 变 $\alpha$（$n{=}1500$ 固定） | CRC-E 风险证书/决策损失全程低于基线 | 各鲁棒水平下都更高效 |
| 变 $\alpha$ 的覆盖率 | CRC 覆盖率远低于鲁棒水平 | 直接验证"覆盖非必要"的动机 |
| 变 $n$（$\alpha{=}0.1$ 固定） | CRC-E 各指标稳定占优 | 不同样本量下优势稳定 |

### 关键发现
- **过度保守被实证**：CRO/E2E 的鲁棒性常年顶在 96%–99.9%（远超名义目标），代价是更高的风险证书与决策损失；CRC 把鲁棒性精准压回 $1-\alpha$，换来更低的风险与损失。
- **覆盖率远低于鲁棒性**：合成实验里 CRC 的覆盖率显著低于鲁棒水平，直接坐实了"覆盖是充分非必要条件"——不追覆盖反而更高效。
- **跨形状、跨任务稳定**：box 与椭球两种预测集、组合投资 / 股票 / 电池储能三类任务上结论一致，椭球版（能建模分量相关性）通常风险证书更低。

## 亮点与洞察
- **把"约束选错了"这件事讲透并修好**：很多保形 + 决策的工作默认"要鲁棒就得保覆盖"，本文点破覆盖只是充分条件、是过保守的根源，并给出 Theorem 3.1 证明换成鲁棒性约束、只优化预测集不损失最优性——动机和理论咬得很紧。
- **不可微约束的工程化处理干净利落**：指示函数 → 高斯 erf 光滑代理 → 拉格朗日交替梯度，配上隐式微分算 minmax 子问题的梯度，是一套可直接复用到"带覆盖/风险约束的端到端预测集学习"的范式。
- **训练 + 校准两段式**：训练给渐近最优、校准给有限样本鲁棒，把"统计有效性"和"决策效率"解耦——只调一个半径 $t$ 的嵌套保形校准便宜又保真，这个"只校准一维半径"的技巧很值得迁移。
- **理论自洽**：覆盖数刻画复杂度，鲁棒性 gap 与最优性 gap 同以 $O(\sqrt{d\log n/n})$ 收敛，给出了"样本越多越接近理想"的明确速率。

## 局限与展望
- **依赖 i.i.d. / 可交换性**：Theorem 4.1 的有限样本保证完全建立在数据可交换上，分布漂移、时间序列依赖（如真实股市）下保证可能失效，而风险敏感场景恰恰常有非平稳性。
- **预测集形状受限**：实验主要用 box / 椭球（附录有多面体），这些规则形状是为了保 minmax 子问题凸、可解；复杂多峰的条件分布可能无法被良好刻画。
- **可微性前提较强**：理论与高效求解都依赖"CRO 子问题能转成光滑凸规划"，一旦损失 $\phi$ 或约束让子问题非凸/不可微，隐式微分和 Lipschitz 条件都要重新论证。
- **校准遍历开销**：Cal-CRC 需遍历候选标签 $y\in\mathcal Y$，连续/高维标签空间下即便用离散化也可能偏贵，论文未充分量化这部分计算成本。
- **改进方向**：把可交换性放松到协变量漂移下的加权保形；探索更灵活（如神经参数化、非凸）预测集的可解化；给 Cal-CRC 设计更省的校准搜索。

## 相关工作与启发
- **vs CRO + 保形预测（Sun et al. 2023；Johnstone & Cox 2021）**：他们在覆盖约束下构造预测集再做 minmax，覆盖是充分非必要条件导致过保守；本文直接在鲁棒性约束下优化预测集，风险证书严格更低（Appendix B.3），决策更高效。
- **vs RA-CPO / RAC（Kiyani et al. 2025）**：RA-CPO 同样优化期望风险证书但在覆盖约束下，给出依赖最小化 VaR 的最优预测集闭式解，连续决策空间下 VaR 不可解；CRC 换成鲁棒性约束并参数化求解，天然适配连续决策空间。
- **vs End-to-End 决策学习（Chenreddy & Delage 2024；Yeh et al. 2024）**：E2E 直接最小化下游期望决策风险来训练保形预测集；本文实验显示 E2E 仍偏保守（鲁棒性常超目标），CRC 把鲁棒性卡得更准、风险更低。
- **vs Conformal Risk Control（Angelopoulos et al. 2024b）**：CRC 可看作把"鲁棒性"当作一种特殊风险来控制，但该风险关于模型参数**非单调**，且本文优化的是更复杂的风险证书函数 $r(\cdot)$ 而非单一阈值参数，二者不能简单套用。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 点破"覆盖非必要"并把约束换成鲁棒性本身，配等价性定理，角度清晰且有理论支撑
- 实验充分度: ⭐⭐⭐⭐ 合成 + 真实股票 + 电池储能三类任务、两种预测集形状，但多为投资/控制类，场景广度可再扩
- 写作质量: ⭐⭐⭐⭐⭐ 动机—方法—理论—实验逻辑闭环，Figure 1 的对比直观，定理与算法配合紧凑
- 价值: ⭐⭐⭐⭐⭐ 为风险敏感决策提供了"既保鲁棒又更高效"的可落地框架，光滑代理 + 嵌套保形校准范式可迁移性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Multilevel Control Functional](multilevel_control_functional.md)
- [\[ICLR 2026\] LEGACY: A Lightweight Dynamic Gradient Compression Strategy for Distributed Deep Learning](legacy_a_lightweight_dynamic_gradient_compression_strategy_for_distributed_deep_.md)
- [\[ICLR 2026\] Proving the Limited Scalability of Centralized Distributed Optimization via a New Lower Bound Construction](proving_the_limited_scalability_of_centralized_distributed_optimization_via_a_ne.md)
- [\[NeurIPS 2025\] One Sample is Enough to Make Conformal Prediction Robust](../../NeurIPS2025/optimization/one_sample_is_enough_to_make_conformal_prediction_robust.md)
- [\[ICLR 2026\] Differentiable Model Predictive Control on the GPU](differentiable_model_predictive_control_on_the_gpu.md)

</div>

<!-- RELATED:END -->
