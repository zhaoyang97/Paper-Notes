---
title: >-
  [论文解读] Diversified Multinomial Logit Contextual Bandits
description: >-
  [ICLR2026][学习理论][MNL bandit] 本文把"组合多样性"直接嵌进 MNL（多项 logit）选择概率里，提出 DMNL 上下文 bandit 模型，并设计了一个不依赖黑盒优化预言机的白盒 UCB 算法 OFU-DMNL，用逐项贪心构造组合，在 $O(NK)$ 单轮开销下证明了至少 $(1-\frac{1}{e+1})$ 的近似 regret 界 $\tilde{O}(d\sqrt{T/K})$。
tags:
  - "ICLR2026"
  - "学习理论"
  - "在线学习"
  - "组合 bandit"
  - "MNL bandit"
  - "多样性"
  - "子模函数"
  - "近似 regret"
  - "UCB"
---

# Diversified Multinomial Logit Contextual Bandits

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=swwelQtLRn](https://openreview.net/forum?id=swwelQtLRn)  
**代码**: 待确认  
**领域**: 学习理论 / 在线学习 / 组合 bandit  
**关键词**: MNL bandit, 多样性, 子模函数, 近似 regret, UCB

## 一句话总结
本文把"组合多样性"直接嵌进 MNL（多项 logit）选择概率里，提出 DMNL 上下文 bandit 模型，并设计了一个不依赖黑盒优化预言机的白盒 UCB 算法 OFU-DMNL，用逐项贪心构造组合，在 $O(NK)$ 单轮开销下证明了至少 $(1-\frac{1}{e+1})$ 的近似 regret 界 $\tilde{O}(d\sqrt{T/K})$。

## 研究背景与动机

**领域现状**：在序贯组合推荐（电商商品坑位、流媒体片单、应用商店栏目）里，平台每轮展示一个商品集合（assortment），用户最多选一个，平台据反馈在线学习。把"展示集合 → 用户选择概率"联系起来的经典模型是多项 logit（MNL）：每个商品有一个相关性效用 $x_{ti}^\top\theta^*$，用户按 softmax 在集合内（外加一个"啥都不选"的外部选项）做选择。MNL 的好处是组合优化可解（均匀收益下选 top-K 相关性最高的即可）、统计学习有干净的保证，因此衍生了一大批 MNL assortment bandit 与上下文变体的工作。

**现有痛点**：实际选品很少只看相关性——集合内的**多样性**同样关键。一组近似重复的商品会互相蚕食（cannibalization），多花的坑位几乎没带来额外价值；而跨品类/品牌/风格互补的集合往往更受用户青睐。但标准 MNL 的选择概率只通过单个商品的效用进入，**集合内商品之间的相互作用（相似导致的蚕食、互补效应）完全没被建模**。

**核心矛盾**：相关性与多样性之间存在内在 trade-off——多样但不相关的集合照样表现差，所以多样性不能替代相关性，二者要同时考虑。另一条建模路线是子模/组合 bandit，用一个子模奖励函数直接刻画多样性（边际递减、鼓励覆盖），但它们把奖励定义成集合函数，**丢掉了"用户按结构化离散选择模型最多选一个"这一 assortment 场景的核心反馈机制**，也没把相关性效用和多样性效应耦合进同一个概率选择模型。于是"多样性感知的集合选择"与"基于相关性、由选择模型驱动的动态 assortment 学习"之间留下一道建模鸿沟。

**本文目标**：在单一概率选择模型里同时刻画相关性与多样性，并给出端到端的统计 + 计算保证。难点在于：一旦把多样性塞进选择概率，经典 MNL 那种"精确组合优化可解"的性质就没了——最优集合可能需要穷举 $\binom{N}{K}$，即使参数已知也算不动。很多组合 bandit 工作靠假设一个"$\gamma$-近似黑盒预言机"绕过去，但这种假设不现实、还掩盖了近似的来源。

**切入角度**：作者观察到，MNL 奖励函数本身在集合上具有单调子模结构，子模函数的逐项贪心天然有 $(1-\frac1e)$ 近似——这意味着可以**不靠黑盒预言机、自己构造一个透明的逐项贪心当近似预言机**，并直接证明它的近似率；进一步利用 MNL 的特殊结构，还能把 $(1-\frac1e)$ 改进到 $(1-\frac{1}{e+1})$。

**核心 idea**：用一个一般的子模多样性函数 $g_t(S)$ 去**调制 MNL 的外部选项权重**，把相关性–多样性 trade-off 形式化进选择概率（DMNL 模型）；再用逐项乐观构造（item-wise optimistic construction）替代黑盒预言机，实现白盒、可证明近似、计算高效的在线学习。

## 方法详解

### 整体框架
DMNL bandit 的目标是：每轮 $t$ 收到 $N$ 个商品的特征 $X_t=\{x_{t1},\dots,x_{tN}\}$ 和一个给定的多样性度量 $g_t$，要在线选出一个大小 $\le K$ 的集合 $S_t$，使 $T$ 轮累积 regret 最小。整条流程由三块拼成：（1）**DMNL 选择模型**——把子模多样性塞进 MNL 概率，定义"展示集合 → 用户选择"的概率与期望奖励；（2）**逐项贪心近似预言机**——证明在已知参数下，逐项贪心构造能达到比通用子模更好的 $(1-\frac{1}{e+1})$ 近似，从而无需黑盒预言机；（3）**OFU-DMNL 在线算法**——用在线镜像下降估计参数、构造乐观奖励、逐项乐观构造集合，并在多样性参数置信度不足时触发自适应探索。最后给出与 MNL bandit 下界相匹配的近似 regret 上界。

整体是"模型定义 → 离线近似保证 → 在线学习算法 → regret 分析"的理论链条，不是多模块神经 pipeline，因此不画框架图，下面按模型、算法、理论三层讲清。

### 关键设计

**1. DMNL 选择模型：用子模多样性调制外部选项，把相关性–多样性 trade-off 写进选择概率**

痛点是标准 MNL 的选择概率只依赖单品效用，集合内交互无从体现。本文的做法是引入一个单调子模多样性函数 $g_t:S\to\mathbb{R}_{\ge0}$，并让它只作用在"外部选项"（$i=0$，啥都不选）那一项上。选择概率定义为
$$p_t(i\mid S,\theta^*,\lambda^*)=\frac{\exp(x_{ti}^\top\theta^*)}{\exp(-\lambda^* g_t(S))+\sum_{j\in S}\exp(x_{tj}^\top\theta^*)},$$
外部选项概率为 $p_t(0\mid S)=\frac{\exp(-\lambda^* g_t(S))}{\exp(-\lambda^* g_t(S))+\sum_{j\in S}\exp(x_{tj}^\top\theta^*)}$，其中 $\theta^*$ 是相关性参数、$\lambda^*$ 是多样性参数（假设 $\lambda^*>l>0$）。直觉很清楚：当两个集合的单品效用完全相同时，多样性得分 $g_t(S)$ 更高的那个会把外部选项的权重 $\exp(-\lambda^* g_t(S))$ 压低，于是集合内每个商品被选中的概率都上升，集合的期望奖励也更高；反之低多样性会让"啥都不选"更有吸引力。集合 $S$ 的期望奖励就是 $R_t(S,\theta^*,\lambda^*)=\sum_{i\in S}p_t(i\mid S,\theta^*,\lambda^*)$，本文聚焦**均匀收益**（$r_{ti}=1$）设定——只有在均匀收益下，提升多样性才直接等价于提升总点击概率与期望奖励；非均匀收益下高收益单品会主导最优集合，多样性反而可能与目标冲突。这个模型严格泛化了已有 MNL：当 $g_t$ 在所有集合上取常数时就退化回标准 MNL。

**2. 逐项贪心近似预言机：靠 MNL 结构把子模通用界从 $1-\frac1e$ 改进到 $1-\frac{1}{e+1}$**

引入多样性后最优集合需穷举 $\binom{N}{K}$，必须近似。本文不假设黑盒预言机，而是显式构造一个逐项贪心：第 $k$ 步加入边际增益最大的商品 $a_k=\arg\max_{a}R_t(\{a_1,\dots,a_{k-1}\}\cup\{a\})$。关键是先证明 $R_t(S)$ 是单调子模——把奖励写成 $R_t(S)=\frac{\exp(f_t(S))}{1+\exp(f_t(S))}$，其中 $f_t(S)=\log\sum_{i\in S}\exp(x_{ti}^\top\theta^*)+\lambda^* g_t(S)$ 是 LogSumExp（子模）与子模 $g_t$ 之和故仍子模，再叠一个非降凹函数 $\frac{e^x}{1+e^x}$，复合后保持子模。这样逐项贪心自动有通用的 $(1-\frac1e)$ 近似。本文进一步用 MNL 奖励的特殊结构证明了更强的界（Theorem 1）：$R_t(S^{\text{greedy}})\ge\frac{\psi_0(1+\psi_0^\alpha)}{\psi_0^\alpha(1+\psi_0)}R_t(S^*)$，其中 $\alpha=\frac{e}{e-1}$、$\psi_0$ 是 $x^\alpha=\alpha x+\alpha-1$ 的解；粗放下界给出 $1-\frac{1}{e+1}$，严格优于一般子模只能保证的 $1-\frac1e$。这条改进是独立的理论贡献，也正是"白盒替代黑盒预言机"的底气所在。

**3. OFU-DMNL：增广特征 + 在线镜像下降 + 逐项乐观构造的白盒 UCB 算法**

把未知参数合成 $w^*=[\theta^*,\lambda^*]$、把多样性拼进特征 $z_{ti}(S)=[x_{ti},g_t(S)]$，DMNL 概率就写成单参数 MNL 形式 $p_t(i\mid S,w^*)=\frac{\exp(z_{ti}(S)^\top w^*)}{1+\sum_{j\in S}\exp(z_{tj}(S)^\top w^*)}$，于是可沿用 Lee & Oh (2024) 的高效估计：用在线镜像下降在多项 logit 损失 $\ell_t(w)=-\sum_i y_{ti}\log p_t(i\mid S_t,w)$ 上更新 $w_{t+1}$，并维护 Hessian 累积矩阵 $H_t$ 得到置信半径 $\|w_t-w^*\|_{H_t}\le\alpha_t$。据此构造乐观效用 $\text{ucb}(z_{ti}(S))=z_{ti}(S)^\top w_t+\alpha_t\|z_{ti}(S)\|_{H_t^{-1}}$ 与乐观奖励 $\tilde R_t(S)$，再对 $\tilde R_t$ 做逐项乐观构造 $a_k=\arg\max_a\tilde R_t(\{a_1,\dots,a_{k-1}\}\cup\{a\})$（注意因 $z_{ti}(S)$ 依赖整个集合 $S$，乐观奖励不能逐品独立算，所以必须逐项加入），单轮开销 $O(NK)$。

**4. 自适应探索：在多样性参数置信不足时随机探索，弥补联合估计无法解耦的不确定性**

$\theta$ 与 $\lambda$ 通过增广特征联合估计，二者的单独不确定性无法解耦，联合置信宽度对 $\lambda^*$ 可能不够紧，导致逐项乐观构造的"乐观性"得不到保证。为此算法加了一道触发条件：当 $\|[0_d,1]\|_{H_t^{-1}}\ge\frac{\nu}{\hat\lambda_t}\alpha_t$（即沿多样性方向 $[0_d,1]$ 的不确定性过大）时，本轮随机均匀地选一个大小为 $K$ 的集合做探索，否则才走逐项乐观构造。作者证明这类强制探索的轮数至多 $O(\sqrt{d}\log T)$，因此不会破坏总体 regret 阶。

### 损失函数 / 训练策略
参数估计用在线镜像下降最小化多项 logit 损失 $\ell_t(w)=-\sum_{i\in S_t}y_{ti}\log p_t(i\mid S_t,w)$，更新式
$$w_{t+1}=\arg\min_{w\in\mathcal W}\Big\{\langle\nabla\ell_t(w_t),w\rangle+\tfrac{1}{2\eta}\|w-w_t\|_{\tilde H_t}^2\Big\},$$
其中 $\mathcal W=\{w:\|w\|_2\le1\}$，$\tilde H_t=H_t+\eta G_t(w_t)$，$H_t=\Lambda I_{d+1}+\sum_{s<t}G_s(w_{s+1})$，$G_t(w)$ 是 MNL 概率诱导的 Hessian。算法超参设为 $\alpha_t=O(\sqrt{d\log t}\log K)$、$\eta=\frac12\log(K+1)+2$、$\Lambda=84\sqrt{(d+1)\eta}$、$\nu=\frac\omega2$，且**不需要知道** $l$。

## 实验关键数据

### 主实验
合成实验：上下文特征独立采自 $\mathcal N(0_d,I_d)$ 并裁剪到 $[-1/\sqrt d,1/\sqrt d]^d$，每个商品分配一个类别，多样性函数取"指数衰减类别覆盖"函数；固定多样性参数 $\lambda^*$ 于若干取值，$\theta^*$ 采自均匀分布并缩放使 $\|\theta^*\|_2+\lambda^*=1$；每组配置跑 10 次独立实验取均值。对比对象包含纯 MNL 基线 UCB-MNL、TS-MNL、OFU-MNL+，以及两个改造版：OFU-MNL-DR（沿用标准 MNL 选择模型但奖励加一个需手调 $\lambda$ 的子模项）和 OFU-DMNL-FULL（精确实现 DMNL 但每轮穷举 $\binom{N}{K}$，开销约 $O((eN/K)^K)$）。

| 设定（N, K, d, λ*） | 累积 regret 表现 | 运行时间 |
|--------|------|------|
| N=10/20, K=5, d=5, λ*=0.4 | OFU-DMNL 显著低于纯 MNL 基线与 OFU-MNL-DR | OFU-DMNL ≪ OFU-DMNL-FULL（数量级优势） |
| N=50/100, K=5/10, d=5, λ*=0.4 | OFU-DMNL 与穷举的 OFU-DMNL-FULL regret 相当 | OFU-DMNL 单轮 $O(NK)$，FULL 随 N、K 爆炸 |

核心结论：OFU-DMNL 在多种规模下都优于忽略/手调多样性的基线，并且**逼近穷举最优解的 regret，却以数量级更低的运行时间达成**。

### 消融 / 对照

| 对照算法 | 与本文的差别 | 说明 |
|------|---------|------|
| OFU-MNL+ | 不建模多样性 | 在 DMNL 环境下因忽视多样性而吃亏 |
| OFU-MNL-DR | 标准 MNL + 手调 $\lambda$ 的加性子模奖励 | 需人工调多样性权重，且选择模型未耦合多样性 |
| OFU-DMNL-FULL | 精确 DMNL 但穷举 | regret 可作上界参照，但计算不可行 |
| OFU-DMNL（本文） | 逐项乐观构造 + 联合学 $\lambda$ | 无需调参、$O(NK)$、近似最优 |

### 关键发现
- 在 $\lambda$ 控制的不同相关性–多样性配比下（Figure 5），OFU-DMNL 均稳健占优，说明它能自适应不同 trade-off 区间，而不是只在某一档好。
- "学习多样性" vs "手调多样性"的对比（vs OFU-MNL-DR）显示直接联合学 $\lambda^*$ 免去了人工调参，且因为多样性进了选择概率而非外挂在奖励上，建模更贴合用户行为。
- 与穷举 FULL 的 regret 几乎持平但运行时间数量级更低，定量印证了逐项乐观构造"统计 + 计算双高效"。

## 亮点与洞察
- **多样性进概率、不进奖励**：把 $g_t$ 放在外部选项权重上，让"集合越多样、外部选项越不诱人、各单品被选概率越高"，既保留了 MNL 的离散选择反馈机制，又自然写出了相关性–多样性 trade-off——比把多样性硬塞进加性奖励更贴合真实选品行为。
- **白盒替代黑盒预言机**：先证明 DMNL 奖励单调子模（LogSumExp 子模 + 子模和 + 非降凹复合），从而逐项贪心本身就是一个有证明近似率的预言机，再用 MNL 结构把 $1-\frac1e$ 抬到 $1-\frac{1}{e+1}$。这条"自造可证近似预言机"的思路可迁移到其他需要组合优化预言机的 bandit 问题。
- **增广特征统一估计**：$z_{ti}(S)=[x_{ti},g_t(S)]$ 把"多学一个多样性参数"折叠成单参数 MNL 估计问题，复用现成的近极小极大最优 MNL 估计器，使得 regret 上界 $\tilde O(d\sqrt{T/K})$ 在阶上不因多学 $\lambda$ 而变差。
- **针对耦合参数的自适应探索**：当不能解耦 $\theta$ 与 $\lambda$ 的不确定性时，用一道沿 $[0_d,1]$ 方向的置信判据触发随机探索、并证明其轮数仅 $O(\sqrt d\log T)$，是处理"联合估计但需保证某分量乐观性"的实用手法。

## 局限与展望
- **仅限均匀收益**：作者明确只研究 $r_{ti}=1$，因为非均匀收益下高收益单品会主导最优集合、多样性目标与奖励最大化可能冲突；带收益的多样化 assortment 仍是 open 的。
- **依赖较强的结构假设**：需要非退化（特征张成 $\mathbb R^d$、$g_t$ 非常数）和 $\omega$-严格子模（Definition 2）等假设；现实中多样性度量未必严格子模，$\lambda^*>l>0$ 的"多样性效应恒正"假设也排除了多样性无关的情形。
- **近似率而非精确最优**：因穷举不可行，目标是 $\gamma$-近似 regret（$\gamma\ge1-\frac{1}{e+1}$），与精确 regret 之间天然有 $\gamma$ 倍差距；改进近似率仍有空间。
- **实验为合成**：只在合成数据上验证，缺真实推荐数据集上的端到端效果；多样性函数取的是单一类别覆盖型，换其他子模度量的鲁棒性待验。

## 相关工作与启发
- **vs 经典 MNL assortment bandit（Oh & Iyengar；Lee & Oh 2024/2025 等）**：他们的选择概率只依赖单品相关性、top-K 即最优；本文把子模多样性嵌进概率，最优集合需近似，但 regret 阶 $\tilde O(d\sqrt{T/K})$ 与之匹配（代价是多学一个 $\lambda$）。
- **vs 子模/组合 bandit（Yue & Guestrin 2011；Chen 等 2013/2016/2017；Hwang 等 2023）**：他们用集合函数定义奖励、且常假设黑盒近似预言机或需要逐项/交互式边际增益反馈；本文用 MNL 离散选择反馈（只看最终选择），白盒构造预言机，并证明严格子模假设足以在无中间反馈下恢复乐观奖励的子模性。
- **vs OFU-MNL-DR（加性多样性基线）**：DR 在标准 MNL 上外挂一个需手调权重的加性子模奖励；本文把多样性放进选择概率并联合学习，免调参且建模更自洽，实验上也更优。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个把多样性直接写进 MNL 选择概率的 bandit 模型，并自造可证近似预言机替代黑盒。
- 实验充分度: ⭐⭐⭐⭐ 合成实验覆盖多规模与多 trade-off 配比、含穷举对照，但缺真实数据。
- 写作质量: ⭐⭐⭐⭐⭐ 模型–近似–算法–regret 链条清晰，假设与定理讨论到位。
- 价值: ⭐⭐⭐⭐ 为不确定性下的多样性感知 assortment 优化提供了有理论保证又计算高效的范式。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Variance-Dependent Regret Lower Bounds for Contextual Bandits](variance-dependent_regret_lower_bounds_for_contextual_bandits.md)
- [\[ICLR 2026\] Queue Length Regret Bounds for Contextual Queueing Bandits](queue_length_regret_bounds_for_contextual_queueing_bandits.md)
- [\[ICML 2026\] Optimal Design for Multinomial Logit Model with Applications to Best Assortment Identification](../../ICML2026/learning_theory/optimal_design_for_multinomial_logit_model_with_applications_to_best_assortment_.md)
- [\[ICLR 2026\] Towards a Sharp Analysis of Offline Policy Learning for f-Divergence-Regularized Contextual Bandits](towards_a_sharp_analysis_of_offline_policy_learning_for_f-divergence-regularized.md)
- [\[ICLR 2026\] Contextual Multi-Armed Bandits with Minimum Aggregated Revenue Constraints](contextual_multi-armed_bandits_with_minimum_aggregated_revenue_constraints.md)

</div>

<!-- RELATED:END -->
