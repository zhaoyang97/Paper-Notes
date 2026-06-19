---
title: >-
  [论文解读] Multi-Level Strategic Classification: Incentivizing Improvement Through Promotion and Relegation Dynamics
description: >-
  [ICML 2026][策略性分类] 本文把传统单次"策略性分类"扩展成一个由多级三元分类器（通过/弃判/不通过 = 晋升/留级/降级）构成的序贯机制，证明仅靠折现因子 $\beta$、技能保留率 $\gamma$ 与"高位增益"$\delta$ 这三种跨期效应，就能把不可激励区域从 $c^+>c^-$ 缩小到 $(1-\beta\gamma)c^+>c^-$；进一步给出 $\mu_l = \delta(l-1)/(1-\gamma)$ 的稳态阈值序列，证明在温和条件下可激励诚实努力把属性推到任意高水平。
tags:
  - "ICML 2026"
  - "策略性分类"
  - "多级机制"
  - "晋升-降级"
  - "马尔可夫决策过程"
  - "激励相容"
---

# Multi-Level Strategic Classification: Incentivizing Improvement Through Promotion and Relegation Dynamics

**会议**: ICML 2026  
**arXiv**: [2602.11439](https://arxiv.org/abs/2602.11439)  
**代码**: 无  
**领域**: 策略性分类 / 机制设计 / 算法公平性  
**关键词**: 策略性分类、多级机制、晋升-降级、马尔可夫决策过程、激励相容  

## 一句话总结
本文把传统单次"策略性分类"扩展成一个由多级三元分类器（通过/弃判/不通过 = 晋升/留级/降级）构成的序贯机制，证明仅靠折现因子 $\beta$、技能保留率 $\gamma$ 与"高位增益"$\delta$ 这三种跨期效应，就能把不可激励区域从 $c^+>c^-$ 缩小到 $(1-\beta\gamma)c^+>c^-$；进一步给出 $\mu_l = \delta(l-1)/(1-\gamma)$ 的稳态阈值序列，证明在温和条件下可激励诚实努力把属性推到任意高水平。

## 研究背景与动机
**领域现状**：策略性分类研究主体——决策者部署分类器、被分类的自利个体可以选择"诚实改进"或"低成本作弊"。经典结论非常负面：单次交互下当作弊成本严格低于真实改进成本（$c^- < c^+$）时，理性 Agent 永远选择作弊，除非引入外部补贴或惩罚等额外工具。

**现有痛点**：（1）单次模型把人当无记忆的优化器，忽略真实场景里"昨天的努力会影响今天的状态"这种跨期耦合；（2）现有序贯策略性分类研究大多围绕"如何动态更新分类器权重"做文章，对阈值设计、级别进度和"晋升带来的边际增益"几乎没有系统刻画；（3）经典工作（Harris 等 2021）虽涉及序贯回归，但既不考虑分类的离散反馈，也不引入技能折旧。

**核心矛盾**：要让 Agent 自愿选择更贵的诚实改进，需要存在某种"未来收益放大器"，而单次或仅靠权重调节的机制都缺乏这种放大器的显式表达。

**本文目标**：(1) 形式化一个多级、有晋升-降级动态的序贯机制；(2) 完整刻画两级（单分类器）下 Agent 的最优长期策略；(3) 给出多级阈值设计的可行性条件与最优解，证明诚实改进可以把属性推到任意高水平 $M$。

**切入角度**：作者注意到现实里"考试 → 升级 → 更多资源 → 更易考更高级别"这类正反馈是天然存在的，把它显式刻画成 leg-up 因子 $\delta$，再叠加 retention $\gamma$ 与 farsightedness $\beta$，三者共同压低改进的"等效长期单位成本"。

**核心 idea**：用三元分类器（pass/abstain/fail）构造级别进阶机制，把"诚实成本 $c^+$"的真实经济含义改写为 $(1-\beta\gamma)c^+$，再用 $\delta$ 提供持续上拉，从而无需外部补贴就能激励改进。

## 方法详解
所有动力学都发生在一个连续状态连续动作的 MDP $\{(l_t, x_t)\}_{t\ge0}$ 上：$l_t$ 是离散级别，$x_t\ge0$ 是私有属性（不可观测），$z_t = x_t + a_t^+ + a_t^-$ 是可观测特征。Agent 在每一步同时选择改进量 $a_t^+\ge0$（单位成本 $c^+$）与作弊量 $a_t^-\ge0$（单位成本 $c^-<c^+$），二者对特征 $z_t$ 同等贡献，但只有 $a_t^+$ 真正改进属性。

### 整体框架
单步动作后属性立刻变为 $x_{t_+} = x_t + a_t^+$；然后进入下一步前再经过两次修正：折旧 $\gamma\in(0,1)$ 把属性缩到 $\gamma x_{t_+}$，再加上一个与级别相关的 leg-up 增益 $\delta(l_{t+1}-1)$。综合起来：$x_{t+1}=\gamma x_{t_+}+\delta(l_{t+1}-1)$。分类器是按级别索引的三元函数：$\theta z_t \ge \mu_{l+1}$ 则晋升，$\mu_l \le \theta z_t < \mu_{l+1}$ 则留级，$\theta z_t \le \mu_l$ 则降级（边界级别只能单向）；不失一般性令 $\theta=1, \mu_1\equiv0$。Agent 目标是最大化无限期折现总收益 $\sum_t \beta^t (R_{l_{t+1}} - \vec c^\top \vec a_t)$，其中 $R_l = r(l-1)$ 与级别线性相关。Principal 的任务是设计最短阈值序列 $\vec\mu$ 使得 (i) Agent 永不作弊；(ii) 长期属性 $\liminf_t x_{t_+}\ge M$；(iii) 最终到达最高级别。

### 关键设计

**1. 三元分类多级机制：把"单次不可激励"翻译成可调节的几何约束**

单次模型里 Agent 没有任何未来回报可言，只要作弊更便宜（$c^-<c^+$）就永远占优，这是策略性分类领域最经典也最悲观的不可能结论。本文的破局点是把每一级做成一个**能弃判的选择性分类器**：通过 $\theta z_t\ge\mu_{l+1}$ 晋升、$\mu_l\le\theta z_t<\mu_{l+1}$ 留级、$\theta z_t\le\mu_l$ 降级，原来的"通过/不通过"二元被扩展成"晋升/留级/降级"三元——abstain 不再是统计上的弃判，而是经济上的"留在原级"。这一抽象让阈值从"终极结果"退化成"短期门槛"，于是机制设计者获得了把跨期效应映射成静态约束的杠杆。

杠杆由三种跨期效应提供，各有清晰的经济解释：折现因子 $\beta$ 表示 Agent 在意未来、保留率 $\gamma$ 反映技能折旧、leg-up $\delta$ 是高级别带来的资源溢出。命题 2.1 把它们压成一个干净的等效成本表达式——只要 $(1-\beta\gamma)c^+<c^-$，就存在让 Agent 自愿改进的设计，相当于把单次不可激励区域 $c^+>c^-$ 严格收缩到 $(1-\beta\gamma)c^+>c^-$。换言之，未来回报把诚实改进的"等效长期单位成本"从 $c^+$ 压到了 $(1-\beta\gamma)c^+$，这是整篇文章无需外部补贴就能突破激励墙的根本原因。

**2. 两级最优策略的完整相图：给多级设计提供可解的原子构件**

要在多级机制上用动态规划，必须先知道"在某一级 Agent 究竟会怎么响应"，而两级（单分类器）是这个问题的最小可解情形，也正是后面贪心算法每一步的子问题（定理 5.1）。本文把它彻底解成了一张相图。定理 3.1 处理低阈值情形：当 $\mu<\delta/(1-\gamma)$ 时存在临界点 $x^\circ\in[0,\mu]$，Agent 在 $[x^\circ,\mu]$ 区间纯作弊、在 $[0,x^\circ)$ 区间混合改进与作弊。定理 3.2 处理高阈值情形，给出两个与 $\delta$ 无关的常数 $\underline\mu,\overline\mu$，把 $\mu\ge\delta/(1-\gamma)$ 进一步切成三段：靠近门槛时纯改进、否则不动；中段按距离"近改进、中作弊、远不动"；当 $\mu\ge\overline\mu+\delta/(1-\gamma)$ 时改进已不值得，只剩作弊或放弃。

相图里藏着一个关键的不对称：$\underline\mu,\overline\mu$ 都随 $\beta,\gamma$ 单调增，但 $\gamma\to1$ 会把两者推到无穷、彻底消除纯作弊区，而 $\beta\to1$ 只能把它们推到有限上限 $r/((1-\gamma)c^+)$。也就是说，**技能保留率比远见对消除作弊更有效**——这条结论后面在 FICO 实验里被反复验证，也直接决定了哪种系统更容易设计。

**3. 稳态阈值序列 $\mu_l=\delta(l-1)/(1-\gamma)$：把阈值钉在自然均衡上**

有了两级相图，剩下的问题是怎么排布整条阈值序列才能把属性一路推到任意目标 $M$。朴素直觉是"小步快跑"密集设阈值鼓励攀登，但太密集会让 Agent 仅靠 leg-up 滚雪球（promotion begets promotion）、太稀疏又让折旧吃光属性，两头都翻车。本文给出的最简闭式序列把阈值正好设在 Agent 若停留该级会自然收敛到的属性稳态上：

$$\mu_l = \frac{\delta(l-1)}{1-\gamma}$$

这个取值的妙处在于折旧项 $-\gamma\mu_l$ 与 leg-up 项 $+\delta(l-1)$ 在此精确抵消，于是从下一级晋升过来时属性不会回落——Principal 是在借力自然均衡而不是与之对抗。定理 4.2 给出它的可行边界：当 $\delta>0$ 且 $r<\frac{1-\beta}{1-\gamma}c^+\delta$ 时对任意 $M$ 都不可行；反之当 $r\ge\frac{1-\beta}{1-\gamma}c^+\delta$ 且

$$c^-\ge\max\Big\{(1+\tfrac{\beta\gamma}{2})(1-\beta\gamma)c^+,\ \beta\gamma(1-\beta^2\gamma^2)c^+\Big\}$$

时可行，所需级数仅 $L=\lceil(1-\gamma)M/\delta\rceil$，且 $r$ 取边界值时上述序列最优。作为对照，没有 leg-up（$\delta=0$）时定理 4.1 给出硬性不可行上界 $M\ge r/((1-\beta)(1-\gamma)^2c^+)$，其中 $(1-\gamma)$ 的二次方再次印证了保留率的主导地位。

### 损失函数 / 训练策略
没有学习损失。Agent 端用 ValueIterate（值迭代 + 属性空间离散化 + 线性插值）求解 MDP，证明收敛速率为 $O(\log(1/\varepsilon)/|\log\beta|)$、值函数误差上界 $c^+\Delta x/(2(1-\beta))$。Principal 端在松弛目标下用 CMA-ES 进行黑箱优化，并配套一个贪心阈值搜索算法（Algorithm 1，定理 5.1 保证返回的序列在 $M\le \mu_L$ 时可行）。

## 实验关键数据

### 主实验
FICO 信用评分数据（归一化到 $[0,10]$）模拟多级信用产品系统，固定 $\beta=\gamma=0.8, \delta=0.01, \alpha=0.95, \xi=0.01, \lambda=5$，对 $L\in[2,8]$ 搜索 Principal 最优设计：

| Case | $(c^+, c^-)$ | $L^*$ | $r^*$ | $\mu_L^*$ | $U^*$ |
|------|--------------|-------|-------|-----------|-------|
| I 易学难骗 | (0.8, 0.7) | 6 | 1.80 | 10.76 | 630.4 |
| II 双高 | (1.5, 1.2) | 7 | 2.51 | 11.92 | 629.9 |
| III 易学易骗 | (0.8, 0.4) | 2 | 4.48 | 11.98 | 628.8 |
| IV 难学易骗 | (1.5, 0.4) | 8 | 0.63 | 7.98 | 107.9 |

### 消融实验

| 配置 | 关键现象 | 说明 |
|------|----------|------|
| 完整机制 (Case I) | 全程纯改进、属性单调上升 | 激励对齐成立，达成理想轨迹 |
| 缺乏奖励 (Case IV) | $r^*$ 被压到 0.63，作弊占优 | 假设 2.2 不成立时机制退化 |
| 折扣 $\beta\to0$ | 不可激励区域回到 $c^+>c^-$ | 跨期效应丧失 |
| 保留 $\gamma\to1$ | $\underline\mu,\overline\mu\to\infty$，纯作弊区消失 | 技能不折旧时 Principal 自由度最大 |
| $\delta=0$ | 受定理 4.1 上界 $r/((1-\beta)(1-\gamma)^2c^+)$ 制约 | 没有 leg-up 时存在硬上限 |

### 关键发现
- 单次问题里不可激励区域是 $c^+>c^-$，多级机制把它缩小到 $(1-\beta\gamma)c^+>c^-$，定理给出的紧的几何缩小被 FICO 实验中的相变（gaming cost 跨过 $(1-\beta\gamma)c^+$ 时激励能力突然消失）实证。
- $\gamma$（技能保留率）比 $\beta$（折现因子）更有效——前者推动 $\underline\mu,\overline\mu$ 二次扩张、把作弊区消除，后者只能线性逼近一个有限上限。
- 阈值序列 $\mu_l = \delta(l-1)/(1-\gamma)$ 在大 $M$ 时近似最优，且经验上 leg-up 弱时（$\delta$ 很小）也几乎不损失效率，说明把阈值钉在自然均衡上是个鲁棒选择。
- Case III 揭示了一个反直觉现象——当作弊成本远低于改进成本时，最优设计是把级别压成 2 级、把阈值拉得极高，用一次性大门槛而不是渐进阶梯来阻止持续作弊。

## 亮点与洞察
- 把"三元分类 + 多级"作为机制设计原语是一个非常聪明的封装：abstain 自然映射到"留级"，把弃判的统计学动机翻译成 Agent 经济学含义。
- 命题 2.1 给出的 $(1-\beta\gamma)c^+$ 这个等效成本表达式干净到可以直接用作政策指引——只要算出系统的折现率与技能折旧率，就能立刻判断激励是否可行。
- 把"在自然稳态上钉阈值"这一直觉用 $\mu_l = \delta(l-1)/(1-\gamma)$ 写成闭式，避免了对每个 $M$ 重新做凸优化，工程上极易部署。
- 经济解释贯穿全文：每条定理都给出了直观说明，让一个偏 ML 背景的读者也能轻松映射到现实场景（学位证书、信用评级、职业认证）。

## 局限与展望
- 模型假设属性、特征均为标量，作者明确说明可推广到多维但未给出多维分析；现实中信用、教育的"qualification"几乎一定是多维向量，跨维耦合可能让 leg-up 与 retention 难以分开估计。
- 三元分类器假设统一的模型权重 $\theta$，并默认 $\theta$ 来自非策略数据估计；当策略反馈污染训练分布时（即 Hardt 等的反馈循环问题），$\theta$ 估计偏差会直接抹掉理论保证。
- 实验仅在合成 + FICO 上做，缺少在真实序贯任务（如多次考试、信贷续贷）上的纵向验证；同时对 Agent 异质性、群体公平性的讨论几乎没有，存在公平性盲点。
- 改进方向：把权重 $\theta$ 与阈值 $\vec\mu$ 一起放入序贯设计，并引入 Agent 类型分布显式建模；在更现实的"reset"事件（如换工作、换平台）下重新刻画稳态。

## 相关工作与启发
- **vs Harris 等 2021**：他们做的是序贯回归 + 努力累积，未考虑分类的离散反馈，也没建模属性折旧与 leg-up；本文显式把这三种跨期效应写进 MDP，给出可解析的可行性边界。
- **vs Hardt 等 2015 / Milli 等 2019**：单次策略性分类的负面结论 $c^+>c^-$ 不可激励在本文中被严格弱化为 $(1-\beta\gamma)c^+>c^-$，是该领域少有的"机制设计本身就能突破激励墙"的结果。
- **vs Jin 等 2022**：他们靠外部补贴来打破不可激励性，需要额外预算；本文证明在多级机制 + 自然 leg-up 下不需要外部货币转移就能达到同样效果。
- **vs Kleinberg & Raghavan 2019**：他们关注努力激励的拓扑刻画，本文给出可计算的阈值序列与数值实验，对现实政策更具可操作性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把"多级、三元、leg-up + retention + farsightedness"放进同一个序贯框架并给出闭式最优解，在策略性分类领域极少见。
- 实验充分度: ⭐⭐⭐ FICO 与合成实验充分验证理论，但缺少真实纵向数据与多维属性扩展。
- 写作质量: ⭐⭐⭐⭐ 定理与经济解释紧密配合，相图（图 3）非常直观；唯一不足是部分关键证明放在附录、主文阅读时需要频繁跳转。
- 价值: ⭐⭐⭐⭐ 对设计教育、信用、认证等多级决策系统的算法机制提供了直接可用的分析框架与设计原则。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DF²-VB: Dual-level Fuzzy Fusion with View-specific Boosting for Multi-view Multi-label Classification](../../CVPR2026/others/df2-vb_dual-level_fuzzy_fusion_with_view-specific_boosting_for_multi-view_multi-.md)
- [\[ICML 2026\] Networked Information Aggregation for Binary Classification](networked_information_aggregation_for_binary_classification.md)
- [\[CVPR 2026\] Prototype-based Causal Intervention for Multi-Label Image Classification](../../CVPR2026/others/prototype-based_causal_intervention_for_multi-label_image_classification.md)
- [\[AAAI 2026\] DcMatch: Unsupervised Multi-Shape Matching with Dual-Level Consistency](../../AAAI2026/others/dcmatch_unsupervised_multi-shape_matching_with_dual-level_consistency.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](../../ICLR2026/others/distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)

</div>

<!-- RELATED:END -->
