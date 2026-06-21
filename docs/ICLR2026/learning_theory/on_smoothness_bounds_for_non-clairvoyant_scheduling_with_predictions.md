---
title: >-
  [论文解读] On Smoothness Bounds for Non-Clairvoyant Scheduling with Predictions
description: >-
  [ICLR 2026][学习增强算法理论][带预测算法] 本文重新定义了「带预测算法」里的平滑性（smoothness）指标——只在「预测确实提供了额外信息」的实例子集上度量竞争比，从而避免旧定义被无信息实例污染；并在此新指标下为三类非透视调度问题给出了更紧的下界与匹配算法（单机总完成时间 $\eta$ 下界 + $\eta^2$ 算法、并行同速机 makespan $2-O(\eta^{-2})$ 下界 + $O(\eta^2)$ 算法、相关速度机 makespan 紧的 $\lceil\log\eta\rceil$ 界）。
tags:
  - "ICLR 2026"
  - "学习增强算法理论"
  - "在线调度"
  - "带预测算法"
  - "平滑性"
  - "非透视调度"
  - "竞争比"
  - "对抗下界"
---

# On Smoothness Bounds for Non-Clairvoyant Scheduling with Predictions

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Zcn4n57lHg](https://openreview.net/forum?id=Zcn4n57lHg)  
**代码**: 无  
**领域**: 学习增强算法理论 / 在线调度  
**关键词**: 带预测算法, 平滑性, 非透视调度, 竞争比, 对抗下界

## 一句话总结
本文重新定义了「带预测算法」里的平滑性（smoothness）指标——只在「预测确实提供了额外信息」的实例子集上度量竞争比，从而避免旧定义被无信息实例污染；并在此新指标下为三类非透视调度问题给出了更紧的下界与匹配算法（单机总完成时间 $\eta$ 下界 + $\eta^2$ 算法、并行同速机 makespan $2-O(\eta^{-2})$ 下界 + $O(\eta^2)$ 算法、相关速度机 makespan 紧的 $\lceil\log\eta\rceil$ 界）。

## 研究背景与动机
**领域现状**：「带预测算法」（algorithms with predictions / learning-augmented algorithms）是在线决策的一个活跃方向：给在线算法喂一个对未知输入的预测，再用两个指标评价——**一致性**（consistency，预测完美时的竞争比）和**鲁棒性**（robustness，预测最坏时的竞争比）。除了这两端，人们还希望算法是**平滑的**：随着预测误差变大，性能优雅退化，而不是断崖式崩掉。通行做法是把竞争比写成预测误差 $\eta$ 的函数，这个函数就叫平滑性曲线。

**现有痛点**：作者指出，旧的平滑性定义「竞争比作为误差的函数」常常**毫无意义**。原因是：对一个给定的误差 $\eta$，往往能构造出一批实例，在这些实例上预测根本不提供任何额外信息——尤其当误差用「绝对量」（如 $L_1$ 距离 $\sum_j|p_j^*-p_j|$）来定义时。论文给了一个直白的反例：取任意实例 $I$，把所有作业大小缩放到 $\sum_j p_j'^* = \eta$，再把所有预测设为 0。这样误差恰好是 $\eta$，但预测完全无信息，算法表现等同于「没有预测」的最坏情况。由于这对**任意** $\eta$ 都成立，旧平滑性曲线可以被这些无信息实例顶到任意高，根本反映不出「合理预测」带来的边际收益。

**核心矛盾**：平滑性想衡量的是「预测带来的边际价值」，但旧定义把「预测有用」和「预测无用」的实例混在一起取上确界，结果被无用实例主导，失去了诊断意义。

**本文目标**：重新定义一个平滑性指标，让它**只在预测保证提供额外信息的实例上**度量，从而下界能真正解读为「即便预测总在提供信息，任何算法也好不过这个比值」——这同时刻画了预测的极限与潜力。

**切入角度**：把全体实例 $\mathcal{I}$ 按预测误差切成三块——完美预测、误差大到预测在最坏情况下区分不出任何东西、误差小到预测保证带来信息——分别对应一致性、鲁棒性、平滑性。关键是要有一个谓词来精确刻画「预测无信息」这件事。

**核心 idea**：用谓词 $R(I)$ 判定「是否存在一个误差恰为 $\eta(I)$ 的常数预测向量」来界定「无信息」实例，把它们从平滑性的度量域里剔除；在剩下的「合理预测」子集上重做经典调度问题的下界与算法分析。

## 方法详解
本文是纯理论工作，核心不是某个算法 pipeline，而是**一个新指标的定义 + 三类调度问题在该指标下的下界证明与匹配算法**。整体框架是：先用谓词 $R(I)$ 把实例宇宙划成三块、据此重定义平滑性，然后在三个经典非透视调度问题上分别构造对抗实例求下界、并设计算法求上界，验证新指标既能给出更紧的界、又能揭示预测的边际价值。这是一个「理论框架 + 多问题求解」式结构，没有清晰的数据流 pipeline，故不配框架图。

### 整体框架
**问题设定**。$m$ 台机器、$n$ 个独立作业，作业 $J_j$ 的真实大小 $p_j^*$ 在它完成前未知（这就是「非透视 non-clairvoyant」），机器 $M_i$ 速度 $s_i$。作业到达时给出大小预测 $p_j$。预测误差用**乘性**定义：单作业 $\eta_j=\max\{p_j/p_j^*,\,p_j^*/p_j\}$，实例误差 $\eta(I)=\max_j\eta_j\ge 1$，$\eta=1$ 当且仅当预测完美。乘性误差相比 $L_1$ 的好处是**尺度无关**：把所有作业大小同乘 100，$L_1$ 误差放大 100 倍，但乘性误差不变，而调度顺序若不变竞争比也不变，所以乘性度量更贴合算法实际表现。

**三分实例 + 重定义指标**。给定误差函数 $\eta(\mathbf{p},\mathbf{p}^*)$，谓词 $R(I)$ 判定「是否存在常数向量 $\mathbf{p}$ 使 $\eta(\mathbf{p},\mathbf{p}^*(I))=\eta(I)$」——若存在，说明在该误差水平上预测可以表现为「所有作业预测都一样」，无法对作业排序或分组，等于没有预测。据此划分：
$$\mathcal{I}_c=\{I\mid\eta(I)=\eta(\mathbf{p}^*,\mathbf{p}^*)\},\quad \mathcal{I}_r=\{I\mid R(I),\,I\notin\mathcal{I}_c\},\quad \mathcal{I}_s=\{I\mid\neg R(I),\,I\notin\mathcal{I}_c\}$$
三块完备且互斥。一致性、鲁棒性、平滑性分别是竞争比 $A(I)/\mathrm{Opt}(I)$ 在 $\mathcal{I}_c,\mathcal{I}_r,\mathcal{I}_s$ 上的上确界，其中平滑性 $s_A(\eta)=\sup_{I\in\mathcal{I}_s,\,\eta(I)\le\eta}A(I)/\mathrm{Opt}(I)$。在乘性误差下可解析出 $\mathcal{I}_s=\{I\mid 1<\eta(I)<\sqrt{P(I)}\}$，其中 $P(I)$ 是实例内最大作业大小比——直观即「误差还没大到能用常数向量伪装」。

**三个问题的求解**。框架落到三个 Graham 记号下的非透视调度问题，逐一给下界（对抗构造）+ 上界（算法），下面三个关键设计依次对应这三个问题，加上贯穿其上的指标定义本身。

### 关键设计

**1. 精炼的平滑性定义：用谓词 $R(I)$ 剔除无信息实例**

这一设计直击「旧平滑性被无信息实例污染」的痛点。机制是：不再在全体非完美实例上取竞争比上确界，而是先用 $R(I)$ 把「存在等误差常数预测向量」的实例（$\mathcal{I}_r$，预测可被伪装成无信息）划走，只在补集 $\mathcal{I}_s$ 上度量。形式化地，平滑性 $s_A(\eta)=\sup_{I\in\mathcal{I}_s,\,\eta(I)\le\eta}A(I)/\mathrm{Opt}(I)$。这样得到的下界有一个干净的解读：**即使预测总在提供额外信息，任何确定性算法的竞争比也不可能优于 $s_A(\eta)$**——它同时是预测能力的上限和潜力的下限。论文进一步论证这个指标在实践上比一致性/鲁棒性更有用：真实预测通常带些误差但非对抗，从业者虽难以先验知道 $\eta$，却能通过历史回测估出误差范围，于是可在该范围上优化平滑性；而 $\Omega(\eta)$ 这样的下界还能告诉业务方「把预测误差砍半最多能换来目标翻倍的改善」，从而指导是否值得投入做更准的预测模型。作者也在讨论里给出一个不剔除 $\mathcal{I}_r$ 的备选定义 $s_A'(\eta)$，并说明本文所有下界对 $s_A'$ 同样成立（因为 $\mathcal{I}_s$ 是更大集合的子集），但本文定义能完全分离两指标考虑的实例、且界更强。

**2. 单机总完成时间：min-max 对抗博弈给出 $\eta$ 下界与 $\eta^2$ 算法**

针对 $1\,|\,\text{online-time-nclv, pmtn}\,|\,\sum C_j$ 问题。痛点是旧工作都用尺度敏感的 $L_1$ 误差，且从 Azar et al. (2022) 只能直接得到 $O(\eta^2\log\eta)$ 上界。本文换用乘性误差后，下界证明呈现一个清晰的**调度者 vs 对手**博弈：让前 $n$ 个作业大小非增但预测相同（彼此不可区分），最后一个作业极小但预测正确（保证落在 $\mathcal{I}_s$、预测确实有信息，且这个小作业对总完成时间影响微乎其微，可「免费」做掉以简化讨论）。由于前 $n$ 个作业不可区分，对手能逼任何算法按非增序调度、而最优是非减序，于是对手求解线性分式规划
$$\max_{\eta p\ge p_1^*\ge\cdots\ge p_n^*\ge p/\eta}\ \frac{\sum_{j=1}^n(n-j+1)p_j^*}{\sum_{j=1}^n j\,p_j^*}$$
最优解在多面体顶点取得，即每个作业大小取两端极值 $\eta p$ 或 $p/\eta$，最优分界比例 $\beta^*=\frac{1}{\eta+1}$。但静态对手不够：作者设计了一个**自适应对手**——让算法跑到 $t^*=(\eta p)\beta^* n$ 时刻，把已处理满 $\eta p$ 的记为大作业、已处理 $p/\eta$ 的立即完成记为中作业、其余设为小作业 $p/\eta$。这迫使算法在小误差时完成所有大作业、不留中作业（得下界 $\eta$），大误差时退化成 Round-Robin 均分处理器（得下界 2），中等误差时混用 SPJF 与 RR（得分段下界 $\lambda(\eta)$）。最终 Theorem 4.1 给出分段下界（$\eta\le\eta^*\approx1.835$ 时为 $\eta$，$\eta^*$ 是 $x^4-4x-4=0$ 的正根），Theorem 4.2 证明 **Shortest Predicted Job Size First (SPJF)** 达到 $\eta^2$ 竞争比。下界 $\eta$ 与上界 $\eta^2$ 间仍有 gap，闭合它是 open problem。

**3. 并行同速机 makespan：动态对手 + list-adjusting 技术**

针对 $Pm\,|\,\text{online-time-nclv, pmtn-restart}\,|\,C_{\max}$。Bampis et al. (2023) 首次给出该问题的带预测界：下界 $2-1/\lfloor\eta^2\rfloor$（静态对手）、上界 $\min\{2(\eta^2+1)/3,\,2\}$（Longest Predicted Job Size First）。本文两端都改进。**下界**用随误差变化的动态对手：核心观察是 makespan 主要由大作业的分配决定，于是让 $J_1$ 取最大尺寸 $\eta^2$（预测 $\eta$）、$J_2..J_{n-1}$ 尺寸接近 1（预测 $\eta$，与 $J_1$ 不可区分）、最小作业 $J_n$ 可见但影响微小。以 3 作业 2 机器为最简例可逼出 $\frac{1+\eta^2}{2}$；随 $\eta$ 增大用 7 作业 3 机器逼出 $\frac{\eta^2+2}{3}$，推广后得 Theorem 5.1 的 $\max\{1+\frac{\lfloor\eta^2\rfloor-1}{\lfloor\eta^2\rfloor},\,1+\frac{\eta^2-1}{\lceil\eta^2\rceil}\}=2-O(\eta^{-2})$，不超过 2（与 list scheduling 的 2-竞争一致）。**上界**的关键是 list-adjusting 技术：直接把 $(1+\epsilon)$-近似离线算法当成「预测即真值」来跑（记为 SIMPLE）能拿到 $1+\epsilon$ 一致性和 $O(\eta^2)$ 平滑性，但鲁棒性差到 $\Theta(m)$（最坏会把一堆作业堆到一台机器上）。修法是：若某机器 $M_i$ 做完了 SIMPLE 分给它的最后一个作业、而别处还有等待作业，就让它去跑一个等待作业而非空转。这保证最终调度可被 list scheduling 复现（故名 list-adjusting），从而继承 list scheduling 的 $2-\frac{1}{m}$ 鲁棒性。Theorem 5.4 因此给出 $\min\{\eta^2+\epsilon,\,2-\frac{1}{m}\}$ 竞争比，做到近最优 $1+\epsilon$ 一致性、$\eta^2+\epsilon$ 平滑性（$\eta^2<2$ 时）、最优 $2-\frac{1}{m}$ 鲁棒性。

**4. 相关速度机 makespan：紧的 $\lceil\log\eta\rceil$ 下界填补空白**

针对 $Qm\,|\,\text{online-time-nclv, pmtn-restart}\,|\,C_{\max}$。痛点是 Zhao et al. (2024) 已给出 $O(\min\{\log\eta,\log m\})$-竞争算法，但文献缺一个下界，无从判断它是否最优。本文用一个精巧的对抗实例补上这个空白（Theorem 6.1：$s_A(\eta)\ge\lceil\log\eta\rceil$）。构造借用 Cho-Sahni (1980) 攻击非透视调度的实例族：令 $k=\lceil2\log\eta\rceil+1$，建 $k$ 组机器 $G_i$ 与 $k$ 组作业 $T_i$，$G_i$ 中机器速度均为 $2^i$、$T_i$ 中作业大小均为 $2^i$，$|G_i|=|T_i|=2^{2k-2i-1}$，且这些作业预测都设为 $\eta$（彼此不可区分）；额外一组 $G_k,T_k$ 各含一台速度 $\delta$ 的机器和一个大小 $\delta$、预测正确的极小作业。这个实例满足：预测误差恰为 $\eta$、属于 $\mathcal{I}_s$、最优 makespan $C^*_{\max}=1$。证明的两个支柱是引理 6.3（完成任一组作业至少花 1 单位时间）和引理 6.4（相邻两组作业的完成时间差至少 $\frac{1}{2}$），二者叠加即得 makespan $\ge\lceil\log\eta\rceil$。由于已有 $O(\log\eta)$ 算法，这个下界**渐近紧**——意味着在相关速度机上，预测能带来的平滑性改进存在硬上限，已有算法已渐近最优地同时拿到一致性、平滑性、鲁棒性三者。

## 实验关键数据
本文为纯理论论文，没有数值实验，「关键数据」即三个问题在新平滑性指标下的界。下表汇总本文结果与已有界的对比。

### 主结果：三个调度问题的平滑性界

| 问题（Graham 记号） | 误差度量 | 已有界 | 本文下界 | 本文上界（算法） |
|---|---|---|---|---|
| 单机 $\sum C_j$ | 乘性 $\eta$ | $O(\eta^2\log\eta)$（由 Azar 2022 推得） | $\eta$（$\eta\le1.835$），$\lambda(\eta)$，$\to 2$ | $\eta^2$（SPJF） |
| 并行同速机 $C_{\max}$ | 乘性 $\eta$ | 下界 $2-1/\lfloor\eta^2\rfloor$；上界 $\min\{\tfrac{2(\eta^2+1)}{3},2\}$（Bampis 2023） | $2-O(\eta^{-2})$ | $\min\{\eta^2+\epsilon,\,2-\tfrac1m\}$（SIMPLE + list-adjusting） |
| 相关速度机 $C_{\max}$ | 乘性 $\eta$ | $O(\log\eta)$ 算法，**无下界** | $\lceil\log\eta\rceil$（渐近紧） | $O(\log\eta)$（Zhao 2024，已有） |

### 三个核心指标的对照

| 指标 | 度量域 | 直观含义 |
|---|---|---|
| 一致性 $c_A$ | $\mathcal{I}_c$（完美预测） | 预测无误差时的竞争比 |
| 鲁棒性 $r_A$ | $\mathcal{I}_r$（预测可伪装为无信息） | 预测最坏时的竞争比 |
| 平滑性 $s_A(\eta)$（本文精炼） | $\mathcal{I}_s$（预测保证有额外信息） | 误差为 $\eta$ 的合理预测下的竞争比 |

### 关键发现
- **三个问题的下界都不超过对应的鲁棒性常数**（如并行机 $\le 2$、单机 $\to 2$），这与「无论误差多大都存在常数竞争算法」自洽，说明新指标没有人为放大界。
- **单机问题仍留有 gap**：下界 $\eta$、上界 $\eta^2$，且分析显示 SPJF 对所构造的对手只达到 $\eta$，闭合该 gap 是公开问题；并行同速机同样存在上下界 gap。
- **相关速度机是唯一三界都渐近紧的问题**：配合 Shmoys et al. (1995) 的 $\Omega(\log m)$ 非透视下界，已有算法在一致性、平滑性、鲁棒性上**同时**渐近最优，展示了三指标可同时达最优的可能性。
- **新指标只收紧下界、不削弱已有上界**：所有下界（Thm 4.1/4.2/5.1/5.4/6.1）对不剔除 $\mathcal{I}_r$ 的备选定义 $s_A'$ 也成立。

## 亮点与洞察
- **「预测约束对手」这一视角很漂亮**：传统竞争分析里对手可以任意造实例，而有了预测后，对手被限制只能造「与预测一致到误差 $\eta$」的实例，于是对手要解一个**约束优化**（线性分式规划），策略变得复杂得多——平滑性下界本质上量化了这个约束给调度者带来的优势。这个 min-max 博弈框架可迁移到其他带预测在线问题。
- **谓词 $R(I)$ 是个可复用的「无信息」探测器**：作者在讨论里直接把它推广到单向交易（one-way trading）问题，给出对应的 $R(I)$ 形式。任何带预测的在线问题，只要能定义「存在等误差但不可区分的预测」就能套用这套三分 + 重定义平滑性的范式。
- **list-adjusting 是个朴素却有效的鲁棒化补丁**：把一个高一致性但不鲁棒的离线近似算法（SIMPLE），仅靠「空闲机器去捡等待作业」这一条规则就强行拉回 list scheduling 的鲁棒性保证，思路简单、可复用到其他「近似离线解 + 在线兜底」的组合算法。
- **下界的业务解读**：把 $\Omega(\eta)$ 下界翻译成「预测误差减半 → 目标最多翻倍改善」，给「该不该投资更准的预测模型」提供了量化依据，这种把理论界对接到决策的写法少见且实用。

## 局限与展望
- **上下界仍有 gap**：单机（$\eta$ vs $\eta^2$）和并行同速机问题都没闭合，作者明确把它列为未来工作；只有相关速度机做到了渐近紧。
- **只覆盖确定性算法与三个具体调度问题**：随机化算法、其他目标函数（如加权完成时间、流时间）下新平滑性指标会给出什么界，尚未触及。
- **新定义是否真能改进上界仍是 open**：作者诚实指出，目前所有上界对剔除/不剔除 $\mathcal{I}_r$ 两种定义都一样，是否存在某问题上 $s_A(\eta)<s_A'(\eta)$（即精炼定义真带来更优算法）还不知道。
- **$\mathcal{I}_s$ 的划分依赖误差度量**：三分结果（如 $\mathcal{I}_s=\{1<\eta<\sqrt{P}\}$）是针对乘性误差解析出来的，换一种误差度量需重新推导 $R(I)$ 与各子集的形状，迁移到新问题有一定门槛。

## 相关工作与启发
- **vs Azar et al. (2023) 的 discrete-smoothness**: 他们也把性能写成误差的函数来度量平滑性，但要求预测必须与优化问题的解同形；本文的平滑性不限制预测形式，而是从「实例是否含等误差常数向量」这个信息论角度切入，适用面更广。
- **vs Purohit et al. (2018) / Wei-Zhang (2020) / Bampis et al. (2022)**: 这些单机/多机完成时间工作都用尺度敏感的 $L_1$ 误差；本文改用尺度无关的乘性误差，并指出 $L_1$ 会因纯缩放而虚报 100 倍误差，乘性度量更贴合竞争比的实际行为。
- **vs Bampis et al. (2023)（并行同速机首个界）**: 他们用静态对手 + Longest Predicted Job Size First；本文用随误差自适应的动态对手收紧下界，并用 SIMPLE + list-adjusting 把上界从 $\min\{2(\eta^2+1)/3,2\}$ 改进到 $\min\{\eta^2+\epsilon,2-\frac1m\}$。
- **vs Zhao et al. (2022; 2024)（相关速度机）**: 他们给了 $O(\log\eta)$ 上界算法但缺下界；本文借 Cho-Sahni 实例族补上 $\lceil\log\eta\rceil$ 紧下界，使该问题成为三指标同时渐近最优的范例。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 重新定义一个被广泛使用却有缺陷的核心指标（平滑性），并用谓词 $R(I)$ 给出可推广的形式化，概念贡献扎实。
- 实验充分度: ⭐⭐⭐⭐ 纯理论论文，三个问题的下界 + 匹配算法覆盖完整、对比清晰，但单机/并行机仍留 gap。
- 写作质量: ⭐⭐⭐⭐⭐ 动机用反例讲得极清楚，对抗构造与博弈直觉解释到位，理论文章里属易读的。
- 价值: ⭐⭐⭐⭐ 为带预测算法社区提供了更有诊断力的平滑性框架，且可迁移到其他在线问题，但短期内偏理论、落地间接。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Non-Clairvoyant Scheduling with Progress Bars](../../NeurIPS2025/learning_theory/non-clairvoyant_scheduling_with_progress_bars.md)
- [\[ICLR 2026\] ATLAS: Alibaba Dataset and Benchmark for Learning-Augmented Scheduling](atlas_alibaba_dataset_and_benchmark_for_learning-augmented_scheduling.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](../../AAAI2026/learning_theory/a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[ICLR 2026\] Semi-Parametric Contextual Pricing with General Smoothness](semi-parametric_contextual_pricing_with_general_smoothness.md)
- [\[ICLR 2026\] Strong Correlations Induce Cause Only Predictions in Transformer Training](strong_correlations_induce_cause_only_predictions_in_transformer_training.md)

</div>

<!-- RELATED:END -->
