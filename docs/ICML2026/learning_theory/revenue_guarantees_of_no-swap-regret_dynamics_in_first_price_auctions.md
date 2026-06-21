---
title: >-
  [论文解读] Revenue Guarantees of No-Swap-Regret Dynamics in First Price Auctions
description: >-
  [ICML2026][学习理论][一价拍卖] 在离散一价拍卖里证明：任意 $\epsilon$-近似相关均衡的收益至少是 $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$（$v_2$ 是次高估值），由此首次给出无交换遗憾竞拍者在一价拍卖中收益的**多项式收敛速率**——若用最优 $O(\sqrt{kT})$ 交换遗憾算法，只需 $O(k^5/\epsilon^2)$ 轮就能让时均收益逼近次高估值，把此前的拟多项式界 $k^{O(\log k)}$ 大幅改进。
tags:
  - "ICML2026"
  - "学习理论"
  - "在线学习"
  - "算法博弈论"
  - "一价拍卖"
  - "无交换遗憾"
  - "相关均衡"
  - "收敛速率"
  - "对偶拟合"
---

# Revenue Guarantees of No-Swap-Regret Dynamics in First Price Auctions

**会议**: ICML2026  
**arXiv**: [2606.06085](https://arxiv.org/abs/2606.06085)  
**代码**: 无（理论论文）  
**领域**: 学习理论 / 在线学习 / 算法博弈论  
**关键词**: 一价拍卖, 无交换遗憾, 相关均衡, 收敛速率, 对偶拟合

## 一句话总结
在离散一价拍卖里证明：任意 $\epsilon$-近似相关均衡的收益至少是 $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$（$v_2$ 是次高估值），由此首次给出无交换遗憾竞拍者在一价拍卖中收益的**多项式收敛速率**——若用最优 $O(\sqrt{kT})$ 交换遗憾算法，只需 $O(k^5/\epsilon^2)$ 轮就能让时均收益逼近次高估值，把此前的拟多项式界 $k^{O(\log k)}$ 大幅改进。

## 研究背景与动机

**领域现状**：广告拍卖是百亿级产业，近年从二价拍卖大规模转向一价拍卖（Google Ad Manager 2019 年 3 月就切换了）。二价拍卖里诚实出价是占优策略、好分析；一价拍卖则迫使竞拍者偏离直白策略，复杂且不可预测。现实中竞拍者越来越多地用在线学习/自动出价算法来根据对手行为调整出价，这类算法带有"无遗憾"（no-regret）保证——重复 $T$ 轮后，不管对手怎么出价，自己的总效用都逼近最优固定出价。于是一条研究线在问：当所有人都用在线学习算法出价时，一价拍卖会产生多少收益？

**现有痛点**：已有工作（[17, 19]）刻画了一价拍卖里相关均衡的集合，证明**精确**相关均衡的收益至少 $v_2-\Theta(1/k)$。结合"无交换遗憾算法的时均行为收敛到相关均衡"，可以推出 $T\to\infty$ 时收益逼近 $v_2$。但这只是渐近结论，**完全没给出达到该收益需要多少轮**。

**核心矛盾**：现实中轮数虽大（广告拍卖每毫秒一次）但有限。在有限 $T$ 下，无交换遗憾动态收敛到的是**近似**相关均衡而非精确相关均衡，而 [17, 19] 的论证只对精确相关均衡成立。把它们的论证勉强推广到 $\epsilon$-近似相关均衡，收益保证会对 $k$ 呈现**拟多项式**依赖：收益至少 $v_2-\Theta(1/k)-\epsilon\cdot k^{\text{poly}(\log k)}$。这意味着即便用最优算法（交换遗憾 $O(\sqrt{kT})$），所需轮数上界也高达 $k^{O(\log k)}$——大到拍卖方实际上永远等不到那个高收益。

**本文目标**：回答 Question 1.5——任意 $\epsilon$-近似相关均衡的收益能否被界为 $v_2-\Theta(1/k)-\epsilon\cdot\text{poly}(k)$（即对 $\epsilon$ 只付多项式而非拟多项式代价）？

**切入角度**：作者完全抛弃 [17, 19] 的论证路线，转而引入近似算法里的**对偶拟合（dual-fitting）**技术。据作者所知，这是首次把对偶拟合用于建立收敛速率。

**核心 idea**：构造一个线性规划，其最优值 $Z^\star_{\text{LP}}$ 是任意 $\epsilon$-近似相关均衡收益的下界；再对其对偶给出一组显式可行赋值，用弱对偶把收益下界拉到 $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$。

## 方法详解

### 整体框架
论文的"方法"是一套证明流程，目标是给离散一价拍卖里 $\epsilon$-近似相关均衡的收益一个尽可能紧的多项式下界。设定：$n$ 个竞拍者，离散出价集 $B=\{0,1/k,\dots,1-1/k,1\}$，竞拍者 $i$ 私有估值 $v_i$、出价不超过估值，最高出价者赢、平局随机分配。收益定义为 $\sum_{s}\mu(s)\cdot r(s)$，其中 $r(s)=\max(s_1,\dots,s_n)$ 是获胜出价。整条论证分三段串起来：**(1)** 把"求任意 $\epsilon$-近似相关均衡的最小收益"松弛成一个线性规划，其最优值是真实收益下界；**(2)** 用对偶拟合给对偶变量一组显式赋值，证明它对偶可行、目标值达到 $v_2-3/k-\Theta(\epsilon k^2)$，再用弱对偶得到主结果；**(3)** 把这个均衡级的收益界，通过"无交换遗憾时均行为 → 近似相关均衡"的转换，翻译成具体的轮数收敛速率。最后再补一个匹配上界说明 $\epsilon$ 依赖无法去掉。

### 关键设计

**1. LP 松弛：把收益下界问题写成线性规划**

作者先把"在所有 $\epsilon$-近似相关均衡中收益最小是多少"写成一个以 $\mu(\cdot)$ 为变量的线性规划（Definition 3.1）：目标 $\min\sum_s \mu(s)r(s)$，约束是近似相关均衡的偏离条件 $\sum_{s_{-i}}\mu(s_i,s_{-i})(U_i(s_i',s_{-i})-U_i(s_i,s_{-i}))\le \epsilon$（对每个竞拍者 $i$、每对出价 $s_i,s_i'$），外加 $\mu$ 是概率分布。关键观察（Lemma 3.2）是：任意 $\epsilon$-近似相关均衡都满足这些约束、是 LP 的可行解，所以 LP 最优值 $Z^\star_{\text{LP}}$ 就是任意近似相关均衡收益的下界。这一步把一个"对所有均衡求下确界"的难题，转化成一个干净的优化对象——证一个 $Z^\star_{\text{LP}}$ 的下界就够了。

**2. 对偶拟合：给对偶变量一组显式可行赋值**

直接求 LP 最优值很难，作者转向它的对偶（Lemma 3.3）。对偶变量是一个标量 $\mu\in\mathbb{R}$ 和一族 $\lambda^i_{s_is_i'}\ge 0$，对偶可行要求 $\mu+\sum_i\sum_{s_i'}\lambda^i_{s_is_i'}(U_i(s_i,s_{-i})-U_i(s_i',s_{-i}))\le r(s)$ 对所有出价组合成立。对偶拟合的精髓就是**猜一组好的赋值**：作者令所有 $i\ge 3$ 的竞拍者 $\hat\lambda^i=0$，只给前两名（估值最高的两位）赋非零值——

$$\hat\lambda^i_{s_is_i'} := \begin{cases} 1, & s_i+1/k \le s_i' \le v_2-2/k \\ 0, & \text{otherwise} \end{cases}\quad (i\in\{1,2\})$$

并取 $\hat\mu := v_2-3/k$。然后证明这组赋值确实对偶可行（Lemma 3.7：对所有出价组合 $\hat b_s\ge 1-3/k$，通过把出价空间分成 4 类逐一验证），且对偶目标 $\hat\mu-\epsilon\sum_{i,s_i,s_i'}\hat\lambda^i_{s_is_i'}\ge v_2-3/k-2\epsilon k^2$。一旦对偶可行 + 目标值有下界，由**弱对偶**立即得 $\sum_s\mu(s)r(s)\ge Z^\star_{\text{LP}}\ge v_2-3/k-\Theta(\epsilon k^2)$，即主结果 Theorem 2.6。这套思路之所以有效，是因为对偶拟合不需要解出最优对偶解，只要构造**一个**可行解就能给主问题最优值一个下界——把原本拟多项式的论证一下压到多项式 $k^2$。

**3. 从均衡界到收敛速率：补上"多少轮"的答案**

主结果是关于均衡的静态界，作者再把它接到动态上。已知（Theorem 2.3）若每个竞拍者用遗憾 $R_i(T)$ 的无交换遗憾算法，则 $T=O(\max_i R_i(T)/\epsilon+\log(n/\delta)/\epsilon)$ 轮后时均分布以高概率是 $\epsilon$-近似相关均衡。把它和 Theorem 2.6 组合（Corollary 2.7）得：时均收益 $\frac{1}{T}\sum_t r(s^t)\ge v_2-3/k-O(\epsilon)$，所需轮数

$$T=O\!\left(k^2\max_i R_i(T)/\epsilon + k^2\log(n/\delta)/\epsilon\right).$$

若所有人用 [10] 的最优算法（交换遗憾 $O(\sqrt{kT})$），则 $T=O(k^5\log(n/\delta)/\epsilon^2)$ 轮足矣——相比此前 $k^{O(\log k)}/\epsilon^2$ 是质的飞跃。这一步才真正回答了 Question 1.1"多少轮"。

**4. 匹配上界：证明对 $\epsilon$ 的依赖无法消除**

为说明界的形状是对的，作者还给出一个收益上界（Theorem 2.8）：存在两个竞拍者 $v_1=v_2=1$ 的离散一价拍卖，其某个 $\epsilon$-近似相关均衡收益 $\le 1-\Theta(1/k)-\Theta(\epsilon)$。这说明收益里的 $O(\epsilon)$ 项是本质的、去不掉的。注意下界 Theorem 2.6 是 $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$、上界 Theorem 2.8 是 $v_2-\Theta(1/k)-\Theta(\epsilon)$，两者在 $\epsilon$ 的系数（$k^2$ vs 常数）上还有 gap，作者据实验猜测正确答案是上界那一侧。

### 损失函数 / 训练策略
本文为纯理论论文，无训练目标。唯一的"算法"是被分析的对象——[10] 提出的最优无交换遗憾在线学习算法，其交换遗憾 $R^{\text{swap}}_A(T)=O(\sqrt{|S_i|T})=O(\sqrt{kT})$，是可能达到的最优无交换遗憾。

## 实验关键数据

### 主实验
理论结果对比（核心是收敛轮数从拟多项式改进到多项式）：

| 结果 | 收益下界 | 达到 $v_2-\Theta(1/k)$ 所需轮数 $T$ |
|------|---------|-----------------------------------|
| 此前推广 [17,19]（Theorem 1.4） | $v_2-\Theta(1/k)-\epsilon\cdot k^{\text{poly}(\log k)}$ | $k^{O(\log k)}/\epsilon^2$（拟多项式） |
| **本文主结果（Theorem 2.6）** | $v_2-\Theta(1/k)-\Theta(\epsilon k^2)$ | $O(k^5/\epsilon^2)$（多项式） |
| **本文上界（Theorem 2.8）** | $v_2-\Theta(1/k)-\Theta(\epsilon)$（存在性，反向） | — |

数值实验验证（用 [10] 的 $O(\sqrt{kT})$ 算法跑重复一价拍卖，时均收益约束 $\frac{1}{T}\sum_t r(t)\ge v_2-3/k-O(k^{2.5}/\sqrt{T})$）：

| 实验设定 | 参数 | 观察 |
|---------|------|------|
| 确定性估值 | $k=10$，3 人，$v_1=1\ge v_2\ge v_3=0$，$v_2\in\{0,0.2,0.5,0.8,1\}$ | 各情形时均收益均超过 $v_2-0.1$，远快于 $O(k^7)$ |
| 离散化 $k$ 的权衡 | 3 人，$v_1=v_2=1,v_3=0$，$k\in\{10,30,50\}$ | $k$ 越大长期收益越高但收敛越慢，仍比理论预测快得多 |
| 贝叶斯 i.i.d. 设定 | $v_i\in\{0,1\}$ 各 1/2，$n\in\{2,3,4,5\}$，$k=10$ | 时均收益收敛到 $\mathbb{E}[v_2]=1-(n+1)/2^n$ |
| 贝叶斯混合纳什 | $n=2$，连续空间唯一均衡 CDF $F_P(x)=1/(1-x)-1$ | 时均出价行为非常接近贝叶斯混合纳什均衡 |

### 关键发现
- **实验收敛比理论快**：所有数值实验里时均收益都比 Theorem 2.6 预测的更快逼近 $v_2$，这成为作者猜测"正确界是上界 Theorem 2.8（$\Theta(\epsilon)$ 而非 $\Theta(\epsilon k^2)$）"的主要依据。
- **离散化的双刃剑**：增大 $k$ 让长期可达收益更高（$v_2-1/k$ 更接近 $v_2$），但收敛速率变慢——这正是 $T=O(k^5/\epsilon^2)$ 里 $k$ 高次幂的现实体现。
- **贝叶斯设定外推**：即便估值随机，无交换遗憾动态产生的收益仍逼近期望次高估值 $\mathbb{E}[v_2]$，且出价行为逼近贝叶斯混合纳什均衡，暗示对偶拟合有望推广到贝叶斯设定。

## 亮点与洞察
- **首次把对偶拟合用于收敛速率分析**：对偶拟合本是近似算法的工具，作者把它搬到博弈动态的收益分析里——只构造一个对偶可行解就能给收益一个紧下界，绕开了对所有近似相关均衡直接论证的拟多项式难题。这个思路本身可能对其它"动态收敛速率"问题有独立价值。
- **只给前两名赋非零对偶变量**：对偶赋值（Definition 3.5/3.6）抓住了一价拍卖收益由前两高估值主导的结构，让 $i\ge 3$ 的变量全清零、把组合复杂度压下来——这是把界从 $k^{O(\log k)}$ 压到 $k^2$ 的关键技巧。
- **上下界配套**：不只给下界，还配一个反向构造的上界证明 $\epsilon$ 依赖本质存在，把问题的边界划清楚，留下一个明确的 $k^2$ vs 常数 的 gap 作为后续靶子。

## 局限与展望
- **上下界仍有 gap**：下界 $\Theta(\epsilon k^2)$、上界 $\Theta(\epsilon)$，作者自己猜测正确答案是上界一侧但未证明，关掉这个 gap、拿到更快收敛速率是明确的开放问题。
- **只覆盖无交换遗憾**：结果针对无交换遗憾动态（收敛到相关均衡）。更弱的无（外部）遗憾动态收敛到的是粗相关均衡，本文方法是否适用未涉及。
- **贝叶斯设定只有实验**：i.i.d./贝叶斯设定下只给了实验观察（收益逼近 $\mathbb{E}[v_2]$），对偶拟合在贝叶斯设定下的理论分析作者列为未来方向。

## 相关工作与启发
- **vs [17, 19]（相关均衡刻画）**: 它们刻画一价拍卖里精确相关均衡的收益（连续/极限），但推广到近似相关均衡只能得拟多项式界；本文换用对偶拟合得多项式界，且直接给出有限轮收敛速率。
- **vs [20, 31, 15]（无遗憾收敛）**: [20] 需初始探索阶段、[31] 证明若收敛则收敛到次高价、[15] 限两个最高估值竞拍者；本文对**所有**无交换遗憾动态给出无条件收敛结果且带显式速率。
- **vs [2]（强对偶 → 纳什）**: [2] 用强对偶证明投影梯度上升收敛到纳什均衡；本文用弱对偶 + 对偶拟合给收益下界，针对无交换遗憾这一更广动态类。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把对偶拟合引入收敛速率分析，把拟多项式界压到多项式
- 实验充分度: ⭐⭐⭐⭐ 理论论文，数值实验覆盖确定性/贝叶斯多设定且支撑了猜想
- 写作质量: ⭐⭐⭐⭐ 问题动机（有限轮 vs 渐近）提得清楚，论证主线清晰
- 价值: ⭐⭐⭐⭐ 给自动出价/在线学习拍卖设计提供首个收益收敛轮数保证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] On the Robustness of Langevin Dynamics to Score Function Error](on_the_robustness_of_langevin_dynamics_to_score_function_error.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](../../ICLR2026/learning_theory/the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICML 2026\] Formalizing Learning from Language Feedback with Provable Guarantees](formalizing_learning_from_language_feedback_with_provable_guarantees.md)
- [\[ICML 2026\] On Regret Bounds of Thompson Sampling for Bayesian Optimization](on_regret_bounds_of_thompson_sampling_for_bayesian_optimization.md)
- [\[ICLR 2026\] A Faster Parameter-Free Regret Matching Algorithm](../../ICLR2026/learning_theory/a_faster_parameter-free_regret_matching_algorithm.md)

</div>

<!-- RELATED:END -->
