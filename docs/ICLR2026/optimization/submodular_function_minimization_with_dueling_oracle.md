---
title: >-
  [论文解读] Submodular Function Minimization with Dueling Oracle
description: >-
  [ICLR 2026][优化/理论][子模函数最小化] 在只能拿到"两个集合谁的函数值更大"这种带噪声成对比较反馈（dueling oracle）、完全没有函数值的情况下做子模函数最小化，作者用 Lovász 扩展 + SGD 构造子梯度估计器，针对线性传递函数给出 $O(n^{3/2}/\sqrt{T})$ 的误差界并配上匹配下界（在受限算法类里最优），针对 sigmoid 传递函数用 Firth 偏差校正法给出 $O(n^{7/5}/T^{2/5})$ 的误差界。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "子模函数最小化"
  - "对决预言机"
  - "Lovász 扩展"
  - "随机梯度下降"
  - "上下界"
---

# Submodular Function Minimization with Dueling Oracle

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=BeMtzSH1d7](https://openreview.net/forum?id=BeMtzSH1d7)  
**代码**: 补充材料附带（论文未给公开仓库链接）  
**领域**: 优化 / 子模优化 / 学习理论  
**关键词**: 子模函数最小化, 对决预言机, Lovász 扩展, 随机梯度下降, 上下界

## 一句话总结
在只能拿到"两个集合谁的函数值更大"这种带噪声成对比较反馈（dueling oracle）、完全没有函数值的情况下做子模函数最小化，作者用 Lovász 扩展 + SGD 构造子梯度估计器，针对线性传递函数给出 $O(n^{3/2}/\sqrt{T})$ 的误差界并配上匹配下界（在受限算法类里最优），针对 sigmoid 传递函数用 Firth 偏差校正法给出 $O(n^{7/5}/T^{2/5})$ 的误差界。

## 研究背景与动机
**领域现状**：子模函数最小化（SFM）是组合优化、机器学习、运筹学里的核心问题（图割、聚类、图像分割、价格优化都能归约到它）。经典做法（椭球法、组合强多项式算法）以及后来基于 Lovász 扩展把 SFM 转成凸最小化、再跑 SGD 的方法（Hazan & Kale 2012、Ito 2019），**都假设有一个值预言机**——查询一个集合 $S$ 就能拿到精确或带噪声的函数值 $f(S)$。

**现有痛点**：在很多真实场景里精确函数值根本拿不到，或者拿到也不可靠。比如给大模型做偏好标注、RLHF、推荐系统，人类很难给出可靠的绝对打分，但**很容易表达"A 和 B 哪个更好"这种相对偏好**。已有的"对决预言机（dueling oracle）"研究只覆盖了多臂赌博机和凸优化，**SFM 在纯 dueling 反馈、完全没有值预言机的设定下从没被研究过**，这篇是第一篇。

**核心矛盾**：基于 Lovász 扩展跑 SGD 的方法，命脉在于能否构造**无偏的子梯度估计器**——已有工作靠值预言机的反馈直接算出子梯度的无偏估计。但 dueling oracle 只返回一个带噪二元符号 $o\in\{\pm 1\}$，且这个符号的正确概率由一个**传递函数（transfer function）**$\rho$ 调制：函数值差越大、符号越可靠，差越小、几乎是抛硬币。当 $\rho$ 非线性（如 sigmoid）时，根本无法从这种反馈里凑出无偏子梯度，而带偏的估计器在 SGD 里会逐轮累积误差。

**本文目标**：把 SFM 的可解范围从"值预言机"推到"对决预言机"，并对两个最重要的传递函数——线性 $\rho(x)=ax$ 和 sigmoid $\rho(x)=2/(1+e^{-bx})-1$——分别给出**算法 + 误差上界 + 信息论下界**，回答"这个问题到底能做到多好"。

**切入角度**：dueling oracle 的响应概率写成 $\Pr(o=+1)=\tfrac12+\tfrac12\rho(f(S)-f(S'))$。线性情形下 $\rho$ 与函数值差成正比，$\mathbb{E}[o/a]$ 恰好等于函数值差，于是子梯度的无偏估计自然浮现；sigmoid 情形对应一个**逻辑回归模型**（也正是 Bradley–Terry 偏好模型），可以借统计学里的 Firth 偏差校正法把估计器的偏差压到 $O(1/k^2)$。

**核心 idea**：用对决反馈代替函数值来构造（线性时无偏、sigmoid 时小偏差的）子梯度估计器，在 Lovász 扩展上跑投影 SGD；对 sigmoid 引入 Firth 惩罚似然估计来对抗偏差累积，并重新设计学习率 $\eta$ 与每步重复查询次数 $k$ 来平衡"SGD 项 + 方差项 + 偏差项"三者。

## 方法详解

### 整体框架
整篇论文的算法骨架是同一条：**Lovász 扩展把离散 SFM 变成凸最小化 → 用投影 SGD 在超立方体 $K=[0,1]^n$ 上迭代 → 每一步用 dueling oracle 的反馈估计一个子梯度 → 平均迭代点再随机阈值取整还原成一个集合**。所有创新都落在"怎么从对决反馈里估子梯度"和"怎么配参数让误差最小"这两点上，传递函数线性还是 sigmoid，只换估计器和参数设置，主循环不变。

具体地，Lovász 扩展 $\hat f(w)$ 把定义在 $2^{[n]}$ 上的子模函数 $f$ 连续延拓到 $K$，且满足 $\min_{X} f(X)=\min_{w\in K}\hat f(w)$（Theorem 1），把组合问题转成凸问题。$\hat f$ 是分段线性的，每个线性区域里子梯度恒定：在点 $w$ 处按降序排出置换 $\pi$、令 $B_i=\{\pi(1),\dots,\pi(i)\}$，则子梯度第 $\pi(i)$ 个分量是 $g_{\pi(i)}=f(B_i)-f(B_{i-1})$（Proposition 2）。**关键观察**：这个分量正好是相邻两个嵌套集合的函数值差，而 dueling oracle 比较的就是两个集合的大小关系——于是查询 $(B_{\tau(i)},B_{\tau(i)-1})$ 这一对，就能拿到第 $i$ 个子梯度分量的信息。SGD 更新为 $w^{(t+1)}=\Pi_K(w^{(t)}-\eta\hat g_t)$，投影 $\Pi_K$ 就是逐分量裁剪到 $[0,1]$。跑完 $T'$ 步后取平均 $\bar w=\frac{1}{T'}\sum_t w^{(t)}$，再用 Lovász 扩展的积分表示 $\hat f(w)=\mathbb{E}_{z\sim U[0,1]}[f(\{i\mid w_i\ge z\})]$（Proposition 1），随机抽阈值 $z$ 输出集合 $\hat S=\{i\mid \bar w_i\ge z\}$，保证 $\mathbb{E}[f(\hat S)]=\hat f(\bar w)$。

### 关键设计

**1. 用对决反馈构造无偏子梯度：线性传递函数下的"免费午餐"**

线性 SFM 的核心难点是：没有函数值，怎么得到子梯度？作者抓住 Proposition 2 的结构——子梯度的每个分量就是一对嵌套集合的函数值差。线性传递函数 $\rho(x)=ax$ 下，dueling oracle 对查询 $(B_{\tau(i)},B_{\tau(i)-1})$ 返回 $o_{ti}\in\{\pm1\}$，满足 $\mathbb{E}[o_{ti}]=\rho(f(B_{\tau(i)})-f(B_{\tau(i)-1}))=a\,g_i$。于是 $\hat g_{ti}=o_{ti}/a$ 直接就是**无偏估计**：$\mathbb{E}[\hat g_{ti}]=g_i$。这一步是整篇论文最干净的地方——把"成对比较"和"分段线性子梯度"两个结构对齐，无偏估计器几乎是白送的，主循环和 Hazan & Kale (2012)、Ito (2019) 完全一样，唯一的差别就是估计器从"值预言机"换成了"对决反馈"。每个 SGD 步要查 $n$ 对（每个分量一对），所以总步数 $T'=T/n$，学习率取 $\eta=\frac{a}{2\sqrt{nT}}$。

**2. Firth 偏差校正：sigmoid 传递函数下抢救无偏性的失效**

sigmoid 情形 $\rho(x)=2/(1+e^{-bx})-1$ 不再线性，$\mathbb{E}[o]$ 和函数值差之间是非线性关系，**无偏子梯度估计器原则上构造不出来**。一个自然的退路是用带偏估计，但 SGD 会把每步的偏差逐轮累加，最终误差爆掉。前人 Saha et al. (2025) 在 dueling 凸优化里靠 $\beta$-光滑 + $\alpha$-强凸来控偏差，但 Lovász 扩展得到的凸函数既不光滑也不强凸（分段线性），那套技术用不上。

作者的关键一招是把问题**重新看成逻辑回归**：对查询 $(B_{\tau(i)},B_{\tau(i)-1})$ 重复查 $k$ 次，记 $k_+,k_-$ 为返回 $+1/-1$ 的次数，参数 $\theta=g_i$ 的响应概率 $\Pr(X=+1\mid\theta)=1/(1+e^{-b\theta})$ 正是 logistic 模型。直接用极大似然估计 $\frac1b\log(k_+/k_-)$ 有 $O(1/k)$ 的一阶偏差；作者引入 **Firth (1993) 的惩罚似然法**——对对数似然加上 $\frac12\log|I(\theta)|$（$I$ 是 Fisher 信息）这一项，能消掉指数族自然参数估计的一阶偏差。落到 logistic 上，惩罚极大似然估计有一个漂亮的闭式：
$$\hat\theta^*=\frac1b\log\frac{k_++\tfrac12}{k_-+\tfrac12}.$$
分子分母各加 $\tfrac12$ 这个"半计数"修正，把偏差从 $O(1/k)$ 压到 $O(1/k^2)$：$|\mathbb{E}[\hat\theta^*]-\theta|\le \frac{C(b)}{k^2}+O(1/k^3)$，其中 $C(b)=\big|\frac{2\psi-1}{24b\psi^2(1-\psi)^2}\big|$、$\psi=1/(1+e^{-b})$。这就是 sigmoid 算法（Algorithm 2）用作子梯度估计的 $\hat g_{ti}=\frac1b\log\frac{k_++1/2}{k_-+1/2}$。

**3. 三项平衡的参数设置：把偏差作为新增的第三个误差源统一调度**

无偏估计器消失后，不仅算法要改，**分析也整个变难**。前人 SFM 的误差分析里只有两项随学习率 $\eta$ 此消彼长（SGD 的"距离收缩项" $\propto n/\eta$ 和"方差项" $\propto \eta\sum\|\hat g_t\|^2$），选 $\eta$ 平衡两项就得到界。sigmoid 情形冒出来**第三项——子梯度估计的偏差**，而且多了一个旋钮：每步重复查询次数 $k$（$k$ 越大偏差越小，但消耗的预言机调用越多、SGD 步数 $T'$ 越少）。

作者据此给出一套新的参数规则，把三项一起压平。对 Theorem 2 给出的基础界 $\mathbb{E}[\hat f(\bar w)]-\min\hat f\le\frac1{T'}\big(\frac{n}{8\eta}+\frac\eta2\sum\mathbb{E}\|\hat g_t\|^2\big)$ 再叠加偏差项后做联合最优化，解得 $\eta=b^{6/5}C(b)^{1/5}n^{2/5}/T^{2/5}$、步数 $T'=T^{4/5}/(n^{4/5}b^{2/5}C(b)^{2/5})$、重复次数 $k=b^{2/5}C(b)^{2/5}T^{1/5}/n^{1/5}$（满足 $T'\cdot n\cdot k=T$ 的预算约束）。这套配比正是把上界做到 $O(n^{7/5}/T^{2/5})$ 的来源；$T$ 的指数从线性情形的 $1/2$ 退化成 $2/5$，正是"无法无偏"付出的代价。论文还顺带把一般非线性 $\rho$ 的上界做到 $O(n^{4/3}/T^{1/3})$。

**4. 成对查询下的下界：把单点查询的信息论论证推广到"查询对的分布"**

光有上界不知道好不好，作者还证了匹配下界。构造硬实例沿用 Ito (2019)，但**证明本身要重做**：前人的下界假设算法每次只查一个子集，而这里每次查一**对**子集，可选的数据采集方式多得多，信息论论证必须把"查询对上的概率分布"显式纳入，再去界定不同实例诱导出的分布之间的 KL 散度。技术路线是 Yao 原理（用确定性算法在精心构造的实例分布下的误差去界随机算法）+ 沿输入序列累加 KL 散度，最终说明：对两个最优解不同的目标函数，任何算法的输出分布都差不太开，因而必然在其中一个上吃到不可忽略的误差。

结论分两档：在 **Restriction 1**（每次查询的两集合对称差 $|S_t\triangle S_t'|=O(1)$，即只能查"邻近"的一对——Algorithm 1/2 都满足，因为它们查的 $(B_{\tau(i)},B_{\tau(i)-1})$ 对称差恒为 1）下，下界是 $\Omega(n^{3/2}/\sqrt T)$，**与线性算法上界完全吻合，证明 Algorithm 1 在受限算法类里常数因子最优**；去掉限制后下界降为 $\Omega(n/\sqrt T)$，意味着即使允许任意查询，误差最多只能再改进 $O(1/\sqrt n)$ 因子。

### 损失函数 / 训练策略
没有传统意义的训练损失——优化目标就是 Lovász 扩展 $\hat f$ 的凸最小化，"训练"即投影 SGD 迭代。关键超参在算法里写死成 $n,T,a$（或 $b$）的解析式：线性取 $\eta=\frac{a}{2\sqrt{nT}},\,T'=T/n$；sigmoid 取上面的三项平衡配比。最终解通过迭代平均 $\bar w$ + 随机阈值取整还原成离散集合，保证期望意义上 $\mathbb{E}[f(\hat S)]=\hat f(\bar w)$。

## 实验关键数据
这是一篇理论算法论文，核心贡献是上下界而非 benchmark。主结果以"误差率（error rate，$\mathbb{E}[E_T]=f(\hat S)-\min_S f(S)$ 关于预言机调用数 $T$ 和基集大小 $n$ 的渐近行为）"形式给出（Table 1）。

### 主结果：上下界对照

| 传递函数 | 上界（本文算法） | 下界（受 Restriction 1 约束） | 下界（无约束） | 上下界是否匹配 |
|----------|------------------|-------------------------------|----------------|----------------|
| 线性 $\rho(x)=ax$ | $O(n^{3/2}/\sqrt T)$ | $\Omega(n^{3/2}/\sqrt T)$ | $\Omega(n/\sqrt T)$ | $T$ 依赖匹配，受限类内常数因子最优 |
| sigmoid $\rho(x)=\tfrac{2}{1+e^{-bx}}-1$ | $O(n^{7/5}/T^{2/5})$ | $\Omega(n^{3/2}/\sqrt T)$ | $\Omega(n/\sqrt T)$ | $T$ 依赖有缺口（$2/5$ vs $1/2$） |
| 一般非线性 | $O(n^{4/3}/T^{1/3})$ | $\Omega(n^{3/2}/\sqrt T)$ | $\Omega(n/\sqrt T)$ | 缺口更大 |

> 误差界也可反读成查询复杂度：要达到误差 $\le\epsilon$，线性情形需 $T\propto n^3/\epsilon^2$ 次查询。

### 关键发现
- **线性是"干净"的情形**：因为能造无偏估计器，上下界在 $T$ 上完全吻合（都是 $1/\sqrt T$），算法在 Restriction 1 类里达到常数因子最优。
- **sigmoid 的代价写在指数上**：无偏估计失效后，$T$ 的指数从 $1/2$ 掉到 $2/5$，这正是 Firth 校正只能把偏差压到 $O(1/k^2)$、无法完全消除所付的价。作者据此猜想 sigmoid 上界还能收紧到 $O(1/\sqrt T)$，列为 open question。
- **$b$ 不能取极端值**：sigmoid 误差因子 $C(b)^{1/5}/b^{4/5}$ 在 $b\to 0^+$（反馈几乎纯噪声）和 $b\to+\infty$（反馈几乎确定但梯度信息退化）时都发散，说明算法只在"信噪适中"的对决预言机下有效。
- 附录 B 的数值实验进一步佐证：当真实传递函数非线性、但只在零点附近小邻域探索时，线性算法依然管用（因为任意光滑 $\rho$ 在零点附近可被线性近似）。

## 亮点与洞察
- **把成对比较和分段线性子梯度对齐**，是全文最优雅的一步：Lovász 扩展的子梯度分量天然就是"两个嵌套集合的函数值差"，而 dueling oracle 干的就是比较两个集合——两个看似无关的结构一拍即合，线性情形的无偏估计器几乎免费。
- **Firth 偏差校正法从统计学搬到优化里**很有迁移价值：凡是"只能拿带噪二元偏好、底层是逻辑回归/Bradley–Terry"的场景（RLHF、偏好对齐、点击建模），都可以用 $\frac1b\log\frac{k_++1/2}{k_-+1/2}$ 这个半计数修正把估计偏差从 $O(1/k)$ 降到 $O(1/k^2)$，且有闭式、零额外计算。
- **"三项平衡"的分析范式**点出了一个普适教训：当无偏估计器不可得时，误差分析里会多出一个偏差项和一个新旋钮（重复采样次数 $k$），正确做法不是各管各，而是把学习率 $\eta$ 和 $k$ 联立起来对三项做整体最优化。
- **下界证明把单点查询推广到查询对的分布**，给"成对比较型"在线优化问题提供了一套可复用的信息论模板（Yao 原理 + 查询对分布的 KL 散度）。

## 局限与展望
- **sigmoid/一般非线性上下界仍有缺口**：上界 $T^{-2/5}$、下界 $T^{-1/2}$，作者自己猜测上界能收紧到 $1/\sqrt T$ 但未证，是最大的未竟之处。
- **假设传递函数 $\rho$ 已知**：算法把 $a$ 或 $b$ 当输入。现实中 $\rho$ 的形状（噪声水平 $b$）往往未知，需要先估计或做无参版本，作者把这点列为重要 future work，提示可借 dueling 凸优化与无导数优化的技术。
- **只覆盖线性和 sigmoid 两个具体 $\rho$**：虽给了一般非线性的上界，但下界对所有 $\rho$ 都是同一档，未必紧；$n$ 依赖在线性情形是否能改进也是开放问题。
- **纯理论、数值实验仅在附录小规模验证**：没有真实推荐/RLHF 场景的端到端实证，"对决预言机能替代值预言机"在工程上的实际收益还需检验。
- **连续松弛的固有开销**：作者也提到 Lovász 扩展这条"连续化"路线未必最优，组合式方法（如 FTPL）可能更高效，是另一条可探索的路。

## 相关工作与启发
- **vs Hazan & Kale (2012)（在线子模最小化）**：他们在 bandit 设定下用值预言机 + Lovász 扩展 + OGD，无偏子梯度直接由观测损失构造，regret $O(nT^{2/3})$（折算 SFM 误差 $O(n/T^{1/3})$）。本文继承了"Lovász + SGD"骨架，但把反馈从"单集合的损失值"换成"成对比较符号"，线性情形误差直接做到 $O(n^{3/2}/\sqrt T)$ 且配匹配下界。
- **vs Ito (2019)（带噪值预言机的 SFM）**：Ito 用满足 $\mathbb{E}[\hat f(S)]=f(S)$ 的带噪**值**观测，误差 $O(n^{3/2}/\sqrt T)$。本文设定更弱（连带噪函数值都没有，只有相对比较），却在线性 $\rho$ 下达到同阶误差——说明"成对比较"在信息量上并不比"带噪绝对值"差。下界构造也沿用并扩展了 Ito 的硬实例。
- **vs Saha et al. (2025)（dueling 凸优化）**：同样面对"非线性 $\rho$ 下无法构造无偏子梯度"，但他们靠 $\beta$-光滑 + $\alpha$-强凸 + 缩放梯度来控偏差。Lovász 扩展得到的凸函数分段线性、既不光滑也不强凸，那套假设全失效；本文改用 Firth 惩罚似然从源头压偏差，是绕开光滑/强凸假设的另一条路。
- **vs Bradley–Terry / RLHF 偏好建模**：sigmoid 传递函数下的 dueling oracle 恰好是 Bradley–Terry 模型（$\Pr(o=+1)=\exp(\beta_{S'})/(\exp(\beta_{S'})+\exp(\beta_S))$，$\beta_S=-bf(S)$）。这把 SFM 和大模型偏好对齐里最常用的成对偏好模型在数学上接上了头，给"从偏好反馈做组合优化"提供了理论支点。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次研究纯对决预言机（无值预言机）下的 SFM，线性情形给出匹配上下界，Firth 校正用法新颖
- 实验充分度: ⭐⭐⭐⭐ 理论论文，上下界完整、附录有数值验证；但缺真实偏好场景的端到端实证
- 写作质量: ⭐⭐⭐⭐⭐ 问题设定、算法、上下界、证明思路层层递进，三项平衡和下界扩展讲得清楚
- 价值: ⭐⭐⭐⭐ 把组合优化与偏好/比较反馈接上，对 RLHF 时代"从成对比较做优化"有理论指导意义，sigmoid 缺口留有后续空间

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Activation Function Design Sustains Plasticity in Continual Learning](activation_function_design_sustains_plasticity_in_continual_learning.md)
- [\[ICLR 2026\] Reducing Contextual Stochastic Bilevel Optimization via Structured Function Approximation](reducing_contextual_stochastic_bilevel_optimization_via_structured_function_appr.md)
- [\[ICLR 2026\] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization](towards_understanding_the_calibration_benefits_of_sharpness-aware_minimization.md)
- [\[ICLR 2026\] Hyperbolic Aware Minimization: Implicit Bias for Sparsity](hyperbolic_aware_minimization_implicit_bias_for_sparsity.md)
- [\[ICLR 2026\] Efficient Submodular Maximization for Sums of Concave over Modular Functions](efficient_submodular_maximization_for_sums_of_concave_over_modular_functions.md)

</div>

<!-- RELATED:END -->
