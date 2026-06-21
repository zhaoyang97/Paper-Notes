---
title: >-
  [论文解读] Proving the Limited Scalability of Centralized Distributed Optimization via a New Lower Bound Construction
description: >-
  [ICLR 2026][优化/理论][联邦学习] 这篇论文证明：在经典联邦学习（中心化分布式优化）里，一旦把"服务器→worker"方向的通信代价 $\tau_s$ 算进来，再用无偏稀疏化压缩器也无法让"通信项 $\tau_s d L\Delta/\varepsilon$"和"方差项 $h\sigma^2 L\Delta/\varepsilon^2$"同时随 worker 数 $n$ 改善——即便在所有 worker 共享同一函数的同质（i.i.d.）设定下，二者的可改善幅度也最多只有 $n$ 的多项式对数 $\mathrm{poly}\log n$ 倍；为此作者造了一个全新的链式 worst-case 函数并发展了把下界归约成"随…
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "联邦学习"
  - "通信压缩"
  - "非凸优化下界"
  - "worst-case 构造"
  - "可扩展性"
---

# Proving the Limited Scalability of Centralized Distributed Optimization via a New Lower Bound Construction

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=0KXI6lDM9C](https://openreview.net/forum?id=0KXI6lDM9C)  
**代码**: 待确认（理论论文，无代码）  
**领域**: 分布式优化理论 / 下界  
**关键词**: 联邦学习, 通信压缩, 非凸优化下界, worst-case 构造, 可扩展性

## 一句话总结
这篇论文证明：在经典联邦学习（中心化分布式优化）里，一旦把"服务器→worker"方向的通信代价 $\tau_s$ 算进来，再用无偏稀疏化压缩器也无法让"通信项 $\tau_s d L\Delta/\varepsilon$"和"方差项 $h\sigma^2 L\Delta/\varepsilon^2$"同时随 worker 数 $n$ 改善——即便在所有 worker 共享同一函数的同质（i.i.d.）设定下，二者的可改善幅度也最多只有 $n$ 的多项式对数 $\mathrm{poly}\log n$ 倍；为此作者造了一个全新的链式 worst-case 函数并发展了把下界归约成"随机和集中不等式"的证明框架。

## 研究背景与动机

**领域现状**：分布式优化用很多 worker（CPU/GPU/手机）连到一个中心服务器，大家各自算随机梯度、传给服务器聚合再广播回来，目标是求 $L$-光滑非凸函数 $f$ 的 $\varepsilon$-稳定点（$\mathbb{E}[\|\nabla f(\bar x)\|^2]\le\varepsilon$）。用很多 worker 的根本动机就是"可扩展性"：worker 数 $n$ 越多，跑得越快。理论上这点是成立的——比如同步 SGD 的时间复杂度是 $O\!\big(h(\tfrac{L\Delta}{\varepsilon}+\tfrac{\sigma^2 L\Delta}{n\varepsilon^2})\big)$，其中"统计项" $h\sigma^2 L\Delta/(n\varepsilon^2)$ 随 $n$ 线性变好；再配上无偏稀疏化压缩器（如 RandK），连"通信项"也能随 $n$ 改善，甚至出现一个随 $\sqrt n$ 改善的耦合项 $\sqrt{d\tau_w h\sigma^2/(n\varepsilon)}\cdot L\Delta/\varepsilon$。

**现有痛点**：上面这套"越多 worker 越快"的乐观结论，几乎都建立在一个不老实的假设上——服务器→worker 的广播是免费的，即 $\tau_s=0$。但在真实的互联网 / 4G/5G 网络里，下行广播同样昂贵。一旦认账 $\tau_s>0$，Batch QSGD 的复杂度里就会多出一项 $\tau_s d L\Delta/\varepsilon$（见式 (4)），而这一项**完全不随 $n$ 改善**：服务器每轮都得把 $d$ 维的 $x^k$ 发给所有 worker，再多的 worker 也分担不了这份下行带宽。

**核心矛盾**：异质（non-i.i.d.）情形下，Gruntkowska et al. (2024) 已经证明了迭代复杂度不随 $n$ 改善。但同质（i.i.d.，所有 worker 拿到同一个 $f$/分布）按直觉"更简单"——既然大家函数相同，是不是能让 worker 们协同探索坐标、把下行代价 $\tau_s d$ 也摊薄到随 $n$ 变好？这就是矛盾所在：直觉觉得有希望，但没人证明过到底行不行。

**本文目标**：在最简单的同质设定下，问一个干净的判定问题——能不能设计一个用无偏稀疏化压缩器的方法，让下行通信项 $\tau_s d L\Delta/\varepsilon$ 和方差项 $h\sigma^2 L\Delta/\varepsilon^2$ **同时**随 $n$（线性或 $\sqrt n$）改善？

**切入角度**：下界。证"做不到"比证"能做到"更彻底——只要构造出一个 worst-case 函数 + worst-case 压缩器，让任何"零尊重"（zero-respecting）算法在它上面都快不起来，就一劳永逸地堵死了整类方法。难点是：异质情形的旧构造（把式 (6) 的不同 block 分给不同 worker）在同质下失效，因为同质时所有 worker 都能同时往下一个坐标推进，旧论证里那个能省下 $n$ 倍的 $\min_{i\in[n]}$ 反而帮了算法的忙。

**核心 idea**：把 worst-case 函数从"发现下一个坐标只需上一个坐标非零"改成"必须连续 $K$ 个坐标都非零才能往前一步"。这条 $K$-链让"最幸运的 worker"也得同时凑齐 $K$ 个稀有事件，于是把下界里 $\min_{i\in[n]}$ 带来的 $1/n$ 衰减削弱成 $1/n^{1/K}$；再取 $K=\Theta(\log n)$ 让 $n^{1/K}=\Theta(1)$，$n$ 的红利就被彻底抹平，只剩多项式对数。

## 方法详解

### 整体框架

这是一篇纯下界论文，没有算法、没有实验图，"方法"指的是**怎么把"无法随 $n$ 扩展"这件事证出来**。整条证明链是：① 先固定一个足够一般的分布式方法族（Protocol 1 + 零尊重算法 $\mathcal{A}_{zr}$），它涵盖 GD/Adam/MARINA-P 等几乎所有"基于已见坐标 span 行动"的方法；② 给所有 worker 喂同一个新构造的 worst-case 函数 $F_{T,K,a}$，并用 worst-case 的随机稀疏化压缩器（RandK 类）作下行通信；③ 证明在这套构造下，算法"发现第 $T$ 个坐标（从而才可能到 $\varepsilon$-稳定点）"所需的最短随机时间等于一个特定的随机和 $t_B$（式 (13)）；④ 对 $t_B$ 证一个集中不等式，得到 $t_B$ 的高概率下界；⑤ 代入 $T,p_\sigma,p_K$ 的取值并选 $K=\Theta(\log n)$、$a=1+1/K$，收尾成主定理。

为什么不画 pipeline 框架图：这里没有"输入→模块→输出"的数据流，整体是"构造函数 → 归约成随机和 → 集中不等式"的证明推理，画 flowchart 反而失真，下面用文字 + 公式讲清三块贡献即可。三块的对应关系是：**关键设计 1** = 第②步的新函数；**关键设计 2** = 第③④步的随机和归约与集中；**关键设计 3** = 第⑤步那个一锤定音的 $K=\Theta(\log n)$ 选择；末尾再补一句把上行 $\tau_w>0$ 也并进来的扩展。

先回顾被替换掉的经典构造：Carmon et al. (2020) 的链式函数
$$F_T(x)=\sum_{i=1}^{T}\big(\Psi(-x_{i-1})\Phi(-x_i)-\Psi(x_{i-1})\Phi(x_i)\big),\quad x_0\equiv1,$$
它的关键性质是"梯度只在 $\mathrm{prog}(x)+1$ 处可能非零"——从 $x_i=0$ 出发，一次梯度查询最多"点亮"下一个坐标，所以要找到第 $T$ 个坐标至少得算 $T=\Theta(L\Delta/\varepsilon)$ 次梯度。异质下界（Gruntkowska et al. 2024）把这条链的不同 block 分给不同 worker，于是"同一时刻只有一个 worker 能推进"，下行代价拿不到 $1/n$ 折扣；但同质下只能给所有 worker 同一个 $F_T$，于是所有 worker 都能同时推进，下界里冒出 $\min_{i\in[n]}$，反而只证得出 $\tau_s\sum_j\min_{i\in[n]}\eta_{ji}\gtrsim\tau_s\tfrac{d}{n}\tfrac{L\Delta}{\varepsilon}$（式 (8)）——白白随 $n$ 变好。这正是必须换构造的原因。

### 关键设计

**1. K-链 worst-case 函数 $F_{T,K,a}$：把"推进一步"从需要 1 个非零坐标改成需要 K 个**

旧构造里 worker 只要上一坐标 $x_{i-1}\ne0$ 就能发现第 $i$ 个坐标，这个"门槛太低"让同质下所有 worker 都能轻易同步推进。作者把负 block $-\Psi(x_{i-1})\Phi(x_i)$ 换成 $K$ 个 $\Psi_a$ 连乘：
$$F_{T,K,a}(x)=-\sum_{i=1}^{T}\Psi_a(x_{i-K})\cdots\Psi_a(x_{i-2})\Psi_a(x_{i-1})\Phi(x_i)+\sum_{i=1}^{T}\Gamma(x_i),$$
其中 $x_0=\cdots=x_{-K+1}\equiv1$，$\Gamma(x)=-x e^{1/x+1}$（$x<0$）/ $0$（$x\ge0$）。新的进度量是
$$\mathrm{prog}_K(x):=\max\{i\ge0:\ x_i\ne0,\ x_{i-1}\ne0,\dots,x_{i-K+1}\ne0\},$$
即"必须连续 $K$ 个坐标都非零，才算推进到这里"。Lemma 3.1 保证 $\mathrm{supp}(\nabla F_{T,K,a}(x))\subseteq\{1,\dots,\mathrm{prog}_K(x)+1\}\cup\mathrm{supp}(x)$，且只要 $\mathrm{prog}_K(x)<T$ 就有 $\|\nabla F_{T,K,a}(x)\|>1$——梯度还没到底就不可能是 $\varepsilon$-稳定点。$\Gamma$ 项替代了旧构造里防止算法"向上爬到负侧附近稳定点抄近路"的正 block。

代价是连乘 $K$ 个 $\Psi$ 会让函数几何"变野"：函数值落差、光滑常数、$\ell_\infty$ 梯度上界都可能随 $K$ 指数级膨胀。作者引入参数 $a\in(1,e]$ 把 $\Psi$ 改成带 $\log a$ 的版本来压制（$a=e$ 时退回原 $\Psi$）。Lemma 3.2 给出可控的常数：$\Delta_0(K,a)=\sqrt{2\pi e}\,a^K$、$\ell_1(K,a)=\tfrac{12\sqrt{2\pi}e^{5/2}K^2 a^K}{\log a}$、$\gamma_\infty(K,a)=6\sqrt{2\pi}e^{3/2}\tfrac{K a^K}{\sqrt{\log a}}$。关键在于取 $1<a\ll e$ 能抵消 $K$ 带来的爆炸——这是后面能选 $K=\Theta(\log n)$ 而不让常数失控的前提。

**2. 把下界归约成随机和 $t_B$ 的集中不等式：用"双重计数 + 求和"对冲掉 $\min_{i\in[n]}$**

有了 $K$-链，作者把"算法发现第 $T$ 个坐标所需的最短随机时间"精确刻画成一个随机和。worker 想点亮新坐标有两条路：要么本地算到一个"幸运"的随机梯度（最后一个坐标没被随机梯度 oracle 置零，概率 $p_\sigma=\Theta(\varepsilon\gamma_\infty^2(K,a)/\sigma^2)$），要么从服务器收到的坐标流里收到一个落在 $[K]$ 内的坐标（概率 $p_K:=2K/T$）。定义 $\eta_{b,i,k}$ 为"第 $k$ 个幸运随机梯度前算了多少次梯度"、$\mu_{b,i,k}$ 为"收到第 $k$ 个 $[K]$-坐标前收了多少个坐标"，二者近似服从参数 $p_\sigma$、$p_K$ 的几何分布。要凑齐一段 $K$ 坐标，两条路至少有一条得贡献 $K/2$ 个，于是发现第 $T$ 个坐标的总时间不小于
$$t_B:=\sum_{b=1}^{B}\min_{i\in[n]}\Big\{\min\Big\{\,h\!\sum_{k=1}^{K/2}\eta_{b,i,k},\ \ \tau_s\!\sum_{k=1}^{K/2}\mu_{b,i,k}\Big\}\Big\},\qquad B=\lfloor T/K\rfloor.$$
和式 (8) 一样这里也有 $\min_{i\in[n]}$（只需等"最幸运的 worker"），照理会随 $n$ 改善。**但新构造的精髓**：每个 worker 内部多了 $\sum_{k=1}^{K/2}$ 这一层求和，幸运的 worker 不再是"碰一次运气"，而要"连中 $K/2$ 次"，于是 $\min_{i\in[n]}$ 的衰减被求和大幅对冲。作者为 $t_B$ 证了一个集中不等式：
$$t_B\ \gtrsim\ \frac{BK}{n^{1/K}}\,\min\{h/p_\sigma,\ \tau_s/p_K\}\quad(\text{高概率}),$$
注意第一项分母是 $n^{1/K}$ 而不是 $n$——这正是 $K$-链换来的：$K$ 越大，$n$ 的影响越小。

**3. 取 $K=\Theta(\log n)$、$a=1+1/K$：把 $n$ 的红利压成多项式对数屏障**

前两步留下一个张力：$K$ 越大越能压住 $n^{1/K}$，但 Lemma 3.2 的常数又随 $K$ 增长，不能取太大。代回 $T,p_\sigma,p_K$ 后得到
$$t_B\ \gtrsim\ \frac{L\Delta}{n^{1/K}\Delta_0(K,a)\,\ell_1(K,a)\,\varepsilon}\,\min\Big\{\max\Big\{h,\ \tfrac{h\sigma^2}{\varepsilon\gamma_\infty^2(K,a)}\Big\},\ \tfrac{\tau_s d}{K}\Big\}.$$
最后一锤定音：选 $K=\Theta(\log n)$，则 $n^{1/K}=n^{1/\Theta(\log n)}=\Theta(1)$，$n$ 直接从主项里消失；再取 $a=1+1/K$ 让 $\Delta_0,\ell_1,\gamma_\infty$ 只剩 $\mathrm{poly}\log n$ 量级。于是得到 Theorem 4.2（informal 式 (11)）：
$$\widetilde\Omega\Big(\min\Big\{\tau_s d\,\tfrac{L\Delta}{\varepsilon},\ \ h\tfrac{L\Delta}{\varepsilon}+h\tfrac{\sigma^2 L\Delta}{\varepsilon^2}\Big\}\Big).$$
含义：要么改善对 $d$ 的依赖、要么改善对 $\sigma^2/\varepsilon$ 的依赖，**二者不可兼得**——且 $d$ 与 $\sigma^2/\varepsilon$ 随 $n$ 的可改善幅度分别封顶在 $\log^4(n+1)$、$\log^6(n+1)$，比线性 $n$、$\sqrt n$ 差了一整个量级。

**4. 并入上行通信 $\tau_w>0$ 得到完整下界**

Theorem 4.2 只算了下行 $\tau_s$。作者另证 Theorem F.1，扩展并改进了 Tyurin et al. (2024) 在本场景下的结果（原结果对此设定不够用），把上行代价 $\tau_w$ 也并进来，得到 Theorem 1.6 的完整下界 (5)。当 $\tau_s\simeq\tau_w$ 时它化简为
$$\widetilde\Omega\Big(\min\Big\{h\big(\tfrac{\sigma^2}{n\varepsilon}+1\big)\tfrac{L\Delta}{\varepsilon}+\tau_s d\,\tfrac{L\Delta}{\varepsilon},\ \ h\tfrac{L\Delta}{\varepsilon}+h\tfrac{\sigma^2 L\Delta}{\varepsilon^2}\Big\}\Big),$$
恰好能被**不压缩**的 Batch Synchronized SGD（式 (2)）或本地裸 SGD 匹配——也就是说 $\tau_s\simeq\tau_w$ 时无偏稀疏化压缩**毫无用处**。只有当 $\tau_s\lesssim\tau_w$、且 $\tau_w d/n+\sqrt{d\tau_w h\sigma^2/(n\varepsilon)}$ 比 $\tau_s d$ 大时，worker 侧压缩才还有价值。

## 实验关键数据

这是理论论文，"实验"即各场景下的时间复杂度对比与下界。下面两张表分别给出"上界随 $n$ 的可扩展性"和"本文下界 vs 可匹配算法"。

### 主结果：三种通信场景下复杂度随 n 的可扩展性

| 场景 | 代表方法 | 时间复杂度（关键项） | 随 $n$ 扩展？ |
|------|---------|------|------|
| 通信免费 $\tau_s=\tau_w=0$ | 同步 SGD | $h\big(\tfrac{L\Delta}{\varepsilon}+\tfrac{\sigma^2 L\Delta}{n\varepsilon^2}\big)$ | 统计项随 $n$ 线性变好 ✅ |
| 仅上行 $\tau_w>0,\ \tau_s=0$ | Batch QSGD | $h(1+\tfrac{\sigma^2}{n\varepsilon})\tfrac{L\Delta}{\varepsilon}+\tau_w(\tfrac{d}{n}+1)\tfrac{L\Delta}{\varepsilon}+\sqrt{\tfrac{d\tau_w h\sigma^2}{n\varepsilon}}\tfrac{L\Delta}{\varepsilon}$ | 通信项随 $n$、耦合项随 $\sqrt n$ ✅ |
| 双向 $\tau_s,\tau_w>0$ | Batch QSGD | 多出一项 $\tau_s d\,\tfrac{L\Delta}{\varepsilon}$（式 4） | 下行项**不随 $n$ 改善** ❌ |

### 本文下界与可匹配算法

| 量 | 结论 | 说明 |
|------|------|------|
| 下界 (11)（仅 $\tau_s$） | $\widetilde\Omega\big(\min\{\tau_s d\tfrac{L\Delta}{\varepsilon},\ h\tfrac{L\Delta}{\varepsilon}+h\tfrac{\sigma^2 L\Delta}{\varepsilon^2}\}\big)$ | $d$ 与 $\sigma^2/\varepsilon$ 不可同时随 $n$ 改善 |
| $d$ 可随 $n$ 改善幅度 | 至多 $\log^4(n+1)$ | 远不如线性 $n$ |
| $\sigma^2/\varepsilon$ 可随 $n$ 改善幅度 | 至多 $\log^6(n+1)$ | 远不如 $\sqrt n$ |
| 下界匹配者（$\tau_s\simeq\tau_w$） | Batch Synchronized SGD（不压缩）/ 裸 SGD | 此时压缩无增益 |

### 关键发现
- **同质并不比异质"更容易"**：直觉以为大家函数相同就能协同摊薄下行代价，结果在 $\tau_s\simeq\tau_w$ 时下界与异质情形结论一致——压缩省不下下行带宽。
- **对数屏障极其陡峭**：例如 $n=10{,}000$ 时，要想把 $d$ 扩大 10 倍而保持 $\tau_s d/\log^4(n+1)$ 不变，得把 $n$ 再扩大 $10^3$ 倍（多两个数量级），可扩展性近乎"假性"。
- **$K=\Theta(\log n)$ 是平衡点**：$K$ 太小压不住 $n^{1/K}$，太大又被 Lemma 3.2 的常数反噬，对数恰好是甜点。
- **下界对偏置压缩器同样成立**：无偏压缩器族包含偏置压缩器，故 TopK/RankK 等也存在让其快不起来的 worst-case 压缩器。

## 亮点与洞察
- **"提高门槛"对冲掉 $\min$ 红利**：把"推进 1 步需 1 个非零坐标"改成"需连续 $K$ 个"，本质是逼最幸运的 worker 从"碰一次运气"变成"连中 $K/2$ 次"，从而把 $\min_{i\in[n]}$ 的 $1/n$ 衰减软化成 $1/n^{1/K}$——这是整篇论文最巧的杠杆。
- **$n^{1/K}\to\Theta(1)$ 的取对数技巧**：$K=\Theta(\log n)$ 让指数衰减蒸发成常数，是连接"构造"与"poly-log 屏障"的临门一脚，思路干净到可以当模板。
- **参数 $a$ 解耦"构造表达力"与"几何可控性"**：连乘 $K$ 个 $\Psi$ 必然让光滑常数等爆炸，引入 $a=1+1/K$ 单独把这些常数压回 poly-log，这种"加一个旋钮换可控性"的手法可迁移到其他链式下界构造。
- **把下界归约成随机和的集中**：式 (13) 把抽象的"最短发现时间"翻译成可用集中不等式硬算的几何随机变量之和，是该证明框架最可复用的部分。

## 局限与展望
- 作者承认：下界只精确到对数因子，去掉或改进 $\log$ 的幂次很难，可能需要全新构造。
- worst-case 压缩器只取了随机稀疏化（RandK 类）；作者**猜测**（基于 Safaryan et al. 2022 的不确定性原理）结论对整个无偏压缩器族都成立，但严格证明需更复杂构造——这是悬而未决的口子。
- 下界是悲观的极端构造：实践中 TopK/RankK 的压缩性质远好于 worst-case，配合 EF/EF21-P 在服务器侧用，可能缓解 $\tau_s dL\Delta/\varepsilon$ 这一项；换言之"理论堵死"不等于"工程没救"。
- 额外假设（凸性、二阶光滑性）下这条悲观下界有可能被打破，是值得探的正向方向。

## 相关工作与启发
- **vs Carmon et al. (2020)**：本文的 $F_{T,K,a}$ 直接扩展其经典链式函数 $F_T$，把单坐标门槛升级为 $K$-坐标链并用 $\Gamma$ 项 + 参数 $a$ 控几何；$K=1,a=e$ 时退回原构造。
- **vs Gruntkowska et al. (2024)**：他们在**异质**设定下证下行不随 $n$ 改善（不同 block 分给不同 worker）；本文攻克更难的**同质**设定——所有 worker 同函数，旧的 block 划分论证失效，必须靠 $K$-链。
- **vs Tyurin et al. (2024)（Shadowheart SGD / Batch QSGD）**：本文既把其方法当作下界的匹配算法，又扩展改进其下界结果（Theorem F.1）以并入 $\tau_w>0$。
- **vs Woodworth et al. (2020/2021)**：他们在凸、间歇通信下得到"最优即加速 local 与 minibatch SGD 取其优"的类似结论，但本文聚焦"压缩能否同时改善噪声与通信"的非凸可扩展性，二者相关但不可直接比较。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ $K$-链构造 + $n^{1/K}$ 取对数技巧是真正原创的下界工具，可独立复用。
- 实验充分度: ⭐⭐⭐⭐ 理论论文无实验，但下界与可匹配算法的对照清晰、覆盖三种通信场景。
- 写作质量: ⭐⭐⭐⭐⭐ 从经典构造的失效讲起、层层递进到新构造与归约，动机与技术衔接极顺。
- 价值: ⭐⭐⭐⭐⭐ 给"用更多 worker + 压缩就能无限加速"的乐观叙事划了一条硬下界，重塑了对中心化分布式优化可扩展性的认知。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm](bilevel_optimization_with_lower-level_uniform_convexity_theory_and_algorithm.md)
- [\[ICLR 2026\] LEGACY: A Lightweight Dynamic Gradient Compression Strategy for Distributed Deep Learning](legacy_a_lightweight_dynamic_gradient_compression_strategy_for_distributed_deep_.md)
- [\[ICLR 2026\] Conformal Robustness Control: A New Strategy for Robust Decision](conformal_robustness_control_a_new_strategy_for_robust_decision.md)
- [\[ICLR 2026\] Towards Better Branching Policies: Leveraging the Sequential Nature of Branch-and-Bound Tree](towards_better_branching_policies_leveraging_the_sequential_nature_of_branch-and.md)
- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)

</div>

<!-- RELATED:END -->
