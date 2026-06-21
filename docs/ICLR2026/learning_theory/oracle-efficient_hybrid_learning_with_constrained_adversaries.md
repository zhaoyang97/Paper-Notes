---
title: >-
  [论文解读] Oracle-Efficient Hybrid Online Learning with Constrained Adversaries
description: >-
  [ICLR 2026][学习理论][混合在线学习] 本文研究"特征 i.i.d.、标签对抗"的混合在线学习问题，通过给对抗者加上"标签函数必须取自固定函数类 $R$"的结构性约束，设计出一个只需调用线性优化 oracle 就能运行、且 regret 随复合类 $\ell\circ(H\times R)$ 的 Rademacher 复杂度缩放的算法，首次在该设定下同时逼近统计最优与计算高效。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "在线学习"
  - "混合在线学习"
  - "oracle 高效"
  - "Rademacher 复杂度"
  - "Frank-Wolfe"
  - "随机零和博弈"
---

# Oracle-Efficient Hybrid Online Learning with Constrained Adversaries

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=AKUSUkWj6p](https://openreview.net/forum?id=AKUSUkWj6p)  
**代码**: 待确认  
**领域**: 学习理论 / 在线学习  
**关键词**: 混合在线学习, oracle 高效, Rademacher 复杂度, Frank-Wolfe, 随机零和博弈

## 一句话总结
本文研究"特征 i.i.d.、标签对抗"的混合在线学习问题，通过给对抗者加上"标签函数必须取自固定函数类 $R$"的结构性约束，设计出一个只需调用线性优化 oracle 就能运行、且 regret 随复合类 $\ell\circ(H\times R)$ 的 Rademacher 复杂度缩放的算法，首次在该设定下同时逼近统计最优与计算高效。

## 研究背景与动机
**领域现状**：在线学习按数据生成方式分两个极端——统计设定（数据 i.i.d. 来自固定未知分布）和完全对抗设定（数据由自适应对手挑选）。两者的可学性差距极大：比如学阈值函数，统计设定下用少量样本就能搞定，完全对抗设定下却根本不可学（Littlestone 维度无穷）。混合在线学习（Hybrid Online Learning）是介于两者之间的折中模型：**特征 $x_t$ 仍 i.i.d. 抽自未知分布 $D$，但对应标签由可能恶意的对手生成**。它刻画了现实中"实例服从统计规律、但标签受策略性参与者或最坏情况力量左右"的场景。

**现有痛点**：该领域存在一道"计算-统计鸿沟"。一类算法（Lazaric & Munos 2009；Wu et al. 2023）统计最优，但时间和空间复杂度都随假设类 $H$ 的大小线性增长，对很多自然假设类计算上不可行；另一类算法计算高效，但要么假设学习者完全知道或能无限采样底层特征分布，要么 regret 次优——Wu et al. (2024) 给出首个 oracle 高效算法，却只能拿到 $\tilde{O}(d^{1/2}T^{3/4})$ 的次优 regret。

**核心矛盾**：在混合设定里，**统计最优与计算高效似乎不可兼得**。根本原因在于：对手每轮可以自由挑任意标签函数，使得学习者必须在一个表达力极强、复杂度不受控的标签空间上做最坏情况防御，这迫使统计最优算法去枚举/覆盖整个 $H$，从而丧失计算效率。

**本文目标**：在混合设定下同时做到统计最优与 oracle 高效。为攻克这一难题，作者退一步，研究问题的一个**结构化版本**。

**切入角度**：作者的关键观察是——如果对手不能任意乱标，而是**被约束只能从一个固定但富有表达力的函数类 $R\subseteq[0,1]^X$ 里挑标签函数 $r_t$**，那么学习者面对的复杂度就不再是无穷的标签空间，而是"假设类 $H$ 与对手类 $R$ 交互后"诱导出的复合损失类 $\ell\circ(H\times R)=\{x\mapsto\ell(h(x),r(x))\mid h\in H,r\in R\}$。只要这个复合类的统计复杂度可控，就有望恢复统计速率。

**核心 idea**：用"对对手类 $R$ 加结构约束 + FTRL 配截断熵正则 + Frank-Wolfe 归约到线性 oracle"三件套，把 regret 锚定到 $\mathrm{rad}_T(\ell\circ(H\times R))$ 上，从而在结构化混合设定里同时拿到近最优 regret 和 oracle 高效性。

## 方法详解

### 整体框架
问题设定：每轮 $t$ 学习者先出假设 $h_t$，对手在不知未来特征 $x_t$ 的情况下从 $R$ 中挑标签函数 $r_t$，自然从 $D$ 抽 $x_t$，学习者吃到损失 $\ell(h_t(x_t),r_t(x_t))$ 并观测到 $(x_t,r_t)$。损失 $\ell$ 关于第一个参数凸且 $L$-Lipschitz。目标是相对最优固定假设 $h\in H$ 最小化 regret。

整体算法分两层。**外层**是一个跑在 $H$ 上的 Follow-The-Regularized-Leader（FTRL）：但因为分布 $D$ 未知、且样本是边在线边到达的，作者没法把它写成"固定向量空间 + 固定线性损失"的标准 OCO，于是构造了一种**自适应的截断熵正则**，让正则项只依赖已观测样本的坐标，并证明它在"相关坐标"上强凸，从而拿到 $O(\sqrt{T\log T})$ 的 in-expectation regret。**内层**是把外层每步需要的"熵正则化 $\ell$-ERM oracle"用 **Frank-Wolfe（条件梯度）** 归约到一个更弱、更现实的**线性优化 oracle**，用多项式次调用近似求解。最后用一个新的"混合鞅差序列"尾界（Proposition 1.3）把 in-expectation 弱基准的 regret 转成标准 regret 强基准。整条链路把最终 regret 拆成三块复杂度项之和。

这是一篇纯理论论文，方法是"约束设定 + 算法设计 + 收敛/集中分析"的组合，不存在多阶段数据流水线，故不配框架图。

### 关键设计

**1. 约束对手到函数类 $R$：把无穷标签空间换成可控的复合复杂度**

混合学习难就难在对手每轮能挑任意标签 $r_t$，使学习者要防御的空间复杂度不受控；若 $R$ 不受限（如 $R=[0,1]^X$），复合类 $\ell\circ(H\times R)$ 可以任意丰富，其 Rademacher 复杂度不随 $T$ 衰减，统计速率无从谈起。本文的根本性结构假设是**对手只能从一个固定、学习者已知的函数类 $R$ 里挑 $r_t$**。这把统计复杂度从"无穷标签空间"压缩成复合损失类 $\ell\circ(H\times R)$ 的复杂度。主定理（Theorem 1.1）由此给出高概率 regret 界：

$$\sum_{t=1}^{T}\ell(h_t(x_t),r_t(x_t))-\min_{h\in H}\sum_{t=1}^{T}\ell(h(x_t),r_t(x_t))\le O\!\Big(T\,\mathrm{rad}_T(\ell\circ H\times R)+L\,T\,\mathrm{rad}_T(H)+L\sqrt{T\log(T/\delta)}\Big).$$

其中 $\mathrm{rad}_T(F)=\mathbb{E}_{x_{1:T}\sim D}\,\mathbb{E}_\sigma[\sup_{f\in F}\frac1T\sum_t\sigma_t f(x_t)]$ 是分布相关的 Rademacher 复杂度。直觉是：复合类里每个函数把 $x$ 映到 $(h,r)$ 这对组合诱导的损失 $\ell(h(x),r(x))$，恰好刻画了学习者与对手的"交互复杂度"，regret 就随这族可能损失的统计复杂度缩放。当 $H$ 是 VC 维 $d$ 的二值类、复合类也是 VC 维 $d^*$ 的二值类时，$\mathrm{rad}_T(F)=O(\sqrt{d/T})$，界简化为 $O(\sqrt{Td^*}+L\sqrt{Td}+L\sqrt{T\log(T/\delta)})$。这个界在 $R\subseteq H$ 时与统计学习的下界 $L\,T\,\mathrm{rad}_T(H)+L\sqrt{T\log(1/\delta)}$ 仅差对数因子，即近最优；唯一"代价"是对手类 $R$ 复杂度的依赖。

**2. 截断熵正则的 FTRL：在"永远看不到完整损失向量"的自适应空间里做 OCO**

退一步想：若一开始就有 $m$ 个 i.i.d. 样本 $S$，复合类 $F$ 上的一致收敛（速率 $\mathrm{rad}_m(F)$）会保证经验均值损失 $\frac1m\sum_i\ell(h(s_i),r_t(s_i))$ 逼近真期望，于是问题退化成在 $[0,1/m]^m$ 单纯形子集上的 OCO，FTRL 拿 $\sqrt{T\log m}$ regret 即可。但混合设定里**样本是边走边到的**：第 $t$ 轮只能用前 $t-1$ 个样本构造经验损失 $\tilde\ell_t(v)=\frac{1}{t-1}\sum_{s=1}^{t-1}\ell(v(s),r_t(x_s))$，损失函数的结构随数据集增长动态变化，没法套固定向量空间的标准 OCO。

作者的解法是构造**自适应熵正则** $\psi_t(v)=\frac1\eta\sum_{s=1}^{t-1}v(s)\log(v(s)+1)$，并让第 $t$ 步输出 $F_t(v)=\sum_{s<t}\tilde\ell_s(v)+\psi_t(v)$ 的 $\varepsilon$-近似最小元（对应 $h_t\in\mathrm{conv}(H)$）。关键技巧有两点：其一，用 $\log(h+1)$ 而非 $\log h$，保证参数落在 $[0,1]$ 上良定义，且 $a\log(a+1)$ 在整个 $[0,1]$ 上一致强凸；其二，**正则项不在 $T$ 维全空间上强凸**（因为算法在 $t<T$ 时从未观测到"全 $T$ 样本上的损失向量"），但由于 $\tilde\ell_t,\psi_t$ 只依赖前 $t$ 个坐标、其梯度在 $s>t$ 处为零，作者证明 $\psi_t$ 在**前 $t-1$ 个坐标的 $\ell_1$ 范数下强凸**，这恰好够用。由此 Lemma 2.2 给出经验 regret 界 $\frac{T\log 2}{\eta}+\frac{4\eta L^2\log T}{3}+5L\sqrt{\eta\varepsilon T}$，配 $\eta=\sqrt{T/L^2\log T}$ 得到 in-expectation regret $O(T\,\mathrm{rad}_T(\ell\circ H\times R)+L\sqrt{T\log T})$（Theorem 2.1）。

**3. Frank-Wolfe 归约：把熵正则 ERM oracle 落地成只需线性优化 oracle**

上面的 FTRL 每步要调用一个"熵正则化 $\ell$-ERM oracle"（Definition 2.1：在 $\mathrm{conv}(H)$ 上最小化 $\sum_i w_i\ell(h(x_i),y_i)+\frac1\eta\sum_{s\in S}h(s)\log(h(s)+1)$），但这种 oracle 本身计算上未必现实。作者用 **Frank-Wolfe（条件梯度，projection-free）** 把它归约到一个更弱的**线性优化 oracle**——后者只需给定点集与权重，返回最小化 $\sum_s c_s h(s)$ 的 $h\in H$，对很多自然假设类都可实现。

具体把目标写成在 $K_S=\mathrm{conv}\{(h(s))_{s\in S}\mid h\in H\}$ 上最小化光滑凸函数 $G(z)=\eta\sum_i w_i\ell(z_{x_i},y_i)+\sum_s z_s\log(z_s+1)$。每轮算 $G$ 的梯度分量 $c_s=\eta\sum_{i:x_i=s}w_i\,\partial_h\ell(h_t(s),y_i)+\log(h_t(s)+1)+\frac{h_t(s)}{h_t(s)+1}$，喂给线性 oracle 取极点 $h'_t$，再以步长 $\gamma_t=\frac{2}{t+1}$ 做凸组合更新 $h_{t+1}=(1-\gamma_t)h_t+\gamma_t h'_t$。Lemma 3.1 证明 $O\big(\frac{|S|(\eta W_{\max}\beta+1)}{\epsilon}\big)$ 次迭代即得 $\epsilon$-近似解（$W_{\max}$ 是单特征上权重绝对值之和的上界）。若 $\ell$ 只 Lipschitz 不光滑，用标准 OCO-to-OLO 把 $\ell$ 在当前预测处线性化即可，不损失 regret。整体算法每轮 $O(T^2)$ 时间、全程 $O(T^2)$ 次线性 oracle 调用，瓶颈在构造规模 $|S_t|=\sum_{s=2}^t(s-1)=O(t^2)$ 的三元组数据集。

**4. 混合鞅差尾界：把"弱基准 in-expectation regret"升级成"强基准标准 regret"**

FTRL 给的是相对"期望损失之和"的弱基准 regret，而标准 regret 是相对"实际观测样本上损失之和"。要跨越这道鸿沟，需要一个一致收敛界，难点在于**函数 $r_t$ 是基于前序样本自适应选出来的**——直接对 $H$ 做 Azuma-Hoeffding 加并集界只对有限类成立、且会退化到 $\log|H|$ 的次优界。

作者证明 Proposition 1.3：对 i.i.d. 样本 $x_{1:T}$ 与自适应序列 $r_t$（$r_t$ 只依赖 $x_1,\dots,x_{t-1}$），以概率 $\ge1-\delta$ 对所有 $h\in H$ 有

$$\Big|\frac1T\sum_{t=1}^T\ell(h(x_t),r_t(x_t))-\frac1T\sum_{t=1}^T\mathbb{E}_{x\sim D}[\ell(h(x),r_t(x))]\Big|\le O\!\Big(L\cdot\mathrm{rad}_T(H)+L\sqrt{\tfrac{\log(T/\delta)}{T}}\Big).$$

关键在于 $\ell(h(x_t),r_t(x_t))-\mathbb{E}_D[\ell(h(x),r_t(x))]$ 是鞅差序列（因为 $x_t$ 在 $r_t$ 选定之后才采样），作者用对称化技巧配 Rakhlin et al. (2011) 的分布相关序列 Rademacher 复杂度来证明，**$\ell$ 关于第一参数的 $L$-Lipschitz 性是核心**——它保证界只依赖 $H$ 的复杂度与 $L$，而不依赖自适应序列 $r_t$ 本身的复杂度。最终在 Theorem 1.1 的证明里，把标准 regret 拆成三项 $A+B+C$：$A$ 是鞅差（Azuma 控住，$O(L\sqrt{T\log(1/\delta)})$）、$B$ 是 in-expectation regret（Theorem 2.1）、$C$ 是最优假设的泛化间隙（Proposition 1.3，$L\,T\,\mathrm{rad}_T(H)+O(\sqrt{T\log(T/\delta)})$），各分配 $\delta/3$ 失败概率求和得证。

### 一个完整示例：随机零和博弈求均衡
方法的一个关键推论（Corollary 1.2 / 4.1）是把混合学习器用到博弈求解。考虑鞍点问题 $\min_{h\in H}\max_{r\in R}\mathbb{E}_{x\sim D}[u(h(x),r(x))]$，其中 $u$ 是凸-凹、关于第一参数 $L$-Lipschitz 的收益函数。已知一般零和博弈不存在 oracle 高效求均衡算法（Hazan & Koren 2016），但当**收益函数能分解成"双变量凸-凹 Lipschitz 函数 ∘ 各玩家动作的标量值函数"**时，博弈就有一种低维结构。

作者把 $m$ 个 $D$ 的样本顺序喂给混合学习器：每步取 $r_t$ 为对当前 $h_t$ 的最佳响应 $r_t=\arg\max_{r\in R}\sum_i u(h_t(x_i),r(x_i))$（用同一批样本算，靠一致收敛逼近真最佳响应），配合标准 minimax 分析，过程收敛到 $\epsilon(m)$-近似鞍点，$\epsilon(m)=\mathrm{rad}_m(F)+O(L\sqrt{\log m/m})$，且 $\mathrm{rad}_m(F)\to0$ 正是收益矩阵一致收敛所必需。这样即便玩家动作集本身（潜在）高维，只要收益函数有上述低维分解，就能 oracle 高效地求近似均衡——这正是把混合在线学习框架变成实际算法工具的范例。

## 实验关键数据
本文是纯理论工作，**不含实证实验**，贡献以定理形式给出。下表汇总核心理论结果，便于对比定位。

### 主要结果（理论界）

| 结果 | 内容 | 计算代价 |
|------|------|----------|
| Theorem 1.1（主定理） | 高概率 regret $O(T\,\mathrm{rad}_T(\ell\circ H\times R)+L\,T\,\mathrm{rad}_T(H)+L\sqrt{T\log(T/\delta)})$ | 每轮 $O(T^2)$ 时间，全程 $O(T^2)$ 次线性 oracle 调用 |
| Theorem 2.1（弱基准） | in-expectation regret $T\,\mathrm{rad}_T(\ell\circ H\times R)+O(L\sqrt{T\log T})$ | 每轮 $O(T^2)$ 时间，$T$ 次熵正则 ERM oracle |
| Lemma 3.1（FW 归约） | 熵正则 ERM oracle 的 $\epsilon$-近似解 | $O(|S|(\eta W_{\max}\beta+1)/\epsilon)$ 次线性 oracle 调用 |
| Corollary 1.2（博弈） | $\epsilon(m)=\mathrm{rad}_m(F)+O(L\sqrt{\log m/m})$ 近似鞍点 | $m$ 与 $H,R$ 复杂度的多项式时间 |

### 与已有工作的速率对比

| 方法 | regret | 是否 oracle 高效 | 是否统计最优 |
|------|--------|------------------|---------------|
| Lazaric & Munos (2009) | $O(\sqrt{dT\log T})$（二值类，绝对损失） | 否（需覆盖 $H$） | 近最优 |
| Wu et al. (2023) | 近最优期望 regret（实值类、一般凸损失） | 否（构造随机覆盖，可能不可行） | 是 |
| Wu et al. (2024) | $\tilde{O}(d^{1/2}T^{3/4})$ | 是（ERM oracle + FPL） | 否（次优） |
| **本文** | $O(\sqrt{Td^*}+L\sqrt{Td}+\dots)$（约束对手下） | **是（线性 oracle）** | **近最优（仅差 $R$ 依赖与 log 因子）** |

### 关键发现
- **结构假设的边界**：若 $R$ 完全不受限，复合类 Rademacher 复杂度不随 $T$ 衰减，本文界失效（此时反而是 Wu et al. 2024 给出 sublinear regret）；而当 $R\subseteq H$ 时，本文界恰好匹配统计学习下界（差 log 因子），定量说明"约束对手"到底买回了多少统计速率。
- **下界不可绕过**：混合学习至少和对应统计学习一样难，故 $L\,T\,\mathrm{rad}_T(H)+L\sqrt{T\log(1/\delta)}$ 是无法回避的下界——即便没有对抗结构，对 $H$ 复杂度的依赖也必然存在。
- **Lipschitz 性是关键杠杆**：Proposition 1.3 的界只依赖 $H$ 的复杂度与 $L$、而非自适应序列 $r_t$ 的复杂度，全靠 $\ell$ 对第一参数的 $L$-Lipschitz 性，这是把弱基准升级到强基准的核心。

## 亮点与洞察
- **"约束对手"这一刀切得巧**：不是去硬刚无穷对抗空间，而是把对手关进固定函数类 $R$，让 regret 自然锚定到复合类 $\ell\circ(H\times R)$ 的 Rademacher 复杂度——一个统计学习里成熟、可控的量。这种"用结构假设换可控复杂度"的思路可迁移到其他计算-统计鸿沟问题。
- **截断熵正则化的构造很精致**：在"永远看不到完整 $T$ 维损失向量"的自适应在线场景里，放弃在全空间强凸、转而证明"在已观测的前 $t-1$ 个坐标上强凸即够"，是对标准 FTRL 分析的实质性扩展；$\log(h+1)$ 的小改动同时解决了定义域与一致强凸两个问题。
- **Frank-Wolfe 把强 oracle 降级成弱 oracle**：projection-free 的条件梯度让算法只依赖线性优化 oracle（很多假设类可实现），是把理论算法变成"现实可调用"的关键工程化一步。
- **博弈推论是漂亮的副产品**：把"收益函数可分解 → 低维结构 → 最佳响应即混合学习"这条链路打通，在一般 oracle 高效求均衡被证不可能的前提下，刻画出一类可解的结构化零和博弈。

## 局限与展望
- **依赖对手类 $R$ 的复杂度**：界里 $\mathrm{rad}_T(\ell\circ H\times R)$ 直接吃 $R$ 的复杂度，当 $R$ 富表达力时这一项可能很大；$R$ 完全不受限时方法直接失效，适用范围被结构假设框死。
- **计算代价偏高**：每轮 $O(T^2)$ 时间、全程 $O(T^2)$ 次 oracle 调用，瓶颈在构造 $O(t^2)$ 规模的三元组数据集，对长 horizon $T$ 不够轻量，作者未讨论如何降到近线性。
- **纯理论、无实证**：所有结论是 worst-case 上界，未给任何数值实验验证常数因子或实际假设类上的表现，落地效果未知。
- **刻画工具留白**：作者自己指出，能否用 Hu & Peale (2023) 的 mutual VC dimension 之类的度量来刻画混合学习的样本复杂度，仍是开放问题；与 comparative learning 的区别（本文对手每轮可换 $r_t$）也提示了更细粒度刻画的空间。

## 相关工作与启发
- **vs Wu et al. (2023)**：他们用构造假设类随机覆盖的方式拿到实值类、一般凸损失的统计近最优 regret，但覆盖对很多自然假设类计算上不可行；本文牺牲一点（依赖 $R$ 结构）换来 oracle 高效，定位是"在结构化子问题上同时要统计+计算"。
- **vs Wu et al. (2024)**：他们用 relaxation-based FPL 给出首个 oracle 高效算法，但 regret $\tilde{O}(d^{1/2}T^{3/4})$ 统计次优；本文在约束对手下把速率拉回 $\sqrt{T}$ 量级的近最优，代价是对手不能任意乱标。
- **vs Block et al. (2024)（smoothed online learning）**：他们证明 ERM 在未知底分布的 smoothed 模型里也能 oracle 高效，但其结论对应混合学习的 realizable 情形（所有标签来自单一 $h\in H$）；本文设定严格更一般——对手每轮可换 $r_t\in R$，即便 $R=H$，揭示的标签也不必与任何单个 $h$ 全程一致。
- **vs Hu & Peale (2023)（comparative learning）**：两者都把标签函数限制在已知函数类里，但 comparative learning 的标签函数是固定的，本文对手每轮可自适应换函数，是更强的对抗模型。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在结构化混合设定下同时逼近统计最优与 oracle 高效，截断熵正则与混合鞅差尾界都是新工具。
- 实验充分度: ⭐⭐⭐ 纯理论无实证，结论以定理形式给出，常数与实际表现未验证（理论论文按其标准评）。
- 写作质量: ⭐⭐⭐⭐ 技术综述与三项分解的证明思路清晰，定理与已有工作对比定位明确。
- 价值: ⭐⭐⭐⭐ 推进了在线学习计算-统计鸿沟的理解，并衍生出结构化零和博弈的 oracle 高效求解，工具有独立价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Online Decision-Focused Learning](online_decision-focused_learning.md)
- [\[ICML 2025\] Improved and Oracle-Efficient Online $\ell_1$-Multicalibration](../../ICML2025/learning_theory/improved_and_oracle-efficient_online_ell_1-multicalibration.md)
- [\[ICLR 2026\] Online Learning and Equilibrium Computation with Ranking Feedback](online_learning_and_equilibrium_computation_with_ranking_feedback.md)
- [\[ICLR 2026\] Sublinear Spectral Clustering Oracle with Little Memory](sublinear_spectral_clustering_oracle_with_little_memory.md)
- [\[ICLR 2026\] Pseudo-Non-Linear Data Augmentation: A Constrained Energy Minimization Viewpoint](pseudo-non-linear_data_augmentation_a_constrained_energy_minimization_viewpoint.md)

</div>

<!-- RELATED:END -->
