---
title: >-
  [论文解读] A Tale of Two Geometries: Adaptive Optimizers and Non-Euclidean Descent
description: >-
  [ICLR 2026][优化/理论][自适应优化器] 这篇论文用"两种几何 / 两种平滑度"统一刻画了 Adam/Shampoo 这类自适应优化器与 SignGD/Muon 这类归一化最速下降（NSD）的关系：两者都在利用损失函数的非欧几何，但自适应优化器依赖一个更强的「自适应平滑度」$\Lambda_{\mathcal H}(f)$，而 NSD 依赖标准平滑度 $L_{\|\cdot\|_{\mathcal H}}(f)$；论文把自适应平滑度的分析从凸推广到非凸，并证明这个更强的假设确实能换来"标准平滑度下拿不到"的好处——Nesterov 加速率 $\tilde O(T^{-2})$ 与维度无关的随机收敛率。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "自适应优化器"
  - "非欧几何"
  - "自适应平滑度"
  - "收敛分析"
  - "Nesterov加速"
---

# A Tale of Two Geometries: Adaptive Optimizers and Non-Euclidean Descent

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=iaoAKDRAJQ](https://openreview.net/forum?id=iaoAKDRAJQ)  
**代码**: 无（纯理论论文）  
**领域**: 优化理论  
**关键词**: 自适应优化器, 非欧几何, 自适应平滑度, 收敛分析, Nesterov加速

## 一句话总结
这篇论文用"两种几何 / 两种平滑度"统一刻画了 Adam/Shampoo 这类自适应优化器与 SignGD/Muon 这类归一化最速下降（NSD）的关系：两者都在利用损失函数的非欧几何，但自适应优化器依赖一个更强的「自适应平滑度」$\Lambda_{\mathcal H}(f)$，而 NSD 依赖标准平滑度 $L_{\|\cdot\|_{\mathcal H}}(f)$；论文把自适应平滑度的分析从凸推广到非凸，并证明这个更强的假设确实能换来"标准平滑度下拿不到"的好处——Nesterov 加速率 $\tilde O(T^{-2})$ 与维度无关的随机收敛率。

## 研究背景与动机
**领域现状**：训练大模型长期由 Adam 这类自适应优化器主导，但近期 Muon、Lion 等结构更简单的「归一化最速下降」（Normalized Steepest Descent, NSD）方法表现出惊人的竞争力。学界逐渐形成一个共识：这两类优化器之所以强，关键都在于它们能利用损失曲面的**非欧几何**（例如 $\ell_\infty$ 几何、矩阵谱范数几何），而不是默认的欧氏 $\ell_2$ 几何。

**现有痛点**：Bernstein & Newhouse (2024) 给出过一个惊人的连接——把指数滑动平均（EMA）关掉后，某些自适应优化器会**精确退化**成对应的 NSD：没有 EMA 的 Adam 等于 $\ell_\infty$ 范数下的 NSD（即 SignGD），没有 EMA 的 Shampoo 等于谱范数下的 NSD（即 Muon）。两个家族看似只隔着一个 EMA，但除了这种"算法形式上的退化"之外，**没有任何形式化结果系统刻画两者的关系**——它们利用非欧几何的"方式"到底是不是一回事？

**核心矛盾**：作者从理论视角切入，发现即便在**同一个几何**下，也会冒出两种本质不同的平滑度假设。一种是一般范数下的**标准平滑度**（Definition 2.3），它支配 NSD 的收敛率；另一种是 Xie et al. (2025b) 提出的**自适应平滑度**（Definition 2.4），此前只被证明在凸情形支配自适应优化器的收敛率。问题在于：这两种平滑度到底差在哪、谁强谁弱、强的那个是不是白强。

**本文目标**：拆成两个问题。Q1：Adam/Shampoo 与对应的 NSD（Lion/Muon）是否以"相同的方式"利用非欧几何？Q2：自适应方法依赖的更强平滑度假设，是否真能带来优化上的好处？

**切入角度**：用平滑度假设作为比较两类算法的"通用尺子"。关键观察是——自适应平滑度 $\Lambda_{\mathcal H}(f)$ 永远不小于同一几何下的标准平滑度 $L_{\|\cdot\|_{\mathcal H}}(f)$（两者最多差一个维度因子 $d$）。既然是更强的条件，就要回答：这份"强"是纯粹的技术负担，还是能兑换成更好的收敛率？

**核心 idea**：用「自适应平滑度 vs 标准平滑度」这一对几何，把自适应优化器与 NSD 的差异讲透；先把自适应平滑度的分析从凸推广到非凸（答 Q1：两者确实靠不同平滑度），再证明这个更强假设能解锁标准平滑度下不可能的加速率与维度无关率（答 Q2：强假设有真实回报），并把同样的"标准 vs 自适应"二分平移到随机噪声上。

## 方法详解

### 整体框架
论文是一篇**纯优化理论**工作，没有"模型/网络"，它要搭的是一套能同时容纳 Adam、AdaGrad、单边 Shampoo 等的**统一分析框架**，然后在这个框架里比较两套几何度量。整条逻辑链是：先用一个元算法（Algorithm 1）把一大类自适应优化器统一写出来 → 定义刻画它们收敛行为的「自适应平滑度」并和 NSD 的「标准平滑度」对照 → 在非凸设定下证明自适应平滑度确实支配自适应优化器的收敛率（答 Q1）→ 再分别从"加速"和"噪声"两个角度证明这个更强假设能换来更好的率（答 Q2）。

统一的元算法把更新写成预条件子（preconditioner）形式：每步用历史梯度的二阶矩 $M_t$（累积 / EMA / 加权三种聚合方式之一）解一个矩阵优化得到预条件子 $V_t$，再做预条件梯度步
$$V_t \leftarrow \arg\min_{H\in\mathcal H}\ \langle M_t+\epsilon I_d,\, H^{-1}\rangle + \mathrm{Tr}(H),\qquad x_{t+1}\leftarrow x_t-\eta\, V_t^{-1} g_t.$$
这里 $\mathcal H\subseteq \mathbb S^d_+$ 是一个**凸锥（预条件子集合）**：取 $\mathcal H=$ 所有对角 PSD 矩阵就恢复出 Adam/AdaGrad，取 $\mathcal H=\{c I_d\}$ 恢复 AdaGrad-Norm，取 $\mathcal H=\mathbb S^d_+$ 恢复全矩阵 AdaGrad，取 $\mathcal H=\mathbb S^{d_L}_+\otimes I_{d_R}$ 恢复单边 Shampoo/ASGO。换句话说，"选哪个几何"被编码成"选哪个预条件子集合 $\mathcal H$"，整套分析因此一次覆盖一大家族优化器。

### 关键设计

**1. 两种几何即两种平滑度：标准平滑度支配 NSD，自适应平滑度支配自适应优化器**

这是全文的"棱镜"。在固定的非欧几何 $\|\cdot\|_{\mathcal H}$ 下，作者区分两把尺子。标准平滑度 $L_{\|\cdot\|}(f)$ 是经典定义——使 $\|\nabla f(x)-\nabla f(y)\|_*\le L\|x-y\|$ 成立的最小 $L$，它支配 NSD 的收敛。自适应平滑度 $\Lambda_{\mathcal H}(f)$ 则是"在 $\mathcal H$ 诱导的所有范数里取**最小**的那个平滑度"：
$$\Lambda_{\mathcal H}(f):=\min_{H\in\mathcal H,\ \mathrm{Tr}(H)\le 1} L_{\|\cdot\|_H}(f)=\min_{H\in\mathcal H,\ \forall x:\,-H\preceq\nabla^2 f(x)\preceq H}\mathrm{Tr}(H).$$
对 Adam（$\mathcal H=$ 对角 PSD）它就是 $\Lambda_{\mathrm{diag}}(f)=\min_{-H\preceq\nabla^2 f\preceq H,\,H\text{ 对角}}\mathrm{Tr}(H)$。为什么对角自适应平滑度会"长成 $\ell_\infty$ 几何"？因为良结构预条件子集合有一条对偶性质（Lemma 2.2）：原范数取上确界、对偶范数取下确界，恰好

$$\sup_{H\text{ 对角},\,\mathrm{Tr}(H)\le 1}\|\cdot\|_H=\|\cdot\|_\infty,\qquad \inf_{H\text{ 对角},\,\mathrm{Tr}(H)\le 1}\|\cdot\|_{H,*}=\|\cdot\|_1.$$

于是把 NSD 在每个 $\|\cdot\|_H$ 下的收敛率 $O(\sqrt{\Delta_0 L_{\|\cdot\|_H}(f)/T})$ 对 $H$ 取最优，右端就收敛到 $O(\sqrt{\Delta_0\Lambda_{\mathrm{diag}}(f)/T})$——正是 Adam 的率。这解释了 Adam 的"自适应"本质：它能自动找到并贴合给定损失最合适的对角矩阵诱导范数，**无需事先知道最优的 $H$**。关键的对照结论是 Proposition 2.5：$L_{\|\cdot\|_{\mathcal H}}(f)\le \Lambda_{\mathcal H}(f)\le d\cdot L_{\|\cdot\|_{\mathcal H}}(f)$——自适应平滑度永远不小于标准平滑度，作为假设它更强，但两者最多差维度因子 $d$。这一条直接回答 Q1：自适应优化器和 NSD 即便在同一几何下，也是靠**两种不同的平滑度**在收敛。

**2. 良结构预条件子集合 + 一个新矩阵不等式：把自适应平滑度的分析从凸打通到非凸**

要让结论成立，需要一个能容纳非对角预条件子的统一分析。论文沿用「良结构预条件子集合」的定义（Definition 2.1）：$\mathcal H=\mathbb S^d_+\cap K$，其中 $K$ 是含单位阵、对数乘/加法/矩阵乘封闭的子代数。此前对单边 Shampoo 等非对角预条件子的收敛分析基本只在**凸**或**只含对角矩阵**（可交换）的情形成立——因为对角情形可逐分量拆开、标量望远镜求和即可收尾。难点在于一般 $\mathcal H$ 下矩阵**不可交换**，二阶项无法这样拆。论文的核心技术贡献是 Lemma 3.3：对任意良结构集合，二阶项之和被

$$\sum_{t=0}^{T-1}\|V_t^{-1}g_t\|_H^2\le \mathrm{Tr}(H)\,\|S_T\|_{\mathrm{op}},\qquad S_T=\sum_{t=0}^{T-1}V_t^{-1}(V_t^2-\beta V_{t-1}^2)V_t^{-1}$$

控制住，且 $\|S_T\|_{\mathrm{op}}$ 的上界在不可交换时只比对角情形多出一个 $\log d$ 因子（可交换时这个因子消失）。它依托一条新的矩阵不等式（Lemma C.1，把两个正定矩阵之差与它们对数之差联系起来），本身具有独立价值。靠它，作者得到非凸收敛率（Theorem 3.2）：取最优学习率后

$$\frac1T\sum_{t=0}^{T-1}\|\nabla f(x_t)\|_{H,*}\le \tilde O\!\Big(\log d\cdot \sqrt{\tfrac{\Delta_0\,\Lambda_{\mathcal H}(f)}{T}}\Big),$$

匹配最优的 $\tilde O(T^{-1/4})$ 量级（对 Adam 这是关于梯度 $\ell_1$ 范数的保证）。这正式确认：**非凸**情形下自适应优化器同样由 $\Lambda_{\mathcal H}(f)$ 支配，而 NSD 由 $L_{\|\cdot\|_{\mathcal H}}(f)$ 支配，两者用的几何度量不同。

**3. Nesterov 加速：更强的自适应平滑度换来标准平滑度拿不到的 $\tilde O(T^{-2})$ 率**

既然 $\Lambda_{\mathcal H}$ 更强，Q2 问它强得值不值。论文给出肯定回答的第一条证据：在凸设定下给自适应优化器装上 Nesterov 动量（Algorithm 2，把每步加速解读为在修正损失 $f^{\alpha,\bar x}(x)=\alpha^{-2}f(\alpha x+(1-\alpha)\bar x)$ 上做一步标准梯度），可得加速率（Theorem 4.3）

$$\mathbb E[f(\bar x_T)-f(x^*)]=\tilde O\!\Big(\frac{\Lambda_{\mathcal H}(f)D^2\log^2 d}{T^2}+\frac{\sigma_{\mathcal H}D\log d}{\sqrt T}\Big),$$

确定性部分是 $\tilde O(\Lambda_{\mathcal H}(f)D^2/T^2)$ 的加速率。对照之下，Guzmán & Nemirovski (2015) 证明：在标准 $\ell_\infty$ 平滑度下，**任何**一阶优化器都不可能优于 $\Omega(L_{\ell_\infty}(f)/(T\log T))$。这构成一条干净的**分离**：自适应平滑度让非欧几何下的加速成为可能，而标准平滑度做不到。也就是说，自适应平滑度对加速是**必要**的，更弱的非欧平滑度替代不了它——这份"强"换来了实打实的算法收益。（论文还给出一个投影变体 Algorithm 8，去掉对未知量 $D=\max_t\|x_t-x^*\|_{\mathcal H}$ 的依赖，保持同样的率。）

**4. 自适应方差：把"标准 vs 自适应"平移到噪声，解锁维度无关的随机收敛率**

第二条证据来自随机设定。作者发现平滑度上的二分在**梯度噪声**上有完全平行的版本。标准梯度方差 $\sigma_{\|\cdot\|}$ 在固定范数下度量噪声；自适应梯度方差（Definition 4.1）则要求对 $\mathcal H$ 里每个预条件子诱导的几何一致控制：
$$\sigma_{\mathcal H}^2=\min_{H\in\mathcal H,\,\mathrm{Tr}(H)\le 1}\ \sup_{t,x}\ \mathbb E\big[\|\nabla f_t(x)-\mathbb E\nabla f_t(x)\|_{H^{-1}}^2\big].$$
和平滑度一样，自适应方差永远不小于标准方差，但它是比"有界协方差"更弱的假设。背后的机制是同一个直觉：**在对偶空间里平均并不能有效缩小范数**。对 $n$ 个零均值向量，$\mathbb E\|\frac1n\sum x_i\|_2^2\le\sigma^2/n$，但 $\mathbb E\|\frac1n\sum x_i\|_1^2$ 可能差一个 $d$ 因子，这导致 NSD 的随机收敛率被一个维度失真因子 $\rho=\sup_x \|x\|_{\mathcal H,*}/\|x\|_2$ 拖累（对角 $H$ 下 $\rho=\Theta(\sqrt d)$，$T\ll d$ 时界变空洞）。论文证明（Theorem 4.5）：换成自适应方差假设，带动量的 NSD 能拿到**维度无关**的率 $O((\Delta_0 L_{\|\cdot\|_{\mathcal H}}(f))^{1/4}\sqrt{\sigma_{\mathcal H}}/T^{1/4})$，只依赖标准平滑度与 $\sigma_{\mathcal H}$，彻底甩掉 $\rho$。而 Theorem 4.6/4.7 给出**下界**：在标准方差假设（$\|\cdot\|=\|\cdot\|_\infty$）下，这种 $d$ 依赖在最坏情况无法避免，达到精度 $\epsilon$ 需要 $T=\Omega(\epsilon^{-2}(dL\Delta_0\sigma^2)^{1/2})$，对维度是 $\Omega(\sqrt d)$。一上一下夹出一个本质 gap：自适应方差这个更强假设，同样换来了标准方差下不可能的维度无关率。

## 实验关键数据

> 本文是纯理论论文，**没有数值实验**；下面把核心理论结论整理成对照表，作为"关键数据"。$\tilde O/\Omega$ 隐去对数因子，$d$ 为参数维度，$T$ 为迭代步数，$D=\max_t\|x_t-x^*\|_{\mathcal H}$。

### 主结果：两种平滑度下的收敛率对照

| 设定 | NSD（标准平滑度 $L_{\|\cdot\|_{\mathcal H}}$） | 自适应优化器（自适应平滑度 $\Lambda_{\mathcal H}$） |
|------|------|------|
| 非凸·确定性 | $O\big(\sqrt{\Delta_0 L_{\|\cdot\|_{\mathcal H}}(f)/T}\big)$ | $\tilde O\big(\log d\,\sqrt{\Delta_0\Lambda_{\mathcal H}(f)/T}\big)$（Thm 3.2，匹配最优 $\tilde O(T^{-1/4})$） |
| 凸 + Nesterov 加速 | $\Omega\big(L_{\ell_\infty}(f)/(T\log T)\big)$ 下界，**无法加速** | $\tilde O\big(\Lambda_{\mathcal H}(f)D^2/T^2\big)$ 加速（Thm 4.3） |
| 假设强弱 | 较弱 | $L_{\|\cdot\|_{\mathcal H}}\le \Lambda_{\mathcal H}\le d\cdot L_{\|\cdot\|_{\mathcal H}}$（Prop 2.5），更强 |

### 噪声分析：标准方差 vs 自适应方差（NSD with momentum）

| 噪声假设 | NSD 随机收敛率 | 维度依赖 |
|------|------|------|
| 标准方差 $\sigma_{\|\cdot\|_*}$ | 含范数失真因子 $\psi(\|\cdot\|_*,\|\cdot\|_2)$，随 $d$ 增长 | $\Omega(\sqrt d)$ 下界不可避免（Thm 4.6/4.7） |
| 自适应方差 $\sigma_{\mathcal H}$ | $O\big((\Delta_0 L_{\|\cdot\|_{\mathcal H}}(f))^{1/4}\sqrt{\sigma_{\mathcal H}}/T^{1/4}\big)$ | **维度无关**（Thm 4.5） |

### 关键发现
- **自适应平滑度是收敛行为的"真支配量"**：非凸情形下自适应优化器的率由 $\Lambda_{\mathcal H}(f)$ 而非 $\|\nabla f\|_2$ 决定，与凸情形的已有结论接上，把整条比较线补全到非凸。
- **强假设有真实回报**：加速率 $\tilde O(T^{-2})$ 与维度无关率，都被证明在更弱的标准平滑度/标准方差下**不可达**——"自适应假设更强"不是技术负担而是算法优势的来源。
- **不可交换性的代价是一个 $\log d$**：从对角预条件子推广到一般良结构集合，唯一多出的就是 $\log d$ 因子（Lemma 3.3），可交换时消失；这量化了"对角 Adam"与"矩阵型 Shampoo"分析难度的本质差距。

## 亮点与洞察
- **"一个 EMA 之隔"被升级成"一种平滑度之隔"**：把 Bernstein & Newhouse 那个算法形式上的退化连接，提炼成 $\Lambda_{\mathcal H}$ vs $L_{\|\cdot\|_{\mathcal H}}$ 的几何/假设层面的本质区分，给两大优化器家族提供了统一坐标系，非常解释性。
- **对偶性是全篇的支点**：良结构预条件子集合的"原范数取 sup、对偶范数取 inf"对偶（Lemma 2.2），既解释了 Adam 为何呈现 $\ell_\infty$ 几何，又是把 NSD 逐 $H$ 率最优化成自适应率的关键一跃——一个干净的数学结构撑起两个家族的对照。
- **可迁移的技术工具**：Lemma 3.3 与其底层矩阵对数不等式（Lemma C.1）是处理"不可交换预条件子二阶项"的通用利器，凡是分析 Shampoo/Muon 类矩阵优化器、想从凸推非凸的工作都可能复用。
- **"平均在对偶空间失效"是统一直觉**：加速和噪声两个看似不同的结论，被归因于同一现象——非欧几何下平均不能有效缩范数。一个直觉串起两条独立的分离结果，是这篇论文最"啊哈"的地方。

## 局限与展望
- **纯理论、零实验**：所有结论都是收敛率与上下界，没有任何数值或真实训练验证 $\Lambda_{\mathcal H}$、$\sigma_{\mathcal H}$ 等量在实际损失上的大小与可估计性；这些量本身往往难以计算，实践指导意义需要后续实证补足。
- **加速结果限于凸**：$\tilde O(T^{-2})$ 的加速只在凸设定证明，深度学习的非凸损失上能否（以何种形式）保留加速，论文未涉及。
- **覆盖范围仍受"良结构/子代数"约束**：分析依赖 $\mathcal H=\mathbb S^d_+\cap K$ 的子代数结构，对不满足该结构的更一般预条件方案不直接适用。
- **常数与对数因子**：$\tilde O$ 隐去的 $\log d$ 等因子在高维实际中未必可忽略，最坏情况下界也不代表典型情况；如何刻画"典型损失"上的真实差距是有意义的开放问题。

## 相关工作与启发
- **vs Bernstein & Newhouse (2024)**：他们指出关掉 EMA 后 Adam→SignGD、Shampoo→Muon 的算法等价；本文承接这一连接，但更进一步从"平滑度假设"层面证明二者收敛机制不同，把"形式等价"与"分析差异"区分开。
- **vs Xie et al. (2025b)**：自适应平滑度（原称 $\mathcal H$-smoothness）由其提出并在**凸**情形证明支配自适应优化器；本文的主要推进是把这一刻画扩展到**非凸**，并依赖新矩阵不等式覆盖一般良结构（非对角）预条件子。
- **vs Kovalev (2025a) / Kovalev & Borodich (2025)**：同样研究自适应噪声/加速框架，但其分析对一般 $\mathcal H$ 施加了对损失与噪声的限制性假设；本文借 Lemma 3.3 绕开这些条件，且在维度无关率上用标准平滑度（而非自适应平滑度）给出**严格更优**的界。
- **vs Guzmán & Nemirovski (2015)**：提供了标准 $\ell_\infty$ 平滑度下任何一阶法不超过 $\Omega(1/(T\log T))$ 的下界，本文正是用它作为对照，凸显自适应平滑度对加速的必要性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用"两种几何/两种平滑度"统一两大优化器家族，并证明强假设可兑换加速与维度无关率，视角与结论都很新。
- 实验充分度: ⭐⭐⭐ 纯理论论文，定理与上下界自洽完整，但完全没有数值/实证支撑，关键量的实际可估计性未验证。
- 写作质量: ⭐⭐⭐⭐ 问题驱动（Q1/Q2）、对照清晰、直觉（对偶空间平均失效）贯穿全篇，理论论文里属易读的一档。
- 价值: ⭐⭐⭐⭐⭐ 为 Adam/Shampoo vs Lion/Muon 之争提供了统一理论框架与可复用的矩阵分析工具，对优化理论社区有较强的奠基意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A Convergence Analysis of Adaptive Optimizers under Floating-Point Quantization](a_convergence_analysis_of_adaptive_optimizers_under_floating-point_quantization.md)
- [\[ICLR 2026\] DES-LOC: Desynced Low Communication Adaptive Optimizers for Foundation Models](des-loc_desynced_low_communication_adaptive_optimizers_for_foundation_models.md)
- [\[ICLR 2026\] MT-DAO: Multi-Timescale Distributed Adaptive Optimizers with Local Updates](mt-dao_multi-timescale_distributed_adaptive_optimizers_with_local_updates.md)
- [\[ICLR 2026\] Towards Dynamic Interleaving Optimizers](towards_dynamic_interleaving_optimizers.md)
- [\[ICLR 2026\] Directional Convergence, Benign Overfitting of Gradient Descent in leaky ReLU two-layer Neural Networks](directional_convergence_benign_overfitting_of_gradient_descent_in_leaky_relu_two.md)

</div>

<!-- RELATED:END -->
