---
title: >-
  [论文解读] On the Convergence Direction of Gradient Descent
description: >-
  [ICLR 2026][优化/理论][梯度下降] 本文证明：当梯度下降（GD）收敛到一个局部强凸极小点时，它的轨迹并不是朝任意方向逼近，而是要么**对齐到一个固定方向**（小学习率），要么**沿一条直线来回振荡收敛**（大学习率），分界线恰好是 $\eta = 2/(\lambda_1+\lambda_n)$；这个离散版"梯度猜想"还顺带为 Edge of Stability 里 sharpness 的振荡现象给出了一个解释。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "梯度下降"
  - "收敛方向"
  - "Edge of Stability"
  - "sharpness"
  - "梯度猜想"
---

# On the Convergence Direction of Gradient Descent

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=3U6wH7uAPZ](https://openreview.net/forum?id=3U6wH7uAPZ)  
**代码**: 待确认  
**领域**: 优化理论 / 梯度下降动力学  
**关键词**: 梯度下降, 收敛方向, Edge of Stability, sharpness, 梯度猜想

## 一句话总结
本文证明：当梯度下降（GD）收敛到一个局部强凸极小点时，它的轨迹并不是朝任意方向逼近，而是要么**对齐到一个固定方向**（小学习率），要么**沿一条直线来回振荡收敛**（大学习率），分界线恰好是 $\eta = 2/(\lambda_1+\lambda_n)$；这个离散版"梯度猜想"还顺带为 Edge of Stability 里 sharpness 的振荡现象给出了一个解释。

## 研究背景与动机

**领域现状**：GD 是深度学习里最基础、被研究得最透的优化算法。经典分析（凸 + $L$-smooth）告诉我们，只要学习率满足 $0<\eta<2/L$，GD 就稳定收敛并有标准的收敛速率。围绕极小点的二阶近似，更精细的结论是：收敛通常要求 $0<\eta<2/\lambda_n$，其中 $\lambda_n$ 是极小点处 Hessian 的最大特征值。

**现有痛点**：这些经典框架只回答了"会不会收敛、收敛多快"，却几乎没有刻画**轨迹长什么样**——尤其是 GD 临近极小点时是沿哪个方向逼近的。更尴尬的是，实验里 GD 经常在 $2/\eta$ 比当前 sharpness 还小（即违反经典稳定性条件）的情况下照样往前走，这就是 Cohen 等人观察到的 Edge of Stability（EoS）现象：loss 在短时间尺度上非单调抖动，sharpness 在 $2/\eta$ 附近徘徊。经典理论对这种"不稳定却仍在前进"的动力学无能为力。

**核心矛盾**：连续时间的梯度流早有漂亮答案——René Thom 提出、Parusinski 等人（2000）证明的**梯度猜想**断言：解析函数的梯度流轨迹在临界点附近，其归一化割线 $\frac{x(t)-x_0}{\|x(t)-x_0\|}$ 一定收敛（方向存在）。但离散的 GD 多了一个学习率 $\eta$，步长不再无穷小，方向是否还存在、会不会因为 $\eta$ 大而出现连续时间没有的新行为，一直没有答案。

**本文目标**：把连续时间的梯度猜想"翻译"到离散 GD 上，并刻画方向究竟由什么决定。

**切入角度**：作者从一个最简单的二次例子出发观察。取 $f(x,y)=x^2/2+2y^2$，GD 解析解为 $x_k=(1-\eta)^k x_0,\ y_k=(1-4\eta)^k y_0$。看归一化向量 $v_k=\frac{(x_k,y_k)}{\|(x_k,y_k)\|}$：当 $0<\eta<2/5$ 时 $v_k\to(\mathrm{sign}(x_0),0)$（锁定在小特征值方向 $x$ 轴）；当 $2/5<\eta<1/2$ 时 $(-1)^k v_k\to(0,\mathrm{sign}(y_0))$（在大特征值方向 $y$ 轴上交替振荡）。加上高阶扰动项后规律依旧。这个"相变"暗示方向由学习率与特征值的相对关系决定。

**核心 idea**：用"特征值 + 学习率"刻画 GD 的渐近方向——**小学习率走最平缓方向（最小特征值），大学习率沿最陡方向（最大特征值）来回振荡**，分界是 $2/(\lambda_1+\lambda_n)$。

## 方法详解

本文是纯理论工作，核心是一个定理（Theorem 1）及其证明，外加对 EoS 的解释和一组验证实验，没有可画成 pipeline 的工程流程，故不配框架图。整体逻辑链是：先用一个二次例子建立直觉 → 给出一般 $C^3$ 强凸极小点下的主定理 → 通过 Hessian 对角化 + 分量分离 + 不变集论证完成证明 → 把"方向 + 特征值扰动"组合起来解释 EoS 的 sharpness 振荡 → 在真实网络（SGD/Adam）上验证方向对齐普遍存在。

### 整体框架

设损失 $f\in C^3(\mathbb{R}^n)$，GD 迭代 $x_{k+1}=F(x_k)=x_k-\eta\nabla f(x_k)$，$x^*$ 是孤立局部极小点，$V_\eta=\{x_0:\lim_k x_k=x^*\}$ 是收敛到它的初值集合。再加一个技术性假设：任何零测集 $W$ 的原像 $F^{-1}(W)$ 也是零测的（GD 动力学分析里的惯例假设，用来排除一组测度为零的"病态"初值）。在 $x^*$ 处 Hessian 特征值满足 $0<\lambda_1<\lambda_2\le\cdots\le\lambda_{n-1}<\lambda_n$（即局部强凸）。在此设定下，本文给出二分式的主结论，并围绕"如何严格证明这个二分"以及"这个二分能解释什么"展开。各个关键设计点之间是层层递进的：主定理是骨架，证明技术是支撑，EoS 解释和现代优化器实验是它向外延伸的两个出口。

### 关键设计

**1. 主定理：收敛方向随学习率发生二分相变**

这是全文的骨架，直接回应"GD 临近极小点时沿哪个方向走"的痛点。对几乎所有初值 $x_0\in V_\eta$：

- 当 $0<\eta<\dfrac{2}{\lambda_1+\lambda_n}$（小学习率），**收敛方向存在**：$\displaystyle\lim_{k\to\infty}\frac{x_k-x^*}{\|x_k-x^*\|}$ 存在，且对齐到**最小特征值** $\lambda_1$ 对应的特征向量方向（最平缓方向）。
- 当 $\dfrac{2}{\lambda_1+\lambda_n}<\eta<\dfrac{2}{\lambda_n}$（大学习率），**交替收敛方向存在**：$\displaystyle\lim_{k\to\infty}(-1)^k\frac{x_k-x^*}{\|x_k-x^*\|}$ 存在，轨迹沿**最大特征值** $\lambda_n$ 对应的特征向量（最陡方向）逐迭代翻转符号、振荡逼近。

之所以分界点恰是 $2/(\lambda_1+\lambda_n)$，可以从线性主项 $1-\eta\lambda_i$ 的符号直观理解：当 $\eta<2/(\lambda_1+\lambda_n)$ 时，对最小特征值方向有 $1-\eta\lambda_1$ 最大且为正、收缩最慢，于是该方向"幸存"下来主导整条轨迹；当 $\eta$ 越过 $2/(\lambda_1+\lambda_n)$，最大特征值方向的因子 $1-\eta\lambda_n$ 变成绝对值最大的负数，每步翻号又收缩最慢，于是轨迹改由 $\lambda_n$ 方向主导并交替振荡。这把连续时间梯度猜想的"方向必收敛"推广到了离散 GD，还揭示出连续时间没有的新分支——振荡收敛。

**2. 证明技术：对角化 + 分量分离 + 前向不变集，把非线性问题锁回主导方向**

主定理的难点在于 $f$ 是一般非线性的，不能直接用二次解析解。本文的处理是：不失一般性设 $x^*=0$ 且 $\nabla^2 f(0)=\mathrm{diag}(\lambda_1,\dots,\lambda_n)$，把 GD 迭代逐分量写成线性主项 + 高阶余项

$$x_{k+1,i}=(1-\eta\lambda_i)x_{k,i}+g_i(x_k),\qquad g_i(x)=\eta(\lambda_i x_i-\partial_i f(x)),$$

其中 $g_i$ 是把非线性"残差"打包进去的二阶小量。记 $a=1-\eta\lambda_1$，$b=\max_{2\le i\le n}|1-\eta\lambda_i|$；由 $\eta<2/(\lambda_1+\lambda_n)<1/\lambda_1$ 可得 $a>b\ge0$，即最小特征值方向的收缩因子严格大于其余方向。

证明分三层。**第一层（Lemma 1）**借用作者前作（Chen et al., 2024）的前向不变性定理，构造一个含 $0$ 的开集 $\Omega\subset V_\eta$ 与常数 $C$，使轨迹永不逃逸，并给出关键估计 $|g_{k,i}|\le C\|x_k\|^2$、$\left|\partial g_{k,i}/\partial x_{k,j}\right|\le C\|x_k\|$——即高阶项相对线性主项可控。**第二层（Lemma 2）**选足够小的 $\varepsilon$ 让小球 $B(0,\varepsilon)$ 也前向不变。**第三层**定义"坏初值集" $S=\{x_0\in B(0,\varepsilon):\forall k,\ |x_{k,1}|<\sum_{i\ge2}|x_{k,i}|\}$ 并证明它零测；对其余初值，一旦某步 $|x_{k^*,1}|\ge\sum_{i\ge2}|x_{k^*,i}|$，就能用 $a>b$ 归纳出此后第一分量恒占优、且符号不变，进而对每个 $i\ge2$ 有

$$\frac{|x_{k,i}|}{|x_{k,1}|^\alpha}\le C_1\quad(1<\alpha<2)\ \Longrightarrow\ \lim_{k\to\infty}\frac{x_{k,i}}{x_{k,1}}=0,$$

于是 $\frac{x_k}{\|x_k\|}\to(\mathrm{sign}(x_{k^*,1}),0,\dots,0)$，方向锁定在 $\lambda_1$ 方向。大学习率情形（沿 $\lambda_n$ 振荡）的证明结构同理，放在附录。这套"对角化 → 分离主导分量 → 不变集 + 比值归零"的技术，本质是把一般函数的渐近行为压缩回二次主项的行为。

**3. sharpness 振荡：用方向结论解释 Edge of Stability**

EoS 的标志是训练中 sharpness（Hessian 最大特征值）$\lambda_n(x_k)$ 在阈值 $2/\eta$ 附近上下抖动。本文把"方向结论"和"特征值的一阶扰动"拼起来给出解释。由矩阵扰动理论，极小点邻域内 $\lambda_n(x)$ 可微，Taylor 展开 $\lambda_n(x)=\lambda_n+\omega^\top x+o(\|x\|)$。代入主定理的方向结论：

- 小学习率：$\lambda_n(x_k)=\lambda_n+C_\eta\|x_k\|+o(\|x_k\|)$，由于 $\|x_k\|$ 单调缩小，sharpness **单调**收敛到 $\lambda_n$（升或降取决于 $C_\eta$ 符号）。
- 大学习率：因为轨迹沿主方向逐步翻号，扰动项带上 $(-1)^k$：
$$\lambda_n(x_k)=\lambda_n+(-1)^k C_\eta\|x_k\|+o(\|x_k\|),$$
于是偶数步与奇数步分别从两侧逼近 $\lambda_n$，sharpness **振荡**收敛。更关键的是，由于此时 $\lambda_n<2/\eta<\lambda_1+\lambda_n$，在轨迹早期（$k$ 较小、$\|x_k\|$ 还大）$\lambda_n(x_k)$ 容易**冲过** $2/\eta$，这正对应 EoS 里 sharpness 偶尔越过 $2/\eta$ 红线、但长期包络仍收敛到 $\lambda_n$ 的现象。这等于把 EoS 的 sharpness 抖动从"经验观察"提升为方向定理的一个推论。

**4. 现代优化器的方向对齐：用相邻更新的余弦相似度验证**

理论只覆盖确定性 vanilla GD，作者好奇这种方向对齐是否在实用优化器里也成立。他们在 CIFAR-10 上用一个 CNN，分别跑带动量的 SGD 和 Adam，跟踪相邻两次参数更新的余弦相似度

$$\cos\langle\Delta x_{k+1},\Delta x_k\rangle=\frac{\langle\Delta x_k,\Delta x_{k+1}\rangle}{\|\Delta x_k\|\,\|\Delta x_{k+1}\|},\qquad \Delta x_k=x_{k+1}-x_k.$$

实验发现：当 loss 稳定下降收敛时，所有三种方法（GD/SGD/Adam）的该余弦值都趋向 $1$，即更新方向逐渐对齐到一个稳定方向，与理论预测一致。这把一个只对理想 GD 证明的结论，经验性地外推到了真实训练，暗示方向对齐可能是一种更普遍的优化几何性质（可用于设计学习率调度、初始化或新优化器）。

### 一个完整示例

以 $f(x,y)=x^2+y^2/2+x^2y+x^3$ 为例（局部极小点 $(0,0)$，$\lambda_1=1,\lambda_2=2$，故分界 $2/(\lambda_1+\lambda_2)=2/3$）：

- 取 $\eta=0.1<2/3$（小学习率）：轨迹沿 $v=(0,\mp1)$ 方向收敛到极小点，sharpness $\lambda_2(x_k)$ **单调**收敛到 $\lambda_2=2$，符合小学习率预测。
- 取 $\eta=0.95\in(2/3,\,1)$（大学习率，$2/\lambda_2=1$）：轨迹沿最陡方向 $v=(1,0)$ **振荡**前进，sharpness 在渐近值 $\lambda_2$ 附近抖动，并偶尔越过理论上界 $2/\eta\approx2.105$（图中红虚线）——这正是 EoS 现象，但振荡包络长期收敛到 $\lambda_2$ 印证了定理。

## 实验关键数据

本文是理论论文，"实验"主要用来印证定理而非刷指标。

### 方向相变（合成函数）

| 函数 | 学习率 $\eta$ | 理论分界 | 观察到的收敛方向 |
|------|------|------|------|
| $x^2/2+2y^2$ | $0.1$ | $2/5$ | 锁定 $x$ 轴（最小特征值方向） |
| $x^2/2+2y^2$ | $0.42$ | $2/5$ | 沿 $y$ 轴交替振荡（最大特征值方向） |
| $x^2/2+2y^2+xy^2+y^3$ | $0.1$ | — | 仍锁定 $x$ 轴（高阶扰动下规律不变） |
| $x^2/2+2y^2+xy^2+y^3$ | $0.42$ | — | 仍沿 $y$ 轴交替振荡 |

### sharpness 行为与现代优化器

| 配置 | 现象 | 与理论的关系 |
|------|------|------|
| $f=x^2+y^2/2+x^2y+x^3$，$\eta=0.1$ | $\lambda_2(x_k)$ 单调 $\to 2$ | 小学习率：方向固定 ⇒ sharpness 单调 |
| 同上，$\eta=0.95$ | $\lambda_2(x_k)$ 振荡 $\to 2$，偶尔越过 $2/\eta\approx2.105$ | 大学习率：方向翻号 ⇒ sharpness 振荡（EoS） |
| CIFAR-10 + CNN，SGD-momentum | $\cos\langle\Delta x_{k+1},\Delta x_k\rangle\to 1$ | 方向对齐在随机优化器上也出现 |
| CIFAR-10 + CNN，Adam | $\cos\langle\Delta x_{k+1},\Delta x_k\rangle\to 1$ | 方向对齐在自适应优化器上也出现 |

### 关键发现
- 收敛方向的"相变"完全由 $\eta$ 相对 $2/(\lambda_1+\lambda_n)$ 的位置决定，且对高阶扰动鲁棒，说明它是局部二次主项主导的渐近性质。
- EoS 中 sharpness 越过 $2/\eta$ 不是反常，而是大学习率下方向振荡带来的必然早期行为，长期包络仍收敛——这把经验现象纳入了统一解释。
- 方向对齐不局限于理想 GD：SGD、Adam 在稳定下降阶段同样出现相邻更新高度对齐，提示这是更普适的几何规律。

## 亮点与洞察
- **把连续时间梯度猜想搬到离散 GD**：连续梯度流"方向必收敛"是经典结果，本文用对角化 + 不变集 + 比值归零的离散技术复刻了它，还发现了连续时间没有的"振荡收敛"新分支，这个对照本身就很有启发。
- **一个干净的阈值 $2/(\lambda_1+\lambda_n)$ 统一了两类行为**：小于它走最平缓方向、大于它沿最陡方向振荡，且这个阈值恰好把 EoS 的 sharpness 振荡和经典稳定收敛切开，理论上很优雅。
- **方向 + 特征值一阶扰动 = EoS 解释**：用 $\lambda_n(x_k)=\lambda_n+(-1)^k C_\eta\|x_k\|+o(\|x_k\|)$ 一行就把 sharpness 的奇偶振荡说清楚，是个可复用的分析套路——只要知道轨迹方向，就能预测沿途任何光滑标量的振荡模式。
- **余弦相似度作为方向对齐的可观测代理**：$\cos\langle\Delta x_{k+1},\Delta x_k\rangle\to1$ 简单可测，可作为判断优化是否进入"方向锁定"阶段的实用信号。

## 局限与展望
- **只覆盖局部强凸极小点**：要求 Hessian 特征值严格正且彼此可分（$0<\lambda_1<\cdots<\lambda_n$），鞍点、平坦极小、退化 Hessian、一般非凸景观都不在覆盖范围内，作者也把推广到更一般函数（如满足 KL 条件的函数）列为未来工作。
- **理论只严格证明了 vanilla GD**：SGD/Adam 上的方向对齐目前仅有 CIFAR-10 的经验证据，没有证明；动量、自适应步长如何改变方向相变仍是开放问题。
- **只给方向，没给速率**：定理回答"朝哪个方向收敛"，但没量化"沿该方向收敛多快"，作者认为补上方向收敛速率能带来更锐的理论保证，并可能指导利用方向性的新优化器设计。
- **依赖一个技术性零测假设**（$F^{-1}$ 保零测集），以及对几乎所有初值成立——存在一组测度为零的病态初值不满足结论。

## 相关工作与启发
- **vs 梯度猜想（Parusinski et al., 2000）**：他们在连续时间梯度流上证明归一化割线收敛；本文把结论搬到离散 GD，多出一个学习率自由度，于是除"方向收敛"外还冒出"振荡收敛"分支，是离散化带来的新物理。
- **vs EoS（Cohen et al., 2021）**：他们经验观察到 sharpness 在 $2/\eta$ 附近徘徊；本文用方向定理 + 特征值一阶扰动给出一个解析解释，说明越过 $2/\eta$ 是大学习率下方向翻号的必然早期行为。
- **vs 作者前作（Chen et al., 2024，不稳定收敛）**：本文直接复用其前向不变性定理（Theorem 2）作为 Lemma 1 的基石，可视为同一研究线在"方向刻画"上的推进。
- **vs SAM / 长步长 / 动量等 EoS 变体（Long & Bartlett 2024；Grimmer 2024；Phunyaphibarn et al. 2024）**：这些工作从正则化、加速、catapult 等角度研究大学习率行为；本文提供了一个互补的"渐近方向"视角，并把它们都指向同一个可能的统一框架。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次把连续时间梯度猜想严格推广到离散 GD，并发现学习率驱动的方向相变与振荡收敛新分支
- 实验充分度: ⭐⭐⭐ 理论为主，合成例子 + CIFAR-10 验证到位，但现代优化器只有经验证据、无证明
- 写作质量: ⭐⭐⭐⭐ 直觉例子 → 主定理 → 证明 → EoS 解释链条清晰（个别拼写小瑕疵不影响理解）
- 价值: ⭐⭐⭐⭐ 为 GD 长期动力学与 EoS 提供了干净的理论视角，可启发利用方向性的新优化器设计

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Convergence Behavior of Preconditioned Gradient Descent Toward the Rich Learning Regime](on_the_convergence_behavior_of_preconditioned_gradient_descent_toward_the_rich_l.md)
- [\[ICLR 2026\] Gradient Descent with Large Step Sizes: Chaos and Fractal Convergence Region](gradient_descent_with_large_step_sizes_chaos_and_fractal_convergence_region.md)
- [\[ICML 2026\] On the Convergence Rate of LoRA Gradient Descent](../../ICML2026/optimization/on_the_convergence_rate_of_lora_gradient_descent.md)
- [\[ICLR 2026\] Fast Convergence of Natural Gradient Descent for Over-parameterized Physics-Informed Neural Networks](fast_convergence_of_natural_gradient_descent_for_over-parameterized_physics-info.md)
- [\[NeurIPS 2025\] Learning Provably Improves the Convergence of Gradient Descent](../../NeurIPS2025/optimization/learning_provably_improves_the_convergence_of_gradient_descent.md)

</div>

<!-- RELATED:END -->
