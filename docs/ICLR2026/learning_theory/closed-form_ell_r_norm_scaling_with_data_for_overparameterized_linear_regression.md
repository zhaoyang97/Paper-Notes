---
title: >-
  [论文解读] Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias
description: >-
  [ICLR2026][学习理论][过参数化] 对过参数化线性回归（各向同性高斯设计、最小 $\ell_p$ 插值，$p\in(1,2]$），本文用一个简单的"对偶射线"分析，给出整族参数范数 $\{\|\hat w_p\|_r\}_{r\in[1,p]}$ 随样本量 $n$ 缩放的**闭式高概率刻画**：存在一个数据相关的转折点 $n^\star$（"肘点"）和一个普适阈值 $r^\star=2(p-1)$，把会随 $n$ 饱和的范数和会继续增长的范数一刀切开；并把同一套规律迁移到由梯度下降训练的对角线性网络（DLN）。
tags:
  - "ICLR2026"
  - "学习理论"
  - "过参数化线性回归"
  - "过参数化"
  - "最小 $\\ell_p$ 插值"
  - "范数缩放律"
  - "对偶射线分析"
  - "对角线性网络"
  - "隐式偏置"
---

# Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=qPKTDOJ5Xs](https://openreview.net/forum?id=qPKTDOJ5Xs)  
**代码**: 待确认  
**领域**: 学习理论 / 过参数化线性回归  
**关键词**: 过参数化, 最小 $\ell_p$ 插值, 范数缩放律, 对偶射线分析, 对角线性网络, 隐式偏置

## 一句话总结
对过参数化线性回归（各向同性高斯设计、最小 $\ell_p$ 插值，$p\in(1,2]$），本文用一个简单的"对偶射线"分析，给出整族参数范数 $\{\|\hat w_p\|_r\}_{r\in[1,p]}$ 随样本量 $n$ 缩放的**闭式高概率刻画**：存在一个数据相关的转折点 $n^\star$（"肘点"）和一个普适阈值 $r^\star=2(p-1)$，把会随 $n$ 饱和的范数和会继续增长的范数一刀切开；并把同一套规律迁移到由梯度下降训练的对角线性网络（DLN）。

## 研究背景与动机
**领域现状**：现代机器学习的泛化度量越来越多地不靠参数个数、而靠**参数范数**来锚定（如各种基于 norm 的泛化界与诊断指标）。但绝大多数过参数化回归的分析对"范数"这件事处理得很笼统——默认就用 $\ell_2$，很少认真问：到底该用哪个 $\ell_r$？

**现有痛点**：一旦真的要用参数范数当泛化代理，就绕不开两个被长期忽视的问题。其一，选哪个 $r$ 会和"选出插值解的归纳偏置"（比如最小 $\ell_p$）发生耦合，二者交互效应不清楚。其二，已有对范数的精确刻画大多只盯住 $\ell_2$、或只针对极端稀疏的 $w^\star=e_1$ 特例（如 Donhauser et al. 2022），缺一个覆盖**整族** $\{\|\hat w_p\|_r\}_{r\in[1,p]}$ 的统一结论。

**核心矛盾**：实验里 sweep $(r,p)$ 会出现"非平凡"现象——同一个插值预测器，有的 $\ell_r$ 范数随 $n$ 增大很快**平台化（plateau）**，有的却**持续增长**且斜率各不相同；改变 $p$ 还会移动肘点、并重新分配"哪些 $r$ 会平台化"。也就是说，norm-based 代理的结论可能对 $r$ 的选择极其敏感，而这种敏感性又取决于底层的 $\ell_p$ 偏置。如果训练管线的有效 $p$ 未知（深度网络几乎总是如此），盲目挑一个 $r$ 来做泛化诊断是危险的。

**本文目标**：在最干净的设定——过参数化线性回归——里，给出 $\{\|\hat w_p\|_r\}_{r\in[1,p]}$ 随 $n$ 缩放的闭式、高概率刻画；再把这套图景桥接到隐式偏置主导的 DLN，连通"显式 $\ell_p$ 惩罚"与"优化诱导的隐式几何"两条路径。

**切入角度**：作者不去直接追踪优化器解 $\hat w_p$，而是看它的**对偶证书**——一个既要拟合标签、又要在过设计后满足范数预算的向量。把对偶沿标签方向压成一条一维"射线"，就能暴露出一个决定一切的诊断尺度 $t_\star$。这个角度的好处是把高维问题降到一维，且天然分离出 $X^\top Y$ 里的"信号尖峰"与"噪声体（bulk）"两股竞争力量。

**核心 idea**：用一维对偶射线平衡"信号尖峰 vs. 高维噪声体"，直接读出转折样本量 $n^\star$ 与普适阈值 $r^\star=2(p-1)$，从而一张图说清整族 $\ell_r$ 谁饱和、谁增长、以什么指数增长。

## 方法详解

### 整体框架
研究对象是过参数化线性模型：设计矩阵 $X\in\mathbb{R}^{n\times d}$（$d>n$，行 i.i.d. $\mathcal{N}(0,I_d)$），$Y=Xw^\star+\xi$，$\xi\sim\mathcal{N}(0,\sigma^2 I_n)$；估计器取最小 $\ell_p$ 插值

$$\hat w_p\in\arg\min_{w\in\mathbb{R}^d}\|w\|_p\quad\text{s.t.}\quad Xw=Y,\qquad p\in(1,2].$$

记真信号支撑大小 $s=\|w^\star\|_0$、$\tau_s^2:=\|w^\star\|_2^2+\sigma^2$。目标是刻画整族 $\{\|\hat w_p\|_r\}_{r\in[1,p]}$ 随 $n$ 的缩放。

整体逻辑是一条"从对偶到判据再到两相"的链：先把原始约束问题写成对偶，把对偶限制在沿标签 $Y$ 的射线上得到一个一维标量 $t_\star$；再把 $X^\top Y$ 拆成"信号尖峰 + 噪声体"两块，看谁主导 $t_\star$；两块力量的平衡点定义出转折样本量 $n^\star$；最后通过 KKT 把对偶尺度翻译回原始坐标的范数，得到尖峰主导（$n\gg n^\star$）与体主导（$n\ll n^\star$）两个区间的闭式表达式，并在尖峰区读出阈值 $r^\star=2(p-1)$。这套分析既适用于显式最小 $\ell_p$，又通过一次 $\alpha\mapsto p_{\text{eff}}$ 标定迁移给 DLN。

### 关键设计

**1. 对偶射线约简：把高维插值压成一维诊断尺度**

最小 $\ell_p$ 插值的难点在于要追踪一个高维优化器。作者改走对偶：约束问题 $\min_w \tfrac1p\|w\|_p^p$ s.t. $Xw=Y$ 的无约束对偶是 $\max_\lambda\ \lambda^\top Y-\tfrac1q\|X^\top\lambda\|_q^q$（$q=\tfrac{p}{p-1}$ 是 $p$ 的共轭指数），KKT 条件为 $Xw=Y$ 且 $X^\top\lambda=\nabla f(w)$。关键一步是把对偶变量限制在沿标签的射线 $\lambda=tY$ 上，于是最优 $t_\star$ 满足

$$t_\star^{\,q-1}=\frac{\|Y\|_2^2}{\|X^\top Y\|_q^q}.$$

这把一个 $d$ 维优化坍缩成一个标量平衡，$t_\star$ 就是控制全部 $\ell_r$ 行为的单一诊断量。相比 Koehler et al. (2021)、Donhauser et al. (2022) 用高斯极小极大定理（GMT/CGMT）从头推范数上界，本文的对偶射线论证更"第一性"，不调用 GMT/CGMT，却能直接产出闭式的 $n$-缩放律。

**2. 信号尖峰 vs. 噪声体的分块竞争**

$t_\star$ 的分母 $\|X^\top Y\|_q^q$ 决定一切，而它由两股力量竞争。把 $X^\top Y$ 按坐标分成两块：落在真信号支撑 $S$ 上的"**尖峰（spike）**"坐标，随 $n$ 相干累积；以及大量零坐标构成的"**体（bulk）**"，每个都是小而嘈杂的贡献、但数量 $\sim d-s$ 极大。高概率下

$$t_\star^{\,q-1}\ \asymp\ \underbrace{\frac{\tau_s^2 n}{n^q W_q}}_{\text{spike}}\ +\ \underbrace{(d-s)\,m_q\,\tau_s^q\, n^{q/2}}_{\text{bulk}}\ +\ O(\cdot),$$

其中 $W_q:=\|w^\star\|_q^q$、$m_t:=\mathbb{E}|Z|^t$（$Z\sim\mathcal N(0,1)$）。"spike/bulk"二字指的正是 $\|X^\top Y\|_q$ 的哪一部分主导 $t_\star$。当 $n$ 小，体项靠数量取胜、解的质量被摊到许多坐标上；当 $n$ 大，尖峰相干增长占上风、质量集中到支撑上。这个分块完全靠高斯设计的标准集中不等式（Gaussian concentration、blockwise 分解）撑起，是后面所有相变结论的物理来源。

**3. 普适阈值 $r^\star=2(p-1)$ 与转折点 $n^\star$：一张图说清谁饱和谁增长**

让尖峰项与体项相等，就解出数据相关的转折样本量

$$n^\star\ \asymp\ \Big(\kappa_{\text{bulk}}\frac{\tau_s^q}{W_q}\Big)^{\frac{2}{q-2}},$$

即图里肉眼可见的"肘点"（$\kappa_{\text{bulk}}=\lim (d-s)/n$）。把 $t_\star$ 经 KKT 翻回原始坐标后，主定理给出统一界 $\|\hat w_p\|_r\asymp\max\{\cdots\}$，并在两个极端坍缩成干净形式。**尖峰主导（$n\gg n^\star$）**：

$$\|\hat w_p\|_r\ \asymp\ \begin{cases}\dfrac{\tau_s^{q+1}}{W_q}\,n^{\frac1r-\frac{1}{2(p-1)}},& r\le 2(p-1),\\[2mm]\dfrac{\tau_s^2}{W_q}\,\|w^\star\|_{(q-1)r}^{q-1},& r>2(p-1).\end{cases}$$

这里就跳出了**普适阈值** $r^\star=2(p-1)$：当 $r>r^\star$，指数 $\tfrac1r-\tfrac{1}{2(p-1)}<0$，范数停在 $\asymp\tau_s^2/W_q$ 的平台上不再涨；当 $r\le r^\star$，范数以显式斜率 $\tfrac1r-\tfrac{1}{2(p-1)}$ 继续增长。阈值只由 $p$ 决定（$p$ 越接近 1，$r^\star$ 越小，越多 $r$ 会增长）。**体主导（$n\ll n^\star$）**：诸 $\ell_r$ 大致按 $n^{1/r-1/2}$ 增长，呈现典型的 $n^{1/2}$ 型趋势与跨 $r$ 的固定排序。这一条把"哪些范数饱和、哪些增长、以什么指数"压进一个闭式判据，正是论文的核心贡献。作者还把它具体化到两个范例靶（单尖峰 $w^\star=e_1$ 与平坦 $s$-稀疏向量），给出 $n^\star$、平台高度与增长斜率的可代入公式：两者阈值与指数相同，差别只在尺度（平坦靶的 $n^\star$ 随 $s$ 近似线性右移）。

**4. DLN 桥接：把初始化尺度 $\alpha$ 标定成有效 $p_{\text{eff}}(\alpha)$**

显式最小 $\ell_p$ 是"摆明的"归纳偏置；而由梯度下降训练的对角线性网络（DLN，权重对角、有效预测器是各层参数的逐坐标乘积）则是"隐式"几何。已有理论指出 DLN 的可分梯度流势 $Q_\alpha$ 让初始化尺度 $\alpha$ 连续调节隐式偏置：$\alpha\to 0$ 趋向 $\ell_1$ 式稀疏几何，$\alpha\to\infty$ 趋向 $\ell_2^2$ 式稠密几何。为把 DLN 跟显式 $\ell_p$ 实验对齐，作者做一次**与数据无关**的标定：在 $k$-稀疏、单位 $\ell_2$ 的探针上估 $Q_\alpha$ 关于 $k$ 的 log-log 斜率，匹配 $\|\cdot\|_p^p$ 的精确 $k^{1-p/2}$ 律，得到单调映射 $\alpha\mapsto p_{\text{eff}}(\alpha)$（极限 $p_{\text{eff}}\to 1$（$\alpha\to0$）、$\to 2$（$\alpha\to\infty$））。标定后 DLN 的 $\ell_r$-vs-$n$ 曲线**继承同一套肘点/阈值结构**，从而在显式偏置与隐式偏置之间架起可预测的桥。作者还指出有限学习率会改变图景：在有标签噪声时，更大的 lr 把（随机）梯度下降变成带"有效温度"的噪声动力学，把预测器推向更圆（更不稀疏）的几何、相当于抬高 $p_{\text{eff}}$，使质量漏进体坐标、推迟尖峰主导并在最终平台前抬高 $\ell_r$。

### 损失函数 / 训练策略
显式部分无训练——直接解最小 $\ell_p$ 插值的凸规划。DLN 部分用梯度下降训练对角参数化模型；实验固定 $d=50{,}000$、$\sigma=0.1$，扫 $\alpha\in\{0.00102,0.0664,0.229\}$（按标定 $\approx p\in\{1.1,1.5,1.9\}$），并研究学习率与噪声对 $\ell_r$ 缩放的影响。

## 实验关键数据

### 主实验
实验意在**验证理论曲线**而非刷 SOTA：固定 $d=50{,}000$、$\sigma=0.1$，扫 $p\in\{1.1,1.5,1.9\}$ 与 $n$，每张图叠加测试 MSE（左轴）与代表性 $\ell_r$ 曲线（右轴），并用理论 overlay（bulk/spike）对照。

| 设定 | 观测现象 | 与理论是否吻合 |
|------|----------|----------------|
| 单尖峰 $w^\star=e_1$，$p=1.5$ | 在 $n^\star\approx1.4\times10^3$ 附近出现清晰肘点；越过后 $r>2(p-1)$ 曲线平台、其余保持斜率 | 与 $n^\star$ 公式及尖峰区表达式一致 |
| 单尖峰，$p=1.9$（稠密倾向） | 全程体主导，$\ell_r$ 近似按 $n^{1/2}$ 增长且按 $r$ 排序 | 与体主导公式一致 |
| 单尖峰，$p=1.1$（稀疏倾向） | $r>2(p-1)$ 早早平台，$r$ 较小者继续增长 | 与阈值规则一致 |
| 平坦靶 $w^\star$（$s=50$） | 同样的平台/增长规则，但转折尺度更大（$p=1.5$ 肘点后移甚至离屏） | 与 $n^\star$ 随 $s$ 近似线性右移一致 |

### 消融 / 分析实验

| 配置 | 关键现象 | 说明 |
|------|----------|------|
| 加大稀疏度 $s\in\{500,5000\}$ | 定性图景重现但整体右移；$p=1.1$、$s=5000$ 时出现**双下降**（泛化误差先升后降），$\ell_{1.1}$ 沿 bulk 导线持续上升 | 印证 $n^\star$ 随 $s$ 增长 |
| DLN（标定 $\alpha$ 后），$w^\star=e_1$ | 更小 $\alpha$（更小 $p_{\text{eff}}$）更早进入尖峰主导、$r>2(p-1)$ 平台；更大 $\alpha$ 维持体主导的 $n^{1/2}$ 增长 | DLN 继承显式 $\ell_p$ 的肘点/阈值结构 |
| 有限学习率 + 噪声（$\sigma\in\{0.1,0.5\}$） | 增大 lr 使 $\ell_{1.1}$ 稳步上升、肘点右移；$\sigma=0$ 时 $\ell_{1.1}$ 快速平台且对 lr 不敏感 | lr×噪声相当于抬高有效 $p_{\text{eff}}$ |

### 关键发现
- **阈值 $r^\star=2(p-1)$ 是分水岭**：它只由 $p$ 决定，跨单尖峰/平坦靶不变；越过它范数从"增长"切到"饱和"。
- **肘点 $n^\star$ 随支撑大小 $s$ 右移**：稀疏越强，体主导窗口越长，越容易看到双下降。
- **隐式与显式偏置同律**：一次数据无关的 $\alpha\mapsto p_{\text{eff}}$ 标定就让 DLN 复现同一套相变，说明这套规律不止于显式 $\ell_p$。

## 亮点与洞察
- **一维对偶射线把高维问题降维**：不追踪优化器、只看沿标签方向的对偶标量 $t_\star$，绕开了 GMT/CGMT 的重机械，从第一性直接拿到闭式 $n$-缩放律——这是最漂亮的一招，可迁移到其他带 $\ell_p$ 惩罚的高维估计问题。
- **"spike vs bulk"是一个统一的物理图像**：把 $X^\top Y$ 拆成相干信号与高维噪声体两股力量的竞争，既给出 $n^\star$ 的来源，又解释了双下降为何在强稀疏时显现。
- **对实践的告诫很实在**：既然大量 norm-based 泛化代理依赖 $\|\hat w\|_r$，而不同 $(r,p)$ 能给出**相反**的缩放（平台 vs 增长），用范数界做诊断（尤其深网，有效 $p$ 未知）要格外谨慎——这把一个看似纯理论的结论落到了可操作的警示上。

## 局限与展望
- **设定理想化**：各向同性高斯设计、$d/n\to\kappa$ 的比例渐近、$p\in(1,2]$（不含 $\ell_1$ 极端稀疏的角点几何）。相关分析依赖高斯集中，换成重尾/相关设计是否仍成立未验证。
- **DLN 部分以经验为主**：$\alpha\mapsto p_{\text{eff}}$ 标定后"继承同律"是实验观察而非定理；有限学习率的"有效温度"解释也是启发式的物理图像，缺严格刻画。
- **只到范数缩放、未给新泛化界**：作者明确说不发展新的泛化界，与泛化的联系是间接的；把 $\ell_r$ 缩放律真正接到一个可用的泛化上界仍是开放问题。
- 可改进方向：推广到各向异性/相关协方差、把 $p_{\text{eff}}$ 的隐式偏置标定做成有理论保证的定理、以及在真实深网上检验"对 $r$ 敏感"的实证后果。

## 相关工作与启发
- **vs Donhauser et al. (2022)**：他们用 CGMT 给最小 $\ell_p$ 插值的范数上界，但主要局限在 $w^\star=e_1$ 的极端稀疏极限；本文不限于该极限、覆盖整族 $\{\|\hat w_p\|_r\}_{r\in[1,p]}$，并用对偶射线给出闭式 $n$-缩放而非仅上界。
- **vs Koehler et al. (2021)**：同样把范数上界当作通往泛化的中间步骤，走 GMT；本文方法论上更轻、从对偶平衡第一性出发，直接产出转折点与阈值的显式表达。
- **vs ridge/lasso/elastic-net/bridge 等经典 $\ell_p$ 估计刻画**：以往关心"哪个 $p$ 选出哪种估计器"；本文把问题转成"给定 $p$，所有 $\ell_r$ 诊断随数据怎么变"，是对该脉络的正交补充。
- **vs 隐式正则 / DLN 路线（Soudry, Gunasekar, Woodworth 等）**：本文不另造隐式偏置理论，而是借 $\alpha\mapsto p_{\text{eff}}$ 把 DLN 的隐式几何映回显式 $\ell_p$ 几何，从而把"显式偏置的相变律"直接搬过去验证。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个覆盖整族 $\ell_r$ 的统一闭式缩放律，对偶射线论证干净且回避了 GMT/CGMT。
- 实验充分度: ⭐⭐⭐⭐ 系统 sweep $(r,p,s,\alpha,\text{lr},\sigma)$ 充分支撑理论，但全是合成数据、无真实网络验证。
- 写作质量: ⭐⭐⭐⭐ 主定理与两区间推论结构清晰，spike/bulk 图像贯穿全文；公式密度高、对非理论读者门槛偏陡。
- 价值: ⭐⭐⭐⭐ 对"用哪个范数做泛化代理"给出可操作告诫，理论桥接显式/隐式偏置，对理解过参数化有参考价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Covariate-Guided Clusterwise Linear Regression for Generalization to Unseen Data](covariate-guided_clusterwise_linear_regression_for_generalization_to_unseen_data.md)
- [\[ICLR 2026\] Learning under Quantization for High-Dimensional Linear Regression](learning_under_quantization_for_high-dimensional_linear_regression.md)
- [\[ICLR 2026\] Larger Datasets Can Be Repeated More: A Theoretical Analysis of Multi-Epoch Scaling in Linear Regression](larger_datasets_can_be_repeated_more_a_theoretical_analysis_of_multi-epoch_scali.md)
- [\[ICLR 2026\] Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks](implicit_bias_produces_neural_scaling_laws_in_learning_curves_from_perceptrons_t.md)
- [\[ICLR 2026\] A New Approach to Controlling Linear Dynamical Systems](a_new_approach_to_controlling_linear_dynamical_systems.md)

</div>

<!-- RELATED:END -->
