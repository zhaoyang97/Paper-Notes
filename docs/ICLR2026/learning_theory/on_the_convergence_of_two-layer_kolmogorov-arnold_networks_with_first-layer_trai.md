---
title: >-
  [论文解读] On the Convergence of Two-Layer Kolmogorov-Arnold Networks with First-Layer Training
description: >-
  [ICLR 2026][学习理论][Kolmogorov-Arnold 网络] 本文在过参数化、只训练第一层系数的设定下，证明两层 KAN 用梯度下降必然收敛到全局最优（零训练误差），给出一个由"标签在 KAN 切核特征谱上的投影"决定的细粒度收敛速率，并指出 KAN 只需 $m=O(n^2)$ 的隐层宽度就能保证收敛——相比经典 ReLU 两层网络的 $m=O(n^6)$ 是一个多项式级的巨大改进。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "优化收敛"
  - "Kolmogorov-Arnold 网络"
  - "切核 (NTK)"
  - "过参数化"
  - "全局收敛"
  - "lazy training"
---

# On the Convergence of Two-Layer Kolmogorov-Arnold Networks with First-Layer Training

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=buuwRBYfrP](https://openreview.net/forum?id=buuwRBYfrP)  
**代码**: 待确认  
**领域**: 学习理论 / 优化收敛  
**关键词**: Kolmogorov-Arnold 网络, 切核 (NTK), 过参数化, 全局收敛, lazy training

## 一句话总结
本文在过参数化、只训练第一层系数的设定下，证明两层 KAN 用梯度下降必然收敛到全局最优（零训练误差），给出一个由"标签在 KAN 切核特征谱上的投影"决定的细粒度收敛速率，并指出 KAN 只需 $m=O(n^2)$ 的隐层宽度就能保证收敛——相比经典 ReLU 两层网络的 $m=O(n^6)$ 是一个多项式级的巨大改进。

## 研究背景与动机

**领域现状**：Kolmogorov-Arnold 网络（KAN）把可学习的一维样条函数放在边上、节点只做求和，被认为比 MLP 更可解释、更省参数，近两年在视觉、时序、PDE 求解等任务上拿到了不错的经验结果。但它"为什么能被梯度下降训好"这件事，理论上几乎还是空白。

**现有痛点**：过参数化网络的训练动力学已有成熟的切核（NTK）分析框架——对两层 ReLU 网络，Du et al. (2019) 证明只要宽度多项式级地大于样本数 $n$，梯度下降就能收敛到全局最优。但这套分析直接搬到 KAN 上并不成立：KAN 的边函数是基函数的线性组合（如 RBF、多项式），其切核形式完全不同，没人推导过它的闭式表达，也没人量化过它需要多宽。Gao & Tan (2025) 证明了"两层全部训练"的 KAN 会收敛，但宽度需求高（$\tilde O(g^9 n^3/\lambda_0^4)$）、对最小特征值 $\lambda_0$ 的依赖很差（$\lambda_0^{-4}$）。

**核心矛盾**：要把训练动力学讲清楚，就必须有一个"稳定且可分析"的切核；但同时训练两层会让分析变复杂、宽度与稳定性的代价都变高。能不能用一个更受限、更干净的设定，换来更紧的宽度界和更鲁棒的保证？

**切入角度**：作者借鉴标准神经网络分析里的经典简化——**只训练第一层、固定随机初始化的第二层**（Du et al. 2019 对 MLP 用过同样的招）。这样切核只对第一层系数求导，动力学更接近核回归，能推出闭式切核并做精细分析。

**核心 idea**：在"只训第一层 + 过参数化"下，建立 **KAN 切核（KAN-TK）**，用它一举证明全局收敛、给出标签依赖的收敛速率，并量化出 KAN 相对 MLP 的参数效率优势（$O(n^2)$ vs $O(n^6)$ 宽度）。

## 方法详解

### 整体框架

本文是一篇纯理论分析，对象是如下两层 KAN：输入 $x\in[0,1]^d$，隐层宽度 $m$，标量输出

$$f(x)=\frac{1}{\sqrt m}\sum_{p=1}^{m}\sum_{l=1}^{g}\beta_{pl}\,\phi_l(z_p),\qquad z_p=\sum_{k=1}^{d}\sum_{j=1}^{g}\alpha_{pjk}\,\phi_j(x_k)$$

其中 $\{\phi_j\}_{j=1}^g$ 是 $g$ 个基函数（分析时取 RBF），$\alpha_{pjk}$ 是第一层可学习系数，$\beta_{pl}$ 是第二层系数，$\tfrac{1}{\sqrt m}$ 是过参数化分析里的标准缩放。训练协议：$\alpha$ 从 $\mathcal N(0,\sigma^2)$ 初始化、用全批量梯度下降更新；$\beta$ 从 $\{-1,+1\}$ 均匀初始化后**全程冻结**；损失是 MSE $L=\tfrac12\|y-u\|_2^2$。

整条论证链路是：先**定义并推出 KAN 切核** $H_{ij}=\langle\nabla_\alpha f(x_i),\nabla_\alpha f(x_j)\rangle$ 的无穷宽闭式（针对 RBF 基），再**证明在足够宽时这个切核全程稳定**（lazy training），由稳定性导出**损失线性收敛到 0**；接着把收敛速率**按切核特征谱细化**，得到标签依赖的误差界；最后把宽度、参数量、学习率与两类基线**对比**，量化 KAN 的优势与代价。这是一条"建核 → 证稳定 → 证收敛 → 谱细化 → 比效率"的标准 NTK 分析骨架，没有 pipeline 式的工程模块，因此不配框架图。

### 关键设计

**1. KAN 切核（KAN-TK）的闭式推导：给 KAN 找一把"决定动力学"的尺子**

NTK 理论的核心是：在 lazy training 区，网络输出可以用初始点附近的一阶 Taylor 展开近似，复杂的非线性动力学就退化成一个固定核的核回归。对本文的两层 KAN，由于只训第一层 $\alpha$，切核就对 $\alpha$ 求梯度内积。作者针对一维输入 + RBF 基 $\phi_j(x)=\exp\!\big(-\tfrac{(x-\mu_j)^2}{2\sigma^2}\big)$，在无穷宽极限 $m\to\infty$ 下推出了 $H^\infty$ 的闭式表达（Proposition 3.1）：每个元素 $(H^\infty)_{qr}$ 是一串对基函数下标 $j,l$ 求和、含一组辅助张量（$A^{qr},b^{qr},G^{qr}=(I+A^{qr}/\sigma^2)^{-1},T^{qr}_l,Z^{qr}_l,Y^{qr}_{sl},X^{qr}_{psl}$）的表达式。它形式上高度复杂、随样本数 $n$ 多项式增长，不适合大规模直接计算；但它的价值在于**能算出 KAN-TK 的特征值与特征向量**——这正是后面分析标签对齐和收敛行为的钥匙。换句话说，这一步是把"KAN 怎么训"翻译成"一个具体核的谱长什么样"。

**2. 只训第一层 + lazy training，证明全局收敛到零误差：用更干净的设定换更紧的界**

在三条标准假设下（基函数有界、一二阶导有界且 $\phi_l(0)=0$；无穷宽核正定即 $\lambda_0=\lambda_{\min}(H^\infty)>0$；标签有界 $|y_i|\le1$），定理 4.2 证明：只要隐层足够宽且初始化方差足够小，

$$m\gtrsim\max\!\Big(\frac{d^2 g^6 n^2}{\lambda_0^2}\log\frac n\delta,\;n\Big),\qquad \sigma=O\!\Big(\frac{\delta}{\sqrt{mng^3 d}}\Big),$$

则以至少 $1-O(\delta)$ 的概率，梯度下降满足线性收敛 $L(t+1)\le\big(1-\tfrac{\eta\lambda_0}{2}\big)L(t)$，学习率 $\eta=O\!\big(\tfrac{\lambda_0}{n^3 d^2 g^6}\big)$。证明用归纳法走"lazy training"路线：靠三条引理把动力学钉在 $H^\infty$ 附近——系数稳定性（$|\alpha_{ijk}(t)-\alpha_{ijk}(0)|\le R$，$R=O(\tfrac{g\sqrt n}{\lambda_0\sqrt m}\|u(0)-y\|_2)$）、时间维核稳定性（$\|H(t)-H(0)\|_2\le 2n^2 d^2 g^4 R$）、初始核集中（$\|H(0)-H^\infty\|_2\le\tfrac{dg^3 n}{\sqrt m}\sqrt{\log(2n^2/\delta)}$）。三者合起来保证每步损失都按固定比例下降。**只训第一层**是关键简化：它让权重几乎不动、核几乎不变，从而把宽度需求压到 $O(n^2)$、把对 $\lambda_0$ 的依赖从 $\lambda_0^{-4}$ 改善到 $\lambda_0^{-2}$。

**3. 标签依赖的收敛速率：收敛快不快，取决于标签落在哪些特征向量上**

定理 4.2 给的是一个由 $\lambda_0$ 决定的"最坏情况"速率，但实践中不同任务收敛快慢差别很大。定理 4.6 把速率沿切核特征谱细化：设 $H^\infty=\sum_{i=1}^n\lambda_i v_i v_i^\top$，则误差向量满足

$$\|y-u(t)\|_2\le\sqrt{\sum_{i=1}^{n}(1-\eta\lambda_i)^{2t}(v_i^\top y)^2}\;\pm\;\epsilon,$$

其中 $\epsilon$ 随 $m\to\infty$ 消失。证明把输出更新近似成 $u(t+1)-u(t)\approx-\eta H^\infty(u(t)-y)$，递推展开后取范数并代入特征分解即得。它的解读（Remark 1）很直观：与**大特征值** $\lambda_i$ 对应的特征向量方向上的误差衰减最快；若标签 $y$ 在这些 top 特征向量上投影很强（即标签结构正好是核"擅长学"的），整体收敛会远快于标签随机或对齐到小特征值方向时。这把"为什么有些任务好学"量化成了"标签与核谱的对齐程度"。

**4. 宽度/参数/学习率三方对比：把 KAN 的效率优势与代价摆上台面**

作者把本文方法与两个基线对齐对比（同一套 TK 稳定性分析方法）：经典两层 ReLU NN（Du et al. 2019）和两层全训 KAN（Gao & Tan 2025）。宽度上，ReLU NN 需 $O(n^6/(\lambda_0^4\delta^3))$、全训 KAN 需 $\tilde O(g^9 n^3/\lambda_0^4)$，而本文只训第一层只需 $O(d^2 g^6 n^2/\lambda_0^2)$——对 $n$ 是 $O(n^2)$ vs $O(n^6)$ 的多项式级改进，对 $\lambda_0$ 的依赖也从四次方降到二次方（$\lambda_0$ 若缩小 $k$ 倍，NN 要多 $k^4$ 倍宽度，KAN 只要 $k^2$ 倍，更鲁棒）。代价是对输入维度 $d$ 和基函数个数 $g$ 的依赖更强，但因实际中样本量 $n$ 通常主导，作者认为这是划算的。Remark 2 给出机理解释：KAN 把神经元级激活换成边上**平滑的一维样条**，中间表示是光滑一维函数的复合，其 NTK 只依赖样条的有界导数、且最多只涉及样本间的**两两交互**，于是宽度只需对 $n$ 二次；而 ReLU 网络要维持离散激活模式的稳定性、需控制样本间的**高阶交互**，才放大成 $O(n^6)$。另一面的代价见 Remark/Table 2：本文允许的学习率 $\eta=O(\lambda_0/(n^3 d^2 g^6))$ 比两个基线都小，由于线性速率正比于 $\eta\lambda_0$，**单步收敛更慢**——这是"用收敛速度换参数效率与稳定性"的权衡。

## 实验关键数据

实验用基于 FastKAN 的两层 RBF-KAN，只训第一层、全批量 GD，目的是验证而非刷点。

### 主实验：宽度对收敛的影响（验证定理 4.2）

| 设置 | 现象 | 对应理论 |
|------|------|---------|
| $n=100,d=100$，标签 $\sim\mathcal N(0,1)$，$m\in\{500,\dots,32000\}$ | $m$ 越大训练误差下降越快 | 定理 4.2：更宽更快收敛 |
| 同上，看 $\|\alpha(t)-\alpha(0)\|_\infty$ | $m$ 越大权重偏移越小 | 引理 4.3：lazy training |

### 标签结构对收敛的影响（验证定理 4.6）

| 标签类型 | 谱投影特征 | 收敛速度 |
|----------|-----------|---------|
| 结构化（$y=\sin^2(0.7x/2)/\sin^2(x/2)$） | 能量集中在 top 特征向量 | 最快 |
| 随机（i.i.d. $\mathcal N(0,1)$） | 能量在谱上较均匀分布 | 中等 |
| 反结构化（$H^\infty$ 最小特征值对应的特征向量） | 集中在最小特征值方向 | 最慢 |

（1D 数据，$n=30/50$，$m=5000$，3000 epochs。）

### 关键发现
- **lazy training 被双重证实**：宽网络既收敛更快、权重又偏移更小，两条曲线同时印证"足够宽时权重几乎不动、核几乎不变"的核心假设。
- **收敛速度由标签—谱对齐主导**：结构化 → 随机 → 反结构化的收敛速度单调变慢，定量支持了"误差在大特征值方向衰减最快"的 Remark 1。
- **正定性假设在实践中成立**：用 FastKAN 在多种输入分布上观察到 $H^\infty$ 的最小特征值严格为正（如 $[-1,1]$ 上 linspace 取到 $3.29\times10^{-4}$），支撑了 $\lambda_0>0$ 的假设。

## 亮点与洞察
- **把"KAN 为什么省参数"从直觉变成定理**：之前都说 KAN 更省参数，本文用 NTK 框架给出了量化答案——$O(n^2)$ vs $O(n^6)$ 的宽度需求，且把根因归到"边上平滑样条只产生样本两两交互、不像 ReLU 要控高阶激活模式翻转"，这个机理解释很有迁移价值。
- **闭式 KAN-TK 是可复用的分析工具**：RBF 基下的切核闭式表达虽然计算重，但它给出了谱，能直接拿来做核回归、分析标签对齐——是后续研究 KAN 动力学的基础设施。
- **"只训第一层"换来更鲁棒的 $\lambda_0$ 依赖**：把对最小特征值的依赖从 $\lambda_0^{-4}$ 降到 $\lambda_0^{-2}$，在 $\lambda_0$ 很小（实测可到 $10^{-4}$ 量级）时意义重大，是个值得借鉴的"简化设定换稳定性"思路。

## 局限与展望
- **设定高度受限**：只训第一层、第二层冻结、一维输入 + RBF 基才有闭式核——这与实际中两层全训、多维输入、B-spline/多项式等基的 KAN 有差距，结论能否外推到深层 KAN 尚未知。
- **以参数效率换收敛速度**：作者坦承本方法学习率更小、单步收敛更慢，纯粹是优化理论上的权衡，未必是实际训练的最优选择。
- **切核计算不可扩展**：KAN-TK 闭式随 $n$ 多项式爆炸，只能用于小规模理论验证，无法直接用于大规模任务。
- **基线选择偏保守**：对比锚定 Du et al. (2019) 的 $O(n^6)$；而 Polaczyk & Cyranka (2023) 用 Clarke 次微分能把 ReLU 网络压到 $O(n^{1.25})$，作者为方法一致性没用更紧的工具，因此 $O(n^2)$ 的优势相对的是"经典界"而非"最紧界"。
- **展望**：扩展到深层 KAN、分析 Adam 等随机优化、用 NTK 之外的工具（如微分包含）求更紧界、推导多维输入与其他基的闭式 KAN-TK。

## 相关工作与启发
- **vs Du et al. (2019)（两层 ReLU NN）**：同样用 TK 稳定性分析，他们证 MLP 需 $O(n^6)$ 宽度，本文证 KAN 只需 $O(n^2)$；本文优势是宽度与 $\lambda_0$ 依赖都更好，劣势是允许的学习率更小、收敛更慢。
- **vs Gao & Tan (2025)（两层全训 KAN）**：他们训两层、需 $\tilde O(g^9 n^3/\lambda_0^4)$ 宽度且 $\lambda_0^{-4}$ 依赖；本文只训第一层，宽度降到 $O(n^2)$、$\lambda_0$ 依赖改善到 $\lambda_0^{-2}$，代价是更强的 $d,g$ 依赖和更慢的收敛——是"受限设定换稳定性与效率"的权衡。
- **vs Polaczyk & Cyranka (2023)**：他们用 Clarke 次微分/微分包含把 ReLU 网络宽度界精炼到 $O(n^{1.25})$；本文为与基础结果方法一致，未采用该工具，但指出把它移植到 KAN 是求更紧界的有希望方向。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次给出"只训第一层"两层 KAN 的闭式切核与全局收敛证明，并量化参数效率优势。
- 实验充分度: ⭐⭐⭐ 实验仅为小规模合成数据上的理论验证，无真实任务，但精准对应每条定理。
- 写作质量: ⭐⭐⭐⭐ 论证链路清晰，Remark 把机理与权衡讲得很到位。
- 价值: ⭐⭐⭐⭐ 为 KAN 训练理论打下扎实地基，机理解释（平滑样条→两两交互→$O(n^2)$）有迁移价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Two-Layer Convolutional Autoencoders Trained on Normal Data Provably Detect Unseen Anomalies](two-layer_convolutional_autoencoders_trained_on_normal_data_provably_detect_unse.md)
- [\[ICML 2026\] Two-Layer Linear Auto-Regressive Models Estimate Latent States](../../ICML2026/learning_theory/two-layer_linear_auto-regressive_models_estimate_latent_states.md)
- [\[ICLR 2026\] High-Dimensional Analysis of Single-Layer Attention for Sparse-Token Classification](high-dimensional_analysis_of_single-layer_attention_for_sparse-token_classificat.md)
- [\[ICLR 2026\] Tractability via Low Dimensionality: The Parameterized Complexity of Training Quantized Neural Networks](tractability_via_low_dimensionality_the_parameterized_complexity_of_training_qua.md)
- [\[ICLR 2026\] Convergence Analysis of Tsetlin Machines under Noise-Free and Noisy Training Conditions: From 2 Bits to k Bits](convergence_analysis_of_tsetlin_machines_under_noise-free_and_noisy_training_con.md)

</div>

<!-- RELATED:END -->
