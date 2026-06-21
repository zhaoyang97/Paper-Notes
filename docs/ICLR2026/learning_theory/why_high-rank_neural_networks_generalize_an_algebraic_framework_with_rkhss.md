---
title: >-
  [论文解读] 为什么高秩神经网络也能泛化？：基于 RKHS 的代数框架
description: >-
  [ICLR 2026][学习理论][泛化界] 本文用 Koopman 算子、群表示和再生核希尔伯特空间（RKHS）把深层网络写成"算子乘积"的代数形式，推导出一个新的 Rademacher 复杂度界——其分母里出现权重矩阵的行列式 $\det(W_l^*W_l)^{1/4}$，从而在理论上解释了"高秩、大奇异值的权重矩阵反而泛化得好"这一经验现象，并且首次把这套 Koopman 理论扩展到 tanh、sigmoid、Leaky ReLU 等非光滑激活和有界数据空间上。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "泛化理论"
  - "泛化界"
  - "Rademacher 复杂度"
  - "Koopman 算子"
  - "再生核希尔伯特空间"
  - "群表示"
---

# 为什么高秩神经网络也能泛化？：基于 RKHS 的代数框架

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=nCE7Sli461](https://openreview.net/forum?id=nCE7Sli461)  
**领域**: 学习理论 / 泛化理论  
**关键词**: 泛化界, Rademacher 复杂度, Koopman 算子, 再生核希尔伯特空间, 群表示

## 一句话总结
本文用 Koopman 算子、群表示和再生核希尔伯特空间（RKHS）把深层网络写成"算子乘积"的代数形式，推导出一个新的 Rademacher 复杂度界——其分母里出现权重矩阵的行列式 $\det(W_l^*W_l)^{1/4}$，从而在理论上解释了"高秩、大奇异值的权重矩阵反而泛化得好"这一经验现象，并且首次把这套 Koopman 理论扩展到 tanh、sigmoid、Leaky ReLU 等非光滑激活和有界数据空间上。

## 研究背景与动机

**领域现状**：理解深度网络的泛化能力是机器学习的核心难题。经典的做法是用 VC 维、范数界（norm-based）或压缩界（compression-based）来控制泛化误差。范数界依赖权重矩阵的 $(p,q)$ 范数，压缩界则考察网络能被压缩多少。

**现有痛点**：这两类界都暗示"低秩、小奇异值（近似低秩）的权重矩阵对泛化有利"。但实验上反复观察到一个矛盾现象——**权重矩阵是高秩、且奇异值很大的网络往往也泛化得很好**（Goldblum et al., 2020）。范数界和压缩界只盯着低秩情形，根本无法解释这类现象。

**核心矛盾**：要解释"高秩为何泛化"，需要一个分母里带"行列式/奇异值乘积"的界——奇异值越大，行列式越大，界反而越小。Hashimoto et al. (2024) 提出的 Koopman 界正是这种形式：界正比于 $\prod_l G_l\|K_{\sigma_l}\|_{H_l}\|W_l\|^{s_l-1}/(\sqrt{S}\,\det(W_l^*W_l)^{1/4})$，行列式在分母，所以高秩也能小。但这个界**强依赖模型光滑性和数据空间的无界性**：它把 Koopman 算子的范数定义在 Sobolev 空间 $H_l$ 里，因此把 tanh、sigmoid、ReLU 这类非光滑或作用在有界域上的激活全部排除在外；而且界里的因子 $\|K_{\sigma_l}\|_{H_l}$、$G_l$ 在多数情况下极难评估，激活函数到底如何影响复杂度也说不清。

**本文目标**：保留 Koopman 界"行列式在分母、能解释高秩泛化"的优点，同时把它从光滑无界的理想模型推广到 tanh/sigmoid/Leaky ReLU + 有界数据空间的现实模型，并让界里的每个因子都能被显式计算。

**切入角度**：作者发现，把 Koopman 算子的工作空间从 Sobolev 空间 $H_l$ 换成更大的 $L^2$ 空间 $L_l$（$H_l\subset L_l$），就能同时容纳非光滑激活和有界域；而推导泛化界所需的"再生性"，则可以通过在**参数空间**上额外构造一个 RKHS 来获得。

**核心 idea**：把深层网络代数化为"群表示/Koopman 算子的乘积" $\rho(g_1)A_1\cdots A_{L-1}\rho(g_L)v$，在参数空间上用一个核函数构造 RKHS，借再生性把 Rademacher 复杂度界由一串算子范数控制，从而得到一个适用范围更广、因子可计算的高秩泛化界。

## 方法详解

### 整体框架

本文是一篇纯理论推导：输入是一个 $L$ 层深层网络（一串权重矩阵 + 激活函数），输出是这个网络函数类的 Rademacher 复杂度上界（进而由 Mohri et al. 的定理转成泛化误差界）。整条推导链分四步走：

1. **代数化**：把"线性变换 + 非线性激活的复合"统一写成希尔伯特空间 $H$ 上的算子乘积 $f(g_1,\dots,g_L)=\rho(g_1)A_1\rho(g_2)A_2\cdots A_{L-1}\rho(g_L)v$，其中 $\rho$ 是群 $G$ 的酉表示（描述权重的作用），$A_l$ 取成激活函数对应的 Koopman 算子，$v$ 是末端非线性变换。
2. **正则化逼近**：模型所在的空间 $H$（典型如 $L^2(\mathbb{R}^d)$）本身没有再生性，于是先把模型与一个核 $p_{c,x}$ 做内积，得到正则化模型 $F_c(g,x)=\langle\rho(g_1)A_1\cdots A_{L-1}\rho(g_L)v,\,p_{c,x}\rangle$；当 $c\to\infty$ 时 $p_{c,x}$ 趋于以 $x$ 为中心的 Dirac delta，$F_c$ 收敛回原模型 $f$。
3. **构造参数空间上的 RKHS**：定义核 $k\big((g_1,\dots,g_L),(\tilde g_1,\dots,\tilde g_L)\big)=\langle\rho(g_1)A_1\cdots\rho(g_L)v,\,\rho(\tilde g_1)A_1\cdots\rho(\tilde g_L)v\rangle_H$，它在参数空间 $G^L$ 上张成一个 RKHS $R_k$，并通过等距同构 $\iota$ 把"模型所在的子空间 $K\subseteq H$"与 $R_k$ 对应起来（Proposition 3.4）。这样就能把数据空间上的模型当成参数空间上的函数，享用 RKHS 的再生性。
4. **导界**：用再生性把 Rademacher 复杂度展开成算子范数之积，得到主定理 $\hat R(F_c)\le \|A_1\|\cdots\|A_{L-1}\|\,\|v\|\,E(c)/\sqrt{S}$（Theorem 4.1）。代入具体网络（可逆网络、非常宽度网络、CNN）后，分母里就会自然冒出 $\det(W_l^*W_l)^{1/4}$，于是高秩权重对应小界。

这是矩阵算子层面的机制推导而非数据流水线，因此不配框架图；下面四个关键设计依次对应"代数化 → RKHS 与导界 → $L^2$ 空间替换 → 推广到现实结构"。

### 关键设计

**1. 把网络代数化为算子乘积，用群表示与 Koopman 算子统一线性层与激活**

要让"行列式进分母"，第一步得把网络写成可以拆解范数的算子形式。本文把 $L$ 层网络表达为 $f(g_1,\dots,g_L)=\rho(g_1)A_1\cdots A_{L-1}\rho(g_L)v$，其中两类线性算子各司其职：**群表示** $\rho:G\to B(H)$ 描述权重对模型的作用，例如取仿射群 $G=GL(d)\ltimes\mathbb{R}^d$、$\rho(g)h(x)=|\det W|^{1/2}h(W(x-b))$，就刚好对应一个带可逆权重的（缩放）神经网络——注意 $|\det W|^{1/2}$ 这个因子是后面行列式进分母的来源；**Koopman 算子** $K_\sigma$ 则描述激活函数的复合，定义为 $K_\sigma h(x)=h(\sigma(x))$，它把"非线性复合"线性化成一个线性算子，这正是 Koopman 算子在分析网络时最关键的性质。作者还给出带 Heisenberg 群的新结构作为示范，说明这套代数表示不只复刻已有网络，还能孵化新模型。这一步的意义在于：一旦网络被写成算子乘积，整个函数类的复杂度就有希望被这串算子的范数之积控制。

**2. 在参数空间上构造 RKHS，借再生性把 Rademacher 复杂度界由算子范数控制**

模型所在的 $H$（如 $L^2(\mathbb{R}^d)$）没有再生性，无法直接套 RKHS 导界。本文的巧思是**另起炉灶在参数空间上造一个 RKHS**：用核 $k$（两条参数路径所对应模型函数的内积）张成 $R_k$，并证明 $\iota:K\to R_k$ 是等距同构（Proposition 3.4），从而把数据空间上的模型"搬"到参数空间上当函数用。有了再生性，正则化模型满足 $F_c(\cdot,x)=\iota(p_{c,x})\in R_k$，于是可以把 Rademacher 复杂度逐层剥开，得到主界

$$\hat R(F_c,x_1,\dots,x_S)\le\frac{\|A_1\|\cdots\|A_{L-1}\|\,\|v\|\,E(c)}{\sqrt{S}}.$$

把它落到 Example 3.1 的可逆网络上，由于 $\rho$ 不可约、$A_l$ 可逆（Lemma 3.6 保证 $K=H$，即模型具有"万有性"），界自动满足前提，并化为 $\hat R(\overline{NN}_c)\le E(c)\|v\|\prod_l\|A_l\|/\sqrt{S}\cdot\sup\prod_l|\det W_l|^{-1/2}$。**$\det W_l$ 是 $W_l$ 全部奇异值之积、且位于分母**——所以即使 $W_l$ 奇异值很大、矩阵高秩，界也能很小，这就在数学上回答了标题之问。

**3. 用 $L^2$ 空间替代 Sobolev 空间，让界覆盖非光滑激活与有界数据，并能显式评估激活因子**

已有 Koopman 界把算子范数定义在 Sobolev 空间 $H_l$ 上，这要求激活光滑、数据空间无界，因此排除了 tanh/sigmoid/ReLU 和有界域。本文改在更大的 $L^2$ 空间 $L_l$（满足 $H_l\subset L_l$）上度量算子范数，新界形如 $O\big(\prod_l G_l\|K_{\sigma_l}\|_{L_l}/(\sqrt{S}\det(W_l^*W_l)^{1/4})\big)$。换空间带来两个实打实的好处：其一，**适用范围变大**，非光滑激活和有界数据空间都能分析；其二，**因子可显式计算**——作者给出三条引理把 Koopman 算子范数上界化：Lemma 2.3 用 $\sigma^{-1}$ 的 Jacobian 给出 $\|K_\sigma\|\le\sup_x|J_{\sigma^{-1}}(x)|^{1/2}$；Lemma 2.4 给出 tanh、sigmoid 在有界矩形域上的显式上界；Lemma 2.5 给出 Leaky ReLU 的 $\|K_\sigma\|\le\max\{1,1/a^d\}^{1/2}$。借这些引理，Remark 5.4 进一步说清了激活函数怎样影响复杂度：当 $\sigma$ 像 tanh/sigmoid 那样趋于饱和时，$\det W_l$ 一大、$X_l$ 体积变大就会把 $\|A_l\|$ 顶上去；当 $\sigma$ 像 Leaky ReLU 那样无界时，则是末端变换 $v$ 的范数被顶大。这是已有 Koopman 界完全看不清的。

**4. 推广到非常宽度、非单射权重与卷积网络，用加权 Koopman 算子兜底**

前面假设单一希尔伯特空间 $H$，对应常宽度网络。现实里宽度会变、数据有界，作者于是引入逐层不同的空间 $H_0,\dots,H_{L-1},\tilde H_1,\dots,\tilde H_L$，把权重 $W_l$ 也表示成 Koopman 算子 $\eta_l(W_l)=K_{W_l}$，得到非常宽度网络的界（Theorem 5.1），其中分母换成 $\prod_l|\det W_l^*W_l|^{1/4}$。当 $W$ 不是单射时 $K_W$ 无界，作者改用**加权 Koopman 算子** $\tilde K_{\psi,W}h(x)=\psi(x)h(\sigma(x))$，靠权重 $\psi$（在核空间方向上把贡献置零）把界做出来（Theorem 5.5）；Remark 5.6 给了一个漂亮的解释——如果 $\ker(W_{l+1})$ 恰好对准噪声方向、被 $W_{l+1}$ 滤掉，那么 $\mu_{\ker(W_{l+1})}(Y_l)$ 这个因子就小、模型泛化好，这与 Arora et al. (2018) 的噪声稳定性结论一致。对卷积网络，则把卷积写成在傅里叶分量 $\gamma_m(\theta)$ 上对角化的线性算子、池化写成加权 Koopman 算子，得到 Proposition 5.7。四种结构（可逆/非常宽度/一般非单射/卷积）的界都保持"行列式或傅里叶分量之积在分母"的同一形态。

## 实验关键数据

论文是理论工作，数值实验只用来验证界的有效性（Appendix B 给细节），共三组：

### 主实验

| 实验 | 模型 / 数据 | 设置 | 结论 |
|------|------------|------|------|
| 界的有效性 | 合成回归，$X_0=[-1,1]^3$，tanh，2 层，$S=1000$ | 目标 $t(x)=e^{-\|2x-1\|^2}$，加 $0.1r$ 正则（$r$ 正比于本文界） | 泛化误差随本文界**成比例下降**（Fig 4a 散点随训练变深） |
| 与已有界对比 | MNIST 4 层 dense，smooth Leaky ReLU + softmax，$S=1000$ | 本文正则 $0.01(r_1+r_2+r_3)$ vs Hashimoto 2024 的正则 | 本文正则的**测试精度更高**（Fig 4b） |
| 实用模型 (LeNet) | LeNet on MNIST，tanh + softmax | 加本文正则 $0.1(r_1+r_2+r_3)$ vs 不加 | 加正则**精度更高**；而已有界对 tanh+softmax 根本无效（Fig 4c） |

这里三个正则项分别对应界里的三个因子：$r_1$ 控 $\|A_l\|$（用 Lemma 2.3/2.4 让 $\inf_x(1-x^2)$ 变大）、$r_2$ 控 $1/\det(W_l^*W_l)^{1/2}$（让最小奇异值 $s_{\min}(W_l)$ 变大）、$r_3$ 控 $\|v\|$。

### 与已有 Koopman 界的对比（定性）

| 维度 | Hashimoto 2024（Sobolev $H_l$） | 本文（$L^2$ 空间 $L_l$） |
|------|----------------------------------|--------------------------|
| 适用激活 | 仅光滑、无界 | tanh / sigmoid / Leaky ReLU 等 |
| 数据空间 | 无界 | 可有界 |
| 激活因子是否可评估 | 极难，效果不清 | Lemma 2.3–2.5 给显式上界 |
| 宽度 | 常宽 | 非常宽（Koopman 算子表示权重） |
| 网络结构 | 受限 | 可逆 / 非单射 / 卷积均覆盖 |

### 关键发现
- **行列式进分母是解释高秩泛化的关键**：界正比于 $1/\prod_l\det(W_l^*W_l)^{1/4}$，奇异值越大行列式越大、界越小，这与范数/压缩界"偏好低秩"的结论正好互补。
- **换空间（$H_l\to L_l$）是这篇相对前作的核心增量**：不是降低界的数值，而是把界的**适用域**从理想模型扩到现实模型，同时把原本算不动的激活因子变成可显式上界。
- **激活函数的角色被讲清了**：饱和型激活（tanh/sigmoid）通过放大 $\|A_l\|$ 增加复杂度，无界型激活（Leaky ReLU）通过放大 $\|v\|$ 增加复杂度（Remark 5.4）。
- 三组实验都用"按界构造正则项"间接验证：能依据界改善测试精度，说明界确实抓住了泛化的相关量。

## 亮点与洞察
- **"另起炉灶在参数空间造 RKHS"** 是全文最巧的一步：模型所在的 $L^2$ 空间没有再生性，硬套 RKHS 导界走不通；作者绕到参数空间上用核 $k$ 造 RKHS、再用等距同构 $\iota$ 把两边接上，既拿到再生性又不破坏原空间结构。
- **把"高秩为何泛化"这个经验现象落成一行公式**：分母的 $\det(W^*W)^{1/4}$ 让"大奇异值 → 小界"一目了然，是对范数/压缩界叙事的有力补充。
- **Koopman 算子把非线性复合线性化**的思路可迁移：凡是"函数复合"占核心地位的结构（RNN、归一化流、扩散采样链），都可能套用这套算子代数 + 算子范数控复杂度的框架。
- 用加权 Koopman 算子处理不可逆权重、并把 $\ker(W)$ 解释成"噪声方向被滤掉"，把抽象的算子技巧和"噪声稳定性 → 泛化"的直觉对接上了。

## 局限与展望
- **不覆盖精确 ReLU**：当激活的导数在某段恒为零（如 exact ReLU）时，本框架失效——作者承认这是当前最大缺口，并提示加权 Koopman 算子的变体可能是出路，但留作未来工作。
- **正则常数 $E(c)$ 与逼近精度存在 trade-off**：取 $p_{c,x}=(c/\pi)^{d/2}e^{-c\|y-x\|^2}$ 时 $E(c)=(2c/\pi)^{d/2}$，$c$ 越大 $F_c$ 越逼近原模型、但 $E(c)$ 越大，界越松（Remark 4.3）。
- **实验偏验证性、规模小**：仅合成回归 + MNIST/LeNet，$S=1000$，且只通过"按界正则"间接验证，未在大模型/大数据上检验界的紧度与可用性。
- 界里仍含 $\alpha(f_l)$、$\mu_{\ker}(Y_l)$ 等需要"把 $\tilde X_l$ 设得足够大"才能控制到 1 的辅助因子，实际网络里是否容易满足这些设定尚需更细的考察。

## 相关工作与启发
- **vs 范数界 / 压缩界**（Bartlett 2017, Arora 2018 等）：它们依赖权重的 $(p,q)$ 范数或可压缩性，结论偏好低秩 / 近似低秩；本文界的分母带行列式，专门解释高秩 / 大奇异值情形，二者互补而非替代。
- **vs Hashimoto et al. (2024) 的 Koopman 界**：同样让行列式进分母解释高秩泛化，但前作把范数定义在 Sobolev 空间、只接受光滑无界激活、激活因子算不动；本文换到 $L^2$ 空间，覆盖 tanh/sigmoid/Leaky ReLU、有界数据与非常宽度，并用 Lemma 2.3–2.5 把因子显式化，是对前作适用性的关键扩展。
- **vs 群表示视角的网络分析**（Sonoda et al., 2025）：本文把群表示与 Koopman 算子并用——前者刻画可逆网络，后者刻画更一般（含非单射、卷积）的网络，二者共同支撑起统一的代数表示。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用 RKHS + 群表示 + Koopman 算子把高秩泛化界推广到现实模型，框架原创性强
- 实验充分度: ⭐⭐⭐ 仅合成 + MNIST/LeNet 小规模验证，偏理论佐证
- 写作质量: ⭐⭐⭐⭐ 推导链清晰、引理分工明确，但符号密集、对读者数学背景要求高
- 价值: ⭐⭐⭐⭐ 在理论上回答"高秩为何泛化"并打通到非光滑现实模型，对泛化理论有实质推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Sampling Complexity of TD and PPO in RKHS](sampling_complexity_of_td_and_ppo_in_rkhs.md)
- [\[ICLR 2026\] The Price of Robustness: Stable Classifiers Need Overparameterization](the_price_of_robustness_stable_classifiers_need_overparameterization.md)
- [\[ICLR 2026\] Towards a Theoretical Understanding of In-Context Learning: Stability and Non-i.i.d. Generalisation](towards_a_theoretical_understanding_of_in-context_learning_stability_and_non-iid.md)
- [\[ICLR 2026\] Random Label Prediction Heads for Studying Memorization in Deep Neural Networks](random_label_prediction_heads_for_studying_memorization_in_deep_neural_networks.md)
- [\[ICLR 2026\] Resurfacing the Instance-only Dependent Label Noise Model through Loss Correction](resurfacing_the_instance-only_dependent_label_noise_model_through_loss_correctio.md)

</div>

<!-- RELATED:END -->
