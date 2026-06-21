---
title: >-
  [论文解读] Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime
description: >-
  [ICLR 2026][学习理论][标度律] 这篇论文把两层神经网络（对角网络、二次网络）在带权重衰减训练下的经验风险最小化问题，**精确映射**到 LASSO 和低秩矩阵压缩感知，从而首次在"真正发生特征学习"的设定下解析地刻画出超额风险的完整相图（8 个相、含良性/有害过拟合与插值峰），并进一步把每个标度律相位与训练后权重谱（bulk / spike / 重尾）一一对应，从第一性原理解释了"重尾权重谱 ↔ 更好泛化"这一经验观察。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "神经标度律"
  - "特征学习"
  - "标度律"
  - "权重谱"
  - "压缩感知"
  - "近似消息传递"
---

# Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Q3yLIIkt7z](https://openreview.net/forum?id=Q3yLIIkt7z)  
**代码**: 无  
**领域**: 学习理论 / 神经标度律 / 特征学习  
**关键词**: 标度律, 特征学习, 权重谱, 压缩感知, 近似消息传递

## 一句话总结
这篇论文把两层神经网络（对角网络、二次网络）在带权重衰减训练下的经验风险最小化问题，**精确映射**到 LASSO 和低秩矩阵压缩感知，从而首次在"真正发生特征学习"的设定下解析地刻画出超额风险的完整相图（8 个相、含良性/有害过拟合与插值峰），并进一步把每个标度律相位与训练后权重谱（bulk / spike / 重尾）一一对应，从第一性原理解释了"重尾权重谱 ↔ 更好泛化"这一经验观察。

## 研究背景与动机
**领域现状**：神经标度律（test loss 随样本量 $n$、参数量、算力按幂律下降）是现代深度学习的支柱性经验规律，但其**理论理解几乎全部局限在"惰性 / 随机特征"机制（lazy / random features）**——在这个机制里特征是固定的，网络退化成一个核方法，标度行为由经典的 source & capacity 条件给出（Caponnetto-De Vito、Cui 等）。

**现有痛点**：惰性机制最关键的缺陷是**特征不会被学习**——第一层权重几乎不动。可现实中深度网络的威力恰恰来自特征学习：权重谱会演化出重尾、出现 outlier、发生 rank collapse / bleed-out 等丰富现象（Martin-Mahoney 的经验观察）。在惰性理论里这些现象根本无从谈起，因此现有标度律理论既解释不了"特征学习如何改变标度指数"，也解释不了"权重谱为什么和泛化挂钩"。

**核心矛盾**：要分析特征学习，就必须直面**非凸、高维**的两层网络 ERM 问题，而这类问题通常没有可解析的端到端刻画；而要保留可解析性，以往只能退回到惰性机制。可解析性与"真特征学习"之间存在张力。

**本文目标**：在 teacher-student 设定下，对带权重衰减 $\lambda$、标签噪声 $\Delta$ 的两层网络 ERM（式 1），(1) 给出超额风险 $R$ 关于有效样本量与正则强度的完整标度相图；(2) 刻画训练后权重谱；(3) 把两者联系起来解释谱-泛化关系。

**切入角度**：作者抓住两类特殊但仍能特征学习的两层网络——**对角线性网络**与**二次激活网络**——它们的 ERM 问题存在**精确的数学等价**：对角网络 $\equiv$ LASSO，二次网络 $\equiv$ 核范数正则的低秩矩阵估计（矩阵压缩感知）。这样就能把信号处理领域为 LASSO / 压缩感知积累的强大工具箱（尤其是近似消息传递 AMP 及其状态演化 SE）直接搬过来，得到端到端的精确预测。

**核心 idea**：用"把网络训练等价成稀疏 / 低秩估计"代替"硬解非凸优化"，从而在保留真实特征学习的同时获得超额风险标度指数与权重谱的解析刻画。

## 方法详解

### 整体框架
全文是一套**解析理论**，没有可训练的算法 pipeline，主线是三步：**建立等价映射 → 用 AMP/SE 推导风险与谱 → 把风险分解项对应到谱特征**。

研究对象是两层网络 $f(x;W,a)=a^\top\sigma(Wx+b)$ 在带 $\ell_2$ 权重衰减下的 ERM（式 1）：

$$\min_{W,a}\sum_{\mu=1}^{n}\big(y_\mu-f(x_\mu;W,a)\big)^2+\lambda\big(\|W\|_F^2+\|a\|_2^2\big)$$

数据 $x_\mu\sim\mathcal N(0,I_d)$，标签由同架构 teacher 生成并加方差为 $\Delta$ 的高斯噪声（式 2）。关心的量是超额风险 $R(W,a)=\mathbb E_x[(f(x;W^\star,a^\star)-f(x;W,a))^2]$。

作者考虑两类网络，且都假设 teacher 的谱服从**幂律 / 准稀疏**（power-law / quasi-sparse）衰减——即系数不是严格稀疏而是按重尾 $i^{-\gamma}$（$\gamma>1/2$）衰减。两类网络通过一个统一的**有效样本量** $n_{\rm eff}$ 被放进同一套结果里：

$$n_{\rm eff}=\begin{cases}n & \text{对角网络}\\ n/d & \text{二次网络}\end{cases}$$

这个简单定义让对角与二次两个看似不同的模型呈现出惊人的**普适性**（同一张相图、同一套风险表达式）。

### 关键设计

**1. 两类网络 → 稀疏 / 低秩估计的精确等价：把非凸训练换成有成熟工具的凸问题**

这是全文的地基，针对"两层网络 ERM 非凸、无法解析"这个痛点。对**对角线性网络** $f(x;W,a)=a^\top(w\odot x)/\sqrt d$（$W=\mathrm{diag}(w)$、线性激活、无 bias），令有效权重 $\theta_i=a_iw_i/\sqrt d$，则带 $\ell_2$ 权重衰减的 ERM 精确等价于一个 **LASSO**：

$$\hat\theta=\arg\min_{\theta}\tfrac12\sum_{\mu=1}^{n}\big(y_\mu-\theta^\top x_\mu\big)^2+\lambda\|\theta\|_1$$

虽然这个网络的表达力和线性模型一样，但 $a_iw_i$ 这种**重参数化产生了隐式的 $\ell_1$ 正则**，从而带来特征选择。对**二次网络** $f(x;W,a)=\mathrm{Tr}[S(xx^\top-I_d)/\sqrt d]$（$S=W^\top W/\sqrt{pd}$、二次激活、$a$ 固定为全 1），ERM 精确等价于 **核范数正则的低秩矩阵估计 / 矩阵压缩感知**：

$$\hat S=\arg\min_{S\succeq0}\sum_{\mu=1}^{n}\big(y_\mu-\mathrm{Tr}[SZ_\mu]\big)^2+\lambda\|S\|_*,\quad Z_\mu=\tfrac{x_\mu x_\mu^\top-I_d}{\sqrt d}$$

有了这两个等价，就能把 LASSO / 压缩感知的 **AMP 及其状态演化（state evolution, SE）** 直接用来给出超额风险与谱的精确预测——这正是惰性理论做不到、而本文能做到端到端分析的根本原因。

**2. 超额风险的完整相图：用一个统一表达式覆盖 8 个标度相位**

针对"特征学习如何改变标度指数"这个核心问题，作者在 $n,d\gg1$、$p\ge d$、噪声 $\Delta>0$ 下给出超额风险关于 $(n_{\rm eff},\lambda)$ 的分段精确速率（Result 1，式 11），并据此画出相图（Figure 1）。最具代表性的几个相位：

- **快速衰减 / minimax 相（Phase IV）**：小正则、$1\ll n_{\rm eff}\ll d$ 时 $R=\Theta(n_{\rm eff}^{-1+1/(2\gamma)})$，正好匹配 Raskutti 等的 $\ell_q$-ball minimax 速率；
- **有害过拟合相（Phase V）**：当 $n_{\rm eff}\to d$、欠正则时，估计器开始拟合噪声，风险被非普适尺度 $\rho(n_{\rm eff}/d)$ 主导（对角网络 $\rho(t)=-1/\log t$，二次网络 $\rho(t)=t^{2/5}$）；
- **插值峰**：在 $n_{\rm eff}\sim d$ 附近风险达到最大 $R\sim\lambda^{-2/3}$，这种插值处的非单调正是 **double descent** 的体现，本文把它从线性模型推广到了非线性模型；
- **二次快速衰减（Phase VIa/VIb）**：$n_{\rm eff}\gg d$ 后 $R\propto d/n_{\rm eff}$。

相图里还有若干**速率不连续的相边界**（红线），如 $n_{\rm eff}=\Theta(d)$ 处的有害过拟合↔快速衰减跨越。作为推论，最优正则 $\lambda_{\rm opt}=\tilde\Theta(\sqrt{n_{\rm eff}/d})$ 能避开有害过拟合相、达到 **Bayes 最优速率**（Corollary 1）。关键的普适性在于：对角与二次两个模型在 $n_{\rm eff}$ 的语言下共享同一张相图，包括从良性到有害过拟合的转变。

**3. 权重谱刻画与"软阈值"结构：训练后的权重就是 teacher 谱的加噪软阈值版本**

针对"权重谱为什么这样长"的问题，作者用 SE 给出训练后权重谱的解析形式（Result 2）。核心结论：学到的权重是 teacher 权重的**加噪声 + 软阈值**版本。对角网络：

$$\hat\theta_i\sim\sigma_d(\theta_i^\star+\delta z_i;\,\epsilon),\quad z_i\sim\mathcal N(0,1)$$

其中 $\sigma_d(x;a)=\max(x-a,0)-\max(-x-a,0)$ 是软阈值函数。两个常数有清晰物理意义：$\delta$ 量化来自标签噪声与有限样本估计的**噪声强度**，$\lambda\epsilon$ 给出**截断阈值**——奇异值低于它就被正则压成零。二次网络的谱（式 16）则由零点 Dirac 质量 $\delta_0$、近零的 bulk、以及与 teacher 顶特征向量对齐的少数 outlier 组成。这套结构自然解释了经验观察到的 rank collapse（全谱为零）、heavy-tail（谱是 teacher 的扰动版）、bleed-out（最小 outlier 并入 bulk 边界）等现象。

**4. "普适"误差分解：把超额风险拆成欠拟合 / 过拟合 / 逼近误差三项，并各自对应一类谱特征**

这是把风险和谱真正"焊死"的一步（Result 3），直接服务于"从第一性原理解释谱-泛化关系"的目标。以二次网络欠正则情形为例，超额风险分解为（式 17）：

$$R_{n,d}=\underbrace{\delta^2\!\int(\cdots)\mu_{sc}+\tfrac1d\delta K'(\delta)(2\delta-\lambda\epsilon)^2}_{\text{过拟合（学到的噪声）}}+\underbrace{\tfrac1d\!\!\sum_{i=K(\delta)+1}^{d}\!\!s_i^2}_{\text{欠拟合（没学到的特征）}}+\underbrace{\tfrac1d\!\sum_{i=1}^{K(\delta)}(\cdots)}_{\text{已学特征的逼近误差}}$$

三项各有谱解读：**过拟合项 = bulk 的二阶矩**（即学到的噪声功率，对应 Wigner 半圆律 $\mu_{sc}$）；**欠拟合项 = 被截断掉、藏在 bulk 之下的 spike 功率**（截断点 $K(\delta)$ 由噪声和正则决定）；**逼近误差项 = 已浮出 bulk 的 outlier 上的平均误差**，取决于有效信噪比 $s_i/\delta$ 与有效正则 $\lambda\epsilon$。这个分解之所以叫"普适"，是因为它**不依赖 teacher 谱、样本量或正则**，对所有 $\Delta\ge0$ 和所有谱相位都成立。

由此得到对谱-泛化关系的第一性原理解释：**bulk = 学到的噪声、藏在 bulk 下的 spike = 没学到的特征、outlier = 学到的特征**。最优正则策略也随之清晰——只截断 bulk（把过拟合第一项压零），尽量少动后两项。更重要的是，在 $d\ll n_{\rm eff}\ll d^2$ 区间，最优性能出现在 Phase II（$\lambda=\sqrt{n_{\rm eff}/d}$），其谱恰好从"outlier 主导"过渡到"重尾"，从理论上**支持了 Martin 等"重尾谱关联更好泛化"的经验论断**。

### 损失函数 / 训练策略
没有专门的损失设计——目标就是式 (1) 的带 $\ell_2$ 权重衰减的平方误差 ERM。数值实验中网络用 PyTorch + LBFGS 训练到全局最小（Appendix F）。理论侧用 AMP 的状态演化方程求解；论文的一个关键技术主张是：SE 方程严格成立只在比例渐近（$n_{\rm eff}/d$ 固定、$\lambda$ 固定）下，但作者**启发式地把它外推到 $n,d,\lambda$ 任意标度**，并用大量数值验证其在远超已证明范围处仍精确到常数。

## 实验关键数据
本文是理论论文，"实验"是用数值模拟验证解析预测（不是和别的方法刷点）。

### 主验证：风险标度律 vs 状态演化
Figure 3 把对角 / 二次网络的真实训练超额风险（PyTorch+LBFGS，$d=100,200,400,800$）与非渐近 SE 预测（实线）对比：

| 设定 | 正则 $\lambda$ | 对比对象 | 结论 |
|------|------|----------|------|
| 二次网络 | $1/d,\,1,\,\sqrt d$ | 模拟点 vs SE 实线 | 跨整个 $n_{\rm eff}$ 范围吻合极好 |
| 对角网络 | $1/d,\,1,\,\sqrt d$ | 模拟点 vs SE 实线 | 同样吻合，且匹配 Result 1 预测的黑色速率线 |

尽管 SE 只在 $n_{\rm eff}/d=\Theta(1)$ 的渐近极限严格成立，模拟显示它在远超该范围处仍精确到常数。

### 谱验证
Figure 2 把训练后权重的特征值直方图（蓝）与理论 bulk（紫）/ spike（橙）预测对比，覆盖 Ia / IV / VIa（$\lambda=1/d$）与 Ib / II / III（$\lambda=\sqrt d$）等相位（$d=800$，Phase III 用 $d=400$）。理论曲线准确捕捉了各相位 bulk 形状与 outlier 位置。

### 关键发现
- **普适性**：对角网络与二次网络在 $n_{\rm eff}$ 语言下共享同一张相图，含从良性到有害过拟合的同一种转变——这是作者认为最值得继续深挖的现象。
- **非单调风险来自谱演化**：随样本增加，bulk 不断收缩、spike 不断"浮出"，但 bulk 二阶矩在插值附近反而增大，导致风险非单调（double descent 的谱级解释）。
- **谱-泛化的因果链被打通**：重尾谱之所以关联好泛化，是因为它对应 outlier 主导→重尾的过渡，恰好落在最优正则 Phase II。
- **SE 的稳健外推**：AMP/SE 在远超其已证渐近假设处仍有预测力，作者据此提出更广的猜想——自旋玻璃理论工具可能在标准渐近假设之外仍然有效。

## 亮点与洞察
- **"网络训练 = 稀疏/低秩估计"的精确等价**是最漂亮的一招：它不是近似，而是恒等映射，因此能把 LASSO / 压缩感知三十年的工具（AMP、SE、minimax 界）原封不动搬到神经网络标度律上。
- **统一有效样本量 $n_{\rm eff}=n$ 或 $n/d$** 让两类形态迥异的网络坍缩到同一套公式，这种"普适性"提示标度律可能由问题的稀疏/低秩结构而非具体架构决定。
- **谱三分法（bulk=噪声 / 隐藏 spike=未学特征 / outlier=已学特征）** 极具迁移价值：它给"读权重谱判断网络学得好不好"提供了第一性原理依据，可启发对大模型权重谱诊断的理论化。
- 把 double descent、benign/harmful overfitting、heavy-tail 谱这几个原本分散的经验现象，**用同一张相图串了起来**。

## 局限与展望
- 只分析了**全局最小**，没有覆盖 GD/SGD 的训练动力学与算力标度律（作者明确指出这是后续方向）。
- 仅限**两层网络 + 二次/线性激活 + 各向同性高斯数据**；更深网络、一般激活、非平凡协方差结构尚未覆盖。
- 核心结果依赖 **SE 在比例渐近外仍成立的启发式外推**——虽有大量数值支撑，但缺严格证明，作者把"严格化 SE 猜想"列为未来工作。
- teacher-student + 幂律谱是理想化设定，与真实数据的标度律之间还有距离；普适性能否延伸到这两个模型之外也仍是开放问题。

## 相关工作与启发
- **vs 随机特征 / 核机制标度律（Bahri、Maloney、Atanasov、Bordelon 等）**：他们在特征固定的惰性机制下分析标度律，问题退化为核方法；本文跳出惰性、直面真特征学习，能解释惰性理论无法触及的权重谱演化与谱-泛化关系。
- **vs Ben Arous et al. (2025)**：同样研究二次网络，但他们聚焦无噪声、无正则下特定 SGD 动力学，只得到本文众多标度指数中的一个；本文系统覆盖任意 $\lambda>0$、$\Delta\ge0$ 的整张相图。
- **vs Ren et al. (2025)**：他们研究大 information exponent 的激活函数，方向与本文正交。
- **vs 经典 LASSO/矩阵压缩感知（Raskutti & Wainwright、Negahban & Wainwright）**：本文不仅复现其在特定 $\lambda$ 下的 minimax 速率，还把结论扩展到所有正则强度与数据规模，给出完整相图，并揭示 minimax 速率与更快 $\Theta(d/n_{\rm eff})$ 速率之间的跨越。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在真特征学习设定下给出两层网络标度律完整相图，并把风险与权重谱从第一性原理焊在一起。
- 实验充分度: ⭐⭐⭐⭐ 理论论文，数值验证覆盖多相位、多 $d$、风险与谱两条线，吻合度高；但仅限两个 toy 架构。
- 写作质量: ⭐⭐⭐⭐ 逻辑链清晰（等价→相图→谱→分解），但相图与公式较稠密，门槛偏高。
- 价值: ⭐⭐⭐⭐⭐ 为神经标度律与"重尾谱↔泛化"提供了罕见的可解析理论基座，对理论与谱诊断实践都有启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Implicit bias produces neural scaling laws in learning curves, from perceptrons to deep networks](implicit_bias_produces_neural_scaling_laws_in_learning_curves_from_perceptrons_t.md)
- [\[ICLR 2026\] Mitigating the Curse of Detail: Scaling Arguments for Feature Learning and Sample Complexity](mitigating_the_curse_of_detail_scaling_arguments_for_feature_learning_and_sample.md)
- [\[ICLR 2026\] Transfer Learning in Infinite Width Feature Learning Networks](transfer_learning_in_infinite_width_feature_learning_networks.md)
- [\[ICLR 2026\] Feature Compression is the Root Cause of Adversarial Fragility in Neural Networks](feature_compression_is_the_root_cause_of_adversarial_fragility_in_neural_network.md)
- [\[ICLR 2026\] Theory of Scaling Laws for In-Context Regression: Depth, Width, Context and Time](theory_of_scaling_laws_for_in-context_regression_depth_width_context_and_time.md)

</div>

<!-- RELATED:END -->
