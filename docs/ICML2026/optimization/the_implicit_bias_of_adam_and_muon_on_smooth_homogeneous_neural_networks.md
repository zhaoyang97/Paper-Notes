---
title: >-
  [论文解读] The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks
description: >-
  [ICML 2026][优化/理论][隐式偏置] 本文证明：在光滑 $L$-同质模型 + 指数尾损失 + 学习率衰减的设定下，Muon（含 Muon-Signum、Muon-Adam）作为带动量的"归一化最速下降"会收敛到对应范数 max-margin 问题的 KKT 点；Adam（无稳定常数）则收敛到 $\ell_\infty$ max-margin 的 KKT 点，从而把以往仅对线性模型成立的隐式偏置结论一次性提升到所有光滑同质网络。
tags:
  - "ICML 2026"
  - "优化/理论"
  - "隐式偏置"
  - "Adam"
  - "Muon"
  - "同质网络"
  - "最大间隔"
  - "谱范数"
---

# The Implicit Bias of Adam and Muon on Smooth Homogeneous Neural Networks

**会议**: ICML 2026  
**arXiv**: [2602.16340](https://arxiv.org/abs/2602.16340)  
**代码**: 无  
**领域**: 优化理论 / 隐式偏置 / 大模型训练  
**关键词**: 隐式偏置, Adam, Muon, 同质网络, 最大间隔, 谱范数  

## 一句话总结
本文证明：在光滑 $L$-同质模型 + 指数尾损失 + 学习率衰减的设定下，Muon（含 Muon-Signum、Muon-Adam）作为带动量的"归一化最速下降"会收敛到对应范数 max-margin 问题的 KKT 点；Adam（无稳定常数）则收敛到 $\ell_\infty$ max-margin 的 KKT 点，从而把以往仅对线性模型成立的隐式偏置结论一次性提升到所有光滑同质网络。

## 研究背景与动机

**领域现状**：过参数化神经网络的泛化之谜被普遍归因于优化器的"隐式偏置"——即便没有显式正则，梯度类算法也倾向于收敛到某种最大间隔解。从 Soudry 等 (2017) 证明 GD 在线性模型上最大化 $\ell_2$-margin 开始，到 Lyu & Li (2019) 把这一结论搬到任意 $L$-同质（深度 ReLU）网络的 KKT 形式，gradient descent 的隐式偏置图景已基本完整。

**现有痛点**：但工业界训练 LLM / ViT 用的早已不是 GD，而是 Adam、Muon 这类带动量的自适应优化器；它们的隐式偏置只在线性预测器（Zhang 等 2024、Fan 等 2025）上有结果，到了真正的非线性网络就缺乏理论。更尴尬的是，Adam 的早期分析（Wang 等 2021）保留了稳定常数 $\varepsilon$，让 $\sqrt{v_t}$ 渐近被 $\varepsilon$ 主导，实质上把 Adam 退化成 GD——与实践中 $\varepsilon$ 可忽略的常态严重脱节。

**核心矛盾**：动量算法（MSD = momentum steepest descent）的更新方向不再严格满足"最速下降"的代数条件（$\langle\dot{\boldsymbol{\theta}}/\|\dot{\boldsymbol{\theta}}\|, -\boldsymbol{g}/\|\boldsymbol{g}\|_\star\rangle=1$），但动量带来的过冲又只是 $o(1)$ 量级，需要新的工具刻画这种"近似最速下降"。Adam 更复杂：它是 $\hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$ 两个不同衰减率动量的比值，不属于任何 MSD 范式。

**本文目标**：(i) 给 Muon / MSD 在光滑同质模型上的 KKT 收敛证明；(ii) 给无 $\varepsilon$ Adam 同类结果；(iii) 把混合算法（Muon-Signum、Muon-Adam）的隐式范数也写下来；(iv) 用一个统一框架（"近似最速下降"）解释为何动量不破坏 max-margin 偏置。

**切入角度**：在连续时间（流极限）下分析，把动量 $\boldsymbol{m}_t=\int_0^t c_1 e^{-c_1(t-s)}\boldsymbol{g}_s\,ds$ 与瞬时梯度的渐近关系刻成 $\boldsymbol{m}_t[j]=\boldsymbol{g}_t[j](1\pm o(1))$（对"瞬时显著"的坐标 $j$），这样动量就成了梯度的渐近一阶逼近。

**核心 idea**：定义"近似最速下降（Approximate Steepest Descent）"——只要更新方向与负梯度对齐度的下确界渐近 $\ge 1$ 且参数范数被某积分上界控制，就足以推出 max-margin KKT，避开对动量做精确刻画。

## 方法详解

### 整体框架
分析对象是连续时间流，统一形式如下：

- **(Normalized) MSD**：$d\boldsymbol{\theta}_t/dt\in\eta(t)\arg\min_{\|\boldsymbol{u}\|=1}\langle\boldsymbol{u},\boldsymbol{m}_t\rangle$，动量 $\boldsymbol{m}_t$ 按 $d\boldsymbol{m}_t/dt=c_1(\boldsymbol{g}_t-\boldsymbol{m}_t)$ 演化（$c_1\sim -\log\beta_1$）。
- **Muon = 矩阵谱范数下的归一化 MSD**：$\|\cdot\|=\|\cdot\|_{\mathrm{sp}}$ 单层、$\|\cdot\|_{\mathrm{msp}}=\max_k\|W_k\|_{\mathrm{sp}}$ 多层。
- **Adam**（去 $\varepsilon$）：$d\boldsymbol{\theta}_t/dt=-\eta(t)\,\hat{\boldsymbol{m}}_t/\sqrt{\hat{\boldsymbol{v}}_t}$，$\boldsymbol{v}_t$ 按 $d\boldsymbol{v}_t/dt=c_2(\boldsymbol{g}_t^2-\boldsymbol{v}_t)$ 演化。

模型假设 (M1) $f\in C^1$、(M2) $L$-同质 $f(\boldsymbol{x};\alpha\boldsymbol{\theta})=\alpha^L f(\boldsymbol{x};\boldsymbol{\theta})$；损失 $\mathcal{L}(\boldsymbol{\theta})=\sum_i e^{-\varphi(y_i f(\boldsymbol{x}_i;\boldsymbol{\theta}))}$ 涵盖指数与 logistic 两种主流。轨迹假设 (T1) 范数不趋零、(T2) 方向收敛且 margin 正、(A1) Adam 初始有效梯度非零。学习率衰减 (LR-MSD/LR-Adam)：$\int\eta=\infty$ 且 $\eta(t)\le o(t^{1/L-1})$（$L>1$ 时 $\eta(t)=1/t$ 即满足）。

### 关键设计

**1. 近似最速下降框架：用"轨迹是否对齐负梯度"一把抓住一大类优化器的 max-margin 偏置**

动量算法的更新方向不再严格满足最速下降的代数条件，Adam 更是两个不同衰减率动量的比值、根本不属于 MSD 范式，逐个精确刻画太难。作者的破局抽象是"近似最速下降"：只要轨迹 $\boldsymbol{\theta}_t$ 存在某 $\nu(t),R_{\max}$ 使 $N(t)=\int_0^t\nu\to\infty$、$\limsup\|\boldsymbol{\theta}_t\|/N(t)\le R_{\max}$，且对齐度下确界 $\operatorname{ess\,liminf} r(t)\ge 1$（$r(t)=\sup_{\boldsymbol{g}_t}\langle\nu^{-1}\dot{\boldsymbol{\theta}}_t,-\boldsymbol{g}_t/\|\boldsymbol{g}_t\|_\star\rangle$），则在 (T2) + $R_{\max}\le 1$ 下，$\bar{\boldsymbol{\theta}}=\lim\boldsymbol{\theta}_t/\|\boldsymbol{\theta}_t\|$ 必为对应 max-margin 问题的 KKT 点。这个框架的威力在于"接口"取得巧——MSD 取 $\nu=\|\dot{\boldsymbol{\theta}}\|$ 就化归经典最速下降，Adam 虽不是 MSD，但只要把 $\nu=\eta(t)$ 选好同样落入框架，于是 Muon、Adam 能被一并处理，动量过冲、动量比值这些技术细节全被吸进 $r(t)\ge 1$ 这一条对齐度要求里。

**2. 动量与梯度的渐近一致性：证明衰减学习率下动量就是梯度的一阶逼近**

要让动量算法满足上面框架的对齐度条件，得证动量方向渐近等于梯度方向。作者在连续时间下把动量写成 $\boldsymbol{m}_t=\int_0^t c_1 e^{-c_1(t-s)}\boldsymbol{g}_s\,ds$，再证在学习率衰减 $\|\dot{\boldsymbol{\theta}}_t\|\le o(t^{1/L-1})$ 下，对所有"瞬时显著"坐标 $J_\varepsilon(t)=\{j:|\boldsymbol{g}_t[j]|/\|\boldsymbol{g}_t\|_\star>\varepsilon\}$ 都有 $\boldsymbol{m}_t[j]=\boldsymbol{g}_t[j](1\pm o(1))$。核心是附录 B 的一个标量事实：只要 $d\log g/dt$ 收敛，标量动量 $m(t)/g(t)$ 就收敛到良定义极限，而学习率衰减恰好让这个收敛条件沿轨迹自动满足，从而 $\boldsymbol{m}_t/\|\boldsymbol{m}_t\|_\star-\boldsymbol{g}_t/\|\boldsymbol{g}_t\|_\star\to 0$。这步同时为 Adam 准备好了关键近似 $\hat{\boldsymbol{m}}_t[j]/\sqrt{\hat{\boldsymbol{v}}_t[j]}=\mathrm{sign}(\boldsymbol{g}_t[j])(1\pm o(1))$——它正是 Adam $\to$ 符号梯度下降 $\to$ $\ell_\infty$ max-margin 的那座桥。

**3. 复合算法 = 最大范数 MSD：把混合训练配方机械翻译成单一范数下的（近似）MSD**

实战里 Muon 通常只管矩阵参数、对 LayerNorm/bias 改用 Adam 或 sign GD，这种分块混合的隐式间隔目标此前没人写得出来。作者证明：若在参数块 $\boldsymbol{\theta}=(W_1,\dots,W_K,\boldsymbol{u})$ 上并行跑各自归一化的（动量）最速下降、共用同一 $\eta(t)$，则整体等价于以

$$\|\boldsymbol{\theta}\|=\max\{\|(W_1,\dots,W_K)\|_{\mathrm{msp}},\ \|\boldsymbol{u}\|_\infty\}$$

为范数的归一化 MSD（Corollary 3.4）。Muon-Adam 还允许两者基础学习率不同，范数相应变成 $\max\{(\eta_0^A/\eta_0^M)\|(W_1,\dots,W_K)\|_{\mathrm{msp}},\|\boldsymbol{u}\|_\infty\}$（Theorem 3.6）。这条"max-norm 等价"第一次把混合训练的隐式 margin 目标写成解析式，也直接带来一个实用含义：调 $\eta_0^A/\eta_0^M$ 就能在矩阵层 margin 与 bias 层 margin 之间显式 trade-off。

### 损失函数 / 训练策略
分析对象本身是优化器，不重写损失；所有结果在指数/logistic 损失下成立，要求 $\varphi$ 二阶连续可导、严格单调凸且一阶/二阶导数有界（Appendix C.1）。模型类涵盖深度线性网络与含光滑同质激活（如 $\mathrm{ReLU}^q\;(q>1)$、平方激活）的非线性网络，但严格意义上不覆盖普通 ReLU（仅在 (T3) 这一额外的子梯度方向收敛假设下成立）。

## 实验关键数据

### 主实验
2 层（1 隐层）同质网络，$m=2048$ MNIST 数字奇偶分类，logistic 损失，训练至损失 $10^{-8}$；学习率 $\eta(t)=\eta_0 t^{-0.8}$（同时满足 (LR-MSD) 与 (LR-Adam)）；Adam 稳定常数取 $\varepsilon=10^{-20}$ 使其在梯度尺度下可忽略。

| 优化器 | 隐式偏置预测 | 经验上获得最大 margin 的范数 |
|--------|--------------|------------------------------|
| Normalized GD (±momentum) | $\ell_2$ | $\ell_2$ |
| Signum | $\ell_\infty$ | $\ell_\infty$（略优于 Adam） |
| Adam (去 $\varepsilon$) | $\ell_\infty$ | $\ell_\infty$ |
| Muon | $\|\cdot\|_{\mathrm{msp}}$ | $\|\cdot\|_{\mathrm{msp}}$ |
| Muon-Adam | $\max\{(\eta_0^A/\eta_0^M)\|\cdot\|_{\mathrm{msp}},\|\cdot\|_\infty\}$ | 与混合范数一致（Figure 2） |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| squared-ReLU 激活 | 满足 (M1)+(M2)，margin 增长曲线与理论吻合 | 文中主声明的合法测试床 |
| 普通 ReLU 激活 | 经验上同样符合各算法的预测范数 | 提示理论可能可放松到非光滑情形（在 (T3) 之外） |
| 关掉动量（NGD vs MSD 等） | margin 行为几乎无差异（流极限解释） | 验证 Theorem 3.2 中"归一化 vs 未归一化、有无动量"在 KKT 层面等价 |
| 方向收敛诊断（Figure 1b） | 训练后半段 $\langle\boldsymbol{\theta}_t,\boldsymbol{\theta}_{\text{last}}\rangle/\|\cdot\|\|\cdot\|>0.99$ | 经验证 (T2) 在所有算法下都满足 |
| 4 层网络 + CIFAR-10（附录 D） | 趋势与 MNIST 一致 | 验证结论可外推到更深网络 |

### 关键发现
- 算法 → 隐式范数的映射在实验上严格成立：Muon 总是最大化 $\|\cdot\|_{\mathrm{msp}}$ margin，Adam 与 Signum 总是最大化 $\ell_\infty$ margin。
- Signum 在 $\ell_\infty$ margin 上略优于 Adam——与"Adam 仅是符号梯度下降的近似"这一理论解释一致。
- NGD 在 $\|\cdot\|_{\mathrm{msp}}$ 上是次优——因为最后一层若是单行矩阵，其谱范数恰是 $\ell_2$ 范数，NGD 自然部分受益。

## 亮点与洞察
- **"近似最速下降"是个轻巧好用的抽象**：把动量带来的过冲、Adam 中两个动量比值这类复杂技术细节都吸进 $r(t)\ge 1$ 的对齐度条件，未来分析 Lion、Shampoo、Scion 等"非纯净"优化器有现成模板。
- **Muon-Adam 的混合范数公式很实用**：实战中 Muon 与 Adam 学习率比 $\eta_0^A/\eta_0^M$ 会"重新加权"两个范数——这意味着调整这个比值就能在矩阵层 margin 与 bias 层 margin 之间显式 trade-off，可能对 LLM 训练稳定性有直接指导。
- **去 $\varepsilon$ 的 Adam 才是"真 Adam"**：本文用一句话掰回了实践与理论的脱节——稳定常数实测可忽略，理论上必须去掉，否则结论会被 $\varepsilon$ 退化成 GD。

## 局限与展望
- 关键假设 (T2)（方向收敛）只在实验里看到，理论上未证明 Adam / Muon 必然满足；GD 用了多年才被 Ji & Telgarsky (2020) 证完，这里大概率是个长期 open 问题。
- 仅覆盖光滑同质模型；想推到 ReLU 必须依赖 (T3)（子梯度方向收敛），而本文实验在 2 层 ReLU MNIST 上观察到 (T3) 似乎不成立——这意味着 ReLU 网络上 Adam/Muon 的隐式偏置可能本质不同于谱/符号 margin。
- 二分类设定；多分类（特别是 LLM 的下一 token 分类）下结论是否还成立需另证；Fan 等 2025 在线性多分类已有但未扩展到同质网络。
- 未触及 AdamW：本文模型不带显式 weight decay，AdamW 的 max-margin 性质另有 Xie 等 2024 的"约束 $\ell_\infty$ 范数有界"结果，与本文形式上仍有差距。

## 相关工作与启发
- **vs Lyu & Li (2019)**：他们证 GD 在同质网络收敛到 $\ell_2$ max-margin KKT；本文用近似最速下降把这一结论"原样"复制到 Muon、Signum、Adam，区别在于范数从 $\ell_2$ 换成各算法对应的范数。
- **vs Tsilivis 等 (2025)**：他们把 Lyu-Li 推广到任意最速下降；本文 Theorem 3.2 在 MSD 上重做了一遍，并多走一步到 Adam（不属于 MSD 范式）。
- **vs Zhang 等 (2024) / Fan 等 (2025)**：他们在线性模型证 Adam 最大化 $\ell_\infty$、Muon 最大化谱范数；本文把线性模型这一限制升级到光滑同质网络，是真正意义上的"非线性扩展"。
- **vs Xie 等 (2024)**（AdamW）：那篇证 AdamW 的轨迹极限是 $\|\boldsymbol{\theta}\|_\infty$ 有界约束下的 KKT；本文不分析 weight decay，但 $\ell_\infty$ 范数的角色一致，暗示 Adam 系算法的 $\ell_\infty$ 偏置具有普适性。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一次把 Muon 与去 $\varepsilon$ Adam 的隐式偏置推到光滑同质网络，并提出可复用的"近似最速下降"框架。
- 实验充分度: ⭐⭐⭐ MNIST + 一个 CIFAR-10 附录实验，理论文章足够但不算丰富。
- 写作质量: ⭐⭐⭐⭐ 把流极限、动量积分、KKT 三套机器拼装得很清晰，附录的 B/C 章对每个 Lemma 都给了直观说明。
- 价值: ⭐⭐⭐⭐⭐ 对 LLM / Muon 实战的优化器选型提供了"你要哪种 margin"的清晰理论指南，是 2024-2026 这波优化器分析浪潮中最系统的一篇。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] The Rich and the Simple: On the Implicit Bias of Adam and SGD](../../NeurIPS2025/optimization/the_rich_and_the_simple_on_the_implicit_bias_of_adam_and_sgd.md)
- [\[ICLR 2026\] Implicit Bias of Per-sample Adam on Separable Data: Departure from the Full-batch Regime](../../ICLR2026/optimization/implicit_bias_of_per-sample_adam_on_separable_data_departure_from_the_full-batch.md)
- [\[NeurIPS 2025\] Implicit Bias of Spectral Descent and Muon on Multiclass Separable Data](../../NeurIPS2025/optimization/implicit_bias_of_spectral_descent_and_muon_on_multiclass_separable_data.md)
- [\[ICML 2026\] LiMuon: Light and Fast Muon Optimizer for Large Models](limuon_light_and_fast_muon_optimizer_for_large_models.md)
- [\[NeurIPS 2025\] Understanding the Generalization of Stochastic Gradient Adam in Learning Neural Networks](../../NeurIPS2025/optimization/understanding_the_generalization_of_stochastic_gradient_adam_in_learning_neural_.md)

</div>

<!-- RELATED:END -->
