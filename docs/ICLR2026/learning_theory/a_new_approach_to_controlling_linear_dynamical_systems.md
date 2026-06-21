---
title: >-
  [论文解读] A New Approach to Controlling Linear Dynamical Systems
description: >-
  [ICLR2026][学习理论][在线非随机控制] 本文提出 Online Spectral Control（OSC）：把对抗扰动下的线性动力系统控制问题，用一组与具体系统无关的「谱滤波器」（某个 Hankel 矩阵的特征向量）做凸松弛，从而在保持 $\tilde O(\gamma^{-4}\sqrt T)$ 最优遗憾的同时，把每步运行时间对稳定裕度 $\gamma$ 的依赖从多项式 $O(\gamma^{-1})$ 降到对数级 $O(\mathrm{polylog}(1/\gamma))$。
tags:
  - "ICLR2026"
  - "学习理论"
  - "在线控制"
  - "在线非随机控制"
  - "谱滤波"
  - "Hankel 矩阵"
  - "遗憾界"
  - "凸松弛"
---

# A New Approach to Controlling Linear Dynamical Systems

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=BQIzu1T6F0](https://openreview.net/forum?id=BQIzu1T6F0)  
**代码**: 待确认  
**领域**: 学习理论 / 在线控制  
**关键词**: 在线非随机控制, 谱滤波, Hankel 矩阵, 遗憾界, 凸松弛

## 一句话总结
本文提出 Online Spectral Control（OSC）：把对抗扰动下的线性动力系统控制问题，用一组与具体系统无关的「谱滤波器」（某个 Hankel 矩阵的特征向量）做凸松弛，从而在保持 $\tilde O(\gamma^{-4}\sqrt T)$ 最优遗憾的同时，把每步运行时间对稳定裕度 $\gamma$ 的依赖从多项式 $O(\gamma^{-1})$ 降到对数级 $O(\mathrm{polylog}(1/\gamma))$。

## 研究背景与动机
**领域现状**：控制对抗扰动下的线性动力系统（LDS）$x_{t+1}=Ax_t+Bu_t+w_t$ 是控制与在线学习交叉处的核心问题。经典的 LQR（线性二次调节器）假设代价是二次的、噪声是 i.i.d. 高斯，这在现代应用里几乎从不成立——风扰、故障、对手都可能让扰动和代价变得结构化、相关甚至对抗。于是「在线非随机控制」（online nonstochastic control）成为更通用的框架：动态 $(A,B)$ 已知且时不变，状态可观测，代价是凸 Lipschitz 但对抗给出，目标是相对「事后最优固定策略」最小化遗憾（regret）。

**现有痛点**：这条线的奠基工作 GPC（Agarwal et al., 2019a）用「扰动-动作映射」（disturbance-action policy，DAC）参数化控制器：把过去若干步扰动 $w_{t-1},\dots,w_{t-m}$ 线性组合成当前控制 $u_t$。问题出在记忆长度 $m$ 上——要让旧扰动的影响衰减到可忽略，$m$ 必须取到 $O(\gamma^{-1}\log T)$，其中 $\gamma$ 是闭环系统的稳定裕度（谱半径 $\rho=1-\gamma$）。参数量和每步运行时间因此**多项式地依赖 $1/\gamma$**。

**核心矛盾**：很多实际场景偏偏喜欢**边际稳定**（marginally stable），即 $\gamma$ 很小、$\rho$ 接近 1。这种系统保留长期记忆、控制更平滑省能（机器人、热力系统、卫星动力学都是这样）。可是 $\gamma$ 越小，GPC 这类方法需要的记忆窗口越长、计算越贵——「想要的物理特性」和「算得起的计算成本」直接打架。

**切入角度**：作者的关键观察是——学习「扰动→控制」的线性映射，本质上是在一个高维但**高度冗余**的特征（原始扰动序列）上做回归。谱滤波（spectral filtering）这套工具最初是为「序列预测」绕开学习 LDS 的非凸性而生的：用某个固定 Hankel 矩阵的顶部特征向量当通用基，能把长记忆压成少数几个分量。能不能把它搬到「控制」上？

**核心 idea**：用「谱滤波器」（一个由 $\gamma$ 决定的 Hankel 矩阵的顶 $h$ 个特征向量）作为**与具体 LDS 无关的通用特征映射**，把控制问题松弛成「在扰动与谱滤波器卷积后的特征上做凸回归」。这样参数量从 $\sim 50d$（记忆 $m$ 乘维度）降到 $hd$（$h$ 只是 polylog 级），遗憾界还更优。

## 方法详解

### 整体框架
OSC（Algorithm 1）是 Projected Online Gradient Descent（投影在线梯度下降）的一个具体实例，但作用在一组精心设计的谱特征上。整条流程是：

1. **离线一次性**计算一个 $m\times m$ Hankel 矩阵 $H$ 的顶 $h$ 个特征对 $\{(\sigma_j,\phi_j)\}_{j=1}^h$。这里 $H$ 的元素只依赖假定的稳定裕度 $\gamma$：
$$H_{ij}=\frac{(1-\gamma)^{i+j-1}}{i+j-1}.$$
这组滤波器是「通用」的——与具体的 $(A,B)$、初始状态、扰动、代价函数都无关。
2. **在线**地维护一组参数矩阵 $M^t_{1:h}\in\mathbb R^{h\times n\times d}$。每一步 $t$：取最近 $m$ 步扰动拼成 $\tilde W_{t-1:t-m}=[w_{t-1},\dots,w_{t-m}]\in\mathbb R^{d\times m}$，计算控制
$$u_t=\sum_{i=1}^{h}\sigma_i^{1/4}\,M^t_i\,\tilde W_{t-1:t-m}\,\phi_i .$$
3. 观测新状态 $x_{t+1}$，**反解**出本步真实扰动 $w_t=x_{t+1}-Ax_t-Bu_t$（这一步显式利用了已知动态 $A,B$）。
4. 对一个无记忆代理损失 $\ell_t$ 做一步梯度下降，再投影回凸约束集 $K$：$M^{t+1}_{1:h}=\Pi_K[M^t_{1:h}-\eta\nabla_M\ell_t(M^t_{1:h})]$。

直观上：GPC 学的是「过去扰动的线性映射」，OSC 学的是「过去扰动**与谱滤波器卷积之后**的线性映射」。后者用更少、更有表达力的特征，把长记忆里真正有用的结构提炼出来。

### 关键设计

**1. 通用谱滤波器：把「与系统相关的长记忆」换成「与系统无关的固定基」**

痛点是 GPC 的记忆窗口 $m$ 必须随 $1/\gamma$ 增长，参数和算力都被拖累。OSC 的破法是：注意到边际稳定 LDS 的状态演化 $x_t=\sum_i (A+BK)^{i-1}w_{t-i}$ 里，闭环矩阵的幂次给响应注入了一种 Hankel 结构。作者构造的 $H_{ij}=(1-\gamma)^{i+j-1}/(i+j-1)$ 恰好编码了「谱半径 $\le 1-\gamma$ 的所有系统」的公共记忆衰减模式。它的顶 $h$ 个特征向量 $\phi_i$ 就是一组**与具体 $(A,B)$ 无关**的通用滤波器：只要系统满足裕度假设，同一组滤波器就够用。$\sigma_i^{1/4}$ 这个四次根权重来自逼近误差分析（控制谱误差在各分量上的累积），让少数滤波器就能覆盖记忆的主要能量。这与序列预测里的谱滤波同源，但作者强调一个**关键技术差异**：预测时过去响应是可得的，而在线控制必须利用系统稳定性、在一组**不同的特征值**上做积分，于是得到的 Hankel 结构（式 2.1）与预测版本不同——这正是把谱滤波从预测搬到控制的难点所在。

**2. 谱控制器类与凸松弛：把非凸的「找最优 K」变成凸的「学 M」**

直接在状态反馈策略 $u_t=Kx_t$ 上优化是非凸的（$K$ 出现在闭环矩阵的幂次里）。作者沿用「改进学习」（improper learning）思路：不直接学 $K$，而是与一个更宽的可凸优化的策略类竞争。具体定义**谱控制器类**
$$\Pi^{SC}_{h,m,\gamma}=\Big\{\pi(w_{t-1:t-m})=\sum_{i=1}^{h}\sigma_i^{1/4} M_i \tilde W_{t-1:t-m}\phi_i\Big\},$$
由参数 $M_{1:h}$ 完全刻画。控制对 $M$ 是**线性**的，所以代价对 $M$ 是凸的——这就把原问题松弛成了凸在线学习。配套地，作者定义有界参数集（Definition 4.2）$K=\{M_{1:h}:\|x^M_t\|,\|u^M_t\|\le 3\kappa^3 W/\gamma,\ \|M_{1:h}\|\le \kappa^3\sqrt{2h/\gamma}\}$，保证投影梯度下降在一个有界凸集上运行、状态/控制不爆。OSC 因此就是 Zinkevich(2003) 的投影在线梯度下降，套在这个凸集与无记忆损失上。

**3. 无记忆损失 + 逼近引理：把「带记忆的真实代价」化简成「可分析的单步损失」**

真实代价 $c_t(x_t,u_t)$ 依赖整段历史（状态由过去所有控制累积而来），是「带记忆」的，标准 OGD 遗憾分析套不进去。作者引入**无记忆损失**（Definition 4.4）$\ell_t(M_{1:h})=c_t(x_t(M_{1:h}),u_t(M_{1:h}))$——假装当前参数从头到尾一直被使用、由此算出的「反事实」状态与控制上的代价。然后用两个引理把缺口补上：(i) **逼近引理**（Lemma 4.3）证明谱控制器类能以任意精度逼近目标策略类 $S$（可对角化稳定策略），即 $\sum_t[c_t(x^M_t,u^M_t)-c_t(x^K_t,u^K_t)]\le \varepsilon T$，只要 $m,h$ 取够大；(ii) **无记忆-真实代价逼近**（Lemma 4.5）证明 $|c_t(x_t,u_t)-\ell_t(M^t_{1:h})|\le 6C_1\sqrt{mh}\,\gamma^{-7/2}T^{-1/2}\log^{1/4}(2/\gamma)$，即用当前参数算的单步损失与真实代价足够接近。两步合起来，OGD 在 $\ell_t$ 上的遗憾就能转回真实代价上的遗憾。

**4. 快速在线卷积：把每步算 $\tilde W\phi_i$ 从 $O(m)$ 摊薄到 $O(\mathrm{polylog})$**

即便参数少了，每步还要算 $h$ 个矩阵乘 $\tilde W_{t-1:t-m}\phi_i$，朴素实现每个要 $O(m)$。作者把 $\phi_i$ 零填充到 $T$ 维、对扰动流 $\{w_t\}$ 做**在线卷积**，并采用 Agarwal et al.(2024a) 的高效在线卷积技术，使内层循环每步的**摊还运行时间**降到 $O(\log^4(T/\gamma))$（Corollary 2.2）。这是把「对数级参数」真正兑现成「对数级算力」的最后一块拼图——否则卷积本身又会把 $m$（含 $1/\gamma$）拖回来。

### 损失函数 / 训练策略
优化器就是投影 OGD，超参由主定理给定（$\gamma$ 为稳定裕度）：记忆 $m=\lceil\frac1\gamma\log(8C_1\sqrt T/\gamma^3)\rceil$、参数数 $h=\lceil 4\log T\log(900C_1dT/\gamma^3)\rceil$（仅 polylog 级）、步长 $\eta=C_2\sqrt{\gamma^3/(Tmh)}$，约束集取 Definition 4.2 的 $K$。实验里则用 Adam（lr $10^{-4}$）并每步投影到单位球。

## 实验关键数据

实验对照对象是 GPC，重点验证「更少参数、相当甚至更好性能」。控制一个状态维 $d=10$、单控制输入 $n=1$ 的 LDS，$A$ 对角元在 $[0.5,0.95]$、$B\sim N(0,1)$，高斯扰动 + 随机生成的固定二次代价，20 次运行取均值。两法都用最近 50 步扰动：GPC 因此有 $50\times d$ 参数，OSC 只用顶 15 个 Hankel 特征向量、即 $15\times d$ 参数。论文报告的是 loss（而非 regret，因为事后最优比较器对单条实现的扰动序列不可观测）。

### 理论对比（Table 1）

| 方法 | 遗憾 | 每步时间 | 扰动 | 代价 |
|------|------|----------|------|------|
| LQR | $O(1)$ | $O(1)$ | i.i.d. | 固定已知二次 |
| Online LQ (Cohen 2018) | $O(\gamma^{-2.5}\sqrt T)$ | $O(1)$ | i.i.d. | 在线二次 |
| GPC (Agarwal 2019a) | $\tilde O(\gamma^{-5.5}\sqrt T)$ | $O(\gamma^{-1}\log T)$ | 对抗 | 在线凸 Lipschitz |
| **OSC（本文）** | $\tilde O(\gamma^{-4}\sqrt T)$ | $O(\log^4(\gamma^{-1}T))$ | 对抗 | 在线凸 Lipschitz |

OSC 是表中唯一在最一般设定（对抗扰动 + 在线凸 Lipschitz 代价）下、同时拿到最好运行时间的方法：遗憾对 $\gamma$ 的指数从 GPC 的 $-5.5$ 改善到 $-4$，每步时间从 $O(\gamma^{-1}\log T)$ 改善到 polylog。

### 经验对比（Figure 2–4）

| 配置 | 现象 | 说明 |
|------|------|------|
| 线性 LDS，无隐层 | OSC 长程 loss 与 GPC 相当 | 参数仅 15d vs 50d |
| 线性 LDS，100 隐单元 MLP 头 | 两法都变好，OSC 仍持平/更优 | 谱特征可作为更强模型的输入 |
| 非线性 LDS ReLU（$x_{t+1}=\mathrm{ReLU}(Ax_t+Bu_t)+w_t$） | OSC 同样有效 | 谱特征在非线性系统上仍管用 |
| 状态维消融 $d\in\{2,5,10,20\}$ | $d$ 增大性能下降（loss 已按 $d$ 归一化） | 单输入控大状态空间本身更难 |
| 参数量消融 $h$ | $h$ 增到约 20 收益递减；$h=5$ 已可与 GPC 竞争、$h=20$ 几乎一致 | 谱特征高度紧凑 |

### 关键发现
- 谱滤波带来的核心收益是「紧凑性」：$h=5$ 个滤波器就能逼平 GPC 的 50 维记忆，印证了「记忆里真正有用的结构维度很低」。
- 谱特征对模型无关：换成 ReLU MLP 头或非线性动态后仍有效，说明它是一个通用特征映射，可接 CNN/Transformer 等更强架构。
- 单输入控制大状态空间是内在困难，与算法无关——$d$ 增大时性能下降是任务本身的难度。

## 亮点与洞察
- **把「序列预测里的谱滤波」迁移到「在线控制」**，且点明关键差异：控制端因为要利用稳定性、在不同特征值集合上积分，Hankel 结构和预测端不同。这种「同一工具、不同结构」的迁移是最漂亮的地方。
- **用「通用、与系统无关的固定基」换掉「随系统裕度增长的长记忆」**：这是把 $1/\gamma$ 依赖从多项式压到对数的根因，思路可迁移到其他长记忆在线学习问题（凡是衰减由谱半径主导的，都可能套这套 Hankel 滤波）。
- **凸松弛三件套（谱控制器类 + 有界约束集 + 无记忆损失）**是把一个非凸带记忆控制问题塞进标准 OGD 框架的范式化做法，值得复用。
- **$\sigma_i^{1/4}$ 这个四次根权重**是个容易被忽略但关键的细节——它让逼近误差在少数滤波器上收敛，是「少参数也够用」的理论保证来源。

## 局限与展望
- 假设较强：动态 $(A,B)$ 已知且时不变、状态全观测、$\gamma$ 已知（作为 Hankel 矩阵的输入）。未知动态、部分观测、时变系统都不在本文范畴。
- 比较器类是「可对角化稳定策略」（Definition 3.4），要求 $L$ 有实非负特征值（作者说可放宽但为简化而保留）；这比一般强稳定略窄。
- 遗憾对 $\gamma$ 的依赖仍是 $\gamma^{-4}$，在极端边际稳定（$\gamma\to0$）时常数项不小；实验规模也偏小（$d\le 20$、单输入），大规模/真实机器人系统的表现待验证。
- 实验报告的是 loss 而非 regret（事后最优不可观测），因此「逼近最优比较器」的程度只能间接看出。

## 相关工作与启发
- **vs GPC (Agarwal et al., 2019a)**：两者都在对抗设定下学「扰动→控制」映射；GPC 直接回归原始扰动、记忆 $m\sim O(\gamma^{-1}\log T)$，OSC 回归「扰动⊗谱滤波器」、参数仅 polylog 级。OSC 遗憾 $\gamma$ 指数从 $-5.5$ 改到 $-4$、每步时间从 $O(\gamma^{-1}\log T)$ 改到 polylog，是直接的严格改进。
- **vs Online LQ (Cohen et al., 2018)**：后者只处理在线二次代价 + i.i.d. 噪声，OSC 处理对抗扰动 + 一般凸 Lipschitz 代价，设定更一般。
- **vs 谱滤波用于预测 (Hazan et al., 2017; Marsden & Hazan, 2025)**：那条线把谱滤波用于序列预测、绕开学 LDS 的非凸性；本文首次把它系统地用于**在线控制**（此前仅 Arora et al., 2018 在离线 LQR 中当黑盒子程序），并处理了控制特有的 Hankel 结构差异。
- **vs Sinkhorn 类加性逼近 / 经典 LQR**：LQR 要求二次代价 + i.i.d. 噪声，遗憾 $O(1)$ 但假设最强；OSC 牺牲一点遗憾换取对抗鲁棒性与对数级算力。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把谱滤波首次系统地引入在线非随机控制，并指出与预测端的 Hankel 结构差异，思路新且漂亮。
- 实验充分度: ⭐⭐⭐ 验证了「少参数、相当性能」与若干消融，但规模偏小（$d\le20$、单输入），仅 loss 无 regret。
- 写作质量: ⭐⭐⭐⭐ 动机清晰、定理与证明路线图组织得当，理论贡献交代到位。
- 价值: ⭐⭐⭐⭐ 对边际稳定系统的对数级算力是实打实的改进，谱特征的通用性也利于迁移到更强模型。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On learning linear dynamical systems in context with attention layers](on_learning_linear_dynamical_systems_in_context_with_attention_layers.md)
- [\[ICLR 2026\] An Optimal Diffusion Approach to Quadratic Rate-Distortion Problems: New Solution and Approximation Methods](an_optimal_diffusion_approach_to_quadratic_rate-distortion_problems_new_solution.md)
- [\[ICML 2026\] A Perturbation Approach to Unconstrained Linear Bandits](../../ICML2026/learning_theory/a_perturbation_approach_to_unconstrained_linear_bandits.md)
- [\[ICLR 2026\] Dynamical properties of dense associative memory](dynamical_properties_of_dense_associative_memory.md)
- [\[ICLR 2026\] Closed-form $\ell_r$ norm scaling with data for overparameterized linear regression and diagonal linear networks under $\ell_p$ bias](closed-form_ell_r_norm_scaling_with_data_for_overparameterized_linear_regression.md)

</div>

<!-- RELATED:END -->
