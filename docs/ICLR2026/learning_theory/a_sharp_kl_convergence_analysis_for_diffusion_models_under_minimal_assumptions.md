---
title: >-
  [论文解读] A Sharp KL Convergence Analysis for Diffusion Models under Minimal Assumptions
description: >-
  [ICLR2026][学习理论][扩散模型] 本文为扩散模型（DDPM 采样器）在"只假设 score 估计 L2 准确、不假设任何光滑性"的最小假设下，给出 KL 散度收敛的更尖锐分析：把生成过程建模为"一步概率流 ODE + 一小步加噪"，并发展出一套处理 score 二阶空间导数（Laplacian）的新证明技术，把达到 $\varepsilon^2$-KL 所需迭代数从此前最好的 $\tilde O(d/\varepsilon^2)$ 改进到 $\tilde O(d/\varepsilon)$——在保持维度 $d$ 线性依赖的同时，把对精度 $\varepsilon$ 的依赖从二次降到一次。
tags:
  - "ICLR2026"
  - "学习理论"
  - "扩散模型收敛分析"
  - "扩散模型"
  - "KL 收敛"
  - "概率流 ODE"
  - "随机局部化"
  - "离散化误差"
---

# A Sharp KL Convergence Analysis for Diffusion Models under Minimal Assumptions

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=c8Ft3246KD](https://openreview.net/forum?id=c8Ft3246KD)  
**代码**: 无  
**领域**: 学习理论 / 扩散模型收敛分析  
**关键词**: 扩散模型, KL 收敛, 概率流 ODE, 随机局部化, 离散化误差

## 一句话总结
本文为扩散模型（DDPM 采样器）在"只假设 score 估计 L2 准确、不假设任何光滑性"的最小假设下，给出 KL 散度收敛的更尖锐分析：把生成过程建模为"一步概率流 ODE + 一小步加噪"，并发展出一套处理 score 二阶空间导数（Laplacian）的新证明技术，把达到 $\varepsilon^2$-KL 所需迭代数从此前最好的 $\tilde O(d/\varepsilon^2)$ 改进到 $\tilde O(d/\varepsilon)$——在保持维度 $d$ 线性依赖的同时，把对精度 $\varepsilon$ 的依赖从二次降到一次。

## 研究背景与动机

**领域现状**：扩散模型（score-based generative models）的采样可以走两条等价路径——模拟反向 SDE，或模拟与之共享所有时刻边缘分布的概率流 ODE（probability flow ODE）。近年大量理论工作在分析这套生成过程的收敛速率，关心的是"采样器跑多少步 $K$ 才能让生成分布逼近真实数据分布"。

**现有痛点**：早期分析需要额外的正则性条件（score 光滑、数据有界支撑等）。更现实的一派工作转向"最小假设"——只假设训练得到的 score 估计在 L2 意义下准确。在这个设定下，目前不带任何光滑性假设的最好 KL 保证（Benton et al. 2024，走反向 SDE）是：要让 KL 落到 $\varepsilon^2$ 以内，需要 $\tilde O(d/\varepsilon^2)$ 步——对维度 $d$ 是令人满意的线性，但对精度 $\varepsilon$ 是二次依赖，看起来并不最优。另一派走 ODE 的工作（如 Li & Yan 2024）能拿到 TV 距离 $O(d/\varepsilon)$，但 TV 只是 KL 的平方根上界，把它换算回 KL 反而更弱。

**核心矛盾**：反向 SDE 路径每个区间的离散化误差是 $O(h_k^2)$（$h_k$ 是步长），这是 $\varepsilon$ 二次依赖的根源；而概率流 ODE 路径本可以拿到更好的步长依赖（每区间误差降到 $O(h_k^3)$），但直接用 ODE 聚合误差时会爆炸，过去要么靠 Langevin corrector 加噪平滑（Chen et al. 2023b，需要光滑性假设），要么需要对 score 的 Jacobian/散度加额外假设。换句话说：**想要更好的 $\varepsilon$ 依赖就得走 ODE，但走 ODE 在"无光滑性假设"下又很难把维度 $d$ 压到线性。**

**本文目标**：在只用 score 估计假设的前提下，能否改进 KL 对 $\varepsilon$ 的依赖，同时仍保持对维度 $d$ 的线性依赖？

**切入角度**：借鉴二阶/随机中点离散化的近期工作（Li & Cai 2024、Li & Jiao 2024、Jain et al. 2025）——它们把生成的一个区间拆成"沿反向 ODE 走一步 + 沿前向方向加一小步噪声"。ODE 步用来控制 Wasserstein 型误差（不需要光滑性），加噪步把这个 Wasserstein 误差转换成 KL，且转换时对步长有更好的依赖。

**核心 idea**：用"ODE 步 + 小噪声步"替代反向 SDE/Langevin corrector 来分析 DDPM 采样器，并为这条 ODE 路径专门发展出一套能处理 score 二阶空间导数（Laplacian）的分析框架，从而在无光滑性假设下同时拿到 $d$ 线性和 $\varepsilon$ 线性。

## 方法详解

### 整体框架

这是一篇纯理论分析论文，"方法"指的是收敛证明的算法设定与证明技术，没有新模型或新训练流程。

设定：前向过程取标准 OU 过程 $dx(t) = -x(t)dt + \sqrt{2}\,dw_t$，对应 $x(t) = e^{-t}y + \sqrt{1-e^{-2t}}\,\epsilon$，$y\sim p_{\text{data}}$。反向生成既可用反向 SDE，也可用概率流 ODE $dx(t) = -x(t)dt - s(t,x(t))dt$，其中 $s(t,\cdot)=\nabla\log p_t$ 是 score 函数，训练时只能拿到近似 $\hat s$。

分析对象是作者给出的推断算法（Algorithm 1）：在区间 $[t_{k-1},t_k]$ 上，先用指数积分器（Exponential Integrator）离散化经验概率流 ODE 走"一步半"，再沿前向方向加一小步高斯噪声：
$$\hat x_{k-0.5} = e^{h_k+h_{k-1}}\hat x_k + (e^{h_k+h_{k-1}}-1)\,\hat s(t_k,\hat x_k),\qquad \hat x_{k-1} = e^{-h_{k-1}}\hat x_{k-0.5} + \sqrt{1-e^{-2h_{k-1}}}\,\eta_k.$$
这个"ODE 步 + 加噪步"组合可以解释为反向 SDE 生成过程的一种等价模拟。整条证明链是：先用链式法则把端到端 KL 拆成各区间的条件 KL，再把每个区间的条件 KL 写成 Wasserstein 型误差 $\|x_{k-0.5}-\hat x_{k-0.5}\|^2$，最后把这个误差拆成"离散化误差 $T_d$ + score 估计误差 $T_s$"分别界定。最难的是 $T_d$——它牵出 score 对时间的全导数，进而牵出 score 的二阶空间导数（Laplacian），这是反向 SDE 分析里不会出现的项，也是本文新技术的发力点。

最终主定理（Theorem 3.1）给出的界为：
$$\mathrm{KL}(p_{t_1}\,\|\,\hat p_{t_1}) \lesssim (d+m_2)e^{-T} + d^2 c^3 K + T\varepsilon_{\text{score}}^2,$$
三项分别是初始化误差（用 $N(0,I_d)$ 起步）、离散化误差、score 估计误差，$m_2$ 是数据二阶矩，$c$ 是步长系数。配合指数衰减步长 $h_k = c\min\{t_k,1\}$，整理得到迭代复杂度 $K=\Theta\!\big(d\log^{3/2}(1/\delta)/\varepsilon\big)$ 即可把 KL 压到 $\tilde O(\varepsilon^2)$。

### 关键设计

**1. ODE 步 + 小噪声步：替代反向 SDE / Langevin corrector，换取更优步长依赖**

反向 SDE 路径每区间的离散化误差形如 $\int_{t_{k-1}}^{t_k}\mathbb{E}\|s(t_k,x_k)-s(t,x(t))\|^2 dt$，用 score 的 Jacobian 界定后只能拿到 $O(h_k^2)$ 的步长依赖，这正是 $\varepsilon$ 二次的来源。本文改走概率流 ODE：在区间内先沿确定性 ODE 走一步控制 Wasserstein 型误差（这一步不需要任何光滑性假设），再加一小步前向噪声把 Wasserstein 误差转成 KL。关键好处是 ODE 步的离散化误差是 $O(h_k^3)$（Taylor 积分余项给出三阶），比 SDE 的 $O(h_k^2)$ 更优。与 Chen et al. (2023b) 用 Langevin 动力学做 corrector 不同，本文只是简单地"沿前向方向加噪"，因而绕开了对真/近似 score 光滑性的要求。

**2. 链式分解 + Wasserstein→KL 的逐区间转换**

借助数据处理不等式与链式法则（Lemma A.2），端到端 KL 被拆成初始化项与各区间条件 KL 之和：$\mathrm{KL}(p_{t_1}\|\hat p_{t_1}) \le \mathrm{KL}(p_{t_{K+1}}\|\hat p_{t_{K+1}}) + \sum_k \mathbb{E}\,\mathrm{KL}(p_{t_{k-1}|t_k}\|\hat p_{t_{k-1}|t_k})$。当真过程与生成过程从同一点 $x_k$ 出发时，单区间条件 KL 可精确写成高斯之间的 KL：$\mathrm{KL}(p_{t_{k-1}|t_k}\|\hat p_{t_{k-1}|t_k}) = \frac{e^{-2h_{k-1}}\|x_{k-0.5}-\hat x_{k-0.5}\|_2^2}{2(1-e^{-2h_{k-1}})}$（Lemma A.1）。于是问题归约为界定 Wasserstein 型量 $\mathbb{E}\|x_{k-0.5}-\hat x_{k-0.5}\|^2$，再进一步拆成离散化误差 $T_d$ 和 score 估计误差 $T_s=(e^{h_k+h_{k-1}}-1)^2\mathbb{E}\|s-\hat s\|^2$，后者用假设 2.1 聚合后是 $O(h_k\varepsilon_{\text{score}}^2)$。

**3. Fokker-Planck 把时间导数转成空间导数，正面处理 Laplacian 项**

界定离散化误差 $T_d$ 时，作者先做尺度变换 $z(t)=e^t x(t)$ 简化计算，用 Taylor 积分余项得到 $\mathbb{E}\|z_{k-0.5}-\tilde z_{k-0.5}\|^2 \lesssim (h_k+h_{k-1})^3\int e^{4t}\mathbb{E}\|s_r'(t,z)\|^2 dt$，其中 $s_r'=\frac{d}{dt}s_r$ 是 score 沿轨迹的全导数。这个全导数含对时间的偏导 $\partial_t s_r$，这正是 ODE 路径比 SDE 难的地方。作者用 score 版的 Fokker-Planck 方程（Lemma A.9）把时间导数换成空间导数：$\partial_t s_r(t,z) = e^{2t}\Delta s_r(t,z) + 2e^{2t}\nabla s_r(t,z)^\top s_r(t,z)$，于是
$$\mathbb{E}_{q_t}\|s_r'\|^2 = e^{4t}\,\mathbb{E}_{q_t}\!\big[\|\Delta s_r\|_2^2 + \|\nabla s_r^\top s_r\|_2^2\big] + \mathbb{E}_{q_t}\big[(\Delta s_r)^\top(\nabla s_r^\top s_r)\big].$$
这里出现了 score 的 Laplacian $\Delta s_r$（二阶空间导数）——它在以往的扩散模型分析（无论 SDE 还是一阶 ODE）里都不曾出现，已有技术无法直接界定，是本文必须啃下的硬骨头。

**4. 把随机局部化论证推广到 ODE，建立 score 一阶/二阶导的新恒等式，拿到线性 $d$**

直接的做法（把 Li & Cai 2024 的二阶分析套到 DDPM）虽然改善了 $h_k$ 依赖，但会让目标积分项 $\int\mathbb{E}\|s_r'\|^2 dt$ 出现 $d^3$，进而 KL 得到 $d^{3/2}$，比 Benton et al. (2024) 的 $d$ 更差。要把 $d$ 压回线性，关键观察是 $\mathbb{E}\|s_r\|^2$ 量级为 $O(d/(e^{2t}-1))$，而 $\mathbb{E}\|\nabla s_r\|_F^2$ 量级为 $O(d^2/(e^{2t}-1)^2)$——前者维度更友好。Benton et al. (2024) 在反向 SDE 下靠"等价到随机局部化（stochastic localization）再引用已知结果"拿到线性 $d$，但那只需界定 Jacobian 项 $\mathbb{E}\|\nabla s_r\|_F^2$；ODE 路径却要界定含 Laplacian 的完整全导数项。本文把这套论证推广到 ODE：先证 $\frac{d}{dt}\mathbb{E}_{q_t}[\|s_r\|^2] = -2e^{2t}\mathbb{E}_{q_t}[\|\nabla s_r\|_F^2]$（Lemma A.11），把对 Jacobian 的时间积分换成对 $\|s_r\|^2$ 时间导的积分，从而把 $d^2$ 贡献降到 $d$；再给出对一般幂次 $m$ 的推广 $e^{-2t}\frac{d}{dt}\mathbb{E}_{q_t}[\|s_r\|_2^m]=-m\mathbb{E}_{q_t}[\|s_r\|_2^{m-2}\|\nabla s_r\|_F^2]-\frac{m(m-2)}{4}\mathbb{E}_{q_t}[\|s_r\|_2^{m-4}\|\nabla\|s_r\|_2^2\|_2^2]$（Lemma A.12），并构造一系列新恒等式把 Laplacian 项 $\mathbb{E}_{q_t}\|\Delta s_r\|^2$ 反解为可控量（Lemma A.15–A.17），最终得到 $\mathbb{E}_{q_t}\|s_r'\|^2 \lesssim \frac{d^2 e^{4t}}{(e^{2t}-1)^3} - \cdots$。沿时间积分、跨区间求和、取指数衰减步长后，离散化项是 $\tilde O(d^2/K^2)$，配合主定理给出迭代复杂度的线性 $d$。

### 损失函数 / 训练策略
本文不涉及训练，唯一关于"学习"的假设是 score 估计的 L2 准确性：$\frac{1}{T}\sum_k h_k\,\mathbb{E}_{x\sim p_{t_k}}\|\hat s(t_k,x)-s(t_k,x)\|^2 \le \varepsilon_{\text{score}}^2$（假设 2.1），外加数据分布有限二阶矩 $\mathbb{E}\|x_0\|^2 = m_2 < \infty$（假设 2.2）。没有任何 score 光滑性、Jacobian 有界或数据有界支撑的假设。

## 实验关键数据

本文为纯理论论文，**没有数值实验**，贡献是收敛速率的理论改进。下表汇总在"最小假设（仅 L2 score 估计准确）"下、达到目标精度所需迭代复杂度的对比。

### 收敛速率对比（最小假设设定）

| 工作 | 路径 | 度量 | 迭代复杂度 | 备注 |
|------|------|------|-----------|------|
| Benton et al. 2024 | 反向 SDE | KL ($\le\varepsilon^2$) | $\tilde O(d/\varepsilon^2)$ | 此前 KL 最好结果，$d$ 线性、$\varepsilon$ 二次 |
| Li & Yan 2024 | 反向 SDE | TV ($\le\varepsilon$) | $\tilde O(d/\varepsilon)$ | TV 度量；换算回 KL 更弱 |
| Li et al. 2024a | ODE | TV ($\le\varepsilon$) | $O(d/\varepsilon)$ | 需额外的 Jacobian 假设 |
| 朴素套 Li & Cai 2024 到 DDPM | ODE+加噪 | KL ($\le\varepsilon^2$) | $\tilde O(d^{3/2}/\varepsilon)$ | $\varepsilon$ 改善但 $d$ 退化到 $3/2$ 次 |
| **本文** | ODE+加噪 | KL ($\le\varepsilon^2$) | $\tilde O(d/\varepsilon)$，精确为 $\Theta\!\big(d\log^{3/2}(1/\delta)/\varepsilon_{\text{score}}\big)$ | $d$ 线性、$\varepsilon$ 线性，新 SOTA |

### 关键发现
- **$\varepsilon$ 从二次降到一次**：相比 Benton et al. (2024) 的 $\tilde O(d/\varepsilon^2)$，本文做到 $\tilde O(d/\varepsilon)$，且维度依赖仍是线性，这是无光滑性假设下 KL 收敛的新最优界。
- **走 ODE 不必牺牲维度**：朴素地把已有二阶离散化分析搬到 DDPM 会让 $d$ 退化为 $d^{3/2}$；本文通过新建立的 score 一阶/二阶导恒等式把它救回线性 $d$，说明"ODE 的更优步长依赖"与"线性 $d$"可以兼得。
- **由 KL 自动得到 TV 保证**：由 Pinsker 不等式 $\mathrm{TV}^2\le\frac{1}{2}\mathrm{KL}$，本文的 KL 界同时给出比 Li & Yan (2024) 更强的 TV 收敛保证。

## 亮点与洞察
- **把"时间导数"系统化转成"空间导数"**：用 Fokker-Planck 方程把 $\partial_t s_r$ 改写为 $e^{2t}\Delta s_r + 2e^{2t}\nabla s_r^\top s_r$，是处理概率流 ODE 全导数项的关键一招；这套"全导数→空间导数"的转换思路可迁移到其他确定性采样器（如一致性模型、二阶/随机中点求解器）的收敛分析。
- **把随机局部化论证从 SDE 推广到 ODE**：Benton et al. (2024) 在 SDE 下只需界定 Jacobian 的时间积分；本文识别出 ODE 路径会额外冒出 Laplacian 项，并通过一族"对 $\frac{d}{dt}\mathbb{E}\|s_r\|^m$ 求恒等式"的技巧把高阶导数项逐个降维，是真正的新证明技术，而非已有结果的直接套用。
- **算法侧极简却有效**：在分析中用"一步 ODE + 一小步加噪"替代 Langevin corrector，既保住了 ODE 的步长优势，又用加噪自然完成 Wasserstein→KL 的转换，避开了对 score 光滑性的依赖——这种"用加噪换光滑性假设"的思路本身就很有启发。

## 局限与展望
- **仅是理论上界，无实证验证**：论文没有任何数值实验来佐证 $\tilde O(d/\varepsilon)$ 在实际采样器上的体现，常数因子（隐藏在 $\lesssim$ 与 $\tilde O$ 里）也未量化。
- **依赖早停（early stopping）**：界是相对于前向过程在 $t_1>\delta>0$ 时刻的分布 $p_{t_1}$（即真实数据被方差 $\delta$ 的高斯扰动后的版本），而非数据分布本身，这是为了绕开对数据分布的光滑性假设而付出的代价。
- **$\varepsilon$ 依赖是否还能更优**：作者自己指出，"ODE 步 + 加噪"框架下能否把步长依赖进一步改进、从而把 $\varepsilon$ 依赖压得更低、实现更快收敛，是值得探索的方向。
- **分析绑定特定离散化与 OU 过程**：结果基于指数积分器离散化和标准 OU 前向过程，对其他噪声调度或求解器是否平移，文中未展开。

## 相关工作与启发
- **vs Benton et al. (2024)**：两者都在最小假设下追求线性 $d$ 的 KL 界，但 Benton 走反向 SDE、靠等价到随机局部化引用现成结果，步长依赖只到 $O(h^2)$（$\varepsilon$ 二次）；本文走概率流 ODE，步长依赖到 $O(h^3)$（$\varepsilon$ 一次），代价是要自己处理 SDE 里不出现的 Laplacian 项。本文是其 KL 结果的严格改进。
- **vs Li & Yan (2024)**：Li & Yan 在 ODE/反向过程下给出 TV 距离 $O(d/\varepsilon)$，但 TV 比 KL 弱；本文直接在更强的 KL 上拿到同量级速率，并由 Pinsker 反推出更强的 TV 保证。
- **vs Chen et al. (2023b)（predictor-corrector / Langevin）**：Chen 等用 ODE 步 + Langevin corrector 平滑轨迹，但需要真/近似 score 的光滑性假设；本文用"加一小步前向噪声"替代 Langevin，在无光滑性假设下完成同样的 Wasserstein→KL 转换。
- **vs Li & Cai (2024) / Li & Jiao (2024) / Jain et al. (2025)**：这几篇在二阶离散化、随机中点（带光滑性）或一致性模型设定下提出了"ODE 步 + 加噪"的转换思路；本文继承该思路但落到最小假设的 DDPM 采样器上，并新增了把维度压回线性的关键技术，避免了朴素套用导致的 $d^{3/2}$ 退化。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次在无光滑性假设下把 KL 收敛对 $\varepsilon$ 的依赖从二次降到线性，并发展出处理 score Laplacian 的新证明技术。
- 实验充分度: ⭐⭐⭐ 纯理论论文无数值实验，但收敛速率对比清晰、定位明确（这是理论工作的合理形态）。
- 写作质量: ⭐⭐⭐⭐ 动机层层递进、证明 sketch 把"为什么难、新技术解决了什么"讲得清楚。
- 价值: ⭐⭐⭐⭐ 刷新了扩散模型最小假设下 KL 收敛的 SOTA，对采样理论与确定性求解器分析有方法论价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Polynomial Convergence of Riemannian Diffusion Models](polynomial_convergence_of_riemannian_diffusion_models.md)
- [\[ICLR 2026\] Best-of-N through the Smoothing Lens: KL Divergence and Regret Analysis](best-of-n_through_the_smoothing_lens_kl_divergence_and_regret_analysis.md)
- [\[ICLR 2026\] Finite-Time Convergence Analysis of ODE-based Generative Models for Stochastic Interpolants](finite-time_convergence_analysis_of_ode-based_generative_models_for_stochastic_i.md)
- [\[ICLR 2026\] On the Interpolation Effect of Score Smoothing in Diffusion Models](on_the_interpolation_effect_of_score_smoothing_in_diffusion_models.md)
- [\[ICLR 2026\] Provable Separations between Memorization and Generalization in Diffusion Models](provable_separations_between_memorization_and_generalization_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
