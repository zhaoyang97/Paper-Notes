---
title: >-
  [论文解读] Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization
description: >-
  [ICLR2026][优化/理论][bilevel optimization] 通过将 F2SA 方法重新解释为前向差分近似 hyper-gradient，提出利用高阶有限差分的 F2SA-p 方法族，在高阶光滑条件下将随机双层优化的 SFO 复杂度从 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 改进至 $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$，并证明了 $\Omega(\epsilon^{-4})$ 下界表明该方法在 $p$ 足够大时近乎最优。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "bilevel optimization"
  - "stochastic optimization"
  - "finite difference"
  - "hyper-gradient estimation"
  - "complexity lower bound"
---

# Faster Gradient Methods for Highly-Smooth Stochastic Bilevel Optimization

**会议**: ICLR2026  
**arXiv**: [2509.02937](https://arxiv.org/abs/2509.02937)  
**代码**: [GitHub](https://github.com/TrueNobility303/F2BA)  
**领域**: 优化  
**关键词**: bilevel optimization, stochastic optimization, finite difference, hyper-gradient estimation, complexity lower bound

## 一句话总结
通过将 F2SA 方法重新解释为前向差分近似 hyper-gradient，提出利用高阶有限差分的 F2SA-p 方法族，在高阶光滑条件下将随机双层优化的 SFO 复杂度从 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 改进至 $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$，并证明了 $\Omega(\epsilon^{-4})$ 下界表明该方法在 $p$ 足够大时近乎最优。

## 背景与动机
双层优化 (bilevel optimization) 在元学习、超参数调优、对抗训练、强化学习等任务中广泛出现，其目标为 $\min_{\bm{x}} f(\bm{x}, \bm{y}^*(\bm{x}))$，其中 $\bm{y}^*(\bm{x}) = \arg\min_{\bm{y}} g(\bm{x}, \bm{y})$。现有方法主要分两类：

1. **基于 Hessian-向量积 (HVP) 的方法**：如 BSA、stocBiO，需要随机 Hessian 预言机，假设更强
2. **纯一阶方法**：如 F2SA，仅需随机梯度预言机，假设更弱，实践中可扩展到 32B 规模 LLM 训练

F2SA 在标准 SGD 假设下取得了 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 的复杂度上界，但与单层优化的 $\Omega(\epsilon^{-4})$ 下界仍有显著差距。问题在于：**纯一阶方法能否在双层优化中达到最优复杂度？**

## 核心问题
在上层函数非凸、下层函数强凸的随机双层优化设定下，能否利用目标函数的高阶光滑性质，突破现有纯一阶方法的 $\tilde{\mathcal{O}}(\epsilon^{-6})$ 复杂度瓶颈，逼近 $\Omega(\epsilon^{-4})$ 的下界？

## 方法详解

### 整体框架
本文先把纯一阶方法 F2SA 重新解读为对 hyper-gradient 的**前向差分近似**，再顺着这个视角换上更高阶的有限差分格式，把近似误差从 $\mathcal{O}(\nu)$ 压到 $\mathcal{O}(\nu^p)$，得到算法族 F2SA-$p$，并配上匹配的 $\Omega(\epsilon^{-4})$ 下界，说明它在高阶光滑区域近乎最优。整套推导围绕一个核心对象展开：扰动下层问题的最优值函数 $\ell_\nu(\bm{x})$ 关于扰动强度 $\nu$ 的导数，恰好等于 hyper-gradient。本文是一篇理论工作，贡献集中在「一个重新解读的视角、一种算法降阶、一处假设放松、一个匹配下界」上，而非一条数据流水线，因此不配框架图。

### 关键设计

**1. 有限差分视角：把 hyper-gradient 估计翻译成数值微分问题**

这是全文的出发点。作者定义扰动下层问题 $g_\nu(\bm{x}, \bm{y}) = \nu f(\bm{x}, \bm{y}) + g(\bm{x}, \bm{y})$ 及其最优值函数 $\ell_\nu(\bm{x})$，并证明 hyper-gradient 可以写成一个对 $\nu$ 的混合偏导 $\frac{\partial^2}{\partial \nu \partial \bm{x}} \ell_\nu(\bm{x})\big|_{\nu=0} = \nabla \varphi(\bm{x})$。在这一视角下，F2SA 惩罚函数的梯度无非是用前向差分 $(\ell_\nu - \ell_0)/\nu$ 去近似 $\partial \ell_\nu / \partial \nu|_{\nu=0}$，因此天然只带一阶误差 $\mathcal{O}(\nu)$。一旦认清这层关系，"如何更准地估计 hyper-gradient"就被翻译成数值分析里"如何更准地做数值微分"这个成熟问题，改进方向也随之浮现——这正是 F2SA 与最优复杂度差距的根源所在。

**2. F2SA-$p$：用高阶有限差分把近似误差从 $\mathcal{O}(\nu)$ 降到 $\mathcal{O}(\nu^p)$**

既然 F2SA 的瓶颈是数值微分只用了最粗的前向差分，换成更高阶的差分格式即可把误差降到 $\mathcal{O}(\nu^p)$（Lemma 3.1）：$p=1$ 对应系数 $\alpha_0=-1,\alpha_1=1$，即原始前向差分 F2SA；$p=2$ 对应中心差分 $\alpha_{-1}=-1/2,\alpha_1=1/2$，即对称惩罚的 F2SA-2。算法是一个双层循环：内层对每个偏移 $j=-p/2,\ldots,p/2$ 跑 $K$ 步 SGD 求解扰动问题 $g_{j\nu}(\bm{x},\cdot)$，得到近似最优解 $\bm{y}_{j\nu}^*(\bm{x})$；外层用差分系数 $\{\alpha_j\}$ 把各扰动点的梯度线性组合成 hyper-gradient 估计 $\Phi_t$，并走归一化梯度步 $\bm{x}_{t+1} = \bm{x}_t - \eta_x \Phi_t / \|\Phi_t\|$（归一化控制了 $\bm{y}_{j\nu}^*(\bm{x}_t)$ 随 $\bm{x}$ 的变化幅度，从而简化内层分析；作者指出标准梯度步用更复杂的分析也能得到同样保证）。复杂度上界借助高维 Faà di Bruno 公式建立 $\ell_\nu$ 关于 $\nu$ 的高阶导数 Lipschitz 性（Lemma 3.2），最终给出 SFO 复杂度 $\tilde{\mathcal{O}}\left(p \kappa^{9+2/p} \epsilon^{-4-2/p}\right)$；当 $p = \Omega(\log\epsilon^{-1}/\log\log\epsilon^{-1})$ 时它简化为 $\tilde{\mathcal{O}}(\kappa^9 \epsilon^{-4})$，与基于 Hessian-向量积 (HVP) 的方法在随机 Hessian 假设下的最优复杂度持平，却完全不需要 Hessian 预言机。值得强调的是 F2SA-2 与 F2SA 同样只需求解两个下层问题，却把误差从一阶降到二阶——这是近乎"免费"的改进，可直接作为 F2SA 的默认替代；偶数 $p$ 因中心差分有 $\alpha_0=0$ 只需 $p$ 个扰动点、奇数 $p$ 需 $p+1$ 个，即便二阶光滑不成立，F2SA-2 也只退化为一阶误差、不会比 F2SA 更差。

**3. 仅需下层方向高阶光滑：把适用假设压到更弱的形式**

误差阶 $\mathcal{O}(\nu^p)$ 要成立，需要 $\ell_\nu(\bm{x})$ 关于 $\nu$ 高阶可导且导数 Lipschitz。作者只对下层变量 $\bm{y}$ 方向要求高阶光滑（Assumption 2.5），即 $f,g$ 关于 $\bm{y}$ 的高阶导数 Lipschitz，这比要求 $(\bm{x},\bm{y})$ 联合高阶光滑要弱得多。这一假设并非空中楼阁：softmax 关于其输入任意阶光滑，因此 data hyper-cleaning、learn-to-regularize 这类基于逻辑回归的超参数调优问题天然满足，使得高阶差分在实际任务中真的可用，而非只停留在纸面。

**4. 匹配下界：构造可分实例约化到单层，证明 $\epsilon^{-4}$ 近乎不可再降**

为说明 $\epsilon^{-4}$ 已逼近极限，作者构造一个完全可分的双层实例（$f(\bm{x},\bm{y}) \equiv f_{\bm{U}}(\bm{x})$，$g(\bm{x},y)=\mu y^2/2$），把双层问题约化为单层问题，从而直接继承 Arjevani et al. (2023) 的 $\Omega(\epsilon^{-4})$ 下界。这一构造的巧妙之处在于它自动满足所有高阶光滑性条件，避开了先前工作（Dagréou et al. 2024、Kwon et al. 2024a）下界构造违反光滑假设的漏洞，因此下界与上界落在同一假设框架内、可直接对照——这也是上界"近乎最优"这一结论站得住脚的关键。

## 实验关键数据
在 20 Newsgroups 数据集上进行 learn-to-regularize 逻辑回归实验（18000 样本，130107 维特征，20 类），该问题天然满足任意阶高阶光滑。

- 比较方法：F2SA-p（$p \in \{2,3,5,8,10\}$）vs F2SA vs stocBiO vs MRBO vs VRBO
- 内循环 $K=10$，外循环 $T=1000$
- F2SA-p 系列随 $p$ 增大收敛速度逐步改善，验证了理论预测
- 附录中在 5 层 ReLU MLP 上的实验表明方法在非光滑非凸问题上也有潜力

## 亮点
1. **优雅的理论洞察**：将 F2SA 重新解释为有限差分，这一视角自然地导出了算法改进方向
2. **几乎免费的改进**：F2SA-2 只需求解 2 个下层问题（与 F2SA 相同），但获得二阶误差保证
3. **完备的理论体系**：上界 $\tilde{\mathcal{O}}(p\epsilon^{-4-2/p})$ + 下界 $\Omega(\epsilon^{-4})$，证明高阶光滑区域近乎最优
4. **更紧的已知界**：$p=1$ 时改进系数 $\kappa$，$p=2$ 时修正了先前工作的 Hessian 收敛界
5. **假设更弱**：仅需 $\bm{y}$ 方向高阶光滑，不需联合高阶光滑或随机 Hessian 假设

## 局限与展望
1. 当 $p$ 较小（特别是 $p=1$）时上下界仍有差距，即使 $p=1$ 的最优复杂度仍是开放问题
2. 条件数 $\kappa$ 的依赖存在 $\Omega(\kappa^9)$ 的上下界差距
3. 实验仅在凸下层问题上验证，非凸-非凸结构化双层优化的推广尚待研究
4. 未与方差缩减或动量技术结合，有进一步加速空间
5. 高阶光滑性假设限制了适用范围，深度网络中通常不满足

## 与相关工作的对比

| 方法 | 光滑阶 | 复杂度 | 需要 HVP |
|------|--------|--------|----------|
| BSA | 1 阶 | $\tilde{\mathcal{O}}(\epsilon^{-6})$ SFO + $\tilde{\mathcal{O}}(\epsilon^{-4})$ HVP | 是 |
| stocBiO | 1 阶 | $\tilde{\mathcal{O}}(\epsilon^{-4})$ | 是 |
| F2SA | 1 阶 | $\tilde{\mathcal{O}}(\kappa^{12}\epsilon^{-6})$ | 否 |
| **F2SA-p** | **$p$ 阶** | **$\tilde{\mathcal{O}}(p\kappa^{9+2/p}\epsilon^{-4-2/p})$** | **否** |
| 下界 | $p$ 阶 | $\Omega(\epsilon^{-4})$ | — |

相比 Chayti & Jaggi (2024) 仅在元学习和对称近似中建立的有限差分联系，本文推广到一般双层优化和任意阶有限差分。相比 Huang et al. (2025) 需要联合高阶光滑，本文仅需 $\bm{y}$ 方向高阶光滑。

## 启发与关联
- **有限差分视角的力量**：将算法动机从惩罚函数转化为有限差分，为双层优化方法设计提供了全新透镜
- **可推广至其他问题**：有限差分改进思路可能适用于其他涉及隐函数定理的优化问题（如 minimax、compositional optimization）
- **实践指导**：F2SA-2 作为默认替代 F2SA 的方案，几乎无额外代价但理论更优

## 评分
- 新颖性: ⭐⭐⭐⭐ — 有限差分视角新颖且自然，算法族设计优雅
- 实验充分度: ⭐⭐⭐ — 实验相对简单，仅凸下层逻辑回归，大规模实验不足
- 写作质量: ⭐⭐⭐⭐⭐ — 论文结构清晰，从洞察到算法到理论的逻辑链条完整
- 价值: ⭐⭐⭐⭐ — 在双层优化理论中建立了重要的新结果，缩小了上下界差距

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reducing Contextual Stochastic Bilevel Optimization via Structured Function Approximation](reducing_contextual_stochastic_bilevel_optimization_via_structured_function_appr.md)
- [\[NeurIPS 2025\] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization](../../NeurIPS2025/optimization/stochastic_momentum_methods_for_non-smooth_non-convex_finite-sum_coupled_composi.md)
- [\[ICML 2026\] SPSsafe: Safeguarded Stochastic Polyak Step Sizes for Non-smooth Optimization](../../ICML2026/optimization/safeguarded_stochastic_polyak_step_sizes_for_non-smooth_optimization_robust_perf.md)
- [\[ICLR 2026\] Bilevel Optimization with Lower-Level Uniform Convexity: Theory and Algorithm](bilevel_optimization_with_lower-level_uniform_convexity_theory_and_algorithm.md)
- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](../../NeurIPS2025/optimization/nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)

</div>

<!-- RELATED:END -->
