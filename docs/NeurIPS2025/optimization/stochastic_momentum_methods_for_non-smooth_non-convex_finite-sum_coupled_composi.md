---
title: >-
  [论文解读] Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization
description: >-
  [NeurIPS2025][优化/理论][FCCO] 针对非光滑非凸有限和耦合复合优化 (FCCO) 问题，提出两种随机动量方法 SONEX（单循环）和 ALEXR2（双循环），通过外层 Moreau 包络平滑和嵌套平滑技术将迭代复杂度从 $O(1/\epsilon^6)$ 改进至 $O(1/\epsilon^5)$，并在非凸不等式约束优化中取得同等最优复杂度。
tags:
  - "NeurIPS2025"
  - "优化/理论"
  - "FCCO"
  - "非光滑非凸优化"
  - "随机动量法"
  - "Moreau 包络"
  - "约束优化"
  - "复合优化"
---

# Stochastic Momentum Methods for Non-smooth Non-Convex Finite-Sum Coupled Compositional Optimization

**会议**: NeurIPS2025  
**arXiv**: [2506.02504](https://arxiv.org/abs/2506.02504)  
**作者**: Xingyu Chen, Bokun Wang, Ming Yang, Qihang Lin, Tianbao Yang (Texas A&M, Iowa)
**代码**: 待确认  
**领域**: 优化  
**关键词**: FCCO, 非光滑非凸优化, 随机动量法, Moreau 包络, 约束优化, 复合优化

## 一句话总结
针对非光滑非凸有限和耦合复合优化 (FCCO) 问题，提出两种随机动量方法 SONEX（单循环）和 ALEXR2（双循环），通过外层 Moreau 包络平滑和嵌套平滑技术将迭代复杂度从 $O(1/\epsilon^6)$ 改进至 $O(1/\epsilon^5)$，并在非凸不等式约束优化中取得同等最优复杂度。

## 研究背景与动机

有限和耦合复合优化 (FCCO) 问题形如：
$$\min_{\mathbf{w}} F(\mathbf{w}) := \frac{1}{n}\sum_{i=1}^n f_i(g_i(\mathbf{w}))$$

其中 $f_i$ 为外层函数（可能非光滑），$g_i$ 为内层函数（随机估计），广泛应用于 X-risk 优化（AUC、对比损失）、组分布鲁棒优化 (GDRO) 和非凸约束优化等任务。

现有方法存在两个关键局限：
- **复杂度过高**：先前最优方法 SONX 在 stochastic 内函数均值 Lipschitz 连续假设下需 $O(1/\epsilon^6)$ 迭代
- **不适配深度学习**：SONX 依赖 vanilla SGD 更新，缺乏动量机制，不适合训练深度神经网络

本文旨在同时解决这两个问题：设计可证明收敛的随机动量方法，并将复杂度降低一个量级。

## 方法详解

### 核心思路：外层平滑 + 嵌套平滑

**外层平滑 (Outer Smoothing)**：对非光滑外层函数 $f_i$ 使用 Moreau 包络平滑：
$$f_{i,\lambda}(\mathbf{u}) := \min_{\mathbf{v}} f_i(\mathbf{v}) + \frac{1}{2\lambda}\|\mathbf{u} - \mathbf{v}\|^2$$

将原问题转化为光滑复合优化 $\min_{\mathbf{w}} F_\lambda(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n f_{i,\lambda}(g_i(\mathbf{w}))$。当 $\lambda = \epsilon / C_f$ 时，$F_\lambda$ 的 $\epsilon$-稳定点即为原问题的近似 $\epsilon$-稳定解（Theorem 4.2）。

**嵌套平滑 (Nested Smoothing)**：当内层函数 $g_i$ 也非光滑（弱凸）时，对 $F_\lambda$ 再做一层 Moreau 包络平滑，将问题转化为 $\min_{\mathbf{w}} F_{\lambda,\nu}(\mathbf{w})$，使之可被动量方法处理。

### 算法 1: SONEX（单循环，光滑内函数）

适用条件：$f_i$ 弱凸非光滑或凸，$g_i$ 光滑。

关键步骤：
1. **MSVR 跟踪**：维护 $n$ 个序列 $u_{i,t}$ 跟踪 $g_i(\mathbf{w}_t)$，采用坐标式更新（仅被采样的 $i \in \mathcal{B}_1^t$ 更新）
2. **梯度估计**：$G_t = \frac{1}{|\mathcal{B}_1^t|}\sum_{i \in \mathcal{B}_1^t} \nabla f_{i,\lambda}(u_{t,i}) \nabla g_i(\mathbf{w}_t, \mathcal{B}_{i,2}^t)$
3. **动量更新**：$\mathbf{v}_{t+1} = (1-\beta)\mathbf{v}_t + \beta G_t$，$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}$（或 Adam 更新）

复杂度：$O\left(\frac{n}{B_1\sqrt{B_2}} \epsilon^{-5}\right)$

### 算法 2: ALEXR2（双循环，光滑或弱凸内函数）

适用条件：$f_i$ 凸，$g_i$ 光滑或弱凸（弱凸时需 $f_i$ 单调非减）。

将 $F_\lambda$ 改写为 minimax 形式后，通过嵌套 Moreau 平滑得到光滑目标 $F_{\lambda,\nu}$：

关键步骤：
1. **外循环**：动量更新 $\mathbf{w}$，用近似梯度 $G_t = \frac{\beta}{\nu}(\mathbf{w}_t - \mathbf{z}_{t,K_t})$
2. **内循环**：运行 ALEXR 求解强凸-强凹 minimax 子问题，得到近端映射的近似解 $\hat{\mathbf{z}}_t$
3. 支持 Adam 或动量更新

复杂度：$\tilde{O}\left(\frac{n}{B_1 B_2 \epsilon^5} + \frac{1}{\epsilon^5}\right)$

### 约束优化应用

对不等式约束优化 $\min g_0(\mathbf{w})$ s.t. $g_i(\mathbf{w}) \leq 0$，采用平滑铰链罚函数：
$$\Phi_\lambda(\mathbf{w}) = g_0(\mathbf{w}) + \frac{1}{m}\sum_{i=1}^m f_\lambda(g_i(\mathbf{w}))$$

其中 $f_\lambda$ 为铰链函数 $\rho[\cdot]_+$ 的 Moreau 平滑版本。用 SONEX/ALEXR2 优化可达 $O(1/\epsilon^5)$ 找近似 $\epsilon$-KKT 解，且对 $\delta$ 依赖从 $O(\delta^{-6})$ 改进至 $O(\delta^{-4})$。

## 实验关键数据

### 实验1: 组分布鲁棒优化 (GDRO with CVaR)

在 Camelyon17 (30组, DenseNet121)、Amazon (1252组, DistilBERT)、CelebA (16组, ResNet50) 上比较，CVaR ratio $r=0.15$：

| 方法 | 更新类型 | Camelyon17 | Amazon | CelebA 损失 | CelebA 测试准确率 |
|------|----------|------------|--------|-------------|-----------------|
| OOA | SGD | 收敛较慢 | 收敛较慢 | 收敛较慢 | 较低 |
| SONX | SGD | 中等 | 中等 | 中等 | 中等 |
| **SONEX** | **Momentum/Adam** | **最快收敛** | **最快收敛** | **最快收敛** | **最高** |

SONEX 在所有数据集上训练损失下降最快，CelebA 测试精度最优。Amazon (1252组) 上使用 Adam 更新，优势尤为明显。

### 实验2: AUC 最大化 + ROC 公平性约束

在 Adult 和 COMPAS 数据集上，14 个公平性约束（FPR/TPR 差距容忍度 $\kappa=0.005$，阈值 $\Gamma=\{-3,...,3\}$），2 层隐藏层网络：

| 方法 | 罚函数 | Adult AUC | COMPAS AUC | 约束满足 |
|------|--------|-----------|------------|---------|
| SOX | 平方铰链 | 较低 | 较低 | 可接受 |
| SONX | 铰链 | 中等 | 中等 | 可接受 |
| ICPPAC | 双循环 | 中等 | 中等 | 可接受 |
| **ALEXR2** | **平滑铰链** | **最高** | **最高** | **相当** |

ALEXR2 在保持约束满足度相当的前提下，AUC 最优化效果最佳。

### 实验3: 持续学习 + 非遗忘约束

在 BDD100K 上微调 CLIP 模型（目标任务如雾天/阴天分类，约束为对其他天气类别不遗忘）：
- SONEX 的 Target $\Delta$ACC 显著高于 SOX（平方铰链）
- 约束违反程度相当
- SONX (SGD) 在 Transformer 架构上完全失败，无法训练

## 亮点

- **理论突破**：将非光滑非凸 FCCO 的最优复杂度从 $O(\epsilon^{-6})$ 提升至 $O(\epsilon^{-5})$，是该问题的首个随机动量方法
- **双重平滑技术**：外层 Moreau 包络平滑 + 嵌套平滑，巧妙处理外层和内层函数的双重非光滑性
- **实用性强**：支持 Adam 更新，可直接用于深度学习；在 Transformer/CLIP 等模型上有效（SONX 的 SGD 在这些架构上失败）
- **约束优化新结果**：复杂度和对正则化常数 $\delta$ 的依赖均取得改进
- **统一框架**：SONEX 和 ALEXR2 分别覆盖光滑和弱凸内函数场景，适用于 GDRO、AUC 公平性、持续学习等多种应用

## 局限性

- 单循环 SONEX 在内层批量大小 $B_2$ 的依赖上（$1/\sqrt{B_2}$）弱于双循环 ALEXR2（$1/B_2$），作者明确指出此为待改进点
- SONEX 需要均值 Lipschitz 连续 (MLC0) 假设，ALEXR2 则要求外层函数凸性，两者各有限制
- Assumption 4.3（内函数 Jacobian 的最小奇异值下界）虽有实验验证但并非对所有问题成立
- 实验规模有限：仅在较小分类任务上验证，缺乏大规模深度学习基准实验
- 超参数设置依赖理论推导的精确常数，实践中可能需额外调参

## 相关工作

- **FCCO 方法**：SOX、MSVR/STORM、SONX 均要求光滑 $f_i$ 或只能 SGD 更新；本文首次处理非光滑 + 动量
- **平滑技术**：Nesterov 平滑、Moreau 包络已广泛用于弱凸问题；本文推广至 FCCO 的耦合结构
- **非凸约束优化**：OSS/ICPPAC ($O(\epsilon^{-6})$)、SSG ($O(\epsilon^{-8})$)、Li et al. ($O(\epsilon^{-7})$)；本文 $O(\epsilon^{-5})$ 在光滑和弱凸场景下均为最优
- **X-risk 优化**：AUROC、对比损失等目标可建模为 FCCO，本文方法直接适用

## 评分
- 新颖性: ⭐⭐⭐⭐ — 首个非光滑 FCCO 的随机动量方法，外层平滑 + 嵌套平滑的设计思路清晰有效
- 实验充分度: ⭐⭐⭐ — 覆盖 GDRO、公平性约束、持续学习三类任务，但规模偏小且缺少更多基线对比
- 写作质量: ⭐⭐⭐⭐ — 理论推导严谨，符号系统清晰，但内容密度极高，非优化方向读者门槛较高
- 价值: ⭐⭐⭐⭐ — 复杂度改进有坚实的理论意义，动量方法的引入对实际深度学习应用有直接帮助

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] A Near-Optimal Single-Loop Stochastic Algorithm for Convex Finite-Sum Coupled Compositional Optimization](../../ICML2025/optimization/a_near-optimal_single-loop_stochastic_algorithm_for_convex_finite-sum_coupled_co.md)
- [\[NeurIPS 2025\] Non-Stationary Bandit Convex Optimization: A Comprehensive Study](non-stationary_bandit_convex_optimization_a_comprehensive_study.md)
- [\[ICLR 2026\] High Probability Bounds for Non-Convex Stochastic Optimization with Momentum](../../ICLR2026/optimization/high_probability_bounds_for_non-convex_stochastic_optimization_with_momentum.md)
- [\[NeurIPS 2025\] Nonlinearly Preconditioned Gradient Methods: Momentum and Stochastic Analysis](nonlinearly_preconditioned_gradient_methods_momentum_and_stochastic_analysis.md)
- [\[NeurIPS 2025\] Isotropic Noise in Stochastic and Quantum Convex Optimization](isotropic_noise_in_stochastic_and_quantum_convex_optimization.md)

</div>

<!-- RELATED:END -->
