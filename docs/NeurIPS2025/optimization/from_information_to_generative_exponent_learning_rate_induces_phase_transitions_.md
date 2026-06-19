---
title: >-
  [论文解读] From Information to Generative Exponent: Learning Rate Induces Phase Transitions in SGD
description: >-
  [NeurIPS 2025][优化/理论][单指标模型] 系统刻画了在学习高斯单指标模型时，学习率如何在"information exponent 主导"和"generative exponent 主导"两个样本复杂度体制之间引发相变，并提出了一种新的逐层交替 SGD 算法，无需复用样本即可突破 CSQ 下界。
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "单指标模型"
  - "学习率相变"
  - "信息指数"
  - "生成指数"
  - "SGD 样本复杂度"
---

# From Information to Generative Exponent: Learning Rate Induces Phase Transitions in SGD

**会议**: NeurIPS 2025  
**arXiv**: [2510.21020](https://arxiv.org/abs/2510.21020)  
**代码**: 无  
**领域**: 优化  
**关键词**: 单指标模型, 学习率相变, 信息指数, 生成指数, SGD 样本复杂度

## 一句话总结
系统刻画了在学习高斯单指标模型时，学习率如何在"information exponent 主导"和"generative exponent 主导"两个样本复杂度体制之间引发相变，并提出了一种新的逐层交替 SGD 算法，无需复用样本即可突破 CSQ 下界。

## 研究背景与动机

**领域现状**：学习单指标模型 $y = \sigma_*(⟨\mathbf{x}, \boldsymbol{\theta}_*⟩) + \zeta$（$\mathbf{x} \sim \mathcal{N}(0, I_d)$）的样本复杂度由链接函数属性决定。在线 SGD 的复杂度由 information exponent $p = \text{IE}(\sigma_*)$（第一个非零 Hermite 系数的阶数）主导，为 $\tilde{\Theta}(d^{(p-1)\vee 1})$。

**现有痛点**：
   - 通过多次梯度步（batch reuse）可将复杂度降到由 generative exponent $p_* = \text{GE}(\sigma_*) \leq p$ 主导，但这一图景仅在学习率足够大时成立
   - Full-batch gradient flow 的最佳已知界仍依赖 information exponent，暗示学习率的作用被忽视
   - 此前工作未系统研究学习率选择如何影响 CSQ/SQ 体制的切换

**核心矛盾**：batch reuse 直觉上应突破 CSQ 限制，但 full-batch gradient flow（极端复用）的界却未改善——说明学习率而非仅复用决定了查询类别

**切入角度**：将一大类梯度算法统一为 $\mathbf{w}^{(t+1)} \leftarrow \mathbf{w}^{(t)} + \gamma \psi_\eta(y, ⟨\mathbf{x}, \mathbf{w}⟩) P_\mathbf{w}^\perp \mathbf{x}$ 的形式，$\eta$ 控制非关联项的尺度

**核心idea**：样本复杂度由 Hermite 系数 $\mu_i(\eta)$ 的竞争决定，$\eta$ 的大小决定哪个项主导，从而在 information exponent 体制和 generative exponent 体制之间产生不光滑相变

## 方法详解

### 整体框架
考虑两层网络 $f(\mathbf{x}; W, \mathbf{a}, \mathbf{b}) = \frac{1}{N}\sum_{j=1}^N a_j \sigma_j(⟨\mathbf{x}, \mathbf{w}_j⟩ + b_j)$ 学习目标函数。统一更新规则为球面投影梯度形式：$\mathbf{w}^{(t+1)} \leftarrow \mathbf{w}^{(t)} + \gamma \psi_\eta(y, ⟨\mathbf{x}, \mathbf{w}⟩) P_\mathbf{w}^\perp \mathbf{x}$，$\gamma$ 为全局学习率，$\eta$ 控制非关联更新项的缩放。

### 关键设计

1. **通用框架与 Hermite 系数分析**:

    - 功能：将在线 SGD、batch reuse SGD、alternating SGD 统一为同一框架
    - 核心思路：定义关键量 $\mu_i(\eta) = \mathbb{E}[\psi_\eta(\sigma_*(a), b) \mathsf{He}_i(a) \mathsf{He}_{i-1}(b)]$，其中 $(a,b) \sim \mathcal{N}(0, I_2)$。更新方向与目标的对齐为 $\mathbb{E}[⟨\boldsymbol{\theta}_*, \mathbf{g}⟩] = \sum_i i! \mu_i ⟨\boldsymbol{\theta}_*, \mathbf{w}⟩^{i-1}(1-⟨\boldsymbol{\theta}_*, \mathbf{w}⟩^2)$
    - 设计动机：$\mu_i(\eta)$ 的大小和非零性完全决定哪个 Hermite 阶主导对齐增长

2. **主定理 (Theorem 3.3)**:

    - 功能：给出通用样本复杂度公式
    - 核心思路：取 $\gamma \leq C\delta \max_i \mu_i d^{-(i/2 \vee 1)}$，弱恢复所需迭代数为
    $T(\eta) = \min_{\substack{1 \leq i \leq r \\ \mu_i > 0}} \tilde{\Theta}\big(\gamma^{-1} \mu_i(\eta)^{-1} d^{\frac{i-2}{2} \vee 0}\big)$
      选最优 $\gamma$ 后简化为 $T(\eta) = \min_i \tilde{\Theta}(\mu_i(\eta)^{-2} d^{(i-1)\vee 1})$
    - 设计动机：$\min$ 操作导致当最优 $i$ 随 $\eta$ 变化时产生不光滑相变

3. **Batch Reuse SGD 实例化 (Algorithm 1)**:

    - 功能：两步更新——先用 $\eta$ 做一步梯度，再用 $\gamma$ 在同一样本上做第二步
    - 核心思路：Taylor 展开后得 $\psi_\eta(y,z) = y\sigma'(z) + \sum_{k=2}^r \frac{(\eta d)^{k-1}}{(k-1)!}\sigma^{(k)}(z)(\sigma'(z))^{k-1} y^k$。相变条件：$\eta \leq d^{\frac{[(p_j-1)\vee 1]-[(p_i-1)\vee 1]}{2(j-i)}-1}$ 时处于 information exponent 体制
    - 设计动机：验证框架正确性——当 $\eta \lesssim d^{-(p+1)/2}$ 时复杂度退化为在线 SGD 的 $\Theta(d^{(p-1)\vee 1})$，当 $\eta \gtrsim d^{-1}$ 时恢复 $\tilde{\Theta}(d)$

4. **Alternating SGD (Algorithm 2, 本文新算法)**:

    - 功能：逐层更新——先用 $\eta$ 更新第二层 $\tilde{a} \leftarrow a + \eta y \sigma(⟨\mathbf{x}, \mathbf{w}⟩)$，再用更新后的 $\tilde{a}$ 和 $\gamma$ 更新第一层
    - 核心思路：更新函数 $\psi_\eta(y,z) = ya\sigma'(z) + \eta y^2 \sigma(z)\sigma'(z)$，系数 $\mu_i(\eta) = ai \cdot u_i(\sigma_*) u_i(\sigma) + \eta \cdot u_{i-1}(\sigma\sigma') u_i(\sigma_*^2)$。若 $p_2 = \text{IE}(\sigma_*^2) < p$ 且 $\eta$ 足够大，第二项主导，复杂度为 $\tilde{\Theta}(\eta^{-2} d^{(p_2-1)\vee 1})$
    - 设计动机：无需复用样本且使用标准平方损失即可引入非关联项——两层不同学习率的时间尺度分离自然产生 $y^2$ 变换

### 训练策略
- Alternating SGD：对 $\sigma_* = \mathsf{He}_3$（$p=3, p_2=2$），相变阈值 $\eta \asymp d^{-1/2}$
- $\eta$ 以下：$T = \Theta(d^2)$（information exponent 体制）
- $\eta$ 以上：$T = \tilde{\Theta}(d)$（线性复杂度）
- 可推广到 $D$ 层稀疏网络：$T = \max_{1 \leq i \leq D} \tilde{\Theta}(\eta^{-2(i-1)} d^{(p_i-1)\vee 1})$

## 实验关键数据

### 主要理论结果对比

| 算法 | 复杂度 ($\eta$ 小) | 复杂度 ($\eta$ 大) | 相变阈值 |
|------|-------------------|-------------------|---------|
| 在线 SGD | $\tilde{\Theta}(d^{(p-1)\vee 1})$ | 不适用 | 无 |
| Batch Reuse SGD | $\tilde{\Theta}(d^{(p-1)\vee 1})$ | $\tilde{\Theta}(d^{(p_*-1)\vee 1})$ | $\eta \asymp d^{-1}$ |
| Alternating SGD | $\tilde{\Theta}(d^{(p-1)\vee 1})$ | $\tilde{\Theta}(d^{(p_2-1)\vee 1})$ | $\eta \asymp d^{-\frac{p-p_2}{2}}$ |

### 数值实验 ($\sigma_* = \sigma = \mathsf{He}_3$, $d=50$)

| $\eta$ 范围 | 达到 $⟨\mathbf{w}, \boldsymbol{\theta}_*⟩ \geq 0.5$ 所需样本 $n$ | 体制 |
|------------|--------------------------------------------------------------|------|
| $\eta \lesssim d^{-1/2}$ | $n \approx d^2 = 2500$ | information exponent |
| $\eta \gtrsim d^{-1/2}$ | $n$ 随 $\eta^{-2}$ 下降 | 过渡区 |
| $\eta \approx 1$ | $n \approx d \cdot \text{polylog}$ | generative exponent |

### 关键发现
- Figure 1 清晰展示两个不同斜率的区域，相变点与理论预测 $\eta \asymp d^{-1/2}$ 一致
- Alternating SGD 是首个不需要 batch reuse 也不需要改变损失函数即可突破 CSQ 限制的算法
- 深度 $D$ 层网络可进一步利用 $\sigma_*^D$ 的 information exponent，在更多情况下实现线性复杂度

## 亮点与洞察
- **学习率作为算法设计的关键维度**：不仅影响收敛速度，还决定算法属于 CSQ 还是 SQ 类别——这一洞察深刻且此前被忽视
- **Alternating SGD 的简洁性**：仅仅对两层使用不同学习率（两时间尺度）就能产生非关联更新，无需 batch reuse 或改变损失
- **统一框架的表达力**：通过 $\mu_i(\eta)$ 系数统一分析三种算法，使得相变条件和复杂度比较一目了然

## 局限与展望
- 仅考虑单指标模型，多指标模型的推广尚未完成
- 假设高斯输入分布，一般分布的情况需要进一步研究
- Alternating SGD 的分析仅覆盖 $p_2 = \text{IE}(\sigma_*^2)$，深层网络的条件较难验证
- 非多项式激活函数的情况未覆盖

## 相关工作与启发
- **vs Ben Arous et al. (BAGJ'21)**：确立了在线 SGD 的 information exponent 界，本文展示学习率可改变这一界
- **vs Damian et al. / Lee et al.**：他们用 batch reuse 达到 generative exponent 体制，本文揭示这需要 $\eta$ 足够大
- **vs Joshi et al.**：他们通过改变损失函数突破 CSQ，本文的 alternating SGD 保持平方损失不变
- **vs Cutkosky-Wei-Lin et al.**：他们用广义梯度+扰动+平均达到 SQ 最优，本文的视角互补

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 学习率引发相变的发现深刻，alternating SGD 概念简洁有力
- 实验充分度: ⭐⭐⭐ 理论为主，数值验证仅在 $d=50$ 的小规模下
- 写作质量: ⭐⭐⭐⭐⭐ 统一框架阐述清晰，三个实例的逐步展开节奏优秀
- 价值: ⭐⭐⭐⭐⭐ 揭示了SGD复杂度中学习率的基本角色，对理解神经网络特征学习有深远意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Orthogonal Multi-Index Models: A Fine-Grained Information Exponent Analysis](learning_orthogonal_multi-index_models_a_fine-grained_information_exponent_analy.md)
- [\[NeurIPS 2025\] Evaluating LLMs for Combinatorial Optimization: One-Phase and Two-Phase Heuristics for 2D Bin-Packing](evaluating_llms_for_combinatorial_optimization_one-phase_and_two-phase_heuristic.md)
- [\[NeurIPS 2025\] Emergence and Scaling Laws in SGD Learning of Shallow Neural Networks](emergence_and_scaling_laws_in_sgd_learning_of_shallow_neural_networks.md)
- [\[NeurIPS 2025\] Learning Quadratic Neural Networks in High Dimensions: SGD Dynamics and Scaling Laws](learning_quadratic_neural_networks_in_high_dimensions_sgd_dynamics_and_scaling_l.md)
- [\[NeurIPS 2025\] Improving the Straight-Through Estimator with Zeroth-Order Information](improving_the_straight-through_estimator_with_zeroth-order_information.md)

</div>

<!-- RELATED:END -->
