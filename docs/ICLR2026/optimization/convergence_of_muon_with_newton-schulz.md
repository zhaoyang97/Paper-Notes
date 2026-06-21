---
title: >-
  [论文解读] Convergence of Muon with Newton-Schulz
description: >-
  [ICLR2026][优化/理论][Muon optimizer] 首次为实际使用的 Muon 优化器（使用 Newton-Schulz 近似而非精确 SVD 极坐标分解）提供非凸收敛保证：证明收敛速率匹配 SVD 理想化版本（差一个常数因子），该因子随 Newton-Schulz 步数 $q$ 双指数衰减，且 Muon 比向量对应物 SGD-M 少 $\sqrt{r}$ 倍秩损失。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "Muon optimizer"
  - "Newton-Schulz"
  - "polar decomposition"
  - "matrix optimization"
  - "convergence analysis"
---

# Convergence of Muon with Newton-Schulz

**会议**: ICLR2026  
**arXiv**: [2601.19156](https://arxiv.org/abs/2601.19156)  
**代码**: 待确认  
**领域**: 优化/理论  
**关键词**: Muon optimizer, Newton-Schulz, polar decomposition, matrix optimization, convergence analysis

## 一句话总结
首次为实际使用的 Muon 优化器（使用 Newton-Schulz 近似而非精确 SVD 极坐标分解）提供非凸收敛保证：证明收敛速率匹配 SVD 理想化版本（差一个常数因子），该因子随 Newton-Schulz 步数 $q$ 双指数衰减，且 Muon 比向量对应物 SGD-M 少 $\sqrt{r}$ 倍秩损失。

## 研究背景与动机

**领域现状**：Muon 优化器通过正交化动量矩阵（而非像 Adam 那样向量化处理）来更新矩阵参数，在 LLM 训练中表现优异。实际使用 Newton-Schulz (NS) 迭代近似极坐标分解，避免昂贵的 SVD。

**现有痛点**：现有 Muon 理论分析（Shen et al., Li & Hong）都将 NS 替换为精确 SVD——但实际中从不用 SVD。NS 近似误差如何影响收敛？几步 NS 就够吗？why？

**核心矛盾**：实践中 Muon 用少量 NS 步就达到了 SVD 级别效果（更快的 wall-clock），但理论空白——实践远超理论。

**切入角度**：直接分析 NS 近似的极坐标误差 $\varepsilon_q$，证明它随步数双指数衰减。

**核心 idea**：NS 近似误差 $\varepsilon_q$ 双指数衰减→几步 NS 就将 Muon 收敛率拉到 SVD 级别→每步计算远低于 SVD→wall-clock 更快。

## 方法详解

### 整体框架
本文不改动 Muon 算法本身，而是把实际跑的那一版（用 Newton-Schulz 迭代而非精确 SVD 做正交化）放进非凸优化的分析框架，去回答"近似误差到底吃掉多少收敛性"。Muon 每步仍是：随机梯度 $G_t$ 先做动量累积 $M_t = \beta M_{t-1} + G_t$，预缩放成 $X_{t,0} = M_t/\alpha_t$，再跑 $q$ 步 NS 迭代 $X_{t,j} = p_\kappa(X_{t,j-1}X_{t,j-1}^\top)\,X_{t,j-1}$ 把动量推向正交矩阵，最后用近似正交化方向 $O_t$ 走一步 $W_t = W_{t-1} - \eta O_t$。证明目标是给出 nuclear norm 下的平稳性界 $\frac{1}{T}\sum_t \mathbb{E}\big[\|\nabla f(W_{t-1})\|_*\big] \leq \epsilon$。

### 关键设计

**1. NS-Muon 的非凸收敛界：把近似误差隔离成一个常数因子**

既有 Muon 理论（Shen et al.、Li & Hong）都把正交化当成精确 SVD 来分析，可实践里从不用 SVD，于是理论保证悬空。本文直接对带 $q$ 步 NS 的 Muon 做收敛分析，证明在标准光滑、有界方差假设下，达到 $\epsilon$-平稳点所需迭代数为

$$T = O\!\left(\frac{C_q\, L D}{\epsilon^2}\right),$$

其中 $L$ 是光滑常数、$D$ 是初始次优间隙。关键在于把 NS 近似带来的全部退化都压进唯一的常数因子 $C_q$：当 $C_q \to 1$ 时该界就退回 SVD 理想化版本的速率，于是"近似够不够好"被干净地化约成"$C_q$ 离 1 有多远"这一个问题，后面两个设计正是分别回答这个因子从哪来、以及它如何决定 Muon 相对逐元素方法的优势。

**2. 极坐标近似误差的双指数衰减：解释了为什么几步 NS 就够**

上一步把成败押在 $C_q$ 上，那它到底有多接近 1？本文把 $C_q$ 追溯到 NS 对极坐标分解的逼近误差 $\varepsilon_q$，并证明这个误差以 $\varepsilon_q \leq \varepsilon_0^{(2\kappa+1)^q}$ 衰减——指数上还套着一层指数，随 NS 步数 $q$ 与多项式阶 $\kappa$ 双指数收缩。代入实际取值就一目了然：$\kappa=2$ 时底数是 $5$，$q=3$ 步误差已小于 $\varepsilon_0^{125}$（约 $10^{-100}$ 量级），所以 $q=3\!\sim\!5$、$\kappa=2\!\sim\!3$ 就足以让 $C_q\approx 1$，与工程实践中"NS 只跑几步"的经验严丝合缝。这也顺带解释了 wall-clock 优势：NS 每步只是矩阵乘法、在 GPU 上高效，而 SVD 是 $O(mn\min(m,n))$ 的稠密分解，二者迭代数几乎相同时总时间却差很多。

**3. 相比 SGD-M 的 $\sqrt{r}$ 秩优势：度量选对了才看得见**

确认了 NS-Muon 与 SVD-Muon 几乎同速后，剩下的问题是 Muon 这套"正交化动量"相比逐元素的 SGD-M 到底强在哪。本文证明 Muon 的收敛率快 $\sqrt{r}$ 倍（$r=\min(m,n)$ 为参数矩阵的秩维度），而差距的根源在度量的选择：SGD-M 的界写在 Frobenius norm 下，Muon 的正交化则天然匹配 nuclear norm，分析就落在 nuclear norm 度量里，利用了矩阵参数的低秩结构、给出更高效的搜索方向。对 attention 这类高秩大矩阵层，$\sqrt{r}$ 的因子随维度放大，正好对应实践中 Muon 在大模型上观察到的增益。

## 实验关键数据

### 主实验（收敛对比）

| 方法 | 度量 | 收敛率 | 秩依赖 |
|------|------|--------|--------|
| SGD-M | Frobenius 梯度 | $O(1/\sqrt{T})$ | $\sqrt{r}$ 损失 |
| Muon (SVD) | Nuclear 梯度 | $O(1/\sqrt{T})$ | 无 $\sqrt{r}$ 损失 |
| **Muon (NS, $q$ 步)** | Nuclear 梯度 | $O(C_q/\sqrt{T})$ | 无 $\sqrt{r}$ 损失 |

### 消融（$C_q$ vs 步数 $q$）

| NS 步数 $q$ | $\kappa=2$ 时 $C_q$ | $\kappa=3$ 时 $C_q$ |
|-------------|---------------------|---------------------|
| 1 | 大 | 中等 |
| 3 | $\approx 1.01$ | $\approx 1.001$ |
| 5 | $\approx 1.0$ | $\approx 1.0$ |

### 关键发现
- **3-5 步 NS 就匹配 SVD**：$C_q$ 双指数收敛到 1，实际中的选择有充分理论根据
- **Muon vs SGD-M 的 $\sqrt{r}$ 优势**：对高秩矩阵参数（如大 attention 层），优势显著
- **wall-clock 优势解释**：NS 每步成本远低于 SVD，迭代数几乎相同→总时间更少

## 亮点与洞察
- **首次为实际 Muon 提供理论保证**：关闭了 practice-theory gap。之前所有理论都"假装"用 SVD
- **双指数衰减是关键 insight**：$\varepsilon_q \leq \varepsilon_0^{5^q}$（$\kappa=2$ 时）——3步误差 $< 10^{-100}$
- **Nuclear norm 度量的选择**：在矩阵空间用 nuclear norm 而非 Frobenius——自然匹配极坐标分解，揭示秩优势
- **对未来矩阵优化器的启发**：NS 近似的通用分析框架可扩展到其他矩阵优化器

## 局限与展望
- 纯理论贡献，无新实验（但论文目标就是解释已有实践）
- 假设标准光滑+有界方差，未覆盖 Adam 风格自适应
- 未分析 Muon 与 Shampoo/SOAP 等二阶方法的比较

## 相关工作与启发
- **vs Shen et al. / Li & Hong**：他们分析 SVD-Muon。本文首次分析 NS-Muon——唯一匹配实践的理论
- **vs Shampoo/SOAP**：二阶预条件器，维护曲率。Muon 非二阶——正交化动量，机制不同可互补
- **vs Orthogonal-SGDM**：先正交化再动量。Muon 先动量再正交化+用 NS 替代 SVD

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次为实际使用的 NS-Muon 提供收敛理论，双指数衰减结果优美
- 实验充分度: ⭐⭐⭐ 纯理论，无新实验（但合理——补充已有实践的理论解释）
- 写作质量: ⭐⭐⭐⭐⭐ 研究问题清晰、定理层层递进、叙述严谨流畅
- 价值: ⭐⭐⭐⭐⭐ 为当下最热门的矩阵优化器提供了急需的理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Newton Method Revisited: Global Convergence Rates up to $O(1/k^3)$ for Stepsize Schedules and Linesearch Procedures](newton_method_revisited_global_convergence_rates_up_to_o1k3_for_stepsize_schedul.md)
- [\[ICLR 2026\] Error Feedback for Muon and Friends](error_feedback_for_muon_and_friends.md)
- [\[ICLR 2026\] The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton](the_potential_of_second-order_optimization_for_llms_a_study_with_full_gauss-newt.md)
- [\[ICLR 2026\] Hinge Regression Tree: A Newton Method for Oblique Regression Tree Splitting](hinge_regression_tree_a_newton_method_for_oblique_regression_tree_splitting.md)
- [\[ICLR 2026\] MuonBP: Faster Muon via Block-Periodic Orthogonalization](muonbp_faster_muon_via_block-periodic_orthogonalization.md)

</div>

<!-- RELATED:END -->
