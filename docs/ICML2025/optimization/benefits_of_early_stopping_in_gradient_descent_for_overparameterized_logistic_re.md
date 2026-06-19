---
title: >-
  [论文解读] Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression
description: >-
  [ICML2025][优化/理论][early stopping] 在过参数化逻辑回归中，理论证明了早停梯度下降（early-stopped GD）相比渐近 GD 具有统计优势：早停 GD 是校准且一致的，而渐近 GD 的 logistic risk 趋于无穷且校准误差不消失；同时建立了早停与 $\ell_2$ 正则化之间的定量联系。
tags:
  - "ICML2025"
  - "优化/理论"
  - "early stopping"
  - "implicit regularization"
  - "logistic regression"
  - "overparameterization"
  - "gradient descent"
  - "calibration"
  - "maximum margin"
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Benefits of Early Stopping in Gradient Descent for Overparameterized Logistic Regression

**会议**: ICML2025  
**arXiv**: [2502.13283](https://arxiv.org/abs/2502.13283)  
**代码**: 无（理论工作）  
**领域**: 优化  
**关键词**: early stopping, implicit regularization, logistic regression, overparameterization, gradient descent, calibration, maximum margin

## 一句话总结
在过参数化逻辑回归中，理论证明了早停梯度下降（early-stopped GD）相比渐近 GD 具有统计优势：早停 GD 是校准且一致的，而渐近 GD 的 logistic risk 趋于无穷且校准误差不消失；同时建立了早停与 $\ell_2$ 正则化之间的定量联系。

## 研究背景与动机
- **过参数化中的隐式偏置**：在过参数化线性回归中，GD 收敛到最小 $\ell_2$ 范数插值器，且在某些协方差条件下可出现 benign overfitting。但在过参数化逻辑回归中，GD 迭代的范数趋于无穷，方向收敛到最大 $\ell_2$-margin 方向（Soudry et al., 2018; Ji & Telgarsky, 2018），情况更复杂。
- **分类中的早停**：在线性回归中，早停（和单遍 SGD）即使在一般协方差下也能获得消失的 excess risk，且其效果与 $\ell_2$ 正则化可比。但在分类问题（logistic loss / 0-1 loss）中，早停的正则化效果尚未被系统性地理论刻画。
- **核心问题**：过参数化逻辑回归中，早停 GD 是否具有相比渐近 GD（收敛到最大 margin 解）的统计优势？这种隐式正则化与显式 $\ell_2$ 正则化之间有何联系？

## 方法详解

### 问题设定
- **数据模型（Assumption 1）**：特征 $\mathbf{x} \sim \mathcal{N}(0, \boldsymbol{\Sigma})$，标签条件概率 $\Pr(y|\mathbf{x}) = 1/(1+\exp(-y\mathbf{x}^\top \mathbf{w}^*))$，即 well-specified 逻辑回归模型。
- **过参数化与噪声**：$\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}} \lesssim 1$（贝叶斯误差为常数，标签有噪声），且 $\operatorname{rank}(\boldsymbol{\Sigma}) \geq n$（有效参数数超过样本数）。
- **度量**：logistic risk $\mathcal{L}(\mathbf{w})$、0-1 error $\mathcal{E}(\mathbf{w})$、校准误差 $\mathcal{C}(\mathbf{w})$，三者满足链式不等式 $\mathcal{E} - \min\mathcal{E} \leq 2\sqrt{\mathcal{C}} \leq \sqrt{2}\sqrt{\mathcal{L}-\min\mathcal{L}}$。

### 贡献一：早停 GD 的上界（校准性与一致性）

**Theorem 3.1（偏差主导上界）**：对任意索引 $k$，存在停止时间 $t$ 使得：

$$\mathcal{L}(\mathbf{w}_t) - \min\mathcal{L} \lesssim \sqrt{\frac{\max\{1, \operatorname{tr}(\boldsymbol{\Sigma})\|\mathbf{w}^*_{0:k}\|^2\}\ln^2(n/\delta)}{n}} + \|\mathbf{w}^*_{k:\infty}\|^2_{\boldsymbol{\Sigma}}$$

- 其中 $\mathbf{w}^*_{0:k}$ 是按 $\lambda_i(\mathbf{u}_i^\top \mathbf{w}^*)^2$ 排序后前 $k$ 个分量的投影。
- 选取 $k$ 随 $n$ 增长→两项均趋于零→早停 GD 对**所有** well-specified 逻辑回归问题都是校准且一致的。
- 在 source/capacity 条件 $\lambda_i \asymp i^{-a}, \lambda_i(\mathbf{u}_i^\top \mathbf{w}^*)^2 \asymp i^{-b}$ 下，速率为 $\tilde{O}(n^{-1/2})$（$b > a+1$）或 $\tilde{O}(n^{(1-b)/(a+b-1)})$（$b \leq a+1$）。

**Theorem 3.2（方差主导上界）**：当 $\|\mathbf{w}^*\| < \infty$ 时可获得更快速率：

$$\mathcal{L}(\mathbf{w}_t) - \min\mathcal{L} \lesssim \|\mathbf{w}^*\|\left(\frac{k}{n} + \sqrt{\frac{\sum_{i>k}\lambda_i}{n}} + \frac{\operatorname{tr}(\boldsymbol{\Sigma})^{1/2}\ln(\cdots)}{n}\right)$$

- 在有限维 $\boldsymbol{\Sigma} = \mathbf{I}_d$ 下达到 $\tilde{O}(d/n)$，恢复经典速率。

### 贡献二：插值估计器的下界

**Theorem 4.1（logistic risk 与校准下界）**：任何范数趋于无穷、方向收敛的估计器序列（包括渐近 GD），其 logistic risk 趋于无穷，校准误差有常数下界 $\mathcal{C} \geq \exp(-C\|\mathbf{w}^*\|_{\boldsymbol{\Sigma}})$。

**Theorem 4.2（0-1 error 下界）**：在 $\boldsymbol{\Sigma}^{1/2}\mathbf{w}^*$ 为 $k$-稀疏、$\operatorname{rank}(\boldsymbol{\Sigma}) \geq Cn\ln(n)\ln(1/\delta)$ 的设定下，**任何**插值估计器的 excess 0-1 error 至少为 $\Omega(1/\sqrt{\ln(n)\ln(1/\delta)})$。
- 这意味着插值估计器需要**指数多**的样本才能获得小的 0-1 error，而早停 GD 只需要**多项式多**的样本——形成指数级的样本复杂度分离。

### 贡献三：早停与 $\ell_2$ 正则化的联系

**核心引理（Lemma 3.3，早停的隐式偏置）**：对凸光滑的经验风险，GD 迭代满足：
$$\frac{\|\mathbf{w}_t - \mathbf{u}\|^2}{2\eta t} + \hat{\mathcal{L}}(\mathbf{w}_t) \leq \hat{\mathcal{L}}(\mathbf{u}) + \frac{\|\mathbf{u}\|^2}{2\eta t}$$
即早停 GD 在获得小经验风险的同时保持较小范数——这正是 $\ell_2$ 正则化的效果。

**Theorem 5.1（全局联系）**：设 $\lambda = 1/(\eta t)$，则对**所有**凸光滑问题：
- $\|\mathbf{w}_t - \mathbf{u}_\lambda\| \leq \|\mathbf{w}_t\|/\sqrt{2}$
- GD 路径与 $\ell_2$ 正则化路径的夹角 $\leq \pi/4$，范数比在 $[0.585, 3.415]$ 之间。

**Theorem 5.2（渐近联系）**：在支持向量条件（Assumption 2）下，$\|\mathbf{w}_t - \mathbf{u}_{\lambda(t)}\| \to 0$，即两条路径的绝对距离趋于零（尽管两者范数均趋于无穷）。

**Theorem 5.3（反例）**：当 Assumption 2 不满足时，存在简单的二维例子使得两条路径的 $\ell_2$ 距离趋于无穷（$\gtrsim \ln\ln\|\mathbf{w}_t\|$）。

## 实验关键数据
- **Figure 1 实验**（$d=2000, n=1000, \lambda_i = i^{-2}$，$\mathbf{w}^*_{0:100}=1, \mathbf{w}^*_{100:\infty}=0$）：
    - 早停 GD 的 population logistic risk 和 0-1 error 在适当停止时间处取得最小值。
    - 随着 GD 继续迭代进入插值阶段，两种误差均显著增大。
    - 经验风险单调下降，但 population risk 呈 U 形曲线，验证了过拟合现象。
    - 最优停止时间对应的 $\eta t$ 约为 $10^2$ 量级，此时 excess risk 接近零。
    - 当 $\eta t \to \infty$ 时 population logistic risk 发散，与 Theorem 4.1 一致。
- 本文以理论分析为主，实验仅作为直观验证，但清晰展示了早停的 U 形过拟合曲线。

## 亮点与洞察
1. **完整的正面+负面理论图景**：不仅证明早停 GD 好（上界），还证明不早停一定差（下界），且下界对所有插值估计器成立。
2. **校准性的重要性**：渐近 GD 可能在 0-1 error 上表现尚可（方向正确），但在概率预测（校准）上完全失败——这对需要概率输出的应用（如医疗诊断）至关重要。
3. **指数级样本复杂度分离**：早停 GD 只需多项式样本，插值估计器需指数样本，这是过参数化分类中已知最强的分离结果之一。
4. **Lemma 3.3 的优雅性**：一个简单不等式统一解释了早停的隐式正则化效果，并直接推导出与 $\ell_2$ 正则化的全局联系。
5. **Theorem 5.2 vs 5.3 的相变**：支持向量条件决定了 GD 路径与正则化路径是渐近重合还是发散——揭示了隐式正则化的脆弱性。

## 局限与展望
1. **停止时间依赖 Oracle 信息**：Theorems 3.1/3.2 中的最优停止时间依赖于真实参数 $\mathbf{w}^*$，不是实际可计算的算法。交叉验证等实际方法的理论保证尚未建立。
2. **偏差-方差 trade-off 不够精确**：两个上界分别由偏差或方差主导，未能刻画真正的偏差-方差最优平衡（不像线性回归中已有精确结果）。
3. **仅限 well-specified 模型**：主要结果假设数据由逻辑模型生成（Assumption 1），misspecified 情形仅有部分推广（仅 logistic risk 上界，无校准/0-1 error 保证）。
4. **最大 margin 估计器的 0-1 error 下界**：Theorem 4.2 对所有插值估计器给出下界，但猜测对最大 margin 估计器应能证明常数级下界（未完成）。
5. **缺乏与 SGD 的比较**：实际训练常用 SGD 而非 GD，SGD 早停的类似理论尚未建立。
6. **技术瓶颈**：线性回归中的精确分析依赖于显式闭式解和固定 Hessian 矩阵，但逻辑回归中 Hessian 随迭代变化且 GD 路径发散到无穷，需要全新的分析工具。
7. **未覆盖非高斯设计**：Assumption 1 要求高斯特征分布，对 sub-Gaussian 设计的推广仅在 Theorem 3.1 中部分讨论。

## 相关工作与启发
- **Benign overfitting 系列**（Bartlett et al., 2020; Chatterji & Long, 2021; Cao et al., 2021）：本文结果不矛盾，而是揭示了即使 benign overfitting 成立（0-1 error 可控），渐近 GD 在校准性和 logistic risk 上仍劣于早停 GD。
- **线性回归中的早停**（Ali et al., 2019; Zou et al., 2021, 2023; Bühlmann & Yu, 2003）：本文将早停正则化理论从线性回归推广到逻辑回归，但技术难度更大（Hessian 非固定，GD 路径发散）。
- **隐式偏置**（Soudry et al., 2018; Ji & Telgarsky, 2018）：本文以最大 margin 方向收敛为基础，进一步分析早停时刻的统计性质。
- **M-estimators**（Ostrovskii & Bach, 2021; Hsu & Mazumdar, 2024）：经典有限维设定中 ERM 的 $O(d/n)$ 速率已知，本文 Theorem 3.2 在特化到该设定时恢复了可比速率 $\tilde{O}(d/n)$。
- **Boosting 与早停**（Zhang & Yu, 2005; Bühlmann & Yu, 2003）：Boosting 文献中已有早停一致性结果，但仅限有限维，未提供插值估计器的下界。
- 对于过参数化深度学习的启发：训练不应完全收敛到插值，适当早停可能在校准性和泛化上都更优。
- **分类校准理论**（Zhang, 2004; Bartlett et al., 2006）：本文提供了具体的统计速率和早停后果，超越了先前的抽象分析。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首次系统建立过参数化逻辑回归中早停 GD 的正/负面理论，指数级样本复杂度分离和正则化路径联系都是重要新结果)
- 实验充分度: ⭐⭐⭐ (理论为主，仅一组说明性实验，但对理论工作而言足够)
- 写作质量: ⭐⭐⭐⭐⭐ (结构清晰，主要结论在 Contributions 中总结到位，技术细节安排合理)
- 价值: ⭐⭐⭐⭐⭐ (为过参数化学习中的优化与统计交互提供了深刻理论洞察，对理解深度学习中的早停实践有重要参考价值)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Large Stepsizes Accelerate Gradient Descent for Regularized Logistic Regression](../../NeurIPS2025/optimization/large_stepsizes_accelerate_gradient_descent_for_regularized_logistic_regression.md)
- [\[ICML 2025\] Constant Stepsize Local GD for Logistic Regression: Acceleration by Instability](constant_stepsize_local_gd_for_logistic_regression_acceleration_by_instability.md)
- [\[CVPR 2025\] Stop Walking in Circles! Bailing Out Early in Projected Gradient Descent](../../CVPR2025/optimization/stop_walking_in_circles_bailing_out_early_in_projected_gradient_descent.md)
- [\[ICML 2025\] Quantum Optimization via Gradient-Based Hamiltonian Descent](quantum_optimization_via_gradient-based_hamiltonian_descent.md)
- [\[ICML 2025\] Incremental Gradient Descent with Small Epoch Counts is Surprisingly Slow on Ill-Conditioned Problems](incremental_gradient_descent_with_small_epoch_counts_is_surprisingly_slow_on_ill.md)

</div>

<!-- RELATED:END -->
