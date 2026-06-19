---
title: >-
  [论文解读] Near-Exponential Savings for Mean Estimation with Active Learning
description: >-
  [NeurIPS 2025][优化/理论][主动学习] 提出 PartiBandits 算法，结合基于分歧的主动学习与 UCB 风格的分层抽样，在辅助信息 $X$ 对目标变量 $Y$ 有预测力时，实现了均值估计的近指数级标签节省。 估计一个 $k$ 类随机变量 $Y$ 的总体均值 $\mathbb{E}[Y]$ 是统计学和机…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "主动学习"
  - "均值估计"
  - "分层抽样"
  - "UCB算法"
  - "极小化极大最优"
---

# Near-Exponential Savings for Mean Estimation with Active Learning

**会议**: NeurIPS 2025  
**arXiv**: [2511.05736](https://arxiv.org/abs/2511.05736)  
**代码**: [R Package: PartiBandits](https://cran.r-project.org/)  
**领域**: 优化  
**关键词**: 主动学习, 均值估计, 分层抽样, UCB算法, 极小化极大最优

## 一句话总结

提出 PartiBandits 算法，结合基于分歧的主动学习与 UCB 风格的分层抽样，在辅助信息 $X$ 对目标变量 $Y$ 有预测力时，实现了均值估计的近指数级标签节省。

## 研究背景与动机

估计一个 $k$ 类随机变量 $Y$ 的总体均值 $\mathbb{E}[Y]$ 是统计学和机器学习中的基本问题。当标签昂贵但辅助信息（协变量 $X$）丰富时，如何利用 $X$ 来减少所需标签数量？

**简单随机抽样（SRS）** 的误差率为 $\mathcal{O}(\mathrm{Var}(Y)/N)$，不利用 $X$ 的信息；**分层随机抽样（StRS）** 可以利用 $X$ 但需要预先知道好的分层方案，且层内方差未知时分配策略可能次优。

现有主动学习文献主要关注分类任务中的标签效率，或关注预定义分层下的子组均值估计。但没有工作系统研究：**当最优分层方案未知时，如何用主动学习实现高效的总体均值估计**。

核心挑战有二：(1) 需要从无标签数据中学习一个好的分层方案；(2) 需要在估计方差的同时自适应地分配标签预算。

## 方法详解

### 整体框架

PartiBandits 是一个两阶段算法：

- **阶段一**: 使用基于分歧的主动学习算法 $\mathcal{S}$（如 $A^2$ 算法），用一半标签预算学习一个分类器 $\hat{h}$，再由 $\hat{h}$ 的预映射诱导分层方案 $\mathcal{G} = \{A_i = \hat{h}^{-1}(i)\}$。
- **阶段二**: 在学到的分层上，用 WarmStart-UCB 子程序自适应分配剩余标签预算来估计总体均值。

### 关键设计

1. **WarmStart-UCB 子程序**: 在给定分层方案 $\mathcal{G}$ 下估计总体均值。核心思想是维护每个层的条件方差的 UCB 上界 $\mathrm{UCB}_t(\sigma_g) = \hat{\sigma}_{g,t} + C_N(\delta)/\sqrt{n_{g,t}}$，每轮选择 $\mathrm{UCB}_t(\sigma_g)/n_{g,t}$ 最大的层进行采样。创新点是引入"热启动"阶段：先将 $\tau$ 比例的标签预算均匀分配给所有层，保证每层至少有最低采样量。这消除了对 $\sigma_{\min}$（最小层条件方差）的病态依赖，保证非渐近高概率界在**所有**标签预算下成立。

   **定理 1**: $|\hat{\mu}_{\text{WS-UCB}} - \mathbb{E}[Y]|^2 = \tilde{\mathcal{O}}(\Sigma_1(\mathcal{G})/N)$，其中 $\Sigma_1(\mathcal{G}) = \sum_g \sigma_g'^2 P_g$ 是平均层内方差。由全方差公式，$\Sigma_1(\mathcal{G}) \leq \mathrm{Var}(Y)$，故速率至少与 SRS 一样快。

2. **PartiBandits 主算法的收敛保证**: 通过基于分歧的主动学习算法学到的分类器 $\hat{h}$，其超额风险以近指数速率衰减到贝叶斯最优分类器的风险 $\nu$。由偏差-方差分解，分类器的平方损失上界了诱导分层方案的平均层内方差。因此：

   **定理 3**: $|\hat{\mu}_{\text{PB}} - \mathbb{E}[Y]|^2 = \tilde{\mathcal{O}}\left(\frac{\nu + \exp(c \cdot (-N/\log N))}{N}\right)$
   
   当 $\nu$ 很小（$X$ 对 $Y$ 有强预测力）时，速率远快于 SRS 的 $\mathcal{O}(1/N)$。

3. **极小化极大最优性**: 定理 2 和定理 4 分别为 WarmStart-UCB 和 PartiBandits 建立了匹配的下界，证明了在经典设置下两个速率都是极小化极大最优的。下界基于阈值分类问题 $Y = \mathbf{1}\{X \geq t\}$ 加随机标签翻转。

### 损失函数 / 训练策略

- 使用平方损失 $\mathrm{er}(h) = \mathbb{E}[(h(X)-Y)^2]$ 作为分类器的损失（也支持不对称损失）。
- 关键假设（假设 1）：联合分布 $(X,Y)$ 和假设类 $\mathcal{C}$ 使得主动学习算法可以实现分类超额风险的指数衰减。
- 推论 1-4 展示了不同子算法 $\mathcal{S}$ 如何适配不同问题设置（二分类/多分类、硬间隔/弱噪声条件等）。
- 推论 4 提出了"异质性感知"的 $\mathcal{S}$，通过将弃权区域与确定区域分开，进一步细分高低方差层。

## 实验关键数据

### 主实验

| 设置 | 方法 | $N=50$ 时90%误差界 | $N=100$ 时90%误差界 | 趋势 |
|------|------|-------------------|-------------------|------|
| $\nu=0$（完美可分） | PartiBandits | ~0.002 | ~0.0005 | 远优于SRS |
| $\nu=0$（完美可分） | SRS | ~0.020 | ~0.010 | 标准率 |
| $\nu=0.05$（5%噪声） | PartiBandits | ~0.015 | ~0.004 | 优于SRS |
| $\nu=0.10$（10%噪声） | PartiBandits | ~0.020 | ~0.008 | 略优于SRS |

### 消融实验（WarmStart-UCB）

| 分层方案 $\Sigma_1(\mathcal{G})$ | $N=100$ | $N=200$ | 说明 |
|----------------------------------|---------|---------|------|
| 最优分层（最低$\Sigma_1$） | ~0.003 | ~0.001 | 最快收敛 |
| 次优分层（中等$\Sigma_1$） | ~0.006 | ~0.003 | 中等提升 |
| 较差分层（高$\Sigma_1$） | ~0.009 | ~0.005 | 仍优于SRS |
| SRS（$\Sigma_1 = \mathrm{Var}(Y)$） | ~0.012 | ~0.006 | 最慢 |

### 关键发现

- 实际电子健康记录（AFC数据集，600万条记录）上的模拟验证了理论预测：PartiBandits 在标签预算约30-50时开始超过 SRS，且差距随预算增大而显著扩大。
- 即使 $X$ 不是在整个群体上都有高预测力，只要在分布的尾部有预测力，PartiBandits 就能在这些子区域节省标签。

## 亮点与洞察

- **桥接两大主动学习范式**: 将基于分歧的主动学习（用于分类）与 UCB 风格的自适应采样（用于均值估计）优雅结合，证明了分类中的指数节省可以传递到均值估计任务。
- **消除了 $\sigma_{\min}$ 依赖**: 通过热启动机制解决了分层主动学习文献中的一个公开问题。
- **理论严谨**: 上下界匹配的极小化极大最优结果非常漂亮。
- **实用性强**: 提供了 R 包实现，模拟展示了真实数据上的效果。

## 局限与展望

- 假设 1（指数节省条件）的满足取决于假设类 $\mathcal{C}$ 的选择和数据分布，需要先验知识。
- 阶段一使用一半标签预算可能在标签极度稀缺时浪费；自适应的阶段划分可能更优。
- 实验主要在简单的阈值模型和二分类上验证，高维或复杂数据上的效果有待检验。
- 当 $\nu$ 接近 $\mathrm{Var}(Y)$ 时（$X$ 对 $Y$ 没有预测力），PartiBandits 退化为 SRS，没有额外收益。

## 相关工作与启发

- 直接扩展了 Aznag et al. (2023) 的多组均值估计框架，改进了对 $\sigma_{\min}$ 和组数 $G$ 的依赖。
- 与 Hanneke (2011) 和 Puchkin & Zhivotovskiy (2022) 的分类主动学习理论深度关联。
- 提示了一个更广泛的思路：分类中的标签效率结果可以通过分层机制转化到估计问题中。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次将分类主动学习的指数节省传递到均值估计，桥接两大范式
- 实验充分度: ⭐⭐⭐⭐ 有模拟和真实数据验证，但场景较简单
- 写作质量: ⭐⭐⭐⭐ 理论严谨清晰，但符号较多，入门门槛高
- 价值: ⭐⭐⭐⭐ 理论贡献扎实，对调查抽样和医疗公平领域有实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning](covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)
- [\[NeurIPS 2025\] Robust Estimation Under Heterogeneous Corruption Rates](robust_estimation_under_heterogeneous_corruption_rates.md)
- [\[NeurIPS 2025\] On Minimax Estimation of Parameters in Softmax-Contaminated Mixture of Experts](on_minimax_estimation_of_parameters_in_softmax-contaminated_mixture_of_experts.md)
- [\[ICML 2025\] A Unified View on Learning Unnormalized Distributions via Noise-Contrastive Estimation](../../ICML2025/optimization/a_unified_view_on_learning_unnormalized_distributions_via_noise-contrastive_esti.md)
- [\[ICML 2025\] FSL-SAGE: Accelerating Federated Split Learning via Smashed Activation Gradient Estimation](../../ICML2025/optimization/fsl-sage_accelerating_federated_split_learning_via_smashed_activation_gradient_e.md)

</div>

<!-- RELATED:END -->
