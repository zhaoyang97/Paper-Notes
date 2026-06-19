---
title: >-
  [论文解读] Distributionally Robust Feature Selection
description: >-
  [NeurIPS 2025][特征选择] 本文提出一种模型无关的分布鲁棒特征选择方法，通过向协变量注入可控高斯噪声实现离散选择的连续松弛，并优化 Bayes 最优预测器的条件方差，使选出的特征子集能在多个子群体上同时训练出高质量下游模型。 领域现状：特征选择是机器学习中的基础问题。经典方法如 Lasso、前向/后向选择、XG…
tags:
  - "NeurIPS 2025"
  - "特征选择"
  - "分布鲁棒优化"
  - "Group DRO"
  - "噪声注入"
  - "模型无关"
---

# Distributionally Robust Feature Selection

**会议**: NeurIPS 2025  
**arXiv**: [2510.21113](https://arxiv.org/abs/2510.21113)  
**代码**: 有（论文中提供链接）  
**领域**: 其他  
**关键词**: 特征选择, 分布鲁棒优化, Group DRO, 噪声注入, 模型无关

## 一句话总结
本文提出一种模型无关的分布鲁棒特征选择方法，通过向协变量注入可控高斯噪声实现离散选择的连续松弛，并优化 Bayes 最优预测器的条件方差，使选出的特征子集能在多个子群体上同时训练出高质量下游模型。

## 研究背景与动机

**领域现状**：特征选择是机器学习中的基础问题。经典方法如 Lasso、前向/后向选择、XGBoost 特征重要性等都针对单一分布优化。在分布鲁棒优化（DRO）方面，Group DRO 等方法致力于找到一个在所有子群上表现良好的单一模型，但不涉及特征选择。

**现有痛点**：实际场景中常需在预算约束下选择少量特征（如医疗问卷只能问有限问题），且选出的特征要能为不同群体分别训练出好模型。现有特征选择方法不考虑跨分布鲁棒性，DRO 方法不做特征选择——两者的交集完全未被研究。

**核心矛盾**：特征选择是离散组合优化（NP-hard），DRO 需要 minimax 优化，而这两层优化的耦合使得直接求解极其困难——每换一组特征，所有子群体的模型都要重新训练。

**本文目标**：选择 $k$ 个特征，使得在这 $k$ 个特征上为每个子群体 $P_i$ 分别训练的模型，其最差群体性能最优。

**切入角度**：(a) 用噪声注入替代硬选择实现连续松弛；(b) 转向优化 Bayes 最优预测器的性能以消除对特定模型的依赖；(c) 利用高斯噪声模型推导闭式核权重表达式。

**核心 idea**：通过特征级噪声注入 + Bayes 最优预测器方差优化，将不可解的离散-minimax 问题转化为可微分的连续优化，实现模型无关的分布鲁棒特征选择。

## 方法详解

### 整体框架
输入是多个子群体的带标签数据 $\{(X_i^j, Y_i^j)\}$，输出是 $k$ 个最优特征的索引。方法分三步：(1) 为每个群体拟合一次条件期望 $\hat{\mu}_i(X)$；(2) 用噪声注入参数 $\alpha$ 控制每个特征的信息量，通过梯度下降优化 minimax 目标；(3) 选择 $\alpha$ 最小的 $k$ 个特征。

### 关键设计

1. **噪声注入连续松弛**:

    - 功能：将离散的二值掩码 $\alpha \in \{0,1\}^m$ 松弛为连续噪声参数 $\alpha \in \mathbb{R}_{\geq 0}^m$
    - 核心思路：不是直接缩放特征（$\alpha \odot X$，可被模型逆向补偿），而是注入高斯噪声 $S_i(\alpha)|X \sim \mathcal{N}(X_i, \alpha_i)$。$\alpha_i=0$ 表示完全保留信息，$\alpha_i \to \infty$ 表示丢弃该特征
    - 设计动机：确定性缩放不改变信息量（模型可学 $w_i/\alpha_i$ 补偿），随机噪声才真正降低信噪比
    - 与 Lasso 的区别：Lasso 的 $\ell_1$ 正则化只在线性模型中有效，且假设单一分布

2. **Bayes 最优预测器方差优化**:

    - 功能：用 Bayes 最优损失替代具体模型的损失，规避内层优化
    - 核心思路（定理1）：在 MSE 损失下，问题等价于 $\min_\alpha \max_{P_i} -\mathbb{E}_{S(\alpha)}[\mathbb{E}_X[\mu_i(X)|S(\alpha)]^2] + \lambda \text{Reg}(\alpha)$，其中 $\mu_i(X) = \mathbb{E}_{P_i}[Y|X]$
    - 设计动机：(a) 不需要对每个 $\alpha$ 重新训练模型；(b) 与下游模型架构无关；(c) 只需在开始时对每个群体拟合一次 $\mu_i$

3. **核形式闭式解（定理2）**:

    - 功能：将条件期望的经验估计转化为高斯核加权和
    - 闭式权重：$w_i^j(S,\alpha) = \frac{\exp(-\frac{1}{2}(X_i^j - S)^T \text{diag}(\alpha)^{-1}(X_i^j - S))}{\sum_k \exp(-\frac{1}{2}(X_i^k - S)^T \text{diag}(\alpha)^{-1}(X_i^k - S))}$
    - 设计动机：$\alpha$ 直接控制核带宽——$\alpha_i$ 小则该维度核窄（保留信息），$\alpha_i$ 大则该维度核宽（忽略信息）

### 损失函数 / 训练策略
最终优化：$\min_\alpha \max_{P_i} -\frac{1}{b}\sum_{\ell=1}^b (\sum_j w_i^j(S^\ell, \alpha) \mu_i(X_i^j))^2$，用 reparameterization trick $S = X + \sqrt{\alpha} \odot \epsilon$ 保证梯度可传播。内层 max 用 softmax（温度 $\beta$）近似。$\lambda \cdot \text{Reg}(\alpha) = \lambda / \|\alpha\|_1$ 鼓励稀疏。实际中仅对 $k$ 近邻求和以加速。

## 实验关键数据

### 主实验：合成数据集 1（线性模型，3群体，15特征）

| 方法 | Budget=5 群体A MSE | 群体B MSE | 群体C MSE | Budget=10 最差群体 |
|------|-------------------|----------|----------|-------------------|
| **Ours** | **最低** | **均衡** | **均衡** | **与最佳持平** |
| DRO-Lasso | 中等 | 中等 | 较好 | 次优 |
| DRO-XGBoost | 较好 | 中等 | 较好 | 与Ours持平 |
| Vanilla Lasso | 差 | 差 | 最好 | 差 |
| Embedded MLP | 差 | 差 | 差 | 差 |

### 主实验：真实数据集 ACS（收入预测，3个州）

| 方法 | CA MSE↓ | FL MSE↓ | NY MSE↓ | CA R²↑ | FL R²↑ | NY R²↑ |
|------|---------|---------|---------|--------|--------|--------|
| **Ours** | **最低（数量级领先）** | **最低** | **最低** | **最高** | **最高** | **最高** |
| DRO-XGBoost | 高 | 高 | 高 | 低 | 低 | 低 |
| DRO-Lasso | 中 | 中 | 中 | 中 | 中 | 中 |

### 消融：方法特性对比

| 特性 | Ours | Lasso | DRO-Lasso | XGBoost | DRO-XGB | Embedded MLP |
|------|------|-------|-----------|---------|---------|-------------|
| 模型无关 | ✓ | ✗(线性) | ✗(线性) | ✗(树) | ✗(树) | ✗(MLP) |
| 分布鲁棒 | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ |
| 处理非线性 | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| 低方差 | ✓ | 中 | 中 | 高 | 高 | 高 |

### 关键发现
- 在 ACS 数据集上，本方法的 MSE 比所有 baseline 低一个数量级，R² 也显著领先
- 合成数据中，当群体间系数符号反转（群体A和B）时，普通 Lasso 完全失效，本方法保持均衡
- 本方法跨随机种子的方差始终最低，体现了选择稳定性
- 下游预测模型（随机森林 vs MLP）的选择不影响各特征选择方法的相对排名

## 亮点与洞察
- **噪声注入 vs 确定性缩放的洞察**：确定性缩放不改变互信息（线性模型可逆向补偿），只有随机噪声才真正降低特征的信息承载量
- **从具体模型到 Bayes 最优的跳跃**：放弃优化具体模型的损失、转向优化理论最优损失，不仅绕开了 bilevel 优化的计算困难，还使方法天然与下游模型解耦
- **核带宽 = 特征重要性**：最终的高斯核权重中 $\alpha$ 恰好控制各维度的核带宽，建立了特征选择与核方法的直接联系

## 局限与展望
- 理论推导基于 MSE 损失的 bias-variance 分解，推广到交叉熵等其他损失需要非平凡扩展
- 需要为每个群体预先拟合 $\mu_i(X)$，当群体样本极少时估计质量影响最终效果
- K-近邻加速引入近似误差，高维稀疏数据中近邻质量可能下降
- 未考虑特征间的组合效应——某些特征单独无用但组合后很有信息

## 相关工作与启发
- **vs Lasso**: Lasso 假设线性、单分布，本方法模型无关、多分布鲁棒
- **vs Group DRO (Sagawa et al. 2019)**: Group DRO 训练单一鲁棒模型，本方法先选特征再为每个群体单独训练——更灵活
- **vs MAML (Finn et al. 2017)**: 两者都有 bilevel 结构，但 MAML 需要差异化内层训练，本方法通过 Bayes 最优完全规避

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次正式研究分布鲁棒特征选择问题，噪声注入+Bayes最优的组合新颖
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据覆盖回归和分类，但缺少大规模高维实验
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，推导层层递进，从动机到方法到实验逻辑完整
- 价值: ⭐⭐⭐⭐ 问题本身有很强的实际意义（医疗问卷、传感器部署），方法简洁实用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Feature Selection for Latent Factor Models](../../CVPR2025/others/feature_selection_for_latent_factor_models.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](../../ICLR2026/others/distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[ICLR 2026\] Mitigating Spurious Correlation via Distributionally Robust Learning with Hierarchical Ambiguity Sets](../../ICLR2026/others/mitigating_spurious_correlation_via_distributionally_robust_learning_with_hierar.md)
- [\[NeurIPS 2025\] Manipulating Feature Visualizations with Gradient Slingshots](manipulating_feature_visualizations_with_gradient_slingshots.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](robust_sampling_for_active_statistical_inference.md)

</div>

<!-- RELATED:END -->
