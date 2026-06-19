---
title: >-
  [论文解读] Prediction-Powered Semi-Supervised Learning with Online Power Tuning
description: >-
  [NeurIPS 2025][半监督学习 / 统计推断][半监督学习] 将预测驱动推断（PPI）框架扩展到半监督学习训练过程中，提出无偏梯度估计器，并设计在线AdaGrad算法动态调节伪标签与真实标签的相对权重 $\lambda$，在保证无偏性的同时实现与最优固定 $\lambda$ 匹配的收敛速率。
tags:
  - "NeurIPS 2025"
  - "半监督学习 / 统计推断"
  - "半监督学习"
  - "预测驱动推断"
  - "在线学习"
  - "伪标签去偏"
  - "AdaGrad"
---

# Prediction-Powered Semi-Supervised Learning with Online Power Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2510.22586](https://arxiv.org/abs/2510.22586)  
**代码**: [GitHub](https://github.com/noashoham/PP-SSL)  
**领域**: 半监督学习 / 统计推断  
**关键词**: 半监督学习, 预测驱动推断, 在线学习, 伪标签去偏, AdaGrad

## 一句话总结

将预测驱动推断（PPI）框架扩展到半监督学习训练过程中，提出无偏梯度估计器，并设计在线AdaGrad算法动态调节伪标签与真实标签的相对权重 $\lambda$，在保证无偏性的同时实现与最优固定 $\lambda$ 匹配的收敛速率。

## 研究背景与动机

半监督学习（SSL）利用大量无标签数据来增强少量有标签数据的训练效果，伪标签（pseudo-labeling）是其核心策略之一：用教师模型为无标签数据生成人工标签来扩充训练集。然而，当教师模型在某些子群体上表现不佳时，伪标签会引入偏差（confirmation bias），导致学生模型在这些子群体上性能反而比仅用有标签数据更差。

PPI框架通过在有标签数据上同时计算真实标签和伪标签的损失差来"去偏"：$L_{\text{PPI}} = L_n + \tilde{L}_N^f - L_n^f$，期望上伪标签偏差恰好被抵消。PPI++进一步引入插值参数 $\lambda \in [0,1]$ 来平衡有标签和伪标签的贡献。但现有方法存在两个关键不足：

**分析仅在渐近最优点成立**，没有提供训练过程中有限时间内伪标签如何影响收敛的保证

**$\lambda$ 的选择依赖于离线估计**，需要预先知道标签数据方差 $\sigma^2$ 和教师模型精度 $\mathcal{E}^f$，这些在实践中未知，固定估计值可能严重次优

本文的核心动机就是将PPI从离线统计推断扩展到在线SSL训练，同时用在线学习来自适应调节 $\lambda$。

## 方法详解

### 整体框架

PP-SSL是一个teacher-student框架下的无偏半监督学习算法。每轮迭代中：

1. 获取 $n$ 个有标签样本和 $N$ 个无标签样本（$N \gg n$）
2. 用固定的教师模型 $f$ 为无标签数据生成伪标签
3. 构建预测驱动梯度估计器（prediction-powered gradient）
4. 同时更新模型参数 $w$ 和插值参数 $\lambda$

### 关键设计

1. **预测驱动梯度估计器（PP Gradient）**: 核心公式为 $g_{\text{PP}}^{\lambda} = g^n + \lambda(\tilde{g}^{N,f} - g^{n,f})$，其中 $g^n$ 是有标签数据的标准mini-batch梯度，$\tilde{g}^{N,f}$ 是无标签数据上伪标签的梯度，$g^{n,f}$ 是有标签数据上伪标签的梯度。**无偏性保证**：由于有标签和无标签数据来自同一特征分布 $\mathbb{P}_X$，$\mathbb{E}[\tilde{g}^{N,f} - g^{n,f}] = 0$，因此 $\mathbb{E}[g_{\text{PP}}^{\lambda}] = \nabla \mathcal{L}(w)$。设计动机：修正了伪标签引入的偏差，使得即使教师模型不准确，梯度估计仍然无偏。

2. **方差分析与最优 $\lambda$**: 梯度方差上界为 $(1-\lambda)^2 \sigma^2 + \lambda^2(r\sigma^2 + (1+r)\sigma_e^2)$，其中 $r = n/N$，$\sigma_e^2$ 与教师预测误差 $\mathcal{E}^f$ 通过 $\sigma_e^2 \leq L_Y^2 \cdot \mathcal{E}^f$ 关联。最优化得 $\lambda^* = \frac{1}{1+r} \cdot \frac{\sigma^2}{\sigma^2 + \sigma_e^2}$。当教师准确（$\sigma_e^2 \ll \sigma^2$）时 $\lambda^* \to 1/(1+r)$，方差大幅降低；教师不准时 $\lambda^* \to 0$，自动退化为纯有标签训练。设计动机：建立教师质量、数据量比和方差减少之间的定量关系，为在线调参提供理论目标。

3. **AdaGrad在线调节 $\lambda$**: 将累积方差最小化转化为在线学习问题。定义 $h_t(\lambda) = \|g_t^n + \lambda(\tilde{g}_t^{N,f} - g_t^{n,f})\|^2$，这是 $\lambda$ 的凸函数。使用一维AdaGrad更新 $\lambda_{t+1} = \text{clamp}(\lambda_t - \gamma_t \nabla h_t(\lambda_t); 0, 1)$，学习率 $\gamma_t = (2\sum_{s=1}^{t}\|\nabla h_s(\lambda_s)\|^2)^{-1/2}$ 自适应调整。设计动机：AdaGrad无需知道 $\sigma^2$ 或 $\sigma_e^2$，能自动适应教师模型的质量变化；online regret bound保证了动态调节只引入低阶附加项。

### 损失函数 / 训练策略

模型参数 $w$ 同样使用AdaGrad步长更新：$w_{t+1} = w_t - \eta_t g_t^{\lambda_t}$，$\eta_t = \eta_0 (\sum_{s=1}^{t}\|g_s^{\lambda_s}\|^2)^{-1/2}$。收敛率为 $\mathcal{O}(\sqrt{M\beta V^*/T} + M\beta/T + \sqrt{M\beta}G/T)$，其中第三项是在线学习的附加代价，衰减速度远快于主项 $\mathcal{O}(\sqrt{V^*/T})$，因此整体收敛率与使用最优 $\lambda^*$ 匹配。

## 实验关键数据

### 主实验

| 数据集 | 指标 | PP-SSL | PPI++ | SSL | Only Labeled | 说明 |
|--------|------|--------|-------|-----|-------------|------|
| 合成线性回归（高偏差 $\mu=7$） | MSE（全集） | **最低** | 次低 | 高 | 中等 | PP-SSL在高偏差下优势明显 |
| 合成线性回归（Group B） | MSE | **最低** | 次低 | 最高 | 中等 | SSL反而比仅用标签更差 |
| California Housing | MSE | 与PPI++接近 | 与PP-SSL接近 | 较高 | 较高 | 两种去偏方法都优于基线 |
| UTKFace年龄估计 | MAE | **最低** | 次低 | 最高 | 中等 | PP-SSL在深度模型上也有优势 |
| CIFAR-10（含corruption） | Accuracy | **最高** | 次高 | 低 | 中等 | 对Group B准确率提升最显著 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 固定 $\lambda$（扫描0-1） | MSE | PP-SSL与最佳固定 $\lambda$ 性能相当，无需手动调参 |
| 不同偏差程度 $\mu$ | Group B MSE | 偏差越大PP-SSL相对SSL优势越明显 |
| 不同 $N_B/N_A$ 比例 | MSE | 教师质量差时去偏效果最显著 |
| 有/无group indicator | MSE | 即使不提供组别信息，PP-SSL仍然有效 |

### 关键发现

- 传统SSL在教师模型对子群表现差时**反而不如仅用有标签数据**，本方法通过去偏有效解决
- 在线调节 $\lambda$ 自动适应教师质量，无需事先估计方差和误差
- 该方法首次提供了PPI类方法的有限时间收敛保证（非仅渐近分析）
- 在深度神经网络（ResNet50）上也展现了一致的改善

## 亮点与洞察

- **理论与实践的优雅统一**：从方差分析出发推导最优 $\lambda$，发现其依赖未知量后自然过渡到在线学习，整个推导逻辑链非常紧密
- **AdaGrad for $\lambda$的巧妙构造**：$h_t(\lambda)$ 恰好是 $\lambda$ 的凸二次函数，使得AdaGrad有理论保证；同时用于 $w$ 和 $\lambda$ 的AdaGrad步长可以隐式适应 $\sigma^2$ 和 $\sigma_e^2$
- **实际价值高**：当前LLM时代大量使用伪标签/合成数据，该方法提供了一个有原则的框架来平衡真实与合成数据的贡献

## 局限与展望

- 假设有标签和无标签数据来自相同分布，实际中无标签数据可能来自不同domain或由生成模型产生
- 理论分析假设固定教师模型，self-training（教师持续更新）场景需要更复杂的动态遗憾分析
- 当前收敛分析针对非凸优化的近似平稳点，对全局最优解没有保证
- 方差上界的松紧程度依赖于梯度Lipschitz常数 $L_Y$，该假设对某些损失函数可能较强

## 相关工作与启发

- **与Doubly-Robust Self-Training的对比**：后者也使用PPI-like损失但用预定义的阶梯函数控制 $\lambda$，且只提供渐近保证
- **与PPI++的对比**：PPI++离线调 $\lambda$（最小化渐近方差），PP-SSL在线调（最小化累积二阶矩），后者在实践中更鲁棒
- **启发**：这种"去偏+在线调参"的范式可能对knowledge distillation、学习合成数据等场景同样有用

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 从PPI推广到在线SSL的扩展思路清晰且有实质贡献，在线调 $\lambda$ 的设计巧妙
- **实验充分度**: ⭐⭐⭐⭐ — 合成+真实+深度模型全覆盖，消融分析到位，但缺少大规模基准（ImageNet）
- **写作质量**: ⭐⭐⭐⭐⭐ — 行文严谨，理论推导完整，符号一致，易于跟随
- **价值**: ⭐⭐⭐⭐ — 在伪标签/合成数据大量使用的当下极具实用性，理论基础扎实

## 补充笔记

- Lemma 3.1建立了损失梯度关于标签的Lipschitz性与教师误差 $\mathcal{E}^f$ 之间的桥梁，适用于平方损失和逻辑回归损失
- 该方法天然适配teacher-student框架，其中教师固定、学生模型独立训练，稳定性优于self-training

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning](keep_it_on_a_leash_controllable_pseudo-label_generation_towards_realistic_long-t.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)
- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](../../ICML2026/learning_theory/semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[NeurIPS 2025\] Computable Universal Online Learning](computable_universal_online_learning.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](conformal_online_learning_of_deep_koopman_linear_embeddings.md)

</div>

<!-- RELATED:END -->
