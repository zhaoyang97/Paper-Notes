---
title: >-
  [论文解读] Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies
description: >-
  [NeurIPS 2025][优化/理论][自然梯度下降] 研究使用自然梯度下降优化器 iVON 替代标准 SGD 来优化变分推断中的 BNN 参数，在射电星系分类中获得更好的不确定性校准，同时保持与 HMC 和 BBB-VI 相当的预测性能。 未来射电天文巡天预计产出 EB 级数据，需要统计上鲁棒的 ML 模型…
tags:
  - "NeurIPS 2025"
  - "优化/理论"
  - "自然梯度下降"
  - "变分推断"
  - "贝叶斯神经网络"
  - "不确定性校准"
  - "射电星系分类"
---

# Natural Gradient Descent for Improving Variational Inference Based Classification of Radio Galaxies

**会议**: NeurIPS 2025  
**arXiv**: [2511.13224](https://arxiv.org/abs/2511.13224)  
**代码**: [有](https://github.com/devinamhn/RadioGalaxies-BNNs)  
**领域**: 贝叶斯深度学习 / 射电天文  
**关键词**: 自然梯度下降, 变分推断, 贝叶斯神经网络, 不确定性校准, 射电星系分类

## 一句话总结

研究使用自然梯度下降优化器 iVON 替代标准 SGD 来优化变分推断中的 BNN 参数，在射电星系分类中获得更好的不确定性校准，同时保持与 HMC 和 BBB-VI 相当的预测性能。

## 研究背景与动机

未来射电天文巡天预计产出 EB 级数据，需要统计上鲁棒的 ML 模型。**贝叶斯神经网络（BNN）** 提供原则性的不确定性建模方式。

**前期基准工作**发现：
- HMC（哈密顿蒙特卡洛）在预测精度、校准和分布偏移检测上整体最佳，但计算成本极高（7天）
- BBB-VI（Bayes by Backprop）性能尚可但面临初始化敏感、收敛慢、冷后验效应等问题

**核心问题**：标准 VI 使用 SGD 优化变分参数，但 VI 的参数空间是**统计流形**（黎曼流形），每个点对应一个概率分布。SGD 假设欧几里得空间，可能不是在分布空间中最高效的优化方向。

**自然梯度下降（NGD）** 用 Fisher 信息矩阵（FIM）的逆对梯度进行预条件化，账户了统计流形的几何结构，提供更直接的分布空间路径。

## 方法详解

### 整体框架

使用 **iVON（Improved Variational Online Newton）** 算法替代 SGD 优化 BNN 的变分参数。iVON 基于**贝叶斯学习规则（BLR）**框架，将变分推断与自然梯度下降统一。

### 关键设计

1. **BLR 框架下的自然梯度更新**：

    - 变分分布选择多元高斯 $q(\boldsymbol{\theta}) = \mathcal{N}(\boldsymbol{\theta}|\mathbf{m}, \mathbf{S}^{-1})$
    - 自然梯度更新自然参数：$\boldsymbol{\lambda}_{t+1} \leftarrow \boldsymbol{\lambda}_t - \alpha \nabla_{\boldsymbol{\mu}}\{\mathbb{E}_{q}[l(\boldsymbol{\theta})] - \mathcal{H}(q)\}$
    - 等价于类牛顿法更新，需要梯度和 Hessian 信息

2. **iVON 的可扩展近似**：

    - **对角 Hessian 近似**：降低计算从 $O(d^2)$ 到 $O(d)$
    - **重参数化技巧**估计二阶信息：$\hat{\mathbf{h}} = \hat{\mathbf{g}} \cdot (\boldsymbol{\theta} - \mathbf{m}) / \boldsymbol{\sigma}^2$——通过测量梯度对参数随机扰动的响应来近似曲率
    - **几何修正项**：保证精度矩阵正定性，确保变分分布在训练全程有效
    - 均值和标准差更新：$\mathbf{m} \leftarrow \mathbf{m} - \alpha \frac{(\hat{\mathbf{g}} + \delta\mathbf{m})}{(\mathbf{h} + \delta)}$，$\boldsymbol{\sigma} \leftarrow \frac{1}{\sqrt{\text{ess}(\mathbf{h} + \delta)}}$

3. **与 BBB-VI 的关键区别**：

    - BBB-VI 在欧几里得空间分别对均值和方差计算梯度：$\mathbf{m} \leftarrow \mathbf{m} - \alpha \nabla_\mathbf{m} \mathcal{L}$
    - iVON 使用自然梯度，分母中的曲率估计 $(\mathbf{h} + \delta)$ 提供自适应步长
    - iVON 使用1个 MC 样本即可，与 BBB 一致

### 训练策略

- 数据集：MiraBest Confident（FRI/FRII 射电星系二分类，584训练/145验证/104测试）
- 架构：LeNet-like 网络
- 学习率 0.2 + 余弦退火，warmup 5 epochs
- Hessian 初始化 $h_0 = 0.5$（从 {0.01, 0.1, 0.5, 1, 5} 中选出）
- **有效样本量 ESS**：$\text{ess} = 10N$ 给出最佳结果（冷后验）
- 权重衰减 $\delta = 10^{-4}$，训练 1000 epochs，10 个随机种子

## 实验关键数据

### 主实验——预测性能与校准（表1）

| 推断方法 | 测试错误率 ↓ | UCE ↓ | 训练时间 |
|----------|-------------|-------|---------|
| HMC | 4.16 ± 0.45 | 14.76 ± 0.95 | **7 天** |
| BBB-VI | 3.94 ± 0.01 | 12.77 ± 6.11 | 40 分钟 |
| **iVON (ess=10N)** | **3.07 ± 1.47** | **8.37 ± 4.12** | **25 分钟** |
| iVON (ess=100N) | 3.36 ± 1.23 | 12.19 ± 6.57 | 25 分钟 |

### 分布偏移检测（能量分数分析）

| 数据集 | HMC | BBB-VI | iVON |
|--------|-----|--------|------|
| MiraBest (iD) | 低能量 ✓ | 低能量 ✓ | 低能量 ✓ |
| GalaxyMNIST (远 OoD, 光学) | 高能量 ✓ | 高能量 ✓ | 高能量 ✓ |
| MIGHTEE (近 OoD, 不同射电望远镜) | 高能量 ✓ | 中等区分 ✓ | 无法可靠区分 ✗ |

### 关键发现

- **iVON 在 UCE 上全面最优**（8.37 vs BBB-VI 12.77 vs HMC 14.76），不确定性校准显著改善
- 预测性能与最佳方法相当（3.07%），且**训练速度比 BBB-VI 快 37.5%**，比 HMC 快 400 倍+
- **冷后验效应持续存在**：ess=10N（冷后验）优于 ess=N（标准后验），说明即使更好的优化器也未消除该效应
- **近 OoD 检测减弱**：iVON 对 MIGHTEE（不同望远镜分辨率/灵敏度的射电星系）无法可靠检测，但对远 OoD（光学星系）仍然有效
- 不同优化器收敛到**定性不同的解**——更好的校准但较差的近 OoD 检测，说明优化器引入的归纳偏置不可忽视

## 亮点与洞察

- **优化器作为归纳偏置**：本文揭示了一个重要观察——优化器的选择不仅影响收敛速度，还决定了模型学到的表示类型（分布式/冗余 vs 压缩/局部），从而影响不同下游任务的表现
- 自然梯度下降利用参数空间的黎曼几何，为 VI 优化提供了更「自然」的方向
- 实验虽小但**分析深入**，每个发现都有具体的物理应用意义

## 局限与展望

- 仅在小规模 LeNet 和小数据集上验证，需要在更大架构/数据上确认
- 近 OoD 检测能力下降是重要缺陷，限制了在需要检测不同望远镜数据偏移的场景中的应用
- 冷后验效应未被解决，暗示更深层的模型误设问题
- ESS 超参数选择需要领域知识，缺乏自动选择策略
- 对角 Hessian 近似可能丢失参数间重要关联

## 相关工作与启发

- **iVON**（Lin et al.）和 **BLR**（Khan & Rue）提供了将多种学习算法统一在贝叶斯学习规则下的优雅理论框架
- **Noisy Natural Gradient**（Zhang et al.）是早期的自然梯度 VI 尝试
- 对天文学和其他科学领域中使用 BNN 的研究者有直接指导意义：选择优化器时需考虑对不同评估维度的影响
- 启发未来研究形式化不同优化器的归纳偏置及其对后验近似质量的影响

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将已有优化器应用于新领域，理论贡献有限但观察有价值
- 实验充分度: ⭐⭐⭐⭐ — 小规模但分析细致，复现了前期基准结果
- 写作质量: ⭐⭐⭐⭐⭐ — 背景知识讲解清晰，数学推导完整
- 价值: ⭐⭐⭐⭐ — 对射电天文 BNN 社区有直接参考价值，对更广泛 BNN 社区的洞察有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VIKING: Deep Variational Inference with Stochastic Projections](viking_deep_variational_inference_with_stochastic_projections.md)
- [\[NeurIPS 2025\] NeuSymEA: Neuro-symbolic Entity Alignment via Variational Inference](neuro-symbolic_entity_alignment_via_variational_inference.md)
- [\[NeurIPS 2025\] Least Squares Variational Inference](least_squares_variational_inference.md)
- [\[NeurIPS 2025\] Optimal Rates for Generalization of Gradient Descent for Deep ReLU Classification](optimal_rates_for_generalization_of_gradient_descent_for_deep_relu_classificatio.md)
- [\[NeurIPS 2025\] VERA: Variational Inference Framework for Jailbreaking Large Language Models](vera_variational_inference_framework_for_jailbreaking_large_language_models.md)

</div>

<!-- RELATED:END -->
