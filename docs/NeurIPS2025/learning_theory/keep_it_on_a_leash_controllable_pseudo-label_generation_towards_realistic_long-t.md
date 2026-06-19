---
title: >-
  [论文解读] Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning
description: >-
  [NeurIPS 2025][半监督学习 / 长尾学习][长尾分布] 提出 Controllable Pseudo-label Generation (CPG) 框架，通过可控的自强化优化循环将可靠伪标签逐步纳入标注集，在已知分布上构建 Bayes-optimal 分类器，从而在未标注数据分布完全未知的 Realistic LTSSL 场景下实现最高 15.97% 的准确率提升。
tags:
  - "NeurIPS 2025"
  - "半监督学习 / 长尾学习"
  - "长尾分布"
  - "半监督学习"
  - "伪标签"
  - "分布不匹配"
  - "Logit Adjustment"
---

# Keep It on a Leash: Controllable Pseudo-label Generation Towards Realistic Long-Tailed Semi-Supervised Learning

**会议**: NeurIPS 2025  
**arXiv**: [2510.03993](https://arxiv.org/abs/2510.03993)  
**代码**: [https://github.com/yaxinhou/CPG](https://github.com/yaxinhou/CPG)  
**领域**: 半监督学习 / 长尾学习  
**关键词**: 长尾分布, 半监督学习, 伪标签, 分布不匹配, Logit Adjustment

## 一句话总结

提出 Controllable Pseudo-label Generation (CPG) 框架，通过可控的自强化优化循环将可靠伪标签逐步纳入标注集，在已知分布上构建 Bayes-optimal 分类器，从而在未标注数据分布完全未知的 Realistic LTSSL 场景下实现最高 15.97% 的准确率提升。

## 研究背景与动机

长尾半监督学习（LTSSL）是一个重要的实际场景：标注数据呈长尾分布（常见类别样本多、罕见类别样本少），同时利用大量未标注数据提升模型性能。现有方法存在一个关键假设缺陷：

**传统 SSL 方法**（FixMatch、FreeMatch、SoftMatch）假设标注和未标注数据分布均衡且一致，不适用于长尾场景

**LTSSL 方法**（ABC、CoSSL、BaCon）放松了均衡假设，但仍假设标注和未标注数据分布基本一致

**ReaLTSSL 方法**（ACR、CPE、SimPro）尝试处理分布不匹配，但 ACR/CPE 假设未标注数据服从某种预定义分布（长尾/均匀/逆长尾），SimPro 虽不需要先验但在分布偏差大时估计不准

**核心挑战**：实际场景中未标注数据的分布完全未知，可能是任意分布。例如城市交通监控中，不同区域、不同时段的车辆类型分布差异巨大。现有方法依赖高置信伪标签估计分布再指导伪标签生成，当分布未知且任意时，高置信伪标签本身就可能包含大量错误，形成确认偏差（confirmation bias）。

## 方法详解

### 整体框架

CPG 框架包含三个核心组件：
1. **可控自强化优化循环**（Controllable Self-Reinforcing Optimization Cycle）—— 核心机制
2. **类别感知自适应增强**（Class-Aware Adaptive Augmentation, CAA）—— 增强少数类表征
3. **辅助分支**（Auxiliary Branch）—— 最大化数据利用

### 关键设计

#### 1. 动态可控过滤（Dynamic Controllable Filtering）

与传统方法仅用弱增强视图生成伪标签不同，CPG 同时利用弱增强和强增强视图的预测来识别可靠伪标签。对每个未标注样本 $x_u$：

- 分别从弱增强视图 $\Omega_w(x_u)$ 和强增强视图 $\Omega_s(x_u)$ 获得预测和置信度
- 可靠伪标签的二值掩码定义为：$\mathbb{I} = \mathbb{I}(\tilde{q}_w > \tau) \cdot \mathbb{I}(\tilde{q}_s > \tau) \cdot \mathbb{I}(\hat{q}_w = \hat{q}_s)$
- 三重条件：两个视图的置信度都超过阈值 $\tau$，且预测类别一致
- 额外引入**投票策略**确保同一样本在不同训练步骤中获得一致的伪标签分配

**设计动机**：传统方法将弱增强预测传播到强增强视图，在分布不匹配时容易传播错误预测。CPG 选择只纳入双视图一致且高置信的样本，虽然利用率可能低但可靠性高。

#### 2. 迭代标注数据集构建与 Bayes-Optimal 分类器

这是 CPG 的核心自强化循环：

- 维护标注集频率 $n = \{n_1, \dots, n_C\}$ 和伪标注集频率 $m = \{m_1, \dots, m_C\}$
- 更新后的标注集类别分布 $\pi_c = \phi_c / \sum_{c'} \phi_{c'}$，其中 $\phi_c = n_c + m_c$
- 基于已知分布 $\pi$ 使用 Logit Adjustment 损失构建 Bayes-optimal 分类器
- 改进的分类器在下一轮识别更多可靠伪标签 → 形成正循环

**关键洞察**：不需要估计未标注数据的分布，而是在已知分布（标注集 + 可靠伪标签）上训练，使模型完全不受未标注数据分布的影响。

#### 3. 类别感知自适应增强（CAA）

针对少数类表征不足的问题：

- 定义类别紧凑度 $\alpha(c) = \frac{1}{\phi_c} \sum_{i} \frac{\langle h_i, \mu(c) \rangle}{\|h_i\| \cdot \|\mu(c)\|}$
- 少数类通常类内多样性低（紧凑度高），需要较小增强半径 $r = 1/\alpha$
- 表征合成：$h'_i = h_i + \frac{h_i}{\|h_i\|} \cdot r(c) \cdot \delta_i$，其中 $\delta_i \sim \mathcal{N}(0, I)$
- 对每个少数类样本合成 10 个增强表征

#### 4. 辅助分支

在训练初期可靠伪标签数量有限，引入辅助分支利用所有标注和未标注样本：

- 采用类似 FixMatch 的一致性正则化范式
- 辅助损失 $\ell_{aux}$ 强制弱增强和强增强视图的预测一致

### 损失函数 / 训练策略

总损失为：$\ell_{overall} = \ell_{la} + \omega \cdot \ell_{aux}$

其中 $\ell_{la}$ 是 Logit-Adjusted 交叉熵损失，$\omega$ 是二值指示函数（辅助分支时 $\omega = 1$，主分支时 $\omega = 0$）。

训练分两阶段：
- **初始阶段**（前 30 个 epoch）：仅在标注数据集上训练
- **迭代阶段**：开始纳入可靠伪标签并在更新的标注集上优化

### 理论分析

Theorem 1 给出泛化误差上界：

$$R_T \leq R_0 - \sum_{t=1}^T \lambda_t + U\sum_{t=1}^T \epsilon_t + 4\sqrt{2}\rho\sum_{t=1}^T\sum_{y=1}^C \mathcal{R}_{O_t}(\mathcal{H}_y) + 2U\sum_{t=1}^T \sqrt{\frac{\log(2/\upsilon)}{2O_t}}$$

核心含义：当训练步数增加时，训练样本 $O_t$ 增长且伪标签错误率 $\epsilon_t$ 下降或不变，配合 Bayes-optimal 分类器最大化风险降低 $\lambda_t$，泛化误差可被有效最小化。

## 实验关键数据

### 主实验

在四个数据集上对比 SSL（FixMatch、FreeMatch、SoftMatch）和 ReaLTSSL（ACR、SimPro、CDMAD）方法：

| 数据集 | 场景 | 本文 CPG | SimPro | FreeMatch | 提升 |
|--------|------|---------|--------|-----------|------|
| CIFAR-10-LT (γ=100, Arbitrary) | 分布不匹配 | **82.10** | 65.81 | 65.41 | +15.97 pp |
| CIFAR-10-LT (γ=100, Inverse) | 逆长尾 | **82.37** | 63.70 | 68.91 | +13.46 pp |
| CIFAR-10-LT (γ=100, Consistent) | 一致分布 | **76.93** | 64.13 | 70.08 | +6.85 pp |
| CIFAR-100-LT (γ=10, Arbitrary) | 分布不匹配 | **51.48** | 44.26 | 45.97 | +5.51 pp |
| Food-101-LT (γ=10, Inverse) | 逆长尾 | **25.52** | 17.31 | 21.89 | +3.63 pp |
| SVHN-LT (γ=100, Arbitrary) | 分布不匹配 | **93.99** | 90.10 | 85.78 | +3.89 pp |

**平均提升**：CIFAR-10-LT +11.14 pp，CIFAR-100-LT +3.09 pp，SVHN-LT +4.06 pp，Food-101-LT +2.57 pp

### 消融实验

| 配置 (w/ AB, CSOC, CAA) | CIFAR-10-LT Arbitrary | CIFAR-100-LT Arbitrary | 说明 |
|-------------------------|----------------------|----------------------|------|
| 无任何组件 | 65.18 | 46.32 | Baseline |
| +AB (辅助分支) | 65.98 (+0.80) | 48.37 (+2.05) | 数据利用率提升 |
| +AB +CAA | 68.97 (+3.79) | 48.90 (+2.58) | 少数类增强有效 |
| +AB +CSOC | 80.39 (+15.21) | 49.25 (+2.93) | 核心组件贡献最大 |
| +AB +CSOC +CAA (完整) | **82.33 (+17.15)** | **51.85 (+5.53)** | 各组件互补 |

CSOC 单独贡献平均 6.97 pp，CAA 单独贡献 2.35 pp，组合后 10.65 pp（超加性效应）。

### 关键发现

1. **任意分布场景优势最大**：CPG 在分布不匹配场景下优势尤其明显（CIFAR-10-LT 最高 +15.97 pp）
2. **伪标签质量显著提升**：CPG 的伪标签错误率远低于 FreeMatch 和 SimPro，尤其在少数类上
3. **鲁棒性强**：随未标注数据不平衡度增大，baseline 性能大幅下降，CPG 基本保持稳定
4. **任意标注分布也适用**：在标注数据也是任意分布的极端设定下（Table 4），CPG 仍一致领先
5. **统计显著性**：所有对比通过 pairwise t-test (0.05 显著性)，CPG 在所有数据集上胜出

## 亮点与洞察

- **核心创新**：不估计未标注数据分布，而是通过逐步扩展已知分布的标注集来规避分布估计的难题——"堵不如疏"的思路
- **自强化循环设计精巧**：好的分类器 → 更多可靠伪标签 → 更大更均衡的训练集 → 更好的分类器，形成良性循环
- **理论支撑扎实**：Theorem 1 从泛化误差角度证明了循环机制的有效性，不只是 heuristic
- **实用性强**：不需要关于未标注数据分布的任何先验知识，真正适合"现实"场景

## 局限与展望

1. **CIFAR-100 上提升有限**：100 类的细粒度分类本身上限低（SL 仅 64.62%），限制了绝对增益
2. **初始 30 epoch 纯标注训练**：这个超参数的选择缺乏自适应机制
3. **伪标签筛选阈值 $\tau$ 的敏感性**：论文未详细分析阈值选择对性能的影响
4. **计算开销**：需维护两个分支（主分支 + 辅助分支），且 CAA 需要计算类别紧凑度和合成表征
5. **投票策略的稳定性**：在极端分布偏移下，早期投票可能被错误伪标签主导

## 相关工作与启发

- **与 SimPro 的对比**：SimPro 用 EM 估计分布再调整伪标签概率，本质上还是依赖分布估计。CPG 彻底绕过了分布估计
- **Logit Adjustment 的巧妙运用**：LA 原本需要已知类别先验，CPG 通过构建已知分布的更新标注集来自然提供先验
- **对其他领域的启发**：这种"在已知分布上训练"的思路可迁移到其他分布不匹配场景（如域适应、联邦学习中的非 IID 数据）

## 评分
- 新颖性: ⭐⭐⭐⭐ — 绕过分布估计的思路新颖，自强化循环设计巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ — 四数据集、多种分布设定、消融实验、统计显著性检验齐全
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、方法系统，图表丰富
- 价值: ⭐⭐⭐⭐ — 在 ReaLTSSL 领域取得显著进展，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Prediction-Powered Semi-Supervised Learning with Online Power Tuning](prediction-powered_semi-supervised_learning_with_online_power_tuning.md)
- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](../../ICML2026/learning_theory/semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[ICML 2025\] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](../../ICML2025/learning_theory/heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)
- [\[NeurIPS 2025\] Product Distribution Learning with Imperfect Advice](product_distribution_learning_with_imperfect_advice.md)
- [\[NeurIPS 2025\] Learning-Augmented Online Bipartite Fractional Matching](learning-augmented_online_bipartite_fractional_matching.md)

</div>

<!-- RELATED:END -->
