---
title: >-
  [论文解读] Gradient Extrapolation for Debiased Representation Learning
description: >-
  [ICCV 2025][社会计算][去偏差] 提出 GERNE 方法，通过构建具有不同虚假相关程度的两个 batch 并对其梯度进行线性外推，引导模型学习去偏差表征，在已知和未知属性情况下均优于 SOTA。 深度学习分类模型在使用经验风险最小化 (ERM) 训练时，常常无意中依赖虚假相关 (spurious correlat…
tags:
  - "ICCV 2025"
  - "社会计算"
  - "去偏差"
  - "虚假相关"
  - "梯度外推"
  - "鲁棒性"
  - "表征学习"
---

# Gradient Extrapolation for Debiased Representation Learning

**会议**: ICCV 2025  
**arXiv**: [2503.13236](https://arxiv.org/abs/2503.13236)  
**代码**: [项目主页](https://gerne-debias.github.io/)  
**领域**: 社会计算  
**关键词**: 去偏差, 虚假相关, 梯度外推, 鲁棒性, 表征学习

## 一句话总结

提出 GERNE 方法，通过构建具有不同虚假相关程度的两个 batch 并对其梯度进行线性外推，引导模型学习去偏差表征，在已知和未知属性情况下均优于 SOTA。

## 研究背景与动机

深度学习分类模型在使用经验风险最小化 (ERM) 训练时，常常无意中依赖**虚假相关 (spurious correlations)**。例如在 Waterbirds 数据集中，模型可能基于背景（水/陆地）而非鸟类本身特征进行分类。当测试数据中这些虚假关联不存在时，泛化性能急剧下降。

**现有方法的局限**：

**需要完整属性标注的方法**（如 Group DRO）：直接最小化最差组损失，但标注成本高昂

**仅在验证集使用属性的方法**（如 DFR、JTT）：用 ERM 预训练模型推断伪属性，但精度受限

**重采样/重加权方法**：虽然简单有效，但性能受限——在虚假相关强烈时，模型仍然优先学习"捷径特征"

**核心矛盾**：ERM 优化的是平均性能，天然偏向学习对多数样本有预测力的捷径特征。即使通过平衡采样（Resampling），模型依然倾向于先学习容易的虚假特征。

本文的**关键 idea**：从模型优化角度出发，利用两个具有不同虚假相关程度的 batch 的梯度差异来定义"去偏方向"，将目标梯度设定为沿该方向的**线性外推**。外推因子 $\beta$ 的调节可以灵活优化 Group-Balanced Accuracy 或 Worst-Group Accuracy。

## 方法详解

### 整体框架

1. 构建两类 batch：偏差 batch $B_b$（保持原始数据偏差分布）和较少偏差 batch $B_{lb}$（更均衡的属性分布）
2. 分别计算两个 batch 的损失和梯度
3. 将目标梯度定义为两个梯度的线性外推
4. 使用外推后的梯度更新模型参数

### 关键设计

1. **Batch 采样策略**：

    - 偏差 batch：$p_b(a|y) = \alpha_{ya} = \frac{|\mathcal{X}_{y,a}|}{|\mathcal{X}_y|}$，反映数据固有偏差
    - 较少偏差 batch：$p_{lb}(a|y) = \alpha_{ya} + c \cdot (\frac{1}{A} - \alpha_{ya})$，参数 $c \in (0,1]$ 控制偏差减少程度
    - 两类 batch 都保证类间均匀采样且组内均匀采样

2. **梯度外推**：

    - 目标损失：$\mathcal{L}_{ext} = \mathcal{L}_{lb} + \beta \cdot (\mathcal{L}_{lb} - \mathcal{L}_b)$
    - 目标梯度：$\nabla_\theta \mathcal{L}_{ext} = \nabla_\theta \mathcal{L}_{lb} + \beta \cdot (\nabla_\theta \mathcal{L}_{lb} - \nabla_\theta \mathcal{L}_b)$
    - 等价于模拟条件属性分布为 $p_{ext}(a|y) = \alpha_{ya} + c \cdot (\beta + 1) \cdot (\frac{1}{A} - \alpha_{ya})$ 的采样

3. **GERNE 作为通用框架**：

    - $\beta = -1$：退化为 ERM
    - $c = 1, \beta = 0$：等价于 Resampling
    - $c \cdot (\beta + 1) = 1$：期望等价于均衡采样，但损失方差不同
    - $c \cdot (\beta + 1) > 1$：**过采样少数组**，更强的去偏效果

4. **$\beta$ 的理论上下界**：

    - 下界 $\beta_{\min} = -1$（退化为 ERM）
    - 上界 $\beta_{\max}$ 由最大组比例 $\alpha_{y''a''}$ 和 $c$ 决定
    - 增大 $\beta$ 超过 $\frac{1}{c} - 1$ 会赋予少数组更高权重，从而优化最差组风险

5. **未知属性情况**：

    - 先训练 ERM 模型 $\tilde{f}$，根据预测置信度划分易/难样本生成伪属性
    - 通过梯度外推可以模拟超出伪组分布范围的条件属性分布（Proposition 1）

### 损失函数 / 训练策略

- 所有实验使用交叉熵损失
- SGD 优化器（视觉任务），AdamW（NLP 任务）
- 超参数 $c, \beta$ 通过网格搜索调优
- 未知属性情况下，阈值 $t$ 作为额外超参数

## 实验关键数据

### 主实验

C-MNIST 和 C-CIFAR-10 数据集（GBA %，已知属性）：

| 方法 | C-MNIST 0.5% | C-MNIST 1% | C-CIFAR-10 0.5% | C-CIFAR-10 1% |
|------|-------------|------------|----------------|---------------|
| Group DRO | 63.12 | 68.78 | 33.44 | 38.30 |
| Resampling | 77.68 | 84.36 | 45.10 | 50.08 |
| **GERNE** | **77.79** | **84.47** | **45.34** | **50.84** |

Waterbirds/CelebA/CivilComments（WGA %，已知属性）：

| 方法 | Waterbirds | CelebA | CivilComments |
|------|-----------|--------|---------------|
| Group DRO | 78.60 | 89.00 | 70.60 |
| DFR | 91.00 | 90.40 | 69.60 |
| **GERNE** | **90.20** | **91.98** | **74.65** |

### 消融实验

$\beta$ 对去偏效果的影响（C-MNIST 0.5%，$c=0.5$）：

| $\beta$ | 等效分布 | 少数组训练精度 | 非偏测试精度 | 稳定性 |
|---------|---------|-------------|------------|--------|
| -1 | ERM | 低 | ~35% | 稳定 |
| 0 | 弱去偏 | ~100% | ~70% | 稳定 |
| 1 | 强去偏 | ~100% | ~77% | 稳定 |
| 1.2 | 接近上界 | 波动 | ~74% | 高方差 |
| >1.22 | 超出界 | - | 发散 | 不可用 |

GERNE vs Resampling 的方差分析表明：GERNE 通过可控的损失方差帮助逃离尖锐极小值，而等效采样+加权方法在去偏极端设置下方差趋近于零，易陷入局部最优。

### 关键发现

- GERNE 在 bFFHQ 上比 Resampling 高 13%+，说明梯度外推方向比简单平衡更有效
- 在少数样本极少时（0.5% minority），GERNE 的优势最明显
- 未知属性情况下，GERNE 依然竞争力强，验证了伪属性+外推的有效性
- 阈值 $t$ 的选择对最优 $\beta$ 有影响：高精度的伪属性允许更低的 $\beta$

## 亮点与洞察

- 将 ERM 和 Resampling 统一为特殊情况的通用框架设计非常优雅
- 理论分析完整：外推因子的上下界推导、与最差组风险的直接联系
- 超越简单平衡：通过外推可以模拟"反转偏差"的采样，这是 Resampling 无法实现的
- 可控损失方差的理论分析为理解为什么外推比等效采样更好提供了新视角

## 局限与展望

- $\beta$ 的最优值对数据集敏感，尤其当 $c$ 较大时可行范围窄
- 未知属性情况依赖 ERM 预训练模型的质量来生成伪属性
- 没有动态调整 $\beta$ 的机制——理想情况下应随训练进展自适应
- CelebA 在无验证属性时性能下降明显，说明对验证集质量有较强依赖
- 通信/计算开销：每步需要计算两个 batch 的梯度

## 相关工作与启发

- Group DRO 直接优化最差组但需要完整标注，GERNE 可以看作"软化"版本
- JTT 的两阶段训练思路与 GERNE 未知属性方案的伪属性生成类似
- DFR 在验证集上重训最后一层的思路与 GERNE 互补
- 梯度外推的 idea 可以启发其他领域的鲁棒性优化

## 评分

- 新颖性：⭐⭐⭐⭐ 梯度外推去偏视角新颖
- 技术深度：⭐⭐⭐⭐ 理论推导完整
- 实验充分度：⭐⭐⭐⭐⭐ 6 个基准全面验证
- 实用价值：⭐⭐⭐⭐ 简单有效，易于集成
- 总体推荐：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning from Neighbors: Category Extrapolation for Long-Tail Learning](../../CVPR2025/social_computing/learning_from_neighbors_category_extrapolation_for_long-tail_learning.md)
- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[ICML 2026\] The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence](../../ICML2026/social_computing/the_geometric_mechanics_of_contrastive_representation_learning_alignment_potenti.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](../../ACL2026/social_computing/toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)
- [\[ICML 2025\] Learning Survival Distributions with the Asymmetric Laplace Distribution](../../ICML2025/social_computing/learning_survival_distributions_with_the_asymmetric_laplace_distribution.md)

</div>

<!-- RELATED:END -->
