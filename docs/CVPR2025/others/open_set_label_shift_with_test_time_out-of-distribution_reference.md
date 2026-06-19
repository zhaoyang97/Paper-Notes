---
title: >-
  [论文解读] Open Set Label Shift with Test Time Out-of-Distribution Reference
description: >-
  [CVPR 2025][开集标签偏移] 本文针对开集标签偏移（OSLS）问题——目标分布包含源分布中没有的OOD类且标签分布变化——提出无需重训练的三阶段估计方法：利用已有的ID分类器和OOD检测器，通过EM算法估计目标域的标签分布和OOD比例，并校正分类器以适应目标分布。 领域现状：标签偏移（Label Shift）是常见…
tags:
  - "CVPR 2025"
  - "开集标签偏移"
  - "分布外检测"
  - "EM算法"
  - "最大似然估计"
  - "分类器校正"
---

# Open Set Label Shift with Test Time Out-of-Distribution Reference

**会议**: CVPR 2025  
**arXiv**: [2505.05868](https://arxiv.org/abs/2505.05868)  
**代码**: [GitHub](https://github.com/ChangkunYe/OpenSetLabelShift)  
**领域**: 分布偏移 / OOD检测  
**关键词**: 开集标签偏移, 分布外检测, EM算法, 最大似然估计, 分类器校正

## 一句话总结
本文针对开集标签偏移（OSLS）问题——目标分布包含源分布中没有的OOD类且标签分布变化——提出无需重训练的三阶段估计方法：利用已有的ID分类器和OOD检测器，通过EM算法估计目标域的标签分布和OOD比例，并校正分类器以适应目标分布。

## 研究背景与动机

**领域现状**：标签偏移（Label Shift）是常见的分布偏移类型——训练和测试时标签分布 $p(y)$ 不同但条件分布 $p(x|y)$ 不变。闭集标签偏移（CSLS）方法已较成熟（如MLLS、BBSE），但现实场景中测试时常遇到训练时未见的OOD类别。

**现有痛点**：Garg et al.首次研究OSLS，但其方法需要重训练ID/OOD分类器以适应目标域，这在分类器冻结或重训代价高时不可行。且重训需要标注，与实际无标签目标域的设定矛盾。

**核心矛盾**：需要在不重训任何分类器的情况下，同时估计目标域K个ID类的标签分布和OOD比例。OOD检测器可能不完美（存在ID/OOD分类误差），进一步增加了估计难度。

**本文目标**：利用现有的冻结ID分类器和任意OOD检测器（来自OOD检测文献），无需重训即可估计目标标签分布并校正分类器。

**切入角度**：将OSLS视为潜变量模型，标签和ID/OOD状态为潜变量，通过EM算法极大化似然。

**核心 idea**：三阶段——(1)估计源域OOD比例 $\rho_s$，(2)EM算法联合估计目标ID标签分布 $\pi$ 和ID比例 $\rho_t$，(3)放松OOD检测器假设后修正 $\rho_t$ 估计。估计结果用于通过importance weighting校正ID分类器。

## 方法详解

### 整体框架
给定：源域ID标记数据、目标域无标签数据、冻结的K类ID分类器 $f$、冻结的ID/OOD检测器 $h$。可选提供一个OOD参考数据集。输出：目标域标签分布估计、校正后的分类器。

### 关键设计

1. **源域OOD比例估计**:

    - 功能：获取 $\rho_s = p_s(b=1)$，即源域中ID数据的比例
    - 核心思路：利用OOD参考数据集和源域数据，通过OOD检测器 $h$ 的预测值构造 $\rho_s$ 的估计量。提供了基于集中不等式的采样误差上界
    - 设计动机：EM算法需要 $\rho_s$ 作为输入；OOD参考数据集提供了校准OOD检测器的锚点

2. **EM算法估计目标分布**:

    - 功能：联合估计目标ID标签分布 $\pi$ 和ID比例 $\rho_t$
    - 核心思路：将目标域数据的标签视为潜变量，用ID分类器 $f$ 的软预测和OOD检测器 $h$ 的ID概率构造完整数据似然。E步计算标签后验，M步更新 $\pi$ 和 $\rho_t$。可选加入Dirichlet先验得到MAP估计
    - 设计动机：EM算法是CSLS的经典方法（MLLS），本文将其推广到包含OOD类的开集设定

3. **OOD检测器不完美时的修正**:

    - 功能：放松对OOD检测器完美的假设
    - 核心思路：当 $h(x)$ 不是真正的 $p_s(b=1|x)$（Assumption 3.3B不成立）时，$\rho_t$ 估计会有偏差。利用OOD参考数据集和ID数据上 $h$ 的统计量构造修正项，提供误差上界
    - 设计动机：现实中OOD检测器基于启发式原理，不可能完美

### 损失函数 / 训练策略
无需训练——方法完全在推理时运行，利用已有分类器的输出。

## 实验关键数据

### 主实验（CIFAR10/100, ImageNet-200）

| 方法 | 标签估计误差↓ | 校正准确率↑ |
|------|------------|-----------|
| Garg et al. (需重训) | 中等 | 中等 |
| BBSE-OVA | 较大 | 较低 |
| **Ours (无需重训)** | **最小** | **最高** |

### 消融实验

| 配置 | 效果说明 |
|------|---------|
| 完美OOD检测器 | 估计精度最高 |
| 不完美OOD检测器+修正 | 近似完美性能 |
| 不同OOD数据集作为参考 | 对参考数据选择不敏感 |
| MAP vs MLE | MAP在小样本时更稳定 |

### 关键发现
- 无需重训任何分类器即可优于需要重训的baseline
- OOD检测器的选择灵活——可直接使用现有文献中的任意方法
- 修正步骤有效处理了OOD检测器不完美的情况
- 在CIFAR10/100和ImageNet-200上均展示了一致的优越性

## 亮点与洞察
- "不重训"的设计极大提高了实用性——冻结模型、隐私约束、计算受限等场景均适用
- 将CSLS经典方法（EM/MLE）优雅地推广到开集设定，理论推导严谨
- 集中不等式给出的采样误差上界提供了可靠性保证

## 局限与展望
- 假设 $p(x|y)$ 在源和目标域间不变（label shift assumption），covariate shift不在考虑范围
- 需要一个OOD参考数据集（虽然选择灵活）
- 当ID分类器本身很差时，估计质量也会下降
- 可拓展到同时处理label shift和covariate shift的设定

## 相关工作与启发
- **vs Garg et al. (OSLS)**: 需要重训OOD分类器；本文直接使用现有OOD检测器
- **vs MLLS/MAPLS (CSLS)**: 本文将闭集EM方法推广到含OOD类的开集设定
- **vs BBSE**: BBSE通过线性系统求解，不处理OOD；本文的EM方法自然处理OOD比例

## 评分
- 新颖性: ⭐⭐⭐⭐ CSLS到OSLS的推广优雅，无需重训是关键优势
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多OOD检测器、理论保证
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，问题定义清晰
- 价值: ⭐⭐⭐⭐ 对实际部署中的分布偏移问题有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Sufficient Invariant Learning for Distribution Shift](sufficient_invariant_learning_for_distribution_shift.md)
- [\[CVPR 2025\] Rethinking Epistemic and Aleatoric Uncertainty for Active Open-Set Annotation: An Energy-Based Approach](rethinking_epistemic_and_aleatoric_uncertainty_for_active_open-set_annotation_an.md)
- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [\[NeurIPS 2025\] Out-of-distribution Generalisation is Hard: Evidence from ARC-like Tasks](../../NeurIPS2025/others/out-of-distribution_generalisation_is_hard_evidence_from_arc-like_tasks.md)
- [\[ECCV 2024\] Bidirectional Uncertainty-Based Active Learning for Open-Set Annotation](../../ECCV2024/others/bidirectional_uncertainty-based_active_learning_for_open-set_annotation.md)

</div>

<!-- RELATED:END -->
