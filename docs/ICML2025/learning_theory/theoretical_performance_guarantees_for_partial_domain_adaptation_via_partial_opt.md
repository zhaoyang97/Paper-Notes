---
title: >-
  [论文解读] Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport
description: >-
  [ICML 2025][迁移学习 / 最优传输][域适应] 本文基于部分最优传输理论推导了部分领域自适应（PDA）的泛化界，证明了部分 Wasserstein 距离作为领域对齐项和提出的理论驱动权重方案的合理性，并据此开发了实用算法 WARMPOT。 领域现状：领域自适应（DA）旨在将源域的知识迁移到标签稀缺的目标域…
tags:
  - "ICML 2025"
  - "迁移学习 / 最优传输"
  - "域适应"
  - "partial optimal transport"
  - "Wasserstein distance"
  - "generalization bounds"
  - "WARMPOT"
---

# Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport

**会议**: ICML 2025  
**arXiv**: [2506.02712](https://arxiv.org/abs/2506.02712)  
**代码**: 无  
**领域**: 迁移学习 / 最优传输  
**关键词**: partial domain adaptation, partial optimal transport, Wasserstein distance, generalization bounds, WARMPOT

## 一句话总结
本文基于部分最优传输理论推导了部分领域自适应（PDA）的泛化界，证明了部分 Wasserstein 距离作为领域对齐项和提出的理论驱动权重方案的合理性，并据此开发了实用算法 WARMPOT。

## 研究背景与动机
**领域现状**：领域自适应（DA）旨在将源域的知识迁移到标签稀缺的目标域。部分领域自适应（PDA）是一种更现实的设置：目标域的标签空间是源域的子集（如源域有100类，目标域只有50类）。

**现有痛点**：PDA 方法通常最小化领域对齐项 + 加权源域经验损失，但：(1) 对齐项的选择缺乏理论依据；(2) 加权方案大多是启发式的（如基于类别预测概率的权重）；(3) 缺乏泛化保证。

**核心矛盾**：实践中 PDA 方法有效，但理论基础薄弱，不同加权方案的优劣缺乏理论解释。

**本文目标**：提供 PDA 的理论泛化界，从中推导出有原则的权重公式和域对齐度量。

**切入角度**：部分最优传输（Partial OT）——一种只传输部分质量的最优传输变体，天然适合处理标签空间不对称的情况。

**核心idea**：部分 Wasserstein 距离是 PDA 中正确的领域对齐度量，且泛化界中自然给出了源域样本的最优权重。

## 方法详解

### 整体框架
输入：有标签源域数据 $D_S$（$C_S$ 类）、无标签目标域数据 $D_T$（$C_T \subseteq C_S$ 类）
输出：在目标域上准确的分类器 $h$

### 关键设计

1. **基于部分最优传输的泛化界**:

    - 功能：推导目标域风险的上界
    - 核心思路：
    $\mathcal{R}_T(h) \leq \sum_i w_i \ell(h(x_i^S), y_i^S) + \lambda \cdot W_s^{(p)}(\hat{\mu}_S^w, \hat{\mu}_T) + \text{复杂度项}$
      其中 $W_s^{(p)}$ 是部分 Wasserstein 距离（只传输 $s = |C_T|/|C_S|$ 比例的质量），$w_i$ 是源样本权重
    - 设计动机：标准 Wasserstein 距离在 PDA 中不合适（会强制对齐源域独有类的样本），部分 Wasserstein 自动忽略不相关的源域样本

2. **理论驱动的权重方案**:

    - 功能：从泛化界中推导出源域样本的最优权重
    - 核心思路：在部分 OT 的对偶问题中，运输方案 $\pi^*$ 自然指示了哪些源样本与目标域相关。最优权重正比于该运输方案的边际：
    $w_i^* \propto \sum_j \pi^*(x_i^S, x_j^T)$
      即一个源样本的权重取决于它被"运输"到多少目标样本
    - 设计动机：启发式权重（如基于分类器预测的权重）忽略了数据分布的几何结构，OT 权重直接基于样本间的距离关系

3. **WARMPOT 算法**:

    - 功能：实用的 PDA 算法
    - 核心思路：
        - Step 1：用特征提取器 $g$ 将源和目标数据映射到特征空间
        - Step 2：计算特征空间中的部分最优传输方案 $\pi^*$
        - Step 3：从 $\pi^*$ 提取源样本权重 $w_i$
        - Step 4：最小化 $\sum_i w_i \ell(h(g(x_i^S)), y_i^S) + \lambda W_s^{(p)}(g_\#\hat{\mu}_S^w, g_\#\hat{\mu}_T)$
        - 迭代更新 $g, h, w$
    - 设计动机：将理论直接转化为可实现的算法

### 损失函数 / 训练策略
$$\mathcal{L} = \sum_i w_i \cdot \text{CE}(h(g(x_i^S)), y_i^S) + \lambda \cdot \hat{W}_s^{(p)}(g(X_S), g(X_T))$$
部分 Wasserstein 距离通过 Sinkhorn 算法的部分传输变体高效近似。

## 实验关键数据

### 主实验

| 数据集 | 指标(ACC) | WARMPOT | PADA | ETN | BA3US |
|---|---|---|---|---|---|
| Office-Home (A→C) | Target ACC | **72.3** | 68.7 | 69.1 | 71.5 |
| Office-Home (R→P) | Target ACC | **81.5** | 78.3 | 79.0 | 80.8 |
| Office-31 (A→W) | Target ACC | **93.2** | 89.5 | 90.1 | 92.4 |
| VisDA (12→6类) | Target ACC | **85.7** | 81.2 | 82.5 | 84.3 |

### 消融实验

| 权重方案 | Office-Home 平均ACC | 说明 |
|---|---|---|
| WARMPOT (OT权重) | **72.3** | 本文理论权重 |
| 均匀权重 | 64.8 | 不加权 |
| 基于分类器预测的权重 | 69.5 | PADA 风格 |
| 基于域判别器的权重 | 70.1 | DANN 风格 |
| 换用标准 Wasserstein | 68.2 | 非部分OT |
| 换用 MMD 对齐 | 67.5 | 非OT方法 |

### 关键发现
- WARMPOT 在多个 PDA 基准上具有竞争力或超过最新方法
- OT 权重显著优于启发式权重方案（均匀、基于预测、基于判别器）
- 部分 Wasserstein 距离比标准 Wasserstein 和 MMD 更适合作为 PDA 的对齐项
- 理论界与实际性能趋势一致

## 亮点与洞察
- 理论 <-> 实践的良好闭环：从泛化界推导方法，方法验证理论
- 部分 OT 与 PDA 的自然对应：传输部分质量 ↔ 只对齐共有类
- OT 权重的几何直觉清晰：远离目标域的源样本自动获得低权重

## 局限与展望
- 部分比例 $s$ 需要预设或估计（通常假设已知目标域的类别数比例）
- 高维特征空间中部分 OT 的计算效率需要优化
- 与开放集领域自适应（OSDA）的联系值得探索

## 相关工作与启发
- 与 Ben-David et al. (2010) 的 DA 泛化界互补：本文专注于部分设置
- 部分 OT 的理论由 Caffarelli & McCann (2010), Figalli (2010) 奠基
- 为 PDA 实践提供了理论指导：选择什么对齐度量、如何加权

## 评分
- 新颖性: ⭐⭐⭐⭐ 部分OT + PDA泛化界的组合有理论价值
- 实验充分度: ⭐⭐⭐⭐ 多数据集+充分消融
- 写作质量: ⭐⭐⭐⭐⭐ 理论和实验衔接流畅
- 价值: ⭐⭐⭐⭐ 为PDA提供了理论基础

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Semi-Supervised Noise Adaptation: Transferring Knowledge from Noise Domain](../../ICML2026/learning_theory/semi-supervised_noise_adaptation_transferring_knowledge_from_noise_domain.md)
- [\[NeurIPS 2025\] How Many Domains Suffice for Domain Generalization? A Tight Characterization via the Domain Shattering Dimension](../../NeurIPS2025/learning_theory/how_many_domains_suffice_for_domain_generalization_a_tight_characterization_via_.md)
- [\[ICLR 2026\] A Statistical Learning Perspective on Semi-dual Adversarial Neural Optimal Transport Solvers](../../ICLR2026/learning_theory/a_statistical_learning_perspective_on_semi-dual_adversarial_neural_optimal_trans.md)
- [\[ICLR 2026\] SEINT: An Efficient SE(p)-Invariant Transport Metric Driven by Polar Transport Discrepancy-based Representation](../../ICLR2026/learning_theory/an_efficient_sep-invariant_transport_metric_driven_by_polar_transport_discrepanc.md)
- [\[ICML 2025\] Near Optimal Best Arm Identification for Clustered Bandits](near_optimal_best_arm_identification_for_clustered_bandits.md)

</div>

<!-- RELATED:END -->
