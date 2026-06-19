---
title: >-
  [论文解读] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model
description: >-
  [NeurIPS 2025][图像生成][最优传输] 提出A2A-FM方法，通过一种新颖的代价函数在FlowMatching框架中同时学习所有条件分布对之间的最优传输映射，理论证明在无限样本极限下收敛至逐对最优传输，尤其适用于连续条件变量的非分组数据场景。 条件迁移（condition transfer）是条件生成建模中的核…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "最优传输"
  - "流匹配"
  - "条件迁移"
  - "逐对最优传输"
  - "分子优化"
---

# Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model

**会议**: NeurIPS 2025  
**arXiv**: [2504.03188](https://arxiv.org/abs/2504.03188)  
**代码**: [GitHub](https://github.com/kotatumuri-room/A2A-FM)  
**领域**: 扩散模型 / 流匹配  
**关键词**: 最优传输, 流匹配, 条件迁移, 逐对最优传输, 分子优化

## 一句话总结

提出A2A-FM方法，通过一种新颖的代价函数在FlowMatching框架中同时学习所有条件分布对之间的最优传输映射，理论证明在无限样本极限下收敛至逐对最优传输，尤其适用于连续条件变量的非分组数据场景。

## 研究背景与动机

条件迁移（condition transfer）是条件生成建模中的核心任务：给定一个来自条件分布$P_{c_1}$的样本，生成满足目标条件$c_2$的样本$x \sim P_{c_2}$。典型应用如图像风格迁移、分子属性修改等。

现有方法面临两个关键挑战：

**连续条件问题**：当条件变量$c$为连续值时，每个条件$c$可能只对应一个观测样本$x$，无法为单个条件估计分布$P_c$。大多数现有方法（如Multimarginal SI、EFM）要求"分组数据"（grouped data），即每个条件都有足够多的i.i.d.样本。

**全对全可扩展性**：条件对$(c_1, c_2)$可能有无穷多种组合，逐对学习传输映射的计算开销无法承受。

核心矛盾：如何在非分组数据（non-grouped data）上学习任意条件对之间的最优传输映射？

本文的切入角度是将条件最优传输（COT）的技术推广到逐对最优传输场景，设计一种新的代价函数，使得在mini-batch级别的耦合能够收敛到所有条件对之间的最优传输。

## 方法详解

### 整体框架

A2A-FM基于Flow Matching框架，学习一个以$(c_1, c_2)$为参数的速度场$v(x, t | c_1, c_2)$，通过ODE将$P_{c_1}$传输到$P_{c_2}$：

$$\dot{x}_{c_1, c_2}(t) = v(x_{c_1, c_2}(t), t | c_1, c_2)$$

其中$x_{c_1,c_2}(0) \sim P_{c_1}$，$x_{c_1,c_2}(1) \sim P_{c_2}$。

### 关键设计

1. **逐对最优传输代价函数**：A2A-FM的核心创新在于耦合策略。从数据集$D = \{(x^{(i)}, c^{(i)})\}$中独立抽取两个batch $B_1$和$B_2$，通过最小化以下代价函数获得最优耦合$\pi_\beta^*$：

$$\sum_{i=1}^N \|x_1^{(i)} - x_2^{\pi(i)}\|^2 + \beta \left(\|c_1^{(i)} - c_1^{\pi(i)}\|^2 + \|c_2^{(i)} - c_2^{\pi(i)}\|^2\right)$$

这一代价函数的关键在于：$\beta$项同时约束了**源条件和目标条件**的匹配，而不仅仅是单个条件（区别于COT的代价函数式(6)）。当$\beta$较大时，耦合倾向于将具有相似$(c_1, c_2)$的样本对匹配在一起；当$\beta$较小时，允许不同条件之间共享传输信息。

2. **理论保证（Proposition 3.1）**：证明了当$\beta \to \infty$且样本量$N$同步增大时，上述代价函数的最优耦合收敛到真正的逐对最优传输。即对几乎所有$(c_1, c_2)$：

$$\int \|x_1 - x_2\|^2 d\Pi^*(x_1, x_2 | c_1, c_2) = W_2^2(P_{c_1}, P_{c_2})$$

这意味着在数据充足时，mini-batch上的近似能够捕捉到每对条件分布之间的真实最优传输。

3. **对非分组数据的适用性**：A2A-FM的核心优势在于无需对条件进行分组或离散化。通过$\beta$的平衡作用，即使每个条件只有一个样本，方法仍能通过条件相近的样本共享信息来近似逐对最优传输。实践中$\beta = N^{1/(2d_c)}$（$d_c$为条件维度）是有效的启发式选择。

### 训练策略

训练流程遵循标准CFM：
1. 从数据集抽取batch $B_1, B_2$
2. 通过最优耦合算法OPTC最小化上述代价得到$\pi_\beta^*$
3. 构建线性路径$\psi_i(t) = (1-t)x_1^{(i)} + tx_2^{\pi_\beta^*(i)}$
4. 更新速度场参数$\theta$最小化FM损失$L(\theta) = \sum_i \|v_\theta(\psi_i(t_i), t_i | c_1^{(i)}, c_2^{\pi_\beta^*(i)}) - \dot{\psi}_i(t_i)\|^2$

推理时直接从$t=0$到$t=1$求解ODE进行条件迁移。

## 实验关键数据

### 合成数据验证

| 数据设置 | 方法 | 与逐对OT的MSE |
|----------|------|---------------|
| 分组数据 | A2A-FM | **(5.81±2.22)×10⁻²** |
| 分组数据 | Generalized geodesic | (1.03±0.04)×10⁰ |
| 非分组数据 | A2A-FM | **(1.51±0.17)×10⁻²** |
| 非分组数据 | Partial diffusion | (6.77±0.14)×10⁻² |
| 非分组数据 | Multimarginal SI | (4.90±0.28)×10⁻² |

### 分子优化（QED近邻采样）

| 方法 | 成功率(%) |
|------|-----------|
| A2A-FM | **97.5** |
| COATI-LDM | 95.6 |
| MolMIM | 94.6 |
| QMO | 92.8 |
| DESMILES | 76.9 |

### LogP-TPSA多属性迁移（AUC）

| 方法 | AUC值 |
|------|-------|
| A2A-FM | **0.990** |
| OT-CFM | 0.819 |
| SI (K=10) | 0.583 |
| PD+CFG (T=300) | 0.450 |

### 关键发现

- 在分组和非分组数据上，A2A-FM的耦合和训练后的速度场都更接近真实逐对最优传输
- Multimarginal SI在非分组数据上因离散化效果显著下降；Partial Diffusion产生近乎随机的耦合
- 分子优化中A2A-FM以更少的oracle调用达到更高的成功率，采样效率大幅领先
- 反对称约束$v_{c_1,c_2} = -v_{c_2,c_1}$在QED实验中将成功率从94.6%提升至97.5%

## 亮点与洞察

- 代价函数设计的巧妙之处在于同时约束源和目标条件，相比COT的单条件约束实现了不同条件对之间的迁移
- 理论结果优美：$\beta \to \infty$的极限行为与有限$|\mathcal{C}|$时的直观解释一致，且可推广到连续条件
- 与函数表示定理的联系提供了为什么逐对OT在条件迁移中有效的深层理解
- 计算成本仅依赖于$|D|$而非条件对数$K^2$，比需要分组的方法更可扩展

## 局限与展望

- $\beta$的选择仍存在精度与OT近似之间的权衡，虽然$\beta = N^{1/(2d_c)}$的启发式有效但缺乏严格的最优性保证
- 不保证循环一致性（$T_{c_2 \to c_3} \circ T_{c_1 \to c_2} = T_{c_1 \to c_3}$），因为OT本身不满足此性质
- 在条件维度$d_c$较高时，所需$\beta$的收敛速率可能变慢
- 实验系统规模相对有限，未验证在大规模图像数据集上的表现

## 相关工作与启发

- 与OT-CFM的关系：A2A-FM可视为将OT-CFM从"源到目标的单向传输"推广到"任意条件对之间的全对全传输"
- COT方法（Chemseddine et al.）提供了证明技术的灵感，但其代价函数只支持条件生成而非条件迁移
- 在药物设计、材料科学等条件为连续物理量的场景中有重要应用前景

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 提出全新的逐对OT代价函数，理论证明严谨，填补了非分组数据条件迁移的空白
- **实验充分度**: ⭐⭐⭐⭐ 合成数据验证理论、分子优化展示应用价值，但缺乏更多领域的实验
- **写作质量**: ⭐⭐⭐⭐⭐ 理论推导严谨清晰，直觉解释到位，与相关工作的区别讨论详尽
- **价值**: ⭐⭐⭐⭐⭐ 解决了条件迁移的基本问题，适用范围广，化学应用展示了实际价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICCV 2025\] The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation](../../ICCV2025/image_generation/the_curse_of_conditions_analyzing_and_improving_optimal_transport_for_conditiona.md)
- [\[ECCV 2024\] AFreeCA: Annotation-Free Counting for All](../../ECCV2024/image_generation/afreeca_annotation-free_counting_for_all.md)
- [\[CVPR 2026\] One Algorithm to Align Them All](../../CVPR2026/image_generation/one_algorithm_to_align_them_all.md)

</div>

<!-- RELATED:END -->
