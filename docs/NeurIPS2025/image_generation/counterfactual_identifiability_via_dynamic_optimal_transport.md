---
title: >-
  [论文解读] Counterfactual Identifiability via Dynamic Optimal Transport
description: >-
  [NeurIPS 2025][图像生成][counterfactual identification] 利用动态最优传输 (dynamic OT) 理论，首次解决了高维多变量 Markovian SCM 中反事实的可辨识性问题——证明 OT flow 机制产生唯一的单调保序反事实传输映射，并扩展至非 Markovian 设置（IV/BC/FC 准则）。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "counterfactual identification"
  - "optimal transport"
  - "flow matching"
  - "structural causal model"
  - "monotone transport map"
---

# Counterfactual Identifiability via Dynamic Optimal Transport

**会议**: NeurIPS 2025  
**arXiv**: [2510.08294](https://arxiv.org/abs/2510.08294)  
**代码**: 待确认  
**领域**: 因果推断 / 生成模型 / 最优传输  
**关键词**: counterfactual identification, optimal transport, flow matching, structural causal model, monotone transport map

## 一句话总结
利用动态最优传输 (dynamic OT) 理论，首次解决了高维多变量 Markovian SCM 中反事实的可辨识性问题——证明 OT flow 机制产生唯一的单调保序反事实传输映射，并扩展至非 Markovian 设置（IV/BC/FC 准则）。

## 研究背景与动机

**领域现状**：深度生成模型（VAE、扩散模型、flow）越来越多被用于参数化结构因果模型 (SCM) 来推断反事实，但这些方法缺少可辨识性保证——对同一观测数据可能存在多个 observationally equivalent 的模型给出不同反事实答案。

**现有痛点**：(a) Pearl 强调反事实查询必须满足可辨识性要求，否则无法做出可靠的因果声明；(b) 经典的符号化辨识方法（Tian & Pearl, Shpitser & Pearl）未被推广到高维变量；(c) Nasr-Esfahany et al. (2023) 建立了双射机制的反事实辨识，但对多维变量 ($d>1$) 在 Markovian 设置下如何推广单调性条件是未解决的开放问题。

**核心矛盾**：多维变量下，双射性不足以保证反事实辨识——由于旋转对称性，同一观测分布可由无穷多双射机制产生。需要一个多维单调性的正确推广来打破对称性。

**切入角度**：利用 Brenier 定理——在标准正则性条件下，最优传输映射 $T = \nabla \phi$ 是唯一的且单调的（由凸函数的梯度给出）。将此与 SCM 的因果机制联系，证明 OT flow 机制天然满足多维反事实辨识所需的单调性。

## 方法详解

### 问题设定

考虑 SCM $\mathfrak{C} = (\mathbf{U}, \mathbf{X}, \mathcal{F})$，聚焦于某个多维 ($d>1$) 变量 $X$ 的因果机制 $f$：$X = f(\mathbf{PA}, U)$，其中 $\dim(X) = \dim(U) = d$。

反事实查询："给定观察到 $X=x$（父变量 $\mathbf{PA}=\mathbf{pa}$），若父变量变为 $\mathbf{pa}^*$，$X$ 会是什么？"

反事实传输映射：$T^*(\mathbf{pa}^*, \mathbf{pa}, x) = f(\mathbf{pa}^*, f^{-1}(\mathbf{pa}, x))$

### 核心理论

**Definition 4.3（单调算子）**：映射 $f$ 在 $u$ 上单调，如果：
$$\langle f(\mathbf{pa}, u_1) - f(\mathbf{pa}, u_2), u_1 - u_2 \rangle \geq 0, \quad \forall u_1, u_2$$

**Proposition 4.4**：若机制 $f$ 在 $u$ 上单调，则反事实传输映射 $T^*$ 在 $x$ 上单调——保证在给定干预下，事实结果的排序在反事实中保持不变（rank preservation）。

**Lemma 4.6（唯一且单调的 Dynamic OT 机制）**：在 Markovian 设置 ($U \perp\!\!\!\perp \mathbf{PA}$) 下，令 $T$ 为 dynamic OT flow 的 time-1 映射，将 $P_U$ 推前到 $P_{X|\mathbf{PA}}$。在标准正则性条件下，存在凸函数 $\phi$ 使得 $T(u; \mathbf{pa}) = \nabla_u \phi(u; \mathbf{pa})$，且 $T$ 是单调的、几乎处处双射的、由 $(P_U, P_{X|\mathbf{PA}})$ 唯一确定的。

**Theorem 4.12（Markovian SCM 中的反事实辨识）**：设 $P_U$ 为 $[0,1]^d$ 上的均匀分布，$T$ 为 Lemma 4.6 中的 OT 映射，则反事实传输映射 $T^*$ 在 $x$ 上**严格单调**：

$$\langle T^*(\mathbf{pa}^*, \mathbf{pa}, x_1) - T^*(\mathbf{pa}^*, \mathbf{pa}, x_2), x_1 - x_2 \rangle > 0, \quad \forall x_1 \neq x_2$$

这保证了 $\mathcal{L}_3$-equivalence 辨识性——从观测数据恢复的反事实是唯一的。

### 非 Markovian 扩展

将理论扩展到三种标准因果准则下的非 Markovian 设置：
- **工具变量 (IV)**：利用 Lemma 4.6 的单调性，将 $d=1$ 的 IV 结果推广到 $d>1$
- **后门准则 (BC)**：双射性 + 充分变异性即够（继承 Nasr-Esfahany et al. 的结果）
- **前门准则 (FC)**：在类似 BC 的条件下证明辨识性（新结果）

### 实际推断方法：Flow Matching

用连续时间 flow 模型参数化因果机制，通过 flow matching 训练：

$$\min_{\theta} \int_0^1 \mathbb{E}_{X_1 \sim p_{\text{data}}} \left[\|v_t(X_t; \theta) - v_t^*(X_t | X_1)\|^2\right] dt$$

反事实推断流程（abduction-action-prediction）：
1. **Abduction**：沿 ODE 反向积分恢复外生噪声 $u = x - \int_0^1 v_t(x_t; \mathbf{pa}, \theta) dt$
2. **Action**：设定反事实父变量 $\mathbf{pa}^*$
3. **Prediction**：沿 ODE 正向积分 $x^* = u + \int_0^1 v_t(x_t; \mathbf{pa}^*, \theta) dt$

**Markovian Batch-OT Coupling**：修正标准 Batch-OT flow matching 中隐式破坏 $U \perp\!\!\!\perp \mathbf{PA}$ 独立性的问题——在每个固定的 $\mathbf{pa}$ 值下独立求解 OT 耦合。

## 实验关键数据

### 实验1：反事实椭圆生成（合成，有 ground truth）

| 方法 | NFE | μ_APE (%) ↓ (Markov) | μ_APE (%) ↓ (Front-door) |
|------|-----|----------------------|-------------------------|
| Baseline (Nasr-Esfahany) | - | 607 | - |
| EBM | 50 | 2.32 | 1.79 |
| Flow | 50 | 2.30 | 1.67 |
| **OT-EBM** | 2 | 1.21 | 1.64 |
| **OT-Flow** | 2 | **1.06** | **1.60** |
| Naive Batch-OT | 2 | 违反 Markov 假设，反事实不正确 | - |

- OT 变体仅用 2 个函数评估 (NFE) 即可实现 ~1% 误差，而基线方法需要 50 NFE
- 按理论预测，Markovian 设置下 OT map 显著优于普通 flow；Front-door 设置下双射性即足够

### 实验2：胸部 X 光反事实生成 (MIMIC-CXR, 192×192)

| 干预 | 指标 | 基线 (Ribeiro 2023) | Flow (Ours) |
|------|------|-------------------|-------------|
| do(Sex=s) | |Δ_AUC| ↓ | 0.370% | **0.173%** |
| do(Race=r) | |Δ_AUC| ↓ | 8.640% | **0.050%** |
| do(Age=a) | Δ_MAE ↓ | 0.288 yr | 0.333 yr |
| do(Disease=d) | |Δ_AUC| ↓ | 2.490% | **0.023%** |

- 在 Race 干预上改进显著（8.64% → 0.05%），因 OT 提供了更一致的反事实
- Markovian OT coupling 显著优于 naive OT flow baseline

## 亮点与洞察

- **理论贡献是核心价值**：首次解决了多维 Markovian 反事实辨识的开放问题，Brenier 定理 + SCM 的连接非常优雅
- **将 OT 的数学唯一性保证转化为因果推断的辨识性保证**——两个看似不相关的领域产生了深刻联系
- **Markovian Batch-OT coupling 的修正**是一个重要的细节贡献——指出了标准做法的隐含缺陷
- **rank preservation 的多维推广**：单调性保证了反事实不会产生 rank inversion，对公平性应用至关重要

## 局限与展望

- **正则性假设较强**：要求密度严格正、有界、定义域有界凸——排除了很多实际分布
- **OT 计算在高维的可扩展性**：Batch-OT 在高维需要大 batch size，计算成本高
- **反事实有效性指标的局限**：composition/effectiveness/reversibility 不等于辨识性——作者自己也承认
- **Markovian 假设在真实数据中难以验证**：MIMIC-CXR 的因果图是假设的，可能存在未观测混淆
- **先验分布 $P_U$ 的选择**：假设均匀分布或标准 Gaussian，但真实外生分布未知

## 相关工作对比

- **vs Nasr-Esfahany et al. (2023)**：他们的 spline flow 在 Markovian $d>1$ 下失败 (μAPE=607%)，本文 OT flow 解决了该问题
- **vs Pawlowski et al. (2020) / Ribeiro et al. (2023)**：这些工作用 VAE/扩散模型参数化 SCM 但缺少辨识性，本文提供理论基础
- **vs 经典符号化方法 (Tian & Pearl)**：经典方法不适用于高维变量，本文填补了这一空白
- **vs Brenier 理论**：将纯数学的 OT 唯一性结果应用于因果推断是新颖的跨学科贡献

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 解决了因果推断中的重要开放问题，OT-因果的理论联系原创性强
- 实验充分度: ⭐⭐⭐⭐ 合成实验有 ground truth 验证理论，真实数据实验展示实用性
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，但前置知识要求较高，可读性略受限
- 价值: ⭐⭐⭐⭐⭐ 为深度因果推断提供了急需的理论基础，影响范围广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[ICCV 2025\] The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation](../../ICCV2025/image_generation/the_curse_of_conditions_analyzing_and_improving_optimal_transport_for_conditiona.md)
- [\[NeurIPS 2025\] V-CECE: Visual Counterfactual Explanations via Conceptual Edits](v-cece_visual_counterfactual_explanations_via_conceptual_edits.md)
- [\[ICML 2026\] Pareto-Guided Optimal Transport for Multi-Reward Alignment](../../ICML2026/image_generation/pareto-guided_optimal_transport_for_multi-reward_alignment.md)
- [\[NeurIPS 2025\] LeapFactual: Reliable Visual Counterfactual Explanation Using Conditional Flow Matching](leapfactual_reliable_visual_counterfactual_explanation_using_conditional_flow_ma.md)

</div>

<!-- RELATED:END -->
