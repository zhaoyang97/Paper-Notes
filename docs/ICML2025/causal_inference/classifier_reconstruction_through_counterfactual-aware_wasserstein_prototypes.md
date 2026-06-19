---
title: >-
  [论文解读] Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes
description: >-
  [ICML 2025][因果推理][反事实解释] 提出利用 Wasserstein 重心将原始样本与反事实样本融合为类别原型，从而在有限查询预算下高保真地重建目标二分类器，有效缓解了朴素使用反事实样本导致的决策边界偏移问题。 反事实解释（Counterfactual Explanation）通过找到使模型预测翻转的最小输入改…
tags:
  - "ICML 2025"
  - "因果推理"
  - "反事实解释"
  - "Wasserstein 重心"
  - "模型重建"
  - "原型分类"
  - "决策边界"
---

# Classifier Reconstruction Through Counterfactual-Aware Wasserstein Prototypes

**会议**: ICML 2025  
**arXiv**: [2512.10878](https://arxiv.org/abs/2512.10878)  
**代码**: 无  
**领域**: 因果推断  
**关键词**: 反事实解释, Wasserstein 重心, 模型重建, 原型分类, 决策边界

## 一句话总结

提出利用 Wasserstein 重心将原始样本与反事实样本融合为类别原型，从而在有限查询预算下高保真地重建目标二分类器，有效缓解了朴素使用反事实样本导致的决策边界偏移问题。

## 研究背景与动机

反事实解释（Counterfactual Explanation）通过找到使模型预测翻转的最小输入改变，为用户提供可操作的洞察（如"如果收入增加 1 万就能通过贷款审核"）。然而，反事实也可能暴露模型内部结构：攻击者可以利用反事实查询来训练代理模型（surrogate model），从而实现模型提取（model extraction），威胁 MLaaS 平台的知识产权。另一方面，模型重建也有正面用途——申请人可以在不正式提交敏感数据的情况下评估审批概率。

**现有方法的核心问题**：

**决策边界偏移**：此前方法（Aïvodji et al., 2020）将反事实直接当作带标签的训练样本。由于反事实自然位于决策边界附近，在类别不平衡或单侧反事实（只有"拒绝→接受"方向的反事实）的场景下，代理模型的决策边界会显著偏离目标模型。

**过拟合**：Counterfactual Clamping 损失（Dissanayake & Dutta, 2024）虽然通过修改交叉熵损失来处理反事实，但在查询样本有限时容易过拟合，尤其是使用复杂神经网络作为代理模型时。

**双侧查询限制**：双侧反事实查询可以缓解偏移，但现实中往往只能获得单侧反事实（例如仅有被拒绝申请的反事实）。

**本文动机**：反事实样本虽然信息丰富（靠近决策边界），但不能代替两个类别的典型样本。如何在有限查询下，既利用反事实的信息又避免边界偏移？作者的核心洞察是：把反事实视为两个类别的"软样本"（标签 0.5），与原始数据一起通过最优传输理论计算类别原型。

## 方法详解

### 整体框架

本方法的整体流程如下：

1. **数据组织**：将数据分为三类——类别 0 样本集 $\mathcal{D}_0$、类别 1 样本集 $\mathcal{D}_1$、反事实样本集 $\mathcal{D}_{cf}$（标签为 0.5）。
2. **Wasserstein 重心计算**：为每个类别 $c \in \{0, 1\}$ 计算融合了原始样本和反事实样本的重心分布 $\mathbb{Q}_c$。
3. **分类推断**：对新样本计算其到两个重心的 Wasserstein 距离，取距离更近者为预测类别。

目标模型 $m: \mathbb{R}^d \to [0,1]$ 是一个二分类器，输出概率分数，以 0.5 为阈值决定类别。反事实生成器 $g_m$ 仅在模型预测为类别 0 时触发（即单侧反事实设定）。目标是学习代理模型 $\hat{m}$ 以最少查询量复现目标模型行为。

### 关键设计

#### 1. 反事实的软标签处理

传统做法直接将反事实作为目标类别的训练样本，但这会引入边界偏移。本文将反事实赋予软标签 0.5：

$$\mathcal{Y} = \{0, 0.5, 1\}$$

这个设计有两个好处：

- **缓解类别不平衡**：反事实不再单独归入某一类别，避免了一侧样本过多导致边界偏移。
- **利用边界信息**：0.5 标签自然地表达了反事实位于两类之间的语义。

#### 2. 基于 Wasserstein 重心的类别原型

对每个类别 $c$，计算一个重心分布 $\mathbb{Q}_c$，使其同时靠近该类别的原始分布 $\mathbb{P}_c$ 和反事实分布 $\mathbb{P}_{cf}$：

$$\mathbb{Q}_c = \arg\min_{\mathbb{Q} \in \mathbb{P}(\mathcal{X})} \left( W_2^2(\mathbb{Q}, \mathbb{P}_c) + \lambda_c W_2^2(\mathbb{Q}, \mathbb{P}_{cf}) \right)$$

其中 $\lambda_c = 0.5$ 平衡了原始分布和反事实分布的影响。这里使用 2-Wasserstein 距离作为分布间度量，它考虑了底层空间的几何结构，比 KL 散度等更适合捕捉分布形状。

与传统原型网络（Prototypical Network, Snell et al., 2017）使用嵌入空间均值作为原型不同，Wasserstein 重心保留了类内的分布变异性，在小样本下提供了更鲁棒的类别表示。

#### 3. 对称性正则化

为确保反事实在两个类别原型之间保持对称位置（即等距于两个重心），引入正则项：

$$\mathcal{R}(\mathbb{Q}_0, \mathbb{Q}_1) = \left( W_2(\mathbb{Q}_0, \mathbb{P}_{cf}) - W_2(\mathbb{Q}_1, \mathbb{P}_{cf}) \right)^2$$

该正则惩罚反事实到两个重心距离的不对称性，鼓励决策边界恰好通过反事实分布的中心区域，与目标模型的决策边界对齐。

#### 4. 带 margin 的分类规则

对测试样本 $x$，用其 Dirac 分布 $\delta_x$ 到两个重心的 Wasserstein 距离进行分类：

$$\hat{y}(x) = \begin{cases} 0 & \text{if } W_2(\delta_x, \mathbb{Q}_0) < W_2(\delta_x, \mathbb{Q}_1) - \tau \\ 1 & \text{if } W_2(\delta_x, \mathbb{Q}_1) < W_2(\delta_x, \mathbb{Q}_0) - \tau \end{cases}$$

阈值参数 $\tau \geq 0$ 引入了一个间隔，避免在决策边界附近做出过于自信的预测。

### 损失函数 / 训练策略

整体优化目标将类别重心损失与对称性正则合并：

$$\min_{\mathbb{Q}_0, \mathbb{Q}_1} \sum_{c \in \{0,1\}} \left( W_2^2(\mathbb{Q}_c, \mathbb{P}_c) + \lambda_c W_2^2(\mathbb{Q}_c, \mathbb{P}_{cf}) \right) + \gamma \mathcal{R}(\mathbb{Q}_0, \mathbb{Q}_1)$$

- $\lambda_c = 0.5$：反事实对两个类别原型的影响权重
- $\gamma > 0$：对称性正则强度
- 使用 Python 最优传输库 POT (Flamary et al., 2021) 中的 free-support barycenter 算法迭代求解

训练策略要点：

- **无需训练神经网络**：不同于 Baseline 2 需要训练代理分类器，本方法直接通过优化问题计算 Wasserstein 重心，避免了神经网络训练带来的过拟合风险。
- **低查询效率**：方法仅需几百次查询即可达到高保真度，适合查询预算受限的场景。

## 实验关键数据

### 主实验

在 Adult Income、COMPAS、DCCC、HELOC 四个表格数据集上，与两个 SOTA 方法对比（目标分类器为逻辑回归，反事实由 MCCF 方法生成）：

| 数据集 | 查询量 | Baseline 1 | Baseline 2 | **本文** | 提升 (vs B2) |
|--------|--------|-----------|-----------|---------|-------------|
| Adult Income | 500 | 91±3.2 | 94±3.2 | **96±2.5** | +2.0 |
| COMPAS | 500 | 92±3.2 | 94±2.0 | **96±2.3** | +2.0 |
| DCCC | 500 | 89±8.9 | 91±0.9 | **97±1.5** | +6.0 |
| HELOC | 500 | 91±4.7 | 93±2.2 | **95±2.0** | +2.0 |
| Adult Income | 300 | 87±3.8 | 90±3.8 | **93±3.2** | +3.0 |
| COMPAS | 300 | 88±3.8 | 90±2.6 | **94±3.0** | +4.0 |
| DCCC | 300 | 85±9.5 | 87±1.5 | **93±2.1** | +6.0 |
| HELOC | 300 | 87±5.3 | 89±2.8 | **93±2.6** | +4.0 |

### 消融实验

不同反事实生成方法对 Adult 数据集保真度的影响：

| 反事实方法 | 特性 | 保真度表现 | 说明 |
|-----------|------|----------|------|
| MCCF (Wachter) | L2 距离最小化 | 高 | 默认方法，表现稳健 |
| L1-Sparse | 稀疏变化 | 高 | 特征改变最少 |
| DiCE | 可操作性约束 | 高 | 支持不可变特征 |
| Nearest Neighbor | 现实性（近邻） | 高 | 贴近数据流形 |
| C-CHVAE | VAE 生成 | 较低 | 生成模型在表格数据上表现较弱 |
| ROAR | 鲁棒性 | 高 | 对模型漂移具有鲁棒性 |

### 关键发现

1. **查询量越少，优势越大**：从 500→300 查询，本方法的保真度下降幅度远小于两个基线，展现了在低查询预算下的优越性。
2. **复杂代理模型下更稳定**：当 Baseline 2 使用更复杂的神经网络作为代理模型时，其保真度反而下降（受限查询下过拟合），而本方法依赖 Wasserstein 重心无此问题。
3. **反事实质量影响显著**：越贴近数据流形的反事实（如最近邻方法）效果越好，基于 VAE 的 C-CHVAE 因生成质量有限而表现较差。
4. **方差更低**：本方法在多数设定下标准差小于两个基线（如 DCCC 数据集从 8.9 降至 1.5），说明结果更稳定。

## 亮点与洞察

1. **视角新颖**：将反事实解释的"副作用"（暴露模型信息）转化为一个正式的优化问题，并巧妙利用最优传输理论给出优雅解法。
2. **无需训练神经网络**：不同于主流的代理模型训练范式，本方法直接计算分布重心作为原型，在低数据场景下天然避免了过拟合。
3. **理论与实践统一**：软标签 0.5 + Wasserstein 重心 + 对称性正则三者有机结合，每个组件都有清晰的几何直觉和理论支撑。
4. **反事实生成方法的全面比较**：系统地评估了 6 种反事实生成方法对模型重建的影响，这在文献中较为少见。

## 局限与展望

1. **仅限二分类**：当前方法围绕二分类器设计，扩展到多分类需要处理更复杂的多类原型和边界。
2. **仅验证表格数据**：实验全部在表格型数据集上进行，对图像/文本等高维数据的有效性未知。
3. **目标模型限制**：实验目标模型为逻辑回归，对更复杂的目标分类器（如深度神经网络）的重建效果有待验证。
4. **Wasserstein 距离计算成本**：在高维空间中精确计算 Wasserstein 距离计算量大，可考虑使用 Sliced Wasserstein 或 Sinkhorn 距离等近似方案。
5. **单侧反事实假设固定**：仅考虑了 class 0→class 1 方向的反事实，可探索混合单/双侧的更灵活设定。

## 相关工作与启发

- **Aïvodji et al. (2020)**：首次提出利用反事实进行模型重建（Baseline 1），直接将反事实作为训练样本的朴素方案。
- **Dissanayake & Dutta (2024)**：提出 Counterfactual Clamping 损失改进代理训练（Baseline 2），但在低查询量下因过拟合而受限。
- **Prototypical Networks (Snell et al., 2017)**：原型网络在 few-shot learning 中使用类别均值作为原型的思想启发了本文，但本文用 Wasserstein 重心代替了简单均值。
- **最优传输 (Flamary et al., 2021, POT)**：提供了 Wasserstein 重心的高效实现。
- **启发**：本文的思路可迁移至联邦学习中的模型隐私保护分析——评估在何种信息共享条件下对手可以重建本地模型。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 将最优传输引入反事实模型重建是新颖结合 |
| 技术深度 | 4 | 数学推导严谨，框架设计合理 |
| 实验充分性 | 3 | 数据集偏少且限于表格数据和简单模型 |
| 写作质量 | 4 | 问题定义清晰，方法动机充分 |
| 实用性 | 3 | 限于二分类表格数据，应用场景有限 |
| **综合** | **3.5** | 方法优雅但实验规模和场景泛化需加强 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Adjusting Prediction Model Through Wasserstein Geodesic for Causal Inference](../../ICLR2026/causal_inference/adjusting_prediction_model_through_wasserstein_geodesic_for_causal_inference.md)
- [\[ICCV 2025\] A Visual Leap in CLIP Compositionality Reasoning through Generation of Counterfactual Sets](../../ICCV2025/causal_inference/a_visual_leap_in_clip_compositionality_reasoning_through_gen.md)
- [\[ICML 2025\] Learning Time-Aware Causal Representation for Model Generalization in Evolving Domains](learning_time-aware_causal_representation_for_model_generalization_in_evolving_d.md)
- [\[ACL 2025\] Counterfactual Explanations for Aspect-Based Sentiment Analysis](../../ACL2025/causal_inference/counterfactual_explanations_for_aspect-based_sentiment_analysis.md)
- [\[ICML 2025\] Exogenous Isomorphism for Counterfactual Identifiability](exogenous_isomorphism_for_counterfactual_identifiability.md)

</div>

<!-- RELATED:END -->
