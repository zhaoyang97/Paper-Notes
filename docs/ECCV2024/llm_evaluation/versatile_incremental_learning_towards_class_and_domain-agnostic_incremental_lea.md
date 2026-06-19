---
title: >-
  [论文解读] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning
description: >-
  [ECCV 2024][LLM评测][增量学习] 首次定义 Versatile Incremental Learning (VIL) 场景——后续任务的类别或领域增量类型未知，并提出 ICON 框架，通过 CAST 损失控制学习方向避免与历史任务冲突、IC 增量分类器动态扩展输出节点处理跨域同类覆写问题，在三个基准上全面超越现有 CIL/DIL 方法。
tags:
  - "ECCV 2024"
  - "LLM评测"
  - "增量学习"
  - "灾难性遗忘"
  - "类别-领域联合增量"
  - "适应偏移控制"
  - "增量分类器"
---

# Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning

**会议**: ECCV 2024  
**arXiv**: [2409.10956](https://arxiv.org/abs/2409.10956)  
**代码**: [KHU-AGI/VIL](https://github.com/KHU-AGI/VIL)  
**领域**: LLM评测  
**关键词**: 增量学习, 灾难性遗忘, 类别-领域联合增量, 适应偏移控制, 增量分类器

## 一句话总结

首次定义 Versatile Incremental Learning (VIL) 场景——后续任务的类别或领域增量类型未知，并提出 ICON 框架，通过 CAST 损失控制学习方向避免与历史任务冲突、IC 增量分类器动态扩展输出节点处理跨域同类覆写问题，在三个基准上全面超越现有 CIL/DIL 方法。

## 研究背景与动机

增量学习 (IL) 旨在从顺序到达的任务中持续积累知识，同时克服灾难性遗忘。现有 IL 场景分为两类：

- **Class IL (CIL)**：任务间类别不同但领域相同（如持续学习新类别的物体）
- **Domain IL (DIL)**：任务间领域不同但类别相同（如相同类别在不同天气/环境下）

**核心问题**：现有方法强假设后续任务只会增加类别或只会增加领域，但现实中二者可能随机出现。例如自动驾驶中，模型既要学新类别物体，也要适应新的环境条件，且无法预知下一个任务会增加什么。

**VIL 带来的新挑战**：

**类内领域混淆 (Intra-class domain confusion)**：同一类在不同领域下的分布差异导致分类器权重被覆写

**跨域类别混淆 (Inter-domain class confusion)**：DIL 方法假设类别不变，遇到新类别时无法适应

**分类器漂移**：学习已有类别的新领域时，分类器权重被严重覆写

## 方法详解

### 整体框架

ICON (Incremental Classifier with Adaptation Shift cONtrol) 基于冻结的 ViT backbone + 可训练 adapter 架构，包含两个核心组件：

1. **CAST (Cluster-based Adaptation Shift conTrol)**：基于适配器权重偏移的聚类正则化，控制学习方向
2. **IC (Incremental Classifier)**：根据类别难度动态扩展分类器输出节点

### 关键设计

1. **CAST 损失 — 基于聚类的适应偏移控制**：

核心观察：当 IL 类型在连续任务间发生变化时（如从 CIL 切换到 DIL），adapter 权重的偏移方向差异显著。若不加约束，模型在不同类型任务间的学习方向会相互冲突。

具体机制：
- 每个任务学习前后记录 adapter 权重差 $V_{t-1} = A_{t-1}^{after} - A_{t-1}^{prev}$，存入偏移池 (shift pool)
- 对偏移池中所有历史偏移做 K-Means 聚类
- 训练当前任务时，计算当前迭代的偏移 $V_t^i = A_t^i - A_t^{prev}$
- 找到 $V_t^i$ 所属的聚类 $S_t^i$，将其他聚类 $S_t^{i'}$ 中的偏移视为"不同类型"的历史学习方向
- 正则化当前偏移与不同聚类中的偏移正交：

$$\mathcal{L}_{CAST} = \sum_j w_j \cdot \frac{V_t^i \cdot V_j}{\|V_t^i\| \|V_j\|}$$

其中 $w_j = \frac{\|V_t^i - V_j\|_2}{\sum_{V_k \in S_t^{i'}} \|V_t^i - V_k\|_2}$，$V_j \in S_t^{i'}$

权重 $w_j$ 使距离更远的历史偏移获得更大的正则化权重，从而差异性地控制学习方向。偏移差的数学本质就是累积梯度（由梯度下降公式推导得出），因此偏移方向等价于学习方向。

2. **IC — 增量分类器**：

针对 VIL 中"同一类别出现在不同领域"时分类器权重被覆写的问题，IC 根据需要动态扩展分类器输出节点：

- **动态阈值决策**：对每个类别 $i$，计算在已学领域上的平均准确率与新领域准确率的差距：

$$\delta_i = \tanh(p_i), \quad p_i = \gamma \cdot \frac{\frac{1}{|D^{prev}|}\sum_{d \in D^{prev}} Acc(C_i^d) - Acc(C_i^{d_{new}})}{\frac{1}{|D^{prev}|}\sum_{d \in D^{prev}} Acc(C_i^d)}$$

若新领域准确率显著低于历史平均（即该类在新领域"困难"），则为其新增输出节点。

- **节点选择策略**：推理时对同一类的多个节点取 max logit（基于能量模型理论——低能量=高 logit 的节点更 in-distribution）
- **知识蒸馏**：对未被选中的旧节点，用 KL 散度损失从前一任务的分类器蒸馏知识

### 损失函数 / 训练策略

$$\mathcal{L}_{Total} = \beta \mathcal{L}_{CAST} + \mathcal{L}_{IC}$$

其中 $\mathcal{L}_{IC} = \mathcal{L}_{CE}(O^t, y) + \alpha \mathcal{L}_{KL}(O^t, O^{t-1})$。ViT 参数冻结，仅更新 adapter 参数和分类器参数。

## 实验关键数据

### 主实验

VIL 场景下的 Average Accuracy (%)：

| 方法 | iDigits | CORe50 | DomainNet | 平均 |
|------|---------|--------|-----------|------|
| Fine-tuning | 19.89 | 14.04 | 20.35 | 18.09 |
| L2P | 59.07 | 64.85 | 48.98 | 57.63 |
| CODA-Prompt | 63.30 | 69.28 | 49.45 | 60.68 |
| LAE | 59.34 | 77.11 | 49.01 | 61.82 |
| **ICON (Ours)** | **75.11** | **83.18** | **53.37** | **70.55** |

跨所有场景 (CIL+DIL+VIL) 的平均准确率：

| 方法 | iDigits | CORe50 | DomainNet |
|------|---------|--------|-----------|
| CODA-Prompt | 70.95 | 74.52 | 58.73 |
| LAE | 68.12 | 75.89 | 55.26 |
| **ICON** | **77.15** | **84.34** | **59.74** |

### 消融实验

VIL 场景下 CAST 和 IC 的消融（平均 Avg. Acc %）：

| CAST | IC | iDigits | CORe50 | DomainNet | 平均 |
|------|-----|---------|--------|-----------|------|
| ✗ | ✗ | 59.34 | 77.11 | 49.01 | 61.82 |
| ✓ | ✗ | 68.34 | 79.20 | 50.56 | 66.03 |
| ✗ | ✓ | 66.97 | 81.13 | 51.60 | 66.57 |
| ✓ | ✓ | **75.11** | **83.18** | **53.37** | **69.98** |

IC 的进一步分解（iDigits VIL）：

| 节点扩展 | 知识蒸馏 | Avg. Acc | Forgetting |
|---------|---------|----------|------------|
| ✗ | ✗ | 59.34 | 29.32 |
| ✓ | ✗ | 63.10 | 25.50 |
| ✓ | ✓ | **66.97** | **14.32** |

### 关键发现

- CAST 和 IC 各自贡献显著（平均约 +4%），且组合使用有协同效应
- 聚类数 K=2 对短序列 (iDigits, 20 tasks) 最佳，K=3 对长序列 (CORe50, 40 tasks) 最佳
- 仅做节点扩展（不蒸馏）就能带来可观提升，说明输出节点分离本身对解决权重覆写问题很关键
- ICON 在 Cross-Domain IL 中同样 SOTA（平均 72.88% vs CODA-P 69.08%），证明方法在 VIL 子场景中也有效

## 亮点与洞察

1. **场景定义有价值**：VIL 统一了 CIL 和 DIL，更贴近真实世界。现有方法在 VIL 上的显著退化说明这是一个值得研究的问题
2. **CAST 的设计巧妙**：通过权重偏移（=累积梯度）间接度量学习方向，再用聚类区分"相似"和"不同"的历史任务，把正则化集中在"不同类型"任务上——避免了一刀切的正则化
3. **IC 的低成本扩展**：不像 DER/DyTox 那样扩展整个网络，仅扩展分类器最后一层的部分节点，成本极低

## 局限与展望

1. VIL 场景中每个任务的类别数和领域数是固定的，论文也承认"变化的类别/领域数量是更现实的场景"
2. CAST 依赖 K-Means 聚类，聚类数 K 需要调参，且对不同数据集的最优值不同
3. IC 的动态阈值计算需要在各领域上评估分类器精度，引入额外计算开销
4. 仅在 ViT+adapter 架构上验证，对其他架构（如 CNN-based）的适用性未知
5. 三个数据集中领域区分较为明显（如不同手写体、不同风格），更细微的领域变化可能挑战更大

## 相关工作与启发

- **CODA-Prompt / DualPrompt / L2P**：Prompt-based IL 方法，在 CIL 上表现好但在 VIL 上退化明显
- **S-Prompts**：DIL 特化方法，在 VIL 上因缺乏类别增量处理而大幅退化
- **DER / DyTox**：模型扩展方法，未考虑同一类跨域覆写问题
- **EWC / LwF**：经典正则化方法，在 VIL 中效果有限

## 评分

- 新颖性: ⭐⭐⭐⭐ — VIL 场景定义有实际意义，CAST 和 IC 都是合理的新设计
- 实验充分度: ⭐⭐⭐⭐⭐ — 三个数据集、三种场景、多个基线、详细消融和分析
- 写作质量: ⭐⭐⭐⭐ — 问题动机阐述清晰，方法推导完整；数据表格较多但组织良好
- 综合价值: ⭐⭐⭐⭐ — 开创了 VIL 新场景，方法在各场景上全面 SOTA，兼具理论贡献和实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Image-Feature Weak-to-Strong Consistency: An Enhanced Paradigm for Semi-Supervised Learning](image-feature_weak-to-strong_consistency_an_enhanced_paradigm_for_semi-supervise.md)
- [\[ECCV 2024\] Instance-dependent Noisy-label Learning with Graphical Model Based Noise-rate Estimation](instance-dependent_noisy-label_learning_with_graphical_model_based_noise-rate_es.md)
- [\[ICLR 2026\] In-Context Learning for Pure Exploration](../../ICLR2026/llm_evaluation/in-context_learning_for_pure_exploration.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](../../ICML2025/llm_evaluation/sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[ECCV 2024\] Learn from the Learnt: Source-Free Active Domain Adaptation via Contrastive Sampling and Visual Persistence](learn_from_the_learnt_source-free_active_domain_adaptation_via_contrastive_sampl.md)

</div>

<!-- RELATED:END -->
