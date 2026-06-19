---
title: >-
  [论文解读] Continual Multimodal Contrastive Learning
description: >-
  [NeurIPS 2025][多模态VLM][持续学习] 本文首次形式化定义了持续多模态对比学习（CMCL）问题，提出双侧零空间梯度投影（DNS）方法，将新数据的梯度投影到不影响旧知识的子空间上，在 7 个数据集上实现了稳定性和可塑性的最佳平衡。 领域现状：多模态对比学习（MCL）通过对比学习将不同模态（视觉、音频、文本等）…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "持续学习"
  - "多模态对比学习"
  - "梯度投影"
  - "灾难性遗忘"
  - "模态绑定"
---

# Continual Multimodal Contrastive Learning

**会议**: NeurIPS 2025  
**arXiv**: [2503.14963](https://arxiv.org/abs/2503.14963)  
**代码**: [https://github.com/Xiaohao-Liu/CMCL](https://github.com/Xiaohao-Liu/CMCL)  
**领域**: 多模态VLM  
**关键词**: 持续学习, 多模态对比学习, 梯度投影, 灾难性遗忘, 模态绑定

## 一句话总结

本文首次形式化定义了持续多模态对比学习（CMCL）问题，提出双侧零空间梯度投影（DNS）方法，将新数据的梯度投影到不影响旧知识的子空间上，在 7 个数据集上实现了稳定性和可塑性的最佳平衡。

## 研究背景与动机

**领域现状**：多模态对比学习（MCL）通过对比学习将不同模态（视觉、音频、文本等）对齐到统一表示空间中，代表性方法如 CLIP、ImageBind、LanguageBind 等已展示出强大的跨模态表示能力。

**现有痛点**：现有 MCL 方法通常假设所有模态数据可以一次性收集并联合训练，但现实中多模态数据往往是分批采集的——新的模态对数据不断涌现——从头重训代价极高。直接在已有模型上持续训练又会导致灾难性遗忘，破坏已学习的模态对齐关系。

**核心矛盾**：传统持续学习方法（如 EWC、GEM、DER++）设计之初面向类增量或任务增量场景，无法处理 MCL 中独特的跨模态复杂性：任务目标始终是对比学习，但涉及的模态对不断变化。这些方法在 CMCL 场景中效果不佳，稳定性和可塑性之间的 trade-off 严重。

**本文目标** (1) 如何形式化定义持续多模态对比学习，明确稳定性和可塑性的数学定义；(2) 如何设计一种方法，在更新参数时既保证旧模态对的对齐不受干扰，又能有效学习新模态对。

**切入角度**：从梯度更新视角出发，观察到模型参数更新实质上是对全局参数矩阵的修改。如果能将梯度投影到不影响旧数据表示的子空间中，就能同时兼顾两个目标。这借鉴了零空间投影（null space projection）在单模态持续学习中的应用，但多模态场景需要解决从两个模态侧同时投影的问题。

**核心 idea**：将梯度从两个模态侧同时投影到各自旧数据特征的零空间中，使参数更新不干扰已学习的跨模态对齐。

## 方法详解

### 整体框架

给定一个预训练的模态绑定模型（如 ImageBind），在其上添加可训练的线性映射层。当新的模态对数据到来时，通过对比学习优化映射层参数。DNS 方法的核心是：在每步训练中，将梯度从两个"侧面"投影到特定子空间，使投影后的梯度更新不影响先前学习的模态对齐分数。

### 关键设计

1. **CMCL 问题形式化与双目标定义**:

    - 功能：为持续多模态对比学习提供严格的数学框架
    - 核心思路：定义稳定性为用当前模型在旧数据上计算的对齐分数 $\mathbf{A}_{t-1;t}^{m_1,m_2}$ 应与旧模型一致 $\mathbf{A}_{t-1;t-1}^{m_1,m_2}$；可塑性要求模型能从新模态对中有效学习。作者证明若全局参数更新被投影为 $\bar{\mathbf{W}} = \tilde{\mathbf{W}} - \mathbf{P}'\tilde{\mathbf{W}}\mathbf{P}$，其中 $\mathbf{P}'$ 和 $\mathbf{P}$ 是旧数据特征的空间投影器，则稳定性条件自动满足
    - 设计动机：CMCL 的目标函数（对比损失）保持一致，变化的是模态对——这需要专门的稳定性/可塑性定义，不同于分类持续学习

2. **双侧零空间梯度投影（DNS）**:

    - 功能：将全局稳定性条件分解为局部可操作的梯度投影
    - 核心思路：将全局参数更新展开为与两个模态各自梯度相关的三项。Theorem 4 证明可以将每个模态的梯度分别投影——模态 $m_1$ 的梯度投影为 $\Delta\mathbf{W}_t^{m_1} = \nabla\mathbf{W}_t^{m_1} - \tilde{\mathbf{P}}\nabla\mathbf{W}_t^{m_1}\mathbf{P}'$，其中 $\tilde{\mathbf{P}}$ 和 $\mathbf{P}'$ 分别基于对应旧模态的特征构建。投影后的梯度替换原始梯度，参数更新就不会干扰旧的跨模态对齐
    - 设计动机：直接在全局参数上做投影不可行，将投影分解到每个模态的梯度上，大幅降低实现复杂度

3. **多步骤与多模态对扩展**:

    - 功能：将两步两模态的理论推广到任意步数和任意模态对
    - 核心思路：通过递增式的无中心特征协方差矩阵 $\bar{\mathbf{Z}}_{<t}^m$ 累积所有历史步骤的特征信息，只需在每步结束时用 SVD 更新投影器。对不同模态对只需将未参与训练的模态梯度设为零
    - 设计动机：实际训练中涉及多步多模态对，需要高效方式维护历史信息而不存储旧数据，协方差矩阵的递推更新实现了无回放的持续学习

### 损失函数 / 训练策略

使用标准 CLIP-style InfoNCE 对比损失，AdamW 优化器（lr=0.0001, weight decay=0.001），batch size=64。零空间投影用截断 SVD 近似，最小特征值阈值 $\lambda_{\min}$ 为 0.01（ImageBind/UniBind）或 0.0001（LanguageBind）。

## 实验关键数据

### 主实验

在 7 个数据集（UCF101、ESC50、NYUDv2、VGGSound-S、Clotho、TVL、LLVIP）上评估，涉及 11 个训练步骤、7 种模态，使用三种 backbone。

| Backbone | 方法 | 分类 Acc | BWT_A | R@10 | BWT_R10 |
|----------|------|---------|-------|------|---------|
| ImageBind | Vanilla | 47.32 | -5.72 | 38.56 | -3.34 |
| ImageBind | Co2L | 50.13 | -3.74 | 38.66 | -1.77 |
| ImageBind | **DNS** | **52.52** | **-0.02** | **40.89** | **-1.07** |
| LanguageBind | Vanilla | 51.86 | -15.71 | 36.02 | -10.19 |
| LanguageBind | CILA | 59.30 | -2.63 | 40.48 | -1.09 |
| LanguageBind | **DNS** | **64.07** | **-0.09** | **42.44** | -3.00 |
| UniBind | C-FLAT | 51.25 | -4.36 | 40.51 | -1.48 |
| UniBind | **DNS** | **52.86** | **+0.31** | **41.44** | **-1.19** |

### 消融实验

| 分析维度 | 实验结果 | 说明 |
|---------|---------|------|
| 稳定性偏差 | 各步对齐分数偏差接近 0 | 实证验证 Theorem 5 的理论界 |
| 可塑性高阶项 | $o(\eta)/\eta < 0$ | 满足 Theorem 6 的可塑性条件 |
| 训练损失 | 各步均正常收敛 | 可塑性在约束下仍得以保持 |
| 训练时间开销 | DNS 仅增加 <1s | 相比 replay 方法更高效 |

### 关键发现

- DNS 的分类 BWT 接近 0 甚至为正（UniBind 上 +0.31），说明新知识学习甚至能促进旧任务——在持续学习中极为罕见
- LanguageBind 场景下遗忘最严重（Vanilla BWT=-15.71），DNS 将其控制到 -0.09，改善约 170 倍
- DNS 是无回放方法，不需存储旧数据，额外训练时间不到 1 秒

## 亮点与洞察

- **双侧投影的巧妙性**：不同于单模态持续学习只需单侧投影，CMCL 需同时考虑两个模态的特征空间。作者将全局稳定性条件巧妙分解为两个模态各自的局部梯度投影，理论优雅且实现简单
- **无回放设计**的效率优势显著。相比需 replay buffer 的方法，DNS 只维护一个递推更新的特征协方差矩阵，存储和计算开销极低
- **跨模态对的灵活扩展**：当新步骤涉及不同模态对时，只需将未参与模态的梯度设零，无需额外处理——在实际多模态场景中非常实用

## 局限与展望

- 实验在预训练模型上加线性层评估，未探索直接 fine-tune 整个编码器的场景，可能限制大规模训练中的适用性
- SVD 截断阈值对不同 backbone 需分别调参（0.01 vs 0.0001），缺乏自适应策略
- 仅考虑两两模态对的对齐，未涉及三模态或更多模态同时训练的场景
- 理论分析假设线性映射层，对非线性编码器的推广需要进一步研究

## 相关工作与启发

- **vs Adam-NSCL**: 同样使用零空间投影但只处理单模态持续学习，无法应对多模态对齐中的跨模态复杂性。DNS 的双侧投影是零空间方法在多模态场景下的自然推广
- **vs Co2L**: 用对比蒸馏防止遗忘，需旧数据回放，LanguageBind 场景 BWT 仍为 -5.79。DNS 无需回放且 BWT 接近 0
- **vs EWC**: 通过参数重要性权重防遗忘，隐式假设参数独立，多模态交叉影响下效果有限

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次形式化 CMCL 问题并提出专门方法，双侧投影有理论创新
- 实验充分度: ⭐⭐⭐⭐⭐ 7 数据集、3 backbone、多 baseline，稳定性/可塑性分析详尽
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，符号体系完整，但可读性受大量数学符号影响
- 价值: ⭐⭐⭐⭐ 填补了多模态对比学习与持续学习交叉领域的重要空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Generalized Contrastive Learning for Universal Multimodal Retrieval](generalized_contrastive_learning_for_universal_multimodal_re.md)
- [\[CVPR 2026\] Re-evaluating Continual VQA: Toward Fair and Robust Evaluation for Multimodal Continual Learning](../../CVPR2026/multimodal_vlm/re-evaluating_continual_vqa_toward_fair_and_robust_evaluation_for_multimodal_con.md)
- [\[CVPR 2026\] Towards Dynamic Modality Alignment in Multimodal Continual Learning](../../CVPR2026/multimodal_vlm/towards_dynamic_modality_alignment_in_multimodal_continual_learning.md)
- [\[ICLR 2026\] KeepLoRA: Continual Learning with Residual Gradient Adaptation](../../ICLR2026/multimodal_vlm/keeplora_continual_learning_with_residual_gradient_adaptation.md)
- [\[NeurIPS 2025\] BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)

</div>

<!-- RELATED:END -->
