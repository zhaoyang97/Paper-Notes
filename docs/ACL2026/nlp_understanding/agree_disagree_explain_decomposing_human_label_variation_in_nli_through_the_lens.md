---
title: >-
  [论文解读] Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations
description: >-
  [ACL 2026 Findings][NLP理解][标注分歧] 将LiTEx推理分类法从"标签一致下的解释变异"扩展到"标签不一致"场景，发现标注者可能标签不同但推理类似，推理类别的一致性比标签一致性更好地反映解释的语义相似度。 领域现状：NLI数据集中普遍存在标注者分歧，理解这些分歧对构建可靠的NLU系统至关重要…
tags:
  - "ACL 2026 Findings"
  - "NLP理解"
  - "标注分歧"
  - "自然语言推理"
  - "LiTEx分类法"
  - "推理策略"
  - "人类标注变异"
---

# Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations

**会议**: ACL 2026 Findings  
**arXiv**: [2510.16458](https://arxiv.org/abs/2510.16458)  
**代码**: 无  
**领域**: NLI / Annotation Analysis  
**关键词**: 标注分歧, 自然语言推理, LiTEx分类法, 推理策略, 人类标注变异

## 一句话总结

将LiTEx推理分类法从"标签一致下的解释变异"扩展到"标签不一致"场景，发现标注者可能标签不同但推理类似，推理类别的一致性比标签一致性更好地反映解释的语义相似度。

## 研究背景与动机

**领域现状**：NLI数据集中普遍存在标注者分歧，理解这些分歧对构建可靠的NLU系统至关重要。基于解释的方法通过分析标注者决策背后的推理来揭示分歧的本质。

**现有痛点**：LiTEx分类法将自由文本解释归类为8种推理策略，但此前仅用于分析"标签一致、解释不同"的within-label变异，忽略了标签本身的不一致。

**核心矛盾**：标签不一致可能掩盖推理一致（同样的推理导致不同标签），而标签一致也可能掩盖推理分歧（不同推理碰巧得到同一标签）。仅看标签无法揭示真实的认知分歧。

**本文目标**：将LiTEx扩展到标签变异场景，从标签、解释类别和解释文本相似度三个维度分析NLI标注变异。

**切入角度**：在LiveNLI和VariErr两个带解释的NLI数据集上标注LiTEx类别，追踪个体标注者的标签偏好和推理策略偏好。

**核心 idea**：推理类别的一致性比标签一致性本身更能反映解释之间的语义相似度，说明应更关注推理过程而非最终标签。

## 方法详解

### 整体框架

在三个数据集（e-SNLI, LiveNLI, VariErr）上应用LiTEx分类法标注解释，然后从三个维度分析变异：(1) NLI标签一致性；(2) 推理类别一致性（LiTEx）；(3) 解释文本的语义相似度。通过追踪个体标注者揭示行为模式。

### 关键设计

**1. LiTEx 分类法的跨数据集扩展：把只在 e-SNLI 上验证过的推理分类搬到标签会变的场景**

LiTEx 原本只在 e-SNLI 上开发，且只分析"标签一致、解释不同"的 within-label 变异。本文把它迁移到 LiveNLI 和 VariErr 两个标签本身就有分歧的数据集，由经过训练的标注者对所有自由文本解释打类别。8 种推理类别分为两组：文本型（共指、句法、语义、语用、信息缺失、逻辑冲突）和世界知识型（事实知识、推理知识）。这样做一来检验 LiTEx 能否跨数据集泛化，二来把它的适用范围从"标签一致"推广到"标签不一致"，为后面的多维度对比铺好统一的标注基础。

**2. 多维度一致性分析：在同一条 NLI 实例上同时比标签、推理类别和解释文本三种一致性**

单看 NLI 标签无法分辨分歧到底来自哪里——同样的推理可能落到不同标签，不同的推理也可能碰巧落到同一标签。本文对每条实例的标注者并排比较三个维度的一致性：(1) NLI 标签是否一致；(2) LiTEx 推理类别是否一致；(3) 解释文本的语义相似度。由此能把"标签不同但推理类别相同"和"标签相同但推理类别不同"这两种被标签掩盖的情形显式拆出来，揭示标签一致性与推理一致性之间的非对称关系。

**3. 个体标注者追踪：把分歧归因到具体的人，找系统性偏好**

标注分歧未必只来自文本歧义，也可能来自标注者各自固定的推理风格。本文追踪 LiveNLI 和 VariErr 中各 4 位标注者的标签分布和推理类别偏好，逐人统计他们的倾向（如 VariErr 标注者 2 有近 60% 偏 neutral）。这把群体层面的分歧落到个体层面，说明有些"分歧"其实是稳定的个人偏好，对标注流程设计有直接启示。

### 损失函数 / 训练策略

本文为实证分析研究，不涉及模型训练。标注者间一致度用Cohen's Kappa衡量（LiveNLI κ=0.828，VariErr κ=0.792）。

## 实验关键数据

### 主实验

| 数据集 | 标注数量 | κ值 | 主要发现 |
|--------|---------|-----|---------|
| e-SNLI | 原有标注 | - | 推理知识和信息缺失是主要类别 |
| LiveNLI | 1404对 | 0.828 | 信息缺失偏向neutral标签 |
| VariErr | 1933对 | 0.792 | 信息缺失是最频繁类别 |

### 关键发现

| 发现 | 说明 |
|------|------|
| 标签不一致但推理一致 | 标注者用相同推理策略但得出不同标签，说明分歧在判断而非理解 |
| 推理类别-标签共现稳定 | 尽管数据集间绝对分布不同，推理类别对应的标签分布高度一致 |
| 个体标签偏好显著 | 如VariErr标注者2有近60%的neutral偏好，w7有52%的neutral偏好 |
| 推理相似度 > 标签相似度 | 推理类别一致性更好地预测解释的语义相似度 |

### 关键发现
- LiTEx分类法跨数据集泛化良好，类别-标签共现模式在三个数据集上高度一致
- "标签不一致但解释相似"的情况频繁出现，表面分歧可能掩盖深层理解的一致
- 信息缺失（Absence of Mention）类别与neutral标签的强关联在所有数据集中一致
- 个体标注者展现出稳定的标签偏好和推理策略偏好

## 亮点与洞察
- "标签不代表理解"的核心发现挑战了将标签视为ground truth的传统做法
- 解释不仅是可解释性的工具，更是理解标注分歧的窗口
- 个体标注者追踪揭示了系统性的个人偏好，这对标注流程设计有直接启示

## 局限与展望
- 仅覆盖英语NLI数据集，跨语言泛化性未验证
- LiTEx的8个类别可能无法覆盖所有推理类型
- 标注者数量有限（每个数据集仅追踪4位），统计效力受限
- 未来可将方法扩展到其他NLU任务的标注分歧分析

## 相关工作与启发
- **vs 传统标注一致性研究**: 不仅看标签，还看推理过程，提供更细粒度的分析
- **vs ChaosNLI/AmbiEnt**: 这些数据集关注分歧的量化，本文关注分歧的认知来源
- **vs LiTEx原始工作**: 将分析范围从within-label扩展到label variation

## 评分
- 新颖性: ⭐⭐⭐⭐ 将推理分析扩展到标签变异场景的视角新颖
- 实验充分度: ⭐⭐⭐ 分析深入但规模较小
- 写作质量: ⭐⭐⭐⭐ 案例说明清晰，分析层次分明
- 价值: ⭐⭐⭐⭐ 对标注流程和数据质量研究有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Exploring Concreteness Through a Figurative Lens](exploring_concreteness_through_a_figurative_lens.md)
- [\[ACL 2026\] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/nlp_understanding/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] TruthSplit: Operationalizing Conditional Validity in Arguments Through Multi-Perspective Reasoning](truthsplit_operationalizing_conditional_validity_in_arguments_through_multi-pers.md)

</div>

<!-- RELATED:END -->
