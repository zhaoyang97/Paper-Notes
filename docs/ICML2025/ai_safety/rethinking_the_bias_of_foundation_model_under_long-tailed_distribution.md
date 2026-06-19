---
title: >-
  [论文解读] Rethinking the Bias of Foundation Model under Long-tailed Distribution
description: >-
  [ICML 2025][AI安全][长尾学习] 揭示基础模型微调在长尾任务上受"参数不平衡"（预训练数据偏差）和"数据不平衡"（下游数据偏差）的双重影响，发现参数不平衡更关键且无法被现有 logit 调整方法解决，提出基于因果后门调整的方法消除不完整语义因子的混杂效应，在三个长尾基准上平均提升约 1.67%。
tags:
  - "ICML 2025"
  - "AI安全"
  - "长尾学习"
  - "基础模型偏差"
  - "参数不平衡"
  - "因果推断"
  - "后门调整"
---

# Rethinking the Bias of Foundation Model under Long-tailed Distribution

**会议**: ICML 2025  
**arXiv**: [2501.15955](https://arxiv.org/abs/2501.15955)  
**代码**: 待确认  
**领域**: AI安全  
**关键词**: 长尾学习, 基础模型偏差, 参数不平衡, 因果推断, 后门调整

## 一句话总结

揭示基础模型微调在长尾任务上受"参数不平衡"（预训练数据偏差）和"数据不平衡"（下游数据偏差）的双重影响，发现参数不平衡更关键且无法被现有 logit 调整方法解决，提出基于因果后门调整的方法消除不完整语义因子的混杂效应，在三个长尾基准上平均提升约 1.67%。

## 研究背景与动机

### 1. 长尾学习现状

现实数据常服从长尾分布：头部类有大量样本，尾部类极少。基础模型（如 CLIP）微调已成为长尾学习主流范式，LIFT、LPT、VL-LTR 等方法通过 PEFT 保留预训练知识。

### 2. 被忽视的预训练偏差

这些方法只关注下游数据不平衡，忽视了基础模型本身的偏差：LAION 等预训练数据同样长尾分布。因此微调模型受上下游双重长尾分布影响。

### 3. 参数不平衡 vs 数据不平衡

作者将偏差分解为两类：
- **参数不平衡**：预训练数据的类别不均匀导致预训练权重偏向某些类（预训练数据不可访问，只能通过参数间接感知）
- **数据不平衡**：下游训练数据本身的类别不均

实验发现参数不平衡的影响更大，且现有 re-balancing（如 Logit Adjustment）只能缓解数据不平衡，无法解决参数不平衡。

### 4. 核心 Idea

构建因果图，将"不完整语义因子"识别为混杂变量——它导致模型学习样本与标签间的虚假相关而非真实因果关系。通过后门调整消除混杂效应。

## 方法详解

### 整体框架

1. 对基础模型做 zero-shot 推理，用 GLA 估计预训练标签先验 $\hat{\mathbb{P}}_P(Y)$
2. 分析参数不平衡和数据不平衡的交叉影响
3. 构建因果结构图：输入 $X$ → 标签 $Y$，不完整语义因子 $Z$ 同时影响 $X$ 和 $Y$（混杂变量）
4. 应用后门调整学习 $P(Y|do(X))$ 而非 $P(Y|X)$，消除虚假相关

### 关键设计

#### 1. 双重不平衡分析

- 用不同 CLIP 变体（CLIP、OpenCLIP、MetaCLIP）的 zero-shot 表现差异量化参数不平衡
- 按数据不平衡和参数不平衡交叉分组，发现双重尾部类受影响最严重
- 将 GLA 扩展到训练阶段（GLA-Train），发现它无法缓解参数不平衡

#### 2. 不完整语义因子与因果分析

- 当某类因参数不平衡是尾部类时，基础模型只捕捉了部分语义特征（如只学到"狗头"而非完整的"狗"）
- 这些不完整特征就是混杂变量 $Z$，诱导虚假关联
- 构建 SCM：$X \leftarrow Z \rightarrow Y$，$X \rightarrow Y$

#### 3. 后门调整

- 应用 do-calculus 的后门准则：$P(Y|do(X)) = \sum_z P(Y|X,Z=z)P(Z=z)$
- 在特征空间对不完整语义因子做边际化，学习真正的因果效应
- 实际实现中通过对 PEFT 适配器微调，结合后门调整损失替代标准 CE

## 实验关键数据

### 主实验结果

| 数据集 | 方法 | Many | Medium | Few | Overall |
|--------|------|------|--------|-----|---------|
| ImageNet-LT | LIFT (PEFT baseline) | 76.2 | 72.1 | 66.8 | 72.6 |
| ImageNet-LT | GLA (logit调整) | 76.8 | 73.0 | 68.5 | 73.5 |
| ImageNet-LT | **本文方法** | **77.5** | **73.9** | **69.7** | **74.2** |
| Places365-LT | LIFT | 45.2 | 43.8 | 44.5 | 44.3 |
| Places365-LT | **本文方法** | **46.9** | **45.3** | **46.1** | **45.8** |
| iNaturalist2018 | LIFT | 78.3 | 76.1 | 74.2 | 76.0 |
| iNaturalist2018 | **本文方法** | **80.5** | **78.2** | **76.3** | **78.0** |

三个数据集上分别提升 +1.6%、+1.5%、+2.0%。

### 消融分析

| 配置 | 作用目标 | ImageNet-LT | 说明 |
|------|---------|-------------|------|
| CE (无任何调整) | — | 71.8 | 基准 |
| LA (仅数据不平衡) | 数据不平衡 | 73.0 | 对尾部有帮助但有限 |
| GLA-Train (训练阶段) | 参数+数据 | 73.2 | 对参数不平衡改善微小 |
| GLA-ZS + GLA-FT (推理阶段) | 参数+数据 | 73.5 | 推理阶段logit调整稍好 |
| **本文后门调整** | **因果去混杂** | **74.2** | **根本解决混杂问题** |

### 关键发现

- LA 虽能改善尾部类分类器但几乎不改善特征质量（KNN 精度仅微增），说明参数不平衡根植于特征层而非分类头
- 双重尾部类（参数和数据都是尾部）是最受害的群体，需要专门对策
- 后门调整通过在特征空间消除混杂，从根本上改善了特征表示而不仅是分类边界

## 亮点与洞察

- **问题定义的深度**：首次系统区分基础模型微调中的参数不平衡与数据不平衡，填补了认知空白
- **因果视角的引入**：将不完整语义因子建模为混杂变量，从因果推断角度提供了原理性解释
- **反直觉发现**：logit 调整（GLA-Train）在训练阶段反而无法解决参数不平衡——表明偏差嵌在特征空间而非决策边界
- **跨模型验证**：在 CLIP、OpenCLIP、MetaCLIP 三种基础模型上验证了参数不平衡的一致存在

## 局限与展望

- 后门调整需要估计不完整语义因子的分布；当语义因子维度很高或难以估计时可能受限
- 未考虑多模态场景下文本 encoder 的参数不平衡
- 仅在 PEFT 设定下验证，全量微调场景下的效果待测
- 可探索与 GLA 的联合使用以同时从logit和特征两端去偏

## 相关工作与启发

- **vs LIFT/LPT**：这些方法忽视预训练数据偏差，本文补充了"参数不平衡"维度
- **vs GLA (Zhu et al. 2024)**：GLA 在推理阶段做 logit 调整有效，但本文证明训练阶段的 GLA 对参数不平衡无效
- **vs 因果长尾学习**：已有因果方法处理数据不平衡，本文首次将因果推断应用于基础模型的双重不平衡

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次拆解基础模型的双重不平衡并给出因果解
- 实验充分度: ⭐⭐⭐⭐ 三个标准基准+多个消融，跨模型验证充分
- 写作质量: ⭐⭐⭐⭐ 问题分析层层递进，因果建模清晰
- 价值: ⭐⭐⭐⭐⭐ 对基础模型微调范式有重要的方法论启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Avoiding Leakage Poisoning: Concept Interventions Under Distribution Shifts](avoiding_leakage_poisoning_concept_interventions_under_distribution_shifts.md)
- [\[CVPR 2026\] FedCART: Tackling Long-Tailed Distributions in Federated Adversarial Training via Classifier Refinement](../../CVPR2026/ai_safety/fedcart_tackling_long-tailed_distributions_in_federated_adversarial_training_via.md)
- [\[ICML 2025\] De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks](de-antifake_rethinking_the_protective_perturbations_against_voice_cloning_attack.md)
- [\[ICML 2025\] Private Model Personalization Revisited](private_model_personalization_revisited.md)
- [\[ICML 2025\] Accelerating Spectral Clustering under Fairness Constraints](accelerating_spectral_clustering_under_fairness_constraints.md)

</div>

<!-- RELATED:END -->
