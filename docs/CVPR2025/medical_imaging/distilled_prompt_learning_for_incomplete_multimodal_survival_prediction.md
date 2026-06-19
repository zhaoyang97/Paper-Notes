---
title: >-
  [论文解读] Distilled Prompt Learning for Incomplete Multimodal Survival Prediction
description: >-
  [CVPR 2025][医学图像][生存预测] 本文提出DisPro (Distilled Prompt Learning)，通过两阶段提示学习——UniPro蒸馏各模态知识分布 + MultiPro利用LLM从可用模态推断缺失模态——同时补偿缺失模态的特异性和共享信息，在5个TCGA生存预测数据集上取得SOTA。
tags:
  - "CVPR 2025"
  - "医学图像"
  - "生存预测"
  - "缺失模态"
  - "提示学习"
  - "LLM鲁棒性"
  - "病理WSI"
  - "基因组学"
  - "知识蒸馏"
---

# Distilled Prompt Learning for Incomplete Multimodal Survival Prediction

**会议**: CVPR 2025  
**arXiv**: [2503.01653](https://arxiv.org/abs/2503.01653)  
**代码**: [Innse/DisPro](https://github.com/Innse/DisPro)  
**领域**: 医学影像  
**关键词**: 生存预测, 缺失模态, 提示学习, LLM鲁棒性, 病理WSI, 基因组学, 知识蒸馏

## 一句话总结

本文提出DisPro (Distilled Prompt Learning)，通过两阶段提示学习——UniPro蒸馏各模态知识分布 + MultiPro利用LLM从可用模态推断缺失模态——同时补偿缺失模态的特异性和共享信息，在5个TCGA生存预测数据集上取得SOTA。

## 研究背景与动机

多模态生存预测整合病理图像和基因组数据进行精确预后分析，是计算病理学的重要任务。但在临床实践中模态缺失是常态：

1. **数据获取限制**：基因测序成本高（尤其欠发达地区），病理切片可能丢失或质量不足
2. **现有多模态模型脆弱**：MCAT、MOTCat、CMTA等假设所有模态可用，缺失任一模态时性能急剧下降

现有应对缺失模态的方法存在根本性局限：
- **生成式补全** (Diffusion/VAE)：只能推断模态共享信息，无法"凭空生成"缺失模态的特异性信息
- **检索式补全** (M3Care)：单个检索样本随机性大，难以全面捕获模态独有知识
- **无补全方法** (MUSE, MAP)：学习模态不变表示，同样忽略模态特异性知识

核心洞察：需要同时补偿缺失模态的 **共享信息** (modality-common) 和 **特异性信息** (modality-specific)。

## 方法详解

### 整体框架

DisPro分为两个阶段：
- **Stage 1 - UniPro (单模态提示)**：为每个模态独立训练可学习提示，蒸馏该模态在不同风险等级下的知识分布
- **Stage 2 - MultiPro (多模态提示)**：以可用模态为LLM的提示推断缺失模态表示，并注入Stage 1学到的模态特异性知识

### 关键设计

**1. 单模态提示蒸馏 (UniPro)**

受CoOp启发，但扩展至MIL（多示例学习）范式以适配超大WSI（100,000×100,000像素）：
- 为每个风险等级（$2I_t$类）设计可学习的上下文token $[P]_1...[P]_k$
- 将病理patch/基因pathway特征通过adapter映射到LLM文本空间（768维）
- 基于CLIP式对比学习：计算每个patch与各类文本表示的相似度
- TopK max-pooling聚合得到slide级预测，NLL损失优化

关键产出：每个模态的文本类表示 $\mathbf{t}_p^{(j)}$ / $\mathbf{t}_g^{(j)}$ 编码了该模态不同风险等级的知识分布。

**2. 多模态提示推断 (MultiPro)**

以病理可用、基因缺失为例：
- 将病理patch特征作为LLM（BERT）输入token
- 基因组位置用可学习的placeholder token代替
- LLM通过self-attention从病理信息推断基因组表示 → 补偿模态共享信息

**UniPro Scoring**：智能选择输入token（解决LLM 512 token长度限制+信息冗余）：
$$\mathbf{s}_{n,\#}^{(i)} = \mathbf{s}_{n,p}^{(i,\tau)} + \mathbf{s}_{n,g}^{(i,\tau)} + \mathbf{a}_{n,p}^{(i)}$$
- 第一项：与本模态UniPro的相关性分数（选discriminative token）
- 第二项：与缺失模态UniPro的相关性分数（选cross-modal相关token）
- 第三项：可学习的self-scoring（动态适应当前输入）

**3. UniPro Distillation**

将LLM输出的缺失模态部分 $[\tilde{\mathbf{g}}_n]$ 与Stage 1的基因组文本类表示对齐：
- 计算推断表征与各类文本表示的相似度
- 通过生存损失强制推断的风险概率分布匹配UniPro学到的分布
- 从而注入缺失模态的特异性知识

### 损失函数

$$\mathcal{L} = \mathcal{L}_{surv}^{cls} + \alpha_1 \mathcal{L}_{ud}^p + \alpha_2 \mathcal{L}_{ud}^g$$

- $\mathcal{L}_{surv}^{cls}$：基于[CLS] token的生存预测NLL损失
- $\mathcal{L}_{ud}^p$, $\mathcal{L}_{ud}^g$：UniPro蒸馏损失，分别监督病理和基因组缺失时的补偿学习

## 实验关键数据

### 5个TCGA数据集 (60%模态缺失率)

| 方法 | 缺失场景 | BLCA | BRCA | COADREAD | LUAD | UCEC | Avg |
|------|----------|------|------|----------|------|------|-----|
| MOTCat | 完整 | 0.627 | 0.672 | 0.650 | 0.675 | 0.721 | 0.669 |
| SurvPath | 完整 | 0.657 | 0.707 | 0.708 | 0.680 | 0.739 | 0.698 |
| COM | P有G缺 | 0.602 | 0.674 | 0.678 | 0.634 | 0.699 | 0.657 |
| M3Care | P有G缺 | 0.621 | 0.669 | 0.657 | 0.622 | 0.703 | 0.654 |
| MAP | P有G缺 | 0.592 | 0.628 | 0.597 | 0.649 | 0.693 | 0.632 |
| **DisPro** | **P有G缺** | **0.632** | **0.690** | **0.688** | **0.661** | **0.727** | **0.680** |
| **DisPro** | **双模态有** | **0.664** | **0.722** | **0.703** | **0.674** | **0.748** | **0.702** |

### 关键对比

- **60%缺失下**：DisPro平均C-index 0.680 vs MAP 0.632 vs M3Care 0.654，显著领先
- **完整模态下**：DisPro (0.702) 甚至超越SurvPath (0.698)，说明框架本身的优越性
- **不同缺失率 (0%~60%)**：DisPro性能下降最小，鲁棒性最优

### 消融实验

| 配置 | 效果 |
|------|------|
| 无UniPro蒸馏 | 下降显著，丢失模态特异性信息 |
| 无UniPro Scoring | 随机选token导致信息丢失 |
| 无UniPro Distillation | 无法注入模态特异性知识 |
| 完整DisPro | 最优 |

## 亮点与洞察

1. **信息论视角的清晰分析**：将缺失模态信息明确拆分为modality-common和modality-specific两部分，指出现有方法只能补偿前者
2. **CoOp到MIL的扩展**：将提示学习优雅地扩展到超大WSI，通过TopK池化桥接patch-level和slide-level预测
3. **LLM鲁棒性的利用**：无需微调LLM参数，仅通过prompt engineering和adapter利用其对缺失输入的推理能力
4. **UniPro的三重复用**：Stage 1的产出在Stage 2中作为蒸馏目标、token评分器和推理指导

## 局限性

1. 两阶段训练增加实现复杂度，Stage 1需为每个模态单独训练
2. 基于BERT的LLM上下文长度限制（512 token）要求对WSI进行大幅下采样，可能丢失细粒度信息
3. 仅考虑两种模态（病理+基因），未验证三模态及以上的扩展性
4. UniPro的知识蒸馏质量依赖于可学习prompt的表达能力，在小数据集上可能不足

## 相关工作

- **多模态生存预测**：MCAT → MOTCat → CMTA → PIBD → MMP → SurvPath
- **缺失模态学习**：SMIL (Bayesian meta-learning) → M3Care (检索补全) → MUSE (图对比) → MAP (LLM提示)
- **提示学习**：CoOp → CoCoOp → 本文扩展至MIL范式
- **计算病理学**：TransMIL → CLAM → CONCH → UNI

## 评分

- **新颖性**：5/5 — 两阶段提示学习同时补偿模态共享和特异性信息，理论motivation清晰
- **有效性**：4/5 — 5个数据集一致性优势，60%缺失下超越完整模态方法
- **清晰度**：4/5 — 框架复杂但图示清晰，符号系统一致
- **意义**：5/5 — 解决了多模态医学AI从实验室到临床的关键障碍（模态缺失）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] H2-Surv: Hierarchical Hyperbolic Multimodal Representation Learning for Survival Prediction](../../CVPR2026/medical_imaging/h2-surv_hierarchical_hyperbolic_multimodal_representation_learning_for_survival_.md)
- [\[CVPR 2026\] MUST: Modality-Specific Representation-Aware Transformer for Diffusion-Enhanced Survival Prediction with Missing Modality](../../CVPR2026/medical_imaging/must_modality-specific_representation-aware_transformer_for_diffusion-enhanced_s.md)
- [\[ICLR 2026\] Inference-Time Dynamic Modality Selection for Incomplete Multimodal Classification](../../ICLR2026/medical_imaging/inference-time_dynamic_modality_selection_for_incomplete_multimodal_classificati.md)
- [\[CVPR 2026\] MDCS-MoAME: Multi-directional Composite Scanning with Mixture of Attention and Mamba Experts for Cancer Survival Prediction](../../CVPR2026/medical_imaging/mdcs-moame_multi-directional_composite_scanning_with_mixture_of_attention_and_ma.md)
- [\[CVPR 2025\] Residual SODAP: Residual Self-Organizing Domain-Adaptive Prompting with Structural Knowledge Preservation for Continual Learning](residual_sodap_residual_self-organizing_domain-adaptive_prompting_with_structura.md)

</div>

<!-- RELATED:END -->
