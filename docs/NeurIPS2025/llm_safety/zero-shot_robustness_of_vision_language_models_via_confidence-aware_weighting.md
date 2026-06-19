---
title: >-
  [论文解读] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting
description: >-
  [NeurIPS 2025][LLM安全][CLIP] 提出 CAW（Confidence-Aware Weighting），一种针对CLIP模型的对抗微调损失函数，通过置信度感知加权重点关注困难对抗样本，结合特征对齐正则化保留预训练语义知识，在AutoAttack下实现零样本鲁棒性SOTA，且内存占用更低。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "CLIP"
  - "对抗鲁棒性"
  - "零样本"
  - "对抗微调"
  - "置信度加权"
---

# Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting

**会议**: NeurIPS 2025  
**arXiv**: [2510.02913](https://arxiv.org/abs/2510.02913)  
**代码**: 暂无  
**领域**: 多模态VLM  
**关键词**: CLIP, 对抗鲁棒性, 零样本, 对抗微调, 置信度加权

## 一句话总结

提出 CAW（Confidence-Aware Weighting），一种针对CLIP模型的对抗微调损失函数，通过置信度感知加权重点关注困难对抗样本，结合特征对齐正则化保留预训练语义知识，在AutoAttack下实现零样本鲁棒性SOTA，且内存占用更低。

## 研究背景与动机

CLIP等视觉语言模型展现了强大的零样本泛化能力，但面对对抗攻击极其脆弱——小的、人眼不可见的扰动就能导致预测严重偏移。对抗训练是提升鲁棒性最有效的方法，但直接应用于CLIP这样的大规模预训练模型面临两个关键挑战：(1) 容易过拟合并遗忘预训练知识；(2) 难以同时维持干净数据精度和对抗鲁棒性。

已有工作的不足：
- **TeCoA** 首次研究大规模VLM的零样本鲁棒性，引入对比对抗损失+文本监督，但无法同时提升干净精度和鲁棒精度
- **PMG-AFT** 在TeCoA基础上增加新损失项增强鲁棒性，但内存开销大
- **TGA-ZSR** 利用语义文本监督提升鲁棒性和可解释性，但在强攻击下鲁棒精度仍不理想

核心出发点：**不是所有对抗样本都同等重要**——模型对某些样本已经很有信心，对另一些则非常不确定。如果能让训练重点关注那些"困难的"对抗样本（即模型最不自信的样本），就能更高效地提升鲁棒性。

## 方法详解

### 整体框架

CAW使用冻结的原始CLIP模型和可微调的目标CLIP模型（仅微调图像编码器），通过两个新损失项联合优化：(1) 置信度感知损失关注困难样本；(2) 特征对齐正则化保留预训练知识。训练采用标准的对抗训练框架——内循环用PGD生成对抗样本（最大化交叉熵损失），外循环用新损失函数更新模型参数。

### 关键设计

1. **Confidence-Aware Loss（置信度感知损失）**：核心思想是用KL散度对齐冻结CLIP在干净图像上的预测分布 $P^{\text{clean}}$ 和微调CLIP在对抗图像上的预测分布 $P^{\text{adv}}$，并以 $(1 - P^{\text{adv}}_{i,y_i})$ 作为权重因子放大困难样本的贡献：

    $L_{\text{CA}} = \frac{1}{N}\sum_{i=1}^{N}\left[\text{KL}(P^{\text{adv}}_i \| P^{\text{clean}}_i)(1 - P^{\text{adv}}_{i,y_i})\right]$

   其中 $P^{\text{adv}}_{i,y_i}$ 是对抗输入下真实类别的预测概率——概率越低说明模型越不确定，权重越大，训练越关注该样本。与ARoW方法不同，CAW使用 $\text{KL}(P^{\text{adv}} \| P^{\text{clean}})$ 的方向（对抗分布在前），实验表明这产生更好的效果。

2. **Feature Alignment Regularization（特征对齐正则化）**：通过最小化微调编码器和冻结编码器在对抗输入上的特征$\ell_2$距离，在文本对齐之前的图像特征层面保持语义一致性：

    $L_{\text{Reg}} = \frac{1}{N}\sum_{i=0}^{N}\|f(x_{\text{adv}})_{\text{tar}} - f(x_{\text{adv}})_{\text{ori}}\|_2$

   跟蒸馏而不是在logit层做对齐，这在更丰富的特征空间中保留了预训练CLIP的视觉语义知识，减少微调时的过拟合风险。

3. **总损失函数**：

    $L_{\text{total}} = L_{\text{CE}} + \alpha \cdot L_{\text{CA}} + \beta \cdot L_{\text{Reg}}$

   $L_{\text{CE}}$ 是标准交叉熵损失，$\alpha$ 和 $\beta$ 控制各项权重。

### 损失函数 / 训练策略

- 内循环：使用PGD-2生成对抗样本，扰动范围 $\epsilon = 1/255$
- 外循环：用 $L_{\text{total}}$ 更新图像编码器参数
- 文本模板使用CLIP标准模板构建类别描述
- 仅在TinyImageNet上训练，然后零样本迁移到14个其他数据集

## 实验关键数据

### 主实验

**AutoAttack零样本鲁棒精度（$\epsilon=1/255$，15个数据集）**

| 方法 | TinyImageNet | CIFAR-10 | CIFAR-100 | STL-10 | SUN397 | Caltech-101 | 平均 |
|------|-------------|----------|-----------|--------|--------|-------------|------|
| CLIP | 0.02 | 0.01 | 0.08 | 0.03 | 0.04 | 0.43 | 0.09 |
| FT-Adv | 50.48 | 37.55 | 20.39 | 69.14 | 16.25 | 49.90 | 29.08 |
| TeCoA | 35.03 | 28.18 | 16.09 | 66.08 | 17.41 | 54.54 | 27.23 |
| PMG-AFT | 44.26 | 44.12 | 23.66 | 73.90 | 19.63 | 60.57 | 31.55 |
| TGA-ZSR | 49.45 | 40.53 | 22.38 | 72.06 | 20.36 | 57.16 | 31.63 |
| **CAW** | **50.52** | **47.35** | **26.35** | **74.27** | **19.64** | **62.79** | **33.51** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅$L_{\text{CE}}$ | 基线鲁棒精度 | 标准对抗训练 |
| $L_{\text{CE}} + L_{\text{CA}}$ | 鲁棒精度↑ | 加入置信度加权后困难样本学习更充分 |
| $L_{\text{CE}} + L_{\text{Reg}}$ | 干净精度↑ | 特征正则化有效保留预训练知识 |
| $L_{\text{CE}} + L_{\text{CA}} + L_{\text{Reg}}$ | 两者均↑ | 两个损失项互补，最优性能 |
| KL(clean‖adv) 方向 | 鲁棒精度↓ | 对抗分布在前的KL方向更优 |

**与基线的内存对比**

| 方法 | 内存使用（相对） | 鲁棒精度（平均） |
|------|----------------|-----------------|
| PMG-AFT | 高 | 31.55 |
| TGA-ZSR | 高 | 31.63 |
| **CAW** | **低** | **33.51** |

### 关键发现

- 原始CLIP在AutoAttack下精度接近0（平均0.09%），说明零样本鲁棒性问题极其严重
- CAW在所有15个数据集上的平均鲁棒精度达到33.51%，比之前最优的TGA-ZSR高约2%
- 仅在TinyImageNet上微调即可泛化到14个不同分布的数据集，说明学到的鲁棒特征具有良好迁移性
- CIFAR-10上CAW比PMG-AFT提升了3.23个百分点（47.35 vs 44.12）

## 亮点与洞察

- **简洁有效的核心思想**：通过 $(1 - P^{\text{adv}}_{i,y_i})$ 加权实现对困难样本的自然关注，无需额外的样本挖掘或课程学习策略
- 在特征空间而非logit空间做正则化，更好地保留了CLIP预训练的丰富语义
- 整体方法轻量——无需额外的文本生成、注意力机制或复杂的多阶段训练
- 内存效率优于PMG-AFT和TGA-ZSR，对资源受限场景更友好

## 局限与展望

- 仅在分类任务上验证，未扩展到检测、分割等更复杂的下游任务
- 文中主要针对 $\ell_\infty$ 范数扰动，对 $\ell_2$ 等其他攻击类型的鲁棒性未充分讨论
- 作为Workshop论文，实验规模和分析深度有所限制
- 未与最新的对抗训练方法（如利用生成模型增强对抗训练数据的方法）进行对比
- 超参 $\alpha$、$\beta$ 的选择策略未详细说明

## 相关工作与启发

- TeCoA 首次建立了大规模VLM零样本鲁棒性的研究范式
- PMG-AFT 引入预训练模型引导的微调框架
- ARoW 提出优先关注脆弱样本的思想，CAW将其适配并改进用于视觉语言模型
- 启发：置信度加权的思想简单且通用，可推广到其他对抗训练和鲁棒学习场景

## 评分

- 新颖性: ⭐⭐⭐ 核心思路（置信度加权+特征正则化）相对直觉，但在VLM场景下组合有效
- 实验充分度: ⭐⭐⭐⭐ 15个数据集评估、多种攻击方式、内存对比
- 写作质量: ⭐⭐⭐⭐ 简洁明了，方法公式清晰
- 价值: ⭐⭐⭐⭐ 提供了一种低资源消耗的CLIP鲁棒性增强方案，实际部署意义大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Visual Language Models as Zero-Shot Deepfake Detectors](../../ICML2025/llm_safety/visual_language_models_as_zero-shot_deepfake_detectors.md)
- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](../../CVPR2025/llm_safety/hyperbolic_safety-aware_vision-language_models.md)
- [\[NeurIPS 2025\] Approximate Domain Unlearning for Vision-Language Models](approximate_domain_unlearning_for_visionlanguage_models.md)
- [\[NeurIPS 2025\] Enhancing CLIP Robustness via Cross-Modality Alignment](enhancing_clip_robustness_via_crossmodality_alignment.md)

</div>

<!-- RELATED:END -->
