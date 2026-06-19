---
title: >-
  [论文解读] On the Generalization of Handwritten Text Recognition Models
description: >-
  [CVPR 2025][手写文本识别] 本文首次系统性地分析了 HTR 模型在域外（OOD）数据上的泛化能力，通过对 8 个 SOTA 模型在 7 个数据集（5 种语言）上的 336 种 OOD 评估发现：文本差异是影响泛化的最关键因素，OOD 误差在 70% 的情况下可以被可靠预估（偏差 <10 个百分点）。
tags:
  - "CVPR 2025"
  - "手写文本识别"
  - "域泛化"
  - "分布外泛化"
  - "跨语言"
  - "因子分析"
---

# On the Generalization of Handwritten Text Recognition Models

**会议**: CVPR 2025  
**arXiv**: [2411.17332](https://arxiv.org/abs/2411.17332)  
**代码**: [https://github.com/carlos10garrido/HTR-OOD](https://github.com/carlos10garrido/HTR-OOD)  
**领域**: OCR/文本识别  
**关键词**: 手写文本识别, 域泛化, 分布外泛化, 跨语言, 因子分析

## 一句话总结

本文首次系统性地分析了 HTR 模型在域外（OOD）数据上的泛化能力，通过对 8 个 SOTA 模型在 7 个数据集（5 种语言）上的 336 种 OOD 评估发现：文本差异是影响泛化的最关键因素，OOD 误差在 70% 的情况下可以被可靠预估（偏差 <10 个百分点）。

## 研究背景与动机

**领域现状**：手写文本识别（HTR）近年来在标准基准上取得了显著进步，主流方法包括 CTC 解码（CRNN、VAN）、序列到序列（Transformer）和混合方法（CTC+CE）。然而，这些进展建立在训练和测试数据同分布（i.i.d.）的假设上。

**现有痛点**：(1) 现有"泛化"评估仅限于同分布的训练-测试划分（同一手稿不同行），并非真正的跨域泛化；(2) 当模型面对全新手稿、不同语言、不同历史时期的文本时，性能急剧下降——初步实验显示 OOD 场景下 CER 平均从 ~7% 飙升到 ~35%；(3) Transfer Learning 和 Domain Adaptation 需要目标域数据，在完全未知的目标域上不适用。

**核心矛盾**：HTR 领域对"泛化"的定义过于狭隘，未探索真正的 OOD 泛化场景（零接触新手稿/新语言），且缺乏对影响 OOD 性能的关键因素的系统理解。

**本文目标**：(1) 在域泛化（DG）框架下分析 HTR 模型的 OOD 表现；(2) 识别影响泛化的核心因素；(3) 评估 OOD 误差是否可预估。

**切入角度**：构建 visual divergence 和 textual divergence 两个代理指标量化源域和目标域之间的差异，通过因子分析揭示影响 OOD 性能的显著因素。

**核心 idea**：通过大规模系统实验（8 模型 × 7 数据集 × 6 个 OOD 目标 = 336 种 OOD 评估），发现文本差异（语言/字母表差异）比视觉差异（书写风格）对泛化影响更大，且 OOD 误差可通过这些代理指标可靠预测。

## 方法详解

### 整体框架

实验分两部分：(1) **实践性分析**——在标准化条件下评估 8 个 HTR 模型的 ID 和 OOD 性能，探讨模型容量、选择策略和合成数据对泛化的影响；(2) **因子分析**——定义视觉差异和文本差异指标，进行多因素方差分析（ANOVA）确定影响 OOD 性能的显著因子，并基于这些因子构建 OOD 误差预测模型。

### 关键设计

1. **标准化跨域评估框架**:

    - 功能：确保公平可比地评估不同 HTR 模型的 OOD 泛化能力
    - 核心思路：8 个模型（CRNN、VAN、C-SAN、HTR-VT、Kang Transformer、Michael、LT、VLT）覆盖 CTC/Seq2Seq/Hybrid 三大类，在 7 个数据集（IAM/Rimes/Bentham/Saint-Gall/G.W./Rodrigo/ICFHR2016，跨英/法/拉丁/西/德 5 种语言）上从头训练，统一使用 Unicode 合并的 94 字符字母表。每个模型在一个源域训练后在其他所有域上 OOD 测试，共 336 种评估。
    - 设计动机：以往研究使用不同的训练设置、不同的字母表处理方式，结果不可比。标准化后才能公平比较模型间的泛化差异，揭示架构本身对泛化的影响。

2. **视觉差异与文本差异指标（Visual/Textual Divergence）**:

    - 功能：量化源域和目标域之间的视觉和语言层面差异
    - 核心思路：**视觉差异**使用 FID（Fréchet Inception Distance）度量源域和目标域图像特征分布的距离。**文本差异**度量两域文本内容的语言差异——基于字母表重叠度和字符频率分布的 KL 散度。这两个指标直觉上应该与 OOD 性能下降程度正相关。
    - 设计动机：OOD 性能下降可能来自视觉原因（字迹风格差异大）或文本原因（不同语言/字母表），区分两者的影响有助于针对性地改进模型。

3. **因子分析与 OOD 误差预测**:

    - 功能：识别影响 OOD 泛化的显著因子并预测 OOD 误差
    - 核心思路：以 OOD CER 为因变量，进行多因素方差分析（ANOVA），考察模型类型、源域、目标域、视觉差异、文本差异等因子的显著性。基于显著因子构建回归模型预测 OOD 误差。结果表明文本差异是最显著因子，其次是视觉差异。在 70% 的情况下，预测误差与实际误差的偏差不超过 10 个 CER 百分点。
    - 设计动机：如果 OOD 误差可预测，部署 HTR 系统时就能预先评估模型在新数据上的可靠性，而不需要实际标注测试数据。

### 损失函数 / 训练策略

取决于模型架构——CTC 模型使用 CTC loss，Seq2Seq 使用 CE loss，Hybrid 使用 $\mathcal{L} = \lambda \mathcal{L}_{\text{ctc}} + (1-\lambda) \mathcal{L}_{\text{ce}}$（$\lambda=0.5$）。所有模型从头训练 500 epochs，基于验证集 CER 选择最佳模型，100 epoch 不改善则提前停止。

## 实验关键数据

### 主实验（ID vs OOD CER%，选取代表性模型）

| 数据集 | CRNN ID | CRNN OOD | VAN ID | VAN OOD | HTR-VT ID | HTR-VT OOD |
|--------|---------|----------|--------|---------|-----------|------------|
| IAM (En) | 6.4 | 34.9 (+28.5) | 6.6 | 28.6 (+22.0) | 5.8 | 33.7 (+27.9) |
| Rimes (Fr) | 3.7 | 25.0 (+21.2) | 5.6 | 21.3 (+15.6) | 7.9 | 28.3 (+20.4) |
| Bentham (En) | 4.7 | 25.3 (+20.6) | 7.4 | 26.6 (+19.2) | 8.4 | 33.3 (+24.9) |
| S.G. (Lat) | 7.2 | 33.6 (+26.3) | 7.8 | 39.8 (+32.0) | 17.1 | 36.5 (+19.3) |
| Rodrigo (Sp) | 4.1 | 36.5 (+32.4) | 4.2 | 29.9 (+25.7) | 5.1 | 34.2 (+29.1) |

### 消融实验（因子分析 - ANOVA 结果）

| 因子 | F-statistic | p-value | 显著性 |
|------|------------|---------|--------|
| 文本差异 | 最高 | <0.01 | 极显著 |
| 视觉差异 | 中等 | <0.01 | 显著 |
| 源域 | 中等 | <0.01 | 显著 |
| 模型架构 | 较低 | <0.05 | 弱显著 |

### 关键发现

- **ID-OOD gap 巨大**：8 个模型平均 OOD CER 约 35%，比 ID 的 ~7% 高出约 28 个百分点，说明当前 HTR 模型完全缺乏 OOD 泛化能力。
- **文本差异是第一影响因子**：当源域和目标域的语言/字母表差异大时，OOD 性能下降最严重。视觉差异（书写风格）是第二因子。
- **模型架构的影响相对较小**：不存在一个在所有 OOD 场景下稳定最优的架构。CTC 模型（CRNN、VAN）在 OOD 上相对稳健，大参数量的 Transformer 模型反而容易过拟合。
- **合成数据有帮助但有限**：合成数据作为源域训练可提升部分 OOD 场景的性能，但无法完全弥补域差距。
- OOD 误差预测在 70% 的情况下偏差 <10 CER 点，为实际部署提供了可行的预评估方案。

## 亮点与洞察

- **首次系统揭示 HTR 的 OOD 泛化缺陷**：336 种评估的大规模实验提供了可靠的统计结论，打破了"基准越来越好=问题已解决"的假象。
- **文本差异 > 视觉差异**的发现意外但合理：HTR 模型本质上学习了语言模型（字符序列的统计规律），当目标域语言完全不同时，学到的语言先验完全失效。这提示改进方向应侧重于语言无关的视觉特征提取。
- OOD 误差可预估的结论有实际价值：部署 HTR 时可根据源域和目标域的文本/视觉差异评估系统可靠性。

## 局限与展望

- 仅使用行级 HTR，未涉及端到端文档识别。
- 统一字母表为 94 字符，可能对某些语言特有字符处理不够精细。
- 未探索多源域训练和域泛化算法（如 DRO、IRM 等）对 HTR 的效果。
- 未来可以研究语言无关的 HTR 架构设计，以及大规模预训练和自监督学习对 OOD 泛化的影响。

## 相关工作与启发

- **vs TrOCR 等大规模预训练方法**: TrOCR 通过大规模预训练提升了泛化性，但需要海量数据和计算。本文的分析可帮助这类方法识别最有价值的训练数据（应覆盖更多语言而非更多视觉风格）。
- **vs Domain Adaptation 方法**: DA 需要目标域数据，适用场景受限。本文的 DG 框架更贴近实际——部署时通常不知道目标域是什么。
- **vs 以往 HTR 评估**: 以往评估只关注 ID 性能，忽视了 OOD 场景。本文揭示的巨大 ID-OOD gap 呼吁社区重新审视评估范式。

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在 HTR 领域进行系统的 OOD 泛化分析，视角新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 336 种评估、8 模型、7 数据集、5 语言，统计分析严谨
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，结论有说服力
- 价值: ⭐⭐⭐⭐ 揭示了 HTR 的关键盲点，为未来研究指明方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Do ImageNet-trained Models Learn Shortcuts? The Impact of Frequency Shortcuts on Generalization](do_imagenet-trained_models_learn_shortcuts_the_impact_of_frequency_shortcuts_on_.md)
- [\[CVPR 2025\] Gradient-Guided Annealing for Domain Generalization](gradient-guided_annealing_for_domain_generalization.md)
- [\[ICML 2025\] Provably Improving Generalization of Few-Shot Models with Synthetic Data](../../ICML2025/others/provably_improving_generalization_of_few-shot_models_with_synthetic_data.md)
- [\[ACL 2025\] Principled Understanding of Generalization for Generative Transformer Models in Arithmetic Reasoning Tasks](../../ACL2025/others/principled_generalization_arithmetic.md)
- [\[CVPR 2025\] Feature Selection for Latent Factor Models](feature_selection_for_latent_factor_models.md)

</div>

<!-- RELATED:END -->
