---
title: >-
  [论文解读] Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning
description: >-
  [NeurIPS 2025][多模态VLM][CLIP组合推理] 提出 CLIC，通过拼接两张图像并基于跨图词汇交换生成 hard negatives，同时创建多个正样本描述，仅微调 CLIP 文本编码器就能同时提升组合推理能力（SugarCrepe++ SOTA）和下游检索性能，打破了之前方法中组合性与检索性不可兼得的困局。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "CLIP组合推理"
  - "SugarCrepe++"
  - "图像拼接"
  - "困难负样本"
  - "对比学习"
---

# Advancing Compositional Awareness in CLIP with Efficient Fine-Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2505.24424](https://arxiv.org/abs/2505.24424)  
**代码**: [https://clic-compositional-clip.github.io/](https://clic-compositional-clip.github.io/)  
**领域**: 多模态VLM / CLIP微调  
**关键词**: CLIP组合推理, SugarCrepe++, 图像拼接, 困难负样本, 对比学习

## 一句话总结

提出 CLIC，通过拼接两张图像并基于跨图词汇交换生成 hard negatives，同时创建多个正样本描述，仅微调 CLIP 文本编码器就能同时提升组合推理能力（SugarCrepe++ SOTA）和下游检索性能，打破了之前方法中组合性与检索性不可兼得的困局。

## 研究背景与动机

**领域现状**：CLIP 等视觉语言模型在零样本分类和检索任务上表现出色，但在组合推理（compositional reasoning）上存在显著缺陷——它们倾向于学习"词袋"表示，无法区分"红衬衫的人骑灰马"和"灰衬衫的人骑红马"。已有多种方法（NegCLIP、DAC、TripletCLIP、SVLC）试图通过引入 hard negatives 来改善组合性。

**现有痛点**：最近的 SugarCrepe++ 基准揭示了一个令人尴尬的事实——之前声称提升了组合性的方法（如 DAC 在 SugarCrepe 上达到 89.4%），在 SugarCrepe++ 上的表现甚至不如预训练 CLIP（DAC 降至 53.7%）。这意味着这些方法只是学会了检测词汇层面的变化（lexical sensitivity），而没有真正理解语义差异。此外，提升组合性的方法通常会降低检索性能，尽管直觉上组合理解力增强应该有助于检索。

**核心矛盾**：组合性提升方法过度针对特定基准的 hard negative 模式（如固定交换颜色/动作），导致对新基准过拟合而对真正的语义理解提升有限。同时，强化 hard negative 训练往往破坏了 CLIP 原有的通用表示能力。

**本文目标** 如何在提升 CLIP 组合推理能力的同时保持甚至增强其检索性能？

**切入角度**：不是单独为每张图像生成 hard negative，而是通过拼接两张图像创造一个组合场景，然后通过跨图词汇交换自然地生成 hard negatives，同时利用多句描述创建多个正样本增强语义不变性。

**核心 idea**：图像拼接 + 跨图词汇交换 = 低成本、高多样性的组合性训练数据。

## 方法详解

### 整体框架

CLIC 在每个训练迭代中采样一批图像-文本对 $\{x_i, y_i\}$，为每张图像随机选择另一张方向一致的图像，将两张图像拼接成新图像 $u_i$，并基于两张图像的描述构造 4 个正样本和 1 个负样本。整个训练仅微调文本编码器，视觉编码器冻结，且每隔一步使用标准 CLIP 训练防止偏离预训练表示。

### 关键设计

1. **图像拼接与多正样本生成**:

    - 功能：创造组合场景并提供丰富的正样本描述
    - 核心思路：拼接图像 $u_i = \text{RandomConcat}(x_i, x_{i+m})$，然后构造 4 个正样本：$p_1$ = 两张图首句按序拼接；$p_2$ = 首句顺序交换（教模型对语序不变性）；$p_3, p_4$ = 从各图描述中随机选取其他句子拼接（增加描述多样性）。图像拼接使得组合数量随数据集大小平方增长，且正样本天然覆盖了同一图像的多角度描述
    - 设计动机：单图 hard negative 的组合空间有限且容易过拟合特定基准；图像拼接本身就是一种组合操作，能自然地产生需要组合理解的场景

2. **跨图词汇交换生成 Hard Negatives**:

    - 功能：低成本生成语义有意义的困难负样本
    - 核心思路：用 spaCy 解析 $p_1$ 中两个句子的词汇类别，随机选择两句共有的某个词类（名词/动词/形容词等），从每句各选一个该类词汇进行交换，生成负样本 $n$。交换后的描述不再正确描述拼接图像（除非恰好交换的词同义）。关键是不限定特定词类，避免过拟合某类基准测试
    - 设计动机：相比 DAC 需 LLM 生成、TripletCLIP 需额外生成合成图像，这种方法几乎零额外计算成本且不针对特定基准

3. **三重损失函数设计**:

    - 功能：分别优化对比学习、hard negative 区分和语义不变性
    - 核心思路：总损失 $\mathcal{L} = \lambda_{Cont}\mathcal{L}_{Cont} + \lambda_{S\text{-}Neg}\mathcal{L}_{S\text{-}Neg} + \lambda_{Uni}\mathcal{L}_{Uni}$。对比损失 $\mathcal{L}_{Cont}$ 扩展到 4 个正样本；$\mathcal{L}_{S\text{-}Neg}$ 是每个正样本与负样本的二元对比（确保 hard negative 始终影响训练）；$\mathcal{L}_{Uni}$ 最小化 $p_1$ 和 $p_2$（首句顺序交换）的文本嵌入距离，教模型对语序无关但语义相同的文本保持一致
    - 设计动机：标准对比损失中 hard negative 可能被 batch 内其他容易区分的样本"掩盖"，单独的 S-Neg 损失保证 hard negative 始终参与优化

### 训练策略

仅微调文本编码器，224x224 分辨率，每隔一步切换到标准单图 CLIP 训练以防偏离。使用 ~1M 样本子集（来自 PixelProse-RedCaps/CC12M 或 CogVLM 重标注的 Laion），训练数据独立于 MS-COCO（保证零样本评估的公平性）。

## 实验关键数据

### 主实验（ViT-B/32, SugarCrepe++ ITT）

| 方法 | SC++ Replace ITT | SC++ Swap ITT | WG Image | COCO I→T | COCO T→I |
|------|-----------------|---------------|----------|----------|----------|
| CLIP | 69.5 | 45.7 | 11.0 | 74.1 | 54.6 |
| NegCLIP | 70.5 | 56.4 | 11.0 | 83.6* | 72.2* |
| DAC-LLM | 53.7 | 32.2 | 10.5 | 63.3 | 58.1 |
| TripletCLIP | 73.5 | 43.4 | 11.2 | 72.3 | 56.8 |
| **CLIC-RedCaps** | **76.0** | **61.5** | **12.2** | 76.0 | 59.5 |
| **CLIC-CC12M** | **74.4** | **60.6** | **11.8** | **76.9** | **60.8** |

### 跨架构泛化（ViT-L/14, CLIPS）

| 方法 | SC++ Replace ITT | COCO I→T | COCO T→I |
|------|-----------------|----------|----------|
| CLIPS | 75.5 | 87.3 | 69.9 |
| CLIPS + CLIC | **84.5 (+9%)** | **88.6 (+1.3%)** | **72.1 (+2.2%)** |

### 关键发现

- CLIC 是唯一在 SugarCrepe++ 和检索任务上同时取得一致提升的方法，打破了组合性与检索性的 trade-off
- DAC 在 SugarCrepe 上表现优异（89.4% Replace ITT）但在 SugarCrepe++ 上暴跌至 53.7%，揭示它只学到了词汇级别的敏感性而非真正的语义理解
- CLIC 在不同数据源（LAION、CC12M、RedCaps、COCO）上都能稳定提升，证明方法不依赖特定训练数据
- 将 CLIC 视觉编码器接入 LLaVA-1.5-7b 后，VQAScore 提升但不影响 QA/captioning 等下游任务能力

## 亮点与洞察

- **图像拼接是一个被低估的数据增强策略**：简单地拼接两张图就创造了一个需要组合理解的新场景，组合数平方增长，成本几乎为零。这个思路可以迁移到其他需要组合推理的任务
- **跨图词汇交换 vs LLM 生成 hard negatives**：前者零额外成本且不针对特定基准，后者昂贵且容易过拟合。CLIC 证明简单方法在正确框架下优于复杂方法
- **SugarCrepe++ 暴露了"伪组合性"**：之前方法在 SugarCrepe 上的高分是虚假的——它们只是学会了检测词汇变化而非理解语义。这对该领域的评估标准提出了重要质疑

## 局限与展望

- 仅对文本编码器微调，未探索联合微调或适配器方案的效果
- 词汇交换生成的 hard negative 有一定概率交换语义相同的词（如同义词），这类样本会引入噪声
- 图像拼接改变了图像的宽高比和分辨率，在高分辨率模型上可能引入伪影
- WinoGround 上的提升幅度有限（11.0→12.2），表明对需要精细空间推理的场景仍有改善空间

## 相关工作与启发

- **vs NegCLIP**: NegCLIP 也用词汇交换但只在单张图像描述内交换特定词类（形容词/名词等），CLIC 的跨图交换更自然且不限定词类
- **vs DAC**: DAC 使用 LLM/SAM 生成高质量描述和负样本，计算成本高且在 SugarCrepe++ 上失败，证明复杂不等于有效
- **vs TripletCLIP**: TripletCLIP 额外用扩散模型生成合成图像，成本高且零样本分类性能大幅退化（ImageNet 降 8.5%）

## 评分

- 新颖性: ⭐⭐⭐⭐ 图像拼接+跨图交换的数据构造方案简洁高效，思路新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 跨架构、跨数据源、跨基准的全面验证，包括 LLaVA 集成实验
- 写作质量: ⭐⭐⭐⭐ 论文清晰且自洽，对比实验公平
- 价值: ⭐⭐⭐⭐ 同时解决组合性和检索性的 trade-off，对 CLIP 微调实践有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Breaking the Limits of Open-Weight CLIP: An Optimization Framework for Self-supervised Fine-tuning of CLIP](../../ICLR2026/multimodal_vlm/breaking_the_limits_of_open-weight_clip_an_optimization_framework_for_self-super.md)
- [\[CVPR 2025\] VladVA: Discriminative Fine-tuning of LVLMs](../../CVPR2025/multimodal_vlm/vladva_discriminative_fine-tuning_of_lvlms.md)
- [\[NeurIPS 2025\] Towards Evaluating Proactive Risk Awareness of Multimodal Language Models](towards_evaluating_proactive_risk_awareness_of_multimodal_language_models.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[ICCV 2025\] From Holistic to Localized: Local Enhanced Adapters for Efficient Visual Instruction Fine-Tuning](../../ICCV2025/multimodal_vlm/from_holistic_to_localized_local_enhanced_adapters_for_efficient_visual_instruct.md)

</div>

<!-- RELATED:END -->
