---
title: >-
  [论文解读] Open Ad-Hoc Categorization with Contextualized Feature Learning
description: >-
  [CVPR 2025][可解释性][Ad-Hoc分类] 本文提出了 OAK（Open Ad-hoc Categorization with Contextualized Feature Learning），通过在冻结 CLIP 的输入层引入少量可学习的上下文 token，联合 CLIP 的图文对齐目标和 GCD 的视觉聚类目标，在仅有少数标注样本的条件下实现了自适应的 ad-hoc 类别发现和上下文切换，Stanford Mood 数据集新类别准确率达 87.4%，超过 CLIP 和 GCD 50% 以上。
tags:
  - "CVPR 2025"
  - "可解释性"
  - "Ad-Hoc分类"
  - "上下文学习"
  - "CLIP"
  - "广义类别发现"
---

# Open Ad-Hoc Categorization with Contextualized Feature Learning

**会议**: CVPR 2025  
**arXiv**: [2512.16202](https://arxiv.org/abs/2512.16202)  
**代码**: [https://github.com/Wayne2Wang/OAK](https://github.com/Wayne2Wang/OAK)  
**领域**: 可解释性  
**关键词**: Ad-Hoc分类, 上下文学习, CLIP, 广义类别发现, 可解释性

## 一句话总结

本文提出了 OAK（Open Ad-hoc Categorization with Contextualized Feature Learning），通过在冻结 CLIP 的输入层引入少量可学习的上下文 token，联合 CLIP 的图文对齐目标和 GCD 的视觉聚类目标，在仅有少数标注样本的条件下实现了自适应的 ad-hoc 类别发现和上下文切换，Stanford Mood 数据集新类别准确率达 87.4%，超过 CLIP 和 GCD 50% 以上。

## 研究背景与动机

**领域现状**：传统视觉分类假设类别体系是固定的通用类别（如植物、动物），但实际场景中经常需要 ad-hoc（临时性）分类——比如"在车库促销中可以卖的东西"，这类类别缺乏视觉或语义相似性，是为特定目标动态创建的。CLIP 等开放词汇分类模型通过视觉-语言对齐实现了灵活的分类，但依赖固定的全局语义空间，无法适应不同上下文。GCD（广义类别发现）通过视觉聚类发现新类别，但缺乏语义引导。

**现有痛点**：同一张图片在不同上下文下应被分到完全不同的类别（如"喝水"属于 Action 类、"居民区"属于 Location 类、"专注"属于 Mood 类），但现有方法只能提供单一固定的解释。CLIP 的图像编码器不会根据上下文调整注意力（始终关注显著性物体），GCD 缺乏语义线索容易在复杂 ad-hoc 类别上失败。

**核心矛盾**：ad-hoc 分类与通用分类依赖相同的感知机制，但额外需要上下文化来适应不同目标——如何让模型在保留通用感知能力的同时，根据上下文动态调整特征表示？

**本文目标**：提出 open ad-hoc categorization 任务，给定少量标注样本和大量无标注数据，模型需要（1）推断潜在的上下文，（2）通过语义扩展和视觉聚类来扩展 ad-hoc 类别。

**切入角度**：受认知科学启发——人类识别 ad-hoc 类别使用与通用类别相同的感知机制，但需要上下文化来适应不同目标。因此不修改 CLIP 的感知机制，而是通过可学习的上下文 token 来捕捉数据中隐含的上下文语义。

**核心 idea**：用少量可学习的 context tokens 注入冻结 CLIP 的输入层，配合 GCD 聚类目标和 CLIP 文本引导目标联合训练，实现上下文感知的特征调制和 ad-hoc 类别发现。

## 方法详解

### 整体框架

OAK 基于冻结的 CLIP ViT 图像编码器，对每个上下文独立学习一组 context tokens $\mathbf{z}_c$，将其与图像 patch tokens 一起输入 ViT：$f_c(\mathbf{x}_i) := f([\mathbf{x}_i, \mathbf{z}_c])$。训练使用两个目标的联合：GCD 的对比学习聚类目标 + CLIP 的图文分类目标。推理时通过切换 context tokens 即可获得不同上下文下的分类结果。

### 关键设计

1. **上下文感知的视觉注意力 (Context-aware Visual Attention)**:

    - 功能：使图像特征根据上下文动态调整，引导编码器关注相关区域
    - 核心思路：为每个上下文学习一组 context tokens，作为ViT 的额外输入 token。这些 token 类似于 register tokens，但为每个上下文独立优化，backbone 保持冻结。通过 self-attention 机制，context tokens 的存在会改变注意力模式——例如在 Action 上下文下关注手部，在 Location 下关注背景，在 Mood 下关注面部
    - 设计动机：不同上下文需要关注图像的不同区域。通过在输入层注入可学习的上下文信号，以最小的参数量实现特征空间的上下文化，同时完全保留 CLIP 的预训练感知能力

2. **自下而上的视觉聚类 (Bottom-up Visual Clustering)**:

    - 功能：通过聚类视觉特征发现新类别
    - 核心思路：采用 GCD 的对比学习框架，对无标注数据使用自监督对比损失 $\ell_{\text{self-con}}$，对有标注数据使用监督对比损失 $\ell_{\text{sup-con}}$，联合训练：$\ell_{\text{GCD}}(\mathbf{z}) = (1-\lambda)\ell_{\text{self-con}}(\mathbf{z}; \mathcal{D_U}) + \lambda\ell_{\text{sup-con}}(\mathbf{z}; \mathcal{D_L})$。这个目标只优化 context tokens
    - 设计动机：纯文本引导可能遗漏视觉上可区分但语义不直观的新类别，视觉聚类可以作为互补发现新的类别结构

3. **自上而下的文本引导 (Top-down Text Guidance)**:

    - 功能：利用 CLIP 的语义知识引导聚类与语义类别对齐
    - 核心思路：冻结文本编码器 $g$，在已知类 $\mathcal{Y}_\mathcal{L}$ 和 LLM 生成的潜在新类 $\hat{\mathcal{Y}}_\mathcal{N}$ 上构建分类损失。有标注数据用真实标签，无标注数据通过 SS-KMeans + 匈牙利匹配生成伪标签。总目标为 $\ell_{\text{OAK}}(\mathbf{z}_c) = \ell_{\text{GCD}}(\mathbf{z}_c) + \lambda_{\text{text}} \cdot \ell_{\text{text}}(\mathbf{z}_c)$
    - 设计动机：GCD 目标将类别视为独立实体，忽略了语义关系；文本引导可以利用 CLIP 丰富的语义知识，将视觉聚类与有意义的语义标签对齐

### 损失函数 / 训练策略

- 总体目标是 GCD 对比损失 + 文本引导分类损失的加权和
- 文本引导中对无标注数据使用伪标签（每个 epoch 更新），通过半监督 K-means + 匈牙利匹配获得
- 仅优化 context tokens，CLIP backbone 和文本编码器完全冻结
- 对于每个上下文独立训练一组 context tokens

## 实验关键数据

### 主实验

Stanford 数据集整体准确率（Overall）：

| 方法 | Action | Location | Mood | Omni |
|------|--------|----------|------|------|
| CLIP-ZS + LLM vocab | 65.2 | 47.5 | 55.0 | 43.0 |
| CLIP-ZS + GT vocab | 86.7 | 59.7 | 72.1 | 38.3 |
| GCD | 78.3 | 77.8 | 52.1 | 52.3 |
| **OAK** | **86.9** | **85.9** | **78.4** | **70.3** |

Stanford Novel 类别准确率：

| 方法 | Action | Location | Mood |
|------|--------|----------|------|
| CLIP-ZS + LLM vocab | 38.6 | 34.2 | 35.4 |
| GCD | 67.8 | 80.8 | 40.6 |
| **OAK** | **85.1** | **88.4** | **87.4** |

OAK 在 Mood 新类别上以 87.4% 的准确率大幅超越 CLIP（35.4%）和 GCD（40.6%）。

### 消融实验

Clevr-4 数据集验证了 OAK 在合成数据上同样有效：OAK 在 Texture 上的新类别准确率为 47.8%（GCD 43.6%），Color 上达到 100%。

### 关键发现

- OAK 的显著性图展示了清晰的上下文切换：Action 关注手部，Location 关注背景，Mood 关注面部表情，与人类直觉高度一致
- Omni 准确率（跨所有上下文一致预测正确）方面 OAK（70.3%）远超基线（GCD 52.3%，CLIP 43.0%），显示了出色的上下文一致性
- 文本引导在 CLIP 不太熟悉的概念（如 Location、Mood）上特别有用

## 亮点与洞察

- 提出了一个新颖且有认知科学依据的问题定义——open ad-hoc categorization，很有实际意义
- 方法设计极其简洁：只需在 CLIP 输入层加几个可学习的 token，不修改任何模型架构
- 显著性图结果非常有说服力，直观展示了上下文切换能力
- Omni 准确率指标的引入很有意义，评估了模型在多上下文间无缝切换的能力
- 将 GCD 和 CLIP 的优势有机融合，互补性很强

## 局限与展望

- 新类别名称的发现依赖 LLM 提示，质量受 LLM 局限性影响
- 目前每个上下文独立训练 context tokens，上下文间的知识共享尚未探索
- Stanford 数据集规模较小，在大规模真实场景中的表现有待验证
- 如何自动发现上下文本身（而非给定上下文的类别名称）是更具挑战性的开放问题

## 相关工作与启发

- 与 IC||TC、OpenSMC 等基于 LLM/VLM 的多重聚类方法相比，OAK 不依赖外部 caption，更高效
- GCD（广义类别发现）提供了视觉聚类基础，OAK 在其上增加了上下文化能力
- Visual prompt tuning（VPT）是 context tokens 的技术基础，但 OAK 将其从适配单任务扩展为适配多上下文
- 对 few-shot 视觉分类和上下文化表示学习领域有重要启发

## 评分

- **新颖性**: 9/10 — 问题定义新颖，方法简洁优雅
- **实验充分度**: 8/10 — 多数据集多指标验证，显著性图分析出色
- **写作质量**: 9/10 — 逻辑清晰，与认知科学的联系阐述深入
- **价值**: 8/10 — 开创了 ad-hoc 分类新方向，方法可扩展性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](../../ICCV2025/interpretability/svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)
- [\[CVPR 2025\] Language Guided Concept Bottleneck Models for Interpretable Continual Learning](language_guided_concept_bottleneck_models_for_interpretable_continual_learning.md)
- [\[CVPR 2025\] Learning Visual Composition through Improved Semantic Guidance](learning_visual_composition_through_improved_semantic_guidance.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[AAAI 2026\] Quiet Feature Learning in Algorithmic Tasks](../../AAAI2026/interpretability/quiet_feature_learning_in_algorithmic_tasks.md)

</div>

<!-- RELATED:END -->
