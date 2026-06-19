---
title: >-
  [论文解读] SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding
description: >-
  [ECCV 2024][目标检测][temporal grounding] 针对组合时序定位任务中现有方法负样本构造不合理、DETR 模型对负查询无法产生合理显著性响应的问题，提出利用 LLM（GPT-3.5 Turbo）生成语义可行的分层硬负样本，并设计粗到细的显著性排序策略建立视频片段与层次负查询之间的多粒度语义关系，显著提升组合泛化能力。
tags:
  - "ECCV 2024"
  - "目标检测"
  - "temporal grounding"
  - "compositional generalization"
  - "hard negatives"
  - "saliency ranking"
  - "DETR"
---

# SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding

**会议**: ECCV 2024  
**arXiv**: [2407.05118](https://arxiv.org/abs/2407.05118)  
**代码**: [https://github.com/zxccade/SHINE](https://github.com/zxccade/SHINE)  
**领域**: 目标检测  
**关键词**: temporal grounding, compositional generalization, hard negatives, saliency ranking, DETR

## 一句话总结

针对组合时序定位任务中现有方法负样本构造不合理、DETR 模型对负查询无法产生合理显著性响应的问题，提出利用 LLM（GPT-3.5 Turbo）生成语义可行的分层硬负样本，并设计粗到细的显著性排序策略建立视频片段与层次负查询之间的多粒度语义关系，显著提升组合泛化能力。

## 研究背景与动机

时序定位（temporal grounding）旨在根据自然语言查询定位视频中的对应片段。自然语言的组合性使得查询可以描述超出预定义事件范围的新场景，这就要求模型具备组合泛化能力——在训练时只见过已知词汇的组合，推理时能定位由已知词的新组合描述的视频片段。

**现有痛点**：

**负样本构造不合理**：现有方法（VISA、SSL2CG）仅关注主要词性（动词和名词），忽略了介词、副词等非主导词在语义中的关键作用（如 "on/under the table"、"turn on/off the light"，换一个介词/副词就完全改变语义）。Deco 通过随机采样和重组构造负样本，产生大量语义不合理的组合（如 "eating the table"、"reading the door"），反而误导模型学习不现实的差异。

**DETR 模型的显著性响应不合理**：近年来 DETR-based 方法（Moment-DETR、QD-DETR）在时序定位中兴起，它们在定位片段的同时预测每个视频 clip 对查询的显著性分数。但实验发现这些模型面对负查询时的显著性响应不合理——硬负查询的显著性分数甚至超过正查询，表明模型无法捕捉原始查询与细微变化后查询之间的语义差异。

**核心矛盾**：要实现组合泛化，模型需要精确理解每个词汇在查询中的作用；但现有负样本的随机构造方式要么产生不现实的组合、要么只覆盖部分词性，无法教会模型细粒度的组合语义。

**本文切入角度**：(1) 用 LLM 的语言知识保证负样本的语义可行性，同时覆盖所有五类词性；(2) 利用 DETR 框架中现有的显著性分数机制，建立正查询与多层次负查询之间的显著性排序约束。

## 方法详解

### 整体框架

给定视频-查询对 $(V_p, Q_p)$：
1. 通过渐进式 mask-and-predict 策略生成 3 层硬负查询 $\{Q_{hn}^1, Q_{hn}^2, Q_{hn}^3\}$
2. 从同一 mini-batch 随机采样一个无关负查询 $Q_n$
3. 所有查询与视频分别经视频编码器和文本编码器提取特征
4. 经 DETR 编码器交互，预测各查询的显著性分数 $\{S_p, S_{hn}^1, S_{hn}^2, S_{hn}^3, S_n\}$
5. 用粗粒度排序损失 $\mathcal{L}_{cr}$ 拉开正/负查询的显著性差距
6. 用细粒度排序损失 $\mathcal{L}_{fr}$ 约束层次负查询之间的显著性梯度
7. 与 baseline 基础损失 $\mathcal{L}_{base}$ 联合优化

### 关键设计

1. **LLM 驱动的层次硬负样本构造**

    - **功能**：为每个查询生成 3 个语义递进变化的硬负查询
    - **核心思路**：
     - 先用 spaCy 对训练集所有查询做词性标注，按五类词性（动词、名词、形容词、介词、副词）构建词典 $D$
     - 按语言学重要性（动词→名词→形容词→介词→副词）渐进 mask 原始查询中的词汇，mask 比例分别为 25%、50%、75%（Charades-CG）
     - 不用随机采样填充，而是将 mask 后的查询和词典子集交给 GPT-3.5 Turbo，让 LLM 生成语义可行但与原查询不同的替换词
    - **设计动机**：随机采样产生大量不合理组合（"eating the table"），误导模型。LLM 具有语言常识，能确保生成的负查询在语义上合理（如 "person picks up the book" → "person throws the pen"），从而让模型从真实可能的混淆场景中学习辨别。渐进 mask 则确保 3 层负查询的语义与正查询的差距逐级递增，为细粒度排序提供基础。

2. **粗粒度显著性排序 (Coarse-Grained Saliency Ranking)**

    - **功能**：在视频级别建立显著性先验约束
    - **核心思路**：包含两条约束：
     - **Intra-ranking** $\mathcal{L}_{intra}$：正查询在 ground-truth 区间内的显著性应高于区间外
     - **Inter-ranking** $\mathcal{L}_{inter}$：正查询在 ground-truth 区间内的显著性应高于负查询的显著性

     $$\mathcal{L}_{cr} = \max(0, h_1 + S_p^- - S_p^+) + \max(0, h_2 + S_n^+ - S_p^+)$$

     其中 $S^+$ 取 ground-truth 区间内的 top-k 平均值（$k = \max(1, \lfloor T^+/q \rfloor)$），通过 $q$ 自适应不同长度的区间。
    - **设计动机**：现有 DETR 的显著性损失只约束了正查询自身的内部一致性，缺乏正/负查询之间的对比。增加 inter-ranking 约束可以显式拉大正负查询的显著性差距。

3. **细粒度显著性排序 (Fine-Grained Saliency Ranking)**

    - **功能**：约束层次硬负查询之间的显著性呈梯度递降
    - **核心思路**：要求显著性分数满足严格的层次结构——正查询 > 1 层负 > 2 层负 > 3 层负 > 无关负，通过多级 margin ranking loss 实现：

     $$\mathcal{L}_{fr} = \sum_{i=0}^{3} \max(0, m_i + d(S_p, S_{hn}^i) - d(S_p, S_{hn}^{i+1}))$$

     其中 $d(\cdot)$ 是负对数似然度量，衡量显著性分数在时间维度上的分布差异。
    - **设计动机**：3 层负查询的语义与正查询的差距不同——1 层负替换了少量词（语义相近），3 层负替换了大量词（语义差异大）。模型应该对语义越接近的负查询产生越高的显著性（但仍低于正查询），反之越低。这种层次约束迫使模型学习细粒度的词汇-视频对应关系。

### 损失函数 / 训练策略

$$\mathcal{L} = \mathcal{L}_{base} + \alpha \mathcal{L}_{cr} + \beta \mathcal{L}_{fr}$$

- $\mathcal{L}_{base}$：DETR 基础损失（二部图匹配 + 时刻定位 + 显著性损失）
- $\alpha = \beta = 1.0$（经网格搜索确定），margin $h_1=1.0, h_2=2.0, m_0 \sim m_3=0.25$
- Charades-CG 上 QD-DETR 学习率 0.0001，Moment-DETR 学习率 0.0002
- 单卡 NVIDIA A100，batch size 32，训练 200 epochs

## 实验关键数据

### 主实验

**Charades-CG**：

| 方法 | Test-Trivial R1@0.5 | Novel-Composition R1@0.5 | Novel-Composition R1@0.7 | Novel-Composition mIoU |
|------|---------------------|-------------------------|--------------------------|----------------------|
| QD-DETR | 59.24 | 42.30 | 21.09 | 38.55 |
| **QD-DETR + SHINE** | **60.66** | **50.23 (+7.93)** | **27.69 (+6.60)** | **44.14 (+5.59)** |
| Moment-DETR | 49.48 | 39.42 | 18.62 | 36.61 |
| **MD + SHINE** | **57.14 (+7.66)** | **44.65 (+5.23)** | **23.21 (+4.59)** | **39.86 (+3.25)** |
| DeCo (prev SOTA) | 58.75 | 47.39 | 21.06 | 40.70 |

**ActivityNet-CG**：

| 方法 | Test-Trivial R1@0.5 | Novel-Composition R1@0.5 | Novel-Composition mIoU |
|------|---------------------|-------------------------|----------------------|
| QD-DETR | 41.80 | 26.91 | 31.01 |
| **QD-DETR + SHINE** | **43.76** | **29.56 (+2.65)** | **32.44 (+1.43)** |

### 消融实验

**粗粒度排序约束消融（Charades-CG Novel-Composition）**：

| 配置 | R1@0.5 | R1@0.7 | mIoU | 说明 |
|------|--------|--------|------|------|
| $\mathcal{L}_{base}$ only | 42.30 | 21.09 | 38.55 | baseline |
| + $\mathcal{L}_{intra}$ | 44.02 | 22.84 | 39.23 | intra 提升 |
| + $\mathcal{L}_{inter}$ | 46.69 | 24.87 | 41.74 | inter 贡献更大 |
| + 两者 | 46.25 | 24.93 | 41.88 | 组合进一步提升 |

**LLM vs 随机采样（Charades-CG Novel-Composition）**：

| 负样本来源 | R1@0.5 | R1@0.7 | mIoU |
|-----------|--------|--------|------|
| 随机采样 | 47.41 | 25.33 | 42.50 |
| Llama 3 | 48.75 | 25.22 | 42.89 |
| Gemini-1.5 Flash | 48.69 | 25.60 | 43.54 |
| **GPT-3.5 Turbo** | **50.23** | **27.69** | **44.14** |

**介词和副词的贡献（Charades-CG Novel-Composition）**：

| 配置 | R1@0.5 | mIoU |
|------|--------|------|
| QD+Ours w/o prep & adv | 48.87 | 43.30 |
| **QD+Ours** | **50.23** | **44.14** |

### 关键发现

- **$\mathcal{L}_{inter}$ 贡献大于 $\mathcal{L}_{intra}$**：正负查询之间的显著性差距约束比区间内外的约束对组合泛化更重要
- **细粒度约束需要完整层次才能发挥最大作用**：单独添加 $\mathcal{L}_{fr}^3$（最深层）反而轻微降低性能，但在完整层次约束中加入后提升 R1@0.7 达 3.17%
- **GPT-3.5 Turbo 生成的负样本最优**：比随机采样高 2.82% R1@0.5，LLM 的语言常识确保了负样本的合理性
- 加入介词和副词的负样本构造有效提升模型对非主导词的感知

## 亮点与洞察

- **首个将 DETR 显著性分数与组合泛化建立联系**：发现现有 DETR 的显著性响应对负查询不合理这一问题本身就是重要洞察
- **LLM 作为负样本生成器**：利用 LLM 的语言常识保证负样本语义可行性，比随机替换更有效的负样本构造范式
- **即插即用设计**：SHINE 可以无缝集成到任何 DETR-based 时序定位模型，不修改模型架构，只增加训练时损失
- **覆盖全部五类词性**：介词和副词虽非主导词性但对语义影响巨大，将它们纳入负样本构造是一个值得借鉴的思路
- **粗到细层次约束**：从视频级别到时间分布级别的多粒度排序约束策略，可迁移到其他需要层次对比学习的任务

## 局限与展望

- 层次负样本的 mask 比例（25/50/75%）是手动设定的，不同数据集的最优比例不同（ActivityNet-CG 用 10/30/50%）
- LLM 调用增加训练前的预处理成本（GPT-3.5 Turbo API 费用），且 LLM 生成质量不完全可控
- 仅在两个组合泛化基准上验证，未测试在标准时序定位基准（Charades-STA、ActivityNet Captions）上的泛化性能
- 查询较长时 LLM 相对于随机替换的优势减弱（ActivityNet-CG 上提升不如 Charades-CG 显著）
- 细粒度排序的 pseudo saliency score（区间内 1/区间外 0）是粗糙的二值标签，未考虑边界附近的渐变

## 相关工作与启发

- **vs DeCo**: DeCo 也用 decompose-reconstruct 策略构造负样本，但采用随机采样产生大量不合理组合；SHINE 用 LLM 保证语义可行性，且加入粗到细排序约束
- **vs SSL2CG**: SSL2CG 通过 mask 不同词汇生成等变和不变样本做对比学习，但仅考虑动词和名词；SHINE 扩展到五类词性并用 LLM 生成合理替换
- **vs QD-DETR**: SHINE 作为插件不修改 QD-DETR 架构，仅增加训练约束即可大幅提升其组合泛化（R1@0.5 +7.93%）

## 评分

- 新颖性: ⭐⭐⭐⭐ 将 LLM 用于负样本构造与显著性排序的结合是有创意的组合
- 实验充分度: ⭐⭐⭐⭐ 两个基准、两个 baseline、详细消融、多个 LLM 对比
- 写作质量: ⭐⭐⭐⭐ 问题分析透彻，方法描述清晰，可视化说服力强
- 价值: ⭐⭐⭐⭐ 即插即用的设计实用性好，但受限于组合泛化这一细分方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Sim-DETR: Unlock DETR for Temporal Sentence Grounding](../../ICCV2025/object_detection/sim-detr_unlock_detr_for_temporal_sentence_grounding.md)
- [\[ECCV 2024\] BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos](bam-detr_boundary-aligned_moment_detection_transformer_for_temporal_sentence_gro.md)
- [\[ECCV 2024\] Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)
- [\[ECCV 2024\] ReGround: Improving Textual and Spatial Grounding at No Cost](reground_improving_textual_and_spatial_grounding_at_no_cost.md)
- [\[CVPR 2026\] FALCON: False-Negative Aware Learning of Contrastive Negatives in Vision-Language Alignment](../../CVPR2026/object_detection/falcon_false-negative_aware_learning_of_contrastive_negatives_in_vision-language.md)

</div>

<!-- RELATED:END -->
