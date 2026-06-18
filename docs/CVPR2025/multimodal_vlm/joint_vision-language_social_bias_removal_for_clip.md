---
title: >-
  [论文解读] Joint Vision-Language Social Bias Removal for CLIP
description: >-
  [CVPR 2025][信息检索/RAG][CLIP去偏] 本文揭示了CLIP模型中图像和文本偏见分布不一致导致的"过度去偏"问题，提出一种双模态偏见对齐+反事实去偏的联合框架，在有效减少性别/年龄/种族偏见的同时保持视觉-语言对齐能力，并设计了ABLE指标综合评估去偏效果与下游性能。 CLIP等视觉-语言预训练模型在分类、…
tags:
  - "CVPR 2025"
  - "信息检索/RAG"
  - "CLIP去偏"
  - "社会偏见消除"
  - "视觉语言对齐"
  - "反事实去偏"
  - "公平性"
---

# Joint Vision-Language Social Bias Removal for CLIP

**会议**: CVPR 2025  
**arXiv**: [2411.12785](https://arxiv.org/abs/2411.12785)  
**代码**: [https://github.com/](https://github.com/) (有，论文中提及)  
**领域**: 多模态VLM  
**关键词**: CLIP去偏、社会偏见消除、视觉语言对齐、反事实去偏、公平性

## 一句话总结
本文揭示了CLIP模型中图像和文本偏见分布不一致导致的"过度去偏"问题，提出一种双模态偏见对齐+反事实去偏的联合框架，在有效减少性别/年龄/种族偏见的同时保持视觉-语言对齐能力，并设计了ABLE指标综合评估去偏效果与下游性能。

## 研究背景与动机
CLIP等视觉-语言预训练模型在分类、检索等下游任务上表现优异，但从web数据中继承了严重的社会偏见（如将"职业"与特定性别关联）。现有去偏方法主要从单一模态的embedding中移除偏见信息，但这带来一个核心矛盾：**去偏后V-L对齐能力大幅下降**，即所谓的"过度去偏"（over-debiasing）问题。

作者进一步探究发现：(1) 社会偏见同时存在于图像和文本两个模态中；(2) 两个模态中的偏见分布差异很大（如gender-career偏见在图像中显著、gender-science偏见在文本中显著）。因此，像CLIP-clip那样假设两个模态偏见相同并用相同维度去偏是不合理的。

**核心idea**: 先对齐两个模态的偏见分布，再联合移除偏见，同时通过反事实目标保持V-L对齐能力。

## 方法详解

### 整体框架
冻结原始CLIP编码器，在其后接一个可学习的偏见对齐模块 $\mathrm{BA}(\cdot;\theta_{ba})$。训练数据为带属性标签的人脸图像-文本对（如FairFace）。训练时通过偏见对齐损失 $\mathcal{L}_{ba}$ 和反事实去偏损失 $\mathcal{L}_{cd}$ 联合优化；推理时通过 $\bar{\phi}(t) = f(t) - \mathrm{BA}(f(t))$ 得到去偏embedding。

### 关键设计
1. **偏见信息解耦**:
    - 功能：将CLIP embedding分解为偏见分量和中性分量
    - 核心思路：$f(t) = \phi(t) + \bar{\phi}(t)$，其中 $\phi(t)$ 为偏见信息，$\bar{\phi}(t)$ 为中性信息。BA模块输出 $\phi(t)$，减去即得去偏embedding
    - 设计动机：社会偏见作为可加性分量嵌入embedding中，可通过学习并减去的方式消除

2. **双模态偏见对齐（Dual-Bias Alignment）**:
    - 功能：在去偏前先将图像和文本的偏见分布对齐
    - 核心思路：维护图像和文本embedding队列 $\mathcal{Q}_v, \mathcal{Q}_t$（类似MoCo），计算偏见embedding与队列的相似度伪分布 $p(t_i), p(v_i)$，通过KL散度损失 $\mathcal{L}_{ba} = \frac{1}{N}\sum D_{KL}(p(t_i) \| p(v_i))$ 对齐两个分布
    - 设计动机：直接element-wise匹配会丢失背景信息和特征多样性，通过分布层面的对齐更灵活且保留信息

3. **反事实去偏（Counterfactual Debiasing）**:
    - 功能：拉近同一中性概念不同属性的去偏embedding，同时保持V-L对齐
    - 核心思路：对文本构造反事实对（如"male dancer"↔"female dancer"），用交叉熵损失拉近去偏后的相似度分布与原始分布：$\mathcal{L}_{cd}^t = -\frac{1}{N}\sum\sum s_t(t_i,v,\mathcal{V}_q)\log\bar{s}_t(a(t_i,t'_i),v,\mathcal{V}_q)$，其中 $a(t_i,t'_i)$ 以50%概率随机选择文本或其反事实版本
    - 设计动机：仅对齐偏见不够，还需确保去偏后保持原始的V-L对齐能力，避免下游任务性能退化

### 损失函数 / 训练策略
总损失为：$\mathcal{L} = \alpha \mathcal{L}_{cd} + (1-\alpha)\mathcal{L}_{ba}$，其中 $\alpha \in [0,1]$ 平衡两个目标。CLIP编码器始终冻结，仅训练BA模块参数 $\theta_{ba}$。推理时BA模块作为即插即用组件。

## 实验关键数据

### 主实验（ViT-B/16，FairFace训练）

| 设置 | 方法 | MaxSkew↓(域内) | NDKL↓(域内) | IN1K Top1↑ | Flickr TR↑ | ABLE↑ |
|------|------|-------------|------------|-----------|-----------|-------|
| Gender | Original CLIP | 0.218 | 0.088 | 68.31 | 96.4 | 73.87 |
| Gender | CLIP-clip | 0.103 | 0.026 | 68.00 | 95.4 | 77.55 |
| Gender | Biased-prompts | 0.161 | 0.048 | 65.07 | 94.3 | 73.78 |
| Gender | **Ours** | **0.080** | **0.025** | 68.05 | **96.6** | **78.35** |
| Age | Original CLIP | 0.657 | 0.433 | 68.31 | 96.4 | 58.94 |
| Age | **Ours** | **0.608** | **0.294** | **68.34** | 96.0 | **60.61** |

### 消融实验（ViT-B/16, Gender, FairFace）

| 配置 | MaxSkew↓ | NDKL↓ | IN1K Top1↑ | ABLE↑ | 说明 |
|------|---------|-------|-----------|-------|------|
| Ours (complete) | 0.080 | 0.025 | 68.05 | 78.35 | 完整方法 |
| w/o $\mathcal{L}_{cd}$ | 0.167 | 0.056 | 68.28 | 75.58 | 去掉反事实损失，偏见增加 |
| w/o $\mathcal{L}_{ba}$ | 0.095 | 0.033 | 67.84 | 77.71 | 去掉对齐损失，性能略降 |

### 关键发现
- 两个损失均不可或缺：$\mathcal{L}_{cd}$ 对减少偏见贡献更大，$\mathcal{L}_{ba}$ 对保持V-L对齐更关键
- 方法在4种ViT backbone（B/16, B/32, L/14, H/14）上均一致有效
- 域外泛化性强：在FairFace上训练，UTKFace和FACET上也能有效去偏
- 可同时去除多种偏见（性别+年龄+种族），更适合实际部署

## 亮点与洞察
- **问题发现有价值**：证明了V-L模型中偏见在两个模态的分布不同，直接解释了现有方法的失败原因
- **ABLE指标设计巧妙**：用调和平均数综合评估去偏程度和下游性能，解决了以往只看一面的问题
- **方法简洁高效**：仅需训练一个轻量BA模块，CLIP完全冻结，可即插即用

## 局限与展望
- 依赖带属性标签的人脸数据集（FairFace/UTKFace）训练
- 图像侧无法构造反事实样本（生成模型质量不够），只能用单向图像去偏损失
- 仅在检索和分类任务上评估，未验证对文生图等生成任务的影响
- 偏见类型受限于训练数据的标注类别

## 相关工作与启发
- 与CLIP-clip（基于互信息维度裁剪）和Biased-prompts（基于投影矩阵）形成互补
- 分布对齐思路借鉴了MoCo的动量队列机制，应用到偏见对齐场景
- 反事实去偏思路可推广到其他V-L模型（如BLIP系列）

## 评分
- 新颖性: ⭐⭐⭐⭐ 问题分析深入，双模态偏见对齐思路新颖，但基本框架（对齐+去偏）并不复杂
- 实验充分度: ⭐⭐⭐⭐⭐ 4种backbone、3种偏见类型、域内域外评估、消融实验齐全
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，从问题分析到方法设计环环相扣
- 价值: ⭐⭐⭐⭐ 为V-L模型公平性研究提供了新的分析视角和实用方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] HiMo-CLIP: Modeling Semantic Hierarchy and Monotonicity in Vision-Language Alignment](../../AAAI2026/information_retrieval/himo-clip_modeling_semantic_hierarchy_and_monotonicity_in_vi.md)
- [\[ICLR 2026\] Efficient Discriminative Joint Encoders for Large Scale Vision-Language Re-ranking](../../ICLR2026/information_retrieval/efficient_discriminative_joint_encoders_for_large_scale_vision-language_rerankin.md)
- [\[ICCV 2025\] ViLU: Learning Vision-Language Uncertainties for Failure Prediction](../../ICCV2025/information_retrieval/vilu_learning_vision-language_uncertainties_for_failure_prediction.md)
- [\[ACL 2025\] Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms](../../ACL2025/information_retrieval/llm_reranking_harmful_content.md)
- [\[ACL 2025\] Semantic Outlier Removal with Embedding Models and LLMs](../../ACL2025/information_retrieval/semantic_outlier_removal_with_embedding_models_and_llms.md)

</div>

<!-- RELATED:END -->
