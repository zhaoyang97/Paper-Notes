---
title: >-
  [论文解读] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning
description: >-
  [ICCV 2025][多模态VLM][数据高效预训练] 受人类婴儿高效学习能力的启发，提出BabyVLM框架，包括合成训练数据集（将通用数据转化为儿童导向的格式）和多个发展对齐的评估基准，实现了紧凑VLM在有限数据下的高效预训练，性能优于仅用SAYCam或通用数据训练的模型。 当前大规模视觉语言模型（如LLaVA、CLIP…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "数据高效预训练"
  - "婴儿学习启发"
  - "视觉语言模型"
  - "发展心理学"
  - "合成数据"
---

# BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning

**会议**: ICCV 2025  
**arXiv**: [2504.09426](https://arxiv.org/abs/2504.09426)  
**代码**: [https://github.com/shawnking98/BabyVLM](https://github.com/shawnking98/BabyVLM) (项目主页: shawnking98.github.io/BabyVLM)  
**领域**: 多模态VLM  
**关键词**: 数据高效预训练, 婴儿学习启发, 视觉语言模型, 发展心理学, 合成数据

## 一句话总结
受人类婴儿高效学习能力的启发，提出BabyVLM框架，包括合成训练数据集（将通用数据转化为儿童导向的格式）和多个发展对齐的评估基准，实现了紧凑VLM在有限数据下的高效预训练，性能优于仅用SAYCam或通用数据训练的模型。

## 研究背景与动机
当前大规模视觉语言模型（如LLaVA、CLIP等）的训练依赖海量数据和昂贵计算资源，训练成本动辄数千GPU小时，独立研究者难以承受。然而，人类婴儿能从极其有限的感官输入中快速习得复杂的认知和感知能力，这暗示着精心设计的小规模数据同样可以训练出有效的表征。

现有的婴儿启发数据集如SAYCam虽然提供了婴儿视角的视听数据，但存在两大问题：（1）SAYCam仅记录了婴儿部分日常经验，覆盖面不足；（2）现有评估基准要么过于简单（如Labeled-S仅评估分类），要么与训练数据领域不对齐（如VQA、Winoground为大模型设计），无法准确衡量紧凑模型的发展对齐能力。

本文的核心切入点是：通过"儿童导向转化"将通用大规模数据转化为符合婴儿学习环境的格式，并设计与训练域对齐的多样化评估任务，填补当前婴儿启发VLM训练与评估之间的空白。

## 方法详解

### 整体框架
BabyVLM框架包含四个核心组件：（1）过滤后的SAYCam数据集子集；（2）通过"儿童导向转化"生成的合成训练数据；（3）BabyLLaVA生成式基线模型；（4）四个发展对齐的评估基准。

### 关键设计

1. **SAYCam数据过滤**:

    - 功能：从SAYCam原始视频中提取图像-语句对，并基于CLIP相似度过滤
    - 核心思路：保留CLIP相似度阈值 > 0.2的高质量图文对，确保图像与文本的高相关性
    - 过滤后获得约67K个图文对
    - 设计动机：原始SAYCam数据中存在大量低质量的图文配对，直接使用会引入噪声

2. **合成数据转化（Transferred Dataset）**:

    - 功能：将CC3M、LAION、SBU等通用数据集转化为婴儿学习风格的数据
    - 核心思路：分两步实现
        - **第一步**：用GPT-4o将原始图像描述重写为简单的"儿童导向语句"，模拟与两岁幼儿对话的语言风格。GPT-4o同时过滤与婴儿日常经验不相关的图文对
        - **第二步**：使用匈牙利算法（Hungarian matching），以CLIP相似度为距离度量，从转化后的数据中选择与SAYCam图像视觉最相似的子集，确保视觉一致性
    - 设计动机：SAYCam仅覆盖婴儿有限的日常经验，需要更多样化的数据来模拟婴儿从更广泛环境中学习的过程

3. **BabyLLaVA生成式基线**:

    - 功能：构建完全在发展数据上从头训练的紧凑生成式VLM
    - 核心思路：受LLaVA启发，将紧凑的语言模型GPT-2（7M参数）与视觉编码器ResNeXt-50（23M参数）通过轻量级MLP连接器整合。也提供了更大的变体（Llama-1.1B + ViT-L）
    - 设计动机：验证紧凑模型在发展数据约束下能否学到有意义的多模态表征

4. **评估基准设计**:

    - **Labeled-S**：经典分类任务，从4个候选图像中匹配目标类别
    - **视觉双词测试（VTWT）**：灵感来自18-24月龄的"双词阶段"，测试组合语义推理（如"wash cup" vs. "fill cup"）。使用GPT-4o生成5117个短语对，人工筛选后保留967对
    - **Baby Winoground**：扩展VTWT，需同时匹配两组图文对，负样本图像通过Stable Diffusion生成，测试更高级的视觉-语言组合推理
    - **SAYCam Caption**：生成式字幕评估，使用METEOR指标衡量模型生成儿童导向描述的能力

### 损失函数 / 训练策略
BabyLLaVA沿用LLaVA的标准训练流程，在编译的发展数据上训练。CVCL（对比模型）使用标准对比学习损失。模型设计遵循三个原则：（1）发展合理的复杂度；（2）有限的泛化边界；（3）语言和视觉的简单性。

## 实验关键数据

### 主实验

| 模型 | Labeled-S | VTWT | Baby Winoground (Overall) | SAYCam Caption |
|------|-----------|------|---------------------------|----------------|
| CLIP-large（上界） | 0.710 | 0.863 | 0.674 | N/A |
| LLaVA-v1.5-7B（上界） | 0.740 | 0.785 | 0.427 | 0.166 |
| CVCL（对比式baby模型） | 0.609 | 0.649 | 0.093 | N/A |
| BabyLLaVA-GPT2 | 0.420 | 0.625 | 0.066 | 0.138 |
| BabyLLaVA-Llama | 0.420 | 0.603 | 0.052 | 0.129 |
| 随机猜测 | 0.250 | 0.500 | 0.167 | N/A |

### 消融实验

| 训练配置 | Labeled-S | VTWT | Baby Winoground (Overall) | 说明 |
|----------|-----------|------|---------------------------|------|
| CVCL-filtered | 0.609 | 0.649 | 0.093 | 仅SAYCam |
| CVCL-filtered-aug | 0.581 | 0.702 | 0.203 | SAYCam+合成数据（↑显著） |
| CVCL-filtered-random | 0.602 | 0.684 | 0.107 | SAYCam+随机通用数据 |
| BabyLLaVA-filtered | 0.420 | 0.625 | 0.066 | 仅SAYCam |
| BabyLLaVA-filtered-aug | 0.536 | 0.693 | 0.082 | SAYCam+合成数据（↑显著） |
| BabyLLaVA-aug-only | 0.500 | 0.624 | 0.063 | 仅合成数据 |

### 关键发现
- 对比模型（CVCL）在判别任务上始终优于生成式模型（BabyLLaVA），符合对比学习更适合判别任务的认知
- 更大的BabyLLaVA-Llama（参数量约为GPT2版的50倍）性能相当甚至更差，说明在有限数据上存在过拟合
- 合成数据带来的提升显著优于随机通用数据，验证了"儿童导向转化"的有效性
- Baby Winoground揭示分布内/外的不对称性：baby模型在正向上下文（分布内）表现尚可，但在负向上下文（分布外）低于随机猜测
- VTWT去除视觉输入后准确率降至~53%（接近随机），证明该任务确实测试了真正的多模态推理

## 亮点与洞察
- 将发展心理学的洞见（婴儿双词阶段、名词优先偏好等）与VLM训练相结合，跨领域思路新颖
- 合成数据方法可推广到其他资源受限领域的高效训练
- 组合推理分析发现模型在名词差异上表现最好，与语言发展文献中"婴儿名词使用频率是动词两倍"的发现一致
- 证明了"精心策划的小数据 + 紧凑模型"可以学到有意义的表征，为资源受限的模型训练提供了新范式

## 局限与展望
- 生成式字幕任务表现仍然不佳，所有模型的METEOR分数都很低
- 在Baby Winoground上baby模型表现远低于上界模型，组合推理能力仍有巨大提升空间
- 合成数据仍然依赖GPT-4o和大规模源数据集，"发展合理性"存在争议
- 未探索时序上下文、更丰富的物体交互等额外模态信号
- 框架主要在SAYCam域内验证，跨域泛化能力有待探索

## 相关工作与启发
- **vs CVCL (Vong et al.)**: CVCL是对比式baby模型，本文在其基础上引入生成式模型和更丰富的合成数据
- **vs BabyLM Challenge**: BabyLM关注纯语言的发展式训练，本文扩展到多模态视觉-语言场景
- **vs LLaVA**: BabyLLaVA借鉴了LLaVA的架构但大幅缩小模型规模，聚焦发展约束下的学习

## 评分
- 新颖性: ⭐⭐⭐⭐ 将发展心理学与VLM训练结合，视角独特，评估基准设计有创意
- 实验充分度: ⭐⭐⭐⭐ 消融实验详实，多角度分析模型行为与语言发展理论的对应
- 写作质量: ⭐⭐⭐⭐ 结构清晰，动机阐述充分，跨学科表述到位
- 价值: ⭐⭐⭐ 更偏认知科学导向，直接工程应用有限，但为数据高效训练提供了新思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining](../../NeurIPS2025/multimodal_vlm/adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)
- [\[ICCV 2025\] Training-free Generation of Temporally Consistent Rewards from VLMs](training-free_generation_of_temporally_consistent_rewards_from_vlms.md)
- [\[CVPR 2026\] Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning](../../CVPR2026/multimodal_vlm/similarity-as-evidence_calibrating_overconfident_vlms_for_interpretable_and_labe.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[ACL 2026\] Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding](../../ACL2026/multimodal_vlm/learning_more_from_less_exploiting_counterfactuals_for_data-efficient_chart_unde.md)

</div>

<!-- RELATED:END -->
