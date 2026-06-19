---
title: >-
  [论文解读] VAQUUM: Are Vague Quantifiers Grounded in Visual Data?
description: >-
  [ACL 2025][vague quantifiers] 本文发布了VAQUUM数据集（20,300条人类评分，1,089张图片），系统评估视觉语言模型在模糊量词（few/many等）使用上与人类的一致性，发现VLM像人类一样受物体数量影响，但不同评估范式下模型表现差异大，表明判断和生成模糊量词依赖不同认知过程。
tags:
  - "ACL 2025"
  - "vague quantifiers"
  - "visual grounding"
  - "VLM evaluation"
  - "human judgment"
  - "多模态"
---

# VAQUUM: Are Vague Quantifiers Grounded in Visual Data?

**会议**: ACL 2025  
**arXiv**: [2502.11874](https://arxiv.org/abs/2502.11874)  
**代码**: [https://github.com/hughmee/vaquum](https://github.com/hughmee/vaquum)  
**领域**: 其他  
**关键词**: vague quantifiers, visual grounding, VLM evaluation, human judgment, multimodal

## 一句话总结

本文发布了VAQUUM数据集（20,300条人类评分，1,089张图片），系统评估视觉语言模型在模糊量词（few/many等）使用上与人类的一致性，发现VLM像人类一样受物体数量影响，但不同评估范式下模型表现差异大，表明判断和生成模糊量词依赖不同认知过程。

## 研究背景与动机
1. **领域现状**：模糊量词（如 a few、many）在日常对话中广泛使用，其含义受上下文因素影响（如物体数量、大小、个人信念等）。在NLP和多模态研究中，大部分量词研究聚焦于精确量词（all、none），对模糊量词的建模关注较少。
2. **现有痛点**：现有研究多将模糊量词定义为固定比例（如few = <17%），消除了其"模糊性"本质；缺乏大规模的人类判断数据来衡量VLM在视觉场景中对模糊量词的理解；评估方法单一，无法全面反映模型行为。
3. **核心矛盾**：模糊量词的使用依赖复杂的视觉和语境因素（数量、面积、真实世界大小），我们不知道VLM是否能像人类一样根据这些因素灵活调整量词使用，更不清楚不同评估方法是否能得到一致的结论。
4. **本文目标** (1) 视觉场景中哪些因素影响人类对模糊量词的判断？(2) VLM是否在这些因素上与人类行为一致？(3) 不同评估范式（生成概率、数值评分、多选题）是否给出一致结果？
5. **切入角度**：构建一个包含丰富视觉特征标注（物体数量、分割面积、真实世界大小规范）的数据集，使用三种不同评估方式（概率提取、提示评分、多选题）来全面评估VLM与人类的对齐程度。
6. **核心 idea**：通过三种评估范式揭示VLM在模糊量词使用上与人类部分对齐但高度依赖评估方法的事实。

## 方法详解

### 整体框架
输入是自然图片+量化陈述（"There are [QUANT] [OBJECT] in the image"，其中QUANT ∈ {few, a few, some, many, a lot of}），输出是该陈述的"恰当性"评分。从人类和VLM两侧收集评分，然后比较对齐程度。VLM侧使用三种评估方式：生成概率提取、LLM数值评分和多选题。

### 关键设计

1. **VAQUUM数据集构建**:

    - 功能：提供多维视觉特征标注的人类量词判断基准
    - 核心思路：从FSC-133（物体计数数据集，7-3731个物体）和TallyQA（2-15个物体）合并图片，将99种不同计数分为33个bin，每bin采样33张图片得到1,089张。标注三种视觉特征：(a) 物体计数（分bin均匀采样）；(b) 分割面积（用CLIPSeg估计物体占图片面积比例）；(c) 真实世界大小规范（从THINGSplus数据库获取人类评定的物体典型大小）。招募203名英语母语参与者，每人评100个陈述，用滑块评分恰当性
    - 设计动机：现有数据集要么用人工图片（不真实），要么用固定比例定义量词（消除了模糊性），本数据集使用自然图片+连续评分+多维特征

2. **线性混合效应模型分析人类判断**:

    - 功能：量化各视觉因素对人类量词判断的影响
    - 核心思路：以量词、计数、分割面积、大小规范为固定效应，参与者和物体类别为随机效应，拟合LMM预测人类评分。模型解释了50.3%的总方差（$R^2_c = 0.503$）。交互效应显示few/a few与计数负相关（$\beta = -0.37/-0.38$），many/a lot of与计数正相关（$\beta = 0.38/0.42$），面积和大小规范效应方向一致但更弱
    - 设计动机：精确建立人类行为的统计模型，作为VLM评估的参考基线

3. **三种VLM评估范式**:

    - 功能：从不同角度全面评估VLM的量词使用能力
    - 核心思路：(a) 生成概率（Experiment 1）：提示VLM"How would you describe the amount of [OBJECT]?"并提取对各量化陈述的log概率，按token长度归一化；(b) 数值评分（Experiment 2）：提示VLM直接给出陈述恰当性的数值分数；(c) 多选题（Experiment 3）：要求VLM从六个量化陈述中选最恰当的一个。对5个VLM（BLIP-2、InstructBLIP、LLaVA-NeXT、LLaVA-OneVision、Molmo）比较各评估方式下与人类判断的相关性
    - 设计动机：不同评估方式测量的可能是不同能力——概率反映内在语言建模偏好，数值评分需要元认知能力，多选题需要比较和判断能力

## 实验关键数据

### 主实验 — 生成概率与人类判断的Spearman相关

| 模型 | few | a few | some | many | a lot of |
|------|-----|-------|------|------|---------|
| BLIP-2 | -0.18 | -0.19 | -0.06 | 0.14 | 0.13 |
| InstructBLIP | 0.06 | 0.04 | -0.03 | -0.01 | -0.04 |
| LLaVA-NeXT | **0.34** | **0.39** | **0.21** | **0.43** | **0.52** |
| LLaVA-OneVision | 0.30 | 0.40 | 0.22 | 0.52 | - |
| Molmo | ~0 | ~0 | ~0 | ~0 | ~0 |

### 人类判断：计数的影响 (LMM交互效应 $\beta$)

| 量词 | 计数交互$\beta$ | 面积交互$\beta$ | 大小规范交互$\beta$ |
|------|----------------|----------------|-------------------|
| few | -0.37 | -0.07 | -0.13 |
| a few | -0.38 | -0.10 | -0.11 |
| some | -0.20 | -0.05 | -0.07 |
| many | +0.38 | +0.08 | +0.14 |
| a lot of | +0.42 | +0.06 | +0.17 |

### 关键发现
- LLaVA系列模型在生成概率上与人类判断最为一致（few/a few随数量增加概率下降，many/a lot of概率上升），而InstructBLIP和Molmo完全无法区分不同量词
- 三种评估范式给出的结论不一致：概率提取和多选题方法显示较好的人类对齐，但数值评分提示方法的对齐程度差得多。这暗示VLM的"产出"和"判断"能力可能由不同机制驱动
- 物体计数是影响人类量词判断的最强因素（$|\beta|$ 0.20-0.42），分割面积（0.05-0.10）和真实世界大小（0.07-0.17）影响较弱
- 人类判断中few和a few行为几乎一致，但many和a lot of也高度相似——量词的"模糊"边界似乎比语言学理论预测的更为一致
- LMM模型解释了50.3%的方差，参与者间差异（0.042）远大于物体类别间差异（0.002），说明量词使用差异主要是个体差异而非物体特异性

## 亮点与洞察
- 三种评估范式（概率/评分/多选）的不一致是本文最有价值的发现——它警示VLM评估中方法选择会系统性地影响结论，单一范式可能误导对模型能力的判断
- 使用滑动条连续评分而非离散选择来收集人类判断是很好的设计，保留了模糊量词的连续性本质
- 心理语言学理论（近似数系统、subitizing阈值）与计算模型的结合为VLM评估提供了更深层的认知科学基础

## 局限与展望
- 仅测试了5个VLM，且均为开源模型，未包含GPT-4V等闭源模型
- 图片中物体种类限制在FSC-133的子集，未覆盖复杂场景（如多种物体共存）
- 仅测试英语量词，不同语言的模糊量词体系可能差异很大
- 未探索VLM在什么条件下会出现"反人类"判断（如把10个物体称为"few"），这类失败模式分析有助于理解模型缺陷
- 人类评分收集仅通过Prolific平台（英国/爱尔兰），文化和语言背景可能影响量词使用
- 大型闭源VLM（GPT-4V、Gemini）未纳入评估，可能遗漏了更强模型的行为模式

## 相关工作与启发
- **vs Testoni et al. (2024)**: 他们用合成图片测试3个VLM的量词选择，本文使用自然图片、5个VLM和三种评估范式，更加全面
- **vs Enyan et al. (2024)**: 他们比较LLM在精确vs模糊量词上的表现（纯文本），本文扩展到视觉场景中的grounded理解
- **vs Sorodoc et al. (2016/2018)**: 他们用固定比例定义量词进行训练，本文保留量词的模糊性并用人类连续评分建立ground truth
- 该数据集可进一步用于训练VLM的量词使用能力，或作为探测VLM计数能力的诊断工具

## 评分
- 总体评价: 跨学科融合研究的典范，对VLM评估方法论有重要警示
- 新颖性: ⭐⭐⭐⭐ 首个全面评估VLM模糊量词grounding能力的工作
- 实验充分度: ⭐⭐⭐⭐ 三种评估范式+统计建模很充分，但模型覆盖略窄
- 写作质量: ⭐⭐⭐⭐⭐ 跨学科融合（语言学+心理学+NLP）流畅自然
- 价值: ⭐⭐⭐⭐ 揭示了VLM评估范式依赖性的重要问题

<!-- VAQUUM: 20,300 human ratings, 1,089 images, 5 quantifiers, 5 VLMs, 3 evaluation paradigms -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] LAQuer: Localized Attribution Queries in Content-grounded Generation](laquer_localized_attribution.md)
- [\[ACL 2025\] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction](visual_cues_enhance_predictive_turn-taking_for_two-party_human_interaction.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ACL 2025\] AIDE: Attribute-Guided Multi-Hop Data Expansion for Data Scarcity in Task-Specific Fine-tuning](aide_attribute-guided_multi-hop_data_expansion_for_data_scarcity_in_task-specifi.md)
- [\[ACL 2025\] What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma](what_is_stigma_attributed_to_a_theory-grounded_expert-annotated_interview_corpus.md)

</div>

<!-- RELATED:END -->
