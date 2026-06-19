---
title: >-
  [论文解读] Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly
description: >-
  [CVPR 2025][多模态VLM][MLLM评估] 揭示MLLM"**理解了视觉内容但仍给出错误回答**"的普遍现象，构建包含12类正负样本对的**MMVU基准**，发现根因在于训练数据正样本偏倚和视觉token注意力不足，提出**MMVU-Train数据集**（112K正负样本对）+ **内容引导精炼（CGR）**+ **视觉注意力精炼（VAR）**三管齐下的解决方案。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "MLLM评估"
  - "视觉理解"
  - "误导性问题"
  - "正负样本对"
  - "注意力分析"
  - "鲁棒性"
---

# Unveiling the Ignorance of MLLMs: Seeing Clearly, Answering Incorrectly

**会议**: CVPR 2025  
**arXiv**: [2406.10638](https://arxiv.org/abs/2406.10638)  
**代码**: [https://github.com/BAAI-DCAI/MMVU](https://github.com/BAAI-DCAI/MMVU)  
**领域**: 多模态VLM  
**关键词**: MLLM评估, 视觉理解, 误导性问题, 正负样本对, 注意力分析, 鲁棒性

## 一句话总结
揭示MLLM"**理解了视觉内容但仍给出错误回答**"的普遍现象，构建包含12类正负样本对的**MMVU基准**，发现根因在于训练数据正样本偏倚和视觉token注意力不足，提出**MMVU-Train数据集**（112K正负样本对）+ **内容引导精炼（CGR）**+ **视觉注意力精炼（VAR）**三管齐下的解决方案。

## 研究背景与动机
MLLM在视觉理解任务上表现出色，但一个被忽视的现象是：模型能**正确回答直接相关的正面问题**（证明它理解了图像），却在**间接或误导性的负面问题**上犯错。例如，模型能识别出图中是"一只黑色的猫"（正面问题），但被问"这只猫是什么颜色的红色部分？"时却信以为真回答了错误选项。

**现有痛点**：
1. 现有幻觉基准（POPE、MADBench等）不区分"不理解导致的错误"和"理解了但仍犯错"
2. 指令微调数据集以正面、直接的视觉问答为主，缺乏负面/误导性样本，导致模型对正面回答有系统性偏倚
3. 生成token时，模型对视觉token的attention远低于对系统提示和问题token的attention

**核心矛盾**：模型具备视觉理解能力（正面问题答对了），但缺乏利用理解来抵抗误导的能力。问题不在于"看不到"，而在于"知识应用不当"。

**切入角度**：构建成对的正/负问题来精确量化这种"理解但犯错"的程度，分析注意力分布找出根因，然后从数据和推理两个角度提出解决方案。

## 方法详解

### 整体框架
工作包含三部分：(1) MMVU基准测试集（893对正负问题，12类，手动标注）+ 新评估指标；(2) MMVU-Train训练集（112K对正负样本，自动构建管线）；(3) 推理阶段的CGR和VAR策略。三个组件互补——训练数据从源头减少偏倚，推理策略在部署时增强视觉聚焦。

### 关键设计

1. **MMVU基准设计与评估指标**
    - 功能：精确评估MLLM"理解但犯错"的程度
    - 核心思路：每张图像配对两个选择题——**正面问题**直接考察视觉理解（"图中的花是什么颜色？"），**负面问题**引入误导信息测试鲁棒性（"图中的花有几片红色花瓣？"——实际花是蓝色的）。涵盖3个层次12个类别：
        - 字符层：字符/数字识别
        - 属性层：颜色/纹理、数量、形状、姿态、位置
        - 上下文层：抽象知识、具体知识、专业知识、行为、关系
    - 评估指标：
        - **RA（Response Accuracy）**：综合正负问题的准确率，↑越高越好
        - **MR（Misresponse Rate）**：在正面问题答对的前提下，负面问题答错的比例，↓越低越好。MR直接量化了"理解但犯错"的程度
    - 设计动机：现有基准将"答错"等同于"不理解"，而MMVU通过正负配对区分了"真的不理解"和"理解了但被误导"

2. **正负样本配对数据构建管线（MMVU-Train）**
    - 功能：构建112K对正负训练样本，从数据源头减少MLLM对正面回答的偏倚
    - 核心思路：从图像中自动提取视觉信息（文字、数字、物体、属性、关系、局部/全局上下文），然后基于这些信息构建正面问题（直接考察可观察内容）和负面问题（引入假设性修改、无关描述或错误前提），每个问题4个选项（含解释性内容的干扰项）
    - 设计动机：手动标注成本过高（MMVU测试集仅893对），自动管线可规模化生成。关键：负面问题不是随机噪声，而是基于图像真实内容精心构造的反面——这样模型学到的是"基于视觉理解进行判断"而非简单的"拒绝所有负面问题"

3. **内容引导精炼（CGR）+ 视觉注意力精炼（VAR）**
    - 功能：推理阶段增强模型对视觉内容的关注和利用
    - **CGR（Content Guided Refinement）**：
        - 两步推理：先让模型对图像进行详细内容分析（提取结构化信息），再基于分析结果和原始图像回答问题
        - 动机：强制模型在回答前"先看后想"，将视觉理解显式化，减少语言偏倚对回答的影响
    - **VAR（Visual Attention Refinement）**：
        - 提取问题token与视觉token之间的attention分数，对高分区域增强、低分区域遮蔽，生成聚焦mask引导模型关注与问题相关的视觉区域
        - 动机：分析发现MLLM对视觉token的attention远低于系统/问题token（Fig. 4a），且负面问题时视觉attention更低（Fig. 4b）——VAR通过显式增强视觉注意力来修复这一失衡

## 实验关键数据

### 15个MLLM在MMVU上的表现（Tab. 1）

| 模型 | Avg. RA↑ | Avg. MR↓ |
|------|---------|---------|
| GPT-4o | 65.06 | 19.53 |
| Ovis1.6-Gemma2-9B | 66.74 | 19.78 |
| Llama3.2-90B | 66.74 | 18.47 |
| LLaVA-OneVision-7B | 60.58 | 27.77 |
| InternVL2-8B | 49.72 | 30.63 |
| VILA1.5-13B | 25.76 | 62.30 |
- 即使是最强的GPT-4o也有19.53%的误导率——每5次理解了内容的情况中仍有1次答错
- VILA1.5-13B的MR高达62.30%——超过一半的"理解了但答错"

### MMVU-Train微调效果（Tab. 2）

| 模型 | 基础RA | +MMVU-Train RA | 提升 |
|------|--------|---------------|------|
| InternVL2-8B | 49.72 | 58.01 | **+8.29** |
| VILA1.5-13B | 25.76 | 37.74 | **+11.98** |
| LLaVA-OneVision-7B | 60.58 | 63.49 | +2.91 |

| 模型 | 基础MR | +MMVU-Train MR | 降低 |
|------|--------|---------------|------|
| LLaVA-OneVision-7B | 27.77 | 21.03 | **-6.74** |
| InternVL2-8B | 30.63 | 25.04 | **-5.59** |

### 注意力分析关键发现（Fig. 4）
- 答案token对视觉token的attention仅占~10%，远低于对系统token（~55%）和问题token（~35%）
- 负面问题时，问题-视觉attention比正面问题时更低（约低15-25%）
- 负面问题时模型输出概率更低（置信度下降），但仍给出错误答案——说明模型"犹豫但仍偏向错误"

### CGR + VAR策略效果
- CGR平均提升RA 2-4个百分点，CGR+VAR组合效果更好
- 在通用基准（MMBench、SEED等）上性能不下降，说明增强鲁棒性不损害通用能力

## 亮点与洞察
- **独特视角**：区分"不理解导致的错误"和"理解了但仍犯错"，揭示了MLLM一个被忽视的系统性缺陷
- **因果分析**：通过attention map和logit分布分析，精确定位了两个根因（数据偏倚 + 视觉注意力不足），而非仅凭经验猜测
- **数据+推理双管齐下**：MMVU-Train从训练数据源头减少偏倚，CGR+VAR在推理时增强视觉利用——两条路径互补
- **正负配对设计精巧**：负面问题不是随机生成，而是基于正面问题的视觉内容精心构造的"反面"——这确保了错误确实来自"理解但犯错"而非"不理解"
- **通知性发现**：即使是GPT-4o也有~20%的MR——说明这不是小模型特有的问题，而是当前MLLM训练范式的系统性缺陷

## 局限性
- MMVU测试集规模较小（893对），12个类别中某些类别样本偏少（如关系类仅71对）
- 仅使用选择题格式，未验证在开放式问答场景下的表现
- VAR策略需要访问中间attention map，对闭源API模型不适用
- 自动构建的MMVU-Train可能存在噪声样本，未做人工质量审核
- CGR的两步推理增加了推理延迟（约翻倍）

## 相关工作与启发
- POPE、MADBench → 关注物体幻觉和对抗误导，但不区分"不理解"和"理解但犯错"
- NaturalBench → 关注一致性评估，但未分析注意力根因
- VCD（Visual Contrastive Decoding）→ 修改解码策略减少幻觉，与VAR的思路互补
- 启发：MLLM的"理解"和"应用理解"是不同的能力维度——即使模型"看到"了正确信息，如果注意力分配不合理或训练数据有偏倚，仍然会给出错误答案。这一发现对MLLM的指令微调数据构建有深远启示。

## 评分
⭐⭐⭐⭐ — 发现了一个重要且被忽视的MLLM缺陷，分析方法（正负配对+attention分析）客观有力。MMVU基准和MMVU-Train数据集对社区有实际贡献。解决方案（数据+推理策略）虽然不算颠覆性，但实用有效。整体工作完整严谨。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)
- [\[CVPR 2025\] Seeing the Abstract: Translating the Abstract Language for Vision Language Models](seeing_the_abstract_translating_the_abstract_language_for_vision_language_models.md)
- [\[ACL 2025\] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](../../ACL2025/multimodal_vlm/unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [\[CVPR 2025\] MARTEN: Visual Question Answering with Mask Generation for Multi-Modal Document Understanding](marten_visual_question_answering_with_mask_generation_for_multi-modal_document_u.md)
- [\[ACL 2025\] VLSBench: Unveiling Visual Leakage in Multimodal Safety](../../ACL2025/multimodal_vlm/vlsbench_unveiling_visual_leakage_in_multimodal_safety.md)

</div>

<!-- RELATED:END -->
