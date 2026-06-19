---
title: >-
  [论文解读] Minerva: Evaluating Complex Video Reasoning
description: >-
  [ICCV 2025][可解释性][视频推理] 提出 Minerva——一个包含 1515 个手工标注的复杂视频推理问答数据集，每题配有 5 个选项和详细推理链（reasoning trace），用于评估多模态大模型的视频推理能力，并建立了视频推理错误分类体系（Temporal/Perceptual/Logical/Completeness）和 MiRA 自动评估框架。
tags:
  - "ICCV 2025"
  - "可解释性"
  - "视频推理"
  - "推理链评估"
  - "视频问答"
  - "benchmark"
  - "推理错误分类"
---

# Minerva: Evaluating Complex Video Reasoning

**会议**: ICCV 2025  
**arXiv**: [2505.00681](https://arxiv.org/abs/2505.00681)  
**代码**: [GitHub](https://github.com/google-deepmind/neptune?tab=readme-ov-file#minerva)  
**领域**: 可解释性  
**关键词**: 视频推理, 推理链评估, 视频问答, benchmark, 推理错误分类

## 一句话总结

提出 Minerva——一个包含 1515 个手工标注的复杂视频推理问答数据集，每题配有 5 个选项和详细推理链（reasoning trace），用于评估多模态大模型的视频推理能力，并建立了视频推理错误分类体系（Temporal/Perceptual/Logical/Completeness）和 MiRA 自动评估框架。

## 研究背景与动机

现有视频基准测试的核心问题：**只看最终答案，不看推理过程**。

**正确答案不等于正确推理**：模型可能因为语言偏差、排除法或纯粹运气答对，而非真正理解视频内容

**错误答案不等于完全失败**：模型可能距离正确答案只差一步，但最终答案错了就被判为完全失败

**视频推理的特殊性**：不同于文本推理，视频推理需要时序定位、视觉感知（识别物体/动作/事件）和逻辑推理的多步骤协作，每步可能用到不同技能和模态

**现有数据集缺陷**：大多依赖半自动 LLM 标注流水线，不提供中间推理步骤；少数提供辅助信息的数据集要么视频太短（VideoCoT 用 Kinetics700），要么推理链质量低（自动生成含大量无关信息）

核心需求：需要一个**全手工标注、高质量推理链、跨领域长视频**的基准，不仅评估最终答案对错，还能诊断模型在推理链上哪一步出了问题。

## 方法详解

### 整体框架

Minerva 的构建和评估包含三个层次：
1. **数据集构建**：视频选择→人工标注→质量审核→对抗过滤
2. **MCQ 评估**：多模态模型在 5 选 1 上的准确率
3. **推理链评估**：用 Minerva Rubric 对推理过程打分（人工+LLM-as-Judge）

### 关键设计

1. **视频选择策略**——四大领域确保复杂性：

    - **短片（Short Films）**：复杂故事线、关系弧线、事件弧线，避免主流电影以减少训练数据污染
    - **体育与棋盘游戏**：需要基于规则推理、细粒度动作识别、棋子/球员位置判断
    - **教育/STEM 讲座**：数学/科学推理，但仅占 8%（因为语音常主导）
    - **生活方式**：烹饪、旅行视频，事件因果链、自中心视角、空间推理

2. **标注设计**——确保复杂多步推理：

    - 每题要求至少使用 2 种以上技能：时序推理、计数、因果关系、目标推理、情境感知、事件检测、状态变化、OCR、语音理解、空间感知、数值推理、物体识别、反事实推理
    - 推理链（reasoning trace）详细、自由格式，包含时间戳（99.6% 包含，平均 4 个/题）和关键动作描述
    - 平均推理链长度 92 词，视频长度 2 分钟到 100+ 分钟（均值 12 分钟）

3. **对抗过滤——去除文本偏差**：

    - 用 Deepseek、GPT-4o、Gemini-flash、Qwen2.5-VL 做纯文本（QAD-only 和 ASR-only）测试
    - 取多模型共识，过滤仅凭文本就能答对的题目，避免误删仅因运气答对的难题

4. **推理错误分类体系（Minerva Rubric）**：

    - **Perceptual Correctness**：视觉感知错误（物体/动作/事件识别、OCR、ASR 解析）
    - **Temporal Localization**：指向视频中错误的时间段
    - **Logical Reasoning**：推理逻辑错误（含算术/数值推理）
    - **Completeness**：推理链缺少关键步骤

5. **MiRA (Minerva Reasoning Assessment)**：

    - 基于 Minerva Rubric 的 LLM-as-Judge 自动评估
    - 支持 Reference-based（给 ground truth 推理链参考）和 Reference-free 两种模式
    - 使用 3 分 Likert 量表打分

### 损失函数 / 训练策略

本文是评估数据集，不涉及模型训练。Prompt 设计包含三种策略：直接回答、逐步推理、提供 Minerva Rubric 辅助推理。

## 实验关键数据

### 主实验（MCQ 准确率）

| 模型 | 帧数 | ASR | MCQ-Acc% |
|------|------|-----|----------|
| Random | - | - | 20.0 |
| Qwen2.5-VL (开源) | 768 | ✓ | 35.05 |
| VideoLLaMA3 (开源) | 180 | ✓ | 35.91 |
| InternVideo2.5 (开源) | 256 | ✓ | 35.18 |
| Claude 3.5 Sonnet v2 | 64 | ✓ | 31.28 |
| GPT-4o | 250 | ✓ | 45.54 |
| GPT-4.1 | 256 | ✓ | 53.99 |
| Gemini 2.0 Flash | 256 | ✓ | 53.47 |
| Gemini 2.5 Pro Thinking | 1024 | ✓ | **66.2** |
| **人类** | all | ✓ | **92.54** |

### 推理链评估（MiRA + Human）

| 评估方式 | Temporal | Perceptual | Logical | Completeness |
|---------|----------|-----------|---------|-------------|
| Human | 0.440 | 0.625 | 0.770 | 0.725 |
| RF-MiRA (Pearson r) | 0.711 (0.56) | 0.684 (0.45) | 0.920 (0.21) | 0.871 (0.07) |
| RB-MiRA (Pearson r) | 0.434 (0.79) | 0.484 (0.59) | 0.848 (0.17) | 0.748 (0.24) |

### Prompt 消融

| Prompt 方式 | MCQ Accuracy | MiRA |
|------------|-------------|------|
| 直接回答 | 46.47 | - |
| + 逐步推理 | 51.22 | 0.65 |
| + Minerva Rubric | **53.47** | **0.75** |

### 关键发现

- **人机差距巨大**：最强模型（Gemini 2.5 Pro Thinking）66.2% vs 人类 92.5%，差距近 30%
- **开源与闭源差距缩小**：Qwen2.5-VL 和 InternVideo2.5 已超越 Claude Sonnet
- **时序定位是最大瓶颈**：在推理链评估中 Temporal 得分最低（Human 评分 0.440），远低于 Logical（0.770）
- **正确答案 ≠ 正确推理**：Table 6 展示了模型答对但推理链严重错误的案例（如捏造视频中不存在的内容来"推理"出正确答案）
- **提供 Rubric 可提升表现**：仅在 prompt 中告诉模型评估标准（无额外计算），就能将 MCQ 准确率从 51.22% 提升到 53.47%
- **Reference-based MiRA 在 Temporal 维度上与人类相关性最高（r=0.79）**，说明给参考推理链后 LLM 判分更可靠
- **Thinking 模式有帮助**：Gemini 2.5 Pro 开启 thinking 后 1024 帧下从 63.9% 提升到 66.2%

## 亮点与洞察

- **推理链标注的路线选择正确**：全人工标注 + 自由格式文本（非结构化模板），兼顾了质量和表达灵活性
- **错误分类体系（Rubric）有双重价值**：既是评估工具，又能作为 prompt 提升模型表现
- **揭示了视频理解的真正瓶颈**：不是逻辑推理能力不足，而是时序定位和视觉感知——这是"视频"相比"文本"推理的独特挑战
- **Table 6 的案例极具说服力**：模型能用虚构的推理过程得到正确答案，说明仅靠 MCQ 准确率评估视频理解是不够的

## 局限与展望

- 数据规模有限（1515 题），虽然质量高但可能不够涵盖所有视频理解场景
- 推理链评估的 LLM-as-Judge 在 Logical 和 Completeness 维度上与人类相关性较低（r<0.25），自动化评估仍需改进
- 未提供训练集——不能直接用于微调模型的推理能力
- 视频来源以 YouTube 为主，某些专业领域（如医学、工业检测）未覆盖

## 相关工作与启发

- 推理链标注的思路可以推广到其他多模态任务（如文档理解、3D 场景推理）
- Minerva Rubric 的四维错误分类体系可以作为视频 AI 系统的通用诊断框架
- "提供评估标准到 prompt 中可提升模型表现"是一个值得深入研究的现象

## 评分

- 新颖性：⭐⭐⭐⭐⭐ （首个带推理链的视频理解 benchmark，错误分类体系原创）
- 技术深度：⭐⭐⭐⭐ （标注设计严谨，对抗过滤+多级评估体系完备）
- 实验充分度：⭐⭐⭐⭐⭐ （10+ 模型、人类基线、帧数/ASR/prompt/thinking 多维消融）
- 实用价值：⭐⭐⭐⭐⭐ （直接可用于诊断视频模型的推理瓶颈）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](../../ACL2026/interpretability/meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)
- [\[CVPR 2026\] Back to the Feature: Explaining Video Classifiers with Video Counterfactual Explanations](../../CVPR2026/interpretability/back_to_the_feature_explaining_video_classifiers_with_video_counterfactual_expla.md)
- [\[NeurIPS 2025\] Evaluating LLMs in Open-Source Games](../../NeurIPS2025/interpretability/evaluating_llms_in_open-source_games.md)
- [\[ICML 2025\] Evaluating Neuron Explanations: A Unified Framework with Sanity Checks](../../ICML2025/interpretability/evaluating_neuron_explanations_a_unified_framework_with_sanity_checks.md)
- [\[CVPR 2025\] KVQ: Boosting Video Quality Assessment via Saliency-Guided Local Perception](../../CVPR2025/interpretability/kvq_boosting_video_quality_assessment_via_saliency-guided_local_perception.md)

</div>

<!-- RELATED:END -->
