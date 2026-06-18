---
title: >-
  [论文解读] RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video
description: >-
  [NeurIPS 2025][多模态VLM][实时视频理解] 提出 RTV-Bench，一个面向多模态大模型（MLLM）实时视频连续分析能力的细粒度评测基准，包含552个视频和4608个QA对，通过多时间戳问答、层次化问题结构和多维度评估来全面测试模型在动态视频流中的感知、理解和推理能力。 领域现状：多模态大模型在感知、理解…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "实时视频理解"
  - "多模态大模型评测"
  - "连续分析"
  - "多时间戳问答"
  - "视频基准"
---

# RTV-Bench: Benchmarking MLLM Continuous Perception, Understanding and Reasoning through Real-Time Video

**会议**: NeurIPS 2025  
**arXiv**: [2505.02064](https://arxiv.org/abs/2505.02064)  
**代码**: [https://ljungang.github.io/RTV-Bench](https://ljungang.github.io/RTV-Bench)  
**领域**: 多模态VLM  
**关键词**: 实时视频理解, 多模态大模型评测, 连续分析, 多时间戳问答, 视频基准

## 一句话总结
提出 RTV-Bench，一个面向多模态大模型（MLLM）实时视频连续分析能力的细粒度评测基准，包含552个视频和4608个QA对，通过多时间戳问答、层次化问题结构和多维度评估来全面测试模型在动态视频流中的感知、理解和推理能力。

## 研究背景与动机

**领域现状**：多模态大模型在感知、理解和推理方面取得快速进展，但现有基准主要评估静态或离线视频理解能力，难以衡量模型在连续动态视频流中的表现。

**现有痛点**：VStream、StreamingBench、OVOBench 等基准虽然在视频长度和评测类型上做了改进，但对实时响应能力的评估不够充分——它们忽略了模型捕捉视觉输入中转换和瞬间细节的能力。

**核心矛盾**：现实世界的视频是连续变化的，同一个问题在不同时间点可能有不同的正确答案，而现有基准通常只在单一时间点提出问题，无法测试模型对动态状态变化的敏感度。

**本文目标**：设计一个能够全面评测 MLLM 在实时视频场景中连续分析能力的基准，涵盖感知、理解和推理三个层次。

**切入角度**：通过三个核心创新——多时间戳QA机制、层次化问题结构和多维度评估——来构建更严格的实时视频理解评测。

**核心 idea**：同一个概念性问题在视频不同时间点重复提问，正确答案随场景演变而改变，从而直接测试模型的连续时间追踪和状态更新能力。

## 方法详解

### 整体框架
RTV-Bench 由552个多样化视频（总时长167.2小时，平均18.2分钟/视频）和4608个精心标注的QA对组成。视频主要来自智能驾驶、体育赛事和第一人称视频三大领域，涵盖16个子类别。

### 关键设计

1. **多时间戳QA机制（MTQA）**:

    - 功能：评估模型对视频动态变化的实时追踪能力
    - 核心思路：对同一概念性问题在视频不同时间点重复提问。例如"守门员在做什么？"，随着比赛进展，正确答案会从"扑球"变为"站立"再变为"开球"。标注者为每个答案选项确定最早有效时间戳
    - 设计动机：不同于 OVOBench 在不同时间戳提出不同问题，MTQA 在不同时间复用同一问题，更严格地测试模型的连续分析能力

2. **层次化问题结构**:

    - 功能：确保模型具备可靠的顺序推理能力
    - 核心思路：每组问题包含约3个选择题，前两个是基础感知/理解题，第三个是需要综合上下文的高级推理题。高级问题逻辑上依赖于对基础问题的正确回答
    - 设计动机：防止模型通过认知捷径得到正确答案，确保高级推理建立在扎实的基础理解之上

3. **多维度评估体系**:

    - 功能：提供细粒度的模型能力诊断
    - 核心思路：将评估分为8个维度——时间感知(TP)、场景感知(SP)、视觉感知(VP)、未来预测(FP)、现象理解(PU)、意图分析(IA)、全局理解(GU)和时空推理(SR)。引入条件Score指标：仅当基础问题全部正确时才计高级问题得分
    - 设计动机：超越单一聚合分数，对模型能力和局限提供更有信息量的视角

### 损失函数 / 训练策略
这是一个评测基准，不涉及训练。标注流程采用 DeepSeek 生成初始问题模板，再由人工标注者精心修改以反映动态场景需求，确保高质量标注。

## 实验关键数据

### 主实验

| 模型 | 类型 | Overall Acc(%) | Score | MTQA Acc(%) |
|------|------|---------------|-------|------------|
| GPT-4o | 闭源 | 50.02 | 22.10 | 44.73 |
| IXC2.5-OL | 在线7B | 47.33 | 15.40 | 38.21 |
| VITA-1.5 | 在线7B | 44.51 | 11.80 | 36.32 |
| Qwen2.5-VL | 离线7B | 40.41 | 7.13 | 37.46 |
| VideoLLaMA2 | 离线7B | 39.55 | 7.90 | 34.95 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 在线 vs 离线模型 | +7.78% Acc | 在线模型显著优于离线模型 |
| 增加帧数 | 非单调提升 | 更多帧并不总能提升表现 |
| 模型规模 | 正相关 | 更大模型通常表现更好 |

### 关键发现
- 大部分模型准确率低于50%，实时视频理解仍是巨大挑战
- 在线模型（如IXC2.5-OL）在MTQA任务上显著优于离线模型，但仍远落后于GPT-4o
- 增加采样帧密度的收益非单调，说明简单增加帧数不能解决问题，需要专门设计的流式处理架构

## 亮点与洞察
- MTQA设计极具新意：同一问题+不同时间=不同答案，这比传统的"不同时间问不同问题"更能测试连续分析能力。这一设计思路可迁移到其他动态场景评测
- 条件Score指标巧妙地防止了"蒙对高级题但基础题错误"的虚假成功，提高了评估的可靠性

## 局限与展望
- 视频来源偏向驾驶、体育和第一人称三个领域，多样性有限
- 评估仅基于选择题，未考虑开放式问答
- 未来可扩展到更多场景类型，并考虑模型的延迟和实时响应速度评测

## 相关工作与启发
- **vs StreamingBench**: StreamingBench 虽评估实时场景，但问题设计较简单，缺乏多时间戳复用问题的机制
- **vs OVOBench**: OVOBench 在不同时间戳问不同问题，而 RTV-Bench 用同一问题测试动态追踪，评估更严格

## 评分
- 新颖性: ⭐⭐⭐⭐ 多时间戳QA和层次化评估设计有新意
- 实验充分度: ⭐⭐⭐⭐ 覆盖了多种在线/离线/闭源模型
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，动机阐述充分
- 价值: ⭐⭐⭐⭐ 填补了实时视频理解评测的重要空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] In the Eye of MLLM: Benchmarking Egocentric Video Intent Understanding with Gaze-Guided Prompting](in_the_eye_of_mllm_benchmarking_egocentric_video_intent_understanding_with_gaze-.md)
- [\[NeurIPS 2025\] GEM: Empowering MLLM for Grounded ECG Understanding with Time Series and Images](gem_empowering_mllm_for_grounded_ecg_understanding_with_time_series_and_images.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] DynamicVL: Benchmarking MLLMs for Dynamic City Understanding](dynamicvl_benchmarking_multimodal_large_language_models_for_dynamic_city_underst.md)
- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)

</div>

<!-- RELATED:END -->
