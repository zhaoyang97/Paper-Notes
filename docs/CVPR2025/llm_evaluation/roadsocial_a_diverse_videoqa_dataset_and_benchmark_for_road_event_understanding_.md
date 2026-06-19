---
title: >-
  [论文解读] RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives
description: >-
  [CVPR 2025][LLM评测][VideoQA] 本文提出RoadSocial，一个来源于社交媒体的大规模多样化VideoQA数据集（13.2K视频、260K问答对），覆盖全球多地域多视角的道路事件场景，通过半自动标注框架和12类QA任务系统性评测了18种Video LLM的道路事件理解能力。
tags:
  - "CVPR 2025"
  - "LLM评测"
  - "VideoQA"
  - "道路事件"
  - "社交媒体"
  - "数据集"
  - "Video LLM"
---

# RoadSocial: A Diverse VideoQA Dataset and Benchmark for Road Event Understanding from Social Video Narratives

**会议**: CVPR 2025  
**代码**: 无  
**领域**: LLM评测  
**关键词**: VideoQA, 道路事件, 社交媒体, 数据集, Video LLM

## 一句话总结

本文提出RoadSocial，一个来源于社交媒体的大规模多样化VideoQA数据集（13.2K视频、260K问答对），覆盖全球多地域多视角的道路事件场景，通过半自动标注框架和12类QA任务系统性评测了18种Video LLM的道路事件理解能力。

## 研究背景与动机

### 领域现状

**领域现状**：视频问答（VideoQA）是评估视频语言模型理解能力的重要任务。在自动驾驶和交通安全领域，准确理解道路事件（事故、违规行为、天气影响等）对安全系统至关重要。

**现有痛点**：(1) 地域和视角偏差——现有道路事件数据集（如BDD100K、Drive-LM）主要来自特定地区的车载摄像头，未能反映全球道路条件的差异性和多样性。(2) 标注偏差——现有数据集依赖专家标注，覆盖的事件类型有限、标注成本高。(3) 缺乏社交语境——社交媒体上的道路事件视频包含丰富的评论、叙事信息，但现有数据集未利用这些上下文信息。(4) QA任务类型有限——大多数数据集只考虑简单的事件描述或因果推理，缺乏空间推理、时间定位、社交影响分析等深度理解任务。

**核心矛盾**：道路事件的复杂性和多样性要求大规模、高多样性的数据来训练和评估模型，但人工标注成本极高，且现有数据源（车载摄像头）的视角单一。

**本文目标** 如何构建一个覆盖全球、多视角、包含丰富社交语境的大规模道路事件VideoQA数据集？

**切入角度**：从社交媒体平台（YouTube等）收集道路事件视频，利用Text LLM和Video LLM构建半自动标注流水线，大规模生成高质量QA对。

**核心 idea**：社交媒体视频+LLM半自动标注，构建全球最大最多样的道路事件VideoQA数据集。

## 方法详解

### 整体框架

RoadSocial的构建流程：(1) 数据收集——从社交媒体平台按关键词（accident、road rage、traffic violation等）搜索和过滤道路事件视频，确保地域和视角多样性。(2) 元数据提取——收集视频标题、描述、评论等社交媒体元数据。(3) 半自动QA生成——利用Video LLM理解视频内容，Text LLM结合社交元数据生成覆盖12类任务的QA对。(4) 质量控制——人工审核和过滤，确保QA对的准确性和多样性。

### 关键设计

1. **半自动标注框架（LLM-Powered Annotation）**：
    - 功能：低成本高效率地生成大规模高质量QA对
    - 核心思路：分两阶段标注。第一阶段：Video LLM（如GPT-4V、VideoChat等）观看视频并生成事件描述、时间标注和要素识别。第二阶段：Text LLM（如GPT-4）基于视频描述和社交媒体评论生成结构化QA对，覆盖事件识别、因果推理、空间关系、时间定位等12类任务。最后人工审核过滤低质量样本
    - 设计动机：纯人工标注260K QA对成本不可接受，纯自动生成质量不可控。两阶段LLM+人工审核实现了规模和质量的平衡

2. **12类挑战性QA任务设计**：
    - 功能：全面评估Video LLM对道路事件的理解深度
    - 核心思路：设计12个不同难度和类型的QA任务：事件类型识别、事件描述、因果推理、参与者识别、空间关系、时间定位、交通规则判断、天气影响分析、情感分析（从评论中）、严重程度评估、风险预测、社交影响分析。每类任务有对应的评估指标
    - 设计动机：现有benchmark只考虑2-3类简单QA任务，无法全面评估模型的道路场景理解能力

3. **多样性保障机制**：
    - 功能：确保数据集的地域、视角和事件类型多样性
    - 核心思路：(1) 地域多样性——覆盖亚洲、欧洲、北美、南美等全球多个地区，不同交通规则和道路条件。(2) 视角多样性——包含CCTV固定监控、手持拍摄、行车记录仪、无人机航拍等多种视角。(3) 事件类型平衡——确保事故、违规、恶劣天气、道路冲突等各类事件有足够样本
    - 设计动机：单一来源数据集的模型在其他场景泛化差

## 实验关键数据

### 关键发现

- 数据集规模：13.2K视频、14M帧、414K社交评论、674个标签、260K QA对
- 18种Video LLM（包括GPT-4V、VideoChat2、Video-LLaVA等）的全面评测
- 所有模型在因果推理和时间定位任务上表现最差，显示这些能力仍是瓶颈
- 社交媒体元数据的加入可提升事件理解准确率约5-10%
- 闭源模型（GPT-4V）仍显著领先开源模型，差距约15-20%
- 多视角样本上的评估表明：模型对无人机视角表现最差

## 亮点与洞察

- **数据源创新**：首次系统性地利用社交媒体作为道路事件数据源，自然包含多样性和社交语境
- **评测全面**：12类QA任务的设计覆盖了道路事件理解的各个维度
- **标注流水线可复制**：半自动框架可被其他领域的VideoQA数据集构建所借鉴

## 局限与展望

- 社交媒体视频质量参差不齐（画质、稳定性），可能影响模型评估公平性
- LLM自动标注仍可能引入系统性偏差
- 隐私问题——社交媒体视频需要脱敏处理
- 未来可扩展到多语言标注和更多社交媒体平台

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] StreamMind: Unlocking Full Frame Rate Streaming Video Dialogue through Event-Gated Cognition](../../ICCV2025/llm_evaluation/streammind_unlocking_full_frame_rate_streaming_video_dialogue_through_event-gate.md)
- [\[ACL 2025\] skLEP: A Slovak General Language Understanding Benchmark](../../ACL2025/llm_evaluation/sklep_a_slovak_general_language_understanding_benchmark.md)
- [\[ACL 2025\] BelarusianGLUE: Towards a Natural Language Understanding Benchmark for Belarusian](../../ACL2025/llm_evaluation/belarusian_glue.md)
- [\[ACL 2025\] CulturalBench: A Robust, Diverse, and Challenging Cultural Benchmark by Human-AI CulturalTeaming](../../ACL2025/llm_evaluation/culturalbench_a_robust_diverse_and_challenging_cultural_benchmark_by_human-ai_cu.md)
- [\[ACL 2025\] NorEval: A Norwegian Language Understanding and Generation Evaluation Benchmark](../../ACL2025/llm_evaluation/noreval_a_norwegian_language_understanding_and_generation_evaluation_benchmark.md)

</div>

<!-- RELATED:END -->
