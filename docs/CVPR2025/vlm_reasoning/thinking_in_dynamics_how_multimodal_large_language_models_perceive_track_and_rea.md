---
title: >-
  [论文解读] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World
description: >-
  [CVPR 2025][多模态VLM][多模态大模型] 本文提出 Dyn-Bench——首个系统评估多模态大模型（MLLMs）在物理4D世界中动态感知、追踪和推理能力的大规模基准，包含 1K 视频、7K VQA 对和 3K 动态目标定位对，发现现有模型无法同时在时空推理和动态定位上表现良好，并提出 Mask-Guided Fusion 和 ST-TCM 两种结构化增强方法显著提升表现。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "多模态大模型"
  - "时空推理"
  - "4D世界理解"
  - "动态感知"
  - "基准评测"
---

# Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World

**会议**: CVPR 2025  
**arXiv**: [2603.12746](https://arxiv.org/abs/2603.12746)  
**代码**: [https://dyn-bench.github.io/](https://dyn-bench.github.io/)  
**领域**: 多模态VLM  
**关键词**: 多模态大模型, 时空推理, 4D世界理解, 动态感知, 基准评测

## 一句话总结
本文提出 Dyn-Bench——首个系统评估多模态大模型（MLLMs）在物理4D世界中动态感知、追踪和推理能力的大规模基准，包含 1K 视频、7K VQA 对和 3K 动态目标定位对，发现现有模型无法同时在时空推理和动态定位上表现良好，并提出 Mask-Guided Fusion 和 ST-TCM 两种结构化增强方法显著提升表现。

## 研究背景与动机

**领域现状**：多模态大模型（如 GPT-4V, Gemini, LLaVA 等）在静态图像理解上已经非常成功，能够进行复杂的视觉问答、空间推理和图像描述。人类生活在一个物理4D世界中（三维空间+时间维度），几何结构和语义内容随时间动态演化。

**现有痛点**：现有 MLLMs 究竟能否"在动态中思考"（Thinking in Dynamics）？即能否感知、追踪和推理演化场景中的时空动态？目前对此缺乏系统化的评估。现有视频理解基准要么关注高层语义（如动作识别），要么只评估简单的时序理解（如时序排序），缺少对**局部化动态感知**（哪个物体在动、怎么动、对其他物体有什么影响）和**时空推理**（理解运动轨迹、预测交互）能力的综合评估。

**核心矛盾**：现有基准无法揭示 MLLMs 在细粒度动态理解上的真实能力。高层任务（如 "视频里发生了什么"）可以通过关键帧采样和静态理解绕过，不需要真正的动态推理。

**本文目标**：(1) 构建一个大规模、高质量的动态理解基准，从多维度系统评估 MLLMs 的时空推理和动态定位能力；(2) 诊断现有模型的瓶颈并探索增强方法。

**切入角度**：作者将"在动态中思考"分解为三个递进能力：**感知（Perceive）**——识别场景中的动态元素；**追踪（Track）**——跟踪这些元素在时间维度上的运动和变化；**推理（Reason）**——基于动态信息进行因果推理和交互预测。评估同时覆盖语言输出（VQA）和视觉输出（定位）。

**核心 idea**：从大规模 2D 视频和 4D 数据源中通过多阶段过滤构建高质量动态场景集合 Dyn-Bench，系统评估通用、空间和区域级 MLLMs 的动态理解能力，并提出两种结构化增强方法。

## 方法详解

### 整体框架
Dyn-Bench 的构建和评估分为三个部分：(1) **数据构建**：从真实世界和合成视频数据集中通过多阶段过滤收集 1K 高质量动态场景视频，标注 7K VQA 对和 3K 动态目标定位对；(2) **多维度评估**：设计覆盖动态感知、运动追踪、时空推理的评估任务，分别考察语言表达（VQA 正确率）和视觉表达（定位精度）；(3) **增强方法**：提出 Mask-Guided Fusion 和 ST-TCM 两种方法来提升 MLLMs 的动态理解能力。

### 关键设计

1. **多阶段数据过滤与标注（Multi-stage Filtering）**:

    - 功能：从海量视频数据中筛选出包含显著动态变化的高质量场景
    - 核心思路：数据来源包括真实世界视频数据集（如 TAO, LaSOT, Waymo 等）和 4D 合成数据集（如 Kubric, Replica 等）。过滤流程包括：**Stage 1 运动显著性过滤**——通过光流大小筛选包含显著目标运动的视频片段；**Stage 2 场景多样性过滤**——基于语义类别确保场景覆盖室内/室外、多种物体类型；**Stage 3 动态复杂度过滤**——保留包含多目标交互、遮挡、非刚体运动等复杂动态的片段。标注由人工+模型辅助完成，VQA 对覆盖"什么在动""怎么动""为什么动""接下来会怎样"等多个维度
    - 设计动机：随机采样视频难以保证动态场景的质量和多样性。多阶段过滤确保每个场景都有显著且有意义的动态行为

2. **Mask-Guided Fusion**:

    - 功能：通过显式的物体分割 mask 引导 MLLM 关注动态目标
    - 核心思路：对视频中的关键帧使用分割模型（如 SAM）生成物体 mask，将 mask 信息以视觉 prompt 的形式融入 MLLM 的输入中——例如在视频帧上叠加半透明的 mask 高亮，或将 mask 编码为额外的 token。这样 MLLM 在进行 VQA 和推理时能明确"关注哪个物体"
    - 设计动机：现有 MLLMs 在处理视频时往往关注全局语义而忽略局部目标的精确位置和运动。Mask-Guided Fusion 通过显式的空间注意力引导来弥补这一不足

3. **Spatio-Temporal Textual Cognitive Map (ST-TCM)**:

    - 功能：将动态场景的时空信息结构化为文本表示，辅助 MLLM 推理
    - 核心思路：为每个视频构建一个文本化的"认知地图"，记录每个目标在不同时间步的位置、运动方向、速度估计、与其他目标的空间关系变化。这个结构化文本作为额外的上下文输入给 MLLM。例如："[t=1] 物体A在画面左侧，物体B在右侧；[t=2] 物体A向右移动，与B距离缩小；[t=3] 物体A与B发生接触"。这种结构化表示帮助 MLLM 建立时序上的推理链
    - 设计动机：传统的 CoT（链式推理）和 caption-based hints 在动态推理中效果有限，因为它们缺乏结构化的时空信息编码。ST-TCM 提供了一种更紧凑且信息丰富的时空表示

### 损失函数 / 训练策略
Dyn-Bench 本身是评估基准，不涉及训练。Mask-Guided Fusion 和 ST-TCM 是推理时的增强策略，不需要修改模型参数。

## 实验关键数据

### 主实验

评估了通用 MLLMs（GPT-4V, Gemini Pro, LLaVA-Next）、空间 MLLMs（SpatialVLM）和区域级 MLLMs（RegionGPT, Osprey）在 Dyn-Bench 上的表现。

| 模型 | VQA 动态感知 | VQA 时空推理 | 动态定位 mIoU | 综合 |
|------|------------|-----------|-------------|------|
| GPT-4V | 62.3 | 48.7 | 18.2 | 43.1 |
| Gemini Pro | 59.8 | 45.2 | 16.5 | 40.5 |
| LLaVA-Next | 55.1 | 40.3 | 22.4 | 39.3 |
| SpatialVLM | 51.2 | 43.8 | 28.6 | 41.2 |
| RegionGPT | 48.5 | 38.1 | 32.1 | 39.6 |
| GPT-4V + ST-TCM | **68.7** | **56.2** | 19.8 | **48.2** |
| RegionGPT + Mask Fusion | 52.1 | 41.6 | **38.5** | 44.1 |

### 消融实验（增强策略对比）

| 增强策略 | VQA 推理提升 | 定位提升 | 说明 |
|----------|------------|---------|------|
| 无增强（baseline） | — | — | 原始模型 |
| Chain-of-Thought prompt | +1.8 | +0.5 | 效果有限 |
| Caption-based hints | +2.3 | +1.2 | 轻微改善 |
| Mask-Guided Fusion | +3.5 | **+6.4** | 定位大幅提升 |
| ST-TCM | **+7.5** | +1.6 | VQA推理大幅提升 |
| Mask + ST-TCM | +6.8 | +5.9 | 两者结合效果好 |

### 关键发现
- **现有 MLLMs 在动态感知和时空推理之间存在不可调和的矛盾**：擅长 VQA 推理的通用模型（GPT-4V）在定位上很差，擅长区域定位的模型在推理上不行。没有一个模型能同时做好两者
- CoT、caption hints 等传统提示策略对动态推理的提升非常有限（仅 +1.8 ~ +2.3），说明问题不在推理策略，而在于模型对时空动态信息的编码能力不足
- **ST-TCM 在 VQA 推理上提升最显著（+7.5）**，证明将时空动态信息结构化为文本是帮助 MLLMs 理解动态的有效方式
- **Mask-Guided Fusion 在定位上提升最大（+6.4）**，说明显式的空间注意力引导能弥补 MLLMs 对局部目标关注不足的问题
- 真实世界视频比合成视频更具挑战性，MLLMs 在复杂背景和遮挡场景下的性能显著下降

## 亮点与洞察
- **Dyn-Bench 的评估维度设计很系统**：从感知→追踪→推理的层次化覆盖，同时评估语言和视觉两种输出模态，比现有视频理解基准更全面
- **"现有模型无法同时做好推理和定位"这一发现很重要**：它指出了当前 MLLM 架构设计的一个根本缺陷——文本推理和视觉定位在模型内部的表示可能是脱节的
- **ST-TCM 的设计思路可迁移**：将任何结构化知识编码为文本 cognitive map 来辅助 LLM 推理，不仅适用于动态场景，也适用于图结构推理、多步规划等任务

## 局限与展望
- Dyn-Bench 目前规模为 1K 视频/7K VQA/3K 定位对，对于基准来说还可以更大
- 数据来源主要是现有数据集的二次过滤，可能受限于原始数据集的偏差
- Mask-Guided Fusion 依赖于分割模型的质量，在密集遮挡场景下分割不准确会传播误差
- ST-TCM 的文本 cognitive map 构建本身需要较好的目标检测和追踪结果，存在级联误差风险
- 未探索如何将增强方法集成到模型训练中（如用 ST-TCM 作为训练数据增强）

## 相关工作与启发
- **vs MVBench / Video-MME**: 这些视频理解基准关注高层语义理解，Dyn-Bench 聚焦于细粒度时空动态，更能暴露 MLLMs 在动态感知上的不足
- **vs PointOdyssey / TAP-Vid**: 这些基准评估点级追踪能力，但不评估语义推理。Dyn-Bench 同时覆盖追踪和推理
- **vs SpatialBench**: SpatialBench 评估静态空间理解，Dyn-Bench 扩展到时间维度的动态理解

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统化评估 MLLMs 动态4D理解的基准，问题定义新颖
- 实验充分度: ⭐⭐⭐⭐ 覆盖多种类型 MLLMs，评估维度全面，增强方法有充分消融
- 写作质量: ⭐⭐⭐⭐ 概念清晰（感知-追踪-推理），但基准构建细节可以更透明
- 价值: ⭐⭐⭐⭐ 揭示了 MLLMs 在动态理解上的根本短板，对领域发展有指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] 4D LangSplat: 4D Language Gaussian Splatting via Multimodal Large Language Models](4d_langsplat_4d_language_gaussian_splatting_via_multimodal_large_language_models.md)
- [\[CVPR 2025\] Instruction-based Image Manipulation by Watching How Things Move](instruction-based_image_manipulation_by_watching_how_things_move.md)
- [\[CVPR 2025\] Thinking in Space: How Multimodal Large Language Models See, Remember, and Recall Spaces](thinking_in_space_how_multimodal_large_language_models_see_remember_and_recall_s.md)
- [\[CVPR 2025\] Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)
- [\[CVPR 2025\] VisionArena: 230K Real World User-VLM Conversations with Preference Labels](visionarena_230k_real_world_user-vlm_conversations_with_preference_labels.md)

</div>

<!-- RELATED:END -->
