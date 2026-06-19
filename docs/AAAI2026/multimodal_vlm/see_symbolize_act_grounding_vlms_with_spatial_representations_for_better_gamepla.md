---
title: >-
  [论文解读] See, Symbolize, Act: Grounding VLMs with Spatial Representations for Better Gameplay
description: >-
  [AAAI 2026][多模态VLM][符号接地] 系统性评估了符号化空间表示（物体坐标）对VLM游戏能力的影响，发现符号信息仅在检测准确时有益，当VLM自提取符号时效果取决于模型能力和场景复杂度，视觉帧始终不可或缺。 领域现状 VLM越来越多地被用于构建通用AI智能体，不仅要理解视觉场景，还要在交互环境中做出决策…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "符号接地"
  - "VLM游戏智能体"
  - "空间推理"
  - "物体检测"
  - "Atari"
---

# See, Symbolize, Act: Grounding VLMs with Spatial Representations for Better Gameplay

**会议**: AAAI 2026  
**arXiv**: [2603.11601](https://arxiv.org/abs/2603.11601)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 符号接地, VLM游戏智能体, 空间推理, 物体检测, Atari

## 一句话总结
系统性评估了符号化空间表示（物体坐标）对VLM游戏能力的影响，发现符号信息仅在检测准确时有益，当VLM自提取符号时效果取决于模型能力和场景复杂度，视觉帧始终不可或缺。

## 研究背景与动机

### 领域现状
VLM越来越多地被用于构建通用AI智能体，不仅要理解视觉场景，还要在交互环境中做出决策。当VLM从被动感知转向交互式决策时（如机器人、具身AI、游戏环境），它们面临精确空间理解的挑战——这是当前VLM尚未可靠提供的能力。

### 核心痛点

**Atari游戏**提供了研究空间推理挑战的可控环境。Pong、Breakout、Space Invaders等游戏要求精确追踪挡板、球和外星人的位置。在这些条件下，当前VLM频繁出现：
- **物体误识别**
- **重复无效动作**
- **精确控制失败**

### 已有方法局限
- **微调方法**（Zhai et al. 2024）：在数千条游戏轨迹上微调VLM，但牺牲了零样本泛化能力
- **纯视觉方法**（Atari-GPT）：仅使用视觉帧，空间推理仍是瓶颈
- **关键空白**：没有系统性研究评估VLM**识别物体和坐标的准确性**，以及坐标精度**如何影响决策**

### 本文切入点
如果给VLM同时提供视觉帧和场景的**符号化表示**（物体名称+精确坐标），能否提升其交互式决策能力？通过四种管线（纯帧、帧+自提取符号、帧+真实符号、纯符号）的系统对比，隔离各因素的贡献。

## 方法详解

### 整体框架
设计四种实验管线，在Atari、VizDoom和AI2-THOR三个环境中评估三个SOTA VLM（Claude-4-Sonnet、GPT-4o、Gemini-2.5-Pro）。

### 关键设计

1. **四种实验管线**:

    - **F+S-GT（帧+真实符号）**：从游戏RAM直接读取物体坐标（OCAtari），作为性能上界
    - **F（纯帧）**：VLM仅接收原始游戏帧，测试纯视觉空间推理
    - **F+S-self（帧+自提取符号）**：两阶段——VLM先从帧中提取物体坐标，再结合帧和坐标做决策
    - **S-GT（纯符号）**：仅提供RAM中的物体坐标，无视觉帧，测试符号信息的独立贡献
    - 设计动机：通过控制变量法分离视觉帧、符号精度、自提取能力的各自影响

2. **评估指标**:

    - **游戏指标**：600帧累计奖励，归一化到0-100区间
    - **检测指标**：F1分数（物体识别精度）和IoU（坐标重叠度），在100帧上评估
    - 设计动机：将游戏表现与感知质量解耦，分析因果关系

3. **消融研究设计**:

    - **分辨率消融**：测试160×210到1280×720四种分辨率对检测质量的影响
    - **噪声消融**：向真实坐标注入高斯噪声 $x' = x + \mathcal{N}(0, \sigma \times W)$，σ从0到0.4，10个种子各300帧
    - 设计动机：量化"符号从有用到有害的转折点"

### 训练策略
完全**零样本**设置，不进行任何微调。使用通用提示模板，不包含任务特定指令或策略提示。

## 实验关键数据

### 主实验（Atari环境 - 600帧累计奖励）

| 模型 | 管线 | Pong | Breakout | Space Invaders |
|------|------|------|----------|----------------|
| Claude-4-Sonnet | F (纯帧) | -16.0 | 0.0 | 80.0 |
| Claude-4-Sonnet | F+S-self | **-3.0** | **12.0** | **150.0** |
| Claude-4-Sonnet | F+S-GT (上界) | -1.0 | 12.0 | 175.0 |
| GPT-4o | F (纯帧) | -5.0 | 7.5 | **130.0** |
| GPT-4o | F+S-self | -6.5 | 8.0 | 65.0 ↓ |
| GPT-4o | F+S-GT (上界) | -3.0 | 13.0 | 185.0 |
| Gemini-2.5-Pro | F (纯帧) | -7.0 | 7.0 | **95.0** |
| Gemini-2.5-Pro | F+S-self | -3.0 | 10.0 | 80.0 ↓ |
| Gemini-2.5-Pro | F+S-GT (上界) | -1.0 | 12.0 | 170.0 |

关键发现：Claude自提取符号后大幅提升，GPT-4o和Gemini在复杂场景（Space Invaders）反而下降！

### 消融实验一：物体检测质量

| 模型 | F1 Score | IoU |
|------|----------|-----|
| Claude-4-Sonnet | **0.715** | **0.533** |
| Gemini-2.5-Pro | 0.189 | 0.202 |
| GPT-4o | 0.124 | 0.128 |

**Claude的检测F1是GPT-4o的5.8倍**，解释了为什么只有Claude从自提取符号中获益。

### 消融实验二：坐标噪声对Breakout性能的影响

| 噪声级别σ | Claude奖励 | GPT-4o奖励 | 说明 |
|-----------|-----------|-----------|------|
| 0.0 (无噪声) | 5.0 | 5.0 | 基线 |
| 0.1 (~16-20px误差) | 4.3 | 4.0 | 已降30-40% |
| 0.2 (~32-40px误差) | 3.4 | 3.0 | 无优势 |
| 0.3 | 3.4 | 2.3 | 低于纯帧 |
| 0.4 (~64-80px误差) | 2.8 | 2.6 | 严重退化 |

### 扩展环境（VizDoom + AI2-THOR）

| 模型 | VizDoom F | VizDoom F+S-self | AI2-THOR F | AI2-THOR F+S-self |
|------|-----------|-----------------|------------|-------------------|
| Claude-4-Sonnet | 5 | **9** | -1.0 | **2.0** |
| GPT-4o | **12** | 8↓ | 7.0 | **9.0** |
| Gemini-2.5-Pro | **11** | 4↓ | 5.0 | 1.0↓ |

3D环境验证了核心发现：符号有用当且仅当检测准确。

### 关键发现

1. **"符号有益"不是绝对的**：仅当检测精度足够高时（Claude的F1=0.715），自提取符号才有帮助
2. **视觉帧不可或缺**：即使提供完美符号坐标，移除视觉帧后性能暴跌（纯符号管线表现最差）
3. **噪声容忍度极低**：仅16-20像素的坐标误差就导致30-40%的性能下降
4. **场景复杂度是关键变量**：Pong（2-4物体）→成功，Space Invaders（20-50物体）→失败
5. **分辨率是简单有效的提升手段**：从160×210到1280×720，F1从0.31翻倍到0.68
6. **感知质量是瓶颈**：不是符号接地这个idea有问题，而是VLM的感知还不够准确

## 亮点与洞察

- **控制变量法的典范**：四种管线+三种环境+三个模型，实验设计极为系统，每个因素都被隔离分析
- **核心结论简明有力**："Symbolic grounding helps only when symbols are accurate"——一句话概括全文
- **揭示了VLM间的感知差异**：Claude的物体检测远优于GPT-4o和Gemini，这个发现本身就有独立价值
- **噪声消融**首次量化了"坐标误差→决策退化"的传播关系
- 实验扩展到VizDoom和AI2-THOR证明了发现的通用性

## 局限与展望

- **环境有限**：尽管扩展到3个环境，但仍是游戏场景，与真实世界机器人控制有差距
- **API成本**：每帧调用VLM API，延迟和费用使实时游戏不可行
- **未探索鲁棒的符号提取方法**：如混合检测器、轻量级微调
- **固定提示模板**：未尝试针对不同场景复杂度自适应调整提示策略
- **仅测试3个VLM**：未包括开源模型（如LLaVA、Qwen-VL）

## 相关工作与启发

- 与Atari-GPT的区别：系统研究符号信息的效果，而非仅用帧做零样本
- 与OCAtari的关系：利用OCAtari提供真实符号坐标作为实验上界
- **符号接地问题（Harnad 1990）** 的新视角：不是符号-感知的对齐问题，而是符号提取的精度问题
- PoE-World假设可靠的符号输入，本文证明这个假设往往不成立
- 启发：**提升VLM的空间感知精度**可能比设计更复杂的推理策略更重要

## 评分
- 新颖性: ⭐⭐⭐（实验驱动而非方法创新，但问题定义新颖）
- 实验充分度: ⭐⭐⭐⭐⭐（极其系统的控制变量实验+多环境+消融）
- 写作质量: ⭐⭐⭐⭐⭐（结构清晰，结论明确，图表丰富）
- 价值: ⭐⭐⭐⭐（为VLM智能体研究提供了重要的实证基础）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] See What I Mean: Aligning Vision and Language Representations for Video Fine-grained Object Understanding](../../CVPR2026/multimodal_vlm/see_what_i_mean_aligning_vision_and_language_representations_for_video_fine-grai.md)
- [\[NeurIPS 2025\] See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model](../../NeurIPS2025/multimodal_vlm/seetrek_training-free_spatial_prompting_for_multimodal_large_language_model.md)
- [\[CVPR 2026\] See and Fix the Flaws: Enabling VLMs and Diffusion Models to Comprehend Visual Artifacts via Agentic Data Synthesis](../../CVPR2026/multimodal_vlm/see_and_fix_the_flaws_enabling_vlms_and_diffusion_models_to_comprehend_visual_ar.md)
- [\[AAAI 2026\] The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations](the_triangle_of_similarity_a_multi-faceted_framework_for_comparing_neural_networ.md)
- [\[NeurIPS 2025\] T-Rex: Task-Adaptive Spatial Representation Extraction for Robotic Manipulation with VLMs](../../NeurIPS2025/multimodal_vlm/t-rex_task-adaptive_spatial_representation_extraction_for_robotic_manipulation_w.md)

</div>

<!-- RELATED:END -->
