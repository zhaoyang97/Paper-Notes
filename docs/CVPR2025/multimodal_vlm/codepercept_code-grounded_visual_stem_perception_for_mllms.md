---
title: >-
  [论文解读] CodePercept: Code-Grounded Visual STEM Perception for MLLMs
description: >-
  [CVPR 2025][多模态VLM][STEM感知] 通过 scaling 分析发现 STEM 视觉推理的真正瓶颈是感知而非推理，提出用可执行 Python 代码作为精确感知媒介——构建 ICC-1M 数据集（Image-Caption-Code 三元组）训练模型，在 STEM 感知基准上 CodePercept-8B 比 Qwen3-VL-8B 提升 +3.0%-12.3%。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "STEM感知"
  - "代码生成"
  - "视觉推理"
  - "数据合成"
  - "感知瓶颈"
---

# CodePercept: Code-Grounded Visual STEM Perception for MLLMs

**会议**: CVPR 2025  
**arXiv**: [2603.10757](https://arxiv.org/abs/2603.10757)  
**代码**: [https://github.com/TongkunGuan/Qwen-CodePercept](https://github.com/TongkunGuan/Qwen-CodePercept)  
**领域**: 多模态VLM  
**关键词**: STEM感知, 代码生成, 视觉推理, 数据合成, 感知瓶颈

## 一句话总结

通过 scaling 分析发现 STEM 视觉推理的真正瓶颈是感知而非推理，提出用可执行 Python 代码作为精确感知媒介——构建 ICC-1M 数据集（Image-Caption-Code 三元组）训练模型，在 STEM 感知基准上 CodePercept-8B 比 Qwen3-VL-8B 提升 +3.0%-12.3%。

## 研究背景与动机

**领域现状**：当前 MLLM 在 STEM 领域的改进集中在推理能力——cold-start 数据、RL 训练、text-only thinking data 迁移。大量工作用 RL reward 来提升数学/科学推理。

**现有痛点**：模型在 STEM 任务上失败时，不知道是感知不够还是推理不够。传统 benchmark 只测问题求解准确率，无法分离两种能力。

**核心矛盾**：作者通过 scaling 实验揭示——将 STEM 推理解耦为 perception (image→caption) 和 reasoning (caption→answer)，独立 scale 两者时，perception scaling 始终优于 reasoning scaling。这表明感知才是真正的杠杆。

**本文目标** 如何系统性提升 MLLM 在 STEM 领域的视觉感知能力？

**切入角度**：自然语言描述 STEM 图像时存在"描述失语"——复杂空间关系、精确数值无法用自然语言完整表达。但可执行代码天然具有精确语义，与 STEM 图像的结构化特性高度匹配。

**核心 idea**：用可执行 Python 代码作为 STEM 视觉感知的 ground truth 和训练媒介——能准确重构图像才证明真正理解了图像。

## 方法详解

### 整体框架

三大组件：(1) ICC-1M 数据集构建（1M Image-Caption-Code 三元组），(2) 两个 Code-Grounded 训练任务（代码驱动的 caption 生成 + 图像到代码翻译），(3) STEM2Code-Eval 基准（通过代码重构评估感知能力）。训练分两阶段：SFT + RL。

### 关键设计

1. **Scaling 分析（关键发现）**：

    - 功能：将 STEM 推理解耦为 感知（MLLMcaptioner → 描述）和 推理（LLMsolver → 答案）两阶段
    - 实验设计：固定一方用 4B 模型，另一方分别用 4B/8B/32B 模型，观测性能变化
    - 关键结论：scaling perception 的提升始终大于 scaling reasoning。用 Qwen3-VL-Thinking 模型在 MathVision 上验证。这回答了一个长期存在的问题——STEM 失败的根因是感知

2. **ICC-1M 数据集构建（三条 pipeline）**：

    - **Image Reproduce (IR)**：用 MLLM 为现有 STEM 图像生成重构代码——先 caption 理解内容，再据此 + 原图生成代码
    - **Image Diversity (ID)**：从种子图像提取底层 STEM 原理，然后在不同视觉上下文中重新实例化（如从多米诺逻辑谜题→圆形多米诺轮、三角排列等），扩展多样性
    - **Solid Geometry (SG)**：用参数化模板生成立体几何图像+代码，解决 LLM 无法生成准确 3D 空间代码的问题
    - 统一质量控制：图像质量 + 代码质量 + 图像-代码一致性三重过滤

3. **Code-Grounded Caption Generation**：

    - 功能：利用可执行代码作为 ground truth 来生成精确 caption
    - 核心思路：先生成 native caption（可能有幻觉）→ 分析代码+执行日志提取 verified visual facts → 用 visual facts 修正 caption 中的错误
    - 亮点：execution tracer 自动记录几何坐标、数量、颜色等精确信息，解决代码复杂逻辑难以直接分析的问题
    - 设计动机：直接让 MLLM 描述 STEM 图像会产生数值、空间关系的幻觉

4. **STEM Image-to-Code Translation**：

    - 功能：训练模型从图像直接生成可执行重构代码
    - 核心思路：先生成解释性代码草稿（有步骤说明但可能错误）→ 用 ground truth 代码修正错误，保留解释性结构
    - 代码作为"结构化 caption"，与自然语言 caption 互补

### 训练策略

- **SFT**：联合训练 image→caption 和 image→code 两个任务，让语义理解和结构化理解相互增强
- **RL（GRPO）**：只对代码生成做 RL，reward = format reward（合法 Python 块）+ content reward（可执行 + GPT-4o 评分代码语义 + GPT-4o 评分图像相似度）
- 基于 Qwen3-VL 系列，32×A100 训练

## 实验关键数据

### 主实验（Captioner-Solver 感知评估）

| 模型 | MathVision | MathVista | MathVerse | DynaMath | WeMath | LogicVista | Avg |
|------|-----------|-----------|-----------|---------|--------|------------|-----|
| Qwen3-VL-8B-Instruct | 54.37 | 69.60 | 63.75 | 72.19 | 45.43 | 56.82 | 60.36 |
| **CodePercept-8B-S1** | 59.31 (+5.0) | 70.20 (+0.6) | 66.52 (+2.8) | 73.20 (+1.0) | 49.14 (+3.7) | 61.52 (+4.7) | **63.32 (+3.0)** |
| CodePercept-32B-S1 | 62.27 (+3.7) | 72.90 | 71.70 | 77.41 | 54.19 (+6.2) | 65.33 | **67.30 (+2.7)** |

### STEM2Code-Eval（代码重构评估）

| 模型 | Image Score | Code Score | Avg | Exec Rate |
|------|-----------|-----------|-----|-----------|
| Qwen3-VL-8B-Instruct | 28.59 | 28.23 | 28.41 | 85.3% |
| CodePercept-8B-S1 | 44.53 | 46.78 | 45.66 | 87.6% |
| CodePercept-8B-R1 | **50.25** | **47.04** | **48.65** | 93.4% |
| Gemini2.5-Pro-Thinking | 68.89 | 75.41 | 72.15 | 91.7% |

### 消融实验

| 数据配置 | Avg Score |
|---------|----------|
| Qwen3-VL-8B baseline | 60.36 |
| + IR-CodeCap | 60.91 (+0.6) |
| + ID-CodeCap | 62.15 (+1.8) |
| + SG-CodeCap | 62.75 (+2.4) |
| NativeCap (直接caption，无代码) | 60.78 |
| CodeCap (代码驱动caption) | 62.75 (+2.0) |
| CodeCap + ImCode (联合训练) | **63.32** (+2.5) |

### 关键发现

- **感知是瓶颈的实证**：scaling 分析非常有说服力——独立放大感知能力的收益始终超过放大推理能力
- **代码驱动 caption 比直接 caption 好 +2.0%**：验证了代码消除幻觉的有效性
- **Caption 和 Code 互补**：联合训练比单独 caption (+0.6%) 或单独 code 效果更佳
- **RL 对代码生成帮助巨大**：CodePercept-8B-R1 vs S1 在 STEM2Code-Eval 上 +3.0，Exec Rate +5.8%
- **CodePercept-8B 超越多个 72B 模型**：在 captioner-solver 评估中超越 Qwen2.5-VL-72B

## 亮点与洞察

- **"感知是瓶颈"是一个被忽视但极重要的发现**。当前 STEM AI 研究几乎全聚焦推理（RL、CoT），但这篇论文用 controlled experiment 证明应先解决感知。这可能改变该领域的研究优先级。
- **"代码作为感知媒介"的 insight 非常精妙**：代码天然具有精确性（坐标、颜色值）、可验证性（能执行）和结构性（层次化描述），这些都是自然语言缺乏的。
- **Execution tracer**：执行代码并记录所有渲染细节（坐标、z-order、颜色 RGB），作为代码分析的"说明书"，巧妙地解决了 LLM 难以分析复杂代码逻辑的问题。
- **STEM2Code-Eval benchmark 的设计理念**：只有能重构图像才算真正理解图像——这比回答问题更全面地评估感知。

## 局限与展望

- **代码生成限于 matplotlib**：某些 STEM 图像（如真实照片、手绘图）无法用 matplotlib 重构
- **RL reward 依赖 GPT-4o**：GPT-4o 评分不稳定且昂贵，可探索基于像素相似度的自动 reward
- **只测试了 STEM 领域**：代码作为感知媒介的思路是否适用于其他领域（医学影像、遥感）？
- **感知 vs 推理的解耦不完美**：caption 质量同时影响感知和表达能力，解耦不彻底

## 相关工作与启发

- **vs Vision-R1/Video-R1 等 RL 推理工作**：这些工作全力提升推理，CodePercept 指出他们可能在优化错误的方向——应先提升感知。
- **vs Chart/UI 代码生成**：那些是面向下游应用的代码生成，CodePercept 是用代码作为感知增强手段，目标不同但技术可互通。
- **启发**：也许其他模态（音频→MIDI、3D→渲染代码）也可以用"代码作为感知媒介"的思路来提升理解能力。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "感知是瓶颈"+代码作为感知的双重洞察，非常原创
- 实验充分度: ⭐⭐⭐⭐⭐ 6个STEM基准，scaling分析，全面消融，3个模型规模（4/8/32B）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但公式稍多
- 价值: ⭐⭐⭐⭐⭐ 可能改变STEM AI研究的优先级，benchmark + 数据集 + 方法贡献全面

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] StarVector: Generating Scalable Vector Graphics Code from Images and Text](starvector_generating_scalable_vector_graphics_code_from_images_and_text.md)
- [\[CVPR 2025\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2025\] Evaluating Model Perception of Color Illusions in Photorealistic Scenes](evaluating_model_perception_of_color_illusions_in_photorealistic_scenes.md)
- [\[CVPR 2025\] VidComposition: Can MLLMs Analyze Compositions in Compiled Videos?](vidcomposition_can_mllms_analyze_compositions_in_compiled_videos.md)
- [\[CVPR 2025\] PEACE: Empowering Geologic Map Holistic Understanding with MLLMs](peace_empowering_geologic_map_holistic_understanding_with_mllms.md)

</div>

<!-- RELATED:END -->
