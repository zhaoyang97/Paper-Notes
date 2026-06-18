---
title: >-
  [论文解读] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation
description: >-
  [ICCV 2025][多模态VLM][视角感知推理] 提出 Abstract Perspective Change (APC) 框架，通过利用视觉基础模型构建场景抽象表示并执行透视变换，使 VLM 能够从任意视角进行空间推理，在合成与真实图像基准上大幅优于现有 VLM 和微调模型。 领域现状：视觉语言模型 (VLM) 在空…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视角感知推理"
  - "心理意象模拟"
  - "视觉语言模型"
  - "空间推理"
  - "透视变换"
---

# Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation

**会议**: ICCV 2025  
**arXiv**: [2504.17207](https://arxiv.org/abs/2504.17207)  
**代码**: [https://github.com/KAIST-Visual-AI-Group/APC-VLM](https://github.com/KAIST-Visual-AI-Group/APC-VLM)  
**领域**: 多模态VLM  
**关键词**: 视角感知推理, 心理意象模拟, 视觉语言模型, 空间推理, 透视变换

## 一句话总结

提出 Abstract Perspective Change (APC) 框架，通过利用视觉基础模型构建场景抽象表示并执行透视变换，使 VLM 能够从任意视角进行空间推理，在合成与真实图像基准上大幅优于现有 VLM 和微调模型。

## 研究背景与动机

**领域现状**：视觉语言模型 (VLM) 在空间推理任务上取得了显著进展，包括物体位置关系判断、深度估计辅助的空间问答等。目前主流的做法是直接将图像和文本问题输入 VLM 进行端到端推理，或通过微调加入空间推理数据来增强模型能力。

**现有痛点**：近期研究表明，现有 VLM 在需要从非相机视角（即非自我中心视角）进行推理时表现极差，存在严重的"自我中心偏差"——模型默认从拍照者的视角回答问题，无法切换到场景中其他人或位置的视角。例如当问"从桌子对面的人看来，杯子在盘子的左边还是右边"时，VLM 几乎总是从拍照者视角回答，导致答案错误。

**核心矛盾**：VLM 缺乏"心理旋转"能力——人类可以在脑中想象场景的抽象表示并自由旋转以切换视角，但 VLM 的 2D 视觉编码器天然绑定于输入图像的相机视角，无法进行这种透视变换。这不是简单的数据增强或微调能解决的结构性缺陷。

**本文目标**：(1) 从输入图像构建场景的 3D 抽象表示；(2) 将抽象表示变换到目标视角；(3) 将变换后的信息以 VLM 能理解的方式传递给模型。

**切入角度**：作者从认知心理学中"心理意象" (mental imagery) 的概念出发——人类在进行视角切换时，并非在脑中重建完整的 3D 场景，而是形成一个简化的抽象表示（比如只记住物体的相对位置和朝向），然后在这个抽象表示上进行旋转。这种"先抽象、再变换"的策略比直接进行 novel view synthesis 更高效且更稳健。

**核心 idea**：用现成的视觉基础模型（检测、分割、深度估计、朝向估计）从图像中提取 3D 场景抽象，通过坐标变换模拟视角切换，再以数值或视觉提示的形式反馈给 VLM，实现任意视角的空间推理。

## 方法详解

### 整体框架

APC 框架接收一张场景图像和一个需要从特定参考视角回答的空间问题，输出从该视角看到的正确空间关系答案。整体 pipeline 分为三个阶段：(1) Scene Abstraction——利用视觉基础模型构建场景的 3D 抽象表示；(2) Perspective Change——将 3D 抽象变换到参考视角的坐标系；(3) Perspective Prompting——将变换后的信息编码为 VLM 能理解的提示。

### 关键设计

1. **Scene Abstraction（场景抽象构建）**:

    - 功能：从单张 2D 图像构建包含物体 3D 位置和朝向的简化场景表示
    - 核心思路：首先用 Grounding DINO 进行开放词汇检测定位问题中提及的物体，然后用 SAM 获取精细分割 mask 确定物体中心像素位置，接着用 Depth Pro 估计每个物体的深度值从而反投影到 3D 空间，最后用 Orient Anything 估计物体的朝向向量。最终得到每个物体的 3D 坐标 $(x, y, z)$ 和朝向 $(\theta)$
    - 设计动机：不需要重建完整的 3D 场景（如 NeRF），只需要物体级别的位置和朝向信息，这与人类心理意象中"粗略但足够"的抽象策略一致，同时避免了 3D 重建的高计算成本和质量不稳定问题

2. **Perspective Change（透视变换）**:

    - 功能：将场景抽象从相机坐标系变换到参考视角（allocentric viewpoint）的坐标系
    - 核心思路：根据问题确定参考视角的位置和朝向（例如"从 Alice 的角度看"，则将 Alice 的位置和面部朝向作为新坐标系原点和正前方），对场景中所有物体的 3D 坐标进行旋转和平移变换 $\mathbf{p}' = R(\theta)(\mathbf{p} - \mathbf{t})$，得到从参考视角看到的各物体相对位置
    - 设计动机：这一步是整个框架的核心——直接在 3D 抽象层面做坐标变换比用 novel view synthesis 重新渲染整张图像更精确、更快，且不受渲染质量的影响

3. **Perspective Prompting（透视提示）**:

    - 功能：将变换后的场景信息以 VLM 能理解的方式输入，引导其从正确视角回答
    - 核心思路：提供两种形式——(a) **数值提示**：将每个物体变换后的 3D 坐标直接以文本形式写在 prompt 中，如"From the reference perspective, object A is at (1.2, -0.3, 0.5)"；(b) **视觉提示**：在每个物体的 3D 位置放置彩色方块，然后从参考视角渲染一张抽象的鸟瞰图/正面图，连同颜色-物体映射一起输入 VLM
    - 设计动机：两种提示方式适用于不同场景——数值提示对空间推理能力强的 VLM（如 GPT-4o）效果好，视觉提示对视觉理解更强的模型（如 Qwen2.5-VL）更友好

### 损失函数 / 训练策略

APC 是一个 training-free 框架，不需要额外的训练或微调。所有组件（检测、分割、深度估计、朝向估计）都使用现成的预训练模型，坐标变换是确定性计算，VLM 推理使用原始的预训练权重配合精心设计的 prompt。

## 实验关键数据

### 主实验

在合成场景基准 (Spatial-Map) 和真实图像基准上的视角推理准确率对比：

| 方法 | Spatial-Map Ego (%) | Spatial-Map Allo (%) | Real-Image Allo (%) | 类型 |
|------|-------|--------|---------|------|
| GPT-4o (baseline) | 78.5 | 42.3 | 38.7 | 原始VLM |
| Qwen2.5-VL (baseline) | 72.1 | 35.8 | 33.2 | 原始VLM |
| Cambrian-1 (baseline) | 65.4 | 30.5 | 28.9 | 原始VLM |
| SpatialRGPT (fine-tuned) | 70.2 | 45.6 | 41.3 | 微调模型 |
| NVS-based approach | 68.3 | 48.2 | 39.8 | NVS辅助 |
| **APC + GPT-4o** | **79.1** | **68.7** | **62.5** | 本文 |
| **APC + Qwen2.5-VL** | **73.5** | **63.2** | **57.8** | 本文 |

### 消融实验

| 配置 | Allo Accuracy (%) | 说明 |
|------|---------|------|
| Full APC (visual prompt) | 63.2 | 完整模型 + 视觉提示 |
| Full APC (numerical prompt) | 68.7 | 完整模型 + 数值提示 |
| w/o Depth (2D only) | 48.5 | 去掉深度估计，仅用 2D 坐标 |
| w/o Orientation | 55.3 | 去掉朝向估计 |
| w/o Scene Abstraction | 42.3 | 直接用原图 (baseline) |
| Random perspective guess | 25.0 | 随机猜测（4选1） |

### 关键发现

- 深度估计是最关键的模块，去掉后 allocentric 准确率从 68.7% 降至 48.5%，说明 3D 位置信息是视角变换的基础
- 数值提示在 GPT-4o 上优于视觉提示（68.7% vs 63.2%），但在视觉能力更强的模型上视觉提示可能更有优势
- APC 在相机与参考视角偏差角度 $\theta$ 较大时仍保持稳定准确率，而 baseline VLM 随角度增大急剧下降——说明框架确实实现了真正的视角解耦
- 真实图像上的提升比合成环境略小，主要受限于深度估计和物体检测在复杂真实场景中的准确度

## 亮点与洞察

- **"先抽象再变换"的范式非常巧妙**：不试图在像素级别解决视角变换（避免 NVS 的质量问题），而是在物体级别的抽象表示上做简单的坐标变换。这与人类认知过程高度一致，且计算成本极低
- **完全 training-free 的模块化设计**：所有组件即插即用，可以随着视觉基础模型的进步自动获得提升，无需重新训练
- **透视提示的双模态设计思路可迁移**：将 3D 信息同时编码为文本和视觉两种形式供 VLM 使用的思路，可以推广到任何需要向 VLM 注入 3D 空间信息的任务（如机器人导航、具身问答）

## 局限与展望

- 场景抽象的质量高度依赖于物体检测和深度估计的准确性，在遮挡严重或深度模糊的场景中可能失效
- 当前只处理刚体物体的位置和朝向，无法处理非刚体变形或细粒度的空间关系（如"A 叠在 B 上面"的垂直关系）
- 假设参考视角可以从问题中明确确定（需要知道参考人物的位置和朝向），在自然语言中这一信息有时是隐含的
- 未来可以结合 3D scene graph 或 world model 来构建更丰富的场景抽象，支持更复杂的空间推理

## 相关工作与启发

- **vs SpatialRGPT**: SpatialRGPT 通过在空间推理数据上微调 VLM 来提升能力，但微调数据主要是自我中心视角的，对 allocentric 推理提升有限。APC 的优势在于不需要训练数据，且通过显式的坐标变换实现了真正的视角无关推理
- **vs Novel View Synthesis (NVS)**: NVS 方法（如 Zero-1-to-3）尝试先合成新视角图像再让 VLM 推理，但生成质量不稳定且计算昂贵。APC 跳过了像素级重建，直接在抽象层面做变换，更高效稳健
- **vs 3DSRBench/3D-PC**: 这些是视角推理的评估基准，APC 在这些基准上展示了显著提升，可以作为未来视角推理研究的强 baseline

## 评分

- 新颖性: ⭐⭐⭐⭐ 心理意象模拟的认知启发角度新颖，但各个模块本身都是现有工具的组合
- 实验充分度: ⭐⭐⭐⭐ 合成和真实场景都有测试，消融实验覆盖了关键组件，但缺少在更大规模 benchmark 上的评估
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、方法描述直观、从认知科学到技术方案的逻辑链完整流畅
- 价值: ⭐⭐⭐⭐ 指出并解决了 VLM 的一个重要能力缺陷，training-free 设计实用性强，对具身智能应用有直接推动

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](../../ICML2026/multimodal_vlm/3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICCV 2025\] Understanding Museum Exhibits using Vision-Language Reasoning](understanding_museum_exhibits_using_vision-language_reasoning.md)
- [\[ICCV 2025\] Global and Local Entailment Learning for Natural World Imagery](global_and_local_entailment_learning_for_natural_world_imagery.md)
- [\[ICCV 2025\] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks](geobench-vlm_benchmarking_vision-language_models_for_geospatial_tasks.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](llava-cot_let_vision_language_models_reason_step-by-step.md)

</div>

<!-- RELATED:END -->
