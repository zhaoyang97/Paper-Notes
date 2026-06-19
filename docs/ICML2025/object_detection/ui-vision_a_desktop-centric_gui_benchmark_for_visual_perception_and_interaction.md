---
title: >-
  [论文解读] UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction
description: >-
  [ICML 2025][目标检测][GUI benchmark] 提出 UI-Vision——首个面向桌面环境的综合离线评估基准，覆盖 83 个软件应用，提供密集的 bounding box、UI 标签和操作轨迹标注，定义从细粒度到粗粒度的三级评估任务（Element Grounding → Layout Grounding → Action Prediction），系统评估并揭示 SOTA 模型在专业软件理解、空间推理和复杂操作上的关键短板。
tags:
  - "ICML 2025"
  - "目标检测"
  - "GUI benchmark"
  - "desktop agent"
  - "element grounding"
  - "layout grounding"
  - "action prediction"
---

# UI-Vision: A Desktop-centric GUI Benchmark for Visual Perception and Interaction

**会议**: ICML 2025  
**作者**: Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Juan A. Rodriguez, Montek Kalsi 等  
**arXiv**: [2503.15661](https://arxiv.org/abs/2503.15661)  
**代码**: 开源（MIT License）  
**领域**: 目标检测  
**关键词**: GUI benchmark, desktop agent, element grounding, layout grounding, action prediction  

## 一句话总结

提出 UI-Vision——首个面向桌面环境的综合离线评估基准，覆盖 83 个软件应用，提供密集的 bounding box、UI 标签和操作轨迹标注，定义从细粒度到粗粒度的三级评估任务（Element Grounding → Layout Grounding → Action Prediction），系统评估并揭示 SOTA 模型在专业软件理解、空间推理和复杂操作上的关键短板。

## 研究背景与动机

**领域现状**：自主 GUI 智能体通过导航图形用户界面自动化文档编辑、文件管理等任务，是 LLM Agent 的重要应用方向。当前已有多个 Web 和移动端 GUI 基准（如 Mind2Web、ScreenSpot、OSWorld），推动了 GUI 理解模型的快速发展。

**现有痛点**：(1) **桌面环境严重缺失**——桌面原生应用（IDE、图像编辑器、终端等）的界面复杂度远超 Web 页面，但因数据采集困难和软件许可问题几乎没有对应基准；(2) **在线评估不可复现**——现有基准多依赖在线交互环境，每次运行结果可能不同，无法标准化比较；(3) **标注稀疏**——现有桌面数据集缺乏密集标注（精确 bounding box、功能标签、完整操作轨迹），评估粒度粗糙；(4) **缺乏层次化评估**——无法精确定位模型在视觉感知链路中的能力瓶颈。

**核心矛盾**：桌面环境是最重要的专业工作场景之一，但当前 GUI 智能体的视觉感知能力在桌面环境下的真实表现完全未知。

**切入角度**：构建许可友好（MIT License）的离线静态基准，通过三级任务设计从"能否看到元素"到"能否理解布局"再到"能否执行操作"逐层诊断模型能力。

**核心 idea**：用密集标注的桌面 GUI 截图构建层次化离线基准，精确定位 GUI 智能体的视觉感知瓶颈。

## 方法详解

### 整体框架

UI-Vision 包含三个核心组件：(1) 覆盖 83 个桌面应用的密集标注数据集；(2) 由细到粗的三级评估任务体系；(3) 每级任务配套的标准化评估指标。所有评估采用离线模式——基于静态截图而非在线交互，确保完全可复现。

### 关键设计

1. **数据集与标注体系**:

    - 功能：提供高质量、高密度的桌面 GUI 评估数据
    - 核心思路：收集人类在 83 个桌面软件中的真实操作示范，为每个截图提供三类密集标注：(a) UI 元素的精确 bounding box；(b) 每个元素的功能描述标签（UI labels）；(c) 完整的操作轨迹，包括点击坐标、拖拽起终点、键盘输入序列。软件类别涵盖创意工具（Photoshop、Blender）、开发环境（VS Code、Terminal）、办公套件、系统工具、通信工具等
    - 设计动机：密集标注使得同一数据集可以支撑多粒度评估，避免为不同任务分别构建数据集

2. **三级评估任务**:

    - 功能：由细到粗逐层诊断模型在视觉感知链路中的能力瓶颈
    - 核心思路：
        - **Task 1: Element Grounding（元素定位）**——给定截图和元素描述，预测目标 UI 元素的 bounding box。进一步细分为 Basic（基础元素识别）、Functional（基于功能描述定位）和 Spatial（涉及空间关系推理）三个子集。评估指标：IoU 匹配准确率
        - **Task 2: Layout Grounding（布局定位）**——给定截图，识别和定位所有 UI 元素及其层次关系，评估模型对整体界面布局的理解能力
        - **Task 3: Action Prediction（动作预测）**——给定截图和任务描述，预测下一步操作（点击位置、拖拽轨迹或键盘输入）。这是最具挑战性的端到端任务，同时考验视觉理解和操作推理
    - 设计动机：层次化设计使得研究者可以精确定位失败发生在哪个环节——是看不到元素、不理解布局，还是不知道该做什么操作

3. **离线评估设计**:

    - 功能：确保评估的可复现性、标准化和可扩展性
    - 核心思路：所有评估基于相同的静态截图集合，不需要与实际操作系统交互。新模型只需在固定输入上推理即可快速评估，避免了在线环境的非确定性
    - 设计动机：在线基准的运行结果受环境状态影响（弹窗、窗口位置变化等），难以公平比较不同模型

### 损失函数

本工作为基准构建，不涉及模型训练。

## 实验

### 主实验——跨模型评估

| 模型 | Element (Basic) | Element (Spatial) | Layout Grounding | Action (Click) | Action (Drag) |
|------|----------------|-------------------|-----------------|----------------|---------------|
| UI-TARS-72B | 最高 | 中等 | 较高 | 中等 | 低 |
| Claude 3.5 Sonnet | 中等 | 较低 | 中等 | 较低 | 低 |
| GPT-4o | 中等 | 较低 | 较低 | 较低 | 低 |
| Qwen2-VL | 中等 | 较低 | 中等 | 较低 | 低 |

### 消融实验——按挑战维度的错误分析

| 挑战维度 | 表现特征 | SOTA 典型错误 |
|---------|---------|--------------|
| 专业软件理解 | 所有模型显著退化 | 无法识别专业工具栏中的特殊图标和控件 |
| 空间推理 | Spatial 子集准确率大幅低于 Basic | "右侧第二个按钮"等方位描述判断错误 |
| 拖拽操作 | Action Prediction 中最低分 | 起终点坐标预测偏差大，路径规划能力弱 |
| 键盘输入 | 次于点击但优于拖拽 | 快捷键组合预测困难 |
| 跨应用泛化 | 常用应用 >> 专业应用 | 对小众/专业软件界面理解能力骤降 |

### 关键发现

- 即使最强的 UI-TARS-72B（72B 参数、专门训练），在 Spatial 元素定位和拖拽操作预测上仍表现不佳
- 专业软件（3D 建模、音视频编辑）是所有模型的盲区——训练数据中几乎不包含这类界面
- Element Grounding 从 Basic → Functional → Spatial，难度阶梯式增加，Spatial 是最普遍的瓶颈
- 动作预测中拖拽操作远难于点击——模型需要理解起点、终点、路径方向三个维度
- 数据集覆盖 83 个应用，横跨创意、开发、办公、系统、通信五大类别

## 亮点与洞察

- 填补了桌面 GUI 智能体评估的重大空白，MIT License 完全开放，社区可直接使用
- 三级任务阶梯设计精巧——从"看到"到"理解"到"操作"逐层诊断，比端到端评估更有诊断价值
- 83 个应用的广泛覆盖揭示了"应用长尾"问题——常见应用上的能力不能推广到专业领域
- 离线静态设计解决了在线基准不可复现的老问题，降低了评估成本

## 局限性

- 数据规模有限（约 1464 条标注样本），相比 Web 基准数量偏少
- 主要针对特定桌面操作系统，缺乏跨 OS（Windows/macOS/Linux）的泛化评估
- 离线评估无法捕捉动态交互中的时序依赖和状态反馈循环
- 仅评估单步预测，未衡量多步骤任务的端到端成功率
- 空间推理子集的标注可能存在歧义（"旁边"的精确定义因人而异）

## 相关工作与启发

- **与 ScreenSpot 的关系**：ScreenSpot 侧重移动端，UI-Vision 侧重桌面端，在 GUI 智能体评估体系中互补
- **与 OSWorld 的差异**：OSWorld 是在线交互基准，UI-Vision 是离线静态基准，后者牺牲了交互真实性但换取了可复现性
- **启发**：在训练数据中增加专业桌面软件截图是提升 GUI 智能体泛化能力的关键

## 评分

| 维度 | 分数 | 理由 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首个桌面离线 GUI 基准，三级任务设计新颖 |
| 技术深度 | ⭐⭐⭐ | 基准构建工作，技术深度适中 |
| 实验完整度 | ⭐⭐⭐⭐ | 覆盖多个 SOTA 模型，多维度错误分析详尽 |
| 写作质量 | ⭐⭐⭐⭐ | 结构清晰，问题定义明确，标注流程透明 |
| 实用性 | ⭐⭐⭐⭐⭐ | MIT 开源 + HuggingFace 发布，社区直接可用 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](../../CVPR2026/object_detection/mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [\[ICCV 2025\] VisRL: Intention-Driven Visual Perception via Reinforced Reasoning](../../ICCV2025/object_detection/visrl_intention-driven_visual_perception_via_reinforced_reasoning.md)
- [\[ICML 2025\] FG-CLIP: Fine-Grained Visual and Textual Alignment](fg-clip_fine-grained_visual_and_textual_alignment.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](../../ICCV2025/object_detection/visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[ICML 2025\] Self-Organizing Visual Prototypes for Non-Parametric Representation Learning](self-organizing_visual_prototypes_for_non-parametric_representation_learning.md)

</div>

<!-- RELATED:END -->
