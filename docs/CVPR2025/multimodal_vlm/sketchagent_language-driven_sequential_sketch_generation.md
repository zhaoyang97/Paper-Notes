---
title: >-
  [论文解读] SketchAgent: Language-Driven Sequential Sketch Generation
description: >-
  [CVPR 2025][多模态VLM][草图生成] SketchAgent 无需任何训练或微调，通过为预训练多模态 LLM 设计网格画布坐标系统 + 上下文示例 + 贝塞尔曲线拟合的后处理流水线，使模型以逐笔画方式生成语义丰富、接近人类风格的草图，Top-1 识别率达人类水平的 85%，并支持交互式协作绘图和对话编辑。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "草图生成"
  - "语言驱动"
  - "零训练"
  - "贝塞尔曲线"
  - "逐笔画"
  - "人机协作"
---

# SketchAgent: Language-Driven Sequential Sketch Generation

**会议**: CVPR 2025  
**arXiv**: [2411.17673](https://arxiv.org/abs/2411.17673)  
**代码**: 待确认  
**领域**: 多模态VLM / Agent  
**关键词**: 草图生成、语言驱动、零训练、贝塞尔曲线、逐笔画、人机协作

## 一句话总结
SketchAgent 无需任何训练或微调，通过为预训练多模态 LLM 设计网格画布坐标系统 + 上下文示例 + 贝塞尔曲线拟合的后处理流水线，使模型以逐笔画方式生成语义丰富、接近人类风格的草图，Top-1 识别率达人类水平的 85%，并支持交互式协作绘图和对话编辑。

## 研究背景与动机

**领域现状**：草图生成研究主要有两条路线——(1) 基于 RNN 的方法（如 SketchRNN）学习编码矢量笔画序列，但泛化能力差、缺乏语义控制；(2) 基于扩散模型的优化方法（如 SVGDreamer），效果好但耗时长（~1.6 小时/张），且输出更像设计图而非手绘草图。现有方法普遍缺乏人类绘画的渐进式、语义性特征。

**现有痛点**：(1) 传统方法需要在草图数据集上训练/微调，限制了泛化到新概念的能力；(2) 直接让 LLM 输出 SVG 代码会产生机械、僵硬的线条，缺乏手绘感；(3) LLM 的空间推理能力差，直接用像素坐标容易出错；(4) 现有方法不支持人类与 AI 的协作绘图——无法中途暂停让用户添加笔画再继续。

**核心矛盾**：多模态 LLM 拥有丰富的视觉和语义先验知识，理论上能"理解"物体的视觉结构，但缺乏将语义理解转化为精确空间坐标的能力。

**本文目标** 如何在不训练的前提下，利用预训练 LLM 的先验知识生成人类风格的逐笔画草图，并支持交互式编辑？

**切入角度**：不改模型，改接口——设计一套简洁的"绘画语言"（网格坐标 + 笔画格式），通过上下文学习让 LLM 理解如何画草图，再用贝塞尔曲线拟合把粗糙坐标变成流畅线条。

**核心 idea**：用网格画布弥补 LLM 空间推理弱点 + 上下文学习教会 LLM 绘画格式 + 贝塞尔曲线后处理生成平滑手绘风格的草图，全程零训练。

## 方法详解

### 整体框架
Pipeline: 用户文本提示 → LLM 在 `<thinking>` 标签中规划绘图策略 → 按笔画输出网格坐标序列 → 贝塞尔曲线拟合 → 渲染为矢量草图。支持三种模式：独立生成、协作绘图（用户可在中途加笔画）、对话式编辑（将已画内容反馈给模型并用语言指令修改）。

### 关键设计

1. **网格画布坐标系统（The Canvas）**:

    - 功能：将 LLM 的坐标输出从像素空间转换到低分辨率网格空间，大幅降低空间推理难度
    - 核心思路：定义 50×50 的编号网格，每个单元格用坐标标识（如 x2y8）。LLM 输出笔画时只需给出网格坐标序列，格式为 `<points>x1y1, x15y20, ...</points>`。每笔画带有语义 ID 标签用于分析
    - 设计动机：直接用像素坐标时 LLM 的空间推理会崩溃。50×50 的网格足够表达草图的粗略结构，同时让 LLM 更容易理解和生成。虽然牺牲了精度，但草图本身就是抽象的，正好匹配

2. **贝塞尔曲线拟合（Bézier Curve Smoothing）**:

    - 功能：将 LLM 输出的粗糙网格坐标点变成平滑自然的手绘线条
    - 核心思路：每条笔画的坐标序列拟合为三次贝塞尔曲线 $B(t) = (1-t)^3P_0 + 3(1-t)^2tP_1 + 3(1-t)t^2P_2 + t^3P_3$。LLM 同时输出每个点对应的参数 t 值，用最小二乘法 $P = \text{argmin}_P ||AP - B||$ 求解控制点。拟合误差大时递归拆分曲线段
    - 设计动机：网格坐标直接连线会产生锯齿、机械的折线；贝塞尔拟合让线条获得手绘的流畅感。让 LLM 输出 t 值是巧妙的设计——这等于让模型表达"这个点在笔画行程的哪个位置"，提供了比纯坐标更丰富的信息

3. **上下文学习 + Chain-of-Thought 绘图规划**:

    - 功能：无需训练即让 LLM 学会绘图格式和策略
    - 核心思路：System prompt 介绍网格画布和绘画语言规则；User prompt 包含一个完整的房屋草图示例（关键）+ 若干单笔画基元示例。模型在 `<thinking>` 标签中先描述绘图策略——分解物体组件、规划笔画顺序和各部分位置，然后逐笔画输出坐标
    - 设计动机：消融实验证明完整示例（Modified ICL）对性能影响最大（去掉后 Top-1 从 0.23 降到 0.07），说明 LLM 需要"看到一个完整的画完的例子"才能理解任务。CoT 规划则帮助模型分解复杂物体的组件结构

### 损失函数 / 训练策略
**无需训练或微调**——模型完全使用预训练的 Claude 3.5-Sonnet，只有贝塞尔曲线拟合的最小二乘优化（闭式解）。生成速度约 20 秒/张。

## 实验关键数据

### 主实验

| 方法 | Top-1 识别率 | Top-5 识别率 |
|--------|------|------|
| GPT-4o | 0.15 ± 0.04 | 0.30 ± 0.06 |
| Claude-3.5-Sonnet (SVG直出) | 0.23 ± 0.05 | 0.44 ± 0.03 |
| **SketchAgent** | **0.23 ± 0.04** | **0.43 ± 0.06** |
| Human (QuickDraw) | 0.27 ± 0.07 | 0.49 ± 0.06 |

2AFC 用户研究：74.9% 的人认为 SketchAgent 的草图比直接 prompting 更像人画的。与真人草图对比时，SketchAgent 被选为"更像人画的"概率为 45.3%（非常接近 50/50）。

### 消融实验

| 配置 | Top-1 | Top-5 |
|------|---------|---------|
| **SketchAgent (完整)** | **0.23** | **0.43** |
| w/o System Prompt | 0.20 | 0.42 |
| w/o CoT | 0.14 | 0.29 |
| Modified ICL (去掉完整示例) | 0.07 | 0.16 |

### 关键发现
- **完整示例是最关键因素**：去掉后 Top-1 骤降 70%（0.23→0.07），说明 LLM 需要 "示范" 才能理解绘画任务的输出格式和策略
- **CoT 规划贡献显著**：去掉后 Top-1 降 39%（0.23→0.14），证明先规划再画比直接画好得多
- **SVG 直出 CLIP 分数可比但视觉风格差异大**：SVG 方法产生机械、设计感的图形，SketchAgent 产生手绘、渐进式的草图，两者识别率相当但人类偏好显著不同
- **协作绘图有效**：30 名参与者的用户研究中，协作草图的识别率接近单人绘制，且双方都有贡献
- **对话编辑准确率 92%**：空间关系指令（"左边"）94% 准确，语义关系推断（"帽子放在动物头上"）88% 准确

## 亮点与洞察
- **零训练范式的优雅性**：不训练、不微调、不优化，纯靠 prompt 工程 + 后处理就达到接近人类的草图生成能力。这证明大模型的先验知识远比我们利用的多，关键在于设计正确的接口
- **网格画布是一个通用的空间推理增强方案**：50×50 网格把连续空间离散化为 LLM 更容易处理的形式。这个思路可以推广到任何需要 LLM 做空间推理的任务（布局设计、地图导航等）
- **人机协作草图的可能性**：通过 stopping token 和坐标转换机制，实现了 AI 和人类交替画笔画的流畅协作，这种交互模式在创意工具中有巨大应用潜力

## 局限与展望
- 受制于 LLM 的视觉先验——能用语言描述复杂概念（如独角兽的角）但画不好，语义理解和空间表达之间存在 gap
- 人物画和文字/数字画效果差，LLM 的人体/字符空间结构先验不足
- 50×50 网格抽象度高，细节表现力有限（但这某种程度上也是草图的特点）
- 依赖特定模型（Claude 3.5-Sonnet），其他 LLM 效果可能差很多
- 生成速度 ~20 秒，比 SketchRNN (~4秒) 慢但比优化方法 (~1.6小时) 快很多

## 相关工作与启发
- **vs SketchRNN**: SketchRNN 需要训练且泛化差；SketchAgent 零训练且能画任何概念，但精度较低
- **vs SVGDreamer**: SVGDreamer 通过优化生成高质量 SVG 但耗时 1.6 小时；SketchAgent 20 秒即出、风格更像手绘
- **vs 直接 SVG prompting**: 识别率相当但 SketchAgent 的笔画更自然、人更偏好。关键区别在于贝塞尔拟合和逐笔画生成策略

## 评分
- 新颖性: ⭐⭐⭐⭐ 零训练方案非常新颖，网格+贝塞尔+ICL 的组合设计巧妙
- 实验充分度: ⭐⭐⭐⭐ 识别率、用户研究、消融、协作/编辑都有测试
- 写作质量: ⭐⭐⭐⭐⭐ 图文并茂，方法动机和设计选择描述清晰
- 价值: ⭐⭐⭐⭐ 展示了 LLM 在视觉创意任务上的潜力，交互设计有启发性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SldprtNet: A Large-Scale Multimodal Dataset for CAD Generation in Language-Driven 3D Design](sldprtnet_a_large-scale_multimodal_dataset_for_cad_generation_in_language-driven.md)
- [\[AAAI 2026\] PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography](../../AAAI2026/multimodal_vlm/pet2rep_towards_vision-language_model-drived_automated_radiology_report_generati.md)
- [\[AAAI 2026\] O3SLM: Open Weight, Open Data, and Open Vocabulary Sketch-Language Model](../../AAAI2026/multimodal_vlm/o3slm_open_weight_open_data_and_open_vocabulary_sketch-language_model.md)
- [\[CVPR 2025\] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)
- [\[CVPR 2025\] Global-Local Tree Search in VLMs for 3D Indoor Scene Generation](global-local_tree_search_in_vlms_for_3d_indoor_scene_generation.md)

</div>

<!-- RELATED:END -->
