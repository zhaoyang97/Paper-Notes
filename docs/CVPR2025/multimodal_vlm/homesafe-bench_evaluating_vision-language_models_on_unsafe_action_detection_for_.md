---
title: >-
  [论文解读] HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios
description: >-
  [CVPR 2025][多模态VLM][家庭安全] HomeSafe-Bench是首个评估VLM在家庭场景中不安全行为检测的benchmark（438个案例覆盖6个功能区域），并提出HD-Guard层次化流式架构协调轻量FastBrain和大规模SlowBrain实现实时安全监控。 领域现状：家庭机器人快速发展…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "家庭安全"
  - "不安全行为检测"
  - "VLM评估"
  - "具身智能"
  - "双脑架构"
---

# HomeSafe-Bench: Evaluating Vision-Language Models on Unsafe Action Detection for Embodied Agents in Household Scenarios

**会议**: CVPR 2025  
**arXiv**: [2603.11975](https://arxiv.org/abs/2603.11975)  
**代码**: 有  
**领域**: 多模态VLM / AI安全  
**关键词**: 家庭安全, 不安全行为检测, VLM评估, 具身智能, 双脑架构

## 一句话总结
HomeSafe-Bench是首个评估VLM在家庭场景中不安全行为检测的benchmark（438个案例覆盖6个功能区域），并提出HD-Guard层次化流式架构协调轻量FastBrain和大规模SlowBrain实现实时安全监控。

## 研究背景与动机

**领域现状**：家庭机器人快速发展，但家庭环境引入不可预测的安全风险（如感知延迟、缺乏常识导致危险操作）。现有安全评估多局限于静态图像、文本或通用危害。

**现有痛点**：（1）缺乏动态不安全行为检测的标准化benchmark；（2）家庭场景比工业环境更复杂多变，需要理解上下文才能判断行为是否安全；（3）VLM在安全检测中的能力和瓶颈不清楚。

**核心矛盾**：实时安全监控需要低延迟，但准确的不安全行为检测需要深度多模态推理——两者难以兼顾。

**本文目标**：构建评估benchmark + 设计实时安全监控架构。

**切入角度**：（1）通过物理仿真+视频生成的混合管线构建多样的不安全行为数据集；（2）用双脑架构平衡推理效率和检测精度。

**核心 idea**：FastBrain做高频轻量筛查，SlowBrain做异步深度推理，两者协调实现实时安全。

## 方法详解

### 整体框架
HomeSafe-Bench包含438个不安全案例覆盖厨房、客厅等6个功能区域，带多维度细粒度标注。HD-Guard在推理时用快慢双脑协同：FastBrain连续高频筛查视频帧，发现可疑行为时触发SlowBrain进行深度多模态分析。

### 关键设计

1. **混合数据构建管线**:

    - 功能：生成多样逼真的不安全行为视频
    - 核心思路：物理仿真器生成基础场景和动作，结合先进视频生成模型增强视觉真实度，人工标注不安全类型、严重程度和上下文
    - 设计动机：纯仿真不够真实，纯真实数据难以覆盖足够多的不安全场景

2. **Hierarchical Dual-Brain Guard (HD-Guard)**:

    - 功能：实时安全监控架构
    - 核心思路：FastBrain是轻量模型（如小型ViT），以高频率扫描视频帧，输出每帧的快速安全评分。当评分超过阈值时，异步触发SlowBrain（大型VLM如GPT-4V）进行深度多模态推理，综合视觉、语言和常识知识做出最终判断
    - 设计动机：类比人类的快慢系统（System 1/2）——大多数时间快速直觉判断就够了，只在需要时启动深度推理

3. **多维度细粒度标注**:

    - 功能：支持系统化评估
    - 核心思路：每个案例标注了不安全类型（如碰撞、跌落、火灾）、严重程度、涉及的物体和上下文依赖性。6个功能区域的划分使评估覆盖家庭的各个典型空间
    - 设计动机：粗粒度的"安全/不安全"二分类不足以诊断模型的具体弱点

### 损失函数 / 训练策略
FastBrain可以用少量标注数据微调，SlowBrain使用预训练VLM做zero/few-shot推理。

## 实验关键数据

### 主实验

| 方法 | 检测准确率 | 延迟 | 说明 |
|------|-----------|------|------|
| HD-Guard | 最佳trade-off | 低 | 快慢脑协同 |
| 仅大型VLM | 最高准确率 | 很高 | 不适合实时 |
| 仅轻量模型 | 较低 | 最低 | 漏检严重 |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| FastBrain + SlowBrain | 最佳 | 互补协同 |
| 仅FastBrain | 漏检多 | 缺乏深度推理 |
| 仅SlowBrain | 延迟高 | 无法实时 |
| 不同触发阈值 | 有trade-off | 阈值低→更多SlowBrain调用 |

### 关键发现
- 现有VLM在不安全行为检测上表现远非完美，尤其在需要常识推理的场景（如判断"刀朝向幼儿"是否危险）
- HD-Guard的快慢脑协同显著优于单一模型策略
- 上下文依赖性是关键瓶颈——同一行为在不同上下文中可能安全或不安全

## 亮点与洞察
- **重要的安全评估缺口**：首个系统性评估VLM在家庭不安全行为检测上的能力，对具身AI安全有直接意义
- **快慢脑架构的实用性**：System 1/2的类比直观且有效，这种架构可以迁移到其他需要实时监控+深度分析的场景
- **混合数据构建**：仿真+生成的管线是解决安全数据稀缺的实用方案

## 局限与展望
- 438个案例规模较小，可能不覆盖所有不安全场景
- 视频生成的不安全行为可能与真实行为有分布偏差
- FastBrain的误报率和SlowBrain的调用频率之间的最优平衡需要场景化调优
- 未考虑多人交互场景中的安全问题

## 相关工作与启发
- **vs SafetyBench (文本安全)**：文本安全评估不涉及视觉和物理交互，HomeSafe-Bench更贴近具身场景
- **vs RoboCasa/Habitat**：仿真平台提供环境但不专注安全评估，HomeSafe-Bench填补了这个空白
- **vs System 1/2 架构**：HD-Guard是认知科学双系统理论在AI安全中的具体实现

## 评分
- 新颖性: ⭐⭐⭐⭐ 新benchmark+双脑架构组合新颖
- 实验充分度: ⭐⭐⭐⭐ 多VLM对比+消融完整
- 写作质量: ⭐⭐⭐⭐ 问题动机和方法描述清晰
- 价值: ⭐⭐⭐⭐⭐ 对具身AI安全有重要实际意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)
- [\[CVPR 2025\] Embodied Scene Understanding for Vision Language Models via MetaVQA](embodied_scene_understanding_for_vision_language_models_via_metavqa.md)
- [\[CVPR 2025\] Evaluating Vision-Language Models as Evaluators in Path Planning](evaluating_vision-language_models_as_evaluators_in_path_planning.md)
- [\[CVPR 2025\] Taxonomy-Aware Evaluation of Vision-Language Models](taxonomy-aware_evaluation_of_vision-language_models.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](../../NeurIPS2025/multimodal_vlm/mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)

</div>

<!-- RELATED:END -->
