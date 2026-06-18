---
title: >-
  [论文解读] Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence
description: >-
  [CVPR 2025][多模态VLM][视频推理] 提出 VAEX-Bench 基准，首次系统评估 MLLM 的"抽象时空推理"能力——不是从单帧提取信息，而是需要跨房间/跨时间整合观察来推断全局空间布局、跨场景计数等，发现所有 SOTA 模型（包括 GPT-5.2、Gemini-3 Pro）在抽象推理上表现远低于人类。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "视频推理"
  - "时空推理"
  - "抽象推理"
  - "自我中心视频"
  - "benchmark"
---

# Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence

**会议**: CVPR 2025  
**arXiv**: [2603.13091](https://arxiv.org/abs/2603.13091)  
**代码**: 即将发布  
**领域**: 多模态VLM  
**关键词**: 视频推理, 时空推理, 抽象推理, 自我中心视频, benchmark

## 一句话总结

提出 VAEX-Bench 基准，首次系统评估 MLLM 的"抽象时空推理"能力——不是从单帧提取信息，而是需要跨房间/跨时间整合观察来推断全局空间布局、跨场景计数等，发现所有 SOTA 模型（包括 GPT-5.2、Gemini-3 Pro）在抽象推理上表现远低于人类。

## 研究背景与动机

**领域现状**：现有视频时空基准（VSI-Bench、VSTI-Bench）主要测试"提取式推理"——答案可从单帧或局部时空事件中直接提取（如物体出现顺序、相对方向）。

**现有痛点**：提取式推理不能评估模型是否能形成全局一致的空间表示——是否能从碎片化的第一人称观察中重建房屋平面图？能否跨房间计数？能否理解房间之间的全局方位关系？

**核心矛盾**：具身智能需要的"抽象时空推理"能力（整合分散观察、推断隐含空间结构）几乎未被评估。

**本文目标** 构建可控基准来系统评估 MLLM 的抽象 vs 提取式时空推理能力。

**切入角度**：不是在已有视频上标问题（证据固定、难以设计抽象推理问题），而是"先设计问题、再生成环境和视频"——query-conditioned video construction。

**核心 idea**：从提取式推理扩展到抽象式推理的一对一任务对比，用可控合成环境暴露 MLLM 的时空推理瓶颈。

## 方法详解

### 整体框架

VAEX-Bench = 10 个室内场景 × 10 个任务（5 个提取式 + 5 个抽象式）× 3 个问题 = 300 个 query。核心 pipeline：场景规范设计 → 查询构建 → SketchUp 建模 → Enscape 渲染 → 自我中心视频录制 → 人工验证。

### 关键设计

1. **提取→抽象的一对一任务扩展**：

    - Appearance Order → **Memory-Action**：从"物体出现顺序"扩展为"第三个房间能做什么活动"（需长程记忆）
    - Relative Direction → **Map Direction**：从单视角相对方向扩展为"room3 相对 room4 什么方向"（需全局方位建模）
    - Relative Distance → **Map Scale**：从局部距离扩展为给定参考距离后估算房间间距离（需全局度量推理）
    - Route Plan → **Simulation**：从局部导航扩展为"哪个房间在厨房正对面"（需平面布局推理）
    - Object Counting → **Global Counting**：从单房间计数扩展为跨所有房间的全局计数（需消除重复的跨场景聚合）

2. **Query-Conditioned Video Construction**：

    - 先设计问题决定需要什么证据 → 据此构建场景和轨迹 → 确保证据按受控方式在时空中分布
    - 关键约束：temporal cue separation（决定性证据分散在视频不同位置）+ spatial mapping（不能靠局部导航线索解答）
    - 每个视频约需 2-3 周制作

3. **合成环境的可控性**：

    - SketchUp 建模 + Enscape 渲染，室内场景
    - 14 个 MLLM 统一评估：32 帧采样、温度 0.7、Accuracy@5（5 次生成取平均）

## 实验关键数据

### 主实验

| 模型 | 抽象 Avg | 提取 Avg | 记忆 | 地图方向 | 全局计数 |
|------|---------|---------|------|---------|---------|
| Human | 81.7% | 88.0% | 89.3% | 83.3% | 82.7% |
| Gemini-3 Flash | **40.3%** | 50.0% | 60.7% | 34.0% | 31.3% |
| GPT-5.2 | 30.1% | 44.5% | 38.0% | 26.0% | 23.3% |
| Qwen3-VL-32B | 29.9% | 45.5% | 40.0% | 26.0% | 17.3% |
| Qwen3-VL-235B | 26.7% | 49.7% | 43.3% | 16.7% | 13.3% |
| Random | 26.5% | 24.8% | 30.7% | 22.0% | N/A |

### 关键发现

- **抽象 vs 提取的巨大落差**：所有模型在抽象任务上的表现远低于提取任务。最佳模型 Gemini-3 Flash 抽象平均 40.3% vs 提取 50.0%，人类则高达 81.7% vs 88.0%
- **模型排名反转**：Gemini-3 Flash 在抽象任务上显著优于 Gemini-3 Pro（40.3% vs 29.7%），但提取任务上 Pro 更好。说明短程识别能力不能转化为抽象推理
- **规模扩大不等于抽象推理提升**：Qwen3-VL-32B/235B 在抽象任务上并不比 8B 好多少（29.9/26.7% vs 24.5%）
- **全局计数是最大瓶颈**：所有模型在 Global Counting 上远低于人类（13-31% vs 82.7%），说明模型无法消除重复和跨场景聚合
- **MCQ → 自由生成性能下降**：模型依赖选项线索，移除选项后表现进一步恶化
- **人类在 Map Scale 上也表现不佳（60%）**：距离度量推理对人和模型都是困难的

## 亮点与洞察

- **"提取 vs 抽象"的二分法非常清晰有力**：一对一的任务对比让人直观看到模型在哪里失败。这不是简单的"换个更难的问题"，而是在语义意图相同的条件下测试不同层次的认知能力。
- **Query-conditioned pipeline 是关键创新**：先设计问题再生成视频，确保每个问题都有精确可控的证据分布，比在已有视频上标注问题质量高得多。
- **合成数据但评估真实能力**：虽然场景是合成的，但评估的推理能力（全局空间建模、跨场景聚合）是具身智能的核心需求。

## 局限与展望

- **规模小**：只有 10 个场景、300 个 query，统计显著性有限
- **合成 vs 真实鸿沟**：SketchUp/Enscape 渲染的场景与真实环境有视觉差距
- **制作成本极高**：每个视频 2-3 周，难以大规模扩展
- **32 帧均匀采样可能不够**：对于较长视频（~37s），32 帧可能遗漏关键证据
- **只测试了室内场景**：户外、开放世界的抽象时空推理未涉及

## 相关工作与启发

- **vs VSI-Bench**：VSI-Bench 专注 3D 视觉空间智能但仍是提取式。VAEX-Bench 系统性引入抽象推理维度。
- **vs 文本抽象推理（HotpotQA等）**：文本领域的多跳推理已被广泛研究，但视频时空的抽象推理远更困难——需要处理 partial observability 和空间推理。
- **启发**：models 在全局计数上的失败暴露了"entity persistence under partial observability"这一根本问题——模型不知道在 room 1 和 room 3 看到的椅子是否是同一把。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首次系统定义并评估视频"抽象时空推理"，任务分类体系优雅
- 实验充分度: ⭐⭐⭐⭐ 14个模型全面对比，包含人类基线；但数据集规模较小
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，例子直观，分类体系系统
- 价值: ⭐⭐⭐⭐⭐ 对具身智能的视频理解研究有重要指导意义——揭示了当前模型的根本局限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[CVPR 2025\] Reasoning to Attend: Try to Understand How \<SEG\> Token Works](reasoning_to_attend_try_to_understand_how_seg_token_works.md)
- [\[CVPR 2026\] Select Less, Reason More: Prioritizing Evidence Purity for Video Reasoning](../../CVPR2026/multimodal_vlm/select_less_reason_more_prioritizing_evidence_purity_for_video_reasoning.md)
- [\[CVPR 2025\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[CVPR 2025\] Thinking in Dynamics: How Multimodal Large Language Models Perceive, Track, and Reason Dynamics in Physical 4D World](thinking_in_dynamics_how_multimodal_large_language_models_perceive_track_and_rea.md)

</div>

<!-- RELATED:END -->
