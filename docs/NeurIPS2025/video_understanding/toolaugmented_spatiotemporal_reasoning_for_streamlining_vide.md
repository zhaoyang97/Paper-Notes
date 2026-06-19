---
title: >-
  [论文解读] Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task
description: >-
  [NeurIPS2025][视频理解][VideoQA] 提出了包含 22 个工具的视频工具包和 STAR（Spatiotemporal Reasoning）框架，通过时间-空间工具交替调度策略渐进式定位 3D RoI，在 VideoMME 上将 GPT-4o 提升 8.2%，同时大幅减少处理帧数和计算开销。
tags:
  - "NeurIPS2025"
  - "视频理解"
  - "VideoQA"
  - "tool-augmented LLM"
  - "spatiotemporal reasoning"
  - "agentic framework"
  - "video toolkit"
---

# Tool-Augmented Spatiotemporal Reasoning for Streamlining Video Question Answering Task

**会议**: NeurIPS2025  
**arXiv**: [2512.10359](https://arxiv.org/abs/2512.10359)  
**代码**: [GitHub](https://github.com/fansunqi/VideoTool)  
**领域**: 视频理解 / 多模态推理  
**关键词**: [VideoQA, tool-augmented LLM, spatiotemporal reasoning, agentic framework, video toolkit]

## 一句话总结

提出了包含 22 个工具的视频工具包和 STAR（Spatiotemporal Reasoning）框架，通过时间-空间工具交替调度策略渐进式定位 3D RoI，在 VideoMME 上将 GPT-4o 提升 8.2%，同时大幅减少处理帧数和计算开销。

## 研究背景与动机

**领域现状**：视频问答（VideoQA）是评估模型动态场景理解能力的关键任务。当前方法分为两类：Video-LLM（如 Qwen-VL）直接处理大量帧，计算冗余；工具增强 LLM（如 DoraemonGPT）引入外部工具辅助推理。

**现有痛点**：现有工具增强方法存在三个根本缺陷：(1) 工具单维度——要么只关注时间要么只关注空间，无法同时建模帧内空间关系和帧间时间因果；(2) 工具数量与多样性不平衡——简单堆砌工具导致 LLM 调用混乱；(3) 工具调度策略不足——缺乏有效调度机制导致工具链"走捷径"（Toolchain Shortcut），即 LLM 跳过逐步推理直接调用通用工具回答。

**核心矛盾**：需要同时在时间和空间维度上渐进细化，但无约束的工具调度会导致 LLM 取捷径。

**本文目标**：构建一套全面的视频工具包并设计有效的时空工具交替调度策略，解决工具链捷径问题。

**切入角度**：受链式思维（CoT）启发，将视频理解分解为时间定位和空间分析的交替迭代。

**核心 idea**：时间工具和空间工具交替调用，渐进式缩小时空搜索范围定位 3D RoI，如同视觉推理版的 System 2 思维。

## 方法详解

### 整体框架

STAR 是一个无需训练的、可扩展的 Agent 推理框架。包含三类工具集（时间工具集 $T_t$、空间工具集 $T_s$、通用工具集 $T_g$）和一个核心 LLM Planner。初始化时稀疏均匀采样帧填入可见帧字典 $V$；然后交替调用时间和空间工具，时间工具负责选择/缩减帧索引，空间工具负责处理指定帧并更新 $V$ 中的信息；LLM Planner 每步判断信息是否充分，不足则继续调工具，最后必要时调用通用工具生成答案。

### 关键设计

1. **22 工具视频工具包**:

    - 功能：构建涵盖时间、空间、通用三个维度的全面工具集，包括 Frame Selector（基于 LLM 的帧选择）、Temporal Grounding（时间定位）、Object Detection（YOLO/Grounding DINO）、Patch Zoomer（区域放大）、OCR、Image Captioner、Person ReID 等 22 个工具
    - 核心思路：所有工具采用标准化 Tool Card 封装，即插即用。空间工具支持三种 bbox 利用方式：文本化描述、区域放大、Set-of-Mark 标注。时间工具支持帧级和片段级操作（单帧选择、视频剪辑、连续片段提取）
    - 设计动机：视频处理任务天然分解为时间和空间维度，工具需要在两个维度上都提供细粒度能力；通过 Tool Card 标准化接口保证可扩展性

2. **时空交替调度策略（STAR 算法）**:

    - 功能：约束工具链使时间和空间工具交替调用，通用工具仅作最后兜底
    - 核心思路：算法维护可见帧字典 $V$（键为帧索引，值为各工具收集的信息）。第一步自动选择时间或空间工具；之后奇偶步交替使用另一维度工具。每步 LLM Planner 生成 Thought → 选择工具 → 执行 → Observation → 更新 $V$。如果时间工具缩小了时间范围，空间工具可以在更少帧上做精细分析；空间分析的结果又反馈影响后续时间工具的选择，形成渐进定位 3D RoI 的闭环
    - 设计动机：解决 Toolchain Shortcut 问题——无约束时 LLM 倾向于直接调用 VLM 一步回答，绕过多步推理。交替约束强制 LLM 进行时空渐进推理，类似于 CoT 中的 System 2 模式

### 损失函数 / 训练策略

STAR 是完全无训练的框架，所有工具即插即用。STAR 版本使用 GPT-4o 作为 Planner，搭配最大 3B 参数的开源模型工具（QwenVL-2.5-3B 等）；STAR-mini 版本使用 GPT-3.5-turbo 作为 Planner，工具不超过 500M 参数（BLIP 等），可在个人电脑上运行。

## 实验关键数据

### 主实验

| 方法 | 参数量 | 帧数↓ | 运行时间↓ | VideoMME↑ | LongVideoBench↑ |
|------|--------|-------|----------|-----------|----------------|
| GPT-4o (32帧) | - | 32 | <30s | 61.8 | 52.6 |
| GPT-4o (1fps/384) | - | 384 | >10min | 71.9 | 66.7 |
| Qwen2.5-VL-7B | 7B | - | - | 65.1 | 53.7 |
| InternVL3-8B | 8B | 64 | <30s | 66.3 | 48.9 |
| Qwen2-VL-72B | 72B | 2fps | 6-8min | 71.2 | - |
| **STAR (ours)** | - | **30.2** | **15.8s** | **70.0 (+8.2)** | **57.2 (+4.6)** |

NExT-QA 测试集（STAR vs 最优基线 AKeyS 78.1%）：

| 方法 | 帧数↓ | 因果↑ | 时间↑ | 描述↑ | 总计↑ |
|------|-------|-------|-------|-------|-------|
| AKeyS (GPT-4o) | 7.6 | 72.9 | 79.0 | 86.1 | 78.1 |
| **STAR (GPT-4o)** | **7.2** | **81.1** | **81.5** | **86.3** | **82.1 (+1.2)** |

### 消融实验

| 工具链策略 | 准确率↑ | 帧数↓ | 链长↑ | 工具种类↑ |
|-----------|--------|-------|-------|----------|
| 无约束 | 61.2 | 112.6 | 2.9 | 1.3 |
| Prompting | 60.4 | 98.7 | 3.6 | 1.9 |
| In-Context Learning | 63.2 | 50.1 | 5.4 | 3.2 |
| 时空解耦 | 68.6 | 40.6 | 5.6 | 3.4 |
| **STAR（交替）** | **70.0** | **30.2** | **8.7** | **6.3** |

### 关键发现

- 时空交替比时空解耦提升 1.4%，同时减少 10.4 帧——时空之间的信息反馈至关重要
- 无约束策略导致工具链极短（2.9 步），仅用 1.3 种工具，帧数却高达 112.6——典型的 Toolchain Shortcut
- STAR 以约 30 帧 / 15 秒的开销接近 GPT-4o 使用 384 帧 / 10 分钟以上的性能

## 亮点与洞察

- **Toolchain Shortcut 概念**：首次定义并分析了工具链捷径现象，揭示了无约束 LLM Agent 的典型失败模式
- **效率-精度帕累托最优**：以 30 帧达到 70% VideoMME，接近 72B 模型用数千帧的性能
- **可扩展架构**：22 个工具全部即插即用，新工具可通过 Tool Card 标准化接入
- **EgoSchema 可扩展性**：随帧数增加 STAR 的性能持续增长，展现强 scaling 特性

## 局限与展望

- 仍依赖 GPT-4o API 作为 Planner，带来成本和延迟，替换为开源轻量模型是重要方向
- 当前仅聚焦视频内容，未整合字幕和音频信息，这是 Gemini 等模型领先的关键因素
- 工具调用次数与 API 成本线性相关，长视频的多轮交互开销可能较高
- 工具输出质量受底层模型（YOLO、BLIP 等）限制，存在误差传播风险

## 相关工作与启发

- **DoraemonGPT**：使用 text-to-SQL 查询工具输出数据库，但 SQL 查询经常失败；STAR 通过可见帧字典避免了这一问题
- **AKeyS / T***：LLM 驱动的帧选择方法，本文在此基础上增加了空间维度的工具支持
- **ReAct 框架**：STAR 基于 ReAct 的 Thought-Action-Observation 循环，但加入了时空交替约束
- **启发**：工具增强 Agent 的关键不仅是工具数量，更是调度策略的设计——适当约束反而能释放更强的推理能力

## 评分

- 新颖性: ⭐⭐⭐⭐ 时空交替调度策略和 Toolchain Shortcut 概念有洞察力
- 实验充分度: ⭐⭐⭐⭐ 四个 benchmark 全面评测，消融研究详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表直观
- 价值: ⭐⭐⭐⭐ 为视频 Agent 系统设计提供了可复现的框架和设计原则

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[NeurIPS 2025\] EgoGazeVQA: Egocentric Gaze-Guided Video Question Answering Benchmark](egogazevqa_egocentric_gaze_guided_video_question_answering.md)
- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[CVPR 2025\] QA-TIGER: Question-Aware Gaussian Experts for Audio-Visual Question Answering](../../CVPR2025/video_understanding/question-aware_gaussian_experts_for_audio-visual_question_answering.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](../../CVPR2025/video_understanding/egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)

</div>

<!-- RELATED:END -->
