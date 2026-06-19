---
title: >-
  [论文解读] UVLM: Benchmarking Video Language Model for Underwater World Understanding
description: >-
  [AAAI 2026][视频理解][underwater video] 构建首个水下视频语言理解基准 UVLM（2109 段视频、419 类海洋生物、20 种子任务、~4 万 video-text pairs），通过 human-AI 协同标注注入海洋领域知识，在 UVLM 上微调后 7B VidLM 可达到接近 GPT-4o 的性能（73.04 vs 77.95 Overall）。
tags:
  - "AAAI 2026"
  - "视频理解"
  - "underwater video"
  - "VidLM"
  - "benchmark"
  - "marine biology"
  - "fine-grained recognition"
  - "human-AI annotation"
---

# UVLM: Benchmarking Video Language Model for Underwater World Understanding

**会议**: AAAI 2026  
**arXiv**: [2507.02373](https://arxiv.org/abs/2507.02373)  
**代码**: [GitHub](https://github.com/Cecilia-xue/UVLM-Benchmark)  
**领域**: 视频语言理解 / 水下视觉  
**关键词**: underwater video, VidLM, benchmark, marine biology, fine-grained recognition, human-AI annotation

## 一句话总结

构建首个水下视频语言理解基准 UVLM（2109 段视频、419 类海洋生物、20 种子任务、~4 万 video-text pairs），通过 human-AI 协同标注注入海洋领域知识，在 UVLM 上微调后 7B VidLM 可达到接近 GPT-4o 的性能（73.04 vs 77.95 Overall）。

## 研究背景与动机

**领域现状**：视频语言模型（VidLM）在视频描述、时序定位、视觉问答等任务上取得显著进展，但现有工作主要聚焦地面场景（人类活动、运动、日常生活），水下环境几乎未被覆盖。

**现有痛点**：(1) **视觉特征退化**——水下环境存在光衰减、波长相关色彩失真、浑浊度波动等问题，标准 VidLM 难以有效处理低质量视觉线索；(2) **领域知识缺失**——水下内容需要分类学鉴定、形态特征、行为状态、生态关系等专业生态学知识，模型训练以常见物体和人类活动为主，存在巨大知识鸿沟；(3) **数据资源稀缺**——现有水下数据集主要针对目标跟踪（WebUOT）、实例分割（UIIS）等单一视觉任务，缺乏多模态视频语言数据集。

**核心矛盾**：水下环境具有巨大科学价值（海洋生物多样性监测、生态健康评估）和工程应用潜力（AUV、海上基础设施检测），但现有 VidLM 基准无一面向水下场景。

**本文目标** 为水下视频语言理解构建首个专业化基准数据集，注入海洋领域知识，并验证微调后的 VidLM 在水下场景中的表现。

**切入角度**：通过 human-AI 协同标注（人工逐帧标注 + GPT-4o 辅助生成 + 人工修正），构建覆盖生物和环境两大维度的 20 种子任务基准。

**核心 idea**：用人工逐帧标注的细粒度分类学信息引导 GPT-4o 生成多样化 video-text pairs，再经专家三轮审核确保质量，从而构建首个大规模水下视频语言基准。

## 方法详解

### 整体框架

视频采集（网络爬取 + WebUOT 重标注）→ 人工逐帧标注（边界框 + 五界分类法分类学信息）→ GPT-4o 辅助生成 Q&A → 人工两轮修正 → 最终数据集（2109 视频、0.86M 帧、419 类、~40K video-text pairs）→ 评估 8 种指标。

### 关键设计

1. **水下特异性视频采集与质量控制**
    - 双路径采集策略：(a) 从 YouTube/Bilibili 爬取约 400 段视频，覆盖海洋/湖泊/河流/鱼缸等环境，含水面波纹、浑浊水体、光散射等典型退化；(b) 从 WebUOT 重标注，包括数据清洗（去除字幕水印）、场景清洗（排除非自然水下环境）、SAM+LaMa 去除小面积水印
    - 设计动机：确保数据覆盖水下环境的独特挑战性，而非仅收集清晰水族馆视频

2. **三阶段细粒度分类学标注**
    - 第一阶段：4 名具有海洋生物学知识的标注员独立标注物种级鉴定和详细分类学分类（界、门、纲、目等，依据 Whittaker 五界系统）
    - 第二阶段：标注员配对交叉验证，分歧由第三方标注员多数共识解决
    - 第三阶段：高级海洋生物学专家审查验证标注，疑问案例由 5 名专家集体讨论确定最终分类
    - 设计动机：细粒度分类学信息作为先验信息注入 GPT-4o 文本生成过程，确保生态学准确性

3. **结构化 Human-AI 协同标注管道**
    - 从海洋生物学关键课题出发设计 prompt：生物维度（静态：物种鉴定、形态属性；动态：行为分析、运动模式）+ 环境维度（静态：基底类型、珊瑚结构；动态：光照变化、能见度波动）
    - 每段视频生成 16-20 个 video-text pairs（含选择题和开放式问答）
    - 两轮人工修正：第一轮一般审核员检测信息冲突，第二轮高级专家深度编辑确保事实精确性
    - 设计动机：在保证数据量的同时，通过多轮人工审核确保生态学知识的可靠性

### 评估指标体系

2 类客观指标 + 5 类 LLM-based 判断指标：
- 客观指标：Multiple Choice Accuracy (MCA)、Fine-grained Taxonomic Classification (FGC)
- LLM 判断指标（GPT-4o 作为评估 backbone）：Semantic Accuracy (SA)、Detail Completeness (DC)、Visual Perception Accuracy (VPA)、Environmental Description Accuracy (EDA)、Species Behavior Matching (SBM)

## 实验关键数据

### 闭源模型与大模型基线

| 模型 | MCA | FGC | SA | DC | VPA | EDA | SBM | Overall |
|------|-----|-----|-----|-----|-----|-----|-----|---------|
| GPT-4o | 77.72 | 81.47 | 77.67 | 73.40 | 78.23 | 80.07 | 79.73 | 77.95 |
| Claude3.7-Sonnet | 76.61 | 82.64 | 73.35 | 73.58 | 74.10 | 79.71 | 76.35 | 76.09 |
| Gemini2.5-Flash | 78.22 | 86.27 | 72.43 | 73.34 | 70.53 | 78.32 | 74.92 | 75.00 |
| Qwen2.5VL-72B | 75.97 | 80.57 | 74.22 | 71.94 | 74.85 | 78.45 | 77.40 | 75.49 |

### 开源模型微调前后对比

| 模型 | 微调前 Overall | 微调后 Overall | 提升 |
|------|---------------|---------------|------|
| InternVL2.5-1B | 46.73 | 59.14 | +12.41 |
| VideoLLaMA3-2B | 58.39 | 66.67 | +8.28 |
| Qwen2.5VL-2B | 52.97 | 58.44 | +5.47 |
| InternVL2.5-8B | 60.15 | 69.45 | +9.30 |
| VideoLLaMA3-7B | 62.70 | 73.04 | +10.34 |
| Qwen2.5VL-7B | 63.57 | 68.08 | +4.51 |

### 关键发现

- VideoLLaMA3-7B 微调后 Overall 73.04，仅落后 Qwen2.5VL-72B 2.45 分，接近闭源 Gemini 75.00
- FGC（细粒度分类学）是唯一"大小模型差距难以通过微调弥合"的任务——需要复杂生物学领域知识，小模型容量不足
- UVLM 微调还能提升通用基准（VideoMME、Perception Test）表现

## 亮点与洞察

- 首个水下视频语言理解基准，填补了 VidLM 在水下领域的空白
- 三阶段分类学标注流程体现了极高的标注严谨性——四人独立标注 + 配对验证 + 专家终审 + 五人集体讨论
- 从 GPT-4o 蒸馏知识到小模型的范式有效：7B 模型在 UVLM 上微调后接近 GPT-4o，同时大幅降低了部署成本
- FGC 任务揭示了一个有趣发现：专业领域知识（分类学鉴定）无法通过简单微调弥合大小模型差距，指向了模型容量与领域知识的根本矛盾

## 局限与展望

- 数据集规模（2109 段视频）相对有限，与地面视频基准（HowTo100M、Kinetics）差距甚大
- 生物类别分布呈长尾分布，稀有物种样本不足可能影响微调效果
- 评估依赖 GPT-4o 作为 judge（5 个 LLM-based 指标），引入 LLM-as-judge 的已知偏差
- 仅测试了 3 个开源VidLM 家族，缺少多样性
- 未探索如何利用 UVLM 做水下视觉任务（如跟踪、分割）的跨任务迁移

## 相关工作与启发

- **vs WebUOT**：WebUOT 是最接近的水下视频数据集（1500 序列、1M 帧、408 类），但仅支持单目标跟踪任务，无语言理解维度。UVLM 在其基础上进行了重标注和扩展
- **vs MarineInst**：MarineInst 支持图像分割和描述，但缺乏视频时序理解能力
- **vs Video-MME / Perception Test**：通用视频基准，但完全不涉及水下场景。UVLM 微调兼顾提升水下和通用性能
- 标注方法论上，human-AI 协同标注 + 领域专家终审的范式对专业领域数据集构建有通用启发价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个水下视频语言理解基准，选题具有明确的差异化价值
- 实验充分度: ⭐⭐⭐ 测试了多个闭源和开源模型，但缺少更多消融实验（如标注策略、数据量影响）
- 写作质量: ⭐⭐⭐ 结构清晰，标注流程描述详尽，但整体行文偏冗长
- 价值: ⭐⭐⭐⭐ 填补了重要空白，数据集和代码开源，对水下视觉社区有直接推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] 4D-Bench: Benchmarking Multi-modal Large Language Models for 4D Object Understanding](../../ICCV2025/video_understanding/4dbench_benchmarking_multimodal_large_language_models_for_4d.md)
- [\[CVPR 2026\] ELV-Halluc: Benchmarking Semantic Aggregation Hallucinations in Video Understanding](../../CVPR2026/video_understanding/elv-halluc_benchmarking_semantic_aggregation_hallucinations_in_video_understandi.md)
- [\[CVPR 2026\] InternVideo-Next: Towards World-Understanding Video Models](../../CVPR2026/video_understanding/internvideo-next_towards_world-understanding_video_models.md)
- [\[CVPR 2026\] Enhancing Video Vision Language Model with Hippocampal Sensing](../../CVPR2026/video_understanding/enhancing_video_vision_language_model_with_hippocampal_sensing.md)
- [\[CVPR 2026\] Towards Streaming Referring Video Segmentation via Large Language Model](../../CVPR2026/video_understanding/towards_streaming_referring_video_segmentation_via_large_language_model.md)

</div>

<!-- RELATED:END -->
