---
title: >-
  [论文解读] VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM
description: >-
  [CVPR 2025][视频理解][视频区域理解] VideoRefer Suite 从数据集（700K 目标级视频指令数据）、模型（时空目标编码器实现像素级区域理解）和基准（多维度评估）三方面构建完整体系，使 Video LLM 具备对视频中任意目标在任意时刻的感知、推理和检索能力。 领域现状：Video LLM（如 Vi…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频区域理解"
  - "目标级指令数据"
  - "时空目标编码器"
  - "视频LLM"
  - "细粒度理解"
---

# VideoRefer Suite: Advancing Spatial-Temporal Object Understanding with Video LLM

**会议**: CVPR 2025  
**arXiv**: [2501.00599](https://arxiv.org/abs/2501.00599)  
**代码**: 有（Project Page + Code 链接）  
**领域**: 视频理解  
**关键词**: 视频区域理解, 目标级指令数据, 时空目标编码器, 视频LLM, 细粒度理解

## 一句话总结
VideoRefer Suite 从数据集（700K 目标级视频指令数据）、模型（时空目标编码器实现像素级区域理解）和基准（多维度评估）三方面构建完整体系，使 Video LLM 具备对视频中任意目标在任意时刻的感知、推理和检索能力。

## 研究背景与动机

**领域现状**：Video LLM（如 VideoLLaMA2）在整体视频理解上表现出色，但主要关注场景级理解，无法精确聚焦用户指定的特定目标。图像领域的区域理解方法（GPT4RoI、Ferret、Osprey）已较成熟，但视频领域的目标级理解研究仍然有限。

**现有痛点**：(1) 现有视频区域理解方法（如 Artemis）只支持单目标、粗框级特征，无法分析多目标关系和复杂推理；(2) 将边界框坐标直接转为文本提示（如 VTimeLLM）导致区域理解不精确；(3) 缺乏高质量的目标级视频指令数据和全面的评估基准。

**核心矛盾**：视频中的细粒度理解需要同时具备精确的空间定位（像素级 mask）和丰富的时间上下文（跨帧追踪），现有架构和数据都不足以支撑这一需求。

**本文目标**：构建 VideoRefer Suite——一个覆盖数据、模型和基准的完整解决方案，赋能 Video LLM 进行任意目标在任意时刻的细粒度视频理解。

**切入角度**：采用多智能体数据引擎自动构建高质量目标级标注，设计统一的空间-时间目标编码器支持单帧和多帧混合输入，并构建涵盖描述生成和选择题两种形式的综合基准。

**核心 idea**：用像素级 mask 作为统一的区域表示，通过空间 Token 提取器（Mask Pooling）获取目标表示，再通过时间 Token 合并模块自适应聚合跨帧信息，将目标级 token 与场景级 token 交错输入 LLM。

## 方法详解

### 整体框架
基于 VideoLLaMA2.1 构建。输入视频经共享视觉编码器提取帧级特征图 $\mathbf{F}_I$，用户通过 mask 指定感兴趣的目标。空间-时间目标编码器（REnc）处理目标 mask 和特征图，生成目标级 token $\mathcal{T}_R$。场景级 token $\mathcal{T}_Z$、目标级 token $\mathcal{T}_R$ 和文本 token $\mathcal{T}_x$ 交错输入 LLM，实现细粒度视频目标理解。支持单帧和多帧两种模式。

### 关键设计

1. **多智能体数据引擎（VideoRefer-700K）**:

    - 功能：自动构建大规模高质量目标级视频指令数据
    - 核心思路：5 个协作智能体串联工作：(1) Analyzer（Qwen2-7B）从原始字幕提取名词；(2) Annotator（InternVL2-26B）分两次查询生成动态动作描述和静态外观描述；(3) Segmentor（Grounding-DINO + HQ-SAM + SAM2）生成像素级 mask；(4) Reviewer（Qwen2-7B）用 Osprey 区域描述验证 mask-描述的对应关系，仅保留 40% 通过验证的样本；(5) Refiner（GPT-4o）总结精化最终描述
    - 设计动机：利用多个专长不同的专家模型协作，自动化流水线保证了数据规模（700K），严格的 Reviewer 过滤和 GPT-4o 精化保证了数据质量

2. **空间-时间目标编码器（Spatial Token Extractor + Temporal Token Merge）**:

    - 功能：从视频帧中提取精确的目标级表示
    - 核心思路：空间层面，将 2D 二值 mask resize 到特征图尺寸，通过 Mask Pooling 提取区域内所有特征的聚合，经 MLP 得到目标 token $\mathbf{O} \in \mathbb{R}^{1 \times C}$。时间层面，对多帧目标 token $\mathbf{O} \in \mathbb{R}^{k \times C}$ 计算相邻帧间余弦相似度 $\mathbf{S}_{m,m+1}$，选择相似度最高的 $k-u$ 对合并（平均池化），最终保留 $u$ 个代表性 token
    - 设计动机：Mask Pooling 比 RoI Align 更精确（像素级 vs 框级）。时间合并通过合并相似帧的表示消除冗余，同时保留关键的时间变化信息

3. **VideoRefer-Bench 综合基准**:

    - 功能：全面评估 Video LLM 的区域视频理解能力
    - 核心思路：两个子基准——Bench$^D$（描述生成, 400 样本, GPT-4o 从主体一致性/外观/动态/幻觉 4 维度评分 0-5）和 Bench$^Q$（选择题, 1000 题, 涵盖基础/序列/关系/推理/预测 5 类型）。所有问答都必须关联特定视频区域，防止模型不看视频就能回答
    - 设计动机：现有基准要么只有描述任务，要么不要求区域理解，Bench 覆盖多维度确保评估全面性

### 损失函数 / 训练策略
标准自回归语言建模损失 $\mathcal{L} = \sum \log P(y|V, R_1,...,R_n, x)$。两阶段训练：Stage 1 用 500K 短描述预训练目标编码器与 LLM 的对齐；Stage 2 用 125K 详细描述 + 75K QA 微调全部可训练参数。

## 实验关键数据

### 主实验（VideoRefer-Bench$^Q$）

| 方法 | 基础理解 | 序列理解 | 关系理解 | 推理 | 未来预测 | 总体 |
|------|---------|---------|---------|------|---------|------|
| Qwen2-VL-7B | 52.0 | 49.2 | 50.0 | 43.3 | 45.0 | 48.7 |
| Artemis-7B | 48.0 | 45.2 | 40.0 | 36.7 | 37.5 | 43.0 |
| **VideoRefer-7B** | **72.0** | **66.4** | **60.0** | **60.0** | **57.5** | **64.8** |

### 消融实验

| 配置 | Bench$^D$ Avg | Bench$^Q$ Avg |
|------|---------------|---------------|
| Box-level 特征（RoI） | 2.51 | 57.2 |
| 无 Temporal Token Merge | 2.68 | 61.5 |
| 无 Reviewer 过滤 | 2.55 | 59.8 |
| **完整 VideoRefer** | **2.82** | **64.8** |

### 关键发现
- Mask-level 特征比 Box-level 显著更好（Bench$^Q$ 64.8 vs 57.2），说明像素级精度对区域理解至关重要
- Temporal Token Merge 提升 3.3 个点，证明跨帧时间聚合的必要性
- Reviewer 质量过滤（仅保留 40%）带来 5 个点的提升，强调了数据质量 > 数据数量
- VideoRefer 在通用视频理解基准上也有提升（如 MVBench +2.4%），说明目标级理解能力全面增强了视频理解

## 亮点与洞察
- **完整的数据-模型-基准体系**：这种系统化的研究方式为视频区域理解领域建立了扎实的基础设施
- **多智能体数据引擎**：通过多模型协作 + 严格审核构建高质量数据的方法论可迁移到其他领域
- **统一的 mask 表示**：将框、点、自由形状区域统一为二值 mask，简化了模型设计并提升了灵活性
- 可与 SAM2 无缝集成，点击任意位置即可理解对应目标

## 局限与展望
- 目标编码器的 Mask Pooling 是简单的平均操作，可能丢失区域内部的空间结构信息
- 时间 Token Merge 基于简单的余弦相似度合并，在快速运动场景中可能错合关键帧
- 数据引擎依赖多个大模型（特别是 GPT-4o 作为 Refiner），成本较高
- 未来可探索更精细的区域表示（如多粒度特征金字塔）

## 相关工作与启发
- **vs Artemis**: Artemis 用外部 RoI tracker + 框级特征做单目标引用。VideoRefer 用 mask 级特征 + 多目标支持，在复杂推理上大幅领先
- **vs Osprey**: Osprey 是图像区域理解的先驱，VideoRefer 将其核心思路扩展到视频领域并加入时间维度
- 数据引擎中 Reviewer 的设计思路（用独立模型验证标注一致性）值得在其他数据构建流程中采用

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统化工作，各组件设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 自建基准、通用基准、详细消融、数据质量分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示丰富
- 价值: ⭐⭐⭐⭐⭐ 为视频区域理解提供了完整的基础设施

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding](stop_integrated_spatial-temporal_dynamic_prompting_for_video_understanding.md)
- [\[CVPR 2025\] M-LLM Based Video Frame Selection for Efficient Video Understanding](m-llm_based_video_frame_selection_for_efficient_video_understanding.md)
- [\[CVPR 2025\] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](seq2time_sequential_knowledge_transfer_for_video_llm_temporal_grounding.md)
- [\[CVPR 2025\] FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding](fsbench_a_figure_skating_benchmark_for_advancing_artistic_sports_understanding.md)
- [\[ICCV 2025\] TimeExpert: An Expert-Guided Video LLM for Video Temporal Grounding](../../ICCV2025/video_understanding/timeexpert_an_expert-guided_video_llm_for_video_temporal_grounding.md)

</div>

<!-- RELATED:END -->
