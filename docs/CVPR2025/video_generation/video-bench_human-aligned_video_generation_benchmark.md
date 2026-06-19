---
title: >-
  [论文解读] Video-Bench: Human-Aligned Video Generation Benchmark
description: >-
  [CVPR 2025][视频生成][视频生成评估] 本文提出 Video-Bench，一个全面的视频生成评估基准，通过 Chain-of-Query 和 Few-Shot Scoring 两种技术系统性地利用多模态大语言模型（MLLM）自动评估生成视频，在所有评估维度上实现了与人类偏好最高的对齐度。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频生成评估"
  - "Benchmark"
  - "MLLM评估"
  - "人类偏好对齐"
  - "Chain-of-Query"
---

# Video-Bench: Human-Aligned Video Generation Benchmark

**会议**: CVPR 2025  
**arXiv**: [2504.04907](https://arxiv.org/abs/2504.04907)  
**代码**: [https://github.com/Video-Bench/Video-Bench.git](https://github.com/Video-Bench/Video-Bench.git)  
**领域**: 扩散模型  
**关键词**: 视频生成评估, Benchmark, MLLM评估, 人类偏好对齐, Chain-of-Query

## 一句话总结

本文提出 Video-Bench，一个全面的视频生成评估基准，通过 Chain-of-Query 和 Few-Shot Scoring 两种技术系统性地利用多模态大语言模型（MLLM）自动评估生成视频，在所有评估维度上实现了与人类偏好最高的对齐度。

## 研究背景与动机

**领域现状**：视频生成评估基准主要分为两类——基于指标和嵌入的方法（如 FID、FVD、CLIP 得分）提供定量评估但常与人类判断不一致；基于 LLM 的方法虽有推理能力但面临跨模态比较困难和文本评分标准模糊两大限制。

**现有痛点**：指标类方法（VBench、EvalCrafter）通过组合计算指标评估视频质量，但评估结果与人类偏好显著不一致。LLM 类方法（CompBench、T2VScore）虽尝试引入推理能力，但仅在文本-视频对齐维度使用 LLM，其他方面仍依赖传统指标。具体来说存在两个瓶颈：(1) 视频-文本对齐评估中 MLLM 容易产生文本偏见和幻觉，难以准确检测跨模态不一致；(2) 视频质量评估中文本标准的模糊性导致模型倾向给出"平均分"。

**核心矛盾**：自动评估方法的评分与人类感知之间存在显著差距，特别是在需要跨模态理解和细粒度质量判断的维度上。

**本文目标**：构建一个全维度、与人类偏好高度对齐的自动视频生成评估框架。

**切入角度**：作者认为直接让 MLLM 评分面临两个本质困难——跨模态比较和模糊评分标准，提出将问题分解：对齐类维度先将视频转为文本再比较，质量类维度用多视频参照校准评分。

**核心 idea**：通过 Chain-of-Query（迭代式多轮视频-文本转换与对比）和 Few-Shot Scoring（批量视频互为参照校准评分）两种策略，系统性解决 MLLM 在视频评估中的两大瓶颈。

## 方法详解

### 整体框架

Video-Bench 包含两个层面：(1) 评估维度体系，分为视频-条件对齐（5 个维度）和视频质量（4 个维度）共9个维度；(2) MLLM 自动评估框架，针对不同类型的维度设计不同的评估策略——对齐类用 Chain-of-Query，质量类用 Few-Shot Scoring。配套提供 419 个提示词和 35,196 条人类标注。

### 关键设计

1. **Chain-of-Query（对齐类维度评估）**:

    - 功能：通过多轮迭代将跨模态比较转化为同模态文本比较，解决 MLLM 的跨模态幻觉问题
    - 核心思路：分为四步——(1) MLLM 生成视频的初始文本描述和摘要；(2) LLM 根据描述和原始提示词生成 N 组针对性问题链（如"视频中考拉的颜色是否与提示匹配？"）；(3) MLLM 逐一回答问题并重新生成描述，补充维度相关细节；(4) MLLM 利用视频内容和多轮对话历史综合评分。整个过程避免了直接的跨模态比较
    - 设计动机：直接让 MLLM 比较视频和文本容易产生幻觉，先将视频信息转化为文本形式后再比较能显著降低模态差异带来的误判

2. **Few-Shot Scoring（质量类维度评估）**:

    - 功能：解决 MLLM 对视频质量评分时倾向给出"平均分"的问题
    - 核心思路：将同一提示词生成的多个视频组成批次，评分时每个视频以批次中其他视频作为参照。评第二个视频时，第一个视频的评分和所有批次内视频都作为隐式参照，形成比较框架。类似于"把参考答案和待评作品放在一起对比打分"
    - 设计动机：仅靠文本标准（如"有点模糊"和"非常清晰"之间的界限）过于模糊，MLLM 无法区分不同等级；提供具体视频参照相当于给出"锚点"，使评分更有区分度

3. **维度体系设计**:

    - 功能：提供全面的评估维度覆盖
    - 核心思路：视频-条件对齐包含物体类别一致性、动作一致性、颜色一致性、场景一致性（3分制）和视频-文本整体一致性（5分制）；视频质量包含成像质量、美学质量、时序一致性和运动质量（均5分制）
    - 设计动机：现有基准维度不完整，且不同维度难度不同需要不同的评分量表

### 损失函数 / 训练策略

本工作不涉及模型训练，评估框架基于 GPT-4o（多模态输入）和 GPT-4o-mini（纯文本推理）。提示词套件设计结合了 Kinetics-400 人体动作数据和 VBench 的相关提示词，每个维度 70-90 个提示词，每个提示词采样 3 次以减少随机性偏差。

## 实验关键数据

### 主实验

| 评估方法 | 成像质量 | 美学质量 | 时序一致性 | 运动效果 | 整体对齐 | 物体类别 | 颜色 | 动作 | 场景 |
|---------|---------|---------|----------|---------|---------|---------|------|------|------|
| MUSIQ | 0.363 | - | - | - | - | - | - | - | - |
| CLIP | - | - | 0.260 | - | - | - | - | - | - |
| CompBench* | - | - | - | - | 0.633 | 0.611 | 0.696 | 0.633 | 0.631 |
| **Video-Bench** | **0.733** | **0.702** | **0.402** | **0.514** | **0.732** | **0.735** | **0.750** | **0.718** | **0.733** |

### 消融实验

| 配置 | 对齐类平均 Spearman | 质量类平均 Spearman | 说明 |
|------|-------------------|-------------------|------|
| 无 Chain-of-Query | 0.679 | - | 单轮评估 |
| + Chain-of-Query | 0.7336 | - | 多轮迭代提升 +0.054 |
| 无 Few-Shot Scoring | - | 0.561 | 独立评分 |
| + Few-Shot Scoring | - | 0.620 | 批量参照提升 +0.059 |

### 关键发现

- Video-Bench 的人机一致性（Krippendorff α = 0.50）与人类标注者间一致性（α = 0.52）几乎持平，说明自动评估已接近人类评估水平
- 在所有 9 个维度上 Video-Bench 都优于现有方法，其中 Chain-of-Query 对对齐类维度提升最大（Color Consistency 从 0.699 提升到 0.750）
- 模型排行榜显示 Gen3 在视频质量上最优，CogVideoX 在条件对齐上最优
- 在某些 Video-Bench 评分与人类评估不一致的案例中，Video-Bench 实际上提供了更客观准确的判断

## 亮点与洞察

- **Chain-of-Query 的"模态转换"策略**很巧妙——不让 MLLM 直接做跨模态比较，而是先把视频"翻译"成文本再做文本间比较，有效规避了多模态模型的幻觉问题。这种思路可推广到任何涉及跨模态对齐评估的场景
- **Few-Shot Scoring 的"批量互参"机制**本质上引入了相对评分的思想，解决了绝对评分标准模糊的问题。这个 trick 在任何需要 LLM 打分的场景都可复用
- 提出的 9 维度评估体系较为全面，从物体级别到视频级别逐层覆盖

## 局限与展望

- 依赖 GPT-4o 作为评估模型，API 成本较高且可能存在版本更新导致的不一致性
- 提示词套件规模有限（419 个），对某些长尾场景覆盖不足
- 仅评估文本到视频生成，未覆盖图像到视频、视频编辑等新兴任务
- Few-Shot Scoring 要求同 prompt 生成多个视频，增加了评估成本
- 运动效果维度的 Spearman 相关系数仍较低（0.514），表明动态质量评估仍是难点

## 相关工作与启发

- **vs VBench**: VBench 主要基于计算指标组合评估，与人类偏好对齐度较低；Video-Bench 全面使用 MLLM 并通过 Chain-of-Query 和 Few-Shot Scoring 显著提升对齐度
- **vs CompBench**: CompBench 仅在对齐维度使用单轮 LLM 评估，Video-Bench 的多轮迭代在 Video-Condition Alignment 上平均提升 0.093
- **vs EvalCrafter**: EvalCrafter 用线性回归拟合用户评分与指标的关系，方法较为间接

## 评分

- 新颖性: ⭐⭐⭐⭐ Chain-of-Query 和 Few-Shot Scoring 是有效的方法创新，但整体框架是组合式的
- 实验充分度: ⭐⭐⭐⭐⭐ 35K 人类标注、7 个生成模型、9 个维度、详细消融，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，但部分符号使用略繁重
- 价值: ⭐⭐⭐⭐ 为视频生成评估提供了一个强基线，Chain-of-Query 思想有较好的通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VMBench: A Benchmark for Perception-Aligned Video Motion Generation](../../ICCV2025/video_generation/vmbench_a_benchmark_for_perception-aligned_video_motion_generation.md)
- [\[CVPR 2025\] VEU-Bench: Towards Comprehensive Understanding of Video Editing](veu-bench_towards_comprehensive_understanding_of_video_editing.md)
- [\[ICML 2026\] V2V-Bench: A Comprehensive Benchmark for Video-to-Video Generation Evaluation](../../ICML2026/video_generation/v2v-bench_a_comprehensive_benchmark_for_video-to-video_generation_evaluation.md)
- [\[CVPR 2025\] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation](tokenmotion_decoupled_motion_control_via_token_disentanglement_for_human-centric.md)
- [\[CVPR 2025\] HOIGen-1M: A Large-Scale Dataset for Human-Object Interaction Video Generation](hoigen-1m_a_large-scale_dataset_for_human-object_interaction_video_generation.md)

</div>

<!-- RELATED:END -->
