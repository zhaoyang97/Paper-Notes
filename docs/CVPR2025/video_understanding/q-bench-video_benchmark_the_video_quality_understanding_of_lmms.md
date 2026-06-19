---
title: >-
  [论文解读] Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs
description: >-
  [CVPR 2025][视频理解][视频质量评估] 首个系统评估大型多模态模型视频质量理解能力的基准 Q-Bench-Video，涵盖自然/AIGC/CG视频、四维质量关注和多题型设计。 核心矛盾 核心矛盾：领域现状：大型多模态模型 (LMM) 在高层语义视频理解任务上取得显著进展，但对视频质量理解的系统评估严重缺乏…
tags:
  - "CVPR 2025"
  - "视频理解"
  - "视频质量评估"
  - "大模型基准"
  - "LMM"
  - "AIGC失真"
  - "时序一致性"
---

# Q-Bench-Video: Benchmark the Video Quality Understanding of LMMs

**会议**: CVPR 2025  
**arXiv**: [2409.20063](https://arxiv.org/abs/2409.20063)  
**代码**: [https://github.com/Q-Future/Q-Bench-Video](https://github.com/Q-Future/Q-Bench-Video)  
**领域**: 视频理解 / 质量评估  
**关键词**: 视频质量评估, 大模型基准, LMM, AIGC失真, 时序一致性

## 一句话总结

首个系统评估大型多模态模型视频质量理解能力的基准 Q-Bench-Video，涵盖自然/AIGC/CG视频、四维质量关注和多题型设计。

## 研究背景与动机

### 核心矛盾

**核心矛盾**：**领域现状**：大型多模态模型 (LMM) 在高层语义视频理解任务上取得显著进展，但对视频质量理解的系统评估严重缺乏。视频质量对压缩优化、观看体验提升和视频生成标准制定至关重要，涉及的低层级信息（模糊、噪声、压缩伪影等）与高层语义理解有本质区别。现有 LMM 视频基准（如 MVBench、Video-MME）专注语义理解，遗漏了质量感知维度。另一方面，AIGC 视频生成的爆发式发展引入了全新的失真类型（不自然纹理、光照不一致等），急需专门的评估框架。本文系统填补了这一空白。

### 解决思路

**本文目标**：### 整体框架

Q-Bench-Video 的构建遵循三个原则：(1) 广泛的视频内容覆盖——自然场景1000、AIGC 600、CG 200 共1800视频；(2) 基于质量标注的均匀采样确保质量分布平衡；(3) 聚焦影响观看体验的四维质量关注。


## 方法详解

### 整体框架

Q-Bench-Video 的构建遵循三个原则：(1) 广泛的视频内容覆盖——自然场景1000、AIGC 600、CG 200 共1800视频；(2) 基于质量标注的均匀采样确保质量分布平衡；(3) 聚焦影响观看体验的四维质量关注。每个数据条目为元结构（V, Q, A, C），总计 2378 个问答对。评估了12个开源 + 5个闭源 LMM。

### 关键设计

1. **三种题型设计**：(a) Yes-or-No 题：二元判断视频质量，通过标注调整保证正确答案50:50平衡，避免 LMM 的偏向倾向；(b) What-How 题：What 识别具体失真类型，How 区分失真程度的精细差异；(c) Open-ended 开放题：不限定答案集合，评估 LMM 在真实场景中感知视频质量的能力，如"请列出导致该视频清晰度低的可能因素并解释"。此外增加视频对比较任务评估相对质量判断能力。

2. **四维质量关注**：(a) 技术失真：模糊、噪声、压缩伪影等底层退化；(b) 美学失真：构图、色彩、光照等主观审美偏差；(c) 时序失真：画面抖动、闪烁、运动不一致、卡顿等时域问题；(d) AIGC失真：AI生成内容特有的不自然纹理、诡异面部、不现实物体行为等。单个问题可同时涉及多个维度。

3. **视频来源多样性**：自然视频来自 LSVQ（39K采600）、MaxWell（4.5K采350）、WaterlooSQoE 系列；AIGC 视频来自 T2VQA-DB（10K采200）和 VideoFeedback（37.6K采400）；CG 视频来自 LIVE-YT-Gaming（600采200）。大部分数据集含 ITU 标准 MOS 标注，确保质量采样的科学性。

### 损失函数 / 训练策略

- 纯评估基准，无训练组件
- 开放题使用 GPT-4 辅助评分
- 多选题使用准确率
- 视频对比较使用一致率

## 实验关键数据

### 主实验

| 模型 | Yes-or-No↑ | What-How↑ | Open-ended↑ | 平均↑ |
|------|-----------|----------|-------------|------|
| GPT-4o | 最高 | 最高 | 最高 | 最高 |
| InternVL2 | 次高 | 次高 | - | 次高 |
| VideoLLaMA2 | 中等 | 中等 | - | 中等 |
| 人类表现 | **远高于所有LMM** | **远高于所有LMM** | **远高于所有LMM** | **显著领先** |

### 消融实验

| 维度 | LMM 表现差异 |
|------|-------------|
| 技术失真 | 较好（LMM对模糊/噪声有基本感知） |
| 美学失真 | 中等 |
| 时序失真 | 差（LMM 难以捕捉时域问题） |
| AIGC失真 | 差（LMM 对AI生成artifact不敏感） |

### 关键发现

- LMM 对视频质量有基础理解但不完整不精确，与人类表现差距显著
- 闭源模型（GPT-4o等）显著优于开源模型
- LMM 在时序失真和 AIGC 失真维度表现最差——这正是视频质量最独特的两个方面
- 视频对比较任务比单视频评估更具挑战性
- 开放题暴露了 LMM 在解释质量原因方面的不足

## 亮点与洞察

- 首个将 LMM 视频质量理解作为独立研究方向提出的工作，填补重要空白
- AIGC 失真维度的引入非常及时——随视频生成模型普及，此类评估需求急增
- Yes-or-No 题的平衡设计和开放题的引入提升了评估的全面性和真实性
- 基准揭示了 LMM 在低层信息感知上的根本局限性

## 局限与展望

- 2378 个问答对规模可以进一步扩大
- 开放题评估依赖 GPT-4 可能引入偏差
- 未评估 LMM 的视频质量评分能力（定量打分 vs 定性描述）
- 可扩展到更多视频生成模型的输出评估

## 相关工作与启发

- **vs Video-MME/MVBench**: 专注语义理解；Q-Bench-Video 专注低层质量理解，互为补充
- **vs 传统 VQA 方法**: 传统方法输出质量分数；Q-Bench-Video 评估 LMM 的质量理解和解释能力
- **vs Q-Bench (图像版)**: 将图像质量基准范式扩展到视频，增加时序和 AIGC 维度

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个视频质量 LMM 基准，方向开创性
- **实验充分度**: ⭐⭐⭐⭐⭐ — 17个模型、4维度、3题型的全面评估
- **写作质量**: ⭐⭐⭐⭐ — 基准设计原则清晰，分类系统完整
- **实用价值**: ⭐⭐⭐⭐⭐ — 为视频质量理解研究提供标准化评测平台

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)
- [\[CVPR 2026\] FPS-Bench: A Benchmark for High Frame-Rate Video Understanding](../../CVPR2026/video_understanding/fps-bench_a_benchmark_for_high_frame-rate_video_understanding.md)
- [\[CVPR 2025\] SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding](seriesbench_a_benchmark_for_narrative-driven_drama_series_understanding.md)
- [\[CVPR 2025\] FSBench: A Figure Skating Benchmark for Advancing Artistic Sports Understanding](fsbench_a_figure_skating_benchmark_for_advancing_artistic_sports_understanding.md)
- [\[CVPR 2025\] Towards Universal Soccer Video Understanding](towards_universal_soccer_video_understanding.md)

</div>

<!-- RELATED:END -->
