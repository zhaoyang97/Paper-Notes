---
title: >-
  [论文解读] VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation
description: >-
  [ACL 2025][视频生成][视频描述] 提出 VidCapBench，首个专为可控文生视频（T2V）设计的视频描述评估 benchmark，从美学/内容/运动/物理规律四个维度评估 caption 质量，643 个视频+10,644 个 QA 对，实验证明 VidCapBench 分数与 T2V 生成质量高度正相关。
tags:
  - "ACL 2025"
  - "视频生成"
  - "视频描述"
  - "benchmark"
  - "文生视频"
  - "caption评估"
  - "多维度评价"
---

# VidCapBench: A Comprehensive Benchmark of Video Captioning for Controllable Text-to-Video Generation

**会议**: ACL 2025  
**arXiv**: [2502.12782](https://arxiv.org/abs/2502.12782)  
**代码**: [https://github.com/VidCapBench/VidCapBench](https://github.com/VidCapBench/VidCapBench)  
**领域**: 视频理解 / 视频生成  
**关键词**: 视频描述, benchmark, 文生视频, caption评估, 多维度评价

## 一句话总结
提出 VidCapBench，首个专为可控文生视频（T2V）设计的视频描述评估 benchmark，从美学/内容/运动/物理规律四个维度评估 caption 质量，643 个视频+10,644 个 QA 对，实验证明 VidCapBench 分数与 T2V 生成质量高度正相关。

## 研究背景与动机

**领域现状**：可控 T2V 生成依赖高质量视频描述（caption）与视频的对齐。现有 caption 评估 benchmark（MSR-VTT、VATEX）使用 CIDEr 等传统指标评估短描述，不适合评估 T2V 生成所需的详细多维 caption。

**现有痛点**：(1) 现有评估未覆盖 T2V 关键维度（美学、运动、物理规律）；(2) 自动评估不稳定——VDC benchmark 中仅 41% 的 QA 在多次评估中结果一致；(3) Caption 评估与 T2V 生成质量的关联性未被验证。

**核心矛盾**：T2V 模型训练需要高质量 caption，但缺乏衡量 caption 是否满足 T2V 需求的评估标准。

**本文目标**：(1) 定义 T2V 导向的 caption 评估维度；(2) 构建稳定可靠的评估 benchmark；(3) 验证 caption 评估与 T2V 质量的正相关性。

**切入角度**：从 T2V 生成模型关注的核心要素（美学、内容、运动、物理规律）出发设计评估维度，而非从 caption 本身出发。

**核心 idea**：将 QA 对分为"自动评估子集"（评估稳定的）和"人工评估子集"（评估困难的），兼顾效率和准确性。

## 方法详解

### 整体框架
数据收集 → 四维度标注（视频美学/内容/运动/物理规律）→ 生成 QA 对 → 按评估稳定性分层 → 自动 + 人工混合评估。

### 关键设计

1. **四维度评估体系**:

    - **视频美学 (VA)**：拍摄技术、后期处理、画面构图等
    - **视频内容 (VC)**：叙事内容、主体/背景/场景描述
    - **视频运动 (VM)**：前景主体运动、背景物体运动、镜头运动
    - **物理规律 (PL)**：物理现象的合理性和一致性
    - 设计动机：与 T2V 模型评估核心维度（VBench、EvalCrafter 等）完全对齐

2. **数据标注流水线**:

    - 功能：为 643 个视频创建多维度 QA 标注
    - 核心思路：(1) 收集多源视频（开源数据集 + YouTube + UGC），确保主体多样性（人/动物/植物/食物/物体/风景等 10 类均匀分布）；(2) 使用专家模型（姿态估计、目标检测、光流等）自动标注视频属性；(3) 基于属性生成 QA 对，人工审核精修
    - 设计动机：结合自动标注和人工精修，平衡数据质量和标注成本

3. **分层评估策略**:

    - 功能：将 QA 对分为自动评估子集和人工评估子集
    - 核心思路：对 QA 对进行多次重复评估（3 次 × 5 个模型），仅将**所有评估一致**的 QA 归为"自动评估子集"（约 41%），其余需人工评估
    - 设计动机：发现仅用自动评估会导致显著偏差（如短 caption 在自动评估中得分虚高），分层策略同时满足"快速迭代"和"精确验证"需求

4. **四维度评估指标**:

    - **Accuracy (Acc)**：完全正确的比例
    - **Precision (Pre)**：已提及内容中正确的比例
    - **Coverage (Cov)**：QA 内容被 caption 覆盖的比例
    - **Conciseness (Con)**：每个 token 对 Acc 的贡献（鼓励简洁）
    - 设计动机：单一指标（如 CIDEr）无法全面反映 caption 质量，四个指标从不同角度互补

### 与 T2V 的关联验证
用不同模型的 caption 作为 prompt 输入 T2V 模型（CogVideoX、Hunyuan Video），计算 T2V 质量指标（VBench）与 VidCapBench 分数的相关性。

## 实验关键数据

### 主实验（VidCapBench-AE 自动评估部分）

| 模型 | Overall Acc | Video Aesthetics | Video Content | Video Motion | Physical Laws |
|------|-----------|-----------------|---------------|-------------|--------------|
| GPT-4o | **16.8** | 14.1 | 17.5 | 10.2 | **27.9** |
| Gemini 1.5 Pro | 17.1 | 16.4 | 16.9 | 9.8 | 28.4 |
| Qwen2-VL-72B | 15.2 | 14.3 | 15.0 | 5.0 | 25.9 |
| CogVLM2-Caption | 13.1 | 12.5 | 12.7 | 5.7 | 27.9 |
| Tarsier-34B | 11.1 | 10.7 | 10.2 | 3.2 | 26.2 |
| LLaVA-Next-Video-7B | 10.6 | 11.3 | 9.6 | 4.4 | 24.4 |

### T2V 相关性验证

| Caption 模型 | VidCapBench Acc | CogVideoX VBench ↑ |
|-------------|----------------|-------------------|
| GPT-4o | 16.8 | 最高 |
| CogVLM2-Caption | 13.1 | 中等 |
| LLaVA-Next-Video | 10.6 | 较低 |

Pearson 相关系数 r > 0.8，证明 VidCapBench 分数与 T2V 质量高度正相关。

### 关键发现
- **所有模型在运动维度表现最差**：Video Motion Acc 普遍 < 10%，说明当前 VLM 难以准确描述视频运动——这是 T2V 对齐的关键瓶颈
- **闭源模型 > 开源模型**：GPT-4o / Gemini 在 Overall Acc 上领先，但优势不如预期大
- **长 caption 不一定好**：Conciseness 指标揭示过长 caption 的信息密度低（InternVL2 Acc 10.2 但 Con 仅 2.5）
- **VDC 评估不稳定**：仅 41% QA 对在多次自动评估中一致，验证了分层评估的必要性
- **Caption 质量→T2V 质量**：VidCapBench 分数与 VBench 分数显著正相关，为 caption 优化提供量化指导

## 亮点与洞察
- **首个 T2V 导向的 caption benchmark**：从 T2V 评估维度反推 caption 评估维度，建立了 caption→T2V 的评估闭环
- **分层自动+人工评估**：巧妙利用评估一致性将 QA 分层，是解决自动评估不稳定问题的通用策略
- **Video Motion 维度的诊断价值**：所有模型在运动描述上表现极差（<10% Acc），直接指出了 caption 模型的改进方向

## 局限与展望
- 643 个视频规模相对较小，覆盖场景有限
- 自动评估仍依赖 GPT-4o 作为 judge，judge 能力是评估的天花板
- QA 对的粒度设计对结果影响大，但最优粒度未深入探索
- 仅验证了与 CogVideoX/Hunyuan 两个 T2V 模型的相关性
- 物理规律（PL）维度的标注难度大，可能存在标注噪声

## 相关工作与启发
- **vs VDC**: VDC 有 ~100K QA 但评估不稳定（仅 41% 一致）。VidCapBench 规模小但稳定，且覆盖更多维度（美学、物理规律）
- **vs DREAM-1K**: DREAM-1K 评估事件级描述，但不评估美学和物理合理性
- **vs MSR-VTT/VATEX**: 传统短 caption 评估，使用 CIDEr 等指标，与 T2V 需求脱节

## 评分
- 新颖性: ⭐⭐⭐⭐ 从 T2V 视角评估 caption 是好的切入点，分层评估策略有亮点
- 实验充分度: ⭐⭐⭐⭐ 多模型评估充分，T2V 相关性验证有说服力，但数据集偏小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题定义准确
- 价值: ⭐⭐⭐⭐ 对 T2V 领域的 caption 质量优化有直接指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VABench: A Comprehensive Benchmark for Audio-Video Generation](../../CVPR2026/video_generation/vabench_a_comprehensive_benchmark_for_audio-video_generation.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](../../ICLR2026/video_generation/drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[CVPR 2026\] ActivityForensics: A Comprehensive Benchmark for Localizing Manipulated Activity in Videos](../../CVPR2026/video_generation/activityforensics_a_comprehensive_benchmark_for_localizing_manipulated_activity_.md)
- [\[CVPR 2025\] Video-Bench: Human-Aligned Video Generation Benchmark](../../CVPR2025/video_generation/video-bench_human-aligned_video_generation_benchmark.md)
- [\[ICML 2026\] V2V-Bench: A Comprehensive Benchmark for Video-to-Video Generation Evaluation](../../ICML2026/video_generation/v2v-bench_a_comprehensive_benchmark_for_video-to-video_generation_evaluation.md)

</div>

<!-- RELATED:END -->
