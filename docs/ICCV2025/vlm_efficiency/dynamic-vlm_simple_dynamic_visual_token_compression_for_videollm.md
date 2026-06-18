---
title: >-
  [论文解读] Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM
description: >-
  [ICCV 2025][多模态VLM][VideoLLM] 提出 Dynamic-VLM，通过动态视觉Token压缩器根据视频长度灵活调整每帧Token数量，配合200万级高质量合成视频QA数据集，在 VideoMME 上比 LLaVA-OneVision 提升 2.7%，在 MuirBench 上提升 10.7%。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "VideoLLM"
  - "视觉Token压缩"
  - "动态压缩"
  - "合成数据"
  - "多模态"
---

# Dynamic-VLM: Simple Dynamic Visual Token Compression for VideoLLM

**会议**: ICCV 2025  
**arXiv**: [2412.09530](https://arxiv.org/abs/2412.09530)  
**代码**: 无  
**领域**: 多模态大模型 / 视频理解  
**关键词**: VideoLLM, 视觉Token压缩, 动态压缩, 合成数据, 多模态

## 一句话总结

提出 Dynamic-VLM，通过动态视觉Token压缩器根据视频长度灵活调整每帧Token数量，配合200万级高质量合成视频QA数据集，在 VideoMME 上比 LLaVA-OneVision 提升 2.7%，在 MuirBench 上提升 10.7%。

## 研究背景与动机

视频大语言模型面临两个核心挑战：

**数据缺口**: 图像领域有大量高质量合成数据（LLaVA, ShareGPT4V 等），但视频领域的微调数据仍严重依赖低质量的旧数据集，问题类型单一（仅活动识别、物体计数等）

**架构瓶颈**: 现有 VideoLLM 方案要么：
   - 将视频特征压入外部记忆模块 → 丢失帧级细节
   - 简单扩展LLM上下文窗口 → 计算成本爆炸且性能退化
   - 使用固定压缩率 → 短视频信息丢失、长视频Token过多

核心问题：**如何在上下文窗口和处理帧数之间取得平衡？** 短视频应保留更多细节（少压缩），长视频应增加帧数（高压缩）。

## 方法详解

### 整体框架

Dynamic-VLM = ViT视觉编码器 + 动态Token压缩器 + LLM。
- 视觉编码器: CLIP-ViT-Large@336p（默认）
- LLM: Qwen-2.5 系列（7B/14B）
- 训练分三阶段：预训练 → 图像指令微调 → 视频指令微调

输入视频 $\mathcal{V} = \{X_0, ..., X_{N-1}\}$，每帧独立通过ViT得到视觉特征 $F_i \in \mathbb{R}^{(H \times W) \times C}$，然后通过动态压缩器将Token数从576压缩到目标数量 $M$。

### 关键设计

1. **动态视觉Token压缩器**: 探索了三种候选方案：

    - **Dynamic Spatial Pooling（自适应空间池化）**: $\hat{\mathcal{F}} = \text{AdaptiveAvgPool2d}(\mathcal{F}, [H, W])$，$H=W \in [4, 28]$，Token数 $M = H \times W$。**最终选用此方案**
    - **Dynamic Token Merging**: 基于 ToMe 的双边软匹配，按余弦相似度合并Token
    - **Token Pruning**: 使用 MLP + Gumbel Softmax 打分，保留 Top-K Token

   推理时的动态策略：每帧Token数 = $\max(N_{max}/T, 576)$，其中 $N_{max}$ 是最大视觉Token总预算（7B: 12K，14B: 10K），$T$ 是视频帧数。短视频（少帧）→ 每帧多Token；长视频（多帧）→ 每帧少Token。

2. **200万级合成视频QA数据集**: 从闭源模型 GPT-4V/GPT-4o 收集，原始视频来自三个数据集：

    - **WebVid-10M**（349k视频）: 通过caption去重+低频名词下采样
    - **InternVid-10M**（547k视频）: 仅提供视频（caption质量低）
    - **HDVILA-100M**（3.3M原始视频）: 包含小时级长视频

   任务设计覆盖五大类：
    - **感知任务**: 实体识别、属性、位置、运动描述
    - **通用任务**: 重描述、情感分析、故事写作
    - **时序任务**: 密集视频描述、带时间戳QA、通用时序QA
    - **推理任务**: 视觉推理，提升细节理解
    - **格式化任务**: 多选QA格式引导

3. **训练流程**:

    - 预训练: llava-558K 上先冻结backbone只训练压缩器，再端到端训练caption数据
    - 图像指令微调: 大规模公开数据（General VQA + OCR 类）
    - 视频指令微调: 自制200万 + PerceptionTest + NextQA，上下文窗口16K

### 损失函数 / 训练策略

- 标准自回归语言模型损失（next token prediction）
- 训练时Token数随机采样：图像 $M \in [16, 576]$，视频 $M \in [16, \min(N_{max}/T, 576)]$
- 系统提示: "You are a helpful visual assistant."
- 学习率：LLM+压缩器 $2 \times 10^{-5}$，ViT $4 \times 10^{-6}$（1/5倍率）
- 视频帧以自然时间戳格式输入: "1s: \<image\>; 2s: \<image\>; ..."

## 实验关键数据

### 主实验 (表格)

**多选视频QA (Multi-choice VideoQA)：**

| Method | LLM Size | VideoMME (w/o/w sub) | MLVU | TempCompass | EgoSchema | PerceptionTest |
|--------|---------|---------------------|------|-------------|-----------|----------------|
| GPT-4o | N/A | 71.9/77.2 | 64.6 | 71.0 | - | - |
| LLaVA-OneVision | 7B | 58.2/61.5 | 64.7 | 64.8 | 60.1 | 57.1 |
| LLaVA-OneVision | 72B | 66.2/69.5 | 68.0 | - | 62.0 | 66.9 |
| **Dynamic-VLM** | **7B** | **60.9/64.0** | **65.0** | **62.2** | **68.6** | **68.8** |
| **Dynamic-VLM** | **14B** | **64.6/68.8** | **70.1** | **66.2** | **75.2** | **72.1** |

Dynamic-VLM-7B 在 VideoMME 上比 LLaVA-OneVision-7B 提升 **2.7%**，在 EgoSchema 上提升 **8.5%**。Dynamic-VLM-14B 甚至接近 GPT-4o mini 的 VideoMME 水平。

### 消融实验 (表格)

**压缩器架构对比：**

| Compressor | VideoMME (w/o sub) | MSVD-QA Acc/Score |
|------------|-------------------|-------------------|
| Token Merging | 51.6% | 61.9/3.6 |
| Token Pruning | 47.6% | 59.5/3.5 |
| **Pooling** | **52.0%** | **62.0/3.6** |

**Token数/帧 vs 最大帧数 权衡 (12K token预算)：**

| Tokens/Frame | Max Frames | VideoMME |
|-------------|-----------|---------|
| 36 | 333 | 58.7% |
| 64 | 187 | 59.4% |
| **100** | **120** | **60.9%** |
| 144 | 83 | 59.7% |
| 256 | 46 | 59.3% |

最优点在 100 tokens/frame，说明中等压缩率在信息保留和帧覆盖之间取得了最佳平衡。

### 关键发现

- **池化最简单但最有效**: 三种压缩器中自适应池化效果最好且实现最简单
- **100 tokens/frame 是甜蜜点**: 在12K预算下对应最多120帧，兼顾细节与时序覆盖
- **零样本多图像理解惊人**: 未训练多图像数据却在 MuirBench 上比 LLaVA-OneVision 提升 10.7%，MuirBench 达到 50.7%
- **数据质量 > 数据数量**: 精心设计的prompt生成的200万合成数据带来显著提升

## 亮点与洞察

- **动态策略优雅简洁**: 一个参数（token预算）自动决定压缩率，无需per-video人工调参
- **合成数据工程全面**: 覆盖感知/通用/时序/推理/格式化五类任务，且有明确的质量过滤
- **小模型击败大模型**: 7B的Dynamic-VLM在多个任务上超越72B的LLaVA-OneVision
- **多图像泛化**: 视频训练的模型自然获得了多图像理解能力，说明视频理解能力可泛化
- **时间戳格式**: 将时间信息以文本方式"1s: \<image\>"注入，提供了时序线索

## 局限与展望

- 依赖闭源模型生成训练数据，可重复性受限
- 自适应池化是手工设计的压缩方式，未考虑场景内容自适应
- 最大256帧/视频，对超长视频（如电影级）支持有限
- 未探索音频模态的融合
- 压缩率在推理时固定，未实现内容感知的动态压缩

## 相关工作与启发

- 与 LLaVA-OneVision 路线一致但在视频方向更深入
- 动态Token思想可借鉴到3D点云、医学影像等需要处理大量Token的场景
- 合成数据工程方法具有普适参考价值：去重 → 过滤 → 多任务prompt设计

## 评分

- **新颖性**: ⭐⭐⭐ 架构设计相对简单，主要贡献在数据集和工程
- **实验充分度**: ⭐⭐⭐⭐⭐ 覆盖开放式/多选/零样本多图像三大类任务，消融全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据对比详尽
- **价值**: ⭐⭐⭐⭐ 实用性强，提供了VideoLLM训练的完整方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Dynamic Group Detection using VLM-augmented Temporal Groupness Graph](dynamic_group_detection_using_vlm-augmented_temporal_groupness_graph.md)
- [\[ICCV 2025\] Dynamic Multimodal Prototype Learning in Vision-Language Models](dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [\[CVPR 2026\] OmniZip: Audio-Guided Dynamic Token Compression for Fast Omnimodal Large Language Models](../../CVPR2026/multimodal_vlm/omnizip_audio-guided_dynamic_token_compression_for_fast_omnimodal_large_language.md)
- [\[CVPR 2025\] Dynamic Updates for Language Adaptation in Visual-Language Tracking](../../CVPR2025/multimodal_vlm/dynamic_updates_for_language_adaptation_in_visual-language_tracking.md)
- [\[CVPR 2026\] Dynamic Token Reweighting for Robust Vision-Language Models](../../CVPR2026/multimodal_vlm/dynamic_token_reweighting_for_robust_vision-language_models.md)

</div>

<!-- RELATED:END -->
