---
title: >-
  [论文解读] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding
description: >-
  [CVPR 2025][多模态VLM][长视频理解] 利用 LLM 内部的 KV 稀疏化能力实现长视频 token 压缩——引入视觉摘要 token（VST）将每段视频的视觉信息压缩到其 KV 中并卸载原始视觉 KV，配合动态压缩和课程学习，在单 A100 上处理 2048 帧，MLVU Dev 上超越 GPT-4o。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "长视频理解"
  - "KV缓存压缩"
  - "视觉摘要token"
  - "课程学习"
  - "小时级视频"
---

# Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding

**会议**: CVPR 2025  
**arXiv**: [2409.14485](https://arxiv.org/abs/2409.14485)  
**代码**: [https://github.com/VectorSpaceLab/Video-XL](https://github.com/VectorSpaceLab/Video-XL)  
**领域**: 多模态VLM  
**关键词**: 长视频理解、KV缓存压缩、视觉摘要token、课程学习、小时级视频

## 一句话总结
利用 LLM 内部的 KV 稀疏化能力实现长视频 token 压缩——引入视觉摘要 token（VST）将每段视频的视觉信息压缩到其 KV 中并卸载原始视觉 KV，配合动态压缩和课程学习，在单 A100 上处理 2048 帧，MLVU Dev 上超越 GPT-4o。

## 研究背景与动机

**领域现状**：长视频理解（小时级）是 VLM 的重要应用方向，但数千帧意味着数十万视觉 token，远超 LLM 的上下文窗口。

**现有痛点**：现有压缩方法在 LLM 之前做（如池化、Q-Former、C-Abstractor），在高压缩比（16×）时性能大幅下降——因为视觉信息的压缩和利用被分离了，压缩器不知道哪些信息对后续推理重要。

**核心矛盾**：长视频需要高压缩比（16× 甚至更高），但预压缩方法在高比率下丢失太多关键信息。

**本文目标** 利用 LLM 内部的注意力机制做语义感知的 KV 压缩，而非在 LLM 外部做盲目的 token 压缩。

**切入角度**：在视频间插入可学习的 Visual Summarization Token（VST），让 LLM 的注意力机制自然地将视觉信息压缩到 VST 的 KV 中，然后卸载原始视觉 token 的 KV，仅保留 VST KV。

**核心 idea**：将视觉 token 压缩从 LLM 外部移到 LLM 内部——用 VST 的 KV 缓存替代原始视觉 KV，利用 LLM 自身的注意力做语义感知压缩。

## 方法详解

### 整体框架
视频帧经视觉编码器提取 token → 每 $n$ 帧间插入一组 VST token → 送入 LLM → LLM 处理时自注意力将关键视觉信息压缩到 VST 的 KV 中 → 卸载原始视觉 token KV，只保留 VST KV + 文本 KV → 后续推理用 VST KV 代表视觉上下文。

### 关键设计

1. **Visual Summarization Token（VST）**:

    - 功能：在 LLM 内部做视觉信息压缩的载体
    - 核心思路：在每段视频帧序列后插入可学习的 VST token。LLM 的因果注意力使 VST 能看到它之前的所有视觉 token，因此其 KV 自然编码了前序视觉信息。处理完后卸载原始视觉 KV 仅保留 VST KV
    - 设计动机：消融实验显示 LLM 内部压缩（VST）在 16× 压缩比下 MLVU 41.4，远超池化 33.7、Q-Former 35.1、C-Abstractor 37.1

2. **动态压缩**:

    - 功能：根据视频内容的信息密度自适应调整压缩粒度
    - 核心思路：用 CLIP 深度分数衡量相邻帧的语义变化幅度——变化大的区域（如场景切换）用更细的粒度（更多 VST），变化小的区域（如静态画面）用更粗的粒度
    - 设计动机：均匀压缩会在信息密集区域丢失关键细节，在信息稀疏区域浪费容量

3. **课程学习**:

    - 功能：渐进式引导模型学会越来越高的压缩比
    - 核心思路：训练初期使用低压缩比（2×, 4×），逐步增加到高压缩比（8×, 12×, 16×）。避免模型一开始就面对极高压缩比导致学习崩溃
    - 设计动机：直接用 16× 训练效果差（MLVU 37.2 vs 课程学习 41.4），渐进学习让模型逐步习得压缩技能

### 损失函数 / 训练策略
标准 next-token prediction 损失。训练数据混合图像、多图和长视频数据。VICO 合成数据集（从 CinePile 视频生成的视觉线索排序任务）增强长程理解。

## 实验关键数据

### 主实验

| 模型 | 大小 | MLVU Dev | VideoMME | VNBench | LongVidBench |
|------|------|---------|----------|---------|-------------|
| GPT-4o | - | 64.6 | 71.9 | 64.4 | 66.7 |
| LongVA | 7B | 56.3 | 52.6 | 41.5 | 47.8 |
| **Video-XL** | **7B** | **64.9** | **55.5** | **61.6** | **50.7** |

### 消融实验

| 压缩方法 (16×) | MLVU | VideoMME | MME | MMB |
|---------------|------|---------|-----|-----|
| 池化 | 33.7 | 41.0 | 1405 | 62.3 |
| Q-Former | 35.1 | 42.1 | 1410 | 61.9 |
| C-Abstractor | 37.1 | 46.3 | 1440 | 65.1 |
| **Video-XL (VST)** | **41.4** | **52.0** | **1510** | **70.9** |

### 关键发现
- **7B 模型超越 GPT-4o**：在 MLVU Dev 上 Video-XL 7B（64.9）超越 GPT-4o（64.6），证明内部压缩策略的有效性
- **16× 压缩下 VST >> 外部压缩**：VST 比最强外部方法 C-Abstractor 高 4.3 个点（MLVU），因为 LLM 内部注意力知道哪些信息对推理重要
- **2048 帧单 A100**：在 Needle-in-Haystack 测试中保持 95% 准确率，实现小时级视频理解

## 亮点与洞察
- **"在 LLM 内部压缩"颠覆了外部压缩范式**——让 LLM 自己决定保留什么比外部压缩器猜测要准确得多
- **课程学习是高压缩比的关键**：直接 16× 训练 vs 渐进训练差 4.2 个点
- **VST 思路可推广**到任何需要长序列压缩的 LLM 应用（如长文档、多图理解）

## 局限与展望
- VST 的 KV 缓存仍需占用内存，极长视频（10+ 小时）可能仍然受限
- 动态压缩的 CLIP 深度分数计算有额外开销
- 仅在 7B 模型上验证，更大模型的效果未知

## 相关工作与启发
- **vs LongVA / LLaMA-VID**：这些方法用外部 token 压缩，在高压缩比下性能急剧下降。Video-XL 的内部压缩在 16× 下仍保持高精度
- **vs MovieChat / StreamingLLM**：这些方法处理流式视频但不保证长程理解。Video-XL 的 VST 显式编码了长程依赖

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ LLM 内部 KV 压缩替代外部 token 压缩是范式创新
- 实验充分度: ⭐⭐⭐⭐ 多个长视频基准 + Needle-in-Haystack + 压缩方法对比
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，压缩对比实验有说服力
- 价值: ⭐⭐⭐⭐⭐ 对长视频理解有重大贡献，7B 超越 GPT-4o 是亮点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](../../CVPR2026/multimodal_vlm/scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)
- [\[CVPR 2026\] ReMoRa: Multimodal Large Language Model based on Refined Motion Representation for Long-Video Understanding](../../CVPR2026/multimodal_vlm/remora_multimodal_large_language_model_based_on_refined_motion_representation_fo.md)
- [\[CVPR 2025\] SVLTA: Benchmarking Vision-Language Temporal Alignment via Synthetic Video Situation](svlta_benchmarking_vision-language_temporal_alignment_via_synthetic_video_situat.md)
- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](efficient_motion-aware_video_mllm.md)

</div>

<!-- RELATED:END -->
