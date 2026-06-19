---
title: >-
  [论文解读] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges
description: >-
  [ICCV 2025][视频理解][长视频理解] 提出 VideoLLaMB，通过 SceneTiling 语义分段、循环记忆桥接层和记忆缓存检索机制，以线性 GPU 内存扩展实现长流式视频理解，在 4 个 VideoQA 基准上平均提升 4.2 分。 大规模视频语言模型（如 GPT-4o）在理解流式视频方面展示了巨大潜力…
tags:
  - "ICCV 2025"
  - "视频理解"
  - "长视频理解"
  - "循环记忆"
  - "流式视频"
  - "视频语言模型"
  - "帧检索"
---

# VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges

**会议**: ICCV 2025  
**arXiv**: [2409.01071](https://arxiv.org/abs/2409.01071)  
**代码**: [https://github.com/bigai-nlco/VideoLLaMB](https://github.com/bigai-nlco/VideoLLaMB)  
**领域**: 视频理解  
**关键词**: 长视频理解, 循环记忆, 流式视频, 视频语言模型, 帧检索

## 一句话总结

提出 VideoLLaMB，通过 SceneTiling 语义分段、循环记忆桥接层和记忆缓存检索机制，以线性 GPU 内存扩展实现长流式视频理解，在 4 个 VideoQA 基准上平均提升 4.2 分。

## 研究背景与动机

大规模视频语言模型（如 GPT-4o）在理解流式视频方面展示了巨大潜力，但面临以下挑战：

**计算资源瓶颈**：长视频的高维数据对学术界研究者不可承受

**压缩策略的信息丢失**：采样、聚合、语义合并等方法损失关键视觉线索

**分段方法的语义断裂**：将视频分割为短片段会打断语义流，影响整体理解

**评估偏差**：现有基准存在静态偏差和语言偏差，无法全面评估长视频能力

核心动机：设计一个高效框架，在不丢弃视觉信息的前提下，通过循环记忆机制编码整个视频序列并保持语义连续性。

## 方法详解

### 整体框架

VideoLLaMB 包含三个核心模块：(1) SceneTiling 语义分割器，(2) 循环记忆桥接层，(3) 记忆缓存检索器。视频经 ViT 编码后由 SceneTiling 分段，循环记忆层在语义段间递归编码，记忆缓存通过检索机制维持长程依赖，最终将增强的表示送入 LLM。

### 关键设计

1. **SceneTiling 语义分段算法**：受 TextTiling 启发的无模型场景分割算法。计算相邻帧 [CLS] token 的余弦相似度 $c_i = S_C(\text{ViT}(v_i), \text{ViT}(v_{i+1}))$，然后计算深度分数 $d_i = (cl_i + cr_i - 2c_i)/2$。以 $\mu + \alpha \cdot \sigma$ 为阈值分段。该算法保证段内语义一致性，无需训练即可适配流式视频字幕生成。

2. **循环记忆桥接层（Recurrent Memory Bridge Layers）**：在 Bridge Layer（单层 Transformer）中引入循环记忆 token。对每个语义段 $s_i$，前置记忆 token $[m_i; s_i]$，经自注意力得到 $[m_{i+1}; o_i] = \text{BridgeLayer}([m_i; s_i])$。递归遍历所有语义段更新记忆 token。这样既能将历史视频压缩到记忆中，又通过投影保留当前帧的细节信息。

3. **记忆缓存与检索（Memory Cache with Retrieval）**：在每个时间步 $i$ 存储所有历史记忆 token $M_i = [m_1, ..., m_i]$。通过交叉注意力自检索机制更新当前记忆：$m_{i+1} = \text{Softmax}(W_i^Q m_i (W_i^K M_i)^\top / \sqrt{d_k}) W_i^V M_i$，缓解梯度消失问题并维持长程依赖。

### 损失函数 / 训练策略

- 使用与 PLLaVA 相同的视频数据协议进行训练
- LLM 基座为 Vicuna-7B-v1.5，视觉骨干为 ViT-L/14
- 训练和评估均使用 16 帧、4 个语义段
- 时间复杂度 $\mathcal{O}(K^2)$，空间复杂度 $\mathcal{O}(K)$（$K$ 为段数），GPU 内存线性扩展

## 实验关键数据

### 主实验

EgoSchema 零样本准确率：

| 模型 | LLM | 帧数 | 准确率 |
|------|-----|------|--------|
| GPT-4o | OpenAI API | 16 | 72.2 |
| Video-LLaVA | Vicuna-7B | 8 | 40.2 |
| PLLaVA | Vicuna-7B | 16 | 45.6 |
| PLLaVA | Vicuna-7B | 32 | 43.8 |
| **VideoLLaMB** | Vicuna-7B | 32(训练8) | **53.8** |

NExT-QA 准确率比较：

| 模型 | Temporal | Causal | Description | All |
|------|----------|--------|-------------|-----|
| PLLaVA* | 62.2 | 68.5 | 79.7 | 68.2 |
| **VideoLLaMB*** | **66.8** | **71.6** | 78.4 | **71.1** |

### 消融实验

| 配置 | 关键改进 | 说明 |
|------|---------|------|
| 基础线性投影 | - | 细节保留好但记忆差 |
| + Resampler | 压缩语义 | 语义压缩强但丢细节 |
| + 循环记忆桥接层 | +4.2 avg | 平衡压缩与细节 |
| + 记忆缓存检索 | +长视频鲁棒性 | 解决梯度消失 |
| + SceneTiling | +语义连贯性 | 无训练流式字幕 |

### 关键发现

- 视频长度扩展到 8× 原始长度时仍保持稳健性能
- 在 NIAVH（Needle in a Video Haystack）测试中，1-320 秒视频内均能准确检索目标帧
- 单张 A100 可处理 320 帧（训练仅用 16 帧）
- 在 EgoPlan 任务上，在所有 7B 模型中取得最佳表现，比 PLLaVA 提升 2.06 分

## 亮点与洞察

- SceneTiling 巧妙地将 TextTiling 的思想迁移到视频分段，无需训练即可保持语义一致性
- 循环记忆桥接层在 Bridge 层实现，不修改视觉编码器和 LLM 架构，插拔式设计
- 线性内存扩展使得长视频理解在学术界可行
- NIAVH 基准填补了帧级检索评估的空白

## 局限与展望

- 基于 7B 模型，与 GPT-4o 等大模型仍有差距
- 语义分段质量依赖 ViT [CLS] token 的表征能力
- 记忆缓存随视频增长需要更高效的淘汰/压缩策略
- 训练帧数有限（16 帧），超长视频的泛化能力有待进一步验证

## 相关工作与启发

- 循环记忆 + 检索的组合思路可推广到其他需要长程依赖的多模态任务
- SceneTiling 的免训练流式处理范式对实时视频理解有实用价值
- 桥接层理念：在投影与压缩之间取平衡，值得其他视频-语言模型借鉴

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体 | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Online Dense Point Tracking with Streaming Memory](online_dense_point_tracking_with_streaming_memory.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](../../NeurIPS2025/video_understanding/videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[CVPR 2025\] ReWind: Understanding Long Videos with Instructed Learnable Memory](../../CVPR2025/video_understanding/rewind_understanding_long_videos_with_instructed_learnable_memory.md)
- [\[NeurIPS 2025\] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding](../../NeurIPS2025/video_understanding/infinipot-v_memory-constrained_kv_cache_compression_for_streaming_video_understa.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)

</div>

<!-- RELATED:END -->
