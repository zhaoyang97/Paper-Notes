---
title: >-
  [论文解读] Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference
description: >-
  [ICCV 2025][多模态VLM][长视频理解] 提出免训练方法Free-MoRef，受MoE启发将长视频token分割为多个短序列作为多参考(multi-reference)，通过MoRef注意力机制并行查询并融合统一激活值，在单卡A100上实现2×到8×更长帧输入的高效全面理解，在VideoMME/MLVU/LongVideoBench上超越专训长视频模型。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "长视频理解"
  - "Video-MLLM"
  - "免训练推理"
  - "MoE启发"
  - "注意力机制"
---

# Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference

**会议**: ICCV 2025  
**arXiv**: [2508.02134](https://arxiv.org/abs/2508.02134)  
**代码**: [github.com/wkfdb/Free-MoRef](https://github.com/wkfdb/Free-MoRef)  
**领域**: 视频理解 / 多模态大模型  
**关键词**: 长视频理解, Video-MLLM, 免训练推理, MoE启发, 注意力机制

## 一句话总结

提出免训练方法Free-MoRef，受MoE启发将长视频token分割为多个短序列作为多参考(multi-reference)，通过MoRef注意力机制并行查询并融合统一激活值，在单卡A100上实现2×到8×更长帧输入的高效全面理解，在VideoMME/MLVU/LongVideoBench上超越专训长视频模型。

## 研究背景与动机

Video-MLLM在视频理解任务上取得显著进展，但受限于底层LLM的上下文长度限制，长视频场景表现不佳。现有解决方案各有缺陷：

**Token压缩**：减少视觉token数量以容纳更多帧，但压缩率越高信息损失越严重

**流式推理**：保留历史KV-CACHE实现超长上下文依赖，但延迟与上下文长度成正比（2×上下文=2×延迟）

**上下文扩展**：后训练延长上下文窗口，计算负担大

核心问题：能否在单次推理中实现更长上下文感知，同时保证全面理解和高效推理？

## 方法详解

### 整体框架

Free-MoRef工作流程：
1. **多参考分割**：将长视频token序列按时间维度分割为N个短块(chunk)，每个chunk代表原视频的一个抽象
2. **MoRef注意力**：在浅层decoder中，用相同问题并行查询各chunk并融合统一响应
3. **参考融合**：在深层decoder中间层，基于注意力权重选取关键视觉token合并为全局参考

### 关键设计

1. **多参考分割(Multi-Reference Partition)**：首先将视频token按时间关系分为M个单元(unit)，每个单元再分为N个片段(fragment)，聚合不同单元的片段组成N个参考chunk。M控制参考间的时间交叉程度：M=1时N个chunk时间互不重叠，M越大交叉越显著。每个chunk附加相同的系统提示和问题，形成并行推理序列。

2. **MoRef注意力(Mixture of Reference Attention)**：核心步骤。对并行chunk执行Flash Attention获得初始结果 $O = [O^{sys}, O^{vis}, O^{ques}]$。由于因果注意力，$O^{sys}$ 各chunk相同，但 $O^{vis}$ 和 $O^{ques}$ 因视觉参考不同而有差异。保持 $O^{vis}$ 的差异性，通过门控加权融合 $O^{ques}$：

$$O^{fusion} = (\sum_{i=1}^N \omega_i \cdot O_i^{ques}).repeat(N)$$

门控权重 $\omega_i$ 由query-vision跨模态注意力图计算：$\omega_i = \frac{max(A[i])}{\sum max(A[i])}$，其中 $A = softmax(Q^{ques} \times (K^{vis})^T)$，捕获了query与各reference的相关度。这样每层decoder中所有视觉token都有效参与了query的更新。

3. **参考融合(Reference Fusion)**：基于FastV的观察——视觉token在浅层decoder贡献均匀，但在深层注意力更集中于query token。利用此特性，在第L层进行合并：基于注意力图 $A$ 评估每个视觉token的重要性矩阵 $E$，在每个chunk中剪枝 $1-1/N$ 的不重要token，按时间关系聚合剩余token为全局参考。此步骤补偿了MoRef注意力中缺失的跨chunk视觉交互。

### 损失函数 / 训练策略

**完全免训练**。所有设计直接在推理阶段应用，无需额外训练或微调。

- 基础模型：LLaVA-Video-7B（默认最大64帧）
- 帧输入倍增：128(2×)、256(4×)、512(8×)
- M=64个时间单元，N=帧数/64
- 参考融合层：N=2时L=3，N=4时L=6，N=8时L=12
- 支持Flash-Attention，可与流式推理和token压缩策略组合

## 实验关键数据

### 主实验

**不同帧数下的性能对比**：

| 上下文 | FLOPs | MLVU | VideoMME (Medium/Long/Overall) | LongVideoBench |
|--------|-------|------|-------------------------------|----------------|
| 64帧 (基线) | 100% | 70.3 | 62.1/53.4/64.3 | 58.8 |
| 128帧 (原生) | 400% | 70.2 | 63.2/54.1/64.9 | 58.7 |
| 128帧@MoRef | **110.4%** | **70.8** | **65.8/55.8/66.3** | **59.3** |
| 256帧 (原生) | 1600% | 67.2 | 61.4/54.1/63.1 | 56.7 |
| 256帧@MoRef | **163.2%** | **72.5** | **66.4/55.3/66.3** | **59.3** |
| 512帧@MoRef | 400% | **72.8** | **67.3/56.0/66.9** | **59.9** |

512帧原生推理直接OOM或性能暴跌；Free-MoRef用400% FLOPs（而非6400%）实现最佳性能。

**与其他7B-8B模型对比**：

| 方法 | MLVU | LVideoBench | VideoMME Long | VideoMME Overall |
|------|------|-------------|---------------|-----------------|
| LLaVA-Video | 70.2 | 58.2 | 53.4 | 64.3 |
| Qwen2-VL | 64.8 | 55.6 | 55.7 | 63.3 |
| InternVL2.5 | 68.4 | 57.5 | 53.0 | 64.5 |
| Video-XL | 64.9 | 50.7 | - | 55.5 |
| RETAKE | 69.8 | - | 56.2 | 63.9 |
| **LLaVA-Video@MoRef** | **72.8** | **59.9** | **56.0** | **66.9** |

超越所有同规模模型，包括专门训练的长视频模型。

### 消融实验

**各组件效果**（128帧，VideoMME Overall）：

| Multi-Ref | MoRef Attn | Ref Fusion | Overall |
|-----------|------------|------------|---------|
| ✗ | ✗ | ✗ | 64.9 |
| ✗ | ✗ | ✓ | 63.9 |
| ✓ | ✗ | ✓ | 62.0 |
| ✓ | ✓ | ✗ | 65.8 |
| ✓ | ✓ | ✓ | **66.3** |

仅分割不融合反而下降；MoRef注意力是核心提升来源（+3.8）；参考融合进一步优化（+0.5）。

**并行chunk数N的影响**：

| N | FLOPs | Overall |
|---|-------|---------|
| 1 (默认) | 100% | 64.9 |
| 2 | 27.6% | **66.3** |
| 4 | 25% | 66.1 |
| 8 | 23.6% | 65.9 |

N=2时效果最佳且计算量仅为原来的27.6%。

### 关键发现

- Free-MoRef核心优势在于MoRef注意力实现了全视觉token的有效参与，等效于全注意力但计算量大幅降低
- 时间单元M影响时间感知(TP)和空间感知(SP)任务的平衡：M小时SP好但TP差，M大时反之
- 参考融合层L的选择影响性能：过早融合导致信息丢失，过晚则跨chunk交互补偿不及时
- 在VideoMME的几乎所有问题类型上都有提升，唯一例外是属性感知任务（这类问题只涉及视频小片段，扩展上下文反而引入冗余）

## 亮点与洞察

- **免训练的即插即用设计**：无需任何训练或额外参数，直接提升现有Video-MLLM的长视频能力
- **MoE思路的创新迁移**：把"多专家处理不同数据"转化为"同一模型查询不同参考视频片段"
- **计算效率惊人**：8×帧数仅需27.6%-400%原始FLOPs，单卡A100可处理1024帧
- **兼容性好**：支持Flash-Attention，可与流式推理或token压缩方案叠加

## 局限与展望

- 多参考分割打断了跨chunk的视觉特征连续性，参考融合仅部分补偿
- 超参数(M, N, L)需要手动配置，缺乏自适应机制
- 仅在LLaVA-Video-7B上验证，未测试更大模型或其他架构
- 属性感知类任务(AP)反而因上下文扩展而轻微下降
- MoRef注意力设计可能启发训练时的长上下文学习方案，但本文未探索

## 相关工作与启发

- FastV揭示了LLM浅层和深层对视觉token处理方式的差异，启发了参考融合的时机选择
- MoE的混合专家范式被巧妙类比为混合参考
- 流式推理(如INF-MLLM)和token压缩(如PruneVid)与Free-MoRef正交可组合
- LLaVA-Video的默认64帧限制凸显了帧数与上下文长度的矛盾

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 免训练方法创新性强，MoRef注意力设计巧妙，MoE到多参考的类比有启发
- **实验充分度**: ⭐⭐⭐⭐ 三个长视频benchmark全面评测，消融详细，但仅限单一基础模型
- **写作质量**: ⭐⭐⭐⭐ 框架图清晰，注意力模式可视化有说服力
- **价值**: ⭐⭐⭐⭐⭐ 实用价值极高，任何Video-MLLM都可即时获益，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](../../ACL2025/multimodal_vlm/activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[ACL 2025\] Inference Compute-Optimal Video Vision Language Models](../../ACL2025/multimodal_vlm/inference_compute_optimal_video_vlm.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](../../NeurIPS2025/multimodal_vlm/training-free_online_video_step_grounding.md)
- [\[CVPR 2026\] Benchmarking Single-Factor Physical Video-to-Audio Generation](../../CVPR2026/multimodal_vlm/benchmarking_single-factor_physical_video-to-audio_generation.md)

</div>

<!-- RELATED:END -->
