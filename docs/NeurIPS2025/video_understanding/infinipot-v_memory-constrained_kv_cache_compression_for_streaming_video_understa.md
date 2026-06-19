---
title: >-
  [论文解读] InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding
description: >-
  [NeurIPS 2025][视频理解][KV Cache压缩] 提出首个无需训练、查询无关的流式视频理解框架InfiniPot-V，通过时序冗余度（TaR）和值范数（VaN）两个度量实现KV缓存的在线压缩，在固定内存约束下支持任意长度的流式视频理解。 现代多模态大语言模型（MLLM）已经具备处理小时级长视频的推理能力…
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "KV Cache压缩"
  - "流式视频理解"
  - "多模态大模型"
  - "时序冗余"
  - "边缘部署"
---

# InfiniPot-V: Memory-Constrained KV Cache Compression for Streaming Video Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2506.15745](https://arxiv.org/abs/2506.15745)  
**代码**: [GitHub](https://github.com/aiha-lab/InfiniPot-V)  
**领域**: 视频理解/模型效率  
**关键词**: KV Cache压缩, 流式视频理解, 多模态大模型, 时序冗余, 边缘部署  

## 一句话总结

提出首个无需训练、查询无关的流式视频理解框架InfiniPot-V，通过时序冗余度（TaR）和值范数（VaN）两个度量实现KV缓存的在线压缩，在固定内存约束下支持任意长度的流式视频理解。

## 研究背景与动机

现代多模态大语言模型（MLLM）已经具备处理小时级长视频的推理能力，但随之而来的是一个严峻的实际问题：**KV缓存随视频帧数线性增长，快速超出设备内存限制**。

这一问题在以下场景尤为突出：

**移动端/边缘设备**：手机、AR眼镜、机器人等设备的显存固定且有限

**流式视频**：视频长度未知且持续增长，无法预先分配内存

**多轮对话**：用户可能在视频播放过程中多次提问，缓存需要长期维护

现有的KV缓存压缩方案存在两大局限：
- **离线假设**：大多数方法假设整个视频和用户查询在处理前可用，不适用于实时流式场景
- **内存仍随长度缩放**：即使是"压缩"方法也需要先构建完整KV缓存再压缩，内存峰值仍与视频长度成正比

InfiniPot-V的目标是：**在视频编码过程中在线压缩，强制维持一个与视频长度无关的固定内存上限**。

## 方法详解

### 整体框架

InfiniPot-V采用分块处理（Block-wise Processing）策略：

1. 将输入视频按帧分组为固定大小的块（block），逐块编码
2. 每编码完一个块后，检查当前KV缓存大小是否超过用户设定的阈值
3. 若超过阈值，启动轻量压缩过程，将缓存降至预算以下
4. 压缩过程结合两个互补的度量：时序冗余度（TaR）和值范数（VaN）

整个过程**无需训练**、**无需知道用户查询**，是完全即插即用的。

### 关键设计

**1. Temporal-axis Redundancy (TaR) 时序冗余度**

TaR度量用于识别和移除时间维度上冗余的token：
- 计算相邻帧对应位置的KV向量之间的余弦相似度
- 高相似度表示该位置在时间上变化小（如静态背景），可以安全移除
- 公式：$\text{TaR}(i) = \text{CosSim}(\mathbf{k}_i^{(t)}, \mathbf{k}_i^{(t-1)})$
- 移除TaR分数最高（最冗余）的token

直觉：视频中大量token对应静态背景或缓慢变化的区域，这些token在时间上高度冗余。

**2. Value-Norm (VaN) 值范数排序**

VaN度量用于保留语义最重要的token：
- 计算每个token的Value向量的L2范数
- 较大的值范数通常对应语义更显著的内容（活动的物体、关键动作等）
- 保留VaN分数最高的token

直觉：Value范数大的token在注意力聚合时对输出贡献更大。

**3. 两阶段压缩流程**

- 第一阶段（TaR过滤）：移除时间冗余度最高的 $r_1\%$ token
- 第二阶段（VaN排序）：在剩余token中保留VaN最高的 $r_2\%$ token
- 两阶段协作：TaR去冗余 + VaN保重要，实现精准压缩

### 损失函数 / 训练策略

- **无需训练**：InfiniPot-V完全基于统计度量，不引入任何可学习参数
- 无需微调底层MLLM，直接在推理时应用
- 兼容多种开源MLLM（Qwen2-VL、Qwen2.5-VL等）

## 实验关键数据

### 主实验：长视频理解基准

在Qwen2.5-VL-7B模型上，使用InfiniPot-V将KV缓存从~32K token压缩到~4K token的效果：

| 基准 | Full Cache | Uniform压缩 | SWA | InfiniPot-V | 压缩比 |
|------|-----------|-------------|-----|-------------|--------|
| MLVU | 70.2 | 64.8 | 66.1 | 69.5 | 8× |
| Video-MME | 63.4 | 58.2 | 59.7 | 62.8 | 8× |
| LongVideoBench | 55.1 | 49.6 | 51.3 | 54.7 | 8× |
| EgoSchema | 67.8 | 61.5 | 63.2 | 67.1 | 8× |

InfiniPot-V在8倍压缩下仅损失0.5-1.0分，显著优于均匀采样和滑动窗口注意力。

### 跨模型泛化实验

在不同MLLM上的表现（MLVU基准，8倍压缩）：

| 模型 | Full Cache | InfiniPot-V | 精度保持率 |
|------|-----------|-------------|-----------|
| Qwen2-VL-7B | 65.3 | 64.1 | 98.2% |
| Qwen2.5-VL-7B | 70.2 | 69.5 | 99.0% |
| Qwen2-VL-72B | 78.1 | 77.3 | 99.0% |
| Qwen2.5-VL-72B | 80.5 | 79.8 | 99.1% |

模型越大，InfiniPot-V的精度保持率越高。

### 消融实验

**压缩组件有效性**：

| 配置 | MLVU | Video-MME |
|------|------|-----------|
| Full Cache（无压缩） | 70.2 | 63.4 |
| 仅TaR | 67.2 | 60.8 |
| 仅VaN | 66.8 | 60.1 |
| TaR + VaN（InfiniPot-V） | 69.5 | 62.8 |

两个度量互补：TaR擅长去除背景冗余，VaN擅长保留前景重要信息。

**内存节省效率**：

| 配置 | 峰值GPU内存 | 相对Full Cache |
|------|------------|----------------|
| Full Cache (768帧) | ~48 GB | 100% |
| InfiniPot-V (4K预算) | ~3 GB | 6% (节省94%) |
| InfiniPot-V (8K预算) | ~5 GB | 10% (节省90%) |

### 关键发现

1. **GPU内存最高可降低94%**：从48GB降至3GB，使小时级视频处理可在消费级GPU上运行
2. **精度几乎无损**：在多个基准上精度保持率超过98%，部分场景甚至超越Full Cache
3. **实时生成速度**：压缩后的推理速度不低于Full Cache，因token数更少每步计算更快
4. **多轮对话支持**：固定内存预算下可持续接收新帧和新查询，天然支持流式多轮交互

## 亮点与洞察

1. **第一个真正的流式方案**：之前所有的"长视频理解"方法都是"先全读再处理"，InfiniPot-V真正实现了逐块处理、固定内存
2. **无需训练的优雅设计**：TaR和VaN两个度量直觉清晰、计算轻量，无需任何额外训练
3. **视频特性的巧妙利用**：TaR利用了视频固有的时序冗余性，这是视频相比文本/图像独特的可压缩性来源
4. **工业部署友好**：即插即用、跨模型泛化、固定内存预算——非常适合边缘设备部署

## 局限与展望

1. 当前TaR基于相邻帧比较，对于快速场景切换可能误删关键token
2. VaN的语义显著性假设可能不总是成立（Value范数大不一定语义重要）
3. 固定的压缩比例（$r_1$, $r_2$）未能根据视频内容复杂度自适应调整
4. 仅在Qwen系列模型上验证，对其他架构（如LLaVA系列）的泛化性待确认
5. 未评估在极端场景（如监控视频数小时无变化后突然出现异常）的鲁棒性

## 相关工作与启发

- **StreamingLLM**：通过保留attention sink和最近token实现流式推理，但不考虑语义重要性
- **FastV / FreeVideoLLM**：token剪枝方法，但不是为流式设计
- **KVzip**：查询感知的KV缓存压缩，但需要知道查询内容
- **LiveVLM**：流式视频理解的同期工作，采用retrieval-based方法
- 启发：TaR度量可推广到其他时序数据（音频、传感器流）的token压缩

## 评分

- 新颖性：⭐⭐⭐⭐（首个流式+固定内存的方案）
- 技术深度：⭐⭐⭐⭐
- 实验充分度：⭐⭐⭐⭐
- 实用性：⭐⭐⭐⭐⭐（直接可部署）
- 写作质量：⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MuKV: Multi-Grained KV Cache Compression for Long Streaming Video Question-Answering](../../CVPR2026/video_understanding/mukv_multi-grained_kv_cache_compression_for_long_streaming_video_question-answer.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[CVPR 2026\] FluxMem: Adaptive Hierarchical Memory for Streaming Video Understanding](../../CVPR2026/video_understanding/fluxmem_adaptive_hierarchical_memory_for_streaming_video_understanding.md)
- [\[ICCV 2025\] VideoLLaMB: Long Streaming Video Understanding with Recurrent Memory Bridges](../../ICCV2025/video_understanding/videollamb_long_streaming_video_understanding_with_recurrent_memory_bridges.md)
- [\[NeurIPS 2025\] INST-IT: Boosting Instance Understanding via Explicit Visual Prompt Instruction Tuning](inst-it_boosting_instance_understanding_via_explicit_visual_prompt_instruction_t.md)

</div>

<!-- RELATED:END -->
