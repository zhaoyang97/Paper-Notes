---
title: >-
  [论文解读] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference
description: >-
  [ICCV 2025][多模态VLM][VLM] 提出SparseVILA——首个解耦prefill和decode阶段视觉稀疏性的VLM推理加速框架：prefill阶段进行query-agnostic的冗余token剪枝，decode阶段进行query-aware的相关token检索，实现最高4.0×prefill加速、2.5×decode吞吐提升、2.6×端到端加速，同时在多轮对话场景中保持精度（现有方法因永久删除token而在多轮中急剧退化）。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "VLM"
  - "剪枝"
  - "KV-Cache"
  - "Decoupled Sparsity"
  - "Multi-turn Conversation"
  - "Prefill-Decode"
---

# SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference

**会议**: ICCV 2025  
**arXiv**: [2510.17777](https://arxiv.org/abs/2510.17777)  
**代码**: 未开源（基于TinyChat + AWQ推理框架）  
**领域**: 多模态大模型 / 推理加速 / Token剪枝  
**关键词**: VLM, Token Pruning, KV-Cache, Decoupled Sparsity, Multi-turn Conversation, Prefill-Decode  

## 一句话总结
提出SparseVILA——首个解耦prefill和decode阶段视觉稀疏性的VLM推理加速框架：prefill阶段进行query-agnostic的冗余token剪枝，decode阶段进行query-aware的相关token检索，实现最高4.0×prefill加速、2.5×decode吞吐提升、2.6×端到端加速，同时在多轮对话场景中保持精度（现有方法因永久删除token而在多轮中急剧退化）。

## 研究背景与动机

### 问题定义
VLM（如LLaVA、Qwen2-VL）在处理高分辨率图像、长视频和多轮对话时，视觉token占据输入序列的90%-99%，主导推理延迟和显存消耗。需要在不损失精度的前提下加速推理。

### 现有方法的局限

**Query-agnostic剪枝（PruMerge/VisionZip）**：
   - 仅基于视觉显著性/冗余性剪枝，不考虑文本query
   - 高稀疏度下丢失细粒度视觉细节
   - 不能适应query-specific的信息需求

**Query-aware剪枝（FastV/SparseVLM/PDrop）**：
   - 利用query-vision注意力分数选择token
   - **多轮对话的致命问题**：为第一个query剪掉的token无法恢复，后续问题若需要这些token则精度急剧下降
   - 实验发现：即使构建"贪心最优oracle"（用GT答案选最优token子集），多轮评估性能仍然严重退化

**延迟分布不匹配**：
   - 图像任务中decode阶段占50-70%延迟
   - 视频任务中decode阶段占70-90%延迟
   - 现有方法主要优化prefill，忽略了真正的延迟瓶颈

### 核心洞察
视觉稀疏性不应在整个推理pipeline上统一施加——prefill和decode有不同的计算特性和功能需求。解耦(decouple)稀疏性可以"两全其美"：prefill保留足够覆盖度，decode针对当前query激进检索。

## 方法详解

### 整体框架
SparseVILA在推理的两个阶段分别施加不同类型的稀疏性：

**Prefill阶段**（执行一次，构建多模态上下文）：
- Query-agnostic剪枝：基于视觉编码器的自注意力估计token显著性
- 剪除冗余token，但保留足够视觉覆盖以支持后续多轮对话
- 典型稀疏度：45%-75%

**Decode阶段**（逐token生成，主导延迟）：
- Query-aware检索：从KV Cache中选取与当前query最相关的视觉token激活
- 未激活的token保留在cache中供后续轮次使用
- 典型稀疏度：75%-95%

### 关键设计1：Prefill阶段的Query-Agnostic剪枝

**Token显著性估计**：
- 对有summary token的编码器（如CLIP）：每个token对summary token的注意力贡献
- 对有多个summary token的编码器（如RADIO）：对所有summary token的平均注意力
- 对无summary token的编码器（如SigLIP/QwenVL）：所有token间的平均self-attention

**高效实现**：
自定义Triton kernel——流式计算softmax和显著性累积，无需显式构建完整注意力矩阵：
- SigLIP编码器加速3×
- QwenVL编码器加速10×

### 关键设计2：Decode阶段的Query-Aware检索

在decode开始前计算每个视觉token与当前query的相关性：
- 度量query embeddings与视觉KV Cache之间的注意力强度
- 保留得分最高的token子集用于decode attention计算
- 低得分token**不删除**——保留在cache中供后续轮次重新检索

**关键区别**：这不是永久剪枝，而是"软选择"——每轮对话可以选择不同的token子集。

**Triton kernel优化**：与FlashAttention2的prefill路径并行执行，实现1.5×加速。

### 关键设计3：位置编码处理

不同VLM使用不同的位置编码策略：
- **统一RoPE（LLaVA-NeXT等）**：保留剪枝后视觉token的连续位置索引
- **多模态RoPE（Qwen2.5-VL）**：在temporal/height/width维度重建最小连续位置网格，再平移后续文本位置

### 关键设计4：多轮评估协议

发现现有benchmark存在信息泄露问题（Q1透露Q2的答案），设计KV Cache部分驱逐策略：每轮结束后仅移除前一轮Q&A的KV条目，保留视觉KV cache。

## 实验关键数据

### 推理设置
- 量化基线：视觉编码器W8A8 (SmoothQuant) + LLM W4A16 (AWQ)，已有2.4×加速
- 所有结果在量化基线之上报告
- 硬件：单卡NVIDIA A6000

### 图像Benchmark结果（LLaVA-NeXT-7B）

| 方法 | Prefill稀疏 | Decode稀疏 | E2E加速 | AI2D | DocVQA | GQA | POPE | TextVQA |
|------|:-----------:|:----------:|:-------:|------|--------|-----|------|---------|
| 无压缩 | 0% | 0% | 1.0× | 63.9 | 63.6 | 63.5 | 84.5 | 58.2 |
| FastV | 80% | 0% | 1.2× | 61.8 | 33.5 | 55.3 | 76.7 | 52.7 |
| SparseVLM | 75% | 0% | 1.2× | 63.2 | 41.8 | 59.7 | 83.4 | 57.6 |
| VisionZip | 80% | 0% | 1.2× | 62.9 | 48.5 | 60.3 | 84.1 | 57.1 |
| **SparseVILA** | **60%** | **75%** | **1.2×** | **64.1** | **58.0** | **62.7** | **85.8** | **59.1** |

关键发现：同等加速比下，SparseVILA在DocVQA上比FastV高24.5个点，比VisionZip高9.5个点。在GQA/POPE/TextVQA上甚至超越无压缩基线。

### 视频理解Benchmark结果

| 模型(帧数) | Prefill | Decode | E2E加速 | LVB | MLVU | NExT-QA | Video-MME |
|------------|:-------:|:------:|:-------:|-----|------|---------|-----------|
| LongVILA-7B(256f) 无压缩 | - | - | 1.0× | 53.8 | 64.9 | 78.6 | 58.8 |
| + VisionZip 95% | 0.9× | 1.5× | 2.1× | 47.0 | 60.4 | 75.5 | 52.2 |
| + PruMerge 95% | 0.9× | 1.5× | 2.1× | 47.9 | 60.9 | 75.7 | 52.0 |
| **+ SparseVILA 75%/90%** | 1.0× | **1.6×** | **2.1×** | **54.1** | **65.3** | **79.0** | **58.7** |

关键发现：SparseVILA在视频任务上甚至**超越无压缩基线**（如MLVU 65.3 vs 64.9），原因是更精确的token检索让模型聚焦在语义最重要的视觉线索上。

### 解耦稀疏性消融

| Prefill稀疏 | Decode稀疏 | Prefill加速 | Decode加速 | E2E加速 | RoboVQA |
|:-----------:|:----------:|:-----------:|:----------:|:-------:|---------|
| 0% | 0% | 1.0× | 1.0× | 1.0× | 86.4 |
| 90% | 0% | 14.6× | 1.1× | 1.4× | 80.0 |
| **70%** | **85%** | **4.9×** | **1.2×** | **1.4×** | **89.1** |

关键发现：同等1.4×端到端加速下，将稀疏性从prefill(90%)重新分配到decode(70%/85%)，RoboVQA从80.0提升到89.1——比无压缩基线(86.4)还高！

### 检索token的功能分析

SparseVILA检索出的token呈现两种角色：
1. **Visual Attention Sinks**：跨query稳定激活的锚点token，维持注意力稳定性
2. **Visual Retrieval Tokens**：随query动态变化的语义相关token，捕获任务特定信息

## 亮点与洞察

1. **Prefill-Decode解耦的范式创新**：首次明确指出VLM推理的两个阶段应使用不同类型的稀疏策略——这一洞察简单但深刻，改变了"统一压缩"的惯性思维
2. **多轮对话的根本性解决**：query-aware剪枝本质上不可逆（oracle实验证明上限也很差），而SparseVILA的"软检索"设计保留了所有信息——只是每轮活性子集不同
3. **"less is more"效应**：在视频任务上，稀疏推理反而超越完整推理——类似StreamingLLM的发现，少即是多
4. **工程完整度高**：自定义Triton kernel处理显著性计算和cache紧凑打包，实测端到端加速而非理论FLOP减少
5. **RoPE兼容性**：针对统一/多模态RoPE分别设计位置重构策略，确保跨模态位置一致性

## 局限性

1. **prefill和decode使用恒定稀疏比**：未探索逐层/逐head的自适应稀疏策略，可能有进一步优化空间
2. **文档理解场景的精度下降**：虽然相比其他方法好很多，但DocVQA仍有~5.6个点的下降（58.0 vs 63.6），因为文档中每个细节都可能重要
3. **依赖视觉编码器的注意力图**：对于不提供注意力图的黑盒编码器不适用
4. **仅单GPU评测**：batch=1设置，分布式/多batch场景的效果未知
5. **量化+稀疏的组合效应**：所有结果建立在AWQ量化基线上，独立稀疏效果需额外验证

## 相关工作与启发

- 与SparseMM的互补性：SparseMM从head角度分配不对称预算，SparseVILA从阶段角度解耦稀疏——两者可结合
- **Prefill-Decode解耦**的思想可推广到LLM的一般推理加速（不限于多模态）
- 多轮评估协议（防信息泄露的KV部分驱逐）本身是方法论贡献，值得在评测框架中推广
- Visual Attention Sink + Retrieval Token的双角色发现，与VisionZip和VAR的观察一致，指向VLM注意力的某种普遍结构

## 评分 ⭐⭐⭐⭐⭐
- 创新性：⭐⭐⭐⭐⭐（prefill-decode解耦是范式级贡献，多轮对话的洞察深刻）
- 实验：⭐⭐⭐⭐⭐（图像9种benchmark+视频4种+多模型+多轮评测+端到端实测，极其完整）
- 写作：⭐⭐⭐⭐⭐（动机-方法-实验的逻辑链极为清晰，oracle实验令人信服）
- 实用性：⭐⭐⭐⭐⭐（免训练+架构无关+2.6×端到端加速，直接可集成到生产系统）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs](sparsemm_head_sparsity_emerges_from_visual_concept_responses_in_mllms.md)
- [\[CVPR 2026\] DUET-VLM: Dual Stage Unified Efficient Token Reduction for VLM Training and Inference](../../CVPR2026/multimodal_vlm/duet-vlm_dual_stage_unified_efficient_token_reduction_for_vlm_training_and_infer.md)
- [\[ICML 2025\] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](../../ICML2025/multimodal_vlm/sparsevlm_visual_token_sparsification_for_efficient_vision-language_model_infere.md)
- [\[CVPR 2026\] VLM-Pruner: Buffering for Spatial Sparsity in an Efficient VLM Centrifugal Token Pruning Paradigm](../../CVPR2026/multimodal_vlm/vlm-pruner_buffering_for_spatial_sparsity_in_an_efficient_vlm_centrifugal_token_.md)
- [\[ICCV 2025\] Sparsity Outperforms Low-Rank Projections in Few-Shot Adaptation](sparsity_outperforms_low-rank_projections_in_few-shot_adaptation.md)

</div>

<!-- RELATED:END -->
