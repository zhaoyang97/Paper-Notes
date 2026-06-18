---
title: >-
  [论文解读] ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers
description: >-
  [ICCV 2025][多模态VLM][多模态大模型] 发现 MLLM 中存在显著的**层级冗余**——多数层对视觉 token 的变换贡献极小，据此提出 ShortV：在约 60% 的层中冻结视觉 token（跳过其注意力和 FFN 计算），在 LLaVA-NeXT-13B 上实现 50% FLOPs 减少，性能几乎无损。方法免训练，且与 token 剪枝方法正交可叠加。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "多模态大模型"
  - "推理效率"
  - "层冗余"
  - "视觉token冻结"
  - "免训练"
  - "MLLM加速"
---

# ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers

**会议**: ICCV 2025  
**代码**: [https://github.com/icip-cas/ShortV](https://github.com/icip-cas/ShortV)  
**领域**: 多模态VLM  
**关键词**: 多模态大模型, 推理效率, 层冗余, 视觉token冻结, 免训练, MLLM加速

## 一句话总结

发现 MLLM 中存在显著的**层级冗余**——多数层对视觉 token 的变换贡献极小，据此提出 ShortV：在约 60% 的层中冻结视觉 token（跳过其注意力和 FFN 计算），在 LLaVA-NeXT-13B 上实现 50% FLOPs 减少，性能几乎无损。方法免训练，且与 token 剪枝方法正交可叠加。

## 研究背景与动机

MLLM（如 LLaVA-1.5、LLaVA-NeXT）的计算开销主要来自两个方面：

**LLM 主干规模大**：7B-13B 参数

**视觉 token 序列长**：单张图像 576-2880 个 token

**已有工作的角度**：FastV 等方法从 **token 维度**发现冗余——许多视觉 token 不重要可以裁剪。但这些方法减少的是 token 数量。

**本文的新视角**：从 **层维度** 研究冗余——同一个视觉 token 是否真的需要在每一层都被更新？

**关键发现**：文本 LLM 中约 25% 的层对文本 token 是低效的。但对于 MLLM 中的视觉 token，由于存在**模态鸿沟**（modality gap），视觉和文本 token 在嵌入空间中的分布差异大，MLLM 可能对两种模态采取截然不同的处理策略。

## 方法详解

### Layer Contribution (LC) 指标

为了量化某一层对特定类型 token 的贡献，提出 LC 指标。具体做法：

1. 将第 $i$ 层替换为**稀疏层**，其中目标 token（如视觉 token）被冻结——即保持其隐藏状态不变
2. 在稀疏层中，被冻结的 token 不参与 query 计算且跳过 FFN
3. 计算模型输出的 KL 散度来衡量冻结的影响：

$$LC_i^X = KL(logits(M), logits(\mathcal{M}_i^X))$$

其中 $M$ 是原始模型，$\mathcal{M}_i^X$ 是第 $i$ 层中 token $X$ 被冻结后的模型。LC 越低 → 该层对该类 token 越不重要。

### 为什么不用其他指标？

- **困惑度**：不适合视觉 token，因为即使完全移除视觉 token，MLLM 仍能生成合理文本（困惑度变化小），但视觉任务性能严重下降
- **余弦相似度**：只看层输入输出的相似度，忽略了层在模型中的位置——浅层的微小变换会影响所有后续层，深层则不然。实验证实余弦相似度一致性地高估浅层冗余、低估深层冗余

### ShortV 方法

1. 构建小型数据集（2000 个样本，来自 Flickr30K + GQA）
2. 计算每层对视觉 token 的 LC 分数
3. 按 LC 分数升序排列，选择最低的 $N$ 层
4. 将这些层替换为 ShortV 层（视觉 token 冻结）

**ShortV 层的具体设计**（如 Figure 4 所示）：
- 只有**文本 token** 通过 $W_Q$ 和 $W_O$ 矩阵及 FFN
- 视觉 token 不作为 query（不 attend to 其他 token）
- 视觉 token 不通过 FFN
- 视觉 token 的 key/value 仍参与计算（文本 token 可以注意到视觉 token）

### FLOPs 计算

原始层：$FLOPs = 2(t+v)(4h+3m)h + 4(t+v)^2 h$

ShortV 层：$FLOPs^* = 2t(4h+3m)h + 4vh^2 + 4t(t+v)h$

整体 FLOPs 比例：$r = \frac{(L-N) \times FLOPs + N \times FLOPs^*}{L \times FLOPs}$

## 实验关键数据

### 主实验：与其他免训练方法对比

| 模型 | 方法 | TFLOPs | FLOPs 比例 | MME ↑ | MMBench ↑ | MMMU ↑ | MMStar ↑ | SEED ↑ | GQA ↑ | Flickr30K ↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | Vanilla | 8.5 | 100% | 1510.7 | 64.1 | 36.3 | 33.7 | 66.1 | 61.9 | 74.9 |
| LLaVA-1.5-7B | FastV (K=2,R=50%) | 4.9 | 58% | 1475.6 | 64.3 | 35.8 | 32.4 | 65.4 | 60.2 | 67.5 |
| LLaVA-1.5-7B | VTW (K=16) | 4.7 | 55% | 1497.0 | 64.0 | 36.1 | 32.8 | 66.2 | 55.1 | 44.5 |
| LLaVA-1.5-7B | **ShortV (N=19)** | **4.7** | **55%** | **1503.1** | **64.8** | **36.2** | **33.3** | **66.2** | **60.9** | **71.3** |
| LLaVA-NeXT-13B | Vanilla | 81.8 | 100% | 1570.0 | 69.3 | 35.9 | 39.9 | 71.9 | 65.7 | 66.7 |
| LLaVA-NeXT-13B | FastV (K=2,R=50%) | 42.1 | 51% | 1546.4 | 68.5 | 35.9 | 39.6 | 71.5 | 62.9 | 66.0 |
| LLaVA-NeXT-13B | VTW (K=20) | 41.7 | 51% | 1569.4 | 69.1 | 34.8 | 39.8 | 71.8 | 61.5 | 56.6 |
| LLaVA-NeXT-13B | **ShortV (N=24)** | **41.0** | **50%** | **1553.0** | **70.2** | **36.2** | **39.9** | **71.8** | **63.6** | **67.5** |

ShortV 在所有模型上性能均优于或持平其他方法，而计算量相当或更低。VTW 在 Flickr30K 上严重退化（44.5-56.6），ShortV 几乎无损。

### 消融：不同层选择策略

| 策略 | FLOPs 比例 | MMBench | MMMU | SEED-Bench | GQA |
|---|---|---|---|---|---|
| Vanilla | 100% | 64.0 | 36.3 | 66.1 | 61.9 |
| Random | 55% | 58.4 | 33.6 | 60.5 | 56.1 |
| Cosine Similarity | 55% | 60.8 | 34.2 | 62.7 | 59.5 |
| **LC (Ours)** | **55%** | **64.8** | **36.2** | **66.2** | **60.9** |

LC 指标远优于随机选择和余弦相似度——后者虽优于随机但仍无法匹配原始性能。

### 消融：冻结不同类型 token

| 冻结 Token | MMBench | MMMU | SEED-Bench | GQA |
|---|---|---|---|---|
| None (Vanilla) | 64.0 | 36.3 | 66.1 | 61.9 |
| Text token | 2.1 | 23.7 | 8.9 | 2.9 |
| All token | 1.3 | 26.6 | 0.8 | 0.0 |
| Random token | 1.5 | 22.9 | 5.5 | 2.3 |
| **Visual token (Ours)** | **64.8** | **36.2** | **66.2** | **60.9** |

冻结文本 token 导致性能**崩溃**（GQA 从 61.9 → 2.9），但冻结视觉 token 几乎无影响——强有力地证明了 MLLM 层对视觉和文本 token 的处理确实存在根本差异。

### ShortV + FastV 兼容性

| 方法 | FLOPs 比例 | MMBench | MMMU | SEED-Bench | GQA |
|---|---|---|---|---|---|
| Vanilla | 100% | 64.0 | 36.3 | 66.1 | 61.9 |
| FastV | 58% | 64.3 | 35.8 | 65.4 | 60.2 |
| ShortV | 55% | 64.8 | 36.2 | 66.2 | 60.9 |
| **ShortV + FastV** | **29%** | **64.2** | **37.1** | **65.1** | **59.3** |

两种方法叠加后 FLOPs 降至 29%，性能仍保持在合理范围。

### 关键发现

1. **60% 的层可以冻结视觉 token 而不降低性能**，远超 LLM 中文本 token 的 ~25%
2. 实际加速比：LLaVA-NeXT-13B 上 ShortV(N=24) 达到 **1.52× 加速**，FastV 仅 1.31×
3. 80% 层冻结后仍保持 >90% 性能

## 亮点与洞察

1. **层级冗余的模态不对称性**：这是一个深刻且令人意外的发现——MLLM 层对视觉 token 的冗余远大于文本 token，反映了当前 MLLM 架构对视觉信息的处理效率很低
2. **LC 指标的设计哲学**：通过直接测量冻结对输出的影响，避免了余弦相似度等代理指标的系统性偏差
3. **与 token 剪枝正交的新维度**：token 剪枝减少 token 数量，ShortV 减少每个 token 的计算量，两者可叠加到 29% FLOPs
4. **免训练无参数更新**：可即插即用地应用于任意 MLLM

## 局限性

1. **仅在 LLaVA 系列上验证**：未测试 Qwen-VL、InternVL 等其他架构
2. **视觉 token 仍保留为 KV**：虽然视觉 token 不作为 query，但其 K、V 仍参与注意力计算，这部分计算未省去
3. **LC 计算需要小数据集**：虽然只需 2000 个样本，但每换一个模型都需要重新计算
4. **仅验证了图像理解任务**：视频理解、长文档理解等更长序列场景未探索
5. **隐含假设**：假定层对视觉 token 的低效性是与输入无关的（使用固定的层选择），但不同类型的图像可能需要不同层的处理

## 相关工作与启发

- **FastV**：发现 token 级冗余并剪枝；ShortV 发现层级冗余并冻结——两个互补维度的结合暗示 MLLM 在处理视觉信息方面有巨大的效率提升空间
- **LLM 层裁剪**：Men et al. 发现 LLM 中 ~25% 层可移除，但 MLLM 对视觉 token 有 ~60% 层可冻结——模态鸿沟导致了更大的冗余
- **SAISA / NAAViT**：ShortV 层的设计灵感来源于这些在自注意力中集成多模态交叉注意力的架构
- **对 MLLM 架构设计的启示**：既然大部分层对视觉 token 不重要，是否应该从一开始就设计只在少数层处理视觉信息的架构？

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — LC 指标和层级冗余的模态不对称性发现很有价值
- **实验完整性**: ⭐⭐⭐⭐ — 消融实验全面（层选择策略、冻结对象、兼容性），多模型验证
- **实用性**: ⭐⭐⭐⭐⭐ — 免训练、实际加速 1.5×+、可与 FastV 叠加
- **写作质量**: ⭐⭐⭐⭐ — 论证逻辑清晰，从发现到方法到验证一气呵成

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] VLsI: Verbalized Layers-to-Interactions from Large to Small Vision Language Models](../../CVPR2025/multimodal_vlm/vlsi_verbalized_layers-to-interactions_from_large_to_small_vision_language_model.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)
- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)
- [\[ICCV 2025\] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers](falcon_resolving_visual_redundancy_and_fragmentation_in_high.md)
- [\[CVPR 2026\] Grounding Everything in Tokens for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/grounding_everything_in_tokens_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
