---
title: >-
  [论文解读] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance
description: >-
  [ICCV 2025][多模态VLM][视觉Token压缩] 提出 FOLDER——一种即插即用的视觉 token 压缩模块，通过系统分析信息损失的三个关键因素（压缩影响、传播效应、聚合方式），在视觉编码器的最后几层进行激进的 token 合并，实现最多 70% 的 token 削减，同时保持甚至提升模型性能。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视觉Token压缩"
  - "MLLM加速"
  - "即插即用"
  - "Token合并"
  - "推理加速"
---

# FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance

**会议**: ICCV 2025  
**arXiv**: [2501.02430](https://arxiv.org/abs/2501.02430)  
**代码**: [GitHub](https://github.com/anakin-skywalker-Joseph/Folder)  
**领域**: 多模态VLM  
**关键词**: 视觉Token压缩, MLLM加速, 即插即用, Token合并, 推理加速

## 一句话总结

提出 FOLDER——一种即插即用的视觉 token 压缩模块，通过系统分析信息损失的三个关键因素（压缩影响、传播效应、聚合方式），在视觉编码器的最后几层进行激进的 token 合并，实现最多 70% 的 token 削减，同时保持甚至提升模型性能。

## 研究背景与动机

MLLM 在多模态任务上表现强大，但视觉骨干网络产生的长序列视觉 token 带来了显著的计算挑战：

- 高分辨率 MLLM（如 LLaVA-NeXT）可能产生超过 2000 个视觉 token
- 多视觉编码器架构进一步加剧序列长度问题
- 视频理解模型（如 VideoLLaVA）即使仅 8 帧也超过 2000 token
- 注意力机制的二次计算复杂度使实时部署困难

现有方法的不足：

**训练时方案**（Q-Former、Resampler、Pooling）：性能下降明显，可扩展性差，与特定架构绑定

**推理时方案**（ToMe、Turbo）：采用各层均匀压缩的策略，未考虑信息损失的差异性，效果欠优

**FastV**：在 LLM 内部基于注意力剪枝视觉 token，但需要全程保留 KV-cache 中的视觉 token，不适合长对话场景

核心问题：**信息损失到底来自哪里？** 作者通过系统的实证分析回答了这个问题，进而设计出高效的 token 压缩策略。

## 方法详解

### 整体框架

FOLDER 作为即插即用模块，集成在视觉骨干网络的最后几个 block 中。核心思想是在信息冗余度最高、传播影响最小的位置进行激进的 token 压缩。

### 关键设计

1. **Token 压缩影响分析（Reduction Impact）**：
   利用 SVD 分解来估算保持特定能量阈值所需的最少 token 数。对 token 序列 $\mathbf{X} \in \mathbb{R}^{n \times d}$，通过 SVD 获得奇异值 $\sigma_i$，定义能量保持比：
    $E(k) = \frac{\sum_{i=1}^{k} \sigma_i}{\sum_{i=1}^{n} \sigma_i}$
   实验发现：**后面的 block 需要的 token 数远少于前面的 block**，说明后层存在大量冗余。

2. **传播效应分析（Propagation Effect）**：
   由于 Transformer 的顺序结构，早期 block 的 token 压缩误差会逐层累积放大（蝴蝶效应）。使用 Earth Mover's Distance（EMD）测量在不同 block 进行 token 压缩后对最终输出分布的影响：
    $\text{EMD}(P_Y, P_{\tilde{Y}_b}) = \min_{\gamma} \langle \gamma, \mathbf{M} \rangle_F$
   实验证明：**在早期 block 压缩导致的 EMD 远大于后期 block**。即使 75% 的压缩比，在最后几层的 EMD 也极低。综合两项分析得出结论：**token 压缩应集中在网络末端**。

3. **Token 聚合方式分析（Aggregation Method）**：
   对比了三种聚合方式——直接丢弃（$\alpha_{i_{max}}=1$, 其余为0）、平均合并（$\alpha_i = 1/m$）、加权合并（$\alpha_i = \|x_i\|_2 / \sum_j \|x_j\|_2$）。实验表明平均合并和加权合并的 EMD 远低于直接丢弃，且平均合并更简洁，效果相当，因此选用**平均合并**。

4. **FOLDER 算法**：
   基于上述分析，在最后 block 中执行二部图匹配的 token 合并。为突破传统二部图匹配最多压缩 1/2 的限制，设计了**迭代 FOLD 操作**：
    - 将 token 分为等大的集合 $\mathbb{A}$ 和 $\mathbb{B}$
    - 为 $\mathbb{A}$ 中每个 token 找到 $\mathbb{B}$ 中最相似的 token（基于匹配函数 $S$）
    - 按匹配分数排序，保留 top-$r_{\text{fold}}$ 个匹配进行合并
    - 若需要压缩数超过 $\lfloor n/2 \rfloor$（reduction overflow），自动执行多次 FOLD 迭代
    - 每次 FOLD 将 token 数减半，直到剩余压缩量可在一次 FOLD 中完成

### 损失函数 / 训练策略

FOLDER 有两种使用模式：
- **推理加速**：无需训练，直接插入预训练模型的视觉编码器，减少推理时的视觉 token 数
- **训练加速/性能增强**：在 MLLM 的预训练阶段集成 FOLDER，起到训练加速器和正则化器的双重作用

## 实验关键数据

### 主实验（LLaVA-1.5 7B 推理加速）

| 方法 | 压缩比 | 加速比 | MMBench | MME | POPE | SEEDBench | 平均 |
|------|--------|--------|---------|-----|------|-----------|------|
| Original-7B | 0% | 1× | 62.8 | 1338.9 | 79.7 | 60.2 | 55.0 |
| Pooling | 50% | 1.5× | 59.5 | 1308.6 | 84.0 | 59.2 | 53.7 |
| FastV | 50% | 1.3× | 63.3 | 1345.1 | 81.0 | 59.4 | 54.9 |
| Turbo | 50% | 1.5× | 60.4 | 1311.2 | 83.0 | 58.1 | 52.3 |
| **FOLDER** | **50%** | **1.5×** | **62.4** | **1338.2** | **85.4** | **59.5** | **55.5** |
| FastV | 66% | 1.5× | 62.5 | 1353.2 | 79.3 | 58.3 | 54.6 |
| **FOLDER** | **66%** | **1.7×** | **61.4** | **1350.0** | **85.4** | **59.7** | **54.8** |

### 消融实验（LLaVA-1.5 13B）

| 方法 | 压缩比 | 加速比 | MMBench | MME | POPE | 平均 |
|------|--------|--------|---------|-----|------|------|
| Original-13B | 0% | 1× | 66.6 | 1371.1 | 86.4 | 56.3 |
| Pooling (50%) | 50% | 1.5× | 64.1 | 1316.6 | 85.7 | 55.1 |
| FastV (50%) | 50% | 1.3× | 66.4 | 1386.3 | 85.4 | 55.9 |
| **FOLDER (50%)** | **50%** | **1.5×** | **65.4** | **1383.7** | **86.9** | **56.8** |
| **FOLDER (66%)** | **66%** | **1.6×** | **65.8** | **1366.9** | **86.1** | **56.1** |
| FOLDER+FastV (75%) | 75% | 1.8× | 65.1 | 1368.6 | 85.8 | 56.2 |

### 关键发现

- FOLDER 在 50% 压缩比下不仅不降低性能，在 POPE 和 ScienceQA 上甚至有明显提升
- 在 MiniGPT4v2 上，60% 压缩比时 FOLDER 的 MME 分数（859.9）竟大幅超过原始模型（631.4），表现出强正则化效果
- FOLDER 与 FastV 互补（一个在视觉编码器中压缩，一个在 LLM 中剪枝），组合使用可达 75% 压缩比、1.8× 加速
- 作为训练加速器，FOLDER 在所有基准上实现了性能提升，压缩比高达 70%

## 亮点与洞察

- **系统的实证分析**：通过 SVD 能量分析、EMD 传播效应、聚合方式对比三位一体地回答了 "信息损失来自哪里" 这个根本问题
- **违反直觉的发现**：虽然 SVD 能量分析建议在前几层压缩，但传播效应分析推翻了这一建议——前层压缩误差会被放大
- **激进压缩的合理性**：最后层的 token 高度冗余，允许无损的激进压缩
- **双重用途**：推理时作为加速器，训练时作为正则化器，体现了方法的灵活性

## 局限与展望

- 匹配函数的选择空间未充分探索（沿用了 ToMe 的余弦相似度）
- 多次 FOLD 迭代引入少量额外计算开销，虽然作者声称可忽略，但在极端压缩比下可能不可忽视
- 仅在 ViT 系列视觉编码器上验证，未测试 CNN 骨干或混合架构
- 视频理解场景的实验验证有限

## 相关工作与启发

- 继承了 ToMe 的二部图匹配思想，但突破了其最多 50% 的压缩限制
- 与 FastV 互补：FOLDER 在编码器端压缩，FastV 在 LLM 端剪枝
- 信息传播效应分析的方法论（利用 EMD 追踪输出分布变化）可推广到其他模型压缩场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 系统性分析框架是主要贡献，FOLD 迭代策略是巧妙的工程创新
- **实验充分度**: ⭐⭐⭐⭐⭐ 多模型（LLaVA、MiniGPT4v2）、多规模（7B/13B）、多场景（推理/训练）、详细消融
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，从"三个问题"到解决方案的叙事结构优秀
- **价值**: ⭐⭐⭐⭐⭐ 即插即用、开源、实用性强，对 MLLM 部署有直接意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ICCV 2025\] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval](bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](../../NeurIPS2025/multimodal_vlm/efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)

</div>

<!-- RELATED:END -->
