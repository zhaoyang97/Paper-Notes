---
title: >-
  [论文解读] UniTok: A Unified Tokenizer for Visual Generation and Understanding
description: >-
  [NeurIPS 2025 Spotlight][多模态VLM][统一tokenizer] 提出 UniTok，一种统一视觉生成和理解的tokenizer，通过多码本量化（MCQ）突破离散token表示容量瓶颈，在ImageNet上实现0.38 rFID和78.6%零样本精度的双项记录，并可无缝集成到MLLM中同时启用生成和理解能力。
tags:
  - "NeurIPS 2025 Spotlight"
  - "多模态VLM"
  - "统一tokenizer"
  - "VQVAE"
  - "CLIP"
  - "多码本量化"
  - "视觉生成与理解"
---

# UniTok: A Unified Tokenizer for Visual Generation and Understanding

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2502.20321](https://arxiv.org/abs/2502.20321)  
**代码**: [GitHub](https://github.com/FoundationVision/UniTok)  
**领域**: 多模态VLM  
**关键词**: 统一tokenizer, VQVAE, CLIP, 多码本量化, 视觉生成与理解

## 一句话总结

提出 UniTok，一种统一视觉生成和理解的tokenizer，通过多码本量化（MCQ）突破离散token表示容量瓶颈，在ImageNet上实现0.38 rFID和78.6%零样本精度的双项记录，并可无缝集成到MLLM中同时启用生成和理解能力。

## 研究背景与动机

统一多模态大语言模型（如GPT-4o）需要一个同时适用于视觉生成和理解的tokenizer，但现有方案面临根本性矛盾：CLIP tokenizer在理解任务上出色但输出连续高维特征，不适合自回归生成；VQVAE tokenizer适合离散生成但缺乏语义捕捉能力，理解性能差。

直觉上的解决方案是将CLIP监督融入VQVAE训练，但实际操作中会出现严重的收敛问题，理解性能大幅落后于CLIP基线。**先前的研究将此归因于语义损失和像素级重建损失之间的冲突**——但这个结论是否正确？

作者进行了系统分析，得到了一个**反直觉的发现**：重建监督和语义监督本身并不冲突。性能退化的真正原因是**离散token空间的表示容量不足**。具体来说，两个关键操作导致了信息瓶颈：

**Token下投影（Factorization）**：将768维特征压缩到16维用于码本索引查找，严重损害了token的表达力

**离散化（Discretization）**：将连续特征映射到小型码本（通常4K-16K个条目），导致大量信息丢失

这与语言tokenizer的20万+词汇量形成鲜明对比。

## 方法详解

### 整体框架

UniTok联合使用VQVAE重建损失和CLIP对比损失训练。图像经编码器得到连续特征后，通过多码本量化机制将特征分块并在多个子码本中独立量化，再经解码器重建图像。同时，量化后的特征通过池化用于图像-文本对比学习。

### 关键设计

1. **多码本量化（Multi-Codebook Quantization, MCQ）**：核心创新。将latent向量 $f \in \mathbb{R}^d$ 均匀分为 $n$ 个chunk $\{f_1, f_2, \ldots, f_n\}$，每个chunk在独立的子码本中进行量化：

    $\hat{f} = \text{Concat}(\mathcal{Q}(Z_1, f_1), \mathcal{Q}(Z_2, f_2), \ldots, \mathcal{Q}(Z_n, f_n))$

   关键优势是词汇量指数级增长：将子码本数从1增到4（每个16K条目），理论词汇量从 $2^{14}$ 暴增到 $2^{56}$。与此同时，latent维度也随子码本数线性增长（如从16维到64维），进一步增强表示容量。每个子码本规模不变，因此避免了大码本的优化问题（低利用率、死码本）。

   MCQ与残差量化（RQ）的关键区别：RQ采用由粗到细的定量顺序，MCQ采用分治策略。在高维latent空间中MCQ优势巨大——64维latent下MCQ的量化损失比RQ低15-45倍。

2. **注意力投影（Attention Projection）**：替代传统的线性/卷积投影层用于token下投影。将多头注意力中拼接多头输出的操作换为平均池化来实现通道压缩。尽管简单，但有效增强了下投影后token的表达力，并稳定了训练。

3. **统一MLLM集成**：基于Liquid框架，图像被编码为 $H \times W \times K$ 个离散码（K是子码本数）。通过MLP投射器将UniTok码嵌入映射到MLLM token空间。输入时每K个连续码合并为一个视觉token；生成时每个token自回归预测下一组K个码，使用深度transformer头。

### 损失函数 / 训练策略

统一损失函数：$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{contra}} \mathcal{L}_{\text{contra}}$

其中重建损失 $\mathcal{L}_{\text{recon}} = \mathcal{L}_R + \lambda_{VQ}\mathcal{L}_{VQ} + \lambda_P\mathcal{L}_P + \lambda_G\mathcal{L}_G$，包括像素重建、VQ量化、感知（LPIPS）和对抗损失。对比损失 $\lambda_{\text{contra}}=1$。

- 架构：ViTamin-L/16混合架构，8个子码本×4096条目×8维
- 判别器用DINOv2-S初始化
- 在DataComp-1B（12.8亿图文对）上训练1 epoch，图像256×256
- MLLM用Llama-2-7B，预训练70M数据，微调3M数据

## 实验关键数据

### 主实验

**ImageNet 重建FID与零样本分类精度**

| 方法 | 类型 | Token数 | rFID↓ | 零样本精度 |
|------|------|---------|-------|-----------|
| VQGAN | VQVAE | 256 | 4.98 | — |
| VAR | VQVAE | 680 | 0.90 | — |
| CLIP | CLIP | 256 | — | 76.2% |
| SigLIP | CLIP | 256 | — | 80.5% |
| VILA-U | 统一 | 256 | 1.80 | 73.3% |
| **UniTok** | **统一** | 256 | **0.38** | **78.6%** |

**统一MLLM VQA基准对比**

| 方法 | LLM | Token类型 | VQAv2 | GQA | TextVQA | MME |
|------|-----|-----------|-------|-----|---------|-----|
| Chameleon | 34B | 离散 | 69.6 | — | — | — |
| Liquid | Gemma-7B | 离散 | 71.3 | 58.4 | 42.4 | 1119 |
| VILA-U | Llama-2-7B | 离散 | 75.3 | 58.3 | 48.3 | 1336 |
| **UniTok** | Llama-2-7B | 离散 | **76.8** | **61.1** | **51.6** | **1448** |

### 消融实验

| 监督方式 | rFID↓ | gFID↓ | VQAv2 | TextVQA | MME |
|---------|-------|-------|-------|---------|-----|
| 仅对比 | — | — | 68.95 | 49.89 | 1373 |
| 仅重建 | 0.82 | 3.59 | 56.33 | 43.65 | 902 |
| 重建+对比 | 0.72 | 3.26 | 69.14 | 49.22 | 1333 |

**MCQ vs RQ 对比（64维latent）**

| 量化方法 | 码形状 | rFID↓ | 零样本精度 |
|---------|--------|-------|-----------|
| RQ | 16×16×8 | 3.46 | 58.8% |
| **MCQ** | 16×16×8 | **0.55** | **63.7%** |

**子码本数量消融**

| 码本配置 / 词汇量 | rFID↓ | 零样本精度 |
|-------------------|-------|-----------|
| 1×16384 / $2^{14}$ | 1.50 | 41.0% |
| 2×8192 / $2^{26}$ | 0.98 | 43.9% |
| 4×4096 / $2^{48}$ | 0.54 | 44.7% |
| 8×2048 / $2^{88}$ | 0.33 | 46.1% |

### 关键发现

- 重建监督和语义监督**不内在冲突**——联合训练的理解性能与纯CLIP训练相当（69.14 vs 68.95 VQAv2），rFID甚至优于纯重建（0.72 vs 0.82）
- 性能瓶颈在**离散化**而非损失冲突——从CLIP到VQ-CLIP，下投影和离散化分别导致了显著的VQA性能下降
- MCQ在64维高维latent空间中量化误差比RQ低15-45倍
- 增加子码本数量对重建和理解性能均持续有益，且不会引发码本利用率问题
- CFG-free生成中，UniTok将gFID从14.6降到2.5，说明语义监督产生了更结构化的码分布

## 亮点与洞察

- **核心洞察**：统一tokenizer的瓶颈不是"生成vs理解的损失冲突"，而是离散token空间表示容量不足。这个发现纠正了领域内的普遍误解
- MCQ的分治策略非常优雅——将大码本问题分解为多个小码本问题，词汇量指数增长且避免优化问题
- 一个令人惊讶的发现：CLIP权重初始化对下游VQA未必有益（从头训练的UniTok在LLaVA框架下反而更好），暗示统一视觉特征空间与CLIP特征空间可能存在根本差异
- CFG-free生成的大幅提升（14.6→2.5 gFID）验证了语义监督对latent空间结构化的积极作用，与扩散模型社区的发现一致

## 局限与展望

- 计算资源限制下仅训练1 epoch，语义表征学习未收敛，全量训练可能进一步提升理解性能
- 仅在256×256分辨率训练，高分辨率下的表现有待验证
- MLLM实验基于Llama-2-7B，新一代LLM可能释放更大潜力
- 子码本在MLLM中的KV码合并方案较为简单粗暴，可能有更优的多码融合策略
- 生成质量仍与顶级扩散模型有差距（GenEval Overall 0.59 vs DALL-E 3 0.67）

## 相关工作与启发

- VILA-U 首次尝试统一tokenizer但受限于单码本，理解性能不佳
- Liquid 提出了统一生成-理解的MLLM框架，UniTok在其上用更好的tokenizer实现显著提升
- LlamaGen 提供了自回归图像生成的框架基线
- 启发：分治策略（MCQ）在表示学习中的成功可推广到其他需要离散化的场景（如音频token化、视频token化）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 纠正了领域内的关键误解，MCQ机制设计精巧
- 实验充分度: ⭐⭐⭐⭐⭐ 从tokenizer到MLLM再到生成全链路评估，消融极其详尽
- 写作质量: ⭐⭐⭐⭐⭐ 分析层层递进，从CLIP到VQ-CLIP到UniTok的roadmap说理清晰
- 价值: ⭐⭐⭐⭐⭐ 统一tokenizer是多模态统一建模的关键基础设施，0.38 rFID刷新SOTA

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Rosetta Stone for Unified MLLMs: A Unified Tokenizer to Decipher Understanding and Generation](../../CVPR2026/multimodal_vlm/rosetta_stone_for_unified_mllms_a_unified_tokenizer_to_decipher_understanding_an.md)
- [\[ICCV 2025\] Harmonizing Visual Representations for Unified Multimodal Understanding and Generation](../../ICCV2025/multimodal_vlm/harmonizing_visual_representations_for_unified_multimodal_un.md)
- [\[CVPR 2026\] AToken: A Unified Tokenizer for Vision](../../CVPR2026/multimodal_vlm/atoken_a_unified_tokenizer_for_vision.md)
- [\[ICCV 2025\] Unified Multimodal Understanding via Byte-Pair Visual Encoding](../../ICCV2025/multimodal_vlm/unified_multimodal_understanding_via_byte-pair_visual_encoding.md)
- [\[NeurIPS 2025\] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
