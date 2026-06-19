---
title: >-
  [论文解读] FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers
description: >-
  [ICCV 2025][多模态VLM][视觉寄存器] 提出 FALCON，通过在 ViT 中引入可学习的视觉寄存器（Visual Register），利用 ReCompact 机制在编码阶段直接消除视觉冗余（9 倍 token 压缩），并用 ReAtten 模块通过寄存器间交互解决裁切导致的视觉碎片化问题。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "视觉寄存器"
  - "Token压缩"
  - "视觉冗余"
  - "视觉碎片化"
  - "高分辨率MLLM"
---

# FALCON: Resolving Visual Redundancy and Fragmentation in High-resolution Multimodal Large Language Models via Visual Registers

**会议**: ICCV 2025  
**arXiv**: [2501.16297](https://arxiv.org/abs/2501.16297)  
**代码**: [有](https://github.com/JiuTian-VL/JiuTian-FALCON)  
**领域**: Multimodal VLM / High-resolution Understanding  
**关键词**: 视觉寄存器, Token压缩, 视觉冗余, 视觉碎片化, 高分辨率MLLM

## 一句话总结

提出 FALCON，通过在 ViT 中引入可学习的视觉寄存器（Visual Register），利用 ReCompact 机制在编码阶段直接消除视觉冗余（9 倍 token 压缩），并用 ReAtten 模块通过寄存器间交互解决裁切导致的视觉碎片化问题。

## 研究背景与动机

高分辨率 MLLM 普遍采用裁切策略（cropping-based）：将高分辨率图像裁切为多个匹配编码器预训练分辨率的子图，独立编码后拼接。这带来两个核心问题：

1. **视觉冗余**：token 数量随分辨率急剧增长（如 16 个子图 × 576 token = 9216 token），大量来自背景区域的 token 是冗余的，严重增加 LLM 计算负担。现有压缩方法（池化/QFormer/Abstractor）要么效果有限要么需要海量预训练数据。

2. **视觉碎片化**：独立编码子图时破坏了语义连贯性。典型案例："pineapple" 被切成 "pine"+"apple" 导致 OCR 错误；"鲁宾花瓶"被切分后无法识别。

## 方法详解

### 整体框架

FALCON 基于 SigLIP-L/16-384px（视觉编码器）+ Llama-3.1-8B-Instruct（LLM），核心创新集中在视觉编码阶段：

1. 形状自适应裁切 → 获得 $N_c$ 个子图 + 全局缩略图
2. 每个子图的 image token $I_k$ 与共享的可学习视觉寄存器 $R = \{r_1, \ldots, r_M\}$ 拼接后输入 ViT
3. ViT 每层先执行标准自注意力（ReCompact），再执行跨子图的寄存器交互（ReAtten）
4. 输出仅保留每个子图的 $M$ 个寄存器特征，经 MLP 投影后送入 LLM

设 $M = 64$（而原始 image token $N = 576$），压缩率为 $576/64 = 9\times$。

### 关键设计

1. **ReCompact：基于寄存器的表示压缩**

    功能：引入 $M \ll N$ 个可学习视觉寄存器，与 image token 一起输入 ViT，通过自注意力让寄存器自适应聚合 image token 中的关键信息，编码完成后仅保留寄存器输出。

    核心思路：ViT 自注意力公式 $\hat{X}_{k,l} = \text{Softmax}(\frac{X_{k,l} X_{k,l}^T}{\sqrt{D_{key}}}) X_{k,l}$，不添加任何注意力掩码，利用预训练 ViT 天然将全局信息汇聚到特定 token 的能力（已有研究观察到这一现象）。

    设计动机：与 QFormer/Abstractor 等查询-交叉注意力方案相比，ReCompact 直接复用预训练 ViT 参数，所需适配数据极少（<3M 样本 vs QFormer 的 129M、Abstractor 的 400M）。

2. **ReAtten：寄存器交互注意力**

    功能：在 ViT 每层的自注意力之后，收集所有子图的寄存器隐状态 $\hat{X}_l^R \in \mathbb{R}^{M \cdot N_c \times D}$，通过 Cross-ViT-Attention 进行交互：$\bar{X}_l^R = \hat{X}_l^R + \text{Cross-ViT-Atten}(\hat{X}_l^R)$，然后与各自的 image token 重新拼接送入 FFN。

    核心思路：利用寄存器的紧凑性（$M \cdot N_c \ll N \cdot N_c$）使全图信息交换在计算上可行。Cross-ViT-Atten 参数用同层 ViT 自注意力参数初始化，实现平滑启动。

    设计动机：直接拼接所有子图的 image token 做全局注意力的二次复杂度不可承受；Shifted Window Attention 仅限局部交互信息不充分；互补图像金字塔 (CIP) 会引入额外冗余。

### 损失函数 / 训练策略

**四阶段渐进训练**：

- **Stage 0**（静态对齐，低分辨率）：冻结 ViT 和 LLM，仅训练寄存器和 MLP 投影，使用 image caption 数据
- **Stage 1**（粗对齐，低分辨率）：解冻所有参数，使用高质量长描述 + 全文 OCR 数据，让寄存器学习从粗到细的信息捕获
- **Stage 2**（精对齐，高分辨率）：引入高分辨率输入，使用细粒度描述和 OCR 任务（区域描述、文本定位）
- **Stage 3**（指令微调，高分辨率）：冻结 ViT，引入 ReAtten 模块，使用多任务指令数据微调

## 实验关键数据

### 主实验（表格）

**MME-RealWorld Benchmark（高分辨率理解）**：

| 模型 | Token 数/子图 | 感知 Avg-C | 推理 Avg-C |
|------|-------------|-----------|-----------|
| MiniCPM-V 2.5 | 96×9 | 44.0 | 36.0 |
| Monkey | 256×4 | 36.3 | 28.8 |
| GPT-4o | - | 41.9 | 42.3 |
| Claude 3.5 Sonnet | - | 47.7 | 49.2 |
| LLaVA-OneVision | 729×9 | 55.8 | 44.2 |
| **FALCON** | **64×16** | **50.3** | **43.4** |

FALCON 仅用 64 token/子图（9× 压缩），在感知和推理上超越 GPT-4o、GPT-4o-mini 和 Gemini-1.5-pro，在推理上与 LLaVA-OneVision（用 729×9 token）相当。

### 消融实验（表格）

**压缩方法对比（MME-RealWorld Avg-C 总分）**：

| 压缩方法 | Avg-C（感知+推理） |
|----------|------------------|
| Pooling | 较低 |
| Pixel Shuffle | 中等 |
| Abstractor | 低于池化 |
| **ReCompact** | **最高** |

**视觉连续性方法对比**：

| 方法 | V*_Avg | MME-RW 感知 | MME-RW 推理 | POPE |
|------|--------|------------|------------|------|
| Baseline（无连续性保持） | 51.3 | 38.2 | 35.6 | 85.7 |
| CIP | 50.3 | 38.4 | 37.2 | 86.3 |
| W-Atten | 60.2 | 41.0 | 38.1 | 86.4 |
| **ReAtten** | **61.3** | **42.1** | **39.0** | **87.3** |

ReAtten 在所有指标上最优，且有效减少幻觉（POPE 87.3）。

### 关键发现

- **寄存器数量消融**：从 36→64→144，性能持续提升但边际收益递减；64 寄存器是效率-性能最优平衡点（144 寄存器训练时间 2.41× 但提升有限）
- **注意力可视化**：每个寄存器聚焦图像的特定区域（人脸/文字等），几乎不关注背景，证实冗余消除的有效性
- **碎片化可视化**：无 ReAtten 时子图间注意力模式高度碎片化；加入 ReAtten 后注意力模式连续，表明有效实现了全图信息交换
- **同数据下对比**：使用与 LLaVA-v1.5 完全相同的数据训练时，FALCON 在 SQA (68.9 vs 66.8)、POPE (87.5 vs 85.9)、MMB (66.0 vs 64.3) 等多个基准上均优

## 亮点与洞察

- 核心创新优雅简洁：不需要额外压缩模块，直接在 ViT 内部通过寄存器机制同时解决冗余和碎片化两个问题
- 数据效率优势显著：与需要上亿样本预训练的 QFormer/Abstractor 相比，仅需 <3M 样本适配
- 四阶段渐进训练设计合理，确保寄存器从低分辨率到高分辨率平滑过渡

## 局限与展望

- 64 寄存器的固定数量可能不适应所有场景——简单图像分配过多、复杂图像分配不足
- 未探索动态寄存器数量分配（如根据图像复杂度调整）
- ReAtten 在每层 ViT 都执行，极端子图数量下仍有计算开销
- 未与最新的动态分辨率方法（如 NaViT）进行对比

## 相关工作与启发

- 利用了 ViT 中 register token 汇聚全局信息的已有发现 (Darcet et al., 2024)，将其从单纯的"注意力图伪影修复"扩展为功能性的表示压缩工具
- 与 TextMonkey 的 Shifted Window Attention 和 MiniMonkey 的 CIP 形成对比，ReAtten 实现了真正的全局交互
- 启发方向：寄存器技术可推广到视频 MLLM 中的时间冗余消除

## 评分

⭐⭐⭐⭐ 方法简洁有效，9× 压缩率下超越 GPT-4o 级商用模型令人印象深刻，设计动机清晰且有充分消融支撑，是高分辨率 MLLM 的优秀工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](../../ACL2025/multimodal_vlm/teaching_vlm_ask_ambiguity.md)
- [\[ICCV 2025\] HRScene: How Far Are VLMs from Effective High-Resolution Image Understanding?](hrscene_how_far_are_vlms_from_effective_high-resolution_image_understanding.md)
- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[ICML 2026\] CVSearch: Empowering Multimodal LLMs with Cognitive Visual Search for High-Resolution Image Perception](../../ICML2026/multimodal_vlm/cvsearch_empowering_multimodal_llms_with_cognitive_visual_search_for_high-resolu.md)

</div>

<!-- RELATED:END -->
