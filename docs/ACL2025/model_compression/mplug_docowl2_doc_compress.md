---
title: >-
  [论文解读] mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding
description: >-
  [ACL 2025][模型压缩][文档理解] 提出布局感知的High-resolution DocCompressor模块，用全局低分辨率视觉特征作为query、子图特征作为key/value进行分组交叉注意力，将每张高分辨率文档图片从数千tokens压缩至324 tokens，配合三阶段训练框架在多页文档理解上达到SOTA且First Token Latency降低50%以上。
tags:
  - "ACL 2025"
  - "模型压缩"
  - "文档理解"
  - "视觉token压缩"
  - "布局感知"
  - "多页文档"
  - "OCR"
---

# mPLUG-DocOwl2: High-resolution Compressing for OCR-free Multi-page Document Understanding

**会议**: ACL 2025  
**arXiv**: [2409.03420](https://arxiv.org/abs/2409.03420)  
**代码**: [GitHub](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl2)  
**领域**: 模型压缩 / 多模态VLM  
**关键词**: 文档理解, 视觉token压缩, 布局感知, 多页文档, OCR-free

## 一句话总结

提出布局感知的High-resolution DocCompressor模块，用全局低分辨率视觉特征作为query、子图特征作为key/value进行分组交叉注意力，将每张高分辨率文档图片从数千tokens压缩至324 tokens，配合三阶段训练框架在多页文档理解上达到SOTA且First Token Latency降低50%以上。

## 研究背景与动机

**领域现状**：OCR-free文档理解通过将高分辨率文档图片裁剪为多个低分辨率子图来捕获细粒度文字信息，已取得显著进展（如InternVL 2在DocVQA上达91.6%）。然而这种策略的代价是每张文档图片需要数千视觉tokens（InternVL 2平均3k+ tokens），导致GPU内存消耗大、推理速度慢。

**现有痛点**：大量视觉tokens使多页文档理解几乎不可行——10页文档需要3万+ tokens，远超大多数LLM的上下文窗口。现有压缩方法存在明显缺陷：(1) 独立压缩每个子图仍需大量tokens（如TokenPacker仍1.8k+ tokens）；(2) 可学习query（如Resampler/Q-former）缺乏布局先验，难以有效压缩文档中密集的文字信息；(3) 基于token相似度选择的方法（如TextMonkey）可能遗漏部分区域。

**核心矛盾**：文档图片中的文字信息密度远高于自然图片，简单的视觉token压缩方法会严重丢失文字信息；但不压缩又无法实现多页/多图联合理解。

**本文目标** 如何在将高分辨率文档图片大幅压缩的同时保留完整的布局和文字信息？

**切入角度**：两个关键观察——(1) NLP领域已证明文本段落可以被压缩为少量向量而保留大部分语义；(2) 经过vision-to-text模块对齐后的视觉tokens本质上已是"文本token"，可以视为编码了图像不同区域文字信息的文本token，因此可以用类似文本压缩的方式来处理。全局低分辨率图片天然编码了整体布局信息，可以作为压缩的语义引导。

**核心 idea**：用全局低分辨率特征作为query引导跨注意力压缩高分辨率子图特征，利用位置对应关系进行分组注意力，并在vision-to-text对齐后执行以更好保留文字语义。

## 方法详解

### 整体框架

DocOwl2的编码流程：高分辨率图片 → Shape-adaptive Cropping切为$R \times C$个子图+全局图 → ViT独立编码每个子图和全局图 → H-Reducer（卷积+FC）做vision-to-text对齐并将每个子图的token数减到1/4 → **High-resolution DocCompressor**进一步压缩到与全局特征图同样大小（324 tokens） → 多张图的压缩tokens拼接后送入LLM。三阶段训练：Single-image Pretraining → Multi-image Continue-Pretraining → Multi-task Finetuning。

### 关键设计

1. **布局感知的分组交叉注意力压缩**:

    - 功能：将每张高分辨率文档图片从 $(R \times C + 1) \times h \times w/4$ tokens压缩到 $h \times w/4$ tokens（如从2560压到324）
    - 核心思路：全局特征图 $\hat{V}^g$ 中的每个token $\hat{v}_{ij}^g$ 作为query，对应的 $R \times C$ 个高分辨率子图tokens $\hat{v}_{ij}^s$ 作为key/value进行分组交叉注意力：$\bar{v}_{ij} = \text{softmax}(\frac{W^q \hat{v}_{ij}^g \cdot W^k \hat{v}_{ij}^s}{\sqrt{d_k}}) W^v \hat{v}_{ij}^s + \hat{v}_{ij}^g$（含residual connection）。位置对应关系由图片裁剪的空间映射自然确定
    - 设计动机：区别于让每个query attend所有高分辨率tokens（计算量大且信息压缩更困难），分组注意力利用了全局图与子图之间天然的空间对应关系，每个query只需关注同一物理区域的 $R \times C$ 个对应位置的tokens，更容易按布局区域聚合语义信息

2. **压缩位置：vision-to-text对齐之后**:

    - 功能：将DocCompressor放置在H-Reducer（V2T模块）之后而非ViT和H-Reducer之间
    - 核心思路：先通过H-Reducer的卷积层聚合水平方向4个特征并通过FC层与LLM特征空间对齐，使视觉特征已编码为"类文本token"，再进行压缩
    - 设计动机：消融实验（r4 vs r3）证实在V2T对齐后压缩优于直接压缩ViT输出。直觉是：压缩已对齐的特征类似于NLP中的文本摘要（在语义空间中操作），而直接压缩视觉特征会丢失更多文字信息

3. **三阶段训练框架**:

    - 功能：分阶段逐步赋予模型单图理解→多图关联→多任务泛化能力
    - 核心思路：**Stage 1**（Single-image Pretraining）：在DocStruct4M上学习文档/表格/图表的结构解析，确保压缩tokens编码足够信息。**Stage 2**（Multi-image Continue-Pretraining）：在MP-DocStruct1M上学习多页文本解析和文本查找两个对称任务。**Stage 3**（Multi-task Finetuning）：混合单图（DocDownstream-1.0, DocReason25K）和多图（MP-DocVQA, DUDE, NewsVideoQA, MP-DocReason51K）指令微调数据集
    - 设计动机：Stage 2的两个对称任务（给页码解析文字 + 给文字找页码）是多页理解的基础能力——模型需要在多张图片间建立页码-内容的双向映射。消融实验（r3 vs r2）证实这一阶段对10+页文档的理解至关重要

### 损失函数 / 训练策略

基于mPLUG-Owl2初始化。Stage 1训练12k步（batch 1024, lr 1e-4），冻结LLM主参数只调MAM；Stage 2训练2.4k步（batch 1024, lr 2e-5），冻结ViT；Stage 3训练9k步（batch 256, lr 2e-5），除ViT外全部参数可训练。DocCompressor仅含2层交叉注意力。

## 实验关键数据

### 主实验

单页文档理解（visual tokens < 1k组）：

| 模型 | TokenV | DocVQA | ChartQA | TextVQA | InfoVQA |
|------|--------|--------|---------|---------|---------|
| TextMonkey | 768 | 73.0 | 66.9 | 65.9 | 28.6 |
| UReader | ~841 | 65.4 | 59.3 | 57.6 | 42.2 |
| **DocOwl2** | **324** | **80.7** | **70.0** | **66.7** | **46.4** |

多页文档理解：

| 模型 | TokenV | MP-DocVQA (ANLS) | DUDE (ANLS) | FTL(s)↓ |
|------|--------|------------------|-------------|---------|
| LongVA-7B | ~2029 | 60.80 | 38.37 | 2.13 |
| Idefics3-8B | ~838 | 67.15 | 38.65 | 2.26 |
| **DocOwl2-8B** | **324** | **69.42** | **46.77** | **0.95** |

### 消融实验

压缩架构对比（相同训练流程，单图评估）：

| 方法 | 压缩方式 | TokenV | DocVQA | WTQ | ChartQA |
|------|---------|--------|--------|-----|---------|
| Resampler | 可学习query | 256 | 69.0 | 29.4 | 66.6 |
| CAbstractor | 自适应均值池化 | 256 | 73.0 | 32.6 | 67.6 |
| **DocCompressor** | 分组交叉注意力 | 256 | **76.1** | **35.1** | **69.2** |
| DocCompressor (after ViT) | 分组交叉注意力 | 256 | 75.7 | 33.3 | 68.7 |
| DocCompressor (完整注意力) | 全局交叉注意力 | 256 | 74.4 | 33.7 | 68.2 |
| DocCompressor (均值) | 分组均值池化 | 256 | 74.6 | 31.9 | 68.2 |

### 关键发现

- **压缩10倍性能仅降2%**：与DocOwl 1.5（~1698 tokens）相比，DocOwl2（324 tokens）在DocVQA上达到98%性能，但First Token Latency降低55%（0.26s vs 0.58s）
- **分组注意力优于全局注意力**：利用空间对应的分组注意力（r3）比全局注意力（r5）+2个百分点，且计算更低
- **V2T对齐后压缩更好**：放在H-Reducer之后（r3）比之前（r4）好0.4-1.8个百分点
- **多图训练阶段贡献显著**：加入Stage 2后10+页文档的准确率从5.8%跳到37.9%（r4 vs r1）
- **与>1k tokens的SOTA比**：用<20%的tokens达到了InternVL 2 / IXC 2.5在7/10基准上>80%的性能

## 亮点与洞察

- **布局感知压缩的直觉精准**：文档中同一布局区域的文字语义连贯，用全局特征图的空间结构作为压缩引导非常自然。这个设计比可学习query、token选择等方法都更贴合文档的物理结构
- **"视觉token = 文本token"的洞察**：经V2T对齐后的视觉tokens已进入文本语义空间，因此压缩它们等价于文本摘要，而非视觉信息的有损编码。这个insight解释了为什么在V2T后压缩效果更好
- **三阶段训练中的对称任务设计**：多页文本解析（页码→文本）和文本查找（文本→页码）互为逆向，同时训练能建立更强的页码-内容双向关联

## 局限与展望

- **基座模型较老**：基于LLaMA-1和mPLUG-Owl2，性能天花板受限。用更强的基座（如LLaMA-3/Qwen2）可能显著提升
- **固定压缩比**：所有图片统一压缩到324 tokens，但不同文档的信息密度差异大。自适应压缩比是值得探索的方向
- **与SOTA差距仍存**：在需要精细OCR的任务上（如DocVQA），与InternVL 2（91.6%）差距仍有10+个百分点
- **仅验证文档类图片**：对自然图片的压缩效果未充分验证

## 相关工作与启发

- **vs TokenPacker**: TokenPacker用下采样特征作为query压缩每个子图，但仍需拼接所有子图结果（1.8k tokens）；DocOwl2用全局特征引导压缩全局性做得更彻底（324 tokens）
- **vs TextMonkey**: TextMonkey基于token相似度选择有价值的tokens作为压缩引导，可能遗漏某些区域；DocOwl2的全局特征图覆盖所有区域
- **vs Mini-Gemini**: Mini-Gemini需要额外的高分辨率编码器；DocOwl2复用同一ViT编码全局图和子图，更简洁
- **启发**：这种"用低分辨率全局特征引导高分辨率细节压缩"的范式可以迁移到视频理解（关键帧引导压缩相邻帧）和超长文档处理

## 评分

- 新颖性: ⭐⭐⭐⭐ 布局感知压缩的设计自然且有效，V2T后压缩的insight有深度
- 实验充分度: ⭐⭐⭐⭐⭐ 10个单页+3个多页基准，充分消融，延迟分析，定性示例
- 写作质量: ⭐⭐⭐⭐ 结构清晰，消融设计系统，动机阐述到位
- 价值: ⭐⭐⭐⭐ 为多页文档理解的token效率问题提供了实用且可复现的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Understanding Multi-layered Transmission Matrices](../../CVPR2025/model_compression/understanding_multi-layered_transmission_matrices.md)
- [\[ICCV 2025\] Variance-Based Pruning for Accelerating and Compressing Trained Networks](../../ICCV2025/model_compression/variance-based_pruning_for_accelerating_and_compressing_trained_networks.md)
- [\[ICML 2026\] Hierarchical Image Tokenization for Multi-Scale Image Super Resolution](../../ICML2026/model_compression/hierarchical_image_tokenization_for_multi-scale_image_super_resolution.md)
- [\[ACL 2025\] TaDA: Training-free recipe for Decoding with Adaptive KV Cache Compression and Mean-centering](tada_training-free_recipe_for_decoding_with_adaptive_kv_cache_compression_and_me.md)
- [\[CVPR 2026\] High Resolution Neural Video Coding with Bi-directional Confidence-Guided Reference Information Modeling](../../CVPR2026/model_compression/high_resolution_neural_video_coding_with_bi-directional_confidence-guided_refere.md)

</div>

<!-- RELATED:END -->
