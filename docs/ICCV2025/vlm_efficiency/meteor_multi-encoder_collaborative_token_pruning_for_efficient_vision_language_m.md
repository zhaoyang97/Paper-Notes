---
title: >-
  [论文解读] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models
description: >-
  [ICCV 2025][多模态VLM][多编码器VLM] METEOR 提出首个面向多编码器 MLLM 的三阶段渐进式 token 剪枝框架：在编码阶段用特征秩分配各编码器的稀疏比例，在融合阶段通过协同剪枝消除跨编码器冗余，在解码阶段根据文本提示自适应调整剪枝比例，将视觉 token 减少 76% 而性能仅降 0.3%。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "多编码器VLM"
  - "视觉token剪枝"
  - "协同压缩"
  - "自适应剪枝"
  - "特征秩分配"
---

# METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models

**会议**: ICCV 2025  
**arXiv**: [2507.20842](https://arxiv.org/abs/2507.20842)  
**代码**: [https://github.com/YuchenLiu98/METEOR](https://github.com/YuchenLiu98/METEOR)  
**领域**: 多模态VLM  
**关键词**: 多编码器VLM、视觉token剪枝、协同压缩、自适应剪枝、特征秩分配

## 一句话总结

METEOR 提出首个面向多编码器 MLLM 的三阶段渐进式 token 剪枝框架：在编码阶段用特征秩分配各编码器的稀疏比例，在融合阶段通过协同剪枝消除跨编码器冗余，在解码阶段根据文本提示自适应调整剪枝比例，将视觉 token 减少 76% 而性能仅降 0.3%。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）通过将图像编码为视觉 token 并与文本 token 拼接来实现多模态理解。单编码器（如 CLIP）存在泛化局限，近期 EAGLE、Cambrian-1 等工作通过融合多个视觉编码器（CLIP、ConvNeXt、Pix2Struct、EVA-02 等）来增强视觉感知能力。

**现有痛点**：多编码器方法带来了严重的计算开销。例如 Mini-Gemini 中双编码器处理 672×672 图像产生 2880 个视觉 token，self-attention 的计算复杂度随 token 数量二次增长。现有的 token 压缩方法（FastV、Pdrop、SparseVLM）都针对单编码器设计，无法解决多编码器特有的问题：如何为不同信息丰度的编码器分配合理的 token 预算？如何消除跨编码器间重叠的冗余信息？此外，这些方法使用固定剪枝比例，无法适应不同任务的需求，导致在 OCR 等细粒度任务上性能严重下降。

**核心矛盾**：多编码器带来性能增益但计算成本剧增，且各编码器产生的视觉 token 存在大量冗余——既有编码器内部的冗余，也有跨编码器的信息重叠。现有剪枝方法无法在多编码器场景下进行协同优化。

**本文目标**：设计一个全流程的多编码器 token 剪枝框架，在编码、融合、解码三个阶段渐进式消除冗余。

**切入角度**：深入分析了多编码器 MLLM 的 token 冗余模式，发现三个关键 insight——浅层用注意力识别冗余不可靠但平均 token 相似度可用、特征图的秩可稳定衡量信息丰度、不同任务需要不同数量的视觉 token。

**核心 idea**：基于五个关键发现，从编码→融合→解码三个阶段逐步精简视觉 token，其中编码阶段用秩引导分配、融合阶段用互冗余消除、解码阶段用文本引导的自适应剪枝。

## 方法详解

### 整体框架

METEOR 在多编码器 MLLM（基于 EAGLE 架构）的三个处理阶段中分别引入 token 剪枝。Stage 1（编码阶段）：在各编码器内部独立剪枝冗余 token，使用特征秩分配各编码器的稀疏比例。Stage 2（融合阶段）：通过独立投影器适配各编码器 token 后，用协同剪枝消除跨编码器冗余。Stage 3（解码阶段）：在 LLM 层中根据文本提示自适应确定保留 token 数量。

### 关键设计

1. **编码阶段的秩引导协同 token 分配（Stage 1）**:

    - 功能：在各编码器内部渐进式剪枝冗余 token，并为不同编码器合理分配 token 预算
    - 核心思路：将每个编码器等分为三个阶段。**浅层阶段**用与平均 token 的余弦相似度衡量冗余（Finding 1：浅层注意力值不稀疏且不稳定，但平均 token 对应低频背景信息，与之相似的 token 冗余度高）。**深层阶段**用 cls token 与视觉 token 间的注意力值衡量冗余（Finding 2：深层注意力稀疏且可靠）。为分配各编码器的 token 预算，对特征图做 SVD 分解计算秩 $r_b^l$，秩越高表示信息越丰富，按秩比例分配保留数量：$k_b^l = k_b \cdot r_b^l / \sum_{l=1}^{C} r_b^l$。秩的期望对输入图像鲁棒（方差极小），可离线在小批量数据上计算
    - 设计动机：不同编码器（如 CLIP 偏语义、Pix2Struct 偏 OCR）产生的特征信息丰度不同，均匀分配 token 预算是次优的。秩提供了一个数学上有根据的信息丰度度量

2. **融合阶段的跨编码器协同剪枝（Stage 2）**:

    - 功能：消除不同编码器 token 间的信息重叠冗余
    - 核心思路：采用 post-projection 融合策略：每个编码器配备独立的 MLP 投影器，先将各编码器的 token 独立映射到共享语义空间，再拼接融合。在共享空间中计算跨编码器的互冗余度：$\mathcal{R}_i^j = \sum_{l=1, l \neq j}^{L} \sum_{m=1}^{n_l} \mathcal{S}(z_i^j, z_m^l)$，保留互冗余最低的 top-k token。Finding 3 表明：消除跨编码器冗余比编码器内部冗余更有效
    - 设计动机：现有方法用共享投影器（pre-projection fusion），这样不够灵活且忽略了跨编码器冗余。独立投影器 + 协同剪枝在共享语义空间中精准识别并消除信息重叠

3. **解码阶段的文本感知自适应剪枝（Stage 3）**:

    - 功能：根据文本提示内容动态调整保留 token 数量，适应不同任务需求
    - 核心思路：在 LLM 的特定层，使用文本 token 到视觉 token 的注意力值来识别冗余 token。关键改进是 **注意力头筛选**（Finding 4）：不是平均所有注意力头，而是选择 Visual Attention Value（VAV）最大的 top-k 注意力头，因为大多数头与视觉定位无关甚至产生幻觉。**自适应 token 保留**（Finding 5）：top-k 头的 VAV 总和与任务复杂度高度相关（AI2D 等粗粒度任务 VAV 低，DocVQA 等 OCR 任务 VAV 高），据此动态计算保留 token 数：$K = \lambda \cdot \sum_{h=1}^{k} \sum_{i=1}^{N} a_{i, \mathbf{I}(h)}$
    - 设计动机：固定剪枝比例在 OCR 任务上严重掉点（OCR 需要保留更多细粒度 token）。VAV 天然反映了视觉信息对当前任务的贡献程度，用它来自适应调整 token 预算是合理且高效的

### 损失函数 / 训练策略

Stage 1 和 Stage 2 的剪枝策略融入到预训练和 SFT 过程中联合训练（预训练 558K 数据冻结除投影器外的参数，SFT 1M 或 1.8M 数据全模型微调）。Stage 3 是 training-free 的，直接在推理时应用。

## 实验关键数据

### 主实验

在 11 个 benchmark 上与 EAGLE 和其他高效方法对比（基于 Vicuna-7B）：

| 方法 | 视觉Token | TFLOPS↓ | SQA | AI2D | GQA | POPE | TextVQA | DocVQA | OCRBench | 均值 |
|------|----------|---------|-----|------|-----|------|---------|--------|----------|------|
| EAGLE | 1024 | 26.21 | 71.0 | 72.2 | 64.8 | 88.4 | 71.7 | 73.2 | 538 | 69.3 |
| FastV | 256 | 16.83 | 70.5 | 72.5 | 61.8 | 86.4 | 70.5 | 54.2 | 431 | 64.9 |
| SparseVLM | 256 | 17.89 | 70.7 | 72.2 | 59.8 | 86.5 | 70.4 | 51.9 | 383 | 63.9 |
| **METEOR** | **242*** | **13.42** | **71.4** | **73.4** | 63.5 | 87.9 | **71.1** | **71.4** | **533** | **69.0** |

### 消融实验

融合策略和剪枝方法消融（SFT 1M 数据）：

| 融合方式 | 剪枝方法 | Token数 | 知识 | 通用 | OCR | 均值 |
|---------|---------|---------|------|------|-----|------|
| Pre-proj | - | 1193 | 64.4 | 73.7 | 64.5 | 67.5 |
| Post-proj | - | 1193 | 65.5 | 74.0 | 65.4 | 68.3 |
| Post-proj | Random | 576 | 63.4 | 72.2 | 48.5 | 61.4 |
| Post-proj | Resampler | 576 | 61.1 | 70.6 | 49.2 | 60.3 |
| Post-proj | **Ours** | 576 | **65.4** | **74.0** | **65.5** | **68.3** |

### 关键发现

- **76% token 压缩仅损失 0.3%**：METEOR-242 相比 EAGLE-1024 减少了 76% 的视觉 token，TFLOPS 降低 49%，吞吐量提升 46%，平均性能仅下降 0.3%
- **OCR 任务优势巨大**：相比 FastV/Pdrop/SparseVLM 等固定比例方法，METEOR 在 OCR 任务上领先 8.8-12.3%，因为自适应剪枝为 OCR 任务保留了更多 token
- **跨编码器协同剪枝有效**：协同剪枝比编码器内部独立剪枝和随机剪枝分别高出 1.4% 和 6.9%（均值），且核范数分析证实了特征多样性的提升
- **秩引导分配优于均匀分配**：按秩分配比均匀分配在 OCR 任务上高出约 2%，因为低秩编码器产生的 token 信息冗余更高
- **自适应比例 vs 固定比例**：在相同平均 token 数（242）下，自适应比例比固定比例在 OCR 上高 2.6%

## 亮点与洞察

- **特征秩作为信息度量的妙用**：利用 SVD 的秩来量化各编码器特征图的信息丰度，发现秩对输入图像的方差极小，可以离线计算一次复用。这个发现从 CNN 扩展到了 Vision Transformer，是一个有理论基础且实用的贡献
- **五个 Finding 驱动的设计范式**：每个设计选择都有对应的实证发现支撑（浅深层用不同准则、秩的稳定性、跨编码器冗余、注意力头非均匀、VAV 与任务复杂度相关），methodical 且有说服力
- **Instance-adaptive 思想**：根据每个输入实例的 VAV 动态调整剪枝比例，这种"因题施策"的思路可以迁移到其他需要效率-精度权衡的场景中

## 局限与展望

- 目前仅在 EAGLE 和 Cambrian-1 两种多编码器架构上验证，对更多样化的架构适配性有待检验
- Stage 3 的自适应剪枝是 training-free 的，如果能引入少量训练可能进一步提升效果
- 秩的离线计算假设秩对输入鲁棒，在分布偏移严重的场景下是否仍成立需要验证
- 未考虑视频输入场景下多帧视觉 token 的时序冗余消除

## 相关工作与启发

- **vs FastV / Pdrop**: 这些方法仅在 LLM 阶段做固定比例剪枝，忽略了编码和融合阶段的加速机会以及 OCR 等任务的特殊需求。METEOR 全流程+自适应的设计全面领先
- **vs EAGLE**: EAGLE 是 METEOR 的基座模型，使用全量 1024 token。METEOR 以仅 24% 的 token 数达到 99.7% 的性能，证明了多编码器 MLLM 中视觉 token 冗余极为严重
- **vs Cambrian-1 (SVA)**: 使用相同编码器组合时，METEOR 在 OCR 任务上比 Cambrian-1 的 SVA 聚合方式高 3.3%，且 token 数少 44%

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个多编码器 MLLM 的全流程 token 剪枝框架，秩引导分配和自适应剪枝设计新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 11 个 benchmark、多种消融、两种编码器组合、效率分析，极为全面
- 写作质量: ⭐⭐⭐⭐ 五个 Finding 的组织方式清晰有力，方法推导逻辑性强
- 价值: ⭐⭐⭐⭐⭐ 解决了多编码器 MLLM 的关键效率瓶颈，实用价值极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CoMP: Collaborative Multi-Mode Pruning for Vision-Language Models](../../CVPR2026/multimodal_vlm/comp_collaborative_multi-mode_pruning_for_vision-language_models.md)
- [\[ICCV 2025\] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration](feather_the_throttle_revisiting_visual_token_pruning_for_vision-language_model_a.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[NeurIPS 2025\] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization](../../NeurIPS2025/multimodal_vlm/balanced_token_pruning_accelerating_vision_language_models_b.md)
- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](../../NeurIPS2025/multimodal_vlm/scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)

</div>

<!-- RELATED:END -->
