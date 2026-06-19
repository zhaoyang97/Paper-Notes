---
title: >-
  [论文解读] M3-JEPA: Multimodal Alignment via Multi-gate MoE based on JEPA
description: >-
  [ICML 2025][多模态VLM][JEPA] 将 JEPA（联合嵌入预测架构）推广到任意模态组合的多模态对齐中，用 Multi-gate MoE 作为跨模态预测器在潜在空间对齐（而非 token 空间），门控函数解耦模态特定和共享信息，通过交替梯度下降避免多方向任务间的梯度冲突，仅 140M 可训练参数在多个检索和分类任务上超越 BLIP-2（1.2B）等 SOTA。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "JEPA"
  - "Mixture-of-Experts"
  - "多模态"
  - "energy-based model"
  - "alternating gradient descent"
---

# M3-JEPA: Multimodal Alignment via Multi-gate MoE based on JEPA

**会议**: ICML 2025  
**arXiv**: [2409.05929](https://arxiv.org/abs/2409.05929)  
**代码**: [GitHub - M3-JEPA](https://github.com/HongyangLL/M3-JEPA)  
**领域**: 多模态VLM  
**关键词**: JEPA, Mixture-of-Experts, multimodal alignment, energy-based model, alternating gradient descent

## 一句话总结

将 JEPA（联合嵌入预测架构）推广到任意模态组合的多模态对齐中，用 Multi-gate MoE 作为跨模态预测器在潜在空间对齐（而非 token 空间），门控函数解耦模态特定和共享信息，通过交替梯度下降避免多方向任务间的梯度冲突，仅 140M 可训练参数在多个检索和分类任务上超越 BLIP-2（1.2B）等 SOTA。

## 研究背景与动机

**领域现状**：现代多模态学习的主流采用生成式架构，分两类：一类从头训练（如 OFA、BEiT-3），需要大量数据和计算；另一类使用预训练 LLM 作为骨干微调轻量连接器（如 BLIP-2、LLaVA），计算效率更高。两类方法都在原始 token 空间进行跨模态对齐优化。

**现有痛点**：在 token 空间对齐容易出现模态崩溃（modality collapse）—— 一种模态的信号主导另一种。根源包括：(1) 多方向任务的梯度冲突；(2) 连续域（图像/视频）与离散域（文本）的分布不匹配；(3) 信息不确定性和冗余（同一张图可有多种语义等价但文本不同的描述）。这些因素使跨模态对齐难以收敛，且可能遗漏关键信息。

**核心矛盾**：token 空间的对齐过于"表面"——它要求预测出精确的 token sequence，但跨模态信息天然存在不确定性和一对多映射。需要一种在更抽象的潜在空间中对齐的方案，只保留跨模态共享的核心语义信息。

**本文目标** (1) 如何避免 token 空间对齐导致的模态崩溃？(2) 如何设计一个通用的 any-to-any 多模态对齐框架？(3) 如何在多方向任务（如 image→text 和 text→image）之间避免梯度冲突？

**切入角度**：JEPA（Joint-Embedding Predictive Architecture）从能量基模型的角度出发，不在 token 空间做生成式预测，而是用一个预测器将输入嵌入投影到输出嵌入空间，在潜在空间进行对齐。I-JEPA 和 V-JEPA 已在视觉自监督学习中验证了该范式的有效性，但尚无通用多模态版本。

**核心 idea**：用 Multi-gate MoE 实现 JEPA 的跨模态预测器，门控函数自动解耦模态特定和共享信息，配合交替梯度下降避免多任务冲突，实现首个 any-to-any 的多模态 JEPA 对齐框架。

## 方法详解

### 整体框架

给定 $M$ 种模态和 $T$ 个任务，每种模态由冻结的预训练单模态编码器（LLama3-8B 编码文本、DINOv2-Large 编码图像、LanguageBind 编码音频）产生嵌入。对于任务 $t$，输入嵌入 $e_x^t$ 和输出嵌入 $e_y^t$ 分别由对应模态编码器生成。一个 Multi-gate MoE 预测器 $\mathcal{P}$ 将 $e_x^t$ 投影到 $e_y^t$ 的潜在空间，形成 $e_{x \to y}^t = \mathcal{P}(e_x^t)$。然后在潜在空间中最小化 $e_{x \to y}^t$ 和 $e_y^t$ 之间的能量函数 $\mathcal{F}^t(x,y)$。编码器仅 3 层 LoRA 微调（rank=64），其余参数冻结。MoE 预测器随机初始化，全参数训练。

### 关键设计

1. **Multi-gate MoE 跨模态预测器**:

    - 功能：作为轻量级跨模态连接器，将输入模态的嵌入投影到输出模态的潜在空间
    - 核心思路：为每种模态实现 $N=12$ 个专家网络，总计 $M \times N$ 个专家，采用 Top-$K$（$K=4$）稀疏激活。门控函数接受输入嵌入 $e_x$ 和可学习的模态嵌入 $e_m$ 的拼接作为输入：$\mathbb{G} = \text{softmax}(g \cdot [e_x \oplus e_m])$，其中 $g$ 是共享的投影矩阵。通过 $L=2$ 个并行门控分别服务对比损失和正则化损失。总可训练参数仅 140M
    - 设计动机：(1) 模态特定路径（模态专家 + 模态嵌入 $e_m$）捕获各模态独有信息；(2) 共享投影矩阵 $g$ 创建跨模态公共子空间；(3) 轻量 MoE 比全量微调编码器高效得多

2. **对比+正则化双损失的能量函数**:

    - 功能：从两个互补角度优化跨模态对齐
    - 核心思路：正则化损失 $\mathcal{L}_{\text{reg}} = |e_{x \to y} - e_y|_2^2$ 直接拉近正对的嵌入距离（最小化条件熵 $\mathcal{H}(y|x)$）。对比损失 $\mathcal{L}_{\text{cl}}$ 使用 batch 内负样本通过 InfoNCE 将正对拉近、负对推远（最大化互信息 $\mathcal{I}(x;y)$）。总损失 $\mathcal{L} = \alpha \mathcal{L}_{\text{reg}} + (1-\alpha)\mathcal{L}_{\text{cl}}$，理论分析证明最优 $\alpha = 0.5$（对应自由能最小化的临界温度），实验也验证了这一结论
    - 设计动机：单用对比损失可能导致表示坍塌（所有嵌入趋同），单用正则化损失无法区分负对。双损失构成完整的能量函数，从信息论角度同时最大化互信息和最小化条件熵

3. **交替梯度下降（AGD）**:

    - 功能：解决多方向多模态任务之间的梯度冲突
    - 核心思路：在不同训练步轮流切换 $T$ 个任务，每步仅对当前任务做前向传播和反向传播：$\theta(i+1) \leftarrow \theta(i) - \eta \nabla_\theta \mathcal{L}^t$，其中 $\text{mod}(i, T) = t$。与联合优化不同，AGD 解耦了各任务的参数更新，避免了如 image→text 和 text→image 争夺同一连接器权重导致的梯度冲突
    - 设计动机：传统联合优化在多方向任务中容易出现梯度冲突（seesaw effect），AGD 借鉴了多任务学习中交替训练的成功经验

### 损失函数

总损失 $\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{reg}} + 0.5 \cdot \mathcal{L}_{\text{cl}}$，其中正则化损失为 L2 距离，对比损失为 InfoNCE。训练使用 Adam 优化器，batch size 128，cosine 学习率调度，warmup 0.1，weight decay 0.005。

## 实验关键数据

### 主实验表格：视觉-语言检索（Flickr30K / COCO）

| 方法 | 可训练参数 | Flickr30K I→T R@1 | Flickr30K T→I R@1 | COCO I→T R@1 | COCO T→I R@1 |
|------|----------|------------------|------------------|-------------|-------------|
| CLIP | 428M | 88.0 | 68.7 | - | - |
| BLIP-2 (ViT-g) | 1.2B | 97.6 | 89.7 | 85.4 | 68.3 |
| BEiT-3 | 1.9B | 94.9 | 81.5 | 84.8 | 67.2 |
| **M3-JEPA** | **140M** | **97.8** | **97.8** | **87.7** | **89.7** |

M3-JEPA 仅用 140M 参数在 COCO T→I R@1 上达 89.7%，远超 BLIP-2 的 68.3%（+21.4 pt），在 Flickr30K 上两个方向均达 97.8%。

### 消融实验表格：方法组件对 COCO 检索的影响

| MoE | AGD | I→T R@1 | I→T R@5 | I→T R@10 | T→I R@1 | T→I R@5 | T→I R@10 |
|-----|-----|---------|---------|----------|---------|---------|----------|
| ✗ | ✓ | 74.4 | 86.0 | 92.2 | 82.3 | 89.5 | 92.6 |
| ✓ | ✗ | 68.2 | 68.7 | 81.1 | 74.2 | 88.7 | 92.4 |
| ✓ | ✓ | **87.7** | **99.6** | **99.9** | **89.7** | **99.7** | **99.9** |

MoE 和 AGD 缺一不可：去掉 MoE（换 MLP）I→T R@1 降至 74.4%，去掉 AGD 降至 68.2%，两者结合达 87.7%。

### 关键发现

- 在 ImageNet-1K 分类上 M3-JEPA 达 86.6% accuracy，超越 CLIP-ViT（82.1%）和 DINOv2（83.2%），证明 JEPA 范式可处理分类任务
- 在音频-文本检索（Clotho/Audiocaps）上 zero-shot 也超越 LanguageBind 等方法，证明框架的模态可扩展性
- VQA 任务上（VQAv2 test-dev 82.3%）接近 BLIP-2 水平，证明多模态输入场景的适应能力
- 理论预测的最优 $\alpha=0.5$ 与实验验证完全吻合

## 亮点与洞察

- 首个 any-to-any 多模态 JEPA 框架，将 I-JEPA/V-JEPA 从单模态自监督扩展到跨模态对齐
- 140M 参数超越 1.2B 的 BLIP-2：轻量 MoE 预测器 + 冻结编码器的效率优势明显
- MoE 门控的模态解耦有信息论支撑：共享矩阵 $g$ 对应互信息，模态嵌入 $e_m$ 对应条件熵
- 理论分析与实验的 $\alpha=0.5$ 完美对应，增强了框架的可信度

## 局限性

- 门控信息解耦的质量依赖于数据质量和模态编码器的表示空间
- 冻结编码器限制了细粒度适配能力（VQA 上表现略低于全量训练的 BEiT-3）
- AGD 的任务切换增加了训练复杂度和调参难度
- 3D/触觉等更多模态的验证缺失

## 相关工作与启发

- **vs I-JEPA/V-JEPA**：本文是 JEPA 在多模态上的首次推广，从单模态自监督到跨模态对齐
- **vs CLIP/ALIGN**：CLIP 在 token 空间做对比学习，M3-JEPA 在潜在空间做；后者通过 MoE 预测器过滤了无关信息
- **启发**：JEPA + MoE 的范式可能成为自监督多模态学习的新基础，特别适合信息不确定性高的场景

## 评分

⭐⭐⭐⭐⭐ JEPA 多模态推广的开创性工作，理论分析完整（信息论+最优超参+收敛保证），实验覆盖文本/图像/音频三种模态和检索/分类/VQA 多种任务，140M 参数效率优势突出。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] CHARM: 用 Multimodal JEPA + 通道描述做时间序列 foundation embedding](../../ICML2026/multimodal_vlm/giving_sensors_a_voice_multimodal_jepa_for_semantic_time-series_embeddings.md)
- [\[ICML 2026\] Text-Conditional JEPA for Learning Semantically Rich Visual Representations](../../ICML2026/multimodal_vlm/text-conditional_jepa_for_learning_semantically_rich_visual_representations.md)
- [\[ICML 2025\] Universal Retrieval for Multimodal Trajectory Modeling](universal_retrieval_for_multimodal_trajectory_modeling.md)
- [\[ICML 2025\] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](the_devil_is_in_the_details_tackling_unimodal_spurious_correlations_for_generali.md)
- [\[NeurIPS 2025\] The Narrow Gate: Localized Image-Text Communication in Native Multimodal Models](../../NeurIPS2025/multimodal_vlm/the_narrow_gate_localized_imagetext_communication_in_native.md)

</div>

<!-- RELATED:END -->
