---
title: >-
  [论文解读] Unifying Vision-Language Latents for Zero-Label Image Caption Enhancement
description: >-
  [NeurIPS 2025][多模态VLM][零标签学习] 本文提出ViZer框架，通过统一视觉-语言潜空间对齐的训练范式，在无任何文本标注的情况下提升VLM的图像描述能力——仅使用原始图像数据就能让模型生成更接地、更描述性的caption。 视觉语言模型（VLM）在大规模图像-文本预训练后取得了出色表现…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "零标签学习"
  - "图像描述"
  - "视觉-语言对齐"
  - "联合嵌入"
  - "自监督"
---

# Unifying Vision-Language Latents for Zero-Label Image Caption Enhancement

**会议**: NeurIPS 2025  
**arXiv**: [2510.12931](https://arxiv.org/abs/2510.12931)  
**代码**: 暂无  
**领域**: 多模态VLM  
**关键词**: 零标签学习, 图像描述, 视觉-语言对齐, 联合嵌入, 自监督

## 一句话总结
本文提出ViZer框架，通过统一视觉-语言潜空间对齐的训练范式，在无任何文本标注的情况下提升VLM的图像描述能力——仅使用原始图像数据就能让模型生成更接地、更描述性的caption。

## 研究背景与动机
视觉语言模型（VLM）在大规模图像-文本预训练后取得了出色表现，但它们对标注数据的依赖限制了可扩展性，同时留下了大量未标注图像数据被闲置。这种标注稀缺不仅约束了训练范围，还导致视觉编码器和语言模型之间的**持续性不匹配**——即使在SOTA系统中也会产生幻觉、事实错误的caption和不一致的多模态推理。

表征学习领域的进展（如JEPA、DINO）展示了在不依赖像素重建或密集监督的情况下，通过预测式潜空间建模可以学到更鲁棒、泛化性更强的特征。然而，这些方法主要面向视觉表征学习，并未直接用于生成接地的caption或同步跨模态语义。

核心矛盾在于：当前的对齐策略（如CLIP的对比学习、Q-Former的可学习查询）通常在预训练阶段静态完成，在与下游LLM集成时**不会持续调整对齐**。这留下了一个表征上的鸿沟——视觉和语言模型各自强大，但潜空间没有被直接共适应。

ViZer的切入角度是：**能否在训练过程中主动对齐视觉和语言表征，从而在无标注数据的情况下提升VLM性能？**这不是简单的自监督特征学习，而是直接优化跨模态对齐以服务生成式图像描述任务。

## 方法详解

### 整体框架
ViZer在冻结的视觉编码器和VLM的隐层特征空间之间引入一个轻量级映射器（mapper），通过对比损失训练该映射器实现视觉特征和文本特征的对齐。VLM本身使用LoRA微调，训练信号来自ViZer mapper提供的对齐损失——无需任何文本标签。

### 关键设计

1. **ViZer映射器（Alignment Mapper）**:

    - 定义映射函数 $M_\tau(\cdot) = h_\tau(f_\psi(\cdot))$，其中 $h$ 是MLP（将文本嵌入变换到视觉特征空间），$f_\psi$ 是VLM的transformer层（不含LM头）
    - 视觉特征 $F_I = V_\theta(I)$ 直接从冻结视觉编码器提取
    - 文本特征通过映射器处理：$\hat{F}_T = M_\tau(E_\phi(x_{:t}))$
    - 参数高效，可随数据集或模型规模灵活扩展
    - 设计动机：受联合嵌入原理启发，学习视觉和语言嵌入间的双向映射，不同于静态投影层，ViZer在训练中持续优化对齐

2. **两种映射器变体（ViZer$_{\text{GT}}$ 和 ViZer$_{\text{G}}$）**:

    - **ViZer$_{\text{GT}}$**：映射器在真实图像-文本对上训练，图像caption作为文本输入
    - **ViZer$_{\text{G}}$**：完全零标签——映射器使用VLM自身生成的caption训练，$\hat{F}_T = M_\tau(f_\psi(V_\theta(I) \circ E_\phi(P))_{t+1:})$，其中 $P$ 是caption提示
    - 只有ViZer$_{\text{G}}$是真正的无监督零标签方案
    - 设计动机：ViZer$_{\text{G}}$通过模型自身生成的caption建立对齐，形成自我改进的闭环

3. **零标签VLM训练**:

    - VLM使用LoRA微调（rank=32, alpha=64），仅在未标注的OpenImagesV7数据上训练
    - 损失函数为余弦相似度：$\mathcal{L}_{\text{zero}} = 1 - \frac{F_I \cdot \hat{F}_T}{\|F_I\| \times \|\hat{F}_T\|}$
    - 使用LoRA避免干扰预训练能力，可通过关闭LoRA恢复原始零样本性能
    - 设计动机：通过对齐的潜空间提供梯度信号替代缺失的文本标签，让VLM在无标注数据上自我提升

### 损失函数 / 训练策略
- Mapper和VLM各训练1个epoch
- 使用AdamW优化器，weight decay 0.01
- Mapper深度固定为2层MLP，宽度可变（最优256）
- Mapper在COCO+CC3M混合数据上训练，VLM严格在无标签的OpenImagesV7上训练
- 所有训练在单张RTX 4090 (24GB)上完成

## 实验关键数据

### 主实验

| 方法 | 模型 | COCO BLEU1 | COCO CIDEr | COCO CLIPS | CC3M CLIPS |
|------|------|------|------|------|------|
| Base | SmolVLM | 0.3784 | 0.276 | 0.2529 | 0.2617 |
| RL | SmolVLM | 0.3623 | 0.255 | 0.2506 | 0.2604 |
| ViZer$_{\text{GT}}$ | SmolVLM | **0.5564** | **0.505** | 0.2569 | **0.2647** |
| ViZer$_{\text{G}}$ | SmolVLM | 0.4081 | 0.337 | **0.2571** | 0.2636 |
| Base | Qwen2-VL | 0.5249 | 0.521 | 0.2693 | 0.2766 |
| ViZer$_{\text{G}}$ | Qwen2-VL | **0.5373** | 0.470 | **0.2744** | **0.2774** |

### 消融实验（Mapper超参数，SmolVLM-Base）

| ViZer变体 | 数据量 | 宽度 | COCO BLEU1 | COCO CIDEr | 说明 |
|-------|------|-----|------|------|------|
| ViZer$_{\text{GT}}$ | 10k | 256 | 0.4064 | 0.342 | 数据量少但效果不错 |
| ViZer$_{\text{GT}}$ | 40k | 256 | **0.4169** | **0.355** | 最优配置 |
| ViZer$_{\text{GT}}$ | 100k | 256 | 0.4161 | 0.354 | 过多数据无额外增益 |
| ViZer$_{\text{G}}$ | 10k | 256 | **0.4112** | **0.313** | ViZer$_{\text{G}}$偏好更少数据 |
| ViZer$_{\text{G}}$ | 40k | 256 | 0.3662 | 0.254 | 数据增多反而退化 |

### 关键发现
- CLIPScore在所有ViZer变体上持续提升，表明图像-caption语义一致性确实增强
- 传统指标（BLEU、CIDEr）提升有限甚至不升反降——因为ViZer生成的caption包含更多参考caption中不存在的正确细节，被当作"错误"惩罚
- 定性评估显示ViZer显著改善：将"ITAP of an airplane"提升为"ITAP of an airplane flying over power lines"；将"\<PERSON\> in 2008"改为"Woman surfing in the ocean"
- **小数据反而更好**：ViZer$_{\text{GT}}$的最优数据量约40k，ViZer$_{\text{G}}$仅需约10k，过多数据导致mapper过拟合
- RL baseline（使用奖励模型）效果甚微，因为奖励信号倾向于保守更新以保护预训练表征

## 亮点与洞察
- **范式突破**：证明了VLM可以仅通过未标注图像数据自我提升caption能力，开创了视觉-语言领域的零标签增强训练数据路
- **评估指标反思**：深刻揭示了CIDEr/BLEU等参考依赖指标的局限性——它们惩罚正确但超出参考范围的细节，这对自监督方法极不公平
- **架构通用性**：ViZer可即插即用地集成到任何使用视觉编码器的VLM架构，训练仅需单张24GB GPU

## 局限与展望
- 当baseline caption质量极差时，ViZer改进有限（如消防站的年份猜测）
- 目前仅验证了图像描述任务，向VQA扩展存在挑战——VQA关注局部区域而非全局语义
- 缺乏合适的自动化评估指标——需要开发不依赖参考文本且具有图像理解能力的评估方法
- 在分布外图像（医学、卫星等）上的表现未知

## 相关工作与启发
- **vs CLIP/ALIGN**: CLIP通过对比学习建立静态对齐，ViZer在训练过程中持续动态对齐且服务于生成任务
- **vs BLIP-2 Q-Former**: Q-Former学习可学习查询来桥接模态，但仍依赖标注caption数据；ViZer完全摆脱文本标签
- **vs I-JEPA/DINO**: 这些方法学习视觉表征但不直接服务于跨模态生成任务；ViZer将联合嵌入思想扩展到视觉-语言任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 零标签caption训练是新颖方向，但思路上是联合嵌入/对比学习在VLM上的自然延伸
- 实验充分度: ⭐⭐⭐ 定量结果有限（指标不适用），主要靠定性比较；在更大模型和更多任务上的验证不足
- 写作质量: ⭐⭐⭐⭐ 动机清晰，对评估指标局限性的讨论深入，但某些设计决策的解释可更充分
- 价值: ⭐⭐⭐⭐ 为利用海量未标注图像数据提升VLM提供了实用路径，对标注稀缺场景有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] RATE-Nav: Region-Aware Termination Enhancement for Zero-shot Object Navigation with Vision-Language Models](../../ACL2025/multimodal_vlm/rate-nav_region-aware_termination_enhancement_for_zero-shot_object_navigation_wi.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[CVPR 2026\] UARE: A Unified Vision-Language Model for Image Quality Assessment, Restoration, and Enhancement](../../CVPR2026/multimodal_vlm/uare_a_unified_vision-language_model_for_image_quality_assessment_restoration_an.md)
- [\[NeurIPS 2025\] CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness](capability_a_comprehensive_visual_caption_benchmark_for_eval.md)
- [\[CVPR 2026\] CaptionQA: Is Your Caption as Useful as the Image Itself?](../../CVPR2026/multimodal_vlm/captionqa_is_your_caption_as_useful_as_the_image_itself.md)

</div>

<!-- RELATED:END -->
