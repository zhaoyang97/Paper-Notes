---
title: >-
  [论文解读] ConText: Driving In-context Learning for Text Removal and Segmentation
description: >-
  [ICML 2025][语义分割][视觉上下文学习] 首次将视觉上下文学习（V-ICL）范式应用于OCR任务，提出任务链式提示（task-chaining prompting）、上下文感知聚合（CAA）和自提示策略（self-prompting）三项关键设计，在文本去除和分割任务上大幅超越现有V-ICL通用模型和专用模型，分别取得 +4.50 PSNR 和 +3.34% fgIoU 的提升。
tags:
  - "ICML 2025"
  - "语义分割"
  - "视觉上下文学习"
  - "文本分割"
  - "文本去除"
  - "任务链式推理"
  - "MAE"
---

# ConText: Driving In-context Learning for Text Removal and Segmentation

**会议**: ICML 2025  
**arXiv**: [2506.03799](https://arxiv.org/abs/2506.03799)  
**代码**: [https://github.com/Ferenas/ConText](https://github.com/Ferenas/ConText)  
**领域**: 分割  
**关键词**: 视觉上下文学习, 文本分割, 文本去除, 任务链式推理, MAE

## 一句话总结

首次将视觉上下文学习（V-ICL）范式应用于OCR任务，提出任务链式提示（task-chaining prompting）、上下文感知聚合（CAA）和自提示策略（self-prompting）三项关键设计，在文本去除和分割任务上大幅超越现有V-ICL通用模型和专用模型，分别取得 +4.50 PSNR 和 +3.34% fgIoU 的提升。

## 研究背景与动机

视觉上下文学习（V-ICL）是从NLP中的in-context learning（ICL）范式借鉴而来，核心思想是通过少量输入-输出示例（demonstrations）作为上下文，引导模型对新查询进行预测。现有V-ICL方法主要基于MAE（Masked Autoencoders），将两组 image-label 拼接为网格输入，遮盖查询标签区域后进行掩码重建。

然而，现有方法面临三个核心问题：

**单任务中心**：每次只能进行一个任务的推理，无法利用任务间的内在关联来增强ICL能力

**上下文无关融合**：输入-输出的特征融合仅在单个pair内部进行线性叠加，缺乏跨demonstration的上下文信息交互

**上下文同质性假设**：要求demonstration和query具有相同类别的对象（如飞机对飞机），但文本识别中字体、风格、语言的高度异质性使得找到"同类"demonstration非常困难

本文的核心动机是：**能否像LLM中的Chain-of-Thought一样，将多个相关视觉任务链接为一个整体提示，通过任务间的互补逻辑来增强V-ICL的推理能力？**

## 方法详解

### 整体框架

ConText 基于 ViT-L + decoder 的MAE架构（以SegGPT为基础），通过三个专门设计的模块来增强基线：

1. **任务链式提示（Task Chaining）**：将原始的 image-label 二元组扩展为 image-removal-segmentation 三元组
2. **上下文感知聚合（Context-Aware Aggregation, CAA）**：通过交叉注意力将prompt的模式信息注入query表示
3. **自提示策略（Self-Prompting, SP）**：以一定概率使用相同样本作为demonstration和query，保持ICL可学习性

### 关键设计

#### 1. 任务链式提示（Task Chaining）

核心观察：文本分割和文本去除之间存在隐含的任务级逻辑关联——分割掩码理论上应对应原始图像与去除文本后图像之间的视觉差异。

- 将原始 prompt 从 `[Image, Label]` 扩展为 `[Image, Removal, Segmentation]`
- 新的组合输入：$\mathbf{F} = [\mathbf{F}_I, \mathbf{F}_O, \mathbf{F}_Y] \in \mathbb{R}^{3 \times 2h \times 3w}$
- 训练时对 removal 和 segmentation 标签同时执行掩码重建，且保持两者掩码的空间一致性：$\mathbf{M}_O = \mathbf{M}_Y$
- 使用权重共享的解码器分别重建两个任务的标签
- 推理时遮盖query的removal和segmentation标签位置，端到端生成所有任务输出

先导实验验证：在Painter上引入removal标签辅助分割（Rem-based Seg.），TotalText fgIoU从60.60%提升到63.22%（+2.62%）；反向引入分割辅助去除（Seg-based Rem.），SCUT-Ens PSNR从36.15提升到37.02（+0.87），证实了任务链的有效性。

#### 2. 上下文感知聚合（CAA）

受ICL机制解释性研究的启发（label位置在浅层逐步提取demonstration信息，最终标签吸收所有信息），设计两步融合：

**第一步：上下文无关融合**（类似基线的线性融合）

$$\tilde{\mathbf{F}}_1 = [\mathbf{I}_i + \tilde{\mathbf{O}}_i + \alpha_y \tilde{\mathbf{Y}}_i, \quad \mathbf{I}_j + \alpha_o \tilde{\mathbf{O}}_j + \tilde{\mathbf{Y}}_j]$$

其中 $\alpha_o$, $\alpha_y$ 为可学习权重，控制removal和segmentation特征的贡献，此步聚焦于 demonstration 内部的 inter-task 融合。

**第二步：跨 demonstration 上下文融合（CAA）**

$$\tilde{\mathbf{F}}_2 = [\phi(\tilde{\mathbf{F}}_{O_i}, \tilde{\mathbf{F}}_j), \phi(\tilde{\mathbf{F}}_{Y_i}, \tilde{\mathbf{F}}_j), \phi(\tilde{\mathbf{F}}_{O_j}, \tilde{\mathbf{F}}_i), \phi(\tilde{\mathbf{F}}_{Y_j}, \tilde{\mathbf{F}}_i)]$$

其中 $\phi(\text{query}, \text{key/value})$ 为共享的交叉注意力映射。最终标签表示为 $\tilde{\mathbf{F}}_1 + \tilde{\mathbf{F}}_2$，每个 label 特征都能显式地从另一个 demonstration 提取信息，实现更全面的上下文理解。

#### 3. 自提示策略（Self-Prompting）

针对文本识别中视觉异质性问题（字体、风格、语言差异巨大，难以找到"同类"demonstration），以概率 $p=0.2$ 在训练时使用相同的输入-输出pair作为demonstration和query（$\tilde{\mathbf{F}}_i = \tilde{\mathbf{F}}_j$）。

这迫使模型同时维持特定任务推理能力和泛化的上下文学习能力，防止模型退化为无需demonstration的specialist。但过高的自提示概率（如0.6）会因减少demonstration多样性而降低任务性能。

### 损失函数 / 训练策略

- **重建损失**：对removal和segmentation标签分别使用 smooth-L1 loss 进行像素重建，removal loss权重设为0.3
- **像素级监督**：额外引入基于交叉熵的像素级监督 $\mathcal{L}_{pix}$（权重1.0），通过一个仅在训练时使用的轻量解码器（两层卷积）来增强细粒度文字级别的识别
- **掩码比例**：85%（消融实验验证最优）
- **优化器**：AdamW，学习率0.0001，权重衰减0.1，cosine调度
- **训练规模**：16×A100 (80GB)，batch size 64（单卡2 + 2步梯度累积），150 epoch
- **两种训练模式**：
    - ConText：仅用 HierText 训练（用于消融）
    - ConText$_V$：HierText + TextSeg + TotalText + SCUT-EnsText（用于与specialist对比）

## 实验关键数据

### 主实验

**与V-ICL通用模型对比**（仅在HierText训练）：

| 方法 | 文本去除 PSNR (Avg) | 文本去除 FID (Avg) | 文本分割 fgIoU (Avg) |
|------|---------------------|--------------------|-----------------------|
| MAE-VQGAN | 28.30 | 41.70 | 6.97% |
| Painter Rem.+Seg. | 32.34 | 24.73 | 67.16% |
| SegGPT Rem.+Seg. | 33.05 | 24.34 | 68.34% |
| **ConText** | **38.36** | **11.04** | **76.77%** |

**与文本分割专用模型对比**（ConText$_V$）：

| 方法 | HierText fgIoU | TotalText fgIoU | *FST fgIoU | TextSeg fgIoU |
|------|----------------|-----------------|-----------|-------------|
| Hi-SAM | 77.76 | 84.59 | - | 88.77 |
| EAFormer | - | 82.73 | 72.63 | 88.06 |
| UPOCR | - | - | - | 88.76 |
| **ConText$_V$** | **81.21** | **85.19** | **75.90** | **89.74** |

**与文本去除专用模型对比**（SCUT-EnsText）：

| 方法 | PSNR↑ | MSSIM↑ | MSE↓ | FID↓ |
|------|-------|--------|------|------|
| ViTEraser | 36.87 | 97.51% | 0.05 | 10.15 |
| UPOCR | 37.14 | 97.62% | 0.04 | 10.47 |
| **ConText$_V$** | **40.83** | **98.76%** | **0.03** | 11.63 |

### 消融实验

**各模块有效性**（TotalText分割/SCUT-Ens去除，RS=随机demonstration）：

| 配置 | Seg. fgIoU (RS / GT差值) | Rem. PSNR (RS / GT差值) | 说明 |
|------|--------------------------|-------------------------|------|
| Baseline (SegGPT) | 68.53 / +1.57 | 34.42 / +0.17 | 多任务微调SegGPT |
| + Linear Fusion | 72.14 / +1.08 | 35.75 / +0.41 | 上下文无关融合，+3.61% |
| + CAA | 79.14 / +0.65 | 38.59 / +0.37 | 交叉注意力融合，+10.61% |
| + CAA + SP-0.2 | 78.02 / +3.98 | 37.67 / +1.42 | 最佳平衡，ICL能力大幅恢复 |
| + CAA + SP-0.6 | 77.14 / +5.83 | 36.12 / +2.13 | ICL能力更强但任务性能下降 |

**计算开销**：

| 配置 | 训练时间/epoch | 推理时间 | FLOPs增量 |
|------|--------------|---------|----------|
| Baseline | 3.8 min | 0.09 sec | 666.76G |
| + SP | 4.2 min | 0.09 sec | 0% |
| + CAA | 4.6 min | 0.12 sec | +2% |
| + CAA + SP | 4.8 min | 0.12 sec | +2% |

### 关键发现

1. **任务链式提示有效**：先导实验证实利用任务间逻辑关联（分割↔去除）可显著提升ICL推理，Rem-based Seg. 提升 +2.62% fgIoU
2. **CAA是性能提升主力**：从baseline到+CAA，分割 fgIoU 提升 +10.61%，去除 PSNR 提升 +4.17，但会导致模型退化为specialist（GT-RS差值很小）
3. **SP恢复ICL可学习性**：SP-0.2 使RS-GT性能差从+0.65扩大到+3.98（分割），证明模型能从demonstration中学到信息
4. **training-free prompting 能力**：在从未参与训练的PromptText数据集上，ConText能理解人工标注的圈/框/笔画等视觉提示，表现远超所有specialist和V-ICL通用模型
5. **多demonstration和双重推理**：5-shot推理带来 +0.62% fgIoU 和 +0.78 PSNR，双重推理带来 +0.25% fgIoU 和 +0.44 PSNR

## 亮点与洞察

1. **Chain-of-Thought在视觉中的体现**：将NLP的链式思维推理理念迁移到视觉任务，通过 image→removal→segmentation 的任务链提供丰富中间表示，是对V-ICL范式的根本性增强
2. **ICL可学习性的量化**：提出用RS和GT demonstration的性能差值来度量模型的in-context可学习性，ConText在分割任务上达到 +5.39% fgIoU 的差值，远超其他方法
3. **轻量但关键的设计**：CAA仅增加2% FLOPs，SP在推理时零开销，但组合起来带来了质的提升
4. **首个OCR领域的V-ICL框架**：开辟了V-ICL在细粒度文本识别任务上的新方向

## 局限与展望

1. **分割标签依赖**：对于缺乏人工标注removal标签的数据集，需依赖ViTEraser生成伪标签，引入噪声
2. **合成-真实域差距**：在SCUT-Syn合成数据集上表现不如部分specialist，暴露了域迁移问题
3. **多demonstration收益有限**：5-shot仅比1-shot提升0.62%，异质性文本的"同类"demonstration筛选仍待探索
4. **自提示概率的sensitivity**：SP-0.2和SP-0.6有明显的任务性能/ICL能力权衡，最优值的自动选择尚未解决
5. **FID指标上的不足**：在SCUT-EnsText上ConText$_V$的FID（11.63）略高于ViTEraser（10.15），生成质量的感知一致性仍有提升空间

## 相关工作与启发

- **V-ICL基线**：MAE-VQGAN、Painter、SegGPT 构成了composited-prompting V-ICL的主流范式
- **任务链灵感**：来源于NLP的Chain-of-Thought (Wei et al., 2022) 和 Chain-of-Thought reasoning without prompting (Wang & Zhou, 2024)
- **ICL机制**：label位置作为信息锚点的假说 (Wang et al., 2023a; Yu & Ananiadou, 2024) 直接启发了CAA的设计
- **文本分割SOTA**：Hi-SAM基于SAM的层次文本分割、EAFormer的边缘感知Transformer
- **文本去除SOTA**：ViTEraser的一阶段ViT方案、UPOCR的统一像素级OCR接口
- **启发方向**：可将task-chaining思想推广到其他具有隐含任务关联的视觉任务组合（如检测↔分割、深度↔法线）

## 评分

| 维度 | 评分 | 说明 |
|------|------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次将V-ICL应用于OCR，任务链式提示是新颖的视觉CoT |
| 技术深度 | ⭐⭐⭐⭐ | CAA和SP设计有理论依据，消融充分 |
| 实验充分性 | ⭐⭐⭐⭐⭐ | 6个数据集，多维指标，丰富消融和分析 |
| 写作质量 | ⭐⭐⭐⭐ | 叙事清晰，从基线特征出发逐步引出改进 |
| 实用价值 | ⭐⭐⭐⭐ | 代码开源，端到端多任务，额外开销仅+2% FLOPs |
| 总评 | ⭐⭐⭐⭐ | 扎实的工作，将V-ICL成功拓展到OCR领域并取得显著SOTA |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](../../ICCV2025/segmentation/cavis_context-aware_video_instance_segmentation.md)
- [\[CVPR 2026\] INSID3: Training-Free In-Context Segmentation with DINOv3](../../CVPR2026/segmentation/insid3_training-free_in-context_segmentation_with_dinov3.md)
- [\[CVPR 2025\] The Power of Context: How Multimodality Improves Image Super-Resolution](../../CVPR2025/segmentation/the_power_of_context_how_multimodality_improves_image_super-resolution.md)
- [\[ICCV 2025\] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation](../../ICCV2025/segmentation/score_scene_context_matters_in_openvocabulary_remote_sensing.md)
- [\[CVPR 2026\] Annotation-Efficient Coreset Selection for Context-dependent Segmentation](../../CVPR2026/segmentation/annotation-efficient_coreset_selection_for_context-dependent_segmentation.md)

</div>

<!-- RELATED:END -->
