---
title: >-
  [论文解读] Pixel-Level Reasoning Segmentation via Multi-turn Conversations
description: >-
  [语义分割] 提出像素级推理分割 (Pixel-level RS) 新任务，通过多轮对话逐步理解用户意图实现细粒度分割，构建了包含 24k 对话轮次的 PRIST 数据集，并设计 MIRAS 框架在分割精度和推理能力上均超越现有基线。 - 现有问题： 当前视觉感知系统仅支持单轮对话中的区域级分割 (region-level…
tags:
  - "语义分割"
---

# Pixel-Level Reasoning Segmentation via Multi-turn Conversations

| 会议 | arXiv | 代码 | 领域 | 关键词 |
|------|-------|------|------|--------|
| ACL 2025 | [2502.09447](https://arxiv.org/abs/2502.09447) | [GitHub](https://github.com/ccccai239/PixelRIST) | segmentation | 像素级推理分割, 多轮对话, MLLM, SAM, 语义区域对齐 |

## 一句话总结

提出像素级推理分割 (Pixel-level RS) 新任务，通过多轮对话逐步理解用户意图实现细粒度分割，构建了包含 24k 对话轮次的 PRIST 数据集，并设计 MIRAS 框架在分割精度和推理能力上均超越现有基线。

## 研究背景与动机

- **现有问题**: 当前视觉感知系统仅支持单轮对话中的区域级分割 (region-level segmentation)，依赖复杂且显式的查询指令，无法在像素级别进行推理，也无法理解用户在交互中动态演变的意图。
- **核心差距**: 现有推理分割方法 (如 LISA, PixelLM) 存在两大局限：(1) 依赖单轮模糊查询，不能充分理解用户不断变化的意图；(2) 缺乏像素级分割能力，只能通过一步解释实现粗糙的区域级分割。
- **研究动机**: 多轮交互可以逐步澄清用户的模糊指令（如"做面包"），通过渐进式对话最终聚焦到具体物体，实现像素级精确分割。
- **新任务定义**: Pixel-level RS 要求系统通过多轮对话追踪用户演变意图，同时生成像素级分割掩码和文本推理链。

## 方法详解

### 整体框架

MIRAS (Multi-turn Interactive ReAsoning Segmentation) 框架包含三个核心组件：
1. **双视觉编码器** (Dual Visual Encoder): 提取多尺度视觉特征
2. **多模态大语言模型 (MLLM)**: 基于 LLaVA 进行多轮对话和推理
3. **掩码解码器** (Mask Decoder): 基于 SAM 生成像素级分割掩码

通过引入特殊 token `[SEG]` 作为分割区域占位符，实现推理与分割的端到端处理。

### 关键设计

1. **双视觉编码器融合**: 高分辨率图像 (768×768) 经 ConvNext-L 处理，低分辨率图像 (336×336) 经 CLIP-L/14 处理，通过交叉注意力模块融合多尺度特征，增强视觉细节捕捉。
2. **语义区域对齐策略**: 设计分割提示模板 `[OBJ]{CLASS}[SEG]`，利用 `[OBJ]` 提取相关子序列，通过交叉注意力注入语义信息到掩码解码器，解决因物体描述长度变化导致的维度不匹配问题。
3. **PRIST 数据集构建**: 基于推理树的三步渐进式对话自动生成流水线——(Step 1) 提取可见元素, (Step 2) 构建推理问题和推理树, (Step 3) 将推理树节点组织为多轮对话格式。

### 损失函数

$$\mathcal{L} = \lambda_t \mathcal{L}_t + \lambda_{bce} \text{BCE}(\mathcal{M}, \hat{\mathcal{M}}) + \lambda_{dice} \text{DICE}(\mathcal{M}, \hat{\mathcal{M}})$$

其中 $\lambda_t=1.0$, $\lambda_{bce}=2.0$, $\lambda_{dice}=0.5$。两阶段训练：Stage-1 掩码-文本对齐预训练，Stage-2 在 PRIST 数据集上指令微调。仅训练掩码解码器和投影层，冻结图像编码器和 MLLM。

## 实验

### 主实验

| 模型 | CIoU | Prec. | Recall | F1 | BLEU-4 | ROUGE_L | METEOR |
|------|------|-------|--------|-----|--------|---------|--------|
| GPT-4o (zero-shot) | 14.13 | 17.35 | 35.01 | 23.18 | 4.30 | 26.35 | 28.55 |
| OMG-LLaVA (zero-shot) | 9.67 | 16.67 | **77.80** | 27.46 | 8.70 | 23.47 | 27.90 |
| LISA (fine-tuned) | 11.23 | 26.23 | 29.22 | 27.64 | 7.81 | 27.84 | 30.74 |
| OMG-LLaVA (fine-tuned) | 13.84 | 21.54 | 49.31 | 29.98 | 11.21 | 30.59 | 39.18 |
| **MIRAS (Stage-2)** | **14.72** | **24.22** | 40.61 | **30.34** | 8.51 | **30.82** | **40.06** |

### 消融实验 (推理质量)

| 模型 | PR | LC | CC | TR | Win Rate(%) |
|------|-----|-----|-----|-----|-------------|
| Human | 4.03 | 4.04 | — | — | — |
| MIRAS | **最高** | **最高** | **最高** | **最高** | **42%** |

微调后各模型平均 Win Rate 提升约 10%，MIRAS 在四项推理指标上均达到 SOTA。

### 关键发现

1. **PRIST 微调的普适性**: 在 PRIST 上微调后，所有分割模型的 CIoU 和 Precision 均显著提升（如 OMG-LLaVA CIoU ↑43%，LISA Precision ↑71%）。
2. **精度-召回权衡**: 微调后模型优先提高分割特异性而非泛化能力，召回率有所下降但精确度大幅提升，符合像素级 RS 的任务目标。
3. **MIRAS 双重能力**: 在分割和对话响应上同时优化，Dist-1/2 达 15.7/49.2，表明生成文本的多样性最高。

## 亮点

- 定义了像素级推理分割新任务，填补了多轮对话驱动的细粒度分割空白
- 构建了 PRIST 数据集（24k 对话, 8.3k 场景, 53% 细粒度目标），基于推理树的自动生成流水线既高效又保证质量
- 语义区域对齐策略通过注入目标语义信息显著提升掩码解码器的分割精度

## 局限性

- 数据集规模相对较小（仅 2,800 张图像），可能限制泛化能力
- 绝对分割性能仍然偏低（CIoU 最高仅 14.72），像素级推理分割仍有大量提升空间
- 仅冻结 MLLM 和图像编码器进行训练，端到端全参数微调可能带来进一步提升
- 推理质量评估依赖 GPT-4o 作为裁判，可能引入评估偏差

## 相关工作

- **推理分割数据集**: ReasonSeg (Lai et al., 2023) 首次提出基于复杂查询的分割数据集，但规模小且不支持多轮交互；后续 GREN (Yuan et al., 2024) 等扩展到多目标但仍限于单轮推理
- **区域级分割模型**: LISA 集成分割模块与 LLM 实现端到端训练；PixelLM 支持多目标分割；OMG-LLaVA 增强区域理解；但这些方法均缺乏多轮推理能力
- **多模态大语言模型**: InternVL2, Qwen2-VL 等通用 MLLM 具有强视觉感知但缺乏像素级分割能力

## 评分

| 维度 | 分数 (1-10) |
|------|------------|
| 创新性 | 8 |
| 技术深度 | 7 |
| 实验充分性 | 8 |
| 写作质量 | 7 |
| **综合** | **7.5** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](../../ICML2025/segmentation/actionpiece_contextually_tokenizing_action_sequences_for_generative_recommendati.md)
- [\[NeurIPS 2025\] UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning](../../NeurIPS2025/segmentation/unipixel_unified_object_referring_and_segmentation_for_pixel-level_visual_reason.md)
- [\[ICML 2025\] unMORE: Unsupervised Multi-Object Segmentation via Center-Boundary Reasoning](../../ICML2025/segmentation/unmore_unsupervised_multi-object_segmentation_via_center-boundary_reasoning.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](../../CVPR2025/segmentation/dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)
- [\[AAAI 2026\] JoDiffusion: Jointly Diffusing Image with Pixel-Level Annotations for Semantic Segmentation Promotion](../../AAAI2026/segmentation/jodiffusion_jointly_diffusing_image_with_pixel-level_annotations_for_semantic_se.md)

</div>

<!-- RELATED:END -->
