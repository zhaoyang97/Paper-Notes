---
title: >-
  [论文解读] Visual Modality Prompt for Adapting Vision-Language Object Detectors
description: >-
  [ICCV 2025][目标检测][视觉提示] 提出 ModPrompt，一种基于编码器-解码器的视觉提示策略，将视觉-语言目标检测器（如 YOLO-World、Grounding DINO）适应到红外和深度等新模态，同时保留零样本检测能力。 视觉-语言目标检测器（如 YOLO-World、Grounding DINO）通过…
tags:
  - "ICCV 2025"
  - "目标检测"
  - "视觉提示"
  - "模态适应"
  - "视觉-语言检测器"
  - "零样本检测"
  - "跨模态迁移"
---

# Visual Modality Prompt for Adapting Vision-Language Object Detectors

**会议**: ICCV 2025  
**arXiv**: [2412.00622](https://arxiv.org/abs/2412.00622)  
**代码**: [GitHub](https://github.com/heitorrapela/ModPrompt)  
**领域**: 目标检测  
**关键词**: 视觉提示, 模态适应, 视觉-语言检测器, 零样本检测, 跨模态迁移

## 一句话总结

提出 ModPrompt，一种基于编码器-解码器的视觉提示策略，将视觉-语言目标检测器（如 YOLO-World、Grounding DINO）适应到红外和深度等新模态，同时保留零样本检测能力。

## 研究背景与动机

视觉-语言目标检测器（如 YOLO-World、Grounding DINO）通过融合文本语义和视觉特征，在 RGB 图像上展现出强大的零样本检测能力。然而，当测试域发生较大模态偏移（如从 RGB 到红外或深度图）时，这些检测器的性能会显著下降。

现有适应方法存在以下局限性：

**全量微调（Full Fine-tuning）**：虽能提升目标模态的检测精度，但会导致灾难性遗忘，丧失零样本检测能力

**传统视觉提示（Visual Prompt）**：对每张图像施加相同的线性提示变换（如固定 patch、随机 padding），不依赖输入图像内容，在大模态偏移场景下效果有限

**图像翻译方法（如 HalluciDet、ModTr）**：仅适用于传统检测器，未探索视觉-语言检测器的跨模态适应；且一些方法会丢失预训练知识

核心动机在于：能否在不修改检测器参数的前提下，通过一个依赖于输入图像的可学习视觉提示，将新模态图像"翻译"为检测器更容易理解的伪 RGB 表示，从而同时实现高精度检测和零样本能力的保留？

## 方法详解

### 整体框架

ModPrompt 的核心思想是在输入空间（像素级别）进行模态适应。整体流程为：将目标模态图像 $x$ 通过一个可学习的编码器-解码器网络 $h_\vartheta$ 生成视觉提示，然后将提示与原始图像相加，形成伪 RGB 图像送入冻结的视觉-语言检测器。同时，引入 MPDR（Modality Prompt Decoupled Residual）机制对文本嵌入进行解耦适应。

### 关键设计

1. **ModPrompt（模态提示编码器-解码器）**:

    - 功能：根据输入图像动态生成视觉提示，实现像素级的模态翻译
    - 核心思路：采用基于 U-Net 的编码器-解码器结构（可使用 MobileNet 或 ResNet 作为骨干网络），将输入图像映射为 3 通道的提示图像，输出值约束在 $[0,1]$ 范围内。训练目标为检测损失而非重建损失：
    $\mathcal{C}_{\text{mp}}(\vartheta) = \frac{1}{|\mathcal{D}|}\sum_{(x,Y)\in\mathcal{D}} \mathcal{L}_{det}(f_\theta(x + h_\vartheta(x)), Y)$
      其中 $f_\theta$ 为冻结的检测器，$h_\vartheta(x)$ 为依赖输入的视觉提示
    - 设计动机：与固定视觉提示不同，ModPrompt 是输入条件化的——不同图像生成不同的提示,能更好地增强目标区域并抑制背景噪声，特别适用于模态差异大的场景

2. **MPDR（模态提示解耦残差）**:

    - 功能：在文本嵌入空间进行高效的模态适应，同时保留原始零样本知识
    - 核心思路：预先计算每个目标类别的文本嵌入（离线生成），然后学习一组可训练的残差参数 $\phi$，将其加到冻结的嵌入上。最终训练目标为：
    $\mathcal{C}_{\text{mp-tp}}(\vartheta, \phi) = \mathcal{C}_{\text{mp}}(\vartheta) + \mathcal{C}_{\text{tp}}(\phi)$
    - 设计动机：通过解耦策略，在测试时可通过零掩码关闭 MPDR 恢复完全的零样本嵌入知识，或开启以使用适应后的嵌入，无推理开销

3. **检测器无关设计**:

    - 功能：使 ModPrompt 可灵活集成到不同架构的视觉-语言检测器中
    - 核心思路：由于提示作用在输入像素层面而非特征层面，与检测器骨干类型无关（CNN 或 Transformer 均可）
    - 设计动机：已有方法大多和特定检测器绑定，而 ModPrompt 在 YOLO-World（CNN 骨干 + CLIP）和 Grounding DINO（Swin Transformer + BERT）上均适用

### 损失函数 / 训练策略

- 训练损失为原始检测器的检测损失 $\mathcal{L}_{det}$，包含分类和回归损失
- 仅训练编码器-解码器参数 $\vartheta$ 和 MPDR 参数 $\phi$，检测器所有参数冻结
- YOLO-World 训练 80 个 epoch，Grounding DINO 训练 60 个 epoch
- 文本嵌入分别用 CLIP-ViT-base-patch32（YOLO-World）和 BERT-base-uncased（Grounding DINO）离线提取

## 实验关键数据

### 主实验

| 数据集 | 方法 | YOLO-World AP50 | YOLO-World AP | Grounding DINO AP50 | Grounding DINO AP |
|-------|------|----------------|--------------|--------------------|--------------------|
| LLVIP-IR | Zero-Shot | 81.00 | 53.20 | 85.50 | 56.50 |
| LLVIP-IR | Full FT | 97.43 | 67.73 | 97.17 | 67.83 |
| LLVIP-IR | Visual Prompt (WM) | 82.00 | 50.90 | 69.57 | 40.77 |
| LLVIP-IR | **ModPrompt** | **92.80** | **62.87** | **93.13** | **60.10** |
| NYUv2-Depth | Zero-Shot | 4.80 | 3.00 | 8.30 | 5.30 |
| NYUv2-Depth | Full FT | 49.90 | 33.57 | 51.60 | 35.77 |
| NYUv2-Depth | **ModPrompt** | **37.17** | **24.93** | **21.70** | **14.13** |

ModPrompt 在红外和深度模态上均显著超越所有视觉提示基线，在 LLVIP 上接近全量微调性能。

### 消融实验

| 配置 | LLVIP AP50 | COCO AP50 | 平均 | 说明 |
|------|-----------|----------|------|------|
| Zero-Shot | 81.00 | 51.90 | 66.45 | 基线 |
| Full FT | 97.43 | 0.10 | 48.77 | 丧失零样本能力 |
| Head FT | 93.57 | 0.66 | 47.12 | 同样灾难性遗忘 |
| WM | 87.47 | 51.90 | 69.69 | 保留零样本但精度有限 |
| **ModPrompt** | **95.63** | **51.90** | **73.77** | 高精度 + 完整零样本 |

- 可训练参数：ModPrompt 仅 3.08M，远低于 Full FT 的 76.81M
- MobileNet 骨干即可接近 ResNet 的检测性能，更适合实时应用
- MPDR 对几乎所有视觉提示策略均带来额外增益（+0.6 到 +8.0 AP50）

### 关键发现

1. 传统视觉提示（固定 patch、随机 patch）在模态适应场景下甚至可能劣于零样本，因为它们不依赖输入图像内容
2. ModPrompt 生成的视觉提示会在图像上产生"伪影"来增强目标区域并抑制背景
3. 在 NYUv2（深度图）上零样本性能极低（AP 仅 3-5%），说明 RGB 预训练模型在跨模态场景面临巨大挑战

## 亮点与洞察

- **实用性强**：ModPrompt 在保留零样本能力的同时实现了接近全量微调的精度，这在实际部署中极有价值——同一模型可同时处理 RGB 和新模态任务
- **设计简洁优雅**：输入条件化的视觉提示思想直观且有效，将 U-Net 用于检测引导的图像翻译是一个巧妙的转变
- **首次系统研究**：据我们所知，这是首个聚焦于将 VLM 检测器适应到新视觉模态的工作

## 局限与展望

1. 在精细定位（AP75 和 AP）上仍与全量微调有差距，尤其是小目标场景
2. 目前仅在红外和深度两种模态上验证，未涉及 SAR、热成像灰度等
3. 编码器-解码器会引入额外推理延迟，虽然 MobileNet 版本较轻量，但仍非零开销
4. 仅在 LLVIP（行人类别）和 NYUv2（室内场景）上验证，数据集规模和类别多样性相对有限

## 相关工作与启发

- 与 Co-op/VPT 等文本或特征层面的提示方法形成对比，ModPrompt 在像素层面操作，更适合处理大模态偏移
- HalluciDet 和 ModTr 的图像翻译思路是 ModPrompt 的先驱，但未利用视觉-语言模型的优势
- 残差解耦嵌入学习（MPDR）的思想来自 Task Residual，但首次应用于检测器的文本嵌入适应

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统探索 VLM 检测器的跨模态适应，设计简洁有效
- 实验充分度: ⭐⭐⭐⭐ 两个检测器、三个数据集、多种基线、丰富消融
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，方法表述直观
- 价值: ⭐⭐⭐⭐ 对多模态检测部署有实际意义，开源代码可复现

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] EvRT-DETR: Latent Space Adaptation of Image Detectors for Event-based Vision](evrt-detr_latent_space_adaptation_of_image_detectors_for_event-based_vision.md)
- [\[AAAI 2026\] T-Rex-Omni: Integrating Negative Visual Prompt in Generic Object Detection](../../AAAI2026/object_detection/t-rex-omni_integrating_negative_visual_prompt_in_generic_object_detection.md)
- [\[AAAI 2026\] Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](../../AAAI2026/object_detection/correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [\[ICCV 2025\] Visual-RFT: Visual Reinforcement Fine-Tuning](visual-rft_visual_reinforcement_fine-tuning.md)
- [\[ICCV 2025\] Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights](revisiting_adversarial_patch_defenses_on_object_detectors_unified_evaluation_lar.md)

</div>

<!-- RELATED:END -->
