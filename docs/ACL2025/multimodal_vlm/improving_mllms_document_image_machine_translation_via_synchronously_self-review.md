---
title: >-
  [论文解读] Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency
description: >-
  [ACL 2025 (Findings)][多模态VLM][文档图像翻译] 本文提出 Synchronously Self-Reviewing (SSR) 范式，通过在文档图像翻译过程中让 MLLM 先生成 OCR 文本再生成翻译文本，利用"双语认知优势"缓解微调导致的灾难性遗忘，同时提升 OCR 和文档图像机器翻译（DIMT）的性能。
tags:
  - "ACL 2025 (Findings)"
  - "多模态VLM"
  - "文档图像翻译"
  - "多模态大模型"
  - "OCR"
  - "灾难性遗忘"
  - "自审查机制"
---

# Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency

**会议**: ACL 2025 (Findings)  
**arXiv**: [2507.08309](https://arxiv.org/abs/2507.08309)  
**代码**: 无  
**领域**: 多模态VLM / 机器翻译  
**关键词**: 文档图像翻译, 多模态大模型, OCR, 灾难性遗忘, 自审查机制

## 一句话总结

本文提出 Synchronously Self-Reviewing (SSR) 范式，通过在文档图像翻译过程中让 MLLM 先生成 OCR 文本再生成翻译文本，利用"双语认知优势"缓解微调导致的灾难性遗忘，同时提升 OCR 和文档图像机器翻译（DIMT）的性能。

## 研究背景与动机

**领域现状**：多模态大模型（MLLMs）在文档图像任务上表现出色，尤其在光学字符识别（OCR）方面已具备很强的能力。文档图像机器翻译（DIMT）是一项更复杂的任务，要求模型同时处理跨模态（图像→文本）和跨语言（源语言→目标语言）两个维度的转换。

**现有痛点**：当通过监督微调（SFT）在 DIMT 数据集上训练 MLLM 时，模型会出现灾难性遗忘——翻译能力虽然提升了，但原有的 OCR 能力会显著退化。这是因为单纯的翻译任务微调改变了模型对文档图像的理解方式，导致模型"忘记"了如何准确识别源语言文本。

**核心矛盾**：DIMT 本质上涉及两步——先精确识别图像中的源语言文本（OCR），再将其翻译为目标语言。但标准 SFT 只用翻译对训练模型，没有显式保持 OCR 能力，导致两个能力此消彼长。

**本文目标**：设计一种微调范式，在提升 DIMT 翻译能力的同时保持甚至增强模型的 OCR 能力。

**切入角度**：作者从认知科学中"双语认知优势"（Bilingual Cognitive Advantage）理论获得启发——研究表明双语者在认知任务中表现更好，因为两种语言在脑中的交互增强了认知灵活性。类似地，让模型在翻译前先进行 OCR 识别，可以利用两种任务的协同效应。

**核心 idea**：在微调 MLLM 进行 DIMT 时，让模型在生成翻译文本之前先生成 OCR 识别文本（即 "自审查" 其 OCR 能力），通过这种同步自审查机制保持模型的单语能力同时学习跨语言翻译。

## 方法详解

### 整体框架

SSR 方法的整体流程非常简洁：输入是一张文档图像和翻译提示，模型需要依次输出两部分——首先输出源语言的 OCR 识别文本，然后输出目标语言的翻译文本。这两部分在同一次前向传播中顺序生成，使用特殊分隔符分开。训练时的监督信号同时包含 OCR 标签和翻译标签。

### 关键设计

1. **同步自审查（Synchronously Self-Reviewing, SSR）**:

    - 功能：在翻译生成过程中强制激活模型的 OCR 能力
    - 核心思路：将 DIMT 任务的输出格式定义为 "[OCR] {源语言文本} [Translation] {目标语言文本}"。训练时，OCR 部分使用文档的 ground truth 源文本作为监督，翻译部分使用参考译文。推理时，模型先自主识别源文本再翻译，OCR 输出可以作为翻译的上下文参考
    - 设计动机：通过强制模型在翻译前执行 OCR，模型在微调过程中始终需要维持 OCR 能力，从而避免灾难性遗忘；同时 OCR 输出为翻译提供了显式的源文本参考，提高翻译准确性

2. **双任务联合训练框架**:

    - 功能：同时优化 OCR 和翻译两个目标
    - 核心思路：损失函数包含两部分：OCR 识别损失和翻译损失，在同一序列中通过位置标记区分。两个任务共享模型参数和视觉编码器，OCR 任务的梯度信号有助于维持视觉特征提取的准确性，翻译任务的梯度信号则推动跨语言映射能力的学习
    - 设计动机：相比在 SFT 中单独加入 OCR 数据进行多任务训练，SSR 的优势在于两个任务在同一个输入上执行，确保了 OCR 和翻译的语义一致性

3. **渐进式训练策略**:

    - 功能：分阶段引入不同难度的训练数据
    - 核心思路：第一阶段用简单的、OCR 准确率高的文档训练，让模型先学会格式和基本能力；第二阶段引入更复杂的文档（手写体、复杂排版等），逐步提升难度。这种课程学习策略避免了一开始就在困难样本上训练导致的不稳定
    - 设计动机：DIMT 数据的质量参差不齐，渐进式训练可以让模型更稳定地学习

### 损失函数 / 训练策略

总损失为标准的自回归交叉熵损失，计算范围覆盖 OCR token 和翻译 token。通过 attention mask 确保 OCR 部分只关注图像输入，而翻译部分可以同时关注图像和前面的 OCR 输出。

## 实验关键数据

### 主实验

在多个文档图像翻译基准上的 BLEU 评分对比：

| 方法 | Zh→En BLEU | En→De BLEU | OCR CER↓ | OCR F1 |
|------|-----------|-----------|---------|--------|
| 基线 MLLM (SFT) | 28.3 | 21.7 | 8.2% | 86.5% |
| 基线 MLLM (无微调) | 15.6 | 12.4 | 3.1% | 94.2% |
| Pipeline (OCR+NMT) | 26.8 | 20.9 | 3.1% | 94.2% |
| SSR (本文) | 31.5 | 24.2 | 3.8% | 93.1% |

SSR 方法在翻译质量上超越所有基线（BLEU +3.2/+2.5），同时 OCR 性能仅轻微下降（CER 增加 0.7%），远好于标准 SFT（CER 从 3.1% 退化到 8.2%）。

### 消融实验

| 配置 | BLEU | OCR CER↓ | 说明 |
|------|------|---------|------|
| SSR 完整 | 31.5 | 3.8% | 完整模型 |
| 仅翻译 SFT | 28.3 | 8.2% | 严重灾难性遗忘 |
| 翻译 + OCR 多任务 | 29.7 | 5.1% | 分离式多任务一定程度缓解 |
| SSR 无 OCR 监督 | 29.1 | 6.7% | OCR 部分无监督退化明显 |
| SSR + 渐进训练 | 32.1 | 3.6% | 渐进训练进一步提升 |

### 关键发现

- **灾难性遗忘是 DIMT 的核心瓶颈**：标准 SFT 让 OCR CER 从 3.1% 退化到 8.2%，说明单纯的翻译微调会严重破坏视觉理解能力
- SSR 完整模型相比分离式多任务训练提升 1.8 BLEU，说明同步生成（OCR→翻译）比独立多任务更有效
- OCR 输出为翻译提供了类似 "中间步骤" 的效果，类似于 Chain-of-Thought 的思想在多模态翻译中的应用
- 渐进训练策略带来稳定的额外提升，尤其在复杂文档（手写体、多列排版）上效果显著

## 亮点与洞察

- **认知科学启发的简单有效方案**：从双语认知优势到 SSR 的映射非常自然且有说服力，方法实现起来也非常简单——只需修改输出格式和添加 OCR 标签，不需要额外的模块或复杂的训练策略
- **灾难性遗忘的优雅解决方案**：不是通过数据重放、正则化等间接手段，而是直接在任务设计层面让模型必须保持原有能力，思路新颖
- 可推广性强：SSR 的思想可以推广到任何多能力 MLLM 微调场景——在训练新能力时显式保持旧能力的生成

## 局限与展望

- 论文主要在中英、英德翻译对上验证，对更多语言对（尤其是形态丰富的低资源语言）的泛化性未知
- SSR 增加了生成长度（需要先生成 OCR 再生成翻译），推理时间增加，对长文档效率影响较大
- 未探讨 OCR 错误对翻译质量的级联影响——如果 OCR 识别错误，错误信息可能误导翻译
- 可以扩展到多模态问答中的类似场景，如先描述图像再回答问题

## 相关工作与启发

- **vs Pipeline 方法（OCR→NMT）**: Pipeline 方法将两步完全分离，错误级联且无法端到端优化；SSR 在端到端框架中整合了两步，允许联合优化
- **vs 标准 SFT**: 标准 SFT 只关注最终翻译输出，忽略了中间能力的保持；SSR 通过显式监督 OCR 来保持中间能力
- **vs Chain-of-Thought**: SSR 可以看作 CoT 在多模态翻译中的变体——先"思考"（OCR）再"回答"（翻译），利用中间步骤辅助最终输出

## 评分

- 新颖性: ⭐⭐⭐⭐ 认知科学启发的自审查机制新颖简洁
- 实验充分度: ⭐⭐⭐⭐ 消融和对比全面，多维度验证了方法有效性
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法描述直观易懂
- 价值: ⭐⭐⭐½ 问题聚焦于文档图像翻译这一相对小众的领域，但方法思想有更广泛的适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Single-to-mix Modality Alignment with Multimodal Large Language Model for Document Image Machine Translation](single-to-mix_modality_alignment_with_multimodal_large_language_model_for_docume.md)
- [\[AAAI 2026\] Seeing Justice Clearly: Handwritten Legal Document Translation with OCR and Vision-Language Models](../../AAAI2026/multimodal_vlm/seeing_justice_clearly_handwritten_legal_document_translation_with_ocr_and_visio.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](../../ICCV2025/multimodal_vlm/dogr_towards_versatile_visual_document_grounding_and_referring.md)
- [\[ACL 2026\] TeXOCR: Advancing Document OCR Models for Compilable Page-to-LaTeX Reconstruction](../../ACL2026/multimodal_vlm/texocr_advancing_document_ocr_models_for_compilable_page-to-latex_reconstruction.md)

</div>

<!-- RELATED:END -->
