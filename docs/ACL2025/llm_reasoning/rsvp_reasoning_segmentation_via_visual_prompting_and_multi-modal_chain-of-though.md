---
title: >-
  [论文解读] RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought
description: >-
  [ACL 2025 (Main)][LLM推理][推理分割] 本文提出 RSVP 框架，通过两阶段结构（推理驱动定位 + 分割精炼）将多模态大模型的推理能力与视觉分割相统一，利用多模态思维链视觉提示在 ReasonSeg 上超越 SOTA 达 +6.5 gIoU / +9.2 cIoU，零样本 SegInW 达到 49.7 mAP。
tags:
  - "ACL 2025 (Main)"
  - "LLM推理"
  - "推理分割"
  - "视觉提示"
  - "多模态思维链"
  - "视觉定位"
  - "语言-视觉分割"
---

# RSVP: Reasoning Segmentation via Visual Prompting and Multi-modal Chain-of-Thought

**会议**: ACL 2025 (Main)  
**arXiv**: [2506.04277](https://arxiv.org/abs/2506.04277)  
**代码**: 无  
**领域**: LLM推理  
**关键词**: 推理分割, 视觉提示, 多模态思维链, 视觉定位, 语言-视觉分割

## 一句话总结

本文提出 RSVP 框架，通过两阶段结构（推理驱动定位 + 分割精炼）将多模态大模型的推理能力与视觉分割相统一，利用多模态思维链视觉提示在 ReasonSeg 上超越 SOTA 达 +6.5 gIoU / +9.2 cIoU，零样本 SegInW 达到 49.7 mAP。

## 研究背景与动机

**领域现状**：多模态大语言模型（MLLM）在视觉推理任务上展现了优秀的认知推理能力，能够理解复杂的视觉场景并回答关于图像内容的问题。然而，这些模型缺乏将推理结论直接落地为精确视觉输出（如分割掩码）的机制，存在认知推理与视觉感知之间的鸿沟。

**现有痛点**：现有的推理分割方法大多采用端到端的方式，将推理和分割压缩在单一模型中。这种方式面临两个问题：(1) 推理过程不透明——模型直接从查询跳到分割结果，中间的推理链条不可见，难以调试和理解；(2) 分割精度受限——MLLM 虽然擅长语义推理，但其输出通常是文本或粗粒度的区域指示，无法直接产生像素级精确的分割掩码。

**核心矛盾**：高层语义推理（理解"哪个物体最可能导致交通事故"这类复杂查询）需要 MLLM 的强大推理能力，但精确的像素级分割需要专门的视觉模型。将两者硬塞进同一个模型会导致两方面都做不到最好。

**本文目标**：设计一个可解释的推理分割框架，让 MLLM 专注于推理和定位，让专业的分割模块专注于产生精确掩码，两者通过结构化的视觉表示连接。

**切入角度**：MLLM 虽然无法直接输出分割掩码，但它们具备固有的目标定位能力——可以通过文本描述或边界框来指示目标区域。利用这种能力，让 MLLM 生成结构化的视觉提示（如区域建议），作为下游分割模块的输入。

**核心 idea**：两阶段解耦框架——第一阶段用 MLLM 进行多步推理并生成可解释的区域建议（reasoning stage），第二阶段用视觉-语言分割模块将区域建议精炼为像素级掩码（segmentation stage）。

## 方法详解

### 整体框架

RSVP 是一个两阶段 pipeline：输入为一张图像和一个需要推理的自然语言查询（如"找到可能导致交通事故的物体"），输出是对应目标的精确分割掩码。第一阶段（推理阶段）使用 MLLM 解析查询、进行多步推理、识别目标并生成区域建议；第二阶段（分割阶段）使用 Vision-Language Segmentation Module (VLSM) 融合文本线索和视觉线索，将区域建议精炼为精确的分割掩码。两个阶段之间通过结构化的视觉提示（如边界框或区域标记）连接。

### 关键设计

1. **多模态思维链视觉提示 (Multimodal Chain-of-Thought Visual Prompting)**:

    - 功能：引导 MLLM 进行结构化的多步推理，从复杂查询逐步推导到目标定位
    - 核心思路：设计了一套 prompt 工程策略，让 MLLM 像"思维链"一样逐步推理：(a) 先理解查询的语义意图；(b) 分析图像中的相关物体及其关系；(c) 基于推理锁定目标物体；(d) 生成目标的区域建议（边界框坐标或视觉标记）。整个推理过程以自然语言输出，具有高度可解释性。视觉提示的引入使得推理不仅停留在语言层面，还能结合视觉信息进行定位
    - 设计动机：传统方法让模型直接从问题跳到答案，丢失了推理过程。思维链范式在文本推理中已证明有效，将其扩展到多模态场景可以同时提升推理准确性和可解释性

2. **视觉-语言分割模块 (Vision-Language Segmentation Module, VLSM)**:

    - 功能：将第一阶段的粗粒度区域建议和文本描述精炼为像素级精确的分割掩码
    - 核心思路：VLSM 接收三路输入——原始图像的视觉特征、第一阶段输出的区域建议（作为视觉先验）、以及推理过程中产生的文本描述。模块通过跨模态注意力机制融合文本和视觉线索，利用区域建议作为空间注意力先验来聚焦分割区域，最终输出目标的分割掩码。这种设计使得分割模块能够利用语义信息来消除视觉歧义
    - 设计动机：单纯依赖区域建议（如边界框）进行分割会丢失语义信息，而纯文本驱动的分割在处理精细边界时不够准确。VLSM 通过融合两种模态的信息来取长补短

3. **推理-分割解耦架构 (Reasoning-Segmentation Decoupling)**:

    - 功能：将认知推理和视觉感知分配给各自最擅长的模块
    - 核心思路：MLLM（如 GPT-4V、LLaVA）负责高层推理——理解查询意图、分析场景、推断目标身份；VLSM 负责底层感知——生成精确的像素级分割。两者通过结构化的中间表示（区域建议 + 文本描述）解耦连接。这种架构允许独立替换或升级任一模块
    - 设计动机：端到端方法将推理和分割混在一起，导致推理能力和分割精度相互制约。解耦后，每个模块可以在各自的任务上做到最优，且整体系统更模块化、更易于维护和升级

### 损失函数 / 训练策略

VLSM 的训练使用标准的分割损失（交叉熵 + Dice loss 组合）。Reasoning 阶段利用 MLLM 的 in-context learning 能力，不需要额外训练。整体框架侧重推理与组合的方式，而非端到端微调。

## 实验关键数据

### 主实验

| 数据集/指标 | RSVP | 之前SOTA | 提升 |
|------------|------|---------|------|
| ReasonSeg gIoU | SOTA | 之前最佳 | **+6.5** |
| ReasonSeg cIoU | SOTA | 之前最佳 | **+9.2** |
| SegInW mAP (零样本) | **49.7** | - | 零样本新SOTA |

在 ReasonSeg 基准上，RSVP 在 gIoU 和 cIoU 两个指标上都大幅超越所有已有方法。在 Segmentation in the Wild (SegInW) 的零样本设置下，RSVP 无需任何目标域训练数据即达到 49.7 mAP。

### 消融实验

| 配置 | gIoU | cIoU | 说明 |
|------|------|------|------|
| Full RSVP | 最优 | 最优 | 完整两阶段框架 |
| 无 Chain-of-Thought | 下降 | 下降 | 移除多步推理，直接定位 |
| 无 Visual Prompting | 下降 | 下降 | 仅用文本推理，不结合视觉 |
| 无 VLSM（用标准分割器） | 显著下降 | 显著下降 | 没有文本-视觉融合的精炼 |
| 单阶段端到端 | 显著下降 | 显著下降 | 验证解耦架构的必要性 |

### 关键发现

- 两阶段解耦架构相比端到端方法提升显著，验证了"推理-感知分离"策略的有效性
- Visual prompting 对定位质量至关重要——没有视觉线索的辅助，纯文本推理容易产生定位偏差
- Chain-of-Thought 推理对复杂查询效果尤其显著，对简单查询的边际收益较小
- 在零样本设置下的出色表现说明 RSVP 的推理能力具有强泛化性，不依赖目标域的分割标注

## 亮点与洞察

- **可解释性与精度的双赢**：RSVP 的推理阶段输出完整的自然语言推理链条，用户可以看到模型"为什么"选择某个目标，这在医疗影像、自动驾驶等高风险场景下特别有价值。同时，解耦设计并没有牺牲分割精度
- **利用 MLLM 的固有定位能力**：大多数工作聚焦于训练 MLLM 产生分割输出，RSVP 反其道而行之——不修改 MLLM，而是利用其已有的理解和定位能力，通过 prompt 工程来引导推理。这种方式零训练成本且能跟随 MLLM 的能力提升而自动升级
- **模块化设计的可扩展性**：reasoning 模块可以替换为更强的 MLLM，segmentation 模块可以替换为更精确的分割器，两者独立进化。这种思路可以迁移到其他"推理→执行"的多步任务中

## 局限与展望

- 两阶段方法引入了额外的推理延迟，不适合需要实时响应的场景
- 高度依赖 MLLM 的推理质量——如果第一阶段推理错误地定位了目标，下游分割无论多精确都无法挽回
- 对复杂的遮挡场景和细粒度区分（如"第三朵从左边数的花"）的处理能力有待验证
- 目前主要在英文查询上验证，多语言推理分割场景的性能未知
- 改进方向：可引入推理自验证机制，让模型回看分割结果并修正推理链中的错误

## 相关工作与启发

- **vs LISA (Lai et al., 2024)**: LISA 将分割 token 嵌入 MLLM 进行端到端训练，推理过程不可见。RSVP 通过解耦设计保证可解释性，且零样本能力更强
- **vs SEEM/GroundedSAM**: 这些方法主要依赖显式的视觉引用（如指定类名或框），缺乏复杂语义推理能力。RSVP 能处理需要多步推理的隐含查询
- **vs Set-of-Mark (SoM) Prompting**: SoM 在图像上标记区域作为视觉提示，RSVP 进一步将这种思路与 chain-of-thought 推理结合，形成结构化的推理-定位-分割流水线

## 评分

- 新颖性: ⭐⭐⭐⭐ 将多模态 CoT 与视觉分割解耦组合的新范式，思路清晰且符合直觉
- 实验充分度: ⭐⭐⭐⭐ 在 ReasonSeg 和 SegInW 上均有显著提升，零样本能力令人印象深刻
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，两阶段设计的动机论述充分
- 价值: ⭐⭐⭐⭐ 为可解释的视觉推理分割提供了实用的范式参考

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](../../AAAI2026/llm_reasoning/cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)
- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](../../CVPR2025/llm_reasoning/interleaved-modal_chain-of-thought.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](../../CVPR2026/llm_reasoning/rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)
- [\[ICLR 2026\] Fine-R1: Make Multi-modal LLMs Excel in Fine-Grained Visual Recognition by Chain-of-Thought Reasoning](../../ICLR2026/llm_reasoning/fine-r1_make_multi-modal_llms_excel_in_fine-grained_visual_recognition_by_chain-.md)

</div>

<!-- RELATED:END -->
