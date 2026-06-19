---
title: >-
  [论文解读] Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models
description: >-
  [CVPR 2025][目标检测][零样本异常检测] 首个专用于零样本异常检测和推理的 MLLM（Anomaly-OV），通过 Look-Twice Feature Matching 机制生成异常显著性图，配合视觉 Token 选择器聚焦可疑区域，在 9 个基准上实现 88.6% 平均 AUROC 的零样本异常检测 SOTA。
tags:
  - "CVPR 2025"
  - "目标检测"
  - "零样本异常检测"
  - "MLLM专家系统"
  - "视觉特征匹配"
  - "异常推理"
  - "工业检测"
---

# Towards Zero-Shot Anomaly Detection and Reasoning with Multimodal Large Language Models

**会议**: CVPR 2025  
**arXiv**: [2502.07601](https://arxiv.org/abs/2502.07601)  
**代码**: [https://xujiacong.github.io/Anomaly-OV/](https://xujiacong.github.io/Anomaly-OV/)  
**领域**: 多模态VLM  
**关键词**: 零样本异常检测、MLLM专家系统、视觉特征匹配、异常推理、工业检测

## 一句话总结
首个专用于零样本异常检测和推理的 MLLM（Anomaly-OV），通过 Look-Twice Feature Matching 机制生成异常显著性图，配合视觉 Token 选择器聚焦可疑区域，在 9 个基准上实现 88.6% 平均 AUROC 的零样本异常检测 SOTA。

## 研究背景与动机

**领域现状**：零样本异常检测（ZSAD）旨在不使用目标类别正常样本的情况下检测异常。现有方法如 WinCLIP、AnomalyCLIP 使用 CLIP 文本编码器构建正常/异常描述做匹配，但依赖文本编码器的语义能力有限。

**现有痛点**：(1) 通用 MLLM（如 GPT-4o）虽能检测异常但无法准确描述和定位——检测准确率 70% 但推理描述不精确。(2) 现有 ZSAD 方法只能给出二分类结果，无法解释"为什么异常"。(3) 工业、医疗、3D 等不同领域的异常模式差异大，单一模型难以覆盖。

**核心矛盾**：MLLM 有推理能力但缺乏异常检测的专业视觉感知；专用异常检测模型有视觉精度但无法推理解释。

**本文目标** 将异常检测的专业视觉能力注入 MLLM，使其同时具备高精度检测和自然语言推理能力。

**切入角度**：设计一个"异常专家"模块——用多层 ViT 特征 + 可学习的正常/异常 embedding 做 Look-Twice Feature Matching，生成异常显著性图。显著性图指导视觉 token 选择，让 MLLM 聚焦于可疑区域。

**核心 idea**：用多层视觉特征匹配生成异常显著性图作为 MLLM 的"放大镜"，使其既能精准定位异常又能用自然语言解释。

## 方法详解

### 整体框架
两阶段训练：Stage 1 训练异常专家（多层 ViT 特征 + 可学习 $e^+$/$e^-$ embedding → LTFM 生成显著性图）→ Stage 2 冻结专家和视觉编码器，训练投影层 + LLM，使用 Anomaly-Instruct-125K 指令数据。

### 关键设计

1. **Look-Twice Feature Matching（LTFM）**:

    - 功能：生成逐像素的异常显著性图
    - 核心思路：第一次"看"：将多层 ViT 特征与可学习正常 embedding $e^+$ 和异常 embedding $e^-$ 做协方差匹配。第二次"看"（look-back path）：用第一次的匹配结果调制原始特征再匹配，类似"回头仔细看"。两次匹配结果融合为最终显著性图
    - 设计动机：单次匹配对细微异常不够灵敏，look-back 机制提供了自校正能力。消融显示去掉 look-back AUROC 降 1.2 个点

2. **视觉 Token 选择器**:

    - 功能：让 MLLM 聚焦于显著性高的可疑区域
    - 核心思路：将视觉 token 与显著性图相乘 → 空间池化 → Q-Former 聚合成精选 token。同时用指示 prompt（$\langle adv \rangle$ suspicious feature: 其中 adv $\in$ {highly, moderately, slightly}）桥接原始 token 和精选 token
    - 设计动机：MLLM 不需要看所有视觉 token——聚焦于可疑区域使推理更精确

3. **Anomaly-Instruct-125K 数据集**:

    - 功能：覆盖多领域的异常检测指令微调数据
    - 核心思路：125K 样本涵盖工业（MVTec、VisA）、医疗（BrainMRI、HeadCT）、3D（MVTec-3D）、野外（WebAD 72K 张网络图片）。包含检测、定位、描述、推理四种任务类型
    - 设计动机：WebAD 贡献了 MVTec 上 +5.5% AUROC 的提升，说明野外异常数据对学习通用异常语义至关重要

### 损失函数 / 训练策略
Stage 1：异常专家用二分类 + 显著性图损失训练。Stage 2：冻结专家 + ViT，标准 next-token prediction 训练 LLM + 投影层。

## 实验关键数据

### 主实验

| 方法 | MVTec | VisA | AITEX | BrainMRI | HeadCT | 9 基准平均 |
|------|-------|------|-------|----------|--------|----------|
| WinCLIP | 91.8 | 78.8 | 73.0 | 92.6 | 90.0 | 79.2 |
| AnomalyCLIP | 91.5 | 82.1 | 62.2 | 90.3 | 93.4 | 84.5 |
| **Anomaly-OV** | **94.0** | **91.1** | **72.0** | **93.9** | **97.6** | **88.6** |

### 消融实验

| 配置 | MVTec | VisA | HeadCT |
|------|-------|------|--------|
| 完整模型 | 94.0 | 91.1 | 97.6 |
| 无 look-back | 92.8 | 90.5 | 96.6 |
| 无 $e^+$/$e^-$ | 92.1 | 90.1 | 94.7 |
| 无 WebAD | 88.5 | 88.9 | 91.2 |

### 关键发现
- **文本编码器不是必须的**：Anomaly-OV 不使用文本编码器做匹配（纯视觉），仍超越所有 CLIP 基方法
- **WebAD 是关键**：72K 野外异常图片贡献了 MVTec 上 +5.5% AUROC，通用异常语义的预训练至关重要
- **检测+推理一体化**：GPT-4o 在 VisA-D&R 上检测 Acc 70% 但 F1 仅 68%，Anomaly-OV 达 79% Acc 和 83% F1

## 亮点与洞察
- **"异常专家 + MLLM"的架构**巧妙——专家提供专业视觉感知，MLLM 提供推理和语言输出，各司其职
- **显著性图作为"放大镜"**的思路可推广到其他需要聚焦特定区域的 MLLM 应用（如医学影像分析）
- **零样本跨域能力**：在工业和医疗上都达到 SOTA，说明异常的视觉模式有跨领域的共性

## 局限与展望
- 异常专家的 LTFM 需要额外的训练阶段和计算开销
- 像素级异常定位（segmentation）的精度未详细报告
- 125K 训练数据中工业数据占比大，对自然场景异常可能不够

## 相关工作与启发
- **vs WinCLIP / AnomalyCLIP**：这些方法用 CLIP 文本-视觉匹配。Anomaly-OV 用纯视觉特征匹配+MLLM，精度更高且能推理
- **vs GPT-4o**：GPT-4o 能检测但不精确。Anomaly-OV 在检测和推理上都显著领先

## 评分
- 新颖性: ⭐⭐⭐⭐ 异常专家+MLLM的结合、LTFM机制新颖
- 实验充分度: ⭐⭐⭐⭐⭐ 9个异常检测基准+VisA-D&R推理基准+详尽消融
- 写作质量: ⭐⭐⭐⭐ 方法逻辑清晰，数据集贡献有价值
- 价值: ⭐⭐⭐⭐⭐ 对工业/医疗异常检测有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP](aa-clip_enhancing_zero-shot_anomaly_detection_via_anomaly-aware_clip.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](../../CVPR2026/object_detection/visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](../../CVPR2026/object_detection/back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)
- [\[ICCV 2025\] LMM-Det: Make Large Multimodal Models Excel in Object Detection](../../ICCV2025/object_detection/lmm-det_make_large_multimodal_models_excel_in_object_detection.md)
- [\[CVPR 2025\] Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection](large_self-supervised_models_bridge_the_gap_in_domain_adaptive_object_detection.md)

</div>

<!-- RELATED:END -->
