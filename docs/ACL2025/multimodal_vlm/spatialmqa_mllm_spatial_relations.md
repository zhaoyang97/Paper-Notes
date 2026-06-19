---
title: >-
  [论文解读] Can Multimodal Large Language Models Understand Spatial Relations?
description: >-
  [多模态VLM] 提出 SpatialMQA 基准，以多选题形式评估 MLLM 的空间关系推理能力，发现 SOTA 模型仅达 48.14% 准确率，远低于人类 98.40%。 - 核心问题： 现有空间关系推理基准存在依赖边界框、忽视视角替换、仅凭先验知识可答题等问题，无法真正评估 MLLM 对图像中空间关系的理解能力…
tags:
  - "多模态VLM"
---

# Can Multimodal Large Language Models Understand Spatial Relations?

- **会议**: ACL 2025
- **arXiv**: [2505.19015](https://arxiv.org/abs/2505.19015)
- **代码**: [GitHub](https://github.com/ziyan-xiaoyu/SpatialMQA)
- **领域**: 多模态VLM
- **关键词**: 空间关系, 多模态基准, MLLM评估, SpatialMQA, 视角替换

## 一句话总结

提出 SpatialMQA 基准，以多选题形式评估 MLLM 的空间关系推理能力，发现 SOTA 模型仅达 48.14% 准确率，远低于人类 98.40%。

## 研究背景与动机

- **核心问题**: 现有空间关系推理基准存在依赖边界框、忽视视角替换、仅凭先验知识可答题等问题，无法真正评估 MLLM 对图像中空间关系的理解能力。
- **现有方法局限**:
    - **依赖边界框**: SpatialVOC2K、Rel3D、SpatialSense+ 等需要 bbox 标注主客体，但某些实体（如"太阳"）无法用 bbox 框定。
    - **标注不基于客观世界**: SpatialSense 将"天空在森林后方"标注为 behind，与人类认知不一致。
    - **忽视视角替换**: 如 VSR 中仅 6% 样本使用第一人称视角，缺乏对复杂场景（如自动驾驶）的评估能力。
    - **先验知识可答题**: 如"书在公交车上方"可仅凭常识回答 No，无需理解图像。
- **本文动机**: 构建一个高质量的人工标注基准，使模型必须理解图像才能回答，同时覆盖多种视角替换场景。

## 方法详解

### 整体框架

SpatialMQA 是一个基于 COCO2017 的多选题空间关系推理基准，包含 5,392 个样本、128 种主客体类型、6 种空间关系（left of / right of / in front of / behind / on/above / below）。任务形式为：给定图像 I 和问题 Q，从 k(k=2,...,6) 个选项中选择正确的空间关系。

### 关键设计

1. **以客观世界为参考系的空间坐标系**: 以重力方向为下、观察者为原点，X 轴从左到右，Y 轴从后到前，Z 轴从下到上，确保标注符合人类直觉认知。
2. **视角替换机制**: 题目分为两类——图像外视角（第三方观察者）和图像内视角（第一人称/第三人称），其中图像内视角占 60%，要求模型理解不同观察者视角下的空间关系。
3. **三轮标注质量控制**: 一轮标注 → 二轮检查（是否可仅凭先验知识回答 + 主客体是否清晰）→ 三轮审查（主作者抽检 20%），每轮设置通过率阈值（90%/95%）。

### 损失函数

开源模型微调时使用标准的交叉熵损失，LoRA 用于参数高效微调。

## 实验

### 主实验结果

| 模型 | 设置 | Acc (%) |
|------|------|---------|
| **SpaceLLaVA** | LoRA | **48.14** |
| LLaVA1.5-7B | LoRA | 46.85 |
| InstructBLIP-3B | LoRA | 42.38 |
| GPT-4o | 0-shot | 40.20 |
| Gemini-1.5-flash | 3-shot | 38.00 |
| BLIP-vqa-base | Full | 33.64 |
| Random Choose | - | 27.20 |
| **Human** | - | **98.40** |

### 消融实验（按问题类型和答案维度分析）

| 模型 | Q1(图外) | Q2(第一人称) | Q3(第三人称) | Ax(左右) | Ay(前后) | Az(上下) |
|------|----------|-------------|-------------|----------|----------|----------|
| SpaceLLaVA (LoRA) | 54.87 | 42.37 | 58.82 | 56.00 | 51.85 | 31.41 |
| GPT-4o (0-shot) | 44.09 | 33.74 | 61.76 | 37.08 | 47.50 | 36.00 |
| LLaVA1.5-7B (LoRA) | 53.14 | 40.99 | 64.71 | 55.71 | 29.64 | 48.13 |

### 关键发现

1. **MLLM 与人类差距巨大**: SOTA 模型（SpaceLLaVA LoRA）48.14% vs 人类 98.40%，差距超过 50 个百分点。
2. **视角替换是主要难点**: 图像内第一人称视角（Q2）准确率普遍最低（如 SpaceLLaVA 仅 42.37%），说明模型难以进行视角切换推理。
3. **LoRA 微调显著提升表现**: 经指令微调后，SpaceLLaVA 从 31.32% 提升至 48.14%，LLaVA 从 29.28% 提升至 46.85%。
4. **纯文本无法回答**: 仅给文本不给图像时准确率仅 24.40%，低于随机选择，验证了基准的"必须看图"特性。
5. **GPT-4o 的 few-shot 未必优于 zero-shot**: GPT-4o 0-shot (40.20%) 优于 3-shot (37.80%)，可能因为 ICL 样例引入了干扰。

## 亮点

- 首个系统化排除先验知识干扰、涵盖第一/第三人称视角替换的空间关系推理基准
- 三轮人工标注质量控制流程设计严谨，确保每个样本必须看图才能作答
- 揭示了当前 MLLM 在空间关系理解上的巨大不足（50+个百分点差距），为后续研究提供了明确方向
- 提供了按视角类型和空间维度的细粒度分析，精确定位了模型的薄弱环节
- 基准和代码完全开源，标注指南详尽可复现

## 局限性

- 空间关系仅覆盖 6 种基本类型，未包含更复杂的空间描述（如"旁边""之间""环绕"）
- 基于 COCO2017 的图像，场景多样性受限于该数据集的覆盖范围
- 仅评估了有限数量的闭源模型（GPT-4o 和 Gemini），未覆盖更多最新模型（如 Claude、Qwen-VL）
- 第三人称视角样本较少（仅 185 个），可能影响该类别评估的统计可靠性
- 选项数量（2/4/6）的分布不均匀（75% 为 4 选项），对不同选项数量的难度差异分析不足
- 未探讨空间关系的模糊性（如"on"和"above"的边界情况）如何影响标注一致性

## 相关工作

- **空间关系基准**: SpatialVOC2K (Belz et al. 2018)、SpatialSense (Yang et al. 2019)、Rel3D (Goyal et al. 2020)、VSR (Liu et al. 2023a)、EmbSpatial (Du et al. 2024)、SpatialRGPT (Cheng et al. 2024)
- **MLLM**: GPT-4o (Achiam et al. 2023)、Gemini-1.5-flash、LLaVA (Liu et al. 2024)、SpaceLLaVA (Chen et al. 2024)、BLIP/BLIP2/InstructBLIP 系列
- **空间推理增强**: SpaceLLaVA 通过空间关系指令微调提升理解能力；SpatialVLM (Chen et al. 2024) 通过开放式 QA 评估空间感知
- **基准设计方法论**: COCO2017 (Lin et al. 2014) 提供多实体场景图像；三轮标注流程参考了严格的质量控制范式

## 评分

- **创新性**: ⭐⭐⭐⭐ — 首次系统化地解决了空间关系基准中的视角替换和先验知识问题
- **实用性**: ⭐⭐⭐⭐ — 为社区提供了高质量评估工具，暴露了当前模型的明确短板
- **严谨性**: ⭐⭐⭐⭐⭐ — 三轮标注质量控制流程非常扎实
- **综合**: ⭐⭐⭐⭐

<!-- SpatialMQA: human-annotated, perspective-aware spatial relation reasoning benchmark for MLLMs -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](can_vision_language_models_understand_mimed_actions.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] Teaching Vision-Language Models to Ask: Resolving Ambiguity in Visual Questions](teaching_vlm_ask_ambiguity.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)

</div>

<!-- RELATED:END -->
