---
title: >-
  [论文解读] Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation
description: >-
  [NeurIPS 2025][医学图像][视觉语言模型] 构建首个越南语 PET/CT 图像-报告数据集 ViMed-PET（2,757 例全身 PET/CT 体积 + 完整临床报告），通过数据增强策略和三阶段微调流程显著提升 VLM 在医学报告生成和 VQA 任务上的表现，并提出基于临床关键信息的评估指标。
tags:
  - "NeurIPS 2025"
  - "医学图像"
  - "视觉语言模型"
  - "PET/CT"
  - "越南语医学报告"
  - "数据集"
  - "报告生成"
---

# Toward a Vision-Language Foundation Model for Medical Data: Multimodal Dataset and Benchmarks for Vietnamese PET/CT Report Generation

**会议**: NeurIPS 2025  
**arXiv**: [2509.24739](https://arxiv.org/abs/2509.24739)  
**代码**: 暂无  
**领域**: 医学图像  
**关键词**: 视觉语言模型, PET/CT, 越南语医学报告, 数据集, 报告生成

## 一句话总结

构建首个越南语 PET/CT 图像-报告数据集 ViMed-PET（2,757 例全身 PET/CT 体积 + 完整临床报告），通过数据增强策略和三阶段微调流程显著提升 VLM 在医学报告生成和 VQA 任务上的表现，并提出基于临床关键信息的评估指标。

## 研究背景与动机

视觉语言基础模型 (VLM) 在通用领域已取得巨大成功，但在医学影像中的应用面临两大系统性障碍：

**影像模态覆盖窄**：现有医学 VLM 主要聚焦于 X-ray、CT 和 MRI 等成像方式，而**PET/CT 扫描在肿瘤学、心脏病学和神经学中至关重要**，却在现有数据集和模型中严重缺失。PET/CT 能非侵入性地评估肿瘤代谢和扩散，是早期诊断和分期的不可替代工具。

**语言覆盖单一**：现有医学 VLM 和数据集几乎完全以英语为主，拥有超过 1 亿母语人口的越南语在医学 AI 中几乎没有资源，加剧了医疗数据不公平。

作者的初步实验清楚揭示了问题的严重性：LLaVA-Med、M3D、RadFM 在越南语 PET/CT 报告生成任务上的 BLEU-4 分数接近零（0.01-0.06），即使 GPT-4o 在 few-shot 设置下也仅达到 31.12%。这不仅反映了模型能力不足，更说明了**训练数据的根本缺失**。

此外，现有公开 PET/CT 数据集（RIDER Lung、FDG-PET-CT-Lesions 等）虽然提供了影像数据，但**均不包含配对的临床报告**，限制了生成式建模和多模态推理的发展。

## 方法详解

### 整体框架

ViMed-PET 的贡献分为三个层面：数据集构建 → 数据增强策略 → 模型微调流程。

### 关键设计

1. **ViMed-PET 数据集**：来源于越南某国家三级综合医院，包含 2,757 例独立患者的 PET/CT 体积（约 1,567,062 对 CT-PET 切片）和对应的完整越南语临床报告。数据涵盖 2017-2023 年，包含肺癌、甲状腺癌等多种疾病。采用 DICOM 格式存储，包含像素数据和采集参数元数据。所有数据经过 IRB 伦理审批和去标识化处理。

   **解剖区域分割**：将每个全身扫描分为头颈、胸部、腹盆三个区域，根据患者身高自适应确定分割比例，引入 20 切片重叠避免边界信息丢失。这将样本量从 2,757 扩展到 8,271 对图像-报告。

2. **数据增强框架**：构建四个任务专用子集：

    - **VQA 数据集**（8,271 对话）：单轮描述性问答 + GPT-4o 生成的多轮对话
    - **报告生成数据集**（5,571 报告）：用 GPT-4o 对原始报告进行临床准确的改写
    - **研究对比数据集**（10,000 对）：将两个 PET 图像拼接，配合描述差异性的文本
    - **医学测试集**（398 病灶）：专注肺癌诊断，提取病灶类型、位置、FDG 代谢等结构化关键信息

3. **三阶段微调流程**：

    - **Stage 1**：3D 视觉编码器微调。CT-ViT 用 CLIP 式文本-图像对比学习，Cosmos Tokenizer 用重建损失
    - **Stage 2**：概念特征对齐。冻结视觉编码器和语言模型，仅训练线性投影层，将视觉特征映射到文本嵌入空间
    - **Stage 3**：指令微调。使用 LoRA 高效微调语言模型和投影层，在 VQA 数据集上进行指令调优

   视觉编码器选择：CT-ViT（3D 医学 CT 预训练）和 Cosmos Tokenizer（通用视频预训练，去除因果注意力）
   语言模型选择：Mistral-7B 和 LLaMA-2-7B

### 临床评估指标

传统 NLP 指标（BLEU、ROUGE）无法反映临床准确性。本文提出四个临床 F1 指标：
- **F1-T**：仅评估病灶类型
- **F1-TP**：病灶类型 + 位置
- **F1-TF**：病灶类型 + FDG 代谢
- **F1-TPF**：类型 + 位置 + FDG 摄取（最严格）

## 实验关键数据

### 主实验：PET/CT 报告生成

| 模型 | 设置 | BLEU-4↑ | ROUGE-1↑ | BERT↑ | F1-T↑ | F1-TP↑ | F1-TPF↑ |
|------|------|---------|---------|-------|-------|--------|---------|
| LLaVA-Med | baseline | 0.01 | 50.08 | 64.63 | - | - | - |
| M3D | baseline | 0.04 | 41.01 | 67.21 | - | - | - |
| RadFM | baseline | 0.06 | 54.23 | 69.49 | - | - | - |
| GPT-4o* | few-shot | 31.12 | 67.96 | 81.09 | 24.21 | 13.62 | 7.87 |
| CT-ViT+Mistral (O-G) | 微调 | **58.07** | **80.11** | **89.98** | **51.11** | **30.66** | **22.65** |
| Cosmos+Mistral (O-G-C) | 微调 | 58.87 | 80.10 | 90.55 | 47.93 | 22.78 | 15.68 |

### VQA 任务（O-G-C 设置）

| 模型 | BLEU-4↑ | ROUGE-1↑ | ROUGE-L↑ | BERT↑ |
|------|---------|---------|---------|-------|
| GPT-4o* | 3.01 | 49.35 | 30.09 | 71.92 |
| CT-ViT+Mistral | **31.14** | **65.61** | **51.22** | **82.50** |
| Cosmos+LLaMA-2 | 28.40 | 63.29 | 48.76 | 79.35 |

### 消融：数据增强效果

| 数据设置 | BLEU-4 | F1-T | F1-TPF | 说明 |
|---------|--------|------|--------|------|
| O (仅原始) | 53.30 | 43.49 | 18.26 | 基线 |
| O+G (原始+改写) | 58.07 | 51.11 | 22.65 | +改写报告 |
| O+G+C (全部) | 58.05 | 51.96 | 21.23 | +对比数据 |

### 关键发现

- 微调后 BLEU-4 从接近 0 提升至 58+（vs GPT-4o 的 31.12），提升幅度巨大
- 临床 F1 指标最高约 51%（F1-T），F1-TPF 约 22%，说明模型在精确诊断上仍有很大提升空间
- CT-ViT 在临床指标上一致优于 Cosmos Tokenizer，说明 3D 医学预训练的重要性
- Mistral-7B 在扩展数据后表现优于 LLaMA-2-7B，归因于更高效的架构
- 报告改写增强（G）贡献最大，研究对比增强（C）的边际效益较小

## 亮点与洞察

- 填补了 PET/CT + 低资源语言的双重空白，是该领域首个此类数据集
- 临床评估指标的设计具有实际参考价值——传统 NLP 指标确实无法衡量医学关键信息的准确性
- 数据增强策略（解剖分割、报告改写、研究对比）可推广到其他影像-报告数据集
- 结果诚实地指出了当前模型的严重局限：F1-TPF 仅约 22%，距离临床可用仍有很大差距

## 局限与展望

- 临床测试集仅聚焦肺癌，覆盖面有限
- 虽然数据集包含 CT 体积，但 benchmark 仅使用 PET 信息（报告主要基于 PET）
- 数据增强大量依赖 GPT-4o，引入了 AI 生成内容的偏差
- 仅 2,757 例的数据量相对有限，收集更多数据应能进一步提升性能
- 未开源数据集（仅描述了数据集，涉及医院数据伦理限制）

## 相关工作与启发

- 与现有 PET/CT 数据集（RIDER、FDG-PET-CT-Lesions 等）的关键区别：本数据集首次提供配对临床报告
- 三阶段微调流程借鉴了 LLaVA 系列的对齐方法，但适配了 3D 医学场景
- 临床 F1 指标的思路可推广到其他需要评估"医学关键信息"准确性的生成任务

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据集贡献为主，填补PET/CT+越南语空白
- 实验充分度: ⭐⭐⭐⭐ 多种encoder/LLM组合+增强策略+临床指标，但缺少更多baseline
- 写作质量: ⭐⭐⭐⭐ 数据集描述详尽，pipeline清晰
- 价值: ⭐⭐⭐⭐ 数据集资源价值高，推动低资源语言医学AI发展

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Brain Harmony: A Multimodal Foundation Model Unifying Morphology and Function into 1D Tokens](brain_harmony_a_multimodal_foundation_model_unifying_morphology_and_function_int.md)
- [\[NeurIPS 2025\] NeurIPT: Foundation Model for Neural Interfaces](neuript_foundation_model_for_neural_interfaces.md)
- [\[NeurIPS 2025\] Magical: Medical Lay Language Generation via Semantic Invariance and Layperson-tailored Adaptation](magical_medical_lay_language_generation_via_semantic_invariance_and_layperson-ta.md)
- [\[NeurIPS 2025\] Few-Shot Learning from Gigapixel Images via Hierarchical Vision-Language Alignment and Modeling](few-shot_learning_from_gigapixel_images_via_hierarchical_vision-language_alignme.md)
- [\[NeurIPS 2025\] Mind the (Data) Gap: Evaluating Vision Systems in Small Data Applications](mind_the_data_gap_evaluating_vision_systems_in_small_data_applications.md)

</div>

<!-- RELATED:END -->
