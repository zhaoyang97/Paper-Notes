---
title: >-
  [论文解读] PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography
description: >-
  [AAAI 2026][多模态VLM][PET影像] 本文提出 PET2Rep，首个专用于正电子发射断层扫描（PET）放射报告生成的大规模基准数据集（565例全身 PET/CT 图像-报告对），并设计了 PET 临床效能（CE）评估指标，对 30 个前沿通用和医疗专用 VLM 进行系统评估，发现当前 SOTA VLM 在 PET 报告生成任务上表现不佳，甚至无法超越简单的模板基线。
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "PET影像"
  - "放射报告生成"
  - "视觉语言模型"
  - "基准评估"
  - "临床效能指标"
---

# PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography

**会议**: AAAI 2026  
**arXiv**: [2508.04062](https://arxiv.org/abs/2508.04062)  
**代码**: [https://github.com/YichiZhang98/PET2Rep](https://github.com/YichiZhang98/PET2Rep)  
**领域**: 多模态VLM  
**关键词**: PET影像, 放射报告生成, 视觉语言模型, 基准评估, 临床效能指标

## 一句话总结

本文提出 PET2Rep，首个专用于正电子发射断层扫描（PET）放射报告生成的大规模基准数据集（565例全身 PET/CT 图像-报告对），并设计了 PET 临床效能（CE）评估指标，对 30 个前沿通用和医疗专用 VLM 进行系统评估，发现当前 SOTA VLM 在 PET 报告生成任务上表现不佳，甚至无法超越简单的模板基线。

## 研究背景与动机

PET 是现代肿瘤学和神经学影像的基石，通过追踪放射性示踪剂分布来可视化代谢信息，能够在解剖结构变化之前检测早期疾病征兆。放射报告是临床决策的关键，但手动撰写耗时费力，给放射科医师带来沉重行政负担。

近年来 VLM 在医学应用中展现出巨大潜力。然而，现有 VLM 的医学应用主要集中在结构性成像模态（X 光、CT），而 PET 影像的独特特征（分子水平成像、代谢信息、示踪剂摄取模式解读）被严重忽视。

核心问题：**VLM 距离有效的 PET 影像放射报告生成还有多远？** 目前缺乏专用数据集和评估框架来回答这一问题。

PET 报告生成的独特挑战包括：（1）需要整合功能和解剖信息的能力；（2）需要解读示踪剂摄取模式的专业知识；（3）全身成像涵盖数十个器官，要求广泛的医学知识；（4）现有 NLG 指标无法评估诊断准确性。

## 方法详解

### 整体框架

PET2Rep 是一个评估基准，而非一个新的模型架构。其核心贡献在于数据集构建、评估流水线设计和 30 个 VLM 的系统评测。

Pipeline：PET/CT 图像预处理（CT 重采样到 PET 分辨率、z-score 标准化、SUV 归一化、PET/CT 融合）→ 关键切片选择（冠状面）→ VLM 推理（结合标准化 prompt）→ 与 ground-truth 对比评估（NLG 指标 + CE 指标 + 人工评分）。

### 关键设计

1. **数据集构建**:

    - 功能：构建首个 PET/CT 报告生成专用数据集
    - 核心思路：从真实临床场景收集 565 例全身 FDG PET/CT 影像及配对结构化放射报告，报告按照放射科培训模板设计，从头到脚系统性描述所有检出异常
    - 设计动机：（1）现有医学基准多局限于特定解剖区域（胸部 X 光、腹部 CT），PET2Rep 覆盖从头颈到近端四肢的全身范围；（2）数据来自实际临床，非公共影像库再加工，避免数据泄漏风险和表面化任务设计

2. **关键切片选择策略**:

    - 功能：将 3D PET/CT 体积转换为 2D 切片输入现有 VLM
    - 核心思路：选择冠状面作为切片采样视图（临床惯例），设计两种输入模式：
        - 分离输入：每个关键位置提供 PET 和 CT 各一张切片（共 6 张图像）
        - 融合输入：将伪彩 PET 叠加到灰度 CT 上（共 3 张融合图像）
    - 设计动机：分离输入考验模型学习功能-结构关联的能力，融合输入模拟放射科医师最终使用的可视化方式

3. **PET 临床效能（CE）指标**:

    - 功能：评估生成报告中对关键器官放射性示踪剂摄取模式描述的质量
    - 核心思路：定义 19 个关键器官/结构，为每个器官提取四种摄取状态（摄取增高、摄取减低、摄取缺失、正常），将报告评估从文本匹配问题转化为多标签分类评估。计算三个阳性类别（增高/减低/缺失）的宏平均精确率、召回率和 F1 值
    - 设计动机：NLG 指标（BLEU、ROUGE-L、METEOR）只关注文本相似度，无法区分诊断结论相反但用词相似的报告。CE 指标更贴近临床诊断的核心需求

4. **标准化 Prompt 设计**:

    - 功能：为 VLM 提供统一的输入格式
    - 核心思路：包含影像模态说明、临床任务描述和基于放射科培训指南的结构化报告模板
    - 设计动机：确保图像解读以与专家撰写的报告一致的格式表达，便于公平比较

### 评估设置

- 零样本评估（zero-shot），测试模型泛化能力
- 评估 30 个 VLM：19 个通用 VLM + 11 个医疗专用 VLM
- 通用模型：Qwen2.5-VL 系列、InternVL3 系列、Yi-VL 系列、LLaVA、DeepSeek-VL2 等
- 医疗模型：LLaVA-Med、Med-Flamingo、MedGemma 系列、Lingshu 系列、MedVLM-R1 等
- 闭源模型：Gemini 2.5 Pro、GPT-4o、Moonshot-v1、Qwen-VL-Max

## 实验关键数据

### 主实验（代表性模型性能，融合输入模式）

| 模型 | BL-4 | MTR | RG-L | CE-Pre | CE-Rec | CE-F1 | 综合(%) |
|------|------|-----|------|--------|--------|-------|---------|
| 模板基线 | 0.315 | 0.148 | 0.511 | 0.228 | 0.222 | 0.225 | 27.5 |
| Qwen2.5-VL-7B | 0.306 | 0.139 | 0.509 | 0.228 | 0.202 | 0.214 | 26.6 |
| MedGemma-4B | 0.287 | 0.121 | 0.488 | 0.236 | 0.225 | 0.230 | 26.4 |
| Lingshu-32B | 0.299 | 0.153 | 0.494 | 0.233 | 0.207 | 0.219 | 26.8 |
| GPT-4o | 0.213 | 0.032 | 0.417 | 0.254 | 0.073 | 0.113 | 18.5 |
| Gemini 2.5 Pro | 0.154 | 0.020 | 0.403 | 0.239 | 0.031 | 0.055 | 15.0 |

### 消融实验（模型规模 vs 性能）

| 模型系列 | 小模型 | 大模型 | 趋势 |
|----------|--------|--------|------|
| Qwen2.5-VL | 7B: 26.6% | 72B: 18.7% | 大模型反而更差 |
| InternVL3 | 8B: 24.4% | 78B: 22.7% | 大模型反而更差 |
| MedGemma | 4B: 26.4% | 27B: 20.1% | 大模型反而更差 |

### 关键发现
- **全面失败**：所有 VLM 在 PET 报告生成上表现有限，多数无法超越简单模板基线
- **SOTA 仅勉强持平基线**：最好的模型（Lingshu-32B、MedGemma-4B）综合分仅约 26-27%，仅与模板基线持平
- **更大不等于更好**：同系列中更大模型反而表现更差，可能因缺乏领域特定数据和任务导向训练
- **闭源模型表现不佳**：GPT-4o（18.5%）、Gemini 2.5 Pro（15.0%）远逊于开源小模型
- **NLG 高≠诊断准确**：部分模型 NLG 指标尚可但 CE 指标极低，说明生成了通顺但临床无用的文本
- **人工评分确认**：两位放射科医师评估也确认现有模型输出基本不可用
- **多种失败模式**：拒绝回答、空输出、不遵循模板、生成无关信息（如编造患者姓名年龄）

## 亮点与洞察

- **领域开创性**：首个 PET/CT 报告生成基准，填补了功能影像学在 VLM 评估中的空白
- **全身覆盖**：涵盖 19 个关键器官/结构，远超现有局限于胸部/腹部的基准
- **CE 指标设计**：将报告评估从文本匹配转化为多标签分类问题，更贴近临床诊断本质
- **"越大越笨"现象**：揭示了大模型在高度专业化结构化任务上可能不如小模型的反直觉现象
- **真实临床数据**：避免了公共数据集的数据泄漏问题，真实反映 VLM 的泛化能力

## 局限与展望

- 仅使用 2D 切片，未充分利用 3D 空间信息和体积信息
- 未纳入 SUV 值、病灶体积等关键定量指标
- 当前仅支持中文报告，缺乏多语言评估
- 数据量（565 例）相对有限
- 仅做零样本评估，未探索微调后的性能
- 冠状面切片可能遗漏某些轴位面更清晰的病变

## 相关工作与启发

- **vs CT2Rep**: CT 报告生成集中在特定解剖区域，PET2Rep 要求全身综合评估
- **vs 胸部 X 光基准 (GEMeX)**: X 光聚焦胸部病理，PET 需要代谢信息解读
- **vs GMAI-MMBench**: 现有医学多模态基准多为 VQA 形式，测试表面理解；PET2Rep 要求深层临床推理

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个 PET 报告生成基准，CE 指标设计创新，填补重要空白
- 实验充分度: ⭐⭐⭐⭐⭐ 30 个模型全面评测，NLG+CE+人工三维评估，多种输入模式
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，实验分析深入，失败模式分类有价值
- 价值: ⭐⭐⭐⭐⭐ 揭示了 VLM 在功能影像报告生成上的巨大差距，为后续研究指明方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] ReCAD: Reinforcement Learning Enhanced Parametric CAD Model Generation with Vision-Language Models](recad_reinforcement_learning_enhanced_parametric_cad_model_generation_with_visio.md)
- [\[AAAI 2026\] LLMC+: Benchmarking Vision-Language Model Compression with a Plug-and-play Toolkit](llmc_benchmarking_vision-language_model_compression_with_a_plug-and-play_toolkit.md)
- [\[ICML 2026\] WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](../../ICML2026/multimodal_vlm/weathersyn_an_instruction_tuning_mllm_for_weather_forecasting_report_generation.md)
- [\[ACL 2026\] CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation](../../ACL2026/multimodal_vlm/coggen_a_cognitively_inspired_recursive_framework_for_deep_research_report_gener.md)
- [\[AAAI 2026\] Towards Long-window Anchoring in Vision-Language Model Distillation](towards_long-window_anchoring_in_vision-language_model_distillation.md)

</div>

<!-- RELATED:END -->
