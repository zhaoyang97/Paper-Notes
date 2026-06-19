---
title: >-
  [论文解读] MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding
description: >-
  [ICML 2025][医学图像][医学 QA 基准] MedXpertQA 构建了包含 4460 题、覆盖 17 个专科和 11 个身体系统的专家级医学 QA 基准，通过严格的筛选增强和数据合成防泄漏，评估了 18 个主流模型，并专门设计了推理子集用于评估 o1 类推理模型。 领域现状：医学 QA 是评估 LLM/MLLM…
tags:
  - "ICML 2025"
  - "医学图像"
  - "医学 QA 基准"
  - "专家级推理"
  - "多模态评估"
  - "数据泄漏防护"
  - "o1 推理评估"
---

# MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding

**会议**: ICML 2025  
**arXiv**: [2501.18362](https://arxiv.org/abs/2501.18362)  
**代码**: [https://github.com/TsinghuaC3I/MedXpertQA](https://github.com/TsinghuaC3I/MedXpertQA)  
**领域**: 医疗NLP  
**关键词**: 医学 QA 基准, 专家级推理, 多模态评估, 数据泄漏防护, o1 推理评估

## 一句话总结
MedXpertQA 构建了包含 4460 题、覆盖 17 个专科和 11 个身体系统的专家级医学 QA 基准，通过严格的筛选增强和数据合成防泄漏，评估了 18 个主流模型，并专门设计了推理子集用于评估 o1 类推理模型。

## 研究背景与动机

**领域现状**：医学 QA 是评估 LLM/MLLM 医学能力的核心 benchmark 类型。现有基准如 MedQA、MedMCQA、PubMedQA 等已被广泛使用。GPT-4 等模型在多个医学 QA 上已接近或超过人类水平。

**现有痛点**：(1) 难度不足——GPT-4 在 MedQA 上已达 90%+，天花板效应严重；(2) 数据泄漏——训练数据中可能包含测试题；(3) 多模态不足——现有多模态医学 benchmark 多为简单的图像 caption QA，缺乏真正的临床推理题；(4) 推理评估缺失——无专门评估 o1 类推理能力的医学 benchmark。

**核心矛盾**：需要一个足够难（区分模型能力）、无泄漏（公平评估）、临床相关（不是百科问答而是专家级诊断推理）的 benchmark。

**本文目标**：创建一个真正的专家级医学推理和理解 benchmark。

**切入角度**：从专科委员会考试题出发，经过严格筛选增强（过滤简单题）+ 数据合成（防泄漏）+ 专家多轮审核。

**核心 idea**：用专科委员会级别难度 + 数据合成防泄漏 + 推理导向子集，构建真正能区分当前最强模型的医学 benchmark。

## 方法详解

### 整体框架

- **数据来源**：医学专科委员会考试题（如 USMLE Step 3、各专科 board exam）
- **文本子集(Text)**：4460 题纯文本 QA
- **多模态子集(MM)**：包含医学图像（CT/MRI/X-ray/病理等）+ 患者记录 + 检查结果的复杂 QA
- **推理子集**：专为评估 o1 类模型设计的需要多步推理的题目

### 关键设计

1. **严格过滤与增强机制**:

    - 第一轮：去除 GPT-4 能轻松答对的"简单"题目
    - 第二轮：增强难度——修改干扰项、增加临床背景复杂度
    - 保留策略：只保留需要专科知识和多步推理的题目
    - **设计动机**：现有 benchmark 的主要问题就是太简单

2. **数据合成防泄漏**:

    - 对原始题目进行改写/合成——改变临床场景、数值、选项
    - 确保改写后的题目在语义上等价但文本上不同
    - 多轮自动+人工检查确保不被直接搜索到
    - **设计动机**：LLM 训练集中可能包含公开考试题，必须防止数据泄漏

3. **多模态子集设计 (MM)**:

    - 不是简单的"看图说话"——每题包含：医学图像 + 患者主诉 + 病史 + 实验室检查结果
    - 需要综合多种信息源进行诊断推理
    - 图像类型多样：CT、MRI、X-ray、超声、皮肤镜、病理切片等
    - **设计动机**：真实临床场景是多信息源融合推理

4. **推理导向子集**:

    - 专门筛选需要≥3步推理的题目
    - 适合评估 o1、o3 等推理增强模型
    - 包含诊断推理链标注
    - **设计动机**：医学是评估推理能力的天然领域（复杂但有明确答案）

### 损失函数 / 训练策略（Benchmark，无训练）

- 评估指标：选择题准确率
- 评估方式：zero-shot、few-shot、Chain-of-Thought
- 对开卷/闭卷分别评估
- 多模态题额外评估视觉理解能力

## 实验关键数据

### 主实验

| 模型 | Text 准确率 | MM 准确率 | 推理子集 |
|------|-----------|----------|---------|
| GPT-4o | ~65% | ~55% | ~60% |
| Claude 3.5 | ~63% | ~53% | ~58% |
| o1 | ~70% | - | ~68% |
| Gemini 1.5 Pro | ~60% | ~50% | ~55% |
| Med-PaLM 2 | ~58% | - | - |
| LLaMA 3 70B | ~50% | - | ~45% |
| 开源 MLLM | <50% | <45% | <40% |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 全部题 vs 增强后 | 准确率差 15-20% | 过滤有效提升难度 |
| 原始题 vs 合成题 | 准确率接近 | 合成不改变难度 |
| 文本 vs 多模态 | MM 更难 | 多信息源融合具挑战性 |
| 标准 QA vs 推理子集 | 推理更难 | 多步推理对模型要求高 |
| 无 CoT vs CoT | CoT 帮助 | 推理题尤其受益 |

### 关键发现

- **当前最强模型仍达不到专家水平**：GPT-4o 在 Text 上约 65%，远低于专科医生水平
- **多模态是软肋**：所有模型在 MM 子集上比 Text 低 ~10%
- **o1 类模型有优势但有限**：推理增强在推理子集上提升约 5-8%
- **开源模型差距大**：与闭源模型差距 15-20%
- 数据泄漏防护有效：合成后的题目准确率与原始题接近

## 亮点与洞察

1. **难度足够**：当前最强模型也只有 65-70%，有足够区分度
2. **数据安全**：多层防泄漏机制确保评估公平
3. **多模态创新**：不是简单 VQA，而是临床级多信息源推理
4. **推理评估**：首个针对 o1 类模型的医学推理 benchmark
5. **超越医学**：为推理能力评估提供了一个丰富的现实世界测试场景

## 局限与展望

1. 以选择题为主，未覆盖自由文本生成的诊断报告
2. 英语为主，多语言医学 QA 需扩展
3. 部分专科覆盖不均匀（某些小专科题量少）
4. 多模态图像质量和分辨率受限于来源

## 相关工作与启发

- MedQA、MedMCQA 是前代医学 QA benchmark
- MMMU、ScienceQA 是多学科多模态评估的对照
- 启发：数据合成防泄漏的方法论可推广到其他 benchmark 构建

## 评分
- 新颖性: ⭐⭐⭐⭐ Benchmark 工作，设计创新在于难度控制和推理子集
- 实验充分度: ⭐⭐⭐⭐⭐ 18 个模型的全面评估
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，设计合理
- 价值: ⭐⭐⭐⭐⭐ 填补了专家级医学推理评估的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] THUNDER: Tile-level Histopathology image UNDERstanding benchmark](../../NeurIPS2025/medical_imaging/thunder_tile-level_histopathology_image_understanding_benchmark.md)
- [\[ICML 2025\] Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees](mastering_multiple-expert_routing_realizable_h-consistency_and_strong_guarantees.md)
- [\[NeurIPS 2025\] SMMILE: An Expert-Driven Benchmark for Multimodal Medical In-Context Learning](../../NeurIPS2025/medical_imaging/smmile_an_expert-driven_benchmark_for_multimodal_medical_in-context_learning.md)
- [\[CVPR 2025\] Interactive Medical Image Analysis with Concept-based Similarity Reasoning](../../CVPR2025/medical_imaging/interactive_medical_image_analysis_with_concept-based_similarity_reasoning.md)
- [\[CVPR 2026\] IBISAgent: Reinforcing Pixel-Level Visual Reasoning in MLLMs for Universal Biomedical Object Referring and Segmentation](../../CVPR2026/medical_imaging/ibisagent_reinforcing_pixel-level_visual_reasoning_in_mllms_for_universal_biomed.md)

</div>

<!-- RELATED:END -->
