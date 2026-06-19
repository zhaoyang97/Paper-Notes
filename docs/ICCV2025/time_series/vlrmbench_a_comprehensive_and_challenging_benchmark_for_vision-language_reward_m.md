---
title: >-
  [论文解读] VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models
description: >-
  [ICCV 2025][时间序列][奖励模型] 提出 VLRMBench，一个包含 12634 个问题、12 项任务的综合且具有挑战性的视觉语言奖励模型（VLRM）基准，覆盖过程理解、结果判断和批评生成三大方面，在 26 个模型上的广泛实验揭示了当前 VLRM 的显著不足。 奖励模型（RM）在大模型的训练和推理阶段扮演重要角…
tags:
  - "ICCV 2025"
  - "时间序列"
  - "奖励模型"
  - "视觉语言理解"
  - "benchmark"
  - "过程推理"
  - "多模态评估"
---

# VLRMBench: A Comprehensive and Challenging Benchmark for Vision-Language Reward Models

**会议**: ICCV 2025  
**arXiv**: [2503.07478](https://arxiv.org/abs/2503.07478)  
**代码**: [https://github.com/JCruan519/VLRMBench](https://github.com/JCruan519/VLRMBench)  
**领域**: Time Series / Multimodal Evaluation  
**关键词**: 奖励模型, 视觉语言理解, benchmark, 过程推理, 多模态评估

## 一句话总结

提出 VLRMBench，一个包含 12634 个问题、12 项任务的综合且具有挑战性的视觉语言奖励模型（VLRM）基准，覆盖过程理解、结果判断和批评生成三大方面，在 26 个模型上的广泛实验揭示了当前 VLRM 的显著不足。

## 研究背景与动机

奖励模型（RM）在大模型的训练和推理阶段扮演重要角色：训练前用于过滤高质量样本、训练中用于偏好优化（如 RLAIF）、推理时用于测试时缩放（TTS）。然而，现有 VLRM 基准存在严重不足：

**评估维度单一**：VLRewardBench 仅包含 Pairwise Comparison（两两比较），无法全面评估 VLRM 能力

**缺乏步骤级标注**：多数基准不包含推理步骤级别的标签

**聚焦语言领域**：现有 RM 基准（如 PRMBench、ProcessBench）主要针对纯文本，不适用于视觉语言场景

**挑战性不足**：简单的基准无法暴露 VLRM 的潜在缺陷

## 方法详解

### 整体框架

VLRMBench 构建流程分三阶段：(1) 数据收集与过滤；(2) 推理过程生成与步骤分割；(3) 基于三大主题设计 12 项任务。

### 关键设计

1. **协作式数据过滤与生成管线（Collaborative Data Filtering & Generation Pipeline）**: 

    - **质量过滤**：使用 Qwen2VL-7B 在无图像条件下回答问题，答对则说明质量低（纯靠文本就能解答），剔除
    - **难度过滤**：加入图像后 Qwen2VL-7B 仍能答对则太简单，剔除。从 16550 个样本过滤至 6715 个
    - **推理过程生成**：QVQ-72B-preview 生成推理过程（仅保留正确答案的样本）
    - **步骤分割**：GPT-4o 进行语义级步骤分割
    - **人工审核**：3 名博士生验证和纠正推理过程中的错误
    - 最终保留 1000 个高质量样本，覆盖数学推理、幻觉理解、多图理解

2. **步骤级任务（Step-based，8 项）**: 评估 VLRM 的推理过程理解能力

    - **SC（Step Correctness）**：检测推理步骤中的注入错误
    - **RD（Redundancy Detection）**：识别推理过程中的冗余信息
    - **CM（Confidence Misdirection）**：在错误步骤中加入高置信表述，测试鲁棒性
    - **EH（Existential Hallucination）**：检测推理中不存在于图像的实体
    - **AH（Attribute Hallucination）**：检测实体属性的错误描述
    - **DE（Detail Error）**：检测数值计算或符号的细粒度错误
    - **SR（Spatial Relationship）**：检测空间关系描述的错误
    - **IC（Image Confusion）**：检测多图任务中的图像引用错误

3. **结果级任务（Outcome-based，2 项）**: 评估 VLRM 的结果判断能力

    - **MJ（Multi-solution Judgment）**：比较同一问题的不同推理过程质量
    - **FF（Forecasting Future）**：基于前 m 步推理预测最终答案的正确性

4. **批评级任务（Criticism-based，2 项）**: 评估 VLRM 的错误分析和纠正能力

    - **ERA（Error Reason Analysis）**：分析错误推理步骤的原因
    - **EC（Error Correction）**：直接纠正错误并输出正确推理

### 损失函数 / 训练策略

本文为基准测试工作，无需训练。评估指标包括：
- 步骤级任务：F1-Score（平衡精确率和召回率）
- 结果级任务：Accuracy
- 批评级任务：Win Rate（使用 GPT-4o 作为裁判）

## 实验关键数据

### 主实验

**各模型在步骤级任务上的平均 F1-Score（部分模型）**:

| 模型 | SC | RD | CM | EH | AH | DE | 步骤均值 |
|------|-----|-----|-----|-----|-----|-----|---------|
| GPT-4o | 73.7 | 50.6 | 66.6 | 57.6 | 58.6 | 71.8 | 62.4 |
| Claude-3.5-Sonnet | 70.8 | 53.7 | 65.7 | 63.9 | 62.8 | 63.4 | 62.9 |
| Qwen2.5VL-72B | 72.8 | 41.7 | 70.4 | 64.6 | 59.9 | 72.4 | 62.6 |
| Qwen2.5VL-7B | 43.4 | 33.2 | 37.8 | 22.8 | 23.9 | 45.5 | 33.4 |
| InternVL2.5-8B | 36.6 | 28.4 | 31.1 | 21.9 | 21.2 | 36.5 | 28.6 |
| Ovis2-34B | 65.3 | 51.1 | 64.5 | 54.5 | 51.6 | 59.6 | 57.0 |

**结果级和批评级任务表现**:

| 模型 | MJ(Acc) | FF(Acc) | 结果均值 | ERA(WinRate) | EC(WinRate) |
|------|---------|---------|---------|-------------|-------------|
| GPT-4o | 58.4 | 76.0 | 66.3 | 0.0 | 0.0 |
| Claude-3.5-Sonnet | 82.2 | 75.1 | 79.0 | 60.6/25.5/13.9 | 21.2/53.9/24.9 |
| Qwen2.5VL-72B | 65.6 | 80.2 | 72.1 | 74.1/15.1/10.8 | 15.6/77.0/7.3 |
| Qwen2.5VL-7B | 26.0 | 70.7 | 46.0 | 37.7/22.0/40.3 | 9.3/51.9/38.8 |

### 消融实验

**模型规模对步骤级任务的影响**:

| 模型组 | 规模 | 步骤均值 | 结果均值 |
|--------|------|---------|---------|
| 小型组（<10B） | 2B-8B | 29.8 | 45.2 |
| 中型组（10-40B） | 11B-38B | 45.9 | 54.7 |
| 大型组（>40B） | 72B-90B | 46.1 | 56.8 |
| 闭源组 | - | 62.4+ | 66.3+ |

### 关键发现

- **即使是最先进的 GPT-4o，在 FF（预测未来）任务中也仅达 76.0% 准确率**，在步骤级任务中平均 F1 仅 62.4%
- **开源模型正在追赶闭源模型**：Qwen2.5VL-72B 在批评任务中胜过 GPT-4o（ERA win rate 74.1% vs 0.0%）
- **CM 任务证实 VLRM 容易被高置信表述误导**：CM 的 F1 普遍低于 SC
- **冗余检测（RD）是最具挑战性的步骤级任务**：所有模型在 RD 上的 F1 分数最低
- 模型从小型到中型提升明显（29.8→45.9），但中型到大型提升有限（45.9→46.1）

## 亮点与洞察

- **12 项任务的全面设计**：从过程、结果、批评三个维度全面评估 VLRM，远超现有仅做两两比较的基准
- **双重过滤机制**：无图回答正确→低质量、有图回答正确→太简单，确保保留高质量高难度样本
- **Confidence Misdirection 任务的创新**：测试模型是否会被"definitely""without a doubt"等置信词汇误导
- **Forecasting Future 的实用价值**：如果能预判推理正确性，可大幅加速 TTS 推理

## 局限与展望

- 推理过程依赖 QVQ-72B-preview 生成，可能引入该模型的偏差
- 数学推理类样本占比大，幻觉和多图样本相对较少
- 批评级任务使用 GPT-4o 作为裁判存在评判偏差风险
- 当前仅评估文本形式的奖励模型，未考虑数值型 RM
- 基准的动态更新机制缺失，模型可能过拟合固定测试集

## 相关工作与启发

- PRMBench：过程奖励模型的细粒度评估（纯文本领域）
- VLRewardBench：首个视觉语言 RM 基准，但仅含单一任务
- ProcessBench：步骤级错误检测的数学推理评估
- 启发：奖励模型的能力评估需要像 VLRMBench 这样的多维度框架，单一任务（如 pairwise comparison）远不够全面

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个综合的 VLRM 基准，12 项任务设计独特
- **实验充分度**: ⭐⭐⭐⭐⭐ 26 个模型的广泛评估，分小/中/大/闭源四组分析
- **写作质量**: ⭐⭐⭐⭐ 任务设计阐述清晰，表格丰富
- **价值**: ⭐⭐⭐⭐ 填补 VLRM 综合评估的空白，揭示了当前模型的关键弱点

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Omni-iEEG: A Large-Scale, Comprehensive iEEG Dataset and Benchmark for Epilepsy Research](../../ICLR2026/time_series/omni-ieeg_a_large-scale_comprehensive_ieeg_dataset_and_benchmark_for_epilepsy_re.md)
- [\[NeurIPS 2025\] CausalDynamics: A Large-Scale Benchmark for Structural Discovery of Dynamical Causal Models](../../NeurIPS2025/time_series/causaldynamics_a_large-scale_benchmark_for_structural_discovery_of_dynamical_cau.md)
- [\[ICML 2026\] Building Social World Models with Large Language Models](../../ICML2026/time_series/building_social_world_models_with_large_language_models.md)
- [\[ACL 2025\] G2S: A General-to-Specific Learning Framework for Temporal Knowledge Graph Forecasting with Large Language Models](../../ACL2025/time_series/g2s_a_general-to-specific_learning_framework_for_temporal_knowledge_graph_foreca.md)
- [\[NeurIPS 2025\] Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models](../../NeurIPS2025/time_series/causal_masking_on_spatial_data_an_information-theoretic_case_for_learning_spatia.md)

</div>

<!-- RELATED:END -->
