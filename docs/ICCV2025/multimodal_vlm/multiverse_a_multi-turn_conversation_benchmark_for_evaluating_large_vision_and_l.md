---
title: >-
  [论文解读] MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models
description: >-
  [ICCV 2025][多模态VLM][多轮对话] 提出 MultiVerse 多轮对话评估基准，从 12 个 VLM 评估数据集中收集 647 段对话，覆盖 484 种任务和 484 种交互目标，采用 checklist 评估方法发现即使最强的 GPT-4o 在复杂多轮对话中仅达 50% 的成功率。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "多轮对话"
  - "VLM评估"
  - "基准数据集"
  - "checklist评估"
  - "上下文学习"
---

# MultiVerse: A Multi-Turn Conversation Benchmark for Evaluating Large Vision and Language Models

**会议**: ICCV 2025  
**arXiv**: [2510.16641](https://arxiv.org/abs/2510.16641)  
**代码**: [MultiVerse](https://passing2961.github.io/multiverse-project-page/)  
**领域**: 多模态VLM  
**关键词**: 多轮对话, VLM评估, 基准数据集, checklist评估, 上下文学习  

## 一句话总结

提出 MultiVerse 多轮对话评估基准，从 12 个 VLM 评估数据集中收集 647 段对话，覆盖 484 种任务和 484 种交互目标，采用 checklist 评估方法发现即使最强的 GPT-4o 在复杂多轮对话中仅达 50% 的成功率。

## 研究背景与动机

VLM 在单轮基准上表现优异，但真实场景往往涉及多轮交互式对话。现有多轮评估基准存在显著不足：

**MMDU 的局限**：以知识导向图像为主（风景、动物、艺术），任务类型不到 15 种，查询风格有限（词汇多样性低），来源局限于 WIT

**ConvBench 的局限**：虽然覆盖 219 个子任务，但缺乏数学、编程等高级推理任务，用户查询语言结构简单

**广度和深度不足**：两者都未充分覆盖现实场景中的多样任务类型和推理深度

核心问题：**在单轮基准上表现出色的 VLM，是否也能满足用户在更交互式、多轮场景中的需求？**

## 方法详解

### 数据集构建管线

MultiVerse 的构建分为五个步骤：

**Step 1: 源图像收集**
从 12 个 VLM 评估基准（MegaBench、CharXiv、MMMU、MMMU-Pro、NaturalBench 等）收集 49,700 张图像，经过：
- 去重（pHash）：去除 29.1% 重复
- 质量评分（GPT-4o 1-5 分）：去除 64.77% 低质量图像
- 类别分类：57 个类别，去除小类
- 加权采样：上限 1K 张

**Step 2: 个人背景生成**
为每段对话创建虚构角色（年龄、职业、爱好）和场景上下文。这一步受真实用户行为启发——人们通常带着特定目标提问。GPT-4o 生成角色背景和对话目标。实验发现纯图像生成的对话多样性和质量较低。

**Step 3: 多轮对话生成**
基于角色和目标，由 GPT-4o 生成 4 轮对话（8 条消息），遵循四个原则：
- 详尽且信息丰富的回答
- **递增复杂度**：后续查询逐步更具挑战性
- 多样语言风格（非"describe"/"what is"等模板化提问）
- 固定 4 轮长度

**Step 4: 人工审查**
按三个标准筛选：
- 自然性和真实性（去除 48 条）
- 正确性（去除 65 条）
- 盲测（去除 104 条，确保需要图像才能回答）

最终保留 **647 段对话**。

**Step 5: Checklist 生成**
为每个 turn 的每个查询生成实例级 checklist，包含多个二元问题，覆盖 37 个关键方面（感知准确性、语言清晰度、事实正确性等）。使用 GPT-4o 和 Claude-3.5-Sonnet 生成后人工验证。

### 评估指标

采用两个子指标：
- **Checklist Completion Ratio**：回答满足 checklist 项目的比例（"Yes" 占比）
- **Quality Assessment**：1-10 整数评分（缩放至 10-100）

最终分数 = 两者乘积。实验验证两者强正相关（$R^2 = 0.44$）。

### 数据集统计

- 647 段对话，平均 3.91 轮
- 8 大交互目标 / 484 子交互目标
- 9 大任务类型 / 484 子任务
- 25 个图像类别 / 384 子类别
- 平均查询长度 30.53 token，回答长度 221.51 token
- 21,995 个唯一 checklist 项
- 查询/回答词汇多样性（MTLD）分别为 112.0 / 118.0

## 实验

### 18 个 VLM 的 Oracle 设置评估

| 模型 | Turn 1 | Turn 2 | Turn 3 | Turn 4 | 平均 | 斜率 r |
|------|------|------|------|------|------|------|
| GPT-4o | 48.56 | 50.28 | 50.54 | 49.12 | **49.63** | 0.19 |
| Qwen2.5-VL-7B | 45.13 | 47.60 | 49.31 | 50.58 | 48.15 | 1.81 |
| Qwen2.5-VL-72B | 52.05 | 47.72 | 45.82 | 46.19 | 47.95 | -1.95 |
| Claude-3.5-Sonnet | 46.60 | 47.16 | 48.30 | 45.00 | 46.76 | -0.37 |
| Gemini-2.0-Flash | 42.03 | 49.37 | 51.23 | 48.41 | 47.76 | 2.10 |
| LLaVA-1.5-7B | 9.10 | 26.43 | 29.14 | 31.81 | 24.12 | 7.08 |
| InternVL2.5-1B | 13.93 | 21.36 | 23.51 | 26.08 | 21.22 | 3.86 |

**关键发现**：
- **即使 GPT-4o 也仅达 49.63%**，说明多轮交互仍是巨大挑战
- 弱模型（LLaVA-1.5-7B）斜率最大（7.08），说明提供 ground-truth 对话历史的上下文学习效应对弱模型帮助最大
- Qwen2.5-VL-72B 斜率为负（-1.95），可能是其语言风格与 GPT-4o 生成的参考对话不匹配

### Oracle vs. Self-Prediction

| 模型 | Oracle | Self-Prediction | 差距 |
|------|------|------|------|
| GPT-4o | 49.63 | - | - |
| Qwen2.5-VL-72B | 47.95 | - | -44.64% (max) |
| Qwen2.5-VL-7B | 48.15 | - | -30.44% |

所有模型在 Self-Prediction 下性能显著下降，最大降幅达 44.64%（Qwen2.5-VL-72B），说明提供精确对话上下文对性能至关重要。

### 按交互目标分析

| 模型 | 验证 | 分析 | 探索 | 优化 | 计算 | 理解 | 研究 | 创作 |
|------|------|------|------|------|------|------|------|------|
| GPT-4o | 46.80 | **53.67** | 42.70 | 46.36 | 50.54 | **56.41** | 43.50 | 44.51 |
| Qwen2.5-VL-7B | 46.14 | 54.23 | 44.23 | 40.64 | 44.86 | 54.71 | 39.73 | 44.45 |

**关键发现**：
- 模型在"分析"和"理解"任务上表现较好
- "优化"和"研究"等需要创新思维的任务表现较弱
- 弱模型在"验证"和"计算"等可验证任务上尤其困难

### 模型缩放效应

在 InternVL2.5、LLaMA-3.2、Qwen2.5-VL 三个系列中，更大模型通常表现更好，但效果因任务而异：
- Qwen2.5-VL-72B 在可验证任务（数学、编程）上更强
- Qwen2.5-VL-7B 在创意任务上反而更好

### 冗长偏差分析

GPT-4o 评估器的回答长度与性能的线性相关 $R^2$ 随轮次增加而减弱，表明 checklist 评估方法可有效减轻冗长偏差。

## 亮点与洞察

1. **任务广度与深度并重**：覆盖数学、编程、图表解读等高级推理任务，弥补了 MMDU 和 ConvBench 的不足
2. **Personal Background → Conversation 管线**：通过虚构角色背景驱动对话生成，显著提升了语言多样性和对话的真实感
3. **上下文学习效应的发现**：弱模型通过 ground-truth 对话历史获得最大提升，揭示了对话上下文作为"指导"的潜力
4. **Checklist 评估的鲁棒性**：比简单评分更稳健，减轻了冗长偏差，覆盖 37 个评估方面

## 局限性

1. 参考对话由 GPT-4o 生成，语言风格偏差可能不利于与 GPT-4o 风格差异大的模型
2. 固定 4 轮对话可能无法反映更长对话（10+ 轮）中的退化问题
3. 647 段对话的规模相对较小，统计显著性可能在细分类上受限
4. 依赖 GPT-4o 作为评估器存在已知偏差（如对自身风格的偏好）

## 相关工作

- **单轮 VLM 基准**：MMBench、MMMU、MathVista、MM-Vet 等评估感知与推理
- **多轮 LLM 基准**：MT-Bench-101、Multi-IF 等在 NLP 领域已有探索
- **多轮 VLM 基准**：MMDU（知识导向）、ConvBench（感知/推理/创作三类）

## 评分

| 维度 | 分数 |
|------|------|
| 创新性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 总体推荐 | 7.5/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] VisNumBench: Evaluating Number Sense of Multimodal Large Language Models](visnumbench_evaluating_number_sense_of_multimodal_large_language_models.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[CVPR 2026\] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](../../CVPR2026/multimodal_vlm/llavashield_multimodal_multiturn_safety.md)
- [\[ICCV 2025\] CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting](capture_evaluating_spatial_reasoning_in_vision_language_models_via_occluded_obje.md)
- [\[ICCV 2025\] GRAB: A Challenging GRaph Analysis Benchmark for Large Multimodal Models](grab_a_challenging_graph_analysis_benchmark_for_large_multimodal_models.md)

</div>

<!-- RELATED:END -->
