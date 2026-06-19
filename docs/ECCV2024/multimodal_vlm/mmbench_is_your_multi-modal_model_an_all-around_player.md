---
title: >-
  [论文解读] MMBench: Is Your Multi-modal Model an All-Around Player?
description: >-
  [ECCV2024][多模态VLM][VLM benchmark] 提出 MMBench——一个包含 3217 道多选题、覆盖 20 个细粒度能力维度的双语（英/中）视觉语言模型评测基准，并设计了 CircularEval 循环评测策略和基于 LLM 的选项提取机制，显著提升了评测的鲁棒性和公平性。
tags:
  - "ECCV2024"
  - "多模态VLM"
  - "VLM benchmark"
  - "多模态"
  - "CircularEval"
  - "choice extraction"
  - "bilingual benchmark"
---

# MMBench: Is Your Multi-modal Model an All-Around Player?

**会议**: ECCV2024  
**arXiv**: [2307.06281](https://arxiv.org/abs/2307.06281)  
**代码**: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)  
**领域**: 多模态VLM  
**关键词**: VLM benchmark, multi-modal evaluation, CircularEval, choice extraction, bilingual benchmark

## 一句话总结

提出 MMBench——一个包含 3217 道多选题、覆盖 20 个细粒度能力维度的双语（英/中）视觉语言模型评测基准，并设计了 CircularEval 循环评测策略和基于 LLM 的选项提取机制，显著提升了评测的鲁棒性和公平性。

## 背景与动机

大型视觉语言模型（VLM）近年进展迅猛，但缺乏系统、可靠的定量评测手段：

- **传统客观基准**（VQAv2、COCO Caption 等）存在「假阴性」问题——预测 "bicycle" 而标准答案是 "bike" 就判错；且只能评测单一任务，无法给出**细粒度能力画像**
- **主观评测**（OwlEval、LVLM-eHub）依赖人工标注，成本高、偏差大、不可扩展、难以复现
- 不同 VLM 的**指令跟随能力**参差不齐，许多模型无法直接输出选项标签（A/B/C/D），导致精确匹配式评测严重低估其真实能力

因此需要一个设计系统、评测鲁棒、能力覆盖全面的客观基准。

## 核心问题

1. 如何构建一个**能力覆盖全面**且**数据质量可控**的 VLM 评测基准？
2. 如何解决不同 VLM 指令跟随能力差异带来的**选项提取**困难？
3. 如何消除多选题评测中模型**随机猜测**和**选项偏好**带来的偏差？

## 方法详解

### 1. 层次化能力体系

MMBench 设计了三级能力分类体系：

- **L-1**（2 类）：Perception（感知）、Reasoning（推理）
- **L-2**（6 类）：Coarse Perception (CP)、Fine-grained Perception - Single Instance (FP-S)、Fine-grained Perception - Cross Instance (FP-C)、Attribute Reasoning (AR)、Logic Reasoning (LR)、Relation Reasoning (RR)
- **L-3**（20 类）：涵盖物体定位、动作识别、空间关系、社交推理等细粒度能力

每个 L-3 能力至少包含 125 道题，保持均衡分布。

### 2. 数据收集与质量控制

- **来源**：超过 80% 的题目从互联网收集，其余约 20% 基于公开数据集验证集构造
- **纯文本过滤**：用多个 SOTA LLM（GPT-4、Gemini-Pro 等）仅凭文本推理，若超过半数答对则移除该题（说明无需图像即可作答，不适合评测多模态能力）
- **错误样本过滤**：将所有题目送入多个 SOTA VLM，若所有模型均答错，则人工复查并移除确实有误的样本
- **双语版本**：基于 GPT-4 翻译为中文，保留专有名词，并经人工校验→ MMBench-CN

### 3. LLM 辅助选项提取

针对 VLM 自由文本输出无法直接匹配选项的问题，设计两步提取流程：

- **Step 1**（启发式匹配）：尝试从模型输出中直接提取选项标签 A/B/C/D
- **Step 2**（LLM 提取）：若 Step 1 失败，将题目、选项和模型输出一同发给 GPT-4，让其判断模型预测最匹配哪个选项
- GPT-4 作为选项提取器与人工标注的**对齐率达 91.5%**，远高于 GPT-3.5-Turbo（约 85%）

### 4. CircularEval 循环评测策略

为消除多选题中随机猜测（4 选 1 有 25% 基线）和选项位置偏好带来的偏差：

- 对每道 $N$ 选项的题目进行 $N$ 次推理，每次对选项做**循环移位**（circular shift）
- 仅当模型在**所有 $N$ 次**推理中都答对，才算答对该题
- 实际中模型一旦某次答错即可提前终止，计算开销低于 $N$ 倍
- 效果：相比 VanillaEval（单次推理），CircularEval 普遍降低准确率 8–34 个百分点，能更有效地拉开模型间差距

## 实验关键数据

| 模型 | Overall | CP | FP-S | FP-C | AR | LR | RR |
|---|---|---|---|---|---|---|---|
| InternLM-XComposer2 | **78.1** | 80.4 | 83.5 | 73.0 | 83.7 | 63.6 | 74.4 |
| Qwen-VL-Max | 75.4 | 74.8 | **87.2** | 67.0 | 85.3 | 54.9 | 70.5 |
| GPT-4v | 74.3 | 77.6 | 73.8 | 71.5 | **85.3** | **63.6** | 68.6 |
| LLaVA-InternLM2-20B | 72.3 | 78.3 | 76.6 | **68.2** | 78.4 | 46.2 | 69.4 |
| Gemini-Pro-V | 70.2 | 70.0 | 78.9 | 65.9 | 82.9 | 46.2 | 65.9 |
| Yi-VL-34B | 68.4 | 72.0 | 78.0 | 54.7 | 81.2 | 38.6 | 68.2 |
| OpenFlamingo v2 | 2.3 | 1.1 | 3.5 | 1.5 | 5.3 | 0.0 | 2.7 |

关键发现：

- **LLM 底座至关重要**：同为 LLaVA 架构，将 LLM 从 Vicuna-7B 换成 InternLM2-20B，整体准确率从 63.4% 升至 72.3%，推理能力提升尤为显著
- **模型缩放有效**：MiniGPT4 从 7B 到 13B 提升 8.3%，LLaVA v1.5 从 7B 到 13B 提升 3.5%
- **小模型潜力**：MiniCPM-V（≤3B 参数）在 CircularEval 下仍达 61.4%
- **双语差距小**：Top 模型在 MMBench 与 MMBench-CN 间的差距仅约 1–2%，InternLM-XComposer2 差距不足 1%
- **内容审查影响**：GPT-4v 拒绝回答 1.8% 的测试（主要为名人识别），Gemini-Pro-V 拒答 1.6%

## 亮点

1. **CircularEval 设计巧妙**：通过选项循环移位消除位置偏好和随机猜测，在可接受开销下大幅提升评测鲁棒性
2. **LLM 选项提取器**：优雅地解决了不同 VLM 指令跟随能力差异的问题，与人工对齐率 91.5%
3. **三级能力分类体系**：20 个 L-3 能力维度提供细粒度诊断，可直接定位模型短板
4. **质量控制流程系统**：纯文本过滤 + 全模型答错过滤的双重机制，确保数据质量
5. **双语对齐评测**：英中两版完全对应，可公平比较 VLM 的跨语言能力

## 局限与展望

- 多选题格式本身有局限——无法评测开放式生成、多轮对话、长文本推理等能力
- 质量控制依赖 SOTA 模型，当所有模型都犯同样的错误时可能漏检
- CircularEval 对选项数敏感，2 选项和 4 选项的难度差异大
- 选项提取依赖 GPT-4 API，成本不低且存在 API 版本变化风险
- 評測维度虽覆盖 20 项但未涉及 OCR、图表理解、数学推理等近年热点能力

## 与相关工作的对比

| 基准 | 题数 | 能力维度 | 评测方式 | 双语 | 鲁棒性策略 |
|---|---|---|---|---|---|
| **MMBench** | 3217 | 20（三级） | 多选 + CircularEval | ✓ | CircularEval + LLM 提取 |
| MME | ~2400 | 14 | 是/否 | ✗ | 无 |
| OwlEval | 82 | 多种 | 主观/人工 | ✗ | 无 |
| SEED-Bench | 19K | 12 | 多选 | ✗ | 无 |
| VQAv2 | 1.1M | 单一 | 开放式 | ✗ | 精确匹配 |

相比 MME 的简单是非题，MMBench 的多选题更接近真实推理；相比 SEED-Bench 题量更大但缺乏鲁棒性策略，MMBench 用 CircularEval 保证评测可靠性。

## 启发与关联

- CircularEval 的「多次推理 + 一致性校验」思想可推广到其他多选式评测场景（如代码能力、数学推理基准）
- LLM 辅助选项提取为评测开放式模型提供了通用范式——不再要求模型严格遵循输出格式
- 论文指出 LLM 底座对 VLM 性能的决定性影响，启发后续研究应更关注语言模型本身的选择与对齐
- 评测代码集成在 VLMEvalKit 中，已成为后续 VLM 研究的标准评测工具

## 评分

- 新颖性: ⭐⭐⭐⭐ — CircularEval 和 LLM 选项提取器是有意义的方法创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 评测 21 个 VLM，多维度分析，消融充分
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，图表丰富，动机论述充分
- 价值: ⭐⭐⭐⭐⭐ — 已成为 VLM 评测标配，VLMEvalKit 被广泛采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)
- [\[ECCV 2024\] m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks](m_ampmaposs_a_benchmark_to_evaluate_tool-use_for_multi-step_multi-modal_tasks.md)
- [\[CVPR 2025\] EgoLM: Multi-Modal Language Model of Egocentric Motions](../../CVPR2025/multimodal_vlm/egolm_multi-modal_language_model_of_egocentric_motions.md)
- [\[ECCV 2024\] Elevating All Zero-Shot Sketch-Based Image Retrieval Through Multimodal Prompt Learning](elevating_all_zero-shot_sketch-based_image_retrieval_through_multimodal_prompt_l.md)
- [\[CVPR 2026\] ROSE: Rotate Your Large Language Model to See](../../CVPR2026/multimodal_vlm/rose_rotate_your_large_language_model_to_see.md)

</div>

<!-- RELATED:END -->
