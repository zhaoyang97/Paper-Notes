---
title: >-
  [论文解读] MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts
description: >-
  [CVPR 2025][多模态VLM][多图数学推理] 本文提出 MV-MATH 基准，包含 2,009 道高质量多图数学题（来自真实 K-12 场景），系统评估了 25 个多模态大模型在多图数学推理场景下的能力，发现所有模型远低于人类水平（最佳 Claude 仅 33.9%），揭示了多图数学推理仍是 MLLM 的重大挑战。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "多图数学推理"
  - "多模态基准"
  - "K-12数学"
  - "多图理解"
  - "MLLM评估"
---

# MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts

**会议**: CVPR 2025  
**arXiv**: [2502.20808](https://arxiv.org/abs/2502.20808)  
**代码**: [https://eternal8080.github.io/MV-MATH.github.io/](https://eternal8080.github.io/MV-MATH.github.io/)  
**领域**: 多模态VLM  
**关键词**: 多图数学推理, 多模态基准, K-12数学, 多图理解, MLLM评估

## 一句话总结

本文提出 MV-MATH 基准，包含 2,009 道高质量多图数学题（来自真实 K-12 场景），系统评估了 25 个多模态大模型在多图数学推理场景下的能力，发现所有模型远低于人类水平（最佳 Claude 仅 33.9%），揭示了多图数学推理仍是 MLLM 的重大挑战。

## 研究背景与动机

**领域现状**：多模态大模型（MLLM）在数学推理领域取得了显著进展，MathVista 等基准上最好的模型甚至已超过人类表现。然而，现有的多模态数学基准（MathVista、MathVision、MathVerse 等）几乎都局限于单图场景——每道题只包含一张图片。

**现有痛点**：单图设置与真实数学应用场景严重脱节。在实际 K-12 教学中，学生经常需要同时理解多张图表、坐标系、几何图形之间的关系来解题。虽然已有 MathVerse-mv（788 题）和 CMM-Math（765 多图样本）尝试填补空白，但它们要么通过人工改写单图题生成多图题（引入分布偏差），要么包含低质量图片，且都缺乏细粒度分类和多样化题型。

**核心矛盾**：现有多图数学数据集在数量和多样性上都严重不足，无法全面评估 MLLM 在多图情境下的数学推理能力。MathVerse-mv 的变异系数（CV）仅 0.19，而真实场景的题目长度分布远比这丰富。

**本文目标** (1) 构建大规模、高质量、来自真实场景的多图数学基准；(2) 系统评估 MLLM 在多图数学推理上的表现；(3) 深入分析模型的错误模式和性能瓶颈。

**切入角度**：作者直接从超过 30 万道真实 K-12 数学题中筛选，经过三阶段过滤和交叉验证，保证每道题都是真实的多图题而非人工拼接。

**核心 idea**：用真实 K-12 场景构建大规模多图数学基准 MV-MATH，系统揭示 MLLM 在多图推理上的巨大不足。

## 方法详解

### 整体框架

MV-MATH 的构建流程包括：数据收集 → 三阶段过滤 → 数据标注 → 基准评估。输入是 30 万+ 原始数学题 PDF，输出是 2,009 道经过严格筛选和标注的多图数学题，涵盖 11 个学科、3 个难度等级。

### 关键设计

1. **三阶段数据过滤流程**:

    - 功能：从 30 万原始题中筛选出高质量多图题
    - 核心思路：第一阶段验证文本与图片的对齐（Mathpix OCR 经常出错），从 49,538 道多图题保留 35,562 道；第二阶段检查文本字段缺失和语义错误，分类为选择题和填空题；第三阶段人工过滤低质量图片（模糊、含文字等），最终得到 1,109 道选择题 + 900 道填空题。每一步由至少两名研究生交叉验证。
    - 设计动机：自动化 OCR 工具的错误率很高，必须通过多阶段过滤确保数据质量

2. **图片关联性分类（MD/ID）**:

    - 功能：将题目分为"相互依赖型"（MD）和"独立型"（ID）两个子集
    - 核心思路：MD 型题目中图片之间存在关联，理解一张图需要参考另一张（如同一几何图形的不同视角）；ID 型题目中图片相互独立。分类通过 GPT-4o、Claude-3.5-Sonnet、Qwen-VL-Max 三模型投票，再人工校验。
    - 设计动机：区分图片关联性可以更深入分析模型在需要跨图推理 vs 独立推理时的表现差异

3. **多维度难度和学科标注**:

    - 功能：提供细粒度的题目分类
    - 核心思路：难度通过题目长度（权重 0.4）和解析长度（权重 0.6）的加权组合将题目分为 Easy/Medium/Hard 三档；学科通过三模型投票分为 11 个类别（解析几何、代数、度量几何、组合等）
    - 设计动机：细粒度标注让研究者可以精确定位模型的薄弱环节

### 损失函数 / 训练策略

本文是评估基准而非训练方法，不涉及损失函数设计。评估采用多种配置：原始提示、CoT 提示、CoT + 2-shot 等。

## 实验关键数据

### 主实验

| 模型 | Overall | Easy | Medium | Hard |
|------|---------|------|--------|------|
| Claude-3.5-sonnet | **33.9%** | 35.7 | 37.5 | 26.6 |
| GPT-4o | 32.1% | 40.3 | 32.7 | 22.9 |
| LLaVA-OV-72B | 26.2% | 34.6 | 26.0 | 19.2 |
| Qwen2VL-7B | 16.5% | 18.8 | 17.1 | 13.9 |
| Human | ~60%+ | - | - | - |

### 消融实验（CoT 策略对比）

| 模型 | Original | CoT | CoT+2-shot |
|------|----------|-----|------------|
| Claude-3.5 | 29.2 | 32.6 (+3.4) | **33.9** (+1.3) |
| GPT-4o | **31.8** | 30.9 (-0.9) | 32.1 (+1.2) |
| Gemini-1.5 | **29.8** | 28.3 (-1.5) | 29.1 (+0.8) |
| LLaVA-OV-72B | **27.3** | 26.7 (-0.6) | 26.2 (-0.5) |

### 关键发现

- CoT 提示对 Claude 有明显提升（+3.4），但对 GPT-4o、Gemini 等反而降低性能，说明 CoT 在多图数学任务中并不总是有效
- 所有模型在 Hard 难度上表现急剧下降，最好的 Claude 也仅 26.6%，表明多步推理仍是核心挑战
- 开源模型 LLaVA-OneVision-72B（26.2%）表现不俗，超过了 GPT-4V（24.5%）
- 在图片依赖型（MD）题目上，模型表现明显低于图片独立型（ID），说明跨图关联推理是主要瓶颈
- 顺序输入多图优于合并输入，表明模型对图片顺序信息的利用很重要

## 亮点与洞察

- **真实数据 vs 人工改写**：从 30 万真实 K-12 题中筛选而非改写已有数据集，避免了 MathVerse-mv 那种人工拼接导致的分布偏差，这种"大池筛选"的策略在基准构建中很实用
- **CV 指标衡量分布多样性**：用变异系数（CV=σ/μ）量化题目长度分布的丰富程度（MV-MATH 0.74 vs MathVerse-mv 0.19），简洁有效的指标选择
- **MD/ID 分类揭示跨图推理瓶颈**：通过区分相互依赖型和独立型图片，首次定量证明了跨图关联推理是当前模型的核心弱点

## 局限与展望

- 数据全部来自中国 K-12 教育体系，可能存在文化/教育体系偏差，不一定能反映其他国家的数学教育场景
- 难度定义依赖于题目/解析长度的加权，这种方式比较粗糙，可能无法准确反映认知难度
- 仅评估了 2024 年之前的模型（包括 Claude-3.5），未涵盖 GPT-4o 后续版本和更新的开源模型
- 缺少对模型内部表征的分析——知道模型做错了，但不清楚错在哪一步（视觉理解还是数学推理）

## 相关工作与启发

- **vs MathVerse-mv**: MathVerse-mv 通过改写单图题生成多图题，仅 788 题且只有选择题，CV=0.19；MV-MATH 从真实场景筛选 2,009 题，包含多题型，CV=0.74，数据更真实多样
- **vs CMM-Math**: CMM-Math 聚焦中文场景，部分图片质量不佳；MV-MATH 提供英文版本并增加了图片关联性标注和更细粒度的学科分类
- **vs MathVista/MathVision**: 这些基准限于单图场景，MV-MATH 首次在大规模上系统评估多图数学推理

## 评分

- 新颖性: ⭐⭐⭐⭐ 多图数学推理基准的空白较大，填补有价值但方法上偏数据工程
- 实验充分度: ⭐⭐⭐⭐⭐ 评估了 25 个模型，包含多种配置和深入的错误分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，统计详实
- 价值: ⭐⭐⭐⭐ 对社区了解多图推理瓶颈有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] We-Math: Does Your Large Multimodal Model Achieve Human-like Mathematical Reasoning?](../../ACL2025/multimodal_vlm/wemath_knowledge_reasoning.md)
- [\[ACL 2025\] Can Vision-Language Models Evaluate Handwritten Math?](../../ACL2025/multimodal_vlm/can_vision-language_models_evaluate_handwritten_math.md)
- [\[ECCV 2024\] MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](../../ECCV2024/multimodal_vlm/mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)
- [\[CVPR 2025\] Reasoning over Video: Evaluating How MLLMs Extract, Integrate, and Reconstruct Spatiotemporal Evidence](reasoning_over_video_evaluating_how_mllms_extract_integrate_and_reconstruct_spat.md)
- [\[CVPR 2025\] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)

</div>

<!-- RELATED:END -->
