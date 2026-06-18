---
title: >-
  [论文解读] From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning
description: >-
  [ICCV 2025][多模态VLM][multi-image reasoning] 提出 MIR 基准，包含 22,257 个多图像交错推理问答对及五阶段推理步骤，并设计渐进式课程学习策略，从"简单到困难"逐步提升 MLLM 的多图像交错推理能力。 多图像交错推理（Multi-image Interleaved Reaso…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "multi-image reasoning"
  - "interleaved data"
  - "benchmark"
  - "curriculum learning"
  - "MLLM"
---

# From Easy to Hard: The MIR Benchmark for Progressive Interleaved Multi-Image Reasoning

**会议**: ICCV 2025  
**arXiv**: [2509.17040](https://arxiv.org/abs/2509.17040)  
**代码**: [https://github.com/Shelly-coder239/MIRBench](https://github.com/Shelly-coder239/MIRBench)  
**领域**: 多模态VLM  
**关键词**: multi-image reasoning, interleaved data, benchmark, curriculum learning, MLLM

## 一句话总结

提出 MIR 基准，包含 22,257 个多图像交错推理问答对及五阶段推理步骤，并设计渐进式课程学习策略，从"简单到困难"逐步提升 MLLM 的多图像交错推理能力。

## 研究背景与动机

多图像交错推理（Multi-image Interleaved Reasoning）要求模型联合理解多幅图像及其关联的文本上下文，是超越单图像和非交错多图像任务的独特挑战。现有多图像基准存在三个关键问题：

**多图像推理评估不足**：现有基准主要关注基础多图像问答，未整合交错文本信息，忽略了 Text2Region 和 Region2Region 等复杂推理范式

**缺乏逐步推理**：仅关注最终答案准确率，忽略中间推理步骤

**数据泄漏风险**：大多数数据集源自已被 MLLM 预训练使用的公开数据集

论文通过构建包含 138,277 张图像的大规模基准和渐进式训练方法来解决上述问题。

## 方法详解

### 整体框架

MIR 基准 + 阶段式课程学习方法。基准包含三大类（Sequential/Spatial/Analytical）、12 个细分任务。每个实例配有 5 步结构化推理步骤：Summary → Caption → Text2Region → Region2Region → Conclusion。训练方法先用简单样本建立基础理解，再渐进式引入困难样本。

### 关键设计

1. **自适应难度过滤器 (Adaptive Difficulty Filter)**：将每个问题输入 Qwen-VL2.5 推理 10 次，正确率 ≥70% 的归为简单样本，<70% 的归为困难样本。这种基于概率的分类方法确保了客观性和可靠性。

2. **阶段式课程学习策略 (Stage-wise Curriculum Learning)**：分两大阶段——首先用简单样本微调建立基础能力；然后对困难样本进行多阶段渐进训练。每个阶段从困难样本池中采样 40%，并按以下方式逐步减少推理步骤的引导：

    - Stage 1: Q + 全部推理步骤 → A（模型仅需输出答案）
    - Stage 2: Q + 前4步 → A + 第5步（模型需生成结论）
    - Stage 3-5: 逐步将更多推理步骤从输入移到输出端，最终模型需从原始问题自主生成完整推理过程并得出答案

3. **多策略数据构建流程**：针对不同任务类型采用不同策略——Spatial 任务使用 3D 几何平台生成多视角投影；Sequential 任务通过视频帧提取和位置变化计算；Analytical 任务结合定向网络爬取和文生图合成。Q&A 生成结合预定义模板和 LLM，推理步骤通过人工标注关键步骤 + LLM 辅助 + 多轮人工校验。

### 损失函数 / 训练策略

- 基于各 MLLM 标准微调策略（如 Qwen2-VL、LLaVA-OneVision 等）
- 简单样本先训练建立基础，困难样本分阶段渐进训练
- 每阶段采样 40% 困难数据，结构化调整输入输出分配
- 实现从"学习推理"到"自主推理"的范式转换

## 实验关键数据

### 主实验

| 方法 | Spatial | Sequential | Analytical | Avg (in-domain) | MIRBENCH | BLINK | MUIRBench |
|------|---------|------------|------------|-----------------|----------|-------|-----------|
| Qwen2-VL | 29.00% | 56.60% | 45.96% | 40.44% | 50.04% | 50.34% | 45.67% |
| Qwen2-VL (Tuning) | 37.31% | 58.33% | 49.59% | 45.15% | 53.45% | 52.19% | 45.51% |
| Qwen2-VL (Ours) | **40.42%** | **59.65%** | **52.93%** | **51.76%** | **54.47%** | **56.61%** | **46.67%** |
| LLaVA-OneVision | 42.31% | 36.53% | 38.12% | 39.31% | 45.91% | 49.00% | 40.61% |
| LLaVA-OneVision (Tuning) | 45.17% | 39.29% | 41.20% | 42.36% | 47.04% | 50.34% | 42.22% |
| LLaVA-OneVision (Ours) | **50.34%** | **41.27%** | **45.99%** | **47.60%** | **49.23%** | **52.93%** | **45.34%** |

### 消融实验

Qwen2-VL 在 MIR 上的各阶段表现：

| 阶段 | Spatial | Sequential | Analytical | Average |
|------|---------|------------|------------|---------|
| Stage 1（全步骤引导） | 35.19% | 32.22% | 44.85% | 38.01% |
| Stage 2 | 34.60% | 35.55% | 38.97% | 36.73% |
| Stage 3 | 40.57% | 33.05% | 43.52% | 41.03% |
| Stage 4 | 46.29% | 34.44% | 48.52% | 45.97% |
| Stage 5（自主推理） | **50.96%** | **36.66%** | **58.82%** | **51.76%** |

### 关键发现

- 提出的课程学习方法在 domain 内提升 7%+，在 domain 外提升 1%-5%，远超普通微调的 2%
- Stage 2 出现性能下降（36.73%），因为任务复杂度突然增加；Stage 3-5 持续恢复并大幅提升
- 指令遵循能力对渐进学习至关重要——指令遵循能力越强，学习效果越好
- Sequential 任务提升最小，因为从视频提取的静态帧难以完整表示时序信息
- 模型经训练后能维持结构化推理流程：Summary → Caption → Text2Region → Region2Region → Conclusion

## 亮点与洞察

- **推理步骤设计精妙**：5 步结构化推理（Summary/Caption/Text2Region/Region2Region/Conclusion）清晰对应了多图像交错理解的核心认知过程
- **课程学习的"先易后难"策略**在多模态推理场景中效果显著，比直接微调有明显优势
- **数据构建的严谨性**：多源采集（自拍/短视频平台/教育资源）+ 半自动标注 + 多轮人工校验，有效避免数据泄漏
- MIR 数据集规模大（22,257 QA 对，138,277 张图像），平均每实例 6 张图，推理步骤约 3,970 字符

## 局限与展望

- 目前仅支持 MCQ 格式评估，开放式问答未涉及
- Sequential 任务的图像来自视频帧提取，可能丢失关键时序信息
- Stage 2 的性能下降暗示阶段过渡策略可能需要更精细的设计（如平滑过渡或更多数据）
- 对闭源模型（如 GPT-4o、Claude）未进行课程学习训练评估
- 推理步骤的生成依赖 LLM，可能引入偏差

## 相关工作与启发

- 与 MUIRBench、MMIU 等多图像基准相比，MIR 首次系统引入交错推理步骤
- 课程学习的"渐进减少引导"思路可推广到其他需要步骤化推理的任务
- 与 Chain-of-Thought 不同，本方法通过训练让模型内化推理过程，而非仅在推理时提示

## 评分

- **新颖性**: ⭐⭐⭐⭐ 交错多图像推理 + 渐进课程学习的组合新颖实用
- **实验充分度**: ⭐⭐⭐⭐ 5 个 MLLM × 3 种训练策略 × 多个基准，消融充分
- **写作质量**: ⭐⭐⭐⭐ 结构完整，图表清晰，案例分析直观
- **价值**: ⭐⭐⭐⭐ 为多图像交错推理提供了重要基准和有效训练策略

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](../../CVPR2025/multimodal_vlm/opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)
- [\[ICCV 2025\] ReasonVQA: A Multi-hop Reasoning Benchmark with Structural Knowledge for Visual Question Answering](reasonvqa_a_multi-hop_reasoning_benchmark_with_structural_knowledge_for_visual_q.md)
- [\[ACL 2025\] FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation](../../ACL2025/multimodal_vlm/finmme_benchmark_dataset_for_financial_multi-modal_reasoning_evaluation.md)
- [\[ACL 2025\] Progressive Multimodal Reasoning via Active Retrieval](../../ACL2025/multimodal_vlm/progressive_multimodal_reasoning_via_active_retrieval.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)

</div>

<!-- RELATED:END -->
