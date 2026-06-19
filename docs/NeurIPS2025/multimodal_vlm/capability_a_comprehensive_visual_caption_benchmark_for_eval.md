---
title: >-
  [论文解读] CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness
description: >-
  [NeurIPS 2025][多模态VLM][视觉描述评测] 提出CAPability，一个涵盖6大视角12个维度的综合视觉描述评测基准，通过人工标注近11K图像/视频的视觉元素（而非句子），同时评估描述的正确性（precision）和全面性（hit），并引入"知道但说不出"（$K\bar{T}$）指标揭示MLLM在QA与caption任务之间的显著能力差距。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "视觉描述评测"
  - "多维度基准"
  - "多模态大模型"
  - "准确性与全面性"
  - "Caption评估"
---

# CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness

**会议**: NeurIPS 2025  
**arXiv**: [2502.14914](https://arxiv.org/abs/2502.14914)  
**代码**: [项目主页](https://capability-bench.github.io)  
**领域**: 多模态VLM  
**关键词**: 视觉描述评测, 多维度基准, 多模态大模型, 准确性与全面性, Caption评估

## 一句话总结

提出CAPability，一个涵盖6大视角12个维度的综合视觉描述评测基准，通过人工标注近11K图像/视频的视觉元素（而非句子），同时评估描述的正确性（precision）和全面性（hit），并引入"知道但说不出"（$K\bar{T}$）指标揭示MLLM在QA与caption任务之间的显著能力差距。

## 研究背景与动机

随着多模态大语言模型（MLLM）的快速发展，传统视觉描述基准（如MS-COCO、MSR-VTT）已严重过时，原因有二：（1）传统基准的ground truth通常是简短句子，无法评估现代MLLM生成的详细描述；（2）传统指标（BLEU、CIDEr等）基于N-gram匹配，对句子风格高度敏感，评估不可靠。

近年来出现的新基准虽有改进但仍有局限。DetailCaps、Dream-1K、VDC等采用"模糊视角"评估——从ground truth描述中提取关键词再比较，容易受人类偏见和LLM累积误差影响。CompreCap采用"物体视角"评估——仅关注物体相关信息，覆盖范围有限，忽略了场景、文本、风格、相机等重要维度。

作者认为，全面的视觉描述评测需要**多视角**评估，且应同时衡量描述的**正确性**和**全面性**。后者在之前的研究中几乎被忽视——大多数基准只评估"说对了多少"，却不关心"说全了多少"。这一洞察催生了CAPability的设计。

## 方法详解

### 整体框架

CAPability的设计借鉴了视觉生成基准（如GenEval、VBench、T2VCompBench）的思路——既然生成任务从多个方面评估生成质量，描述任务也应如此。整体流程为：维度设计 → 数据收集 → MLLM预标注 → 数据平衡 → 人工标注（准确率>97%） → 数据过滤 → 多维度独立评估。

### 关键设计

1. **6大视角12个维度的分类体系**：将视觉描述分解为如下维度，每个维度独立收集约1000个样本并独立评估：

    - **物体相关**（Object-Related）：物体类别、物体颜色、物体数量、空间关系
    - **全局相关**（Global-Related）：场景、风格
    - **文本相关**（Text-Related）：OCR
    - **相机相关**（Camera-Related）：相机角度、相机运动
    - **时序相关**（Temporal-Related）：动作、事件
    - **知识相关**（Knowledge-Related）：角色识别

   其中9个静态维度适用于图像和视频，4个动态维度仅适用于视频。物体数量同时涵盖静态和动态。设计动机：单帧可获取的信息为静态，需要完整视频的信息为动态。

2. **"以一代全"（One Represents All）标注策略**：对于可能包含多个物体/动作的样本，不追求标注所有元素，而是**随机选择一个**进行标注。核心原理是基于大数定律——大量样本的随机选择可以近似覆盖不同粒度的期望分布。为避免人类选择偏见，使用三个SOTA MLLM（GPT-4o、Gemini-1.5-pro、Qwen-VL-Max）列出所有候选元素，再由Qwen2.5-Max合并结果后随机选择一个作为预标注。

3. **三态评估与双指标体系**：将每个样本的描述评判为三种状态，然后计算两个核心指标：

    - **MIS**（Missing）：描述中未提及该维度内容
    - **COR**（Correct）：描述中提及且正确  
    - **INC**（Incorrect）：描述中提及但错误

    $\text{Precision} = \frac{|S(\text{COR})|}{|S(\text{COR}) \cup S(\text{INC})|}$

    $\text{Hit} = \frac{|S(\text{COR})|}{|S(\text{ALL})|}$

   Precision仅衡量正确性（描述了的内容有多准），Hit同时衡量正确性和全面性（所有应描述的内容中说对了多少）。

4. **$K\bar{T}$（知道但说不出）指标**：将标注转换为QA对格式，对比模型在QA和Caption两种任务上的表现差异：

    $K\bar{T} = \frac{|S_{qa}(\text{COR}) \cap [S(\text{INC}) \cup S(\text{MIS})]|}{|S_{qa}(\text{COR})|}$

   该指标衡量"模型在被提问时能答对，但在主动描述时却没有表达"的比例，揭示了MLLM"被动知识"与"主动表达"之间的差距。这是之前工作完全没有量化过的维度。

### 损失函数 / 训练策略

本文是评测基准工作，不涉及模型训练。评估使用GPT-4 Turbo（1106-preview）作为评判器，对每个维度的生成描述进行三态判定（MIS/COR/INC）。对于有特定类别的维度（风格、相机角度、相机运动）和开放式描述维度，分别设计了不同的prompt模板。

## 实验关键数据

### 主实验——闭源及72B模型

| 模型 | Precision Avg | Hit Avg | 最突出能力 |
|------|:---:|:---:|------|
| GPT-4o (0806) | 79.2% | 56.5% | 相机角度Precision 67.0%（领先9.6%） |
| Gemini-1.5-pro | 77.3% | **60.4%** | 物体计数Hit领先10%+，全面性最佳 |
| Gemini-2.0-flash | **79.3%** | 56.2% | Precision并列最高 |
| Qwen2.5VL-72B | 75.9% | 53.4% | 开源最佳Hit，场景和相机运动好 |
| InternVL2.5-78B | 71.2% | 47.0% | - |
| LLaVA-OV-72B | 74.7% | 46.6% | - |

### 维度难度分析

| 维度 | 最佳Precision | 最佳Hit | 整体难度 |
|------|:---:|:---:|:---:|
| 场景 | 97.0% | 86.9% | 简单 |
| OCR | 95.9% | 88.8% | 简单 |
| 风格 | 91.4% | 91.4% | 简单 |
| 物体类别 | 89.8% | 86.3% | 较易 |
| 物体颜色 | 90.4% | 67.7% | 中等 |
| 物体数量 | 78.6% | 40.0% | 难 |
| 相机角度 | 67.0% | 67.0% | 难 |
| 动作 | 56.8% | 51.4% | 难 |
| 相机运动 | 35.4% | 35.2% | 很难 |
| 角色识别 | 90.9% | 37.9% | Precision高但Hit低 |

### 关键发现

- **全面性差距巨大**：所有模型从Precision到Hit均显著下降（平均20%+），表明模型倾向于只描述有把握的内容而牺牲全面性
- **模型策略分化**：GPT-4o偏保守（高Precision中Hit，"宁可不说也不说错"），Gemini-1.5-pro偏激进（多说多覆盖）
- **共同薄弱点**：物体数量、相机角度/运动、角色识别、动作是所有模型的瓶颈维度
- **$K\bar{T}$发现**：所有模型都存在显著的$K\bar{T}$ gap，表明MLLM的主动描述能力显著弱于被动问答能力

## 亮点与洞察

1. **全面性评估的开创性**：首次在多维度框架下系统评估描述的全面性，揭示了被忽视的"会做不会说"问题
2. **$K\bar{T}$指标的启发性**：定量展示了MLLM在主动描述和被动回答之间的能力差距，为训练策略（如caption-aware训练）提供了方向
3. **"以一代全"策略的统计学巧思**：利用大数定律解决多粒度标注的可行性问题
4. **跨图像-视频的统一评估框架**：12个维度覆盖静态和动态，提供了首个统一的视觉描述评估体系

## 局限与展望

- 每个维度的数据独立收集，未评估模型在单个样本上同时覆盖多个维度的能力
- 依赖GPT-4 Turbo作为评判器，可能引入评判偏差
- 动态维度仅使用视频、静态维度仅使用图像，未探索交叉评估
- 每个维度约1000样本，对某些细分子类别可能不够充分

## 相关工作与启发

- **与CompreCap的对比**：CompreCap只评估物体相关信息，CAPability扩展到6大视角并同时评估正确性和全面性
- **来自视觉生成基准的启发**：维度设计借鉴GenEval、VBench等——描述和生成是逆任务，应该有对称的评估维度
- **$K\bar{T}$作为RLHF信号的潜力**：可用来识别模型"知道但不说"的知识，指导后续描述能力的定向训练

## 评分

- **新颖性**: ⭐⭐⭐⭐ 多视角+双指标+$K\bar{T}$的评测框架设计新颖，但核心思路（分维度评测）并非全新
- **实验充分度**: ⭐⭐⭐⭐⭐ 测试了大量闭源和开源模型（7B到72B多个规模），维度分析详尽
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，图表丰富，动机阐述充分
- **价值**: ⭐⭐⭐⭐⭐ 填补了视觉描述全面性评估的空白，发现了重要的能力差距，对社区有重要指导价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)
- [\[ACL 2025\] Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration](../../ACL2025/multimodal_vlm/evaluating_visual_and_cultural_interpretation_the_k-viscuit_benchmark_with_human.md)
- [\[CVPR 2025\] Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning](../../CVPR2025/multimodal_vlm/mosaic_of_modalities_a_comprehensive_benchmark_for_multimodal_graph_learning.md)
- [\[NeurIPS 2025\] Unifying Vision-Language Latents for Zero-Label Image Caption Enhancement](unifying_vision-language_latents_for_zero-label_image_caption_enhancement.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)

</div>

<!-- RELATED:END -->
