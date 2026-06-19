---
title: >-
  [论文解读] CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models
description: >-
  [ICCV 2025][多模态VLM][类别无关姿态估计] 首次将多模态大语言模型（MLLM）引入类别无关姿态估计（CAPE），仅需查询图像和文本描述即可预测任意类别的关键点位置，无需传统的支持图像和标注，在MP-100基准上超越5-shot SOTA。 类别无关姿态估计（CAPE）旨在泛化到训练集中未见过的物体类别…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "类别无关姿态估计"
  - "MLLM"
  - "无支持集"
  - "关键点检测"
  - "分布建模"
---

# CapeLLM: Support-Free Category-Agnostic Pose Estimation with Multimodal Large Language Models

**会议**: ICCV 2025  
**arXiv**: [2411.06869](https://arxiv.org/abs/2411.06869)  
**代码**: 有（论文提供链接）  
**领域**: 多模态VLM  
**关键词**: 类别无关姿态估计, MLLM, 无支持集, 关键点检测, 分布建模

## 一句话总结

首次将多模态大语言模型（MLLM）引入类别无关姿态估计（CAPE），仅需查询图像和文本描述即可预测任意类别的关键点位置，无需传统的支持图像和标注，在MP-100基准上超越5-shot SOTA。

## 研究背景与动机

类别无关姿态估计（CAPE）旨在泛化到训练集中未见过的物体类别。现有方法（如GraphCAPE、CapeX）依赖"支持集"——即同类别的标注参考图像来建立关键点对应关系。这带来两个核心问题：

**性能不稳定**：同一查询图像搭配不同质量的支持图像会产生不同预测结果

**标注负担重**：新类别出现时需要重新标注支持集

CapeX尝试用关键点名称文本替代支持图像，但仍依赖骨架结构信息，且仅使用简单的关键点名称，未充分利用大语言模型中的丰富语义先验。

本文的核心思路是：**能否完全摆脱支持集，只用图像+详细文本描述来做CAPE？** 答案是肯定的——通过MLLM强大的视觉-语言理解能力，无需任何辅助查询。

## 方法详解

### 整体框架

CapeLLM由三个核心组件构成：（1）DINOv2视觉编码器提取图像特征，（2）线性投影层将视觉token映射到语言空间，（3）LLaMA3.1作为基础语言模型进行推理。输入为一张查询图像和包含关键点名称+详细描述的文本指令，输出为浮点数格式的关键点坐标。

### 关键设计

1. **浮点坐标解码策略 (Floating-Point Decoding)**:
   采用 `[0.abc, 0.def]` 格式直接生成坐标值，每个小数位作为独立token预测。对于标量值 $y$，通过 $K=3$ 个数字token的自回归分解来近似：
    $p(y|\mathbf{x}) \approx \prod_{k=1}^{K} p_{\phi,\theta}(\mathbf{Y}_k | \mathbf{x}, \mathbf{Y}_1, ..., \mathbf{Y}_{k-1})$
   这等价于以 $10^{-K}$ 精度建模截断条件密度，相比高斯参数化有更强的灵活性。可以隐式建模任意概率分布，避免了固定方差高斯分布的局限（如关键点在物体边缘时概率不应扩散到背景区域）。

2. **精心设计的指令 (Instruction Design)**:
   核心创新之一是**详细空间关系描述**替代简单关键点名称。描述包含精确的空间位置和关键点间的相对关系（如"在第五个轮子旁边，在第二个轮子前面"），避免模糊表述。这种设计使mPCK提升约6%。

3. **动态轮次训练 (Dynamic-Round Training)**:

    - **固定轮次**：将每个类别的关键点分成固定大小 $k$ 的组，每组与图像配对形成多轮对话
    - **动态轮次**：每对中关键点数量随机变化，促使模型利用其他关键点信息进行空间推理
   动态轮次训练尤其适配"累积推理"推断策略，使模型能利用已预测关键点的上下文进一步改进后续预测。

4. **累积推理 (Cumulative Reasoning)**:
   推理阶段，将前一个关键点的预测prompt拼接到下一个关键点的输入中，形成累积上下文。配合动态轮次训练，这种策略在PCK@0.2上带来1.2%的提升。

### 损失函数 / 训练策略

- 使用LoRA对视觉编码器和LLM的query/value投影层进行微调，rank=8
- 优化器：AdamW，学习率 $5 \times 10^{-4}$
- 训练12个epoch，3%总步数的warm-up
- 输入图像resize至224×224
- 每条指令默认4轮对话
- 4×RTX-A6000 GPU，梯度累积32步

## 实验关键数据

### 主实验

| 方法 | 设定 | Split1 | Split2 | Split3 | Split4 | Split5 | 平均PCK@0.2 |
|------|------|--------|--------|--------|--------|--------|-------------|
| GraphCAPE (1-shot) | 需要支持集 | 94.63 | 89.79 | 90.30 | 87.81 | 90.07 | 90.52 |
| GraphCAPE (5-shot) | 需要支持集 | 95.81 | 90.78 | 90.94 | 90.42 | 92.27 | 92.04 |
| CapeX | 文本+骨架 | 95.29 | 91.08 | 88.94 | 89.83 | 92.96 | 91.62 |
| **CapeLLM** | **纯文本(无支持集)** | **97.01** | **92.40** | **90.58** | **90.90** | **92.11** | **92.60** |

### 消融实验

| 配置 | PCK@0.05 | PCK@0.2 | mPCK |
|------|---------|---------|------|
| LocLLM风格训练 | 55.15 | 94.85 | 84.00 |
| Fixed-Round训练 | 78.43 | 96.98 | 91.98 |
| Dynamic-Round训练 | 76.55 | 96.05 | 90.92 |
| Dynamic + 累积推理 | — | 97.28 | — |
| 模糊描述 | 69.82 | 91.97 | 85.98 |
| 空间关系描述 | 78.43 | 96.98 | 91.98 |

### 关键发现

- **无支持集超越5-shot**: CapeLLM仅用文本描述就超越了GraphCAPE的5-shot结果（+0.56%），说明MLLM的语义理解能力可完全替代支持集
- **指令质量至关重要**: 空间关系描述 vs 模糊描述，mPCK差距达6%
- **隐式分布建模优势**: 浮点解码天然建模非高斯分布，在边缘处的关键点预测更合理
- **对输入变化鲁棒**: 用GPT-4o改写关键点名称和描述后，性能基本不变
- **遮挡鲁棒性强**: 无需骨架信息也能应对自遮挡、物体遮挡等情况

## 亮点与洞察

- **范式转变**: 从"support-dependent"到"support-free"，完全用MLLM的先验知识替代了传统的视觉支持集
- **浮点解码 = 隐式密度估计**：这个发现非常有趣——通过逐位数字预测坐标，MLLM实际上在建模一个截断的条件密度函数，其灵活性远超固定高斯
- **详细文本描述 > 支持图像**：说明在CAPE中，精确的语义信息比视觉相似度更有价值

## 局限与展望

- MP-100数据集规模有限（每类约200张），更大规模数据集上的效果有待验证
- 浮点精度固定为3位（$10^{-3}$），对需要更高精度的任务可能不够
- 文本描述需要人工设计，自动生成高质量描述是值得探索的方向
- 推理速度比专用模型慢（LLM自回归推理开销大）
- 未探索多实例场景（同张图中多个相同类别物体）

## 相关工作与启发

- 与LocLLM的核心区别在于category-agnostic能力——LocLLM是category-specific的直接应用表现很差
- 累积推理类似Chain-of-Thought，说明关键点间的空间关系推理对MLLM是可学习的
- 未来可考虑结合主动学习机制——模型对低置信度关键点主动请求更多上下文信息

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首次将MLLM用于CAPE，范式创新显著
- **实验充分度**: ⭐⭐⭐⭐⭐ 消融极其详尽（训练策略、描述类型、分辨率、解码策略等）
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，实验设计系统性强
- **价值**: ⭐⭐⭐⭐ 证明了MLLM在精细视觉定位任务中的可行性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AutoComPose: Automatic Generation of Pose Transition Descriptions for Composed Pose Retrieval Using Multimodal LLMs](autocompose_automatic_generation_of_pose_transition_descriptions_for_composed_po.md)
- [\[ICCV 2025\] Bidirectional Likelihood Estimation with Multi-Modal Large Language Models for Text-Video Retrieval](bidirectional_likelihood_estimation_with_multi-modal_large_language_models_for_t.md)
- [\[ICCV 2025\] Visual-Oriented Fine-Grained Knowledge Editing for MultiModal Large Language Models](visual-oriented_fine-grained_knowledge_editing_for_multimodal_large_language_mod.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
