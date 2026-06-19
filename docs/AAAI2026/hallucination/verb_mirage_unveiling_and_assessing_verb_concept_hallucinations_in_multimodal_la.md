---
title: >-
  [论文解读] Verb Mirage: Unveiling and Assessing Verb Concept Hallucinations in Multimodal Large Language Models
description: >-
  [AAAI 2026][幻觉检测][动词幻觉] 首次系统研究多模态大语言模型（MLLM）中的动词概念幻觉问题，构建了多维度基准测试，发现现有幻觉缓解方法对动词幻觉无效，并提出基于丰富动词知识微调的基线方法，显著缓解动词幻觉。 多模态大语言模型（MLLMs）在OCR、VQA、图像描述等任务上取得了显著进展…
tags:
  - "AAAI 2026"
  - "幻觉检测"
  - "动词幻觉"
  - "MLLM"
  - "幻觉评测"
  - "动作理解"
  - "细粒度评估"
---

# Verb Mirage: Unveiling and Assessing Verb Concept Hallucinations in Multimodal Large Language Models

**会议**: AAAI 2026  
**arXiv**: [2412.04939](https://arxiv.org/abs/2412.04939)  
**代码**: [github](https://github.com/davidwang200099/Verb_Mirage)  
**领域**: 幻觉检测  
**关键词**: 动词幻觉, MLLM, 幻觉评测, 动作理解, 细粒度评估

## 一句话总结
首次系统研究多模态大语言模型（MLLM）中的动词概念幻觉问题，构建了多维度基准测试，发现现有幻觉缓解方法对动词幻觉无效，并提出基于丰富动词知识微调的基线方法，显著缓解动词幻觉。

## 研究背景与动机

多模态大语言模型（MLLMs）在OCR、VQA、图像描述等任务上取得了显著进展，但**幻觉**问题一直是制约其可靠性的核心瓶颈。现有的幻觉研究和缓解方法**几乎全部集中在物体/名词概念的幻觉**上，例如POPE基准测试物体是否存在、CHAIR评估描述中的物体幻觉等。

然而，**动词概念**对于理解人类行为至关重要——我们不仅需要知道图中有什么物体，更需要理解这些物体之间发生了什么动作。但动词幻觉长期被忽视，主要原因包括：

**数据集偏差**：常用MLLM预训练数据中，名词数量是动词的4-10倍（如图4(a)所示），导致模型对名词理解远好于动词

**评测空白**：没有专门针对动词幻觉的评测基准

**直觉误区**：人们认为解决了物体幻觉，动词幻觉也就迎刃而解——但本文证明这是错误的

本文的核心切入点：**动词幻觉与物体幻觉是本质不同的问题**。物体幻觉的缓解方法对动词幻觉无效，甚至会加重幻觉。这一发现揭示了MLLM在语义理解上的深层缺陷。

## 方法详解

### 整体框架

本文的工作主要分为三个部分：(1) 构建动词幻觉评测基准；(2) 从多维度探测和分析动词幻觉现象；(3) 提出基于动词知识微调的基线缓解方法。

### 关键设计

#### 1. **多维度动词幻觉基准构建**

基于HICO和CharadesEgo数据集，构建了首个动词幻觉评测基准，无需额外人工标注。基准涵盖两种问题格式：

- **是/否（YN）问题**：如"图中是否有人在拿着杯子？"
- **多选（MC）问题**：给定一个正确动词和三个干扰动词，采用circular evaluation

核心设计思路是：**改变动词而保持物体不变**，从而隔离动词幻觉与物体幻觉。例如，如果图中有人拿着杯子，可以问"有人在拿杯子吗？有人在洗杯子吗？"

#### 2. **多角度探测设计**

从三个维度系统探测动词幻觉：

**(a) 查询条件探测**：
- **问题格式**：MQ vs YN，发现模型在MC上表现更好但仍有大量幻觉
- **物体关联**：对比"有人在拿杯子吗"vs"有人在拿东西吗"，发现MLLMs严重依赖物体参考来理解动词

**(b) 图像条件探测**：
- **图像质量**：添加椒盐噪声（75%像素受影响），发现视觉失真对动词理解的影响**远大于**对物体理解的影响（Cohen's Kappa一致性差异显著）
- **视角差异**：使用CharadesEgo对比第一人称（ego）与第三人称（exo）视角，发现MLLM在第一人称视角下动词理解能力显著下降

**(c) 语义条件探测**：
- **稀有vs常见动词**：模型倾向于拒绝存在的稀有动词、接受不存在的常见动词
- **内容模糊性**：在拥挤、遮挡、人物-物体尺寸不平衡的场景中，动词幻觉更严重

#### 3. **模型行为深度分析**

以LLaVA V1.5为例，从视觉-语言交互和token不确定性两个角度分析动词幻觉的根源：

- **关键图像区域注意力**：幻觉时模型对关键区域注意力较低，但差距不大——即使注意力正确，也无法保证理解动词语义
- **视觉token注意力**：与物体幻觉不同，更多关注视觉token**并不能**排除动词幻觉（这解释了OPERA方法失效的原因）
- **token不确定性**：幻觉答案通常以低概率给出，且模型倾向于高置信度回答"Yes"
- **mAP vs Acc分析**：LLaVA V1.5虽然准确率低（52.16），但mAP（68.41）超过HICO微调的CLIP（60.45），说明**动词幻觉的一个来源是token校准错误**而非完全不理解动词

### 损失函数 / 训练策略

基线缓解方法采用LoRA微调LLaVA V1.5，使用Pangea数据集中的60K样本构建指令微调数据集。Pangea组织了异构动作数据集，建立了从动作标签到VerbNet中290个抽象动词节点的映射，覆盖280/290个动词节点，涵盖广泛的动词语义。

## 实验关键数据

### 主实验

**各模型在YN和MC任务上的动词幻觉表现**：

| 模型 | YN+obj acc | YN+obj prec | YN+obj recall | MC+obj acc | MC verb-only acc |
|------|-----------|-------------|---------------|-----------|-----------------|
| Qwen2-VL-7B | 75.51 | 58.37 | 93.75 | 71.47 | 65.31 |
| MiniCPM-Llama3-V2.5 | 80.91 | 66.83 | 85.41 | 66.39 | 60.77 |
| LLaVA V1.5 | 52.16 | 40.99 | 97.35 | 57.37 | 51.00 |
| Molmo-7B-D | 59.16 | 44.91 | 96.63 | 60.64 | 56.78 |

**关键发现**：所有模型recall极高（85-97%）但precision极低（40-67%），说明模型倾向于无论动词是否存在都回答"Yes"。

**缓解方法对比**（以LLaVA V1.5为基准）：

| 方法 | YN+obj acc | YN+obj F1 | MC+obj acc | MC verb-only acc |
|------|-----------|----------|-----------|-----------------|
| LLaVA V1.5 | 52.16 | 57.69 | 57.37 | 51.00 |
| OPERA | 42.46 | 53.69 | 57.28 | 51.13 |
| VCD | 52.38 | 58.04 | 54.26 | 48.94 |
| Haloquest | 70.57 | 64.89 | 55.20 | 47.45 |
| **Ours (Pangea FT)** | **78.48** | **68.13** | **61.73** | **60.79** |

### 消融实验

| 配置 | YN+obj acc | MC verb-only acc | 说明 |
|------|-----------|-----------------|------|
| OPERA | 42.46 | 51.13 | 惩罚summary token注意力，反而恶化 |
| VCD | 52.38 | 48.94 | 语言先验无法被轻易移除（18.6K/20K样本KL=0） |
| Nullu | 51.99 | 53.17 | 模型层未形成可靠的动词真值/幻觉区分 |
| REVERIE | 40.67 | 41.32 | 训练集缺乏动词知识 |
| Ours | 78.48 | 60.79 | 丰富动词知识微调有效 |

**图像质量影响消融**：

| 模型 | YN无噪声 acc | YN有噪声 acc | 错误一致性(Cohen's κ) |
|------|-------------|-------------|---------------------|
| MiniCPM | 79.14 | 67.40 | 26.12（差） |
| Qwen-VL-Chat | 79.24 | 66.64 | 38.47（中） |
| LLaVA V1.5 | 59.16 | 51.29 | 73.85（好但基础差） |

### 关键发现

1. **动词幻觉普遍且严重**：所有SOTA MLLM在动词理解上都表现不佳，即使它们在物体幻觉基准（POPE）上得分很高
2. **动词理解严重依赖物体参考**：去掉物体后MC准确率大幅下降
3. **现有物体幻觉缓解方法对动词幻觉无效**：OPERA、VCD、Nullu等方法全部失败
4. **模型共享相似偏差**：集成三个模型也无法显著改善
5. **动词幻觉源于token校准错误**而非完全不理解：mAP高但accuracy低
6. **视觉失真对动词理解的影响远大于物体理解**

## 亮点与洞察

1. **问题定义有开创性**：首次将动词幻觉从物体幻觉中独立出来研究，填补了重要空白
2. **评测维度全面**：从查询条件、图像条件、语义条件、模型行为四个维度系统分析，实验设计严谨
3. **深层洞察有价值**：发现"更多视觉注意力≠更少动词幻觉"颠覆了OPERA等方法的核心假设
4. **mAP与accuracy的对比分析**揭示了幻觉的真正来源是校准问题，为未来研究指明了方向
5. **Pangea微调实验**证明丰富动词知识可以缓解幻觉，且不严重影响其他能力

## 局限与展望

1. 提出的缓解方法是baseline水平，性能远未令人满意（MC verb-only acc仅60.79%）
2. 是否存在有效的training-free动词幻觉方法仍是开放问题
3. 评测仅限于静态图像场景，未涉及视频中的动词理解
4. 仅测试了开源7B级别模型，未充分测试更大规模模型
5. 微调数据仅60K样本，可能通过更大规模、更多样化的动词数据获得更好效果

## 相关工作与启发

- **POPE** (Li et al., 2023)：物体幻觉评测先驱，本文将其扩展到动词概念
- **OPERA** (Huang et al., 2024)：通过惩罚summary token注意力缓解幻觉，本文证明其对动词无效
- **VCD** (Leng et al., 2024)：对比解码方法，发现语言先验在动词理解中根深蒂固
- **Pangea** (Li et al., 2024)：统一异构动作数据集，提供丰富动词知识
- 启发：未来可以将动词幻觉评测纳入MLLM的标准评估体系；动词理解需要专门的数据和训练策略

## 评分
- 新颖性: ⭐⭐⭐⭐⭐（首次定义并系统研究动词幻觉问题）
- 实验充分度: ⭐⭐⭐⭐⭐（多维度、多模型、多条件的全面评测）
- 写作质量: ⭐⭐⭐⭐（逻辑清晰，但篇幅较长）
- 价值: ⭐⭐⭐⭐⭐（揭示了MLLM的重要盲区，对社区有重要警示作用）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Beyond Hallucinations: A Composite Score for Measuring Reliability in Open-Source Large Language Models](beyond_hallucinations_a_composite_score_for_measuring_reliability_in_open-source.md)
- [\[CVPR 2026\] MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models](../../CVPR2026/hallucination/mad_modality-adaptive_decoding_for_mitigating_cross-modal_hallucinations_in_mult.md)
- [\[CVPR 2025\] ODE: Open-Set Evaluation of Hallucinations in Multimodal Large Language Models](../../CVPR2025/hallucination/ode_open-set_evaluation_of_hallucinations_in_multimodal_large_language_models.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/hallucination/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[ICML 2026\] Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](../../ICML2026/hallucination/instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat.md)

</div>

<!-- RELATED:END -->
