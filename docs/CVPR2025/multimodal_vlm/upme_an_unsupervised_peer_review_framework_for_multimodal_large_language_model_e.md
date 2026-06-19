---
title: >-
  [论文解读] UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation
description: >-
  [CVPR 2025][多模态VLM][MLLM评估] 提出UPME框架，通过**无监督同行评审机制**、**视觉-语言评分系统**和**动态权重优化**，仅使用图像数据就能让多个MLLM互相出题评审，在MMStar上与人工评估的Pearson相关性达0.944，有效缓解了MLLM评估对人工标注的依赖和评审偏差问题。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "MLLM评估"
  - "无监督评估"
  - "同行评审"
  - "视觉语言评分"
  - "动态权重优化"
---

# UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation

**会议**: CVPR 2025  
**arXiv**: [2503.14941](https://arxiv.org/abs/2503.14941)  
**代码**: 未公开  
**领域**: 多模态VLM  
**关键词**: MLLM评估, 无监督评估, 同行评审, 视觉语言评分, 动态权重优化

## 一句话总结
提出UPME框架，通过**无监督同行评审机制**、**视觉-语言评分系统**和**动态权重优化**，仅使用图像数据就能让多个MLLM互相出题评审，在MMStar上与人工评估的Pearson相关性达0.944，有效缓解了MLLM评估对人工标注的依赖和评审偏差问题。

## 研究背景与动机
MLLM评估领域面临两大核心挑战。第一，传统VQA基准方法（如MMMU、MMStar）依赖大量人工设计问答对，工作量巨大且评估范围受限。第二，MLLM-as-a-Judge方法虽然减轻了人工负担，但引入了**冗余偏差**（偏好更长回答）和**自我偏好偏差**（偏好自己的输出），导致评估偏离视觉内容的真实理解。核心矛盾在于：**如何在完全无人工标注的情况下实现与人工评估高度一致的客观MLLM评价**？本文的切入角度是借鉴学术界同行评审的思想——让模型自己出题、互相评分、通过置信权重迭代优化消除弱模型偏差。

## 方法详解

### 整体框架
UPME包含三个核心模块：(1) **同行评审机制**——每轮迭代从MLLM池中随机抽取2个候选模型和1个评审模型，评审模型对图像生成问题并评估候选模型的回答；(2) **视觉-语言评分系统**——从回答正确性、视觉理解与推理、图文相关性三个维度综合评分；(3) **动态权重优化**——每个模型初始化置信权重，通过MSE损失迭代优化权重与估计分数的一致性。

### 关键设计
1. **同行评审机制（Peer Review Mechanism）**:
    - 功能：实现完全无监督的多模型互评，消除对人工标注的依赖
    - 核心思路：给定图像集 $\mathcal{I}$ 和模型池 $\mathcal{M}$，每轮随机选取评审模型 $M_r$ 对图像 $I_i$ 生成问题 $Q_i^r = M_r(I_i)$，两个候选模型 $M_j, M_k$ 分别回答，评审模型根据评分系统 $S_{VL}$ 对两者进行成对比较评分
    - 设计动机：成对比较评估比独立评分更准确（已有研究支持），且随机轮换角色使每个模型既是候选人又是评审者，通过大量轮次收敛到稳定评价

2. **视觉-语言评分系统（Vision-Language Scoring System）**:
    - 功能：构建多维度评分标准，弥补纯文本评审对视觉内容的忽视
    - 核心思路：最终评分为三项加权和 $S_{VL} = \gamma_1 S_{Correct} + \gamma_2 S_{Visual} + \gamma_3 S_{Clip}$，其中 $S_{Correct}$ 基于成对比较的胜负（1/0.5/0）评估回答正确性；$S_{Visual}$ 通过函数 $\Gamma$ 从描述、推理、定位、关系四个视觉维度综合评估；$S_{Clip}$ 使用CLIP模型计算图文对齐得分
    - 设计动机：引入CLIP分数是关键创新——它作为独立于评审模型的客观指标，有效缓解了冗余偏差（长回答未必CLIP分更高）和自我偏好偏差（CLIP不会偏袒任何模型）

3. **动态权重优化（Dynamic Weight Optimization）**:
    - 功能：通过迭代优化让更强模型获得更高评审权重，提升整体评估准确性
    - 核心思路：每个模型的估计分数为 $\hat{G}_{M_j} = \sum_i \sum_{k \neq j} \sum_{r \neq k, r \neq j} \text{Review}_i^{j,k,r} \times w_r$，使用MSE损失 $\mathcal{L}_{MSE} = \frac{1}{m} \sum_{j=1}^{m} (\hat{G}_{M_j} - w_{M_j})^2$ 迭代更新权重 $w$ 与分数 $\hat{G}$ 的一致性
    - 设计动机：初步实验表明将更高权重分配给更强模型能显著提升评估准确性，动态优化自动发现这种权重分配

## 实验关键数据

### 主实验（Pearson / Spearman与人工评估的相关性）

| 方法 | MMStar Pearson↑ | MMStar Spearman↑ | ScienceQA Pearson↑ | ScienceQA Spearman↑ |
|------|----------------|------------------|--------------------|--------------------|
| GPT-4o (单模型评审) | 0.878 | 0.875 | 0.617 | 0.625 |
| Peer Review (原始) | 0.725 | 0.771 | 0.463 | 0.686 |
| Majority Vote | 0.757 | 0.757 | 0.509 | 0.524 |
| Rating Vote | 0.795 | 0.743 | 0.623 | 0.629 |
| PRD (半监督) | 0.838 | 0.864 | 0.636 | 0.694 |
| **UPME** | **0.944** | **0.972** | **0.814** | **0.886** |

### 消融实验

| 评分组件 | MMStar Pearson | ScienceQA Pearson |
|----------|----------------|-------------------|
| 仅Correctness | 0.854 | 0.713 |
| 仅Visual | 0.873 | 0.701 |
| 仅CLIP | 0.785 | 0.548 |
| Visual + CLIP | 0.903 | 0.775 |
| 全部（UPME） | **0.944** | **0.814** |

### 人工偏好对齐

| 数据集 | 方法 | 人工一致率 | 模型一致率 |
|--------|------|-----------|-----------|
| MMStar | Peer Review | 71.1% | 67.5% |
| MMStar | UPME | **95.9%** | **89.8%** |
| ScienceQA | Peer Review | 68.2% | 61.8% |
| ScienceQA | UPME | **87.4%** | **82.6%** |

### 关键发现
- 样本量25张图片即可达到稳定收敛，继续增加样本量（到100）指标变化很小
- UPME中各模型的评审准确率平均从61.56%提升到74.48%（正确性维度），视觉理解维度达73.93%
- 冗余偏差和自我偏好偏差在UPME中被有效抑制（Chi-Square检验p-value无显著性）

## 亮点与洞察
- **真正的无监督评估**：不需要任何人工标注的QA对，仅需图像即可完成评估，这是评估领域的重要范式转变
- **CLIP分数的巧妙引入**：作为第三方客观裁判，从模型外部引入图文对齐信号，是消除偏差的关键
- **收敛性保证**：64种不同初始化设置下均在30个epoch内收敛，框架稳定可靠
- **与人工偏好的超高一致性**：MMStar上95.9%的人工一致率说明UPME几乎可以替代人工评估

## 局限性
- 模型池仅6个模型（5个闭源+1个开源），需要验证更大规模模型池下的效果
- CLIP分数本身在某些场景下可能不可靠（如细粒度空间关系理解）
- 当前框架对评审模型的提问质量没有显式控制，低能力模型可能生成低质量问题
- ScienceQA上的相关性（0.814）低于MMStar（0.944），说明在需要深度推理的科学问题上仍有提升空间

## 相关工作与启发
- 承接Chatbot Arena的"让用户投票"思想，但将人工投票替换为模型自动评审
- CLIP分数作为偏差校正器的思路可推广到其他MLLM评估场景
- 启发：在构建新的MLLM评估基准时，可以先用UPME快速筛选出有价值的评估图像，再进行人工精标

## 补充分析
- UPME的计算成本主要来源于多轮模型推理（每轮需要3个模型参与），25张图片×多轮迭代的API调用成本约为单个benchmark评测的1/10
- 框架对模型池中最弱模型的容错能力较强，实验表明即使包含LLama-3.2-11B（Pearson仅0.314），整体评估质量仍保持0.944
- 动态权重优化的本质是一种自洽性约束——强模型自然在被评审和评审他人时都表现更好，这种一致性被数学化地捕捉

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个完全无监督的MLLM互评框架，同行评审+动态权重优化思路新颖
- 实验充分度: ⭐⭐⭐⭐ 消融、偏差分析、人工偏好对齐实验完整，但模型池规模偏小
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，数学符号规范，图表直观
- 价值: ⭐⭐⭐⭐ 为MLLM评估提供了低成本高效路径，但实用性需更大规模验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] PRISMM-Bench: A Benchmark of Peer-Review Grounded Multimodal Inconsistencies](../../ICLR2026/multimodal_vlm/prismm-bench_a_benchmark_of_peer-review_grounded_multimodal_inconsistencies.md)
- [\[ACL 2025\] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation](../../ACL2025/multimodal_vlm/flagevalmm_a_flexible_framework_for_comprehensive_multimodal_model_evaluation.md)
- [\[CVPR 2025\] Distraction is All You Need for Multimodal Large Language Model Jailbreaking](distraction_is_all_you_need_for_multimodal_large_language_model_jailbreaking.md)
- [\[CVPR 2025\] Period-LLM: Extending the Periodic Capability of Multimodal Large Language Model](period-llm_extending_the_periodic_capability_of_multimodal_large_language_model.md)
- [\[CVPR 2025\] CoLLM: A Large Language Model for Composed Image Retrieval](collm_a_large_language_model_for_composed_image_retrieval.md)

</div>

<!-- RELATED:END -->
