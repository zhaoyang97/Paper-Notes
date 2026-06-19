---
title: >-
  [论文解读] Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration
description: >-
  [ICML2025][多模态VLM][大型视觉语言模型] 提出 Self-Rationale Calibration (SRC) 框架，通过轻量级 rationale 微调引导 LVLM 输出推理过程，再利用句子级 beam search 生成多样候选响应，结合专门设计的 R-Scorer 配对评分策略筛选优劣 rationale-answer 对，以 DPO 偏好对齐方式迭代校准模型的推理-答案一致性，在感知、推理和泛化多个基准上取得显著提升。
tags:
  - "ICML2025"
  - "多模态VLM"
  - "大型视觉语言模型"
  - "推理对齐"
  - "偏好优化"
  - "自校准"
  - "Rationale生成"
---

# Towards Rationale-Answer Alignment of LVLMs via Self-Rationale Calibration

**会议**: ICML2025  
**arXiv**: [2509.13919](https://arxiv.org/abs/2509.13919)  
**作者**: Yuanchen Wu, Ke Yan, Shouhong Ding, Ziyin Zhou, Xiaoqiang Li  
**代码**: 未公开  
**领域**: 多模态VLM  
**关键词**: 大型视觉语言模型, 推理对齐, 偏好优化, 自校准, Rationale生成  

## 一句话总结

提出 Self-Rationale Calibration (SRC) 框架，通过轻量级 rationale 微调引导 LVLM 输出推理过程，再利用句子级 beam search 生成多样候选响应，结合专门设计的 R-Scorer 配对评分策略筛选优劣 rationale-answer 对，以 DPO 偏好对齐方式迭代校准模型的推理-答案一致性，在感知、推理和泛化多个基准上取得显著提升。

## 研究背景与动机

### 问题背景
大型视觉语言模型（LVLMs）在视觉问答（VQA）等任务上取得了显著进展，但视觉与文本模态之间的对齐问题（hallucination、事实错误等）仍然突出。现有的后训练策略主要分为两类：

**监督微调（SFT）**：通过额外标注数据进行指令微调

**偏好对齐（DPO 等）**：通过构造正负样本对，引导模型远离反事实描述

这些方法虽然在视觉描述任务上有效，但都**忽略了推理过程（rationale）的质量**，即模型的答案是否真正来自合理的、基于事实的推理链。

### 核心观察
作者对 LVLMs（如 LLaVA-1.5-7B）进行了深入分析，发现：

- **正确答案不等于正确推理**：模型经常给出正确答案，但背后的推理过程存在严重问题
- 推理缺陷的三种典型模式：
    - **反事实推理（Counterfactual）**：推理中包含与图像不符的描述
    - **推理不充分（Insufficient）**：推理过程缺乏关键信息支撑
    - **推理不合理（Unreasonable）**：推理逻辑链本身存在问题
- 当前指令微调主要依赖短答案数据集，缺乏对推理过程的显式监督，导致模型建立的是指令到答案的虚假因果关联，而非视觉到推理到答案的正确路径

### 核心问题
> "Does a correct answer stem from a reasonable and factually grounded rationale?"
> （正确的答案是否源自合理且基于事实的推理过程？）

## 方法详解

### 整体框架
SRC 框架分为四个核心阶段，可迭代执行以持续改进模型：

**阶段 1: Rationale 微调（Rationale Fine-Tuning）**
- 增强部分 VQA 样本，构造 Rationale-Answer Pairs (RAPs)，即包含推理过程和最终答案的完整响应格式
- 使用轻量级 LoRA 微调 LVLM，使模型在不需要显式提示的情况下自动输出先推理后回答的格式
- 核心价值：将模型从直接给答案转变为先给推理再给答案，为后续评估和校准创造条件

**阶段 2: 多样化候选生成（Diverse Candidate Generation）**
- 以 rationale 微调后的模型作为种子模型
- 对每个视觉指令样本，执行**句子级 beam search**在输出空间中搜索多样化的响应候选
- 关键设计：句子级而非 token 级搜索，产生推理路径差异更大的候选集

**阶段 3: R-Scorer 配对评分（Pairwise Scoring with R-Scorer）**
- 设计专门的轻量级 LLM 评分模型 **R-Scorer**
- 采用**配对评分策略**：将候选两两配对进行相对比较，而非独立打分
- 评估维度：推理质量（rationale quality）和事实一致性（factual consistency）
- 配对比较的优势：开放式推理难以绝对量化，A 比 B 推理更好的相对判断更可靠
- 结合 LLM-as-judge 范式，捕捉候选间推理过程的相对优越性

**阶段 4: 置信度加权偏好构建与对齐**
- 聚合配对评分中的置信度得分，识别最优候选和最劣候选
- 置信度加权解决中性评分模糊性和评分偏差问题
- 将优劣候选对组成偏好数据，通过 DPO 进行偏好微调
- 阶段 2-4 可迭代执行，每轮用更新后的模型生成新候选

### 关键设计详解

#### Rationale-Answer Pair (RAP) 格式
RAP 将模型输出结构化为 Rationale（视觉分析与推理过程）和 Answer（最终答案）两部分，使推理过程可检查、可评估，是整个框架的基础设计。

#### R-Scorer 评分模型
- 基于轻量级 LLM 构建，专门评估 rationale 质量而非仅看答案正确性
- 评估标准：推理是否与图像事实一致、逻辑链是否完整、rationale 是否充分支撑答案
- 配对评分避免了开放式推理难以绝对量化的困难

#### 句子级 Beam Search
区别于 token-level beam search，在句子粒度分支搜索，能产生推理路径本质不同的候选而非仅措辞差异，保证候选集多样性。

#### 迭代校准机制
每轮用改进模型生成新候选，R-Scorer 做更精细区分，构建更高质量偏好数据，进一步提升模型。形成良性循环：更好的模型产生更好的候选，更精确的偏好带来更好的模型。

## 实验关键数据

### 主要结果：MMStar 基准上的表现

| 模型 | 方法 | MMStar 表现 | 提升效果 |
|------|------|------------|---------|
| LLaVA-1.5-7B | Baseline | 基准水平 | — |
| LLaVA-1.5-7B | + SRC | 显著提升 | 多个百分点 |
| LLaVA-Next-8B | Baseline | 基准水平 | — |
| LLaVA-Next-8B | + SRC | 显著提升 | 多个百分点 |

论文 Figure 1 展示了 SRC 在 MMStar 上的效果，验证了框架在不同规模 LVLM 上的通用性。

### 与现有后训练方法的对比

| 方法类别 | 代表方法 | 核心思路 | 关注 Rationale |
|---------|---------|---------|---------------|
| 视觉描述偏好对齐 | RLHF-V, POVID | 扰动图像描述构建偏好对 | 否，仅描述准确性 |
| 输出采样偏好对齐 | LLaVA-RLHF, STIC | 从输出采样构建偏好 | 否，仅答案正确性 |
| 专家模型辅助 | DRESS, HA-DPO | 引入外部专家评估 | 部分关注 |
| **SRC（本文）** | **R-Scorer + 迭代校准** | **评估推理质量+答案一致性** | **核心关注推理质量** |

### 多维度能力提升

| 评估维度 | 代表基准 | SRC 改进效果 |
|---------|---------|-------------|
| 感知能力（Perception） | VQA 系列基准 | 显著提升 |
| 推理能力（Reasoning） | MMStar 等推理基准 | 最为明显的提升 |
| 泛化能力（Generalization） | 跨域测试 | 强泛化表现 |

论文强调相比仅关注视觉-文本对齐的后训练方法，SRC 在 QA 场景中的提升尤为显著。

## 亮点与洞察

- **问题定义精准**：首次系统研究 LVLM 中正确答案不等于正确推理的现象，将 rationale-answer alignment 作为独立研究问题
- **自监督闭环**：仅利用模型自身输出空间的质量差异进行校准，不依赖外部标注或专家模型
- **配对评分哲学**：相对比较比绝对评分更可靠，与 RLHF 中 Bradley-Terry 模型一脉相承
- **迭代渐进改进**：避免一次性训练的信息利用不充分问题
- **轻量级实现**：LoRA 微调加轻量 R-Scorer，整体计算开销可控
- **CoT 思维嵌入**：通过偏好学习而非纯 SFT 确保推理质量，是 Chain-of-Thought 在 VLM 训练中的深度应用

## 局限与展望

- **R-Scorer 可靠性**：评分模型偏差可能传播错误偏好信号
- **迭代计算开销**：多轮候选生成加评分加 DPO 的总成本较高
- **多样性上限**：种子模型推理能力弱时，候选集可能缺乏高质量样本
- **评估主观性**：rationale 质量评估即使配对比较也无法完全消除噪音
- **模型覆盖有限**：主要基于 LLaVA 系列验证，其他架构（Qwen-VL、InternVL 等）适用性待确认
- **格式刚性**：固定 RAP 格式可能影响自由对话场景的自然性

## 相关工作与启发

### LVLM 偏好对齐
RLHF-V、POVID、LLaVA-RLHF、STIC 等工作关注描述是否准确，SRC 进一步关注推理是否合理。

### LLM-as-Judge
R-Scorer 是 LLM-as-judge 在视觉推理评估的创新应用，配对而非独立评分提高判断质量。

### Chain-of-Thought 与推理
CoT 仅在推理时引导输出推理过程，SRC 通过训练时偏好学习从根本上提升推理质量。

### 自我改进范式
Self-play、Self-reward 利用模型自身能力迭代改进，SRC 的闭环与之一致但专注推理-答案对齐。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统研究 LVLM 的 rationale-answer 对齐，配对评分策略有创新
- 实验充分度: ⭐⭐⭐⭐ — 多基准多模型验证，感知/推理/泛化三维度
- 写作质量: ⭐⭐⭐⭐ — 问题动机清晰，框架逻辑自洽，例子直观
- 价值: ⭐⭐⭐⭐ — 指出 LVLM 后训练的重要盲区，方法实用可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Self-Prophetic Decoding to Unlock Visual Search in LVLMs](../../ICML2026/multimodal_vlm/self-prophetic_decoding_to_unlock_visual_search_in_lvlms.md)
- [\[ICML 2025\] CoMemo: LVLMs Need Image Context with Image Memory](comemo_lvlms_need_image_context_with_image_memory.md)
- [\[CVPR 2026\] Multimodal Learning on Low-Quality Data with Conformal Predictive Self-Calibration](../../CVPR2026/multimodal_vlm/multimodal_learning_on_low-quality_data_with_conformal_predictive_self-calibrati.md)
- [\[CVPR 2025\] VladVA: Discriminative Fine-tuning of LVLMs](../../CVPR2025/multimodal_vlm/vladva_discriminative_fine-tuning_of_lvlms.md)
- [\[NeurIPS 2025\] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs](../../NeurIPS2025/multimodal_vlm/video-safetybench_a_benchmark_for_safety_evaluation_of_video_lvlms.md)

</div>

<!-- RELATED:END -->
