---
title: >-
  [论文解读] Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards
description: >-
  [NeurIPS 2025][多模态VLM][chain-of-step] 提出Chain-of-Step (CoS)推理框架，将VLM的推理链拆解为由Name+Thought+Reflection组成的结构化步骤，训练步骤级Process Reward Model (PRM)提供精细奖励信号，配合迭代DPO和step-level beam search系统性提升VLM推理能力——在InternVL-2.5-MPO-8B上6个benchmark平均73.4%（+4.0%），在LLaVA-NeXT-8B上平均64.2%（+12.1%），并揭示了"VLM推理中质量远比长度重要"这一与LLM领域相反的发现。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "chain-of-step"
  - "process reward model"
  - "step-level reasoning"
  - "iterative DPO"
  - "inference-time scaling"
---

# Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards

**会议**: NeurIPS 2025  
**arXiv**: [2509.19003](https://arxiv.org/abs/2509.19003)  
**代码**: [https://github.com/baaivision/CoS](https://github.com/baaivision/CoS)  
**领域**: 多模态VLM / 视觉推理 / 过程奖励模型  
**关键词**: chain-of-step, process reward model, step-level reasoning, iterative DPO, inference-time scaling

## 一句话总结

提出Chain-of-Step (CoS)推理框架，将VLM的推理链拆解为由Name+Thought+Reflection组成的结构化步骤，训练步骤级Process Reward Model (PRM)提供精细奖励信号，配合迭代DPO和step-level beam search系统性提升VLM推理能力——在InternVL-2.5-MPO-8B上6个benchmark平均73.4%（+4.0%），在LLaVA-NeXT-8B上平均64.2%（+12.1%），并揭示了"VLM推理中质量远比长度重要"这一与LLM领域相反的发现。

## 研究背景与动机

**领域现状**：Chain-of-Thought推理已在LLM领域取得巨大成功，OpenAI-o1和DeepSeek-R1通过大规模RL+CoT实现了推理能力的飞跃。VLM领域也在积极探索CoT推理（LLaVA-CoT、Insight-V、URSA等），但整体仍处于粗粒度阶段。

**现有痛点**：当前VLM的CoT推理输出是一大段缺乏结构的"thought"——没有统一格式、没有清晰的步骤划分，导致两个核心问题：①推理过程容易变得冗长混乱，难以执行系统化的结构化推理；②无法评估中间推理步骤的质量，使得RL训练和inference-time scaling都缺乏有效的reward信号。

**核心矛盾**：LLM领域的PRM（如Math-Shepherd、Let's Verify Step by Step）已证明步骤级reward的价值，但在VLM领域面临两个非平凡的挑战——如何定义"步骤"（将推理链分解为逻辑连贯的渐进步骤）以及如何评估"步骤"（提供精细的步骤级reward信号）。

**本文目标**：为VLM建立一套完整的步骤级推理框架：从结构化推理格式的定义、SFT数据构建、过程奖励模型训练，到基于精细reward的RL训练和推理时scaling。

**切入角度**：从推理链的结构化设计入手，用特殊token划分步骤边界，每步引入Reflection组件连接视觉内容以缓解幻觉，使得步骤拆分稳定可解析，从而为PRM训练和RL提供坚实基础。

**核心 idea**：通过将VLM推理链结构化为可评估的离散步骤，并训练PRM提供步骤级精细奖励，使RL训练和inference-time scaling都能从中间步骤质量中获益。

## 方法详解

### 整体框架

三阶段pipeline：
1. **SFT on ShareGPT-Step-300K**：在300K结构化步骤推理数据上做SFT，教模型输出步骤化推理链
2. **训练PRM**：用Monte Carlo估计和GPT-4o-as-Judge两种方法标注步骤级数据（各100K），训练InternVL-2.5-MPO-38B作为过程奖励模型
3. **Iterative DPO with PRM**：用PRM对采样的推理路径评分，选择正负样本对做3轮迭代DPO，渐进增强推理能力

### 关键设计

1. **结构化推理步骤模板（Structured Reasoning Template）**

    - **功能**：将VLM的自由形式推理链分解为格式稳定、可解析、可评估的离散步骤
    - **核心思路**：每个推理步骤包含三个组件——**Name**（步骤概要，如"识别几何形状"）、**Thought**（详细推理内容）、**Reflection**（与视觉内容和前序步骤建立关联以缓解幻觉）。用特殊token（`<|reasoning_start|>`、`<|reasoning_proceed|>`、`<|reasoning_end|>`等）标记步骤边界，步骤数量和长度由模型自回归生成时自主决定
    - **设计动机**：基于prompt的格式控制不稳定且需要额外数据清洗，用特殊token嵌入格式保证输出稳定性；Reflection组件是专门为VLM设计的——LLM不需要回看视觉内容，但VLM容易生成与图像矛盾的内容，显式的反思步骤可以缓解这一问题

2. **ShareGPT-Step-300K数据集构建**

    - **功能**：为SFT阶段提供高质量的结构化步骤推理训练数据
    - **核心思路**："从结果推理"策略——将问题和ground-truth答案一起提供给GPT-4o，让其逆向生成步骤化推理过程。覆盖17个数据集、4大类任务（数学推理、科学推理、图表文档分析、世界知识），经严格格式清洗后保留300K高质量样本
    - **设计动机**：直接让LLM生成推理过程容易出错，给定参考答案可大幅降低推理难度、显著提升生成质量；17个数据集的多样性确保模型学到的推理能力具有泛化性

3. **过程奖励模型（Process Reward Model, PRM）**

    - **功能**：对推理链的每个步骤给出质量评分，为RL训练和inference-time scaling提供精细reward信号
    - **核心思路**：双管齐下采集标注——Math-Shepherd方法（MC估计，每步采样16条后续路径验证正确率）和GPT-4o-as-Judge方法（Good/Neutral/Bad三级评分）各标注100K步骤级数据；以InternVL-2.5-MPO-38B为基座，BCE loss训练2 epochs得到PRM。评估时step score权重20% + answer score权重80%加权求和作为综合评分
    - **设计动机**：MC估计从统计角度客观评价步骤通向正确答案的概率，LLM-as-Judge从语义角度评价步骤的逻辑正确性——两者互补可训练更鲁棒的PRM；选择38B大模型做PRM（而非8B）是因为更大模型对步骤评估更准确（unseen data上step accuracy 87.3% vs 83.7%）

### 损失函数 / 训练策略

- **SFT阶段**：标准的next-token prediction loss，在ShareGPT-Step-300K上训练1 epoch
- **PRM训练**：Binary Cross Entropy loss，对每个步骤预测其正确性概率
- **Iterative DPO**：标准DPO loss，$\mathcal{L}_{\text{DPO}}(\pi_\theta;\pi_{\text{ref}}) = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_+|x)}{\pi_{\text{ref}}(y_+|x)} - \beta\log\frac{\pi_\theta(y_-|x)}{\pi_{\text{ref}}(y_-|x)})]$，其中$\beta=0.1$。每轮从SFT模型初始化policy和reference，生成16条推理路径用PRM评分选正负对（分差须超过阈值$t$），每轮约20K pairs，共3轮迭代
- **训练成本**：SFT约9小时/单节点A800，3轮DPO共约6小时/单节点A800
- **Step-level Beam Search（推理时）**：每步采样N个候选→PRM打分→选最佳步骤→以此为基础继续采样下一步→重复直到输出答案。与Best-of-N计算量完全相同

## 实验关键数据

### 主实验

| 方法 | MathVista | MMStar | MMMU | M3CoT | AI2D | ChartQA | Avg |
|------|-----------|--------|------|-------|------|---------|-----|
| InternVL2.5-MPO-8B (baseline) | 65.0 | 60.7 | 53.8 | 67.5 | 84.2 | 85.0 | 69.4 |
| + CoS SFT | 65.9 | 61.0 | 53.7 | 75.7 | 81.6 | 88.3 | 71.0 |
| + CoS Iterative DPO | **67.8** | **63.5** | **55.5** | **81.0** | **84.9** | **87.4** | **73.4** |
| LLaVA-NeXT-8B (baseline) | 45.9 | 43.1 | 36.9 | 45.6 | 71.5 | 69.4 | 52.1 |
| + CoS SFT | 51.4 | 54.7 | 39.6 | 67.4 | 76.1 | 75.7 | 60.8 |
| + CoS Iterative DPO | **54.7** | **58.9** | **41.8** | **71.7** | **79.2** | **79.1** | **64.2** |

### 消融实验

**RL reward策略消融**（LLaVA-NeXT-SFT基础上）：

| 方法 | MathVista | MMStar | M3CoT |
|------|-----------|--------|-------|
| LLaVA-NeXT-SFT | 51.4 | 54.7 | 67.4 |
| Answer Only (PRM) | 53.1 | 57.3 | 69.7 |
| Outcome (GT labels) | 53.5 | 58.1 | 70.0 |
| Step&Answer (PRM) | **54.7** | **58.9** | **71.7** |

**推理模式消融**（LLaVA-NeXT基础上）：

| 方法 | Reward | MathVista | MMStar | M3CoT | Avg |
|------|--------|-----------|--------|-------|-----|
| No Reason SFT→RL | outcome | 51.5 | 56.4 | 63.4 | 57.1 (+2.1) |
| Direct Prompt SFT→RL | outcome | 53.1 | 58.2 | 69.3 | 60.2 (+2.7) |
| CoS SFT→RL | PRM | **54.7** | **58.9** | **71.7** | **61.8 (+4.0)** |

**GRPO验证**：

| 方法 | MathVista | MMStar | M3CoT | Avg |
|------|-----------|--------|-------|-----|
| Outcome GRPO | 54.3 | 57.9 | 71.4 | 61.2 |
| CoS GRPO (PRM) | **56.3** | **59.1** | **73.7** | **63.0** |

### 关键发现

- **Step weight最优20%**：纯step score或纯answer score都非最优，Best-of-16 accuracy在step权重20%时达到峰值，表明综合考量步骤和答案质量最有效
- **Step-level beam search > Best-of-N**：在N=64时PRM-BS比Self-Consistency高5%+，且与Best-of-N PRM计算量完全相同但更优
- **推理长度反直觉现象**：PRM DPO训练初期模型主动缩短推理长度以提升质量，稳定后才缓慢增长；而Outcome DPO则持续增长长度。这表明VLM推理中质量远比长度重要——与LLM领域"更长=更强"的规律相反
- **强弱模型差异**：弱模型（LLaVA-NeXT）SFT和DPO都大幅提升（+12.1%），强模型（InternVL2.5-MPO）SFT提升有限但DPO仍显著，说明RL对强模型更为关键
- **Step-wise DPO失败**：尝试在每步构造preference pair进行步骤级DPO，但因为chosen和rejected太相似导致模型拒绝输出——需要足够大的正负差异才能形成有效学习信号

## 亮点与洞察

- **结构化设计的实用性**：Name+Thought+Reflection三组件分工明确——Name提供导航，Thought承载推理，Reflection专职连接视觉信息和前序推理。用特殊token而非prompt控制格式的决策非常务实
- **"质量>长度"的洞察**：这一发现具有重要指导意义——视觉推理更依赖视觉信息的有效利用和知识连接触发，而非像纯数学那样需要冗长的中间推导
- **PRM的scale efficiency**：38B PRM只需训练一次即可服务多个8B模型的RL训练和inference scaling，是一种高效的资源配置
- **全面的failure analysis**：诚实报告了Step-wise DPO的失败案例，揭示了preference learning中正负样本差异性的重要性
- **框架的透明性和可复现性**：数据集、PRM、代码全部开源，是VLM细粒度推理的solid baseline

## 局限与展望

- **数据构建依赖闭源模型**：ShareGPT-Step-300K和GPT-4o-as-Judge标注都依赖GPT-4o，增加成本和不确定性
- **模型规模验证有限**：仅在8B模型上全面验证，更大模型（如72B）上的效果和推理长度变化规律尚不明确
- **Reflection组件的实际效果存疑**：论文未提供Reflection组件移除的消融实验，无法定量验证其对缓解幻觉的贡献
- **PRM推理成本**：38B PRM在生产部署中inference成本偏高，可探索知识蒸馏至更小的PRM
- **步骤定义的通用性**：当前的步骤划分方式主要针对QA和数学推理，对于更开放式的视觉任务（如创意生成、长文档理解）的适用性未验证

## 相关工作与启发

- **vs LLaVA-CoT**：LLaVA-CoT用粗粒度的SUMMARY/CAPTION/REASONING/CONCLUSION四段式，CoS用细粒度的Name/Thought/Reflection步骤+PRM——粒度差异直接影响了reward信号的精度和inference scaling的效果
- **vs URSA**：同样使用PRM但推理链仍是粗粒度的，CoS的结构化步骤使得PRM对每步的评估更准确，step-level beam search因此成为可能
- **vs Insight-V**：Insight-V用多agent系统（reasoning agent + summary agent），CoS用单模型+PRM的更简洁架构——提示我们复杂问题不一定需要复杂系统
- **vs NoisyRollout/Sherlock**：NoisyRollout增强exploration diversity，Sherlock做response-level自纠正——都与CoS的step-level精细reward互补，可以组合使用
- **与inference-time scaling方向的关系**：PRM+step-level beam search为VLM提供了一种新的inference-time compute利用方式，与Self-Consistency和Best-of-N相比更高效

## 评分

- **新颖性**: ⭐⭐⭐⭐ 理由：结构化步骤+PRM在VLM领域是较新的组合，Name+Thought+Reflection的三组件设计和step-level beam search有创新性，但各单独组件（PRM、iterative DPO、结构化推理）并非全新概念
- **实验充分度**: ⭐⭐⭐⭐⭐ 理由：消融极其全面——覆盖step weight、PRM基座选择、推理长度动态、reasoning pattern对比、GRPO验证、step-wise DPO失败分析等，每个结论都有实验支撑
- **写作质量**: ⭐⭐⭐⭐⭐ 理由：逻辑递进清晰（定义步骤→评估步骤→利用步骤），复杂设计用Figure 1一图概览，失败实验也诚实报告，论文结构紧凑信息密度高
- **对我的价值**: ⭐⭐⭐⭐⭐ 理由：VLM推理后训练的complete framework（SFT+PRM+DPO+beam search全链路开源），"质量>长度"的insight对adaptive inference有直接指导价值，PRM+step-level beam search可作为VLM inference scaling的标准方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](../../ICCV2025/multimodal_vlm/fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[ACL 2025\] A Parameter-Efficient and Fine-Grained Prompt Learning for Vision-Language Models](../../ACL2025/multimodal_vlm/a_parameter-efficient_and_fine-grained_prompt_learning_for_vision-language_model.md)
- [\[ICCV 2025\] LLaVA-CoT: Let Vision Language Models Reason Step-by-Step](../../ICCV2025/multimodal_vlm/llava-cot_let_vision_language_models_reason_step-by-step.md)

</div>

<!-- RELATED:END -->
