---
title: >-
  [论文解读] To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning
description: >-
  [NeurIPS 2025 Spotlight][多模态VLM][强化微调] 系统研究了基于规则的强化微调（RFT）中显式思维过程的必要性，发现视觉感知任务中"不思考"的RFT（No-Thinking-RFT）往往优于传统的"先思考再回答"策略，并提出了自适应思维方法让模型根据自身能力和任务复杂度决定是否思考。
tags:
  - "NeurIPS 2025 Spotlight"
  - "多模态VLM"
  - "强化微调"
  - "思维链"
  - "多模态大语言模型"
  - "GRPO"
  - "视觉推理"
---

# To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2503.16188](https://arxiv.org/abs/2503.16188)  
**代码**: [https://github.com/minglllli/CLS-RL](https://github.com/minglllli/CLS-RL)  
**领域**: 多模态VLM  
**关键词**: 强化微调, 思维链, 多模态大语言模型, GRPO, 视觉推理

## 一句话总结

系统研究了基于规则的强化微调（RFT）中显式思维过程的必要性，发现视觉感知任务中"不思考"的RFT（No-Thinking-RFT）往往优于传统的"先思考再回答"策略，并提出了自适应思维方法让模型根据自身能力和任务复杂度决定是否思考。

## 研究背景与动机

自DeepSeek-R1以来，基于规则的强化微调（RFT）在LLM中取得了巨大成功，其核心机制是鼓励模型在回答前进行显式思维（CoT），这被广泛认为是RFT成功的关键因素。许多多模态RFT工作也试图复现R1中的"思维长度渐增"和"aha moment"效果。

然而一个被忽视的关键问题是：**显式思维在RFT中是否总是必要且有益的？** 已有研究表明，在常识任务上推理带来的收益有限，过度思考甚至会损害推理性能。但这些发现仅聚焦于推理阶段，对训练阶段显式思维的影响尚未系统探索。此外，RFT需要生成多个冗长回复，其微调时间和GPU内存远超SFT，这使得思维过程的效率问题更加紧迫。

本文的切入点是：从分类任务出发，观察到RFT过程中回复长度在特定步骤出现骤降（而非渐增），同时准确率显著提升——这表明在某些视觉任务中思维并非必需。基于此洞察，作者提出了No-Thinking-RFT并系统比较了四种思维策略。

## 方法详解

### 整体框架

本文围绕GRPO（Group Relative Policy Optimization）优化算法，系统比较了四种RFT思维策略：Thinking-RFT、No-Thinking-RFT、Think-After-Answer和Adaptive-Thinking。所有方法均采用R1-zero训练方式（直接在base model上RL，不做SFT）。

### 关键设计

1. **Thinking-RFT（基线）**: 采用标准R1思维范式。Prompt设计为"请在`<think>`标签中输出思维过程，在`<answer>`标签中输出最终答案"。奖励函数包含格式奖励（$R_{\text{format}}$）和准确率奖励（$R_{\text{accuracy}}$），两者均为0/1二值。

2. **No-Thinking-RFT**: 核心创新。Prompt设计为"请直接输出答案"，完全禁止思维过程。**取消格式奖励**，仅使用严格等式匹配的准确率奖励——只有当模型输出与标签完全匹配时才给1分。这种设计迫使模型跳过任何思维过程，直接输出答案，从而大幅缩短训练和推理时间。

3. **Think-After-Answer**: 为验证"显式思维在可验证答案之前会阻碍学习"这一假设而设计。让模型先回答问题，再输出简短推理过程。这样思维过程以答案为条件，而非答案以思维为条件，缓解了显式思维对可验证答案的负面影响。

4. **Adaptive-Thinking**: 让模型自主决定是否思考。Prompt要求模型先判断问题是否需要推理，如需要则输出思维过程再回答，否则直接回答。无论选择哪种格式，格式奖励均为1。实验发现模型最终会收敛到单一策略（要么总是思考，要么总是不思考），且该策略恰好对应最优方案。

### 损失函数 / 训练策略

所有方法均使用GRPO算法进行优化，采用二值奖励函数（格式+准确率或仅准确率）。学习率统一为 $1 \times 10^{-6}$，$\beta = 0.04$，温度为1.0。不同任务的rollout数量（2B为4-8，7B为4）和训练轮次有所不同。

## 实验关键数据

### 主实验

| 数据集/任务 | 指标 | No-Thinking-RFT | Thinking-RFT | 提升/对比 |
|------------|------|-----------------|--------------|----------|
| CVBench (2B) | Overall Acc | **76.76** | 70.36 | +6.40 |
| CVBench (7B) | Overall Acc | 80.67 | **80.36** | +0.31 |
| PuzzleVQA (2B) | Acc | **70.85** | 52.50 | +18.35 |
| PuzzleVQA (7B) | Acc | 80.65 | 66.60 | +14.05 |
| MathVista (2B) | Overall Acc | **48.80** | 44.90 | +3.90 |
| MathVista (7B) | Overall Acc | 59.10 | **64.60** | -5.50 |

训练时间对比（CVBench 2B）：No-Thinking-RFT仅需139分钟 vs Thinking-RFT的599分钟，快4.3倍。

### 消融实验

| 配置 | CVBench (2B) | PuzzleVQA (2B) | 说明 |
|------|-------------|----------------|------|
| Thinking-RFT | 70.36 | 52.50 | 完整思维 |
| Think-After-Answer | 73.65 | 64.70 | 答后思考，收敛更快 |
| No-Thinking-RFT | **76.76** | **70.85** | 无思维最优 |
| Adaptive-Thinking | 77.03 | 75.45 | 自适应，收敛至不思考 |
| Thinking-RFT + 空think推理 | - | - | 部分提升但仍落后No-Thinking |

### 关键发现

- **发现1**：能力较弱的小模型（2B）在Thinking-RFT下倾向于生成trivial推理，导致性能不如No-Thinking-RFT
- **发现2**：视觉感知和谜题任务不需要思维——No-Thinking-RFT在所有模型尺寸上均匹配或超越Thinking-RFT
- **发现3**：Thinking-RFT中think标签和answer标签之间存在不一致性，不一致回复的准确率低于整体平均
- **发现4**：No-Thinking-RFT的增益来自两方面——训练中更好的学习和推理时避免了过度思考
- **发现5**：显式思维置于答案之前会导致更慢的奖励收敛和更低的性能
- **发现6**：2B模型在数学任务上收敛至不思考策略，7B模型收敛至思考策略，说明模型能自适应决定

## 亮点与洞察

- 挑战了"RFT中显式思维必不可少"的传统认知，提供了扎实的实验证据
- No-Thinking-RFT极其简洁（仅用等式匹配奖励），却在多个任务上表现优异且训练效率提升数倍
- Adaptive-Thinking展示了模型自主学习何时思考的潜力，且最终策略恰好匹配最优方案
- 参数变化分析揭示了不同思维策略对模型权重分布的差异化影响

## 局限与展望

- Adaptive-Thinking目前只能在任务级别收敛到统一策略，尚未实现问题级别的自适应
- 实验主要在2B-7B模型上进行，更大规模模型的行为有待验证
- 对"为什么trivial推理会出现"缺乏深入的理论解释

## 相关工作与启发

- **vs DeepSeek-R1**: R1在数学任务上展示了思维长度渐增的现象，本文发现视觉分类任务中长度会骤降，说明不同任务对思维的需求截然不同
- **vs SFT**: Thinking-RFT在分类任务上显著优于SFT，且展现出跨数据集迁移效果（在一个数据集上RFT可以提升其他数据集的性能）
- **vs 推理阶段CoT研究**: 本文将"思维是否有益"的问题从推理阶段扩展到训练阶段，提供了更全面的视角

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统质疑RFT中思维的必要性，切入角度新颖但方法本身较简单
- 实验充分度: ⭐⭐⭐⭐⭐ 六个任务、多种模型尺寸、四种思维策略的全面对比，发现丰富
- 写作质量: ⭐⭐⭐⭐ 结构清晰，发现逐步深入，实验设计有说服力
- 价值: ⭐⭐⭐⭐⭐ 对多模态RFT社区极具指导意义，挑战了盲目追随思维链的趋势

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](../../ICCV2025/multimodal_vlm/docthinker_explainable_multimodal_large_language_models_with.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](../../CVPR2026/multimodal_vlm/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[CVPR 2025\] Spatial Reasoning is Not a Free Lunch: A Controlled Study on LLaVA](../../CVPR2025/multimodal_vlm/spatial_reasoning_is_not_a_free_lunch_a_controlled_study_on_llava.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](visual_instruction_bottleneck_tuning.md)

</div>

<!-- RELATED:END -->
