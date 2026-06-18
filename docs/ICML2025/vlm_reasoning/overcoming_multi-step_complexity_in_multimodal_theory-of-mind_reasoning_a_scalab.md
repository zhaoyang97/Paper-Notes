---
title: >-
  [论文解读] Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner
description: >-
  [ICML2025 Spotlight][多模态VLM][Theory-of-Mind] 提出一种可扩展的贝叶斯心智理论（ToM）规划器，通过将多步推理分解为逐步贝叶斯更新，并利用弱到强控制机制将小模型的 ToM 专项能力迁移至大模型（最高 405B），在多模态 ToM 基准上超越 SOTA 4.6%。
tags:
  - "ICML2025 Spotlight"
  - "多模态VLM"
  - "Theory-of-Mind"
  - "Bayesian Inverse Planning"
  - "Weak-to-Strong Control"
  - "多模态推理"
  - "心智理论"
---

# Overcoming Multi-step Complexity in Multimodal Theory-of-Mind Reasoning: A Scalable Bayesian Planner

**会议**: ICML2025 Spotlight  
**arXiv**: [2506.01301](https://arxiv.org/abs/2506.01301)  
**作者**: Chunhui Zhang, Zhongyu Ouyang, Kwonjoon Lee, Nakul Agarwal, Sean Dae Houlihan, Soroush Vosoughi, Shao-Yuan Lo
**代码**: 待确认  
**领域**: 多模态VLM  
**关键词**: Theory-of-Mind, Bayesian Inverse Planning, Weak-to-Strong Control, 多模态推理, 心智理论

## 一句话总结

提出一种可扩展的贝叶斯心智理论（ToM）规划器，通过将多步推理分解为逐步贝叶斯更新，并利用弱到强控制机制将小模型的 ToM 专项能力迁移至大模型（最高 405B），在多模态 ToM 基准上超越 SOTA 4.6%。

## 研究背景与动机

### 心智理论（Theory-of-Mind）的核心挑战

心智理论（ToM）是人类社会认知的基石，使人能够推断他人的信念（beliefs）、意图（desires）和目标（intentions）。在 AI 中，ToM 任务要求模型感知可观察信号（如动作、视觉上下文），进而预测智能体的目标和信念状态。

现有方法主要分两大路线：

**结构化规划方法**：利用 ToM 特定先验进行结构化工作流设计（Baker et al., 2017; Jara-Ettinger, 2019; Shu et al., 2021）

**模型微调方法**：将 ToM 先验通过专门训练集成进语言模型（Rabinowitz et al., 2018; Sclar et al., 2022; Jin et al., 2024）

### 多模态环境下的可扩展性瓶颈

论文通过 VirtualHome 模拟器实验揭示了两个根本性问题：

- **推理边界存在上限**：随着规划步数增加，CoT 推理和 o1/r1 类推理时间扩展方法的效果出现平台效应（plateau），边际收益递减。即使用推理时间扩展方法（如 o1-mini、CoT），小模型（Llama3.1-8B, 70B）在多步规划中准确率迅速下降
- **世界知识依赖模型规模**：多模态 ToM 推理不是封闭的逻辑推理，而是需要丰富的社会知识和世界知识做支撑。研究表明只有大模型（如 Llama3.1-405B）才能在多步规划中维持性能

这两点发现说明：单纯的微调或推理时间扩展都不足以提升 ToM 推理的可扩展性，需要结构化框架 + 大规模模型的双管齐下。

## 方法详解

### 整体框架：可扩展贝叶斯 ToM 规划器

本文提出的方法由两个核心组件构成：

```
┌─────────────────────────────────────────────────┐
│        Scalable Bayesian ToM Planner            │
├─────────────────────────────────────────────────┤
│                                                 │
│  组件1: 贝叶斯逆向规划 (BIP)                     │
│  ┌───────────────────────────────────────────┐  │
│  │ 多模态 ToM → 逐步贝叶斯更新               │  │
│  │ • 状态转移 (state transitions)            │  │
│  │ • 信念更新 (belief updates)               │  │
│  │ • 动作似然 (action likelihoods)           │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
│  组件2: 弱到强控制 (Weak-to-Strong Control)      │
│  ┌───────────────────────────────────────────┐  │
│  │ 小模型(专项训练) → 大模型(知识整合)         │  │
│  │ • 小 LM: ToM 似然估计专精化               │  │
│  │ • 大 LM(7B→405B): 世界知识 + 贝叶斯推理  │  │
│  └───────────────────────────────────────────┘  │
│                                                 │
└─────────────────────────────────────────────────┘
```

### 关键设计 1：贝叶斯逆向规划（Bayesian Inverse Planning, BIP）

BIP 的核心思想是将复杂的多模态 ToM 推理分解为模块化的逐步贝叶斯更新。具体而言：

- **行为建模采用 POMDP 框架**：定义为元组 ⟨S, A, T, G, R, Ω, O, γ⟩，其中 s^t 为状态，a^t 为动作，T 为状态转移概率，g 为目标，R 为奖励函数，o^t 为观测
- **信念动态更新**：智能体的信念 b(s) 是对状态的概率分布，在信念演化过程中动态更新
- **逐步分解**：将原本需要端到端完成的多步推理拆分为独立的子模块（状态转移、信念更新、动作似然），每步通过贝叶斯准则迭代精化假设，确保在复杂环境中仍然可处理（tractable）

这种分解策略的优势在于：即使任务步数增加，每一步的推理复杂度保持可控，从而突破传统方法的推理边界上限。

### 关键设计 2：弱到强控制（Weak-to-Strong Control）

这是本文最核心的创新。先前方法（如 Jin et al., 2024）依赖小型 LM 进行似然估计，但小模型的世界知识容量有限，在丰富的 ToM 场景中泛化能力不足。

弱到强控制的工作机制：

1. **小模型专精化训练**：通过后训练（post-training）使较小的 LM 专精于 ToM 特定任务（如似然估计），学习到 ToM 推理的特殊行为模式
2. **行为迁移至大模型**：将小模型学到的 ToM 推理行为迁移至更大的 LM（从 7B 扩展到 405B）
3. **大模型作为主策略模型**：大模型承担主要推理角色，利用其丰富的预训练世界知识，同时通过迁移来的 ToM 行为保持贝叶斯一致性
4. **理论保障**：通过 Theorem 1 利用 KL 散度分析形式化证明了该方法的有效性

这种设计实现了两全其美：小模型的专项能力 + 大模型的知识广度。

### 关键设计 3：多模态信号整合

ToM 环境要求模型综合处理多种模态信息：

- **视觉信息**：环境场景、物体位置、智能体动作的视觉表征
- **文本信息**：动作描述、任务指令、上下文说明
- **上下文信息**：历史交互、时序关系、社会规范

模型需要将这些多模态线索整合为连贯的心理状态推断，这正是贝叶斯框架的优势所在——通过概率推理自然融合不同来源的证据。

## 实验关键数据

### 主实验：多模态 ToM 基准测试

| 方法类别 | 代表方法 | 多步规划表现 | 可扩展性 |
|---------|---------|-------------|---------|
| CoT 推理 | Chain-of-Thought | 步数增加后准确率下降 | 差 |
| 推理时间扩展 | o1-mini | 提供增量改进但无法维持 | 中等 |
| 小模型微调 | Fine-tuned experts | 受限于推理边界 | 差 |
| 小模型 (Llama3.1-8B) | 基础推理 | 快速退化 | 差 |
| 中等模型 (Llama3.1-70B) | 基础推理 | 中等退化 | 中等 |
| 大模型 (Llama3.1-405B) | 基础推理 | 维持性能 | 好 |
| **本文方法** | **Bayesian ToM Planner** | **SOTA + 4.6%** | **最佳** |

### 不同模型规模下的性能对比

| 模型规模 | 基础推理能力 | + 本文 BIP | + 弱到强控制 | 改进幅度 |
|---------|------------|-----------|------------|---------|
| 7B | 低 | 中等提升 | 显著提升 | 较大 |
| 70B | 中 | 明显提升 | 进一步提升 | 中等 |
| 405B | 高 | 稳定提升 | 最优表现 | SOTA |

### 关键定量结果

- 在多模态 ToM 基准测试中，总体准确率较 SOTA 提升 **4.6%**
- 在**未见过的场景**（unseen scenarios）中同样有效，验证了泛化能力
- 随着规划步数增加，本文方法的性能衰减明显低于所有对比方法

## 亮点与洞察

1. **问题定位精准**：通过 VirtualHome 实验清晰揭示了多步 ToM 推理的两个根本瓶颈（推理边界 + 世界知识依赖），为方法设计提供了明确的指导方向

2. **弱到强控制是核心创新**：不同于简单的模型蒸馏或知识迁移，弱到强控制让小模型专精于 ToM 似然估计，再将这种"专精行为"注入大模型，兼顾专项能力与通用知识。这种设计思想可推广至其他需要同时具备专项能力和广泛知识的任务

3. **贝叶斯分解缓解推理瓶颈**：将端到端的多步推理拆分为模块化的贝叶斯更新步骤，使得每步推理保持可控复杂度，有效突破了推理边界的限制

4. **理论与实验双重验证**：通过 Theorem 1（KL 散度分析）提供了理论保障，同时在基准测试和未见场景中验证了实际效果

5. **对推理时间扩展的反思**：论文指出 CoT、o1 等推理时间扩展方法在 ToM 场景中存在根本性局限，这一发现对理解推理扩展的边界具有启发价值

## 局限与展望

1. **计算成本高**：需要同时维护小模型（专精训练）和大模型（推理执行），405B 级别的模型推理开销巨大，限制了实际部署场景

2. **实验环境相对受限**：主要基于 VirtualHome 模拟器环境，真实世界中更加复杂、模糊的社会场景（如讽刺、隐喻、文化差异）的泛化能力有待验证

3. **依赖大模型的世界知识**：方法的有效性在根本上依赖于大模型的预训练知识质量，对于冷门领域或特殊文化背景的 ToM 推理可能受限

4. **贝叶斯框架的假设约束**：POMDP 框架要求明确定义状态空间、动作空间和转移函数，在开放域场景中这些定义可能不自然

5. **弱到强迁移的理论边界**：虽然 Theorem 1 提供了 KL 散度分析，但迁移效率如何随模型规模差异变化、是否存在最优的大小模型配比，尚未充分探讨

## 相关工作与启发

### 心智理论建模

- **经典贝叶斯方法**：Baker et al. (2007, 2009, 2017) 提出贝叶斯逆向规划，Shum et al. (2019) 扩展至更复杂场景
- **基于 LM 的方法**：Jin et al. (2024) 将贝叶斯推理与语言模型结合，但受限于小模型容量
- **本文贡献**：首次将贝叶斯 ToM 推理扩展至 405B 规模，突破了小模型的泛化瓶颈

### 推理时间扩展

- CoT（Chain-of-Thought）推理、o1/r1 系统在封闭逻辑任务中有效
- 本文发现这些方法在多模态 ToM 中的效果受限于推理边界
- 启发：对于需要世界知识的复杂推理任务，模型规模扩展可能比推理步数扩展更关键

### 弱到强学习

- 弱到强控制思想与 weak-to-strong generalization（OpenAI, 2023）相关但不同
- 本文是将小模型的专项行为迁移给大模型，而非用弱监督训练强模型
- 启发：专项能力和通用能力的解耦与组合是一个值得深入探索的方向

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 弱到强控制机制是新颖的设计，将贝叶斯 ToM 推理扩展至 405B 规模是首次尝试
- **实验充分度**: ⭐⭐⭐⭐ — 多模态基准测试 + 未见场景验证 + 不同模型规模消融，较为充分
- **写作质量**: ⭐⭐⭐⭐ — 动机分析清晰，Fig.1 的问题可视化有说服力，方法描述层次分明
- **价值**: ⭐⭐⭐⭐ — 对多模态 ToM 推理的可扩展性研究具有重要参考价值，弱到强控制思想可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] From Black Boxes to Transparent Minds: Evaluating and Enhancing the Theory of Mind in Multimodal Large Language Models](from_black_boxes_to_transparent_minds_evaluating_and_enhancing_the_theory_of_min.md)
- [\[CVPR 2026\] Video-Only ToM: Enhancing Theory of Mind in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/video-only_tom_enhancing_theory_of_mind_in_multimodal_large_language_models.md)
- [\[CVPR 2026\] MindPower: Enabling Theory-of-Mind Reasoning in VLM-based Embodied Agents](../../CVPR2026/multimodal_vlm/mindpower_enabling_theoryofmind_reasoning_in_vlmba.md)
- [\[ICCV 2025\] ToolVQA: A Dataset for Multi-step Reasoning VQA with External Tools](../../ICCV2025/multimodal_vlm/toolvqa_a_dataset_for_multistep_reasoning_vqa_with_external.md)
- [\[NeurIPS 2025\] Can Multi-Modal LLMs Provide Live Step-by-Step Task Guidance?](../../NeurIPS2025/multimodal_vlm/can_multi-modal_llms_provide_live_step-by-step_task_guidance.md)

</div>

<!-- RELATED:END -->
