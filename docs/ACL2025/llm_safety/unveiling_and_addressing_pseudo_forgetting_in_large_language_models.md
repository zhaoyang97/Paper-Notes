---
title: >-
  [论文解读] Unveiling and Addressing Pseudo Forgetting in Large Language Models
description: >-
  [ACL2025][LLM安全][continual learning] 揭示 LLM 持续学习中的"伪遗忘"现象：性能下降并非因为模型丧失了旧任务能力，而是指令无法正确激活已有能力。通过归因分析证明遗忘模型的指令依赖度降低，并提出基于 Rationale-Guidance Difficulty（RGD）的动态数据回放框架 RGD-R 来缓解伪遗忘。
tags:
  - "ACL2025"
  - "LLM安全"
  - "continual learning"
  - "catastrophic forgetting"
  - "pseudo forgetting"
  - "instruction dependence"
  - "replay-based learning"
---

# Unveiling and Addressing Pseudo Forgetting in Large Language Models

**会议**: ACL2025  
**arXiv**: [2411.11932](https://arxiv.org/abs/2411.11932)  
**代码**: [GitHub](https://github.com/BIT-NLP/Pseudo-Forgetting)  
**领域**: LLM安全  
**关键词**: continual learning, catastrophic forgetting, pseudo forgetting, instruction dependence, replay-based learning

## 一句话总结

揭示 LLM 持续学习中的"伪遗忘"现象：性能下降并非因为模型丧失了旧任务能力，而是指令无法正确激活已有能力。通过归因分析证明遗忘模型的指令依赖度降低，并提出基于 Rationale-Guidance Difficulty（RGD）的动态数据回放框架 RGD-R 来缓解伪遗忘。

## 研究背景与动机

**领域现状**：持续学习使 LLM 能增量学习多个任务，但灾难性遗忘（catastrophic forgetting）仍是核心痛点——学习新任务后在旧任务上性能显著下降。学界已提出大量缓解方法（正则化、架构扩展、数据回放），但对遗忘机制本身的理解仍然不足。

**现有痛点**：Kotha 等提出"任务推断"假设（微调偏向新能力而非丧失旧能力），但仅在合成数据和小模型上验证；Jiang 等认为遗忘源于指令跟随能力下降而非知识丢失，但使用了不一致的实验设置（训练用指令跟随，探测用前缀补全），削弱了结论说服力。

**核心矛盾**：现有工作缺乏直接、有力的经验证据证明 LLM 在自然语言任务上的遗忘是"伪"的，即模型仍保有旧任务能力但无法被正确激活。

**本文目标**：(1) 直接证明 LLM 持续学习中存在伪遗忘；(2) 分析伪遗忘的内在原因；(3) 提出量化指标和缓解方案。

**切入角度**：通过两种干预手段（提供部分正确 rationale、搜索无意义后缀）来恢复遗忘模型在旧任务上的表现，证明能力仍在；再用归因分析揭示遗忘模型对指令的依赖度降低。

**核心 idea**：灾难性遗忘中的性能下降很大程度上是"伪遗忘"——模型能力未丧失，只是原始指令无法有效激活它，通过强化指令依赖即可恢复。

## 方法详解

### 整体框架

论文分为两大部分：(1) 揭示伪遗忘（Section 2）——通过两个探测实验证明遗忘模型仍保有能力，并用归因分析揭示原因；(2) 解决伪遗忘（Section 3）——提出 RGD 指标量化伪遗忘程度，并基于此设计 RGD-R 动态回放框架。

### 关键设计 1：被动恢复实验（External Rationale Guidance）

- **功能**：给遗忘模型提供 ground truth rationale 的前 $k$ 比例（$k \in [0,1]$），观察任务性能是否恢复。
- **为什么**：如果模型真的丧失了旧任务能力，仅提供少量 rationale 前缀不应能恢复性能；反之，若少量引导即可恢复，说明能力仍在模型参数中。
- **怎么做**：在 `<|assistant|>` 后追加 ground truth rationale 的前 $k$ 部分。实验表明当 $k \leq 0.2$ 时，提供的内容不包含直接的任务关键信息。结果：Llama2-13B 在 RTE 任务上 $k=0.3$ 即可恢复到遗忘前水平，模型越大/任务越简单需要的引导越少。

### 关键设计 2：主动恢复实验（GCG Suffix Search）

- **功能**：使用 Greedy Coordinate Gradient（GCG）搜索一个语义无意义的后缀，附加到原始指令后，帮助遗忘模型自主生成正确 rationale。
- **为什么**：如果语义无关的后缀就能让模型恢复表现，那就排除了"模型靠外部信息补全"的解释，直接证明能力未被遗忘。
- **怎么做**：优化目标为最小化 $\mathcal{L}(S) = -\log p(T|[I,S])$，其中 $T$ 是部分正确 rationale（前 20%）。对每个遗忘样本搜索独立后缀。结果：各任务恢复率（Recovery Rate）均超 90%，部分达到 100%。

### 关键设计 3：归因分析（Attribution Analysis）

- **功能**：用归因分数（attribution scores）量化模型在生成 rationale 时对指令部分的依赖程度。
- **为什么**：揭示伪遗忘的根本原因——不是知识丢失，而是指令依赖度降低。
- **怎么做**：计算指令 $I$ 与 rationale $R$ 之间的依赖分数 $Q^{(l)}_{IR}$（逐层），对比遗忘前后模型的差异、以及遗忘模型生成正确 vs 错误 rationale 时的差异。结果：遗忘模型生成错误 rationale 时，浅层的指令依赖度显著低于生成正确 rationale 时。

### 关键设计 4：RGD 指标与 RGD-R 框架

- **功能**：定义 Rationale-Guidance Difficulty (RGD) 指标衡量伪遗忘程度，基于此设计动态回放数据分配策略。
- **为什么**：等量回放（每个旧任务回放相同数据量）效率不高，应根据各任务的实际遗忘程度动态分配。
- **怎么做**：
    - RGD 定义：$\text{RGD}(I, R_g, A_g) = \frac{\text{PPL}_{a\text{-}f}(R_g|I)}{\text{PPL}_{b\text{-}f}(R_g)}$，即遗忘后模型在指令引导下生成正确 rationale 的困惑度与遗忘前的比值。RGD 越高 → 伪遗忘越严重。
    - 动态分配：第 $j$ 个旧任务的回放数据比例 $\alpha_j = \frac{\text{RGD}_{D_j}}{\sum_{k=1}^{i-1} \text{RGD}_{D_k}}$，遗忘越严重的任务获得更多回放数据。

### 损失函数/训练策略

采用标准的指令微调损失，在回放数据上通过 RGD 动态调整各旧任务的数据配比。训练遵循顺序学习范式：每学一个新任务时，根据 RGD 分数计算旧任务回放量。

## 实验关键数据

### 主实验：Long Sequence Benchmark（Table 1）

| 模型 | 方法 | FAP↑ | F.Ra↓ | BWT↑ | FWT↑ |
|------|------|------|-------|------|------|
| Qwen2-0.5B | SEQ | 20.73 | 53.18 | -53.04 | 21.46 |
| Qwen2-0.5B | EA | 64.13 | 5.43 | -4.90 | 33.34 |
| Qwen2-0.5B | **RGD-R** | **65.99** | **3.64** | **-3.29** | 31.87 |
| Mistral-7B | SEQ | 51.48 | 30.19 | -29.97 | 47.91 |
| Mistral-7B | EA | 72.15 | 7.59 | -6.96 | 51.17 |
| Mistral-7B | **RGD-R** | **74.91** | **4.37** | **-3.92** | 50.77 |
| Llama2-7B | SEQ | 62.79 | 17.87 | -17.85 | 43.95 |
| Llama2-7B | EA | 76.10 | 3.52 | -2.49 | 50.91 |
| Llama2-7B | **RGD-R** | **77.03** | **2.65** | **-1.25** | 51.06 |
| Llama2-13B | SEQ | 68.38 | 13.54 | -13.20 | 51.69 |
| Llama2-13B | EA | 76.98 | 4.73 | -3.70 | 56.92 |
| Llama2-13B | **RGD-R** | **78.25** | **3.68** | **-2.29** | 57.83 |

### 消融/对比实验：RGD-R vs InsCL（Table 3）

| 模型 | 方法 | FAP↑ | F.Ra↓ | BWT↑ | FWT↑ |
|------|------|------|-------|------|------|
| Mistral-7B | EA | 72.15 | 7.59 | -6.96 | 51.17 |
| Mistral-7B | InsCL | 76.17 | 4.43 | -4.02 | 54.08 |
| Mistral-7B | RGD-R | 74.91 | 4.37 | -3.92 | 50.77 |
| Llama2-7B | EA | 76.10 | 3.52 | -2.49 | 50.91 |
| Llama2-7B | InsCL | 76.73 | 2.78 | -1.96 | 50.25 |
| Llama2-7B | RGD-R | 77.03 | 2.65 | -1.25 | 51.06 |

### GCG 恢复率实验（Figure 5）

各模型在多个遗忘任务上的恢复率：

| 模型 | 任务 | Recovery Rate |
|------|------|--------------|
| Mistral-7B | MNLI | 100% |
| Mistral-7B | QQP | ~95% |
| Llama2-7B | RTE | ~93% |
| 所有模型 | 平均 | >90% |

### 关键发现

1. **伪遗忘真实存在**：仅提供 10-20% 的正确 rationale 前缀（不含答案信息），遗忘模型即可开始恢复表现。
2. **无意义后缀可激活能力**：语义毫无意义的 GCG 后缀恢复率超 90%，直接否定了"能力丧失"假说。
3. **指令依赖度下降是根因**：归因分析显示遗忘模型在浅层的指令依赖显著降低，导致无法正确激活内部能力。
4. **模型越大抗伪遗忘越强**：Qwen2-0.5B 的 F.Ra=53.18 vs Qwen2-7B 的 11.78，模型规模对抗伪遗忘有天然优势。
5. **RGD-R 全面优于等量回放**：在稳定性指标（FAP/F.Ra/BWT）和可塑性指标（FWT）上同时提升。
6. **等量回放已大幅缓解遗忘**：这本身就支持了伪遗忘假说——简单回放就能恢复，说明能力未真正丢失。
7. **数据回放确实恢复了指令依赖**：归因实验验证回放后模型对旧任务指令的依赖度回升。

## 亮点与洞察

1. **概念贡献大于算法贡献**：提出"伪遗忘"概念并用两套正交实验直接证明，为持续学习领域提供了新理解范式——不是"忘了什么"，而是"无法被正确调用"。
2. **实验设计精巧**：GCG 后缀实验尤为巧妙——用语义无关的扰动恢复性能，排除了外部信息补全的混淆因素，是非常有力的因果证据。
3. **理论支撑**：从信息论角度推导 RGD 可近似衡量指令激活正确能力的概率 $P_\theta(c^*|i) = \frac{P_\theta(r^*|i)}{P_\theta(r^*)}$，赋予指标理论合理性。
4. **实用且轻量**：RGD-R 无需修改模型架构或训练方式，仅调整回放数据配比，易于集成到现有持续学习流程中。
5. **多模型多任务验证**：在 Qwen2(0.5B/7B)、Mistral-7B、Llama2(7B/13B) 五个模型上验证，覆盖不同规模和架构。

## 局限与展望

1. **未分析伪遗忘的发生时间点**：在学习新任务的哪个阶段开始出现指令依赖降低？这一动态过程未被追踪。
2. **任务/领域特异性未探索**：伪遗忘是否在某些任务类型（如生成 vs 分类）上更严重？与数据分布的关系未研究。
3. **RGD 需要遗忘前模型**：计算 $\text{PPL}_{b\text{-}f}$ 需要保存遗忘前的模型快照，增加存储成本。
4. **仅测试了 replay-based 方法**：未与正则化、架构扩展等方法结合验证 RGD 的普适性。
5. **评估指标单一**：RGD 是单维度指标，论文本身承认伪遗忘的量化可能是多维问题。
6. **GCG 搜索成本高**：逐样本搜索后缀仅用于分析目的，不适合作为实际缓解方案。

## 相关工作与启发

### vs Kotha et al., 2024（Task Inference 假说）

Kotha 等提出微调偏向新任务推断而非丧失旧能力，但仅在合成数据集和小 Transformer 上验证。本文在 **自然语言数据集 + 7B/13B 规模 LLM** 上直接证明了类似假说，且通过 GCG 后缀实验提供了更强的因果证据——不仅证明了"能力还在"，还进一步定位了原因（指令依赖度降低）。

### vs Jiang et al., 2024（Instruction Vector）

Jiang 等也认为遗忘源于指令跟随能力下降，但使用训练时用指令跟随、探测时用前缀补全的不一致设置。本文全程保持指令跟随设定（包括 GCG 实验也是在指令模板内操作），实验设计的一致性和说服力更强。

### vs InsCL（Wang et al., 2024）

InsCL 基于任务指令相似度分配回放数据，RGD-R 基于模型对伪遗忘的易感程度分配。两者互补：InsCL 关注任务间关系，RGD-R 关注模型状态。实验显示两者效果可比，但 RGD-R 提供了更好的可解释性（直接对应伪遗忘程度）。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "伪遗忘"概念本身很有启发性，GCG 后缀实验设计巧妙，但核心假说（能力未丢失）已有前人提出，贡献在于提供了更直接有力的证据。
- **实验充分度**: ⭐⭐⭐⭐⭐ — 双重验证（被动+主动恢复）、归因分析、理论推导、5 个模型/多个任务、与 SOTA 对比，实验链条非常完整。
- **写作质量**: ⭐⭐⭐⭐ — 逻辑链清晰（证据→原因→度量→解决），但公式较多，部分符号可简化。
- **价值**: ⭐⭐⭐⭐ — 对持续学习领域的理解有重要推动，RGD-R 框架实用轻量，但算法改进幅度不算特别大（相对于概念贡献）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Exploring Forgetting in Large Language Model Pre-Training](exploring_forgetting_in_large_language_model_pre-training.md)
- [\[ACL 2025\] Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)
- [\[ACL 2025\] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models](ualign_leveraging_uncertainty_estimations_for_factuality_alignment_on_large_lang.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ACL 2025\] Improved Unbiased Watermark for Large Language Models](improved_unbiased_watermark_for_large_language.md)

</div>

<!-- RELATED:END -->
