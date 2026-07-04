---
title: >-
  [论文解读] FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering
description: >-
  [ACL2025][社会计算][debiasing] 提出 FairSteer，一种推理时去偏框架，通过轻量线性分类器检测激活中的偏见信号，再用对比 prompt 对计算的去偏转向向量（DSV）动态调整隐藏层激活，无需重训即可在多任务上有效缓解 LLM 的社会偏见。 领域现状：LLM 会从训练语料中继承种族、性别、年龄等社会…
tags:
  - "ACL2025"
  - "社会计算"
  - "debiasing"
  - "fairness"
  - "activation steering"
  - "linear representation"
  - "inference-time"
---

# FairSteer: Inference Time Debiasing for LLMs with Dynamic Activation Steering

**会议**: ACL2025  
**arXiv**: [2504.14492](https://arxiv.org/abs/2504.14492)  
**代码**: [GitHub](https://github.com/LiYichen99/FairSteer)  
**领域**: 社会计算  
**关键词**: debiasing, fairness, activation steering, linear representation, inference-time

## 一句话总结

提出 FairSteer，一种推理时去偏框架，通过轻量线性分类器检测激活中的偏见信号，再用对比 prompt 对计算的去偏转向向量（DSV）动态调整隐藏层激活，无需重训即可在多任务上有效缓解 LLM 的社会偏见。

## 研究背景与动机

**领域现状**：LLM 会从训练语料中继承种族、性别、年龄等社会偏见，对弱势群体产生负面影响。去偏是 AI 对齐的重要组成部分。

**现有痛点**：Prompt 方法（如 CAL）对措辞敏感、效果不稳定；微调方法（投影、对比学习、强化学习）需要大量计算资源和标注数据，还有灾难性遗忘风险；解码策略方法（如约束搜索、重排序）会降低输出多样性且多针对旧模型。

**核心矛盾**：如何在不重训、不依赖复杂 prompt 设计的前提下，精准地在推理阶段缓解偏见，同时保持模型原有能力？

**本文目标**：设计一种推理时的去偏框架，仅在检测到偏见时进行干预，避免对无偏输出的破坏。

**切入角度**：基于线性表示假说——真值、情感、幽默等语义特征在 LLM 激活空间中编码为线性可分方向——验证公平性特征是否同样线性可分。

**核心 idea**：偏见特征在中间层激活空间中呈线性可分（>90% 分类准确率），可以通过几何干预——沿去偏方向平移激活向量——实现去偏。

## 方法详解

### 整体框架

FairSteer 分三步：(1) 训练线性分类器检测偏见激活（BAD）；(2) 用对比 prompt 对计算去偏转向向量（DSV）；(3) 推理时仅在检测到偏见时动态施加 DSV 调整激活（DAS）。

### 关键设计 1：Biased Activation Detection (BAD)

- **功能**：在每层训练一个轻量线性分类器 $C^l$，判断最后一个 token 的激活向量 $\mathbf{a}^l$ 是否对应偏见输出。
- **为什么**：需要条件性干预——只在偏见被检测到时才施加 DSV，避免破坏无偏输出。无差别施加 DSV 会导致正确率显著下降（消融实验证实）。
- **怎么做**：用 BBQ（58,492 条）+ MMLU（10,266 条）混合数据集构建训练集。模型选择刻板答案标记为偏见（y=0），选择中性答案标记为无偏（y=1）。提取各层最后 token 激活，用带 L2 正则的交叉熵损失训练 $\hat{y} = \sigma(\mathbf{w}^T \mathbf{a}^l + b)$。混入 MMLU 防止过拟合到特定偏见域。

### 关键设计 2：Debiasing Steering Vector (DSV) 计算

- **功能**：计算一个几何上可解释的干预方向，捕获偏见→无偏的激活空间偏移。
- **为什么**：PCA 可视化表明偏见和无偏激活在中间层形成清晰分离的聚类（图 3），它们之间的均值差异向量就是最自然的去偏方向。
- **怎么做**：从 BBQ 的 9 个偏见类别 + 2 个交叉偏见中各采样 10 条，共 110 条对比 prompt 对 $(\mathcal{P}^+, \mathcal{P}^-)$。$\mathcal{P}^+$ 和 $\mathcal{P}^-$ 共享相同上下文但答案选项不同（分别引导无偏和偏见响应）。DSV 计算为：$\mathbf{v}^l = \frac{1}{N}\sum[{\mathbf{a}^l(\mathcal{P}^+) - \mathbf{a}^l(\mathcal{P}^-)}]$。DSV 既编码方向（最优去偏轨迹）也编码幅度（子空间间距）。

### 关键设计 3：Dynamic Activation Steering (DAS)

- **功能**：推理时在选定层 $l^*$ 进行条件性激活调整。
- **为什么**：中间层（13-15 层）在所有测试模型上兼容性分类准确率最高，且兼顾低级 token 表示和高级语义特征。
- **怎么做**：提取输入在层 $l^*$ 的最后 token 激活 $\mathbf{a}^{l^*}$，送入分类器得到偏见概率 $\hat{y}$。若 $\hat{y} < 0.5$（检测到偏见），则施加：$\mathbf{a}^{l^*}_{\text{adj}} = \mathbf{a}^{l^*} + \mathbf{v}^{l^*}$。调整后的激活传播至后续层，引导生成走向无偏输出。

### 训练策略

- 整个框架仅需训练线性分类器（几秒级）和计算 DSV（一次前向传播 110 条样本）
- 无需修改模型参数，作为即插即用的推理插件
- 层选择：生成 2200 条 BBQ 样本评估各层分类准确率，取最高层（通常为 13-15 层）

## 实验关键数据

### 主实验：BBQ 问答去偏（部分模型）

| 模型 | 方法 | ZS Acc ↑ | ZS BS(a) ↓ | FS Acc ↑ | FS BS(a) ↓ |
|------|------|:---:|:---:|:---:|:---:|
| Llama2-13B | Base | 48.60 | 5.86 | 47.94 | 16.31 |
| | CAL | 51.29 | 1.41 | 53.27 | 9.82 |
| | **FairSteer** | **74.02** | **-0.82** | **80.26** | **1.58** |
| Llama3-8B | Base | 71.00 | 13.62 | 84.74 | 13.53 |
| | CAL | 55.51 | 0.08 | 82.65 | 2.61 |
| | **FairSteer** | **90.22** | 1.46 | **92.12** | 4.39 |
| Vicuna-13B | Base | 63.71 | 4.97 | 64.74 | 15.72 |
| | CAL | 47.99 | 0.72 | 63.72 | 12.11 |
| | **FairSteer** | **77.74** | **0.10** | **86.56** | **1.28** |

### 消融实验：BAD 的作用（BBQ）

| 模型 | 方法 | ZS Acc ↑ | FS Acc ↑ |
|------|------|:---:|:---:|
| Llama2-13B | Base | 48.60 | 47.94 |
| | DSV only | 52.84 | 55.46 |
| | **FairSteer** | **74.02** | **80.26** |
| Llama3-8B | Base | 71.00 | 84.74 |
| | DSV only | 62.21 | 74.11 |
| | **FairSteer** | **90.22** | **92.12** |
| Vicuna-7B | Base | 41.33 | 43.89 |
| | DSV only | 55.48 | 55.66 |
| | **FairSteer** | **65.38** | **71.28** |

### 通用能力保持

| 模型 | 方法 | MMLU ↑ | ARC-E ↑ | ARC-C ↑ | OBQA ↑ |
|------|------|:---:|:---:|:---:|:---:|
| Llama3-8B | Base | 68.37 | 93.56 | 83.53 | 81.60 |
| | FairSteer | 68.34 | 93.56 | 83.53 | 81.60 |
| Vicuna-13B | Base | 55.88 | 83.25 | 68.26 | 64.40 |
| | FairSteer | 55.76 | 83.25 | 68.26 | 64.40 |

### 关键发现

- FairSteer 在 BBQ 上对所有 6 个模型均大幅提升准确率（最高 Llama2-13B 零样本 +25.42），同时降低偏见分数。
- 去除 BAD 后（仅用 DSV），准确率显著下降（Llama3-8B 零样本 90.22→62.21），甚至低于基准模型，说明条件性干预至关重要——无差别施加 DSV 会误伤无偏样本。
- 通用任务上几乎无损：MMLU/ARC/OBQA 准确率变化基本在 0.5% 以内，PPL 变化也极小。
- CrowS-Pairs（反事实评估）和 CEB（开放生成）上也持续降低偏见分数，验证跨任务泛化性。
- CAL 在部分模型上去偏过度导致准确率反而低于基准（Vicuna-7B、Llama2-7B），FairSteer 更稳定。

## 亮点与洞察

1. **线性可分性验证**：首次系统验证了公平性特征在 6 个 LLM 的中间层激活空间中呈线性可分（>90%），为几何干预去偏提供了坚实的理论依据。
2. **条件性干预设计**：BAD 作为"门控"仅在偏见被检测时触发 DSV，是保持通用能力的关键。消融实验清楚展示了无条件干预的危害。
3. **极低数据需求**：DSV 仅需 110 条对比样本即可计算，远少于微调方法的数据量。
4. **即插即用**：不修改模型参数，不改变架构，不需要特殊 prompt，作为推理时插件适用于任何 Transformer LLM。

## 局限与展望

1. **线性分类器的局限**：线性分类器可能无法捕捉更复杂的非线性偏见模式，未来可探索轻量非线性探针。
2. **DSV 提取方式可能非最优**：当前直接取均值差可能不是最佳的去偏方向提取方法，论文自述这是"概念验证"而非最优技术。
3. **对比 prompt 质量依赖**：DSV 的有效性取决于对比 prompt 对的质量和代表性，可能无法覆盖所有现实偏见类型。
4. **泛化性未完全验证**：仅在 6 个开源模型（7B-13B）上测试，对更大规模模型和闭源系统的适用性未知。
5. **偏见类型覆盖**：主要聚焦 BBQ 的 9 类偏见，对更微妙、隐性的偏见（如经济地位、政治倾向）缺乏讨论。

## 相关工作与启发

### vs CAL（In-context Prompting）
CAL 通过因果引导主动学习识别偏见模式，用 in-context learning 进行去偏。但 CAL 在多个模型上表现不稳定：Vicuna-7B 上准确率甚至低于基准模型，说明 prompt 方法对模型和措辞非常敏感。FairSteer 通过直接操作激活空间，避免了 prompt 工程的不确定性，在所有测试模型上表现更稳定。

### vs Activation Steering（如 RepE / Refusal Removal）
Arditi et al. (2024) 展示了拒绝行为可通过激活转向移除，Li et al. (2024) 用线性探针检测真值方向。FairSteer 将这一范式扩展到公平性领域，并创新性地加入了 BAD 门控机制——这是关键区别。无门控的直接转向（如消融中的 DSV only）在公平性任务上效果远不如有门控版本，说明偏见去除比真值/情感等特征更需要条件性干预。

### vs 微调去偏方法
投影方法（Ravfogel et al. 2020）和对比学习方法（He et al. 2022）通过修改模型参数去偏，效果直接但代价高昂（需完整微调、大量标注数据、灾难性遗忘风险）。FairSteer 仅需 110 条样本和一个线性分类器，计算成本忽略不计，是更实用的替代方案。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将线性表示假说扩展到公平性领域并加上条件性干预门控是有意义的创新，但激活转向本身已有前例
- **实验充分度**: ⭐⭐⭐⭐⭐ — 6 个模型 × 4 个数据集 × 3 类任务，含消融、通用能力测试、分类别分析和 case study，非常全面
- **写作质量**: ⭐⭐⭐⭐ — 框架清晰，三步流程自然递进，图表丰富；但部分数学符号过于冗余
- **价值**: ⭐⭐⭐⭐ — 提供了一条实用的推理时去偏路径，即插即用特性使其有很强的工程应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Adaptive Debiasing Tsallis Entropy for Test-Time Adaptation](../../ICLR2026/social_computing/adaptive_debiasing_tsallis_entropy_for_test-time_adaptation.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](../../ACL2026/social_computing/livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[NeurIPS 2025\] Concept-Level Explainability for Auditing & Steering LLM Responses](../../NeurIPS2025/social_computing/concept-level_explainability_for_auditing_steering_llm_responses.md)
- [\[ACL 2025\] Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation](exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)
- [\[ICML 2025\] DEFAME: Dynamic Evidence-based FAct-checking with Multimodal Experts](../../ICML2025/social_computing/defame_dynamic_evidence-based_fact-checking_with_multimodal_experts.md)

</div>

<!-- RELATED:END -->
