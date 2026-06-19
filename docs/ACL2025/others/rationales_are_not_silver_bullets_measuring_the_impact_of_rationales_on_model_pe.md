---
title: >-
  [论文解读] Rationales Are Not Silver Bullets: Measuring the Impact of Rationales on Model Performance and Reliability
description: >-
  [ACL 2025 Findings][理由增强] 本文通过对 18 个数据集、7 类任务的系统实验，发现在训练数据中加入 rationale（推理过程）并非总是有益——rationale 有时会削弱模型性能，但可以提升模型可靠性（校准度），且性能和可靠性的改善呈线性相关，两者都受任务固有难度驱动。
tags:
  - "ACL 2025 Findings"
  - "理由增强"
  - "模型可靠性"
  - "校准"
  - "训练数据质量"
  - "任务难度"
---

# Rationales Are Not Silver Bullets: Measuring the Impact of Rationales on Model Performance and Reliability

**会议**: ACL 2025 Findings  
**arXiv**: [2505.24147](https://arxiv.org/abs/2505.24147)  
**代码**: [https://github.com/Ignoramus0817/rationales](https://github.com/Ignoramus0817/rationales)  
**领域**: 其他  
**关键词**: 理由增强, 模型可靠性, 校准, 训练数据质量, 任务难度

## 一句话总结

本文通过对 18 个数据集、7 类任务的系统实验，发现在训练数据中加入 rationale（推理过程）并非总是有益——rationale 有时会削弱模型性能，但可以提升模型可靠性（校准度），且性能和可靠性的改善呈线性相关，两者都受任务固有难度驱动。

## 研究背景与动机

**领域现状**：Rationale augmentation（在训练数据中加入推理过程/解释）已成为提升 LLM 能力的主流做法。无论是 chain-of-thought 数据、过程标注，还是 RLHF 中的推理反馈，"有推理过程的训练数据更好"几乎已成为社区共识。

**现有痛点**：尽管大量工作报告了 rationale 的正面效果，但这些结论通常基于少量任务的局部观察，缺乏全面系统的验证。更重要的是，几乎没有工作从**模型可靠性**（reliability）角度审视 rationale 的影响——模型不仅要回答正确，还要"知道自己什么时候可能出错"（即校准度）。

**核心矛盾**：社区对 rationale 的积极态度可能过于乐观。在不同任务类型、不同难度级别上，rationale 的效果是否真的一致？rationale 对模型的"自知之明"（校准度）又有什么影响？

**本文目标**：系统地、大规模地测量 rationale 对模型性能（准确率）和可靠性（校准度）的影响，找出潜在的规律和不一致之处。

**切入角度**：作者注意到现有研究只关注 rationale 对准确率的影响，忽略了可靠性这个同等重要的维度。他们同时引入了"任务难度"这一变量来解释不同场景下的差异化结果。

**核心 idea**：对 rationale 的效果进行全面审计——跨 7 类任务、18 个数据集测量性能和可靠性两个维度，并发现两者与任务难度之间存在线性关系。

## 方法详解

### 整体框架

整个实验框架分为四步：（1）数据准备——收集 18 个数据集覆盖 7 类 NLP 任务；（2）Rationale 合成——使用 GPT-4 为每个训练样本生成推理过程；（3）对比训练——分别在"有 rationale"和"无 rationale"两种条件下微调模型；（4）双维度评估——同时测量准确率（performance）和校准度（reliability/calibration）。

### 关键设计

1. **大规模多任务实验设计**:

    - 功能：在广泛的任务类型和难度级别上系统对比 rationale 的效果
    - 核心思路：选取 7 类任务：自然语言推理（NLI）、常识推理、阅读理解、情感分析、语义相似度、词义消歧和指代消解。每类任务包含 2-3 个代表性数据集（如 NLI 包含 SNLI、MNLI、ANLI），共计 18 个数据集。对每个数据集，分别训练"有 rationale"（question + rationale + answer）和"无 rationale"（question + answer）两个版本的模型，严格控制其他变量
    - 设计动机：之前的研究通常只在 2-3 个任务上验证结论。全覆盖 7 类任务能避免选择偏差，得到更可靠的结论

2. **Self-Consistency 校准度评估**:

    - 功能：测量模型对自身预测的置信度与实际准确率是否匹配
    - 核心思路：对每个测试问题，模型生成 10 个独立回答（通过温度采样），选出票数最多的答案作为最终预测，其出现频率作为模型置信度（confidence）。然后在不同置信度区间统计实际准确率，绘制可靠性图（reliability diagram）。校准误差通过 ECE（Expected Calibration Error）度量：$ECE = \sum_{b=1}^{B} \frac{n_b}{N} |acc(b) - conf(b)|$，其中 $b$ 是置信度桶。ECE 越低说明模型越"自知"
    - 设计动机：准确率只反映"模型答对了多少"，校准度还反映"模型知不知道自己答对了"。一个校准良好的模型在表示"很有信心"时确实大概率正确，在表示"不确定"时也真的容易出错——这对实际部署至关重要

3. **任务难度-效果线性关系分析**:

    - 功能：揭示 rationale 效果差异的深层原因
    - 核心思路：将每个数据集的固有难度（用无 rationale 模型的准确率近似）作为自变量，将 rationale 带来的性能/可靠性变化作为因变量，拟合线性回归模型。发现两者之间存在显著的线性相关：简单任务上 rationale 容易带来负面效果，中等难度任务上 rationale 收益最大，极难任务上 rationale 效果趋于消失。性能变化和可靠性变化之间也呈线性正相关
    - 设计动机：不同论文报告 rationale 效果不一致的根本原因在于它们测试的任务难度不同。线性关系为"何时使用 rationale"提供了定量决策依据

### 损失函数 / 训练策略

微调采用标准的 causal LM 交叉熵损失。有 rationale 版本的训练格式为"Question: {q} Rationale: {r} Answer: {a}"（loss 只计算 Answer 部分），无 rationale 版本为"Question: {q} Answer: {a}"。基础模型使用 Llama2-7B 和 Llama2-13B，训练时采用 LoRA 微调以控制实验成本。每个配置训练 3 个 epoch，学习率 5e-6。

## 实验关键数据

### 主实验（Rationale 对性能的影响）

| 任务类别 | 数据集 | 无 Rationale Acc | 有 Rationale Acc | 变化 |
|---------|--------|-----------------|-----------------|------|
| NLI | SNLI | 89.2 | 88.5 | -0.7 ↓ |
| NLI | MNLI | 83.6 | 84.1 | +0.5 ↑ |
| NLI | ANLI-R3 | 52.3 | 55.8 | +3.5 ↑ |
| 常识推理 | WinoGrande | 72.8 | 71.5 | -1.3 ↓ |
| 常识推理 | HellaSwag | 78.4 | 79.2 | +0.8 ↑ |
| 阅读理解 | BoolQ | 86.1 | 85.3 | -0.8 ↓ |
| 情感分析 | SST-2 | 94.7 | 93.9 | -0.8 ↓ |

### Rationale 对可靠性（ECE）的影响

| 任务类别 | 数据集 | 无 Rationale ECE ↓ | 有 Rationale ECE ↓ | 变化 |
|---------|--------|-------------------|-------------------|------|
| NLI | SNLI | 6.8 | 5.2 | -1.6 ✓ |
| NLI | ANLI-R3 | 15.3 | 11.7 | -3.6 ✓ |
| 常识推理 | WinoGrande | 12.4 | 10.1 | -2.3 ✓ |
| 情感分析 | SST-2 | 3.1 | 3.8 | +0.7 ✗ |
| 阅读理解 | BoolQ | 8.5 | 7.2 | -1.3 ✓ |

### 关键发现

- **发现 1：Rationale 有时会损害性能**。在 18 个数据集中约 1/3 的场景下，加入 rationale 训练反而使准确率下降。尤其在简单任务（准确率 >85%）上，rationale 几乎总是带来负面影响
- **发现 2：Rationale 通常改善可靠性**。即使在性能下降的场景中，rationale 也往往能降低 ECE 校准误差，使模型更"自知"。这说明 rationale 虽然可能引入噪声但让模型的不确定性估计更准确
- **发现 3：性能变化与可靠性变化呈线性正相关**。两者都受任务固有难度驱动：中等难度任务（准确率 55-75% 区间）同时在性能和可靠性上获益最大
- 这些发现对"何时使用 rationale"提供了实用指导：对简单任务不建议加 rationale，对中等难度任务强烈建议，对极难任务效果有限

## 亮点与洞察

- **挑战"理所当然"的社区共识非常有价值**：在 rationale/CoT 训练"必定有益"的普遍认知下，提供系统性的反例和边界条件。这种"audit"型工作在快速发展的领域中尤为重要
- **校准度视角是关键创新**：几乎所有 rationale 研究只看准确率，本文首次系统引入可靠性维度，发现了"性能可能下降但可靠性改善"这一反直觉但有实用价值的现象
- **任务难度-效果线性关系提供了定量决策工具**：不再是"该不该用 rationale"的二元问题，而是"在这个难度级别的任务上，rationale 的预期收益大概是多少"的定量预测

## 局限与展望

- 实验仅使用 Llama2 系列（7B/13B），对更大模型或不同架构的泛化性未知
- Rationale 全部由 GPT-4 合成，人工标注的高质量 rationale 效果可能不同
- "任务难度"用无 rationale 模型准确率近似，这个代理变量可能不够准确
- 作者自评"结论可能显得过时"（工作完成于 2024 年 1 月），但核心发现对于理解训练数据构造仍有持续价值
- 未来可以探索 rationale 质量（而非有/无）对效果的影响——从二元分析扩展到质量光谱分析

## 相关工作与启发

- **vs STaR/Self-Taught Reasoner**: STaR 通过自我生成 rationale 迭代训练来提升推理能力，总体报告正面结果。本文指出这些正面结果可能有任务选择偏差——如果在简单任务上测试，结论可能相反
- **vs Scaling CoT Data** (Mukherjee et al.): 该工作发现增加 CoT 数据量能持续提升性能。本文补充了另一个维度：不仅数量重要，匹配任务难度更重要
- **vs Efficient Reasoning** (最近工作): 与最近"减少不必要推理步骤"的趋势高度一致——本文从训练角度验证了"推理过程并非越多越好"

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次系统审视 rationale 对性能和可靠性的双重影响，任务难度-效果线性关系是新发现
- 实验充分度: ⭐⭐⭐⭐⭐ 18 数据集、7 类任务、双维度评估，规模大且设计严谨
- 写作质量: ⭐⭐⭐⭐ 论点清晰，数据分析详实
- 价值: ⭐⭐⭐⭐ 为训练数据构造策略提供了有价值的实证指导，信息密度高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Measuring Model Performance in the Presence of an Intervention](../../AAAI2026/others/measuring_model_performance_in_the_presence_of_an_intervention.md)
- [\[ACL 2025\] EpiCoDe: Boosting Model Performance Beyond Training with Extrapolation and Contrastive Decoding](epicode_boosting_model_performance_beyond_training_with_extrapolation_and_contra.md)
- [\[ACL 2025\] Do not Abstain! Identify and Solve the Uncertainty](do_not_abstain_identify_and_solve_the_uncertainty.md)
- [\[ACL 2025\] LaTIM: Measuring Latent Token-to-Token Interactions in Mamba Models](latim_measuring_latent_token-to-token_interactions_in_mamba_models.md)
- [\[ACL 2025\] Measuring the Effect of Transcription Noise on Downstream Language Understanding Tasks](measuring_the_effect_of_transcription_noise_on_downstream_language_understanding.md)

</div>

<!-- RELATED:END -->
