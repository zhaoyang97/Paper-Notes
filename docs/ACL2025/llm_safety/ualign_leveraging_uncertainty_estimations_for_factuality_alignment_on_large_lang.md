---
title: >-
  [论文解读] UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models
description: >-
  [ACL2025][LLM安全][不确定性估计] 提出 UAlign 框架，利用置信度分数和语义熵两种不确定性估计来显式建模 LLM 知识边界，并将其作为输入特征融入 PPO 对齐训练，使模型自信回答已知问题、坚定拒绝未知问题，在多个知识 QA 数据集上显著提升可靠性与泛化性。 LLM 在预训练阶段学到了大量知识…
tags:
  - "ACL2025"
  - "LLM安全"
  - "不确定性估计"
  - "事实性对齐"
  - "知识边界"
  - "PPO"
  - "语义熵"
  - "置信度"
---

# UAlign: Leveraging Uncertainty Estimations for Factuality Alignment on Large Language Models

**会议**: ACL2025  
**arXiv**: [2412.11803](https://arxiv.org/abs/2412.11803)  
**代码**: [AmourWaltz/UAlign](https://github.com/AmourWaltz/UAlign)  
**领域**: LLM安全  
**关键词**: 不确定性估计, 事实性对齐, 知识边界, PPO, 语义熵, 置信度

## 一句话总结

提出 UAlign 框架，利用置信度分数和语义熵两种不确定性估计来显式建模 LLM 知识边界，并将其作为输入特征融入 PPO 对齐训练，使模型自信回答已知问题、坚定拒绝未知问题，在多个知识 QA 数据集上显著提升可靠性与泛化性。

## 研究背景与动机

LLM 在预训练阶段学到了大量知识，但在下游任务中经常无法准确表达它所掌握的事实知识。
核心问题在于 LLM 的 **知识边界模糊**，具体体现在三个层面：

**弱已知知识被丢弃**：模型对某些问题实际上"知道"但不确定，多次采样中只有部分回答正确。先前方法（如 R-Tuning）会直接将这类问题标记为"未知"并训练模型拒绝，导致本可正确回答的知识被浪费。

**未知知识被过度自信地回答**：模型对完全不熟悉的问题也会生成看似合理的答案，造成严重的幻觉问题，损害用户信任。

**现有对齐方法的不足**：先前的事实性对齐工作（R-Tuning、RLKF、RL-DPO）没有显式利用知识边界信息。它们要么仅做已知/未知二分类，要么通过知识探测间接估计，均未将不确定性度量作为模型的直接输入。

UAlign 的核心洞察是：如果能显式量化 LLM 对每个问题的不确定性，并将这些信息作为额外输入特征融入对齐训练，就能让模型更好地理解自身的知识边界。
这相当于在 prompt 中加入"自信程度"和"答案分散度"提示，帮助模型做出更审慎的决策——对已知问题大胆回答，对未知问题果断拒绝。

## 方法详解

### 整体框架

UAlign 分为两大阶段：

- **阶段一：数据集准备** — 对知识 QA 数据集进行多次采样，计算置信度和语义熵
- **阶段二：UAlign 训练** — 先 SFT 训练不确定性估计模型和奖励模型，再用 PPO 进行策略模型对齐

### 阶段一：数据集准备

**多次采样策略**：
对数据集中的每个问题，使用 K=10 个不同的 1-shot prompt 模板、采样温度 T=0.2 重复生成。
每次采样得到一个候选答案，与标准答案比较后标注正确性。
若所有 K 次采样全部错误，则将该问题归为"未知"，标准答案改写为拒绝回复 "Sorry, I don't know."。

**不确定性度量 1 — 置信度分数（Confidence Score）**：
定义为 K 次采样中正确答案的比例，反映模型对该问题的"答对概率"。
直觉上，某个问题的置信度越高，LLM 对该知识越确定。

**不确定性度量 2 — 语义熵（Semantic Entropy）**：
先用 NLI 模型将语义等价的回答聚类到同一个语义集合，然后计算聚类分布的熵。
语义熵衡量的是生成答案在语义层面的分散程度——即使置信度低，如果所有答案都集中在少数语义上，熵也会较低。

**两种度量的互补性**：
置信度衡量"模型多大概率答对"，语义熵衡量"模型的回答在语义上有多分散"。
关键场景：某问题的置信度仅 40%（正确率低），但语义熵很高（其他答案更分散），此时正确答案虽不占优势但仍是最集中的，模型应被引导输出该答案而非拒绝。

### 阶段二：UAlign 训练

**SFT 子阶段 — 训练估计模型和奖励模型**：

- **不确定性估计模型**（预测置信度和语义熵）：以 vanilla LLM 为底座，LoRA rank=4 微调。输入仅为问题，目标为预测对应的置信度或语义熵值。
- **奖励模型**：同样以 LLM 为底座，LoRA rank=4。输入为问题 + 预测的两个不确定性值 + 候选答案，输出为正确性判定概率，使用二元交叉熵损失训练。

关键设计：奖励模型的输入显式包含了不确定性估计，使其能利用知识边界信息来更准确地判断答案质量。

**PPO 子阶段 — 策略模型对齐**：

- 策略模型的输入为：问题 + 预测的置信度 + 预测的语义熵
- 参考模型的输入为：仅问题（无不确定性信息）
- 奖励函数包含两部分：奖励模型的评分信号和 KL 散度惩罚项
- 通过 PPO 最大化该奖励，引导策略模型根据知识边界信息生成更事实性的回答

所有 LLM 均使用 LoRA（rank=16）微调，在 4x NVIDIA A100-40GB 上训练。

## 实验关键数据

### 实验设置

- **模型**：Llama-3-8B、Mistral-7B
- **训练集**：TriviaQA (TVQA)、SciQ、NQ-Open 三个知识 QA 数据集
- **测试集**：上述三个数据集的验证/测试集（ID）+ LSQA 多语言 QA 数据集（OOD）
- **评价指标**：Precision（已知问题中正确回答的比例）、Truthfulness（正确回答已知 + 正确拒绝未知 的总比例）

### 主实验结果（Table 1，Llama-3-8B）

| 方法 | TVQA Prec. | TVQA Truth. | SciQ Prec. | NQ Prec. | Avg ID Prec. | LSQA OOD Prec. |
|------|-----------|------------|-----------|---------|-------------|---------------|
| ICL | 76.15 | 56.55 | 70.43 | 50.28 | 65.62 | 77.35 |
| R-Tuning | 72.93 | 55.44 | 71.38 | 47.81 | 64.04 | 71.54 |
| RL-PPO | 76.32 | 55.19 | 75.70 | 54.07 | 68.03 | 72.18 |
| RLKF | 77.12 | 56.07 | 72.36 | 54.86 | 68.11 | 74.95 |
| **UAlign** | **79.14** | **57.04** | **76.44** | **56.60** | **70.72** | **79.56** |

在 Mistral-7B 上，UAlign 在 TVQA 上达到 Prec. 82.10、Truth. 59.05，同样全面领先。
值得注意的是，多数训练方法在 OOD 数据集上性能下降，但 UAlign 在 LSQA 上仍超越所有方法（含 prompt-based 基线）。

### 消融实验：不确定性度量对奖励模型准确率的影响（Table 2）

| 置信度 | 语义熵 | TVQA | SciQ | NQ-Open | LSQA (OOD) |
|--------|--------|------|------|---------|-----------|
| x | x | 82.31 | 79.00 | 67.45 | 70.12 |
| o | x | 85.41 | 84.30 | 70.37 | 75.09 |
| x | o | 82.05 | 77.90 | 67.85 | 70.40 |
| o | o | **86.73** | **86.40** | **72.00** | **74.59** |

以上为 Llama-3-8B 结果。关键发现：

- 置信度贡献最大，单独加入可提升奖励模型准确率 3-5 个百分点
- 语义熵单独使用效果不稳定，在部分数据集上甚至轻微下降
- 两者联合使用在大多数设置下达到最优

### 采样次数 K 的影响

K 从 1 增至 4、7、10 时，Prec. 和 Truth. 持续提升但增幅递减。
K=10 时性能基本收敛，进一步增加 K 的边际收益有限。
在 4x A100 上对 10000 个 QA 样本进行 K=10 次采样的时间成本可控（答案为实体级短文本）。

## 亮点

- **新颖的显式知识边界建模**：首次将不确定性估计作为显式 prompt 输入融入 RLHF 对齐流程，思路直觉且有效
- **互补度量的精巧设计**：置信度（"答对概率"）+ 语义熵（"答案分散度"）联合使用，能挽救低置信但正确的弱已知知识
- **OOD 泛化性突出**：UAlign 是唯一在 LSQA (OOD) 上持续超越 prompt-based 基线的训练方法
- **与 Test-Time Scaling 的关联**：多次采样后计算不确定性再指导对齐的流程，与推理时计算量分配趋势相呼应

## 局限性

- **任务范围窄**：仅在短答案知识 QA 上验证，未扩展到开放式生成、长文本写作或推理任务
- **依赖标准答案**：置信度计算需要 ground-truth 标签进行采样答案的正误比较，难以直接迁移到无标注场景
- **计算成本线性增长**：数据集构建需 K 次采样，成本随 K 和数据规模线性增长
- **语义熵的不稳定性**：消融实验显示语义熵单独使用时效果波动，在部分数据集上甚至略微降低性能

## 与相关工作的对比

- **R-Tuning**：通过采样判定已知/未知后做 SFT，不使用 RL，无显式知识边界输入
- **RLKF**：用知识探测和一致性检查训练 reward model 后做 PPO，知识边界信息仅隐式体现在奖励信号中
- **RL-DPO**：构建事实性偏好对做 DPO 对齐，不涉及不确定性估计
- **ITI**：推理时干预注意力头激活，无需训练但效果有限
- **UAlign**：将置信度和语义熵显式作为 prompt 输入传递给奖励模型和策略模型，是区别于所有先前方法的核心创新

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将不确定性估计显式融入对齐流程的思路新颖，但各组件（置信度、语义熵、PPO）均为已有技术的组合
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 2 个模型、4 个数据集、多种基线和详细消融；缺少开放式生成任务验证
- 写作质量: ⭐⭐⭐⭐ — 逻辑清晰、图表丰富，知识边界的可视化解释直观易懂
- 价值: ⭐⭐⭐⭐ — 为事实性对齐提供了新视角，但实际应用受限于短答案 QA 场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Chinese SimpleQA: A Chinese Factuality Evaluation for Large Language Models](chinese_simpleqa_a_chinese_factuality_evaluation_for_large_language_models.md)
- [\[ACL 2026\] FAITH: Factuality Alignment through Integrating Trustworthiness and Honestness](../../ACL2026/llm_safety/faith_factuality_alignment_through_integrating_trustworthiness_and_honestness.md)
- [\[ACL 2026\] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models](../../ACL2026/llm_safety/learning_uncertainty_from_sequential_internal_dispersion_in_large_language_model.md)
- [\[ACL 2025\] ComparisonQA: Evaluating Factuality Robustness of LLMs Through Knowledge Frequency Control and Uncertainty](comparisonqa_evaluating_factuality_robustness_of_llms_through_knowledge_frequenc.md)
- [\[ACL 2025\] Unveiling and Addressing Pseudo Forgetting in Large Language Models](unveiling_and_addressing_pseudo_forgetting_in_large_language_models.md)

</div>

<!-- RELATED:END -->
