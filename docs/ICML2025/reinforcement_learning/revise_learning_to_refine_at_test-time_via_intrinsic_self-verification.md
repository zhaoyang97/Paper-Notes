---
title: >-
  [论文解读] ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification
description: >-
  [ICML2025][强化学习][LLM self-correction] 提出 ReVISE 框架，通过引入 [refine] 特殊 token 和两阶段课程学习（先学自验证、再学自纠错），使 LLM 在推理时能内省式地验证并修正自身推理轨迹，无需外部验证器或复杂 RL 训练。 核心问题：LLM 在复杂推理任务中…
tags:
  - "ICML2025"
  - "强化学习"
  - "LLM self-correction"
  - "测试时缩放"
  - "偏好学习"
  - "课程学习"
  - "self-verification"
---

# ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification

**会议**: ICML2025  
**arXiv**: [2502.14565](https://arxiv.org/abs/2502.14565)  
**代码**: [github.com/seunghyukoh/revise](https://github.com/seunghyukoh/revise)  
**作者**: Hyunseok Lee, Seunghyuk Oh, Jaehyung Kim, Jinwoo Shin, Jihoon Tack (KAIST)
**领域**: 强化学习  
**关键词**: LLM self-correction, 测试时缩放, 偏好学习, 课程学习, self-verification

## 一句话总结

提出 ReVISE 框架，通过引入 `[refine]` 特殊 token 和两阶段课程学习（先学自验证、再学自纠错），使 LLM 在推理时能内省式地验证并修正自身推理轨迹，无需外部验证器或复杂 RL 训练。

## 研究背景与动机

**核心问题**：LLM 在复杂推理任务中，早期步骤的错误会逐步累积，而模型自身检测和纠正错误的能力（self-awareness）严重不足。自回归生成的本质也限制了模型回顾和修正先前步骤的能力。

**现有方案的不足**：

**外部验证器方法**（如 Luo et al., 2024）：依赖大规模外部模型做验证并触发重新生成，计算开销大

**RL 方法**（如 SCoRe, Kumar et al., 2024）：训练不稳定、计算量巨大（约 150 万次生成/3000步），且不显式建模中间推理步骤的验证

**Self-Refine 类方法**（Madaan et al., 2023）：在复杂任务上实际性能下降，且 LLM 本身缺乏自纠错能力（Huang et al., 2024 已论证）

**关键问题**：能否让 LLM 拥有一个内部机制，显式验证自身推理过程并据此纠正错误？

## 方法详解

### 3.1 问题建模与 `[refine]` Token

给定输入 $x$，模型先生成初始输出 $y_{\text{init}} \sim \mathcal{M}(\cdot|x)$，然后预测一个验证 token $v \in \{[\text{eos}], [\text{refine}]\}$：

- 若 $v = [\text{eos}]$：模型认为答案正确，终止生成
- 若 $v = [\text{refine}]$：模型认为答案有误，继续生成修正后的推理 $y_{\text{refined}} \sim \mathcal{M}(\cdot|[\text{refine}], y_{\text{init}}, x)$

**关键优势**：可以直接获取模型对验证 token 的 softmax 概率，作为自验证置信度。

### 3.2 两阶段课程学习

两阶段都使用 SFT + DPO 联合损失，避免了不稳定的 RL 训练。

**Stage 1：学习自验证**

从初始模型 $\mathcal{M}_0$ 对每个输入采样多个回复，根据 ground-truth 区分正确/错误路径，构建偏好对：

- 正确回答 $\hat{y} = y_{\text{correct}}$ → 偏好 $(x, \hat{y} \oplus [\text{eos}], \hat{y} \oplus [\text{refine}])$
- 错误回答 $\hat{y} = y_{\text{wrong}}$ → 偏好 $(x \oplus \hat{y}, [\text{refine}], [\text{eos}])$

目标函数：

$$\mathcal{L}_{\text{verify}} = \mathcal{L}_{\text{SFT}}(\mathcal{D}_{\text{verify}}) + \lambda \mathcal{L}_{\text{Pref}}(\mathcal{D}_{\text{verify}})$$

其中 DPO 损失中的隐式奖励为 $r(x,y) = \beta \log \frac{\mathcal{M}(y|x)}{\mathcal{M}_0(y|x)}$，$\lambda = 0.1$。

**Stage 2：学习自纠错**

以 Stage 1 输出的 $\mathcal{M}_1$ 为起点，构建新的偏好数据集 $\mathcal{D}_{\text{correct}}$：

- 正确回答：同 Stage 1，鼓励 `[eos]`
- 错误回答 $\hat{y} = y_{\text{wrong}}$：正样本为 $[\text{refine}] \oplus y$（拼接 ground-truth），负样本为 $[\text{eos}]$

$$\mathcal{L}_{\text{correct}} = \mathcal{L}_{\text{SFT}}(\mathcal{D}_{\text{correct}}) + \lambda \mathcal{L}_{\text{Pref}}(\mathcal{D}_{\text{correct}})$$

课程学习的关键：将自验证和自纠错解耦为两阶段，避免两个任务目标冲突。

### 3.3 验证置信度感知采样

推理时利用 `[eos]` token 的 softmax 概率 $c_i = \mathcal{M}([\text{eos}]|y_i, x)$ 作为置信度，替代传统多数投票中的等权计数：

$$y^* = \arg\max_{y \in \mathcal{Y}} \sum_{i: y_i = y} c_i$$

即对相同答案的置信度求和，选择累积置信度最高的答案。

## 实验设置与主要结果

**模型**：Llama-3.2-1B、Llama-3.1-8B（非指令微调版本）
**数据集**：GSM8K（训练集 8.8K）、MATH（用 MetaMath 50K 子集训练）、MBPP（GPT-4o 生成 CoT）
**基线**：SFT、RFT、STaR+（STaR + SFT 数据）、DPO、SCoRe
**训练**：AdamW，lr ∈ {1e-4, 1e-5}，cosine decay，1 epoch

### 主实验（Table 1）

| 方法 | Llama-3.2-1B GSM8K Maj@1/5 | MATH-500 Maj@1/5 | Llama-3.1-8B GSM8K Maj@1/5 | MATH-500 Maj@1/5 |
|------|:---:|:---:|:---:|:---:|
| Few-shot CoT | 5.7 / 7.2 | 3.0 / 3.2 | 56.7 / 58.3 | 23.4 / 23.2 |
| SFT | 22.1 / 26.4 | 10.4 / 11.4 | 58.2 / 64.8 | 27.8 / 33.2 |
| RFT | 26.2 / 28.6 | 12.6 / 12.8 | 58.9 / 65.3 | 30.8 / 35.6 |
| STaR+ | 26.2 / 29.9 | 11.4 / 13.4 | 59.2 / 64.9 | 30.4 / 32.8 |
| **ReVISE** | **28.1 / 32.8** | **13.4 / 14.8** | **61.6 / 69.2** | **33.6 / 37.6** |

### 代码任务 MBPP（Table 2, Llama-3.2-1B）

| 方法 | Pass@1 |
|------|:---:|
| Few-shot CoT | 24.5 |
| SFT | 30.0 |
| STaR+ | 30.7 |
| **ReVISE** | **33.1** |

### 指令微调模型上的效果（Table 3, Llama-3.2-1B-Instruct）

| 方法 | GSM8K | GSM240K |
|------|:---:|:---:|
| Zero-shot CoT | 48.6 | 48.6 |
| SFT | 41.9 | 54.8 |
| RFT | 44.0 | 50.9 |
| **ReVISE** | **52.3** | **59.4** |

注：SFT/RFT 在指令微调模型上反而低于 zero-shot CoT（灾难性遗忘），ReVISE 因为只在第二次尝试中使用 gold label 而非直接微调，避免了此问题。

### 与 SCoRe 对比（Table 6, Gemma-2-2B, MATH-500）

| 方法 | 准确率 | 训练效率 |
|------|:---:|:---:|
| SCoRe | 23.0% | ×1 |
| ReVISE | 23.2% | **×30 更高效** |
| ReVISE + iter2 | 25.8% | ×15 更高效 |

ReVISE 每个样本只需生成 1 条推理路径（共 50K 次），SCoRe 需约 150 万次生成。

### 验证能力量化（Table 7, Llama-3.2-1B, GSM8K AUROC）

| 方法 | AUROC |
|------|:---:|
| V-STaR 外部验证器 | 69.5% |
| **ReVISE 内部验证** | **76.0%** |

### 关键消融实验

- **课程学习有效性**：无课程 22.6% → Stage 1 only ~26% → ReVISE 完整 28.1%（GSM8K Maj@1）
- **DPO 损失关键性**：去掉 DPO 后性能下降 10.3%
- **迭代修正**：可支持多轮 refine（1→2→3 次），MATH-500 上 8B 模型精度持续提升
- **跨域泛移**：MATH 上训练 → GSM8K 评估，8B 模型 ReVISE 61.5% > SFT 60.3%

## 亮点与洞察

1. **设计极简但有效**：仅增加一个 `[refine]` 特殊 token，用偏好学习替代 RL，实现了自验证+自纠错双能力
2. **训练效率极高**：相比 SCoRe 减少 30× 训练计算量，同等性能下效率优势显著
3. **内建置信度信号**：`[eos]` 概率天然作为验证置信度，无需额外验证器训练
4. **对指令微调友好**：避免灾难性遗忘——因为 gold label 仅作为"修正后的第二次尝试"而非直接 SFT 目标
5. **自然支持 test-time scaling**：refine 机制本身就是计算扩展，配合置信度加权投票进一步增益

## 局限与展望

1. **Stage 2 训练导致验证能力轻微退化**：AUROC 从 Stage 1 的最优值略有下降，存在灾难性遗忘问题
2. **纠错数据依赖 ground-truth**：Stage 2 正样本直接拼接 ground-truth label，限制了在无标注场景下的应用
3. **仅验证最终答案正确性**：不做中间步骤级别的验证（process-level verification），可能在更长推理链上效果有限
4. **实验模型规模有限**：仅测试了 1B 和 8B 模型，未验证在更大模型或 reasoning-specific 模型上的效果
5. **单次修正训练**：虽然推理时可多次 refine，但训练仅针对单次修正，多轮 refine 的训练方案待探索
6. **基准任务相对简单**：GSM8K/MATH-500 是数学推理标准任务，未在更具挑战性的竞赛题或多模态推理上测试

## 相关工作与启发

- **Backtracking (Zhang et al., 2024c)**：类似地引入 reset token，但聚焦安全场景而非推理纠错
- **STaR / V-STaR**：自我改进范式但不做显式验证
- **SCoRe (Kumar et al., 2024)**：RL-based self-correction，效果相当但训练成本高 30×
- **Self-Refine (Madaan et al., 2023)**：推理时迭代反馈，但在复杂任务上反而降低性能
- **启发**：`[refine]` token 的设计思路可推广到其他需要"自我审查"的场景（安全、代码审查），课程学习拆解复杂能力的训练策略值得借鉴

## 评分

- 新颖性: ⭐⭐⭐⭐ — `[refine]` token + 两阶段课程学习的组合设计简洁而有效，将自验证和自纠错用偏好学习统一建模是很好的思路
- 实验充分度: ⭐⭐⭐⭐ — 消融全面（课程学习、DPO、置信度采样、迭代修正、跨域泛化、指令微调），与 SCoRe 等强基线对比，但缺少更大模型和更难任务的验证
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，公式表达规范，图示直观
- 价值: ⭐⭐⭐⭐ — 提供了一种比 RL 更高效的自纠错训练范式，实用性强，但在 o1/DeepSeek-R1 等强推理模型时代，方法的竞争力还需进一步验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Test-Time Adaptation with Binary Feedback](test-time_adaptation_with_binary_feedback.md)
- [\[ICLR 2026\] Self-Harmony: Learning to Harmonize Self-Supervision and Self-Play in Test-Time Reinforcement Learning](../../ICLR2026/reinforcement_learning/self-harmony_learning_to_harmonize_self-supervision_and_self-play_in_test-time_r.md)
- [\[NeurIPS 2025\] Reinforcement Learning Teachers of Test Time Scaling](../../NeurIPS2025/reinforcement_learning/reinforcement_learning_teachers_of_test_time_scaling.md)
- [\[ICML 2025\] Optimizing Language Models for Inference Time Objectives using Reinforcement Learning](optimizing_language_models_for_inference_time_objectives_using_reinforcement_lea.md)
- [\[ICML 2025\] Learning Progress Driven Multi-Agent Curriculum](learning_progress_driven_multi-agent_curriculum.md)

</div>

<!-- RELATED:END -->
