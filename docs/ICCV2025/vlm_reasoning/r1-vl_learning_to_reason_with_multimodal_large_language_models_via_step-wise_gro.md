---
title: >-
  [论文解读] R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization
description: >-
  [多模态VLM] 提出 StepGRPO，一种新的在线强化学习框架，通过两种无需过程奖励模型的规则化步级推理奖励（StepRAR 步级推理准确性奖励 + StepRVR 步级推理有效性奖励），解决 MLLM 在 RL 训练中的稀疏奖励问题，使 MLLM 能够自主探索和改进推理能力。 当前增强 MLLM 推理能力的主流方法是在…
tags:
  - "多模态VLM"
---

# R1-VL: Learning to Reason with Multimodal Large Language Models via Step-wise Group Relative Policy Optimization

## 论文信息
- **会议**: ICCV 2025
- **arXiv**: [2503.12937](https://arxiv.org/abs/2503.12937)
- **代码**: [链接见论文](https://arxiv.org/abs/2503.12937)
- **领域**: Multimodal VLM / MLLM推理
- **关键词**: MLLM推理, 在线强化学习, step-wise奖励, GRPO, 稀疏奖励问题
- **作者**: Jingyi Zhang, Jiaxing Huang, Huanjin Yao, Shunyu Liu, Xikun Zhang, Shijian Lu, Dacheng Tao (NTU Singapore)

## 一句话总结

提出 StepGRPO，一种新的在线强化学习框架，通过两种无需过程奖励模型的规则化步级推理奖励（StepRAR 步级推理准确性奖励 + StepRVR 步级推理有效性奖励），解决 MLLM 在 RL 训练中的稀疏奖励问题，使 MLLM 能够自主探索和改进推理能力。

## 研究背景与动机

当前增强 MLLM 推理能力的主流方法是在高质量 CoT 数据上做 SFT（如 Mulberry），但 SFT 仅让模型**被动模仿**正确推理路径，不理解错在哪里。

受 DeepSeek-R1 启发，自然想到用 GRPO 的在线 RL 方式让 MLLM 自主探索。但直接应用 outcome-level reward 到 MLLM 存在**稀疏奖励问题**：
- MLLM（尤其小模型）长链推理准确率低
- 很少有推理路径能获得正奖励
- 正反馈信号不足 → 探索效率低 → 训练不稳定

核心需求：在结果级奖励之外，提供更密集、细粒度的步级奖励来指导学习。

## 方法详解

### 整体框架

StepGRPO 分两个阶段：
1. **Policy Warm-up**：在 CoT 数据上 SFT，赋予基础推理能力
2. **Step-wise Online Policy Optimization**：在线 RL，自我探索 + 步级奖励优化

### Step-wise Reasoning Accuracy Reward (StepRAR)

评估推理路径是否包含必要的中间推理步骤。

**关键步骤提取**：
- 从 CoT 数据的标准推理路径中预提取关键步骤 $\mathbf{v} = \{v_1, v_2, ...\}$
- 仅保留核心数值/方程等几个关键词
- 增强多种等价格式（如 $\frac{6}{3}=2$ → "6/3 = 2" → "6 divided by 3 equals 2"）

**软匹配计算**：
$$k^i = |\mathbf{v}_{match}| / |\mathbf{v}|$$

**StepRAR 定义**：
$$r_{auc}^i = \begin{cases} 1 + \alpha k^i, & \text{答案正确} \\ \alpha k^i, & \text{有答案但错误} \\ 0, & \text{无答案} \end{cases}$$

即使最终答案错误，只要包含正确的中间步骤也能获得奖励（$\alpha k^i$），缓解稀疏奖励。

### Step-wise Reasoning Validity Reward (StepRVR)

评估推理路径的结构完整性和逻辑一致性，基于两个规则：
- **推理完整性** $\delta^c$：必须包含背景分析、逐步推理过程、最终答案三部分
- **推理逻辑性** $\delta^l$：背景分析在推理步骤之前，最终答案在推理步骤之后

$$r_{val}^i = \begin{cases} 1, & \mathbb{I}(\delta^c) \cdot \mathbb{I}(\delta^l) = 1 \\ 0, & \text{otherwise} \end{cases}$$

### 策略优化

总奖励 $r^i = r_{auc}^i + r_{val}^i$，组内相对优势：

$$\hat{A}^i = \frac{r^i - \text{mean}(\{r^1, ..., r^M\})}{\text{std}(\{r^1, ..., r^M\})}$$

RL 目标（带 KL 散度正则化）：

$$\mathcal{L}_{StepRL} = -\mathbb{E}\left[ \frac{1}{M} \sum_{i=1}^M \frac{\pi_\theta(\mathbf{c}^i|Q)}{[\pi_\theta(\mathbf{c}^i|Q)]_{\text{no grad}}} \hat{A}^i - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right]$$

### 训练细节

- 基础模型：Qwen2-VL-2B 和 Qwen2-VL-7B
- Warm-up：Mulberry-260k 数据 SFT
- RL：从 Mulberry-260k 随机采 10K 数据
- 每问 4 rollouts（$M=4$），温度 1.2，最大长度 1024
- 4× H100-80GB，lr=1e-6，$\alpha=0.1$，$\beta=0.04$

## 实验关键数据

### 主实验结果（Table 1：8 个基准平均）

| 模型 | MathVista | MMStar | Math-V | ChartQA | DynaMath | HallBench | MathVerse | AVG |
|------|-----------|--------|--------|---------|----------|-----------|-----------|-----|
| GPT-4o | 63.8 | 63.9 | 30.3 | 85.7 | 63.7 | 55.0 | 39.4 | 56.2 |
| Qwen2-VL-2B | 43.0 | 48.0 | 12.4 | 73.5 | 24.9 | 41.7 | 19.7 | 37.5 |
| **R1-VL-2B** | **52.1** | 49.8 | **17.1** | **75.2** | **29.4** | **44.0** | **26.2** | **41.6** |
| Qwen2-VL-7B | 58.2 | 60.7 | 16.3 | 83.0 | 42.1 | 50.6 | 32.5 | 48.7 |
| **R1-VL-7B** | **63.5** | 60.0 | **24.7** | **83.9** | **45.2** | **54.7** | **40.0** | **52.1** |
| Mulberry-7B | 63.1 | 61.3 | — | 83.9 | 45.1 | 54.1 | — | — |
| LlamaV-o1-11B | 54.4 | 59.4 | — | — | — | 63.5 | — | — |

R1-VL-7B 在 MathVista 上达到 63.5，接近 GPT-4o 的 63.8。R1-VL-2B 甚至超越 LLaVA-CoT-11B（+9.3 MathVista）。

### 消融实验（Table 2：各奖励组件效果）

| Warm-up | StepRAR | StepRVR | MathVista |
|---------|---------|---------|-----------|
| ✗ | ✗ | ✗ | 58.2 (基线) |
| ✓ | ✗ | ✗ | 61.2 |
| ✓ | ✓ | ✗ | 62.4 |
| ✓ | ✗ | ✓ | 61.9 |
| ✓ | ✓ | ✓ | **63.5** |

每个组件贡献明确：warm-up +3.0，StepRAR +1.2，StepRVR +0.7，联合 +2.3。

### 步级 vs 结果级奖励（Table 4）

| 方法 | MathVista |
|------|-----------|
| Warm-up only | 61.7 |
| Warm-up + Outcome-level reward | 62.3 |
| **Warm-up + Step-wise reward** | **63.5** |

步级奖励比结果级奖励多提升 1.2%。

### M 的参数敏感性（Table 3）

| M (每问生成数) | 2 | 3 | 4 | 5 | 6 |
|-------------|---|---|---|---|---|
| MathVista | 62.5 | 62.8 | 63.5 | 63.2 | 63.7 |

$M=4$ 在性能和计算成本之间取得最佳平衡。

### 关键发现

1. **StepGRPO 一致性超越 SFT**：在相同训练步数下始终优于 SFT
2. **小模型也能大幅提升**：R1-VL-2B 比基线 Qwen2-VL-2B 平均提升 4.1%
3. **步级奖励解决稀疏问题**：即使答案错误，有用的中间步骤也能获得正反馈
4. **规则化奖励足够有效**：无需额外训练过程奖励模型
5. **StepRVR 保证推理结构**：强制要求背景分析→推理→结论的逻辑流

## 亮点与洞察

- **密集步级奖励的巧妙设计**：不需要训练 PRM（Process Reward Model），通过规则化的关键步匹配和结构验证即可实现细粒度奖励
- **对 MLLM 推理的深入理解**：区分了推理准确性（StepRAR）和推理结构有效性（StepRVR）两个正交维度
- **实用且高效**：仅需 10K 训练数据、4 GPU 即可完成 RL 训练
- **软关键步匹配**：格式增强机制优雅地处理了数学表达式的多样写法

## 局限性

- 关键步骤的提取依赖 GPT-4，引入了额外依赖和成本
- 仅在数学推理为主的基准上验证，在空间推理、常识推理等场景的效果未知
- 基于 Qwen2-VL 验证，其他基础模型的适用性待确认
- 训练数据仅 10K，更大规模数据的 scaling 行为未探索
- StepRVR 的规则较为刚性（要求固定的推理结构），可能限制推理多样性

## 相关工作与启发

- **DeepSeek-R1 → MLLM 版本**：将 GRPO 从 LLM 扩展到 MLLM，核心贡献在于解决多模态场景的稀疏奖励问题
- **与 Mulberry 的互补**：Mulberry 提供 SFT 数据（warm-up），StepGRPO 在此基础上做 RL 自我改进
- **PRM-free 的路线**：规则化步级奖励可能是一种更实用的替代方案

## 评分 ⭐⭐⭐⭐

将 GRPO 扩展到 MLLM 的合理且有效的工作。步级奖励设计简洁实用，消融实验清晰。在多个基准上的一致性提升令人信服。主要不足是评估范围偏向数学推理，且关键步提取依赖外部大模型。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](../../ACL2025/multimodal_vlm/mammoth_vl_multimodal_reasoning.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](../../ACL2025/multimodal_vlm/spatialmqa_mllm_spatial_relations.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)

</div>

<!-- RELATED:END -->
