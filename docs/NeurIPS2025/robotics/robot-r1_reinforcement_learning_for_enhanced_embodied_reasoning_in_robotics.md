---
title: >-
  [论文解读] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics
description: >-
  [NeurIPS 2025][机器人][具身推理] Robot-R1 提出利用强化学习（GRPO）训练大视觉语言模型（LVLM）进行具身推理，通过将下一关键状态预测转化为多选题并用 RL 优化推理路径，仅凭 7B 参数在低级控制推理任务上超越 GPT-4o。 大视觉语言模型（LVLMs）在机器人领域展现了巨大潜力…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "具身推理"
  - "强化学习"
  - "机器人控制"
  - "GRPO"
  - "大视觉语言模型"
---

# Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics

**会议**: NeurIPS 2025  
**arXiv**: [2506.00070](https://arxiv.org/abs/2506.00070)  
**代码**: [GitHub](https://github.com/hiyouga/EasyR1)  
**领域**: 强化学习  
**关键词**: 具身推理, 强化学习, 机器人控制, GRPO, 大视觉语言模型

## 一句话总结

Robot-R1 提出利用强化学习（GRPO）训练大视觉语言模型（LVLM）进行具身推理，通过将下一关键状态预测转化为多选题并用 RL 优化推理路径，仅凭 7B 参数在低级控制推理任务上超越 GPT-4o。

## 研究背景与动机

大视觉语言模型（LVLMs）在机器人领域展现了巨大潜力，能同时处理视觉输入和自然语言指令来实现对机器人的高层控制。然而当前 LVLMs 在**精细的具身推理**方面仍面临挑战：

**SFT 数据集的局限性**：现有的具身推理训练数据集多为启发式构建，语言描述无法捕捉低级机器人控制所需的精确数量信息，且数据集并未针对实际机器人动作预测进行优化。

**灾难性遗忘问题**：SFT 方法在面对训练分布外的输入输出格式时表现急剧下降，会遗忘已学到的通用对话能力。

**泛化性不足**：SFT 训练的模型在空间推理和运动推理等对机器人控制至关重要的能力上提升有限，难以泛化到新任务。

受 DeepSeek-R1 的启发，作者认为**强化学习可以更好地激发和强化推理路径**，比 SFT 具有更好的泛化能力和样本效率。核心洞察在于：如果让模型通过 RL 自主探索如何推理来预测下一个关键状态，这种推理能力可以自然迁移到其它具身推理任务。

## 方法详解

### 整体框架

Robot-R1 的核心思路分为三步：（1）从专家示范中提取元数据和关键帧，构建训练数据；（2）将连续状态预测问题转化为多选问答（MCQA）形式；（3）使用 GRPO 算法训练 LVLM 生成推理过程并优化。整体框架还包含一个专门设计的评估基准 Robot-R1 Bench。

### 关键设计

1. **元数据条件化的数据生成**：
   由于 LVLM 无法直接从图像推断低级状态，Robot-R1 引入环境元数据 $M$，包含三类信息：（a）环境固定参考点（如桌面中心）；（b）3D 坐标系定义及各轴正方向；（c）机器人末端执行器的尺寸信息用于尺度估计。这些元数据作为问题提示的条件输入，帮助模型建立对场景的量化理解。

2. **三类多选问答任务设计**：

    - **路点预测 QA**（主任务）：给定当前观测 $o_t$、状态 $s_t$ 和元数据 $M$，预测下一个关键帧状态 $s_{k^*}$。问题包含正确答案和 3 个从合法状态空间随机采样的干扰项。
    - **当前状态预测 QA**（辅助任务）：从视觉观测中识别当前状态 $s_t$，增强模型对自身状态的理解。
    - **运动预测 QA**（辅助任务）：预测从当前状态到下一关键帧的运动方向（如"向上移动"、"略微向后"）。运动标签通过基于规则的启发式方法从 3D 笛卡尔位移中提取。

   将连续状态预测离散化为多选题的核心动机是：连续动作空间太大，RL 探索困难；离散化后动作空间收窄，学习更高效。

3. **GRPO 优化推理路径**：
   训练时模型需在 `<think>...</think>` 中输出推理过程，在 `<answer>...</answer>` 中输出最终答案。采用 GRPO 算法，对每个查询生成 $G$ 个不同回答，利用组内相对优势 $A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$ 来更新策略，同时施加 KL 散度惩罚防止策略偏离过远。

### 损失函数 / 训练策略

奖励信号由两部分组成：
- **格式奖励 $r_f$**：鼓励模型遵循指定的输出结构（`<think>`/`<answer>` 标签）
- **答案正确性奖励 $r_a$**：基于规则的正向反馈，模型答案与多选题正确选项精确匹配时给予奖励

总奖励 $R = r_f + r_a$。训练使用 batch size 128、5 个 epoch、学习率 $1.0 \times 10^{-6}$、每个 prompt 生成 5 个样本。

### Robot-R1 Bench 评估基准

专门设计的开放式问答基准，评估四种具身推理能力：规划、高级动作推理、运动推理和空间推理。共 215 个问题，由经验丰富的研究者手工创建，使用 GPT-4o 作为裁判进行 0-3 分评分。

## 实验关键数据

### 主实验：Robot-R1 Bench 低级控制推理

| 模型 | 运动推理(In) | 运动推理(Out) | 运动推理(Avg) | 空间推理(In) | 空间推理(Out) | 空间推理(Avg) |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| GPT-4o | 0.92 | 0.52 | 0.72 | 1.70 | 1.07 | 1.43 |
| Gemini-2.0-Flash | 0.52 | 0.40 | 0.46 | 1.76 | 1.14 | 1.49 |
| Qwen2.5-VL-7B-Ins | 0.64 | 0.52 | 0.58 | 1.62 | 1.11 | 1.40 |
| Direct SFT | 0.12 | 0 | 0.06 | 0.08 | 0.04 | 0.06 |
| CoT SFT | 0.84 | 0.56 | 0.70 | 0.46 | 0.07 | 0.29 |
| **Robot-R1 (Ours)** | **0.96** | **0.56** | **0.76** | **1.76** | **1.18** | **1.51** |

Robot-R1 在所有低级控制推理指标上取得最高分，超越所有商业模型。

### EmbodiedBench 操作基准

| 模型 | Base | Common | Complex | Spatial | Visual | Avg |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen2.5-VL-7B-Ins | 6.3 | 6.3 | 6.3 | 14.6 | 11.1 | 8.92 |
| Direct SFT | 0 | 0 | 0 | 0 | 0 | 0 |
| CoT SFT | 0 | 0 | 0 | 0 | 0 | 0 |
| **Robot-R1 (Ours)** | **12.5** | **8.3** | **6.3** | **14.6** | **16.7** | **11.68** |

Robot-R1 相比基线提升约 31%，SFT 方法完全无法完成任何任务。

### 消融实验

| RL 算法 | 规划 | 高级动作 | 运动 | 空间 |
|---------|:---:|:---:|:---:|:---:|
| GRPO | 1.44 | 1.30 | 0.76 | 1.51 |
| RLOO | 1.56 | 1.54 | 0.68 | 1.52 |
| REINFORCE++ | 1.40 | 1.08 | 0.62 | 1.38 |

GRPO 和 RLOO 表现相当，REINFORCE++ 因为批级奖励归一化方差更大而表现较差。

### 关键发现

- 仅通过低级控制任务训练，模型在高级动作推理上也获得显著提升（SFT 无法实现此迁移）
- Robot-R1 在 SpatialRGPT 基准上定量指标提升约 40%、定性指标提升约 60%
- GRPO/RLOO 等具有方差缩减机制的 RL 算法更适合 Robot-R1 训练
- 跨三个随机种子实验表明训练结果具有鲁棒性

## 亮点与洞察

- **连续到离散的巧妙转化**：将连续状态预测转为多选题，解决了 RL 探索困难的核心问题
- **推理能力的自然迁移**：仅训练低级状态预测就获得了高级推理能力的提升
- **RL 优于 SFT 的实证**：SFT（包括 CoT-SFT）在迁移到分布外任务时表现极差甚至崩溃，而 RL 训练的模型泛化能力显著更强

## 局限与展望

- 训练数据仅来自 RLBench 的 5 个桌面操作任务，场景多样性有限
- 规划能力略有下降，因为训练目标聚焦于下一关键帧预测而非长期规划
- 仅验证了 7B 模型，更大规模模型的效果未知
- 多选题的选项数量和干扰项生成策略对训练效果的影响未充分探索

## 相关工作与启发

- **DeepSeek-R1**：RL 训练推理模型的范式，Robot-R1 将此思路拓展到具身领域
- **ARM (James et al.)**：关键帧提取方法的来源
- **EmbodiedBench、SpatialRGPT**：外部评估基准，验证泛化能力
- 启发：RL 训练的具身推理能力可能是通往通用机器人智能的重要路径

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将 R1 式 RL 推理训练范式迁移到机器人具身推理是新颖且有意义的
- **实验充分度**: ⭐⭐⭐⭐ 三个基准评估 + 消融 + 种子鲁棒性 + 人类一致性验证
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，动机阐述充分
- **价值**: ⭐⭐⭐⭐ 为机器人领域引入了有效的 RL 推理训练范式，具有较强的实践指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[CVPR 2025\] UniAct: Universal Actions for Enhanced Embodied Foundation Models](../../CVPR2025/robotics/universal_actions_for_enhanced_embodied_foundation_models.md)
- [\[NeurIPS 2025\] Memo: Training Memory-Efficient Embodied Agents with Reinforcement Learning](memo_training_memory-efficient_embodied_agents_with_reinforcement_learning.md)
- [\[NeurIPS 2025\] VIKI-R: Coordinating Embodied Multi-Agent Cooperation via Reinforcement Learning](viki-r_coordinating_embodied_multi-agent_cooperation_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
