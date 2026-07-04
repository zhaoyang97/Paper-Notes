---
title: >-
  [论文解读] Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model
description: >-
  [AAAI2026][自动驾驶][扩散模型] 提出 M-Diffusion Planner，基于多头扩散模型和 GRPO 后训练，实现策略级（strategy-level）运动规划，允许用户通过自然语言切换激进/保守/舒适等驾驶风格，同时保持 SOTA 规划性能。 - 现有学习型规划器经过监督训练后策略固定…
tags:
  - "AAAI2026"
  - "自动驾驶"
  - "扩散模型"
  - "GRPO"
  - "Motion Planning"
  - "Driving Preferences"
---

# Drive As You Like: Strategy-Level Motion Planning Based on A Multi-Head Diffusion Model

**会议**: AAAI2026  
**arXiv**: [2508.16947](https://arxiv.org/abs/2508.16947)  
**代码**: 待确认  
**领域**: 自动驾驶  
**关键词**: autonomous driving, diffusion model, GRPO, Motion Planning, Driving Preferences

## 一句话总结

提出 M-Diffusion Planner，基于多头扩散模型和 GRPO 后训练，实现策略级（strategy-level）运动规划，允许用户通过自然语言切换激进/保守/舒适等驾驶风格，同时保持 SOTA 规划性能。

## 背景与动机

- 现有学习型规划器经过监督训练后策略固定，产生平滑但千篇一律的轨迹，无法体现个人驾驶偏好
- 部分可控规划方法在动作级（action-level）交互，需用户逐步下达指令，与自动驾驶"解放驾驶员"的初衷相矛盾
- 传统行为克隆和监督学习难以建模多模态的人类驾驶行为分布
- 扩散模型具有强大的多样化生成能力，但直接后训练容易导致规划能力退化

## 核心问题

如何在保持高质量轨迹规划能力的前提下，让规划器支持多种驾驶策略（激进/保守/舒适），并通过自然语言实现实时策略切换？

## 方法详解

### 整体框架

M-Diffusion Planner 包含三个核心组件：

1. **编码器**：MLP-Mixer + Transformer
    - MLP-Mixer 对车道线、导航路线、动态物体、静态障碍等异构输入在 token 和 channel 维度交替混合，生成紧凑定长嵌入
    - Transformer 通过自注意力建模交通参与者之间的时空依赖关系

2. **多头扩散解码器**：基于 DiT（Diffusion Transformer）架构
    - 多个输出头对应不同驾驶策略（base/aggressive/conservative/comfortable）
    - 基于场景编码和高层策略标识符条件生成轨迹

3. **LLM 语义解释器**：作为用户与规划器之间的桥梁
    - 将自然语言指令（如"请开快一点"、"注意安全"）解析为结构化策略标识符
    - 策略在执行期间持续生效，除非用户显式修改

### 训练阶段

**基础模型训练**：

- 采用 score-based 生成框架（VP-SDE 公式），对真值轨迹加高斯噪声后训练模型预测噪声
- 损失函数：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{neighbor}} + \alpha \cdot \mathcal{L}_{\text{ego}}$
- 关键设计：训练阶段所有解码器头共享参数，避免分别训练，确保各头均获得充分学习

**GRPO 后训练**：

- 冻结编码器和其他输出头，仅微调目标策略头的输出层参数
- 采样 $S$ 条轨迹 → 奖励函数评估 → 标准化为相对优势 $A_i = \frac{r_i - \mu}{\sigma + \epsilon}$
- 总损失 = 策略梯度项 + KL 散度正则项：$\mathcal{L} = \sum_{i=1}^{S} A_i \cdot \log \pi_i + \beta \cdot \log \sigma$
- KL 散度约束防止更新过激导致规划能力退化
- 仅需 nuPlan-mini 中 10,000 条轨迹、训练 30 个 epoch 即可完成

### 推理阶段

- 用 DPM-Solver++ 二阶多步法确定性求解反向 ODE
- 对当前状态施加硬约束，采样过程中保持不变
- 用户通过自然语言 → LLM 解析 → 策略 ID → 选择对应头 → 生成轨迹
- 支持实时策略切换，无需重新训练或重新加载模型

## 实验关键数据

### 闭环评估（nuPlan Val14）

| 规划器 | 非反应式 (NR) | 反应式 (R) |
|---|---|---|
| PDM-Closed | 92.84 | 92.12 |
| PLUTO | 92.88 | 76.88 |
| Diffusion Planner | 89.87 | 82.80 |
| **M-Diffusion (base, 本文)** | **93.43** | **85.65** |
| M-Diffusion (Conservative) | 85.51 | 78.69 |
| M-Diffusion (Aggressive) | 82.63 | 75.11 |
| M-Diffusion (Comfortable) | 88.72 | 79.80 |

- 基础模型在非反应式和反应式两种设置下均达到 SOTA
- GRPO 微调后的策略头在性能下降可控范围内表现出明显的行为差异

### 开环评估（2000 场景）

| 策略 | 速度 (m/s) | 加速度 (m/s²) | 急动度 (m/s³) | 高速占比 |
|---|---|---|---|---|
| Base | 10.59 | 1.97 | 2.66 | 16.54% |
| Aggressive | 12.50 | 2.31 | 2.43 | 26.56% |
| Conservative | 9.57 | 1.80 | 2.58 | 7.68% |
| Comfortable | 11.03 | 1.72 | **1.85** | 17.7% |

- Aggressive 策略速度最高，高速段占比达 26.56%
- Conservative 策略速度最低，高速段仅 7.68%
- Comfortable 策略急动度最低（1.85），符合舒适驾驶目标

## 亮点

1. **策略级交互**：首次提出在策略层面注入人类意图的运动规划框架，策略设定后持续生效，无需逐步干预
2. **高效后训练**：GRPO 仅需 10,000 条轨迹和 30 个 epoch，即可为各策略头学到差异化行为，训练开销极小
3. **共享参数设计**：基础训练阶段多头共享权重，确保所有策略头都具备高质量规划基础
4. **零样本策略切换**：推理时通过 LLM 即时翻译自然语言为策略 ID，实时切换无需重新加载

## 局限与展望

- 策略种类有限（仅 3 种 + base），更细粒度的偏好（如"稍微快一点"）无法精确建模
- 闭环评估中策略头（尤其 Aggressive）性能相比 base 有明显下降（82.63 vs 93.43），安全性存疑
- 开环评估中未报告碰撞率、偏离车道等安全指标
- LLM 解释器的可靠性和延迟未做评估，真实部署场景下可能成为瓶颈
- 仅在 nuPlan 数据集上验证，缺乏跨数据集或真车实验

## 与相关工作的对比

| 方法 | 交互方式 | 多样性 | 性能保持 |
|---|---|---|---|
| Diffusion Planner | 无交互 | 有限 | 好 |
| SceneControl | 动作级指令 | 中 | 中 |
| PLUTO（对比学习） | 无交互 | 有限 | 好 |
| **M-Diffusion（本文）** | **策略级自然语言** | **高** | **好** |

- 相比动作级可控方法（如 SceneControl），本文在策略层面交互更符合真实使用场景
- 相比 Diffusion Planner，通过 GRPO 后训练显著增强轨迹多样性
- 相比 PLUTO 的对比学习，本文通过 RL 后训练实现偏好对齐

## 启发与关联

- GRPO 在扩散模型后训练中的成功应用表明，LLM 领域的对齐技术可迁移至连续决策任务
- 多头共享训练 + 单头微调的范式可推广到其他需要多模态输出的生成任务
- 策略级交互思路可扩展至更多维度：变道风格、跟车距离偏好、路口通行策略等

## 评分
- 新颖性: 7/10（策略级交互 + GRPO 微调扩散模型的组合较新颖）
- 实验充分度: 6/10（闭环 SOTA 有说服力，但缺安全指标和真车验证）
- 写作质量: 7/10（结构清晰，公式完整，但部分表述冗余）
- 价值: 7/10（为可控驾驶规划提供了实用范式，但策略粒度有限）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Think Before You Drive: World Model-Inspired Multimodal Grounding](../../CVPR2026/autonomous_driving/think_before_you_drive_world_model-inspired_multimodal_grounding.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[AAAI 2026\] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)
- [\[ICLR 2026\] Multi-Head Low-Rank Attention (MLRA)](../../ICLR2026/autonomous_driving/multi-head_low-rank_attention.md)
- [\[CVPR 2026\] Diffusion Forcing Planner: History-Annealed Planning with Time-Dependent Guidance for Autonomous Driving](../../CVPR2026/autonomous_driving/diffusion_forcing_planner_history-annealed_planning_with_time-dependent_guidance.md)

</div>

<!-- RELATED:END -->
