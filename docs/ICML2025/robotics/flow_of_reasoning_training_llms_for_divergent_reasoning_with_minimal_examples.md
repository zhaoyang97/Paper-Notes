---
title: >-
  [论文解读] Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples
description: >-
  [ICML2025][机器人][GFlowNet] 提出 Flow of Reasoning (FoR)，将多步 LLM 推理建模为 DAG 上的马尔可夫流，借助 GFlowNet 的轨迹平衡目标微调 LLM，使其仅用极少训练样本（如15个）即可采样出概率正比于奖励的多条高质量且多样化的推理路径。 - 发散性推理的重要性：人…
tags:
  - "ICML2025"
  - "机器人"
  - "GFlowNet"
  - "发散性推理"
  - "多样化采样"
  - "少样本微调"
  - "马尔可夫流"
  - "轨迹平衡"
---

# Flow of Reasoning: Training LLMs for Divergent Reasoning with Minimal Examples

**会议**: ICML2025  
**arXiv**: [2406.05673](https://arxiv.org/abs/2406.05673)  
**代码**: [Yu-Fangxu/FoR](https://github.com/Yu-Fangxu/FoR)  
**领域**: 强化学习  
**关键词**: GFlowNet, 发散性推理, 多样化采样, 少样本微调, 马尔可夫流, 轨迹平衡

## 一句话总结

提出 Flow of Reasoning (FoR)，将多步 LLM 推理建模为 DAG 上的马尔可夫流，借助 GFlowNet 的轨迹平衡目标微调 LLM，使其仅用极少训练样本（如15个）即可采样出概率正比于奖励的多条高质量且多样化的推理路径。

## 研究背景与动机

- **发散性推理的重要性**：人类智能的核心标志之一是对同一问题生成多条不同解法（divergent reasoning），这对增强鲁棒性（如 self-consistency 投票）和辅助科学发现至关重要。
- **现有方法的局限**：
    - **推理时方法**（CoT/ToT/RAP）：高度依赖基座模型能力，搜索式推理计算开销大，多样性受限于解码策略。
    - **SFT**：需要大量标注数据才能覆盖解的多样性，标注成本高。
    - **奖励最大化 RL**（PPO）：目标是找到最高奖励的单一解，天然忽略解的多样性。
- **核心差距**：缺乏一种数据高效的方法，在保证推理质量的同时发现多条不同的正确推理路径。

## 方法详解

### 核心思想：推理即马尔可夫流

将多步推理问题建模为有向无环图（DAG）上的流网络：

- **状态节点** $s_t$：推理过程中的中间状态（如 BlocksWorld 的方块配置）
- **边（动作）** $s_t \to s_{t+1}$：一步推理操作
- **完整轨迹** $\tau = (s_0 \to s_1 \to \cdots \to s_n)$：从初始状态到终态的完整推理路径
- **目标**：学习前向策略 $P_F(s_t | s_{t-1}; \theta, g)$，使采样轨迹的概率正比于终态奖励 $R(s_n)$

与传统 RL（PPO）追求奖励最大化不同，FoR 的目标是按奖励成比例地采样多条不同路径。

### 流的分解与前向策略

轨迹概率通过马尔可夫假设分解为逐步条件概率：

$$P(\tau) = \prod_{t=1}^{n} P_F(s_t | s_{t-1})$$

前向策略用 LLM 参数化：$P_F(s_{t+1} | s_t; \theta, g) = P_{\text{LLM}}(a_t | s_t; \theta, g, c)$。

### 轨迹平衡（Trajectory Balance）目标

核心训练约束：

$$Z(s_0, g) \prod_{t=1}^{n} P_F(s_t | s_{t-1}; \theta, g) = R(s_n) \prod_{t=1}^{n} P_B(s_{t-1} | s_t)$$

其中 $P_B$ 为后向策略，设为均匀分布 $P_B(s_{t-1}|s_t) = 1/|\text{Pa}(s_t)|$。

### 对数方差损失（Log-Variance Loss）

为避免直接学习 $\log Z$，采用对数方差近似：

$$\Phi(\tau; \theta) = \log R(s_n) + \sum_{t=1}^{n} \log P_B(s_{t-1}|s_t) - \sum_{t=1}^{n} \log P_F(s_t|s_{t-1}; \theta, g)$$

最终损失函数：

$$\mathcal{L}_V(\tau; \theta) = \left(\Phi(\tau; \theta) - \mathbb{E}_\tau[\Phi(\tau; \theta)]\right)^2$$

最小化不同轨迹上 $\Phi$ 的方差，使流满足终态流等于奖励的条件。

### 高效探索策略

- **On-policy**：用当前策略 $P_F$ 及温度变体采样轨迹
- **Off-policy**：优先回放缓冲区（优先高奖励轨迹）+ $\epsilon$-采样
- **局部搜索（Local Search）**：选取批次中最高奖励轨迹，截断后半段，用随机策略 $P_U$ 重构，高效探索高奖励区域的邻域

### 与 GFN-CoT 的关键区别

FoR 在**推理步骤级别**建模（每步对应一个推理操作），而非 token 级别，将搜索式推理的计算开销摊销（amortize）到训练阶段。

## 实验关键数据

**基座模型**：Llama-3-8B，所有微调方法使用相同数据集。

### BlocksWorld（具身推理）

| 方法 | 2步 Acc(%) | 4步 Acc(%) | 6步 Acc(%) | 6步 Diversity | 6步 Creativity(%) |
|------|-----------|-----------|-----------|--------------|------------------|
| CoT (1-shot) | 48.88 | 28.57 | 15.82 | 1.05 | 0.00 |
| CoT (GPT-4o) | 93.33 | 54.76 | 67.67 | 1.06 | 0.79 |
| RAP | 100.00 | 92.86 | 69.70 | - | - |
| O1-mini | 100.00 | 100.00 | 93.93 | 1.05 | 2.38 |
| SFT (α=1.0) | 44.44 | 42.06 | 34.68 | 1.04 | 4.76 |
| SFT + PPO | 46.66 | 44.44 | 24.58 | 1.08 | 3.17 |
| **FoR** | **100.00** | **98.41** | **78.44** | **1.33** | **9.52** |

### 跨任务汇总（核心结果）

- **Game24**（数学谜题）：FoR 发现 3+ 种不同正确解法，基线方法（SFT/CoT）仅能反复生成 1 种
- **Rubik's Cube**（空间推理）：FoR 在准确率和多样性上均大幅领先
- **1D-ARC**（抽象推理）：FoR 表现优异
- **GSM8K**（数学推理）& **ProntoQA**（逻辑推理）：FoR 在保持高准确率的同时显著提升解的多样性
- 整体比基线提升 **20%–85%**，仅需 **15 个训练样本**

### 关键消融实验

- 局部搜索对训练效率和最终性能贡献显著
- 混合探索（on-policy + off-policy + local search）效果最佳
- 奖励设计中，正确解给高奖励、错误解给低奖励的方式对多样性至关重要

## 亮点与洞察

1. **极致数据效率**：仅 15 个训练样本即可微调 LLM 实现多样化推理，远优于 SFT 对大量标注的需求
2. **原理优雅**：将 GFlowNet 的流匹配理论与 LLM 推理结合，奖励成比例采样天然鼓励多样性
3. **推理步骤级建模**：不同于 token 级 GFlowNet，在推理步骤粒度建模更符合多步推理的结构特性
4. **训练时摊销推理开销**：搜索代价在训练阶段完成，推理时仅需前向采样，效率远高于 ToT/RAP
5. **六个任务全面验证**：涵盖具身推理、数学、空间、抽象、逻辑等多个推理类型
6. **Creativity 指标新颖**：提出衡量方法独特发现正确解的比例，不仅看多样性还看"创造力"

## 局限与展望

1. **DAG 结构假设**：要求推理过程可建模为 DAG，对自由形式的开放式推理（如创意写作）未必适用
2. **奖励函数依赖**：需要明确的终态奖励 $R(s_n)$，而很多实际推理任务缺乏清晰的自动奖励信号
3. **状态转移函数**：部分任务需要额外的环境模拟器或 LLM 辅助确定状态转移 $T(s_t, a_t)$
4. **基座模型规模**：仅在 Llama-3-8B 上验证，更大规模模型上的效果未知
5. **可扩展性**：随推理步数增长，轨迹空间指数膨胀，探索效率可能下降
6. **与 RLHF/DPO 的结合**：未探讨与对齐方法的整合潜力

## 相关工作与启发

- **GFlowNet**（Bengio et al., 2021）：奖励成比例采样的理论基础
- **CoT/ToT/RAP**：推理时搜索的代表性方法，FoR 将搜索摊销到训练
- **GFN-CoT**（Hu et al., 2023）：token 级 GFlowNet + LLM，FoR 提升到推理步骤级
- **启发**：GFlowNet 的"按奖励比例采样"范式为 LLM 推理多样性提供了理论优雅的解决方案，后续可扩展到代码生成、定理证明等结构化推理场景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (GFlowNet + LLM推理的首次步骤级结合，框架优雅)
- 实验充分度: ⭐⭐⭐⭐ (6个任务覆盖广泛，消融扎实，但仅单一基座模型)
- 写作质量: ⭐⭐⭐⭐⭐ (流的类比直观易懂，公式推导清晰)
- 价值: ⭐⭐⭐⭐⭐ (填补LLM多样化推理的重要空白，数据效率极高)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](../../NeurIPS2025/robotics/nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[CVPR 2026\] Obstruction Reasoning for Robotic Grasping](../../CVPR2026/robotics/obstruction_reasoning_for_robotic_grasping.md)
- [\[NeurIPS 2025\] EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT](../../NeurIPS2025/robotics/egothinker_unveiling_egocentric_reasoning_with_spatiotempora.md)
- [\[NeurIPS 2025\] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics](../../NeurIPS2025/robotics/robot-r1_reinforcement_learning_for_enhanced_embodied_reasoning_in_robotics.md)
- [\[CVPR 2026\] Parse, Search, and Confirmation: Training-Free Aerial Vision-and-Dialog Navigation with Chain-of-Thought Reasoning and Structured Spatial Memory](../../CVPR2026/robotics/parse_search_and_confirmation_training-free_aerial_vision-and-dialog_navigation_.md)

</div>

<!-- RELATED:END -->
