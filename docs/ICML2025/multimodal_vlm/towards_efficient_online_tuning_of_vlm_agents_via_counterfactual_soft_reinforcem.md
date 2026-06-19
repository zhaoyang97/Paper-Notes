---
title: >-
  [论文解读] Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning
description: >-
  [ICML 2025][多模态VLM][VLM Agent] 提出 Counterfactual Soft Reinforcement Learning (CoSo)，利用反事实推理评估每个 token 对最终动作的因果影响，通过因果加权熵正则优化集中探索关键 token，解决 VLM 智能体在线 RL 微调中文本动作空间爆炸问题，在 Android 控制、卡牌游戏、具身 AI 上分别提升 12.3%、9.3%、16.7%。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "VLM Agent"
  - "Counterfactual Reasoning"
  - "Soft RL"
  - "Exploration Efficiency"
  - "causal inference"
---

# Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning

**会议**: ICML 2025  
**arXiv**: [2505.03792](https://arxiv.org/abs/2505.03792)  
**代码**: [github.com/langfengQ/CoSo](https://github.com/langfengQ/CoSo)  
**领域**: 多模态VLM  
**关键词**: VLM Agent, Counterfactual Reasoning, Soft RL, Exploration Efficiency, causal inference

## 一句话总结

提出 Counterfactual Soft Reinforcement Learning (CoSo)，利用反事实推理评估每个 token 对最终动作的因果影响，通过因果加权熵正则优化集中探索关键 token，解决 VLM 智能体在线 RL 微调中文本动作空间爆炸问题，在 Android 控制、卡牌游戏、具身 AI 上分别提升 12.3%、9.3%、16.7%。

## 研究背景与动机

VLM 作为决策智能体在设备控制、游戏、机器人等场景中应用广泛。RL 在线微调使 VLM 能迭代与环境交互优化多步目标，但面临两大挑战：

**探索空间爆炸**：传统 RL 动作空间 $|\mathcal{A}|=6$，VLM 的文本动作空间达 $|V|^n = 32100^{100}$

**动作生成非端到端**：VLM 输出需经解析函数 $f^{\text{parse}}$ 转为可执行动作，大量 token（CoT、格式固定等）对最终动作无影响

核心洞察：**只有少量"动作关键"token（< 10%）真正决定最终动作**，标准熵正则化对所有 token 施加均匀不确定性，导致低效甚至有害的探索。

## 方法详解

### 整体框架

CoSo 包含三个阶段：
1. **Rollout**：VLM 智能体与环境交互采集数据
2. **反事实推理**：计算每个 token 的因果权重
3. **CoSo 更新**：用因果加权熵正则化优化 VLM

### Token-to-Action 因果分析

用结构因果模型 (SCM) 刻画 token $y^i$ 与动作 $a$ 的关系：

$$\mathcal{B}_{y \to a}^i = |\mathbb{P}(a|y, \epsilon_a) - \mathbb{P}(a|y^{-i} \cup y^i_{\text{null}}, \epsilon_a)|$$

对每个 token 做"反事实干预"——用空值替换该 token，观察解析后动作概率的变化。变化大→因果重要。

SCM 用轻量 BERT 模型（0.01B 参数）实例化。

### 因果加权熵正则

标准 soft RL：
$$\mathcal{H}(\pi(\cdot|s)) = \sum_{i=1}^n \mathcal{H}(y^i|y^{1:i-1})$$

CoSo 改为：
$$\mathcal{H}^{\mathcal{B}}(\pi(\cdot|s)) = \sum_{i=1}^n \mathcal{B}_{y \to a}^i \cdot \mathcal{H}(y^i|y^{1:i-1})$$

关键 token（高因果权重）获得更多探索；低影响 token 的探索被抑制。

### 理论保证

- **引理 4.2（策略评估收敛）**：CoSo 的 Bellman 备份算子 $\mathcal{T}^{\mathcal{B}}$ 收敛到不动点
- **引理 4.3（策略改进）**：$Q^{\tilde{\pi}}(s,a) \geq Q^{\pi}(s,a)$
- **命题 4.4（策略迭代）**：反复应用策略评估和改进收敛到最优

### 实现

- 离线阶段：SFT 初始化 VLM（学习任务格式和有效输出）
- 在线阶段：RL + CoSo 微调（可基于 AWR 或 PPO）

## 实验关键数据

### Android-in-the-Wild

| 方法 | General Train | General Test | Web Train | Web Test | 平均 |
|------|------|------|------|------|------|
| GPT-4V + AppAgent | 13.5 | 17.7 | 12.5 | 8.3 | 13.0 |
| DigiRL | 64.6 | 62.7 | 68.1 | 64.2 | 64.9 |
| **CoSo** | **72.9** | **71.3** | **77.0** | **70.5** | **72.9** |

### Gym Cards（卡牌推理）

| 方法 | NL | EZP | P24 | BJ | 平均 |
|------|------|------|------|------|------|
| RL4VLM | 88.4 | 50.0 | 2.5 | 39.3 | 45.1 |
| **CoSo** | **100.0** | **50.0** | **5.8** | **41.5** | **49.3** |

### ALFWorld（具身 AI）

CoSo 达到 26.5%，较 RL4VLM 的 22.7% 提升 16.7%。

### 消融实验

- RL（无熵）：基线
- RL + $\mathcal{H}$（均匀熵）：仅微弱提升
- RL + $\mathcal{H}^{\mathcal{B}}$（CoSo）：**显著提升收敛速度和最终性能**

关键 token 仅占总 token 的 < 10%，> 80% 的 token 因果权重 < 0.2。

## 亮点与洞察

1. **精准定位关键 token**：通过反事实推理将探索空间有效缩小约 $32100^{90}$ 倍
2. **通用框架**：可自然适配 AWR、PPO 等不同 RL 目标
3. **理论完备**：收敛和策略改进保证与标准 soft RL 一致
4. **实际案例生动**：Android 场景中 CoSo 能在误操作后采样到 "Home" 恢复动作，而标准 RL 反复选择不可点击的按钮

## 局限性

- 实验中最长 utterance 不超过 300 token，超长 CoT 的效果未验证
- SCM 的 BERT 模型需与 VLM 同步更新
- 反事实推理增加了每轮计算开销

## 相关工作

- VLM 智能体（DigiRL、RL4VLM、AutoUI）
- RL 探索（intrinsic motivation、entropy regularization）
- RLHF（PPO、DeepSeek-R1）

## 评分

⭐⭐⭐⭐⭐ — 问题定义清晰（VLM 文本动作空间的探索爆炸），解决思路优雅（反事实因果 + 加权熵），理论扎实，三类任务上一致显著提升。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning](../../NeurIPS2025/multimodal_vlm/praxisvlm_visiongrounded_decision_making_via_textdriven_rein.md)
- [\[CVPR 2025\] DocVLM: Make Your VLM an Efficient Reader](../../CVPR2025/multimodal_vlm/docvlm_make_your_vlm_an_efficient_reader.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-Based VLM Agent](../../ICCV2025/multimodal_vlm/gtr_guided_thought_reinforcement_prevents_thought_collapse_in_rl-based_vlm_agent.md)
- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/unified_reinforcement_and_imitation_learning_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
