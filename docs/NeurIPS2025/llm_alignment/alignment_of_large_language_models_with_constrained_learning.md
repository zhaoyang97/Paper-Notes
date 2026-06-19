---
title: >-
  [论文解读] Alignment of Large Language Models with Constrained Learning
description: >-
  [NeurIPS 2025][LLM对齐][约束对齐] 本文提出 CAID（Constrained Alignment via Iterative Dualization），通过迭代对偶方法交替更新 LLM 策略和对偶变量，在理论上证明了对偶方法可以找到最优约束 LLM 策略（至多存在参数化间隙），并在 PKU-SafeRLHF 数据集上显著改善了约束满足和 helpfulness-safety 权衡。
tags:
  - "NeurIPS 2025"
  - "LLM对齐"
  - "约束对齐"
  - "拉格朗日对偶"
  - "多目标优化"
  - "安全RLHF"
  - "DPO"
---

# Alignment of Large Language Models with Constrained Learning

**会议**: NeurIPS 2025  
**arXiv**: [2505.19387](https://arxiv.org/abs/2505.19387)  
**代码**: 无  
**领域**: 对齐RLHF  
**关键词**: 约束对齐, 拉格朗日对偶, 多目标优化, 安全RLHF, DPO

## 一句话总结

本文提出 CAID（Constrained Alignment via Iterative Dualization），通过迭代对偶方法交替更新 LLM 策略和对偶变量，在理论上证明了对偶方法可以找到最优约束 LLM 策略（至多存在参数化间隙），并在 PKU-SafeRLHF 数据集上显著改善了约束满足和 helpfulness-safety 权衡。

## 研究背景与动机

**领域现状**：RLHF 是 LLM 对齐的主流方法，但单一奖励模型难以充分表达多维人类偏好。现有方法分为两个方向：多目标对齐（通过加权聚合奖励）和约束对齐（在最大化主奖励的同时满足次要约束）。约束对齐在安全性场景中更自然——例如，在保证 LLM 有用性的同时，要求其安全性改进达到某个阈值。

**现有痛点**：基于拉格朗日的 LLM 策略搜索存在两大问题：(1) 迭代原始-对偶方法（如 Safe RLHF）在最坏情况下策略不收敛；(2) 非迭代对偶方法（如 One-shot Safety Alignment）虽然可以在分布空间找到最优解，但无法保证在 LLM 参数空间中达到最优。

**核心矛盾**：分布空间中的凸优化性质不能直接迁移到 LLM 参数空间。分布空间中的强对偶性不意味着参数空间中的最优性。现有工作缺乏关于"对偶方法能否在 LLM 参数空间找到最优约束策略"的理论分析。

**本文目标**：(1) 设计一种实用的迭代对偶对齐方法；(2) 建立其在 LLM 参数空间中的最优性理论保证；(3) 实验验证在安全对齐任务上的有效性。

**切入角度**：作者从约束学习理论出发，利用拉格朗日对偶性将非凸参数空间问题与凸分布空间问题联系起来，通过分析"参数化间隙"来桥接两者的最优性差距。

**核心 idea**：通过多轮迭代（multi-shot）交替执行 LLM 策略更新和对偶变量下降，以 one-shot 方法的解作为 warm-start 初始化，既保证理论最优性又提升实践效果。

## 方法详解

### 整体框架

输入一个预训练参考模型 $\pi_{\text{ref}}$、奖励模型 $r$（helpfulness）和效用模型 $g$（safety），CAID 交替执行两步：(1) 固定对偶变量 $\lambda$，通过 DPO 最大化拉格朗日函数来更新 LLM 策略；(2) 固定策略，通过对偶子梯度下降更新 $\lambda$。最终输出满足安全约束的对齐 LLM。

### 关键设计

1. **迭代对偶对齐算法 (CAID)**:

    - 功能：在 LLM 参数空间中寻找满足约束的最优策略
    - 核心思路：将约束对齐问题分解为对偶问题，交替更新。在每次迭代 $t$ 中，先通过对偶子梯度方向 $u(\lambda^{(t)}) = \mathbb{E}[\mathbb{E}_{y \sim \pi}[g(x,y)] - \mathbb{E}_{y \sim \pi_{\text{ref}}}[g(x,y)]] - b$ 更新对偶变量 $\lambda^{(t+1)} = [\lambda^{(t)} - \eta u(\lambda^{(t)})]_+$，然后通过 DPO 最大化复合奖励 $r_\lambda = r + \lambda^\top g$ 来更新策略。使用 one-shot 方法的解作为 warm-start 初始化。
    - 设计动机：one-shot 方法在分布空间可求解最优对偶变量，但参数空间有误差；multi-shot 迭代可以通过多轮调整修正这一误差，且 warm-start 显著加速收敛。

2. **两种实用实现（MoCAID 和 PeCAID）**:

    - 功能：分别在有/无显式奖励模型的场景下实现 CAID
    - 核心思路：MoCAID（基于模型）直接使用奖励和效用模型打分，通过 Bradley-Terry 模型构造伪偏好对，喂给 DPO 训练。PeCAID（基于偏好）在只有人类偏好标注时，先用 DPO 分别对 $r$ 和 $g$ 进行预对齐，获取隐式奖励模型 $\beta \log(\pi_{\theta_r}/\pi_{\text{ref}})$，再构造复合偏好。
    - 设计动机：覆盖不同实际场景——有些应用有显式打分器，有些只有偏好数据。

3. **最优性理论分析**:

    - 功能：证明 CAID 可以逼近最优约束 LLM 策略
    - 核心思路：定义参数化间隙 $\nu$ 衡量 LLM 参数空间覆盖分布空间的能力。证明原始-对偶间隙 $|D_p^* - P^*| \lesssim \nu$（Theorem 2），以及学到的策略在奖励和约束函数上的最优性间隙均为 $O(\nu)$（Theorem 3/4）。
    - 设计动机：首次为约束 LLM 对齐方法提供了完整的最优性保证，填补了理论空白。

### 损失函数 / 训练策略

训练使用 DPO 损失，复合奖励为 $r_\lambda = r + \lambda^\top g$。对偶变量通过投影子梯度下降更新。在线数据集通过从当前策略采样 600 prompts × 64 responses 构建。训练使用 LoRA（rank=8, alpha=16）、cosine lr schedule（lr=5e-4）、4 次迭代。

## 实验关键数据

### 主实验

| 数据集/设置 | 指标 | Multi-shot (CAID) | One-shot | 说明 |
|---|---|---|---|---|
| PKU-SafeRLHF (b=5) | Safety Improvement | 5.758 | 2.285 | Multi-shot 更准确地满足约束 |
| PKU-SafeRLHF (b=5) | Helpfulness Improvement | 9.769 | 7.248 | Multi-shot 同时提升有用性 |
| PKU-SafeRLHF (b=9) | Safety Improvement | 11.420 | 9.574 | 高约束阈值下差距缩小 |
| PKU-SafeRLHF (b=9) | Helpfulness Improvement | 6.879 | 4.271 | Multi-shot 仍有明显优势 |
| GPT-4o-mini 评估 | Helpfulness Win Rate | ~55-60% | baseline | Multi-shot 在多数 b 值下胜出 |
| GPT-4o-mini 评估 | Safety Win Rate | ~55-65% | baseline | Multi-shot 安全性更优 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|---|---|---|
| DPO (仅 helpfulness) | H.I.=高, S.I.=低 | 只优化单目标无法满足安全约束 |
| DPO (仅 safety) | H.I.=低, S.I.=高 | 牺牲过多有用性 |
| One-shot (小 b) | Safety < threshold | 小约束时不足 |
| One-shot (大 b) | Safety > threshold | 大约束时过度满足 |
| Multi-shot (warm-start) | 4 iterations converge | Warm-start 使对偶变量快速收敛 |
| Multi-shot (AdvBench) | Safety Win >60% | 对抗评估下仍保持安全优势 |

### 关键发现

- Multi-shot 方法在所有约束阈值 $b \in \{3,...,9\}$ 上都更接近目标安全约束，而 one-shot 在小 $b$ 时不足、大 $b$ 时过度满足
- Multi-shot 在 helpfulness-safety trade-off 的 Pareto 前沿上严格优于 one-shot
- 对抗数据（AdvBench）评估表明 multi-shot 模型在所有约束水平上安全性评分更高
- Red teaming 案例显示 multi-shot 模型能有效拒绝有害请求，而 one-shot 仍可能给出部分有害内容

## 亮点与洞察

- 首次为约束 LLM 对齐提供了完整的最优性理论保证，证明对偶方法在参数化间隙 $\nu$ 内可以找到最优策略
- Warm-start策略非常实用——用 one-shot 解初始化对偶变量，multi-shot 只需在小邻域内微调即可，训练时间仅增加约 170 分钟
- 理论和实验的良好对应：Theorem 2 预测参数化间隙决定优化质量，实验中 multi-shot 确实通过迭代修正了 one-shot 在参数空间中的误差

## 局限与展望

- 实验仅在 7B 规模模型（Alpaca-7b）上验证，需要在更大模型上验证 multi-shot 的有效性
- 仅考虑单约束（safety），多约束场景（如同时约束安全性、毒性、偏见等）的理论和实验需要探索
- 理论依赖参数化间隙假设，但该间隙在实际 LLM 中的量化尚不清楚
- 对偶变量收敛速度与 prompt 数量和 response 采样量有关，实际部署时的计算开销需要权衡

## 相关工作与启发

- **vs Safe RLHF (Dai et al., 2024)**: Safe RLHF 使用迭代原始-对偶方法同时更新策略和对偶变量，最坏情况下策略不收敛。CAID 通过先更新对偶再更新策略的交替策略避免了这一问题
- **vs One-shot Safety Alignment (Huang et al., 2024)**: One-shot 在分布空间计算最优对偶变量然后一次性对齐，但无法保证参数空间最优性。CAID 是 one-shot 的 multi-shot 推广，既保留其优点又修正参数化误差
- **vs SimPO / DPO variants**: CAID 框架与具体 alignment 算法正交，可以替换 DPO 为 SimPO 等变体

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次建立约束 LLM 对齐的完整最优性理论，multi-shot warm-start 设计精巧
- 实验充分度: ⭐⭐⭐⭐ 包含 scoring 评估、GPT-4o 评估、对抗评估、red teaming 案例，多角度验证
- 写作质量: ⭐⭐⭐⭐⭐ 51页含附录，理论推导完整严谨，实验细节充分
- 价值: ⭐⭐⭐⭐ 为约束对齐提供了理论基础和实用方法，但 7B 规模限制了直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](reinforcement_learning_finetunes_small_subnetworks_in_large_language_models.md)
- [\[ACL 2025\] Safety Alignment via Constrained Knowledge Unlearning](../../ACL2025/llm_alignment/safety_alignment_via_constrained_knowledge_unlearning.md)
- [\[NeurIPS 2025\] LASeR: Learning to Adaptively Select Reward Models with Multi-Armed Bandits](laser_learning_to_adaptively_select_reward_models_with_multi-armed_bandits.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)

</div>

<!-- RELATED:END -->
