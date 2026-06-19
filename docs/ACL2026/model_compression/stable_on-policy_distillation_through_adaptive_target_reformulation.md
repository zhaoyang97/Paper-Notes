---
title: >-
  [论文解读] Stable On-Policy Distillation through Adaptive Target Reformulation
description: >-
  [ACL 2026 Findings][模型压缩][知识蒸馏] 本文提出 Veto，一种目标层面的重构方法，通过在 logit 空间构建教师-学生的几何桥接分布来稳定 on-policy 知识蒸馏，单一参数 $\beta$ 同时在 forward KL 中充当自适应梯度否决器（抑制低置信度 token 的有害梯度）和在 reverse KL 中充当果断性旋钮（平衡奖励驱动和输出多样性），在 GSM8K 上比 SFT 提升 9.2%。
tags:
  - "ACL 2026 Findings"
  - "模型压缩"
  - "知识蒸馏"
  - "On-policy蒸馏"
  - "梯度稳定性"
  - "KL散度"
  - "目标重构"
---

# Stable On-Policy Distillation through Adaptive Target Reformulation

**会议**: ACL 2026 Findings  
**arXiv**: [2601.07155](https://arxiv.org/abs/2601.07155)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 知识蒸馏, On-policy蒸馏, 梯度稳定性, KL散度, 目标重构

## 一句话总结

本文提出 Veto，一种目标层面的重构方法，通过在 logit 空间构建教师-学生的几何桥接分布来稳定 on-policy 知识蒸馏，单一参数 $\beta$ 同时在 forward KL 中充当自适应梯度否决器（抑制低置信度 token 的有害梯度）和在 reverse KL 中充当果断性旋钮（平衡奖励驱动和输出多样性），在 GSM8K 上比 SFT 提升 9.2%。

## 研究背景与动机

**领域现状**：知识蒸馏（KD）是将大型语言模型能力转移到小型学生模型的广泛技术。传统监督 KD 在固定的教师生成轨迹上训练，但存在暴露偏差（exposure bias）——训练时用教师数据、推理时用自生成数据，导致自回归任务性能退化。On-policy KD 通过在学生自生成的输出上学习来缓解这一问题。

**现有痛点**：On-policy KD 面临严重的训练不稳定性，因为新手学生和专家教师之间的分布差距太大：(1) Forward KL 目标在学生对教师偏好 token 赋予近零概率时会产生梯度爆炸（$P_T(y)/P_S(y) \to \infty$）；(2) Reverse KL 目标虽然数值稳定，但缺乏对模式寻找强度的显式控制，容易导致模式坍缩和多样性丧失。

**核心矛盾**：现有方法主要在数据层面混合教师和学生 token 来弥合差距，却忽视了优化目标本身的稳定性。即使使用混合数据，迫使新手学生立即匹配专家的尖锐分布仍会创造陡峭的优化悬崖。问题根源在于散度目标的几何性质。

**本文目标**：提出一种目标层面的重构，在 logit 空间构建教师与学生之间的分布桥接，同时解决 forward KL 的梯度爆炸和 reverse KL 的模式坍缩问题。

**切入角度**：不在数据层面混合样本，而是在分布层面混合——在 logit 空间创建一个中间目标分布，强调教师和学生的共识区域，有效地"否决"低置信度 token 上的有害更新。

**核心 idea**：构建几何桥接分布 $Q \propto P_T \cdot P_S^\beta$，作为 Product of Experts 形式的共识过滤器，只有教师（质量）和学生（置信度）都支持的 token 才获得高目标概率，单一参数 $\beta$ 统一控制 forward KL 的梯度抑制和 reverse KL 的果断性-多样性权衡。

## 方法详解

### 整体框架

Veto 在标准 on-policy KD 的基础上修改目标分布：不直接用教师分布 $P_T$ 作为目标，而是构建中间目标 $Q$，通过 logit 空间的几何插值实现。对于每个 token 位置，计算教师和学生的 logits $z_T$ 和 $z_S$，构建 $Q \propto \exp(z_T + \beta \cdot z_S)$，然后最小化 $D_{KL}(Q \| P_S)$（forward KL）或 $D_{KL}(P_S \| Q)$（reverse KL）。$\beta$ 按线性衰减从初始值递减到 0。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["学生 on-policy 自生成输出"] --> B["逐 token 取教师 logits z_T 与学生 logits z_S"]
    B --> C["几何桥接分布 Q ∝ exp(z_T + β·z_S)<br/>Product of Experts 共识过滤器"]
    C --> D{"选择散度目标"}
    D -->|forward KL| E["自适应梯度否决<br/>P_S^β 门控掐掉低置信 token 梯度爆炸"]
    D -->|reverse KL| F["果断性旋钮<br/>β 调节模式寻找 ↔ 多样性"]
    E --> G["更新学生 + 锐化效应与线性衰减调度<br/>β ← β·(1 − i/N)，最优不动点 P_S ∝ P_T^(1/(1−β))"]
    F --> G
    G -->|下一训练步| A
```

### 关键设计

**1. 自适应梯度否决（Adaptive Gradient Veto, Forward KL）：在 forward KL 中掐掉低置信度 token 上的梯度爆炸**

标准 forward KL 在 on-policy 早期最危险：当学生对教师偏好的 token 赋了近零概率（$P_S(y)\to 0$ 而 $P_T(y)>0$）时，比值 $P_T(y)/P_S(y)$ 发散，实验里梯度能冲到 $10^7$ 以上。Veto 把目标分布换成几何桥接 $Q=P_T\cdot P_S^\beta$ 后，损失项变成 $\mathcal{L}(y)\approx P_S(y)^\beta \log P_S(y)$；由 L'Hôpital 法则，多项式项 $P_S^\beta(y)$ 衰减到零的速度快过对数项 $\log P_S(y)$ 发散的速度，于是这一项自然趋零，相当于给“学生还无知的 token”装了一道门控。它不改模型架构、也不改数据生成策略，纯从目标函数层面消掉了早期噪声输出带来的优化不稳定。

**2. 果断性旋钮（Decisiveness Knob, Reverse KL）：用同一个 $\beta$ 在 reverse KL 里显式调节模式寻找与多样性**

reverse KL 数值上稳定，却没有显式机制控制“模式寻找”的强度，容易模式坍缩、丢掉多样性。Veto 注意到 reverse KL 的梯度其实等价于一次策略梯度更新

$$\nabla_\theta \mathcal{L}_{\text{REV}} = \mathbb{E}_{y \sim P_S}[\nabla_\theta \log P_S(y) \cdot A(y)]$$

其中优势函数 $A(y) = -\log P_T(y) + (1-\beta) \log P_S(y)$。这样 $\beta$ 就成了一个从 KD 到 RL 的连续旋钮：$\beta=0$ 是标准 reverse KD（完全匹配教师），$0<\beta<1$ 是几何 KD（在高奖励区域寻找、同时保留多样性预算），$\beta\to 1$ 退化成纯 REINFORCE（零熵正则、坍缩到单一最高奖励模式）。用户因此能按任务需求在“果断”和“多样”之间调档。

**3. 锐化效应与线性衰减调度：保证学生收敛到教师的锐化版本，并随训练逐步贴近教师**

在最优不动点 $P_S^* = Q$ 处，可解出 $P_S^*(y|x) \propto P_T(y|x)^{1/(1-\beta)}$。因为 $0 \leq \beta < 1$，指数 $1/(1-\beta) > 1$，学生分布天然比教师更尖锐、更果断。而 $\beta$ 不取定值，按线性衰减 $\beta \leftarrow \beta \cdot (1 - i/N)$ 随训练步 $i$ 递减到 0：训练初期大 $\beta$ 给噪声学生强保护，后期学生改善后逐步恢复标准 KD、缩小与教师的差距。这正呼应了“早期强保护、后期逐步放松”的整体设计，消融里线性衰减也确实优于恒定 $\beta$。

### 损失函数 / 训练策略

使用 Qwen2-0.5B-IT 作为学生，Qwen2-7B-IT 作为教师。先在任务数据上监督微调教师，再从训练集采样 1K 实例进行学生 on-policy 训练。学习率 1e-5，warmup 比例 0.1，dropout 0.1，训练 3 个 epoch，2 张 H100 GPU。$\beta$ 通过网格搜索选择并线性衰减。不同任务用不同 $\beta$：推理 $\beta=0.8$，代码 $\beta=1.0$，摘要 $\beta=0.3$。

## 实验关键数据

### 主实验

**跨三个领域的性能对比**

| 方法 | GSM8K (Accuracy) | HumanEval (Pass@1) | HumanEval (Pass@10) | DialogSum (Win-rate) |
|------|------|------|------|------|
| Teacher SFT | 74.7 | 64.7 | 72.2 | 65.0 |
| Student SFT | 30.7 | 26.9 | 34.6 | 54.0 |
| Supervised KD | 33.4 | 26.8 | 34.5 | 54.3 |
| SKD | 33.6 | 24.8 | 34.8 | 53.6 |
| On-policy KD | 35.1 | 22.9 | 35.3 | 54.3 |
| **Veto (Ours)** | **39.9** | **29.0** | **37.7** | **56.5** |

### 消融实验

| 配置 | GSM8K Accuracy | 说明 |
|------|------|------|
| Student SFT | 30.7 | 基线 |
| Supervised KD | 33.4 | +2.7 |
| On-policy KD | 35.1 | +4.4 |
| Veto ($\beta=0.8$) | **39.9** | **+9.2**，最佳 |
| Veto (无衰减) | — | 衰减调度有益 |

**不同 $\beta$ 值的影响**：
- $\beta=0$ 退化为标准 on-policy KD
- $\beta=0.8$ 在 GSM8K 上最优
- $\beta=1.0$ 在代码生成上最优
- $\beta=0.3$ 在摘要生成上最优
- $\beta$ 过大会导致过度锐化，过小则保护不足

### 关键发现

- Veto 相比 Student SFT 在 GSM8K 上提升 9.2 个百分点（30.7%→39.9%），相比 on-policy KD 提升 4.8 个百分点
- 标准 Forward KL 在无知 token 上梯度超过 $10^7$，Veto 有效将其抑制在稳定范围内
- HumanEval Pass@1 从 22.9 提升到 29.0（+6.1），DialogSum Win-rate 从 54.3 提升到 56.5（+2.2）
- 不同任务的最优 $\beta$ 不同，反映了推理（需要高果断性）和生成（需要多样性）任务的本质差异
- 线性 $\beta$ 衰减优于恒定 $\beta$，验证了"早期强保护、后期逐步放松"的策略

## 亮点与洞察

- 从散度目标的几何性质入手解决 on-policy KD 的稳定性问题，比数据层面的混合更加根本
- 单一参数 $\beta$ 统一解决 forward KL 梯度爆炸和 reverse KL 模式坍缩两个问题，理论优雅
- Theorem 3 揭示 Veto 在 reverse KL 下等价于带缩放熵正则化的 REINFORCE，建立了 KD 与 RL 的桥梁
- Product of Experts 形式的"共识过滤器"直觉清晰：只有教师和学生都支持的 token 才获得高权重

## 局限与展望

- 实验仅使用 Qwen2-0.5B 作为学生和 Qwen2-7B 作为教师，未在更大规模（如 7B→70B）上验证
- 不同任务需要不同的 $\beta$，最优超参数需通过网格搜索确定
- 理论分析主要在 token 级别，序列级别的动态特性未深入探讨
- 与其他先进的 on-policy 方法（如 RLHF/DPO）的关系和组合潜力未充分探索

## 相关工作与启发

- **vs GKD (On-policy KD)**: GKD 提出了 on-policy 蒸馏框架但未解决目标稳定性，Veto 从目标层面提供稳定性保证
- **vs SKD (Interleaved Sampling)**: SKD 通过交错采样改善反馈质量，但仍在数据层面操作；Veto 在分布层面操作，两者正交
- **vs MiniLLM/f-distill (Reverse KL)**: 使用 reverse KL 鼓励模式寻找但缺乏多样性控制，Veto 通过 $\beta$ 提供显式果断性-多样性权衡

## 评分

- 新颖性: ⭐⭐⭐⭐ 从目标函数几何性质出发统一解决两个问题的思路优雅，KD-RL 桥梁有理论深度
- 实验充分度: ⭐⭐⭐ 三个任务验证有效，但模型规模单一（0.5B-7B），缺少更多 baseline 对比
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，直觉解释到位，图示质量高
- 价值: ⭐⭐⭐⭐ 为 on-policy KD 提供了简洁有效的稳定化方案，理论与实践结合良好

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](../../ICML2026/model_compression/entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ACL 2025\] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](../../ACL2025/model_compression/aligndistil_token_level_alignment.md)
- [\[ICLR 2026\] π-Flow: Policy-Based Few-Step Generation via Imitation Distillation](../../ICLR2026/model_compression/pi-flow_policy-based_few-step_generation_via_imitation_distillation.md)
- [\[CVPR 2026\] WPT: World-to-Policy Transfer via Online World Model Distillation](../../CVPR2026/model_compression/wpt_world-to-policy_transfer_via_online_world_model_distillation.md)
- [\[NeurIPS 2025\] ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation](../../NeurIPS2025/model_compression/orpo-distill_mixed-policy_preference_optimization_for_cross-architecture_llm_dis.md)

</div>

<!-- RELATED:END -->
