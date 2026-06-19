---
title: >-
  [论文解读] Entropy-Aware On-Policy Distillation of Language Models
description: >-
  [ICML 2026][模型压缩][知识蒸馏] 针对在策略蒸馏中 reverse KL 在教师高熵区域引发多样性坍缩和梯度不稳的问题，提出根据教师 token 级熵值自适应混合 forward KL 与 reverse KL 的蒸馏策略，在六个数学推理基准上 Pass@8 最高提升 +5.05。 领域现状：在策略（on-po…
tags:
  - "ICML 2026"
  - "模型压缩"
  - "知识蒸馏"
  - "在策略蒸馏"
  - "KL散度"
  - "熵感知"
  - "语言模型"
---

# Entropy-Aware On-Policy Distillation of Language Models

**会议**: ICML 2026  
**arXiv**: [2603.07079](https://arxiv.org/abs/2603.07079)  
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: 知识蒸馏, 在策略蒸馏, KL散度, 熵感知, 语言模型

## 一句话总结
针对在策略蒸馏中 reverse KL 在教师高熵区域引发多样性坍缩和梯度不稳的问题，提出根据教师 token 级熵值自适应混合 forward KL 与 reverse KL 的蒸馏策略，在六个数学推理基准上 Pass@8 最高提升 +5.05。

## 研究背景与动机

**领域现状**：在策略（on-policy）蒸馏是语言模型知识迁移的主流范式——学生模型在自身采样轨迹上，利用教师提供的 dense token-level 信号进行学习。标准做法采用 reverse KL 散度 $D_{\mathrm{KL}}(p_\theta \| p_T)$ 作为训练目标，鼓励学生模型聚焦于教师分布的高置信模式。

**现有痛点**：reverse KL 是 mode-seeking 的，它让学生把概率质量集中在教师分布的峰值上。当教师分布具有高熵——即存在多个合理续写（如数学解题的多条推理路径）——时，reverse KL 会强迫学生只拟合其中一种，导致生成多样性骤降。更严重的是，在高熵区域教师的梯度信号方差大，训练不稳定。

**核心矛盾**：mode-seeking（reverse KL，精确但窄）与 mode-covering（forward KL，全面但散）之间存在根本性 trade-off。现有方法一刀切地选择 reverse KL，忽略了教师输出不确定性随 token 位置动态变化的事实。

**本文目标**：设计一种能在蒸馏过程中自适应感知教师不确定性、在高熵时切换为 forward KL 的蒸馏框架，同时保持在策略训练的效率优势。

**切入角度**：作者观察到教师 token 级熵 $H(p_T(\cdot|x_{<t}))$ 可以作为"什么时候该 mode-seek、什么时候该 mode-cover"的天然指示器。低熵 = 教师确信 → 用 reverse KL 精确模仿；高熵 = 教师不确定 → 用 forward KL 保留多样性。

**核心 idea**：根据教师逐 token 熵值，自适应地将标准 reverse KL 目标增强为 forward KL，在一个统一的在策略框架中同时兼顾精确模仿与多样性保持。

## 方法详解

### 整体框架
方法要解决的核心问题是：在策略蒸馏一律用 reverse KL，会在教师不确定的高熵 token 上逼着学生坍缩到单一续写、还把梯度搅得不稳（§3 实测：蒸馏后的学生只保留 6.8% 的高熵 token，而教师有 18.5%）。本文（EOPD）不改变在策略蒸馏的标准流程（学生采样轨迹 → 教师逐 token 打分 → 算损失 → 梯度更新），只在「算损失」这一步动手：reverse KL 始终保留（低熵区精确模仿、保住高效收敛），但对教师条件分布熵 $H_t^{\mathrm{te}}=H(p_T(\cdot|x_{<t}))$ 超过阈值 $\tau$ 的那些高熵 token，额外**加挂**一项 forward KL，把教师的多模态分布覆盖回来。换句话说，它不是在两种 KL 之间平滑插值，而是用一个熵阈值**门控**、只在不确定处把 forward KL 加上去。

### 关键设计

**1. 熵感知的 KL 目标增强：用教师熵阈值门控地加挂 forward KL**

标准在策略蒸馏（OPD）一刀切地优化（带 PPO 截断的）reverse KL，它是 mode-seeking 的——把学生概率质量往教师峰值挤。这在教师确信（低熵）时高效，但教师面对多条合理推理路径（高熵）时会强行只拟合一种，多样性骤降、梯度还不稳（§3 的玩具实验里，高熵教师下学生 top-1 索引每步平均变化 84 次，低熵时仅 7 次）。EOPD 把"什么时候要保多样性"交给教师逐 token 熵 $H_t^{\mathrm{te}}=-\sum_x \pi_{\mathrm{te}}(x|\mathbf{c}_t)\log\pi_{\mathrm{te}}(x|\mathbf{c}_t)$ 来裁决，在 reverse KL 基础上**加挂**一项受指示函数门控的 forward KL：

$$\mathcal{L}_t^{\mathrm{EOPD}} = \mathcal{L}_t^{\mathrm{OPD}} + \alpha \cdot \mathbb{I}\!\left[H_t^{\mathrm{te}} > \tau\right] \mathcal{L}_t^{\mathrm{FKL}}$$

其中 $\mathcal{L}_t^{\mathrm{OPD}}$ 就是那项截断 reverse KL，$\mathcal{L}_t^{\mathrm{FKL}}=D_{\mathrm{KL}}(\pi_{\mathrm{te}}\|\pi_\theta)$ 是 forward KL。关键在于这是**硬阈值门控、不是平滑插值**：低熵 token 指示函数取 0，目标退化成标准 reverse KL，保住效率与收敛速度；只有当教师熵越过阈值 $\tau$，forward KL 才以权重 $\alpha$ 介入，强迫学生在这些不确定位置保留多个合理续写的概率质量。实验取 $\tau=0.8$、$\alpha=1.0$。这样正好对症 reverse KL 一刀切忽略「教师不确定性随 token 位置动态变化」这一事实——精确模仿留给低熵区、多样性让给高熵区。

**2. forward KL 的 top-k 高效近似：熵感知不额外掏采样成本**

加挂 forward KL 的天然麻烦是它定义为对教师分布求期望，朴素实现要从教师采样、还会逼学生去拟合教师分布的低概率长尾，既增开销又损效率。EOPD 绕开采样：把 forward KL 近似成只在教师 **top-k（k=16）** 个 token 上、用重归一化后的教师分布 $\tilde{\pi}_{\mathrm{te}}$ 求的期望：

$$\mathcal{L}_t^{\mathrm{FKL}} \approx \sum_{x\in\mathcal{S}_t^k} \tilde{\pi}_{\mathrm{te}}(x|\mathbf{c}_t)\,\log\frac{\tilde{\pi}_{\mathrm{te}}(x|\mathbf{c}_t)}{\pi_\theta(x|\mathbf{c}_t)}$$

只取 top-k 既把低概率长尾挡在外面（不让小容量学生白学尾巴）、又省显存，作者实测 $k=16$ 在累计概率质量与内存间最划算。而门控用的熵 $H_t^{\mathrm{te}}$ 直接由教师 logits 算出——这些 logits 在标准 OPD 里本就前向算好了，取熵只是多做一次归一化求和，几乎免费。于是整套熵感知机制完全嵌进原有 pipeline：教师本来每个 token 就要查询一次（拿 logprob），现在顺带返回熵和 top-k 集合即可，不引入额外的前向传播或向教师采样，在策略蒸馏 10× 于 GRPO 的效率优势原封不动。

### 训练策略
训练流程沿用标准在策略蒸馏的 PPO 式实现（论文 Algorithm 1）：每轮先用旧策略 $\pi_{\theta_{\mathrm{old}}}$ 在学生自身轨迹上采样，对每个 token 查询教师得到 $(\log\pi_{\mathrm{te}}(x_t|\mathbf{c}_t),\,H_t^{\mathrm{te}},\,\text{top-}k\text{ 集合})$ 存入 rollout buffer；再按上式算 EOPD 损失——reverse KL 项始终计入，forward KL 项只在 $H_t^{\mathrm{te}}>\tau$ 时加上——并用标准优化器更新学生参数。教师模型为 Qwen3-8B（非 thinking 模式），学生分别为 Qwen3-0.6B-Base、Qwen3-1.7B-Base 和 Qwen3-4B-Base；0.6B、1.7B 用 MATH 数据集训练，4B 用更难的 DAPO-Math-14k。

## 实验关键数据

### 主实验

| 学生模型 | 方法 | 6个数学基准 Pass@8 (avg) | vs. 基线 |
|----------|------|-------------------------|----------|
| Qwen3-0.6B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-0.6B-Base | Entropy-Aware (本文) | baseline + 1.37 | **+1.37** |
| Qwen3-1.7B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-1.7B-Base | Entropy-Aware (本文) | baseline + 2.39 | **+2.39** |
| Qwen3-4B-Base | On-Policy (reverse KL) | baseline | — |
| Qwen3-4B-Base | Entropy-Aware (本文) | baseline + 5.05 | **+5.05** |

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| 纯 Reverse KL | 基线水平 | 标准在策略蒸馏，高熵区域多样性差 |
| 纯 Forward KL | 略低于基线 | 全局 mode-covering 导致低熵区域拟合不精确 |
| 固定混合权重 | 小幅提升 | 不随 token 熵变化的静态混合无法最优适配 |
| 熵感知自适应混合（本文） | 最优 | 动态切换兼得精确性与多样性 |

### 关键发现
- 增益随学生模型规模增大而增大（0.6B: +1.37, 1.7B: +2.39, 4B: +5.05），说明更大的学生模型更能受益于保持高熵区域的多样性
- 在 token 级分析中，本文方法显著维持了学生模型的 token 级熵，避免了生成多样性坍缩
- 在高熵 token 上，学生与教师之间的 forward KL 显著降低，表明 student-teacher 对齐更好
- Pass@8 指标的提升比 Pass@1 更显著，进一步验证了多样性保持的重要性——多条推理路径中至少有一条正确的概率更高

## 亮点与洞察
- 将教师 token 级熵作为 mode-seeking / mode-covering 切换信号，简洁而有效——这个设计几乎不增加计算开销（熵从已有 logits 直接算），却带来显著收益
- 揭示了 reverse KL 在语言模型蒸馏中被忽视的"高熵盲区"问题，为蒸馏目标的选择提供了新视角
- 方法的通用性好：可以作为插件应用到任何在策略蒸馏框架中，不需要修改采样策略或网络结构

## 局限与展望
- 当前仅在数学推理任务上验证，尚未在代码生成、开放域对话等其他多样性要求高的任务上验证泛化性
- 熵阈值 $\tau$ 与权重 $\alpha$ 依赖经验调参（取 $\tau=0.8$、$\alpha=1.0$），自适应阈值选择机制有待探索
- 主实验为 Qwen3 同家族蒸馏，附录补充了 Llama-3.1-8B → Llama-3.2-3B 的跨家族验证，但更大规模或更异构的教师-学生组合仍待探索
- 未探讨与其他蒸馏增强技巧（如数据增强、课程学习）的组合效果

## 相关工作与启发
本文延续了语言模型知识蒸馏的研究线，与 GKD（Generalized Knowledge Distillation）、MiniLLM 等在策略蒸馏方法形成对比。关键启发在于：蒸馏目标不应该是全局固定的，而应该根据教师的局部不确定性动态调整。这个 insight 可以迁移到强化学习中的 reward shaping（不确定区域降低 reward 权重）和对比学习中的 hard negative mining（根据 anchor 的熵选择负样本策略）。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](../../ACL2026/model_compression/stable_on-policy_distillation_through_adaptive_target_reformulation.md)
- [\[ICLR 2026\] Distillation of Large Language Models via Concrete Score Matching](../../ICLR2026/model_compression/distillation_of_large_language_models_via_concrete_score_matching.md)
- [\[ICML 2026\] WinQ: Accelerating Quantization-Aware Training of Language Models Around Saddle Points](winq_accelerating_quantization-aware_training_of_language_models_around_saddle_p.md)
- [\[ICML 2026\] Don't Ignore the Tail: Decoupling top-K Probabilities for Efficient Language Model Distillation](dont_ignore_the_tail_decoupling_top-k_probabilities_for_efficient_language_model.md)
- [\[ICML 2026\] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)

</div>

<!-- RELATED:END -->
