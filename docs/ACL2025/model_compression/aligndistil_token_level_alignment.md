---
title: >-
  [论文解读] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation
description: >-
  [ACL 2025][模型压缩][LLM对齐] AlignDistil 证明了 RLHF 目标函数与 token 级蒸馏过程的理论等价性，并据此设计了一种简单的蒸馏方法：用 DPO 模型和反向 DPO 模型的 logit 分布线性组合构造教师分布，配合 token 自适应外推机制实现 token 级奖励优化，在 AlpacaEval 2.0、MT-Bench 和 Arena-Hard 上优于现有方法且收敛更快。
tags:
  - "ACL 2025"
  - "模型压缩"
  - "LLM对齐"
  - "DPO"
  - "蒸馏"
  - "Token级奖励"
  - "偏好优化"
---

# AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation

**会议**: ACL 2025  
**arXiv**: [2503.02832](https://arxiv.org/abs/2503.02832)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: LLM对齐, DPO, 蒸馏, Token级奖励, 偏好优化  

## 一句话总结
AlignDistil 证明了 RLHF 目标函数与 token 级蒸馏过程的理论等价性，并据此设计了一种简单的蒸馏方法：用 DPO 模型和反向 DPO 模型的 logit 分布线性组合构造教师分布，配合 token 自适应外推机制实现 token 级奖励优化，在 AlpacaEval 2.0、MT-Bench 和 Arena-Hard 上优于现有方法且收敛更快。

## 研究背景与动机

**领域现状**：LLM 对齐主要通过 RLHF 和 DPO 实现，但这些方法使用稀疏的 response 级奖励/偏好标注来优化所有 token。

**现有痛点**：response 级反馈是粗粒度的，无法反映每个 token 的个体贡献——可能错误地惩罚高质量 token 或鼓励低质量 token，导致次优性能和慢收敛。

**核心矛盾**：需要 token 级的细粒度奖励信号，但人类标注只能提供 response 级偏好。

**本文目标** 从理论上将 RLHF 的 response 级目标分解为 token 级优化，并实现高效的 token 级对齐。

**切入角度**：利用 DPO 奖励的 token 级分解性质，证明 RLHF 目标等价于一个 token 级蒸馏过程。

**核心 idea**：RLHF = token级蒸馏，教师分布 = DPO logits + 参考模型 logits 的线性组合。

## 方法详解

### 整体框架

从 RLHF 目标出发，引入 DPO 奖励的 token 级分解 → 推导等价的 token 级蒸馏目标 → 学生策略 $\pi_\theta$ 向教师分布 $\pi^*$ 学习，教师分布由 DPO 模型和反向 DPO 模型的 logit 自适应外推构成。

### 关键设计

1. **RLHF-蒸馏等价性**:

    - 功能：证明 RLHF 的 sequence 级目标可以分解为 token 级 KL 散度蒸馏
    - 核心思路：DPO 隐式奖励可以分解到每个 token 位置 $r_t = \beta \log \frac{\pi_{DPO}(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$。将此代入 RLHF 目标后，最优策略在每个 token 位置上等价于教师分布 $\pi^*(t) \propto \exp(\text{logit}_{ref}(t) + \alpha \cdot \text{logit}_{DPO}(t))$
    - 设计动机：将不可处理的 RL 优化问题转化为简单的蒸馏问题

2. **对比 DPO 奖励 (Contrastive DPO Reward)**:

    - 功能：提升 DPO 隐式奖励的准确性
    - 核心思路：训练一个正常 DPO 模型和一个反向 DPO 模型（交换 chosen/rejected），用两者的对比构造更鲁棒的奖励：正向 DPO 强化好 token，反向 DPO 削弱坏 token
    - 设计动机：单独的 DPO 奖励比纯奖励模型准确性差，对比策略弥补了这个差距

3. **Token 自适应 Logit 外推**:

    - 功能：为每个 token 位置构造合适强度的教师分布
    - 核心思路：根据每个 token 位置上 DPO 模型与参考模型的分歧程度 $\alpha_t$，自适应调整外推权重——分歧大的 token 用较小权重避免过度优化，分歧小的 token 用较大权重加强对齐
    - 设计动机：统一用相同权重会导致某些 token 过度优化而另一些欠优化

### 损失函数 / 训练策略
蒸馏损失：$\mathcal{L} = \text{KL}(\pi^*(t) \| \pi_\theta(t))$，对所有 token 位置求和。支持 on-policy（自采样）和 off-policy（用已有数据）训练模式的灵活切换。

## 实验关键数据

### 主实验

基座模型：Llama-3-8B-Instruct. 偏好数据：UltraFeedback.

| 方法 | AlpacaEval 2.0 LC WR ↑ | MT-Bench ↑ | Arena-Hard ↑ |
|------|----------------------|------------|-------------|
| DPO | 25.4 | 7.82 | 28.1 |
| SimPO | 30.2 | 7.88 | 33.5 |
| TDPO | 28.9 | 7.85 | 31.2 |
| **AlignDistil** | **34.7** | **8.05** | **37.8** |

### 消融实验

| 配置 | AlpacaEval 2.0 LC WR | 说明 |
|------|---------------------|------|
| AlignDistil (完整) | **34.7** | 对比DPO + 自适应外推 |
| 无对比 (仅正向DPO) | 31.5 | 收到DPO奖励不准确的影响 |
| 无自适应 (固定α) | 32.3 | 部分token过度/欠优化 |
| Response级奖励蒸馏 | 29.1 | 验证token级别的优势 |

### 关键发现
- **Token 级分布奖励 > Token 级标量奖励 > Response 级奖励**：分布奖励提供最丰富的梯度信号
- **收敛速度显著更快**：AlignDistil 在约一半的训练步骤就达到 DPO 的最终性能
- **对比 DPO 奖励有效弥补了 DPO 作为奖励模型的不足**

## 亮点与洞察
- **RLHF=蒸馏的理论等价性**：将复杂的 RL 优化问题优雅地转化为标准的知识蒸馏问题，极大简化了实现
- **反向 DPO 模型的巧妙使用**：通过交换 chosen/rejected 训练一个"做相反事情"的模型，两者对比增强了奖励信号的准确性

## 局限与展望
- 需要训练两个 DPO 模型（正向+反向），增加了前置计算成本
- 理论等价性依赖 DPO 奖励的 token 级分解假设，在实际中可能不精确
- 未与最新的 GRPO/RLVR 等方法对比

## 相关工作与启发
- **vs DPO**: DPO 用 response 级偏好直接优化，AlignDistil 通过蒸馏实现 token 级优化，更精细
- **vs TDPO**: TDPO 也做 token 级 DPO，但缺少蒸馏视角和自适应机制

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ RLHF-蒸馏等价性的理论发现非常优雅
- 实验充分度: ⭐⭐⭐⭐ 多基准对比和消融实验完整
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导严谨，公式清晰
- 价值: ⭐⭐⭐⭐ 为 LLM 对齐提供了新的理论视角和实用方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] ConfPO: Exploiting Policy Model Confidence for Critical Token Selection in Preference Optimization](../../ICML2025/model_compression/confpo_exploiting_policy_model_confidence_for_critical_token_selection_in_prefer.md)
- [\[ACL 2025\] Quantification of Large Language Model Distillation](quantification_of_large_language_model_distillation.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](efficient_long_context_language_model_retrieval_with_compression.md)
- [\[ACL 2026\] SRA: Span Representation Alignment for Large Language Model Distillation](../../ACL2026/model_compression/sra_span_representation_alignment_for_large_language_model_distillation.md)
- [\[ACL 2026\] MTA: Multi-Granular Trajectory Alignment for Large Language Model Distillation](../../ACL2026/model_compression/mta_multi-granular_trajectory_alignment_for_large_language_model_distillation.md)

</div>

<!-- RELATED:END -->
