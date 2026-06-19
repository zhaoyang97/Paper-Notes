---
title: >-
  [论文解读] Align-then-Unlearn: Embedding Alignment for LLM Unlearning
description: >-
  [ICML 2025][LLM安全][LLM 遗忘] 提出 Align-then-Unlearn 框架，通过在语义嵌入空间（而非 token 级别）执行遗忘操作，先训练嵌入预测模块对齐未来语义表示，再微调 LLM 使预测嵌入远离目标概念嵌入，实现对 prompt 改写鲁棒的概念级知识遗忘。 领域现状：LLM 在大规模数据上训…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "LLM 遗忘"
  - "嵌入空间"
  - "语义遗忘"
  - "隐私保护"
  - "概念级遗忘"
---

# Align-then-Unlearn: Embedding Alignment for LLM Unlearning

**会议**: ICML 2025  
**arXiv**: [2506.13181](https://arxiv.org/abs/2506.13181)  
**代码**: [https://github.com/ExplainableML/align-then-unlearn](https://github.com/ExplainableML/align-then-unlearn)  
**领域**: AI安全  
**关键词**: LLM 遗忘, 嵌入空间, 语义遗忘, 隐私保护, 概念级遗忘

## 一句话总结
提出 Align-then-Unlearn 框架，通过在语义嵌入空间（而非 token 级别）执行遗忘操作，先训练嵌入预测模块对齐未来语义表示，再微调 LLM 使预测嵌入远离目标概念嵌入，实现对 prompt 改写鲁棒的概念级知识遗忘。

## 研究背景与动机

**领域现状**：LLM 在大规模数据上训练后会不可避免地保留敏感信息（个人隐私、版权内容等），machine unlearning 旨在从已训练模型中选择性移除特定数据影响。

**现有痛点**：现有 SOTA 方法（如 Gradient Ascent、DPO、NPO）都在 token 级别操作——通过 forget set 中的特定文本序列来定义遗忘目标。这导致两个问题：(a) 遗忘范围难以精确控制，因为 forget set 可能很大；(b) 对 prompt 改写不鲁棒，简单换个问法就能绕过遗忘。

**核心矛盾**：token 级遗忘只是在输出层面做"掩盖"，并未真正从模型的语义表示中移除目标知识，导致相关概念仍可通过其他路径被提取。

**本文目标**：如何实现概念级别的、对改写鲁棒的知识遗忘？

**切入角度**：既然 token 粒度太细，不如在语义嵌入空间操作——用一个嵌入向量整体表示"需遗忘的概念"，将模型的内部表示推离该概念。

**核心 idea**：先训练嵌入预测头对齐语义空间，再利用该预测头作为"探针"引导 LLM 的隐状态远离目标概念嵌入。

## 方法详解

### 整体框架
Align-then-Unlearn 分两阶段：
- **Phase 1 - Alignment Pre-training**：给 LLM 加一个小型嵌入预测模块 $E$，训练它将 LLM 隐状态映射到预训练文本编码器（MPNet）生成的未来 token 语义嵌入空间。
- **Phase 2 - Unlearning**：冻结 $E$，用目标概念（如"Stephen King"）的嵌入 $e_{\text{unlearn}}$ 作为锚点，微调 LLM 使其预测嵌入 $\hat{e}_t$ 与 $e_{\text{unlearn}}$ 的相似度最小化。

输入是 token 序列 $(x_1, \dots, x_T)$，模型生成隐状态 $(h_1, \dots, h_T)$，嵌入预测头将 $h_t$ 映射到 $\hat{e}_t = E(h_1, \dots, h_t)$，该嵌入整体代表接下来 $k$ 个 token 的语义。

### 关键设计

1. **嵌入预测模块 (Embedding Prediction Head)**:

    - 功能：将 LLM 隐状态映射到语义嵌入空间，预测未来 $k$ 个 token 的整体语义
    - 核心思路：用 6 层网络、隐藏维度 768，对齐损失为余弦距离 $\mathcal{L}_{\text{align}} = 1 - \text{sim}(\hat{e}_t, e_t)$，其中 $e_t$ 由冻结的 MPNet 对未来窗口 $(x_{t+1}, \dots, x_{t+k})$ 编码后得到
    - 设计动机：相比逐 token 预测，嵌入空间的表示是整体语义的，因此可以捕捉概念级信息而非字面信息

2. **嵌入空间遗忘 (Unlearning in Embedding Space)**:

    - 功能：微调 LLM 参数使预测嵌入远离目标概念
    - 核心思路：遗忘损失 $\mathcal{L}_{\text{unlearn}} = \max(0, \text{sim}(\hat{e}_t, e_{\text{unlearn}}) - \tau)$，只有当相似度超过阈值 $\tau$ 时才施加惩罚
    - 设计动机：阈值 $\tau$ 提供精细控制，防止过度遗忘破坏模型剩余能力；只需要一个文本描述就能定义遗忘目标（如"Stephen King"），无需大量 forget set

3. **迭代对齐-遗忘交替训练**:

    - 功能：交替执行嵌入头重新对齐和 LLM 遗忘更新
    - 核心思路：遗忘后 LLM 隐状态分布改变，嵌入头失效；重新对齐后嵌入头恢复探测能力，可继续推动更深层遗忘
    - 设计动机：形成对抗动态——LLM 试图从 $E$ 角度隐藏目标概念，而 $E$ 不断恢复探测能力，迫使 LLM 在更深层表示上执行真正的知识删除

### 损失函数 / 训练策略
- 对齐阶段：$\theta_E^* = \arg\min_{\theta_E} \mathbb{E}[\mathcal{L}_{\text{align}}]$，仅训练嵌入头
- 遗忘阶段：$\theta_M^* = \arg\min_{\theta_M} \mathbb{E}[\mathcal{L}_{\text{unlearn}}]$，冻结嵌入头，微调 LLM
- 动态阈值递减策略：逐步降低 $\tau$ 实现渐进式遗忘

## 实验关键数据

### 主实验
在 RWKU 基准上，基于 Phi-3-mini-4k-instruct，与 SOTA 方法对比（15 个遗忘目标的平均）：

| 方法 | Forget FB ↓ | Forget QA ↓ | Forget AA ↓ | Neighbor QA ↑ | MMLU ↑ |
|------|------------|------------|------------|--------------|--------|
| Before Unlearning | 47.1 | 47.4 | 55.8 | 61.4 | 64.4 |
| GA (Full) | 17.8 | 14.3 | 26.3 | 51.7 | 64.3 |
| DPO (Full) | 25.0 | 19.1 | 29.9 | 39.6 | 63.0 |
| NPO (Full) | 22.5 | 16.9 | 27.3 | 53.6 | 64.2 |
| **ATU (20%)** | **13.5** | **15.3** | **25.9** | 52.3 | **64.5** |

### 消融实验

| 配置 | Forget QA ↓ | Neighbor QA ↑ | MMLU ↑ | 说明 |
|------|------------|--------------|--------|------|
| ATU (50% threshold) | 40.5 | 64.4 | 64.2 | 轻度遗忘，邻居知识保留最好 |
| ATU (35% threshold) | 24.8 | 56.4 | 64.8 | 中等遗忘 |
| ATU (20% threshold) | 15.3 | 52.3 | 64.5 | 深度遗忘 |
| Layer 10 | 54.32* | - | - | 遗忘效果因层而异 |
| Layer 20 | 12.40* | - | - | 某些目标在中间层效果最好 |

*单个目标（Warren Buffett）的结果

### 关键发现
- ATU 在 20% 阈值下达到最低 Forget FB（13.5%），同时 MMLU 保持 64.5%（甚至略高于原始模型的 64.4%）
- 不同遗忘目标在不同层的表现差异很大，暗示概念知识在网络中的分布不均匀
- 遗忘与邻居知识保留存在持续的 trade-off

## 亮点与洞察
- **概念级 vs token 级遗忘**的视角转换非常巧妙——一个嵌入向量就能定义遗忘目标，数据效率极高，不需要精心构造大规模的 forget set
- **对抗式交替训练**设计精巧：嵌入头不断"追赶"LLM 的变化，迫使遗忘发生在深层表示而非浅层遮蔽，类似 GAN 的思路但用于遗忘
- 阈值 $\tau$ 提供了可调的遗忘-性能 trade-off 开关，比大多数方法更具可控性

## 局限与展望
- 阈值 $\tau$ 缺乏自适应调整机制，在不同目标间迁移效果不稳定
- 邻居知识的丧失仍然显著，说明嵌入空间中概念间的纠缠难以完全避免
- 仅在 Phi-3-mini 上验证，未测试更大模型（如 70B+）的表现
- 目前仅聚焦实体级遗忘（人名），更复杂的概念（如技术知识、推理模式）的遗忘效果未知
- 未讨论 membership inference attack 下的鲁棒性

## 相关工作与启发
- **vs GA/NPO**: token 级 gradient ascent 方法可以快速降低 forget 分数但对改写不鲁棒；ATU 在嵌入空间操作，理论上更鲁棒
- **vs DPO**: DPO 需要正负样本对，ATU 只需一个概念描述文本
- **vs ICU**: ICU 效果较差（Forget QA 仅降到 34.6%），ATU 大幅超越

## 评分
- 新颖性: ⭐⭐⭐⭐ 嵌入空间遗忘的视角新颖，对抗交替训练设计巧妙
- 实验充分度: ⭐⭐⭐ 仅一个基准、一个模型，缺少改写鲁棒性的定量对比
- 写作质量: ⭐⭐⭐⭐ 清晰简洁，图示直观
- 价值: ⭐⭐⭐⭐ 提出有前景的概念级遗忘范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Improving LLM Safety Alignment with Dual-Objective Optimization](improving_llm_safety_alignment_with_dual-objective_optimization.md)
- [\[ICML 2025\] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)
- [\[ICML 2025\] Targeted Unlearning with Single Layer Unlearning Gradient](targeted_unlearning_with_single_layer_unlearning_gradient.md)
- [\[ICML 2025\] Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning](invariance_makes_llm_unlearning_resilient_even_to_unanticipated_downstream_fine-.md)
- [\[ICLR 2026\] Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges](../../ICLR2026/llm_safety/align_to_misalign_automatic_llm_jailbreak_with_meta-optimized_llm_judges.md)

</div>

<!-- RELATED:END -->
