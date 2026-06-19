---
title: >-
  [论文解读] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach
description: >-
  [ACL 2025][LLM安全][上下文鲁棒性] 提出 Grft（Gated Representation Fine-Tuning），一种轻量级即插即用的门控表示微调方法，仅需不到 200 个训练样本和模型 0.0004% 的参数，即可让 LLM 在面对矛盾、无用的外部上下文时表现出类似人类的鲁棒认知行为。
tags:
  - "ACL 2025"
  - "LLM安全"
  - "上下文鲁棒性"
  - "表示工程"
  - "门控机制"
  - "RAG"
  - "知识冲突"
---

# Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach

**会议**: ACL 2025  
**arXiv**: [2502.14100](https://arxiv.org/abs/2502.14100)  
**代码**: 无  
**领域**: LLM安全  
**关键词**: 上下文鲁棒性, 表示工程, 门控机制, RAG, 知识冲突

## 一句话总结

提出 Grft（Gated Representation Fine-Tuning），一种轻量级即插即用的门控表示微调方法，仅需不到 200 个训练样本和模型 0.0004% 的参数，即可让 LLM 在面对矛盾、无用的外部上下文时表现出类似人类的鲁棒认知行为。

## 研究背景与动机

RAG 等技术通过为 LLM 提供外部上下文来增强其事实准确性，已广泛应用于医疗、法律、金融等领域。然而，LLM 在处理**不完美证据**时面临严重问题：

**过度依赖外部知识**：即使 LLM 自身拥有正确答案，面对矛盾性上下文时准确率会从 ~99% 暴跌到 ~25-35%

**无用上下文干扰**：语义相关但实际无助于回答的上下文同样会导致性能严重退化（从 ~99% 降到 ~44-53%）

**与人类认知的差距**：人类会自然地权衡外部信息与内部知识，而 LLM 缺乏这种能力

论文将**上下文鲁棒 LLM** 定义为应具备四种行为：
- (a) 缺乏内部知识时依赖外部上下文
- (b) 内外知识匹配时使用两者
- (c) 内外知识矛盾时识别矛盾并提供两种答案
- (d) 上下文无用时忽略它，依赖内部知识

现有方法（如 system prompt、ICL、CoT、直接微调）均无法可靠地实现这些行为。

## 方法详解

### 整体框架

Grft 在 LLM 的隐藏层表示上引入轻量级干预函数，由两个组件组成：

$$\text{Grft}(\mathbf{h}_l) = \mathbf{h}_l + \text{Gate}(\mathbf{h}_l) \cdot \text{Intervention}(\mathbf{h}_l)$$

**核心思想**：LLM 的表示在处理矛盾/匹配/有用/无用输入时展现出**内在的不同模式**，通过在表示空间进行精准干预，可以高效地修改模型行为。

### 关键设计

**门控函数（Gate Function）**：

$$\text{Gate}(\mathbf{h}_l) = \sigma(\mathbf{W}_g \mathbf{h}_l + \mathbf{b}_g)$$

- 输入：第 $l$ 层的隐藏表示 $\mathbf{h}_l \in \mathbb{R}^d$
- 输出：0到1的标量，控制干预强度
- 设计目标：对"正常"输入（未知问题+有用上下文、已知问题+匹配上下文）输出低值；对"异常"输入（矛盾/无用上下文）输出高值
- 关键：仅增加约 4.1K 参数

**干预函数（Intervention）**：

$$\text{Intervention}(\mathbf{h}_l) = \mathbf{R}^\top(\mathbf{W}\mathbf{h}_l + \mathbf{b} - \mathbf{R}\mathbf{h}_l)$$

- 采用低秩表示微调，在低维空间中学习对表示的干预
- $\mathbf{W}$ 和 $\mathbf{R}$ 为低秩矩阵（rank=4），$\mathbf{b}$ 为偏置
- 与 ReFT 的关键区别在于引入了 Gate 门控机制

**训练数据构建**（仅需100个已知问题 + 100个未知问题）：
- 未知问题对（gate label=0）：LLM 不知道答案，提供正确上下文
- 匹配样本（gate label=0）：上下文与内部知识一致
- 矛盾样本（gate label=1）：上下文与内部知识冲突，期望输出识别矛盾并给出两种答案
- 无用样本（gate label=1）：上下文无法帮助回答，期望忽略并用内部知识回答

### 损失函数 / 训练策略

总损失由两部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{FT}}(\hat{y}_i, y_i) + \mathcal{L}_{\text{gate}}(\text{Gate}(\mathbf{h}_l^i), z_i)$$

- $\mathcal{L}_{\text{FT}}$：标准交叉熵损失，监督输出行为
- $\mathcal{L}_{\text{gate}}$：二元交叉熵损失，监督门控值的正确激活

训练时冻结基础模型参数，仅更新 Grft 的可学习参数。rank=4，batch size=5，训练100轮。

**推理策略**：
- **Grft 直接生成**：直接使用干预后的模型生成输出
- **Grft-requery**：当输出包含"CONTRADICTORY"或"UNHELPFUL"标记时，重新查询原始 LLM 获取内部答案

## 实验关键数据

### 主实验

在 ConflictQA 子集上，以 Llama-2-7B-Chat 为主模型：

**矛盾上下文**（已知问题 + 矛盾证据）：

| 方法 | 短上下文 | 长上下文 |
|---|---|---|
| LLM 原始 | 34.55% | 25.33% |
| CoT | 41.83% | 36.02% |
| Astute-RAG | 59.60% | 46.70% |
| **Grft** | **60.88%** | **61.19%** |
| **Grft-requery** | **82.49%** | **88.15%** |

**无用上下文**（已知问题 + 干扰/随机证据）：

| 方法 | 随机 | 干扰 |
|---|---|---|
| LLM 原始 | 53.14% | 44.62% |
| **Grft** | **73.22%** | **68.86%** |
| **Grft-requery** | **97.98%** | **98.46%** |

**正常输入（匹配/有用上下文）**：Grft 保持了接近原始 LLM 的高性能（99.07%/98.23%/97.03%），不会损害正常场景。

### 消融实验

- **去掉 Gate**（Grft-W/O Gate）：在噪声输入上有效但在正常输入上退化（有用长上下文：89.03% vs 98.23%），证明 Gate 对避免过度干预至关重要
- **有 Gate 但无 gate loss**（Grft-W/O Loss）：性能介于有/无 Gate 之间，说明 gate loss 的监督信号很重要
- **参数效率**：Grft 仅需 36.9K 参数（0.0005%），远低于 LoRA 的 2.1M（0.0311%）和全量微调的 6.74B

### 关键发现

1. **门控值可视化**：矛盾输入的平均 gate 值 >0.7，无用输入接近 1.0；匹配/有用输入 <0.3。Gate 成功区分了正常与异常输入
2. **泛化性**：在 COUNTERFACT 和 NQ 数据集上（未参与训练），Grft 仍然有效（COUNTERFACT 矛盾：62.52%，比原始 LLM 高 20.28%）
3. **Grft-requery 一致性优势**：重新查询策略在所有场景下都大幅提升性能，且不损害正常输入

## 亮点与洞察

- **极致的参数效率**：仅 0.0004% 的参数量和不到 200 个训练样本，却能实现显著的行为改变，这得益于在表示空间而非参数空间进行干预
- **门控机制设计精巧**：自适应地决定是否干预，避免了"一刀切"导致的正常场景退化，这是相比 ReFT 的核心改进
- **从表示工程角度解决 RAG 鲁棒性问题**：不同于修改 prompt 或大规模微调，直接在 LLM 内部表示上进行最小化干预，方向新颖且有效
- 即插即用的设计使其可以轻松集成到现有 RAG 系统中

## 局限与展望

- 目前仅处理**单个**上下文-问题对，未考虑多文档交互场景下更复杂的内外知识关系
- 实验主要基于 Llama-2-7B 和 Llama-3-8B，需要在更多模型上验证泛化性
- 矛盾检测的"二元"标签可能过于简化：现实中矛盾程度可能是连续谱
- 对于 LLM 内部知识本身就是错误的情况，该方法可能过度信任内部知识
- 推理时 Grft-requery 需要两次查询模型，增加了计算开销

## 相关工作与启发

- 表示工程（Representation Engineering, Zou et al., 2023）发现 LLM 表示在处理对比概念时有特征模式，是本工作的核心启发
- ReFT（Wu et al., 2024b）提供了低秩表示微调的基础框架，Grft 在此基础上加入门控机制
- 知识冲突研究（Xie et al.; Ying et al., 2024b）揭示了 LLM 在内外知识冲突时的脆弱性
- 可以启发更广泛的 LLM 行为控制方法：通过表示空间的精准干预来实现特定行为

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将门控机制与表示微调结合来解决 RAG 鲁棒性是新颖的思路
- **实用性**: ⭐⭐⭐⭐⭐ — 极低成本、即插即用，直接可用于 RAG 系统增强
- **实验充分度**: ⭐⭐⭐⭐ — 消融全面、泛化验证充分，但模型范围可扩展
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，方法动机阐述充分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs](from_misleading_queries_to_accurate_answers_a_three-stage_fine-tuning_method_for.md)
- [\[ICML 2025\] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs](../../ICML2025/llm_safety/tuco_measuring_the_contribution_of_fine-tuning_to_individual_responses_of_llms.md)
- [\[ACL 2025\] Bias in the Mirror: Are LLMs' Opinions Robust to Their Own Adversarial Attacks](bias_in_the_mirror_are_llms_opinions_robust_to_their_own_adversarial_attacks.md)
- [\[ICLR 2026\] Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!](../../ICLR2026/llm_safety/be_careful_when_fine-tuning_on_open-source_llms_your_fine-tuning_data_could_be_s.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)

</div>

<!-- RELATED:END -->
