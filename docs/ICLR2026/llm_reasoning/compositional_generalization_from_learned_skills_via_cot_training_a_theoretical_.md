---
title: >-
  [论文解读] Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning
description: >-
  [ICLR 2026][Reasoning][组合泛化] 本文通过信息论泛化界和可解释性分析证明，CoT 训练的核心机制是组合泛化：——模型学会系统性地组合已学的简单技能来解决新颖复杂问题，并内化为两阶段组合推理电路，使中间结果在更浅层提取，释放深层专注于后续推理步骤。 领域现状：CoT 训练（如 DeepSeek-R1 的…
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "组合泛化"
  - "思维链训练"
  - "信息论泛化界"
  - "推理电路"
  - "OOD泛化"
---

# Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning

**会议**: ICLR 2026  
**arXiv**: [2502.04667](https://arxiv.org/abs/2502.04667)  
**代码**: [https://github.com/chen123CtrlS/T-CotMechanism](https://github.com/chen123CtrlS/T-CotMechanism)  
**领域**: AI Safety / LLM推理  
**关键词**: 组合泛化, 思维链训练, 信息论泛化界, 推理电路, OOD泛化

## 一句话总结
本文通过信息论泛化界和可解释性分析证明，CoT 训练的核心机制是**组合泛化**——模型学会系统性地组合已学的简单技能来解决新颖复杂问题，并内化为两阶段组合推理电路，使中间结果在更浅层提取，释放深层专注于后续推理步骤。

## 研究背景与动机

**领域现状**：CoT 训练（如 DeepSeek-R1 的长 CoT 冷启动、OpenAI O1 的 RFT）已成为提升 LLM 推理能力的核心范式，但其增强泛化能力的**机制**仍然不清楚。

**现有痛点**：
   - 先前理论分析聚焦于 CoT 提升 Transformer 的表达力/计算复杂度类，未解释训练过程中能力如何涌现
   - 无 CoT 训练的模型在 ID 泛化上存在"组合性缺口"——知道所有基本事实但无法组合
   - 更关键的是，无 CoT 模型完全无法实现 OOD 泛化（新的组合模式）

**核心矛盾**：CoT 为什么能让模型从 ID 泛化扩展到 OOD 泛化？是只学了"what to think"（正确答案），还是学了"how to think"（推理方法）？

**本文目标**
   - (Q1) CoT 训练是否提升 ID 和 OOD 泛化？理论原理是什么？
   - (Q2) 这种泛化能力在模型内部如何实现？

**切入角度**：将 CoT 分解为 $P(Y|X) = \sum_C P(Y|X,C) \cdot P(C|X)$，其中 $C$ 是推理链。CoT 训练显式学习 $P(C|X)$ 和 $P(Y|X,C)$，而无 CoT 训练只学 $P(Y|X)$。

**核心 idea**：CoT 训练教会模型"如何思考"——通过将复杂问题分解为已学简单技能的组合（$P(C|X)$ + $P(Y|X,C)$），从而将 OOD 问题拉近到 ID 分布，实现系统性泛化。

## 方法详解

### 整体框架
全文不提新模型，而是回答一个机制问题：CoT 训练凭什么能把模型从"只会背已见过的组合"推到"会处理没见过的新组合"？作者沿两条线展开。一条是**理论线**：把 CoT 写成 $P(Y|X)=\sum_C P(Y|X,C)\cdot P(C|X)$ 的分解，推导一个信息论泛化界，把测试误差拆成 ID 和 OOD 两块，再逐项说明为什么 CoT 训练能把 OOD 那块压下去；顺着这个界还能进一步分析训练链里掺入错误步骤时机制的鲁棒性。另一条是**结构线**：拿一个 8 层 GPT-2 在两跳推理任务 $(e_1,r_1,r_2)\to e_3$ 上做 Logit Lens 和因果追踪，打开模型内部，看 CoT 训练到底改了什么计算路径。两条线最后汇到同一个结论——CoT 训练学的是"how to think"（把难题拆成已学技能再组合），而不是"what to think"（死记答案）。

> 本文是理论 + 可解释性分析，没有可画的多阶段处理流水线（"框架"是「信息论泛化界」和「内部电路解剖」两条并行分析线索汇到同一结论），故不配框架图；下面的关键设计与上述两条线索一一对应。

### 关键设计

**1. 信息论泛化界：把 OOD 误差拆开，证明 CoT 训练能逐项压低它**

要解释 CoT 为什么能 OOD 泛化，得先有个能区分 ID 和 OOD 的误差刻度。作者推导出泛化误差上界正比于

$$\sqrt{\tfrac{1}{N}\big[(1-\alpha)D_{KL}(P_{test}^{ID}\,\|\,P_{train}) + \alpha\,D_{KL}(P_{test}^{OOD}\,\|\,P_{train})\big]}$$

ID 那一项不难压：测试和训练共享同样的组合模式，$D_{KL}\to 0$。难点全在 OOD 项。无 CoT 训练时，模型只学了 $P(Y|X)$，对推理链 $C$ 相当于持有一个均匀先验，碰到新组合时这个 $D_{KL}$ 很大，于是 OOD 误差压不下去。有 CoT 训练时，这一项可以进一步分解为 $D_{KL}(P_{test}^{OOD}(C|X)\,\|\,P_{train}(C|X)) + \mathbb{E}[D_{KL}(P_{test}^{OOD}(Y|X,C)\,\|\,P_{train}(Y|X,C))]$。关键在于：训练数据里已经包含了构成这些新组合的简单技能（即 $P(C|X)$ 怎么走链、$P(Y|X,C)$ 每步怎么算），所以这两项都能很小。换句话说，OOD 的新问题被"拉回"到由已学技能张成的 ID 分布里——这正是把"训练集上表现好"升级成"未见组合上也表现好"的理论依据（Theorem 1 & 2）。

**2. 两阶段组合推理电路：CoT 训练把中间结果的提取挪到更浅的层**

理论说 CoT 学到了组合能力，那它在模型内部长什么样？作者用 Logit Lens 和因果追踪解剖 8 层 GPT-2 在两跳推理上的中间状态，发现 CoT 训练后模型固化成一个清晰的**两阶段电路**：第一阶段在浅层（第 0 层到第 $l$ 层）从输入 $e_1,r_1,r_2$ 里提取出桥接实体 $e_2$，把它存进状态 $E[l,r_2]$；第二阶段在深层（第 $l$ 层到第 8 层）拿 $e_2$ 做第二跳推理得到最终答案 $e_3$。更有意思的是这个分界层 $l$ 的位置：CoT 训练让 $e_2$ 在第 3 层（ID 设置）就被提取出来，而无 CoT 训练要拖到第 5 层。早提取意味着后面留给第二跳推理的层数更多——CoT 不只是让模型"把中间结果说出来"，而是从根上重排了计算的深度分配，等于扩大了模型的"有效深度"。

**3. 噪声鲁棒性分析：训练链里掺错也照样能组合泛化**

实际的 CoT 训练数据（如 DeepSeek-R1 的 60 万条长 CoT）里推理步骤难免有错，那这套组合机制扛不扛得住？作者在训练数据的中间步骤里按比例 $\xi$ 注入噪声来测（第二跳噪声扫 $\xi\in\{0.05,0.2,0.4,0.6,0.8\}$）。结论是只要噪声不过分，组合泛化仍然成立：$\xi<0.2$ 时模型几乎不受影响，ID 和 OOD 泛化都保住；$\xi$ 再加大，两者的泛化都随之单调下滑，泛化误差上界同步抬高（与 Theorem 3 的预测一致）。机理上，由于噪声只掺在第二跳，第一跳电路照样学得很干净，受冲击的主要是第二跳电路。这就解释了为什么实践中带瑕疵的大规模 CoT 数据依然有效——错误率落在可容忍区间内，组合泛化对噪声是稳健的。

### 损失函数 / 训练策略
两种训练的差别就在 loss 里要不要显式建桥接实体 $e_2$。无 CoT 训练只盯最终答案，$\mathcal{L}=\mathbb{E}[\ell(e_3,\mathcal{M}(e_1,r_1,r_2))]$；有 CoT 训练则同时预测桥接实体和最终答案，$\mathcal{L}=\mathbb{E}[\ell(e_3,\mathcal{M}(e_1,r_1,r_2,\hat{e}_2))+\ell(e_2,\mathcal{M}(e_1,r_1,r_2))]$，正是后一项把"先求 $e_2$ 再求 $e_3$"的两阶段结构压进了模型。训练用自回归 next-token prediction（非 teacher-forcing）。

## 实验关键数据

### 主实验（可控环境，2000实体×200关系）

| 方法 | ID 准确率 | OOD 准确率 | 收敛步数 |
|------|----------|----------|---------|
| 无 CoT (grokking) | ~100% (延迟) | ~0% | >1M 步 |
| **有 CoT** | **~100%** | **~90%+** | **~4000 步** |

### 消融实验（λ = 两跳/单跳数据比，CoT 训练）

| λ 值 | OOD 泛化速度 | 最终 OOD 准确率 |
|------|-------------|--------------|
| 0.001 (极少两跳数据) | 最快 | ~85% |
| 0.9 | 快 | ~90% |
| 7.2 | 中等 | ~95% |
| 12.6 | 慢 | ~95% |

### 关键发现
- **CoT 训练将收敛速度加速 250 倍**：~4000 步 vs >1M 步（无 CoT）
- **CoT 实现 OOD 组合泛化**：无 CoT 模型即使训练百万步仍为 0%，CoT 模型在 ~4000 步达到 90%+
- **更少的两跳数据反而加速 OOD 泛化**：$\lambda=0.001$（极少两跳数据）时 OOD 泛化反而更快——类似 OpenAI O1 的 RFT（少量数据微调即可获得推理能力）
- **浅层提取中间结果**：CoT 训练使 $e_2$ 在 layer 3 就被提取（无 CoT 需要 layer 5），释放更多层用于后续推理
- **两层 Transformer 就足够学习组合电路**：CoT 训练的组合电路可在 2 层模型中完整涌现
- **噪声鲁棒**：$\xi < 0.2$ 时几乎不影响泛化，解释了实际 CoT 数据中错误的可容忍性

## 亮点与洞察
- **"how to think vs what to think"的形式化**：将 CoT 的优势精确表述为将 $P(Y|X)$ 分解为 $P(C|X) \cdot P(Y|X,C)$——CoT 教会模型推理过程（$P(C|X)$），而非直接答案。这个信息论框架非常优雅。
- **浅层提取中间结果**的发现是对 Transformer 推理机制的重要补充——CoT 训练本质上是在教模型更高效地利用深度，将不同推理步骤分配到不同层。这解释了为什么 CoT 训练增大了模型的"有效深度"。
- **少数据加速泛化的反直觉发现**有很强的实践指导意义——RFT/SFT 不需要大量 CoT 数据，少量高质量 CoT 数据可能比大量数据更快激发泛化能力。

## 局限与展望
- 主要在合成数据（entity-relation）上验证，真实 NLP 任务的验证仅在附录中
- 仅分析两跳推理，多跳（>3 步）场景的组合电路结构待研究
- 信息论界是上界，可能不紧——实际泛化可能比理论预测更好或更差
- 未分析在 RL 微调（GRPO/PPO）设置下的组合泛化行为
- "简单已学技能"在真实世界任务中如何定义和识别仍是开放问题

## 相关工作与启发
- **vs Wang et al. [102]**: 同样研究 Transformer 的组合推理电路，但他们发现系统化电路仅在 ID 设置下出现；本文证明 CoT 训练将其扩展到 OOD
- **vs Feng et al. [17]**: 他们从表达力角度证明 CoT 增加有效深度；本文从泛化和内部结构角度补充了"如何实现"
- **vs COCONUT/CoT2**: 这些工作探索连续推理空间；本文的组合泛化理论可以作为理解这些方法的基础

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从信息论+内部电路双视角形式化 CoT 训练的泛化机制
- 实验充分度: ⭐⭐⭐⭐ 合成实验非常系统，但真实任务验证不够充分
- 写作质量: ⭐⭐⭐⭐⭐ 理论与实验完美交织，论证链条清晰有力
- 价值: ⭐⭐⭐⭐⭐ 对 CoT 训练的理论理解有里程碑意义，对 RFT/SFT 实践有直接指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] The CoT Encyclopedia：分析、预测并控制推理模型的思考方式](the_cot_encyclopedia_analyzing_predicting_and_controlling_how_a_reasoning_model_.md)
- [\[ICLR 2026\] Analytica: Soft Propositional Reasoning for Robust and Scalable LLM-Driven Analysis](analytica_soft_propositional_reasoning_for_robust_and_scalable_llm-driven_analys.md)
- [\[ICLR 2026\] TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis](tumorchain_interleaved_multimodal_chain-of-thought_reasoning_for_traceable_clini.md)
- [\[ICLR 2026\] STAT: Skill-Targeted Adaptive Training](stat_skill-targeted_adaptive_training.md)
- [\[ICLR 2026\] RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems](rlad_training_llms_to_discover_abstractions_for_solving_reasoning_problems.md)

</div>

<!-- RELATED:END -->
