---
title: >-
  [论文解读] Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks
description: >-
  [ICLR 2026 Oral][LLM对齐][MoE] 系统研究 MoE 语言模型的稀疏度如何不同地影响记忆性任务和推理性任务：记忆任务偏好更高稀疏度（更多参数），而推理任务在 TPP≈20 附近达到最优，且该趋势在 GRPO 后训练和测试时计算增加后仍然不变。 经典 scaling laws（Kaplan et al.…
tags:
  - "ICLR 2026 Oral"
  - "LLM对齐"
  - "MoE"
  - "scaling laws"
  - "sparsity"
  - "推理"
  - "记忆"
  - "tokens per parameter"
  - "GRPO"
  - "测试时计算"
---

# Optimal Sparsity of Mixture-of-Experts Language Models for Reasoning Tasks

**会议**: ICLR 2026 Oral  
**arXiv**: [2508.18672](https://arxiv.org/abs/2508.18672)  
**代码**: [GitHub](https://github.com/rioyokotalab/optimal-sparsity)  
**领域**: LLM对齐  
**关键词**: MoE, scaling laws, sparsity, 推理, 记忆, tokens per parameter, GRPO, 测试时计算

## 一句话总结

系统研究 MoE 语言模型的稀疏度如何不同地影响记忆性任务和推理性任务：记忆任务偏好更高稀疏度（更多参数），而推理任务在 TPP≈20 附近达到最优，且该趋势在 GRPO 后训练和测试时计算增加后仍然不变。

## 研究背景与动机

经典 scaling laws（Kaplan et al., 2020; Hoffmann et al., 2022）建立了预训练损失与模型规模/数据量/计算预算之间的幂律关系，成为模型规划的基石。但这些定律有重要局限：

**系数不通用**：架构或数据管道改变后需重新估计

**MoE 引入新维度**：MoE 模型通过稀疏路由以固定 FLOPs 获得高容量，成为 Gemini 2.5 Pro、DeepSeek-V3、Qwen3 等旗舰模型的标准配置，但稠密模型的 scaling 前沿无法覆盖稀疏度这一维度

**损失≠性能**：相同预训练损失的模型在下游推理基准上可能表现迥异（GLM-4.5 Team 也有此观察）

**后训练和 TTC 的影响未知**：GRPO 和测试时计算是否能弥补预训练稀疏度选择的不足？

## 方法详解

### 整体框架

这篇论文不提出新模型，而是用一套**控制变量的扫描实验**回答两个问题：MoE 的稀疏度对"记忆"和"推理"两类能力的影响是否一样、预训练阶段定下的稀疏度能否事后补救。整体做法分三步走。第一步训练一个 Mixtral 架构的 MoE 模型家族，系统扫描模型宽度 $d \in \{512, 1024, 2048\}$、每层专家数 $E \in \{8, 16, 32, 64, 128, 256\}$、激活专家数 top-$k \in \{2, 4, 8, 16\}$（固定 16 层、统一训练 125B token），并把稀疏度定义为 $\text{sparsity} = 1 - k/E$（激活越少越稀疏）。第二步对每个模型同时测三个层次的指标，把"损失好"和"答得对"拆开看。第三步沿 Active FLOPs 和 TPP 两个轴去解释扫描出来的趋势，再用 GRPO 后训练和测试时计算两组实验检验这些趋势会不会被事后抹平。

为了让结论可归因到稀疏度本身而非调参，所有模型共用同一套训练配方：AdamW 优化器，峰值学习率 $4 \times 10^{-4}$，2k 步线性 warmup 接 cosine decay，weight decay 0.1；数据为 125B token 的平衡混合（web 43B、math 32B、STEM 49B、code 1B）。统一配方让宽度 / 专家数 / top-k 的扫描成为干净的控制变量实验。

> 本文是 scaling-law / 分析型论文，方法即"如何设计扫描与归因"，没有多阶段 pipeline 或多模块协同的数据流，**故不画 Mermaid 框架图**；下面三个关键设计已按整体框架的"测量 → 归因 → 稳健性检验"三步同序对齐。

### 关键设计

**1. 三层测量：把"损失好"和"答得对"拆成可分别量化的两件事**

经典 scaling law 只盯预训练损失，但本文要回答的恰恰是"损失继续降为什么没换来推理变强"。为此对每个模型同时测三个层次：预训练数据上的 train/val loss、下游基准上的 task loss（只在答案 token 上算交叉熵）、以及下游基准上的准确率。这三层拆开后，"训练分布 → 测试分布"的泛化差距和"损失 → 准确率"的映射差距就能分别量化——后面发现的"推理任务上 train loss 单调下降、task loss 却走 U 型先降后升、准确率随过度优化反而退化"正是靠这套测量才暴露的；而记忆任务（TriviaQA、HellaSwag）三层始终同向单调，对照之下分歧才看得清楚。

**2. 两轴归因：用 Active FLOPs 和 TPP 解释稀疏度的相反效应**

这是全文的分析骨架——把稀疏度对性能的影响拆到两个正交的轴上。**Active FLOPs 轴**指训练和推理时实际激活的算力（由 top-$k$ 决定）：在相同预训练损失下，激活算力更大的模型推理准确率一致更高，说明推理质量并不只由损失决定，还取决于实际激活的 FLOPs。**Total Tokens Per Parameter (TPP) 轴**定义为训练 token 总量除以参数量，

$$\text{TPP} = N_{\text{tokens}} / N_{\text{params}}$$

本文 $N_{\text{tokens}}$ 固定为 125B，故 TPP 随参数量反向变化；它刻画模型是"参数饥渴"还是"数据饥渴"：记忆任务属参数饥渴型，TPP 越低（参数越多）越好；推理任务属数据饥渴型，在 $\text{TPP} \approx 20$ 时最优，过低或过高都会退化。两轴合起来看，才能解释为什么"堆参数"对记忆和推理是两套相反的策略，也才支撑起"最优稀疏度须由 Active FLOPs 与 TPP 联合决定、而非仅看计算预算"这一核心结论。

**3. GRPO + TTC 稳健性检验：验证预训练的稀疏度选择能否被事后补救**

预训练阶段定下的稀疏度，能不能靠后训练或测试时多花算力追回来？本文用两组实验把同一批模型重评一遍：**GRPO** 沿用 DeepSeek-R1 的算法，在 GSM8K 训练集上做强化微调；**TTC**（测试时计算）用 zero-shot Self-Consistency 解码，每题采样 $2^7 = 128$ 个候选答案后取多数投票。两组都让所有模型整体变强，但"损失-准确率非单调"的权衡保持不变、较稀疏模型仍追不上较稠密模型——说明稀疏度造成的推理损失无法被后训练或 TTC 抹平，预训练阶段的稀疏度选择才是关键。作者还另外扫了学习率与初始化方案，发现它们引起的泛化差距与稀疏度引起的高度一致，进一步确认这套现象并非 MoE 独有、传统超参也能复现。

### 损失函数

贯穿所有扫描的标准 MoE 训练损失：

$$\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{LB} + \beta \mathcal{L}_{RZ}$$

其中交叉熵 $\mathcal{L}_{CE}$ 是主项，load-balancing 损失 $\mathcal{L}_{LB}$（$\alpha = 10^{-2}$）防止专家坍缩，router z-loss $\mathcal{L}_{RZ}$（$\beta = 10^{-3}$）惩罚过大的路由器 logits 以保持数值稳定。

## 实验关键数据

### 主实验

**记忆 vs 推理任务在增加总参数时的分歧（Figure 1-3）**：

| 维度 | TriviaQA/HellaSwag (记忆) | GSM8K/GSM-Plus (推理) |
|------|--------------------------|---------------------|
| 预训练损失 | 随总参数增加单调下降 ✓ | 随总参数增加单调下降 ✓ |
| Task Loss | 随预训练损失下降单调改善 ✓ | U 型曲线：先降后升 ✗ |
| 准确率 | 随预训练损失下降单调提升 ✓ | 非单调：过度优化反而损害 ✗ |

**Iso-FLOP 分析下的最优稀疏度（Figure 5）**：

| 任务类型 | 低 FLOPs 预算 | 高 FLOPs 预算 |
|---------|-------------|-------------|
| 记忆型 | 更高稀疏度更优 | 更高稀疏度更优（一致） |
| 推理型 | 更高稀疏度更优 | **更稠密模型反超**（逆转） |

**TPP 对性能的影响（Figure 7）**：

| 任务类型 | TPP 趋势 | 最优 TPP |
|---------|---------|---------|
| TriviaQA/HellaSwag | 单调：TPP 越低越好 | 尽可能低 |
| GSM8K/GSM-Plus | 非单调倒 U 型 | **≈20** |

### 消融实验

**Top-k 在固定激活参数下的影响**：
- 固定激活参数量时改变 top-k 对预训练损失影响可忽略
- 但在推理任务上，更大 top-k 一致优于更小 top-k（即使 TPP 固定）

**GRPO 后训练效果（Figure 6 右）**：
- 所有模型性能提升，但预训练损失与准确率之间的非单调关系**保持不变**
- 较稀疏模型在 GRPO 后仍不如较稠密模型

**TTC 效果（Figure 6 左，Self-Consistency $2^7$ 采样）**：
- 性能随模型规模扩展，但损失-准确率权衡**保持不变**
- TTC 无法弥补预训练稀疏度的不足

**超参数控制实验**：
- 扫描学习率和初始化方案，其效果与稀疏度导致的泛化差距惊人一致
- 确认记忆/推理性能差距不仅由稀疏度引起，传统超参数也可复现

**代码任务消融（HumanEval, MBPP）**：
- 代码生成表现出与数学推理类似的趋势：高 FLOPs 预算下稠密模型更优

### 关键发现

1. 预训练损失下降不一定带来推理性能提升——在 MoE 中可能反而有害
2. 最优稀疏度必须由 Active FLOPs 和 TPP 联合决定，而非仅考虑计算预算
3. GRPO 和 TTC 均无法消除预训练稀疏度造成的推理性能损失
4. 记忆偏好稀疏（更多参数），推理偏好适度稠密（更多数据/参数）

## 亮点与洞察

1. **挑战经典 scaling wisdom**：揭示了"更多参数总是更好"在 MoE 推理任务上不成立的反直觉结论
2. **实验设计精巧**：通过控制 top-k/宽度/专家数的系统扫描，clean 地解耦了多个混杂因素
3. **实用指导**：为 MoE 模型的预训练规划提供了清晰的决策框架——记忆任务堆参数，推理任务控制 TPP≈20
4. **后训练无法弥补**：GRPO 和 TTC 均不改变底层权衡，强调预训练阶段稀疏度选择的关键性
5. **全面开源**：checkpoints、代码、训练日志均开源，可复现程度高

## 局限性

1. 所有模型仅训练 125B token，更大数据集可能改变最优稀疏度（作者承认）
2. 仅使用 Mixtral 架构，未验证 shared experts、QK-norm 等现代变体
3. 评估基准有限（GSM8K/TriviaQA/HellaSwag），缺乏 MATH、ARC-C 等更难推理基准
4. 最大宽度 d=2048 限制了对真正大模型行为的外推
5. 深度固定为 16 层，深度与稀疏度的交互未充分探索

## 相关工作与启发

与 Abnar et al.（2025）的 MoE 参数-FLOPs 前沿分析不同，本文进一步区分了记忆和推理任务的不同最优策略。与 Jelassi et al.（2025）的理论分析（MoE 增加专家改善记忆多于推理）在实验上相互印证。与 Roberts et al.（2025）的 TPP 分析（记忆参数饥渴、推理数据饥渴）高度一致。

**核心启发**：在规划大规模 MoE 训练时，不能仅看 perplexity 曲线。必须同时监控下游推理基准，并根据目标任务类型（记忆 vs 推理）选择不同的稀疏度策略。TPP≈20 是推理任务的"甜蜜点"，值得作为工程经验法则。

## 评分

- 新颖性: ⭐⭐⭐⭐ (针对 MoE 推理性能的稀疏度分析填补了重要空白)
- 实验充分度: ⭐⭐⭐⭐⭐ (大量系统性扫描、多种消融、GRPO+TTC 验证、代码任务扩展)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，图表信息密度高，讨论诚实)
- 价值: ⭐⭐⭐⭐⭐ (对 MoE 模型工程和 scaling law 研究具有直接实用价值)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Upcycling Instruction Tuning from Dense to Mixture-of-Experts via Parameter Merging](../../ACL2025/llm_alignment/upcycling_instruction_tuning_from_dense_to_mixture-of-experts_via_parameter_merg.md)
- [\[ICLR 2026\] Multi-objective Large Language Model Alignment with Hierarchical Experts](multi-objective_large_language_model_alignment_with_hierarchical_experts.md)
- [\[ICLR 2026\] Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks](hierarchy-of-groups_policy_optimization_for_long-horizon_agentic_tasks.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](../../NeurIPS2025/llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)
- [\[ICLR 2026\] Cognitive models can reveal interpretable value trade-offs in language models](cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models.md)

</div>

<!-- RELATED:END -->
