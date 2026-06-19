---
title: >-
  [论文解读] LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation
description: >-
  [ACL 2025 (XLLM Workshop)][模型压缩][结构化推理] 本文提出 Less is More 框架，在仅有 24 个标注样本的极端低资源条件下，通过逆向提示归纳、GPT-4o 增强的检索式推理合成和双阶段奖励引导过滤三个阶段，蒸馏出高质量的结构化推理数据来微调 LLaMA3-8B 多智能体系统，在 XLLM@ACL2025 共享任务中获得第三名。
tags:
  - "ACL 2025 (XLLM Workshop)"
  - "模型压缩"
  - "结构化推理"
  - "多智能体框架"
  - "知识蒸馏"
  - "低资源学习"
  - "质量引导过滤"
---

# LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation

**会议**: ACL 2025 (XLLM Workshop)  
**arXiv**: [2504.16408](https://arxiv.org/abs/2504.16408)  
**代码**: [GitHub](https://github.com/JhCircle/Less-is-More)  
**领域**: 模型压缩 / LLM推理  
**关键词**: 结构化推理, 多智能体框架, 知识蒸馏, 低资源学习, 质量引导过滤

## 一句话总结

本文提出 Less is More 框架，在仅有 24 个标注样本的极端低资源条件下，通过逆向提示归纳、GPT-4o 增强的检索式推理合成和双阶段奖励引导过滤三个阶段，蒸馏出高质量的结构化推理数据来微调 LLaMA3-8B 多智能体系统，在 XLLM@ACL2025 共享任务中获得第三名。

## 研究背景与动机

**领域现状**：结构化推理任务——例如将问题分解为逻辑约束、验证推理链的每一步——要求 LLM 生成可解释的、逐步的推理过程。XLLM@ACL2025 共享任务-III 正瞄准这一挑战，要求参赛者仅从 24 个标注样本中学习，涵盖四个子任务：问题解析（QP）、CoT 解析（CP）、CoT 陈述（CS）和验证（CV）。

**现有痛点**：(1) 标注数据极度稀缺，无法直接微调高容量模型；(2) 需要在多个推理模块之间保持步骤级一致性和逻辑连贯性。已有的 CoT 提示方法通常依赖大规模指令微调或启发式提示，在标注稀缺且需要结构粒度的场景下表现不佳。

**核心矛盾**：数据质量与数量之间的张力——低资源环境下不可能获得大量高质量标注，但模型需要足够多且足够好的训练信号来学习结构化推理。

**本文目标**：从最少的种子数据出发，通过可控的数据蒸馏管道生成高质量训练数据，使小模型也能完成结构化推理。

**切入角度**：作者假设"少而精"的数据比"多而杂"的数据更有效——通过质量引导的过滤，可以从大量合成数据中筛选出真正有用的训练信号。

**核心 idea**：用逆向思维自动归纳任务提示，通过 GPT-4o 合成大量推理数据，再用双阶段过滤（结构+奖励）确保数据质量，最终数据质量而非数量驱动模型性能的提升。

## 方法详解

### 整体框架

输入为自然语言逻辑推理问题（来自 LogiQA 数据集），输出为结构化的推理过程，包括问题解析、CoT 分解和逐步验证。整个管道分为训练阶段（提示归纳→数据合成→质量过滤→模型微调）和推理阶段（三个专用智能体级联处理）。

### 关键设计

1. **逆向提示归纳（Reverse-Prompt Induction）**:

    - 功能：从少量种子样本中自动推导出最优的任务特定提示
    - 核心思路：受"逆向思维"（RoT）启发，给定种子数据 $\{(x_i, y_i)\}$，通过逆向提示指令 $\mathcal{P}_{\text{reverse}}$ 让 LLM 推断"什么样的指令能产生这些输出"。生成候选提示集 $\Pi$，再根据生成分数 $S_{\text{gen}}$ 和偏好分数 $S_{\text{pref}}$ 联合选取最优提示 $\pi_t^* = \arg\max_{\pi}[S_{\text{gen}}(\pi) + S_{\text{pref}}(\pi)]$
    - 设计动机：避免人工设计提示的主观性和低效性，自动化地从种子数据中挖掘最优指令模板

2. **检索增强推理合成（Retrieval-Augmented Reasoning Synthesis）**:

    - 功能：利用 GPT-4o 为未标注的 LogiQA 数据生成结构化推理标注
    - 核心思路：对每个未标注问题 $x$，用预训练编码器计算嵌入 $\mathbf{h}_x$，从种子集中检索 $k$ 个语义最近邻作为 few-shot 示范。分别构造 QP 和 UCoT 两个提示，调用 GPT-4o 生成结构化的 JSON 输出，包含 CoT 步骤、文本证据和验证标签
    - 设计动机：利用强大的闭源模型（GPT-4o）生成监督信号，检索最相似的示范确保上下文一致性

3. **双阶段奖励过滤（Dual-Stage Reward-Based Filtering）**:

    - 功能：从合成数据中筛选出高质量样本用于下游微调
    - 核心思路：第一阶段为结构过滤，移除格式不合规的输出（如 JSON 解析失败、推理步骤不足两步）。第二阶段使用 LLaMA3-based 奖励模型对每条数据进行打分，分别在 few-shot 和 zero-shot 两种提示下评估：$s_{\text{avg}} = \frac{1}{2}(s_{\text{few}} + s_{\text{zero}})$，只保留 $\mathcal{S}(x) > 0$ 的样本。双提示打分策略兼顾了上下文连贯性和通用质量
    - 设计动机：合成数据不可避免包含噪声，需要严格筛选才能保证微调效果

### 损失函数 / 训练策略

三个子任务模型分别从 Meta-Llama-3-8B-Instruct 独立微调，使用 LoRA+（rank=16, $\alpha=32$, lorap_lr_ratio=16）。通过 ms-swift 框架训练 5 个 epoch，学习率 $2 \times 10^{-5}$，batch size 4，梯度累积 4 步，warmup ratio 0.03，使用两张 NVIDIA A100-80G GPU。

## 实验关键数据

### 主实验

| 过滤策略 | Ques._F1 | Stmt._F1 | Evid._F1 | Reason._F1 |
|----------|----------|----------|----------|------------|
| 仅结构过滤 | 56.87 | 36.72 | 10.80 | 5.20 |
| Zero-shot 奖励过滤 | 62.76 | 38.05 | 12.79 | 7.15 |
| Few-shot 奖励过滤 | 65.89 | 38.26 | 14.45 | 7.70 |
| 平均奖励过滤 | **66.71** | **39.21** | **14.92** | **8.98** |

### 消融实验（训练数据量）

| 配置 | QP数量 | CP数量 | CV数量 | 说明 |
|------|--------|--------|--------|------|
| 原始 LogiQA | 7,376 | - | - | 未过滤 |
| 结构过滤 | 1,940 | 1,940 | 13,818 | 格式合规 |
| Zero-shot 过滤 | 1,309 | 1,309 | 9,434 | 质量筛选 |
| Few-shot 过滤 | 1,377 | 1,377 | 9,858 | 质量筛选 |
| 平均过滤 | 1,346 | 1,346 | 9,688 | 最佳平衡 |

### 关键发现

- 平均奖励过滤策略一致性地在所有指标上取得最佳，Reasoning F1 从 5.20 提升到 8.98（+72.7%）
- 即使 Question F1 不直接参与奖励计算，也从 56.87 提升到 66.71——这说明高质量的中间监督信号能间接提升模型的全局结构理解
- 数据量从 7,376 减少到 ~1,346 后性能反而更好，直接证明了"Less is More"的核心论点
- Few-shot 过滤略优于 Zero-shot 过滤，说明上下文信息有助于更准确的质量评估

## 亮点与洞察

- **"数据质量驱动"的实证证据**：仅用约 18% 的过滤数据就显著超越全量结构过滤数据，这对低资源场景下的 LLM 微调具有重要指导意义
- **逆向提示归纳**是一个优雅的冷启动策略——从目标输出反推提示，避免了人工提示工程的负担。这个技巧可以迁移到任何需要从少量示范中学习的任务
- **意外的跨模块迁移**：奖励信号仅应用于 CoT 模块，但 QP 模块也获得显著提升，暗示了结构化推理子任务之间的内在关联性

## 局限与展望

- 任务第三名而非第一，说明方法仍有改进空间，尤其在步骤级验证（Reason_F1 仅 8.98）方面
- 依赖 GPT-4o 进行数据合成，成本不低且存在闭源依赖
- 仅在 LogiQA 数据集上评估，泛化到其他结构化推理任务（如数学证明、代码调试）有待验证
- 奖励模型的选择对过滤质量影响很大，但论文未深入探讨不同奖励模型的对比
- 未来可探索迭代蒸馏（用过滤后的数据训练的模型再次生成合成数据）来进一步提升质量

## 相关工作与启发

- **vs LIMA (Less is More for Alignment)**: 两者共享"数据质量>数量"的理念，但本文针对结构化推理而非通用对齐，且引入了自动化的质量过滤机制
- **vs 标准 CoT Prompting**: 标准 CoT 依赖大量人工设计，本文通过逆向归纳+RA-ICL 实现全自动的结构化 CoT 生成
- **vs Knowledge Distillation 方法**: 传统蒸馏关注 soft label，本文蒸馏 explicit reasoning traces 并用质量过滤保证纯度

## 评分

- 新颖性: ⭐⭐⭐⭐ 各组件（逆向提示、RA-ICL、奖励过滤）均非全新，但组合方式在低资源结构化推理场景下有效
- 实验充分度: ⭐⭐⭐⭐ 消融充分展示了各过滤策略的效果，但只在单一数据集上评测
- 写作质量: ⭐⭐⭐⭐⭐ 方法描述清晰，框架图直观，公式表达规范
- 价值: ⭐⭐⭐⭐ 作为共享任务方案有参考价值，"数据质量>数量"的结论对社区有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Less is More but Where: Dynamic Token Compression via LLM-Guided Keyframe Prior](../../NeurIPS2025/model_compression/less_is_more_but_where_dynamic_token_compression_via_llm-guided_keyframe_prior.md)
- [\[ICML 2026\] Critique-Guided Distillation for Robust Reasoning via Refinement](../../ICML2026/model_compression/critique-guided_distillation_for_robust_reasoning_via_refinement.md)
- [\[CVPR 2025\] Less is More: Efficient Model Merging with Binary Task Switch](../../CVPR2025/model_compression/less_is_more_efficient_model_merging_with_binary_task_switch.md)
- [\[ICML 2026\] Beyond Tokens: Enhancing RTL Quality Estimation via Structural Graph Learning](../../ICML2026/model_compression/beyond_tokens_enhancing_rtl_quality_estimation_via_structural_graph_learning.md)
- [\[ACL 2025\] MoRE: A Mixture of Low-Rank Experts for Adaptive Multi-Task Learning](more_a_mixture_of_low-rank_experts_for_adaptive_multi-task_learning.md)

</div>

<!-- RELATED:END -->
