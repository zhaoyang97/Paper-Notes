---
title: >-
  [论文解读] From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs
description: >-
  [ACL 2025][LLM安全][misleading queries] 提出三阶段微调方法（误导检测-查询纠正-准确回答）增强 LLM 处理含误导信息输入的能力，在误导检测和 QA 任务上显著提升准确率，同时减少幻觉生成。 领域现状 领域现状：LLM 对输入质量高度敏感，当查询包含不准确或误导性信息时容易产生幻觉…
tags:
  - "ACL 2025"
  - "LLM安全"
  - "misleading queries"
  - "幻觉"
  - "微调"
  - "query correction"
  - "鲁棒性"
---

# From Misleading Queries to Accurate Answers: A Three-Stage Fine-Tuning Method for LLMs

**会议**: ACL 2025  
**arXiv**: [2504.11277](https://arxiv.org/abs/2504.11277)  
**代码**: [https://github.com/cong03/FMQAA](https://github.com/cong03/FMQAA)  
**领域**: LLM安全  
**关键词**: misleading queries, 幻觉, 微调, query correction, 鲁棒性

## 一句话总结
提出三阶段微调方法（误导检测->查询纠正->准确回答）增强 LLM 处理含误导信息输入的能力，在误导检测和 QA 任务上显著提升准确率，同时减少幻觉生成。

## 研究背景与动机

### 领域现状

**领域现状**：LLM 对输入质量高度敏感，当查询包含不准确或误导性信息时容易产生幻觉。

**现有痛点**：现有方法（RAG、self-correction）聚焦纠正输出，忽略了纠正输入本身的潜力。RAG 需要外部知识库导致延迟高，self-correction 依赖模型自评估能力。

**核心矛盾**：如何让 LLM 主动识别并纠正输入中的误导信息，而非被动接受导致错误输出？

**本文目标** 训练 LLM 依次完成：检测误导->纠正查询->基于纠正后查询回答。

**切入角度**：将"处理误导输入"分解为三个可独立训练的子任务。

**核心 idea**：教会 LLM"先检查问题再回答"——通过三阶段微调让模型主动清洁输入。

## 方法详解

### 整体框架
用 Qwen2.5-72B 生成含误导信息的查询（质量过滤：编辑距离相似度 >0.8 且错误率 >0.5）-> 三阶段微调：(1) 检测查询是否含误导 (2) 纠正误导查询 (3) 基于纠正后查询生成答案。

### 关键设计

1. **误导数据构建**

    - 用大模型为每个原始查询生成 3 个变体，保持高相似度但引入误导
    - 双重过滤：表面相似度 $S_{sim}$ > 0.8 + 答案错误率 $E_{error}$ > 0.5
    - 设计动机：确保误导查询与原始查询足够相似（不易察觉）且确实能欺骗模型

2. **三阶段训练**

    - Stage 1 — 二分类检测（YES/NO），交叉熵损失
    - Stage 2 — 生成纠正后的查询，可结合内置知识或外部知识
    - Stage 3 — 基于纠正查询生成准确答案
    - 设计动机：分解为可独立优化的子任务，每阶段聚焦一种能力

## 实验关键数据

### 主实验 -- 误导查询下的 QA 准确率

| 方法 | HaluEval-QAmis | CQAmis | 提升 |
|------|---------------|--------|------|
| 原始 LLM | ~45% | ~40% | 基线 |
| RAG 增强 | ~55% | ~50% | +10% |
| Self-correction | ~50% | ~48% | +7% |
| **三阶段方法** | **~72%** | **~68%** | **+25%** |

### 消融实验

| 配置 | 准确率 | 说明 |
|------|--------|------|
| 仅 Stage 1 | +5% | 检测有帮助但不够 |
| Stage 1+2 | +15% | 纠正进一步提升 |
| **Stage 1+2+3** | **+25%** | 完整流水线最优 |

### 关键发现
- **三阶段方法在误导输入上远超基线**（+25%），且对正常输入也有改善
- **标准数据集中也存在误导查询**：模型发现并去除后，原始模型性能也提升
- **兼顾检测和回答**：幻觉检测能力也同步提升
- **对输入的鲁棒性显著增强**：无论是否有误导，性能稳定

## 亮点与洞察
- **"先纠正输入再回答"**是一个简单但有效的范式——类似人类在回答问题前先质疑问题本身
- **误导数据构建方法**可迁移到其他鲁棒性研究

## 局限与展望
- 误导数据由 LLM 生成，可能不完全反映真实用户误导
- 三阶段串行增加推理延迟
- 改进方向：端到端训练、与 RAG 结合

## 相关工作与启发
- **vs RAG**：RAG 纠正输出，本文纠正输入——可以互补
- **vs Self-correction (Madaan et al.)**：Self-correction 依赖模型自评估，本文通过专门训练增强检测能力

## 评分
- 新颖性: ⭐⭐⭐⭐ "先纠正输入再回答"的三阶段设计有创新
- 实验充分度: ⭐⭐⭐⭐ 多数据集 + 消融 + 误导/正常双场景
- 写作质量: ⭐⭐⭐⭐ 方法清晰
- 价值: ⭐⭐⭐⭐ 对 LLM 鲁棒性有实际应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach](towards_context-robust_llms_a_gated_representation_fine-tuning_approach.md)
- [\[ICML 2025\] TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs](../../ICML2025/llm_safety/tuco_measuring_the_contribution_of_fine-tuning_to_individual_responses_of_llms.md)
- [\[ICLR 2026\] Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!](../../ICLR2026/llm_safety/be_careful_when_fine-tuning_on_open-source_llms_your_fine-tuning_data_could_be_s.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](../../NeurIPS2025/llm_safety/adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[ACL 2025\] Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)

</div>

<!-- RELATED:END -->
