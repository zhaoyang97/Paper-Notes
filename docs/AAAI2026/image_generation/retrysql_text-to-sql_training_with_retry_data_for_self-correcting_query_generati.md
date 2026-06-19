---
title: >-
  [论文解读] RetrySQL: Text-to-SQL Training with Retry Data for Self-Correcting Query Generation
description: >-
  [AAAI 2026][图像生成][Text-to-SQL] 提出 RetrySQL 训练范式，通过在推理步骤中注入 retry data（错误步骤 + [BACK] 标记 + 正确步骤）来持续预训练小型编码模型，使 1.5B 参数的开源模型学会自纠正能力，在 BIRD 和 SPIDER 基准上分别提升整体执行准确率最高 4 和 3.93 个百分点，挑战性样例提升高达 9 个百分点。
tags:
  - "AAAI 2026"
  - "图像生成"
  - "Text-to-SQL"
  - "自纠正"
  - "retry data"
  - "推理链"
  - "小型语言模型"
---

# RetrySQL: Text-to-SQL Training with Retry Data for Self-Correcting Query Generation

**会议**: AAAI 2026  
**arXiv**: [2507.02529](https://arxiv.org/abs/2507.02529)  
**代码**: [https://github.com/allegro/RetrySQL](https://github.com/allegro/RetrySQL)  
**领域**: 图像生成  
**关键词**: Text-to-SQL, 自纠正, retry data, 推理链, 小型语言模型

## 一句话总结

提出 RetrySQL 训练范式，通过在推理步骤中注入 retry data（错误步骤 + [BACK] 标记 + 正确步骤）来持续预训练小型编码模型，使 1.5B 参数的开源模型学会自纠正能力，在 BIRD 和 SPIDER 基准上分别提升整体执行准确率最高 4 和 3.93 个百分点，挑战性样例提升高达 9 个百分点。

## 研究背景与动机

### 领域现状

Text-to-SQL 是 NLP 的核心挑战之一，旨在将自然语言问题转换为 SQL 查询。当前方法主要分为：
- **端到端管线**：结合检索（schema linking）、生成和纠正三个阶段
- **大模型方案**：GPT-4o、Gemini 等大模型在 BIRD 和 SPIDER 基准上仍显著落后于人类
- **微调方案**：对开源模型进行参数高效微调（LoRA 等），但主要集中在较大模型（7B+）

### 现有痛点

**小模型潜力未被挖掘**：1.5B 参数的编码模型（如 OpenCoder、Qwen2.5-Coder）在零样本下表现很差（EX_overall ~30-37%），但训练效率高、推理成本低，适合实际部署

**自纠正能力未被应用于 Text-to-SQL**：虽然 DeepSeek-R1 等已证明 RL 可以学习自纠正，更近期研究表明标准自回归训练配合 retry data 就能学到自纠正（数学推理任务），但这一方法**尚未被应用于 Text-to-SQL 领域**

**现有纠正都是后处理**：如 DIN-SQL 的纠正步骤在生成之后进行，而非在生成过程中自纠正

### 核心矛盾

小模型在 Text-to-SQL 任务中性能不足，但增大模型规模面临训练数据不足（BIRD 仅 ~9K 样例）和计算资源限制。如何在**不增加模型参数**的情况下显著提升小模型的生成能力？

### 切入角度

受 "Physics of Language Models" 系列工作启发，在数学推理中 retry data 能教模型自纠正。本文将这一范式迁移到 Text-to-SQL：构造包含错误+纠正的推理步骤数据，通过持续预训练让模型学会"犯错→识别→回溯→纠正"的能力。

## 方法详解

### 整体框架

RetrySQL 训练范式的三个阶段：

1. **推理步骤生成**（图 1a）：用 GPT-4o 为训练集中每个 SQL 查询生成一系列推理步骤（chain of reasoning）
2. **retry data 构造**（图 1b）：对推理步骤施加随机扰动（替换为错误步骤），用 [BACK] 标记标注错误并附上正确步骤
3. **持续预训练**（图 1c）：用含 retry data 的训练样例对开源编码模型进行持续预训练

### 关键设计

#### 1. **推理步骤生成**

**核心思路**：为每个训练样例的 SQL 查询用 GPT-4o 生成合成推理链，格式类似于解题步骤。

**具体实现**：
- 使用 DDL（数据定义语言）作为 schema 表示，因为它提供了简洁的表/列名+数据类型+关系信息
- 采用 perfect schema linking（仅包含查询涉及的表/列），聚焦于 SQL 生成能力的研究
- 对 100 个随机样例人工验证了推理步骤的正确性

**设计动机**：先前研究表明 retry data 仅在有推理步骤的情况下有效。推理步骤为模型提供了**可回溯的中间目标**，使"犯错→纠正"成为可能。

#### 2. **Retry Data 构造**

**核心思路**：对推理步骤序列 $(r_1, r_2, ..., r_N)$ 随机施加扰动。给定步骤 $r_i$，以概率 $p_{retry}$ 选择一个错误步骤 $r_{error}$ 替换它，然后用 [BACK] 标记标注并附上正确的 $r_i$。

四种扰动策略：
- **FS (Forward Single)**：从**后面的步骤**中随机选一个作为错误（一次）
- **FM (Forward Multiple)**：从后面的步骤中随机选多个作为连续错误
- **FBS (Forward-Back Single)**：从**所有其他步骤**中选一个作为错误
- **FBM (Forward-Back Multiple)**：从所有其他步骤中选多个作为错误

每个 $r_i$ 被替换为：$(r_{error}, \text{[BACK]}, r_i)$

**设计动机**：
- FS 策略最有效，因为在 SQL 生成中，错误通常来自**提前使用还未到的步骤**（"跳步"错误）
- 向后引用的错误（FBS/FBM）不符合 SQL 生成中的典型错误模式
- 多次扰动（FM/FBM）可能引入过多噪声

#### 3. **训练数据格式**

引入特殊 token：[CONTEXT]、[QUESTION]、[REASONING]、[SQL]，将数据库 schema、外部知识、推理步骤和 SQL 查询用这些 token 分隔：

```
[CONTEXT] DDL schema + external knowledge
[QUESTION] natural language question  
[REASONING] step1 [BACK] correct_step1 step2 step3 ...
[SQL] SELECT ...
```

推理时，模型输入截至 [REASONING] token，由模型自行生成推理步骤和 SQL。

#### 4. **线性探针验证**

**关键前提验证**：在训练前先验证基座模型是否**内在具有区分正确/错误推理步骤的能力**。

方法：冻结 OpenCoder 1.5B 权重，训练二分类头判断推理步骤的正确性。结果：
- balanced_accuracy = 82%
- f1_score = 71%

远高于随机猜测的 50%，证明模型确实具有隐藏的自纠正潜力，retry data 只是"解锁"了这一能力。

### 训练策略

- 使用 OpenCoder 1.5B 和 Qwen2.5-Coder 1.5B 作为基座模型
- 2× NVIDIA A100 80GB，DeepSpeed Zero-2
- 有效 batch size 128，5 个 epoch
- 学习率 5e-5，余弦调度
- AdamW 优化器（β₁=0.9, β₂=0.95）
- 每次训练约 4.47 GPU 小时

## 实验关键数据

### 主实验

OpenCoder 1.5B 在 BIRD 数据集上的执行准确率（EX）：

| 数据集变体 | EX_simple | EX_moderate | EX_challenging | EX_overall |
|-----------|-----------|-------------|---------------|------------|
| 零样本 | 47.14 | 27.63 | 17.52 | 38.44 |
| 无推理步骤 | 43.78 | 28.88 | 24.83 | 37.48 |
| 有推理步骤（无 retry） | 62.70 | 43.53 | 39.45 | 54.71 |
| **Retry FS 0.2** | **68.22** | **45.47** | 40.28 | **58.70** |
| **Retry FS 0.3** | 68.00 | 44.91 | **43.31** | 58.68 |

Qwen2.5-Coder 1.5B 在 SPIDER 数据集上：

| 数据集变体 | EX_easy | EX_medium | EX_hard | EX_extra | EX_overall |
|-----------|---------|-----------|---------|----------|------------|
| 无 retry | 92.34 | 74.75 | 65.52 | 48.80 | 73.25 |
| **Retry FS 0.2** | 90.40 | **80.94** | **69.66** | **55.18** | **77.18** |

### 消融实验

不同 retry 策略对比（OpenCoder 1.5B, BIRD）：

| 策略 | $p_{retry}$ | EX_overall | EX_challenging | 说明 |
|------|------------|------------|---------------|------|
| 无 retry | - | 54.71 | 39.45 | 基线 |
| **FS 0.2** | 0.2 | **58.70** | 40.28 | 最优整体 |
| **FS 0.3** | 0.3 | 58.68 | **43.31** | 最优挑战性 |
| FM 0.3 | 0.3 | 57.17 | 37.24 | 多次扰动不如单次 |
| FBS 0.4 | 0.4 | 57.63 | 35.86 | 向后引用效果差 |
| FBM 0.3 | 0.3 | 57.43 | 37.93 | 综合不如 FS |
| FM 0.5 | 0.5 | 48.63 | 28.28 | 过多扰动有害 |

### 端到端管线结果

| 模型 | EX_overall (BIRD) | 参数量 |
|------|------------------|-------|
| GPT-4o-mini | 32.53 | ~8B |
| **RetrySQL (OpenCoder 1.5B)** | **51.36** | 1.5B |
| GPT-4o | 54.99 | ~200B |

### 自纠正行为分析

模型在 [BACK] token 前后的置信度变化：

| 位置 | 平均最大 softmax 分数 | softmax 标准差 |
|------|-------------------|--------------|
| [BACK] 之前（错误 token） | 较低 | 较高（不确定） |
| [BACK] 之后（纠正 token） | 较高 | 较低（确定） |

这证明自纠正是**主动学习到的行为**，而非简单的数据增强效果。

### 关键发现

- **推理步骤本身就很重要**：加入推理步骤（无 retry）使 EX_overall 从 37.48 提升到 54.71（+17.2 p.p.）
- **FS 策略最优**：四种扰动策略中，Forward Single 一致最好
- **挑战性样例提升最大**：Retry FS 0.3 在 BIRD 挑战性样例上从 39.45 提升到 43.31（+3.86 p.p.），Qwen2.5-Coder 在 BIRD 上从 30.21 提升到 39.10（+8.89 p.p.）
- **$p_{retry}$ = 0.2-0.3 最优**：过低（0.1）或过高（0.5）的 retry 概率效果下降
- **RetrySQL 减少了推理步骤长度**：有 retry 的模型平均生成 9.47 步 vs 无 retry 的 12.48 步，因为自纠正避免了"继续错下去"
- **1.5B 模型可以挑战 GPT-4o**：在端到端管线中，RetrySQL 训练的 1.5B 模型达到 51.36 EX_overall，超过 GPT-4o-mini（32.53），接近 GPT-4o（54.99）

## 亮点与洞察

1. **自纠正是可学习的普遍规律**：从数学推理到 Text-to-SQL，retry data 的有效性证明了自纠正能力是语言模型的通用属性，不限于特定任务
2. **线性探针提供了先验验证**：在投入训练资源前就通过探针实验确认模型有潜力，这种"先验证再训练"的方法论值得推广
3. **置信度分析的说服力**：不仅展示了性能提升，还通过 softmax 分数的统计分析**证明**了自纠正行为的存在
4. **小模型的实际价值**：1.5B 模型 + RetrySQL 在实际管线中接近 GPT-4o，对成本敏感的生产环境有重要意义
5. **4.47 GPU 小时的训练成本**：极低成本即可获得显著提升

## 局限与展望

- 训练数据有限（BIRD 仅 ~9K 样例），未探索合成数据生成
- 未探索模型规模的扩展效果（仅 1.5B）
- 端到端管线的 schema linking 和纠正阶段较简单，未使用最先进的方案
- retry data 的最优 $p_{retry}$ 可能依赖于数据集特性
- 推理步骤由 GPT-4o 生成，存在对强 LLM 的依赖
- 未探索 retry data 与 RLHF/DPO 等对齐方法的结合

## 相关工作与启发

- **与 DeepSeek-R1 的关系**：R1 通过 RL 学习自纠正，RetrySQL 通过数据构造+标准预训练学习自纠正，后者更简单高效
- **与 s1（budget forcing）的关系**：s1 在 SFT 中强制思考预算来引导推理，RetrySQL 通过 retry data 自然学习回溯
- **Physics of LMs 的验证**：本文验证了 "self-correction as a universal law" 假说在新领域（Text-to-SQL）和新模型（Qwen2.5-Coder）上的成立
- **启发**：retry data 范式可能适用于更多需要推理步骤的任务，如代码生成、定理证明、规划等

## 评分

- 新颖性: ⭐⭐⭐⭐ （将 retry data 范式首次应用于 Text-to-SQL，验证了普遍性）
- 实验充分度: ⭐⭐⭐⭐⭐ （多模型、多基准、四种策略、多概率、置信度分析、管线评估）
- 写作质量: ⭐⭐⭐⭐⭐ （结构清晰、分析深入、可复现性强）
- 价值: ⭐⭐⭐⭐ （小模型+自纠正的范式对实际部署有启发，但需更多领域验证）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Targeted Data Protection for Diffusion Model by Matching Training Trajectory](targeted_data_protection_for_diffusion_model_by_matching_training_trajectory.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](difficulty_controlled_diffusion_model_for_synthesizing_effec.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[ICLR 2026\] Neon: Negative Extrapolation From Self-Training Improves Image Generation](../../ICLR2026/image_generation/neon_negative_extrapolation_image_generation.md)
- [\[AAAI 2026\] Infinite-Story: A Training-Free Consistent Text-to-Image Generation](infinite-story_a_training-free_consistent_text-to-image_gene.md)

</div>

<!-- RELATED:END -->
