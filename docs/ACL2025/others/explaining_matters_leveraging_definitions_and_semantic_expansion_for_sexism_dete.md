---
title: >-
  [论文解读] Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection
description: >-
  [ACL 2025][性别歧视检测] 针对在线性别歧视检测中的数据稀疏和细粒度分类歧义问题，提出两种基于prompt的数据增强技术——定义驱动数据增强（DDA）利用类别定义生成语义对齐的合成样本，上下文语义扩展（CSE）通过分析模型错误的语义特征丰富训练数据——并结合 Mistral-7B 回退集成策略，在 EDOS 数据集上实现全任务 SOTA。
tags:
  - "ACL 2025"
  - "性别歧视检测"
  - "数据增强"
  - "定义驱动"
  - "语义扩展"
  - "集成学习"
---

# Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection

**会议**: ACL 2025  
**arXiv**: [2506.06238](https://arxiv.org/abs/2506.06238)  
**代码**: 有 ([https://github.com/Sahrish42/explaining_matters_sexism_detection_acl2025](https://github.com/Sahrish42/explaining_matters_sexism_detection_acl2025))  
**领域**: 其他  
**关键词**: 性别歧视检测, 数据增强, 定义驱动, 语义扩展, 集成学习

## 一句话总结

针对在线性别歧视检测中的数据稀疏和细粒度分类歧义问题，提出两种基于prompt的数据增强技术——定义驱动数据增强（DDA）利用类别定义生成语义对齐的合成样本，上下文语义扩展（CSE）通过分析模型错误的语义特征丰富训练数据——并结合 Mistral-7B 回退集成策略，在 EDOS 数据集上实现全任务 SOTA。

## 研究背景与动机

在线性别歧视语言严重影响女性和边缘群体，自动检测系统面临两大核心挑战：

**挑战一：数据稀疏**  
即使在最大最精心策划的 EDOS 数据集中，类别极度不平衡。"Threats of harm"仅56个样本（1.1%），"Supporting mistreatment of individual women"仅75个（1.3%）。这种稀疏性严重阻碍了模型在低资源类别上的泛化。

**挑战二：细粒度分类歧义**  
性别歧视语言本质上微妙且边界模糊，连人类标注者也难以达成一致。论文对 EDOS 测试集的分析揭示了惊人的标注不一致：
- "Descriptive attacks": 54.1% 部分分歧
- "Backhanded gendered compliments": 83.3% 完全不一致
- "Threats of harm": 0% 完全一致（没有一组标注者完全同意！）

这些不一致不仅反映了任务难度，更在训练时引入了矛盾信号，损害模型性能。

## 方法详解

### 整体框架

流水线包含四个阶段：
1. **预训练**：在 EDOS 的 200 万未标注数据上做 MLM 预训练
2. **数据增强**：DDA 或 CSE
3. **微调**：在增强数据上做监督微调
4. **集成**：Mistral-7B 回退集成（M7-FE）

### 关键设计

1. **定义驱动数据增强（Definition-based Data Augmentation, DDA）**：

   核心思想：将类别定义显式注入数据增强的 prompt 中，生成语义对齐的合成样本。

   对每个训练样本 (x_i, y_i)，DDA 的 prompt 包含三部分：
    - 明确的生成指令：保持原始性别歧视意图
    - 风格引导：模拟 Reddit/Gab 的非正式社交媒体语言
    - **类别定义 φ(y_i)**：从 EDOS 分类体系中提取的语义定义（如"2.1 Descriptive Attacks"的定义明确了什么算描述性攻击而非情感攻击）

   为什么定义很重要？传统增强（如 EDA、回译）只关注语言多样性，无法理清相邻类别的语义边界。DDA 通过显式定义帮助生成器"理解"每个类别的核心特征，减少类别间的语义重叠。

   DDA 仅针对最不平衡的 c=5 个类别生成合成数据。

2. **上下文语义扩展（Contextual Semantic Expansion, CSE）**：

   核心思想：不是生成更多样本，而是为模型错分的样本生成**语义解释**，拼接在原文本后作为增强上下文。

   CSE 的工作流程：
    - 用基线 DeBERTa 在训练数据上做预测，找出所有误分类样本
    - 发现模型以高置信度（p > 0.9）做出错误预测——说明是系统性偏差而非随机错误
    - 对每个误分类样本，用结构化prompt让LLM做6步语义分析：
      1. 分析语言模式和风格特征
      2. 检查中性/贬义语言
      3. 评估与性别相关的情感偏见
      4. 考虑情境上下文
      5. 识别刻板印象和潜在偏见
      6. 评估文本意图

   输出的语义扩展拼接在原文本后：[x; e(x)]，作为增强训练数据。

   CSE 处理了 2,518 个被错分为非性别歧视的性别歧视样本 + 2,328 个被错分为性别歧视的非性别歧视样本。

3. **Mistral-7B 回退集成（M7-FE）**：

   结合 DeBERTa-v3-Large、Mistral-7B、DTFN 三个模型的预测：
    - 常规情况：多数投票决定最终预测
    - 平局（二选一）：由 Mistral-7B 作为回退模型裁决
    - 完全分歧（三方各不同）：使用 Mistral-7B 的预测

   为什么用 Mistral-7B 做回退？预实验表明它在模糊性别歧视案例上更鲁棒。

   设计动机类比人类标注流程：当标注者意见不一致时，需要引入第三方裁判。不同模型在不同数据和目标上训练，自然提供"多元视角"。

### 损失函数 / 训练策略

- 预训练：MLM（15% token masking），10 epochs
- 微调：标准交叉熵损失，DeBERTa/RoBERTa 训练30 epochs，Mistral-7B 训练10 epochs
- 硬件：4×A100 GPU

## 实验关键数据

### 主实验：EDOS 数据集

| 方法 | Task A (二分类) | Task B (4类) | Task C (11类) |
|------|----------------|-------------|--------------|
| DeBERTa-v3-large (baseline) | 0.8479 | 0.6875 | 0.5088 |
| SemEval-2023 第一名 | 0.8746 | 0.7326 | 0.5606 |
| SEFM (增强baseline) | 0.8538 | 0.6619 | 0.4641 |
| M7-FE (仅集成) | 0.8603 | 0.7027 | 0.5213 |
| + Baseline Prompt | 0.8783 | 0.7049 | 0.5601 |
| **+ DDA** | 0.8769 | **0.7277** | **0.6018** |
| **+ CSE** | **0.8819** | 0.7243 | 0.5639 |

### 消融：DDA vs 无定义prompt

DDA 在 Task C 上的改善最为显著：从 0.5601 提升至 0.6018（+4.17 F1），远超所有 SemEval 2023 参赛系统。

论文通过差异混淆矩阵详细分析了 DDA 的改善来源：
- "2.3 Dehumanising attacks" 正确预测增加 42 例
- "3.4 Condescending explanations" 正确预测增加 8 例
- "3.1 Casual slurs" 和 "3.2 Gender stereotypes" 之间的互混从 48 降至 35（降低 ~27%）

### 关键发现

1. **CSE 在二分类上效果最好**（Task A: 0.8819），因为二分类的决策边界简单，更适合纠正系统性偏差
2. **DDA 在细粒度分类上效果最好**（Task C: 0.6018，比上一SOTA +4.1 F1），因为类别定义直接帮助区分相邻细粒度类别
3. **传统增强方法（SEFM, HULAT/EDA）在细粒度任务上反而降低性能**——缺乏语义约束的增强只会引入更多噪声
4. **标注者不一致率与模型困惑高度相关**：0% Full Agreement 的类别正是模型最容易错分的类别
5. 模型在错分时置信度高达 p > 0.9——表明这是系统性bias而非不确定性，传统的置信度校准/self-correction无法解决

## 亮点与洞察

- **从标注者不一致出发分析问题**：不仅看数据不平衡，更深入分析了标注者分歧的分布——在某些类别上甚至0%完全一致，这是比样本数少更根本的困难
- **DDA 的核心洞察**：LLM 做数据增强时，给定类别定义比仅给指令有质的提升。定义充当了"语义锚"，约束生成器在正确的语义边界内生成
- **CSE 的"内省"式纠错**：不是简单的self-training或置信度过滤，而是让LLM解释为什么一个样本可能被错分——类似 Chain-of-Thought 但用于分类而非生成
- **工程上务实**：回退集成不追求方法新颖性，而是解决实际问题（多类别下投票平局），选择 Mistral-7B 做裁判基于实证而非理论

## 局限与展望

1. DDA 和 CSE 依赖 LLM（GPT-4o）做增强，可能引入预训练数据中的偏见
2. 仅在英语 EDOS 数据集上评估，多语言和低资源语言效果未知
3. M7-FE 采用简单的多数投票+回退，加权投票或置信度聚合可能更优
4. CSE 的语义扩展增加了输入长度，可能影响推理效率
5. DDA 的类别定义来自 EDOS 官方分类体系——其他数据集可能缺乏如此清晰的定义

## 相关工作与启发

- **SemEval-2023 Task 10**（Kirk et al., 2023）：建立了 EDOS 基准，最佳系统使用 DeBERTa 集成。本文在此基础上通过增强和集成策略大幅提升
- **EDA**（Wei & Zou, 2019）：经典文本增强方法，但对细粒度分类效果有限。DDA 通过注入定义实现了质的超越
- **Chain-of-Thought Prompting**：CSE 借鉴了 CoT 的结构化推理思路，但用于分类任务的语义扩展而非生成任务的推理链
- **启发**：定义驱动增强可推广到其他细粒度分类任务（如情感分析的细粒度、仇恨言论子类型），关键是有明确的类别语义定义

## 评分

- **新颖性**: ⭐⭐⭐⭐ — DDA（定义驱动增强）和 CSE（语义扩展纠错）都是新提出的技术，设计动机清晰。集成策略虽然简单但有效
- **实验充分度**: ⭐⭐⭐⭐ — EDOS 三个任务层次的全面评估，消融对比了有/无定义、不同增强方法，错误分析详尽
- **写作质量**: ⭐⭐⭐⭐ — 标注者不一致分析表格直观有力，pipeline图清晰，prompt设计展示完整
- **价值**: ⭐⭐⭐⭐ — Task C 的 +4.1 F1 提升是实质性进步，DDA 思路对其他细粒度NLP任务有广泛参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SEOE: A Scalable and Reliable Semantic Evaluation Framework for Open Domain Event Detection](seoe_semantic_eval.md)
- [\[ACL 2025\] S3 - Semantic Signal Separation](s3_-_semantic_signal_separation.md)
- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ACL 2025\] Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking](inner_thinking_transformer_leveraging_dynamic_depth_scaling_to_foster_adaptive_i.md)
- [\[ACL 2025\] CADReview: Automatically Reviewing CAD Programs with Error Detection and Correction](cadreview_automatically_reviewing_cad_programs_with_error_detection_and_correcti.md)

</div>

<!-- RELATED:END -->
