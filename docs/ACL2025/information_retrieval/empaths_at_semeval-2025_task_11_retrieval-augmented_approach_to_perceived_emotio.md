---
title: >-
  [论文解读] Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction
description: >-
  [ACL 2025 (SemEval Workshop)][信息检索/RAG][情感检测] 提出 EmoRAG 系统，用检索增强生成（RAG）管道结合多 LLM 集成聚合，在 SemEval-2025 Task 11 多标签情感检测任务上无需额外训练即在 28 种语言中取得有竞争力的结果，平均 F1-micro 0.638。
tags:
  - "ACL 2025 (SemEval Workshop)"
  - "信息检索/RAG"
  - "情感检测"
  - "RAG"
  - "多语言"
  - "LLM集成"
  - "多标签分类"
---

# Empaths at SemEval-2025 Task 11: Retrieval-Augmented Approach to Perceived Emotions Prediction

**会议**: ACL 2025 (SemEval Workshop)  
**arXiv**: [2506.04409](https://arxiv.org/abs/2506.04409)  
**代码**: 无  
**领域**: 信息检索  
**关键词**: 情感检测, RAG, 多语言, LLM集成, 多标签分类

## 一句话总结

提出 EmoRAG 系统，用检索增强生成（RAG）管道结合多 LLM 集成聚合，在 SemEval-2025 Task 11 多标签情感检测任务上无需额外训练即在 28 种语言中取得有竞争力的结果，平均 F1-micro 0.638。

## 研究背景与动机

SemEval-2025 Task 11 聚焦**感知情感检测**（perceived emotion detection）：判断大多数读者会从给定文本中推断说话者正在经历哪些情感（joy, sadness, fear, anger, surprise, disgust, neutral）。这不是分析读者被唤起的情感，也不是推断说话者的真实情感，而是关注情感的**社会共识性理解**。

**核心挑战**：

**多语言覆盖**：涵盖 28 种语言，包括大量低资源语言（Hausa、Kinyarwanda、Emakhuwa、isiZulu 等）

**多标签任务**：每段文本可能同时包含多种情感

**文化差异**：不同文化背景下的情感表达和解读存在显著差异

**传统方法**（微调预训练 Transformer + 线性分类头）在单语场景表现好，但跨语言泛化面临语言变异性和文化差异的额外挑战。

**EmoRAG 的核心思路**：利用 RAG 不需要额外训练的特性，以标注训练数据作为检索语料库，让模型在推理时参考相关情感实例，提高跨语言和跨文化的鲁棒性。

## 方法详解

### 整体框架

EmoRAG 由四个组件串联：Database → Retriever → Generators (LLM集合) → Aggregation Model。

### 关键设计

1. **数据库（Database）**：

    - 直接使用标注训练数据构建
    - 基于 BRIGHTER 数据集（28 种语言的多标签情感标注数据）和 EthioEmo 数据集（4 种埃塞俄比亚语言）

2. **检索器（Retriever）**：

    - 两种检索器对比：
        - **n-gram 检索器**（来自 LangChain 模块）：假设对低资源语言更好，因为依赖表面文本特征
        - **BGE-M3 句子嵌入检索器**：多语言嵌入模型
    - K 值设定：低资源语言 K=30（token 消耗大），高资源语言 K=100
    - 检索得到的样本作为 LLM 的 few-shot prompt

3. **生成器（Decoder Models）**：

    - 使用四个 LLM 集成：
        - Llama-3.1-70B
        - Qwen2.5-72B-Instruct
        - gpt-4o-mini
        - gemma-2-27b-it
    - 系统提示均为英语（实验发现英语提示效果优于目标语言提示）
    - 每个模型独立输出情感预测

4. **聚合策略（Aggregation）**：

    - **单模型**：直接使用某一模型的输出
    - **多数投票**：每个标签的预测取所有模型的多数投票
    - **Macro/Micro 加权投票**：按模型在 dev 集上的 F1 分数加权
    - **Label-F1 加权投票**：对每个标签独立加权，权重基于该标签在各模型上的 F1 分数
    - **GPT-4o 聚合**：将所有模型结果 + few-shot 样例输入 gpt-4o-mini 做聚合

### 训练策略

- **无需额外训练**：不做任何模型微调，纯推理时检索+生成
- 仅在 dev 集上确定最佳聚合策略和检索器配置

## 实验关键数据

### 主实验 — 代表性语言测试集表现（表格）

| 语言 | 最佳模型 | Dev F1-micro | Dev F1-macro | Test F1-micro | Test F1-macro |
|------|----------|-------------|-------------|--------------|--------------|
| English | L-F1 Vote | 0.821 | 0.818 | 0.807 | 0.789 |
| Spanish | L-F1 Vote | 0.813 | 0.809 | 0.820 | 0.817 |
| Russian | L-F1 Vote | 0.880 | 0.880 | 0.883 | 0.879 |
| Hindi | L-F1 Vote | 0.842 | 0.849 | 0.866 | 0.866 |
| Marathi | L-F1 Vote | 0.943 | 0.947 | 0.856 | 0.864 |
| Hausa | L-F1 Vote | 0.735 | 0.731 | 0.704 | 0.695 |
| Swahili | L-F1 Vote | 0.440 | 0.409 | 0.430 | 0.386 |
| Emakhuwa | gpt-4o-mini | 0.300 | 0.211 | 0.256 | 0.216 |

### 各模型 Dev 集平均表现（表格）

| 模型/策略 | F1-micro | F1-macro |
|-----------|----------|----------|
| llama-3.1-70b | 0.563 | 0.515 |
| qwen2.5-70b | 0.590 | 0.556 |
| gpt-4o-mini | 0.631 | 0.590 |
| gpt-4o-mini + n-gram | 0.641 | 0.601 |
| gemma-2-27b | 0.617 | 0.576 |
| majority_vote | 0.661 | 0.617 |
| **majority_vote_by_label_f1** | **0.678** | **0.634** |

### 关键发现

- **Label-F1 加权投票**在绝大多数语言上是最优聚合策略（21/28 种语言的最佳选择）
- 高资源语言表现好（Russian 0.883, Hindi 0.866），低资源语言差异大（Emakhuwa 仅 0.256, Tigrinya 0.260）
- n-gram 检索器对部分低资源语言更有效（Oromo、Sundanese、Mandarin、Kinyarwanda 选择了 n-gram 配置）
- gpt-4o-mini 在单模型中表现最佳（F1-micro 0.631），但集成聚合（0.678）显著优于任何单模型
- 英语系统提示效果优于目标语言提示，可能因为 LLM 的英语指令跟随能力更强
- 部分语言 dev/test 性能差异大（German: dev 0.745 → test 0.269；Brazilian Portuguese: dev 0.766 → test 0.481），暗示分布偏移问题

## 亮点与洞察

- **零训练范式**：RAG + LLM 集成完全免去微调，对资源受限场景有吸引力
- **Label-F1 加权策略**比简单多数投票或全局加权更精细：不同情感标签的最优模型不同
- n-gram vs 嵌入检索器的选择揭示了低资源 vs 高资源语言的不同特征：低资源语言的嵌入质量差，表面特征匹配可能更可靠
- 多 LLM 集成的互补性强：单个模型在不同语言上各有所长

## 局限与展望

- 仅覆盖 6 种基本情感 + neutral，未测试更细粒度的情感分类
- 对高度不均衡的类别分布和显著的分布偏移（如 German、Portuguese）处理不足
- K 值（30 vs 100）的设定较粗糙，未做系统的超参数搜索
- 推理成本高（4 个大模型 + 检索），对于低延迟场景不适用
- 未与微调基线直接对比，难以判断 RAG 方式是否真的优于微调

## 相关工作与启发

- 传统方法（微调 Transformer + 分类头）在多语言情感任务上的局限性是本文的出发点
- RAG 在知识密集型 NLP 任务中的成功扩展到了情感分类领域
- BRIGHTER 数据集首次覆盖 28 种语言的情感标注，为多语言情感研究提供基础
- 与跨语言情绪检测的 LLM 方法形成呼应，但本文更强调实用性和可扩展性

## 评分

- **新颖性**: 5/10 — RAG + LLM 集成是成熟范式的组合应用，缺乏技术创新
- **实验充分度**: 7/10 — 覆盖 28 种语言很全面，但缺乏与微调基线的对比
- **写作质量**: 6/10 — 内容组织清晰但篇幅短，分析不够深入
- **价值**: 6/10 — 作为 SemEval 系统描述论文，展示了 RAG 在多语言情感分析中的可行性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Health-LLM: Personalized Retrieval-Augmented Disease Prediction System](health-llm_personalized_retrieval-augmented_disease_prediction_system.md)
- [\[NeurIPS 2025\] RMIT-ADM+S at the MMU-RAG NeurIPS 2025 Competition](../../NeurIPS2025/information_retrieval/rmit-adms_at_the_mmu-rag_neurips_2025_competition.md)
- [\[ACL 2025\] MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation](memerag_a_multilingual_end-to-end_meta-evaluation_benchmark_for_retrieval_augmen.md)
- [\[ACL 2025\] Controllable and Reliable Knowledge-Intensive Task-Oriented Conversational Agents with Declarative Genie Worksheets](genie_worksheets_tod_agent.md)
- [\[NeurIPS 2025\] Learning Task-Agnostic Representations through Multi-Teacher Distillation](../../NeurIPS2025/information_retrieval/learning_task-agnostic_representations_through_multi-teacher_distillation.md)

</div>

<!-- RELATED:END -->
