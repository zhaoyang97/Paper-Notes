---
title: >-
  [论文解读] Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience
description: >-
  [ACL 2025 Findings][信息检索/RAG][结构化知识推理] 本文提出 CoRE 框架，通过蒙特卡洛树搜索（MCTS）构建包含成功和失败经验的记忆库，并在推理时通过对比式上下文学习（Contrastive ICL）检索正负样例来增强 LLM 对结构化数据（表格、数据库）的推理能力，在 Text-to-SQL 和 TableQA 上分别平均提升 3.44% 和 4.24%。
tags:
  - "ACL 2025 Findings"
  - "信息检索/RAG"
  - "结构化知识推理"
  - "对比检索增强生成"
  - "经验记忆"
  - "蒙特卡洛树搜索"
  - "Text-to-SQL"
---

# Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience

**会议**: ACL 2025 Findings  
**arXiv**: [2506.00842](https://arxiv.org/abs/2506.00842)  
**代码**: [GitHub](https://github.com/Kuvvius/CoRE)  
**领域**: 信息检索  
**关键词**: 结构化知识推理、对比检索增强生成、经验记忆、蒙特卡洛树搜索、Text-to-SQL

## 一句话总结

本文提出 CoRE 框架，通过蒙特卡洛树搜索（MCTS）构建包含成功和失败经验的记忆库，并在推理时通过对比式上下文学习（Contrastive ICL）检索正负样例来增强 LLM 对结构化数据（表格、数据库）的推理能力，在 Text-to-SQL 和 TableQA 上分别平均提升 3.44% 和 4.24%。

## 研究背景与动机

**领域现状**：LLM 在纯文本任务上表现出色，但在处理结构化数据（如数据库表、知识图谱）时表现显著退化。主流方法分为两类：(1) 基于 agent 的多步分解方法（MAC-SQL、DIN-SQL）使用固定的 few-shot 示例指导 SQL 生成；(2) 基于 RAG 的方法从外部知识库检索辅助信息。

**现有痛点**：(1) 已有 RAG 方法对结构化知识推理帮助有限，因为检索到的通用文档缺乏与目标结构化格式的对齐；(2) 基于 agent 的方法使用固定的 few-shot 示例，缺乏动态适应能力——固定示例可能与新问题相关性低；(3) 现有方法缺乏系统的正负对比信号——不知道什么是"对的路径"和"错的路径"，无法从错误中学习。

**核心矛盾**：LLM 的预训练数据中结构化数据占比极低，导致它难以理解表格中隐含的关系（如外键关联、列间约束）。而人类专家能通过"刻意练习"——接触大量成功和失败案例来构建结构化推理的心理模型，LLM 缺乏这一能力。

**本文目标**：(1) 构建丰富的推理经验记忆库（包含成功和失败轨迹）；(2) 设计动态检索和对比学习机制来利用这些经验；(3) 提供即插即用、无需训练的解决方案。

**切入角度**：受认知科学中"刻意练习"理论启发——人类专家通过反复接触多样化的成功和失败案例建立高效的心理表征，CoRE 模仿这一过程为 LLM 构建"经验记忆"。

**核心 idea**：用 MCTS 模拟人类的试错过程来自动生成大量成功/失败推理轨迹，建立经验记忆库；推理时检索相关的正负例，通过对比 ICL 让 LLM "学习成功经验、避免失败教训"。

## 方法详解

### 整体框架

CoRE 包含三个模块：(1) **Experience Memory Builder**——用 MCTS 探索推理路径，收集带奖励标签的经验轨迹；(2) **Retriever**——为新问题检索相关的正例（高奖励）和负例（低奖励）；(3) **Contrastive Thinker**——将正负例组织为对比提示，引导 LLM 生成答案。整个框架是 training-free 的，不修改 LLM 参数。

### 关键设计

1. **Experience Memory Builder（基于 MCTS 的经验记忆构建）**:

    - 功能：自动生成大量多样化的推理轨迹，包含中间步骤的奖励标签
    - 核心思路：将结构化推理问题分解为子问题序列 $\tau = \{(q_1, a_1), (q_2, a_2), ..., (q_n, a_n)\}$，用 MCTS 搜索最优轨迹。四个阶段：**Selection**——用 UCT 公式 $q_k^* = \arg\max [Q_{value} + w\sqrt{\ln N(s)/N(c)}]$ 平衡探索与利用；**Expansion**——用 LLM 生成 $d$ 个候选子问题；**Simulation**——前向模拟到终端状态，用混合奖励函数 $f_r = r_1^\alpha \cdot r_2^{1-\alpha}$ 评估（$r_1$ 是一致性奖励，$r_2$ 是自评估奖励）；**Back-propagation**——沿路径回传更新 Q 值。最终产生的经验记忆中每条记录 $(q, a, r)$ 包含问题、答案和奖励标签
    - 设计动机：相比直接用 LLM 生成 few-shot 示例，MCTS 能系统性地探索推理空间、产生多样化的成功和失败案例。原始训练集被扩展 8-9 倍，大幅增加了覆盖度

2. **Retriever（双排序检索器）**:

    - 功能：为新问题找到最相关的正例和负例
    - 核心思路：采用两阶段排序策略。首先用语义相似度 $\text{Sim}(Q_{current}, Q_{e_i})$ 检索 top-$k$ 条经验。然后用奖励标签进行二次排序：正例排序为 $\text{rank}_{sim}$ 与 $\text{rank}_{reward}$ 的线性组合（高奖励优先）；负例排序为 $\text{rank}_{sim}$ 与 $\text{rank}_{reward}$ 反向的组合（低奖励优先）。SQL 查询附带 AST 的自然语言描述以提高检索准确性
    - 设计动机：仅靠语义相似度检索的负例可能"不够相关"或"不够典型"，通过奖励标签的二次排序确保负例既与新问题相关又确实是错误案例

3. **Contrastive Thinker（对比式推理器）**:

    - 功能：利用检索到的正负例引导 LLM 生成正确答案
    - 核心思路：构建对比提示模板："参考成功经验，避免重蹈失败教训"。正例作为应学习的模式，负例（附带错误分析）作为应避免的模式。支持两种对比方式：(a) 单轮对比——在一个 prompt 中同时展示正负例；(b) 多轮对比——先给正例让模型生成初始答案，再给负例让模型修正。两种方式效果相近，多轮方式可适应 token 限制
    - 设计动机：仅给正例可能信息不够，仅给负例反而会"教坏"模型（LLM 会不自觉模仿错误案例）。对比式学习通过同时展示"对比对"让模型更准确理解正确推理模式

### 损失函数 / 训练策略

CoRE 是 training-free 框架，不涉及模型训练。MCTS 过程中的奖励函数是一致性奖励（回答频率）和自评估奖励的加权几何平均。

## 实验关键数据

### 主实验

Text-to-SQL（Bird 数据集，LlaMA-3-70b 或 GPT-4）：

| 方法 | EX (%) | +CoRE | 提升 |
|------|--------|-------|------|
| DIN-SQL | 30.5 | 34.0 | +3.5 |
| DAIL-SQL | 31.6 | 35.2 | +3.6 |
| MAC-SQL | 34.9 | 40.8 | +5.9 |
| MAC-SQL (GPT-4) | 46.6 | 51.6 | +5.0 |
| **平均提升** | — | — | **+3.44** |

TableQA（WikiTQ 和 FinQA，GPT-3.5）：

| 数据集 | 方法 | Accuracy | +CoRE | 提升 |
|--------|------|----------|-------|------|
| WikiTQ | StructGPT | 64.4 | 66.1 | +1.7 |
| WikiTQ | Dater | 58.4 | 63.5 | +5.1 |
| FinQA | StructGPT | 51.2 | 53.1 | +1.9 |
| FinQA | Dater | 52.4 | 59.0 | +6.6 |

### 消融实验

GPT-4 在 2-shot 设置下的对比方式分析（MAC-SQL + Bird）：

| 配置 | EX (%) | 说明 |
|------|--------|------|
| 固定示例 (baseline) | 56.40 | 原始固定 few-shot |
| 正例+正例 | 58.92 | 只用正面示例，动态检索 |
| 负例+负例 | 45.20 | 只用负面示例，性能下降 |
| 正例+负例（单轮） | 58.92 | 对比 ICL 单轮方式 |
| 正例→负例（多轮） | 58.08 | 对比 ICL 多轮方式 |
| 完整 CoRE | **58.92** | 对比 ICL 最优配置 |

### 关键发现

- **对比 ICL 显著优于仅用正例**：单纯用正例检索（+正+正）的提升约 2.52%，但加入负例对比后在困难问题上提升更大（高达 17.2%），说明"从错误中学习"对复杂推理至关重要
- **仅用负例会导致性能下降**：只给负面示例（负+负配置）反而让性能降至 45.2%，比不用 RAG 更差。这验证了 LLM 会不自觉模仿上下文中的错误案例的现象
- **MCTS 生成的经验记忆数据量大幅增加**：Bird 数据集的训练集从约 9400 条扩展到 85,956 条（8-9倍），WikiTQ 从 11,321 条扩展到 98,586 条，极大提高了检索覆盖率
- **困难问题上提升最显著**：在 Bird 的 challenging 子集上，CoRE 将 MAC-SQL 的 EX 从约 20% 提升到 40%（接近翻倍），而在 simple 子集上提升相对较小

## 亮点与洞察

- **"刻意练习"的计算化实现**：将认知科学中的学习理论迁移到 LLM 推理中，MCTS 扮演"练习"的角色、经验记忆扮演"心理模型"的角色，这个类比优雅且有效。同样的思路可迁移到代码生成、数学推理等其他需要系统性试错的场景
- **Training-free 的即插即用设计**：CoRE 不修改任何模型参数，仅通过改变 prompt 中的示例来提升性能，兼容性极强。一次性的 MCTS 投资可跨多个下游任务复用
- **"负例有害但对比有效"的发现**：只给负例会让模型变差，但负例配合正例形成对比反而带来最大提升。这个洞察对所有使用 ICL 的方法都有参考价值——上下文中放什么比放多少更重要

## 局限与展望

- **经验记忆需要针对新任务重建**：每换一个新的数据库或领域，都需要用 MCTS 重新生成经验记忆，这个离线成本虽然是一次性的但不可忽略（需要大量 LLM 调用）
- **高度依赖 grounding 阶段的质量**：CoRE 只改进推理阶段，如果前序的 schema 识别和表格筛选（grounding）出错，CoRE 无法弥补。实验中使用 golden schema 时性能大幅提升证实了这一点
- **与 Self-Consistency 策略不兼容**：CoRE + DAIL-SQL (SC) 的性能反而下降 0.2%，说明对比上下文的信息量与高温采样投票之间存在冲突
- 改进方向：可以探索在线更新经验记忆（随着新问题的产生持续积累经验），或者将 grounding 和 reasoning 阶段统一到对比框架中

## 相关工作与启发

- **vs DAIL-SQL**：DAIL-SQL 通过骨架相似度检索训练集中的 few-shot 示例，但仅用正例且来源有限。CoRE 的经验记忆通过 MCTS 扩展了 8-9 倍数据量，且引入了正负对比信号
- **vs ExpeL**：ExpeL 也用 LLM 从经验中学习，但依赖人工标注数据集来总结经验。CoRE 通过 MCTS 自动生成经验和奖励标签，更加自动化
- **vs Self-RAG / Corrective RAG**：这些工作关注"何时检索"和"检索质量评估"，而 CoRE 关注"如何利用检索结果"——通过对比机制提高检索内容的利用效率

## 评分

- 新颖性: ⭐⭐⭐⭐ MCTS构建经验记忆+对比ICL的组合思路新颖，认知科学类比优雅
- 实验充分度: ⭐⭐⭐⭐ 多数据集多基线的全面评测，消融实验详尽
- 写作质量: ⭐⭐⭐⭐ 框架描述清晰，但符号较多需要仔细阅读
- 价值: ⭐⭐⭐⭐ 即插即用的实用框架，对结构化数据推理有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](../../NeurIPS2025/information_retrieval/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](hybgrag_hybrid_rag_skb.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](../../ICLR2026/information_retrieval/g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)

</div>

<!-- RELATED:END -->
