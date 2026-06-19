---
title: >-
  [论文解读] TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction
description: >-
  [AAAI 2026][强化学习][RAG] 提出 TAdaRAG，一个任务自适应的 RAG 框架，通过意图驱动的模板路由、监督微调和 REINFORCE 强化学习实现实时知识图谱构建，有效解决传统 RAG 的分块截断幻觉、推理链断裂和无关信息干扰三大问题，在 6 个公开数据集和 1 个商业场景基准上取得 SOTA。
tags:
  - "AAAI 2026"
  - "强化学习"
  - "RAG"
  - "知识图谱"
  - "任务自适应"
  - "REINFORCE"
  - "长文本理解"
---

# TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction

**会议**: AAAI 2026  
**arXiv**: [2511.12520](https://arxiv.org/abs/2511.12520)  
**代码**: [github.com/IAAR-Shanghai/TAdaRAG](https://github.com/IAAR-Shanghai/TAdaRAG)  
**领域**: 强化学习  
**关键词**: RAG, 知识图谱, 任务自适应, REINFORCE, 长文本理解

## 一句话总结

提出 TAdaRAG，一个任务自适应的 RAG 框架，通过意图驱动的模板路由、监督微调和 REINFORCE 强化学习实现实时知识图谱构建，有效解决传统 RAG 的分块截断幻觉、推理链断裂和无关信息干扰三大问题，在 6 个公开数据集和 1 个商业场景基准上取得 SOTA。

## 研究背景与动机

### 问题背景

RAG（检索增强生成）通过检索外部知识来增强 LLM 的生成质量，是缓解幻觉问题的主流方案。然而，当前 RAG 系统在实际应用中面临三个核心瓶颈：

### 三大核心问题

**问题 1：分块截断导致的幻觉**。检索到的长文档因输入窗口限制必须切分为小块（chunk），这种截断导致完整知识的信息丢失。例如一段法律条款被拆成多个 chunk 后，每个 chunk 都不完整，模型无法正确整合信息，产生事实性错误。

**问题 2：推理链断裂**。离散的 chunk 无法捕获语料内在的逻辑关系。在多跳推理任务（如 HotpotQA、2WikiMQA）中，答案需要跨文档推理，但独立的 chunk 之间缺乏结构化的逻辑连接，导致推理链不连贯。

**问题 3：无关信息干扰**。传统 RAG 检索的是非结构化文本，包含大量与问题无关的细节，干扰模型提取关键信息，影响实用性。

### 与现有图增强 RAG 的区别

GraphRAG、HippoRAG、PathRAG 等方法虽然利用知识图谱组织信息，但它们**依赖预构建的 KG**——需要人工维护、缺乏可扩展性、且在新领域/任务上适应性差。TAdaRAG 的核心创新在于：**将 KG 构建集成到推理过程中（而非检索阶段）**，实现实时、任务自适应的知识图谱动态生成。

## 方法详解

### 整体框架

TAdaRAG 采用两阶段训练：
- **阶段 1**：监督知识抽取微调（SFT）—— 学习高质量 KG 抽取能力
- **阶段 2**：任务自适应 KG 构建（RL）—— 通过 REINFORCE 算法优化 KG 构建以最大化下游任务性能

推理时：模型根据查询和检索文档动态构建任务自适应 KG，整合到生成管线中产生回答。

### 关键设计

#### 1. **意图驱动的模板路由机制**

预训练语言模型在实体抽取方面常产生无关或冗余实体，尤其在工业场景中。TAdaRAG 设计了领域特定的抽取模板来规范化知识抽取：

- 首先识别输入文本的应用领域（健康、法律、新闻等）
- 通过提示进行意图检测，选择对应的模板 $t$
- 模板指定：该领域需要哪些实体类型、实体描述规范、实体间关系的定义

构建指令集 $I = \{q, r, t\}$（查询、外部知识、模板），然后用强 LLM（GPT-4o/DeepSeek）执行知识抽取生成高质量 KG。

基于此构建了覆盖 4 个问题领域、7 个子数据集、共 9,548 个样本的 SFT 数据集。

#### 2. **并行子图构建 + 混合网络**

模型为每个输入构建 $p$ 个并行子图 $g_i = \{g_i^1, g_i^2, \ldots, g_i^p\}$，使用可学习 token `<|startextraction|>` 和 `<|endextraction|>` 标记知识抽取的起止。

**混合网络（Mixing Network）**用于融合有无 KG 的信息：

给定指令-回答对 $(x_i, y_i)$ 和子图 $g_i^k$：

1. 计算无图隐藏状态 $H_{i,j}^{\text{base}}$ 和有图隐藏状态 $H_{i,j,k}^{\text{graph}}$
2. 通过三层 MLP+ReLU 计算融合权重：
    $\omega_{i,j,k} = \text{MLP}(\text{concat}(H_{i,j}^{\text{base}}, H_{i,j,k}^{\text{graph}}))$
3. 加权融合两种 log-likelihood：
    $l_{i,j,k}^{\text{mix}} = \omega_{i,j,k} \cdot l_{i,j,k}^{\text{w/ graph}} + (1-\omega_{i,j,k}) \cdot l_{i,j,k}^{\text{w/o graph}}$

这让模型能自动判断何时依赖 KG、何时直接回答。

#### 3. **REINFORCE 优化图构建**

目标：找到最优子图 $\tilde{g}^{(i)}$ 使 $\pi_\theta(y_i | x_i, \tilde{g}^{(i)})$ 最大化。

奖励函数设计：

$$R_{i,k} = \max(0, \mathcal{L}_i^{\text{base}} - \mathcal{L}_{i,k}^{\text{graph}} - \bar{R}_i)$$

直觉：如果引入 KG 后 loss 比不用 KG 降低得超过平均水平，则给予正奖励。$\bar{R}_i$ 是所有子图的平均收益基线。

REINFORCE 损失：
$$\mathcal{L}^{\text{REINFORCE}} = -R_{i,k} \cdot \log\pi_\theta(g_i^k | x_i)$$

### 损失函数 / 训练策略

总损失函数：

$$\mathcal{L} = \alpha \cdot \mathcal{L}^{\text{base}} + (1-\alpha) \cdot \mathcal{L}^{\text{graph}} + \beta \cdot \mathcal{L}^{\text{REINFORCE}}$$

- $\mathcal{L}^{\text{base}}$：无 KG 直接回答的 loss（保留模型独立回答能力）
- $\mathcal{L}^{\text{graph}}$：有 KG 辅助回答的 loss（学习整合 KG 信息）
- $\mathcal{L}^{\text{REINFORCE}}$：优化 KG 构建质量

训练配置：
- 骨干模型：Mistral-7B-Instruct、Qwen2.5-7B-Instruct、Qwen2.5-14B-Instruct
- 阶段 1（SFT）：5 epoch，最大序列长度 20480，学习率 5e-5，cosine 调度
- 阶段 2（RL）：3 epoch，学习率 5e-7，ZeRO-2，AdamW，bfloat16
- 采样温度 T=0.6（训练），贪心解码（评估）
- 最大 KG 长度 2048 tokens
- 训练总时间：约 16 小时（8×A100 80GB），SFT 4 小时，RL 12 小时

## 实验关键数据

### 主实验

**基于 Mistral-7B-Instruct（F1 / ROUGE-L）**：

| 方法 | Health | Biology | Legal | HotpotQA | 2WikiMQA | GovReport |
|------|--------|---------|-------|----------|----------|-----------|
| NaïveRAG | 34.80 | 34.10 | 35.80 | 37.60 | 20.60 | 27.40 |
| GraphRAG | 35.60 | 34.80 | 37.65 | 38.00 | 36.50 | 25.60 |
| MEMORAG | 37.40 | 35.70 | 51.20 | 42.90 | 30.30 | 31.60 |
| **TAdaRAG (w/ reinforce)** | **40.77*** | **39.31*** | 49.88 | **44.83*** | **39.31*** | **36.41*** |

**基于 Qwen2.5-7B-Instruct**：

| 方法 | Health | Biology | Legal | HotpotQA | 2WikiMQA | GovReport |
|------|--------|---------|-------|----------|----------|-----------|
| MEMORAG | 36.87 | 36.00 | 47.60 | 37.99 | 35.32 | 31.13 |
| **TAdaRAG (w/ reinforce)** | **42.38*** | **40.75*** | 46.83 | **49.23*** | **43.79*** | **36.95*** |

* 标记表示 p < 0.01 的统计显著改进。

### 消融实验

| 阶段 | Health | Biology | 2WikiMQA | GovReport | 说明 |
|------|--------|---------|----------|-----------|------|
| NaïveRAG | 34.80 | 34.10 | 20.60 | 27.40 | 基线 |
| w/ graph（提示式 KG） | 38.19 | 36.87 | 38.48 | 33.72 | 仅靠提示就有大幅提升 |
| w/ sft（SFT 微调） | 40.00 | 38.92 | 38.86 | 35.39 | SFT 进一步优化抽取质量 |
| **w/ reinforce（完整）** | **40.77** | **39.31** | **39.31** | **36.41** | RL 在所有数据集上达到最优 |

**KG 规模在训练阶段间的变化**（以 Mistral-7B 为例）：

| 阶段 | Health 图大小 | Health 实体数 | HotpotQA 图大小 | HotpotQA 实体数 |
|------|-------------|-------------|----------------|----------------|
| Base（提示） | 7303 | 58.3 | 1894 | 16.1 |
| SFT | 5146 | 50.0 | 573 | 12.5 |
| Reinforce | **2006** | **44.2** | **257** | **10.1** |

随训练推进，KG 越来越精简——REINFORCE 学会了只保留任务相关的关键信息。

### 关键发现

1. **KG 构建从粗到精**：从阶段 1 到阶段 2，图大小大幅缩小（7303→2006），实体数也减少（58→44），但性能持续提升——RL 有效地学会了"精简"
2. **2WikiMQA 上提升最大**（20.60→39.31，+18.71），说明结构化 KG 对多跳推理帮助最大
3. **并行子图数量**：最优为 3 个，太少不够多样、太多引入噪声（但 Qwen2.5 对此更鲁棒）
4. **商业场景验证**：在 NowNewsQA（新闻 QA）上，TAdaRAG 在简洁性（8.25 vs 7.63）和事实性（8.45 vs 7.85）上显著优于 PathRAG（p < 0.0001）
5. **LLM 评估与人工评估高度一致**：Pearson 相关系数在 0.706-0.925 之间

## 亮点与洞察

- **将 KG 构建从检索阶段移到推理阶段**是核心创新，实现了真正的"实时"和"任务自适应"
- **REINFORCE 让模型自动学会精简 KG**——不是手动设定抽取规则而是端到端优化
- **Mixing Network 的设计巧妙**：让模型自主决定对每个 token 依赖 KG 还是直接回答
- 已在**商业系统（Xinyu AI Search）中部署**，具有实际落地价值
- 覆盖了从开放域 QA 到法律、医学、到长文本摘要的多种场景，泛化性强

## 局限与展望

- **动态 KG 构建增加了计算开销**——每次推理都需要生成 KG，延迟比标准 RAG 更高
- 部分依赖**手工设计的领域模板**（尽管只是冷启动用），限制了全自动化程度
- Legal 数据集上不及 MEMORAG（49.88 vs 51.20），可能因法律文本的特殊性需要更专业的模板
- 并行子图数量的最优值（3）是经验选定的，缺乏理论分析
- 当前只在 7B 和 14B 规模上验证，更大模型的效果和效率权衡未知

## 相关工作与启发

- **与 GraphRAG 的本质区别**：GraphRAG 预构建全局 KG + 社区摘要，TAdaRAG 按需实时构建任务特定 KG
- **与 PathRAG 互补**：PathRAG 从索引图中提取关键路径，TAdaRAG 从原始文本构建全新的 KG
- **MEMORAG** 使用记忆模块压缩数据库并生成检索线索，是最强基线
- **RL 在 RAG 中的应用趋势**：从检索策略优化到本文的知识结构优化，RL 在 RAG 领域的角色越来越重要
- 为"Agentic RAG"提供了思路：智能体动态选择知识表示形式（文本 vs KG vs 摘要）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 实时任务自适应 KG 构建的思路新颖，但各组件（意图路由、SFT、REINFORCE）较为标准
- **实验充分度**: ⭐⭐⭐⭐⭐ — 6+1 个数据集、3 个骨干模型、统计检验、人工评估、商业部署验证
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰（三个案例图），方法层次分明
- **价值**: ⭐⭐⭐⭐⭐ — 在多跳推理（+18.71）和长文本摘要（+9.01）上提升巨大，已商业部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](../../CVPR2026/reinforcement_learning/reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)
- [\[NeurIPS 2025\] Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/improving_retrieval-augmented_generation_through_multi-agent_reinforcement_learn.md)
- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](../../NeurIPS2025/reinforcement_learning/knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [\[ICML 2026\] DRIVE: Distributional and Retrieval-Augmented Bidding with Value Evaluation](../../ICML2026/reinforcement_learning/drive_distributional_and_retrieval-augmented_bidding_with_value_evaluation.md)
- [\[AAAI 2026\] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)

</div>

<!-- RELATED:END -->
