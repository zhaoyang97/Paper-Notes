---
title: >-
  [论文解读] ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search
description: >-
  [NeurIPS 2025][LLM安全][RAG安全] ReliabilityRAG 提出了一种利用文档可靠性信号（如搜索排名）进行对抗防御的 RAG 框架，通过在矛盾图上寻找最大独立集（MIS）来识别一致的文档子集并优先选择高可靠性文档，提供可证明的鲁棒性保证，同时在良性场景和长文本生成任务上保持高准确率。
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "RAG安全"
  - "对抗鲁棒性"
  - "最大独立集"
  - "可证明防御"
  - "文档可靠性"
---

# ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search

**会议**: NeurIPS 2025  
**arXiv**: [2509.23519](https://arxiv.org/abs/2509.23519)  
**代码**: 无  
**领域**: AI安全 / RAG防御  
**关键词**: RAG安全, 对抗鲁棒性, 最大独立集, 可证明防御, 文档可靠性

## 一句话总结

ReliabilityRAG 提出了一种利用文档可靠性信号（如搜索排名）进行对抗防御的 RAG 框架，通过在矛盾图上寻找最大独立集（MIS）来识别一致的文档子集并优先选择高可靠性文档，提供可证明的鲁棒性保证，同时在良性场景和长文本生成任务上保持高准确率。

## 研究背景与动机

RAG（检索增强生成）通过检索外部文档增强 LLM 输出的时效性和准确性，已被 Google AI Overview、Bing Chat、Perplexity 等搜索产品广泛采用。然而 RAG 系统的检索语料库容易受到语料投毒和提示注入等对抗攻击。

**现有防御的局限**：

**RobustRAG**：基于关键词或 next-token 概率的多数投票，信息损失大，在良性场景下性能差，长文本生成任务上严重退化

**AstuteRAG / InstructRAG**：设计目标是良性性能而非对抗鲁棒性，遇到攻击时防御力不足

**共同盲点**：所有现有防御将检索到的文档视为无序集合，忽略了检索系统提供的排名/可靠性信号

**关键机遇**：RAG 搜索系统天然包含经过数十年 anti-SEO 优化的文档排名信号。低排名文档本身更可能含噪声，也更容易被攻击者操纵。利用这一可靠性信号可以构建纵深防御。

**核心idea**：从图论角度寻找检索文档中的"一致多数"——在矛盾图上找最大独立集，即最大的互不矛盾文档子集，并优先选择可靠性更高的文档。

## 方法详解

### 整体框架

ReliabilityRAG 的完整流程分三阶段：
1. **检索**：获取按可靠性排序的 $k$ 个文档
2. **基于排名感知的 MIS 选择**：构建矛盾图，寻找最大独立集，优先选择高排名文档
3. **查询**：用选出的文档子集查询 LLM 生成答案

### 关键设计

1. **矛盾图构建**:

    - **隔离回答**：对每个文档 $x_i$ 单独查询 LLM 得到隔离答案 $y_i$
    - **矛盾检测**：用 NLI 模型（DeBERTa-v3-large）对每对答案 $(y_i, y_j)$ 判断是否矛盾（阈值 $\beta=0.5$）
    - **图编码**：文档为节点（继承检索排名），矛盾关系为边，构建无向图 $G=(V,E)$

2. **排名感知 MIS 搜索**:

    - 穷举所有 $2^{|V|}$ 子集（$k \leq 20$ 时可在毫秒内完成），筛选独立集
    - 选择最大的独立集；若有多个同等大小的 MIS，选择字典序最小的（即优先包含高排名文档）
    - $S^* = \arg\max_{S \text{ independent}} (|S|, -\text{lex}(S))$
    - 最终用 $S^*$ 中的文档生成答案

3. **加权采样聚合框架（Sampling + MIS）**:

    - 针对大规模检索（$k > 20$）的可扩展方案
    - 每轮按文档权重采样 $m$ 个文档组成上下文，生成中间答案
    - 执行 $T$ 轮采样后，用 MIS 对中间答案进行聚合
    - 权重采用指数衰减：$w(x_i) \propto \gamma^{i-1}$（$\gamma=0.9$）
    - 将 MIS 的计算从 $k$ 个文档降到 $T$ 个中间答案

### 可证明的鲁棒性保证

**定理 1**：假设攻击者最多腐蚀 $k' \leq k/5$ 个文档，NLI 模型在良性文档间的错误率 $\leq \epsilon_1$，在良性-恶意文档间的错误率 $\leq \epsilon_2$。当 $\epsilon_1, \epsilon_2$ 满足一定条件时，MIS 不包含恶意文档的概率至少为 $1 - e^{-O(k)}$。即随着检索文档数 $k$ 增大，误选恶意文档的概率指数级下降。

完美 NLI（$\epsilon_1 = \epsilon_2 = 0$）时，只要 $k' < k/2$，MIS 恰好是所有良性文档的集合。

## 实验关键数据

### 主实验（k=10, 提示注入攻击）

| 模型 | 方法 | RQA (良性) | RQA (@Pos1) | RQA (@Pos10) | Bio (良性) | Bio (@Pos1) | Bio (@Pos10) |
|------|------|-----------|-------------|-------------|-----------|-------------|-------------|
| Mistral-7B | Vanilla RAG | 64% | 49% | 12% | 72.9 | 65.5 | 11.5 |
| Mistral-7B | RobustRAG | 56% | 53% | 55% | 58.6 | 56.5 | 57.1 |
| Mistral-7B | **MIS** | **70%** | **68%** | **60%** | **73.5** | **69.7** | **71.5** |
| GPT-4o-mini | Vanilla RAG | 77% | 49% | 64% | 81.0 | 65.6 | 9.8 |
| GPT-4o-mini | RobustRAG | 71% | 68% | 70% | 61.2 | 60.4 | 61.4 |
| GPT-4o-mini | **MIS** | **76%** | **70%** | **76%** | **80.1** | **77.9** | **79.0** |

### 长文本生成表现（Bio 数据集）

| 方法 | Llama3.2-3B 良性 | Llama3.2-3B @Pos1 | Llama3.2-3B @Pos10 |
|------|-----------------|-------------------|-------------------|
| RobustRAG | 56.0 | 53.0 | 51.9 |
| AstuteRAG | 62.7 | 46.7 | 38.6 |
| **MIS** | **73.0** | **71.0** | **72.1** |

### 关键发现
- MIS 在良性场景下准确率与 Vanilla RAG 持平甚至更高，远超 RobustRAG
- 在长文本生成（Bio）上优势巨大：MIS 73.5 vs RobustRAG 58.6（Mistral-7B 良性）
- 具有排名感知特性：攻击低排名文档（Pos 10）时防御效果通常更好
- Sampling + MIS 可扩展到 k=50 个文档，保持同级别的鲁棒性
- MIS 在所有三个 LLM（Mistral-7B、Llama3.2-3B、GPT-4o-mini）上一致表现最优

## 亮点与洞察

- **图论视角解决 NLP 问题**：将"从多个文档中选择可靠子集"转化为矛盾图上的 MIS 问题，既优雅又有理论保证
- **利用现有信号而非创造新信号**：搜索排名是现成的可靠性指标，利用它构建纵深防御
- **良性性能和鲁棒性的双赢**：不像 RobustRAG 以牺牲良性性能换取鲁棒性
- **隔离回答的策略**：对每个文档单独查询 LLM，避免了文档间的交叉污染，使矛盾检测更精确
- **可证明鲁棒性**：在合理假设下提供指数级衰减的风险界，不只是经验性防御

## 局限与展望

- 隔离回答需要对每个文档单独调用 LLM，k=10 时需 10 次额外 LLM 调用，延迟成本高
- NLI 模型（DeBERTa-v3-large）本身可能被对抗性攻击绕过
- MIS 计算在 k>20 时需要使用近似方法，牺牲部分理论保证
- 假设攻击者旨在产生与良性文档矛盾的输出——对于更隐蔽的攻击（与良性一致但误导性信息）防御力可能不足
- 主要在 QA 数据集上验证，对多步推理、代码生成等场景的适用性待探索
- 对不相关但良性的文档需要额外的"I don't know"过滤机制

## 相关工作与启发

- **vs RobustRAG**: RobustRAG 用关键词/token 投票，信息损失大；MIS 在语句级别检测矛盾，保留更多信息
- **vs AstuteRAG**: AstuteRAG 处理知识冲突但非对抗设计；MIS 专注对抗鲁棒性且有理论保证
- **vs InstructRAG**: InstructRAG 通过指令让 LLM 自行去噪，依赖 LLM 能力；MIS 用独立的 NLI 模型更可靠

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 图论+NLI+可靠性信号的结合非常新颖，提供可证明鲁棒性在 RAG 领域是首创
- 实验充分度: ⭐⭐⭐⭐ 三个 LLM × 多个 QA 数据集 + 长文本生成 + 多攻击位置
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义严谨，理论证明完整，实验解读清晰
- 价值: ⭐⭐⭐⭐⭐ 对 RAG 安全部署有重大实际意义，特别是搜索引擎场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents](agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)
- [\[ACL 2025\] Towards Effective Extraction and Evaluation of Factual Claims](../../ACL2025/llm_safety/towards_effective_extraction_and_evaluation_of_factual_claims.md)
- [\[NeurIPS 2025\] DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents](drift_dynamic_rulebased_defense_with_injection_isolation_for.md)
- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)
- [\[ACL 2025\] AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection](../../ACL2025/llm_safety/agrail_a_lifelong_agent_guardrail_with_effective_and_adaptive_safety_detection.md)

</div>

<!-- RELATED:END -->
