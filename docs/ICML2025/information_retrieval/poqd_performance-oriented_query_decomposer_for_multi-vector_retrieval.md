---
title: >-
  [论文解读] POQD: Performance-Oriented Query Decomposer for Multi-Vector Retrieval
description: >-
  [ICML 2025][信息检索/RAG][Multi-Vector Retrieval] 提出 POQD，一个面向性能的查询分解框架，利用 LLM-based Prompt Optimizer 迭代优化查询分解 prompt，并通过交替训练算法联合优化 prompt 和下游 RAG 模型参数，在检索和端到端 QA 任务上大幅超越现有方法。
tags:
  - "ICML 2025"
  - "信息检索/RAG"
  - "Multi-Vector Retrieval"
  - "Query Decomposition"
  - "RAG"
  - "提示学习"
  - "LLM-based Optimizer"
---

# POQD: Performance-Oriented Query Decomposer for Multi-Vector Retrieval

**会议**: ICML 2025  
**arXiv**: [2505.19189](https://arxiv.org/abs/2505.19189)  
**代码**: [PKU-SDS-lab/POQD-ICML25](https://github.com/PKU-SDS-lab/POQD-ICML25)  
**领域**: 信息检索  
**关键词**: Multi-Vector Retrieval, Query Decomposition, RAG, Prompt Optimization, LLM-based Optimizer

## 一句话总结

提出 POQD，一个面向性能的查询分解框架，利用 LLM-based Prompt Optimizer 迭代优化查询分解 prompt，并通过交替训练算法联合优化 prompt 和下游 RAG 模型参数，在检索和端到端 QA 任务上大幅超越现有方法。

## 研究背景与动机

**Multi-Vector Retrieval（MVR）** 通过将查询和文档分解为细粒度单元（如 token/phrase），利用 MaxSim 操作计算相似度，在信息检索任务中表现优异。代表方法 ColBERT 将查询按 token 粒度分解。然而，作者发现：

**Token 级分解的致命缺陷**：ColBERT 按 token 分解查询会导致语义歧义。例如查询 "Hong Kong" 被拆为 "Hong" 和 "Kong" 两个 token，其中 "Kong" 可能匹配到黑猩猩图片（King Kong），导致检索到完全不相关的文档。将 "Hong Kong" 作为整体 phrase 则可正确检索。

**手动 prompt 分解质量差**：使用手工 prompt 让 LLM 分解查询（如 ICL-QD），可能产生不相关的子查询（如额外生成 "type" 这种无信息量的词），干扰 MaxSim 相似度计算，导致检索错误。

**优化的核心难点**：(a) 子查询搜索过程不可微，无法反向传播梯度；(b) 评估候选子查询需要训练完整的下游模型，计算开销极大。

这些问题引出核心研究问题：**如何自动生成任意粒度的子查询以最大化下游检索系统性能？**

## 方法详解

### 整体框架

POQD 包含两个核心 LLM 和一个交替训练算法：

- **Query Decomposer（查询分解器）**：接收 prompt $p$ 和输入查询 $Q$，输出分解后的子查询集合 $\{q_i\}_{i=1}^K$
- **Prompt Optimizer（提示优化器）**：基于历史 solution-score 对，迭代生成更优的 prompt prefix $p_0$
- **End-to-End Training（Algorithm 2）**：交替优化 prompt $p$ 和下游模型参数 $\Theta$

整体损失函数定义为：

$$\mathcal{L}(\Theta; p) = -\log\left(\sum_{D \in D_K} P_\theta(a|Q,D) P_\beta(D|Q)\right)$$

其中 $D_K$ 是基于 MVR 相似度检索的 Top-K 文档，$P_\theta$ 是生成器模型的答案似然度，$P_\beta$ 是检索模型的文档相关性概率。检索模型 $\beta$ 保持冻结，仅训练生成器 $\theta$。

### 关键设计

#### 1. MVR 相似度计算（MaxSim）

对查询 $Q$ 分解为子查询 $\{q_i\}_{i=1}^K$，文档 $D$ 分解为片段 $\{d_j\}_{j=1}^m$：

$$\text{SIM}_\theta(Q, D) = \frac{1}{K} \sum_{i=1}^K \max_{1 \le j \le m} E_\theta(q_i)^\top E_\theta(d_j)$$

核心思想：每个子查询找到文档中最相似的片段，聚合所有子查询的最大相似度作为整体评分。

#### 2. 查询分解优化（Algorithm 1）

给定固定的下游模型参数 $\Theta$，通过两步迭代搜索最优 prompt：

- **Step 1 - Prompt 生成**：Prompt Optimizer 接收两段 meta-prompt 和历史 solution-score 对列表 $LS$，生成候选 prompt prefix $p_0$。$p_0$ 与固定的任务描述模板拼接，构成完整 prompt $p$
- **Step 2 - 评估与记录**：用 $p$ 驱动 Query Decomposer 分解训练集所有查询，计算训练损失 $\mathcal{L}(\Theta; p)$，将 $(p, \mathcal{L}(\Theta; p))$ 加入 $LS$

**收敛条件**：当 $\mathcal{L}(\Theta; p) - \mathcal{L}(\Theta; p^{\text{old}}) \le \alpha$ 或迭代达到 $\kappa$ 次。

**幻觉过滤**：Query Decomposer 可能生成不在原查询中的 token，POQD 通过过滤步骤移除这些无关 token。

#### 3. 端到端交替训练（Algorithm 2）

```
输入：训练查询集 Q^train，下游模型参数 Θ
初始化随机 p^old
while 未收敛:
    调用 Algorithm 1 获取新 prompt p^new 和优化后的子查询
    if p^new == p^old: break
    用优化后子查询训练 Θ 共 τ 步（最小化 L(Θ; p^new)）
    p^old ← p^new
最终：固定 p^new，训练 Θ 至收敛
```

关键设计点：每轮不将 $\Theta$ 训练至收敛，而只训练 $\tau$ 步（默认 $\tau=3$），大幅节省计算成本。

### 损失函数 / 训练策略

**理论保证（Theorem 4.4）**：在 $\mu$-PL* 条件和 $L$-smoothness 假设下，当 prompt 从 $p^{\text{old}}$ 更新为 $p^{\text{new}}$ 时：

$$\mathcal{L}(\Theta^*(p^{\text{old}}); p^{\text{old}}) - \mathcal{L}(\Theta^*(p^{\text{new}}); p^{\text{new}}) \ge \alpha - (1 - \frac{\mu}{2L})^\tau M$$

其中 $M$ 是损失上界。由于 $(1 - \frac{\mu}{2L}) \in (0, 1)$，适当设置 $\tau$（例如 $\tau = \log_{1-\frac{\mu}{2L}}(\frac{\alpha}{2M})$）可保证收敛损失严格递减。默认 $\tau=3$ 在效率和性能间取得平衡。

**弱监督优化**：POQD 优化的是下游 RAG 的端到端性能，而非中间检索指标，因此是弱监督方式。这使得它在多跳 QA 等动态生成查询的场景中也能有效工作。

**超参数默认配置**：$\alpha=0.02$，$\tau=3$，$\kappa=5$。

## 实验关键数据

### 主实验

**检索精度（Hit@K）**：

| 数据集 | 指标 | POQD | ColBERT-orig | ICL-QD | Dense Retrieval | 最大提升 |
|--------|------|------|-------------|--------|----------------|---------|
| WebQA (Image) | Hit@1 | **42.33** | 38.95 | 39.39 | 41.38 | +0.64 |
| MultiModalQA (Image) | Hit@1 | **58.26** | 36.96 | 54.34 | 50.00 | +3.92 |
| ManyModalQA (Image) | Hit@1 | **28.67** | 21.05 | 27.76 | 27.38 | +0.78 |
| WebQA (Text) | Hit@2 | **53.24** | 52.16 | 41.37 | 43.52 | +1.08 |
| MultiModalQA (Text) | Hit@2 | **80.58** | 79.89 | 71.43 | 66.44 | +0.69 |
| ManyModalQA (Text) | Hit@2 | **92.35** | 87.07 | 85.14 | 49.25 | +5.28 |

**端到端 QA 准确率（Exact Match）**：

| 数据集 | POQD | ColBERT-orig | ICL-QD | w/o RAG | 最大提升 |
|--------|------|-------------|--------|---------|---------|
| WebQA (Image) | **82.83** | 81.96 | 82.37 | 80.31 | +0.46 |
| MultiModalQA (Image) | **61.74** | 49.13 | 46.78 | 16.52 | **+12.61** |
| ManyModalQA (Image) | **37.92** | 32.29 | 34.70 | 30.98 | +3.22 |
| WebQA (Text) | **62.22** | 61.14 | 58.63 | 56.47 | +1.08 |
| MultiModalQA (Text) | **68.10** | 61.73 | 61.86 | 40.36 | +6.24 |
| ManyModalQA (Text) | **81.27** | 77.66 | 76.69 | 32.28 | +3.61 |
| StrategyQA (Text) | **75.55** | 65.50 | 60.70 | 58.76 | +10.05 |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| $\alpha=0.01$ | 收敛慢 | 每轮 prompt 更新要求的损失降低太小，优化效率低 |
| $\alpha=0.02$（默认）| 最佳平衡 | 训练损失平滑下降，无突变 |
| $\alpha=0.05$ | underfitting | Algorithm 1 难以找到满足条件的 prompt |
| $\tau=0$ | 最差性能 | 不训练下游模型直接优化 prompt |
| $\tau=3$（默认）| 效率/性能最佳 | 3 步训练已足够为 prompt 优化提供信号 |
| $\tau=5$ | 性能略优但训练时间显著增大 | 收益递减，不划算 |
| GPT-4 作为 Decomposer | 59.71 QA acc | POQD 在不同 LLM 下均优于 ICL-QD/ICLF-QD |
| DeepSeek-V3 作为 Decomposer | 60.43 QA acc | 同上，一致性强 |
| vs Query Rewrite | 61.87 vs 52.16 | POQD 大幅优于 query rewrite 策略 |

### 关键发现

1. **检索提升 → QA 提升**：POQD 在检索精度和端到端 QA 准确率上全面领先，验证了优质子查询对 RAG 系统的全链路价值
2. **训练效率高**：prompt 优化阶段的时间开销几乎可忽略，主要开销来自生成器训练；推理时查询分解时间远小于模型推理时间
3. **多跳 QA 同样有效**：在 StrategyQA 上 POQD 比 ColBERT-orig 提升 10.05%，说明 POQD 在动态查询生成场景也能发挥作用
4. **ColBERT token 级分解的局限性**：ColBERT（不使用原始编码器时）性能显著低于其他方法，验证了 token 级分解的不合理性

## 亮点与洞察

1. **将 prompt 优化引入检索系统**：首次将 LLM-based optimizer 用于 MVR 查询分解，巧妙绕过不可微问题
2. **交替训练 + 理论保证**：Algorithm 2 的设计既实用（$\tau=3$ 只需极低成本）又有严格理论基础（Theorem 4.4）
3. **弱监督优化范式**：不直接优化检索指标，而是优化端到端 QA 性能，让查询分解自然适配下游任务
4. **动机示例极具说服力**："Hong Kong" → "Kong" 匹配 King Kong 的案例直观展示了 token 级分解的荒谬性
5. **通用性强**：POQD 可无缝集成到任意基于检索的系统中，不限于 RAG

## 局限与展望

1. **LLM 调用成本**：每次优化 prompt 和分解查询都需要调用 LLM，对大规模部署可能有额外开销
2. **检索模型冻结**：当前仅训练生成器，若能联合更新检索模型可能带来更大收益（但计算开销也更大）
3. **子查询仅来源于原始查询 token**：强制子查询只能由原查询的 token 组成，可能限制了查询扩展的能力
4. **数据集规模有限**：实验数据集相对较小，在大规模工业场景（如搜索引擎）的效果有待验证
5. **Prompt Optimizer 的可控性**：LLM-based optimizer 的优化过程不可控，可能在某些领域收敛到局部最优

## 相关工作与启发

- **ColBERT** (Khattab & Zaharia, 2020)：MVR 开山之作，token 级分解 + late interaction，POQD 的出发点
- **OPRO** (Yang et al., 2024)：LLM as Optimizer 的开创工作，POQD 的 Prompt Optimizer 直接基于此
- **Search-in-the-Chain** (Xu et al., 2024)：多跳 QA 的 SOTA，POQD 在 StrategyQA 上采用其框架
- **启发**：将 LLM-based optimizer 用于优化其他不可微的系统组件（如 RAG 的 reranking 策略、retrieval 策略选择等）是一个有前景的方向

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | ⭐⭐⭐⭐ | LLM optimizer + MVR 查询分解的组合新颖 |
| 理论性 | ⭐⭐⭐⭐ | 有完整的收敛性理论保证 |
| 实验 | ⭐⭐⭐⭐ | 多数据集、多模态、多消融，覆盖全面 |
| 实用性 | ⭐⭐⭐⭐ | 可直接集成到现有 RAG 系统，代码开源 |
| 写作 | ⭐⭐⭐⭐ | 动机示例清晰，框架描述完整 |
| 总评 | ⭐⭐⭐⭐ | 扎实的工作，理论+实验+实用性兼备 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] LEMUR: Learned Multi-Vector Retrieval](../../ICML2026/information_retrieval/lemur_learned_multi-vector_retrieval.md)
- [\[ACL 2026\] Hybrid-Vector Retrieval for Visually Rich Documents: Combining Single-Vector Efficiency and Multi-Vector Accuracy](../../ACL2026/information_retrieval/hybrid-vector_retrieval_for_visually_rich_documents_combining_single-vector_effi.md)
- [\[NeurIPS 2025\] Reliable Decision Making via Calibration Oriented Retrieval Augmented Generation](../../NeurIPS2025/information_retrieval/reliable_decision_making_via_calibration_oriented_retrieval_augmented_generation.md)
- [\[AAAI 2026\] OPERA: A Reinforcement Learning--Enhanced Orchestrated Planner-Executor Architecture for Reasoning-Oriented Multi-Hop Retrieval](../../AAAI2026/information_retrieval/opera_a_reinforcement_learning--enhanced_orchestrated_planner-executor_architect.md)
- [\[ACL 2025\] Investigating the Robustness of Retrieval-Augmented Generation at the Query Level](../../ACL2025/information_retrieval/investigating_the_robustness_of_retrieval-augmented_generation_at_the_query_leve.md)

</div>

<!-- RELATED:END -->
