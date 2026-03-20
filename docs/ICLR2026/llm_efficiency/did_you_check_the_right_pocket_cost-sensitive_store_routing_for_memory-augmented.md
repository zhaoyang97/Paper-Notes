# Did You Check the Right Pocket? Cost-Sensitive Store Routing for Memory-Augmented Agents

**会议**: ICLR 2026 Workshop  
**arXiv**: [2603.15658](https://arxiv.org/abs/2603.15658)  
**代码**: 无  
**领域**: Agent  
**关键词**: memory-augmented agents, store routing, cost-sensitive retrieval, RAG, memory architecture  

## 一句话总结
将记忆增强 Agent 的多存储检索形式化为代价敏感的存储路由问题（store routing），证明选择性检索相比全量检索可在减少 62% context token 的同时提升 QA 准确率（86% vs 81%），并提出基于语义信号的启发式路由基线。

## 研究背景与动机
1. **领域现状**：记忆增强 Agent（如 MemGPT）通常维护多个专用存储——短期记忆（STM, 当前对话）、摘要存储（Summary, 压缩用户事实）、长期记忆（LTM, 历史对话摘要）、情景记忆（Episodic, 原始转录）。但大多数系统对每个查询都从所有存储中检索。
2. **现有痛点**：全量检索带来两个代价——① 计算浪费（查询不可能包含答案的存储）；② 准确率下降（无关/噪声上下文降低信噪比，尤其在长上下文下模型需要在大量无关文本中寻找答案）。
3. **核心矛盾**：更多上下文 ≠ 更好性能。在长上下文设置中，无关存储引入的干扰信息实际上会误导模型——例如 LTM 中过时信息可能与 Summary 中最新信息冲突，模型有时会错误选择旧信息。
4. **本文要解决**：在检索之前决定"查哪个口袋"（which stores to search），将存储选择与存储内排序解耦，使 accuracy-cost tradeoff 显式化。
5. **切入角度**：从信息检索领域的联邦搜索（federated search）和认知科学的记忆分类（episodic vs semantic memory）获得启发，将查询路由到语义角色不同的存储。
6. **核心idea**：路由决策是记忆增强 Agent 设计的一等公民（first-class component），而非事后问题。形式化为 $\pi^*(q) = \arg\max_{G \subseteq \mathcal{S}} [\mathbb{E}[\text{Acc}(q,G)] - \lambda \sum_{s \in G} c_s]$。

## 方法详解

### 整体框架
四个记忆存储构成存储集合 $\mathcal{S} = \{\text{STM}, \text{Sum}, \text{LTM}, \text{Epi}\}$。给定查询 $q$，路由策略 $\pi$ 选择子集 $\hat{G} = \pi(q) \subseteq \mathcal{S}$，系统仅从选中的存储检索内容并拼接送入 LLM 生成答案。整个框架分两个评估阶段：① 合成路由评估（验证存储选择质量）→ ② LLM QA 评估（验证下游任务性能）。

### 关键设计

1. **路由评估指标体系**:
   - 做什么：量化存储选择的质量
   - Coverage = $\frac{1}{N}\sum_i \mathbf{1}[G_i \subseteq \hat{G}_i]$：是否包含了所有必要存储（漏存储 = 不可回答）
   - Exact Match = $\frac{1}{N}\sum_i \mathbf{1}[G_i = \hat{G}_i]$：是否精确选择了恰好必要的存储
   - Waste = $\frac{1}{N}\sum_i |\hat{G}_i \setminus G_i|$：多检索了多少不必要的存储
   - 设计动机：分离覆盖率（不漏）和精确度（不多），使 accuracy-cost tradeoff 可度量。Coverage 是硬约束，Waste 是软代价。

2. **混合启发式路由器（Hybrid Heuristic）**:
   - 做什么：基于查询语义信号选择目标存储
   - 核心规则：数量信号（"list all"） → {LTM, Epi}；时间信号（"before", "changed"） → {LTM, Epi}；多跳信号（"compare", "relate"） → {Sum, LTM}；当前会话（"just said", "today"） → {STM}；事实查找（"what is my"） → {Sum}
   - 无匹配时 fallback 到 {Sum, LTM}（六种两存储组合中覆盖率最高 89%）
   - 额外使用 query-store embedding similarity 作为 tiebreaker，贡献 +4% coverage
   - 设计原则：优先保 coverage（漏存储 = 不可回答），有信号时才收窄路由范围

3. **代价敏感决策理论框架**:
   - 做什么：为存储路由提供数学基础
   - 核心公式：$\pi^*(q) = \arg\max_{G \subseteq \mathcal{S}} [\mathbb{E}[\text{Acc}(q,G)] - \lambda \sum_{s \in G} c_s]$
   - $\lambda = 0$ 退化为全量检索（Uniform）；Oracle routing 是 $\pi^*$ 的近似上界
   - 解释力：当无关存储被检索时，有效检索代价增加而正确抽取概率可能下降（因上下文噪声），因此选择性检索在两端都有收益
   - 与检索门路由（retriever routing）的区别：存储路由是 memory-architecture level 的决策，存储间语义角色差异大（STM vs LTM vs Summary），粒度比 passage-level 路由更粗

### 路由策略谱系
从简到强排列：Uniform（全量，$\lambda=0$）→ Fixed Subset（如 STM+Sum+LTM）→ Hybrid Heuristic（规则+fallback）→ Oracle（理论上界）。论文评估了 12 种策略的完整谱系。

## 实验关键数据

### 合成路由评估（1000 查询，7 种类型）
| 策略 | Coverage | Exact Match | Waste |
|------|----------|-------------|-------|
| Uniform | 100% | 8% | 2.9 |
| Rule-based（仅语言学） | 57% | 35% | 0.5 |
| **Hybrid (Ours)** | **94%** | **58%** | **1.2** |
| Oracle | 100% | 100% | 0.0 |

### LLM QA 评估（150 问题）
| 模型 | 策略 | 总体准确率 | Short | Long | Token数 |
|------|------|-----------|-------|------|---------|
| GPT-4o-mini | Oracle | **86.7%** | 94% | **72%** | **299** |
| GPT-4o-mini | STM+Sum+LTM | 84.7% | 92% | 70% | 591 |
| GPT-4o-mini | Uniform | 81.3% | 92% | 60% | 787 |
| GPT-4o-mini | Hybrid | 70.7% | 80% | 52% | 379 |
| GPT-3.5 | Oracle | 85.3% | 93% | 70% | 299 |
| GPT-3.5 | Uniform | 83.3% | 91% | 68% | 787 |

### 特征消融
| 特征组 | Coverage | Δ |
|--------|----------|---|
| 语言学特征（代词、时态） | 57% | baseline |
| + 语义信号（数量、时间、多跳） | 90% | +33% |
| + Embedding 相似度 | 94% | +4% |

### 关键发现
- **Oracle 用 62% 少的 token 却获得更高准确率**（86.7% vs 81.3%），铁证"更多上下文 ≠ 更好"
- **长上下文放大过度检索惩罚**：Long 场景下 Oracle 72% vs Uniform 60%，差距从 Short 的 2% 扩大到 12%
- **固定策略 STM+Sum+LTM 接近 Oracle**（84.7% vs 86.7%），是实用的 deployable 方案
- **Coverage-Accuracy Gap**：Hybrid 94% coverage 但仅 70% QA 准确率。12% 错误来自路由失误（漏存储），18% 来自抽取失败（存储正确但模型提取错误）

## 全量检索为何反而更差？
两个机制：① **Needle in haystack**：787 token 中寻找少量相关信息，信噪比低；② **信息冲突**：不同存储含过时/冲突信息。典型案例："Who is my current manager?"——Summary 存储有正确答案 "Jennifer Williams"，但 LTM 中有历史记录 "Before the reorg...reported to Michael Torres"，全量检索时模型有时错误提取了更详细的旧信息。

## 亮点与洞察
- **"更多不是更好"的严格实证**——这对所有 RAG/记忆系统都有警示意义：盲目增加上下文长度可能适得其反
- **存储路由 vs 检索路由的概念区分**：存储路由是 architecture-level 决策（语义角色不同的存储），比 passage-level 检索路由影响更大但被忽视
- **两阶段评估设计**：先用合成标签验证路由质量，再用真实 LLM 验证下游性能，有效分离了路由决策与模型能力的影响
- **Coverage-Accuracy Gap 的分解分析**：区分路由错误（12%）和抽取错误（18%），指明了改进方向

## 局限性 / 可改进方向
- 标签来自查询分类规则而非人工标注，可能不完全反映真实场景的存储需求
- 启发式路由器与 Oracle 差距 16 点（70% vs 86%），需要端到端学习的路由策略（如 RL 优化 $\lambda$-tradeoff）
- 仅测试了 GPT-3.5 和 GPT-4o-mini 两个模型家族，上下文处理策略不同的模型可能响应不同
- 使用全存储内容拼接而非 top-k 检索，与生产系统设置有差异——路由与存储内检索的交互尚待研究
- 仅 150 个测试问题，统计效力有限

## 相关工作与启发
- **vs Self-RAG / FLARE**: 它们决定"是否检索"（when），本文决定"从哪个存储检索"（where），是互补维度
- **vs MemGPT**: MemGPT 关注记忆的组织和管理操作（读/写/整合），本文关注记忆访问的路由决策——两者可结合
- **vs ExpertRAG / RAP-RAG**: ExpertRAG 用 MoE 路由上下文选择，RAP-RAG 规划多跳检索序列；本文的存储路由粒度更粗（存储级 vs 段落级）
- **vs 联邦检索文献**: 信息检索中的资源选择（resource selection）算法估计每个集合的相关性分布，可直接迁移到 Agent 记忆路由
- 可启发 multi-store RAG 系统设计：先路由再检索 > 全量检索后让 LLM 自己过滤

## 评分
- 新颖性: ⭐⭐⭐ 问题定义清晰（存储路由作为一等公民），但技术方案（规则+fallback）较简单
- 实验充分度: ⭐⭐⭐ 两阶段评估设计合理，但样本规模小（150 问题）
- 写作质量: ⭐⭐⭐⭐ 论述逻辑清晰，failure case 分析深入，决策框架优雅
- 价值: ⭐⭐⭐⭐ "路由是一等公民"观点对记忆增强系统有实际指导意义，16点的 Oracle gap 激励后续研究
