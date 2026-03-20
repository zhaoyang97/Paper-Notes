# Agentic NL2SQL to Reduce Computational Costs

**会议**: NeurIPS 2025  
**arXiv**: [2510.14808](https://arxiv.org/abs/2510.14808)  
**代码**: 无  
**领域**: Agent  
**关键词**: NL2SQL, text-to-SQL, agentic reasoning, token reduction, datalake

## 一句话总结
提出 Datalake Agent，一个基于交互循环的 agentic NL2SQL 系统，通过分层的信息获取策略（GetDBDescription -> GetTables -> GetColumns -> DBQueryFinalSQL）让 LLM 按需请求数据库 schema 信息而非一次性接收全部，在 319 张表的场景下将 token 使用量减少 87%、成本降低 8 倍，同时在复杂查询上保持更好的性能。

## 研究背景与动机

1. **领域现状**：LLM 在 NL2SQL 领域表现出色，但在企业级大规模数据库集合上部署时，需要将所有数据库的 schema 信息都放入 prompt，导致 prompt 极长且成本高。
2. **现有痛点**：(1) 研究表明 prompt 超过 1000 tokens 时 GPT 模型在表格数据上的效果退化为猜测；(2) 大多数 NL2SQL 任务只需要子集数据库的 schema，大量 meta-information 是冗余的；(3) 现有 benchmark（Bird、Spider）都只提供相关 schema，不反映企业真实场景。
3. **核心矛盾**：直接将所有 schema 信息给 LLM 既昂贵又降低性能，但 LLM 事先不知道哪些数据库/表是相关的。
4. **本文要解决什么？** 如何让 LLM 在大规模数据库集合上高效地只获取必要的 schema 信息？
5. **切入角度**：设计分层的信息获取命令（数据库描述->表列举->列详情），让 LLM 在交互循环中逐步缩小范围。
6. **核心 idea 一句话**：用 agentic 交互循环替代一次性 schema dumping，让 LLM 像调查员一样逐步获取、精炼、定位所需信息。

## 方法详解

### 整体框架
用户查询 -> LLM 进入推理循环 -> 通过预定义命令（GetDBDescription/GetTables/GetColumns）逐步获取 schema 信息 -> 必要时回退到更粗粒度 -> 构建 SQL 查询 -> DBQueryFinalSQL 执行。

### 关键设计

1. **分层信息获取**:
   - 做什么：提供四个命令让 LLM 按需获取不同粒度的 schema 信息。
   - 核心思路：GetDBDescription（获取数据库高层摘要）-> GetTables（列举特定数据库的表）-> GetColumns（暴露列级元数据，包括名称和类型）-> DBQueryFinalSQL（执行最终 SQL）。层次结构从粗到细，LLM 可以随时回退到更粗的层级。
   - 设计动机：模拟人类数据分析师的工作流程——先了解有哪些数据库，再深入看表结构，最后关注具体列。

2. **迭代精炼**:
   - 做什么：允许 LLM 在信息获取过程中灵活调整策略。
   - 核心思路：LLM 可以在任何点回退到更粗的信息粒度，然后重新精炼。这种反馈驱动的循环允许 LLM 自主推理复杂的数据库结构。
   - 设计动机：NL2SQL 不是线性过程——LLM 可能需要探索多个数据库才能找到正确的表。

### 损失函数 / 训练策略
无训练。使用 GPT-4-mini，temperature=0.1。作为 benchmark，人工构建了 100 个 Table QA 任务，跨 23 个数据库。

## 实验关键数据

### 主实验

| 设置 | Direct Solver Token | Datalake Agent Token | Agent 节省 |
|------|--------------------|--------------------|-----------|
| 42 tables | 7,407 | 3,670 | 50% |
| 159 tables | ~18,000 | ~3,900 | ~78% |
| 319 tables | 34,602 | 4,264 | **87%** |

### 消融/成本分析

| 模型 | 方法 | 1000任务成本 (319表) |
|------|------|---------------------|
| GPT-4-mini | Direct Solver | ~$100 |
| GPT-4-mini | Datalake Agent | ~$12 |
| o1 | Direct Solver | >$500 |
| o1 | Datalake Agent | ~$55 |

### 关键发现
- **Token 使用量几乎不随表数增长**：Datalake Agent 从 3,670（42表）到 4,264（319表）仅增长 16%，而 Direct Solver 增长 4.7 倍。
- **在复杂查询上优势更大**：简单查询（单表）Direct Solver 略优，但复杂查询（多表 join）Datalake Agent 明显更好。
- **成本差异随规模放大**：319 表时成本差距达 8 倍，对 o1 等更贵模型差距超过 $450/1000任务。
- **局限：可能产生无限推理循环**：LLM 有时找不到正确的表，导致重复请求。

## 亮点与洞察
- **token 使用量与数据库规模解耦**是核心价值：无论有 42 个还是 319 个表，agent 只获取它需要的信息量，这在企业级数据湖场景中极为重要。
- **简单但有效**：不需要任何训练，仅通过 agentic 交互设计就实现了显著的成本节省。
- **更接近真实企业场景**：用户不知道数据在哪个数据库中——这比标准 NL2SQL benchmark（提供相关 schema）更现实。

## 局限性 / 可改进方向
- **仅用 GPT-4-mini 评估**：未验证在更强/更弱模型上的效果。
- **无限循环问题**：需要添加早停/回退机制。
- **仅 100 个任务**：benchmark 规模较小。
- **模拟数据库**：18 个数据库是模拟的（只有 schema 无数据），不完全代表真实场景。

## 相关工作与启发
- **vs Direct NL2SQL**: 直接方法将所有 schema 放入 prompt，cost 和 performance 都随规模恶化。
- **vs Text2API**: Text2API 通过 API 端点获取数据但受限于 API 设计，NL2SQL 更灵活。
- **vs Spider/Bird 评估范式**: 这些 benchmark 只提供相关 schema，Datalake Agent 的设置更接近真实企业场景。

## 评分
- 新颖性: ⭐⭐⭐ 思路直觉但不算很新（agentic information retrieval）
- 实验充分度: ⭐⭐⭐ 只用一个模型，100 个任务，部分模拟数据
- 写作质量: ⭐⭐⭐⭐ 清晰简洁
- 价值: ⭐⭐⭐⭐ 企业 NL2SQL 场景很实用，cost 节省显著
