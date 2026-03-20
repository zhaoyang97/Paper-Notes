# AgentiQL: An Agent-Inspired Multi-Expert Framework for Text-to-SQL Generation

**会议**: NeurIPS 2025  
**arXiv**: [2510.10661](https://arxiv.org/abs/2510.10661)  
**代码**: 将公开  
**领域**: Agent  
**关键词**: Text-to-SQL, multi-expert, question decomposition, adaptive routing, Spider benchmark

## 一句话总结
提出 AgentiQL，一个多专家 agent 框架用于 Text-to-SQL：reasoning agent 分解问题为子问题，coding agent 生成子查询，refinement 步骤校正列选择，adaptive router 在基线解析器和模块化 pipeline 之间智能路由，使用 14B 开源模型达到 86.07% EX（Spider），接近 GPT-4 SOTA(89.65%)。

## 研究背景与动机

1. **领域现状**：LLM 显著提升了 NL2SQL 能力，但单体 LLM 架构在复杂推理和多样 schema 处理上仍然困难。静态集成方法计算开销大，且部署大规模 LLM 成本高。
2. **现有痛点**：(1) 单体 LLM 对复杂多表 join、嵌套聚合等查询表现差；(2) 现有系统可解释性差——用户不知道 SQL 是怎么生成的；(3) 服务大模型（如 GPT-4）在实际应用中成本高、不切实际。
3. **核心矛盾**：大模型性能好但成本高，小模型成本低但对复杂查询能力不足。
4. **本文要解决什么？** 用小模型（14B 参数）通过多 expert 协作达到大模型级别的 NL2SQL 性能，同时保持可解释性和效率。
5. **切入角度**：将 SQL 生成分解为推理（问题分解）、编码（子查询生成）、精炼（列选择校正）三个专业组件，用 adaptive router 按查询复杂度选择路径。
6. **核心 idea 一句话**：通过 Divide-and-Merge 将复杂 SQL 生成分解为子问题-子查询对，结合 Column Selection 精炼和 Adaptive Routing，让 14B 开源模型接近 GPT-4 水平。

## 方法详解

### 整体框架
查询输入 -> Adaptive Router 判断复杂度 -> 简单查询走 baseline 直接生成 / 复杂查询走 AgentiQL pipeline（Table Selection -> Question Decomposition -> Sub-Query Generation -> Merge -> Column Selection）。

### 关键设计

1. **Divide（分解）**:
   - 做什么：将复杂自然语言查询分解为可管理的子问题。
   - 核心思路：三步——(1) Table Selection：reasoning LLM 过滤不相关的表得到缩减 schema $\tilde{s}$；(2) Question Decomposition：reasoning LLM 将查询分解为子问题 $\{x_1, ..., x_k\}$；(3) Query Generation：coding LLM 为每个子问题生成 SQL 子查询，错误时最多重试 R=3 次。
   - 设计动机：复杂查询需要多步推理，分解后每步更简单，也暴露了中间推理步骤提升可解释性。

2. **Merge（合并）**:
   - 做什么：将子查询合并为最终完整 SQL。
   - 核心思路：两种策略——(1) Last Sub-query：取最后一个子查询作为完整答案（假设推理 agent 将子问题排序使最后一个对应完整解）；(2) Planner&Executor：reasoning LLM 作为 planner 决定如何合并，coding LLM 作为 executor 实现合并。后者更通用但计算开销更大。
   - 设计动机：简单的 Last Sub-query 在很多情况下有效（子问题天然递进），但 Planner&Executor 处理需要真正合并多个子查询的复杂情况。

3. **Column Selection（列选择精炼）**:
   - 做什么：确保 SELECT 子句的列和列顺序与用户意图匹配。
   - 核心思路：reasoning LLM 对合并后的 SQL 检查 SELECT 子句，调整列选择和排序以精确匹配查询需求。
   - 设计动机：合并过程中 planner 可能引入多余列或错误排序，这个精炼步骤修复这些问题。一致性地提升 2-5% EX。

4. **Adaptive Routing**:
   - 做什么：根据查询复杂度选择 baseline 直接生成或 AgentiQL pipeline。
   - 核心思路：可用 XGBoost 分类器或 reasoning agent 作为 judge，使用 schema 大小（表数量）等简单信号作为复杂度代理。
   - 设计动机：简单查询用 baseline 更高效准确，只对复杂查询启用完整 pipeline，平衡效率和准确率。

### 损失函数 / 训练策略
无训练（推理时方法）。使用 Qwen2.5-Coder 系列（7B/14B/32B）作为 coding LLM，few-shot prompting。

## 实验关键数据

### 主实验（Spider Benchmark）

| 方法 | 7B EX | 14B EX | 32B EX |
|------|-------|--------|--------|
| Baseline (Qwen2.5-Coder) | 80.69 | 82.10 | 84.57 |
| Last Sub-query + CS | 74.44 | 80.21 | 83.16 |
| Planner&Executor + CS | **75.85** | **83.40** | **86.07** |
| GPT-4 SOTA (CHASE-SQL) | - | - | **89.65** |

### 消融实验

| 配置 | 7B EX | 14B EX |
|------|-------|--------|
| Last Sub-query w/o CS | 72.26 | 78.38 |
| Last Sub-query + CS | 74.44 (+2.18) | 80.21 (+1.83) |
| Planner&Executor w/o CS | 66.77 | 79.39 |
| Planner&Executor + CS | 75.85 (+9.08) | 83.40 (+4.01) |

### 关键发现
- **Column Selection 贡献一致**：在所有配置下提升 2-9% EX。对 Planner&Executor 策略尤为关键（修复 planner 引入的列错误）。
- **14B + Planner&Executor + CS 达到 86.07%**：缩小了与 GPT-4 SOTA(89.65%) 的差距至 3.5%，使用的是开源小模型。
- **Routing 提升鲁棒性**：baseline 在简单查询上更强，AgentiQL 在复杂查询上更强，routing 结合两者优势。
- **可解释性提升**：中间的子问题和子查询暴露了推理过程。

## 亮点与洞察
- **Column Selection 精炼步骤**看似简单但效果显著（最高 +9%）：说明 NL2SQL 的最后一公里在于 SELECT 子句的对齐，这常被忽视。
- **Adaptive Routing 的思路**很实用：不是所有查询都需要复杂 pipeline，按需分配计算资源。
- **可扩展到并行执行**：子查询生成可以并行，提升吞吐量。

## 局限性 / 可改进方向
- **仅在 Spider 上评估**：需要在 BIRD、SQL-Eval 等更多 benchmark 上验证。
- **大模型代价过高**：235B 模型在 4 个 A100 上每个问题需要约 60 分钟。
- **分解失败是主要失败模式**：如果推理 agent 分解问题不当，后续步骤都会失败。
- **建议方向**：引入 RL 优化查询生成质量（如 SkyRL-SQL）。

## 相关工作与启发
- **vs DIN-SQL**: 也做问题分解，但 AgentiQL 增加了 Column Selection 和 Adaptive Routing。
- **vs CHASE-SQL**: CHASE-SQL 用 GPT-4 达到 SOTA，AgentiQL 用 14B 开源模型接近其性能。
- **vs CodeS**: CodeS 用微调方法，AgentiQL 是纯推理方法（无需训练）。

## 评分
- 新颖性: ⭐⭐⭐ 多 expert 分解思路不算新，但 Column Selection + Routing 组合有效
- 实验充分度: ⭐⭐⭐ 只在 Spider 上，但消融详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐ 在小模型上接近 GPT-4 SOTA 的实用方案
