# Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection

**会议**: NeurIPS 2025  
**arXiv**: [2510.16499](https://arxiv.org/abs/2510.16499)  
**作者**: Michelle Yuan, Khushbu Pahwa, Shuaichen Chang, Mustafa Kaba, Jiarong Jiang, Xiaofei Ma, Yi Zhang, Monica Sunkara (AWS Agentic AI)
**代码**: 无  
**领域**: LLM Agent / 组件选择 / 在线优化  
**关键词**: agent composition, knapsack problem, component selection, online optimization, sandbox testing, multi-agent

## 一句话总结
将 Agent 组件选择问题形式化为在线背包问题，提出 Composer Agent 框架：通过沙盒实测（而非静态语义检索）评估组件真实能力，结合 ZCL 在线算法在预算约束下动态选取最优组件组合，单 Agent 工具选择成功率提升最高 31.6%，多 Agent 子代理选择成功率从 37% 跃升至 87%。

## 研究背景与动机
1. **选择悖论**：Agent 生态中工具、API、子 Agent 激增，开发者面临组合爆炸式的配置选择问题，手工策展和元数据检索难以胜任
2. **语义检索三大缺陷**：(a) 组件描述常与真实能力不符——如 "get_article_content" 的描述与 "Web Browsing" 技能高度匹配，但实际并非 web search 工具；(b) 不考虑成本-性能权衡；(c) 静态架构在需求演化时失效
3. **组件动态性**：工具 API 更新、模型版本迭代导致静态元数据过期，库中存在 "过时" 组件
4. **冗余浪费**：纯语义检索易选出功能高度重叠的组件（如 Offline Knapsack 为 GAIA 选了 10 个工具，Online 只需 4 个即达更优性能）
5. **缺乏理论框架**：Agent 组件选择此前无形式化问题定义和可证明的算法保证

## 问题定义
本文将 Agent 组件选择定义为 ADAS（Automated Design of Agentic Systems）的子问题：给定任务 τ（含描述 x）、组件库 A（每个组件 a_i 有成本 c_i 和描述 d_i）、预算 B，目标为找到最优子集 S* ⊆ A 使成功率 p_τ(S) 最大且总成本不超预算：

$$S^* = \arg\max_{S \subseteq A} p_\tau(S) \quad \text{s.t.} \quad \sum_{a_i \in S} c_i \leq B$$

该形式化直接对应经典背包问题，但有三个关键区别：(1) 真实成功率未知，需要通过迭代测试估计→在线背包；(2) 组件间可能存在非加性交互（协同或冲突）；(3) 组件库动态演化，需支持增量更新。

## 方法：四种 Composer Agent

### 1. Identity Composer（基线）
直接返回全部组件 f(A,τ,B) = A。120 个工具全部装配，不做任何选择。成本极高（$398），且在多 Agent 场景中 Supervisor 无法有效委派导致性能崩溃（Travel 域成功率仅 7%）。

### 2. Retrieval Composer
先用 LLM 将任务描述解析为最多 6 个技能（每个技能含名称、描述、重要性权重），再用嵌入模型（BGE-Large-EN-v1.5）逐技能检索 top-1 组件。**核心局限**：纯语义匹配——在 GAIA 实验中，"Web Browsing" 技能匹配到了 get_article_content 而非真正的 web_search 工具，导致成功率仅 0.19。

### 3. Offline Knapsack Composer
在检索基础上增加预算约束：检索 top-K 候选，以语义相似度分数为价值 v_i^OFF = Σ SIM(a_i, m)，用线性规划求解多选择背包问题，同时满足预算约束和技能覆盖约束。**问题**：价值仍基于语义相似度而非真实能力，容易选入冗余工具（GAIA 实验中选了 10 个工具，包含大量无用项）。

### 4. Online Knapsack Composer（核心贡献）
关键创新：在选择前通过沙盒实测评估组件真实能力。完整流程：
- **技能+测试查询生成**：LLM 将任务描述解析为技能列表，每技能生成 2-3 个单步测试问题及参考计划
- **候选检索**：逐技能用嵌入模型检索 top-K 候选组件
- **沙盒测试**：对每个候选组件在沙盒中运行测试查询，LLM-as-Judge 评判三档结果：helpful(1) / not helpful(0) / broken(-1)
- **价值计算**：v_i = Σ(未覆盖技能的权重 × 是否通过测试)，仅计算尚未被已选组件覆盖的技能贡献
- **ZCL 在线决策**：计算 value-to-cost 比 ρ_i = v_i/c_i，与动态阈值 Ψ 比较，仅当 ρ_i ≥ Ψ 时接纳组件

**ZCL 阈值公式**：Ψ = (Ue/L)^z · L/e，其中 L = 1/max(c_a)，U = Σw_j/min(c_a)，z = 已消耗预算占比。阈值随预算消耗指数上升→后期准入门槛越高→自然实现成本控制。理论竞争比为 ln(U/L)+1。

**运行时优化**：(a) 损坏组件永久标记跳过，避免重复测试；(b) 已覆盖技能不再为新候选评估该维度，减少冗余测试并防止功能重叠选择。

### 可选：AvaTaR 提示优化
复用沙盒测试产生的 trajectory 作为反馈信号，通过 AvaTaR 框架优化 Agent 系统提示词。优化后产生更系统化的工具调用指导：任务分解、工具-子任务匹配、查询重构、错误恢复策略。无额外数据收集成本。

## 实验

### 单 Agent 实验（工具选择，Claude 3.5 Sonnet）

| 方法 | GAIA | SimpleQA | MedQA | 工具数 | 预算花费 |
|------|------|----------|-------|--------|----------|
| Identity（全部 120 工具）| 0.47 | 0.80 | 0.92 | 122 | $398 |
| Retrieval | 0.19 | 0.24 | 0.87 | 4-6 | $12-23 |
| Offline Knapsack ($30) | 0.41 | 0.88 | 0.91 | 10 | $30 |
| Online Knapsack ($30) | 0.47 | 0.92 | 0.93 | 2-4 | $6-12 |
| Online Knapsack + AvaTaR ($30) | **0.47** | **0.92** | **0.93** | 2-4 | $6-14 |

- Online Knapsack 以 $12 达到 Identity ($398) 同等水平，Pareto 最优
- $10 预算下 Online Knapsack + AvaTaR 在 SimpleQA 上从 0.24 跃升至 **0.82**（提示优化的显著效果）

### 多 Agent 实验（子 Agent 选择，117 Agent 库）

| 方法 | Travel GSR | Mortgage GSR | 预算 |
|------|------------|--------------|------|
| Identity（全部 117 Agent）| 0.07 | 0.07 | $117 |
| Retrieval | 0.23 | 0.37 | $5-6 |
| Offline Knapsack ($6) | 0.17 | 0.70 | $3 |
| Online Knapsack ($6) | **0.40** | **0.87** | $5 |

- 多 Agent 场景中 Identity 完全失效（117 Agent 让 Supervisor 无法有效分工）
- Retrieval 误选语义相似但缺少实际工具的 "distractor" Agent
- Online Knapsack 通过沙盒测试识别无工具的冒牌 Agent，Mortgage 域成功率达 **87%**

### 跨模型验证
在 Claude 3.5 Haiku、Claude 3.7 Sonnet、Llama 4 Maverick/Scout、Qwen 2.5 72B、Llama 3.3 70B 上均观察到一致优势：Online Knapsack 始终优于 Retrieval 和 Offline 基线。多次运行方差极小。

## 关键案例分析
**GAIA 工具选择对比**（Table 2）：
- Retrieval 选了 pub_med, read_file, wolfram_alpha 等 6 个工具，**完全没有 web search**
- Offline Knapsack 选了 web_search_free + 9 个其他工具（含大量冗余），花完 $30 预算
- Online Knapsack 仅选 web_search_paid, arxiv, wikipedia 共 3 个工具，花费 $12，精准覆盖核心需求

**AvaTaR 优化效果案例**（SimpleQA "Reinhold Rudenberg 何时退休？"）：
- 优化前：4 次工具调用，web_search_free 被限频两次，最终放弃
- 优化后：3 次工具调用，战略性使用 wikipedia→semanticscholar→web_search_paid，正确回答 "1952"

## 亮点与贡献
- **理论贡献**：首次将 Agent 组件选择形式化为背包问题，ZCL 算法提供 ln(U/L)+1 竞争比的理论保证
- **沙盒测试范式**：实际运行候选组件评估真实能力，根本性解决语义检索中描述-能力不一致问题
- **统一框架**：单 Agent 工具选择与多 Agent 团队构建在同一 Composer Agent 框架下，仅需切换组件库
- **工程实用性**：120 工具/117 Agent 规模接近真实部署场景；成本从 $398 降至 $12 达到同等性能

## 局限性
1. **沙盒耗时**：测试过程需 10-30 分钟（取决于预算和候选数量），不适合实时场景
2. **任务描述要求**：需要开发者提供清晰的任务描述，对模糊/探索性需求支持不足
3. **独立评估假设**：仅逐个测试组件，未考虑组件间的协同或冲突效应（问题定义中提到但未解决）
4. **提示优化不稳定**：AvaTaR 在部分设置（如 MedQA $10 预算）出现性能回退
5. **安全隐患**：自动化组件选择可能无意引入恶意工具，缺乏安全审查机制
6. **缺少替代算法对比**：未与贪心、MDP 等其他优化范式做对比

## 评分
- 新颖性: ⭐⭐⭐⭐ Agent 组件选择→背包问题建模清晰自然，沙盒实测替代语义匹配是核心创新
- 实验充分度: ⭐⭐⭐⭐ 5 数据集 × 7 模型 × 多预算档 + Pareto 分析 + 多次运行方差验证
- 写作质量: ⭐⭐⭐⭐ 四种 Composer 层层递进，问题形式化到算法设计逻辑链完整
- 价值: ⭐⭐⭐⭐ 对 Agent 系统工程中工具/子 Agent 选择有直接实用指导价值
