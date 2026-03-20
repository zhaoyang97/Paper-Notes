# Divide-Then-Aggregate: An Efficient Tool Learning Method via Parallel Tool Invocation

**会议**: ACL 2025  
**arXiv**: [2501.12432](https://arxiv.org/abs/2501.12432)  
**代码**: https://corn0205.github.io/  
**领域**: LLM Agent  
**关键词**: 工具学习, 并行工具调用, DAG结构, Process/Thread推理, LLM Agent

## 一句话总结
提出 DTA-Llama，将传统树搜索的串行工具调用路径转换为有向无环图（DAG）结构实现并行调用，设计 Process/Thread 推理框架使 LLM 在每轮中可分解任务并并行执行多个工具，在 StableToolBench 上使 Llama2-7B 达到 GPT-3.5 Parallel Function Calling 的水平。

## 研究背景与动机

1. **领域现状**：工具学习（tool learning）使 LLM 能够调用外部 API 完成真实世界任务。当前方法分为流水线式（CoT/ReAct，每轮调用一个工具）和树搜索式（DFSDT，通过深度优先搜索回溯提高容错率）。
2. **现有痛点**：
   - CoT/ReAct：每轮仅调用一个工具，感知范围窄，需要更多轮次
   - DFSDT：回溯机制导致工具调用序列更长，token 消耗和推理时间大幅增加
   - 两类方法都无法在单轮中并行调用多个工具
3. **核心矛盾**：如何在保持任务完成率的同时减少工具调用轮次和计算开销。
4. **切入角度**：类比操作系统的 Process/Thread 机制——每轮"Process"分解任务为可并行的子任务，多个"Thread"并行执行工具调用，执行后聚合结果。
5. **核心 idea 一句话**：将树搜索的串行路径转为 DAG 并行结构训练数据 + Process/Thread 推理框架，实现每轮多工具并行调用。

## 方法详解

### 整体框架
**数据构建**：DFSDT 搜索树 → 提取成功路径 → GPT-4 判断哪些工具可并行 → 转换为 DAG 结构 → 层次遍历生成并行训练数据 DTA-Tool (~20K 条)  
→ **模型训练**：在 DTA-Tool 上微调 Llama 系列模型  
→ **推理**：Process（LLM 分析任务状态 + 分解并行工具调用计划）→ Thread（并行执行工具 API）→ Intermediate State Lock（聚合结果）→ 循环直到完成

### 关键设计

1. **串行→并行数据转换**:
   - 从树搜索轨迹中提取成功路径 $\mathcal{P}$
   - 用 GPT-4 判断路径中哪些工具调用可以并行（无输入输出依赖 + 无因果关系）
   - 构建 DAG $\mathcal{G}$，层次遍历同一层的工具可并行执行
   - 数据过滤：去除循环调用、不完整调用、无法聚合的结构

2. **Process/Thread 推理框架**:
   - **Process**：LLM 评估任务状态 → 分析当前步骤需求 → 生成多个可并行的工具调用计划（名称+参数）
   - **Thread**：并行执行 Process 提出的所有工具调用
   - **Intermediate State Lock**：等待所有 Thread 完成后聚合结果，作为下一轮 Process 的输入

3. **训练设计**：
   - 从 Thought-Action-Observation 框架简化为 Thought-Observation（Action 集成到 Thought 中作为工具调用计划）
   - 损失函数：$\mathcal{L}(\theta) = -\log \sum_{i=1}^n p_\theta(y^i | q, y^{[1:i-1]}, o^{[1:i-1]})$

## 实验关键数据

### 主实验（StableToolBench 平均 SoPR/SoWR）

| 方法 | SoPR ↑ | SoWR ↑ |
|------|--------|--------|
| GPT-3.5 (ReAct) | 47.9 | - |
| GPT-3.5 (DFSDT) | 66.7 | 65.5 |
| GPT-3.5 (Parallel) | 61.9 | 53.0 |
| ToolLLaMA (DFSDT) | 54.2 | 47.1 |
| **DTA-Llama2-7B** | **60.7** | **53.5** |

DTA-Llama2-7B（开源 7B 模型）的性能与 GPT-3.5 Parallel Function Calling 相当。

### 效率对比（token 消耗）

| 方法 | 平均 Token 消耗 | 推理时间 |
|------|----------------|---------|
| DFSDT | 最高（大量回溯） | 最慢 |
| ReAct | 中等 | 中等 |
| **DTA (并行)** | **最低** | **最快** |

### 关键发现
- **并行调用显著减少调用轮次**：DTA 平均每条数据只需 2.46 轮工具调用
- 99.1% 的训练数据包含并行工具调用
- 在 Llama2-7B、Llama3-8B 等多个模型上验证了方法的泛化性
- DAG 数据质量高于原始串行数据——结构优化排除了冗余路径

## 亮点与洞察
- **串行→并行的数据转换是核心贡献**：利用 GPT-4 自动识别工具调用间的依赖关系，将树搜索路径重组为 DAG，这个数据工程思路可迁移到其他多步骤任务的数据构建
- **Process/Thread 类比操作系统的设计**直观且有效：Intermediate State Lock 的聚合机制是保证并行可靠性的关键
- 开源 7B 模型逼近 GPT-3.5 的并行函数调用能力，展示了数据结构优化的力量

## 局限性 / 可改进方向
- 依赖 GPT-4 做串行→并行转换，引入成本和潜在偏差
- 并行调用假设 API 支持并发且不会过载，实际部署时可能面临限流等问题
- 仅在 ToolBench 生态下验证，其他工具调用基准（如 API-Bank）的效果未知
- 未探索动态并行度调整——有些情况并行可能不如串行（信息依赖链较长时）

## 相关工作与启发
- **vs DFSDT (ToolLLM)**：DFSDT 通过回溯提高容错但代价巨大；DTA 通过并行化缩短路径且保持效果
- **vs GPT-3.5 Parallel FC**：GPT-3.5 的并行函数调用是黑盒实现；DTA 提供开源可复现方案且 7B 模型即可媲美
- **vs LLMCompiler**：LLMCompiler 用编译器方法并行化工具调用；DTA 的 DAG 数据转换+Process/Thread 框架更系统化

## 评分
- 新颖性: ⭐⭐⭐⭐ DAG 数据转换和 Process/Thread 推理框架的设计有创新性
- 实验充分度: ⭐⭐⭐⭐ StableToolBench 全面评估、效率分析、多模型泛化
- 写作质量: ⭐⭐⭐⭐ 图表清晰，类比直观，方法描述系统
- 价值: ⭐⭐⭐⭐ 为工具学习提供了实用的并行化方案和高质量数据集
