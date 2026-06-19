---
title: >-
  [论文解读] AutoTool: Efficient Tool Selection for Large Language Model Agents
description: >-
  [AAAI 2026][LLM Agent][tool selection] 提出 AutoTool，一种基于图的工具选择框架，利用工具使用惯性（tool usage inertia）构建工具惯性图（TIG），通过统计结构绕过重复的 LLM 推理来选择工具和填充参数，在保持任务完成率的同时减少最多 30% 的推理开销。
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "tool selection"
  - "LLM agent efficiency"
  - "tool usage inertia"
  - "graph-based planning"
  - "inference cost reduction"
---

# AutoTool: Efficient Tool Selection for Large Language Model Agents

**会议**: AAAI 2026  
**arXiv**: [2511.14650](https://arxiv.org/abs/2511.14650)  
**代码**: [GitHub](https://github.com/jiajingyyyyyy/AutoTool)  
**领域**: Agent  
**关键词**: tool selection, LLM agent efficiency, tool usage inertia, graph-based planning, inference cost reduction

## 一句话总结

提出 AutoTool，一种基于图的工具选择框架，利用工具使用惯性（tool usage inertia）构建工具惯性图（TIG），通过统计结构绕过重复的 LLM 推理来选择工具和填充参数，在保持任务完成率的同时减少最多 30% 的推理开销。

## 研究背景与动机

1. **领域现状**: LLM Agent 已成为自动化复杂任务的强大工具，ReAct 等框架通过"思考-行动-观察"循环驱动多步决策。
2. **现有痛点**: 当前框架在每一步都依赖 LLM 推理来选择工具，导致高计算开销和延迟，尤其在多步骤任务中 LLM 调用次数极多。
3. **核心矛盾**: 并非所有决策步骤都需要 LLM 的完整推理能力，许多工具调用发生在高度模式化的上下文中，当前做法属于过度使用 LLM。
4. **关键观察**: 作者发现**工具使用惯性**现象——工具调用遵循可预测的序列模式。例如在 ScienceWorld 中，`go_to` 之后 88.7% 的情况下跟 `look_around`。
5. **理论验证**: 通过 k 阶马尔可夫链分析，0 阶熵 3.50 bits → 1 阶 2.52 bits → 2 阶 1.93 bits，似然比检验显著（$p<.001$），证实序列依赖性。
6. **核心 idea**: 用图结构捕捉工具调用的统计规律，在高置信时直接选择工具，仅在不确定时回退到 LLM 推理。

## 方法详解

### 整体框架

AutoTool 在每次标准 LLM 调用前尝试"惯性调用"，包含两个阶段：① **Inertia Sensing**（惯性感知）预测下一个工具；② **Parameter Filling**（参数填充）自动化参数赋值。仅当两阶段都成功时绕过 LLM，否则回退到标准推理。

### 模块 1：Tool Inertia Graph (TIG) 构建

- **层次化节点结构**: Tool Nodes（工具节点）包含功能描述和执行状态，每个工具节点内嵌 Parameter Nodes（参数节点）子图
- **两类有向边**:
    - Tool Sequence Edges：连接工具节点，编码序列依赖关系
    - Parameter Dependency Edges：连接参数节点，建模工具间数据流
- **在线增量构建**: 从历史执行轨迹动态学习，边权通过成功/失败反馈进行正/负强化
- 仅从 LLM 生成的高置信序列中强化边权，防止惯性调用的错误传播

### 模块 2：Inertia Sensing（CIPS 评分）

综合惯性潜力分数：

$$\text{CIPS} = (1-\alpha) \cdot \text{Score}_{\text{freq}} + \alpha \cdot \text{Score}_{\text{ctx}}$$

- **Frequency Score**: 基于 TIG 边权的历史使用频率
- **Contextual Score**: 用 SimCSE 计算当前 agent 直觉与候选工具描述的语义相似度
- 仅当 $\text{CIPS}(v^*) > \theta_{\text{inertial}}$ 时进入参数填充阶段

### 模块 3：Hierarchical Parameter Filling

严格优先级的非 LLM 参数填充策略：
1. **依赖回溯**: 沿 TIG 中的参数依赖边回溯，从前序工具的输出中获取参数值
2. **环境状态匹配**: 从 agent 维护的关键状态（如当前位置）中提取
3. **启发式填充**: 基于当前状态或任务目标推断
- 任一参数无法确定则放弃惯性调用，回退 LLM

### 安全约束

- 惯性调用不超过总操作数的 30%
- 禁止连续惯性调用
- 容错机制：检测到连续工具失败时触发恢复路径

## 实验关键数据

### 主实验：效率提升（ReAct + AutoTool）

| 数据集 | Progress Rate | Token-In 加速 | Token-Out 加速 | LLM 调用加速 |
|--------|--------------|---------------|----------------|-------------|
| AlfWorld | 0.531 (↑ vs 0.394) | 1.60× | 2.87× | 1.18× |
| ScienceWorld | 0.708 (≈ 0.716) | 1.30× | 1.41× | 1.31× |
| ToolQuery-Academic | 0.895 (≈ 0.901) | 1.15× | 0.92× | 1.20× |

### 主实验：效率提升（Reflexion + AutoTool）

| 数据集 | Progress Rate | Token-In 加速 | Token-Out 加速 | LLM 调用加速 |
|--------|--------------|---------------|----------------|-------------|
| AlfWorld | 0.453 (≈ 0.481) | 1.33× | 1.20× | 1.29× |
| ScienceWorld | 0.712 (≈ 0.730) | 0.93× | 1.20× | 1.28× |
| ToolQuery-Academic | 0.923 (↑ vs 0.917) | 1.33× | 1.19× | 1.26× |

### 开销分析

| 模块 | 耗时占比 |
|------|---------|
| 语义计算（SimCSE 嵌入 + 相似度） | 总任务时间的 2.7% ± 1.5% |
| 非语义模块（图构建/搜索/解析） | 秒级，可忽略不计 |

## 亮点与洞察

- **工具使用惯性的发现与量化**: 首次系统性地用信息论（条件熵）和统计检验验证工具调用的序列依赖性，为方法提供了坚实的理论基础
- **LLM 卸载思路新颖**: 不是优化 LLM 本身，而是识别出哪些决策可以不用 LLM，体现了"不是所有步骤都需要推理"的重要洞察
- **参数级数据流建模**: 不仅建模工具序列，还追踪参数在工具间的流动，实现自动参数填充
- **即插即用**: 可以直接增强 ReAct/Reflexion 等现有框架，无需微调

## 局限性

- 冷启动问题：在线构建图需要积累轨迹，初期惯性预测不够准确
- 惯性窗口固定为 2，未探索动态调整
- 仅在 3 个数据集上验证，未涉及更复杂的真实世界 API 调用场景
- 语义相似度模块（SimCSE）增加了额外的模型依赖
- 30% 的惯性调用上限较为保守，可能限制了进一步的效率提升

## 相关工作与启发

- 与 ToolNet、AnyTool 等工具选择方法相比，AutoTool 是唯一同时实现 LLM 卸载、惯性感知和参数流建模的方法
- 与 Agent Workflow Memory 思路类似（从历史交互中学习），但 AutoTool 专注于工具级别的统计模式
- 惯性图的思路可推广到其他需要重复决策的 agent 场景（如代码生成、数据分析 pipeline）

## 评分

- 新颖性: ⭐⭐⭐⭐ 工具使用惯性是一个新颖且有力的实证发现，图结构设计合理
- 实验充分度: ⭐⭐⭐⭐ 3 个数据集 + 多模型验证 + 详细的开销/敏感性分析，但数据集规模有限
- 写作质量: ⭐⭐⭐⭐ 动机清晰、理论分析严谨、图表丰富
- 价值: ⭐⭐⭐⭐ 为 agent 效率优化提供了新的思路，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](../../ACL2026/llm_agent/meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[AAAI 2026\] Time, Identity and Consciousness in Language Model Agents](time_identity_and_consciousness_in_language_model_agents.md)
- [\[ACL 2026\] Context-Value-Action Architecture for Value-Driven Large Language Model Agents](../../ACL2026/llm_agent/context-value-action_architecture_for_value-driven_large_language_model_agents.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](../../ICLR2026/llm_agent/toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[NeurIPS 2025\] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning](../../NeurIPS2025/llm_agent/zero-shot_large_language_model_agents_for_fully_automated_radiotherapy_treatment.md)

</div>

<!-- RELATED:END -->
