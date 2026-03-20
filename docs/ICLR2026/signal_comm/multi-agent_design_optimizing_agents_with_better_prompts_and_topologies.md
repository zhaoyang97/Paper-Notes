# Multi-Agent Design: Optimizing Agents with Better Prompts and Topologies

**会议**: ICLR 2026  
**arXiv**: [2502.02533](https://arxiv.org/abs/2502.02533)  
**代码**: 待确认  
**领域**: Agent  
**关键词**: multi-agent system, prompt optimization, topology search, automated MAS design, workflow optimization  

## 一句话总结
深入分析多智能体系统中 prompt 和拓扑设计的影响，发现 prompt 优化是最关键的设计因素（仅优化 prompt 的单 Agent 即可超越复杂多 Agent 拓扑），提出 Mass 三阶段框架（block-level prompt → topology → workflow-level prompt）在 8 个 benchmark 上取得 SOTA。

## 研究背景与动机
1. **领域现状**：多智能体系统（MAS）通过 Debate、Reflect、Aggregate 等拓扑组织多个 LLM Agent 协作。近期出现自动化 MAS 设计方法（如 ADAS、AFlow）。
2. **现有痛点**：不清楚 MAS 性能提升究竟来自"多 Agent 拓扑"还是"更好的 prompt"。许多复杂拓扑反而降低性能，但原因不明。
3. **核心矛盾**：增加 Agent 和拓扑复杂度的收益不确定——有时有帮助，有时反而有害。
4. **本文要解决**：① 量化 prompt vs 拓扑的贡献；② 设计统一的自动化框架同时优化两者。
5. **切入角度**：控制变量分析——先只优化 prompt 看效果，再叠加拓扑搜索。
6. **核心idea**：Prompt 优化 >> 拓扑选择；但两者联合优化 > 任何单独优化。

## 方法详解

### 整体框架
Mass 三阶段交替优化：① Block-level prompt optimization（对每个 agent 模块独立优化 instruction + exemplar）→ ② Workflow topology optimization（基于模块增量影响力剪枝搜索空间）→ ③ Workflow-level prompt optimization（在最优拓扑上全局联合优化）。

### 关键设计

1. **Block-level Prompt 优化（热身阶段）**:
   - 对每个 agent 模块独立进行 instruction + exemplar 联合优化
   - 使用验证集反馈迭代改进
   - 作为拓扑搜索的"预训练"——确保每个模块的 prompt 质量
   - 设计动机：实验发现仅 prompt 优化的单 Agent 已超越 SC/Reflect/Debate 等复杂拓扑

2. **Workflow Topology 优化**:
   - 计算每个模块的增量影响力 $I_{a_i}$
   - 基于 softmax 概率采样剪枝搜索空间
   - 评估候选拓扑时使用第一阶段优化好的 prompt
   - 发现：不是所有拓扑都有正面影响（如 HotpotQA 上仅 debate 带来 3% 增益）

3. **Workflow-level Prompt 优化**:
   - 在最优拓扑确定后，全局联合优化所有 agent 的 prompt
   - 考虑 agent 间的交互效应（拓扑中的信息流如何影响 prompt 设计）
   - 细粒度调整以适配最终拓扑

## 实验关键数据

### 主实验（8 benchmark）
| 方法 | MATH | HotpotQA | MMLU | 平均 |
|------|------|----------|------|------|
| SC (Self-Consistency) | 基线 | 基线 | 基线 | 基线 |
| Reflect | 略高 | 略低 | 略高 | 混合 |
| Debate | 略高 | +3% | 略低 | 混合 |
| ADAS | 高 | 高 | 高 | 强基线 |
| **Mass** | **最高** | **最高** | **最高** | **SOTA** |

### 关键消融
| 对比 | 结论 |
|------|------|
| 单 Agent + prompt 优化 vs 多 Agent (无 prompt 优化) | **单 Agent 更优** |
| Mass (3阶段) vs 仅 prompt vs 仅 topology | Mass **显著最优** |
| Gemini 1.5 Pro → Claude 3.5 Sonnet 迁移 | **结论可迁移** |

### 关键发现
- **仅 prompt 优化的单 Agent 已超越 SC/Reflect/Debate**——挑战了"多Agent必然更好"的直觉
- Mass 在所有 8 个 benchmark 上均为 SOTA，显著超越 ADAS、AFlow 等自动化基线
- 并非所有拓扑都有正面影响——约 50% 情况下额外拓扑反而降低性能
- 结论跨模型可迁移（Gemini → Claude → Mistral）

## 亮点与洞察
- **"Prompt > Topology"**是核心发现——对 MAS 社区有重要校正作用：在追求复杂拓扑前先优化 prompt
- **三阶段交替优化**的设计合理——先热身再搜索再微调，避免冷启动
- **增量影响力剪枝**有效缩小了拓扑搜索空间
- 可推广到任何 MAS 框架——Mass 的方法论不绑定特定拓扑类型

## 局限性 / 可改进方向
- 搜索空间依赖预定义的构建模块（Aggregate/Reflect/Debate），缺乏任意结构发现能力
- 优化过程需要验证集反馈，计算成本随 agent 数量增长
- 拓扑构建规则是固定顺序，限制了非常规拓扑的发现
- 未考虑推理时的动态拓扑选择（根据输入难度选择不同拓扑）

## 相关工作与启发
- **vs ADAS**: ADAS 搜索 agent 架构，Mass 联合优化 prompt+topology
- **vs AFlow**: AFlow 用代码生成搜索工作流，Mass 用验证集反馈优化
- **vs DSPy**: DSPy 优化 prompt pipeline，Mass 同时优化拓扑
- 启发：MAS 效果的瓶颈可能不在拓扑而在每个 agent 的 prompt 质量

## 补充讨论

### 为什么复杂拓扑常常无效？
多个 Agent 协作引入了额外的「通信开销」——每个 Agent 的输出可能包含噪声或无关信息，堆积后反而干扰最终决策。Debate 拓扑在 HotpotQA 上有效（因为多角度讨论有助于事实核查），但在数学任务上反而降低性能（因为数学推导是确定性的， debate 引入变异）。这说明拓扑选择应该是任务相关的，而非“一套拓扑适用所有任务”。

### 三阶段优化的必要性

实验证明，单独做 prompt 优化或单独做 topology 搜索都不如三阶段联合优化。

关键原因是 prompt 和 topology 之间存在交互作用：最优的 prompt 在不同拓扑下可能不同，而最优拓扑也依赖于 prompt 的质量。这说明 MAS 设计是一个联合优化问题，不能拆解为独立的子问题。

## 评分
- 新颖性: ⭐⭐⭐⭐ "Prompt > Topology"的发现有价值，三阶段设计合理
- 实验充分度: ⭐⭐⭐⭐⭐ 8 benchmark，跨模型验证，消融全面
- 写作质量: ⭐⭐⭐⭐ 控制变量分析清晰
- 价值: ⭐⭐⭐⭐ 对 MAS 设计实践有直接指导意义
