# Talk, Evaluate, Diagnose: User-aware Agent Evaluation with Automated Error Analysis

**会议**: ICLR2026  
**arXiv**: [2603.15483](https://arxiv.org/abs/2603.15483)  
**代码**: [GitHub](https://github.com/SAP-samples/agent-quality-inspect)  
**领域**: llm_agent / evaluation  
**关键词**: Agent评估, 用户感知, LLM-as-judge, 错误诊断, 会话效率  

## 一句话总结
提出 TED 框架（Talk-Evaluate-Diagnose），通过可复用的专家/非专家 persona 模板、基于 grading notes 的 LLM-as-judge 评估和自动化错误分析，实现跨领域的用户感知型 Agent 评估。

## 背景与动机
1. Agent 应用跨越多领域（航空订票、消息、提醒等），异构评估方法难以统一
2. 现有评估各自为政：数据库查询、正则匹配、工具签名检查等，缺乏通用框架
3. 大多数评估忽略用户角色：用户专业度影响 Agent 表现，但未被系统考量
4. 动态对话模拟中，用户 persona 与任务指令紧耦合导致难以隔离影响因素
5. 现有指标（success rate）过于粗粒度，无法区分「快速完成」vs「缓慢完成」
6. 缺乏系统的错误分析——大多数工作止步于报告最终指标

## 方法详解
**Talk 阶段**：
- 解耦用户 persona 与任务指令：persona 模板（expert / non-expert）+ 任务指令独立组合
- 可复用的通用 persona 模板，跨领域适用（航空、零售、ToolSandbox）

**Evaluate 阶段**：
- **Grading Notes**：将子目标（工具调用、响应内容）统一表示为自然语言检查清单
- **LLM-as-judge**：用 gpt-4.1 评判每个子目标是否达成，多次运行取多数票
- **新指标体系**：
  - MaxProgressRate@k：k 次试验中最佳进度（细粒度替代 pass@k）
  - MaxAUC@k：进度曲线下面积，奖励早期进展
  - MaxPPT@k：每轮进度提升，对顺序不敏感

**Diagnose 阶段**：
- 绘制每个样本的进度期望和方差，分离 judge 不一致性和 Agent 不一致性
- 两步自动错误发现：(1) 低层错误识别 → (2) 语义聚类为高层类别
- 错误反馈可注入 Agent 设计，实现 8-10% 性能提升

## 实验关键数据
| 模型 | τ²-bench Easy MaxProg@k (Expert\|Non-expert) | ToolSandbox MaxProg@k |
|------|------|------|
| gpt-4.1 | 1.00 \| 1.00 | 0.98 \| 0.97 |
| gpt-4o | 1.00 \| 1.00 | 0.99 \| 1.00 |
| gpt-4o-mini | 0.90 \| 0.90 | 0.95 \| 0.93 |
| gpt-5 | 1.00 \| 1.00 | 0.97 \| 0.91 |
| mistral-nemo | 1.00 \| 0.80 | 0.92 \| 0.96 |

- Expert vs Non-expert：MaxAUC@k 一致显示非专家场景效率更低
- 错误修复后 Agent 峰值提升 8-10%
- AUC 和 PPT 可区分 pass@k 无法区分的模型

## 亮点
- **通用评估框架**：grading notes 抽象统一不同领域的评估，无需访问系统状态
- **用户感知维度**：首次系统分离用户 persona 对 Agent 表现的影响
- **AUC + PPT 指标组合**：区分"早进步"和"均匀进步"两种对话效率模式
- 自动错误发现闭环：评估 → 诊断 → 改进

## 局限性 / 可改进方向
- 仅有 expert/non-expert 二分类 persona，缺乏更细粒度用户建模
- grading notes 仍需人工或半自动创建，标注成本非零
- LLM-as-judge 自身存在不一致性，多数票缓解但未根除
- 实验限于任务型对话场景，未涵盖开放域或创造性任务

## 与相关工作的对比
- vs τ²-bench 原始评估：TED 揭示 pass@k 饱和时 AUC/PPT 仍能区分模型
- vs AgentBoard：从 agent-environment 扩展到多轮对话 + 用户感知
- vs MINT：不仅衡量最终成功，还捕获每轮进度和效率

## 评分
- 新颖性: ⭐⭐⭐⭐ 用户感知 + 会话效率指标组合为评估带来新维度
- 实验充分度: ⭐⭐⭐⭐ 多模型、多数据集、多指标对比
- 写作质量: ⭐⭐⭐⭐ 数学定义严谨，框架层次清晰
- 价值: ⭐⭐⭐⭐ 对 Agent 评估社区有实用价值
