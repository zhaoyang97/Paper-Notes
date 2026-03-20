# ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering

**会议**: NeurIPS 2025 (Datasets & Benchmarks Track)  
**arXiv**: [2506.09050](https://arxiv.org/abs/2506.09050)  
**代码**: [https://github.com/SakanaAI/ALE-Bench](https://github.com/SakanaAI/ALE-Bench)  
**领域**: LLM评估 / 算法工程 / Agent基准  
**关键词**: 算法工程基准, 长时间跨度推理, 分数制编程竞赛, 迭代优化, LLM Agent  

## 一句话总结
提出 ALE-Bench，首个面向分数制算法工程竞赛（AtCoder Heuristic Contest）的 AI 基准，评估 LLM 和 Agent 在 NP-hard 优化问题上的长时间迭代改进能力，发现当前最强模型（o3-high）仅达人类平均水平，且在问题一致性和长时间改进方面与人类专家差距显著。

## 背景与动机
现有编程基准（HumanEval、CodeContests、LiveCodeBench）聚焦于短时间、通过/不通过的精确解题，LLM 已接近饱和。但现实中更多优化问题（物流路径、排产调度、电网平衡）是 NP-hard 的，没有精确解，需要通过长时间迭代改进逐步提高分数——这正是 AtCoder Heuristic Contest (AHC) 的模式。AHC 每场吸引约 1000 名参与者，人类选手通常花数周迭代优化，而 AI 在这类长时间推理任务上的能力尚未被系统评估。

## 核心问题
如何系统评估 AI 系统在**长时间跨度、分数驱动的算法工程任务**上的能力？现有基准都是短时间单次提交模式，无法衡量 AI 的**迭代改进**和**跨问题一致性**。

## 方法详解

### 整体框架
ALE-Bench 收集了 40 道 AHC 赛题（覆盖路径规划、排产、拼图、贝叶斯推断等多种类型），提供：1）Markdown 格式题目描述；2）Rust 评分器；3）可视化工具（Web + 命令行）；4）排行榜数据。AI 系统通过 Session 接口参与：读题 → 测试运行（获得反馈分数）→ 可视化 → 最终提交，整个过程模拟真实比赛体验。

### 关键设计
1. **Session 交互系统**：AI 在时间限制内迭代提交，每次获得公开测试集反馈分数。支持四种操作：查看题目、测试运行（含自定义输入）、可视化、最终提交。Docker 沙箱确保执行与 AtCoder 环境一致。

2. **ALE-Agent（专用 Agent）**：两个核心技术——(a) 领域知识注入：将模拟退火、beam search 等优化算法知识嵌入 prompt；(b) 多样性导向搜索：基于 best-first search 的树搜索策略，每次从最优节点扩展 k=30 个子节点，利用 LLM 并行生成。

3. **评估体系**：采用 AHC 的 Performance 指标（类 Elo rating），支持按问题评分和跨问题汇总。特别提出了"average performance"作为推荐指标（rating 会高估 AI 能力）。

### 损失函数 / 训练策略
无训练——这是评估基准。但 ALE-Agent 使用带反馈的迭代 prompt 策略：历史摘要 + 当前最优代码 + 改进方向引导。

## 实验关键数据
| 模型/设置 | Avg Perf | Rating | Rank%(越低越强) | 备注 |
|-----------|----------|--------|---------|------|
| o3-high (one-shot) | 1044 | 1456 | 43.2% | 推理模型最强 |
| Claude 3.7 Sonnet (one-shot) | 833 | 1197 | 63.2% | 非推理最强 |
| o4-mini (迭代4h) | 1520 | 2104 | 11.8% | 迭代设置最强 |
| ALE-Agent+M1&2 (lite) | 1879 | 2222 | 8.6% | Agent最强 |
| 人类平均 | 1260 | 1414 | — | 基准线 |

### 消融实验要点
- 迭代优化 vs one-shot：所有模型在迭代设置下 average performance 提升 400+ 点
- ALE-Agent 消融：领域知识(M1)小幅提升，多样性搜索(M2)大幅提升（1264→1879）
- 长时间改进 vs 暴力搜索：150 次独立 one-shot 的最高分远低于迭代改进，证明基准确实测量了长时间推理而非并行探索
- 无污染：跨知识截止日期分析未发现性能突变

## 亮点
- 填补空白：首个分数制算法工程基准，与人类直接可比
- 开放且可持续：随 AHC 持续更新，分数无上限，即使 AI 超过人类也有区分度
- 真实洞察：揭示 AI 在模拟退火类问题上较强，但在需要问题特定洞察的规划类问题上较弱
- ALE-Agent 实战参与 AHC046 获得第 154 名（performance 1915），验证了基准的真实性

## 局限性 / 可改进方向
- 数据集规模有限（40题），但每题允许长时间迭代，方差较小
- Rating 系统倾向高估 AI（个别高分题会大幅拉高），推荐用 average performance
- 未充分利用可视化和多模态能力——AI 主要用文本反馈，未像人类那样利用可视化调试
- 未来可生成合成题目扩展数据集，或训练 RL 策略

## 与相关工作的对比
与 **SWE-Bench** 对比：SWE-Bench 是通过/不通过的代码修复任务，ALE-Bench 是分数制优化任务，允许无限提升。两者评估维度正交——SWE-Bench 测代码理解和修复，ALE-Bench 测算法设计和迭代改进。

与 **MLE-Bench** 对比：MLE-Bench 来自 Kaggle ML 竞赛，侧重数据科学技能且需要 GPU。ALE-Bench 侧重算法工程，仅需 CPU，评估成本更低。

与 **FunSearch** 对比：FunSearch 用 LLM 改进人类模板代码解决优化问题，ALE-Bench 要求 AI 从头设计和迭代改进完整解决方案。

## 启发与关联
- ALE-Agent 的树搜索策略（beam=30 并行生成）是一种有趣的 inference-time scaling 方法
- 迭代改进能力是未来 Agent 系统的核心需求，ALE-Bench 提供了标准化评估平台
- 领域知识注入（模拟退火 prompt）显著提升效果，暗示 LLM Agent 需要结构化的领域知识

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个分数制长时间算法工程基准
- 实验充分度: ⭐⭐⭐⭐⭐ 22个模型×3种语言×3种设置，分析深入
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但附录偏长
- 价值: ⭐⭐⭐⭐⭐ AI能力评估的重要新维度
