# AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks

**会议**: NeurIPS 2025  
**arXiv**: [2508.00890](https://arxiv.org/abs/2508.00890)  
**代码**: 未提及  
**领域**: LLM Agent  
**关键词**: test-time scaling, compute-optimal, multi-stage tasks, LLM agent, budget allocation, model selection  

## 一句话总结
提出 AgentTTS，一个用 LLM agent 自动搜索多阶段复杂任务中**测试时计算最优缩放策略**（模型选择+预算分配）的框架，通过迭代反馈驱动的交互显著提升搜索效率和性能。

## 背景与动机
测试时缩放（Test-time Scaling, TTS）通过在推理时分配更多计算资源来提升 LLM 性能（如 o1 的 thinking/chain-of-thought）。但现有 TTS 研究主要针对**单阶段任务**（如单个推理问题）。现实世界中的许多任务是**多阶段复杂任务**——由一系列异构子任务组成，每个子任务需要不同能力的 LLM。

例如，一个 RAG 问答任务包含：检索 → 重排序 → 推理 → 生成四个阶段，每个阶段最适合的模型和最优的计算预算可能不同。

两个核心挑战：
1. **组合爆炸**：模型选择 × 预算分配 × 子任务数的搜索空间巨大，暴力搜索不可行
2. **子任务间依赖**：一个子任务的模型/预算选择会影响后续子任务的最优配置

## 核心问题
如何在多阶段复杂任务中**自动搜索**每个子任务的最优模型和计算预算，使整体性能最大化且搜索成本可控？

## 方法详解

### 整体框架
AgentTTS 是一个 LLM-agent-based 搜索框架：
1. LLM agent 分析任务结构，提出候选模型+预算配置
2. 在执行环境中运行候选配置
3. 收集反馈（性能、成本、错误模式）
4. Agent 根据反馈迭代优化搜索方向

### 关键设计
1. **经验性洞察驱动**: 通过在 4 个任务 6 个数据集上的大量实验，提炼出三个关键经验洞察：
   - 不同子任务的最优模型和预算差异显著
   - 子任务间存在性能耦合——某些子任务用强模型可以弥补其他子任务的不足
   - 计算预算的边际效用在不同子任务上不同

2. **LLM Agent 搜索**: 不是用传统优化算法（如贝叶斯优化、网格搜索），而是让 LLM agent 理解任务结构和反馈信号，利用其推理能力指导搜索。Agent 能理解"为什么这个配置不好"并有方向性地探索。

3. **迭代反馈驱动**: 每轮搜索后，agent 收到：性能指标、计算成本、子任务级错误分析，据此调整下一轮搜索策略。

### 训练策略
无训练——完全基于推理时的搜索。Agent 使用 prompt engineering 驱动，不需要额外微调。

## 实验关键数据
- 在四个多阶段任务（包括 RAG、multi-hop QA 等）的六个数据集上评估
- AgentTTS 在搜索效率上显著优于传统搜索方法和其他 LLM-based baseline
- 对训练集大小更鲁棒
- 搜索过程具有可解释性——agent 的推理轨迹可以解释为什么选择了某个配置

### 消融实验要点
- Agent 搜索 vs 随机搜索/网格搜索/贝叶斯优化：Agent 搜索效率更高
- 反馈信息的重要性：移除子任务级反馈后性能显著下降
- 不同 LLM 作为 agent 的效果比较

## 亮点
- 首次研究多阶段复杂任务的 test-time compute-optimal scaling
- LLM agent 做搜索是一个优雅的元学习/元优化思路
- 三个经验洞察为社区提供了有价值的设计指南
- 可解释性——agent 的搜索理由可以被理解和验证

## 局限性 / 可改进方向
- 搜索本身需要调用 LLM，有一定计算开销
- 动态任务（子任务数/类型变化）下的适应性待验证
- 经验洞察可能与任务类型相关，泛化性需更多验证
- 未考虑延迟约束下的 compute-optimal 策略

## 与相关工作的对比
- **vs 传统 TTS (o1/Best-of-N)**: 传统 TTS 对单阶段任务做 scaling；AgentTTS 处理多阶段的组合优化
- **vs AutoML/NAS**: AutoML 搜索模型架构；AgentTTS 搜索推理时的资源分配策略
- **vs AdaVideoRAG（同系列笔记）**: AdaVideoRAG 也做自适应路由但基于规则；AgentTTS 用 agent 做更灵活的搜索

## 启发与关联
- "用 LLM agent 优化 LLM 推理策略"是一个有趣的递归思路——meta-reasoning
- 多阶段预算分配的框架可用于 VLM pipeline（视觉编码→投影→LLM推理→生成）的资源优化
- 与 PrefixKV 的关联：PrefixKV 在层级分配 KV 缓存预算，AgentTTS 在子任务级分配计算预算——都是"自适应资源分配"

## 评分
- 新颖性: ⭐⭐⭐⭐ 多阶段 TTS 问题定义新颖，agent 搜索思路有趣
- 实验充分度: ⭐⭐⭐⭐ 4 任务 6 数据集，多基线对比
- 写作质量: ⭐⭐⭐⭐ 经验洞察清晰，问题定义明确
- 价值: ⭐⭐⭐⭐ 对复杂 agent pipeline 的资源优化有实际意义
