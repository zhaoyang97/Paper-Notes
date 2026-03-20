# Substance over Style: Evaluating Proactive Conversational Coaching Agents

**会议**: ACL 2025  
**arXiv**: [2503.19328](https://arxiv.org/abs/2503.19328)  
**代码**: 无  
**领域**: LLM Agent  
**关键词**: Conversational Coaching, Proactive Agent, User Study, Human Evaluation, Mixed-Initiative Dialogue  

## 一句话总结

通过健康教练领域的专家访谈和用户研究（31 名参与者、155 段对话），系统评估了五种不同对话风格（Directive、Interrogative、Facilitative）的 LLM 教练 Agent，发现用户高度重视核心功能性（substance）而对缺乏功能性时的风格修饰（style）持负面态度，同时揭示了用户第一人称评价与专家/LLM 第三方评价之间的显著不一致。

## 研究背景与动机

当前 NLP 研究在对话任务上取得了显著进展，但大多聚焦于具有以下特征的任务场景：单一明确目标、单一正确答案、单轮或短交互、可由第三方客观评估、清晰的交互结构。然而，教练式对话（coaching conversations）呈现出截然不同的挑战：

1. **开放式多轮交互**：没有预设的结束条件
2. **初始任务定义模糊**：目标需要通过多轮了解用户才能明确
3. **目标不断变化**：需要动态调整优先级
4. **可能偏题**：用户可能在长对话中偏离主题
5. **多样化偏好**：不同用户对对话风格有不同偏好
6. **混合主动性（mixed-initiative）**：教练需在满足用户目标和赋能用户之间取得平衡
7. **未言明的需求**：用户的真实需求可能未被直接表达
8. **无单一正确答案**：评估本质上是主观的

这些特征使得教练对话成为 proactive Agent 研究中最具挑战性的场景之一，且目前缺乏系统性的设计和评估框架。

## 方法详解

### 整体框架

研究包含三个阶段：

1. **健康专家访谈**（N=11）→ 提炼教练关键能力
2. **五种教练 Agent 设计与实现** → 基于不同对话范式组合
3. **用户研究**（N=31, 155 段对话）→ 多维度评估

### 关键设计

#### 专家洞察的分类：Style vs Substance

通过对 11 名健康教练（经验 4-46 年）的访谈，识别出六个关键洞察：

**Substance（核心功能）**：
- **I1 目标与目的理解**：理解用户的目标和动机，保持目标导向的对话
- **I2 上下文澄清**：收集用户的约束、偏好和过往尝试以个性化建议
- **I3 相关建议**：提供相关、可操作且上下文敏感的建议
- **I4 反馈寻求**：征求用户反馈并据此更新建议

**Style（风格）**：
- **I5 积极倾听**：偶尔反述以确保正确理解和目标对齐
- **I6 用户赋能**：建立信任并引导用户自己探索解决方案

#### 三种对话范式

1. **Interrogative（审讯式）**：一方持续提问，另一方仅回答；最大化信息获取但最小化互动
2. **Directive（指令式）**：LLM 主动提供持续的解决方案和指示（如 ChatGPT 的默认行为）
3. **Facilitative（引导式）**：引导用户自己找到解决方案，而非直接给答案——这是人类教练推荐但 LLM 不自然具备的模式

#### 五种 Agent 设计

基于两个维度的组合：

**Coaching Expertise Variations**：
- **Base Module**：良好意图但无正式训练的教练，包含主动提问指南但不定义具体话题
- **Expert Module**：有经验的教练，明确定义目标导向的提问线路（先目标→约束→偏好→障碍→建议），强调激励行为、积极倾听和用户赋能

**Conversation Flow Variations**：
- **Probing Module**：在用户表述模糊或不确定时进一步澄清
- **Recommendation Module**：决定何时提出建议并寻求反馈
- **Resolution Module**：决定对话何时达到合理结论

最终五种 Agent：
1. **Base-Interrogative**：Base + Interrogative flow（强调 I2，弱化 I1/I3/I4）
2. **Expert-Interrogative**：Expert + Interrogative flow
3. **Directive**：简单 prompt 实现的指令式（LLM 默认的推荐优先模式）
4. **Base-Facilitative**：Base + Facilitative flow（强调 I1/I2/I4，中等 I3）
5. **Expert-Facilitative**：Expert + Facilitative flow

#### 显式对话流控制

为实现非 Directive 的对话模式，引入 **Explicit Conversation Flow**——每轮 Agent 回复前运行一系列 LLM 推理链：

- **第一次推理**：在用户发言后输出二值决策（是否需要探测/建议/结束）
- **第二次推理**：根据上一步决策生成具体的 Agent 回复
- 多模块的第一次推理并行运行，当多个模块返回正决策时，优先对话延续（提问优先于建议）

所有 Agent 使用 Gemini 1.5 Pro 作为基础 LM。

### 损失函数 / 训练策略

本文不涉及模型训练。所有 Agent 基于 prompt engineering 和多级 LM 推理链实现，核心创新在于对话控制流的设计而非模型参数优化。

## 实验关键数据

### 主实验

**用户研究设置**：
- 31 名参与者，每人 1.5 小时
- 每人与 5 个 Agent 各进行一次对话
- 33 个开放式健康场景（睡眠、健身、日常习惯等）
- balanced Latin square 顺序设计

**总体排名（Win Rate = Top 1 + Top 2 排名占比）**：

| Agent | Win Rate | Top 1 占比 |
|-------|----------|-----------|
| Expert-Facilitative | **61.29%** | **41.9%** |
| Base-Facilitative | 58.06% | 25.8% |
| Directive | 41.94% | 22.6% |
| Expert-Interrogative | 35.48% | 6.5% |
| Base-Interrogative | 3.22% | 3.2% |

**Substance 维度 Win Rate**：

| Agent | Purpose | Context | Rec. | Personalized | Feedback | Avg. |
|-------|---------|---------|------|-------------|----------|------|
| Expert-Facilitative | 61.29 | 67.74 | 67.74 | 67.74 | 64.52 | **65.81** |
| Base-Facilitative | 51.61 | 54.84 | 51.61 | 45.16 | 54.84 | 51.61 |
| Directive | 48.39 | 41.94 | 41.94 | 45.16 | 41.94 | 43.87 |
| Expert-Interrogative | 6.45 | 6.45 | 3.23 | 9.68 | 9.68 | 7.10 |

**Style 维度 Win Rate**：

| Agent | Length | Concise | Tone | Encourage | Credibility | Empathy | Avg. |
|-------|--------|---------|------|-----------|-------------|---------|------|
| Base-Facilitative | 54.84 | 54.84 | 51.61 | 61.29 | 61.29 | 61.29 | **57.36** |
| Expert-Facilitative | 35.48 | 29.03 | 48.39 | 41.94 | 48.39 | 61.29 | 44.42 |
| Directive | 51.61 | 45.16 | 38.71 | 51.61 | 45.16 | — | — |

### 关键发现

1. **Substance > Style**：用户对核心功能性的重视远超对话风格。Expert-Facilitative 在 substance 维度遥遥领先，而 style 维度上 Base-Facilitative 反而略胜——说明用户对"有实质内容但风格一般"的教练远比"风格好但缺乏实质"的教练满意。

2. **过度提问的负面效果**：Interrogative Agent 表现最差。用户反馈："The agent initially asked a lot of open ended questions… in the end responded with a suggestion. This made me feel the conversation was one sided." —P37。过多提问让用户觉得 Agent 偏离了原始目标，降低了参与度。

3. **强制结束 vs 自然结束**：Interrogative Agent 的对话 58%-71% 的时间被强制结束，而 Facilitative Agent 仅 16%-19%。这说明纯审讯式对话在开放式场景中用户体验极差。

4. **LLM 经验偏差**：有丰富 LLM 使用经验的用户中 29.2% 首选 Directive Agent，而经验较少的用户中 0% 选择 Directive——说明 LLM 老用户对指令式回复有熟悉度偏好。

5. **评价不一致性**：用户第一人称评价、健康专家第三方评价和 LLM 自动评价之间存在显著差异。LLM 在主观的人类中心任务上作为自动评估器表现不佳，但在客观任务上与用户评价合理一致。

## 亮点与洞察

1. **"Substance over Style" 的核心发现**：标题即洞察——在缺乏核心功能时，风格修饰不仅无益反而有害。这对 LLM Agent 的设计有深远启示：应先确保功能完备再追求交互优美。
2. **Facilitative 范式的引入**：识别出第三种对话范式——引导式交互，填补了 Directive 和 Interrogative 之间的空白，且最符合人类教练的专业实践。
3. **显式对话流控制**：通过多级 LLM 推理链实现复杂的对话控制逻辑，解决了单一 prompt 难以做出"何时停止提问/何时给建议"等控制流决策的问题。
4. **评估方法论贡献**：系统对比第一人称（用户）、第三方（专家）和自动（LLM）三种评估方式，揭示其不一致性——警示研究者不能简单依赖 LLM 评估替代用户研究。

## 局限性

1. **样本量有限**：31 名参与者可能不足以捕捉人口统计学上的细微差异。
2. **健康领域专属**：结论可能不能直接推广到其他教练领域（如职业发展、学习辅导）。
3. **单次交互**：每个 Agent 只与每个用户进行一次对话，无法评估长期教练关系的建立。
4. **性别偏差**：参与者中 81.7% 为男性，可能影响对话风格偏好的结论。
5. **Gemini 1.5 Pro 特定**：结果可能受基础 LM 能力的影响，其他 LM 可能表现不同。

## 相关工作与启发

- **Proactive Dialogue**：Li et al. (2024) 的主动信息搜索、Deng et al. (2024) 的个性化目标导向对话，本文扩展到复杂的教练场景。
- **Mixed-Initiative 交互**：传统上分为用户主导和系统主导，本文的 Facilitative 模式代表了一种更平衡的交互范式。
- **Coaching 研究**：Schwarz and Davidson (2008) 的引导式教练理论提供了理论基础。
- **LLM 评估**：大量工作探索 LLM-as-judge，本文发现其在主观评估任务上的局限性。
- **启发**：该研究框架（专家访谈→Agent 设计→用户研究→多维评估）可作为评估其他人类中心 AI 应用的范式参考。对于 LLM Agent 的产品化，用户研究不可或缺，不能仅靠自动评估。

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 3.5 |
| 实验充分性 | 4.5 |
| 实用价值 | 4.5 |
| 写作质量 | 4.5 |
| 总体评分 | 4.2 |
