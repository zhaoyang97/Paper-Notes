---
title: >-
  [论文解读] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making
description: >-
  [ICML 2025][强化学习][反事实推理] 提出一种双层因果分解框架，将多智能体序列决策中某动作的总反事实效应（TCFE）系统地分解为"通过智能体行为传播的效应"（tot-ASE）和"通过状态转移传播的效应"（r-SSE），并分别用 Shapley 值和内在因果贡献（ICC）进一步归因到每个智能体和每个状态变量。
tags:
  - "ICML 2025"
  - "强化学习"
  - "反事实推理"
  - "因果解释"
  - "多智能体MDP"
  - "Shapley值"
  - "可解释性"
---

# Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making

**会议**: ICML 2025  
**arXiv**: [2410.12539](https://arxiv.org/abs/2410.12539)  
**代码**: [GitHub](https://github.com/stelios30/cf-effect-decomposition)  
**领域**: 强化学习  
**关键词**: 反事实推理, 因果解释, 多智能体MDP, Shapley值, 可解释性

## 一句话总结

提出一种双层因果分解框架，将多智能体序列决策中某动作的总反事实效应（TCFE）系统地分解为"通过智能体行为传播的效应"（tot-ASE）和"通过状态转移传播的效应"（r-SSE），并分别用 Shapley 值和内在因果贡献（ICC）进一步归因到每个智能体和每个状态变量。

## 研究背景与动机

在多智能体序列决策场景中（如人-AI协作医疗决策），反事实推理是追溯性分析决策影响的核心工具。给定一条实际轨迹，我们想知道"如果在某时刻采取了不同的动作，结果会如何改变"——这就是**总反事实效应（TCFE）**。

然而 TCFE 本身只是一个标量，无法解释效应**为什么**以及**如何**产生：
- 效应是通过改变了后续其他智能体的行为而传播的？
- 还是通过改变了环境状态转移而传播的？
- 哪个智能体对效应贡献最大？
- 哪些状态变量最关键？

传统因果中介分析（mediation analysis）通过枚举因果路径来分解效应，但在多智能体 MDP 中，动作到结果之间的因果路径呈指数级增长，且很多路径缺乏直观的操作含义。因此需要一种与 MMDP 结构天然匹配的分解方法。

## 方法详解

### 整体框架

本文的核心是一个**双层分解框架（Bi-level Decomposition）**：

**Level 1 — 因果解释公式（Causal Explanation Formula）**：将 TCFE 拆分为两个有明确物理含义的分量：

$$\text{TCFE} = \text{tot-ASE} - \text{r-SSE}$$

- **tot-ASE（总智能体特定效应）**：效应仅通过所有后续智能体的行为变化传播（状态转移保持"原始"机制）
- **r-SSE（反向状态特定效应）**：假设所有智能体已经按反事实行动，但状态转移未受干预影响时"丢失或增加"的效应

这个公式是 Pearl (2001) 经典因果中介公式在多智能体 MDP 上的推广。

**Level 2a — ASE-SV**：用 Shapley 值将 tot-ASE 进一步归因到每个智能体。

**Level 2b — r-SSE-ICC**：用内在因果贡献将 r-SSE 进一步归因到每个状态变量。

### 关键设计

#### 形式化基础：MMDP-SCM

将多智能体 MDP 与结构因果模型（SCM）结合，构建 MMDP-SCM $\langle \mathbf{V}, \mathbf{U}, P(\mathbf{u}), \mathcal{F} \rangle$：

- $\mathbf{V}$：观测变量（所有状态和动作变量）
- $\mathbf{U}$：互相独立的噪声变量（捕获随机性）
- $\mathcal{F}$：结构方程，状态转移 $S_t := f^S(S_{t-1}, \mathbf{A}_{t-1}, U^{S_t})$，智能体策略 $A_{i,t} := f^{A_i}(S_t, U^{A_{i,t}})$

每个噪声实例 $\mathbf{u}$ 唯一确定一条轨迹 $\tau$，反事实通过对结构方程的干预（intervention）来定义。

#### Level 1：tot-ASE 与 r-SSE 的分解

**tot-ASE** 通过"自然干预"（natural intervention）定义：对所有后续智能体的动作施加自然干预，让它们取在反事实世界中会自然采取的动作，测量结果差异。

**r-SSE** 通过"反向"状态特定效应定义：假设所有智能体已按反事实行动，但状态 $S_{t+1}$ 未受干预影响，测量"丢失"的效应。

关键定理（Theorem 3.3）证明了 $\text{TCFE} = \text{tot-ASE} - \text{r-SSE}$ 恒成立。注意：直觉上可能认为 TCFE = tot-ASE + SSE，但作者通过实验证明这并不总是成立，正确的分解需要用 r-SSE 而非 SSE。

#### Level 2a：ASE-SV（Shapley 值分解 tot-ASE）

利用智能体特定效应（ASE）的概念——$\mathbf{N}$-specific effect 衡量干预效应仅通过子集 $\mathbf{N}$ 中智能体传播的部分。以此构造合作博弈，用 Shapley 值计算每个智能体的贡献分数：

$$\phi_j = \sum_{S \subseteq \{1,...,n\} \setminus \{j\}} w_S \cdot [\text{ASE}^{S \cup \{j\}} - \text{ASE}^S]$$

**公理化保证**（Theorem 5.3）：ASE-SV 是唯一同时满足以下四个性质的归因方法：
1. **效率性（Efficiency）**：所有智能体贡献之和等于 tot-ASE
2. **不变性（Invariance）**：不贡献的智能体得分为零
3. **对称性（Symmetry）**：等贡献的智能体得分相同
4. **贡献单调性（Contribution Monotonicity）**：得分仅依赖边际贡献且单调

#### Level 2b：r-SSE-ICC（内在因果贡献分解 r-SSE）

利用 Janzing et al. (2024) 的内在因果贡献（ICC）概念，量化每个状态变量对 r-SSE 不确定性的减少程度：

$$\text{ICC}(S_k \to \Delta Y | \tau) = \text{Unc}^{<S_k} - \text{Unc}^{\leq S_k}$$

其中 $\text{Unc}$ 是条件方差的期望。直观含义：如果已知状态 $S_k$ 的反事实值能显著降低对 r-SSE 估计的不确定性，则 $S_k$ 对 r-SSE 的贡献大。

最终归因分数按 ICC 的相对比例分配 r-SSE（Theorem 4.2 保证了效率性）。

### 损失函数 / 训练策略

本文属于**因果推断分析框架**而非训练新模型，不涉及传统的损失函数。核心计算流程是：

1. **Abduction（溯因）**：给定观测轨迹 $\tau$，从后验分布 $P(\mathbf{u}|\tau)$ 采样噪声
2. **Action（干预）**：在 MMDP-SCM 中施加目标干预
3. **Prediction（预测）**：前向推演获得反事实结果

实验中使用 100 个后验样本估计反事实效应，20 个额外样本用于条件方差估计。环境中的策略网络通过深度 RL 训练（Gridworld 实验），但这不是本文的核心贡献。

## 实验关键数据

### 主实验

作者在两个环境上验证方法：

| 环境 | 任务描述 | 智能体数 | 时间步长 | 主要验证 |
|------|---------|---------|---------|---------|
| Gridworld（LLM辅助） | 两个 Actor + LLM Planner 完成物品配送 | 3（2 Actor + 1 Planner） | ~20步 | 分解公式正确性、ASE-SV 可解释性、r-SSE-ICC 精准归因 |
| Sepsis 模拟器 | 临床医生 + AI 联合治疗 ICU 患者 | 2（临床医生 + AI） | 20轮 | ASE-SV 随信任度变化的合理性、r-SSE-ICC 稀疏性 |

**Gridworld 关键结果**：
- 验证了 TCFE ≠ tot-ASE + SSE，但 TCFE = tot-ASE - r-SSE 成立（Theorem 3.3）
- ASE-SV 将 tot-ASE 全部归因于 Actor 2（施加干预的对象），Actor 1 和 Planner 得分为零——符合预期
- r-SSE-ICC 精确定位到 4 个状态变量有显著贡献，恰好对应 Actor 2 穿越有色区域的时间步

**Sepsis 关键结果**：
- 从 600 条失败轨迹中筛选出 8728 个 TCFE ≥ 0.8 的替代动作
- 单独的临床医生特定效应 + AI特定效应 ≠ tot-ASE，差异高达 95%，而 ASE-SV 始终保证效率性

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 信任度 μ 从低到高 | 临床医生 ASE-SV 得分 ↓, AI 得分 ↑ | 信任度越高→临床医生越少覆盖AI决策→贡献减少 |
| 信任度 μ → 1（完全信任） | 临床医生得分 → 0, AI 得分 → 全部 | AI 完全掌控时承担全部责任 |
| 轮次差异 4-10 | Gini 系数集中在 0.6-0.9 | r-SSE-ICC 只归因到少数关键状态，具有稀疏性 |
| 噪声单调性假设检验 | 分解结果稳健 | 对假设违反具有较好的鲁棒性 |
| 估计误差分析 | 100 样本即可获得稳定估计 | 标准误差在合理范围内 |

### 关键发现

1. **分解公式的严格性**：直觉性的 "TCFE = tot-ASE + SSE" 并不总成立（在 Gridworld 中被反例验证），正确公式为 TCFE = tot-ASE - r-SSE
2. **ASE-SV 的有意义归零机制**：不贡献的智能体被归因为零的两种原因——(a) 对干预无响应（如 Actor 1），(b) 响应了但无法影响状态（如 Planner）
3. **r-SSE-ICC 的稀疏归因**：在 Sepsis 中无论轨迹长度如何，大多数轨迹的 Gini 系数超过 0.6，意味着只有少数关键状态变量主导了 r-SSE
4. **Sepsis 中的责任分配符合直觉**：临床医生在特定信任度下承担约 73.5% 的直接状态相关责任

## 亮点与洞察

1. **框架设计巧妙**：Level 1 的分解将效应按 MMDP 的两个核心组件（智能体行为 vs 环境状态）分离，比传统路径中介分析更具操作含义
2. **公理化保证**：ASE-SV 的唯一性定理为归因方法提供了坚实的理论基础，不依赖启发式
3. **r-SSE 而非 SSE**：发现需要用"反向"状态特定效应才能获得正确分解，这看似反直觉但有严格的数学保证
4. **LLM + RL 的实验设计**：Gridworld 中采用 LLM Planner + RL Actor 的架构，展示了框架在现代 AI 系统中的适用性
5. **稀疏归因的实际价值**：r-SSE-ICC 的稀疏性意味着实践中只需推断少数关键状态的反事实值即可准确估计效应

## 局限与展望

1. **计算复杂度**：ASE-SV 需要枚举智能体子集（$2^n$ 复杂度），r-SSE-ICC 需要逐状态计算条件方差，当智能体数量多或时间步长大时开销显著
2. **噪声单调性假设**：虽然实验显示对假设违反具有鲁棒性，但在实际场景中该假设可能不成立，缺乏部分识别版本
3. **实验规模有限**：两个实验环境的智能体数 ≤ 3、时间步 ≤ 20，缺少大规模多智能体场景的验证
4. **因果模型需已知**：假设可以访问环境的结构因果模型，但现实中通常需要从数据中学习模型
5. **仅处理离散/分类 MMDP-SCM**：对连续状态空间的扩展尚未讨论

## 相关工作与启发

- **因果中介分析**（Pearl 2001, VanderWeele 2016）：本文可视为将经典中介公式推广到多智能体 MDP 结构上
- **Agent-Specific Effects**（Triantafyllou et al. 2024）：本文直接建立在此概念上，进一步实现了完整的效应分解
- **Shapley 值在因果推断中的应用**（Janzing et al. 2024, Heskes et al. 2020）：将博弈论的公平分配概念用于因果归因
- **可解释 AI 中的反事实**：本文为"如果智能体采取不同行动"这类反事实问题提供了比单一 TCFE 更丰富的解释
- **启发**：该框架可直接应用于 LLM Agent 系统中的行为审计——分析多个 LLM Agent 协作时某个决策失误的责任分配

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ - 首次系统性地分解多智能体 MDP 中的反事实效应，理论创新性强
- 实验充分度: ⭐⭐⭐⭐ - 两个环境验证了方法的可解释性，但规模和多样性有限
- 写作质量: ⭐⭐⭐⭐⭐ - 论文结构清晰，定义-定理-实验环环相扣，示例贯穿始终
- 价值: ⭐⭐⭐⭐ - 对多智能体责任归因有重要理论意义，实际应用还需进一步完善

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Sequential Multi-Agent Dynamic Algorithm Configuration](../../NeurIPS2025/reinforcement_learning/sequential_multi-agent_dynamic_algorithm_configuration.md)
- [\[ICML 2026\] Multi-Agent Decision-Focused Learning via Value-Aware Sequential Communication](../../ICML2026/reinforcement_learning/multi-agent_decision-focused_learning_via_value-aware_sequential_communication.md)
- [\[ICML 2025\] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning](divide_and_conquer_grounding_llms_as_efficient_decision-making_agents_via_offlin.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)
- [\[AAAI 2026\] Think, Speak, Decide: Language-Augmented Multi-Agent Reinforcement Learning for Economic Decision-Making](../../AAAI2026/reinforcement_learning/think_speak_decide_language-augmented_multi-agent_reinforcement_learning_for_eco.md)

</div>

<!-- RELATED:END -->
