# MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization

**会议**: AAAI 2026  
**arXiv**: [2503.16874](https://arxiv.org/abs/2503.16874)  
**代码**: [https://github.com/exoskeletonzj/MARS](https://github.com/exoskeletonzj/MARS)  
**领域**: LLM Agent / 自动提示优化 / 多Agent协作  
**关键词**: 自动提示优化, 苏格拉底对话, POMDP, Teacher-Critic-Student, 伪梯度  

## 一句话总结
提出 MARS 五智能体框架做自动提示优化（APO）：Planner 生成任务特定的优化轨迹，Teacher-Critic-Student 三体进行苏格拉底对话式迭代精炼 prompt（模拟文本空间中的伪梯度下降），Target 执行并反馈，整体建模为 POMDP，在 17 个数据集上平均超越前 SOTA（PE2）6.04%（通用任务）和 6.42%（领域任务），且仅需 1-shot 训练数据。

## 研究背景与动机

1. **领域现状**：自动提示优化（APO）旨在克服手工 prompt 的认知偏差，自动探索更优的 prompt 设计空间。现有方法分两大类——生成-搜索法（APE/ProTeGi/PoisonedRAG：生成候选 prompt + 搜索最优）和元 prompt 法（OPRO/PE2：设计精细的 meta-prompt 指导优化）。
2. **现有痛点**：(a) **模板刚性**：固定的元 prompt 模板无法动态适应不同任务需求，难以捕捉任务特定的优化方向；(b) **探索低效**：生成-搜索方法只在初始候选附近做局部搜索，可能过早收敛或遗漏更好的 prompt 空间。
3. **核心矛盾**：prompt 优化的搜索空间是离散、高维、不可微的，无法直接用梯度下降；但又需要类似梯度的"方向性引导"来避免盲目搜索。
4. **切入角度**：受苏格拉底教学法启发——通过提问（而非直接告知）引导学生自主发现答案。将 prompt 优化过程建模为 POMDP，用多 Agent 协作模拟梯度式的迭代精炼。
5. **核心 idea 一句话**：五智能体 POMDP 框架——Planner 规划路径 + Teacher-Critic-Student 苏格拉底对话做伪梯度精炼 + Target 评估反馈。

## 方法详解

### 整体框架
五个 LLM Agent 协作，建模为 POMDP $\langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \mathcal{O} \rangle$：
- **Planner**：分解优化目标为子目标序列 $\mathbf{ST} = [st_1, \ldots, st_n]$
- **Teacher**：根据当前子目标和前一版 prompt 提出苏格拉底式问题 $q_i$
- **Critic**：评估问题的质量和引导方向的合理性，给出反馈 $c_i$
- **Student**：综合问题和批评，更新内部状态并生成新版 prompt $p_i$
- **Target**：在下游任务上执行 prompt 并返回性能奖励 $\mathcal{R}$

### 关键设计

1. **Planner — 优化轨迹规划**:
   - 做什么：将抽象的"优化 prompt"目标分解为具体的、有序的子目标序列
   - 核心思路：$\mathbf{ST} = \pi_{\text{plan}}(g, x, p_0)$，引入隐变量 $z$ 建模任务语义，通过 $\arg\max_{\mathbf{ST}} \mathbb{E}_{z \sim q(z|g,x)}[\log P(\mathbf{ST}|z, p_0)]$ 生成结构化计划
   - 设计动机：静态 meta-prompt 是"一刀切"的，Planner 为每个任务定制优化路径，实现自适应

2. **Teacher-Critic-Student 苏格拉底对话**:
   - 做什么：通过迭代的问答-批评-修改循环精炼 prompt
   - 核心思路：每步 $i$——Teacher 提问 $q_i = \pi_t(st_i, p_{i-1}, \mathcal{H}_{<i})$（引导 Student 思考特定方向）→ Critic 评估 $c_i = \pi_c(q_i, \mathcal{H}_{<i})$（确保问题质量和方向正确） → Student 响应更新 $p_i = \pi_s((q_i, c_i), p_{i-1}, \mathcal{H}_{<i})$。所有 Agent 都有对话历史 $\mathcal{H}_{<i}$ 的完整上下文
   - 设计动机：模拟离散 prompt 空间中的"伪梯度"——Teacher 的问题相当于梯度方向，Critic 确保方向正确，Student 执行"步进"。Proposition 1 形式证明：累计改进有下界 $\geq \sum_i (\bar{A}_i - \sigma^2/2\lambda)$

3. **自适应终止**:
   - 做什么：基于边际收益自动决定何时停止优化
   - 核心思路：$\Delta\mathcal{R}^{(t)} = \mathcal{R}^{(t)} - \mathcal{R}^{(t-1)} > \delta$ 且 $t < I$ 则继续。Proposition 2 证明 Lipschitz 条件下奖励变化有界，小步长时收敛
   - 设计动机：避免过度精炼浪费计算资源

### 训练效率亮点
仅需 **1 个训练样本**做优化——Planner 从单个示例即可推断任务结构和语义，这是因为 APO 的核心是理解"任务是什么"而非记忆"数据是什么"。

## 实验关键数据

### 主实验 — 通用任务（BBH + MMLU，6+6=12 个任务）

| 方法 | BBH 平均 | MMLU 平均 | 总平均 |
|------|---------|----------|-------|
| Origin (原始 prompt) | 53.71 | 76.39 | 64.95 |
| CoT (Zero-Shot) | 61.40 | 78.20 | 69.79 |
| PE2 (前 SOTA) | 69.45 | 88.44 | 78.81 |
| **MARS** | **79.52** | **90.94** | **85.11** |

### 主实验 — 领域任务（C-Eval + LSAT + GSM8K，5 个任务）

| 方法 | C-Eval | GSM8K | LSAT-AR | 平均 |
|------|--------|-------|---------|------|
| PE2 | 66.47 | 83.46 | 34.50 | 69.39 |
| **MARS** | **77.13** | **89.22** | **38.42** | **75.81** |

### 消融实验

| 配置 | BBH 平均 | MMLU 平均 | 变化 |
|------|---------|----------|------|
| MARS (完整) | 79.52 | 90.94 | — |
| w/o Socratic | 68.28 | — | **-11.31** |
| w/o Planner | 72.82 | — | -6.77 |
| w/o Critic | 76.04 | — | -3.55 |

### 关键发现
- **MARS 在所有 17 个数据集上全面 SOTA**：通用任务超前 SOTA（PE2）6.04%，领域任务超 6.42%
- **苏格拉底对话机制贡献最大**：去掉后平均掉 11.31%，远大于 Planner（-6.77%）和 Critic（-3.55%）
- **1-shot 训练已足够**：0-shot 平均 77.77%，1-shot 79.59%，3-shot 79.81%——增加训练数据的边际收益极小
- **收敛快**：通常 5 轮优化即收敛（vs OPRO 10 轮仍未收敛），大幅节省推理计算
- **跨模型泛化**：在 DeepSeek-V2.5 上优化的 prompt 直接用于 GPT-4o 仍有效，说明优化出的 prompt 是模型无关的
- **推理时 scaling law**：在相同 token 消耗下，MARS 性能最高；达到相同性能水平，MARS 消耗最少

## 亮点与洞察
- **将 APO 建模为 POMDP**是理论上的关键创新——Student 的内部推理状态是隐状态，Teacher/Critic 的交互是动作，prompt 是观测，任务性能是奖励，形成完整的数学框架
- **苏格拉底教学法→伪梯度下降**的类比非常精妙——Teacher 提问=梯度方向，Critic 评估=梯度校正，Student 更新=参数更新，形式证明了累计改进的下界
- **仅需 1 个训练样本**是极其亮眼的结果——说明 APO 的本质是"理解任务规范"而非"拟合训练数据"，Planner 的任务理解能力是核心
- **附录提供了所有 17 个任务的最终优化 prompt**，直接可复用

## 局限性 / 可改进方向
- 五个 Agent 的推理开销较大（每轮需要 5 次 LLM 调用），对计算预算敏感
- 依赖 DeepSeek-V2.5 / GPT-4o 作为 Agent backbone，小模型可能无法胜任 Teacher/Planner 角色
- POMDP 的隐状态转移 $\mathcal{T}$ 实际上无法精确建模（是 LLM 隐式实现的），理论分析的假设（Lipschitz 连续、有界方差）在实际中可能不严格成立
- 只在文本分类/QA/数学任务上验证，缺少生成任务（如摘要、翻译）的评估
- Planner 生成的子目标序列是否真的比手动设计更优，缺少与人类 expert prompt engineer 的对比

## 相关工作与启发
- **vs OPRO**：OPRO 用 meta-prompt 直接让 LLM 生成优化后的 prompt，是单 Agent 方法；MARS 用多 Agent 苏格拉底对话做迭代精炼，收敛更快（5 轮 vs 10+轮），性能更高
- **vs PE2**：PE2 是前 SOTA meta-prompt 方法；MARS 在所有任务上超越 PE2 6%+，且计算效率更高
- **对 Agent 研究的启示**：Teacher-Critic-Student 三体模式可以迁移到其他需要"迭代精炼"的 Agent 任务（如代码调试、文本修改、方案优化）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ POMDP建模+苏格拉底五Agent框架是APO领域的范式创新，理论和方法都很完整
- 实验充分度: ⭐⭐⭐⭐⭐ 17个数据集+消融+收敛分析+跨模型验证+1-shot分析+推理scaling law，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ POMDP形式化严谨，Proposition证明完整，附录极其详出（所有prompt+完整优化过程）
- 价值: ⭐⭐⭐⭐⭐ 1-shot即达SOTA的APO方法有极高实用价值，多Agent苏格拉底范式可广泛迁移
