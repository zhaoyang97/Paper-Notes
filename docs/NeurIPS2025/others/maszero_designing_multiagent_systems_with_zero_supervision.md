# MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision

**会议**: NeurIPS 2025 (SEA Workshop, Oral)  
**arXiv**: [2505.14996](https://arxiv.org/abs/2505.14996)  
**代码**: [https://github.com/SalesforceAIResearch/MAS-Zero](https://github.com/SalesforceAIResearch/MAS-Zero)  
**领域**: LLM Agent / 多智能体系统  
**关键词**: multi-agent system, automatic MAS design, meta-agent, zero supervision, inference-time optimization  

## 一句话总结
MAS-ZERO 是首个推理时自动 MAS 设计框架，通过 meta-agent 迭代设计、批评和改进 MAS 配置（包括任务分解和 sub-MAS 分配），无需验证集和训练，在推理（+16.69%）、编程（+16.66%）和搜索代理（+5.45%）任务上均超越手动和自动 MAS baseline，同时保持 Pareto 最优的准确率-成本权衡。

## 研究背景与动机

1. **领域现状**：LLM 多智能体系统（MAS）通过多个 agent 协作（辩论、分工、验证）来解决单模型无法有效处理的复杂任务。当前 MAS 设计主要有两种：手动设计（Debate、Self-Refine 等）和自动设计（ADAS、AFlow 用验证集搜索最优配置）。

2. **现有痛点**：(1) 手动 MAS 依赖人工定义角色和通信协议，难以与底层 LLM 能力对齐且不适应新任务；(2) 自动 MAS 需要标注验证集进行调优，得到的是固定架构（一个配置用于所有问题），缺乏逐问题的适应性；(3) 现有自动方法不能在复杂 MAS 无帮助时动态退化为简单 MAS 或单 agent（如 CoT），导致在困难任务上多 baseline 甚至不如 CoT。

3. **核心矛盾**：有效的自动 MAS 需要同时满足三个条件——(a) 能分解复杂问题并在不需要时退化为简单系统，(b) 自动学习 LLM agent 能力并设计匹配的架构，(c) 推理时逐问题适配而非依赖验证集。现有方法无一同时满足。

4. **本文要解决什么**：设计一个无需验证集、在推理时自动为每个问题实例定制 MAS 配置的框架。

5. **切入角度**：引入 meta-agent 工作在 MAS 层面（而非 agent 层面），通过迭代的 meta-design（分解问题 + 分配 sub-MAS）和 meta-feedback（评估可解性和完整性）来自我进化，并在最终阶段从所有候选答案（包括简单 building block 的输出）中选择最优。

6. **核心idea一句话**：用 meta-agent 将 MAS 设计变成一个推理时的迭代优化过程——每步设计一个 MAS（代码形式），执行后基于可解性和完整性反馈改进，最终从 building block 和多轮进化的候选中选择最佳答案。

## 方法详解

### 整体框架
MAS-ZERO 分三步：(1) **MAS-Init**——运行 4 个 building block（CoT、CoT-SC、Debate、Self-Refine），收集初始候选答案；(2) **MAS-Evolve**——meta-agent 迭代 5 轮，每轮设计 MAS（分解问题 + 分配 sub-MAS）、执行、评估反馈、存入经验库；(3) **MAS-Verify**——从 9 个候选答案（4 个 building block + 5 轮进化）中选择最可靠的。

### 关键设计

1. **Meta-Design（任务分解 + sub-MAS 分配）**：
   - 做什么：将原始问题分解为可管理的子任务，为每个子任务分配由 building block 组成的 sub-MAS
   - 核心思路：Meta-agent 被要求 (a) 将问题分解为"可管理但相互依赖"的子任务，(b) 为每个子任务指定 sub-MAS（可以是单个 block 如 CoT，或 chain 如 CoT→Self-Refine，或并行 {CoT ∥ Debate}）。设计以可执行 Python 代码形式表达
   - 设计动机：允许 meta-agent 修改 block 之间的连接和参数（如温度、辩论轮次），但不允许发明新 block——这在探索和可靠性之间取得平衡

2. **Meta-Feedback（可解性 + 完整性评估）**：
   - 做什么：评估 meta-design 产出的 MAS 质量并生成改进反馈
   - 核心思路：执行 MAS 获取两级中间输出（sub-task 级和 agent 级），评估两个标准：
     - **Solvability**：每个子任务是否被其 sub-MAS 独立且完整地解决？如果不可解，建议进一步分解或更换 sub-MAS
     - **Completeness**：所有子任务是否覆盖了原始问题的全部必要信息？如果有遗漏，建议修改分解策略
   - 设计动机：不同于只看最终答案正确与否的验证集方法，solvability+completeness 提供更丰富的过程级反馈

3. **Experience Library + MAS-Verify**：
   - Experience Library：每轮的 MAS 设计、中间输出和反馈存入经验库，作为后续轮次的上下文
   - MAS-Verify：(a) 按最终答案频率排序候选答案（majority vote 启发），(b) 过滤无效答案，(c) 让 meta-agent 从排序后的候选中选择最佳——包括 building block 的输出，实现动态退化

### 训练策略
完全 training-free，zero supervision。所有操作通过 prompting 实现，只需黑盒 LLM 访问。相同 prompt 模板用于所有任务。

## 实验关键数据

### 主实验（GPT-4o 为 backbone）

| 方法 | AIME24 | GPQA | 平均 | vs MAS-ZERO |
|---|---|---|---|---|
| CoT | 8.33 | 45.78 | 27.06 | -14.91 |
| CoT-SC | 16.67 | 43.37 | 30.02 | -11.95 |
| Debate | 4.17 | 46.99 | 25.58 | -16.39 |
| AFlow (SOTA auto MAS) | 20.83 | 46.99 | 33.91 | -8.05 |
| ADAS | × | 45.20 | — | — |
| **MAS-ZERO** | **33.33** | **50.60** | **41.97** | — |

AIME24 上 ADAS 完全失败（0%，标注其验证集不可靠）。MAS-ZERO 比 AFlow 平均高 8.05%。

### 消融实验

| 配置 | AIME24 | GPQA | 平均 | 影响 |
|---|---|---|---|---|
| MAS-ZERO | 33.33 | 50.60 | 41.97 | — |
| - MAS-Init | 12.50 | 48.43 | 30.46 | -11.50（失去 building block 退化能力）|
| - MAS-Evolve | 20.00 | 48.73 | 34.37 | -7.60 |
| - meta-design（无分解）| 20.83 | 45.18 | 33.01 | -8.96 |
| - meta-feedback | 25.00 | 42.17 | 33.59 | -8.38 |
| → ensemble feedback | 16.67 | 46.88 | 31.77 | -10.19（反而更差）|
| - MAS-Verify（取最后轮）| 6.70 | 33.83 | 20.27 | -21.70（最大损失）|

MAS-Verify 是最关键组件——去掉后性能暴跌 21.70%。Oracle verifier 可将 GPQA 提升至近 95%，说明验证是主要瓶颈。

### 关键发现
- **简单策略有时最强**：CoT、CoT-SC 在某些 benchmark 上优于所有复杂 MAS（包括自动方法）。MAS-ZERO 是唯一能动态退化到这些简单策略的自动 MAS
- **ADAS 在 AIME24 上完全失败**：即使有验证集，基于验证集的方法在困难数据分布上可能完全不可靠
- **Ensemble feedback 不如单次 feedback**：反直觉发现——当前 straightforward 的 meta-feedback 已经足够强，简单集成反而引入噪声
- **异构 agent 配置有潜力但受限**：用 o3-mini 做 meta-agent + GPT-4o 做 individual agent 的提升有限，因为整体性能受限于 individual agent 能力
- **成本效率 Pareto 最优**：MAS-ZERO 在 Figure 1 的准确率-成本散点图上在所有三个 benchmark 上都位于 Pareto 前沿

## 亮点与洞察
- **推理时 MAS 设计是全新范式**：不依赖验证集、不产出固定架构，而是为每个问题实例定制 MAS——这是 test-time compute scaling 在多智能体领域的自然延伸
- **Code-as-MAS 的表示方式**：将 MAS 设计以可执行 Python 代码表达，使得 meta-agent 的设计可以被直接运行和评估，大大降低了"设计"到"执行"的转换损失
- **动态退化能力**：这是 MAS-ZERO 相对于其他自动 MAS 的最独特优势——当复杂 MAS 无帮助时自动回退到 CoT/CoT-SC，避免了"用力过猛"的问题

## 局限性 / 可改进方向
- Meta-agent 需要足够强的 LLM（GPT-4o 级别），弱 LLM（7B-20B）无法可靠执行 meta 层面的代码生成和反馈
- 推理时 token 成本较高（9 个候选答案 = 4 building block + 5 轮进化），虽然 Pareto 最优但绝对成本不低
- Ensemble meta-feedback 失败表明当前 feedback 机制仍有改进空间
- 在搜索代理任务上提升较小（+5.45%），因为搜索错误被内容"锚定"后难以被 meta-verification 识别

## 相关工作与启发
- **vs AFlow / ADAS**：AFlow 和 ADAS 用 MCTS / rejection sampling 在验证集上搜索最优 MAS，得到固定架构。MAS-ZERO 在推理时逐问题设计，无需验证集。AFlow 在 AIME24 上 20.83%，MAS-ZERO 33.33%
- **vs GiGPO (2505.10978)**：GiGPO 解决 Agent RL 训练的 step 级 credit assignment；MAS-ZERO 解决 Agent 配置的自动设计。两者互补——MAS-ZERO 设计的 MAS 可以用 GiGPO 做 RL 训练
- **vs DRIFT (2506.12104)**：DRIFT 保护 Agent 不被外部注入攻击，MAS-ZERO 设计 Agent 如何协作。两者面向 Agent 系统的不同层面

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个推理时 MAS 自动设计框架，solvability/completeness 双维度反馈，动态退化能力独特
- 实验充分度: ⭐⭐⭐⭐⭐ 3 个 LLM backbone，3 个领域（推理/编程/搜索），11 个 baseline，全面消融，Pareto 分析，标准差报告
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1-3 的概念对比和 Pareto 图极其清晰，框架描述条理分明
- 价值: ⭐⭐⭐⭐⭐ 揭示了"简单策略有时最强"和"验证集方法不可靠"的重要 insight，开源代码有直接复用价值
