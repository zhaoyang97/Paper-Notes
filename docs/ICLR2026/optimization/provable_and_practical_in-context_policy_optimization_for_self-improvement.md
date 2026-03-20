# Provable and Practical In-Context Policy Optimization for Self-Improvement

**会议**: ICLR 2026  
**arXiv**: [2603.01335](https://arxiv.org/abs/2603.01335)  
**代码**: https://github.com/UNCSciML/ICPO  
**领域**: LLM推理 / 测试时扩展  
**关键词**: in-context learning, policy optimization, test-time scaling, self-reflection, mathematical reasoning

## 一句话总结
提出 In-Context Policy Optimization (ICPO) 框架，理论证明单层线性自注意力 Transformer 经充分预训练后可在上下文中模拟策略优化算法，并设计实用的 ME-ICPO 算法通过最小熵选择和自评估奖励实现测试时多轮自反思，在数学推理任务上取得显著提升（AIME 2024 上 Qwen2.5-Math-7B 从 11% 提升到 30%）。

## 研究背景与动机

1. **领域现状**：测试时扩展（test-time scaling）已成为提升 LLM 推理能力的重要范式——模型在推理时通过多轮自反思逐步改进回答，无需更新参数。代表方法包括 Chain-of-Thought、Tree-of-Thoughts、Best-of-N、Self-Refine 等。

2. **现有痛点**：(a) 自反思能力为何从预训练中涌现？现有工作（如 Park et al. 2024）直接假设 LLM 具有后验采样/策略优化能力，但未解释这种能力的来源；(b) in-context learning 的理论分析主要集中在监督学习（线性回归）和值函数学习（TD learning），尚无关于策略优化的理论；(c) 已有方法如 Tree-of-Thoughts 需要多步搜索，计算开销大。

3. **核心矛盾**：在上下文中怎样利用历史尝试和奖励反馈来优化自身的输出策略？理论上 Transformer 能否在不更新参数的情况下实现这种策略优化？

4. **本文要解决什么？** (1) 为 LLM 的自反思/自改进行为提供理论基础；(2) 设计一个实用的测试时扩展算法。

5. **切入角度**：将自反思形式化为 K-臂 bandit 问题中的策略优化——agent 生成回答（action），获得奖励（reward），然后在上下文中积累历史 $\{(\mathbf{x}_1, r_1), ..., (\mathbf{x}_t, r_t)\}$ 来优化下一步行为。

6. **核心idea一句话**：Transformer 的自注意力机制天然具有模拟 FTRL 策略优化的归纳偏置，经充分预训练后可在上下文中执行策略优化。

## 方法详解

### 整体框架
ICPO 框架：给定问题 → 模型生成回答 $\mathbf{x}_t$ → 获得奖励 $r_t$（自评估或外部）→ 将 $(\mathbf{x}_t, r_t)$ 加入上下文历史 → 模型基于更新后的历史生成改进的回答 $\mathbf{x}_{t+1}$ → 循环。

理论分析在线性自注意力（LSA）Transformer 上进行，证明其可精确模拟一个基于 FTRL 的策略优化算法。

### 关键设计

1. **Fisher-weighted logit-matching 预训练目标**:
   - 做什么：设计一种新的监督预训练损失函数，使 Transformer 学会在上下文中执行策略优化
   - 核心思路：损失为 $\mathcal{L}(\theta) = \frac{1}{2} \mathbb{E}_{\tau \in \mathcal{D}} [\sum_t \| \text{Proj}(\hat{\mathbf{s}}_{\tau,t+1} - \mathbf{s}_{\tau,t+1}^{\text{PO}}) \|_{\Gamma}^2]$，其中 $\Gamma$ 是策略的 Fisher 信息矩阵，Proj 投影掉常数偏置（因其不影响 softmax 策略）
   - 设计动机：Fisher 加权使损失与 KL 散度成正比（Theorem 4.1），这解释了为什么标准 KL 损失就足以让 Transformer 学会自反思

2. **Population Equivalence 和有限样本保证**:
   - 做什么：证明在充分预训练后，单层 LSA 可以精确模拟目标策略优化算法
   - 核心思路：Theorem 4.2 证明最优参数 $\theta^*$ 使 LSA 在所有可能历史上精确复现策略优化行为；Theorem 4.3 给出 $\tilde{O}(N^2 K / c_\lambda^2)$ 的样本复杂度
   - 设计动机：为 LLM 的 in-context 策略优化能力提供理论证据——单层注意力就足够

3. **鲁棒性保证（Reward Shock Stability）**:
   - 做什么：分析 ICPO 循环对单次奖励扰动的稳定性
   - 核心思路：Theorem 4.8 证明当学习率 $\eta_t = c/t$ 足够小时，一次性奖励扰动 $\delta_r$ 对策略的影响随时间衰减至零：$\mathbb{E}[\|\Delta \hat{\mathbf{p}}_{t+1}^s\|_2] \leq \frac{a(1+C_b)}{s} (\frac{t}{s})^{b-1} |\delta_r|$
   - 设计动机：为使用噪声自评估奖励提供理论支撑

4. **ME-ICPO 实用算法**:
   - 做什么：基于理论框架设计实际可用的测试时推理算法
   - 核心流程：(1) 每轮采样 k 个候选回答；(2) Majority Vote 确定自评估奖励 $r_j^{(t)} = \mathbb{1}[a_j^{(t)} = \hat{a}_t]$；(3) 将回答的 CoT 进行摘要压缩；(4) **最小熵选择**——选择使后续回答熵最小的候选加入上下文
   - 设计动机：最小熵选择遵循离线 RL 的"悲观主义"原则——选择能使 agent 最确信的方向，避免被噪声奖励误导

### 损失函数 / 训练策略
ME-ICPO 在测试时无参数更新——是纯 inference-time 算法。核心策略：
- 每轮采样 $k=16$ 个回答
- Majority vote 作为奖励估计
- CoT 摘要压缩上下文长度
- 最小熵选择确保鲁棒性
- 迭代 $n$ 轮后输出最终回答

## 实验关键数据

### 主实验

| 模型 | Benchmark | Base Mean@16 | w/ ME-ICPO Mean@16 | 提升 |
|------|-----------|-------------|-------------------|------|
| Qwen2.5-Math-7B | AIME 2024 | 11.04 | **30.42** | +19.38 |
| Qwen2.5-Math-7B | AMC | 41.42 | **47.06** | +5.64 |
| Qwen2.5-Math-7B | MATH-L5 | 30.58 | **38.71** | +8.13 |
| Qwen2.5-Math-1.5B | AIME 2024 | 6.46 | **9.79** | +3.31 |
| Qwen2.5-Math-1.5B | MATH-L1 | 49.27 | **57.06** | +12.38 |

在 AIME 2024 上的提升最为显著：7B 模型 +19.38，1.5B 模型 +3.31。ME-ICPO 的 Mean@16 可以超过基线模型的 Maj@k 上限。

### 消融实验

| 配置 | AIME 2024 Accuracy (%) |
|------|----------------------|
| w/o Reward | 19.30 |
| w/o Entropy | 5.77 |
| w/o Entropy & Reward | 6.21 |
| **ME-ICPO (full)** | **30.05** |
| ME-ICPO Oracle | 38.19 |

### 关键发现
- **最小熵选择是最关键组件**：去掉后精度从 30.05% 暴跌至 5.77%，比不做任何操作（6.21%）还差——说明没有合理的选择策略，随机上下文反而有害
- **奖励信号也很重要**：去掉后从 30.05% 降至 19.30%
- **理论验证实验**：LSA 的策略匹配误差快速收敛到数值精度，单次奖励冲击的影响确实随时间衰减
- ME-ICPO 的 Mean@16 可超越基线的 Maj@k 上限——这意味着 in-context 策略优化确实学到了超越简单投票的信息

## 亮点与洞察
- **理论-实践闭环**：从线性自注意力的理论分析出发，推导出实用算法设计原则（奖励引导 + 最小熵选择），再在实际 LLM 上验证——完整的研究闭环
- **最小熵选择的洞察**：不选择奖励最高的候选，而是选择让模型最确信的候选。这在自评估奖励噪声大的场景下尤为重要——高奖励可能是偶然，但低熵意味着模型对某个方向有稳定共识
- **单层就够的理论结论**：与 Lin et al. (2023) 需要 $O(\sqrt{T})$ 层不同，ICPO 仅需单层 LSA，且不随上下文长度增长而需要更多层——更贴近实际 LLM 的长上下文场景

## 局限性 / 可改进方向
- 理论分析基于线性自注意力和线性 bandit 假设，与实际 LLM 和数学推理问题有巨大差距
- ME-ICPO 每轮需采样 16 个候选回答，多轮迭代的计算开销仍然可观
- 自评估奖励基于 Majority Vote，当模型系统性出错时 MV 本身可能给出错误信号
- 仅在数学推理任务上验证，未验证代码生成、逻辑推理等其他领域
- CoT 摘要可能丢失关键推理步骤信息

## 相关工作与启发
- **vs Self-Refine/Reflexion**: 这些工作通过自然语言反馈进行自反思，但没有理论解释为什么有效；ICPO 提供了策略优化视角的理论基础
- **vs Tree-of-Thoughts**: ToT 在每一步搜索，ME-ICPO 每轮优化整个 CoT——更粗粒度但计算效率更高
- **vs TTRL**: TTRL 在测试时进行梯度更新，ME-ICPO 纯 in-context 无参数更新——更轻量
- **vs Best-of-N**: BoN 最终选最好的一个，ME-ICPO 利用多轮迭代积累上下文信息逐步改进——理论上更有优势

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次从策略优化角度为 LLM 自反思提供理论分析，最小熵选择设计新颖
- 实验充分度: ⭐⭐⭐⭐ 理论验证充分，但 LLM 实验仅限数学任务和 Qwen 系列
- 写作质量: ⭐⭐⭐⭐ 理论推导严谨，但理论到实践的过渡可以更紧密
- 价值: ⭐⭐⭐⭐ 为测试时扩展提供理论基础，最小熵选择策略有实用价值
