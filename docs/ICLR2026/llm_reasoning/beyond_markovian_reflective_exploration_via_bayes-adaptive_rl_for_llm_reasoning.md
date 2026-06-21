---
title: >-
  [论文解读] Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning
description: >-
  [ICLR 2026][Reasoning][Bayes-Adaptive RL] 本文用贝叶斯强化学习重新解释 LLM 的"自我反思"行为——把反思看作在 MDP 不确定性下的信息收集，并提出 BARL 算法，通过维护对候选答案的 MDP 假设后验、在信念与奖励反馈冲突时切换策略，从而在数学推理上同时提升准确率和 token 效率。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "Bayes-Adaptive RL"
  - "反思探索"
  - "MDP 后验"
  - "不确定性自适应策略"
  - "Token 效率"
---

# Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning

**会议**: ICLR 2026  
**代码**: [https://github.com/shenao-zhang/BARL](https://github.com/shenao-zhang/BARL)  
**领域**: LLM 推理 / 强化学习  
**关键词**: Bayes-Adaptive RL, 反思探索, MDP 后验, 不确定性自适应策略, Token 效率  

## 一句话总结
本文用贝叶斯强化学习重新解释 LLM 的"自我反思"行为——把反思看作在 MDP 不确定性下的信息收集，并提出 BARL 算法，通过维护对候选答案的 MDP 假设后验、在信念与奖励反馈冲突时切换策略，从而在数学推理上同时提升准确率和 token 效率。

## 研究背景与动机
**领域现状**：RL 训练的推理模型（DeepSeek-R1 等）会涌现出"重新思考""纠错"等自我反思行为，被普遍视作 test-time scaling 的来源，靠生成更长的 CoT 提升表现。

**现有痛点**：但人们并不清楚这些反思**为什么有用、何时该出现**。常规 RL 的最优策略是马尔可夫的——策略只依赖当前状态 $s_t$，对同一个状态无论历史如何都给出相同动作分布。这意味着"回到同一个状态、用新获得的上下文换个走法"这种反思行为，对马尔可夫最优策略的价值毫无增益，因此 RL 根本没有动机去学它。

**核心矛盾**：常规 RL 里探索只发生在训练期（trial-and-error），部署后参数冻结、$\epsilon\approx0$，没有任何机制鼓励"边推理边探索"。所以反思要么不稳定地涌现，要么即使出现也无法被理论解释——近期工作也发现反思频率与性能相关性很弱。

**本文目标**：给反思探索一个**有原则的理论基础**，并据此设计算法，指导 LLM 何时、如何反思性探索。

**核心 idea**：**【贝叶斯化 RL 目标】** 训练数据 $D$ 并不能唯一确定真实 MDP $M^*$，由此诱导出 MDP 上的后验 $p(M\mid D)$。把目标从"在 $M^*$ 上最大化回报"改成"在后验上最大化贝叶斯期望回报" $J(\pi)=\mathbb{E}_{M\sim p(M\mid D)}[J_M(\pi)]$。**【不确定性自适应策略】** 这个目标的最优策略必然依赖完整历史（信念），通过信念更新自然激励信息收集动作，反思就成了"消除假设、缩小 MDP 不确定性"的合理行为——论文进一步证明自适应策略相比任意马尔可夫策略的回报差距可以**任意大**。

## 方法详解

### 整体框架
BARL（Bayes-Adaptive RL for LLM Reasoning）把每个 prompt 的多条 CoT rollout 各自对应一个"MDP 假设"，用后验信念给每个假设的价值加权，并在预测奖励与观测奖励不一致时压低相应假设的权重作为"切换策略"信号。整体是在常规 policy gradient 上把 $M^*$ 下的价值替换成**后验加权价值**。

```mermaid
flowchart TD
    A[Prompt s0] --> B[采样 |M| 条 CoT rollout]
    B --> C[抽取候选答案 → 构造 MDP 假设 M_i]
    C --> D[计算各假设价值 Q_Mi]
    D --> E["后验加权: 模型可信度 × 奖励一致性"]
    E --> F[后验加权价值 → policy gradient 更新 θ]
    F -->|信念与奖励冲突时下调假设权重| C
```

### 关键设计

**1. 后验加权策略梯度：把"真 MDP"换成"假设集合"。** 常规策略梯度 $\nabla_\theta J = \mathbb{E}[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid h_t)\,Q^{\pi_\theta}_{M^*}(h_t,a_t)]$ 依赖未知的真实 MDP。BARL 把它替换成后验上的期望 $\mathbb{E}_{M\sim p(M\mid D,h_t)}[Q^{\pi_\theta}_M(h_t,a_t)]$。这里刻意用 Q 值而非 advantage，因为后者要在每一步做分支式 Monte-Carlo rollout，代价太高；用 Q 值则可复用整条 CoT 的 KV cache。

**2. 后验分解：可信度 × 奖励一致性。** 由贝叶斯公式，假设 $M$ 的后验拆成两项 $p(M\mid D,h_t)\propto p(M\mid D,s_{0:t})\cdot p(r_{0:t-1}\mid s_{0:t},a_{0:t-1},M)$。第一项"只看 CoT 不看奖励"，建模为策略生成该假设对应答案 $y^M_{s_0}$ 的概率（即**模型对该假设的可信度**）；第二项是观测奖励在该假设下的似然，按 $p(r_t\mid s_t,a_t,M)\propto \exp(-\beta|r_t - r_M(s_t,a_t)|)$ 累乘。最终后验加权价值写成三因子相乘：

$$\mathbb{E}_M[Q^{\pi_\theta}_M(h_t,a_t)] = \sum_{i=0}^{|M|} \underbrace{Q^{\pi_\theta}_{M_i}(h_t,a_t)}_{\text{假设 }M_i\text{ 下的价值}}\;\underbrace{\pi_\theta(y^{M_i}_{s_0}\mid s_t)}_{\text{模型可信度}}\;\underbrace{\prod_{t'=0}^{t-1}\exp(-\beta|r_{t'}-r_{M_i}(s_{t'},a_{t'})|)}_{\text{观测奖励与 }M_i\text{ 的一致性}}$$

**3. 反思信号 = 信念与奖励的冲突。** 第三个乘积项是 BARL 的灵魂：当某个假设虽然模型可信度高（belief 大），但它预测的奖励与实际观测奖励持续不符时，这个 $\exp(-\beta|\cdot|)$ 因子会把它的权重压下去——这恰好就是"该切换策略"的显式信号。换句话说，BARL 把"何时反思"形式化为：**当内部信念与累积奖励反馈出现矛盾时，下调那些高可信但不太可能最优的假设**。候选集 $\{M_i\}$ 由 prompt 的 ground-truth 答案 $M_0$ 加上从模型 CoT 抽取的候选答案构成，提议分布取支撑集上的均匀分布，重要性比在假设间自归一化。

**4. 进度奖励作为稠密信号。** 在稀疏的 outcome verifier 之外引入进度奖励 $r(s_t,a_t)=\pi_\phi(y^*_{s_0}\mid s_t+a_t+\texttt{</think>}) - \pi_\phi(y^*_{s_0}\mid s_t+\texttt{</think>})$，衡量追加一步推理后模型输出正确答案概率的提升。相比 Monte-Carlo 过程奖励，它避免逐步分支 rollout、且能复用 KV cache，计算上更省。

## 实验关键数据

### 主实验表格（pass@1，三次独立训练的均值±标准误）

| 模型 | GSM8K | MATH | College | Olympiad | AIME | AMC | Average |
|---|---|---|---|---|---|---|---|
| Qwen-1.5B base | 40.0 | 34.1 | 6.6 | 21.8 | 16.7 | 32.5 | 25.3 |
| GRPO | 83.9 | 71.5 | 45.1 | 31.6 | 17.8 | 55.8 | 51.0 |
| Progress | 84.8 | 72.1 | 45.9 | 35.5 | 14.4 | 55.8 | 51.4 |
| **BARL** | **85.8** | **72.7** | **46.8** | **35.8** | **17.8** | **60.8** | **53.3** |
| Qwen-7B GRPO | 90.3 | 77.6 | 47.0 | 38.5 | 24.5 | 65.0 | 57.1 |
| Qwen-7B **BARL** | **91.7** | **79.2** | **47.5** | **42.0** | **29.0** | **66.7** | **59.4** |
| Llama-8B GRPO | 85.7 | 74.3 | 39.6 | 36.0 | 16.7 | 60.4 | 52.1 |
| Llama-8B **BARL** | 85.4 | 73.9 | **40.4** | **37.2** | **17.8** | **61.7** | **52.7** |

跨三种模型 BARL 平均准确率均领先，在需要有效探索的难 benchmark（Olympiad/AMC/AIME）上增益最明显。

### 消融实验（token 效率 + 反思频率）

| 维度 | 关键结果 |
|---|---|
| Token 效率（pass@k vs 总 token 数） | BARL 用更少 token 达到更高准确率：比 Progress 少最多 **1.63×**、比 GRPO 少 **2×**、比 base 少 **10×+** |
| 反思频率（Figure 6） | base 模型反思更频繁却准确率更低；反思频率与性能**弱相关** |
| 合成任务（重复 prompt token 三次） | 常规 RL 记住训练解但无法泛化到未见 token；BARL 靠消除假设切换策略，最终找到真 MDP |
| CoT 有效性（贝叶斯 Q 值） | BARL 的 CoT 平均贝叶斯状态-动作价值更高，兼顾探索与利用 |

### 关键发现
- **反思频率 ≠ 性能**：base 模型触发更多"反思"关键词却表现更差，说明它在预训练中学到的是**表面的风格化反思**；BARL 学到的是更有效的反思探索。
- **决定因素是 thinking token 的有效性与探索效率**，而非响应长度或反思次数。
- **候选集质量很关键**：合成实验中给定"奖励三元组是重复模式"的先验（$|M|=3$）比无先验（$|M|=27$）收敛更快，候选既要够多样以覆盖部署不确定性、又要限制在最可信的少数以缩小假设空间。

## 亮点与洞察
- **把"反思"从玄学变成数学**：用 Bayes-Adaptive RL 给出了反思探索"为什么、怎么样、何时"的统一解释，并用定理 4.1/4.2/4.3 严格刻画了马尔可夫策略与自适应策略的本质差距（可任意大）。
- **反思信号的工程化落地很优雅**：把"信念与奖励冲突→切换策略"做成一个 $\exp(-\beta|\cdot|)$ 乘积因子，类比"线性化的 best-of-N"，但带上了显式的何时/如何探索指引。
- **计算几乎零额外开销**：候选答案概率在每步末尾算、复用 CoT 前缀 cache，避免了过程奖励常见的分支 rollout。
- **一个反直觉但重要的实证结论**：更长、更频繁的反思未必更好，关键是 token 的"有效性"。

## 局限与展望
- 候选集 $\{M_i\}$ 由模型自身 CoT 抽取，若模型本身覆盖不到正确策略，假设空间可能漏掉真 MDP，BARL 的优势会受限（论文也强调候选多样性 vs 可信度的平衡）。
- 进度奖励依赖把"输出正确答案概率提升"作为代理，需要 ground-truth 答案 $y^*_{s_0}$，对无标准答案的开放任务不直接适用。
- 实验集中在数学推理 + 1.5B/7B/8B 规模，是否在更大模型、代码/agent 等任务上同样有效尚待验证。
- $\beta$、$|M|$ 等超参对候选权重影响较大，论文用固定值（$\beta=1,|M|=5$），自适应调节空间未深入。

## 相关工作与启发
- **与 meta-RL 的联系**：不确定性自适应策略可看作"用 in-context learning 代替参数更新"，BARL 相当于"learning to do in-context learning"，但更强调贝叶斯目标下探索-利用权衡的最优解。
- **与进度奖励/MRT（Qu et al. 2025）的区别**：MRT 奖励"朝正确答案推进"的策略，BARL 在此之上额外鼓励在贝叶斯框架下探索多个可信策略。
- **与 PSRL/Thompson sampling 的 LLM 工作（Arumugam & Griffiths 2025；Dwaracherla et al. 2024）的区别**：它们把贝叶斯探索放在外部算法脚手架或数据采集层，BARL 则**直接在 RL 微调目标里优化贝叶斯目标**，提供 step-level 的反思指引。
- **启发**：把"模型的某种涌现行为"先用一个规范化的目标（这里是贝叶斯回报）解释清楚、再反推算法，是一条比"启发式加 reward"更扎实的研究范式。

## 评分
- **新颖性** ⭐⭐⭐⭐⭐：用 Bayes-Adaptive RL 重新框定反思探索，理论（定理 4.1–4.3）与算法（三因子后验加权）都有原创性，回答了"反思为什么有用"的根本问题。
- **实验充分度** ⭐⭐⭐⭐：合成任务 + 三模型四 benchmark + token 效率/反思频率/CoT 有效性多角度消融，三随机种子；但规模偏小、任务局限在数学。
- **写作质量** ⭐⭐⭐⭐⭐：Why/How/When 三问贯穿全文，理论铺垫到算法落地的逻辑链条清晰，公式与直觉解释配合到位。
- **价值** ⭐⭐⭐⭐：既给出了理解 RL 推理模型反思行为的理论透镜，又交付了即插即用、计算开销小的算法，对 test-time scaling 和高效推理有直接借鉴意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MAGO: Beyond Fixed Hyperparameters with Multi-Objective Pareto Optimization for Hybrid LLM Reasoning](mago_beyond_fixed_hyperparameters_with_multi-objective_pareto_optimization_for_h.md)
- [\[ICLR 2026\] Attention as a Compass: Efficient Exploration for Process-Supervised RL in Reasoning Models](attention_as_a_compass_efficient_exploration_for_process-supervised_rl_in_reason.md)
- [\[ICLR 2026\] Tricks or Traps? A Deep Dive into RL for LLM Reasoning](tricks_or_traps_a_deep_dive_into_rl_for_llm_reasoning.md)
- [\[ICLR 2026\] Beyond Magnitude: Leveraging Direction of RLVR Updates for LLM Reasoning](beyond_magnitude_leveraging_direction_of_rlvr_updates_for_llm_reasoning.md)
- [\[ICLR 2026\] RL of Thoughts: Navigating LLM Reasoning with Inference-Time Reinforcement Learning](rl_of_thoughts_navigating_llm_reasoning_with_inference-time_reinforcement_learni.md)

</div>

<!-- RELATED:END -->
