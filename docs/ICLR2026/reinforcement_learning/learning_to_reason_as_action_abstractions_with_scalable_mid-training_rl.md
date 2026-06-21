---
title: >-
  [论文解读] Learning to Reason as Action Abstractions with Scalable Mid-Training RL
description: >-
  [ICLR 2026][强化学习][mid-training] 本文首次从理论上刻画了"中训练 (mid-training) 如何塑造后训练 RL"，指出有效的中训练应在**时间动作抽象 (temporal action abstraction)** 而非原始 token 空间中学习，并据此提出 RA3——一个用自监督 RL 发现潜在推理结构、再 SFT 回灌的可扩展中训练算法。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "mid-training"
  - "动作抽象"
  - "时间抽象 (options)"
  - "自监督 RL"
  - "EM"
  - "RLVR"
  - "代码生成"
---

# Learning to Reason as Action Abstractions with Scalable Mid-Training RL

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=uWd9A1zp0Y](https://openreview.net/forum?id=uWd9A1zp0Y)  
**代码**: 待确认  
**领域**: 强化学习 / LLM Mid-Training  
**关键词**: mid-training, 动作抽象, 时间抽象 (options), 自监督 RL, EM, RLVR, 代码生成  

## 一句话总结
本文首次从理论上刻画了"中训练 (mid-training) 如何塑造后训练 RL"，指出有效的中训练应在**时间动作抽象 (temporal action abstraction)** 而非原始 token 空间中学习，并据此提出 RA3——一个用自监督 RL 发现潜在推理结构、再 SFT 回灌的可扩展中训练算法。

## 研究背景与动机
**领域现状**：当下 LLM 的训练已固化为"预训练 → 中训练 → RLVR 后训练"三段式，其中中训练（在最优策略采样的专家数据上继续预训练）被广泛用来强化策略先验，是后续 RL 能不能起飞的关键一步。

**现有痛点**：尽管中训练几乎成了标配，但它在后训练 RL 中究竟扮演什么角色一直说不清楚。大家只能靠"初始策略的准确率""熵"这类启发式指标隔靴搔痒地判断中训练好不好，这些信号既间接、又不保证下游真的变好，导致中训练算法的设计基本是拍脑袋。

**核心矛盾**：中训练用的是普通的 next-token prediction (NTP)，它在**原始 token 动作空间**上做模仿学习——动作集合巨大、规划视野极长。而后训练 RL 想要的是一个动作空间已被剪枝干净、规划视野已被压短的良好起点。NTP 给的先验和 RL 真正需要的先验之间存在结构性错位。

**本文目标**：把"中训练有效性"拆成可证明的量——后训练 RL 的 regret 可分解为「动作集剪枝误差」+「在剪枝空间内做 RL 的误差」，进而回答应该在什么粒度、用什么目标做中训练。

**核心 idea**：**【在动作抽象空间做中训练】** 理论上证明时间抽象能同时缩小动作集大小 $|Z|$ 和决策视野，于是用一个**时间变分下界 (temporal ELBO)** 把 NTP 重写成"发现隐藏推理 + 回灌模仿"的 EM 过程，让模型自己用 RL 长出可迁移的高层"技能"。

## 方法详解

### 整体框架
方法由"一套理论 + 一个算法"组成。理论侧（第 3 节）把后训练 regret 分解为剪枝误差与 RL 误差，证明（i）剪枝所需专家样本数正比于近似最优动作子集的最小基数 $|Z_\epsilon|$，（ii）动作时间越长 RL 收敛越快——两者都指向"应在时间抽象空间训练"。算法侧（RA3）给 NTP 推一个引入潜变量 $z_{0:T}$ 的时间 ELBO，用 EM 交替优化：E 步用自监督 RL 发现能"解释"专家动作的隐藏推理，M 步在被推理回灌的数据上做 SFT，并用一个 Bernoulli 先验逼迫潜变量在时间上保持一致，从而既是动作抽象、又把 rollout 成本压住。

```mermaid
flowchart LR
    A[专家代码数据 D_E<br/>原始动作=逐行代码] --> B[E 步: 自监督 RL<br/>奖励=专家动作 log-likelihood<br/>采样隐藏推理 z]
    B --> C[回灌数据<br/>code 行 + 推理注释 z]
    C --> D[M 步: NTP/SFT<br/>π 在回灌数据上模仿]
    D -->|EM 迭代 i+1| B
    D --> E[RLVR 后训练<br/>GRPO 收敛更快/上限更高]
```

### 关键设计

**1. Regret 分解 → 把"中训练好不好"变成可证明的量**：本文先把后训练 RL 目标写成最小化跨任务 regret $\min_\pi \mathbb{E}_{M}[V^*_M(s_0)-V^\pi_M(s_0)]$，再证明对任意动作子空间 $Z'$ 它恰好分解为「动作集剪枝误差 $\mathbb{E}[\Delta(M,Z')]$」加「在 $Z'$ 内做 RL 的误差」两项。这一步把模糊的"先验好不好"翻译成了两个清晰的优化目标——剪得准 + RL 跑得动，也直接说明了为什么单看初始准确率/熵不够。剪枝效率定理进一步给出：要达到给定剪枝误差，所需专家 rollout 数 $|D_E|=\Theta(|Z_\epsilon|\log(|Z|/\delta)/\sigma)$，即动作空间越紧凑（$|Z_\epsilon|$、$|Z|$ 越小），同样数据下剪枝越干净，这正是把动作抽象引进来的理论动机。

**2. 时间抽象同时压缩动作集与决策视野**：动作抽象 $z\in Z$ 被定义成 Markov option——一个高层意图执行一段长度 $\tau\sim p(\cdot|s,z)$ 的原始动作序列，而原始 token 只是 $\tau=1$ 的特例。收敛定理给出达到 $\varepsilon$-最优所需迭代数 $N\ge \frac{1}{1-\bar\gamma}\log\frac{R_{\max}}{\varepsilon(1-\bar\gamma)}$，其中 $\bar\gamma=\sup E[\gamma^\tau|s,z]\le\gamma$。动作持续越久 $\bar\gamma$ 越小，每次 Bellman 备份一口气跨过 $\tau$ 步，等价于把有效规划视野缩短、收敛更快。于是"动作抽象"在剪枝效率和 RL 收敛两条线上同时占便宜，构成全文核心论点。

**3. 时间变分下界 + EM：把 NTP 改写成"先长推理再模仿"**：本文给 NTP 目标推了一个 ELBO，$J_{\text{NTP}}(\pi)\ge \mathbb{E}_{z_t\sim q}\big[\sum_t \log\pi(a_t|s_t,z_t)-D_{\mathrm{KL}}(q(z_t|\cdot)\,\|\,p(z_t|\cdot))\big]$，引入一串隐藏意图 $z_{0:T}$ 来解释专家动作。它按 EM 交替优化：**E 步**固定 $\pi$、把"专家动作的对数似然"当作每步奖励做一次 $T$-视野 RL，让采出的潜变量序列去"解释"专家决策（式 4.1）；**M 步**固定 $q$、在被潜变量回灌后的轨迹上做普通 NTP 模仿（式 4.2）。本质就是：模型自己用 RL 把数据里没写出来的"思考"挖出来，再当成额外条件去拟合下一步。

**4. Bernoulli 先验保证时间一致性 + 控成本**：要让 $z_t$ 真的代表一段时间，需要 $z_t=z_{t+1}=\dots=z_{t+\tau}$，本文用先验 $p(z_t|s_t,z_{t-1})=\alpha\,\delta(z_{t-1})+(1-\alpha)\,U(z_t)$ 实现——Dirac 项把质量压在"延续上一个潜变量"上以保持时间一致，均匀项鼓励多样推理。工程上落成两类潜变量：$z=$`<act>` 表示"沿用上一意图、直接出下一行"，以 `<think>` 开头则触发新推理 rollout。Proposition 5.1 把 KL 拆成 Bernoulli-KL 加熵项，只要把固定惩罚 $c$ 加在 $z_t\ne$`<act>` 上，就给"是否要新思考"设了一个阈值：只有当新推理把 log-likelihood 奖励抬高超过 $c$ 才值得想，否则停在 `<act>`。这样 full rollout 只在 `<think>` 时发生，把中训练规模（10 亿 token）下的 RL 推理成本压住，$\alpha\to 1$ 时算法退化为纯 NTP。

## 实验关键数据

设置：Python 逐行代码为原始动作，`<act>=\n`、`<think>=\n#`（即可选地生成一行注释作为高层抽象）。在 Qwen-2.5-1.5B、Llama-3.2-1B、Llama-3.1-8B 上中训练，数据为 3.5M 代码片段 / 1B token；后训练用 GRPO（DeepCoder 代码库 + AReaL-boba-2-RL-Code 7.7K 数据）。

### 主实验表格（中训练 pass@k，平均分摘录）

| 模型 | 方法 | HumanEval p@1 | MBPP p@1 | HE+ p@1 | MBPP+ p@1 | Avg p@1 | Avg p@5 |
|------|------|------|------|------|------|------|------|
| Llama-3.2-1B | Base | 18.9 | 25.8 | 17.1 | 31.5 | 23.3 | 35.3 |
| | NTP | 21.3 | 27.8 | 17.7 | 34.4 | 25.3 | 40.3 |
| | **RA3** | **25.0** | **32.8** | **22.0** | **39.4** | **29.8** | **43.1** |
| Qwen-2.5-1.5B | Base | 37.2 | 38.6 | 32.3 | 43.4 | 37.9 | 54.8 |
| | NTP | 41.5 | 43.4 | 35.4 | 46.6 | 41.7 | 58.7 |
| | **RA3** | **48.2** | **45.8** | **42.7** | **49.7** | **46.6** | **62.2** |
| Llama-3.1-8B | Base | 36.6 | 45.2 | 30.5 | 51.6 | 41.0 | 59.3 |
| | NTP | 48.2 | 48.6 | 42.7 | 51.1 | 47.7 | 63.1 |
| | **RA3** | **50.0** | 48.0 | **44.5** | **53.2** | **48.9** | **64.6** |

RA3 平均比 NTP 高约 4 分、比 base 高约 8 分；M 步的 CE Loss 在三个模型上都明显低于 NTP，说明回灌推理后"下一 token 更好预测"。后训练 RLVR 上，RA3 起点更好，且在 HumanEval+/MBPP+/LiveCodeBench/Codeforces 上**收敛更快、渐近性能更高**。

### 消融实验表格（惩罚 $c$ 的作用，Qwen）

| 惩罚 $c$ | z 平均长度 | full rollout 占比 | HE+MBPP 均值 | 解读 |
|------|------|------|------|------|
| 0.01 | 长 | 高 | 较低 | 几乎每步都想，退化成 NTP 无优势且开销大 |
| 0.05 (默认) | <6 | <40% | 最高 (~40.6) | 性能/算力最佳折中 |
| 0.2 | 短 | 低 | 略降 | 几乎只出 `<act>`，开销≈NTP |

### 关键发现
- E 步 RL 奖励几步内即收敛，故算力可主要分配给 M 步回灌与微调。
- M 步用回灌数据微调显著降低 CE Loss（Fig 3），印证"专家轨迹背后存在隐藏推理、把它用 RL 挖出来后下一步预测更易"。
- 回灌示例（Fig 2）中模型确实抽象出了 dummy head 创建、BFS 遍历这类可复用"技能"注释，证明潜变量在语义上对应高层意图而非噪声。
- 与合成推理 NTP 基线对比（Fig 7）：用外部 LLM 蒸馏注释再 NTP，效果不如 RA3——作者归因于 RA3 的潜变量是模型自学且被证明是对数似然下界，更易被后续 RL 进一步优化、可学习性更好。
- 与 BRiTE 关系：决策视野为 1 时 RA3 退化为 BRiTE；多步问题里直接把整段代码 log-prob 当单步奖励会高方差、训练不稳，RA3 的时间分解解决了这点。

## 亮点与洞察
- **把"中训练玄学"做成可证明的优化目标**：regret 分解 + 剪枝效率 + 收敛率三条定理，第一次给"中训练为什么有用、该在什么粒度做"提供了理论答案，而不是堆指标。
- **option/HRL 思想优雅嫁接到 LLM**：用时间一致潜变量把经典分层 RL 的"option 压缩视野"翻译到 token 世界，且工程上用 `<act>`/`<think>` 双 token 把理论落成可扩展实现。
- **成本可调旋钮**：单个惩罚 $c$ 同时控制推理频率与算力开销，$\alpha\to1$ 平滑退化为 NTP，工程友好。

## 局限与展望
- **只在 Python 代码生成上验证**：原始动作=逐行代码这一设定让 `<act>/<think>` 的语法对齐很自然，但在数学（单步答案）、agentic、自然语言等领域如何定义"行级动作"与抽象粒度仍待验证；作者自己也把数学排除在外。
- **模型规模到 8B 为止**：1B–8B 上成立，但更大模型下"自学推理 vs 直接蒸馏强模型推理"的优劣是否反转未知。
- **理论假设较强**：剪枝/收敛定理依赖近似最优动作子集等理想化设定，与真实 LLM 训练动力学之间仍有 gap。
- **EM + 异步 rollout 的工程复杂度**：相比一行 NTP，RA3 需要多轮 EM、自监督 RL、SGLang 异步采样，落地门槛更高。

## 相关工作与启发
- **LLM Mid-Training**：相比直接蒸馏前沿模型推理来增广数据（有分布漂移、且大规模重标注昂贵），RA3 让模型自学推理，更适合代码这类"互联网人类代码为主、无现成推理标注"的语料。
- **自监督 RL**：以专家动作 log-prob 为奖励的一脉（BRiTE 等），RA3 把单步 ELBO 推广到时间序列，是其多步泛化。
- **Markov Options / 分层 RL**：动作抽象的定义与决策空间分析借鉴 Sutton 的 options 与 Brunskill & Li 的迁移分析，但落点放在"中训练算法设计及其对后训练 RL 的影响"。
- **启发**：把"应该在什么动作粒度学习"作为可证明的设计变量，而非默认 token——这个视角对 agentic、多步工具调用等长视野任务的预训练/中训练都有借鉴意义。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个刻画 mid-training→post-training RL 的理论框架，并由理论直接导出算法，思路自洽且少见。
- 实验充分度: ⭐⭐⭐⭐ 三模型 × 四基准 + RLVR + 消融 + 合成推理对比扎实，但仅限代码生成、规模到 8B。
- 写作质量: ⭐⭐⭐⭐ 理论与算法衔接清晰，定理动机讲得明白；公式密度高，对非 RL 背景读者门槛偏高。
- 价值: ⭐⭐⭐⭐⭐ 给"中训练该怎么做"提供了原理性指导与可扩展实现，对 LLM 推理训练范式有方法论价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reinforcement Mid-Training](reinforcement_mid-training.md)
- [\[ICLR 2026\] DEAS: DEtached value learning with Action Sequence for Scalable Offline RL](deas_detached_value_learning_with_action_sequence_for_scalable_offline_rl.md)
- [\[ICLR 2026\] ExGRPO: Learning to Reason from Experience](exgrpo_learning_to_reason_from_experience.md)
- [\[ICLR 2026\] Learning to Reason Efficiently with Discounted Reinforcement Learning](learning_to_reason_efficiently_with_discounted_reinforcement_learning.md)
- [\[ICLR 2026\] Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty](beyond_binary_rewards_training_lms_to_reason_about_their_uncertainty.md)

</div>

<!-- RELATED:END -->
