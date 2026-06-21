---
title: >-
  [论文解读] Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring
description: >-
  [ICLR 2026][强化学习][LLM 奖励设计] PoRSE 让 LLM 不只生成目标导向奖励，还顺手设计一个"功能可供性状态空间"来驱动任务相关的探索，再用一套在线策略改进流程动态权衡两者，在 24 个机器人操作/运动任务上刷新 SOTA 并首次攻克两个此前无解的难任务。 领域现状：用 RL 训练机器人技能的关键瓶颈…
tags:
  - "ICLR 2026"
  - "强化学习"
  - "LLM 奖励设计"
  - "好奇心探索"
  - "affordance 状态空间"
  - "奖励-策略协同进化"
  - "PPO"
---

# Master Skill Learning with Policy-Grounded Synergy of LLM-based Reward Shaping and Exploring

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=1vXMfIYFZp](https://openreview.net/forum?id=1vXMfIYFZp)  
**代码**: 待确认  
**领域**: 强化学习 / 机器人技能学习  
**关键词**: LLM 奖励设计, 好奇心探索, affordance 状态空间, 奖励-策略协同进化, PPO  

## 一句话总结
PoRSE 让 LLM 不只生成目标导向奖励，还顺手设计一个"功能可供性状态空间"来驱动任务相关的探索，再用一套在线策略改进流程动态权衡两者，在 24 个机器人操作/运动任务上刷新 SOTA 并首次攻克两个此前无解的难任务。

## 研究背景与动机
**领域现状**：用 RL 训练机器人技能的关键瓶颈是奖励函数设计——传统做法靠专家手工调，费时费力。Eureka、ROSKA 等工作让 LLM 从语言指令直接生成可执行奖励代码，大幅降低人工成本，ROSKA 还进一步用奖励-策略协同进化避免每次改奖励都从头训。

**现有痛点**：现有 LLM 奖励方法只盯着"任务文本描述"做设计，生成的奖励**过度目标导向、忽略状态探索**。在自由度高、目标状态稀疏的灵巧操作（如抛接物体）里，策略搜索空间巨大，智能体极易卡在局部最优。传统 RL 用探索 bonus 来缓解，但通用探索 bonus 不区分任务相关性，把资源浪费在无关状态上；而且探索/利用的权衡比例往往一开始就写死，无视后续策略的实际进展。

**核心矛盾**：奖励要"准"（目标导向）又要"广"（鼓励探索），但两者此消彼长，且最优权衡随训练阶段动态变化——固定配比和任务无关的探索都解决不了。

**本文目标**：构建一个统一框架，让 LLM 同时产出任务感知的目标奖励和任务相关的探索机制，并能随策略进展在线调整两者的权衡，无需为每个奖励-探索组合都从头训策略。

**核心 idea**：**[LLM 设计 affordance 探索空间]** 把高维状态压缩成与任务紧密相关的低维离散 affordance 状态空间，在此空间上做基于访问计数的好奇心探索，使探索天然对齐任务目标；**[在线策略改进接地 IPG]** 用实时策略性能反馈引导 LLM 不断提出、评估、淘汰奖励-探索配置并动态调权衡，配合策略继承避免从头训练。

## 方法详解

### 整体框架
PoRSE 把"目标奖励 $R_g$、探索 bonus $R_e$、策略 $\theta$"三者放进一个自强化循环：LLM 先根据任务描述生成目标奖励函数，并构造一个 affordance 状态映射函数把原始状态投影到低维任务相关空间；在该空间用访问计数算出好奇心探索 bonus，两者按权重 $\beta$ 融合成总奖励训练 PPO 策略；策略性能反馈再喂回 LLM，让它迭代精修奖励、调整映射函数、生成候选权重，并通过淘汰-扩张机制和策略继承持续逼近最优配置。

```mermaid
flowchart TD
    T[任务描述 Id + 环境代码 Ie] --> L[LLM deepseek-v3]
    L -->|生成| RG[目标导向奖励 Rg]
    L -->|生成| M[Affordance 状态映射 M]
    M --> RE[访问计数好奇心 bonus Re]
    RG --> F[加权融合 Rtotal = β·Re + (2-β)·Rg]
    RE --> F
    F --> RL[PPO 策略训练]
    RL -->|性能反馈 V θ| IPG[IPG: LEF 淘汰-扩张 + 策略继承]
    IPG -->|精修 Rg / M / β / α| L
    IPG -->|最优策略 θbest| RL
```

### 关键设计

**1. 目标相关探索 bonus：让 LLM 设计 affordance 状态空间。** PoRSE 的第一个突破是把"探索往哪走"也交给 LLM 决定。它用映射函数 $M: S \to S_o$ 把高维原始状态压成低维 affordance 状态空间（AFS），每一维度量当前状态与目标在某种"行为可供性"上的距离——例如 DoorOpenInward 任务里 $S_o=[s_o^d, s_a^l]$，分别是到门把手的欧氏距离和门的角位移。再把连续 AFS 离散化为 $C$ 个状态，按访问频率给出好奇心 bonus：

$$R_e(s_o) = \frac{\lambda}{\sqrt{\sum_{t=1}^{T} \mathbb{I}(s_{o,d}^t = s_{o,d}^c)}}$$

访问越少的状态 bonus 越高。映射函数 $M$ 本身由 LLM 从任务描述自动生成（$M_n = \text{LLM}(I_d, I_e)$），省去逐任务手工定义。因为探索锚定在任务相关的 affordance 维度上，智能体被引导去探"真正有助于完成任务"的状态，而非满世界乱撞。总奖励初步写作 $R_{total}^k = R_e^k + R_g^k$。

**2. 奖励-bonus 协同精修。** 单次生成的奖励和映射函数往往不够好，PoRSE 借鉴 Eureka 的多轮迭代思路，把策略评估数据反馈给 LLM 同时精修两者。对目标奖励仍按 Eureka 式迭代优化；对探索机制则把上一轮最优映射函数 $M_{best}^{n-1}$ 作为参考样本喂回 LLM：

$$M_n = \text{LLM}(I_d, I_e, M_{best}^{n-1}, V(\theta))$$

这让奖励函数和 affordance 映射函数**协同进化**，探索效率随迭代逐步提升，而不是用一个一成不变的探索 bonus（论文实验证明静态 bonus 的 LLMCount 明显更差）。

**3. 在线策略改进接地（IPG）+ 淘汰-扩张过滤（LEF）。** 目标奖励与探索 bonus 的最优配比 $\beta$ 随训练阶段漂移，穷举所有组合从头训不现实。PoRSE 把总奖励写成动态加权形式：

$$R_{total}^k = \beta \cdot R_e^k + (2-\beta) R_g^k, \quad \beta \in [0,2]$$

并让 LLM 根据策略反馈生成一组候选 $\beta$ 值。LEF 借鉴 L-SHADE 的种群线性缩减和 PSO 的扩张思想：并行训练 $N$ 个"探索-目标奖励对"，每 $H$ 个 epoch 淘汰当前最差的一对；淘汰后以幸存者中最优组合为基底，变异扩张出 $J$ 个新 $\beta$，如此交替。这套坐标优化（固定一个变量优化另一个）把纠缠的搜索空间大幅简化。其内在逻辑是**自适应权衡**：当策略停滞或退化就调大 $\beta$ 强调探索以逃离局部最优，当策略稳步逼近目标就调小 $\beta$ 强化目标奖励加速收敛。

**4. 策略继承避免从头训练。** 为复用历史知识，PoRSE 沿用 ROSKA 的策略融合，把最优历史参数 $\theta_{best}$ 与随机策略 $\theta_{random}$ 按融合比 $\alpha$ 混合：

$$\theta_f(\alpha) = \alpha \cdot \theta_{best} + (1-\alpha)\cdot \theta_{random}$$

但 ROSKA 用贝叶斯优化找 $\alpha$，需反复训练采样、代价高。PoRSE 改用同一套 LEF 机制让 LLM 直接给出并变异 $\alpha$（$\alpha_{new}=\text{LLM}(I_d, V(\theta))$），无需收集"比例-性能"样本对，在保持探索-利用平衡的同时显著降低计算成本。训练时在 $\beta$ 搜索和 $\alpha$ 搜索之间交替进行。

## 实验关键数据

### 主实验
- **环境/任务**：24 个机器人技能任务，20 个来自 Bi-DexHands 双手灵巧操作，4 个来自 Isaac Gym（腿足运动 + 机械臂）。策略用 PPO，LLM 为 DeepSeek-V3；$N=5$ 轮、每轮 $K=6$ 个奖励-映射对，单策略训 3000 epoch，5 个随机种子取最优。
- **评价指标**：MTS（Maximum Training Success，训练中达到的最高稀疏奖励）。
- **基线**：Sparse 奖励、Human 专家奖励、Eureka、ROSKA。

| 难度档 | 结果概览 |
|---|---|
| 中等难度（16 任务） | PoRSE 在 **15/16** 任务领先；Pen/Scissors 等 5 个任务 MTS=1.000；GraspAndPlace 达 0.984（比人类基线 0.785 高 25.3%）；仅 Humanoid 运动略输 ROSKA（8.454 vs 8.917） |
| 困难（8 任务） | PoRSE 在 **全部 8 个**难任务领先；DoorCloseOutward/Kettle 达 1.000（Eureka 仅 0.553/0.742）；**首次解出 TwoCatch**（0.349）；BlockStack 0.753 远超 ROSKA 0.148 |

迭代收敛上，简单任务 LiftUnderarm PoRSE 第 5 轮达 0.97（Eureka 0.6 / ROSKA 0.8）；复杂任务 TwoCatch 中 Eureka 与 ROSKA 完全失败（0），PoRSE 达 0.4 且第 1 轮到第 5 轮提升十倍。

### 消融实验

组件消融（节选 MTS，↓为相对完整 PoRSE 的下降）：

| 变体 | Anymal | Franka | BlockStack | TwoCatch |
|---|---|---|---|---|
| PoRSE w/o $R_g$ | −0.128 | 0.883 | 0.328 | 0.190 |
| PoRSE w/o $R_e$ | −0.097 | 0.912 | 0.393 | 0.193 |
| PoRSE w/o $\theta_{fusion}$ | −0.346 | 0.671 | 0.603 | 0.017 |
| PoRSE w/o $R_{ratio}$（固定 $\beta$） | −0.066 | 0.946 | 0.296 | 0.297 |
| PoRSE w/o $\theta_{ratio}$（固定 $\alpha$） | −0.020 | 0.940 | 0.590 | 0.276 |
| **PoRSE（完整）** | **−0.012** | **0.957** | **0.753** | **0.349** |

与静态探索 bonus 的 LLMCount 对比（Tab.3）：PoRSE 在 6 个任务全面碾压，如 TwoCatch 0.349 vs 0.000、BlockStack 0.753 vs 0.140、Franka 0.957 vs 0.706。

AFS 鲁棒性（Tab.4，随机拼装 AFS）：PoRSE-AFS-Random 仍明显强于所有基线，BlockStack 0.680±0.080（Eureka 0.254 / ROSKA 0.148）、PushBlock 0.324±0.034（最好基线 0.069），虽不及用任务相关 AFS 的完整 PoRSE（0.753 / 0.378），说明淘汰机制能剪掉无用维度、对不完美 affordance 设定有较强容错。

### 关键发现
- 去掉任一组件都会掉点，$\theta_{fusion}$（策略继承）对长程优化任务最关键（Franka 0.957→0.671，TwoCatch 0.349→0.017）；$R_e$ 对需探索任务关键（BlockStack 0.753→0.393）。
- 固定 $\beta$ 或 $\alpha$ 都会破坏探索-利用平衡，**交替搜索两个系数**才得最优。
- $\beta$ 配比敏感：单一奖励（纯 $R_g$ 的 Eureka 或纯 $R_e$ 的 LLMCount）在难任务上 0% 成功，PoRSE 在 1.5:0.5 配比下 DoorOpenInward 达 97%、TwoCatch 25%。

## 亮点与洞察
- **把"探索方向"也变成 LLM 可设计对象**：以往 LLM 只管目标奖励，PoRSE 让 LLM 生成 affordance 状态映射，使好奇心探索天然对齐任务，从机理上解决"通用探索浪费在无关状态"的老问题。
- **动态权衡而非固定配比**：用策略实时性能驱动 $\beta$ 的升降，停滞时探、收敛时利，是对"探索-利用权衡随阶段漂移"这一真实现象的直接回应。
- **工程上的降本**：用统一的 LEF 机制同时搜索 $\beta$ 和融合比 $\alpha$，把 ROSKA 昂贵的贝叶斯优化替换为 LLM 引导的淘汰-变异，几乎不增算力即超越前作。
- **AFS 鲁棒性实验**很加分：即便随机拼 AFS 仍强于基线，说明收益不只来自"LLM 恰好设计得好"，淘汰机制本身提供了容错。

## 局限与展望
- **依赖仿真环境**：全部 24 个任务在 Bi-DexHands 和 Isaac Gym 仿真里，未见真机迁移，affordance 状态的可获取性在真实感知噪声下存疑。
- **LLM 与算力成本**：每轮要并行训多个策略并多次调用 DeepSeek-V3，虽与 Eureka/ROSKA 算力相当，但绝对开销仍高，单策略 3000 epoch × 多对 × 5 轮。
- **超参偏多**：$N, K, H, J$ 以及 $\beta/\alpha$ 的搜索调度需人工设定，离"完全免调"还有距离。
- **运动类任务略弱**：Humanoid 双足运动上输给 ROSKA，作者归因于这类任务更偏目标导向、探索增益有限——说明 affordance 探索的收益是任务相关的。
- **展望**：把 affordance 映射扩展到视觉/多模态观测、引入真机闭环、或让 LLM 自适应决定搜索预算，都是自然的下一步。

## 相关工作与启发
- **LLM 奖励设计谱系**：L2R（语言到模块化奖励 + MPC）→ Eureka（进化搜索奖励代码）→ ROSKA（奖励-策略协同进化）→ REvolve（LLM+人类反馈进化）。PoRSE 的位置是首个把"探索机制"也纳入 LLM 设计与协同进化的工作。
- **好奇心驱动探索**：ICM（Pathak）、基于计数的内在奖励（Tang/Kolter&Ng）。PoRSE 的差异在于探索不再独立于任务目标，而是建立在 LLM 设计的任务相关 affordance 空间上。
- **进化/群体优化借鉴**：LEF 的淘汰来自 L-SHADE 的线性种群缩减、扩张来自 PSO，体现了把经典演化算法思想嫁接到"LLM 候选配置筛选"的可行路径。
- **启发**：当 RL 的某个设计维度（奖励、探索、甚至课程）能被结构化表达成 LLM 可生成的代码/映射时，"LLM 生成 + 性能反馈淘汰 + 历史策略继承"这套三段循环就可能复用——PoRSE 给出了在"探索"维度上的一个范例。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把 LLM 从"只设计目标奖励"推进到"同时设计任务相关探索空间并协同进化"，affordance 探索 + 动态权衡的组合有清晰增量，但单看每个零件（LLM 奖励、计数探索、策略融合）都是已有思想的拼装。
- **实验充分度**: ⭐⭐⭐⭐ — 24 任务、4 类基线、5 种子，组件/策略/配比/精修/AFS 鲁棒性五维消融齐全，且首次解出 TwoCatch 有说服力；缺真机验证。
- **写作质量**: ⭐⭐⭐⭐ — 动机—痛点—方法对应清晰，Fig.1/Fig.2 把与前作差异讲得明白；公式与符号偶有冗余、部分图表压缩较狠。
- **价值**: ⭐⭐⭐⭐ — 为"自动奖励设计"补上了长期被忽视的探索维度，对稀疏奖励、高自由度操作任务有实际推动，工程降本路径也可迁移。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reference Grounded Skill Discovery](reference_grounded_skill_discovery.md)
- [\[ICLR 2026\] Learn to Reason Efficiently with Adaptive Length-based Reward Shaping](learn_to_reason_efficiently_with_adaptive_length-based_reward_shaping.md)
- [\[ICLR 2026\] Skill Learning via Policy Diversity Yields Identifiable Representations for Reinforcement Learning](skill_learning_via_policy_diversity_yields_identifiable_representations_for_rein.md)
- [\[ICLR 2026\] Self-Improving Skill Learning for Robust Skill-based Meta-Reinforcement Learning](self-improving_skill_learning_for_robust_skill-based_meta-reinforcement_learning.md)
- [\[ICLR 2026\] TIPS: Turn-Level Information-Potential Reward Shaping for Search-Augmented LLMs](tips_turn-level_information-potential_reward_shaping_for_search-augmented_llms.md)

</div>

<!-- RELATED:END -->
