---
title: >-
  [论文解读] Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs
description: >-
  [ICLR 2026][强化学习][RLVR] 针对"RLVR 到底是真提升了推理能力、还是只提高了采样效率"这场争论，本文提出新指标 CoT-Pass@K（要求答案对且推理过程也对），并用一个 GRPO 的理论框架证明：只要预训练模型具备"对的 CoT 更容易导出对答案"这一逻辑先验，仅靠答案正确性的奖励就会**隐式地**把生成正确推理的概率推上去，从而把基座模型的推理边界真正向外扩展。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "RLVR"
  - "GRPO"
  - "链式思维"
  - "Pass@K"
  - "推理能力边界"
---

# Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=jGbRWwIidy](https://openreview.net/forum?id=jGbRWwIidy)  
**代码**: 验证数据公开于 [HuggingFace: AIME24-25_CoT_Verification](https://huggingface.co/datasets/XumengWen/AIME24-25_CoT_Verification)  
**领域**: LLM推理 / 强化学习 / RLVR  
**关键词**: RLVR、GRPO、链式思维、Pass@K、推理能力边界

## 一句话总结
针对"RLVR 到底是真提升了推理能力、还是只提高了采样效率"这场争论，本文提出新指标 CoT-Pass@K（要求答案对且推理过程也对），并用一个 GRPO 的理论框架证明：只要预训练模型具备"对的 CoT 更容易导出对答案"这一逻辑先验，仅靠答案正确性的奖励就会**隐式地**把生成正确推理的概率推上去，从而把基座模型的推理边界真正向外扩展。

## 研究背景与动机
**领域现状**：DeepSeek-R1 用 GRPO 把长链式思维（CoT）推理跑通后，RLVR（基于可验证奖励的强化学习）成了让 LLM 学会推理的主流范式——模型当策略，生成 CoT 当动作序列，再由确定性验证器只对最终答案的对错给二值奖励。大家寄望它能让模型通过自由探索"从经验里学习"。

**现有痛点**：一批工作发现，RLVR 后的模型确实提高了 Pass@1，但在 Pass@K（采样 K 次至少对一次）上往往追不上、甚至不如基座模型。Yue et al. (2025) 由此提出一个很有冲击力的假设：**所有正确推理路径在基座模型里本就存在，RLVR 只是调高了它们的采样概率，代价是缩小了整体推理容量**。这个假设获得不少支持，但也有反例——有人在竞赛编程上看到 Pass@K 持续提升，有人在不同难度的谜题上看到相反趋势。

**核心矛盾**：争论的根子在于 Pass@K 这个指标本身**不可靠**。数学题答案常常是个简单整数，基座模型完全可能"推理过程是错的，却撞对了答案"，尤其是难题反复采样多次后总能蒙中一次。于是 Pass@K 把"蒙对"也算成功，自然让基座模型显得很强，掩盖了 RLVR 的真实贡献。

**本文目标**：（1）找一个能区分"真会做"和"蒙对"的评测指标；（2）从理论上解释清楚——为什么只奖励答案对错，却能让推理过程变对；（3）用训练动态和 CoT 质量实验，验证这种推理提升是真实发生的。

**切入角度**：作者认为评判推理必须把 CoT 的对错也纳入，而不是只看答案；同时他们抓住 LLM 与传统 RL 的关键区别——LLM 预训练后带有强知识/逻辑先验，"对的推理"和"错的推理"导向正确答案的概率天然不同，这个差距正是 GRPO 能起效的杠杆。

**核心 idea**：用"答案对 + 推理对"双重判定的 CoT-Pass@K 揭示被掩盖的能力提升，再用"逻辑先验"假设证明 GRPO 梯度会单调推高正确 CoT 的生成概率，把 RLVR 的有效性从经验观察上升为可解释的机制。

## 方法详解

### 整体框架
本文不是提出新算法，而是一套"重新评测 + 理论解释 + 训练动态验证 + 质量复核"的论证体系，用来回答 RLVR 是否真正强化了推理。它分四块协同：先用新指标 **CoT-Pass@K** 把"蒙对"从成功里剔除，重新测量数学/代码任务上的推理边界，看 RLVR 后模型与基座模型是否仍有稳定差距；再给出一个 **GRPO 隐式激励** 的理论框架，从"逻辑先验"假设推出"正确 CoT 的期望优势为正、错误 CoT 为负"，解释为什么只靠答案奖励就能扶持正确推理；接着复现 DAPO 训练、记录关键指标，验证训练动态与定理一致、且从早期就在提升；最后用 **SFT 反推 CoT 质量**——拿不同阶段模型生成的 CoT 去微调同一基座，用泛化性能反向衡量这些 CoT 的好坏。四块层层互证：指标负责"看见"，理论负责"解释为什么"，训练动态负责"验证何时发生"，SFT 负责"确认质量真的变好"。

整篇的对象是基座模型 Qwen2.5-32B 与其 RLVR 版本 DAPO-Qwen-32B（数学），以及蒸馏模型 DeepSeek-R1-Distill-Qwen-7B 与其 RLVR 版本 AceReason-Nemotron-7B（代码）。

### 关键设计

**1. CoT-Pass@K：把"蒙对"从成功里剔除的可靠推理指标**

Pass@K 的致命缺陷是只看答案，导致基座模型靠多次采样蒙中简单整数答案就被算成"会推理"，从而虚高、追平甚至反超 RLVR 模型。本文定义 CoT-Pass@K：只有当最终答案 $I_{\text{Ans}}(a_i)=1$ **且**中间推理 $I_{\text{CoT}}(c_i)=1$ 同时成立才算成功。其中 $I_{\text{CoT}}$ 定义为"中间 token 表达了导向正确答案所必需且准确的逻辑"。难点是数学 CoT 冗长、无结构、知识密集，难以规模化判对，作者采用 **LLM-as-a-CoT-Judge** 范式，用轻量专用模型 DeepSeek-R1-0528-Qwen3-8B 对每条 CoT 多次验证，并给出三种聚合策略——any-correct（至少一次判对）、all-correct（全部判对）、majority-correct（多数表决），三者构成结果的上下界阴影带，同时人工抽查 Pass@K 为正但 CoT-Pass@K 为零的样本以确保可靠。换用这个指标后，在 AIME 2024/2025 上 RLVR 模型与基座模型在所有 K（直到 1024）上都保持一致且显著的差距——尤其 AIME 2025 因发布于基座模型训练截止之后、无数据污染，差距最明显。这直接戳破了"基座模型 Pass@K 能追上"的表象：它追上的是蒙对，不是会做。

**2. 逻辑先验假设 + 定理 1：只奖励答案为何能扶持正确推理**

这是全文的理论内核，回答"reward 只看答案、模型却学会了对的推理"这个反直觉现象。关键区别在于：传统 RL（如围棋）里每个合法动作都有效，只要轨迹回报高就强化；而预训练 LLM 可能用错误 CoT 撞对答案。作者引入 **逻辑先验（Logic Prior）** 假设——正确 CoT 比错误 CoT 更容易导出正确答案：

$$P(I_{\text{Ans}}=1 \mid I_{\text{CoT}}=1)=\alpha > P(I_{\text{Ans}}=1 \mid I_{\text{CoT}}=0)=\beta$$

在 GRPO 中，组内 $G$ 个回答的优势为 $\hat{A}(y_i)=\dfrac{R(y_i)-\mu_Y}{\sigma_Y}$，奖励 $R(y_i)=I_{\text{Ans}}(a_i)$ 只由答案决定。**定理 1** 证明：在该假设与可学习组（$\sigma_Y>0$）、足够大 $G$ 的条件下，

$$\mathbb{E}\big[\hat{A}(y_i)\mid I_{\text{CoT}}(c_i)=1\big]>0,\qquad \mathbb{E}\big[\hat{A}(y_i)\mid I_{\text{CoT}}(c_i)=0\big]<0$$

即正确 CoT 的期望优势为正、错误 CoT 为负，于是策略梯度会单调推高生成正确 CoT 的概率 $p_c^\theta$。驱动量正是 $\alpha-\beta>0$：训练中 $\alpha$ 升高（推理更稳健）、$\beta$ 下降（伪相关与知识错误减少），差距越拉越大，正向激励加速；当 $p_c\to 1$ 时 $(\alpha-\beta)$ 趋近 1，正确 CoT 的期望优势趋零，保证收敛。这套推导把"为什么有效"讲清了：哪怕初始 $p_c^\theta$ 很低（常蒙对），只要预训练打下的知识/逻辑先验在，GRPO 就会把概率往正确 CoT 上挤。

**3. 训练动态指标：验证激励从早期就发生且能泛化**

理论需要落到训练过程里检验。作者复现 DAPO（R1-Zero 配方）训练，对每个 prompt 的 $G$ 个回答定义两个关键指标：答案通过数 $C=\sum_i I_{\text{Ans}}(a_i)$ 与"CoT 和答案都通过"数 $D=\sum_i I_{\text{CoT}}(c_i)\cdot I_{\text{Ans}}(a_i)$，进而估计产生正确答案的概率 $P(CA)(q)=\tfrac{C}{G}$、以及"答对时推理也对"的概率 $P(CC\mid CA)(q)=\tfrac{D}{C}$。观察到：对已被充分优化的训练题，$P(CA)$ 几乎到 1 的同时 $P(CC\mid CA)$ 也在上升——即模型不只优化最终奖励，确实在隐式提升推理正确率，与定理 1 吻合。在测试集上，Pass@K 与 CoT-Pass@K 的泛化提升从训练**最早期**就出现，说明推理边界的扩展不是后期才挤出来的。这一设计的意义在于：它把"隐式激励"从纸面定理变成可观测、可复现的训练信号。

**4. SFT 反推 CoT 质量：用泛化性能给推理打分**

LLM-as-a-Judge 只能识别"明显错误"，无法度量 CoT 的整体质量好坏。作者换一个学习视角：好 CoT 应当让在它上面做监督微调的模型泛化更好。于是从同一基座 Qwen2.5-32B 出发，分别用基座 CoT、不同 RLVR 阶段（如 step 0–20、180–200）模型产生的 CoT 做 SFT，用测试集 Pass@1 当代理指标。结果显示：随 RLVR 进程推进，对应 CoT 训出的模型泛化稳步变好；用 DAPO CoT 做 SFT 几乎能复现 DAPO-Qwen-32B 的 Pass@1——这意味着只要有足够训练题和后 RLVR 模型的 CoT 数据，**仅靠 SFT 就能低成本复刻一个昂贵 RLVR 模型的能力**。更有意思的是，无论 CoT 里是否含可识别错误，越靠后期的 CoT 质量越高，说明 RLVR 后期那些"含瑕疵"的 CoT 整体也在变好。这个设计独立验证了第 1、2 点的结论：推理质量确实被 RLVR 从根本上提升了。

## 实验关键数据

### 主实验（推理边界是否真扩展）

| 任务/基准 | 对比对象 | 指标 | 结论 |
|------|------|------|------|
| AIME 2024 / 2025（数学） | DAPO-Qwen-32B vs Qwen2.5-32B | Pass@K | 基座随 K 增大追平甚至反超（符合 Yue 等的观察） |
| AIME 2024 / 2025（数学） | 同上 | **CoT-Pass@K** | 所有 K（≤1024）上 RLVR 模型稳定显著领先，AIME 2025 差距最明显 |
| MATH-500 / AMC23（数学） | 同上 | CoT-Pass@K | RLVR 效果不明显（题目较简单或疑似在预训练数据中） |
| Minerva（数学） | 同上 | CoT-Pass@K | 无提升，疑因训练-测试域不匹配（含物理、自由格式答案） |
| LiveCodeBench v1–v6（代码） | AceReason-Nemotron-7B vs DeepSeek-R1-Distill-Qwen-7B | Pass@K | 多数版本上 RLVR 模型持续领先；代码靠执行验证，几乎无法蒙，Pass@K 本身可靠 |

### 训练动态与 CoT 质量分析

| 分析 | 关键指标 | 发现 |
|------|---------|------|
| 优化效果 | $P(CA)(q)$、$P(CC\mid CA)(q)$ | 充分优化题 $P(CA)\to1$ 的同时 $P(CC\mid CA)$ 上升，验证隐式激励 |
| 泛化行为 | Pass@K / CoT-Pass@K（测试集） | 从训练早期即提升，推理边界一开始就在扩展 |
| DAPO 局限 | $P(CC\mid CA)$ 中位数 | 400 步后 $P(CA)\approx1$（题不再可学），但仍约 0.7，残留瑕疵 CoT 无法仅靠答案奖励纠正 |
| CoT 质量（SFT 代理） | 测试集 Pass@1 | 用 DAPO CoT 做 SFT 几乎复现 DAPO-Qwen-32B 的 Pass@1，质量随阶段单调提升 |

### 关键发现
- **指标决定结论**：同一对模型，Pass@K 说"RLVR 没用甚至有害"，CoT-Pass@K 说"RLVR 真扩展了边界"——争论的核心其实是评测口径。代码任务因可执行验证、蒙不了，Pass@K 才可靠。
- **基座模型的 Pass@K 是"假强"**：难题答案简单，反复采样能蒙中，CoT 却是错的；这正是 Pass@K 让基座虚高的根源。
- **DAPO 的天花板**：当训练题被刷到全对，GRPO 优势无法计算（组内方差为 0），剩下约 30% 的瑕疵 CoT 失去纠正信号——这是只用答案奖励的内在局限，作者推测也是 R1-Zero 可读性差、多语言混杂的根因。

## 亮点与洞察
- **一个指标终结一场争论**：把"答案对"升级为"答案对 + 推理对"，CoT-Pass@K 用极小的概念改动就揭穿了 Pass@K 的系统性偏差，是典型的"换把尺子，结论就翻转"的洞察。
- **逻辑先验是 LLM-RL 与传统 RL 的分水岭**：定理把 $\alpha-\beta>0$ 这个看似朴素的不等式，转成"答案奖励隐式监督推理过程"的严格结论，解释了一个反直觉现象，且对"何时失效"（先验不成立时强化错误 CoT）也给了清晰边界。
- **SFT 可低成本复刻 RLVR**：用后 RLVR 模型的 CoT 做 SFT 就能逼近其 Pass@1，提示一条"昂贵 RL 跑一次、便宜蒸馏给多个"的工程路线。
- **LLM-as-a-CoT-Judge 的可迁移性**：在数学/代码这类无结构推理上规模化判 CoT 对错的范式，可迁移到其它难以程序化验证推理的任务，但也呼吁为"LLM 验证器可靠性"建专门的评测基准。

## 局限与展望
- **结论依赖 LLM 验证器**：CoT-Pass@K 的可靠性建立在 DeepSeek-R1-0528-Qwen3-8B 判对/判错的准确度上，作者虽做人工抽查与多策略上下界，但验证器本身的系统偏差难以完全排除。
- **逻辑先验假设并非总成立**：当基座模型带有顽固偏见或致命知识错误，且这些错误 CoT 恰好导出正确答案时，GRPO 会反向强化错误推理——这是理论框架自承的失效模式，也是 R1-Zero 行为退化的可疑根源。
- **只覆盖中小模型与数学/代码域**：受 RLVR 算力所限，实验集中在 ≤32B 模型与数学、竞赛编程；更大模型、更开放的推理任务上是否成立尚待验证。
- **DAPO 后期失去可学信号**：题目刷满后无法再纠正残留瑕疵 CoT，提示需要超越"纯答案奖励"的过程级监督或课程设计。

## 相关工作与启发
- **vs Yue et al. (2025)（Pass@K 假设）**：他们用 Pass@K 得出"RLVR 只提采样效率、缩推理容量"的悲观结论；本文指出该指标会把"蒙对"算成功，换用 CoT-Pass@K 后结论反转——RLVR 确实扩展了推理边界。两者的分歧本质是评测口径而非事实。
- **vs Chen et al. (2025b)（代码 Pass@K 提升）**：他们在竞赛编程上报告了持续的 Pass@K 提升但未覆盖数学；本文把实验扩到更多 LiveCodeBench 版本并补充 Skywork-OR1，统一解释了"代码可靠、数学需 CoT 校正"的差异来源。
- **vs 合成可验证推理任务（Arcuschin et al. 2025 等）**：那类工作构造易于验证 CoT 对错的合成任务，本文则主张用 LLM-as-a-CoT-Judge 处理数学/代码这类真实无结构推理，并强调要为验证器可靠性建立基准。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ CoT-Pass@K + 逻辑先验定理把一场经验争论提升到机制层面，角度独到。
- 实验充分度: ⭐⭐⭐⭐ 数学/代码双域、多基准、训练动态 + SFT 反推质量四重互证，但模型规模与任务域偏窄。
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，理论与实验衔接紧密，失效模式也交代到位。
- 价值: ⭐⭐⭐⭐⭐ 直接回应"RLVR 是否真有用"这一核心争论，并给出可复刻 RLVR 的 SFT 工程启示。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains](rubrics_as_rewards_reinforcement_learning_beyond_verifiable_domains.md)
- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] From Verifiable Dot to Reward Chain: Harnessing Verifiable Reference-based Rewards for RL of Open-ended Generation](from_verifiable_dot_to_reward_chain_harnessing_verifiable_reference-based_reward.md)
- [\[ICLR 2026\] RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewards for Robust Long-Horizon Agents](rlvmr_reinforcement_learning_with_verifiable_meta-reasoning_rewards_for_robust_l.md)
- [\[ICLR 2026\] Lookahead Tree-Based Rollouts for Enhanced Trajectory-Level Exploration in Reinforcement Learning with Verifiable Rewards](lookahead_tree-based_rollouts_for_enhanced_trajectory-level_exploration_in_reinf.md)

</div>

<!-- RELATED:END -->
