---
title: >-
  [论文解读] All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning
description: >-
  [ICLR2026][强化学习][RLHF] 这篇论文从信息几何、受控实验和复杂度直觉三个角度解释为什么语言模型微调中“两阶段奖励模型 + 在线 RL”常常优于直接离线最大似然，核心结论是 RL 的价值不在于凭空创造新信息，而在于借助更容易学习的 verifier 把策略搜索限制到一小类由简单奖励诱导的生成器。
tags:
  - "ICLR2026"
  - "强化学习"
  - "RLHF"
  - "偏好优化"
  - "DPO"
  - "奖励模型"
  - "生成-验证差距"
---

# All Roads Lead to Likelihood: The Value of Reinforcement Learning in Fine-Tuning

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=sCL5mSTpKm](https://openreview.net/forum?id=sCL5mSTpKm)  
**代码**: 待公开  
**领域**: 强化学习 / 语言模型偏好微调  
**关键词**: RLHF, 偏好优化, DPO, 奖励模型, 生成-验证差距

## 一句话总结
这篇论文从信息几何、受控实验和复杂度直觉三个角度解释为什么语言模型微调中“两阶段奖励模型 + 在线 RL”常常优于直接离线最大似然，核心结论是 RL 的价值不在于凭空创造新信息，而在于借助更容易学习的 verifier 把策略搜索限制到一小类由简单奖励诱导的生成器。

## 研究背景与动机
**领域现状**：大模型后训练里的偏好微调通常有两条路线：一条是 DPO、IPO、SLiC-HF 这类离线方法，直接在偏好对上优化策略参数；另一条是传统 RLHF，先用偏好数据训练奖励模型，再让策略通过 PPO、online DPO 等在线 RL 过程最大化这个奖励。工程实践中，许多强模型仍然采用后一种更复杂的两阶段路线。

**现有痛点**：从第一性原理看，这件事反而有些奇怪。偏好数据本身已经告诉我们哪个 completion 更好，如果目标只是提高偏好 completion 的似然，为什么不直接做最大似然或分类式偏好优化？更麻烦的是，奖励模型只是偏好数据的函数，on-policy 采样也只是从当前策略生成新文本；按照数据处理不等式，这两个步骤都不能产生新的真实人类偏好信息。

**核心矛盾**：理论直觉说“绕一圈奖励模型再 RL”不应比直接离线学习更有信息优势，但实证上 online PFT/RLHF 往往更强。作者要解释的不是“在线数据有没有用”这种表层问题，而是：在相同偏好数据、相同初始 SFT 模型、甚至相同 DPO loss 的条件下，为什么经过奖励模型的路线仍然能让策略更好？

**本文目标**：论文把这个问题拆成三步。第一步，用信息几何把离线 PFT 和在线 PFT 写成同一个似然目标下的两种投影，说明理想条件下它们应当等价。第二步，设计受控实验排查几种常见解释，例如 online samples 自带价值、离线正则化失败、在线优化更容易、奖励模型能用更多数据或 OOD 泛化天然更好。第三步，提出并检验“生成-验证差距”假说：有些任务中判断答案好坏比生成好答案简单得多。

**切入角度**：作者把 policy 看成 generator，把 reward model 看成 verifier。若 verifier 的函数复杂度显著低于生成优质答案所需的策略或 Q 函数，那么用有限偏好数据先学 verifier 会比直接学 generator 更省样本。RL 并不提供新标签，而是把已经学到的简单 verifier 转换成其 soft-optimal policy。

**核心 idea**：两阶段在线微调的真正价值，是先学习一个相对简单的奖励模型，再通过 RL 只在“简单 verifier 的最优策略集合”里找策略，从而把离线 PFT 面对的全策略空间搜索问题缩小成一个更容易的数据-计算权衡问题。

## 方法详解

### 整体框架
这篇论文的方法不是提出一个新的 RLHF 算法，而是建立一套解释框架，并用实验去反驳或支持不同假说。整体流程可以概括为：先把偏好微调统一成 KL 正则化的似然最大化问题，再证明在奖励类和策略类同构时 online RLHF 与 offline DPO/MLE 的最优解应当一致；接着在 TL;DR summarization 上做严格对照实验，确认现实中 online DPO 仍优于 offline DPO；最后用生成-验证差距解释这个差异，并通过缩短生成 horizon、改用 ROUGE-L 这类更复杂 verifier 的实验检查该解释是否站得住。

在形式化上，作者把 completion 视作轨迹 $\xi$，策略 $\pi$ 在自回归 MDP 中产生轨迹分布 $P_\pi(\xi)$。偏好数据由同一 prompt 下的成对轨迹 $(\xi_i^+, \xi_i^-)$ 构成。离线 PFT 直接在策略类 $\Pi$ 上最大化偏好似然，而在线 PFT 先在奖励类 $R$ 上拟合 Bradley-Terry 奖励模型，再解一个带熵或 KL 正则的 soft RL 问题。两条路线表面上不同，实质上都在试图解释偏好数据，只是一个直接投影到策略空间，一个先投影到奖励空间再反向投影回策略空间。

经验部分围绕 Pythia 系列模型和 TL;DR summarization 展开。作者尽量固定混杂因素：offline 和 online 都使用 DPO loss，reward model 与 policy 从同一个 SFT checkpoint 开始，使用同一份偏好数据；online DPO 只是从 SFT 或 DPO policy 采样 25 个 completion，由同一个 global RM 排序后取 top/bottom 作为新的偏好对。这样如果 online 仍然更好，差异就很难简单归因于 loss、训练步数或额外人工信息。

### 关键设计
**1. 信息几何等价：先证明“理论上不该有差距”**

论文先把 online/offline PFT 放进统一目标中。偏好学习的核心是让数据分布下的偏好更可能，同时用熵或到参考策略的 KL 做正则。对 reward model，作者使用 Bradley-Terry 形式 $P_r^{BT}(\xi_1 \succ \xi_2 \mid s_0)=\sigma(r(\xi_1)-r(\xi_2))$；对 policy，则把 trajectory-level reward 写成 token log probability 的和 $r_\pi(\xi)=\sum_h \log \pi(a_h\mid s_h)$。这一步说明 DPO 这类离线方法本质上是在学习一个 local reward model，只是这个 reward 由策略 logits 隐式给出。

接着作者证明 soft RL 可以看成从奖励诱导的最优轨迹分布 $P_r^*(\xi\mid s_0)\propto \exp(r(\xi))$ 到策略诱导分布 $P_\pi$ 的 reverse KL projection。于是 online RLHF 变成两步投影：先用 forward KL/MLE 把数据投影到 reward class，再用 reverse KL 把 reward 的 soft-optimal distribution 投影到 policy class。若 $R=R(\Pi)$，即奖励类和策略诱导的 local reward 类表达同一组函数，并且优化足够理想，那么两阶段 RLHF 与直接 MLE/DPO 有相同最优解。这个结论很关键，因为它先排除了“RLHF 当然更强”的直觉，把后续实证差距变成一个必须解释的反常现象。

**2. 受控反证：把 online DPO 的优势从常见解释中剥离出来**

为了让解释不落入“也许只是 PPO 比 DPO 好”或“也许只是多训练了一轮”的模糊说法，论文把 offline DPO 和 online DPO 控制到几乎只差训练数据。online DPO 从当前策略采样 25 个 completion，用同一 global RM 排序，取最高和最低样本形成偏好对，再用同一个 DPO objective 训练；offline DPO 则用原始偏好数据训练。所有策略和奖励模型都从同一 SFT checkpoint 起步，超参也保持一致。

在这个设置下，online DPO 仍然稳定优于 offline DPO。作者随后逐个检查五类替代解释。若 online sample 本身提供新信息，它会违背数据处理不等式，因为标签仍来自同一份数据训练出的 RM；若问题只是离线 DPO 正则化到 $\pi_{ref}$ 不好，那么相同正则的实验不应仍有差距；若 online 只是优化更容易，额外 prompt augmentation 应显著提升，但结果几乎不变；若 global RM 的优势来自能吃更多、更广数据，那么用 SFT policy 生成并由 gpt-4o 标注的窄分布数据后差距应缩小，但 online DPO 仍有明显提升；若只是 OOD 泛化更好，问题又会退回到为什么 global RM 在分布内 validation likelihood 就更好。

**3. 生成-验证差距：把两阶段 RLHF 解释为受约束的 proper learning**

论文最终支持的假说是 H6：在很多后训练问题中，判断一个答案好不好比生成一个好答案简单。奖励模型是 verifier，策略是 generator；如果 underlying reward function 可以由较浅或较简单的函数表示，而 soft-optimal policy/Q function 需要编码更复杂的多步生成结构，那么用偏好数据直接拟合 policy 等价于在一个更大的复杂函数类里找 generator，而先学 RM 再 RL 等价于先找到简单 verifier，再只考虑这些 verifier 的最优策略集合 $\Pi(R_{sim})$。

作者把这个说法形式化为定理：若 $R_{sim}\subset R$ 是简单奖励模型集合，$\Pi(R_{sim})$ 是这些奖励模型诱导的 soft-optimal policies，那么在 RL 的 reverse KL projection 不损失的条件下，RLHF 恢复的是 $\Pi(R_{sim})$ 上的最大似然解。也就是说，online PFT 并没有摆脱 likelihood；它只是通过奖励模型这条路，把“在整个策略空间 $\Pi$ 中搜索”改成“在简单 verifier 的最优策略子集里搜索”。这也是标题“All Roads Lead to Likelihood”的含义：目标仍是似然，但 RL 给了一条穿过简单 verifier 的捷径。

**4. 复杂度直觉：local reward 更像 Q function 而不是普通 reward**

一个容易误解的点是：如果 reward 和 policy 在 soft RL 中同构，为什么学习 reward 会比学习 policy 省样本？作者的回答是，同构并不意味着两个端点同样容易表示。DPO 隐式学习的 local RM 更接近 soft Q function，因为 token logits 需要在每个 prefix 上表达“从这里继续生成会带来多好结果”。相比之下，global RM 只需对完整 completion 打分。迷宫例子很直观：reward 只要标记目标位置，Q function 却要在每个格子上编码通往目标的路径价值；horizon 越长，Q function 覆盖的状态越广，也越复杂。

这解释了为什么 online PFT 是一种数据换计算的策略。离线 DPO 直接从有限偏好数据学习复杂的 policy/Q-like object，需要支付统计代价；online PFT 先学习较简单的 reward，再用 RL 计算由它诱导的策略，更多支付的是计算代价。对需要长程规划、多步推理或 agentic 行为的任务，这种差距可能更明显；对接近 bandit 或 verifier 与 generator 同样复杂的任务，差距则应减弱。

### 损失函数 / 训练策略
理论部分的核心目标可以写成一个 KL 正则化的偏好似然问题。简化地看，策略要最小化数据偏好分布与策略诱导偏好分布之间的 forward KL，同时通过熵或到参考策略的 reverse KL 避免偏离过远。reward model 的训练是标准 logistic regression：

$$
\hat r_{mle}=\arg\max_{r\in R}\sum_i \log \sigma(r(\xi_i^+)-r(\xi_i^-)).
$$

离线 DPO/local RM 则把 $r(\xi)$ 替换为策略 log probability 的轨迹和：

$$
r_\pi(\xi)=\sum_h \log \pi(a_h\mid s_h),
$$

从而直接优化 preferred completion 相对 dispreferred completion 的似然差。在线阶段给定训练好的 global RM 后，解的是 soft RL：

$$
\pi_r^*=\arg\max_{\pi\in\Pi}\mathbb{E}_{\xi\sim\pi}[r(\xi)]+H(\pi),
$$

或带参考策略的 KL 正则版本。实验里，global RM 训练使用 logistic loss，batch size 64、learning rate $3\times10^{-6}$、AdamW、cosine decay；DPO 训练 batch size 128、learning rate $3\times10^{-7}$、$\beta=0.05$、linear decay。online DPO 每个 prompt 采样 25 个 completion，按 RM 排序取 top/bottom，不做连续在线采样，而是批量生成一次，以减少计算并保持 offline/online 对照干净。

## 实验关键数据

### 主实验
主实验在 TL;DR summarization 上比较 SFT、offline DPO 和 online DPO，评价指标是 gpt-4o 相对人工参考摘要的 winrate。即使使用相同偏好数据、相同 SFT 初始点和相同 DPO loss，online DPO 仍显著优于 offline DPO；更有意思的是，先做 DPO 再用同一 RM 进行 online DPO，还能在没有新增人工反馈的情况下继续提升。

| 模型规模 | 方法 | gpt-4o Winrate(↑) | 说明 |
|--------|------|------------------|------|
| Pythia-1.4B | SFT | 26.2 | 只做监督微调 |
| Pythia-1.4B | DPO | 49.7 | 原始偏好数据上的离线 DPO |
| Pythia-1.4B | DPO (2x) | 52.2 | 离线 DPO 多训一轮 |
| Pythia-1.4B | Online DPO (SFT) | 56.1 | 从 SFT 采样，由 RM 排序 |
| Pythia-1.4B | Online DPO (DPO) | 59.3 | 从 DPO 策略采样，由 RM 排序 |
| Pythia-2.8B | SFT | 30.5 | 更大 SFT baseline |
| Pythia-2.8B | DPO | 54.9 | 离线 DPO |
| Pythia-2.8B | Online DPO (SFT) | 60.8 | online 仍明显更强 |
| Pythia-2.8B | Online DPO (DPO) | 59.4 | 继续提升但略低于从 SFT online 的设置 |

另一组关键结果是“缩小或消除生成-验证差距”后的表现。如果 H6 正确，当生成不再显著难于验证时，online PFT 的优势应该减弱。作者用两个设置验证这一点：一个是两词摘要，显著缩短 horizon；另一个是 ROUGE-L reward，它需要隐藏参考摘要，verifier 本身也变复杂。

| 设置 | 方法 | 指标 | 数值 | 结论 |
|------|------|------|------|------|
| Two-Word TL;DR | SFT | gpt-4o Winrate(↑) | 6.3 | 极短生成任务下 baseline 很弱 |
| Two-Word TL;DR | DPO | gpt-4o Winrate(↑) | 21.9 | 离线 DPO 明显提升 |
| Two-Word TL;DR | Online DPO (DPO) | gpt-4o Winrate(↑) | 23.2 | online 只多约 1.3 个点 |
| ROUGE-L TL;DR | DPO | Val. ROUGE-L(↑) | 0.354 | 离线 DPO 基线 |
| ROUGE-L TL;DR | Online DPO (DPO) | Val. ROUGE-L(↑) | 0.352 | online 没有提升 |
| ROUGE-L TL;DR | DPO (2x) | Val. ROUGE-L(↑) | 0.358 | 多训离线 DPO 反而略升 |

### 消融实验
论文的消融并不是传统模块拆卸，而是围绕不同解释假说做 falsification。下面几组最能说明问题：prompt augmentation 几乎不提高 online DPO；窄分布 gpt-4o 标签下 online 仍能提升；global RM 在 validation likelihood 和 BoN 上优于 local/DPO RM，但这更像生成-验证差距的表征，而不是独立根因。

| 假说 / 设置 | 关键结果 | 对解释的影响 |
|-------------|----------|--------------|
| Prompt augmentation | 1.4B Online DPO 从 56.1 到 56.6；2.8B 从 60.8 到 61.2 | 额外冗余 prompt 只带来极小变化，不支持“online 样本只是提供更多约束所以优化更容易” |
| GPT-label 窄分布数据 | 1.4B DPO 为 56.8，Online DPO (SFT) 为 58.8，Online DPO (DPO) 为 65.2 | 即使偏好数据只来自 SFT policy 且由 gpt-4o 标注，online on top of DPO 仍显著提升，不支持“global RM 只是能用更广数据” |
| RM validation likelihood | 1.4B: DPO 0.526、Local 0.567、Global 0.593；2.8B: DPO 0.545、Local 0.598、Global 0.610 | global RM 分布内拟合更好，说明 local/Q-like 表达更难学 |
| BoN OOD 排序 | 高 $N$ 时，global/local/DPO RM 的 BoN 表现与 validation likelihood 排序一致 | OOD 更好可能来自分布内 margin 更好，但根因仍需解释为什么 global RM 更容易学 |
| RM 尺度变化 | 用 1.4B、2.8B、6.9B global RM 给同一 policy 做 BoN，性能差异很小；扩大 policy 则更有用 | verifier 似乎可由更小模型近似，而 generator 扩大后收益更明显，支持“验证比生成简单” |

### 关键发现
- 在理想同构条件下，online RLHF、offline MLE 和 DPO 都指向同一类似然最优解；如果现实中 online 更好，原因必须来自这些理想条件之外的统计或表示复杂度差异。
- 严格控制 loss、初始 checkpoint、偏好数据和训练超参后，online DPO 仍然优于 offline DPO，说明“PPO 技巧更好”“多训一步”“使用额外人工信息”都不是充分解释。
- global RM 比 local/DPO RM 更容易在验证集上拟合偏好，也在 BoN 排序中表现更好；这与“完整 completion 打分比 token-level Q-like credit assignment 更简单”的解释一致。
- 当作者主动缩短 horizon 或让 verifier 变得和 generator 一样复杂时，online DPO 的优势基本消失，这正是生成-验证差距假说的可检验预测。
- 论文把 RLHF 的价值表述成“数据换计算”：先用少量偏好数据学简单 reward，再用计算密集的 RL 把 reward 转成 policy，而不是直接用偏好数据学复杂 policy。

## 亮点与洞察
- 最有价值的地方是它没有把 RLHF 的成功简单归因于“在线交互更强”，而是先用等价定理把这种直觉压低：如果 reward/policy class 真同构、优化理想、表示复杂度无差别，那么两阶段路线不应凭空赢。
- “local RM 更像 Q function”这个观察非常有启发。很多 DPO 解释会说“policy secretly contains reward”，但这篇论文提醒我们：token logits 要对所有 prefix 做 credit assignment，它隐含的是面向生成过程的价值函数，学习难度可能远高于完整回答级别的 verifier。
- 生成-验证差距把 RLHF、inverse RL、reasoning verifier、agent process reward model 串到了一起。只要任务是“验证容易、生成难”，先学 verifier 再搜索/规划/采样就可能比直接 imitation 或 preference MLE 更省数据。
- 论文对替代假说的处理比较克制。作者没有声称完全排除所有因素，而是说在他们的受控设置下最少证据反对 H6；这种 Popper 式写法比强行宣布“找到唯一原因”更可信。
- 对实践的启发是：后训练资源有限时，可能应优先投资高质量 verifier/reward model 和高效 RL/搜索过程，而不是只把偏好对直接喂给 policy。尤其在长程推理、多轮 agent、机器人等 horizon 更长的问题上，两阶段路线可能更有优势。

## 局限与展望
- 实验主要集中在 TL;DR summarization 和 Pythia 模型族上，虽然作者引用了 reasoning、SFT、inverse RL 的相关现象，但核心实证还没有覆盖真正的大规模 instruction-following、数学推理或 agentic workflow。
- 评价大量依赖 gpt-4o winrate。它适合做相对比较，但仍然是模型裁判，不等同于人类偏好；若 reward model 和评价模型共享某些偏置，结论可能会被放大或压缩。
- H6 是一个很有说服力的机制假说，但“reward function 更简单”在深度网络里并不容易直接测量。论文用模型大小、validation likelihood、horizon 缩短等间接证据支持它，后续仍需要更直接的复杂度度量或可控合成任务。
- 作者讨论了 reward model 无法表达 intransitive preferences 的问题，但没有系统实验。现实人类偏好常常多目标、群体异质且上下文相关，简单标量 reward 可能不是足够好的 verifier。
- online PFT 需要更多计算，论文强调这是数据-计算权衡，但没有详细比较相同算力预算下“更强 offline 训练 / 更多数据 / 更大 verifier / 更多 RL rollout”之间的最优配置。
- 一个自然延伸是研究 off-policy RL 或离线规划方法：如果能在不频繁 on-policy 采样的情况下利用 learned RM 施加 $\Pi(R_{sim})$ 约束，可能兼具离线训练的稳定性和 verifier 路线的数据效率。

## 相关工作与启发
- **vs DPO**: DPO 直接把策略看成隐式 reward model，在偏好对上最大化 preferred completion 相对 dispreferred completion 的 log probability。本文不是否定 DPO，而是指出 DPO 学到的 local RM 更像 Q function，可能比完整回答级别的 global RM 更难从有限偏好数据中学习。
- **vs 传统 RLHF / PPO**: 传统 RLHF 常被经验性地认为更强，但原因容易被归结到 PPO、在线采样或工程细节。本文把两阶段流程抽象成“先 forward KL 拟合 reward，再 reverse KL/soft RL 投影到 policy”，从而解释它在统计上的潜在优势。
- **vs online DPO / online preference optimization**: online DPO 是本文实验里的主要工具，但作者强调工具本身不是结论。online DPO 的作用是提供一个受控平台，让 offline 和 online 使用同一 loss，只比较“是否经过 global RM 生成新偏好对”。
- **vs inverse RL / imitation learning**: Ng et al. 式 inverse RL 的经典观点是 reward 比 policy 更紧凑，先学 reward 再规划比行为克隆更省样本。本文把这个观点迁移到语言模型偏好微调，并用生成-验证差距解释为什么它在 LLM 后训练里仍然成立。
- **vs reward model OOD generalization 解释**: 以往工作观察到 global RM 比 DPO 隐式 RM 更能排序 OOD 样本。本文接受这个现象，但认为它不是终点：真正的问题是为什么 global RM 在分布内就更好学，而生成-验证差距给出了更底层的解释。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 不是提出新算法，而是把 RLHF 优势重新解释为 proper learning 与生成-验证差距，理论和实验问题意识都很强。
- 实验充分度: ⭐⭐⭐⭐☆ 控制变量做得扎实，主实验、反证实验和 gap-closing 实验相互呼应；不足是任务和模型范围仍偏集中。
- 写作质量: ⭐⭐⭐⭐⭐ 论文结构清楚，先证明等价再展示反常，再逐个排除假说，最后提出 H6，读者很容易跟上论证链。
- 价值: ⭐⭐⭐⭐⭐ 对理解 RLHF、DPO、reward model、reasoning verifier 都有直接启发，尤其适合作为“为什么还需要 RL”的理论读物。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Proximal Supervised Fine-Tuning](proximal_supervised_fine-tuning.md)
- [\[ICLR 2026\] Fine-tuning Behavioral Cloning Policies with Preference-Based Reinforcement Learning](fine-tuning_behavioral_cloning_policies_with_preferencebased_reinforcement_learn.md)
- [\[ICLR 2026\] SRFT: A Single-Stage Method with Supervised and Reinforcement Fine-Tuning for Reasoning](srft_a_single-stage_method_with_supervised_and_reinforcement_fine-tuning_for_rea.md)
- [\[ICLR 2026\] Escaping Policy Contraction: Contraction-Aware PPO (CaPPO) for Stable Language Model Fine-Tuning](escaping_policy_contraction_contraction-aware_ppo_cappo_for_stable_language_mode.md)
- [\[ICLR 2026\] On-Policy RL Meets Off-Policy Experts: Harmonizing Supervised Fine-Tuning and Reinforcement Learning via Dynamic Weighting](on-policy_rl_meets_off-policy_experts_harmonizing_supervised_fine-tuning_and_rei.md)

</div>

<!-- RELATED:END -->
