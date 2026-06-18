---
title: >-
  [论文解读] Confidence Estimation for LLMs in Multi-turn Interactions
description: >-
  [ACL 2026 Findings][视频理解][多轮对话] 首次系统研究多轮对话场景下的 LLM 置信度估计，提出两个核心准则（per-turn 校准 + 信息增加时单调性）、对应的 InfoECE 指标和 Kendall's $\tau$ 评估、Hinter-Guesser 数据集构造范式，并提出新颖的 P(SUFFICIENT) logit 探针——结果发现现有方法（verbalized / SC / P(TRUE)）在多轮场景中校准和单调性都很差，而 P(SUFFICIENT) 在 GUESS 上 InfoECE 降到 5.27（vs P(TRUE) 79.97）、$\tau$ 达 81.51，但任务远未解决。
tags:
  - "ACL 2026 Findings"
  - "视频理解"
  - "多轮对话"
  - "置信度估计"
  - "P(SUFFICIENT)"
  - "InfoECE"
  - "单调性"
---

# Confidence Estimation for LLMs in Multi-turn Interactions

**会议**: ACL 2026 Findings  
**arXiv**: [2601.02179](https://arxiv.org/abs/2601.02179)  
**代码**: 论文提及 GitHub（有，链接见原文）  
**领域**: LLM 校准 / 对话 / 置信度估计  
**关键词**: 多轮对话, 置信度估计, P(SUFFICIENT), InfoECE, 单调性

## 一句话总结
首次系统研究多轮对话场景下的 LLM 置信度估计，提出两个核心准则（per-turn 校准 + 信息增加时单调性）、对应的 InfoECE 指标和 Kendall's $\tau$ 评估、Hinter-Guesser 数据集构造范式，并提出新颖的 P(SUFFICIENT) logit 探针——结果发现现有方法（verbalized / SC / P(TRUE)）在多轮场景中校准和单调性都很差，而 P(SUFFICIENT) 在 GUESS 上 InfoECE 降到 5.27（vs P(TRUE) 79.97）、$\tau$ 达 81.51，但任务远未解决。

## 研究背景与动机
**领域现状**：置信度估计是减轻 LLM 幻觉的核心方向，但绝大多数工作（FActScore、Tian 2023、Xiong 2024）都聚焦单轮 QA，假设输入是一次性给的完整问题。

**现有痛点**：(1) 真实人机交互是多轮、增量的——用户分次澄清需求、模型反复问询、假设空间逐渐收窄——这种动态信息累积下的置信度行为完全没人研究；(2) 现有方法在多轮上能否保持校准、能否反映"信息越多越确定"这种直觉，未知；(3) 缺少针对多轮的评测指标和数据集——单轮 ECE 不能处理对话长度差异，标准 QA 数据集没有增量信息结构。

**核心矛盾**：在多轮场景中，置信度不应该是"对单一答复的固定属性"，而应该是"随对话演化、随信息积累而升高的动态信号"。但能否真的做到、用什么方法做、怎么测都没系统答案。

**本文目标**：(1) 形式化多轮置信度的两个 desiderata：per-turn 校准 + 单调性；(2) 设计长度归一化的 InfoECE 指标和 Kendall's $\tau$ 单调性指标；(3) 构造适合多轮的数据集（under-specified 用 Hinter-Guesser 范式生成 20Q / GUESS；fully-specified 用现成 GRACE / TrickMe）；(4) 实测主流置信度方法 + 提出新方法 P(SUFFICIENT)。

**切入角度**：作者观察到 under-specified 场景下"答案对了不等于信息足够" —— 模型可能蒙对正确答案但还有多个候选未被排除，这时置信度应该低；而 P(TRUE) 只问"答案对不对"，无法捕捉这种 identifiability 缺失。

**核心 idea**：把置信度探针的语义从"答案是否正确（P(TRUE)）"换成"现有信息是否足以唯一确定答案（P(SUFFICIENT)）"，让置信度对应 identifiability 而非 incidental correctness。

## 方法详解

### 整体框架
对每个对话 $d$ 的每个 turn $i$，模型输出答案 $\hat{y}_{d,i}$ 和置信度 $c_{d,i} \in [0, 1]$，记录正确性 $z_{d,i} = \mathbb{I}[\hat{y}_{d,i} = y_d]$。为消除对话长度差异，把 turn $i$ 归一化为信息级别 $s_{d,i} = i / L_d \in (0, 1]$，划分 $B$ 个 bin。两个核心指标：(a) **InfoECE** = $\frac{1}{B}\sum_b |\text{acc}_b - \text{conf}_b|$，衡量每个信息级别上的校准；(b) **Kendall's $\tau$** 衡量对话内置信度随 turn 单调递增的程度。再加上 5 种置信度估计方法（3 现有 + 1 新提）和 2 类数据集（under-spec / fully-spec）。

### 关键设计

**1. Hinter-Guesser 范式：用结构化角色保证信息严格累积，让置信度方法的失败可归因**

如果直接让两个 LLM 自由地玩 20Q / GUESS，早期常出现"问题撞墙、问到不相关线索"的情况，导致置信度起伏甚至倒退——这会污染多轮置信度评估的实验基础。Hinter-Guesser 范式用结构化角色破解这个问题：Hinter（LLM）被分配一个秘密实体，每轮给出"helpful 但非 trivial"的 hint；Guesser 做 best guess，同时做 **uniqueness probing**——即使猜对也要标记"是否还有其他候选同样符合当前证据"，从而把"碰巧蒙对"和"信息足以唯一锁定"区分开。对话一直持续到 Guesser 既猜对又认证唯一性才停止，只保留成功收敛的对话、丢弃无法收敛的轨迹。这样构造出的数据满足 C1-C3（信息单调增 + 步步可答 + 置信度应单调），最终 20Q 收集 1848 turn / 226 实体，GUESS 收集 1625 turn / 223 实体。正因为信息被强制严格累积，后续观察到的置信度失败就可以归因于方法本身而非数据噪声。

**2. InfoECE 指标：把绝对 turn 索引归一化成信息进度条，让变长对话可比**

不同对话长度不一，直接用 turn 索引 $i$ 做 ECE 会让短对话和长对话不可比。InfoECE 先把每个 turn 位置归一化成对话内的分数信息级别 $s_{d,i} = i/L_d \in (0, 1]$，再分 $B$ 个等宽 bin；每个 bin 内对所有跨对话的 turn 求平均置信度 $\text{conf}_b$ 和平均准确率 $\text{acc}_b$，得到

$$\text{InfoECE} = \frac{1}{B}\sum_b |\text{acc}_b - \text{conf}_b|.$$

归一化到 $[0,1]$ 之后，不同长度的对话被对齐到同一"信息进度条"上，"信息级别 = 50% 时该有多自信"这类问题才有了 well-defined 的答案。单调性则用 Kendall's $\tau = \frac{1}{N}\sum_d \frac{N^{(d)}_{con} - N^{(d)}_{dis}}{\binom{L_d}{2}}$ 衡量对话内置信度随 turn 递增的程度。

**3. P(SUFFICIENT) logit 探针：把置信度语义从"答案对不对"切换到"信息是否充分"**

P(TRUE) 在 under-specified 场景下有结构性缺陷——它只问"这个答案对吗"，但 turn 1 的 best guess 即使正确也只是"在所有候选里蒙中的"，按 P(TRUE) 会给出高置信度，可 epistemic 上此时本该低置信度。P(SUFFICIENT) 保留 P(TRUE) 的 logit-based 二选一探针形式，但把 prompt 改成"基于上述信息，是否足以推断正确答案就是 $\hat{y}$"，输出强制为 A（足够）或 B（不够）的单一大写字母，置信度取 confidence $= \Pr[\text{A} \mid p_{d,i}, \hat{y}_{d,i}]$。这样即便模型偶然蒙对了正确答案，只要 hints 还排除不掉其他候选，置信度就保持在低位——置信度对应的是 identifiability 而非 incidental correctness。论文没有给形式化证明，但实验直接验证了这次语义切换的关键作用：GUESS 上 InfoECE 从 P(TRUE) 的 79.97 暴降到 5.27，$\tau$ 从 3.29 飙到 81.51。

### 损失函数 / 训练策略
不训练任何模型，全部用现成开源 LLM：Llama3.1 Instruct (8B / 70B)、Qwen2.5 Instruct (7B / 72B)。生成温度 1，置信度估计温度 0。Self-Consistency 用 $m=20$ 次采样。为公平起见对每个方法都先让模型先答一次得到 $a$，再估计 $a$ 的置信度。

## 实验关键数据

### 主实验（InfoECE↓ / $\tau$↑，Llama3.1-70B 为例）

| Method | 20Q InfoECE | 20Q $\tau$ | GUESS InfoECE | GUESS $\tau$ | GRACE InfoECE | TRICKME InfoECE |
|--------|-------------|------------|---------------|---------------|---------------|-----------------|
| Vanilla-Verb | 59.63 | 17.60 | 65.52 | 16.92 | 39.06 | 47.47 |
| CoT-Verb | 58.39 | 34.49 | 70.16 | 18.24 | 96.04 | 80.97 |
| SC (m=20) | 32.99 | 28.98 | 56.88 | 2.59 | 15.91 | 19.90 |
| P(TRUE) | 67.82 | 40.82 | 79.97 | 3.29 | 37.04 | 35.62 |
| **P(SUFFICIENT)** | **13.05** | **48.43** | **5.27** | **81.51** | **11.52** | 23.16 |

P(SUFFICIENT) 在 4 数据集 ×  4 模型的大多数 InfoECE 上都是最佳，尤其 GUESS 上 InfoECE 仅 **5.27**（vs P(TRUE) 79.97，提升 15×）、$\tau$ 达 **81.51**（vs P(TRUE) 3.29）。对 ground-truth 答案评估时所有方法的 $\tau$ 都大幅提升，P(SUFFICIENT) 在 GUESS 上 Qwen2.5-72B 达到 $\tau = 93.91$，证明模型能部分识别 hints 是否对齐正确答案。

### 控制实验：placebo vs informative turn（Llama3.1-70B on GUESS）

| 方法 | Conf at $i-1$ | Conf placebo at $i'$ | Conf informative at $i$ |
|------|---------------|----------------------|--------------------------|
| Vanilla-Verb | 71.30 | 73.70 (+2.40) | 83.83 (+12.53) |
| CoT-Verb | 78.39 | 78.77 (+0.38) | 88.41 (+10.02) |
| SC | 52.42 | 53.18 (+0.76) | 72.33 (+19.91) |
| P(TRUE) | 88.16 | 88.14 (−0.02) | 95.17 (+7.01) |
| **P(SUFFICIENT)** | 14.27 | **2.97 (−11.30)** | 27.58 (+13.31) |

P(SUFFICIENT) 是唯一在 placebo（无信息废话 hint）下**显著降低**置信度的方法，证明它真的在跟踪信息而非 turn 计数；P(TRUE) 在 Llama3.1-8B / Qwen2.5-72B 的 GUESS 上 placebo 加 +11.75 / +14.61，揭示它有长度artifact。

### 关键发现
- **现有方法在多轮上普遍不校准**：Verbalized 类（VANILLA-VERB / COT-VERB）和 P(TRUE) 的 InfoECE 通常在 40-80 之间，远超合理水平。SC 是 fully-specified 场景下校准最好的默认选择，但在 under-specified 上 $\tau$ 经常单位数。
- **P(SUFFICIENT) 在 under-specified 上压倒性最佳**：因为 GUESS / 20Q 中"信息累积 → 候选缩减"的语义和 sufficiency 探针完美匹配；而 fully-specified 数据集上优势缩小但仍领先（GRACE InfoECE 11.52 vs SC 15.91）。
- **Placebo 实验是关键的诊断工具**：5 种方法 × 4 模型 × 2 数据集 = 40 对比，informative turn 显著变化 27 次 vs placebo 仅 18 次，证明置信度增长**部分**来自真实信息但部分是 turn count artifact。P(SUFFICIENT) 把这两个因素分得最清。
- **Multi-turn vs Single-turn summary**：作者把多轮 hints 浓缩成单轮 prompt 再测，发现准确率差异 <1%，没出现 Laban et al. (2025) 的"get lost in conversation"效应（因为此处任务非复杂数学推理）；但置信度行为差异巨大——P(SUFFICIENT) 在 single-turn 上骤降（20Q: Qwen2.5-7B 从 63.13 降到 13.23），说明它依赖对话结构线索。
- **模型规模效应**：参数增加 $\tau$ 提升明显（Qwen2.5-72B 在 GUESS 上 P(SUFFICIENT) $\tau = 83.76$ vs 7B 的 51.44），但 InfoECE 改善更微妙，有时小模型反而绝对校准更好。

## 亮点与洞察
- **从"答案对不对"切换到"信息是否充分"**：这是非常深刻的语义切换——P(TRUE) 测的是 outcome correctness，P(SUFFICIENT) 测的是 epistemic identifiability。在信息逐步揭示的场景下，后者才是真正想要的"理性自信"信号。这种"重新定义探针的语义而非堆叠复杂度"是少见的优雅。
- **Hinter-Guesser 范式 + uniqueness probing**：用结构化角色 + 唯一性认证巧妙解决了多轮对话数据集构造的混乱问题，让评测能聚焦在置信度方法本身。这个数据构造范式可推广到任何需要"信息严格累积"的多轮评测（如多步推理、医疗诊断对话）。
- **InfoECE 的长度归一化设计**：用对话内的分数信息级别替代绝对 turn 索引，让 ECE 在变长对话间可比。这个归一化思想可迁移到所有变长序列的校准评测（如 chain-of-thought 步骤校准）。
- **Placebo 控制实验**：用"插入无信息废话 turn"作为 adversarial baseline 来分离"真信息驱动的置信度增长"和"turn count artifact"，是非常巧妙的实验设计，可标准化为任何"动态置信度"研究的必备控制。
- **跨模型方差揭示稳健性差异**：Qwen2.5 系列在 verbalized 上偶尔 $\tau$ 最高但绝对校准差，提示读者"$\tau$ 高 ≠ 可用"，应该联合看 InfoECE + $\tau$。

## 局限与展望
- 作者承认数据集都是简化的信息检索游戏，没有真实对话中的话题切换、错误修复、混合意图等现象，对真实开放域对话的可迁移性有限。
- 多轮置信度评测只限于 information-seeking 任务；开放生成、创造性协作场景下的置信度动态完全没研究。
- 评测只覆盖校准 + 单调性两个维度，没量化对下游应用的真实价值（如什么时候触发澄清问题、什么时候调用工具）。
- 只研究 confidence 没研究 uncertainty——后者更难，但在 agentic 应用中可能更重要。
- 自己发现：P(SUFFICIENT) 的优势在 fully-specified 场景（GRACE / TRICKME）明显缩小，说明它本质是"借候选缩减"获得校准；如果初始候选空间已经很小，sufficiency 与 truth 等价，方法优势消失。
- 没有评估模型规模 + finetune 的影响——如果给模型 SFT 一下 sufficiency 判断，效果会怎样？
- 只对 4 个开源模型评测，闭源 LLM（GPT-4 / Claude）行为可能不同。

## 相关工作与启发
- **vs Tian et al. (2023) Verbalized Confidence**：单轮自报置信度的经典方法，本文证明其在多轮上严重失校准（InfoECE 经常 50+）。
- **vs Kadavath et al. (2022) P(TRUE)**：单轮 logit 探针的标杆，本文证明它在 under-specified 多轮上既不校准也不单调，根本问题在"答案对错 ≠ 信息充分"。P(SUFFICIENT) 就是针对这个语义缺陷设计的替代。
- **vs Self-Consistency (Manakul et al. 2023)**：在 fully-specified 上是最佳校准 baseline，但在 under-specified 上 $\tau$ 极低（GUESS 单位数），因为多次采样模型只是反复确认同一个错误猜测。
- **vs Laban et al. (2025) "LLMs get lost in multi-turn"**：作者用渐进式信息揭示数据集发现自家场景**没有** lost in conversation 效应，因为任务不涉及复杂数学；这暗示"get lost"效应高度任务特异。
- **vs Zhang et al. (2026)（同期）Conformity in Multi-turn Persuasion**：讨论对抗 persuasion 下的置信度抵抗；本文是其互补——cooperative 信息累积下的置信度增长。
- **vs Sung et al. (2025) GRACE / Wallace et al. (2019) TrickMe**：直接借用这两个 incremental QA 数据集作为 fully-specified regime 的评测基准。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统化多轮置信度评测，方法论（InfoECE + Hinter-Guesser + P(SUFFICIENT) + placebo 控制）每一项都有原创贡献。
- 实验充分度: ⭐⭐⭐⭐⭐ 4 模型 × 5 方法 × 4 数据集 + ground-truth $\tau$ + placebo 控制 + multi/single-turn 对比 + 模型规模扫描，非常彻底。
- 写作质量: ⭐⭐⭐⭐⭐ 形式化定义清晰、动机推导有力、所有 prompt 在附录给全、图表组织清晰。
- 价值: ⭐⭐⭐⭐⭐ 为多轮 LLM 校准奠定基础方法学，"sufficiency vs truth"的语义切换洞察对后续置信度研究有深远启发。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] VISTA: Verification In Sequential Turn-based Assessment](vista_verification_in_sequential_turn-based_assessment.md)
- [\[ECCV 2024\] Benchmarks and Challenges in Pose Estimation for Egocentric Hand Interactions with Objects](../../ECCV2024/video_understanding/benchmarks_and_challenges_in_pose_estimation_for_egocentric_hand_interactions_wi.md)
- [\[ICML 2026\] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](../../ICML2026/video_understanding/video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)
- [\[NeurIPS 2025\] SAMA: Towards Multi-Turn Referential Grounded Video Chat with Large Language Models](../../NeurIPS2025/video_understanding/sama_towards_multi-turn_referential_grounded_video_chat_with_large_language_mode.md)
- [\[NeurIPS 2025\] When One Moment Isn't Enough: Multi-Moment Retrieval with Cross-Moment Interactions](../../NeurIPS2025/video_understanding/when_one_moment_isnt_enough_multi-moment_retrieval_with_cross-moment_interaction.md)

</div>

<!-- RELATED:END -->
