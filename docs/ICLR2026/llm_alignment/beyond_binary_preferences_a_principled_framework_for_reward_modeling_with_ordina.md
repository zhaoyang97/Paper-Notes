---
title: >-
  [论文解读] Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback
description: >-
  [ICLR2026][LLM对齐][奖励模型] 这篇论文指出现有奖励模型只会用"A 好于 B"的二元偏好，面对人类打的 Likert 分级反馈（"明显更好/更好/略好"）只能靠加 margin、乘权重这种拍脑袋的启发式补丁；作者把奖励建模重新表述成**离散序数回归**问题，从有序 logit 模型自然推出两个有理论根据的损失（NLL 与 all-threshold），让分隔各偏好等级的"阈值"直接从数据里学出来，在 RewardBench / RM-Bench 上一致追平或超过启发式基线，并把错误严重度降低 87%。
tags:
  - "ICLR2026"
  - "LLM对齐"
  - "奖励模型"
  - "序数回归"
  - "Likert 偏好"
  - "阈值学习"
  - "RLHF"
---

# Beyond Binary Preferences: A Principled Framework for Reward Modeling with Ordinal Feedback

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=mteZOi0xyu](https://openreview.net/forum?id=mteZOi0xyu)  
**代码**: 待确认（作者称基于 TRL 库扩展，验收后开源）  
**领域**: 对齐RLHF / 奖励建模  
**关键词**: 奖励模型, 序数回归, Likert 偏好, 阈值学习, RLHF

## 一句话总结
这篇论文指出现有奖励模型只会用"A 好于 B"的二元偏好，面对人类打的 Likert 分级反馈（"明显更好/更好/略好"）只能靠加 margin、乘权重这种拍脑袋的启发式补丁；作者把奖励建模重新表述成**离散序数回归**问题，从有序 logit 模型自然推出两个有理论根据的损失（NLL 与 all-threshold），让分隔各偏好等级的"阈值"直接从数据里学出来，在 RewardBench / RM-Bench 上一致追平或超过启发式基线，并把错误严重度降低 87%。

## 研究背景与动机

**领域现状**：现代大模型对齐（RLHF、DPO）本质都建立在 Bradley-Terry（BT）模型上——它假设"偏好 A 胜过 B 的概率随两者奖励差单调增长"，即 $p^\star(y \succ y' \mid x) = \sigma\big(r^\star(x,y) - r^\star(x,y')\big)$。奖励模型就是用这个二元 logistic 损失训练出来的判别器。

**现有痛点**：BT 模型生来只为**二元比较**设计——A 要么被偏好、要么不被偏好。但越来越多的偏好数据集（HelpSteer2/3）让标注者不只标"哪个更好"，还标"好多少"——在 Likert 量表上给出"明显更好""更好""略好""几乎一样好"这种**序数等级**。这些等级信号比二元标签丰富得多，可现有方法被二元根基锁死，基本只能把它当噪声丢掉，或者打补丁硬塞。

**核心矛盾**：现有补丁分三类——Llama-2 的 **Margin BT** 在损失里减去一个手工 margin $m(z)$；HelpSteer2 的 **Scaled BT** 把损失整体乘以偏好强度 $m(z)$；Gunter 等的 **Soft Label** 把序数等级解释成软概率 $p(z)$。它们共有两个根本缺陷：其一，**没有任何描述"人类如何给出序数标签"的数学模型**，所有改动都是凭直觉拼上去的，说不清背后是什么假设；其二，**全靠手调超参**——"略好"和"明显更好"之间该隔多大 margin？强偏好该贡献两倍还是三倍梯度？这些值无清晰解释，而且一旦偏好等级的数量或定义变了就得重调，方法因此脆弱且数据集相关。

**本文目标**：给序数偏好奖励建模一个**有原理的统一数学框架**——损失函数应当从明确的建模假设里自然导出，所有参数（包括各等级之间的边界）都从数据中学习，而不是手工指定。

**切入角度**：作者注意到，"带序关系的离散类别预测"在统计学里早有成熟答案——**序数回归（ordinal regression）**，它专门处理评分系统（1–5 星）、问卷（强烈反对到强烈同意）、严重程度（轻/中/重）这类有序分类数据。机器学习社区为它发展了大量方法（有序 logit、阈值学习、大间隔法），却从没人把这套机器搬到偏好学习上。

**核心 idea**：用一组**从数据中学出的阈值**把连续的奖励差空间切成若干有序区间，每个区间对应一个偏好等级；用序数回归的成熟损失（概率式的 NLL、间隔式的 all-threshold）替换掉那些手工 margin/权重，让"略好与明显更好之间隔多远"变成可学参数而非超参。

## 方法详解

### 整体框架

整篇方法的输入是带 Likert 分级的成对偏好数据 $D = \{(x_i, y_i, y'_i, z_i)\}$，其中 $x$ 是 prompt，$y/y'$ 是两个候选回复，$z_i \in [-K] \cup \{0\} \cup [K]$ 是序数偏好标签：$z>0$ 表示 $y$ 在第 $z$ 级上被偏好、$z<0$ 表示 $y'$ 被偏好、$z=0$ 表示两者约等。输出是一个标量奖励模型 $r_\phi(x,y)$，要求它给出的**奖励差** $s_\phi(x,y,y') = r_\phi(x,y) - r_\phi(x,y')$ 不仅符号对（哪个回复更好），**幅度也对**（差多少应反映偏好强度）。

整体只需把"二元判别"换成"序数回归"四步：① 沿用 BT 的奖励差作为序数回归的连续预测量 $s_\phi$；② 引入 $2K$ 个有序阈值 $\zeta_{-K} < \dots < \zeta_{-1} < \zeta_1 < \dots < \zeta_K$ 把实数轴切成 $2K+1$ 段，每段对应一个偏好等级（注意刻意不设 $\zeta_0$，落在 $(\zeta_{-1}, \zeta_1)$ 之间即 $z=0$）；③ 从这套阈值结构导出两类损失（概率式 NLL / 间隔式 AT），并对阈值加 L2 正则保证解有界；④ 把奖励参数 $\phi$ 和阈值 $\zeta$ 通过重参数化联合优化。这是一个纯损失/概率建模层面的改造，没有额外的多模块流水线，因此不强加框架图，下面用公式把每个设计讲清。

### 关键设计

**1. 把奖励建模重写成离散序数回归：用学出来的阈值切奖励差空间**

针对"BT 只懂二元、没法表达强度"这个根本痛点，作者借用序数回归的潜变量框架。标准序数回归学一个打分函数把输入映到一维潜空间，再用一组有序阈值 $-\infty = \zeta_0 < \zeta_1 < \dots < \zeta_{K-1} < \zeta_K = +\infty$ 把潜空间切成 $K$ 个区间，潜分落在哪段就属于哪个序数等级——当 $K=2$ 退化成只有一个阈值的二元分类，正好对回 BT。作者把这套搬到偏好学习上：潜变量直接取奖励差 $s_\phi(x,y,y')$，并因为偏好有方向（$y$ 好 / $y'$ 好两侧对称）而引入 $2K$ 个阈值、对称地排在 0 两侧。关键转变在于：旧方法里"略好 vs 明显更好该隔多远"是一个写死的超参 $m(z)$，而这里它变成了**模型参数 $\zeta$**，从数据中和奖励一起学出来——这就是"有原理"和"打补丁"的分水岭。

**2. 两个从建模假设自然导出的损失：概率式 NLL 与间隔式 All-Threshold**

针对启发式损失"说不清是什么假设导出来的"，作者给出两条各有清晰来源的损失。**概率式（NLL）**假设标注者遵循**有序 logit 模型**：给定奖励差，观测到等级 $z$ 的概率由相邻阈值的 sigmoid 之差给出，例如 $z\in[K]$ 时 $p(y \succ_z y' \mid x) = \sigma(\zeta_{z+1} - s_\phi) - \sigma(\zeta_z - s_\phi)$，$z=0$ 时为 $\sigma(\zeta_1 - s_\phi) - \sigma(\zeta_{-1} - s_\phi)$；它天然保证各等级概率非负且求和为 1，最大似然即最小化 $L_{\text{NLL}} = -\log p(y \succ_z y' \mid x)$，直观上"惩罚模型给真实等级所在区间分配过低的概率质量"。**间隔式（AT，All-Threshold）**不假设具体的人类行为概率模型，而是受大间隔法启发，直接惩罚奖励差落在每个阈值错误一侧：

$$L_{\text{AT}}(r_\phi, \zeta) = \sum_{l \in [-K]\cup[K]} -\log \sigma\big(\nu(l;z)\cdot(\zeta_l - s_\phi)\big),$$

其中 $\nu(l;z) = -1$ 当 $l<z$、$\nu(l;z) = +1$ 当 $l \ge z$。它要求 $s_\phi$ 大于所有 $l<z$ 的阈值、小于所有 $l\ge z$ 的阈值，且因为**累加了所有阈值**的违例惩罚，错得越离谱（跨越越多阈值）罚得越重——这正是序数回归比普通分类多出来的"误分有轻重"的特性。作者特意弃用了只看目标区间两端的 Immediate-Threshold 损失，因为前人实验已证明它逊于 AT。

**3. 对称 vs 非对称阈值：用一条定理把"偏好对称"翻译成"阈值对称"**

阈值要不要约束对称，是一个与损失选择正交的建模决策。**对称模型**假设"在第 $k$ 级上偏好 $y$ 胜 $y'$"和"在第 $k$ 级上偏好 $y'$ 胜 $y$"强度相同，即 $\zeta_{-k} = -\zeta_k$。论文用 **Theorem 3.2** 给了它理论依据：在有序 logit 模型下，若偏好数据满足对称性质 $P(y \succ_k y' \mid s = r) = P(y' \succ_{-k} y \mid s = r)$，则阈值**必然**满足 $\zeta_{-k} = -\zeta_k$——人类偏好的对称性会强制阈值对称。对称模型把阈值参数从 $2K$ 砍到 $K$，减少过拟合。**非对称模型**则放开全部 $2K$ 个阈值独立学习，用来捕捉认知偏差（如损失厌恶、负面偏置导致"强烈否定"和"强烈肯定"不等价）。这是一个偏差-方差权衡，实验里对称版反而常胜出，反过来印证"人类偏好确实近似对称"。

**4. 正则化保证有界解 + 重参数化无约束优化：让联合训练真能收敛**

奖励参数 $\phi$ 和阈值 $\zeta$ 同时学会引入一个优化病态。**Theorem 3.1** 证明：无正则时若存在把所有样本正确排序的解 $(r^\star_\phi, \zeta^\star)$，则对任意 $c>1$，放缩解 $(c r^\star_\phi, c\zeta^\star)$ 损失严格更低且 $c\to\infty$ 时损失趋于 0——正确样本贡献趋零、错误样本只线性增长，于是梯度下降会把阈值推向无穷大，数值不稳、标定崩坏。作者据此对阈值加 L2 正则：

$$\min_{\phi, \zeta \in C} \sum_{(x,y,y',z)\in D} L(r_\phi, \zeta) + \lambda \lVert \zeta \rVert_2^2,$$

其中 $C$ 是阈值有序的可行域，$\lambda$ 在收敛速度与灵活性之间做权衡。为处理"阈值必须严格递增"这个约束，又用**单调重参数化**：$\zeta_{-K} = \alpha_0$，$\zeta_k = \zeta_{k-1} + \exp(\alpha_k)$——指数映射保证增量恒正、阈值天然递增，把带约束问题转成对 $(\phi, \alpha)$ 的无约束优化，标准梯度法即可解（对称模型只参数化正侧阈值）。每步同时更新奖励头与阈值。

### 损失函数 / 训练策略

最终落地三个实例：**NLL-Symmetric**（NLL + 对称阈值）、**NLL-Asymmetric**（NLL + 全独立阈值）、**All-Threshold**（AT + 非对称阈值）。底座是三个 7–8B 指令模型（Llama-3.1-8B、Mistral-7B、Zephyr-7B）的 SFT checkpoint，把语言头换成单层奖励头输出标量。训练 8 个 epoch、有效 batch 64、8 张 H100/H200 上 FSDP，奖励与阈值参数联合优化、阈值带 L2 正则。

## 实验关键数据

### 主实验

在 HelpSteer2/3（7 级强度标注，-3 到 +3）上训练，于 RM-Bench 和 RewardBench 评测。**NLL-Symmetric 是最稳的方法**，多数配置取得最高平均分，平均超过基线 2–5%。下表为 RewardBench（HelpSteer2 训练）的平均分对比（Avg 越高越好）：

| 模型 | 方法 | Chat-Hard | Safety | Reason | Avg |
|------|------|-----------|--------|--------|-----|
| Llama | Margin BT | 0.660 | 0.885 | 0.703 | 0.802 |
| Llama | Soft Label | 0.581 | 0.689 | 0.680 | 0.722 |
| Llama | **NLL-Sym** | **0.728** | **0.897** | **0.804** | **0.843** |
| Llama | NLL-Asym | 0.695 | 0.837 | 0.794 | 0.809 |
| Llama | All-Thresh | 0.689 | 0.872 | 0.798 | 0.820 |

NLL-Symmetric 一致优于 NLL-Asymmetric，印证 Theorem 3.2——人类偏好确实近似对称。在 RM-Bench 上 NLL-Sym 同样在多数 model×dataset 组合里拿到最高或次高 Total（如 Mistral/HelpSteer2 上 0.647，明显高于 Soft Label 的 0.535）。

### 错误严重度与序数预测

| 分析 | Simple BT | NLL-Symmetric | 改善 |
|------|-----------|---------------|------|
| 错误数（RewardBench, Llama） | 433 | 282 | -35% |
| 平均错误 margin | 3.827 | 0.501 | **-87%** |
| 最大错误 margin | 高达 20 | 不超过 2.5 | — |

序数预测（HelpSteer2 测试集 448 例，对比 post-hoc 标定）：

| 方法 | MAE ↓ | Acc@0 ↑ | Acc@1 ↑ | Acc@2 ↑ |
|------|-------|---------|---------|---------|
| Margin BT (post-hoc) | 2.181 | 16.1% | 48.7% | 59.6% |
| Soft Label (post-hoc) | 1.725 | 12.9% | 47.8% | 78.1% |
| **NLL-Symmetric (joint)** | **1.060** | **29.7%** | **72.5%** | **92.9%** |

### 关键发现
- **降的不是错误数量，而是错误的"离谱程度"**：NLL-Sym 把错误数降 35%，但平均错误 margin 降 87%——犯错时都是在真·模糊样本上低置信度地错，几乎不会再高置信度地把差回复打高分。这对下游 RL 至关重要，因为"自信地给错奖励"最能把策略优化带偏。
- **联合训练不可由事后标定替代**：把基线奖励模型冻结后再拟合阈值（post-hoc），MAE 1.725、Acc@1 不到一半；联合训练 MAE 1.060、Acc@1 达 72.5%。说明细粒度序数结构必须在训练中和奖励一起学，事后补不回来。
- **对噪声鲁棒**：在系统性平移噪声下（标签整体偏一级），即便 100% 污染性能几乎不变（RewardBench 0.808–0.846 vs 0.843），因为学出的阈值能吸收系统偏置；随机噪声下退化平缓，25% 污染时仍持平干净基线、50% 仍有意义学习。
- **必须正则**：无正则时阈值会无界增长（Theorem 3.1 的实证），数值不稳。

## 亮点与洞察
- **"换框架而非打补丁"的范式价值**：最让人"啊哈"的是，作者没有再发明一个新 margin，而是指出整个问题本就是统计学里有标准答案的序数回归——margin/权重/软标签这些启发式不过是它的退化或近似。这种"把临时补丁还原成成熟数学对象"的思路，可迁移到任何"被二元假设卡住、其实有序结构"的偏好/反馈任务。
- **阈值可学 = 超参消失**：把"略好与明显更好隔多远"从手调超参变成可学参数，一举解决了"换数据集就要重调、且无清晰解释"的脆弱性，学出的阈值结构本身还能反映标注者如何区分各等级，是手调永远发现不了的可解释信号。
- **理论与实验闭环**：Theorem 3.1 解释了为什么必须正则（否则无界），Theorem 3.2 解释了为什么对称版更好（人类偏好对称），两条定理都被实验直接验证，理论不是装饰。
- **可直接迁移到 DPO**：奖励差是 DPO 的隐式量，作者指出整套序数损失能平移到 DPO 风格算法（Appendix A），潜在影响面不止奖励模型。

## 局限与展望
- **依赖 Likert 分级数据**：方法的全部增益来自序数标注，对只有二元标签的数据集退化回 BT、无优势；目前只在 HelpSteer2/3 这类 7 级标注上验证。
- **有序 logit 是一个建模假设**：NLL 路线把人类标注假设成有序 logit 生成过程，真实标注者行为未必严格符合；AT 路线虽不依赖此假设但实验里整体逊于 NLL。
- **只评了奖励模型本身，未端到端验证下游 RL**：论文反复强调"低严重度错误利好下游 RLHF/策略优化"，但并未真正跑一遍 RL 看最终策略质量的提升，这一步是后续最该补的。
- **更复杂反馈结构待扩展**：多维度评分、带不确定性的成对比较等更丰富的标注形式，作者列为未来工作但尚未实现。

## 相关工作与启发
- **vs Margin BT (Llama-2)**：它在 BT 损失里减一个手工 margin $m(z)$；本文把 margin 替换成从数据学出的有序阈值，且有有序 logit 概率模型背书。本文优势是无需手调、可解释、换数据集免重调；Margin BT 胜在实现极简。
- **vs Scaled BT (HelpSteer2)**：它把损失整体乘强度权重 $m(z)$；本文不靠加权而靠阈值划分区间来表达强度，且 NLL 给出完整的 $P(z\mid x)$ 概率分布，能预测偏好强度而不只是排序。
- **vs Soft Label**：它把序数等级当软概率标签；本文证明这同样是缺乏生成模型的启发式，且 post-hoc 实验显示软标签拟合不出联合训练才能得到的细粒度序数结构（Acc@1 47.8% vs 72.5%）。
- **vs 经典序数回归（Chu & Keerthi、Rennie & Srebro）**：本文是把这套成熟机器**首次**引入偏好学习，并针对偏好的方向性引入对称 $2K$ 阈值、给出对称性定理与有界性定理两条新理论结果。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个把序数回归引入奖励建模的有原理框架，"还原成熟数学对象"的视角清新且有说服力。
- 实验充分度: ⭐⭐⭐⭐ 三模型×两数据集×两 benchmark 覆盖扎实，错误严重度/联合训练/噪声鲁棒分析到位；唯独缺端到端 RL 验证。
- 写作质量: ⭐⭐⭐⭐⭐ 动机—理论—方法—实验环环相扣，两条定理把直觉钉成定论，叙述清晰。
- 价值: ⭐⭐⭐⭐⭐ 随着标注走向更细粒度反馈，"如何有原理地利用序数信号"会越来越关键，本文提供了可直接落地且可扩展到 DPO 的数学地基。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Learning Ordinal Probabilistic Reward from Preferences (OPRM)](learning_ordinal_probabilistic_reward_from_preferences.md)
- [\[ICLR 2026\] RLBFF: Binary Flexible Feedback to Bridge Between Human Feedback & Verifiable Rewards](rlbff_binary_flexible_feedback_to_bridge_between_human_feedback_verifiable_rewar.md)
- [\[ICLR 2026\] Omni-Reward: Towards Generalist Omni-Modal Reward Modeling with Free-Form Preferences](omni-reward_towards_generalist_omni-modal_reward_modeling_with_free-form_prefere.md)
- [\[ICLR 2026\] Robust Reward Modeling via Causal Rubrics](robust_reward_modeling_via_causal_rubrics.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)

</div>

<!-- RELATED:END -->
