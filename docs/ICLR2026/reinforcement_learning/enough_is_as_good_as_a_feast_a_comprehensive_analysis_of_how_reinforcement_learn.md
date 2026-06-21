---
title: >-
  [论文解读] Enough is as good as a feast: A Comprehensive Analysis of How Reinforcement Learning Mitigates Task Conflicts in LLMs
description: >-
  [ICLR2026][强化学习][模型融合] 这篇论文系统地比较了 SFT 和 RL 两种后训练范式对「模型融合（model merging）」的影响，发现 RL 训练出的模型在被融合后性能掉得远比 SFT 少，并从 on-policy 数据、RL 优化目标的自适应衰减、正负样本联合优化三个角度给出了实证与理论解释。
tags:
  - "ICLR2026"
  - "强化学习"
  - "模型融合"
  - "任务冲突"
  - "SFT"
  - "on-policy"
---

# Enough is as good as a feast: A Comprehensive Analysis of How Reinforcement Learning Mitigates Task Conflicts in LLMs

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=N4l4Jp50R4](https://openreview.net/forum?id=N4l4Jp50R4)  
**代码**: 待确认  
**领域**: 强化学习 / LLM 后训练 / 模型融合  
**关键词**: 模型融合、任务冲突、强化学习、SFT、on-policy

## 一句话总结
这篇论文系统地比较了 SFT 和 RL 两种后训练范式对「模型融合（model merging）」的影响，发现 RL 训练出的模型在被融合后性能掉得远比 SFT 少，并从 on-policy 数据、RL 优化目标的自适应衰减、正负样本联合优化三个角度给出了实证与理论解释。

## 研究背景与动机
**领域现状**：模型融合（model merging）是把多个独立微调过的专家模型直接在参数空间里合并成一个统一模型的技术——不需要原始训练数据、不需要重新训练、也不用维护多个 checkpoint，因此在 LLM 时代特别有吸引力。主流做法基于「任务向量」$\tau_i := \theta_i - \theta_0$（微调后参数减去共享基座参数），再用 Averaging、TIEs、Task-Arithmetic、DARE 等策略去合并这些向量。

**现有痛点**：融合的核心难题是「任务冲突（task conflicts）」——不同任务的参数更新方向相互干扰，导致合并后的模型在各任务上都掉点。已有研究几乎全部把精力放在「怎么设计更聪明的合并算子」上，却有一个被系统性忽略的前提：这些待融合的专家模型本身是怎么训出来的？

**核心矛盾**：LLM 的后训练范式可以粗分为监督微调（SFT）和强化学习（RL）两大类，但现有融合工作默认研究的都是 SFT 训练的模型，SFT 与 RL 在融合行为上的差异几乎是一片空白。直觉上，训练范式决定了参数更新的「形状」，从而直接决定融合时冲不冲突，但没人量化过。

**本文目标**：(1) 在受控实验下系统比较 SFT 与 RL 训练的模型在融合后的性能保持能力；(2) 如果 RL 确实更适合融合，要从机制层面解释清楚「为什么」。

**切入角度**：作者在数学、代码、指令遵循、逻辑谜题、排序五个可自动验证的任务上做配对融合实验，固定基座和合并算法做横向对照，再用参数范数、性能 landscape、冲突范数等工具去拆解 RL 与 SFT 的更新差异。

**核心 idea**：RL 天生更适合被融合——因为 RL 的 on-policy 采样、随收敛而衰减的 advantage、以及对正负样本的联合优化，三者共同把参数更新压成「小幅、低冲突、任务正交」的形状，正应了那句「enough is as good as a feast（够用即可，不必贪多）」。

## 方法详解
本文不是提出一个新算法，而是一项分析性研究。它的「方法」是一套严谨的对照实验设计加上三层机制解释，因此下面把研究设计当作「整体框架」、把三个解释因子当作「关键设计」来讲。

### 整体框架
作者构造了一个两层结构的研究：**第一层「现象层」**先证明 RL 训练的模型确实比 SFT 更耐融合，并验证这个结论对不同合并方法（Averaging / TIEs / Arithmetic / DARE）、不同 RL 算法（PPO / GRPO / REINFORCE++）、不同基座（Llama-3.2-3B / Llama-3.1-8B / Mistral-Small-24B）都成立，排除是某个特例造成的；**第二层「机制层」**再追问为什么，把 RL 与 SFT 的差别拆成三个可单独检验的维度——训练数据是不是 on-policy、优化目标的内在动力学、是否同时用了负样本——并对每个维度配上量化指标（参数更新范数、冲突范数、收敛精度）和理论界来佐证。整条逻辑链是「先量化现象 → 再用性能 landscape 区分『随机扰动』和『任务冲突』两种掉点来源 → 最后逐一拆解三个机制」。

度量任务冲突的关键工具是「冲突范数」：先定义冲突指示矩阵 $C(\Delta\theta_{t_i}, \Delta\theta_{t_j}) = \Delta\theta_{t_i} \odot \Delta\theta_{t_j}$（$\odot$ 是逐元素乘积），其中负元素表示两个任务在该参数上方向相反、属于破坏性干扰；再只取负元素求范数得到冲突范数 $\|C\|_{\text{conflict}} = \big\|\,C_{ij}\cdot \mathbb{1}_{\{C_{ij}<0\}}\,\big\|_2$，它度量了两个任务参数更新相互对抗的总强度。

### 关键设计

**1. on-policy 数据让梯度更新「天生更小」：从数据来源切入解释**

SFT 与 RL 最表层的区别是训练数据来源——SFT 用固定数据集（off-policy），RL 用模型自己采样出来的回答（on-policy）。作者直接测量三种范式下整个模型更新的范数 $\|\Delta\theta\|$：在数学任务上 SFT 是 6.5，而把 SFT 改成「从原模型采样数据」的拒绝采样微调 RFT 只有 2.36，纯 RL 更是低到 0.78（见 Table 2）。为隔离「数据来源」这一变量，RFT 的设计很巧妙——它本质是 SFT，但数据采自原模型，于是 RFT 介于 SFT 与 RL 之间正好说明 on-policy 数据本身就能压低更新幅度。进一步看分布：RL 中更新幅度超过 $10^{-5}$ 的参数比例在 math/code/IF 上只有 25.0%/20.7%/24.1%，而 SFT 高达 79.9%/78.0%/73.9%。也就是说 RL 的更新是「大量小幅改动」，这种低幅更新更不容易覆盖掉模型里属于其它任务的知识，因此融合时冲突更少。

**2. RL 优化目标随收敛自适应衰减：「够用即可」的内在动力学**

第二个区别是优化目标。作者证明 RL 的更新强度会随训练收敛而自动变小，而 SFT 不论收没收敛都按固定强度更新。理论上分两步：定理 1 给出单个 query 下期望绝对 advantage 的上界 $\mathbb{E}_{a\sim\pi_\theta}\big[|A(a,x)|\big] \le \sqrt{\mathrm{Var}(r(a,x))}$（其中 $A(a,x)=r(a,x)-b(x)$ 是 advantage，即观测奖励减去基线）；定理 2 进一步指出当 reward 落在有界区间、且每个状态的 advantage 均值为零时，advantage 在期望意义下趋于零，$\lim_{n\to\infty}\mathbb{E}(|A_n(a,x)|)=0$（对 GRPO 那种标准化 advantage $A=\frac{r-\mathbb{E}(r)}{\mathrm{std}(r)}$ 同样成立）。

把它代回参数更新就看出关键差别：SFT 的累计更新是 $\|\Delta\theta^{\text{SFT}}\|_2 = \|\sum_s \eta G_s\|_2$，每步等权；而 RL 是 $\|\Delta\theta^{\text{RL}}\|_2 = \|\sum_s \eta A_s G_s\|_2$，每步都被 advantage $A_s$ 缩放，而 $A_s$ 随收敛衰减到零（$G_s=\nabla_\theta\log\pi_\theta(y_s|x_s)$ 是样本梯度）。于是在可验证场景 $r\in\{0,1\}$ 下 $|A|\le\frac12$，代入冲突范数可得 $\mathbb{E}[\|C\|^{\text{RL}}_{\text{conflict}}] \ll \mathbb{E}[\|C\|^{\text{SFT}}_{\text{conflict}}]$——RL 通过让 advantage 消失，把跨任务冲突一并压下去了。实证上（Figure 5、6）RL 的更新范数和冲突范数随训练步数的增长都明显比 SFT 慢，验证了这套「收敛即停手」的自适应特性。这正是标题「enough is as good as a feast」的来源：模型一旦做对了就不再猛改，省下来的「克制」恰好换来了融合时的低冲突。

**3. 正负样本联合优化导向无偏的任务子空间**

第三个区别是 RL 同时在正样本和负样本上优化，而 SFT 只用正样本（标注答案）。作者设计了一个受控变体 RL-Pos：把所有负样本的 advantage 强行置零、从而排除其梯度贡献，但保留 KL 正则和 on-policy 采样，以此单独隔离「负样本」这一个因素。然后检验两个假设——H1：用正负样本一起训应收敛到更高的单任务精度；H2：相同训练预算下，用正负样本训出的模型融合后掉点更少。Table 3 验证 H1：RL-Pos 虽优于 SFT，但仍明显不如完整 RL（IF 上 RL 90.0 vs RL-Pos 86.1），说明负样本帮助了单任务优化；Figure 7 验证 H2：在 Averaging 和 TIEs 两种合并下，完整 RL 比 RL-Pos 的掉点都更小。结论是正负样本联合优化把模型推向一个「无偏的、任务专属的参数子空间」，既保住单任务性能，又进一步避免了参数冲突。

### 损失函数 / 训练策略
论文复用标准范式而非自创损失：SFT 最小化负条件似然 $\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y)\sim D_{\text{SFT}}}\big[\sum_t \log\pi_\theta(y_t|x,y_{<t})\big]$；RL 最大化期望回报 $J_{\text{RL}}(\theta) = \mathbb{E}_{x\sim D_{\text{RL}}, y\sim\pi_\theta}[r(y,x)]$，主实验用 GRPO（critic-free 的 PPO 变体），并以 PPO、REINFORCE++ 做泛化性验证。除特别说明外，默认基座是 Llama-3.1-8B、默认算法是 GRPO，融合统一做「配对融合（两两合并）」并报对其它任务的平均性能。

## 实验关键数据

### 主实验
五个任务（数学、代码、指令遵循 IF、逻辑谜题 Puzzle、排序 Rank）、四种合并策略下 SFT 与 RL(GRPO) 的融合后性能对比（括号内为相对原始未融合模型的相对掉点，越小越好）：

| 训练 / 合并 | Math | Code | IF | Puzzle | Rank | 平均 |
|------|------|------|------|------|------|------|
| SFT（原始） | 61.9 | 60.5 | 63.9 | 86.2 | 52.8 | 61.5 |
| SFT + Averaging | 52.0 (-16%) | 56.0 (-7.4%) | 49.2 (-23%) | 30.8 (-65%) | 51.6 (-2.3%) | 47.9 (-22%) |
| SFT + TIEs | 56.8 (-8.3%) | 58.0 (-4.1%) | 47.5 (-25%) | 35.8 (-58%) | 51.3 (-2.7%) | 49.9 (-19%) |
| SFT + DARE | 58.2 (-6.1%) | 58.0 (-4.1%) | 46.7 (-27%) | 38.0 (-56%) | 49.3 (-6.7%) | 50.0 (-19%) |
| RL(GRPO)（原始） | 64.6 | 65.6 | 90.0 | 85.2 | 55.7 | 72.2 |
| RL + Averaging | 62.1 (-3.9%) | 61.7 (-5.9%) | 84.4 (-6.2%) | 37.8 (-56%) | 54.4 (-2.3%) | 60.1 (-17%) |
| RL + TIEs | 63.3 (-2.0%) | 64.3 (-2.0%) | 90.0 (-0%) | 64.6 (-24%) | 53.1 (-4.7%) | 67.1 (-7.1%) |
| RL + DARE | 63.5 (-1.7%) | 64.2 (-2.1%) | 89.9 (-0.1%) | 65.0 (-24%) | 53.1 (-4.7%) | 67.1 (-7.1%) |

最直观的对比：SFT 在 TIEs 下平均掉 19%，而 RL 只掉 7.1%；最脆弱的 Puzzle 任务上 SFT 一度掉到 -65%，RL 同样脆弱但相对更稳。结论是 RL 训练的模型在几乎所有任务、所有合并算法下都更抗融合。

### 消融实验
机制层的关键量化（数值取自 Table 2 / Table 3 与正文）：

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| SFT 更新范数 $\|\Delta\theta\|$ | math 6.50 / code 7.75 / IF 4.83 | off-policy，更新幅度最大 |
| RFT 更新范数 | math 2.36 / code 2.17 / IF 1.70 | 仅把数据改成 on-policy，幅度已大降 |
| RL 更新范数 | math 0.78 / code 0.71 / IF 0.64 | on-policy + advantage 衰减，幅度最小 |
| RL 大幅更新参数占比 | 25.0% / 20.7% / 24.1% | 超过 $10^{-5}$ 的参数比例（math/code/IF） |
| SFT 大幅更新参数占比 | 79.9% / 78.0% / 73.9% | 同上，远高于 RL |
| RL-Pos 收敛精度 | Math 58.5 / Code 61.7 / IF 86.1 | 去掉负样本梯度，单任务精度低于完整 RL |
| RL 收敛精度 | Math 64.6 / Code 65.6 / IF 90.0 | 正负样本联合，精度最高 |

### 关键发现
- **掉点来源被区分清楚**：性能 landscape 实验（Figure 4）显示，沿随机方向 $\theta_{\text{rand}}$ 扰动时 SFT 和 RL 都很稳，说明两者对参数噪声都鲁棒；但沿任务诱导方向 $\Delta\theta$ 扰动时，SFT 性能明显下滑、RL 几乎不动——证明 SFT 的掉点是真·任务冲突（参数纠缠），RL 学到的更新则更接近任务正交。
- **三个因子可单独验证、且层层递进**：on-policy（RFT 的中间值证明数据来源就够压幅度）→ advantage 衰减（理论界 + Figure 5/6 的增长趋势）→ 负样本（RL-Pos 受控实验同时验证 H1、H2）。
- **泛化性扎实**：跨 PPO/GRPO/REINFORCE++ 三种 RL 算法、跨 3B/8B/24B 三种基座，RL 优于 SFT 的结论都稳定成立（Figure 2、3），IF 任务上 SFT 一度掉 28.7%~35.6%，而 RL 几乎不掉甚至略涨（+0.4%）。

## 亮点与洞察
- **把「训练范式」抬成融合的一等公民**：以往融合研究只盯着合并算子，这篇论文指出「模型是 SFT 还是 RL 训的」可能比「用哪种合并算法」影响更大——这是一个被长期忽略却很实用的视角。
- **RFT 这个中间变量设计得很聪明**：用「采自原模型的 SFT」把「数据是否 on-policy」从「是否是 RL」里干净地剥离出来，让因果归因更可信，而不是笼统说「RL 就是好」。
- **理论与实证闭环**：advantage 趋零的两条定理直接推出冲突范数的不等式 $\mathbb{E}[\|C\|^{\text{RL}}_{\text{conflict}}] \ll \mathbb{E}[\|C\|^{\text{SFT}}_{\text{conflict}}]$，再被 Figure 5/6 的曲线趋势印证，形成「定理预测 → 曲线验证」的闭环。
- **可迁移的启发**：如果你要训一批将来准备融合的专家模型，这篇论文给出了实操建议——优先用 RL（哪怕只是为了让参数更新更小、更正交），而不是默认 SFT。

## 局限与展望
- **任务局限于「可自动验证」场景**：五个任务都依赖明确的奖励信号（答对/答错），论文的 advantage 趋零分析也建立在 $r\in\{0,1\}$ 的可验证设定上；对开放式生成、偏好对齐这类奖励本身有偏/有噪的任务，结论是否成立尚不清楚。
- **横向数字不可直接比大小**：不同任务的脆弱度差异极大（Puzzle 普遍掉 50%+，Rank 几乎不掉），把不同任务的掉点百分比直接横向比较意义有限，需结合任务本身难度看。
- **「为什么 RL 不掉」更多是相关性解释**：三个因子是作者提出并验证的「假设」，彼此可能存在交互（on-policy 与 advantage 衰减天然耦合），论文没有完全解耦三者的独立贡献量级。
- **改进方向**：能否把「RL 让更新小且正交」的特性反过来用于设计更好的合并算子，或在 SFT 中显式加入 advantage 式的自适应衰减/负样本，去逼近 RL 的低冲突特性，是很自然的后续。

## 相关工作与启发
- **vs 传统模型融合策略（TIEs / DARE / Task-Arithmetic / Fisher 加权）**：它们都在「给定已训练好的模型」前提下设计更聪明的合并算子去剪冗余、对齐符号；本文换了个上游视角，指出「上游怎么训」决定了下游融合的难易，两条路线正交、可叠加。
- **vs 用 loss landscape / 线性模式连通性解释融合的理论工作**：这些工作几乎只分析 SFT 微调的模型；本文首次把分析对象扩展到 RL 后训练，填补了「不同后训练范式如何塑造任务冲突」的空白。
- **vs 单纯讨论 RL 后训练能力提升的工作**：以往关注 RL 让模型在单任务上更强，本文揭示了 RL 一个「副产品」式的优势——其参数更新形态恰好天然适合被融合，这是对 RL 价值的一个新认识。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把「训练范式 × 模型融合」这一被忽视的交叉点系统化，视角新。
- 实验充分度: ⭐⭐⭐⭐⭐ 跨 5 任务、4 合并法、3 RL 算法、3 基座，外加 RFT/RL-Pos 受控变体和理论界，证据链很全。
- 写作质量: ⭐⭐⭐⭐ 现象层到机制层的逻辑递进清晰，个别公式排版有小瑕疵但不影响理解。
- 价值: ⭐⭐⭐⭐ 给「训一批待融合专家模型该用什么范式」提供了可操作的实证答案，实用性强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Reward is Enough: LLMs are In-Context Reinforcement Learners](reward_is_enough_llms_are_in-context_reinforcement_learners.md)
- [\[ICLR 2026\] Mirage or Method? How Model–Task Alignment Induces Divergent RL Conclusions](mirage_or_method_how_modeltask_alignment_induces_divergent_rl_conclusions.md)
- [\[ICLR 2026\] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)
- [\[ICLR 2026\] RL Grokking Recipe: How Does RL Unlock and Transfer New Algorithms in LLMs?](rl_grokking_recipe_how_does_rl_unlock_and_transfer_new_algorithms_in_llms.md)
- [\[ACL 2026\] Good Reasoning Makes Good Demonstrations: Implicit Reasoning Quality Supervision via In-Context Reinforcement Learning](../../ACL2026/reinforcement_learning/good_reasoning_makes_good_demonstrations_implicit_reasoning_quality_supervision_.md)

</div>

<!-- RELATED:END -->
