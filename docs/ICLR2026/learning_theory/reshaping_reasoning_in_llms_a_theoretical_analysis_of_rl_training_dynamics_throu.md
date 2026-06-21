---
title: >-
  [论文解读] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection
description: >-
  [ICLR 2026][学习理论][RL训练动态] 本文把 LLM 的推理抽象成「先选推理模式 $r$、再据此推答案 $a$」的两阶段过程 $q\to r\to a$，用 tabular policy + 梯度流刻画 RLVR 与 RLIF 的训练动态，证明 RLVR 会稳定收敛到成功率最高的推理模式（强基座快收敛、弱基座要经历"纠缠期"），而 RLIF 初期提升、长训却有 50% 概率收敛到最差模式，从理论上解释了二者实测曲线的差异。
tags:
  - "ICLR 2026"
  - "学习理论"
  - "LLM推理"
  - "强化学习"
  - "RL训练动态"
  - "RLVR"
  - "RLIF"
  - "推理模式选择"
  - "收敛分析"
---

# Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2OO399hRD6](https://openreview.net/forum?id=2OO399hRD6)  
**代码**: 待确认  
**领域**: 学习理论 / LLM推理 / 强化学习  
**关键词**: RL训练动态, RLVR, RLIF, 推理模式选择, 收敛分析

## 一句话总结
本文把 LLM 的推理抽象成「先选推理模式 $r$、再据此推答案 $a$」的两阶段过程 $q\to r\to a$，用 tabular policy + 梯度流刻画 RLVR 与 RLIF 的训练动态，证明 RLVR 会稳定收敛到成功率最高的推理模式（强基座快收敛、弱基座要经历"纠缠期"），而 RLIF 初期提升、长训却有 50% 概率收敛到最差模式，从理论上解释了二者实测曲线的差异。

## 研究背景与动机
**领域现状**：RL（尤其是可验证奖励 RLVR 和内部反馈 RLIF）已成为提升 DeepSeek-R1、Qwen3 等推理模型的关键手段。但人们对"RL 训练过程中模型内部到底发生了什么"知之甚少——已有工作分别从 pass@k、策略熵、稀疏关键 token 等角度做过观察，却缺一个能把"实测现象"和"训练动态"串起来的统一解释。

**现有痛点**：实证侧的几个观察互相孤立——有人说 RL 没引入新推理能力、有人说 RL 降低策略熵、有人说 RL 只动了一小撮关键 token，但这些都停留在 high-level 描述，没回答**为什么 RLVR 稳定涨点而 RLIF 会先涨后崩**，也没法预测什么时候 RL 会优化困难。理论侧则因为真实 LLM 太复杂，很难给出可证明的优化保证。

**核心矛盾**：RL 的训练效果高度依赖基座模型质量，但"基座质量"如何通过训练动态最终决定收敛行为，一直是个黑箱。RLVR 和 RLIF 用的是同一套 RL 目标，奖励却根本不同——一个看答案对不对，一个只看模型对自己输出的自信度——这种奖励差异如何传导成完全相反的训练曲线，缺乏机理刻画。

**本文目标**：(1) 用更干净的实证分析确认"RL 只优化稀疏关键 token、从而重塑推理模式分布"；(2) 建一个可分析的数学框架，把 RLVR 和 RLIF 的训练动态（收敛点、收敛速度、何时退化）证出来。

**切入角度**：作者观察到一个关键实证事实——**单个推理模式的内在成功率在训练全程基本不变**，变的是模型选择各模式的概率分布。这提示可以把"答案推导"那一层冻住、只把"模式选择"那一层当作 RL 真正优化的对象，从而把高维的 token 级优化降维成对模式选择概率的分析。

**核心 idea**：用「问题→推理模式→答案」$q\to r\to a$ 的两阶段抽象 + tabular policy 的梯度流，证明 RL 本质是在重排各推理模式的选择概率，由此统一解释 RLVR 的稳定收敛与 RLIF 的先升后降。

## 方法详解

### 整体框架
本文不是提出新算法，而是给 RL 训练 LLM 的过程建立一套「实证 → 理论」的解释框架，分两块。

**实证块**：固定 Qwen2.5-3B 在 MATH 上，分别跑 RLVR 与 RLIF，做三层分析。① 训练曲线层：RLVR 单调涨并收敛，RLIF 先涨后跌甚至跌破基座。② 推理模式层：用 GPT-4o 把基座模型的回答按关键词和逻辑结构聚成若干"推理模式"，跟踪各模式的占比和成功率——发现 RLVR 持续把概率挪向高成功率模式，RLIF 则在各模式间乱晃，而**任一模式自身的成功率全程稳定**。③ token 层：对比 base 与 RL 后模型在每个 token 位置的概率排名，发现每条回答里发生排名变化的位置不到 10%，证明 RL 只在稀疏的关键决策点动手。

**理论块**：把上面的实证事实形式化。给定问题 $q$，模型先以 $\pi_\theta(r|q)$ 选一个推理模式 $r\in R$，再以 $\pi_\theta(a|q,r)$ 推出答案 $a\in A$；每个模式有固定成功率 $p^*(r)=\pi_\theta(a^*|q,r)$。采用 tabular policy（每个输出 token 对应一列可训练 logit，$\pi_\theta(y_l|y_{l-1})=\mathrm{softmax}(\theta_{:,y_{l-1}})_{y_l}$），在小学习率极限下用梯度流 $\frac{d}{dt}\theta(t)=\nabla\phi_{RL}(\theta(t))$ 分析。RL 目标沿用标准形式

$$\phi_{RL}(\theta)=\mathbb{E}_{x\sim D,\,y\sim\pi_\theta}[r_\phi(x,y)]-\beta D_{KL}[\pi_\theta(y|x)\,\|\,\pi_{ref}(y|x)].$$

其中 RLVR 奖励是 $r_\phi=\mathbb{1}[y=\text{ground truth}]$，RLIF 奖励是模型 next-token 分布与词表均匀分布的负平均 KL。框架先给出通用最优策略，再分别对 RLVR、RLIF 推训练动态。

### 关键设计

**1. 稀疏关键 token 重塑推理模式分布：把"RL 改了什么"测干净**

这一块针对"已有实证观察互相孤立、说不清 RL 到底动了什么"的痛点。作者做了模式级与 token 级两套对照实验，得到三个被反复验证的事实：RLVR 持续把选择概率挪向高成功率模式；RLIF 的模式分布不稳定、无法专注到有效模式；**任一模式的内在成功率 $p^*(r)$ 在训练全程几乎不变**。token 级上，用排名变化比例（Ranking Change Ratio）量化——统计每条回答里 RL 前后 token 概率排名发生变化的位置占比，结果在 GSM8K/MATH/AIME 上多数低于 10%（如 MATH-RLVR 从 step20 的 5.3% 到 step100 的 6.6%），而 RLIF 的变化比例普遍更高且随训练递增（MATH-RLIF 升到 8.8%）。这两组实证直接支撑了后续理论的核心简化假设：既然单模式成功率不变，就可以把"模式→答案"这层固定住，只分析"问题→模式"的选择概率怎么演化。

**2. 两阶段推理框架与最优策略刻画：把高维优化降成模式概率的重排**

针对"真实 LLM 太复杂、证不出优化保证"的痛点，本文把推理形式化为 $q\to r\to a$ 两阶段，并引入假设 5.1——每个模式的成功率 $p^*(r)$ 训练中恒定（来自实证观察，理由是"从问题映射到模式"比"从模式映射到正确答案"好优化得多）。在此框架下，作者解出 RL 目标的最优策略（命题 5.2）：

$$\pi_{opt}(r|q)=\frac{1}{Z}\exp\!\Big(\tfrac{1}{\beta}R(r)\Big)\pi_{ref}(r|q),\qquad Z=\sum_{r\in R}\exp\!\Big(\tfrac{1}{\beta}R(r)\Big)\pi_{ref}(r|q),$$

其中 $R(r)$ 是该推理路径的奖励：RLVR 取 $R_{RLVR}(r)=p^*(r)$（直接就是成功率），RLIF 取 $R_{RLIF}(r)=-\frac{1}{|A|}\sum_{a\in A}\log\pi_\theta(a|q,r)$（只是对最终答案分布的"自信度"，**不区分答案对错**）。这条公式是全文枢纽：它说明 RLVR 把概率推向成功率高的模式因而稳定涨点，而 RLIF 的奖励里压根没有"正确"这个信号，最优解不保证比基座更准。又因实践中 $\beta$ 很小，取极限 $\beta\to 0$ 时策略退化为确定性地只选 $\arg\max_r R(r)$ 的模式——RLVR 选成功率最高的，RLIF 选自信度最高的。

**3. RLVR 的两种收敛 regime 与"纠缠期"：基座强弱决定收敛快慢**

命题 5.2 只给了终点，没说怎么走到、走多久。这一设计用 tabular policy 的梯度流刻画 RLVR 到达最优模式 $r^*=\arg\max_r p^*(r)$ 的全过程，分两种情形。**Regime 1（定理 5.3）**：若基座整体准确率 $\mathrm{ACC}_{\theta_{ref}}=\sum_r\pi_{ref}(r|q)p^*(r)$ 已超过所有非最优模式的成功率（$\mathrm{ACC}_{\theta_{ref}}>p^*(r),\,\forall r\neq r^*$），则 $\pi_\theta(r^*|q)$ 单调升到 1，收敛时间 $T_1=O(1/\epsilon)$，多项式级、很快。**Regime 2（定理 5.4）**：若有一个次优模式 $r'$ 的成功率反超了初始整体准确率（$p^*(r')>\mathrm{ACC}_{\theta_{ref}}>p^*(r),\,\forall r\neq r^*,r'$），模型会先陷入"纠缠期"——次优模式 $r'$ 把优化往自己这边拽、拖慢 $r^*$ 上升，必须熬过一段转换时间 $T_0$ 才能进入 Regime 1 的快速收敛。关键在于 $T_0$ 随

$$\gamma_{\pi_{ref}}:=\sum_{r\in R/\{r'\}}\frac{\pi_{ref}(r|q)}{\pi_{ref}(r^*|q)}$$

**超指数级**增长（$T_0$ 量级含 $\gamma_{\pi_{ref}}^{\,C_2\gamma_{\pi_{ref}}}$）。直观含义：当弱基座给最优模式 $r^*$ 的初始概率很小，$\gamma_{\pi_{ref}}$ 就很大，$T_0$ 大到训练实际上动不了——这正是"RLVR 在某些场景优化困难"的理论根源，也对应实测里 $\gamma=(0.25+0.05)/0.05=6$ 的纠缠案例。

**4. RLIF 的"先升后崩"：初期 100% 涨、长训 50% 收敛到最差模式**

针对"RLIF 为何先涨后跌"这个最反直觉的现象，本设计在一个干净设定下证清楚（定理 5.6）：$\beta=0$、tabular policy、基座对各模式高熵均匀选择 $\pi_{base}(r|q)=1/|R|$、$|A|=2$、各路径成功率服从 $U[0,1]$，并设基座满足"正确答案概率最高"（假设 5.5，由 majority voting 有效这一经验事实佐证）。结论是：当 $|R|\to\infty$ 时，(1) 初始时刻准确率导数以概率 1 为正，$P(\frac{d}{dt}\mathrm{ACC}_\theta(t)|_{t=0}>0)=1$——解释了 RLIF 的早期增益；(2) 以 0.5 的概率，$\arg\max_r R_{RLIF}(r)=\arg\min_r p^*(r)$，即长训收敛点有一半概率恰好是**最差**推理模式。两条合起来：RLIF 因为奖励只看自信度、不看对错，初期靠提升整体自信度蹭到涨点，但 $\beta\to 0$ 的确定性收敛会把概率压到自信度最高的模式上，而该模式有 50% 概率正是成功率最低的那个，于是长训跌破基座。仿真（$|R|=10^3,10^4$）显示收敛到改善态与退化态的概率各约一半，且 $|R|$ 越大初期涨点概率越逼近 1，与理论吻合。

## 实验关键数据

### 主实验
实证部分以 Qwen2.5-3B 在 MATH 上做受控对比（RLVR vs RLIF，其余设置相同），核心证据是 token 级排名变化比例——RL 只动稀疏关键 token。

| 任务 | 方法 | Step20 | Step60 | Step100 |
|------|------|--------|--------|---------|
| GSM8K | RLVR | 5.2% | 6.6% | 7.3% |
| GSM8K | RLIF | 6.8% | 8.9% | 10.1% |
| MATH | RLVR | 5.3% | 6.1% | 6.6% |
| MATH | RLIF | 6.1% | 7.7% | 8.8% |
| AIME24 | RLVR | 5.1% | 5.9% | 6.3% |

排名变化比例多数 <10%，证明 RL 选择性地只改一小撮关键决策点；RLIF 的比例更高且随训练递增，对应其分布不稳定。

### 数值仿真（验证理论）

| 现象 | 理论预测 | 仿真结果 |
|------|---------|---------|
| RLVR Regime 1 | $T_1=O(1/\epsilon)$ 快速收敛到 $r^*$ | $\pi(r^*)$ 单调升到 1 |
| RLVR Regime 2 | $T_0$ 超指数（$\gamma=6$） | 出现明显纠缠期再收敛 |
| RLIF 初期 | $\frac{d}{dt}\mathrm{ACC}|_{t=0}>0$ 概率 →1 | $|R|$ 越大越逼近 1 |
| RLIF 长训 | 50% 概率收敛到最差模式 | 改善态 vs 退化态约 48.7% : 51.3% |

### 关键发现
- **单模式成功率全程稳定**是整个理论的实证基石——它让"冻住答案推导层、只分析模式选择"这一降维合法。
- **RLVR 的成败由基座决定**：基座整体准确率是否超过次优模式成功率，直接区分"秒收敛"与"超指数纠缠"两种命运。
- **RLIF 先升后崩有精确刻画**：初期涨点是几乎必然（概率→1），后期崩盘是因为奖励里没有正确性信号，约 50% 概率锁死到最差模式。

## 亮点与洞察
- **把 token 级优化降维成模式选择概率的重排**：通过实证确认"单模式成功率不变"，作者得以冻住 $\pi(a|q,r)$ 只分析 $\pi(r|q)$，是让真实 LLM 的 RL 动态变得可证的关键一招，这个降维思路可迁移到其他"先选策略再执行"的 RL 分析。
- **一条最优策略公式统一两种奖励**：$\pi_{opt}\propto\exp(R(r)/\beta)\pi_{ref}$ 里 RLVR 与 RLIF 只差 $R(r)$ 的定义，直接暴露 RLIF "不看对错"的结构性缺陷，比起堆实验更有说服力。
- **"纠缠期" $T_0$ 的超指数刻画**：用 $\gamma_{\pi_{ref}}$ 一个量把"基座给最优模式的概率太小→训练卡死"讲清楚，给"RLVR 何时优化困难"提供了可计算的判据。

## 局限与展望
- 假设 5.1（单模式成功率训练中恒定）虽有实证支撑，但本质是把"答案推导层不变"理想化了，真实训练中模式内部也会被微调。
- 理论分析依赖 tabular policy、$\beta=0$、$|A|=2$、成功率均匀分布等强简化设定，定理 5.4、5.6 都坦承只在特定 regime 成立，能否推广到一般 LLM 仍待验证。
- 推理模式由 GPT-4o 聚类得到，其可解释性和稳定性需要更多验证——"模式"本身是否客观存在、边界如何，会影响所有下游结论。
- 作者承认框架需扩展到更复杂的真实推理场景，并放宽对基座模型的特定假设。

## 相关工作与启发
- **vs Yue et al. 2025a / Huan et al. 2025（稀疏关键 token 观察）**：他们指出 RL 只优化少量关键 token，本文用更干净的模式级 + token 级对照把这一点测得更扎实，并进一步把它接进可证明的训练动态理论，而不止于现象描述。
- **vs Cui et al. / Zhang et al.（熵视角）**：他们从策略熵下降解释 RL，本文换成"推理模式选择"的视角，能同时解释 RLVR 稳定涨点和 RLIF 先升后崩，比单纯熵塌缩更贴合两类奖励的差异。
- **vs Agarwal et al. 2025（用熵最小化解释 RLIF 增益）**：他们解释了 RLIF 为何能涨，本文补上"为何长训会崩"——指出 RLIF 奖励不区分对错、$\beta\to 0$ 时有 50% 概率锁到最差模式，给出了更完整的双向刻画。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次用 $q\to r\to a$ 两阶段框架把 RLVR/RLIF 训练动态证出来，视角和结论都新。
- 实验充分度: ⭐⭐⭐⭐ 实证 + 仿真双线验证理论，但缺真实大模型上对定理预测的直接检验。
- 写作质量: ⭐⭐⭐⭐ 实证与理论衔接清晰，定理表述严谨；强简化设定对读者门槛较高。
- 价值: ⭐⭐⭐⭐⭐ 给"RLVR 何时难收敛、RLIF 何时会崩"提供了可计算判据，对 RL 后训练实践有直接指导意义。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Theoretical Analysis of Contrastive Learning under Imbalanced Data: From Training Dynamics to a Pruning Solution](theoretical_analysis_of_contrastive_learning_under_imbalanced_data_from_training.md)
- [\[ICLR 2026\] A Theoretical Analysis of Mamba's Training Dynamics: Filtering Relevant Features for Generalization in State Space Models](a_theoretical_analysis_of_mambas_training_dynamics_filtering_relevant_features_f.md)
- [\[ICLR 2026\] Optimizing Data Augmentation through Bayesian Model Selection](optimizing_data_augmentation_through_bayesian_model_selection.md)
- [\[ICLR 2026\] On the Computational Limits of AI4S-RL：A Unified $\varepsilon$-$N$ Analysis](on_the_computational_limits_of_ai4s-rl_a_unified_varepsilon-n_analysis.md)
- [\[ICLR 2026\] High-dimensional Analysis of Synthetic Data Selection](high-dimensional_analysis_of_synthetic_data_selection.md)

</div>

<!-- RELATED:END -->
