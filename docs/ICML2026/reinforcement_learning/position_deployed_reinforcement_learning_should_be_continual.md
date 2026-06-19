---
title: >-
  [论文解读] Position: Deployed Reinforcement Learning should be Continual
description: >-
  [ICML 2026][强化学习][持续强化学习] 本文是一篇立场论文：作者主张凡是部署后仍能拿到评价性奖励信号、且环境复杂度超出 agent 表征/计算能力的 RL 系统，本质都是一个持续强化学习（CRL）问题，应当抛弃"训练完就冻结"的范式，让 agent 在部署中持续更新策略。 领域现状：RL 的标志性成就（TD-Ga…
tags:
  - "ICML 2026"
  - "强化学习"
  - "持续强化学习"
  - "可测量部署"
  - "历史过程"
  - "非平稳性"
  - "train-then-fix"
---

# Position: Deployed Reinforcement Learning should be Continual

**会议**: ICML 2026  
**arXiv**: [2606.04029](https://arxiv.org/abs/2606.04029)  
**代码**: 无（position paper）  
**领域**: 强化学习 / 持续学习 / 部署后适应  
**关键词**: 持续强化学习, 可测量部署, 历史过程, 非平稳性, train-then-fix

## 一句话总结
本文是一篇立场论文：作者主张凡是部署后仍能拿到评价性奖励信号、且环境复杂度超出 agent 表征/计算能力的 RL 系统，本质都是一个持续强化学习（CRL）问题，应当抛弃"训练完就冻结"的范式，让 agent 在部署中持续更新策略。

## 研究背景与动机
**领域现状**：RL 的标志性成就（TD-Gammon、AlphaGo、OpenAI Five、GT Sophy、平流层气球、Tokamak 控制）几乎都遵循 train-then-fix 范式——离线大量训练后冻结策略部署。这一惯例既来自工程稳定性的需要，也来自 MDP 形式化下"收敛到 $\pi^\star$"的数学传统。

**现有痛点**：冻结策略在真实部署中无法持续保持性能，需要靠周期性 retraining 维持，性能曲线呈锯齿状（先衰减、再人工触发重训）。Cursor Tab、Lyft 这类系统每天处理数亿请求，固定策略无法跟上用户行为、库版本、市场结构的变化；机器人 sim-to-real 也证明固定策略遇到磨损、光照、传感器漂移就失效。

**核心矛盾**：传统 MDP 形式化假设环境平稳、状态可访问、存在不动点 $\pi^\star$，因此学习被建模成"一次性求解"。但 Big World Hypothesis 指出真实环境的复杂度远超任何 agent 的表征容量，optimal policy 既不可表达也不可达；同时部署后还有 action-induced 非平稳、动态漂移、目标演化、emergent novelty 四类非平稳源。一个被"求解-冻结"思维约束的 agent，注定要不断让出性能给环境。

**本文目标**：(1) 把"部署后仍能收到评价反馈"这一常见场景正式命名为 measurable deployment；(2) 用 history process 形式化论证它本质就是 CRL 问题；(3) 给从业者和研究者各开一份行动清单。

**切入角度**：从 Abel 等人 2023 年对 CRL 的定义出发——"best agent 永远不停止学习的问题"——结合 Bowling 等人提出的 history process 形式化，把"是否需要持续学习"从算法属性还原为问题属性。

**核心 idea**：当奖励信号还在、而最优策略不在可达策略集中时，"停止搜索"就是次优行为；measurable deployment 的最优解就是把 deployment 本身当成学习过程。

## 方法详解
作为立场论文，本文没有新算法，而是给出一套形式化论证 + 三个真实部署案例 + 两类受众的行动清单。

### 整体框架
论证链条由四块拼成：(1) 用 history process 重写 RL 形式化，绕开 MDP 的平稳性/可重置假设；(2) 列举 measurable deployment 中四类非平稳源，论证它必然是 CRL 问题；(3) 用 Cursor Tab、Lyft、Sim-to-Real 三个案例对应不同非平稳源；(4) 引入 continual learner vs non-continual learner 的二分，把"是否持续"还原为 learning rule $\sigma$ 是否会终止策略集搜索。

### 关键设计

**1. Measurable Deployment 的形式化定义：把"该不该继续学"变成一个可判定条件**

MDP 这套语言自带"存在不动点 $\pi^\star$"的暗示，把研究者诱导进"训练完就结束"的思维定势。作者改用 history process 来描述环境——$e:\mathcal H\times\mathcal A\to\Delta(\mathcal O)$，其中 $\mathcal H=\bigcup_{n=0}^\infty(\mathcal A\times\mathcal O)^n$ 是所有有限历史，agent 是策略 $\pi:\mathcal S\to\Delta(\mathcal A)$ 加 learning rule $\sigma:\mathcal H\to\Delta(\Pi)$。这套语言不假设可重置、不假设 Markov、不假设可重访状态，因此更贴合真实部署。在此之上，一个部署被称为 measurable 当且仅当：(i) 它处于 big world regime，最优策略 $\pi^\star$ 落在可达策略集 $\Pi$ 之外或计算上不可达；(ii) 部署后仍能持续收到 evaluative reward。一旦两条同时满足，best agent 就不可能终止搜索，问题必然落入 CRL。这一步把"部署后该不该继续学"从一场哲学辩论变成了一个可勾选的形式化判据。

**2. 四类部署后非平稳源：把"为什么必须 CRL"拆成可对照自查的四个维度**

光说"环境非平稳"太抽象，工程师没法判断自家系统算不算。作者把非平稳拆成四个可识别的来源：(i) Action-induced——agent 自己的动作改变未来 history 分布，如推荐系统重塑用户偏好、交易策略改变市场，和 performative prediction 文献紧密相关；(ii) 环境动态变化——季节、硬件老化、市场结构、监管这类外因；(iii) 目标演化——按 reward hypothesis，目标本身会变，多目标场景里权重也会随时间漂移；(iv) Emergent novelty——Big World Hypothesis 保证任何有限容量 agent 必然撞上训练时没见过的 action-observation 序列，黑天鹅是其极端形态。论文再用 Cursor Tab、Lyft、Sim-to-Real 三个案例对照，标出每类源在哪些系统里是 Primary / Present / Implicit，让"我的系统是不是 CRL"成为一道可勾选的检查题。

**3. Continual vs Non-Continual Learner 的二分：把"是否持续"进一步还原到 learning rule**

很多人把 catastrophic forgetting 或 plasticity loss 误当作 CRL 的定义性特征，但那其实是算法在 CRL 上的副作用，不是问题本身的特征。作者在 history process 视角下把 learning 看成对策略集 $\Pi$ 的搜索：agent 要么在某个 history 停止搜索、锁定一个策略（non-continual learner），要么永不终止搜索（continual learner）。一个最小例子说得很清楚——64 参数小网络 + SGD，若 step-size 退火到 0 就是 non-continual；若用 IDBD 这类 meta-gradient 让 step-size 永不归零，就是 continual。于是 CRL 的定义可以精确表述为"其 best agent 不能终止搜索的问题"，而且这个定义同时能切问题侧和算法侧——换 $\Pi$ 或换 $\sigma$ 都可能把同一物理环境从 CRL 翻成 non-CRL 或反过来。严格区分 problem-side characterization 和 solution-side challenges，正是为了避免社区把"我们解决了 forgetting"误当成"我们解决了 CRL"。

### 论证策略
论文用三个真实部署案例（Cursor Tab、Lyft 出行调度、Sim-to-Real 机器人）做存在性证明：在已经成功的工业 CRL 系统中，每一类非平稳源都至少在一处是 primary driver，且采用持续学习确实带来量化收益（Cursor Tab：建议减少 21%、接受率提升 28%；Lyft：年增 \$30M 收入、数百万次额外完单）。同时论文给出 Rusting Pendulum 这一极简实验，证明 joint friction 累积下 train-then-fix 策略失效而持续学习者维持性能。

## 实验关键数据

### 案例对照表
本文用一张关键表把三个真实部署系统按四类非平稳源对齐：

| 非平稳源 | Cursor Tab | Lyft | Sim-to-Real |
|----------|------------|------|-------------|
| Action-induced NS | Implicit | **Primary** | Implicit |
| 环境动态变化 | Implicit | Present | **Primary** |
| 目标演化 | Present | Implicit | Implicit |
| Emergent Novelty | **Primary** | Present | Present |

Primary 表示该案例的主导驱动力，Present 表示明显存在，Implicit 表示存在但不突出。表中三列横跨推荐/匹配/控制三类典型部署，覆盖全部四类源。

### 工业部署收益

| 系统 | 量化收益 | 持续学习节奏 |
|------|----------|--------------|
| Cursor Tab | 每日 4 亿次请求；建议数 −21%、接受率 +28% | 1.5–2 小时一轮策略更新 |
| Lyft 匹配 | 每年数百万次额外完单、+\$30M 收入 | 在线 RL + switchback 安全验证 |
| Rusting Pendulum | Train-then-fix 随摩擦累积退化；continual learner 维持性能 | 实验性 toy 环境 |

### 关键发现
- 三个工业系统都靠 evaluative reward（接受率、完单率、性能指标）做在线更新；本文强调这种信号在已有部署中往往已经存在，只是被弃用。
- Cursor Tab 选 policy gradient → 强制 on-policy → 强制 1.5–2 小时迭代周期，说明 solution-level 约束会反向塑造工程实践；这是 "solution challenge" 而非 "problem characteristic"。
- Lyft 工程师承认"信任一个会自我更新的算法很难"，靠 switchback 实验做安全验证；本文据此推荐"部署前验证 + 持续在线验证 + fallback 策略"的三层保障。

## 亮点与洞察
- **把 deployment 直接定义成学习过程**：传统 MLOps 把 deployment 当作"训练终点+服务起点"，本文把它翻转成"deployed model 就是学习系统、生产数据就是训练数据"，这一视角转换比任何算法贡献都更有杠杆。
- **history process 形式化的工程意义**：从 MDP 切到 history process 不只是数学美化，而是把"是否可重置"、"是否可重访"这些被默认成立的工程假设暴露出来；任何真实部署其实都违反这些假设，因此 MDP 下的收敛性保证在部署后基本失效。
- **problem vs solution 区分**：把 catastrophic forgetting / plasticity loss 还原为算法层挑战、把非平稳源还原为问题层特征，避免社区继续把"我们解决了 forgetting"等同于"我们解决了 CRL"。
- **可迁移 trick**：用 controlled non-stationarity（扰动 reward、偏移观测、模拟 concept drift）压力测试系统适应性，作者把它推荐为 CRL 部署前的标准 dev practice，思路可直接借到生产 ML pipeline 的 chaos engineering。

## 局限与展望
- 作者承认 measurable deployment 的范围限定较窄——奖励稀疏、延迟、噪声、不可观测的场景（如家用 Roomba 无法获得清扫质量评价）不在 thesis 覆盖范围内；Monitored-MDP 是一个候选扩展但在 history process 下尚无收敛保证。
- 本文的 Rusting Pendulum 是极简 demo，工业案例都是事后回溯解释，缺少受控对照来量化"持续 vs 固定"的差距；社区急需 big-world simulator 类基准但目前仍是空白。
- 在 safety 论证上，作者主张"adaptation safer than stagnation"，但对如何在持续学习下做形式化安全验证只给了方向（shielded RL、constrained MDP、cautious agent），没有给出 deployment-ready 的方案。
- 立场层面的空白：本文没有详细讨论 reward hacking / Goodhart 效应在持续部署下如何恶化，这可能是 measurable deployment 真正落地时的最大障碍。

## 相关工作与启发
- **vs Abel et al. (2023) "A Definition of Continual RL"**：Abel 给出 CRL 的形式化定义，本文把这一定义直接套到工业部署，并加入 measurable deployment 这一中间概念，把研究 community 的兴趣引向已经在用 RL 的真实系统。
- **vs Big World Hypothesis (Javed & Sutton 2024)**：BWH 论证 agent 容量永远小于世界复杂度；本文把这一假设作为 measurable deployment 必须 CRL 的存在性论据。
- **vs Khetarpal et al. (2022) CRL survey**：survey 整理了 CRL 算法挑战，本文则把焦点拉回 problem setting，呼吁不要把"算法解决了 forgetting"和"解决了 CRL 问题"混为一谈。
- **vs Alberta Plan (Sutton et al. 2022)**：Alberta Plan 是长期研究 roadmap，本文是短期 deployment 行动手册，两者互补。

## 评分
- 新颖性: ⭐⭐⭐⭐ 概念合成性强（measurable deployment + 4 类非平稳源 + 工业案例对照表），但底层定义来自 Abel 等人。
- 实验充分度: ⭐⭐⭐ 作为 position paper 主要靠工业案例和 toy demo，缺少受控对照实验。
- 写作质量: ⭐⭐⭐⭐⭐ 论证链条清晰，案例-理论穿插得当，把抽象的 CRL 问题翻译成工程师能立刻自查的清单。
- 价值: ⭐⭐⭐⭐⭐ 给 RL 落地社区指出一个明确方向，且配套有可执行的实践建议；ICML 2026 position track 的高价值代表。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Position: Lifetime Tuning is Incompatible with Continual Reinforcement Learning](../../ICML2025/reinforcement_learning/position_lifetime_tuning_is_incompatible_with_continual_reinforcement_learning.md)
- [\[ICML 2026\] Shapley Neuron Values for Continual Learning: Which Neurons Matter Most?](shapley_neuron_values_for_continual_learning_which_neurons_matter_most.md)
- [\[ICML 2025\] Continual Reinforcement Learning by Planning with Online World Models](../../ICML2025/reinforcement_learning/continual_reinforcement_learning_by_planning_with_online_world_models.md)
- [\[ICLR 2026\] Principled Fast and Meta Knowledge Learners for Continual Reinforcement Learning](../../ICLR2026/reinforcement_learning/principled_fast_and_meta_knowledge_learners_for_continual_reinforcement_learning.md)
- [\[CVPR 2026\] Resolving the Stability-Plasticity Dilemma in Reinforcement Learning via Complementary Continual Critics](../../CVPR2026/reinforcement_learning/resolving_the_stability-plasticity_dilemma_in_reinforcement_learning_via_complem.md)

</div>

<!-- RELATED:END -->
