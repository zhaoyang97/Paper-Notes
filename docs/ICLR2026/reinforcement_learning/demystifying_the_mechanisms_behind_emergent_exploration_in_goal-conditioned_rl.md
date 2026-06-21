---
title: >-
  [论文解读] Demystifying The Mechanisms Behind Emergent Exploration in Goal-Conditioned RL
description: >-
  [ICLR2026][强化学习][目标条件RL] 本文用认知科学的"理性分析 + 干预实验 + 简化建模"三件套，拆解了无奖励的 Single-Goal Contrastive RL（SGCRL）为什么能自发探索——结论是：actor 实际在最大化一个由 critic 表征塑造出来的隐式奖励（状态与目标的表征相似度 $\psi$-similarity），而这套探索-利用动态来自对比学习得到的**低秩表征**，而非神经网络函数逼近。
tags:
  - "ICLR2026"
  - "强化学习"
  - "目标条件RL"
  - "涌现探索"
  - "对比学习"
  - "隐式奖励"
  - "低秩表征"
---

# Demystifying The Mechanisms Behind Emergent Exploration in Goal-Conditioned RL

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=mwgYORsqtv](https://openreview.net/forum?id=mwgYORsqtv)  
**代码**: 项目页 https://mahsa-bastankhah.github.io/demystifying-single-goal-exploration/  
**领域**: 强化学习  
**关键词**: 目标条件RL, 涌现探索, 对比学习, 隐式奖励, 低秩表征

## 一句话总结
本文用认知科学的"理性分析 + 干预实验 + 简化建模"三件套，拆解了无奖励的 Single-Goal Contrastive RL（SGCRL）为什么能自发探索——结论是：actor 实际在最大化一个由 critic 表征塑造出来的隐式奖励（状态与目标的表征相似度 $\psi$-similarity），而这套探索-利用动态来自对比学习得到的**低秩表征**，而非神经网络函数逼近。

## 研究背景与动机
**领域现状**：近年深度 RL 里出现了不少"涌现行为"——智能体在没有显式奖励、没有人为课程的情况下也能学会复杂技能。一个代表是 SGCRL（Liu et al., 2025）：它只盯着**一个固定的难目标**收集数据，用时序对比学习训练 critic，就能在长程目标到达任务里学会操作、移动等技能，甚至比基于"目标分布 / 子目标课程"的方法还强。

**现有痛点**：大家观察到了这个现象，却说不清它**为什么**会发生。坊间常把涌现归因于"模型够大"，但本文指出这条经验法则在这里站不住——词向量能做类比推理是损失函数的功劳而非架构的功劳，那 SGCRL 的探索到底是算法本身的性质，还是神经网络泛化的副产品？不搞清楚驱动因素，就无法预测探索何时、如何、为何出现，也就无法安全可靠地使用这类系统。

**核心矛盾**：SGCRL 训练时**完全不用外部奖励**，却表现出明确的探索-利用切换。一个无奖励的算法，凭什么知道往哪走、什么时候该停下来利用？这个"奖励从哪来"的问题是全文要回答的核心。

**本文目标**：回答一个具体问题——SGCRL 在没有任何明显内/外在奖励时，为什么能高效探索？并把答案拆成可被实验证伪的预测。

**切入角度**：作者刻意不走"在 benchmark 上调性能"的标准 ML 套路，而是借认知科学研究智能行为的工具箱——**理性分析**（看智能体最优化什么目标）、**干预实验**（改变某个变量看行为如何变）、**简化建模**（用尽量小的模型复现行为）。

**核心 idea**：把 SGCRL 的 actor 目标重新解读为"最大化一个隐式奖励"，这个奖励就是 critic 学到的表征相似度 $\psi$-similarity；再证明这套机制在去掉神经网络的**表格化**设定下依然成立，从而把涌现探索归因到低秩对比表征本身。

## 方法详解

### 整体框架
本文是一篇**分析性论文**，"方法"是一套理解机制的工具链而非新算法。被研究的对象 SGCRL 是一个 actor-critic 框架：critic 用 $\phi(s,a)^\top\psi(s_f)$ 估计状态-动作对 $(s,a)$ 通向未来状态 $s_f$ 的可能性，用后向 InfoNCE 损失训练（正样本是沿轨迹往前 $\Delta\sim\text{Geom}(1-\gamma)$ 步采到的未来状态，负样本来自边缘分布）；所有表征做 $\ell_2$ 归一化。训练好后 critic 等价于一个 log-Q：$\phi(s,a)^\top\psi(s_f)=\log p^\pi_\gamma(s_f\mid s,a)-\log p(s_f)$。actor 则选动作去最大化到达目标的似然 $\phi(s,a)^\top\psi(g)+\tau H(\pi)$，离散动作下就是对 $\frac{1}{\tau}\phi(s,a)^\top\psi(g)$ 做 softmax。

作者的分析分三层推进：先在 **4.1** 证明 actor 其实在最大化一个隐式奖励（$\psi$-similarity）；再在 **4.2** 分析这个隐式奖励如何随表征更新而演化——找到目标前抑制已探索区、找到目标后强化成功路径；最后用 **表格化 SGCRL** 这个去掉神经网络的极简模型，验证整套动态来自对比学习的低秩表征而非网络逼近。整条逻辑围绕"actor 和 critic 两个玩家的相互作用"展开：actor 奔向高相似度区域，critic 在没目标时把这些区域的相似度压下来。

### 关键设计

**1. 理性分析：把无奖励的 actor 目标解读成最大化隐式奖励**

这一步直接回答"奖励从哪来"。作者基于 InfoNCE 的一个对齐性质（Assumption 1：优化收敛时，状态-动作对的表征会对齐到其未来状态表征的期望，$\phi(s,a)=\mathbb{E}_{s_f\sim p^\pi_\gamma}[\psi(s_f)]$；附录给出了在各向同性高斯分布下可证成立的条件）推出 Theorem 1：尽管 SGCRL 名义上只在最大化"到达目标 $g$ 的概率"，它**等价于**最大化如下回报

$$Q_\psi(s,a):=\mathbb{E}_{p^\pi_t}\!\left[\sum_{t=0}^{\infty}\gamma^t\,\psi(s_t)^\top\psi(g)\;\Big|\;s_0=s,a_0=a\right].$$

也就是说，逐步奖励就是当前状态与目标的表征相似度 $\psi$-similarity $:=\psi(s_f)^\top\psi(g)$。它可以理解成智能体对"目标在哪"的当前信念，随着数据增多由 critic 不断修正，类似后验采样里维护一个对奖励/转移的后验来指导探索。这就解释了为什么无外部奖励还能有方向——奖励是表征内生的，且 $\psi$-similarity 越密、结构越好，探索越有效（初始时起点附近的高相似度给了"离开起点"的梯度，但相似度必须随时间在无目标区域衰减，否则智能体会被困住）。作者也点明这与 successor representation/features 形似但本质不同：后者要显式训练状态特征再学后继特征，而 SGCRL 的特征与其后继表征都从 InfoNCE 目标里自然涌现，且重点是"表征本身如何驱动探索"。

**2. 找到目标前：InfoNCE 把已探索状态推到与目标正交，自动剪枝搜索空间**

光有隐式奖励还不够，得解释探索为什么不会原地打转。Theorem 2（非正式）分析一个简化设定：$\psi(g)$ 固定、智能体在不含目标的区域活动。设初始表征都带一个平行于 $\psi(g)$ 的共享分量加上独立噪声，$\phi^{(0)}(s_i,a_i)=c\,\psi(g)+\zeta_i$、$\psi^{(0)}(s_{f,i})=c\,\psi(g)+\kappa_i$，在足够大 batch、足够小学习率下用 InfoNCE 梯度更新，则系统**高概率收敛到** $\phi(s_i,a_i)^\top\psi(g)=\psi(s_{f,i})^\top\psi(g)=0$，即这些状态被推到与目标正交。直觉是：InfoNCE 对所有表征加同一向量是不变的，平行于 $\psi(g)$ 的共享分量学不到时序差异，而归一化表征容量有限，于是会**压掉这个冗余分量**省给真正有用的方向。后果是已频繁访问、却没找到目标的状态相似度被持续拉低，actor 自然不再回头，相当于一个**自动生成的子目标课程**在剪枝搜索空间。这个证明只用 InfoNCE 的不动点分析，**完全不依赖神经网络逼近**——这是"机制来自对比学习而非网络"的关键理论支撑。作者还说明该结论对 $\psi(g)$ 的轻微漂移稳定（漂 $\epsilon$，正交性最多差 $\epsilon$），且在连续设定下经验上依然成立。

**3. 找到目标后：成功轨迹留下高相似度"痕迹"，触发探索到利用的切换**

机制的另一半是利用。由于对比学习会对齐正样本表征（Assumption 1），一旦目标 $g$ 被到达，它就作为成功轨迹上各状态的正样本出现，于是这条路径上的状态表征会**向 $\psi(g)$ 对齐**，沿途留下一条从起点到目标的高 $\psi$-similarity "痕迹"。虽然完全对齐无法理论保证，但实验中这条痕迹稳定出现并可靠地把智能体引回目标，标志着行为从探索切换到利用。与传统内在奖励（新颖性奖励、计数奖励、预测误差奖励等）相比，SGCRL 的内在奖励不是人为设计后再加到任务奖励上、靠超参调平衡，而是**直接从 actor 目标里涌现**，无需额外调参，且自带一个直观保证：它恰好对应"在智能体当前对候选目标状态的认知下，最大化到达目标的概率"。

**4. 简化建模：表格化 SGCRL 隔离出"低秩表征"才是真正的功臣**

最后一步是因果归因。既然 Theorem 2 不需要网络，作者就做一个去掉神经网络的**表格化 SGCRL**：每个状态 $s$ 在查找表里存一个嵌入 $\psi(s)$，用 InfoNCE 梯度规则更新；假设确定性转移并直接用真值动力学 $s_{t+1}=p(s_t,a_t)$ 替代学习 $\phi(s,a)$，policy 按 softmax 选动作，所有表征初始化为共享高斯种子加微小独立噪声。在表格化 Tower of Hanoi 与 FourRooms 上，这个极简模型同样表现出"找到目标前负相关、找到后正相关"的两阶段动态。更关键的是一个**对照消融**：把向量化表征换成一张 $|S|\times|S|$ 的状态-目标相似度标量查找表、用同样的对比目标更新，智能体就探索失败，需要约 100 倍样本。这说明驱动探索的不是对比目标本身、也不是网络泛化，而是**对比学习得到的低秩（低维归一化）表征**所施加的几何约束——正是这种"容量有限"让 InfoNCE 能压掉冗余共享分量、把无效状态推开。

## 实验关键数据

实验全部围绕两个研究问题：RQ1（critic 表征如何演化以促成探索/利用）和 RQ2（critic 表征如何影响 actor 的数据收集）。任务包括 2D 点迷宫导航（FourRooms、L 形墙、螺旋墙）和 Tower of Hanoi 目标到达；不用奖励/子目标，到达即成功，曲线为 8 个随机种子平均，阴影为一个标准误。

### 主实验（RQ1：表征演化出现清晰的两阶段）

| 设定 | 现象 | 说明 |
|------|------|------|
| 表格化 SGCRL（Tower of Hanoi, 3 盘，4/5 盘同样成立） | 状态访问量与目标相似度的 Pearson $r$：找到目标前转为**负相关**、找到后转为**正相关** | 验证 Theorem 2 预测的两阶段动态，且无需神经网络 |
| 标准 SGCRL（连续，固定朝某方向收集数据） | 频繁经过路径上的状态 $\psi$-similarity 随训练**系统性下降** | 连续设定下复现"已探索区被推离目标" |
| 假想目标 $\psi(g)=z$ + PCA 投影 | 表征初始聚在目标附近，逐渐漂向与 $z$ 正交的"赤道"，但仍保留房间级局部结构；先访问的房间先被推离 | 直观印证正交化与访问顺序的对应 |

### 消融 / 干预实验

| 配置 | 关键结果 | 说明 |
|------|---------|------|
| 向量化低秩表征（完整） | 高效探索 | 完整机制 |
| 换成 $|S|\times|S|$ 标量相似度查找表 | 需 **~100×** 样本，探索失败 | 证明低秩表征是关键，而非对比目标本身 |
| 单目标 vs 均匀采样多目标数据收集（FourRooms） | 均匀采样**无法**把频繁访问状态推离目标，探索失败 | actor 通过数据收集主动塑造表征，不只是被动消费 |
| 干预：把右上房间一块 patch 表征设为 $\psi(g)$ | 智能体强烈被吸引、该房间访问量大增，甚至绕路去 patch | RQ2：行为由表征相似度这个隐式奖励驱动 |
| 干预：把某房间表征设为 $-\psi(g)$ | 智能体在训练与测试时都**系统性避开**该区域，并找到替代路径 | 表征设计可实现安全感知探索 |

### 关键发现
- **低秩表征是因，不是果**：把同样的对比更新作用在标量查找表上就崩，说明"容量受限的低维归一化嵌入"提供的几何约束才是探索动态的来源；自监督学习里通常嫌弃低秩损害下游分类，但在 SGCRL 里低秩反而是成功的必要条件。
- **actor 既是消费者也是生产者**：单目标数据收集策略会主动把频繁访问状态推离目标，均匀多目标采样做不到，这解释了 Liu et al. (2025) "单目标反超子目标课程"的反直觉结果。
- **可干预、可用于安全**：直接编辑表征（设成 $\psi(g)$ 或 $-\psi(g)$）就能让智能体趋近或回避特定区域，提示"用表征设计代替奖励工程"来控制行为的可能。
- **与经典探索算法同构**：$\psi$-similarity 的演化类似 R-MAX——一开始假设所有状态都给最高奖励（等价于初始高相似度），随访问逐步修正；为把 R-MAX/PSRL 的良好理论性质带到高维长程任务指了条路。

## 亮点与洞察
- **把"无奖励"翻译成"隐式奖励"**：Theorem 1 用一个对齐假设就把看似无目标函数的 actor 重写成最大化 $\psi$-similarity 的回报，这是全文最"啊哈"的一步——它让一个自监督算法的行为变得可解释、可预测、可干预。
- **用表格化模型做因果隔离**：去掉神经网络后机制仍在，再加一个"标量查找表反而崩"的对照，干净利落地把功劳判给低秩表征而非网络，是非常漂亮的可解释性实验设计，方法论上可迁移到其他 RL 算法的归因研究。
- **认知科学工具箱迁移到 ML**：理性分析 + 干预 + 简化建模这套组合，给"理解涌现行为"提供了一个不依赖刷 benchmark 的范式，值得在分析其他复杂学习系统时借用。
- **低秩=优点的反转**：把通常被视为缺陷的"表征容量受限"重新论证成探索机制的必要条件，这个视角反转对表征学习也有启发。

## 局限与展望
- **依赖理想化假设**：核心理论建立在 InfoNCE 的对齐假设（Assumption 1）和 $\psi(g)$ 固定（表格化设定下严格成立）之上；连续/共享编码器下只能保证近似（$\epsilon$ 漂移、经验验证），找到目标后"沿成功轨迹完全对齐"也无法理论保证、只有经验支持。
- **任务规模偏小**：实验集中在 2D 点迷宫和 Tower of Hanoi 这类低维/结构化任务，是否能解释高维像素、真实机器人等更复杂场景下的涌现探索，仍需进一步验证。
- **安全应用尚属初步**：用表征干预实现安全感知探索的结果是 preliminary 的概念验证，离实用的安全约束方法还有距离。
- **可改进方向**：把这套"理性分析 + 干预 + 简化建模"框架系统化，用于解释更多 RL 算法的意外行为；以及探索如何把 R-MAX/PSRL 的可证效率真正搬到高维长程任务中。

## 相关工作与启发
- **vs 透明性/事后可解释方法（如低维表征、子目标分层 RL、决策树蒸馏、显著图）**：这些方法要么改架构、加辅助训练任务，要么把算法当黑盒事后解释；本文从**算法目标本身**出发解释行为，无需额外训练开销。
- **vs successor representation / successor features**：形式上 $\psi(s)$ 像状态特征、$\phi(s,a)$ 像后继特征预测，但后继特征方法需显式训练状态特征再学后继，且服务于快速奖励迁移；SGCRL 的特征与后继表征都从 InfoNCE 涌现，且关注表征如何**自身驱动探索**。
- **vs 传统内在奖励（新颖性/计数/预测误差奖励）**：传统方法启发式设计、手动叠加到任务奖励、靠超参调平衡；SGCRL 的内在奖励从 actor 目标涌现，无需调参，并有"最大化当前认知下到达概率"的理论解释。
- **vs 经典可证高效探索（R-MAX / PSRL）**：二者共享"先乐观假设、随访问修正"的动态，本文借此把经典算法的理论性质与高维长程任务连起来。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把无奖励 actor 解读成最大化隐式表征奖励、并用表格化模型把功劳归给低秩表征，是少见且扎实的机制解释
- 实验充分度: ⭐⭐⭐⭐ 理论预测都有对应的受控/干预实验佐证，但任务规模偏小、安全应用仍属初步
- 写作质量: ⭐⭐⭐⭐⭐ 理论-实验逐条对应、研究问题清晰，认知科学方法论叙事完整
- 价值: ⭐⭐⭐⭐⭐ 既解释了 SGCRL 的反直觉成功，又提供了一套可迁移的涌现行为分析范式与安全控制思路

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] First-Order Representation Languages for Goal-Conditioned RL](../../AAAI2026/reinforcement_learning/first-order_representation_languages_for_goal-conditioned_rl.md)
- [\[ICLR 2026\] Multistep Quasimetric Learning for Scalable Goal-Conditioned Reinforcement Learning](scaling_goal-conditioned_reinforcement_learning_with_multistep_quasimetric_dista.md)
- [\[CVPR 2026\] MangoBench: A Benchmark for Multi-Agent Goal-Conditioned Offline Reinforcement Learning](../../CVPR2026/reinforcement_learning/mangobench_a_benchmark_for_multi-agent_goal-conditioned_offline_reinforcement_le.md)
- [\[ICML 2026\] Direction-Conditioned Policies via Compositional Subgoal Scoring for Online Goal-Conditioned Reinforcement Learning](../../ICML2026/reinforcement_learning/direction-conditioned_policies_via_compositional_subgoal_scoring_for_online_goal.md)
- [\[ICLR 2026\] Occupancy Reward Shaping: Improving Credit Assignment for Offline Goal-Conditioned Reinforcement Learning](occupancy_reward_shaping_improving_credit_assignment_for_offline_goal-conditione.md)

</div>

<!-- RELATED:END -->
