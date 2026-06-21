---
title: >-
  [论文解读] In-Context Compositional Q-Learning for Offline Reinforcement Learning
description: >-
  [ICLR 2026][强化学习][离线强化学习] ICQL 把离线 RL 的 Q 学习重写成"上下文推断"问题——给定查询状态先从离线数据集检索 top-k 相似 transition，再用线性 Transformer 从这组局部上下文里**就地推断出一个局部 Q 函数**，从而绕过"一个全局 Q 网络硬拟合所有子任务"的困境，在 D4RL 的 Kitchen / MuJoCo / Adroit 上分别提升最高 16.4% / 8.8% / 6.3%。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "离线强化学习"
  - "上下文学习"
  - "Transformer"
  - "局部 Q 函数"
  - "组合性价值估计"
  - "检索增强"
---

# In-Context Compositional Q-Learning for Offline Reinforcement Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ZBbKLvH0w4](https://openreview.net/forum?id=ZBbKLvH0w4)  
**代码**: 待确认  
**领域**: 强化学习 / 离线 RL  
**关键词**: 离线强化学习, 上下文学习, 线性 Transformer, 局部 Q 函数, 组合性价值估计, 检索增强  

## 一句话总结
ICQL 把离线 RL 的 Q 学习重写成"上下文推断"问题——给定查询状态先从离线数据集检索 top-k 相似 transition，再用线性 Transformer 从这组局部上下文里**就地推断出一个局部 Q 函数**，从而绕过"一个全局 Q 网络硬拟合所有子任务"的困境，在 D4RL 的 Kitchen / MuJoCo / Adroit 上分别提升最高 16.4% / 8.8% / 6.3%。

## 研究背景与动机
- **领域现状**：离线 RL 的核心是从固定数据集里学到好策略而不与环境交互，关键障碍是分布漂移导致的 Q 值过估计。主流解法分两派——策略约束（TD3+BC、DT）把策略拉回行为策略附近；价值正则（CQL、IQL）给 OOD 动作的 Q 值加保守惩罚。
- **现有痛点**：无论哪一派，本质上都在训练**一个全局共享的 Q 函数 / 策略**去覆盖整个状态空间。但很多控制任务天然由多个子任务拼成（locomotion 里"加速行走"和"从非正常姿态恢复"是两回事，Kitchen 里开灯/开柜子是不同阶段），一个子任务学到的价值知识不一定能迁移到另一个。
- **核心矛盾**：论文用 t-SNE 可视化指出一个反直觉现象——**几何上相似的状态簇可能对应语义完全不同、长程回报差异很大的行为**。全局价值拟合在数据不足、无法探索时根本抓不住这种局部结构，硬拟合反而引入误差。
- **本文目标**：不再追求一个能精确覆盖全局的价值近似器，而是承认价值函数的"组合性 / 局部性"，学一**族**随状态区域灵活变化的局部价值函数。
- **核心 idea**：**【上下文推断式价值学习】** 把"估计某个状态的 Q 值"重新表述为"给定该状态附近检索到的一小撮 transition，在线推断一个局部线性 Q 函数"。利用线性 Transformer 的 in-context learning 能力，无需任何子任务标签或预定义结构，就能自适应地对每个查询给出局部价值估计。

## 方法详解

### 整体框架
ICQL 的推理-训练流程是一条"检索 → 上下文 → 局部 Q"的链路：给定查询 $(s_{\text{query}}, a_{\text{query}})$，先用特征提取器 $\phi$ 嵌入，从离线数据集 $\mathcal{D}$ 检索 top-k 最相似 transition 拼成局部上下文集 $\Omega^{d_k}_{s_{\text{query}}}$，把它编码成"prompt 矩阵"喂进线性 Transformer，由后者**在前向过程中模拟出一次 in-context TD 学习**、直接吐出局部权重 $w^L_{s_{\text{query}}}$，再线性投影得到局部 Q 值。训练沿用 IQL 的 expectile 回归学价值、advantage-weighted 回归抽策略；推理时策略可独立部署，**无需再做检索**。

```mermaid
flowchart LR
    Q["查询 (s_query, a_query)"] --> PHI["特征提取器 φ"]
    PHI --> RET["从数据集 D 检索<br/>top-k 相似 transition"]
    RET --> CTX["局部上下文 Ω^dk<br/>(prompt 矩阵 Z0)"]
    CTX --> LT["线性 Transformer<br/>= 隐式 in-context TD"]
    LT --> WQ["局部权重 w^L → 局部 Q 函数"]
    WQ --> IQL["IQL 式 critic / policy 更新"]
```

### 关键设计

**1. 局部 Q 函数：用邻域定义代替全局拟合。** 论文不再假设存在一个全局权重向量，而是为每个状态 $s$ 定义一个由其邻域决定的局部线性 Q 函数。具体地，对状态相近、状态转移也相近的 transition 集合 $\Omega^{(d,\bar d)}_s$（满足 $\|s_i-s\|_2^2\le d^2$ 且 $\|s_i'-s_i\|_2^2\le\bar d^2$），存在一个最优局部权重 $w^*_s$ 使得 $\hat Q^{\Omega}_s(\bar s,\bar a) \triangleq w^{*\top}_s \phi(\bar s,\bar a)$ 在该邻域内逼近真实 Q 值（近似误差 $\le\varepsilon^s_{\text{approx}}$）。这把"全局价值近似"拆成了"一族按状态区域各自成立的局部线性近似"，每个局部域有自己的结构。由于邻域半径 $d$ 取决于数据密度、算法无法直接调，论文改用检索集大小 $k$ 来在实践中控制"局部性"。

**2. 检索机制：用相似 transition 拼出局部上下文。** 给定查询状态，默认采用 **State-Similar Retrieval**——取与 $s_{\text{query}}$ 的 $\ell_2$ 距离最小的 k 个 transition：$\Omega_{s_{\text{query}}} \triangleq \{(s_i,a_i,r_i,s_i',a_i')\in\mathcal{D} \mid s_i\in\arg\text{top-}k(-\|s_{\text{query}}-s_i\|_2^2)\}$。论文还讨论了另两种策略：随机检索（保留多样性但局部信息弱）、相似且高回报检索（进一步过滤出高质量 transition）。检索集大小 k 直接等价于隐式控制邻域半径 $d_k$，是连接"理论上的局部域"与"工程上的上下文窗口"的桥梁。

**3. 上下文推断 = 线性 Transformer 隐式跑 TD。** 这是 ICQL 的核心。把检索到的上下文构造成 prompt 矩阵 $Z_0$（每列是一个 transition 的特征 $\phi_i$、折扣后的下一步特征 $\gamma\phi_i'$、奖励 $r_i$，最后一列放查询），喂进 L 层线性 Transformer，每层用形如 $\text{LinAttn}(Z;P,G)=PZM(Z^\top G Z)$ 的线性注意力。论文从理论上证明：在精心构造的权重矩阵 $P_\ell, G_\ell$ 下，**每一层线性注意力恰好等价于对局部权重做一步 SARSA / TD 更新**：

$$w^{l+1}_{s} = w^l_{s} + \alpha\Big(r + \gamma\, w_s^\top\phi(s',a') - w_s^{l\top}\phi(s,a)\Big)\phi(s,a)$$

于是 L 层前向就等价于在检索到的局部数据上跑 L 步 in-context TD 学习，末端取出 $w^L_{s_{\text{query}}}$，得到 $\hat Q(s_{\text{query}}, a_{\text{query}}\mid\Omega^{d_k}) = w^{L\top}_{s_{\text{query}}}\phi(s_{\text{query}}, a_{\text{query}})$。换句话说，Transformer 不是在"记忆"价值，而是在前向里**为每个查询临时训练出一个局部价值估计器**。

**4. IQL 式训练 + 理论近优性保证。** critic 用 expectile 回归拟合局部 Q：$L_{\text{critic}}=\mathbb{E}_{\mathcal{D}}[\rho_\tau(\hat Q(s,a\mid\Omega^{d_k}_s)-y)]$，其中 $y=r+\gamma V(s'\mid\Omega^{d_k}_{s'})$；策略用 advantage-weighted 回归 $L_{\text{policy}}=\mathbb{E}[\exp(\beta(\hat Q-V))\log\pi(a\mid s)]$ 抽取。理论上，在"局部 Q 可线性近似"和"检索集对理想局部域的覆盖率 $\ge\sigma$"两个假设下，论文用 performance difference lemma 证明贪婪策略的性能差被界住：$J(\pi^*)-J(\pi)\le \frac{2}{1-\gamma}\mathbb{E}[\varepsilon^s_{\text{approx}}(1+B_\phi)+CB_\phi\sqrt{(d+\log(1/\delta))/(\sigma|\Omega^{d_k}_s|)}]$，把误差清晰地拆成"近似误差"和"权重估计误差"两项，后者随上下文覆盖量增大而衰减。

## 实验关键数据

### 主实验表格（D4RL，5 个随机种子均值）

| 任务族 | BC | DT | TD3+BC | CQL | IQL | **ICQL** | Gain |
|---|---|---|---|---|---|---|---|
| MuJoCo (9 任务均值) | 51.9 | 58.8 | 62.9 | 74.0 | 72.4 | **80.6** | **+8.8%** |
| Adroit (6 任务均值) | 17.5 | 27.9 | 24.2 | 15.5 | 33.2 | **35.3** | **+6.3%** |
| Kitchen (3 任务均值) | 51.5 | 55.8 | 52.6 | 48.2 | 52.8 | **66.8** | **+16.4%** |

代表性单项：Walker2d-Medium-Replay 81.9（次优 CQL 77.2）、HalfCheetah-Medium-Expert 89.1（次优 IQL 83.4）、Door-Human 17.1（IQL 9.8，+73%）、Kitchen-Complete 79.3（BC 65.0，+22%）。

### 消融实验表格（检索策略，节选）

| Dataset | Random | State-Similar | Similar+HighReward |
|---|---|---|---|
| Walker2d-Medium | 78.1 | 80.3 | **83.9** |
| Walker2d-Medium-Replay | 67.5 | **81.9** | 75.1 |
| Hopper-Medium-Replay | 81.0 | **96.4** | 90.8 |
| Pen-Human | 75.1 | **85.6** | 84.8 |
| Kitchen-Complete | 70.0 | **79.3** | 71.3 |

其余消融：① 层数（=in-context TD 步数）从 4→20，MuJoCo 多数任务分数随层数增加而上升，印证更多层 = 更充分的 in-context 价值学习；② 上下文长度从 {10,20,30,40} 中以 **20 最优**，太长会让查询状态与上下文距离变大、破坏"局部性"并引入噪声。

### 关键发现
- **更准的 Q 估计是性能来源**：在 Walker2d-Medium 上，ICQL 的 Q 估计分布与在线 SAC 的相似度达 0.69，而 IQL 仅 0.29——说明局部价值建模在噪声数据上给出了更接近"真值"的 Q。
- **组合性任务收益最大**：Kitchen 这种多阶段长程任务提升最显著（+16.4%），直接支撑"价值函数本质是组合性的"这一动机。
- **失败案例诚实呈现**：Hammer-Human 上 ICQL 反而落后部分基线，论文归因于该数据集规模小、查询状态与检索状态距离大，使 in-context learning 更困难。

## 亮点与洞察
- **范式重述**：把"训练一个 Q 网络"换成"为每个查询在线推断一个局部 Q 函数"，是 in-context learning 思想在价值估计（而非 DT 那种动作生成）上的一次干净落地——论文明确指出已有 in-context RL 工作都在生成动作/策略，没人专门攻价值估计。
- **理论与机制对齐漂亮**：线性 Transformer 的每层注意力 = 一步 TD 更新，这个等价关系让"为什么 Transformer 能做价值推断"有了可证明的机制解释，而非黑盒。
- **局部性即先验**：用检索半径 k 显式编码"价值的局部结构"，把一个难调的连续超参 $d$ 转成可操作的离散 k，工程友好且与理论覆盖率假设直接挂钩。

## 局限与展望
- **依赖检索质量**：理论近优性的核心假设是检索集对理想局部域的覆盖率 $\ge\sigma$，数据稀疏/查询离群时（如 Hammer-Human）覆盖不足直接掉点，方法对数据集密度敏感。
- **线性局部近似的天花板**：局部 Q 假设可线性近似，遇到局部域内本身高度非线性的价值结构时近似误差 $\varepsilon_{\text{approx}}$ 不可忽略。
- **额外计算开销**：每次价值估计都要检索 + 跑 L 层 Transformer 前向，虽论文称开销"适中"，但相比单次网络前向仍更重（推理时策略可独立部署缓解了这点）。
- **可拓展方向**：自适应选择 k / 上下文长度、更强的特征提取器、把"相似+高回报"检索做成可学习的检索器，都可能进一步提升覆盖率与价值精度。

## 相关工作与启发
- **离线 RL**：CQL（保守 Q 惩罚）、IQL（expectile + AWR，本文训练框架基底）、TD3+BC、ReBRAC、FQL 等，共同点是全局价值/策略建模，ICQL 用局部估计形成对照。
- **RL 中的 in-context learning**：Decision Transformer、Gato、Algorithm Distillation、DPT、PreDeToR 等，多聚焦轨迹建模/动作生成；ICQL 强调自己是首个用线性注意力做**组合性价值估计**的工作。
- **理论根基**：线性 Transformer 实现 in-context（TD）学习的理论（Von Oswald 2023、Wang 2025b）+ 非参回归的覆盖率分析，被本文借来证明近优性。
- **启发**：这条"检索 → in-context 推断局部模型"的思路不止于 Q 学习，对任何"全局模型难以覆盖、但局部结构清晰"的预测任务（如局部动力学建模、奖励建模）都有迁移潜力。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把离线 RL 价值学习重述为上下文推断、并证明线性注意力每层 = 一步 TD，视角和机制解释都新颖。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 D4RL 三大任务族、5 种基线、检索策略/层数/上下文长度多维消融，并诚实报告失败案例；缺更大规模/像素任务验证。
- **写作质量**: ⭐⭐⭐⭐ 动机—理论—方法—实验闭环清晰，t-SNE 可视化有力支撑核心假设。
- **价值**: ⭐⭐⭐⭐ 在组合性长程任务上提升显著，为"检索增强 + in-context 价值估计"开了一条有理论保证的新路线。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] LongRLVR: Long-Context Reinforcement Learning Requires Verifiable Context Rewards](longrlvr_long-context_reinforcement_learning_requires_verifiable_context_rewards.md)
- [\[ICLR 2026\] Scalable In-Context Q-Learning](scalable_in-context_q-learning.md)
- [\[ICLR 2026\] 3D-aware Disentangled Representation for Compositional Reinforcement Learning](3d-aware_disentangled_representation_for_compositional_reinforcement_learning.md)
- [\[ICLR 2026\] Offline Reinforcement Learning with Adaptive Feature Fusion](offline_reinforcement_learning_with_adaptive_feature_fusion.md)
- [\[ICLR 2026\] Context and Diversity Matter: The Emergence of In-Context Learning in World Models](context_and_diversity_matter_the_emergence_of_in-context_learning_in_world_model.md)

</div>

<!-- RELATED:END -->
