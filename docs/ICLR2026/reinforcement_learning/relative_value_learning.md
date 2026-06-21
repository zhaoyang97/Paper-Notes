---
title: >-
  [论文解读] Relative Value Learning
description: >-
  [ICLR 2026][强化学习][相对价值] 针对"控制只关心价值之差、绝对价值尺度是冗余自由度"这一观察，本文提出 Relative Value Learning（RV），让 critic 直接学一个反对称函数 $\Delta_\theta(s_i,s_j)=V^\pi(s_i)-V^\pi(s_j)$，配套给出成对 Bellman 算子（证明是 $\gamma$-压缩、唯一不动点等于真实价值差）、良定义的 1-step / n-step / λ-return 目标，以及从成对差重建出的无偏优势估计 R-GAE；接到 PPO 上在 49 个 Atari 游戏上与标准 PPO 持平甚至更好。
tags:
  - "ICLR 2026"
  - "强化学习"
  - "相对价值"
  - "反对称函数"
  - "成对 Bellman 算子"
  - "R-GAE"
  - "PPO"
---

# Relative Value Learning

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=ulTRUwrzt9](https://openreview.net/forum?id=ulTRUwrzt9)  
**代码**: [https://github.com/Hauf3n/relative-value-learning](https://github.com/Hauf3n/relative-value-learning)  
**领域**: 强化学习 / 价值函数 / 策略梯度  
**关键词**: 相对价值、反对称函数、成对 Bellman 算子、R-GAE、PPO

## 一句话总结
针对"控制只关心价值之差、绝对价值尺度是冗余自由度"这一观察，本文提出 Relative Value Learning（RV），让 critic 直接学一个反对称函数 $\Delta_\theta(s_i,s_j)=V^\pi(s_i)-V^\pi(s_j)$，配套给出成对 Bellman 算子（证明是 $\gamma$-压缩、唯一不动点等于真实价值差）、良定义的 1-step / n-step / λ-return 目标，以及从成对差重建出的无偏优势估计 R-GAE；接到 PPO 上在 49 个 Atari 游戏上与标准 PPO 持平甚至更好。

## 研究背景与动机
**领域现状**：主流 value-based RL（TD(λ)、DQN、Rainbow、A2C/PPO 等）都让 critic 去逼近**绝对**状态价值 $V^\pi(s)$ 或动作价值 $Q^\pi(s,a)$，即"单独评估某个状态/动作有多好"，再把价值之差当作派生量算出来。

**现有痛点**：在控制里，动作是靠**比较**选出来的——贪心选 $\max_a Q^\pi(s,a)$、策略梯度用优势 $A^\pi(s,a)$，这些都只依赖价值之差。给 $V^\pi$ 整体加一个常数 $c$（配合相应的奖励整形）不会改变任何优势、不会改变贪心选择。也就是说，绝对尺度在行为上**没有意义**，是一个"未被钉死的偏移量"（gauge freedom，规范自由度）。

**核心矛盾**：绝对 critic 偏偏要去预测这个行为上无意义的标量。这个多余的自由度会带来三类麻烦：① 奖励整形 / baseline 变化时容易发生漂移；② 在只有"比较"或隐式反馈的场景（偏好式 RL、human-in-the-loop RL）里，绝对尺度本身就是模糊的、问题甚至 ill-posed，而成对关系却始终良定义；③ 函数类的不变性与决策问题的不变性不匹配。

**本文目标**：把价值之差直接当成首要学习对象，让 critic 的函数类从设计上就吻合"只有差有意义"这一不变性，从而消除规范自由度。

**切入角度**：学一个反对称函数 $\Delta_\theta:S\times S\to\mathbb{R}$，强制 $\Delta_\theta(s_i,s_j)=-\Delta_\theta(s_j,s_i)$（于是自动有 $\Delta_\theta(s,s)=0$）去逼近 $V^\pi(s_i)-V^\pi(s_j)$。这样规范自由度被"构造性地"消掉，而且优势可以从成对差里重建出来、无需知道任何状态的绝对价值。

**核心 idea**：用一个反对称的成对价值差网络替代绝对 critic，并为它配一套自洽的 Bellman 理论（压缩、目标、优势重建），让"相对价值"成为有干净解析基础的一等公民。

## 方法详解

### 整体框架
RV 把传统 actor-critic 里的"绝对 critic"换成一个**成对价值差 critic** $\Delta_\theta(s_i,s_j)$，整套方法围绕"如何良定义地学这个差、再用它喂给 PPO"展开，分四块：（1）在反对称函数空间上定义**成对 Bellman 算子** $T_\pi$，证明它是 $\gamma$-压缩且唯一不动点正好等于真实价值差，给出理论合法性；（2）把单步/多步/λ-return 的 bootstrapping 目标全部**改写成只含可观测奖励与非终止成对差**的形式，解决终止状态导致的 ill-posed 问题；（3）从成对差出发，沿一条轨迹做 telescoping 重建出相对价值，进而得到 **R-GAE** 优势估计，并证明它与标准 GAE 只差一个轨迹常数、对策略梯度无偏；（4）在一个 batch 含多条轨迹时，用 **trajectory ranking** 给每条轨迹估一个偏移量，把 R-GAE 的额外方差压下去。最后 critic 用一个共享 CNN 编码器 + siamese 差分头实现，损失就是把 PPO 的 GAE 换成 R-GAE。

### 关键设计

**1. 成对 Bellman 算子与压缩性：给"学价值差"一个干净的理论地基**

要直接学 $\Delta^\pi(s_i,s_j)=V^\pi(s_i)-V^\pi(s_j)$，首先得证明这个目标本身满足一条自洽的递归方程、并且能被迭代逼近，否则"学差"只是个工程 trick。本文把两个状态各自的 Bellman 方程相减，得到成对 Bellman 恒等式：

$$\Delta^\pi(s_i,s_j)=r^\pi(s_i)-r^\pi(s_j)+\gamma\,\mathbb{E}_{s_i'\sim P^\pi(\cdot|s_i),\,s_j'\sim P^\pi(\cdot|s_j)}\big[\Delta^\pi(s_i',s_j')\big]$$

其中两个后继 $s_i',s_j'$ 独立采样。这个式子只依赖可观测的单步奖励差和后继处的成对差，且对 $V^\pi$ 的任意整体平移不变。在有界反对称函数的 Banach 空间 $\mathcal{F}$（配 $\ell_\infty$ 范数）上定义算子 $(T_\pi\Delta)(s_i,s_j):=\Delta r^\pi(s_i,s_j)+\gamma(\hat P^\pi\Delta)(s_i,s_j)$，作者证明（Theorem 3.1）：对任意 $\Delta_1,\Delta_2$ 有 $\|T_\pi\Delta_1-T_\pi\Delta_2\|_\infty\le\gamma\|\Delta_1-\Delta_2\|_\infty$。证明的关键一步是即时奖励差 $\Delta r^\pi$ 在相减时直接抵消，只剩 $\gamma$ 倍的差。由 Banach 不动点定理，$T_\pi$ 有唯一不动点，且这个不动点**恰好等于真实价值差** $V^\pi(s_i)-V^\pi(s_j)$。这就保证了"学差"不是近似游戏，而是和标准价值迭代一样有收敛保证。

**2. 良定义的成对价值目标：处理终止状态这个隐藏陷阱**

直接套用成对 Bellman 算子去构造 bootstrapping 目标会踩坑：当某个后继是终止状态（done flag $d_i=1$）时，朴素地写 $\Delta(s_{i+1},s_{j+1})=0-V(s_{j+1})=-V(s_{j+1})$ 需要单点的**绝对价值**，而 RV 的函数类里根本拿不到绝对价值，目标因此 ill-posed。本文把所有 bootstrapping 目标**重排成只含可观测奖励与非终止成对差**的形式。1-step 目标写作 $y^{(1)}_{ij}=(r_i-r_j)+\gamma\delta_{ij}$，其中 bootstrap 项按两条轨迹的终止标志分四种情况：

$$\delta_{ij}=\begin{cases}\Delta_\theta(s_{i+1},s_{j+1}),&d_i=0,d_j=0\\\Delta_\theta(s_{i+1},s_j)+r_j,&d_i=0,d_j=1\\\Delta_\theta(s_i,s_{j+1})-r_i,&d_i=1,d_j=0\\\Delta_\theta(s_i,s_j)+r_j-r_i,&d_i=1,d_j=1\end{cases}$$

当两条都终止时，默认取 $\delta_{ij}=0$（两个吸收态都是零价值）以降低 episode 末尾的方差。n-step 目标 $y^{(n)}_{ij}=\sum_{k=0}^{n-1}\gamma^k(r_{i+k}-r_{j+k})+\gamma^n\Delta_\theta(s_{i+n},s_{j+n})$ 假设窗口内不终止（否则取较短轨迹长度），λ-return 则按 $y^{(\lambda)}_{ij}=(1-\lambda)\sum_{n\ge1}\lambda^{n-1}y^{(n)}_{ij}$ 在 n 上指数加权、并在首个终止处截断。这套改写让目标对算子 $T_\pi$ 保持兼容、对终止鲁棒。

**3. R-GAE：从成对差重建出无偏的优势估计**

RV 要给 PPO 当 critic，就必须能产出优势。但 RV 不知道任何绝对价值，怎么算 GAE？本文沿一条 rollout $(s_0,\dots,s_T)$ 做 telescoping 构造**相对价值**：令 $\tilde V_\theta(s_0):=0$，$\tilde V_\theta(s_t):=\sum_{k=0}^{t-1}\Delta_\theta(s_{k+1},s_k)$（也可直接用 $\Delta_\theta(s_t,s_0)$ 大步算）。若 $\Delta_\theta=\Delta^\pi$，则 $\tilde V_\theta(s_t)=V^\pi(s_t)-V^\pi(s_0)$，即相当于把整条轨迹锚到起点为零。再仿照 GAE 定义相对 TD 残差 $\tilde\delta_t=r_t+\gamma\tilde V_\theta(s_{t+1})-\tilde V_\theta(s_t)$ 和 $\tilde A_t=\sum_{l=0}^{T-t}(\gamma\lambda)^l\tilde\delta_{t+l}$。关键的理论结果（Lemma 3.2）是：$\tilde A_t=A_t+B_t$，其中 $A_t$ 是用真实 $V^\pi$ 算出的标准 GAE，$B_t=(1-\gamma)C\sum_{l=0}^{T-t}(\gamma\lambda)^l$，$C:=V^\pi(s_0)$ 是**轨迹常数**。进一步（Corollary 3.3），把策略梯度里的优势换成 $\tilde A_t$ 后梯度**不变**：因为 $B_t$ 在给定 $s_t$ 时与动作 $a_t$ 无关，乘上 score function $\nabla_\phi\log\pi_\phi(a_t|s_t)$ 取期望为零，所以 $\nabla_\phi\tilde J(\phi)=\nabla_\phi J(\phi)$。这意味着 RV 当 critic 时策略梯度是**无偏**的——锚到零只是一次"逐轨迹的规范固定"，类比 average-reward MDP 里的相对价值归一化，区别只在于这里是 per-trajectory 而非全局。

**4. Trajectory Ranking 初始化：把多轨迹 batch 的额外方差压下去**

无偏不等于零代价：$\tilde A_t=A_t+B_t$ 里的 $B_t$ 正比于未知轨迹常数 $C=V^\pi(s_0)$，$|C|$ 大时会放大 $\tilde A_t$ 的幅度、抬高策略梯度方差，让 credit assignment 更难。当一个训练 batch 里混了多条来自不同 episode 的轨迹时，给每条都锚到零是不对的——它们之间还有相对高低。本文用一个数据相关的初始化来让 $\mathbb{E}_t[B_t]\approx0$：从 batch 里取出所有 start states（每条 rollout 开头以及每个新 sub-episode 开头），用学到的（带噪）$\Delta_\theta$ 构造 $N\times N$ 成对差矩阵 $\Delta_{ij}=\Delta_\theta(s^{(i)}_{\text{start}},s^{(j)}_{\text{start}})$，对每行求均值得到偏移估计 $O(s^{(n)}_{\text{start}})=\frac1N\sum_j\Delta_{nj}$，再减去 batch 最小值得到非负且可排序的 $\hat V_\theta(s^{(n)}_{\text{start}})=O(s^{(n)}_{\text{start}})-\min_\ell O(s^{(\ell)}_{\text{start}})$。最后对 rollout 里任意状态 $s$，用同一 rollout 中最近的 start state $s_{\text{start}}$ 给它配偏移：$\bar V_\theta(s)=\hat V_\theta(s_{\text{start}})+\Delta_\theta(s,s_{\text{start}})$，并用这套带偏移的相对价值跑 R-GAE。行均值 + 减最小值的设计让偏移对预测噪声鲁棒、且在 batch 内"可识别到一个常数"——对排序而言已经足够。

### 损失函数 / 训练策略
critic 用共享 CNN 编码器（与标准 PPO 架构完全一致）把两个状态各自编码 $f_{\text{enc}}(s)\in\mathbb{R}^d$，再投影它们 embedding 的差：$\Delta_\theta(s_i,s_j)=\Phi(f_{\text{enc}}(s_i)-f_{\text{enc}}(s_j))$。为保证反对称，$\Phi$ 取**无偏置**的单个学习向量 $w\in\mathbb{R}^d$（这样 $\Delta_\theta(s_i,s_j)=-\Delta_\theta(s_j,s_i)$、$\Delta_\theta(s_i,s_i)=0$ 天然成立），不用额外 target encoder、不用 stop-gradient。总损失把 PPO 的 GAE 换成 R-GAE：

$$L(\theta)=-L_{\text{policy}}(\theta)+c_v L_{\text{critic}}(\theta)+c_e L_{\text{ent}}(\theta)$$

其中策略项是用 $\tilde A_t$ 的 PPO clip 目标 $L_{\text{policy}}=\mathbb{E}_t[\min(r_t(\theta)\tilde A_t,\,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\tilde A_t)]$，critic 项是回归 n-step 目标的 MSE $L_{\text{critic}}=\frac12\mathbb{E}_{(i,j)}(\Delta_\theta(s_i,s_j)-y^{(n)}_{ij})^2$，外加熵奖励。所有 49 个游戏共用一套超参。

## 实验关键数据

### 主实验
在 ALE Atari 的 49 个游戏上把 PPO+RV 当成 on-policy 策略梯度的 drop-in critic，与标准 PPO、DAE 对比，训练 40M 帧（10M 环境步），每个游戏 10 个随机种子，报告最后 100 个 episode 的平均得分。

| 游戏（节选） | PPO | DAE | PPO + RV (ours) |
|--------------|-----|-----|-----------------|
| BattleZone | 17366.7 | 16302.0 | **21780.0** |
| RoadRunner | 25076.0 | 16146.3 | **43346.3** |
| Robotank | 5.5 | 6.9 | **19.5** |
| TimePilot | 4342.0 | 7252.7 | **10212.7** |
| VideoPinball | 37389.0 | 23958.6 | **138564.8** |
| Enduro | 758.3 | 0.0 | **1080.2** |
| Gravitar | 737.2 | 443.5 | **1441.0** |
| Centipede | **4386.4** | 3915.8 | 1226.1 |
| Pong | **20.7** | 20.7 | 16.8 |
| Zaxxon | 5008.7 | 5612.2 | 845.8 |

聚合指标（human-normalized，95% 分层 bootstrap 置信区间）上，PPO+RV 在 Median、IQM、Mean 三项都高于 PPO 与 DAE，Optimality Gap 更低——即整体上 RV 这个相对 critic 是绝对 critic 的一个有效替代，而非仅仅持平。

### 消融 / 关键设置

| 设置 | 做法 | 结果 / 说明 |
|------|------|-------------|
| R-GAE vs $\Delta_\gamma$ 变体 | 用非反对称的 $\Delta_\gamma(s',s)=\gamma V(s')-V(s)$ 精确匹配 TD 残差（理论上可与 GAE 逐点相等） | 实测效果**更差**，故仍采用 R-GAE |
| 双终止时的 $\delta_{ij}$ | 取 $\delta_{ij}=0$ 而非公式第四行 | 降低 episode 末尾方差，默认采用 |
| 投影头 $\Phi$ | 单向量 $w$ vs 反对称非线性 MLP（tanh、无 bias） | 非线性头**未观察到额外提升** |
| Trajectory Ranking | 给多轨迹 batch 估偏移使 $\mathbb{E}_t[B_t]\approx0$ | 压低 R-GAE 因轨迹常数 $B_t$ 带来的方差 |

### 关键发现
- RV 在像 VideoPinball、RoadRunner、Robotank、Gravitar 这类游戏上对 PPO 有大幅领先，但在 Centipede、Zaxxon、Pong、Tennis 等少数游戏上明显落后，说明相对 critic 并非全面占优、而是整体聚合更好。
- 理论上能逐点等于 GAE 的 $\Delta_\gamma$ 变体反而在实践中更差，作者最终坚持反对称的 R-GAE，并靠 trajectory ranking 控制其额外方差——可见"无偏 + 控方差"这条工程链路是落地关键。
- 全程不用 target network、不用 stop-gradient，critic 仅靠 siamese 差分头就稳定训练，说明反对称约束本身带来了良好的训练性质。
- 计算成本：每次 40M 帧约 65 分钟（单 A100 + 12 CPU），490 次运行共 530 GPU-小时（22.1 A100-天）。

## 亮点与洞察
- **把"规范自由度"这件被默认无害的事变成方法论起点**：大家都知道给 $V$ 加常数不改变行为，但只有本文把它当成应当从函数类里直接消掉的冗余，用反对称网络结构性地钉死，思路很干净。
- **成对 Bellman 算子的压缩证明很优雅**：即时奖励差在相减时抵消，一行就得到 $\gamma$-压缩，且不动点恰好是真实价值差——理论自洽度高。
- **R-GAE 的无偏性论证可迁移**："重建出的优势只差一个与动作无关的常数，故策略梯度无偏"这一论证，本质上是 baseline 不影响策略梯度的推广，可以借鉴到任何"只学相对量再重建优势"的设定（如偏好式 RL）。
- **终止状态的目标改写**是容易被忽视但必须处理的细节，本文把它显式列成四种情况，给后来者省了踩坑。

## 局限与展望
- **只在 Atari + PPO 上验证**：RV 作为 drop-in critic 仅在 ALE 离散动作、on-policy 的设定下测过，对连续控制、off-policy（如 Q-learning 系）是否同样有效未知。
- **成对结构带来额外开销与方差**：需要构造状态对、跑成对差，trajectory ranking 还要在 batch 内算 $N\times N$ 矩阵；虽然单向量头很轻，但相对价值的方差控制仍依赖额外机制。
- **在部分游戏明显掉点**（Centipede、Pong、Zaxxon、Tennis 等），论文未深入分析这些失败模式的成因。
- **动机里强调的偏好式 / human-in-the-loop RL 场景并未实验验证**——这正是相对价值最该发光的地方，留作未来工作会更有说服力。
- 反对称非线性头、$\Delta_\gamma$ 变体目前都没带来增益，相对 critic 的表达能力上限还没探到。

## 相关工作与启发
- **vs 绝对 critic（DQN / Rainbow / 分布式 critic）**: 它们都在绝对价值空间里预测一个任意尺度的标量；分布式方法只是重构了 target、仍在绝对空间。RV 在**模型层面**就去掉了偏移自由度，让函数类吻合决策的不变性。
- **vs Direct Advantage Estimation (DAE)**: DAE 直接学优势 $A^\pi(s,a)$ 绕开价值学习；RV 则学状态对之间的价值差 $\Delta(s_i,s_j)$ 并配一个显式的成对 Bellman 算子，本文实验里 PPO+RV 整体强于 DAE。
- **vs Dueling 网络**: Dueling 把 $Q=V+A$ 分解以提升样本效率，但仍在绝对空间；RV 用 siamese 差分头从设计上强制反对称与零自差。
- **vs 偏好式 / human-in-the-loop RL 与 inverse RL**: 这些工作里也出现过成对目标，但没把它做成一个 Bellman-自洽的价值 critic；本文填了这个空白。average-reward MDP 里的相对价值迭代（只到加性常数）则是其规范固定思想的近亲。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把"只有价值差有意义"做成有完整 Bellman 理论的反对称 critic，视角新且自洽
- 实验充分度: ⭐⭐⭐⭐ 49 游戏 × 10 种子很扎实，但只覆盖 Atari+PPO，且动机强调的偏好式 RL 未验证
- 写作质量: ⭐⭐⭐⭐⭐ 动机—理论—算法—实验环环相扣，定理与算法推导清晰
- 价值: ⭐⭐⭐⭐ 提供了一个干净的相对价值框架，对偏好式/隐式反馈 RL 有迁移潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Relative Entropy Pathwise Policy Optimization](relative_entropy_pathwise_policy_optimization.md)
- [\[ICLR 2026\] Value Flows](value_flows.md)
- [\[ICLR 2026\] Reinforcement Learning via Value Gradient Flow](reinforcement_learning_via_value_gradient_flow.md)
- [\[ICLR 2026\] Inter-Agent Relative Representations for Multi-Agent Option Discovery](inter-agent_relative_representations_for_multi-agent_option_discovery.md)
- [\[ICLR 2026\] Transitive RL: Value Learning via Divide and Conquer](transitive_rl_value_learning_via_divide_and_conquer.md)

</div>

<!-- RELATED:END -->
