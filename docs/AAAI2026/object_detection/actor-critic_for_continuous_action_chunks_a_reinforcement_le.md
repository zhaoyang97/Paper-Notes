# Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward

**会议**: AAAI 2026  
**arXiv**: [2508.11143](https://arxiv.org/abs/2508.11143)  
**代码**: [https://github.com/flyfaerss/ac3](https://github.com/flyfaerss/ac3) (有)  
**领域**: 机器人操作 / 强化学习  
**关键词**: action chunking, actor-critic, sparse reward, long-horizon manipulation, self-supervised reward shaping  

## 一句话总结
AC3 提出了一个直接学习连续动作序列（action chunk）的 actor-critic 框架，通过"仅从成功轨迹更新 actor"的非对称更新规则和基于自监督锚点的内在奖励来稳定稀疏奖励下的长时域机器人操作学习，在 BiGym 和 RLBench 的 25 个任务上取得优于现有方法的成功率。

## 背景与动机
长时域机器人操作任务（如移动盘子、翻转三明治）需要执行多个子任务的连续动作序列，现有 RL 方法在这类稀疏奖励场景下面临两大困境：（1）探索空间随动作序列长度指数增长，智能体难以发现有效策略；（2）只有任务完成时才有正奖励，中间步骤缺乏有效引导信号。

**Action Chunking（动作分块）** 范式在模仿学习中已非常成功（如 ACT、Diffusion Policy），但它们的上限受制于专家数据质量，且遇到分布偏移（未见过的状态）就容易失败。将 action chunking 引入 RL 理论上能通过在线探索突破这一上限，但直接学习连续动作块面临 Q 值爆炸和训练不稳定的严重问题。现有尝试要么离散化动作空间牺牲精度（CQN-AS），要么依赖大规模离线数据和复杂蒸馏管线（Q-Chunking），均不够直接高效。

## 核心问题
如何在**稀疏奖励**和**少量专家演示**的条件下，让 RL 直接学习**高维连续动作序列**？核心挑战是：(1) 动作空间维度随 chunk 长度线性增长，导致 Q 值估计不稳定甚至爆炸；(2) actor 在大量失败状态上获得的梯度信号是有害的，会导致策略崩溃。

## 方法详解

### 整体框架
AC3 建立在 DDPG 风格的 off-policy actor-critic 框架之上。输入是多视角 RGB 图像（84×84）加上本体感觉状态，输出是一个 $C \times d_a$ 维的连续动作块（默认 $C=16$）。整个流程分三层设计：
1. **Actor 网络** 直接预测连续动作块
2. **Critic 网络** 用 chunk 内 $n$-step return 来估计 Q 值
3. **自监督奖励塑形模块** 在锚点处提供内在奖励信号

### 关键设计

1. **非对称更新规则（Asymmetric Actor Update）**：这是最核心的设计。Actor 只从成功轨迹（包括专家演示和在线成功 rollout）中学习，而非全部经验。原因在于：高维稀疏奖励下 Critic 对绝大多数状态空间的 Q 值估计极不准确，如果 Actor 在这些区域做策略梯度更新，会被错误梯度带偏导致策略崩溃。将训练限制在"可信区域"（value function 最可靠的区域），等价于只在成功的数据流形上优化策略。Actor 的总损失为 BC 损失和 Q 损失的加权组合：$\mathcal{L}_\theta = \lambda_{BC} \mathcal{L}_\theta^{BC} + \lambda_Q \mathcal{L}_\theta^Q$，其中 $\lambda_{BC}=1.0, \lambda_Q=0.1$。

2. **Chunk 内 $n$-step Return（Intra-Chunk $n$-step TD）**：Critic 不直接评估整个 chunk 的价值（会引入过大方差），而是用 $n$-step 回报（$n < C$，默认 $n=4, C=16$）。这解决了一个关键理论问题——**无约束动作子空间问题（Unconstrained Action Subspace）**：当 $n < C$ 时，动作块的尾部 $\{a_{t+n}, ..., a_{t+C-1}\}$ 不影响 TD 目标，Actor 可以任意操纵这些维度来"欺骗"Critic 最大化 Q 值，形成 Actor-Critic 恶性循环导致 Q 值爆炸。适中的 $n$ 值（4 或 8）利用神经网络的平滑性对尾部动作施加隐式约束，有效抑制爆炸。同时采用 TD3 的 clipped double-Q 来压制过估计。

3. **自监督奖励塑形（Self-Supervised Reward Shaping）**：预训练一个 Goal Network $G_\omega$，使用对比学习（triplet loss）在专家演示数据上学习状态表征——时间上相近的状态拉近，时间上远或不同轨迹的状态推远。在线训练时，每隔 $K=C=16$ 步设置一个**锚点（Anchor Point）**，计算当前状态到示范状态集的最近嵌入距离，若小于阈值 $m=0.5$ 则给予固定内在奖励 $a=0.1$（远小于任务成功奖励 1）。这种半稠密的奖励设计与 chunk 决策节奏自然对齐，避免了过度奖励导致的 Q 值爆炸。

### 损失函数 / 训练策略
- **Critic 损失**：双 Q 网络的 MSE 损失，目标使用 $n$-step return + clipped double-Q
- **Actor 损失**：BC 损失（模仿成功轨迹）+ Q 损失（策略梯度），仅在成功轨迹上计算
- **Goal Network**：triplet loss 预训练，5 seeds × 5 epochs = 25 epochs，预训练仅需不到1小时
- **网络架构**：简单的 MLP+GRU，Actor 和 Critic 共用状态编码器结构，隐藏维度 512
- 使用 ACT 的时序集成（temporal ensemble）来平滑动作执行

## 实验关键数据

| 基准 | AC3 vs BC baseline | AC3 vs CQN-AS | AC3 vs DrQ-v2 |
|------|-------------------|---------------|---------------|
| BiGym (15任务, 10 demos) | 大多数任务显著超越 | 大多数任务更优 | 大幅领先 |
| RLBench (10任务, 100 demos) | 多数任务超越 | 性能相当（架构更简单） | 大幅领先 |

| 方法 | 可训练参数 | 推理速度 |
|------|-----------|---------|
| Chunk-wise BC | 6.56M (Actor-only) | 2.9ms |
| CQN-AS | 28.58M (Q-Network) | 9.5ms |
| AC3 | 14.44M (Actor-Critic) | 2.9ms (仅用Actor推理) |

### 消融实验要点
- **Chunk 长度 $C$**：复杂双臂任务（如 move plate）需要长 chunk（$C=16$），$C=4$ 几乎完全失效；简单任务适当 chunk 即可
- **$n$-step 设置**：$n=1$ 在高维任务中导致 Q 值爆炸；$n=C=16$ 方差过大也不行；$n=4, 8$ 是最佳折中
- **非对称更新**：让 Actor 从全部经验（含失败）学习会导致策略崩溃或严重退化
- **奖励塑形**：$r_{int}=0.1$ 效果最好；线性增长或等于任务奖励的设计都导致 Q 值爆炸
- **与 QC-FQL 对比**：当离线数据少时，flow matching 的 BC 预训练效果差，且用全部经验训练在双臂任务中导致策略完全失败

## 亮点
- **非对称更新规则**是一个简洁而有效的 insight：在稀疏奖励下 Critic 的 Q 值在大部分状态空间不可靠，所以 Actor 应该只在 Q 值可信的区域（成功轨迹）做策略优化。这个想法可以迁移到其他稀疏奖励 RL 问题
- **Unconstrained Action Subspace**的理论分析非常到位，清晰解释了为什么 $n < C$ 时会出现 Q 值爆炸，以及为什么适中的 $n$ 能通过神经网络平滑性缓解这个问题
- 架构极简（MLP+GRU），不依赖 Transformer 或扩散模型，推理 2.9ms，3倍快于 CQN-AS——非常适合实时控制部署
- 用**少量**演示（BiGym 仅 10 条）就能通过在线 RL 超越纯模仿学习的上限

## 局限性 / 可改进方向
- **未在真实机器人上验证**：全部实验在仿真 benchmark 上完成，sim-to-real gap 未知
- **Chunk 长度固定**：$C$ 是手工设定的超参数，不同复杂度的子任务可能需要不同的 chunk 长度，自适应 chunk 长度是一个自然的扩展方向
- **Goal Network 依赖刚性阈值**：锚点奖励的阈值 $m$ 和奖励值 $a$ 是手工调的，鲁棒性有待验证
- **假设成功轨迹可获得**：非对称更新依赖 replay buffer 中存在成功轨迹，在极难任务中初始可能无成功经历（仅靠少量专家 demo bootstrap）
- **单任务训练**：每个任务独立训练一个策略，无跨任务泛化能力
- **没有与 Diffusion Policy + RL 的方法（如 DPPO）对比**

## 与相关工作的对比
- **vs CQN-AS**：CQN-AS 将动作块离散化后做 Q-learning，精度和灵活性受限；AC3 直接生成连续 chunk，更精确，且推理快 3x+
- **vs Q-Chunking (QC-FQL)**：QC 用 flow matching + 大规模离线数据 + $n=C$ 的全 chunk 设计；AC3 用简单 MLP+GRU + 少量 demo + $n<C$ 的折中设计，在少数据场景下更实用
- **vs DrQ-v2 (单步 RL)**：单步 RL 在长时域稀疏奖励任务上几乎完全失败，充分说明了 action chunking 的必要性
- **vs ACT/Diffusion Policy (纯 IL)**：IL 方法的性能上限受制于演示数据，AC3 通过在线 RL 突破这一上限

## 启发与关联
- **非对称更新思想**在其他稀疏奖励/高维连续控制问题中可能有普遍价值，比如自动驾驶等长时序决策任务
- **Chunk 内 $n$-step 的 bias-variance-explosion 三角权衡分析**提供了 RL + action chunking 领域的理论框架，对后续工作有参考价值
- 对比学习做 reward shaping 的思路可以迁移到其他需要中间信号引导的稀疏奖励任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 非对称更新和 chunk 内 n-step 的理论分析有新意，但整体框架是 DDPG 的自然扩展
- 实验充分度: ⭐⭐⭐⭐⭐ 25 个任务、多项消融、与多种 baseline 对比，分析非常详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，理论分析透彻，附录的无约束动作子空间分析尤为出色
- 价值: ⭐⭐⭐⭐ 解决了 RL + continuous action chunking 中的关键稳定性问题，方法简洁实用
