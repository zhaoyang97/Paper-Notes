# DA-AC: Distributions as Actions — A Unified RL Framework for Diverse Action Spaces

**会议**: ICLR 2026  
**arXiv**: [2506.16608](https://arxiv.org/abs/2506.16608)  
**代码**: [GitHub](https://github.com/hejm37/da-ac)  
**领域**: 强化学习 / Agent  
**关键词**: 统一动作空间, 分布参数化, 确定性策略梯度, 离散连续混合控制, 方差缩减  

## 一句话总结
DA-AC 提出将动作分布的参数（如 softmax 概率或 Gaussian 均值/方差）作为 Agent 的"动作"输出，将动作采样过程移入环境，从而用统一的确定性策略梯度框架处理离散/连续/混合动作空间，理论证明方差严格低于 LR 和 RP 估计器，并在 40+ 环境上取得 competitive 或 SOTA 性能。

## 研究背景与动机
1. **领域现状**：当前 RL 算法与动作空间类型紧密耦合——离散用 DQN/DSAC，连续用 DDPG/TD3/SAC，混合动作需要 PADDPG 等专用算法。不同估计器架构完全不同，难以设计跨域统一的通用算法。
2. **现有痛点**：
   - LR（Likelihood Ratio）估计器虽通用，但方差高，需要精心设计 baseline
   - DPG/RP 估计器方差低，但只能用于连续动作空间
   - 混合动作空间（同时含离散和连续维度）需要额外的工程设计
3. **核心矛盾**：需要低方差的梯度估计器，但低方差的 DPG/RP 又要求连续动作空间——如何在离散动作上也享受 DPG 的低方差优势？
4. **本文要解决**：设计一个统一的 actor-critic 算法，能在任意类型的动作空间上工作，且理论上保证低方差。
5. **切入角度**：重新思考 Agent-环境的边界——Agent 的"动作"不一定要是环境定义的原始动作，可以是**分布参数**。策略通常可以分解为 $\bar{\pi}_\theta$ (输出分布参数) + $f$ (从分布中采样)。如果把 $f$ 移到环境侧，Agent 的动作空间就变成了连续的参数空间 $\mathcal{U}$，无论原始动作空间是什么类型。
6. **核心idea**：Distributions-as-Actions——分布参数就是动作，采样是环境的一部分。

## 方法详解

### 整体框架
在经典 RL 中，Agent 的策略 $\pi_\theta$ 包含两部分：$\bar{\pi}_\theta$ (映射状态到分布参数) 和 $f$ (从分布采样得到动作)。DA 框架将 $f$ 移入环境侧，Agent 直接输出分布参数 $u = \bar{\pi}_\theta(s)$。这定义了一个新的 MDP——DA-MDP $\langle \mathcal{S}, \mathcal{U}, \bar{p}, d_0, \bar{r}, \gamma \rangle$，其中转移和奖励变为对原始动作的期望：

$$\bar{p}(s'|s,u) = \mathbb{E}_{A \sim f(\cdot|u)}[p(s'|s,A)], \quad \bar{r}(s,u) = \mathbb{E}_{A \sim f(\cdot|u)}[r(s,A)]$$

关键不变量：$\bar{v}_{\bar{\pi}}(s) = v_\pi(s)$——状态价值不变；$\bar{q}_{\bar{\pi}}(s,u) = \mathbb{E}_{A \sim f(\cdot|u)}[q_\pi(s,A)]$——分布参数的 Q 值等于原始 Q 值在分布下的期望。

### 关键设计

1. **DA-PG 梯度估计器（Theorem 4.2）**:
   - 做什么：在分布参数空间上的确定性策略梯度
   - 核心公式：$\hat{\nabla}_\theta^{\text{DA-PG}} = \nabla_\theta \bar{\pi}_\theta(S_t)^\top \nabla_U \bar{Q}_w(S_t, U)|_{U=\bar{\pi}_\theta(S_t)}$
   - 与 DPG 的关系：数学形式相同，但 $\bar{\pi}$ 输出分布参数而非单一动作，$\bar{Q}$ 估计分布下的期望回报
   - DPG 是 DA-PG 的特例（Prop 4.3）：当 $f(\cdot|u)$ 退化为 Dirac delta 时两者等价
   - 设计动机：由于 $\mathcal{U}$ 总是连续的，可以在任何动作空间上使用 DPG 风格的梯度

2. **方差严格降低的理论保证（Prop 4.4 & 4.5）**:
   - DA-PG 是 LR 估计器的条件期望（对动作 $A$ 取期望），根据全方差公式，方差严格更低
   - DA-PG 同样是 RP 估计器的条件期望（对噪声 $\epsilon$ 取期望），方差也严格更低
   - 代价：可能增加偏差（critic 的输入空间变大，更难学习）
   - 这是首个在离散动作空间上提供无偏 RP 风格低方差估计器的方法

3. **ICL（Interpolated Critic Learning）**:
   - 做什么：改善分布参数空间中 critic 的学习质量
   - 核心思路：标准 TD 更新只在当前策略的分布参数 $U_t$ 处学习 critic，导致 critic 在参数空间其他位置不准确。ICL 在当前参数 $U_t$ 和采样动作对应的确定性参数 $U_{A_t}$ 之间线性插值：$\hat{U}_t = \omega U_t + (1-\omega) U_{A_t}$, $\omega \sim \text{Uniform}[0,1]$
   - 设计动机：鼓励 critic 在分布参数空间中学到平滑的曲率信息，使策略梯度能指向高价值区域。类似 off-policy 学习，但操作对象是分布而非策略
   - 在 bandit 实验中可视化验证：ICL 学到的 critic 有更丰富的曲率，标准更新的 critic 仅在当前策略附近准确

4. **DA-AC 算法**:
   - 基于 TD3 构建：双 critic、延迟策略更新、目标噪声
   - 用 DA-PG 替换 DPG 更新 actor，ICL 替换标准 TD 更新 critic
   - 移除了 actor 目标网络（消融显示无需）
   - 对不同动作空间的适配：Gaussian（连续）、Softmax（离散）、Gaussian+Softmax（混合）

## 实验关键数据

### 主实验——连续控制（MuJoCo + DMC, 20 环境, 1M steps）
| 算法 | MuJoCo (归一化) | DMC (归一化) |
|------|----------------|-------------|
| TD3 | ~0.82 | ~0.70 |
| SAC | ~0.78 | ~0.65 |
| RP-AC | ~0.80 | ~0.72 |
| PPO | ~0.55 | ~0.48 |
| **DA-AC** | **~0.85** | **~0.78** |

在 20 个单独环境对比中，DA-AC 在多数环境优于 TD3，尤其在高维动作空间（如 Humanoid、Dog）中优势显著。

### 离散控制（Classic Control + MinAtar, 9 环境）
DA-AC 在 Classic Control 和 MinAtar 上均与 DQN 可比较，显著优于 LR-AC 和 ST-AC。

### 高维离散控制（$7^{17}$ 动作空间的 Humanoid）
DQN、DSAC、EAC 完全无法扩展（需要枚举不可行动作空间），DA-AC 保持与连续版本相当的性能。

### 混合控制（7 个 PAMDPs 环境）
DA-AC 与 PATD3（专为混合动作设计的算法）可比较或更优。

### 消融实验——ICL 的贡献
| 配置 | MuJoCo | DMC | MinAtar | Hybrid |
|------|--------|-----|---------|--------|
| DA-AC (w/ ICL) | **0.85** | **0.78** | **0.73** | **0.82** |
| DA-AC w/o ICL | 0.80 | 0.72 | 0.68 | 0.77 |

ICL 在所有设置中带来一致的提升，paired t-test 显示统计显著。

### 关键发现
- **统一性验证**：一个算法在离散、连续、混合三种动作空间上都 competitive——这是首次实现
- **DA-PG 的方差优势**：bias-variance 分析实验确认 DA-PG 方差低于 LR 和 RP，验证了理论预测
- **ICL 对 critic 质量至关重要**：可视化显示 ICL 使 critic 在分布参数空间中有更好的梯度信号

## 亮点与洞察
- **Agent-环境边界的重新思考**极其优雅——这不是 trick，而是一个深刻的概念转变：改变看待问题的方式可以统一原本不兼容的方法
- **DA-PG 作为 LR 和 RP 的条件期望**——这个理论结果非常漂亮，用全方差公式直接证明方差降低
- **ICL 的设计直觉**：标准 critic 只在策略附近准确，但策略优化需要知道其他区域的 Q 值；ICL 通过插值采样让 critic "看得更远"——类似 off-policy 但在分布空间操作
- **高维离散控制**的实验特别有说服力：$7^{17}$ 的动作空间让 DQN/DSAC/EAC 完全失效，但 DA-AC 无缝处理

## 局限性 / 可改进方向
- ICL 引入偏差（无重要性采样修正），对收敛性的影响尚待理论分析
- 仅基于 TD3 实现，与 SAC（最大熵）或 PPO（on-policy）的整合是自然的下一步
- critic 学习在高维分布参数空间中可能更困难——Gaussian 策略中 critic 输入翻倍（均值+方差）
- 未探索 model-based RL、层次控制等扩展方向

## 相关工作与启发
- **vs TD3/DDPG**：DA-AC 是 TD3 的自然推广，DPG 是 DA-PG 的特例
- **vs SAC**：SAC 用 RP 估计器+熵正则，DA-AC 用 DA-PG（方差更低）+无熵，两者可结合
- **vs EPG (Expected Policy Gradient)**：EPG 也追求零方差但仅在低维离散或特殊连续情况可行，DA-AC 推广到任意高维
- **vs Gumbel-Softmax/ST**：这些是有偏的离散松弛方法，DA-PG 在离散空间上提供首个无偏低方差估计
- 可启发 Agent 系统设计：当 Agent 需要同时做离散决策和连续控制时，DA 框架提供统一接口

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 重新定义 Agent-环境边界的概念非常优雅且深刻
- 实验充分度: ⭐⭐⭐⭐⭐ 40+ 环境覆盖 4 种动作空间类型，消融全面
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰，可视化有效
- 价值: ⭐⭐⭐⭐⭐ 为 RL 统一框架迈出重要一步，实际可用
