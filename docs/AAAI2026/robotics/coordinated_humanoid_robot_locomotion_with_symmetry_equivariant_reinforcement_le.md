---
title: >-
  [论文解读] Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy
description: >-
  [AAAI 2026][机器人][人形机器人] 提出 SE-Policy，将严格的对称等变性（actor）和对称不变性（critic）直接嵌入神经网络架构，无需额外超参数即可使人形机器人产生时空协调的自然运动，速度跟踪误差相比 DreamWaQ 降低 40%，并成功部署到 Unitree G1 实体机器人。
tags:
  - "AAAI 2026"
  - "机器人"
  - "人形机器人"
  - "对称等变"
  - "深度强化学习"
  - "运动协调"
  - "PPO"
  - "仿真到现实"
---

# Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy

**会议**: AAAI 2026  
**arXiv**: [2508.01247](https://arxiv.org/abs/2508.01247)  
**代码**: 无  
**领域**: 视频理解  
**关键词**: 人形机器人, 对称等变, 深度强化学习, 运动协调, PPO, 仿真到现实

## 一句话总结

提出 SE-Policy，将严格的对称等变性（actor）和对称不变性（critic）直接嵌入神经网络架构，无需额外超参数即可使人形机器人产生时空协调的自然运动，速度跟踪误差相比 DreamWaQ 降低 40%，并成功部署到 Unitree G1 实体机器人。

## 研究背景与动机

人形机器人具有天然的双侧对称形态——左右臂和左右腿结构镜像对称，与人类神经系统的双侧对称性一致。然而，现有基于深度强化学习（DRL）的控制策略通常将网络视为黑盒，完全忽略了这一形态学先验：

**对称观测产生不对称动作**：对于镜像对称的状态输入（如"左脚悬空"与"右脚悬空"），策略网络可能输出完全不同的动作响应，导致左右关节运动风格不一致。

**运动不协调**：例如 DreamWaQ 策略下左右脚的步幅分别为 0.13m 和 0.25m，步态严重不对称，既不自然也降低了跟踪精度。

**用户体验差**：不协调的运动模式看起来不自然，且降低了任务性能。

现有的对称性利用方法大致分三类，但各有缺陷：

- **时序对称方法**（周期信号、CPG）：仅鼓励运动的周期性，约束较松，对称性得不到保证。
- **数据增强**（对称经验回放）：通过将采集到的 transition 及其镜像副本一起训练来诱导等变性，但只是软约束，推理时仍可能违反对称性。
- **损失正则化**：添加等变惩罚项 $\mathcal{L}_{reg} = \|\pi(\mathcal{F}_o(o)) - \mathcal{F}_a(\pi(o))\|^2$，需要额外超参数调节且可能干扰策略优化过程。
- **严格等变网络**：虽然在经典控制任务中有效，但在真实人形机器人上的效果尚未被充分验证。

**核心动机**：能否设计一种无需额外超参数、将严格对称等变性直接编码进网络结构的方法，让人形机器人自然地产生时空协调的运动？

## 方法详解

### 整体框架

SE-Policy 基于 actor-critic 架构（PPO 算法），由四个组件构成：

- **历史编码器** $f_{en}$：输入过去 $h$ 步的观测序列 $o_{[t-h:t]}$，输出隐特征 $z$
- **观测解码器** $f_{de}$：预测下一步观测 $\hat{o}_{t+1}$，用于自监督训练编码器
- **策略网络** $f_\pi$（actor）：基于当前观测 $o_t$ 和隐特征 $z$，输出关节目标位置
- **价值网络** $V$（critic）：利用观测和特权地形信息评估状态值

关键创新在于：**actor 中所有网络模块满足严格等变性，critic 满足严格不变性**——这是通过网络架构本身的参数共享实现的，而非通过训练损失软约束。

### 对称 MDP 的数学基础

人形机器人的 MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{O}, \mathcal{A}, P, R, \gamma \rangle$ 具有反射对称性。定义对称变换函数 $\mathcal{F}_s$、$\mathcal{F}_o$、$\mathcal{F}_a$ 分别作用于状态、观测和动作，则：

- **转移不变性**：$P(\mathcal{F}_s(s')|\mathcal{F}_s(s), \mathcal{F}_a(a)) = P(s'|s,a)$
- **奖励不变性**：$R(\mathcal{F}_s(s), \mathcal{F}_a(a)) = R(s,a)$

由此可推导出最优策略的等变性质：$\pi^*(\mathcal{F}_s(s)) = \mathcal{F}_a(\pi^*(s))$，以及最优值函数的不变性质：$V(\mathcal{F}_s(s)) = V(s)$。

直观理解：当机器人左脚悬空时最优动作是放下左脚；那么在对称状态（右脚悬空）下，最优动作就是放下右脚——即对称状态对应对称动作。

### 观测与动作空间的对称变换设计

观测空间维度 96，包含 7 个分量。每个分量都有精心设计的对称变换规则：

- **角速度** $\omega$：$(ω_x, ω_y, ω_z) \to (-ω_x, ω_y, -ω_z)$，反射翻转 x 和 z 轴
- **重力投影** $g$：$(g_x, g_y, g_z) \to (g_x, -g_y, g_z)$，翻转 y 轴
- **速度指令** $c$：$(c_x, c_y, c_\omega) \to (c_x, -c_y, -c_\omega)$，翻转横向速度和角速度
- **关节位置/速度/上一步动作**：左右互换并取反，如 $\theta_{left}^{arm} \leftrightarrow -\theta_{right}^{arm}$
- **相位输入** $\Phi$：$(\Phi_{sin}, \Phi_{cos}) \to (-\Phi_{sin}, -\Phi_{cos})$，半周期相移
- **高度图** $H$：左右子图互换，中间不变

动作空间维度 27（27 个关节目标位置），变换规则与关节位置相同。

### 等变神经网络构造

所有网络使用基于 ESCNN 框架的参数共享线性层 + ReLU 构建等变 MLP。核心思想是：网络权重矩阵被约束为满足等变性的特定结构，使得：

$$f(\mathcal{F}_{input}(x)) = \mathcal{F}_{output}(f(x))$$

对于隐特征 $z$（偶数维），定义对称变换 $\mathcal{F}_z$：相邻元素互换，即 $[\mathcal{F}_z(z)]_i = z_{i+1}$（$i$ 为奇数）或 $z_{i-1}$（$i$ 为偶数）。这是一个简洁但有效的设计——确保编码器和解码器可以桥接输入输出的不同对称变换。

### 训练策略

训练基于标准 PPO 算法，损失函数包含：

1. **PPO 策略损失**：$\mathcal{L}_{PPO} = \mathbb{E}[\min(\rho_\pi A, \text{clip}(\rho_\pi, 1-\xi, 1+\xi) A)]$
2. **自编码器损失**：$\mathcal{L}_{AE} = \text{MSE}(\hat{o}, o_{t+1})$，训练历史编码器
3. **价值损失**：$\mathcal{L}_V = \text{MSE}(V(H_t, o_t), y)$，$y$ 为 reward-to-go

辅助训练技术：
- **课程学习**：地形难度（平坦→粗糙→离散→斜坡）、任务指令范围、传感器噪声逐步递增
- **域随机化**：摩擦系数 [0.7,1.0]、基座质量 ±5kg、电机强度 [0.9,1.1]、电机延迟 [0.02,0.1]s 等，用于缩小 sim-to-real gap

奖励函数设计涵盖速度跟踪（权重 2.0）、存活奖励（2.0）、z 轴速度惩罚（-1.0）、动作平滑性惩罚（-0.01）、关节正则化等共 14 项。

## 实验关键数据

### 表1：核心指标对比（仿真环境，速度跟踪任务）

| 指标 | DreamWaQ | DreamWaQ-Regu | SE-Policy (actor only) | **SE-Policy** |
|------|----------|---------------|----------------------|---------------|
| TE-V (cm/s) ↓ | 16.43±9.54 | 13.91±8.53 | 11.06±8.63 | **9.85±1.54** |
| Temp-S (×10⁻²rad) ↓ | 22.52±2.70 | 16.58±2.88 | 9.20±2.19 | **7.86±1.44** |
| Spat-S (×10⁻²rad) ↓ | 30.84±5.20 | 8.18±1.46 | 0.00±0.00 | **0.00±0.00** |

**核心结论**：SE-Policy 的速度跟踪误差仅 9.85 cm/s，比 DreamWaQ 低 40.0%，比 DreamWaQ-Regu 低 29.2%。空间对称性达到完美的 0.00（严格等变性的理论保证）。时序对称性也最优，说明等变约束同时改善了步态的周期一致性。

### 表2：消融分析与运动可视化关键数据

| 对比维度 | DreamWaQ | SE-Policy | 改善 |
|----------|----------|-----------|------|
| 左脚平均步幅 | 0.13m | 一致 | 步态对称 |
| 右脚平均步幅 | 0.25m | 一致 | 步态对称 |
| 轨迹对称性 | 左转偏差明显 | 镜像对称 | 轨迹精度↑ |
| 累积位置误差(TE-P) | 随时间发散 | 增长缓慢 | 长期稳定性↑ |
| 累积朝向误差(TE-O) | 随时间发散 | 增长缓慢 | 方向控制↑ |

**消融发现**：去掉不变 critic（SE-Policy actor only）后 TE-V 升高 12.2%、Temp-S 升高 17.0%，说明 critic 的不变性对策略优化过程同样重要——不变 critic 为等变 actor 提供了一致的价值评估信号。

## 亮点与洞察

1. **从软约束到硬约束的范式跃迁**：不同于数据增强或正则化的"期望等变"，SE-Policy 通过网络架构本身保证"必然等变"，消除了推理时违反对称性的可能。这一思路适用于任何具有形态对称性的机器人。

2. **零 Spat-S 的理论保证**：空间对称分数恒为 0 是架构约束的直接数学结果，而非训练优化的近似结果。这意味着对称观测下的动作输出在数学上完全等价。

3. **等变性带来的间接收益**：严格等变不仅改善了空间对称，还显著提升了时序对称（Temp-S 降低 65%），说明空间等变约束有效限制了策略搜索空间，使优化更容易找到周期协调的步态。

4. **Sim-to-Real 成功验证**：策略在真实 Unitree G1 上可处理草地、斜坡、沙地、碎石等多种地形，证明等变性在 domain gap 存在时仍保持鲁棒。

5. **隐特征对称变换设计巧妙**：$\mathcal{F}_z$ 采用相邻元素互换的简单规则，既满足群论要求又便于实现，是连接输入/输出不同对称群表示的关键桥梁。

## 局限性

1. **仅验证反射对称**：当前方法专注于人形机器人的左右镜像对称（$\mathbb{Z}_2$ 群），未拓展到更丰富的对称群（如旋转对称），限制了对非双侧对称机器人的适用性。
2. **任务局限**：仅在速度跟踪任务上验证，未涉及操作任务（抓取、搬运）或高动态任务（跑步、跳跃），这些场景下对称性假设可能被打破。
3. **隐空间对称设计依赖手工**：$\mathcal{F}_z$ 的"相邻互换"规则是人为指定的，对于更复杂的机器人结构，最优的隐空间对称变换不一定显而易见。
4. **缺少与模仿学习的对比**：现有人形机器人方法中基于运动捕捉参考的方法（如 AMP）已能产生自然运动，缺少与此类方法的直接比较。
5. **计算开销未详细分析**：等变层的参数共享可能减少参数量，但 ESCNN 的等变约束计算是否引入额外开销未做讨论。

## 相关工作与启发

- **DreamWaQ**（nahrendra2023dreamwaq）：本文直接基线。基于 PPO 的无模型方法，使用历史编码器提取运动上下文。SE-Policy 完全兼容其架构，仅将 MLP 替换为等变 MLP。
- **ESCNN**（cesa2022program）：提供等变网络的实现框架，基于参数共享实现线性层的群等变约束。
- **MDP 中的对称性**（zinkevich2001symmetry; van2020mdp）：理论基础——对称 MDP 中最优策略必然等变、最优值函数必然不变。
- 对 **四足机器人** 的启发：SE-Policy 的思想可直接迁移到四足机器人（具有更丰富的 $C_2$ 或 $C_4$ 对称性），可能带来更大的性能提升。
- 对 **多智能体系统** 的启发：同构智能体间的策略共享本质上也是一种等变性，可借鉴 SE-Policy 的架构设计。

## 评分

| 维度 | 分数 |
|------|------|
| 新颖性 | ⭐⭐⭐ |
| 理论深度 | ⭐⭐⭐⭐ |
| 实验充分性 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| **综合** | **⭐⭐⭐⭐** |

理论推导完整，从对称 MDP 性质出发自然引出架构设计；实验覆盖仿真+实物，消融设计合理。但核心创新在于将已有的等变网络技术（ESCNN）工程化地适配到人形机器人，新颖性偏工程应用层面而非方法论突破。任务场景较单一（仅速度跟踪），期待在更复杂任务上的验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](../../NeurIPS2025/robotics/adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[ICLR 2026\] Partially Equivariant Reinforcement Learning in Symmetry-Breaking Environments](../../ICLR2026/robotics/partially_equivariant_reinforcement_learning_in_symmetry-breaking_environments.md)
- [\[AAAI 2026\] RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms](rlslm_a_hybrid_reinforcement_learning_framework_aligning_rule-based_social_locom.md)
- [\[CVPR 2026\] Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control](../../CVPR2026/robotics/do_you_have_freestyle_expressive_humanoid_locomotion_via_audio_control.md)
- [\[CVPR 2026\] PvP: Data-Efficient Humanoid Robot Learning with Proprioceptive-Privileged Contrastive Representations](../../CVPR2026/robotics/pvp_data-efficient_humanoid_robot_learning_with_proprioceptive-privileged_contra.md)

</div>

<!-- RELATED:END -->
