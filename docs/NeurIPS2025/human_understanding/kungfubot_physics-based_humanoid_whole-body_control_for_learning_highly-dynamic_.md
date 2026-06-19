---
title: >-
  [论文解读] KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills
description: >-
  [NeurIPS 2025][人体理解][humanoid control] 提出 PBHC 框架，通过物理感知运动处理流水线和自适应跟踪因子的双层优化，使人形机器人（Unitree G1）学会功夫、舞蹈等高动态全身动作，跟踪误差显著优于现有方法并成功实机部署。 - 高动态动作模仿极具挑战：现有人形机器人运动模仿方法（ExB…
tags:
  - "NeurIPS 2025"
  - "人体理解"
  - "humanoid control"
  - "motion imitation"
  - "reinforcement-learning"
  - "adaptive tracking"
  - "sim-to-real"
---

# KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills

**会议**: NeurIPS 2025  
**arXiv**: [2506.12851](https://arxiv.org/abs/2506.12851)  
**代码**: [项目主页](https://kungfu-bot.github.io)  
**领域**: 视频理解  
**关键词**: humanoid control, motion imitation, reinforcement-learning, adaptive tracking, sim-to-real

## 一句话总结

提出 PBHC 框架，通过物理感知运动处理流水线和自适应跟踪因子的双层优化，使人形机器人（Unitree G1）学会功夫、舞蹈等高动态全身动作，跟踪误差显著优于现有方法并成功实机部署。

## 研究背景与动机

- **高动态动作模仿极具挑战**：现有人形机器人运动模仿方法（ExBody、OmniH2O 等）仅能跟踪低速平滑动作，对功夫踢腿、旋转跳跃等高动态行为无能为力。
- **参考运动物理可行性问题**：从视频提取的人体运动序列可能违反机器人物理约束（关节极限、动力学），直接用 RL 最大化跟踪奖励难以收敛。
- **固定跟踪容差的不足**：现有方法使用固定的跟踪奖励参数（tracking factor σ），无法适应不同难度的动作——σ 太大则奖励对误差不敏感，σ 太小则奖励趋近零。
- **数据集预处理代价高**：H2O 需训练特权策略过滤不可行动作，ExBody2 需训练初始策略评估动作难度——都是昂贵的前置步骤。

## 方法详解

### 阶段一：运动处理流水线

**1. 视频运动提取**：使用 GVHMR 从单目视频估计 SMPL 格式运动，其重力-视角坐标系消除身体倾斜问题。

**2. 物理感知运动过滤**：基于 CoM-CoP 距离判断动态稳定性。令 $\bar{\mathbf{p}}_t^{\text{CoM}}$ 和 $\bar{\mathbf{p}}_t^{\text{CoP}}$ 分别为质心和压力中心在地面的投影：

$$\Delta d_t = \|\bar{\mathbf{p}}_t^{\text{CoM}} - \bar{\mathbf{p}}_t^{\text{CoP}}\|_2 < \epsilon_{\text{stab}}$$

一个 $N$ 帧序列被认为稳定需满足：① 首尾帧均稳定；② 连续不稳定帧最长不超过阈值 $\epsilon_N$。

**3. 接触掩码与运动修正**：基于零速度假设从踝关节位移估计接触掩码：

$$c_t^{\text{left}} = \mathbb{I}[\|\mathbf{p}_{t+1}^{\text{l-ankle}} - \mathbf{p}_t^{\text{l-ankle}}\|_2^2 < \epsilon_{\text{vel}}] \cdot \mathbb{I}[p_{t,z}^{\text{l-ankle}} < \epsilon_{\text{height}}]$$

对悬浮伪影施加垂直偏移修正：$\psi_{t,z}^{\text{corr}} = \psi_{t,z} - \Delta h_t$，其中 $\Delta h_t = \min_{v \in \mathcal{V}_t} p_{t,z}^v$，随后用 EMA 平滑消除抖动。

**4. 运动重定向**：使用基于逆运动学的可微优化，在满足关节极限的前提下对齐末端执行器轨迹。

### 阶段二：自适应运动跟踪

**指数形式跟踪奖励**：

$$r(x) = \exp(-x / \sigma)$$

其中 $x$ 为跟踪误差（如关节角 MSE），$\sigma$ 为跟踪因子。$\sigma$ 过大→奖励对误差不敏感；$\sigma$ 过小→奖励趋零。

**最优跟踪因子的双层优化**：将 σ 的选择形式化为双层优化问题：

$$\max_{\sigma \in \mathbb{R}_+} J^{\text{ex}}(\mathbf{x}^*), \quad \text{s.t.} \quad \mathbf{x}^* \in \arg\max_{\mathbf{x} \in \mathbb{R}_+^N} J^{\text{in}}(\mathbf{x}, \sigma) + R(\mathbf{x})$$

其中内层目标 $J^{\text{in}} = \sum_{i=1}^N \exp(-x_i/\sigma)$ 是简化累积奖励，外层目标 $J^{\text{ex}} = \sum_{i=1}^N -x_i^*$ 最小化总跟踪误差。求解得最优跟踪因子等于平均最优跟踪误差：

$$\sigma^* = \left(\sum_{i=1}^N x_i^*\right) / N$$

**自适应更新机制**：维护跟踪误差的 EMA 估计 $\hat{x}$，在训练中动态收紧 σ：

$$\sigma \leftarrow \min(\sigma, \hat{x})$$

σ 单调不增，从较大初始值开始，随策略改善逐步收紧，形成闭环反馈。

### RL 训练框架

- **非对称 Actor-Critic**：Actor 仅观测本体感知 + 时间相位；Critic 额外接收参考运动位置、根线速度和随机物理参数
- **奖励向量化**：每个奖励分量 $r_i$ 配一个独立的价值头 $V_i(\mathbf{s})$，避免标量聚合导致的值估计不准
- **参考状态初始化（RSI）**：从参考运动中随机采样时间点初始化，并行学习不同运动阶段
- **域随机化 + 零样本迁移**：变化仿真物理参数，策略直接部署到真实 G1 机器人

## 实验

### 表1：主要跟踪性能对比（均值±标准差，粗体为最优）

| 方法 | $E_{\text{g-mpbpe}}$↓(mm) | $E_{\text{mpbpe}}$↓(mm) | $E_{\text{mpjpe}}$↓($10^{-3}$rad) |
|:---|:---:|:---:|:---:|
| **Easy** | | | |
| OmniH2O | 233.54±4.0 | 103.67±1.9 | 1805.1±12.3 |
| ExBody2 | 588.22±11.4 | 332.50±3.6 | 4014.4±21.5 |
| **PBHC** | **53.25±17.6** | **28.16±6.1** | **725.6±16.2** |
| **Medium** | | | |
| OmniH2O | 433.64±16.2 | 151.42±7.3 | 2333.9±49.5 |
| ExBody2 | 619.84±26.2 | 261.01±1.6 | 3738.7±26.9 |
| **PBHC** | **126.48±27.0** | **48.87±7.6** | **1043.3±104** |
| **Hard** | | | |
| OmniH2O | 446.17±12.8 | 147.88±4.1 | 1939.5±23.9 |
| ExBody2 | 689.68±11.8 | 246.40±1.3 | 4037.4±16.7 |
| **PBHC** | **290.36±139** | **124.61±54** | **1326.6±379** |

PBHC 在所有难度级别和所有指标上一致超越可部署基线（OmniH2O、ExBody2），Easy 级别全局位置误差从 234mm 降至 53mm（降幅 77%）。

### 表2：真实世界太极拳跟踪性能

| 平台 | $E_{\text{mpbpe}}$↓ | $E_{\text{mpjpe}}$↓ | $E_{\text{mpbve}}$↓ | $E_{\text{mpjve}}$↓ |
|:---|:---:|:---:|:---:|:---:|
| MuJoCo | 33.18±2.7 | 1061±83.3 | 2.96±0.34 | 67.71±6.7 |
| Real | 36.64±2.6 | 1130±9.5 | 3.01±0.13 | 65.68±2.0 |

真实世界指标与仿真高度接近，验证了零样本 sim-to-real 迁移的有效性。

### 自适应机制消融

固定 σ 配置（Coarse/Medium/UpperBound/LowerBound）在不同动作上表现不一致——某些配置在特定动作上好但在其他动作上差。自适应机制在所有动作类型上一致达到近最优性能。

### 物理过滤有效性

对 10 条运动序列应用过滤，4 条被拒绝、6 条被接受。接受的动作 ELR（Episode Length Ratio）均高，拒绝的动作最高仅 54%，验证了物理可行性指标的有效性。

## 亮点

- **自适应跟踪因子理论优雅**：将 σ 选择形式化为双层优化并推导出闭式解 σ*=平均误差，避免了人工调参
- **高动态能力突出**：成功在真实 G1 机器人上展示功夫出拳、360°旋转踢、太极等高难度动作
- **运动处理流水线完整**：从视频提取→物理过滤→接触修正→重定向，形成端到端管道
- **零样本 sim-to-real**：无需真实世界微调即可成功部署

## 局限性

- **单策略单动作**：每个动作需训练独立策略，无法高效扩展到大规模动作库
- **缺乏环境感知**：不具备地形感知和避障能力，限制了非结构化环境中的部署
- **依赖 MoCap 视频质量**：GVHMR 提取的运动可能不准确，物理过滤虽能剔除部分但非万能
- **Hard 级别方差较大**：Hard 动作的跟踪误差标准差很高，说明稳定性仍有提升空间

## 相关工作

- **人形运动模仿**（DeepMimic [Peng+ 2018]、ExBody2 [Ji+ 2024]、OmniH2O [He+ 2024]）：仅适用于低速动作，PBHC 将能力边界推至高动态
- **人形全身控制**（HugWBC、ASAP [He+ 2025]）：ASAP 使用多阶段+残差策略弥补 sim-to-real gap，PBHC 纯仿真内解决
- **运动处理**（IPMAN [Tripathi+ 2023]、GVHMR [Shen+ 2024]）：PBHC 在 GVHMR 基础上增加了物理过滤和接触修正

## 评分

- 新颖性: ⭐⭐⭐⭐ — 双层优化推导自适应 σ 的理论贡献扎实，物理过滤管道完整
- 实验充分度: ⭐⭐⭐⭐⭐ — 仿真对比 + 消融 + 真实部署 + sim-to-real 定量验证
- 写作质量: ⭐⭐⭐⭐ — 公式推导清晰，实验组织系统
- 价值: ⭐⭐⭐⭐ — 大幅推进人形机器人高动态技能学习的前沿

## 实验关键数据

## 亮点

## 局限与展望

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AudioAvatar: Personalized Audio-driven Whole-body Talking Avatars](../../CVPR2026/human_understanding/audioavatar_personalized_audio-driven_whole-body_talking_avatars.md)
- [\[CVPR 2026\] InterPrior: Scaling Generative Control for Physics-Based Human-Object Interactions](../../CVPR2026/human_understanding/interprior_scaling_generative_control_for_physics-based_human-object_interaction.md)
- [\[CVPR 2026\] CIGPose: Causal Intervention Graph Neural Network for Whole-Body Pose Estimation](../../CVPR2026/human_understanding/cigpose_causal_intervention_graph_neural_network_for_whole-body_pose_estimation.md)
- [\[CVPR 2026\] InterPhys: Physics-aware Human Motion Synthesis in a Dynamic Scene](../../CVPR2026/human_understanding/interphys_physics-aware_human_motion_synthesis_in_a_dynamic_scene.md)
- [\[CVPR 2026\] PAMotion: Physics-Aware Motion Generation for Full-Body Interaction with Multiple Objects](../../CVPR2026/human_understanding/pamotion_physics-aware_motion_generation_for_full-body_interaction_with_multiple.md)

</div>

<!-- RELATED:END -->
