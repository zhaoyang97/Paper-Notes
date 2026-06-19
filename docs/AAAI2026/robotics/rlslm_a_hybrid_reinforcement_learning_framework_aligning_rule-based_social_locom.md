---
title: >-
  [论文解读] RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms
description: >-
  [AAAI 2026][机器人][社会导航] 本文提出RLSLM，一种将心理学实验驱动的规则式社交运动模型（SLM）嵌入强化学习奖励函数的混合框架，使智能体在人群环境中高效学习符合人类社交规范的导航策略，VR实验验证其舒适度评分显著优于现有规则式基线。 问题背景 在人类密集环境中进行不引起不适的导航是社交智能体的核心能力…
tags:
  - "AAAI 2026"
  - "机器人"
  - "社会导航"
  - "强化学习"
  - "社交运动模型"
  - "VR实验"
  - "人机交互"
---

# RLSLM: A Hybrid Reinforcement Learning Framework Aligning Rule-Based Social Locomotion Model with Human Social Norms

**会议**: AAAI 2026  
**arXiv**: [2511.11323](https://arxiv.org/abs/2511.11323)  
**代码**: [github.com/kouyitian/RLSLM](https://github.com/kouyitian/RLSLM)  
**领域**: 强化学习  
**关键词**: 社会导航, 强化学习, 社交运动模型, VR实验, 人机交互

## 一句话总结

本文提出RLSLM，一种将心理学实验驱动的规则式社交运动模型（SLM）嵌入强化学习奖励函数的混合框架，使智能体在人群环境中高效学习符合人类社交规范的导航策略，VR实验验证其舒适度评分显著优于现有规则式基线。

## 研究背景与动机

### 问题背景

在人类密集环境中进行不引起不适的导航是社交智能体的核心能力。现有方法主要分两类：

**规则式方法**：基于心理学原理（如个人空间理论proxemics、社会力模型）设计确定性规则
- 优点：可解释性强、计算开销低
- 缺点：难以精确量化、泛化性差、可能产生不自然的振荡路径

**数据驱动方法**：使用RL/模仿学习从大规模数据集学习
- 优点：能学习复杂行为、表现力强
- 缺点：高度依赖数据质量、训练昂贵、缺乏可解释性、难以与人类直觉对齐

### 核心动机

**关键问题**：能否将两种方法整合，构建高效、可适应、可解释且与真实人类社会行为对齐的模型？

现有导航研究中社会规则的设计大多基于直觉或数据统计，而非来自严格控制的人类行为实验。第三人称用户研究存在生态效度问题。作者提出：**将前沿心理学研究的定量发现直接嵌入RL训练流程**，并通过沉浸式VR实验验证。

### 独特创新点

- 首次将心理学行为实验得到的定量社交运动模型作为RL奖励信号
- 通过VR第一人称沉浸式实验评估舒适度（高生态效度）
- 在极少的训练步数（10,000步）内即可学习社交对齐的导航策略

## 方法详解

### 整体框架

RLSLM框架遵循三阶段决策循环：
1. **环境观察**：捕获智能体自身位置及周围行人的相对位置和朝向
2. **动作选择**：基于Actor-Critic网络生成导航动作
3. **策略更新**：通过多维反馈机制（机械能+目标进度+社交影响）更新策略

核心思路是：自上而下的规则式方法提供先验知识→编码为奖励函数→自下而上的RL在真实场景数据上优化策略。

### 关键设计

#### 1. **环境观察模块**

观察向量包含智能体自身位置及周围 $n$ 个行人的相对位置和朝向，拼接为结构化输入：

$$s_t \in \mathbb{R}^{3n+2}$$

每个行人用3个参数描述（相对x/y坐标 + 朝向角），智能体自身用2个参数（位置坐标）。

#### 2. **Actor-Critic动作选择**

采用A2C（Advantage Actor-Critic）算法：
- **Actor网络**：表示策略 $\pi(a_t|s_t)$，输出动作概率分布，支持探索与利用平衡
- **Critic网络**：估计价值函数 $V(s_t)$，预测从当前状态的期望回报
- 网络架构：5层MLP（64-128-256-128-64），优化器RMSprop，学习率 $5 \times 10^{-4}$

#### 3. **多维反馈机制（核心创新）**

奖励由三个分量组成，平衡社会影响和机械能消耗：

**a) 机械能惩罚 $R_e$**：
$$R_e(s_t) = -\alpha$$
每步固定能量消耗惩罚（$\alpha=1$），激励智能体用最少步数到达目标。

**b) 目标进度奖励 $R_d$**：
$$R_d(s_t, s_{t-1}) = \frac{D_{t-1} - D_t}{l}$$
与到目标距离的减少量成正比，鼓励向目标方向前进。

**c) 社交影响惩罚 $R_s$（核心）**：

基于Zhou et al.的行为实验数据，构建朝向敏感的非对称社交不适场（social comfort field）。社交影响由三个子分量组成：

- **HRSC（朝向相关社交分量）**：$m \times f(\theta_h)$，其中 $f(\theta_h) = \max(\cos(\theta_h), 0)$。捕获"面对面时不适感更强"的心理学规律
- **HISC（朝向无关社交分量）**：常数 $n$，表示基础个人空间不适
- **CAC（碰撞避免分量）**：$c \times I_{CA}$，将人体近似为椭圆，建模物理碰撞风险

总社交影响公式：
$$F = \frac{I_{\text{agent}} \times I_{\text{person}}}{d^2}$$
$$I_{\text{human}} = m \times f(\theta_h) + n + c \times I_{CA}$$

标准化后：$F' = \min(F/K, 1)$，其中 $K=10.180$ 为行为数据拟合的上限。

所有参数（$m_a=0.321, n_a=0.856, m_p=0.438, n_p=0.630, a=0.285, b=0.175, c=1.430$）均来自心理学实验的拟合结果。

**总奖励函数**：
$$r_t = R_d(s_t, s_{t-1}) + R_e(s_t) + \sigma R_s(s_t), \quad 0 < t < T$$
$$r_T = \pm C \quad (\text{终止奖惩})$$

其中 $\sigma=0.5$（社交影响权重）、$C=500$（终止奖惩）、$\gamma=0.9$（折扣因子）。

### 损失函数 / 训练策略

- 算法：A2C（Stable-Baselines3实现）
- 网络：MLP策略，5层（64-128-256-128-64）
- 优化器：RMSprop，lr = $5\times10^{-4}$
- 折扣因子：$\gamma = 0.8$
- 训练预算：每次运行10,000步（极低训练成本）
- 硬件：NVIDIA 3090 GPU + CUDA
- 环境：OpenAI Gymnasium
- 分别为单人和多人场景训练独立模型

## 实验关键数据

### VR人机交互实验

**参与者**：30名大学生/教职员工（11男19女，18-29岁），正常/矫正视力

**设备**：HTC Vive Pro头显（双目分辨率2880×1600，刷新率90Hz，视场角110°）

**实验设计**：50个场景（25单人+25多人），每个场景×3种算法=150 trials。参与者以第一人称视角体验并在1-5量表上评分。

### 主实验

**用户舒适度评分对比：**

| 模型 | 单人场景均分 | 多人场景均分 | 总均分 | 统计显著性 |
|------|-----------|-----------|-------|----------|
| **RLSLM** | ~3.8 | ~4.5 | **4.21/5** | - |
| COMPANION | ~2.8 | ~2.9 | ~3.09 | P < 0.001 |
| n-Body | ~2.5 | ~3.1 | ~2.80 | P < 0.001 |

**统计分析**：重复测量ANOVA显示模型类型对舒适度有显著主效应（$F_{(2,58)}=219.589$, $P<0.001$, $\eta_G^2=0.525$）。RLSLM在单人和多人场景中均显著优于两个基线（Bonferroni校正 $P<0.001$）。

**关键发现**：RLSLM的舒适度增量 $\Delta\text{rating}=1.12$，相对最佳基线提升36%。

### 消融实验

**社交行为权重 $\sigma$ 敏感性分析：**

| $\sigma$ 值 | 行为模式 | MLD（最大横向偏移） |
|-------------|---------|-------------------|
| 0 | 严格最短路径 | 最小 |
| 0.5 | 适度绕行 | 中等 |
| 1.0 | 较大绕行 | 较大 |
| 2.0 | 过度保守 | 最大 |

$\sigma$ 有效控制了社交敏感度，验证了奖励函数的可解释性。

**HRSC消融（朝向相关分量）：**

| 配置 | 从正面经过人的次数 (42场景) | 说明 |
|------|--------------------------|------|
| 完整模型 | 5次 (11.9%) | 对朝向敏感 |
| w/o HRSC | 23次 (57.76%) | 失去朝向意识，随机绕行 |

移除HRSC后智能体失去对行人朝向的感知，在正面经过的比例从12%跃升至58%。

**HISC和CAC消融（朝向无关分量）：**

| 配置 | MLD变化 | 说明 |
|------|---------|------|
| 完整模型 | 基准 | 稳定导航 |
| w/o HISC | 降低 | 个人空间感知减弱 |
| w/o CAC | 降低 | 碰撞避免能力减弱 |

### 关键发现

1. **混合框架优于纯规则方法**：RLSLM在用户舒适度上显著超越COMPANION和n-Body
2. **心理学先验大幅降低训练成本**：仅需10,000步即收敛，远少于纯数据驱动方法
3. **社交影响权重 $\sigma$ 提供直观调控旋钮**：可根据应用场景调整社交保守程度
4. **三个社交分量各司其职**：消融证明每个分量对特定社交行为不可或缺
5. **多人场景优势明显**：RLSLM在多人交互场景中的提升更为显著
6. **生态效度高**：VR第一人称实验比传统第三人称视频评估更真实

## 亮点与洞察

1. **跨学科融合的典范**：将定量心理学实验结果直接嵌入RL奖励函数，实现认知科学与机器学习的有机结合
2. **训练效率极高**：10,000步收敛，证明良好的先验知识能戏剧性降低数据需求
3. **可解释性强**：每个奖励分量和参数都有明确的心理学含义，而非黑箱
4. **双向价值**：框架不仅服务机器人导航，还可作为心理学研究的计算工具
5. **VR评估管线可复用**：开源的VR评估工具链可用于其他社交导航研究的标准化测试

## 局限与展望

1. **静态行人假设**：当前实验中行人是静止的，未考虑动态行人避障
2. **参数固定**：SLM参数来自特定实验，未必适用于所有文化和场景
3. **动作空间简单**：仅输出移动方向，未建模速度变化
4. **场景规模受限**：15m×15m虚拟环境，未验证大规模开放场景
5. **单一评估指标**：主要依赖主观舒适度评分，缺乏导航效率的定量权衡分析
6. **缺乏与深度学习基线的比较**：如Social Force GAN、STGCNN等数据驱动方法

## 相关工作与启发

- **Zhou et al. (2022)**：提供SLM的核心行为实验数据，是本文的心理学基础
- **Social Force Model (Helbing 1995)**：经典的物理力模型，但不如SLM有人类行为实验支撑
- **COMPANION (Kirby 2009)**：约束优化导航方法，本文基线之一
- **n-Body (van den Berg 2011)**：基于互惠碰撞避免的方法，本文基线之一
- 启发：**将领域专家知识编码为奖励函数是提升RL训练效率和可解释性的有效策略**，尤其在数据稀缺场景中

## 评分

- 新颖性: ⭐⭐⭐⭐（心理学模型嵌入RL奖励的混合范式新颖）
- 实验充分度: ⭐⭐⭐⭐（VR人机实验+消融+敏感性分析全面）
- 写作质量: ⭐⭐⭐⭐（清晰易懂，跨学科工作表达良好）
- 价值: ⭐⭐⭐⭐（为社交导航提供了可复用的混合范式和评估工具链）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy](coordinated_humanoid_robot_locomotion_with_symmetry_equivariant_reinforcement_le.md)
- [\[CVPR 2025\] SOLAMI: Social Vision-Language-Action Modeling for Immersive Interaction with 3D Autonomous Characters](../../CVPR2025/robotics/solami_social_vision-language-action_modeling_for_immersive_interaction_with_3d_.md)
- [\[AAAI 2026\] Theory of Mind for Explainable Human-Robot Interaction](theory_of_mind_for_explainable_human-robot_interaction.md)
- [\[AAAI 2026\] Towards Affordance-Aware Robotic Dexterous Grasping with Human-like Priors](towards_affordance-aware_robotic_dexterous_grasping_with_human-like_priors.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)

</div>

<!-- RELATED:END -->
