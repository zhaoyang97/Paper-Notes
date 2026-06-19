---
title: >-
  [论文解读] Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration
description: >-
  [ICML2025][强化学习][无监督技能预训练] 提出 SUPE 方法，将无标签离线轨迹数据"用两次"——既用于 VAE 技能预训练，又通过 UCB 伪标签转化为高层 off-policy 数据加速在线探索，在 42 个稀疏奖励任务上全面超越已有方法。 无监督预训练在 NLP/CV 中已取得巨大成功…
tags:
  - "ICML2025"
  - "强化学习"
  - "无监督技能预训练"
  - "层次强化学习"
  - "离线数据利用"
  - "探索策略"
  - "伪标签"
---

# Leveraging Skills from Unlabeled Prior Data for Efficient Online Exploration

**会议**: ICML2025  
**arXiv**: [2410.18076](https://arxiv.org/abs/2410.18076)  
**作者**: Max Wilcoxson, Qiyang Li, Kevin Frans, Sergey Levine (UC Berkeley)
**代码**: [rail-berkeley/supe](https://github.com/rail-berkeley/supe)  
**领域**: 强化学习  
**关键词**: 无监督技能预训练, 层次强化学习, 离线数据利用, 探索策略, 伪标签

## 一句话总结

提出 SUPE 方法，将无标签离线轨迹数据"用两次"——既用于 VAE 技能预训练，又通过 UCB 伪标签转化为高层 off-policy 数据加速在线探索，在 42 个稀疏奖励任务上全面超越已有方法。

## 研究背景与动机

无监督预训练在 NLP/CV 中已取得巨大成功，但将其迁移到强化学习（RL）有独特挑战：RL 的在线微调不是模仿数据，而是通过**探索**找到解并自我迭代改进。因此，RL 预训练的核心问题不只是学好表征，更是学会**高效探索策略**。

无标签轨迹数据（来自未知策略或任务无关收集策略）是最容易获取的，但存在**纠缠问题**（entanglement problem）：通用环境知识与任务特定行为混在一起。例如 locomotion 数据中，我们要学会在世界中移动，但不一定要去训练数据中的位置。

已有方法通常只在一个阶段使用离线数据：
- **技能预训练方法**（SPiRL, OPAL 等）：从离线数据提取技能后就丢弃数据，在线从头学高层策略
- **Offline-to-online RL**（ExPLORe 等）：直接把离线数据当 off-policy 数据，但不做技能预训练

本文的关键洞察：无标签轨迹提供**双重价值**——(1) 学习多样化低层技能，(2) 作为组合这些技能的 off-policy 样本。两者结合可产生叠加效果。

## 方法详解

### 整体框架

SUPE 分为两个阶段：

**阶段一：离线技能预训练**
- 从无标签离线数据 $\mathcal{D}$ 中提取长度为 $H$ 的轨迹片段
- 用 trajectory-segment VAE 学习低层技能空间

**阶段二：在线 RL + 伪标签**
- 用预训练的轨迹编码器为离线数据打上高层动作标签 $\hat{z}$
- 用 UCB 奖励估计为离线数据打上乐观奖励标签 $\hat{r}$
- 将伪标签后的离线数据与在线 replay buffer 合并，训练高层策略 $\pi_\psi(z|s)$

### 轨迹 VAE 预训练

轨迹片段 $\tau_{[H]} = \{s_0, a_0, s_1, \ldots, s_{H-1}, a_{H-1}\}$ 输入编码器 $f_\theta(z|\tau)$ 得到技能潜变量 $z$ 的分布，技能策略 $\pi_\theta(a|s,z)$ 重构动作序列。同时学习状态依赖先验 $p_\theta(z|s_0)$。

**VAE 损失函数（核心公式）：**

$$\mathcal{L}_\theta(\tau) = \beta D_{\mathrm{KL}}(f_\theta(z|\tau) \| p_\theta(z|s_0)) - \mathbb{E}_{z \sim f_\theta(z|\tau)} \left[\sum_{h=0}^{H-1} \log \pi_\theta(a_h|s_h, z)\right]$$

其中第一项是 KL 正则化（编码器后验与状态先验对齐），第二项是动作重构对数似然。

### 在线伪标签机制

对每个离线轨迹片段 $\tau_{[H]}$，构造高层转移元组：

$$(s_0, \quad \hat{z} \sim f_\theta(z|\tau), \quad \hat{r} = r_{\mathrm{UCB}}(s_0, \hat{z}), \quad s_H)$$

- **高层动作标签 $\hat{z}$**：通过 VAE 编码器推断，在线学习开始前一次性计算
- **乐观奖励标签 $\hat{r}$**：使用 UCB 估计，每次更新前重新计算

**UCB 奖励估计：**

$$r_{\mathrm{UCB}}(s, z) = r_\zeta(s, z) + \alpha \|g_\phi(s, z) - \bar{g}(s, z)\|_2^2$$

其中 $r_\zeta$ 是从在线数据学习的奖励预测模型，后一项是 RND（Random Network Distillation）探索奖励，$\alpha$ 控制乐观程度。

### 实现细节

- **高层策略**：SAC（Soft Actor-Critic）+ 10 个 critic 集成，使用 RLPD 框架
- **策略参数化**：采用 tanh 变换 + 熵正则（替代 SPiRL 的 KL 约束，效果更好更稳定）
- **采样策略**：在线和离线各采 128 样本平衡训练
- **技能空间**：潜变量维度 10，轨迹片段长度 $H=4$
- **UTD Ratio**：状态域 20，视觉域 40

## 实验关键数据

### 实验设置

在 **8 个领域、42 个稀疏奖励任务**上评估：

| 领域类别 | 环境 | 任务数 | 特点 |
|---------|------|-------|------|
| 状态导航 | antmaze (medium/large/ultra × 4 goals) | 12 | D4RL 标准基准 |
| 状态导航 | humanoidmaze (medium/large/giant × navigate/stitch) | 6 | 高维动作空间（21 vs. 8） |
| 状态导航 | antsoccer (arena/medium) | 2 | 需控球到目标位置 |
| 状态操作 | kitchen (mixed/partial/complete) | 3 | 多步组合操作 |
| 状态操作 | cube-single (5 tasks) | 5 | 抓放排列 |
| 状态操作 | cube-double (5 tasks) | 5 | 双方块操作 |
| 状态操作 | scene (5 tasks) | 5 | 多物体交互（抽屉/窗/锁/方块） |
| 视觉导航 | visual-antmaze (4 goals) | 4 | 64×64 图像观测 |

### 基线方法

- **ExPLORe**：不预训练技能，直接用 RND+UCB 标注离线数据做在线 RL
- **DBC+JSRL**：扩散策略模仿离线数据 → JSRL 式初始化在线探索
- **Trajectory Skills**：VAE 技能预训练 → 丢弃离线数据 → 纯在线学高层策略
- **HILP Skills**：HILP 技能预训练 → 丢弃离线数据 → 纯在线学高层策略
- **SUPE (HILP)**：HILP 技能 + 离线伪标签（本文提出的 HILP 变体）

### 主要结果

**聚合性能（Figure 3）：** SUPE 在 8 个领域中的 6 个上最优，SUPE(HILP) 在 antsoccer 和 scene 上更好。

**关键发现：**

1. **双重利用离线数据至关重要**：SUPE > Trajectory Skills，SUPE(HILP) > HILP Skills，表明在线阶段使用离线数据显著加速学习
2. **技能预训练不可或缺**：SUPE > ExPLORe，尤其在难度大的环境中 ExPLORe 完全失败
3. **HumanoidMaze 上绝对优势**：SUPE 是唯一在 large 和 giant 迷宫上获得非零回报的方法
4. **探索效率（antmaze first-goal-time）**：SUPE 在所有布局上最快到达目标，证实了更好的探索策略

**超参敏感性：**
- RND 系数 $\alpha$：在 {2, 8, 16} 范围内性能稳定，$\alpha=0$ 时大幅下降
- 技能长度 $H=4$ 全局最优；$H=2$ 或 $H=8$ 在多数任务上表现更差

### 数据鲁棒性（Appendix K）

在 antmaze-large 上测试不同质量离线数据：
- **Navigate/Stitch 数据**：SUPE 保持优势，尤其在更难的 Stitch 上
- **纯随机 Explore 数据**：所有技能方法失败（技能质量太差）
- **数据不充分（5% 数据/去除目标附近数据）**：SUPE 学习变慢但渐近性能不变，仍优于基线

## 亮点与洞察

1. **简洁而深刻的核心思想**："用两次"离线数据——这个想法直觉上合理但此前被忽视，作者通过精心设计的伪标签机制有效实现了这一点
2. **UCB 奖励伪标签**：将 ExPLORe 的 state-action 级别乐观估计提升到高层 state-skill 级别，与技能抽象的时间尺度匹配
3. **去掉 KL 约束**：用 tanh + 熵正则替代 SPiRL 的 KL 惩罚，更简单且性能更好——是工程层面的重要简化
4. **实验规模大且系统**：42 个任务、8 个领域、多种基线、敏感性分析、数据质量分析，极为充分

## 局限与展望

1. **技能冻结问题**：预训练技能在在线阶段保持固定，当技能学得不好或需要适应在线分布变化时会受限。允许低层技能在线微调是自然的改进方向
2. **依赖 RND**：UCB 估计基于 RND，在高维观测空间中可能不够可靠。虽然在 visual-antmaze 上不需要 ICVF 就能工作，但更复杂的视觉域可能需要更稳健的探索信号
3. **技能长度固定**：$H=4$ 全局使用，但不同环境可能需要不同的时间抽象粒度。变长技能（options framework）可能更灵活
4. **纯随机离线数据不可用**：当离线数据质量极差（Explore 数据集）时所有技能方法都失败，方法对数据质量有下限要求
5. **计算成本**：总计约 16600 GPU 小时（A5000/V100），实验规模的可复现性门槛较高

## 相关工作与启发

- **SPiRL / OPAL**（Pertsch et al. 2021; Ajay et al. 2021）：SUPE 的直接前身，VAE 技能预训练框架基本相同，但只在离线阶段使用数据
- **ExPLORe**（Li et al. 2024）：提出 UCB 伪标签在线使用离线数据，但不预训练技能。SUPE 将其提升到高层技能空间
- **HILP**（Park et al. 2024b）：替代的技能发现方法，通过 Hilbert 表示学习方向性技能。SUPE(HILP)表明框架对技能类型通用
- **RLPD**（Ball et al. 2023）：SUPE 在线 RL 的底层算法，高效的 off-policy 框架

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 核心思想（双重利用离线数据）简洁且有效，伪标签从 action 空间到 skill 空间的提升是合理的创新
- **实验充分度**: ⭐⭐⭐⭐⭐ — 42 个任务、8 个领域、系统的消融和敏感性分析，附录极为详尽
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰、方法描述完整、图表设计良好
- **价值**: ⭐⭐⭐⭐ — 为利用无标签先验数据加速在线 RL 探索提供了实用且有效的方案，在 HRL+探索领域具有参考意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Online Pre-Training for Offline-to-Online Reinforcement Learning](online_pre-training_for_offline-to-online_reinforcement_learning.md)
- [\[ICML 2025\] KEA: Keeping Exploration Alive by Proactively Coordinating Exploration Strategies](kea_keeping_exploration_alive_by_proactively_coordinating_exploration_strategies.md)
- [\[CVPR 2025\] SkillMimic: Learning Basketball Interaction Skills from Demonstrations](../../CVPR2025/reinforcement_learning/skillmimic_learning_basketball_interaction_skills_from_demonstrations.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](../../ICLR2026/reinforcement_learning/sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning](../../ICLR2026/reinforcement_learning/one_model_for_all_tasks_leveraging_efficient_world_models_in_multi-task_planning.md)

</div>

<!-- RELATED:END -->
