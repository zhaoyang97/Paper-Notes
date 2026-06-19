---
title: >-
  [论文解读] Learning from Demonstrations via Capability-Aware Goal Sampling
description: >-
  [NeurIPS 2025][强化学习][模仿学习] 提出Cago方法，通过动态追踪智能体在专家演示轨迹上的达成能力，自适应采样处于能力边界的中间目标，构建隐式课程引导长视野稀疏奖励任务学习。 领域现状：模仿学习通过专家演示训练智能体，方法包括行为克隆（BC）、GAIL、逆强化学习等，但长视野复杂任务中仍存在严重挑战…
tags:
  - "NeurIPS 2025"
  - "强化学习"
  - "模仿学习"
  - "课程学习"
  - "目标条件强化学习"
  - "能力感知"
  - "世界模型"
---

# Learning from Demonstrations via Capability-Aware Goal Sampling

**会议**: NeurIPS 2025  
**arXiv**: [2601.08731](https://arxiv.org/abs/2601.08731)  
**代码**: [GitHub](https://github.com/RU-Automated-Reasoning-Group/Cago)  
**领域**: 强化学习  
**关键词**: 模仿学习, 课程学习, 目标条件强化学习, 能力感知, 世界模型

## 一句话总结

提出Cago方法，通过动态追踪智能体在专家演示轨迹上的达成能力，自适应采样处于能力边界的中间目标，构建隐式课程引导长视野稀疏奖励任务学习。

## 研究背景与动机

**领域现状**：模仿学习通过专家演示训练智能体，方法包括行为克隆（BC）、GAIL、逆强化学习等，但长视野复杂任务中仍存在严重挑战。

**现有痛点**：
   - BC存在复合误差问题
   - 分布匹配方法（GAIL等）在早期训练阶段进行"平坦匹配"，无法区分已掌握和未掌握的部分
   - 反向课程方法需要将智能体重置到演示的任意状态，在真实世界中不现实（关节速度等难以精确复现）

**核心矛盾**：现有方法未考虑智能体能力的动态演化——不知道哪些部分已掌握、哪些仍具挑战。

**本文目标**：在不需要任意状态重置的前提下，构建与智能体能力匹配的自适应学习课程。

**切入角度**：将演示视为结构化路线图而非直接模仿对象，持续监控智能体的能力上限来选择中间目标。

**核心 idea**：通过观测访问频率追踪能力边界，采样刚好超出当前能力的目标引导Go-Explore式探索。

## 方法详解

### 整体框架

三步闭环：(1) **观测访问追踪**——记录智能体在演示轨迹各位置的访问频率；(2) **能力感知目标采样**——在能力边界附近采样中间目标；(3) **Go-Explore式训练**——目标条件策略先到达目标，BC Explorer继续探索，收集的数据训练World Model和策略。

### 关键设计

1. **观测访问追踪（Observation Visit Tracking）**：

    - 功能：维护字典 $\text{Dict}_{visit}$ 记录智能体在每条演示轨迹各步骤的访问频率
    - 为什么：访问频率直接反映智能体到达对应状态的能力
    - 怎么做：每个环境步骤更新：$\text{Dict}_{visit}[\tau^{(i)}][j] += 1$ 当 $\text{sim}(s_t, s_j^{(i)}) \leq \epsilon$
    - 支持L2距离（状态空间）和MSE（视觉环境）
    - 区别：仅重置到演示初始状态，不需要重置到任意中间状态

2. **能力感知目标采样（Capability-Aware Goal Sampling）**：

    - 功能：在能力边界附近采样合适难度的目标
    - 为什么：太简单的目标无学习信号，太难的目标导致发散
    - 怎么做：
        - 找能力上限：$j^* = \max\{j | \text{Dict}_{visit}[\tau^{(i)}][j] \geq \lambda_{visit}\}$
        - 采样范围：$\mathcal{G}_{cap}(\pi^G, \tau^{(i)}) = \{s_k \in \tau^{(i)} | |k - j^*| \leq \delta \cdot L_i\}$
    - $\lambda_{visit}$：频率阈值（如100），$\delta$：窗口大小（如10%轨迹长度）
    - 区别：与JSRL的均匀课程不同，本方法真正感知智能体能力

3. **Go-Explore式数据收集**：

    - 功能：每个episode分为Go阶段和Explore阶段
    - 为什么：双阶段确保数据既靠近演示分布又有探索性
    - 怎么做：
        - Go阶段：目标条件策略 $\pi^G(\cdot|s, g)$ 尝试到达采样目标
        - Explore阶段：BC Explorer $\pi^E$（行为克隆策略）从到达点继续探索
    - 区别：BC Explorer提供高质量探索，优于随机探索

4. **World Model + 策略训练**：

    - 功能：用模型想象轨迹训练目标条件策略
    - 为什么：Go-Explore收集的数据靠近演示分布，学到的World Model在这些区域更准确
    - 怎么做：基于Dreamer框架，用时间距离函数 $D_t(s,g)$ 作为奖励 $r^G(s,g) = -D_t(s,g)$
    - 理论保证：Theorem 1证明BC Explorer有效降低模型预测误差上界

5. **目标预测器（Goal Predictor）**：

    - 功能：测试时从当前观测推断最终目标
    - 为什么：测试时无法访问演示轨迹
    - 怎么做：$\mathcal{P}_\phi: s \mapsto \hat{g}$，最小化MSE $\|\mathcal{P}_\phi(s_t^{(i)}) - s_L^{(i)}\|_2^2$
    - 最终策略：$\pi(s) = \pi^G(s, \mathcal{P}(s))$

### 损失函数 / 训练策略

- World Model：Dreamer框架的监督学习损失
- 策略：Actor-Critic + 时间距离奖励
- Goal Predictor：MSE回归损失
- BC Explorer：行为克隆损失
- 每个任务仅用10-20条演示

## 实验关键数据

### 主实验

**MetaWorld Very Hard 任务（Success Rate %，8种子平均）**：

| 方法 | Disassemble | PickPlaceWall | ShelfPlace | StickPull | StickPush |
|------|-------------|---------------|------------|-----------|-----------|
| Dreamer | ~10% | ~5% | ~10% | ~5% | ~15% |
| JSRL | ~25% | ~20% | ~30% | ~20% | ~30% |
| MoDem | ~40% | ~35% | ~40% | ~30% | ~45% |
| Cal-QL | ~15% | ~10% | ~15% | ~10% | ~20% |
| **Cago** | **~70%** | **~60%** | **~65%** | **~55%** | **~70%** |

**Adroit 灵巧手任务（1M步后Success Rate）**：

| 方法 | Door | Hammer | Pen |
|------|------|--------|-----|
| MoDem | ~60% | ~70% | ~55% |
| **Cago** | **~80%** | **~85%** | **~75%** |

**ManiSkill 难任务**：Cago是唯一能在有限演示下达到高成功率的方法。

### 消融实验

**各组件重要性（Disassemble/StickPush/Pen，5种子）**：

| 变体 | 描述 | 效果 |
|------|------|------|
| Cago (完整) | 能力感知采样 + BC Explorer | 最佳 |
| Cago-FinalGoal | 仅BC Explorer，总是选最终目标 | 显著下降 |
| Cago-StepBased | 按训练步数比例采样目标 | 下降 |
| Cago-NoExplorer | 仅能力感知采样，无BC Explorer | 明显下降 |
| Cago-RandomExplorer | 随机探索替代BC Explorer | 下降 |

### 关键发现

- 能力感知目标采样是核心贡献：去除后性能显著恶化
- 目标采样的归一化位置随训练自然从0→1递进，验证了自适应课程的有效性
- BC Explorer对数据质量至关重要，随机探索效果差
- 仅用10条演示即可有效工作
- 视觉输入version（Cago-Visual）保持相近性能，泛化能力强

## 亮点与洞察

- **"能力感知"的核心洞察**：不同于现有方法假设固定课程或全局匹配，Cago真正追踪了智能体的动态学习状态
- **仅重置到初始状态**：比反向课程方法实际得多，不需要复现关节速度等隐变量
- **理论+实验双重验证**：Theorem 1提供误差界理论保证，实验覆盖三大benchmark
- **Go-Explore范式的目标条件化扩展**：将经典探索策略与演示引导优雅结合

## 局限与展望

- 依赖重置到演示初始状态（比任意状态重置弱得多，但仍有限制）
- 相似度度量 $\text{sim}(\cdot,\cdot)$ 和阈值 $\epsilon$ 对不同任务可能需要调整
- 目标预测器在分布外场景的泛化能力有待验证
- 可探索结合LLM/VLM作为目标预测器处理更抽象任务

## 相关工作与启发

- 与JSRL的课程对比：JSRL使用预定义课程而非能力感知
- 与MoDem对比：MoDem通过过采样演示快速收敛但最终性能有限
- Go-Explore范式在Cago中被目标条件化和演示引导重新诠释
- Dreamer世界模型为能力感知采样提供了想象训练的基础设施

## 评分

- 新颖性: ⭐⭐⭐⭐ 能力感知目标采样思路直观且有效
- 实验充分度: ⭐⭐⭐⭐⭐ 三大benchmark、11个任务、全面消融+视觉扩展
- 写作质量: ⭐⭐⭐⭐ 动机清晰，方法阐述流畅，理论分析完整
- 价值: ⭐⭐⭐⭐ 对长视野稀疏奖励任务有显著提升，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Variance-Aware Feel-Good Thompson Sampling for Contextual Bandits](variance-aware_feel-good_thompson_sampling_for_contextual_bandits.md)
- [\[NeurIPS 2025\] Distribution Learning Meets Graph Structure Sampling](distribution_learning_meets_graph_structure_sampling.md)
- [\[ICLR 2026\] RuleReasoner: Reinforced Rule-based Reasoning via Domain-aware Dynamic Sampling](../../ICLR2026/reinforcement_learning/rulereasoner_reinforced_rule-based_reasoning_via_domain-aware_dynamic_sampling.md)
- [\[CVPR 2025\] SkillMimic: Learning Basketball Interaction Skills from Demonstrations](../../CVPR2025/reinforcement_learning/skillmimic_learning_basketball_interaction_skills_from_demonstrations.md)
- [\[ICML 2025\] Learning Utilities from Demonstrations in Markov Decision Processes](../../ICML2025/reinforcement_learning/learning_utilities_from_demonstrations_in_markov_decision_processes.md)

</div>

<!-- RELATED:END -->
