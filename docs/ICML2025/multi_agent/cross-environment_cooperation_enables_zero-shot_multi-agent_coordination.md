---
title: >-
  [论文解读] Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination
description: >-
  [ICML 2025][多智能体][Zero-shot Coordination] 提出跨环境合作（CEC）范式，通过在程序化生成的大量多样化环境中进行自对弈训练（而非增加伙伴多样性），使智能体学习到通用的合作规范，从而在从未见过的新环境中与从未见过的新伙伴实现零样本协调。 问题定义 零样本协调（Zero-Shot Coor…
tags:
  - "ICML 2025"
  - "多智能体"
  - "Zero-shot Coordination"
  - "Cross-Environment Cooperation"
  - "Procedural Generation"
  - "Multi-agent RL"
  - "Human-AI Collaboration"
---

# Cross-environment Cooperation Enables Zero-shot Multi-agent Coordination

**会议**: ICML 2025  
**arXiv**: [2504.12714](https://arxiv.org/abs/2504.12714)  
**代码**: [https://kjha02.github.io/publication/cross-env-coop](https://kjha02.github.io/publication/cross-env-coop)  
**领域**: 强化学习 / 多智能体协作 / 零样本协调  
**关键词**: Zero-shot Coordination, Cross-Environment Cooperation, Procedural Generation, Multi-agent RL, Human-AI Collaboration  

## 一句话总结

> 提出跨环境合作（CEC）范式，通过在程序化生成的大量多样化环境中进行自对弈训练（而非增加伙伴多样性），使智能体学习到通用的合作规范，从而在从未见过的新环境中与从未见过的新伙伴实现零样本协调。

## 研究背景与动机

### 问题定义

零样本协调（Zero-Shot Coordination, ZSC）是构建人类兼容AI的关键能力——智能体需要在没有预先协商的情况下与新伙伴在新任务中即时合作。人类天然擅长这种即兴合作：一个厨师在家和父母学会了一道菜的做法，换个厨房和配偶也能顺利完成同样（甚至更多）的烹饪任务。

### 现有方法的局限

**自对弈（Self-Play, SP）**：在合作博弈中，一旦找到一个均衡策略，双方都没有动力去探索其他均衡，导致策略脆弱，无法适应采用不同策略的新伙伴。

**基于种群的训练（Population-Based Training, PBT）**：如 Fictitious Co-Play (FCP)，通过维护多样化的伙伴池来增加"伙伴多样性"。虽然能在单一环境中适应不同伙伴，但**完全无法泛化到哪怕是微小变化的新环境**。每次环境变化都需要重新训练整个种群，计算开销巨大且不可扩展。

**E3T（Efficient End-to-End Training）**：通过给伙伴策略加噪声并训练辅助网络预测他人行为，在单任务ZSC中达到SOTA，但同样局限于训练时见过的那个环境。

### 核心洞察

作者提出一个关键假设：**环境多样性 > 伙伴多样性**。与其在单一任务上训练多样化的伙伴策略，不如让智能体在大量多样化的环境中与同一个伙伴（自身的副本）进行自对弈。多样化的环境迫使智能体学习高层次的任务结构（"烹饪洋葱并送餐"），而不是低层次的动作序列（"向左移动三格然后交互"），从而自然地获得对新伙伴和新环境的泛化能力。

## 方法详解

### 整体框架

CEC（Cross-Environment Cooperation）的核心流程：

1. **程序化环境生成**：构建能产生海量可解协调任务的生成器
2. **跨环境自对弈训练**：在采样的多样化任务上，用 IPPO 进行自对弈
3. **（可选）单任务微调**：在特定目标环境上低学习率微调（CEC-FT）

### 形式化定义

两人合作马尔可夫博弈定义为 $\langle S, A, \mathcal{T}, R, H \rangle$，其中任务 $m \sim \mathcal{M}$ 定义初始状态分布 $p(s_0|m)$，但共享转移动态 $\mathcal{T}$ 和奖励函数 $R$。

**PBT目标函数**（单任务，多伙伴）：

$$J(\pi_C) = \mathbb{E}_{\pi_i \sim P}[S(\pi_i, \pi_C, m)]$$

**CEC目标函数**（多环境，自对弈）：

$$J(\pi_C) = \mathbb{E}_{m_i \sim \mathcal{M}}[S(\pi_C, \pi_C, m_i)]$$

其中合作得分定义为：

$$S(\pi_p, \pi_C, m) = \mathbb{E}_{\substack{s_0 \sim m, s \sim \mathcal{T} \\ a^p \sim \pi_p, a^C \sim \pi_C}} \left[\sum_{t=0}^{H} R(s_t, a_t^p, a_t^C)\right]$$

关键对比：PBT 在伙伴分布 $P$ 上求期望，CEC 在任务分布 $\mathcal{M}$ 上求期望。CEC 只需训练单一策略，无需维护伙伴种群。

### 程序化环境生成

#### Dual Destination（玩具环境）

- 两个智能体在网格中，需要分别到达不同的绿色目标格
- 到达对侧目标格 +3 奖励，每步 -1 惩罚
- CEC版本：随机化智能体起始位置和目标位置

#### Overcooked（主实验环境）

基于 JaxMARL 项目的 Overcooked 扩展，基于五种经典布局（Asymmetric Advantages、Coordination Ring、Counter Circuit、Cramped Room、Forced Coordination）的墙体结构：

1. 从五种布局中随机采样一种作为基础结构
2. 移除所有物品和智能体，只保留墙壁
3. 在可达墙壁上随机放置必需物品（盘子堆、洋葱堆、锅、目标位置）
4. 在剩余墙壁上随机放置额外物品
5. 随机采样智能体初始位置（分隔布局确保双方在隔板两侧）
6. 50% 概率旋转90°，嵌入 9×9×26 观测空间
7. 检查是否与评估关卡重复，重复则重新生成

该生成器可产生 $1.16 \times 10^{17}$ 种不同的可解厨房配置。整个流程基于 JAX 实现，单GPU训练速度达到 **1000万步/分钟**。

### 网络架构

| 组件 | 层 | 细节 |
|------|------|------|
| 观测编码器 | Conv1 | 2×2卷积核, 64滤波器, ReLU |
| | Conv2 | 2×2卷积核, 32滤波器, ReLU |
| | FC1 | 全连接, 512单元, ReLU |
| | FC2 | 全连接, 512单元, ReLU |
| 循环核心 | LSTM | 特征维度256, episode边界重置状态 |
| Actor头 | FC1-FC4 | 256→192→128→64, ReLU |
| | Output | 6个动作logits（Overcooked） |
| Critic头 | FC1-FC4 | 512→256→192→128, ReLU |
| | Output | 1个标量值预测 |

**循环网络的必要性**：实验表明去掉LSTM后，CEC在300M步训练中甚至无法获得正奖励，因为LSTM提供了基本的元学习能力，使智能体能够在episode内推断伙伴意图。

### PPO训练参数

- 学习率：$3 \times 10^{-4}$（退火）
- 总训练步数：$3 \times 10^9$
- GAE参数：$\gamma=0.99, \lambda=0.95$
- PPO裁剪：$\epsilon=0.2$
- 熵系数：0.005
- 梯度裁剪：0.5

### CEC-Finetune

训练完CEC通用策略后，针对5个评估布局分别创建5个副本，在每个布局上以降低的学习率继续自对弈训练 $10^8$ 步，获得 CEC-FT 模型。

## 实验关键数据

### 玩具环境实验（Dual Destination）

| 方法 | 固定任务XP | 程序生成任务XP |
|------|------------|----------------|
| IPPO (SP) | ~0.2 | ~0.05 |
| FCP | ~0.6 | ~0.15 |
| **CEC** | **~0.93** | **~0.97** |

- CEC 归一化奖励 0.931（标准误差0.013），仅比理想oracle低约2.5%
- 统计显著：CEC vs FCP 和 IPPO 均 $p < 0.001$（t检验）

### Overcooked AI-AI 评估

#### 5个经典布局（XP性能）

| 方法 | 平均XP奖励 |
|------|------------|
| IPPO | ~50 |
| FCP | ~80 |
| E3T | ~90 |
| **CEC** | **~130** |
| **CEC-FT** | **~155** |

- CEC-FT 在经典布局上显著优于 FCP 和 IPPO（$p < 0.01$）
- CEC（未见过这些布局）仍优于所有单任务基线

#### 100个程序生成布局（XP性能）

| 方法 | 平均XP奖励 |
|------|------------|
| IPPO | ~0 |
| FCP | 0 |
| E3T | 0 |
| **CEC** | **~70** |
| CEC-FT | ~42 |

- FCP 和 E3T 在新布局上获得**零奖励**，完全无法泛化
- CEC 显著超越所有基线（$p < 0.0001$）
- CEC-FT 泛化能力下降，体现通用性与专业化的权衡

### 跨算法协作分析

通过经验博弈论分析（Empirical Game-Theoretic Analysis），将跨算法合作分数作为元博弈的支付矩阵，计算复制器动态梯度。结果显示：在5个经典布局和100个程序生成布局上，梯度方向均指向CEC和CEC-FT，表明它们是可能的均衡策略。

### 人类实验结果

80名参与者在 Counter Circuit 和 Coordination Ring 两个布局上与各种AI合作。

#### 合作得分（定量）

| 方法 | 人类合作得分 |
|------|-------------|
| IPPO | ~2.0 |
| FCP | ~4.0 |
| **CEC** | **~7.5** |
| **CEC-FT** | **~8.0** |
| E3T | ~9.5 |

- CEC 显著优于 FCP（$p < 0.001$）
- CEC 接近 E3T 的性能，尽管从未见过评估布局

#### 主观评价（定性，7项指标）

CEC和CEC-FT在以下所有主观维度上获得最高用户评分：
- 适应性、一致性、愉悦度、协调性、低挫折感、合作能力、整体偏好
- CEC-FT 显著超越 E3T（$p < 0.01, t=3.1233$）
- Cronbach's alpha = 0.874，验证主观指标的内部一致性

#### 碰撞分析

CEC 与人类的平均碰撞次数最低，表明CEC学到了"避让"这种通用合作规范。尽管这可能降低即时奖励，但大幅提升了用户体验。

### 消融实验

| 消融项 | 关键发现 |
|--------|---------|
| 部分可观测 | 在3×3视窗下CEC(0.74) > PBT(0.61) > SP(0.03)，结论一致 |
| 多任务变体 | 4种有效策略时CEC(0.404) > PBT(0.251) > SP(0.083) |
| CEC+E3T | 结合伙伴多样性，新布局泛化优于CEC-FT但经典布局性能下降 |
| 去掉LSTM | 无法收敛，300M步内无法获得正奖励 |
| CEC-FT | 提升特定布局性能但损失泛化能力 |

### 行为模式分析

通过可视化 Counter Circuit 布局上的格子访问频率：
- **IPPO**：访问分布高度集中，固定路线（顺时针或逆时针），策略脆弱
- **CEC**：访问分布更均匀，高频区域集中在任务相关物品（锅、洋葱堆、盘子堆、目标位置），说明CEC学到了任务结构的丰富表征

## 亮点与洞察

1. **颠覆性发现**：自对弈在合作博弈中"不够用"的传统观点被挑战——关键不在于增加伙伴多样性，而在于增加环境多样性。CEC用自对弈训练却超越了PBT在零样本协调上的表现。

2. **通用合作规范的涌现**：跨环境训练不仅提升了环境泛化能力，还意外地提升了伙伴泛化能力。智能体学到了如"避让他人"、"关注任务相关物品"等通用合作规范。

3. **计算效率优势**：CEC只需训练单一策略，PBT则需要为每个任务训练整个种群。在相同计算预算下，CEC将计算投资于环境多样性而非伙伴多样性，效果更好。

4. **量化与定性评价的分离**：E3T在奖励分数上更高，但CEC在人类主观评价上更优。这提示我们**奖励最大化不等于好的合作**——过度贪婪的策略可能迫使人类适应AI而非AI适应人类。

5. **JAX加速的工程贡献**：提供了基于JAX的程序化Overcooked环境生成器，单GPU达到10M步/分钟，为大规模多智能体协作研究提供了可扩展的基础设施。

## 局限性

1. **训练尚未收敛**：3B步训练后CEC的学习曲线尚未饱和，受限于计算资源未能探索其性能上限。
2. **环境复杂度有限**：Overcooked虽然是标准基准，但与真实世界合作场景（如家庭机器人）相比仍较简单。
3. **人类实验偏差**：参与者限制为英语流利者，可能引入文化偏差影响合作行为和评价。
4. **CEC+PBT结合效果受限**：CEC与E3T结合后训练效率下降，可能需要更大网络和更长训练时间。
5. **泛化与专业化的权衡**：CEC-FT提升特定布局性能的同时损失了对新布局的泛化能力，如何平衡二者仍是开放问题。

## 相关工作

- **自对弈方法**：AlphaStar (Vinyals et al., 2019)、AlphaGo (Silver et al., 2017) 在零和博弈中取得巨大成功，但在合作博弈中因多均衡问题而失效
- **PBT方法**：FCP (Strouse et al., 2022)、MEP (Zhao et al., 2022)、E3T (Yan et al., 2023) 通过伙伴多样性提升ZSC，但局限于单一环境
- **程序化环境生成**：Procgen (Cobbe et al., 2020)、MAESTRO (Samvelyan et al., 2023) 在单智能体或零和设定中展示了环境多样性的价值
- **环境多样性与合作**：McKee et al. (2022) 发现环境多样性提升合作泛化，但未与人类测试、未比较环境vs伙伴多样性的相对效果
- **UED方法**：Ruhdorfer et al. (2024) 在Overcooked中使用无监督环境设计，但未保证生成任务的可解性，泛化效果不佳

## 评分

| 维度 | 评分 |
|------|------|
| 新颖性 | ⭐⭐⭐⭐ |
| 技术深度 | ⭐⭐⭐⭐ |
| 实验充分度 | ⭐⭐⭐⭐⭐ |
| 写作质量 | ⭐⭐⭐⭐ |
| 实用价值 | ⭐⭐⭐⭐ |
| 综合评分 | ⭐⭐⭐⭐ |

> 本文提出的CEC范式简洁而有效，通过环境多样性替代伙伴多样性来实现零样本协调是一个优雅且具有启发性的思路。实验覆盖了从玩具环境到大规模Overcooked、从AI-AI到Human-AI的完整评估链条，人类实验的设计尤为扎实。主要遗憾在于环境复杂度仍有限，且3B步训练未收敛这一限制使得CEC的真实潜力尚未充分展现。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] AgentDet: A Shared-Blackboard Multi-Agent Framework for Zero-/Few-Shot Object Detection](../../CVPR2026/multi_agent/agentdet_a_shared-blackboard_multi-agent_framework_for_zero-few-shot_object_dete.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](../../ACL2026/multi_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[NeurIPS 2025\] MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision](../../NeurIPS2025/multi_agent/maszero_designing_multiagent_systems_with_zero_supervision.md)
- [\[ACL 2025\] Multi-Agent Collaboration via Cross-Team Orchestration](../../ACL2025/multi_agent/multi-agent_collaboration_via_cross-team_orchestration.md)
- [\[AAAI 2026\] Learning to Generate and Extract: A Multi-Agent Collaboration Framework for Zero-shot Document-level Event Arguments Extraction](../../AAAI2026/multi_agent/learning_to_generate_and_extract_a_multi-agent_collaboration_framework_for_zero-.md)

</div>

<!-- RELATED:END -->
