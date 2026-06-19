---
title: >-
  [论文解读] Learning Parameterized Skills from Demonstrations
description: >-
  [NeurIPS 2025][机器人][参数化技能] 提出 DEPS，一种端到端从专家示范中发现参数化技能的算法，通过三层层次策略（离散技能选择→连续参数选择→底层动作）和信息瓶颈设计，学习可解释且可泛化的技能抽象，在LIBERO和MetaWorld上显著优于基线。 领域现状： 标准RL应用于长时域序列决策问题时未能利用内在…
tags:
  - "NeurIPS 2025"
  - "机器人"
  - "参数化技能"
  - "示范学习"
  - "层次策略"
  - "变分推断"
  - "机器人操作"
---

# Learning Parameterized Skills from Demonstrations

**会议**: NeurIPS 2025  
**arXiv**: [2510.24095](https://arxiv.org/abs/2510.24095)  
**代码**: [GitHub](https://github.com/guptbot/DEPS)  
**领域**: Optimization (机器人学习/技能发现)  
**关键词**: 参数化技能, 示范学习, 层次策略, 变分推断, 机器人操作

## 一句话总结

提出 DEPS，一种端到端从专家示范中发现参数化技能的算法，通过三层层次策略（离散技能选择→连续参数选择→底层动作）和信息瓶颈设计，学习可解释且可泛化的技能抽象，在LIBERO和MetaWorld上显著优于基线。

## 研究背景与动机

**领域现状**: 标准RL应用于长时域序列决策问题时未能利用内在行为模式，导致样本效率低。Options框架旨在发现模块化、时间延展的技能，但已有方法要么学习纯离散技能要么学习纯连续技能。

**现有痛点**: (i) 离散技能缺乏灵活性，难以泛化到新情境；(ii) 连续技能结构性差、难以解释；(iii) 现有参数化技能方法（如da Silva等）需要标注的任务参数，或依赖VLMs预训练聚类（如LOTUS、EXTRACT），假设相同技能发生在视觉相似的环境中；(iv) 潜变量模型易出现退化——底层策略直接记忆行为，不学习有意义的技能抽象。

**核心矛盾**: 如何在不依赖额外标注或预定义技能库的前提下，端到端地从示范中发现既有离散结构又可连续调制的参数化技能？

**本文目标**: 从多任务专家示范中自动发现参数化技能，使学到的技能能快速泛化到未见任务。

**切入角度**: 将技能建模为参数化轨迹流形，通过极端的状态压缩（压缩到1维）迫使潜变量编码有意义的技能信息。

**核心 idea**: 通过将观测空间压缩到1维"索引"来创造信息不对称，迫使离散技能和连续参数承载关键的任务语义信息。

## 方法详解

### 整体框架

DEPS 训练一个三层层次结构：
- **层级1 - 离散技能策略** $\pi^K(k_t | s_{1:t}, a_{1:t-1}, l)$：从技能库中选择技能
- **层级2 - 连续参数策略** $\pi^Z(z_t | s_{1:t}, a_{1:t-1}, k_t, l)$：为选定技能输出连续参数
- **层级3 - 底层子策略** $\pi^A(a_t | s'_t, k_t, z_t)$：基于压缩状态和参数化技能生成动作

### 关键设计

1. **时序变分推断 (Temporal Variational Inference)**:

    - **功能**: 通过变分下界最大化示范轨迹的对数似然
    - **为什么**: 直接计算 $\log p(\tau, l)$ 需要对所有可能的技能序列进行边际化，不可行
    - **怎么做**: 引入变分分布 $q(\kappa, \zeta | \tau, l)$，利用KL散度非负性得到ELBO：
    $\mathcal{L} = \mathbb{E}[\sum_t \log\pi^A(a_t|s'_t, k_t, z_t)] - \mathbb{E}[\sum_t D_{KL}(q(k_t|\tau,l) \| \pi^K(\cdot)) + \mathbb{E}_{k_t}[D_{KL}(q(z_t|\tau,k_t,l) \| \pi^Z(\cdot))]]$
    - **区别**: 与Shankar\&Gupta的方法不同，DEPS同时处理离散和连续潜变量，且支持高维状态空间

2. **投影状态压缩到一维 (Projective State Compression)**:

    - **功能**: 将底层子策略的输入状态压缩到标量
    - **为什么**: (a) 增加跨任务状态空间重叠，提升泛化；(b) 压缩后的状态不足以独立决定动作，迫使子策略依赖 $(k_t, z_t)$ 编码关键信息
    - **怎么做**: $s'_t = \tanh(\mathbf{w}_{(k_t, z_t)} \cdot s_t^{\text{proj}} + b_{(k_t, z_t)})$，其中投影向量 $\mathbf{w}$ 和偏置 $b$ 由技能条件MLP生成。tanh归一化到 $[-1, 1]$
    - **区别**: 极端的1维压缩是本文最大创新，将技能概念化为"参数化轨迹流形"上的索引

3. **信息不对称设计**:

    - **功能**: 高层策略接收完整观测（图像+本体感觉），底层子策略只接收压缩的本体感觉状态
    - **为什么**: 防止子策略过拟合视觉细节，强制通过技能变量传递任务信息
    - **补充约束**: 连续参数按离散技能粒度预测（而非每步），防止连续参数退化为编码每步动作的快捷方式；引入技能参数范数惩罚防止过拟合

### 损失函数 / 训练策略

- 最大化变分下界（ELBO），包含三项：行为克隆项 + 离散KL项 + 连续KL项
- 变分网络用双向GRU，离散/连续策略用单向GRU
- 预训练后在未见任务上微调500步评估泛化
- LIBERO: 80个任务预训练、20 epochs；MetaWorld: 10个任务、40 epochs

## 实验关键数据

### 主实验

**跨评估设置的平均成功率 (LIBERO + MetaWorld):**

| 评估集 | 算法 | Mean Success | Mean Highest Success |
|--------|------|-------------|---------------------|
| LIBERO-OOD | **DEPS** | **0.34±0.08** | **0.66±0.12** |
| | PRISE | 0.10±0.09 | 0.27±0.23 |
| | BC | 0.15±0.04 | 0.36±0.08 |
| LIBERO-3-shot | **DEPS** | **0.26±0.03** | **0.49±0.03** |
| | PRISE | 0.07±0.07 | 0.19±0.14 |
| | BC | 0.11±0.05 | 0.22±0.08 |
| MW-Vanilla | **DEPS** | **0.45±0.03** | **0.65±0.03** |
| | PRISE | 0.21±0.07 | 0.33±0.10 |
| | BC | 0.35±0.02 | 0.51±0.01 |

### 消融实验

**预训练数据量鲁棒性 (LIBERO-OOD):**

| 预训练轮数 | DEPS Mean Highest | BC Mean Highest | PRISE Mean Highest |
|-----------|-------------------|-----------------|--------------------|
| 5 epochs | 0.64±0.09 | 0.30±0.08 | 0.11±0.13 |
| 10 epochs | 0.75±0.01 | 0.30±0.12 | 0.27±0.33 |
| 15 epochs | 0.74±0.04 | 0.32±0.07 | 0.33±0.26 |

**关键消融结论:**
- 1D状态压缩对DEPS性能至关重要
- 仅学习离散技能或仅学习连续技能都无法复现DEPS的性能
- 改变最大离散技能数量可以进一步提升性能

### 关键发现

- DEPS在LIBERO-OOD上的Mean Success是BC的2倍+、PRISE的3倍+
- 在极端数据稀缺场景（3-shot）下DEPS仍保持强劲性能（0.26 vs BC 0.11）
- 预训练量越少DEPS的优势越大，说明参数化技能学习也提升了数据效率
- 学到的离散技能可解释：对应抓取、移动、释放等基本操作
- 连续参数的变化导致策略的平滑变化（如抓取位置的连续变化）
- 压缩后的1D状态在单个技能内单调变化，确认其作为"轨迹索引"的功能

## 亮点与洞察

- **技能作为参数化轨迹流形**: 概念新颖且直观——同一技能的不同执行对应流形上的不同点，1维索引足以确定轨迹位置
- **极端压缩的反直觉有效性**: 将状态压缩到单个标量竟然有效，且是性能的关键驱动因素
- **信息不对称的精心设计**: 高层丰富观测 + 底层极度压缩，迫使层次结构各层各司其职
- **可解释性**: 学到的技能对应直觉上合理的行为原语，增强了方法的可信度

## 局限与展望

- 仅在机器人操作任务上验证，更复杂的环境（导航、双手操作）未测试
- 假设所有任务共享状态空间和动作空间
- 离散技能数量需要作为超参数设定
- 1维压缩可能在需要更丰富状态信息的任务上失效
- 仅从离线示范学习，未与在线RL结合
- 计算效率分析（训练时间、推理延迟）缺失
- 扩展到更大规模任务集和更长时域问题的能力待验证

## 相关工作与启发

- 与Options框架的经典工作（Sutton et al.）一脉相承，但实现了端到端的参数化技能发现
- 对比PRISE（基于VLMs的动作token化）展示了端到端方法的优越性
- 信息瓶颈/状态压缩思想可推广到其他层次化学习场景
- 启发了将变分推断与信息论正则化结合以避免退化的通用策略

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 参数化轨迹流形概念和1维压缩设计极具创新性
- 实验充分度: ⭐⭐⭐⭐ 多基准多设置评估全面，定性可视化优秀，但缺少更复杂环境
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，概念阐述清晰，图示直观
- 价值: ⭐⭐⭐⭐ 对机器人技能学习和层次化策略学习领域有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [\[CVPR 2026\] Beyond Mimicry: Learning Whole-Body Human-Humanoid Interaction from Human-Human Demonstrations](../../CVPR2026/robotics/beyond_mimicry_learning_whole-body_human-humanoid_interaction_from_human-human_d.md)
- [\[CVPR 2026\] RoboWheel: A Data Engine from Real-World Human Demonstrations for Cross-Embodiment Robotic Learning](../../CVPR2026/robotics/robowheel_a_data_engine_from_real-world_human_demonstrations_for_cross-embodimen.md)
- [\[CVPR 2026\] SPEAR-1: Scaling Beyond Robot Demonstrations via 3D Understanding](../../CVPR2026/robotics/spear-1_scaling_beyond_robot_demonstrations_via_3d_understanding.md)
- [\[ICLR 2026\] Abstracting Robot Manipulation Skills via Mixture-of-Experts Diffusion Policies](../../ICLR2026/robotics/abstracting_robot_manipulation_skills_via_mixture-of-experts_diffusion_policies.md)

</div>

<!-- RELATED:END -->
