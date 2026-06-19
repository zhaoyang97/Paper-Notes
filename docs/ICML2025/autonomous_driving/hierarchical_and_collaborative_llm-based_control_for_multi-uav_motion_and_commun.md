---
title: >-
  [论文解读] Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks
description: >-
  [ICML 2025 (Workshop on ML4Wireless)][自动驾驶][多无人机控制] 提出一种基于 LLM 的层次化协作控制框架，通过 HAPS 端部署的元控制器 LLM 和 UAV 端部署的边缘控制器 LLM 的双层协同，实现多 UAV 在 3D 空中高速公路场景下的运动规划与通信接入联合优化。
tags:
  - "ICML 2025 (Workshop on ML4Wireless)"
  - "自动驾驶"
  - "多无人机控制"
  - "大语言模型"
  - "层次化协作"
  - "空地一体网络"
  - "运动与通信联合优化"
---

# Hierarchical and Collaborative LLM-Based Control for Multi-UAV Motion and Communication in Integrated Terrestrial and Non-Terrestrial Networks

**会议**: ICML 2025 (Workshop on ML4Wireless)  
**arXiv**: [2506.06532](https://arxiv.org/abs/2506.06532)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 多无人机控制, 大语言模型, 层次化协作, 空地一体网络, 运动与通信联合优化

## 一句话总结

提出一种基于 LLM 的层次化协作控制框架，通过 HAPS 端部署的元控制器 LLM 和 UAV 端部署的边缘控制器 LLM 的双层协同，实现多 UAV 在 3D 空中高速公路场景下的运动规划与通信接入联合优化。

## 研究背景与动机

多 UAV 系统在物流、通信覆盖等领域广泛应用，但其联合控制面临严峻挑战：

**运动动态复杂**：现有研究多只考虑运动方向，忽略了加速、减速、变道等精细运动动态，以及多 UAV 之间的碰撞避免问题

**通信与交通耦合**：UAV 提速可提高交通流量，但会导致频繁切换基站（handover），降低通信质量；两者目标天然矛盾

**异构网络复杂性**：UAV 需在地面基站（BS）与高空平台站（HAPS）构成的空地一体网络中进行接入选择，涉及水平切换（BS 间）和垂直切换（BS-HAPS 间）

**现有 RL 方法局限**：传统 DDQN、MORL 等方法需要大量 task-specific 训练，且难以同时处理运动和通信的多目标优化

作者认为 LLM 的预训练知识和 in-context learning 能力可以天然适配这类多目标决策问题，无需针对每个场景重新训练。

## 方法详解

### 整体框架

本文设计了一个**双层 LLM 协作架构**（LLM-LLM Dual Agents）：

- **HAPS 层（元控制器）**：部署在高空平台站上的 LLM，负责全局网络编排，管理 UAV 与 HAPS 的连接关系（接入/卸载决策）
- **UAV 层（边缘控制器）**：部署在每个 UAV 上的 LLM，负责实时运动控制（加速/减速/变道）和基站选择策略

两层控制器通过状态共享实现协同：HAPS 元控制器的卸载指令会约束 UAV 的垂直切换选项，UAV 的本地状态变化又会反馈影响 HAPS 的全局决策。

### 关键设计

#### 1. HAPS 元控制器 MDP

建模为 MDP $(S_{meta}, A_{meta}, P_{meta}, r_{meta})$：

- **状态**：当前 HAPS 负载、每个 UAV 的数据速率、地面 BS 覆盖情况
- **动作**：$\text{Offload}\{ID\}$（将 UAV 卸载到地面 BS）、$\text{Recall}\{ID\}$（将 UAV 重新接入 HAPS）、$\text{Idle}$（保持不变）
- **决策逻辑**：当 HAPS 总带宽使用 $B_t > C$ 时，选择链路质量最差的 UAV 卸载；当有空闲容量时，将卸载的 UAV 召回

元控制器奖励函数：

$$r_{meta} = \eta_1 \sum_{j} WR_t^{ij} - \eta_2 \cdot \text{Sat}_{HAPS} - \eta_3 \cdot \mu$$

其中第一项为总吞吐量，第二项惩罚 HAPS 饱和事件，第三项为切换惩罚。

#### 2. UAV 边缘控制器 MDP

每个 UAV 独立运行一个 MDP，动作空间受 HAPS 元控制器约束：

- **状态空间**（7 维/UAV）：3D 位置 $(x, y, z)$、前进速度 $v$、航向角 $\psi$、可用地面 BS 数 $n_R$、可用 HAPS 信道数 $n_H$
- **运动动作**（5 种）：左变道、保持车道、右变道、加速、减速
- **通信动作**（3 种）：$a_{tele1}$（最大加权数据速率 BS）、$a_{tele2}$（考虑负载均衡的 BS）、$a_{tele3}$（最大瞬时速率 BS）

#### 3. 基于 LLM 的 Prompt 工程

核心创新在于将整个优化问题编码为结构化自然语言 prompt，包含六个模块：

1. **任务描述**：声明使命目标
2. **任务目标**：具体优化指标（最大速度、避碰、最小切换）
3. **状态定义**：环境特征枚举
4. **观测值**：离散化后的状态矩阵
5. **经验记忆**：Top-5 最相似的 good/bad 经验（按状态空间欧氏距离选取）
6. **回复规则**：约束输出格式为一个运动动作 + 一个通信策略

通过 Ollama 框架在边缘服务器部署 LLM，保证实时推理延迟。

#### 4. 信道模型

- **G2A 信道**：采用 3GPP 天线模式规范，考虑方位角/仰角增益、LoS 概率、路径损耗
- **UAV-HAPS 信道**：LoS 链路 + 自由空间路径损耗 + Rician 衰落
- **加权数据速率**：$WR_t^{ij} = \frac{R_t^{ij}}{\min(Q_i, n_i)} (1 - \mu)$，其中 $\mu$ 为切换惩罚系数

### 损失函数 / 训练策略

UAV 边缘控制器的奖励由运输奖励和通信奖励两部分组成：

**运输奖励**：
$$r_t^{j, tran} = w_1 \frac{v_t^j - v_{min}}{v_{max} - v_{min}} - w_2 \delta_c - w_3 \chi_t^j$$

- $w_1$：速度归一化奖励
- $w_2 \delta_c$：碰撞惩罚（$\delta_c \in \{0,1\}$）
- $w_3 \chi_t^j$：变道频率惩罚

**通信奖励**：
$$r_t^{j, tele} = w_4 \cdot WR_{i^*, j, t} \cdot (1 - \min(1, \xi_t^j))$$

- $\xi_t^j$：累计切换概率
- 权重设计优先保障安全（$w_2$）和连通性（$w_4$）

推理过程不涉及梯度训练，完全依赖 LLM 的 in-context learning 能力。

## 实验关键数据

### 实验设置

| 参数 | 值 |
|------|------|
| UAV 数量 | 5（训练），5-40（评估） |
| 车道数 | 5 |
| 飞行速度范围 | 5-20 m/s |
| 载波频率 | 2.1 GHz（BS），2 GHz（HAPS） |
| BS 发射功率 | 40 dBm |
| 每 BS 最大用户数 | 3 |
| 每轮最大时间步 | 30 |
| 硬件 | 2× Intel E5-2650 v4 + 2× NVIDIA P100 |

### 主实验

| 方法 | 总奖励 | 通信奖励优势 | 碰撞率 | 说明 |
|------|--------|-------------|--------|------|
| DDQN | ~23 | 基线 | 较高 | 传统 DRL |
| Envelope-MORL | <20 | 中等 | 中等 | 多目标 RL |
| Llama 3.1 8B + DDQN | 中等 | 中等 | 中等 | 混合 LLM+RL |
| Llama 3.1 70B + DDQN | 较高 | 较高 | 较低 | 更大模型混合 |
| **LLM-LLM Dual Agent** | **~30** | **+2~3 units** | **<0.08** | **本文方法** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| UAV=5 → UAV=40 | 总奖励先降后升（U 型） | 低密度有编队效应，高密度拥堵主导 |
| 碰撞率 vs 密度 | <0.12（低密度），M>30 急升 | Dual Agent 始终低于 0.08 |
| BS 数量 5/10/15/20 | 网络可扩展性验证 | BS 增多提升通信性能 |
| 收敛速度 | Dual Agent ~1.5k episodes | 比 DDQN 快约一个数量级 |

### 关键发现

1. **收敛加速**：LLM-LLM 框架在约 1500 个 episode 内收敛，而 DDQN 需要更长时间且最终奖励更低
2. **Pareto 改进**：在全部评估的 UAV 密度下（5-40），本文方法在总奖励上平均提升 **16.3%**
3. **碰撞控制**：在高密度场景（>30 UAV）下，碰撞率比 DDQN 低 **21%**
4. **通信优势**：LLM 驱动的策略在 V2I 吞吐量上持续保持 2-3 单位优势，DDQN 在 >25 UAV 时因不协调切换导致性能急剧下降

## 亮点与洞察

1. **LLM 作为控制器的新范式**：直接将优化问题编码为 prompt，利用 LLM 的 in-context learning 代替传统 RL 训练，避免了针对特定任务的大量训练开销
2. **经验记忆的精巧设计**：仅选取状态空间中欧氏距离最近的 top-5 good/bad 经验作为上下文，在保持决策质量的同时大幅降低推理成本
3. **层次化解耦**：将全局网络管理（慢时间尺度）与局部运动控制（快时间尺度）分离到两个独立 LLM，降低了单个模型的决策复杂度
4. **通信-交通联合建模**：加权数据速率公式巧妙地将负载均衡和切换惩罚统一到一个指标中

## 局限与展望

1. **可扩展性存疑**：实验仅在 5-40 个 UAV 的规模下验证，大规模（数百 UAV）场景下的 LLM 推理延迟和 prompt 长度可能成为瓶颈
2. **LLM 推理成本**：虽然使用 Ollama 本地部署，但每个 UAV 每步都需调用一次 LLM 推理，实际部署中的算力和延迟需进一步评估
3. **缺乏与更先进 RL 的对比**：未与 PPO、SAC 等主流策略梯度方法比较
4. **仿真环境简化**：使用智能驾驶员模型（IDM）模拟运动，未考虑风场、GPS 精度等真实环境因素
5. **Workshop 论文**：作为 ICML Workshop 论文，实验深度和消融分析有限，缺乏详细的计算开销对比
6. **未来方向**：作者提到计划引入多模态感知输入、GPS 拒止环境下的定位、在线自适应策略

## 相关工作与启发

- **LLM for 6G**：与 Zhou et al. (2024, 2025) 的 LLM 功率控制和 prompt 工程工作互补，本文拓展到了多 LLM 协作场景
- **RL for UAV**：Cherif et al. (2022, 2023) 的 RL 轨迹规划是直接对标，本文用 LLM 替代了 RL 训练过程
- **Vehicle-to-Infrastructure**：Yan et al. (2025) 的 LLM+DDQN 混合方法是本文的前序工作，本文将混合方案升级为纯 LLM 的双代理方案
- **启发**：这种 "LLM 替代 RL" 的思路可以迁移到其他多智能体协调场景（如自动驾驶车队、机器人集群），但需注意推理延迟的实际约束

## 评分

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 双层 LLM 协作控制架构有创意 |
| 技术深度 | 3 | 信道建模详实，但 LLM 部分缺乏深入分析 |
| 实验充分性 | 3 | 基线合理但缺乏消融和计算开销对比 |
| 写作质量 | 3.5 | 结构清晰，但部分公式符号定义不够统一 |
| 实用价值 | 3 | 概念验证有意义，但实际部署需解决延迟问题 |
| **综合** | **3.3** | 有趣的探索方向，但作为 Workshop 论文深度有限 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] CoLC: Communication-Efficient Collaborative Perception with LiDAR Completion](../../CVPR2026/autonomous_driving/colc_communication-efficient_collaborative_perception_with_lidar_completion.md)
- [\[NeurIPS 2025\] Continuous Simplicial Neural Networks](../../NeurIPS2025/autonomous_driving/continuous_simplicial_neural_networks.md)
- [\[ICCV 2025\] MGSfM: Multi-Camera Geometry Driven Global Structure-from-Motion](../../ICCV2025/autonomous_driving/mgsfm_multi-camera_geometry_driven_global_structure-from-motion.md)
- [\[ICCV 2025\] CoLMDriver: LLM-based Negotiation Benefits Cooperative Autonomous Driving](../../ICCV2025/autonomous_driving/colmdriver_llm-based_negotiation_benefits_cooperative_autonomous_driving.md)
- [\[CVPR 2026\] Open-Ended Instruction Realization with LLM-Enabled Multi-Planner Scheduling in Autonomous Vehicles](../../CVPR2026/autonomous_driving/open-ended_instruction_realization_with_llm-enabled_multi-planner_scheduling_in_.md)

</div>

<!-- RELATED:END -->
