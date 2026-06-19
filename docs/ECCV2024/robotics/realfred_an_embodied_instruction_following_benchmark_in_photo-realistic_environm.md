---
title: >-
  [论文解读] ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments
description: >-
  [ECCV 2024][机器人][具身智能] 提出 ReALFRED 基准，使用 150 个真实世界 3D 扫描的多房间可交互环境替代 ALFRED 的合成单房间场景，提供 30,696 条自由格式语言指令，揭示了现有具身指令跟随方法在真实环境中性能显著下降的问题。 构建能执行日常家务任务的自主机器人助手是长期研究目标…
tags:
  - "ECCV 2024"
  - "机器人"
  - "具身智能"
  - "指令跟随"
  - "3D 扫描环境"
  - "多房间导航"
  - "基准数据集"
---

# ReALFRED: An Embodied Instruction Following Benchmark in Photo-Realistic Environments

**会议**: ECCV 2024  
**arXiv**: [2407.18550](https://arxiv.org/abs/2407.18550)  
**代码**: [GitHub](https://github.com/snumprlab/realfred)  
**领域**: 机器人  
**关键词**: 具身智能, 指令跟随, 3D 扫描环境, 多房间导航, 基准数据集

## 一句话总结

提出 ReALFRED 基准，使用 150 个真实世界 3D 扫描的多房间可交互环境替代 ALFRED 的合成单房间场景，提供 30,696 条自由格式语言指令，揭示了现有具身指令跟随方法在真实环境中性能显著下降的问题。

## 研究背景与动机

构建能执行日常家务任务的自主机器人助手是长期研究目标。为训练此类智能体，需要提供可交互的模拟环境让其在大量交互中学习任务完成技能。

当前基准和环境存在**三个核心差距**：

**视觉域差距**：以 ALFRED 为代表的基准使用 Unity 游戏引擎和合成 CAD 资产构建环境，视觉风格与真实世界差异明显。研究表明这种域差距会导致部署时性能显著下降。

**空间规模受限**：ALFRED 仅提供单房间粒度的环境（总可导航面积 1,356 m²），而真实家庭场景通常涉及跨多个房间的导航。合成环境中构建大规模高保真空间极具挑战。

**交互能力不足**：3D 扫描环境（如 Matterport3D、HM3D）虽然视觉逼真，但场景是静态的——物体无法交互（不能拾取、加热、冷却等）。Habitat-Web 虽支持拾放，但仅限基础交互，且使用模板化语言指令。

**存在矛盾**：3D 扫描环境视觉逼真但缺乏交互；合成环境支持丰富交互但视觉失真。ReALFRED 旨在**同时满足**四个条件：真实感视觉、多房间导航、丰富物体交互、自由格式语言指令——这是此前没有基准能完全覆盖的（Fig.2）。

## 方法详解

### 整体框架

ReALFRED 是一个完整的基准数据集，核心工作包括三部分：(1) 构建可交互的 3D 扫描环境，(2) 生成专家演示，(3) 收集自由格式语言指令。基于 AI2-THOR 模拟器，扩展了 ALFRED 基准以支持更大空间和更小视觉域差距。

### 关键设计

1. **可交互的 3D 扫描场景构建**：团队实地访问住宅，使用与 Matterport3D 相同的 3D 扫描仪（配备三个 RGB 相机和深度传感器），以 2.5 米间隔扫描，针对家具遮挡区域补充扫描。核心挑战在于：扫描数据中物体与背景融为一体，无法交互。解决方案是**手动分离**：将 3D 扫描分解为背景元素和可交互物体，为物体添加状态变化纹理（如"脏"纹理），然后在 Unity 编辑器中重建，使其兼容 AI2-THOR 模拟器。提供 150 个场景、112 种物体类别（86 种可拾取 + 26 种容器），总楼层面积 10,060 m²——远超 ALFRED 的 120 场景/2,555 m²。

2. **多房间任务设计与专家演示**：采用 PDDL（规划域定义语言）规则和规划器生成 7 种任务类型的专家演示。相比 ALFRED 的单房间任务，ReALFRED 的任务需要**跨房间导航**——智能体需穿过门和走廊从一个房间到另一个房间，完成更长步数的任务（Fig.7 显示 ReALFRED 的步数和轨迹长度显著超过 ALFRED）。数据按 135 个 seen 场景和 15 个 unseen 场景划分。

3. **自由格式语言指令收集**：通过 93 名 Amazon Mechanical Turk "Master" 工人收集 30,696 条语言指令，每条包含高层目标描述和分步指令。通过额外投票验证保证质量，无效指令被替换重收。

### 损失函数 / 训练策略

ReALFRED 本身是一个基准而非模型。作者评估了两类方法：
- **模仿学习**（Seq2Seq, MOCA, ABP）：直接从视觉观察和语言指令映射到动作序列
- **空间地图重建**（HLSM, FILM, LLM-Planner, CAPEAM）：基于预测深度图构建语义空间表示后规划动作

所有方法使用与 ALFRED 相同的评估指标：Success Rate (SR) 和 Goal-Condition Success Rate (GC)。

## 实验关键数据

### 主实验

**各方法在 ReALFRED 上的表现（%）**：

| 方法 | 类别 | Val Seen SR | Val Seen GC | Val Unseen SR | Val Unseen GC | Test Unseen SR | Test Unseen GC |
|------|------|------------|------------|--------------|--------------|---------------|---------------|
| Seq2Seq | 模仿学习 | 0.77 | 6.93 | 0.00 | 4.03 | 0.00 | 3.50 |
| MOCA | 模仿学习 | 12.64 | 20.95 | 1.44 | 6.76 | 0.62 | 5.14 |
| **ABP** | **模仿学习** | **24.71** | **33.80** | **4.22** | **11.71** | **3.54** | **10.57** |
| HLSM | 空间地图 | 4.23 | 9.14 | 1.08 | 6.12 | 0.49 | 4.28 |
| FILM | 空间地图 | 7.08 | 11.93 | 4.44 | 9.26 | 2.15 | 6.56 |
| CAPEAM | 空间地图 | 13.45 | 18.16 | 4.92 | 9.47 | 2.87 | 7.36 |
| **人类** | - | - | - | - | - | **85.00** | **91.30** |

### 消融实验

**Sim2Real 迁移对比**：

| 设置 | 域适配方法 | 多+单房间 SR | 仅单房间 SR | 说明 |
|------|-----------|------------|------------|------|
| Sim2Real | 无 | 0.115 | 0.0 | 合成环境训练，直接评估 |
| Sim2Real | CycleGAN | 0.115 | 0.327 | 加域适配 |
| Sim2Real | UVCGAN-v2 | 0.115 | 0.327 | 更好的域适配 |
| **Real2Real** | **无** | **2.405** | **2.614** | **真实扫描环境训练** |

**环境规模对比**：

| 基准 | 场景数 | 总楼层面积 (m²) | 总可导航面积 (m²) | 导航复杂度 | 物体类别 |
|------|-------|----------------|------------------|-----------|---------|
| ReplicaCAD | 111 | 8,824.5 | - | - | 92 |
| ALFRED | 120 | 2,555 | 1,356 | 2.549 | 82 |
| **ReALFRED** | **150** | **10,060** | **4,251** | **3.020** | **112** |

### 关键发现

- **所有 SOTA 方法性能全面下降**：最佳方法 ABP 在 ALFRED unseen 的 SR 为 ~26%，在 ReALFRED unseen 仅 4.22%，下降超 80%
- **与 ALFRED 相反的趋势**：在 ALFRED 上空间地图方法优于模仿学习，而在 ReALFRED 上**模仿学习反超空间地图方法**——因为多房间环境中有限视野导致空间地图重建受限
- **导航是主要瓶颈**：ABP 在 ReALFRED unseen 的导航成功率仅 59.18%，而在 ALFRED 为 84.82%
- **大空间导致性能进一步下降**：小于 30.44 m² 的场景 SR=5.46%，大于该阈值的场景仅 1.77%
- **门口和狭窄通道**是碰撞热点（Fig.8），空间地图方法将障碍物感知得过大，进一步阻塞了狭窄通道
- **Sim2Real 差距显著**：Real2Real 的 SR 是 Sim2Real 的 20+ 倍，即使加了 CycleGAN 域适配效果也有限
- 人类表现 85% SR vs 最佳方法 3.54% SR，差距巨大

## 亮点与洞察

- **填补了重要空白**：首次同时满足真实感视觉、多房间规模、丰富交互、自由语言指令四个条件
- **精细的环境工程**：手动分离 3D 扫描的每个物体、添加状态纹理、重建为可交互资产，工作量巨大但提供了高质量数据
- **揭示了深刻问题**：合成环境中表现良好的方法在真实环境中几乎失效，说明当前方法的泛化能力远远不足
- 多房间导航引入的挑战（狭窄通道、大空间探索）是现有方法未解决的根本问题

## 局限与展望

- **任务类型有限**：仅 7 种任务类型，未涵盖更复杂的真实场景（如需双手操作的任务）
- **仅支持英语**：实际部署需要多语言支持
- 手动分离物体的过程劳动密集，限制了数据集规模的进一步扩展
- 环境中可交互物体的物理仿真精度（如物体碰撞、流体等）相比合成环境可能有限
- 当前方法的极低成功率也意味着该基准可能对近期方法过于困难，需要更多中间难度的评估指标

## 相关工作与启发

- 与 ALFRED 的关系：ReALFRED 是 ALFRED 的现实主义升级版，保留相同的任务框架和评估指标，便于直接对比
- 与 Habitat-Web 的区别：HW 支持 3D 扫描环境中的拾放，但仅用模板语言；ReALFRED 支持更复杂的交互（加热、冷却、切割等）和自由语言
- Sim2Real 实验启发：单纯的视觉域适配（CycleGAN）远不足以弥合差距，需要从环境布局、物体分布、任务复杂度等更深层次解决
- 对 LLM-based agent 的启示：即使 LLM-Planner 利用了语言模型的知识，在 ReALFRED 上表现仍然不佳，说明**低层视觉感知和导航能力**是当前瓶颈

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 构建了第一个同时满足四项关键需求的具身基准，工程创新量大
- **实验充分度**: ⭐⭐⭐⭐⭐ — 评估了 7 种 SOTA 方法、Sim2Real 迁移、人类表现、多维度环境分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，对比全面，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ — 为具身 AI 社区指出了关键差距，高价值基准数据集将推动更鲁棒的方法研发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models](../../ICML2025/robotics/hi_robot_open-ended_instruction_following_with_hierarchical_vision-language-acti.md)
- [\[ICLR 2026\] Test-Time Mixture of World Models for Embodied Agents in Dynamic Environments](../../ICLR2026/robotics/test-time_mixture_of_world_models_for_embodied_agents_in_dynamic_environments.md)
- [\[ECCV 2024\] See and Think: Embodied Agent in Virtual Environment](see_and_think_embodied_agent_in_virtual_environment.md)
- [\[AAAI 2026\] Realistic Synthetic Household Data Generation at Scale](../../AAAI2026/robotics/realistic_synthetic_household_data_generation_at_scale.md)
- [\[CVPR 2025\] Robotic Visual Instruction](../../CVPR2025/robotics/robotic_visual_instruction.md)

</div>

<!-- RELATED:END -->
