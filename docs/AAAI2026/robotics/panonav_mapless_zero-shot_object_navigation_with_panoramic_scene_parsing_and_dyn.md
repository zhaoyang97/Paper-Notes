---
title: >-
  [论文解读] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory
description: >-
  [AAAI 2026][机器人][零样本目标导航] 提出 PanoNav，一个仅使用 RGB 图像的无地图零样本目标导航框架，通过全景场景解析（Panoramic Scene Parsing）释放 MLLM 的空间推理能力，并引入动态有界记忆队列（Dynamic Bounded Memory Queue）避免局部死锁问题。
tags:
  - "AAAI 2026"
  - "机器人"
  - "零样本目标导航"
  - "全景场景解析"
  - "动态记忆"
  - "无地图导航"
  - "MLLM"
---

# PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory

**会议**: AAAI 2026  
**arXiv**: [2511.06840](https://arxiv.org/abs/2511.06840)  
**代码**: 无  
**领域**: 机器人  
**关键词**: 零样本目标导航, 全景场景解析, 动态记忆, 无地图导航, MLLM

## 一句话总结

提出 PanoNav，一个仅使用 RGB 图像的无地图零样本目标导航框架，通过全景场景解析（Panoramic Scene Parsing）释放 MLLM 的空间推理能力，并引入动态有界记忆队列（Dynamic Bounded Memory Queue）避免局部死锁问题。

## 研究背景与动机

目标导航（ObjectNav）是家用机器人的基础能力，要求机器人在未知环境中定位并导航至指定物体。现有方法存在三个主要局限：

**1. 依赖深度传感器和预建地图**：大量方法（如 VLFM、ESC、VoroNav、L3MVN）需要 RGB-D 输入构建 2.5D 场景表示或度量地图。这增加了硬件负担，降低了在嘈杂或动态环境中的鲁棒性。

**2. 仅限于封闭类别集合**：许多方法只能识别预定义的物体类别，无法泛化到开放词汇的真实场景。

**3. 无地图方法的"局部死锁"问题**：现有无地图方法（如 ZSON、PixNav）仅基于当前观测做决策，忽略历史轨迹信息，容易陷入反复访问已探索区域的死循环。这是因为 LLM 过度依赖物体-房间先验（如客厅可能有沙发），而不考虑"我已经搜索过这个区域了"。

**核心动机**：能否设计一个仅用 RGB 图像、不需要地图和深度信息的开放词汇导航系统，同时解决局部死锁的问题？

## 方法详解

### 整体框架

PanoNav 在每个时间步采集 6 个方向的 RGB 图像（60度间隔）构成全景视图，经过两个核心模块处理：

1. **Panoramic Scene Parsing**（全景场景解析）：利用 MLLM（Qwen-2.5-VL）从 6 视角 RGB + 点阵图（dot matrix）中提取局部方向描述和全局场景摘要
2. **Memory-Guided Decision-Making**（记忆引导决策）：结合动态有界记忆队列，由 LLM（DeepSeek-V3）做出导航决策

### 关键设计

#### 1. **局部方向解析（Local Directional Parsing）**

对每个方向视图提取丰富的上下文信息，包括物体存在、空间关系、房间类型、目标出现概率等。关键创新是引入**点阵图（dot matrix）**作为辅助输入：

$$\mathbf{M}_t^i = \text{SCA}(\mathbf{V}_t^i), \quad i \in \{1, \ldots, 6\}$$

其中 SCA 是 Scaffold 处理方法。原始 RGB 图像捕捉几何距离线索，点阵图增强平面位置理解。两者互补，使 MLLM 能在两个空间维度上推理。构建空间关系图：

$$G_t^i = \mathcal{P}(\Psi(\mathbf{V}_t^i), \Phi(\mathbf{V}_t^i, \mathbf{M}_t^i))$$

其中 $\Psi(\cdot)$ 提取几何距离关系，$\Phi(\cdot)$ 解析平面位置关系，$\mathcal{P}(\cdot)$ 聚合为统一图。

#### 2. **全局全景摘要（Global Panoramic Summary）**

超越局部解析，进行全局分析以获得更高层语义理解——识别周围环境中的物体并确定当前所处的房间/场景类型（如厨房、走廊）。提供一种隐式的自我定位感知能力。

#### 3. **动态有界记忆队列（Dynamic Bounded Memory Queue）**

这是本文解决局部死锁的核心设计。维护一个最大长度为 $n$ 的队列 $\mathcal{Q}$，存储最近的全局摘要描述：

$$\mathcal{Q}_t = \{\mathbf{gs}_{t-n}, \ldots, \mathbf{gs}_{t-2}, \mathbf{gs}_{t-1}\}$$

- 初始阶段队列为空，智能体仅基于当前观测决策
- 队列满后（$f_t = 1$），新路点描述入队，最旧的描述出队
- 决策函数根据队列状态切换：

$$\begin{cases} \mathbf{r}_t = \mathcal{F}(\mathbf{ld}_t, \mathbf{gs}_t), & f_t = 0 \\ \mathbf{r}_t = \mathcal{F}(\mathbf{ld}_t, \mathbf{gs}_t, \mathcal{Q}_t), & f_t = 1 \end{cases}$$

这样 LLM 能意识到"我已经在客厅搜索过了，应该去走廊看看"，避免局部死锁。

### 损失函数 / 训练策略

PanoNav 是一个**零样本（zero-shot）框架**，不需要任何训练。所有模块基于现成的预训练模型：
- 场景解析：Qwen-2.5-VL（冻结参数）
- 导航决策：DeepSeek-V3（冻结参数）
- 运动控制：使用 PixNav 作为运动控制器

## 实验关键数据

### 主实验

在 HM3D 数据集上评估 200 个随机 episode，对比多种导航方法：

| 方法 | 输入模态 | 开放词汇 | 地图 | SR↑ | SPL↑ |
|---|---|---|---|---|---|
| FBE | RGB-D, GPS | 封闭集 | 地图 | 33.7 | 15.3 |
| SemExp | RGB-D, GPS | 封闭集 | 地图 | 37.9 | 18.8 |
| Habitat-Web | RGB-D, GPS | 封闭集 | 无地图 | 41.5 | 16.0 |
| OVRL | RGB-D, GPS | 封闭集 | 无地图 | 62.0 | 26.8 |
| VLFM | RGB-D, GPS | 开放集 | 地图 | 52.2 | 30.4 |
| L3MVN | RGB-D, GPS | 开放集 | 地图 | 50.4 | 23.1 |
| ZSON | RGB only | 开放集 | 无地图 | 25.5 | 12.6 |
| PixNav | RGB only | 开放集 | 无地图 | 37.9 | 20.5 |
| **PanoNav (Ours)** | **RGB only** | **开放集** | **无地图** | **43.5** | **23.7** |

**关键结论**：在同等设置（RGB-only、无地图、开放词汇）下，PanoNav 的 SR 比 PixNav 提升 14.76%，SPL 提升 15.61%。甚至超越了多个使用深度传感器和地图的方法。

### 消融实验

| 配置 | SR↑ | SPL↑ | 说明 |
|---|---|---|---|
| 3 视角（无全景） | 19.5 | 9.97 | 视野受限导致严重性能下降 |
| 全景 + 一步决策（无解耦） | 35.0 | 20.47 | 端到端推理能力不足 |
| 全景 + 解耦（无记忆） | 38.5 | 22.57 | 缺少历史信息易入死锁 |
| **全景 + 解耦 + 记忆（完整）** | **43.5** | **23.73** | 完整框架最优 |

### 死锁避免测试

专门设计了具有高欺骗性的 episode（如"在客厅中寻找沙发但沙发不在客厅"），每个episode 重复 10 次实验：

| 配置 | SR↑ | SPL↑ | DTS(fail)↓ | 逃逸率↑ |
|---|---|---|---|---|
| 无记忆 | 12.0 | 4.9 | 6.7 | 32.0% |
| **有记忆** | **48.0** | **19.2** | **4.7** | **82.0%** |

记忆引导的方法成功率提升 4 倍，逃逸率从 32% 提升至 82%。

### 关键发现

1. **6 视角全景比 3 视角（仅前方）SR 提升 24%**——全方位感知至关重要
2. **解耦感知与决策优于端到端**——分步处理减轻 MLLM 的认知负荷
3. **动态记忆是避免死锁的关键**——即使失败案例中，有记忆的智能体也更接近目标

## 亮点与洞察

1. **全 RGB、无地图、零样本**——技术栈极简，不需要深度传感器、GPS、预建地图或任何训练，完全依赖预训练的 MLLM 和 LLM
2. **点阵图（dot matrix）的巧妙设计**：通过 Scaffold 方法将 RGB 转为点阵图，增强 MLLM 对平面空间关系的理解，弥补了 MLLM 在精确空间推理上的不足
3. **动态记忆队列的简洁有效**：仅存储文本摘要而非复杂的空间表示，计算开销极低
4. **死锁避免测试的实验设计**具有很强的说服力

## 局限与展望

1. **仅在 HM3D 上测试**，200 个 episode 的评估规模偏小
2. **记忆队列仅存储文本摘要**，可能丢失细粒度的空间信息，未来可考虑结合轻量级拓扑图
3. **运动控制器依赖 PixNav**，导航的底层控制精度受限
4. **无真实机器人实验**——仅在仿真环境中验证
5. **固定的 6 视角、60 度间隔**——在狭窄空间中可能存在冗余或不足

## 相关工作与启发

- **与 VLFM、VoroNav 等地图方法的对比**：PanoNav 证明了在简单架构下也能取得有竞争力的结果
- **MLLM 作为场景理解引擎**：未来可在更多机器人任务中使用 MLLM 替代传统的感知模块
- **记忆机制在导航中的重要性**：从简单的文本队列到更复杂的空间记忆，是一个值得探索的谱系

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 全景解析+动态记忆的组合新颖，但各个模块相对简单
- **实验充分度**: ⭐⭐⭐⭐ — 消融和死锁测试设计巧妙，但数据集规模偏小
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，图表直观
- **价值**: ⭐⭐⭐⭐ — 为无地图导航提供了实用的解决方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TrajRAG: Retrieving Geometric-Semantic Experience for Zero-Shot Object Navigation](../../CVPR2026/robotics/trajrag_retrieving_geometric-semantic_experience_for_zero-shot_object_navigation.md)
- [\[CVPR 2026\] Memory-Augmented Scene Understanding and Exploration for Open-World Aerial Object-Goal Navigation](../../CVPR2026/robotics/memory-augmented_scene_understanding_and_exploration_for_open-world_aerial_objec.md)
- [\[ECCV 2024\] Prioritized Semantic Learning for Zero-shot Instance Navigation](../../ECCV2024/robotics/prioritized_semantic_learning_for_zero-shot_instance_navigation.md)
- [\[AAAI 2026\] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation](manilong-shot_interaction-aware_one-shot_imitation_learning_for_long-horizon_man.md)
- [\[CVPR 2026\] Humanoid Generative Pre-Training for Zero-Shot Motion Tracking](../../CVPR2026/robotics/humanoid_generative_pre-training_for_zero-shot_motion_tracking.md)

</div>

<!-- RELATED:END -->
