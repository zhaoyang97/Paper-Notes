---
title: >-
  [论文解读] ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation
description: >-
  [AAAI 2026][机器人][单次模仿学习] 提出 ManiLong-Shot 框架，通过交互感知的任务分解、不变区域预测和区域匹配三个模块，仅在10个短序列任务上训练即可泛化到20个未见长序列操作任务，单次模仿成功率 30.2%，相对SOTA提升22.8%。 问题定义 单次模仿学习（OSIL）：从单个演示中学习新技能…
tags:
  - "AAAI 2026"
  - "机器人"
  - "单次模仿学习"
  - "长序列操作"
  - "交互感知"
  - "不变区域"
  - "任务分解"
---

# ManiLong-Shot: Interaction-Aware One-Shot Imitation Learning for Long-Horizon Manipulation

**会议**: AAAI 2026  
**arXiv**: [2512.16302](https://arxiv.org/abs/2512.16302)  
**代码**: [网站](https://sites.google.com/view/manilong-shot)  
**领域**: 强化学习  
**关键词**: 单次模仿学习, 长序列操作, 交互感知, 不变区域, 任务分解

## 一句话总结

提出 ManiLong-Shot 框架，通过交互感知的任务分解、不变区域预测和区域匹配三个模块，仅在10个短序列任务上训练即可泛化到20个未见长序列操作任务，单次模仿成功率 30.2%，相对SOTA提升22.8%。

## 研究背景与动机

### 问题定义

单次模仿学习（OSIL）：从单个演示中学习新技能，无需额外训练。机器人需在日常生活中快速学习并执行多样化的长序列操作任务（如"摆餐桌""整理厨房"），这些任务涉及与多个物体的顺序交互。

### 现有方法局限

**短序列限制**：大多数 OSIL 方法仅适用于短序列技能（如 IMOP, zhang2024oneshot），无法扩展到多步骤长序列任务

**任务变体依赖**：部分方法要求新任务是训练任务的微小变体，或依赖已知的3D物体模型

**预定义原语库**：wu2024one 依赖预定义的原语库来组合长序列操作，缺乏灵活性

### 核心动机 —— 人类学习的启示

人类面对未见任务（如摆放餐具）时，自然地将其分解为短序列原语并推断关键交互区域：
1. 拿起盘子（边缘） → 2. 放置盘子（桌上目标位置） → 3. 拿起叉子（手柄）

每个原语以物理交互（接触/释放）为界，模仿通过在这些区域上复制动作实现。核心问题：机器人能否从未标注的演示中推断子任务边界和关键交互区域？

## 方法详解

### 整体框架

围绕物理交互事件构建三个核心模块：
1. **交互感知任务分解**：将演示分解为预接触/抓取/后接触三阶段的原语序列
2. **交互感知区域预测网络**：为每个原语预测功能不变的交互区域
3. **交互感知区域匹配网络**：将预测区域与当前观测对齐，计算目标末端执行器姿态

推理流程：分解演示 → 预测不变区域 → 匹配当前场景 → 姿态回归 → 运动规划 → 执行 → 迭代直至任务完成。

### 关键设计

#### 1. **交互感知任务分解**

将演示轨迹组织为基于物理交互阶段的原语序列：

- **预接触阶段**：夹爪打开并接近物体，以关节速度降为零（对齐就位）结束
- **抓取阶段**：夹爪从打开到关闭，夹持物体
- **后接触阶段**：成功抓取后，夹爪从关闭到打开，放置物体或与其他物体交互

两种分解策略可互换：
- **规则法**：分析关节速度和夹爪状态变化推断阶段边界，稳定可靠
- **VLM法**：使用 GPT-4o 通过结构化轨迹表示自动识别交互阶段，具有语义感知能力

短序列任务包含一个交互周期（3个阶段），长序列任务重复多个周期。

设计动机：物理交互事件是自然的子任务边界，比语义分割更鲁棒，跨任务和环境可迁移。

#### 2. **交互感知区域预测网络**

识别每个交互阶段中功能和语义不变的交互区域。不变区域定义：3D几何子集，在具有相同最优策略的状态中保持对 $SE(3)$ 变换等变的结构。

架构：
- 输入：连续状态对 $\{s_i, s_{i+1}\}$ 的 RGB-D 密集点云
- 骨干：Point Cloud Transformer V3 (PTV3)，渐进下采样+跨场景交叉注意力+场景内自注意力
- 输出：交互概率分布→激活相关区域 $\mathcal{I}(s_i)$

关键区分：
- **预接触+抓取阶段**：联合训练（功能相似），预测抓取表面区域
- **后接触阶段**：额外启用**定位网络**（Positioning Network），使用注意力机制将抓取物体与目标区域对齐

训练监督：使用仿真中的实例分割掩码作为 ground-truth。仅在短序列任务 $\mathcal{T}^{\text{sh}}$ 上训练。

设计动机：不变区域概念使知识从短序列任务迁移到长序列成为可能——同一"拿杯子边缘"区域在不同任务中保持一致。

#### 3. **交互感知区域匹配网络**

将演示中预测的不变区域与当前执行状态对齐，计算目标姿态。

流程：
1. **状态路由网络**：选择演示轨迹中与当前状态最相似的帧
2. **特征融合**：裁剪不变区域点云 $\mathcal{I}(s_i)$ 与当前场景点云联合下采样
3. **双阶段注意力**：交叉-自-交叉注意力模块增强空间和几何特征对齐
4. **对应矩阵计算**：双 softmax 匹配算法计算对应矩阵 $\mathbf{C}$

基于对应矩阵的姿态回归：

$$\mathbf{T}_j = \arg\min_{\mathbf{T} \in SE(3)} \|\mathbf{T} \mathbf{T}_i^{-1} P_{\mathcal{I}(s_i)} \mathbf{C} - P_{s_j}\|$$

运动规划使用 RRT-Connect 算法寻找无碰撞轨迹。

设计动机：基于对应而非直接动作预测的姿态估计对物体姿态和场景布局更鲁棒。

### 训练策略

- 仅在10个短序列任务（每个100个演示轨迹）上训练
- 前/侧/腕部摄像头 RGB-D 观测
- PTV3 骨干网络
- 监督学习：使用 ground-truth 实例掩码和对应矩阵

## 实验关键数据

### 实验设置

- **仿真基准**：RLBench-Oneshot（30个任务：10个 SH + 20个 LH）
    - LH 三个难度级别：Level 1（13任务，6交互），Level 2（4任务，9交互），Level 3（3任务，12交互）
- **真实机器人**：UFactory xArm7 + RealSense D435/D415，3个 LH 任务
- **基线**：ARP, 3DDA, RVT2（SOTA IL模型）, IMOP（OSIL SOTA）
- 每任务25次试验（5个随机种子），报告均值±标准差

### 主实验 —— 短序列任务

| 模型 | 平均成功率(%) | 平均排名 | 最佳任务数 |
|------|-------------|---------|-----------|
| IMOP | 65.24 | 3.9 | 1 |
| RVT2 | 79.2 | 3.1 | 0 |
| 3DDA | 85.0 | 2.9 | 2 |
| ARP | 86.6 | 2.7 | 3 |
| **ManiLong-Shot** | **90.4** (+3.8%) | **1.9** | **6** |

### 主实验 —— 未见长序列任务（OSIL）

| 模型 | 平均成功率(%) | 平均排名 |
|------|-------------|---------|
| RVT2+FT | 4.1 | 3.7 |
| 3DDA+FT | 4.5 | 3.3 |
| ARP+FT | 4.7 | 3.3 |
| IMOP | 7.4 | 2.6 |
| **ManiLong-Shot** | **30.2** (+22.8%) | **1.0** |

代表性任务对比：

| 任务 | IMOP | ManiLong-Shot |
|------|------|---------------|
| Empty Container | 4.0% | **28.0%** |
| Empty Dishwasher | 1.3% | **42.7%** |
| Put Item in Drawer | 40.0% | **65.3%** |
| Take Item out Drawer | 38.7% | **76.0%** |
| Set Table | 2.7% | **17.3%** |
| Stack Blocks | 1.3% | **8.0%** |

### 消融实验

| 配置 | Level 1 | Level 2 | Level 3 | 说明 |
|------|---------|---------|---------|------|
| ManiLong-Shot (Rule) | 最高 | 最高 | 最高 | 基于规则的分解 |
| ManiLong-Shot (VLM) | 低于Rule | 低于Rule | 低于Rule | VLM推理不稳定 |
| w/o Positioning | 下降 | 下降明显 | 下降最大 | 后接触阶段放置不准 |

### 真实机器人实验

| 任务 | IMOP | ManiLong-Shot |
|------|------|---------------|
| Stack Blocks | 60% | **80%** |
| Stack Cups | 20% | **60%** |
| Place Cups | 20% | **40%** |
| 平均 | 33.3% | **60.0%** (+26.7%) |

### 关键发现

1. **短序列全面领先**：90.4% 平均成功率，在10个训练任务中6个最优
2. **长序列巨大优势**：30.2% vs IMOP 7.4%，22.8% 绝对提升；即使微调后的 SOTA 模型也仅4-5%
3. **规则法优于VLM法**：VLM 推理不稳定，任务越复杂差距越大；但 VLM 法仍显著优于所有基线
4. **定位网络关键**：去除后后接触阶段放置不准，影响后续子任务执行链
5. **Sim-to-Real 有效**：60% 平均成功率，比 IMOP 提升 26.7%

## 亮点与洞察

1. **物理交互作为通用分解原语**：预接触/抓取/后接触三阶段是抓取操作的自然结构，比语义分割更鲁棒
2. **仅用短序列训练泛化长序列**：10个短任务→20个未见长任务，展示了交互原语的组合泛化能力
3. **不变区域的优雅抽象**：将"杯子边缘适合抓取"这一功能性质形式化为 $SE(3)$ 等变的不变区域
4. **两种分解策略的灵活性**：规则法稳定 + VLM法可学习语义模式，可按需选择
5. **端到端一次性泛化**：无需任务特定微调，真正的 zero-shot

## 局限与展望

1. **仅限抓取操作**：三阶段分解基于物理接触，不适用于非抓取行为（如擦拭、倒水等连续交互）
2. **平行夹爪假设**：框架围绕平行夹爪设计，灵巧手操作未涉及
3. **桌面环境限制**：实验场景为桌面操作，更复杂环境（移动操作、多房间）未验证
4. **绝对成功率仍低**：长序列任务平均30.2%，虽远超基线但实用性仍有限
5. **VLM 推理不稳定**：GPT-4o 作为任务分解器的不一致性在复杂任务中放大

## 相关工作与启发

- **IMOP (zhang2024oneshot)**：不变区域概念的先驱工作，本文的直接基础
- **GravMAD/DECO (chen2024/2025)**：基于物理交互的子任务分解方法
- **RLBench (james2020)**：机器人操作仿真基准，本文构建了 RLBench-Oneshot 子集
- **PTV3**：点云 Transformer，用作几何特征提取骨干
- 对长序列操作研究的启发：组合短序列原语比直接学习长序列更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ — 交互感知的三阶段分解+不变区域预测组合新颖
- 实验充分度: ⭐⭐⭐⭐⭐ — 30个任务基准、消融、VLM对比、真实机器人实验全面
- 写作质量: ⭐⭐⭐⭐ — 图示清晰，问题定义严谨，方法描述详尽
- 价值: ⭐⭐⭐⭐ — 为长序列单次模仿学习提供实用框架

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](../../ICLR2026/robotics/vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)
- [\[AAAI 2026\] PanoNav: Mapless Zero-Shot Object Navigation with Panoramic Scene Parsing and Dynamic Memory](panonav_mapless_zero-shot_object_navigation_with_panoramic_scene_parsing_and_dyn.md)
- [\[AAAI 2026\] Actor-Critic for Continuous Action Chunks: A Reinforcement Learning Framework for Long-Horizon Robotic Manipulation with Sparse Reward](actor-critic_for_continuous_action_chunks_a_reinforcement_le.md)
- [\[CVPR 2026\] AGiLe: Learning Robust Long-Horizon Manipulation via Affordance-Grounded Bidirectional Latent Planning](../../CVPR2026/robotics/agile_learning_robust_long-horizon_manipulation_via_affordance-grounded_bidirect.md)
- [\[ICML 2026\] Drift is a Sampling Error: SNR-Aware Power Distributions for Long-Horizon Robotic Planning](../../ICML2026/robotics/drift_is_a_sampling_error_snr-aware_power_distributions_for_long-horizon_robotic.md)

</div>

<!-- RELATED:END -->
