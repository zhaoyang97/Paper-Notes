---
title: >-
  [论文解读] AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving
description: >-
  [AAAI 2026 Oral][自动驾驶][端到端自动驾驶] 识别出端到端自动驾驶中ego status过度依赖的架构根源（BEV编码器中ego status的过早融合），提出AdaptiveAD双分支架构：场景驱动分支（去除ego status）和自我驱动分支独立生成决策，再通过场景感知融合模块自适应整合，配合路径注意力、BEV单向蒸馏和自回归在线建图辅助任务，在nuScenes上达到SOTA规划性能。
tags:
  - "AAAI 2026 Oral"
  - "自动驾驶"
  - "端到端自动驾驶"
  - "因果混淆"
  - "ego-status过度依赖"
  - "双分支架构"
  - "多上下文融合"
---

# AdaptiveAD: Decoupling Scene Perception and Ego Status for End-to-End Autonomous Driving

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.13079](https://arxiv.org/abs/2511.13079)  
**代码**: 无  
**领域**: 自动驾驶  
**关键词**: 端到端自动驾驶, 因果混淆, ego-status过度依赖, 双分支架构, 多上下文融合

## 一句话总结

识别出端到端自动驾驶中ego status过度依赖的架构根源（BEV编码器中ego status的过早融合），提出AdaptiveAD双分支架构：场景驱动分支（去除ego status）和自我驱动分支独立生成决策，再通过场景感知融合模块自适应整合，配合路径注意力、BEV单向蒸馏和自回归在线建图辅助任务，在nuScenes上达到SOTA规划性能。

## 研究背景与动机

端到端自动驾驶模型普遍存在"凭惯性驾驶"而非"凭视觉驾驶"的问题（因果混淆），在新场景或长尾场景下表现糟糕。

**问题根源分析**：
- 现有架构（如UniAD、VAD）在BEV编码器的上游就将ego status（车辆运动学状态）融入感知特征
- 这种过早融合创造了信息捷径：规划模块可以直接依赖ego status绕过复杂的场景理解
- 高速行驶时遇到突发障碍物，"凭惯性"模型会产生致命轨迹

**现有缓解方案的不足**：
- 数据层面（如平衡采样）：缓解数据集偏差但不改变模型内部信息流
- 正则化（如dropout、对比模仿学习）：改善特征质量但可能加剧多任务学习的困难
- 这些方法都在"修饰输入"，而非重构决策过程本身

## 方法详解

### 整体框架

AdaptiveAD采用**双分支+自适应融合**策略，将场景感知和ego状态inference在架构层面显式解耦：

1. **场景驱动分支（Scene-driven Branch）**：基于多任务学习的环境感知推理，从BEV编码器中**刻意移除**ego status增强
2. **自我驱动分支（Planning-only Branch）**：仅基于规划任务的自我状态推理，保留ego status增强
3. **多上下文决策融合模块**：自适应整合两分支的互补决策

### 关键设计

**（1）场景驱动分支**

基于VAD架构，包含BEV编码器、向量化场景解码器和决策生成器。**核心修改**：移除BEV encode中的BEV query增强步骤（该步骤通常注入ego status），得到纯环境BEV特征 $B_{woes} \in \mathbb{R}^{C \times H_{bev} \times W_{bev}}$。

向量化场景解码器将密集BEV转换为稀疏的agent queries $A$ 和 map queries $M$，决策生成器初始化多模态ego query $E_{woes}$ 并依次与 $A$、$M$、$B_{woes}$ 交互。

**（2）自我驱动分支**

保留BEV query增强操作，产生运动补偿的BEV特征 $B_{wes}$。跳过显式场景解码器，ego query $E_{wes}$ 直接与 $B_{wes}$ 交互。初始参考点直接从ego status预测——体现运动外推的强先验。

**（3）路径注意力（Path Attention）**

替代标准可变形注意力，引入轨迹引导的语义采样：
- 先解码初步轨迹，沿轨迹均匀采样 $T$ 个参考点
- 每个参考点分配独立注意力头，在其邻域内学习采样 $K$ 个局部特征
- 模拟人类驾驶员沿规划路线扫视的行为

$$\text{PathAttn}(E^i, P^i, B) = \sum_{t=1}^T W_t [\sum_{k=1}^K a^{i,t,k} W_t' B_{samp}^{i,t,k}]$$

权重在每个头内归一化（$\sum_k a^{i,t,k} = 1$），利用头间的特征分离同时建模长程上下文和局部细节。

**（4）多上下文决策融合**

融合ego query $E_{fusion}$ 先用场景感知初始化——从场景BEV做全局平均池化：

$$E_{fusion}^{com} = \text{GAP}(B_{woes})$$

再通过transformer融合层进行上下文对齐：将两分支决策拼接后做自注意力，实现丰富的跨上下文交互和内上下文精炼，然后 $E_{fusion}$ 通过交叉注意力从对齐后的多上下文表示中自适应加权。

**（5）BEV单向蒸馏**

场景驱动分支因缺乏ego运动补偿可能产生运动模糊。将 $B_{wes}$（有运动补偿）作为teacher、$B_{woes}$ 作为student：

$$L_{distill} = \alpha L_{distill}^{DF} + \beta L_{distill}^{IK} + \gamma L_{distill}^{IC}$$

包括密集特征蒸馏（agent引导加权）、关键点间相关和通道间相关蒸馏。梯度不回传teacher。

**（6）自回归在线建图**

建立规划→感知的反馈回路：在预测轨迹和GT轨迹的重叠感知区域内，对地图实例施加masked L1损失：

$$L_{autoreg}^{MAP} = \frac{1}{T} \sum_{\tau=1}^T \frac{1}{\|\mathcal{M}\|_1 + \epsilon} \|(\hat{P}_M - P_M) \odot \mathcal{M}\|_1$$

确保感知到的地图在跟随预测轨迹和跟随GT轨迹时保持一致，缓解建图和规划头之间的优化冲突。

### 损失函数 / 训练策略

- 蒸馏和自回归损失权重 $(\alpha, \beta, \gamma, \delta, \lambda) = (0.01, 0.1, 0.01, 0.01, 0.01)$
- 60 epochs，32×A100 GPU，AdamW + CosineAnnealing，batch size 2/GPU
- 基于VAD架构，预测3秒轨迹，使用2秒历史数据，60m×30m感知范围
- 6层ego-BEV交互 + 6层多上下文融合

## 实验关键数据

### 主实验

**Table 1: nuScenes 开环规划性能**

| 方法 | L2 Avg↓ | CR Avg↓ | FPS |
|------|---------|---------|-----|
| UniAD | 0.73 | 0.61 | 1.8 |
| VAD | 0.61 | 0.28 | 3.4 |
| PPAD | 0.58 | 0.19 | 2.6 |
| SparseDrive | 0.61 | 0.10 | 5.2 |
| BridgeAD | 0.58 | 0.08 | 3.1 |
| **AdaptiveAD** | **0.47** | **0.12** | 3.0 |

相比VAD：L2误差降低22%，碰撞率降低57%。

**Table 2: 场景泛化能力（直行ST vs 转弯LR）**

| 方法 | ST L2↓ | LR L2↓ | ST CR↓ | LR CR↓ |
|------|--------|--------|--------|--------|
| VAD | 0.62 | 0.91 | 0.33 | 0.18 |
| VAD (Turning-nuScenes) | - | 0.92 | - | 0.38 |
| **Ours** | **0.47** | **0.63** | **0.11** | **0.16** |
| **Ours** (Turning-nuScenes) | - | **0.63** | - | **0.28** |

转弯场景下优势尤为显著。

**Table 3: Ego状态依赖程度**

| 方法 | 正常 L2 | velocity×0.0 L2 | velocity×0.5 L2 | 100m/s L2 |
|------|---------|-----------------|-----------------|-----------|
| VAD | 0.61 | 5.54 (+808%) | 3.05 | 14.93 |
| **Ours** | **0.47** | **4.08** (+768%) | **2.41** | **5.06** |

VAD在ego velocity归零时L2爆炸810%，AdaptiveAD在极端噪声（100m/s）下仍比VAD正常值低17%。

### 消融实验

**Table 5: 组件逐步累加消融**

| ID | 双分支 | BEV蒸馏 | 场景初始化 | 自回归建图 | L2 Avg↓ | CR Avg↓ |
|----|--------|---------|-----------|-----------|---------|---------|
| 1 | - | - | - | - | 0.57 | 0.22 |
| 2 | ✓ | - | - | - | 0.62 | 0.15 |
| 3 | ✓ | ✓ | - | - | 0.58 | 0.08 |
| 4 | ✓ | ✓ | ✓ | - | 0.52 | 0.12 |
| 5 | ✓ | ✓ | ✓ | ✓ | **0.47** | **0.12** |

- 双分支引入后L2暂时变差（运动模糊），但CR显著下降（场景理解改善）
- BEV蒸馏恢复L2并将CR降低60%（关键组件）
- 场景感知初始化进一步改善L2约10%

### 关键发现

1. **路径注意力优于可变形注意力**（Table 6）：CR Avg从0.15降至0.12，且计算开销相同
2. **组件具备通用性**（Table 7）：Path Attention和自回归在线建图作为插件加入UniAD和SparseDrive均有提升
3. **NAVSIM/Bench2Drive闭环验证**：在ego status噪声下，AdaptiveAD的PDMS/DS/SR均显著优于VAD

## 亮点与洞察

1. **架构级解法**：不同于data/regularization层面的缓解方案，直接在信息流层面切断ego-status捷径
2. **双分支互补逻辑清晰**：场景分支提供环境感知决策（复杂场景关键），ego分支提供惯性运动先验（简单场景高效），融合模块按场景复杂度自适应加权
3. **BEV蒸馏设计精巧**：解决了解耦带来的运动模糊副作用，teacher-student范式自然且有效
4. **自回归建图的因果一致性**：从world model思想出发建立规划→感知反馈回路

## 局限与展望

1. 仅在nuScenes开环评估为主，闭环性能的验证深度不足
2. 双分支架构增加了模型复杂度，FPS从VAD的3.4降至3.0
3. 场景驱动分支完全去除ego status可能过于激进，部分场景（如高速公路匀速行驶）ego status是有用信息
4. 未讨论与LiDAR融合的情况
5. 可探索将decoupling思想扩展到generative world model中

## 相关工作与启发

- **规划导向端到端**：UniAD（CVPR 2023）→ VAD（ICLR 2024）→ SparseDrive → BridgeAD
- **因果混淆缓解**：EgoStatus分析（Li 2024）、PLUTO对比模仿学习（Cheng 2024）
- **多传感器融合启发**：从sensor fusion转向decision context fusion是概念创新
- **World model反馈**：Think2Drive、Vista的自回归思想
- **启发**：decoupling + adaptive fusion的范式可推广到多模态决策和人机混合驾驶

## 评分

| 维度 | 分数 | 说明 |
|------|------|------|
| 新颖性 | ★★★★★ | 架构级解决因果混淆，双分支解耦思路开创性 |
| 技术深度 | ★★★★☆ | 路径注意力+蒸馏+自回归建图组合完整 |
| 实验质量 | ★★★★★ | 开环+闭环+ego噪声压力测试+场景分类评估 |
| 写作质量 | ★★★★★ | 问题定义精准，动机论述有力，结构清晰 |
| 实用价值 | ★★★★☆ | 组件可插拔（已验证），但双分支增加复杂度 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Percept-WAM: Perception-Enhanced World-Awareness-Action Model for Robust End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/percept-wam_perception-enhanced_world-awareness-action_model_for_robust_end-to-e.md)
- [\[AAAI 2026\] Rethinking the Spatio-Temporal Alignment of End-to-End 3D Perception](rethinking_the_spatio-temporal_alignment_of_end-to-end_3d_perception.md)
- [\[CVPR 2026\] ActiveAD: Planning-Oriented Active Learning for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/activead_planning-oriented_active_learning_for_end-to-end_autonomous_driving.md)
- [\[CVPR 2026\] ResAD: Normalized Residual Trajectory Modeling for End-to-End Autonomous Driving](../../CVPR2026/autonomous_driving/resad_normalized_residual_trajectory_modeling_for_end-to-end_autonomous_driving.md)
- [\[ICLR 2026\] ResWorld: Temporal Residual World Model for End-to-End Autonomous Driving](../../ICLR2026/autonomous_driving/resworld_temporal_residual_world_model_for_end-to-end_autonomous_driving.md)

</div>

<!-- RELATED:END -->
