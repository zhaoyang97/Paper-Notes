---
title: >-
  [论文解读] CarFormer: Self-Driving with Learned Object-Centric Representations
description: >-
  [ECCV2024][自动驾驶][object-centric learning] 提出 CarFormer，首次将自监督 slot attention 学到的 object-centric 表征用于自动驾驶，在 CARLA Longest6 基准上超越了使用精确物体属性的 PlanT，同时具备世界模型预测未来状态的能力。
tags:
  - "ECCV2024"
  - "自动驾驶"
  - "object-centric learning"
  - "注意力机制"
  - "self-driving"
  - "bird's eye view"
  - "Transformer"
---

# CarFormer: Self-Driving with Learned Object-Centric Representations

**会议**: ECCV2024  
**arXiv**: [2407.15843](https://arxiv.org/abs/2407.15843)  
**代码**: [https://kuis-ai.github.io/CarFormer/](https://kuis-ai.github.io/CarFormer/)  
**领域**: 自动驾驶  
**关键词**: object-centric learning, slot attention, self-driving, bird's eye view, autoregressive transformer

## 一句话总结

提出 CarFormer，首次将自监督 slot attention 学到的 object-centric 表征用于自动驾驶，在 CARLA Longest6 基准上超越了使用精确物体属性的 PlanT，同时具备世界模型预测未来状态的能力。

## 背景与动机

- 自动驾驶中场景表征的选择至关重要。BEV (Bird's Eye View) 近年表现突出，但维度仍然很高——大部分像素属于道路区域，车辆仅占 BEV 的很小部分，却是导致违规的主要原因
- 已有 object-centric 方法（如 PlanT）使用精确物体属性向量（位置、大小、朝向、速度），但这些属性需要人工指定，可能不完备且难以泛化到多种物体类型
- Slot attention 等自监督方法已经能在合成场景中有效地将场景分解为物体，但在复杂驾驶序列上仍是挑战。BEV 序列因其类似合成数据的特性，为 slot extraction 提供了可行的输入空间
- 核心动机：能否用自监督学习的 slot 表征替代手工指定的属性向量，让模型自动从时空上下文中学到驾驶所需的物体信息（位置、速度、朝向等）？

## 核心问题

1. 如何从 BEV 驾驶序列中自监督地提取 object-centric slot 表征？
2. 基于 slot 表征如何建模场景动态并学习驾驶策略？
3. Slot 表征能否同时支持动作预测和未来状态预测（世界模型）？

## 方法详解

### 整体框架：两阶段流水线

**第一阶段：Slot 提取（SAVi）**

- 采用冻结的 SAVi (Slot Attention for Video) 模型从 BEV 序列中提取 slot 表征
- 给定过去 T 个时间步的 BEV 帧，CNN 编码器处理每帧得到视觉特征 $\mathbf{h}_i$
- 初始化 K 个 slot 向量，通过 Slot Attention 机制更新：$\mathcal{Z}_i = f_{SA}(\tilde{\mathcal{Z}_i}, \mathbf{h}_i)$
- 时间一致性通过预测器维护：$\mathcal{Z}_{i+1} = f_{pred}(\mathcal{Z}_i)$
- 为提升 slot extraction 质量的关键技巧：(1) 给不同车辆分配不同颜色；(2) 放大小型车辆（摩托车、自行车等）至 $4.9m \times 2.12m$；(3) 使用轻量化解码器以支持更多 slot 数量

**第二阶段：CarFormer 行为学习**

- 基于 GPT-2 架构的自回归 Transformer 解码器
- 轨迹定义为混合模态的 token 序列：$\tau_t = \{g_t^x, g_t^y, l_t, v_t, \mathbf{z}_t^1, \dots, \mathbf{z}_t^K, \mathbf{r}_t^1, \mathbf{r}_t^2, q_t^1, \dots, q_t^{2W}\}$
    - 目标点 $(g_t^x, g_t^y)$、红绿灯标志 $l_t$、车速 $v_t$（离散化，k-means 量化）
    - K 个 slot 特征 $\mathbf{z}_t^i \in \mathbb{R}^{1 \times d}$（连续，MLP 投影）
    - 路线向量 $\mathbf{r}_t^1, \mathbf{r}_t^2 \in \mathbb{R}^6$（连续，MLP 投影）
    - 量化 waypoint $q_t^i$（离散，embedding lookup）

### Block Attention 机制

- 将标准因果注意力掩码替换为 block 三角掩码
- Slot 特征和路线向量作为一个 block，block 内部允许双向交叉注意力
- 使所有物体和路线可以充分交互，更好地建模场景动态

### 双头动作预测

- **GRU 头**：取 backbone 最后一个隐向量，拼接红绿灯标志，自回归预测 W 个连续 waypoint
- **Quantization 头**：将 waypoint 离散化为 token，作为 next-token prediction 问题
- 实验表明 GRU 头效果显著优于量化头

### 训练损失

- 总损失：$\mathcal{L} = \mathcal{L}_{wp} + \alpha \mathcal{L}_{forecast}$
- $\mathcal{L}_{wp} = \mathcal{L}_{GRU} + \mathcal{L}_{LM}$：GRU 的 L1 损失 + 量化 waypoint 的交叉熵
- $\mathcal{L}_{forecast}$：预测未来 slot 表征的 MSE 损失，$\alpha = 40$
- 模态感知编码器统一连续和离散输入到相同隐藏维度 $H = 768$

## 实验关键数据

### CARLA Longest6 基准对比

| 模型 | 表征类型 | DS↑ | IS↑ | RC↑ |
|------|---------|-----|-----|-----|
| AIM-BEV | Scene (BEV) | 45.06±1.68 | 0.55±0.01 | 78.31±1.12 |
| ROACH | Scene (BEV) | 55.27±1.43 | 0.62±0.02 | 88.16±1.52 |
| PlanT | Attributes | 73.36±2.97 | **0.84±0.01** | 87.03±3.91 |
| CarFormer | Attributes | 71.53±3.52 | 0.78±0.06 | 90.01±1.60 |
| **CarFormer** | **Slots** | **74.89±1.44** | 0.79±0.02 | **92.90±1.28** |

- Slot 表征的 DS 最高且方差最低（±1.44 vs PlanT ±2.97），表明鲁棒性优势
- RC 高达 92.90%，远超 PlanT 的 87.03%

### 消融实验

- 移除 Block Attention：DS 从 74.89 降至 70.42
- 移除 Forecasting：IS 从 0.79 暴降至 0.63，DS 降至 57.25（最大影响）
- 移除 Creeping：RC 从 92.90 降至 80.52
- GRU vs 量化头：GRU 的 DS = 74.89 远高于量化头的 66.87

### Slot 数量与放大的影响

- 7 slots → 30 slots：DS 从 48.17 提升到 71.48（无放大）/ 从 62.93 到 74.89（有放大）
- 放大小型车辆对少量 slot 设置提升尤为显著（7 slots：48.17→62.93）

### 未来状态预测

- CarFormer 预测 t+1：ARI=0.795, mIoU=0.702（优于 Input-Copy 的 0.641/0.561）
- CarFormer 预测 t+4：ARI=0.540, mIoU=0.454（大幅优于 Input-Copy 的 0.412/0.375）

## 亮点

1. **首次将自监督 slot 表征用于自动驾驶**：无需手工指定物体属性，slot 能自动从时空上下文中隐式编码位置、朝向、速度等驾驶关键信息
2. **驾驶与世界模型联合训练**：forecasting 辅助任务不仅提供额外监督，还让智能体能预判其他车辆意图，IS 提升 0.17
3. **低方差高鲁棒性**：slot 表征的跨运行方差仅为 PlanT 的一半，说明对场景变化更稳定
4. **工程细节扎实**：车辆着色、小型车辆放大、轻量化解码器等技巧有效解决了 slot extraction 在驾驶场景中的实际困难

## 局限与展望

1. **依赖 ground truth BEV**：当前假设可获取真实 BEV 地图，实际部署需从相机图像估计 BEV，级联误差会影响 slot 质量
2. **SAVi 的感知瓶颈**：slot extraction 的质量直接约束下游预测，SAVi 在转弯场景中的模糊预测和拥挤场景中的漏检会传播到 CarFormer
3. **仅限模仿学习**：目前为单步策略的 imitation learning，未利用自回归架构做多步推理或引入强化学习的奖励信号
4. **幻觉问题**：world model 预测未来时会出现 false positive（幻觉车辆），在复杂多车场景中动态预测仍有困难
5. **泛化性待验**：实验仅在 CARLA 模拟器中验证，真实世界驾驶场景的复杂度远超 BEV 中的合成序列

## 与相关工作的对比

| 维度 | PlanT | AIM-BEV / ROACH | CarFormer |
|------|-------|-----------------|-----------|
| 表征 | 精确物体属性向量 | 场景级 BEV | 自监督 slot 表征 |
| 信息获取 | 需要精确位置/速度/朝向 | 全局像素 | 自动从 BEV 序列学习 |
| 架构 | Transformer Encoder | CNN / RL | Autoregressive Transformer Decoder |
| 世界模型 | 有（属性预测） | 无 | 有（slot 预测） |
| 扩展性 | 受限于预定义属性集 | 维度过高 | 可泛化到任何可被 slot 捕获的物体 |
| DS 表现 | 73.36 | 45.06 / 55.27 | **74.89** |

## 启发与关联

- **Object-centric representations 在其他领域的潜力**：slot attention 的成功暗示它可以用于其他需要物体级推理的任务（机器人操作、视频理解）
- **BEV 作为中间表征的桥梁作用**：BEV 的"类合成"特性使真实场景中难以直接应用的 slot extraction 方法变得可行，值得探索更多 BEV 上的自监督方法
- **联合预测与规划**：forecasting 带来的巨大性能提升（IS +0.17）表明，将预测和规划联合建模是自动驾驶的重要范式
- **可与端到端方法结合**：未来可探索从相机直接提取 BEV slot，跳过显式 BEV 估计，减少级联误差

## 评分

- 新颖性: 8/10 — 首次在自动驾驶中使用自监督 slot 表征，思路清晰且有说服力
- 实验充分度: 8/10 — 消融实验全面，涵盖表征类型、注意力机制、动作头、slot 数量等多个维度
- 写作质量: 7/10 — 结构清晰，但部分数学符号较密集，可视化分析有待加强
- 价值: 7/10 — 在特权设置下验证了 slot 的有效性，但依赖 GT BEV 限制了实际影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Equivariant Spatio-Temporal Self-Supervision for LiDAR Object Detection](equivariant_spatio-temporal_self-supervision_for_lidar_object_detection.md)
- [\[NeurIPS 2025\] Self-Supervised Learning of Graph Representations for Network Intrusion Detection](../../NeurIPS2025/autonomous_driving/self-supervised_learning_of_graph_representations_for_network_intrusion_detectio.md)
- [\[ECCV 2024\] FSD-BEV: Foreground Self-Distillation for Multi-View 3D Object Detection](fsd-bev_foreground_self-distillation_for_multi-view_3d_object_detection.md)
- [\[CVPR 2025\] UniScene: Unified Occupancy-centric Driving Scene Generation](../../CVPR2025/autonomous_driving/uniscene_unified_occupancy-centric_driving_scene_generation.md)
- [\[ECCV 2024\] SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving](seflow_a_self-supervised_scene_flow_method_in_autonomous_driving.md)

</div>

<!-- RELATED:END -->
