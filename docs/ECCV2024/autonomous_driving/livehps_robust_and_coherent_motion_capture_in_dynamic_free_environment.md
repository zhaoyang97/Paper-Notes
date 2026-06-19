---
title: >-
  [论文解读] LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment
description: >-
  [ECCV 2024][自动驾驶][LiDAR动作捕捉] 提出 LiveHPS++，一种基于单 LiDAR 的鲁棒人体动作捕捉方法，通过轨迹引导身体追踪器、噪声不敏感速度预测器和运动学感知姿态优化器三个模块，隐式和显式建模人体运动的动力学和运动学特征，在复杂噪声环境下实现精确且连贯的全局人体运动捕捉。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "LiDAR动作捕捉"
  - "抗噪声"
  - "运动连贯性"
  - "SMPL"
  - "运动学优化"
---

# LiveHPS++: Robust and Coherent Motion Capture in Dynamic Free Environment

**会议**: ECCV 2024  
**arXiv**: [2407.09833](https://arxiv.org/abs/2407.09833)  
**代码**: [有 (项目页面)](https://4dvlab.github.io/project_page/LiveHPS2.html)  
**领域**: 自动驾驶  
**关键词**: LiDAR动作捕捉, 抗噪声, 运动连贯性, SMPL, 运动学优化

## 一句话总结

提出 LiveHPS++，一种基于单 LiDAR 的鲁棒人体动作捕捉方法，通过轨迹引导身体追踪器、噪声不敏感速度预测器和运动学感知姿态优化器三个模块，隐式和显式建模人体运动的动力学和运动学特征，在复杂噪声环境下实现精确且连贯的全局人体运动捕捉。

## 研究背景与动机

**领域现状**：在大规模动态环境中精确捕捉人体运动对数字电影、AR/VR、机器人等下游应用至关重要。LiDAR 因其远距离深度感知能力、不受光照条件限制的特性，成为户外大场景动作捕捉的理想传感器。近年来，LiDARCap、MOVIN 等方法已展示了 LiDAR 动作捕捉的可行性。

**现有方法的痛点**：

**仅适用于干净数据**：LiDARCap、PointHPS 等方法依赖于干净分割的人体点云输入，在实际场景中表现急剧下降——当人与人近距离接触或与物体交互时，上游感知算法的分割结果往往包含大量噪声点

**噪声特征等同处理**：LiveHPS 虽然考虑了噪声干扰，但其网络将真实人体点和噪声点的特征同等对待，严重噪声会显著影响估计精度

**缺乏全局运动连贯性**：现有方法仅考虑关节间的交互，忽略全局运动学信息，导致预测的全局姿态和轨迹出现抖动和不连贯

**合成数据不足**：现有合成数据集仅模拟随机噪声和随机平移，未准确反映人-物交互等真实场景的复杂噪声模式

**核心矛盾**：如何在实际部署中面对严重噪声干扰（来自物体遮挡、人群靠近、分割错误），仍保证动作捕捉结果的精确性和时序连贯性？

**本文切入角度**：从动力学（dynamic）和运动学（kinematic）两个维度建模人体运动特征——通过轨迹信息恢复归一化后丢失的全局运动动态，通过速度预测和运动学优化显式排除噪声影响并强化时序连贯性。同时构建 NoiseMotion 合成数据集模拟复杂的人-物交互噪声场景。

## 方法详解

### 整体框架

LiveHPS++ 接收序列化的含噪点云，输出序列化的 SMPL 参数（姿态 $\theta$、体型 $\beta$、平移 $\mathbf{T}$）。流水线由三个核心模块串联：(1) 轨迹引导身体追踪器 → 预测关节位置和平移；(2) 噪声不敏感速度预测器 → 回归每个关节的速度，消除噪声影响；(3) 运动学感知姿态优化器 → 利用速度信息优化姿态的精度和连贯性。最终通过 SMPL solver 回归人体参数。

输入点云经最远点采样固定为 $N_{input}=256$ 点，减去均值位置 $\mathbf{Loc}(t)$ 进行归一化。SMPL 模型定义 $N_J=24$ 个关节和 $N_V=6890$ 个网格顶点。

### 关键设计

1. **轨迹引导身体追踪器（Trajectory-guided Body Tracker, TBT）**

   **问题**：传统逐帧归一化（减去每帧均值位置）在噪声数据下会导致相邻帧间的点云分布剧烈波动，破坏轨迹的空间连续性；而序列归一化（减去首帧均值）虽保留物理轨迹但牺牲了精度。

   **解决方案**：设计专用编码器捕获轨迹嵌入（trajectory embedding），隐式建模人体运动的动态特征。采用**顶点-轨迹引导的自适应蒸馏机制**：
    - 引导网络：以 GT 顶点采样 $\mathbf{V}_{pc}^{GT}(t)$ 和 GT 轨迹 $\mathbf{Traj}^{GT}(t)$ 为输入
    - 学习网络：以输入点云 $\mathbf{PC}(t)$ 和计算轨迹 $\mathbf{Traj}(t) = \mathbf{Loc}(t) - \mathbf{Loc}(1)$ 为输入
    - 通过 KL 散度蒸馏损失 $\mathcal{L}_{distillation}$ 使学习网络逼近引导网络的特征分布
    - 增加平移预测分支

   损失函数：$\mathcal{L}_{TBT} = \lambda_1 \mathcal{L}_{distillation} + \lambda_2 \mathcal{L}_{mse}(\mathbf{J}_{pc}) + \lambda_3 \mathcal{L}_{mse}(\hat{\mathbf{T}}_{pc})$，其中 $\lambda_1=10^3, \lambda_2=1, \lambda_3=1$。推理时不需要引导网络。

2. **噪声不敏感速度预测器（Noise-insensitive Velocity Predictor, NVP）**

   **问题**：TBT 基于骨架的父子关节结构进行回归，当父关节因噪声而偏移时，误差会沿骨架链累积传播。

   **核心思路**：利用**交叉注意力机制**让每个关节去搜索原始点云中真正有价值的点特征（而非被噪声污染的特征），预测每个全局关节的速度 $\mathbf{K}(n)$ 和平移速度 $\mathbf{K}_{ts}$（时间窗口 $L=32$）：

    $\mathcal{L}_{mse}(\mathbf{K}(n)) = \sum_n \|\mathbf{K}(n) - \mathbf{K}^{GT}(n)\|_2^2$

   其中速度 GT 定义为相邻帧关节位置差：$\mathbf{K}^{GT}(n,t) = \mathbf{J}^{GT}(n,t+1) - \mathbf{J}^{GT}(n,t)$。

   **设计动机**：速度信息反映运动学特征，通过交叉注意力让网络学会区分真实人体点和噪声点，从而在特征层面消除噪声影响。

3. **运动学感知姿态优化器（Kinematic-aware Pose Optimizer, KPO）**

   **核心思路**：利用预测的速度和关节位置生成候选关节：

    $\mathbf{J}_{cds}(n_i, t_i, t_j) = \mathbf{J}_{pc}(n_i, t_i) + \Delta t \sum_{t=t_i}^{t_j} \mathbf{K}(n_i, t)$

   每帧可生成 $(L-1)$ 个候选关节。使用交叉注意力架构在候选关节和原始输入之间建立连接，综合短期和长期运动学信息进行关节矫正，输出连贯且精确的全局关节 $\mathbf{J}_c(t)$ 和平移 $\mathbf{T}_c(t)$。

   **设计动机**：短期优化增强相邻帧连贯性但可能引入长序列抖动，长期优化保持整体连贯但累积误差大——KPO 同时考虑两者以实现精度和连贯性的平衡。

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \mathcal{L}_{TBT} + \mathcal{L}_{NVP} + \mathcal{L}_{KPO} + \mathcal{L}_{smpl}$

SMPL solver 损失：$\mathcal{L}_{smpl} = \lambda_6 \mathcal{L}_{mse}(\mathbf{J}_{smpl}) + \lambda_7 \mathcal{L}_{mse}(\mathbf{V}_{smpl}) + \lambda_8 \mathcal{L}_{mse}(\theta) + \lambda_9 \mathcal{L}_{mse}(\beta)$，其中 $\lambda_6 = 100/N_J, \lambda_7=100/N_V, \lambda_8=1/5, \lambda_9=1$。

训练配置：PyTorch 1.10.0 + CUDA 11.4，200 epochs，batch size 64，序列长度 32，学习率 $10^{-3}$，4×NVIDIA A40 GPU。训练数据包括 FreeMotion、Sloper4D 和自建的 NoiseMotion 数据集。

**NoiseMotion 数据集**：基于 SURREAL（1,021,802 人体运动）和 ShapeNet（51,300 个3D物体模型）构建，模拟人-物交互的动态/静态噪声，远比 FreeMotion-OBJ 的不到10种动态物体丰富。

## 实验关键数据

### 主实验

**与 SOTA 方法在多数据集上的对比（Table 1）：**

| 方法 | NoiseMotion J/V Err(PST)↓ | NoiseMotion Jitter↓ | FreeMotion-OBJ J/V Err(PST)↓ | FreeMotion-OBJ Jitter↓ |
|---|---|---|---|---|
| LiDARCap | 400.66/402.58 | 765.89 | 181.82/189.32 | 62.47 |
| LIP | 192.79/198.66 | 451.74 | 158.38/170.90 | 60.19 |
| LiveHPS | 74.70/83.84 | 68.65 | 146.78/158.00 | 117.79 |
| LiveHPS* | 561.49/611.40 | 884.24 | 133.82/146.12 | 100.82 |
| **Ours** | **58.53/64.51** | **59.35** | **128.60/136.94** | **30.96** |

**FreeMotion 和 Sloper4D 数据集：**

| 方法 | FreeMotion J/V Err(PST)↓ | FreeMotion Jitter↓ | Sloper4D J/V Err(PST)↓ | Sloper4D Jitter↓ |
|---|---|---|---|---|
| LiveHPS | 130.41/141.08 | 85.38 | 88.35/95.85 | 73.56 |
| LiveHPS* | 119.22/128.55 | 86.07 | 77.73/85.83 | 97.41 |
| **Ours** | **112.13/120.39** | **33.16** | **76.98/81.67** | **59.97** |

在最具挑战性的 FreeMotion-OBJ 上：全局顶点误差降低 **6.28%**，抖动降低 **69.29%**。在 NoiseMotion 上：全局顶点误差降低 **23.05%**，抖动降低 **13.54%**。

### 消融实验

**FreeMotion-OBJ 上的模块消融（Table 2）：**

| 配置 | J/V Err(PS)↓ | Jitter↓ | 说明 |
|---|---|---|---|
| w/o TBT | 71.68/90.08 | 33.89 | 去掉轨迹模块→精度大幅下降 |
| w/o NVP & KPO | 68.37/84.92 | 71.82 | 去掉速度+优化→连贯性崩溃 |
| Frame-wise normalization | 68.57/86.15 | 43.53 | 传统逐帧归一化 |
| Sequence-wise normalization | 85.42/106.91 | 31.32 | 序列归一化→精度差但连贯 |
| Short-term optimizer | 64.04/79.27 | 55.50 | 仅短期优化→长序列抖动 |
| Long-term optimizer | 68.06/83.62 | 42.51 | 仅长期优化→累积误差 |
| **完整 LiveHPS++** | **58.11/72.55** | **30.96** | 全部模块配合最优 |

### 关键发现

- **TBT 和 KPO 互补**：TBT 主要贡献精度（w/o TBT 精度大降），KPO 主要贡献连贯性（w/o NVP&KPO 抖动从30.96飙升到71.82）
- **轨迹引导归一化兼顾精度和连贯性**：逐帧归一化精度好但不连贯，序列归一化连贯但不精确，TBT 兼得两者优势
- **时间窗口32最优**：窗口从1增大到32，抖动持续降低，精度略有提升
- **NoiseMotion数据量越多越好**：逐步增加合成数据比例（0%→100%），性能持续改善，证明合成噪声数据的价值
- **LiveHPS 在 NoiseMotion 上训练后泛化不稳定**（FreeMotion/Sloper4D 反而变差），而 LiveHPS++ 在所有数据集上均达到 SOTA

## 亮点与洞察

1. **动力学+运动学的双重建模**：TBT 隐式建模动态特征（通过轨迹嵌入），KPO 显式建模运动学特征（通过速度预测和候选关节生成），两者互补
2. **交叉注意力消噪**：NVP 中让关节"主动搜索"有价值的点特征，而非被动接收所有点（包括噪声），网络学会区分人体点和噪声点
3. **归一化策略的深层洞察**：揭示了逐帧/序列归一化的精度-连贯性权衡，轨迹嵌入是解决这一矛盾的优雅方案
4. **NoiseMotion 数据集**：利用 SURREAL+ShapeNet 大规模模拟真实人-物交互噪声，弥补了现有合成数据的空白

## 局限与展望

- 仅针对单人场景，多人同时动作捕捉尚未解决
- 依赖上游感知算法提供（含噪的）人体点云分割结果，端到端方案可能更鲁棒
- NoiseMotion 的物体噪声分布仍可能与真实场景有差距（合成 vs 真实域差距）
- 时间窗口固定为32帧，自适应窗口可能在不同运动场景下更灵活
- 未讨论实时性能指标（推理速度）

## 相关工作与启发

- **LiveHPS**：本文的直接前身，提出场景级人体姿态估计和顶点引导蒸馏机制，但对噪声和连贯性处理不足
- **LiDARCap / LIP**：早期 LiDAR 动作捕捉方法，分别用图卷积和稀疏IMU，但仅适用于干净环境
- **PointHPS**：级联网络从密集点云估计姿态，不适合户外稀疏场景
- **启发**：在噪声环境中，运动学信息（速度、加速度）是比静态姿态更鲁棒的特征——这一思想可推广到其他序列估计任务

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 动力学+运动学双重建模框架设计巧妙，轨迹引导归一化和候选关节生成都是有洞察力的设计
- **实验充分度**: ⭐⭐⭐⭐⭐ — 4个数据集（含自建 NoiseMotion），5项指标，丰富的定量/定性对比和详尽消融
- **写作质量**: ⭐⭐⭐⭐ — 问题动机清晰，每个模块的设计理由解释充分，消融分析有深度
- **价值**: ⭐⭐⭐⭐ — 显著推进了 LiDAR 动作捕捉在真实噪声环境下的实用性，FreeMotion-OBJ 抖动降低 69% 具有很强的应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Multimodal Data Fusion to Capture Dynamic Interactions between Built Environment and Vulnerable Older Adults](../../AAAI2026/autonomous_driving/multimodal_data_fusion_to_capture_dynamic_interactions_between_built_environment.md)
- [\[ECCV 2024\] DySeT: A Dynamic Masked Self-distillation Approach for Robust Trajectory Prediction](dyset_a_dynamic_masked_self-distillation_approach_for_robust_trajectory_predicti.md)
- [\[ECCV 2024\] Train Till You Drop: Towards Stable and Robust Source-free Unsupervised 3D Domain Adaptation](train_till_you_drop_towards_stable_and_robust_source-free_unsupervised_3d_domain.md)
- [\[CVPR 2026\] Bezier Degradation Modeling for LiDAR-based Human Motion Capture](../../CVPR2026/autonomous_driving/bezier_degradation_modeling_for_lidar-based_human_motion_capture.md)
- [\[ECCV 2024\] Rethinking Data Augmentation for Robust LiDAR Semantic Segmentation in Adverse Weather](rethinking_data_augmentation_for_robust_lidar_semantic_segmentation_in_adverse_w.md)

</div>

<!-- RELATED:END -->
