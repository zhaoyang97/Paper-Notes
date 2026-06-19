---
title: >-
  [论文解读] Neural Augmented Kalman Filters for Road Network Assisted GNSS Positioning
description: >-
  [ICML 2025][遥感][GNSS positioning] 提出用时序图神经网络（TGNN）将开源道路网络信息集成到 GNSS 卡尔曼滤波中——TGNN 在图结构上预测最可能的道路段并动态估计其不确定性，在真实城市数据中 P95 定位误差从 77.23m 降至 55.02m（降幅 29%）。
tags:
  - "ICML 2025"
  - "遥感"
  - "GNSS positioning"
  - "Kalman filter"
  - "road network"
  - "图神经网络"
  - "urban localization"
---

# Neural Augmented Kalman Filters for Road Network Assisted GNSS Positioning

**会议**: ICML 2025  
**arXiv**: [2507.00654](https://arxiv.org/abs/2507.00654)  
**代码**: 待确认  
**领域**: 遥感  
**关键词**: GNSS positioning, Kalman filter, road network, temporal graph neural network, urban localization

## 一句话总结

提出用时序图神经网络（TGNN）将开源道路网络信息集成到 GNSS 卡尔曼滤波中——TGNN 在图结构上预测最可能的道路段并动态估计其不确定性，在真实城市数据中 P95 定位误差从 77.23m 降至 55.02m（降幅 29%）。

## 研究背景与动机

**领域现状**：GNSS 提供全球定位能力，2023 年约 56 亿接收器中 10% 用于道路交通。然而在密集城市中，高楼反射（多径效应）和遮挡（非视线误差）导致定位误差可达 20 米，即使使用 3D 模型也难低于 10 米。车道级应用（导航、配送）需 2-10 米精度。

**现有痛点**：额外传感器（IMU/LiDAR）成本高或不可用。OpenStreetMap 等开源道路网络是全球免费的替代信息源，但其观测本质是多模态高斯（多条候选道路），不能直接集成到假设单峰的 KF 中。现有方案先用启发式或 HMM+Viterbi 选择单一道路段再构建高斯观测，但：(1) Viterbi 仅用历史信息不看未来，噪声场景下选错路；(2) 道路观测的标准差用经验值固定，不能适应变化场景。

**核心矛盾**：需要选择正确道路段并根据置信度动态调整权重，但现有方法在选择和权重两方面都不够灵活。

**本文目标** (1) 可学习的道路段选择替代启发式/Viterbi；(2) 动态预测道路观测不确定性实现自适应 KF 更新。

**切入角度**：双向 Viterbi（Oracle）能利用未来信息做更好选择，但在线不可用。训练 TGNN 在仅用历史信息时逼近 Oracle 质量。

**核心 idea**：用 TGNN 在道路图上学习时序-空间联合推理，预测道路段概率和不确定性，端到端增强 KF 定位。

## 方法详解

### 整体框架

每个时间步：(1) GNSS 伪距 → KF 预测和 GNSS 更新得到 $\mathbf{x}_{\text{GNSS}}^+$；(2) 以当前位置为中心取 50m 半径道路子图 → TGNN 预测每段概率和标准差；(3) 选最可能道路段，用预测标准差构建高斯观测 → KF 道路量测更新得到最终 $\mathbf{x}_{\text{RN}}^+$。

KF 状态 8 维：3D 位置 + 3D 速度 + 时钟偏差/漂移。道路量测更新：$\mathbf{x}_{\text{RN}}^+ = \mathbf{x}_{\text{GNSS}}^+ + \mathbf{K}(\mathbf{z} - \mathbf{H}\mathbf{x}_{\text{GNSS}}^+)$，$\mathbf{z}$ 是选中道路段位置和航向，$\mathbf{V}$ 是 TGNN 预测的协方差。

### 关键设计

1. **TGNN 架构**:

    - 功能：同时处理道路图空间拓扑和用户轨迹时序信息
    - 核心思路：双路径处理用户级和道路级特征，通过 $L$ 个重复块：(a) 特征变换——两个 MLP 分别投影；(b) 局部消息传递——LSTM 捕捉轨迹时序（$x \to x$），GCN 在道路图上传播邻域信息（$r \to r$）；(c) 交叉消息传递——用户特征与道路均值池化交互（$x,r \to x$），用户特征与每段拼接交互（$r,x \to r$）。最后线性层+softmax 得道路概率
    - 设计动机：GCN 考虑图连通性（相邻道路上下文），LSTM 利用历史轨迹（速度/航向变化），交叉传递让两类信息多尺度融合

2. **标准差预测头**:

    - 功能：动态估计道路观测不确定性
    - 核心思路：在 TGNN 最后一层用户特征上加线性投影头，输出道路平行 $\sigma_\parallel^2$ 和垂直 $\sigma_\perp^2$ 方差，经 exp 保证正值。不确定时输出大方差→KF 更依赖 GNSS；确信时输出小方差→道路权重提高
    - 设计动机：固定标准差在所有场景下同一权重融合道路信息过于僵化。端到端优化 MSE 损失使网络学会场景自适应

3. **Oracle 监督与训练目标**:

    - 功能：为 TGNN 提供高质量训练标签
    - 核心思路：双向 Viterbi（利用过去+未来位置）作为 Oracle 得最优道路序列。训练损失两项：(a) 交叉熵 $L_{\text{CE}} = \text{CE}(r_{\text{oracle}}^*, P_\phi(\mathbf{r}))$——学模拟 Oracle；(b) MSE $L_{\text{MSE}} = \text{MSE}(\mathbf{x}_{\text{gt}}, \mathbf{x}_{\text{RN}}^+)$——端到端优化 KF 输出定位误差。总损失 $L = L_{\text{CE}} + \lambda L_{\text{MSE}}$
    - 设计动机：KF 全可微，标准差预测头可通过 KF 输出误差直接反向传播优化

### 输入特征

道路段：端点坐标、段长、车道数、限速、道路类型、单行道标记。用户：KF 状态均值/不确定性、Viterbi 概率先验。缺失信息用默认值。

## 实验关键数据

### 定位误差对比（真实世界 4 城市数据，3 折 CV，Table 1）

| 方法 | 道路选择 | 标准差 | HE@50th (m)↓ | HE@95th (m)↓ |
|------|---------|-------|-------------|-------------|
| 最小二乘 (LS) | — | — | 20.43 | 115.97 |
| GNSS-only KF | — | — | 10.75 | 77.23 |
| KF + Instant | 最近距离 | 网格搜索 | 7.96 | 68.86 |
| KF + Viterbi | HMM | 网格搜索 | 8.02 | 68.27 |
| **KF + TGNN** | **TGNN** | **TGNN** | **8.74±0.15** | **55.02±2.21** |
| KF + Oracle | 双向 Viterbi | 0 | 3.72 | 11.40 |

### 消融实验（Table 2-3）

| 消融项 | HE@50th (m) | HE@95th (m) |
|--------|-------------|-------------|
| TGNN + TGNN 标准差 | 8.74±0.15 | **55.02±2.21** |
| Viterbi + TGNN 标准差 | 8.69±0.18 | 67.08±3.49 |
| TGNN + 网格搜索标准差 | 8.36±0.38 | 63.72±9.00 |
| MLP（无 GCN 无 LSTM） | 9.13±0.31 | 59.68±3.94 |
| GNN（无 LSTM） | 8.82±0.17 | 56.90±2.83 |

### 关键发现

1. TGNN 将 P95 误差从 77.23m 降至 55.02m（**降幅 29%**），相比 Viterbi 再降 13m
2. **标准差预测头贡献关键**：替换为网格搜索后 P95 从 55.02 退化到 63.72（+8.7m）
3. LSTM 贡献约 2m（GNN→TGNN），GCN 贡献约 3m（MLP→GNN）
4. 视野范围敏感：太小（<30m）错过正确道路，太大（>80m）候选过多，最优约 50m
5. 10 个随机初始化标准差仅 ±0.15/±2.21m，方法鲁棒
6. P50 TGNN(8.74) 略逊 Viterbi(8.02)，但 P95 大幅领先——困难场景获益最大

## 亮点与洞察

- **端到端可微 KF**：标准差预测头通过 KF 输出定位误差直接反向传播，让网络学会"何时信任道路信息"
- **Oracle 监督巧妙**：用双向 Viterbi（数据层面可计算）作为训练标签，无需额外标注
- **全球适用性**：仅依赖 OpenStreetMap，无需商业地图或额外传感器
- **KF 框架保持工程特性**：实时性、可解释性、与其他传感器可扩展融合

## 局限性

- 数据规模有限——仅 4 城市，跨城市/跨国泛化未充分验证
- 仅用道路中心线，缺车道级信息导致平行多车道场景存在系统偏移
- 未集成 IMU——极差 GNSS 信号时错误道路选择导致 KF 持续偏移
- 推理延迟未量化
- 不处理多层道路结构（高架桥与地面重叠）

## 相关工作与启发

- Revach et al. (2022) KalmanNet：学习最优卡尔曼增益但不处理道路网络
- Jalalirad et al. (2023)：用 GNN 估计伪距误差，本文使用其数据集
- Map matching（Hu et al., 2023）：后处理确定道路序列，不提供实时位置改善
- 启发：TGNN+KF 范式可推广到其他地理先验辅助定位（建筑高度图、地形模型等）

## 评分

⭐⭐⭐⭐ — 首个深度学习+道路网络+GNSS KF 的端到端方案。标准差预测头简洁有效。P95 困难场景 29% 改善有工程意义。遗憾是数据规模偏小、未集成 IMU。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Information-Bottleneck Driven Binary Neural Network for Change Detection](../../ICCV2025/remote_sensing/information-bottleneck_driven_binary_neural_network_for_change_detection.md)
- [\[CVPR 2026\] Beyond Endpoints: Path-Centric Reasoning for Vectorized Off-Road Network Extraction](../../CVPR2026/remote_sensing/beyond_endpoints_path-centric_reasoning_for_vectorized_off-road_network_extracti.md)
- [\[CVPR 2026\] LNEM: Lunar Neural Elevation Model](../../CVPR2026/remote_sensing/lnem_lunar_neural_elevation_model.md)
- [\[CVPR 2026\] Spectrally Distilled Representations Aligned with Instruction-Augmented LLMs for Satellite Imagery](../../CVPR2026/remote_sensing/spectrally_distilled_representations_aligned_with_instruction-augmented_llms_for.md)
- [\[CVPR 2026\] RoadGIE: Towards A Global-Scale Aerial Benchmark for Generalizable Interactive Road Extraction](../../CVPR2026/remote_sensing/roadgie_towards_a_global-scale_aerial_benchmark_for_generalizable_interactive_ro.md)

</div>

<!-- RELATED:END -->
