---
title: >-
  [论文解读] A Data-Driven Model Predictive Control Framework for Multi-Aircraft TMA Routing Under Travel Time Uncertainty
description: >-
  [AAAI 2026][自动驾驶][终端区运行] 提出闭环 MPC 框架用于樟宜机场 50 海里半径终端区（TMA）的多飞机无冲突路径规划与调度，集成 XGBoost 预测 TMA 边界到达时间、MILP 优化（含路径选择/速度调整/等待控制/安全间隔约束）和滚动时域仿真器，在峰值 36 架/小时拥堵场景下实现 7 倍计算加速且 Monte Carlo 鲁棒性验证中可行性远优于 Dijkstra 基线。
tags:
  - "AAAI 2026"
  - "自动驾驶"
  - "终端区运行"
  - "STAR路径规划"
  - "旅行时间不确定性"
  - "模型预测控制"
  - "MILP"
---

# A Data-Driven Model Predictive Control Framework for Multi-Aircraft TMA Routing Under Travel Time Uncertainty

**会议**: AAAI 2026  
**arXiv**: [2511.19452](https://arxiv.org/abs/2511.19452)  
**领域**: 空中交通管理 / 终端区路径规划  
**关键词**: 终端区运行, STAR路径规划, 旅行时间不确定性, 模型预测控制, MILP

## 一句话总结

提出闭环 MPC 框架用于樟宜机场 50 海里半径终端区（TMA）的多飞机无冲突路径规划与调度，集成 XGBoost 预测 TMA 边界到达时间、MILP 优化（含路径选择/速度调整/等待控制/安全间隔约束）和滚动时域仿真器，在峰值 36 架/小时拥堵场景下实现 7 倍计算加速且 Monte Carlo 鲁棒性验证中可行性远优于 Dijkstra 基线。

## 研究背景与动机

**领域现状**：航班延误给乘客、航空公司和环境带来巨大成本。空中交通流量管理（ATFM）通常建模为动态多商品网络流问题，但终端区（TMA）——机场附近高密度空域——的优化文献较少。现有研究主要集中在跑道调度（ASP），对整体 TMA 空域优化关注不足。

**现有痛点**：
(1) **TMA 网络模型过于简化**——已有方法使用简化地图，航路点有限，无法表达真实 STAR（标准终端进场航路）的复杂结构；
(2) **缺乏闭环实时控制**——大多数方法采用一次性优化，无法适应实时系统的动态变化；
(3) **未考虑旅行时间不确定性**——等待和雷达引导在 TMA 中频繁发生，飞行时间方差大（TMA 内方差 204s vs 航路段 81s），但现有方法忽略了此不确定性对调度的影响。

**本文切入角度**：将预测、优化和仿真三个组件整合为闭环 MPC 框架——用数据驱动的 TMA 边界到达时间预测替代不可靠的跑道 ETA 预测，用完整 STAR 网络的 MILP 优化替代简化模型，用滚动时域策略实现实时计算和鲁棒控制。

## 方法详解

### 整体框架

系统由四个组件构成：历史数据库、XGBoost 流量预测器、MILP 实时 MPC 控制器和自定义交通仿真器。ADS-B 雷达数据和气象 METAR 数据作为输入，预测器给出 TMA 边界到达时间，MPC 在滚动窗口内优化着陆时间、路径和速度，优化指令应用于仿真器更新系统状态，迭代直至仿真完成。

### 关键设计

1. **数据驱动 TMA 边界到达时间预测**

    - **功能**：用 XGBoost 预测每架飞机到达 50NM TMA 边界的时间 $T_f^t = \mathcal{F}(X_f^t)$
    - **输入特征**：ADS-B 实时动态（经纬度/地速/距离）、跑道方向、出发机场代码、尾流等级、时间属性、METAR 气象信息（风速/风向/能见度/云层）
    - **设计动机**：预测 TMA 边界到达时间比直接预测跑道 ETA 更可靠——TMA 内部由于等待和引导操作，飞行时间方差极大（为航路段的 2.5 倍），而 TMA 边界前的航路段行为相对稳定。将不确定性留给优化器处理是更合理的架构

2. **完整 STAR 网络 MILP 优化模型**

    - **功能**：在 50NM 半径的真实 STAR 网络上求解多飞机无冲突最优路径
    - **核心约束**：
        - 路径守恒律（类似车辆路径问题的流守恒）
        - 到达/出发时间约束（含离散速度等级选择）
        - 安全间隔约束（任意两架飞机在同一航路点到达/出发间隔 ≥ $t_{f,f'}$）
        - 防追尾约束（同一路段上先出发的飞机先到达）
        - 等待约束（仅在指定航路点允许等待）
    - **目标函数**：最小化所有飞机平均着陆时间 $J = \min \frac{1}{|F|}\sum_{f \in F, j \in E} AR_j^f$
    - **设计动机**：基于完整 AIP（航空信息出版物）构建的真实 STAR 地图，而非简化网络，能更准确反映实际运行约束

3. **滚动时域闭环控制**

    - **功能**：将全规划域分割为若干子问题，通过 MPC 与仿真器双向交互实现实时控制
    - **核心思路**：
        - 前视窗口（look-ahead horizon）= 10min，控制窗口（control horizon）= 5min
        - 每个时刻只优化前视窗口内的飞机，但仅执行控制窗口内的指令
        - 仿真器按秒更新系统状态，分三种情况处理飞机位置：在途、到站等待、离站
        - 扰动模式下引入到达时间偏差 $\xi_{arr}$ 和状态观测偏差 $\xi_{ob}$
    - **设计动机**：一次性 MILP 优化在高拥堵场景下计算时间指数增长不可行，滚动时域将大问题拆分为可在秒级求解的子问题

## 实验关键数据

### 计算复杂度对比（10-27 架飞机，10 分钟窗口）

| Case | 飞机数 | MILP 时间(s) | MILP 成本(s) | Dijkstra 时间(s) | Dijkstra 成本(s) | 差距 |
|------|--------|-------------|-------------|-----------------|-----------------|------|
| 1 | 10 | 2.38 | 1189.2 | 14.99 | 1191.5 | 0.19% |
| 6 | 15 | 20.50 | 1084.7 | 21.97 | 1103.8 | 1.76% |
| 8 | 25 | 173.51 | 1072.5 | 24.48 | ~1080.6 | *(2架不可行)* |
| 9 | 27 | 2609.6 | 1083.9 | 24.82 | ~1085.3 | *(2架不可行)* |

### 滚动时域 vs 一次性优化（历史真实数据，1 小时）

| Case | 飞机数 | 一次性时间(s) | 一次性成本(s) | 滚动时域时间(s) | 滚动成本(s) |
|------|--------|-------------|-------------|---------------|------------|
| 2 | 25 | 169.30 | 2298.1 | 42.41 | 2302.2 |
| 5 | 36 | 784.69 | 2736.3 | 102.75 | 2736.3 |

### Monte Carlo 鲁棒性测试（36 架飞机，100 次 × 4 种扰动强度）

- 随机值 0.05-0.2 范围内，MILP 的不可行案例数远低于 Dijkstra
- Dijkstra 在随机值 = 0.2 时仅剩 1 个可行解，而 MILP 仍保持较高可行率
- MILP 计算时间稳定在 30-40s 范围，不随扰动强度明显波动

### 关键发现

- 峰值拥堵（36 架/小时）下滚动时域实现 7 倍以上计算加速（784.69s → 102.75s），且成本无损
- Dijkstra 贪心策略在高拥堵下无法保证可行性（大规模 Case 出现不可行飞机），而 MILP 始终可行
- MILP 计算时间随问题规模指数增长（27 架 = 2609s），证明了滚动时域的必要性
- TMA 内旅行时间方差（204s）是航路段（81s）的 2.5 倍，验证了预测 TMA 边界而非跑道 ETA 的合理性

## 亮点与洞察

1. **系统级闭环框架**：将预测、优化和仿真无缝整合，具备真实部署潜力
2. **基于真实 AIP 的 STAR 网络**：相比简化模型更贴近实际运行，约束覆盖全面（路径/速度/等待/安全间隔/防追尾）
3. **预测环节设计务实**：选择预测 TMA 边界到达时间（不确定性小）而非跑道 ETA（不确定性大），将难以预测的部分留给优化器处理
4. **Monte Carlo 验证鲁棒性**：100 次 × 4 种扰动的系统性验证具有说服力

## 局限与展望

1. 仅考虑 2D 路径规划，未涉及垂直分层策略
2. 速度选择为预设离散等级，未支持连续速度优化
3. Case 9（27 架/10min）已需 2609s，更大规模需要进一步加速（如列生成、分支定价）
4. 未考虑多跑道选择优化和起飞-到场交互
5. 仿真器为自制简化版，与真实空管系统集成验证尚未完成

## 相关工作与启发

- MPC + MILP 的闭环框架可推广到其他实时调度问题（如港口调度、仓储机器人路径规划）
- 将预测不确定性留给在线优化器处理的架构设计值得借鉴
- Monte Carlo 鲁棒性验证方法论可应用于其他安全关键系统

## 评分

⭐⭐⭐⭐

- **新颖性** ⭐⭐⭐：增量式创新，各组件有先例但系统整合有价值
- **实验充分度** ⭐⭐⭐⭐⭐：真实 ADS-B 数据、计算复杂度分析、滚动时域对比、Monte Carlo 鲁棒性验证，极为全面
- **写作质量** ⭐⭐⭐⭐：数学建模清晰严谨
- **价值** ⭐⭐⭐⭐：对实际空管决策支持有直接应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Task Prototype-Based Knowledge Retrieval for Multi-Task Learning from Partially Annotated Data](task_prototype-based_knowledge_retrieval_for_multi-task_lear.md)
- [\[ICCV 2025\] Adaptive Dual Uncertainty Optimization: Boosting Monocular 3D Object Detection under Test-Time Shifts](../../ICCV2025/autonomous_driving/adaptive_dual_uncertainty_optimization_boosting_monocular_3d_object_detection_un.md)
- [\[CVPR 2026\] Den-TP: A Density-Balanced Data Curation and Evaluation Framework for Trajectory Prediction](../../CVPR2026/autonomous_driving/den_tp_a_density_balanced_data_curation_and_evaluation_framework_for_trajectory.md)
- [\[CVPR 2026\] Test-Time Training for LiDAR Semantic Segmentation under Corruption via Geometric Inlier Discrimination](../../CVPR2026/autonomous_driving/test-time_training_for_lidar_semantic_segmentation_under_corruption_via_geometri.md)
- [\[AAAI 2026\] TimeBill: Time-Budgeted Inference for Large Language Models](timebill_time-budgeted_inference_for_large_language_models.md)

</div>

<!-- RELATED:END -->
