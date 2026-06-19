---
title: >-
  [论文解读] Recovering Parametric Scenes from Very Few Time-of-Flight Pixels
description: >-
  [ICCV 2025][飞行时间传感器] 本文探索用极少量（低至 15 个像素）低成本广视场 ToF 传感器恢复 3D 参数化场景几何的可行性，设计了前馈预测+可微渲染的分析-合成框架，在 6D 物体位姿估计等任务上展示了令人惊讶的效果。 飞行时间（ToF）相机是现代 3D 视觉的关键技术。主流方法依赖高分辨率密集 3D 数…
tags:
  - "ICCV 2025"
  - "飞行时间传感器"
  - "6D位姿估计"
  - "可微渲染"
  - "SPAD"
  - "参数化场景恢复"
---

# Recovering Parametric Scenes from Very Few Time-of-Flight Pixels

**会议**: ICCV 2025  
**arXiv**: [2509.16132](https://arxiv.org/abs/2509.16132)  
**代码**: [项目主页](https://cpsiff.github.io/recovering_parametric_scenes)  
**领域**: 其他  
**关键词**: 飞行时间传感器, 6D位姿估计, 可微渲染, SPAD, 参数化场景恢复

## 一句话总结

本文探索用极少量（低至 15 个像素）低成本广视场 ToF 传感器恢复 3D 参数化场景几何的可行性，设计了前馈预测+可微渲染的分析-合成框架，在 6D 物体位姿估计等任务上展示了令人惊讶的效果。

## 研究背景与动机

飞行时间（ToF）相机是现代 3D 视觉的关键技术。主流方法依赖高分辨率密集 3D 数据，形成了"密集深度测量是精确 3D 视觉的必要条件"这一共识。然而近年出现了一类**超低成本** ToF 传感器（<$3/颗，<5mm），已部署在手机和可穿戴设备中。这些传感器的特点是：

- **极低空间分辨率**：甚至只有单个像素
- **宽视场**：每个像素覆盖约 30° 的视角
- **丰富的时间信息**：通过瞬态直方图（transient histogram）捕获精细的飞行时间数据

传统做法将这些直方图简化为单个深度值（峰值检测），丢失了大量信息。本文的核心假设是：**即使只有少数几个瞬态直方图，它们编码的场景信息也足以在强几何先验下恢复 3D 场景**。

本文追问：在有参数化形状模型先验的条件下，恢复 3D 场景所需的**最少深度测量数量**是多少？

## 方法详解

### 整体框架

方法由两个核心组件构成：
1. **前馈预测网络**：从稀疏瞬态直方图 $\{\mathbf{h}_i\}_{i=1}^n$ 直接预测场景参数 $\mathbf{P}_{\text{FF}}$
2. **分析-合成精炼器**：利用可微渲染器 $\mathcal{R}$ 在 $\mathbf{P}_{\text{FF}}$ 基础上迭代优化

### 关键设计

1. **SPAD 瞬态成像模型**：每个传感器发射 $N_{\text{emit}}$ 个光子，光子沿方向 $\boldsymbol{\omega}$ 到达场景点 $\mathbf{x}$ 并反射回传感器。第 $i$ 个时间 bin 接收的期望光子数为：

$$N[i] = N_{\text{emit}} \int_{\Omega} I(\boldsymbol{\omega}) \frac{\rho(\mathbf{x})}{\pi} \frac{\langle -\boldsymbol{\omega}, \hat{\mathbf{n}}(\mathbf{x}) \rangle}{\|\mathbf{x}\|^2} W\left(\frac{2\|\mathbf{x}\|}{c}, t_i\right) d\boldsymbol{\omega}$$

其中 $\rho(\mathbf{x})$ 是反照率，$\hat{\mathbf{n}}(\mathbf{x})$ 是表面法向，$W$ 是时间 binning 函数。关键是这个模型考虑了广视场内**所有方向**的光子贡献，而非仅一个峰值深度。再卷积经验抖动核 $\mathbf{s}$ 得到最终直方图。

2. **可微渲染器**：将积分离散化为 $h \times w$ 网格的加权求和，使用 Nvdiffrast 实现可微光栅化。对不连续的 binning 函数 $W$ 用 sigmoid 近似：

$$W(t, t_i) = \sigma(k(t-t_i)) - \sigma(k(t-t_i-\Delta t))$$

激光强度分布 $I(\boldsymbol{\omega})$ 用可微高斯核拟合：$I(\boldsymbol{\omega}) = K_1 \exp(-K_2(\omega_x^2+\omega_y^2) - K_3(\omega_x^4+\omega_y^4))$。整个渲染管线对场景参数完全可微。

3. **前馈 Transformer 网络**：输入 $n$ 个归一化直方图，通过 MLP 嵌入后加位置编码，经 4 层 Transformer blocks 处理，拼接输出嵌入后经 MLP 预测场景参数。关键挑战是真实数据匮乏，因此**使用渲染器生成大规模合成数据训练**，并通过域随机化（传感器位置噪声 ±1.5cm、反照率随机化）实现 sim-to-real 迁移。

### 损失函数 / 训练策略

- **前馈网络**：对非对称物体使用旋转损失 + 平移损失 + 点匹配损失；对对称物体使用 ADD-S 损失
- **精炼阶段**：Adam 优化器最小化 $\sum_{i=1}^n \|\mathcal{R}(\mathbf{P})_i - \mathbf{h}_i\|$，旋转学习率 0.01，平移 0.001，200 步迭代
- 同时优化物体和平面的反照率参数
- 旋转用 6D 连续表示以避免万向锁

## 实验关键数据

### 主实验

**YCB 物体 6D 位姿估计（对称物体，AUC-ADD-S ↑）**：

| 方法 | 像素数 | Crackers | Mustard | SPAM | Basketball | Tennis | Mean |
|------|--------|----------|---------|------|-----------|--------|------|
| 1Px 点云(仿真) | 15 | 78.36 | 82.12 | 85.07 | 82.92 | 88.09 | 82.96 |
| **Ours: FF+Refiner(真实)** | **15** | **90.04** | **90.07** | **90.00** | **95.76** | **96.06** | **92.20** |
| 16² 点云(仿真) | 3840 | 95.17 | 97.23 | 97.19 | 97.67 | 97.57 | 97.06 |
| 单视图 RGB(真实) | 407K | 60.71 | 87.93 | 58.95 | 65.46 | 77.68 | 66.18 |
| 单视图 RGB-D(真实) | 407K | 90.49 | 92.10 | 93.80 | 94.24 | 86.67 | 92.01 |

仅用 15 个 ToF 像素就接近了使用 40 万像素的 RGB-D 方法！

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅前馈 vs FF+Refiner（3D打印物体） | AUC-ADD 70.96→80.71 | 精炼器提升约 10 个点 |
| 仅前馈 vs FF+Refiner（YCB） | AUC-ADD-S 91.83→92.20 | 对称物体提升较小 |
| 球体位置/尺寸恢复 | 位置误差<1cm, 直径误差<0.35cm | 时间分辨率 1.4cm 下仍有效 |
| 手部姿态（仅仿真训练） | PA-MPJPE 19.56mm | 近距离 sim-to-real 差距大 |
| 手部姿态（仿真预训练+真实微调） | PA-MPJPE 8.18mm | 迁移学习有效 |

### 关键发现

- 瞬态直方图蕴含的信息远超单个深度值，在 5-100 像素范围内本文方法始终优于点云基线
- 点云基线在极稀疏条件下因覆盖不足而严重退化，但瞬态数据的广视场特性缓解了此问题
- Sim-to-real 迁移在 6D 位姿估计中效果良好，但在手部追踪（近距离强光）中退化
- 超越单视图 RGB 方法 26 个百分点（66.18→92.20），接近 RGB-D（92.01 vs 92.20）

## 亮点与洞察

- **极端稀疏感知的可行性论证**：15 个<$3 的传感器就能做 6D 位姿估计，颠覆了对密集感知的依赖
- **信息利用最大化**：不做峰值检测简化，而是直接利用完整瞬态直方图的所有时间 bin
- **端到端可微管线**：从成像物理到位姿优化，梯度全程可达
- **硬件原型验证**：不仅是仿真，用真实机器人臂+TMF8820 传感器在多个物体上验证

## 局限与展望

- 假设朗伯表面 + 共位传感器/光源，真实复杂反射场景性能待验证
- 传感器距离限制（TMF8820 最远 1.5m），仅能做桌面场景
- 固定传感器配置需要重训网络，灵活性有限
- 手部追踪在近距离（<15cm）sim-to-real 差距大，因未建模的门控/pile-up 效应
- 当前仅处理单物体，多物体遮挡场景有待探索

## 相关工作与启发

- 与 Pixels2Pose（4×4 SPAD 估人体姿态）类似但更极端（1 pixel/view），且引入可微渲染精炼
- 可微渲染+分析-合成范式（Analysis-by-Synthesis）在低数据量条件下特别有效
- 对可穿戴计算和低功耗机器人感知有直接应用前景
- 启发思考：哪些视觉任务可以用极少量物理传感器替代高分辨率相机？

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 极端稀疏 ToF 的问题设定非常新颖
- **实验充分度**: ⭐⭐⭐⭐ 仿真+真实硬件, 多种场景, 但仅桌面尺度
- **写作质量**: ⭐⭐⭐⭐ 问题定义清晰, 物理建模详尽
- **价值**: ⭐⭐⭐⭐ 开辟了超低成本 3D 感知的新方向, 有显著实用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] AdaptiveAE: An Adaptive Exposure Strategy for HDR Capturing in Dynamic Scenes](adaptiveae_an_adaptive_exposure_strategy_for_hdr_capturing_i.md)
- [\[ICCV 2025\] Doodle Your Keypoints: Sketch-Based Few-Shot Keypoint Detection](doodle_your_keypoints_sketch-based_few-shot_keypoint_detection.md)
- [\[NeurIPS 2025\] TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels](../../NeurIPS2025/others/trackingworld_world-centric_monocular_3d_tracking_of_almost_all_pixels.md)
- [\[CVPR 2026\] Basis-Oriented Low-rank Transfer for Few-Shot and Test-Time Adaptation](../../CVPR2026/others/basis-oriented_low-rank_transfer_for_few-shot_and_test-time_adaptation.md)
- [\[ICCV 2025\] Is Meta-Learning Out? Rethinking Unsupervised Few-Shot Classification with Limited Entropy](is_meta-learning_out_rethinking_unsupervised_few-shot_classification_with_limite.md)

</div>

<!-- RELATED:END -->
