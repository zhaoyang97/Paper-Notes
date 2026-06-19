---
title: >-
  [论文解读] OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving
description: >-
  [ECCV 2024][自动驾驶][3D Occupancy] OccWorld 提出在 3D 占用空间中学习世界模型，用 VQ-VAE 对 3D occupancy 进行 token 化，再通过 GPT 风格的时空生成 Transformer 自回归预测未来场景演化和自车轨迹，在 nuScenes 上无需实例和地图标注即可实现有竞争力的规划性能。
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "3D Occupancy"
  - "世界模型"
  - "4D预测"
  - "轨迹规划"
---

# OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving

**会议**: ECCV 2024  
**arXiv**: [2311.16038](https://arxiv.org/abs/2311.16038)  
**代码**: [https://github.com/wzzheng/OccWorld](https://github.com/wzzheng/OccWorld)  
**领域**: 自动驾驶  
**关键词**: 3D Occupancy, 世界模型, 自动驾驶, 4D预测, 轨迹规划

## 一句话总结
OccWorld 提出在 3D 占用空间中学习世界模型，用 VQ-VAE 对 3D occupancy 进行 token 化，再通过 GPT 风格的时空生成 Transformer 自回归预测未来场景演化和自车轨迹，在 nuScenes 上无需实例和地图标注即可实现有竞争力的规划性能。

## 研究背景与动机

**领域现状**：自动驾驶主流方案遵循"感知→预测→规划"的级联管线，如 UniAD、VAD 等，依赖 3D 检测框、语义地图、运动预测等中间表示，每个阶段都需要大量人工标注。

**现有痛点**：传统方法只预测目标的 bounding box 运动，无法捕捉更细粒度的 3D 场景信息（如路面形状变化、可通行区域演化）。同时，级联设计导致每一阶段的训练都依赖独立的标注数据，标注成本极高。

**核心矛盾**：自动驾驶需要理解完整 3D 场景的时序演变来做决策，而现有框架在实例级别建模远不能覆盖全部场景要素（如静态结构、可驾驶区域的变化等）。

**本文目标**  
   - 如何用一个统一模型同时预测周围场景演化和自车运动？  
   - 如何在不使用实例框和 HD Map 标注的前提下完成规划？  
   - 如何高效地将高维 3D occupancy 序列建模为可预测的 token 序列？

**切入角度**：作者观察到 3D occupancy 具有三大优势——表达力强（包含完整 3D 结构和语义）、获取高效（可从稀疏 LiDAR 点生成）、模态通用（适用于视觉和 LiDAR），因此以 occupancy 替代 bounding box 作为世界模型的场景表示。

**核心 idea**：在 3D occupancy 空间中训练类 GPT 的时空生成 Transformer，自回归预测未来场景和自车轨迹，无需实例级监督。

## 方法详解

### 整体框架
输入为过去若干帧的 3D semantic occupancy（$\mathbf{y} \in \mathbb{R}^{H \times W \times D}$）和自车位姿，输出为未来多帧的 3D occupancy 和自车轨迹。整个流程分两阶段：
1. **Stage 1**：训练 3D occupancy scene tokenizer（VQ-VAE），将高维 occupancy 压缩为离散 token；
2. **Stage 2**：冻结 tokenizer，训练 spatial-temporal generative transformer，自回归预测未来场景 token 和 ego token，再通过解码器重建 occupancy 和轨迹。

### 关键设计

1. **3D Occupancy Scene Tokenizer (VQ-VAE)**:

    - 功能：将 3D occupancy 压缩为离散 token 序列，编码高层语义概念
    - 核心思路：先将 3D occupancy 通过类别嵌入转为 BEV 表示 $\hat{\mathbf{y}} \in \mathbb{R}^{H \times W \times DC'}$（沿高度维度拼接），再用 2D 卷积 encoder 下采样 $d$ 倍得到 $\hat{\mathbf{z}} \in \mathbb{R}^{H/d \times W/d \times C}$。同时学习 codebook $\mathbf{C} \in \mathbb{R}^{N \times C}$，对每个空间特征做最近邻量化：$\mathbf{z}_{ij} = \arg\min_{\mathbf{c} \in \mathbf{C}} \|\hat{\mathbf{z}}_{ij} - \mathbf{c}\|_2$。解码器用 2D 反卷积恢复原始分辨率后，沿通道维 split 恢复高度维，再 softmax 分类重建 occupancy
    - 设计动机：3D occupancy 维度太高（200×200×16），直接建模困难。VQ-VAE 将其压缩到 50×50 的离散 token，大幅降低后续 Transformer 建模难度，同时 codebook 编码了可复用的高层场景概念

2. **Spatial-Temporal Generative Transformer**:

    - 功能：建模 token 序列的时空演化关系，自回归预测下一帧 token
    - 核心思路：先对每帧 token 做 spatial aggregation（self-attention），然后在 2×2 窗口内 merge 形成多尺度 token $\{\mathbf{T}_0, \ldots, \mathbf{T}_K\}$（K=3 级）。对每个尺度独立做 spatial-wise temporal causal attention：$\hat{\mathbf{z}}_{j,i}^{T+1} = \text{TA}(\mathbf{z}_{j,i}^T, \ldots, \mathbf{z}_{j,i}^{T-t})$，即同一空间位置的 token 跨时间做 masked attention 预测未来。最后用 U-Net 结构聚合多尺度预测，确保空间一致性
    - 设计动机：直接像 NLP-GPT 逐 token 预测太慢（token 数太多）。分离时空建模+多尺度设计既保证了全局感知（spatial mixing），又实现了高效的时序预测（按位置的 temporal attention）

3. **Ego Token 与轨迹解码**:

    - 功能：将自车运动也统一到 token 序列中，与场景 token 联合建模
    - 核心思路：引入 ego token $\mathbf{z}_0 \in \mathbb{R}^C$ 编码自车位置，与场景 token 一起参与 spatial aggregation 和 temporal attention。预测的 ego token 通过 MLP ego decoder 解码为位移：$\hat{p}^{T+1} = d_{ego}(\hat{z}_0^{T+1})$
    - 设计动机：传统方法将场景预测和自车规划分开处理，忽略了二者的高阶交互。将 ego token 融入世界模型能捕捉"自车移动导致场景变化"的耦合关系

### 损失函数 / 训练策略
- **Stage 1 (Tokenizer)**：$J_{e,d} = L_{soft}(d(e(\mathbf{y})), \mathbf{y}) + \lambda_1 L_{lovasz}(d(e(\mathbf{y})), \mathbf{y})$，即 softmax CE + Lovász-softmax loss
- **Stage 2 (World Model)**：$J_{w} = \sum_t \sum_j L_{soft}(\hat{\mathbf{z}}_{j,0}^t, \mathbf{C}(\mathbf{z}_{j,0}^t)) + \lambda_2 L_{L2}(d_{ego}(\hat{\mathbf{z}}_0^t), \mathbf{p}^t)$，即 token 分类损失 + 轨迹 L2 损失
- 训练时用 ground-truth token 作为输入（teacher forcing），推理时用自回归预测 token 逐帧推进
- 使用 2s 历史预测 3s 未来，batch=1/GPU，8× RTX 4090

## 实验关键数据

### 主实验：4D Occupancy Forecasting

| 方法 | 输入 | 额外监督 | mIoU(1s) | mIoU(2s) | mIoU(3s) | mIoU(Avg) | IoU(Avg) |
|------|------|----------|----------|----------|----------|-----------|----------|
| Copy&Paste | 3D-Occ | 无 | 14.91 | 10.54 | 8.52 | 11.33 | 20.52 |
| **OccWorld-O** | 3D-Occ | 无 | **25.78** | **15.14** | **10.51** | **17.14** | **26.63** |
| OccWorld-D | Camera | 3D-Occ | 11.55 | 8.10 | 6.22 | 8.62 | 16.53 |
| OccWorld-T | Camera | Sem-LiDAR | 4.68 | 3.36 | 2.63 | 3.56 | 8.34 |

### 主实验：Motion Planning (L2 ↓ / Collision Rate ↓)

| 方法 | 输入 | 额外监督 | L2 Avg(m) | Col. Avg(%) |
|------|------|----------|-----------|-------------|
| UniAD | Camera | Map+Box+Motion+Tracklets+Occ | 1.03 | 0.31 |
| VAD-Base | Camera | Map+Box+Motion | 1.22 | 0.53 |
| **OccWorld-O** | 3D-Occ | **无** | **1.17** | 0.60 |
| OccWorld-D | Camera | 3D-Occ | 1.40 | 0.87 |

### 消融实验：Scene Tokenizer 超参

| 设置 (token数², 特征维, codebook) | 重建 mIoU | 预测 mIoU(Avg) | 规划 L2(Avg) | FPS |
|----------------------------------|-----------|---------------|-------------|-----|
| (50², 128, 512) | 66.38 | 17.14 | 1.17 | 18.0 |
| (50², 128, 256) | 63.40 | 16.24 | 1.15 | 17.8 |
| (25², 256, 512) | 36.28 | 8.81 | 6.53 | 28.1 |
| (100², 128, 512) | 78.12 | 12.38 | 1.36 | 6.7 |

### 关键发现
- 50×50 token 分辨率是最优折中：太小（25²）重建质量差，太大（100²）虽然重建好但预测和规划反而变差，说明过多 token 让 Transformer 建模更困难
- OccWorld-O 在**零额外监督**下（不用任何 box/map 标注）的规划性能已超过 VAD-Base（需要 Map+Box+Motion 监督）
- Codebook 大小 512 最优，太大（1024）导致利用率低，反而降低性能
- Occupancy-based world model 能成功预测移动物体运动和可驾驶区域变化，甚至生成比 GT 更合理的可驾驶区域

## 亮点与洞察
- **Occupancy 作为世界模型的统一表示**：用 occupancy 替代 bbox 作为世界模型的操作对象，天然包含动态和静态场景要素，无需分别建模。这一思路可以迁移到室内机器人导航等需要精细 3D 场景理解的领域
- **场景 Token 化 + GPT 风格自回归**：将连续的 3D 空间量化为离散 token 后用语言模型范式预测，巧妙地将"场景预测"转化为"序列预测"问题，复用了 Transformer 在序列建模上的强大能力
- **时空分离的多尺度设计**：spatial mixing + per-position temporal attention + U-Net fusion，既保证计算效率又维持全局一致性，比全时空 attention 高效得多

## 局限与展望
- 无法预测视野外新出现的车辆（因为输入中没有这些物体的信息），需要结合生成模型或概率预测来处理
- 自监督 occupancy (OccWorld-S) 质量较差，mIoU 仅 0.26，说明目前的自监督 occ 方法还不够成熟
- 只在 nuScenes 上验证，场景多样性有限；需要在更大规模数据集上验证 scaling 能力
- VQ-VAE 的重建存在较大信息损失（mIoU 66.38 vs 理论上限），限制了预测精度上界
- 可以考虑结合 diffusion model 替代 GPT 自回归，避免多步生成的误差累积

## 相关工作与启发
- **vs UniAD**：UniAD 用感知-预测-规划的级联管线，需要大量中间标注（box、map、tracklets、motion、occ），OccWorld 仅用 occupancy 无需实例标注。UniAD 规划略优（L2 1.03 vs 1.17），但监督信号多得多
- **vs MILE/GAIA**：这些方法在 2D 图像空间做 world model，缺乏 3D 理解能力。OccWorld 在 3D occupancy 空间建模，输出可直接用于 3D 规划
- **vs PointCloud Forecasting**：如 4D-OCC、NTP 预测点云未来帧，但忽略语义信息且不支持视觉输入。OccWorld 包含语义 occupancy，兼容视觉和 LiDAR

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次提出在 3D occupancy 空间构建世界模型，VQ-VAE + GPT 的组合有新意
- 实验充分度: ⭐⭐⭐⭐ 4D forecasting 和 planning 两个任务都有详细实验和消融
- 写作质量: ⭐⭐⭐⭐ 论文逻辑清晰，从动机到方法到实验层层递进
- 价值: ⭐⭐⭐⭐⭐ 开创了 occupancy world model 方向，后续大量工作跟进，影响力大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] OccGen: Generative Multi-modal 3D Occupancy Prediction for Autonomous Driving](occgen_generative_multi-modal_3d_occupancy_prediction_for_autonomous_driving.md)
- [\[ECCV 2024\] Neural Volumetric World Models for Autonomous Driving](neural_volumetric_world_models_for_autonomous_driving.md)
- [\[ECCV 2024\] Fully Sparse 3D Occupancy Prediction](fully_sparse_3d_occupancy_prediction.md)
- [\[CVPR 2025\] GaussianWorld: Gaussian World Model for Streaming 3D Occupancy Prediction](../../CVPR2025/autonomous_driving/gaussianworld_gaussian_world_model_for_streaming_3d_occupancy_prediction.md)
- [\[ECCV 2024\] Risk-Aware Self-Consistent Imitation Learning for Trajectory Planning in Autonomous Driving](risk-aware_self-consistent_imitation_learning_for_trajectory_planning_in_autonom.md)

</div>

<!-- RELATED:END -->
