---
title: >-
  [论文解读] T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning
description: >-
  [ECCV 2024][3D视觉][自监督学习] T-MAE 提出了一种时序掩码自编码器预训练策略，以时序相邻两帧为输入，通过掩码当前帧并借助历史帧信息重建来学习时序依赖关系，配合设计的 SiamWCA（孪生编码器+窗口交叉注意力）架构，在 Waymo 和 ONCE 数据集上以更少标注数据和更少训练迭代次数超越 SOTA 自监督方法。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "自监督学习"
  - "点云表征学习"
  - "掩码自编码器"
  - "时序建模"
  - "3D目标检测"
---

# T-MAE: Temporal Masked Autoencoders for Point Cloud Representation Learning

**会议**: ECCV 2024  
**arXiv**: [2312.10217](https://arxiv.org/abs/2312.10217)  
**代码**: [https://github.com/weijie-wei/T-MAE](https://github.com/weijie-wei/T-MAE)  
**领域**: 自动驾驶  
**关键词**: 自监督学习, 点云表征学习, 掩码自编码器, 时序建模, 3D目标检测

## 一句话总结
T-MAE 提出了一种时序掩码自编码器预训练策略，以时序相邻两帧为输入，通过掩码当前帧并借助历史帧信息重建来学习时序依赖关系，配合设计的 SiamWCA（孪生编码器+窗口交叉注意力）架构，在 Waymo 和 ONCE 数据集上以更少标注数据和更少训练迭代次数超越 SOTA 自监督方法。

## 研究背景与动机
LiDAR 点云理解中标注数据的稀缺严重制约了表征学习。例如 nuScenes 仅标注了 10% 的帧，ONCE 仅标注了 0.8%。自监督预训练是解决这一问题的有效途径。

**已有 SSL 方法的局限**：
- **对比学习方法**（PointContrast、SegContrast 等）：需要精心调节超参数和复杂的预/后处理来建立对应关系
- **掩码重建方法**（GD-MAE、MV-JAR 等）：虽然有效，但都在单帧场景下操作，忽视了 LiDAR 数据天然的时序连续性
- **利用时序信息的方法**（STSSL、TARL）：虽然引入多帧输入，但核心仍是对比学习，把不同时间的扫描当作同一场景的增强版本，未将时序对应关系纳入建模

**核心洞察**：时序相邻帧之间不仅包含冗余信息（可用于增强表征），更重要的是包含运动信息。ego 车辆的移动改变了同一物体的视角，这是一种天然且强大的数据增强。

**本文方案**：将掩码自编码范式从单帧扩展到双帧 — 以完整的历史帧和高比例掩码的当前帧作为输入，预训练任务是重建当前帧的掩码体素，从而迫使网络学习时序对应和运动建模能力。

## 方法详解

### 整体框架
T-MAE 的完整 pipeline：
1. **时序批次采样**：从点云序列中采样一对帧 $(\mathcal{P}^{t_1}, \mathcal{P}^{t_2})$，间隔在适当范围内（过近信息冗余，过远重叠不足）
2. **对齐与体素化**：用 ego-pose 将历史帧对齐到当前帧坐标系，两帧各自体素化为 pillar 特征
3. **掩码**（仅预训练）：对当前帧以 75% 的高掩码率随机遮蔽 pillar
4. **孪生编码器**：两帧共享权重的编码器（SPT）提取稀疏 token
5. **窗口交叉注意力（WCA）**：当前帧 token 从历史帧 token 中获取信息
6. **稠密特征恢复**：稀疏 token 放回空间位置形成稠密特征图，四层稠密卷积填充空位
7. **任务头**：预训练时用重建头恢复掩码点坐标（Chamfer Distance loss）；微调时用检测头（CenterPoint 风格）

### 关键设计
1. **SiamWCA 架构（孪生编码器 + 窗口交叉注意力）**：

    - **功能**：建立双帧点云处理的基础架构，使当前帧能利用历史帧信息
    - **孪生编码器**：两个分支共享相同的配置和权重（Siamese），对两帧的 pillar 特征编码为稀疏 token。比较了三种方案：
        - 非对称编码器（历史帧编码器减半通道）→ 较差
        - SimSiam 风格（历史帧编码器不接收梯度）→ 最差
        - **孪生编码器（梯度累积）→ 最优**
    - **设计动机**：共享权重确保两帧特征在同一潜在空间中，有利于交叉注意力；梯度累积方式让两个编码器同步优化

2. **Windowed Cross-Attention (WCA) — 窗口交叉注意力**：

    - **功能**：在窗口内执行交叉注意力，让当前帧 token 从历史帧中获取信息
    - **核心流程**：
        - **联合 token 分组**：将 3D 空间划分为非重叠窗口，两帧（已对齐坐标系）token 按空间位置分配到对应窗口
        - **稀疏区域交叉注意力（SRCA）**：$\hat{\mathcal{F}}^{t_2} = \text{MCA}(\mathcal{F}^{t_2} + \text{PE}(\mathcal{I}^{t_2}), \mathcal{F}^{t_1} + \text{PE}(\mathcal{I}^{t_1}), \mathcal{F}^{t_1})$
        - Query 来自当前帧 $\mathcal{P}^{t_2}$，Key-Value 来自历史帧 $\mathcal{P}^{t_1}$
        - 若某窗口在历史帧中为空，当前帧 token 保持不变
        - **窗口移位 + 重复操作**：移位半个窗口大小后重新分组并执行第二次 SRCA，扩大感受野
    - **设计动机**：全局交叉注意力在 3D 点云的高分辨率特征图（468×468 vs ViT 的 14×14）上计算代价不可接受，窗口化实现将计算复杂度控制在可承受范围

3. **T-MAE 预训练策略**：

    - **功能**：通过重建掩码的当前帧来学习时序对应和运动建模
    - **核心思路**：
        - 历史帧完整输入编码器（提供参考信息）
        - 当前帧以 75% 高掩码率遮蔽后输入编码器（制造信息缺失）
        - WCA 模块利用历史帧 token 增强当前帧残余 token
        - 稠密恢复后，重建头从特征图中检索掩码 pillar 的特征，重建每个 pillar 内固定数量 $K^O$ 个点的相对坐标
        - 损失：重建点与 GT 点之间的 Chamfer Distance
    - **与单帧 MAE 的根本区别**：
        - 单帧方法（GD-MAE）仅重用编码器权重
        - T-MAE 保留整个 SiamWCA 的权重（编码器+WCA），因此时序对应能力也被保留用于下游任务
    - **设计动机**：掩码重建迫使网络同时学习两种能力：(a) 稀疏点云的强表征；(b) 从历史观测推理当前帧的时序建模

### 损失函数 / 训练策略
- **预训练 loss**：Chamfer Distance，每个被掩码 pillar 重建 $K^O$ 个点
- **下游检测 loss**：使用 CenterPoint 风格的 center-based head + 相同目标分配策略
- **数据增强**：随机翻转、缩放、旋转应用于双帧；微调时加 copy-n-paste 增强处理类别不平衡
- **时序批次采样**：从序列中采样连续 $n$ 帧组成 batch，两帧分别从前 1/3 和后 1/3 采样，保证适当时间间隔
- 基于 OpenPCDet 框架实现

## 实验关键数据

### 主实验

**Waymo 数据集 (val, Level 2, 不同标注比例)**：

| 标注比例 | 方法 | 初始化 | Overall mAPH | Vehicle APH | Ped APH | Cyclist APH |
|---------|------|--------|-------------|-------------|---------|-------------|
| 5% | Random Init | 从零训练 | 40.29 | 53.50 | 44.76 | 22.61 |
| 5% | MV-JAR | SSL | 46.68 | 56.01 | 47.69 | 36.33 |
| 5% | GD-MAE | SSL | 44.56 | 55.76 | 46.22 | 31.69 |
| 5% | **T-MAE** | **SSL** | **49.46 (+9.17)** | **56.63** | **55.28** | **36.48** |
| 10% | MV-JAR | SSL | 54.06 | 58.00 | 54.66 | 49.52 |
| 10% | **T-MAE** | **SSL** | **57.99 (+4.86)** | **59.77** | **61.10** | **53.09** |
| 100% | GD-MAE | SSL | 67.64 | 68.29 | 65.47 | 69.16 |
| 100% | MV-JAR | SSL | 66.20 | 65.12 | 65.28 | 68.20 |
| 100% | Random Init | 从零训练 | 69.13 | 68.62 | 68.80 | 69.97 |
| 100% | **T-MAE** | **SSL** | **70.52 (+1.39)** | **68.89** | **72.01** | **70.65** |

**ONCE 数据集 (val)**：

| 方法 | 预训练 | mAP | Vehicle | Pedestrian | Cyclist |
|------|--------|-----|---------|------------|---------|
| SiamWCA | ✗ | 63.71 | 76.47 | 47.27 | 67.40 |
| GD-MAE | ✓ | 64.92 | 76.79 | 48.84 | 69.14 |
| **T-MAE** | **✓** | **67.00 (+3.29)** | **78.35** | **52.57** | **70.09** |

### 消融实验

**架构设计对比（Waymo, 5% 数据）**：

| 模型 | 编码器 | 融合方式 | Overall mAPH | 说明 |
|------|--------|---------|-------------|------|
| (a) | 非对称 | WCA | 44.78 | 编码器不共享权重 |
| (b) | SimSiam | WCA | 42.05 | 一个编码器不接受梯度 |
| (c) | Siamese | WCA+WSA | 45.11 | 加自注意力层 |
| (d) | Siamese | WSA | 40.90 | 仅自注意力（拼接两帧） |
| **(e) Ours** | **Siamese** | **WCA** | **46.78** | **最优方案** |

**双帧 vs 单帧 vs 帧拼接（Waymo, 5% 数据）**：

| 方法 | 输入方式 | Overall mAPH | Cyclist APH | 说明 |
|------|---------|-------------|-------------|------|
| GD-MAE | 单帧 | 44.56 | 31.69 | 基线 |
| GD-MAE | 拼接两帧 | 43.69 | 26.12 | 骑行者反而下降 |
| **T-MAE** | **双帧WCA** | **46.78** | **32.37** | **一致提升** |

### 关键发现
- **T-MAE 预训练效果随标注数据减少而增强**：从 100% 数据的 +1.39 mAPH 提升到 5% 数据的 +9.17 mAPH
- **行人检测提升最显著**：T-MAE 用 5% 标注数据的行人 mAPH（55.28）超过 MV-JAR 用 10% 数据的结果，说明时序建模对方向感知（APH）尤为有效
- **SiamWCA 本身就是强 backbone**：100% 数据时从零训练（69.13 mAPH）已超过所有 SSL 方法，但数据少时严重依赖标注
- **帧拼接不如学习型融合**：简单拼接两帧虽能增加点密度，但对运动物体会引入"幽灵点"，导致骑行者检测下降
- **收敛速度快**：T-MAE 以 1.6× 到 2.4× 更少的微调迭代次数即超过 MV-JAR
- **兼容性好**：T-MAE 在不同编码器（SST、SpCNN、SPT）和检测头（CenterPoint、Graph R-CNN）上均有显著提升

## 亮点与洞察
1. **时序信息的自监督利用** — 首次将掩码自编码范式从单帧扩展到双帧用于 LiDAR 点云，让网络通过重建任务自然学习时序对应
2. **ego 运动 = 天然数据增强** — 自车移动改变同一物体的观察视角，是一种强大且免费的数据增强，无需人为设计
3. **WCA 设计的高效性** — 基于窗口的交叉注意力将复杂度控制在可承受范围，同时通过窗口移位扩大感受野
4. **行人方向检测的显著改善** — 时序建模让网络更好地理解行人朝向，对下游行人意图预测有实际价值
5. **SiamWCA 的发现** — 证明双帧架构本身就很强大，SSL 的作用在于降低其对标注数据的依赖

## 局限与展望
- 目前仅使用两帧，扩展到更多帧可能进一步提升效果
- WCA 模块在预训练和推理时都需要双帧输入，增加了一定的计算开销
- 掩码策略为随机 pillar 掩码，未探索基于运动区域的自适应掩码
- 对快速运动物体（如骑行者的方向变化），时序对齐仍存在挑战
- 未探索与对比学习方法的结合

## 相关工作与启发
- **GD-MAE** — 单帧掩码重建的 SOTA baseline，T-MAE 在其基础上扩展到双帧
- **SiameseMAE** — 视频理解中的双帧掩码自编码，启发了 T-MAE 将类似范式迁移到点云
- **SST** — 提出的稀疏区域自注意力（SRA）是 WCA 的基础
- **TARL/STSSL** — 利用时序信息的对比学习方法，但依赖复杂预处理（HDBSCAN 获取 segments），T-MAE 避免了这些
- **启发**：时序维度是点云 SSL 的一个被低估的信息来源，即使仅用两帧就能带来显著提升

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DAP-MAE: Domain-Adaptive Point Cloud Masked Autoencoder for Effective Cross-Domain Learning](../../ICCV2025/3d_vision/dap-mae_domain-adaptive_point_cloud_masked_autoencoder_for_effective_cross-domai.md)
- [\[ECCV 2024\] DG-PIC: Domain Generalized Point-In-Context Learning for Point Cloud Understanding](dg-pic_domain_generalized_point-in-context_learning_for_point_cloud_understandin.md)
- [\[ECCV 2024\] 3D Single-Object Tracking in Point Clouds with High Temporal Variation](3d_single-object_tracking_in_point_clouds_with_high_temporal_variation.md)
- [\[AAAI 2026\] Distilling Future Temporal Knowledge with Masked Feature Reconstruction for 3D Object Detection](../../AAAI2026/3d_vision/distilling_future_temporal_knowledge_with_masked_feature_reconstruction_for_3d_o.md)
- [\[ICCV 2025\] StruMamba3D: Exploring Structural Mamba for Self-supervised Point Cloud Representation Learning](../../ICCV2025/3d_vision/strumamba3d_exploring_structural_mamba_for_self-supervised_point_cloud_represent.md)

</div>

<!-- RELATED:END -->
