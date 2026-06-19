---
title: >-
  [论文解读] SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection
description: >-
  [AAAI2026 Oral][目标检测][多模态遥感] SM3Det提出了遥感领域的M2Det新任务（多模态数据集+多任务目标检测），通过网格级稀疏MoE骨干网络和动态子模块优化（DSO）机制，用单一模型同时处理SAR/光学/红外三种模态的水平/旋转框检测，显著超越各模态独立训练的三个专用模型组合。
tags:
  - "AAAI2026 Oral"
  - "目标检测"
  - "多模态遥感"
  - "稀疏MoE"
  - "动态学习率优化"
  - "统一模型"
---

<!-- SM3Det 论文笔记 -->
# SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection

**会议**: AAAI2026 Oral  
**arXiv**: [2412.20665](https://arxiv.org/abs/2412.20665)  
**代码**: [github.com/zcablii/SM3Det](https://github.com/zcablii/SM3Det)  
**领域**: 目标检测  
**关键词**: 多模态遥感, 目标检测, 稀疏MoE, 动态学习率优化, 统一模型  

## 一句话总结

SM3Det提出了遥感领域的M2Det新任务（多模态数据集+多任务目标检测），通过网格级稀疏MoE骨干网络和动态子模块优化（DSO）机制，用单一模型同时处理SAR/光学/红外三种模态的水平/旋转框检测，显著超越各模态独立训练的三个专用模型组合。

## 背景与动机

1. **多模态数据日益丰富**：遥感平台（无人机/卫星）通常搭载多种传感器（SAR/光学/红外），获取多模态图像，但目前检测模型通常只针对单一模态单一数据集训练。
2. **传统方法忽视跨模态共享知识**：各模态独立训练错失了跨模态共享知识的利用机会，如不同模态中目标的形状、尺度等共性特征。
3. **已有多源检测依赖对齐**：此前的多源检测方法严重依赖稀缺且不灵活的空间对齐配对图像和对齐算法，实用性差。
4. **表征空间拥挤问题**：用一套共享参数的密集模型（dense model）拟合多模态多任务数据会导致表征空间拥挤，单一参数集难以有效拟合各模态的不同分布。
5. **优化不一致性**：不同模态和任务的学习难度不同，导致优化速率不同步、优化方向冲突，相互干扰各损失函数的收敛。
6. **低空经济应用需求**：飞行汽车、无人机、卫星等低空经济场景急需能同时处理多模态的统一检测能力，减少设备上的模型维护成本。

## 方法详解

### 任务定义：M2Det

作者首次定义了遥感领域的 **M2Det** 任务——用统一模型检测来自任意传感器模态的图像中目标，同时处理水平框（HBB）和旋转框（OBB）两种检测格式。

### 整体架构

经典的多任务学习架构：共享骨干网络 + 各任务独立轻量检测头。SA数据用GFL头（水平框），DOTA和DroneVehicle用O-RCNN头（旋转框）。

### 网格级稀疏MoE骨干

**核心创新**：在骨干网络中引入即插即用的网格级稀疏MoE架构。

- **不同于已有方法的图像级路由**（整张图像路由到单个专家），SM3Det在空间特征的网格粒度做专家选择
- 对CNN骨干（如ConvNeXt），用MoE替换 $1 \times 1$ 卷积层；对Transformer骨干，将MoE集成到FFN中
- 门控函数：$G(x_{ij}) = \text{TOP}_k(\text{Softmax}(\frac{E^T W x_{ij}}{\tau \|Wx_{ij}\| \|E\|}))$
- 最终输出为top-k专家的加权和：$f_{MoE}(x_{ij}) = \sum_{n=1}^N G_n(x_{ij}) \cdot Conv_n^{1\times 1}(x_{ij})$
- **初始化策略**：用预训练权重的 $1 \times 1$ 卷积复制初始化所有专家，确保起始时各专家被均匀选中

**优势**：共享专家学习跨模态通用知识（如目标形状/尺度），专属专家学习各模态独有特征（如SAR散射特性）。

### 动态子模块优化（DSO）

DSO分两部分分别调节检测头和骨干的学习率：

**检测头学习率调节**（平衡各任务收敛速率）：
- 维护每个任务损失的EMA历史值 $his\_L_i^t$
- 计算收敛速率逆指标 $w_i^t = his\_L_i^t / cur\_L_i^t$
- 通过带温度的Softmax重加权各头学习率：$\lambda_i^t = T \cdot e^{w_i^t/\theta} / \sum_k e^{w_i^k/\theta}$
- 效果：收敛快的任务降低学习率、收敛慢的提高，保持同步

**骨干学习率调节**（保证优化方向一致性）：
- 计算当前损失分布与历史损失分布的KL散度：$C = 1 - D_{KL}(P(cur\_L) \| P(his\_L))$
- 一致性得分 $C$ 高→当前batch稳定→可以大步更新；$C$ 低→不同任务学习难度失衡→谨慎更新
- 通过Sigmoid调节：$\gamma_i = 2 \cdot \text{Sigmoid}((C-b) \cdot \tau)$

### 基准数据集SOI-Det

合并SARDet-100K（SAR/水平框）+ DOTA-v1.0（光学/旋转框）+ DroneVehicle（红外/旋转框），采样比2:1:1。

## 实验关键数据

### 表1: SOI-Det基准主要结果（ConvNeXt-T骨干）

| 方法 | FLOPs | 参数量 | 整体mAP | @50 | @75 |
|------|-------|--------|---------|-----|-----|
| 3个独立模型组合 | 403G | 126M | 48.23 | 79.39 | 51.26 |
| 简单联合训练 | 403G | 66M | 47.05 | 77.56 | 50.11 |
| DA + ConvNeXt-T | 403G | 66M | 48.37 | 79.76 | 51.66 |
| UniDet (Partitioned) | 403G | 66M | 48.47 | 79.55 | 52.01 |
| Uncertainty loss | 403G | 66M | 48.79 | 79.99 | 52.50 |
| SM3Det (DSO only) | 403G | 66M | 49.40 | 80.19 | 52.93 |
| **SM3Det (完整)** | 487G | 178M | **50.20** | **80.68** | **53.79** |

- 完整SM3Det比3个独立模型组合提升 **+1.97 mAP**
- 仅DSO的轻量版本（无MoE、参数量不变）已超越所有SOTA方法

### 表5: 参数效率对比（不同骨干规模）

| 配置 | 参数量 | mAP |
|------|--------|-----|
| 3模型(Small) | 192M | 49.17 |
| SM3Det(Tiny) | 178M | **50.20** |
| 3模型(Base) | 309M | 50.18 |
| SM3Det(Small) | 275M | **50.28** |
| 3模型(Large) | 636M | 50.50 |
| SM3Det(Base) | 459M | **51.33** |
| SM3Det(Large) | 770M | **52.16** |

- SM3Det-Tiny（178M）超过3模型-Small（192M）的同时参数还少7.3%
- SM3Det-Base（459M）超过3模型-Large（636M），参数少27.8%

### 消融实验关键发现

- **专家数量与top-k**：8个专家+top-2为最优配置，平衡性能与计算效率
- **网格级 vs 图像级MoE**：网格级（50.20）显著优于图像级（48.25），证明空间细粒度路由对检测任务至关重要
- **MoE层位置**：后3个stage的偶数层加MoE效果最佳（49.53），全部层反而下降（49.47）
- **DSO必要性**：去掉DSO后mAP从50.20降至49.47
- **DSO超参数稳健性**：bias参数 $b$ 变化对性能影响很小，方法稳健

## 亮点

1. **新任务定义**：首次系统定义遥感M2Det任务，填补多模态统一检测的研究空白
2. **网格级MoE设计精巧**：不同于图像级粗粒度路由，网格级专家能感知空间局部模式，同时学习共享和模态专有表征
3. **DSO机制独特**：不同于GradNorm等修改损失权重/梯度的方法，DSO直接调节子模块学习率，控制更精细且高效
4. **即超即省**：单一SM3Det模型用更少参数超越多个独立模型组合
5. **泛化性强**：在ConvNeXt/VAN/LSKNet/PVT-v2等多种骨干和单/两阶段检测器上均有效
6. **专家激活可视化有洞察力**：SAR模态使用独有专家集，RGB与红外共享较多专家，符合模态特性认知

## 局限与展望

1. **缺少多光谱模态**：因大规模多光谱检测数据集稀缺，未能纳入多光谱成像实验
2. **仅限遥感场景**：虽然方法可迁移至医学影像、自动驾驶等多模态场景，但未实际验证
3. **MoE引入额外参数**：完整SM3Det的参数量（178M）比基线（66M）多了近3倍，虽然仍少于3个独立模型（126M），但部署时需权衡
4. **训练资源要求**：使用8块RTX 3090训练，计算资源需求不低
5. **数据集采样策略固定**：2:1:1的采样比例未做充分消融，可能不是最优

## 与相关工作的对比

### vs UniDet（统一标签空间多数据集检测）
UniDet使用分区检测头和统一标签空间进行多数据集训练，在通用目标检测（光学概念数据集）上有效。但面对遥感多模态图像（SAR/光学/红外具有根本不同的模式概念），UniDet（48.47 mAP）仅略高于简单联合训练（47.05），提升有限。SM3Det（50.20）通过MoE解决表征空间拥挤问题，通过DSO解决优化不一致问题，优势显著（+1.73 mAP）。

### vs DA网络（域特定SE注意力）
DA使用SE层作为域特定注意力机制，是此前多数据集检测的代表方法。但DA的域特定机制是硬编码的图像级路由，无法在空间特征层面做灵活的专家选择。在SOI-Det上DA（48.37）甚至不如Uncertainty loss（48.79），远逊于SM3Det（50.20）。

### vs GradNorm（梯度平衡多任务学习）
GradNorm通过调整各任务梯度大小来平衡学习。SM3Det的DSO则直接调节子模块学习率——对检测头用损失比率平衡收敛速率，对骨干用KL散度一致性分数调节更新步幅。这种双层策略既保证各任务同步收敛，又防止共享权重被某个任务的困难样本过度带偏。

## 评分
- 新颖性: ⭐⭐⭐⭐ — M2Det任务定义+网格级MoE+DSO三重创新
- 实验充分度: ⭐⭐⭐⭐⭐ — 多骨干/多检测器/详尽消融/可视化分析非常全面
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，方法动机充分，实验组织有逻辑
- 价值: ⭐⭐⭐⭐ — 对遥感多模态统一检测有开创性意义，方法可迁移至其他多模态场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection](../../CVPR2026/object_detection/a_semantically_disentangled_unified_model_for_multi-category_3d_anomaly_detectio.md)
- [\[CVPR 2026\] UniMMAD: Unified Multi-Modal and Multi-Class Anomaly Detection via MoE-Driven Feature Decompression](../../CVPR2026/object_detection/unimmad_unified_multi-modal_and_multi-class_anomaly_detection_via_moe-driven_fea.md)
- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](../../CVPR2026/object_detection/fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[CVPR 2026\] Rotation Invariant and Symmetry Aware Pixel Difference Network for Remote Sensing Object Detection](../../CVPR2026/object_detection/rotation_invariant_and_symmetry_aware_pixel_difference_network_for_remote_sensin.md)
- [\[ICCV 2025\] OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images](../../ICCV2025/object_detection/openrsd_towards_open-prompts_for_object_detection_in_remote_sensing_images.md)

</div>

<!-- RELATED:END -->
