---
title: >-
  [论文解读] Masked Angle-Aware Autoencoder for Remote Sensing Images
description: >-
  [ECCV 2024][遥感][自监督学习] 提出 MA3E，在 MAE 预训练中显式引入角度变化（通过 scaling center crop 构建旋转裁剪），并用最优传输损失自动分配重建目标，使模型感知遥感目标的多样角度，学习旋转不变表示。 遥感（RS）图像与自然图像存在显著的域差距。自然图像中的物体因重力通常具有固定朝…
tags:
  - "ECCV 2024"
  - "遥感"
  - "自监督学习"
  - "掩码图像建模"
  - "旋转不变性"
  - "最优传输"
  - "遥感图像"
---

# Masked Angle-Aware Autoencoder for Remote Sensing Images

**会议**: ECCV 2024  
**arXiv**: [2408.01946](https://arxiv.org/abs/2408.01946)  
**代码**: [GitHub](https://github.com/benesakitam/MA3E)  
**领域**: 遥感  
**关键词**: 自监督学习, 掩码图像建模, 旋转不变性, 最优传输, 遥感图像

## 一句话总结

提出 MA3E，在 MAE 预训练中显式引入角度变化（通过 scaling center crop 构建旋转裁剪），并用最优传输损失自动分配重建目标，使模型感知遥感目标的多样角度，学习旋转不变表示。

## 研究背景与动机

遥感（RS）图像与自然图像存在显著的域差距。自然图像中的物体因重力通常具有固定朝向，而遥感图像从鸟瞰视角拍摄，物体呈现**各种角度**——同一目标在不同角度下呈现截然不同的形状和外观。

现有的遥感自监督方法（如 SatMAE、ScaleMAE、RingMo）虽然考虑了多分辨率、多尺度、多光谱等因素，但**忽视了遥感目标的角度多样性**。这些方法仅关注像素值重建，角度信息的学习只是隐式地伴随重建过程。

作者通过实验（Fig.1）直观展示了问题所在：用标准 MAE 预训练后做旋转目标检测，模型仅在水平方向（0° 或 90°）附近的目标上表现较好，而对有大倾斜角（10°-80°）的目标检测性能明显下降。这说明现有 MIM 方法未能有效学习角度感知表示。

**核心问题**：如何在预训练阶段让模型显式感知和学习遥感目标的角度信息，从而获得旋转不变的视觉表示？

## 方法详解

### 整体框架

MA3E 沿用 MAE 的非对称编码器-解码器架构。核心改进是：在原始图像上构建一个具有随机方向的**旋转裁剪区域（rotated crop）**，将其嵌入原图形成合成图像作为输入，训练目标是**重建原始图像**（即同时完成像素重建和角度还原）。

流程：原图 → scaling center crop 创建旋转裁剪 → 替换原位置场景 → 添加角度嵌入 → 分别掩码 → 编码解码 → 背景用 MSE 损失重建 + 旋转区域用 OT 损失重建。

### 关键设计

1. **Scaling Center Crop（缩放中心裁剪）**：构建旋转裁剪区域的核心操作。对图像中一个边长为 $h$ 的正方形区域，以随机角度旋转后，取其**最大内切圆中的最大内接正方形**作为旋转裁剪，边长 $a = \frac{\sqrt{2}}{2}h$。这样做的动机是：直接随机旋转会导致三个问题——(i) 出现无意义的零值背景，(ii) 场景丢失，(iii) 场景尺度变化。而 scaling center crop 在保留主要场景的同时引入了任意角度变化。旋转裁剪替换原位置后形成合成图像，为模型提供显式的角度变化信号。

2. **Angle Embedding（角度嵌入）**：为旋转裁剪中的每个 patch 添加一个可学习的角度嵌入向量（同一裁剪区域内共享）。该嵌入作为**隐式提示**，让模型感知旋转裁剪的角度变化，同时将旋转区域与背景区分开来。这是一个轻量但有效的设计——不需要显式的角度标签，仅通过额外的嵌入向量让模型自动学习角度感知。

3. **分区域随机掩码（Separate Random Masking）**：将旋转裁剪的 $N_r$ 个 patch 和背景的 $N_b$ 个 patch 分别以 75% 的比率独立掩码。动机：标准 MAE 的全局随机掩码可能导致旋转裁剪区域的 patch 被过度甚至完全掩盖（因为旋转区域相对较小），从而无法学习角度信息。分区域掩码确保两个区域都有足够的可见 patch。

4. **Optimal Transport Loss（OT 损失）**：旋转裁剪后，裁剪区域的 patch 与原始图像同位置 patch 存在场景偏移，直接用 MSE 重建会引入明显偏差。MA3E 将此视为最优传输问题：把 $N_r$ 个原始图像 patch 视为供应商，$N_r$ 个预测 patch 视为需求方，运输代价为 L2 距离：

$$c_{ij} = \|r_i - \hat{r}_j\|_2^2$$

使用 Sinkhorn-Knopp 快速迭代算法求解运输方案 $\Omega$，OT 损失自动为每个预测 patch 分配相似的原始 patch 作为重建目标：

$$\mathcal{L}_{OT}(r, \hat{r}) = \sum_{i=1}^{N_r}\sum_{j=1}^{N_r} \|r_i - \hat{r}_j\|_2^2 \omega_{ij}$$

### 损失函数 / 训练策略

总损失由背景 MSE 损失和旋转区域 OT 损失组成：

$$\mathcal{L}_{rec} = \mathcal{L}_{MSE}(b^m, \hat{b}^m) + \mathcal{L}_{OT}(r, \hat{r})$$

- 背景区域：仅对被掩码的 patch 计算 MSE 损失（与标准 MAE 一致）
- 旋转裁剪区域：对所有 patch（包括可见和掩码的）计算 OT 损失

预训练在 MillionAID 数据集（约 99 万张 RS 图像）上进行，输入 $224 \times 224$，patch size=16，旋转裁剪边长 $a=96$，旋转范围 $[-45°, +45°]$，编码器 ViT-B，解码器 8 层 ViT blocks（512-D）。

## 实验关键数据

### 主实验

**场景分类（Fine-tuning）**：

| 数据集 | 指标 | MA3E (300ep) | MA3E (1600ep) | MAE (1600ep) | MAE+RVSA (1600ep) |
|--------|------|-------------|--------------|-------------|-------------------|
| NWPU-RESISC45 | Top-1 Acc | 95.77 | **96.23** | 95.40 | 95.49 |
| AID | Top-1 Acc | 98.44 | **99.04** | 98.36 | 98.33 |
| UC Merced | Top-1 Acc | 99.05 | **99.81** | 99.44 | 99.70 |

**旋转目标检测 & 语义分割**：

| 数据集 | 指标 | MA3E (1600ep) | MAE+RVSA (1600ep) | MAE+ViTAE+RVSA (1600ep) |
|--------|------|--------------|-------------------|------------------------|
| DOTA1.0 | mAP | **79.47** | 78.75 | 78.96 |
| DIOR-R | mAP | **71.82** | 70.67 | 70.95 |
| iSAID | mIoU | **64.06** | 63.76 | 63.48 |
| Potsdam | mF1 | **91.50** | 90.60 | 91.22 |

### 消融实验

**各组件消融（300 epochs, ViT-B）**：

| 配置 | NU45 (ft) | DOTA1.0 (det) | iSAID (seg) | 说明 |
|------|-----------|--------------|-------------|------|
| MAE baseline | 95.31 | 75.85 | 60.96 | 标准 MAE |
| + SCC | 95.43 | 76.12 | 61.24 | 增加 scaling center crop |
| + SCC + AE | 95.47 | 76.41 | 61.86 | 增加角度嵌入 |
| + SCC + OT | 95.36 | 76.46 | 61.88 | 增加 OT 损失 |
| + SCC + Mask. | 95.06 | 77.23 | 62.17 | 增加分区域掩码 |
| + SCC + AE + Mask. | 95.53 | 76.70 | 61.93 | 三组件组合 |
| **MA3E (全部)** | **95.77** | **77.93** | **62.74** | 所有组件 |

**旋转范围消融**：

| 旋转范围 | NU45 (ft) | DOTA1.0 (det) | iSAID (seg) |
|---------|-----------|--------------|-------------|
| [-30°, +30°] | 95.78 | 77.68 | 62.49 |
| **[-45°, +45°]** | **95.77** | **77.93** | **62.74** |
| [-60°, +60°] | 95.32 | 77.22 | 62.55 |
| [-90°, +90°] | 94.89 | 76.45 | 61.90 |

### 关键发现

- MA3E 显著提升了 10°-80° 大倾斜角目标的检测 AP50（Fig.1），验证了角度感知的有效性
- 旋转裁剪边长 $a=96$（36 patches）效果最佳，过大或使用多个裁剪反而降低性能
- Scaling center crop 比简单随机旋转在三个任务上分别提升 1.95/1.79/1.51
- ±45° 旋转范围最优，过大范围使角度还原过于困难

## 亮点与洞察

- **问题定义精准**：遥感目标角度多样性是一个被忽视但重要的问题，直接关系到旋转目标检测等任务
- **OT 损失设计巧妙**：将场景偏移后的重建问题转化为最优传输问题，避免了 one-to-one 硬匹配的偏差
- **计算代价小**：相比 MAE，仅增加约 0.2 小时/epoch 的训练时间，额外参数极少
- 仅用简单的 ViT-B backbone 即超越了使用更复杂架构（ViTAE+RVSA）的方法

## 局限与展望

- 角度感知对**人造目标**（车辆、建筑等）更有价值，对大面积自然地物（林地、水域）收益有限
- 未考虑**尺度因素**——遥感图像的多尺度特性与角度同样重要，两者的联合建模值得探索
- 旋转裁剪位置的选择较为随机，虽然 selective search 能带来微小提升但代价较高，更高效的目标区域选择策略有待研究
- 仅在 ViT-B 上验证，未探索更大模型（ViT-L/H）的扩展性

## 相关工作与启发

- 对比 MixMAE（混合多张图像输入重建）的思路，MA3E 是在**单图内**创建合成输入，更适合遥感场景
- OT 在预训练损失中的应用可推广到其他存在空间偏移的场景（如形变、透视变换）
- 方法启发：针对特定领域图像的物理特性（如遥感的俯视角度、医学的多模态）设计定制化预训练策略

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 将角度感知引入 MIM 预训练，OT 损失处理旋转重建是创新点
- **实验充分度**: ⭐⭐⭐⭐⭐ — 7 个数据集、3 个下游任务、详尽的消融实验
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰，动机充分，图示直观
- **价值**: ⭐⭐⭐⭐ — 对遥感预训练有实际推动作用，方法简洁有效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images](../../ICCV2025/remote_sensing/smarties_spectrum-aware_multi-sensor_auto-encoder_for_remote_sensing_images.md)
- [\[CVPR 2026\] NeighborMAE: Exploiting Spatial Dependencies between Neighboring Earth Observation Images in Masked Autoencoders Pretraining](../../CVPR2026/remote_sensing/neighbormae_exploiting_spatial_dependencies_between_neighboring_earth_observatio.md)
- [\[ECCV 2024\] Learning Representations of Satellite Images From Metadata Supervision](learning_representations_of_satellite_images_from_metadata_supervision.md)
- [\[NeurIPS 2025\] ChA-MAEViT: Unifying Channel-Aware Masked Autoencoders and Multi-Channel Vision Transformers for Improved Cross-Channel Learning](../../NeurIPS2025/remote_sensing/chamaevit_unifying_channelaware_masked_autoencoders_and_mult.md)
- [\[CVPR 2026\] SegEarth-R2: Towards Comprehensive Language-guided Segmentation for Remote Sensing Images](../../CVPR2026/remote_sensing/segearth-r2_towards_comprehensive_language-guided_segmentation_for_remote_sensin.md)

</div>

<!-- RELATED:END -->
