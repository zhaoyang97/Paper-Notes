---
title: >-
  [论文解读] Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors
description: >-
  [CVPR 2025][3D视觉][3D重建] 提出 Pow3R，一个在 DUSt3R 基础上增强的通用 3D 视觉回归模型，能灵活接收相机内参、相对位姿、稀疏/稠密深度等任意组合的辅助信息，在多项 3D 视觉任务上取得 SOTA 并解锁原生分辨率推理等新能力。 DUSt3R 是 3D 视觉回归模型的里程碑…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D重建"
  - "DUSt3R"
  - "多模态先验"
  - "深度补全"
  - "位姿估计"
---

# Pow3R: Empowering Unconstrained 3D Reconstruction with Camera and Scene Priors

**会议**: CVPR 2025  
**arXiv**: [2503.17316](https://arxiv.org/abs/2503.17316)  
**代码**: [项目页面](https://europe.naverlabs.com/pow3r)  
**领域**: 3D视觉  
**关键词**: 3D重建, DUSt3R, 多模态先验, 深度补全, 位姿估计

## 一句话总结

提出 Pow3R，一个在 DUSt3R 基础上增强的通用 3D 视觉回归模型，能灵活接收相机内参、相对位姿、稀疏/稠密深度等任意组合的辅助信息，在多项 3D 视觉任务上取得 SOTA 并解锁原生分辨率推理等新能力。

## 研究背景与动机

DUSt3R 是 3D 视觉回归模型的里程碑，能从未标定、未定位的图像对中回归 3D 点图。然而它存在一个根本限制——仅接受 RGB 图像输入，无法利用实际应用中常有的额外信息：

- **相机标定信息浪费**：很多应用场景提供已标定的相机内参，但 DUSt3R 无法利用
- **深度传感器数据闲置**：RGB-D 相机或 LiDAR 提供的稀疏/稠密深度无法融入模型
- **相对位姿约束未利用**：IMU 或其他传感器提供的位姿估计无法作为额外约束
- **分辨率受限**：DUSt3R 训练于固定分辨率，无法泛化到更高分辨率
- **需要双向推理**：提取两张图像的深度需要两次前向传播(正反各一次)

## 方法详解

### 整体框架

Pow3R 基于 DUSt3R 的 ViT 编解码器架构，引入轻量级条件注入模块来接收辅助信息。编码器接收图像、内参和深度；解码器接收相对位姿。训练时随机采样辅助信息子集，使模型适应不同条件。此外，预测三个点图 $X^{1,1}$, $X^{2,1}$, $X^{2,2}$（多于 DUSt3R 的两个），实现单次推理即可提取所有信息。

### 关键设计1: 多模态辅助信息灵活注入 — 轻量级条件化

**功能**: 在不改变核心架构的前提下，将相机内参、深度图和相对位姿灵活注入到 ViT 中。

**核心思路**: 针对不同模态设计专用注入方式：
- **内参** $K$: 计算相机射线 $K^{-1}[i,j,1]$，patch 化后通过专用 MLP 注入编码器
- **深度** $D$: 归一化后与有效掩码 $M$ 拼接为 $[D', M] \in \mathbb{R}^{W \times H \times 2}$，patch 化注入编码器
- **相对位姿** $P_{12}$: 通过嵌入层+MLP 编码后添加到解码器的 CLS token

采用 "inject-1" 策略：每个模态的 MLP 仅在第一个 Transformer 块中注入，以 token-wise 加法方式融合。

**设计动机**: 训练时随机丢弃子集（"dropout" 辅助信息），使单一模型适应从无先验到完全先验的所有条件。实验发现 "inject-1" 与更深层注入效果相当但更高效。

### 关键设计2: 三点图预测 — 单次推理提取完整信息

**功能**: 通过额外预测 $X^{2,2}$（第二张图像在自身坐标系下的点图），实现单次推理提取两幅图像的深度、焦距和相对位姿。

**核心思路**: 解码器分支 Head$^2$ 同时输出 $X^{2,1}$, $X^{2,2}$ 及对应置信度。相对位姿通过 Procrustes 对齐 $X^{2,2}$ 和 $X^{2,1}$ 直接获得：

$$R^*, t^* = \arg\min_{\sigma,R,t} \sum_{i,j} \sqrt{C^{2,2}_{i,j} C^{2,1}_{i,j}} \|\sigma(RX^{2,2}_{i,j} + t) - X^{2,1}_{i,j}\|^2$$

**设计动机**: DUSt3R 需要两次前向传播 $(I_1, I_2)$ 和 $(I_2, I_1)$ 才能得到两幅图像的深度和位姿。三点图设计将推理效率提升一倍。Procrustes 对齐虽对噪声敏感，但在此场景下与 RANSAC+PnP 精度相当，速度提升一个数量级。

### 关键设计3: 原生分辨率推理 — 内参条件化的新能力

**功能**: 利用内参信息处理任意裁切，实现滑动窗口式高分辨率推理。

**核心思路**: 内参编码为相机射线，包含了裁切位置信息（即焦距和主点）。因此可以将高分辨率图像分成多个裁切块，每个块配合其裁切区域的内参独立处理，最后通过重叠区域的中值尺度对齐和置信度加权混合拼接。

**设计动机**: DUSt3R 训练于固定分辨率且假设主点居中，无法处理非中心裁切或高分辨率图像。内参条件化自然地解除了这一限制，无需额外设计。

### 损失函数

$$\mathcal{L} = \mathcal{L}^{\text{conf}}(1,1) + \mathcal{L}^{\text{conf}}(2,1) + \beta \mathcal{L}^{\text{conf}}(2,2)$$

其中置信度加权回归损失 $\mathcal{L}^{\text{conf}}(n,m) = \sum C^{n,m}_{i,j} \mathcal{L}^{\text{regr}}_{i,j}(n,m) - \alpha \log C^{n,m}_{i,j}$，使用尺度不变的 3D 回归损失。

## 实验关键数据

### 多视图深度预测 (ScanNet++)

| 方法 | AbsRel↓ | δ<1.25↑ |
|------|---------|---------|
| DUSt3R | 0.068 | 0.967 |
| MASt3R | 0.061 | 0.974 |
| **Pow3R (无先验)** | 0.063 | 0.972 |
| **Pow3R (有内参+深度)** | **0.041** | **0.988** |

### 相对位姿估计 (Map-free benchmark)

| 方法 | AUC@5°↑ | AUC@10°↑ | 推理时间 |
|------|---------|----------|---------|
| DUSt3R + PnP | 52.3 | 64.1 | ~100ms |
| MASt3R + PnP | 60.1 | 71.5 | ~100ms |
| **Pow3R + Procrustes** | **62.8** | **73.2** | **~10ms** |

### 深度补全 (NYUv2)

| 方法 | RMSE↓ (500点) | RMSE↓ (200点) |
|------|-------------|-------------|
| CompletionFormer | 0.089 | 0.105 |
| **Pow3R** | **0.082** | **0.094** |

### 关键发现

- 无先验时 Pow3R 与 DUSt3R 性能持平，有先验时显著提升（**40%** AbsRel 改善with内参+深度）
- Procrustes 位姿估计精度与 RANSAC+PnP 相当但**快约 10 倍**
- 原生分辨率推理产生更锐利、更精细的 3D 重建，无需重训练
- 深度补全任务超越专门方法，证明了通用性
- 随机丢弃辅助信息的训练策略是适应不同条件的关键

## 亮点与洞察

1. **灵活的辅助信息注入是实用创新**：实际应用中各种先验信息的可用性差异很大，单一模型适应所有条件是理想方案
2. **三点图设计一箭双雕**：效率翻倍的同时使 Procrustes 位姿估计成为可能
3. **内参条件化解锁高分辨率推理**是一个意外但极有价值的副产品
4. **与 DUSt3R 生态的兼容性**：MASt3R、MonST3R 等后续工作都可从中受益

## 局限与展望

- 高分辨率滑动窗口推理需要手动设置裁切策略和重叠区域
- Procrustes 对齐对严重遮挡场景可能不够鲁棒
- 目前只处理图像对，多视图扩展需要全局对齐后处理
- 辅助信息的噪声容忍度值得更深入分析
- 未来可探索主动选择最有价值的辅助信息进行查询

## 相关工作与启发

- **DUSt3R / MASt3R**: 基础 3D 回归框架，Pow3R 直接在其上扩展
- **UniDepth**: 单目深度估计中可选内参条件化的先驱
- **CompletionFormer**: CNN+ViT 深度补全方法
- **RayDiffusion**: 基于射线扩散的位姿估计，内参射线编码的启发来源

## 评分

⭐⭐⭐⭐⭐ — 工程和学术价值都很高。在 DUSt3R 这一强大基础上做出了正确且重要的增强方向。灵活的先验注入使一个模型适应广泛场景；三点图设计效率翻倍；原生分辨率推理是令人兴奋的新能力。实验覆盖多个下游任务且都取得了 SOTA。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Scene Coordinate Reconstruction Priors](../../ICCV2025/3d_vision/scene_coordinate_reconstruction_priors.md)
- [\[CVPR 2025\] Empowering Large Language Models with 3D Situation Awareness](empowering_large_language_models_with_3d_situation_awareness.md)
- [\[CVPR 2025\] MUSt3R: Multi-view Network for Stereo 3D Reconstruction](must3r_multi-view_network_for_stereo_3d_reconstruction.md)
- [\[CVPR 2025\] MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors](mast3r-slam_real-time_dense_slam_with_3d_reconstruction_priors.md)
- [\[CVPR 2025\] Fast3R: Towards 3D Reconstruction of 1000+ Images in One Forward Pass](fast3r_towards_3d_reconstruction_of_1000_images_in_one_forward_pass.md)

</div>

<!-- RELATED:END -->
