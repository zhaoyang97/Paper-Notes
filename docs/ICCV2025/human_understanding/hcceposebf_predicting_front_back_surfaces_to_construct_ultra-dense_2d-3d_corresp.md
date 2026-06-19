---
title: >-
  [论文解读] HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation
description: >-
  [ICCV 2025][人体理解][位姿估计] 提出同时预测物体前后表面的3D坐标并在两表面间密集采样，构建超密集2D-3D对应关系，配合新颖的层级连续坐标编码（HCCE），在BOP七大核心数据集上超越现有SOTA方法。 在已知物体的6D位姿估计中，主流pipeline是通过神经网络预测物体表面的3D坐标…
tags:
  - "ICCV 2025"
  - "人体理解"
  - "位姿估计"
  - "2D-3D对应"
  - "前后表面预测"
  - "层级连续编码"
  - "PnP"
---

# HccePose(BF): Predicting Front & Back Surfaces to Construct Ultra-Dense 2D-3D Correspondences for Pose Estimation

**会议**: ICCV 2025  
**arXiv**: [2510.10177](https://arxiv.org/abs/2510.10177)  
**代码**: [https://github.com/WangYuLin-SEU/HCCEPose](https://github.com/WangYuLin-SEU/HCCEPose)  
**领域**: 人体理解  
**关键词**: 位姿估计, 2D-3D对应, 前后表面预测, 层级连续编码, PnP

## 一句话总结

提出同时预测物体前后表面的3D坐标并在两表面间密集采样，构建超密集2D-3D对应关系，配合新颖的层级连续坐标编码（HCCE），在BOP七大核心数据集上超越现有SOTA方法。

## 研究背景与动机

在已知物体的6D位姿估计中，主流pipeline是通过神经网络预测物体表面的3D坐标，然后建立2D-3D对应关系，最后用PnP算法求解位姿。现有方法存在两个关键局限：

**只利用前表面**：现有方法仅预测物体正面可见表面的3D坐标，忽略了背面和内部区域的潜在信息。更密集的2D-3D对应关系能帮助RANSAC-PnP求解器得到更精确的位姿

**编码精度不足**：现有的层级二值编码（如ZebraPose）中，神经网络在明暗条纹边界附近难以准确学习二值码，导致坐标预测精度受限

## 方法详解

### 整体框架

输入裁剪后的RGB图像，通过神经网络同时预测物体mask、前表面坐标和后表面坐标。前后表面坐标通过HCCE编码为多级连续码由网络预测，推理时将连续码转为二值码并解码得到表面坐标。利用预测的前后表面坐标在两者之间密集采样3D点，构建超密集2D-3D对应关系，最后通过RANSAC-PnP求解位姿。

### 关键设计

1. **超密集2D-3D对应关系构建**：网络同时预测前表面坐标 $\tilde{Q}_f$ 和后表面坐标 $\tilde{Q}_b$。对于同一2D坐标对应的前后3D坐标 $\tilde{q}_1$ 和 $\tilde{q}_2$，计算采样数 $n = \lfloor \|\tilde{q}_1 - \tilde{q}_2\|_2 / \bar{d} \rfloor$（$\bar{d}$ 为最近点对的平均距离），然后均匀插值采样：$s(\tilde{q}_1, \tilde{q}_2, a) = a \times \tilde{q}_1 + (1-a)\tilde{q}_2$。为避免多个3D点共享同一2D投影导致PnP不可靠，约束每次RANSAC迭代中每个2D像素最多采样一个3D点。

2. **层级连续坐标编码（HCCE）**：对坐标分量 $x, y, z$ 分别编码为多级连续码。第1级 $Cx_{1,k} = x_k$，后续通过镜像操作生成更高级别的连续码：当 $x_k < 0.5$ 时复制前级码，当 $x_k \geq 0.5$ 时镜像前级码。关键优势是消除了二值编码中的明暗条纹边界，使网络更容易学习。解码时先将连续码转为二值码：$Bx_{i,k} = g(Cx_{i,k})$ 或 $1 - g(Cx_{i,k})$（取决于前一级是否经过镜像），然后按 $x_k \approx \sum_{i=1}^{8} 2^{-k} \times Bx_{i,k}$ 恢复坐标。

3. **基于多直方图的层级学习**：为每个坐标分量分别计算误差直方图，记录不同层级的错误预测比例 $r_{f,x,i}$，然后计算权重 $h_{f,x,i} = \exp(\sigma \cdot \min\{r_{f,x,i}, 0.5 - r_{f,x,i}\})$。归一化后得到各级损失权重，使训练从低级到高级渐进式学习。相比ZebraPose的单直方图，多直方图方法更稳定。

### 损失函数 / 训练策略

总损失 $L = L_M + \gamma \times (L_{xyz}^{Front} + L_{xyz}^{Back})$：
- **Mask损失** $L_M$：L1范数计算预测mask与GT之间的差异
- **层级损失**：对前后表面分别计算，以前表面x分量为例：$L_x^{Front} = \sum_{i=1}^{8} (w_{f,x,i} \cdot \sum_{j=1}^{n} \|Cx_{f,i,j} - \widetilde{Cx}_{f,i,j}\|_1)$

网络输出49个通道（前表面3×8 + 后表面3×8 + mask 1）。使用ResNet34（消融实验）或EfficientNet-B4（对比实验），输入/输出分辨率256×256/128×128。每个物体单独训练约24小时（NVIDIA RTX 4090）。

## 实验关键数据

### 主实验 (表格)

在BOP七大核心数据集上与SOTA方法对比（BOP Score，%）：

| 方法 | LM-O | T-LESS | TUD-L | IC-BIN | ITODD | HB | YCB-V | Core平均 |
|------|------|--------|-------|--------|-------|----|-------|----------|
| GPose | 69.9 | 79.9 | 83.1 | 62.6 | 46.0 | 87.6 | 80.9 | 72.9 |
| ZebraPose | 72.9 | 82.1 | 85.0 | 59.2 | 50.4 | 92.2 | 82.8 | 74.9 |
| **Ours (RGB)** | **75.5** | **85.6** | **86.9** | **63.5** | **54.2** | 91.9 | **83.9** | **77.3** |
| GDRNPP (RGB-D) | 79.2 | 87.2 | 93.6 | 70.2 | 58.8 | 90.9 | 83.4 | 80.5 |
| **Ours (RGB→RGB-D)** | **80.5** | **87.9** | 94.4 | **72.4** | **73.4** | **93.1** | **91.1** | **84.7** |

RGB模式下Core平均BOP Score超越ZebraPose 2.4%，RGB训练RGB-D测试超越GDRNPP 4.2%。

### 消融实验 (表格)

IC-BIN数据集上不同编码方法消融（ADD(-S) AR%）：

| 方法 | AR of ADD(-S) | AR of ADD-S | AUC ADD(-S) | AUC ADD-S |
|------|---------------|-------------|-------------|-----------|
| ZebraPose (表面编码) | 55.85 | 61.82 | 72.91 | 76.40 |
| HBCE+f (坐标编码) | 56.82 | 63.50 | 73.15 | 76.56 |
| HCCE+f(h0) 无权重调整 | 61.35 | 67.00 | 76.35 | 79.82 |
| HCCE+f(h1) 单直方图 | 60.44 | 65.23 | 74.92 | 78.68 |
| **HCCE+f(h3) 多直方图** | **61.95** | **68.30** | **77.67** | **81.11** |

前后表面信息的消融（LM-O/TUD-L/IC-BIN平均BOP AP%）：

| 配置 | 平均AP |
|------|--------|
| 仅前表面(f) | 82.2 |
| 仅后表面(b) | 81.3 |
| 前后表面(bf) | 82.6 |
| **前后表面+采样(bfu)** | **83.3** |

### 关键发现

- 编码坐标分量（HBCE）比编码表面区域（ZebraPose）更有效，ADD(-S)提升0.97%
- HCCE相比HBCE进一步提升5.13%，证明连续码比二值码更易学习
- 多直方图权重调整相比不调整提升0.6%，而单直方图反而降低0.91%
- 同时利用前后表面并密集采样（bfu）相比仅用前表面提升1.1% AP
- 2D分割任务上也超越ZebraPose 3.7%，说明前后表面预测有助于更精确的物体定位

## 亮点与洞察

- **前后表面预测是一个简单但有效的想法**：现有方法都只考虑前表面，而几何上后表面信息确实能为PnP提供更受约束的求解空间
- **HCCE编码消除了二值码的条纹边界问题**，是对ZebraPose编码方案的重要改进
- 多直方图层级学习策略使训练更稳定，权重峰值从低级到高级逐步转移，实现了自然的课程学习
- 方法框架通用性强，可以与不同backbone（ResNet34/EfficientNet-B4）配合

## 局限与展望

- 每个物体需要单独训练一个网络（约24小时），不适合大规模场景（数百个物体）
- 仅针对已知物体（seen objects），未涉及未知物体的位姿估计
- 前后表面定义依赖渲染深度测试（GL_LESS/GL_GREATER），对透明或自遮挡严重的物体可能不适用
- 推理速度约30ms/物体，在多物体场景中可能成为瓶颈

## 相关工作与启发

- ZebraPose提出的层级二值编码是重要的前序工作，本文的HCCE是其自然演进
- StereoPose也预测前后表面但仅用于透明物体的立体图像，本文将其推广到通用物体的单目RGB
- 未来可以结合foundation model（如SAM/DINOv2）的特征来替代per-object训练
- 超密集对应关系的思路可推广到6DoF抓取、机器人操作等需要精确位姿的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ 前后表面预测+HCCE编码两个创新点互相配合，设计自然
- **实验充分度**: ⭐⭐⭐⭐⭐ BOP七大数据集全面评测，消融实验细致
- **写作质量**: ⭐⭐⭐⭐ 逻辑清晰，公式推导完整
- **价值**: ⭐⭐⭐⭐ 实际提升显著，技术路线可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](../../NeurIPS2025/human_understanding/pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[CVPR 2026\] Differentially Private 2D Human Pose Estimation](../../CVPR2026/human_understanding/differentially_private_2d_human_pose_estimation.md)
- [\[NeurIPS 2025\] Learning Dense Hand Contact Estimation from Imbalanced Data](../../NeurIPS2025/human_understanding/learning_dense_hand_contact_estimation_from_imbalanced_data.md)
- [\[ICCV 2025\] High-Resolution Spatiotemporal Modeling with Global-Local State Space Models for Video-Based Human Pose Estimation](high-resolution_spatiotemporal_modeling_with_global-local_state_space_models_for.md)

</div>

<!-- RELATED:END -->
