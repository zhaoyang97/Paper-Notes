---
title: >-
  [论文解读] REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion
description: >-
  [AAAI 2026][目标检测][雷达感知] 将 DiffusionDet 的 2D BBox 扩散范式提升到 3D 雷达空间，提出 REXO 框架：通过含噪 3D BBox 的投影引导显式跨视图雷达特征关联，并引入地面约束减少扩散参数，在 HIBER 和 MMVR 两个室内雷达数据集上分别超越 SOTA +4.22 AP 和 +11.02 AP。
tags:
  - "AAAI 2026"
  - "目标检测"
  - "雷达感知"
  - "多视图融合"
  - "扩散模型"
  - "3D 目标检测"
  - "室内人体检测"
  - "跨视图特征关联"
---

# REXO: Indoor Multi-View Radar Object Detection via 3D Bounding Box Diffusion

**会议**: AAAI 2026  
**arXiv**: [2511.17806](https://arxiv.org/abs/2511.17806)  
**代码**: [https://github.com/merlresearch/radar-bbox-diffusion](https://github.com/merlresearch/radar-bbox-diffusion)  
**领域**: 目标检测  
**关键词**: 雷达感知, 多视图融合, 扩散模型, 3D 目标检测, 室内人体检测, 跨视图特征关联  

## 一句话总结

将 DiffusionDet 的 2D BBox 扩散范式提升到 3D 雷达空间，提出 REXO 框架：通过含噪 3D BBox 的投影引导显式跨视图雷达特征关联，并引入地面约束减少扩散参数，在 HIBER 和 MMVR 两个室内雷达数据集上分别超越 SOTA +4.22 AP 和 +11.02 AP。

## 研究背景与动机

室内雷达感知因成本低、隐私侵入小而备受关注，但多视图（水平+垂直）雷达热力图的跨视图特征融合一直是核心难题：

**RFMask**：在水平视图上用 Faster R-CNN 生成 proposal，再配固定高度的垂直窗口进行配对 → 隐式且受限

**RETR**：基于 DETR 的 decoder cross-attention 将 query 隐式关联到两个视图 → 特征匹配模糊

**DiffusionDet 直接移植到雷达**：在水平雷达热力图上做 2D BBox 扩散，仍需固定高度垂直配对

以上方法的共同问题是**跨视图关联是隐式的**，在复杂室内场景下容易出现模糊匹配，导致检测性能退化。特别是当多个被检测人处于相同深度时，垂直视图的反射信号高度混叠，隐式方案更难分辨。

## 方法详解

### 整体框架

REXO 将扩散过程的参数从 2D 平面提升到 3D 雷达空间 $\boldsymbol{x}_t = \{c_x^t, c_y^t, c_z^t, w^t, h^t, d^t\}^\top \in \mathbb{R}^6$。训练时将 GT 3D BBox 加噪，推理时从随机 3D BBox 逐步去噪。两个雷达视图的特征图分别由共享参数的 ResNet 骨干提取，通过 FPN 生成多尺度特征。

### 关键设计 1：显式跨视图雷达特征关联

在每个扩散时间步，将含噪 3D BBox $\boldsymbol{x}_t$ 分别投影到水平/垂直两个 2D 视图得到对应的 2D BBox，然后通过 RoIAlign 从各自的特征图裁剪出 $C \times r \times r$ 的特征，拼接得到 $\boldsymbol{Z}_{\text{radar}}^{\text{crop}} \in \mathbb{R}^{C \times r \times 2r}$。

**核心优势**：这种 BBox 引导的关联方式复杂度随视图数**线性**增长（proposal 或 query 方案是**二次**增长），语义上也更加精准——因为 3D BBox 已经编码了空间位置信息。

### 关键设计 2：雷达条件去噪检测器 DenoisingDet

将拼接后的跨视图雷达特征 $\boldsymbol{Z}_{\text{radar}}^{\text{crop}}$ 注入时间依赖的 Predictor（自注意力 + 动态卷积 + 时间嵌入），再通过 BBox Head 预测 3D BBox 偏移量和 Class Head 预测分类。去噪步骤**天然地以跨视图雷达特征为条件**，无需额外设计融合模块。

### 关键设计 3：地面约束 (Ground-Level Constraint)

利用室内场景中"人站立在地面"的先验知识，将垂直方向中心 $c_y^t$ 直接设置为 $h^t/2$，从而将扩散参数从 6 维降至 5 维。这一约束不仅减少了搜索空间，还使 3D 和 2D 的梯度联合流动，实现了更快的收敛和更好的泛化。

### 损失函数

采用几何感知的双域损失：

$$\mathcal{L}_{\text{box}}^{\text{GA}} = \lambda_{3D} \mathcal{L}_{\text{box}}^{3D}(\boldsymbol{x}_{\text{radar}}, \hat{\boldsymbol{x}}_{\text{radar}}) + \lambda_{2D} \mathcal{L}_{\text{box}}^{2D}(\boldsymbol{b}_{\text{image}}, \hat{\boldsymbol{b}}_{\text{image}})$$

其中每个 BBox 损失 = GIoU 损失 + L1 损失的加权组合。3D BBox 通过标定的旋转矩阵和平移向量映射到相机坐标系后，投影到图像平面得到 2D BBox；再经可学习的 Refinement 模块修正投影过大问题。匈牙利匹配用于确定最优分配。

## 实验

### 主实验表：MMVR 数据集 (4 种数据划分)

| 方法 | P1S1 AP | P1S2 AP | P2S1 AP | P2S2 AP |
|------|---------|---------|---------|---------|
| RFMask | 25.53 | 24.46 | 31.37 | 6.03 |
| RFMask3D | 34.84 | 30.75 | 39.89 | 12.26 |
| DETR | 35.64 | 28.51 | 29.53 | 9.29 |
| RETR | 39.62 | 30.16 | 46.75 | 12.45 |
| **REXO** | **39.23** | **36.48** | **48.35** | **23.47** |

在最具挑战性的 P2S2（完全未见环境）上，REXO 将 AP 从 12.45 提升到 23.47，**提升 +11.02**，展示了极强的泛化能力。

### 消融实验表：P2S2 on MMVR

| 消融维度 | 配置 | AP |
|----------|------|-----|
| 地面约束 | ✗ | 22.67 |
| 地面约束 | ✓ | **23.47** |
| λ₃D=0, λ₂D=1 | 纯 2D 监督 | 0.98 |
| λ₃D=1, λ₂D=0.1 | 弱 2D | 15.55 |
| λ₃D=1, λ₂D=1 | 均衡监督 | **23.47** |
| 推理步数 S=1 | — | 23.48 |
| 推理步数 S=10 | — | **24.27** |
| DiffusionDet (水平) | 基线 | 20.75 |
| REXO (双视图) | — | **23.47** |

### 关键发现

1. **3D 监督是关键**：没有 3D 损失 ($\lambda_{3D}=0$) 时 AP 几乎为 0，说明雷达坐标系中的 3D 精度对图像平面 2D 预测至关重要
2. **推理 BBox 数量鲁棒性**：REXO 在推理 BBox 从 2 增加到 80 时性能下降极小（23.48→21.70），而 RETR 急剧退化（12.45→2.16）
3. **深度近似挑战**：两人深度差 <20cm 时 AP 从 23.47 骤降到 9.93，揭示了垂直视图信号混叠的根本困难
4. **HIBER 数据集**：REXO 达到 25.33 AP，超过 RETR 的 22.09，**+3.24 AP**

## 亮点

- **概念简洁**：将 2D 扩散直接提升到 3D 空间的想法自然且优雅，一步到位解决跨视图关联
- **显式关联的计算优势**：线性复杂度 vs. 二次复杂度，可扩展到更多视图
- **地面约束的巧妙利用**：物理先验融入扩散过程减少参数空间，同时加速训练收敛
- **泛化能力突出**：在完全未见环境（P2S2）上的压倒性优势证明了方法的鲁棒性
- **代码开源**

## 局限性

1. 两人深度近似时性能严重下降，垂直视图的角度分辨率不足以区分重叠反射
2. 地面约束假设人站在地面上，跳跃或站在障碍物上时不准确
3. 仅在室内雷达场景验证，未扩展到户外自动驾驶雷达
4. 推理步数增加带来显著延迟（1 步 17 FPS → 10 步 2 FPS）

## 相关工作

- **DiffusionDet** (Chen et al., 2023): 2D 图像检测的扩散模型范式，REXO 的直接基础
- **RETR** (Yataka et al., 2024): 基于 DETR 的多视图雷达检测 Transformer，REXO 的主要基线
- **RFMask** (Wu et al., 2023): Faster R-CNN + 固定高度垂直窗口的跨视图方案
- **Diffusion-SS3D** (Ho et al., 2023): 半监督 3D 检测中的扩散模型应用

## 评分

- 新颖性: ⭐⭐⭐⭐ （2D→3D 扩散提升 + 跨视图关联思路独到）
- 实验充分度: ⭐⭐⭐⭐ （两个数据集 + 丰富消融 + 可视化分析）
- 写作质量: ⭐⭐⭐⭐ （图示清晰，技术路线描述完整）
- 价值: ⭐⭐⭐⭐ （室内雷达感知的实用推进，开源代码增加可复现性）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](../../ECCV2024/object_detection/adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)
- [\[AAAI 2026\] CountSteer: Steering Attention for Object Counting in Diffusion Models](countsteer_steering_attention_for_object_counting_in_diffusion_models.md)
- [\[CVPR 2026\] GS-CLIP: Zero-shot 3D Anomaly Detection by Geometry-Aware Prompt and Synergistic View Representation Learning](../../CVPR2026/object_detection/gs-clip_zero-shot_3d_anomaly_detection_by_geometry-aware_prompt_and_synergistic_.md)
- [\[AAAI 2026\] AerialMind: Towards Referring Multi-Object Tracking in UAV Scenarios](aerialmind_towards_referring_multi-object_tracking_in_uav_sc.md)
- [\[CVPR 2026\] A Semantically Disentangled Unified Model for Multi-category 3D Anomaly Detection](../../CVPR2026/object_detection/a_semantically_disentangled_unified_model_for_multi-category_3d_anomaly_detectio.md)

</div>

<!-- RELATED:END -->
