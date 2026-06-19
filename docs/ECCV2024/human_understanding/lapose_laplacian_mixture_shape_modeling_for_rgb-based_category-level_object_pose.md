---
title: >-
  [论文解读] LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation
description: >-
  [ECCV 2024][人体理解][类别级物体位姿估计] 提出 LaPose 框架，通过拉普拉斯混合模型 (LMM) 建模物体形状不确定性，结合 DINOv2 通用3D流和卷积专用特征流的双流架构预测 NOCS 坐标分布，并引入尺度无关的位姿表示解决 RGB-only 场景下的固有尺度歧义，在 NOCS 数据集上取得 SOTA。
tags:
  - "ECCV 2024"
  - "人体理解"
  - "类别级物体位姿估计"
  - "RGB-only"
  - "拉普拉斯混合模型"
  - "尺度不变表示"
  - "PnP"
---

# LaPose: Laplacian Mixture Shape Modeling for RGB-Based Category-Level Object Pose Estimation

**会议**: ECCV 2024  
**arXiv**: [2409.15727](https://arxiv.org/abs/2409.15727)  
**代码**: [github.com/lolrudy/LaPose](https://github.com/lolrudy/LaPose)  
**领域**: 人体理解  
**关键词**: 类别级物体位姿估计, RGB-only, 拉普拉斯混合模型, 尺度不变表示, PnP

## 一句话总结

提出 LaPose 框架，通过拉普拉斯混合模型 (LMM) 建模物体形状不确定性，结合 DINOv2 通用3D流和卷积专用特征流的双流架构预测 NOCS 坐标分布，并引入尺度无关的位姿表示解决 RGB-only 场景下的固有尺度歧义，在 NOCS 数据集上取得 SOTA。

## 研究背景与动机

类别级物体位姿估计需预测 9DoF 位姿（旋转、平移、尺寸），RGBD 方法虽然效果好但受限于深度传感器的可用性。RGB-only 方法面临两个核心挑战：

**形状不确定性**：缺少深度信息使得预测物体3D形状更加困难，尤其在类内形状变化大的区域（如相机镜头长度从正面无法确定），增加了 NOCS 坐标预测的不确定性

**尺度歧义**：仅凭 RGB 图像无法区分"大物体远离相机"和"小物体靠近相机"，使得绝对尺度预测成为病态问题

现有 RGB 方法（MSOS、OLD-Net、DMSR）的局限：将所有像素的 NOCS 预测等权对待、依赖 RANSAC 过滤异常值（慢且不鲁棒），对尺度歧义处理不足。

## 方法详解

### 整体框架

LaPose 的整体流程：
1. MaskRCNN 检测并裁剪目标物体
2. **双流特征提取**：DINOv2 通用3D信息流 + 卷积网络专用特征流
3. 两流分别独立预测像素级 Laplacian 分布的均值和方差
4. 合并为 **拉普拉斯混合模型 (LMM)** 建立2D-3D对应关系
5. 卷积 PnP 模块求解旋转和归一化平移
6. 独立 size head 预测归一化物体尺寸

### 关键设计

1. **拉普拉斯混合模型 (LMM)**：核心创新。将每个像素的 NOCS 坐标建模为概率分布而非确定性值，显式量化形状不确定性。选择 Laplacian 分布而非 Gaussian 的原因是其更强的异常值鲁棒性（L1 距离 vs L2 距离）。两流分别预测 $\text{Laplace}(\mu_{dino}, \sigma^2_{dino})$ 和 $\text{Laplace}(\mu_{conv}, \sigma^2_{conv})$，通过 Laplacian aleatoric uncertainty loss 同时学习均值和方差：
    $\mathcal{L}_{3D} = \frac{\lambda}{\sigma^2} \| \mathbf{M}_{vis} \cdot (\mathbf{C}^{3D}_{gt} - \mu) \|_1 + \mathbf{M}_{vis} \cdot \log(\sigma^2)$
   当坐标误差大时 $\sigma^2$ 被迫增大以减小第一项，误差小时 $\sigma^2$ 被鼓励减小以降低对数项——实现自监督的不确定性学习。PnP 模块据此动态聚合双流信息并过滤高方差区域的错误对应。

2. **双流特征架构**：

    - **通用3D信息流 (DINOv2)**：提取类别无关的、SE(3)-一致的局部特征，与 NOCS 坐标的 SE(3)-不变性天然契合。但 DINOv2 缺乏类别特定知识
    - **专用特征流 (ConvNet)**：训练卷积网络提取类别特定特征，补充 DINOv2 的不足
    - 消融显示：仅用 DINOv2 或仅用 ConvNet 性能均不如双流组合，双流互补带来 6.7% 的 $10°0.5d$ 提升

3. **尺度无关位姿表示 (SAP)**：解决 RGB-only 的固有尺度歧义。将物体归一化到紧致包围盒对角线长度为1：

    - 归一化尺寸：$\mathbf{s}_{norm} = \{s_x/d, s_y/d, s_z/d\}$，其中 $d = \sqrt{s_x^2 + s_y^2 + s_z^2}$
    - 归一化平移：$\mathbf{t}_{norm} = \{t_x/d, t_y/d, t_z/d\}$
    - 预测归一化 size 的残差 $\mathbf{s}_{out} = \mathbf{s}_{norm} - \mathbf{s}_{avg}$，降低预测难度
    - 绝对尺度 $d$ 由独立的 MobileNet 预测，解耦尺度与位姿，阻止尺度预测误差向位姿传播
    - 对应提出归一化 IoU (NIoU) 和 $10°0.2d$ 等尺度无关评估指标

### 损失函数 / 训练策略

$$\mathcal{L} = \lambda_{pose} \mathcal{L}_{pose} + \lambda_{3D}(\mathcal{L}_{3D\text{-}dino} + \mathcal{L}_{3D\text{-}conv})$$

- **$\mathcal{L}_{pose}$**：监督尺度无关的 9DoF 参数（旋转向量、归一化平移、归一化尺寸）
- **$\mathcal{L}_{3D}$**：Laplacian aleatoric uncertainty loss，驱动双流的 NOCS 预测和不确定性估计
- 超参数 $\{\lambda_1, \lambda_2, \lambda_{pose}, \lambda_{3D}\} = \{15, 15, 1, 0.1\}$
- Ranger 优化器，学习率 $10^{-3}$，余弦退火，batch size 32
- Dynamic Zoom-In 增强检测鲁棒性，100 epoch（单模型）或 150 epoch（每类单独模型）

## 实验关键数据

### 主实验

**NOCS-REAL275 (尺度无关指标 mAP %)**:

| 方法 | NIoU25 | NIoU50 | NIoU75 | 10°0.2d | 10°0.5d | 10° |
|------|--------|--------|--------|---------|---------|-----|
| MSOS | 36.9 | 9.7 | 0.7 | 3.3 | 15.3 | 17.0 |
| OLD-Net | 31.5 | 6.2 | 0.1 | 2.8 | 12.2 | 14.8 |
| DMSR | 57.2 | 38.4 | 9.7 | 26.0 | 44.9 | 36.9 |
| **LaPose** | **70.7** | **47.9** | 15.8 | 37.4 | **57.4** | **60.7** |
| **LaPose (M)** | 66.4 | **48.8** | **20.5** | **39.7** | 55.4 | 60.2 |

**NOCS-REAL275 (绝对尺度指标 mAP %)**:

| 方法 | IoU25 | IoU50 | IoU75 | 10°10cm |
|------|-------|-------|-------|---------|
| DMSR | 37.4 | 16.3 | 3.2 | 25.2 |
| **LaPose (M)** | **40.2** | **18.3** | **4.1** | 27.7 |
| **LaPose** | 41.2 | 17.5 | 2.6 | **30.5** |

### 消融实验

**NOCS-REAL275 组件消融**:

| 配置 | $\mathcal{F}_{conv}$ | $\mathcal{F}_{dino}$ | $\sigma^2$ | SAP | NIoU25 | 10°0.5d |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| (A) Conv only | ✓ | | | ✓ | 60.3 | 46.1 |
| (B) 无SAP | ✓ | | | | 37.8 | 33.2 |
| (C) DINO only | | ✓ | | ✓ | 61.7 | 38.5 |
| (D) 双流无分布 | ✓ | ✓ | | ✓ | 64.9 | 52.8 |
| (E) Conv+Lap | ✓ | | Lap | ✓ | 65.5 | 51.2 |
| (F) Conv+Gaus | ✓ | | Gaus | ✓ | 59.1 | 44.7 |
| **(G) Full LaPose** | **✓** | **✓** | **Lap** | **✓** | **70.7** | **57.4** |

### 关键发现

- **SAP 至关重要**：无 SAP 时 NIoU25 从 60.3% 暴跌至 37.8%，证实尺度歧义对训练的严重干扰
- **Laplacian 优于 Gaussian**：Gaussian 分布反而降低性能 (59.1% vs 60.3%)，L1 损失的异常值鲁棒性在此任务中关键
- **双流互补**：DINOv2 和 ConvNet 各有所长，组合后 NIoU25 提升约5%
- **LMM 建模**：方差图与 NOCS 误差高度相关，能有效引导 PnP 模块聚焦低不确定性区域
- 在旋转预测上优势尤为明显（$10°$ 指标达 60.7% vs DMSR 的 36.9%），提升 23.8%

## 亮点与洞察

- **概率建模形状**：将确定性 NOCS 预测升级为概率分布，自然解决了"有些像素更可信"的问题，思路优雅
- **DINOv2 的3D能力挖掘**：利用 DINOv2 的 SE(3)-一致特征辅助 NOCS 预测，为基础模型在3D任务中的应用提供了新思路
- **尺度解耦设计**：SAP 不仅提升训练效率，还提供了更有意义的评估指标（NIoU），推动了领域指标标准化
- 推理速度约 10 FPS，具备一定实时性

## 局限与展望

- 绝对尺度 $d$ 由独立的 MobileNet 预测，与位姿估计完全解耦可能丢失有用的联合信息
- 依赖 MaskRCNN 检测结果，检测失败直接导致后续全部失败
- DINOv2 骨干冻结使用，微调可能进一步提升性能但增加训练成本
- 目前仅在 NOCS 的6个类别上验证，未扩展到更多类别或开放词汇设定
- 多模型版本 (M) 每类训练独立模型，扩展性受限

## 相关工作与启发

- DMSR 利用预训练 DPT 模型的法线和相对深度作为额外输入，启发了"引入外部3D先验"的思路
- EPro-PnP 的端到端概率 PnP 框架影响了 PnP 模块的设计
- DINOv2 在3D任务中的有效性（SecondPose、SD-3D 等已验证）为本文提供了动机
- Laplacian aleatoric uncertainty loss 源自 MonoPair 等单目3D检测工作

## 评分

- **新颖性**: ⭐⭐⭐⭐ — LMM 建模形状不确定性 + 双流架构 + SAP 三个创新点组合紧密
- **实验充分度**: ⭐⭐⭐⭐ — 消融全面，发现了评估脚本 bug 并修正，但数据集仅限 NOCS
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题动机到方案设计逻辑严谨，图示清晰专业
- **价值**: ⭐⭐⭐⭐ — RGB-only 类别级位姿SOTA，推动了尺度无关评估指标标准化

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] GS-Pose: Category-Level Object Pose Estimation via Geometric and Semantic Correspondence](gs-pose_category-level_object_pose_estimation_via_geometric_and_semantic_corresp.md)
- [\[ECCV 2024\] U-COPE: Taking a Further Step to Universal 9D Category-Level Object Pose Estimation](u-cope_taking_a_further_step_to_universal_9d_category-level_object_pose_estimati.md)
- [\[CVPR 2025\] GCE-Pose: Global Context Enhancement for Category-Level Object Pose Estimation](../../CVPR2025/human_understanding/gce-pose_global_context_enhancement_for_category-level_object_pose_estimation.md)
- [\[ECCV 2024\] FoundPose: Unseen Object Pose Estimation with Foundation Features](foundpose_unseen_object_pose_estimation_with_foundation_features.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)

</div>

<!-- RELATED:END -->
