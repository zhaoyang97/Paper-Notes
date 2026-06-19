---
title: >-
  [论文解读] RAPTR: Radar-Based 3D Pose Estimation Using Transformer
description: >-
  [NeurIPS 2025][人体理解][雷达感知] 提出RAPTR，首个利用弱监督（3D BBox + 2D关键点标签）进行雷达3D人体姿态估计的Transformer框架，通过伪3D可变形注意力和结构化损失函数在两个室内数据集上大幅超过基线。 雷达在室内人体感知中有独特优势：隐私保护、穿墙能力、对光照/烟雾鲁棒…
tags:
  - "NeurIPS 2025"
  - "人体理解"
  - "雷达感知"
  - "3D人体姿态估计"
  - "Transformer"
  - "弱监督"
  - "可变形注意力"
---

# RAPTR: Radar-Based 3D Pose Estimation Using Transformer

**会议**: NeurIPS 2025  
**arXiv**: [2511.08387](https://arxiv.org/abs/2511.08387)  
**代码**: [GitHub](https://github.com/merlresearch/radar-pose-transformer)  
**领域**: 人体理解  
**关键词**: 雷达感知, 3D人体姿态估计, Transformer, 弱监督, 可变形注意力

## 一句话总结

提出RAPTR，首个利用弱监督（3D BBox + 2D关键点标签）进行雷达3D人体姿态估计的Transformer框架，通过伪3D可变形注意力和结构化损失函数在两个室内数据集上大幅超过基线。

## 研究背景与动机

雷达在室内人体感知中有独特优势：隐私保护、穿墙能力、对光照/烟雾鲁棒。但现有方法依赖昂贵的细粒度3D关键点标签（通常需VICON等动捕系统采集），在复杂室内环境（杂物、遮挡、多人）中标注成本极高且难以扩展。

相比之下，2D关键点标签（从相机图像获取）和粗粒度3D BBox标签（从深度传感器或雷达获取）采集成本低得多。本文的核心动机是：能否仅用这些廉价的弱监督标签训练高质量的雷达3D姿态模型？关键挑战是深度模糊——2D标签无法提供深度信息，3D BBox仅提供位置但无关节级精度。

## 方法详解

### 整体框架

RAPTR接收多视角雷达热图（水平-深度 $\mathbf{Y}_{\text{hor}} \in \mathbb{R}^{T \times W \times D}$ 和垂直-深度 $\mathbf{Y}_{\text{ver}} \in \mathbb{R}^{T \times H \times D}$），通过共享骨干网络提取多尺度特征，经交叉视角编码器融合双视角信息，再由两阶段解码器（姿态解码器 + 关节解码器）逐步估计3D人体姿态。

### 关键设计

1. **伪3D可变形注意力**: 核心创新。参考点和采样偏移量定义在3D雷达空间 $(x,y,z)$ 中，通过投影到两个2D雷达视图提取特征：$\mathbf{f}_{\text{hor}}^{(i)} = \mathbf{F}_{\text{hor}}(x+\Delta x_i, z+\Delta z_i)$ 和 $\mathbf{f}_{\text{ver}}^{(i)} = \mathbf{F}_{\text{ver}}(y+\Delta y_i, z+\Delta z_i)$。相比QRFPose的逐视图独立2D注意力，本方法在3D空间统一处理偏移量，避免了冗余的逐视图偏移估计，随视图数增加更好扩展。多视图注意力权重通过query的线性投影+softmax获得。

2. **两阶段解码器架构**: 借鉴RGB姿态估计PETR的设计。**姿态解码器**处理 $N$ 个姿态query，每层迭代更新参考姿态 $\tilde{\mathbf{P}}_{\text{radar}}^{(l)} = \sigma(\sigma^{-1}(\tilde{\mathbf{P}}_{\text{radar}}^{(l-1)}) + \Delta \tilde{\mathbf{P}}_{\text{radar}}^{(l-1)})$，输出初始3D姿态和置信度。**关节解码器**对每个匹配到的姿态进一步逐关节精调，输入为姿态解码器的预测结果。

3. **结构化损失函数（核心）**: 精心设计以利用弱监督标签。

    - **3D模板损失（T3D）**: 在姿态解码器处使用。从3D BBox标签计算重心 $\mathbf{g}_{\text{world}}$，结合预定义关键点模板 $\mathbf{K}_{\text{world}}$ 生成模板姿态 $\mathbf{T}_{\text{world}} = \mathbf{K}_{\text{world}} + \mathbf{1}^\top \mathbf{g}_{\text{world}}$，约束初始姿态与模板对齐，缓解深度模糊
    - **3D重力损失（G3D）**: 在关节解码器处，约束精调姿态的重心与BBox重心一致
    - **2D关键点损失（K2D）**: 将精调的3D姿态投影到图像平面，与2D关键点标签计算欧式距离和OKS损失
    - 总损失: $\mathcal{L} = \frac{1}{N'}\sum(\lambda_1 \mathcal{L}_{\text{template}} + \lambda_2 \mathcal{L}_{\text{gravity}} + \lambda_3 \mathcal{L}_{\text{kpt2D}} + \lambda_4 \mathcal{L}_{\text{OKS}}) + \lambda_5 \mathcal{L}_{\text{cls}}$

### 损失函数 / 训练策略

使用DETR风格的二部图匹配关联预测和GT。骨干网络使用ResNet。输入 $T=4$ 帧时序上下文。姿态query数 $N=10$。使用focal loss作为分类损失。

## 实验关键数据

### 主实验

| 数据集 | 方法 | 整体MPJPE(cm) | 水平误差 | 垂直误差 | 深度误差 |
|--------|------|---------------|---------|---------|---------|
| HIBER-WALK | Person-in-WiFi 3D | 58.25 | 25.60 | 23.94 | 36.20 |
| HIBER-WALK | QRFPose | 38.20 | 14.78 | 13.40 | 26.76 |
| HIBER-WALK | HRRadarPose | 33.96 | 15.14 | 13.13 | 19.85 |
| HIBER-WALK | **RAPTR** | **22.32** | **8.41** | **4.85** | **17.73** |
| HIBER-MULTI | HRRadarPose | 33.19 | 16.77 | 10.75 | 21.84 |
| HIBER-MULTI | **RAPTR** | **18.99** | **7.80** | **4.38** | **14.54** |

### 消融实验

| 配置 | 指标 | 说明 |
|------|------|------|
| 无T3D模板损失 | MPJPE上升 | 深度方向误差显著增大 |
| 无G3D重力损失 | MPJPE上升 | 丢失3D位置约束 |
| 无K2D二维损失 | MPJPE显著上升 | 丢失关键的细粒度关节位置信息 |
| 2D注意力(QRFPose式) vs 伪3D注意力 | 伪3D更优 | 统一3D偏移估计更高效 |
| 单阶段 vs 两阶段解码器 | 两阶段更优 | 粗到精的渐进式估计更有效 |

### 关键发现

- RAPTR在WALK上降低MPJPE 34.3%（vs HRRadarPose），在MULTI上降低42.7%
- 在MMVR数据集上降低76.9%的关节位置误差
- 垂直方向误差仅4.85cm（WALK），远低于对手的13.13cm，说明2D标签有效约束了垂直位置
- 多人场景（MULTI）中RAPTR保持一致性能，而基线明显退化

## 亮点与洞察

- 首次系统探索用廉价弱监督标签（2D关键点+3D BBox）训练雷达3D姿态模型，具有很高的实用价值
- 结构化损失设计精巧：T3D用模板提供初始3D骨架先验，G3D+K2D从不同维度约束，三者互补解决深度模糊
- 伪3D注意力在3D空间统一处理采样点，比逐视图独立注意力更优雅且可扩展
- 多人场景下的鲁棒性（18.99cm vs 33.19cm）表明DETR式query匹配适合雷达人体感知

## 局限与展望

- 3D模板假设固定的人体骨架比例，不适应儿童或特殊体型
- 仅在两个室内数据集上验证，室外/非受控环境未评估
- 弱监督下的深度估计仍是最大误差来源（17.73cm vs 4.85cm垂直），可考虑引入时序一致性约束
- 对严重遮挡场景（如大型家具遮挡）的鲁棒性仍需评估
- 计算开销分析（相比简单CNN基线）可进一步讨论

## 相关工作与启发

- 继承PETR的两阶段query-based姿态估计思想，适配到雷达模态
- 弱监督3D姿态估计在RGB领域有丰富研究（lifting 2D到3D），本文首次在雷达上实现
- 多视角雷达特征融合可启发其他多模态传感器融合方案
- 对隐私保护的室内智能（老人监护、智慧建筑）有直接应用前景
- 与QRFPose的核心区别：伪3D注意力在3D空间统一偏移估计 vs 逐视图独立2D注意力
- 模板损失的设计思路可推广：任何有粗粒度3D标注的场景都可用人体骨架先验作为正则化
- 交叉视角编码器的双向交叉注意力比简单拼接更好地融合互补视角信息

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次在雷达3D姿态估计中系统利用弱监督，损失设计新颖
- 实验充分度: ⭐⭐⭐⭐ 两个数据集、多个基线、详细消融和定性分析
- 写作质量: ⭐⭐⭐⭐ 图示清晰，架构描述详尽
- 价值: ⭐⭐⭐⭐ 降低雷达姿态估计的标注成本，贴近实用需求

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] HiPART: Hierarchical Pose AutoRegressive Transformer for Occluded 3D Human Pose Estimation](../../CVPR2025/human_understanding/hipart_hierarchical_pose_autoregressive_transformer_for_occluded_3d_human_pose_e.md)
- [\[NeurIPS 2025\] PandaPose: 3D Human Pose Lifting from a Single Image via Propagating 2D Pose Prior to 3D Anchor Space](pandapose_3d_human_pose_lifting_from_a_single_image_via_propagating_2d_pose_prio.md)
- [\[CVPR 2025\] 3D Face Reconstruction From Radar Images](../../CVPR2025/human_understanding/3d_face_reconstruction_from_radar_images.md)
- [\[CVPR 2025\] GA3CE: Unconstrained 3D Gaze Estimation with Gaze-Aware 3D Context Encoding](../../CVPR2025/human_understanding/ga3ce_unconstrained_3d_gaze_estimation_with_gaze-aware_3d_context_encoding.md)
- [\[CVPR 2025\] Analyzing the Synthetic-to-Real Domain Gap in 3D Hand Pose Estimation](../../CVPR2025/human_understanding/analyzing_the_synthetic-to-real_domain_gap_in_3d_hand_pose_estimation.md)

</div>

<!-- RELATED:END -->
