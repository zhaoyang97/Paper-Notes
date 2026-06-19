---
title: >-
  [论文解读] Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures
description: >-
  [ECCV 2024][语义分割][管状结构分割] 利用分形理论将分形维数（FD）从图像级扩展到像素级，生成分形特征图（FFM）作为深度学习模型的额外输入和损失权重，并设计包含边缘解码器和骨架解码器的多解码器网络（MD-Net），在五个管状结构数据集上显著提升分割性能。 管状结构的精确分割在多个领域至关重要： - 生物学：内…
tags:
  - "ECCV 2024"
  - "语义分割"
  - "管状结构分割"
  - "分形特征图"
  - "拓扑自相似性"
  - "多解码器网络"
  - "边界与骨架"
---

# Representing Topological Self-Similarity Using Fractal Feature Maps for Accurate Segmentation of Tubular Structures

**会议**: ECCV 2024  
**arXiv**: [2407.14754](https://arxiv.org/abs/2407.14754)  
**代码**: 有 ([https://github.com/cbmi-group/FFM-Multi-Decoder-Network](https://github.com/cbmi-group/FFM-Multi-Decoder-Network))  
**领域**: 分割  
**关键词**: 管状结构分割, 分形特征图, 拓扑自相似性, 多解码器网络, 边界与骨架

## 一句话总结

利用分形理论将分形维数（FD）从图像级扩展到像素级，生成分形特征图（FFM）作为深度学习模型的额外输入和损失权重，并设计包含边缘解码器和骨架解码器的多解码器网络（MD-Net），在五个管状结构数据集上显著提升分割性能。

## 研究背景与动机

管状结构的精确分割在多个领域至关重要：

- **生物学**：内质网（ER）分割用于人类疾病机制研究
- **医学**：血管分割用于视网膜病变和中风的早期诊断
- **遥感**：道路提取用于导航和路线规划

管状结构分割面临的核心挑战：
1. 复杂的形态和几何形状
2. 低信噪比和低对比度

**互连结构的断裂问题**——分割结果中管状结构频繁出现中断

现有深度学习方法主要从三个方面改进：卷积核设计（如可变形卷积）、模型架构设计、损失函数设计。但这些方法主要关注网络本身的优化，未给模型提供**额外的结构先验信息**。

本文的核心观察：管状结构具有**拓扑自相似性**——大型复杂管状结构在不同尺度上展现相似的拓扑模式（"一个节点连接多条边"这一基本组件在不同尺度重复出现）。这一特性可用分形理论量化。

## 方法详解

### 整体框架

本文贡献包含两个独立但互补的模块：

1. **分形特征图（FFM）**：将分形维数扩展到像素级，作为模型输入和损失权重
2. **多解码器网络（MD-Net）**：在 U-Net 基础上增加边缘解码器和骨架解码器

### 关键设计

**分形维数的像素级扩展**：

传统分形维数针对整幅图像计算一个标量值。本文通过**滑动窗口**技术将其扩展到每个像素：

1. 使用 5×5 窗口在图像上滑动（步长为1）
2. 每个窗口内用盒计数法（Box-counting）计算局部 FD
3. 所有像素位置的 FD 值组成 FFM

**盒计数法**：将灰度图像建模为3D空间 $(x,y,z)$，z 为灰度值。以不同尺度 $k$ 将空间划分为 $k \times k \times h$ 的立方体，统计覆盖整个灰度面所需的最少盒数 $N_r$。FD 通过 $\log N_r$ vs. $\log(1/r)$ 的最小二乘线性拟合得到。

**FFM 的两种用法**：

| FFM 类型 | 来源 | 用途 |
|----------|------|------|
| $FFM_{image}$ | 原始图像 | 作为模型的额外输入通道 |
| $FFM_{label}$ | 标注掩码 | 作为损失函数的像素级权重 |

$FFM_{image}$ 帮助模型感知纹理复杂度和自相似结构，$FFM_{label}$ 使模型对更复杂区域（FD 更高）分配更大的损失权重。

**多解码器网络（MD-Net）**：

在 U-Net 的编码器-解码器基础上，增加两个平行解码器：

- **目标解码器**：预测分割掩码
- **边缘解码器**：预测管状结构的边界
- **骨架解码器**：预测管状结构的骨架

三个解码器共享编码器的特征，通过跳跃连接获取多尺度信息。推理时仅使用目标解码器的输出。

### 损失函数 / 训练策略

**全局损失**：

$$\mathcal{L}_{global} = \alpha\mathcal{L}_{object} + \beta\mathcal{L}_{edge} + \gamma\mathcal{L}_{skeleton}$$

- 目标分割使用 Soft IoU Loss
- 边缘和骨架使用 BCE Loss
- 权重默认 $\alpha=1.0, \beta=0.5, \gamma=0.5$

**分形约束损失**：用 $FFM_{label}$ 作为目标分割损失的像素级权重：

$$\mathcal{L}_{constrained} = \alpha\mathcal{L}_{object} \cdot FFM_{label} + \beta\mathcal{L}_{edge} + \gamma\mathcal{L}_{skeleton}$$

训练细节：SGD 优化器，初始学习率 0.05，对所有数据集使用固定 batch size 32，训练 50 个 epoch。

## 实验关键数据

### 主实验（表格）

ER（内质网）和 MITO（线粒体）数据集上的分割性能：

| 模型 | 损失 | ER IoU↑ | ER clDice↑ | ER β Error↓ | MITO IoU↑ | MITO clDice↑ |
|------|------|---------|------------|-------------|-----------|--------------|
| U-Net | $\mathcal{L}_{iou}$ | 75.44 | 94.63 | 28.72 | 79.77 | 96.91 |
| U-Net++ | $\mathcal{L}_{iou}$ | 75.02 | 94.67 | 26.02 | 79.70 | 97.30 |
| DSC-Net | $\mathcal{L}_{iou}$ | 75.51 | 94.44 | 34.51 | 80.32 | 97.16 |
| U-Net* (+ FFM) | $\mathcal{L}_{iou}$ | 76.59 | 95.43 | 20.78 | 80.71 | 97.42 |
| HR-Net* (+ FFM) | $\mathcal{L}_{iou}$ | 76.43 | 95.47 | 20.52 | 80.62 | 97.29 |
| **MD-Net*** | $\mathcal{L}_{constrained}$ | **77.09** | 95.74 | 19.52 | **81.18** | 97.61 |

### 消融实验（表格）

FFM 作为插件模块的效果（IoU 提升）：

| 基础模型 | 原始 IoU (ER) | + FFM IoU (ER) | 提升 |
|----------|-------------|----------------|------|
| U-Net | 75.44 | 76.59 | +1.15 |
| HR-Net | 75.83 | 76.43 | +0.60 |
| MD-Net | 77.01 | 77.09 | +0.08 |

FFM + 约束损失 vs. 仅 FFM 输入：MD-Net 使用约束损失后 ER 的 HD（Hausdorff Distance）从 6.77 降至 6.72，ACC 从 92.06 升至 92.14。

### 关键发现

1. **FFM 作为通用插件有效**：无论是 U-Net 还是 HR-Net，加入 $FFM_{image}$ 均提升性能，验证了其通用性
2. **多解码器设计比单解码器更优**：MD-Net 在所有指标上超越基线 U-Net
3. **分形约束损失提供额外增益**：使用 $FFM_{label}$ 加权的损失进一步提升了边界精度（HD 下降）
4. **拓扑错误显著减少**：在 ER 数据集上，β Error 从 U-Net 的 28.72 降至 MD-Net* 的 19.52（降低 32%）
5. FFM 在 ROSE、STARE 等视网膜血管数据集和 ROAD 遥感道路数据集上同样有效

## 亮点与洞察

1. **分形理论与深度学习的创新融合**：首次将像素级分形特征引入深度学习分割模型，为 "结构先验注入" 提供了新思路
2. **即插即用的设计**：FFM 不依赖特定网络架构，可作为任意分割模型的插件模块
3. **双重利用 FFM**：既作为输入通道增强特征表示，又作为损失权重引导优化方向
4. **边缘+骨架辅助解码器**：虽然推理时不使用，但训练时的多任务学习有效提升了主解码器的分割质量
5. 使用标准差替代灰度值计算 FD，增强了对图像噪声的鲁棒性

## 局限与展望

1. FFM 的滑动窗口计算有一定开销（但为离线预计算，不影响推理速度）
2. 滑动窗口大小（5×5）和盒计数法的尺度参数为手工设定，可探索自适应方案
3. FFM 在非管状结构（如 NUCLEUS 数据集的椭圆形细胞核）上提升有限，说明其优势主要体现在拓扑自相似的结构上
4. 仅在2D图像上验证，可扩展到3D管状结构分割（如3D血管）
5. 骨架和边缘的 ground truth 由算法自动提取，可能引入噪声

## 相关工作与启发

- **clDice**：基于骨架交集的分割指标和损失函数，与本文的骨架解码器互补
- **DSC-Net**：使用动态蛇形卷积捕捉管状特征，关注微观卷积核设计
- **Dconn-Net**：关注连通性的分割网络
- 启发：为分割模型**提供结构先验信息**（如分形特征、拓扑约束）是提升复杂结构分割的有效策略

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 新颖性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 5 |
| 写作质量 | 4 |
| 实用价值 | 4 |
| **综合** | **4.2** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Self-supervised Co-salient Object Detection via Feature Correspondences at Multiple Scales](self-supervised_co-salient_object_detection_via_feature_correspondences_at_multi.md)
- [\[ECCV 2024\] Eliminating Feature Ambiguity for Few-Shot Segmentation](eliminating_feature_ambiguity_for_few-shot_segmentation.md)
- [\[ECCV 2024\] FREST: Feature Restoration for Semantic Segmentation under Multiple Adverse Conditions](frest_feature_restoration_for_semantic_segmentation_under_multiple_adverse_condi.md)
- [\[ECCV 2024\] LiFT: A Surprisingly Simple Lightweight Feature Transform for Dense ViT Descriptors](lift_a_surprisingly_simple_lightweight_feature_transform_for_dense_vit_descripto.md)
- [\[ICCV 2025\] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation](../../ICCV2025/segmentation/topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)

</div>

<!-- RELATED:END -->
