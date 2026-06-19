---
title: >-
  [论文解读] DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis
description: >-
  [ICCV 2025][图像生成][Deepfake Detection] 提出 DeepShield，一种结合局部 patch 级引导（LPG）和全局伪造多样化（GFD）的深度伪造视频检测框架，通过时空伪影建模提供 patch 级监督、分布级特征增强合成多样伪造表征，在跨数据集和跨操控类型评估中显著超越 SOTA。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Deepfake Detection"
  - "CLIP-ViT"
  - "Patch-Level Supervision"
  - "Feature Augmentation"
  - "Cross-Domain Generalization"
---

# DeepShield: Fortifying Deepfake Video Detection with Local and Global Forgery Analysis

**会议**: ICCV 2025  
**arXiv**: [2510.25237](https://arxiv.org/abs/2510.25237)  
**代码**: [GitHub](https://github.com/lijichang/DeepShield)  
**领域**: 深度伪造检测 / 图像生成  
**关键词**: Deepfake Detection, CLIP-ViT, Patch-Level Supervision, Feature Augmentation, Cross-Domain Generalization

## 一句话总结

提出 DeepShield，一种结合局部 patch 级引导（LPG）和全局伪造多样化（GFD）的深度伪造视频检测框架，通过时空伪影建模提供 patch 级监督、分布级特征增强合成多样伪造表征，在跨数据集和跨操控类型评估中显著超越 SOTA。

## 研究背景与动机

深度生成模型（GAN、VAE、扩散模型）使人脸视频篡改门槛大幅降低，伪造检测成为关键问题。现有检测器面临两大挑战：

**局部敏感性不足**：基于 CLIP 等大模型的方法主要利用全局特征，倾向于关注最显著的篡改痕迹（如面部转换的夸张过渡），忽略了细微的伪造线索（如混合边界、微小纹理不一致）

**跨域泛化差**：模型容易过拟合训练时见过的特定篡改类型，面对未知篡改手段性能大幅下降。依赖重新训练或数据增强的方案成本高、扩展性差

本文核心思想：通过局部-全局学习范式，让模型既能捕捉细粒度的 patch 级伪造痕迹，又能通过特征空间的伪造多样化增强跨域泛化能力。

## 方法详解

### 整体框架

DeepShield 基于 CLIP-ViT-B/16 + ST-Adapter，包含两个互补组件：
- **Local Patch Guidance (LPG)**：patch 级监督学习 + 时空伪影建模
- **Global Forgery Diversification (GFD)**：域特征增强 + 对比学习目标

### 关键设计

1. **时空伪影建模（SAM, Spatiotemporal Artifact Modeling）**：

    - 将 SBI 技术从图像级扩展到视频级，生成带有精确 mask 的伪造视频
    - **空间伪影**：对每帧图像分别增强内层（面部）和外层（背景），通过面部关键点凸包 mask 混合，产生统计不一致性
    - **时间伪影**：跨 $T$ 帧保持一致的增强方向和 mask 变换策略，模拟真实 deepfake 的生成模式
    - 关键：同一增强类型跨帧一致，mask 的随机变形和模糊跨帧保持微小差异

2. **Local Patch Guidance (LPG)**：

    - 将每帧和对应 mask 划分为 $P$ 个非重叠 patch，通过 PatchMaskScore 和阈值 $\theta$ 分配 patch 级二分类标签
    - 每个 patch token 独立作为训练样本，通过二分类器 $\phi$ 和交叉熵损失 $\mathcal{L}_{\text{LPG}}$ 训练
    - 效果：patch 嵌入同类聚集、异类分离，cls token 通过自注意力聚合更丰富的局部语义

3. **Global Forgery Diversification (GFD)**：

    - **Domain-Bridging Feature Generation (DFG)**：随机配对不同篡改类型的视频，通过 Beta(0.1, 0.1) 采样权重 $\lambda$，用 AdaIN 混合两个域的均值/方差统计量，生成跨域桥接特征
    - **Boundary-Expanding Feature Generation (BFG)**：将标准差缩放 $\alpha=1.1$ 倍，将特征推向现有域边界外，扩展检测范围
    - 训练目标：交叉熵损失 + 监督对比损失 $\mathcal{L}_{\text{GFD}} = \mathcal{L}^{\text{cls}} + \upsilon \mathcal{L}^{\text{supCon}}$

### 损失函数 / 训练策略

$$\mathcal{L}^{\text{overall}} = \omega \mathcal{L}_{\text{LPG}} + \mathcal{L}_{\text{GFD}}$$

- $\omega = 0.5$ 平衡局部与全局损失
- $\upsilon = 0.5$ 平衡交叉熵与对比损失
- Adam 优化器，余弦学习率衰减，80 epochs
- 输入：每视频采样 4 个 clip，每 clip 12 帧连续帧

## 实验关键数据

### 主实验（跨数据集评估，FF++ HQ 训练，Video-level AUC %）

| 方法 | 架构 | FF++ | CDF | DFDCP | DFDC | DFD |
|------|------|------|-----|-------|------|-----|
| SBI | ResNet50 | - | 85.7 | - | - | 94.0 |
| AltFreezing | 3D | 99.7 | 89.0 | - | - | 93.7 |
| TALL | Swin-B | 99.9 | 90.8 | - | 76.8 | - |
| LSDA | EfficientNet | - | 91.1 | 81.2 | 77.0 | 95.6 |
| SeeABLE | EfficientNet | - | 87.3 | 86.3 | 75.9 | - |
| VB-StA | CLIP ViT-B/16 | - | 86.6 | - | 77.8 | - |
| **DeepShield** | **CLIP ViT-B/16+ST-Adapter** | **99.2** | **92.2** | **93.2** | **82.8** | **96.1** |

在最具挑战性的 DFDCP 和 DFDC 上分别超越此前最优 **6.9%** 和 **5.8%**。

### 消融实验

**LPG 和 GFD 组件贡献（跨数据集 AUC %）**：

| 变体 | CDF | DFDC | DFDCP | Avg |
|------|-----|------|-------|-----|
| DeepShield (full) | **92.2** | **82.8** | **93.2** | **89.4** |
| w/o LPG | 89.1 | 81.3 | 87.1 | 85.8 |
| w/o GFD | 89.0 | 81.9 | 91.9 | 87.6 |
| w/o LPG & GFD | 85.4 | 78.4 | 88.9 | 84.2 |

**DFA 组件贡献**：

| DFG | BFG | CDF | DFDC | DFDCP | Avg |
|:---:|:---:|-----|------|-------|-----|
| ✔ | ✔ | **92.2** | **82.8** | **93.2** | **89.4** |
| ✗ | ✔ | 91.1 | 81.9 | 92.9 | 88.6 |
| ✔ | ✗ | 90.3 | 82.1 | 92.5 | 88.3 |
| ✗ | ✗ | 92.0 | 81.3 | 91.6 | 88.3 |

### 关键发现

- LPG 移除导致 Avg AUC 下降 3.6%，特别在 DFDCP 上下降 6.1%——局部感知对复杂伪造至关重要
- GFD 移除导致下降 1.8%，主要影响跨域泛化
- 两者同时移除下降 5.2%，证明局部与全局分析的协同效应
- DFG 和 BFG 单独均有正贡献且互补——DFG 桥接已知域，BFG 扩展边界
- 跨操控评估中，以 FS 训练时平均 AUC 超越 WATCHER **15.95%**
- GradCAM 可视化显示 LPG 使模型均匀关注整个篡改区域，而非仅关注最显著伪影
- t-SNE 可视化显示完整 DeepShield 的真/假特征分离度远优于退化版

## 亮点与洞察

- **局部-全局协同范式**：LPG 提供细粒度局部感知，GFD 提供多样化全局表征，两者深度耦合形成迭代协作
- **SAM 的精巧设计**：将 SBI 扩展到视频级不是简单的复制——需要保证跨帧增强一致性和 mask 变化的自然性
- **分布级特征增强优于线性插值**：DFG 的 AdaIN 混合和 BFG 的标准差缩放比简单线性插值产生更丰富的非线性增强
- 在 CLIP-ViT 微调基础上的设计，保持了预训练模型的泛化优势

## 局限与展望

- FF++ 内部测试（in-domain）性能略低于部分方法，说明局部-全局策略在过拟合场景下有一定代价
- BFG 的缩放系数 $\alpha=1.1$ 是经验值，不同数据集可能需要调整
- 未探讨对最新扩散模型生成的伪造内容的检测效果
- patch 级标签依赖 SAM 生成的 mask 质量，mask 不精确可能引入噪声标签

## 相关工作与启发

- **SBI**：空间伪影合成的基础方法，本文将其扩展到时空维度
- **LSDA**：特征空间线性增强的先驱，本文用分布级增强取代线性插值
- **AltFreezing**：通过交替训练空间/时间卷积层捕获时空信息，但从零训练泛化性受限
- **ST-Adapter**：参数高效微调 CLIP-ViT 的关键组件

## 评分

- **新颖性**: ⭐⭐⭐⭐ 局部 patch 引导+全局伪造多样化的组合有新意，SAM 的视频级扩展设计精细
- **实验充分度**: ⭐⭐⭐⭐⭐ 跨数据集+跨操控双重评估，含详细消融、GradCAM、t-SNE 可视化
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，但公式符号较多需要仔细对应
- **价值**: ⭐⭐⭐⭐⭐ 跨域检测性能提升显著（DFDC +5.8%），对实际部署有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection](forgelens_data-efficient_forgery_focus_for_generalizable_forgery_image_detection.md)
- [\[ICCV 2025\] Semantic Discrepancy-aware Detector for Image Forgery Identification](semantic_discrepancy-aware_detector_for_image_forgery_identification.md)
- [\[NeurIPS 2025\] FerretNet: Efficient Synthetic Image Detection via Local Pixel Dependencies](../../NeurIPS2025/image_generation/ferretnet_efficient_synthetic_image_detection_via_local_pixel_dependencies.md)
- [\[CVPR 2026\] LogCD: Local-to-global Consistency Distillation for Few-step Image Generation](../../CVPR2026/image_generation/logcd_local-to-global_consistency_distillation_for_few-step_image_generation.md)
- [\[ICML 2026\] From Talking to Singing: A New Challenge for Audio-Visual Deepfake Detection](../../ICML2026/image_generation/from_talking_to_singing_a_new_challenge_for_audio-visual_deepfake_detection.md)

</div>

<!-- RELATED:END -->
