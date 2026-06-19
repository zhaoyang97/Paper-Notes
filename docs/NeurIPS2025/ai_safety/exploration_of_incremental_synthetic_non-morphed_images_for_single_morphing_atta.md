---
title: >-
  [论文解读] Exploration of Incremental Synthetic Non-Morphed Images for Single Morphing Attack Detection
description: >-
  [NeurIPS 2025 (LXAI Workshop)][AI安全][S-MAD] 系统研究了在单图像变形攻击检测（S-MAD）训练中增量引入合成非变形人脸图像的效果，发现适量的合成数据（~75%增量）可提升跨数据集泛化能力（EER从6.17%降至6.10%），但过度使用或仅用合成数据会导致性能严重退化（EER升至~38%）。
tags:
  - "NeurIPS 2025 (LXAI Workshop)"
  - "AI安全"
  - "S-MAD"
  - "合成人脸数据"
  - "增量训练"
  - "EfficientNet"
  - "MobileNet"
  - "跨数据集泛化"
---

# Exploration of Incremental Synthetic Non-Morphed Images for Single Morphing Attack Detection

**会议**: NeurIPS 2025 (LXAI Workshop)  
**arXiv**: [2510.09836](https://arxiv.org/abs/2510.09836)  
**代码**: 无公开代码  
**领域**: AI安全  
**关键词**: S-MAD, 合成人脸数据, 增量训练, EfficientNet, MobileNet, 跨数据集泛化  

## 一句话总结
系统研究了在单图像变形攻击检测（S-MAD）训练中增量引入合成非变形人脸图像的效果，发现适量的合成数据（~75%增量）可提升跨数据集泛化能力（EER从6.17%降至6.10%），但过度使用或仅用合成数据会导致性能严重退化（EER升至~38%）。

## 背景与动机
人脸变形攻击（Morphing Attack）是AI时代生物识别系统面临的重要安全威胁：通过合并两个或多个主体的人脸图像生成一张变形图像，该图像可同时匹配多个主体的生物特征。变形攻击检测（MAD）分为两类：
- **D-MAD**（差分MAD）：对比现场照片和证件照
- **S-MAD**（单图像MAD）：仅基于单张图像判断是否为变形

S-MAD面临的核心数据困境：由于隐私法规限制，大规模真实人脸（bona fide）数据集获取困难，而深度学习方法对数据量需求大。合成数据是一种潜在的解决方案，但其引入方式和比例对检测性能的影响尚不明确。

## 核心问题
如何通过控制性地在训练集中引入合成非变形图像来增强S-MAD的泛化能力？合成数据的最优比例是多少？仅使用合成数据是否可行？

## 方法详解

### 数据集构成

三个数据集：

| 数据集 | 主体数 | Bona fide | Morph | 变形工具 | 备注 |
|--------|--------|-----------|-------|---------|------|
| FERET | 529 | 529×3 | 529×4×3 | FaceFusion, FaceMorpher, OpenCV, UBO | PS300/PS600/Resized |
| FRGCv2 | 533 | 984×3 | 964×4×3 | 同上 | PS300/PS600/Resized |
| SMDD | - | 15,000 | 25,000 | StyleGAN2-ADA | 完全合成 |

三种图像处理方法：打印/扫描300dpi（PS300）、打印/扫描600dpi（PS600）、数字缩放。四种变形工具覆盖从低质量（FaceMorpher）到高质量（FaceFusion）的攻击。

### 训练框架
- **预处理**：MTCNN人脸对齐（缩放因子0.9，输出369×369），数据增强（随机翻转/旋转/色彩抖动），ImageNet归一化
- **模型**：EfficientNet-B2（2.9M参数）和MobileNetV3-large（7.7M参数），ImageNet预训练权重初始化
- **优化器**：Adam（$\beta_1=0.99$, $\beta_2=0.999$），学习率$1\times10^{-5}$（网格搜索最优），batch size 64，100 epochs
- **损失函数**：分类交叉熵

$$L_i = H(y_k, \hat{y_k}) = -\frac{1}{n}\sum_x (y_k) \log(\hat{y_k})$$

### 增量合成数据策略
从SMDD的非变形子集中随机采样$S_{(SMDD,j)}$，按比例$j$加入训练集：

$$S_{(SMDD,j)} \subset D_{SMDD}, \quad |S_{(SMDD,j)}| = m < n$$

三种实验配置：
1. **不加合成数据**：FERET训练/FRGCv2测试（或反向）
2. **增量加入**：分别加入10%, 20%, 30%, 50%, 75%, 100%比例的SMDD非变形图像
3. **纯合成训练**：仅用SMDD训练，真实数据集测试

### 评估指标
- **MACER**（变形攻击分类错误率）：变形样本被误分为bona fide的比例
- **BPCER**（Bona fide分类错误率）：真实样本被误分为变形的比例
- **D-EER**（等错误率）：MACER=BPCER的交叉点
- **BPCER@5/10/20**：MACER为5%/10%/20%时的BPCER

## 实验关键数据

### 训练FERET → 测试FRGCv2

| 模型 | 合成数据比例 | Bona fide总量 | D-EER(%) ↓ | BPCER5(%) ↓ | BPCER10(%) ↓ |
|------|------------|-------------|------------|-------------|-------------|
| MobileNetV3 | 0% | 1,587 | 6.17 | 1.42 | 3.59 |
| EfficientNet-B2 | 0% | 1,587 | 6.47 | 1.97 | 4.57 |
| EfficientNet-B2 | 50% | 2,387 | 6.09 | 1.83 | 3.70 |
| **MobileNetV3** | **75%** | **2,787** | **6.10** | **1.05** | **3.05** |
| **EfficientNet-B2** | **75%** | **2,787** | **6.09** | **1.39** | **4.20** |
| MobileNetV3 | 100% | 3,174 | 8.17 | 2.58 | 6.23 |
| EfficientNet-B2 | 纯合成 | 25,000 | 37.96 | 61.24 | 76.51 |
| MobileNetV3 | 纯合成 | 25,000 | 38.95 | 62.32 | 73.73 |

### 训练FRGCv2 → 测试FERET

| 模型 | 合成数据比例 | D-EER(%) ↓ | BPCER5(%) ↓ |
|------|------------|------------|-------------|
| **EfficientNet-B2** | **10%** | **8.68** | **3.47** |
| **MobileNetV3** | **10%** | **10.20** | **3.66** |
| EfficientNet-B2 | 0% | 9.61 | 2.64 |
| EfficientNet-B2 | 75% | 14.05 | 8.95 |
| EfficientNet-B2 | 纯合成 | 37.57 | 57.54 |

### 关键发现

1. **最优合成比例因数据集而异**：FERET→FRGCv2最优为75%，FRGCv2→FERET最优为10%，说明合成数据的最佳比例取决于原始训练集的特性
2. **过度合成数据有害**：100%比例时性能开始下降（8.17% vs 75%时的6.10%）
3. **纯合成训练严重失败**：EER飙升至37-39%，合成-真实域差距导致泛化崩溃
4. **轻量架构表现良好**：EfficientNet-B2和MobileNetV3在参数量受限的情况下均达到竞争性能力，适合实际部署（边境检查、移动验证）

### 平均EER统计

| 配置 | EfficientNet-B2 $\overline{EER}$ | MobileNetV3 $\overline{EER}$ |
|------|--------------------------------|------------------------------|
| FERET→FRGCv2（全实验） | 6.86% | 6.93% |
| FRGCv2→FERET（全实验） | 10.46% | 12.05% |

## 亮点
- **系统性的增量研究**：从0%到100%再到纯合成的完整比例扫描，提供了清晰的合成数据引入指导
- **跨数据集评估严格性**：训练和测试使用完全不同的数据集（FERET vs FRGCv2），真实评估泛化能力
- **实际部署导向**：选用轻量级架构（2.9M-7.7M参数），适合移动端和边境检查等资源受限场景
- **多变形工具覆盖**：四种不同质量的变形工具（FaceFusion/FaceMorpher/OpenCV/UBO）+ GAN生成，覆盖实际攻击的多样性

## 局限与展望
- **Workshop论文篇幅限制**：个别实验细节和分析深度不够，如缺乏对合成数据域偏移(domain shift)的定量分析
- **合成数据生成方式单一**：仅使用SMDD（StyleGAN2-ADA），未探索其他生成方法（如扩散模型）
- **缺少最新backbone**：未测试Vision Transformer或更新的高效架构
- **无隐私-效用的量化分析**：虽然动机是隐私约束，但未量化不同合成比例对隐私保护的实际贡献
- **最优比例的可预测性**：不同数据集的最优比例差异大（10% vs 75%），缺乏理论指导来预判最优比例
- **数据增强策略探索不足**：未尝试print-scan仿真、压缩伪影等更贴近实际操作条件的增强方法
- **弱变形工具的影响**：FaceMorpher产生的变形伪影较为明显，可能会拉低整体检测难度

## 评分
- 新颖性: ⭐⭐⭐⭐ 增量合成数据研究系统但方法本身创新有限
- 实验充分度: ⭐⭐⭐⭐ 比例扫描全面，但缺乏深层分析和消融实验
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但Workshop篇幅限制了深度
- 价值: ⭐⭐⭐⭐ 为安全领域合成数据使用提供了实用指南

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Stacking Brick by Brick: Aligned Feature Isolation for Incremental Face Forgery Detection](../../CVPR2025/ai_safety/stacking_brick_by_brick_aligned_feature_isolation_for_incremental_face_forgery_d.md)
- [\[NeurIPS 2025\] Unifying Proportional Fairness in Centroid and Non-Centroid Clustering](unifying_proportional_fairness_in_centroid_and_non-centroid_clustering.md)
- [\[NeurIPS 2025\] Influence Functions for Edge Edits in Non-Convex Graph Neural Networks](influence_functions_for_edge_edits_in_non-convex_graph_neural_networks.md)
- [\[CVPR 2025\] Dynamic Integration of Task-Specific Adapters for Class Incremental Learning](../../CVPR2025/ai_safety/dynamic_integration_of_task-specific_adapters_for_class_incremental_learning.md)
- [\[AAAI 2026\] Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference](../../AAAI2026/ai_safety/diversifying_counterattacks_orthogonal_exploration_for_robust_clip_inference.md)

</div>

<!-- RELATED:END -->
