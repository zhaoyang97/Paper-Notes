---
title: >-
  [论文解读] CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance
description: >-
  [ICCV 2025][3D视觉][Feature Matching] 提出 CasP，一种级联匹配流水线，将匹配阶段分解为 1/16 尺度的一对多先验匹配和 1/8 尺度的一对一精细匹配，在保持精度的同时实现最高 2.2× 加速，并显著提升跨域泛化能力。 半稠密特征匹配（以 LoFTR 为代表）通过将特征图中每个 toke…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "Feature Matching"
  - "Cascaded Matching"
  - "Semi-Dense"
  - "Efficiency"
  - "Cross-Domain Generalization"
---

# CasP: Improving Semi-Dense Feature Matching Pipeline Leveraging Cascaded Correspondence Priors for Guidance

**会议**: ICCV 2025  
**arXiv**: [2507.17312](https://arxiv.org/abs/2507.17312)  
**代码**: [GitHub](https://github.com/pq-chen/CasP)  
**领域**: 三维视觉 / 特征匹配  
**关键词**: Feature Matching, Cascaded Matching, Semi-Dense, Efficiency, Cross-Domain Generalization

## 一句话总结

提出 CasP，一种级联匹配流水线，将匹配阶段分解为 1/16 尺度的一对多先验匹配和 1/8 尺度的一对一精细匹配，在保持精度的同时实现最高 2.2× 加速，并显著提升跨域泛化能力。

## 研究背景与动机

半稠密特征匹配（以 LoFTR 为代表）通过将特征图中每个 token 视为候选匹配点取代了显式特征检测阶段，在低纹理和重复模式场景中表现强大。但其效率瓶颈在于匹配阶段需要在整个特征图上进行全局搜索：

**延迟瓶颈**：随着输入分辨率增加，匹配阶段的 token 数量急剧增长，占总运行时间的比例越来越大。ELoFTR 虽引入聚合注意力，但在 1152 分辨率下匹配阶段仍占大量时间

**准确性-效率权衡失败**：ELoFTR 尝试直接移除 dual-softmax (DS) 算子来加速，但导致显著精度下降，因为匹配阶段仅依赖描述符相似度而忽略全局置信度

**跨域泛化不足**：现有方法在 ScanNet（室内）等跨域基准上性能增益有限

核心设计思路：将主要计算操作延迟到更粗的尺度执行，用级联先验约束精细匹配的搜索范围。

## 方法详解

### 整体框架

CasP 流水线包含四个阶段：
1. **特征提取**：轻量 CNN（低级 1/2~1/8）+ Context Cluster（高级 1/16~1/32）
2. **特征交互**：混合交互模块（聚合注意力 + Cross-CoC）
3. **级联匹配**：一对多匹配（1/16）→ RSCA → 一对一匹配（1/8）
4. **匹配精细化**：两阶段单应性精细化（像素级 + 亚像素级）

### 关键设计

1. **高级特征提取（Self-CoC）**：

    - 采用 Context Cluster 机制替代卷积提取 1/16 和 1/32 尺度特征
    - 通过聚类-聚合-分发三阶段进行间接的全局点对点交互
    - 通过控制锚点数量管理计算成本，实现全局感受野的上下文理解

2. **级联匹配模块**：

    - **一对多匹配（1/16 尺度）**：构建相关性矩阵 $S_{1/16}$，为每个 token 选取 top-$k$（$k=8$）对应先验 $\pi_{1/16}$。训练时使用 DS 算子生成置信度矩阵并注入 ground-truth 加速收敛；推理时直接用原始相关性做 top-k 选择
    - **RSCA（Region-based Selective Cross-Attention）**：将特征图按 $r \times r$ 分块，每个块仅对先验区域做交叉注意力，增强先验候选间的特征判别力。每个 query 长度 $r^2$，key/value 长度 $k \cdot r^2$
    - **一对一匹配（1/8 尺度）**：引入 Partial Softmax——仅对 RSCA 关注的 token 计算 softmax，其余置零。匹配必须满足双向先验约束：$j \in \phi_r(\pi^A)[i]$ 且 $i \in \phi_r(\pi^B)[j]$

3. **训练-推理解耦策略**：

    - 训练时：两阶段均使用 DS 算子提供监督信号；向 top-k 中注入 ground-truth 先验加速 RSCA 学习
    - 推理时：一对多匹配省略 DS 算子；一对一匹配使用 Partial Softmax 替代完整 softmax
    - 效果：训练时最大化表征能力，推理时最大化效率

### 损失函数 / 训练策略

$$L = \lambda_1 L_{1/16}^c + \lambda_2 L_{1/8}^c + \lambda_3 L_{1/1}^f + \lambda_4 L_{\text{sub}}^f$$

- $L_{1/s}^c$：粗匹配负对数似然损失（$s \in \{8, 16\}$）
- $L_{1/1}^f$：像素级精细化损失
- $L_{\text{sub}}^f$：亚像素 $\ell_2$ 损失
- 权重：$\lambda_1=0.5, \lambda_2=0.5, \lambda_3=0.25, \lambda_4=1.0$
- MegaDepth 训练，8×V100，batch 8，30 epochs

## 实验关键数据

### 主实验（相对位姿估计 AUC + 运行时间）

| 方法 | MD-1500 @5° | @10° | @20° | SN-1500 @5° | @10° | @20° | MD 时间(ms) | SN 时间(ms) |
|------|-------------|------|------|-------------|------|------|-------------|-------------|
| LoFTR | 52.8 | 69.2 | 81.2 | 16.9 | 33.6 | 50.6 | 347.6 | 71.7 |
| ELoFTR | 56.4 | 72.2 | 83.5 | 19.2 | 37.0 | 53.6 | 238.3 | 45.2 |
| AffineFormer | 57.3 | 72.8 | 84.0 | 22.0 | 40.9 | 58.0 | ≥347.6 | ≥71.7 |
| **Ours-full** | **57.1** | **72.7** | **83.9** | **23.0** | **41.6** | **58.7** | **147.2** | **40.1** |
| **Ours-lite** | 55.6 | 71.7 | 83.3 | 21.6 | 40.1 | 57.0 | **108.1** | **33.1** |

### 消融实验（ETH3D 跨域评估 + 效率）

| 方法 | ETH3D[O] @5° | @10° | @20° | ETH3D[I] @5° | @10° | @20° | GMACs | Runtime(ms) |
|------|--------------|------|------|--------------|------|------|-------|-------------|
| EL w/ DS | 56.7 | 63.2 | 69.1 | 49.1 | 55.0 | 59.4 | 909.1 | 238.3 |
| EL w/o DS | 53.4 | 60.1 | 66.3 | 44.7 | 50.8 | 55.7 | 909.1 | 185.3 |
| EL+CM-full | 60.1 | 65.6 | 70.6 | 52.3 | 57.2 | 60.7 | 708.1 | 144.6 |
| **Ours-full** | **61.8** | **66.8** | **71.5** | **56.1** | **60.7** | **64.0** | **691.0** | **147.2** |
| **Ours-lite** | 60.3 | 65.9 | 71.2 | 54.3 | 59.2 | 62.7 | 365.1 | 108.1 |

**低级特征提取参数对比**：

| 方法 | 类型 | 通道配置 | 参数量(M) |
|------|------|----------|-----------|
| LoFTR | ResNet | [128,196,256] | 5.9 |
| ELoFTR | RepVGG | [64,128,256] | 9.5 |
| Ours-full | RepVGG | [64,128,192] | 2.0 |
| Ours-lite | RepVGG | [64,64,128] | 0.8 |

### 关键发现

- **跨域泛化显著**：SN-1500 上 full 模型 @5° 比 ELoFTR 高 3.8%，ETH3D[I] @5° 高 7.0%——级联先验约束有效提升泛化
- 直接移除 DS 算子会在 ETH3D[I] 上掉 4.4%（@5°），说明简单加速策略不可行
- EL+CM（仅替换匹配模块为级联）的 lite 版在 ETH3D[O] 上已超过原 EL-full，证明 CasP 流水线本身的贡献
- lite 模型特征提取参数仅 0.8M（ELoFTR 的 1/12），但准确率可比
- 在 1152 分辨率下 lite 模型比 ELoFTR 快 **2.2×**，比 LoFTR 快 **3.2×**
- HPatches 单应性估计中@3px 达到 71.8%，接近稠密方法 DKM

## 亮点与洞察

- **加速核心思想精准**：将计算密集操作推迟到更粗尺度，用廉价的 one-to-many 先验约束昂贵的 one-to-one 匹配搜索范围
- **训练-推理解耦设计精巧**：训练时保留 DS 算子的监督信号和 GT 注入，推理时全部省略以最大化效率
- **Partial Softmax 替代全局 softmax**：仅在先验区域内计算归一化，大幅减少无关计算
- **跨域泛化的惊喜发现**：级联先验不仅加速，还成为泛化的正则化——先验约束减少了错误匹配的搜索空间

## 局限与展望

- top-$k$ 值固定（$k=8$），不同场景的最优值可能不同，自适应 $k$ 是潜在改进方向
- MegaDepth 训练的模型在 InLoc（室内）上改进幅度有限，室内场景的匹配仍有提升空间
- 两阶段单应性精细化假设局部区域满足刚性变换，对大形变场景可能不适用
- 未探讨与学习型 RANSAC 等后处理方法的结合

## 相关工作与启发

- **LoFTR**：半稠密匹配的奠基工作，本文分析了其匹配阶段的效率瓶颈
- **ELoFTR**：引入聚合注意力和 RepVGG，本文进一步用级联流水线解决其残余瓶颈
- **ASpanFormer / AffineFormer**：多层次交叉注意力提升精度但增加运行时间
- **EcoMatcher**：Cross-CoC 机制的来源，本文在混合交互和高级特征提取中复用
- **HomoMatcher**：两阶段单应性精细化的灵感来源

## 评分

- **新颖性**: ⭐⭐⭐⭐ 级联先验+训练推理解耦+Partial Softmax 的系统性设计有新意
- **实验充分度**: ⭐⭐⭐⭐⭐ 姿态/单应性/定位多任务+ETH3D 跨域消融+详细效率分析+可视化
- **写作质量**: ⭐⭐⭐⭐ 流水线图清晰，公式推导完整，效率分析透彻
- **价值**: ⭐⭐⭐⭐⭐ SLAM/UAV 等实时系统的重要工程贡献，2.2× 加速且精度不降

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] TextFM: Robust Semi-dense Feature Matching with Language Guidance](../../CVPR2026/3d_vision/textfm_robust_semi-dense_feature_matching_with_language_guidance.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [\[ICCV 2025\] Sat2City: 3D City Generation from A Single Satellite Image with Cascaded Latent Diffusion](sat2city_3d_city_generation_from_a_single_satellite_image_with_cascaded_latent_d.md)
- [\[ICCV 2025\] S3R-GS: Streamlining the Pipeline for Large-Scale Street Scene Reconstruction](s3r-gs_streamlining_the_pipeline_for_large-scale_street_scene_reconstruction.md)
- [\[ICCV 2025\] Vivid4D: Improving 4D Reconstruction from Monocular Video by Video Inpainting](vivid4d_improving_4d_reconstruction_from_monocular_video_by_video_inpainting.md)

</div>

<!-- RELATED:END -->
