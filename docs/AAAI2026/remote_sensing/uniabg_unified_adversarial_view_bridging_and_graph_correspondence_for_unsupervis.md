---
title: >-
  [论文解读] UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization
description: >-
  [AAAI 2026 Oral][遥感][cross-view geo-localization] 提出双阶段无监督跨视角地理定位框架 UniABG，通过对抗式视角桥接 (VAAB) 消除无人机/卫星视角域差距，再用异构图过滤校准 (HGFC) 净化跨视角关联，在 University-1652 上 Satellite→Drone AP 达 93.29%，超过多数有监督方法。
tags:
  - "AAAI 2026 Oral"
  - "遥感"
  - "cross-view geo-localization"
  - "unsupervised"
  - "adversarial learning"
  - "graph filtering"
  - "伪标签"
---

# UniABG: Unified Adversarial View Bridging and Graph Correspondence for Unsupervised Cross-View Geo-Localization

**会议**: AAAI 2026 Oral  
**arXiv**: [2511.12054](https://arxiv.org/abs/2511.12054)  
**代码**: [GitHub](https://github.com/chenqi142/UniABG)  
**领域**: 遥感  
**关键词**: cross-view geo-localization, unsupervised, adversarial learning, graph filtering, pseudo-label  

## 一句话总结

提出双阶段无监督跨视角地理定位框架 UniABG，通过对抗式视角桥接 (VAAB) 消除无人机/卫星视角域差距，再用异构图过滤校准 (HGFC) 净化跨视角关联，在 University-1652 上 Satellite→Drone AP 达 93.29%，超过多数有监督方法。

## 背景与动机

- 跨视角地理定位 (CVGL) 要求将无人机查询图像与卫星图像匹配，但有监督方法依赖大规模成对标注，成本高
- 无监督方法 (UCVGL) 避免标注成本，但直接跨视角做伪标签关联面临两个核心困难：(1) 无人机/卫星视角的特征分布不对齐，同类跨视角距离可能大于不同类同视角距离；(2) 聚类不纯导致噪声实例引发灾难性关联错误

## 核心问题

如何在无标注条件下同时解决跨视角域差距和伪标签噪声传播问题？

## 方法详解

### 整体框架

UniABG 采用双阶段设计：Stage 1 通过 VAAB 学习视角不变特征 + DBSCAN 聚类生成伪标签；Stage 2 通过 HGFC 构建高质量跨视角关联对用于对比学习。Backbone 为 ConvNeXt-B。

### Stage 1: View-Aware Adversarial Bridging (VAAB)

1. **辅助伪视角 (APV) 生成**：将无人机图像在 Lab 色彩空间中做全局颜色迁移到卫星域：$l'_c = \frac{\sigma^s_c}{\sigma^d_c}(l_c - \mu^d_c) + \mu^s_c$，保留结构语义同时模拟视角过渡
2. **三视角对抗训练**：Backbone $F_B$ 提取三视角特征，视角判别器 $D_v$ 尝试分类来源，Backbone 对抗学习使特征视角不可区分：$\mathcal{L}_{\text{VAAB}} = \sum_{v \in \mathcal{V}} \text{CE}(D_v(f^v), t^v) + \sum_{v \in \mathcal{V}} \text{CE}(D_v(f^v), t^p)$
3. **视角内对比损失**：基于 DBSCAN 聚类伪标签和 memory dictionary 做 InfoNCE 对比学习

### Stage 2: Heterogeneous Graph Filtering Calibration (HGFC)

1. **异构图构建**：构建 Real-to-Real Graph $G_{RU}$（无人机-卫星 kNN 图）和 Pseudo-to-Real Graph $G_{PU}$（APV-卫星 kNN 图）
2. **拓扑一致性对齐**：计算交叉图一致性分数 $s_{ij}^{\text{cross}} = \frac{|N_k^{RU}(f_j^s) \cap N_k^{PU}(f_j^s)|}{k}$，仅保留超过阈值 $\tau$ 的关联
3. **语义引导加权投票**：$\omega_{ij} = \text{sim}(f_i^d, f_j^s) \cdot s_{ij}^{\text{cross}}$，在簇内加权投票得到最终伪标签

### 总损失

- Stage 1: $\mathcal{L}_{\text{stage1}} = \mathcal{L}_{iv} + \lambda \cdot \mathcal{L}_{\text{VAAB}}$，$\lambda=0.1$
- Stage 2: 使用 $\mathcal{L}_{\text{sup}} = \mathcal{L}_{\text{InfoNCE}} + \mathcal{L}_{\text{MSE}} + \mathcal{L}_{\text{CE}}$

## 实验关键数据

| 方法 | 类型 | Drone→Sat R@1 | Drone→Sat AP | Sat→Drone R@1 | Sat→Drone AP |
|------|------|------|------|------|------|
| Wang et al. | U | 85.95 | 90.33 | 94.01 | 82.66 |
| **UniABG** | **U** | **93.62** | **94.61** | **95.43** | **93.29** |
| DAC | S | 94.67 | 95.50 | 96.43 | 93.79 |
| QDFL | S | 95.00 | 95.83 | 97.15 | 94.57 |

- 消融实验：HGFC 贡献最大（Drone→Sat R@1 +54.89%），VAAB 进一步提升 +2.79%
- SUES-200 150m（最具挑战）：Drone→Sat R@1 92.40%（比 SOTA 无监督 +15.5%）

## 亮点

- 首个将对抗学习与图过滤结合的双阶段 UCVGL 框架
- APV 作为几何中间视角的设计巧妙，既辅助域对齐又用于图过滤中的多视角验证
- 无监督性能接近甚至超过有监督方法

## 局限与展望

- 依赖 DBSCAN 超参（如 eps），对不同数据集可能需要调参
- APV 的颜色迁移方式较简单（全局统计量），未考虑局部语义
- 仅在两个数据集上验证，缺少更大规模/更多样场景的测试

## 相关工作对比

| 维度 | Wang et al. (2025b) | UniABG |
|------|------|------|
| 域对齐 | 无显式对齐 | VAAB 对抗桥接 |
| 关联策略 | 直接聚类匹配 | HGFC 拓扑一致性过滤 |
| Sat→Drone AP | 82.66 | 93.29 (+10.63) |

## 启发

- 异构图的互 kNN 过滤思路可迁移到其他跨模态匹配任务
- 利用合成中间视角作为对抗训练的"桥"是减小域差距的通用策略

## 评分

⭐⭐⭐⭐ — 方法设计完整，实验充分，无监督接近有监督水平，值得关注

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] UniGeoRS: A Unified Benchmark for Tri-view Geo-Localization](../../CVPR2026/remote_sensing/unigeors_a_unified_benchmark_for_tri-view_geo-localization.md)
- [\[CVPR 2026\] RHO: Robust Holistic OSM-Based Metric Cross-View Geo-Localization](../../CVPR2026/remote_sensing/rho_robust_holistic_osm-based_metric_cross-view_geo-localization.md)
- [\[ECCV 2024\] ConGeo: Robust Cross-View Geo-Localization Across Ground View Variations](../../ECCV2024/remote_sensing/congeo_robust_cross-view_geo-localization_across_ground_view_variations.md)
- [\[CVPR 2026\] MOGeo: Beyond One-to-One Cross-View Object Geo-localization](../../CVPR2026/remote_sensing/mogeo_beyond_one-to-one_cross-view_object_geo-localization.md)
- [\[CVPR 2026\] Geo2: Geometry-Guided Cross-view Geo-Localization and Image Synthesis](../../CVPR2026/remote_sensing/geo2_geometry-guided_cross-view_geo-localization_and_image_synthesis.md)

</div>

<!-- RELATED:END -->
