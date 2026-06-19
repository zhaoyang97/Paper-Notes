---
title: >-
  [论文解读] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 提出遮挡感知的场景划分策略和基于区域的渲染技术,通过相机共可见性图聚类实现与场景布局对齐的分区,显著提升大场景3DGS重建质量和渲染速度。 大场景重建在自动驾驶、文化遗产保护和虚拟/增强现实等领域至关重要。3D Gaussian Splatti…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "大场景重建"
  - "遮挡感知"
  - "场景划分"
  - "渲染加速"
---

# OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering

**会议**: ICCV 2025  
**arXiv**: [2503.16177](https://arxiv.org/abs/2503.16177)  
**代码**: [项目页面](https://occlugaussian.github.io)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 大场景重建, 遮挡感知, 场景划分, 渲染加速

## 一句话总结

提出遮挡感知的场景划分策略和基于区域的渲染技术,通过相机共可见性图聚类实现与场景布局对齐的分区,显著提升大场景3DGS重建质量和渲染速度。

## 研究背景与动机

大场景重建在自动驾驶、文化遗产保护和虚拟/增强现实等领域至关重要。3D Gaussian Splatting (3DGS) 由于内存密集的表示方式,在大场景中面临可扩展性挑战,因此通常采用**分治策略**将场景划分为多个小区域分别重建。

然而,现有场景划分方法存在一个共同缺陷:**对遮挡不感知**。它们主要基于相机位置或点云进行均匀划分,忽视了场景布局和遮挡关系。这导致:

**区域内包含严重遮挡** — 例如被墙壁或建筑物隔开的相机被划分到同一区域

**相机间相关性低** — 区域内相机的共可见内容少,平均贡献度低

**重建质量下降** — 训练资源被分散到不相关的区域

特别是在地面级采集的场景中,频繁出现的墙壁、建筑物等遮挡物使问题更加突出。此外,重建完成后的大场景包含大量Gaussian基元,渲染速度也是一个关键瓶颈。

## 方法详解

### 整体框架

OccluGaussian包含两个核心创新:
1. **遮挡感知场景划分** — 基于属性图聚类的相机分区策略
2. **基于区域的渲染加速** — 剔除对当前视点不可见的被遮挡Gaussian

### 遮挡感知场景划分

#### 属性视图图构建

构建无向属性图 $\mathcal{G}=(\mathcal{V},\mathcal{E},X)$:
- **节点**: 每个节点对应一个相机
- **边**: 如果两个相机共享可见内容,则建立边,权重为匹配特征点数量,得到邻接矩阵 $A \in \mathbb{R}^{n \times n}$
- **特征**: 每个相机的3D坐标经位置编码后作为节点特征 $X \in \mathbb{R}^{n \times d}$

被遮挡或距离较远的相机通常共享极少的重叠视角,可在图中被有效区分。

#### 图聚类

使用属性图聚类算法,先进行图卷积生成平滑特征,再进行谱聚类:

$$L_s = I - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$$

图卷积滤波器: $G = (I - \frac{1}{2}L_s)^r$

通过滤波获得 $\bar{X} = GX$,计算相似度矩阵后应用谱聚类,使共享大量重叠视角或空间接近的相机聚类到同一区域。

#### 自适应聚类数确定

从初始聚类数 $K$ 开始递归细化:
- 分裂包含过多相机的聚类
- 忽略过少相机的聚类或被其他聚类凸包完全覆盖的聚类
- 递归直到所有聚类达到平衡大小

### 区域重建

每个区域选择三类训练相机:
- **基础集**: 位于区域内的相机
- **扩展集**: 区域外但能捕获足够可见内容的相机
- **边界集**: 面向区域但被遮挡的相机,约束边界附近的Gaussian基元

### 基于区域的渲染加速

为每个区域记录对其内部所有训练相机可见的3D Gaussian。渲染时只处理当前视点所在区域记录的Gaussian,有效剔除了被遮挡的不可见Gaussian。还可进一步细分为更小的子区域以进一步加速。

## 实验

### 主实验 — OccluScene3D数据集

| 场景 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | FPS↑ |
|------|------|-------|-------|--------|------|
| Gallery | VastGaussian | 25.09 | 0.903 | 0.095 | 215 |
| Gallery | CityGaussian | 21.98 | 0.808 | 0.294 | 120 |
| Gallery | Hierarchical-GS | 22.23 | 0.800 | 0.182 | 216 |
| Gallery | **OccluGaussian** | **25.81** | **0.903** | **0.094** | **289** |
| Canteen | VastGaussian | 24.60 | 0.890 | 0.105 | 211 |
| Canteen | **OccluGaussian** | **25.25** | **0.900** | **0.100** | **312** |
| ClassBuilding | VastGaussian | 24.05 | 0.884 | 0.111 | 270 |
| ClassBuilding | **OccluGaussian** | **25.33** | **0.921** | **0.083** | **340** |

### Zip-NeRF数据集对比

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| MERF | 23.49 | 0.747 | 0.445 |
| SMERF | 27.28 | 0.829 | 0.340 |
| Zip-NeRF | 27.37 | 0.836 | — |

### 关键发现

1. OccluGaussian在所有场景上均取得最优重建质量,PSNR提升1-3 dB
2. 渲染速度显著提升,FPS从200+提升到280-340,基于区域的渲染有效避免了对不可见Gaussian的冗余计算
3. 遮挡感知的划分策略使每个区域内相机具有更强的相关性和更高的平均贡献度

## 亮点与洞察

1. **问题定义精准** — 从"相机Co-visibility"角度重新审视场景划分问题,抓住了遮挡导致重建质量下降的核心原因
2. **综合利用SfM信息** — 直接复用SfM的匹配信息构建视图图,无额外计算开销
3. **渲染加速与划分策略统一** — 遮挡感知的区域划分天然支持基于区域的渲染剔除,一举两得
4. **地面级场景适用性强** — 特别适合室内/城市等遮挡频繁的场景

## 局限性

- 依赖SfM提供的相机位姿和匹配信息
- 主要针对地面级采集场景,对航拍等开放场景改进有限
- 聚类数的自适应确定策略需递归细化,可能在极大场景上开销较大

## 相关工作

- **大场景重建**: BlockNeRF, Mega-NeRF, VastGaussian, CityGaussian 等分治策略
- **相机聚类**: COLMAP的Metis图分割, Out-of-Core-BA
- **渲染加速**: Octree-GS, LightGaussian, Hierarchical-3DGS 等LoD方法

## 评分

- 新颖性: ⭐⭐⭐⭐ (遮挡感知划分+区域渲染的统一设计新颖)
- 技术深度: ⭐⭐⭐⭐ (图聚类方法完整且合理)
- 实验充分度: ⭐⭐⭐⭐ (多数据集验证,定量定性全面)
- 实用价值: ⭐⭐⭐⭐⭐ (直接改善大场景重建的实际痛点)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)
- [\[NeurIPS 2025\] GOATex: Geometry & Occlusion-Aware Texturing](../../NeurIPS2025/3d_vision/goatex_geometry_occlusion-aware_texturing.md)
- [\[ICCV 2025\] Momentum-GS: Momentum Gaussian Self-Distillation for High-Quality Large Scene Reconstruction](momentum-gs_momentum_gaussian_self-distillation_for_high-quality_large_scene_rec.md)
- [\[ICCV 2025\] Curve-Aware Gaussian Splatting for 3D Parametric Curve Reconstruction](curve-aware_gaussian_splatting_for_3d_parametric_curve_reconstruction.md)
- [\[ICCV 2025\] CoMoGaussian: Continuous Motion-Aware Gaussian Splatting from Motion-Blurred Images](comogaussian_continuous_motionaware_gaussian_splatting_from.md)

</div>

<!-- RELATED:END -->
