---
title: >-
  [论文解读] LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth
description: >-
  [ICML 2025][遥感][海岸线距离计算] 提出了一个全球10米分辨率的海岸线数据集和毫秒级查询库 Lighthouse，通过融合 ESA WorldCover 与 OpenStreetMap 数据，结合分层 BallTree + 球面 Voronoi 索引实现仅需1 CPU/2GB RAM的实时海岸距离查询，精度比已有数据集提升100倍以上。
tags:
  - "ICML 2025"
  - "遥感"
  - "海岸线距离计算"
  - "球面Voronoi剖分"
  - "BallTree"
  - "高分辨率地表覆盖"
  - "实时地理查询"
---

# LIGHTHOUSE: Fast and Precise Distance to Shoreline Calculations from Anywhere on Earth

**会议**: ICML 2025  
**arXiv**: [2506.18842](https://arxiv.org/abs/2506.18842)  
**代码**: [github.com/allenai/lighthouse](https://github.com/allenai/lighthouse) (Apache 2.0)  
**领域**: 遥感  
**关键词**: 海岸线距离计算, 球面Voronoi剖分, BallTree, 高分辨率地表覆盖, 实时地理查询

## 一句话总结

提出了一个全球10米分辨率的海岸线数据集和毫秒级查询库 Lighthouse，通过融合 ESA WorldCover 与 OpenStreetMap 数据，结合分层 BallTree + 球面 Voronoi 索引实现仅需1 CPU/2GB RAM的实时海岸距离查询，精度比已有数据集提升100倍以上。

## 研究背景与动机

海岸线距离数据在海岸侵蚀监测、基础设施规划、栖息地变化追踪以及卫星图像目标检测/分割等任务中至关重要。现有全球海岸线数据集分辨率极低（典型为1-4km），无法满足精细化需求：

- **GSHHG**：分辨率约1855米，仅覆盖公海区域
- **NASA OBPG**：分辨率4000米（也有1km插值版本），仅覆盖公海
- **ArcGIS/ESRI**：分辨率200km，仅覆盖陆地

这些粗分辨率数据在近岸和内陆区域的应用价值极为有限。同时，即使拥有高分辨率海岸线数据，如何在10米级精度下实现全球范围的快速查询也是一个巨大的计算挑战——简单的预计算方案需要约100TB的RAM存储，完全不切实际。

## 方法详解

### 整体框架

Lighthouse 的设计分为两个阶段：**离线数据构建**和**在线实时查询**。

**离线阶段**：
1. 融合 ESA WorldCover V2（10米分辨率卫星图像衍生地表覆盖图）和 OpenStreetMap 众包标注数据
2. 将全球划分为1°×1°瓦片，对每个瓦片提取海岸线点并构建 BallTree 索引
3. 对全球海岸线点进行球面 Voronoi 剖分，用于定位海上（非瓦片内）查询点的最近海岸线

**在线阶段**：
1. 根据查询点坐标判断所属瓦片或通过 Voronoi 查找最近瓦片
2. 加载对应 BallTree，以 Haversine 距离度量查询最近海岸点
3. 通过 h5 文件查询该点的地表覆盖类别（陆地/水体）

### 关键设计

#### 1. 数据融合策略

ESA WorldCover V2 是目前永久水体标注精度最高的全球10米地表覆盖图（基于Sentinel-2），但存在关键缺失：
- **密克罗尼西亚**数百个岛屿未覆盖
- **南极洲**在其发布时数据不可用
- 格陵兰北部、夏威夷、南大西洋部分岛屿缺失

解决方案：用 OpenStreetMap 的众包海岸线标注补全缺失区域。OSM 全球分辨率不均匀，但覆盖了所有岛屿且包含南极洲的最新标注。两者融合后得到完整的全球海岸线地图。

#### 2. 海岸线点提取（Algorithm 1）

对每个1°×1°瓦片执行以下流程：
1. **二值化**：将地表覆盖标签转为水体 vs. 非水体的二值掩码
2. **Sobel边缘检测**：提取水陆交界线像素
3. **BallTree构建**：对边缘坐标以 Haversine 度量构建 BallTree
4. **无压缩存储**：牺牲少量磁盘空间换取最低读取延迟

选择 Haversine 而非更精确的 Vincenty 公式，因为后者计算复杂度显著更高，而 Haversine 在实际应用中精度已足够。

#### 3. 球面 Voronoi 剖分

对于落在公海（不在任何陆地瓦片内）的查询点，需要确定其最近的海岸线瓦片。直接遍历所有瓦片不可行，因此预计算全球海岸线点的球面 Voronoi 剖分。

关键约束的下采样策略：
- 由于 Voronoi 构建的时间复杂度为 $O(n^2)$，不能包含所有海岸线点
- **约束1**：原始数据集中每个线段至少保留一个代表点
- **约束2**：相连点之间距离不超过1km阈值

这保证了不会遗漏任何关键岛屿或海岸结构。

#### 4. h5格式单点查询

地表覆盖数据存储为 h5 格式而非标准 GeoTIFF，关键优势在于可以查询单个像素的类别标签而无需将整个瓦片加载到内存中，大幅降低内存开销。

### 损失函数 / 训练策略

本文不涉及深度学习训练，核心是数据工程和算法优化。关键的工程决策包括：

- **延迟优先**：所有设计选择以最小化查询延迟为第一目标（如 BallTree 不压缩、使用 h5 单点查询）
- **瓦片缓存**：在线查询时维护瓦片缓存，避免重复加载
- **批量查询优化**：支持向量化计算和缓存复用的批量查询模式

## 实验关键数据

### 主实验

| 数据集 | 分辨率 (m) | 覆盖范围 | 开源 | 备注 |
|--------|-----------|---------|------|------|
| **Lighthouse (本文)** | **~10** | **全球** | **是** | 含内陆水体 |
| GSHHG | 1855 | 公海 | 是 | 1弧分分辨率 |
| NASA OBPG | 4000 | 公海 | 是 | 也有1km插值版 |
| ArcGIS/ESRI | 200000 | 仅陆地 | 是 | 陆地到公海 |

分辨率提升：相比现有最优数据集 GSHHG，Lighthouse 实现了 **~185倍** 的分辨率提升。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 仅 ESA WorldCover | 覆盖不完整 | 密克罗尼西亚、南极洲等缺失 |
| 仅 OpenStreetMap | 分辨率较粗 | 南极洲中位段间距35米 |
| ESA + OSM 融合 | 10m全球覆盖 | 互补优势，完整覆盖 |
| BallTree 有压缩 | 延迟增加 | 磁盘节省小，延迟代价大 |
| BallTree 无压缩 | 毫秒级延迟 | 本文采用方案 |
| GeoTIFF 格式 | 需加载整个瓦片 | 内存开销大 |
| h5 格式 | 单像素查询 | 本文采用方案 |

### 关键发现

1. **资源效率极高**：仅需1个CPU和2GB RAM即可实现毫秒级在线推理
2. **单次查询延迟远低于10ms**，批量查询通过缓存和向量化进一步加速
3. **OSM在南极洲的中位标注段间距为35米**，作为最差情况的分辨率下界
4. 该方法可扩展到商用卫星图像的最高分辨率（截至2025年6月为10cm）
5. 生产环境验证：已在 Skylight Global（卫星船只检测平台）中部署运行

## 亮点与洞察

1. **问题选择精准**：海岸线距离看似简单，但在10m分辨率下实现全球实时查询是一个非平凡的系统设计问题
2. **分层搜索架构**：BallTree（局部精确查询）+ Voronoi（全局路由）的两层设计优雅地解决了"高分辨率≠高成本"的矛盾
3. **数据融合的互补性**：WorldCover 的高精度 + OSM 的高覆盖率，两者优缺点互补
4. **工程导向的设计哲学**：始终优先延迟而非存储，例如不压缩 BallTree、使用 h5 单点查询
5. **曼德博分形几何的呼应**：论文引用了曼德博关于海岸线复杂度无穷的经典论述，点明高分辨率带来的误差放大问题

## 局限与展望

1. **分辨率非均匀**：ESA 部分为10m，但 OSM 补充区域分辨率不一致（南极洲约35m）
2. **时间动态性**：海岸线随海平面上升、冰川消退等不断变化，数据集是某一时间点的快照
3. **海岸线定义模糊**：悬崖、沙滩、港湾、湿地等复杂地形的边界定义存在固有歧义
4. **标注误差**：混合了 CV 模型的误分类和众包标注的人为错误
5. **Haversine 近似**：Haversine 在长距离上不如 Vincenty 精确，但本文认为在实际应用中足够

## 相关工作与启发

- **ESA WorldCover V2** (Zanaga et al., 2022)：基于 Sentinel-2 的全球10m地表覆盖图，本文的主要数据源
- **Dynamic World** (Brown et al., 2022)：Google 的近实时全球10m地表覆盖图，未被选择因为永久水体精度不如 WorldCover
- **NOAA C-CAP**：美国大陆1m分辨率地表覆盖图，高精度但仅区域覆盖
- **分形几何** (Mandelbrot, 1982)：海岸线长度的无穷性讨论，提示高分辨率数据的固有限制
- 对遥感领域的启发：将高精度地表覆盖数据与高效空间索引结合，可为其他地理要素（河流、道路边界等）的全球实时查询提供参考范式

## 评分

| 维度 | 分数 (1-10) | 说明 |
|------|------------|------|
| 新颖性 | 7 | 方法不复杂但问题定义和系统设计有价值 |
| 实用性 | 9 | 已在生产环境部署，开源数据和代码 |
| 技术深度 | 6 | 核心是数据工程，无深度学习创新 |
| 写作质量 | 8 | 清晰简洁，幽默（库名缩写），非常务实 |
| 可复现性 | 10 | 代码和数据完全开源 |
| **综合** | **7.5** | 一篇典型的"做对的事比做难的事更重要"的工作 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents](../../ICLR2026/remote_sensing/earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents.md)
- [\[ICML 2025\] High-Resolution Live Fuel Moisture Content (LFMC) Maps for Wildfire Risk from Multimodal Earth Observation Data](high-resolution_live_fuel_moisture_content_lfmc_maps_for_wildfire_risk_from_mult.md)
- [\[ICCV 2025\] Towards a Unified Copernicus Foundation Model for Earth Vision](../../ICCV2025/remote_sensing/towards_a_unified_copernicus_foundation_model_for_earth_vision.md)
- [\[CVPR 2026\] Local Precise Refinement: A Dual-Gated Mixture-of-Experts for Enhancing Foundation Model Generalization against Spectral Shifts](../../CVPR2026/remote_sensing/local_precise_refinement_a_dual-gated_mixture-of-experts_for_enhancing_foundatio.md)
- [\[CVPR 2025\] EarthDial: Turning Multi-sensory Earth Observations to Interactive Dialogues](../../CVPR2025/remote_sensing/earthdial_turning_multi-sensory_earth_observations_to_interactive_dialogues.md)

</div>

<!-- RELATED:END -->
