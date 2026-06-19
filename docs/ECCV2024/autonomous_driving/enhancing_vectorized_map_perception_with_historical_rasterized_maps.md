---
title: >-
  [论文解读] Enhancing Vectorized Map Perception with Historical Rasterized Maps
description: >-
  [ECCV2024][自动驾驶][vectorized map perception] 提出 HRMapNet，通过维护一张低成本的全局历史栅格化地图（historical rasterized map），为在线矢量化地图感知提供互补先验信息，在 BEV 特征聚合和 query 初始化两个层面增强现有方法，在 nuScenes 和 Argoverse 2 上取得显著提升。
tags:
  - "ECCV2024"
  - "自动驾驶"
  - "vectorized map perception"
  - "historical rasterized map"
  - "BEV"
  - "bird's-eye-view"
  - "HD map"
---

# Enhancing Vectorized Map Perception with Historical Rasterized Maps

**会议**: ECCV2024  
**arXiv**: [2409.00620](https://arxiv.org/abs/2409.00620)  
**代码**: [HXMap/HRMapNet](https://github.com/HXMap/HRMapNet)  
**领域**: 自动驾驶  
**关键词**: vectorized map perception, historical rasterized map, BEV, bird's-eye-view, HD map, autonomous driving

## 一句话总结

提出 HRMapNet，通过维护一张低成本的全局历史栅格化地图（historical rasterized map），为在线矢量化地图感知提供互补先验信息，在 BEV 特征聚合和 query 初始化两个层面增强现有方法，在 nuScenes 和 Argoverse 2 上取得显著提升。

## 背景与动机

- 高精地图（HD map）对自动驾驶至关重要，但传统离线构建方式成本极高，促使研究者转向基于车载传感器的在线地图感知。
- 以 MapTR 为代表的在线矢量化地图感知方法在 BEV 空间直接预测矢量化地图元素，但仅依赖当前帧车载传感器，在遮挡、恶劣天气、夜间等挑战场景下精度和鲁棒性会大幅下降。
- 时序信息是一种可行的补充，但现有方法（如 StreamMapNet）仅利用短期前几帧的时序信息，无法充分发挥历史观测的全部价值。
- 论文的核心洞察：历史预测结果可以低成本地栅格化并累积为全局地图，作为在线感知的先验信息。栅格化地图具有易合并、易检索、语义清晰、内存占用小等优势。

## 核心问题

如何以低成本方式维护并利用历史地图信息，弥补单帧车载传感器在困难场景下的感知不足？

## 方法详解

### 1. 全局栅格化地图的构建与维护

- 每一帧的在线预测结果（矢量化地图）被栅格化为局部栅格图 $M_i^l \in \{0,1\}^{H \times W \times N}$，其中 $N=3$ 代表 lane divider、pedestrian crossing、road boundary 三种地图元素类别。
- 基于 ego-pose 将局部坐标映射到全局坐标，利用类似占据栅格地图（occupancy grid map）的更新规则维护全局地图 $M^g$：
    - 若局部预测该位置存在地图元素，全局值增加 $S^+$（默认 30）
    - 若不存在，全局值减少 $S^-$（默认 1）
- 检索时根据当前 ego-pose 从全局地图裁取局部栅格图，用阈值 $S_{th}$ 二值化。
- 全局地图使用 8-bit 无符号整型存储，内存开销约每公里 1 MB（nuScenes 波士顿地图仅约 120 MB，而 NMP 中基于 BEV 特征的地图需 11 GB）。

### 2. BEV 特征聚合模块（Feature Aggregation）

- 现有方法从车载图像提取 BEV 特征 $F_I \in \mathbb{R}^{H \times W \times C}$。
- HRMapNet 在检索到的局部栅格图中有地图元素的位置额外放置 BEV queries，通过 spatial cross-attention 从图像中提取对应特征，得到补充 BEV 特征 $F_M$。无地图元素的位置填零。
- 最终特征融合：$F_{BEV} = \text{Conv}(\text{Concat}(F_I + F_M, M^l))$，即将图像 BEV 特征、地图补充特征与栅格图本身的语义信息拼接后卷积融合。

### 3. Query 初始化模块（Query Initialization）

- 在 DETR 范式中，可学习 queries 需要从随机位置搜索地图元素。历史栅格图提供了元素可能存在位置的先验。
- 对栅格图中每个有效位置 $p$，编码位置嵌入 $PE(p)$ 和语义标签嵌入 $LE(p)$，相加得到地图先验嵌入 $ME(p) = PE(p) + LE(p)$。
- 基础 queries 通过 cross-attention 与地图先验嵌入交互，再送入原始 decoder 层。这使 queries 能更高效地定位目标元素。
- 为控制内存，局部栅格图在提取先验嵌入前进行下采样（默认分辨率 0.6 m）。

### 4. 训练与推理

- 损失函数与原始方法完全一致（分类损失、点对点损失、边方向损失等）。
- 训练时每个 epoch 全局地图从空开始逐步更新。
- 推理时全局地图同样默认初始为空，随测试帧按时间顺序推进而增量更新。

## 实验关键数据

### nuScenes 主要结果

| 方法 | 额外信息 | Epoch | mAP |
|------|----------|-------|-----|
| MapTRv2 | 无 | 24 | 61.5 |
| **HRMapNet (MapTRv2)** | HRMap | 24 | **67.2 (+5.7)** |
| MapTRv2 | 无 | 110 | 68.7 |
| **HRMapNet (MapTRv2)** | HRMap | 110 | **73.6 (+4.9)** |
| StreamMapNet | 无 | 24 | 60.4 |
| **HRMapNet (StreamMapNet)** | HRMap | 24 | **66.3 (+5.9)** |

### Argoverse 2 主要结果

| 方法 | mAP |
|------|-----|
| MapTRv2 | 64.3 |
| **HRMapNet (MapTRv2)** | **68.3 (+4.0)** |
| StreamMapNet | 61.5 |
| **HRMapNet (StreamMapNet)** | **64.3 (+2.8)** |

### 消融实验

- 仅 BEV 特征聚合：+3.1 mAP（61.5→64.6）
- 再加 Query 初始化：+2.6 mAP（64.6→67.2）
- 两个模块均有显著贡献

### 初始地图对性能的影响

| 初始地图 | mAP |
|----------|-----|
| 空地图（默认） | 67.2 |
| 验证集自建地图（跑两遍） | 72.6 (+5.4) |
| 训练集地图 | 83.7 (+16.5) |

### 定位误差鲁棒性

- 平移误差 0.1 m + 旋转误差 0.01 rad 时，mAP 仅下降约 1.2（67.2→66.0）
- 最严重噪声下（0.2 m + 0.02 rad）仍达 63.8，仍优于无地图的基线 61.5

### 推理速度

- HRMapNet + MapTRv2：17.0 FPS（基线 19.6 FPS）
- HRMapNet + StreamMapNet：21.1 FPS（基线 22.5 FPS）
- 速度下降可接受，仍满足实时需求

## 亮点

1. **思路简洁有效**：将历史预测栅格化为全局地图并回馈在线感知，概念简单但效果显著，mAP 提升 4-6 个点。
2. **即插即用**：框架可与多数现有矢量化地图感知方法集成，论文验证了 MapTRv2 和 StreamMapNet 两种代表性方法。
3. **极低存储成本**：全局栅格图仅需约 1 MB/km，远低于基于 BEV 特征的方法（NMP 需 11 GB）。
4. **鲁棒性好**：对定位误差鲁棒，在常见定位精度范围内性能几乎不受影响。
5. **实际应用潜力大**：天然适合众包地图感知场景——多车共同维护全局地图。

## 局限与展望

1. 论文主要关注如何利用历史栅格图，但栅格图的质量取决于在线感知精度，在首次到达新区域时无历史信息可用，退化为纯在线感知。
2. 全局地图更新策略较简单（固定增减值），缺少基于置信度的自适应更新机制。
3. Query 初始化模块在 0.3 m 分辨率下训练显存达 65 GB，需要下采样来控制开销，这可能丢失细粒度信息。
4. 仅在 nuScenes 和 Argoverse 2 上验证，两者训练集与验证集有地理位置重叠，在完全未见区域的效果有待更多验证。
5. 未考虑地图过时/场景变化的问题——历史地图中的信息可能因施工等原因不再有效。

## 与相关工作的对比

| 方法 | 信息来源 | 特点 |
|------|----------|------|
| MapTRv2 | 单帧图像 | 基线方法，无额外信息 |
| StreamMapNet | 短期时序 | query propagation + BEV fusion |
| SQD-MapNet | 短期时序 | stream query denoising |
| P-MapNet | SD Map（OSM） | 需外部数据，提升有限 |
| NMP | 历史 BEV 特征 | 内存占用巨大（11 GB） |
| **HRMapNet** | 历史栅格图 | 低成本、全历史、即插即用，效果最优 |

## 启发与关联

- 栅格化地图作为轻量级历史信息载体的思路可以推广到其他 BEV 感知任务（如 3D 目标检测、占据预测）。
- 众包地图感知是一个很有价值的实际应用方向——多车共建共享全局地图，可进一步提升每辆车的感知能力。
- Query 初始化的设计思路（用先验信息引导 DETR queries 搜索）具有通用性，可应用于其他检测/分割任务中的先验注入。
- 初始地图实验（训练图达 83.7 mAP）暗示：如果有高质量历史地图，在线感知精度可以大幅跃升，这对实际部署意义重大。

## 评分

- 新颖性: ⭐⭐⭐⭐ （思路虽简单但 insight 扎实，栅格图作为低成本历史信息载体是好的工程选择）
- 实验充分度: ⭐⭐⭐⭐⭐ （两数据集、两基线、消融完善、鲁棒性/初始地图等额外实验丰富）
- 写作质量: ⭐⭐⭐⭐ （逻辑清晰，图表信息量大）
- 价值: ⭐⭐⭐⭐ （对实际自动驾驶地图感知部署有直接参考价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Stream Query Denoising for Vectorized HD-Map Construction](stream_query_denoising_for_vectorized_hd-map_construction.md)
- [\[NeurIPS 2025\] SDTagNet: Leveraging Text-Annotated Navigation Maps for Online HD Map Construction](../../NeurIPS2025/autonomous_driving/sdtagnet_leveraging_text-annotated_navigation_maps_for_online_hd_map_constructio.md)
- [\[CVPR 2026\] OptiMVMap: Offline Vectorized Map Construction via Optimal Multi-vehicle Perspectives](../../CVPR2026/autonomous_driving/optimvmap_offline_vectorized_map_construction_via_optimal_multi-vehicle_perspect.md)
- [\[CVPR 2025\] MapGCLR: Geospatial Contrastive Learning of Representations for Online Vectorized HD Map Construction](../../CVPR2025/autonomous_driving/mapgclr_geospatial_contrastive_learning_of_representations_for_online_vectorized.md)
- [\[CVPR 2025\] Driving by the Rules: A Benchmark for Integrating Traffic Sign Regulations into Vectorized HD Map](../../CVPR2025/autonomous_driving/driving_by_the_rules_a_benchmark_for_integrating_traffic_sign_regulations_into_v.md)

</div>

<!-- RELATED:END -->
