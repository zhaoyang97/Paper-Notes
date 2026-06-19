---
title: >-
  [论文解读] GOATex: Geometry & Occlusion-Aware Texturing
description: >-
  [NeurIPS 2025][3D视觉][3D纹理生成] GOATex 提出首个遮挡感知的 3D 网格纹理生成框架，通过基于光线投射的 hit level 分层机制将网格分解为由外到内的可见性层，配合法线翻转和残差面聚类的两阶段可见性控制策略以及基于可见性权重的 UV 空间融合，实现了对外表面和被遮挡内表面的高质量纹理生成。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D纹理生成"
  - "遮挡感知"
  - "扩散模型"
  - "UV纹理融合"
  - "多层纹理"
---

# GOATex: Geometry & Occlusion-Aware Texturing

**会议**: NeurIPS 2025  
**arXiv**: [2511.23051](https://arxiv.org/abs/2511.23051)  
**代码**: 项目主页有展示，代码待确认  
**领域**: 3D视觉  
**关键词**: 3D纹理生成, 遮挡感知, 扩散模型, UV纹理融合, 多层纹理

## 一句话总结
GOATex 提出首个遮挡感知的 3D 网格纹理生成框架，通过基于光线投射的 hit level 分层机制将网格分解为由外到内的可见性层，配合法线翻转和残差面聚类的两阶段可见性控制策略以及基于可见性权重的 UV 空间融合，实现了对外表面和被遮挡内表面的高质量纹理生成。

## 研究背景与动机

**领域现状**：当前 3D 网格纹理生成方法（如 TEXTure、SyncMVD、Paint3D、TEXGen）主要依赖文本到图像扩散模型，通过多视图渲染后反投影到 UV 空间来生成纹理。这些方法在可见外表面上效果不错。

**现有痛点**：现有方法本质上只能处理从外部视角可见的表面。对于被遮挡的内部表面（如房屋内部、汽车仪表盘、帐篷内壁），它们要么完全不处理，要么使用 Voronoi 填充等启发式方法生成低质量的内部纹理，导致明显的接缝和色彩不连贯。

**核心矛盾**：基于多视图渲染-反投影的范式天然缺乏对被遮挡几何体的访问能力，而 UV 空间内的修复方法又缺乏几何上下文信息，难以生成语义合理的内部纹理。

**本文目标** (1) 如何系统性地识别和暴露被遮挡的内部表面？(2) 如何在暴露内部几何时保持整体结构一致性？(3) 如何将多层次独立生成的纹理无缝融合？

**切入角度**：将网格视为洋葱状的分层结构，通过光线投射计算每个面的"命中层级"(hit level)，从外向内逐层暴露并纹理化。

**核心 idea**：基于光线的遮挡感知分层机制 + 两阶段可见性控制 + 加权 UV 融合，实现内外表面一体化的高质量纹理生成。

## 方法详解

### 整体框架
输入为一个未纹理化的 3D 网格和文本提示。Pipeline 分为五步：(1) 超面构建——将细粒度三角面聚合为连贯区域；(2) Hit level 分配——通过多视角光线投射为每个超面分配可见性深度层级；(3) 可见性控制——两阶段策略逐层暴露内部几何；(4) 纹理合成——对每层使用深度条件扩散模型生成纹理；(5) UV 融合——基于可见性权重的软融合。最终输出完整的 UV 纹理贴图。

### 关键设计

1. **超面构建与 Hit Level 分配**:

    - 功能：将大量小三角面聚合为低曲率的连贯超面，然后为每个超面分配一个表示其可见性深度的 hit level。
    - 核心思路：使用 Xatlas 库对网格进行过分割得到超面。然后从多个视点投射光线，记录每条光线与面的交叉顺序 $k$。每条光线的影响权重由光线方向与面法线的余弦相似度加权：$W(f,k) = \sum_{r \in R_k(f)} \max(-n(f) \cdot d(r), 0)$。超面的最终 hit level 取所有面上权重最大的交叉顺序：$H(SF_i) = \arg\max_k \sum_{f \in SF_i} W(f,k)$。
    - 设计动机：直接对单个面分配 hit level 会导致同一平面区域的相邻面被分到不同层级，破坏语义一致性。超面聚合保证了同一连贯区域被统一处理。

2. **两阶段可见性控制策略**:

    - 功能：解决逐层暴露内部几何时面集合稀疏和碎片化的问题。
    - 核心思路：
        - **残差面聚类**：不是只渲染当前层的面，而是渲染所有尚未纹理化的面集合 $F_k^{res} = F - \bigcup_{i=1}^{k-1} F_i^{init}$，类似剥洋葱。
        - **法线翻转 + 背面剔除**：对已纹理化的面翻转法线，使其在当前视角下被背面剔除隐藏，暴露出内部未纹理化的面。最终渲染面集合 $F_k = F_k^{res} \cup (\overline{F} - \overline{F_k^{res}})$。
    - 设计动机：单纯按层级渲染会导致深层 hit level 的面集合非常稀疏和碎片化，产生的深度图偏离自然物体分布（OOD），扩散模型无法生成合理纹理。残差聚类增加密度，法线翻转保持整体形状。

3. **加权 UV 空间纹理融合**:

    - 功能：将各 hit level 独立生成的纹理无缝合并为最终纹理。
    - 核心思路：对每个视角 $v$ 在每个 hit level $k$，计算 UV 空间权重 $W_k^{(v)}$（基于视角方向与面法线的余弦相似度）。跨层级用修正 softmax 归一化：$\overline{W_k} = \frac{e^{W_k} \odot \mathcal{M}_k}{\sum_{j=1}^{H} e^{W_j} \odot \mathcal{M}_j}$。最终纹理 $\text{UV}_F = \sum_{k=1}^{H} \overline{W_k} \odot \text{UV}_k$。
    - 设计动机：简单的覆盖或均匀平均会导致内外表面边界模糊或出现伪影。基于可见性置信度的软融合实现了层间平滑过渡。

### 训练策略
GOATex 完全不需要微调预训练扩散模型，使用预训练的 Stable Diffusion 1.5 + ControlNet 作为纹理生成器。这大幅降低了计算成本并保证了泛化能力。同时支持外部和内部使用不同的文本提示（双提示），实现对分层外观的细粒度控制。

## 实验关键数据

### 数据集与评估
从 Objaverse/Objaverse-XL 中精选 226 个具有复杂内部几何的高质量网格（12 类物体：房屋、汽车、巴士、帐篷等）。由于缺乏内部纹理的 ground truth，采用用户研究和 GPT 评估的 A/B 偏好测试。

### 主实验（偏好率 %，对比基线 vs GOATex）

| 基线方法 | 人类偏好 GOATex | GPT-4o-mini | GPT-4o | GPT-4.1 | GPT-o3 |
|----------|----------------|-------------|--------|---------|--------|
| TEXTure | >60% | ~55% | ~52% | ~58% | ~55% |
| SyncMVD | >60% | ~55% | ~53% | ~57% | ~55% |
| Paint3D | >65% | ~70% | ~68% | ~72% | ~75% |
| TEXGen | >65% | ~72% | ~70% | ~75% | ~78% |

人类评估者在所有对比中一致强烈偏好 GOATex，尤其在内部纹理质量上优势明显。

### 消融实验（在 SyncMVD 基线上逐步添加组件的胜率 %）

| 配置 | GPT-4o-mini | GPT-4o | GPT-4.1 | GPT-o3 | 平均 |
|------|-------------|--------|---------|--------|------|
| Hit Level 分配 | 82.50 | 66.67 | 75.00 | 77.50 | 75.68 |
| + 超面构建 | 84.62 | 70.00 | 75.86 | 89.74 | 80.27 |
| + 软 UV 融合 | 79.49 | 82.05 | 90.00 | 95.00 | 86.49 |
| + 残差面聚类 | 77.50 | 72.50 | 88.89 | 84.84 | 81.17 |
| + 法线翻转（完整模型） | 86.84 | 92.31 | 86.67 | 97.50 | **91.16** |

### 关键发现
- 法线翻转 + 背面剔除是最关键组件，将整体胜率从 81.17% 提升到 91.16%。没有它，残差聚类单独使用反而会轻微降低效果，因为遮挡面无法被正确暴露。
- GPT-4.1 和 GPT-o3 与人类评估者的相关性更高（Pearson r 分别为 0.43 和 0.34），GPT 间一致性（Cohen's κ=0.54）高于人类间一致性（κ=0.31）。
- GPT 评估者倾向于偏好内部更"平滑"的结果，对显式内部纹理的评分略低于人类。

## 亮点与洞察
- **首个遮挡感知纹理生成**：hit level 的概念简洁而有效，将看似复杂的内部纹理问题转化为逐层渲染问题。法线翻转的 trick 巧妙利用了渲染管线的背面剔除特性来暴露内部面。
- **零微调方案**：完全不需要微调扩散模型，仅通过几何操作（分层、法线翻转、权重融合）就实现了显著提升，部署成本低。
- **双提示控制**：可以对内部和外部指定不同的文本提示，这为 3D 资产创作引入了新的控制维度，有望在游戏和 VR 场景制作中得到应用。

## 局限与展望
- Hit level 分配仅基于几何可见性，不考虑语义一致性。在薄壁或嵌套腔体等复杂几何中，语义统一的区域可能被分到不同层级，导致边界纹理不连贯。
- 方法依赖于超面分割的质量，Xatlas 的分割结果不一定总能保证语义正确性。
- 计算时间随 hit level 数量线性增长（每层需要一次完整的扩散推理）。5000 面的网格约需 2 分钟用于 hit level 分配 + 2 分钟纹理合成。
- 未来可以引入语义分割先验来改进 hit level 分配，或将多层级融合集成到去噪过程中以提升跨层一致性。

## 相关工作与启发
- **vs TEXTure/SyncMVD**: 它们通过迭代绘制/多视图同步来处理外部纹理，但对被遮挡区域只能用 Voronoi 填充。GOATex 的分层渲染机制从根本上解决了内部纹理问题。
- **vs Paint3D/TEXGen**: 它们在 UV 空间做修复/生成，但 UV 空间缺乏几何上下文，且高质量内部纹理训练数据匮乏。GOATex 在图像空间操作，利用了扩散模型更强的图像生成先验。
- **vs SDS 优化方法 (PaintIt, DreamMat)**: 它们通过分数蒸馏采样优化 UV 贴图，计算量大且同样无法处理遮挡区域。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统性处理 3D 网格内部纹理的方法，hit level 概念和法线翻转技巧有创意
- 实验充分度: ⭐⭐⭐⭐ 人类和多个 GPT 模型评估，消融全面，但缺少传统量化指标
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，方法描述逐步推进，图示丰富
- 价值: ⭐⭐⭐⭐ 解决了实际存在的痛点，双提示功能有应用价值，但局限于封闭/半封闭物体

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] GeoComplete: Geometry-Aware Diffusion for Reference-Driven Image Completion](geocomplete_geometry-aware_diffusion_for_reference-driven_image_completion.md)
- [\[ICCV 2025\] OccluGaussian: Occlusion-Aware Gaussian Splatting for Large Scene Reconstruction and Rendering](../../ICCV2025/3d_vision/occlugaussian_occlusion-aware_gaussian_splatting_for_large_scene_reconstruction_.md)
- [\[CVPR 2026\] Lafite: A Generative Latent Field for 3D Native Texturing](../../CVPR2026/3d_vision/lafite_a_generative_latent_field_for_3d_native_texturing.md)
- [\[CVPR 2025\] RoomTour3D: Geometry-Aware Video-Instruction Tuning for Embodied Navigation](../../CVPR2025/3d_vision/roomtour3d_geometry-aware_video-instruction_tuning_for_embodied_navigation.md)
- [\[CVPR 2026\] Geometry-Aligned and Anomaly-Aware Reconstruction for 3D Anomaly Detection](../../CVPR2026/3d_vision/geometry-aligned_and_anomaly-aware_reconstruction_for_3d_anomaly_detection.md)

</div>

<!-- RELATED:END -->
