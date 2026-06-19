---
title: >-
  [论文解读] TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels
description: >-
  [NeurIPS 2025][3D跟踪] 提出TrackingWorld，一个从单目视频实现几乎所有像素的稠密3D跟踪的流水线，通过跟踪上采样器将稀疏2D轨迹提升为稠密轨迹、迭代跟踪所有帧中新出现的物体、以及基于优化的框架将2D轨迹提升到世界坐标系3D空间并显式分离相机运动和物体运动。 现有3D跟踪方法的两个根本缺陷 缺陷一…
tags:
  - "NeurIPS 2025"
  - "3D跟踪"
  - "单目视频"
  - "世界坐标系"
  - "稠密跟踪"
  - "相机姿态估计"
---

# TrackingWorld: World-centric Monocular 3D Tracking of Almost All Pixels

**会议**: NeurIPS 2025  
**arXiv**: [2512.08358](https://arxiv.org/abs/2512.08358)  
**代码**: [项目页](https://igl-hkust.github.io/TrackingWorld.github.io/)  
**领域**: 视频理解  
**关键词**: 3D跟踪, 单目视频, 世界坐标系, 稠密跟踪, 相机姿态估计

## 一句话总结

提出TrackingWorld，一个从单目视频实现几乎所有像素的稠密3D跟踪的流水线，通过跟踪上采样器将稀疏2D轨迹提升为稠密轨迹、迭代跟踪所有帧中新出现的物体、以及基于优化的框架将2D轨迹提升到世界坐标系3D空间并显式分离相机运动和物体运动。

## 研究背景与动机

### 现有3D跟踪方法的两个根本缺陷

**缺陷一：无法分离相机运动和物体运动**

OmniMotion、SpatialTracker、DELTA等方法均假设相机静态，在相机坐标系中建模3D flow。但下游任务（如运动分析、新视角合成）普遍需要区分相机运动和动态物体运动。最近的MotionGS工作也表明，显式考虑相机姿态能提升3D跟踪质量。虽有ST4RTrack和TAPIP3D尝试世界坐标系跟踪，但前者有长期漂移问题，后者仅限稀疏跟踪且无法恢复相机运动。

**缺陷二：仅能跟踪首帧像素**

现有方法局限于跟踪视频首帧中的稀疏像素，无法追踪后续帧中新出现的动态目标。DELTA虽提出上采样器产生稠密3D轨迹，但仍限于首帧。如何在所有帧中估计所有像素的稠密3D轨迹仍是未解问题。

### "Almost All Pixels"的含义

"几乎所有"是指在最终结果中会过滤掉一些噪声和离群轨迹以确保鲁棒性。这是有意义的工程权衡而非方法局限。

## 方法详解

### 整体框架

TrackingWorld分为两个主要阶段：
1. **稠密2D跟踪**：将稀疏2D轨迹提升为稠密并覆盖所有帧
2. **2D到3D提升**：通过三阶段优化框架估计相机姿态并将2D轨迹转换为世界坐标系3D轨迹

输入为单目视频及基础模型的预处理结果（稀疏轨迹、深度图、动态掩码），输出为稠密3D轨迹和每帧相机姿态。

### 关键设计

#### 1. 稀疏到稠密的2D轨迹上采样

**核心发现**：DELTA的上采样模块可泛化到任意2D轨迹（不限于DELTA自身生成的轨迹）。

给定稀疏2D轨迹 $\mathbf{P}_{\text{sparse}} \in \mathbb{R}^{(\frac{H}{s} \times \frac{W}{s}) \times T \times 2}$，上采样器通过特征预测权重矩阵 $\mathbf{W}$：

$$\mathbf{P}_{\text{dense}} = \mathbf{W}^T \mathbf{P}_{\text{sparse}}$$

实际仅关联每个稠密点与其空间邻近的稀疏轨迹，计算高效。

**逐帧跟踪与去冗余**：在所有帧上执行2D跟踪和上采样，但大部分区域已在前序帧被跟踪过。因此，若某像素位于任何已有可见2D轨迹附近，则丢弃该像素，并通过连通分量分析删除面积小于阈值 $\tau=50$ 的孤立区域。实验证明此过滤策略一致提升精度。

#### 2. 初始相机姿态估计（Stage 1）

利用前景动态掩码选择静态区域的2D轨迹 $\mathbf{P}_{\text{static}}$，通过单目深度反投影到3D空间后，定义重投影损失：

$$\mathcal{L}_{\text{proj}} = \sum_i^{N_{\text{inliers}}} \sum_{t_1}^{T} \sum_{t_2}^{T} \|\pi_{t_2} \pi_{t_1}^{-1}(\mathbf{P}_{\text{static}}(i,t_1), \mathbf{D}_{\text{static}}(i,t_1)) - \mathbf{P}_{\text{static}}(i,t_2)\|_2^2$$

计算效率优化：先将视频分为C个clip并行估计clip内姿态，再估计clip间姿态合并全局姿态。

#### 3. 动态背景精化（Stage 2）

**动机**：前景动态掩码通常不够准确，背景中仍可能存在运动物体（如滚动的苹果），干扰bundle adjustment。

引入**尽可能静态约束（As-Static-As-Possible, ASAP）**：将"静态"区域的每个点也建模为有时变偏移量 $\mathbf{O}_{\text{static}}$：

$$\mathbf{T}'_{\text{static}}(i,t) = \mathbf{T}_{\text{static}}(i) + \mathbf{O}_{\text{static}}(i,t)$$

同时优化相机姿态和静态3D坐标，使用bundle adjustment损失加ASAP正则：

$$\mathcal{L}_{\text{asap}} = \sum_{i,t} \|\mathbf{O}_{\text{static}}(i,t)\|_1$$

L1范数迫使大多数偏移为零（真正静态的点），而非零偏移的点则被识别为动态背景点。联合目标为：

$$\mathcal{L}_{\text{static}} = \lambda_{\text{ba}} \mathcal{L}_{\text{ba}} + \lambda_{\text{dc}} \mathcal{L}_{\text{dc}} + \lambda_{\text{asap}} \mathcal{L}_{\text{asap}}$$

其中 $\lambda_{\text{ba}}=1, \lambda_{\text{dc}}=1, \lambda_{\text{asap}}=5$。$\mathcal{L}_{\text{dc}}$ 为深度一致性损失，约束投影深度与单目深度估计一致。

#### 4. 动态物体跟踪（Stage 3）

将 $\|\mathbf{O}_{\text{static}}(i,\cdot)\|_2 \geq \varepsilon$ 的背景点也归入动态类别。动态3D轨迹直接优化 $\mathbf{T}_{\text{dynamic}} \in \mathbb{R}^{N_{\text{dynamic}} \times T \times 3}$，训练目标包含重投影损失、深度一致性损失、尽可能刚性约束 $\mathcal{L}_{\text{arap}}$ 和时间平滑约束 $\mathcal{L}_{\text{ts}}$：

$$\mathcal{L}_{\text{dyn}} = \lambda_{\text{ba}} \mathcal{L}_{\text{ba}} + \lambda_{\text{dc}} \mathcal{L}_{\text{dc}} + \lambda_{\text{arap}} \mathcal{L}_{\text{arap}} + \lambda_{\text{ts}} \mathcal{L}_{\text{ts}}$$

其中 $\lambda_{\text{arap}}=100, \lambda_{\text{ts}}=10$。

### 训练策略

整体为优化框架（非学习框架），在RTX 4090上处理30帧视频约20分钟。利用clip级并行和静态点下采样（下采样因子 $\varpi$）加速优化，实测从60分钟降至8分钟且精度几乎无损。

## 实验关键数据

### 相机姿态估计

| 方法 | Sintel ATE↓ | Sintel RTE↓ | Bonn ATE↓ | TUM-D ATE↓ |
|------|-------------|-------------|-----------|------------|
| MonST3R | 0.111 | 0.044 | 0.029 | 0.063 |
| Align3R | 0.128 | 0.042 | 0.023 | 0.027 |
| Uni4D* | 0.116 | 0.046 | 0.017 | 0.039 |
| **Ours (DELTA)** | **0.088** | **0.035** | **0.016** | **0.016** |

### 稠密3D跟踪深度精度

| 方法 | Sintel Abs Rel↓ | Sintel δ<1.25↑ | Bonn Abs Rel↓ | TUM-D Abs Rel↓ |
|------|----------------|----------------|---------------|----------------|
| DELTA+UniDepth（无优化） | 0.636 | 63.1 | 0.153 | 0.178 |
| **Ours (DELTA)** | **0.218** | **73.3** | **0.058** | **0.084** |

### 消融实验

| 配置 | ATE↓ | RTE↓ | RRE↓ | Abs Rel↓ | δ<1.25↑ |
|------|------|------|------|----------|---------|
| w/o 逐帧跟踪 | 0.171 | 0.047 | 0.748 | / | / |
| w/o 初始姿态 | 0.659 | 0.153 | 1.382 | 0.230 | 72.4 |
| w/o 动态物体跟踪 | 0.088 | 0.035 | 0.410 | 0.468 | 73.0 |
| w/o $\mathbf{O}_{\text{static}}$ | 0.092 | 0.036 | 0.459 | 0.224 | 72.6 |
| w/o 深度一致性损失 | 0.093 | 0.036 | 0.441 | 0.234 | 71.2 |
| **完整模型** | **0.088** | **0.035** | **0.410** | **0.218** | **73.3** |

### 关键发现

1. **逐帧跟踪至关重要**：去掉后ATE从0.088恶化到0.171（+94%），因为丢失了许多后续帧的关键姿态估计线索
2. **初始姿态必不可少**：无良好初始化时相机姿态几乎无法恢复（ATE 0.659），联合优化难以同时收敛
3. **ASAP约束有效**：去掉 $\mathbf{O}_{\text{static}}$ 后RRE从0.410恶化到0.459，可视化显示动态背景物体（如苹果）被错误投影
4. 2D上采样器泛化性强：应用于CoTrackerV3轨迹后EPE下降（1.45→1.24），运行时间3.00→0.25分钟（12×加速）
5. 对不同深度估计模型（ZoeDepth、Depth Pro、UniDepth）均稳健提升，证明方法对深度先验质量有容忍度

## 亮点与洞察

1. **显式世界坐标系建模**：与DELTA/SpatialTracker在相机坐标系中操作不同，显式分离相机运动和物体运动带来了质量上的显著提升
2. **ASAP约束设计巧妙**：通过L1稀疏正则自动识别动态背景，不依赖完美的分割掩码
3. **模块化+基础模型组合**：灵活接入不同的2D跟踪器、深度估计器和动态分割器，形成可扩展的流水线
4. **物理一致的3D跟踪**：通过bundle adjustment强制几何一致性，深度精度相比原始单目估计提升约3倍
5. **副产品**：可直接输出时间一致的视频深度序列，在多个基准上超过现有video depth方法

## 局限与展望

- 依赖多个辅助模型（2D跟踪器、深度估计器、动态掩码），引入额外计算开销和对组件质量的要求
- 优化方式处理30帧需约20分钟（加速后8分钟），距实时仍有差距
- 前馈解决方案（如受VGGT启发联合处理所有帧直接预测状态）可能是更高效的未来方向
- ST4RTrack的配对匹配存在漂移，但前馈设计的思路值得借鉴
- 对极端遮挡或大视角变化的场景鲁棒性未充分验证

## 相关工作与启发

- **2D点跟踪**: CoTrackerV3, TAPIR, LocoTrack, TAP-Net
- **3D点跟踪**: SpatialTracker, DELTA, OmniMotion, ST4RTrack, TAPIP3D
- **4D重建**: Uni4D, MonST3R, Align3R, MegaSaM
- **深度估计**: UniDepth, Depth Pro, DepthCrafter
- **启发**: ASAP约束可推广到任何需要静动分离的场景理解任务；跟踪上采样器的泛化能力提示了模块化设计的优势

## 评分
- 新颖性: ⭐⭐⭐⭐☆ — 世界坐标系稠密3D跟踪+逐帧扩展是重要进展，但各组件多基于已有技术
- 实验充分度: ⭐⭐⭐⭐⭐ — 相机姿态、深度精度、稀疏3D跟踪、稠密2D跟踪四维度全面评估+丰富消融
- 写作质量: ⭐⭐⭐⭐☆ — 流水线描述清晰，问题定义准确
- 价值: ⭐⭐⭐⭐⭐ — 为稠密3D跟踪建立了新的性能标杆，可作为多个下游任务的基础模块

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Ultrametric Cluster Hierarchies: I Want 'em All!](ultrametric_cluster_hierarchies_i_want_em_all.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](../../ICLR2026/others/latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[NeurIPS 2025\] MetaFind: Scene-Aware 3D Asset Retrieval for Coherent Metaverse Scene Generation](metafind_scene-aware_3d_asset_retrieval_for_coherent_metaverse_scene_generation.md)
- [\[CVPR 2026\] Multi-view Crowd Tracking Transformer with View-Ground Interactions Under Large Real-World Scenes](../../CVPR2026/others/multi-view_crowd_tracking_transformer_with_view-ground_interactions_under_large_.md)
- [\[ICCV 2025\] Recovering Parametric Scenes from Very Few Time-of-Flight Pixels](../../ICCV2025/others/recovering_parametric_scenes_from_very_few_time-of-flight_pixels.md)

</div>

<!-- RELATED:END -->
