---
title: >-
  [论文解读] Reliable Spatial-Temporal Voxels For Multi-Modal Test-Time Adaptation
description: >-
  [ECCV 2024][3D视觉][多模态测试时适应] 本文提出 Latte（ReLiable Spatial-temporal Voxels），一种多模态测试时适应方法，通过滑动窗口帧聚合构建时空体素（ST voxels）并计算时空熵（ST entropy）来评估预测可靠性，进而实现自适应跨模态学习，在三个 MM-TTA 基准上取得 SOTA 性能。
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "多模态测试时适应"
  - "3D语义分割"
  - "时空体素"
  - "跨模态学习"
  - "在线适应"
---

# Reliable Spatial-Temporal Voxels For Multi-Modal Test-Time Adaptation

**会议**: ECCV 2024  
**arXiv**: [2403.06461](https://arxiv.org/abs/2403.06461)  
**代码**: [https://sites.google.com/view/eccv24-latte](https://sites.google.com/view/eccv24-latte) (项目主页)  
**领域**: 3D视觉  
**关键词**: 多模态测试时适应, 3D语义分割, 时空体素, 跨模态学习, 在线适应

## 一句话总结

本文提出 Latte（ReLiable Spatial-temporal Voxels），一种多模态测试时适应方法，通过滑动窗口帧聚合构建时空体素（ST voxels）并计算时空熵（ST entropy）来评估预测可靠性，进而实现自适应跨模态学习，在三个 MM-TTA 基准上取得 SOTA 性能。

## 研究背景与动机

**领域现状**：3D 语义分割是自动驾驶和机器人导航的基础任务，多模态传感器（相机+LiDAR）被广泛采用。多模态测试时适应（MM-TTA）旨在测试阶段在线适应模型到未标注的目标域，无需访问源域数据。此前的 MMTTA 方法在单帧级别通过跨模态伪标签和预测一致性进行适应。

**现有痛点**：现有 MM-TTA 方法（如 MMTTA）依赖每帧的跨模态信息进行适应，但忽略了一个重要事实：连续帧中几何邻域的预测是高度相关的。由于域偏移，单帧预测通常不稳定——同一物体在连续帧中可能被预测为不同类别（如某辆车在某帧被正确识别，但在相邻帧中被错误分类）。这种时间不一致性导致：(1) 不可靠的预测被错误地视为"可靠"并传播到另一模态，造成错误累积；(2) 多增广帧的平均虽能缓解，但计算量随增广帧数线性增长。

**核心矛盾**：MM-TTA 需要可靠的监督信号来在线更新模型，但单帧预测在域偏移下恰恰不可靠。如何在不增加大量计算的前提下获得更稳定的可靠性估计，是关键挑战。

**本文目标** (1) 如何利用连续帧之间的时空关联来获得更可靠的预测估计？(2) 如何高效地评估每个模态在不同空间区域的可靠性？(3) 如何基于可靠性估计进行自适应的跨模态学习？

**切入角度**：作者观察到，3D 空间可以被划分为体素，同一体素内的不同帧的点可以视为对同一语义对象的不同观察。可靠的预测在时空邻域内应当是一致且确定的。通过聚合相邻帧在同一体素内的预测并评估其一致性，可以比单帧预测更可靠地判断每个模态、每个区域的可信度。

**核心 idea**：通过滑动窗口体素化聚合连续帧预测，用时空熵衡量预测可靠性，实现自适应跨模态测试时适应。

## 方法详解

### 整体框架

Latte 的 pipeline：(1) 对每个模态（2D/3D），使用 student-teacher 架构生成逐帧预测；(2) 通过滑动窗口聚合连续帧的点云，进行体素化构建 ST voxels；(3) 计算每个 ST voxel 内 teacher 预测的时空熵（ST entropy），低熵表示可靠，高熵的不可靠体素被过滤；(4) 基于 ST entropy 的跨模态加权，让 student 学习更可靠模态的 teacher 预测。

### 关键设计

1. **滑动窗口帧聚合与体素化（Slide Window Aggregation & Voxelization）**:

    - 功能：以高效的方式建立连续帧之间的时空对应关系
    - 核心思路：给定当前帧 $i$ 作为查询，在时间窗口 $\{j : |j-i| \leq w_t\}$ 内的帧通过位姿变换 $\mathbf{T}_{j \to i}$ 对齐到同一坐标系，合并后进行体素化，体素大小为 $\mathbf{s}$。同一体素内来自不同帧的点被视为时空对应。滑动窗口（$w_t=3$）使得每帧都有重叠的评估区间，比"全帧合并"或"无重叠分块"更能捕捉局部一致性
    - 设计动机：全帧合并无法突出短时间窗口内的不一致，帧到帧对应点数太少不够代表性。滑动窗口取折中，既有足够的对应点，又聚焦于局部一致性。利用 KISS-ICP 提供在线位姿估计，计算开销可控

2. **时空体素与时空熵（ST Voxels & ST Entropy）**:

    - 功能：量化每个模态在每个空间区域的预测可靠性
    - 核心思路：一个 ST voxel 包含查询（当前帧 student 预测 $\mathbf{p}_q^m$）和参考（窗口内多帧 teacher 预测 $\mathbf{p}_r^m$）。参考预测的平均类别概率的 Shannon 熵作为 ST 熵：$E_{i,k}^m = -\sum_c \bar{p}_{r,c}^m \log \bar{p}_{r,c}^m$。高 ST 熵表示参考预测在时空邻域内不一致或不确定——这些体素被 $\alpha$-分位过滤掉（$\alpha=0.9$）。低 ST 熵表示多帧 teacher 对该区域的预测一致且确信，应被信任
    - 设计动机：比起单帧的点级熵，ST 熵融合了多帧时空邻域的信息，提供更鲁棒的可靠性度量。实验证明 ST 熵比点级熵在三个基准上平均高 2.0% mIoU

3. **ST 体素辅助的自适应跨模态学习**:

    - 功能：根据模态可靠性自适应地进行跨模态知识迁移
    - 核心思路：在体素级别，计算跨模态加权：$w_v^{2D} = \frac{\exp(E_{i,k}^{2D})}{\exp(E_{i,k}^{2D}) + \exp(E_{i,k}^{3D})}$（高熵的模态权重高，即 KL 散度损失中该模态作为学生被引导）。跨模态一致性损失：$\mathcal{L}_{xM} = w_v^{2D} D_{KL}(\bar{p}_q^{3D} \| \bar{p}_r^{2D}) + w_v^{3D} D_{KL}(\bar{p}_q^{2D} \| \bar{p}_r^{3D})$。在点级别，ST 熵也传播到每个点，用于生成加权的跨模态伪标签：$\mathbf{p}^{xM} = w_p^{2D} \mathbf{p}^{2D} + w_p^{3D} \mathbf{p}^{3D}$
    - 设计动机：不同场景下，2D 和 3D 模态的可靠性可能截然不同（如户外场景 LiDAR 更可靠，但低纹理区域相机更可靠）。自适应加权让可靠模态引导不可靠模态，避免噪声传播

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \sum_t \mathcal{F}(\mathbf{p}_t^m, \mathbf{y}_t^{xM}) + \frac{\lambda_{xM}}{B} \sum_t \sum_k \mathcal{L}_{t,k}^{xM}$，其中 $\mathcal{F}$ 是交叉熵，$\mathbf{y}_t^{xM}$ 是跨模态伪标签。Teacher 通过 EMA 更新：$\tilde{\theta}_t^m = \lambda_s \tilde{\theta}_{t-1}^m + (1-\lambda_s) \theta_t^m$，$\lambda_s=0.99$。只更新 BN 层参数。严格遵循 one-pass 协议：先评估再更新。$\lambda_{xM}=0.3$。

## 实验关键数据

### 主实验

| 方法 | MM | U-to-S (xM) | A-to-S (xM) | S-to-S (xM) | Avg |
|------|-----|-------------|-------------|-------------|-----|
| Source only | ✗ | 43.9 | 44.3 | 38.2 | 42.1 |
| TENT | ✗ | 41.1 | 49.1 | 37.5 | 42.6 |
| SAR | ✗ | 43.9 | 50.3 | 32.9 | 42.4 |
| xMUDA+PL | ✓ | 43.0 | 50.9 | 36.3 | 43.4 |
| MMTTA | ✓ | 45.4 | 53.7 | 35.5 | 44.9 |
| **Latte** | ✓ | **46.0** | **54.3** | **41.6** | **47.3** |

### 消融实验

| No. | 配置 | U-to-S (xM) | A-to-S (xM) | S-to-S (xM) |
|-----|------|-------------|-------------|-------------|
| 0 | Source only | 43.9 | 44.3 | 38.2 |
| 7 | w/o 分位过滤 | 45.4 | 53.2 | 41.7 |
| 8 | 点级熵替代 ST 熵 | 45.1 | 52.7 | 38.2 |
| 9 | Full Latte | **46.0** | **54.3** | **41.6** |

### 关键发现

- **S-to-S 基准最具说服力**：这是合成→真实的挑战性场景，所有之前的方法（包括 MMTTA）都不如 Source only，只有 Latte 实现了 +3.4% 的正向适应。这证明了时空一致性在极端域偏移下的独特价值
- ST 熵比点级熵平均提升 2.0%+ mIoU，验证了多帧聚合比单帧评估更可靠
- 分位过滤（$\alpha=0.9$）在 U-to-S 和 A-to-S 上提升 0.6-1.1%，在 S-to-S 上差异较小
- 体素大小 0.2m 在所有基准上最优——过大导致语义边界模糊，过小则对应点不够
- 小窗口（$w_t=3$）比大窗口效果好，说明局部时间一致性比全局一致性更有信息量
- 几乎所有跨模态方法优于单模态 TTA，说明跨模态信息对 TTA 有重要价值

## 亮点与洞察

- **时空体素的巧妙设计**：利用 3D 点云天然的时空结构来评估预测可靠性——同一体素在不同时刻的预测应该一致。这种"以一致性度量可靠性"的思想可迁移到其他时序感知任务（如目标跟踪、SLAM）
- **滑动窗口 vs 全局聚合**：全局聚合不可取，帧到帧太稀疏，滑动窗口取折中。这种设计选择的思路在时序建模中很有参考价值——不是信息越多越好，而是要找到合适的时间粒度
- **自适应跨模态权重**：在体素级别根据可靠性动态调整跨模态学习方向，比全局固定的模态融合策略更精细。对自动驾驶等需要多传感器融合的场景有直接借鉴意义

## 局限与展望

- Latte 难以纠正在时间上一致性地犯错的预测——如果某个区域在所有帧中都被错误预测，ST 熵仍然很低，模型会错误地认为其可靠
- 依赖 SLAM 算法提供位姿估计，位姿不准确会影响体素化质量
- 只更新 BN 层参数限制了适应能力，可以探索更多可适应参数的策略
- 体素大小和窗口大小需要手动调参，虽然在三个基准上最优值一致（0.2m, $w_t=3$），但新数据集可能不同
- 未考虑目标检测等其他 3D 感知任务

## 相关工作与启发

- **vs MMTTA (Shin et al., CVPR 2022)**: MMTTA 通过单帧跨模态伪标签精炼来适应，但在 S-to-S 等挑战性场景下单帧预测不稳定导致失败。Latte 通过时空聚合获得更稳定的可靠性估计
- **vs CoTTA (Wang et al., CVPR 2022)**: CoTTA 用增广不变性缓解时间不稳定，但计算量随增广数线性增长。Latte 利用自然存在的帧间一致性，计算效率更高
- **vs GIPSO (Saltori et al., ECCV 2022)**: GIPSO 全局合并帧进行传播但无法突出局部不一致。Latte 的滑动窗口更好地评估局部一致性

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次将时空对应引入 MM-TTA，ST 体素和 ST 熵的设计有创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 三个基准、多种基线、详尽消融（组件有效性、聚合机制、参数敏感性、定性分析）
- 写作质量: ⭐⭐⭐⭐ 动机阐述清晰，方法描述严谨，数学公式完整
- 价值: ⭐⭐⭐⭐ 对自动驾驶在线适应有直接实用价值，在挑战性场景下优势突出

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] CloudFixer: Test-Time Adaptation for 3D Point Clouds via Diffusion-Guided Geometric Transformation](cloudfixer_test-time_adaptation_for_3d_point_clouds_via_diffusion-guided_geometr.md)
- [\[ICCV 2025\] 3D Test-time Adaptation via Graph Spectral Driven Point Shift](../../ICCV2025/3d_vision/3d_testtime_adaptation_via_graph_spectral_driven_point_shift.md)
- [\[NeurIPS 2025\] PointMAC: Meta-Learned Adaptation for Robust Test-Time Point Cloud Completion](../../NeurIPS2025/3d_vision/pointmac_meta-learned_adaptation_for_robust_test-time_point_cloud_completion.md)
- [\[CVPR 2026\] Anatomical Domain Shifts: Test-time Heterogeneous Adaptation for 3D Human Pose Prediction](../../CVPR2026/3d_vision/anatomical_domain_shifts_test-time_heterogeneous_adaptation_for_3d_human_pose_pr.md)
- [\[ECCV 2024\] Zero-Shot Multi-Object Scene Completion](zero-shot_multi-object_scene_completion.md)

</div>

<!-- RELATED:END -->
