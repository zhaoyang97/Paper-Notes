---
title: >-
  [论文解读] RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes
description: >-
  [NeurIPS 2025 Spotlight][3D视觉][camera parameter estimation] ROS-Cam 提出仅用单个RGB视频作为监督的动态场景相机参数（焦距+位姿）优化方法，通过Patch-wise跟踪过滤器建立稀疏鲁棒对应关系、Cauchy分布异常值感知联合优化自适应降权运动物体、以及基于Softplus/凸极小分析的两阶段优化策略，在5个数据集上以最少监督实现最优精度和最快速度。
tags:
  - "NeurIPS 2025 Spotlight"
  - "3D视觉"
  - "camera parameter estimation"
  - "dynamic scene"
  - "RGB-only supervision"
  - "outlier-aware optimization"
  - "visual odometry"
---

# RGB-Only Supervised Camera Parameter Optimization in Dynamic Scenes

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2509.15123](https://arxiv.org/abs/2509.15123)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: camera parameter estimation, dynamic scene, RGB-only supervision, outlier-aware optimization, visual odometry

## 一句话总结
ROS-Cam 提出仅用单个RGB视频作为监督的动态场景相机参数（焦距+位姿）优化方法，通过Patch-wise跟踪过滤器建立稀疏鲁棒对应关系、Cauchy分布异常值感知联合优化自适应降权运动物体、以及基于Softplus/凸极小分析的两阶段优化策略，在5个数据集上以最少监督实现最优精度和最快速度。

## 研究背景与动机

**领域现状**：COLMAP是静态场景相机参数估计的事实标准，但处理动态场景需要真值运动掩码来排除运动物体。近年来出现了大量改进方法（表1分类），但绝大多数依赖额外先验：GT焦距（CF-3DGS、Nope-NeRF）、GT运动掩码（GFlow、LEAP-VO）、度量深度（DROID-SLAM）、GT 3D点云和位姿（DUSt3R、Monst3R、Cut3R）。这些先验在随手拍摄的视频中通常不可获得。

**现有痛点**：(a) 仅有的几个RGB-only方法（VGGSfM、FlowMap、casualSAM）要么无法处理动态场景，要么依赖多个预训练稠密预测模型（RAFT/CoTracker/MiDAS）做伪监督——其中任一模型失效就导致整体性能下降；(b) 无法在不依赖GT运动先验的情况下自适应排除运动离群点；(c) 计算延迟高。

**核心矛盾**：能否仅用RGB视频——最基本的监督形式——在动态场景中准确高效地估计相机焦距和位姿？这需要同时解决三个难题：稀疏鲁棒的跟踪关系、运动物体的自适应排除、高效稳定的优化收敛。

**切入角度**：建立"最大程度稀疏"的铰链式跟踪关系（仅依赖点跟踪模型而非稠密预测），用Cauchy分布建模不确定性来降权运动离群点（而非分割/检测它们），两阶段优化策略平衡快速收敛和精确收敛。**核心 idea**：用最少的依赖（仅PT模型）提取最鲁棒的信息（稀疏高梯度跟踪点），用最robust的不确定性模型（Cauchy重尾分布）联合优化相机参数和3D校准点。

## 方法详解

### 整体框架
给定N帧RGB视频，ROS-Cam首先通过Patch-wise Tracking Filters从PT模型的输出中提取H条稀疏鲁棒跟踪轨迹作为伪监督。每条轨迹对应一个可学习的3D校准点P^{cali}。然后联合优化校准点、焦距f、旋转四元数Q、平移t和不确定性参数Γ。最后将估计的相机参数输入4DGS进行4D场景重建。

### 关键设计

1. **Patch-wise Tracking Filters（逐Patch跟踪过滤器）**:

    - 功能：从预训练点跟踪模型的输出中提取最稀疏、最鲁棒的跟踪轨迹作为优化的伪监督
    - 核心思路：四级过滤器级联——(a) Patch-wise Texture Filter：将图像划分为w×w的patch，计算每个patch的强度方差，仅保留纹理丰富的patch（高方差=易跟踪）；(b) Patch-wise Gradient Filter：在每个选中的patch内选择梯度模最大的像素作为跟踪点；(c) Visibility Filter：删除任何时刻变得不可见的轨迹（避免遮挡后重出现的跟踪误差）；(d) Patch-wise Distribution Filter：当多条轨迹落入同一patch时只保留梯度最大的那条，保证空间均匀分布
    - 设计动机：PT模型的注意力机制对纹理丰富/高梯度的点跟踪更准确（利用而非对抗PT模型的特性）。"铰链式"最大稀疏——大幅减少可学习参数数量和计算量，同时提高鲁棒性

2. **Outlier-aware Joint Optimization（异常值感知联合优化）**:

    - 功能：在不依赖任何运动先验/掩码的情况下，自适应降低运动物体对应点对优化的影响
    - 核心思路：为每个3D校准点P^{cali}_h关联一个不确定性参数Γ_h（用Softplus保证正值）。提出Average Cumulative Projection (ACP) Error——将每个校准点在所有帧上的投影误差累积取平均。构造Cauchy Loss: L = (1/H)Σlog(Γ + E²_ACP/Γ)，其中运动物体的校准点因三角测量不一致而产生高ACP误差→学出大Γ→被降权。使用四元数表示旋转（避免正交性约束）
    - 设计动机：Cauchy分布比高斯分布更能处理重尾（运动离群点产生的大误差），且其对数似然形式产生的Cauchy Loss天然具有对大误差的鲁棒性。不确定性关联在稀疏3D点上（而非2D像素上）大幅减少参数——NeRF-DS场景casualSAM有424×270×480个不确定性参数，ROS-Cam仅440个

3. **Two-stage Optimization Strategy（两阶段优化策略）**:

    - 功能：加速收敛并避免局部极小
    - 核心思路：Stage 1——固定Γ^{raw}=1（不学不确定性），仅优化P^{cali}/f/Q/t，利用Softplus(1)≈ln2的近似快速收敛到粗略解。Stage 2——用Stage 1的ACP误差初始化Γ^{raw}（基于Cauchy Loss内凸项Φ=x+O/x的最优解x*=√O），然后联合优化所有参数，运动物体被正确降权后进一步精化
    - 设计动机：如果从头联合优化Γ，Cauchy Loss的非凸性容易导致收敛不稳定。两阶段策略基于对Softplus渐近行为和Cauchy Loss凸子项的解析分析——是带有理论指导的工程设计而非启发式trick

### 损失函数 / 训练策略
总损失: L_total = L_cauchy + R_depth。L_cauchy为Cauchy Loss（主项），R_depth = (1/N)Σ-ReLU(P^{proj-homo}[:,3])为深度正则化（鼓励正深度）。Stage 1做200次迭代，Stage 2做50次迭代。

## 实验关键数据

### 主实验

| 方法 | 监督类型 | NeRF-DS PSNR↑ | DAVIS PSNR↑ | TUM ATE↓ | TUM RPE_t↓ | 运行时间 |
|------|---------|---------------|-------------|----------|-----------|---------|
| COLMAP(w/ mask) | GT运动掩码 | 32.17 | - | - | - | 1.5h |
| casualSAM | RGB-only | 21.23 | 19.03 | 0.071 | 0.010 | 10.5h |
| Robust-CVD | RGB-only | - | - | 0.153 | 0.026 | - |
| **ROS-Cam** | **RGB-only** | **33.55** | **22.29** | **0.065** | **0.010** | **0.83h** |

| 方法 | iPhone Avg. PSNR↑ | 方法类型 |
|------|------------------|---------|
| Record3D | ~25.5 | LiDAR传感器 |
| COLMAP(w/o mask) | ~21.0 | RGB-only(静态) |
| **ROS-Cam** | **~25.2** | RGB-only |

### 消融实验

| 配置 | NeRF-DS PSNR↑ | 说明 |
|------|---------------|------|
| Full (ROS-Cam) | 33.55 | 完整方法 |
| w/o two-stage | 25.95 | 去掉两阶段→不稳定收敛 |
| w/o Γ | 26.44 | 去掉不确定性→无法排除运动离群点 |
| w/o E_ACP | 23.56 | 去掉ACP误差→最差 |
| w/o texture filter | 25.99 | 跟踪点质量下降 |
| w/o gradient filter | 26.04 | 跟踪点质量下降 |
| w/o distribution filter | 26.02 | 跟踪点聚集不均匀 |

### 关键发现
- ROS-Cam在NeRF-DS上PSNR 33.55甚至超越使用GT运动掩码的COLMAP(32.17)——纯RGB监督反超GT掩码监督
- 运行时间线性增长（约1/800小时/帧），而COLMAP近似指数增长——长视频优势更大
- 在TUM-dynamics上位姿精度(ATE=0.065)优于需要GT焦距+度量深度的DROID-SLAM(0.043)和需要GT 3D点云的Monst3R(0.098)——相当于或超越使用更多监督的方法
- 消融显示每个组件都有显著贡献，ACP误差和不确定性参数是最关键的两个设计

## 亮点与洞察
- **"最少监督=最强泛化"的哲学**：通过极致减少对预训练模型和外部先验的依赖，反而避免了任何一个先验源出错的级联风险
- 不确定性参数的稀疏关联（3D点而非2D像素）是一个elegant的工程决策——parameter数量降低几个数量级同时保持效果
- Cauchy Loss的选择有理论支撑（重尾鲁棒性）且与ACP误差天然配合——形成了一个自洽的鲁棒估计框架
- 两阶段优化不是简单的coarse-to-fine，而是基于对Softplus渐近行为和Cauchy Loss凸子项的解析分析得到的理论指导设计

## 局限与展望
- 假设针孔相机模型和恒定焦距，不适用于鱼眼/变焦镜头
- 极端动态场景（几乎所有物体都在运动，静态点极少）可能导致三角测量退化
- RGB-only方法的精度天花板仍低于LiDAR等直接测量
- MPI-Sintel上一些高速运动场景表现不如casualSAM（如ambush_4/5）

## 相关工作与启发
- **稀疏vs稠密的取舍**：本文证明对相机估计而言"稀疏但高质量"的对应关系远优于"稠密但带噪"的伪监督
- **Cauchy分布在鲁棒估计中的应用**：可推广到更多需要抗离群点的优化问题
- **对4D重建pipeline的影响**：ROS-Cam可以作为任何4D重建方法的前端替代COLMAP，在动态场景中尤其有价值

## 评分
- 新颖性: ⭐⭐⭐⭐ 最小监督形式+Cauchy鲁棒估计的组合设计有独特性
- 实验充分度: ⭐⭐⭐⭐⭐ 5个数据集（3真实+1合成+1iPhone）、NVS+位姿+运行时间多维评估、详尽消融
- 写作质量: ⭐⭐⭐⭐ 方法动机清晰，理论分析扎实
- 价值: ⭐⭐⭐⭐ 对casually captured动态视频的3D/4D重建有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] D$^2$USt3R: Enhancing 3D Reconstruction for Dynamic Scenes](d2ust3r_enhancing_3d_reconstruction_for_dynamic_scenes.md)
- [\[CVPR 2026\] Coverage Optimization for Camera View Selection](../../CVPR2026/3d_vision/coverage_optimization_for_camera_view_selection.md)
- [\[CVPR 2026\] FastEventDGS: Deformable Gaussian Splatting for Fast Dynamic Scenes from a Single Event Camera](../../CVPR2026/3d_vision/fasteventdgs_deformable_gaussian_splatting_for_fast_dynamic_scenes_from_a_single.md)
- [\[CVPR 2025\] Dyn-HaMR: Recovering 4D Interacting Hand Motion from a Dynamic Camera](../../CVPR2025/3d_vision/dyn-hamr_recovering_4d_interacting_hand_motion_from_a_dynamic_camera.md)
- [\[CVPR 2025\] Joint Optimization of Neural Radiance Fields and Continuous Camera Motion from a Monocular Video](../../CVPR2025/3d_vision/joint_optimization_of_neural_radiance_fields_and_continuous_camera_motion_from_a.md)

</div>

<!-- RELATED:END -->
