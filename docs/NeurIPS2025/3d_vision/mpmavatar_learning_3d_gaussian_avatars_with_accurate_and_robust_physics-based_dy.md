---
title: >-
  [论文解读] MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics
description: >-
  [NeurIPS 2025][3D视觉][3D human avatar] MPMAvatar 将 Material Point Method (MPM) 物理仿真器与 3D 高斯溅射渲染相结合，通过各向异性本构模型和面向网格碰撞体的新碰撞处理算法，实现宽松衣物的精确鲁棒物理动画——在 ActorsHQ 和 4D-DRESS 上几何和外观全面超越 PhysAvatar，仿真成功率 100% vs 37.6%，单帧仿真仅需 1.1 秒。
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D human avatar"
  - "physics-based simulation"
  - "MPM"
  - "3D Gaussian Splatting"
  - "garment dynamics"
---

# MPMAvatar: Learning 3D Gaussian Avatars with Accurate and Robust Physics-Based Dynamics

**会议**: NeurIPS 2025  
**arXiv**: [2510.01619](https://arxiv.org/abs/2510.01619)  
**代码**: [https://KAISTChangmin.github.io/MPMAvatar/](https://KAISTChangmin.github.io/MPMAvatar/)  
**领域**: 3D视觉  
**关键词**: 3D human avatar, physics-based simulation, MPM, 3D Gaussian Splatting, garment dynamics

## 一句话总结
MPMAvatar 将 Material Point Method (MPM) 物理仿真器与 3D 高斯溅射渲染相结合，通过各向异性本构模型和面向网格碰撞体的新碰撞处理算法，实现宽松衣物的精确鲁棒物理动画——在 ActorsHQ 和 4D-DRESS 上几何和外观全面超越 PhysAvatar，仿真成功率 100% vs 37.6%，单帧仿真仅需 1.1 秒。

## 研究背景与动机

**领域现状**：从多视角视频创建3D人体化身是计算机视觉和图形学的核心问题，近年来基于3D Gaussian Splatting (3DGS) 的方法在自由视角渲染上取得了显著进展。然而，对宽松衣物（如裙子、风衣）进行物理真实的动画化仍是一大挑战。

**现有痛点**：主流方法使用分段线性变换 (LBS) 或姿态依赖的几何校正来驱动衣物运动，但无法捕捉褶皱、飘动等复杂变形，且严重过拟合训练动作。少数尝试引入物理仿真的工作存在关键限制：Xiang等人采用X-PBD但需要耗时的人工参数搜索；PhysAvatar采用C-IPC仿真器，但当驱动身体网格存在微小自穿透时（这在实用的参数化人体估计中极常见）仿真会彻底崩溃，必须人工调整body mesh才能运行。在外观方面，PhysAvatar依赖mesh-based渲染无法捕捉细粒度纹理。

**核心矛盾**：真实的衣物动画需要物理仿真保证泛化性，但现有物理仿真方案在鲁棒性和精度之间难以兼顾——C-IPC精度尚可但极其脆弱，PBD速度快但物理精度不足。**核心idea**：采用Material Point Method作为仿真核心——MPM天然擅长处理大变形和复杂碰撞，再针对衣物的各向异性物理特性和网格碰撞体进行定制化改造，与3DGS渲染结合实现精度-鲁棒性-效率的全面优势。

## 方法详解

### 整体框架
MPMAvatar包含三个紧密耦合的模块：(1) 混合化身表示——用三角网格+物理参数建模几何和动力学，3D Gaussian Splats建模外观；(2) 基于定制MPM的物理动力学仿真——用各向异性本构模型和新碰撞算法驱动衣物运动；(3) 从多视角视频学习物理参数和外观参数。body区域用LBS驱动，衣物区域由MPM仿真驱动。

### 关键设计

1. **各向异性本构模型（Anisotropic Constitutive Model）**:

    - 功能：精确建模衣物的方向依赖物理行为——沿面内方向容易拉伸，法向方向几乎不可压缩
    - 核心思路：采用Jiang等人的各向异性本构模型，将应变能密度ψ按QR分解后的R矩阵重参数化为三个独立分量：ψ_normal（法向变形惩罚，由刚度κ控制）、ψ_shear（剪切惩罚，由γ控制）、ψ_in-plane（面内变形，由杨氏模量E和泊松比ν控制）。通过对每个粒子的材料方向做跟踪来区分不同方向的力学响应
    - 设计动机：标准MPM使用各向同性本构（如Neo-Hookean），对衣物这种共维流形（codimensional manifold）建模不准——会出现撕裂伪影（图4b）。各向异性模型能正确产生织物特有的褶皱和垂坠

2. **面向网格碰撞体的碰撞处理算法**:

    - 功能：有效解决衣物与SMPL-X体型网格之间的碰撞，防止穿透
    - 核心思路：原始MPM的碰撞处理基于解析level set（如球体），无法处理网格碰撞体。本文提出两阶段方法：(a) Mesh-to-Grid Transfer——将碰撞体每个face的速度和法线通过B-Spline权重传递到附近网格节点；(b) Relative Velocity Projection——在碰撞体参考系中检测穿透方向速度并投影去除。复杂度O(N_f)远低于原始level set方法的O(N_grid³)（20K vs 8M）
    - 设计动机：物理化身的碰撞体是SMPL-X body mesh而非简单几何体，必须支持任意网格碰撞体。且MPM本身的前馈速度投影方式（不同于C-IPC的迭代求解）保证了即使body mesh有轻微自穿透也不会导致仿真崩溃

3. **准阴影渲染（Quasi-Shadowing）+ 逆物理学习**:

    - 功能：提升渲染真实感并从视频中自动学习物理参数
    - 核心思路：准阴影通过神经网络预测每个Gaussian的着色标量w_p来模拟自遮挡阴影效果。物理参数学习方面，固定ν/γ/κ为默认值，通过有限差分法端到端优化杨氏模量E、密度ρ和休息几何参数α（补偿重力引起的初始变形）
    - 设计动机：物理动画中阴影对视觉真实感至关重要；休息几何参数α解决了一个实际问题——从真实视频获得的canonical mesh已经被重力拉变形了，需要反推未受力时的形状作为仿真起点

### 损失函数 / 训练策略
物理参数学习：最小化仿真mesh与跟踪mesh之间的逐顶点L2距离，使用有限差分法优化（非可微仿真）。外观学习：在所有训练帧和视角上最小化渲染图像与GT图像的光度损失，用标准3DGS优化流程。

## 实验关键数据

### 主实验

| 数据集 | 方法 | CD(×10³)↓ | F-Score↑ | LPIPS↓ | PSNR↑ | SSIM↑ |
|--------|------|----------|---------|--------|-------|-------|
| ActorsHQ | ARAH | 1.12 | 86.1 | 0.055 | 28.6 | 0.957 |
| ActorsHQ | GS-Avatar | 0.91 | 89.4 | 0.044 | 30.6 | 0.962 |
| ActorsHQ | PhysAvatar | 0.55 | 92.9 | 0.035 | 30.2 | 0.957 |
| ActorsHQ | **MPMAvatar** | **0.42** | **95.7** | **0.033** | **32.0** | **0.963** |
| 4D-DRESS | PhysAvatar | 0.37 | 96.6 | 0.022 | 33.2 | 0.976 |
| 4D-DRESS | **MPMAvatar** | **0.33** | **97.2** | **0.018** | **34.1** | **0.977** |

| 方法 | 仿真成功率(%)↑ | 单帧仿真时间(s)↓ |
|------|---------------|-----------------|
| PhysAvatar | 37.6 | 170.0 |
| **MPMAvatar** | **100.0** | **1.1** |

### 消融实验

| 配置 | CD(×10³)↓ | PSNR↑ | 说明 |
|------|----------|-------|------|
| Full (MPMAvatar) | 0.42 | 32.0 | 完整方法 |
| − Anisotropy | 6.24 | 28.7 | 去掉各向异性→几何精度暴降15倍，出现撕裂 |
| − Physics | 0.69 | 31.0 | 默认参数不学习→次优动力学 |
| − Shadow | - | 31.8 | 去掉准阴影→PSNR下降0.2 |

### 关键发现
- MPMAvatar在几何和外观上全面超越PhysAvatar，CD降低24%、PSNR提升1.8dB（ActorsHQ）
- 仿真鲁棒性是决定性优势：PhysAvatar 37.6%成功率（因body mesh自穿透导致C-IPC崩溃），MPMAvatar 100%
- 单帧仿真速度提升155倍（1.1s vs 170s）——MPM的前馈投影远快于C-IPC的迭代求解
- 各向异性本构是性能关键：去掉后CD从0.42暴增至6.24，甚至出现布料撕裂

## 亮点与洞察
- **经典物理仿真+现代神经渲染**的结合是一个高价值研究方向——物理保证泛化性，神经渲染保证真实感
- MPM相比C-IPC的核心优势在于"前馈速度投影"——不做迭代求解所以不会因碰撞检测失败而崩溃，这对实际应用（body mesh不完美是常态）至关重要
- 零样本场景交互泛化是物理仿真的独有能力——化身可以与未见过的椅子、沙子自然交互，学习型方法完全做不到
- 休息几何参数α的设计非常实用——解决了"从真实视频学物理但canonical pose已被重力变形"的鸡和蛋问题

## 局限与展望
- 不支持重光照（relighting），PhysAvatar支持
- 非衣物区域（如头发）仍用LBS驱动，可引入strand-based仿真进一步提升
- 衣物材质参数虽然通过逆物理学习，但每个化身需独立优化（非跨身份泛化）
- MPM的时间步长受限影响快速运动的仿真精度

## 相关工作与启发
- **物理仿真器选择的工程洞察**：MPM > C-IPC > X-PBD在衣物化身场景——鲁棒性和效率是最关键因素，而非理论上的精度上限
- **混合表示的价值**：mesh（物理仿真）+ Gaussian Splats（渲染）各取所长，比纯隐式或纯显式方法更适合需要同时仿真和渲染的场景
- **逆物理学习**：从视频中学物理参数是一个通用范式，可扩展到更多可变形物体

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ MPM+3DGS的首次结合，定制化各向异性本构和碰撞处理是solid贡献
- 实验充分度: ⭐⭐⭐⭐ 两个数据集+全面消融+零样本交互演示，但主体对比仅vs PhysAvatar
- 写作质量: ⭐⭐⭐⭐ 方法描述细致，图表清晰
- 价值: ⭐⭐⭐⭐⭐ 物理仿真+神经渲染的典范工作，鲁棒性和效率的巨大提升有实际应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] BecomingLit: Relightable Gaussian Avatars with Hybrid Neural Shading](becominglit_relightable_gaussian_avatars_with_hybrid_neural_shading.md)
- [\[NeurIPS 2025\] Gaussian-Augmented Physics Simulation and System Identification with Complex Colliders](gaussian-augmented_physics_simulation_and_system_identification_with_complex_col.md)
- [\[NeurIPS 2025\] DynaRend: Learning 3D Dynamics via Masked Future Rendering for Robotic Manipulation](dynarend_learning_3d_dynamics_via_masked_future_rendering_for_robotic_manipulati.md)
- [\[ICCV 2025\] TRACE: Learning 3D Gaussian Physical Dynamics from Multi-view Videos](../../ICCV2025/3d_vision/trace_learning_3d_gaussian_physical_dynamics_from_multi-view_videos.md)
- [\[ICLR 2026\] Learning Physics-Grounded 4D Dynamics with Neural Gaussian Force Fields](../../ICLR2026/3d_vision/learning_physics-grounded_4d_dynamics_with_neural_gaussian_force_fields.md)

</div>

<!-- RELATED:END -->
