---
title: >-
  [论文解读] CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images
description: >-
  [ECCV 2024][3D视觉][可驱动3D人体] 提出CanonicalFusion框架,通过联合预测深度图和压缩LBS权重映射图实现直接规范化,并利用前向蒙皮可微渲染融合多张图像信息,从多张输入图像生成可驱动的3D人体Avatar。 从图像生成3D人体Avatar是元宇宙和AR/VR的关键技术。现有方法的局限: 隐式方…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "可驱动3D人体"
  - "规范空间融合"
  - "前向蒙皮"
  - "可微渲染"
  - "LBS权重压缩"
---

# CanonicalFusion: Generating Drivable 3D Human Avatars from Multiple Images

**会议**: ECCV 2024  
**arXiv**: [2407.04345](https://arxiv.org/abs/2407.04345)  
**代码**: [有](https://github.com/jsshin98/CanonicalFusion)  
**领域**: 3D视觉  
**关键词**: 可驱动3D人体, 规范空间融合, 前向蒙皮, 可微渲染, LBS权重压缩

## 一句话总结

提出CanonicalFusion框架,通过联合预测深度图和压缩LBS权重映射图实现直接规范化,并利用前向蒙皮可微渲染融合多张图像信息,从多张输入图像生成可驱动的3D人体Avatar。

## 研究背景与动机

从图像生成3D人体Avatar是元宇宙和AR/VR的关键技术。现有方法的局限:

**隐式方法**(PIFu系列): 分辨率受限于体素空间

**显式方法**(三明治式深度图): 多视图融合困难

**模板驱动方法**(ARCH系列): 远离模板表面的点初始化不准确

CanonicalFusion的核心想法: **在规范空间(canonical space)中融合多视图重建结果**。

## 方法详解

### 整体框架

两步流程:
1. **初始网格预测**: 共享编码器-双解码器网络预测双面深度图和压缩LBS权重图,直接规范化得到初始网格
2. **前向蒙皮可微渲染优化**: 规范网格通过前向蒙皮变形 -> 可微光栅化渲染 -> 最小化几何和光度误差

### 关键设计

#### 1. LBS权重的紧凑表示

用堆叠自编码器MLP将55维蒙皮权重压缩到3维潜空间。训练数据约800K个样本(SMPL-X UV坐标插值)。损失: L1 + 非零元素损失(高斯基函数近似) + KL散度。最后一层softmax保证权重和为1。预训练后推理只用解码器。

#### 2. 联合深度和LBS预测

ATUNet架构,共享编码器+深度解码器+LBS解码器。输入RGB+SMPL-X深度图,输出前后面的深度图和3维压缩LBS权重图。额外UNet纹理预测网络输出去阴影的颜色图。

#### 3. 规范网格重建

从LBS权重直接逆蒙皮到规范空间。不可见区域(腋下、大腿内侧)通过SDF集成填补: 结合重建网格和SMPL-X模板网格的有符号距离,根据点是否在重建网格附近选择SDF。Marching Cubes提取后用Flexicubes转化为可微紧凑网格。

#### 4. 前向蒙皮可微渲染优化

将规范网格通过前向蒙皮变形到各输入姿态,用NDS光栅化渲染,同时优化规范网格顶点位置/颜色和3D姿态参数。渐进式: 先优化姿态,再固定姿态优化形状和颜色。每500次迭代上采样4倍,总计2000迭代。

### 损失函数 / 训练策略

优化目标: Laplacian平滑 + 法线一致性正则 + 法线图L1 + 掩码MSE + Chamfer距离 + 颜色L2。4xRTX 3090训练2天,推理约11分钟。

## 实验关键数据

### 主实验: 单目人体重建对比

| 方法 | 训练数据 | RP P2S↓ | RP CF↓ | TH3.0 P2S↓ | TH3.0 CF↓ |
|------|----------|---------|--------|------------|-----------|
| PIFuHD | RP | 1.420 | 1.434 | 1.534 | 1.527 |
| ICON* | RP | 1.296 | 1.364 | 1.371 | 1.437 |
| 2K2K* | TH2.0+RP | 1.097 | 1.195 | 1.416 | 1.542 |
| TeCH* | N/A | 1.489 | 1.523 | 1.721 | 1.795 |
| **Ours** | **TH2.0+RP** | **0.886** | **0.943** | **1.072** | **1.165** |

(P2S=点到面距离cm, CF=Chamfer距离cm, *=使用GT SMPL-X)

### 与SCANimate对比(规范空间精度)

| 方法 | 视角数 | SET1 P2S↓ | SET2 P2S↓ |
|------|--------|-----------|-----------|
| SCANimate | 5 | 1.362 | 1.076 |
| SCANimate | 15 | 1.103 | 0.997 |
| **Ours** | **5** | **0.244** | **0.180** |
| **Ours** | **15** | **0.199** | **0.149** |

P2S误差仅为SCANimate的约1/5。

### 消融实验

- **姿态误差修正**: 同时优化姿态参数有效纠正初始网格的手臂弯曲等误差
- **宽松衣物**: 从初始规范网格出发(非模板),拓扑结构更接近目标,可恢复宽松衣物
- **多帧融合**: 多帧显著提高模型完整性
- **真实场景**: Actors-HQ和野外拍摄均可生成逼真Avatar

### 关键发现

1. 显式深度预测+SMPL-X引导仍是有效方法,无需复杂隐式技术
2. 多样化数据集一致提升性能
3. 前向蒙皮优于逆蒙皮(SCANimate在姿态误差下严重退化)
4. LBS权重压缩到3维几乎无精度损失

## 亮点与洞察

1. **3维LBS压缩**: 将55维稀疏蒙皮权重压缩到3维,既降低预测难度又可可视化
2. **规范空间融合**: 避免观测空间多视图融合的几何对齐困难
3. **姿态-形状联合优化**: 缓解姿态估计误差的级联影响
4. **SDF集成填补空洞**: 巧妙结合重建网格和模板网格
5. **任意数量输入**: 1张到数十张均可

## 局限与展望

1. **非刚性衣物变形**: 帧间变形过大可能产生模糊
2. **手部细节**: 需借助外部手部替换模块
3. **依赖SMPL-X估计**: 初始深度图依赖SMPL-X参数质量
4. 未来: 处理头发和衣物非刚性变形,结合生成式技术

## 相关工作与启发

- 与PIFu系列相比,显式深度预测允许更高分辨率处理
- SCANimate的循环一致性被规范空间融合继承并改进
- SNARF的前向蒙皮场启发了本文的前向蒙皮可微渲染

## 评分

- 新颖性: ⭐⭐⭐⭐ — LBS压缩和规范空间融合策略新颖实用
- 实用性: ⭐⭐⭐⭐ — 可直接驱动,已开源,支持任意数量输入
- 实验充分度: ⭐⭐⭐⭐ — 多数据集/多方法对比+丰富消融
- 写作质量: ⭐⭐⭐⭐ — 流程清晰,配图丰富

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Divide and Fuse: Body Part Mesh Recovery from Partially Visible Human Images](divide_and_fuse_body_part_mesh_recovery_from_partially_visible_human_images.md)
- [\[ECCV 2024\] Differentiable Convex Polyhedra Optimization from Multi-view Images](differentiable_convex_polyhedra_optimization_from_multi-view_images.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images](../../ICCV2025/3d_vision/auto-regressively_generating_multi-view_consistent_images.md)
- [\[CVPR 2025\] HOI3DGen: Generating High-Quality Human-Object-Interactions in 3D](../../CVPR2025/3d_vision/hoi3dgen_generating_high-quality_human-object-interactions_in_3d.md)
- [\[ECCV 2024\] HeadGaS: Real-Time Animatable Head Avatars via 3D Gaussian Splatting](headgas_real-time_animatable_head_avatars_via_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
