---
title: >-
  [论文解读] Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation
description: >-
  [ICCV2025][3D视觉][dual-pixel] Sdirt 提出基于光线追踪的双像素（DP）图像模拟方案，通过精确计算包含像差和相位分裂信息的空间变化 DP PSF，弥合仿真与真实 DP 数据之间的域间差距，使深度估计模型在真实 DP 图像上具有更好的泛化能力。 - 问题定义：双像素（DP）传感器将每个像素分为左右…
tags:
  - "ICCV2025"
  - "3D视觉"
  - "dual-pixel"
  - "深度估计"
  - "ray tracing"
  - "PSF simulation"
  - "domain gap"
---

# Simulating Dual-Pixel Images From Ray Tracing For Depth Estimation

**会议**: ICCV2025  
**arXiv**: [2503.11213](https://arxiv.org/abs/2503.11213)  
**代码**: [GitHub](https://github.com/LinYark/Sdirt)  
**领域**: 3D视觉  
**关键词**: dual-pixel, depth estimation, ray tracing, PSF simulation, domain gap  

## 一句话总结

Sdirt 提出基于光线追踪的双像素（DP）图像模拟方案，通过精确计算包含像差和相位分裂信息的空间变化 DP PSF，弥合仿真与真实 DP 数据之间的域间差距，使深度估计模型在真实 DP 图像上具有更好的泛化能力。

## 研究背景与动机

- **问题定义**：双像素（DP）传感器将每个像素分为左右子像素，利用微透镜实现相位分裂，一次拍摄获取一对 DP 图像。DP 图像可用于深度估计（DfDP），但 DP-depth 配对数据极其稀缺
- **现有方法局限**：
    - **标定式模拟器**（Xin et al., Li et al.）：需大量时间标定真实相机，存在离散标定点的插值误差，且难以迁移到其他镜头
    - **模型式模拟器**（DDDNet, Pan et al., Punnappurath et al.）：使用理想光学模型直接计算 DP PSF，但忽略了透镜像差和传感器相位分裂特性
    - 如图所示，理想薄透镜模型的 CoC 仿真 DP PSF 与真实 PSF 存在显著域间差距
- **核心问题**：现有模型式模拟器违反了真实光学传播规律，导致仿真 DP 图像训练的模型难以泛化到真实 DP 数据
- **动机**：利用光线追踪准确模拟包含像差和相位信息的 DP PSF，从根本上缩小仿真与真实的域间差距

## 方法详解

### 整体框架

Sdirt 由三个模块组成：
1. **光线追踪 DP PSF 模拟器**：计算空间变化的 DP PSF
2. **DP PSF 预测网络**：MLP 预测每个像素的 DP PSF（减少计算开销）
3. **逐像素 DP 图像渲染模块**：卷积生成仿真 DP 图像

### 关键设计一：光线追踪 DP PSF 模拟器

**透镜光线追踪**：从物点 $p$ 出发，在入瞳上密集采样 $n$ 条光线。光线穿过每个透镜表面时，按 Snell 定律和透镜参数更新位置和方向，最终到达传感器面，得到着陆点 $O$ 和方向 $D$。

**DP 传感器光线追踪**：简化 DP 像素结构，将微透镜建模为薄透镜（半径 $r$、焦距 $f$），定义子像素宽度 $w$、距微透镜距离 $h$、像素尺寸 $ps$。

**分两种情况判断光线进入哪个子像素**：

当光线着陆在微透镜内时，经微透镜折射后进入子像素，通过边界线判断：

$$x_{L1} = x_i + w - (f \cdot \tan\theta - w) \cdot h / (f - h)$$
$$x_{M1} = x_i - (f \cdot \tan\theta) \cdot h / (f - h)$$
$$x_{R1} = x_i - w - (f \cdot \tan\theta + w) \cdot h / (f - h)$$

当光线着陆在微透镜外时，直接进入子像素：

$$x_{L2} = x_i + w - h \cdot \tan\theta, \quad x_{M2} = x_i - h \cdot \tan\theta$$

**左 PSF 计算**（对所有 $n$ 条光线在各左子像素上的能量分布积分）：

$$PSF_L(i,j) = \sum_{k=1}^{n} A_k \cdot \delta_{L,k}(i,j)$$

### 关键设计二：MLP PSF 预测网络

为降低光线追踪的计算成本，训练一个 MLP 预测 DP PSF：
- 输入：有效成像区域内的归一化坐标 $p$（视场和传感器定义的锥台空间）
- 网络：5 个隐藏层 × 512 神经元 + 输出层 $2 \times ks^2$ 神经元
- 训练损失：

$$Loss = L_2(\widehat{PSF_L}, PSF_L) + L_2(\widehat{PSF_R}, PSF_R)$$

- 训练时使用最大值归一化，推理时使用求和归一化（近似相机的渐晕补偿）

### 关键设计三：逐像素 DP 图像渲染

将深度图中每个像素视为物点，用训练好的 MLP 预测其 DP PSF，然后与全聚焦 RGB 图像做逐像素局部卷积，渲染出包含像差和相位信息的仿真 DP 图像。

### DfDP 模型适配

选用 AANet 作为 DfDP 模型，关键调整：在代价体积生成步骤中增加反向视差（蓝色箭头），因为 DP 图像中焦距前后物点的视差方向相反，不同于双目图像的单向视差。

训练损失：$Loss = L_1(\hat{I_D}, I_D)$

## 实验关键数据

### 主实验：DP PSF 模拟精度

| 方法 | NCC ↑ | NSD ↓ |
|------|-------|-------|
| DDDNet | 0.589 | 0.625 |
| L2R | 0.638 | 0.523 |
| CoC | 0.672 | 0.448 |
| Modeling | 0.707 | 0.423 |
| **Sdirt** | **0.915** | **0.133** |

NCC（归一化互相关）衡量相似度，NSD（归一化平方差）衡量误差。Sdirt 的 PSF 相似度远超所有基线。

### 深度估计结果（DP119 测试集）

| 场景 | 方法 | MAE ↓ | MSE ↓ | Acc-1 ↑ | Acc-2 ↑ |
|------|------|-------|-------|---------|---------|
| Planar | **Sdirt** | **0.0845** | **0.0109** | **0.9849** | **0.9997** |
| Planar | CoC | 0.2085 | 0.1001 | 0.6670 | 0.8990 |
| Box | **Sdirt** | **0.1197** | **0.0339** | **0.9474** | **0.9812** |
| Box | CoC | 0.3375 | 0.1804 | 0.4412 | 0.8277 |
| Casual | **Sdirt** | **0.2702** | **0.2294** | **0.8236** | **0.9314** |
| Casual | CoC | 0.7925 | 1.8579 | 0.3318 | 0.6103 |

### 消融/关键发现

1. **PSF 精度差距巨大**：Sdirt NCC=0.915 vs 次优 Modeling=0.707，其他方法因忽略像差和相位分裂产生显著域间差距
2. **深度估计泛化能力**：Planar 场景 Acc-1 从 0.6670（CoC）提升到 0.9849（Sdirt），差距约 30 个百分点
3. **Casual 场景鲁棒性**：即使有无纹理区域干扰，Sdirt 仍达 0.8236 Acc-1，远超次优 0.3318
4. **关键物理现象**：离轴越远，真实 PSF 的相位越不对称、像差越大，现有模拟器完全忽略这些特性
5. 仿真 DP 图像的定量评估（PSNR/SSIM）：Sdirt 在所有深度上均最优，平均 PSNR=37.20/SSIM=0.9845

## 亮点与洞察

1. **物理建模的回归**：在深度学习时代，通过回归物理光学原理（光线追踪）而非简单理想化模型来解决域间差距问题
2. **MLP 加速光线追踪的优雅方案**：用 MLP 近似光线追踪结果，训练阶段离线计算 GT，推理阶段快速预测
3. **新测试集 DP119**：119 个场景（平面/盒子/日常），已知镜头结构+固定对焦距离，填补了评估空白
4. **代价体积的物理适配**：增加反向视差捕捉 DP 图像特有的前后焦视差方向变化，简单但关键
5. **域间差距分析深入**：不仅讨论 PSF 层面的差距，还系统分析了图像层面和深度估计层面的传播效应

## 局限性

1. 仅适用于已知镜头结构参数的定焦镜头+DP 传感器的相机，目前仅 Canon 系列满足
2. 相机厂商不公开 DP 像素结构参数，需自行简化建模
3. 假设色差已校正（使用 550nm 单波长），对色差显著的系统可能不准
4. F/1.8 时因散焦核过大导致 GPU 内存不足，实验限制在 F/4
5. 数据集（DP119）规模相对较小，场景多样性有限

## 相关工作与启发

- **DP 模拟器的演进**：标定式（精确但耗时）→ 模型式（快速但不准）→ **光线追踪式**（本文，准确且可迁移）
- **sim-to-real 域间差距**：本文在 DP 成像领域系统研究仿真与真实的差距，方法论可推广到其他计算成像任务
- **MLP 作为物理场近似器**：与 NeRF 的思路一致，用简单 MLP 近似空间变化的物理量（这里是 PSF）
- **DP 数据的应用前景**：不限于深度估计，还可用于去模糊、重对焦、雨滴去除、反射去除等

## 评分 ⭐⭐⭐⭐

问题定义明确（DP 仿真域间差距），方法基于严格的物理光学建模，实验设置和评估指标设计严谨。深度估计性能提升显著。新数据集 DP119 有独立贡献。局限在于适用范围窄（需要已知镜头结构的 Canon 相机），但在该限定条件下工作扎实完整。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Radiant Foam: Real-Time Differentiable Ray Tracing](radiant_foam_real-time_differentiable_ray_tracing.md)
- [\[ICCV 2025\] One Look is Enough: Seamless Patchwise Refinement for Zero-Shot Monocular Depth Estimation on High-Resolution Images](one_look_is_enough_seamless_patchwise_refinement_for_zero-shot_monocular_depth_e.md)
- [\[CVPR 2026\] Lens Component Deletion based on Differentiable Ray Tracing](../../CVPR2026/3d_vision/lens_component_deletion_based_on_differentiable_ray_tracing.md)
- [\[CVPR 2025\] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing](../../CVPR2025/3d_vision/irgs_inter-reflective_gaussian_splatting_with_2d_gaussian_ray_tracing.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)

</div>

<!-- RELATED:END -->
