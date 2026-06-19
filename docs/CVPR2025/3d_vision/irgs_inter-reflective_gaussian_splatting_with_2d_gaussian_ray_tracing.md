---
title: >-
  [论文解读] IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing
description: >-
  [CVPR 2025][3D视觉][逆渲染] 本文提出IRGS框架，首次在高斯泼溅中集成完整渲染方程（无简化），通过提出的可微分2D高斯光线追踪技术实时计算入射光的可见性和间接辐射，在多个逆渲染基准上取得了显著优于先前方法的重光照和材质估计效果。 1. 领域现状：逆渲染旨在从一组位姿图像中重建几何并估计材质和光照…
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "逆渲染"
  - "高斯泼溅"
  - "光线追踪"
  - "间接光照"
  - "材质估计"
---

# IRGS: Inter-Reflective Gaussian Splatting with 2D Gaussian Ray Tracing

**会议**: CVPR 2025  
**arXiv**: [2412.15867](https://arxiv.org/abs/2412.15867)  
**代码**: [https://fudan-zvg.github.io/IRGS](https://fudan-zvg.github.io/IRGS)  
**领域**: 3D视觉  
**关键词**: 逆渲染, 高斯泼溅, 光线追踪, 间接光照, 材质估计

## 一句话总结
本文提出IRGS框架，首次在高斯泼溅中集成完整渲染方程（无简化），通过提出的可微分2D高斯光线追踪技术实时计算入射光的可见性和间接辐射，在多个逆渲染基准上取得了显著优于先前方法的重光照和材质估计效果。

## 研究背景与动机

1. **领域现状**：逆渲染旨在从一组位姿图像中重建几何并估计材质和光照。3D高斯泼溅(3DGS)因其出色的渲染质量和效率成为有前途的3D场景表示方法。

2. **现有痛点**：由于缺乏高效的高斯光线追踪器，现有基于3DGS的逆渲染方法要么采用简化的渲染方程（如GS-IR使用split-sum近似），要么用可学习参数近似间接辐射（如R3DG），导致材质和光照估计不准确。

3. **核心矛盾**：渲染方程要求精确计算每个入射方向的可见性和间接辐射，而3DGS基于光栅化无法进行射线追踪，所以不得不做出各种妥协。

4. **本文目标** 如何在高效的高斯泼溅框架中，不做任何简化地完整实现渲染方程，准确捕捉inter-reflection（互反射）效果。

5. **切入角度**：受到3DGRT（3D高斯光线追踪）的启发，但3DGRT直接用于预训练的3DGS模型会有明显质量退化，因此作者选择在2D高斯原语上做光线追踪——因为2D高斯盘有明确定义的射线-平面交点。

6. **核心 idea**：在2D高斯泼溅上构建可微分光线追踪，为渲染方程的完整评估提供可见性和间接辐射的实时查询能力。

## 方法详解

### 整体框架
IRGS采用两阶段训练框架。输入为一组位姿RGB图像，输出为分解的几何、材质（albedo和roughness）、光照（环境光cubemap）。第一阶段预训练标准2DGS模型获得可靠几何；第二阶段在此基础上引入材质属性和完整渲染方程，利用2D高斯光线追踪计算入射光的可见性和间接辐射。

### 关键设计

1. **2D高斯光线追踪 (2DGRT)**:

    - 功能：在2D高斯原语上进行高效精确的光线追踪，计算任意射线的累积不透明度和颜色
    - 核心思路：使用自适应正二十面体网格作为每个2D高斯的包围代理，利用OptiX硬件加速的射线-三角形求交构建BVH。对于每个2D高斯，解析计算射线-平面交点 $\boldsymbol{p} = \boldsymbol{r}_o + \tau\boldsymbol{r}_d$，其中 $\tau = \frac{\boldsymbol{n}^\top(\boldsymbol{\mu}-\boldsymbol{r}_o)}{\boldsymbol{n}^\top\boldsymbol{r}_d}$，然后通过k-buffer排序和alpha混合得到最终结果
    - 设计动机：3DGRT在3D高斯上做光线追踪时，射线-高斯交点定义不一致（3D高斯取射线上最大响应点，与splatting的2D投影计算不同），导致直接在预训练3DGS上做光线追踪质量严重退化。而2D高斯盘的法向量明确定义了射线-平面交点，确保了由splatting到ray tracing的一致性，质量退化极小

2. **完整渲染方程的集成**:

    - 功能：在光栅化后的像素级别应用完整渲染方程，无需任何简化
    - 核心思路：将入射辐射分解为直接辐射（环境cubemap）和间接辐射。使用蒙特卡洛分层采样 $N_r=256$ 个入射方向，通过2DGRT查询每个方向的可见性 $V$ 和间接辐射 $L_{ind}$：$(L_{ind}(\omega_i, x), 1-V(\omega_i, x)) \leftarrow \text{Trace}(x, \omega_i)$。采用简化Disney BRDF模型（仅albedo和roughness），将BRDF分解为漫反射项 $f_d = a/\pi$ 和镜面反射项 $f_s$
    - 设计动机：之前方法要么用split-sum近似简化渲染方程（GS-IR），要么用可学习SH参数近似间接辐射（R3DG），无法准确建模真实的inter-reflection。本方法完整保留渲染方程，可微光线追踪允许梯度通过反向传播优化间接光

3. **高效优化方案和重光照策略**:

    - 功能：管理蒙特卡洛采样的计算需求，并在重光照时查询间接辐射
    - 核心思路：每次训练迭代设置最多 $N_{rays}=2^{18}$ 条射线，只对 $\lfloor N_{rays}/N_r \rfloor$ 个随机像素评估渲染方程，使得可以使用大量采样射线（$N_r=256$）提高估计质量。重光照时，对每条入射射线通过光线追踪alpha混合得到该方向的albedo、roughness、normal值，然后用split-sum近似和预过滤环境贴图高效查询间接辐射
    - 设计动机：完整评估密集像素的渲染方程计算量巨大（每像素256条射线），子采样像素策略在保证每像素高采样数的同时控制总计算量。重光照时环境光变化导致优化时的SH颜色不再适用，split-sum近似避免了递归评估

### 损失函数 / 训练策略
第一阶段损失：$\mathcal{L}^1 = \mathcal{L}_c + \lambda_n\mathcal{L}_n + \lambda_d\mathcal{L}_d + \lambda_{s,n}\mathcal{L}_{s,n} + \lambda_o\mathcal{L}_o$，包括RGB重建损失、法线一致性损失、深度畸变损失、边缘感知平滑损失和二元交叉熵mask损失。第二阶段增加PBR颜色L1损失 $\mathcal{L}_1^{pbr}$、白光先验正则 $\mathcal{L}_{light}$、albedo和roughness的边缘感知平滑正则。训练约40分钟（RTX 3090），第一阶段40K迭代/15min，第二阶段20K迭代/25min。

## 实验关键数据

### 主实验

| 数据集 | 指标 | IRGS | R3DG | GS-IR | TensoIR |
|--------|------|------|------|-------|---------|
| Synthetic4Relight | Relight PSNR↑ | **34.90** | 31.00 | 25.40 | 29.69 |
| Synthetic4Relight | Albedo PSNR↑ | **30.81** | 28.31 | 19.48 | 30.58 |
| Synthetic4Relight | Roughness MSE↓ | **0.008** | 0.013 | 0.011 | 0.015 |
| TensoIR | Relight PSNR↑ | **29.907** | 27.367 | 24.374 | 28.580 |
| TensoIR | Albedo PSNR↑ | **33.796** | 26.199 | 30.286 | 29.275 |
| TensoIR | Normal MAE↓ | 4.112 | 5.927 | 4.948 | **4.100** |

### 消融实验

| 配置 | NVS PSNR | Albedo PSNR | Relight PSNR | 说明 |
|------|----------|-------------|--------------|------|
| Full model | 35.48 | 30.81 | 34.68 | 完整模型 |
| Detach indirect | 34.21 | 30.29 | 34.22 | 不反传间接光梯度 |
| w/o indirect (train) | 34.09 | 30.10 | 33.93 | 训练时去掉间接光项 |
| w/o indirect (relight) | - | - | 33.84 | 重光照时无间接光 |
| $N_r=16$ | 34.01 | 30.21 | 29.46 | 采样射线太少，重光照掉5.2dB |
| $N_r=64$ | 34.98 | 30.63 | 33.11 | 64条射线，重光照仍差1.6dB |

### 关键发现
- 采样射线数量对重光照质量至关重要：$N_r$ 从16增加到256，relight PSNR提升超过5dB
- 可微间接光是关键：detach间接光梯度导致不真实的间接辐射估计
- IRGS的训练时间（0.7h）与R3DG（0.9h）相当，远快于NeRF-based方法（3-48h）
- R3DG在NVS上表现更好（36.80 vs 35.48），但这是因为它在Gaussian上做shading，以牺牲重光照性能为代价

## 亮点与洞察
- **2D vs 3D高斯光线追踪**：2D高斯有明确的射线-平面交点，允许直接在预训练2DGS上做光线追踪，质量退化极小。这个insight说明了表示一致性对混合渲染管线的重要性
- **像素级子采样+高采样数的trade-off**：通过只对随机子集像素评估渲染方程，用256条射线获得高质量估计，而不是对所有像素用少量射线——这个策略可以迁移到其他计算密集型渲染任务
- **完整渲染方程的必要性**：实验证明简化假设（split-sum、可学习SH）确实会限制材质/光照估计精度，完整渲染方程虽然计算量大但可以通过工程优化解决

## 局限与展望
- 无法实时渲染（每帧约1秒），限制了实际应用
- 作者提到可以通过bake间接辐射或预计算辐射传输来加速，但未实现
- 假设材质为电介质（dielectric），不支持金属材质的精确建模
- 仅在合成场景和简单真实场景上验证，对复杂真实室内场景的适用性存疑

## 相关工作与启发
- **vs GS-IR**: GS-IR使用split-sum近似+烘焙体积存储遮挡，本文直接用光线追踪查询。IRGS在重光照上大幅领先（34.90 vs 25.40 PSNR），证明完整渲染方程的优势
- **vs R3DG**: R3DG在每个Gaussian上独立着色并用SH参数化间接辐射，本文在像素级做着色并实时追踪间接光。R3DG在NVS上更好但重光照差距明显
- **vs 3DGRT**: 3DGRT提出了高斯光线追踪框架，但在3D高斯上存在射线-平面交点不一致问题。本文改用2D高斯解决了这一关键问题

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将完整渲染方程无简化集成到高斯泼溅，2DGRT是有意义的技术贡献
- 实验充分度: ⭐⭐⭐⭐ 三个数据集+详细消融，但缺少大规模真实场景验证
- 写作质量: ⭐⭐⭐⭐ 公式清晰，动机阐述逻辑性强
- 价值: ⭐⭐⭐⭐ 为高斯泼溅的逆渲染提供了正确的技术路线（完整渲染方程+光线追踪），有望推动后续工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Hardware-Rasterized Ray-Based Gaussian Splatting](hardware-rasterized_ray-based_gaussian_splatting.md)
- [\[CVPR 2026\] Stochastic Ray Tracing for the Reconstruction of 3D Gaussian Splatting](../../CVPR2026/3d_vision/stochastic_ray_tracing_for_the_reconstruction_of_3d_gaussian_splatting.md)
- [\[CVPR 2026\] Geometric-Photometric Event-based 3D Gaussian Ray Tracing](../../CVPR2026/3d_vision/geometric-photometric_event-based_3d_gaussian_ray_tracing.md)
- [\[CVPR 2025\] Ref-GS: Directional Factorization for 2D Gaussian Splatting](ref-gs_directional_factorization_for_2d_gaussian_splatting.md)
- [\[CVPR 2025\] DepthSplat: Connecting Gaussian Splatting and Depth](depthsplat_connecting_gaussian_splatting_and_depth.md)

</div>

<!-- RELATED:END -->
