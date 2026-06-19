---
title: >-
  [论文解读] Ray-Distance Volume Rendering for Neural Scene Reconstruction
description: >-
  [ECCV 2024][3D视觉][室内场景重建] 提出 RS-Recon 方法，用射线方向相关的有符号射线距离函数（SRDF）替代传统 SDF 来参数化体渲染中的密度函数，结合 SRDF-SDF 一致性损失和自监督可见性任务，在多物体室内场景重建中取得更准确的表面和视图合成。 - 领域现状： 基于 NeRF 的神经隐式场景…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "室内场景重建"
  - "神经隐式表面"
  - "SRDF"
  - "体渲染"
  - "可见性预测"
---

# Ray-Distance Volume Rendering for Neural Scene Reconstruction

**会议**: ECCV 2024  
**arXiv**: [2408.15524](https://arxiv.org/abs/2408.15524)  
**代码**: 无  
**领域**: 3D视觉  
**关键词**: 室内场景重建, 神经隐式表面, SRDF, 体渲染, 可见性预测  

## 一句话总结

提出 RS-Recon 方法，用射线方向相关的有符号射线距离函数（SRDF）替代传统 SDF 来参数化体渲染中的密度函数，结合 SRDF-SDF 一致性损失和自监督可见性任务，在多物体室内场景重建中取得更准确的表面和视图合成。

## 研究背景与动机

- **领域现状**: 基于 NeRF 的神经隐式场景重建方法（如 VolSDF、NeuS、MonoSDF）通常用 SDF 的可学习变换来参数化体密度函数，在单物体场景中表现优异
- **现有痛点**: 在多物体室内场景中，沿相机射线采样点的 SDF 可能受到相邻物体表面的影响而产生波动，形成多个局部极小值，导致密度函数出现错误的局部极大值和高权重
- **核心矛盾**: SDF 计算的是点到**整个场景所有表面**的最短距离，但体渲染中沿射线方向，真正重要的只是射线交汇的那个表面——远处不在射线上的表面不应影响该射线的密度分布
- **本文解决什么**: 解决 SDF 在多物体场景中产生虚假密度峰值的问题，使体渲染的权重分布更准确地反映实际2D观测
- **切入角度**: 引入射线相关的 SRDF（Signed Ray Distance Function），它仅计算点到沿射线方向表面的最短距离，消除了非射线方向表面的干扰
- **核心idea**: 用 SRDF 建模密度函数实现更准确的体渲染，同时用 SDF 描述3D表面几何，通过一致性损失和可见性任务将二者耦合

## 方法详解

### 整体框架

RS-Recon 的网络包含三个 MLP 分支：
1. **几何 MLP $f_g$**: 从编码位置预测 SDF $d_\Omega$ 和几何特征 $\mathbf{F}_g$
2. **SRDF MLP $f_s$**: 从几何特征 + 视角方向 + 位置预测 SRDF $\tilde{d}_\Omega$ 和可见性概率
3. **颜色 MLP $f_c$**: 从几何特征 + 视角方向 + 法线预测颜色

体渲染使用 SRDF 推导的密度函数；表面提取使用仅依赖位置的 SDF（通过 Marching Cubes）。

### 关键设计

**模块一：SRDF 密度函数**

传统 SDF 定义为点到场景所有表面的最短距离：

$$d_\Omega(\mathbf{p}) = (-1)^{\mathbf{1}_\Omega(\mathbf{p})} \min_{\mathbf{p}^* \in \mathcal{M}} \|\mathbf{p} - \mathbf{p}^*\|_2$$

SRDF 定义为点沿射线方向到表面的最短距离：

$$\tilde{d}_\Omega(\mathbf{p}, \mathbf{r}) = (-1)^{\mathbf{1}_\Omega(\mathbf{p})} \min_{(\mathbf{p}+\rho\mathbf{r}) \in \mathcal{M}; \rho \in \mathbb{R}} |\rho|$$

SRDF 是射线相关的（view-dependent），因此仅在射线与表面的交叉点附近产生密度峰值，不受旁边物体的干扰。从 SRDF 推导的密度函数为：

$$\sigma^{\text{SRDF}}(\mathbf{p}, \mathbf{r}) = \alpha \Psi_\beta(-\tilde{d}_\Omega(\mathbf{p}, \mathbf{r}))$$

其中 $\Psi_\beta$ 是零均值、$\beta$ 尺度的 Laplace 分布 CDF，$\alpha, \beta$ 为可学习参数。

**模块二：SRDF-SDF 一致性损失**

SRDF 和 SDF 虽然定义不同，但符号含义一致（正=表面外，负=表面内）。由于两者由不同分支预测，符号一致性无法自动保证。用 sigmoid 近似符号函数实现可微约束：

$$\mathcal{L}_{con} = \frac{1}{N_r} \sum_{\mathbf{p}, \mathbf{r}} M_{con} \|\varsigma(\tilde{d}_\Omega) - \varsigma(d_\Omega)\|_2$$

$$\varsigma(d) = \text{Sigmoid}(k \cdot d), \quad M_{con} = [\tilde{d}_\Omega \cdot d_\Omega < 0]$$

仅对符号不一致的点施加惩罚。该损失的梯度有两个优势：(1) 惩罚力度随不一致程度调整；(2) sigmoid 导数在零点附近最大，对表面附近的点提供最强监督。

**模块三：自监督可见性任务**

沿射线方向，在第一个表面交叉点之前的采样点为可见，之后为遮挡。通过检测相邻采样点的 SRDF/SDF 符号变化定位第一个表面。为减少噪声，同时使用 SRDF 和 SDF 的信息：

$$V_{gt} = \begin{cases} 1, & \text{if } V^{\text{SRDF}}=1 \text{ and } V^{\text{SDF}}=1 \\ 0, & \text{if } V^{\text{SRDF}}=0 \text{ and } V^{\text{SDF}}=0 \end{cases}$$

SRDF 和 SDF 判断不一致时不参与训练。采用二值交叉熵损失 $\mathcal{L}_{vis}$ 监督可见性预测。

### 损失函数 / 训练策略

总损失函数：

$$\mathcal{L} = \mathcal{L}_c + \lambda_n \mathcal{L}_n + \lambda_d \mathcal{L}_d + \lambda_e \mathcal{L}_e + \lambda_s \mathcal{L}_s + \lambda_{con} \mathcal{L}_{con} + \lambda_{vis} \mathcal{L}_{vis}$$

包含：RGB 损失 $\mathcal{L}_c$、法线损失 $\mathcal{L}_n$、深度损失 $\mathcal{L}_d$、Eikonal 损失 $\mathcal{L}_e$（约束 SDF 梯度范数为1）、平滑损失 $\mathcal{L}_s$、一致性损失 $\mathcal{L}_{con}$、可见性损失 $\mathcal{L}_{vis}$。训练时同时渲染 SRDF 和 SDF 密度的颜色以获取 SDF 的梯度信号。

## 实验关键数据

### 主实验

**ScanNet（真实世界室内数据集）**：

| 方法 | Acc ↓ | Comp ↓ | Prec ↑ | Recall ↑ | F-score ↑ |
|------|-------|--------|--------|----------|-----------|
| MonoSDF_MLP | 0.035 | 0.048 | 0.799 | 0.681 | 0.733 |
| HelixSurf | 0.038 | 0.044 | 0.786 | 0.727 | 0.755 |
| Occ_SDF_Hybrid | 0.040 | 0.041 | 0.783 | 0.748 | 0.765 |
| **Ours_MLP** | **0.040** | **0.040** | **0.809** | **0.779** | **0.794** |

**Replica（合成室内数据集）/ Tanks and Temples（大规模真实数据集）**：

| 数据集 | 方法 | 关键指标 |
|--------|------|---------|
| Replica (MLP) | MonoSDF → Ours | F-score: 86.18 → **91.72** |
| Tanks and Temples (Grid) | MonoSDF → Ours | F-score: 6.58 → **7.73** |

**视图合成 PSNR**：

| 数据集 | MonoSDF_MLP | Occ_SDF_Hybrid | **Ours_MLP** |
|--------|------------|----------------|-------------|
| ScanNet | 26.40 | 26.98 | **27.77** |
| Replica | 34.45 | 35.50 | **36.06** |
| Tanks and Temples | 24.13 | 24.72 | **25.47** |

### 消融实验

ScanNet 上 MLP 表示的消融（F-score ↑）：

| 配置 | F-score |
|------|---------|
| (a) Baseline (MonoSDF) | 0.733 |
| (b) + SRDF 密度 | 0.745 (+1.2%) |
| (c) + SRDF-SDF 一致性损失 | 0.776 (+3.1%) |
| (d) + 可见性(仅 SDF) | 0.789 |
| (e) + 可见性(仅 SRDF) | 0.788 |
| **(f) + 可见性(SRDF+SDF)** | **0.794** (+6.1%) |

### 关键发现

1. 仅替换密度函数为 SRDF（无额外约束）即可提升 F-score 1.2%，验证了 SRDF 密度的有效性
2. SRDF-SDF 一致性损失贡献最大（+3.1%），表明符号对齐对于双分支架构至关重要
3. 可见性标签同时使用 SRDF 和 SDF 优于单独使用任一，因为互补先验可过滤噪声标签
4. 定性分析中，MonoSDF 在白墙附近产生虚假表面（因 SDF 受旁边表面影响），本方法则更准确
5. 渲染图像中，MonoSDF 因密度双峰产生不准确颜色，本方法单峰权重更精确

## 亮点与洞察

- **问题分析出色**: 用 toy example 清晰展示 SDF 在多物体场景中的密度问题，动机令人信服
- **SRDF 与 SDF 分工明确**: SRDF 负责密度建模（view-dependent），SDF 负责表面提取（view-independent），各司其职
- **自监督可见性**: 不依赖多视图几何或额外标注，利用网络自身的 SRDF/SDF 预测生成伪标签
- **通用性强**: 可以应用于 VolSDF 或 NeuS 基础的重建方法，使用 Grid 或 MLP 表示

## 局限与展望

- SRDF MLP 引入了额外的网络参数和计算开销
- 对于单物体场景，SRDF 和 SDF 差异不大，增益有限
- 可见性伪标签在训练初期（SDF/SRDF 不准时）可能引入噪声
- 可考虑在 3D Gaussian Splatting 框架中引入类似思路
- 大规模室外场景（Tanks and Temples）上改进幅度仍有提升空间

## 相关工作与启发

- **VolSDF**: 用 Laplace CDF 将 SDF 转化为密度 → 本文指出其在多物体场景中的局限
- **MonoSDF**: 利用单目深度和法线先验增强重建 → 本文以此为 baseline 进一步提升
- **VolRecon (CVPR 2023)**: 使用 SRDF 做可泛化多视图重建 → 本文将 SRDF 用于per-scene优化的体渲染密度建模
- **VIP-NeRF**: 用平面扫描体构建可见性标签 → 本文的自监督方法更轻量，不需要多视图几何计算

## 评分

- **新颖性**: ⭐⭐⭐⭐ — SRDF 密度建模在神经场景重建中是新颖的视角
- **实验充分度**: ⭐⭐⭐⭐⭐ — 三个数据集、两种表示、详细消融与定性分析
- **写作质量**: ⭐⭐⭐⭐⭐ — 问题动机通过 toy example 阐述极为清晰
- **实用价值**: ⭐⭐⭐⭐ — 作为即插即用模块适配现有 SDF 基方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] A Probability-guided Sampler for Neural Implicit Surface Rendering](a_probabilityguided_sampler_for_neural_implicit_surface_rend.md)
- [\[ECCV 2024\] Implicit Filtering for Learning Neural Signed Distance Functions from 3D Point Clouds](implicit_filtering_for_learning_neural_signed_distance_functions_from_3d_point_c.md)
- [\[ECCV 2024\] VersatileGaussian: Real-Time Neural Rendering for Versatile Tasks Using Gaussian Splatting](versatilegaussian_real-time_neural_rendering_for_versatile_tasks_using_gaussian_.md)
- [\[ECCV 2024\] Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)

</div>

<!-- RELATED:END -->
