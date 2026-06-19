---
title: >-
  [论文解读] A Physics-Informed Blur Learning Framework for Imaging Systems
description: >-
  [CVPR 2025][图像恢复][PSF estimation] 提出基于物理的 PSF 学习框架，设计新型波前基（每个基仅影响单一 SFR 方向）消除梯度冲突，结合课程学习（中心→边缘），无需镜头参数即可精确估计成像系统的空间变化 PSF。 领域现状：成像系统的空间非均匀像差严重影响图像质量，精确表征 PSF（点扩散函数…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "PSF estimation"
  - "wavefront aberration"
  - "curriculum learning"
  - "SFR"
  - "deblurring"
  - "imaging systems"
---

# A Physics-Informed Blur Learning Framework for Imaging Systems

**会议**: CVPR 2025  
**arXiv**: [2502.11382](https://arxiv.org/abs/2502.11382)  
**代码**: 已公开  
**领域**: 图像复原 / 计算光学  
**关键词**: PSF estimation, wavefront aberration, curriculum learning, SFR, deblurring, imaging systems

## 一句话总结
提出基于物理的 PSF 学习框架，设计新型波前基（每个基仅影响单一 SFR 方向）消除梯度冲突，结合课程学习（中心→边缘），无需镜头参数即可精确估计成像系统的空间变化 PSF。

## 研究背景与动机

**领域现状**：成像系统的空间非均匀像差严重影响图像质量，精确表征 PSF（点扩散函数）对数码摄影、工业检测、自动驾驶、天文观测等至关重要。

**现有痛点**：
   - **非参数模型**（如 Degradation Transfer）：稀疏独立采样，无法捕获 PSF 的高维特征，且不保证空间平滑性
   - **简单参数模型**（如异方差高斯、Fast Two-step）：过度简化，首次入射镜头的模糊不一定符合高斯核
   - **光学仿真模型**：需要详细镜头设计参数，受知识产权限制
   - **Seidel PSF 模型**：球差基 $\rho^2$ 同时影响多个 SFR 方向，导致优化时梯度冲突

**核心矛盾**：现有 Seidel 基将球差等项耦合了多方向 SFR，当实际 0° 和 90° 方向 SFR 不同时，优化单个系数会产生梯度冲突，类似多任务学习中的迭代干扰。

**切入角度**：将 Seidel 基分解为新型波前基，使每个基仅影响单一 SFR 方向；结合课程学习从中心到边缘逐步优化。

**核心 idea**：解耦基 + 课程学习 + MLP 代理模型 = 无需镜头参数的高精度 PSF 估计。

## 方法详解

### PSF 与 SFR 的关系
PSF 通过波前像差定义：
$$\text{PSF}(H, \lambda) = |\mathcal{F}(A(\mathbf{p}) \exp(i \frac{2\pi W(H, \lambda, \mathbf{p})}{\lambda}))|^2$$
其中 $W$ 是波前像差，$\mathbf{p} = (\rho, \theta)$ 是光瞳面坐标。SFR 是 PSF 的方向性切片。

### 关键设计

1. **新型波前基（解耦基）**

    - 功能：将 Seidel 基分解，使每个基函数仅包含 $\cos\theta$ 或 $\sin\theta$ 分量
    - 核心思路：定义索引集 $\mathcal{Q} = \{(2,2,0), (2,0,2), (3,1,0), (3,3,0), (4,2,0), (4,0,2), (5,1,0), (6,2,0), (6,0,2)\}$
    - 效果：每个基独立影响水平或竖直方向的 SFR，消除梯度冲突
    - 设计动机：类比多任务学习中的梯度冲突缓解策略

2. **课程学习优化策略**

    - 功能：从图像中心到边缘逐步学习 PSF
    - 物理依据：根据像差理论，中心仅受球差影响，越往边缘，彗差和场曲逐渐出现
    - 实现：每步限制 $H$ 在狭窄视场内优化，$H$ 从0逐渐增到1

3. **两阶段 PSF 估计**

    - **阶段一：单色 PSF 估计**
        - MLP $\mathcal{G}_{\Theta_1}$ 以归一化像高 $H$ 为输入，输出波前系数
        - 通过物理变换生成 SFR*，与实测 SFR 对比优化
        - 损失：$\Theta_1^* = \arg\min_{\Theta_1} \sum_H \sum_\phi |\text{SFR}^*(H,\phi) - \text{SFR}(H,\phi)|$
    - **阶段二：跨通道 PSF 偏移估计**
        - MLP $\mathcal{G}_{\Theta_2}$ 估计红/蓝通道相对绿通道的 PSF 平移
        - 通过色差面积差 $\Delta\text{CA}$ 作为优化目标
        - 分离单色像差和色差简化优化

4. **标定流程**

    - 受控环境下拍摄棋盘格图案
    - RAW 格式多帧平均降噪
    - 转换为线性 RGB 格式（避免 ISP 非线性影响）

### 训练策略
- 用估计的 PSF 合成模糊图像对训练去模糊网络（MPRNet/Restormer/FFTFormer）
- 训练集：Flickr2K 500 张，测试集：100 张
- 使用地面真实 PSF 合成测试模糊图像

## 实验关键数据

### PSF 精度（PSNR/SSIM，镜头 #63762）

| 方法 | H=0 无噪声 | H=0.7 无噪声 | H=1 无噪声 |
|------|-----------|-------------|-----------|
| Degradation Transfer | 41.98 / 0.937 | — | — |
| Fast Two-step | 42.24 / 0.943 | — | — |
| **Ours** | **42.08 / 0.945** | **49.19 / 0.968** | **50.16 / 0.983** |

### 去模糊性能（PSNR/SSIM，无噪声）

| 去模糊方法 | Degradation Transfer | Fast Two-step | **Ours** |
|-----------|---------------------|---------------|----------|
| MPRNet | 30.55 / 0.873 | 30.22 / 0.870 | **31.24 / 0.894** |
| Restormer | 30.69 / 0.871 | 30.34 / 0.869 | **31.51 / 0.894** |
| FFTFormer | 30.53 / 0.872 | 30.34 / 0.868 | **31.36 / 0.891** |

### 消融实验（PSNR/SSIM）

| 配置 | H=0 | H=0.7 | H=1 |
|------|------|-------|------|
| w/o 窄视场优化 | 42.51/0.934 | 42.68/0.937 | 41.06/0.922 |
| w/o 新型波前基 | 42.18/0.931 | 47.48/0.950 | 44.58/0.954 |
| w/o 课程学习 | 41.56/0.937 | 48.58/0.957 | 46.02/0.955 |
| **完整方法** | **42.64/0.940** | **49.08/0.968** | **49.25/0.981** |

### 关键发现
- 三个组件对边缘视场（H=1）提升最大：新波前基 +4.67dB，课程学习 +3.23dB
- 窄视场优化对极端位置至关重要
- 真实拍摄验证：MUSIQ/MANIQA 指标均优于对比方法

## 亮点与洞察
- **不依赖镜头参数**：可直接应用于量产成像设备
- **波前基解耦**设计优雅，类比多任务学习的梯度冲突缓解
- 两阶段分离单色和色差，降低优化难度
- 对实际成像系统（Edmund 镜头 + onsemi 传感器）验证了有效性

## 局限与展望
- 广视场下色差校正仍不完善
- 目前仅处理 PSF 引起的模糊，未考虑运动模糊
- 需要受控环境标定（棋盘格）

## 评分
- 新颖性: ⭐⭐⭐⭐ 波前基解耦+课程学习组合独特
- 实验充分度: ⭐⭐⭐⭐ 仿真+真实拍摄+消融+多去模糊算法
- 写作质量: ⭐⭐⭐⭐⭐ 物理推导清晰严谨
- 价值: ⭐⭐⭐⭐ 对量产设备图像质量提升有实用意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] PolarFree: Polarization-based Reflection-Free Imaging](polarfree_polarization-based_reflection-free_imaging.md)
- [\[ICML 2026\] Phy-CoSF: Physics-Guided Continuous Spectral Fields Reconstruction and Super-Resolution for Snapshot Compressive Imaging](../../ICML2026/image_restoration/phy-cosf_physics-guided_continuous_spectral_fields_reconstruction_and_super-reso.md)
- [\[CVPR 2025\] Proximal Algorithm Unrolling: Flexible and Efficient Reconstruction Networks for Single-Pixel Imaging](proximal_algorithm_unrolling_flexible_and_efficient_reconstruction_networks_for_.md)
- [\[ICCV 2025\] Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention](../../ICCV2025/image_restoration/consistent_time-of-flight_depth_denoising_via_graph-informed_geometric_attention.md)
- [\[AAAI 2026\] Blur-Robust Detection via Feature Restoration: An End-to-End Framework for Prior-Guided Infrared UAV Target Detection](../../AAAI2026/image_restoration/blur-robust_detection_via_feature_restoration_an_end-to-end_framework_for_prior-.md)

</div>

<!-- RELATED:END -->
