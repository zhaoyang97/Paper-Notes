---
title: >-
  [论文解读] MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video
description: >-
  [AAAI 2026][3D视觉][3D Gaussian Splatting] MoBGS 提出了一种端到端的动态去模糊 3D Gaussian Splatting 框架，通过 Blur-adaptive Latent Camera Estimation (BLCE) 和 Latent Camera-induced Exposure Estimation (LCEE) 两个核心模块，从模糊单目视频中重建清晰的时空新视角，在 Stereo Blur 数据集上大幅超越现有 SOTA 方法。
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "motion deblurring"
  - "dynamic novel view synthesis"
  - "Neural ODE"
  - "monocular video"
---

# MoBGS: Motion Deblurring Dynamic 3D Gaussian Splatting for Blurry Monocular Video

**会议**: AAAI 2026  
**arXiv**: [2504.15122](https://arxiv.org/abs/2504.15122)  
**代码**: 待确认  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, motion deblurring, dynamic novel view synthesis, Neural ODE, monocular video  

## 一句话总结

MoBGS 提出了一种端到端的动态去模糊 3D Gaussian Splatting 框架，通过 Blur-adaptive Latent Camera Estimation (BLCE) 和 Latent Camera-induced Exposure Estimation (LCEE) 两个核心模块，从模糊单目视频中重建清晰的时空新视角，在 Stereo Blur 数据集上大幅超越现有 SOTA 方法。

## 背景与动机

Novel View Synthesis (NVS) 近年来取得了显著进展，广泛应用于 VR、AR 和影视制作等领域。然而，现有的动态 NVS 方法（如 D3DGS、4DGS、SplineGS 等）高度依赖输入 2D 观测的质量。在日常随手拍摄的视频中，由于快速运动的物体或相机抖动，运动模糊 (motion blur) 频繁出现，导致 NVS 方法的渲染质量严重退化。这是因为 NVS 依赖精确的场景几何和外观重建，而模糊造成的锐利细节缺失会被"烘焙"进 3D 表征中，产生模糊或振铃伪影。

此前已有一些去模糊 NVS 方法被提出，如 DeblurNeRF、BAD-NeRF、BAD-GS 等，它们通过估计曝光期间的相机轨迹来建模模糊过程。但这些方法主要关注**静态场景**的去模糊，缺乏对动态物体运动的专门建模。少数面向动态场景的方法（DyBluRF、MoBluRF、Deblur4DGS）虽然开始处理全局相机运动和局部物体运动的联合模糊问题，但仍存在两个关键局限：(1) latent camera pose 估计缺乏来自输入模糊程度的引导；(2) 曝光时间估计未考虑全局相机运动与局部物体运动模糊的同步发生关系。

## 核心问题

如何从模糊单目视频中，同时精确建模全局相机运动模糊和局部动态物体运动模糊，实现端到端的清晰动态新视角合成？

## 方法详解

### 整体框架

MoBGS 构建在 SplineGS 之上，将场景表示为静态 3D Gaussians $\{G^{\text{st}}_i\}_{i=1}^{n^{\text{st}}}$ 和动态 3D Gaussians $\{G^{\text{dy}}_i\}_{i=1}^{n^{\text{dy}}}$，后者通过样条曲线建模平滑的运动轨迹。给定 $N_f$ 帧模糊单目视频 $\{\bm{B}_t\}_{t=1}^{N_f}$ 及其相机位姿 $\{\bm{\mathcal{P}}_t\}$，MoBGS 通过两个核心模块解耦全局相机运动和局部物体运动模糊：(1) BLCE 估计 latent camera poses；(2) LCEE 估计 latent exposure time。最终将 $N_l$ 个 latent sharp frame 取平均来近似模糊帧：

$$\hat{\bm{B}}_t = \frac{1}{N_l} \sum_{k=1}^{N_l} \hat{\bm{C}}_{\hat{\tau}_t^{(k)}, \hat{\bm{\mathcal{P}}}_t^{(k)}}$$

### 关键设计一：Blur-adaptive Latent Camera Estimation (BLCE)

BLCE 的核心思想是利用输入帧的模糊强度作为 prior 来引导 latent camera pose 的估计。

**Blur Score 计算**：利用模糊帧在频域中低频分量占比更高的观察，对输入帧 $\bm{B}_t$ 进行 2D DFT，定义 blur score 为：

$$\beta_t = \frac{\sum_{\xi \in \Lambda} M_t(\xi)}{\sum_{\xi} M_t(\xi)}, \quad M_t = |\tilde{\mathcal{F}}(\bm{B}_t)|$$

其中 $\Lambda$ 是边长为 $s$ 的中心裁剪正方形区域（实验中 $s=20$）。

**Blur Feature 提取**：通过浅层 MLP $F_\theta$ 和 positional encoding $\phi(\cdot)$ 提取 blur feature：$\bm{\Phi}_t = F_\theta(\phi(\beta_t))$。

**Blur-adaptive Neural ODE**：与现有方法（SMURF、CRiM-GS）使用的普通 Neural ODE 不同，MoBGS 将 blur feature $\bm{\Phi}_t$ 注入到 ODE 求解器中，使其能够根据每帧的模糊强度自适应调整。初始 latent feature 由相机位姿编码得到 $\textbf{z}_t(u_0) = F_{\theta_{\text{enc}}}(\bm{\mathcal{P}}_t)$，然后通过积分得到一系列 latent vectors：

$$\textbf{z}_t(u_k) = \textbf{z}_t(u_0) + \int_{u_0}^{u_k} f(\textbf{z}_t(u), u, \bm{\Phi}_t; \psi) \, du$$

每个 $\textbf{z}_t(u_k)$ 通过 decoder 预测 screw axis $(\bm{\omega}_t^{(k)}; \bm{v}_t^{(k)}) \in \mathbb{R}^6$，最终通过残差变换得到 latent camera pose $\hat{\bm{\mathcal{P}}}_t^{(k)} = \bm{\mathcal{P}}_t \Psi(\bm{\omega}_t^{(k)}, \bm{v}_t^{(k)})$。

### 关键设计二：Latent Camera-induced Exposure Estimation (LCEE)

LCEE 的核心洞察是：全局相机运动和局部物体运动的模糊发生在**同一曝光时间区间**内。因此，可以利用 BLCE 估计的 latent camera poses 来推断曝光时长。

具体地，LCEE 通过比较两个区间内静态 3D Gaussian 均值 $\bm{\mu}_i^{\text{st}}$ 在图像平面上的 2D 位移比值来估计 latent exposure time $\hat{\mathcal{T}}_t$：

$$\hat{\mathcal{T}}_t = \frac{2}{n^{\text{st}}} \sum_{i=1}^{n^{\text{st}}} \frac{D(\hat{\bm{\mathcal{P}}}_t^{(1)}, \hat{\bm{\mathcal{P}}}_t^{(N_l)}, \bm{\mu}_i^{\text{st}}) + \epsilon}{D(\bm{\mathcal{P}}_{t-1}, \bm{\mathcal{P}}_{t+1}, \bm{\mu}_i^{\text{st}}) + \epsilon}$$

其中 $D(\cdot)$ 计算静态 Gaussian 在两个相机位姿下的投影 2D 位移。该方法将曝光时间的估计与相机运动幅度直接关联，无需额外的可学习参数或手动调参。

### 优化目标

$$\mathcal{L}_{\text{total}} = \lambda_{\text{rgb}} \mathcal{L}_{\text{rgb}} + \lambda_{\text{depth}} \mathcal{L}_{\text{depth}}$$

其中 $\mathcal{L}_{\text{rgb}}$ 是输入模糊帧与渲染模糊帧之间的 L1 loss，$\mathcal{L}_{\text{depth}}$ 是渲染深度与 GT 深度的 L1 loss。$\lambda_{\text{rgb}}=1.0$，$\lambda_{\text{depth}}=0.2$。

## 实验关键数据

### Stereo Blur 数据集上的动态去模糊 NVS 结果（Full Image）

| 方法 | LPIPS↓ | MUSIQ↑ | tOF↓ | PSNR↑ | FPS |
|------|--------|--------|------|-------|-----|
| SplineGS | 0.141 | 42.88 | 1.409 | 26.41 | 300 |
| GShiftNet + SplineGS | 0.074 | 55.29 | 0.748 | 26.54 | 329 |
| DyBluRF | 0.079 | 50.82 | 0.889 | 25.62 | 0.2 |
| MoBluRF | 0.078 | 51.84 | 0.816 | 25.69 | 0.1 |
| **MoBGS (Ours)** | **0.050** | **57.64** | **0.507** | **28.80** | **480** |

### Dynamic Region 结果

| 方法 | LPIPS↓ | tOF↓ | PSNR↑ |
|------|--------|------|-------|
| SplineGS | 0.168 | 1.417 | 22.31 |
| DyBluRF | 0.158 | 1.367 | 19.41 |
| MoBluRF | 0.155 | 1.456 | 20.63 |
| **MoBGS (Ours)** | **0.096** | **1.093** | **23.41** |

### LCEE 消融实验（Dynamic Region）

| 曝光时间设置 | LPIPS↓ | tOF↓ | PSNR↑ |
|------------|--------|------|-------|
| Fixed $\hat{\mathcal{T}}_t=0.0$ | 0.120 | 1.237 | 23.20 |
| Fixed $\hat{\mathcal{T}}_t=0.5$ | 0.117 | 1.276 | 23.12 |
| Learnable $\hat{\mathcal{T}}_t$ | 0.128 | 1.261 | 23.24 |
| **LCEE (Ours)** | **0.096** | **1.093** | **23.41** |

### $N_l$ 消融

| $N_l$ | LPIPS↓ | MUSIQ↑ | tOF↓ | PSNR↑ | 训练时间 |
|-------|--------|--------|------|-------|---------|
| 3 | 0.069 | 53.66 | 0.594 | 28.79 | 0.8h |
| 5 | 0.055 | 56.48 | 0.526 | 28.78 | 1.0h |
| **9 (Ours)** | **0.050** | **57.64** | **0.507** | **28.80** | **1.5h** |

## 亮点

- **模糊自适应的 Neural ODE**：将 blur score/feature 注入 ODE 求解器，使 latent camera pose 估计能根据每帧模糊程度自适应调整，是对现有 Neural ODE 方法的关键改进
- **优雅的曝光时间估计**：LCEE 方法利用全局相机运动与局部物体运动共享曝光区间的物理先验，通过 2D 投影位移比值估计曝光时长，无需额外可学习参数
- **全面的性能提升**：在 Stereo Blur 数据集上，MoBGS 在 LPIPS 上达到 0.050（比 MoBluRF 的 0.078 提升 36%），PSNR 达到 28.80dB（提升 3.1dB），同时渲染速度达到 ~480 FPS（比 MoBluRF 快 4800×）

## 局限与展望

- **Per-scene 优化范式**：每个新场景都需要重新从头训练，无法泛化到未见过的场景。作者指出，未来可将去模糊模块集成到 feed-forward 可泛化 3DGS 方法中
- **深度监督依赖**：依赖预训练的单目深度估计模型（UniDepth）提供深度 GT，深度质量可能影响重建精度
- **数据集规模有限**：Stereo Blur 数据集仅包含 6 个场景，且 DAVIS 数据集无 NVS GT，评估的全面性有待加强

## 与相关工作的对比

与静态去模糊 NVS 方法（BAD-GS、Deblurring 3DGS）相比，MoBGS 显著优于它们，因为后者无法处理动态场景中的物体运动模糊。与 cascade 方法（先用 2D 去模糊网络预处理再做 NVS）相比，MoBGS 的端到端方式在 PSNR 上有 1.3-2dB 的优势，说明联合优化 3D 重建与去模糊优于分离式处理。与最近的动态去模糊 NVS 方法相比：DyBluRF 使用固定曝光时间且基于 NeRF，渲染极慢（0.2 FPS）；MoBluRF 虽然分解了全局/局部模糊但忽略曝光时间估计，同样基于 NeRF（0.1 FPS）；Deblur4DGS 虽基于 3DGS 但其可学习曝光时间缺乏约束，导致去模糊不一致。MoBGS 在所有指标上均显著领先，同时实现了实时渲染速度。

## 启发与关联

- **频域先验的利用**：blur score 的设计利用了模糊图像低频分量占比高的频域特性，这一思路可推广到其他图像退化感知任务中
- **ODE 求解器的条件化**：将外部先验注入 Neural ODE 的思路（Blur-adaptive Neural ODE）具有通用性，可应用于其他需要条件化连续动力学建模的场景
- **全局-局部运动一致性约束**：LCEE 中全局相机运动与局部物体运动共享曝光区间的约束思想，可启发其他需要跨尺度运动一致性建模的任务（如视频稳定、运动分割等）

## 评分

- 新颖性: ⭐⭐⭐⭐ — BLCE 和 LCEE 两个模块设计巧妙，blur-adaptive Neural ODE 和基于投影位移比的曝光估计均为有意义的创新
- 实验充分度: ⭐⭐⭐⭐ — 提供了详尽的消融实验（BLCE、LCEE、$N_l$、$s$），包含多种 baseline 对比和定性可视化分析
- 写作质量: ⭐⭐⭐⭐ — 论文结构清晰，动机阐述和方法推导逻辑完整，图表设计直观
- 价值: ⭐⭐⭐⭐ — 端到端动态去模糊 NVS 是实际应用中的重要问题，性能和速度双重提升具有很高的实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MSCD-GS: Motion-Separated Cooperative Deblurring Dynamic Reconstruction via Gaussian Splatting](../../CVPR2026/3d_vision/mscd-gs_motion-separated_cooperative_deblurring_dynamic_reconstruction_via_gauss.md)
- [\[CVPR 2026\] 4DEquine: Disentangling Motion and Appearance for 4D Equine Reconstruction from Monocular Video](../../CVPR2026/3d_vision/4dequine_disentangling_motion_and_appearance_for_4d_equine_reconstruction_from_m.md)
- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](../../CVPR2026/3d_vision/motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[CVPR 2026\] Learning Explicit Continuous Motion Representation for Dynamic Gaussian Splatting from Monocular Videos](../../CVPR2026/3d_vision/learning_explicit_continuous_motion_representation_for_dynamic_gaussian_splattin.md)

</div>

<!-- RELATED:END -->
