---
title: >-
  [论文解读] BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream
description: >-
  [ECCV 2024][3D视觉][NeRF] 提出 BeNeRF，仅从单张模糊图像：及其对应的事件流（event stream）联合恢复神经辐射场与相机运动轨迹，无需多视角输入或已知位姿，即可实现高质量去模糊与新视角合成。 NeRF 依赖多视角清晰图像：传统 NeRF 需要多张标定好的清晰 RGB 图像才能重建 3D 场景…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "NeRF"
  - "Event Camera"
  - "Motion Deblurring"
  - "novel view synthesis"
  - "Camera Motion Estimation"
  - "B-Spline"
---

# BeNeRF: Neural Radiance Fields from a Single Blurry Image and Event Stream

**会议**: ECCV 2024  
**arXiv**: [2407.02174](https://arxiv.org/abs/2407.02174)  
**代码**: [wu-cvgl/BeNeRF](https://github.com/wu-cvgl/BeNeRF)  
**领域**: 3D视觉  
**关键词**: NeRF, Event Camera, Motion Deblurring, novel view synthesis, Camera Motion Estimation, B-Spline

## 一句话总结

提出 BeNeRF，仅从**单张模糊图像**及其对应的事件流（event stream）联合恢复神经辐射场与相机运动轨迹，无需多视角输入或已知位姿，即可实现高质量去模糊与新视角合成。

## 研究背景与动机

**NeRF 依赖多视角清晰图像**：传统 NeRF 需要多张标定好的清晰 RGB 图像才能重建 3D 场景，对输入质量和数量要求高。

**模糊图像蕴含丰富信息**：运动模糊虽然降低图像质量，但实际上编码了曝光期间的相机运动轨迹和更多结构信息，可以被有效利用。

**事件相机的互补特性**：事件相机以极低延迟异步捕捉像素亮度变化，拥有极高的时间分辨率，恰好与帧相机的光子积分成像过程互补。

**现有事件去模糊方法的局限**：EDI、eSLNet 等方法只能恢复清晰图像，无法提取相机运动轨迹或进行 3D 重建，限制了其在三维视觉任务中的应用。

**多视角事件 NeRF 方法的不足**：E2NeRF 等方法虽结合了事件流，但仍需多视角图像输入，且采用两阶段策略（先用 EDI 恢复清晰图像再用 COLMAP 估计位姿），会引入累积误差。

**单图 NeRF 的挑战**：从单张图像学习 NeRF 极度 ill-posed，现有方法通常需要大规模数据集预训练来学习先验，泛化能力有限。

## 方法详解

### 整体框架

给定一张模糊图像及其对应的事件流，BeNeRF 联合优化：(1) 用 MLP 表示的神经辐射场（3D 场景）；(2) 用 SE(3) 空间中的三次 B 样条曲线表示的相机运动轨迹。通过物理成像模型合成模糊图像和事件累积图像，再与真实测量的差异最小化来训练整个系统。

### 关键设计一：SE(3) 三次 B 样条运动建模

- 用一组可学习的控制节点 $\boldsymbol{T}_{c_i}^w \in \text{SE}(3)$ 定义三次 B 样条曲线，表示连续的相机运动轨迹。
- 利用 De Boor-Cox 公式的矩阵表示，任意时刻 $t$ 的位姿由相邻四个控制节点插值得到。
- 由于单张图像曝光时间短，仅需 4 个控制节点即可充分表示运动，初始化在单位位姿附近随机采样。
- 相比线性插值，三次 B 样条能更好地建模复杂的非线性运动（PSNR 提升约 3-5 dB）。

### 关键设计二：模糊图像物理成像模型

- 模糊图像建模为曝光时间内 $n$ 张虚拟清晰图像的均值：$\mathbf{B}(\mathbf{x}) \approx \frac{1}{n}\sum_{i=0}^{n-1}\mathbf{I}_i(\mathbf{x})$。
- 每张虚拟清晰图像由 B 样条插值得到的对应位姿从 NeRF 渲染而来。
- 消融实验表明 $n=19$ 在质量和效率之间取得最佳平衡。

### 关键设计三：事件流累积与归一化

- 将时间间隔 $\Delta t$ 内的事件累积为事件图像 $\mathbf{E}(\mathbf{x})$，归一化消除未知对比度阈值 $C$ 的影响。
- 从 NeRF 渲染起止时刻的灰度图像，计算合成事件图像：$\hat{\mathbf{E}}(\mathbf{x})=\log(\mathbf{I}_{end})-\log(\mathbf{I}_{start})$。
- 事件流提供了丰富的时间约束，有效正则化了单图 NeRF 学习中的几何歧义。

### 损失函数与训练

总损失为光度损失与事件损失的加权和：

$$\mathcal{L}_{total} = \mathcal{L}_p + \beta \mathcal{L}_e$$

- $\mathcal{L}_p = \|\mathbf{B} - \hat{\mathbf{B}}\|^2$：合成模糊图与真实模糊图的 MSE。
- $\mathcal{L}_e = \|\mathbf{E}_n - \hat{\mathbf{E}}_n\|^2$：归一化事件图的 MSE。
- $\beta$ 在合成数据上取 0.1，真实数据上取 2。使用两个独立 Adam 优化器分别优化场景模型和位姿，学习率从 $5\times10^{-4}$ 指数衰减，训练 80K 迭代。

## 实验关键数据

### 表1：与单图去模糊方法在合成数据集上的比较（PSNR↑ / LPIPS↓）

| 方法 | Livingroom | Whiteroom | Pinkcastle | Tanabata | Outdoorpool | 平均 |
|---|---|---|---|---|---|---|
| SRN-Deblur | 30.86 / .253 | 27.59 / .250 | 23.12 / .325 | 19.89 / .426 | 27.79 / .359 | 25.85 / .323 |
| NAFNet | 29.92 / .227 | 28.16 / .199 | 22.41 / .306 | 18.96 / .391 | 26.75 / .328 | 25.24 / .290 |
| Restormer | 29.48 / .239 | 27.39 / .249 | 22.22 / .337 | 18.82 / .425 | 27.35 / .366 | 25.05 / .323 |
| **BeNeRF** | **37.11 / .063** | **32.95 / .079** | **29.68 / .076** | **32.14 / .052** | **36.38 / .068** | **33.65 / .068** |

BeNeRF 在 PSNR 上平均超过最优基线方法 **+7.8 dB**，LPIPS 降低约 **75%**。

### 表2：与多视角 NeRF 方法在 E2NeRF 合成数据集上的比较（PSNR↑ / LPIPS↓）

| 方法 | Chair | Ficus | Hotdog | Lego | Materials | Mic | 平均 |
|---|---|---|---|---|---|---|---|
| NeRF | 24.29 / .125 | 22.98 / .104 | 27.75 / .116 | 21.95 / .210 | 19.99 / .151 | 20.50 / .158 | 22.91 / .144 |
| E2NeRF (多视角) | 31.28 / .061 | 30.00 / .036 | 34.34 / .066 | 28.11 / .108 | 27.27 / .092 | 27.60 / .072 | 29.77 / .073 |
| **BeNeRF (单图)** | 31.17 / **.050** | **30.81** / **.030** | 34.31 / **.054** | 28.09 / **.075** | **27.44** / **.071** | 26.13 / .074 | 29.66 / **.059** |

仅用单张模糊图像即达到多视角方法 E2NeRF 的同等 PSNR 水平，LPIPS 指标甚至更优。

### 真实数据集结果（BRISQUE↓）

| 方法 | Camera | Lego | Letter | Plant | Toys | 平均 |
|---|---|---|---|---|---|---|
| EDI | 29.74 | 29.35 | 28.74 | 31.09 | 37.09 | 31.20 |
| E2NeRF (多视角) | 33.40 | 33.85 | 37.41 | 32.02 | 43.00 | 35.94 |
| **BeNeRF (单图)** | **19.47** | **25.86** | **27.37** | **21.46** | **25.20** | **23.87** |

## 亮点

1. **极端设置下的突破**：首次实现从单张模糊图像 + 事件流恢复 NeRF，是该领域输入条件最严苛的工作。
2. **无需预训练或先验**：作为 test-time optimization 方法，不依赖大规模数据集预训练，天然避免了泛化性问题。
3. **与多视角方法性能持平**：仅用单张图像即达到 E2NeRF（多视角 + 长事件流）的同等水平，LPIPS 甚至更优。
4. **物理模型驱动**：模糊成像和事件生成均基于物理模型建模，真实数据上显著优于学习型方法。
5. **联合优化位姿**：无需 COLMAP 或任何外部位姿估计，从零开始联合恢复场景和运动。

## 局限与展望

1. **仅处理单张图像**：无法利用视频序列中的时序一致性，如扩展到连续帧将进一步提升重建质量。
2. **训练开销较大**：每张图像需 80K 迭代优化，未采用 Instant-NGP 等加速表示（文中提到可替换但未实验验证）。
3. **依赖事件相机**：需要额外的事件传感器硬件，限制了实际应用场景。
4. **场景类型受限**：实验主要在室内小场景验证，对大规模户外场景、动态物体等情况未做评估。
5. **虚拟清晰图像数量敏感**：$n=19$ 的选择是经验性的，不同运动模式可能需要不同配置。

## 与相关工作的对比

| 方法 | 输入要求 | 位姿 | 事件流 | 3D 重建 |
|---|---|---|---|---|
| BAD-NeRF | 多视角模糊图像 | COLMAP 初始化 | ✗ | ✓ |
| E2NeRF | 多视角模糊 + 长事件流 | COLMAP (EDI辅助) | ✓ | ✓ |
| EDI | 单图 + 事件流 | ✗ | ✓ | ✗ |
| **BeNeRF** | **单图 + 事件流** | **联合优化** | ✓ | ✓ |

BeNeRF 在输入要求最少的条件下同时实现了相机运动估计、图像去模糊和 3D 场景重建。

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次将单张模糊图像 + 事件流用于 NeRF 恢复，问题设置新颖且具挑战性
- 实验充分度: ⭐⭐⭐⭐ — 合成/真实数据集全面评估，消融实验详尽，与多类基线比较
- 写作质量: ⭐⭐⭐⭐ — 方法推导清晰，物理建模严谨，图表质量高
- 价值: ⭐⭐⭐⭐ — 在极端输入条件下达到多视角方法性能，展示了事件相机在 3D 视觉中的巨大潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Omni-Recon: Harnessing Image-Based Rendering for General-Purpose Neural Radiance Fields](omni-recon_harnessing_image-based_rendering_for_general-purpose_neural_radiance_.md)
- [\[ECCV 2024\] GeometrySticker: Enabling Ownership Claim of Recolorized Neural Radiance Fields](geometrysticker_enabling_ownership_claim_of_recolorized_neural_radiance_fields.md)
- [\[ECCV 2024\] G2fR: Frequency Regularization in Grid-Based Feature Encoding Neural Radiance Fields](g2fr_frequency_regularization_in_grid-based_feature_encoding_neural_radiance_fie.md)
- [\[CVPR 2026\] Evidential Neural Radiance Fields](../../CVPR2026/3d_vision/evidential_neural_radiance_fields.md)
- [\[ECCV 2024\] Vista3D: Unravel the 3D Darkside of a Single Image](vista3d_unravel_the_3d_darkside_of_a_single_image.md)

</div>

<!-- RELATED:END -->
