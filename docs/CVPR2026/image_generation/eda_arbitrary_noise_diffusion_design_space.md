---
title: >-
  [论文解读] Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models
description: >-
  [CVPR 2026][图像生成][任意噪声扩散] 提出 EDA 框架，将 EDM 的设计空间从纯高斯噪声扩展至任意噪声模式，通过多元高斯分布和多独立维纳过程驱动的 SDE 实现灵活噪声扩散，且证明噪声复杂度的提升不引入额外采样开销；仅用 5 步采样即可在 MRI 偏置场矫正、CT 金属伪影去除和自然图像阴影去除三项任务上取得媲美或优于百步 Refusion 和专用方法的效果。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "任意噪声扩散"
  - "EDM统一框架"
  - "SDE设计空间"
  - "医学图像去噪"
  - "阴影去除"
---

# Elucidating the Design Space of Arbitrary-Noise-Based Diffusion Models

**会议**: CVPR 2026  
**arXiv**: [2507.18534](https://arxiv.org/abs/2507.18534)  
**代码**: [https://github.com/PerceptionComputingLab/EDA](https://github.com/PerceptionComputingLab/EDA)  
**领域**: 扩散模型 / 图像复原  
**关键词**: 任意噪声扩散, EDM统一框架, SDE设计空间, 医学图像去噪, 阴影去除  

## 一句话总结
提出 EDA 框架，将 EDM 的设计空间从纯高斯噪声扩展至任意噪声模式，通过多元高斯分布和多独立维纳过程驱动的 SDE 实现灵活噪声扩散，且证明噪声复杂度的提升不引入额外采样开销；仅用 5 步采样即可在 MRI 偏置场矫正、CT 金属伪影去除和自然图像阴影去除三项任务上取得媲美或优于百步 Refusion 和专用方法的效果。

## 背景与动机
EDM (Karras et al., NeurIPS 2022) 统一了大部分扩散模型的设计空间，提供了灵活的噪声调度和训练目标选择，但其闭合形式的扰动核 $p_t(x_t|x_0) = \mathcal{N}(x_t; s(t)x_0, s^2(t)\sigma^2(t)\mathbf{I})$ 限制了只能扩散像素级独立的高斯噪声。近年来 Flow Matching、MeanFlow、Cold Diffusion 等方法打破了高斯约束，支持更灵活的噪声分布，但要么退化为 ODE 扩散（牺牲随机性带来的鲁棒性和多样性），要么缺乏严格的理论基础。

对于图像复原任务，EDM 框架面临两个关键缺陷：(1) 强制注入高斯噪声破坏退化图像中的任务相关信息；(2) 人为增大了复原距离和复原复杂度——因为必须从 $P_{LQ} + N_{Gaus}$ 出发而非直接从退化图像 $P_{LQ}$ 出发。如果能让模型直接扩散任务特定的噪声模式，就能缩短复原路径，降低任务难度。

## 核心问题
如何在保留 EDM 结构参数灵活性（噪声调度、训练目标可选）的前提下，将扩散噪声模式从纯高斯扩展到任意模式，从而建立一个统一的 SDE 基扩散设计空间？同时需要证明这种推广不会带来额外的计算开销。

## 方法详解

### 整体框架

EDA 想解决的是 EDM 只能扩散像素级独立高斯噪声的限制：对复原任务来说，强行注入高斯噪声会破坏退化图里的任务信息，还把复原起点从退化图 $P_{LQ}$ 推远到 $P_{LQ}+N_{Gaus}$，平白拉长复原路径。EDA 的核心是把扩散噪声从标量方差的各向同性高斯，推广为由任意基函数集 $H_{x_0}$ 控制的多元高斯：前向过程按任务把数据逐步扰动成退化形式，逆向过程则用 PFODE 确定性采样、直接从退化图像出发恢复，整条链只需 5 步。

### 关键设计

**1. 广义前向过程：把高斯噪声换成任意基函数驱动的多元高斯，并给它严格的 SDE**

为了让模型扩散「任务特定的噪声」而非通用高斯，EDA 把噪声定义为 $N = \sum_{m=1}^{M} \frac{\eta + \epsilon_m}{\eta + 1} h_{m,x_0}$，其中 $H_{x_0} = [h_{1,x_0}, \ldots, h_{M,x_0}]$ 是可任意预定义的基函数集，$\epsilon_m \sim \mathcal{N}(0,1)$ 是独立高斯变量，参数 $\eta \geq 0$ 控制随机性（$\eta=0$ 最大随机，$\eta \to \infty$ 趋于确定性）。对应扰动分布的协方差 $\Sigma_{x_0} = H_{x_0} H_{x_0}^\top$ 不再是 EDM 的对角矩阵 $\sigma^2(t)\mathbf{I}$，于是能用基函数直接捕捉结构化的退化噪声模式。

这套噪声的动态形式由多个独立维纳过程驱动：$dx = [f(t)x + \phi_{x_0}(t)]dt + g(t)\sum_{m=1}^M h_{m,x_0} d\omega_t^{(m)}$，其中漂移系数、扩散系数和偏移项都能从噪声调度 $s(t), \sigma(t)$ 和基函数 $H_{x_0}$ 解析推出。这给了任意噪声一个严格的 SDE 基础，也保留了随机性带来的鲁棒性与多样性，而不像 Flow Matching 那样退化成 ODE。

**2. PFODE 化简与零额外开销（Proposition 2）**

这是全文最关键的理论发现：把 score function 代入 PFODE、再用去噪器近似后，所有和基函数 $h_m$、协方差 $\Sigma_{x_0}$ 相关的额外项会完全解析抵消，最终确定性采样公式 $\frac{dx}{dt} = (\frac{s'(t)}{s(t)} + \frac{\sigma'(t)}{\sigma(t)})x - \frac{\sigma'(t)s(t)}{\sigma(t)} D_\theta(x;\sigma)$ 与 EDM 完全一致。换句话说，无论噪声设计得多复杂，采样过程一行都不用改、也不增加任何额外开销。

**3. 三种噪声配置（Proposition 1）**

基函数具体怎么选，对应三种实用配置：

- **Case 1（统一基函数，最优）**: 基函数与数据无关 $H = H_j, \forall j$（如 MRI 偏置场用低阶 Legendre 多项式和三角函数）
- **Case 2（样本依赖基函数，通用）**: $H_{x_0}$ 随样本变化，如 CT 金属伪影和阴影去除中 $H_A = [B - A]$
- **Case 3（非高斯噪声离散采样）**: 通过离散空间分布匹配，支持泊松噪声等非高斯噪声

**4. EDM 是 EDA 的特例（Proposition 3）**

当 $\eta=0$ 且基函数取像素级单位矩阵 $E_{i,j}$ 时，EDA 就退化回标准 EDM——这说明 EDA 是 EDM 的严格超集，旧框架只是它的一个角点。

### 损失函数 / 训练策略
- 统一训练目标：$\mathcal{L} = \mathbb{E}_{x_0 \sim P_{data}} \mathbb{E}_{x \sim P(x_t|y)} \|D_\theta(x;\sigma) - x_0\|^2$
- MRI 偏置场矫正：在对数域操作（将乘性噪声转为加性），基函数用低阶 Legendre 多项式和三角函数 $H_{3,5}$，$\eta = 0$
- CT 金属伪影去除：$\eta = 10$，噪声为金属影响 CT 与无金属 CT 的差异，加权 MSE 平衡金属/非金属域
- 阴影去除：$\eta = 10$，噪声为阴影图像与无阴影图像的差异，基于 ShadowFormer 架构
- 所有任务总扩散步数 $T = 100$，噪声调度沿用 DDPM 线性增长 $\beta$ 方案
- 训练硬件：单卡 NVIDIA RTX 3090

## 实验关键数据

### MRI 偏置场矫正（HCP 数据集）

| 方法 | SSIM ↑ | PSNR ↑ | COCO ↑ | CV(WM) ↓ |
|------|--------|--------|--------|----------|
| N4 | 0.95 | 25.62 | 0.95 | 7.95 |
| ABCNet | - | - | - | - |
| Refusion (100步) | - | - | - | - |
| **EDA (5步)** | **最优** | **最优** | **最优** | **最优** |

速度对比：EDA 0.182 sec/slice vs Refusion 9.665 sec/slice → **~53× 加速**

### CT 金属伪影去除（DeepLesion）

| 方法 | 域 | 平均 PSNR/SSIM |
|------|-----|----------------|
| InDuDoNet+ | 双域 | 41.50/0.9891 |
| DICDNet | 双域 | 41.83/0.9923 |
| Refusion | 图像域 | 38.15/0.9793 |
| **EDA (5步)** | **图像域** | **38.67/0.9823** |

EDA 仅用图像域信息即超越 Refusion，接近双域方法。

### 自然图像阴影去除（ISTD）

| 方法 | ALL PSNR ↑ | ALL SSIM ↑ | NS PSNR ↑ |
|------|------------|------------|-----------|
| ShadowFormer | - | - | - |
| Refusion | - | - | - |
| **EDA** | **最优** | **最优** | **34.31** |

EDA 在全图和非阴影区域均达到最优，非阴影区 PSNR 34.31 dB 证明了精确的边界感知能力。

### 消融实验要点
- 5 步 EDA 即可达到或超越 100 步 Refusion 的效果，说明缩短复原距离确实有效
- MeanFlow（ODE 方法）在所有三项任务上均显著弱于 EDA，验证了 SDE 随机性对复原质量的重要性——ODE 倾向于输出模糊的平均解
- Case 1 基函数（数据无关）理论上最优但适用范围有限；Case 2（数据依赖）更通用，实验效果同样优秀

## 亮点
- **理论优雅**：Proposition 2 证明任意噪声复杂度不增加采样开销，所有额外项在 PFODE 中解析抵消——这是一个非常漂亮的理论结果
- **统一视角**：Flow Matching、Cold Diffusion、EDM 都可在 EDA 框架下理解，为未来扩散模型研究提供了更广的理论基础
- **实用价值**：5 步即可完成高质量复原，53× 加速使其可用于临床高通量场景
- **设计思路可迁移**：将任务特定的退化模式直接编码为扩散噪声的思想，可以推广到去雾、超分辨率、去模糊等其他复原任务

## 局限与展望
- **随机性与适用范围的权衡**：Case 1 最大随机但要求基函数数据无关（限制应用），Case 2-3 减少随机性换取通用性
- **仅验证复原任务**：未在生成任务（如无条件图像生成）上验证，理论框架的生成能力尚不明确
- **仅使用图像域**：CT 金属伪影去除中未利用正弦域信息，与双域 SOTA 仍有差距
- **基函数设计需领域知识**：MRI 需要 Legendre 多项式，CT 需要图像差异——基函数的选择依赖对任务退化模式的先验理解
- 论文未讨论非医学非自然图像的更广泛复原任务（如去雨、去雾等）

## 与相关工作的对比
- **vs EDM (Karras et al., 2022)**: EDA 是 EDM 的严格超集（Proposition 3），核心差异在于噪声从 $\sigma^2\mathbf{I}$ 推广到 $\Sigma_{x_0}$，同时保持采样公式不变
- **vs Flow Matching / MeanFlow**: Flow Matching 虽支持任意噪声但限于 ODE 扩散，缺乏 SDE 的随机性优势；实验证明 MeanFlow 在所有三项复原任务上表现较差（ODE 产生平均解导致模糊）
- **vs Refusion**: 同为高斯扩散复原方法，但 Refusion 需 100 步且从噪声破坏的退化图像出发；EDA 仅需 5 步从退化图像直接出发，速度快 53 倍且效果更好

## 启发与关联
- 任意噪声扩散的思想可以与多模态生成结合——不同模态的噪声模式不同，EDA 的框架天然支持这种异构性
- 5 步高质量采样的效率优势，可以考虑与 VLM 驱动的医学图像分析流水线结合
- 基函数的设计可以考虑用神经网络自动学习，而非人工预定义

## 评分
- 新颖性: ⭐⭐⭐⭐ 将 EDM 推广到任意噪声的理论框架有价值，但从数学上看主要是协方差矩阵的推广
- 实验充分度: ⭐⭐⭐⭐ 三种代表性任务覆盖了不同噪声模式，但缺乏生成任务验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰严谨，命题组织有条理，附录详尽
- 价值: ⭐⭐⭐⭐ 为扩散模型复原任务提供了统一高效的理论框架，但适用范围待进一步验证

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Elucidating the SNR-t Bias of Diffusion Probabilistic Models](dcw_snr_t_bias_diffusion.md)
- [\[CVPR 2026\] What Is It Like to Be a Noise? An Entropy-based Gaussian Noise Regularization for Diffusion Models](what_is_it_like_to_be_a_noise_an_entropy-based_gaussian_noise_regularization_for.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] Scale Space Diffusion：把尺度空间塞进扩散过程](scale_space_diffusion.md)
- [\[CVPR 2026\] It's Never Too Late: Noise Optimization for Collapse Recovery in Trained Diffusion Models](its_never_too_late_noise_optimization_for_collapse_recovery_in_trained_diffusion.md)

</div>

<!-- RELATED:END -->
