---
title: >-
  [论文解读] Spatially-Variant Degradation Model for Dataset-free Super-resolution
description: >-
  [ECCV 2024][图像恢复][盲图像超分辨率] 提出首个无需数据集训练的空间变化退化模型 SVDSR，每个像素的退化核由可学习的原子核字典的线性组合表示，系数矩阵通过模糊集的隶属函数从图像纹理信息推导，在 MAP 框架下用 Monte Carlo EM 算法推断，$2\times$ 超分平均提升 1 dB。
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "盲图像超分辨率"
  - "空间变化退化"
  - "无数据集"
  - "模糊核估计"
  - "Monte Carlo EM"
---

# Spatially-Variant Degradation Model for Dataset-free Super-resolution

**会议**: ECCV 2024  
**arXiv**: 2407.08252 (https://arxiv.org/abs/2407.08252)  
**代码**: [https://github.com/shaojieguoECNU/SVDSR](https://github.com/shaojieguoECNU/SVDSR)  
**领域**: 图像恢复  
**关键词**: 盲图像超分辨率, 空间变化退化, 无数据集, 模糊核估计, Monte Carlo EM

## 一句话总结

提出首个无需数据集训练的空间变化退化模型 SVDSR，每个像素的退化核由可学习的原子核字典的线性组合表示，系数矩阵通过模糊集的隶属函数从图像纹理信息推导，在 MAP 框架下用 Monte Carlo EM 算法推断，$2\times$ 超分平均提升 1 dB。

## 研究背景与动机

盲图像超分辨率（BISR）的核心挑战在于准确估计未知退化算子 $\mathcal{D}$。退化模型可表示为 $\boldsymbol{y} = (\mathcal{D}\boldsymbol{x})\downarrow_s + \boldsymbol{n}$。现有方法存在两个维度的局限：

**空间不变性假设**：多数方法假设整幅图像使用同一退化核，但真实场景中平坦区域和纹理密集区域的退化差异显著

**数据集依赖**：空间变化退化方法（如 KOALA, DARM, LARPAR）需要在大规模配对数据集上训练，降低了实用性

本文弥合了这两个维度：**首个同时实现空间变化退化建模和无数据集深度学习的 BISR 方法**。相比之前空间变化方法使用 72 个预定义原子核，本文仅需 5 个可学习原子核（每个仅 3 个参数），大幅降低参数量。

## 方法详解

### 整体框架

整体流程基于 MAP 框架 + MCEM 推断：
1. 构建空间变化退化模型：可学习原子核字典 + 基于纹理的系数矩阵
2. 设计概率 BISR 模型：包含空间频率双域似然、核先验和图像先验
3. 交替执行 E-Step（采样潜变量 $\boldsymbol{z}$）和 M-Step（更新核参数 $\boldsymbol{\Gamma}$ 和网络权重 $\boldsymbol{\phi}$）

### 关键设计

1. **空间变化退化模型**：基于 O'Leary 分解公式，每个像素 $[h,w]$ 的退化表示为 $N_{\mathcal{D}}$ 个原子核的加权组合：$(\mathcal{D}\boldsymbol{x})[h,w] = \sum_{r,c}\sum_{i=1}^{N_{\mathcal{D}}} \boldsymbol{W}_i[h,w] \mathcal{D}_i \boldsymbol{x}[h-r,w-c]$。每个原子核 $\mathcal{D}_i$ 是各向异性高斯核，通过分解仅需 3 个可学习参数 $\{\theta_i, \sigma_{i,1}, \sigma_{i,2}\}$（旋转角 + 两个标准差）。这比先前方法的 72 个预定义核灵活且参数少得多。

2. **模糊集系数矩阵**：系数矩阵 $\boldsymbol{W}_i$ 不通过神经网络学习，而是通过模糊集理论的隶属函数从图像纹理推导：$\boldsymbol{W}_i = \frac{\boldsymbol{\mu}_i(\tilde{\boldsymbol{x}})}{\sum \boldsymbol{\mu}_i(\tilde{\boldsymbol{x}})}$，其中 $\boldsymbol{\mu}_i(\tilde{\boldsymbol{x}}) = \exp\left(-\frac{(N_{\mathcal{D}}-1)}{2\sigma_g^2}(\boldsymbol{h}(\tilde{\boldsymbol{x}}) - \frac{i-1}{N_{\mathcal{D}}-1})^2\right)$。纹理特征 $\boldsymbol{h}(\tilde{\boldsymbol{x}}) = \boldsymbol{H} * (\nabla \tilde{\boldsymbol{x}})$，用一阶导数提取梯度后用中值滤波平滑。这个设计利用了一个关键观察：**退化核形状与所在区域的纹理密度高度相关**。

3. **双域似然函数**：不同于仅在空间域定义似然，本文同时约束空间域和频率域：$y \sim \mathcal{N}(\boldsymbol{y} | (\mathcal{D}\boldsymbol{x})\downarrow_s, \sigma_y) \cdot \mathcal{N}(\mathcal{F}(\boldsymbol{y}) | \mathcal{F}((\mathcal{D}\boldsymbol{x})\downarrow_s), \sigma_f)$。频率域约束增强了重建效果（消融实验中去除后 PSNR 下降）。

4. **图像先验（Deep Image Prior 变体）**：使用 3 层 U-Net 网络 $G(\boldsymbol{z}; \boldsymbol{\phi})$ 作为图像隐式先验，对 $\boldsymbol{z}$ 施加高斯先验 $\mathcal{N}(0, \sigma_z)$，对网络输出施加 Laplace 梯度先验 $\mathcal{L}(\nabla G | 0, \sigma_x)$ 抑制过拟合。额外引入 Instance Normalization 和频域 skip connection 进一步缓解过拟合。

### 损失函数 / 训练策略

MCEM 推断算法交替执行：
- **E-Step**：通过 Stochastic Gradient Langevin Dynamics (SGLD) 采样潜变量 $\boldsymbol{z}$，采样步数 $n_z = 5$
- **M-Step**：用 ADAM 优化器更新核参数 $\boldsymbol{\Gamma}$ 和网络权重 $\boldsymbol{\phi}$，最大化 ELBO

关键参数：$N_{\mathcal{D}} = 5$（原子核数）、$\sigma_g = 0.5$、$\sigma_y = 1$、$\sigma_f = 2$、$\sigma_x = 2.5$、$\sigma_z = 1$、最大迭代 5000 次。

## 实验关键数据

### 主实验

| 数据集 | 缩放 | SVDSR (Ours) | BSRDM (SOTA) | 提升 |
|--------|------|-------------|-------------|------|
| Set5 | ×2 | **33.51/0.92** | 32.76/0.91 | +0.75 |
| Set14 | ×2 | **29.61/0.83** | 28.65/0.81 | +0.96 |
| Urban100 | ×2 | **26.40/0.79** | 25.46/0.76 | +0.94 |
| Manga109 | ×2 | **29.89/0.89** | 28.49/0.87 | +1.40 |
| DIV2K100 | ×2 | **29.46/0.82** | 28.32/0.78 | +1.14 |
| Set5 | ×3 | **31.37/0.89** | 30.96/0.88 | +0.41 |
| Set14 | ×3 | **28.14/0.79** | 27.67/0.77 | +0.47 |
| Set5 | ×4 | **29.29/0.85** | 29.02/0.85 | +0.27 |
| Urban100 | ×4 | **23.90/0.70** | 23.47/0.68 | +0.43 |

$2\times$ 超分平均 PSNR 提升约 1 dB，且在多个数据集上与非盲方法（给定真实核的 ZSSR-NB）性能接近。

### 消融实验

| 配置 | Set14 PSNR (×2) | 说明 |
|------|-----------------|------|
| 完整模型 | **29.61** | — |
| Case1: 去除频域似然 | 29.61 | 频域约束仍有贡献 |
| Case2: 去除频域 skip | 29.52 | -0.09 |
| Case3: 去除 Instance Norm | 29.28 | -0.33，过拟合效应明显 |

| 原子核数 $N_{\mathcal{D}}$ | Set5 | Set14 | Urban100 | Manga109 | DIV2K100 |
|---------------------------|------|-------|----------|----------|----------|
| 1（退化为空间不变） | 33.19 | 29.13 | 25.90 | 29.23 | 29.05 |
| 3 | 33.29 | 29.40 | 26.11 | 29.53 | 29.30 |
| **5** | **33.51** | **29.61** | **26.40** | **29.89** | **29.46** |
| 7 | 33.43 | 29.58 | 26.21 | 29.67 | 29.52 |
| 9 | 33.27 | 29.59 | 26.20 | 29.74 | 29.54 |

### 关键发现

- **空间变化 vs 空间不变**：$N_{\mathcal{D}}=5$ 比 $N_{\mathcal{D}}=1$ 提升 0.3~0.7 dB，验证空间变化建模的价值
- **5 个原子核为最优**：继续增加反而性能下降，说明先前方法的 72 个核是冗余的
- Instance Normalization 对抑制过拟合至关重要（-0.33 dB）
- 模型大小 850K 参数，运行时间 33s（256×256，×2），与 BSRDM 相当

## 亮点与洞察

1. **首创无数据集空间变化退化**：打破了"空间变化退化模型必须在大数据集上训练"的范式
2. **模糊集理论的巧妙引入**：利用图像纹理信息推导系数矩阵，避免了学习巨大的每像素系数矩阵，同时具有物理可解释性
3. **极简参数化**：每个原子核仅 3 个参数（旋转角 + 两个方差），基于各向异性高斯核分解的理论支撑
4. **可视化效果直观**：系数矩阵的可视化清晰反映了图像纹理结构，不同区域的原子核形状差异显著

## 局限与展望

1. **高倍超分性能下降**：$4\times$ 时优势缩小，因纹理信息损失过多导致空间变化建模优势减弱
2. **色彩失真问题**：强噪声图像可能出现色彩偏移，是多数 SR 方法的共性问题
3. **计算效率**：5000 次 EM 迭代、每次含 SGLD 采样，总时间 33s 尚可但难以实时
4. 可探索将模糊集系数矩阵的思想推广到其他图像反问题（如去模糊、去雨）

## 相关工作与启发

- Deep Image Prior (DIP) 框架为无数据集方法提供了基础，本文在此基础上引入空间变化退化建模
- 与 BSRDM 的对比说明：空间变化退化建模可以仅用少量额外参数（88K）带来显著提升
- Monte Carlo EM + SGLD 的推断框架为概率图像恢复提供了优雅的数学框架
- 模糊集理论在计算机视觉中的应用仍有很大探索空间

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首个无数据集空间变化退化模型，模糊集系数矩阵的设计独到
- **实验充分度**: ⭐⭐⭐⭐ — 5 个数据集 × 3 个缩放因子，消融覆盖核心组件，缺少与 LARPAR 等空间变化方法的直接对比
- **写作质量**: ⭐⭐⭐⭐ — 数学推导清晰，但符号密集，概率模型部分阅读门槛较高
- **实用价值**: ⭐⭐⭐⭐ — 无需数据集训练是显著优势，性能提升实在，代码开源

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] A New Dataset and Framework for Real-World Blurred Images Super-Resolution](a_new_dataset_and_framework_for_real-world_blurred_images_super-resolution.md)
- [\[ICCV 2025\] Towards a Universal Image Degradation Model via Content-Degradation Disentanglement](../../ICCV2025/image_restoration/towards_a_universal_image_degradation_model_via_content-degradation_disentanglem.md)
- [\[ECCV 2024\] Rethinking Image Super-Resolution from Training Data Perspectives](rethinking_image_super-resolution_from_training_data_perspectives.md)
- [\[ECCV 2024\] Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks](overcoming_distribution_mismatch_in_quantizing_image_super-resolution_networks.md)
- [\[ECCV 2024\] Raindrop Clarity: A Dual-Focused Dataset for Day and Night Raindrop Removal](raindrop_clarity_a_dual-focused_dataset_for_day_and_night_raindrop_removal.md)

</div>

<!-- RELATED:END -->
