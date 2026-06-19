---
title: >-
  [论文解读] FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis
description: >-
  [ECCV 2024][图像生成][扩散模型] 提出 FouriScale，从频域分析视角出发，通过膨胀卷积+低通滤波替换预训练扩散模型中的卷积层，实现免训练的任意尺寸高分辨率图像生成，理论上证明了膨胀卷积保持结构一致性的有效性。 预训练的扩散模型（如 SD、SDXL）在超出训练分辨率生成图像时会出现严重问题： 重复模式（P…
tags:
  - "ECCV 2024"
  - "图像生成"
  - "扩散模型"
  - "高分辨率生成"
  - "免训练"
  - "频域分析"
  - "膨胀卷积"
---

# FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis

**会议**: ECCV 2024  
**arXiv**: [2403.12963](https://arxiv.org/abs/2403.12963)  
**代码**: [GitHub](https://github.com/LeonHLJ/FouriScale)  
**领域**: 图像生成  
**关键词**: 扩散模型, 高分辨率生成, 免训练, 频域分析, 膨胀卷积

## 一句话总结

提出 FouriScale，从频域分析视角出发，通过膨胀卷积+低通滤波替换预训练扩散模型中的卷积层，实现免训练的任意尺寸高分辨率图像生成，理论上证明了膨胀卷积保持结构一致性的有效性。

## 研究背景与动机

预训练的扩散模型（如 SD、SDXL）在超出训练分辨率生成图像时会出现严重问题：

**重复模式（Pattern Repetition）**：图像中出现重复的物体或结构，如一张脸上出现多只眼睛

**结构失真**：整体构图和局部细节出现异常变形

现有解决方案的不足：

| 方法 | 缺陷 |
|------|------|
| MultiDiffusion / SyncDiffusion | 拼接重叠 patch，缺乏全局方向，无法生成以特定物体为中心的图像 |
| Attn-Entro | 从注意力熵角度出发，但在高倍放大下效果不佳 |
| ScaleCrafter | 发现卷积层是关键，使用重膨胀+卷积分散操作，但（1）结论来自经验观察缺乏理论支撑，（2）需要离线预计算线性变换，缺乏通用性 |

FouriScale 的核心创新在于**从频域分析角度给出了理论解释**，并提出了更简洁、通用、无需预计算的解决方案。

## 方法详解

### 整体框架

FouriScale 将预训练扩散模型的 UNet 中的原始卷积层替换为：

$$Conv_k(F) \rightarrow Conv_{k'}(iDFT(H \odot DFT(F)))$$

即：先对特征图做低通滤波，再用膨胀卷积处理。其中：
- $k'$ 是原始卷积核 $k$ 的膨胀版本
- $H$ 是低通滤波器
- DFT/iDFT 是离散傅里叶变换/逆变换

加上 padding-then-crop 策略处理任意纵横比，以及 FouriScale Guidance 提升细节质量。

### 关键设计

#### 结构一致性：膨胀卷积的频域证明

核心理论问题：如何找到一个新卷积核 $k'$，使得在高分辨率特征上做卷积的结果与在低分辨率（下采样后）特征上做原始卷积等价？

**结构一致性方程**：
$$Down_s(F) \circledast k = Down_s(F \circledast k')$$

**Lemma 1（关键引理）**：空间下采样 $s$ 倍等价于将傅里叶谱分割为 $s \times s$ 个等大块并求平均叠加（缩放 $1/s^2$）：

$$DFT(Down_s(F(x,y))) = \frac{1}{s^2} \sum_{i=0}^{s-1} \sum_{j=0}^{s-1} F_{(i,j)}(u,v)$$

将结构一致性方程变换到频域后发现：**理想的 $k'$ 的傅里叶谱应该是原始核 $k$ 的傅里叶谱的 $s \times s$ 周期重复**。

**定理**：膨胀因子为 $(H/h, W/w)$ 的膨胀卷积恰好满足这一周期性要求！因为膨胀核的 DFT 中指数项在整数倍处恢复为原始核的值：

$$e^{-j2\pi(\frac{p'·m}{d_h·M} + \frac{q'·n}{d_w·N})} = e^{-j2\pi(\frac{pm}{M} + \frac{qn}{N})}$$

这是本文最重要的理论贡献——为 ScaleCrafter 观察到的膨胀操作有效性给出了严格的频域证明。

#### 尺度一致性：低通滤波

仅用膨胀卷积不足以完全解决重复模式问题。原因在于**混叠效应（Aliasing）**：

- 下采样会导致高频折叠到低频并叠加（根据 Theorem 3.1）
- 这改变了原始信号的基本频率成分，破坏了尺度间的一致性

解决方案：在膨胀卷积前引入**低通滤波**，去除可能导致混叠的高频分量。

最优的低通滤波器掩码大小为 $M/s_h \times N/s_w$（频谱居中情况下），恰好保留下采样分辨率内的所有有效频率。

实验验证（图4）：添加低通滤波后，高低分辨率特征的频率分布差距显著缩小。

#### 任意尺寸生成：Padding-then-Crop

当目标和训练分辨率的纵横比不同时，高度和宽度的膨胀率不同，会导致结构变形。

**Padding-then-Crop 策略**：

1. 计算 $r = \max(\lceil H_f/h_f \rceil, \lceil W_f/w_f \rceil)$
2. 将特征图零填充到 $r·h_f \times r·w_f$（统一的膨胀率）
3. 应用膨胀卷积 + 低通滤波
4. 裁剪回目标大小 $H_f \times W_f$

#### FouriScale Guidance

直接使用 FouriScale 修改后的 UNet 可能在背景中引入伪影（源于低通滤波的振铃效应和细节损失）。

解决方案：生成三个噪声估计：
1. 无条件估计（FouriScale 修改的 UNet）
2. 条件估计（FouriScale 修改的 UNet）
3. **额外的条件估计**（使用相同膨胀卷积但**更温和的低通滤波器**）

将第 2 个估计的 attention map 替代到第 3 个估计中（类似 MasaCtrl 的思路），使用第 3 个估计作为最终的条件估计。这样既保持了正确的结构信息，又避免了低通滤波造成的质量下降。

#### 退火策略（Annealing）

- 前 $S_{init}$ 步使用理想的膨胀卷积和低通滤波（建立结构）
- 从 $S_{init}$ 到 $S_{stop}$，逐步将膨胀因子和 $r$ 减小到 1
- $S_{stop}$ 之后使用原始 UNet 精细化细节

### 损失函数 / 训练策略

**完全免训练**！不需要任何微调或离线预计算。所有操作都在推理时通过替换卷积层实现。

**SDXL 特殊设置**：使用更温和的低通滤波器（系数 $\sigma=0.6$），不完全去除高频而是衰减。因为 SDXL 本身在训练时就支持多纵横比，对尺度变化有一定鲁棒性。

## 实验关键数据

### 主实验

**SD 2.1 不同放大倍率的 FID 对比**

| 分辨率 (倍率) | 方法 | FIDr↓ | KIDr↓ | FIDb↓ | KIDb↓ |
|-------------|------|-------|-------|-------|-------|
| 4× (1:1) | Vanilla | 29.90 | 1.11 | 19.21 | 0.54 |
| 4× (1:1) | ScaleCrafter | 25.19 | 0.98 | 13.88 | 0.40 |
| 4× (1:1) | **FouriScale** | **25.17** | **0.98** | **13.57** | **0.40** |
| 16× (1:1) | Vanilla | 84.01 | 3.28 | 82.25 | 3.05 |
| 16× (1:1) | ScaleCrafter | 40.91 | 1.32 | 33.23 | 0.90 |
| 16× (1:1) | **FouriScale** | **39.49** | **1.27** | **28.14** | **0.73** |

**SDXL 不同放大倍率的 FID 对比**

| 分辨率 (倍率) | 方法 | FIDr↓ | KIDr↓ |
|-------------|------|-------|-------|
| 4× (1:1) | Vanilla | 49.81 | 1.84 |
| 4× (1:1) | ScaleCrafter | 49.46 | 1.73 |
| 4× (1:1) | **FouriScale** | **33.89** | **1.21** |
| 16× (1:1) | Vanilla | 116.40 | 5.45 |
| 16× (1:1) | ScaleCrafter | 84.58 | 3.53 |
| 16× (1:1) | **FouriScale** | **56.66** | **2.18** |

**FouriScale 在 SDXL 上的优势极为突出**，ScaleCrafter 在 SDXL 上常常无法生成可接受的图像。

### 消融实验

**SD 2.1, 16× (2048×2048) 各组件贡献**

| 方法 | FIDr↓ |
|------|-------|
| FouriScale (完整) | 39.49 |
| 去掉 Guidance | 43.75 (+4.26) |
| 去掉 Guidance + 低通滤波 | 46.74 (+7.25) |

每个组件都有明确贡献：低通滤波解决尺度一致性问题（-2.99），Guidance 提升细节质量（-4.26）。

### 关键发现

1. **理论优雅且实践有效**：膨胀卷积的频域证明不仅是理论贡献，实际效果也显著优于 ScaleCrafter
2. **SDXL 兼容性突出**：ScaleCrafter 在 SDXL 上表现不佳（FIDr 84.58 vs 56.66），本方法的通用性更强
3. **无需预计算**：相比 ScaleCrafter 需要离线计算线性变换，FouriScale 随时可用
4. **推理速度更快**：16× SDXL 下，ScaleCrafter 577s vs FouriScale 540s
5. **低通滤波的理论解释**：通过 Nyquist 定理给出了最优滤波器尺寸的理论选择

## 亮点与洞察

1. **深刻的频域理论分析**：不仅给出了经验有效的方案，更从傅里叶变换角度严格证明了为什么膨胀卷积能解决重复模式问题
2. **简洁性**：整个方法不需要任何训练、微调或离线计算，只需在推理时替换卷积层
3. **通用性**：适用于 SD 1.5、SD 2.1、SDXL 三种不同的预训练模型，且支持任意纵横比
4. **退火策略的直觉**：前期用 FouriScale 确定结构，后期用原始 UNet 填充细节——这种分阶段思想很有启发性
5. **FouriScale Guidance 的设计**：巧妙地利用 attention map 替换在结构正确性和细节丰富性之间取得平衡

## 局限与展望

1. **超高分辨率仍有挑战**：4096×4096 等极端分辨率下仍可能出现伪影
2. **仅适用于 UNet 架构**：方法聚焦于卷积层操作，不适用于纯 Transformer（如 DiT）架构的扩散模型
3. **SDXL 需要特殊处理**：需要使用更温和的低通滤波器（$\sigma=0.6$），说明方法不是完全自动的
4. **退火参数需要调节**：$S_{init}$ 和 $S_{stop}$ 的选择可能影响结果
5. **未与 patch 拼接方法组合**：与 MultiDiffusion 等方法组合可能实现更高分辨率

## 相关工作与启发

- **ScaleCrafter**：发现卷积层是关键的经验性工作，FouriScale 为其提供了理论基础并给出了更好的方案
- **Spectral Pooling**：低通滤波（频域池化）的灵感来源
- **MasaCtrl**：attention map 替换的思路被用于 FouriScale Guidance
- **FreeU**：同样从频域角度优化扩散模型，FouriScale 在所有实验中默认使用 FreeU
- 启发：频域分析是理解扩散模型行为的重要工具。未来可以进一步研究 attention 层在频域的行为，以及如何将类似思路应用于 Transformer 架构

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4.5 |
| 理论深度 | 5 |
| 实验充分性 | 4.5 |
| 实用价值 | 4 |
| 写作质量 | 4.5 |
| 总体评分 | 4.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models](freediff_progressive_frequency_truncation_for_image_editing_with_diffusion_model.md)
- [\[ECCV 2024\] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [\[ECCV 2024\] Editable Image Elements for Controllable Synthesis](editable_image_elements_for_controllable_synthesis.md)
- [\[ECCV 2024\] Enhancing Perceptual Quality in Video Super-Resolution through Temporally-Consistent Detail Synthesis using Diffusion Models](enhancing_perceptual_quality_in_video_super-resolution_through_temporally-consis.md)
- [\[CVPR 2026\] Toward Diffusible High-Dimensional Latent Spaces: A Frequency Perspective](../../CVPR2026/image_generation/toward_diffusible_high-dimensional_latent_spaces_a_frequency_perspective.md)

</div>

<!-- RELATED:END -->
