---
title: >-
  [论文解读] M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction
description: >-
  [AAAI 2026][遥感][Spectral Reconstruction] 提出 M3SR，一种基于 Mamba 的多尺度多感知架构，通过空间-频率-光谱三分支并行融合结合 U-Net 多尺度结构，以 2.17M 参数和 100.9G FLOPs 的低计算代价在四个光谱重建基准上超越现有 SOTA 方法。
tags:
  - "AAAI 2026"
  - "遥感"
  - "Spectral Reconstruction"
  - "Mamba"
  - "State Space Model"
  - "Multi-Scale"
  - "Hyperspectral Imaging"
---

# M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2601.08293](https://arxiv.org/abs/2601.08293)  
**代码**: [https://github.com/zhangyuzecn/M3SR](https://github.com/zhangyuzecn/M3SR)  
**领域**: 遥感/高光谱图像重建  
**关键词**: Spectral Reconstruction, Mamba, State Space Model, Multi-Scale, Hyperspectral Imaging

## 一句话总结
提出 M3SR，一种基于 Mamba 的多尺度多感知架构，通过空间-频率-光谱三分支并行融合结合 U-Net 多尺度结构，以 2.17M 参数和 100.9G FLOPs 的低计算代价在四个光谱重建基准上超越现有 SOTA 方法。

## 研究背景与动机
高光谱成像（HSI）通过窄波段采集丰富的空间-光谱信息，广泛应用于环境监测、医学成像和农业等领域。然而直接采集 HSI 成本高昂且过程复杂，因此**光谱重建（SR）**——从 RGB 图像生成 HSI——成为重要的替代方案。

现有 SR 方法存在明确的发展脉络和局限：
- **传统方法**（稀疏字典、高斯过程、低秩表示）难以识别复杂模式
- **CNN 方法**（HSCNN+、HRNet）提升了性能，但难以捕获长程依赖
- **Transformer 方法**（MST++、ESSAformer）能建模长程关系，但计算复杂度随图像尺寸急剧增长

**Mamba** 架构基于状态空间模型（SSM），能以线性复杂度处理长序列，但现有 Mamba SR 方法面临两个核心挑战：
1. **单一空间感知**限制了对高光谱图像的全面理解
2. **单一尺度特征提取**难以同时捕获复杂结构和精细细节

M3SR 的核心思路是：设计多感知融合模块（MPF block），将空间、频率、光谱三种感知集成到一个统一块中，再将其嵌入 U-Net 实现多尺度特征提取与融合。

## 方法详解

### 整体框架
M3SR 采用基于 U-Net 的编码器-解码器结构：
1. **输入端**：接收 RGB 图像，通过浅层特征提取得到初始特征
2. **编码器路径**：通过下采样逐步提取多尺度语义特征，分为三个尺度：
    - **全局尺度**：捕获整体结构信息
    - **中间尺度**：聚焦上下文信息
    - **局部尺度**：恢复精细纹理信息
3. **解码器路径**：通过上采样恢复空间分辨率，结合跳跃连接融合多尺度特征
4. **输出端**：生成重建后的高光谱图像

每个尺度级别的核心构建块是**多感知融合块（MPF Block）**，包含空间、频率、光谱三个并行分支。

### 关键设计

**设计一：多感知融合块（MPF Block）—— 三分支并行感知**

MPF Block 包含三个并行分支，分别针对不同维度的信息：

**（1）空间感知分支**：基于 VMamba 的 2D 选择性扫描（SS2D）技术，将 2D 图像展开为 1D 序列，沿四个方向扫描以捕获长程空间依赖。VSS 块定义为：

$$VSS(x) = Lin(LN(SS2D(x')))$$
$$x' = SiLU(DWConv(Lin(x)))$$

空间分支通过 reshape 和 concat 操作增强空间特征：

$$\mathbf{F}_a^2 = Reshape(Concat(VSS(LN(\mathbf{F}_a^1)), \mathbf{F}_a^1))$$

**（2）频率感知分支**：使用离散小波变换（DWT）将输入分解为低频（$I_{LL}$）和三个高频分量（$I_{LH}, I_{HL}, I_{HH}$），分别捕获纹理细节：

$$\mathbf{F}_f^1 = DWT(\mathbf{F}_{in})$$
$$\mathbf{F}_f^2 = IDWT(Concat(VSS(LN(\mathbf{F}_f^1)), \mathbf{F}_f^1))$$

**（3）光谱感知分支**：基于原始 Mamba 块建模光谱维度的连续性依赖。将通道维度 $C$ 扩展为 $C \times G$，分组后用 Mamba 块提取光谱交互特征：

$$\mathbf{F}_e^1 = Mamba(Reshape(Conv(\mathbf{F}_{in})))$$
$$\mathbf{F}_e^2 = Conv(Reshape(\mathbf{F}_e^1))$$

其中 Mamba 块定义为：$Mamba(x) = Lin(x' + x'')$，$x' = SiLU(Lin(x))$，$x'' = S6(SiLU(DWConv(Lin(x))))$，S6 为选择性 SSM。

**设计二：自适应融合机制**

三分支的输出通过可学习权重进行自适应加权融合，并加入残差连接：

$$\mathbf{F}_{out} = \omega_a \cdot \mathbf{F}_a^2 + \omega_f \cdot \mathbf{F}_f^2 + \omega_e \cdot \mathbf{F}_e^2 + \mathbf{F}_{in}$$

其中 $\omega_a, \omega_f, \omega_e$ 随机初始化并通过反向传播更新。与简单的均匀融合或串行融合不同，自适应权重允许模型动态调整三种感知的重要性。

**设计三：多尺度 U-Net 集成**

将 MPF Block 嵌入 U-Net 对称结构，通过下采样-上采样和跳跃连接实现全局、中间、局部三尺度的特征提取与融合，兼顾全局语义一致性和局部纹理细节。

### 损失函数 / 训练策略
使用 MAE（平均绝对误差）作为损失函数：

$$L = \frac{1}{H \times W \times C} \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{k=1}^{C} |Z_{i,j,k} - \hat{Z}_{i,j,k}|$$

训练采用 Adam 优化器（$\beta_1=0.9, \beta_2=0.999$），初始学习率 0.0004，余弦退火调度 100 epochs。数据增强包括随机旋转和翻转，批大小 32，裁剪 128×128 patch。单卡 NVIDIA 4090 训练。

## 实验关键数据

### 主实验

**NTIRE2022 & CAVE 数据集结果：**

| 方法 | Params(M) | FLOPs(G) | NTIRE2022 PSNR↑ | NTIRE2022 RMSE↓ | CAVE PSNR↑ | CAVE RMSE↓ |
|------|-----------|----------|-----------------|----------------|-----------|-----------|
| HSCNN+ | 1.642 | 808.0 | 25.26 | 0.058 | 33.81 | 0.0227 |
| HRNet | 31.705 | 1249.0 | 25.22 | 0.0577 | 34.53 | 0.0205 |
| MST++ | 1.62 | 177.7 | 30.18 | 0.035 | 34.65 | 0.0205 |
| GMSR | 0.019 | 8.0 | 26.92 | 0.0492 | 34.58 | 0.0206 |
| **M3SR** | **2.166** | **100.9** | **31.40** | **0.0343** | **35.61** | **0.0184** |

**NTIRE2020 数据集结果：**

| 方法 | NTIRE2020-Clean PSNR↑ | NTIRE2020-Real PSNR↑ | Clean RMSE↓ | Real RMSE↓ |
|------|----------------------|---------------------|------------|------------|
| MST++ | 36.32 | 35.63 | 0.0198 | 0.0185 |
| HSRNet | 37.17 | 34.55 | 0.0198 | 0.0213 |
| GMSR | 33.97 | 31.90 | 0.0239 | 0.0278 |
| **M3SR** | **37.71** | **36.35** | **0.0196** | **0.0171** |

M3SR 在 NTIRE2022 上 PSNR 达到 31.40dB（超越 MST++ 的 30.18dB 约 1.2dB），同时 FLOPs 仅 100.9G（MST++ 为 177.7G），参数量仅 2.166M。

### 消融实验

**多感知分支消融（NTIRE2022）：**

| 变体 | 空间 | 频率 | 光谱 | PSNR↑ | RMSE↓ | SAM↓ | MSSIM↑ |
|------|------|------|------|-------|-------|------|--------|
| M3SR-V1 | ✗ | ✓ | ✓ | 30.49 | 0.0381 | 12.55 | 0.8827 |
| M3SR-V2 | ✓ | ✗ | ✓ | 30.59 | 0.0365 | 6.52 | 0.9315 |
| M3SR-V3 | ✓ | ✓ | ✗ | 30.36 | 0.0369 | 6.28 | 0.9241 |
| **M3SR** | **✓** | **✓** | **✓** | **31.40** | **0.0343** | **6.62** | **0.9351** |

移除空间分支导致 SAM 急剧恶化至 12.55（完整模型为 6.62），PSNR 下降约 0.9dB。

**分组数 G 消融：**

| G | PSNR↑ | RMSE↓ | Params(M) | FLOPs(G) |
|---|-------|-------|-----------|----------|
| 2 | 31.23 | 0.0351 | 2.066 | 91.3 |
| **4** | **31.40** | **0.0343** | **2.166** | **100.9** |
| 8 | 30.64 | 0.0372 | 2.368 | 120.1 |
| 16 | 30.76 | 0.0361 | 2.770 | 158.6 |

G=4 在性能和效率之间取得最佳平衡。

### 关键发现
- 空间感知分支对光谱角度（SAM）影响最大，移除后 SAM 从 6.62 飙升至 12.55
- 频率感知分支对 MSSIM 贡献显著，移除后 MSSIM 从 0.9351 降至 0.9315
- M3SR 以 100.9G FLOPs 超越 FLOPs 高达 5819.5G 的 FMNet，效率提升约 57 倍

## 亮点与洞察
- **三分支并行设计**将空间（SS2D）、频率（DWT）、光谱（原始 Mamba）三种互补感知统一在一个模块中，是对 Mamba 在底层视觉任务中应用的系统性扩展
- **自适应加权融合**比简单拼接或求和更灵活，允许不同尺度和数据集上动态调整各感知的贡献
- 在保持极低计算开销的同时取得 SOTA，特别适合资源受限的高光谱应用场景

## 局限与展望
- 仅使用 MAE 损失，未探索感知损失、频率域损失等可能进一步提升重建质量的策略
- 消融仅在 NTIRE2022 上进行，缺乏跨数据集的消融验证
- DWT 固定使用 Haar 小波，未探索其他小波基的影响
- 光谱分组数 G 作为超参数需要手动调节，缺乏自适应选择机制

## 相关工作与启发
- **vs MST++**: M3SR 在 PSNR 上超越 1.2dB 的同时 FLOPs 仅为其 56.8%，证明 Mamba 的线性复杂度优势
- **vs GMSR**: 同为 Mamba SR 方法，但 GMSR 仅用单一空间 SSM（PSNR 26.92），M3SR 通过多感知融合提升至 31.40，增幅 4.5dB
- **vs HRNet**: HRNet 参数量 31.7M 远超 M3SR 的 2.17M（约 15 倍），但 PSNR 仅 25.22 vs 31.40

## 评分
- 新颖性: ⭐⭐⭐⭐ 三分支多感知 Mamba 融合是对 SSM 底层视觉应用的系统性创新
- 实验充分度: ⭐⭐⭐⭐ 四个数据集、十种 SOTA 对比、完整消融实验和参数分析
- 写作质量: ⭐⭐⭐ 结构清晰但部分公式符号不够统一
- 价值: ⭐⭐⭐⭐ 为高光谱重建提供了高效实用的 Mamba 方案，有开源代码

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](../../CVPR2026/remote_sensing/exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)
- [\[CVPR 2026\] Regulating Rather than Constraining: Adaptive Guidance for Complex Spectral Reconstruction in Pansharpening](../../CVPR2026/remote_sensing/regulating_rather_than_constraining_adaptive_guidance_for_complex_spectral_recon.md)
- [\[AAAI 2026\] Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments](consistency-based_abductive_reasoning_over_perceptual_errors_of_multiple_pre-tra.md)
- [\[CVPR 2025\] MFogHub: Bridging Multi-Regional and Multi-Satellite Data for Global Marine Fog Detection and Forecasting](../../CVPR2025/remote_sensing/mfoghub_bridging_multi-regional_and_multi-satellite_data_for_global_marine_fog_d.md)
- [\[CVPR 2026\] PhenoYieldNet: Learning Crop-Aware Phenological Responses for Multi-Crop Yield Prediction](../../CVPR2026/remote_sensing/phenoyieldnet_learning_crop-aware_phenological_responses_for_multi-crop_yield_pr.md)

</div>

<!-- RELATED:END -->
