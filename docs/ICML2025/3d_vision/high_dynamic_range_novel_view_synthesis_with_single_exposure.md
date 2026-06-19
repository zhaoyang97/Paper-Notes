---
title: >-
  [论文解读] High Dynamic Range Novel View Synthesis with Single Exposure
description: >-
  [ICML 2025][3D视觉][HDR新视角合成] 首次提出仅使用单曝光LDR图像进行HDR新视角合成（HDR-NVS）的问题设定，并设计了一个基于相机成像原理的元算法框架Mono-HDR-3D，通过LDR→HDR颜色转换器（L2H-CC）和HDR→LDR闭环转换器（H2L-CC）实现无HDR监督下的HDR场景建模。
tags:
  - "ICML 2025"
  - "3D视觉"
  - "HDR新视角合成"
  - "单曝光"
  - "相机成像建模"
  - "NeRF"
  - "3D高斯溅射"
---

# High Dynamic Range Novel View Synthesis with Single Exposure

**会议**: ICML 2025  
**arXiv**: [2505.01212](https://arxiv.org/abs/2505.01212)  
**代码**: [github.com/prinasi/Mono-HDR-3D](https://github.com/prinasi/Mono-HDR-3D)  
**领域**: 3D视觉  
**关键词**: HDR新视角合成, 单曝光, 相机成像建模, NeRF, 3D高斯溅射

## 一句话总结

首次提出仅使用单曝光LDR图像进行HDR新视角合成（HDR-NVS）的问题设定，并设计了一个基于相机成像原理的元算法框架Mono-HDR-3D，通过LDR→HDR颜色转换器（L2H-CC）和HDR→LDR闭环转换器（H2L-CC）实现无HDR监督下的HDR场景建模。

## 研究背景与动机

HDR新视角合成的目标是从LDR图像集建立3D场景的HDR模型，生成任意视角的HDR渲染图像。现有方法（HDR-NeRF、HDR-GS）均依赖**多曝光LDR图像**作为训练数据，存在以下固有缺陷：

**运动伪影**：长曝光帧会累积物体/相机运动导致模糊，不同曝光之间的位移产生鬼影

**对齐困难**：不同曝光时间导致亮度分布、局部对比度差异，增加配准难度

**采集成本高**：需要专业设备、多次拍摄，在动态环境或移动设备上难以实现

作者提出更实用也更具挑战性的新任务：**单曝光HDR-NVS**——仅使用单一曝光时间的LDR图像进行训练。核心难点在于单曝光图像必然存在过曝或欠曝区域，信息不完整，无法直接恢复HDR内容。

## 方法详解

### 整体框架

Mono-HDR-3D是一个**元算法（meta-algorithm）**，可无缝集成到NeRF或3DGS等任意NVS模型上。整体流程分三个阶段：

1. **LDR 3D场景建模**：以单曝光LDR图像+相机位姿为输入，训练一个标准的LDR 3D场景模型（NeRF/3DGS）
2. **LDR→HDR提升**：通过L2H-CC（LDR-to-HDR Color Converter）将LDR颜色空间提升到HDR
3. **HDR→LDR闭环**：通过H2L-CC（HDR-to-LDR Color Converter）将HDR图像转回LDR，形成闭环，实现无HDR标签下的自监督训练

关键设计思想：**先建LDR模型再提升到HDR**，而非直接从单曝光LDR尝试建HDR模型（会失败），这是与先前方法相反的设计路线。

### 关键设计

#### 相机成像机制建模

方法的核心创新在于将L2H-CC和H2L-CC的网络结构设计建立在**物理成像公式**之上。

**LDR图像形成公式**（从HDR到LDR的正向过程）：

$$I^l = \frac{\Delta t}{g} \cdot I^h + I_0 + \epsilon - I_{\text{overflow}}$$

其中$\Delta t$为曝光时间，$g$为传感器增益，$I^h$为HDR像素值，$I_0$为暗电流偏移，$\epsilon$为传感器噪声，$I_{\text{overflow}}$为饱和溢出值。该公式可统一描述饱和与非饱和像素的成像过程，整理为两个功能项：

- $D(\cdot)$：线性缩放HDR亮度到LDR范围
- $B(\cdot)$：学习LDR亮度的偏移与校正

**逆向公式**（从LDR到HDR的反向过程）：

$$I^h = \underbrace{\frac{g}{\Delta t}}_{X(\cdot)} \cdot \underbrace{(I^l - I_0 + I_{\text{overflow}})}_{S(\cdot)} - \underbrace{\frac{g}{\Delta t} \cdot \epsilon}_{Y(\cdot)}$$

分解为三个功能项：
- $X(\cdot)$：线性放大因子，将LDR亮度线性映射到HDR范围
- $S(\cdot)$：偏移校正，调整放大后的LDR值
- $Y(\cdot)$：噪声校正项

#### L2H-CC（LDR→HDR颜色转换器）

L2H-CC是**逐通道（per-channel）** 操作，网络结构严格模拟逆向公式的三个项：

1. **输入映射**：线性层 + ReLU，将LDR颜色嵌入潜在特征空间
2. **三分支模拟**：

    - $X(\cdot)$分支：MLP + ReLU（确保非负，符合物理约束）
    - $S(\cdot)$分支：MLP + ReLU（非负校正值）
    - $Y(\cdot)$分支：MLP **无激活函数**（噪声本质随机，无非负约束）
3. **残差连接**：LDR输入通过残差结构与转换输出相加，保留精细颜色细节，稳定学习过程

#### H2L-CC（HDR→LDR颜色转换器，闭环设计）

H2L-CC模拟正向成像公式，将渲染的HDR图像转回LDR，使得即使没有HDR真值也能通过与LDR训练图像比较进行监督学习：

1. **输入映射**：线性层 + ReLU
2. **双分支模拟**：

    - $D(\cdot)$分支：线性层 + ReLU（非负线性缩放）
    - $B(\cdot)$分支：线性层 + Tanh（偏移校正，允许正负值）
3. **输出映射**：Sigmoid激活，将值约束到[0,1]的LDR范围

### 损失函数 / 训练策略

总体损失函数：

$$\mathcal{L} = \mathcal{L}_{\text{ldr}} + \alpha \mathcal{L}_{\text{hdr}} + \beta \mathcal{L}_{\text{h2l}}$$

**Mono-HDR-GS实例化**（集成3DGS时）：

- $\mathcal{L}_{\text{ldr}}$：L1损失 + D-SSIM损失（标准3DGS损失），权重$\lambda$平衡
- $\mathcal{L}_{\text{hdr}}$：$\mu$-law域下的L2损失，对HDR值做对数压缩后计算
- $\mathcal{L}_{\text{h2l}}$：与$\mathcal{L}_{\text{ldr}}$同样形式，但作用在H2L-CC输出上

**Mono-HDR-NeRF实例化**（集成NeRF时）：三个损失均使用MSE。

**超参数**：$\alpha=0.6$，$\beta=0.01$（NeRF）/$0.05$（3DGS）；L2H-CC学习率$5 \times 10^{-4}$，H2L-CC学习率$1 \times 10^{-3}$。

**关键：在纯单曝光设定下$\alpha=0$**（无HDR真值），此时H2L-CC的闭环设计成为唯一额外监督信号。

## 实验关键数据

### 主实验

数据集：8个合成场景（Blender）+ 4个真实场景。每个场景35张图，5个曝光时间。单曝光设定下随机选1个曝光训练。评估指标：PSNR↑ / SSIM↑ / LPIPS↓。

**合成数据集结果（LDR + HDR NVS）**：

| 方法 | 速度(fps) | LDR-PSNR↑ | LDR-SSIM↑ | LDR-LPIPS↓ | HDR-PSNR↑ | HDR-SSIM↑ | HDR-LPIPS↓ |
|------|----------|-----------|-----------|------------|-----------|-----------|------------|
| HDR-NeRF | 0.26 | 30.62 | 0.658 | 0.285 | 13.76 | 0.511 | 0.443 |
| **Mono-HDR-NeRF** | 0.26 | **38.78** | **0.936** | **0.048** | **32.86** | **0.940** | **0.068** |
| HDR-GS | 147.45 | 39.48 | 0.977 | 0.018 | 35.30 | 0.965 | 0.030 |
| **Mono-HDR-GS** | 136.97 | **41.68** | **0.983** | **0.009** | **38.57** | **0.975** | **0.012** |

**真实数据集结果（LDR NVS，无HDR真值）**：

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| HDR-NeRF | 32.50 | 0.948 | 0.069 |
| Mono-HDR-NeRF | 32.52 | 0.948 | 0.069 |
| HDR-GS | 35.34 | 0.966 | 0.019 |
| **Mono-HDR-GS** | **35.81** | **0.967** | **0.017** |

### 消融实验

**模块设计消融**（合成数据，HDR NVS指标）：

| 配置 | HDR-PSNR | HDR-SSIM | HDR-LPIPS | 说明 |
|------|----------|----------|-----------|------|
| L2H-CC→MLP替换 | 19.02 | 0.778 | 0.327 | L2H-CC是核心，替换后大幅下降 |
| H2L-CC→MLP替换 | 38.43 | 0.974 | 0.015 | 略有下降，闭环设计有正面作用 |
| 完整模型 | **38.57** | **0.975** | **0.012** | - |

**损失组合消融**（合成数据，HDR NVS指标）：

| 配置 | HDR-PSNR | HDR-SSIM | HDR-LPIPS | 说明 |
|------|----------|----------|-----------|------|
| 仅$\mathcal{L}_{\text{ldr}}$ | - | - | - | 无法训练 |
| 仅$\mathcal{L}_{\text{hdr}}$ | 33.93 | 0.925 | 0.050 | 基础可用 |
| $\mathcal{L}_{\text{ldr}}+\mathcal{L}_{\text{hdr}}$ | 38.19 | 0.974 | 0.015 | LDR提供几何正则化 |
| 全部三个损失 | **38.57** | **0.975** | **0.012** | 闭环贡献+0.38dB |

### 关键发现

1. **HDR-NeRF在单曝光下几乎失效**：HDR PSNR仅13.76dB（趋近全黑/全白输出），证明多曝光方法无法直接迁移到单曝光设定
2. **Mono-HDR-NeRF vs HDR-NeRF**：HDR PSNR提升+19.1dB（13.76→32.86），为巨大的质量飞跃
3. **Mono-HDR-GS vs HDR-GS**：HDR PSNR提升+3.27dB（35.30→38.57），在已经很强的基线上仍有显著提升
4. **效率无损**：Mono-HDR-GS的推理速度（136.97fps）与HDR-GS（147.45fps）基本持平
5. **LDR/HDR比例鲁棒性**：即使LDR:HDR=5:1，Mono-HDR-GS仍保留92.6%的峰值PSNR

## 亮点与洞察

1. **问题定义巧妙**：首次将单曝光HDR-NVS作为独立问题提出，降低了数据采集要求同时消除了多曝光固有缺陷
2. **物理驱动的网络设计**：L2H-CC和H2L-CC的结构直接从相机成像公式推导而来，每个网络分支对应一个物理项，这比黑盒MLP显著更有效（消融证实MLP替换L2H-CC后PSNR暴降19dB+）
3. **闭环自监督思想**：H2L-CC闭环设计使得仅用LDR图像也能间接监督HDR空间学习，是一种优雅的无标签学习策略
4. **元算法设计**：作为即插即用模块可以集成到任意NVS backbone中，已验证NeRF和3DGS两种实例化

## 局限与展望

1. **真实场景提升有限**：在真实数据上LDR NVS指标提升微弱（PSNR仅+0.47dB），说明方法在复杂真实光照条件下优势缩小
2. **单曝光信息天花板**：严重过曝/欠曝区域的信息本质上已丢失，仅靠网络学习难以真正恢复
3. **评估局限**：合成数据的HDR真值来自渲染器，可能与真实HDR有分布差异
4. **未探索视频/动态场景**：框架目前仅处理静态场景的多视角图像
5. **可扩展到更多backbone**：论文仅验证了NeRF和3DGS，可考虑集成到Instant-NGP、Zip-NeRF等更高效模型

## 相关工作与启发

- **HDR-NeRF (CVPR 2022)**：首个HDR-NVS方法，基于NeRF学习辐射到HDR颜色的隐式映射，但需多曝光且训推昂贵
- **HDR-GS (NeurIPS 2024)**：基于3DGS的HDR-NVS，效率大幅提升（1000×加速），仍依赖多曝光
- **单图HDR重建**（Eilertsen 2017, DCDR-UNet 2024）：2D图像的LDR→HDR转换，缺乏3D一致性
- **启发**：物理启发的网络架构设计（而非纯数据驱动）在先验信息有限的场景下极为有效；闭环设计思路可推广到其他缺乏对应标签的3D重建任务

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4.5 | 首次提出单曝光HDR-NVS问题，物理驱动闭环设计新颖 |
| 技术深度 | 4.0 | 公式推导严谨，模块设计有物理依据 |
| 实验充分度 | 4.0 | 合成+真实数据，多角度消融，但真实场景实验偏弱 |
| 实用性 | 4.5 | 即插即用元算法，大幅降低数据采集要求 |
| 写作质量 | 4.0 | 结构清晰，动机阐述充分 |
| **总分** | **4.2** | 问题定义有价值，方法设计优雅，实验有说服力 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SeHDR: Single-Exposure HDR Novel View Synthesis via 3D Gaussian Bracketing](../../ICCV2025/3d_vision/sehdr_single-exposure_hdr_novel_view_synthesis_via_3d_gaussian_bracketing.md)
- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](../../ICLR2026/3d_vision/dynamic_novel_view_synthesis_in_high_dynamic_range.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](../../CVPR2025/3d_vision/instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[CVPR 2025\] Dual Exposure Stereo for Extended Dynamic Range 3D Imaging](../../CVPR2025/3d_vision/dual_exposure_stereo_for_extended_dynamic_range_3d_imaging.md)
- [\[NeurIPS 2025\] NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods](../../NeurIPS2025/3d_vision/nerfbaselines_consistent_and_reproducible_evaluation_of_novel_view_synthesis_met.md)

</div>

<!-- RELATED:END -->
