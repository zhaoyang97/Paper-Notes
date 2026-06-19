---
title: >-
  [论文解读] Dynamic Neural Radiance Field from Defocused Monocular Video
description: >-
  [ECCV 2024][3D视觉][动态NeRF] 提出 $D^2RF$，首个从散焦单目视频中恢复清晰动态NeRF的方法，通过将景深(DoF)渲染与体积渲染统一，引入分层DoF体积渲染来建模散焦模糊并恢复清晰新视角。 领域现状 动态NeRF从单目视频进行时空新视角合成已取得优秀成果，现有方法（NSFF、HyperNeRF、D…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "动态NeRF"
  - "散焦模糊"
  - "景深渲染"
  - "体积渲染"
  - "新视角合成"
---

# Dynamic Neural Radiance Field from Defocused Monocular Video

**会议**: ECCV 2024  
**arXiv**: [2407.05586](https://arxiv.org/abs/2407.05586)  
**代码**: [有](https://github.com/xianrui-luo/D2RF)  
**领域**: 3D视觉  
**关键词**: 动态NeRF, 散焦模糊, 景深渲染, 体积渲染, 新视角合成

## 一句话总结

提出 $D^2RF$，首个从散焦单目视频中恢复清晰动态NeRF的方法，通过将景深(DoF)渲染与体积渲染统一，引入分层DoF体积渲染来建模散焦模糊并恢复清晰新视角。

## 研究背景与动机

### 领域现状
动态NeRF从单目视频进行时空新视角合成已取得优秀成果，现有方法（NSFF、HyperNeRF、DVS等）通过建立时空一致性来实现高质量动态场景重建。

### 核心痛点
现有动态NeRF方法假设输入为全聚焦（all-in-focus）图像序列。然而实际视频拍摄中，由于场景深度变化、大光圈设定以及拍摄者对焦不稳定，**散焦模糊（defocus blur）** 几乎不可避免。散焦模糊会导致：
1. 缺乏清晰细节，干扰动态物体运动建模
2. 无法建立输入视角间的时间一致性
3. 当前动态NeRF方法在散焦输入下严重退化，无法恢复清晰内容

### 现有方案局限
- 静态场景去焦方法（Deblur-NeRF、DP-NeRF、DoF-NeRF）：仅针对静态多视角输入，不能处理动态场景
- 这些方法在NeRF最后一步（体积渲染之后）进行模糊建模——属于后处理方式
- 2D单图去模糊+动态NeRF的组合方案：各视角去模糊结果缺乏视角间一致性，效果不稳定

### 核心切入角度
作者发现 **DoF渲染中的层可见性（layer visibility）与体积渲染中的不透明度（opacity）具有相同的物理含义**——两者都描述了光线在某点被吸收/遮挡的程度。基于此，可以将DoF模糊过程无缝嵌入NeRF的体积渲染管线中，而非作为后处理。

## 方法详解

### 整体框架

$D^2RF$ 输入散焦单目视频，输出清晰的动态场景NeRF表示。核心流程：
1. 定义模糊核模板 → 通过MLP预测优化后的稀疏光线及权重
2. 将光线送入静态MLP $G_\theta^{st}$ 和动态MLP $G_\theta^{dy}$ 分别建模场景
3. 通过分层DoF体积渲染融合模糊并监督训练
4. 测试时直接渲染光线（不经过DoF模糊），输出清晰新视角

### 关键设计

#### 1. 分层DoF体积渲染（Layered DoF Volume Rendering）

**功能**：将散焦模糊的建模过程从后处理提升到体积渲染内部，实现模糊感知的NeRF训练。

**核心联系**：体积渲染中采样点的 alpha 值 $\alpha_i = 1 - \exp(-\sigma\delta_i)$ 表示该点的不透明度，与DoF渲染中的层可见性 $W_i$ 具有相同的物理意义。且NeRF的离散采样方式与DoF渲染的层离散化天然兼容。

将传统DoF渲染公式从图像层级转换为单像素/单光线层级后，提出分层DoF体积渲染公式：

$$\hat{C}_{dof}(\mathbf{r}_p) = \frac{\sum_{i=1}^{k} (T_i * K(\mathbf{r}))(1-\exp(-\sigma\delta_i)) \mathbf{c}(\mathbf{r}(t_i), \mathbf{d}) * K(\mathbf{r})}{\sum_{i=1}^{k} (T_i * K(\mathbf{r}))(1-\exp(-\sigma\delta_i)) * K(\mathbf{r})}$$

其中 $T_i$ 为累积透射率，$K(\mathbf{r})$ 为光线级模糊核。

**设计动机**：将模糊建模深度集成到体积渲染采样过程中，而非事后模糊，使NeRF网络能从散焦输入中学习清晰场景表示。

#### 2. 光线级优化稀疏核（Ray-based Optimized Sparse Kernel）

**功能**：将DoF渲染的层级核 $K(\gamma_i)$ 转换为光线级核 $K(\mathbf{r})$，并用稀疏点替代密集核以降低计算成本。

**核心思路**：使用MLP $G_\theta^k$ 预测核点的偏移和权重：

$$(\Delta\mathbf{j}, g_j) = G_\theta^k((u,v), \mathbf{j}, t_l)$$

其中 $(u,v)$ 为核中心的平面坐标，$\mathbf{j}$ 为核模板原始光线，$t_l$ 为时间嵌入。最终优化光线为 $\mathbf{r}_j = \mathbf{j} + \Delta\mathbf{j}$。

稀疏核卷积：$\mathbf{b}_p = \sum_{j \in S(p)} \mathbf{c}_j g_j$，约束 $\sum g_j = 1$ 保证亮度一致性。

**设计动机**：DoF渲染原本基于图像层，NeRF基于光线采样，需要将核从层级转换为光线级。稀疏核（5个点、半径10）大幅降低了计算开销，同时通过可学习的变形适应真实世界的空间变化模糊。

#### 3. 动态-静态场景融合与跨时间渲染

**功能**：用两个独立MLP分别建模静态和动态场景，并通过跨时间渲染建立时间一致性。

- **静态MLP** $G_\theta^{st}$：输出颜色 $\mathbf{c}$、密度 $\sigma$ 和混合权重 $\eta$
- **动态MLP** $G_\theta^{dy}$：输出颜色 $\mathbf{c}_t$、密度 $\sigma_t$、场景流 $f_t$ 和遮挡权重 $\mathcal{W}_t$
- 融合渲染通过 $\eta(t)$ 加权静态和动态颜色
- **跨时间渲染**：利用场景流将相邻帧的采样点变形到目标帧，通过分层DoF体积渲染计算跨时间颜色，建立时间一致性

### 损失函数 / 训练策略

总损失包含：
- **混合渲染损失** $\mathcal{L}_{color}^b$：融合结果与GT的L2损失
- **动态渲染损失** $\mathcal{L}_{color}^t$：约束单独的动态渲染结果
- **跨时间损失** $\mathcal{L}_{cross}$：相邻帧变形渲染与GT的加权L2损失，权重由遮挡置信度控制
- **数据先验损失** $\mathcal{L}_{data}$：尺度不变单目深度损失 + 场景流一致性和L1正则

训练细节：Adam优化器，学习率 $5 \times 10^{-4}$，每场景 250k 迭代，单卡 RTX 3090 约两天。使用 COLMAP 估计相机参数，RAFT 和 DPT 提供光流和深度先验。

## 实验关键数据

### 数据集
从 VDW 立体数据集收集8个动态场景，使用 BokehMe 管线合成散焦模糊。图像分辨率 $940 \times 360$，焦距沿场景视差渐变，模拟真实对焦过程。左视角散焦图像用于训练，右视角清晰图像用于评估。

### 主实验

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| DVS [Gao et al.] | 25.43 | 0.764 | 0.242 |
| RoDynRF [Liu et al.] | 26.18 | 0.770 | 0.227 |
| HyperNeRF [Park et al.] | 26.96 | 0.780 | 0.208 |
| NSFF [Li et al.] | 27.01 | 0.803 | 0.209 |
| [Lee]+RoDynRF (2D去模糊) | 25.79 | 0.776 | 0.196 |
| [Lee]+DVS (2D去模糊) | 24.52 | 0.757 | 0.208 |
| **$D^2RF$ (Ours)** | **27.30** | **0.816** | **0.130** |

$D^2RF$ 在所有指标上全面超越，LPIPS 提升尤为显著（0.130 vs 次优0.196），说明感知质量大幅改善。2D去模糊预处理反而可能引入视角间不一致而降低性能。

### 消融实验

| 配置 | PSNR↑ (全图) | SSIM↑ | LPIPS↓ | 说明 |
|------|-------------|-------|--------|------|
| w/o cross-time | 22.61 | 0.725 | 0.232 | 跨时间渲染对时间一致性至关重要 |
| w/o layered volume | 27.11 | 0.811 | 0.211 | 分层体积渲染带来更精确的模糊建模 |
| w/o optimized kernel | 27.25 | 0.795 | 0.216 | 优化核提供高效的模糊拟合管线 |
| w/o static | 26.20 | 0.769 | 0.177 | 独立静态表示稳定训练 |
| **Full (Ours)** | **27.30** | **0.816** | **0.130** | 所有组件协同达到最佳 |

### 关键发现

1. 去除跨时间渲染导致 PSNR 骤降 4.7dB（动态区域），说明时间一致性对动态场景至关重要
2. 分层DoF体积渲染比后处理式模糊建模更准确，LPIPS从0.211降至0.130
3. 3D空间中建模散焦模糊（本文）显著优于2D逐帧去模糊+动态NeRF的两阶段方案
4. 2D去模糊预处理有时反而使性能更差（RoDynRF+去模糊 < RoDynRF原始），因为各帧独立去模糊破坏了多视角一致性

## 亮点与洞察

1. **优雅的理论联系**：发现DoF渲染层可见性与体积渲染不透明度的等价性，自然地将两个渲染框架统一
2. **从后处理到内嵌**：将模糊建模从渲染后挪到渲染过程中，是范式性的改进
3. **问题定义价值**：首次定义并解决"散焦动态NeRF"这一实际且重要的问题

## 局限与展望

1. 无法处理极端散焦模糊
2. 训练耗时长（单场景约2天），推理速度慢（13秒/帧）
3. 合成数据集评估，未在真实散焦视频上验证
4. 依赖 COLMAP 相机参数估计，散焦模糊可能影响特征匹配精度
5. 可探索与3D Gaussian Splatting结合以加速

## 相关工作与启发

- **Deblur-NeRF** (CVPR 2022)：提出可变形稀疏核建模散焦，但仅限静态场景
- **NSFF** (CVPR 2021)：场景流方法建立动态NeRF时间一致性，本文在其基础上加入模糊处理
- **BokehMe** (ECCV 2022)：高质量散焦合成，本文用其构建数据集
- 启发：将物理成像过程嵌入神经渲染管线是处理退化输入的有效范式

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次统一DoF渲染与NeRF体积渲染，问题定义新颖
- **实验充分度**: ⭐⭐⭐ — 8场景合成数据集充分消融，但缺少真实数据验证
- **写作质量**: ⭐⭐⭐⭐ — 理论推导清晰，图示直观，公式与直觉解释并行
- **价值**: ⭐⭐⭐⭐ — 解决实际拍摄中不可避免的散焦问题，具有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Dynamic Gaussian Splatting from Defocused and Motion-blurred Monocular Videos](../../NeurIPS2025/3d_vision/dynamic_gaussian_splatting_from_defocused_and_motion-blurred_monocular_videos.md)
- [\[ECCV 2024\] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)
- [\[ECCV 2024\] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting](taming_latent_diffusion_model_for_neural_radiance_field_inpainting.md)
- [\[ECCV 2024\] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)
- [\[ECCV 2024\] Lagrangian Hashing for Compressed Neural Field Representations](lagrangian_hashing_for_compressed_neural_field_representations.md)

</div>

<!-- RELATED:END -->
