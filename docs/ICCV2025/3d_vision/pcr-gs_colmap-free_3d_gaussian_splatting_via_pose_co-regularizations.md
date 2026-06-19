---
title: >-
  [论文解读] PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations
description: >-
  [ICCV 2025][3D视觉][3D Gaussian Splatting] 提出 PCR-GS，通过 DINO 特征重投影正则化和基于小波变换的频率正则化对相机位姿进行协同约束，在无需 COLMAP 先验的条件下实现了复杂相机轨迹场景的高质量 3D-GS 重建与位姿估计。 3D Gaussian Splatting (…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "COLMAP-Free"
  - "相机位姿估计"
  - "DINO特征"
  - "小波变换"
  - "新视角合成"
---

# PCR-GS: COLMAP-Free 3D Gaussian Splatting via Pose Co-Regularizations

**会议**: ICCV 2025  
**arXiv**: [2507.13891](https://arxiv.org/abs/2507.13891)  
**作者**: Yu Wei, Jiahui Zhang, Xiaoqin Zhang, Ling Shao, Shijian Lu (NTU, 浙工大, UCAS)
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, COLMAP-Free, 相机位姿估计, DINO特征, 小波变换, 新视角合成

## 一句话总结

提出 PCR-GS，通过 DINO 特征重投影正则化和基于小波变换的频率正则化对相机位姿进行协同约束，在无需 COLMAP 先验的条件下实现了复杂相机轨迹场景的高质量 3D-GS 重建与位姿估计。

## 研究背景与动机

3D Gaussian Splatting (3D-GS) 在新视角合成中表现优异，但严重依赖 COLMAP 提供的精确相机位姿。COLMAP 本身计算量大、且在稀疏纹理或重复模式场景下容易失败。已有的无 COLMAP 方法（如 CF-3DGS）通过估计相邻帧间的相对位姿并利用 RGB 光度损失进行联合优化，但存在以下关键问题：

**复杂相机轨迹下的失效**：当相邻帧之间存在大幅旋转和平移时，相邻视角的重叠区域有限，基于 RGB 的对齐变得不可靠，导致相对位姿估计不准确。

**局部最优陷阱**：不准确的相对位姿会使位姿与 3D-GS 的联合优化陷入局部最优，产生严重的伪影和模糊。

**旋转矩阵优化难**：相机位姿中旋转矩阵的微小误差会导致几何结构和纹理的空间偏移，而 RGB 空间的正则化对这种结构性偏移不敏感。

这些问题在真实场景中非常常见——用户手持拍摄往往伴随大幅度的相机运动。PCR-GS 的动机就是从**语义特征对齐**和**频率域分析**两个互补视角对相机位姿进行协同正则化，突破复杂轨迹下的性能瓶颈。

## 方法详解

### 整体框架

PCR-GS 构建在 CF-3DGS 基础之上，保留其渐进式优化策略（逐帧扩展 3D 高斯），但引入两个核心正则化模块并行工作：

1. **特征重投影正则化 (Feature Reprojection Regularization, FRR)**：利用 DINO 语义特征进行跨视角对齐
2. **基于小波的频率正则化 (Wavelet-based Frequency Regularization, WFR)**：在频率域捕获旋转误差

总损失函数为：

$$\mathcal{L} = \lambda_0 \mathcal{L}_{\text{rgb}} + \lambda_1 \mathcal{L}_{\text{feat}} + \lambda_2 \mathcal{L}_{\text{freq}}$$

其中 $\lambda_0=0.6$，$\lambda_1=0.2$，$\lambda_2=0.2$。

### 关键设计一：特征重投影正则化 (FRR)

**核心思路**：DINO 特征对视角变化具有鲁棒性（即使剧烈旋转，DINO 特征的点对应关系仍然稳定），因此利用 DINO 语义特征替代不稳定的 RGB 信息来约束相对位姿。

**具体流程**：

- 用预训练 DINO 模型（取第 9 层特征）对每帧提取语义特征图 $F_i$、$F_{i+1}$
- 利用已训练的 3D 高斯 $G_i^*$ 渲染深度图，获取每个像素的深度值
- 将帧 $I_i$ 的 2D 像素通过深度反投影到 3D 相机坐标，再用变换矩阵 $T_i$ 转到帧 $I_{i+1}$ 的坐标系，最后投影到 2D
- 最小化原始位置和重投影位置上 DINO 特征的 L2 差异：

$$\mathcal{L}_{\text{feat}} = \|F_i\langle P_i\rangle - F_{i+1}\langle \mathbf{K} P_i T_i \rangle\|_2$$

**位姿初始化策略**：

- 利用 DINO 特征图的显著性图构建前景掩模，过滤背景
- 在前景关键点间用 Best Buddies 算法建立稀疏对应关系
- 随机选取 $N_s=20$ 个稀疏对应点优化初始相对位姿（代替单位矩阵初始化）
- 这一步有效降低了后续优化陷入局部最优的风险

### 关键设计二：基于小波的频率正则化 (WFR)

**核心思路**：旋转误差会导致边缘和纹理的空间偏移，在高频细节中体现明显。RGB 空间主要关注像素强度变化，对旋转引起的结构性偏移不敏感。频率域正则化能更好地捕获这类误差。

**具体流程**：

- 对渲染图和 GT 图施加小波变换，分解为 4 个分量：$LL$（低频）、$LH$（水平高频）、$HL$（垂直高频）、$HH$（对角高频）
- 计算各分量的加权欧氏距离：

$$d = \sum_{x \in \{LL, LH, HL, HH\}} w_x \|W_x(I_t) - W_x(\hat{I}_t)\|$$

**退火策略（Annealing）**：

- 直接优化高频分量容易引入噪声，因此设计从低频到高频的渐进策略：

$$\mathcal{L}_{\text{freq}} = \begin{cases} d_{LL} & 0 < n \leq 100 \\ (1-w_h)d_{LL} + w_h d_H & 100 < n \leq 200 \\ d_H & n > 200 \end{cases}$$

- 前 100 次迭代只优化低频，100-200 次迭代线性增加高频权重，200 次之后只优化高频
- 小波变换的优势在于高频细节保留空间位置信息，不同于 FFT 等全局频率变换

### 损失函数

- **$\mathcal{L}_{\text{rgb}}$**：标准光度损失 = $(1-\lambda)\|I - \hat{I}\| + \lambda \mathcal{L}_{\text{D-SSIM}}$，$\lambda=0.2$
- **$\mathcal{L}_{\text{feat}}$**：DINO 特征重投影 L2 损失
- **$\mathcal{L}_{\text{freq}}$**：小波频率域退火损失

## 实验关键数据

### 主实验：新视角合成（Tanks&Temples，8 场景均值）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| NeRFmm | 14.10 | 0.36 | 0.66 |
| BARF | 14.05 | 0.42 | 0.69 |
| Nope-NeRF | 21.95 | 0.57 | 0.52 |
| CF-3DGS | 19.79 | 0.60 | 0.33 |
| **PCR-GS** | **23.68** | **0.73** | **0.23** |

PCR-GS 在 PSNR 上比 CF-3DGS 高出 **+3.89 dB**，LPIPS 降低 30%。

### 主实验：新视角合成（Free-Dataset，3 场景均值）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| CF-3DGS | 15.00 | 0.37 | 0.55 |
| **PCR-GS** | **17.78** | **0.49** | **0.46** |

在 Free-Dataset 上 PSNR 提升 **+2.78 dB**。

### 主实验：位姿估计（Tanks&Temples，8 场景均值）

| 方法 | RPE_t↓ | RPE_r↓ | ATE↓ |
|------|--------|--------|------|
| NeRFmm | 8.261 | 1.950 | 0.446 |
| BARF | 7.641 | 2.121 | 0.436 |
| Nope-NeRF | 3.519 | 0.751 | 0.403 |
| CF-3DGS | 0.211 | 0.520 | 0.013 |
| **PCR-GS** | **0.109** | **0.350** | **0.008** |

位姿精度全面优于所有基线，RPE_t 比 CF-3DGS 降低 48%，ATE 降低 38%。

### 消融实验（Tanks&Temples - Horse 场景）

| 模型 | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|-------|-------|--------|
| Base (CF-3DGS) | 18.34 | 0.64 | 0.32 |
| Base+FRR | 23.16 | 0.72 | 0.17 |
| Base+WFR (无高频) | 18.40 | 0.65 | 0.32 |
| Base+WFR (有高频) | 19.31 | 0.66 | 0.28 |
| **Base+FRR+WFR (PCR-GS)** | **24.20** | **0.79** | **0.17** |

### 关键发现

1. **FRR 贡献最大**：单独加 FRR 就带来 +4.82 dB PSNR 提升，说明 DINO 特征对齐是核心改进
2. **WFR 的高频分量关键**：不使用高频分量时 WFR 几乎无效（+0.06 dB），使用后提升明显（+0.97 dB），证明高频细节确实能有效捕获旋转误差
3. **FRR+WFR 互补**：在 FRR 基础上加 WFR 又进一步提升 +1.04 dB，两者协同效果显著
4. **复杂轨迹下优势更大**：实验特意将 Tanks&Temples 采样率从 20fps 降至 4fps 以增加相机运动复杂度，CF-3DGS 性能急剧下降，而 PCR-GS 仍保持良好表现

## 亮点与洞察

1. **问题定义精准**：准确识别了 COLMAP-free 3D-GS 在复杂相机轨迹下的核心痛点——RGB 对齐在大视角变化下失效，并从语义特征和频率域两个互补角度提出解决方案
2. **DINO 特征的巧妙利用**：不是简单加入 DINO 损失，而是设计了完整的特征重投影流程（3D 反投影→变换→2D 投影→采样对齐），同时还利用 DINO 显著性图做初始化，形成一套完整的方案
3. **频率域退火策略**：从低频到高频渐进优化的设计很有工程洞察力，避免了高频噪声对训练的干扰
4. **实验设计考究**：没有沿用 CF-3DGS 过于理想化的数据设置（20fps 平滑轨迹），而是自行构建了更具挑战性的 4fps 采样数据，使评估更贴近真实应用场景
5. **小波变换的选择有道理**：相比 FFT 等全局频率变换，小波变换保留空间位置信息，更适合检测旋转导致的局部结构偏移

## 局限性

1. **仅针对视频序列**：方法基于相邻帧间的渐进式优化，不适用于无序图片集合的重建场景
2. **依赖单目深度估计**：使用 DPT 初始化点云，单目深度的不准确可能影响特征重投影质量
3. **DINO 特征提取开销**：每帧需要提取 DINO 特征，增加了计算和内存开销，但论文未提供具体的时间开销对比
4. **退火策略参数固定**：$n_0=100$、$n_1=200$ 是人为设定的超参数，不同场景是否需要调整未做讨论
5. **缺少与更新方法的对比**：未与 InstantSplat、DUSt3R 等近期无位姿重建方法对比
6. **场景规模有限**：实验主要在中等规模室内外场景，未验证大规模场景（如街景、城市级）的泛化能力

## 相关工作与启发

1. **CF-3DGS [Fu et al., CVPR 2024]**：本文的直接基线，首个无位姿 3D-GS 方法，渐进式估计相对位姿。PCR-GS 在其基础上增加了两种正则化
2. **BARF [Lin et al., ICCV 2021]**：渐进式位姿-NeRF 联合优化，采用从粗到细的位置编码策略。启发：coarse-to-fine 的思想在本文的退火策略中有所体现
3. **Nope-NeRF [Bian et al., CVPR 2023]**：利用无畸变深度先验约束相对位姿。在复杂轨迹下的 NVS 质量优于 CF-3DGS 但位姿精度较差
4. **DINO [Caron et al., ICCV 2021]**：自监督 ViT 特征对视角变化鲁棒。本文核心依赖 DINO 特征的跨视角一致性
5. **启发方向**：可以考虑将 DINOv2 或 SAM 等更强的基础模型特征用于位姿正则化；频率域正则化的思路可以扩展到动态场景中物体运动的检测与分离

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 两个正则化模块的设计各有巧思，DINO特征重投影和小波频率退火都不是简单堆叠
- **实验充分度**: ⭐⭐⭐⭐ — 11个场景、NVS+位姿双评估、详细消融；但缺少与最新方法对比和效率分析
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰、公式完整、图示直观，动机阐述充分
- **价值**: ⭐⭐⭐⭐ — 解决了无COLMAP 3D-GS在复杂轨迹下的实际痛点，有明确的应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] No Pose at All: Self-Supervised Pose-Free 3D Gaussian Splatting from Sparse Views](no_pose_at_all_self-supervised_pose-free_3d_gaussian_splatting_from_sparse_views.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[ECCV 2024\] CoR-GS: Sparse-View 3D Gaussian Splatting via Co-Regularization](../../ECCV2024/3d_vision/cor-gs_sparse-view_3d_gaussian_splatting_via_co-regularization.md)
- [\[CVPR 2025\] SelfSplat: Pose-Free and 3D Prior-Free Generalizable 3D Gaussian Splatting](../../CVPR2025/3d_vision/selfsplat_pose-free_and_3d_prior-free_generalizable_3d_gaussian_splatting.md)
- [\[ICCV 2025\] BezierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting](beziergs_dynamic_urban_scene_reconstruction_with_bezier_curve_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
