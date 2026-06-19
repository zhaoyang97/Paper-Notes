---
title: >-
  [论文解读] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior
description: >-
  [ICCV 2025][稀疏视角重建] 提出Neural Volumetric Prior (NVP)，通过融合显式3D特征网格与隐式MLP的混合神经表示，结合基于衍射光学的物理渲染方程，首次实现了从稀疏视角（仅6-7张荧光图像）对半透明生物样本3D折射率的高保真体积重建，所需图像数量减少约50倍、处理时间缩短3倍。
tags:
  - "ICCV 2025"
  - "稀疏视角重建"
  - "神经体积先验"
  - "衍射光学"
  - "折射率重建"
  - "荧光衍射断层扫描"
---

# Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior

**会议**: ICCV 2025  
**arXiv**: [2510.16391](https://arxiv.org/abs/2510.16391)  
**代码**: 无  
**领域**: 其他/计算成像  
**关键词**: 稀疏视角重建, 神经体积先验, 衍射光学, 折射率重建, 荧光衍射断层扫描  

## 一句话总结
提出Neural Volumetric Prior (NVP)，通过融合显式3D特征网格与隐式MLP的混合神经表示，结合基于衍射光学的物理渲染方程，首次实现了从稀疏视角（仅6-7张荧光图像）对半透明生物样本3D折射率的高保真体积重建，所需图像数量减少约50倍、处理时间缩短3倍。

## 研究背景与动机

**领域现状**：光学断层扫描（Optical Tomography）通过多角度2D图像重建3D生物结构，是活细胞无标记成像的重要工具。荧光衍射断层扫描（FDT）利用样本内部荧光作为光源，无需光透射侧即可成像，适合活体成像。

**现有痛点**：
   - FDT需要**数百张2D图像**来重建一个3D体积，要求样本在成像过程中保持静止（秒到分钟级），无法捕捉快速动态过程（如心肌细胞收缩、胚胎发育）
   - 与自然场景的3D重建不同，生物样本是**半透明**的，需重建整个体积而非仅表面，未知体素数量大得多
   - 显微镜的数值孔径限制了可用视角范围，荧光源的空间分布进一步减少了可用角度，形成极端"稀疏视角"设置
   - 现有神经场方法基于**射线光学**（NeRF等），假设光沿直线传播，但微观尺度下衍射效应显著，射线光学模型不适用

**核心矛盾**：如何用极少量（~6张）图像重建包含大量未知体素的3D体积？物理模型需从射线光学升级到波动光学，同时神经表示需能在极稀疏数据下提供足够的正则化。

**切入角度**：设计一种混合神经表示（显式网格+隐式MLP），既保留显式表示的稀疏先验，又通过MLP捕获空间相关性来填补缺失信息，并结合衍射光学物理先验进行物理准确的渲染。

## 方法详解

### 整体框架
NVP的工作流程：(1) 定义随机初始化的3D特征网格 $W_{xyz}$，通过3层MLP $F_{\text{nvp}}$ 映射到折射率体积 $\hat{n}$；(2) 利用多层Born近似的衍射渲染方程，从预测的折射率和自校准的荧光源位置生成预测图像 $\hat{I}$；(3) 通过与真实图像的损失函数反向传播优化网格特征和MLP参数。

### 关键设计

1. **Neural Volumetric Prior（神经体积先验）**：

    - **显式网格** $W_{xyz}(x,y,z) \in \mathbb{R}^F$：在每个体素存储 $F$ 维可学习特征向量，提供直接的空间结构先验和稀疏性
    - **隐式MLP** $F_{\text{nvp}}$：3层全连接网络，将特征向量映射为标量折射率值，捕获显式网格中的隐含空间相关性
    - **混合表示**：$\hat{n}(x,y,z) = F_{\text{nvp}}(W_{xyz}(x,y,z))$
    - **自适应分辨率**：网格分辨率根据折射率的空间变化分布动态调整
    - 与其他表示的对比：纯显式（Plenoxel）缺乏空间相关性编码需要更多视角；纯隐式（NeRF-style MLP）计算效率低；Triplane低秩分解会引入网格伪影

2. **基于衍射光学的物理渲染方程**：

    - 将成像体积建模为 $N_z$ 个薄片，光场 $\hat{E}_{k,i}(\mathbf{r})$ 在每层薄片间传播：
    $\hat{E}_{k,i}(\mathbf{r}) = \mathcal{P}_{\Delta z}\{t_k(\mathbf{r}) \cdot \hat{E}_{k-1,i}(\mathbf{r})\}$
   其中 $\mathcal{P}_{\Delta z}$ 是传播算子，$t_k(\mathbf{r})$ 是第 $k$ 层的透射函数（与折射率相关）
    - 最终相机捕获的强度图像：$\hat{I}_i(\mathbf{r}) = |\hat{E}_{N_z,i}(\mathbf{r})|^2$
    - 通过GPU并行计算和预定义传播核实现高效渲染

3. **相干对齐与自校准**：

    - **相干掩码**：处理实验中部分相干/非相干荧光与模型中相干光的不匹配
    - **视点自校准**：通过高斯拟合从荧光图像估计荧光源位置，与MLP参数联合优化

### 损失函数
$$\mathcal{L} = \mathcal{L}_{\text{img}} + \tau \mathcal{R}_{\text{ri}}$$
- $\mathcal{L}_{\text{img}}$：结合L1、L2和SSIM的图像级损失
- $\mathcal{R}_{\text{ri}}$：折射率的全变分正则化（保持平滑性+保留细节）

## 实验

### 主实验：合成数据量化比较（不同光照数量）

| 方法 | 6张光照 PSNR/SSIM/LPIPS | 7张光照 PSNR/SSIM/LPIPS | 20张光照 PSNR/SSIM/LPIPS |
|:---|:---|:---|:---|
| Explicit | 28.62 / 0.854 / 0.182 | 28.73 / 0.847 / 0.178 | 28.88 / 0.865 / 0.119 |
| Triplane | 28.73 / 0.713 / 0.215 | 30.43 / 0.762 / 0.139 | 30.61 / 0.962 / 0.062 |
| **NVP** | **30.73 / 0.891 / 0.103** | **30.96 / 0.897 / 0.090** | **31.38 / 0.896 / 0.054** |

**关键发现**：NVP用6张图像即可达到Explicit和Triplane用20张图像的效果。从20张减少到6张，NVP的PSNR仅下降0.65dB。

### 真实生物样本实验（MDCK活细胞）

| 方法 | SSIM↑ | LPIPS↓ | PSNR↑ |
|:---|:---|:---|:---|
| Explicit | 0.9944 | 0.0051 | 36.65 |
| Triplane | 0.9762 | 0.0285 | 32.69 |
| **NVP** | **0.9977** | **0.0015** | **40.70** |

NVP在19张荧光图像下重建MDCK活细胞，PSNR比Explicit高4.05dB，比Triplane高8.01dB。Explicit方法出现不连续性和噪声，Triplane出现严重的网格伪影。NVP在20分钟内收敛到SSIM>0.99，而Explicit基线需要60分钟。

### 关键发现
1. NVP实现了约**50倍的测量减少**（从100+张到仅6-7张）和**3倍的处理速度提升**
2. 合成组织数据集上NVP的SSIM为0.4775，远高于Explicit的0.2954和Triplane的0.1323
3. NVP同时重建连续结构（血管）和稀疏结构（神经元），显示出对不同形态的鲁棒性
4. 自校准模块对重建质量有显著贡献（消融实验见附录）

## 亮点与洞察

1. **首次实现稀疏视角生物体积重建**：从衍射荧光图像仅用~6张即可重建3D折射率，开辟了实时动态生物成像的新可能
2. **物理驱动的神经表示设计**：将衍射光学物理先验融入神经场方法，解决了微观尺度下射线光学模型失效的问题
3. **混合表示的优势论证**：清晰展示了显式+隐式混合表示如何突破各自局限——显式提供空间稀疏先验防止过拟合，隐式提供空间相关性编码填补缺失信息

## 局限性

1. 目前仅验证了在FDT成像模态上的效果，未在其他光学成像系统上测试
2. 合成数据与真实数据之间仍存在domain gap，相干对齐仅是近似解决方案
3. 体积尺寸受GPU内存限制，对更大的3D体积可能需要分块处理
4. 折射率重建偏差在组织等复杂散射样本上仍然较大

## 相关工作

- **稀疏视角3D重建**：NeRF系列（RegNeRF, DietNeRF）、深度正则化、扩散先验等，但均基于射线光学
- **衍射光学3D重建**：Born近似、多层Born模型、CNN/隐式神经场的相位恢复，但都需要大量图像
- **混合神经表示**：K-Planes、TensoRF、Instant-NGP等，但缺少波动光学物理先验

## 评分
- 创新性：★★★★☆（物理先验+混合表示的组合创新，首次在FDT上实现稀疏视角重建）
- 实验充分度：★★★★☆（合成+真实实验充分，但baseline较少且都是自行实现）
- 实用价值：★★★★★（实现~50倍测量减少对活体动态成像意义重大）
- 写作质量：★★★★☆（物理模型阐述清晰，图示丰富）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images (MV-AR)](autoregressively_generating_multiview_consistent_images.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ICCV 2025\] A Linear N-Point Solver for Structure and Motion from Asynchronous Tracks](a_linear_n-point_solver_for_structure_and_motion_from_asynchronous_tracks.md)
- [\[ICCV 2025\] Φ-GAN: Physics-Inspired GAN for Generating SAR Images Under Limited Data](ph-gan_physics-inspired_gan_for_generating_sar_images_under_limited_data.md)

</div>

<!-- RELATED:END -->
