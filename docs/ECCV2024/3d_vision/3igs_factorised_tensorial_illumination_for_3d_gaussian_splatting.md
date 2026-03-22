# 3iGS: Factorised Tensorial Illumination for 3D Gaussian Splatting

**会议**: ECCV 2024  
**arXiv**: [2408.03753](https://arxiv.org/abs/2408.03753)  
**作者**: Zhe Jun Tang, Tat-Jen Cham (NTU S-Lab)  
**代码**: [https://github.com/TangZJ/3iGS](https://github.com/TangZJ/3iGS) (有)  
**领域**: 3D视觉 / 新视角合成  
**关键词**: 3D Gaussian Splatting, 光照建模, BRDF, 张量分解, 视角依赖效果  

## 一句话总结
3iGS 用基于张量分解的连续入射光照场替代 3DGS 中每个高斯体独立优化的球谐系数，结合可学习 BRDF 特征和轻量神经渲染器来建模出射辐射，在保持实时渲染速度的同时显著提升了镜面反射等视角依赖效果的渲染质量。

## 研究背景与动机

1. **领域现状**：3D Gaussian Splatting (3DGS) 已成为从多视角图像实时重建和渲染3D场景的主流方法。它通过为每个高斯体独立优化球谐系数 (SH) 来表示出射辐射，结合基于 tile 的光栅化实现实时渲染。

2. **现有痛点**：3DGS 为每个高斯体**独立**优化 SH 系数来描述出射辐射，这种做法完全忽略了场景中相邻高斯体之间的光照交互：一个高斯体的颜色应该受周围物体反射光的影响，但当前框架无法建模这种场景级的间接光照。这导致在具有镜面反射和光泽表面的场景中，渲染效果不佳——反射高光缺乏真实感，表面颜色随视角变化不够准确。

3. **核心矛盾**：一方面希望像 PBR 那样准确建模 BRDF 和光照来产生真实的视角依赖效果；另一方面，从多视角图像精确估计物理材质参数（roughness, albedo, metallicity 等）本身是严重的病态逆问题（ill-posed inverse rendering），现有方法（如 GaussianShader 直接预测 Cook-Torrance BRDF 参数）精度有限，反而引入了误差。

4. **本文要解决什么？** 如何在不依赖精确物理参数估计的前提下，为 3DGS 引入场景级光照建模以提升视角依赖效果？

5. **切入角度**：受游戏引擎中 irradiance volume 和 light probe 的启发——它们不精确求解物理方程，而是在空间中预计算/存储光照信息，渲染时通过插值快速查询。类似地，可以学一个连续光照场，每个高斯体从中查询"入射光照特征"，再结合自身的 BRDF 特征输出颜色。

6. **核心idea一句话**：用张量分解的连续光照场 + 可学习 BRDF 特征 + 神经渲染器，替代 3DGS 中独立优化的球谐系数，在不求解逆渲染的前提下建模视角依赖效果。

## 方法详解

### 整体框架
3iGS 的 pipeline：输入是高斯体的3D位置 $\mathbf{x}_i$，首先从张量分解的光照场 $\mathcal{G}_l$ 中插值得到入射光照特征 $L_i$；然后将 $L_i$、高斯体自带的 BRDF 特征 $\rho_i$、以及观察方向 $\omega_o$（经 IDE 编码）一起输入轻量 MLP $\mathcal{F}$，输出镜面反射颜色 $\mathbf{c_s}$；最后与漫反射颜色 $\mathbf{c_d}$ 线性相加得到最终出射辐射：

$$\mathbf{c}(\omega_o) = \mathbf{c_d} + \mathbf{c_s}(\omega_o)$$

$$\mathcal{F}: \{\rho_i, L_i, \omega_o\} \mapsto \mathbf{c_s}$$

### 关键设计

1. **张量分解的光照场 (Factorised Tensorial Illumination Field)**:
   - 做什么：学习一个覆盖整个场景的连续3D光照场，每个高斯体可以从中查询自己位置的入射光照特征
   - 核心思路：采用 TensoRF 的 VM (Vector-Matrix) 分解，将3D光照体素网格分解为紧凑的向量和矩阵乘积之和。给定高斯体位置 $\mathbf{x}_i$，通过三线性插值从分解后的张量中获取光照特征 $L_i$：
     $$\mathcal{G}_l = \sum_{r=1}^{R_L} \mathbf{A}_{L,r}^{X} \circ \mathbf{b}_{3r-2} + \mathbf{A}_{L,r}^{Y} \circ \mathbf{b}_{3r-1} + \mathbf{A}_{L,r}^{Z} \circ \mathbf{b}_{3r}$$
   - 设计动机：（1）连续光照场让所有高斯体共享场景光照信息，打破了独立优化的限制；（2）VM 分解极其紧凑（$150^3$ 体素，比 TensoRF 少 87.5%），查询只需插值，几乎不影响渲染速度；（3）训练中途会 shrink 边界框并重采样，进一步提高效率

2. **高斯体 BRDF 特征 (Gaussian BRDF Features)**:
   - 做什么：每个高斯体携带可学习的 BRDF 特征向量 $\rho_i$，描述其表面反射属性
   - 核心思路：将原来 3DGS 中的 16x3 SH 系数重新利用为 BRDF 特征通道，并额外增加 4 个参数（base color + roughness）用于 IDE 视角编码。关键在于**不强制物理可解释性**——不像 GaussianShader 那样预测 roughness/albedo/metallicity 等物理参数，而是将 BRDF 特征视为一组权重，在神经渲染器中调制入射光照场
   - 设计动机：物理材质参数的逆渲染估计是严重的病态问题，不如让网络自己学习一组"广义 BRDF"特征。受 SH 卷积理论启发（$B_{lm} = \Lambda_l \rho_l L_{lm}$），BRDF 本质上是对入射光照的滤波器

3. **神经渲染器与 Shading (Neural Renderer)**:
   - 做什么：一个小型 MLP 将光照特征、BRDF 特征和视角方向映射到镜面颜色
   - 核心思路：视角方向用 Integrated Directional Encoding (IDE) 编码（来自 Ref-NeRF），IDE 需要 roughness 参数来调制编码频率——粗糙表面保留低频，光滑表面保留高频
   - 与 GaussianShader 区别：GaussianShader 使用全局可微分环境立方体贴图（cube map）+ GGX BRDF 解析计算；3iGS 使用局部连续光照场 + 神经网络预测，无需依赖特定的渲染方程

### 损失函数 / 训练策略
- 与 3DGS 完全相同的损失函数：$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$，$\lambda=0.2$
- **分阶段训练**：前 3000 步只训练漫反射颜色 $\mathbf{c_d}$，之后加入镜面颜色 $\mathbf{c_s}$，提升训练稳定性
- **光照场边界收缩**：训练中途将光照网格 shrink 到高斯体实际包围盒，用相同体素数重采样
- 同样采用 3DGS 的自适应密度控制

## 实验关键数据

### 主实验

**NeRF Synthetic 数据集（8 场景平均）**：

| 方法 | PSNR | SSIM | LPIPS | 实时渲染? |
|------|------|------|-------|----------|
| NeRF | 31.01 | 0.947 | 0.081 | 否 |
| Ref-NeRF | 31.29 | 0.947 | 0.058 | 否 |
| 3DGS | 33.30 | 0.969 | 0.030 | 是 |
| GaussianShader | 33.38 | 0.968 | 0.029 | 是(慢6.3x) |
| **3iGS (Ours)** | **33.64** | **0.970** | **0.029** | 是(慢2.0x) |

**Tanks and Temples 数据集（真实场景，5 场景平均）**：

| 方法 | PSNR | SSIM | LPIPS |
|------|------|------|-------|
| 3DGS | 29.61 | 0.950 | 0.060 |
| GaussianShader (reproduced) | 28.46 | 0.938 | 0.077 |
| **3iGS (Ours)** | **30.20** | **0.953** | **0.058** |

**Shiny Blender 数据集**：3iGS 在 PSNR 30.77 vs 3DGS 30.37 vs GaussianShader 31.94。GaussianShader 在此数据集上更优，原因是 Shiny Blender 场景几何简单（单个物体），以直接光照为主导，GaussianShader 的全局环境贴图 + GGX BRDF 恰好适合；而在多物体复杂间接光照的 NeRF Synthetic 场景中，3iGS 的局部连续光照场优势明显。

### 消融实验

| 配置 | PSNR | SSIM | LPIPS | 说明 |
|------|------|------|-------|------|
| 直接预测出射辐射场 | 32.38 | 0.965 | 0.035 | 类似NeRF方式直接从位置预测辐射，无BRDF分解 |
| 无 roughness 参数（无IDE） | 33.26 | 0.967 | 0.031 | 去掉roughness参数，用标准Fourier位置编码替代IDE |
| **完整模型** | **33.64** | **0.970** | **0.029** | 全部组件 |

### 关键发现
- **光照场是核心贡献**：直接预测出射辐射 vs 完整模型 PSNR 差 1.26 dB，说明连续光照场对建模视角依赖效果至关重要
- **IDE 视角编码有效**：roughness 控制的 IDE 比标准 Fourier 编码好 0.38 dB PSNR
- **速度优势**：3iGS 渲染速度仅比 3DGS 慢 2.0x，训练慢 3.2x；而 GaussianShader 渲染慢 6.3x，训练慢 12.1x —— 张量分解的光照场比 GaussianShader 的 ray tracing 方式高效得多
- **真实场景泛化好**：在 Tanks and Temples 上 3iGS 大幅优于 GaussianShader（+1.74 dB PSNR），说明在复杂真实场景中连续局部光照建模比全局环境贴图更有效

## 亮点与洞察
- **从游戏引擎借鉴的设计哲学**：irradiance volume / light probe 的思路迁移到可微分渲染——不求解精确物理参数，而是让网络在光照特征空间中学习，既避免了逆渲染的病态问题，又保持了高效
- **张量分解的紧凑性**：$150^3$ 体素 + VM 分解，存储开销极小（远小于百万级高斯体的参数量），查询只需三线性插值，几乎是"免费"的
- **即插即用设计**：方法与 3DGS 框架高度兼容，仅替换了颜色建模部分（SH -> illumination + BRDF + MLP），损失函数、自适应控制等全部复用
- **BRDF "软约束" 思路可迁移**：不强制物理参数而是学习神经 BRDF 特征的做法，可推广到其他需要材质建模的任务

## 局限性 / 可改进方向
- **受限于有界场景**：张量分解的光照网格需要预定义边界框，无法直接处理无界室外大场景，需要 scene warping 技术
- **显存需求高**：继承 3DGS 的大量高斯体 + 额外的光照网格，需要大显存 GPU
- **几何质量未改善**：继承了 3DGS 难以产生精确场景几何的问题——只改善了外观，法线/几何估计仍然不准
- **单一光照场假设**：当前只有一个全局光照场，对于动态光照或极大场景可能需要分区建模
- 可能的改进方向：结合 unbounded scene warping（如 mip-NeRF 360 的 contraction）+ 多分辨率光照场

## 相关工作与启发
- **vs 3DGS**：3DGS 独立优化每个高斯体的 SH，无场景级光照信息；3iGS 引入共享光照场让高斯体"看到"周围环境，镜面效果更真实
- **vs GaussianShader**：GaussianShader 使用全局环境 cube map + Cook-Torrance/GGX BRDF 物理模型；3iGS 使用局部连续光照场 + 神经 BRDF，避免了物理参数估计的病态问题。在简单单物体场景 GaussianShader 有优势，在复杂多物体场景 3iGS 更优
- **vs TensoRF**：TensoRF 用 VM 分解建模整个辐射场（NeRF替代）；3iGS 只用 VM 分解建模光照场，辐射场仍由高斯体承载，二者互补
- **vs Ref-NeRF**：Ref-NeRF 在 NeRF 框架中引入 IDE 编码改善反射建模；3iGS 将 IDE 借鉴到高斯体框架中，结合光照场获得更好效果

## 评分
- 新颖性: ⭐⭐⭐⭐ 将游戏引擎的光照体积思路引入 3DGS 框架，用张量分解实现高效光照建模
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据集全面评测，消融实验清晰，速度对比有说服力
- 写作质量: ⭐⭐⭐⭐ 动机推导流畅，从渲染方程到设计选择的逻辑链清晰
- 价值: ⭐⭐⭐⭐ 为 3DGS 视角依赖效果提供了高效解决方案，张量分解光照场的思路有实际应用价值
