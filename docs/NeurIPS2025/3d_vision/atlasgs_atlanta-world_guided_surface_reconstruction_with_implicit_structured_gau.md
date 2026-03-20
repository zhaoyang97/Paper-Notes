<!-- 由 src/gen_stubs.py 自动生成 -->
# AtlasGS: Atlanta-world Guided Surface Reconstruction with Implicit Structured Gaussians

**会议**: NeurIPS 2025  
**arXiv**: [2510.25129](https://arxiv.org/abs/2510.25129)  
**代码**: 待确认  
**领域**: 3d_vision  
**关键词**: 3D Gaussian Splatting, 表面重建, Atlanta-world 假设, 隐式表示, 室内/城市场景  

## 一句话总结

提出 AtlasGS，通过将 Atlanta-world 结构先验引入隐式结构化高斯表示（implicit-structured Gaussians），在室内和城市场景中实现平滑且保留高频细节的高质量表面重建，全面超越已有隐式和显式方法。

## 研究背景与动机

1. **室内/城市重建是热门课题**：数字孪生、机器人导航、增强现实等应用对高精度高效重建有强烈需求。
2. **低纹理区域是核心难点**：人造场景中的地板、天花板、白墙等缺乏纹理特征，传统多视图立体方法在这些区域重建失败，产生不完整或扭曲的几何。
3. **单目几何先验缺乏全局一致性**：单目深度/法线先验仅提供局部平滑信号，不同视角间常出现不一致，导致凹凸不平的表面。
4. **Manhattan-world 假设过于严格**：Manhattan-world 假设要求场景沿三个正交方向排列，无法处理城市场景中建筑物非正交排列的情况（如斜向建筑）。
5. **2DGS 的离散性导致表面不连续**：2D Gaussian Splatting 使用独立优化的 surfel 原语，在低纹理/欠观测区域产生断裂表面。
6. **隐式 SDF + GS 的简单叠加效果不佳**：已有方法（如 GSRec）尝试用隐式 SDF 场正则化高斯优化，但两者的相互干扰往往导致重建质量下降。

**核心动机**：需要 (1) 全局一致的几何先验来正则化低纹理区域；(2) 兼具高斯效率/高频细节保持和隐式方法平滑性的 3D 表示。

## 方法详解

### 整体框架

给定带位姿的多视图图像和 SfM 点云，构建稀疏特征体素网格，将场景表示为隐式结构化的 2D 高斯（surfel）。通过属性解码器和语义解码器预测高斯属性，经光栅化后用 RGB 图像、单目几何先验和语义标签监督。同时引入基于 Atlanta-world 假设的可学习平面指示器（plane indicators）约束全局结构。

### 三大核心设计

#### 1. 隐式结构化高斯表示（Implicit-Structured Gaussians）

- 基于 SfM 点云构建**稀疏特征体素网格** $\mathcal{V}$，每个体素包含几何特征 $\mathcal{V}_g$、语义特征 $\mathcal{V}_s$、$\mathcal{K}=10$ 个局部高斯的偏移量 $\Delta_k$ 和共享缩放因子 $l$。
- 通过几何 MLP $\mathcal{M}_g$ 解码不透明度 $\alpha$、缩放 $s$、旋转 $q$、颜色 $c$（视角相关），通过语义 MLP $\mathcal{M}_s$ 解码语义属性 $z \in \mathbb{R}^4$（墙/地板/天花板/其他）。
- 高斯位置 $\mathbf{p}_k^i = \mathbf{v}_i + l \cdot \Delta_k^i$，通过体素中心加偏移计算。
- **核心优势**：解码器使每个高斯的优化隐式影响邻域，实现局部几何一致性，同时通过高斯原语保留高频细节。与 2DGS 独立优化各原语的方式形成对比。

#### 2. 语义高斯提升（Gaussian Semantic Lifting）

- 使用预训练语义分割模型生成 2D 伪标签 $\hat{Z}$，定义 4 类语义：墙壁、地板、天花板、其他。
- 将 3D 语义属性 $z$ 渲染到图像空间得到语义概率 $Z$，用交叉熵损失 $\mathcal{L}_{\text{sem}}$ 优化。
- 使用 stop-gradient 阻断语义监督对几何的反向传播，防止不一致标签污染几何优化。

#### 3. Atlanta-world 引导的平面正则化

**可学习平面指示器**：定义地板平面 $\pi_f = (\mathbf{n}_g, d_f)$ 和天花板平面 $\pi_c = (-\mathbf{n}_g, d_c)$，其中 $\mathbf{n}_g$ 为重力方向，$d_f, d_c$ 为距原点距离。城市场景忽略天花板平面。通过 RANSAC 初始化后与高斯联合优化。

**3D 全局平面正则化** $\mathcal{L}_{3D}$：

- **法线对齐**：墙面高斯法线应垂直于重力方向（$1 - |\mathbf{n}_g^\top \mathbf{n}_i|$）；地板/天花板高斯法线应平行于重力方向（$|\mathbf{n}_g^\top \mathbf{n}_i|$）。
- **平面约束**：地板/天花板高斯的位置应位于对应平面上（$|d_f + \mathbf{n}_g^\top \mathbf{p}_i|$ 等）。
- 各项按语义概率加权，软性约束。

**2D 局部表面正则化** $\mathcal{L}_{2D}$：

- 针对墙面区域：高斯位置和法线的显式解耦导致仅优化法线无法约束空间分布。
- 从渲染深度反投影得到 3D 点，计算局部表面法线 $\mathbf{N}_d$，约束其与重力方向的关系。
- 同样按语义概率加权以缓解语义误分类。

### 损失函数

$$\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_1 \mathcal{L}_{\text{depth}} + \lambda_2 \mathcal{L}_{\text{normal}} + \lambda_3 \mathcal{L}_{\text{reg}} + \lambda_4 \mathcal{L}_{\text{sem}} + \lambda_5 \mathcal{L}_{\text{dist}} + \lambda_6 \mathcal{L}_{\text{nc}}$$

其中 $\mathcal{L}_{\text{reg}} = \mathcal{L}_{3D} + \mathcal{L}_{2D}$；$\mathcal{L}_{\text{depth}}$ 使用 scale-shift 后的 L2 对齐单目深度先验；$\mathcal{L}_{\text{normal}}$ 同时约束渲染法线和深度导出法线与先验法线一致。

## 实验

### 数据集与基线

- **室内**：Replica（7 场景合成）、ScanNet（4 场景真实）、ScanNet++（4 场景真实）
- **室外**：MatrixCity（4 city blocks 合成）
- **基线**：隐式方法（ManhattanSDF、MonoSDF）；显式方法（Scaffold-GS、2DGS、DN-Splatter、GSRec）；室外加 GaussianPro

### 关键定量结果

| 数据集 | 指标 | AtlasGS | 最佳基线 | 提升 |
|--------|------|---------|----------|------|
| Replica | F-score ↑ | **87.35** | MonoSDF 73.08 | +14.27 |
| ScanNet++ | F-score ↑ | **87.48** | ManhattanSDF 76.67 | +10.81 |
| ScanNet | F-score ↑ | **77.98** | MonoSDF 71.21 | +6.77 |
| ScanNet | Acc ↓ (cm) | **3.62** | ManhattanSDF 4.25 | -0.63 |
| MatrixCity | CD ↓ | **0.028** | GaussianPro 0.091 | -0.063 |

### 关键发现

1. **全面超越隐式和显式方法**：在所有室内数据集上 F-score 均大幅领先，精度和完整性同时优于基线。
2. **效率优于隐式方法**：ScanNet 上训练 27 分钟 vs 隐式方法 7+ 小时，渲染 70 FPS vs < 10 FPS。
3. **室外场景同样有效**：MatrixCity 上 CD 仅 0.028，远优于所有基线（含 GSRec 的 0.112 和 2DGS 的 0.106）。
4. **新视角合成质量有竞争力**：虽非最优（Replica 上 PSNR 39.58 vs 2DGS 41.59），但在真实数据集 ScanNet++ 上 LPIPS 最优（0.2517），几何精度带来更少伪影。

### 消融实验（ScanNet）

| 配置 | CD ↓ | F-score ↑ |
|------|------|-----------|
| 2DGS + depth/normal 先验 | 12.68 | 39.27 |
| 隐式结构化 GS（无 $\mathcal{L}_{\text{reg}}$） | 4.10 | 74.23 |
| + $\mathcal{L}_{3D}$（无 $\mathcal{L}_{2D}$） | 3.97 | 75.52 |
| 完整模型 | **3.77** | **77.98** |

- 隐式结构化表示本身就大幅提升质量（F-score 39.27 → 74.23）。
- 3D 和 2D 正则化各贡献约 1-2 个 F-score 点，合计提升 3.75。
- 移除 depth 或 normal 先验均导致性能下降，证明几何先验不可或缺。

## 亮点

1. **Atlanta-world 假设比 Manhattan-world 更具通用性**：允许多个非正交水平方向，统一适用于室内和城市场景，是一个很好的结构先验扩展。
2. **隐式结构化高斯的设计精巧**：不是简单叠加隐式和显式表示，而是将体素网格嵌入高斯框架内部，通过共享 MLP 解码实现局部一致性，避免了先前方法中的相互干扰问题。
3. **语义-几何解耦**：stop-gradient 阻断语义监督对几何的反传，是一个细致而重要的设计。
4. **2D 局部表面正则化的洞察**：发现高斯表示中法线和位置的解耦导致仅约束法线不够，需要从渲染深度推导局部表面法线来间接约束位置。

## 局限性

1. **训练和渲染速度慢于纯高斯方法**：27 分钟 vs 11-12 分钟训练，70 FPS vs 118-279 FPS 渲染。额外的 MLP 解码所有高斯属性带来显著开销。
2. **依赖预训练语义分割模型**：语义类别固定为 4 类（墙/地板/天花板/其他），对非典型结构场景（如曲面建筑、自然环境）适用性受限。
3. **Atlanta-world 假设本身有适用范围**：仅适用于存在主导重力方向和平面结构的人造场景，对自然地形、非结构化环境不适用。
4. **新视角合成并非最优**：在合成数据 Replica 上 PSNR 低于 2DGS，说明几何约束对渲染质量有一定代价。

## 相关工作

- **隐式表面重建**：NeRF → NeuS/VolSDF（SDF + 体渲染）→ 加入单目先验（MonoSDF）和语义（ManhattanSDF）。受限于 MLP 表达能力和训练速度。
- **高斯表面重建**：3DGS → 2DGS/Gaussian Surfels（surfel 替代 3D 高斯提升多视角一致性）→ PGSR（平面高斯）→ GSRec（IMLS 正则化）→ DN-Splatter（深度法线先验）。离散性仍是核心问题。
- **结构先验**：Manhattan-world（三正交方向）→ Atlanta-world（一重力 + 多水平方向），后者更灵活。
- **隐式-显式联合方法**：NeuSG、GSDF 等同时学习 SDF 和 GS，但相互干扰导致效果不佳。本文的嵌入式设计避免了这个问题。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — Atlanta-world 假设引入 GS 是新颖组合，隐式结构化高斯设计有独创性，但各组件技术（体素网格、MLP 解码、语义提升）较为标准。
- **实验充分度**: ⭐⭐⭐⭐ — 4 个数据集（含室内外）、6+ 个基线、完整消融，结果令人信服。但缺少复杂非结构化场景的失败案例分析。
- **写作质量**: ⭐⭐⭐⭐ — 动机清晰，方法描述详尽，图表质量高。公式较多但组织合理。
- **价值**: ⭐⭐⭐⭐ — 在室内/城市重建这一重要方向上取得全面 SOTA，工程价值和学术价值均高。主要受限于速度和适用场景范围。
