---
title: >-
  [论文解读] Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation
description: >-
  [AAAI 2026][3D视觉][平面四边形网格] 提出一种基于DGCNN的数据驱动方法高效生成共轭方向场（CDF），避免了传统非线性优化的高计算开销，支持用户笔画引导的可控CDF生成，将CDF生成速度提升了1-2个数量级，同时配套发布了包含50000+自由曲面的大规模数据集。 平面四边形（PQ）网格：在计算机辅助设计中至…
tags:
  - "AAAI 2026"
  - "3D视觉"
  - "平面四边形网格"
  - "共轭方向场"
  - "深度学习"
  - "建筑设计"
  - "可控生成"
---

# Learning Conjugate Direction Fields for Planar Quadrilateral Mesh Generation

**会议**: AAAI 2026  
**arXiv**: [2511.11865](https://arxiv.org/abs/2511.11865)  
**代码**: [https://github.com/jiongtj/Learning-CDF](https://github.com/jiongtj/Learning-CDF)  
**领域**: 3D视觉  
**关键词**: 平面四边形网格, 共轭方向场, 深度学习, 建筑设计, 可控生成

## 一句话总结

提出一种基于DGCNN的数据驱动方法高效生成共轭方向场（CDF），避免了传统非线性优化的高计算开销，支持用户笔画引导的可控CDF生成，将CDF生成速度提升了1-2个数量级，同时配套发布了包含50000+自由曲面的大规模数据集。

## 研究背景与动机

**平面四边形（PQ）网格**在计算机辅助设计中至关重要，尤其是建筑表面的离散化。PQ网格的核心优势包括：(1) 面的平面性大幅降低玻璃等物理材料的制造成本；(2) 相比三角网格，顶点度更低，减少了支撑结构的复杂性；(3) 边缘布局直观美观。

PQ网格生成通常分两步：先生成初始四边形网格布局，再通过几何优化细化使每个面趋于平面。初始布局的质量取决于曲面上的**共轭方向场（CDF）**——CDF的共轭性保证初始网格面近似平面，对后续PQ网格优化至关重要。

**核心挑战**：不同于主曲率方向场（PDF）由曲面几何唯一确定（等距点除外），CDF**不唯一**且自由度很高。需要用户通过笔画指定偏好方向，作为非线性优化约束来计算CDF。但这个非线性优化：
- **计算量大**：随网格规模增长急剧增加（~20k面时需17秒，~60k面时需40秒）
- **迭代次数多**：设计者常需要反复调整笔画和重新计算，交互体验差
- **妨碍探索**：无法实时预览不同CDF方案对应的PQ网格布局

## 方法详解

### 整体框架

输入：三角网格 $\mathcal{M} = \{\mathcal{V}, \mathcal{F}\}$ + 用户笔画 $\mathcal{S} = \{\mathbf{S}_i\}$

输出：每个三角面上的方向向量对 $\{(\mathbf{u}_j, \mathbf{v}_j)\}$，构成CDF

流程：特征提取 → CDF预测 → 全局参数化 → 四边形网格提取 → 顶点扰动优化

### 关键设计

#### 1. **特征表示**

为每个顶点构建9维特征向量，由三部分拼接：

- **顶点位置** $\mathbf{p}_i \in \mathbb{R}^3$：编码网格几何
- **顶点法线** $\mathbf{n}_i \in \mathbb{R}^3$：编码局部曲面朝向
- **笔画投影向量** $\mathbf{l}_i = \mathbf{p}_i^* - \mathbf{p}_i \in \mathbb{R}^3$：从顶点到最近笔画点的向量

笔画投影向量的设计是一个全局笔画表示——它编码了每个顶点与笔画曲线的空间关系，而非单纯处理笔画本身。实验证明这比用Point Cloud Transformer（PCT）提取笔画特征效果好很多（δ: 8.31° vs 20.98°）。

#### 2. **网络架构**

**特征提取模块**：基于DGCNN，使用4层EdgeConv提取顶点特征。与原始DGCNN不同，本文将每个顶点的局部特征与全局形状特征拼接，再通过全连接层得到256维特征表示。DGCNN通过动态图在每层重新计算局部邻域，能自适应学习多尺度几何信息。

**预测模块**：两个独立MLP分别预测 $\{\mathbf{u}_j\}$ 和 $\{\mathbf{v}_j\}$。从顶点特征到面特征通过简单平均。每个MLP包含3层（256→128→64→3），前两层使用BatchNorm+ReLU，最后一层直接输出并归一化到单位长度。

#### 3. **损失函数设计**

本文精心设计了5个损失项：

**方向对齐损失 $\mathcal{L}_d$**：衡量预测CDF与ground truth的对齐程度。使用旋转90°后的ground truth向量来处理符号歧义，并取两种对应关系的最小值：

$$\mathcal{L}_d = \frac{1}{m}\sum_{j=1}^{m} \min(E_j, E_j')$$

其中 $E_j = (\mathbf{u}_j \cdot \mathbf{u}_j^{*\perp})^2 + (\mathbf{v}_j \cdot \mathbf{v}_j^{*\perp})^2$

**法线一致性损失 $\mathcal{L}_{dn}$**：确保预测方向与面法线正交：

$$\mathcal{L}_{dn} = \frac{1}{m}\sum_{j=1}^{m} (\mathbf{u}_j \cdot \mathbf{n}_j)^2 + (\mathbf{v}_j \cdot \mathbf{n}_j)^2$$

**方向平滑损失 $\mathcal{L}_{ds}$**：确保CDF在相邻面间平滑过渡，减少奇点：

$$\mathcal{L}_{ds} = \frac{1}{|\mathcal{N}|}\sum_{(j,k)\in\mathcal{N}} \min(E_{jk}, E_{jk}')$$

通过平行传输处理相邻面法线不同的情况。

**笔画一致性损失 $\mathcal{L}_{dc}$**：确保CDF方向与用户笔画对齐：

$$\mathcal{L}_{dc} = \frac{1}{|\mathcal{S}|}\sum_{\mathbf{S}_i \in \mathcal{S}} \frac{1}{|\mathcal{T}_i|}\sum_{k \in \mathcal{T}_i} D_k$$

**场正则化损失 $\mathcal{L}_{fr}$**：防止预测零向量：

$$\mathcal{L}_{fr} = \frac{1}{m}\sum_{j=1}^{m} (||\mathbf{u}_j||-1)^2 + (||\mathbf{v}_j||-1)^2$$

总损失：$\mathcal{L}_{total} = \mathcal{L}_d + \lambda_1\mathcal{L}_{dn} + \lambda_2\mathcal{L}_{ds} + \lambda_3\mathcal{L}_{dc} + \lambda_4\mathcal{L}_{fr}$

所有权重均设为1.0。

### 损失函数 / 训练策略

- 数据集：50000训练 + 2500验证 + 300测试的B样条曲面
- 每个曲面2601个采样点，5000个面
- 通过PCA归一化位置和朝向
- Adam优化器，学习率 $1.0 \times 10^{-4}$，训练200 epoch
- 硬件：Intel i9-14900K + NVIDIA RTX 4090

## 实验关键数据

### 主实验（计算效率对比）

CDF生成时间对比（vs. 传统优化方法）：

| 模型 | 面数 | 优化方法 | 本文方法 | 加速比 |
|------|------|---------|---------|--------|
| Test Model 1 | 5,000 | 2.851s | 0.200s | 14.3× |
| Test Model 2 | 5,000 | 2.855s | 0.194s | 14.7× |
| Vase | 23,642 | 17.326s | 0.254s | 68.2× |
| Dome | 44,490 | 30.198s | 0.417s | 72.4× |
| Face | 60,077 | 40.412s | 0.571s | 70.8× |
| Garden（建筑） | 8,322 | 4.946s | 0.206s | 24.0× |
| Yas Island（建筑） | 7,029 | 3.766s | 0.204s | 18.5× |
| Aqua Dome（建筑） | 10,790 | 6.522s | 0.217s | 30.1× |

面数越多，加速比越大：5k面时~15×，60k面时~71×。传统方法时间近线性增长，学习方法几乎平坦增长。

### 消融实验

消融损失函数对测试集的影响（300个模型平均）：

| 配置 | 奇点数 | δ (笔画一致性) | θ (CDF接近度) | 说明 |
|------|--------|--------------|-------------|------|
| Full model | 4.91 | 8.31° | 11.30° | 完整模型 |
| w/o $\mathcal{L}_{ds}$ | 7.02 | 7.38° | 10.55° | 奇点大幅增加 |
| w/o $\mathcal{L}_{dc}$ | 4.67 | 10.32° | 11.48° | 笔画一致性下降 |
| PCT笔画特征 | 8.97 | 20.98° | 19.06° | 远差于本文表示 |

### 关键发现

- 平滑损失 $\mathcal{L}_{ds}$ 对减少PQ网格奇点至关重要（7.02→4.91）
- 笔画一致性损失 $\mathcal{L}_{dc}$ 将δ从10.32°降到8.31°
- 本文的笔画投影向量表示远优于PCT（δ降低60%，θ降低41%），验证了曲面上下文感知的重要性
- PQ网格的平面性（初始即很好 $\eta_{\text{mean}} \approx 0.006$），经顶点扰动优化后进一步提升到~0.002
- 方法能泛化到开边界曲面、真实建筑曲面，甚至拓扑各异的封闭模型（如Stanford Bunny）
- 对比VectorHeat和NeurCross：VectorHeat无法保证共轭性，NeurCross限于PDF，均不适用于可控CDF生成

## 亮点与洞察

1. **问题定义精准**：CDF的非唯一性既是灵活性来源也是计算瓶颈，用学习方法绕开非线性优化是自然且有效的方案
2. **笔画表示设计巧妙**：投影向量编码了每个顶点与笔画的空间关系，隐式地将笔画信息传播到整个曲面
3. **损失函数处理方向歧义**：通过旋转90°和取最小值的方式优雅地处理了方向场固有的符号和对应歧义
4. **大规模数据集贡献**：50000+样本的合成数据集也是实际贡献，且通过模拟实际设计工作流（从ground truth CDF追踪流线模拟笔画）来构造训练数据

## 局限与展望

- 在锐特征处CDF无法准确对齐锐边（因为训练数据中未包含锐特征）
- 没有对奇点数量和位置的显式控制
- 仅使用B样条曲面生成的合成数据，可能限制泛化能力
- 未探索无监督方法以获得更好泛化
- 笔画的覆盖范围和密度对结果有影响，但缺乏系统分析

## 相关工作与启发

- 传统方法（Liu et al. 2011）通过约束非线性优化计算CDF，本文用学习完全替代
- Sketch2PQ（Deng et al. 2022）从2D草图预测PQ网格，但在3D曲面上限制较大
- VectorHeatNet学习向量场但不保证共轭性
- 启发：对建筑CAD中的其他计算瓶颈（如曲面展开、结构优化），学习代替优化可能同样适用

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次用深度学习解决CDF生成问题
- 实验充分度: ⭐⭐⭐⭐ — 效率对比充分，消融完整，泛化测试到建筑和通用3D
- 写作质量: ⭐⭐⭐⭐⭐ — 问题定义清晰，数学推导严谨
- 价值: ⭐⭐⭐⭐ — 对建筑设计CAD有直接应用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] QuadGPT: Native Quadrilateral Mesh Generation with Autoregressive Models](../../ICLR2026/3d_vision/quadgpt_native_quadrilateral_mesh_generation_with_autoregressive_models.md)
- [\[CVPR 2026\] Mesh-Pro: Asynchronous Advantage-guided Ranking Preference Optimization for Artist-style Quadrilateral Mesh Generation](../../CVPR2026/3d_vision/mesh-pro_asynchronous_advantage-guided_ranking_preference_optimization_for_artis.md)
- [\[AAAI 2026\] Hierarchical Direction Perception via Atomic Dot-Product Operators for Rotation-Invariant Point Clouds Learning](hierarchical_direction_perception_via_atomic_dot-product_operators_for_rotation-.md)
- [\[CVPR 2026\] Learning Convex Decomposition via Feature Fields](../../CVPR2026/3d_vision/learning_convex_decomposition_via_feature_fields.md)
- [\[AAAI 2026\] TG-Field: Geometry-Aware Radiative Gaussian Fields for Tomographic Reconstruction](tg-field_geometry-aware_radiative_gaussian_fields_for_tomographic_reconstruction.md)

</div>

<!-- RELATED:END -->
