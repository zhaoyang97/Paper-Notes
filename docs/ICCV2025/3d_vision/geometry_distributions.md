---
title: >-
  [论文解读] Geometry Distributions
description: >-
  [ICCV 2025][3D视觉][几何表示] 提出Geometry Distributions (GeomDist)，将3D几何建模为表面点的概率分布并用扩散模型学习，无需假设亏格、连通性或边界条件，可从高斯噪声采样无限多表面点来表示任意拓扑的几何。 现有3D几何表示各有局限： - 网格：数据结构不一致…
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "几何表示"
  - "扩散模型"
  - "表面点分布"
  - "神经压缩"
  - "非水密网格"
---

# Geometry Distributions

**会议**: ICCV 2025  
**arXiv**: [2411.16076](https://arxiv.org/abs/2411.16076)  
**代码**: 未公开  
**领域**: 3D视觉  
**关键词**: 几何表示, 扩散模型, 表面点分布, 神经压缩, 非水密网格

## 一句话总结

提出Geometry Distributions (GeomDist)，将3D几何建模为表面点的概率分布并用扩散模型学习，无需假设亏格、连通性或边界条件，可从高斯噪声采样无限多表面点来表示任意拓扑的几何。

## 研究背景与动机

现有3D几何表示各有局限：
- **网格**：数据结构不一致，不适合学习
- **体素**：内存密集，高分辨率需求大
- **点云**：采样有限，缺乏连通性信息
- **SDF**：无法表示薄结构和非水密几何

核心洞见：任何表面都可以用足够多的采样点逼近，而生成模型理论上可以从分布中采样无限数据。因此将几何建模为表面点的分布 $\Phi_{\mathcal{M}}$，使得 $\mathbf{x} \sim \Phi_{\mathcal{M}} \Rightarrow \mathbf{x} \in \mathcal{M}$。

## 方法详解

### 问题建模

给定表面 $\mathcal{M} \subset \mathbb{R}^3$，学习映射 $\mathcal{E}$：从高斯分布到表面点分布。通过扩散模型 $D_\theta(\cdot, \cdot)$ 学习，满足ODE：

$$\mathrm{d}\mathbf{x} = \frac{\mathbf{x} - D_\theta(\mathbf{x}, t)}{t} \mathrm{d}t$$

### 前向采样（高斯 → 表面）

从高斯噪声 $\mathbf{x}_0 = T\mathbf{n}$ 出发，迭代计算：

$$\mathbf{x}_{i+1} = \mathbf{x}_i + (t_{i+1} - t_i) \cdot \frac{\mathbf{x}_i - D_\theta(\mathbf{x}_i, t_i)}{t_i}$$

终点 $\mathbf{x}_N$ 落在目标表面上。采样任意多高斯点即可任意精度逼近表面。

### 逆向采样（表面 → 高斯）

从表面点出发反向遍历轨迹，映射回噪声空间：

$$\mathbf{x}_{i-1} = \mathbf{x}_i + (t_{i-1} - t_i) \cdot \frac{\mathbf{x}_i - D_\theta(\mathbf{x}_i, t_i)}{t_i}$$

实现表面点与噪声空间的一一对应。

### 训练过程

关键设计：**每个epoch重新采样** $2^{25}$ 个表面点。1000个epoch后网络见过足够多的表面点，模拟无限采样：

$$\arg\min_\theta \mathbb{E}_{\mathbf{x} \in \mathcal{M}} \mathbb{E}_{\mathbf{n} \sim \mathcal{N}} \mathbb{E}_{\sigma > 0} \|D_\theta(\mathbf{x} + \sigma\mathbf{n}, \sigma) - \mathbf{x}\|$$

### 网络设计

受EDM启发的magnitude-preserving层设计，6个块、C=512线性层，共5.53M参数。输入输出标准化为零均值单位方差。

## 实验

### 与SDF对比 (非水密物体)

| 方法 | 参数量 | 非水密支持 | 薄结构 |
|------|--------|-----------|--------|
| SDF (Instant-NGP) | 14M | ✗ | 差 |
| **GeomDist** | **5M** | **✓** | **好** |

GeomDist用更少参数成功表示SDF无法处理的开放和非水密几何。

### 与向量场方法对比

| 方法 | Chamfer Distance (×10³) | 均匀性 |
|------|------------------------|--------|
| 向量场 | 4.886 | 不均匀 |
| **GeomDist** | **3.218** | **均匀** |

GeomDist在均匀性和几何保真度上均优于向量场方法。

### 多分辨率采样

在Wukong网格上从 $n=2^{15}$ 到 $n=2^{19}$ 不同分辨率采样，均能准确逼近目标表面，展示了连续分辨率调整能力。

## 亮点与洞察

1. **表示的普适性**：不假设亏格、水密性、连通性——真正通用的几何表示
2. **无限分辨率**：理论上可采样无限多点，不受固定采样密度限制
3. **紧凑性**：5M参数编码复杂几何，远少于14M的SDF
4. **可逆性**：前向和逆向采样沿同一轨迹，实现表面点与噪声空间的双射
5. **多应用支持**：纹理网格表示、神经压缩、动态建模、高斯渲染

## 局限性

- 训练时间较长（数小时），不适合实时应用
- 每个物体需单独训练一个网络模型
- 表面提取依赖后处理（如Ball Pivoting连接）
- 未展示大规模场景的可扩展性

## 相关工作

- SDF/UDF: 坐标基神经表示
- Point-E, NeuralPoints: 点云生成
- EDM: 扩散模型框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ (几何即分布的全新视角)
- 技术深度: ⭐⭐⭐⭐⭐ (ODE框架+网络设计+训练策略)
- 实验充分度: ⭐⭐⭐⭐ (多类物体+消融+应用)
- 实用价值: ⭐⭐⭐⭐ (通用表示有广泛潜力)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Neural Compression for 3D Geometry Sets](neural_compression_for_3d_geometry_sets.md)
- [\[CVPR 2025\] Guiding Human-Object Interactions with Rich Geometry and Relations](../../CVPR2025/3d_vision/guiding_human-object_interactions_with_rich_geometry_and_relations.md)
- [\[ICCV 2025\] GeoSplatting: Towards Geometry Guided Gaussian Splatting for Physically-based Inverse Rendering](geosplatting_towards_geometry_guided_gaussian_splatting_for_physically-based_inv.md)
- [\[CVPR 2026\] Scenes as Tokens: Multi-Scale Normal Distributions Transform Tokenizer for General 3D Vision-Language Understanding](../../CVPR2026/3d_vision/scenes_as_tokens_multi-scale_normal_distributions_transform_tokenizer_for_genera.md)
- [\[NeurIPS 2025\] GOATex: Geometry & Occlusion-Aware Texturing](../../NeurIPS2025/3d_vision/goatex_geometry_occlusion-aware_texturing.md)

</div>

<!-- RELATED:END -->
