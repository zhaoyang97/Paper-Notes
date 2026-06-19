---
title: >-
  [论文解读] LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS
description: >-
  [NeurIPS 2025][3D视觉][3D语言场] 通过将每个3D高斯视为全局字典上的稀疏编码，LangSplatV2用稀疏系数场替代重量级解码器，实现476.2 FPS的高维特征溅射和384.6 FPS的3D开放词汇查询，较LangSplat加速47倍。 领域现状：3D语言场是视觉语言模型与3D环境建模的交叉领域…
tags:
  - "NeurIPS 2025"
  - "3D视觉"
  - "3D语言场"
  - "高斯溅射"
  - "稀疏编码"
  - "实时推理"
  - "开放词汇查询"
---

# LangSplatV2: High-dimensional 3D Language Gaussian Splatting with 450+ FPS

**会议**: NeurIPS 2025  
**arXiv**: [2507.07136](https://arxiv.org/abs/2507.07136)  
**代码**: [项目主页](https://langsplat-v2.github.io)  
**领域**: 3D视觉  
**关键词**: 3D语言场, 高斯溅射, 稀疏编码, 实时推理, 开放词汇查询

## 一句话总结

通过将每个3D高斯视为全局字典上的稀疏编码，LangSplatV2用稀疏系数场替代重量级解码器，实现476.2 FPS的高维特征溅射和384.6 FPS的3D开放词汇查询，较LangSplat加速47倍。

## 研究背景与动机

**领域现状**：3D语言场是视觉语言模型与3D环境建模的交叉领域，LangSplat通过3D高斯溅射嵌入CLIP特征取得重要进展（比LERF快199倍）。

**现有痛点**：LangSplat仍未达到实时推理（8.2 FPS），重量级MLP解码器占总推理时间的97.1%，严重限制AR、机器人等应用。

**核心矛盾**：CLIP特征是512维高维向量，直接溅射开销巨大（1536维比3维慢15倍）；但通过编码器压缩到低维再解码引入了沉重的MLP瓶颈。

**本文目标**：消除解码器瓶颈，实现高维特征的实时溅射。

**切入角度**：观察到场景中百万级高斯点实际只包含有限数量的独特语义，可用稀疏编码高效表示。

**核心 idea**：每个高斯的语言特征是全局码本上K个基向量的稀疏线性组合，渲染稀疏系数而非高维特征。

## 方法详解

### 整体框架

为每个3D高斯学习一个L维稀疏系数向量（仅K个非零值）和一个共享的全局码本（L个D维基向量）。推理时：(1) 溅射K维稀疏系数 → (2) 矩阵乘法恢复D维CLIP特征 → (3) 计算相关性评分。完全去除MLP解码器。

### 关键设计

1. **3D稀疏系数场（3D Sparse Coefficient Field）**：

    - 功能：用稀疏系数+全局码本替代每个高斯的高维语言特征
    - 为什么：百万高斯点仅对应有限语义，天然适合稀疏表示
    - 怎么做：每个高斯的特征 $\mathbf{f}_i = \mathbf{w}_i \mathcal{S} = \sum_{l=1}^{L} w_{i,l} \mathbf{s}_l$，其中 $\mathbf{w}_i \in \mathbb{R}^{L}$ 仅K个非零值
    - 关键推导：$\mathbf{S} = \sum_{i \in \mathcal{N}} \mathbf{w}_i \mathcal{S} e_i = (\sum_{i \in \mathcal{N}} e_i \mathbf{w}_i) \mathcal{S}$
    - 区别：渲染D维特征等价于先渲染L维系数再乘码本，解耦了渲染维度与特征维度

2. **高效稀疏系数溅射（Efficient Sparse Coefficient Splatting）**：

    - 功能：利用稀疏性加速CUDA alpha-blending
    - 为什么：标准溅射复杂度 $O(|\mathcal{N}| \cdot L)$，L大时成为瓶颈
    - 怎么做：每个高斯仅存储top-K索引和系数值，alpha-blending仅对K个非零元素操作
    - 复杂度从 $O(|\mathcal{N}| \cdot L)$ 降到 $O(|\mathcal{N}| \cdot K)$
    - 实践中K=4，三个语义尺度并行渲染有效维度仅12
    - 区别：渲染速度与特征维度完全解耦

3. **全局码本学习**：

    - 功能：为整个场景学习L个D维基向量
    - 为什么：捕获场景中所有独特语义的紧凑表示
    - 怎么做：L维稀疏系数先softmax归一化，保留top-K后重归一化，与码本联合端到端学习
    - 参数设置：L=64, K=4, D=512（CLIP特征维度）
    - 区别：无维度压缩损失（直接在CLIP空间学习）

### 损失函数 / 训练策略

- 先用RGB监督训练3D高斯30,000轮
- 再固定高斯参数，训练稀疏系数场10,000轮
- 使用OpenCLIP ViT-B/16提取CLIP特征，SAM ViT-H进行语义分割
- 三个SAM层级语义同时建模

## 实验关键数据

### 主实验

**LERF数据集 - 3D开放词汇定位与分割**：

| 方法 | 定位Acc(%) | 分割IoU(%) | 速度(FPS) |
|------|-----------|------------|-----------|
| LERF | — | — | ~0.04 |
| LangSplat | 84.3 | 51.4 | 8.2 |
| GAGS | 81.7 | 54.1 | — |
| **LangSplatV2** | **84.1** | **59.9** | **384.6** |

**推理时间分解 (毫秒, A100 GPU)**：

| 方法 | 渲染 | 解码 | 后处理 | 总计 | FPS |
|------|------|------|--------|------|-----|
| LangSplat | 6.0 | 83.1 | 33.0 | 122.1 | 8.2 |
| LangSplat* | 2.0 | 83.1 | 0.5 | 85.6 | 11.7 |
| **LangSplatV2** | **2.0** | **0.1** | **0.5** | **2.6** | **384.6** |

**不同GPU上的特征渲染时间 (ms)**：
- RTX 3090和RTX 4090在特征维度≥1024时OOM
- LangSplatV2渲染时间不随特征维度增长（始终≈2ms）

### 消融实验

**码本大小L和稀疏度K的影响**（LERF数据集, Overall IoU %）：

| L/K | K=2 | K=4 | K=8 |
|-----|-----|-----|-----|
| L=32 | 56.8 | 58.1 | 58.5 |
| L=64 | 57.5 | **59.9** | 59.7 |
| L=128 | 57.2 | 59.5 | 59.8 |

（L=64, K=4为最佳效率-精度平衡点）

### 关键发现

- 解码器去除后速度提升47倍（8.2→384.6 FPS），分割精度反而提升8.5% IoU
- 精度提升原因：消除了维度压缩带来的信息损失，直接在CLIP空间建模
- K=4已足以高质量表示场景语义，进一步增大K边际收益极小
- 在3D-OVS和Mip-NeRF360数据集上同样优于LangSplat
- 特征渲染速度与维度完全解耦是核心技术贡献

## 亮点与洞察

- **"解码即瓶颈"的深刻洞察**：97.1%推理时间在解码阶段，通过定量分析精准定位问题
- **稀疏编码思想的优雅应用**：从百万高斯点到有限语义的观察极具直觉性
- **数学推导简洁有力**：渲染线性性使得特征维度与渲染维度完美解耦
- **工程与理论兼备**：CUDA稀疏优化实现了理论承诺的加速倍数

## 局限与展望

- 码本大小L和稀疏度K需要手动设置
- 稀疏性假设可能在语义极其丰富的大规模场景中受限
- 训练分两阶段（RGB→语言），端到端联合训练可能进一步提升
- 可探索自适应K值——不同区域语义复杂度不同

## 相关工作与启发

- LangSplat的autoencoder压缩思路与本文的稀疏编码思路形成鲜明对比
- 3D高斯压缩方法（CompGS, LightGaussian）关注RGB压缩，本文关注高维语义特征
- 稀疏编码在传统信号处理中有深厚理论基础，在3D场景理解中的应用是创新
- 对所有需要高维特征溅射的任务（CLIP/DINOv2/SAM特征）都有通用价值

## 评分

- 新颖性: ⭐⭐⭐⭐ 稀疏系数场思路清晰优雅且有效
- 实验充分度: ⭐⭐⭐⭐ 多数据集、详细时间分析、全面消融
- 写作质量: ⭐⭐⭐⭐⭐ 问题分析→方案推导→实验验证的逻辑链极其流畅
- 价值: ⭐⭐⭐⭐⭐ 47倍加速+精度提升，直接推动3D语言场落地应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] PlanarGS: High-Fidelity Indoor 3D Gaussian Splatting Guided by Vision-Language Planar Priors](planargs_high-fidelity_indoor_3d_gaussian_splatting_guided_by_vision-language_pl.md)
- [\[CVPR 2026\] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars](../../CVPR2026/3d_vision/hypergaussians_high-dimensional_gaussian_splatting_for_high-fidelity_animatable_.md)
- [\[CVPR 2025\] InstantHDR: Single-forward Gaussian Splatting for High Dynamic Range 3D Reconstruction](../../CVPR2025/3d_vision/instanthdr_single-forward_gaussian_splatting_for_high_dynamic_range_3d_reconstru.md)
- [\[NeurIPS 2025\] IBGS: Image-Based Gaussian Splatting](ibgs_image-based_gaussian_splatting.md)
- [\[ECCV 2024\] GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](../../ECCV2024/3d_vision/gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)

</div>

<!-- RELATED:END -->
