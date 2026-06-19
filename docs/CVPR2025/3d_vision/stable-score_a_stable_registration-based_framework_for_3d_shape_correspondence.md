---
title: >-
  [论文解读] Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence
description: >-
  [CVPR 2025][3D视觉][3D形状对应] Stable-SCore 重新审视了"配准-对应"范式，通过利用 2D 基础模型（Stable Diffusion + DINO）建立稳健的 2D 字符对应，并提出语义流引导的配准方法（基于 Neural Jacobian Fields）通过可微渲染桥接 2D 对应与 3D 形变，在非等距字符形状对应任务中大幅超越了函数映射系列方法。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D形状对应"
  - "配准方法"
  - "Neural Jacobian Fields"
  - "2D基础模型"
  - "非等距形状"
---

# Stable-SCore: A Stable Registration-Based Framework for 3D Shape Correspondence

**会议**: CVPR 2025  
**arXiv**: [2503.21766](https://arxiv.org/abs/2503.21766)  
**代码**: [https://haolinliu97.github.io/Stable-Score](https://haolinliu97.github.io/Stable-Score)  
**领域**: 3D视觉  
**关键词**: 3D形状对应, 配准方法, Neural Jacobian Fields, 2D基础模型, 非等距形状

## 一句话总结

Stable-SCore 重新审视了"配准-对应"范式，通过利用 2D 基础模型（Stable Diffusion + DINO）建立稳健的 2D 字符对应，并提出语义流引导的配准方法（基于 Neural Jacobian Fields）通过可微渲染桥接 2D 对应与 3D 形变，在非等距字符形状对应任务中大幅超越了函数映射系列方法。

## 研究背景与动机

**领域现状**：3D 形状对应旨在建立不同形状之间的逐点映射，是计算机视觉和图形学的基础任务，直接支撑 re-topology、纹理迁移、骨骼迁移、形状插值等应用。目前有两大范式——**函数映射方法**和**配准-对应方法**。函数映射方法（如 FMNet、ULRSSM）近年来占据主导地位，将点映射转化为函数空间映射。

**现有痛点**：函数映射方法在"受控场景"（形状和拓扑差异小）下表现优异，但在真实世界的非等距场景（如艺术家创作的角色或 AI 生成的 3D 模型之间）表现急剧下降。其根本原因在于函数映射依赖严格对齐的低秩基函数，非等距形变破坏了这一假设。另一方面，传统配准方法虽然更适合非等距场景，但面临两个致命问题：（1）形变过程不稳定（扭曲、失真）；（2）需要高质量的初始 3D 对应或仔细的预对齐，而这在非等距场景下恰恰最难获得。

**核心矛盾**：在非等距设置下，函数映射的数学假设不成立，配准方法又缺乏稳定的形变模型和可靠的初始对应。两大范式各有结构性缺陷。

**本文目标**：修复配准-对应范式的两个核心缺陷——用 Neural Jacobian Fields 替代不稳定的形变模型，用 2D 基础模型替代不可靠的 3D 初始对应——重新激活这一范式在非等距场景下的潜力。

**切入角度**：作者观察到 2D 视觉基础模型（如 Stable Diffusion、DINOv2）在图像级密集对应上展现了惊人的泛化能力，可以"跨模态迁移"到 3D 任务：先在 2D 渲染图上建立对应，再通过可微渲染将 2D 对应引导转化为 3D 形变。

**核心 idea**：（1）用 Stable Diffusion + DINO 特征训练一个轻量级 2D 对应模型；（2）用 Neural Jacobian Fields 作为稳定的形变引擎；（3）通过可微渲染将 2D 语义流作为 3D 配准的监督信号，迭代优化源网格向目标网格的形变。

## 方法详解

### 整体框架

1. **渲染多视角图像**：将源网格和目标网格在固定相机位姿下渲染为多视角的 RGB 或法线图
2. **2D 对应估计**：用 Stable Diffusion + DINO 特征提取器 + 轻量级 adapter 网络，在每个视角下建立源-目标图像之间的 2D 语义流图
3. **语义流引导配准**：在 Neural Jacobian Fields 上迭代优化每面变换矩阵，通过可微渲染将形变后的源网格渲染为正向流图，用 2D 语义流作为监督

### 关键设计

1. **2D 字符对应模型 (2D Character Correspondence Model)**:

    - 功能：建立源和目标渲染图之间的稳健 2D 逐像素对应
    - 核心思路：将源-目标图像输入 Stable Diffusion（提取 UNet 中间层特征，120×120）和 DINOv2（输出 60×60 特征），拼接并通过轻量级 adapter 网络映射到共同嵌入空间。通过最近邻搜索建立 2D 对应，生成语义流图。训练数据包括 3D 对应数据集（3DBiCar、SURREAL）和 2D 对应数据集（SPair-71K）
    - 设计动机：2D 基础模型在亿级数据上预训练，泛化能力远超 3D 方法。通过渲染将 3D 问题降维到 2D，绕过了 3D 大规模数据集稀缺的难题

2. **几何-引导负样本损失 (Geometry-Grounded Negative Loss)**:

    - 功能：解决 2D 对应中的自相似性问题（如左手和右手特征相似）
    - 核心思路：利用参数化模型（RaBit、SMPL）上预计算的测地距离矩阵 $\mathbb{G}$，对测地距离超过阈值的点对 $(p,q)$，惩罚其特征相似度：$\mathcal{L}_{neg} = \sum_{(p,q), \mathbb{G}(p,q) > th} \|\mathcal{X}_{src}(\Pi(p,C_s)) \cdot \mathcal{X}_{tgt}(\Pi(q,C_t))\|_2$。总训练损失为 $L_{2D} = L_{con} + \lambda_{neg} L_{neg}$
    - 设计动机：标准对比损失无法区分几何上远但语义上相似的部位。利用 3D 参数模型的测地先验作为"硬负样本"的指导，有效消除自相似歧义

3. **语义流引导配准 (Semantic Flow Guided Registration)**:

    - 功能：通过 2D 语义流监督 3D 网格形变，实现稳定的配准
    - 核心思路：使用 Neural Jacobian Fields (NJF) 作为形变模型，优化每面变换矩阵 $\tilde{J}_i \in \mathbb{R}^{3 \times 3}$。NJF 通过投影到切空间的 Jacobian 得到面变换 $J_i = \tilde{J}_i \mathcal{B}_i$，然后求解 Poisson 方程得到形变后顶点位置 $\Phi^* = L^{-1} \mathcal{A} \nabla^T J$。将形变后顶点投影到 2D，计算 2D 位移作为颜色渲染正向流图 $\tilde{S}^i$，用语义流 $S^i$ 作为监督：$\mathcal{L}_{flow} = \sum_i \|\tilde{S}^i - S^i\|_1$
    - 设计动机：NJF 在切空间中参数化形变，比直接优化顶点位移更稳定。通过可微渲染桥接 2D 监督和 3D 形变，避免了传统方法需要高质量初始 3D 对应的问题

### 损失函数 / 训练策略

配准阶段的总损失为：
$\mathcal{L} = \lambda_{flow}\mathcal{L}_{flow} + \lambda_{cd}\mathcal{L}_{cd} + \lambda_{normal}\mathcal{L}_{normal} + \lambda_{identity}\mathcal{L}_{identity} + \lambda_{shear}\mathcal{L}_{shear}$

- $\mathcal{L}_{flow}$：语义流引导损失（权重 10.0）
- $\mathcal{L}_{cd}$：Chamfer Distance 几何对齐损失（权重 1.0）
- $\mathcal{L}_{normal}$：法线一致性损失（权重 0.1）
- $\mathcal{L}_{identity}$：恒等保持项 $\|\tilde{J}_i - I_3\|_F$，权重从 0.01 线性衰减到 0.0001
- $\mathcal{L}_{shear}$：抗剪切项 $\|\tilde{J}_i - \tilde{J}_i^{rot}\|_F$，通过极分解提取旋转分量（权重 0.1），鼓励刚性变换、抑制剪切变形

优化 5000 次迭代，10K 面网格约 2 分钟，40K 面网格约 4 分钟。

## 实验关键数据

### 主实验

**跨域设置（在 3DBiCar+SURREAL 上训练，其他数据集上测试），测地误差 ×100↓：**

| 方法 | 监督类型 | FAUST | CharW | DT4D-H std | DT4D-H hard |
|------|---------|-------|-------|------------|-------------|
| ULRSSM | 无监督 | 2.09 | 32.6 | 28.2 | 32.0 |
| Hybrid ULRSSM | 无监督 | 1.55 | 33.5 | 15.5 | 22.1 |
| Diff3f | 零样本 | 12.0 | 12.5 | 24.0 | 22.7 |
| SmoothShell | 零样本 | 2.93 | 11.6 | 13.6 | 12.4 |
| Ours (Zero-shot) | 零样本 | 5.60 | 3.48 | 19.9 | 14.1 |
| **Ours (Normal)** | **有监督** | **1.83** | **2.61** | **4.23** | **4.12** |

### 消融实验

| 配置 | FAUST | CharW (RGB) | DT4D-H hard | 说明 |
|------|-------|-------------|-------------|------|
| Baseline (顶点位移+拉普拉斯平滑) | 2.88 | 3.44 | 6.04 | 没有 NJF 和特殊损失 |
| + Neural Jacobian Field | 2.32 | 2.69 | 4.58 | NJF 贡献最大 |
| + shear-resistant loss | 2.07 | 2.59 | 4.50 | 抗剪切 loss 持续改进 |
| + geo-grounded neg loss (full) | **1.83** | **2.57** | **4.12** | 几何负样本 loss 进一步提升 |

**2D 对应模型中 feature adapter 的必要性：**

| 配置 | FAUST | CharW | DT4D-H hard |
|------|-------|-------|-------------|
| Diff3f (零样本特征) | 12.0 | 12.5 | 22.7 |
| Ours (零样本, 无 adapter) | 5.60 | 3.48 | 14.1 |
| Ours (有 adapter) | **1.83** | **2.61** | **4.12** |

### 关键发现

- **函数映射方法在非等距场景下全面崩溃**：ULRSSM 在 FAUST（等距）上仅 2.09，但在 CharW（非等距）上飙升至 32.6。Stable-SCore 在 CharW 上仅为 2.61，差距超过 10 倍
- **NJF 是最关键的组件**：从 baseline 到 +NJF，DT4D-H hard 从 6.04 降至 4.58，降幅最大
- **训练 adapter 至关重要**：零样本模式下 DT4D-H hard 为 14.1，加 adapter 后降至 4.12，提升近 4 倍
- **语义流监督优于 3D 对应监督**：直接用 3D 对应（Diff3f 式）作为配准监督的误差为 4.56，用语义流监督仅为 4.12。2D 监督保留了更多语义细节
- **抗剪切损失的精妙作用**：恒等保持项过大会阻碍大姿态变化，过小会导致不平滑。抗剪切项通过鼓励旋转抑制剪切，在保持形变自由度的同时提升稳定性

## 亮点与洞察

- **"2D 基础模型 → 3D 任务" 的桥梁设计**十分巧妙：用渲染将 3D 降维到 2D 利用基础模型的泛化能力，再用可微渲染将 2D 监督反传回 3D，形成闭环。这是利用 2D 预训练知识解决 3D 数据稀缺问题的优雅范式
- **几何引导负样本损失**解决了一个长期困扰对应领域的根本问题——自相似性（左右手、前后腿），利用参数化模型的测地先验提供了几何感知的硬负样本
- **CharW 基准**的引入也很有价值——它首次提供了包含艺术家创作和 AI 生成角色的 wild 场景测试集，推动了非等距对应的研究

## 局限与展望

- **需要粗略的旋转对齐**：源和目标网格需要粗略的朝向一致，无法处理大角度旋转
- **面数限制**：NJF 的优化时间随面数线性增长（40K 面约 4 分钟），对高精度网格可能过慢
- **2D 对应的遮挡问题**：多视角渲染缓解但不能完全解决遮挡——被完全遮挡的区域在所有视角都不可见
- **依赖参数化模型训练**：几何负样本损失需要预计算的测地距离矩阵，限制于有参数化模型的类别（如人体、动物）
- 未来可探索：3D 生成先验（如 3D diffusion）替代 2D 渲染路径、无需参数化模型的自监督训练

## 相关工作与启发

- **vs Diff3f**: Diff3f 将 2D 基础特征反投影到 3D 表面再用函数映射求连续映射。Stable-SCore 认为函数映射的低秩投影会丢失语义信息，因此绕过函数映射，直接用语义流引导配准
- **vs SmoothShell**: SmoothShell 是传统配准方法，零样本 CharW 误差 11.6 vs Ours 2.61。核心差异在于 Stable-SCore 使用了 2D 基础模型提供的语义引导
- **vs 函数映射系列 (ULRSSM, GeoFMap)**: 这些方法在等距场景下强大，但在非等距场景下结构性失败。两种范式的互补性值得探索

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ "2D 基础模型→可微渲染→3D 配准"的 pipeline 设计原创性高，抗剪切损失和几何负样本损失是精妙的技术贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 跨域/域内双设置、多基准、丰富消融、下游应用展示齐全
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰但公式较多，需要一定背景知识
- 价值: ⭐⭐⭐⭐⭐ 在非等距形状对应上的突破性结果，对下游应用（re-topology、rig transfer）有直接价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Stable Score Distillation](../../ICCV2025/3d_vision/stable_score_distillation.md)
- [\[CVPR 2025\] SPAR3D: Stable Point-Aware Reconstruction of 3D Objects from Single Images](spar3d_stable_point-aware_reconstruction_of_3d_objects_from_single_images.md)
- [\[CVPR 2025\] Denoising Functional Maps: Diffusion Models for Shape Correspondence](denoising_functional_maps_diffusion_models_for_shape_correspondence.md)
- [\[CVPR 2025\] A Lightweight UDF Learning Framework for 3D Reconstruction Based on Local Shape Functions](a_lightweight_udf_learning_framework_for_3d_reconstruction_based_on_local_shape_.md)
- [\[CVPR 2025\] PrEditor3D: Fast and Precise 3D Shape Editing](preditor3d_fast_and_precise_3d_shape_editing.md)

</div>

<!-- RELATED:END -->
