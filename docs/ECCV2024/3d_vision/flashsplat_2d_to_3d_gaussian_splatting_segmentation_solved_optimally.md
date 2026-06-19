---
title: >-
  [论文解读] FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally
description: >-
  [ECCV 2024][3D视觉][3D高斯溅射] 将3D高斯溅射的2D-to-3D分割问题建模为整数线性规划，利用alpha混合的线性性质得到闭式最优解，仅需30秒完成优化，比现有方法快50倍。 领域现状： 3D高斯溅射（3D-GS）作为一种高效的3D场景表示方法迅速兴起，但从2D mask将分割结果提升到3D（即为每个3…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "3D高斯溅射"
  - "3D分割"
  - "线性规划"
  - "最优求解"
  - "物体移除"
---

# FlashSplat: 2D to 3D Gaussian Splatting Segmentation Solved Optimally

**会议**: ECCV 2024  
**arXiv**: [2409.08270](https://arxiv.org/abs/2409.08270)  
**代码**: [https://github.com/florinshen/FlashSplat](https://github.com/florinshen/FlashSplat)  
**领域**: 3D视觉  
**关键词**: 3D高斯溅射, 3D分割, 线性规划, 最优求解, 物体移除

## 一句话总结

将3D高斯溅射的2D-to-3D分割问题建模为整数线性规划，利用alpha混合的线性性质得到闭式最优解，仅需30秒完成优化，比现有方法快50倍。

## 研究背景与动机

**领域现状**: 3D高斯溅射（3D-GS）作为一种高效的3D场景表示方法迅速兴起，但从2D mask将分割结果提升到3D（即为每个3D高斯分配标签）是一个关键且尚未很好解决的问题。

**现有痛点**: 现有方法（SAGA、Gaussian Grouping）依赖迭代梯度下降优化来训练每个高斯的标签/特征，需要30000次迭代、18-37分钟的额外训练时间，容易陷入次优解。

**核心矛盾**: 直觉上3D-GS分割应该很简单——场景已经重建好了，只是标签分配问题——但现有方法把它当作需要大量优化的学习任务来处理，效率极低。

**本文目标**: 发现3D-GS分割问题的数学结构，提供全局最优且高效的求解方案。

**切入角度**: 关键洞察——在已重建的3D-GS场景中，2D mask的渲染关于每个高斯的标签是**线性函数**，因此最优标签分配可以通过线性规划闭式求解。

**核心 idea**: 利用alpha混合中 $\alpha_i T_i$ 为常数的性质，将3D-GS分割转化为线性规划问题，通过加权多数投票一步求得全局最优解。

## 方法详解

### 整体框架

输入：已重建的3D-GS场景 $\{G_i\}$ + 多视角2D binary/instance masks $\{M^v\}$。
目标：为每个3D高斯 $G_i$ 分配标签 $P_i$。
流程：(1) 利用3D-GS光栅化器遍历所有mask视角，累积每个高斯在前景/背景像素上的加权贡献 $A_0, A_1$；(2) 通过argmax一步确定最优标签分配；(3) 可选地引入背景偏置 $\gamma$ 减少噪声。整个过程约30秒。

### 关键设计

1. **二值分割的整数线性规划建模 (Binary Segmentation as ILP)**:

    - **功能**: 将2D-to-3D分割问题严格建模为整数线性规划。
    - **核心思路**: 3D-GS渲染中，像素值 $X = \sum_i x_i \alpha_i T_i$，其中 $\alpha_i T_i$ 在场景重建完成后是常数。将标签 $P_i$ 作为属性 $x_i$，优化目标为最小化渲染mask与给定mask的MAE：
    $\min_{\{P_i\}} \mathcal{F} = \sum_{v \in L} \sum_{M_{jk}^v \in M^v} \left| \sum_i P_i \alpha_i T_i - M_{jk}^v \right|, \quad P_i \in \{0,1\}$
   由alpha混合的性质（**引理1**）：$0 \leq \sum P_i \alpha_i T_i \leq \sum \alpha_i T_i \leq 1$，可以将绝对值展开得到：
    $\min \mathcal{F} = C + \sum_i P_i (A_0^i - A_1^i)$
   其中 $A_n^i = \sum_{v,j,k} \alpha_i T_i \mathbb{I}(M_{jk}^v, n)$ 是第 $i$ 个高斯在所有标签为 $n$ 的像素上的总贡献。
    - **最优解**: $P_i = \arg\max_n A_n$，即加权多数投票——如果一个高斯对前景像素的加权贡献大于对背景像素的贡献，就标记为前景。
    - **设计动机**: 由于目标函数关于 $P_i$ 是线性的（每个 $P_i$ 独立），且 $P_i \in \{0,1\}$，可以逐个高斯独立求解，所有高斯并发完成——这就是全局最优解。

2. **带背景偏置的正则化 (Regularized ILP with Background Bias)**:

    - **功能**: 通过引入背景偏置参数 $\gamma \in [-1, 1]$ 来处理2D mask中的噪声。
    - **核心思路**: 先对贡献做L1归一化 $\bar{A_e} = A_e / \sum_t A_t$，然后调整 $\hat{A_0} = \bar{A_0} + \gamma$。正的 $\gamma$ 倾向将更多高斯分为背景（减少前景噪声），负的 $\gamma$ 则保留更多前景高斯。
    - **设计动机**: SAM等模型生成的2D mask通常含噪声，导致背景高斯被错误标记为前景，产生锯齿边缘。$\gamma$ 提供了一个简单但灵活的噪声控制旋钮，且因为 $\{A_e\}$ 只需计算一次，调整 $\gamma$ 是即时的（0.4ms）。

3. **从二值分割到场景分割 (Binary to Scene Segmentation)**:

    - **功能**: 将方法扩展到同时分割场景中的所有物体实例。
    - **核心思路**: 将多实例分割重新解释为多次二值分割的组合。对于目标实例 $t$，将其他所有实例视为背景：$A_0 = A_{\text{others}} = \sum_{e \neq t} A_e$，然后 $P_i = \arg\max_n \{A_0, A_t\}$。只需累积 $\{A_e\}$ 集合一次，每个实例的分割只需一次argmax。
    - **设计动机**: 3D高斯天然不互斥——一个高斯可能同时贡献于多个物体（约20%的高斯被多个物体共享），因此允许不同实例的高斯子集重叠是合理的。

4. **深度引导的新视角Mask渲染 (Depth-Guided Novel View Mask Rendering)**:

    - **功能**: 在新视角下渲染分割结果的2D mask。
    - **核心思路**: 二值分割下，只渲染前景高斯的累积alpha值 $\rho_{jk}$，用阈值 $\tau$ 量化得到mask。场景分割下，如果多个实例的alpha超过阈值，用渲染深度选择最近的实例。
    - **设计动机**: 3D高斯的半透明特性导致不做量化时前景mask会有很多空洞（背景高斯的alpha贡献可能超过前景）。深度引导解决多实例重叠的歧义问题。

### 实现细节

- 核心计算在CUDA kernel中实现：利用tile-based光栅化遍历所有mask，对每个高斯原子操作累积 $A_e$
- 累积 $\{A_e\}$ 约26秒（遍历所有视角的所有像素）
- argmax分配仅需0.4ms
- 峰值显存仅8G，是SAGA的一半

## 实验关键数据

### 主实验：NVOS数据集定量比较

| 方法 | mIoU (%)↑ | mAcc (%)↑ |
|------|----------|----------|
| NVOS | 39.4 | 73.6 |
| ISRF | 70.1 | 92.0 |
| SGISRF | 83.8 | 96.4 |
| SA3D | 90.3 | 98.2 |
| SAGA | 90.9 | 98.3 |
| **FlashSplat** | **91.8** | **98.6** |

### 计算开销对比

| 方法 | 额外训练时间 | 优化步数 | 单次分割时间 | 峰值显存 |
|------|------------|---------|------------|---------|
| SAGA | 18分钟 | 30000 | 0.5秒 | 15G |
| Gaussian Grouping | 37分钟 | 30000 | 0.3秒 | 34G |
| **FlashSplat** | **26秒** | **1** | **0.4ms** | **8G** |

### 消融实验

| 背景偏置γ | mIoU (Truck场景) | 说明 |
|----------|-----------------|------|
| -0.8 | 82.4% | 过多前景噪声 |
| -0.4 | 89.6% | - |
| 0 | 92.3% | 无偏置基线 |
| 0.4 | **94.2%** | 最优，减少SAM噪声 |
| 0.8 | 93.8% | 略过度清理 |

### 关键发现

- FlashSplat在NVOS数据集上mIoU达91.8%，超越所有NeRF和3D-GS基线
- 速度提升约**50倍**（26秒 vs 18-37分钟），单次分割时间**750倍**加速（0.4ms vs 0.3-0.5s）
- 仅需约10%的视角mask即可产生不错的分割结果（360度场景下1/8视角仍可用）
- 背景偏置 $\gamma=0.4$ 在Truck场景上将mIoU从92.3%提升到94.2%
- 约20%的3D高斯被2个以上物体共享——3D高斯天然不互斥
- 物体移除质量优于Gaussian Grouping，后者在移除区域附近有严重伪影

## 亮点与洞察

- **问题建模的优雅性**是本文最大的亮点——将一个看似需要优化的问题变成了一个有闭式解的问题。关键洞察是alpha混合后 $\alpha_i T_i$ 为常数，使得目标函数关于标签线性
- 引理1的推导虽然简单但很关键——保证了渲染mask值在[0,1]范围内，使得绝对值可以展开
- $\gamma$ 参数的设计直觉且实用——一旦 $\{A_e\}$ 计算完成，用户可以交互式调整 $\gamma$ 实时看到分割变化
- 将场景分割分解为多次二值分割并允许重叠是对3D-GS特性的正确理解
- 代码简洁——核心算法的PyTorch实现仅约20行

## 局限与展望

- 需要遍历所有mask的所有像素来累积 $\{A_e\}$，对超大场景可能有可扩展性问题
- 3D-GS缺乏显式几何监督导致学到的几何不完全准确，深度引导的mask渲染有时产生模糊结果
- 依赖SAM等2D分割模型的质量和mask关联（视频跟踪器）的准确性
- 面向前方视角（如LLFF数据集）在物体移除后会暴露未观察到的背景区域
- 物体边界处的分割可能不够精细——线性规划假设每个高斯的贡献独立，忽略了空间连续性先验
- 未与语义/开放词汇分割方法结合

## 相关工作与启发

- **vs SAGA**: SAGA需要为每个3D高斯训练额外的特征维度+多种损失函数（约30000步），是一种过度工程化的方案。FlashSplat证明这个问题存在闭式解，根本不需要迭代优化。
- **vs Gaussian Grouping**: 使用视频跟踪器关联2D mask后蒸馏为物体特征+分类器训练，更复杂但不一定更好。FlashSplat直接从mask到3D标签，无需学习特征。
- **vs SAGS**: 将3D高斯中心投影到2D mask判断前景/背景，是一种无需训练的方法但过于简化——忽略了高斯的空间范围和alpha混合权重。FlashSplat考虑了完整的渲染贡献。
- **启发**: 在分析已重建表示的属性分配问题时，应首先检查目标函数的数学结构——线性结构意味着有闭式解，不需要梯度下降。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将优化问题转化为闭式解的洞察非常精彩，简单、优雅、全局最优
- 实验充分度: ⭐⭐⭐⭐ 定量+定性+消融+计算开销对比完整，但定量评估仅在NVOS（8个前向场景）上进行较局限
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，从二值到场景分割的扩展自然，但部分内容有重复
- 价值: ⭐⭐⭐⭐⭐ 50倍加速+全局最优+更少显存+更好质量，实用价值极高，对3D-GS分割任务是近乎终结性的工作

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] 3×2: 3D Object Part Segmentation by 2D Semantic Correspondences](3x2_3d_object_part_segmentation_by_2d_semantic_correspondenc.md)
- [\[ECCV 2024\] GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting](gaussianimage_1000_fps_image_representation_and_compression_by_2d_gaussian_splat.md)
- [\[CVPR 2025\] Rethinking End-to-End 2D to 3D Scene Segmentation in Gaussian Splatting](../../CVPR2025/3d_vision/rethinking_end-to-end_2d_to_3d_scene_segmentation_in_gaussian_splatting.md)
- [\[ECCV 2024\] Surface Reconstruction from 3D Gaussian Splatting via Local Structural Hints](surface_reconstruction_from_3d_gaussian_splatting_via_local_structural_hints.md)
- [\[ECCV 2024\] GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting](gs-lrm_large_reconstruction_model_for_3d_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
