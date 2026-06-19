---
title: >-
  [论文解读] Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details
description: >-
  [AAAI 2026][神经符号距离函数] 提出一种双分支架构（泛化分支+过拟合分支）来学习多个神经SDF的紧凑潜空间，结合共享spatial feature grid和新颖的带宽采样策略，在保持紧凑latent code的同时恢复高保真几何细节，在Stanford Models、ShapeNet和D-FAUST上均达到SOTA。
tags:
  - "AAAI 2026"
  - "神经符号距离函数"
  - "隐式表示"
  - "紧凑潜空间"
  - "体积网格"
  - "几何细节"
---

# Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details

**会议**: AAAI 2026  
**arXiv**: [2511.14539](https://arxiv.org/abs/2511.14539)  
**代码**: [GitHub](https://github.com/eoozbq/Compact-SDF)  
**领域**: 其他  
**关键词**: 神经符号距离函数, 隐式表示, 紧凑潜空间, 体积网格, 几何细节

## 一句话总结

提出一种双分支架构（泛化分支+过拟合分支）来学习多个神经SDF的紧凑潜空间，结合共享spatial feature grid和新颖的带宽采样策略，在保持紧凑latent code的同时恢复高保真几何细节，在Stanford Models、ShapeNet和D-FAUST上均达到SOTA。

## 研究背景与动机

神经符号距离函数（Neural SDF）是三维形状表示的核心方法，通过神经网络参数化一个连续的隐式函数，可以在任意空间坐标处查询到该点到物体表面的符号距离。然而，现有方法在同时表示**多个**SDF并保留高保真几何细节时面临严重瓶颈。

**现有方法的两大路线及其局限**：

**泛化路线**（如DeepSDF）：将多个形状编码到一个共享的全局latent space中，利用MLP解码。优势是能泛化到新形状，但受限于网络对低频信号的偏好（spectral bias），无法恢复高频几何细节（如尖锐边缘、细小孔洞）。

**过拟合路线**（如Instant-NGP、MosaicSDF）：利用体积网格或多分辨率哈希表存储空间特征，能恢复精细细节。但这类方法通常只针对单个形状进行过拟合，缺乏用于表示多个形状的紧凑潜空间。当多个形状共享一个feature grid时，不同形状之间的采样不平衡会导致相互干扰，产生伪影。

**核心矛盾**：泛化能力与高频细节恢复能力难以兼顾。泛化方法有紧凑编码但丢失细节，过拟合方法保留细节但缺少共享空间。

**本文切入角度**：将两种路线的优势结合——用泛化分支处理远离表面的区域（不需密集采样也能产生合理的SDF值），用过拟合分支处理近表面区域（密集采样以恢复高频细节），两个分支共享同一个紧凑的形状编码 $\mathbf{z}$。

## 方法详解

### 整体框架

方法由两个并行分支构成，共享一个可学习的latent code $\mathbf{z}$：

- **泛化分支**（Generalization Branch）：输入位置编码 $PE(\mathbf{x})$ 和形状编码 $\mathbf{z}$，输出粗糙的SDF值 $s_g$。依靠神经网络的泛化能力，即使远离表面区域只有稀疏采样也能给出合理预测。
- **过拟合分支**（Overfitting Branch）：维护一个共享的空间特征网格（$128^3$ 分辨率，每个顶点128维特征），通过三线性插值获取查询点的局部特征 $\mathbf{c}$，与 $\mathbf{x}$ 和 $\mathbf{z}$ 一起输入网络，输出精细的SDF值 $s_o$。利用grid的高频拟合能力恢复表面几何细节。

推理时通过**符号距离融合**策略组合两个分支的输出：先用泛化分支做粗糙重建，确定表面所在的体素带（bandwidth $\mathbb{B}$），带内使用过拟合分支的预测，带外使用泛化分支的预测。

### 关键设计

1. **双分支架构与符号距离融合**:

    - 功能：将SDF的查询空间分为近表面和远表面两个区域，分别用不同分支处理
    - 核心思路：融合公式为 $s = s_o$ 当 $\mathbf{x} \in \mathbb{B}$，否则 $s = s_g$。其中 $\mathbb{B}$ 是表面两侧各扩展 $n$ 层体素形成的带宽区域
    - 设计动机：泛化分支在远表面能消除因采样不平衡导致的伪影（不同形状的特征干扰），过拟合分支在近表面能恢复尖锐边缘和精细结构。两者互补。

2. **带平衡约束的采样策略**:

    - 功能：为过拟合分支设计一种新的训练采样方案
    - 核心思路：先以低分辨率（$128^3$）均匀采样整个空间得到稀疏样本；再以高分辨率（$512^3$）在表面带宽 $\mathbb{B}$ 内密集采样；合并去重
    - 设计动机：直接在高分辨率下对所有形状保证每个体素都有平衡采样，计算代价是 $O(N \times R^3)$ 不可接受。该策略通过只在近表面密集采样、远表面稀疏采样来平衡效率和质量，同时泛化分支弥补了远表面的稀疏采样不足

3. **共享空间特征网格**:

    - 功能：多个形状共享同一个 $128^3 \times 128$ 维的特征网格，每个形状通过latent code $\mathbf{z}$ 区分
    - 核心思路：查询点 $\mathbf{x}$ 通过三线性插值从网格获取特征 $\mathbf{c}$，再与 $\mathbf{z}$ 一起解码
    - 设计动机：unlike哈希网格（Instant-NGP），显式的共享网格避免了哈希碰撞，配合形状编码能更好区分不同形状

### 损失函数 / 训练策略

总损失函数包含四项：

$$\mathcal{L} = \mathcal{L}_{\text{gen}} + \mathcal{L}_{\text{ovf}} + \lambda_1 \mathcal{L}_z + \lambda_2 \mathcal{L}_c$$

- $\mathcal{L}_{\text{gen}}$: 泛化分支的MSE损失
- $\mathcal{L}_{\text{ovf}}$: 过拟合分支的MSE损失
- $\mathcal{L}_z$: 形状编码 $\mathbf{z}$ 的正则化（防过拟合、鼓励紧凑性）
- $\mathcal{L}_c$: 网格特征 $\mathbf{c}$ 的正则化

训练细节：网格特征学习率 $10^{-1}$，形状编码学习率 $10^{-3}$，两个网络均为8层512单元MLP，latent code 256维，共训练4000 epoch。

## 实验关键数据

### 主实验（单个复杂物体重建 — Stanford Models）

| 方法 | CD(↓) | F-Score(↑) | Precision(↑) | Recall(↑) |
|------|-------|-----------|-------------|----------|
| ACORN | 6.76e-05 | 0.982 | 0.967 | 0.998 |
| NGLOD | 6.77e-05 | 0.980 | 0.968 | 0.994 |
| Instant-NGP | 7.37e-05 | 0.976 | 0.962 | 0.990 |
| MosaicSDF | 1.30e-03 | 0.902 | 0.846 | 0.972 |
| HyperDiffusion | 1.20e-04 | 0.835 | 0.847 | 0.823 |
| **Ours** | **6.56e-05** | **0.983** | **0.969** | **0.998** |

### 多物体重建（ShapeNet, CD×10⁻⁴）

| 方法 | Bench | Chair | Plane | Table | Lamp | Sofa | Avg. |
|------|-------|-------|-------|-------|------|------|------|
| DeepSDF | 4.890 | 8.630 | 2.660 | 6.330 | 14.63 | 5.040 | 7.030 |
| IF-NET | 1.340 | 1.000 | 0.225 | 0.857 | 0.817 | 1.100 | 0.890 |
| Instant-NGP | 0.881 | 1.210 | 0.664 | 1.030 | 1.880 | 1.100 | 1.128 |
| HyperDiffusion | 1.640 | 1.490 | 2.310 | 1.470 | 6.990 | 2.830 | 2.788 |
| **Ours** | **0.463** | **0.898** | **0.223** | **0.790** | **0.517** | **0.751** | **0.607** |

### 消融实验

| 配置 | CD(×10⁻⁴) | 说明 |
|------|-----------|------|
| 双分支(Ours) | 4.01 | 完整方法 |
| 仅泛化分支 | 4.05 | 丢失高频细节 |
| 仅过拟合分支 | 11.6 | 远表面出现严重伪影 |
| 带宽n=1 | 4.29 | 带宽过窄，过拟合分支贡献不足 |
| 带宽n=6 | 14.6 | 带宽过大，伪影从过拟合分支传递 |

| 采样策略 | CD(×10⁻⁴) | 采样量 |
|---------|-----------|-------|
| 均匀 $32^3$ | 35.900 | $32^3$ |
| 均匀 $128^3$ | 0.890 | $128^3$ |
| 均匀 $256^3$ | 0.417 | $256^3$ |
| **Ours** | **0.401** | **≈$220^3$** |

### 关键发现

- 多物体重建中，本方法平均CD（0.607×10⁻⁴）比第二名IF-NET（0.890）低32%
- 带宽参数 $n=3$ 是最佳权衡点，过大过小都会降低性能
- latent code维度在64维时性能趋于饱和（CD从d=8的0.443降到d=64的0.397）
- 本方法的采样策略用约 $220^3$ 样本就超过了 $256^3$ 均匀采样的效果

## 亮点与洞察

- 核心洞察很精准：泛化和过拟合本质上是SDF不同空间区域的最优策略，将两者按空间分工融合是一个简洁而有效的思路
- 共享feature grid上不同形状的采样不平衡问题被识别并通过带宽限制+泛化分支补偿巧妙解决
- 在shape interpolation实验中，本方法比DeepSDF有更显著的插值变化，比HyperDiffusion更紧凑（256维 vs 整个MLP参数）

## 局限与展望

- 方法需要watertight mesh作为输入，无法直接处理点云或带噪声的扫描数据
- $128^3$ 的feature grid内存开销随类别数增加线性增长
- 推理需要两步（先粗糙重建确定带宽，再精细融合），增加了推理时间
- 形状编码维度在100个形状时64维饱和，大规模数据集可能需要更高维度，可扩展性有待验证

## 相关工作与启发

- 与MosaicSDF的关键区别：MosaicSDF将grid聚焦在表面附近，但仍是单物体方法；本文将grid扩展到多物体共享并用latent code区分
- Instant-NGP的哈希碰撞问题在多物体场景中尤为严重，这解释了它在多物体任务中性能退化的原因
- 该双分支思想可能可以推广到NeRF或其他隐式表示任务中

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] InstantRetouch: Efficient and High-Fidelity Instruction-Guided Image Retouching with Bilateral Space](../../CVPR2026/others/instantretouch_efficient_and_high-fidelity_instruction-guided_image_retouching_w.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](../../ICLR2026/others/out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)
- [\[ICLR 2026\] A Single Architecture for Representing Invariance Under Any Space Group](../../ICLR2026/others/a_single_architecture_for_representing_invariance_under_any_space_group.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[CVPR 2025\] HotSpot: Signed Distance Function Optimization with an Asymptotically Sufficient Condition](../../CVPR2025/others/hotspot_signed_distance_function_optimization_with_an_asymptotically_sufficient_.md)

</div>

<!-- RELATED:END -->
