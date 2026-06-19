---
title: >-
  [论文解读] MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks
description: >-
  [CVPR 2025][3D视觉][3D Gaussian Splatting] 将 3DGS 中的高斯剪枝从确定性移除改为概率性存在建模，提出 masked-rasterization 技术使未被采样的高斯仍能接收梯度以动态评估其贡献，在 Mip-NeRF360/T&T/DeepBlending 上实现 62-75% 的高斯剪枝率且仅损失 0.02 PSNR。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D Gaussian Splatting"
  - "高斯剪枝"
  - "概率mask"
  - "渲染效率"
  - "masked rasterization"
---

# MaskGaussian: Adaptive 3D Gaussian Representation from Probabilistic Masks

**会议**: CVPR 2025  
**arXiv**: [2412.20522](https://arxiv.org/abs/2412.20522)  
**代码**: [https://github.com/kaikai23/MaskGaussian](https://github.com/kaikai23/MaskGaussian)  
**领域**: 3D视觉  
**关键词**: 3D Gaussian Splatting, 高斯剪枝, 概率mask, 渲染效率, masked rasterization

## 一句话总结

将 3DGS 中的高斯剪枝从确定性移除改为概率性存在建模，提出 masked-rasterization 技术使未被采样的高斯仍能接收梯度以动态评估其贡献，在 Mip-NeRF360/T&T/DeepBlending 上实现 62-75% 的高斯剪枝率且仅损失 0.02 PSNR。

## 研究背景与动机

3D Gaussian Splatting 在新视角合成和实时渲染方面表现出色，但存在高斯冗余问题——单个室内场景可能需要数百万个高斯，导致内存消耗大、渲染速度受限。现有剪枝方法分两类：

- **基于重要性分数的方法**（LightGaussian, RadSplat, Mini-Splatting）：手工设计评分函数，需扫描所有训练图像计算，只能执行一两次剪枝，无法考虑剪枝后场景的动态演化
- **基于可学习 mask 的方法**（Compact3DGS）：将 mask 乘以高斯属性（opacity/scale），但一旦 mask 为 0，opacity/scale 归零 → $\alpha = 0$ → 被 $\alpha$-filter 过滤 → 无法接收梯度 → 永远无法恢复
- **核心问题**：确定性剪枝仅反映当前快照的重要性，不考虑剪枝后场景演化——当前看似不重要的高斯可能在后续训练中变得关键（如精细细节或透明物体），一旦移除则不可逆

核心动机：将高斯建模为概率性存在的实体，通过采样决定每次迭代中哪些高斯参与渲染，并通过特殊设计的 masked-rasterization 让未被采样的高斯也能收到梯度以更新其存在概率。

## 方法详解

### 整体框架

MaskGaussian 为每个高斯维护一个存在概率分布（2 个可学习 mask 分数 + Gumbel-Softmax 采样）。每次迭代：(1) 采样 binary mask 决定高斯的出现/缺席；(2) 所有高斯（包括被 mask 的）正常 splat 计算 $\alpha$，通过 $\alpha$-filter；(3) 在 masked-rasterization 中将 mask 应用于透射率衰减和颜色累积；(4) 被 mask 的高斯虽不影响渲染结果，但仍参与前向计算并接收反向梯度。

### 关键设计

1. **Masked Rasterization（前向）**:
    - 功能：在光栅化过程中应用 mask 而非在高斯属性上，确保被 mask 的高斯不影响渲染但仍参与计算
    - 核心思路：修改 3DGS 的 $\alpha$-blending 方程——颜色累积变为 $c(\mathbf{x}) = \sum_{i=1}^N \mathcal{M}_i \cdot c_i \cdot \alpha_i \cdot T_i$，透射率衰减变为 $T_{i+1} = \mathcal{M}_i \cdot (1-\alpha_i) \cdot T_i + (1-\mathcal{M}_i) \cdot T_i$。当 $\mathcal{M}_i = 0$ 时，颜色贡献被掩盖、透射率不被消耗——高斯表现为"不存在"。关键是 mask 不乘在 opacity/scale 上，所以 $\alpha_i$ 不为 0，高斯仍通过 $\alpha$-filter 参与光栅化
    - 设计动机：Compact3DGS 把 mask 乘在 opacity 上导致 $\alpha=0$，被 filter 剔除后彻底失去梯度反馈；masked-rasterization 将 mask 解耦出高斯属性，保证梯度通路畅通

2. **Masked Rasterization（反向梯度分析）**:
    - 功能：推导 mask 的梯度公式，证明未采样高斯能收到有意义的梯度来更新存在概率
    - 核心思路：mask 梯度为 $\frac{\partial L}{\partial \mathcal{M}_i} = \alpha_i \cdot T_i \cdot \frac{\partial L}{\partial c(\mathbf{x})} \cdot (c_i - b_{i+1})$，其中 $b_{i+1}$ 是第 $i$ 个高斯后方的颜色。解读：(1) $\alpha_i \cdot T_i$ 是该高斯对颜色的影响权重；(2) $\frac{\partial L}{\partial c(\mathbf{x})} \cdot (c_i - b_{i+1})$ 衡量使用该高斯的颜色 $c_i$ 相比不使用（用后方颜色 $b_{i+1}$）的收益。若点积为正，说明该高斯有贡献，其存在概率应增加——即使当前被 mask
    - 设计动机：这个梯度公式已经自然包含了 $\alpha_i \cdot T_i$（即 score-based 方法的重要性准则），并额外捕捉了"当前颜色 vs 后方颜色"的对比信息——score-based 方法无法衡量这一点

3. **概率采样 + 稀疏化训练**:
    - 功能：自动学习每个高斯的存在概率并执行动态剪枝
    - 核心思路：每个高斯有 2 个 mask score，通过 Gumbel-Softmax 采样得到可微 binary mask $\mathcal{M}_i \in \{0,1\}$。使用平方 mask 损失 $L_m = (\frac{1}{N}\sum_i \mathcal{M}_i)^2$ 约束采样高斯的平均数量（比 L1 效果更好）。剪枝策略：每步采样 10 次，从未被采样到的高斯视为低概率并移除；在 densification 期间的每步和之后每 1000 迭代执行
    - 设计动机：确定性 mask 一旦移除不可逆；概率性采样允许高斯在不同迭代中动态出现/消失，适应场景演化

### 损失函数

- 总损失：$L = L_{render} + \lambda_m \cdot L_m$
- $L_{render}$：标准 3DGS 渲染损失（L1 + SSIM）
- $L_m = (\frac{1}{N}\sum_i \mathcal{M}_i)^2$：平方 mask 正则化（约束高斯使用数量）
- $\lambda_m$：权衡系数，Ours-$\alpha$ 在 19K-20K 迭代用 0.1，Ours-$\beta$ 全程 0.0005，Ours-$\gamma$ 全程 0.001

## 实验关键数据

### 主实验

| 方法 | Mip-NeRF360 PSNR↑ | #GS↓(M) | FPS↑ | T&T PSNR↑ | #GS↓(M) | DeepBlend PSNR↑ | #GS↓(M) |
|------|-------------------|---------|------|-----------|---------|----------------|---------|
| 3DGS | 27.45 | 3.204 | 187.8 | 23.74 | 1.825 | 29.53 | 2.815 |
| Compact3DGS | 27.32 | 1.533 | 281.1 | 23.61 | 0.960 | 29.58 | 1.310 |
| RadSplat | 27.45 | 2.184 | 247.8 | 23.61 | 1.053 | 29.55 | 1.515 |
| **MaskGaussian** | **27.43** | **1.205** | **384.7** | **23.72** | **0.590** | **29.69** | **0.694** |

### 与 Compact3DGS 对照消融

| 方法 | Mip PSNR | #GS(M) | T&T PSNR | #GS(M) | DB PSNR | #GS(M) |
|------|---------|--------|---------|--------|---------|--------|
| Compact3DGS | 27.32 | 1.533 | 23.61 | 0.960 | 29.58 | 1.310 |
| Ours-$\beta$ (同设置) | **27.44** | 1.520 | **23.66** | **0.740** | **29.76** | **0.913** |
| Ours-$\gamma$ (强压缩) | 27.42 | **1.171** | 23.59 | **0.549** | **29.74** | **0.570** |

### 关键发现

- 剪枝率达 62.4%/67.7%/75.3%（Mip/T&T/DB），渲染加速 2.05×/2.19×/3.16×，PSNR 仅降 0.02
- **Deep Blending 上渲染质量反而提升**（29.69 vs 3DGS 的 29.53），归因于剪枝的正则化效果
- 同等训练设置下（Ours-$\beta$），MaskGaussian 在所有数据集上同时达到更高 PSNR 和更少高斯
- 在自行车场景中成功保留了 Compact3DGS 丢失的轮胎充气嘴、穿透辐条等精细透明结构
- 可与 Taming-3DGS 等优化框架无缝集成，进一步提升效率
- 将 mask 乘在 opacity/scale 上（Compact3DGS 方式）vs masked-rasterization：后者 PSNR 高约 0.3-0.5

## 亮点与洞察

- **概率 vs 确定性剪枝的本质区别**：确定性剪枝基于"快照"评估——当前不重要的高斯被永久移除；概率性存在允许高斯在训练过程中"复活"，随场景动态演化而调整
- **梯度公式的优雅自然性**：mask 梯度自然包含了 $\alpha_i \cdot T_i$（已有重要性准则）加上"颜色贡献增益"——形式简洁且物理意义清晰
- **mask 应用位置的核心洞察**：mask 不应乘在高斯属性上（否则断梯度），而应在光栅化的 blending 过程中应用——这个看似微小的设计差异导致了质的变化
- **与 Vision Transformer 动态 pruning 的类比**：借鉴 ViT 中 token pruning 让 active 和 inactive token 同时收到梯度的思路，是跨领域知识迁移的好例子

## 局限与展望

- 训练时间略长于 3DGS（多了 mask 采样和 masked-rasterization 的开销）
- $\lambda_m$ 的设置对最终剪枝率和质量影响较大，需要针对不同场景/需求调整
- 仅在 3DGS 基础上验证，未扩展到 2DGS、Scaffold-GS 等变体
- 概率采样引入随机性，训练过程可能不够稳定（尤其是初期）
- CUDA 内核修改增加了工程复杂度，不如简单的后处理剪枝易于部署

## 相关工作与启发

- 与 Compact3DGS 的核心区别：Compact3DGS 将 mask 乘在 opacity/scale 上是确定性的，一旦 mask=0 高斯永失梯度；MaskGaussian 在光栅化中应用 mask，保证概率性动态评估
- 与 LightGaussian/RadSplat 的区别：score-based 方法只在某一时刻的静态快照上评估重要性；MaskGaussian 通过迭代概率采样实现了对场景动态演化的自适应
- 启发：3DGS 剪枝的关键不仅是"评估哪些高斯重要"，更是"在场景不断变化的过程中持续重新评估"——概率性采样 + 梯度回传是实现这一目标的有效框架

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 从确定性到概率性剪枝的范式转变，masked-rasterization 的数学推导和 CUDA 实现均有深度
- 实验充分度: ⭐⭐⭐⭐ 三个数据集全面对比，消融详尽（mask 应用位置、损失形式、超参数），但缺少户外大场景
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，梯度推导严谨，可视化对比（轮胎充气嘴、枯藤等细节）极具说服力
- 价值: ⭐⭐⭐⭐ 为 3DGS 效率优化提供了新思路，masked-rasterization 作为通用 mask 接口有更广泛的应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] EigenGS: Representation from Eigenspace to Gaussian Image Space](eigengs_representation_from_eigenspace_to_gaussian_image_space.md)
- [\[ICCV 2025\] Discretized Gaussian Representation for Tomographic Reconstruction](../../ICCV2025/3d_vision/discretized_gaussian_representation_for_tomographic_reconstruction.md)
- [\[CVPR 2025\] COB-GS: Clear Object Boundaries in 3DGS Segmentation Based on Boundary-Adaptive Gaussian Splitting](cob-gs_clear_object_boundaries_in_3dgs_segmentation_based_on_boundary-adaptive_g.md)
- [\[ICCV 2025\] CLIP-GS: Unifying Vision-Language Representation with 3D Gaussian Splatting](../../ICCV2025/3d_vision/clip-gs_unifying_vision-language_representation_with_3d_gaussian_splatting.md)
- [\[NeurIPS 2025\] DC4GS: Directional Consistency-Driven Adaptive Density Control for 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/dc4gs_directional_consistency-driven_adaptive_density_control_for_3d_gaussian_sp.md)

</div>

<!-- RELATED:END -->
