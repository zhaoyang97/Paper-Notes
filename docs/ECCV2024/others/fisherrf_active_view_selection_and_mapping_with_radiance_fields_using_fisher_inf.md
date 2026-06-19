---
title: >-
  [论文解读] FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information
description: >-
  [ECCV 2024][Fisher信息] 本文提出FisherRF，利用Fisher信息直接量化辐射场（Radiance Fields）模型参数的观测信息量，通过最大化期望信息增益（Expected Information Gain）选择最优视角，在视角选择、主动建图和不确定性量化三个任务上均达到SOTA，且通过稀疏性利用和自定义CUDA核实现了70 fps的视角评估速度。
tags:
  - "ECCV 2024"
  - "Fisher信息"
  - "主动视角选择"
  - "3D高斯溅射"
  - "不确定性量化"
  - "主动建图"
---

# FisherRF: Active View Selection and Mapping with Radiance Fields Using Fisher Information

**会议**: ECCV 2024  
**arXiv**: [2311.17874](https://arxiv.org/abs/2311.17874)  
**代码**: 无  
**领域**: 其他  
**关键词**: Fisher信息, 主动视角选择, 3D高斯溅射, 不确定性量化, 主动建图

## 一句话总结

本文提出FisherRF，利用Fisher信息直接量化辐射场（Radiance Fields）模型参数的观测信息量，通过最大化期望信息增益（Expected Information Gain）选择最优视角，在视角选择、主动建图和不确定性量化三个任务上均达到SOTA，且通过稀疏性利用和自定义CUDA核实现了70 fps的视角评估速度。

## 研究背景与动机

1. **领域现状**: 神经辐射场（NeRF）及3D Gaussian Splatting等技术极大推进了图像渲染和3D重建。然而，训练高质量辐射场模型需要大量不同视角的图像，而图像采集成本高昂。因此，如何高效选择最有信息量的视角进行拍摄成为关键问题。现有的主动视角选择方法主要分为两类：白盒方法（修改模型架构嵌入贝叶斯模型来近似不确定性）和黑盒方法（通过预测分布来间接评估不确定性）。

2. **现有痛点**: 白盒方法依赖特定模型架构且训练速度慢；黑盒方法（如ActiveNeRF通过添加方差输出、BayesRays通过假设扰动场）只能间接近似模型不确定性，从间接近似中选择视角并不保证对模型产生最优的信息增益。此外，这些方法与新型辐射场表征（如3D Gaussian Splatting）不兼容，因为后者不使用查询点（query points）作为输入。

3. **核心矛盾**: 辐射场模型的参数空间巨大（通常超过2亿个可优化参数），直接计算完整的Fisher信息矩阵/Hessian矩阵是不可行的。如何在保证理论合理性的同时实现可计算的信息增益量化是核心技术挑战。

4. **本文目标**: 提供一种理论驱动的、高效的方法来量化辐射场中的观测信息并选择最优视角，同时适用于多种辐射场表征（NeRF、3DGS、Plenoxels），并扩展到批量视角选择、主动建图和像素级不确定性量化。

5. **切入角度**: Fisher信息本质上是似然函数关于模型参数的Hessian矩阵，在体积渲染中这个Hessian不依赖于真实观测（ground truth图像），只需要候选视角的相机参数即可计算。这意味着可以在不实际拍摄的情况下评估每个候选视角的信息增益。此外，由于辐射场模型的局部参数结构，Hessian矩阵高度稀疏，可以利用这种稀疏性实现高效计算。

6. **核心 idea**: 通过Fisher信息直接度量辐射场参数的信息增益来选择下一个最佳视角，不需要修改模型架构、不依赖间接的不确定性近似，且计算高效。

## 方法详解

### 整体框架

FisherRF的核心流程：（1）用初始少量视角训练辐射场模型得到参数 $\mathbf{w}^*$；（2）对每个候选视角，通过Fisher信息计算其相对于训练集的期望信息增益（EIG）；（3）选择EIG最高的候选视角作为下一个拍摄位置；（4）获取该视角的真实图像后加入训练集重新训练模型；（5）重复上述过程直到达到预定视角数。该框架在3D Gaussian Splatting和Plenoxels两种辐射场模型上实现。

### 关键设计

1. **Fisher信息在体积渲染中的推导**: 核心理论贡献。在辐射场中，负对数似然为 $-\log p(\mathbf{y}|\mathbf{x},\mathbf{w}) = (\mathbf{y}-f(\mathbf{x},\mathbf{w}))^T(\mathbf{y}-f(\mathbf{x},\mathbf{w}))$，其Fisher信息即Hessian矩阵为 $\mathbf{H}'' = \nabla_\mathbf{w}f^T \nabla_\mathbf{w}f$。关键在于：这个Hessian只依赖渲染函数的Jacobian，不依赖真实图像 $\mathbf{y}$——这使得可以在不拍摄的情况下评估每个候选视角的信息含量。期望信息增益的优化目标为 $\arg\max_{\mathbf{x}_i^{acq}} \text{tr}(\mathbf{H}''[\mathbf{y}_i|\mathbf{x}_i,\mathbf{w}^*] \cdot \mathbf{H}''[\mathbf{w}^*|\mathcal{D}_{train}]^{-1})$。由于参数空间巨大，使用Laplace近似对Hessian矩阵取对角化：$\mathbf{H}'' \approx \text{diag}(\nabla_\mathbf{w}f^T\nabla_\mathbf{w}f) + \lambda I$。

2. **批量视角选择的贪心算法**: 实际应用中经常需要一次选择多个视角。简单地对每个视角独立最大化EIG会选到相似的视角（信息冗余）。FisherRF利用Fisher信息的可加性，设计了贪心优化算法：每次选择一个最优视角后，将其Hessian加入已有的训练Hessian中再选下一个视角。由于每个参数只受限定空间区域内的射线影响，Hessian矩阵高度稀疏，这使得矩阵乘积的计算与反向传播一样高效。自定义CUDA核函数实现的对角Hessian计算仅需11.3ms，对比传统PyTorch backward引擎的1.1s，快约100倍。

3. **像素级不确定性量化**: FisherRF还能导出像素级的不确定性度量。在3D Gaussian Splatting中，每个参数直接对应3D空间中的一个位置。对于渲染的每个像素，沿射线（ray）方向贡献颜色的3D Gaussian参数的Fisher信息可以通过体积渲染方程聚合：$\mathbf{U}(\mathbf{r}) = \sum_{n=1}^{N_s} T_i(1-\exp(-\sigma_n\delta_n))\text{tr}(\mathbf{G}_n)$，其中 $\mathbf{G}_n$ 是第n个采样点相关参数的Hessian子矩阵。Fisher信息低的区域对应不确定性高的区域——即模型对该区域掌握的信息少。

### 损失函数 / 训练策略

- 辐射场训练使用标准的L2渲染损失（MSE between rendered and ground truth images）
- 视角选择在每次训练迭代的间隔进行（如每100个epoch选一次）
- 3DGS训练遵循原始配置，每次添加新视角后重置opacity以避免退化
- 初始用4个均匀分布的视角训练，然后逐步添加至20个视角
- 主动建图系统基于SplaTAM框架，使用frontier-based exploration生成候选路径

## 实验关键数据

### 主实验

| 数据集 | 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|------|-------|-------|--------|
| Blender (20 views) | 3DGS + Random | 28.73 | 0.939 | 0.053 |
| Blender (20 views) | 3DGS + ActiveNeRF | 26.61 | 0.905 | 0.081 |
| Blender (20 views) | **3DGS + Ours** | **29.53** | **0.944** | **0.043** |
| Blender (10 views) | 3DGS + Random | 22.49 | 0.873 | 0.112 |
| Blender (10 views) | **3DGS + Ours** | **23.68** | **0.883** | **0.102** |
| Mip360 (seq) | 3DGS + Random | 17.91 | 0.564 | 0.430 |
| Mip360 (seq) | 3DGS + ActiveNeRF | 17.89 | 0.533 | 0.414 |
| Mip360 (seq) | **3DGS + Ours** | **20.35** | **0.601** | **0.361** |

### 主动建图结果

| 方法 | Gibson Comp.(%)↑ | Gibson Comp.(cm)↓ | MP3D Comp.(%)↑ | MP3D Comp.(cm)↓ |
|------|------------------|-------------------|----------------|-----------------|
| FBE | 68.91 | 14.42 | 71.18 | 9.78 |
| Active Neural Mapping | 80.45 | 7.44 | 73.15 | 9.11 |
| **Ours** | **92.89** | **5.64** | **89.41** | **2.91** |

### 不确定性量化

| 方法 | AUSE (LF Dataset, avg)↓ |
|------|------------------------|
| CF-NeRF | 0.38 |
| ActiveNeRF | 0.33 |
| BayesRays | 0.23 |
| **Ours** | **0.22** |

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| Random选择 | 基线PSNR | Fisher信息带来显著提升 |
| ActiveNeRF (学习方差) | 低于Random | 间接不确定性近似不可靠 |
| BayesRays (扰动场) | 略优于Random | 不确定性≠信息增益 |
| FisherRF (EIG) | 最优 | 直接优化信息增益的优势明显 |
| Plenoxels后端 | 同样有效 | 框架对不同辐射场模型通用 |

### 关键发现

- **EIG vs 不确定性**: 使用期望信息增益（EIG）选择视角明显优于使用不确定性（uncertainty），因为单纯选择最不确定的视角并不保证对模型参数的整体信息增益最大
- **ActiveNeRF劣于Random**: 在3DGS后端上，ActiveNeRF甚至弱于随机选择，说明修改模型架构来估计不确定性在新表征上不可靠
- **批量选择有效**: 贪心批量选择算法有效避免了信息冗余
- **极端稀疏场景**: 在仅10个训练视角时，FisherRF的优势更加显著
- **主动建图大幅领先**: 场景覆盖率从80.45%提升至92.89%（Gibson），从73.15%提升至89.41%（MP3D）

## 亮点与洞察

1. **理论优雅性**: 从信息论第一原理出发推导视角选择目标，Fisher信息提供了明确的数学框架；Hessian不依赖真实图像的性质使得"选择前不需要拍摄"在理论上清晰自洽
2. **稀疏性洞察**: 辐射场的局部参数结构导致Hessian高度稀疏这一发现，使得对2亿+参数的Fisher信息计算变得可行；自定义CUDA核实现了100倍加速
3. **多任务框架统一**: 视角选择、批量选择、路径规划和像素级不确定性量化都从同一个Fisher信息框架自然导出
4. **强实验结果**: 在三个不同任务、五个数据集上全面超越SOTA，且幅度显著

## 局限与展望

1. 仅适用于静态场景，动态辐射场（如D-NeRF、4D Gaussian Splatting）中的Fisher信息量化仍是开放问题
2. 对角Laplace近似丢弃了参数间的相关性信息，更精确的块对角或低秩近似可能进一步提升性能
3. 主动建图依赖frontier-based exploration提供候选路径，更高级的路径规划策略或强化学习方法可能带来额外收益
4. 像素级不确定性是相对的（基于观测信息），不是绝对的度量，在需要标定不确定性的应用中有限制
5. 当训练视角极少（<5个）时，初始模型质量可能太差导致Fisher信息估计不准确

## 相关工作与启发

- **ActiveNeRF (Pan et al., ECCV 2022)**: 在NeRF中添加方差输出实现主动视角选择，SDFC的直接竞争对手
- **BayesRays (Goli et al., 2023)**: 通过假设扰动场量化NeRF不确定性，不适用于3DGS
- **3D Gaussian Splatting (Kerbl et al., 2023)**: FisherRF在此基础上实现，利用其显式参数化特性高效计算Fisher信息
- **SplaTAM (Keetha et al., 2023)**: 基于3D Gaussian的SLAM系统，FisherRF在此基础上构建主动建图系统
- **Kirsch & Gal, 2022**: 从Fisher信息角度统一深度主动学习中的各种方法，为FisherRF提供了理论灵感
- 启发：Fisher信息作为参数层面的信息度量，可能在其他需要主动数据获取的视觉任务（如主动SLAM、机器人探索）中发挥更大作用

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首次将Fisher信息直接应用于辐射场的参数级信息量化，理论框架优雅完整
- **实验充分度**: ⭐⭐⭐⭐⭐ 三个任务、五个数据集、多种辐射场后端、详尽的消融和对比
- **写作质量**: ⭐⭐⭐⭐ 理论推导清晰，从动机到方法到实验逻辑完整
- **价值**: ⭐⭐⭐⭐⭐ 为辐射场的主动学习提供了理论基础和高效实现，多任务框架具有广泛应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](../../ICCV2025/others/switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[ECCV 2024\] Active Generation for Image Classification](active_generation_for_image_classification.md)
- [\[ECCV 2024\] Mahalanobis Distance-Based Multi-View Optimal Transport for Multi-View Crowd Localization](mahalanobis_distance-based_multi-view_optimal_transport_for_multi-view_crowd_loc.md)
- [\[ICML 2025\] Fishers for Free? Approximating the Fisher Information Matrix by Recycling the Squared Gradient Accumulator](../../ICML2025/others/fishers_for_free_approximating_the_fisher_information_matrix_by_recycling_the_sq.md)
- [\[CVPR 2025\] Which Viewpoint Shows it Best? Language for Weakly Supervising View Selection in Multi-view Instructional Videos](../../CVPR2025/others/which_viewpoint_shows_it_best_language_for_weakly_supervising_view_selection_in_.md)

</div>

<!-- RELATED:END -->
