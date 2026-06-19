---
title: >-
  [论文解读] Click-Gaussian: Interactive Segmentation to Any 3D Gaussians
description: >-
  [ECCV 2024][3D视觉][3D高斯分裂] 提出Click-Gaussian，通过学习两级粒度（粗/细）的可区分3D特征场，结合全局特征引导学习(GFL)解决跨视角mask不一致问题，实现每次点击仅需10ms的实时3D高斯交互式分割，速度比现有方法快15-130倍，同时显著提升分割精度。 3D高斯分裂(3DGS)因实…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "3D高斯分裂"
  - "交互式分割"
  - "特征场"
  - "对比学习"
  - "视角一致性"
---

# Click-Gaussian: Interactive Segmentation to Any 3D Gaussians

**会议**: ECCV 2024  
**arXiv**: [2407.11793](https://arxiv.org/abs/2407.11793)  
**代码**: [https://seokhunchoi.github.io/Click-Gaussian](https://seokhunchoi.github.io/Click-Gaussian) (项目页面)  
**领域**: 3D视觉  
**关键词**: 3D高斯分裂, 交互式分割, 特征场, 对比学习, 视角一致性

## 一句话总结

提出Click-Gaussian，通过学习两级粒度（粗/细）的可区分3D特征场，结合全局特征引导学习(GFL)解决跨视角mask不一致问题，实现每次点击仅需10ms的实时3D高斯交互式分割，速度比现有方法快15-130倍，同时显著提升分割精度。

## 研究背景与动机

3D高斯分裂(3DGS)因实时渲染能力而在3D场景操作中备受关注，对场景中物体的精确分割是编辑、虚拟现实等应用的基础需求。现有基于3DGS的分割方法面临两个核心痛点：

**后处理耗时**：如SAGA等方法需要对噪声分割输出进行大量后处理才能获得清晰的分割结果，严重制约了3DGS实时渲染的效率优势

**细粒度分割困难**：现有方法难以提供精细的分割结果，无法满足精细化3D场景操作需求

**跨视角mask不一致**：从不同视角独立获得的2D分割结果之间存在冲突，导致3D特征学习受阻

核心矛盾：如何在不依赖耗时后处理的前提下，学习到在3D空间中可区分且视角一致的特征场？

本文切入角度：设计两级粒度特征场表示 + 全局特征引导学习策略，从根本上消除对后处理的依赖。

## 方法详解

### 整体框架

Click-Gaussian在预训练的3DGS基础上，为每个3D高斯增加D维特征向量 $\mathbf{f}_i \in \mathbb{R}^D$，分为粗级特征和细级特征。利用SAM自动生成所有训练视角的2D mask，构建两级mask，通过对比学习训练特征场，并引入GFL确保跨视角一致性。训练完成后，用户单击即可在10ms内完成分割。

### 关键设计

1. **两级粒度特征场（Two-level Granularity Feature Fields）**：

    - 将每个高斯的特征 $\mathbf{f}_i$ 分为粗级 $\mathbf{f}_i^c \in \mathbb{R}^{D^c}$ 和补充部分 $\bar{\mathbf{f}}_i^c \in \mathbb{R}^{D-D^c}$
    - 粗级特征直接使用 $\mathbf{f}_i^c$，细级特征通过拼接得到 $\mathbf{f}_i^f = \mathbf{f}_i^c \oplus \bar{\mathbf{f}}_i^c$
    - **粒度先验（Granularity Prior）**：细级特征包含粗级特征，体现了现实世界中的层级依赖关系——若A和B在粗级不同，则 $a \subset A$ 和 $b \subset B$ 在细级自然也不同
    - 实验中设置 $D^c=12$，$D=24$，通过3DGS光栅化器在单次前向传播中计算两级特征

2. **对比学习（Contrastive Learning）**：

    - 正对比损失：对同一mask区域的像素对，最大化其渲染特征的余弦相似度
    $\mathcal{L}_{\text{pos}}^{\text{cont}} = -\frac{1}{|P_1||P_2|}\sum_l \sum_{p_1}\sum_{p_2} \mathbb{1}[M_{p_1}^l = M_{p_2}^l] \mathbf{S}^l(p_1, p_2)$
    - 负对比损失：对不同mask区域的像素对，约束余弦相似度不超过阈值 $\tau^l$（$\tau^f=0.75$, $\tau^c=0.5$）
    - 对细级负对比损失使用stop gradient操作，避免粗级特征在细级区分时被错误更新

3. **全局特征引导学习（Global Feature-guided Learning, GFL）**：

    - **全局特征候选生成**：在指定迭代次数后，对所有训练视角渲染2D特征图，对每个mask区域做平均池化，得到平均特征集合 $\mathcal{F}^l$
    - **HDBSCAN聚类**：对 $\mathcal{F}^l$ 执行HDBSCAN聚类，得到 $C^l$ 个全局特征候选 $\tilde{\mathcal{F}}^l$，这些聚类中心是整个场景最具代表性的特征，有效平滑了跨视角噪声
    - **GFL损失**：引导每个高斯特征接近其最可能的全局聚类，同时远离其他聚类
    $\mathcal{L}_{\text{pos}}^{\text{GFL}} = -\frac{1}{N}\sum_l\sum_i \mathbb{1}[\tilde{\mathbf{S}}^l(i,c_i^l) > \tau^g] \tilde{\mathbf{S}}^l(i,c_i^l)$
    - 聚类定期更新，$\tau^g=0.9$

### 损失函数 / 训练策略

总损失函数：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cont}} + \lambda_1 \mathcal{L}_{\text{GFL}} + \lambda_2 \mathcal{L}_{\text{3D-norm}} + \lambda_3 \mathcal{L}_{\text{2D-norm}} + \lambda_4 \mathcal{L}_{\text{spatial}}$$

正则化项包括：
- **超球正则化** $\mathcal{L}_{\text{3D-norm}}$：约束每个高斯特征的L2范数为1，防止特征范数过大主导渲染
- **渲染特征正则化** $\mathcal{L}_{\text{2D-norm}}$：确保单个像素对应的多个高斯特征方向一致
- **空间一致性正则化** $\mathcal{L}_{\text{spatial}}$：利用KD-tree查询空间最近邻，使邻近高斯特征一致

超参数：$\lambda_1=10.0$，$\lambda_2=0.2$，$\lambda_3=0.2$，$\lambda_4=0.5$。训练3000次迭代，GFL从第2000次开始，总耗时约13分钟（NVIDIA RTX A5000）。

## 实验关键数据

### 主实验

**LERF-Mask数据集（mIoU %）**：

| 方法 | Coarse平均 | Fine平均 | All平均 |
|------|-----------|---------|---------|
| Gau-Group | 72.8 | 31.5 | 49.9 |
| OmniSeg3D | 79.4 | 46.9 | 60.9 |
| Feature3DGS | 65.6 | 63.5 | 63.4 |
| GARField | 80.9 | 71.4 | 75.2 |
| **Click-Gaussian** | **89.1** | **84.3** | **85.4** |

**SPIn-NeRF数据集（mIoU %）**：

| 方法 | mIoU |
|------|------|
| MVSeg | 90.9 |
| SA3D | 92.4 |
| SAGA | 88.0 |
| **Click-Gaussian** | **94.0** |

### 消融实验

| 配置 | All mIoU | 相对变化 |
|------|---------|---------|
| w/o $\mathcal{L}_{\text{2D-norm}}$ | 83.2 | -2.6% |
| w/o $\mathcal{L}_{\text{3D-norm}}$ | 80.3 | -6.0% |
| w/o $\mathcal{L}_{\text{spatial}}$ | 82.0 | -4.0% |
| w/o $\mathcal{L}_{\text{GFL}}$ | 58.6 | **-31.3%** |
| w/o prior | 82.1 | -3.9% |
| **完整模型** | **85.4** | - |

### 关键发现

- GFL是最关键的组件，移除后性能下降31.3%，尤其在细级分割中从84.3降至42.3
- 粒度先验使细级分割提升6个百分点（78.3→84.3）
- 超球正则化对细级分割尤为重要（74.1 vs 84.3）
- 推理速度：每次点击~10ms，比GARField快130倍，比SAGA快15倍

## 亮点与洞察

1. **粒度先验设计精巧**：将细级特征设计为粗级特征的超集，巧妙地编码了现实世界中的层级依赖关系
2. **GFL从根本上解决了跨视角一致性问题**：通过全局聚类提供稳定的监督信号，而非依赖单视角的噪声mask
3. **实用性极强**：10ms级别的交互响应时间使得真正的实时3D编辑成为可能
4. **stop gradient的巧妙运用**：在细级负对比学习中对粗级分量使用stop gradient，避免了粗-细级特征间的梯度冲突

## 局限与展望

1. 依赖预训练3DGS的质量，若单个高斯表示多个语义不同但颜色相似的物体，特征学习会受阻
2. 只有两级粒度，缺少中间层级，对于需要多层级分割的复杂场景可能需要多次交互
3. 未探索语义理解，仅基于外观+空间的特征学习，无法做开放词汇的分割
4. 可考虑扩展到动态场景或4D高斯的分割

## 相关工作与启发

- 与SAGA的对比说明：后处理是瓶颈，端到端学习可区分特征是更好的方向
- GFL思路可推广到其他需要跨视角一致性的任务（如3D编辑、语义理解）
- 两级粒度先验启发：在多尺度学习中编码显式的层级依赖关系比独立学习更有效

## 评分

- 新颖性: ⭐⭐⭐⭐ (GFL和粒度先验设计新颖，整体框架中规中矩)
- 实验充分度: ⭐⭐⭐⭐⭐ (多数据集、多基线、完整消融、应用演示)
- 写作质量: ⭐⭐⭐⭐ (逻辑清晰，公式规范)
- 价值: ⭐⭐⭐⭐ (10ms交互分割有很高实用价值)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] iSegMan: Interactive Segment-and-Manipulate 3D Gaussians](../../CVPR2025/3d_vision/isegman_interactive_segment-and-manipulate_3d_gaussians.md)
- [\[ECCV 2024\] WaSt-3D: Wasserstein-2 Distance for Scene-to-Scene Stylization on 3D Gaussians](wast-3d_wasserstein-2_distance_for_scene-to-scene_stylization_on_3d_gaussians.md)
- [\[ECCV 2024\] Interactive 3D Object Detection with Prompts](interactive_3d_object_detection_with_prompts.md)
- [\[ECCV 2024\] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)
- [\[ECCV 2024\] BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting](bad-gaussians_bundle_adjusted_deblur_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
