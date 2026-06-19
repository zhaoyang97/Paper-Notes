---
title: >-
  [论文解读] S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting
description: >-
  [CVPR 2025][3D视觉][3D高斯溅射] 提出 S2Gaussian 两阶段框架，首次解决稀疏+低分辨率视图联合场景重建问题：第一阶段用深度正则化优化低分辨率高斯并通过 Gaussian Shuffle Split 初始化高分辨率高斯，第二阶段用去模糊不一致性建模和 3D 鲁棒优化策略精炼高分辨率场景。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "3D高斯溅射"
  - "稀疏视图"
  - "超分辨率"
  - "新视角合成"
  - "深度正则化"
---

# S2Gaussian: Sparse-View Super-Resolution 3D Gaussian Splatting

**会议**: CVPR 2025  
**arXiv**: [2503.04314](https://arxiv.org/abs/2503.04314)  
**代码**: [项目页面](https://jeasco.github.io/S2Gaussian/)  
**领域**: 3D视觉/3D重建  
**关键词**: 3D高斯溅射, 稀疏视图, 超分辨率, 新视角合成, 深度正则化

## 一句话总结

提出 S2Gaussian 两阶段框架，首次解决稀疏+低分辨率视图联合场景重建问题：第一阶段用深度正则化优化低分辨率高斯并通过 Gaussian Shuffle Split 初始化高分辨率高斯，第二阶段用去模糊不一致性建模和 3D 鲁棒优化策略精炼高分辨率场景。

## 研究背景与动机

3DGS 在新视角合成中取得了显著进展，但依赖密集、高分辨率输入图像。实际应用中（机器人、互联网图像），可用视图常常**同时**面临稀疏和低分辨率两个问题。

现有方法的局限：
- 稀疏视图方法（FSGS, DNGaussian 等）假设输入是高分辨率的
- 超分辨率方法（SRGS, GaussianSR 等）假设有密集视图监督
- 两类方法存在固有不兼容性：超分辨率需要密集监督恢复细节，而稀疏视图正则化倾向于平滑化
- 简单组合两类方法效果不佳

核心挑战是如何在视角不足和清晰度不足的双重限制下，重建几何精确且细节丰富的 3D 场景。

## 方法详解

### 整体框架

两阶段设计：(1) HR GS 初始化阶段——先用深度正则化优化低分辨率高斯表示，再通过 Gaussian Shuffle Split 致密化初始化高分辨率高斯；(2) HR GS 优化阶段——用 2D 超分辨率模型增强原始视图和伪视图，配合去模糊不一致性建模和 3D 鲁棒优化进行精炼。

### 关键设计一：Gaussian Shuffle Split

- **功能**：将低分辨率粗高斯致密化为适合高分辨率细节表示的紧凑高斯
- **核心思路**：将每个原始高斯替换为沿三个主轴正负方向偏移的 6 个子高斯。偏移距离为 $\alpha \cdot s_i$（默认 $\alpha=0.5$），偏移轴方向的尺度缩小为原来的 1/4，其余轴缩小 $1/\lambda$（$\lambda=1.9$）。仅对不透明度 >0.5 的高斯执行分裂，分裂后重置不透明度以允许自动剪枝
- **设计动机**：低分辨率优化后的高斯椭球体稀疏且粗糙，无法模拟高分辨率细节。标准自适应密度控制在缺乏密集/高质量监督时失效。Shuffle Split 提供了免训练的局部致密化策略

### 关键设计二：去模糊不一致性建模（Blur-Free Inconsistency Modeling）

- **功能**：缓解 2D 超分辨率模型导致的多视图不一致同时保留细节
- **核心思路**：在预训练 SR 模型后接可学习的不一致性建模模块 IM（两个残差块）模拟视图间不一致 $I_{SR}^{IM} = I_{SR} + IM(I_{SR})$；但 IM 倾向于丢失细节以换取一致性，因此额外引入模糊提议模块 BP 预测逐像素模糊核 $\mathcal{B}_k$，用模糊后的渲染图 $R_{HR}^{blur} = R_{HR} * \mathcal{B}_k$ 与 $I_{SR}^{IM}$ 计算损失，避免直接约束 $R_{HR}$ 导致的平滑化
- **设计动机**：2D SR 模型无法保证多视图一致性，直接用 SR 结果监督会让高斯试图表示不一致性导致模糊。IM+BP 的组合让不一致性被显式建模而非由高斯承担

### 关键设计三：3D 鲁棒优化策略

- **功能**：缓解伪视图中未充分优化区域带来的错误梯度更新
- **核心思路**：观察到高质量视图产生稳定一致的梯度，而包含伪影的视图导致梯度混乱和波动。通过对高斯参数（如缩放 $s$）的梯度施加平滑/裁剪约束，抑制异常梯度的影响
- **设计动机**：伪视图中某些区域（如 LR 高斯未覆盖的区域）会产生伪影，直接参与优化会导致 3D 场景收敛到错误结构

### 损失函数

$\mathcal{L}_{SR} = (1-\beta)\mathcal{L}_1(R_{HR}^{blur}, I_{SR}^{IM}) + \beta \mathcal{L}_{D-SSIM}(R_{HR}^{blur}, I_{SR}^{IM})$，其中 $\beta=0.2$。第一阶段使用 Pearson 相关深度损失进行深度正则化。

## 实验关键数据

### 主实验：Blender ×4（8视图）

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 3DGS | 低 | 低 | 高 |
| FSGS + SR | 次优 | 次优 | 次优 |
| **S2Gaussian** | **最优** | **最优** | **最优** |

### LLFF ×4（3视图）- 极端稀疏场景

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|------|--------|--------|---------|
| 稀疏视图方法 | 较优几何 | 较优 | 中等 |
| SR 方法 | 中等 | 中等 | 中等 |
| **S2Gaussian** | **最优** | **最优** | **最优** |

### 消融实验

| 配置 | PSNR | SSIM |
|------|------|------|
| 无 Shuffle Split | 基线 | 基线 |
| + Shuffle Split | +显著提升 | +提升 |
| + IM (不一致性建模) | +进一步提升 | +提升 |
| + BP (模糊提议) | +细节改善 | +提升 |
| + 3D 鲁棒优化 | **最优** | **最优** |

### 关键发现

- S2Gaussian 在 Blender、LLFF、Mip-NeRF360 三个数据集上均达到 SOTA
- 在极端稀疏场景（LLFF 3 视图 ×4 超分）中优势最为明显
- Gaussian Shuffle Split 在不损坏原始 3D 表示的前提下有效致密化
- IM 和 BP 的组合比单独使用任一模块效果更好——IM 处理不一致性，BP 保留细节

## 亮点与洞察

1. **首次联合解决稀疏+低分辨率**：将两个此前独立研究的问题统一到一个框架中，更符合实际应用场景
2. **Shuffle Split 设计优雅**：免训练、局部致密化、不损坏原始表示
3. **IM+BP 的互补设计**：不一致性建模吸收视图差异，模糊提议补偿细节损失

## 局限与展望

- 依赖预训练 2D SR 模型的质量，不同 SR 模型会影响最终效果
- 两阶段流程增加了训练时间
- 对于极端稀疏（如 2 视图）场景的效果有待验证
- 未探索使用 3D 感知的超分辨率方法

## 相关工作与启发

- **FSGS, DNGaussian**：稀疏视图 3DGS 方法，使用深度先验
- **SRGS, GaussianSR**：3D 超分辨率方法
- **SuperGaussian**：利用视频上采样模型的 3D 超分辨率
- S2Gaussian 的两阶段思路（先粗后细）可推广到其他退化条件下的 3D 重建

## 评分

⭐⭐⭐⭐ — 问题定义切合实际，两阶段设计合理。Shuffle Split 和 IM+BP 各有独到之处。在多个 benchmark 上均取得 SOTA 验证了方法的有效性。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SplatSuRe: Selective Super-Resolution for Multi-view Consistent 3D Gaussian Splatting](../../CVPR2026/3d_vision/splatsure_selective_super-resolution_for_multi-view_consistent_3d_gaussian_splat.md)
- [\[CVPR 2025\] CoMapGS: Covisibility Map-based Gaussian Splatting for Sparse Novel View Synthesis](comapgs_covisibility_map-based_gaussian_splatting_for_sparse_novel_view_synthesi.md)
- [\[CVPR 2025\] SfM-Free 3D Gaussian Splatting via Hierarchical Training](sfm-free_3d_gaussian_splatting_via_hierarchical_training.md)
- [\[NeurIPS 2025\] Quantifying and Alleviating Co-Adaptation in Sparse-View 3D Gaussian Splatting](../../NeurIPS2025/3d_vision/quantifying_and_alleviating_co-adaptation_in_sparse-view_3d_gaussian_splatting.md)
- [\[CVPR 2025\] DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting](dropgaussian_structural_regularization_for_sparse-view_gaussian_splatting.md)

</div>

<!-- RELATED:END -->
