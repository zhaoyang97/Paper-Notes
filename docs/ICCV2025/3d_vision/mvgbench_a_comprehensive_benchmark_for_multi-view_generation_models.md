---
title: >-
  [论文解读] MVGBench: a Comprehensive Benchmark for Multi-view Generation Models
description: >-
  [ICCV 2025][3D视觉][多视图生成] 提出 MVGBench——多视图生成模型的综合评估框架，创新性地引入基于 3DGS 自一致性的 3D 一致性指标（无需 3D GT），系统评估了 12 个 SOTA 方法在最佳性能、泛化和鲁棒性三方面的表现，并基于分析提出的最佳实践构建了新方法 ViFiGen。
tags:
  - "ICCV 2025"
  - "3D视觉"
  - "多视图生成"
  - "3D一致性"
  - "benchmark"
  - "3DGS"
  - "视觉语言模型评估"
---

# MVGBench: a Comprehensive Benchmark for Multi-view Generation Models

**会议**: ICCV 2025  
**arXiv**: [2507.00006](https://arxiv.org/abs/2507.00006)  
**代码**: [项目页面](https://virtualhumans.mpi-inf.mpg.de/MVGBench/)  
**领域**: 3D视觉  
**关键词**: 多视图生成, 3D一致性, benchmark, 3DGS, 视觉语言模型评估

## 一句话总结

提出 MVGBench——多视图生成模型的综合评估框架，创新性地引入基于 3DGS 自一致性的 3D 一致性指标（无需 3D GT），系统评估了 12 个 SOTA 方法在最佳性能、泛化和鲁棒性三方面的表现，并基于分析提出的最佳实践构建了新方法 ViFiGen。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：多视图生成 (MVG) 模型是当前 3D 内容创建的核心驱动力，但 **评估方法严重滞后**，存在三大问题：

**与 GT 比较不合理**: 生成模型采样自解空间分布，可能与 GT 不同但仍然正确；现有 PSNR/SSIM 逐视图独立评估忽略了 3D 一致性

**方法不可比**: 不同 MVG 使用不同的相机设置（焦距、距离、仰角），在各自的 GT 下评估产生不可比较的数值

**评估维度不足**: 未覆盖泛化到真实图像的能力、对输入扰动的鲁棒性等重要方面

**核心洞察**: 如果生成的多视图图像是 3D 一致的，那么从不相交子集分别重建的 3DGS 应该是相似的。基于此提出 **自一致性** 指标，无需 3D GT。

## 方法详解

### 整体框架

MVGBench 评估三个方面：
- **最佳设置性能**: 每个方法使用各自最优相机配置
- **泛化到真实图像**: 在人工标注的 CO3D、MVImgnet 数据集上评估
- **鲁棒性**: 对仰角、方位角、光照变化的敏感性

### 关键设计

1. **3D 一致性指标（自一致性）**:

    - 给定 MVG 生成的 $N$ 个视图，分为两个子集 $\mathcal{I}_1, \mathcal{I}_2$
    - 分别用 3DGS 拟合得到 $\mathcal{G}_1, \mathcal{G}_2$
    - **几何一致性**:
        - Chamfer Distance (CD): 从高斯重采样点云计算（利用协方差矩阵采样，非高斯中心），降采样至 60k 点
        - 深度误差 $e_d$: 同一视角渲染深度图，计算 L1 差异
    - **纹理一致性**: 同一视角渲染 RGB 图，计算 cPSNR、cSSIM、cLPIPS
    - **优势**: 可在真实图像上定量评估（无需 3D GT），对不同方法公平（各自用最优相机设置）

2. **公平对比的对齐方案**:

    - 合成数据: 标准化 3D 物体至单位立方体，用各方法各自的训练相机设置渲染输入，重建的 3DGS 已对齐
    - 真实数据: 用 ICP + 均匀缩放将不同方法的 3DGS 对齐到参考 3DGS
    - 对不同输出视图数的方法，允许子集间有少量重叠以对齐 3DGS 拟合精度上界

3. **语义和图像质量指标**:

    - **oFID**: 对象级 FID（非全数据集 FID），每个物体独立计算 FID 后取平均，与人类偏好更一致（Pearson 0.69）
    - **IQ-vlm**: 用预训练 VLM 评估图像质量（二值 yes/no），与人类评分强相关
    - **语义一致性**: VLM 评估类别、颜色、风格是否与输入一致

4. **评估数据集**:

    - 最佳性能: GSO (100物体), OmniObject3D (202物体)
    - 泛化: CO3D (102图像), MVImgnet (230图像)，人工选择正面视图并标注仰角
    - 鲁棒性: GSO30 在不同仰角/方位角/光照下渲染

### 损失函数 / 训练策略

ViFiGen（基于分析发现的最佳实践）：
- 采用视频扩散模型架构（视频先验提供更好的 3D 一致性-质量平衡）
- 用 ConvNextV2 替代 CLIP 编码输入图像（保留更多细节）
- 改进相机嵌入设计

## 实验关键数据

### 主实验

12 个方法在 CO3D（真实）和 GSO（合成）上的性能对比：

| 方法 | CO3D CD↓ | CO3D cPSNR↑ | CO3D IQ-vlm↑ | GSO CD↓ | GSO cPSNR↑ | GSO IQ-vlm↑ |
|------|----------|-------------|-------------|---------|------------|-------------|
| Zero123 | 12.06 | 13.16 | 0.38 | 10.99 | 17.37 | 0.73 |
| SyncDreamer | 3.04 | 25.30 | 0.12 | 2.99 | 26.83 | 0.53 |
| SV3D | 3.48 | 23.72 | 0.29 | 3.47 | 26.75 | 0.77 |
| Hi3D | 5.60 | 20.92 | 0.35 | 3.29 | 24.60 | 0.87 |
| **ViFiGen (Ours)** | **3.02** | **25.82** | 0.29 | **3.15** | **28.93** | 0.82 |

### 消融实验

指标鲁棒性验证（不同视图数/相机设置的 GT 一致性分数）：

| 视图数 | 相机设置 | CD↓ | cPSNR↑ | cSSIM↑ |
|--------|----------|-----|--------|--------|
| 16 | [17] | 1.993 | 30.281 | 0.924 |
| 18 | [9] | 2.119 | 30.688 | 0.934 |
| 20 | [37] | 2.133 | 30.448 | 0.925 |
| 20 | [54] | 2.091 | 30.800 | 0.932 |
| 相对标准差 | - | 0.026 | 0.006 | 0.004 |

**指标验证**: 相对标准差 <8%，说明指标对视图数和相机设置不敏感。

VLM 指标与人类偏好对齐（400 张图、10 用户）：Pearson 置信区间 0.95；oFID 与人类排名匹配度 0.77 vs 传统 FID 的 0.50。

### 关键发现

1. **3D 一致性与图像质量的 trade-off**: 现有方法中无一占据 "右上角"（两者都最佳），一致但缺细节 vs 有细节但不一致
2. **合成-真实差距显著**: 所有方法在真实图像上性能大幅下降，特别是 IQ-vlm
3. **视频模型更优**: SV3D、ViFiGen 等视频扩散模型在一致性和质量间取得更好平衡
4. **关键设计选择**:
    - 输入图像编码器: ConvNextV2 > CLIP（保留更多细节）
    - 相机嵌入: 显式相机参数编码优于隐式方法
    - 视频先验: 时空注意力提供自然的 3D 一致性约束
5. **鲁棒性普遍不足**: 大多数方法对仰角、方位角变化敏感

## 亮点与洞察

- **评估范式创新**: 自一致性指标是评估生成模型 3D 一致性的优雅方案，可推广到其他 3D 生成任务
- **系统分析价值**: 12 个方法、4 个数据集、3 个评估维度的大规模对比分析，识别出关键设计选择
- **从高斯采点而非用中心**: 利用协方差矩阵重采样计算 CD，比直接用高斯中心更准确

## 局限与展望

- 3DGS 拟合本身引入的误差可能混淆一致性评估（虽然实验显示影响 <8%）
- 仅评估物体级 MVG，未覆盖场景级生成
- VLM 评估受限于当前 VLM 的能力，可能随 VLM 更新而变化
- 未评估文本引导的 MVG 方法

## 相关工作与启发

- **Zero123/SV3D/SyncDreamer**: 代表性 MVG 方法，各有一致性-质量权衡
- **MEt3R**: 并发工作，关注 3D 场景生成的一致性，本文聚焦物体级
- **LGM**: 从多视图重建到 3DGS 的快速方法，本文用于渲染评估视图

## 评分

- 新颖性: ⭐⭐⭐⭐ (自一致性指标创新)
- 技术深度: ⭐⭐⭐⭐ (完整的评估框架设计)
- 实验充分度: ⭐⭐⭐⭐⭐ (12方法×4数据集×3维度)
- 实用价值: ⭐⭐⭐⭐⭐ (Benchmark 工具对社区价值极高)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] SpinMeRound: Consistent Multi-View Identity Generation Using Diffusion Models](spinmeround_consistent_multi-view_identity_generation_using_diffusion_models.md)
- [\[ICCV 2025\] FlexGen: Flexible Multi-View Generation from Text and Image Inputs](flexgen_flexible_multi-view_generation_from_text_and_image_inputs.md)
- [\[ICCV 2025\] Auto-Regressively Generating Multi-View Consistent Images](auto-regressively_generating_multi-view_consistent_images.md)
- [\[ICCV 2025\] MV-Adapter: Multi-view Consistent Image Generation Made Easy](mv-adapter_multi-view_consistent_image_generation_made_easy.md)
- [\[ICCV 2025\] MaterialMVP: Illumination-Invariant Material Generation via Multi-view PBR Diffusion](materialmvp_illumination-invariant_material_generation_via_multi-view_pbr_diffus.md)

</div>

<!-- RELATED:END -->
