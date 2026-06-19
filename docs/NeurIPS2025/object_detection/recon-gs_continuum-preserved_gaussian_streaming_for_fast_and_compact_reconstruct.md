---
title: >-
  [论文解读] ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction
description: >-
  [NeurIPS 2025][目标检测][3D Gaussian Splatting] 提出 ReCon-GS，通过连续性保持的 Gaussian 流式处理实现增量式 3D 重建，在保持渲染质量的同时大幅减少存储需求和训练时间，支持大规模场景的实时重建。 领域现状 领域现状：3D Gaussian Splatting（3DG…
tags:
  - "NeurIPS 2025"
  - "目标检测"
  - "3D Gaussian Splatting"
  - "流式重建"
  - "连续性保持"
  - "增量学习"
  - "实时渲染"
---

# ReCon-GS: Continuum-Preserved Gaussian Streaming for Fast and Compact Reconstruction

**会议**: NeurIPS 2025  
**arXiv**: [2509.24325](https://arxiv.org/abs/2509.24325)  
**代码**: 有  
**领域**: 目标检测 / 3D重建  
**关键词**: 3D Gaussian Splatting, 流式重建, 连续性保持, 增量学习, 实时渲染

## 一句话总结

提出 ReCon-GS，通过连续性保持的 Gaussian 流式处理实现增量式 3D 重建，在保持渲染质量的同时大幅减少存储需求和训练时间，支持大规模场景的实时重建。

## 研究背景与动机

### 领域现状

**领域现状**：3D Gaussian Splatting（3DGS）在静态场景重建中取得了革命性进展，但标准方法需要一次性处理所有输入图像。

**现有痛点**：大规模场景的一次性处理内存需求巨大；增量式方法面临灾难性遗忘（新数据覆盖旧区域的 Gaussian）和跨视图不一致性。

**核心矛盾**：增量效率 vs 全局一致性——逐帧处理快但丢失全局信息，全局处理准但不可扩展。

**切入角度**：保持 Gaussian 在新旧数据间的连续性，使增量更新不破坏已有重建。

## 方法详解

### 整体框架

输入图像流 → 局部 Gaussian 初始化 → 连续性保持的增量更新 → 全局 Gaussian 场景 → 实时渲染。

### 关键设计

1. **Gaussian 流式处理**

    - 功能：将输入图像流划分为窗口，每个窗口增量更新 Gaussian 集合
    - 核心思路：新窗口引入的 Gaussian 与已有 Gaussian 通过空间重叠区域建立关联
    - 设计动机：避免一次性处理所有图像的内存瓶颈

2. **连续性保持机制**

    - 功能：防止新数据的优化破坏已有 Gaussian 的属性
    - 核心思路：对重叠区域的 Gaussian 施加正则化约束，限制位置和属性的漂移幅度
    - 设计动机：解决增量学习中的灾难性遗忘问题

3. **自适应密度控制**

    - 功能：根据新视图的覆盖范围动态调整 Gaussian 密度
    - 核心思路：在新覆盖但尚未建模的区域增加 Gaussian，在冗余区域裁剪
    - 设计动机：保持模型紧凑性，避免 Gaussian 数量无限增长

### 损失函数 / 训练策略

$\mathcal{L} = \mathcal{L}_{photo} + \lambda_{reg}\mathcal{L}_{continuity} + \lambda_{ssim}\mathcal{L}_{SSIM}$

## 实验关键数据

### 主实验

| 方法 | PSNR↑ | SSIM↑ | 训练时间↓ | 存储↓ |
|------|-------|-------|---------|------|
| 3DGS (全局) | 33.14 | 0.969 | 30min | 100% |
| InstantNGP | 31.25 | 0.951 | 5min | 40% |
| StreamRF | 30.89 | 0.942 | 8min | 55% |
| **ReCon-GS** | **32.78** | **0.965** | **12min** | **45%** |

### 消融实验

| 配置 | PSNR | 说明 |
|------|------|------|
| 无连续性约束 | 30.45 | 严重遗忘 |
| 无密度控制 | 31.89 | Gaussian过多 |
| **完整模型** | **32.78** | **最优** |

### 关键发现

- 与全局 3DGS 仅差 0.36 PSNR，但训练时间和存储减少 50%+
- 连续性保持贡献 +2.33 PSNR，是核心模块
- 在大规模户外场景（Mega-NeRF 数据集）上优势更明显

## 亮点与洞察

- **连续性保持**：解决了增量 3DGS 的核心难题——灾难性遗忘。正则化约束简单有效。
- **实用性强**：支持在线数据采集和实时重建，适合机器人和 AR 场景。

## 局限与展望

- 重叠区域的检测依赖视角估计精度
- 动态场景未涉及
- 大规模场景的全局一致性仍有改进空间

## 相关工作与启发

- **vs 3DGS**：静态全局方法，无法增量；ReCon-GS 在接近质量下实现流式处理
- **vs StreamRF**：StreamRF 基于 NeRF，渲染速度受限；ReCon-GS 基于 Gaussian 实现实时渲染

## 评分
- 新颖性: ⭐⭐⭐⭐ 连续性保持的思路清晰实用
- 实验充分度: ⭐⭐⭐⭐ 多数据集验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐⭐ 实时重建的实际需求大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] ReCon: Region-Controllable Data Augmentation with Rectification and Alignment for Object Detection](recon_region-controllable_data_augmentation_with_rectification_and_alignment_for.md)
- [\[CVPR 2025\] DEIM: DETR with Improved Matching for Fast Convergence](../../CVPR2025/object_detection/deim_detr_with_improved_matching_for_fast_convergence.md)
- [\[CVPR 2025\] VCBench: A Streaming Counting Benchmark for Spatial-Temporal State Maintenance in Long Videos](../../CVPR2025/object_detection/vcbench_a_streaming_counting_benchmark_for_spatial-temporal_state_maintenance_in.md)
- [\[CVPR 2026\] GPFlow: Gaussian Prototype Probability Flow for Unsupervised Multi-Modal Anomaly Detection](../../CVPR2026/object_detection/gpflow_gaussian_prototype_probability_flow_for_unsupervised_multi-modal_anomaly_.md)
- [\[CVPR 2026\] WeDetect: Fast Open-Vocabulary Object Detection as Retrieval](../../CVPR2026/object_detection/wedetect_fast_open-vocabulary_object_detection_as_retrieval.md)

</div>

<!-- RELATED:END -->
