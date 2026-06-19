---
title: >-
  [论文解读] Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes
description: >-
  [ECCV 2024][自动驾驶][异常分割] 提出 Random Walk on Pixel Manifolds (RWPM)，利用随机游走捕获像素嵌入的流形结构来修正因驾驶场景多样性导致的流形畸变，从而提升异常分割评分函数的准确性，无需额外训练即可即插即用地集成到现有异常分割框架中。 异常分割旨在检测驾驶场景中语义分割模…
tags:
  - "ECCV 2024"
  - "自动驾驶"
  - "异常分割"
  - "随机游走"
  - "流形学习"
  - "像素嵌入"
---

# Random Walk on Pixel Manifolds for Anomaly Segmentation of Complex Driving Scenes

**会议**: ECCV 2024  
**arXiv**: 2404.17961 (https://arxiv.org/abs/2404.17961)  
**代码**: [https://github.com/ZelongZeng/RWPM](https://github.com/ZelongZeng/RWPM)  
**领域**: 自动驾驶  
**关键词**: 异常分割, 随机游走, 流形学习, 像素嵌入, 自动驾驶

## 一句话总结

提出 Random Walk on Pixel Manifolds (RWPM)，利用随机游走捕获像素嵌入的流形结构来修正因驾驶场景多样性导致的流形畸变，从而提升异常分割评分函数的准确性，无需额外训练即可即插即用地集成到现有异常分割框架中。

## 研究背景与动机

异常分割旨在检测驾驶场景中语义分割模型未见过的 outlier 类别目标（如动物、掉落物等），对自动驾驶安全至关重要。当前主流方法采用 Outlier Exposure (OE) 策略，通过异常评分函数基于 inlier 类 logit 预测来推断异常分数。然而，这些方法忽视了一个关键问题：**在真实驾驶场景中，环境多样性（光照、路面材质等）会导致像素嵌入的流形产生畸变**。具体表现为：

1. 部分 inlier 像素偏离其所属类别的高 logit 区域 → 产生假阳性
2. 部分 outlier 像素靠近 inlier 类别的高 logit 区域 → 产生假阴性
3. 直接使用畸变的像素嵌入计算 logit 会严重影响异常分数的准确性

本文是**首个提出并解决像素流形畸变问题**的异常分割工作。

## 方法详解

### 整体框架

RWPM 是一个推理阶段的后处理模块，工作流程为：
1. 使用编码器-解码器提取像素嵌入图 $\boldsymbol{p} \in \mathbb{R}^{d \times H \times W}$
2. 将嵌入图分割为 $n^2$ 个子图（Partial Random Walk）
3. 在每个子图上构建图并执行随机游走，更新像素嵌入
4. 拼接修正后的子图，输入后续分类器和评分函数

### 关键设计

1. **图构建（Graph Construction）**：基于余弦相似度构建亲和矩阵 $\mathbf{W}$，其中 $\mathbf{W}_{ij} = \langle \hat{\boldsymbol{p}}_i^r, \hat{\boldsymbol{p}}_j^r \rangle$（$i \neq j$），自环为 0。通过 softmax 归一化实现局部约束：$\mathbf{S}_{ij} = \frac{\exp(\mathbf{W}_{ij}/\tau)}{\sum \exp(\mathbf{W}_{ij}/\tau)}$，温度参数 $\tau < 1.0$（设为 0.01）来抑制非近邻点的影响，替代传统 k-NN 搜索实现高效的局部约束。

2. **随机游走过程（Random Walk Process）**：与传统流形挖掘方法用随机游走传播标签或状态不同，本文创新性地用随机游走在流形上传播和更新像素嵌入。迭代公式为 $\mathbf{m}^{t+1} = \alpha \mathbf{S} \mathbf{m}^t + (1-\alpha) \mathbf{m}^0$，其中 $\alpha \in (0,1)$ 控制继续游走 vs 重启的概率。闭式解为 $\mathbf{m}^\infty = (1-\alpha)(\mathbf{I} - \alpha\mathbf{S})^{-1} \mathbf{m}^0$。核心思想：同一流形上的像素嵌入经游走后趋于一致，不同流形上的保持差异。

3. **Partial Random Walk 策略**：为应对驾驶场景大尺寸图像的计算挑战（如 512×1024 图像的 $\mathbf{S}$ 矩阵达 524288²），提出两个优化：

    - **嵌入图分割**：将 $\boldsymbol{p}$ 等分为 $n \times n$ 个子图，在每个子图独立构建子图进行随机游走
    - **有限迭代**：使用 $T$ 步迭代（$T=5\sim20$）而非闭式解，时间复杂度从 $O((HW/n^2)^3)$ 降至 $O(Td(HW/n^2)^2)$
    - **校准机制**：当 $n > 2$ 时，对相邻子图边缘的异常分数基准值进行校准

### 损失函数 / 训练策略

RWPM 不需要任何额外训练，仅在推理阶段使用。直接应用于已训练好的异常分割模型的预训练权重上，无需修改网络结构。关键超参数：
- 转移概率 $\alpha = 0.99$
- 温度 $\tau = 0.01$
- 迭代次数 $T = 20$（Road Anomaly）/ $T = 5$（其他数据集）
- 分割参数 $n = 4$（pixel-based）/ $n = 2$（mask-based）

## 实验关键数据

### 主实验

| 数据集 | 指标 | RbA | RbA+RWPM | 提升 |
|--------|------|-----|----------|------|
| Fishyscapes L&F | AP↑ | 70.81 | 71.16 | +0.35 |
| Fishyscapes L&F | FPR95↓ | 6.30 | 6.12 | -0.18 |
| Road Anomaly | AP↑ | 85.42 | 87.34 | +1.92 |
| Road Anomaly | FPR95↓ | 6.92 | 5.27 | -1.65 |
| SMIYC Anomaly Track | AP↑ | 90.86 | 92.00 | +1.14 |
| SMIYC Obstacle Track | AP↑ | 91.85 | 93.30 | +1.45 |
| 平均 | AP↑ | 84.73 | **86.00** | +1.27 |
| 平均 | FPR95↓ | 6.33 | **5.46** | -0.87 |

组件级指标（SMIYC Anomaly Track）：mean F1 从 46.80 提升至 **58.44**（+11.64）。

### 消融实验

| 配置 | AP↑ | FPR95↓ | FPS | GPU 内存 |
|------|-----|--------|-----|----------|
| 无 RWPM | 85.42 | 6.92 | 11.12 | 3.49GiB |
| n=1（无分割） | 87.90 | 5.17 | 0.32 | 74.96GiB |
| n=2 | 87.34 | 5.27 | 2.04 | 7.24GiB |
| n=4† | 87.17 | 5.32 | 4.35 | 3.49GiB |
| n=8† | 86.91 | 5.41 | 6.26 | 3.55GiB |

### 关键发现

- RWPM 在 pixel-based 和 mask-based 两种架构上均一致提升性能
- Partial Random Walk 将内存从 74.96GiB 降至 7.24GiB，速度提升 6 倍以上
- 有限迭代（$T=20$）性能可超越闭式解（$T=\infty$），同时效率大幅提升
- RWPM 甚至能提升 in-distribution 分割性能（Cityscapes mIoU: 82.25→82.43）

## 亮点与洞察

1. **问题定义新颖**：首次揭示驾驶场景多样性导致像素嵌入流形畸变的问题，为异常分割提供了新的优化视角
2. **方法优雅**：借鉴流形学习中的扩散过程思想，用随机游走传播嵌入而非标签/状态，无训练即插即用
3. **t-SNE 可视化**令人信服：清晰展示了 RWPM 前后 inlier/outlier 像素的聚类分离效果
4. 通过 softmax 温度参数替代 k-NN 搜索实现局部约束，巧妙且高效

## 局限与展望

1. 分割子图之间的边界像素可能缺少跨区域流形信息，校准机制仅是近似解决
2. 推理时间开销：以 RbA 为例，FPS 从 11.12 降至 2.04（n=2），实时性受限
3. 随机游走步数和温度参数需要针对不同数据集调整
4. 可以探索轻量化的流形修正方案，或将流形结构引入训练阶段

## 相关工作与启发

- 流形学习方法（扩散过程）在检索任务中已被广泛使用，本文将其推广至像素级密集预测任务
- 与 DiffusionDet 等扩散模型的思路不同，这里的扩散是在特征流形上的几何扩散
- 可将类似思想推广到其他密集预测任务（如深度估计、光流），特别是存在域偏移的场景

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次提出像素流形畸变问题，随机游走更新嵌入的思路新颖
- **实验充分度**: ⭐⭐⭐⭐⭐ — 多数据集、多架构、多指标，消融详尽，可视化有说服力
- **写作质量**: ⭐⭐⭐⭐ — 图 1 的 toy example 和 t-SNE 可视化很好地传达了核心思想
- **实用价值**: ⭐⭐⭐⭐ — 即插即用无需训练，但推理开销限制了实时应用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data](../../CVPR2026/autonomous_driving/climaood_improving_anomaly_segmentation_via_physically_realistic_synthetic_data.md)
- [\[CVPR 2026\] Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation](../../CVPR2026/autonomous_driving/learning_to_identify_out-of-distribution_objects_for_3d_lidar_anomaly_segmentati.md)
- [\[ECCV 2024\] Monocular Occupancy Prediction for Scalable Indoor Scenes](monocular_occupancy_prediction_for_scalable_indoor_scenes.md)
- [\[ECCV 2024\] TOD³Cap: Towards 3D Dense Captioning in Outdoor Scenes](tod3cap_towards_3d_dense_captioning_in_outdoor_scenes.md)
- [\[ECCV 2024\] Reliability in Semantic Segmentation: Can We Use Synthetic Data?](reliability_in_semantic_segmentation_can_we_use_synthetic_data.md)

</div>

<!-- RELATED:END -->
