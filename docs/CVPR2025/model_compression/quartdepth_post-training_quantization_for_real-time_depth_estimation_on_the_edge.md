---
title: >-
  [论文解读] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge
description: >-
  [CVPR 2025][模型压缩][训练后量化] 提出 QuartDepth，一个面向ASIC边缘设备的训练后量化框架，通过LogNP激活磨光（将异常分布的激活值变换为量化友好的分布）、激活量化补偿（更新权重补偿激活量化误差）和Fisher信息引导的权重重建，将深度估计基础模型量化到W4A4/W4A8，并设计可编程硬件加速器实现实时推理。
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "训练后量化"
  - "深度估计"
  - "边缘部署"
  - "ASIC加速器"
  - "4-bit量化"
---

# QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge

**会议**: CVPR 2025  
**arXiv**: [2503.16709](https://arxiv.org/abs/2503.16709)  
**代码**: [GitHub](https://github.com/shawnricecake/quart-depth)  
**领域**: 3D视觉  
**关键词**: 训练后量化, 深度估计, 边缘部署, ASIC加速器, 4-bit量化

## 一句话总结

提出 QuartDepth，一个面向ASIC边缘设备的训练后量化框架，通过LogNP激活磨光（将异常分布的激活值变换为量化友好的分布）、激活量化补偿（更新权重补偿激活量化误差）和Fisher信息引导的权重重建，将深度估计基础模型量化到W4A4/W4A8，并设计可编程硬件加速器实现实时推理。

## 研究背景与动机

- 基础深度估计模型（Metric3D, DepthAnything等）性能优异但计算量巨大，难以部署在边缘设备上
- ASIC是边缘部署的理想平台，但需要低位宽量化（4-bit）以充分利用硬件带宽
- 大规模基础MDE模型的数据集庞大，全量再训练不现实，需要训练后量化（PTQ）
- 深度估计模型解码器中存在严重的异常激活分布：通道间离群值差异大、分布偏离正态
- 逐张量（per-tensor）量化因离群值变异性无法有效处理，逐通道量化中离群值仍导致显著量化误差
- 现有PTQ方法主要针对分类/语言模型，未专门解决深度估计模型的分布特性
- 矩阵乘法和卷积运算占推理时间的绝大部分（~90%+），量化这些操作是关键
- 缺乏针对量化深度估计模型的专用硬件加速器设计

## 方法详解

### 整体框架

QuartDepth 流水线分为三步：(1) 先用 LogNP 磨光变换激活值分布，然后进行激活量化；(2) 更新权重以补偿激活量化引入的误差；(3) 使用Fisher信息引导的AdaRound对更新后的权重进行量化。同时设计了支持kernel fusion和自定义指令可编程性的柔性硬件加速器，包含W4A4/W4A8的专用计算核和可编程向量计算阵列。

### 关键设计

**设计一：LogNP 激活磨光（Activation Polishing）**
- **功能**：将深度估计解码器中的异常激活分布变换为量化友好的正态分布
- **核心思路**：对每个通道 $i$ 的激活值 $x$，施加对数变换 $\Phi(x, \alpha) = \text{sign}(x) \cdot [\log_2(|x| + \alpha) - \log_2(\alpha)]$，其中磨光因子 $\alpha_i = P_\epsilon(\mathbf{x}_i)$ 由第95百分位确定。量化后通过逆变换 $\Phi^{-1}$ 恢复。LogNP有效压缩离群值同时保持主体分布的区分度
- **设计动机**：直接量化含离群值的激活会损失大量信息；对数变换天然地压缩大值、扩展小值，使分布更集中对称

**设计二：激活量化误差补偿**
- **功能**：通过更新权重来最小化激活量化引入的输出误差
- **核心思路**：对每层求解 $\min_{\Delta\mathbf{W}} \|\mathbf{W}\mathbf{x} - (\mathbf{W} + \Delta\mathbf{W})\hat{\mathbf{x}}\|_2^2$，闭式解为 $\Delta\mathbf{W}^* = -\mathbf{W}(\mathbf{x} - \hat{\mathbf{x}})\hat{\mathbf{x}}^T(\hat{\mathbf{x}}\hat{\mathbf{x}}^T)^{-1}$。当 $\hat{\mathbf{x}}\hat{\mathbf{x}}^T$ 不满秩时使用dampening技术
- **设计动机**：激活量化和权重量化分开处理，先补偿激活误差再量化权重，将两步误差累积降到最低

**设计三：Fisher信息引导的权重重建**
- **功能**：使用二阶信息最小化权重量化导致的损失退化
- **核心思路**：将量化误差对损失的影响用Taylor展开近似为 $\frac{1}{2}\Delta\mathbf{w}^T\mathbf{H}_\mathbf{w}\Delta\mathbf{w}$，用KFAC近似逐层Fisher矩阵 $\mathbf{F}_l = \mathbf{G}_l \otimes \mathbf{A}_l$。以此作为AdaRound的优化目标，学习取整参数 $\mathbf{v}$ 使 $\sum_l (\mathbf{w}^{(l)} - \hat{\mathbf{w}}^{(l)})^T\mathbf{F}_l(\mathbf{w}^{(l)} - \hat{\mathbf{w}}^{(l)}) + \lambda h(\mathbf{v})$ 最小化
- **设计动机**：传统round-to-nearest忽略了不同权重对损失的敏感度差异；Fisher信息矩阵提供了比Hessian更实际可计算的二阶近似

### 损失函数/优化目标

激活补偿：$\|\mathbf{W}\mathbf{x} - (\mathbf{W} + \Delta\mathbf{W})\hat{\mathbf{x}}\|_2^2$（层级闭式解）；权重重建：Fisher加权的AdaRound目标 + 正则化项，通过梯度优化学习取整方向。

## 实验关键数据

### 主实验：KITTI/NYUv2深度估计量化比较

| 模型 | 量化配置 | NYUv2 $\delta_1$↑ | NYUv2 AbsRel↓ | KITTI $\delta_1$↑ | KITTI AbsRel↓ |
|------|---------|-------------------|--------------|-------------------|---------------|
| Metric3D (ViT-L) FP32 | W32A32 | 0.977 | 0.064 | 0.975 | 0.052 |
| Metric3D (ViT-L) | W8A8 | 0.975 | 0.065 | 0.974 | 0.053 |
| Metric3D (ViT-L) | W4A8 | 0.970 | 0.069 | 0.970 | 0.056 |
| Metric3D (ViT-L) | **W4A4** | **0.960** | **0.076** | **0.963** | **0.061** |

### 消融实验：各组件贡献（Metric3D ViT-L, W4A4, NYUv2 $\delta_1$↑）

| 方法 | $\delta_1$↑ | AbsRel↓ |
|------|------------|---------|
| Baseline (直接量化) | 0.891 | 0.118 |
| + LogNP polishing | 0.938 | 0.088 |
| + 激活补偿 | 0.949 | 0.081 |
| + Fisher权重重建 | **0.960** | **0.076** |

### 关键发现
- W4A8配置下$\delta_1$仅损失0.7%（0.977→0.970），W4A4损失1.7%
- LogNP磨光是最关键组件，将直接量化的0.891提升至0.938（+4.7%）
- Embodied Road Depth精度不依赖分割模型选择（与GT分割差距<1%）
- ASIC硬件在ViT-L模型上实现30+FPS实时推理
- 与仅使用MSE目标的AdaRound相比，Fisher引导的权重重建提供更准确的量化

## 亮点与洞察

1. **LogNP变换的直觉性**：对数变换天然适合处理长尾分布，百分位自适应磨光因子的设计简洁有效
2. **逐步解耦的量化流水线**：先磨光→再补偿激活→最后量化权重，每步都有清晰的数学推导
3. **软硬件协同设计**：LogNP磨光的计算开销被可编程向量计算阵列的并行执行完全隐藏
4. **通用性**：适用于多种ViT-based深度估计模型（Metric3D, DepthAnything等）

## 局限与展望

- W4A4在部分场景下仍有~2%的$\delta_1$损失，对精度敏感的应用可能需要W4A8
- 当前仅针对线性层和卷积层量化，注意力中的softmax等非线性操作保持浮点
- ASIC设计为特定模型定制，通用性受限
- 未来可探索混合精度量化策略或结合知识蒸馏进一步减小精度损失

## 相关工作与启发

- 与SmoothQuant迁移离群值到权重不同，LogNP直接变换激活分布
- Fisher引导的权重重建结合了OBQ/GPTQ的思路，但避免了逐列求解的高开销
- 激活补偿的闭式解设计可推广到其他模型的PTQ流程

## 评分

⭐⭐⭐⭐ — 系统化的量化框架，从问题分析到解决方案到硬件设计形成完整闭环；LogNP磨光是有价值的技术贡献。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[CVPR 2025\] Towards Practical Real-Time Neural Video Compression](towards_practical_real-time_neural_video_compression.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](../../ICML2025/model_compression/boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)

</div>

<!-- RELATED:END -->
