---
title: >-
  [论文解读] Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention
description: >-
  [ICCV 2025][图像恢复][ToF深度去噪] GIGA-ToF 提出了一种基于运动不变图结构融合的 ToF 深度去噪网络，通过跨帧图注意力机制和 MAP 问题的算法展开，同时增强了时序稳定性和空间锐度，并在合成和真实数据上展现了优秀的泛化能力。 连续波飞行时间（ToF）传感器因其实时响应和低功耗被广泛应用于机器人、3…
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "ToF深度去噪"
  - "图信号处理"
  - "几何注意力"
  - "算法展开"
  - "时序一致性"
---

# Consistent Time-of-Flight Depth Denoising via Graph-Informed Geometric Attention

**会议**: ICCV 2025  
**arXiv**: [2506.23542](https://arxiv.org/abs/2506.23542)  
**代码**: [github.com/davidweidawang/GIGA-ToF](https://github.com/davidweidawang/GIGA-ToF)  
**领域**: 深度图去噪 / 计算成像  
**关键词**: ToF深度去噪, 图信号处理, 几何注意力, 算法展开, 时序一致性

## 一句话总结

GIGA-ToF 提出了一种基于运动不变图结构融合的 ToF 深度去噪网络，通过跨帧图注意力机制和 MAP 问题的算法展开，同时增强了时序稳定性和空间锐度，并在合成和真实数据上展现了优秀的泛化能力。

## 研究背景与动机

连续波飞行时间（ToF）传感器因其实时响应和低功耗被广泛应用于机器人、3D 重建和增强现实等领域。但 ToF 深度图在远距离、低反射率和光滑表面等区域受到严重噪声污染。

现有去噪方法存在三个关键不足：

**单帧方法忽略时序相关性**：大多数 DNN 方法（如 ToFNet、GLRUN）仅处理单帧，无法利用帧间信息，导致时序抖动和去噪不充分

**多帧方法的空间模糊问题**：现有多帧方法（如 MTDNet、DVSR）通过估计场景流或帧间相关性来融合对应像素的深度特征。然而，由于相机运动，同一物体在不同帧中的深度值不同，直接融合深度特征会导致空间模糊和细节丢失

**真实数据泛化差**：纯数据驱动的 DNN 方案在合成数据上训练，由于难以获取真实数据的 ground truth，泛化到真实噪声时性能严重下降

本文的核心洞察：**虽然深度值随运动变化，但邻域像素间的图结构（即相关性模式）具有时序自相似性——它们编码的是物体的形状而非绝对深度。** 因此融合运动不变的图结构而非深度特征可以同时解决空间模糊和时序不一致问题。

## 方法详解

### 整体框架

GIGA-ToF 网络由三部分组成：
1. **特征提取网络**（蓝色）：编码器-解码器结构，提取多尺度几何特征 $\mathbf{F}^t$，估计初始先验权重和帧内图邻接矩阵
2. **图感知几何注意力（GIGA）模块**（黄色）：通过跨帧注意力学习图的边权重，实现运动不变的图结构融合
3. **展开 GLR 模块**（绿色）：将 MAP 优化问题的求解展开为可学习的迭代滤波

### 关键设计

1. **帧内图建模与跨帧图融合**：

    - 对每帧 ToF 原始数据构建 8 连接无向图 $\mathcal{G}^t$，邻接矩阵 $\mathbf{W}^t$ 编码相邻像素间的相关性
    - 构建帧间图 $\mathbf{W}^{t,t-1}$，将参考帧 $t-1$ 的图结构通过 2-hop 和 3-hop 路径映射到当前帧 $t$
    - 映射图的计算：$\hat{\mathbf{W}}^{t-1} = \mathbf{W}^{t,t-1}(\mathbf{W}^{t-1} + \mathbf{I})(\mathbf{W}^{t,t-1})^\top$
    - 通过置信度矩阵 $\boldsymbol{\Phi}^{t,t-1}$ 加权融合得到最终图：$\widetilde{\mathbf{W}}^t = \boldsymbol{\Phi}^{t,t-1}\hat{\mathbf{W}}^{t-1} + \mathbf{W}^t$
    - 设计动机：图结构反映物体形状而非绝对深度，因此具有运动不变性，避免了深度特征融合导致的空间模糊

2. **MAP 问题建模**：

    - **数据保真项**：基于 ToF 深度噪声分布推导似然函数，$\ln P(\mathbf{n}_d^t) \approx -\frac{1}{2\sigma^2}\|(\mathbf{X}_a^t)^{-1}(\mathbf{x}_q^t \odot \mathbf{y}_i^t - \mathbf{x}_i^t \odot \mathbf{y}_q^t)\|_2^2$
    - **图平滑先验**：在融合图上施加图拉普拉斯正则化（GLR），约束去噪结果在图上的平滑性
    - 设计动机：将 ToF 成像机制（噪声分布）和信号先验（图平滑性）嵌入网络设计中，增强可解释性和跨数据集泛化能力

3. **算法展开与图学习**：

    - 将 MAP 问题的交替优化求解展开为可微分的迭代滤波层
    - 每次迭代中，当前估计通过融合图的邻接矩阵进行卷积变换，并与输入加权融合
    - 图边权重通过 GIGA 注意力机制端到端学习：帧内图通过单层卷积估计，帧间图通过 Q-K 注意力计算
    - 设计动机：算法展开将滤波核 $\widetilde{\mathbf{W}}$ 显式参数化为可学习量，既保留了图谱滤波的低通可解释性，又通过 DNN 自适应学习最优参数

### 损失函数 / 训练策略

采用 L1 损失监督 in-phase 和 quadrature 两个分量的去噪结果：

$$L = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}} \sum_{\theta \in \{i,q\}} |\mathbf{x}_\theta^{t,*}(v) - \mathbf{x}_\theta^{t,\text{gt}}(v)|$$

训练配置：Adam 优化器（初始 lr=1e-3，在 epoch 15/30/45 以 0.7 衰减），训练 60 epoch，T=3 帧，R=2 次展开迭代。

## 实验关键数据

### 主实验 — DVToF 合成数据集去噪精度

| 方法 | 类型 | MAE (m) ↓ | AbsRel ↓ | δ₁ ↑ | TEPE (m) ↓ |
|------|------|----------|----------|------|-----------|
| libfreenect2 | 单帧/传统 | 0.1044 | 0.0283 | 0.9746 | 0.1023 |
| GLRUN | 单帧/DNN | 0.0357 | 0.0107 | 0.9929 | 0.0734 |
| WMF | 多帧/传统 | 0.0311 | 0.0116 | 0.9955 | 0.0751 |
| MTDNet | 多帧/DNN | 0.0566 | 0.0642 | 0.9816 | 0.1046 |
| DVSR | 多帧/DNN | 0.0718 | 0.0844 | 0.9777 | 0.1176 |
| **GIGA-ToF** | **多帧/DNN** | **0.0193** | **0.0060** | **0.9974** | **0.0637** |

GIGA-ToF 在 MAE 上超越第二名至少 **37.9%**，在 TEPE 上超越至少 **13.2%**。

### 消融实验 — 各模块贡献

| GLR | 融合方式 | 注意力 | MAE (m) ↓ | TEPE (m) ↓ |
|-----|---------|--------|----------|-----------|
| ✗ | ✗ | ✗ | 0.0409 | 0.0793 |
| 展开 | ✗ | ✗ | 0.0357 | 0.0734 |
| 展开 | 特征融合 | ✗ | 0.0238 | 0.0718 |
| 展开 | 特征融合 | ✓ | 0.0214 | 0.0713 |
| 展开 | **图融合** | ✗ | 0.0219 | 0.0702 |
| 展开 | **图融合** | **✓** | **0.0193** | **0.0637** |

每个组件均有明确贡献：展开 GLR 保留细节、图结构融合优于特征融合并解决空间模糊、注意力机制确保准确的帧间对应。

### 关键发现

1. **图结构融合 > 特征融合**：图融合在所有指标上均优于特征融合，定性结果也显示更锐利的边缘和更少的模糊
2. **强跨数据集泛化**：在合成 DVToF 上训练的模型直接应用于真实 Kinect v2 数据，仍能产生准确且平滑的深度图，而纯数据驱动的 MTDNet 完全失败
3. **帧间步长鲁棒性**：即使在 Δt=8 的大时间步长下，多帧处理仍优于单帧，验证了图结构的时序自相似性

## 亮点与洞察

- "图结构是运动不变的"这一洞察极为巧妙——将关注点从像素值转移到拓扑关系，从根本上解决了多帧深度融合的空间模糊问题
- 将 ToF 物理成像模型（噪声分布）与图信号处理先验结合起来通过算法展开构建网络，兼具性能和可解释性
- 计算效率适中（0.027s/帧），远优于传统 WMF（24.3s/帧），具有实际部署价值

## 局限与展望

- 目前仅考虑前一帧作为参考帧，更早帧的信息未被充分利用
- 帧间图的邻域大小固定为 q=7，可能不适用于所有运动幅度
- 仅在合成数据上训练和评估精度，真实数据仅有定性评估

## 相关工作与启发

- 与 GLRUN 的关系：继承了其算法展开框架，但从单帧扩展到多帧并引入了核心的图融合机制
- 对其他视频去噪任务的启发：图结构的运动不变性这一观察可能同样适用于其他传感器（如事件相机、雷达）
- 图谱滤波的可解释性为理解网络行为提供了理论保证

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ （运动不变图融合的核心洞察非常精妙）
- 实验充分度: ⭐⭐⭐⭐ （合成+真实数据，充分消融）
- 写作质量: ⭐⭐⭐⭐⭐ （数学推导严谨，图示清晰）
- 价值: ⭐⭐⭐⭐ （对 ToF 深度成像和图信号处理社区有良好贡献）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Exploiting Dual-Correlation for Multi-frame Time-of-Flight Denoising](../../ECCV2024/image_restoration/exploiting_dual-correlation_for_multi-frame_time-of-flight_denoising.md)
- [\[ICCV 2025\] Emulating Self-Attention with Convolution for Efficient Image Super-Resolution](emulating_self-attention_with_convolution_for_efficient_image_super-resolution.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)
- [\[CVPR 2026\] Dual Graph Regularized Deep Unfolding Network for Guided Depth Map Super-resolution](../../CVPR2026/image_restoration/dual_graph_regularized_deep_unfolding_network_for_guided_depth_map_super-resolut.md)
- [\[ICCV 2025\] Lightweight and Fast Real-time Image Enhancement via Decomposition of the Spatial-aware Lookup Tables](lightweight_and_fast_real-time_image_enhancement_via_decomposition_of_the_spatia.md)

</div>

<!-- RELATED:END -->
