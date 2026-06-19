---
title: >-
  [论文解读] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks
description: >-
  [AAAI 2026][脉冲神经网络] I2E 提出一个超高效的图像到事件流转换框架，通过模拟微扫视眼动并用高度并行化的卷积实现比先前方法快 300 倍以上的转换速度，首次支持 SNN 训练的在线数据增强，在 I2E-ImageNet 上达到 60.50% 的事件分类 SOTA，并通过合成数据预训练 + 真实数据微调的 sim-to-real 范式在 CIFAR10-DVS 上创下 92.5% 的历史最佳。
tags:
  - "AAAI 2026"
  - "脉冲神经网络"
  - "事件流生成"
  - "图像到事件转换"
  - "数据增强"
  - "仿真到现实"
---

# I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks

**会议**: AAAI 2026  
**arXiv**: [2511.08065](https://arxiv.org/abs/2511.08065)  
**代码**: [GitHub](https://github.com/Ruichen0424/I2E)  
**领域**: 神经形态计算 / 脉冲神经网络  
**关键词**: 脉冲神经网络, 事件流生成, 图像到事件转换, 数据增强, 仿真到现实

## 一句话总结

I2E 提出一个超高效的图像到事件流转换框架，通过模拟微扫视眼动并用高度并行化的卷积实现比先前方法快 300 倍以上的转换速度，首次支持 SNN 训练的在线数据增强，在 I2E-ImageNet 上达到 60.50% 的事件分类 SOTA，并通过合成数据预训练 + 真实数据微调的 sim-to-real 范式在 CIFAR10-DVS 上创下 92.5% 的历史最佳。

## 研究背景与动机

脉冲神经网络（SNN）是一种受大脑稀疏事件驱动计算启发的范式，在专用神经形态芯片（如 Loihi、TrueNorth）上可获得数量级的能效优势。SNN 的天然输入是异步事件流，通常由动态视觉传感器（DVS）捕获——DVS 逐像素报告亮度变化而非整帧采集。然而，对专用硬件的依赖造成了一个根本性的数据瓶颈：大规模事件数据集的获取既昂贵又耗时，现有基准规模有限且质量不一（如显示器闪烁伪影）。

这导致了一个顽固的性能差距：事件 ImageNet 分类的最佳准确率远低于 ANN 对应的 70%+，令 SNN 在复杂任务上的实用性存疑。为规避数据不足，常见做法是在每个时间步重复输入同一静态图像，但这引入了密集冗余计算，从根本上违背了事件驱动范式的能效优势。先前的算法转换方法（如 ES-ImageNet 使用的 ODG 算法）虽然避免了硬件采集的限制，但计算瓶颈严重——处理整个 ImageNet 需要超过 10 小时，无法用于在线数据增强。

I2E 的核心洞察是：通过模拟微扫视眼动（microsaccadic eye movements），将图像差分操作等价为极稀疏的 $3 \times 3$ 卷积，使转换速度提升至可在训练时在线执行的水平。这同时解决了数据稀缺的规模问题和训练方法论的增强问题。

## 方法详解

### 整体框架

I2E 将静态 RGB 图像转换为 8 时间步的二值事件流，分三个阶段：强度图生成 → 时空卷积事件生成 → 自适应事件发放。整个流水线设计为高度并行化的张量操作序列，天然适合 GPU 加速。

### 关键设计

1. **强度图生成（Stage 1）**:

    - 功能：将 RGB 图像转换为单通道强度图
    - 核心思路：取 HSV 颜色空间的 V（Value）通道，即 $V(x,y) = \max(I_R(x,y), I_G(x,y), I_B(x,y))$，以极低计算代价生成类似传感器光感受器的强度表示
    - 设计动机：DVS 响应的是对数亮度变化，V 通道是最简单有效的近似。消融实验证实 V 通道与标准灰度几乎无信息损失

2. **时空卷积事件生成（Stage 2）**:

    - 功能：从静态强度图中模拟微扫视运动产生的亮度变化
    - 核心思路：将图像沿 8 个方向各平移 1 像素并做差分。关键创新在于，每个方向的平移差分等价于一个极稀疏的 $3 \times 3$ 卷积核 $K_t$——核中仅两个位置非零（一个 +1、一个 -1）。8 个方向的差分通过一次分组卷积并行完成：$\Delta V_t = V * K_t$
    - 随机增强策略：每个方向有一组等价的平移向量，训练时随机选取以引入多样性，推理时固定
    - 设计动机：朴素实现需要 8 次图像平移和差分，内存密集且串行。卷积等价性使操作在 GPU 上极度高效，实现了比 ODG 快 300 倍的转换速度

3. **自适应事件发放（Stage 3）**:

    - 功能：将连续的亮度变化图转换为二值脉冲事件
    - 核心思路：像素 $(x,y)$ 在时间步 $t$ 发放 ON 事件当 $\Delta V > S_{th}$，发放 OFF 事件当 $-\Delta V > S_{th}$。阈值采用动态自适应机制：$S_{th} = S_{th_0} \cdot (\max(V) - \min(V))$，其中 $S_{th_0}$ 是唯一的全局超参数
    - 设计动机：固定全局阈值对不同亮度图像产生不一致的事件率。动态阈值根据每张图像的亮度动态范围自适应调整，确保跨数据集的事件稀疏度一致性。ImageNet 上 $S_{th_0} = 0.12$ 对应约 5% 的事件率

### 效率与信息论分析

- **速度**：GPU 上处理单张图像约 0.1 ms，比硬件采集快 30,000 倍，比 ODG 算法快 300 倍
- **能耗**：标准 ANN 首层卷积能耗约 543 μJ，I2E 编码自身仅 0.36 μJ，I2E-SNN 首层仅 28.68 μJ，总体能耗降低 18.9 倍
- **存储**：I2E-ImageNet 以布尔数组存储仅 47 GB，比 JPEG 压缩后的原始 ImageNet（146 GB）减少 67.8%
- **信息保留**：原始灰度图平均 Shannon 熵为 7.14，I2E 事件流仅 1.53（保留不到 22% 的熵），但性能下降有限，说明丢弃的主要是冗余信息（如均匀纹理和背景）

### 训练策略

- 使用 MS-ResNet + LIF 神经元架构，SpikingJelly 框架
- 交叉熵损失 + 标签平滑（$\epsilon = 0.1$）+ SGD
- 在线增强（Baseline-II）：对源图像进行随机裁剪等标准增强后再 I2E 转换，性能大幅优于仅用随机翻转的 Baseline-I

## 实验关键数据

### 主实验（I2E-ImageNet 事件分类）

| 数据集 | 架构 | 方法 | 准确率(%) |
|--------|------|------|----------|
| ES-ImageNet | ResNet18+LIF | baseline | 39.89 |
| ES-ImageNet | ResNet18+LIAF | pre-train | 52.25 |
| N-ImageNet | ResNet34 | EST | 48.93 |
| **I2E-ImageNet** | **ResNet18+LIF** | **Baseline-II** | **57.97** |
| **I2E-ImageNet** | **ResNet34+LIF** | **Baseline-II** | **60.50** |
| I2E-ImageNet | ResNet18+LIF | pre-train | 59.28 |

I2E-ImageNet 上 ResNet34 达到 60.50%，比先前事件 ImageNet 数据集的最佳结果（48.93%）高出超过 11 个百分点。

### CIFAR 数据集 + Sim-to-Real 迁移

| 数据集 | 架构 | 方法 | 准确率(%) |
|--------|------|------|----------|
| CIFAR10-DVS | ResNet18 | transfer-I（从 I2E-ImageNet） | 83.1 |
| **CIFAR10-DVS** | **ResNet18** | **transfer-II（从 I2E-CIFAR10）** | **92.5** |
| CIFAR10-DVS | SpikingResformer | transfer | 84.8 |
| I2E-CIFAR10 | ResNet18 | Baseline-II | 89.23 |
| I2E-CIFAR10 | ResNet18 | transfer-I | 90.86 |
| I2E-CIFAR100 | ResNet18 | Baseline-II | 60.68 |
| I2E-CIFAR100 | ResNet18 | transfer-I | 64.53 |

在真实 DVS 数据集 CIFAR10-DVS 上，通过 I2E 合成数据预训练 + 微调达到 92.5%，比先前 SOTA（84.8%）高出 7.7%，验证了 sim-to-real 范式的有效性。

### 消融实验

| 配置 | 准确率(%) | 说明 |
|------|----------|------|
| 固定阈值 + 无增强 | 47.22 | 最朴素转换 |
| + 动态阈值 | 48.30 | 稳定事件率 |
| + 随机向量选择 | 49.01 | 引入数据多样性 |
| + 标准图像增强（随机裁剪等） | 57.97 | 因实时性而解锁，带来最大收益 |

| 时间步顺序 | CIFAR10 | CIFAR100 | 说明 |
|-----------|---------|----------|------|
| $\gamma\alpha\beta$（高事件率优先） | **89.23** | **60.68** | 最佳序列 |
| $\alpha\beta\gamma$ | 87.96 | 56.10 | 最差序列 |
| $\gamma\beta\alpha$ | 88.60 | 60.12 | 次优 |

### 关键发现

- Baseline-I 到 Baseline-II 的巨大跳升（48.30% → 57.97%）证明在线数据增强是 I2E 最重要的附带价值
- 时间步顺序有显著影响：先呈现高事件率帧（对应更大运动向量）效果更好
- RGB→V 通道转换损失约 3.5%（65.68% → 62.21%），事件化再损失约 3%（62.21% → 59.28%），总转换损失可控
- 时间步数与精度/压缩率之间存在可调节的权衡：仅 2 个时间步仍能达到 51.97%（压缩率 91.95%）

## 亮点与洞察

- 将图像差分等价为稀疏卷积的巧妙工程设计是全文的关键贡献，使转换速度进入可在线使用的量级
- sim-to-real 迁移结果（92.5%）是极有说服力的实验：证明合成事件数据可以作为真实传感器数据的高保真代理
- 该工作本质上将丰富的静态图像数据集"桥接"到事件驱动领域，为 SNN 训练打开了海量数据的大门
- 信息论分析提供了有价值的洞察：事件流虽只保留不到 22% 的原始熵，但保留的恰好是分类任务所需的显著特征

## 局限与展望

- 当前仅验证了分类任务，尚未扩展到检测、分割等更复杂的视觉任务
- 微扫视模拟产生的事件流与真实 DVS 传感器的物理特性（如噪声模式、像素响应延迟）仍有差距
- $S_{th_0}$ 是手动设定的全局超参数，未探索自适应学习阈值的可能
- 8 个时间步的固定设计可能限制了对更复杂动态场景的建模能力

## 相关工作与启发

- ES-ImageNet（ODG 算法）是最接近的先前工作，I2E 在速度和质量上全面超越
- N-ImageNet 通过 DVS 拍摄显示器生成，采集慢且存在伪影；I2E 的纯算法路线完全规避了这些问题
- 该工作的 sim-to-real 范式可借鉴到自动驾驶等依赖事件相机的领域
- 卷积等价性的思路可推广到更复杂的运动模式模拟（如旋转、缩放）

## 评分

- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks](ds-atgo_dual-stage_synergistic_learning_via_forward_adaptive_threshold_and_backw.md)
- [\[AAAI 2026\] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling](tdsnns_competitive_topographic_deep_spiking_neural_networks_for_visual_cortex_mo.md)
- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[ICML 2026\] Bullet Trains: Parallelizing Training of Temporally Precise Spiking Neural Networks](../../ICML2026/others/bullet_trains_parallelizing_training_of_temporally_precise_spiking_neural_networ.md)
- [\[ICLR 2026\] Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks](../../ICLR2026/others/breaking_gradient_temporal_collinearity_for_robust_spiking_neural_networks.md)

</div>

<!-- RELATED:END -->
