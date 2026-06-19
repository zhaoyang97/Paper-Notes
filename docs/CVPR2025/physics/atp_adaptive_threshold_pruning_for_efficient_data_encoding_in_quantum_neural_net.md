---
title: >-
  [论文解读] ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks
description: >-
  [CVPR 2025][物理/科学计算][量子神经网络] 提出 ATP（Adaptive Threshold Pruning），在量子数据编码前自适应地剪除低信息量的数据特征，通过 L-BFGS-B 优化阈值，在 MNIST/FashionMNIST/CIFAR/PneumoniaMNIST 四个数据集的二分类任务上取得最高准确率的同时显著降低纠缠熵。
tags:
  - "CVPR 2025"
  - "物理/科学计算"
  - "量子神经网络"
  - "自适应剪枝"
  - "数据编码"
  - "纠缠熵"
  - "量子机器学习"
---

# ATP: Adaptive Threshold Pruning for Efficient Data Encoding in Quantum Neural Networks

**会议**: CVPR 2025  
**arXiv**: [2503.21815](https://arxiv.org/abs/2503.21815)  
**代码**: 无  
**领域**: 量子计算 / 物理  
**关键词**: 量子神经网络, 自适应剪枝, 数据编码, 纠缠熵, 量子机器学习

## 一句话总结

提出 ATP（Adaptive Threshold Pruning），在量子数据编码前自适应地剪除低信息量的数据特征，通过 L-BFGS-B 优化阈值，在 MNIST/FashionMNIST/CIFAR/PneumoniaMNIST 四个数据集的二分类任务上取得最高准确率的同时显著降低纠缠熵。

## 研究背景与动机

**领域现状**：量子神经网络（QNN）利用量子叠加和纠缠进行数据处理，但受限于量子比特资源不足、噪声和退相干等硬件约束。数据编码是 QML 的关键瓶颈——编码策略直接决定量子比特使用效率。

**现有痛点**：(1) 角度编码（Angle encoding）直接映射像素值到旋转角度，简单但冗余数据增加纠缠和电路复杂度；(2) 幅度编码（Amplitude encoding）紧凑但扩展性受限；(3) 过高的纠缠熵增加计算复杂度并导致贫瘠高原（barren plateau）问题。

**核心矛盾**：更高的纠缠可增强表达能力，但过度纠缠增加硬件开销并降低训练效率，二者需要平衡。

**本文目标**：在编码前减少数据冗余，以更少的量子资源实现更好的分类性能。

**切入角度**：图像中不同区域的信息密度差异很大，低方差区域对分类贡献有限，可以在编码前剪除。

**核心 idea**：基于自适应阈值将低信息量的数据区域置零，减少编码冗余，同时通过双层优化自动找到最佳阈值。

## 方法详解

### 整体框架

ATP 在数据编码之前引入预处理步骤：(1) 计算每个类别的平均像素强度矩阵；(2) 将两类平均值均低于阈值 $\tau$ 的位置置零；(3) 用 L-BFGS-B 优化阈值使测试准确率最大化；(4) 用剪枝后的数据进行角度编码并训练 QNN。

### 关键设计

1. **自适应阈值剪枝函数**:

    - 功能：移除对分类无贡献的低信息量数据区域
    - 核心思路：计算两个二分类类别的平均像素强度矩阵 $\bar{x}_0$ 和 $\bar{x}_1$，对于位置 $(i,j)$，若 $\bar{x}_0(i,j) < \tau$ 且 $\bar{x}_1(i,j) < \tau$，则将该位置值置零。这等效于只保留对区分两类有价值的高信息区域。
    - 设计动机：低方差区域的像素在两类间几乎不可区分，编码这些信息不仅浪费量子比特资源，还增加了不必要的纠缠。

2. **双层阈值优化（L-BFGS-B）**:

    - 功能：自动搜索最优阈值 $\tau^*$
    - 核心思路：外层优化目标是最大化测试集准确率 $\tau^* = \arg\max_\tau \text{Acc}_\text{test}(\mathcal{X}_\tau)$，内层对剪枝后数据训练 QNN。使用 L-BFGS-B 拟牛顿方法在约束 $[0, \tau_\text{max}]$ 内迭代优化，通过近似逆海森矩阵实现高效梯度下降。
    - 设计动机：手动阈值选择不适应不同数据集的分布特性，自动优化确保阈值适配数据。

3. **纠缠熵（EE）作为效率指标**:

    - 功能：衡量编码方案的量子资源使用效率
    - 核心思路：纠缠熵通过约化密度矩阵的冯诺依曼熵计算，量化量子比特间的关联程度。ATP 在降低 EE 的同时提升准确率，实现"用更少的纠缠获得更好的性能"。
    - 设计动机：低 EE 意味着更少的量子比特间交叉干扰，有利于在噪声硬件上的实际部署。

### 损失函数 / 训练策略

使用 COBYLA 优化器训练 3 层参数化量子电路（PQC），包含 XX 和 ZZ 纠缠门。阈值优化和 QNN 训练交替进行。

## 实验关键数据

### 主实验

| 类对 | Angle | Amplitude | ATP | PCA | SQE |
|------|-------|-----------|-----|-----|-----|
| MNIST(0,1) | 96.0 | 95.5 | **99.0** | 99.0 | 88.0 |
| CIFAR(0,1) | 70.0 | 68.5 | **74.2** | 68.0 | 66.0 |
| PneumoniaMNIST | 81.0 | 68.5 | **87.0** | 80.0 | 75.5 |

### 消融实验

| 编码方法 | 平均EE↓ | 平均准确率↑ |
|----------|---------|------------|
| Angle | 0.65 | 85.3 |
| Amplitude | 0.55 | 82.1 |
| PCA | 0.56 | 84.6 |
| SQE | 0.41 | 82.7 |
| **ATP** | **0.38** | **87.5** |

### 关键发现
- ATP 在几乎所有数据集和类对上同时实现最高准确率和最低纠缠熵
- 在去极化噪声（3-10%）下，ATP 和 SQE 表现最鲁棒，准确率下降仅 3-8 个点
- FGSM 对抗攻击后结合对抗训练，ATP 保持最高准确率
- 在 IBM Sherbrooke 真实量子硬件上，ATP 比直接编码平均提升 7%

## 亮点与洞察
- **数据层面而非电路层面的剪枝**：跳出"优化量子电路结构"的思路，直接从输入数据减少冗余，更简单直接
- **噪声鲁棒性**：低纠缠的编码方案天然对去极化噪声更鲁棒，因为减少了量子比特间的串扰放大
- **在真实硬件验证**：在 IBM 量子计算机上的实验增强了方法的实际意义

## 局限与展望
- 仅验证了二分类，多分类需要混合经典-量子方法
- ATP 的阈值优化增加约 15% 的计算开销
- 当前仅在小规模图像上测试，大规模高维数据的适用性待验证
- 未探索与更复杂编码方案（如混合幅度-角度编码）的组合

## 相关工作与启发
- **vs PCA**: PCA 通过线性变换降维，ATP 通过基于方差的自适应剪枝，对数据结构的适应性更强
- **vs SQE**: SQE 将所有特征编码到单量子比特以减少资源使用，ATP 保留多量子比特但剪除冗余特征
- 思路可迁移到经典深度学习中的输入特征选择和数据增强策略

## 评分
- 新颖性: ⭐⭐⭐⭐ 前量子编码剪枝的想法清晰，但方法相对简单
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个数据集、噪声/对抗攻击/真实硬件多角度验证
- 写作质量: ⭐⭐⭐⭐ 结构通顺，但量子背景概念篇幅较大
- 价值: ⭐⭐⭐⭐ 对量子计算社区有价值，但 CVPR 读者可能关注度有限

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DGNet: Discrete Green Networks for Data-Efficient Learning of Spatiotemporal PDEs](../../ICLR2026/physics/dgnet_discrete_green_networks_for_data-efficient_learning_of_spatiotemporal_pdes.md)
- [\[ICLR 2026\] AQER: A Scalable and Efficient Data Loader for Digital Quantum Computers](../../ICLR2026/physics/aqer_a_scalable_and_efficient_data_loader_for_digital_quantum_computers.md)
- [\[ICML 2025\] Compact Matrix Quantum Group Equivariant Neural Networks](../../ICML2025/physics/compact_matrix_quantum_group_equivariant_neural_networks.md)
- [\[ICCV 2025\] ResQ: A Novel Framework to Implement Residual Neural Networks on Analog Rydberg Atom Quantum Computers](../../ICCV2025/physics/resq_a_novel_framework_to_implement_residual_neural_networks_on_analog_rydberg_a.md)
- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](../../NeurIPS2025/physics/f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)

</div>

<!-- RELATED:END -->
