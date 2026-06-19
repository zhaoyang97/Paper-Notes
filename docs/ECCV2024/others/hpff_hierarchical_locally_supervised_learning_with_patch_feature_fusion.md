---
title: >-
  [论文解读] HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion
description: >-
  [ECCV 2024][局部学习] 提出 HPFF，通过层次化局部监督学习（HiLo，将网络划分为独立+级联两级局部模块）和 Patch 特征融合（PFF，将辅助网络的输入切块计算再平均）解决局部学习中的模块间信息缺失和 GPU 内存占用过高问题，在多个数据集上显著超越已有局部学习方法并接近甚至超越 BP。
tags:
  - "ECCV 2024"
  - "局部学习"
  - "梯度隔离"
  - "层次监督"
  - "Patch特征融合"
  - "生物可信性"
---

# HPFF: Hierarchical Locally Supervised Learning with Patch Feature Fusion

**会议**: ECCV 2024  
**arXiv**: [2407.05638](https://arxiv.org/abs/2407.05638)  
**代码**: [https://github.com/Zeudfish/HPFF](https://github.com/Zeudfish/HPFF)  
**领域**: 其他  
**关键词**: 局部学习, 梯度隔离, 层次监督, Patch特征融合, 生物可信性

## 一句话总结

提出 HPFF，通过层次化局部监督学习（HiLo，将网络划分为独立+级联两级局部模块）和 Patch 特征融合（PFF，将辅助网络的输入切块计算再平均）解决局部学习中的模块间信息缺失和 GPU 内存占用过高问题，在多个数据集上显著超越已有局部学习方法并接近甚至超越 BP。

## 研究背景与动机

传统深度学习依赖端到端反向传播（BP），面临两大问题：
- **生物不可信性**：BP 依赖全局误差信号逐层回传，与生物神经网络的学习机制不符
- **更新锁定问题**：隐藏层参数在前向和反向计算完成前无法更新，阻碍了高效的并行训练

局部学习（Local Learning）将网络切分为梯度隔离的局部模块，每个模块通过辅助网络独立优化，缓解了更新锁定问题并降低内存占用。但现有方法存在关键缺陷：
- **近视性（Shortsightedness）**：各模块独立优化局部目标，缺乏全局交互，易陷入局部最优
- **性能差距大**：当网络划分为大量局部模块时（每层一个模块），与 BP 的性能差距显著扩大
- **辅助网络内存开销**：辅助网络的设计本身占用大量 GPU 内存，削弱了局部学习的内存优势

核心问题：如何在保持梯度隔离的同时促进模块间信息交流，并降低辅助网络的内存消耗？

## 方法详解

### 整体框架

HPFF 由两个互补组件构成：
1. **HiLo（Hierarchical Locally Supervised Learning）**：将网络划分为两级局部模块——独立级和级联级，通过权重共享和多级监督实现模块间信息交流
2. **PFF（Patch Feature Fusion）**：将辅助网络的输入特征切分为 patch 分别计算后平均融合，降低 GPU 内存并增强对通用模式的捕捉

### 关键设计

1. **层次化局部模块（HiLo）**：网络被划分为 $K$ 个基础局部模块。每个模块同时属于两级结构：

    - **独立级（Independent Level, IL）**：每个模块 $f_{\theta_j}$ 配备独立辅助网络 $g_{\gamma_j}$，产生局部监督信号 $\hat{y_j} = g_{\gamma_j}(x_{j+1})$
    - **级联级（Cascade Level, CL）**：每 $k$ 个相邻模块组成一个级联模块，共享一个级联辅助网络 $h_{\beta_i}$。级联辅助网络接收级联模块最后一个子模块的输出：$\hat{y_i} = h_{\beta_i}(x_{i+k})$

   关键特性：相邻级联模块之间存在重叠（滑动窗口式），每个局部模块接收 $k+1$ 次监督（1 次来自独立级 + $k$ 次来自级联级）。参数更新规则：

    $\theta_j \leftarrow \theta_j - \eta_d \nabla_{\theta_j} \mathcal{L}(\hat{y_j}, y) - \sum_{n=i}^{i+k-1} \eta_c \nabla_{\theta_j} \mathcal{L}(\hat{y_n}, y)$

   实验中设 $k=2$，即每个级联模块包含两个基础模块。$k$ 过大会退化为 BP 并增加内存。

   **设计直觉**：独立级擅长学习局部特征（类内聚类紧），级联级擅长学习全局特征（类间分离好），两者互补。t-SNE 可视化清晰验证了这一点。

2. **Patch 特征融合（PFF）**：将局部模块输出特征 $x_{j+1}$ 切分为 $n \times n$ 个 patch，分别送入辅助网络后平均融合：

    $\hat{y_j} = \frac{\sum_{k=1}^{n} \sum_{l=1}^{n} g_{\gamma_j}(x_{j+1}^{(k,l)})}{n^2}$

   内存分析：原始方法辅助网络的内存为 $O(D + P + L \times D)$；PFF 下每次仅处理一个 patch，内存降为 $O(D/n^2 + P + L \times D/n^2)$。实验中 $n=2$，理论上辅助网络相关内存降至 $1/4$。

   **额外增益**：patch 级别的平均融合可让网络聚焦于多个 patch 中普遍存在的模式，学习更具泛化性的特征表示。特征可视化显示 PFF 后激活区域更多、更细粒度。

3. **即插即用兼容性**：HPFF 可直接叠加到现有局部学习方法（PredSim、DGL、InfoPro）上，不修改主网络结构。

### 损失函数 / 训练策略

- 每个局部模块使用交叉熵损失，来自独立辅助网络和级联辅助网络的多路监督加权聚合
- 最后一个模块直接连接全局池化层和全连接层输出分类结果
- CIFAR-10/SVHN：SGD + Nesterov，lr=0.8，batch=1024，400 epoch，cosine annealing
- STL-10：lr=0.1，batch=128
- ImageNet：VGG13 lr=0.025，ResNet-101/152 lr=0.05，90 epoch

## 实验关键数据

### 主实验

**CIFAR-10（Test Error↓）：**

| 方法 | ResNet-32 K=16 | ResNet-110 K=55 | 改善 |
|------|---------------|----------------|------|
| DGL | 14.08 | 14.45 | 基线 |
| DGL + HPFF | **8.94** | **8.74** | ↓5.14 / ↓5.71 |
| InfoPro | 12.93 | 13.22 | 基线 |
| InfoPro + HPFF | **8.99** | **8.96** | ↓3.94 / ↓4.26 |
| BP (端到端) | 6.37 | 5.42 | 传统上界 |

**ImageNet（Top-1 Error↓）：**

| 网络 | 方法 | Top-1 Error | Top-5 Error |
|------|------|-------------|-------------|
| ResNet-101 (K=4) | InfoPro | 22.81 | 6.54 |
| ResNet-101 (K=4) | InfoPro + HPFF | **21.14** (↓1.67) | **5.49** (↓1.05) |
| ResNet-152 (K=4) | BP | 21.60 | 5.92 |
| ResNet-152 (K=4) | InfoPro + HPFF | **20.99** (↓1.94) | **5.29** (↓1.42) |
| ResNeXt-101 32×8d (K=4) | BP | 20.64 | 5.40 |
| ResNeXt-101 32×8d (K=4) | InfoPro + HPFF | **19.94** (↓1.75) | **5.09** (↓1.02) |

在 ImageNet 上 HPFF 使局部学习**超越 BP** 成为可能（ResNet-152, ResNeXt-101）。

### 消融实验

**组件贡献（DGL + ResNet-32 K=16, CIFAR-10）：**

| IL | CL | PFF | Test Error | 改善 |
|----|----|----|------------|------|
| ✓ | ✗ | ✗ | 14.08 | 基线 |
| ✗ | ✓ | ✗ | 10.51 | ↓3.57 |
| ✓ | ✓ | ✗ | 9.44 | ↓4.64 |
| ✓ | ✓ | ✓ | **8.94** | ↓5.14 |

**GPU 内存对比：**

| 网络 | 方法 | GPU 内存(GB) | 相对 BP |
|------|------|-------------|---------|
| ResNet-110 K=55 | BP | 9.26 | — |
| ResNet-110 K=55 | DGL + HPFF | 2.44 | ↓73.7% |
| ResNet-110 K=55 | InfoPro + HPFF | 2.38 | ↓74.3% |
| ResNet-110 K=55 | PredSim + HPFF | 1.90 | ↓79.5% |

### 关键发现

- **独立级和级联级互补**：仅用 CL（10.51）已优于 IL（14.08），HiLo 结合两者（9.44）最优；t-SNE 证实 IL 学局部、CL 学全局
- **PFF 同时提升性能和降低内存**：添加 PFF 后 test error 从 9.44 降至 8.94，同时 GPU 内存从 3.13GB 降至 2.31GB
- **早期层精度降低是好事**：CKA 分析和分层线性分类器实验表明，HPFF 使早期层不过度优化局部目标，保留更多对全局有益的特征
- **划分模块越多，HPFF 提升越大**：K=55（每层一个模块）时提升最显著，因为模块间信息丢失最严重
- **通用性强**：在三种不同的局部学习方法、四种数据集、多种网络架构上均有显著提升

## 亮点与洞察

1. **独立级+级联级的层次设计非常优雅**：通过滑动窗口式的级联模块在保持梯度隔离的同时实现跨模块信息传递，不引入新的全局梯度
2. **PFF 一石二鸟**：既降低内存（处理小 patch）、又提升性能（多 patch 平均关注通用模式），设计简洁有效
3. **局部学习超越 BP 成为现实**：在 ImageNet 的 ResNet-152 和 ResNeXt-101 上，HPFF 使局部学习首次在性能上超过端到端 BP
4. **CKA 与分层分析提供深刻洞察**：早期层降低分类精度 → 保留更多全局特征 → 后期层精度显著提高，这一发现挑战了"每层都要分类准确"的直觉

## 局限与展望

- 仍依赖反向传播在局部模块内部计算梯度，未完全消除 BP
- 级联级带来的额外辅助网络增加了模型参数量和计算量（虽然 PFF 部分弥补了内存问题）
- $k=2$ 的设置是固定的，未探索自适应确定级联范围的方法
- 仅验证了图像分类任务，在 NLP、语音等其他模态上的效果未知
- PFF 的 $n=2$ 切分适用于特征图尺寸较大的情况，对于深层小特征图可能不适用

## 相关工作与启发

- **DGL**（Belilovsky et al., ICML 2019）：贪心逐层学习，HPFF 在其基础上改善最显著（error 从 14.08→8.94）
- **InfoPro**（Wang et al.）：通过互信息约束保留全局特征的局部学习方法
- **DNI**（Jaderberg et al., ICML 2017）：通过合成梯度解耦模块，但方向不同
- 层次化设计思路可启发其他需要平衡局部与全局的优化问题（如联邦学习中的客户端间信息共享）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 独立+级联的层次设计和 PFF 均为新颖贡献，且互补性强
- **实验充分度**: ⭐⭐⭐⭐⭐ — 跨三种方法、四个数据集、多种网络、消融+CKA+t-SNE+分层分析，实验极为充分
- **写作质量**: ⭐⭐⭐⭐ — 方法描述清晰，实验分析透彻，可视化有说服力
- **价值**: ⭐⭐⭐⭐ — 使局部学习首次在大规模数据集上超越 BP，对并行训练和节能计算有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Momentum Auxiliary Network for Supervised Local Learning](momentum_auxiliary_network_for_supervised_local_learning.md)
- [\[ECCV 2024\] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)
- [\[NeurIPS 2025\] Depth-Supervised Fusion Network for Seamless-Free Image Stitching](../../NeurIPS2025/others/depth-supervised_fusion_network_for_seamless-free_image_stitching.md)
- [\[ECCV 2024\] Align before Collaborate: Mitigating Feature Misalignment for Robust Multi-Agent Perception](align_before_collaborate_mitigating_feature_misalignment_for_robust_multi-agent_.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)

</div>

<!-- RELATED:END -->
