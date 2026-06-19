---
title: >-
  [论文解读] Function-Space Learning Rates
description: >-
  [ICML2025][模型压缩][function-space learning rate] 提出**逐层函数空间学习率**的高效蒙特卡洛估计方法，并基于此设计 **FLeRM**（Function-space Learning Rate Matching），在小模型上记录函数空间学习率，自动调整大模型的参数空间学习率，实现跨宽度、深度、初始化尺度和 LoRA rank 的超参数迁移。
tags:
  - "ICML2025"
  - "模型压缩"
  - "function-space learning rate"
  - "hyperparameter transfer"
  - "FLeRM"
  - "model scaling"
  - "LoRA"
---

# Function-Space Learning Rates

**会议**: ICML2025  
**arXiv**: [2502.17405](https://arxiv.org/abs/2502.17405)  
**代码**: [GitHub](https://github.com/edwardmilsom/function-space-learning-rates-paper)  
**领域**: 函数空间学习率 / 超参数迁移  
**关键词**: function-space learning rate, hyperparameter transfer, FLeRM, model scaling, LoRA

## 一句话总结

提出**逐层函数空间学习率**的高效蒙特卡洛估计方法，并基于此设计 **FLeRM**（Function-space Learning Rate Matching），在小模型上记录函数空间学习率，自动调整大模型的参数空间学习率，实现跨宽度、深度、初始化尺度和 LoRA rank 的超参数迁移。

## 研究背景与动机

### 领域现状

**领域现状**：神经网络训练的根本目的是学习一个从输入到输出的**函数**，但传统学习率衡量的是参数空间中的变化量，而非函数空间中的变化量。这引出一个核心问题：**能否有意义地量化和控制函数空间中的学习？**

大模型预训练成本高达数百万美元，在全尺度上做超参数搜索不切实际。现有超参数迁移方法（如 µP、Modula）通常需要：

### 现有痛点

**现有痛点**：严格的架构假设（如网络足够宽、接近随机初始化）

### 核心矛盾

**核心矛盾**：复杂的数学工具（Tensor Programs、动态均场理论）

### 解决思路

**解决思路**：使用特定库重写网络架构

这些限制使得现有方法在实际应用中灵活性不足。本文提出一种**经验性**方法，通过直接测量函数空间学习率来避免上述限制。

## 方法详解

### 核心概念：逐层函数空间学习率

第 $\ell$ 层参数 $\mathbf{W}^\ell$ 更新 $\Delta \mathbf{W}^\ell$ 引起网络输出 $f_{nk}$ 变化的一阶 Taylor 近似为：

$$\Delta_\ell f_{nk} = \sum_{ij} \Delta W_{ij}^\ell \frac{df_{nk}}{dW_{ij}^\ell}$$

函数空间学习率定义为输出变化的 RMS 范数：

$$\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^2 = \frac{1}{NK} \sum_{nk} (\Delta_\ell f_{nk})^2$$

直接计算需要 $NK$ 次反向传播，计算代价极高。

### 蒙特卡洛估计

构造标量随机投影 $\phi = \frac{1}{\sqrt{NK}} \sum_{nk} \omega_{nk} f_{nk}$（$\omega_{nk} \sim \mathcal{N}(0,1)$），其关于第 $\ell$ 层参数更新的变化量满足：

$$\Delta_\ell \phi \sim \mathcal{N}(0, \|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^2)$$

因此只需**一次额外反向传播**即可获得一个样本，通过多次采样估计方差即可得到函数空间学习率。

### Kronecker 分解降方差

定义 $Z_{ij} = \Delta W_{ij}^\ell \cdot \frac{d\phi}{dW_{ij}^\ell}$，假设 $Z_{ij}$ 的协方差具有 Kronecker 结构：

$$\text{Cov}[Z_{ij}, Z_{i'j'}] = U_{ii'} V_{jj'}$$

则函数空间学习率可由**3 个标量 EMA** 估计：

$$\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^2 = \frac{\mathbb{E}[\sum_k(\sum_i Z_{ik})^2] \cdot \mathbb{E}[\sum_k(\sum_j Z_{kj})^2]}{\mathbb{E}[\|\mathbf{Z}\|_{\mathcal{F}}^2]}$$

实际操作中，在训练开始时用 40 批数据预热 EMA，之后每 100 步更新一次，计算开销可忽略。

### FLeRM：函数空间学习率匹配

1. **记录阶段**：在小（便宜）的 base 模型上训练，记录各层函数空间学习率 $\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^{\text{base},t}$
2. **迁移阶段**：在大模型上，测量当前函数空间学习率 $\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^{(t)}$，自动设置逐层参数空间学习率：

$$\eta^\ell = \eta_0 \cdot \frac{\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^{\text{base},t}}{\|\Delta_\ell \mathbf{f}\|_{\text{RMS}}^{(t)}}$$

**深度扩展启发式**：若大模型层数多于 base 模型，将 base 模型的函数空间学习率平分给新增的残差块。

## 实验关键数据

### 实验设置


### 主实验

| 模型 | 架构 | 数据集 | 优化器 |
|------|------|--------|--------|
| ResMLP | 4 隐藏层残差 MLP | CIFAR-10 | Adam (恒定 LR) |
| Transformer (PostNorm/PreNorm/PreNormPostMod) | 2 层 decoder-only | Wikitext-103 子集 | Adam (恒定 LR) |
| GPT-2 / Llama-3.2-1B | LoRA 微调 | Cold French Law / Mathpile (~4M tokens) | AdamW |

### FLeRM 宽度迁移

- 标准训练下，宽度增加导致最优学习率**显著偏移**（与 µP 理论一致）
- 使用 FLeRM 后，最优学习率偏移被**完全消除或大幅减少**
- Transformer 在高宽度下，FLeRM 还能**改善训练损失**

### FLeRM 深度迁移

- ResMLP：标准训练有显著偏移，FLeRM 大幅拉近最优 LR
- PostNorm Transformer：标准训练深度增加→不稳定阈值左移→深模型损失更差；FLeRM 使不稳定点对齐，**深模型性能大幅提升**
- PreNorm：小偏移被 FLeRM 修正
- PreNormPostMod：标准训练已深度不变，FLeRM 不破坏该特性

### FLeRM LoRA rank 迁移

- 标准训练下，LoRA rank 从 2 增至 32，最优 LR 偏移**超过一个数量级**
- 使用 FLeRM（rank=2 为 base）后，偏移被**消除或大幅减少**，不稳定点对齐

### 函数空间学习率动态分析

- 在固定参数空间 LR 下，函数空间 LR 随训练**单调下降**（除输入嵌入层）→ 揭示隐式调度
- 不同类型层形成清晰的"频带"：第二个前馈权重矩阵(FF2)对输出影响最大（函数空间 LR 比 readout 层高一个数量级）

## 亮点与洞察

1. **架构无关性**：方法适用于任意 PyTorch 网络，无需重写架构或依赖特定库
2. **计算开销极低**：仅需少量额外反向传播（训练开始 40 次 + 每 100 步 1 次）
3. **统一框架**：用同一套方法同时解决宽度、深度、初始化尺度、LoRA rank 四个维度的超参数迁移
4. **分析工具价值**：函数空间视角揭示了 Adam 优化器的隐式学习率调度——不同层的函数空间贡献随训练动态变化
5. **反直觉发现**：Transformer 中第二个前馈层（而非 readout 层）对输出函数影响最大

## 局限与展望

1. **深度扩展启发式较粗糙**：当大模型层数多于 base 模型时，简单平分函数空间学习率不一定最优；消融实验表明更复杂的匹配方案可能进一步提升性能
2. **一阶 Taylor 近似**：函数空间变化量基于线性近似，大学习率下可能不准确
3. **Kronecker 分解假设**：协方差的 Kronecker 结构是近似，可能引入偏差
4. **实验规模有限**：最大模型约 814M 参数，未验证在更大规模（如数十亿参数）的效果
5. **仅验证了 Adam/AdamW**：未探索 SGD、LAMB 等其他优化器

## 相关工作与启发

- **µP (Yang & Hu, 2022)**：解析推导学习率缩放规则，需无穷宽假设；FLeRM 通过经验测量避免此限制
- **Modula (Large et al., 2024)**：基于 Lipschitz 常数的度量化方法，需设置每个参数的"质量"且需重写架构；FLeRM 自动从 base 模型记录
- **Chizat & Netrapalli (2024)**：用特征更新与反向传播角度量化特征学习；FLeRM 直接用自动微分测量
- **Everett et al. (2024)**：实证发现真实模型中的对齐在训练中高度动态，说明 µP 和均场假设的选择很困难

## 评分

- 新颖性: ⭐⭐⭐⭐ — 函数空间学习率的经验估计和跨尺度迁移是全新视角
- 实验充分度: ⭐⭐⭐⭐ — 覆盖宽度/深度/初始化/LoRA四个维度，含多种架构和消融
- 写作质量: ⭐⭐⭐⭐⭐ — 数学推导清晰，从动机到算法逻辑自然
- 价值: ⭐⭐⭐⭐ — 对大模型训练的超参数调优有直接实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] LoRA Subtraction for Drift-Resistant Space in Exemplar-Free Continual Learning](../../CVPR2025/model_compression/lora_subtraction_for_drift-resistant_space_in_exemplar-free_continual_learning.md)
- [\[CVPR 2025\] Tripartite Weight-Space Ensemble for Few-Shot Class-Incremental Learning](../../CVPR2025/model_compression/tripartite_weight-space_ensemble_for_few-shot_class-incremental_learning.md)
- [\[CVPR 2025\] Dataset Distillation with Neural Characteristic Function: A Minmax Perspective](../../CVPR2025/model_compression/dataset_distillation_with_neural_characteristic_function_a_minmax_perspective.md)
- [\[ICML 2025\] Parameter-Efficient Fine-Tuning of State Space Models](parameter-efficient_fine-tuning_of_state_space_models.md)
- [\[ICLR 2026\] Topology and Geometry of the Learning Space of ReLU Networks: Connectivity and Size](../../ICLR2026/model_compression/topology_and_geometry_of_the_learning_space_of_relu_networks_connectivity_and_si.md)

</div>

<!-- RELATED:END -->
