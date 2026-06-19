---
title: >-
  [论文解读] Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration
description: >-
  [ICCV 2025][模型压缩][数据剪枝] 提出 Partial Forward Blocking (PFB)，在前向传播的浅层阶段计算样本重要性并剪枝，阻断被剪枝样本的后续深层前向传播，实现 ImageNet 上 40% 剪枝下 0.5% 精度提升 + 33% 训练时间缩减。 大规模数据集提升了模型泛化能力但带来了巨大…
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "数据剪枝"
  - "训练加速"
  - "概率密度"
  - "核密度估计"
  - "前向阻断"
---

# Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration

**会议**: ICCV 2025  
**arXiv**: [2506.23674](https://arxiv.org/abs/2506.23674)  
**代码**: 无  
**领域**: 训练加速 / 数据剪枝  
**关键词**: 数据剪枝, 训练加速, 概率密度, 核密度估计, 前向阻断

## 一句话总结

提出 Partial Forward Blocking (PFB)，在前向传播的浅层阶段计算样本重要性并剪枝，阻断被剪枝样本的后续深层前向传播，实现 ImageNet 上 40% 剪枝下 0.5% 精度提升 + 33% 训练时间缩减。

## 研究背景与动机

大规模数据集提升了模型泛化能力但带来了巨大计算开销。现有数据剪枝方法的核心问题：
- **梯度方法**（EL2N, GraNd 等）：需要完整前向+反向传播来计算梯度，额外开销大
- **代理模型方法**（SVP, YOCO 等）：需训练额外代理模型，开销与训练目标模型相当
- **现有动态方法**（InfoBatch, DivBS 等）：虽不需梯度/代理模型，但仍需对所有样本完成完整前向传播

关键问题：能否在更低成本下高效筛选训练样本，同时保持模型泛化性能？

## 方法详解

### 整体框架

PFB 将训练网络分为**浅层子网络** $Net^{sh}$ 和**深层子网络** $Net^{dp}$。每个训练迭代：
1. 所有样本仅通过浅层子网络提取特征
2. 基于特征计算重要性分数
3. 剪枝低重要性样本，**阻断**其深层前向传播
4. 保留样本**复用**浅层特征继续深层前向和反向传播

### 关键设计

1. **Partial Forward Blocking 策略**: 核心创新在于将剪枝时机提前到前向传播早期阶段。被剪枝样本仅经过浅层网络（如 stage-1），深层计算完全跳过。相比之下：

    - 梯度方法需完整前向+额外反向传播
    - 代理方法需完整的代理模型前向
    - InfoBatch 等需完整前向
   
   PFB 的计算成本 = 保留样本的完整前向 + 被剪枝样本的浅层前向，显著低于所有现有方法。

2. **Probability Density Importance (概率密度重要性)**: 使用样本在特征空间中的概率密度作为冗余度衡量：

    - 高概率密度 = 特征空间中密集区域 = 冗余度高 = 重要性低
    - 低概率密度 = 稀疏区域 = 罕见样本 = 重要性高
   
   重要性定义为：
    $\mathcal{I}(z_i^t) = \frac{1}{f_\mathbf{X}(\mathbf{x}_i^t) + r}$
   
   其中 $r = \alpha \cdot \max_{z_i \in B} f_\mathbf{X}(\mathbf{x}_i^t)$，$\alpha \sim U(0, 0.01)$ 引入随机性进一步保证多样性。

3. **Adaptive Distribution Estimation (ADE，自适应分布估计)**: 使用核密度估计 (KDE) 高效估计概率密度：

    - 对浅层特征做空间平均池化 → 通道降维 → 得到紧凑表示 $\mathbf{x}_i^t \in \mathbb{R}^{1 \times D}$
    - 维护一组聚类中心 $C^t = \{\mathbf{c}_j^t\}$，用标准多元正态核函数计算 KDE：
    $\hat{f}_\mathbf{X}(\mathbf{x}_i^t) = \sum_{j=1}^{N_C} \frac{w_j^t}{N_C} K_\mathbf{H}(\mathbf{x}_i^t - \mathbf{c}_j^t)$
    - 使用 Silverman's rule 设置带宽矩阵 $\mathbf{H}$
    - 仅用**保留样本**更新中心（EMA 方式，$\beta=0.01$）
    - 权重 $w_j^{t+1} = n_j^t / (t \cdot (1-p) N_B)$ 平衡不同核的贡献

### 损失函数 / 训练策略

- 标准分类交叉熵损失，仅在保留样本 $S^t$ 上计算
- 剪枝比例 $p$ 为超参数（推荐 30%-50%）
- 聚类中心数量 $N_C$ 设为较小值（如 100），保证 KDE 高效
- 浅层网络选取 stage-1 输出作为特征提取点

## 实验关键数据

### 主实验 (表格)

**ImageNet-1k 上 ResNet-50 结果：**

| Method | 30% Pruned | 40% Pruned | 50% Pruned |
|--------|-----------|-----------|-----------|
| Full Data | 76.4 | 76.4 | 76.4 |
| Random | 72.2 (↓4.2) | - | 69.1 (↓7.3) |
| Forgetting | 74.8 (↓1.6) | - | 72.0 (↓4.4) |
| InfoBatch | 76.5 (↑0.1) | - | 75.8 (↓0.6) |
| MoSo | 76.5 (↑0.1) | - | 73.5 (↓2.9) |
| **PFB (Ours)** | **77.0 (↑0.6)** | - | **76.1 (↓0.3)** |

**ImageNet-1k 上 Swin-T 结果：**

| Method | 30% Pruned | 40% Pruned | 50% Pruned |
|--------|-----------|-----------|-----------|
| Full Data | 79.6 | 79.6 | 79.6 |
| Dyn-Unc | 79.1 (↓0.5) | 78.5 (↓1.1) | 77.6 (↓2.0) |
| InfoBatch | 78.6 (↓1.0) | 78.2 (↓1.4) | 77.5 (↓2.1) |
| **PFB (Ours)** | **79.6 (±0.0)** | **79.2 (↓0.4)** | **78.2 (↓1.4)** |

### 消融实验 (表格)

**CIFAR-10/100 上 ResNet-18 与其他方法对比：**

| Method | CIFAR-10 30%/50%/70% | CIFAR-100 30%/50%/70% |
|--------|---------------------|----------------------|
| Random | 94.6/93.3/90.2 | 73.8/72.1/69.7 |
| InfoBatch | 95.6/95.1/94.7 | 78.2/78.1/76.5 |
| DivBS | 95.4/95.2/95.1 | 78.5/78.2/77.2 |
| **PFB** | **95.9/95.5/95.2** | **79.1/78.8/77.9** |
| Full Data | 95.6 | 78.2 |

**训练时间对比 (ImageNet, ResNet-50, 40% pruning):**

| Method | Top-1 Acc | Training(h) | Overhead(h) | Total(n*h) | Reduction |
|--------|----------|------------|------------|-----------|-----------|
| Full Data | 76.4 | 13.9 | - | 55.6 | - |
| InfoBatch | 76.5 | 10.1 | 0.07 | 40.7 | 26.8% ↓ |
| DivBS | 76.4 | 11.2 | 0.72 | 47.6 | 14.4% ↓ |
| **PFB** | **76.9** | **9.2** | **0.06** | **37.1** | **33.2% ↓** |

### 关键发现

- **无损甚至正增益**: PFB 在 30% 剪枝率下多次超越全数据训练（ImageNet R50: +0.6%, CIFAR-100: +0.9%）
- **最大时间节省**: 40% 剪枝对应 33.2% 训练时间减少，且额外开销仅 0.06h（远低于 DivBS 的 0.72h）
- **CNN 和 Transformer 通用**: 在 ResNet-50 和 Swin-T 上均表现优异
- **概率密度优于损失/梯度**: 密度低的稀有样本比高损失样本更能保障泛化性

## 亮点与洞察

- **范式创新**: 将剪枝时机从"前向传播后"提前到"前向传播中"，根本性地改变了计算开销结构
- **概率密度视角独特**: 从分布维度度量样本冗余度，比梯度/损失更本质地反映样本信息价值
- **ADE 模块设计巧妙**: 通过聚类中心+EMA更新+加权KDE，以极低开销实现对训练分布的自适应追踪
- **正增益现象**: 适度剪枝反而提升性能，暗示全数据中的冗余样本可能干扰学习

## 局限与展望

- 浅层网络的选取（stage-1 vs stage-2）对不同架构可能有不同最优点
- 概率密度方法在极高维特征空间中可能面临"维度灾难"，需要降维
- 仅在分类和分割任务上验证，目标检测等密集预测任务待探索
- 聚类中心数量 $N_C$ 是需要调参的超参数
- 在自监督学习/大语言模型预训练中的效果待验证

## 相关工作与启发

- 与 InfoBatch 的对比最有说明力：InfoBatch 用损失值剪枝但仍需完整前向，PFB 从根本上减少前向计算
- 概率密度重要性可作为通用的数据采样策略，应用到主动学习、课程学习等场景
- "阻断深层前向"可推广到推理阶段的早期退出 (Early Exit) 策略

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 前向阻断范式+概率密度重要性，双重创新
- **实验充分度**: ⭐⭐⭐⭐ CIFAR-10/100/ImageNet + CNN/Transformer + 分割任务全覆盖
- **写作质量**: ⭐⭐⭐⭐ 方法描述清晰，计算开销分析详尽
- **价值**: ⭐⭐⭐⭐⭐ 实现了真正的无损训练加速，33%时间减少+精度提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs](../../NeurIPS2025/model_compression/casspec_cascade_adaptive_selfspeculative_decoding_for_onthef.md)
- [\[ACL 2025\] Disentangling the Roles of Representation and Selection in Data Pruning](../../ACL2025/model_compression/disentangling_the_roles_of_representation_and_selection_in_data_pruning.md)
- [\[CVPR 2025\] Style Quantization for Data-Efficient GAN Training](../../CVPR2025/model_compression/style_quantization_for_data-efficient_gan_training.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)
- [\[CVPR 2026\] LoPrune: Efficient Data Pruning for LoRA-Based Fine-Tuning of Vision Transformer](../../CVPR2026/model_compression/loprune_efficient_data_pruning_for_lora-based_fine-tuning_of_vision_transformer.md)

</div>

<!-- RELATED:END -->
