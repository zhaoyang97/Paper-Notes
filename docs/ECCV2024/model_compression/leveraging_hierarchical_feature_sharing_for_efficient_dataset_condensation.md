---
title: >-
  [论文解读] Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation
description: >-
  [ECCV 2024][模型压缩][数据蒸馏] 提出层级记忆网络（HMN），将数据蒸馏中的合成数据存储为三层结构（数据集级-类级-实例级记忆），通过层级化特征共享提升存储效率，并利用实例级剪枝进一步去除冗余，仅用低GPU内存的 batch-based loss 即超越所有基线方法。 数据蒸馏的核心目标 数据蒸馏（Data C…
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "数据蒸馏"
  - "数据参数化"
  - "层级特征共享"
  - "数据冗余剪枝"
  - "存储效率"
---

# Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation

**会议**: ECCV 2024  
**arXiv**: [2310.07506](https://arxiv.org/abs/2310.07506)  
**代码**: 无（论文未提供代码链接）  
**领域**: 模型压缩 / 数据蒸馏  
**关键词**: 数据蒸馏, 数据参数化, 层级特征共享, 数据冗余剪枝, 存储效率

## 一句话总结

提出层级记忆网络（HMN），将数据蒸馏中的合成数据存储为三层结构（数据集级-类级-实例级记忆），通过层级化特征共享提升存储效率，并利用实例级剪枝进一步去除冗余，仅用低GPU内存的 batch-based loss 即超越所有基线方法。

## 研究背景与动机

### 数据蒸馏的核心目标

数据蒸馏（Data Condensation, DC）旨在从大规模真实数据集 $\mathcal{D}$ 中合成一个极小的数据集 $\mathcal{S}$（$|\mathcal{S}| \ll |\mathcal{D}|$），使得在 $\mathcal{S}$ 上训练的模型能接近在 $\mathcal{D}$ 上训练的效果。随着数据集规模不断增大，DC 在持续学习、神经架构搜索、联邦学习等场景中展现出重要价值。

### 数据参数化的兴起与不足

近期工作提出**数据参数化**，即不直接存储合成图像，而是将数据蒸馏为参数化的数据容器 $f_\theta$，通过共享特征进一步提升存储效率。例如：
- **IDC**：通过下采样图像来提升存储效率
- **HaBa/LinBa**：基于矩阵分解共享图像间的公共信息

然而，这些方法忽略了一个关键事实：**图像的特征共享具有层级结构**。在分类体系中，两张猫的图片共享"猫类"特征，而猫和狗的图片则共享更高层的"动物类"特征。现有分解方法是扁平化的，无法捕捉这种层级共享关系。

### 冗余数据的问题

作者还发现，现有蒸馏数据集中存在明显的数据冗余。通过 AUM（Area Under the Margin）指标衡量数据难度/重要性，发现 HaBa 生成的 CIFAR10 10 IPC 数据集中，至少 10% 的数据可以被剪枝而不影响精度。但由于现有数据容器中权重的耦合关系，直接剪枝非常困难——同一个基向量生成的图像难度可能差异巨大，简单剪枝基向量会误删有价值的数据。

### 设计动机

HMN 的设计动机直接对应两个发现：（1）层级结构对齐分类体系的层级特征共享；（2）实例级记忆的独立性使剪枝变得简单自然。

## 方法详解

### 整体框架

HMN 是一个参数化数据容器，给定图像索引 $i$，输出合成图像 $\mathcal{S}_i$。它包含三层记忆结构和辅助网络，所有组件的参数量计入存储预算。

### 关键设计

#### 1. 三层层级记忆网络

**功能**：用三层可学习参数张量存储蒸馏数据，分别对应不同粒度的特征共享。

**核心思路**：

- **数据集级记忆** $m^{(\mathcal{D})}$：全局共享，存储所有图像的公共信息（如低级纹理、颜色分布等），所有类别共用一份
- **类级记忆** $m_c^{(C)}$：同类共享，存储类别特有特征（如"猫"的形态特征），数量等于类别数
- **实例级记忆** $m_{c,i}^{(I)}$：每张图像独有，存储区分个体的信息，数量决定生成图像数

生成第 $c$ 类第 $i$ 张图像的过程：

$$x_{c,i} = D([f_c(m^{(\mathcal{D})}) \oplus m_c^{(C)} \oplus m_{c,i}^{(I)}])$$

其中 $f_c$ 是类别特定的特征提取器（从共享记忆中提取该类相关特征），$D$ 是统一的解码器（将拼接的记忆转换为图像），$\oplus$ 表示拼接操作。

**设计动机**：层级结构自然对齐分类体系——数据集级对应"通用视觉特征"，类级对应"类别判别特征"，实例级对应"个体差异"。这种信息的层级分离既提升了存储效率（大量信息在高层级共享），又为后续剪枝提供了清晰的接口。

#### 2. 过预算蒸馏 + 双端剪枝

**功能**：先用超出存储预算 $p\%$ 的容量进行蒸馏，再通过剪枝实例级记忆回到预算内，同时提升性能。

**核心思路**：利用 AUM 指标衡量每张合成图像的学习难度，然后进行"双端剪枝"：

$$M^{(t)}(\mathbf{x}, y) = z_y^{(t)}(\mathbf{x}) - \max_{i \neq y} z_i^{(t)}(\mathbf{x})$$

$$\text{AUM}(\mathbf{x}, y) = \frac{1}{T}\sum_{t=1}^{T} M^{(t)}(\mathbf{x}, y)$$

通过网格搜索确定最优的硬样本剪枝比例 $\beta$：
- 剪掉 $\lfloor\beta k\rfloor$ 个最低 AUM（最难）样本
- 剪掉 $k - \lfloor\beta k\rfloor$ 个最高 AUM（最容易）样本

保持每个类别剪枝数量均衡。

**设计动机**：受 CCS 启发，太简单和太难的数据都不利于训练。太简单的数据冗余（容易被分类器学会），太难的数据可能是噪声或异常。HMN 的实例级记忆独立性使得剪枝单张图像只需移除对应的实例级记忆，不影响其他图像——这是 HaBa/LinBa 等分解方法做不到的。

#### 3. 训练优化

**功能**：使用低 GPU 内存的 batch-based loss 优化 HMN。

**核心思路**：采用梯度匹配（gradient matching）作为训练损失：

$$\min_{\mathcal{S}} \mathbf{E}_{\theta_0 \sim P_{\theta_0}} \left[\sum_{t=0}^{T-1} d(\nabla_\theta \mathcal{L}(\theta_t, \mathcal{S}), \nabla_\theta \mathcal{L}(\theta_t, \mathcal{T}))\right]$$

即最小化合成数据和真实数据在模型上产生的梯度距离。相比于 trajectory-based loss（如 MTT），gradient matching 内存需求低得多。

**设计动机**：trajectory-based loss 虽然通常效果更好，但 GPU 内存消耗极大（LinBa 甚至需要 CPU offloading），严重限制了实用性。HMN 的层级架构即使配合简单的 batch-based loss 也能取得优异性能，体现了架构设计的重要性。

### 损失函数 / 训练策略

- 使用 IDC 的梯度匹配变体作为训练 loss
- 过预算率设为 10%（更高的剪枝率会显著降低精度）
- 所有实验重复 3 次数据蒸馏，每次蒸馏后重复 10 次训练取均值和标准差
- 使用 ConvNet（3 层卷积 + 池化）作为蒸馏和评估网络

## 实验关键数据

### 主实验

**CIFAR10 上各方法对比（测试精度 %）**

| 方法 | 容器类型 | 1 IPC | 10 IPC | 50 IPC |
|------|---------|-------|--------|--------|
| DC | 图像 | 28.3 | 44.9 | 53.9 |
| DSA | 图像 | 28.8 | 52.1 | 60.6 |
| DM | 图像 | 26.0 | 48.9 | 63.0 |
| MTT* | 图像 | 46.3 | 65.3 | 71.6 |
| IDC | 参数化 | 50.0 | 67.5 | 74.5 |
| HaBa* | 参数化 | 48.3 | 69.9 | 74.0 |
| LinBa* | 参数化 | **66.4** | 71.2 | 73.6 |
| **HMN (Ours)** | 参数化 | 65.7 | **73.7** | **76.9** |

*注：带 * 的方法使用了高内存 trajectory-based loss，HMN 使用低内存 batch-based loss*

**多数据集表现**

| 数据集 | IPC | HMN | 最佳基线 | 提升/差距 |
|--------|-----|-----|---------|----------|
| CIFAR100 | 1 | **36.3** | 34.0 (LinBa) | +2.3 |
| CIFAR100 | 10 | **45.4** | 42.9 (LinBa) | +2.5 |
| SVHN | 1 | **87.4** | 87.3 (LinBa) | +0.1 |
| SVHN | 10 | **90.0** | 89.1 (LinBa) | +0.9 |
| Tiny-ImageNet | 1 | **19.4** | 16.0 | +3.4 |
| Tiny-ImageNet | 10 | **24.4** | 23.2 | +1.2 |
| ImageNet-10 | 1 | **64.6** | 60.4 | +4.2 |

### 消融实验

**相同 loss 下数据容器对比 (CIFAR10, gradient matching)**

| 数据容器 | 1 IPC | 10 IPC | 50 IPC |
|---------|-------|--------|--------|
| 原始图像 | 36.7 | 58.3 | 69.5 |
| IDC | 50.0 | 67.5 | 74.5 |
| HaBa | 48.5 | 61.8 | 72.4 |
| LinBa | 62.0 | 67.8 | 70.7 |
| **HMN** | **65.7** | **73.7** | **76.9** |

HMN 在公平对比下优势更加明显，分别领先次优方法 3.7%/5.9%/2.4%。

**跨架构迁移性 (CIFAR10, ConvNet 蒸馏 → 其他架构评估)**

| 架构 | HMN (1/10/50) | IDC (1/10/50) | HaBa (1/10/50) |
|------|--------------|--------------|----------------|
| ConvNet | 65.7/73.7/76.9 | 50.0/67.5/74.5 | 48.3/69.9/74.0 |
| VGG16 | 58.5/64.3/70.2 | 28.7/43.1/57.9 | 34.1/53.8/61.1 |
| ResNet18 | 56.8/62.9/69.1 | 32.3/45.1/58.4 | 36.0/49.0/60.4 |
| DenseNet121 | 50.7/56.9/65.1 | 24.3/38.5/50.5 | 34.6/49.3/57.8 |

**GPU 内存对比 (CIFAR10)**

| IPC | HaBa (MTT) | LinBa (BPTT) | HMN (GM) |
|-----|-----------|-------------|---------|
| 1 | 3368M | OOM | **2680M** |
| 10 | 11148M | OOM | **4540M** |
| 50 | 48276M | OOM | **10426M** |

### 关键发现

1. HMN 用低内存 batch-based loss 就能超越使用高内存 trajectory-based loss 的 HaBa 和 LinBa，证明了架构设计比训练策略更重要
2. 在公平对比（相同 loss）下，HMN 的优势被放大到 3.7-5.9% 
3. HMN 的跨架构迁移性远优于其他数据参数化方法（如 VGG16 上领先 IDC 近 30%）
4. GPU 内存仅需 HaBa 的 22%-79%，LinBa 则直接 OOM
5. 实例级记忆大小需要平衡：太小则单图信息不足，太大则生成图像数量过少损害多样性
6. 10% 的过预算剪枝率是最优选择，更高的剪枝率会显著降低精度

## 亮点与洞察

1. **层级思想的自然对齐**：将分类体系的层级结构直接编码进数据容器的架构设计中，是一个既直觉又有效的 insight
2. **设计的克制**：作者尝试了更复杂的设计（如类间不同解码器、更多中间网络），发现简单设计反而更好——过多参数导致过拟合。这体现了数据蒸馏中"简洁即有效"的原则
3. **"过量蒸馏-再剪枝"范式**：类似于模型过参数化训练后剪枝，将这一思路应用于数据蒸馏是新颖的
4. **实用性强**：训练时间从 LinBa 的 14 天降至 15 小时（2080TI），内存需求大幅降低

## 局限与展望

1. **仅使用 batch-based loss**：尽管证明了 HMN + batch loss 已经很好，但与 trajectory-based loss 结合可能进一步提升（受限于 GPU 内存未尝试）
2. **剪枝策略较为简单**：当前的 AUM 双端剪枝需要额外的训练来计算 AUM 值，且最优剪枝参数 $\beta$ 需网格搜索
3. **大规模数据集评估不足**：仅在 Tiny-ImageNet 和 ImageNet-10 上实验，未验证完整 ImageNet 规模
4. **解码器设计固定**：统一的解码器对所有类别使用相同的解码路径，可能限制了类间差异的表达
5. **蒸馏网络与评估网络相同**：主实验以 ConvNet 为主，虽然有跨架构实验但性能有明显下降

## 相关工作与启发

- **与 HaBa/LinBa 的关系**：这两个工作用矩阵分解实现扁平化特征共享，HMN 将其拓展为层级共享
- **与 coreset selection 的关系**：剪枝策略借鉴了 CCS 的双端剪枝思想，将数据重要性度量应用于合成数据
- **启发**：层级特征共享的思想可以推广到其他需要紧凑数据表示的场景（如联邦学习中的数据通信压缩、持续学习中的记忆缓冲区设计）

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 层级记忆网络和"过预算蒸馏+剪枝"范式都是新颖的贡献，但核心思路（层级共享）是自然的推广
- **实验充分度**: ⭐⭐⭐⭐ — 多数据集、跨架构、内存对比、消融实验充分，但缺少大规模数据集验证
- **写作质量**: ⭐⭐⭐⭐ — 动机和方法阐述清晰，图表直观，但表格格式略显拥挤
- **价值**: ⭐⭐⭐⭐ — 大幅降低数据蒸馏的计算和内存成本，实用价值高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Post Training Quantization for Efficient Dataset Condensation](../../AAAI2026/model_compression/post_training_quantization_for_efficient_dataset_condensation.md)
- [\[CVPR 2026\] PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion](../../CVPR2026/model_compression/prism_video_dataset_condensation_with_progressive_refinement_and_insertion_for_s.md)
- [\[ECCV 2024\] Improving Knowledge Distillation via Regularizing Feature Direction and Norm](improving_knowledge_distillation_via_regularizing_feature_direction_and_norm.md)
- [\[CVPR 2026\] ManifoldGD: Training-Free Hierarchical Manifold Guidance for Diffusion-Based Dataset Distillation](../../CVPR2026/model_compression/manifoldgd_training-free_hierarchical_manifold_guidance_for_diffusion-based_data.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](../../NeurIPS2025/model_compression/toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)

</div>

<!-- RELATED:END -->
