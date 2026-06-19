---
title: >-
  [论文解读] Post Training Quantization for Efficient Dataset Condensation
description: >-
  [AAAI 2026 Oral][模型压缩][数据集蒸馏] 首次将训练后量化（PTQ）应用于数据集蒸馏，提出基于补丁的量化框架（PAQ+分组+精炼），在 2-bit 极低比特下将蒸馏数据集的测试精度几乎翻倍（如 DM IPC=1 从 26.0% 提升至 54.1%），作为即插即用框架可应用于各种蒸馏方法。
tags:
  - "AAAI 2026 Oral"
  - "模型压缩"
  - "数据集蒸馏"
  - "训练后量化"
  - "低比特存储"
  - "图像压缩"
  - "补丁量化"
---

# Post Training Quantization for Efficient Dataset Condensation

**会议**: AAAI 2026 Oral  
**arXiv**: [2603.13346](https://arxiv.org/abs/2603.13346)  
**代码**: 无  
**领域**: 模型压缩  
**关键词**: 数据集蒸馏, 训练后量化, 低比特存储, 图像压缩, 补丁量化

## 一句话总结
首次将训练后量化（PTQ）应用于数据集蒸馏，提出基于补丁的量化框架（PAQ+分组+精炼），在 2-bit 极低比特下将蒸馏数据集的测试精度几乎翻倍（如 DM IPC=1 从 26.0% 提升至 54.1%），作为即插即用框架可应用于各种蒸馏方法。

## 研究背景与动机

### 领域现状
数据集蒸馏（Dataset Condensation, DC）通过将大数据集的知识浓缩为小数据集来加速训练和减少存储。现有方法（梯度匹配、分布匹配、轨迹匹配）主要关注生成质量，但忽略了存储效率——每个合成样本仍需全精度存储。参数化 DC（PDC）方法如 IDC（空间降采样）、AutoPalette（颜色缩减）、DDiF（神经场）虽提升了压缩率，但仍依赖 32-bit 表示。

### 现有痛点

**存储冗余**：合成图像按 32-bit 浮点存储，浪费大量空间

**现有 PDC 方法计算成本高**：AutoPalette 需要训练调色板编码器，DDiF 需要神经场网络，推理时需解码

**位级冗余未被利用**：在同等存储预算下，降低比特宽度可以存储更多样本，但全图量化在极低比特下会严重退化

**从未探索 PTQ 用于 DC**：尽管 PTQ 在模型压缩中已广泛应用，但其在合成数据压缩中的潜力完全未被开发

### 核心矛盾
如何在极低比特宽度（如 2-bit）下量化合成图像，同时保持其对下游模型训练的有效性？

### 切入角度
提出补丁级量化：将图像分为非重叠补丁，每个补丁独立量化以保留局部细节。通过聚类分组共享量化参数减少开销，并用精炼模块对齐量化前后特征分布。

## 方法详解

### 整体框架
Pipeline：(I) 合成图像 → 量化感知精炼 → (II) 补丁提取 → k-means 聚类（按量化参数）→ 分组量化 → (III) 组内量化 + 熵编码 → 最终压缩数据集。推理时：解码 → 反量化 → 直接用于训练。

### 关键设计

1. **补丁级非对称量化（PAQ）**:

    - 将图像 x 分为 P 个非重叠补丁 $\{x_i\}_{i=1}^P$，每个补丁 $x_i \in \mathbb{R}^{h \times w \times C}$
    - 每个补丁独立量化：$x_i^q = Q(x_i, \theta_i)$，其中 $\theta_i = (\alpha_i, z_i)$
    - 非对称量化公式：
        - 缩放因子：$\alpha = \frac{\max(x) - \min(x)}{Q_{max} - Q_{min}}$
        - 零点：$z = \lfloor Q_{min} - \frac{\min(x)}{\alpha} \rceil$
        - 量化/反量化：$x^q = \lfloor \frac{x}{\alpha} + z \rceil$，$x^{deq} = (x^q - z) \cdot \alpha$
    - 相比全图量化：2-bit 下 PAQ 达到 47.5% vs 全精度 48.9%，几乎无损
    - 设计动机：全图量化用单一参数覆盖整张图，无法适应空间上纹理和细节的变化

2. **量化感知补丁分组（GAQ）**:

    - PAQ 为每个补丁存储独立参数，增加存储开销
    - 在量化参数空间 $(\alpha_i, z_i)$ 上执行 k-means 聚类
    - 目标：最小化组内量化参数方差
    - $\{\mathcal{C}_g^*, \theta_g^*\}_{g=1}^G = \arg\min \sum_{g=1}^G \sum_{\theta_i \in \mathcal{C}_g} \|\theta_i - \hat{\theta}_g\|^2$
    - **组内重校准**：不直接用聚类中心作为量化参数，而是拼接组内所有补丁重新计算
    - $x_g = \text{concat}(\{x_i\}_{i \in \mathcal{C}_g})$，在展平的 $x_g^{flat}$ 上校准 $\theta_g$
    - 设计动机：在存储开销和量化质量之间平衡——相似补丁共享参数

3. **量化感知精炼模块**:

    - 优化精炼图像 $x^{ft}$ 使其量化后的特征与原始图像对齐
    - 提取特征：$\mathbf{f} = f(x)$，$\tilde{\mathbf{f}} = f((x^{ft})^{deq})$
    - 最小化特征空间 MSE：$\mathcal{L}_{quant} = \mathbb{E}_{x \sim S}[\|\mathbf{f} - \tilde{\mathbf{f}}\|_2^2]$
    - 三种策略：(1) 仅分组前精炼，(2) 仅分组后精炼，(3) 前后都精炼
    - 实验发现分组前精炼效果最好（因为分组依据更准确的量化参数）
    - 设计动机：直接补偿量化噪声导致的特征漂移

4. **存储测量与熵编码**:

    - 总存储 = 组索引 $\mathcal{G}$ + 量化参数 $\mathcal{Q}$ + 量化图像 $\mathcal{X}^q$
    - 对 $\mathcal{X}^q$ 额外应用熵编码（EC）利用统计冗余
    - 约束：$size(\mathcal{G}) + size(\mathcal{Q}) + size(EC(\mathcal{X}^q)) \leq size(\text{IPC})$
    - 在同等预算下可存储更多量化样本，提高数据集的表征密度

### 训练策略
- 默认 2-bit 量化，5×5 非重叠补丁
- 网格搜索确定满足存储约束的最大分组数
- 精炼迭代：CIFAR-10/100 为 500 次，ImageNet 子集为 2000 次
- 评估：在压缩数据集上训练模型，在原始测试集上测试
- 即插即用：适用于 DM、DSA、DATM 等各种蒸馏方法生成的合成图像

## 实验关键数据

### 主实验

| 方法 | CIFAR-10 IPC=1 | IPC=10 | IPC=50 | CIFAR-100 IPC=1 | IPC=10 |
|------|---------------|--------|--------|----------------|--------|
| DM | 26.0 | 48.9 | 63.0 | 11.4 | 29.7 |
| DSA | 28.8 | 52.1 | 60.6 | 13.9 | 32.3 |
| DATM | 46.9 | 66.8 | 76.1 | 27.9 | 47.2 |
| AutoPalette | 58.6 | 74.3 | 79.4 | 38.0 | 52.6 |
| **DM+Ours** | **54.1** | 68.2 | 77.1 | 34.0 | 51.2 |
| **DSA+Ours** | **55.3** | 58.3 | 73.4 | 34.7 | 41.1 |
| **DATM+Ours** | **68.9** | **79.0** | **83.8** | **48.0** | **56.5** |

| 数据集 | I-Nette | I-Woof | I-Fruit | I-Meow | I-Squawk | I-Yellow |
|--------|---------|--------|---------|--------|----------|----------|
| DATM | 65.8 | 38.8 | 41.2 | 45.7 | 56.3 | 61.1 |
| AutoPalette | 73.2 | 44.3 | 48.4 | 53.6 | 68.0 | 72.0 |
| **DATM+Ours** | **81.1** | **53.0** | **56.6** | **61.2** | **80.6** | **78.9** |

### 消融实验

| GAQ | Refinement | EC | CIFAR-10 (IPC=10) | I-Nette |
|-----|-----------|----|--------------------|---------|
| ✗ | ✗ | ✗ | 71.8 | 75.2 |
| ✓ | ✗ | ✗ | 76.1 (+4.3) | 76.5 |
| ✓ | ✓ | ✗ | 77.2 (+1.1) | 77.2 |
| ✓ | ✓ | ✓ | **79.0** (+1.8) | **81.1** |

| 精炼时机 | CIFAR-10 IPC=1 | 说明 |
|---------|---------------|------|
| 仅分组前 | **68.9** | 最佳 |
| 仅分组后 | 68.7 | 略差 |
| 前后都精炼 | 68.9 | 无额外提升 |

### 关键发现

1. **极端压缩下翻倍性能**：DM IPC=1 从 26.0%→54.1%，DSA 从 28.8%→55.3%，证明 2-bit 量化+补丁方法在极低存储预算下极其有效
2. **DATM+Ours 全面 SOTA**：在所有 IPC 设定和数据集上均超越 AutoPalette 等 PDC 方法，且无需额外网络
3. **各组件贡献清晰**：AQ→GAQ（+4.3）→精炼（+1.1）→EC（+1.8），每步均有正向贡献
4. **跨架构泛化**：在 ConvNet、AlexNet、VGG11、ResNet18 上均大幅超越 DATM 基线
5. **跨模态泛化**：在音频（MobileNet/SqueezeNet）和 3D 体素数据上同样有效
6. **跨数据集泛化**：CC3M 和 Places365 等真实世界大规模数据集上的优势尤为明显
7. **精炼时机分析**：分组前精炼效果最好，因为提供了更准确的量化参数给后续分组
8. **可视化对比**：Median Cut 保纹理丢颜色，AQ 保颜色丢纹理，GAQ 在二者间取得更好平衡

## 亮点与洞察
1. 首次将 PTQ 引入数据集蒸馏领域，开辟了全新研究方向
2. 补丁级量化的设计直觉清晰：局部适应空间变化，比全局量化保留更多细节
3. 分组策略在量化参数空间而非像素空间聚类，精确捕捉量化行为的相似性
4. 即插即用设计使其可以与任何蒸馏方法组合，实用价值极高
5. 2-bit 极端量化下依然有效，说明合成图像的信息可以被高度压缩
6. 存储预算公式化为明确的约束优化问题，方便工程实践

## 局限与展望
1. 补丁大小固定为 5×5，可探索自适应补丁大小
2. k-means 聚类的组数通过网格搜索确定，可开发更高效的自动选择方法
3. 精炼模块需要神经网络提取特征，引入了对网络选择的依赖
4. 仅在 2-bit 和 4-bit 上验证，其他比特宽度（如 3-bit）可进一步探索
5. 在 CIFAR-100 IPC=50 超过原始每类 500 张的限制，无法测试
6. 与学习型压缩方法（如变分自编码器）的比较缺失

## 相关工作与启发
- **AutoPalette (Yuan 2024a)**：颜色冗余缩减 → 本文直接做位级冗余缩减，更底层更通用
- **DDiF (Shin 2025)**：神经场编码 → 计算成本高; 本文 PTQ 无需额外网络
- **IDC (Kim 2022)**：空间降采样 → 本文补丁级量化保留更多空间信息
- **PTQ 在模型压缩中的广泛应用** → 首次迁移到数据压缩
- SPEED、FreD、Spectral 等频域/谱方法 → 仍用 32-bit，本文直接降到 2-bit

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首创 PTQ 用于数据集蒸馏，开辟新方向)
- 实验充分度: ⭐⭐⭐⭐⭐ (多数据集+多蒸馏方法+多架构+多模态+消融全面)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，可视化好，但一些公式排版可改进)
- 价值: ⭐⭐⭐⭐⭐ (即插即用框架+极端压缩下翻倍性能，实用价值高)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] MetaAug: Meta-Data Augmentation for Post-Training Quantization](../../ECCV2024/model_compression/metaaug_meta-data_augmentation_for_post-training_quantization.md)
- [\[ECCV 2024\] Leveraging Hierarchical Feature Sharing for Efficient Dataset Condensation](../../ECCV2024/model_compression/leveraging_hierarchical_feature_sharing_for_efficient_dataset_condensation.md)
- [\[ICLR 2026\] Dataset Color Quantization: A Training-Oriented Framework for Dataset-Level Compression](../../ICLR2026/model_compression/dataset_color_quantization_a_training-oriented_framework_for_dataset-level_compr.md)
- [\[AAAI 2026\] QuantVSR: Low-Bit Post-Training Quantization for Real-World Video Super-Resolution](quantvsr_low-bit_post-training_quantization_for_real-world_video_super-resolutio.md)
- [\[CVPR 2026\] CAR-SAM: Cross-Attention Reconstruction for Post-Training Quantization of the Segment Anything Model](../../CVPR2026/model_compression/car-sam_cross-attention_reconstruction_for_post-training_quantization_of_the_seg.md)

</div>

<!-- RELATED:END -->
