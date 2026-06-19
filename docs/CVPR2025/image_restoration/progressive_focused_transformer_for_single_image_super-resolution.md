---
title: >-
  [论文解读] Progressive Focused Transformer for Single Image Super-Resolution
description: >-
  [CVPR 2025][图像恢复][超分辨率] PFT 提出渐进聚焦注意力（PFA）机制，通过在相邻 Transformer 层之间传递注意力图的 Hadamard 乘积，实现逐层筛选不相关 token 并增强关键 token 的权重，在超分辨率任务上达到 SOTA 性能的同时显著降低计算开销。 领域现状：基于 Transf…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "超分辨率"
  - "稀疏注意力"
  - "Transformer"
  - "渐进聚焦"
  - "注意力传递"
---

# Progressive Focused Transformer for Single Image Super-Resolution

**会议**: CVPR 2025  
**arXiv**: [2503.20337](https://arxiv.org/abs/2503.20337)  
**代码**: [https://github.com/LabShuHangGU/PFT-SR](https://github.com/LabShuHangGU/PFT-SR)  
**领域**: 图像复原  
**关键词**: 超分辨率, 稀疏注意力, Transformer, 渐进聚焦, 注意力传递

## 一句话总结

PFT 提出渐进聚焦注意力（PFA）机制，通过在相邻 Transformer 层之间传递注意力图的 Hadamard 乘积，实现逐层筛选不相关 token 并增强关键 token 的权重，在超分辨率任务上达到 SOTA 性能的同时显著降低计算开销。

## 研究背景与动机

**领域现状**：基于 Transformer 的超分辨率方法（如 SwinIR、HAT、ATD）利用自注意力机制捕捉长程依赖来恢复高分辨率细节。由于自注意力的二次复杂度，大多数方法将注意力限制在局部窗口内。

**现有痛点**：现有方法面临两难困境。一类方法（HAT、ATD）试图扩大窗口或引入外部信息来获取更多 token 交互，但更多 token 带来更大计算开销。另一类方法（NLSA、DRSformer）使用稀疏注意力过滤不相关 token，但仍然需要先计算所有 token 对的相似度再选择 top-k，无法在计算前就排除不相关 token。

**核心矛盾**：在相似度计算前识别不相关 token 并跳过其计算是一个关键的未解决问题。标准注意力和稀疏注意力都需要完整计算相似度矩阵，这限制了使用更大窗口的可能性。

**本文目标**：设计一种能在计算相似度之前就过滤不相关 token 的注意力机制，从而在更大窗口上以更少的计算获得更好的聚合效果。

**切入角度**：作者观察到一个关键事实——如果某 token 在前面的层中就被判定为不相关（注意力权重很小），那么在后续层中也大概率仍不相关。因此可以利用前层的注意力图来指导后层跳过计算。

**核心 idea**：将相邻层的注意力图通过 Hadamard 乘积连接起来，让注意力在层间"渐进聚焦"——一致性高相关的 token 权重逐层增强，低相关 token 权重逐层衰减至零，从而实现计算前过滤。

## 方法详解

### 整体框架

PFT 遵循与 SwinIR、HAT 等相似的编码器-重建器架构，由 6 个 PFA Block 组成。与标准 Transformer Block 不同的是，PFA Block 内的多个注意力层共享并传递注意力图，形成从密到疏的渐进过程。输入是低分辨率图像，输出是高分辨率重建结果。窗口大小为 32×32（远大于 SwinIR 的 8×8），使得模型能利用更广范围的信息。

### 关键设计

1. **渐进注意力继承 (Progressive Attention Across Layers)**:

    - 功能：将前层的注意力权重传递到当前层，对注意力进行跨层累积过滤
    - 核心思路：当前层的最终注意力图由计算的注意力图 $\mathbf{A}_{cal}^l$ 与前层注意力图 $\mathbf{A}^{l-1}$ 的 Hadamard 乘积后归一化得到：$\mathbf{A}^l = Norm(\mathbf{A}^{l-1} \odot \mathbf{A}_{cal}^l)$。这意味着只有在多层中一致表现为高相似度的 token 对才能保持大权重，而任何一层中的小权重都会在乘积过程中被放大
    - 设计动机：标准 self-attention 仅根据单步相似度计算决定权重，对高相关和低相关 token 的区分能力有限。通过多层累积，PFA 对 token 相关性做出更全面的评估

2. **稀疏矩阵乘法预过滤 (Sparse Matrix Multiplication)**:

    - 功能：利用前层注意力图中的零位置跳过当前层的相似度计算
    - 核心思路：由于最终注意力图会乘以前层的注意力图，前层中权重已为零的位置不需要计算。通过维护稀疏索引矩阵 $\mathbf{I}^{l-1}$，SMM 操作只对 $\mathbf{I}^{l-1}(i,j)=1$ 的位置计算 $Q^l(i,:)$ 和 $K^l(j,:)^T$ 的点积。每层保留 top-$K^l$ 个非零值，且 $K^l = \alpha K^{l-1}$（$\alpha < 1$），实现逐层递减的关注范围
    - 设计动机：这直接实现了"计算前过滤"——不用先算出所有相似度再筛选，而是根据前面层已确定的不相关位置直接跳过，将计算复杂度从 $O(W^2)$ 指数级下降。开发了专门的 CUDA kernel 来高效实现稀疏乘法

3. **渐进聚焦资源分配策略**:

    - 功能：系统性地安排各层的计算资源，浅层密集、深层稀疏
    - 核心思路：第一层 $K^1 = N$（窗口中所有 token），用标准 self-attention 计算完整注意力图作为初始基础。后续层按 $K^l = \alpha K^{l-1}$ 逐层减少保留数量。具体设置为 6 个 block 中分别保留 [1024, 256, 128, 64, 32, 16] 个注意力值
    - 设计动机：浅层需要广泛探索避免过早排除重要 token，深层已有足够信息可以大胆聚焦。这种资源分配使得 PFT 可以使用 32×32 的超大窗口，而计算开销与使用小窗口的方法相当

### 损失函数 / 训练策略

PFT 使用标准的 L1 像素损失进行训练。模型遵循经典 SR 训练策略：使用 DF2K 数据集训练，输入 LR patch 大小为 64×64。PFT 采用 SwinIR 类似的 shift-window 策略，注意力在奇偶层间交替传递。LePE 位置编码被加入到注意力计算中。

## 实验关键数据

### 主实验

| 方法 | 参数量 | FLOPs | Set5 (×2) | Urban100 (×2) | Manga109 (×2) |
|------|--------|-------|-----------|-------------|-------------|
| SwinIR | 11.8M | 3.04T | 38.42 | 33.81 | 39.92 |
| HAT | 20.6M | 5.81T | 38.63 | 34.45 | 40.26 |
| ATD | 20.1M | 6.07T | 38.61 | 34.70 | 40.37 |
| **PFT** | **19.6M** | **5.03T** | **38.68** | **34.90** | **40.49** |

×3 倍率下：PFT 在 Urban100 上达到 30.56 dB，超越 ATD (30.46) 和 IPG (30.36)。

### 消融实验

| 配置 | 说明 | PSNR 影响 |
|------|------|----------|
| 标准 Self-Attention | 不用 PFA | 基线 |
| Top-k 稀疏注意力 | 用 top-k 但不传递 | 优于标准 SA |
| Progressive Attention (无稀疏) | 乘积传递但不跳过计算 | 优于 top-k |
| PFA (完整) | 渐进聚焦 + 稀疏矩阵乘法 | 最优 |

### 关键发现

- PFT 在参数量（19.6M）和 FLOPs（5.03T）都低于 HAT 和 ATD 的情况下，在全部 5 个测试集上达到最佳 PSNR/SSIM
- 渐进注意力传递带来的增益大于简单 top-k 稀疏选择——验证了跨层信息累积相比单层选择的优越性
- 在 Urban100 等结构丰富的数据集上提升最明显（×2 比 ATD 高 0.20 dB），说明 PFA 更善于利用长程结构相似性
- 使用 $\alpha=0.5$ 时，经过 4 步衰减后计算复杂度降至原来的 6.25%，使得 32×32 的大窗口变得可行

## 亮点与洞察

- **计算前过滤**是本文最巧妙的设计。不同于 top-k 先算再选，PFA 直接跳过不需要的计算，这是对稀疏注意力理念的质的提升。本质上是用前面层的"廉价"信息来指导后续层的"昂贵"计算
- **注意力图的乘积传递**思想可泛化到其他视觉 Transformer 架构。任何需要在较大范围内做注意力的任务（如视频理解、密集预测）都可以借鉴这种"先粗后精"的渐进聚焦策略
- 自定义 CUDA kernel 实现的 SMM 是将理论优势转化为实际加速的关键工程贡献

## 局限与展望

- PFA 的渐进聚焦策略假设一旦某 token 被标记为不相关就永久排除，但实际上有些 token 在深层可能变得重要（如远距离的语义相关 patch）
- 聚焦比率 $\alpha$ 是全局固定的，不同图像区域（纹理丰富 vs 平滑区域）可能需要不同的衰减速度
- 论文只在图像超分任务上验证，是否能有效迁移到去噪、去模糊等其他低级视觉任务还需进一步验证
- 32×32 的窗口虽大但仍是固定的，未来可探索自适应窗口大小

## 相关工作与启发

- **vs HAT**: HAT 通过通道注意力和窗口注意力的结合扩大感受野，但窗口内仍是密集注意力。PFT 通过稀疏化在更大窗口内实现更高效的信息交互
- **vs ATD**: ATD 引入外部 token 字典来弥补局部窗口的信息不足。PFT 则从内部优化注意力机制本身，减少冗余计算以支持更大窗口
- **vs DRSformer**: DRSformer 使用可学习 top-k 选择，但仍需计算完整相似度矩阵后再稀疏化。PFT 在计算前就完成过滤，效率更高

## 评分

- 新颖性: ⭐⭐⭐⭐ 渐进聚焦注意力的跨层传递是对稀疏注意力的有力改进
- 实验充分度: ⭐⭐⭐⭐⭐ 多尺度、多数据集对比，详细的复杂度分析
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，数学推导完整
- 价值: ⭐⭐⭐⭐ 提升了 SR SOTA，且 PFA 机制有较好的通用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Augmenting Perceptual Super-Resolution via Image Quality Predictors](augmenting_perceptual_super-resolution_via_image_quality_predictors.md)
- [\[CVPR 2025\] PIDSR: Complementary Polarized Image Demosaicing and Super-Resolution](pidsr_complementary_polarized_image_demosaicing_and_super-resolution.md)
- [\[CVPR 2025\] Gyro-based Neural Single Image Deblurring](gyro-based_neural_single_image_deblurring.md)
- [\[CVPR 2026\] DreamSR: Towards Ultra-High-Resolution Image Super-Resolution via a Receptive-Field Enhanced Diffusion Transformer](../../CVPR2026/image_restoration/dreamsr_towards_ultra-high-resolution_image_super-resolution_via_a_receptive-fie.md)
- [\[CVPR 2026\] One-Step Diffusion Transformer for Controllable Real-World Image Super-Resolution](../../CVPR2026/image_restoration/one-step_diffusion_transformer_for_controllable_real-world_image_super-resolutio.md)

</div>

<!-- RELATED:END -->
