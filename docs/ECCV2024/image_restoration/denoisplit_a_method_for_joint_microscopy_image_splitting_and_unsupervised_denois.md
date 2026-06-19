---
title: >-
  [论文解读] DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising
description: >-
  [ECCV 2024][图像恢复][图像分解] 提出 DenoiSplit，首个将语义图像分解（image splitting）和无监督去噪（unsupervised denoising）联合解决的方法，通过在层次化 VAE 中整合像素噪声模型和改进的 KL 散度损失加权策略，在荧光显微镜图像上实现了端到端的去噪+分解，性能显著优于先去噪再分解的串行方案。
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "图像分解"
  - "无监督去噪"
  - "荧光显微镜"
  - "变分自编码器"
  - "噪声模型"
---

# DenoiSplit: A Method for Joint Microscopy Image Splitting and Unsupervised Denoising

**会议**: ECCV 2024  
**arXiv**: [2403.11854](https://arxiv.org/abs/2403.11854)  
**代码**: [有 (GitHub)](https://github.com/juglab/denoiSplit)  
**领域**: 图像修复  
**关键词**: 图像分解, 无监督去噪, 荧光显微镜, 变分自编码器, 噪声模型

## 一句话总结

提出 DenoiSplit，首个将语义图像分解（image splitting）和无监督去噪（unsupervised denoising）联合解决的方法，通过在层次化 VAE 中整合像素噪声模型和改进的 KL 散度损失加权策略，在荧光显微镜图像上实现了端到端的去噪+分解，性能显著优于先去噪再分解的串行方案。

## 研究背景与动机

### 荧光显微镜中的图像分解问题

荧光显微镜是研究细胞和亚细胞结构的基础工具，但同时观察多个结构需要**多路复用成像协议（multiplexed imaging）**，费时费力。**语义图像分解**提供了一种替代方案：将一张包含多个结构叠加的图像分解为各自独立的通道图像。

形式化定义：给定叠加图像 $x_j = c_{1,j} + c_{2,j}$（两个通道的逐像素求和），任务是从 $x_j$ 恢复 $(c_{1,j}, c_{2,j})$。虽然两数之和不可唯一分解，但如果每个分量具有**结构先验**（如微管和囊泡的形态差异），则可以通过学习这些先验来实现分解。

### 噪声问题：现有方法的致命弱点

前期工作 μSplit 在相对无噪声的数据上展示了出色的分解性能。但在实际显微镜场景中，图像**总是带有显著噪声**（泊松噪声 + 高斯噪声），这是由成像设备的物理限制和样品对光敏感的约束决定的。

当 μSplit 在含噪数据上训练时，它会不可避免地将噪声也"分配"到预测输出中——即**噪声穿透**问题。这是因为 μSplit 的层次化隐空间中，底层（编码精细像素级结构的层级）对噪声没有足够的约束力。

### 为什么不能简单地"先去噪再分解"？

一种直觉方案是先用 HDN 等去噪方法对数据去噪，再用 μSplit 分解。但这种串行方案存在问题：
- 需要训练**三个独立的 HDN 模型**（分别对输入图像和两个通道去噪）+ 一个 μSplit 模型，计算开销大（约 11 小时 vs DenoiSplit 的 1.5 小时）
- 去噪模型的误差会传播到后续分解步骤
- 细微结构可能在去噪过程中丢失，影响后续生物分析

DenoiSplit 的核心洞察是：**去噪和分解应该联合进行**，通过在训练目标中显式建模噪声来同时实现两个目标。

## 方法详解

### 整体框架

DenoiSplit 基于**变分分解编码器-解码器网络（VSE Network）**，修改自层次化 VAE（HVAE）。与标准自编码器不同，其输出不是输入的重建，而是两个分解后的去噪通道图像 $(\hat{c}_1, \hat{c}_2)$。

训练目标是最大化含噪双通道数据的似然：

$$\boldsymbol{\theta} = \arg\max_{\boldsymbol{\theta}} \sum_{1 \leq j \leq n} \log P(c_{1,j}^N, c_{2,j}^N; \boldsymbol{\theta})$$

使用修改后的 ELBO 并假设两个通道预测在给定隐变量 $z$ 时条件独立。

### 关键设计

1. **层次化 KL 损失加权（Hierarchical KL Loss Weighing）**：解决噪声穿透问题的关键创新。

   HVAE 的隐空间是分层的：底层 $Z[i]$（小 $i$）编码精细像素级结构（分辨率高），高层编码大尺度结构。在 μSplit 中，每层的 KL 散度标量化为：
    $\text{kl}_i = \alpha \cdot \sum_{j,h,w} \frac{\text{KL}_i[j,h,w]}{h_i \cdot w_i}$
   即按**空间大小取平均**。这导致底层的每个像素位置的 KL 约束权重较小，结果是噪声容易通过底层隐空间"泄露"到输出中。

   DenoiSplit 改为**直接求和**（不除以空间大小）：
    $\text{kl}_i = \alpha \cdot \sum_{j,h,w} \text{KL}_i[j,h,w]$

   这使底层隐空间受到更强的高斯先验约束，阻止噪声被编码。这个看似微小的改动效果显著——单独使用此修改（称为 Altered μSplit）就能大幅提升含噪数据上的分解质量。

2. **像素噪声模型集成（Pixel Noise Models）**：实现无监督去噪的核心。

   将两个通道各自的对数似然项替换为噪声模型定义的似然：

    $E_{q(z|x;\phi)}[\log P^{nm}(c_1^N|\hat{c}_1) + \log P^{nm}(c_2^N|\hat{c}_2)] - KL(q(z|x;\phi), P(z))$

   噪声模型 $P_i^{nm}(c_i^N|c_i)$ 定义了从真实像素强度到含噪观测的映射（反之亦然），在像素间相互独立：
    $P_i^{nm}(c_i^N|c_i) = \prod_k P_i^{nm}(c_i^N[k]|c_i[k])$

   这种独立性特别适合显微镜数据中的泊松噪声和高斯噪声。网络不再预测高斯分布参数，而是直接预测无噪像素值 $(\hat{c}_1, \hat{c}_2)$，通过噪声模型计算这些预测值产生观测到的含噪数据的概率。噪声模型可以从显微镜的校准数据中测量获得，也可以从训练数据中估计。

3. **校准的不确定性估计（Calibrated Data Uncertainties）**：提供可靠的预测误差估计。

   利用 VSE 网络的变分特性，对每个输入采样 $k=50$ 次预测，计算逐像素标准差 $\sigma_1, \sigma_2$ 作为不确定性估计。通过学习两个标量参数 $\alpha_1, \alpha_2$ 对不确定性进行校准，使得预测的不确定性与实际误差（RMSE）呈近似线性关系。

   校准评估方法：将像素按缩放标准差排序分成 30 个 bin，计算每个 bin 的 RMV（均方根方差）和 RMSE，理想校准时 RMSE ≈ RMV。

### 损失函数 / 训练策略

总损失为修改后的 ELBO：
$$\mathcal{L} = -E_{q(z|x;\phi)}[\log P^{nm}(c_1^N|\hat{c}_1) + \log P^{nm}(c_2^N|\hat{c}_2)] + \sum_i \text{kl}_i$$

其中 KL 项按层级求和（不做空间归一化），噪声模型项鼓励网络预测与含噪观测一致的去噪值。不确定性校准使用验证集学习缩放标量，在测试集上评估。单个 Tesla V100 训练约 1.5 小时。

## 实验关键数据

### 主实验

**BioSR 数据集上四个分解任务（PSNR / MS-SSIM，噪声水平 σ=1，无泊松噪声）:**

| 任务 | 方法 | 训练时间 (h) | PSNR | MS-SSIM |
|------|------|-------------|------|---------|
| T1: ER vs CCPs | μSplit | 7 | 30.3 | 0.853 |
| | HDN⊕μSplit | 11 | 37.3 | 0.982 |
| | Altered μSplit | 1.3 | 38.9 | 0.988 |
| | **DenoiSplit** | **1.5** | **39.7** | **0.989** |
| T3: CCPs vs MT | μSplit | 7.2 | 30.5 | 0.880 |
| | HDN⊕μSplit | 11 | 38.4 | 0.981 |
| | Altered μSplit | 1.4 | 38.9 | 0.985 |
| | **DenoiSplit** | **1.6** | **40.1** | **0.986** |

**高噪声水平（σ=4）下的性能对比:**

| 任务 | μSplit | HDN⊕μSplit | DenoiSplit |
|------|--------|------------|-----------|
| T1: ER vs CCPs | 25.9 / 0.42 | 29.4 / 0.872 | **31.1 / 0.912** |
| T3: CCPs vs MT | 25.6 / 0.46 | 29.3 / 0.844 | **30.6 / 0.872** |
| T4: F-actin vs ER | 22.4 / 0.331 | 25.8 / 0.725 | **26.0 / 0.725** |

**真实噪声数据（Hagen et al. Actin-Mito 数据集）:**

| 方法 | PSNR | MS-SSIM |
|------|------|---------|
| μSplit | 26.5 | 0.872 |
| HDN⊕μSplit | 28.1 | 0.887 |
| Altered μSplit | **31.1** | **0.936** |
| DenoiSplit | 31.0 | 0.935 |

### 消融实验

**KL 损失加权策略消融（T1: ER vs CCPs, σ=1 无泊松噪声）:**

| 配置 | PSNR | MS-SSIM | 说明 |
|------|------|---------|------|
| μSplit (空间归一化 KL) | 30.3 | 0.853 | 噪声严重穿透 |
| Altered μSplit (直接求和 KL) | 38.9 | 0.988 | +8.6 dB，噪声穿透得到抑制 |
| DenoiSplit (直接求和 KL + 噪声模型) | **39.7** | **0.989** | +0.8 dB，噪声模型进一步提升 |

**全噪声水平系统对比（T1，8 种噪声配置）:**

| 方法 | σ=1,λ=0 | σ=2,λ=0 | σ=4,λ=0 | σ=1,λ=1000 | σ=4,λ=1000 |
|------|---------|---------|---------|------------|------------|
| μSplit | 30.3 | 27.4 | 25.9 | 29.4 | 25.9 |
| HDN⊕μSplit | 37.3 | 33.8 | 29.4 | 36.3 | 29.4 |
| **DenoiSplit** | **39.7** | **35.4** | **31.1** | **37.9** | **31.2** |

### 关键发现

1. **KL 加权是最关键的改进**：仅修改 KL 损失的归一化方式（Altered μSplit），PSNR 就提升了 8.6 dB，证明噪声穿透问题的根源在于底层隐空间约束不足。
2. **端到端优于串行**：DenoiSplit (1.5h 训练) 以更少的训练时间（1.5h vs 11h）取得了更好的结果（39.7 vs 37.3 PSNR），说明联合优化去噪和分解比分开做更高效。
3. **高噪声下优势更明显**：σ=4 时 DenoiSplit 比 μSplit 提升 5.2 dB，比 HDN⊕μSplit 提升 1.7 dB。
4. **校准不确定性有效**：RMSE vs RMV 曲线接近 $y=x$，说明模型能可靠地估计预测误差。
5. **真实噪声数据上同样有效**：在 Hagen et al. 数据集上，Altered μSplit 和 DenoiSplit 显著优于两个基线。

## 亮点与洞察

- **优雅的概率框架**：将噪声模型自然地融入变分推理框架，用似然函数替代 MSE 损失，理论上更合理。噪声模型可以针对特定显微镜配置定制，具有很强的适用性。
- **KL 损失归一化的深刻洞察**：μSplit 的空间归一化 KL 让底层隐空间的每个像素 KL 权重很小，导致噪声可以通过底层"泄露"。改为直接求和后，底层的总 KL 权重自然更大（因为像素数更多），更强地约束了底层的信息编码。这一发现具有通用性，适用于所有使用 HVAE 处理含噪数据的场景。
- **不确定性量化的实用价值**：在生物学应用中，知道"这个预测有多可靠"与预测本身同样重要。DenoiSplit 的校准不确定性估计使研究人员能够识别低置信区域，做出更可靠的生物学判断。
- **训练效率**：单模型端到端训练 1.5 小时，远少于 HDN⊕μSplit 的 11 小时。

## 局限与展望

- 目前仅处理两通道分解，扩展到多通道分解需要更多噪声模型
- 噪声模型需要预先获取或估计，对于未知噪声特性的数据可能需要额外工作
- 目前基于合成噪声（人工添加）的评估为主，真实噪声数据的评估较少
- 补充材料中提到存在失败案例，但正文未充分讨论
- 域适应能力有限，跨显微镜/跨模态的迁移需要进一步研究

## 相关工作与启发

- **与 μSplit 的关系**：DenoiSplit 在 μSplit 的 HVAE + Regular-LC 架构基础上做了两个关键改动（KL 加权 + 噪声模型），改动量不大但效果显著
- **与 HDN 的关系**：HDN 的噪声模型思想被更巧妙地集成到分解任务中，实现了"一石二鸟"
- **变分推理的实用化**：展示了 HVAE 的后验采样和不确定性校准在科学计算场景中的实际应用价值

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 首次提出联合图像分解+无监督去噪的方法框架，KL 加权改进洞察深刻
- **实验充分度**: ⭐⭐⭐⭐ — 4 个分解任务 × 8 种噪声水平 + 真实噪声数据 + 不确定性校准评估，非常系统
- **写作质量**: ⭐⭐⭐⭐ — 问题定义清晰，数学推导严谨，图表精美
- **价值**: ⭐⭐⭐⭐ — 对荧光显微镜图像分析有直接实用价值，方法框架可推广到其他图像分解场景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](../../CVPR2025/image_restoration/rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[ECCV 2024\] Pairwise Distance Distillation for Unsupervised Real-World Image Super-Resolution](pairwise_distance_distillation_for_unsupervised_real-world_image_super-resolutio.md)
- [\[ECCV 2024\] Joint RGB-Spectral Decomposition Model Guided Image Enhancement in Mobile Photography](joint_rgb-spectral_decomposition_model_guided_image_enhancement_in_mobile_photog.md)
- [\[ECCV 2024\] TTT-MIM: Test-Time Training with Masked Image Modeling for Denoising Distribution Shifts](ttt-mim_test-time_training_with_masked_image_modeling_for_denoising_distribution.md)
- [\[ECCV 2024\] EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](edformer_transformer-based_event_denoising_across_varied_noise_levels.md)

</div>

<!-- RELATED:END -->
