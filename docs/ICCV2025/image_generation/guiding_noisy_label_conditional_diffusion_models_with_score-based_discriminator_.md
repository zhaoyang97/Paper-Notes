---
title: >-
  [论文解读] Guiding Noisy Label Conditional Diffusion Models with Score-based Discriminator Correction
description: >-
  [ICCV 2025][图像生成][噪声标签] 提出Score-based Discriminator Correction (SBDC)，通过训练一个轻量判别器在推理时校正噪声标签条件扩散模型的生成轨迹，利用噪声检测将训练集分为干净/腐败子集来训练判别器，并发现仅在采样过程的早中期阶段施加引导即可获得最优效果。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "噪声标签"
  - "判别器引导"
  - "推理时校正"
  - "条件扩散模型"
  - "Score-based Correction"
---

# Guiding Noisy Label Conditional Diffusion Models with Score-based Discriminator Correction

**会议**: ICCV 2025  
**arXiv**: [2508.19581](https://arxiv.org/abs/2508.19581)  
**代码**: 无  
**领域**: 扩散模型/图像生成  
**关键词**: 噪声标签, 判别器引导, 推理时校正, 条件扩散模型, Score-based Correction

## 一句话总结

提出Score-based Discriminator Correction (SBDC)，通过训练一个轻量判别器在推理时校正噪声标签条件扩散模型的生成轨迹，利用噪声检测将训练集分为干净/腐败子集来训练判别器，并发现仅在采样过程的早中期阶段施加引导即可获得最优效果。

## 研究背景与动机

扩散模型依赖大规模数据集实现高质量图像生成，但这些数据集常包含标签错误（mislabeled data）。例如ImageNet存在约6%的标签噪声，LAION-5B等多模态数据集的噪声比例更高。当条件生成模型在这些噪声数据上训练时，会学到噪声分布 $p(X|\tilde{Y})$ 而非真实分布 $p(X|Y)$，导致生成图像质量下降、与条件不一致。

现有解决方案面临几个关键问题：

**重训练成本过高**：基于转移矩阵估计的方法（如TDSM）需要多阶段训练，错误会在阶段间传播；噪声检测方法需要清洗数据后重训练模型，对于大规模模型不切实际

**直接方法的缺陷**：TDSM在高噪声率下严重失效，经常生成标签错误的图像

**推理效率问题**：需要一种计算开销小、无需重训练生成模型的方案

SBDC的核心动机是：**能否在推理时通过一个轻量的辅助信号来校正噪声模型的生成轨迹？** 论文借鉴了噪声对比估计（NCE）的思想，利用现有的噪声检测技术和判别器训练来实现这一目标。

## 方法详解

### 整体框架

SBDC的流程分为两步：
1. **训练阶段**：利用噪声检测方法将数据集分为伪干净集 $\mathcal{D}_r$ 和腐败集 $\mathcal{D}_f$，训练一个时间依赖的判别器 $D_\phi^t$
2. **推理阶段**：将判别器的梯度信号作为引导信号注入到预训练扩散模型的采样过程中，仅在采样的早中期阶段（通过γ-gate控制）施加引导

### 关键设计

1. **噪声条件生成行为分析**：论文将采样过程分为三个阶段：

    - **Phase I（边缘化阶段）**：$t$ 较大时条件信息被遗忘，$C(t)$（置信度）低，$I(t)$（不稳定性）低
    - **Phase II（条件化阶段）**：多个模式竞争影响扰动身份，$I(t)$ 达到峰值，**这是类别不稳定最严重的阶段**
    - **Phase III（精细化阶段）**：后验集中在单一目标上，$C(t)$ 仅反映噪声率

   量化衡量指标为：

    $C(t) = \mathbb{P}[f(\mathbf{x}_\theta(\mathbf{x}_t, \mathbf{y})) = \mathbf{y}]$

    $I(t) = \mathbb{P}[f(\mathbf{x}_\theta(\mathbf{x}_t, \mathbf{y})) \neq f(\mathbf{x}_\theta(\mathbf{x}_{t-1}, \mathbf{y}))]$

   核心洞察：**Phase II的类别变化一旦发生，错误会持续到最终输出**，因此校正应集中在此阶段。

2. **Score-based判别器校正**：假设score网络已完美学到噪声分布，通过以下公式恢复干净分布：

    $\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t|\mathbf{y}) = \nabla_{\mathbf{x}_t} \log p_\theta(\mathbf{x}_t|\tilde{\mathbf{y}}) + \nabla_{\mathbf{x}_t} \log \frac{p(\mathbf{x}_t|\mathbf{y})}{p_\theta(\mathbf{x}_t|\tilde{\mathbf{y}})}$

   右侧第二项（校正项）通过判别器的对数似然比来近似。论文证明了理论界（Theorem 1）：当噪声率不过高时，最优判别器的梯度可以有效估计真实的对数似然比。

3. **γ-gate机制**：基于Phase分析，仅在采样过程的特定区间（γ-gate）内施加判别器引导，避免在Phase I（条件无效）和Phase III（已收敛）浪费计算。实验显示这种限制性引导反而提升了整体性能。

4. **SiMix数据增强**：为缓解判别器的过拟合问题，提出Similarity-based Mixup（SiMix）——在特征空间中找到最近邻样本进行混合，而非随机配对。具体地，对每个样本在batch内找到编码距离最近的样本，按Beta分布采样系数进行线性插值：

    $\mathbf{z}_i \leftarrow \lambda_i \mathbf{z}_i + (1 - \lambda_i) \mathbf{z}_{\arg\min_{j} \|f_i - f_j\|_2}$

### 损失函数 / 训练策略

判别器训练使用时间加权的二元交叉熵损失：

$$\mathcal{L}_{adv} = \mathbb{E}_{t, (\mathbf{x}, \mathbf{y}) \sim p_r, \mathbf{x}_t}[-\log D_\theta^t(\mathbf{x}_t, \mathbf{y})] + \mathbb{E}_{t, (\mathbf{x}, \mathbf{y}) \sim p_f, \mathbf{x}_t}[-\log(1 - D_\theta^t(\mathbf{x}_t, \mathbf{y}))]$$

此外还引入Pseudo-clean Shuffle：从干净集中随机采样一部分样本，将其标签替换为腐败标签并标记为负样本，增加判别器对标签-图像一致性的敏感度。

## 实验关键数据

### 主实验

CIFAR-10上不同噪声设置的生成质量（EDM为基础扩散模型）：

| 噪声类型 | 噪声率 | 方法 | FID ↓ | IS ↑ | CW-FID ↓ | CW-Den. ↑ |
|----------|--------|------|-------|------|----------|-----------|
| Symmetric | 20% | EDM | 1.96 | 9.95 | 11.3 | 98.6 |
| Symmetric | 20% | TDSM | 2.36 | 10.04 | 10.9 | 113.1 |
| Symmetric | 20% | **SBDC** | **2.49** | **10.06** | **10.6** | **114.8** |
| Symmetric | 50% | EDM | 2.07 | 9.69 | 38.6 | 66.8 |
| Symmetric | 50% | TDSM | 2.43 | 9.84 | 18.2 | 95.8 |
| Symmetric | 50% | **SBDC** | 2.24 | **9.87** | **15.6** | **98.1** |
| Symmetric | 80% | EDM | 2.15 | 9.67 | 71.7 | 43.0 |
| Symmetric | 80% | TDSM | 2.25 | 9.76 | 59.8 | 52.0 |
| Symmetric | 80% | **SBDC** | **2.30** | **9.71** | **48.2** | **58.0** |

SBDC在类条件指标（CW-FID、CW-Density、CW-Coverage）上全面超越TDSM，尤其在高噪声率（80%）下优势显著。

### 消融实验

| 配置 | 关键效果 | 说明 |
|------|---------|------|
| 无γ-gate（全程引导） | 性能下降 | 在Phase I和III施加引导反而有害 |
| 无SiMix | 判别器过拟合 | 特别是在小数据集上效果明显下降 |
| 无Pseudo-clean Shuffle | CW指标下降 | 降低了判别器对标签一致性的感知能力 |
| 不同γ范围 | Phase II最优 | 验证了三阶段分析的正确性 |
| 不同引导权重 $w$ | 过大导致失真 | 适中权重平衡校正与生成质量 |

### 关键发现

- SBDC的优势在高噪声率下尤为突出，80%噪声率时CW-FID从TDSM的59.8提升到48.2
- TDSM在高噪声下经常生成标签错误的图像，而SBDC能有效校正
- Instance noise（非对称噪声）比symmetric noise更具挑战性，但SBDC仍保持优势
- 判别器训练快速（相比重训扩散模型极大节省计算），推理时额外开销极小

## 亮点与洞察

- 对条件扩散模型在噪声标签下的行为进行了精细的三阶段分析，发现Phase II是干预的最佳窗口
- 巧妙利用现有噪声检测方法来构建判别器训练数据，避免了额外标注开销
- γ-gate机制的设计既提升了效果又减少了推理开销，是"少即是多"原则的很好体现
- SiMix是一种新颖的Mixup变体，基于特征相似性而非随机配对，对判别器泛化有帮助

## 局限与展望

- 仅在CIFAR-10等小尺度数据集上验证，缺乏大规模文本到图像模型（如Stable Diffusion）的实验
- 噪声检测本身的准确率会影响最终效果，在极高噪声率下可能降级
- 理论界依赖Lipschitz条件和最优判别器假设，实际中可能不完全满足
- 未探索多模态条件（如文本-图像联合）场景下的噪声校正

## 相关工作与启发

- 与Discriminator Guidance的区别：DG用于缩小真实/合成数据差异，SBDC专门针对干净/腐败标签差异
- TDSM通过转移矩阵修改训练目标，而SBDC完全在推理时工作，更具实用性
- 噪声检测文献（如SIMIFEAT、SOP等）为判别器训练提供了可靠的伪标签来源

## 评分

- **新颖性**: ⭐⭐⭐⭐ 将判别器引导与噪声标签校正结合的思路新颖，三阶段分析有洞察
- **实验充分度**: ⭐⭐⭐ 覆盖多种噪声设置但局限于小数据集
- **写作质量**: ⭐⭐⭐⭐ 理论推导清晰，实验组织合理
- **价值**: ⭐⭐⭐⭐ 解决了一个实际且重要的问题，推理时校正方案具有很好的实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] UniCombine: Unified Multi-Conditional Combination with Diffusion Transformer](unicombine_unified_multi-conditional_combination_with_diffusion_transformer.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](../../NeurIPS2025/image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[CVPR 2026\] Guiding Token-Sparse Diffusion Models](../../CVPR2026/image_generation/guiding_token-sparse_diffusion_models.md)
- [\[CVPR 2026\] Guiding Diffusion Models with Semantically Degraded Conditions](../../CVPR2026/image_generation/guiding_diffusion_models_with_semantically_degraded_conditions.md)
- [\[ICCV 2025\] Balanced Image Stylization with Style Matching Score](balanced_image_stylization_with_style_matching_score.md)

</div>

<!-- RELATED:END -->
