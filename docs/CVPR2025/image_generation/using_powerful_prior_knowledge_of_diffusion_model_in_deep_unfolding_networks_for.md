---
title: >-
  [论文解读] Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing
description: >-
  [CVPR 2025][图像生成][压缩感知] 将预训练扩散模型的强大先验知识嵌入深度展开网络（DUN），提出 DMP-DUN 方法，仅需 2 步即可实现高质量图像压缩感知重建。 图像压缩感知（Compressive Sensing, CS）通过低于 Nyquist 采样率的测量恢复原始信号，广泛应用于 MRI、快照压缩成像…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "压缩感知"
  - "深度展开网络"
  - "扩散模型"
  - "图像重建"
  - "消息传递"
---

# Using Powerful Prior Knowledge of Diffusion Model in Deep Unfolding Networks for Image Compressive Sensing

**会议**: CVPR 2025  
**arXiv**: [2503.08429](https://arxiv.org/abs/2503.08429)  
**代码**: [GitHub](https://github.com/FengodChen/DMP-DUN-CVPR2025)  
**领域**: 图像生成  
**关键词**: 压缩感知, 深度展开网络, 扩散模型, 图像重建, 消息传递

## 一句话总结

将预训练扩散模型的强大先验知识嵌入深度展开网络（DUN），提出 DMP-DUN 方法，仅需 2 步即可实现高质量图像压缩感知重建。

## 研究背景与动机

图像压缩感知（Compressive Sensing, CS）通过低于 Nyquist 采样率的测量恢复原始信号，广泛应用于 MRI、快照压缩成像等领域。数学上，CS 问题可表示为 $\mathbf{y} = \mathbf{\Phi x} + \epsilon$，其中 $M \ll N$。

深度展开网络（DUN）将传统迭代优化算法展开为神经网络层，兼具传统算法的可解释性与深度学习的高效性。然而 DUN 的重建质量依赖于学到的先验知识质量。

另一方面，预训练扩散模型拥有强大的图像先验知识，但直接用于 CS 重建需要大量迭代步骤（如 DDNM 需要 1000 步），且在低 CS 比率下表现不佳。

**核心动机**：能否将扩散模型的强先验与 DUN 的快速收敛特性结合？即在 DUN 的每个展开层中嵌入扩散模型的一步反向过程，以少量步数实现高质量重建。

## 方法详解

### 整体框架

DMP-DUN 包含三步流程：(1) 设计 Diffusion Message Passing (DMP) 迭代优化算法，将预训练扩散模型嵌入每次迭代中；(2) 将 DMP 算法深度展开为神经网络 DMP-DUN；(3) 通过端到端训练学习时间步和缩放参数，替代手工调参。整体结构中，DMP step 包含梯度下降、ResBlock 和扩散模型去噪三个子模块，首尾各有一个 ResBlock 用于输入映射和通道转换。

### 关键设计1: Diffusion Message Passing (DMP) 算法

**功能**: 在传统 AMP 算法框架中嵌入预训练扩散模型作为去噪器。

**核心思路**: 基于 AMP 的 Onsager 校正项思想，DMP 通过以下迭代公式进行重建：$\mathbf{s}_t = \mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{\Phi}^T (\mathbf{\Phi x}_t - \mathbf{y})$（梯度下降步），$\mathbf{r}_t = \mathcal{D}_t[\mathbf{s}_t + \sqrt{\bar{\alpha}_t} o_t]$（高斯滤波+Onsager校正），$\mathbf{x}_{t-1} = p_\theta(\mathbf{x}_{t-1}|\mathbf{r}_t)$（扩散模型去噪）。其状态演化表明 $\mathbf{r}_t \sim \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}, \sigma_t^2 \mathbf{I})$，保持数据流形约束。

**设计动机**: 传统方法（如 DDNM、MPGD）直接修改扩散过程进行重建，无法保持数据流形，低 CS 比率下偏离严重。DMP 借助 Onsager 项的误差解耦特性，使每次迭代变量始终保持在扩散模型的中间状态流形上，从而提高重建质量。

### 关键设计2: 深度展开与轻量化

**功能**: 将 DMP 展开为可端到端训练的网络，降低计算开销。

**核心思路**: 用轻量卷积残差块（ResBlock）替代两个高成本计算：(1) Onsager 项 $o_t$ 中的散度计算（传统方法需 Monte-Carlo SURE 近似，开销翻倍）；(2) 高斯滤波器 $\mathcal{D}_t$。首部 ResBlock 将输入 $\mathbf{\Phi}^T \mathbf{y}$ 直接映射到扩散模型中间步骤 $\mathbf{x}_K$ 的分布，跳过前 $T-K$ 步反向扩散。尾部 ResBlock 将 RGB 输出转换为单通道图像。

**设计动机**: 原始 DMP 中散度需要额外一次完整扩散模型前向传播（Monte-Carlo 近似），计算量巨大。通过 ResBlock 学习替代散度计算，可将计算量降低数倍。同时首部映射直接跳过大量扩散步骤，使 2-step 重建成为可能。

### 关键设计3: 可学习时间步与缩放参数

**功能**: 自动优化扩散模型中的超参数。

**核心思路**: 传统扩散模型中，时间步 $t$ 和缩放参数 $\bar{\alpha}_t$ 是预定义的超参数。DMP-DUN 将它们作为可训练参数，通过端到端优化自动发现最优调度策略。

**设计动机**: 手动设定时间步调度（如均匀分布）对于少步数重建并非最优。可学习的调度策略允许模型自适应地选择最有效的扩散时间步组合。

### 损失函数

采用标准 MSE 重建损失，在多个 CS 比率（1%, 4%, 10%, 25%, 50%）下联合训练，使单一模型适用于多种采样率场景。

## 实验关键数据

### 主实验结果 (Set11 数据集, PSNR/SSIM)

| 方法 | CS=1% | CS=10% | CS=25% | CS=50% | FLOPs(G) |
|------|-------|--------|--------|--------|----------|
| ISTA-Net+ (CVPR18) | 17.45/0.413 | 26.49/0.804 | 32.44/0.924 | 38.08/0.968 | 56.2 |
| OCTUF (CVPR23) | 21.75/0.593 | 30.70/0.903 | 36.10/0.960 | 41.34/0.984 | 189.3 |
| CPP-Net (CVPR24) | 22.19/0.614 | 31.27/0.914 | 36.35/0.963 | 41.39/0.983 | 153.5 |
| DDNM (1000步) | 17.95/0.450 | 25.78/0.815 | 27.80/0.893 | 29.01/0.941 | 67039 |
| **DMP-DUN (2步)** | **23.18/0.629** | **32.63/0.921** | **37.58/0.965** | **42.06/0.984** | **157.0** |
| **DMP-DUN (4步)** | **23.32/0.631** | **33.22/0.928** | **38.29/0.968** | **42.82/0.985** | **303.4** |

### 消融实验

| 配置 | Set11 Avg PSNR | Urban100 Avg PSNR |
|------|---------------|-------------------|
| 无展开(直接DMP 10步) | 32.99 | — |
| DMP-DUN (2步+ResBlock) | 32.74 | — |
| DMP-DUN (4步+ResBlock) | 33.26 | — |

### 关键发现

1. **极少步数即可达到 SOTA**: DMP-DUN 仅需 2 步即超越所有传统 DUN 方法和 1000 步 DDNM，PSNR 在 Set11 上高出 8 dB 以上。
2. **FLOPs 大幅降低**: 相比 DDNM 的 67039G FLOPs，2 步 DMP-DUN 仅需 157G（降低 427 倍），4 步版本需 303G。
3. **低 CS 比率优势明显**: 在 CS=1% 极端情况下，DMP-DUN 比次优方法高出约 1 dB，表明扩散先验对严重欠采样场景尤为有效。
4. **流形保持的重要性**: 状态演化分析证明 DMP 在整个重建过程中保持数据流形，这是其优于其他扩散重建方法的关键。

## 亮点与洞察

- **理论优雅**: 从 AMP 的消息传递框架出发，自然地将扩散模型嵌入迭代过程，理论推导完整。
- **效率与质量兼得**: 通过深度展开+轻量 ResBlock 替代高成本计算，解决了扩散模型重建慢的核心瓶颈。
- **流形约束视角新颖**: 通过状态演化方程证明 DMP 变量始终符合扩散中间状态分布，为理解扩散式重建提供了新视角。

## 局限与展望

- 依赖预训练扩散模型的质量，若扩散模型先验与目标数据域不匹配，效果可能下降。
- 仅在 block-based CS 上验证，未扩展到 MRI、SCI 等具体应用场景。
- ResBlock 替代散度近似虽然高效，但丧失了 Onsager 校正的理论保证，实际恢复的流形约束可能弱于理论分析。
- 未来可探索将方法推广到其他逆问题（去模糊、超分辨等）。

## 相关工作与启发

- **AMP 与 DUN 的结合**: 本文延续了 AMP-Net、ISTA-Net+ 等经典工作的思路，但首次引入预训练生成模型的先验。
- **扩散模型用于逆问题**: DDNM、MPGD 等直接修改扩散过程的方法缺乏流形保持，本文通过 AMP 框架优雅地解决了这一问题。
- **启发**: 将强预训练先验与传统优化框架结合的范式值得推广到其他信号处理任务。

## 评分

⭐⭐⭐⭐ — 理论完整、实验效果显著，将扩散模型与深度展开网络结合是一个自然且优雅的设计。2 步即达 SOTA 的效率提升令人印象深刻。但应用场景的扩展性有待进一步验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] DDIS: When Model Knowledge Meets Diffusion Model](../../ICML2025/image_generation/when_model_knowledge_meets_diffusion_model_diffusion-assisted_data-free_image_synthesis.md)
- [\[ICCV 2025\] DIIP: Diffusion Image Prior](../../ICCV2025/image_generation/diffusion_image_prior.md)
- [\[ICCV 2025\] Learning Deblurring Texture Prior from Unpaired Data with Diffusion Model](../../ICCV2025/image_generation/learning_deblurring_texture_prior_from_unpaired_data_with_diffusion_model.md)
- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[CVPR 2025\] SIR-DIFF: Sparse Image Sets Restoration with Multi-View Diffusion Model](sir-diff_sparse_image_sets_restoration_with_multi-view_diffusion_model.md)

</div>

<!-- RELATED:END -->
