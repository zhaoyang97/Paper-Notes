---
title: >-
  [论文解读] Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape
description: >-
  [ICML2025][图像生成][LoRA] 提出 Flat-LoRA，通过在全参数空间：中引入基于贝叶斯期望损失的随机权重扰动，使 LoRA 收敛到全参数空间中更平坦的极小值区域，提升域内和域外泛化性能，且几乎不增加训练时间和显存开销。 LoRA 将微调限制在低秩矩阵空间 $\mathcal{M}_r$ 内。训练完成后…
tags:
  - "ICML2025"
  - "图像生成"
  - "LoRA"
  - "参数高效微调"
  - "平坦极小值"
  - "随机权重扰动"
  - "贝叶斯期望损失"
---

# Flat-LoRA: Low-Rank Adaptation over a Flat Loss Landscape

**会议**: ICML2025  
**arXiv**: [2409.14396](https://arxiv.org/abs/2409.14396)  
**代码**: [nblt/Flat-LoRA](https://github.com/nblt/Flat-LoRA)  
**领域**: 图像生成  
**关键词**: LoRA, 参数高效微调, 平坦极小值, 随机权重扰动, 贝叶斯期望损失

## 一句话总结

提出 Flat-LoRA，通过在**全参数空间**中引入基于贝叶斯期望损失的随机权重扰动，使 LoRA 收敛到全参数空间中更平坦的极小值区域，提升域内和域外泛化性能，且几乎不增加训练时间和显存开销。

## 研究背景与动机

LoRA 将微调限制在低秩矩阵空间 $\mathcal{M}_r$ 内。训练完成后，低秩适应 $\Delta W = BA$ 会合并到预训练权重 $W$ 中用于推理。现有改进（AdaLoRA、DoRA、PiSSA 等）关注的都是 LoRA 参数空间内部的优化质量，**忽视了 LoRA 空间与全参数空间之间的关系**。

核心观察：在 LoRA 参数空间中看起来平坦的极小值，在全参数空间中可能存在尖锐方向（Figure 1）。尖锐极小值会损害泛化能力，尤其是面对分布偏移时。

一个自然的想法是将 SAM（Sharpness-Aware Minimization）与 LoRA 结合（LoRA-SAM），但存在三个问题：

**优化方向受限**：LoRA-SAM 仅在 $A$ 的列空间内优化 sharpness，覆盖不到全参数空间

**训练成本翻倍**：SAM 需要额外梯度步，对大模型不友好

**需存储全参数扰动**：违背参数高效微调的初衷

## 方法详解

### 1. 全参数空间的平坦性目标

理想目标是直接在全参数空间做 SAM：

$$\min_{A,B} \max_{\|\varepsilon_W\|_F \leq \rho} L(W + BA + \varepsilon_W)$$

其中 $\varepsilon_W \in \mathbb{R}^{m \times n}$ 是全参数空间的对抗扰动。但直接求解需额外梯度步且要存储全量扰动。

### 2. 贝叶斯期望损失松弛

将 max 松弛为期望，得到 Flat-LoRA 的核心目标函数：

$$\min_{A,B} \mathbb{E}_{(\varepsilon_W)_{i,j} \sim \mathcal{N}(0, \sigma^2)} L(W + BA + \varepsilon_W)$$

由 Lemma 3.1：若 $L(W)$ 是 $\alpha$-Lipschitz 且 $\beta$-smooth 的，则期望损失函数是 $\min\{\alpha/\sigma, \beta\}$-smooth 的——即噪声方差 $\sigma$ 越大，损失面越光滑，促使优化收敛到更平坦区域。

实际操作：每步训练采样一个噪声矩阵 $\varepsilon_W$，计算扰动后的梯度来更新 $A, B$。**无需额外梯度步**，训练时间几乎不变。

### 3. 精细化随机扰动生成策略

扰动并非简单的 i.i.d. 高斯噪声，而是考虑两个因素：

- **Filter 结构**：按行（filter）生成噪声，范数大的 filter 给更大扰动
- **输入维度缩放**：用 $1/n$ 缩放方差，确保前向传播中扰动引入的方差与输入维度无关

最终扰动生成公式：

$$(\varepsilon_W)_{i,j} \sim \mathcal{N}\left(0, \frac{\sigma^2}{n} \|W'_{i,:}\|_2^2\right)$$

其中 $W' = W + BA$ 是合并后权重。由 Proposition 3.2，这种设计使输出方差增大 $1+\sigma^2$ 倍，与输入维度 $n$ 无关。

### 4. 基于随机种子的显存优化

全参数扰动 $\varepsilon_W$ 本身很大，但实际只需存储：
- 随机种子（一个整数）
- 每个 filter 的范数 $\{\|W'_{i,:}\|_2^2\}$（$m$ 个标量，不到 LoRA 参数的 $1/r$）

训练时按种子重新生成噪声，反向传播后用种子复原并移除扰动，显存开销极小。

### 5. 渐进式扰动增强

实践中建议逐步增大扰动强度 $\sigma$：从 0 线性增长到目标值，使模型先收敛到较好区域再逐步平滑化。

### 与 LoRA-SAM 的理论对比

LoRA-SAM 在全参数空间的等效扰动为：

$$\varepsilon_W \approx c \, (\nabla_W L) A^\top A$$

（当 $B$ 较小时）——只覆盖 $A$ 的列空间，是全参数空间的极小子空间。Flat-LoRA 的随机扰动覆盖全空间所有方向。

## 实验关键数据

### NLP：T5-base 微调 GLUE 子集（r=8/16）

| 方法 | MNLI | SST2 | CoLA | QNLI | MRPC | Avg |
|------|------|------|------|------|------|-----|
| Full FT | 86.19 | 94.15 | 82.84 | 93.10 | 89.22 | 89.10 |
| LoRA (r=8) | 86.24 | 94.25 | 82.87 | 93.06 | 88.56 | 88.99 |
| **Flat-LoRA (r=8)** | 86.20 | **94.75** | **83.61** | 93.16 | **89.59** | **89.47** |
| LoRA (r=16) | 86.49 | 94.52 | 82.89 | 92.97 | 88.89 | 89.15 |
| **Flat-LoRA (r=16)** | 86.51 | **94.84** | **84.08** | **93.28** | **89.83** | **89.72** |

Flat-LoRA r=16 平均比 LoRA 高 **+0.57**，且在 CoLA（+1.19）和 MRPC（+0.94）改进尤为明显。

### CV：CLIP ViT-B/32 微调图像分类

| 方法 | CIFAR-10 | CIFAR-100 | Cars | SVHN | DTD | Avg |
|------|----------|-----------|------|------|-----|-----|
| LoRA (r=8) | 97.90 | 87.74 | 73.22 | 97.49 | 76.86 | 86.64 |
| **Flat-LoRA (r=8)** | **98.09** | **88.64** | **74.17** | **97.59** | **77.51** | **87.20** |
| LoRA (r=16) | 97.99 | 88.12 | 73.80 | 97.56 | 77.34 | 86.92 |
| **Flat-LoRA (r=16)** | **98.21** | **89.27** | **74.89** | **97.71** | **78.24** | **87.66** |

CV 上 r=16 平均提升 **+0.74**，Cars 数据集提升超过 1 个百分点。

### 其他任务

论文还覆盖了数学推理、代码生成、对话、指令跟随和文生图等任务，均观察到一致的提升，说明方法的通用性。

### 与其他 LoRA 变体的正交性

Flat-LoRA 可以即插即用地与 DoRA、PiSSA、LoRA-GA 等结合，在这些变体基础上进一步带来增益。

## 亮点与洞察

1. **问题洞察深刻**：从 LoRA 空间 vs 全参数空间的 landscape 差异切入，揭示了一个被广泛忽视但重要的问题
2. **方法极简高效**：用贝叶斯期望损失替代 SAM 的 min-max，避免了额外梯度步；用随机种子存储避免了显存膨胀——工程实现优雅
3. **理论推导清晰**：从 LoRA-SAM 的等效扰动推导出其局限性，再到 filter-wise 扰动的方差分析，环环相扣
4. **通用性强**：NLP + CV + 生成，跨模态一致有效

## 局限与展望

1. **超参数 $\sigma$ 的选取**：虽然建议渐进式增强，但最优增长策略和目标值仍需逐任务调参
2. **理论保证有限**：期望损失仅是 min-max 的松弛，不是等价替代；平坦性与泛化的理论联系本身也存争议（Dinh et al., 2017 的反例）
3. **单次噪声采样**：每步仅采样一个 $\varepsilon_W$，方差较大，理论上多次采样平均更好但成本更高
4. **未与最新方法充分对比**：如 LoRA-Pro（梯度对齐）等同期工作未做直接实验对比
5. **仅验证了 LoRA 框架**：是否能推广到 Adapter、Prefix-Tuning 等其他 PEFT 方法未探讨

## 相关工作与启发

- **SAM (Foret et al., 2020)**：平坦极小值的经典 min-max 方法，但训练成本翻倍
- **RWP (Bisla et al., 2022)**：随机权重扰动求平坦极小值的先驱，Flat-LoRA 将其适配到 PEFT 场景
- **DoRA / PiSSA / LoRA-GA**：正交的 LoRA 改进方向，可与 Flat-LoRA 组合使用
- **NEFTune (Jain et al., 2024)**：在嵌入层加噪声提升微调，与 Flat-LoRA 在权重空间加噪声的思路可类比

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 创新性 | 4 | 问题视角新颖，方法本身是已有技术的工程化组合 |
| 理论深度 | 3.5 | 推导清晰但不算深刻，核心定理引用自先前工作 |
| 实验充分性 | 4.5 | NLP/CV/生成多场景覆盖，消融实验齐全 |
| 实用性 | 5 | 即插即用，几乎零额外成本，代码开源 |
| 写作质量 | 4 | 结构清晰，Figure 1 直观易懂 |
| **总分** | **4.2** | 高质量工作，实用价值突出 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models](intlora_integral_low-rank_adaptation_of_quantized_diffusion_models.md)
- [\[NeurIPS 2025\] GraLoRA: Granular Low-Rank Adaptation for Parameter-Efficient Fine-Tuning](../../NeurIPS2025/image_generation/gralora_granular_low-rank_adaptation_for_parameter-efficient_fine-tuning.md)
- [\[ICCV 2025\] Transformed Low-rank Adaptation via Tensor Decomposition and Its Applications to Text-to-image Models](../../ICCV2025/image_generation/transformed_low-rank_adaptation_via_tensor_decomposition_and_its_applications_to.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](../../NeurIPS2025/image_generation/stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[CVPR 2026\] CRAFT-LoRA: Content-Style Personalization via Rank-Constrained Adaptation and Training-Free Fusion](../../CVPR2026/image_generation/craft-lora_content-style_personalization_via_rank-constrained_adaptation_and_tra.md)

</div>

<!-- RELATED:END -->
