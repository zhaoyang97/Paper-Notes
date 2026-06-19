---
title: >-
  [论文解读] Detecting Adversarial Data Using Perturbation Forgery
description: >-
  [CVPR 2025][图像生成][对抗检测] 通过建模对抗噪声的高斯分布并证明其近邻性，提出 Perturbation Forgery 方法在训练时持续扰动噪声分布形成开覆盖，配合稀疏掩码生成伪对抗数据训练二分类器，仅需 FGSM 一种攻击的噪声分布就能泛化检测梯度、GAN、扩散和物理等各类未见攻击，AUROC 达 0.99+ 且推理开销极低。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "对抗检测"
  - "扰动伪造"
  - "噪声分布"
  - "开覆盖"
  - "泛化检测"
---

# Detecting Adversarial Data Using Perturbation Forgery

**会议**: CVPR 2025  
**arXiv**: [2405.16226](https://arxiv.org/abs/2405.16226)  
**代码**: [https://github.com/cc13qq/PFD](https://github.com/cc13qq/PFD)  
**领域**: 图像生成  
**关键词**: 对抗检测、扰动伪造、噪声分布、开覆盖、泛化检测

## 一句话总结
通过建模对抗噪声的高斯分布并证明其近邻性，提出 Perturbation Forgery 方法在训练时持续扰动噪声分布形成开覆盖，配合稀疏掩码生成伪对抗数据训练二分类器，仅需 FGSM 一种攻击的噪声分布就能泛化检测梯度、GAN、扩散和物理等各类未见攻击，AUROC 达 0.99+ 且推理开销极低。

## 研究背景与动机

**领域现状**：对抗检测是防御对抗攻击的一种策略，通过识别自然数据和对抗数据之间的分布差异来过滤对抗样本。现有方法主要针对梯度类攻击（如 PGD、FGSM）设计检测器。

**现有痛点**：(1) 现有检测器泛化性差，只能检测训练时见过的攻击类型，对未见攻击失效。(2) 基于生成模型的新型对抗攻击（GAN-based、Diffusion-based）产生不均匀和各向异性的扰动，现有方法难以检测。(3) 部分高性能方法（如 EPSAD）推理开销极大，处理 100 张 ImageNet 图片需要近 400 秒，不实用。

**核心矛盾**：需要一个低开销、能同时泛化到梯度类和生成类对抗攻击的检测器，但现有方法要么泛化差、要么开销大。

**本文目标** 设计一个模型无关的、低推理开销的对抗检测器，能泛化到各类未见攻击，包括梯度类、GAN 类、扩散类和物理攻击。

**切入角度**：作者从数学角度出发，通过将对抗噪声建模为截断高斯分布，发现不同攻击的噪声分布之间存在近邻关系（Wasserstein 距离有界），进而推导出所有对抗噪声分布可以被一个"开覆盖"包含。

**核心 idea**：通过持续扰动已知攻击的噪声分布来形成开覆盖，让检测器在开覆盖上训练即可泛化检测所有类型的未见攻击。

## 方法详解

### 整体框架
Perturbation Forgery 的整体 pipeline 如下：输入自然图像 → 首先用常见攻击（FGSM）估计对抗噪声的高斯分布参数 → 每个训练 batch 对分布参数随机扰动生成新的近邻分布 → 从扰动分布中采样噪声 → 用显著性检测和 GradCAM 生成稀疏掩码 → 将全局噪声转为局部噪声后注入一半自然数据生成伪对抗数据 → 伪对抗数据 + 干净数据训练二分类检测器。推理时直接用训练好的检测器做二分类判断输入是否为对抗样本。

### 关键设计

1. **噪声分布扰动 (Noise Distribution Perturbation)**:

    - 功能：通过连续扰动已知攻击的噪声分布参数，生成大量近邻噪声分布，形成对抗噪声分布的开覆盖
    - 核心思路：先用 FGSM 攻击生成对抗噪声，将噪声展平后估计多元高斯分布的均值 $\hat\mu$ 和协方差 $\hat\Sigma$。每个 batch 训练时，向均值加随机扰动 $\hat\mu_i = \hat\mu + \alpha_i \cdot m_i$（$m_i \sim \mathcal{N}(0, I)$, $\alpha_i \sim U(-\epsilon_\mu, \epsilon_\mu)$），协方差类似处理。经过 n 个 batch，得到近似开覆盖的分布集合
    - 设计动机：作者通过 Theorem 1 证明在同一 $\ell^p$ 范数约束下所有对抗噪声分布互为近邻分布，进而存在开覆盖。通过在开覆盖上训练，检测器学会区分自然数据和对抗数据的分布，从而泛化到未见攻击

2. **稀疏掩码生成 (Sparse Mask Generation)**:

    - 功能：将全局噪声转换为局部稀疏噪声，模拟生成类攻击的不均匀扰动模式
    - 核心思路：对自然样本用显著性检测模型和 GradCAM 生成注意力图，取两者的并集标识高频/显著区域（$\text{Mask}_1$）；再用 Sobel 算子提取梯度图中的高频点进行稀疏化（$\text{Mask}_2$）；最终掩码为两者交集 $\text{Mask} = \text{Mask}_1 \cap \text{Mask}_2$
    - 设计动机：生成类攻击（如 Diff-PGD）的扰动集中在高频和显著区域、在低频和背景区域很弱。物理攻击也倾向于在局部区域添加 patch。稀疏掩码让检测器学会关注这种局部噪声模式，弥补了纯分布扰动只能模拟全局噪声的不足

3. **伪对抗数据生成 (Pseudo-Adversarial Data Production)**:

    - 功能：将采样的噪声与稀疏掩码结合，注入自然样本生成伪对抗数据用于训练
    - 核心思路：从扰动后的分布采样噪声 $\hat\eta_{i,k}$，通过概率密度阈值 $\gamma_p$ 排除低似然样本，然后与掩码做 Hadamard 乘积得到局部噪声，加到自然图像上：$\hat{x}_{i,k} = x_{i,k} + \hat\eta_{i,k} \otimes \text{Mask}(x_{i,k})$
    - 设计动机：用伪对抗数据替换真实对抗数据训练检测器，避免需要真实攻击数据、实现模型无关的检测

### 损失函数 / 训练策略
每个 batch 将一半自然数据转换为伪对抗数据，与另一半干净数据组成训练集，使用标准交叉熵损失训练二分类检测器。整个训练过程中每个 batch 都随机扰动噪声分布参数，确保覆盖足够多的分布变体。

## 实验关键数据

### 主实验

| 数据集/攻击 | 指标 | 本文 PFD | EPSAD | SPAD | 说明 |
|------------|------|---------|-------|------|------|
| ImageNet100 梯度攻击 (8种) | AUROC 平均 | 0.992 | 0.996 | 0.983 | 接近 EPSAD 但推理快 80× |
| ImageNet100 生成攻击 (5种) | AUROC 平均 | 0.947 | 0.472 | 0.903 | 大幅超越 EPSAD |
| 人脸物理攻击 (4种) | AUROC 平均 | 0.992 | 0.959 | 0.983 | 全面领先 |
| 推理时间 (100张) | 秒 | 4.85 | 396.81 | 4.56 | 比 EPSAD 快 ~82× |

### 消融实验

| 配置 | 梯度攻击 AUROC | 生成攻击 AUROC | 说明 |
|------|---------------|---------------|------|
| Full model | 0.992 | 0.947 | 完整模型 |
| w/o 噪声分布扰动 | 0.971 | 0.867 | 去掉分布扰动，泛化性大幅下降 |
| w/o 稀疏掩码 | 0.990 | 0.891 | 去掉掩码后生成攻击检测下降明显 |
| w/o 概率密度过滤 | 0.988 | 0.932 | 过滤低似然噪声有一定作用 |

### 关键发现
- 噪声分布扰动是泛化性的核心贡献者，去掉后生成类攻击检测下降约 8 个点
- 稀疏掩码对生成类攻击检测贡献最大（+5.6 AUROC），因为它模拟了生成攻击的局部扰动特性
- EPSAD 在生成类攻击上几乎失效（AUROC 仅 0.47），而本文方法保持 0.94+，说明基于扩散模型辅助的检测方案对生成类攻击天然不利
- 本文方法在人脸数据集上的物理攻击检测也表现优异（AUROC 0.99+），说明稀疏掩码能有效模拟 patch 类物理攻击

## 亮点与洞察
- **理论驱动的方法设计**：通过将对抗噪声建模为高斯分布并用 Wasserstein 距离证明近邻性，提供了扎实的数学基础，不是纯经验方法。这种"先建模噪声分布、再利用拓扑性质"的思路可以迁移到其他需要泛化未见分布的检测任务
- **伪造替代真实**：用伪对抗数据完全替代真实对抗样本训练检测器，实现模型无关且攻击类型无关，这比传统需要特定攻击算法生成训练数据的方式更实用
- **推理速度与性能的平衡**：在保持 AUROC 0.99+ 的同时推理时间仅 4.85 秒/100张，比 EPSAD 快 82 倍

## 局限与展望
- 理论假设中将对抗噪声建模为各向同性高斯分布（$\sigma^2 I_d$），但真实对抗噪声通常是各向异性的，这个假设的松弛程度值得进一步分析
- 稀疏掩码生成依赖预训练的显著性检测模型和 GradCAM，引入了额外的计算和模型依赖
- 实验主要在 ImageNet100 和人脸数据集上验证，缺少在更大规模数据集（如完整 ImageNet）上的评估
- 检测器本身是一个额外的二分类模型，在真实部署中增加了系统复杂度

## 相关工作与启发
- **vs EPSAD**: EPSAD 利用扩散模型增强检测，达到梯度攻击 0.999 AUROC，但推理慢 80× 且生成攻击检测失效（0.15-0.47）。PFD 通过伪造训练数据避免了推理时的扩散过程
- **vs SPAD**: SPAD 用手工设计的伪噪声做数据增强，但缺乏理论指导，生成攻击检测（0.90）弱于本文（0.95）
- **vs LID/LiBRe**: 这些方法依赖目标模型内部特征，非模型无关，且泛化性不足

## 评分
- 新颖性: ⭐⭐⭐⭐ 理论推导新颖，但"伪造训练数据"的思路在对抗检测中不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖梯度/GAN/扩散/物理4类攻击，多数据集验证，消融完整
- 写作质量: ⭐⭐⭐⭐ 理论部分清晰，但符号较多，需要一定数学背景
- 价值: ⭐⭐⭐⭐ 实用性强，推理快且泛化好，适合实际部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection](../../ICCV2025/image_generation/forgelens_data-efficient_forgery_focus_for_generalizable_forgery_image_detection.md)
- [\[CVPR 2025\] Instant Adversarial Purification with Adversarial Consistency Distillation](instant_adversarial_purification_with_adversarial_consistency_distillation.md)
- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[CVPR 2025\] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification](divide_and_conquer_heterogeneous_noise_integration_for_diffusion-based_adversari.md)
- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](../../NeurIPS2025/image_generation/detecting_generated_images_by_fitting_natural_image_distributions.md)

</div>

<!-- RELATED:END -->
