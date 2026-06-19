---
title: >-
  [论文解读] Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification
description: >-
  [ICCV 2025][图像生成][Protective Perturbation] 提出 AntiPure，一种针对扩散模型净化（purification）过程的对抗扰动方法，通过 Patch-wise Frequency Guidance 和 Erroneous Timestep Guidance 两种引导机制，生成在净化后仍能持续干扰定制化微调的保护性扰动，在"净化-定制化"工作流中全面超越现有防护方法。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Protective Perturbation"
  - "Anti-Purification"
  - "DreamBooth"
  - "扩散模型"
  - "adversarial attack"
---

# Towards Robust Defense against Customization via Protective Perturbation Resistant to Diffusion-based Purification

**会议**: ICCV 2025  
**arXiv**: [2509.13922](https://arxiv.org/abs/2509.13922)  
**代码**: 无（未提及）  
**领域**: 图像生成 / 对抗防御 / 扩散模型安全  
**关键词**: Protective Perturbation, Anti-Purification, DreamBooth, Diffusion Purification, adversarial attack

## 一句话总结
提出 AntiPure，一种针对扩散模型净化（purification）过程的对抗扰动方法，通过 Patch-wise Frequency Guidance 和 Erroneous Timestep Guidance 两种引导机制，生成在净化后仍能持续干扰定制化微调的保护性扰动，在"净化-定制化"工作流中全面超越现有防护方法。

## 研究背景与动机

### 问题背景
Stable Diffusion 等扩散模型的定制化微调技术（如 DreamBooth、LoRA）虽然强大，但也带来了深度伪造和版权侵权的严重安全威胁。保护性扰动（Protective Perturbation）通过在图像中注入不可感知的对抗噪声来干扰微调输出，是一种有前景的防御手段。

### 核心挑战
现有保护性扰动（如 AdvDM、Anti-DreamBooth）可以被扩散模型净化（如 DiffPure、GrIDPure）轻松去除。净化过程通过向对抗图像添加噪声再去噪，有效消除了对抗扰动，使保护失效。在实际场景中，恶意用户可以先净化图像再微调，形成"净化-定制化"（P-C）工作流，导致现有防护方法几乎完全失效。

### 关键洞察
作者深入分析了 anti-purification 比 anti-customization 更困难的三个核心原因：
1. **缺乏脆弱组件**：LDM 有易受攻击的 VAE encoder，而 DDPM 净化模型只有鲁棒的 UNet
2. **无需训练的冻结参数**：净化过程不需要微调，对抗样本无法通过数据毒化影响模型先验
3. **固定高时间步去噪**：净化的去噪从高时间步开始，低频结构已被锁定，攻击被限制在高频分量

## 方法详解

### 整体框架
AntiPure 不试图保留 anti-customization 扰动通过净化，而是直接攻击净化模型本身。核心思路是：即使后续定制化正常运行，净化过程引入的失真也会导致学到的概念偏离原始图像。

### 问题形式化
理想的抗净化扰动优化目标为：

$$\delta^{adv} = \arg\max_{\|\delta\|_\infty \leq \eta} \min_{\theta_c} \mathbb{E}_x \mathcal{L}_{ldm}(\text{Pure}(x_0 + \delta); \theta_c)$$

由于直接反向传播计算图过深，作者将其分解为最大化净化前后的差异：

$$\delta^{adv'} = \arg\max_{\|\delta\|_\infty \leq \eta} \|\text{Pure}(x_0 + \delta) - (x_0 + \delta)\|_\infty$$

### 关键设计一：Patch-wise Frequency Guidance (PFG)
冻结参数中的清洁图像先验使净化模型能良好恢复低频结构，但对高频分量控制较弱。PFG 利用这一弱点：

1. 对噪声对抗样本 $x_t$ 使用 UNet 预测去噪图像 $\widehat{x}_0$
2. 将 $\widehat{x}_0$ 分解为 patch，对每个 patch 做 DCT 变换
3. 提取高频分量（DCT 谱图右下角四分之一）并最大化：

$$\mathcal{L}_{fre}(x_0; \delta^{adv}) = \sigma\left(\mathbb{E}_P \frac{4}{s^2} \sum_{m,n=s/2}^{s-1} \text{PatchDCT}(\widehat{x}_0, s)_{m,n}\right)$$

PFG 增强净化模型预测中的高频分量，间接强化对抗扰动的高频元素，形成均匀网格模式。由于攻击目标是高频，局部结构信息变化最小，保证人类感知一致性。

### 关键设计二：Erroneous Timestep Guidance (ETG)
净化过程可视为高时间步去噪被固定的生成过程。ETG 通过注入对抗噪声，使 UNet 难以区分不同时间步的适当行为：

$$\mathcal{L}_{err\text{-}t}(x_0; \delta^{adv}) = -\|\epsilon_\theta(x_t, t_{err}) - \epsilon_\theta(x_t, t)\|_2^2$$

选择错误时间步 $t_{err}$ 作为 UNet 输入获取更高时间步的噪声预测，最小化错误时间步和正确时间步预测间的差异，瓦解模型的时间步感知能力。

### 总体损失函数
将 PFG 和 ETG 与原始 $\mathcal{L}_{ddpm}$ 结合，通过 PGD 梯度上升优化：

$$\mathcal{L}_{pgd}(x_0; \delta^{adv}) = \mathbb{E}_{\epsilon,t}\left(\mathcal{L}_{ddpm} + \lambda_1 e^{\bar{\alpha}_t - 1} \mathcal{L}_{fre} + \lambda_2 e^{\mathcal{L}_{err\text{-}t}}\right)$$

其中 $\lambda_1 = \lambda_2 = 0.5$，攻击时间步 $t \sim \mathcal{U}(1, t^p)$ 被限制在净化步范围内。系数 $e^{\bar{\alpha}_t - 1}$ 使 PFG 随 $t$ 降低时影响增大；指数函数应用于 ETG 实现更积极的优化。

## 实验

### 实验设置
- **数据集**：CelebA-HQ 和 VGGFace2，各 50 个 ID × 12 张 512×512 图像
- **基线方法**：AdvDM、Mist、Anti-DreamBooth、SimAC
- **净化方法**：GrIDPure（2 轮 × 20 迭代，$t^p=10$）
- **评估指标**：FID↑、ISM↓、FDFR、BRISQUE↑（定制化输出质量），LPIPS↓（扰动感知差异）

### 主实验：DreamBooth P-C 工作流

| 数据集 | 方法 | FID↑ | ISM↓ | BRISQUE↑ |
|--------|------|------|------|----------|
| CelebA-HQ | AdvDM | 77.51 | 0.6561 | 31.33 |
| CelebA-HQ | Mist | 70.23 | 0.6688 | 37.00 |
| CelebA-HQ | Anti-DB | 78.84 | 0.6422 | 31.76 |
| CelebA-HQ | SimAC | 67.37 | 0.6734 | 33.73 |
| CelebA-HQ | **AntiPure** | **81.15** | **0.6112** | **43.60** |
| VGGFace2 | AdvDM | 83.90 | 0.5923 | 37.42 |
| VGGFace2 | Anti-DB | 90.29 | 0.5938 | 38.35 |
| VGGFace2 | **AntiPure** | **90.77** | **0.5475** | **46.01** |

AntiPure 在所有指标和两个数据集上均取得最优表现。

### LoRA 微调验证

| 数据集 | 方法 | FID↑ | ISM↓ | BRISQUE↑ |
|--------|------|------|------|----------|
| VGGFace2 | Anti-DB | 117.89 | 0.5723 | 58.56 |
| VGGFace2 | **AntiPure** | **127.67** | **0.5428** | **69.97** |

在 LoRA 微调场景下仍全面领先，ISM 指标差距尤为显著。

### 净化迭代消融

| 方法 | 迭代=10 ISM | 迭代=20 ISM | 迭代=30 ISM | 迭代=40 ISM |
|------|------------|------------|------------|------------|
| Anti-DB | 0.6020 | 0.6352 | 0.6473 | 0.6391 |
| AntiPure | 0.6362 | 0.6271 | 0.6075 | **0.5994** |

Anti-DB 随迭代增加逐渐失效（ISM 升高），而 AntiPure 反而越来越强——这与其直接攻击净化过程本身的设计一致。

### 感知一致性
AntiPure 在相同 $\eta$ 约束下实现了最小的 LPIPS 感知差异，这归功于 PFG 有效避免了对低频信息的修改。

## 亮点与洞察
- **首次形式化 anti-purification 任务**：系统分析了为什么对抗净化比对抗定制化更困难，为后续研究奠定理论基础
- **"以毒攻毒"的巧妙思路**：不试图让扰动"幸存"净化过程，而是让净化过程本身产生失真，间接扰动后续微调
- **频率域与时间步的双重攻击**：PFG 攻击净化模型的高频控制弱点，ETG 瓦解时间步感知，两者协同增效
- **随净化加深反而增效**：独特的"越净化越有效"特性，展示了方法的鲁棒性

## 局限性
- 无法实现语义结构级别的严重失真（如完全扭曲人脸），只能引入可辨识的伪影
- 依赖白盒攻击，需要知道净化模型的具体架构和参数
- 评估主要限于人脸数据集和 DreamBooth/LoRA 两种微调方式
- 使用 JPEG 压缩时性能有所下降（CelebA-HQ 结果明显差于 VGGFace2）

## 相关工作
- **定制化微调**：DreamBooth、LoRA、Textual Inversion、Custom Diffusion、ControlNet
- **保护性扰动**：AdvDM、Mist、Anti-DreamBooth、SimAC、MetaCloak、CAAT
- **扩散净化**：DiffPure、DensePure、GrIDPure

## 评分
- 新颖性：⭐⭐⭐⭐ — 首次系统定义并解决 anti-purification 问题
- 技术深度：⭐⭐⭐⭐ — 对三个核心挑战的分析透彻
- 实验充分度：⭐⭐⭐⭐ — 多数据集、多微调方法、多净化配置
- 实用价值：⭐⭐⭐ — 白盒假设限制了实际部署

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing](dct-shield_a_robust_frequency_domain_defense_against_malicious_image_editing.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[AAAI 2026\] VoiceCloak: A Multi-Dimensional Defense Framework against Unauthorized Diffusion-based Voice Cloning](../../AAAI2026/image_generation/voicecloak_a_multi-dimensional_defense_framework_against_unauthorized_diffusion-.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](../../NeurIPS2025/image_generation/token_perturbation_guidance_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
