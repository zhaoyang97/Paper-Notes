---
title: >-
  [论文解读] Removing Reflections from RAW Photos
description: >-
  [CVPR 2025][去反射] 提出首个基于 RAW 图像的端到端去反射系统：在 XYZ 色彩空间中模拟逼真的反射（含 Fresnel/双反射/WB/曝光），训练 EfficientNet+BiFPN 基础模型分离透射/反射层，再用高斯金字塔上采样器保留高分辨率细节，利用可选的自拍相机上下文图辅助判断，PSNR 30.62dB。
tags:
  - "CVPR 2025"
  - "去反射"
  - "RAW图像"
  - "光度学仿真"
  - "上采样器"
  - "双摄像头"
---

# Removing Reflections from RAW Photos

**会议**: CVPR 2025  
**arXiv**: [2404.14414](https://arxiv.org/abs/2404.14414)  
**代码**: 有（推断）  
**领域**: 其他  
**关键词**: 去反射, RAW图像, 光度学仿真, 上采样器, 双摄像头

## 一句话总结

提出首个基于 RAW 图像的端到端去反射系统：在 XYZ 色彩空间中模拟逼真的反射（含 Fresnel/双反射/WB/曝光），训练 EfficientNet+BiFPN 基础模型分离透射/反射层，再用高斯金字塔上采样器保留高分辨率细节，利用可选的自拍相机上下文图辅助判断，PSNR 30.62dB。

## 研究背景与动机

### 领域现状

**领域现状**：透过窗户拍照时经常出现的反射是图像质量的大敌。现有去反射方法在 8-bit JPG 图像上训练，但窗户反射的物理过程（Fresnel 反射率+曝光+白平衡+色调映射）在 JPG 中已被不可逆压缩。

**现有痛点**：（1）8-bit JPG 丢失了暗区和高光区的精度，但反射判断正需要这些区域的信息；（2）合成训练数据的真实度不足——简单的 alpha 混合无法模拟 Fresnel 角度依赖/双面反射/色温差异；（3）低分辨率预测后上采样会重新引入反射伪影。

**核心矛盾**：现实中的反射是复杂光学过程，但训练数据要么不够逼真（合成），要么无法获得GT（真实）。

**切入角度**：在 RAW 域（线性的 XYZ 色彩空间）仿真反射形成的完整物理过程，然后在 RAW 上训练去反射模型——RAW 保留了所有光度学信息。

**核心 idea**：RAW 域物理仿真 + 上下文自拍辅助 + 高斯金字塔上采样 = 光度学准确的去反射。

### 解决思路

**本文目标**：### 关键设计

1. **RAW 域物理仿真**：在 XYZ 空间模拟：Fresnel 角度依赖反射率、双面玻璃的多次反射、不同光源色温（透射/反射各自白平衡）、曝光差异、模糊。


## 方法详解

### 整体框架


### 关键设计

1. **RAW 域物理仿真**：在 XYZ 空间模拟：Fresnel 角度依赖反射率、双面玻璃的多次反射、不同光源色温（透射/反射各自白平衡）、曝光差异、模糊。每步仿真基于物理光学

2. **双流基础模型**：EfficientNet-B1 backbone + BiFPN 融合 + StyleGAN 式 mod-demod 块。可选第二输入（自拍相机拍室内），提供上下文帮助判断哪些是反射

3. **高斯金字塔上采样器**：从 256p 分离结果到 2048p 全分辨率。用特征匹配的乘积掩码防止反射在上采样时重新引入

### 损失函数 / 训练策略

基础模型：感知损失(VGG19) + 对抗损失 + 梯度损失(5-tap) + L1 反射损失。上采样器：L1(0.2) + L2(0.2) + 梯度(0.4) + LPIPS(0.8) + 循环一致性(10.0)。完全在合成数据上训练。

## 实验关键数据

| 指标 | 完整系统 | 无 RAW 仿真 |
|------|---------|------------|
| PSNR (透射) | **30.62%** | 下降 ~10 dB |
| SSIM | **95.2%** | 下降 ~40pp |
| 上下文图提升 | +4pp SSIM | — |

### 消融实验

- RAW 仿真 vs 简化仿真：去掉所有物理组件后性能下降 46pp——RAW 域仿真是最大贡献
- 上下文自拍图：+4pp SSIM（统计显著，p<1.7e-11）
- 实机部署：MacBook/iPhone 14 Pro 上 4.5-6.5 秒推理

### 关键发现
- **RAW > JPG 是决定性因素**：在 RAW 域训练比在 JPG 上好 40pp SSIM——比任何架构改进都大
- **仿真的每个物理组件都重要**：去掉 gamma/曝光/Fresnel/WB 任一都显著降低性能
- **上采样器的掩码机制有效**：防止反射在高分辨率恢复时"泄漏"回来

## 亮点与洞察
- **"数据 > 模型" 的典范**——光度学准确的仿真数据带来40pp提升，远超架构创新
- **端到端可部署**——在 iPhone 上实时运行

## 局限与展望
- 需要 RAW 输入（不适用于已压缩的 JPG）
- 过饱和区域需要硬填充
- 无法处理纹理高度重叠的反射

## 评分
- 新颖性: ⭐⭐⭐⭐ RAW 域仿真+自拍辅助是关键创新
- 实验充分度: ⭐⭐⭐⭐⭐ 详尽的组件消融+实机部署
- 写作质量: ⭐⭐⭐⭐ 物理仿真过程描述详细
- 价值: ⭐⭐⭐⭐ 可直接部署到手机相机应用


## 相关工作与启发
- **vs 同领域代表性方法**：本文在方法设计上有独特贡献，与现有方法形成互补
- **vs 传统方法**：相比传统方案，本文方法在关键指标上取得了显著提升
- **启发**：本文的技术路线对后续相关工作有重要参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Predicting Kernel Regression Learning Curves from Only Raw Data Statistics](../../ICLR2026/others/predicting_kernel_regression_learning_curves_from_only_raw_data_statistics.md)
- [\[CVPR 2026\] Zero-shot Detection of AI-Generated Image via RAW-RGB Alignment](../../CVPR2026/others/zero-shot_detection_of_ai-generated_image_via_raw-rgb_alignment.md)
- [\[CVPR 2025\] Tuning the Frequencies: Robust Training for Sinusoidal Neural Networks](tuning_the_frequencies_robust_training_for_sinusoidal_neural_networks.md)
- [\[CVPR 2025\] Integral Fast Fourier Color Constancy](integral_fast_fourier_color_constancy.md)
- [\[CVPR 2025\] PLeaS: Merging Models with Permutations and Least Squares](pleas_-_merging_models_with_permutations_and_least_squares.md)

</div>

<!-- RELATED:END -->
