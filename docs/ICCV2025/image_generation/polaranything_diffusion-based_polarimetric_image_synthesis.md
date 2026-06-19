---
title: >-
  [论文解读] PolarAnything: Diffusion-based Polarimetric Image Synthesis
description: >-
  [ICCV 2025][图像生成][偏振图像合成] 提出 PolarAnything，首个基于单张 RGB 图像生成偏振图像的扩散模型框架，通过对编码后的 AoLP 和 DoLP 进行去噪扩散，实现了物理准确且逼真的偏振属性合成，无需 3D 资产或偏振相机。 偏振成像的价值与痛点：偏振图像为 shape-from-polar…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "偏振图像合成"
  - "扩散模型"
  - "AoLP"
  - "DoLP"
  - "Shape from Polarization"
---

# PolarAnything: Diffusion-based Polarimetric Image Synthesis

**会议**: ICCV 2025  
**arXiv**: [2507.17268](https://arxiv.org/abs/2507.17268)  
**代码**: [项目主页](https://flzt11.github.io/PA_project/)  
**领域**: 3D视觉 / 偏振成像  
**关键词**: 偏振图像合成, diffusion model, AoLP, DoLP, Shape from Polarization, Stable Diffusion

## 一句话总结

提出 PolarAnything，首个基于单张 RGB 图像生成偏振图像的扩散模型框架，通过对编码后的 AoLP 和 DoLP 进行去噪扩散，实现了物理准确且逼真的偏振属性合成，无需 3D 资产或偏振相机。

## 研究背景与动机

**偏振成像的价值与痛点**：偏振图像为 shape-from-polarization（SfP）、去雾、反射去除等任务提供了独特的物理线索（AoLP 编码表面法线信息，DoLP 反映材质属性）。然而偏振相机价格昂贵、普及度极低，导致：

**数据集稀缺**：现有真实偏振数据集要么量极小（Morimatsu 仅 40 组，Qiu 仅 38 组），要么场景单一（HAMMER/HouseCat6D 专注于室内 6D 位姿）

**模拟器局限**：Mitsuba 等基于物理的渲染器需要完整的 3D 资产（mesh + PBR 材质 + 环境光照），且参数化 pBRDF 模型与真实偏振属性存在 gap，无法大规模生成逼真偏振图像

**学习方法受限**：DeepSfP 等学习方法训练数据仅 236 张，严重制约泛化性

**核心洞察**：RGB 图像容易获取且天然覆盖多样场景；预训练扩散模型（Stable Diffusion）从大规模数据中学到了强大的图像先验，可零样本迁移到判别任务。能否利用扩散先验，从单张 RGB 直接生成偏振属性？

## 方法详解

### 整体框架

PolarAnything 基于 Stable Diffusion v1.5 微调，核心架构包含四个模块：

- **图像条件编码器** $\mathcal{E}_{\text{img}}$：卷积层 + SiLU 激活 → 与 U-Net encoder 同构的特征编码器，提取 RGB 图像的多尺度条件特征
- **VAE 编码器** $\mathcal{E}_{\text{vae}}$：将编码后的 AoLP+DoLP 映射到潜空间（权重冻结）
- **去噪 U-Net** $\mu_\theta$：以条件特征做 ControlNet 式的层级引导，在潜空间对偏振属性去噪
- **VAE 解码器** $\mathcal{D}_{\text{vae}}$：将去噪后的潜码还原为 AoLP+DoLP 映射（权重冻结）

推理流程：RGB 图像 → $\mathcal{E}_{\text{img}}$ 提取条件 → 从高斯噪声出发迭代去噪 → 解码得到 AoLP 和 DoLP → 由 Eq.(1) 合成任意偏振角的偏振图像。

### 关键设计：偏振信息编码策略

这是本文最重要的技术贡献。作者对比了三种扩散目标：

| 方案 | 扩散目标 | 问题 |
|------|---------|------|
| (a) 直接生成偏振图像 | $\mathbf{I}_{0°}, \mathbf{I}_{45°}, \mathbf{I}_{90°}, \mathbf{I}_{135°}$ | 保留了 RGB 辐射信息但几乎无法恢复偏振属性，缺乏物理约束 |
| (b) 直接生成 AoLP+DoLP | $\Phi, \mathbf{P}$ | AoLP 值域 $[-90°, 90°]$ 具有 $\pi$-周期性，直接回归会破坏周期结构 |
| (c) 编码 AoLP+DoLP ✓ | $[\cos 2\Phi; \sin 2\Phi; \mathbf{P}]$ | 正弦编码保留周期性且连续，DoLP 归一化到 $[-1,1]$，与 VAE 兼容 |

**为什么编码有效**：AoLP 的 $\pi$-周期性意味着 $-90°$ 和 $90°$ 实际上是相同的偏振方向，直接回归会让网络在边界处产生大误差。正弦编码 $(\cos 2\Phi, \sin 2\Phi)$ 将角度映射到单位圆上的连续表示，自然处理了周期性问题。

### 损失函数

标准的去噪扩散损失：

$$\min_\theta \mathbb{E}_{x \sim \mathcal{E}_{\text{vae}}, t, \epsilon \sim \mathcal{N}(0,1)} \|\epsilon_t - \mu_\theta(z_t, t, c, \mathcal{E}_{\text{img}}(\mathbf{I}_{\text{RGB}}))\|_2^2$$

其中 $c$ 为 CLIP 文本嵌入，$z_t$ 为加噪潜码。值得注意的是，整个 U-Net 权重可训练（不同于 ControlNet 冻结原始权重），实验证明这更有效。

### 数据集构建

作者用偏振相机采集了 1,148 张高质量偏振图像（1224×1024），覆盖：
- **100+ 种物体**：透明/不透明、导体/绝缘体、漫反射/高光
- **19 种光照环境**：8 种室外 + 11 种室内
- 与 Morimatsu 数据合并，1,115 张用于训练，33 张用于测试

## 实验关键数据

### 主实验：偏振属性合成质量（消融实验）

| 扩散目标 | PSNR↑ | SSIM↑ | MAngE↓ | MAbsE↓ |
|---------|-------|-------|--------|--------|
| 偏振图像 (a) | 23.23 | 0.9165 | 45.67 | 0.1233 |
| AoLP+DoLP (b) | 40.57 | 0.9904 | 29.46 | 0.1100 |
| **编码 AoLP+DoLP (c)** | **41.74** | **0.9927** | **25.33** | **0.1075** |

编码策略在所有指标上均最优：PSNR 提升近 19 dB（vs 方案 a），MAngE 降低 44%（vs 方案 a）。

### 下游任务：单视角 SfP（DeepSfP 增强）

| 测试集 | 训练集 | Mean↓ | Median↓ | RMSE↓ | ≤10°↑ | ≤20°↑ | ≤30°↑ |
|-------|-------|-------|---------|-------|-------|-------|-------|
| DP+PN | DeepSfP 原始 | 22.21 | 18.13 | 26.88 | 24.68 | 57.39 | 76.07 |
| DP+PN | +Mitsuba (MSO) | 20.42 | 17.19 | 24.31 | 23.23 | 61.32 | 80.67 |
| DP+PN | **+PolarAnything (PSO)** | **20.13** | **16.81** | **24.15** | **24.47** | **62.00** | **81.92** |

用 PolarAnything 合成 300 张偏振图像（PolarStanford-ORB）扩充训练集，效果超越 Mitsuba 渲染的相同规模数据。在新采集的真实测试集 PN 上提升更显著（Mean: 30.69→22.93）。

### 多视角 SfP（PISR 评估）

| 输入数据 | MAngE (法线)↓ | CD (网格)↓ | MAngE (AoLP) | MAbsE (DoLP) |
|---------|-------------|----------|-------------|-------------|
| 真实偏振图像 | 15.45 | 0.6765 | N/A | N/A |
| PolarAnything 生成 | 15.17 | 0.6564 | 33.68 | 0.1563 |

生成图像的 3D 重建质量与真实偏振图像基本持平甚至略优。

### 关键发现

1. **编码策略是核心**：直接生成偏振图像会丢失偏振物理信息，正弦编码 AoLP 是解决周期性问题的关键
2. **全参数微调优于 ControlNet 式冻结**：与 ControlNet 不同，解锁所有 U-Net 权重效果更好
3. **扩散模型 vs Transformer**：Restormer 在小数据场景下表现明显不如扩散模型，后者的零样本先验在数据稀缺时优势巨大
4. **泛化性强**：在 NeRSP、PANDORA 等分布外数据集上仍能生成合理的偏振属性，覆盖漫反射、金属、透明等多种材质

## 亮点与洞察

1. **问题定义精准**：将偏振图像合成从"需要 3D 资产的渲染问题"重新定义为"单图条件生成问题"，大幅降低了数据获取门槛
2. **编码设计优雅**：$(\cos 2\Phi, \sin 2\Phi)$ 的正弦编码既解决了周期性，又与 VAE 的值域自然兼容，是物理先验与深度学习的完美结合
3. **端到端实用性**：不仅生成偏振图像，还构建了 PolarStanford-ORB 数据集，验证了对下游任务的实际增益
4. **训练效率高**：仅 1,155 张训练图像 + 600 步微调（8×A100，约 10 小时），说明扩散先验的迁移非常高效
5. **从 RGB 到偏振的范式**：开辟了利用海量 RGB 数据集间接扩充偏振训练集的新路径，对所有偏振相关的 CV 任务都有价值

## 局限性

1. **灰度限制**：当前模型将 RGB 转为灰度后计算偏振属性，丢失了颜色通道间的偏振差异信息
2. **训练数据规模**：1,155 张训练图像仍然偏少，可能限制了对复杂场景（大规模室外、极端光照）的泛化
3. **物理一致性无显式约束**：损失函数仅为标准去噪损失，未显式施加偏振物理约束（如 DoLP ∈ [0,1]、AoLP 与法线的 Fresnel 关系），依赖扩散先验隐式学习
4. **评估局限**：与 Mitsuba 的对比因相机标定误差无法做定量评估，只有定性比较
5. **单图输入的固有歧义**：从单张 RGB 推断偏振属性是病态问题（同一外观可对应不同偏振状态），多样性与准确性之间存在 trade-off

## 相关工作与启发

- **Marigold / GeoWizard / StableNormal**：同属"扩散模型做几何预测"范式，PolarAnything 将其扩展到偏振域，说明扩散先验的迁移潜力远未被充分发掘
- **ControlNet**：PolarAnything 的条件注入机制直接借鉴 ControlNet 的层级特征融合，但选择全参数微调而非冻结，值得后续研究探讨何时该冻结何时该解锁
- **DeepSfP / PISR**：下游 SfP 方法的数据瓶颈为 PolarAnything 提供了清晰的应用场景，也暗示了"合成数据 + 真实微调"的训练范式
- **启发**：这种"利用预训练生成模型合成稀缺模态数据"的思路可推广到热红外、SAR、光谱成像等同样面临数据稀缺的传感模态

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | 4 | 首个 RGB→偏振的扩散生成框架，问题定义和编码策略都很新颖 |
| 技术深度 | 3.5 | 偏振编码设计有巧思，但模型架构本身主要是 SD 微调，技术创新幅度有限 |
| 实验充分性 | 4 | 消融实验清晰，下游任务验证完整，多材质多场景定性展示丰富 |
| 实用价值 | 4.5 | 直接降低偏振数据获取门槛，PolarStanford-ORB 数据集对社区有实际贡献 |
| 写作质量 | 4 | 结构清晰，motivation 讲述流畅，图表精心设计 |
| **总分** | **4.0** | 扎实的应用驱动工作，编码策略是核心贡献，开辟了偏振数据合成新范式 |

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Omegance: A Single Parameter for Various Granularities in Diffusion-Based Synthesis](omegance_a_single_parameter_for_various_granularities_in_diffusion-based_synthes.md)
- [\[ICCV 2025\] Efficient Input-Level Backdoor Defense on Text-to-Image Synthesis via Neuron Activation Variation](efficient_input-level_backdoor_defense_on_text-to-image_synthesis_via_neuron_act.md)
- [\[ICCV 2025\] InfGen: A Resolution-Agnostic Paradigm for Scalable Image Synthesis](infgen_a_resolution-agnostic_paradigm_for_scalable_image_synthesis.md)
- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](../../CVPR2025/image_generation/diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[CVPR 2025\] GenDeg: Diffusion-based Degradation Synthesis for Generalizable All-In-One Image Restoration](../../CVPR2025/image_generation/gendeg_diffusion-based_degradation_synthesis_for_generalizable_all-in-one_image_.md)

</div>

<!-- RELATED:END -->
