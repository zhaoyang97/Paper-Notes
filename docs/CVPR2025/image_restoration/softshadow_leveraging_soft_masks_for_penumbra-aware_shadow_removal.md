---
title: >-
  [论文解读] SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal
description: >-
  [CVPR 2025][图像恢复][软阴影掩码] 提出SoftShadow框架，用连续灰度软掩码替代传统二值硬掩码来表示阴影区域，通过SAM+LoRA预测软掩码并引入半影形成约束损失联合训练检测与去阴影网络，在SRD/ISTD+/LRSS/UIUC四个数据集上达到SOTA且无需外部掩码输入。 领域现状：深度学习在阴影移除任务…
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "软阴影掩码"
  - "半影感知"
  - "SAM微调"
  - "物理约束损失"
  - "端到端阴影移除"
---

# SoftShadow: Leveraging Soft Masks for Penumbra-Aware Shadow Removal

**会议**: CVPR 2025  
**arXiv**: [2409.07041](https://arxiv.org/abs/2409.07041)  
**代码**: 有  
**领域**: 图像去阴影 / 图像恢复  
**关键词**: 软阴影掩码、半影感知、SAM微调、物理约束损失、端到端阴影移除

## 一句话总结

提出SoftShadow框架，用连续灰度软掩码替代传统二值硬掩码来表示阴影区域，通过SAM+LoRA预测软掩码并引入半影形成约束损失联合训练检测与去阴影网络，在SRD/ISTD+/LRSS/UIUC四个数据集上达到SOTA且无需外部掩码输入。

## 研究背景与动机

**领域现状**：深度学习在阴影移除任务上取得了显著进展，现有方法分为两类：一类依赖预生成的阴影掩码（如BMNet、HomoFormer使用DHAN/FDRNet检测器输出的二值掩码），另一类采用端到端方式直接去阴影（如DC-ShadowNet、DeS3），但后者由于缺少显式的阴影位置信息，性能往往不及前者。

**现有痛点**：所有依赖掩码的方法都使用二值掩码（$s \in \{0, 1\}$）来标注阴影区域，但真实阴影的半影区(penumbra)存在明显的亮度渐变过渡，二值表示无法编码这种渐变，导致去阴影结果在阴影边界处出现严重伪影。此外，不同阴影检测器生成的掩码质量差异巨大，PSNR波动高达7dB以上，系统鲁棒性极差。

**核心矛盾**：阴影物理形成模型天然包含umbra（全遮挡）和penumbra（半遮挡）两个区域，penumbra区域的光照呈连续渐变，但现有掩码表示丢弃了这一关键信息。需要一种既能精确定位阴影边界、又能编码半影渐变信息的掩码表示。

**本文目标** 设计一种连续软掩码（$s \in [0, 1]$）来替代二值硬掩码，并构建端到端框架自动预测软掩码、指导阴影移除，同时消除对外部掩码检测器的依赖。

**切入角度**：从阴影形成的物理模型出发，阴影图像可被建模为 $\mathbf{y} = \mathbf{a} \cdot \mathbf{s} \cdot \mathbf{x} + (1 - \mathbf{s}) \cdot \mathbf{x}$，其中 $\mathbf{s}$ 即为连续软掩码。利用SAM强大的分割先验知识，通过LoRA微调使其输出连续软掩码而非二值分割结果，并设计物理启发的半影约束来正则化掩码梯度。

**核心 idea**：将阴影掩码从二值离散表示升级为连续软表示，并通过物理约束引导SAM学习半影区域的渐变信息，实现更自然的边界恢复。

## 方法详解

### 整体框架

SoftShadow是一个端到端的统一阴影移除框架，不需要任何外部掩码输入。整体pipeline为：输入阴影图像 $\mathbf{y}$ → SAM+LoRA编码器提取特征并预测连续软掩码 $\hat{\mathbf{s}}$ → 将软掩码与阴影图像一起送入去阴影骨干网络（ShadowDiffusion）→ 输出去阴影结果 $\hat{\mathbf{x}}$。训练时同时优化三个损失：掩码重建损失 $\mathcal{L}_{mask}$、半影约束损失 $\mathcal{L}_{pen}$、去阴影损失 $\mathcal{L}_{rem}$，联合微调SAM和去阴影网络。

### 关键设计

1. **SAM+LoRA软掩码检测器**:
    - 功能：将预训练SAM从二值分割器改造为连续软掩码预测器，输出 $\hat{\mathbf{s}} \in [0,1]$，其中 $s=0$ 表示光照区、$s=1$ 表示全阴影区（umbra）、$s \in (0,1)$ 表示半影区（penumbra）。
    - 核心思路：SAM采用ViT-H作为图像编码器，在所有self-attention层插入LoRA适配层（rank=8），mask decoder则全参数微调。由于SAM原本训练用于二值分割，通过 $\mathcal{L}_{mask}$ 和 $\mathcal{L}_{pen}$ 两个损失引导其输出连续值。掩码gt通过将阴影图像和无阴影图像转到YCbCr空间后取Y通道比值获得：$\mathbf{s}_{gt} = \max(t, f(\mathbf{x}_Y / \mathbf{y}_Y))$，其中 $f$ 为低通滤波器，$t=0.76$ 为阈值。
    - 设计动机：SAM拥有强大的分割先验知识，但直接用SAM预测阴影掩码效果很差（PSNR仅28.16 dB）。使用LoRA微调可以在大幅减少可训练参数的同时有效适配SAM到连续掩码预测任务，兼顾效率与效果。

2. **半影形成约束损失（Penumbra Formation Constraint）**:
    - 功能：正则化预测软掩码在半影区域的梯度方向和幅值，确保掩码反映真实的光照渐变特征。
    - 核心思路：基于两个物理假设——(1)半影区掩码强度应从阴影中心向外逐渐降低，即梯度方向应从中心指向边界；(2)渐变过渡应平滑，即梯度幅值不宜过大。首先通过阈值 $t_1, t_2$ 定义半影区域 $\mathbf{w} = \{(i,j) \mid t_1 \leq \mathbf{s}_{i,j} \leq t_2\}$，计算阴影中心 $\mathbf{c}$ 为半影区坐标均值，定义方向单位向量 $\mathbf{d}(\mathbf{w}) = (\mathbf{w} - \mathbf{c}) / \|\mathbf{w} - \mathbf{c}\|$，约束损失为 $\mathcal{L}_{pen} = \mathbb{E}_{n,\mathbf{w}}[\text{ReLU}(\mathbf{d}(\mathbf{w}) \cdot \nabla M(\mathbf{w}))]$。ReLU过滤掉与期望方向一致的梯度（负值），仅惩罚与期望方向冲突的梯度分量。
    - 设计动机：单纯的掩码重建损失难以精确约束半影区域的局部梯度特性，而半影区恰好是阴影移除中最容易产生伪影的区域。该约束将阴影形成的物理直觉（光照从中心向外渐变）转化为可微分的正则化项，显式引导模型学习平滑过渡的软掩码。

3. **联合端到端训练策略**:
    - 功能：同时优化SAM软掩码检测器和去阴影骨干网络，使两者相互促进。
    - 核心思路：总体损失为 $\mathcal{L} = \mathcal{L}_{mask} + \lambda_1 \mathcal{L}_{pen} + \lambda_2 \mathcal{L}_{rem}$，其中 $\lambda_1 = 0.1$，$\lambda_2 = 1$。掩码重建损失为预测掩码与gt掩码的Frobenius范数：$\mathcal{L}_{mask} = \mathbb{E}_n \|\hat{\mathbf{s}} - \mathbf{s}_{gt}\|_F^2$；去阴影损失为恢复图像与gt的MSE：$\mathcal{L}_{rem} = \mathbb{E}_n \|\hat{\mathbf{x}} - \mathbf{x}\|_F^2$。去阴影损失的梯度同时反传到SAM，使预测的软掩码朝着更有利于去阴影效果的方向优化。
    - 设计动机：如果分别训练检测器和去阴影网络，掩码优化目标可能与去阴影目标不一致。联合训练让去阴影效果直接反馈到掩码预测，形成良性循环——更好的掩码产生更好的去阴影结果，去阴影信号反过来改进掩码质量。

### 损失函数 / 训练策略

三个损失各有分工：$\mathcal{L}_{mask}$ 提供掩码位置的全局监督；$\mathcal{L}_{pen}$ 在半影区域施加局部梯度约束；$\mathcal{L}_{rem}$ 提供任务级别的端到端监督。训练分辨率 $256 \times 256$，batch size 16，使用Adam优化器。去阴影骨干网络采用ShadowDiffusion，推理时使用DDIM采样器、5步采样。软掩码gt的生成不需要额外标注，直接通过阴影/无阴影图像对在YCbCr空间的Y通道比值计算，再经低通滤波和阈值处理。

## 实验关键数据

### 主实验

**SRD数据集（全图PSNR）**：

| 方法 | 需要掩码 | 影子区PSNR↑ | 非影子区PSNR↑ | 全图PSNR↑ | 全图MAE↓ |
|------|---------|-----------|-------------|---------|---------|
| DHAN | DHAN | 33.67 | 34.79 | 30.51 | 5.67 |
| ShadowDiffusion | DHAN | 38.72 | 37.78 | 34.73 | 3.63 |
| HomoFormer | DHAN | 38.81 | 39.45 | 35.37 | 3.33 |
| DeS3 | 否 | 37.91 | 37.45 | 34.11 | 3.56 |
| **SoftShadow** | **否** | **39.08** | **39.36** | **35.57** | **3.11** |

**ISTD+数据集（全图PSNR）**：

| 方法 | 需要掩码 | 影子区PSNR↑ | 全图PSNR↑ | 全图MAE↓ |
|------|---------|-----------|---------|---------|
| HomoFormer | GT | 39.49 | 35.35 | 2.64 |
| ShadowDiffusion | FDRNet | 40.12 | 34.08 | 3.12 |
| HomoFormer | FDRNet | 38.84 | 32.41 | 3.51 |
| DeS3 | 否 | 36.49 | 31.38 | 3.94 |
| **SoftShadow** | **否** | **40.36** | **35.00** | **2.85** |

### 消融实验

**逐步添加损失的消融（SRD + LRSS）**：

| 配置 | SRD PSNR↑ | SRD MAE↓ | LRSS PSNR↑ | LRSS MAE↓ |
|------|----------|---------|-----------|----------|
| 预训练权重 | 31.27 | 4.41 | 21.40 | 12.26 |
| + $\mathcal{L}_{rem}$ | 35.27 | 3.32 | 21.78 | 12.07 |
| + $\mathcal{L}_{rem} + \mathcal{L}_{mask}$ | 35.44 | 3.17 | 23.08 | 9.97 |
| + $\mathcal{L}_{rem} + \mathcal{L}_{mask} + \mathcal{L}_{pen}$ | **35.57** | **3.11** | **23.32** | **9.77** |

**半影区域专项评估（SRD）**：

| 方法 | 半影区PSNR↑ | 半影区MAE↓ |
|------|-----------|----------|
| Inpaint4Shadow | 40.10 | 4.23 |
| DeS3 | 40.91 | 4.08 |
| HomoFormer | 40.82 | 3.91 |
| **SoftShadow** | **41.84** | **3.77** |

### 关键发现

- 无需外部掩码即超越所有需掩码方法：SRD上全图PSNR 35.57 vs HomoFormer(DHAN掩码) 35.37，提升0.2 dB。
- 半影区域改进最为显著：比HomoFormer高1.02 dB PSNR，说明软掩码在半影建模上确实有效。
- ISTD+上即使不用GT掩码也接近GT掩码方法性能（35.00 vs 35.35/35.67），而使用检测器掩码的方法仅32-34 dB。
- 掩码敏感性分析显示传统方法依赖掩码质量（PSNR标准差约0.9-1.0 dB），而SoftShadow完全免疫。
- $\mathcal{L}_{mask}$ 贡献最大（LRSS上+1.3 dB），$\mathcal{L}_{pen}$ 进一步改善半影区边界质量。
- 泛化性强：仅在SRD上训练，在LRSS（23.32 dB）和UIUC（28.85 dB）上均大幅超越DC-ShadowNet等基线。

## 亮点与洞察

- **硬掩码→软掩码的问题重定义**：这是一个"为什么之前没人做"的简单却深刻的改动，将 $\{0,1\}$ 扩展为 $[0,1]$ 完美匹配了阴影的物理形成模型，方法论价值大于技术复杂度。
- **物理约束的可微化**：将"半影区亮度从中心向外渐变"的物理直觉转化为梯度方向约束，通过ReLU选择性惩罚并保证可微性，是一种优雅的先验注入方式。
- **SAM+LoRA的任务适配**：把SAM从二值分割器改造为连续密度预测器，仅用LoRA微调encoder且联合端到端训练，思路通用且参数高效。
- **无需额外标注的软掩码GT生成**：直接用SRD等数据集中已有的阴影/无阴影图像对、通过Y通道比值计算获得软掩码GT，巧妙避免了新标注需求。

## 局限与展望

- 在ISTD+等硬阴影占主导的数据集上优势不显著（硬阴影本身penumbra区域小，软掩码的信息增益有限）。
- 去阴影骨干网络使用ShadowDiffusion（扩散模型），推理速度受限于采样步数，实时应用有较大差距。
- 半影约束假设阴影为简单的单中心渐变，对多光源或复杂几何遮挡场景可能失效。
- 软掩码GT的生成依赖于阈值 $t=0.76$ 的手工设定，对不同数据集的适应性未充分讨论。
- 自阴影(self-shadow)场景的实验有限，UIUC上虽有结果但缺乏深入分析。

## 相关工作与启发

- **vs HomoFormer**：HomoFormer通过均匀化阴影掩码的空间分布来统一恢复，但本质仍依赖二值掩码输入，无法处理半影渐变。SoftShadow不仅免去掩码输入，还通过软掩码提供更丰富的位置信息。
- **vs DeS3**：DeS3采用无掩码端到端方式，利用自监督策略移除self-shadow，但缺乏显式的阴影位置引导。SoftShadow通过内部软掩码检测器弥补了这一不足，SRD上全图PSNR提升1.46 dB。
- **vs SAM-helps-shadow**：该方法直接使用SAM预测二值掩码并输入去阴影网络，属于naive pipeline组合。SoftShadow则将SAM改造为连续掩码预测器，并联合训练使掩码服务于去阴影任务，PSNR从30.72→35.57提升近5 dB。
- **vs ShadowDiffusion**：作为SoftShadow的骨干网络，ShadowDiffusion依赖外部掩码且对掩码质量敏感。SoftShadow通过前置软掩码检测和联合训练，将其性能天花板从34.73提升到35.57 dB。

## 评分

- 新颖性: ⭐⭐⭐⭐ 软掩码概念直观有效，是对阴影掩码表示的根本性改进，但技术手段（SAM+LoRA、物理约束）相对标准
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集、半影专项评估、掩码敏感性分析、逐步消融，实验设计非常全面
- 写作质量: ⭐⭐⭐⭐ 物理动机阐述清晰，从阴影形成模型推导软掩码的逻辑严谨，图示直观
- 价值: ⭐⭐⭐⭐ 对阴影移除领域有实际推动作用，消除外部掩码依赖的方向具有工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Detail-Preserving Latent Diffusion for Stable Shadow Removal](detail-preserving_latent_diffusion_for_stable_shadow_removal.md)
- [\[CVPR 2026\] UniSER: A Foundation Model for Unified Soft Effects Removal](../../CVPR2026/image_restoration/uniser_a_foundation_model_for_unified_soft_effects_removal.md)
- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](../../CVPR2026/image_restoration/phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[CVPR 2025\] Reversible Decoupling Network for Single Image Reflection Removal](reversible_decoupling_network_for_single_image_reflection_removal.md)
- [\[CVPR 2025\] Degradation-Aware Feature Perturbation for All-in-One Image Restoration](degradation-aware_feature_perturbation_for_all-in-one_image_restoration.md)

</div>

<!-- RELATED:END -->
