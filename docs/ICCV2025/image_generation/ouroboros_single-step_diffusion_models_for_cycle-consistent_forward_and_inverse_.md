---
title: >-
  [论文解读] Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering
description: >-
  [ICCV 2025][图像生成][逆渲染] 本文提出 Ouroboros，一个由两个单步扩散模型（分别负责逆渲染 RGB→X 和前向渲染 X→RGB）组成的统一框架，通过循环一致性训练确保双向渲染的一致性，在多个数据集上取得 SOTA 的同时推理速度比多步扩散方法快 50 倍，并可零样本迁移到视频分解。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "逆渲染"
  - "前向渲染"
  - "循环一致性"
  - "单步扩散"
  - "内在图像分解"
---

# Ouroboros: Single-step Diffusion Models for Cycle-consistent Forward and Inverse Rendering

**会议**: ICCV 2025  
**arXiv**: [2508.14461](https://arxiv.org/abs/2508.14461)  
**代码**: [https://siwensun.github.io/ouroboros-project/](https://siwensun.github.io/ouroboros-project/)  
**领域**: 扩散模型 / 3D视觉  
**关键词**: 逆渲染, 前向渲染, 循环一致性, 单步扩散, 内在图像分解

## 一句话总结

本文提出 Ouroboros，一个由两个单步扩散模型（分别负责逆渲染 RGB→X 和前向渲染 X→RGB）组成的统一框架，通过循环一致性训练确保双向渲染的一致性，在多个数据集上取得 SOTA 的同时推理速度比多步扩散方法快 50 倍，并可零样本迁移到视频分解。

## 研究背景与动机

**领域现状**：逆渲染（从图像估计几何、材质、光照等内在属性）和前向渲染（从内在属性合成图像）是计算机图形学和视觉的基础问题。近年来，扩散模型在两个方向上都取得了进展：RGB↔X 首次提出了统一的扩散框架同时支持双向渲染，DiffusionRenderer 将其扩展到视频领域。

**现有痛点**：
   - **计算效率低**：现有扩散方法需要多步去噪（如 50 步 DDIM），推理速度慢
   - **缺乏循环一致性**：逆渲染和前向渲染模型独立训练，当顺序应用时（image → decomposition → reconstruction），重建结果往往与原图不一致
   - **数据域限制**：RGB↔X 仅在室内数据集上训练，对室外场景泛化能力弱
   - **训练数据稀缺**：包含完整 RGB-X 配对的大规模数据集有限，不同数据集的可用内在属性通道不一致

**核心矛盾**：如何在保持高质量的前提下大幅加速扩散渲染，同时确保双向渲染的循环一致性？

**本文目标** (a) 将多步扩散渲染加速为单步推理；(b) 建立逆渲染和前向渲染之间的循环一致性；(c) 扩展到室内外多域场景；(d) 以免训练方式实现视频分解。

**切入角度**：受 E2E（End-to-End Fine-tuning）启发，固定 timestep 为 T 进行单步预测微调。关键创新是将单步微调从感知任务（几何估计）扩展到合成任务（前向渲染），并引入类 CycleGAN 的循环一致性训练使逆渲染和前向渲染相互增强。

**核心 idea**：将 RGB↔X 微调为两个单步扩散模型（50× 加速），通过循环一致性损失联合训练使双向渲染互相正则化，并利用循环结构引入无标注真实数据实现跨域泛化。

## 方法详解

### 整体框架

Ouroboros 由两个互补的单步扩散模型组成：(1) 逆渲染模型（RGB→X），从 RGB 图像预测 5 种内在属性图（法向量 $\mathbf{n}$、反射率 $\mathbf{a}$、粗糙度 $\mathbf{r}$、金属度 $\mathbf{m}$、漫反射辐照度 $\mathbf{E}$）；(2) 前向渲染模型（X→RGB），从内在属性图合成 RGB 图像。两个模型从 RGB↔X 的预训练权重微调，通过循环训练相互增强。

### 关键设计

1. **单步预测微调（Single-step Finetuning）**:

    - 功能：将多步扩散模型微调为单步推理，实现 50× 加速
    - 核心思路：固定 timestep $t = T$，强制模型从最大噪声状态一步生成目标。与 E2E 不同，不使用零噪声而是多分辨率噪声初始化，UNet 的 v-prediction 输出通过 $\hat{\mathbf{z}}_0 = \sqrt{\bar\alpha_T} \mathbf{z}_T - \sqrt{1-\bar\alpha_T} \hat{\mathbf{v}}_\theta$ 转换为预测的 latent，再解码比较
    - 设计动机：使用非确定性噪声初始化（而非零噪声）特别适合内在图像分解这种存在多解性的任务——因为同一图像的分解结果不唯一（如 albedo 和 irradiance 的划分本身就有歧义性）

2. **任务特定的损失函数**:

    - 功能：为不同类型的内在属性设计专门的损失
    - 法向量：角度损失 $\mathcal{L}_\mathbf{n} = \frac{1}{N}\sum_i \arccos \frac{\mathbf{n}_i \cdot \hat{\mathbf{n}}_i}{||\mathbf{n}_i|| \cdot ||\hat{\mathbf{n}}_i||}$
    - 辐照度：仿射不变损失 $\mathcal{L}_\mathbf{E} = |\mathbf{E} - \mathbf{S}\hat{\mathbf{E}} - \mathbf{T}|_F^2$（通过最小二乘拟合每通道的缩放和偏移参数）
    - 其他属性（albedo、roughness、metallicity、RGB）：MSE 损失
    - 设计动机：法向量需要角度一致而非数值一致；辐照度与 albedo 的分解存在缩放歧义，仿射不变损失消除了这种歧义

3. **循环一致性训练（Cycle Training）**:

    - 功能：确保 RGB→X→RGB' 和 X→RGB→X' 两个循环的输出与输入一致
    - 核心思路：给定 $(X, I)$ 对，先用两个模型分别生成 $(\hat{I}, \hat{X})$，再将生成的结果作为输入做第二轮推理得到 $(\tilde{X}, \tilde{I})$，最小化循环损失：$\mathcal{L}_{cycle} = |\mathbf{X} - \tilde{\mathbf{X}}|^2 + |\mathbf{I} - \tilde{\mathbf{I}}|^2$
    - 设计动机：(a) 改善双向渲染的一致性，(b) 更重要的是通过循环结构引入无标注真实图像数据（MSCOCO、Flickr30k）进行自监督训练，减少对稀缺合成数据的依赖

4. **免训练视频推理**:

    - 功能：将图像级别的单步模型扩展到视频分解，无需视频数据训练
    - 核心思路：将 2D 卷积核 $3 \times 3$ 替换为伪 3D 核 $1 \times 3 \times 3$，展平多帧 patch 在空间-时间维度联合做 attention。使用滑动窗口处理长视频，重叠区域的初始化使用前一窗口的预测结果与噪声的加权混合：$\mathbf{z}_{init} = \gamma \cdot \mathbf{z}_{prev} + (1-\gamma) \cdot \epsilon$（$\gamma=0.1$）
    - 设计动机：训练原生视频扩散模型成本高昂，通过简单的架构扩展利用时空局部性实现时序一致性

### 损失函数 / 训练策略

训练分两轮：(1) 初始单步微调——在 Hypersim、InteriorVerse（室内）、MatrixCity（室外）三个数据集上微调，每个采样 17K 图像，使用 channel dropout 处理不同数据集的属性通道差异；(2) 循环训练——将初始微调后的两个模型联合训练，额外加入 MSCOCO 和 Flickr30k 各 20K 无标注图像进行自监督。总损失 = 任务损失 + 循环损失。

## 实验关键数据

### 主实验（逆渲染）

**Albedo 预测**：

| 方法 | Hypersim PSNR↑ | MatrixCity PSNR↑ | InteriorVerse PSNR↑ |
|------|---------------|------------------|---------------------|
| RGB↔X | 18.67 | 12.61 | 16.17 |
| Careaga & Aksoy | 12.01 | 17.30 | 15.51 |
| Kocsis et al. | 12.40 | 15.66 | 14.62 |
| **Ouroboros** | **18.98** | **25.38** | **22.07** |

**Normal 预测**：

| 方法 | Hypersim Mean↓ | MatrixCity Mean↓ | InteriorVerse Mean↓ |
|------|---------------|------------------|---------------------|
| RGB↔X | 17.21 | 23.82 | 12.10 |
| StableNormal | 16.65 | 18.18 | 10.73 |
| E2E | 16.30 | 13.91 | 15.87 |
| **Ouroboros** | **11.98** | **18.12** | **9.58** |

**前向渲染**：

| 方法 | Hypersim PSNR↑ | MatrixCity PSNR↑ | InteriorVerse PSNR↑ |
|------|---------------|------------------|---------------------|
| RGB↔X | 16.37 | 9.24 | 13.70 |
| **Ouroboros** | **18.09** | **21.57** | **15.79** |

### 消融实验

| 配置 | 效果 |
|------|------|
| 无循环训练 | 辐照度估计缺乏细节、重建颜色不准确 |
| 有循环训练 | 辐照度更清晰、重建颜色更忠实 |
| 循环训练无 Wild Data | 对高楼光照和表面材质理解不足 |
| 循环训练有 Wild Data | 更真实的辐照度估计、更连续的金属度预测 |
| 循环训练无 e2e Loss | 金属度和辐照度预测不连续 |
| 循环训练有 e2e Loss | 更好的物理属性连续性和材质理解 |

粗糙度和金属度（MatrixCity，PSNR↑）：

| 方法 | Roughness PSNR | Metallicity PSNR |
|------|---------------|-----------------|
| RGB↔X | 23.82 | 6.83 |
| **Ouroboros** | **24.04** | **26.32** |

### 关键发现

- **室外场景提升最为显著**：在 MatrixCity 上，albedo PSNR 从 12.61 提升至 25.38（+12.77），前向渲染 PSNR 从 9.24 提升至 21.57（+12.33），说明循环训练+多域数据极大增强了室外泛化能力
- **金属度估计提升巨大**：从 6.83 提升至 26.32 PSNR，说明 RGB↔X 原模型对金属度的理解严重不足
- **野外数据的价值**：引入无标注真实图像通过循环自监督显著改善了对真实世界光照和材质的理解，这是循环一致性训练的独特优势
- **50× 加速无明显质量损失**：单步推理在大多数指标上反而优于多步的 RGB↔X，说明 E2E 微调策略在渲染任务上也很有效
- **辐照度跨域泛化**：模型仅在 Hypersim（室内）上训练了辐照度估计，但循环训练使其成功泛化到室外场景

## 亮点与洞察

- **循环一致性带来双赢**：不仅改善了一致性本身，更重要的是打开了利用无标注真实数据的通道。在合成标注数据稀缺的渲染领域，这种"用循环结构引入真实数据"的策略非常巧妙
- **单步扩散渲染的可行性**：证明了 E2E 微调技术不仅适用于深度/法线估计等感知任务，也适用于前向渲染这种合成任务，为扩散模型在实时渲染中的应用铺平了道路
- **非确定性单步预测的设计选择**：使用多分辨率噪声而非零噪声，承认了内在分解的多解性，比追求唯一解更合理

## 局限与展望

- **训练数据质量是主要瓶颈**：作者明确指出现有公开数据集（InteriorVerse、Hypersim）的内在属性图不够可靠，且缺乏准确的光照信息
- 未与 DiffusionRenderer（视频扩散方法）在视频分解上做定量比较
- 伪 3D 的视频推理方案在长视频上可能存在误差累积
- 循环训练增加了训练复杂度，需要同时维护两个模型的更新
- 缺少推理速度的具体对比数据（仅声明 50× 加速但未给出具体秒数）

## 相关工作与启发

- **vs RGB↔X**: Ouroboros 的直接前身，在其基础上实现了单步加速、循环一致性、多域训练三大改进，几乎在所有指标上全面超越
- **vs DiffusionRenderer**: 使用视频扩散模型处理视频分解，而 Ouroboros 通过免训练的伪 3D 扩展实现时序一致的视频分解，更轻量
- **vs CycleGAN**: 循环一致性思路的来源，但将其应用在条件扩散模型的渲染任务中是新的尝试，且利用循环结构引入无标注数据是额外的创新

## 评分

- 新颖性: ⭐⭐⭐⭐ 单步扩散渲染+循环一致性的组合是有价值的贡献，但各组件并非全新
- 实验充分度: ⭐⭐⭐⭐ 覆盖室内外多数据集、多种内在属性，消融充分，但缺少与DiffusionRenderer的对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，但部分表述冗余，数学推导可以更简洁
- 价值: ⭐⭐⭐⭐ 50× 加速+质量提升在实际应用中很有价值，开阔了循环训练在渲染领域的应用前景

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Uni-Renderer: Unifying Rendering and Inverse Rendering via Dual Stream Diffusion](../../CVPR2025/image_generation/uni-renderer_unifying_rendering_and_inverse_rendering_via_dual_stream_diffusion.md)
- [\[CVPR 2025\] Channel-wise Noise Scheduled Diffusion for Inverse Rendering in Indoor Scenes](../../CVPR2025/image_generation/channel-wise_noise_scheduled_diffusion_for_inverse_rendering_in_indoor_scenes.md)
- [\[CVPR 2026\] RenderFlow: Single-Step Neural Rendering via Flow Matching](../../CVPR2026/image_generation/renderflow_single-step_neural_rendering_via_flow_matching.md)
- [\[CVPR 2026\] Cycle-Consistent Tuning for Layered Image Decomposition](../../CVPR2026/image_generation/cycle-consistent_tuning_for_layered_image_decomposition.md)
- [\[ICCV 2025\] 3DSR: Bridging Diffusion Models and 3D Representations for 3D Consistent Super-Resolution](bridging_diffusion_models_and_3d_representations_a_3d_consis.md)

</div>

<!-- RELATED:END -->
