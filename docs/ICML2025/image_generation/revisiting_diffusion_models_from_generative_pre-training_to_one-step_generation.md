---
title: >-
  [论文解读] Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation
description: >-
  [ICML 2025][图像生成][扩散蒸馏] 提出将扩散模型训练视为"生成式预训练"的新视角，发现蒸馏中师生模型收敛到不同局部最优的根本局限，证明仅用 GAN 目标（无需蒸馏损失）即可将预训练扩散模型高效转换为单步生成器（D2O），且冻结 85% 参数的微调版本（D2O-F）仅需 0.2M 图像即可达到强竞争力结果。
tags:
  - "ICML 2025"
  - "图像生成"
  - "扩散蒸馏"
  - "单步生成"
  - "GAN"
  - "生成式预训练"
  - "参数冻结"
---

# Revisiting Diffusion Models: From Generative Pre-training to One-Step Generation

**会议**: ICML 2025  
**arXiv**: [2506.09376](https://arxiv.org/abs/2506.09376)  
**代码**: 无  
**领域**: 扩散模型  
**关键词**: 扩散蒸馏, 单步生成, GAN微调, 生成式预训练, 参数冻结

## 一句话总结

提出将扩散模型训练视为"生成式预训练"的新视角，发现蒸馏中师生模型收敛到不同局部最优的根本局限，证明仅用 GAN 目标（无需蒸馏损失）即可将预训练扩散模型高效转换为单步生成器（D2O），且冻结 85% 参数的微调版本（D2O-F）仅需 0.2M 图像即可达到强竞争力结果。

## 研究背景与动机

扩散模型在图像/视频生成中展现了超越 GAN 的质量，但推理需要多步迭代采样，计算开销极大。为加速采样，扩散蒸馏（Diffusion Distillation）被广泛采用，将多步过程压缩为少步或单步模型。然而现有蒸馏方法存在两大问题：

**训练代价高**：需要大量计算资源和数据进行蒸馏训练

**性能退化**：学生模型往往无法匹配教师模型的生成质量

近期研究发现在蒸馏过程中加入 GAN 损失能显著提升效果，但背后的机制尚不清楚。本文旨在从理论和实验两方面揭示蒸馏的根本限制，并提出一种更优雅的替代方案。

**核心洞察**：蒸馏方法中教师模型的多步推理与学生模型的单步推理导致两者收敛到不同的局部最优解，逐实例对齐（instance-level imitation）本质上是次优的。GAN 目标通过对齐分布而非对齐个体样本，天然绕过了这一限制。

## 方法详解

### 整体框架

本文提出 **D2O（Diffusion to One-Step）** 方法，核心思路分三步：

1. **理论分析**：揭示蒸馏中师生模型"局部最优不一致"的根本问题
2. **D2O 基线**：用预训练扩散模型初始化生成器，仅用 GAN 目标函数微调（无蒸馏损失），将多步扩散模型转化为单步生成器
3. **D2O-F（Frozen）**：进一步冻结 85% 参数（所有卷积层），仅微调归一化层和 skip connection，验证"生成式预训练"假说

### 关键设计

#### 1. 蒸馏局限性的理论分析

作者从 EDM 框架出发，定义了 ODE 求解器：

$$\mathbf{S}_\phi(x_{t_i}, t_i, t_{i-1}) = \frac{t_i - t_{i-1}}{t_i}(\mathbf{g}_\phi(x_{t_i}, t_i) - x_{t_i}) + x_{t_i}$$

在渐进蒸馏（Progressive Distillation）中，教师模型需要多次通过神经网络，而学生模型仅通过一次。这引入了关键的归纳偏差：

- **教师模型**：可在像素空间和潜在空间之间多次变换，参数量等效更多
- **学生模型**：仅能执行一次变换，参数空间受限

作者通过实验验证了这一假设：用 FID 度量不同步数（2/4/6/8/10步）教师模型与单步学生模型之间的差异：
- 对训练集的 FID 相近（都能生成高质量图像）
- 但师生之间的 FID 随教师步数增加而显著增大
- 即使最接近的 2 步教师，与学生之间仍有 FID=1.78 的差距

这证明教师和学生以"不同方式"达到了类似性能，逐实例模仿是次优的。

#### 2. 纯 GAN 目标的 D2O 模型

基于上述分析，D2O 完全抛弃蒸馏损失，仅使用 non-saturating GAN 目标：

- **判别器目标**：$\max_{\mathbf{D}} \mathbb{E}_\mathbf{x}[\log(\mathbf{D}(\mathbf{x}))] + \mathbb{E}_\mathbf{z}[\log(1 - \mathbf{D}(G(\mathbf{z})))]$
- **生成器目标**：$\min_{G} -\mathbb{E}_\mathbf{z}[\log(\mathbf{D}(G(\mathbf{z})))]$

关键设计要点：
- 生成器直接用预训练扩散模型的 U-Net 初始化
- 输入为纯高斯噪声（对应最大噪声水平 $t_{max}$），输出为单步去噪生成的图像
- 使用真实图像（而非教师输出）作为判别器的正样本，直接推动学生逼近真实数据分布
- 无需教师模型的在线推理，避免了蒸馏中的多步前向传播开销

#### 3. D2O-F 参数冻结策略

为验证"扩散训练 = 生成式预训练"的假说，D2O-F 在微调时冻结大部分参数：

- **冻结层（蓝色，约 85%）**：所有卷积层参数保持不变
- **可训练层（红色，约 15%）**：仅微调归一化层（GroupNorm）和 skip connection
- 这一设计的理论依据是：卷积层在扩散预训练中已学会通用的生成式特征变换能力，微调只需调整特征的缩放/偏移和跨层信息流

### 损失函数 / 训练策略

- **初始化**：生成器从 EDM 预训练权重初始化；判别器随机初始化
- **数据需求极低**：D2O 用 5M 图像达到近 SOTA；D2O-F 仅用 0.2M 图像即展现强性能
- **训练效率**：由于大量参数冻结，D2O-F 的可训练参数极少，收敛速度快、显存占用低
- **频域分析**：作者进一步从频域角度解释扩散预训练为何赋予模型单步生成能力——扩散训练过程中，模型逐步学会不同频率分量的生成，低噪声步骤学低频、高噪声步骤学高频，预训练后模型已具备从噪声中一次性恢复全频谱的潜力

## 实验关键数据

### 主实验：ImageNet 64×64 单步生成 FID 对比

| 方法 | 类型 | 训练图像数 | FID ↓ |
|------|------|-----------|-------|
| EDM (63 steps) | 多步扩散 | — | 2.44 |
| Progressive Distillation | 蒸馏 | 大量 | ~3.0+ |
| Consistency Distillation | 蒸馏 | 大量 | ~3.5+ |
| 各种 GAN+蒸馏方法 | 混合 | 大量 | ~2.0-2.5 |
| **D2O (本文)** | **纯 GAN** | **5M** | **≤2.2** |
| **D2O-F (本文)** | **纯 GAN + 冻结** | **0.2M** | **有竞争力** |

### 消融实验：师生模型局部最优不一致性验证

| 教师步数 | vs 训练集 FID | vs 单步学生 FID |
|---------|-------------|---------------|
| 2 步 | ~2.0 | 1.78 |
| 4 步 | ~2.0 | ~3.5 |
| 6 步 | ~2.1 | ~5.0 |
| 8 步 | ~2.1 | ~6.5 |
| 10 步 | ~2.2 | ~8.0 |

> 关键发现：教师步数越多，与学生的分布差异越大，尽管二者对训练集的 FID 相近。这直接支持了"局部最优不一致"的假说。

### 关键发现：CIFAR-10 基线消融

| 设置 | FID ↓ | 说明 |
|------|-------|------|
| 纯蒸馏（无 GAN） | 较高 | 需要大量数据仍难匹配 |
| 蒸馏 + GAN | 中等 | GAN 损失带来显著增益 |
| **纯 GAN（D2O）** | **最优** | 无需蒸馏损失即达最佳 |
| 随机初始化 + GAN | 差 | 需数千万图像，证明预训练的价值 |

## 亮点与洞察

1. **视角创新**："扩散训练即生成式预训练"是一个优雅且有解释力的新视角。类比 NLP 中的 pre-train → fine-tune 范式，扩散训练教会模型通用的降噪/生成能力，下游任务（如单步生成）只需轻量微调
2. **极简设计**：D2O 去掉了蒸馏方法中所有复杂组件（教师模型在线推理、渐进步数缩减、特殊损失函数），仅保留最简单的 GAN 目标，却取得了更好的效果。这是奥卡姆剃刀的典范应用
3. **数据效率惊人**：D2O-F 仅需 0.2M 图像即可工作，比传统蒸馏方法减少 1-2 个数量级的数据需求，这强有力地证明了预训练权重中已编码了丰富的生成能力
4. **冻结实验有说服力**：85% 参数冻结后性能不降反升，这不仅验证了假说，也暗示扩散模型的卷积特征是高度可复用的通用生成基底
5. **频域解释有深度**：从频率分解角度解释扩散预训练如何赋予全频谱生成能力，弥合了理论与实践之间的鸿沟

## 局限与展望

1. **仅在像素空间验证**：实验主要在 ImageNet 64×64 和 CIFAR-10 上进行，未扩展到更高分辨率（如 256×256 或 512×512）或潜在空间扩散模型（LDM/SDXL），泛化性有待验证
2. **GAN 训练不稳定性**：虽然用预训练初始化缓解了 GAN 训练的不稳定问题，但在更大规模或更复杂数据上是否仍稳定有待观察
3. **架构限制**：仅在 U-Net 架构上验证，未探索 DiT（Diffusion Transformer）等新型架构，而后者已成为主流
4. **条件生成**：主要展示了类别条件生成（class-conditional），未涉及文本条件生成等更实用的场景
5. **理论深度可加强**：局部最优不一致的分析主要基于 FID 实证，缺乏更严格的理论证明（如优化景观分析或收敛性证明）
6. **缺乏感知质量评估**：未报告 IS（Inception Score）、CLIP-FID 等互补指标，也缺少人类感知评估

## 相关工作与启发

- **扩散蒸馏**：Progressive Distillation (Salimans & Ho, 2022)、Consistency Models (Song et al., 2023)、Guided Distillation (Meng et al., 2023) 等都需要蒸馏损失，本文证明可以完全去掉
- **GAN + 扩散混合**：SDXL-Turbo (Sauer et al., 2024)、UFOGen (Xu et al., 2023)、DMD (Yin et al., 2024) 等在蒸馏基础上叠加 GAN 损失，本文进一步简化为纯 GAN
- **扩散模型作为预训练**：与 DreamBooth、ControlNet 等"扩散模型微调"工作思路相通，但本文首次明确提出"生成式预训练"的统一视角，并用参数冻结实验提供了直接证据
- **对后续研究的启发**：这一视角可能推动(1)用更大规模扩散预训练再 GAN 微调的两阶段范式；(2)探索哪些层编码了哪些生成能力（类似 NLP 中的 probing）；(3)将冻结微调策略推广到视频/3D 生成

## 评分

- 新颖性: ⭐⭐⭐⭐ "扩散训练即预训练"视角新颖，纯 GAN 目标足矣的洞察有力，但 GAN+扩散的组合并非全新
- 实验充分度: ⭐⭐⭐ 核心论证清晰但规模较小（64×64），缺乏高分辨率和 LDM/DiT 实验
- 写作质量: ⭐⭐⭐⭐ 逻辑清晰，从问题发现→机制分析→方法提出→实验验证的叙事流畅
- 价值: ⭐⭐⭐⭐ 低数据需求和参数冻结策略有实际意义，新视角对后续研究有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](../../CVPR2026/image_generation/duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[CVPR 2025\] OSDFace: One-Step Diffusion Model for Face Restoration](../../CVPR2025/image_generation/osdface_one-step_diffusion_model_for_face_restoration.md)
- [\[ICML 2025\] Task-Agnostic Pre-training and Task-Guided Fine-tuning for Versatile Diffusion Planner](task-agnostic_pre-training_and_task-guided_fine-tuning_for_versatile_diffusion_p.md)
- [\[CVPR 2025\] Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization](../../CVPR2025/image_generation/aesthetic_post-training_diffusion_models_from_generic_preferences_with_step-by-s.md)
- [\[CVPR 2026\] PixelRush: Ultra-Fast, Training-Free High-Resolution Image Generation via One-step Diffusion](../../CVPR2026/image_generation/pixelrush_ultrafast_trainingfree_highresolution_im.md)

</div>

<!-- RELATED:END -->
