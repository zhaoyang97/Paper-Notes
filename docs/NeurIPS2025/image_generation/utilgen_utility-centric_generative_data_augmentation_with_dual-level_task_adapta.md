---
title: >-
  [论文解读] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation
description: >-
  [NeurIPS 2025][图像生成][数据增强] 提出以任务效用为中心的生成式数据增强框架 UtilGen，通过元学习权重网络评估合成数据的下游任务效用，并利用模型级 DPO 和实例级（prompt+noise）双层优化策略，自适应生成高效用的合成训练数据，在8个基准上平均提升3.87%。 现有的生成式数据增强方法主要关…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "数据增强"
  - "任务效用"
  - "扩散模型"
  - "双层优化"
  - "DPO"
---

# UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation

**会议**: NeurIPS 2025  
**arXiv**: [2510.24262](https://arxiv.org/abs/2510.24262)  
**代码**: 暂无  
**领域**: 扩散模型 / 图像生成 / 数据增强  
**关键词**: 数据增强, 任务效用, 扩散模型, 双层优化, DPO

## 一句话总结

提出以任务效用为中心的生成式数据增强框架 UtilGen，通过元学习权重网络评估合成数据的下游任务效用，并利用模型级 DPO 和实例级（prompt+noise）双层优化策略，自适应生成高效用的合成训练数据，在8个基准上平均提升3.87%。

## 研究背景与动机

现有的生成式数据增强方法主要关注合成数据的**内在属性**优化——保真度（fidelity）和多样性（diversity），例如通过 LoRA 微调对齐真实分布，或通过多样化 prompt 增强数据变化。然而，这些方法忽视了一个关键问题：不同的下游任务和模型架构对训练数据的需求截然不同。同一类别中，对某个任务有帮助的样本对另一个任务可能毫无价值。

以往方法缺乏根据下游任务反馈来调整数据生成过程的机制，导致生成的数据虽然视觉质量高，但对特定任务的贡献有限。这促使作者思考：能否从"视觉质量中心"转向**"任务效用中心"**的数据增强范式？

核心挑战在于：（1）如何高效评估合成数据的任务效用而不需要完整的训练-测试循环？（2）如何系统地提升合成数据的任务效用？

## 方法详解

### 整体框架

UtilGen 包含三个核心模块：（1）任务导向数据估值（TODV）通过元学习权重网络量化每个样本的效用；（2）模型级生成能力优化（MLCO）用 DPO 微调扩散模型以适应下游任务偏好；（3）实例级生成策略优化（ILPO）同时优化 prompt 嵌入和初始噪声以最大化单样本效用。

### 关键设计

1. **任务导向数据估值（TODV）**：使用一个单隐层 MLP 权重网络 $\mathcal{W}_\phi$ 预测每个样本的效用权重：$\omega_i = \mathcal{W}_\phi(\mathcal{L}(f(x_i;\theta), y_i))$。通过双层优化训练：内层用加权损失训练分类器 $\theta$，外层最小化验证集损失来优化权重网络 $\phi$。训练好的权重网络可以直接对新生成的样本预测效用分数，避免了昂贵的重训练。核心动机是利用元学习建立"合成数据质量→下游性能"的快速评估通道。

2. **模型级生成能力优化（MLCO）**：利用权重网络将生成样本分为高效用和低效用配对，构建偏好数据集 $\mathcal{D}_{\text{preference}}$，然后用 Diffusion DPO 微调扩散模型的 U-Net。DPO 损失为：$\mathcal{L}_{\text{DPO}}(\psi) = -\mathbb{E}[\log\sigma(-\beta T\omega(\lambda_t)(\Delta\mathcal{L}_w - \Delta\mathcal{L}_l))]$，其中 $\Delta\mathcal{L}_w$ 和 $\Delta\mathcal{L}_l$ 分别为高效用和低效用样本的噪声预测差异。通过迭代 DPO 微调，逐步使扩散模型的生成分布对齐下游任务需求。

3. **实例级生成策略优化（ILPO）**：在每次生成时进行细粒度优化。（a）**Prompt 嵌入优化**：在 textual inversion 学到的类标识符基础上，梯度优化 prompt 嵌入以最大化权重网络预测的效用分数，同时加 CLIP 正则化防止语义偏移：$p^* = \arg\max_p[\mathcal{W}_\phi(\mathcal{L}(f(g(p,\epsilon_T);\theta),y)) - \lambda L_{\text{CLIP}}]$。（b）**噪声优化**：利用 DDIM 正反向过程中 CFG scale 的不对称性，将高效用数据的语义信息隐式注入初始噪声：$\epsilon'_t = \text{DDIM-Inv}_{\omega_w}(\text{DDIM}_{\omega_l}(\epsilon_t, p^*))$，其中 $\omega_l > \omega_w$ 实现语义注入。

### 损失函数 / 训练策略

整体训练流程为迭代式：先训练 TODV 得到效用评估器 → 用 MLCO 迭代微调扩散模型 → 每轮生成时用 ILPO 优化 prompt 和噪声。数据估值阶段使用真实+合成混合数据集进行双层优化。MLCO 和 ILPO 的优化信号均来自训练好的权重网络 $\mathcal{W}_\phi$。

## 实验关键数据

### 主实验

在8个基准数据集上用 ResNet-50 进行分类评估，合成数据量为真实数据的5倍：

| 设置 | 方法 | IN-1k-S | IN-100-S | Cal101 | Flowers | 平均 |
|------|------|---------|----------|--------|---------|------|
| 仅合成数据 | DataDream (前SOTA) | 30.35 | 35.48 | 23.61 | 65.15 | 33.30 |
| 仅合成数据 | **UtilGen** | **33.72** | **40.94** | **29.31** | **67.43** | **37.17** |
| 合成+真实 | DataDream | 52.16 | 57.68 | 73.38 | 89.60 | 58.67 |
| 合成+真实 | **UtilGen** | **54.56** | **61.54** | **75.62** | **93.62** | **62.04** |

平均提升：仅合成 +3.87%，联合训练 +3.37%。

### 消融实验

| 配置 | 准确率 (%) | 说明 |
|------|-----------|------|
| Baseline (SD v2.1) | 27.96 | 无任何优化 |
| +MLCO | 28.68 | 模型级优化 |
| +Prompt Opt | 36.42 | prompt 嵌入优化贡献最大 |
| +Noise Opt | 37.96 | 噪声优化也有独立增益 |
| +MLCO+Prompt+Noise (全) | **40.94** | 三者互补，总提升 +12.98% |

### 关键发现

- UtilGen 是首个仅用3倍合成数据训练的 ResNet-50 在多个基准上超越真实数据训练的方法
- 数据影响力分析显示 UtilGen 生成的正面影响样本比例显著高于 SD v2.1
- 合成数据具有跨架构可复用性：用 ResNet-50 训练的权重网络生成的数据对 WideResNet 和 CLIP 同样有效
- 跨架构泛化：在 ResNeXt-50、WideResNet-50、MobileNetV2 上均保持领先

## 亮点与洞察

- 极具启发性的范式转变：从优化数据的视觉属性转向优化数据对任务的效用，这是数据增强领域的重要思维转变
- 元学习权重网络的双重用途设计巧妙：既能在分类器训练中加权样本，又能为数据生成提供效用评估信号
- ILPO 中噪声优化的设计借鉴了 DDIM 逆过程中 CFG scale 不对称性的发现，实现了零额外训练成本的语义注入

## 局限与展望

- 权重网络的效用评估依赖于分类器的当前状态，当分类器能力较弱时效用评估可能不准确
- TODV 需要预先训练权重网络，增加了整体流程的复杂度和多阶段训练的超参数调节负担
- 当前仅验证了分类任务，对检测、分割等更复杂任务的效用评估方式可能需要重新设计
- 成本分析显示生成5万张图仅需4.7小时/$100，但 DPO 迭代微调的计算开销未详细讨论
- 实验中 textual inversion 使用16-shot 真实图像（与 DataDream 一致），对极低资源场景（如1-2 shot）的表现有待验证
- 噪声优化依赖 CFG scale 不对称性这一特定技巧，在非 DDIM 采样器上的适用性不确定

## 相关工作与启发

- 与 Data Shapley 等数据估值方法的区别在于避免了昂贵的重训练，使用轻量权重网络实现在线估值
- GAP 方法也使用下游模型反馈但仅基于对抗性损失（最大化分类器损失），本文的效用权重提供了更精细的"样本级"有用性信号
- 与 DataDream 的核心区别：DataDream 通过 LoRA 微调提升保真度对齐真实分布，UtilGen 则通过 DPO 微调对齐下游任务偏好，两者优化目标本质不同
- 可启发未来将效用导向的思想扩展到其他生成任务（文本增强、3D 数据增强等）
- 成本效益分析显示生成合成数据的成本远低于人工标注（$100 vs $800 for 同等规模），且质量更优

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 任务效用中心的范式转变具有重要意义，首次用 DPO+效用反馈驱动增强
- **实验充分度**: ⭐⭐⭐⭐⭐ 8个数据集、多架构、消融完整、影响力分析深入
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，方法描述详尽
- **价值**: ⭐⭐⭐⭐⭐ 为数据增强提供了新范式，实验提升显著且具有实际应用价值

<!-- NeurIPS 2025 | image_generation | 2510.24262 -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Non-Asymptotic Analysis of Data Augmentation for Precision Matrix Estimation](non-asymptotic_analysis_of_data_augmentation_for_precision_matrix_estimation.md)
- [\[CVPR 2026\] OntoAug: Rethinking Generative Data Augmentation via Ontology Guidance](../../CVPR2026/image_generation/ontoaug_rethinking_generative_data_augmentation_via_ontology_guidance.md)
- [\[NeurIPS 2025\] Large-Scale Training Data Attribution for Music Generative Models via Unlearning](large-scale_training_data_attribution_for_music_generative_models_via_unlearning.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[NeurIPS 2025\] Aligning Compound AI Systems via System-level DPO](aligning_compound_ai_systems_via_system-level_dpo.md)

</div>

<!-- RELATED:END -->
