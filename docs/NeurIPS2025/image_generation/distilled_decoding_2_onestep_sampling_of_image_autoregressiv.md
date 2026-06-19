---
title: >-
  [论文解读] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation
description: >-
  [NeurIPS 2025][图像生成][自回归模型加速] 本文提出Distilled Decoding 2（DD2），通过将自回归图像模型重新解读为条件分数模型，设计了条件分数蒸馏（CSD）损失，将多步AR采样压缩为一步生成，在ImageNet-256上实现FID从3.40到5.43的微小退化同时获得8.0x加速（VAR）和238x加速（LlamaGen），相比DD1缩小了67%的性能差距且训练快12.3倍。
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "自回归模型加速"
  - "一步生成"
  - "分数蒸馏"
  - "条件分数蒸馏"
---

# Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2510.21003](https://arxiv.org/abs/2510.21003)  
**代码**: [GitHub](https://github.com/imagination-research/Distilled-Decoding-2)  
**领域**: 图像生成 / 模型加速  
**关键词**: 自回归模型加速, 一步生成, 分数蒸馏, 条件分数蒸馏, 图像生成

## 一句话总结

本文提出Distilled Decoding 2（DD2），通过将自回归图像模型重新解读为条件分数模型，设计了条件分数蒸馏（CSD）损失，将多步AR采样压缩为一步生成，在ImageNet-256上实现FID从3.40到5.43的微小退化同时获得8.0x加速（VAR）和238x加速（LlamaGen），相比DD1缩小了67%的性能差距且训练快12.3倍。

## 研究背景与动机

**领域现状**：图像自回归（AR）模型（如VAR、LlamaGen）已经达到了最先进的图像生成质量，甚至超越了VAE、GAN和扩散模型。这些模型通过逐个预测离散token来生成图像，保持了良好的likelihood建模特性。

**现有痛点**：AR模型的核心瓶颈在于逐步采样——VAR需要10步，LlamaGen需要256步。每步都需要完整的模型前向传播，导致生成速度远慢于GAN等一步生成方法。集合预测方法（如MaskGIT式的一次预测多个token）在极端情况（一步生成）下会完全丢失token间的相关性。推测解码（speculative decoding）在图像AR上仅能实现不到3倍的有限压缩。DD1是第一个实现一步采样的方法，但依赖预定义的noise→data映射，这个映射难以学习，导致FID退化明显（VAR-d20从3.40到9.55），且训练较慢。

**核心矛盾**：DD1将AR模型的每步采样转化为flow matching的ODE求解来构建确定性映射，本质上只用了分数信号来建映射。但这种预定义映射（1）对模型学习困难，（2）限制了灵活性——GAN、VAE等不依赖显式映射的模型反而在下游任务中适用性更广。

**本文目标** 能否训练一个一步生成器，让其输出分布匹配给定的AR模型，而不依赖任何预定义映射？

**切入角度**：将AR模型重新解读为条件分数模型——给定前面所有token后，AR模型输出的next-token概率向量可以解析地计算codebook embedding空间中的条件score。然后借鉴扩散模型score distillation的思想，匹配生成器和teacher AR模型在每个token位置上的条件分数。

**核心 idea**：把AR模型的next-token概率向量当作条件score的来源，通过条件分数蒸馏（CSD）在所有token位置上同时对齐生成器与teacher的条件分布，实现无需预定义映射的一步AR图像生成。

## 方法详解

### 整体框架

整个训练分为两个阶段：（1）初始化阶段：将teacher AR模型的分类头替换为MLP头，用GTS loss微调成AR-diffusion模型；（2）CSD训练阶段：用该AR-diffusion模型初始化generator和guidance network，交替训练generator（用CSD loss）和guidance network（用FCS loss）。推理时，generator一次前向传播输出整个token序列。

### 关键设计

1. **Teacher AR模型作为条件分数模型**:

    - 功能：将离散概率向量转化为连续的score信号
    - 核心思路：将token $q_i$的采样视为从"加权Dirac函数之和→Gaussian"的flow matching过程。给定teacher输出的概率向量$p = (p_1,...,p_V)$和flow matching时间步$t$，条件score可以解析计算为：$s(x_t, t, p) = -\frac{\sum_j p_j(x_t - (1-t)c_j)e^{-\frac{(x_t-(1-t)c_j)^2}{2t^2}}}{t^2 \sum_j p_j e^{-\frac{(x_t-(1-t)c_j)^2}{2t^2}}}$。与DD1不同，DD2不用这个score去构建ODE映射，而是直接用于蒸馏
    - 设计动机：AR模型在每个token位置已经隐式包含了完整的条件分数信息，DD1只用了这个信息的一部分（构建映射），DD2要更充分地利用

2. **条件分数蒸馏损失（CSD Loss）**:

    - 功能：训练一步generator，使其输出序列的联合分布匹配teacher AR模型
    - 核心思路：在每个token位置$i$上，对齐teacher的条件score $s_\Phi(q_i^{t_i}, t_i | q_{<i})$和guidance network学到的fake条件score $s_{\text{fake}}(q_i^{t_i}, t_i | q_{<i})$。CSD loss是所有位置的score distillation loss之和：$\mathcal{L}_{CSD} = \mathbb{E}\sum_{i=1}^n d(s_\Phi(q_i^{t_i}, t_i | sg(q_{<i})), s_{\text{fake}}(q_i^{t_i}, t_i | sg(q_{<i})))$，其中$sg(\cdot)$表示stop gradient。采用SiD loss形式。关键理论保证（Proposition 1）：最小化CSD loss → generator的联合分布匹配teacher
    - 设计动机：渐进对齐逻辑——先对齐$q_1$的分布（无条件），再在此基础上对齐$q_2|q_1$，依此类推。这比对齐整个联合分布更容易优化

3. **AR-diffusion初始化策略**:

    - 功能：为generator和guidance network提供良好的初始化
    - 核心思路：直接用teacher AR模型的权重初始化不可行，因为teacher输出离散概率而generator需要输出连续值。解决方案：将teacher的分类头替换为MLP，用Ground Truth Score (GTS) loss微调：$\mathcal{L}_{GTS} = \mathbb{E}\sum_{i=1}^n \|s_\psi(q_i^{t_i}, t_i | q_{<i}) - s_\Phi(q_i^{t_i}, t_i | q_{<i})\|^2$，即直接回归teacher的解析score。微调后的模型同时作为generator和guidance network的初始化
    - 设计动机：实验证明没有良好初始化score distillation会完全collapse。即使只随机初始化generator的最后一层，性能也会严重退化。GTS loss比标准AR-diffusion loss收敛更快更稳定

### 损失函数 / 训练策略

- Generator训练：CSD loss（Eq. 4），采用SiD loss形式
- Guidance network训练：FCS loss（Eq. 5），标准的score matching loss
- 两个网络交替训练
- GTS初始化阶段做全量微调，CSD阶段的训练远短于DD1

## 实验关键数据

### 主实验

ImageNet-256上的一步生成质量对比：

| 方法 | 模型 | FID↓ | IS↑ | 步数 | 加速比 |
|------|------|------|-----|------|--------|
| VAR-d20原始 | 600M | 3.40 | 305.1 | 10 | 1x |
| DD1 | VAR-d20 | 9.55 | 197.2 | 1 | - |
| **DD2** | **VAR-d20** | **5.43** | **233.7** | **1** | **8.0x** |
| LlamaGen-L原始 | 343M | 4.11 | 283.5 | 256 | 1x |
| DD1 | LlamaGen-L | 11.35 | 193.6 | 1 | - |
| **DD2** | **LlamaGen-L** | **8.59** | **229.1** | **1** | **238x** |

DD2在所有config下1步FID优于DD1的2步结果。

### 消融实验

| 配置 | FID (LlamaGen) | FID (VAR-d24) | 说明 |
|------|---------------|---------------|------|
| Gui-Init ✓ Gen-Init ✓ | 14.77 | 11.53 | 完整初始化（最佳） |
| Gui-Init ✓ Gen-Init ✗ | 16.08 | >200 (collapse) | 缺Generator初始化 |
| Gui-Init ✗ Gen-Init ✓ | 21.76 | >200 (collapse) | 缺Guidance初始化 |

训练效率对比：

| 方法 | 模型 | GPU小时(8xA800) | 训练加速比 |
|------|------|----------------|-----------|
| DD1 | LlamaGen-L | 647.7 | 1x |
| DD2 | LlamaGen-L | 52.6 | **12.3x** |
| DD1 | VAR-d24 | 604.2 | 1x |
| DD2 | VAR-d24 | 96.1 | **6.3x** |

### 关键发现

- **DD2大幅超越DD1**：VAR上缩小了teacher和one-step之间67%的性能差距
- **初始化至关重要**：缺少任一组件初始化会导致collapse（VAR-d24上FID>200），即使只随机初始化最后一层也会严重退化
- **训练效率极高**：最多12.3倍训练加速，因为不需要像DD1那样预先构建所有noise-data映射
- **PPL更低**：DD2的Perceptual Path Length（7231.9 vs 18437.6）远低于DD1，说明DD2学到了更光滑的latent space
- **多步采样锦上添花**：DD2 3步（4.88 FID）就优于DD1 6步（5.03 FID），提供灵活的质量-速度trade-off

## 亮点与洞察

- **AR → score model的重新解读极其巧妙**：原本AR模型输出的是next-token的离散概率，本文把它视为连续embedding空间中的score信号。这个视角的转换使得扩散蒸馏的整套理论和工具可以直接迁移到AR加速中。这种跨paradigm的知识迁移值得学习
- **"消除预定义映射"的核心洞察**：DD1的mapping是hard target（必须精确学到每个noise→data的对应），DD2的CSD是soft target（只需分布匹配），后者显然更容易优化。打个比方：DD1是学填空的唯一答案，DD2是学整个答案的分布
- **初始化策略的通用价值**：GTS loss（用teacher的解析score代替Monte Carlo估计）收敛更快更稳定，这个技巧可推广到其他score distillation设置

## 局限与展望

- **仍有性能差距**：VAR-d20从3.40到5.43仍有2.03的FID退化，对质量要求极高的场景可能不够
- **仅验证了VQ-based AR模型**：对MAR等continuous-space AR模型的兼容性仅做了理论讨论，未实验验证
- **仅在ImageNet-256上评测**：未扩展到文本→图像等更实用的场景
- **训练仍需8×A800 GPU**：虽然比DD1快12x，绝对成本仍然不低
- **初始化的强依赖是双刃剑**：高度依赖GTS初始化，如果teacher模型quality差，可能影响DD2效果

## 相关工作与启发

- **vs DD1**：DD1用flow matching构建确定性映射然后拟合，DD2用score distillation做分布匹配。DD2在性能（-67%性能差距）、训练速度（12.3x）、latent smoothness（PPL减半）上全面超越
- **vs 扩散Score Distillation（DMD/SiD）**：虽然都做score matching，但AR模型和diffusion model的生成过程完全不同——AR是顺序条件生成，diffusion是iterative denoising。DD2的创新在于将score distillation从独立变量扩展到具有AR依赖的序列变量
- **vs Set-of-token预测（MaskGIT/VAR）**：一步预测所有token会丢失token间的相关性（如{(0,0),(1,1)}的例子），DD2通过CSD在所有位置做条件score对齐来保持相关性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将score distillation引入AR模型加速，AR→score model的重新解读非常有创见
- 实验充分度: ⭐⭐⭐⭐ 两种AR模型（VAR+LlamaGen）、多种size、完整的消融和对比
- 写作质量: ⭐⭐⭐⭐ 动机和理论推导清晰，但第3节公式密度较高
- 价值: ⭐⭐⭐⭐⭐ 对AR图像生成加速具有里程碑意义，可能改变AR vs Diffusion的速度对比格局

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient](../../CVPR2025/image_generation/collaborative_decoding_makes_visual_auto-regressive_modeling_efficient.md)
- [\[CVPR 2025\] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](../../CVPR2025/image_generation/hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)
- [\[ICCV 2025\] SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation](../../ICCV2025/image_generation/sana-sprint_one-step_diffusion_with_continuous-time_consistency_distillation.md)
- [\[CVPR 2026\] WaDi: Weight Direction-aware Distillation for One-step Image Synthesis](../../CVPR2026/image_generation/wadi_weight_direction-aware_distillation_for_one-step_image_synthesis.md)
- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](../../CVPR2025/image_generation/mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)

</div>

<!-- RELATED:END -->
