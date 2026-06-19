---
title: >-
  [论文解读] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution
description: >-
  [AAAI 2026][图像生成][图像超分辨率] 将稀疏混合专家（MoE）思想引入真实世界图像超分辨率任务，提出 Mixture-of-Ranks（MoR）架构，将 LoRA 的每个 rank 视为独立专家，并设计退化估计模块和退化感知负载均衡损失，实现单步高保真超分辨率重建。 领域现状 真实世界图像超分辨率（Real-I…
tags:
  - "AAAI 2026"
  - "图像生成"
  - "图像超分辨率"
  - "混合专家"
  - "LoRA"
  - "退化感知"
  - "单步扩散"
---

# Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution

**会议**: AAAI 2026  
**arXiv**: [2511.16024](https://arxiv.org/abs/2511.16024)  
**代码**: 无  
**领域**: 图像生成 / 图像超分辨率  
**关键词**: 图像超分辨率, 混合专家, LoRA, 退化感知, 单步扩散

## 一句话总结

将稀疏混合专家（MoE）思想引入真实世界图像超分辨率任务，提出 Mixture-of-Ranks（MoR）架构，将 LoRA 的每个 rank 视为独立专家，并设计退化估计模块和退化感知负载均衡损失，实现单步高保真超分辨率重建。

## 研究背景与动机

### 领域现状

真实世界图像超分辨率（Real-ISR）是计算机视觉中的关键任务，需要从包含模糊、噪声等复杂退化的低分辨率图像中恢复高分辨率细节。扩散模型凭借强大的生成先验已成为 Real-ISR 的主流方案，但其多步迭代采样过程带来巨大的计算延迟。近期的单步超分方法（如 OSEDiff、S3Diff）通过蒸馏或 LoRA 微调预训练扩散模型实现了加速，但仍面临根本性限制。

### 现有痛点

**密集模型难以处理异构退化**：Real-ISR 中样本的退化类型（模糊、噪声）和退化程度差异极大，但现有的密集（dense）LoRA 微调方法对所有输入使用相同的参数，无法自适应地捕捉异构退化特征。

**计算资源分配不灵活**：简单退化的样本和复杂退化的样本使用相同的计算预算，既浪费了简单样本上的计算资源，又限制了对复杂样本的处理能力。

**知识共享不足**：不同退化类型之间存在共性知识（如基本纹理恢复），但密集模型无法在等效计算预算下实现知识共享。

### 核心矛盾

如何在固定计算预算下，让模型根据输入退化的严重程度和类型**动态分配计算资源**，对简单样本少计算、对复杂样本多计算？

### 核心 Idea

受 DeepSeek 等 LLM 中稀疏 MoE 架构的启发，将 MoE 概念引入 LoRA 微调中。**不是将整个 LoRA 模块作为专家，而是将每个 rank 视为独立专家**（细粒度专家分割），结合 CLIP 驱动的退化估计模块为路由提供退化感知信号，并通过零专家（zero-expert）机制和退化感知负载均衡损失实现动态计算资源分配。

## 方法详解

### 整体框架

MoR-DASR 基于 Stable Diffusion 2.1，训练流程交替两个阶段：
1. 优化变分得分网络 $\epsilon_\psi$ 拟合生成样本分布（扩散损失 $\mathcal{L}_{diff}$）
2. 微调带有 MoR 模块的扩散模型 $\epsilon_\theta$ 和编码器 $Enc_\theta$（通过重建损失 $\mathcal{L}_{rec}$、VSD 损失 $\mathcal{L}_{VSD}$ 和 GAN 损失 $\mathcal{L}_{GAN}$）

推理时，LR 图像直接输入编码器，经过一步扩散网络前向传播即可生成 HR 图像。

### 关键设计

#### 1. **Mixture-of-Ranks (MoR) 架构**

**功能**：将 LoRA 中的每个 rank 视为独立专家，实现细粒度的专家分割与知识重组。

**核心思路**：传统 LoRA MoE 将完整的 LoRA 模块（$A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times d}$）作为一个专家单元，这种粗粒度策略限制了知识的灵活分解和重组。MoR 将每个 rank（$A_i, B_i$，rank=1）作为独立专家，分为两类：

- **共享专家（Shared Experts）**：固定位置的 rank，始终激活，捕捉通用知识（如基本纹理恢复、去噪基础操作）
- **路由专家（Routed Experts）**：通过门控机制和 top-k 选择策略动态激活的 rank

前向传播公式：
$$MoR(z_t) = W_0 x + \sum_{i=1}^{k} g_i(s) B_i A_i x + \sum_{j=1}^{m} B_j A_j x$$

其中 $g_i(s)$ 为路由权重，$k$ 为激活的路由专家数，$m$ 为共享专家数。

**设计动机**：类比 DeepSeek-MoE 的两大创新——细粒度专家分割和共享专家隔离。细粒度使得知识可以更灵活地组合，共享专家捕捉跨退化类型的通用特征并减少路由冗余。

**配置**：40 个 rank 总计，8 个共享专家 + 32 个路由专家，训练时 top-8 路由策略。

#### 2. **退化估计模块（Degradation Estimation Module）**

**功能**：利用 CLIP 的跨模态对齐能力计算输入图像的退化程度分数，用于指导路由器的专家选择。

**核心思路**：定义多维度的正/负文本 prompt 对（整体质量、模糊度、噪声、分辨率、边缘清晰度、细节等 7 个维度），通过 CLIP 图像编码器和文本编码器计算 LR 图像与正/负 prompt 的余弦相似度，归一化后得到退化分数：

$$s_i = \frac{\exp(d^{(i,n)})}{\exp(d^{(i,p)}) + \exp(d^{(i,n)})}$$

其中 $d^{(i,p)}$ 和 $d^{(i,n)}$ 分别是图像与第 $i$ 维度正/负 prompt 的余弦距离。退化越严重，$s_i$ 越大。

**设计动机**：语义特征驱动的数据分割不适用于 Real-ISR（文献已证明退化特性更关键）。退化严重的图像需要更多计算资源来重建，退化轻微的图像则不需要，因此退化程度是动态计算分配的最佳依据。

#### 3. **零专家与退化感知负载均衡损失**

**功能**：引入"零专家"（输出恒为零的虚拟专家）和改进的负载均衡损失，实现根据退化程度动态调整激活专家数量。

**核心思路**：在 32 个路由专家中加入若干零专家槽位。传统负载均衡损失对零专家和真实专家一视同仁，会给零专家分配约 $k/N$ 的固定激活概率。改进的退化感知负载均衡损失：

$$\mathcal{L}_{balance} = N \sum_{i=1}^{N} \alpha_i f_i p_i$$

其中：
$$\alpha_i = \begin{cases} \alpha & \text{if } i \leq n \text{ (真实专家)} \\ s \cdot \alpha & \text{if } i > n \text{ (零专家)} \end{cases}$$

退化分数 $s$ 越大（退化越严重），零专家的惩罚权重越大 → 模型倾向激活更多真实专家；$s$ 越小，零专家惩罚减小 → 模型激活更多零专家节省计算。

**设计动机**：解决两个问题：(1) 不同退化程度的样本需要不同计算预算；(2) 网络不同层所需的最优 LoRA rank 可能不同。零专家使得计算资源可以自适应分配。

### 损失函数 / 训练策略

总损失函数：
$$\mathcal{L}_\theta = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{VSD} + \lambda_2 \mathcal{L}_{GAN} + \mathcal{L}_{balance}$$

- $\mathcal{L}_{rec}$：L2 + LPIPS 重建损失
- $\mathcal{L}_{VSD}$：变分得分蒸馏损失（对齐预训练扩散模型的高质量先验）
- $\mathcal{L}_{GAN}$：利用预训练扩散模型特征的多尺度判别器损失
- $\mathcal{L}_{balance}$：退化感知负载均衡损失

训练细节：LSDIR + 10k FFHQ 人脸图像，Real-ESRGAN 退化管线，batch size 16，学习率 5e-5，25000 次迭代。

## 实验关键数据

### 主实验

与单步 Real-ISR 方法的定量对比（三个数据集）：

| 数据集 | 方法 | PSNR↑ | LPIPS↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ | TOPIQ↑ | TRES↑ |
|--------|------|-------|--------|----------|--------|---------|--------|-------|
| DIV2K-Val | OSEDiff | 23.92 | 0.310 | 0.659 | 67.71 | 0.435 | 0.606 | 78.40 |
| DIV2K-Val | S3Diff | 23.53 | **0.258** | 0.699 | 67.92 | 0.452 | 0.633 | 80.72 |
| DIV2K-Val | **MoR-DASR** | **24.01** | 0.289 | 0.681 | **68.09** | **0.475** | **0.663** | **84.14** |
| RealSR | OSEDiff | 25.26 | 0.301 | 0.651 | 68.41 | 0.468 | 0.614 | 80.18 |
| RealSR | **MoR-DASR** | 25.32 | 0.291 | **0.691** | **69.78** | **0.512** | **0.662** | **84.97** |
| DRealSR | OSEDiff | **28.29** | 0.302 | 0.673 | 64.47 | 0.469 | 0.616 | 76.76 |
| DRealSR | **MoR-DASR** | 28.37 | 0.307 | **0.717** | **65.94** | **0.509** | **0.652** | **81.78** |

与多步 Real-ISR 方法对比（关键结果）：

| 数据集 | 方法 | 步数 | CLIPIQA↑ | MANIQA↑ | TRES↑ |
|--------|------|------|----------|---------|-------|
| DIV2K-Val | SeeSR | 50 | 0.693 | **0.504** | **85.80** |
| DIV2K-Val | MoR-DASR | **1** | 0.681 | 0.475 | 84.14 |
| RealSR | SeeSR | 50 | 0.669 | **0.540** | **88.60** |
| RealSR | MoR-DASR | **1** | **0.691** | 0.512 | 84.97 |

MoR-DASR 单步推理即可达到 SeeSR 50 步的 comparable 质量，速度提升约 **40×**。

### 消融实验

MoR 架构消融（在 DRealSR 上）：

| 配置 | CLIPIQA↑ | MANIQA↑ | TRES↑ | 说明 |
|------|----------|---------|-------|------|
| LoRA（基线） | 0.670 | 0.481 | 78.81 | 标准 LoRA 微调 |
| LoRA+MoE | 0.689 | 0.484 | 79.36 | 传统 MoE，整个 LoRA 作为专家 |
| MoR-v1 | 0.704 | 0.491 | 80.32 | MoR 无零专家 |
| MoR-v2 | 0.699 | 0.479 | 79.41 | MoR 有零专家但无退化感知均衡 |
| **MoR-full** | **0.717** | **0.509** | **81.78** | 完整方案 |

关键观察：
- LoRA→LoRA+MoE：验证 MoE 在 Real-ISR 中有效
- LoRA+MoE→MoR-v1：细粒度 rank 级专家优于模块级专家
- MoR-v1→MoR-v2：仅引入零专家但用传统均衡损失反而**性能下降**
- MoR-v2→MoR-full：退化感知均衡损失是零专家生效的关键，最终 CLIPIQA +7%、MANIQA +5.8%、TRES +3.8%

### 关键发现

1. **退化自适应激活**：退化程度越高，激活的零专家越少（即真实专家越多），符合直觉。
2. **层间计算差异**：不同网络层所需的 rank 数量不同，某些层全部激活零专家（该层不需要 LoRA），某些层不激活任何零专家，验证了 MoR 细粒度的必要性。
3. **感知指标优势突出**：MoR-DASR 在 MANIQA、TOPIQ、TRES 等感知指标上表现最佳，这些指标与人眼感知高度相关。

## 亮点与洞察

1. **将 LLM 领域的 MoE 成功迁移到视觉任务**：DeepSeek MoE 的细粒度分割和共享专家隔离思想在 Real-ISR 中同样有效。
2. **退化感知路由是核心创新**：不是用语义特征路由，而是用退化程度路由，这比 image content-based routing 更适合 Real-ISR。
3. **零专家机制优雅**：通过引入输出为零的虚拟专家，实现了"动态 rank"效果，不同层不同样本自适应选择计算量。
4. **单步推理效率**：相比多步方法（SeeSR 50步），速度提升 40× 同时质量相当。

## 局限与展望

1. **退化 prompt 设计手工化**：CLIP 退化估计的正/负 prompt 对是人工设计的（7个维度），可能无法覆盖所有退化类型。
2. **仅验证在 SD 2.1 上**：未在更新的 SDXL 或 SD 3.0 上验证 MoR 的效果。
3. **训练开销**：相比标准 LoRA 微调，MoR 的路由机制和退化估计模块增加了训练复杂度。
4. **专家数量和 top-k 配置**：40 ranks、top-8 的配置是通过经验确定的，缺少系统的超参数分析。

## 相关工作与启发

- **DeepSeek-MoE**：直接启发了 MoR 的细粒度专家分割和共享专家隔离设计。
- **OSEDiff**：提供了 LR-to-HR 直接映射的单步框架基础。
- **CLIP for IQA**：退化估计模块借鉴了 CLIP-IQA 中利用 CLIP 跨模态能力评估图像质量的思路。
- **对其他任务的启发**：MoR 思想可推广到其他需要处理异构输入的视觉任务，如去模糊、去噪、去雨等图像恢复任务。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — MoE 与 LoRA rank 的结合以及退化感知路由设计新颖
- **实验充分度**: ⭐⭐⭐⭐ — 三个数据集、多种对比方法、详细消融和可视化分析
- **写作质量**: ⭐⭐⭐⭐ — 方法动机清晰，图示直观  
- **价值**: ⭐⭐⭐⭐ — 在单步超分方向取得显著进步，MoR 思路可推广

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[AAAI 2026\] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution](continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[ECCV 2024\] AdaDiffSR: Adaptive Region-Aware Dynamic Acceleration Diffusion Model for Real-World Image Super-Resolution](../../ECCV2024/image_generation/adadiffsr_adaptive_region-aware_dynamic_acceleration_diffusion_model_for_real-wo.md)
- [\[CVPR 2026\] Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution](../../CVPR2026/image_generation/bridging_fidelity-reality_with_controllable_one-step_diffusion_for_image_super-r.md)

</div>

<!-- RELATED:END -->
