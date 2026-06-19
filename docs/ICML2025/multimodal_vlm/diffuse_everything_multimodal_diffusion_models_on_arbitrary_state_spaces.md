---
title: >-
  [论文解读] Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces
description: >-
  [ICML 2025][多模态VLM][多模态扩散模型] 提出了一个在任意状态空间上构建多模态扩散模型的统一框架，通过为每种模态引入独立的解耦噪声调度（decoupled noise schedule），在单个模型中同时实现无条件生成和模态条件生成，无需外部的 tokenizer 或 VAE 预处理。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "多模态扩散模型"
  - "混合状态空间"
  - "解耦噪声调度"
  - "文本-图像联合生成"
  - "表格数据合成"
---

# Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces

**会议**: ICML 2025  
**arXiv**: [2506.07903](https://arxiv.org/abs/2506.07903)  
**代码**: [https://github.com/kevinjrojas/DiffuseEverything](https://github.com/kevinjrojas/DiffuseEverything)  
**领域**: 扩散模型 / 多模态生成  
**关键词**: 多模态扩散模型, 混合状态空间, 解耦噪声调度, 文本-图像联合生成, 表格数据合成

## 一句话总结
提出了一个在任意状态空间上构建多模态扩散模型的统一框架，通过为每种模态引入独立的解耦噪声调度（decoupled noise schedule），在单个模型中同时实现无条件生成和模态条件生成，无需外部的 tokenizer 或 VAE 预处理。

## 研究背景与动机

**领域现状**：扩散模型在单模态数据（图像、视频、文本）生成上取得了巨大成功，DDPM、Score-based 模型、Flow Matching 等方法已成为主流。

**现有痛点**：现有多模态扩散方法严重依赖外部预处理协议。例如，要联合生成文本和图像，通常需要先将文本通过 tokenizer 离散化，再通过 VAE 将图像编码到连续潜空间，最后在统一的格式上做扩散。这种 pipeline 对编码器/解码器的精度要求极高。

**核心矛盾**：文本是离散的（categorical），图像是连续的（Gaussian），两者的状态空间本质不同。传统方法试图将所有模态强制映射到同一个状态空间，这引入了额外的信息损失，且在数据有限时编码器质量难以保证。

**本文目标**：如何在不依赖外部编码器/解码器的情况下，直接在不同模态的原始状态空间上构建联合扩散模型？

**切入角度**：从扩散过程的数学框架出发，将连续扩散（Gaussian）和离散扩散（categorical）统一在一个通用的马尔可夫过程框架下，通过为每个模态设计独立的噪声调度来协调不同模态的扩散速率。

**核心 idea**：引入 **解耦噪声调度**（decoupled noise schedule），让每种模态以自己的节奏添加和去除噪声，从而在单个模型中自然地处理混合状态空间数据。

## 方法详解

### 整体框架

输入为多模态耦合数据（如文本-图像对，或混合类型表格数据），每种模态在其原生状态空间中定义各自的前向扩散过程。模型在反向过程中联合去噪，输出各模态的生成结果。

核心思路：
- 连续模态（如图像像素/潜变量）使用 **高斯扩散过程**
- 离散模态（如文本 token）使用 **分类扩散过程**（uniform-state 或 absorbing-state）
- 两者共享去噪网络但使用独立的时间调度

### 关键设计

1. **统一的多模态扩散框架**:

    - 将连续和离散扩散统一描述为一个广义的马尔可夫噪声过程
    - 对于连续模态 $x_c$，前向过程为 $q_t(x_c|x_0) = \mathcal{N}(\alpha_t^c x_0, \sigma_t^c \mathbf{I})$
    - 对于离散模态 $x_d$，前向过程为 $q_t(x_d|x_0) = \text{Cat}(\alpha_t^d x_0 + (1-\alpha_t^d)\pi)$
    - 关键在于 $\alpha_t^c$ 和 $\alpha_t^d$ 可以独立设置
    - **设计动机**：不同模态的信息密度和复杂度差异巨大，统一的噪声调度会导致一种模态过早完成扩散而另一种还未开始有效去噪

2. **解耦噪声调度（Decoupled Noise Schedule）**:

    - 为每种模态定义独立的 signal-to-noise ratio (SNR) 曲线
    - 连续模态使用 $\text{SNR}_c(t) = \alpha_t^{c2} / \sigma_t^{c2}$
    - 离散模态使用对应的 retention probability $\alpha_t^d$
    - 通过独立调整各模态的调度参数，可以控制每种模态的去噪难度和生成质量
    - **为什么这样设计**：文本的离散空间远小于图像的连续空间，如果使用相同的噪声调度，文本会在很早的时间步就被完全破坏，导致模型无法学习有效的文本去噪

3. **条件生成与无条件生成的统一**:

    - 利用解耦调度，可以将某一模态的噪声水平设为 0（即保持该模态为干净数据），而只对另一模态做扩散
    - 这自然地实现了条件生成：给定文本生成图像（text-to-image），或给定图像生成文本（image-to-text）
    - 无条件生成则是所有模态同时从噪声开始
    - **设计动机**：避免训练多个独立模型，单个模型通过调整噪声调度即可切换生成模式

### 损失函数 / 训练策略

- 总体损失为各模态去噪损失的加权和：

$$\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon} \left[ \lambda_c(t) \|\hat{x}_c - x_c\|^2 + \lambda_d(t) \text{CE}(\hat{x}_d, x_d) \right]$$

- 连续模态使用 MSE 损失（标准去噪目标）
- 离散模态使用交叉熵损失
- 权重 $\lambda_c(t)$ 和 $\lambda_d(t)$ 根据各模态的 SNR 自适应调整
- 训练时随机采样时间步 $t$，各模态根据各自的调度获得不同的噪声水平

## 实验关键数据

### 主实验：文本-图像联合生成

| 方法 | 图像 FID ↓ | 文本 PPL ↓ | 联合生成质量 |
|------|-----------|-----------|-------------|
| 本文（解耦调度） | **竞争力水平** | **竞争力水平** | 最佳 |
| 统一调度基线 | 较差 | 较差 | 两模态互相干扰 |
| 独立模型 | 好 | 好 | 无法联合生成 |
| Tokenizer+扩散 | 受限于 tokenizer 质量 | 受限于解码质量 | 依赖外部模块 |

### 消融实验：解耦调度的影响

| 配置 | 图像质量 | 文本质量 | 说明 |
|------|---------|---------|------|
| 完全解耦调度 | ✓ 最佳 | ✓ 最佳 | 各模态独立优化噪声节奏 |
| 统一调度（同步） | ✗ 较差 | ✗ 较差 | 文本被过早破坏，图像去噪不足 |
| 仅图像解耦 | ✓ 好 | △ 中等 | 文本仍受限 |
| 仅文本解耦 | △ 中等 | ✓ 好 | 图像仍受限 |

### 关键发现
- 解耦噪声调度是实现高质量多模态联合生成的关键，统一调度导致严重的模态间干扰
- 该框架同样适用于混合类型表格数据（包含连续列和分类列），在表格数据合成任务上达到有竞争力的性能
- 单模型即可实现无条件生成、单模态条件生成等多种模式，无需针对每种任务单独训练

## 亮点与洞察
- **理论优雅**：将连续和离散扩散统一在同一框架下，解耦调度是一个简洁而有效的设计
- **无需外部依赖**：摆脱了对 tokenizer/VAE 的依赖，直接在原生状态空间上操作
- **灵活性强**：框架可推广到任意数量和类型的模态组合
- **表格数据应用**：展示了扩散模型在非视觉多模态任务上的适用性

## 局限与展望
- 目前主要在相对小规模的数据集上验证，尚未在大规模文生图（如 LAION 级别）上测试
- 解耦调度的超参数选择需要针对具体模态组合进行调优
- 图像生成质量与专用的单模态扩散模型（如 SDXL、DALL-E 3）相比仍有差距
- 未探索更多模态（如音频、视频）的联合生成

## 相关工作与启发
- **Discrete Diffusion**：D3PM、MDLM 等离散扩散模型为文本模态提供了理论基础
- **Multimodal Generation**：UniDiffuser 等工作也尝试了多模态联合扩散，但依赖 VAE 等外部模块
- **启发**：解耦调度的思想可以推广到更多混合状态空间场景，如分子生成（连续坐标 + 离散原子类型）

## 评分
- 新颖性: ⭐⭐⭐⭐ 解耦噪声调度是一个简洁优雅的创新，统一框架有理论价值
- 实验充分度: ⭐⭐⭐ 验证了文本-图像和表格数据两个场景，但规模较小
- 写作质量: ⭐⭐⭐⭐ 数学推导清晰，框架描述完整
- 价值: ⭐⭐⭐⭐ 为多模态原生扩散提供了新的思路，对混合状态空间建模有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Grounding Everything in Tokens for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/grounding_everything_in_tokens_for_multimodal_large_language_models.md)
- [\[CVPR 2025\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](../../CVPR2025/multimodal_vlm/molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)
- [\[CVPR 2026\] Sparse-LaViDa: Sparse Multimodal Discrete Diffusion Language Models](../../CVPR2026/multimodal_vlm/sparse-lavida_sparse_multimodal_discrete_diffusion_language_models.md)
- [\[ICML 2025\] Robust Multimodal Large Language Models Against Modality Conflict](robust_multimodal_large_language_models_against_modality_conflict.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](ranked_from_within_ranking_large_multimodal_models_without_labels.md)

</div>

<!-- RELATED:END -->
