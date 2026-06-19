---
title: >-
  [论文解读] Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization
description: >-
  [ICML 2025][图像生成][扩散模型加速] MoDiff 提出了调制量化(Modulated Quantization)与误差补偿相结合的框架来加速扩散模型，将激活量化从 8-bit 降至 3-bit 且无性能损失，同时继承缓存和量化方法的双重优势。 领域现状 领域现状：扩散模型在生成任务中表现卓越…
tags:
  - "ICML 2025"
  - "图像生成"
  - "扩散模型加速"
  - "量化"
  - "缓存"
  - "误差补偿"
  - "训练后量化"
---

# Modulated Diffusion: Accelerating Generative Modeling with Modulated Quantization

**会议**: ICML 2025  
**arXiv**: [2506.22463](https://arxiv.org/abs/2506.22463)  
**代码**: [https://github.com/WeizhiGao/MoDiff](https://github.com/WeizhiGao/MoDiff)  
**领域**: Image Generation  
**关键词**: 扩散模型加速, 量化, 缓存, 误差补偿, 训练后量化

## 一句话总结
MoDiff 提出了调制量化(Modulated Quantization)与误差补偿相结合的框架来加速扩散模型，将激活量化从 8-bit 降至 3-bit 且无性能损失，同时继承缓存和量化方法的双重优势。

## 研究背景与动机

### 领域现状

**领域现状**：扩散模型在生成任务中表现卓越，但迭代采样带来巨大计算开销。加速技术主要包括：步数减少（蒸馏）、缓存（复用中间计算）和量化（低精度计算）。

**现有痛点**：缓存方法虽然避免冗余计算，但引入的近似误差会随时间步累积；量化方法在低比特（<8 bit）下生成质量显著下降；两者的组合更容易产生叠加误差。

**核心矛盾**：激进的加速（更低比特/更多缓存）与生成质量之间的根本矛盾。量化误差和缓存误差如何在数学上理解和控制？

**本文目标**：提供一个统一框架，深入分析缓存和量化的误差来源，并设计有效的误差补偿机制。

**切入角度**：从理论分析出发，揭示量化和缓存的内在联系，设计"调制量化"来动态调整量化参数以补偿误差。

**核心 idea**：通过调制量化参数来主动补偿扩散过程中的量化误差，在极低比特下保持生成质量。

## 方法详解

### 整体框架

- **输入**：预训练的扩散模型 + 校准数据
- **过程**：(1) 分析每个时间步的激活分布特征；(2) 为每个时间步设计调制参数；(3) 量化时用调制参数动态调整量化范围和步长
- **输出**：量化后的扩散模型，可用于低比特推理
- 整个过程为训练后量化(PTQ)，无需重新训练

### 关键设计

1. **缓存与量化的统一分析**:

    - 理论证明缓存方法本质上是一种特殊的量化——用前一步的值替代当前步等价于无限粗的量化
    - 揭示了两者共同的误差来源：时间步间激活的变化量
    - **设计动机**：统一视角使得可以设计同时处理两种误差的框架

2. **调制量化(Modulated Quantization)**:

    - 为每个时间步 $t$ 学习一组调制参数 $\gamma_t, \beta_t$
    - 量化前对激活做调制变换：$\hat{x} = \gamma_t \cdot Q(x) + \beta_t$
    - 调制参数通过最小化量化后输出与全精度输出的差距来学习
    - **设计动机**：不同时间步的激活分布差异很大（噪声大小不同），固定量化参数无法适配

3. **误差补偿机制**:

    - 跟踪量化引入的残余误差，在后续步骤中补偿
    - 理论推导误差的上界，确保补偿后的总误差可控
    - 支持与缓存方法组合使用
    - **设计动机**：量化误差会在迭代采样中累积，需要显式的误差控制

### 损失函数 / 训练策略

- 调制参数通过最小化 MSE 损失学习：$\min_{\gamma, \beta} \|f(x) - (\gamma \cdot Q(x) + \beta)\|^2$
- 仅需少量校准数据（几百张图片的前向传播），无需完整重训练
- 支持逐层和逐时间步的参数优化

## 实验关键数据

### 主实验

| 数据集 | 模型 | 比特数 | FID↓ | 与全精度差异 |
|--------|------|--------|------|-------------|
| CIFAR-10 | DDPM | 8-bit | 基本无损 | <0.5 |
| CIFAR-10 | DDPM | 4-bit | 轻微下降 | ~1-2 |
| CIFAR-10 | DDPM | **3-bit** | **无明显下降** | **<1** |
| LSUN | LDM | 8-bit | 基本无损 | <0.5 |
| LSUN | LDM | **3-bit** | **无明显下降** | **<1** |

### 消融实验

| 配置 | FID | 说明 |
|------|-----|------|
| 标准 PTQ (3-bit) | 显著下降 | 无调制的直接量化 |
| MoDiff (3-bit) | 基本无损 | 调制量化有效 |
| 仅缓存 | 小幅下降 | 缓存的近似误差 |
| MoDiff + 缓存 | 最优 | 两者互补 |
| 无误差补偿 | 中等 | 补偿对低比特关键 |

### 关键发现

- **3-bit 量化无损**：在 CIFAR-10 和 LSUN 上，3-bit 即可维持全精度性能
- 调制参数的关键性：不同时间步需要不同的量化策略
- 误差补偿对 <4-bit 场景至关重要
- MoDiff 作为通用框架可加速所有扩散模型

## 亮点与洞察

1. **理论创新**：缓存和量化的统一误差分析框架
2. **极端压缩**：3-bit 激活量化（8x 理论加速 vs FP32）无性能损失
3. **通用性**：PTQ 方法，即插即用，不需要修改预训练模型
4. **有理论保证**：误差上界的推导提供了可靠性保证

## 局限与展望

1. 权重量化未充分探索（本文聚焦激活量化）
2. 实际部署的硬件加速比需要低比特运算支持
3. 在更大模型（如 SDXL, DiT-XL）上的验证不足
4. 调制参数的存储开销需要权衡

## 相关工作与启发

- Q-Diffusion、PTQD 等是扩散模型量化的主要基线
- DeepCache 等缓存方法提供了互补的加速策略
- 启发：动态量化参数的思路可能适用于其他迭代推理模型（如 ARM、MCTS）

## 评分
- 新颖性: ⭐⭐⭐⭐ 调制量化的理论框架有新意
- 实验充分度: ⭐⭐⭐⭐ CIFAR-10 和 LSUN 验证充分
- 写作质量: ⭐⭐⭐⭐ 理论推导清晰
- 价值: ⭐⭐⭐⭐ 对扩散模型部署有实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Efficient Generative Modeling with Residual Vector Quantization-Based Tokens](efficient_generative_modeling_with_residual_vector_quantization-based_tokens.md)
- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[ICCV 2025\] DMQ: Dissecting Outliers of Diffusion Models for Post-Training Quantization](../../ICCV2025/image_generation/dmq_dissecting_outliers_of_diffusion_models_for_post-training_quantization.md)
- [\[ICML 2025\] Compositional Scene Understanding through Inverse Generative Modeling](compositional_scene_understanding_through_inverse_generative_modeling.md)
- [\[ICCV 2025\] SummDiff: Generative Modeling of Video Summarization with Diffusion](../../ICCV2025/image_generation/summdiff_generative_modeling_of_video_summarization_with_diffusion.md)

</div>

<!-- RELATED:END -->
