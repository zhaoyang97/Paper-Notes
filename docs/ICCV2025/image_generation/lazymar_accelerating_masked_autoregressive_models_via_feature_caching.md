---
title: >-
  [论文解读] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching
description: >-
  [ICCV 2025][图像生成][MAR加速] LazyMAR针对Masked Autoregressive（MAR）模型的推理效率瓶颈，利用两种冗余——token冗余（相邻解码步中大部分token特征高度相似）和条件冗余（classifier-free guidance中条件/无条件输出的残差在相邻步间变化极小），设计了token cache和condition cache两种缓存机制，实现2.83×加速且几乎不损失生成质量。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "MAR加速"
  - "特征缓存"
  - "token冗余"
  - "条件冗余"
  - "即插即用"
---

# LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching

**会议**: ICCV 2025  
**arXiv**: [2503.12450](https://arxiv.org/abs/2503.12450)  
**代码**: [https://github.com/feihongyan1/LazyMAR](https://github.com/feihongyan1/LazyMAR)  
**领域**: 图像生成  
**关键词**: MAR加速, 特征缓存, token冗余, 条件冗余, 即插即用

## 一句话总结
LazyMAR针对Masked Autoregressive（MAR）模型的推理效率瓶颈，利用两种冗余——token冗余（相邻解码步中大部分token特征高度相似）和条件冗余（classifier-free guidance中条件/无条件输出的残差在相邻步间变化极小），设计了token cache和condition cache两种缓存机制，实现2.83×加速且几乎不损失生成质量。

## 研究背景与动机

**领域现状**：自回归（AR）图像生成模型（如VQGAN+Transformer）通过逐token生成图像，但序列依赖导致效率低。Masked Autoregressive（MAR）模型（如MaskGIT、MAR）通过并行预测被掩码的token来加速，同时保持甚至超越AR模型的生成质量。

**现有痛点**：MAR模型使用双向注意力（bidirectional attention），虽然可以并行解码多个token，但双向注意力要求同时访问所有token，**无法使用传统的KV Cache**。AR模型的KV Cache之所以有效，是因为causal attention下之前token的KV不变。MAR的双向注意力下每个token的表示都可能因新token的解码而改变，传统KV Cache完全不适用。这导致MAR的推理效率出人意料地不如预期。

**核心矛盾**：MAR的并行解码优势被双向注意力的缓存失效所抵消，急需为MAR设计新的缓存机制。

**本文目标** 为MAR模型设计适配的特征缓存机制，在保持生成质量的前提下大幅加速推理。

**切入角度**：通过分析MAR解码过程中的计算冗余，作者发现两种关键冗余：(a) 大部分token在相邻解码步中的特征几乎不变（token冗余）；(b) classifier-free guidance中条件输出与无条件输出的差值在相邻步间变化极小（条件冗余）。

**核心 idea**：利用MAR解码过程中的token冗余和条件冗余，设计token cache和condition cache实现免训练的即插即用加速。

## 方法详解

### 整体框架
LazyMAR在MAR的迭代解码过程中引入两种缓存机制，配合周期性的"cache-reuse-refresh"策略。在前几步执行完整计算建立缓存；之后的步骤中，跳过相似token的计算（token cache）并跳过无条件分支的计算（condition cache）；每隔K步刷新缓存避免误差累积。整个过程无需训练，即插即用。

### 关键设计

1. **Token冗余分析与Token Cache**:

    - 功能：缓存并复用相邻解码步中特征不变的token
    - 核心思路：
        - MAR每步解码时，token可分为四类（Fig.1）：当前步解码的 ($t$)、上一步刚解码的 ($t-1$)、更早解码的 ($<t-1$)、尚未解码的 ($>t$)
        - 关键发现：$<t-1$ 和 $>t$ 类token的特征在相邻步间余弦距离接近0（几乎不变），只有刚解码的 $t-1$ 类token变化较大
        - 实现：前几步执行完整计算并存储所有token的特征到cache。后续步骤中，只在前3层计算所有token（浅层变化大），深层中根据当前特征与cache中特征的余弦相似度决定是否跳过——相似度高的直接复用cache值
        - 结果：平均可以跳过约84%的token计算
    - 设计动机：已解码且稳定的token和未解码的mask token在相邻步间几乎不变，重新计算它们是浪费

2. **条件冗余分析与Condition Cache**:

    - 功能：跳过classifier-free guidance中无条件分支的完整计算
    - 核心思路：
        - Classifier-free guidance需要同时计算条件输出和无条件输出，然后混合。这使计算量翻倍
        - 关键发现（Fig.2）：条件输出和无条件输出各自在相邻步间变化较大（MSE大），但它们的**残差**（条件-无条件）在相邻步间变化极小（约小35.5×）
        - 实现：缓存上一步的残差。当前步只计算条件分支，用缓存的残差近似无条件分支：$uncond_t \approx cond_t - residual_{t-1}$
        - 结果：跳过一个分支减少50%计算量
    - 设计动机：两个分支的绝对值都在变，但变化方向和幅度高度一致，因此差值几乎不变

3. **周期性Cache-Reuse-Refresh策略**:

    - 功能：定期刷新缓存防止误差累积
    - 核心思路：每隔K个解码步，禁用缓存机制，执行所有token和两个分支的完整计算，用计算结果刷新缓存。这样近似误差不会无限累积
    - 设计动机：迭代复用cache会导致指数级的误差放大。周期性刷新将误差控制在可接受范围内

### 损失函数 / 训练策略
- **无需训练**：LazyMAR完全是推理时的加速策略，不涉及任何训练或微调
- 超参数：cache刷新周期K、跳过token的相似度阈值、前几步不使用cache的warmup步数

## 实验关键数据

### 主实验

| 方法 | FID ↓ | IS ↑ | 加速比 | 说明 |
|------|-------|------|--------|------|
| MAR-Diffusion (原始) | 1.55 | 303.7 | 1.0× | baseline |
| 直接减采样步数 | 1.88 | 289.2 | 2.83× | 质量显著下降 |
| **LazyMAR** | **1.65** | **302.9** | **2.83×** | FID仅增0.1 |

### 消融实验

| 配置 | FID | IS | 加速比 | 说明 |
|------|-----|----|----|------|
| Token Cache only | 1.58 | 303.2 | ~1.8× | token cache贡献主要加速 |
| Condition Cache only | 1.60 | 302.5 | ~1.5× | condition cache贡献50%计算节省 |
| Token + Condition | 1.65 | 302.9 | 2.83× | 两者叠加效果相乘 |
| w/o 周期性刷新 | 2.10 | 285.0 | 更高 | 不刷新误差累积严重 |
| w/o warmup | 1.85 | 295.0 | 略高 | 前几步全计算很重要 |

### 关键发现
- Token cache和condition cache的加速效果可以叠加——token cache跳过84%的token计算（约1.8×），condition cache跳过一个分支（约1.5×），组合达到2.83×
- LazyMAR在2.83×加速下FID仅增0.1（1.55→1.65），而直接减步数在同等加速下FID增0.33
- 周期性刷新对维持质量至关重要，去掉后FID从1.65恶化到2.10
- 前几步的warmup（完整计算）也很关键——这些步骤决定了图像的基本内容
- LazyMAR对所有MAR模型通用（MaskGIT、MAGE、MAR-Diffusion等），无需针对性调整

## 亮点与洞察
- **填补MAR缓存空白**：AR模型有KV Cache，扩散模型有DeepCache/FasterDiffusion，但MAR此前没有对应的缓存加速方案。LazyMAR填补了这一空白
- **条件冗余的发现极具洞察力**：条件/无条件输出各自变化大，但残差极稳定——这个发现本身就很有价值，可能适用于所有使用classifier-free guidance的生成模型
- **免训练即插即用**：不需要任何训练或模型修改，对所有MAR模型通用。实际部署成本极低
- **两种cache正交叠加**：token cache（空间维度冗余）和condition cache（guidance维度冗余）彼此独立，效果相乘而非相加

## 局限与展望
- 2.83×加速虽然不错，但MAR总步数较少（通常~20步），因此绝对时间节省空间有限
- 相似度阈值的选择需要在速度和质量之间权衡
- 周期性刷新引入一定的计算开销
- 目前只在类条件生成上验证，文本条件生成的冗余情况可能不同
- 可以考虑自适应的刷新策略而非固定周期

## 相关工作与启发
- **vs KV Cache（AR模型）**: KV Cache利用causal attention的特性，不适用于MAR的bidirectional attention。LazyMAR从特征相似性角度设计了适用于双向注意力的缓存
- **vs DeepCache（扩散模型）**: DeepCache缓存UNet的上采样块输出，专为UNet架构设计。LazyMAR专为Transformer-based MAR设计
- **vs 直接减步数**: 减步数是最简单的加速方式但质量损失大。LazyMAR在同等加速比下质量远优

## 评分
- 新颖性: ⭐⭐⭐⭐ Token冗余和条件冗余的发现有洞察力，但方法本质是特征缓存的标准思路
- 实验充分度: ⭐⭐⭐⭐ 消融充分，对比了多种baseline和MAR模型
- 写作质量: ⭐⭐⭐⭐⭐ 分析图表（Fig.1,2）非常直观，动机清晰
- 价值: ⭐⭐⭐⭐ 填补MAR缓存空白，即插即用实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] DreamCache: Finetuning-Free Lightweight Personalized Image Generation via Feature Caching](../../CVPR2025/image_generation/dreamcache_finetuning-free_lightweight_personalized_image_generation_via_feature.md)
- [\[NeurIPS 2025\] Predictive Feature Caching for Training-free Acceleration of Molecular Geometry Generation](../../NeurIPS2025/image_generation/predictive_feature_caching_for_training-free_acceleration_of_molecular_geometry_.md)
- [\[ICCV 2025\] From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](from_reusing_to_forecasting_accelerating_diffusion_models_with_taylorseers.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[ICCV 2025\] DC-AR: Efficient Masked Autoregressive Image Generation with Deep Compression Hybrid Tokenizer](dc-ar_efficient_masked_autoregressive_image_generation_with_deep_compression_hyb.md)

</div>

<!-- RELATED:END -->
