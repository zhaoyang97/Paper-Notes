---
title: >-
  [论文解读] Visual Generation Without Guidance
description: >-
  [ICML 2025][图像生成][Classifier-Free Guidance] 提出 Guidance-Free Training (GFT)，通过重新参数化条件模型为采样网络与无条件网络的线性插值，直接从数据训练出无需引导的视觉生成模型，在 DiT/VAR/LlamaGen/MAR/LDM 五种模型上匹配 CFG 性能的同时将采样计算量减半。
tags:
  - "ICML 2025"
  - "图像生成"
  - "Classifier-Free Guidance"
  - "无引导生成"
  - "采样效率"
  - "伪温度参数化"
  - "蒸馏替代"
---

# Visual Generation Without Guidance

**会议**: ICML 2025  
**arXiv**: [2501.15420](https://arxiv.org/abs/2501.15420)  
**代码**: [https://github.com/thu-ml/GFT](https://github.com/thu-ml/GFT)  
**领域**: 扩散模型 / 图像生成  
**关键词**: Classifier-Free Guidance, 无引导生成, 采样效率, 伪温度参数化, 蒸馏替代

## 一句话总结
提出 Guidance-Free Training (GFT)，通过重新参数化条件模型为采样网络与无条件网络的线性插值，直接从数据训练出无需引导的视觉生成模型，在 DiT/VAR/LlamaGen/MAR/LDM 五种模型上匹配 CFG 性能的同时将采样计算量减半。

## 研究背景与动机
**领域现状**：CFG 是视觉生成的标配技术，通过在采样时同时运行条件和无条件模型来提升生成质量，但推理计算量直接翻倍。

**现有痛点**：(a) 推理成本翻倍；(b) 复杂化后训练流程（蒸馏、RLHF 时需特殊处理无条件模型）；(c) 与 LLM 中简单温度采样的方式不一致。

**核心矛盾**：CFG 的采样分布 $p^s(\boldsymbol{x}|c) \propto p(\boldsymbol{x}|c)[p(\boldsymbol{x}|c)/p(\boldsymbol{x})]^s$ 没有对应的真实数据集，无法直接最大似然训练。

**本文目标** 能否用单一模型实现 CFG 的质量-多样性权衡？

**切入角度**：重新排列 CFG 公式，将条件模型表示为采样模型和无条件模型的加权组合，直接学采样模型。

**核心 idea**：CFG 的 $\epsilon^c = \frac{1}{1+s}\epsilon^s + \frac{s}{1+s}\epsilon^u$，直接优化 $\epsilon^s$，无需引导即可采样。

## 方法详解

### 整体框架
GFT 保持与 CFG 相同的最大似然训练目标，但对条件模型做不同的参数化：将条件模型定义为采样网络 $\epsilon_\theta^s$ 和无条件网络 $\epsilon_\theta^u$ 的隐式线性组合。引入伪温度 $\beta = 1/(1+s)$ 作为模型的额外输入，允许推理时灵活调节。

### 关键设计

1. **隐式条件参数化**:

    - 功能：让训练直接优化采样模型 $\epsilon_\theta^s$ 而非条件模型 $\epsilon_\theta^c$
    - 核心思路：$\epsilon_\theta^c(\boldsymbol{x}_t|\boldsymbol{c},\beta) = \beta \epsilon_\theta^s(\boldsymbol{x}_t|\boldsymbol{c},\beta) + (1-\beta) \epsilon_\theta^u(\boldsymbol{x}_t)$，用标准条件 loss 训练这个隐式表示
    - 设计动机：虽然 $p^s$ 没有数据集不能直接学 $\epsilon^s$，但 $\epsilon^c$ 是可学的，而 $\epsilon^s$ 可以通过它间接优化

2. **停止梯度技巧**:

    - 功能：提高训练效率和稳定性
    - 核心思路：无条件模型 $\epsilon_\theta^u$ 在 eval 模式下运行并 stop-gradient，只对 $\epsilon_\theta^s$ 反传梯度
    - 设计动机：(a) 与 CFG 训练高度对齐，仅差一次无条件推理；(b) 几乎不增加显存；(c) 仅增加 19% 训练时间

3. **伪温度 $\beta$ 输入**:

    - 功能：让单一模型支持不同温度采样
    - 核心思路：随机采样 $\beta \sim U(0,1)$，用 Fourier embedding + MLP 处理后加到 time/class embedding 上
    - 设计动机：$\beta=1$ 时等价于标准条件生成，$\beta \to 0$ 时趋近低温高质量采样

### 损失函数 / 训练策略
- Diffusion 版本：$\mathcal{L} = \|\beta\epsilon_\theta^s(\boldsymbol{x}_t|\boldsymbol{c}_\varnothing,\beta) + (1-\beta)\mathbf{sg}[\epsilon_\theta^u(\boldsymbol{x}_t|\varnothing,1)] - \boldsymbol{\epsilon}\|_2^2$
- AR/Masked 版本：条件 logits $\ell_\theta^c = \beta \ell_\theta^s + (1-\beta)\mathbf{sg}[\ell_\theta^u]$，再计算标准交叉熵
- 微调预训练 CFG 模型只需 1-5% 的预训练 epoch，零初始化 $\beta$ 的 final MLP 以不影响初始输出

## 实验关键数据

### 主实验

| 模型 | CFG FID ↓ | GFT FID ↓ | GFT 微调/从头 |
|------|-----------|-----------|--------------|
| DiT-XL/2 | 2.11 | **1.99** | 微调 2% epoch |
| DiT-XL/2 (蒸馏) | 2.11 | - | - |
| VAR-d30 | - | 匹配 CFG | 从头训练 |
| LlamaGen | - | 匹配 CFG | 从头训练 |
| MAR | - | 匹配 CFG | 从头训练 |
| LDM (T2I) | - | 匹配 CFG | 微调 |

### 消融实验

| 方法 | 适用域 | 训练额外时间 | 显存增加 | 可从头训练 |
|------|--------|------------|---------|-----------|
| Guidance Distillation | 仅 Diffusion | ×1.19 | ×1.15 | ✗ |
| Contrastive Alignment | 仅 AR/Masked | ×1.69 | ×1.39 | ✗ |
| **GFT (Ours)** | **全部** | **×1.00** | **×1.00** | **✓** |

### 关键发现
- GFT 微调 DiT-XL 仅需 2% epoch 即可达到 FID 1.99，优于 CFG 的 2.11
- GFT 是唯一支持从头训练 + 通用于 diffusion/AR/masked 三类模型的方法
- 通过调节 $\beta$ 可实现与 CFG 相当的多样性-保真度权衡

## 亮点与洞察
- **极简的实现**：基于现有 CFG 代码仅需改几行，大部分超参数直接继承
- **统一性**：首次统一了 diffusion、AR、masked 模型的无引导训练方法
- **定理保证**：Theorem 1 证明了 GFT 的最优解与理想 CFG 采样分布一致

## 局限与展望
- 作者承认 $\beta$ 的采样分布（均匀分布）可能不是最优的
- 大规模 T2I 生成（如 SDXL 级别）的验证尚不充分
- 与 consistency models 等加速方法的结合未探索

## 评分
- 新颖性: ⭐⭐⭐⭐ 参数化视角很巧妙，但核心只是 CFG 公式变形
- 实验充分度: ⭐⭐⭐⭐⭐ 五种模型 + 类条件/文本条件全覆盖
- 写作质量: ⭐⭐⭐⭐⭐ 推导清晰，实验对比公平
- 价值: ⭐⭐⭐⭐⭐ 实用性极强，有望成为替代 CFG 的标准方法

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Learning Visual Generative Priors without Text](../../CVPR2025/image_generation/learning_visual_generative_priors_without_text.md)
- [\[ICCV 2025\] StyleKeeper: Prevent Content Leakage using Negative Visual Query Guidance](../../ICCV2025/image_generation/stylekeeper_prevent_content_leakage_using_negative_visual_query_guidance.md)
- [\[ICML 2025\] Continuous Visual Autoregressive Generation via Score Maximization](continuous_visual_autoregressive_generation_via_score_maximization.md)
- [\[CVPR 2025\] Rectified Diffusion Guidance for Conditional Generation](../../CVPR2025/image_generation/rectified_diffusion_guidance_for_conditional_generation.md)
- [\[ICLR 2026\] SSG: Scaled Spatial Guidance for Multi-Scale Visual Autoregressive Generation](../../ICLR2026/image_generation/ssg_scaled_spatial_guidance_for_multi-scale_visual_autoregressive_generation.md)

</div>

<!-- RELATED:END -->
