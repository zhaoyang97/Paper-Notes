---
title: >-
  [论文解读] Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing
description: >-
  [CVPR 2025][图像恢复][真实世界去雾] IPC-Dehaze 提出了一种基于 VQGAN 码本先验的迭代式 Predictor-Critic 解码框架，通过 Code-Critic 评估码本序列间的相互关联来决定哪些码应保留或重采样，实现了从清晰区域到密集雾区的由易到难渐进去雾，在真实场景中显著超越 SOTA。
tags:
  - "CVPR 2025"
  - "图像恢复"
  - "真实世界去雾"
  - "码本先验"
  - "迭代解码"
  - "GAN"
  - "Predictor-Critic"
---

# Iterative Predictor-Critic Code Decoding for Real-World Image Dehazing

**会议**: CVPR 2025  
**arXiv**: [2503.13147](https://arxiv.org/abs/2503.13147)  
**代码**: 有（论文标注 [Code] [Website]）  
**领域**: 图像复原 / 去雾  
**关键词**: 真实世界去雾, 码本先验, 迭代解码, VQGAN, Predictor-Critic

## 一句话总结

IPC-Dehaze 提出了一种基于 VQGAN 码本先验的迭代式 Predictor-Critic 解码框架，通过 Code-Critic 评估码本序列间的相互关联来决定哪些码应保留或重采样，实现了从清晰区域到密集雾区的由易到难渐进去雾，在真实场景中显著超越 SOTA。

## 研究背景与动机

**领域现状**：真实世界图像去雾是经典的图像复原问题。传统方法依赖手工先验（暗通道先验等），泛化能力差。深度学习方法在合成数据上表现好，但面对真实场景的复杂退化（非均匀雾、色偏、低光）效果不理想。RIDCP 开创性地引入预训练 VQGAN 的高质量码本先验用于去雾，取得了显著进展。

**现有痛点**：现有码本方法采用 one-shot 解码——一次性将雾图 token 匹配到高质量码本。这存在两个问题：(1) 忽视了雾化降质的空间变化性——薄雾区域保留较多信息容易恢复，密集雾区域信息损失严重难以一步到位；(2) 最近邻匹配独立处理每个 token，忽略了码之间的相互依存关系，可能导致相邻区域的码不一致。

**核心矛盾**：one-shot 机制无法利用"已恢复的清晰区域包含对密集雾区的重要线索"这一物理事实。同时，独立采样码的策略忽视了序列间的关联性。

**本文目标** (1) 如何渐进地利用已恢复的高质量码来引导后续的码预测；(2) 如何在每次迭代中决定哪些码应保留、哪些应重新采样。

**切入角度**：借鉴 MaskGIT 的 masked token modeling 思路用于图像复原，但做了关键调整——(1) 用低质量特征作为条件（而非从零生成）；(2) 引入 Code-Critic 评估码间关联性（而非仅看单个码的置信度）。

**核心 idea**：用 Code-Predictor 迭代预测高质量码，用 Code-Critic 评估码间关系来决定保留/重采样，实现由易到难的渐进式去雾。

## 方法详解

### 整体框架

训练分三阶段：(1) 预训练 VQGAN 学习高质量码本（codebook size K=1024, dim=256）；(2) Stage I 训练 Code-Predictor：将雾图和清晰图的 token 按随机 mask 混合，让 Predictor 预测所有位置的高质量码；(3) Stage II 训练 Code-Critic：评估 Predictor 采样的码是否正确。推理时，从雾图 token 出发，经过 T=8 次迭代，每次 Predictor 预测所有码 → Critic 评估并保留最可靠的码 → 下一轮在保留码的引导下重新预测其余位置。

### 关键设计

1. **Code-Predictor（基于 RSTB 的码预测器）**:

    - 功能：从混合了低质量/高质量 token 的输入中预测完整的高质量码序列
    - 核心思路：编码器 $E_L$ 将雾图编码为 $Z_l$，与已恢复的高质量 token $Z_c$ 按 mask $M_t$ 混合得到 $Z_t = Z_l \odot M_t + Z_c \odot (1-M_t)$。Code-Predictor $G_\theta$（4×RSTB 块）以 $Z_t$ 为输入，输出在码本上的概率分布 $p_\theta \in \mathbb{R}^{N \times K}$，用交叉熵损失训练。同时引入 SFT（Spatial Feature Transform）模块在编码器和解码器中对齐雾图和清晰图的特征分布
    - 设计动机：通过随机 mask 混合训练，Predictor 学会了"当部分位置已有高质量码时如何更好地预测其余位置"，这是迭代推理的基础

2. **Code-Critic（码评估器）**:

    - 功能：评估 Predictor 采样的码序列中每个码是否正确，决定保留或重采样
    - 核心思路：以 Predictor 采样的码序列 $S$ 为输入，用 2×RSTB 块输出每个位置的 mask 得分 $p_\phi$。训练标签为 $M = (S \neq S_h)$，即采样码与真实码不一致则应被拒绝。使用 BCE 损失训练。为增加训练多样性，Predictor 的采样温度设为 2 以产生更多错误码供 Critic 学习评估
    - 设计动机：仅凭 Predictor 的输出置信度无法有效判断码的质量，因为这种判断是独立的，不考虑码间关系。Critic 以整个序列为输入，能捕捉码之间的全局关联性（如相邻区域的码应该语义一致），从而做出更合理的保留/拒绝决策

3. **迭代解码调度**:

    - 功能：控制每次迭代保留多少码
    - 核心思路：使用余弦调度函数 $\gamma(t/T)$ 控制每轮保留的码数量。初始时所有位置都需预测（$M_1=1$），随着迭代进行，被保留的码越来越多。Critic 评估后，选择 mask 得分最高的 $\lceil\gamma(t/T) \cdot N\rceil$ 个位置进行重采样，其余保留。T=8 次迭代后所有码都被高质量码替换
    - 设计动机：余弦调度确保早期迭代保留少量最可靠的码（薄雾区域），后期逐步增加，让已恢复的清晰区域自然引导密集雾区的恢复

### 损失函数 / 训练策略

VQGAN 预训练：$\mathcal{L}_{VQGAN} = \mathcal{L}_1 + \mathcal{L}_{code} + \mathcal{L}_{per} + 0.1 \mathcal{L}_{adv}$。Stage I：交叉熵损失 $\mathcal{L}_\theta$。Stage II：BCE 损失 $\mathcal{L}_\phi$。使用 Adam 优化器（lr=1e-4），4 卡 RTX 3090 训练，三阶段分别 400K/100K/20K 迭代。

## 实验关键数据

### 主实验

| 数据集 | 指标 | IPC-Dehaze | RIDCP | KA-Net | 提升(vs RIDCP) |
|--------|------|------------|-------|--------|----------------|
| RTTS | MUSIQ↑ | **59.60** | 55.23 | 54.64 | +4.37 |
| RTTS | Q-Align↑ | **3.49** | 3.24 | 3.09 | +0.25 |
| RTTS | CLIPIQA↑ | **0.44** | 0.30 | 0.28 | +0.14 |
| Fattal | MUSIQ↑ | **66.22** | 65.48 | 64.09 | +0.74 |
| Fattal | Q-Align↑ | **4.234** | 3.799 | 3.982 | +0.435 |
| URHI | MUSIQ↑ | **62.5** | 61.39 | 58.57 | +1.11 |

在 6 个无参考 IQA 指标（MUSIQ、PI、MANIQA、CLIPIQA、Q-Align、TOPIQ）上，IPC-Dehaze 在 RTTS、Fattal、URHI 三个真实数据集上全面有第一或第二。

### 消融实验

| 方法 | MUSIQ↑ | PI↓ | MANIQA↑ | Q-Align↑ | TOPIQ↑ |
|------|--------|-----|---------|----------|--------|
| NN Matching | 58.19 | 3.25 | 0.303 | 3.25 | 0.458 |
| w/o Code-Critic | 57.74 | 3.32 | 0.303 | 3.36 | 0.462 |
| Ours (完整) | **59.60** | **3.22** | **0.327** | **3.49** | **0.500** |

### 关键发现

- Code-Predictor 相比最近邻匹配（NN）在迭代场景中表现更好：NN 增加迭代次数不改变结果（因为是独立匹配），而 Predictor 利用已恢复码能持续改进
- Code-Critic 是性能提升的关键：没有 Critic 时 Predictor 无法有效决定保留哪些码，迭代改进有限；加入 Critic 后 TOPIQ 从 0.462 提升到 0.500
- 可视化显示 Critic 引导的 mask 变化呈现"由近到远、由薄到浓"的规律，验证了由易到难的设计初衷
- 在色偏（沙尘、低光）和密集雾等挑战场景中表现尤为突出

## 亮点与洞察

- **将 MaskGIT 的思想迁移到图像复原是巧妙的**：生成式任务中的"逐步揭示"策略在复原中变成了"从清晰到模糊的渐进修复"，物理直觉很自然——真实世界中雾化确实是空间变化的，薄雾区域的恢复结果可以为密集雾区提供上下文线索
- **Code-Critic 评估码间关联而非独立置信度**：这解决了 MaskGIT 在复原任务中的一个关键缺陷。在图像生成中，独立采样的多样性是优势；但在复原中，码序列必须全局一致才能生成自然图像
- **迭代过程可以即时可视化**：每轮迭代的中间结果清晰展示了去雾进展，有助于理解和调试

## 局限与展望

- 8 次迭代增加了推理时间，对实时应用（如自动驾驶中的去雾）可能不够快
- 码本大小固定为 1024，对极复杂场景的表达能力可能不足
- 依赖 RIDCP 的合成数据生成方法训练 Code-Predictor，合成与真实之间的域差距仍然存在
- Code-Critic 的训练依赖 Code-Predictor 的采样分布，两者存在耦合关系，联合端到端训练可能更优

## 相关工作与启发

- **vs RIDCP**: RIDCP 使用 one-shot 码本匹配，在密集雾和非均匀雾时表现受限；IPC-Dehaze 的迭代机制可以逐步处理不同难度的区域
- **vs MaskGIT**: MaskGIT 设计用于无条件图像生成（HQ→HQ），直接用于 LQ→HQ 复原不合理；IPC-Dehaze 用低质量 token 作为条件并引入 Critic 评估关联性
- **vs KA-Net**: KA-Net 也利用码本先验但仍是 one-shot 方案，在定量指标上整体不如 IPC-Dehaze

## 评分

- 新颖性: ⭐⭐⭐⭐ Predictor-Critic 的迭代解码范式在图像复原中有创新性，但基础构件（VQGAN、RSTB、masked modeling）都是已有的
- 实验充分度: ⭐⭐⭐⭐ 三个真实数据集 + 6 个指标 + 详细消融，但缺少合成数据集上的有参考指标评估
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，算法伪代码规范，可视化对比直观
- 价值: ⭐⭐⭐⭐ 迭代式码本解码的思路可泛化到其他低级视觉任务（超分、去噪等）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] AdcSR: Adversarial Diffusion Compression for Real-World Image Super-Resolution](adversarial_diffusion_compression_for_real-world_image_super-resolution.md)
- [\[ICML 2025\] ε-VAE: Denoising as Visual Decoding](../../ICML2025/image_restoration/epsilon-vae_denoising_as_visual_decoding.md)
- [\[ICCV 2025\] Self-Calibrated Variance-Stabilizing Transformations for Real-World Image Denoising](../../ICCV2025/image_restoration/self-calibrated_variance-stabilizing_transformations_for_real-world_image_denois.md)
- [\[NeurIPS 2025\] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](../../NeurIPS2025/image_restoration/dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [\[ICCV 2025\] IDF: Iterative Dynamic Filtering Networks for Generalizable Image Denoising](../../ICCV2025/image_restoration/idf_iterative_dynamic_filtering_networks_for_generalizable_image_denoising.md)

</div>

<!-- RELATED:END -->
