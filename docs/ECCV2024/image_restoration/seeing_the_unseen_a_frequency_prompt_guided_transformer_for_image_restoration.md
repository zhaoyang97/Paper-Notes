---
title: >-
  [论文解读] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration
description: >-
  [ECCV 2024][图像恢复][提示学习] 提出 FPro，通过频域视角的 prompt learning 指导图像复原：使用 Gated Dynamic Decoupler 将特征解耦为低频/高频分量，再通过 Dual Prompt Block（HPM + LPM）分别对两个频带注入可学习 prompt 并与解码器特征交互，在去雨、去雨滴、去摩尔纹、去模糊、去雾 5 个任务上全面超越 SOTA。
tags:
  - "ECCV 2024"
  - "图像恢复"
  - "提示学习"
  - "Gated Dynamic Decoupler"
  - "Transformer"
---

# Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration

**会议**: ECCV 2024  
**arXiv**: [2404.00288](https://arxiv.org/abs/2404.00288)  
**代码**: [https://github.com/joshyZhou/FPro](https://github.com/joshyZhou/FPro)  
**领域**: 图像复原 / Low-level Vision  
**关键词**: image restoration, Frequency Prompt, Dual Prompt Block, Gated Dynamic Decoupler, Transformer

## 一句话总结

提出 FPro，通过频域视角的 prompt learning 指导图像复原：使用 Gated Dynamic Decoupler 将特征解耦为低频/高频分量，再通过 Dual Prompt Block（HPM + LPM）分别对两个频带注入可学习 prompt 并与解码器特征交互，在去雨、去雨滴、去摩尔纹、去模糊、去雾 5 个任务上全面超越 SOTA。

## 研究背景与动机

**领域现状**：基于深度学习的图像复原方法（Restormer、Uformer、SwinIR 等）取得了显著进展，Transformer 架构凭借全局特征捕获能力成为主流。近期 Prompt Learning 被引入图像复原领域（PromptIR、DA-CLIP 等），通过编码退化信息来调制网络。

**现有痛点**：
1. 现有 prompt 方法（PromptIR, DA-CLIP）专注于空间域的退化信息挖掘，忽略了频域线索——不同退化（雨纹、雨滴、摩尔纹）影响不同频带
2. Self-attention 本质是低通滤波器，会稀释高频信息（纹理、边缘），现有 Transformer 复原模型难以充分利用高频细节
3. 不同退化类型在频域表现迥异：雨纹部分遮挡背景（高频），雨滴造成大范围区域遮挡（低频），但现有方法未针对性处理

**核心矛盾**：空间域 prompt 无法捕获退化的频率特性差异，导致恢复图像残留频域中细微或不可察觉的伪影。

**本文切入角度**：从频率视角设计 prompt 机制——将特征解耦到不同频带，分别注入 prompt 组件来编码退化特有信息。

**核心 idea**：频域 prompt learning——在低频和高频分量上分别生成和调制 prompt，用双分支架构处理全局结构和局部细节。

## 方法详解

### 整体框架

FPro 包含两个分支：(1) 上方的 Restoration Branch——标准 encoder-decoder 结构负责图像复原；(2) 下方的 Prompt Branch——从输入浅层特征中提取频率信息并生成 prompt，与 decoder 特征交互。

**输入**：退化图像 $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$  
**流程**：3×3 Conv 提取浅层特征 $\mathbf{F}_s$ → 同时送入 Restoration Branch（N1=3 级 encoder-decoder）和 Prompt Branch（GDD + DPB）→ Prompt Branch 的输出通过 1×1 Conv 与 decoder 各级特征融合 → 生成残差图 $\mathbf{R}$ → 输出 $\hat{\mathbf{I}} = \mathbf{I} + \mathbf{R}$

**关键架构设计**：Encoder 中移除了 attention 机制（只保留 FFN），因为 early layers 聚焦局部模式而 self-attention 会稀释高频——这是一个重要的设计取舍。

### 关键设计

1. **Gated Dynamic Decoupler (GDD)**:

    - 功能：将输入特征动态解耦为低频和高频分量
    - 核心思路：对输入特征 $\mathbf{F}_s$ 通过 GAP + Conv 生成通道描述子，经 gating 机制（sigmoid 门控）抑制不重要元素后，通过 Softmax 归一化保证生成的是低通滤波器 $\mathbf{F}^L$。将 $\mathbf{F}^L$ 应用于分组输入特征得到低频分量：
    $\mathbf{F}^{lo}_{i,c,h,w} = \sum_{p,q} \mathbf{F}^L_{i,p,q} \mathbf{F}_{i,c,h+p,w+q}$
    - 高通滤波器通过从 identity kernel 减去低通滤波器获得，得到高频分量 $\mathbf{F}_{hi}$
    - 设计动机：不同退化影响不同频带，需要自适应地分离频率成分。Softmax 保证滤波器为低通，gating 抑制冗余元素。每个空间位置和通道组动态学习滤波器

2. **High-frequency Prompt Modulator (HPM)**:

    - 功能：对高频特征注入 prompt 组件并与 decoder 特征进行局部交叉注意力
    - 核心思路：(a) Generation：用深度卷积 + GELU 门控增强高频特征 $\hat{\mathbf{F}}_{hi} = \tilde{\mathbf{F}}_{hi} \odot \sigma(\text{DConv}_{3\times3}(\tilde{\mathbf{F}}_{hi}))$，然后与可学习 prompt $\mathbf{P}_{hi}$ 逐元素相乘 $\mathbf{F}^{prompt}_{hi} = \hat{\mathbf{F}}_{hi} \odot \mathbf{P}_{hi}$
    - (b) Modulation：depth-wise conv 进一步增强高频成分，然后通过局部窗口交叉注意力（window size $M=8$）与 decoder 特征交互
    $\mathbf{F}^{out}_{hi} = \mathbf{V}_{hi} \cdot \text{Softmax}(\mathbf{K}_{hi} \cdot \mathbf{Q}_{hi} / \sqrt{d})$
    - 设计动机：高频信息对应局部细节，用窗口注意力捕获足够且节省计算。Depth-wise conv 天然充当高通滤波器

3. **Low-frequency Prompt Modulator (LPM)**:

    - 功能：对低频特征在傅里叶域注入 prompt 并与 decoder 特征进行全局交叉注意力
    - 核心思路：(a) Generation：将低频特征 FFT 到频域，gating 筛选有用成分 $\hat{\mathbf{F}}_{lo} = \mathcal{F}(\tilde{\mathbf{F}}_{lo}) \odot \sigma(\text{Conv}_{1\times1}(\mathcal{F}(\tilde{\mathbf{F}}_{lo})))$，然后与可学习频域 prompt $\mathbf{P}_{lo}$ 逐元素相乘后 IFFT 回空域
    $\mathbf{F}^{prompt}_{lo} = \mathcal{F}^{-1}(\hat{\mathbf{F}}_{lo} \odot \mathbf{P}_{lo})$
    - (b) Modulation：adaptive average pooling 增强低频后，通过全局交叉注意力（Q 来自 decoder feature，K/V 来自 pooled prompt feature）调制
    $\mathbf{F}^{out}_{lo} = \mathbf{V}_{lo} \cdot \text{Softmax}(\mathbf{K}_{lo} \cdot \mathbf{Q}_{lo} / \alpha)$
    - **关键理论洞察**：根据卷积定理，频域 Hadamard product 等价于空域卷积——Eq.(8) 证明整个 LPM 等价于一个动态大核深度卷积，但在频域实现计算效率更高
    - 设计动机：低频信息对应全局结构，需要全局（非窗口）注意力。在频域操作天然实现全局感受野

### 损失函数 / 训练策略

- 使用与 PromptIR [60] 相同的广泛使用的复原损失函数
- AdamW 优化器，初始学习率 $3 \times 10^{-4}$，余弦退火降至 $1 \times 10^{-6}$
- 架构参数：3 级 encoder-decoder，blocks 数 [2,3,6]，embedding 维度 C=48，注意力头 [2,4,8]，FFN 扩展因子 3
- Pixel-unshuffle/pixel-shuffle 用于下/上采样

## 实验关键数据

### 主实验

**去雨 (SPAD)**:

| 方法 | PSNR↑ | SSIM↑ |
|------|-------|-------|
| Restormer | 47.98 | 0.9921 |
| DRSformer | 48.53 | 0.9924 |
| **FPro** | **48.99** | **0.9936** |

比 DRSformer +0.46 dB，比 SCD-Former +2.1 dB。

**去雨滴 (AGAN-Data)**:

| 方法 | PSNR↑ | SSIM↑ |
|------|-------|-------|
| IDT | 31.63 | 0.936 |
| Restormer | 31.68 | 0.934 |
| **FPro** | **31.96** | **0.937** |

比 Restormer +0.28 dB。

**去摩尔纹 (TIP-2018)**:

| 方法 | PSNR↑ | SSIM↑ |
|------|-------|-------|
| Wang et al. | 28.87 | 0.894 |
| **FPro** | **29.25** | 0.879 |

比前最佳 +0.38 dB。

### 消融实验

**GDD 有效性**:

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| Multi Dynamic Conv | 48.52 | 0.9926 | 用多组动态卷积替代 |
| Multi GDD | 48.91 | 0.9934 | 每个 DPB 各一个 GDD |
| Single GDD (共享) | **48.99** | **0.9936** | 共享一个 GDD 最优 |

**DPB 有效性**:

| 配置 | PSNR | SSIM | 说明 |
|------|------|------|------|
| w/o HPM | 48.77 | 0.9931 | 去掉高频 prompt，-0.22 dB |
| w/o LPM | 48.89 | 0.9933 | 去掉低频 prompt，-0.10 dB |
| Full | **48.99** | **0.9936** | 完整模型 |

**模型效率**:

| 方法 | FLOPs (G) | Params (M) |
|------|-----------|------------|
| Restormer | 174.7 | 26.1 |
| DRSformer | 242.9 | 33.7 |
| **FPro** | **81.9** | 22.3 |

FPro 仅需 Restormer 约 47% 的 FLOPs。

### 关键发现
- HPM 贡献大于 LPM（去掉 HPM 掉 0.22 dB vs 去掉 LPM 掉 0.10 dB），说明高频细节对图像复原更关键
- GDD 替换普通 Dynamic Conv 带来 +0.39 dB 提升，gating 机制有效抑制冗余滤波器元素
- 共享单个 GDD 优于每个 DPB 各用一个（节省 0.02M 参数的同时还提升 0.08 dB），说明频率分解在不同尺度共享是合理的
- FPro 在真实场景（NIQE 5.30 vs DRSformer 5.59）感知质量也最优
- FLOPs 仅 81.9G（vs DRSformer 242.9G），计算高效

## 亮点与洞察

- **频域 prompt 的理论优雅性**：LPM 在频域做 Hadamard product + gating，理论上等价于空域动态大核深度卷积，但计算复杂度低得多。这一理论推导是本文亮点
- **频率解耦 + 双分支处理**：高频用局部窗口注意力，低频用全局注意力——频率特性与注意力范围天然匹配的设计理念值得借鉴
- **Encoder 去掉 attention**：因为 early layers 聚焦局部模式、self-attention 是低通滤波器——这种对 Transformer 特性的深入理解指导架构设计的做法值得学习
- **可学习频域 prompt 组件**：$\mathbf{P}_{hi}$ 和 $\mathbf{P}_{lo}$ 作为可学习参数直接注入频率特征，简洁有效
- **统一框架处理 5 种退化**：虽非 all-in-one 模型，但同一架构在 5 个不同复原任务上都取得最佳或接近最佳成绩

## 局限与展望

- 论文未讨论 all-in-one 设置，每个退化任务需单独训练
- 频率分解仅使用 3×3 滤波器，可能对某些特殊频率模式的捕获不够精细
- GDD 的 gating 机制依赖 GAP 全局信息，在局部退化（如小区域遮挡）场景可能不够精确
- 在去雾和去模糊任务上的详细实验放在了补充材料，主文展示不够完整
- prompt 组件 $\mathbf{P}$ 是固定大小的可学习参数，对不同分辨率输入的适应性有待研究

## 相关工作与启发

- **vs Restormer**: Restormer 用 channel-wise attention 降低计算量，但忽略了频域信息。FPro 以更少 FLOPs 取得更好性能
- **vs PromptIR/DA-CLIP**: 这些方法从空间域提取退化 prompt，忽略频率线索。FPro 明确从频率角度提取 prompt
- **vs DRSformer**: DRSformer 是去雨任务 SOTA，FPro 在 SPAD 上超越 +0.46 dB 且 FLOPs 仅为其 1/3
- **vs SwinIR**: 基于窗口注意力但不区分频率成分。FPro 的 HPM 也用窗口注意力但专门处理高频信息

## 评分

- 新颖性: ⭐⭐⭐⭐ 首次从频率视角设计 prompt learning 用于图像复原，理论推导扎实
- 实验充分度: ⭐⭐⭐⭐ 5 个任务覆盖面广，消融实验充分，可视化分析直观
- 写作质量: ⭐⭐⭐⭐⭐ 论文写作质量高，方法推导严谨，公式-文字-图示配合好
- 价值: ⭐⭐⭐⭐ 频域 prompt 思路有通用性，FLOPs 低性能好，有开源代码

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Teaching Tailored to Talent: Adverse Weather Restoration via Prompt Pool and Depth-Anything Constraint](teaching_tailored_to_talent_adverse_weather_restoration_via_prompt_pool_and_dept.md)
- [\[ECCV 2024\] Contourlet Residual for Prompt Learning Enhanced Infrared Image Super-Resolution](contourlet_residual_for_prompt_learning_enhanced_infrared_image_super-resolution.md)
- [\[ECCV 2024\] Efficient Diffusion Transformer with Step-wise Dynamic Attention Mediators](efficient_diffusion_transformer_with_step-wise_dynamic_attention_mediators.md)
- [\[ECCV 2024\] EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](edformer_transformer-based_event_denoising_across_varied_noise_levels.md)
- [\[ECCV 2024\] OAPT: Offset-Aware Partition Transformer for Double JPEG Artifacts Removal](oapt_offset-aware_partition_transformer_for_double_jpeg_artifacts_removal.md)

</div>

<!-- RELATED:END -->
