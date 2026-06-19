---
title: >-
  [论文解读] MDiff4STR: Mask Diffusion Model for Scene Text Recognition
description: >-
  [AAAI 2026 Oral][图像生成][场景文本识别] 首次将掩码扩散模型（MDM）引入场景文本识别（STR）任务，提出 MDiff4STR，通过六种训练掩码策略（弥合训练-推理噪声差距）和 Token 替换噪声机制（解决过度自信问题），在仅需 3 步去噪的情况下超越 SOTA 自回归模型的准确率，同时实现 3× 推理加速。
tags:
  - "AAAI 2026 Oral"
  - "图像生成"
  - "场景文本识别"
  - "掩码扩散模型"
  - "去噪策略"
  - "Token替换噪声"
  - "全向语言建模"
---

# MDiff4STR: Mask Diffusion Model for Scene Text Recognition

**会议**: AAAI 2026 Oral  
**arXiv**: [2512.01422](https://arxiv.org/abs/2512.01422)  
**代码**: [https://github.com/Topdu/OpenOCR](https://github.com/Topdu/OpenOCR)  
**领域**: 图像生成  
**关键词**: 场景文本识别, 掩码扩散模型, 去噪策略, Token替换噪声, 全向语言建模

## 一句话总结

首次将掩码扩散模型（MDM）引入场景文本识别（STR）任务，提出 MDiff4STR，通过六种训练掩码策略（弥合训练-推理噪声差距）和 Token 替换噪声机制（解决过度自信问题），在仅需 3 步去噪的情况下超越 SOTA 自回归模型的准确率，同时实现 3× 推理加速。

## 研究背景与动机

场景文本识别（STR）是 OCR 系统的核心任务，面临弯曲文本、遮挡、模糊、艺术字体等挑战。现有 STR 方法主要有四种范式：

**自回归模型 (ARM)**：序列建模能力强、准确率高，但逐字符解码效率低

**并行解码模型 (PDM)**：速度快但缺乏上下文建模，准确率低于 ARM

**BERT-like 精炼模型 (ReM)**：先并行预测再双向精炼，但对初始预测错误敏感

**掩码扩散模型 (MDM)**：新兴范式，通过从部分遮掩输入恢复原始序列来学习全向依赖

**MDM 的潜力**：与 ARM 的单向建模不同，MDM 可以捕获更灵活、更全面的全向上下文依赖，这对需要语言理解的 STR 任务至关重要。同时其去噪过程高效可控，仅需几步即可得到准确预测。

**MDM 直接应用于 STR 的两个关键问题**：

**训练-推理噪声差距（Noising Gap）**：训练时使用随机掩码，但推理时从全掩码开始，后续的 remask 模式也与训练不匹配，导致泛化性差

**推理时过度自信（Overconfident Predictions）**：MDM 倾向于给错误预测分配过高的置信分数（如将错误字符"F"的置信度报为 0.95），导致基于置信度的 remask 机制失效，错误无法在后续步骤中被纠正

## 方法详解

### 整体框架

MDiff4STR 的架构包含：
- **视觉编码器**：SVTRv2，专为 STR 设计，提取图像特征 $\mathbf{F}_v \in \mathbb{R}^{\frac{H}{8} \times \frac{W}{4} \times D}$
- **字符嵌入层**：将噪声化字符序列映射为向量
- **掩码扩散解码器（MDiffDecoder）**：条件于视觉特征进行去噪
- **分类器**：将解码 token 映射回字符

### 关键设计

#### 1. **Vanilla MDM 基线与多种解码范式**

MDM 的灵活性体现在可以支持多种解码策略：

- **MDiff-PD**（并行解码）：全部 mask → 一步解码
- **MDiff-AR**（自回归解码）：从左到右逐步 unmask
- **MDiff-Re**（精炼解码）：BERT-like 双向精炼
- **MDiff-LC**（低置信度 remask）：每步将平均置信度以下的 token 重新 mask
- **MDiff-BLC**（块内低置信度 remask）：在固定大小块内执行低置信度 remask，避免"置信度陷阱"（某些 token 反复被 remask）

#### 2. **六种训练掩码策略（弥合噪声差距）**

为消除训练-推理噪声差距，将推理中使用的所有 remask 模式引入训练。原始 MDM 仅用随机掩码训练，而 MDiff4STR 统一使用以下 7 种策略，每次训练时均匀采样：

| 策略 | 描述 |
|-----|------|
| (a) 随机掩码 | 原始 MDM 的训练策略 |
| (b) 全掩码 | 推理第一步的初始状态 |
| (c) 前向自回归 | 从左到右保留部分 |
| (d) 后向自回归 | 从右到左保留部分 |
| (e) BERT-like 精炼 | 保留大部分，mask 少量 |
| (f) 低置信度 remask | 模拟推理中的 remask |
| (g) 块内低置信度 remask | 本文专有的 remask 策略 |

**全掩码策略贡献最大**，因为它是所有推理去噪的第一步基础。

#### 3. **Token 替换噪声机制（解决过度自信）**

这是本文的核心创新。除了传统的掩码噪声，引入一种全新的噪声类型：

将原始序列 $\mathbf{Y}$ 中的某些字符**随机替换为其他字符**，构造错误序列 $\mathbf{Y}_r$，模拟推理中"高置信但错误"的场景。模型需要学会：
- 识别哪些 token 是错误的（但不知道哪些被替换了）
- 纠正这些错误

$$\tilde{\mathbf{T}} = \text{MDiffDecoder}(\mathbf{F}_v, \mathbf{T}_r), \quad \tilde{\mathbf{Y}} = \text{Classifier}(\tilde{\mathbf{T}})$$

关键区别：
- 去噪训练中仅监督 mask 位置
- 纠错训练中监督**所有位置**（因为推理时不知道哪些是错的）

### 损失函数 / 训练策略

$$\mathcal{L}_{total} = \mathcal{L}_{denoising} + \mathcal{L}_{correction}$$

**去噪损失**：

$$\mathcal{L}_{denoising} = -\frac{1}{l_1}\sum_{i=1}^{L}\mathbf{1}[\mathbf{Y}_{l_1}^i = \mathbf{M}]\log p_\theta(\mathbf{Y}^i | \mathbf{Y}_{l_1})$$

**纠错损失**：

$$\mathcal{L}_{correction} = -\frac{1}{L}\sum_{i=1}^{L}\log p_\theta(\mathbf{Y}^i | \mathbf{Y}_{l_2})$$

训练配置：AdamW（weight decay 0.05），LR $5 \times 10^{-4}$，batch size 1024，one-cycle LR scheduler，英文模型 40 epochs / 中文 100 epochs，4× RTX 3090 GPU，最大文本长度 25。

## 实验关键数据

### 主实验

**英文 STR（从头训练，U14M-Filter）**：

| 方法 | 类型 | Com Avg | U14M Avg | OST | 推理时间(ms) |
|-----|------|---------|---------|-----|------------|
| SVTRv2-B | CTC | 96.57 | 86.14 | 80.0 | 19.8 |
| PARSeq | ReM | 96.40 | 84.26 | 79.9 | 23.8 |
| MAERec | ARM | 96.36 | 85.17 | 76.4 | 35.7 |
| IGTR | ARM | 96.48 | 84.86 | 76.3 | 24.1 |
| ARMbase (本文基线) | ARM | 96.88 | 87.34 | 81.03 | 57.95 |
| **MDiff4STR-B-BLC** | **MDM** | **97.30** | **88.44** | **84.25** | **19.21** |

MDiff4STR-B-BLC 在 Com/U14M/OST 上分别超越之前最佳 0.73%/2.30%/4.30%，且推理速度为 ARM 的 **3 倍**。

**预训练后微调**：

| 方法 | Com Avg | OST |
|-----|---------|-----|
| SVTRv2-B (预训练) | 97.83 | 86.9 |
| E2STR | 97.71 | 80.7 |
| CLIP4STR | 97.32 | 82.8 |
| **MDiff4STR-BLC (预训练)** | **98.02** | **87.4** |

**中文 STR（BCTR）**：

| 方法 | Scene | Web | Doc | HW | Avg |
|-----|-------|-----|-----|----|----|
| MAERec | 84.4 | 83.0 | 99.5 | 65.6 | 83.13 |
| SVTRv2-B | 83.5 | 83.3 | 99.5 | 67.0 | 83.31 |
| **MDiff4STR-B-BLC** | **85.7** | **84.7** | **99.6** | **67.0** | **84.25** |

### 消融实验

**核心组件消融**（MDiff-BLC, B 规模）：

| 配置 | Com | U14M | OST | 提升 |
|------|-----|------|-----|------|
| Vanilla MDM (随机mask) | 96.42 | 85.42 | 79.93 | 基线 |
| + 六种训练mask策略 | 96.98 (+0.57) | 87.09 (+1.67) | 81.92 (+2.00) | 弥合噪声差距 |
| + Token替换噪声 | **97.30 (+0.88)** | **88.44 (+3.02)** | **84.25 (+4.33)** | 解决过度自信 |

**去噪步数 vs 精度**：

| 步数 K | Com (BLC) | U14M (BLC) | OST (BLC) | 推理时间(ms) |
|--------|-----------|-----------|----------|------------|
| 1 | 96.88 | 86.69 | 81.31 | 10.52 |
| 2 | 97.19 | 88.05 | 83.69 | 15.56 |
| **3** | **97.30** | **88.44** | **84.25** | **19.21** |
| 5 | 97.28 | 88.50 | 84.42 | 25.70 |
| 8 | 97.24 | 88.65 | 84.11 | 32.74 |

K=3 是精度-效率的最优平衡点，更多步数收效甚微甚至下降。

**六种mask策略的渐进消融**：

| 基线 | +全mask | +前向AR | +后向AR | +ReM | +LC | +BLC |
|------|--------|--------|--------|------|-----|------|
| 85.42 | +1.04 | +1.37 | +1.41 | +1.46 | +1.53 | +1.67 |

每种策略都有正贡献，全 mask 效果最大。

### 关键发现

1. **MDM 的全向上下文建模优势在遮挡场景最突出**：OST 上提升 4.30%（从头训练）/3.22%（vs ARMbase），表明 MDM 的全向依赖建模能力显著优于单向 ARM 和双向 ReM
2. **Token 替换噪声机制效果显著**：在更挑战性的数据集上提升更大（Com +0.88% vs U14M +3.02% vs OST +4.33%）
3. **仅 3 步去噪即可超越 ARM**：说明 MDM 范式在 STR 中的效率优势
4. **MDM 可灵活支持多种解码范式**（PD/AR/Re/LC/BLC），但其专有的 BLC 策略最优
5. **对中英文都有效**：在中文 BCTR 上也取得 SOTA（Scene +1.3%，Web +1.5%）

## 亮点与洞察

- **范式级创新**：将 MDM 引入 STR，建立了与 ARM 平行的新范式
- **问题诊断精准**：准确识别了 vanilla MDM 应用于 STR 的两个关键瓶颈（噪声差距 + 过度自信），并给出针对性解决方案
- **Token 替换噪声**是高度可推广的思想：不仅适用于 STR，可扩展到任何 MDM 应用场景
- **全向语言建模的优势**在需要推理的场景（遮挡、艺术字体）中尤为明显
- **效率-精度兼得**：3 步去噪 = ARM 3× 速度 + 更高准确率

## 局限与展望

- 当前 MDM 默认分离通道 $M$ 和最大文本长度固定为 25，对不定长文本的适应性有待验证
- 低置信度 remask 策略在极端情况下（所有预测都高置信但全错）可能仍受限
- 未探索与视觉编码器联合优化（当前 SVTRv2 编码器和 MDM 解码器是独立设计）
- 可进一步将 Token 替换噪声的替换比例和策略做更细粒度的调整
- 视觉丰富文档（如手写体、古籍）的验证不够充分

## 相关工作与启发

本文建立了 STR 方法的四大范式分类体系（CTC/ARM/PDM/ReM）+ 新增 MDM 范式。关键启发：

- **MDM 不仅是 NLP 范式**：通过本文验证，MDM 在视觉-语言任务中同样展现强大潜力
- **噪声设计是 MDM 的核心**：不同于扩散模型中噪声通常被视为辅助角色，MDM 的性能高度依赖于噪声策略的设计
- **纠错能力**：Token 替换噪声让模型具备了"自我纠错"能力，这在自回归模型中难以实现（auto-regressive 一旦犯错就传播）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（首次将 MDM 引入 STR + Token 替换噪声机制）
- 实验充分度: ⭐⭐⭐⭐⭐（英文/中文多基准 + 5种解码策略 + 详尽消融）
- 写作质量: ⭐⭐⭐⭐⭐（问题诊断→方案设计的逻辑链清晰流畅）
- 价值: ⭐⭐⭐⭐⭐（建立 STR 新范式，同时提升精度和效率）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MTADiffusion: Mask Text Alignment Diffusion Model for Object Inpainting](../../CVPR2025/image_generation/mtadiffusion_mask_text_alignment_diffusion_model_for_object_inpainting.md)
- [\[CVPR 2026\] A Self-Conditioned Representation Guided Diffusion Model for Realistic Text-to-LiDAR Scene Generation](../../CVPR2026/image_generation/a_self-conditioned_representation_guided_diffusion_model_for_realistic_text-to-l.md)
- [\[ECCV 2024\] DCDM: Diffusion-Conditioned-Diffusion Model for Scene Text Image Super-Resolution](../../ECCV2024/image_generation/dcdm_diffusion-conditioned-diffusion_model_for_scene_text_image_super-resolution.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)
- [\[AAAI 2026\] SpecDiff: Accelerating Diffusion Model Inference with Self-Speculation](specdiff_accelerating_diffusion_model_inference_with_self-speculation.md)

</div>

<!-- RELATED:END -->
