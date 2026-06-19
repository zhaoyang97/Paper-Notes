---
title: >-
  [论文解读] One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting
description: >-
  [NeurIPS 2025][图像生成][文本引导图像修复] 提出NTN-Diff频率感知扩散模型，通过将语义一致性问题分解为中频和低频频带各自的一致性任务，利用"空文本-文本-空文本"三阶段去噪策略，同时解决文本引导图像修复中的未遮盖区域保持和遮盖/未遮盖区域语义一致性两大挑战。 文本引导图像修复（text-guided…
tags:
  - "NeurIPS 2025"
  - "图像生成"
  - "文本引导图像修复"
  - "频率感知"
  - "空文本去噪"
  - "扩散模型"
  - "语义一致性"
---

# One Stone with Two Birds: A Null-Text-Null Frequency-Aware Diffusion Models for Text-Guided Image Inpainting

**会议**: NeurIPS 2025  
**arXiv**: [2510.08273](https://arxiv.org/abs/2510.08273)  
**代码**: [GitHub](https://github.com/htyjers/NTN-Diff)  
**领域**: 扩散模型 / 图像修复  
**关键词**: 文本引导图像修复, 频率感知, 空文本去噪, 扩散模型, 语义一致性

## 一句话总结

提出NTN-Diff频率感知扩散模型，通过将语义一致性问题分解为中频和低频频带各自的一致性任务，利用"空文本-文本-空文本"三阶段去噪策略，同时解决文本引导图像修复中的未遮盖区域保持和遮盖/未遮盖区域语义一致性两大挑战。

## 研究背景与动机

文本引导图像修复（text-guided image inpainting）的目标是根据文本提示重建遮盖区域内容，同时保持未遮盖区域不变。该任务面临两个长期存在的挑战：

**未遮盖区域保持**：保证修复过程不改变未遮盖区域的内容

**遮盖/未遮盖区域语义一致性**：确保生成的遮盖区域内容与周围未遮盖区域在语义上和谐统一

现有方法只能解决其中一个，无法同时满足两者。BLD（Blended Latent Diffusion）通过每步混合操作保持未遮盖区域，但由于扩散过程和去噪过程之间的差异，遮盖与未遮盖区域语义不一致。BrushNet通过无交叉注意力的密集纹理特征图保持一致性，但另一个文本引导去噪过程破坏了未遮盖区域。

**关键洞察**：这一矛盾源于不同频率带在去噪过程中对文本提示的鲁棒性差异：
- **低频带**（颜色、光照等）在高噪声阶段容易被文本提示调制，波动大
- **中频带**（布局、结构等）对文本提示更鲁棒，在早期阶段结束时就趋于稳定

本文的核心idea：将全局语义一致性分解为各频带的一致性，利用中频带的稳定性作为"锚点"来引导低频带的去噪。

## 方法详解

### 整体框架

NTN-Diff将去噪过程分为**早期阶段**（高噪声，步数$T$到$\lambda T$）和**晚期阶段**（低噪声，步数$\lambda T$到$0$），共包含四个并行/串行去噪流程：

- (I) 空文本低频感知去噪（早期）
- (II) 文本引导去噪（早期）
- (III) 空文本中频引导去噪（早期）
- (IV) 文本引导去噪（晚期）

### 关键设计

1. **空文本低频感知去噪过程（Sec.2.3.1）**：使用空文本$c_\emptyset$而非真实文本提示进行去噪，避免低频带被文本调制。每步将未遮盖区域替换为扩散过程的结果以保持：

$$\hat{z}_t^{un} = z_{T-t}^{gt} \odot m_z + z_t^{un} \odot (1 - m_z)$$

同时修改自注意力机制，通过掩码抑制遮盖区域的注意力分数，确保修复过程主要依赖未遮盖区域信息。关键作用：得到未被文本干扰的低频带。

2. **文本引导去噪过程 + 低频带替换（Sec.2.3.2）**：并行进行文本引导去噪以对齐遮盖区域的中频带与文本语义。关键操作是通过**去噪低频带层**将文本引导结果的低频替换为空文本去噪的低频：

$$\tilde{z}_t^{text} = \text{IDCT}(\text{DCT}(z_t^{un}) \odot m_{low} + \text{DCT}(z_t^{text}) \odot (1 - m_{low}))$$

低通掩码阈值自适应设置：$th_{lp} = \lambda_{lp}^f + \lambda_{lp}^r \cdot \frac{\|M\|_1}{HW}$，与未遮盖区域比例正相关（更大的未遮盖区域需要更多低频信息替换）。

3. **空文本中频引导去噪过程（Sec.2.3.3）**：利用步骤(II)中已对齐文本的中频带来引导新的空文本去噪过程。通过**去噪中频带层**将中频从文本引导过程注入：

$$\tilde{z}_t^{in} = \text{IDCT}(\text{DCT}(z_t^{text}) \odot m_{mid} + \text{DCT}(z_t^{in}) \odot (1 - m_{mid}))$$

中频带掩码使用带通滤波器提取。这一步的核心作用：在不使用文本提示的情况下，通过中频引导使低频带（特别是遮盖区域的低频）与文本语义对齐。

4. **晚期文本引导去噪（Sec.2.4）**：基于早期三阶段的输出进行最终的文本引导去噪。每步将未遮盖区域替换为扩散过程结果以保持。此时中频带已在早期稳定，低频带已在中频引导下预对齐，因此文本引导能够有效完成最后的精细化。

### 关于频带自适应提取

低频和中频的提取阈值均与未遮盖区域比例$\frac{\|M\|_1}{HW}$相关：更大的未遮盖区域需要更多频率信息从空文本过程替换到文本引导过程中。

## 实验关键数据

### BrushBench Inside Inpainting

| 方法 | IR×10↑ | HPS v2×10²↑ | PSNR↑ | MSE×10³↓ | LPIPS×10³↓ | CLIP Score↑ |
|------|--------|------------|-------|----------|-----------|------------|
| NTN-Diff* | **12.69** | **27.82** | **40.70** | **0.11** | **0.88** | **26.49** |
| BrushNet* | 12.64 | 27.78 | 31.94 | 0.80 | 18.67 | 26.39 |
| NTN-Diff | 12.45 | 27.57 | 23.51 | 6.50 | 40.79 | 26.54 |
| BrushNet | 12.36 | 27.40 | 21.65 | 9.31 | 48.28 | 26.48 |
| BLD | 9.78 | 25.87 | 21.33 | 9.76 | 49.26 | 26.15 |

### 消融实验（BrushBench）

| 配置 | IR×10↑ | PSNR↑ | LPIPS×10³↓ | CLIP Score↑ |
|------|--------|-------|-----------|------------|
| 完整NTN-Diff | **11.12** | **28.10** | **44.09** | **26.09** |
| Case A (无掩码自注意力) | 10.14 | 28.02 | 44.54 | 25.95 |
| Case B (无文本引导过程) | 9.59 | 27.71 | 47.08 | 25.78 |
| Case C (无中频引导过程) | 10.02 | 28.06 | 44.92 | 26.03 |

### 关键发现

- NTN-Diff*（带像素级混合）在PSNR上比BrushNet*高8.76，MSE低86%，LPIPS低95%，同时IR和CLIP Score更高
- 三阶段去噪缺一不可：Case B（缺少文本引导）在IR上降44.8%，说明中频引导需要文本对齐的中频源
- $\lambda = 0.6$是早/晚期划分的最优点：过短的早期（$\lambda=0.9$）无法充分稳定低频，过长（$\lambda=0.5$）则晚期文本引导不足
- 自适应频带提取至关重要：固定阈值会导致语义不一致（如笑脸猫）或色块伪影

## 亮点与洞察

- 频率域分析为扩散模型修复提供了全新视角：不同频带在去噪过程中对文本的响应差异是核心发现
- "null-text-null"三阶段设计巧妙利用了中频的鲁棒性作为桥梁连接低频保持和文本对齐
- 即插即用的频带替换层（DCT域操作）不需要额外训练，简洁高效
- 自适应阈值设计考虑了遮盖比例的影响，提升了对不同掩码的泛化能力

## 局限与展望

- 需要三个并行去噪过程，推理计算量是标准扩散模型的约3倍
- 基于Stable Diffusion v1.5，未在更新的SD-XL或SD3上验证
- 频带分离假设可能在某些极端情况下不成立（如纯低频图像）
- $\lambda$和频率阈值的超参数较多，虽然有启发式设置但仍需调参
- 仅在inpainting场景中验证，推广到outpainting等其他编辑任务需要进一步探索

## 相关工作与启发

- BLD的每步混合思路简单但有效，NTN-Diff在此基础上增加了频率感知的维度
- 频率域扩散模型（FreeDiff、FBSDiff等）启发了频率分离的思路
- 与BrushNet对比展示了纯网络结构方法与频率感知方法的互补性
- "先稳定再精细"的两阶段去噪策略对其他扩散模型编辑任务也有启发

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 频率感知视角新颖，null-text-null三阶段设计独特且有深刻动机
- **实验充分度**: ⭐⭐⭐⭐ BrushBench和EditBench两个benchmark，消融实验充分，超参数分析详细
- **写作质量**: ⭐⭐⭐⭐ 动机阐述清晰，频率分析可视化直观，但方法描述略显冗长
- **价值**: ⭐⭐⭐⭐ 同时解决修复中两大挑战有实际意义，频率视角为后续工作提供新方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] NPN: Non-Linear Projections of the Null-Space for Imaging Inverse Problems](npn_non-linear_projections_of_the_null-space_for_imaging_inverse_problems.md)
- [\[NeurIPS 2025\] Training-Free Safe Text Embedding Guidance for Text-to-Image Diffusion Models](training-free_safe_text_embedding_guidance_for_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Test-Time Alignment of Text-to-Image Diffusion Models via Null-Text Embedding Optimisation](../../CVPR2026/image_generation/test-time_alignment_of_text-to-image_diffusion_models_via_null-text_embedding_op.md)
- [\[NeurIPS 2025\] FairImagen: Post-Processing for Bias Mitigation in Text-to-Image Models](fairimagen_post-processing_for_bias_mitigation_in_text-to-image_models.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)

</div>

<!-- RELATED:END -->
