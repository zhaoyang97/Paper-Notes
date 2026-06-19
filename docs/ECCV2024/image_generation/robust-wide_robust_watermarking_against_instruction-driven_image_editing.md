---
title: >-
  [论文解读] Robust-Wide: Robust Watermarking against Instruction-driven Image Editing
description: >-
  [ECCV 2024][图像生成][鲁棒水印] 本文提出 Robust-Wide，首个针对指令驱动图像编辑的鲁棒水印方法，核心创新是部分指令驱动去噪采样引导（PIDSG）模块——在训练中将编辑过程的最后k步梯度打通，迫使水印嵌入到语义感知区域，实现编辑后仅约2.6% 的64位水印误码率。 领域现状：InstructPix2P…
tags:
  - "ECCV 2024"
  - "图像生成"
  - "鲁棒水印"
  - "图像编辑"
  - "InstructPix2Pix"
  - "语义级扰动"
  - "扩散模型"
---

# Robust-Wide: Robust Watermarking against Instruction-driven Image Editing

**会议**: ECCV 2024  
**arXiv**: [2402.12688](https://arxiv.org/abs/2402.12688)  
**代码**: [https://github.com/hurunyi/Robust-Wide](https://github.com/hurunyi/Robust-Wide)  
**领域**: 图像生成  
**关键词**: 鲁棒水印, 图像编辑, InstructPix2Pix, 语义级扰动, 扩散模型

## 一句话总结
本文提出 Robust-Wide，首个针对指令驱动图像编辑的鲁棒水印方法，核心创新是部分指令驱动去噪采样引导（PIDSG）模块——在训练中将编辑过程的最后k步梯度打通，迫使水印嵌入到语义感知区域，实现编辑后仅约2.6% 的64位水印误码率。

## 研究背景与动机

**领域现状**：InstructPix2Pix 等指令驱动图像编辑模型允许用户通过文本指令快速编辑图像，带来便利的同时也被恶意利用（伪造新闻、盗用画作风格等）。水印技术是追踪溯源的常用手段。

**现有痛点**：现有SOTA水印方法（HiDDeN、MBRS、PIMoG、SepMark等）主要针对像素级扰动（JPEG、模糊、几何变换）设计鲁棒性。但指令驱动编辑在**语义级别**大幅改变图像（如改变表情、替换物体、迁移风格），这些方法在编辑后误码率接近50%——相当于随机猜测，完全失效。

**核心矛盾**：像素级水印嵌入的信息在语义级编辑下无法存活，因为编辑模型会重新生成图像的大部分内容。需要将水印嵌入到编辑过程不会改变的"语义锚点"区域。

**本文目标** 如何在端到端训练框架中模拟指令驱动编辑的扰动，使水印编码器学会将信息嵌入到语义鲁棒区域。

**切入角度**：编辑过程涉及大量去噪采样步骤，直接打通所有步骤的梯度显存开销巨大。作者观察到只打通最后k步就足以让编码器感知编辑过程的特征。

**核心 idea**：选择性打通编辑采样过程最后k步的梯度流，配合多样化的指令注入，迫使水印编码器将信息嵌入到语义感知的鲁棒区域。

## 方法详解

### 整体框架
Robust-Wide 遵循经典的 编码器-噪声层-解码器 架构。输入原始图像 $I_{ori}$ 和随机L位消息 $m$，编码器 $E_m$ 生成水印图像 $I_{wm}$；PIDSG模块模拟指令驱动编辑产生编辑后图像 $I_{wm}^{edit}$；解码器 $E_x$ 从编辑后图像提取水印消息。三部分联合端到端训练。

### 关键设计

1. **水印编码器 $E_m$（U-Net结构）**:

    - 功能：将L位二进制消息嵌入到图像中，生成视觉上与原图一致的水印图像
    - 核心思路：先将 $1 \times \sqrt{L} \times \sqrt{L}$ 形状的消息通过转置卷积扩展到 $C \times H \times W$，再与原图拼接送入U-Net
    - 嵌入约束：像素级 $L_2$ 损失 $L_{em_1} = L_2(I_{ori}, I_{wm})$ + 潜空间级 $L_2$ 损失 $L_{em_2} = L_2(\mathcal{E}(I_{ori}), \mathcal{E}(I_{wm}))$，确保水印图像保持可编辑性

2. **PIDSG（部分指令驱动去噪采样引导）**:

    - 功能：在训练中模拟指令驱动编辑的语义级扰动，使梯度可以从解码器传播回编码器
    - 核心思路：冻结 InstructPix2Pix 所有参数。编辑共T步采样，前 T-k 步截断梯度得到部分去噪潜变量 $Z_k$，后k步保留梯度使整个流程可微分。CLIP编码器处理编辑指令 $Ins$ 引导采样
    - 设计动机：(a) 全部T步打通梯度显存不可行；(b) 最后k步足以捕捉编辑过程的关键特征；(c) 注入多样化指令迫使编码器关注语义区域而非特定编辑模式
    - 与之前方法的区别：现有噪声层只模拟JPEG、模糊等像素级扰动，PIDSG首次将完整的扩散编辑过程引入训练

3. **水印解码器 $E_x$（残差块结构）**:

    - 功能：从编辑后或未编辑的水印图像中提取嵌入的消息
    - 关键发现：必须同时在编辑后图像和未编辑水印图像上训练解码器。只用编辑后图像训练不收敛，因为解码器无法定位水印区域

### 损失函数
总损失 $L_{total} = L_{em_1} + 0.001 \cdot L_{em_2} + 0.1 \cdot L_{ex_1} + 1.0 \cdot L_{ex_2}$

其中 $L_{ex_1} = \text{MSE}(m, E_x(I_{wm}^{edit}))$ 为编辑后提取损失，$L_{ex_2} = \text{MSE}(m, E_x(I_{wm}))$ 为未编辑提取损失。

## 实验关键数据

### 主实验

训练集20k图像-指令对，测试集1.2k样本 + 1.44k真实世界样本。

| 方法 | 图像尺寸 | 水印位数 | BER%(无编辑) | BER%(编辑后) | PSNR↑ | SSIM↑ |
|------|----------|----------|-------------|-------------|-------|-------|
| DWT-DCT | 512 | 32 | 11.94 | 49.23 | 38.71 | 0.966 |
| DWT-DCT-SVD | 512 | 32 | 0.03 | 47.57 | 38.65 | 0.973 |
| RivaGAN | 512 | 32 | 0.63 | 40.53 | 40.61 | 0.972 |
| MBRS | 256 | 256 | 0.00 | 46.77 | 43.98 | 0.987 |
| PIMoG | 256 | 64 | 0.00 | 49.96 | 35.32 | 0.921 |
| SepMark | 256 | 128 | 0.01 | 28.15 | 36.43 | 0.919 |
| **Robust-Wide** | **512** | **64** | **0.00** | **2.66** | **41.91** | **0.991** |

所有基线方法编辑后BER接近50%（≈随机），Robust-Wide仅2.66%。

### 像素级扰动鲁棒性（未在训练中见过）

| 扰动类型 | 编辑前处理 | 编辑后处理(水印图) | 编辑后处理(编辑图) |
|----------|-----------|-------------------|-------------------|
| 无 | 2.66% | 0.00% | 2.66% |
| JPEG | 2.73% | 0.00% | 2.79% |
| 高斯噪声 | 3.07% | 0.07% | 6.05% |
| 亮度 | 12.29% | 0.47% | 9.48% |
| 噪声+去噪 | 8.64% | 3.91% | 9.37% |

### 关键发现
- **PIDSG是核心**：去掉PIDSG后，编辑后BER从2.66%飙升至50.16%——完全失效
- **天然的像素级鲁棒性**：即使训练中从未见过JPEG、模糊等传统扰动，Robust-Wide仍然对其具有鲁棒性。说明语义级嵌入自然涵盖了像素级鲁棒性
- **跨编辑模型泛化**：对ControlNet-InstructPix2Pix (BER 0.96%)、MagicBrush (BER 9.34%)、Inpainting、DDIM Inversion 均有效
- **连续编辑鲁棒**：经过3轮连续编辑后水印仍可准确提取
- 可视化显示水印主要嵌入在主体轮廓和背景的概念性区域

## 亮点与洞察
- **PIDSG设计精妙**：解决了扩散采样过程不可微的难题，只打通最后k步既节省显存又足以引导语义感知嵌入。这个"部分梯度打通"的思路可迁移到任何需要将不可微生成过程纳入训练的场景
- **语义级鲁棒性包含像素级鲁棒性**：一个深刻的洞察——如果水印能在语义大改后存活，那像素级微小扰动自然也不会影响它
- **Le_x2的发现**：必须在未编辑图上也训练解码器才能收敛，揭示了解码器需要先"学会找水印"才能在编辑后的复杂场景中定位

## 局限性
- 当编辑极为剧烈（如 $s_I = 1$，几乎完全重新生成）时BER显著上升
- 训练依赖特定编辑模型（InstructPix2Pix），对MagicBrush等差异较大的模型泛化性有所下降
- 单GPU (A6000) 训练，图像尺寸固定 512×512，更大分辨率的适用性未验证
- PIDSG引入的训练开销较大（需要运行k步去噪采样）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个针对指令驱动编辑的水印方法，PIDSG模块极具创新性
- 实验充分度: ⭐⭐⭐⭐⭐ 多种编辑模型、采样配置、像素级扰动、连续编辑评估极为全面
- 写作质量: ⭐⭐⭐⭐ 问题定义和动机阐述清晰
- 价值: ⭐⭐⭐⭐⭐ 解决了AI安全领域的紧迫问题，对水印和AI内容治理有重要意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](../../CVPR2026/image_generation/rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [\[ECCV 2024\] A High-Quality Robust Diffusion Framework for Corrupted Dataset](a_highquality_robust_diffusion_framework_for_corrupted_datas.md)
- [\[ECCV 2024\] Shedding More Light on Robust Classifiers under the lens of Energy-based Models](shedding_more_light_on_robust_classifiers_under_the_lens_of_energy-based_models.md)
- [\[CVPR 2025\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](../../CVPR2025/image_generation/editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [\[ECCV 2024\] RegionDrag: Fast Region-Based Image Editing with Diffusion Models](regiondrag_fast_region-based_image_editing_with_diffusion_models.md)

</div>

<!-- RELATED:END -->
