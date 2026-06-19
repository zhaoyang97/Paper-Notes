---
title: >-
  [论文解读] Enhancing Image Restoration Transformer via Adaptive Translation Equivariance
description: >-
  [ICCV 2025][图像恢复][平移等变性] 系统研究了平移等变性（Translation Equivariance, TE）对图像修复网络收敛速度和泛化能力的影响，提出滑动键值自注意力（SkvSA）及其自适应版本（ASkvSA）和下采样自注意力（DSA），构建了 TEAFormer，在超分、去模糊、去噪等多个任务上取得 SOTA，同时保持线性复杂度。
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "平移等变性"
  - "图像修复"
  - "Transformer"
  - "自适应注意力"
  - "超分辨率"
---

# Enhancing Image Restoration Transformer via Adaptive Translation Equivariance

**会议**: ICCV 2025  
**arXiv**: [2506.18520](https://arxiv.org/abs/2506.18520)  
**代码**: 无  
**领域**: 图像修复  
**关键词**: 平移等变性, 图像修复, Transformer, 自适应注意力, 超分辨率

## 一句话总结

系统研究了平移等变性（Translation Equivariance, TE）对图像修复网络收敛速度和泛化能力的影响，提出滑动键值自注意力（SkvSA）及其自适应版本（ASkvSA）和下采样自注意力（DSA），构建了 TEAFormer，在超分、去模糊、去噪等多个任务上取得 SOTA，同时保持线性复杂度。

## 研究背景与动机

### 领域现状

图像修复的核心性质是「忠实性」——对目标区域的修复结果应在输入几何变换下保持等变。在 CNN 时代，卷积的滑动特性天然赋予了网络平移等变性。然而 Transformer 引入后，Self-Attention（SA）的全局索引和位置编码，以及 Window Attention（WA）的特征移位操作，都破坏了平移等变性。

### 现有痛点

**Self-Attention 的计算瓶颈 vs 固定感受野的两难**：SA 具有全局感受野但 $O(N^2)$ 复杂度；WA 是线性复杂度但感受野固定，都不满足 TE

**TE 缺失的后果被忽视**：实验证据表明缺失 TE 会导致训练收敛慢（NTK condition number 高 7.5×）和泛化差（SRGA value 高 12%）

**现有修复 Transformer 未从 TE 角度系统设计**：SwinIR、HAT、DAT 等方法的注意力模块均不满足 TE

### 核心切入

从两个基本定理出发——**滑动索引**（Theorem 3.2：如果每个输出位置仅依赖其固定邻域的输入，则满足 TE）和**组件堆叠**（Theorem 3.3：TE 模块的串联和并联仍满足 TE）——重新设计注意力机制，在保持 TE 的同时解决全局感受野和线性复杂度的矛盾。

## 方法详解

### 整体框架

TEAFormer 采用经典的残差套残差结构：输入图像 $I \in \mathbb{R}^{H \times W \times 3}$ 经卷积提取浅层特征 $F_0$，通过 $N_g$ 个 Translation Equivariance Group（TEG）提取深层特征 $F_1$，最终 $F_r = F_0 + F_1$ 送入恢复模块生成高质量图像。每个 TEG 包含 $N_b$ 个 Translation Equivariance Block（TEB），每个 TEB 由一个 TEA 注意力和一个前馈网络组成。

### 关键设计

#### 1. **滑动键值自注意力（SkvSA）**
- **功能**：将 SA 的全局 KV 索引替换为基于滑动窗口的局部 KV 索引
- **核心思路**：对第 $i$ 个 query $Q_i$，仅从其滑动窗口 $[i - \frac{w \cdot s}{2}, i + \frac{w \cdot s}{2}]$ 中提取 KV 对进行注意力计算，其中 $w$ 是窗口大小，$s$ 是步长
- **边界处理**：使用 blocking 方法——边界处的 query 使用预定义的 blocking 窗口内的 KV 对
- **设计动机**：根据 Theorem 3.2，滑动窗口内的注意力计算天然满足 TE，同时保持 $O(N)$ 复杂度

#### 2. **自适应滑动键值自注意力（ASkvSA）**
- **功能**：为每个 query 自适应确定最优的 KV 索引位置
- **核心思路**：将 K 和 V 从 1D 重排为 2D，使用深度可分离卷积（kernel size $k$）生成自适应索引 $\mathcal{F} \in \mathbb{R}^{H \times W \times 2}$，在固定滑动窗口内 shuffle KV 对的位置
- **TE 保证**：convolution 满足 TE，与 SkvSA 串联后根据 Theorem 3.3 仍满足 TE
- **设计动机**：固定窗口限制了 KV 选择的灵活性，自适应索引允许模型为不同 query 找到最相关的 KV 对

#### 3. **下采样自注意力（DSA）+ 自适应融合**
- **功能**：通过下采样提供粗粒度的全局 KV 索引，弥补 ASkvSA 无法覆盖远距离像素的不足
- **核心思路**：使用 average pooling 将 K'和 V' 下采样到 $N_d$ 个 token，然后与 Q 计算全局注意力
- **融合方式**：$\text{TEA}_i(X) = \alpha_s \cdot \text{ASkvSA}(X) + \alpha_d \cdot \text{DSA}(X)$，$\alpha_s, \alpha_d$ 是可学习参数
- **设计动机**：ASkvSA 的自适应索引仅在局部窗口内 shuffle，远距离但相关的像素仍会被遗漏。DSA 以低分辨率补充全局信息

### 计算复杂度

TEA 的总 FLOPs 为 $3ND^2 + 2NDk^2 + 2Nw^2D + 2NN_dD$，当超参数 $w=15, k=3, N_d=16$ 时，计算量与 window size=16 的 WA 相当，均为 $O(N)$。

### 损失函数 / 训练策略

- 超分任务：$L_1$ loss，Adam 优化器，学习率 $2 \times 10^{-4}$，cosine scheduler，DF2K 数据集
- 去模糊/去噪：$L_1$ loss，AdamW 优化器，渐进式学习策略

## 实验关键数据

### 主实验（4× 超分辨率）

| 方法 | 参数量 | FLOPs | Urban100 PSNR | Urban100 SSIM | Manga109 PSNR |
|------|-------|-------|-------------|-------------|-------------|
| SwinIR | 11.8M | 1.848T | 27.45 | 0.8254 | 32.03 |
| HAT | 20.6M | 3.662T | 27.97 | 0.8368 | 32.48 |
| DAT | 14.7M | 2.155T | 27.87 | 0.8343 | 32.51 |
| IPG | 16.8M | 4.732T | 28.13 | 0.8392 | 32.53 |
| **TEAFormer** | **21.8M** | **1.035T** | **28.67** | **0.8489** | **32.99** |

### 消融实验（4× SR, Urban100）

| 模型 | SkvSA | ASkvSA | DSA | PSNR/SSIM | NTK Condition↓ | SRGA Value↓ | 延迟(ms) |
|------|-------|--------|-----|-----------|-------------|------------|---------|
| SwinIR (w=8) | - | - | - | 27.45/0.8254 | 1746.49 | 3.655 | 130.0 |
| SwinIR-Large (w=16) | - | - | - | 27.94/0.8362 | 1554.65 | 3.610 | 214.5 |
| TEAFormer | ✔ | - | - | 28.31/0.8444 | 243.75 | 3.206 | 230.1 |
| TEAFormer | - | ✔ | - | 28.47/0.8457 | 283.99 | 3.298 | 284.4 |
| TEAFormer | ✔ | - | ✔ | 28.49/0.8470 | 203.06 | 3.261 | 340.9 |
| **TEAFormer** | - | **✔** | **✔** | **28.67/0.8489** | **236.78** | **3.275** | 386.7 |

### 去焦模糊（DPDD 数据集，单图输入）

| 方法 | 参数量 | Indoor PSNR | Outdoor PSNR | Combined PSNR |
|------|-------|------------|-------------|-------------|
| Restormer | 26.1M | 28.87 | 23.24 | 25.98 |
| GRL-B | 19.9M | 29.06 | 23.45 | 26.18 |
| **TEAFormer** | **15.4M** | **29.50** | **23.55** | **26.45** |

### 关键发现

- **TE 对收敛和泛化的影响是实质性的**：仅引入 SkvSA（最简单的 TE 方案），NTK condition number 从 1746 降至 244（降 7.2×），SRGA value 从 3.655 降至 3.206
- **ASkvSA + DSA 的组合最优**：自适应局部索引 + 粗粒度全局信息的并行组合，在性能和效率之间取得最佳平衡
- **TEAFormer 在 Urban100 上的优势显著**：4× SR 达到 28.67 dB，比 HAT 高 0.7 dB（同等参数量），比 IPG 高 0.54 dB（更低 FLOPs 和 2× 更快推理）
- **轻量版 TEAFormer-L（829K 参数）也表现出色**：在 Urban100 上比 SwinIR-L 高 0.57 dB

## 亮点与洞察

1. **理论驱动的设计范式**：从 TE 的数学定义出发，推导出两个基本定理，再据此系统设计注意力模块——这种"定理→设计"的范式在视觉领域不多见
2. **NTK 和 SRGA 作为分析工具**：使用 NTK condition number 衡量收敛速度、SRGA 衡量泛化能力，为注意力机制的评估提供了新的度量维度
3. **打破 SA vs WA 的二选一困局**：通过自适应滑动索引 + 下采样全局注意力的并行组合，在 $O(N)$ 复杂度下同时获得灵活的局部索引和全局感受野
4. **跨任务泛化性强**：同一架构在超分、去焦模糊、去噪等多个任务上都取得 SOTA

## 局限与展望

1. **推理延迟较高**：完整 TEAFormer 的延迟为 386.7ms，比 SwinIR 的 130ms 高约 3×
2. **边界处理的 blocking 方法不完美**：边界处的 TE 是近似满足的，严格 TE 需要更精细的设计
3. **下采样使用 average pooling**：虽然效率高但仅近似满足 TE，论文提到可用 learnable polyphase 替代
4. **超参数较多**：$w, s, k, N_d$ 四个超参数需要仔细调节
5. **缺少感知质量评估**：仅使用 PSNR/SSIM，未评估 LPIPS 等感知指标

## 相关工作与启发

- ViTAE 系列的归纳偏置研究指出 ViT 缺乏归纳偏置导致训练困难，本文从 TE 角度提供了更具体的分析
- SwinIR/HAT 的窗口注意力虽然降低了复杂度但牺牲了 TE 和灵活性——TEAFormer 的滑动设计填补了这一空白
- BiFormer 和 DAT 的动态窗口选择思路与 ASkvSA 的自适应索引有相似之处，但 TEAFormer 从 TE 保证的角度提供了更严格的理论基础

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ — 首次系统研究 TE 对修复 Transformer 的影响，理论推导和设计都很扎实
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 SR/去模糊/去噪三大任务 + 完整消融 + 收敛/泛化分析
- **写作质量**: ⭐⭐⭐⭐ — 理论部分清晰，但公式密集，某些符号定义较分散
- **价值**: ⭐⭐⭐⭐⭐ — 为修复 Transformer 设计提供了新的理论视角和实用方案，具有广泛的参考价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Devil is in the Uniformity: Exploring Diverse Learners within Transformer for Image Restoration](devil_is_in_the_uniformity_exploring_diverse_learners_within_transformer_for_ima.md)
- [\[CVPR 2025\] Progressive Focused Transformer for Single Image Super-Resolution](../../CVPR2025/image_restoration/progressive_focused_transformer_for_single_image_super-resolution.md)
- [\[ECCV 2024\] Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration](../../ECCV2024/image_restoration/seeing_the_unseen_a_frequency_prompt_guided_transformer_for_image_restoration.md)
- [\[ICCV 2025\] Metric Convolutions: A Unifying Theory to Adaptive Image Convolutions](metric_convolutions_a_unifying_theory_to_adaptive_image_convolutions.md)
- [\[ICCV 2025\] Learning Pixel-adaptive Multi-layer Perceptrons for Real-time Image Enhancement](learning_pixel-adaptive_multi-layer_perceptrons_for_real-time_image_enhancement.md)

</div>

<!-- RELATED:END -->
