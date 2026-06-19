---
title: >-
  [论文解读] MaskControl: Spatio-Temporal Control for Masked Motion Synthesis
description: >-
  [ICCV 2025][图像生成][Masked Motion Model] MaskControl 首次将空间关节可控性引入生成式掩码运动模型（Masked Motion Model），通过训练时的 Logits Regularizer 隐式对齐运动 token 分布与目标关节位置，以及推理时的 Logits Optimization 显式优化预测 logits 以最小化控制误差，在保持高生成质量（FID 降低77%）的同时实现高精度关节控制（平均误差0.91cm），并支持零样本目标函数控制。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "Masked Motion Model"
  - "关节控制"
  - "Logits优化"
  - "可微期望采样"
  - "零样本目标控制"
---

# MaskControl: Spatio-Temporal Control for Masked Motion Synthesis

**会议**: ICCV 2025  
**arXiv**: [2410.10780](https://arxiv.org/abs/2410.10780)  
**代码**: [项目主页](https://www.ekkasit.com/ControlMM-page/)  
**领域**: 运动生成/空间控制  
**关键词**: Masked Motion Model, 关节控制, Logits优化, 可微期望采样, 零样本目标控制

## 一句话总结
MaskControl 首次将空间关节可控性引入生成式掩码运动模型（Masked Motion Model），通过训练时的 Logits Regularizer 隐式对齐运动 token 分布与目标关节位置，以及推理时的 Logits Optimization 显式优化预测 logits 以最小化控制误差，在保持高生成质量（FID 降低77%）的同时实现高精度关节控制（平均误差0.91cm），并支持零样本目标函数控制。

## 研究背景与动机

### 核心问题
文本驱动的人体运动生成虽具有语义丰富的优势，但文本描述无法精确指定特定关节（如骨盆、手部）的空间位置。在动画、VR/AR、机器人等应用中，精确控制关节轨迹至关重要。

### 现有方法的不足

现有可控运动生成方法几乎全部基于扩散模型（Diffusion Model），存在以下关键问题：

1. **质量与控制精度难以兼顾**：GMD 和 OmniControl 等方法在控制精度和生成质量之间存在明显 trade-off，FID 分数远高于纯文本生成模型
2. **稀疏/稠密信号不兼顾**：部分方法擅长稀疏路径点，部分擅长逐帧密集轨迹，难以统一
3. **无法零样本适应新目标**：扩散方法依赖预训练的控制策略，无法在推理时适应任意目标函数
4. **计算效率低**：在原始运动空间做扩散过程存在冗余，导致生成速度慢

### 本文洞察

掩码运动模型（如 MoMask）通过训练分类器预测被掩盖的 token，然后从学到的分类分布中采样生成运动。这种基于 logits 的生成范式天然适合引入控制信号——可以通过直接操纵分类器的 logits 来改变 token 分布，使生成的运动对齐控制信号。

## 方法详解

### 整体框架

MaskControl 包含四个核心组件：

1. **Motion Tokenizer**：将运动序列编码为离散 token
2. **Logits Regularizer**：训练时隐式扰动 logits 以对齐控制信号
3. **Logits Optimization**：推理时显式优化 logits 以提高控制精度
4. **Differentiable Expectation Sampling (DES)**：解决分类采样不可微的问题

### 关键设计 1：Logits Regularizer（训练时控制）

**架构设计**：采用类似 ControlNet 的设计理念——首次将此原理应用于掩码生成模型。具体地，创建预训练掩码运动模型的可训练副本，每个 Transformer 层与原始模型的对应层通过**零初始化线性层**连接。副本接受两种条件：文本 $W$（通过注意力机制）和关节控制信号 $S$（通过投影层直接加到 token 序列上）。

**关节控制信号**：$S = [s_1, s_2, \ldots, s_F]$，其中 $s_i \in \mathbb{R}^{j \times 3}$ 指定第 $i$ 帧中需要控制的关节的 3D 坐标，未控制关节置零。

**运动一致性损失**：评估生成运动与输入控制信号之间的对齐程度：

$$L_s(e_c, s) = \frac{\sum_n \sum_j \sigma_{nj} \odot \|s_{nj} - R(D(e_c))\|}{\sum_n \sum_j \sigma_{nj}}$$

其中 $\sigma_{nj}$ 为二值掩码指示是否有控制值，$D(\cdot)$ 为 motion tokenizer 解码器，$R(\cdot)$ 将局部坐标变换为全局坐标。

**Logits 一致性损失**：将标准掩码重建损失扩展到所有位置（包括未掩盖位置），条件化于文本 $W$ 和关节信号 $S$：

$$\mathcal{L}_{\text{logits}} = -\sum_{\forall i \in [1,L]} \log p(x_i | X_{\overline{M}}, W, S)$$

**总损失**：$\mathcal{L} = \alpha \mathcal{L}_{\text{logits}} + (1-\alpha) L_s(e_c, s)$

### 关键设计 2：Logits Optimization（推理时控制）

在推理时进一步优化 logits 以增强控制精度。关键思想：在 unmasking 过程的每一步，将 Regularizer 输出的 logits 作为初始值进行梯度下降优化：

$$l_{m+1} = l_m - \eta \nabla_{l_m} L_s(l_m, s)$$

迭代 $I$ 次后得到优化后的 logits $l^+$，再按正常流程采样。在最后一步 unmasking 时，还可直接优化 codebook embedding：

$$e_{m+1} = e_m - \eta \nabla_{e_m} L_s(e_m, s)$$

这种设计的核心优势：**$L_s$ 可以替换为任意可微目标函数**，实现零样本目标控制（如限制人物在方形区域内行走等）。

### 关键设计 3：Differentiable Expectation Sampling (DES)

**动机**：Logits Regularizer 和 Optimization 都需要对 logits 求梯度，但从分类分布中采样 token 是不可微的操作。

**方案一 — Gumbel-Softmax**：用 Straight-Through Gumbel-Softmax 进行可微采样：

$$p_\theta(x_k | \cdot) = \frac{\exp((\ell_k + g_k)/\tau)}{\sum_{j=1}^K \exp((\ell_j + g_j)/\tau)}$$

**方案二 — Token 期望**：用 codebook 向量的加权平均替代 argmax 查表，实现可微的嵌入重建：

$$\mathbb{E}[X_{recon}] = \sum_{k=1}^K p_\theta(x_k | X_{\overline{M}}, W, S) \cdot c_k$$

## 实验

### 主实验：关节控制运动生成（HumanML3D 数据集）

| 方法 | 基础模型 | FID ↓ | R-Precision Top-3 ↑ | 平均误差(cm) ↓ | 轨迹误差>50cm(%) ↓ | 零样本目标 |
|------|---------|-------|---------------------|----------------|-------------------|----------|
| GMD | Motion Diffusion | 0.576 | 0.665 | 14.39 | 9.31 | - |
| OmniControl | Motion Diffusion | 0.218 | 0.687 | 3.38 | 3.87 | ✗ |
| MotionLCM | Latent Diffusion | 0.531 | 0.752 | 18.97 | 18.87 | ✗ |
| TLControl | Feed Forward | 0.271 | 0.779 | 1.08 | 0.00 | ✗ |
| **MaskControl** | **Masked Model** | **0.061 (-77%)** | **0.809** | **0.98** | **0.00** | **✓** |

关键发现：
- MaskControl 在 FID 上大幅领先（0.061 vs 前 SOTA 0.218），说明掩码模型在控制场景中生成质量远优于扩散模型
- 平均误差 0.98cm（多关节）/ 0.91cm（仅骨盆），达到亚厘米级控制精度
- 唯一支持零样本目标函数控制的方法

### 零样本目标控制对比

| 任务 | 方法 | 约束误差 ↓ | 失败率 ↓ | FID ↓ |
|------|------|-----------|---------|-------|
| 头部高度约束 | ProgMoGen | 0.012 | 8.8% | 0.556 |
| 头部高度约束 | **MaskControl** | **0.000** | **0.0%** | **0.246** |
| 方形区域内行走 | ProgMoGen | 0.012 | - | - |
| 方形区域内行走 | **MaskControl** | **0.000** | - | - |

MaskControl 在 HSI 任务上约束误差均为 0，且 FID 远优于 ProgMoGen，说明通过 Logits Optimization 调整 token 分布比直接约束扩散噪声更有效。

### 消融实验

| 配置 | FID ↓ | 平均误差(cm) ↓ |
|------|-------|----------------|
| 无控制 | 0.095 | 63.18 |
| 仅 Logits Optimization（无 Regularizer） | 0.142 | 2.18 |
| 仅 Logits Regularizer（无 Optimization） | 0.128 | 40.41 |
| **完整模型** | **0.061** | **0.98** |

关键发现：
- 移除 Regularizer 后 FID 恶化最严重（0.142），说明 Regularizer 对生成质量至关重要
- 移除 Optimization 后控制误差飙升（40.41），说明推理时优化对精度不可或缺
- 两者互补：Regularizer 保质量，Optimization 保精度

## 亮点与洞察

1. **范式创新**：首次将可控性引入掩码运动模型，开辟了运动控制的新技术路线，避开了扩散模型在质量-精度 trade-off 上的困境
2. **Logits 操纵 = 分布操纵**：通过直接修改分类 logits 来间接调整生成分布，比在连续噪声空间做引导更直接有效
3. **DES 的通用价值**：可微期望采样解决了离散 token 模型的梯度传递问题，不仅限于运动控制，对所有基于 VQ 的生成模型都有参考价值
4. **统一推理框架**：同一模型同时支持 any-joint-any-frame 控制、body-part 时间线控制和零样本目标控制

## 局限性

1. 依赖运动 tokenizer 的质量——如果 VQ 编码损失关键关节信息，控制精度会受限
2. Logits Optimization 在推理时增加了计算开销（多轮梯度下降）
3. 实验仅在 HumanML3D 数据集上验证，未扩展到更复杂的多人交互场景

## 相关工作

- **掩码运动模型**：MoMask、MMM 等建立了通过 bidirectional context 解码掩盖 token 的范式，生成质量和效率已超过扩散方法
- **扩散可控运动**：GMD（仅骨盆）→ OmniControl（任意关节+ControlNet）→ TLControl（高精度但低质量），全部基于扩散
- **零样本控制**：DNO 在扩散噪声上优化，ProgMoGen 在 MDM 上加约束，但都牺牲生成质量

## 评分
- 新颖性：⭐⭐⭐⭐⭐（首次将控制引入掩码运动模型，方法论创新显著）
- 技术深度：⭐⭐⭐⭐⭐（DES、Logits Regularizer/Optimization 理论完整）
- 实验充分度：⭐⭐⭐⭐⭐（多任务、消融、定性定量全面对比）
- 实用价值：⭐⭐⭐⭐（支持多种控制模式，零样本泛化能力强）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Self-Supervised ControlNet with Spatio-Temporal Mamba for Real-World Video Super-Resolution](../../CVPR2025/image_generation/self-supervised_controlnet_with_spatio-temporal_mamba_for_real-world_video_super.md)
- [\[ICCV 2025\] Latent Diffusion Models with Masked AutoEncoders](latent_diffusion_models_with_masked_autoencoders.md)
- [\[ICCV 2025\] LazyMAR: Accelerating Masked Autoregressive Models via Feature Caching](lazymar_accelerating_masked_autoregressive_models_via_feature_caching.md)
- [\[ICCV 2025\] Video Motion Graphs](video_motion_graphs.md)
- [\[ICML 2025\] Broadband Ground Motion Synthesis by Diffusion Model with Minimal Condition](../../ICML2025/image_generation/broadband_ground_motion_synthesis_by_diffusion_model_with_minimal_condition.md)

</div>

<!-- RELATED:END -->
