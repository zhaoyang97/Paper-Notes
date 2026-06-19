---
title: >-
  [论文解读] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models
description: >-
  [NeurIPS 2025 Spotlight][图像生成][数字水印] 提出 Shallow Diffuse，一种利用扩散模型后验均值预测器（PMP）的局部线性性和 Jacobian 低秩性，在扩散过程中间时间步嵌入水印的方法，实现了水印与生成过程的解耦，首次在服务端和用户端两种场景下同时保证了高一致性和高鲁棒性。
tags:
  - "NeurIPS 2025 Spotlight"
  - "图像生成"
  - "数字水印"
  - "扩散模型"
  - "低维子空间"
  - "DDIM"
  - "频域嵌入"
---

# Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2410.21088](https://arxiv.org/abs/2410.21088)  
**代码**: 暂无  
**领域**: 扩散模型 / 水印 / AI生成内容检测  
**关键词**: 数字水印, 扩散模型, 低维子空间, DDIM, 频域嵌入

## 一句话总结

提出 Shallow Diffuse，一种利用扩散模型后验均值预测器（PMP）的局部线性性和 Jacobian 低秩性，在扩散过程中间时间步嵌入水印的方法，实现了水印与生成过程的解耦，首次在服务端和用户端两种场景下同时保证了高一致性和高鲁棒性。

## 研究背景与动机

**领域现状**：扩散模型（Stable Diffusion、DALL-E、Imagen 等）驱动的商业 AI 生成内容引发了三大安全顾虑：(1) AI 生成的虚假信息危害社会稳定；(2) 训练数据记忆导致版权侵权；(3) 模型在 AI 生成内容上迭代训练导致模型坍缩。水印是识别和追踪 AI 生成内容的关键技术。

**现有痛点**：
- 传统水印方法（DWT、RivaGAN 等）主要面向后处理场景，鲁棒性不足
- 基于扩散模型的方法（Tree-Ring、RingID）将水印嵌入初始噪声的傅里叶域低频分量，但大幅改变了高斯噪声分布，导致生成图像一致性差
- 现有方法要么只支持服务端场景（需要控制初始种子），要么只支持用户端场景（后处理嵌入），无法兼顾

**核心矛盾**：鲁棒性要求水印信号强，一致性要求水印对图像改动小——两者天然矛盾。

**本文切入角度**：利用扩散模型 PMP 的 Jacobian 在中间时间步的低秩性，水印的大部分能量落入 Jacobian 的零空间，从而将水印与采样过程解耦——水印不影响预测的 $\hat{x}_0$（保证一致性），同时完整保留在 $x_t$ 中（保证可检测性）。

## 方法详解

### 整体框架

Shallow Diffuse 在扩散过程的中间时间步 $t^* = 0.3T$ 注入水印，而非在初始噪声 $x_T$ 中嵌入。工作流程：
- **服务端**：$x_T \to \text{DDIM采样到}\ x_{t^*} \to 添加水印 \to \text{DDIM采样到}\ x_0^{\mathcal{W}}$
- **用户端**：$x_0 \to \text{DDIM反转到}\ x_{t^*} \to 添加水印 \to \text{DDIM采样到}\ x_0^{\mathcal{W}}$
- **检测**：$\bar{x}_0^{\mathcal{W}} \to \text{DDIM反转到}\ x_{t^*} \to 比对水印$

### 关键设计

1. **基于低秩性的解耦原理**：PMP $\mathbf{f}_{\theta,t}(x_t)$ 预测 $\mathbb{E}[x_0|x_t]$，其 Jacobian $\mathbf{J}_{\theta,t}$ 在 $t \in [0.2T, 0.7T]$ 范围内是**低秩**的（秩比 $< 10^{-2}$），且具有**局部线性性**。当注入水印 $\lambda\Delta\mathbf{x}$ 时：
    $\mathbf{f}_{\theta,t}(x_{t^*} + \lambda\Delta\mathbf{x}) \approx \mathbf{f}_{\theta,t}(x_{t^*}) + \lambda\underbrace{\mathbf{J}_{\theta,t}(x_{t^*})\Delta\mathbf{x}}_{\approx \mathbf{0}}$
   由于 $r_{t^*} \ll d$（秩远小于维度），随机水印 $\Delta\mathbf{x}$ 的大部分能量落入零空间，$\mathbf{J}\Delta\mathbf{x} \approx 0$。因此预测的 $\hat{x}_0$ 几乎不变，保证一致性。
   
   **设计动机**：选择 $t^* = 0.3T$ 是因为此时 Jacobian 秩比最低，同时 PMP 线性度最好。

2. **高频域水印设计**：不同于 Tree-Ring/RingID 修改低频分量，Shallow Diffuse 在频域的高频区域嵌入水印：
    $\lambda\Delta\mathbf{x} = \text{DFT}^{-1}(\text{DFT}(x_{t^*}) \odot (1-\mathbf{M}) + \mathbf{W} \odot \mathbf{M}) - x_{t^*}$
   其中 $\mathbf{M}$ 是高频掩码（不进行零频中心化），$\mathbf{W}$ 是由多环高斯值构成的水印密钥。
   
   **设计动机**：(1) 高频分量能量低，修改后视觉失真小；(2) 水印嵌入在中间时间步 $x_{t^*}$（接近原图），而非纯噪声 $x_T$，高频操作更稳定。

3. **水印检测**：给定可能被攻击的图像 $\bar{x}_0^{\mathcal{W}}$，通过 DDIM 反转恢复 $\bar{x}_{t^*}^{\mathcal{W}}$，计算 p-value：
    $\eta = \frac{\text{sum}(\mathbf{M}) \cdot \|\mathbf{M} \odot \mathbf{W} - \mathbf{M} \odot \text{DFT}(\bar{x}_{t^*}^{\mathcal{W}})\|_F^2}{\|\mathbf{M} \odot \text{DFT}(\bar{x}_{t^*}^{\mathcal{W}})\|_F^2}$
   水印图像 $\eta \approx 0$，非水印图像 $\eta > \eta_0$（阈值）。

4. **T2I 扩展**：对 text-to-image 模型（如 Stable Diffusion），水印注入使用无条件 DDIM（空 prompt），与 CFG 采样过程解耦。服务端用 CFG 采样到 $x_{t^*}$，用户端用 DDIM 反转到 $x_{t^*}$。

### 理论保证

- **Theorem 1（一致性）**：水印引起的预测偏差 $\|\hat{x}_{0,t}^{\mathcal{W}} - \hat{x}_{0,t}\|_2 \leq \lambda L h(r_t)$，其中 $h(r_t) \sim \sqrt{r_t/d}$，仅依赖 Jacobian 秩 $r_t$（$r_t \ll d$），与环境维度 $d$ 弱相关。
- **Theorem 2（可检测性）**：一步 DDIM 后水印恢复误差仅与 $h(\max\{r_{t-1}, r_t\})$ 和 VP 调度参数成正比，两者均很小。

## 实验关键数据

### 主实验1：服务端场景（Stable Diffusion 2-1-base, 5000张图）

| 方法 | CLIP↑ | FID↓ | PSNR↑ | SSIM↑ | Clean TPR | 均攻击TPR↑ |
|------|-------|------|-------|-------|-----------|-----------|
| SD w/o WM | 0.3669 | 25.56 | - | - | - | - |
| Tree-Ring | 0.3645 | 25.82 | 16.61 | 0.64 | 1.00 | 0.77 |
| RingID | 0.3637 | 27.13 | 14.27 | 0.51 | 1.00 | 0.91 |
| Gaussian Shading | 0.3663 | 26.17 | 11.04 | 0.48 | 1.00 | 0.93 |
| **Shallow Diffuse** | **0.3669** | **25.60** | **35.49** | **0.96** | 1.00 | **0.93** |

Shallow Diffuse 的 PSNR 比 Tree-Ring 高 **18.88 dB**，比 RingID 高 **21.22 dB**，一致性遥遥领先。

### 主实验2：用户端场景（COCO 数据集）

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ | Clean TPR | 均攻击 |
|------|-------|-------|--------|-----------|--------|
| Tree-Ring | 28.22 | 0.57 | 0.41 | 1.00 | 0.84 |
| RingID | 12.21 | 0.38 | 0.58 | 1.00 | 0.96 |
| Gaussian Shading | 10.17 | 0.23 | 0.65 | 1.00 | 0.92 |
| RivaGAN | 40.57 | 0.98 | 0.04 | 1.00 | 0.59 |
| **Shallow Diffuse** | **32.11** | **0.84** | **0.05** | 1.00 | **0.93** |

在用户端场景中，Gaussian Shading 和 RingID 的一致性极差（PSNR 仅 10-12），而 Shallow Diffuse 维持 32+ 的高 PSNR。

### 消融实验

| 配置 | PSNR | TPR@1%FPR (avg) | 说明 |
|------|------|-----------------|------|
| $t^* = 0.3T$ (默认) | 35.49 | 0.93 | 最佳平衡点 |
| $t^* = 0.1T$ | ~38 | ~0.5 | 更接近原图但鲁棒性下降 |
| $t^* = 0.5T$ | ~25 | ~0.95 | 鲁棒性略升但一致性下降 |
| 低频水印 | ~20 | ~0.90 | 修改低频导致视觉失真 |
| 高频水印(默认) | 35.49 | 0.93 | 高频修改失真小 |

### 关键发现

- 在服务端场景中，Shallow Diffuse 几乎不影响生成质量（CLIP 和 FID 与无水印基线相当）
- Tree-Ring 和 RingID 在用户端场景下一致性极差，因为它们依赖初始噪声分布的修改
- 15 种攻击测试中，Shallow Diffuse 对失真攻击（JPEG、模糊、噪声）和对抗攻击都表现良好
- Jacobian 秩在 $t^* = 0.3T$ 附近最低（秩比 $< 10^{-2}$），验证了理论分析

## 亮点与洞察

- 利用 PMP 的低秩结构是非常优雅的切入点——零空间中的水印既不影响生成又能保持存在
- 同时适用于服务端和用户端两种场景，这一灵活性是已有方法所不具备的
- 理论分析（Theorem 1 & 2）为一致性和可检测性提供了严格保证
- 整个方法无需训练（training-free），只需现成的扩散模型即可使用

## 局限与展望

- 水印容量有限（当前设计嵌入单个密钥，多密钥识别在附录中讨论但场景受限）
- DDIM 反转的精度影响检测质量，尤其对高 CFG 强度或超长推理步数的模型
- 对非 DDIM 采样器（如 DPM-Solver、Euler 等）的兼容性有待验证
- 面对高级自适应攻击（如知道 $t^*$ 的对手重新噪声化再去噪声）可能需要额外防御

## 相关工作与启发

- 与 Tree-Ring Watermarks 的本质区别：Tree-Ring 在初始噪声的低频域嵌入，强耦合于采样过程；Shallow Diffuse 在中间时间步的高频域嵌入，解耦于采样
- 与图像隐写术的交叉：可利用扩散模型的低维子空间结构设计更高容量的隐写方案
- 对扩散模型 PMP 低秩性的深入利用，也可启发基于此性质的图像编辑、风格迁移等应用

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 利用 PMP 低秩结构实现水印解耦，视角非常独特
- **实验充分度**: ⭐⭐⭐⭐⭐ 15种攻击、两种场景、多个数据集、完整消融
- **写作质量**: ⭐⭐⭐⭐ 图示清晰直观，理论推导完整
- **价值**: ⭐⭐⭐⭐⭐ 同时解决一致性和鲁棒性矛盾，实用性极强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[ICCV 2025\] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](../../ICCV2025/image_generation/invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [\[ICML 2025\] Distillation of Discrete Diffusion through Dimensional Correlations (Di4C)](../../ICML2025/image_generation/distillation_of_discrete_diffusion_through_dimensional_correlations.md)
- [\[ICML 2026\] Diffusion Models Are Statistically Optimal for Learning Low-Dimensional Multi-Modal Distributions](../../ICML2026/image_generation/diffusion_models_are_statistically_optimal_for_learning_low-dimensional_multi-mo.md)
- [\[ECCV 2024\] Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](../../ECCV2024/image_generation/robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)

</div>

<!-- RELATED:END -->
