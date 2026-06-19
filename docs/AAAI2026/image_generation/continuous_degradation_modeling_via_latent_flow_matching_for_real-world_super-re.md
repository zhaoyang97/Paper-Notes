---
title: >-
  [论文解读] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution
description: >-
  [AAAI 2026][图像生成][真实世界超分辨率] 提出 DegFlow，通过残差自编码器 + 潜空间 Flow Matching 从离散尺度的真实 HR-LR 对学习连续退化轨迹，仅需单张 HR 图像即可合成任意连续尺度的逼真 LR 图像，用于训练超分模型达到 SOTA。 深度学习超分辨率方法在 bicubic 降采样…
tags:
  - "AAAI 2026"
  - "图像生成"
  - "真实世界超分辨率"
  - "退化建模"
  - "Flow Matching"
  - "连续尺度"
  - "潜空间"
---

# Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution

**会议**: AAAI 2026  
**arXiv**: [2602.04193](https://arxiv.org/abs/2602.04193)  
**代码**: [GitHub](https://github.com/present091/DegFlow)  
**领域**: 图像生成  
**关键词**: 真实世界超分辨率, 退化建模, Flow Matching, 连续尺度, 潜空间

## 一句话总结

提出 DegFlow，通过残差自编码器 + 潜空间 Flow Matching 从离散尺度的真实 HR-LR 对学习连续退化轨迹，仅需单张 HR 图像即可合成任意连续尺度的逼真 LR 图像，用于训练超分模型达到 SOTA。

## 研究背景与动机

深度学习超分辨率方法在 bicubic 降采样等合成退化下表现出色，但在真实照片上表现不佳，因为真实退化包含未知模糊、噪声和压缩伪影等复杂非线性组合。现有解决路线各有不足：

**手工退化管线**（Real-ESRGAN、BSRGAN）：用模糊核+噪声+下采样+压缩伪影组合增强训练数据，提升了鲁棒性但仍无法表示真实退化的复杂特性。

**物理设备采集**（RealSR、DRealSR、RealArbiSR）：用 DSLR 相机变焦镜头采集配对数据，退化真实但采集成本高、场景多样性有限。

**学习退化模型**（DeFlow、RealDGen）：从少量真实对中学习退化过程并合成 LR 图像，但缺乏**显式尺度控制**，无法生成任意连续尺度的退化。

**InterFlow**：通过在 LR 潜空间插值实现连续尺度合成，但推理时需要两个不同尺度的 LR 图像作为输入，实用性受限。

DegFlow 的优势：从真实 HR-LR 对学习连续退化，**仅需 HR 图像即可推理**，支持任意连续尺度，且退化效果更逼真。

## 方法详解

### 整体框架

DegFlow 采用两阶段顺序训练管线：

- **Stage 1 - Residual Autoencoder (RAE)**：将图像映射到紧凑潜空间，通过 HR 跳接保留细节
- **Stage 2 - Latent Flow Matching (LFM)**：在潜空间中学习连续退化轨迹

推理时：HR 图像 → 编码器 → 潜表示 → FM 模型沿连续时间步演化 → 解码器 → 任意尺度 LR 图像。

### 关键设计

#### 1. Residual Autoencoder (RAE)

编码器 $E_\theta$ 将输入图像 $I \in \mathbb{R}^{C \times H \times W}$（无论 HR 或 LR）映射为紧凑潜码 $z = E_\theta(I) \in \mathbb{R}^{Cr^2 \times H/r \times W/r}$，其中 r 为空间压缩因子。

为弥补高压缩比造成的细节损失，引入**多尺度残差跳接**：

$$\hat{I} = D_\theta(z; H_{\text{HR}})$$

其中 $H_{\text{HR}} = \{h^{(l)}_{\text{HR}}\}_{l=1}^L$ 是从 **HR 图像**提取的多尺度隐特征。关键点：跳接只注入 HR 特征，不管输入是 HR 还是 LR。这样潜码只需编码 LR 与 HR 特征之间的**残差信息**（即退化特定信息），为后续 FM 模型提供信息丰富的表示。

重建损失对 HR 和 LR 输入同时施加：

$$\mathcal{L}_{\text{Recon}} = \|D_\theta(E_\theta(I_{s_1}); H_{\text{HR}}) - I_{s_1}\|_2^2 + \|D_\theta(E_\theta(I_{s_k}); H_{\text{HR}}) - I_{s_k}\|_2^2$$

#### 2. Latent Flow Matching (LFM)

RAE 训练冻结后，将配对 HR-LR 图像编码到潜空间并构建连续退化轨迹。

**尺度归一化**：将退化等级 $s_k$ 线性映射到时间戳 $t_k = (s_k - s_1)/(s_m - s_1)$，归一化到 [0,1]。例如 $\mathcal{S}=\{1,2,4\}$ 对应 $t_1=0, t_2=1/3, t_4=1$。

**Natural Cubic Spline 轨迹**：简单的分段线性插值会偏离非线性的潜流形，且导数不连续违反 ODE 光滑性假设。本文采用自然三次样条在子区间 $[t_k, t_{k+1}]$ 上插值：

$$\mu_t(\epsilon) = a_k(\epsilon)(t-t_k)^3 + b_k(\epsilon)(t-t_k)^2 + c_k(\epsilon)(t-t_k) + d_k(\epsilon)$$

保证一阶和二阶导数连续，满足 flow matching 的正则性要求。

**感知损失（辅助监督）**：中间尺度（如 ×3.34）没有 ground truth LR，通过三阶 Taylor 展开将中间时间步的预测外推到最近的训练退化等级 $s_{k+1}$，然后用 LPIPS 损失提供感知监督：

$$\mathcal{L}_{\text{LPIPS}} = \text{LPIPS}(I_{s_{k+1}}, D_\theta(\hat{z}_{t_{k+1}}))$$

LPIPS 损失通过 Taylor 展开反传梯度到 LFM 网络参数。

### 损失函数 / 训练策略

**Stage 1 (RAE)**：Adam 优化器，200k 迭代，cosine 退火学习率 1e-4→1e-7，batch=16，256×256 patch + 随机翻转。

**Stage 2 (LFM)**：Adam 优化器，400k 迭代，cosine 退火 2e-4→1e-7，batch=32，256×256 patch。
总损失：$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CFM}} + \lambda \mathcal{L}_{\text{LPIPS}}$，$\lambda = 0.1$。

训练集：RealSR-V2 Canon 训练集（×1, ×2, ×4 配对）。

## 实验关键数据

### 主实验

**表2：固定尺度 SR 结果（RealSR ×3 测试集，训练时无 ×3 数据）**

| 模型 | 训练数据 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|---|
| HAT | RealSR ×3 (Oracle) | 30.71 | 0.8645 | 0.3221 |
| HAT | RealSR ×2,×4 | 30.39 | 0.8607 | 0.3248 |
| HAT | InterFlow ×2~×4 | 30.65 | 0.8645 | 0.3135 |
| HAT | **Ours ×2~×4** | **30.86** | **0.8668** | **0.3186** |
| MambaIR | RealSR ×3 (Oracle) | 30.62 | 0.8636 | 0.3208 |
| MambaIR | InterFlow ×2~×4 | 30.51 | 0.8625 | 0.3138 |
| MambaIR | **Ours ×2~×4** | **30.73** | **0.8686** | **0.3152** |

DegFlow 合成的训练集让 HAT 达到 30.86 dB，**超越了使用真实 ×3 数据训练的 Oracle**（30.71 dB）。

**表3：任意尺度 SR 结果（RealSR ×3 + RealArbiSR 多尺度）**

| 模型 | 方法 | RealSR ×3 PSNR | RealArbiSR ×2.5 PSNR |
|---|---|---|---|
| MetaSR | Bicubic | 28.99 | 30.05 |
| MetaSR | InterFlow | 30.42 | 30.71 |
| MetaSR | **Ours** | **30.58** | **30.88** |
| LIIF | InterFlow | 30.44 | 30.70 |
| LIIF | **Ours** | **30.61** | **30.99** |

在任意尺度 SR 中也一致超越 InterFlow，且仅需 HR 输入。

### 消融实验

**表4：各组件贡献**

| 方法 | PSNR↑ | SSIM↑ | LPIPS↓ |
|---|---|---|---|
| 基线（分段线性轨迹） | 30.58 | 0.8640 | 0.3214 |
| + 非线性轨迹（三次样条） | 30.68 | 0.8652 | 0.3209 |
| + LPIPS 三阶 Taylor 近似 | 30.81 | 0.8662 | 0.3200 |
| + RAE 的 HR 跳接 | **30.86** | **0.8668** | **0.3186** |

每个组件贡献清晰：三次样条 > LPIPS 感知监督 > HR 跳接。

### 关键发现

1. **时间步-退化对应分析**：在 RealSR ×3 测试集上，PSNR 和 FID 在 t≈0.73 (s≈3.2) 处峰值，CLIP 在 t≈0.70 (s≈3.1) 处峰值，验证了 DegFlow 学到了有意义的连续退化流形。
2. **外部 HR 数据增强**：用 DIV2K HR 图像额外合成 LR 训练数据可进一步提升 PSNR 0.14 dB（30.86→31.00），展示了仅需 HR 数据就能扩展训练集的优势。
3. 三次样条的导数连续性对 flow matching 的 ODE 求解至关重要。

## 亮点与洞察

- **仅需 HR 图像推理**：对比 InterFlow 需要两个尺度的 LR 输入，DegFlow 只用 HR 更实用。
- **连续退化流形**：三次样条 + flow matching 的组合优雅，在连续性和非线性建模间取得平衡。
- **残差潜空间设计**：通过 HR 跳接使潜码只编码退化信息，大幅简化了 FM 模型的学习任务。
- **LPIPS Taylor 近似**：为无 GT 的中间尺度提供感知监督的巧妙方案。

## 局限与展望

1. 训练依赖真实配对数据集（RealSR），如果目标域相机与训练域差异大可能泛化不佳。
2. RAE 的空间压缩因子 r 需要权衡计算成本和细节保留。
3. 仅训练了 ×2 和 ×4 两个离散点，更多离散点是否能进一步改善连续轨迹待探索。
4. 未与基于扩散模型的退化建模方法（RealDGen）在退化真实性上做直接对比。

## 相关工作与启发

- 与 DeFlow/NAFlow 等 normalizing flow 方法相比，flow matching 更灵活且不需要可逆架构约束。
- 与 InterFlow 相比，DegFlow 的核心优势在推理端：只需 HR 即可生成任意尺度 LR。
- 潜空间退化建模的范式可推广到其他退化任务：去噪、去雾、去模糊等。
- 三次样条替代线性插值的思路在 flow matching 框架中值得推广。

## 评分

- **新颖性**: ★★★★☆ — 潜空间 flow matching + 三次样条轨迹建模是新颖的退化建模方案
- **技术深度**: ★★★★☆ — RAE+LFM 两阶段设计合理，LPIPS Taylor 近似巧妙
- **实验**: ★★★★☆ — 固定/任意尺度 SR 均验证，超越 Oracle 说服力强
- **写作**: ★★★★☆ — 结构清晰，对比表全面
- **实用性**: ★★★★★ — 仅需 HR 推理，开源代码，直接可用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[AAAI 2026\] Symmetrical Flow Matching: Unified Image Generation, Segmentation, and Classification with Score-Based Generative Models](symmetrical_flow_matching_unified_image_generation_segmentation_and_classificati.md)
- [\[ICML 2026\] Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](../../ICML2026/image_generation/q-dit4sr_exploration_of_detail-preserving_diffusion_transformer_quantization_for.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)

</div>

<!-- RELATED:END -->
