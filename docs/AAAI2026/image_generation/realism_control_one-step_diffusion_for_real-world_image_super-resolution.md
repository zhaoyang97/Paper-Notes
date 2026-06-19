---
title: >-
  [论文解读] Realism Control One-step Diffusion for Real-World Image Super-Resolution
description: >-
  [AAAI 2026 Oral][图像生成][图像超分辨率] 提出 RCOD 框架，通过潜在域分组策略和退化感知采样，赋予单步扩散（OSD）超分辨率方法在推理阶段灵活控制保真度-真实感平衡的能力，同时引入视觉提示注入模块替代文本提示来提升恢复精度。 领域现状 真实世界图像超分辨率（Real-ISR）旨在从未知退化的低分辨率图…
tags:
  - "AAAI 2026 Oral"
  - "图像生成"
  - "图像超分辨率"
  - "单步扩散"
  - "真实感控制"
  - "退化感知"
  - "潜在域分组"
---

# Realism Control One-step Diffusion for Real-World Image Super-Resolution

**会议**: AAAI 2026 Oral  
**arXiv**: [2509.10122](https://arxiv.org/abs/2509.10122)  
**代码**: [https://zongliang-wu.github.io/RCOD-SR](https://zongliang-wu.github.io/RCOD-SR)  
**领域**: 图像生成  
**关键词**: 图像超分辨率, 单步扩散, 真实感控制, 退化感知, 潜在域分组

## 一句话总结

提出 RCOD 框架，通过潜在域分组策略和退化感知采样，赋予单步扩散（OSD）超分辨率方法在推理阶段灵活控制保真度-真实感平衡的能力，同时引入视觉提示注入模块替代文本提示来提升恢复精度。

## 研究背景与动机

### 领域现状

真实世界图像超分辨率（Real-ISR）旨在从未知退化的低分辨率图像中恢复高分辨率图像。基于预训练 Stable Diffusion (SD) 的方法（如 DiffBIR、StableSR、SeeSR）通过迭代潜在空间优化取得了出色的感知质量，但多步采样导致的延迟使其难以用于实时应用。

### 现有痛点

为解决效率问题，单步扩散方法（OSEDiff、S3Diff）通过知识蒸馏将多步扩散先验压缩为单步推理，实现了 10-100 倍加速。然而，**OSD 方法面临一个核心矛盾**：
- **多步扩散**可以通过调整采样步数来灵活平衡保真度（fidelity）和真实感（realism）
- **单步扩散**因训练时使用固定时间步 $T$，模型只能学习到"平均退化"的恢复，缺乏对场景特定需求的自适应能力
- 现有 OSD 方法只能输出一个结果，无法满足不同场景下对保真度/真实感的差异化需求

### 核心矛盾

OSD 方法将所有不同退化程度的 LR 输入用单一固定时间步训练，导致模型收敛到一个受限的域内（confined domain），只能生成固定程度的细节，失去了多步方法的灵活性。

### 切入角度

既然时间步条件是扩散去噪网络中不可移除的组件，且它控制着噪声潜在特征的均值和方差，那么可以通过**根据退化程度分配不同时间步**来让单步扩散也获得可控的生成能力。

## 方法详解

### 整体框架

RCOD（Realism Controlled One-step Diffusion）框架包含三个核心组件：
1. **潜在域分组策略（LDG）**：根据退化程度将训练数据分组，分配不同时间步
2. **退化感知采样蒸馏（DAS）**：在蒸馏正则化中对齐 LDG 的分组信息
3. **视觉提示注入模块（VPIM）**：用退化感知的视觉 token 替代文本提示

该框架可以方便地集成到现有 OSD 方法中（如 OSEDiff 和 S3Diff）。

### 关键设计

#### 1. **潜在域分组（Latent Domain Grouping, LDG）**

**核心思路**：不使用固定时间步 $T$，而是根据潜在度量 $M_L$ 自适应选择时间步 $t$：

$$\hat{z}_H = F_\theta(z_L; t, c_y)$$

$$t = k \cdot \left(n - \left\lfloor \frac{n \cdot (M_L - M_{L\text{-min}})}{M_{L\text{-max}} - M_{L\text{-min}}} \right\rfloor \right), \quad k \in \mathbb{Z}^+$$

其中 $M_L$ 是衡量 LR 特征退化程度的潜在度量，$n$ 是分组数（设为 $\leq 4$），$k=250$ 是时间步间隔。

**设计动机**：较大的时间步意味着去噪过程中更强的生成能力（如 SD-Turbo 中 $t$ 越大生成的细节越多）。通过分组策略，退化严重的样本分配更高时间步（更强生成），退化轻微的分配较低时间步（保持保真度）。推理时，用户可直接选择时间步来单调地控制真实感级别。

**潜在度量 $M_L$**：选用余弦相似度作为度量：

$$M_L = \frac{z_L \cdot z_H}{\|z_L\| \|z_H\|}$$

实验表明余弦相似度比 L1/MSE 距离与客观（SSIM）、感知（DISTS）、语义（CLIPIQA）指标的相关系数更高（Spearman 系数分别为 0.78/0.42/0.27 vs L1 的 0.60/0.15/0.06）。

#### 2. **退化感知采样蒸馏（Degradation-Aware Sampling, DAS）**

**核心思路**：在 VSD 蒸馏过程中，将正则化网络的时间步采样与 LDG 对齐：

$$t_r = S(\max(20, t-k), \min(980, t+k))$$

其中 $t_r$ 是正则化网络采样的时间步，$t$ 是 LDG 选择的 OSD 时间步，$S(t_{min}, t_{max})$ 表示均匀随机采样。

**设计动机**：原始 VSD 方法在广范围（20-980）采样时间步用于正则化，这未考虑退化信息。DAS 将采样范围约束在 LDG 分组的邻域内，使蒸馏正则化强度与退化程度对齐。

#### 3. **视觉提示注入模块（Visual Prompt Injection Module, VPIM）**

**核心思路**：用 CLIP 视觉模型 + MLP 替代文本编码器（CLIP 文本模型），直接从 LR 图像提取退化感知视觉 token 作为 U-Net 交叉注意力的输入。

**设计动机**：
- 文本提示（尤其是配合 VLM 的方案）增加计算开销，且文本描述可能与图像内容不完全对齐
- 视觉提示直接绑定到图像像素特征，同时提升保真度和真实感
- 消除对 VLM 的依赖，降低推理延迟

#### 4. **度量估计模块（MEM）**

用于推理阶段自适应时间步选择。利用预训练模型中间层特征和简单 MLP 估计 $M_L$，独立于 OSD 训练。

### 损失函数 / 训练策略

总损失函数（以 RCOD_O 为例）：

$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda_2 \mathcal{L}_{reg} + \lambda_3 \mathcal{L}_{diff} + \lambda_4 \mathcal{L}_{cls}$$

其中：
- $\mathcal{L}_{data} = \mathcal{L}_{MSE}(\hat{x}_H, x_H) + \lambda_1 \mathcal{L}_{LPIPS}(\hat{x}_H, x_H)$：数据一致性损失
- $\mathcal{L}_{reg}$：VSD 正则化损失（配合 DAS 策略）
- $\mathcal{L}_{diff}$：正则化器微调损失
- 使用 LoRA 进行参数高效微调

训练配置：30K+ 迭代，batch size 4，学习率 $2 \times 10^{-5}$。推理时 $n=3$ 对应三个级别：Fidelity ($t=250$)、Neutral ($t=500$)、Realism ($t=750$)。

## 实验关键数据

### 主实验

在 DRealSR（真实世界数据）上的表现：

| 方法 | PSNR↑ | SSIM↑ | MANIQA↑ | CLIPIQA↑ | 说明 |
|------|-------|-------|---------|----------|------|
| OSEDiff | 27.92 | 0.7835 | 0.5899 | 0.6963 | 原始基线 |
| RCOD_O-Fid. | **28.90** | **0.7906** | 0.6275 | 0.7023 | 保真模式优于基线 |
| RCOD_O-Neu. | 28.30 | 0.7775 | 0.6385 | 0.7179 | 中性模式 |
| RCOD_O-Real. | 27.59 | 0.7600 | **0.6295** | **0.7325** | 真实感模式 |
| S3Diff | 27.54 | 0.7491 | 0.6134 | 0.7130 | 原始基线 |
| RCOD_S-Adap. | 27.83 | 0.7661 | 0.6223 | 0.7110 | 自适应模式 |

在 RealSR 数据上，RCOD_O-Fid. 达到 PSNR 26.01（OSEDiff 为 25.15），MANIQA 0.6647（OSEDiff 为 0.6326）。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 余弦相似度 vs L1 | Spearman(SSIM): 0.78 vs 0.60 | CS 与图像质量指标相关性最强 |
| 余弦相似度 vs MSE | Spearman(DISTS): 0.42 vs 0.11 | CS 在感知指标上优势更明显 |
| 不同退化管线训练 | Orig: PSNR 25.15, New Deg: 24.59 | 退化越强→PSNR 降但 MANIQA 升 |

### 效率分析

| 方法 | 推理时间(s) | 可训练参数(M) | PSNR | MANIQA |
|------|------------|-------------|------|--------|
| PiSA-SR | 0.13 | 8.1 | 28.31 | 0.6156 |
| OSEDiff | 0.11 | 8.5 | 27.92 | 0.5899 |
| RCOD_O-Fid. | **0.09** | 9.5 | **28.90** | **0.6275** |
| S3Diff | 0.28 | 34.5 | 27.54 | 0.6134 |

### 关键发现

- 真实感水平随时间步**单调递增**，验证了 LDG 策略的有效性
- RCOD 即使在保真模式下，NR 指标也优于原始方法，说明分组策略本身就能提升整体性能
- VPIM 替代文本提示后推理速度反而更快（不需要文本编码器/VLM）
- 自适应模式（MEM 估计时间步）的 $M_L$ 大多接近 1，与训练数据分布一致

## 亮点与洞察

1. **极简而有效的思路**：核心就是"根据退化程度分配时间步"，通过最基础的扩散条件（时间步）实现可控生成，无需额外可训练参数
2. **通用性强**：框架级别的方法，已验证可以集成到 OSEDiff 和 S3Diff 两种不同架构的 OSD 方法上
3. **余弦相似度的优越性**：通过 Spearman 相关系数分析，说明 CS 比 L1/MSE 更能捕捉高维潜在特征的退化信息
4. **视觉提示 > 文本提示**：消除了对 VLM 的依赖，同时提升了保真度和真实感

## 局限与展望

- 分组数 $n$ 固定为 3-4，更细粒度的分组是否能获得更好的控制效果值得探索
- MEM 估计模块的精度直接影响自适应模式效果，目前用简单 MLP，可考虑更强的估计网络
- 仅验证了 SD 系列模型，尚未在 DiT 架构上验证
- 推理时需要用户手动选择模式或依赖 MEM，缺乏根据内容自动决策的机制

## 相关工作与启发

- 与 PiSA-SR（两个 LoRA + 两步扩散控制保真度/真实感）相比，RCOD 更简单高效（单步+无额外参数）
- 与 OFTSR（轨迹对齐蒸馏）相比，RCOD 具有退化感知能力
- 启发：在其他条件生成任务中，也可以通过分组条件（如噪声级别、引导强度）来实现可控生成

## 评分

- 新颖性: ⭐⭐⭐⭐ （思路简单但有效，核心 idea 清晰）
- 实验充分度: ⭐⭐⭐⭐⭐ （多数据集、多基线、多指标、消融全面）
- 写作质量: ⭐⭐⭐⭐ （结构清晰，动机和方法阐述详细）
- 价值: ⭐⭐⭐⭐ （解决了 OSD 方法一个切实的限制，且通用性好）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[NeurIPS 2025\] DOVE: Efficient One-Step Diffusion Model for Real-World Video Super-Resolution](../../NeurIPS2025/image_generation/dove_efficient_one-step_diffusion_model_for_real-world_video_super-resolution.md)
- [\[AAAI 2026\] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution](continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)
- [\[CVPR 2026\] Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution](../../CVPR2026/image_generation/bridging_fidelity-reality_with_controllable_one-step_diffusion_for_image_super-r.md)
- [\[ICML 2026\] Q-DiT4SR: Exploration of Detail-Preserving Diffusion Transformer Quantization for Real-World Image Super-Resolution](../../ICML2026/image_generation/q-dit4sr_exploration_of_detail-preserving_diffusion_transformer_quantization_for.md)

</div>

<!-- RELATED:END -->
