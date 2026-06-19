---
title: >-
  [论文解读] RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior
description: >-
  [ICML2025][图像生成][扩散模型] 提出 RestoreGrad 框架，通过 Prior Net 和 Posterior Net 联合学习条件 DDPM 的先验分布（而非固定标准高斯），利用退化信号与干净信号之间的相关性构建更具信息量的先验，在语音增强和图像修复任务上实现 5-10× 更快收敛和 2-2.5× 更少推理步数。
tags:
  - "ICML2025"
  - "图像生成"
  - "扩散模型"
  - "信号恢复"
  - "可学习先验"
  - "VAE-DDPM融合"
  - "语音增强"
  - "图像修复"
---

# RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior

**会议**: ICML2025  
**arXiv**: [2502.13574](https://arxiv.org/abs/2502.13574)  
**代码**: 待确认  
**领域**: 扩散模型 / 信号恢复  
**关键词**: 扩散模型, 信号恢复, 可学习先验, VAE-DDPM融合, 语音增强, 图像修复

## 一句话总结
提出 RestoreGrad 框架，通过 Prior Net 和 Posterior Net 联合学习条件 DDPM 的先验分布（而非固定标准高斯），利用退化信号与干净信号之间的相关性构建更具信息量的先验，在语音增强和图像修复任务上实现 5-10× 更快收敛和 2-2.5× 更少推理步数。

## 研究背景与动机

条件 DDPM 已在信号恢复（语音增强、图像修复）中展现出色能力，但存在两个关键瓶颈：

**收敛慢**：标准 DDPM 采用标准高斯 $\mathcal{N}(\mathbf{0}, \mathbf{I})$ 作为先验分布，与真实数据分布差距大，导致训练需要大量迭代才能收敛

**推理步数多**：先验与数据分布的差距也意味着反向过程需要更多步骤才能还原信号

**现有改进有限**：PriorGrad 提出用基于规则的方法从条件输入中提取先验信息，但这种手工设计的先验需要领域知识，且并非所有任务都能设计出合适的规则

**核心洞察**：在信号恢复任务中，退化信号 $\mathbf{y}$ 本身就是干净信号 $\mathbf{x}_0$ 的受损版本，二者之间存在强相关性。标准高斯先验完全丢弃了这种相关信息，造成了效率损失。

## 方法详解

### 核心思想：VAE + DDPM 融合

RestoreGrad 将 DDPM 无缝嵌入 VAE 框架中：

- **DDPM 作为解码器**：利用扩散模型强大的生成能力进行信号重建
- **VAE 编码器学习先验**：通过 Prior Net (ψ) 和 Posterior Net (ϕ) 联合学习更优的潜在空间

### 架构设计

框架包含三个可学习模块：

| 模块 | 参数化 | 输入 | 作用 |
|------|--------|------|------|
| 条件 DDPM (θ) | 噪声估计网络 | $\mathbf{x}_t, \mathbf{y}, t$ | 预测噪声，执行反向去噪 |
| Prior Net (ψ) | 先验编码器 | $\mathbf{y}$ | 推理时提供信息先验 |
| Posterior Net (ϕ) | 后验编码器 | $\mathbf{x}_0, \mathbf{y}$ | 训练时利用干净信号辅助学习 |

先验和后验分布均建模为零均值高斯：

$$p_\psi(\boldsymbol{\epsilon}|\mathbf{y}) = \mathcal{N}(\boldsymbol{\epsilon}; \mathbf{0}, \boldsymbol{\Sigma}_{\text{prior}}(\mathbf{y};\psi))$$

$$q_\phi(\boldsymbol{\epsilon}|\mathbf{x}_0, \mathbf{y}) = \mathcal{N}(\boldsymbol{\epsilon}; \mathbf{0}, \boldsymbol{\Sigma}_{\text{post}}(\mathbf{x}_0, \mathbf{y};\phi))$$

### 新的 ELBO 推导

通过 Proposition 3.1 将扩散过程整合到 VAE 框架中，得到条件数据对数似然的新下界 (Eq. 9)，包含两部分：
- 条件 DDPM 的 ELBO（重建项）
- Prior Net 与 Posterior Net 之间的 KL 散度（先验匹配项）

### 训练损失函数

最终的联合训练目标 (Eq. 11) 由三项组成：

$$\min_{\theta,\phi,\psi} \;\; \eta \cdot \mathcal{L}_{\text{LR}} + \mathcal{L}_{\text{DM}} + \lambda \cdot \mathcal{L}_{\text{PM}}$$

- **潜在正则化 (LR)**：$\mathcal{L}_{\text{LR}} = \bar{\alpha}_T \|\mathbf{x}_0\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{post}}} + \log|\boldsymbol{\Sigma}_{\text{post}}|$，防止后验协方差无限增大
- **去噪匹配 (DM)**：$\mathcal{L}_{\text{DM}} = \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, \mathbf{y}, t)\|^2_{\boldsymbol{\Sigma}^{-1}_{\text{post}}}$，训练 DDPM 预测真实噪声，注意使用后验协方差加权
- **先验匹配 (PM)**：$\mathcal{L}_{\text{PM}} = \log\frac{|\boldsymbol{\Sigma}_{\text{prior}}|}{|\boldsymbol{\Sigma}_{\text{post}}|} + \text{tr}(\boldsymbol{\Sigma}^{-1}_{\text{prior}}\boldsymbol{\Sigma}_{\text{post}})$，对齐先验和后验分布

其中 $\eta, \lambda > 0$ 为超参数，$\eta$ 控制正则化强度，$\lambda$ 控制先验匹配权重。

### 推理流程

推理时 Posterior Net 不再使用（因为没有干净信号 $\mathbf{x}_0$），DDPM 直接从 Prior Net 采样：$\boldsymbol{\epsilon} \sim p_\psi(\boldsymbol{\epsilon}|\mathbf{y}) = \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_{\text{prior}})$。

## 实验关键数据

### 语音增强 (VoiceBank+DEMAND)

| 方法 | 训练epoch | 推理步数 | PESQ↑ | CSIG↑ | CBAK↑ | COVL↑ | SI-SNR↑ |
|------|----------|---------|-------|-------|-------|-------|---------|
| CDiffuSE (基线) | 445 | 6 | 2.44 | 3.66 | 2.83 | 3.03 | - |
| + PriorGrad | 96 | 6 | 2.42 | 3.67 | 2.93 | 3.03 | 14.21 |
| + **RestoreGrad** | **96** | **6** | **2.51** | **3.80** | **3.00** | **3.14** | **14.74** |
| + **RestoreGrad** | **96** | **3** | **2.50** | **3.75** | **2.99** | **3.11** | **14.65** |

**要点**：RestoreGrad 仅用 96 epoch 即超越基线 445 epoch 的结果；推理步数减半性能几乎不降。

### 图像修复 (AllWeather → 多天气测试)

| 方法 | Snow100K-L PSNR/SSIM | Outdoor-Rain PSNR/SSIM | RainDrop PSNR/SSIM |
|------|---------------------|----------------------|-------------------|
| WeatherDiff (1775 epochs) | 30.09/0.904 | 29.64/0.931 | 30.71/0.931 |
| + RestoreGrad (887 epochs) | 30.82/0.916 | **30.83/0.941** | 31.78/0.939 |
| + RestoreGrad (1551 epochs) | **31.16/0.918** | 30.70/0.942 | **32.26/0.941** |

**要点**：训练 epoch 减半即全面超越基线；长训练进一步提升，与天气特定模型 DTPM 可比。

### 编码器开销分析

| 编码器大小 | PESQ↑ | SI-SNR↑ | 延迟占比 | 显存占比 |
|-----------|-------|---------|---------|---------|
| Tiny (24K) | 2.48 | 13.74 | 1.9% | 6.5% |
| Base (93K) | 2.51 | 14.74 | 2.2% | 10.3% |
| Large (370K) | 2.54 | 15.01 | 2.6% | 18.2% |

编码器开销极小（<3% 延迟，<19% 显存），性能随编码器增大稳步提升。

### Posterior Net 消融实验

| 配置 | PESQ↑ | COVL↑ | SI-SNR↑ |
|------|-------|-------|---------|
| CDiffuSE 基线 | 2.32 | 2.89 | 11.84 |
| + PriorGrad | 2.42 | 3.03 | 14.21 |
| + RestoreGrad (完整) | **2.51** | **3.14** | **14.74** |
| + RestoreGrad w/o Posterior (η=0) | — | 训练发散 | — |
| + RestoreGrad w/o Posterior (η=1) | 2.48 | 3.12 | 13.29 |

Posterior Net 对稳定训练和提升性能至关重要。

## 亮点与洞察

1. **VAE-DDPM 融合的理论优雅性**：通过新 ELBO 推导将 DDPM 无缝嵌入 VAE 框架，同时继承了 DDPM 的生成能力和 VAE 的建模效率
2. **极低额外开销**：编码器参数仅为 DDPM 的 0.3%-2%，延迟 <3%，却带来显著收敛加速
3. **跨模态通用性**：同一框架在语音（1D 波形）和图像（2D）上均有效，且 Prior Net 学到的协方差可视化显示它自动捕捉了信号结构
4. **训练与推理双重加速**：5-10× 训练加速 + 2-2.5× 推理步数减少，实用价值突出
5. **Posterior Net 的巧妙设计**：训练时利用 ground truth 信息指导先验学习，推理时丢弃 Posterior Net，无额外推理成本

## 局限与展望

1. **先验形式受限**：目前假设零均值高斯，仅学习协方差；对于均值非零或非高斯分布场景可能不够灵活
2. **仅验证在信号恢复任务**：未探索条件生成（如 text-to-image、text-to-speech）等更广泛的应用
3. **超参数敏感性**：虽然 $\eta$ 在较宽范围内鲁棒，但 $\eta$ 和 $\lambda$ 仍需调节，不同任务的最优值可能不同
4. **编码器架构简单**：使用固定 ResNet-20，未探索更强的编码器或自适应架构
5. **缺少与最新方法的对比**：未与 consistency models、flow matching 等新范式对比

## 相关工作与启发

- **PriorGrad** (Lee et al., 2022)：手工设计先验的先驱，RestoreGrad 可视为其学习型升级
- **CDiffuSE** (Lu et al., 2022)：语音增强 DDPM 基线
- **WeatherDiff** (Özdenizci & Legenstein, 2023)：多天气图像修复 DDPM 基线
- **DTPM** (Ye et al., 2024)：扩散纹理先验模型，需要大规模预训练

## 评分
- 新颖性: ⭐⭐⭐⭐ — VAE-DDPM 融合学习先验的思路新颖，理论推导完整
- 实验充分度: ⭐⭐⭐⭐ — 跨模态验证 + 丰富消融 + OOD 泛化测试
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰，图表丰富
- 价值: ⭐⭐⭐⭐ — 即插即用地加速现有条件 DDPM，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Hiding Images in Diffusion Models by Editing Learned Score Functions](../../CVPR2025/image_generation/hiding_images_in_diffusion_models_by_editing_learned_score_functions.md)
- [\[CVPR 2025\] Navigating Image Restoration with VAR's Distribution Alignment Prior](../../CVPR2025/image_generation/navigating_image_restoration_with_vars_distribution_alignment_prior.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)
- [\[NeurIPS 2025\] AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models](../../NeurIPS2025/image_generation/accuquant_simulating_multiple_denoising_steps_for_quantizing.md)
- [\[ACL 2025\] OZSpeech: One-step Zero-shot Speech Synthesis with Learned-Prior-Conditioned Flow Matching](../../ACL2025/image_generation/ozspeech_one-step_zero-shot_speech_synthesis_with_learned-prior-conditioned_flow.md)

</div>

<!-- RELATED:END -->
