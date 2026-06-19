---
title: >-
  [论文解读] Hiding Images in Diffusion Models by Editing Learned Score Functions
description: >-
  [CVPR 2025][图像生成][神经信息隐藏] 提出在扩散模型的特定时间步编辑learned score function来隐藏图像的方法，结合梯度感知参数选择和LoRA实现参数高效微调，在提取精度（52.90 dB PSNR）、模型保真度（FID变化仅0.02）和隐藏效率（0.04 GPU小时）三个维度上全面超越现有方法数个量级。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "神经信息隐藏"
  - "扩散模型"
  - "参数高效微调"
  - "数据隐写"
  - "Score Function编辑"
---

# Hiding Images in Diffusion Models by Editing Learned Score Functions

**会议**: CVPR 2025  
**arXiv**: [2503.18459](https://arxiv.org/abs/2503.18459)  
**代码**: [https://github.com/haoychen3/DMIH/](https://github.com/haoychen3/DMIH/)  
**领域**: 扩散模型  
**关键词**: 神经信息隐藏, 扩散模型, 参数高效微调, 数据隐写, Score Function编辑

## 一句话总结

提出在扩散模型的特定时间步编辑learned score function来隐藏图像的方法，结合梯度感知参数选择和LoRA实现参数高效微调，在提取精度（52.90 dB PSNR）、模型保真度（FID变化仅0.02）和隐藏效率（0.04 GPU小时）三个维度上全面超越现有方法数个量级。

## 研究背景与动机

**领域现状**：神经隐写术（neural steganography）已从传统比特流操作演进到在神经网络参数中嵌入秘密数据。在生成模型中隐藏数据特别有吸引力——生成模型可以直接产出秘密数据，不需要单独的解码网络，天然解决了传输安全问题。已有工作在GAN（如SinGAN）中成功实现了图像隐藏。

**现有痛点**：现有扩散模型隐写方法（BadDiffusion、TrojDiff、WDM）存在三个关键瓶颈：(1) 提取精度低，对复杂自然图像的重建PSNR≤25dB；(2) 模型保真度差，FID退化超过100%，容易被检测；(3) 隐藏效率低，需要完整重训或微调整个扩散过程（≥10 GPU小时），因为它们在整个反向扩散链中嵌入触发模式。

**核心矛盾**：现有方法将秘密图像的嵌入与提取过程与多步去噪扩散过程纠缠在一起——在初始时间步注入触发模式，在整个反向过程中干预，这导致了精度、保真度和效率三者的全面劣化。

**本文目标** (1) 实现高精度的复杂自然图像提取；(2) 保持模型生成能力几乎不变（样本级和分布级）；(3) 大幅降低嵌入时间；(4) 支持多接收者场景。

**切入角度**：观察到可以只在反向扩散过程的单个时间步编辑score function，在特定 $(z_s, t_s)$ 输入处插入一个秘密key到图像的映射，而不干扰其余时间步的去噪链，从而最大限度保持原始模型行为。

**核心 idea**：在扩散模型的单一私密时间步编辑score function来嵌入图像，通过一步"去噪"即可提取秘密图像，同时用混合PEFT保持模型保真度。

## 方法详解

### 整体框架

隐密通信涉及三方：发送者在预训练扩散模型中嵌入秘密图像（用秘密key $\mathcal{K}_s = \{k_s, t_s\}$），公开分享隐写模型；接收者用私密共享的key通过一步推理提取秘密图像；检查者验证模型的正常生成功能是否异常。嵌入过程仅修改score function在特定输入处的行为，提取通过 $f_{\tilde{\theta}}(z_s, t_s) = \frac{1}{\sqrt{\bar{\alpha}_{t_s}}}(z_s - \sqrt{1 - \bar{\alpha}_{t_s}} \epsilon_{\tilde{\theta}}(z_s, t_s))$ 一步完成。

### 关键设计

1. **单时间步Score Function编辑**:

    - 功能：在不破坏原始反向扩散链的前提下嵌入秘密图像
    - 核心思路：秘密key $\mathcal{K}_s = \{k_s, t_s\}$ 包含两部分：$k_s$ 是生成确定性高斯噪声 $z_s$ 的随机种子，$t_s$ 是选定的嵌入时间步。只在 $(z_s, t_s)$ 这一个特定输入处修改score function的输出，使得该处的"去噪"结果恰好为秘密图像 $x_s$。正常使用模型（随机噪声、从 $T$ 步开始迭代去噪）时，这个特殊输入的概率几乎为零，因此不影响正常生成。
    - 设计动机：现有方法在初始步注入触发模式并在全程干预，相当于修改了整个马尔可夫链。本文的洞察是：只需编辑一个"点"就足够提取高质量图像，且对马尔可夫链的扰动最小化。

2. **双损失优化（提取精度+模型保真度）**:

    - 功能：同时保证秘密图像提取质量和模型生成质量
    - 核心思路：总损失 $\ell = \ell_a + \lambda \ell_f$。提取精度损失 $\ell_a = \|f_\theta(z_s, t_s) - x_s\|^2$ 确保在特定key处重建秘密图像。模型保真度损失 $\ell_f = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon_\theta(x_t, t) - \epsilon_{\bar{\theta}}(x_t, t)\|^2]$ 在所有时间步上约束编辑后的score function与原始模型的偏差。保真度损失跨时间步均匀采样计算，无需重训原始模型或访问原始训练数据。多图像隐藏时，精度损失取所有秘密图像的均值。
    - 设计动机：没有保真度损失，模型行为会偏移被检测；两个损失的解耦设计巧妙——$\ell_a$ 只约束一个点，$\ell_f$ 约束整个函数，两者天然不冲突。

3. **混合参数高效微调（Gradient-based Selection + LoRA）**:

    - 功能：大幅减少可训练参数以提升保真度和效率
    - 核心思路：三步走。(1) **敏感度计算**：对每个参数 $\theta_i$ 累积N次迭代的梯度平方和 $g_i = \sum_{j=1}^N (\frac{\partial \ell}{\partial \theta_i^{(j)}})^2$，识别对编辑目标最敏感的参数。(2) **敏感层选择**：将敏感度二值化（阈值 $\tau$），按每层中敏感参数的数量（而非累积敏感度）排名，选择top-$\eta$ 层。关键地避免了偏向参数量大但不重要的层。(3) **LoRA微调**：仅对选中的层应用低秩分解 $\Delta W = AB$，线性层直接用标准LoRA，卷积层先将4D滤波器reshape为2D矩阵再应用。引入rank stabilization（$O(1/\sqrt{r})$ 缩放）和学习率解耦。相比全量微调减少86.3%可训练参数。
    - 设计动机：全量微调改变太多参数导致模型行为偏移；纯LoRA不区分层的重要性，可能对不重要的层也应用适配。先选层再LoRA，既精准又高效。

### 损失函数 / 训练策略

默认在时间步 $t_s = 500$ 嵌入。敏感度累积N=50次迭代。32×32图像：敏感参数稀疏度0.01，选15层，LoRA rank 64，2000次PEFT迭代。256×256图像：稀疏度0.1，选45层，rank 128。

## 实验关键数据

### 主实验

**提取精度对比（PSNR dB）：**

| 方法 | 32×32 PSNR ↑ | 256×256 PSNR ↑ | 类型 |
|------|-------------|---------------|------|
| BadDiffusion | 22.08 | 17.68 | 扩散 |
| TrojDiff | 46.54 | 24.74 | 扩散 |
| WDM | 36.49 | 17.97 | 扩散 |
| Chen22 (GAN) | 47.72 | 36.44 | GAN |
| **Ours** | **52.90** | **39.33** | 扩散 |

**模型保真度+隐藏效率对比（32×32）：**

| 方法 | FID ↓ | 样本PSNR ↑ | GPU时间 (h) ↓ |
|------|-------|-----------|-------------|
| Original | 4.79 | N/A | N/A |
| BadDiffusion | 6.88 | 23.78 | 4.87 |
| TrojDiff | 4.64 | 28.72 | 12.72 |
| WDM | 5.09 | 22.50 | 2.35 |
| **Ours** | **4.77** | **31.06** | **0.04** |

### 消融实验

**敏感层数的影响（32×32）：**

| 敏感层数 | 提取PSNR ↑ | 模型保真PSNR ↑ |
|---------|-----------|--------------|
| 5 | 47.48 | 31.55 |
| 15 (默认) | 52.90 | 31.06 |
| 45 | 54.04 | 27.87 |

**PEFT vs 全量微调：**

| 策略 | 提取PSNR ↑ | 模型FID ↓ | GPU时间 ↓ |
|------|-----------|---------|---------|
| Full fine-tuning | ~53 | 较差 | ~0.08h |
| PEFT (ours) | 52.90 | **4.77** | **0.04h** |

### 关键发现

- 本文方法的FID仅为4.77 vs 原始4.79，几乎无变化；而BadDiffusion为6.88（+43.6%退化），证明单时间步编辑对模型行为的扰动极小
- 隐藏效率比最快的WDM快约59倍（0.04 vs 2.35 GPU小时），因为只需优化单个时间步而非整个扩散过程
- 选15层是精度和保真度的最佳平衡点：5层精度不够（47.48 dB），45层保真度下降（27.87 dB）
- 多图像隐藏（4张）时提取精度仅轻微下降（52.90→49.38 PSNR），说明不同key对应的score function编辑相互近似独立
- 对时间步选择具有广泛鲁棒性，在大范围 $t_s$ 下性能稳定

## 亮点与洞察

- **单点编辑的深刻洞察**：认识到只需在score function的一个特定输入点处修改行为就能嵌入任意图像，且该点在正常使用中几乎不会被触及。这种对连续函数空间局部编辑的思路，既优雅又实用。
- **混合PEFT的层选择策略**：先用梯度敏感度选层，再用LoRA微调——避免了LoRA应用于不相关层的浪费。按敏感参数"数量"而非"总值"排序避免了大层偏倚，这是一个可推广的PEFT策略。
- **天然的多接收者支持**：不同秘密key对应score function不同输入点的编辑，天然互不干扰，实现独立提取通道，无需复杂的密钥管理。

## 局限与展望

- 仅在像素空间DDPM上验证，未扩展到latent diffusion（如Stable Diffusion），适用范围受限
- 秘密key仅为随机种子+时间步（总共几个bit），key空间较小，理论上可被暴力搜索
- 256×256图像的提取精度（39.33 dB）虽然最优但相比32×32（52.90 dB）下降明显，高分辨率场景仍有改进空间
- 未分析面对模型剪枝、量化等后处理操作的鲁棒性

## 相关工作与启发

- **vs BadDiffusion/TrojDiff**: 它们在整个反向扩散链中注入触发模式，本文只编辑单个时间步，从根本上减少了对模型的干预量级
- **vs Chen22 (SinGAN)**: 在GAN中隐藏图像的前驱工作，但GAN训练不稳定且模型覆盖度有限；扩散模型提供更好的似然优化和训练稳定性
- **vs StableSignature/AquaLoRA**: 水印方法，嵌入不可见信号而非完整图像，且保真度较差；本文实现了完整图像的嵌入与提取

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 单时间步编辑score function的思路非常巧妙，从问题根源解决了效率和保真度的矛盾
- 实验充分度: ⭐⭐⭐⭐⭐ 九种基线对比+五项消融+多图像扩展+视觉对比，极其充分
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，方法描述清晰
- 价值: ⭐⭐⭐⭐ 在AI安全、版权保护和隐蔽通信领域有应用潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] TinyFusion: Diffusion Transformers Learned Shallow](tinyfusion_diffusion_transformers_learned_shallow.md)
- [\[ICML 2025\] RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior](../../ICML2025/image_generation/restoregrad_signal_restoration_using_conditional_denoising_diffusion_models_with.md)
- [\[CVPR 2025\] Traversing Distortion-Perception Tradeoff Using a Single Score-Based Generative Model](traversing_distortion-perception_tradeoff_using_a_single_score-based_generative_.md)
- [\[CVPR 2025\] Diff2Flow: Training Flow Matching Models via Diffusion Model Alignment](diff2flow_training_flow_matching_models_via_diffusion_model_alignment.md)
- [\[CVPR 2025\] Temporal Score Analysis for Understanding and Correcting Diffusion Artifacts](temporal_score_analysis_for_understanding_and_correcting_diffusion_artifacts.md)

</div>

<!-- RELATED:END -->
