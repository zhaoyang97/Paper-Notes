---
title: >-
  [论文解读] Zero-Reference Joint Low-Light Enhancement and Deblurring via Visual Autoregressive Modeling with VLM-Derived Modulation
description: >-
  [AAAI 2026][多模态VLM][低光增强] 提出 VAR-LIDE，一个完全无监督的视觉自回归框架，通过 VLM 感知先验引导自适应光照调制、空间-频率 RoPE 和递归相位域调制三大模块，联合解决低光增强与去模糊问题，在无需配对数据的条件下逼近甚至超越监督方法的感知质量。 问题定义 真实暗光环境下拍摄的图像面临三重…
tags:
  - "AAAI 2026"
  - "多模态VLM"
  - "低光增强"
  - "去模糊"
  - "视觉自回归模型"
  - "VLM感知先验"
  - "无监督"
  - "频域相位调制"
---

# Zero-Reference Joint Low-Light Enhancement and Deblurring via Visual Autoregressive Modeling with VLM-Derived Modulation

**会议**: AAAI 2026  
**arXiv**: [2511.18591](https://arxiv.org/abs/2511.18591)  
**作者**: Wei Dong, Han Zhou, Junwei Lin, Jun Chen
**代码**: [LowLevelAI/VAR-LIDE](https://github.com/LowLevelAI/VAR-LIDE)  
**领域**: 多模态VLM  
**关键词**: 低光增强, 去模糊, 视觉自回归模型, VLM感知先验, 无监督, 频域相位调制

## 一句话总结

提出 VAR-LIDE，一个完全无监督的视觉自回归框架，通过 VLM 感知先验引导自适应光照调制、空间-频率 RoPE 和递归相位域调制三大模块，联合解决低光增强与去模糊问题，在无需配对数据的条件下逼近甚至超越监督方法的感知质量。

## 研究背景与动机

### 问题定义

真实暗光环境下拍摄的图像面临**三重退化叠加**：低可见度/低对比度、复杂传感器噪声、以及长曝光引入的运动模糊。形式化地，退化过程可描述为：

$$\mathbf{x}_{LQ} = \gamma \, f(\mathbf{x}_{HQ}, \mathbf{k}) + \mathbf{n}$$

其中 $\gamma$ 建模动态范围压缩和曝光不足，$f$ 为与模糊核 $\mathbf{k}$ 的卷积操作，$\mathbf{n}$ 为传感器噪声。长曝光虽然提高了光子捕获量，但不可避免地加剧了运动模糊和噪声水平，导致图像质量严重下降。

### 现有方法的三大瓶颈

**分离处理的级联失败**：传统做法将低光增强（LLIE）和去模糊作为独立任务分别处理。LLIE 方法（Zero-DCE、SCI 等）只关注亮度提升但不处理模糊；去模糊方法（Blur2Blur、MambaIR 等）则假设输入具有充足光照。级联管道中，增强阶段可能破坏模糊线索，去模糊阶段又因可见度不足而无法有效恢复运动细节，两者相互制约。

**监督联合方法的配对数据瓶颈**：LEDNet、DarkIR 等联合方法虽取得不错效果，但严重依赖配对训练数据——在真实场景中这种数据几乎不可获取。这限制了它们的泛化能力。

**扩散模型的效率难题**：无监督扩散方法 FourierDiff 虽能避免配对数据需求，但需要约 1000 步采样迭代，推理效率极低，难以用于实际部署。

### 两个关键预备实验

本文的核心动机来自两个精心设计的预备实验：

**实验 1——VAR 的降噪/去模糊潜力**：将含噪声和模糊的图像直接送入预训练 VARSR 模型，发现其天然具有一定的噪声抑制和模糊减轻能力。然而，VAR 无法显著提升图像可见度，也难以恢复精细结构细节。这一观察说明 VAR 骨架是有潜力的，但需要额外的任务特定模块来弥补其在光照增强和精细结构恢复方面的不足。

**实验 2——固定迭代 Zero-DCE 的适应性失败**：Zero-DCE 通过迭代曲线变换调整光照，但固定迭代次数（如默认 $N=8$）在不同光照条件下表现不一致：极暗图像 $n=4$ 或 $n=6$ 次迭代远远不够导致欠增强；而中等亮度图像 $n=8$ 次迭代则产生严重过曝。这揭示了一个根本性缺陷——**单一固定的增强强度无法适应多样化的真实光照条件**，需要一种自适应机制根据图像实际可见度动态决定增强程度。

这两个观察共同驱动了 VAR-LIDE 的三大创新模块设计。

## 方法详解

### 整体框架

VAR-LIDE 以预训练的 VAR 模型（VARSR）为骨架，叠加三个互补的即插即用模块：

1. **VLM-Informed Conditioning Module (VICM)**：解决光照自适应问题
2. **Spatial-Frequency RoPE (SF-RoPE)**：增强 VAR 对模糊退化结构的建模能力
3. **VLM-Guided Phase Modulation (VGPM)**：在频域消除模糊引起的相位重复伪影

完整推理流程：低质图像输入 → VLM 感知先验管线提取可见度评分 $v$ 和模糊评分 $b$ → VICM 根据 $v$ 自适应增强光照作为 VAR 的条件输入 → 含 SF-RoPE 的 VAR 骨架进行多尺度自回归生成 → VGPM 根据 $b$ 在 FFT 相位域递归调制消除残余模糊 → 输出高质量图像。

### 关键设计 1：VLM-Informed Conditioning Module (VICM)

VICM 的核心思想是**用 VLM 的全局感知评分动态控制光照增强的迭代次数**，取代固定参数设定。

Zero-DCE 将增强建模为迭代曲线变换：

$$\bm{E}_N(\mathbf{x}) = \mathbf{x} + \sum_{n=1}^{N} \bm{\mathcal{A}}_n(\mathbf{x}) \cdot \bm{E}_{n-1}(\mathbf{x}) \cdot (1 - \bm{E}_{n-1}(\mathbf{x}))$$

VICM 的改进在于引入**自适应截断**机制：首先，通过 GPP-LLIE 管线从 VLM 获取可见度评分 $v$，再通过轻量 MLP $\boldsymbol{\Theta}_v$ 将其映射为最优迭代次数 $n_v$。曲线估计器 $\bm{\Psi}_I$ 仍然产生 $N$ 组曲线参数 $\{\bm{\mathcal{A}}_n(\mathbf{x})\}_{n=1}^N$，但超过 $n_v$ 的参数被直接置零：

$$\bm{\mathcal{A}}_j(\mathbf{x}) = 0, \quad \forall j > n_v$$

这种自适应截断确保极暗图像得到充分增强（$n_v$ 较大），而中等亮度图像避免过曝（$n_v$ 较小），使增强结果始终处于感知有效范围内。增强后的图像被嵌入并分词为 VAR 模型的条件输入。相比 Zero-DCE 的固定管线，VICM 提供的条件信号在空间上更具适应性，显著改善了下游生成质量。

### 关键设计 2：Content-Aware Spatial-Frequency RoPE (SF-RoPE)

SF-RoPE 旨在增强 VAR 骨架中注意力机制对模糊退化区域的敏感性。VARSR 使用的标准 RoPE 仅依赖固定的位置索引旋转矩阵，缺乏对内容退化的感知能力。SF-RoPE 将**频域相位信息**融入位置编码，实现内容自适应的注意力调制。

**频域 RoPE**：在尺度 $K$ 处，从嵌入 $\mathbf{x}_{K-1}$ 通过 FFT 提取频域相位信息 $\Phi(u,v) = \arg(\text{FFT}(\mathbf{x}_{K-1}))$，构建相位驱动的旋转矩阵 $\mathbf{R}_{\Phi(u,v)}$，使得每个 token 的位置编码能直接反映局部模糊敏感的频率特征。

**空间 RoPE**：并行地，使用尺度归一化的空间坐标计算标准 RoPE，保证多分辨率下的位置一致性。

**自适应融合**：通过可学习混合系数 $\lambda$ 将两种编码融合：

$$\text{RoPE}_{\text{fused}} = \lambda \cdot \text{RoPE}_{\text{freq}} + (1-\lambda) \cdot \text{RoPE}_{\text{spa}}$$

这种融合使注意力模块在保持全局位置对齐的同时，能够对模糊退化区域施加更精细的关注。实验表明 SF-RoPE 带来了更清晰的边缘恢复和更好的空间连贯性。

### 关键设计 3：VLM-Guided Phase Modulation (VGPM)

VGPM 针对模糊图像在 FFT 相位域中出现的**重复边缘伪影**问题。相位信息比空间域特征对遮挡歧义和光照噪声更鲁棒，因此选择在相位域进行模糊抑制。

首先将相位归一化为 $\hat{\phi} = (\phi + \pi) / 2\pi \in [0,1]$，然后执行递归增强：

$$\bm{M}_T(\hat{\phi}) = \hat{\phi} + \sum_{t=1}^{T} \bm{\mathcal{F}}_t(\hat{\phi}) \cdot M_{t-1}(\hat{\phi}) \cdot (1 - \bm{M}_{t-1}(\hat{\phi}))$$

其中 $T=8$ 为调制步数，$\bm{\mathcal{F}}_t \in [0,1]$ 由相位估计器 $\bm{\Psi}_p$ 预测。VLM 的模糊评分 $b$ 通过 MLP $\boldsymbol{\Theta}_b$ 自适应引导调制强度。最终将精炼后的相位逆变换回空间域得到输出图像。这一模块有效消除了模糊引起的鬼影伪影。

### 损失函数

完全无监督训练，联合优化所有参数：

$$\mathcal{L} = \mathcal{L}_{ex} + \lambda_{en}\mathcal{L}_{en} + \lambda_{con}\mathcal{L}_{con} + \lambda_{tv}\mathcal{L}_{tv}$$

- **自适应曝光控制损失** $\mathcal{L}_{ex}$：以 $E=0.45$ 为基准，动态调整 $E_d \in [-0.1, 0.1]$，约束输出均值亮度
- **结构熵损失** $\mathcal{L}_{en}$：对精炼相位的逆 FFT 幅度图计算 Shannon 熵，促进结构保真度
- **结构对比损失** $\mathcal{L}_{con}$：计算 16 个 patch 方差的负均值，增强局部结构区分度
- **全变分损失** $\mathcal{L}_{tv}$：抑制空间噪声和伪影

## 实验关键数据

### LOLBlur 数据集定量比较

| 方法 | 类型 | PSNR↑ | NIQE↓ | LPIPS↓ | FID↓ | CLIPIQA↑ |
|------|------|-------|-------|--------|------|----------|
| EnlightenGAN + Blur2Blur | L+D | 18.16 | 5.02 | 0.396 | 45.73 | 0.206 |
| SSFlow* | w/o R | 19.24 | 5.93 | 0.307 | 42.05 | 0.183 |
| FourierDiff | w/o R | 20.22 | 4.97 | 0.441 | 50.59 | 0.161 |
| LEDNet | w R | 24.36 | 5.37 | 0.227 | 25.19 | 0.207 |
| DarkIR-L | w R | 26.14 | 5.15 | 0.146 | 14.27 | 0.291 |
| LIEDNet-L | w R | 26.42 | 5.17 | 0.127 | 11.38 | 0.305 |
| **Ours** | **w/o R** | **23.39** | **4.80** | **0.191** | **26.04** | **0.262** |

**核心发现**：在无监督方法中，VAR-LIDE 大幅领先 SSFlow（PSNR +4.15dB）和 FourierDiff（PSNR +3.17dB），且在 NIQE 上取得所有方法最优（4.80）。尽管未使用 GT 监督，PSNR 已接近有监督的 LEDNet（24.36 vs 23.39），感知质量指标甚至超过多个有监督方法。

### Real-LOLBlur 数据集泛化评估

| 方法 | 类型 | NIQE↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ |
|------|------|-------|----------|--------|---------|
| SCI + Blur2Blur | L+D | 5.13 | 0.185 | 33.87 | 0.129 |
| SSFlow | w/o R | 5.94 | 0.190 | 30.93 | 0.148 |
| FourierDiff | w/o R | 5.59 | 0.187 | 32.01 | 0.122 |
| JUDE | w R | 4.92 | 0.236 | 50.29 | 0.223 |
| DarkIR-L | w R | 4.90 | 0.262 | 48.72 | 0.216 |
| **Ours** | **w/o R** | **5.16** | **0.226** | **47.53** | **0.223** |

**核心发现**：在真实无配对数据集上，VAR-LIDE 的泛化能力突出。MUSIQ 达 47.53（vs SSFlow 30.93、FourierDiff 32.01），逼近甚至匹配有监督方法 DarkIR-L（48.72）和 JUDE（50.29），说明 VLM 感知先验赋予了极强的跨数据集泛化能力。

## 亮点与洞察

1. **VLM 作为自适应调制器的范式**：不同于将 VLM 用于语义理解或图像描述，本文将 VLM 评分降维为标量可见度/模糊度分数，作为底层图像处理模块的控制信号。这种"VLM → 标量评分 → 模块参数"的链路非常轻量但有效，值得在其他图像复原任务中推广。

2. **Zero-DCE 迭代截断的优雅解法**：作者没有设计新的增强网络，而是保留了 Zero-DCE 的曲线估计器，仅通过动态截断迭代次数实现自适应性，这种最小化修改策略兼顾了方法的简洁性和有效性。

3. **频域相位作为去模糊的切入点**：相比在空间域直接恢复清晰结构，在 FFT 相位域操作能更直接地处理模糊引起的重复边缘现象。相位表示对光照变化的鲁棒性也使其特别适合低光场景。

4. **无监督逼近有监督**：在 Real-LOLBlur 上，本方法的 MANIQA 分数（0.223）与有监督的 JUDE（0.223）持平，MUSIQ 差距仅约 5%，充分验证了利用 VLM 先验替代配对 GT 的可行性。

## 局限性

1. **VLM 依赖的额外开销**：虽然 VLM 评分提取在推理时只需一次，但 GPP-LLIE 管线内部需要调用 VLM（如 CLIP 系列），这引入了额外的模型加载和计算开销，对资源受限的部署场景不够友好。

2. **PSNR 仍有差距**：与最优有监督方法（LIEDNet-L 26.42dB）相比，本方法 23.39dB 仍有约 3dB 差距。在需要精确像素级重建的应用中表现可能不够。

3. **未见真正极端场景测试**：实验仅在 LOLBlur/Real-LOLBlur 上验证，对于如夜间高速运动、超长曝光等更极端的退化组合的鲁棒性尚不清楚。

4. **缺少消融实验的部分细节**：论文中应有完整的消融实验数据来量化每个模块的独立贡献，cache 中缺失此部分。

## 相关工作与启发

- **GPP-LLIE**：提供 VLM 感知先验提取管线，本文直接复用其评分框架
- **VARSR**：VAR 用于图像超分辨的先驱工作，提供了 SA-RoPE 和预训练骨架
- **Zero-DCE / Zero-DCE++**：轻量级无参考 LLIE 基线，本文的 VICM 是其自适应扩展
- **FourierDiff**：无监督扩散方法，本文的主要对比对象，VAR 方案在效率上有显著优势（无需 1000 步采样）
- **启发方向**：VLM 评分引导的自适应参数控制可迁移至去雨、去雾等其他退化场景；SF-RoPE 的频率感知位置编码思路可推广到视频帧间对齐

## 评分

⭐⭐⭐⭐ (4/5)

**理由**：方法设计动机清晰、模块组合合理，VLM 感知先验的引入方式优雅且实用。在无监督设定下取得接近有监督方法的性能具有显著实践价值。扣一分因为 PSNR 差距仍然存在，且 VLM 依赖增加了系统复杂度。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[AAAI 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semantic-sensitive_underwater_image_enhancement_with_vlm.md)
- [\[ICML 2026\] Beyond VLM-Based Rewards: Diffusion-Native Latent Reward Modeling](../../ICML2026/multimodal_vlm/beyond_vlm-based_rewards_diffusion-native_latent_reward_modeling.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)
- [\[NeurIPS 2025\] Unifying Vision-Language Latents for Zero-Label Image Caption Enhancement](../../NeurIPS2025/multimodal_vlm/unifying_vision-language_latents_for_zero-label_image_caption_enhancement.md)

</div>

<!-- RELATED:END -->
