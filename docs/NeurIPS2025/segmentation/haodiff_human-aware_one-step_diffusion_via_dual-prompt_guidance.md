---
title: >-
  [论文解读] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance
description: >-
  [NeurIPS 2025][语义分割][人体图像复原] 提出HAODiff，一种人体感知的单步扩散模型，通过三分支双提示引导（DPG）生成自适应正负提示对，结合显式人体运动模糊（HMB）退化管线和分类器自由引导（CFG），在人体图像复原任务上大幅超越现有SOTA方法。 人体图像退化复杂：真实世界人体图像同时遭受通用退化（噪…
tags:
  - "NeurIPS 2025"
  - "语义分割"
  - "人体图像复原"
  - "运动模糊"
  - "单步扩散"
  - "双提示引导"
  - "分类器自由引导"
---

# HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance

**会议**: NeurIPS 2025  
**arXiv**: [2505.19742](https://arxiv.org/abs/2505.19742)  
**代码**: [有](https://github.com/gobunu/HAODiff)  
**领域**: 分割  
**关键词**: 人体图像复原, 运动模糊, 单步扩散, 双提示引导, 分类器自由引导

## 一句话总结

提出HAODiff，一种人体感知的单步扩散模型，通过三分支双提示引导（DPG）生成自适应正负提示对，结合显式人体运动模糊（HMB）退化管线和分类器自由引导（CFG），在人体图像复原任务上大幅超越现有SOTA方法。

## 研究背景与动机

**人体图像退化复杂**：真实世界人体图像同时遭受通用退化（噪声、压缩、降采样）和人体运动模糊（HMB），现有方法往往只处理其中一种

**退化管线缺失HMB**：主流BIR模型使用Real-ESRGAN退化管线（下采样+压缩+噪声+低通模糊），但**缺少对人体局部运动模糊的模拟**——这是人体图像中最常见且最具挑战的退化类型之一

**负提示设计不足**：现有方法采用固定的负提示（如空文本或固定噪声描述），无法针对每张图像的特定退化模式提供自适应引导

**计算效率**：多步扩散模型（SUPIR需50步/26.67s）推理耗时长，单步扩散模型在保持质量的同时大幅降低计算开销

## 方法详解

### 整体框架

HAODiff采用两阶段训练架构：
- **Stage 1**：训练三分支双提示引导模块（DPG），从LQ图像中分别预测HQ图像（正提示来源）、残差噪声和HMB分割掩码（负提示来源）
- **Stage 2**：将DPG生成的正负提示嵌入向量注入单步扩散模型，通过CFG策略引导LQ→HQ的单步去噪

### 关键设计

**1. 含HMB的退化管线**

核心创新在于在退化过程中显式引入人体运动模糊：

- 使用Sapiens模型对HQ图像进行**人体部件分割**，获得头部、左右上肢、左右下肢、全身共6类掩码
- 随机选择一个部件类别，通过形态学操作（腐蚀→膨胀→高斯模糊）和归一化得到空间权重图：$W_s = (\text{Norm} \circ \text{Morph} \circ \text{Seg})(I_H)$
- 通过马尔可夫过程模拟随机运动轨迹，生成PSF并通过FFT卷积产生全局运动模糊图像 $I_B$
- 将原图与模糊图按空间权重混合：$I_{\text{HMB}} = W_s \odot I_H + (1-W_s) \odot I_B$
- **关键设计**：HMB放在第一阶退化（因运动模糊逻辑上发生在拍摄时），第二阶施加通用退化（噪声/压缩等）
- 第一阶三种可能状态：无退化 / HMB / 通用退化

**2. 三分支双提示引导（DPG）**

基于Swin Transformer构建，核心结构为共享特征提取器+三条独立重建分支：

- **共享backbone** $H_E$：下采样4倍后通过2个RSTB（各含6层STL，6头注意力）提取特征
- **三独立分支** $H_{R_i}$：各含2个RSTB（3层STL，3头注意力），分别预测：
    - 分支1：HQ图像 $\hat{I}_H^P$（正提示来源）
    - 分支2：残差噪声 $\hat{I}_R = I_L - I_H$（负提示来源之一）
    - 分支3：HMB分割掩码 $\hat{M}_{\text{HMB}}$（负提示来源之二，单通道+sigmoid）

关键洞察：**负提示不应是LQ图像本身**，而是残差噪声（仅含退化信息不含结构信息），否则复原图像会丧失保真度。同时HMB掩码提供局部运动模糊的精确定位信息。

**3. 单步扩散模型（OSD）**

- 基于SD2.1-base，使用LoRA（rank=16）微调UNet
- DPG三分支的最后一层特征通过**Prompt Embedder**（Performer Encoder + Attention Pooling）映射为SD兼容的嵌入向量
- 两个负分支特征拼接后共享一个Prompt Embedder生成 $p_{\text{neg}}$，正分支单独生成 $p_{\text{pos}}$
- CFG引导的噪声预测：$z_\varepsilon = z_{\text{neg}} + \lambda_{\text{cfg}} \cdot (z_{\text{pos}} - z_{\text{neg}})$，$\lambda_{\text{cfg}}=3.5$
- UNet通过batch维度拼接正负提示，**一次前向传播**同时获得 $z_{\text{pos}}$ 和 $z_{\text{neg}}$

### 损失函数 / 训练策略

**Stage 1（DPG训练）**：
$$\mathcal{L} = \mathcal{L}_1(\hat{I}_H^P, I_H) + \mathcal{L}_1(\hat{I}_R, I_L - I_H) + \alpha \cdot \mathcal{L}_{\text{Dice}}(\hat{M}_{\text{HMB}}, M_{\text{HMB}})$$
- $\alpha = 0.02$，Adam优化器，lr=2e-3，batch=16，4×A6000训练20K iter

**Stage 2（OSD训练）**：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}}(\hat{I}_H, I_H) + \mathcal{L}_{\text{EA}}(\hat{I}_H, I_H) + \beta \cdot \mathcal{L}_{\mathcal{G}}(\hat{z}_H)$$
- $\mathcal{L}_{\text{EA}}$：边缘感知DISTS感知损失（原图+Sobel边缘图双路DISTS）
- $\mathcal{L}_{\mathcal{G}}$：GAN生成器损失，使用预训练SDXL UNet下采样模块作判别器
- $\beta = 0.01$，AdamW优化器，lr=1e-5，batch=2，2×A6000训练120K iter

## 实验关键数据

### PERSONA-Val合成数据集结果

| 方法 | 步数 | 时间(s) | DISTS↓ | LPIPS↓ | FID↓ | CLIPIQA↑ | NIQE↓ |
|------|------|---------|--------|--------|------|----------|-------|
| SUPIR | 50 | 26.67 | 0.1415 | 0.2929 | 13.84 | 0.7908 | 3.777 |
| SeeSR | 50 | 5.05 | 0.1295 | 0.2555 | 12.82 | 0.7620 | 3.574 |
| OSDHuman | 1 | 0.11 | 0.1356 | 0.2384 | 14.41 | 0.7312 | 3.828 |
| **HAODiff** | **1** | **0.20** | **0.1023** | **0.2046** | **8.36** | **0.7737** | **2.830** |

HAODiff在单步推理下DISTS降低27.8%（vs OSDHuman）、FID降低42%，全面领先。

### MPII-Test真实运动模糊数据集

| 方法 | CLIPIQA↑ | MANIQA↑ | NIQE↓ | HMB-R↓ |
|------|----------|---------|-------|--------|
| SUPIR | 0.6702 | 0.6256 | 4.423 | 0.2776 |
| SeeSR | 0.6478 | 0.6636 | 4.615 | 0.2612 |
| OSDHuman | 0.6726 | 0.6535 | 3.912 | 0.2283 |
| **HAODiff** | **0.7203** | **0.7057** | **3.065** | **0.1167** |

HMB-R（运动模糊残留比）仅0.1167，是OSDHuman的一半，说明HAODiff对运动模糊的去除能力显著更强。

### 消融实验

- **DPG三分支 vs 单分支**：三分支双提示引导相比仅正提示引导在DISTS/LPIPS上有明显提升
- **自适应负提示 vs 固定负提示**：自适应残差噪声负提示优于固定空文本/固定噪声描述
- **含HMB退化管线 vs 不含**：引入HMB模拟后MPII-Test上HMB-R显著降低
- **CFG系数 $\lambda_{\text{cfg}}$**：3.5为最优值，过低引导不足，过高产生伪影

### 关键发现

1. 单步HAODiff在几乎所有指标上超越50步的SUPIR和SeeSR等多步扩散模型
2. 残差噪声作为负提示来源比LQ图像本身更合理——保留结构信息的同时精确描述退化特征
3. HMB分割掩码为局部运动模糊提供空间定位，使模型能针对性处理而非全图统一去噪
4. 推理时间仅0.20s（512×512），在实际应用中具有竞争力

## 亮点与洞察

1. **退化管线创新**：首次在BIR退化管线中显式加入人体运动模糊模拟，通过部件分割+空间权重图实现真实感HMB合成
2. **自适应双提示**：用残差噪声和HMB掩码构造自适应负提示，比固定负提示更有效地利用CFG引导
3. **效率与质量兼得**：单步扩散+LoRA微调，0.2s推理全面超越50步大模型
4. **新基准MPII-Test**：5,427张真实HMB图像+YOLO检测器量化评估，为人体复原提供标准化运动模糊评测

## 局限与展望

1. 退化管线中HMB仅模拟**刚体运动模糊**（基于PSF卷积），未建模关节柔性运动导致的非均匀模糊
2. 依赖Sapiens分割模型的质量——分割失败时HMB模拟可能不准确
3. SD2.1-base分辨率受限于512×512，对高分辨率人体图像需进一步扩展
4. MPII-Test的HMB检测器mAP@0.5仅0.62，HMB-R指标可靠性有待提高
5. 可探索将DPG扩展到视频人体复原场景

## 相关工作与启发

- DPG的三分支设计思路（正/负/定位）可推广到其他需要空间感知修复的任务（如去雾、去雨）
- "残差噪声作为负提示"的思路比"LQ图像作为负提示"更优，可在其他扩散复原模型中采用
- 人体部件级退化模拟为domain-specific退化管线设计提供了参考范式

## 评分

- 新颖性: ⭐⭐⭐⭐ 退化管线和双提示引导均有实质创新
- 实验充分度: ⭐⭐⭐⭐ 合成+真实数据、多指标、消融齐全
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，公式推导完整
- 价值: ⭐⭐⭐⭐ 人体复原的实用性强，新基准有贡献

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Prompt Guidance and Human Proximal Perception for HOT Prediction with Regional Joint Loss](../../ICCV2025/segmentation/prompt_guidance_and_human_proximal_perception_for_hot_prediction_with_regional_j.md)
- [\[NeurIPS 2025\] HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios](hopadiff_holistic-partial_aware_fourier_conditioned_diffusion_for_referring_huma.md)
- [\[NeurIPS 2025\] FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis](fast_foreground-aware_diffusion_with_accelerated_sampling_trajectory_for_segment.md)
- [\[NeurIPS 2025\] STEP: A Unified Spiking Transformer Evaluation Platform for Fair and Reproducible Benchmarking](step_a_unified_spiking_transformer_evaluation_platform_for_fair_and_reproducible.md)
- [\[NeurIPS 2025\] HumanCrafter: Synergizing Generalizable Human Reconstruction and Semantic 3D Segmentation](humancrafter_synergizing_generalizable_human_reconstruction_and_semantic_3d_segm.md)

</div>

<!-- RELATED:END -->
