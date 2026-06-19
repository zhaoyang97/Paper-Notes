---
title: >-
  [论文解读] UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion
description: >-
  [CVPR 2025][图像生成][高动态范围] UltraFusion 首次将曝光融合建模为引导式修复问题，利用欠曝图像作为高光区域的软引导而非硬约束，实现 9 档曝光差的超高动态范围成像，同时对对齐误差和光照变化保持鲁棒。 HDR 成像是相机设计的基本问题。主流方法通过融合不同曝光的图像来增加动态范围…
tags:
  - "CVPR 2025"
  - "图像生成"
  - "高动态范围"
  - "曝光融合"
  - "引导修复"
  - "扩散模型"
  - "超大曝光差"
---

# UltraFusion: Ultra High Dynamic Imaging using Exposure Fusion

**会议**: CVPR 2025  
**arXiv**: [2501.11515](https://arxiv.org/abs/2501.11515)  
**代码**: [项目页面](https://openimaginglab.github.io/UltraFusion)  
**领域**: Image Generation / HDR Imaging  
**关键词**: 高动态范围, 曝光融合, 引导修复, 扩散模型, 超大曝光差

## 一句话总结

UltraFusion 首次将曝光融合建模为引导式修复问题，利用欠曝图像作为高光区域的软引导而非硬约束，实现 9 档曝光差的超高动态范围成像，同时对对齐误差和光照变化保持鲁棒。

## 研究背景与动机

HDR 成像是相机设计的基本问题。主流方法通过融合不同曝光的图像来增加动态范围，但实际限制严重：
- 现有方法仅能处理 3-4 档曝光差（如 HDR+ 仅增加 3 档），远不足以应对超高动态场景
- **对齐问题**：大曝光差时输入亮度差异巨大，光流对齐困难，导致鬼影伪影
- **光照不一致**：欠曝图像不是正常曝光图像的简单暗化版本，物体外观会随曝光变化
- **色调映射伪影**：HDR 方法先生成 HDR 再压缩为 LDR 显示，高动态范围时色调映射引入额外伪影
- 直接使用 ControlNet 无法确定以哪帧为参考，导致不同区域选择不一致的参考帧
- 缺乏大规模动态场景曝光融合训练数据

## 方法详解

### 整体框架

UltraFusion 是两阶段框架：(1) 预对齐阶段——将欠曝图像对齐到过曝图像并掩盖遮挡区域；(2) 引导修复阶段——基于 Stable Diffusion，使用过曝图像为主图、欠曝图像为软引导，修复过曝区域的高光信息。引导修复阶段包含分解融合控制分支和保真控制分支。

### 关键设计1：引导式修复范式

**功能**：将曝光融合重新定义为修复问题，直接输出色调映射的 LDR 结果。

**核心思路**：以正常曝光（过曝）图像 $I_{oe}$ 为基准，修复其高光区域的缺失信息。欠曝图像 $I_{ue}$ 作为软引导而非硬约束提供高光区域的真实内容。预对齐阶段使用 RAFT 估计双向光流，通过前后一致性检查得到遮挡掩码 $\mathcal{M}$，对齐输出 $I_{ue \to oe} = (1-\mathcal{M}) \cdot \mathcal{W}(I_{ue}, f_{oe \to ue})$。

**设计动机**：软引导对对齐误差和光照变化鲁棒（vs 硬约束会放大误差）。直接输出 LDR 避免 HDR→LDR 的级联误差。扩散模型的图像先验确保输出自然可信。

### 关键设计2：分解融合控制分支

**功能**：从极暗的欠曝图像中提取对亮度变化鲁棒的结构和颜色信息，有效引导扩散修复。

**核心思路**：将欠曝图像分解为结构分量 $S_{ue} = (Y_{ue} - \mu(Y_{ue})) / \sigma(Y_{ue})$（YUV 亮度通道归一化）和颜色分量（UV 色度通道）。结构和颜色分别通过独立的卷积提取器提取多尺度特征，然后通过**多尺度交叉注意力**与过曝图像特征融合。控制分支复制 U-Net 编码器结构但独立更新权重，输出通过零卷积注入主 U-Net。

**设计动机**：极暗的欠曝图像直接作为引导会被模型忽略。分解为亮度无关的结构和颜色信息后，引导信号的有效性大幅提升。

### 关键设计3：保真控制分支 + 训练数据合成

**功能**：减轻 VAE 解码器引入的纹理失真；合成动态场景曝光融合训练数据。

**核心思路**：保真控制分支（FCB）结构类似分解融合控制分支，但主提取器采用 VAE 编码器结构（而非 U-Net），为 VAE 解码器提供跳跃连接。训练时以 GT 编码的潜码模拟去噪输出，使用 $\|I_{gt} - \hat{I}_{gt}\|_1$ 重建损失。数据合成：从视频数据集采样帧对模拟大运动，从静态多曝光数据集（SICE）采样欠曝图像块，用伪遮挡掩码模拟动态遮挡，使模型仅用静态数据学会处理动态场景。

**设计动机**：VAE 解码时会引入不良纹理修改；没有现成的大规模动态 HDR 训练数据。

### 损失函数

扩散模型标准去噪损失 + 保真分支的 L1 重建损失 $\|I_{gt} - \hat{I}_{gt}\|_1$。

## 实验关键数据

### 主实验：静态 MEFB 数据集

| 方法 | MUSIQ ↑ | DeQA-Score ↑ | PAQ2PIQ ↑ | HyperIQA ↑ | MEF-SSIM ↑ |
|------|---------|-------------|-----------|-----------|-----------|
| **UltraFusion** | **68.82** | **3.881** | **73.80** | **0.6482** | 0.9385 |
| HSDS-MEF | 66.76 | 3.544 | 72.60 | 0.6026 | **0.9520** |
| HDR-Transformer | 63.10 | 2.983 | 71.36 | 0.5996 | 0.8626 |

### 消融实验：动态 RealHDRV 与 UltraFusion Benchmark

| 方法 | RealHDRV TMQI ↑ | RealHDRV MUSIQ ↑ | Benchmark MUSIQ ↑ | Benchmark DeQA ↑ |
|------|----------------|-----------------|------------------|-----------------|
| **UltraFusion** | **0.8925** | **67.51** | **68.41** | **3.830+** |
| HSDS-MEF | 0.8323 | 61.76 | 64.54 | 3.627 |
| HDR-Transformer | 0.8680 | 62.24 | 63.66 | 2.909 |

### 关键发现

- UltraFusion 是首个能合并 9 档曝光差图像的方法
- 在用户研究中，UltraFusion 在质量和偏好上均大幅领先所有基线
- 软引导机制使方法对大运动和光照变化同时鲁棒
- 分解欠曝图为结构+颜色信息是有效引导暗图中提取信息的关键

## 亮点与洞察

- **范式创新**：将曝光融合从"对齐-合并"范式转变为"引导修复"范式
- **实用性强**：直接输出 LDR 图像避免色调映射步骤，端到端产生可显示的高质输出
- **数据合成策略巧妙**：用视频帧+静态多曝光数据组合模拟动态 HDR 场景

## 局限与展望

- 依赖扩散模型的采样过程，推理速度较慢
- 训练数据合成中的伪遮挡可能不完全反映真实动态场景的复杂性
- 仅处理两帧（一长一短曝光），未扩展到更多曝光括号
- 未来可探索实时推理和视频 HDR 场景

## 相关工作与启发

- 与 ControlNet 的直接应用相比，固定参考帧（过曝图）的策略消除了歧义问题
- 分解欠曝图像为结构+颜色的思路可借鉴到其他跨域引导任务
- 新捕获的 100 场景 UltraFusion Benchmark 为超高动态范围评估提供了标准数据

## 评分

⭐⭐⭐⭐ — 将曝光融合重新定义为引导修复的思路新颖且有效，首次实现 9 档曝光差融合。方法设计完整，分解融合控制分支和数据合成管线都很精巧。在多个基准上取得最优质量。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Diffusion-4K: Ultra-High-Resolution Image Synthesis with Latent Diffusion Models](diffusion-4k_ultra-high-resolution_image_synthesis_with_latent_diffusion_models.md)
- [\[CVPR 2025\] Latent Space Imaging](latent_space_imaging.md)
- [\[CVPR 2025\] LEDiff: Latent Exposure Diffusion for HDR Generation](lediff_latent_exposure_diffusion_for_hdr_generation.md)
- [\[CVPR 2025\] Pursuing Temporal-Consistent Video Virtual Try-On via Dynamic Pose Interaction](pursuing_temporal-consistent_video_virtual_try-on_via_dynamic_pose_interaction.md)
- [\[CVPR 2025\] GlyphMastero: A Glyph Encoder for High-Fidelity Scene Text Editing](glyphmastero_a_glyph_encoder_for_high-fidelity_scene_text_editing.md)

</div>

<!-- RELATED:END -->
