---
title: >-
  [论文解读] MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting
description: >-
  [ECCV 2024][3D视觉][NeRF修复] 提出MALD-NeRF，通过掩码对抗训练和场景定制的潜在扩散模型实现高质量NeRF修复，有效解决扩散模型的多视角不一致和纹理偏移问题。 利用2D潜在扩散模型进行NeRF修复面临两个核心挑战：(1) 扩散模型的修复结果跨视角不一致，使用像素级损失（L1/L2）会导致修复区域模…
tags:
  - "ECCV 2024"
  - "3D视觉"
  - "NeRF修复"
  - "潜在扩散模型"
  - "对抗训练"
  - "3D一致性"
  - "物体移除"
---

# MALD-NeRF: Taming Latent Diffusion Model for Neural Radiance Field Inpainting

**会议**: ECCV 2024  
**arXiv**: [2404.09995](https://arxiv.org/abs/2404.09995)  
**代码**: [项目页面](https://hubert0527.github.io/MALD-NeRF)  
**领域**: 3D视觉  
**关键词**: NeRF修复, 潜在扩散模型, 对抗训练, 3D一致性, 物体移除

## 一句话总结

提出MALD-NeRF，通过掩码对抗训练和场景定制的潜在扩散模型实现高质量NeRF修复，有效解决扩散模型的多视角不一致和纹理偏移问题。

## 研究背景与动机

利用2D潜在扩散模型进行NeRF修复面临两个核心挑战：(1) 扩散模型的修复结果跨视角不一致，使用像素级损失（L1/L2）会导致修复区域模糊如雾；(2) 潜在扩散模型的自编码误差导致修复像素与原始像素之间存在纹理偏移，产生明显的修复边界接缝。虽然感知损失（LPIPS）能一定程度缓解，但并未根本解决问题。

## 方法详解

### 整体框架

MALD-NeRF包含三个核心组件：(1) 掩码对抗训练——替代像素级损失来监督修复区域；(2) 场景定制扩散——通过LoRA微调减少扩散模型的生成多样性；(3) 迭代数据集更新——逐步减少扩散噪声以传播3D一致信息。

### 关键设计

**掩码对抗训练**: 将修复图像的patch作为"真实"样本，NeRF渲染的patch作为"生成"样本进行对抗训练。关键创新在于**掩码设计**：对真实和生成图像都只保留修复掩码内的像素，掩码外用黑色填充，从而向判别器隐藏修复/非修复边界，消除纹理偏移的影响。同时引入判别器特征匹配损失提供更细粒度的监督。

**场景定制扩散**: 对每个场景进行LoRA微调，学习场景特定的文本token。使用自监督修复损失（随机矩形掩码）训练，物体移除掩码区域的损失设为零。微调后扩散模型生成结果跨视角一致性大幅提升。

**迭代数据集更新与噪声调度**: 每U次迭代更新一次修复图像，使用partial DDIM从当前NeRF渲染开始。噪声时间步随训练进度递减：$t = t_{max} - (t_{max} - t_{min}) \cdot \sqrt{k/K}$，实现从粗到细的3D一致信息传播。

### 损失函数

- **重建区域**: $L^r = \lambda_{pix}L_{pix} + \lambda_{inter}L_{inter} + \lambda_{distort}L_{distort} + \lambda_{decay}L_{decay}$
- **修复区域**: $L^m = -\lambda_{adv}L_{adv} + \lambda_{fm}L_{fm}$ + 正则化项
- **判别器**: $L^D = L_{adv} + \lambda_{GP}L_{GP}$（R1正则化）
- 修复区域完全不使用像素/感知损失

## 实验关键数据

### SPIn-NeRF数据集定量对比

| 方法 | LPIPS↓ | M-LPIPS↓ | FID↓ | KID↓ |
|------|--------|----------|------|------|
| SPIn-NeRF | 0.5356 | 0.4019 | 219.80 | 0.0616 |
| SPIn-NeRF (LDM) | 0.5568 | 0.4284 | 227.87 | 0.0558 |
| Inpaint3D | 0.5437 | 0.4374 | 271.66 | 0.0964 |
| InpaintNeRF360 | 0.4694 | 0.3672 | 222.12 | 0.0544 |
| **MALD-NeRF** | **0.4345** | **0.3344** | **183.25** | **0.0397** |

### 消融实验

| 设置 | LPIPS↓ | FID↓ | KID↓ |
|------|--------|------|------|
| 无对抗 + L1重建 | 0.6623 | 305.60 | 0.1177 |
| 无对抗 + LPIPS | 0.4231 | 192.86 | 0.0447 |
| Ours + L1重建 | 0.5106 | 256.82 | 0.0827 |
| Ours + LPIPS | **0.4130** | **185.79** | 0.0419 |
| Ours - 场景定制 | 0.4894 | 224.29 | 0.0596 |
| **MALD-NeRF (完整)** | 0.4345 | 183.25 | **0.0397** |

### 关键发现

- FID从219.80大幅降至183.25，说明对抗训练显著提升生成质量
- 像素级L1损失对修复任务甚至有害（FID 305.60），LPIPS也非最优，对抗训练才是正确选择
- 场景定制减少了上下文外（out-of-context）物体的生成，大幅提升3D一致性
- 掩码对抗训练是消除纹理偏移的关键设计

## 亮点与洞察

1. **深刻的问题分析**：明确指出像素级和感知损失在NeRF修复任务中的局限性
2. 掩码对抗训练思路巧妙——通过信息隐藏让判别器无法利用修复边界的纹理偏移
3. 场景定制+迭代更新+噪声调度的组合形成了完整的3D一致性增强方案

## 局限性

- 使用内部预训练的扩散模型，公开复现可能需替换为开源模型
- 对抗训练可能不稳定
- 大面积遮挡区域的几何重建仍有挑战

## 相关工作与启发

本文的掩码对抗训练设计与AmbientGAN概念上相似，但应用目标不同。对扩散模型在3D任务中"驯服"（taming）策略对后续工作有重要指导意义。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐
- 实用性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] S³D-NeRF: Single-Shot Speech-Driven Neural Radiance Field for High Fidelity Talking Head Synthesis](s3d-nerf_single-shot_speech-driven_neural_radiance_field_for_high_fidelity_talki.md)
- [\[ECCV 2024\] Dynamic Neural Radiance Field from Defocused Monocular Video](dynamic_neural_radiance_field_from_defocused_monocular_video.md)
- [\[ECCV 2024\] LN3Diff: Scalable Latent Neural Fields Diffusion for Speedy 3D Generation](ln3diff_scalable_latent_neural_fields_diffusion_for_speedy_3d_generation.md)
- [\[ECCV 2024\] Mesh2NeRF: Direct Mesh Supervision for Neural Radiance Field Representation and Generation](mesh2nerf_direct_mesh_supervision_for_neural_radiance_field_representation_and_g.md)
- [\[ECCV 2024\] Invertible Neural Warp for NeRF](invertible_neural_warp_for_nerf.md)

</div>

<!-- RELATED:END -->
