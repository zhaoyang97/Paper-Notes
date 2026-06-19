---
title: >-
  [论文解读] Multi-subject Open-set Personalization in Video Generation
description: >-
  [CVPR 2025][视频生成][视频个性化生成] 提出 Video Alchemist，在 Diffusion Transformer 架构中内置多主体、开放集的视频个性化生成能力，支持前景物体和背景的定制，无需测试时优化。 视频个性化生成旨在合成包含特定人物、宠物、场景的视频，但现有方法存在显著局限： 1. 领域受限：…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "视频个性化生成"
  - "多主体定制"
  - "开放集实体"
  - "Transformer"
  - "数据增强"
---

# Multi-subject Open-set Personalization in Video Generation

**会议**: CVPR 2025  
**arXiv**: [2501.06187](https://arxiv.org/abs/2501.06187)  
**代码**: [https://github.com/snap-research/MSRVTT-Personalization](https://github.com/snap-research/MSRVTT-Personalization) (有，benchmark 代码)  
**领域**: 视频生成  
**关键词**: 视频个性化生成, 多主体定制, 开放集实体, Diffusion Transformer, 数据增强

## 一句话总结

提出 Video Alchemist，在 Diffusion Transformer 架构中内置多主体、开放集的视频个性化生成能力，支持前景物体和背景的定制，无需测试时优化。

## 研究背景与动机

视频个性化生成旨在合成包含特定人物、宠物、场景的视频，但现有方法存在显著局限：

1. **领域受限**：很多方法只支持人脸（Magic-Me）、或单一主体（DreamVideo、VideoBooth），无法处理多主体和开放类别
2. **测试时优化代价高**：DreamVideo 等需要对每个新概念进行微调，耗时且易过拟合
3. **前景/背景不可同时定制**：大多数方法仅关注前景物体，无法自定义视频背景
4. **Copy-and-paste 问题**：从同一视频中提取参考帧和目标帧进行训练时，模型倾向于直接复制参考图的光照、姿态、遮挡等无关信息，而非学习身份特征

**核心挑战**：如何构建训练数据，并设计模型架构使其能在不需要微调的情况下支持多主体、开放集、含背景的视频个性化？

## 方法详解

### 整体框架

Video Alchemist 基于 latent Diffusion Transformer (DiT) 构建，输入为文本提示 + 多张参考图像（每个实体一张或多张）。核心创新在于 DiT block 中增加了一个专用的 cross-attention 层处理个性化嵌入（personalization embeddings），实现图像-文本概念的绑定与融合。

### 关键设计

1. **图像-文字概念绑定（Binding of Image and Word Concepts）**：对每个参考实体，用冻结的 DINOv2 编码器提取图像 token $x_n \in \mathbb{R}^{l \times d}$，从文本嵌入中检索对应实体词 token $c_n$，将实体词 token 展平并复制 $l$ 次后与图像 token 沿通道轴拼接，经线性投影 + 残差连接得到个性化嵌入 $f_n$。所有实体的嵌入拼接为 $f = \text{Concat}(f_1, ..., f_N)$，通过独立 cross-attention 层与视频 token 交互。**设计动机**：没有绑定机制时，模型会将参考图像应用到错误的主体上（如把人脸贴到狗上）。

2. **自动化数据构建流水线**：三步流程——(a) LLM 从字幕中提取实体词（subject/object/background），(b) GroundingDINO + SAM 在视频的首/中/末帧分割目标，(c) 腐蚀膨胀后 inpainting 生成干净背景图。选取不同时间点的帧以捕获姿态和光照变化。

3. **抗过拟合数据增强**：针对 copy-and-paste 问题，对参考图像施加多种增强——降采样&高斯模糊（防分辨率过拟合）、颜色抖动&亮度调节（防光照过拟合）、水平翻转/剪切/旋转（防姿态过拟合）。引导模型聚焦于主体身份特征，而非参考图像的无关属性。

### 损失函数 / 训练策略

- 使用 Rectified Flow 公式进行去噪训练
- **两阶段训练**：第一阶段仅训练文本 cross-attention；第二阶段加入个性化 cross-attention 并全模型微调（with warmup）
- 图像编码器使用 DINOv2（冻结），比 CLIP 在主体相似度上更优
- 采用 RoPE 位置编码、Flash Attention、Fused LayerNorm 加速
- Self-conditioning 技术增强视觉质量

## 实验关键数据

### 主实验 — MSRVTT-Personalization（Subject 模式，单参考图）

| 方法 | 测试优化 | Text-S ↑ | Vid-S ↑ | Subj-S ↑ | Dync-D ↑ |
|------|---------|----------|---------|----------|----------|
| ELITE | 否 | 0.245 | 0.620 | 0.359 | - |
| VideoBooth | 否 | 0.222 | 0.612 | 0.395 | 0.448 |
| DreamVideo | 是 | 0.261 | 0.611 | 0.310 | 0.311 |
| **Video Alchemist** | **否** | **0.269** | **0.732** | **0.617** | **0.466** |

### 用户偏好研究

| 方法 | 质量偏好 ↑ | 保真偏好 ↑ |
|------|-----------|-----------|
| ELITE | 2.7% | 0.6% |
| VideoBooth | 0.3% | 0.8% |
| DreamVideo | 0.5% | 0.5% |
| **Video Alchemist** | **96.5%** | **98.1%** |

### 消融实验

| 配置 | Text-S ↑ | Vid-S ↑ | Subj-S ↑ | Dync-D ↑ | 说明 |
|------|----------|---------|----------|----------|------|
| CLIP 编码器 | 0.269 | 0.768 | 0.569 | 0.552 | 文本对齐好 |
| DINOv2 无 word token | 0.256 | 0.790 | 0.566 | 0.569 | 概念绑定缺失 |
| DINOv2 无增强 | 0.251 | 0.781 | 0.609 | 0.506 | copy-paste 严重 |
| **DINOv2 + word token + 增强** | **0.257** | **0.790** | **0.600** | **0.570** | 最佳平衡 |

### 关键发现

- Video Alchemist 在 Subject 相似度上比 VideoBooth 高出 **22.2%**（0.395 → 0.617）
- 即使是开放集模型，Face 相似度也超过面部专用模型 IP-Adapter（0.382 vs 0.269）
- 用户偏好研究中获得 **96.5%** 的质量偏好和 **98.1%** 的保真偏好
- 多参考图输入可进一步提升保真度（单/多 Subj-S: 0.617 → 0.626）
- 背景参考图使视频与 GT 更相似（Vid-S: 0.743 → 0.780），但略降文本对齐
- DINOv2 比 CLIP 更适合捕获独特的物体特征（自监督 vs 文本-图像对齐目标差异）

## 亮点与洞察

- **架构设计优雅**：将个性化能力内置于 DiT block 而非外部适配器，端到端训练
- **概念绑定机制关键**：缺失时多主体场景会出现身份混淆，这个发现对后续工作有参考价值
- **数据增强的精妙**：每种增强都对应一个特定的过拟合模式（分辨率→物体大小，遮挡→生成遮挡物等）
- **MSRVTT-Personalization 基准**：2130 个样本，支持面部/单主体/多主体/前景+背景等多种评测模式
- 开放集 + 无需微调的范式显著优于需要测试时优化的方法

## 局限与展望

- 当前分辨率为 $512 \times 288$，限制了生成质量
- 多参考图有时降低文本对齐度，灵活性和保真度之间存在权衡
- 背景定制仅使用 inpainting 生成的单帧，可能引入伪影
- 未处理极端场景如参考图严重遮挡或极低分辨率
- 训练需要大规模带字幕的视频数据集和多步处理流水线

## 相关工作与启发

- IP-Adapter：解耦 cross-attention 的先驱，但单 attention 层混合文本和图像 token 效果不佳
- DreamBooth / Textual Inversion：优化范式的代表，测试时开销大
- SnapVideo / Sora：DiT 架构的大规模视频生成基础
- 本文的概念绑定 + 数据增强策略可推广到 3D 生成、长视频合成等任务
- 相对标注式 benchmark 设计（segment-level 而非 image-level 相似度）值得借鉴

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个多主体+开放集+前景/背景+无微调的视频个性化模型
- **实验充分度**: ⭐⭐⭐⭐⭐ 提出全新 benchmark，定量/定性/用户研究/消融全面
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，数据构建流程图非常直观
- **价值**: ⭐⭐⭐⭐⭐ 对视频个性化生成领域具有重要推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Mind the Time: Temporally-Controlled Multi-Event Video Generation](mind_the_time_temporally-controlled_multi-event_video_generation.md)
- [\[CVPR 2025\] ShotAdapter: Text-to-Multi-Shot Video Generation with Diffusion Models](shotadapter_text-to-multi-shot_video_generation_with_diffusion_models.md)
- [\[AAAI 2026\] MoFu: Scale-Aware Modulation and Fourier Fusion for Multi-Subject Video Generation](../../AAAI2026/video_generation/mofu_scale-aware_modulation_and_fourier_fusion_for_multi-subject_video_generatio.md)
- [\[CVPR 2026\] ID-Crafter: VLM-Grounded Online RL for Compositional Multi-Subject Video Generation](../../CVPR2026/video_generation/id-crafter_vlm-grounded_online_rl_for_compositional_multi-subject_video_generati.md)
- [\[CVPR 2026\] StoryTailor: A Zero-Shot Pipeline for Action-Rich Multi-Subject Visual Narratives](../../CVPR2026/video_generation/storytailora_zero-shot_pipeline_for_action-rich_multi-subject_visual_narratives.md)

</div>

<!-- RELATED:END -->
