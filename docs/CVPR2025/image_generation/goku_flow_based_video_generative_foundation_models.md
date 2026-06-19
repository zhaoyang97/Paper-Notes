---
title: >-
  [论文解读] Goku: Flow Based Video Generative Foundation Models
description: >-
  [CVPR 2025][图像生成][rectified flow] Goku 是字节跳动与港大提出的 rectified flow Transformer 系列模型（2B/8B），首次将 rectified flow 用于图像-视频联合生成，配合全面的数据管线和大规模训练基础设施优化，在 VBench（84.85）和 GenEval（0.76）等基准上达到 SOTA。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "rectified flow"
  - "video generation"
  - "joint image-video"
  - "3D VAE"
  - "Transformer"
  - "data curation"
---

# Goku: Flow Based Video Generative Foundation Models

**会议**: CVPR 2025  
**arXiv**: [2502.04896](https://arxiv.org/abs/2502.04896)  
**代码**: [项目页面](https://saiyan-world.github.io/goku/)  
**领域**: 图像生成  
**关键词**: rectified flow, video generation, joint image-video, 3D VAE, Transformer, data curation

## 一句话总结

Goku 是字节跳动与港大提出的 rectified flow Transformer 系列模型（2B/8B），首次将 rectified flow 用于图像-视频联合生成，配合全面的数据管线和大规模训练基础设施优化，在 VBench（84.85）和 GenEval（0.76）等基准上达到 SOTA。

## 研究背景与动机

**领域现状**: 视频生成受益于先进的生成算法（GAN、扩散、Flow）、可扩展架构（Transformer）、海量互联网数据和持续增长的算力，已取得显著进展。但工业级联合图像-视频生成模型仍面临多方面挑战。

**现有痛点**:
- 早期方法将时间注意力与空间注意力分离处理（temporal+spatial），难以建模复杂的时间运动
- DDPM 收敛速度较慢，训练大规模模型成本高昂
- 高质量视频数据获取成本远高于图像数据，数据不平衡问题突出
- 长序列（超 220K token）的训练需要高效的并行和内存管理策略

**核心矛盾**: 联合图像-视频训练需要同时学习图像的空间语义和视频的时间运动动态，直接联合优化极具挑战。

**本文目标**: 构建完整的工业级图像-视频联合生成流水线，从数据、模型、训练公式到基础设施全链路优化。

**切入角度**: 采用 rectified flow 替代 DDPM，使用全注意力 Transformer 和 3D 联合 VAE，配合分阶段分辨率渐进训练策略。

**核心 idea**: 用 rectified flow + 全注意力 Transformer 统一图像-视频生成，通过精细的数据管线和多阶段训练实现工业级质量。

## 方法详解

### 整体框架

1. **3D 联合 VAE**: 将图像/视频从像素空间压缩到共享隐空间（视频压缩比 8×8×4，图像 8×8）
2. **Rectified Flow Transformer**: 在隐空间上建模线性插值流，联合训练图像和视频的 token
3. **多阶段训练**: 文本-语义配对 → 图像-视频联合学习 → 模态特定微调
4. **高效基础设施**: 序列并行 + FSDP + 选择性激活检查点 + MegaScale 容错

### 关键设计

#### 1. 全注意力 Transformer 架构

放弃传统的 temporal+spatial 分离注意力，直接对所有图像和视频 token 使用 **plain full attention**。关键增强：
- **Patch n' Pack**: 借鉴 NaViT，将不同分辨率/时长的样本沿序列维度打包到同一 batch，无需数据桶
- **3D RoPE**: 对图像/视频 token 应用三维旋转位置编码，支持分辨率外推，比正弦编码收敛更快
- **Q-K Normalization**: 对 query 和 key 在注意力计算前施加 RMSNorm，防止训练 loss 尖刺导致模型崩溃
- 模型规模：Goku-2B（28层，dim=1792，28头）和 Goku-8B（40层，dim=3072，48头）

#### 2. Rectified Flow 训练公式

定义前向过程为数据与噪声的线性插值 $\mathbf{x}_t = t \cdot \mathbf{x}_1 + (1-t) \cdot \mathbf{x}_0$，模型学习预测速度 $\mathbf{v}_t = d\mathbf{x}_t / dt$。相比 DDPM，RF 提供更直接的插值路径，更好的理论性质和更快的收敛速度。

#### 3. 多阶段渐进训练策略

- **Stage 1 (Text-Semantic Pairing)**: 纯文本到图像预训练，建立语义-视觉映射基础
- **Stage 2 (Joint Learning)**: 图像-视频联合训练，利用全注意力统一跨模态表示；高质量图像数据辅助提升视频帧质量；级联分辨率（288×512 → 480×864 → 720×1280）
- **Stage 3 (Modality-specific Fine-tuning)**: 分别针对 T2I 和 T2V 微调，提升各模态输出质量

### 损失函数

标准 rectified flow 速度预测损失：$\mathcal{L} = \mathbb{E}_{t,\mathbf{x}_0,\mathbf{x}_1}[\|\mathbf{v}_t - f_\theta(\mathbf{x}_t, t)\|^2]$

### 数据管线

- **规模**: 160M 图文对 + 36M 视频文本对
- **视频处理**: 预处理标准化 → PySceneDetect 粗切分 → DINOv2 帧间相似度细切分 → 美学评分/OCR/运动过滤
- **字幕生成**: InternVL2.0 关键帧 + Tarsier2 视频字幕 → Qwen2 合并润色
- **数据均衡**: 视频分类模型标注语义标签，上采样/下采样平衡 9 大类 86 子类

## 实验关键数据

### 主实验表

| 任务 | 基准 | Goku 得分 | 排名 |
|------|------|-----------|------|
| T2I | GenEval | **0.76** | SOTA |
| T2I | DPG-Bench | **83.65** | SOTA |
| T2V | VBench | **84.85** | 第1名 (2025-01-25) |
| T2V | UCF-101 Zero-shot | SOTA | - |

**T2I 对比**: 超越 SD3（GenEval 0.74）、DALL-E 3（GenEval 0.67）、Emu 3（0.66）

### 消融表（ImageNet 256×256 类条件生成）

Rectified Flow 收敛速度验证：

| 损失 | 步数 | FID ↓ | IS ↑ |
|------|------|-------|------|
| DDPM | 400k | 2.52 | 265.1 |
| DDPM | 1000k | 2.26 | 286.6 |
| **RF** | **400k** | **2.16** | **261.1** |

RF 仅需 400k 步即达到 DDPM 1000k 步的 FID 水平。

### 关键发现

1. Rectified flow 比 DDPM 收敛快约 2.5 倍
2. 全注意力优于时空分离注意力，能建模更复杂的时序运动
3. 3D RoPE 比正弦位置编码在跨阶段训练转换时收敛更快
4. 数据均衡显著影响人物类生成质量
5. 8B 模型的检查点保存仅阻塞训练约 4 秒

## 亮点与洞察

1. **工业级完整方案**: 覆盖数据、模型、训练、基础设施全栈，不仅是算法创新
2. **RF 首次用于联合图像-视频生成**: 验证了 rectified flow 在视频生成领域的可行性和优势
3. **Patch n' Pack 灵活打包**: 彻底解决变分辨率/变时长数据的 batch 问题
4. **数据驱动视角独特**: 详细公开了视频过滤阈值（美学 ≥4.3/4.5、运动 0.3-20.0 等），对工程实践极具参考价值
5. **字幕中附加运动分数**: 将 motion score 嵌入 caption 实现运动控制，简单有效

## 局限与展望

1. 论文未公开模型权重和训练代码，可复现性受限
2. 视频质量的主观评价依赖人工，缺乏统一的自动化视频质量指标
3. Image-to-Video 仅支持首帧条件，不支持更灵活的多帧参考
4. 数据管线依赖大量内部私有数据（60M 图像 + 25M 视频），社区难以复现
5. 运动可控性仅通过 caption 中的 motion score 实现，粒度有限

## 相关工作与启发

1. **Sora** (Brooks et al., 2024): 首先提出 3D VAE 压缩视频到隐空间的思路，Goku 延续并完善
2. **GenTron** (Chen et al., 2024): Goku Transformer block 的基础设计来源
3. **NaViT** (Dehghani et al., 2024): Patch n' Pack 灵活打包方案的来源
4. **InternVL2.0**: 用于生成高质量图像/视频字幕
5. **MegaScale** (Jiang et al., 2024): 大规模训练容错机制

**启发**: Rectified flow 在超大规模生成模型中的优势（快速收敛）可能推动更多工作从 DDPM 迁移到 flow-based formulation。数据管线中 motion score 嵌入 caption 的做法值得借鉴。

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性**: ⭐⭐⭐⭐ — RF 首次用于联合图像-视频生成，虽然各组件非全新，但组合方式具工程创新性
- **实验充分度**: ⭐⭐⭐⭐ — 多基准全面评测，但缺少与 CogVideoX 等开源模型的深度对比
- **论文写作**: ⭐⭐⭐⭐ — 技术报告风格，结构清晰，工程细节丰富
- **工程价值**: ⭐⭐⭐⭐⭐ — 对工业级视频生成的数据/训练/基础设施提供全面参考

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Can Generative Video Models Help Pose Estimation?](can_generative_video_models_help_pose_estimation.md)
- [\[CVPR 2025\] VLog: Video-Language Models by Generative Retrieval of Narration Vocabulary](vlog_video-language_models_by_generative_retrieval_of_narration_vocabulary.md)
- [\[CVPR 2025\] ObjectMover: Generative Object Movement with Video Prior](objectmover_generative_object_movement_with_video_prior.md)
- [\[CVPR 2025\] FADE: Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fade_fine_grained_erasure_diffusion.md)
- [\[CVPR 2025\] Fine-Grained Erasure in Text-to-Image Diffusion-based Foundation Models](fine-grained_erasure_in_text-to-image_diffusion-based_foundation_models.md)

</div>

<!-- RELATED:END -->
