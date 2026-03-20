# DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression

**会议**: arXiv 2026  
**arXiv**: [2603.13162](https://arxiv.org/abs/2603.13162)  
**作者**: Junqi Shi, Ming Lu, Xingchen Li, Anle Ke, Ruiqi Zhang et al.
**代码**: 待确认  
**领域**: 模型压缩/高效推理 / LLM/NLP  
**关键词**: dit-ic, aligned, diffusion, transformer, efficient  

## 一句话总结
基于扩散的图像压缩最近表现出了出色的感知保真度，但其实用性却受到过高的采样开销和高内存使用率的阻碍。
## 背景与动机
Diffusion-based image compression has recently shown outstanding perceptual fidelity, yet its practicality is hindered by prohibitive sampling overhead and high memory usage.. Most existing diffusion codecs employ U-Net architectures, where hierarchical downsampling forces diffusion to operate in shallow latent spaces (typically with only 8x spatial downscaling), resulting in excessive computation.

## 核心问题
基于扩散的图像压缩最近表现出了出色的感知保真度，但其实用性却受到过高的采样开销和高内存使用率的阻碍。大多数现有的扩散编解码器都采用 U-Net 架构，其中分层下采样迫使扩散在浅层潜在空间中运行（通常仅进行 8 倍空间缩小），从而导致计算量过多。
## 方法详解

### 整体框架
- To address this, we introduce DiT-IC, an Aligned Diffusion Transformer for Image Compression, which replaces the U-Net with a Diffusion Transformer capable of performing diffusion in latent space entirely at 32x downscaled resolution.
- With these designs, DiT-IC achieves state-of-the-art perceptual quality while offering up to 30x faster decoding and drastically lower memory usage than existing diffusion-based codecs.

### 关键设计
1. **关键组件1**: DiT-IC adapts a pretrained text-to-image multi-step DiT into a single-step reconstruction model through three key alignment mechanisms: (1) a variance-guided reconstruction flow that adapts denoising strength to latent uncertainty for efficient reconstruction; (2) a self-distillation alignment that enforces consistency with encoder-defined latent geometry to enable one-step diffusion; and (3) a latent-conditioned guidance that replaces text prompts with semantically aligned latent conditions, enabling text-free inference.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 通过这些设计，DiT-IC 实现了最先进的感知质量，同时与现有基于扩散的编解码器相比，解码速度提高了 30 倍，内存使用量大幅降低。
## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
