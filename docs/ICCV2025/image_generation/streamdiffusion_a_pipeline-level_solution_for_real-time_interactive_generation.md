---
title: >-
  [论文解读] StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation
description: >-
  [ICCV 2025][图像生成][实时生成] StreamDiffusion 提出管线级实时扩散框架，通过 Stream Batch（去噪步骤批处理）、R-CFG（残差无分类器引导）和 SSF（随机相似性过滤）等策略，在单张 RTX 4090 上实现高达 91 fps 的实时图像生成，比 Diffusers AutoPipeline 快 59.6 倍。
tags:
  - "ICCV 2025"
  - "图像生成"
  - "实时生成"
  - "流式扩散"
  - "Stream Batch"
  - "残差CFG"
  - "随机相似性过滤"
  - "流水线优化"
---

# StreamDiffusion: A Pipeline-level Solution for Real-time Interactive Generation

**会议**: ICCV 2025  
**arXiv**: [2312.12491](https://arxiv.org/abs/2312.12491)  
**代码**: [GitHub](https://github.com/cumulo-autumn/StreamDiffusion)  
**领域**: 扩散模型·图像生成  
**关键词**: 实时生成, 流式扩散, Stream Batch, 残差CFG, 随机相似性过滤, 流水线优化  

## 一句话总结

StreamDiffusion 提出管线级实时扩散框架，通过 Stream Batch（去噪步骤批处理）、R-CFG（残差无分类器引导）和 SSF（随机相似性过滤）等策略，在单张 RTX 4090 上实现高达 91 fps 的实时图像生成，比 Diffusers AutoPipeline 快 59.6 倍。

## 研究背景与动机

扩散模型在图像/视频生成领域表现出色，但在增强/虚拟现实、直播推流、游戏渲染等**实时交互场景**中，其吞吐量远不能满足需求。

现有加速方法主要聚焦于**减少去噪步数**（如 LCM、一致性模型）或**量化**，但这些都是模型级别的优化。作者从**管线级别**出发，发现以下问题：

**串行去噪效率低**：传统方式等待一张图完全去噪后才处理下一张

**CFG 计算冗余**：每步需计算正向和负向条件的 UNet，浪费一半算力

**静态场景重复计算**：输入不变时仍持续运行 GPU，浪费能耗

## 方法详解

### 1. Stream Batch：去噪步骤批处理

核心思想：**不再等单张图完全去噪，而是每完成一步去噪就接收下一输入**，将分属不同图的不同去噪步组成 batch 并行处理。

对于 $n$ 步去噪，Stream Batch 将各帧各步交错排列组成 $n$ 大小的 batch，通过单次 UNet 前向同时完成所有帧的对应去噪步。在时间步 $t$ 编码的图在 $t+n$ 完成生成。

**时间一致性增强**：Stream Batch 天然支持使用**未来帧信息**，通过跨帧注意力提升时间一致性：

$$\text{Attn}(Q_{t,i}, K_{\text{Batch}}, V_{\text{Batch}}) = \text{Softmax}\left(\frac{Q_{t,i} \cdot K_{\text{Batch}}^T}{\sqrt{d}}\right) V_{\text{Batch}}$$

### 2. 残差无分类器引导（R-CFG）

标准 CFG 需要 $2n$ 次 UNet 计算（正向+负向）。R-CFG 利用原始输入图像的**虚拟残差噪声**替代负条件预测：

$$\epsilon_{\tau_i, \bar{c}'} = \frac{x_{\tau_i} - \sqrt{\alpha_{\tau_i}} x_0}{\sqrt{\beta_{\tau_i}}}$$

最终 R-CFG 公式为：

$$\epsilon_{\tau_i, \text{cfg}} = \delta \epsilon_{\tau_i, \bar{c}'} + \gamma(\epsilon_{\tau_i, c} - \delta \epsilon_{\tau_i, \bar{c}'})$$

两个变体：
- **Self-Negative R-CFG**：0 次负条件 UNet 计算（仅 $n$ 次总计算）
- **Onetime-Negative R-CFG**：1 次负条件计算（$n+1$ 次总计算）

### 3. 随机相似性过滤（SSF）

计算当前帧 $I_t$ 与参考帧 $I_{\text{ref}}$ 的余弦相似度，以概率方式决定是否跳过计算：

$$P(\text{skip} | I_t, I_{\text{ref}}) = \max\left\{0, \frac{S_C(I_t, I_{\text{ref}}) - \eta}{1 - \eta}\right\}$$

使用概率采样而非硬阈值，避免视频卡顿，实现更流畅的视觉效果。

## 实验

### 吞吐量对比

| 去噪步数 | StreamDiffusion (ms) | StreamDiffusion w/o TRT (ms) | AutoPipeline (ms) |
|---------|---------------------|----------------------------|--------------------|
| 1 | 10.65 (**59.6x**) | 21.34 (29.7x) | 634.40 |
| 2 | 16.74 (39.3x) | 30.61 (21.3x) | 652.66 |
| 4 | 26.93 (25.8x) | 48.15 (14.4x) | 695.20 |
| 10 | 62.00 (13.0x) | 96.94 (8.3x) | 803.23 |

单步去噪时实现约 91 fps，10 步时仍有 16 fps，显著优于基线。

### R-CFG 加速效果

| 去噪步数 | Self-Negative R-CFG | Onetime-Negative R-CFG | CFG |
|---------|--------------------|-----------------------|-----|
| 1 | 11.04 (1.52x) | 16.55 (1.01x) | 16.74 |
| 5 | 31.47 (**2.05x**) | 36.04 (1.79x) | 64.64 |

5 步去噪时，Self-Negative R-CFG 比标准 CFG 快 2.05 倍。

### 能耗评估

| GPU | 无 SSF 功耗 (W) | 有 SSF 功耗 (W) | 节省倍数 |
|-----|---------------|----------------|---------|
| RTX 3060 | 85.96 | 35.91 | 2.39x |
| RTX 4090 | 238.68 | 119.77 | 1.99x |

### 图像质量

StreamDiffusion 在 FID 上优于 LCM 基线（26.79 vs 29.69），同时保持相似 CLIP 分数（24.99 vs 24.95），证明加速未牺牲质量。

## 亮点与洞察

1. **管线级优化**正交于模型级优化，可与任何加速模型（LCM、TurboSD）兼容
2. **Stream Batch 的通用性**：可推广至视频、音频、机器人动作序列等连续生成任务
3. **R-CFG 以解析方式**替代昂贵的 UNet 负条件计算，几乎零开销
4. 概率采样式 SSF 比硬阈值产生更流畅的视频流

## 局限性

- R-CFG 主要针对 SDEdit 方式的 image-to-image，对纯 text-to-image 的适用性有限
- Stream Batch 引入的延迟与去噪步数线性相关（$n$ 帧延迟）
- SSF 的阈值 $\eta$ 需要根据应用场景手动调整
- 当前主要针对单 GPU 优化，多 GPU 并行收益有限

## 相关工作

- 高效扩散模型：DPM++、LCM、InstaFlow 等减少去噪步数的方法
- 模型加速：量化、TensorRT 等推理加速
- 并行采样：ParaDiGMS（关注延迟，与 StreamDiffusion 关注吞吐量正交）

## 评分

| 维度 | 分数 (1-5) |
|------|-----------|
| 创新性 | 4 |
| 技术深度 | 4 |
| 实验充分性 | 4 |
| 写作质量 | 4 |
| **综合** | **4.0** |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] SemanticDraw: Towards Real-Time Interactive Content Creation from Image Diffusion](../../CVPR2025/image_generation/semanticdraw_towards_real-time_interactive_content_creation_from_image_diffusion.md)
- [\[CVPR 2026\] StreamAvatar: Streaming Diffusion Models for Real-Time Interactive Human Avatars](../../CVPR2026/image_generation/streamavatar_streaming_diffusion_models_for_real-time_interactive_human_avatars.md)
- [\[ICCV 2025\] SDMatte: Grafting Diffusion Models for Interactive Matting](sdmatte_grafting_diffusion_models_for_interactive_matting.md)
- [\[NeurIPS 2025\] Real-Time Execution of Action Chunking Flow Policies](../../NeurIPS2025/image_generation/real-time_execution_of_action_chunking_flow_policies.md)
- [\[ICCV 2025\] GameFactory: Creating New Games with Generative Interactive Videos](gamefactory_creating_new_games_with_generative_interactive_videos.md)

</div>

<!-- RELATED:END -->
