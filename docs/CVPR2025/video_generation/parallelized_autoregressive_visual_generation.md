---
title: >-
  [论文解读] Parallelized Autoregressive Visual Generation
description: >-
  [CVPR 2025][视频生成][自回归生成] 提出 PAR（Parallelized Autoregressive），通过分析视觉 token 依赖性，将空间距离远的弱依赖 token 并行生成而保持局部强依赖 token 的顺序生成，实现 3.6-9.5 倍加速且质量几乎无损。 自回归模型在视觉生成中展现了强大能力…
tags:
  - "CVPR 2025"
  - "视频生成"
  - "自回归生成"
  - "并行解码"
  - "视觉token依赖"
  - "图像生成"
  - "加速推理"
---

# Parallelized Autoregressive Visual Generation

**会议**: CVPR 2025  
**arXiv**: [2412.15119](https://arxiv.org/abs/2412.15119)  
**代码**: [项目主页](https://yuqingwang1029.github.io/PAR-project)  
**领域**: 视频生成  
**关键词**: 自回归生成, 并行解码, 视觉token依赖, 图像生成, 加速推理

## 一句话总结

提出 PAR（Parallelized Autoregressive），通过分析视觉 token 依赖性，将空间距离远的弱依赖 token 并行生成而保持局部强依赖 token 的顺序生成，实现 3.6-9.5 倍加速且质量几乎无损。

## 研究背景与动机

自回归模型在视觉生成中展现了强大能力，但逐 token 顺序预测的推理速度极慢——对于 $24 \times 24 = 576$ 个 token 的图像需要 576 步，严重限制实际应用。

现有加速方案各有局限：(1) 投机解码（speculative decoding）需要额外的草稿模型；(2) MaskGIT 采用非自回归范式改变了建模方式；(3) VAR 需要专用的多尺度 tokenizer 和更长的 token 序列。

核心问题：能否在保持标准自回归模型简洁性和灵活性的同时实现并行生成？

关键洞察：**并行生成的可行性取决于 token 依赖性**。局部相邻 token 具有强依赖——独立采样多个强依赖 token 会导致不一致（如扭曲的虎脸、断裂的斑马纹）；而空间距离远的 token 依赖弱，可安全并行生成。但各区域的**初始 token** 即使距离远也很关键，它们共同决定全局结构，必须顺序生成。

## 方法详解

### 整体框架

PAR 基于标准自回归 Transformer，通过 token 重排序机制实现并行。将图像 token 网格划分为 $M \times M$ 个区域，分两阶段：(1) 顺序生成各区域初始 token 建立全局结构；(2) 并行生成各区域对应位置的 token。使用 2D RoPE 保持空间位置信息，组内双向注意力丰富局部上下文。

### 关键设计一：非局部并行生成策略

**功能**：在保持质量的前提下大幅减少生成步数

**核心思路**：将 $H \times W$ token 网格划分为 $M \times M$ 个区域（如 $M=2$ 得到 4 个区域），然后按对应位置跨区域分组：

$$\{[v_1^{(1)}, \cdots, v_1^{(M^2)}], [v_2^{(1)}, \cdots, v_2^{(M^2)}], \cdots, [v_k^{(1)}, \cdots, v_k^{(M^2)}]\}$$

阶段 1：顺序生成各区域初始 token $v_1^{(i)} \sim \mathbb{P}(v_1^{(i)} | v_1^{(<i)})$

阶段 2：并行生成各区域第 $j$ 个位置的 token $\{v_j^{(r)}\}_{r=1}^{M^2} \sim \mathbb{P}(\{v_j^{(r)}\} | v_{<j})$

以 $M=2$ 为例，总步数从 576 降至 $4 + \frac{576-4}{4} = 147$。

**设计动机**：直接并行预测相邻 token 会因联合分布无法分解为独立分布而导致严重质量下降。空间距离远的 token 弱相关性使独立采样的影响最小化。先顺序生成初始 token 避免全局结构冲突。

### 关键设计二：组内双向注意力 + 全局自回归

**功能**：在并行生成时丰富每个 token 的可见上下文

**核心思路**：朴素因果掩码下，并行生成组中的 token（如 $6b$）只能看到前一组中相同位置之前的 token（到 $5b$）。改为组内双向注意力 + 组间因果注意力：预测当前组 $[6a, 6b, 6c, 6d]$ 时，每个 token 可访问整个前一组 $[5a, 5b, 5c, 5d]$ 作为上下文。

**设计动机**：朴素因果掩码限制了上下文窗口，导致每个 token 获取的信息不足。组内双向注意力在不破坏全局自回归性质的前提下丰富了局部上下文，且兼容 KV-cache 优化。

### 关键设计三：可学习过渡 token

**功能**：帮助模型从顺序生成模式平滑过渡到并行生成模式

**核心思路**：在初始顺序 token $[1, 2, \cdots, n]$ 和并行组之间插入 $n-1$ 个可学习 token $[M_1, M_2, M_3]$。这些 token 维度与常规 token 相同，参与训练但不预测实际视觉内容，仅作为模式切换的"信号"。

**设计动机**：模型需要适应从单 token 预测到多 token 预测的切换。可学习过渡 token 提供了柔和的转换机制，使模型在切换时不会突然改变行为模式。

### 损失函数

标准自回归交叉熵损失，在重排序后的 token 序列上计算。训练时同时监督所有阶段的 token 预测。

## 实验关键数据

### 主实验：ImageNet 256×256 类条件图像生成

| 方法 | 参数 | FID↓ | IS↑ | 步数 | 时间(s)↓ |
|------|------|------|------|------|----------|
| LlamaGen-XXL | 1.4B | 2.34 | 253.9 | 576 | 12.41 |
| MaskGIT | 227M | 6.18 | 182.1 | 8 | 0.13 |
| VAR-d30 | 2B | 1.97 | 334.7 | 10 | 0.27 |
| **PAR-XXL (4×)** | **1.4B** | **2.29** | **271.4** | **147** | **3.46** |
| **PAR-XXL (16×)** | **1.4B** | **3.02** | **247.6** | **40** | **1.31** |

### 消融实验：并行策略对比

| 策略 | FID↓ | IS↑ |
|------|------|-----|
| 标准AR（基线） | 2.34 | 253.9 |
| **PAR-4× (非局部)** | **2.29** | **271.4** |
| 朴素并行（相邻token） | 10.05 | 150.2 |
| 无顺序初始化 | 5.87 | 195.3 |

### 关键发现

- PAR-4× 实现 **3.6 倍实际加速**，FID 从 2.34 降至 **2.29**（反而略有改善！）
- PAR-16× 实现 **9.5 倍加速**，FID 仅从 2.34 升至 3.02（质量降级极小）
- 直接并行相邻 token 产生灾难性质量下降（FID 10.05），验证了 token 依赖性分析的正确性
- 顺序初始化至关重要：不顺序生成初始 token 导致 FID 从 2.29 升至 5.87
- 方法兼容 VQGAN 和 MAGVIT-v2 两种 tokenizer，扩展到视频生成（UCF-101）仅需改用 3D 位置编码

## 亮点与洞察

1. **依赖性分析驱动的设计**：不是盲目并行化，而是基于 token 依赖强度的深入分析来决定哪些 token 可并行
2. **极简且通用**：无需修改架构或 tokenizer，仅通过 token 重排序和少量可学习 token 实现，可无缝集成到任何标准 AR 模型
3. **4× 加速反而提升质量**：非局部并行实际上增加了不同区域间的信息交流

## 局限与展望

- 时序维度的并行化效果不佳（视频中时序依赖太强），仅在空间维度并行
- 大并行度（16×）下的质量下降虽小但存在，极端加速场景仍有改善空间
- 初始 token 的顺序生成步数固定为 $M^2$，自适应选择策略值得探索
- 未来可结合投机解码进一步加速

## 相关工作与启发

- **LlamaGen**：标准 AR 视觉生成基线，PAR 在其上实现加速
- **VAR**：多尺度预测的加速方案，但需专用 tokenizer；PAR 更通用
- **MaskGIT**：非自回归并行生成，但改变了建模范式；PAR 保持自回归性质

## 评分

⭐⭐⭐⭐⭐ — 方法设计精妙且原理清晰，基于 token 依赖性的非局部并行策略兼具理论合理性和实践效果。3.6× 加速+质量不降（甚至略升）的结果极为出色。方法的通用性和简洁性使其具有广泛影响力。对 AR 视觉生成的效率问题给出了优雅解答。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Taming Teacher Forcing for Masked Autoregressive Video Generation](taming_teacher_forcing_for_masked_autoregressive_video_generation.md)
- [\[CVPR 2025\] Teller: Real-Time Streaming Audio-Driven Portrait Animation with Autoregressive Motion Generation](teller_real-time_streaming_audio-driven_portrait_animation_with_autoregressive_m.md)
- [\[CVPR 2025\] From Slow Bidirectional to Fast Autoregressive Video Diffusion Models](from_slow_bidirectional_to_fast_autoregressive_video_diffusion_models.md)
- [\[CVPR 2025\] Visual Prompting for One-Shot Controllable Video Editing Without Inversion](visual_prompting_for_one-shot_controllable_video_editing_without_inversion.md)
- [\[ICLR 2026\] Streaming Autoregressive Video Generation via Diagonal Distillation](../../ICLR2026/video_generation/streaming_autoregressive_video_generation_via_diagonal_distillation.md)

</div>

<!-- RELATED:END -->
