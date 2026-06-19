---
title: >-
  [论文解读] From Slow Bidirectional to Fast Autoregressive Video Diffusion Models
description: >-
  [CVPR 2025][视频生成][自回归扩散] CausVid 通过非对称蒸馏将预训练的双向视频扩散 Transformer 蒸馏为因果自回归 4 步生成器，结合 ODE 初始化和 KV 缓存，实现 9.4 FPS 的流式视频生成（比 CogVideoX 快 160×），在 VBench-Long 基准上以 84.27 分排名第一。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "自回归扩散"
  - "蒸馏"
  - "KV缓存"
  - "实时视频"
---

# From Slow Bidirectional to Fast Autoregressive Video Diffusion Models

**会议**: CVPR 2025  
**arXiv**: [2412.07772](https://arxiv.org/abs/2412.07772)  
**代码**: [https://causvid.github.io/](https://causvid.github.io/) (项目页+代码)  
**领域**: 扩散模型  
**关键词**: 视频生成, 自回归扩散, 蒸馏, KV缓存, 实时视频

## 一句话总结
CausVid 通过非对称蒸馏将预训练的双向视频扩散 Transformer 蒸馏为因果自回归 4 步生成器，结合 ODE 初始化和 KV 缓存，实现 9.4 FPS 的流式视频生成（比 CogVideoX 快 160×），在 VBench-Long 基准上以 84.27 分排名第一。

## 研究背景与动机

**领域现状**：当前最先进的视频扩散模型（如 CogVideoX、MovieGen）基于 Diffusion Transformer (DiT) 架构，使用双向注意力在所有帧之间建立依赖关系。这类模型生成质量出色，但存在严重的延迟问题——生成单帧也需要处理整个视频序列，且需要数十步去噪迭代。

**现有痛点**：(1) **高延迟**：双向依赖导致必须等整个视频生成完才能看到任何结果（CogVideoX 生成 128 帧需要 208 秒）；(2) **无法交互**：当前帧的生成依赖未来帧的条件输入，无法响应实时用户输入；(3) **长度限制**：计算和内存成本随帧数二次增长，长视频生成代价极高。

**核心矛盾**：自回归模型可解决延迟和交互问题，但面临严重的误差累积——每帧基于可能有缺陷的前序帧生成，误差随时间放大，且现有自回归视频模型的质量显著落后于双向模型。

**本文目标** 如何将双向视频扩散模型的质量优势转移到自回归架构中，同时实现快速流式生成和抵抗误差累积？

**切入角度**：作者观察到可以利用 Distribution Matching Distillation (DMD) 的灵活性——DMD 的监督在分布层面而非轨迹层面，允许教师和学生使用不同架构。这意味着可以用双向教师监督因果学生，让学生同时获得双向模型的质量和因果模型的效率。

**核心 idea**：用非对称 DMD 蒸馏将多步双向视频扩散模型蒸馏为 4 步因果自回归生成器，以 ODE 初始化稳定训练，通过 KV 缓存实现流式推理。

## 方法详解

### 整体框架
CausVid 的 pipeline：(1) 用 3D VAE 将视频压缩到潜在空间（每 16 帧 → 5 帧潜在帧为一个 chunk）；(2) 在潜在空间中，因果扩散 Transformer 按 chunk 自回归生成——当前 chunk 内使用双向注意力（保持局部时间一致性），chunk 间使用因果注意力（防止依赖未来帧）；(3) 训练分两阶段：先用教师 ODE 轨迹初始化学生，再用非对称 DMD 损失蒸馏；(4) 推理时用 KV 缓存高效流式生成。

### 关键设计

1. **Block-wise Causal Attention（块级因果注意力架构）**:

    - 功能：将双向 DiT 改造为支持自回归生成的因果架构，同时保持 chunk 内的时间一致性
    - 核心思路：定义注意力掩码 $M_{i,j} = 1$ 当 $\lfloor j/k \rfloor \leq \lfloor i/k \rfloor$，其中 $k$ 是 chunk 大小。即同一 chunk 内的帧可以互相注意（双向），但不能注意到未来 chunk 的帧。类似 decoder-only LLM，在每次迭代中可以利用所有输入帧的监督信号。训练时采用 Diffusion Forcing 策略——每个 chunk 有独立的噪声时间步 $t^i$。
    - 设计动机：纯帧级因果注意力会丧失局部时间一致性，而 3D VAE 需要一整个 chunk 的潜在帧才能解码像素，块级因果不增加额外延迟。从预训练双向模型初始化权重可加速收敛。

2. **Asymmetric Distillation（非对称蒸馏）**:

    - 功能：将多步双向教师模型蒸馏为 4 步因果学生模型，使学生获得教师级别的质量并有效抑制误差累积
    - 核心思路：基于 DMD2 框架，教师 $s_{data}$ 使用双向注意力，学生 $G_\phi$ 使用因果注意力。训练时学生对含噪视频帧做 4 步去噪预测，然后用 DMD loss 对齐学生输出分布和数据分布。核心梯度公式：$\nabla_\phi \mathcal{L}_{DMD} \approx -\mathbb{E}_t[(s_{data} - s_{gen,\xi}) \cdot \frac{dG_\phi}{d\phi}]$，其中 $s_{gen,\xi}$ 是在学生输出上在线训练的 score function。使用 two time-scale update rule（比率 5）交替更新学生和生成器 score function。
    - 设计动机：直接从因果教师蒸馏会继承因果模型的缺陷（质量低、误差累积）。DMD 的分布级监督允许教师和学生使用不同架构——双向教师提供了更高质量的分布目标。实验证明非对称蒸馏的因果学生甚至超越了多步因果模型的质量。

3. **ODE-based Student Initialization（ODE 初始化）**:

    - 功能：在蒸馏前用教师的 ODE 轨迹预训练学生，稳定后续 DMD 训练的收敛
    - 核心思路：先用双向教师生成 1000 组 ODE 轨迹对（从纯噪声到清晰视频的完整路径），取与学生推理时间步匹配的子集。学生用回归损失在这些轨迹对上预训练 3000 迭代：$\mathcal{L}_{init} = \mathbb{E}[\|G_\phi(\{x_{t^i}\}, \{t^i\}) - \{x_0^i\}\|^2]$。
    - 设计动机：由于架构差异（双向 vs 因果），直接用 DMD 损失训练会不稳定。ODE 初始化给学生提供了一个合理的起始点——它已经大致知道如何从噪声映射到清晰视频，后续 DMD 训练只需在此基础上微调分布匹配。

### 损失函数 / 训练策略
两阶段训练：(1) ODE 初始化阶段用 MSE 回归损失，训练 3000 步，学习率 $5 \times 10^{-6}$；(2) 非对称 DMD 蒸馏阶段用 DMD loss + 在线 score function 训练，6000 步，学习率 $2 \times 10^{-6}$，guidance scale 3.5。全部训练在 64 块 H100 上约 2 天完成。推理时使用均匀时间步 [999, 748, 502, 247] 进行 4 步去噪。

## 实验关键数据

### 主实验

| 方法 | 视频长度 | Temporal Quality↑ | Frame Quality↑ | Text Alignment↑ | 延迟(s)↓ | 吞吐(FPS)↑ |
|------|---------|-------------------|----------------|-----------------|---------|-----------|
| CogVideoX-5B | 6s | 89.9 | 59.8 | 29.1 | 208.6 | 0.6 |
| MovieGen | 10s | 91.5 | 61.1 | 28.8 | - | - |
| Pyramid Flow | 10s | 89.6 | 55.9 | 27.1 | 6.7 | 2.5 |
| **CausVid (Ours)** | **10s** | **94.7** | **64.4** | **30.1** | **1.3** | **9.4** |

长视频（30s）：

| 方法 | Temporal Quality↑ | Frame Quality↑ | Text Alignment↑ |
|------|-------------------|----------------|-----------------|
| FIFO-Diffusion | 93.1 | 57.9 | 29.9 |
| Pyramid Flow | 89.0 | 48.3 | 24.4 |
| **CausVid (Ours)** | **94.9** | **63.4** | 28.9 |

### 消融实验

| 配置 | Causal? | #步 | Temporal↑ | Frame↑ | Text↑ |
|------|---------|-----|-----------|--------|-------|
| 双向教师 | ✗ | 100 | 94.6 | 62.7 | 29.6 |
| 因果微调 | ✓ | 100 | 92.4 | 60.1 | 28.5 |
| ODE init + 无教师蒸馏 | ✓ | 4 | 92.9 | 48.1 | 25.3 |
| ODE init + 因果教师 | ✓ | 4 | 91.9 | 61.7 | 28.2 |
| **ODE init + 双向教师** | ✓ | **4** | **94.7** | **64.4** | **30.1** |

### 关键发现
- **非对称蒸馏是关键突破**：因果教师蒸馏 (91.9) 远不如双向教师蒸馏 (94.7)。4 步因果学生的 Temporal Quality 甚至超越了 100 步双向教师 (94.7 vs 94.6)
- **DMD 有效抑制误差累积**：100 步因果模型在 30s 视频中严重退化（Fig.8 橙线），但经 DMD 蒸馏的 4 步因果学生保持稳定质量（蓝线）
- **ODE 初始化不可或缺**：没有 ODE 初始化直接用 DMD 训练不稳定，有 ODE 初始化后 Frame Quality 从 48.1 提升到 64.4
- 延迟降低 **160×**（208.6s → 1.3s），吞吐提升 **16×**（0.6 → 9.4 FPS）
- 人类偏好研究中 CausVid 一致优于 MovieGen、CogVideoX、Pyramid Flow（胜率 >50%）

## 亮点与洞察
- **非对称蒸馏**的核心洞察——DMD 的分布级监督允许教师和学生使用不同架构，这打破了"蒸馏=架构一致"的常规假设。双向教师→因果学生的路径让学生获得了教师的质量优势和自身的效率优势，可推广到任何需要改变推理特性的蒸馏场景
- **蒸馏可以反过来解决误差累积**：这是一个反直觉的发现——4 步蒸馏学生比 100 步因果教师的误差累积更小，原因是 DMD 在分布层面对齐而非逐帧回归，且双向教师的全局一致性知识在蒸馏中传递给了学生
- **KV 缓存 + 块级因果注意力**的组合使视频扩散模型首次实现了类似 LLM 的流式生成范式

## 局限与展望
- 极长视频（>10 分钟）仍有质量退化，误差累积策略需要进一步改进
- 受限于 3D VAE 设计，必须生成 5 帧潜在帧后才能解码像素，帧级 VAE 可进一步降低延迟
- 基于反向 KL 的 DMD 目标可能降低输出多样性，可考虑 EM-Distillation 等替代方案
- 当前分辨率 352×640 较低，扩展到更高分辨率需要更多工程优化
- 视频-视频翻译和图像-视频功能均为零样本，专门微调可能进一步提升质量

## 相关工作与启发
- **vs CogVideoX**: 同为 DiT 架构但 CogVideoX 是双向多步，CausVid 是因果 4 步。CausVid 在质量上超越 CogVideoX（94.7 vs 89.9 Temporal Quality）且快 160×
- **vs Pyramid Flow**: Pyramid Flow 也支持自回归但仍需多步去噪，且在长视频中严重退化（Frame Quality 48.3）。CausVid 通过非对称蒸馏有效解决了退化问题
- **vs FIFO-Diffusion**: FIFO 也实现了流式视频生成（Temporal 93.1），但仍需多步去噪且不是真正的自回归。CausVid 在质量和效率上均更优
- **vs DMD/DMD2**: CausVid 扩展了 DMD 到视频域并引入了教师-学生架构不对称的新范式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 非对称蒸馏（双向→因果）是全新范式，蒸馏解决误差累积的发现反直觉且重要
- 实验充分度: ⭐⭐⭐⭐⭐ VBench 全面评估+人类偏好+长视频+消融+多应用场景，非常充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，算法伪代码完整，消融研究组织良好
- 价值: ⭐⭐⭐⭐⭐ 首个在质量上匹配双向模型的自回归视频生成方法，VBench-Long 第一名，160× 加速具有巨大实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Generating, Fast and Slow: Scalable Parallel Video Generation with Video Interface Networks](../../ICCV2025/video_generation/generating_fast_and_slow_scalable_parallel_video_generation_with_video_interface.md)
- [\[CVPR 2025\] Articulated Kinematics Distillation from Video Diffusion Models](articulated_kinematics_distillation_from_video_diffusion_models.md)
- [\[CVPR 2025\] Mimir: Improving Video Diffusion Models for Precise Text Understanding](mimir_improving_video_diffusion_models_for_precise_text_understanding.md)
- [\[CVPR 2025\] Parallelized Autoregressive Visual Generation](parallelized_autoregressive_visual_generation.md)
- [\[CVPR 2025\] InterDyn: Controllable Interactive Dynamics with Video Diffusion Models](interdyn_controllable_interactive_dynamics_with_video_diffusion_models.md)

</div>

<!-- RELATED:END -->
