---
title: >-
  [论文解读] Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction
description: >-
  [CVPR 2025][图像生成][视频tokenizer] 提出 CoordTok，一种可扩展的视频 tokenizer，将视频编码为因子化 triplane 表示，解码器学习从随机采样的 $(x,y,t)$ 坐标到对应 patch 像素的映射（而非一次重建所有帧），使得可以直接在 128 帧长视频上训练大型 tokenizer，将 128 帧视频编码为仅 1280 个 token（基线需要 6144-8192 个），并驱动 DiT 实现 128 帧一次性视频生成（FVD 369.3 SOTA）。
tags:
  - "CVPR 2025"
  - "图像生成"
  - "视频tokenizer"
  - "Triplane表示"
  - "坐标重建"
  - "长视频编码"
  - "视频压缩"
  - "扩散生成"
---

# Efficient Long Video Tokenization via Coordinate-based Patch Reconstruction

**会议**: CVPR 2025  
**arXiv**: [2411.14762](https://arxiv.org/abs/2411.14762)  
**代码**: [https://huiwon-jang.github.io/coordtok](https://huiwon-jang.github.io/coordtok)  
**领域**: 图像/视频生成  
**关键词**: 视频tokenizer, Triplane表示, 坐标重建, 长视频编码, 视频压缩, 扩散生成

## 一句话总结

提出 CoordTok，一种可扩展的视频 tokenizer，将视频编码为因子化 triplane 表示，解码器学习从随机采样的 $(x,y,t)$ 坐标到对应 patch 像素的映射（而非一次重建所有帧），使得可以直接在 128 帧长视频上训练大型 tokenizer，将 128 帧视频编码为仅 1280 个 token（基线需要 6144-8192 个），并驱动 DiT 实现 128 帧一次性视频生成（FVD 369.3 SOTA）。

## 研究背景与动机

**领域现状**：视频 tokenizer 是视频生成模型的基础组件，负责将高维视频压缩为紧凑的 token 表示。现有 tokenizer（如 TATS-AE、MAGVIT-AE、PVDM-AE）在压缩率上持续提升，但训练时需要重建所有帧，计算和内存成本随视频长度线性增长。

**现有痛点**：(1) 训练成本问题——在单张 4090 GPU 上，PVDM-AE 训练 128 帧视频时直接 OOM，大部分 tokenizer 只能在 16 帧短视频上训练；(2) 时序一致性问题——只能编码短片段的 tokenizer 无法充分利用视频的时序连贯性，当多个短片段拼接编码长视频时，片段交界处出现像素值不一致（如 Figure 1b 所示）。

**核心矛盾**：视频的时序连贯性是高效压缩的关键先验（类似视频编解码器利用关键帧+差异编码），但现有 tokenizer 因训练成本限制只能在短片段上训练，无法利用这一先验。

**本文目标** 如何设计一个可以直接在长视频上训练的 tokenizer，从而充分利用时序连贯性实现更高效的压缩？

**切入角度**：受 3D 生成模型（如 NeRF/triplane）的启发——它们通过学习从随机采样坐标到 RGB/密度值的映射来避免全坐标一次性训练——将视频重建问题类似地转化为从 $(x,y,t)$ 坐标到对应 patch 的映射学习问题。

**核心 idea**：将视频编码为三个2D平面（triplane）的紧凑表示，解码时只需随机采样少量坐标重建对应patch，训练成本与视频长度解耦，从而可直接在128帧长视频上训练获得更高效的tokenization。

## 方法详解

### 整体框架

CoordTok 包含编码器和解码器：编码器将视频 $\mathbf{x}$ 分为时空 patch 经 Transformer 处理后，通过 cross-self attention 层将视频特征投影到因子化 triplane 表示 $\mathbf{z} = [\mathbf{z}^{xy}, \mathbf{z}^{yt}, \mathbf{z}^{xt}]$（分别捕获全局内容、y轴运动、x轴运动）。解码器接收随机采样的 $N$ 个归一化坐标 $(i,j,k) \in [0,1]^3$，通过双线性插值从 triplane 查询坐标特征，经 self-attention 处理后投影为对应 patch 的像素值。训练时仅需重建 3% 的 patch 即可达到良好性能。

### 关键设计

1. **因子化 Triplane 编码器**:

    - 功能：将视频压缩为三个 2D 平面的紧凑表示，避免 3D latent 的高内存开销
    - 核心思路：引入可学习嵌入 $\mathbf{z}_0 = [\mathbf{z}_0^{xy}, \mathbf{z}_0^{yt}, \mathbf{z}_0^{xt}]$（形状分别为 $H' \times W'$、$W' \times T'$、$H' \times T'$），通过 cross-self attention 层将视频特征 $\mathbf{e}$（来自 ViT 处理的时空 patch）聚合到这三个平面中。$\mathbf{z}^{xy}$ 捕获跨时间的全局内容（场景布局、外观），$\mathbf{z}^{yt}$ 和 $\mathbf{z}^{xt}$ 捕获沿两个空间轴的运动信息。每个可学习嵌入被拆分为 4 个子嵌入以增加序列长度、提升模型利用率。
    - 设计动机：三个 2D 平面代替一个 3D latent，token 数量从 $H' \times W' \times T'$ 降为 $H'W' + W'T' + H'T'$。对于 128 帧视频，这一差异使 token 数量从数千降至一千余个。Triplane 的因子化天然分离了内容和运动，有利于下游生成模型分别建模。

2. **坐标采样 + Patch 重建解码器**:

    - 功能：使训练成本与视频长度解耦，允许直接在长视频上训练
    - 核心思路：将视频分为不重叠的时空 patch，将每个 patch 索引转换为归一化坐标 $(i,j,k) \in [0,1]^3$。训练时随机采样 $N$ 个坐标（仅 3% 的 patch），通过双线性插值从 triplane 获取坐标特征 $\mathbf{h} = \text{Concat}(\mathbf{h}^{xy}, \mathbf{h}^{yt}, \mathbf{h}^{xt})$，经 self-attention 层在坐标间交互信息后，用线性投影层映射为对应 patch 的 RGB 像素。损失为 $\ell_2$ 重建损失。
    - 设计动机：传统 tokenizer 必须一次重建所有帧→内存和计算与帧数线性相关→限制为短视频训练。坐标采样 3% 的 patch 即可有效训练，使得 128 帧视频训练的 batch size 仍可保持在 256（对比 PVDM-AE 在 128 帧时 OOM）。

3. **帧采样微调 + LPIPS 损失**:

    - 功能：在坐标采样预训练基础上提升重建的感知质量
    - 核心思路：主训练阶段完成后（1M 次迭代），切换为帧采样模式——随机采样几帧，重建这些帧的所有坐标，同时使用 $\ell_2$ 和 LPIPS 损失微调 50K 次迭代。LPIPS 需要完整帧才能计算，因此不能在坐标采样阶段使用。
    - 设计动机：从训练一开始就用帧采样会因采样多样性不足而效果不佳（Table 4 验证），但在坐标采样充分训练后微调可以有效提升感知质量。两阶段策略结合了坐标采样的高效性和帧采样的高质量。

### 损失函数

- **主训练阶段**：$\ell_2$ 重建损失，$\mathcal{L} = \|\hat{\mathbf{x}}_{ijk} - \mathbf{x}_{ijk}\|_2^2$，仅在随机采样的 $N=1024$ 个坐标上计算
- **微调阶段**：$\ell_2 + $ LPIPS 联合损失，在随机采样的完整帧（$N=4096$ 坐标）上计算

## 实验关键数据

### 长视频重建质量（128帧, 128×128）

| 方法 | Token类型 | Token数量 | 训练帧数 | PSNR↑ | LPIPS↓ | rFVD↓ |
|:--|:--|:--|:--|:--|:--|:--|
| OmniTok-CV | Continuous | 8192 | 17 | 28.3 | 0.081 | 49.5 |
| CosmosTokenizer* | Continuous | 8192 | 17 | 28.5 | 0.119 | 87.8 |
| PVDM-AE | Continuous | 6144 | 16 | 26.5 | 0.120 | 66.5 |
| OmniTok-CV | Continuous | 1024 | 17 | 23.2 | 0.175 | 396.7 |
| PVDM-AE | Continuous | 1152 | 16 | 19.1 | 0.333 | 1270.1 |
| **CoordTok** | **Continuous** | **1280** | **128** | **28.6** | **0.066** | **102.9** |

### 视频生成（128帧, UCF-101）

| 方法 | FVD↓ | 生成时间(s) | 显存(GB) |
|:--|:--|:--|:--|
| StyleGAN-V | 1773.4 | - | - |
| PVDM-L | 505.0 | 116.9 | 4.0 |
| HVDM | 549.7 | 52.1 | 3.9 |
| Latte-L/2 | 1901.8 | 21.4 | 3.1 |
| **CoordTok-SiT-L/2** | **369.3** | **9.8** | **4.5** |

### 消融与分析

| 分析项 | 关键结论 |
|:--|:--|
| 坐标采样比例 | 3% patch 即足够，更多不会显著改善 |
| 模型规模 | Large > Base > Small，更大模型持续改善 |
| Triplane 空间维度 | 16×16 最优，8×8 不足，32×32 过冗余 |
| Triplane 时间维度 | 32 最优，16 不足，64 过冗余 |
| 帧采样 vs 坐标采样 | 纯帧采样（从头开始）不如坐标采样，因为多样性不足 |

### 关键发现

- **长视频训练带来巨大压缩优势**：CoordTok 用 1280 token 达到的重建质量（rFVD 102.9）与基线用 6144-8192 token 的质量相当或更优
- **Triplane 对动态视频更敏感**：Pearson 相关分析表明 CoordTok 的重建质量与视频动态程度的相关性更强（r=0.617），说明运动分解是其关键压力点
- **高效 tokenization 改善下游生成**：用 1280 token 训练的 SiT 比用 3072 token 训练的效果更好（FVD 约低50+），因为更少 token 降低了生成模型的学习难度
- **生成速度极快**：128 帧一次性生成仅需 9.8s（对比 PVDM-L 的 116.9s，快 12 倍）

## 亮点与洞察

1. **跨领域灵感迁移**：将 3D 生成/NeRF 中坐标采样训练的思想迁移到视频 tokenizer 设计中，非常自然且有效
2. **训练成本与视频长度解耦**：这是方法论上的关键突破——通过只重建随机 3% 的 patch，128 帧训练的成本与 16 帧相当
3. **更少 token = 更好生成**：这个反直觉的发现很有启发性——token 数量和重建质量之间的最优点不等于生成模型的最优点，更紧凑的表示反而降低了生成难度
4. **Triplane 的内容-运动分离**：xy平面捕获全局内容、yt/xt平面捕获运动的天然分离在可视化中得到验证

## 局限性与可改进方向

1. **分辨率受限**：所有实验在 128×128 分辨率上进行，高分辨率（如 256×256, 512×512）上的效果未验证
2. **仅在 UCF-101 上实验**：数据集规模和多样性有限，未在大规模数据集上验证泛化性
3. **Triplane 对动态视频的局限**：分析表明越动态的视频重建越难——因为快速运动使内容-运动分解更困难
4. **无条件生成**：下游生成仅做了无条件模型，未与文本条件生成集成
5. **坐标重建 vs 像素级细节**：patch 级重建可能在精细纹理上不如逐像素解码器

## 相关工作与启发

- **PVDM**：同样使用 triplane 表示但一次解码所有帧，是 CoordTok 的直接对比——证明了解码器设计的重要性
- **TiTok**：1D token 的图像 tokenizer，本文将因子化表示的思想扩展到视频维度
- **NeRF / 3D Triplane 生成**：坐标采样训练的灵感来源（如 LRM、Instant3D 等3D生成工作）
- **视频编解码器（HEVC, AV1等）**：关键帧+差异编码是经典的时序冗余利用策略，CoordTok 的 triplane 可视为其学习版本
- **启发**：视频的核心特性是时序冗余——任何视频处理模型都应该思考如何利用这一先验，而不是将每帧独立对待。训练长度与推理效率之间存在有趣的权衡关系

## 评分

⭐⭐⭐⭐ — 思路新颖且简洁优雅，将3D生成的坐标采样思想无缝迁移到视频tokenizer，训练效率突破显著（128帧直接训练），且下游生成效果出色。但分辨率和数据集规模受限，高分辨率大规模场景有待验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Language-Guided Image Tokenization for Generation](language-guided_image_tokenization_for_generation.md)
- [\[CVPR 2026\] Spk2VidNet: A Hierarchical Recurrent Architecture for High-Fidelity Video Reconstruction from Long Spike-Camera Streams](../../CVPR2026/image_generation/spk2vidnet_a_hierarchical_recurrent_architecture_for_high-fidelity_video_reconstr.md)
- [\[ICCV 2025\] Efficient Autoregressive Shape Generation via Octree-Based Adaptive Tokenization](../../ICCV2025/image_generation/efficient_autoregressive_shape_generation_via_octree-based_adaptive_tokenization.md)
- [\[CVPR 2025\] OFER: Occluded Face Expression Reconstruction](ofer_occluded_face_expression_reconstruction.md)
- [\[CVPR 2025\] Divot: Diffusion Powers Video Tokenizer for Comprehension and Generation](divot_diffusion_powers_video_tokenizer_for_comprehension_and_generation.md)

</div>

<!-- RELATED:END -->
