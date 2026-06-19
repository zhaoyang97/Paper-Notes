---
title: >-
  [论文解读] LongDiff: Training-Free Long Video Generation in One Go
description: >-
  [CVPR 2025][视频生成][长视频生成] LongDiff 通过理论分析揭示短视频模型生成长视频时的两个关键挑战——时序位置模糊和信息稀释，并提出 Position Mapping（GROUP+SHIFT）和 Informative Frame Selection（IFS）两个简洁的时序注意力修改策略，无需训练即可让短视频模型一次性生成高质量长视频。
tags:
  - "CVPR 2025"
  - "视频生成"
  - "长视频生成"
  - "无训练推理"
  - "位置映射"
  - "关键帧选择"
  - "时序注意力"
---

# LongDiff: Training-Free Long Video Generation in One Go

**会议**: CVPR 2025  
**arXiv**: [2503.18150](https://arxiv.org/abs/2503.18150)  
**代码**: 无  
**领域**: 扩散模型 / 视频生成  
**关键词**: 长视频生成, 无训练推理, 位置映射, 关键帧选择, 时序注意力

## 一句话总结

LongDiff 通过理论分析揭示短视频模型生成长视频时的两个关键挑战——时序位置模糊和信息稀释，并提出 Position Mapping（GROUP+SHIFT）和 Informative Frame Selection（IFS）两个简洁的时序注意力修改策略，无需训练即可让短视频模型一次性生成高质量长视频。

## 研究背景与动机

**领域现状**：现有视频扩散模型（如 LaVie、VideoCrafter）主要设计和训练用于生成 16 帧以内的短视频。直接将这些模型用于长视频生成会导致时序不一致和视觉细节丢失。

**现有痛点**：训练式长视频方法（自回归、层次化方法等）需要大量计算资源和稀缺的长视频数据集。现有无训练方法要么基于滑动窗口（如 FreeNoise），限制了远距离帧间交互导致全局一致性差；要么在频域混合特征（如 FreeLong），改进仍然有限。

**核心矛盾**：短视频模型中的相对位置编码（RPE）在长序列中失效——模型无法区分大量不同的相对位置，导致帧序混乱（位置模糊）；同时，长序列中时序注意力的信息熵下界随帧数增长而增大，导致每帧的有效信息减少（信息稀释）。

**本文目标** (1) 如何在不训练的情况下解决长序列中的时序位置模糊？(2) 如何避免长序列中视觉细节的信息稀释？

**切入角度**：从伪维度理论和信息熵理论出发，分析得出这两个挑战可以通过对时序 Transformer 的微妙修改来缓解——减少需区分的位置数量、限制信息传递的帧范围。

**核心 idea**：通过 GROUP+SHIFT 操作将大量相对位置映射到可管理范围并保持可区分性，通过 IFS 限制每帧只与邻近帧和关键帧交互，从而一次性生成高质量长视频。

## 方法详解

### 整体框架

LongDiff 是一种无训练（training-free）方法，仅对现有短视频模型（如 LaVie、VideoCrafter）的时序注意力层进行微小修改。在去噪过程的每一步，视频隐状态通过时序 Transformer 层时，先通过 Position Mapping 重新映射相对位置矩阵，再通过 IFS Mask 限制每帧的时序关联范围。输入是扩展的高斯噪声序列（如 128 帧），输出是一次性生成的长视频。

### 关键设计

1. **Position Mapping（PM）: GROUP + SHIFT**:

    - 功能：解决时序位置模糊问题，让模型在长序列中准确区分帧的相对顺序
    - 核心思路：分两步操作。**GROUP**：将 $2N-1$ 个原始相对位置映射到 $2G-1$ 个组索引，公式为 $p_g = \lceil p / \lceil(N-1)/(G-1)\rceil \rceil$，将大范围位置压缩到模型可处理的范围内。**SHIFT**：对分组后的位置矩阵进行反复移位操作，每次将下三角元素下移一位、上三角横移一位来保持反对称性。经过 $M=S-1$ 次移位，每个位置积累了一个唯一的"分配记录"。最终对 $M+1$ 个位置矩阵分别计算时序注意力，取 softmax 注意力的平均作为最终输出。
    - 设计动机：Theorem 1 证明模型区分位置的能力受注意力 logit 上确界限制，长序列中大量位置无法有效区分。简单裁剪（clip）会丢失远距离位置信息，插值（interpolation）实际引入更多需区分的位置。GROUP 减少位置总数，SHIFT 恢复组内可区分性，两者结合兼顾全局和局部。

2. **Informative Frame Selection（IFS）**:

    - 功能：解决信息稀释问题，保持长视频中的视觉细节
    - 核心思路：首先将时序 Transformer 每层的输入特征 $F \in \mathbb{R}^{N \times C \times hw}$ 通过通道维度的 max/avg/min pooling 转换为伪视频 $V$，归一化到 [0,255]。然后将伪视频均匀分为 $n$ 个片段，在每个片段中选取图像熵和帧差分组合得分最高的帧作为关键帧。最后构建 IFS Mask，使每帧只关注邻近 $L$ 帧和所有关键帧：$\text{Mask}_{ij} = 1$ 当 $|i-j| \leq L$ 或 $j$ 是关键帧。
    - 设计动机：Theorem 2 证明信息熵下界为 $\ln N - 2B$，随帧数 $N$ 增长，每帧有效信息减少。固定窗口限制虽能保持细节但阻碍远距离交互。IFS 通过关键帧作为全局信息摘要，在限制信息传递量的同时保持全局一致性。

3. **理论基础（Theorem 1 & 2）**:

    - 功能：为方法设计提供理论依据
    - 核心思路：Theorem 1 基于伪维度分析，证明模型区分 $g(N)$ 个位置组需要注意力 logit 上界满足 $(g(N)/2)^{1/2r} \cdot \epsilon/4e$，长序列中此条件不满足的帧对超过 60%。Theorem 2 基于信息熵分析，证明注意力权重的熵下界随 $N$ 增长，导致信息稀释。
    - 设计动机：理论分析精确定位了问题所在——位置编码的有限表达力和全连接注意力的信息分散，为 PM 和 IFS 的设计提供了有针对性的指导。

### 损失函数 / 训练策略

LongDiff 是完全无训练的推理时方法，不涉及任何训练或损失函数。PM 和 IFS 直接修改时序注意力的计算过程，M 次 SHIFT 可并行计算，仅带来轻微的推理速度下降。

## 实验关键数据

### 主实验

在 LaVie 上生成 128 帧视频：

| 方法 | SC ↑ | BC ↑ | MS ↑ | TF ↑ | IQ ↑ | OC ↑ |
|------|------|------|------|------|------|------|
| Direct | 88.95 | 93.23 | 92.77 | 91.44 | 64.76 | 22.34 |
| FreeNoise | 92.30 | 95.87 | 96.32 | 94.94 | 67.14 | 24.42 |
| FreeLong | 95.16 | 96.80 | 96.85 | 96.04 | 67.55 | 24.56 |
| **LongDiff** | **98.10** | **98.23** | **97.46** | **96.84** | **68.83** | **25.24** |

在 VideoCrafter-512 上：

| 方法 | SC ↑ | BC ↑ | MS ↑ | TF ↑ | IQ ↑ | OC ↑ |
|------|------|------|------|------|------|------|
| FreeNoise | 91.43 | 93.48 | 93.33 | 91.88 | 68.39 | 22.69 |
| FreeLong | 90.84 | 92.37 | 89.11 | 88.46 | 66.62 | 21.85 |
| **LongDiff** | **93.69** | **95.59** | **94.59** | **93.35** | **70.03** | **23.17** |

### 消融实验

| 配置 | SC ↑ | BC ↑ | MS ↑ | TF ↑ | IQ ↑ | OC ↑ |
|------|------|------|------|------|------|------|
| w/o PM | 91.85 | 94.79 | 95.12 | 93.26 | 65.73 | 22.58 |
| w/o IFS | 94.43 | 96.37 | 93.65 | 92.85 | 65.45 | 23.46 |
| Clip (替代PM) | 91.02 | 94.18 | 94.77 | 92.87 | 65.12 | 22.43 |
| Interpolation | 92.52 | 95.29 | 95.25 | 94.76 | 66.43 | 23.01 |
| Group only (无SHIFT) | 94.49 | 96.63 | 96.74 | 95.79 | 67.45 | 24.62 |
| **Full LongDiff** | **98.10** | **98.23** | **97.46** | **96.84** | **68.83** | **25.24** |

### 关键发现

- PM 和 IFS 缺一不可：去掉 PM 后 SC 从 98.10 降至 91.85（时序一致性大幅下降），去掉 IFS 后 IQ 从 68.83 降至 65.45（视觉细节丢失）
- SHIFT 操作的重要性：仅 GROUP 的效果远不如 GROUP+SHIFT，说明组内位置可区分性的恢复至关重要
- 在两个不同架构的模型（LaVie/RoPE、VideoCrafter/RPE）上都取得一致的全面领先，说明方法通用性强
- 相比 clip 和 interpolation 等简单位置处理，PM 的优势非常显著

## 亮点与洞察

- **理论驱动的方法设计**：从伪维度和信息熵两个理论工具出发定位问题，再针对性设计解决方案。这种"先理论分析、再设计方法"的范式值得学习。这在视频生成领域是罕见的理论驱动工作。
- **GROUP+SHIFT 的精巧组合**：GROUP 降低位置总数满足模型的有限表达力，SHIFT 通过累积"分配记录"恢复可区分性，最终通过多次注意力计算的平均来利用这种可区分性。这种思路可以迁移到任何涉及长序列位置编码的场景（如长文本生成的位置外推）。
- **完全无训练、即插即用**：作为推理时方法，LongDiff 可以直接应用于任何使用相对位置编码的短视频模型，不需要额外数据或训练资源，实用性极强。

## 局限与展望

- 仅在 3D U-Net 架构的模型上验证，尚未在更新的 DiT 架构模型上测试
- 128 帧 (约 8 秒) 的"长视频"在实际应用中仍然较短，更长序列的效果有待验证
- 关键帧检测基于图像熵和帧差分，较为简单，可能在复杂场景中不够准确
- SHIFT 操作需要 M+1 次注意力计算，虽然可并行但仍增加了计算量

## 相关工作与启发

- **vs FreeNoise**: FreeNoise 使用滑动窗口时序注意力，限制了远距离帧的交互。LongDiff 通过 IFS 的关键帧机制保持全局信息交流，在所有指标上全面领先。
- **vs FreeLong**: FreeLong 在频域混合空间和时序特征。LongDiff 从时序注意力的位置编码和信息传递两个更根本的角度切入，效果更好且理论基础更扎实。
- **与 NTK-aware Scaling (LLM) 的联系**: LongDiff 的 Position Mapping 思路与 LLM 领域的位置编码外推方法（如 NTK-aware RoPE scaling）有异曲同工之处，都在解决"训练短序列如何泛化到长序列"的位置编码问题。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 理论驱动 + GROUP/SHIFT 设计精巧，IFS 关键帧选择也有新意
- 实验充分度: ⭐⭐⭐⭐ 两个模型上的全面对比和详细消融，但缺少更长序列和 DiT 模型的验证
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰，方法可视化出色
- 价值: ⭐⭐⭐⭐⭐ 无训练即插即用，理论洞察+实践效果俱佳

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] One-Minute Video Generation with Test-Time Training](one-minute_video_generation_with_test-time_training.md)
- [\[CVPR 2025\] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](streamingt2v_consistent_dynamic_and_extendable_long_video_generation_from_text.md)
- [\[CVPR 2025\] MovieBench: A Hierarchical Movie Level Dataset for Long Video Generation](moviebench_a_hierarchical_movie_level_dataset_for_long_video_generation.md)
- [\[ICML 2025\] Diffusion Adversarial Post-Training for One-Step Video Generation](../../ICML2025/video_generation/diffusion_adversarial_post-training_for_one-step_video_generation.md)
- [\[CVPR 2026\] Free-Lunch Long Video Generation via Layer-Adaptive O.O.D Correction](../../CVPR2026/video_generation/free-lunch_long_video_generation_via_layer-adaptive_ood_correction.md)

</div>

<!-- RELATED:END -->
