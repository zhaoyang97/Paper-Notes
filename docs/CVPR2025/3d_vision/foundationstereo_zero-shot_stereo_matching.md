---
title: >-
  [论文解读] FoundationStereo: Zero-Shot Stereo Matching
description: >-
  [CVPR 2025][3D视觉][零样本立体匹配] 提出 FoundationStereo，一个大规模立体深度估计基础模型，通过百万级高保真合成数据集、Side-Tuning Adapter 融合单目深度先验、以及混合代价体过滤（含 Axial-Planar Convolution 和 Disparity Transformer），实现了无需目标域微调的强零样本泛化性能。
tags:
  - "CVPR 2025"
  - "3D视觉"
  - "零样本立体匹配"
  - "基础模型"
  - "单目深度先验"
  - "合成数据"
  - "代价体过滤"
---

# FoundationStereo: Zero-Shot Stereo Matching

**会议**: CVPR 2025  
**arXiv**: [2501.09898](https://arxiv.org/abs/2501.09898)  
**代码**: [https://nvlabs.github.io/FoundationStereo/](https://nvlabs.github.io/FoundationStereo/)  
**领域**: 3D视觉 / 立体匹配  
**关键词**: 零样本立体匹配, 基础模型, 单目深度先验, 合成数据, 代价体过滤

## 一句话总结
提出 FoundationStereo，一个大规模立体深度估计基础模型，通过百万级高保真合成数据集、Side-Tuning Adapter 融合单目深度先验、以及混合代价体过滤（含 Axial-Planar Convolution 和 Disparity Transformer），实现了无需目标域微调的强零样本泛化性能。

## 研究背景与动机

**领域现状**：深度立体匹配在 per-domain fine-tuning 设置下已趋于饱和顶尖 benchmark，主流方法包括代价体过滤（如 GwcNet、IGEV）和迭代精炼（如 RAFT-Stereo），但都依赖目标域微调才能获得竞争性结果。

**现有痛点**：其他视觉任务（如分割 SAM、单目深度 DepthAnything）已展现出强大的零样本泛化能力，但立体匹配领域始终未出现真正的"基础模型"。现有跨域泛化方法主要在 Scene Flow（仅 40K 对）上训练，数据规模和多样性严重不足。网络架构方面，3D CNN 受限于小核尺寸，难以在大视差场景下捕捉全局上下文。

**核心矛盾**：立体匹配的零样本泛化受限于训练数据的规模与多样性，以及网络架构的表示能力——现有结构无法有效利用大规模训练数据。

**本文目标** 构建立体匹配的基础模型，使其无需目标域微调即可在多样化场景中达到甚至超过 fine-tuned 方法的精度。

**切入角度**：从三个维度同时发力——（1）百万级高保真合成数据集消除数据瓶颈；（2）适配单目深度基础模型的丰富先验来弥合 sim-to-real 差距；（3）设计可扩展的架构组件提升跨视差和空间维度的上下文推理能力。

**核心 idea**：通过大规模数据 + 单目先验适配 + 长程代价体过滤三管齐下，将立体匹配提升至基础模型级零样本泛化。

## 方法详解

### 整体框架
输入左右立体图像对，通过 Side-Tuning Adapter (STA) 提取融合了 DepthAnythingV2 先验的多尺度特征，构建混合代价体（分组相关 + 特征拼接），用 Attentive Hybrid Cost Filtering (AHCF) 进行代价体过滤（含 APC 沙漏网络 + Disparity Transformer），soft-argmin 产生初始视差，再经多尺度 GRU 迭代精炼得到最终稠密视差图。

### 关键设计

1. **Side-Tuning Adapter (STA)**:

    - 功能：将预训练单目深度模型（DepthAnythingV2）的丰富语义和几何先验适配到立体匹配任务
    - 核心思路：冻结 DepthAnythingV2 的 ViT 骨干提取特征，将其 DPT head 输出下采样并与 CNN（EdgeNeXt-S）同级特征拼接，形成 1/4 尺度的混合特征。CNN 网络学习适配 ViT 特征到立体匹配任务。作者比较了三种融合策略，发现最简单的"下采样+拼接"显著优于 ViT-Adapter 式交互和直接使用 ViT 特征。
    - 设计动机：DepthAnythingV2 在海量真实图像上训练过，包含丰富的语义和几何先验，能弥补合成训练数据与真实场景的 gap。冻结 ViT 避免破坏已学到的先验，CNN side-tuning 让模型学习如何将单目先验转化为立体匹配所需的特征。

2. **Axial-Planar Convolution (APC)**:

    - 功能：在代价体沙漏过滤中扩大感受野，特别是在大视差场景下
    - 核心思路：将标准 $3\times3\times3$ 3D 卷积解耦为：空间维度卷积 $K_s \times K_s \times 1$ + 视差维度卷积 $1 \times 1 \times K_d$，类似于 3D 版的可分离卷积但不拆分通道。这样可以使用更大的核尺寸（如 $K_s=5, K_d=7$）而不会爆显存。
    - 设计动机：传统 $3\times3\times3$ 卷积在大视差时感受野不足，直接增大到 $5\times5\times5$ 会使 80GB GPU OOM。APC 的解耦设计在同等显存下大幅提升表示能力，让模型能更好地利用大规模训练数据。

3. **Disparity Transformer (DT)**:

    - 功能：在代价体内进行全局视差维度的自注意力推理
    - 核心思路：先用 $4\times4\times4$ 步长的 3D 卷积下采样代价体，reshape 后沿视差维度做 FlashAttention 多头自注意力（4个 transformer encoder block），再三线性插值恢复分辨率并与沙漏输出相加。每个空间位置的不同视差级别之间建模全局依赖。
    - 设计动机：代价体的视差维度编码了匹配概率分布，长程依赖对解决重复纹理、大面积无纹理区域至关重要。3D CNN 即使用 APC 也只能捕捉局部视差上下文，DT 弥补了这一全局推理缺口。

### 损失函数 / 训练策略
损失函数包含两部分：初始视差用 smooth L1 损失，迭代精炼的视差序列用指数增权的 L1 损失（$\gamma=0.9$）。训练在 32 张 A100 上进行，总 batch 128，200K 步，AdamW 优化器，学习率 1e-4。输入裁剪至 320×736，22 次 GRU 迭代。数据集为自有 FSD + 多个公开合成数据集混合。配备自动自筛选管线（iterative self-curation）：用当前模型在 FSD 上评估，BP-2>60% 的样本视为模糊样本并重新生成，交替两轮。

## 实验关键数据

### 主实验（零样本泛化）

| 数据集 | 指标 | FoundationStereo | 之前最佳 | 提升 |
|--------|------|-----------------|---------|------|
| Middlebury | BP-2↓ | **1.1** | 7.5 (NMRF) | -85% |
| ETH3D | BP-1↓ | **0.5** | 1.8 (本文Scene Flow版) | -72% |
| KITTI-12 | D1↓ | **2.3** | 3.2 (本文/S-IGEV*) | -28% |
| KITTI-15 | D1↓ | **2.8** | 4.5 (S-IGEV*) | -38% |

### 消融实验

| 配置 | Middlebury BP-2 | ETH3D BP-1 | 说明 |
|------|----------------|-----------|------|
| Full model | **1.1** | **0.5** | 完整模型 |
| W/o STA | 明显下降 | 明显下降 | 无单目先验，ambiguous 区域预测差 |
| W/o AHCF (用3D CNN) | 下降 | 下降 | 细结构和重复纹理区域退化 |
| STA design (a) 直接ViT | 较差 | 较差 | ViT 特征未充分适配到立体任务 |
| STA design (b) ViT-Adapter | 中等 | 中等 | 交互式融合反而不如简单拼接 |

### 关键发现
- 即使仅在 Scene Flow 上训练，FoundationStereo 也全面超越所有对比方法，说明 STA 引入单目先验的有效性
- STA 对光照不一致区域（如灯具阴影）和几何模糊区域（如吉他音孔）帮助最大
- AHCF 在细长重复结构上的改善最显著
- 自筛选管线有效定位了数据集中的模糊样本（如过度重复纹理、纯色无信息区域），提升了训练稳定性

## 亮点与洞察
- **Side-Tuning 策略的启发性**：冻结预训练基础模型作为"知识源"，用轻量 CNN 做任务适配——这种模式可推广到任何需要利用大模型先验但任务差异较大的场景（如将 CLIP 适配到检测、将 SAM 适配到跟踪）。
- **APC 的工程价值**：3D 可分离卷积的空间-视差解耦是一个实用技巧，显著扩大感受野同时控制显存，适用于所有需要大核 3D 卷积的场景（如视频理解、4D 重建）。
- **自筛选训练管线**：用模型反过来清洗训练数据的闭环设计，在大规模合成数据场景中非常实用，值得在其他依赖合成数据的领域（如 6DoF 位姿、光流）中借鉴。

## 局限与展望
- 百万级数据集使用 NVIDIA Omniverse 生成，复现门槛高，外部研究者难以复制
- DepthAnythingV2 是冻结使用的，STA 的适配能力可能受限于 ViT 特征的表示瓶颈
- 推理时 32 次 GRU 迭代意味着计算开销较大，可探索自适应迭代次数
- 可改进：支持可变分辨率推理避免 resize 带来的精度损失；探索 DepthAnything 的轻量版本降低推理成本

## 相关工作与启发
- **vs RAFT-Stereo / IGEV 系列**：传统方法在特定域微调后很强，但零样本能力弱。FoundationStereo 用数据规模和架构改进弥补了泛化差距，直接用零样本媲美甚至超越精调结果。
- **vs 单目深度模型（DepthAnythingV2）**：FoundationStereo 吸收了单目深度模型的先验但保持了立体匹配的亚像素精度和绝对尺度能力。
- **vs 并发工作**：并发的 monocular prior enhanced 方法也利用单目先验增强 correlation volume，但 FoundationStereo 通过 STA+AHCF+大规模数据的组合取得了更强的综合结果。

## 评分
- 新颖性: ⭐⭐⭐⭐ 各组件单独看并不全新，但组合和工程优化做到了极致
- 实验充分度: ⭐⭐⭐⭐⭐ 5个benchmark全面评测，消融充分，包含in-the-wild定性结果
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，方法描述详细，图表质量高
- 价值: ⭐⭐⭐⭐⭐ 立体匹配领域的里程碑式工作，首次实现了真正的零样本基础模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[CVPR 2026\] Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/fast-foundationstereo_real-time_zero-shot_stereo_matching.md)
- [\[CVPR 2026\] PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts](../../CVPR2026/3d_vision/promptstereo_zero-shot_stereo_matching_via_structure_and_motion_prompts.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](../../ICCV2025/3d_vision/zerostereo_zero-shot_stereo_matching_from_single_images.md)

</div>

<!-- RELATED:END -->
