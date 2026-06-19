---
title: >-
  [论文解读] Uncertainty-Driven Spectral Compressive Imaging with Spatial-Frequency Transformer
description: >-
  [ECCV 2024][模型压缩][高光谱图像重建] 本文提出 Specformer，通过并行的空间局部窗口自注意力（LWSA）和频率域自注意力（FWSA）模块充分捕获高光谱图像（HSI）的空间稀疏性和光谱间相似性先验，并引入不确定性驱动的损失函数增强网络对纹理丰富和边缘区域的重建能力，在模拟和真实 HSI 数据集上以更低计算量超越 SOTA。
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "高光谱图像重建"
  - "编码孔径快照光谱成像"
  - "Transformer"
  - "不确定性驱动"
  - "自注意力"
---

# Uncertainty-Driven Spectral Compressive Imaging with Spatial-Frequency Transformer

**会议**: ECCV 2024  
**代码**: [https://github.com/bianlab/Specformer](https://github.com/bianlab/Specformer)  
**领域**: 模型压缩 / 光谱成像  
**关键词**: 高光谱图像重建, 编码孔径快照光谱成像, 空间-频率Transformer, 不确定性驱动, 自注意力

## 一句话总结
本文提出 Specformer，通过并行的空间局部窗口自注意力（LWSA）和频率域自注意力（FWSA）模块充分捕获高光谱图像（HSI）的空间稀疏性和光谱间相似性先验，并引入不确定性驱动的损失函数增强网络对纹理丰富和边缘区域的重建能力，在模拟和真实 HSI 数据集上以更低计算量超越 SOTA。

## 研究背景与动机

**领域现状**：编码孔径快照光谱成像（CASSI）系统通过单次快照获取 2D 压缩测量，然后通过算法重建出 3D 高光谱数据立方体。近年来基于学习的方法（如 TSA-Net、MST、CST 等 Transformer 方法）在重建质量上取得了显著进步，逐步替代传统的迭代优化方法。

**现有痛点**：现有学习方法存在两个核心问题。第一，它们很少同时利用 HSI 的空间稀疏性先验和光谱间相似性先验——空间方法（如窗口注意力）擅长捕捉局部空间特征但忽略了跨光谱关联，而光谱方法则相反。第二，现有方法对所有图像区域一视同仁，忽略了纹理丰富区域和边缘区域比平滑区域更难重建的事实，导致整体重建质量受限于这些困难区域。

**核心矛盾**：空间信息和频率（光谱）信息的联合建模需要大量计算，传统做法是串行堆叠不同类型的注意力模块，导致计算冗余。同时，均匀的损失权重分配使得网络倾向于优化容易的平滑区域，而非真正需要关注的困难区域。

**本文目标** （1）如何高效地同时建模空间稀疏性和光谱间相似性？（2）如何让网络自适应地关注难以重建的区域？

**切入角度**：作者提出将空间注意力和频率注意力并行化而非串行化，通过并行设计自然建立跨窗口连接，在保持线性复杂度的同时扩大感受野。对于区域自适应问题，作者从贝叶斯深度学习中借鉴不确定性估计思想，让网络自动学习每个像素的重建难度，并将其作为损失权重。

**核心 idea**：用并行的空间-频率注意力模块联合建模 HSI 的两种先验，并用像素级不确定性驱动损失函数自动强化对困难区域的学习。

## 方法详解

### 整体框架
Specformer 采用多尺度 U-shape 网络架构。输入为 CASSI 系统的 2D 压缩测量和对应的光谱编码掩模，输出为重建的 3D 高光谱数据立方体。网络主体由堆叠的 Spatial-Frequency（SF）Block 构成，每个 SF Block 包含并行的 LWSA 和 FWSA 两个注意力分支。网络额外输出一个不确定性图，用于指导损失函数的权重分配。

### 关键设计

1. **空间局部窗口自注意力（LWSA）**:

    - 功能：捕获每个光谱波段内的空间局部特征，关注光谱信息密集的区域
    - 核心思路：将每个光谱波段的特征图划分为非重叠的局部窗口，在窗口内执行标准的多头自注意力。窗口大小固定为 $M \times M$，计算复杂度为 $O(M^2 \cdot H W)$，相对于全局注意力的 $O((HW)^2)$ 是线性的。LWSA 可以引导网络聚焦于光谱信息密集的空间区域，因为这些区域内的 attention score 更高
    - 设计动机：HSI 的空间维度具有稀疏性，空间相邻像素通常具有相似的光谱特征。局部窗口注意力天然匹配这种先验，且计算高效。但单独使用 LWSA 存在跨窗口信息隔离的问题

2. **频率域自注意力（FWSA）**:

    - 功能：捕获光谱间的相似性关系，同时建立跨窗口的长距离依赖
    - 核心思路：FWSA 首先将空间特征通过 FFT 变换到频率域，然后在频率域执行自注意力。频率域的一个关键优势是每个频率分量本质上编码了全局空间信息（频率域的一个点对应空间域的全局模式），因此频率域注意力天然具有全局感受野。具体做法是对特征沿光谱维度做 1D FFT，得到频率特征图，然后计算频率间的注意力得分，最后通过 IFFT 转回空间域
    - 设计动机：HSI 的不同光谱波段之间存在强相关性（相邻波段的光谱响应高度相似）。在频率域中，这种跨光谱相似性可以更紧凑地表示。同时，频率域注意力作为 LWSA 的补充，自然解决了 LWSA 的窗口隔离问题

3. **不确定性驱动损失函数（Uncertainty-Driven Loss）**:

    - 功能：让网络自动识别并强化对纹理丰富/边缘区域等困难区域的学习
    - 核心思路：网络额外输出一个与 HSI 同尺寸的不确定性图 $\sigma(x)$，表示每个像素的重建置信度。损失函数被修改为 $\mathcal{L} = \frac{1}{2\sigma^2(x)} \|y - \hat{y}\|^2 + \frac{1}{2}\log\sigma^2(x)$。第一项使得高不确定性（大 $\sigma$）区域的重建误差权重降低，但第二项的正则化防止 $\sigma$ 无限增大。训练过程中，网络会自动学到纹理/边缘区域的 $\sigma$ 值较小（权重更大），从而强制网络更多关注这些区域
    - 设计动机：传统 MSE 损失对所有像素等权重处理，而 HSI 重建中平滑区域占大多数且容易重建，导致网络训练偏向于优化这些"简单"区域。不确定性驱动损失借鉴了异方差不确定性建模（Kendall & Gal, 2017），为每个像素学习自适应的损失权重

### 损失函数 / 训练策略
总损失为不确定性加权的 L2 损失 + 不确定性正则化项。训练时使用 Adam 优化器，余弦退火学习率调度，在 KAIST 和 CAVE 等模拟数据集上预训练，在真实 CASSI 数据上微调。

## 实验关键数据

### 主实验

| 数据集 | 指标 | Specformer | CST (ECCV22) | MST++ (CVPRW22) | 参数量对比 |
|--------|------|------|----------|------|------|
| 模拟 KAIST | PSNR | **SOTA** | 次优 | 较低 | Specformer 更少参数 |
| 模拟 TSA | PSNR/SSIM | **SOTA** | 次优 | 较低 | 计算量更低 |
| 真实 CASSI | 光谱精度 | **SOTA** | 次优 | 较低 | FLOPs 更低 |

### 消融实验

| 配置 | PSNR | 说明 |
|------|---------|------|
| Full model (LWSA + FWSA + UDL) | 最优 | 完整模型 |
| w/o FWSA (仅空间) | 下降 ~0.5dB | 失去光谱间相似性建模 |
| w/o LWSA (仅频率) | 下降 ~0.3dB | 失去空间局部特征 |
| 串行 LWSA→FWSA | 下降 ~0.2dB 且 FLOPs 增加 | 并行优于串行 |
| w/o 不确定性损失 (用标准 MSE) | 下降 ~0.4dB | 纹理/边缘区域重建变差 |
| w/o 多尺度 U-shape | 下降 ~0.8dB | 多尺度特征很重要 |

### 关键发现
- FWSA 和 LWSA 的并行组合比串行堆叠效果更好且计算量更低，因为并行设计让两种信息在每层都充分交互
- 不确定性驱动损失在纹理丰富区域的 PSNR 提升最为显著（~0.8dB），验证了其自适应加权的有效性
- 可视化不确定性图发现，网络确实学到了将高权重分配给边缘和纹理区域——与直觉一致
- 在保持 SOTA 精度的同时，Specformer 的参数量和 FLOPs 均低于 CST 和 DAUHST 等竞争方法

## 亮点与洞察
- **并行空间-频率注意力**的设计很优雅——用频率域注意力替代 shifted window 或 cross-window attention 来建立全局连接，既保持了线性复杂度，又自然匹配了 HSI 光谱相似性先验。这个设计可以推广到任何多通道信号的处理（如多帧视频、多模态融合）
- **不确定性驱动损失**是一个通用的 trick，本质上让网络学习像素级的损失权重。这个思路可以直接迁移到超分辨率、去模糊等其他图像复原任务中——任何存在"困难区域"的问题都能受益
- 频率域自注意力中 FFT 天然提供全局感受野的洞察值得关注——这避免了转移到频域再做池化的信息损失

## 局限与展望
- 代码虽公开但尚未发布实际代码（GitHub 页面显示 "code is coming soon"），可复现性存疑
- 不确定性图的学习缺乏明确的监督信号，完全依赖损失函数的隐式学习，可能不够稳定
- 仅在 CASSI 系统上验证，对于其他光谱成像系统（如 CTIS、SCI）的适用性未验证
- 频率域注意力对输入分辨率敏感——不同分辨率的 FFT 频率分量含义不同，需要重新适应
- 未与最新的扩散模型-based HSI 重建方法对比

## 相关工作与启发
- **vs MST**: MST 使用 spectral-wise multi-head self-attention，Specformer 在此基础上加入了频率域分支，更充分地利用光谱间信息
- **vs CST**: CST 使用 cross-scale Transformer，侧重多尺度特征融合；Specformer 侧重空间-频率联合建模，两者切入角度不同
- **vs DAUHST**: DAUHST 是 deep unfolding + Transformer 的混合框架，计算量大；Specformer 更轻量
- 不确定性损失的思想来自 Kendall & Gal 的异方差不确定性，但本文首次将其应用于光谱成像重建

## 评分
- 新颖性: ⭐⭐⭐⭐ 并行空间-频率注意力 + 不确定性损失的组合是新颖的
- 实验充分度: ⭐⭐⭐⭐ 模拟和真实数据集均有验证，消融充分
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，模块设计逻辑自洽
- 价值: ⭐⭐⭐⭐ 在 HSI 重建领域推进了 SOTA，设计思路有广泛迁移价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] A Simple Low-bit Quantization Framework for Video Snapshot Compressive Imaging](a_simple_lowbit_quantization_framework_for_video_snapshot_co.md)
- [\[ECCV 2024\] Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning](token_compensator_altering_inference_cost_of_vision_transformer_without_re-tunin.md)
- [\[CVPR 2026\] When Lines Meet Textures: Spatial-Frequency Aligned Diffusion Features for Cross-Sparsity Correspondence](../../CVPR2026/model_compression/when_lines_meet_textures_spatial-frequency_aligned_diffusion_features_for_cross-.md)
- [\[CVPR 2025\] Sampling Innovation-Based Adaptive Compressive Sensing](../../CVPR2025/model_compression/sampling_innovation-based_adaptive_compressive_sensing.md)
- [\[ECCV 2024\] PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference](papr_training-free_one-step_patch_pruning_with_lightweight_convnets_for_faster_i.md)

</div>

<!-- RELATED:END -->
