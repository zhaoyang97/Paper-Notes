---
title: >-
  [论文解读] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers
description: >-
  [CVPR 2025][模型压缩][Transformer] 提出 Q-DiT，针对 Diffusion Transformer (DiT) 的后训练量化方法，通过进化搜索自动分配量化组大小和样本级动态激活量化，在 W4A8 设置下实现高保真图像/视频生成。 扩散模型的架构已从 UNet 演进到 Diffusion Tran…
tags:
  - "CVPR 2025"
  - "模型压缩"
  - "Transformer"
  - "后训练量化"
  - "动态激活量化"
  - "组粒度分配"
  - "DiT加速"
---

# Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers

**会议**: CVPR 2025  
**arXiv**: [2406.17343](https://arxiv.org/abs/2406.17343)  
**代码**: [GitHub](https://github.com/Juanerx/Q-DiT)  
**领域**: 图像生成 / 模型压缩 (Image Generation / Model Compression)  
**关键词**: 扩散Transformer量化, 后训练量化, 动态激活量化, 组粒度分配, DiT加速

## 一句话总结

提出 Q-DiT，针对 Diffusion Transformer (DiT) 的后训练量化方法，通过进化搜索自动分配量化组大小和样本级动态激活量化，在 W4A8 设置下实现高保真图像/视频生成。

## 研究背景与动机

扩散模型的架构已从 UNet 演进到 Diffusion Transformer (DiT)，如 Stable Diffusion 3 和 Sora，显著提升了生成质量和可扩展性。然而，DiT 模型规模庞大，迭代去噪过程计算密集，限制了实时部署。

后训练量化（PTQ）是一种不需要重训练的模型压缩方案。但现有 PTQ 方法（PTQ4DM、Q-diffusion、PTQD 等）主要针对 UNet 架构设计，直接应用于 DiT 会导致显著性能下降。这是因为 DiT 具有两个独特特性：（1）**输入通道维度上的显著方差**——权重和激活在输入通道间的分布差异远大于输出通道，且特定通道存在异常值；（2）**时间步间的激活分布漂移**——不同去噪时间步的激活分布变化剧烈，且不同样本间的变化模式也不同。

现有的基于重建的 PTQ 方法难以同时处理 transformer 架构特性和去噪动态过程。Q-DiT 通过自动量化粒度分配和样本级动态量化同时解决这两个挑战。

## 方法详解

### 整体框架

Q-DiT 对 DiT 模型的每一层进行组量化（group quantization），组大小由进化搜索自动分配，且激活在运行时动态量化。权重和激活使用均匀量化 $\hat{\mathbf{x}} = s \cdot (\text{clip}(\lfloor \frac{\mathbf{x}}{s}\rceil + Z, 0, 2^b - 1) - Z)$。

### 关键设计

1. **自动量化粒度分配（Automatic Quantization Granularity Allocation）**:
    - 功能：为每层自动确定最优的量化组大小
    - 核心思路：将矩阵按输入通道分为大小为 $g_{ll}$ 的组，每组独立量化。关键发现：组大小与性能**不是单调关系**——减小组大小（增加组数）不一定更好（如从 128 减到 96，FID 反而从 17.87 升到 19.97）。因此使用进化算法搜索逐层组大小配置 $\mathbf{g}^* = \arg\min_\mathbf{g} \text{FID}(R, G_\mathbf{g})$，约束为 BitOps 不超过预设阈值 $N_{bitops}$。直接用 FID/FVD 作为搜索指标而非逐层 MSE
    - 设计动机：固定组大小是次优的；逐层 MSE 与最终生成质量不一定相关（非单调性）；进化搜索可以直接优化最终生成指标

2. **样本级动态激活量化（Sample-wise Dynamic Activation Quantization）**:
    - 功能：自适应处理不同时间步和不同样本间的激活分布变化
    - 核心思路：不使用校准集预计算固定的激活量化参数，而是在运行时根据每个样本的实际激活分布动态计算量化参数（缩放因子和零点）。对于每个线性层的输入激活，在组级别上计算 min/max 并即时确定量化参数。计算开销极小——仅需额外的 min/max 统计
    - 设计动机：在固定时间步校准的量化参数无法泛化到所有时间步；DiT 的激活在时间维度和样本维度都有显著变化，静态量化必然引入大误差

3. **与 DiT 特性分析的结合**:
    - 功能：通过对 DiT 模型特性的深入分析指导量化策略设计
    - 核心思路：（i）观察到 DiT 权重和激活在输入通道方向的方差远大于输出通道，证明了沿输入通道分组量化的必要性；（ii）观察到激活的标准差在不同 block 和时间步间变化剧烈（尤其是 MLP 层），证明了动态量化的必要性；（iii）发现异常值集中在特定通道，组量化可以在组级别处理异常值
    - 设计动机：data-driven 的方法设计——先观察问题，再针对性设计方案

### 损失函数 / 训练策略

- **无需训练**: PTQ 方法，不需要重训练或微调
- **进化搜索**: 种群初始化 → 变异/交叉 → FID/FVD 评估 → 选择 → 迭代
- **搜索空间**: 组大小从 {32, 64, 96, 128, ...} 中选择
- **量化配置**: W6A8（权重 6bit，激活 8bit）和 W4A8（权重 4bit，激活 8bit）

## 实验关键数据

### 主实验

DiT-XL/2 在 ImageNet 256×256（W4A8 / W6A8）：

| 方法 | W/A | FID↓ | sFID↓ | IS↑ | 说明 |
|------|-----|------|-------|-----|------|
| Full Precision | 32/32 | 9.62 | 6.17 | 278.24 | 基线 |
| Q-DiT (W6A8) | 6/8 | ~10 | ~6.5 | ~275 | 接近无损 |
| **Q-DiT (W4A8)** | **4/8** | **较低** | — | — | 最低FID |
| PTQ4DiT (W4A8) | 4/8 | +1.09 | — | — | Q-DiT 降低 1.09 |

组大小非单调性验证（ImageNet 256×256, W4A8）：

| 组大小 | FID↓ | sFID↓ |
|--------|------|-------|
| 128 | 17.87 | 20.45 |
| 96 | 19.97 | 21.42 |

### 消融实验

| 配置 | 说明 |
|------|------|
| 固定组大小 (128) | 基线，FID 较高 |
| 自动组大小分配 | FID 显著降低 |
| 静态激活量化 | 时间步变化导致量化误差大 |
| 动态激活量化 | 显著改善，尤其对高变化时间步 |
| 两者结合 | 最佳效果 |

### 关键发现

- 在 W6A8 下实现几乎无损压缩，W4A8 下仍保持高保真生成
- 组大小的非单调性是 DiT 量化独有的现象，区别于 LLM/ViT 量化
- 进化搜索直接优化 FID 比基于逐层 MSE 的方法更有效
- 样本级动态量化的额外开销极小但效果显著
- 方法同时适用于图像（ImageNet）和视频（VBench）生成

## 亮点与洞察

- **对 DiT 独特量化特性的深入分析**：首次系统地指出 DiT 在输入通道方差和时间步激活漂移上与 UNet 的根本差异，为后续 DiT 量化研究提供了观察基础
- **组大小非单调性的发现**极具启发性：打破了"更细粒度量化一定更好"的直觉，说明 DiT 中存在复杂的量化行为需要数据驱动的搜索而非人工启发式

## 局限与展望

- 进化搜索需要多次生成采样计算 FID，搜索成本较高
- 目前仅验证了 DiT-XL/2 等有限模型，对更大规模 DiT 的效果待验证
- 动态量化在部署时需要运行时计算 min/max，可能影响延迟
- 未探索与权重剪枝、知识蒸馏等正交压缩技术的结合

## 相关工作与启发

- **vs PTQ4DiT**: PTQ4DiT 使用通道显著性平衡和 Spearman 校准，但未考虑组大小非单调性和样本级动态；Q-DiT 在 W4A8 下 FID 降低 1.09
- **vs Q-diffusion**: Q-diffusion 针对 UNet 设计，使用逐层重建方法，难以扩展到大模型；Q-DiT 不依赖重建且直接优化生成指标
- **vs LLM 量化 (AWQ, GPTQ)**: LLM 量化方法假设组大小单调对应性能，不适用于 DiT 的非单调行为

## 评分

- 新颖性: ⭐⭐⭐⭐ DiT 量化特性分析深入，自动组分配和动态量化设计针对性强
- 实验充分度: ⭐⭐⭐⭐ ImageNet 和 VBench 双验证，消融全面
- 写作质量: ⭐⭐⭐⭐ 从观察到方法的论证清晰
- 价值: ⭐⭐⭐⭐ 为 DiT 部署提供了实用的量化方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[CVPR 2025\] QuartDepth: Post-Training Quantization for Real-Time Depth Estimation on the Edge](quartdepth_post-training_quantization_for_real-time_depth_estimation_on_the_edge.md)
- [\[ECCV 2024\] AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer](../../ECCV2024/model_compression/adalog_post-training_quantization_for_vision_transformers_with_adaptive_logarith.md)
- [\[NeurIPS 2025\] Quantization Error Propagation: Revisiting Layer-Wise Post-Training Quantization](../../NeurIPS2025/model_compression/quantization_error_propagation_revisiting_layer-wise_post-training_quantization.md)
- [\[CVPR 2025\] Style Quantization for Data-Efficient GAN Training](style_quantization_for_data-efficient_gan_training.md)

</div>

<!-- RELATED:END -->
