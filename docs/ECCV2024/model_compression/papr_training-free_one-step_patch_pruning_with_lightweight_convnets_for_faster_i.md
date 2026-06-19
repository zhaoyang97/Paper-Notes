---
title: >-
  [论文解读] PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference
description: >-
  [ECCV 2024][模型压缩][剪枝] 提出 PaPr，利用轻量级 ConvNet 的卷积特征图生成 Patch Significance Map (PSM)，在**无需重训练**的情况下对 ViT/ConvNet/混合架构进行**一步式** patch 剪枝，实现显著的计算量削减（视频场景最高 3.7× FLOPs 减少），且精度损失极小。
tags:
  - "ECCV 2024"
  - "模型压缩"
  - "剪枝"
  - "Transformer"
  - "ConvNet"
  - "Token Reduction"
  - "推理加速"
---

# PaPr: Training-Free One-Step Patch Pruning with Lightweight ConvNets for Faster Inference

**会议**: ECCV 2024  
**arXiv**: [2403.16020](https://arxiv.org/abs/2403.16020)  
**代码**: [GitHub](https://github.com/tanvir-utexas/PaPr)  
**领域**: 模型压缩  
**关键词**: Patch Pruning, Vision Transformer, ConvNet, Token Reduction, 推理加速

## 一句话总结

提出 PaPr，利用轻量级 ConvNet 的卷积特征图生成 Patch Significance Map (PSM)，在**无需重训练**的情况下对 ViT/ConvNet/混合架构进行**一步式** patch 剪枝，实现显著的计算量削减（视频场景最高 3.7× FLOPs 减少），且精度损失极小。

## 研究背景与动机

随着视觉模型从 ConvNet 演进到 ViT，模型计算量持续攀升。高分辨率图像中存在大量冗余的背景 patch，如何高效剔除这些冗余 patch 是提升推理速度的关键。

现有 patch 剪枝方法存在三大痛点：

**需要额外训练**：大多数方法（DynamicViT、A-ViT、SP-ViT）需要在中间层训练额外的 mask 预测模块，随着基线模型不断更新，重新训练成本高昂

**多步渐进式剪枝**：在网络的多个中间层逐步减少 token，导致早期层仍执行冗余计算，特别是对深层模型不利

**架构绑定**：许多方法依赖特定架构特征（如 class token、attention map），无法推广到 ConvNet 或混合模型

PaPr 的核心洞察在于：**轻量级 ConvNet 虽然 top-1 精度低，但其卷积层具有出色的判别性区域定位能力**。实验发现，当 top-k 中 $k$ 增大时，轻量 ConvNet 与大型 ViT 的精度差距大幅缩小，说明浅层模型已经能准确锁定图像中的判别性区域。全连接层才是 ConvNet 精细分类的瓶颈，而非卷积层的定位能力。

## 方法详解

### 整体框架

PaPr 采用"轻量 ConvNet 提议 + 大模型推理"的两阶段范式。先用超轻量 ConvNet（如 MobileOne-S0，仅 0.27 GFLOPs）提取 Patch Significance Map (PSM)，然后在 patch 提取阶段之后立即剪除低显著性 patch，大模型仅对保留的判别性 patch 进行后续计算。

### 关键设计

#### 1. FC 层权重均匀化（Weight Recalibration）

**功能**：从 ConvNet 最后一个卷积层提取判别性区域提议。

**核心思路**：传统 CAM 方法依赖类激活权重 $w_c^k$ 来加权特征图，但这受限于模型分类精度。PaPr 将 FC 层权重替换为均匀权重 $w_c^k = \frac{1}{KC}$，消除 FC 层对区域定位的干扰：

$$\mathcal{R}(x,y) = \frac{1}{K}\sum_{k} f_k(x,y)$$

其中 $f_k(x,y)$ 是第 $k$ 个卷积核在位置 $(x,y)$ 的特征响应。这样得到的判别区域提议 $\mathcal{R}$ 仅依赖卷积层的空间感知能力，与模型分类精度无关。

**设计动机**：轻量 ConvNet 的卷积层本身就擅长定位判别性区域（top-10 精度与大模型接近），FC 层是分类瓶颈。通过抑制 FC 层，可以从极小的 ConvNet 中获得高质量的区域定位。

#### 2. Patch Significance Map (PSM) 生成与剪枝

**功能**：将低分辨率的区域提议映射到目标特征图分辨率，生成 patch 级别的显著性评分。

**核心思路**：对 $\mathcal{R} \in \mathbb{R}^{h \times w}$ 进行双三次上采样到目标尺寸 $(h', w')$，得到 PSM $\mathcal{P}$，然后通过排序得到剪枝 mask：

$$\mathcal{M} = \text{reshape}(\text{argsort}(\text{flatten}(\mathcal{P})))$$

保留 top-$z\%$ 的 patch，其余直接丢弃。

**设计动机**：ConvNet 天然保持 patch 的位置信息（归纳偏置），因此 PSM 可以直接与目标模型的 patch 建立空间对应关系，无需额外学习。

#### 3. 适配不同架构

**ViT 集成**：在 patch 提取和位置编码之后、进入 transformer block 之前，直接根据 PSM 删除低显著性 token。所有后续 transformer block 仅处理保留的 token。

**层级模型集成（ConvNet/Swin）**：层级模型使用窗口操作，无法直接删除 patch。PaPr 针对**像素算子**（$1 \times 1$ 卷积/线性层，占 ConvNext 96.2%、Swin 63.3% 计算量）进行选择性计算：

- 根据 PSM 将特征分为前景 $A_f$ 和背景 $A_b$
- 仅对 $A_f$ 执行线性运算，$A_b$ 用零填充
- 重组后传入下一个窗口算子

### 训练策略

PaPr **完全不需要训练**。提议 ConvNet 使用预训练权重，大模型使用原始预训练权重，整个流程即插即用。PaPr 也可在训练阶段使用（类似于 masked training），实现 1.6× 训练加速。

## 实验关键数据

### 主实验

ImageNet-1k 上使用 MAE 预训练 ViT 的性能对比：

| 模型 | 方法 | Top-1 Acc (%) | GFLOPs | 吞吐 (img/s) |
|------|------|:---:|:---:|:---:|
| ViT-B-16 | Baseline | 83.74 | 17.59 | 307 |
| ViT-B-16 | ToMe | 78.82 | 8.78 | 615 |
| ViT-B-16 | TokenFusion | 79.23 | 8.78 | 618 |
| ViT-B-16 | **PaPr (z=0.5)** | **82.40** | 8.98 | 605 |
| ViT-L-16 | Baseline | 85.95 | 61.61 | 91 |
| ViT-L-16 | ToMe | 84.24 | 30.99 | 180 |
| ViT-L-16 | **PaPr (z=0.5)** | **85.06** | 30.83 | 183 |
| ViT-H-16 | Baseline | 86.89 | 167.4 | 36 |
| ViT-H-16 | ToMe | 85.48 | 82.53 | 72 |
| ViT-H-16 | **PaPr (z=0.5)** | **86.40** | 83.04 | 71 |

PaPr 在 MAE 模型上优势尤为明显（ViT-B 上比 ToMe 高 **4.5%** 精度），因为 MAE 预训练本身就使用 mask，天然适配 masked inference。

### 消融实验

不同 ConvNet 提议模型对最终精度的影响（z=0.5）：

| 提议模型 | GFLOPs | ViT-B Acc (%) | ViT-L Acc (%) |
|----------|:---:|:---:|:---:|
| MobileOne-S0 | 0.27 | 82.24 | 84.06 |
| MobileOne-S2 | 1.35 | 82.28 | 84.16 |
| ResNet-18 | 1.81 | 81.10 | 83.84 |
| ResNet-50 | 4.09 | 82.33 | 84.09 |
| ResNet-152 | 11.51 | 82.51 | 84.08 |

不同 ConvNet 的精度差异仅 **0.3%**（ViT-B），说明 PSM 具有极强的模型无关性。MobileOne-S0 的计算量仅为 ResNet-152 的 $1/42$，但定位精度几乎一致。

视频识别消融（Kinetics-400, ViT-B MAE）：

| 方法 | Top-1 Acc (%) | GFLOPs |
|------|:---:|:---:|
| Baseline | 81.21 | 180 |
| PaPr (z=0.45) | 81.18 | 76 |
| PaPr (z=0.35) | 80.15 | 59 |

视频场景下约 70% patch 被剪除，FLOPs 降至原始的 1/3，精度损失仅 0.8%。

### 关键发现

1. PaPr 与 ToMe 结合可达到 **Pareto 最优**，在 ViT-B AugReg 上比单独 ToMe 提升 4.5% 精度
2. 在 ConvNext、Swin 等层级模型上同样有效，且无需训练即可跟随模型更新
3. 视频中 PaPr 能通过时空理解抑制冗余帧的背景 patch，实现高达 3.7× 加速

## 亮点与洞察

- **核心洞察极为简洁**：轻量 ConvNet 在定位上并不差，差在分类。通过抑制 FC 层释放卷积层的定位潜力
- **实用性极强**：无需训练、架构无关、支持 batch 处理，可直接应用于 ViT/ConvNet/混合模型
- **与 MAE 天然契合**：MAE 预训练就是 mask 图像，PaPr 提供了"有意义的 mask"替代随机 mask

## 局限与展望

1. 仅验证了判别性任务（分类、视频识别），密集预测任务（检测/分割）尚未探索
2. PSM 是静态的（每张图一次计算），未利用大模型中间层的动态信息
3. 对于前景/背景分布不均的数据集（如医学图像），固定比例 $z$ 可能不是最优

## 相关工作与启发

- **ToMe (ICLR 2023)**：token 合并，PaPr 可作为前置步骤增强其效果
- **DynamicViT (NeurIPS 2021)**：训练额外 predictor 进行 token 稀疏化，需重训练
- **CAM 方法**：依赖梯度和最终预测，不支持 batch 处理；PaPr 使用简单均值替代

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "轻量 ConvNet 定位 + 大模型推理"的解耦思路新颖且直觉上合理
- **实验充分度**: ⭐⭐⭐⭐⭐ — 覆盖 ViT/ConvNet/Swin 多种架构、图像/视频双任务、多种预训练方式
- **写作质量**: ⭐⭐⭐⭐ — 逻辑清晰、图表丰富、PyTorch 伪代码实用
- **价值**: ⭐⭐⭐⭐⭐ — 真正的即插即用方法，对工业界部署价值很高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] SpaceJAM: a Lightweight and Regularization-free Method for Fast Joint Alignment of Images](spacejam_a_lightweight_and_regularization-free_method_for_fast_joint_alignment_o.md)
- [\[NeurIPS 2025\] One-Step Diffusion-Based Image Compression with Semantic Distillation](../../NeurIPS2025/model_compression/one-step_diffusion-based_image_compression_with_semantic_distillation.md)
- [\[ECCV 2024\] Isomorphic Pruning for Vision Models](isomorphic_pruning_for_vision_models.md)
- [\[ECCV 2024\] Auto-DAS: Automated Proxy Discovery for Training-free Distillation-aware Architecture Search](auto-das_automated_proxy_discovery_for_training-free_distillation-aware_architec.md)
- [\[CVPR 2026\] Critical Patch-Aware Sparse Prompting with Decoupled Training for Continual Learning on the Edge](../../CVPR2026/model_compression/critical_patch-aware_sparse_prompting_with_decoupled_training_for_continual_lear.md)

</div>

<!-- RELATED:END -->
