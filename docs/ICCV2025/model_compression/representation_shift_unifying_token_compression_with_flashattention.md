---
title: >-
  [论文解读] Representation Shift: Unifying Token Compression with FlashAttention
description: >-
  [ICCV 2025][模型压缩][剪枝] 提出 Representation Shift，一种无需训练、模型无关的 token 重要性度量方法，通过计算 token 在网络层前后的表征变化量来衡量重要性，从而首次实现 token 压缩与 FlashAttention 的兼容，在视频理解和图像分类上取得高达 5.5× 的加速。
tags:
  - "ICCV 2025"
  - "模型压缩"
  - "剪枝"
  - "注意力机制"
  - "Representation Shift"
  - "Transformer"
  - "模型加速"
---

# Representation Shift: Unifying Token Compression with FlashAttention

**会议**: ICCV 2025  
**arXiv**: [2508.00367](https://arxiv.org/abs/2508.00367)  
**代码**: [https://github.com/mlvlab/Representation-Shift](https://github.com/mlvlab/Representation-Shift)  
**领域**: 信息检索  
**关键词**: Token Pruning, FlashAttention, Representation Shift, Vision Transformer, 模型加速

## 一句话总结

提出 Representation Shift，一种无需训练、模型无关的 token 重要性度量方法，通过计算 token 在网络层前后的表征变化量来衡量重要性，从而首次实现 token 压缩与 FlashAttention 的兼容，在视频理解和图像分类上取得高达 5.5× 的加速。

## 研究背景与动机

现有的 token 压缩方法主要依赖注意力图（attention map）来衡量 token 的重要性，例如利用 class token 的注意力分数或全局平均注意力分数。然而，FlashAttention 为了减少 HBM 内存访问，避免了注意力图的显式构建，导致这些方法无法在 FlashAttention 下使用。这形成了一个矛盾：FlashAttention 能带来显著加速（DeiT-S 上 1.5×，UMT-B 上 2.7×），但无法与训练无关的 token 压缩方法叠加。另外，部分可学习的 token 压缩方法需要额外训练，限制了通用性。

作者的核心动机是：能否找到一种不依赖注意力图、不需要额外训练的 token 重要性度量，使其同时兼容 FlashAttention 和多种架构（Transformer、CNN、SSM）？

## 方法详解

### 整体框架

该方法的核心思想极其简洁：在网络的某一层中，计算每个 token 经过该层变换前后的表征变化量（即 Representation Shift），将变化量小的 token 视为冗余并剪枝。由于只需要层的输入和输出，不依赖注意力图，因此天然与 FlashAttention 兼容。

### 关键设计

1. **Representation Shift 定义**: 给定输入 token $\mathbf{x} \in \mathbb{R}^{L \times C}$ 和某层变换 $F(\cdot)$，重要性分数定义为 $s = \Delta\mathbf{x} = \mathcal{D}(F(\mathbf{x}), \mathbf{x})$，其中 $\mathcal{D}$ 为距离度量。核心假设是：关键 token 倾向于有更高的表征变化，因为网络会强化其核心信息或抑制冗余信号，反之变化小的 token 可能与目标任务无关。

2. **操作选择（Operation Choice）**: 作者比较了三种计算位置：(i) 通过 Attention 层、(ii) 通过 MLP 层、(iii) 通过整个 Attention Block。实验发现 **MLP 层的 representation shift 效果最好**。原因是 Attention 层本身负责跨 token 的信息交换，其变换较为弥散；而 MLP 独立作用于每个 token，能产生更具区分性的表征变化，捕获 token 特定贡献。最终公式为 $\Delta\mathbf{x} = \|\text{MLP}(\text{LN}(\mathbf{x}')) - \mathbf{x}'\|_2$。

3. **距离度量选择（Distance Metric）**: 比较了 L2 范数、L1 范数和余弦距离。L2 距离在各层和模型上表现最稳定且鲁棒。余弦相似度在深层表现不佳，L1 虽然在第一层有优势但总体逊于 L2。因此默认采用 **L2 距离**作为度量。

### 损失函数 / 训练策略

该方法完全 **无需训练（training-free）**，也无需额外的可学习参数。token 剪枝在推理阶段直接应用：在指定层计算 representation shift 分数后，丢弃分数最低的 token。同时该方法也可与 vid-TLDR 等已有 token 合并方法组合，只需替换其重要性度量指标即可。

## 实验关键数据

### 主实验 (表格)

**视频-文本检索 (UMT, 7个基准)**

| 模型 | 方法 | 吞吐量 (vid/s) | 加速比 | MSRVTT R@1 | ActivityNet R@1 | DiDeMo R@1 |
|------|------|---------|--------|-----------|----------------|-----------|
| UMT-B | Base | 32 | - | 50.0 | 57.2 | 62.1 |
| UMT-B | Attn | 57 | 1.78× | 47.6 | 54.2 | 57.7 |
| UMT-B | **Ours** | **175** | **5.47×** | 48.0 | 50.3 | 56.9 |
| UMT-L | Base | 12 | - | 58.7 | 65.6 | 70.8 |
| UMT-L | Attn | 23 | 1.91× | 50.2 | 53.2 | 58.2 |
| UMT-L | **Ours** | **66** | **5.50×** | **56.5** | **62.9** | **67.3** |

**图像分类 (ImageNet1K, DeiT)**

| 模型 | 方法 | Acc-1 | 吞吐量 | GFLOPs |
|------|------|-------|--------|--------|
| DeiT-S | Base | 79.8 | 3002 | 4.6 |
| DeiT-S | Attn | 72.1 | 4844 | 3.0 |
| DeiT-S | **Ours** | **77.8** | **5948** | 3.0 |
| DeiT-B | Base | 81.8 | 1037 | 17.6 |
| DeiT-B | Attn | 76.9 | 2065 | 11.5 |
| DeiT-B | **Ours** | **79.6** | **2428** | 11.5 |

### 消融实验 (表格)

**操作选择与距离度量消融 (DeiT-S, ImageNet1K)**

| 操作位置 | 层 0 剪枝后 Acc | 层 4 剪枝后 Acc | 层 8 剪枝后 Acc |
|---------|----------------|----------------|----------------|
| Attention | ~79.5 | ~76.5 | ~73.0 |
| Entire Block | ~79.5 | ~77.5 | ~74.0 |
| **MLP** | **~79.5** | **~78.0** | **~75.0** |

| 距离度量 | 层 0 | 层 4 | 层 8 |
|---------|------|------|------|
| Cosine | ~79.5 | ~76.0 | ~72.0 |
| L1 | ~79.5 | ~77.5 | ~74.5 |
| **L2** | **~79.5** | **~78.0** | **~75.0** |

**CNN 与 SSM 上的泛化 (ImageNet1K)**

| 模型 | 方法 | Acc | 吞吐量 | GFLOPs |
|------|------|-----|--------|--------|
| ResNet-34 | Base | 73.2 | 5811 | 3.7 |
| ResNet-34 | Line-wise | 72.8 | 7112 | 2.5 |
| ResNet-50 | Base | 76.1 | 2927 | 4.1 |
| ResNet-50 | Line-wise | **76.4** | 3553 | 2.7 |
| ViM-T | Base | 76.1 | 1603 | 1.5 |
| ViM-T | Ours | 75.5 | 1754 | 1.3 |

### 关键发现

- 在 UMT-L 上，Representation Shift 相比 attention-based 剪枝在 R@1 上平均提升 7.2%，且吞吐量比后者高约 2.8×。
- UMT-L + Representation Shift（66 vid/s）比 UMT-B baseline（32 vid/s）更快且性能更优，说明"大模型+token 压缩"优于"直接用小模型"。
- 首次将 token 压缩扩展到 CNN（ResNet）和 SSM（ViM），在 ResNet-50 上 line-wise 剪枝甚至提升了准确率（76.1→76.4）。

## 亮点与洞察

- **极致简洁**: 核心思想仅一个公式：计算 MLP 前后的 L2 距离。无需额外参数、无需训练、无需注意力图，开销可忽略不计。
- **首次统一 FlashAttention + Token 压缩**: 解决了此前两大加速路线互不兼容的问题，叠加后加速效果倍增。
- **模型无关性**: 不仅适用于 ViT，还可泛化到 CNN 和 SSM，是一种通用的 token 重要性度量。
- **可视化直觉好**: 在 DeiT 中间层可视化 representation shift 时，能够自动捕获前景物体，类似显著性检测。

## 局限与展望

- 在视频任务中 UMT-B 的部分基准（MSVD、ActivityNet）上有一定性能下降，说明剪枝比例和层选择需要根据任务调整。
- CNN 上的 token 剪枝由于卷积需要 2D 网格结构，只能采用行列级剪枝，灵活性不如 ViT。
- 未探索 representation shift 与 token merging（如 ToMe）的更深入结合。
- 对多模态大模型（如 LLaVA）的适用性尚未验证。

## 相关工作与启发

- 与 EViT、BAT 等 attention-based 方法相比，representation shift 不依赖注意力图，适用范围更广。
- 与 vid-TLDR 结合使用，替换其重要性度量后可同时享受 token merging + FlashAttention 的加速。
- 为未来大模型推理加速提供了新思路：可考虑在 LLM 中用类似方法评估 KV cache 中的 token 重要性。

## 评分

- **新颖性**: ⭐⭐⭐⭐ 想法极简但有效，首次实现 FlashAttention + token 压缩的统一
- **实验充分度**: ⭐⭐⭐⭐⭐ 涵盖视频检索、视频QA、图像分类，验证了 ViT/CNN/SSM 三种架构
- **写作质量**: ⭐⭐⭐⭐ 结构清晰，消融充分，图表丰富
- **价值**: ⭐⭐⭐⭐⭐ 实用性极强，几行代码即可应用于已有模型

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] UniICL: An Efficient ICL Framework Unifying Compression, Selection, and Generation](../../ACL2025/model_compression/uniicl_icl_framework.md)
- [\[NeurIPS 2025\] VQToken: Neural Discrete Token Representation Learning for Extreme Token Reduction in Video Large Language Models](../../NeurIPS2025/model_compression/vqtoken_neural_discrete_token_representation_learning_for_extreme_token_reductio.md)
- [\[ICCV 2025\] TR-PTS: Task-Relevant Parameter and Token Selection for Efficient Tuning](tr-pts_task-relevant_parameter_and_token_selection_for_efficient_tuning.md)
- [\[ICCV 2025\] FastVAR: Linear Visual Autoregressive Modeling via Cached Token Pruning](fastvar_linear_visual_autoregressive_modeling_via_cached_token_pruning.md)
- [\[ACL 2025\] UniQuanF: Unifying Uniform and Binary-coding Quantization for Accurate Compression of Large Language Models](../../ACL2025/model_compression/uniquanf_unified_quantization.md)

</div>

<!-- RELATED:END -->
