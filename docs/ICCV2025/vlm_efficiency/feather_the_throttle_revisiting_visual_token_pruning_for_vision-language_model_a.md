---
title: >-
  [论文解读] Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration
description: >-
  [ICCV 2025][多模态VLM][VLM加速] 揭示了 VLM 中早期视觉 token 剪枝存在系统性位置偏差（RoPE 导致倾向保留图像底部 token），并提出 FEATHER 方法通过去除 RoPE + 均匀采样 + 多阶段剪枝解决该问题，在定位任务上实现 5× 以上性能提升。 近年来 VLM（如 LLaVA）通…
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "VLM加速"
  - "剪枝"
  - "RoPE位置偏差"
  - "视觉定位"
  - "FEATHER"
---

# Feather the Throttle: Revisiting Visual Token Pruning for Vision-Language Model Acceleration

**会议**: ICCV 2025  
**arXiv**: [2412.13180](https://arxiv.org/abs/2412.13180)  
**代码**: 无  
**领域**: Multimodal VLM / Token Pruning  
**关键词**: VLM加速, Visual Token Pruning, RoPE位置偏差, 视觉定位, FEATHER

## 一句话总结

揭示了 VLM 中早期视觉 token 剪枝存在系统性位置偏差（RoPE 导致倾向保留图像底部 token），并提出 FEATHER 方法通过去除 RoPE + 均匀采样 + 多阶段剪枝解决该问题，在定位任务上实现 5× 以上性能提升。

## 研究背景与动机

近年来 VLM（如 LLaVA）通过将视觉 patch 投影为 LLM 输入 token 实现多模态理解，但大量视觉 token 带来巨大计算开销。以 FastV 为代表的方法在 LLM 浅层后剪枝 50% 视觉 token，声称性能几乎无损。

然而作者发现一个关键问题：虽然剪枝后在多数 VQA benchmark 上表现良好，但在**视觉密集型任务（特别是定位）上性能灾难性下降**——剪枝 75% token 后定位性能下降 88-91%。进一步分析发现这不是剪枝本身的问题，而是**剪枝标准的根本缺陷**：由于 RoPE 的长距离衰减特性，早期层的注意力分数系统性地偏向图像底部的 token，导致上方视觉 token 被错误移除。更令人震惊的是，大多数 benchmark 即使用这种有缺陷的剪枝仍保持高性能，暴露了**现有 benchmark 无法有效评估细粒度视觉能力**的问题。

## 方法详解

### 整体框架

FEATHER（Fast and Effective Acceleration wiTH Ensemble cRiteria）采用两阶段剪枝策略：第一阶段在第 8 层使用去 RoPE 注意力 + 均匀采样的混合标准进行温和剪枝；第二阶段在第 16 层使用去 RoPE 注意力进行激进剪枝。类似赛车手在弯道入口轻踩油门、过弯心后加速的策略。

### 关键设计

1. **去 RoPE 注意力标准 $\phi_{-R}$**: 在计算 token 重要性时去除 RoPE 的位置编码，只计算纯内容级别的注意力分数。这消除了 RoPE 长距离衰减导致的位置偏差，使注意力能真正反映 token 与文本指令的语义相关性。该计算仅在剪枝时执行一次，与 FlashAttention 兼容。在 K=3 时，去 RoPE 使定位性能平均提升 183%。

2. **均匀采样 $\phi_{\text{uniform}}$**: 以固定步长均匀采样视觉 token，确保图像各区域都有覆盖。单独使用时牺牲了对特定区域的关注度，但与注意力标准组合后效果互补。在 $\phi_{-R} + \phi_{\text{uniform}}$ 中，以 stride=3 采样少量均匀 token，其余用去 RoPE 注意力选择，兼顾覆盖率和重要性。

3. **多阶段渐进剪枝**: 不同于 FastV 在第 3 层一次性剪枝，FEATHER 分两阶段：

    - K=8：使用 $\phi_{-R} + \phi_{\text{uniform}}$ 保留 $(1-R)\%$ token
    - K=16：使用 $\phi_{-R}$ 进一步保留 $(1-R)^2\%$ token
    - 动机是注意力标准在更深层更准确，深层可更激进剪枝。最终第 16 层后仅保留 3.3% 的视觉 token。

### 损失函数 / 训练策略

完全 **training-free**，无需修改模型或重新训练。VLM 使用 SigLIP ViT-SO400M 作为视觉编码器，单层 MLP 适配器，LLaMA 2 7B 作为 LLM，在多模态指令微调数据上训练（视觉编码器冻结）。

## 实验关键数据

### 主实验 (表格)

**FEATHER vs FastV vs PyramidDrop (64-68% FLOPS 减少)**

| 方法 | FLOPS减少 | 定位Avg | OCID-Ref | RefCOCO | TextVQA | VQAv2 | POPE |
|------|----------|---------|----------|---------|---------|-------|------|
| Baseline (无剪枝) | 0% | 53.1 | 42.5 | 56.1 | 56.5 | 79.2 | 86.3 |
| FastV (K=3) | 68% | 5.9 | 5.7 | 6.7 | 31.8 | 72.7 | 83.2 |
| PyramidDrop | 65% | ~22 | - | - | ~48 | ~77 | ~86 |
| **FEATHER** | **64%** | **35.6** | **32.0** | **38.8** | **51.7** | **78.1** | **87.4** |

FEATHER 在定位上比 FastV **提升超过 5 倍**（5.9→35.6），比 PyramidDrop 提升 36%。

### 消融实验 (表格)

**不同剪枝标准对比 (K=3, R=0.75)**

| 标准 | FLOPS减少 | 定位Avg | TextVQA | VQAv2 | POPE |
|------|----------|---------|---------|-------|------|
| $\phi_{\text{original}}$ (FastV) | 68% | 5.9 | 31.8 | 72.7 | 83.2 |
| $\phi_{-R}$ (去 RoPE) | 68% | 16.7 | 41.6 | 76.0 | 85.2 |
| $\phi_{\text{uniform}}$ | 66% | 28.0 | 41.4 | 75.9 | 85.2 |
| $\phi_{\text{KNN}}$ | 66% | 23.9 | 39.9 | 74.4 | 81.2 |
| $\phi_{-R} + \phi_{\text{uniform}}$ | 61% | **27.2** | **46.6** | **77.4** | **86.0** |

**不同剪枝标准对比 (K=8, R=0.75)**

| 标准 | 定位Avg | TextVQA | VQAv2 |
|------|---------|---------|-------|
| $\phi_{\text{original}}$ | 23.3 | 45.0 | 76.1 |
| $\phi_{-R}$ | 27.3 | 49.0 | 77.4 |
| $\phi_{-R} + \phi_{\text{uniform}}$ | **35.6** | **51.7** | **78.1** |

### 关键发现

- **RoPE 是罪魁祸首**: RoPE 的长距离衰减使浅层注意力系统性偏向底部 token。Chi-Square 检验确认 token 选择非均匀（p < 0.05），R=0.75 时平均选中 token 位置在图像 80.7% 处。
- **大多数 benchmark 不需要精细视觉理解**: 即使完全移除剪枝 token 不做信息传递（无 LLM 整合），多数 benchmark 性能几乎不变，说明它们主要依赖语言偏差而非视觉理解。
- **深层标准更准确**: 在更深层应用注意力标准时，选中 token 精确集中在与文本描述相关的区域。
- **仅保留 3.3% 视觉 token** 仍可在定位上仅下降 26%，说明关键在于选对 token 而非数量。

## 亮点与洞察

- **深刻的诊断性分析**: 不仅提出方法，更系统性揭示了 FastV 类方法失败的根本原因（RoPE 位置偏差）以及 VLM benchmark 的系统性缺陷。
- **赛车手类比精妙**: "feather the throttle" 直觉地解释了分阶段剪枝的策略——弯道入口温和，过弯心后激进。
- **暴露 benchmark 盲点**: 发现多数 VLM 评测无法区分"看到了"和"理解了"，是对整个领域的重要警示。

## 局限与展望

- 实验仅使用 LLaMA 2 7B 的单一 VLM，未验证在更大或更新模型（如 LLaVA-NeXT、Qwen-VL）上的效果。
- 去除 RoPE 虽消除了位置偏差，但可能引入对注意力权重的非预期影响，更鲁棒的位置编码方案值得探索。
- 剪枝层和比例需要手工设定（K=8, K=16, R=0.75），未实现自适应策略。
- 定位任务虽大幅提升但仍与 baseline 有 33% 差距，说明视觉 token 剪枝对空间精确任务仍有天然劣势。

## 相关工作与启发

- FastV 的"浅层剪枝几乎无损"结论被证明是因为 benchmark 不够挑战性，需谨慎看待此类声明。
- 与 PruMerge、VisionZip 等在 ViT 端的压缩方法互补，FEATHER 在 LLM 端压缩，两者可结合。
- RoPE 的位置偏差问题在 NLP 领域也有讨论（浅层更依赖短距离信息），本文首次在多模态场景中暴露此问题。
- 启发：未来 VLM 加速应专门在定位等视觉密集任务上评估，而非仅看通用 benchmark。

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 深入诊断发现 RoPE 偏差问题，是对 VLM 加速领域的重要洞见
- **实验充分度**: ⭐⭐⭐⭐ 12 个 benchmark 全面评估，消融充分，但仅测一个 VLM
- **写作质量**: ⭐⭐⭐⭐⭐ 故事线清晰、层层递进，从发现问题到分析原因到解决方案
- **价值**: ⭐⭐⭐⭐ 对 VLM 加速和 benchmark 设计都有重要启示

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[ICML 2025\] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models](../../ICML2025/multimodal_vlm/corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun.md)
- [\[ICML 2025\] SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference](../../ICML2025/multimodal_vlm/sparsevlm_visual_token_sparsification_for_efficient_vision-language_model_infere.md)
- [\[CVPR 2026\] TransPrune: Token Transition Pruning for Efficient Large Vision-Language Model](../../CVPR2026/multimodal_vlm/transprune_token_transition_pruning_for_efficient_large_vision-language_model.md)
- [\[CVPR 2025\] Revisiting Model Stitching in the Foundation Model Era](../../CVPR2025/multimodal_vlm/revisiting_model_stitching_in_the_foundation_model_era.md)

</div>

<!-- RELATED:END -->
