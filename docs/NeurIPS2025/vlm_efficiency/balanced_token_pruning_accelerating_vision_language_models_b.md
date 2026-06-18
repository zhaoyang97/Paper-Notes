---
title: >-
  [论文解读] Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization
description: >-
  [NeurIPS 2025][多模态VLM][视觉token剪枝] 提出 Balanced Token Pruning (BTP)，通过联合考虑剪枝对当前层（局部）和后续层（全局）的影响，在浅层侧重多样性保留以维护下游表示质量、在深层侧重注意力选择以保持局部输出一致性，在 LLaVA/Qwen2.5-VL 等多个 LVLM 上仅保留 22% 视觉 token 即保持原模型 98% 性能。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "视觉token剪枝"
  - "局部-全局优化"
  - "注意力剪枝"
  - "多样性剪枝"
  - "LVLM推理加速"
---

# Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2505.22038](https://arxiv.org/abs/2505.22038)  
**代码**: [https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning](https://github.com/EmbodiedCity/NeurIPS2025-Balanced-Token-Pruning)  
**领域**: 多模态VLM / 模型加速  
**关键词**: 视觉token剪枝, 局部-全局优化, 注意力剪枝, 多样性剪枝, LVLM推理加速

## 一句话总结

提出 Balanced Token Pruning (BTP)，通过联合考虑剪枝对当前层（局部）和后续层（全局）的影响，在浅层侧重多样性保留以维护下游表示质量、在深层侧重注意力选择以保持局部输出一致性，在 LLaVA/Qwen2.5-VL 等多个 LVLM 上仅保留 22% 视觉 token 即保持原模型 98% 性能。

## 研究背景与动机

**领域现状**：LVLM 通过视觉编码器将图像转化为大量 token（如 LLaVA-1.5 为 576 个，LLaVA-NeXT 高达 2880 个），视觉 token 数量是推理效率的主要瓶颈，特别是在边缘设备部署场景中。视觉 token 剪枝是目前主流的加速策略。

**现有痛点**：现有剪枝方法分为两类——基于注意力的方法（FastV、PyramidDrop）根据文本对图像 token 的注意力分数选择重要 token，基于多样性的方法（DivPrune）最大化保留 token 的语义多样性。但这两类方法都存在盲点：注意力方法只优化当前层的输出一致性（局部最优），忽略了剪枝对后续层的级联影响——不同层关注的图像区域不同，当前层不重要的 token 可能在深层很关键；多样性方法虽然更好地保留了各层所需信息，但在局部层面无法维持输出一致性。

**核心矛盾**：通过可视化实验，作者发现一个关键现象：注意力剪枝在剪枝层的输出相似度很高，但误差在后续层逐渐累积；多样性剪枝在剪枝层的输出相似度较低，但在更深层反而获得了更好的一致性。这说明两类方法各自只优化了问题的一个方面。

**本文目标** 如何联合考虑剪枝对当前层和后续层的影响，在局部与全局目标之间取得平衡。

**切入角度**：既然注意力方法擅长局部一致性、多样性方法擅长全局表示，那么在不同剪枝阶段动态调整两者的权重——浅层多保留多样性（因为后续层多）、深层多保留注意力焦点（因为 token 少且影响集中）。

**核心 idea**：浅层用多样性为后续层"储备"信息，深层用注意力为当前输出"兜底"。

## 方法详解

### 整体框架

BTP 首先使用一个小型校准集（64 个样本）确定剪枝层位置和阶段划分，然后在每个阶段按照 local-global 目标函数选择保留的 token。输入是完整的视觉 token 序列，输出是逐阶段压缩后的 token 子集。整个方法是 plug-and-play 的，无需修改模型结构或重新训练。

### 关键设计

1. **局部-全局联合目标函数**:

    - 功能：统一注意力和多样性两个优化方向
    - 核心思路：目标函数 $\mathcal{L}_{local-global} = -\sum_{i}(\lambda_i \sum_{j \in P_i} Atten^{(i)}(X_I^{(j)}, X_T) + (1-\lambda_i) F_{dis}(P_i))$，其中第一项是注意力目标（保持当前层输出一致性），第二项是多样性目标（保留后续层所需信息），$\lambda_i$ 随着层数增加而增大——浅层 $\lambda$ 小侧重多样性，深层 $\lambda$ 大侧重注意力
    - 设计动机：浅层保留更多 token，有空间容纳多样信息以供后续层使用；深层 token 数少，需优先保证当前输出的准确性

2. **基于位置的注意力再平衡**:

    - 功能：消除位置编码带来的选择偏差
    - 核心思路：注意力分数受位置编码影响，靠后位置的 token 倾向获得更高分数。BTP 先过选取 $k' > k$ 个 token，然后从前半部分序列保留所有候选 token，从后半部分按注意力排序填充剩余名额 $k - |I_{pre}|$，确保前半部分位置的 token 不被位置偏差淘汰
    - 设计动机：直接按注意力排序会因位置偏差丢失前部有价值的 token

3. **基于空间位置的多样性初始化**:

    - 功能：将 $O(n^2)$ 的 Max-Min Diversity Problem 降低到实用效率
    - 核心思路：观察到空间距离远的 image patch 语义差异大、距离近的相似，因此先在 2D 网格上用曼哈顿距离求解空间 MMDP 作为初始化，然后只需少量额外选择。这避免了在高维语义空间中求解完整 MMDP 的高计算开销
    - 设计动机：DivPrune 的 MMDP 求解是 $O(n^2)$ 且无法被 GPU 加速，导致实际推理延迟甚至超过无剪枝模型

### 剪枝层自动选择

通过计算相邻层间图像 token 隐状态的余弦相似度，找到语义变化剧烈的层——这些层前后是理想的剪枝位置。由于 causal mask 的存在，图像 token 编码独立于输入问题，因此可用固定的 64 个校准样本确定剪枝层，跨任务通用。

## 实验关键数据

### 主实验（LLaVA-1.5-7B, 128 token）

| 方法 | Token数 | GQA | MME | MMB | POPE | SQA | Avg% |
|------|---------|-----|------|-----|------|-----|------|
| Original | 576 | 62.0 | 1510.7 | 64.3 | 85.8 | 69.4 | 100% |
| VTW | 236 | 51.3 | 1475.0 | 63.4 | 82.1 | 68.8 | 89% |
| FastV | 172 | 57.6 | 1465.0 | 61.6 | 81.0 | 68.9 | 96% |
| DivPrune | 128 | 58.8 | 1405.4 | 62.1 | 85.1 | 68.4 | 96% |
| **BTP** | **128** | **59.0** | **1487.0** | **62.7** | **85.6** | **69.1** | **98%** |

### 消融实验（$\lambda$ 影响）

| 配置 | 说明 | 性能趋势 |
|------|------|---------|
| 浅层固定, 中深层调 | 浅层过早偏向注意力 | 性能下降 |
| 中层固定, 浅深层调 | 中层需保持适度多样性 | 最优区间窄 |
| 深层固定, 浅中层调 | 深层应侧重注意力 | 性能稳定 |

### 效率对比（LLaVA-1.5-7B）

| 方法 | Token数 | 延迟 | TFLOPS |
|------|---------|------|--------|
| Original | 576 | 0.145s | 3.82 |
| DivPrune | 128 | 0.224s (54%↑) | 0.83 |
| **BTP** | **128** | **0.134s (7%↓)** | **0.85** |

### 关键发现

- BTP 在 128 token 下保持 98% 原始性能，优于所有对比方法（DivPrune 和 FastV 为 96%）
- DivPrune 虽然 TFLOPS 最低，但实际延迟反而比无剪枝模型高 54%，因为 MMDP 求解无法被 GPU 加速；BTP 通过空间初始化解决了这个问题，实际延迟降低 7%
- 在 Qwen2.5-VL-7B 这种动态分辨率模型上，BTP 在 25% token 保留率下仍保持 97% 性能，而 VTW 骤降至 65%
- KV cache 在 LLaVA-1.6-7B 上从 1.11GB 降至 0.28GB（降低 74.7%）

## 亮点与洞察

- **对注意力 vs 多样性剪枝的深度实验分析**是本文最大亮点——通过逐层可视化隐状态相似度，清晰地揭示了两类方法的互补性，为分阶段策略提供了坚实的实证基础
- **空间位置初始化多样性选择**巧妙地将高维语义多样性问题降维为 2D 空间距离问题，既保证了效果又解决了 DivPrune 的实际部署问题（推理反而变慢）
- **设计理念可迁移**：浅层保多样性、深层保注意力的思路可以应用到其他层级剪枝/压缩任务中

## 局限与展望

- 校准集虽小（64 样本）但仍需额外数据，完全无数据的自适应剪枝策略值得探索
- 仅评估了 4 个模型，没有在更大规模模型（70B+）上验证
- $\lambda$ 的跨阶段递增策略是手动设定的，可以考虑学习 $\lambda$ 的最优调度
- 多样性和注意力目标都假设 token 独立处理，没有考虑 token 间的交互关系

## 相关工作与启发

- **vs FastV**: FastV 在某层后直接按注意力剪枝，属于纯局部优化，BTP 在更少 token 下性能更好
- **vs DivPrune**: DivPrune 纯多样性策略在局部层面损失大，且 MMDP 求解慢导致实际推理变慢。BTP 的空间初始化+平衡策略在效率和质量上全面超越
- **vs PyramidDrop**: 逐阶段剪枝思路类似，但 PyramidDrop 每阶段都用注意力排序，本文证明这不是全局最优

## 评分

- 新颖性: ⭐⭐⭐⭐ 局部-全局联合优化的剪枝视角新颖，分析透彻
- 实验充分度: ⭐⭐⭐⭐ 跨模型、跨压缩率、效率分析全面，消融扎实
- 写作质量: ⭐⭐⭐⭐ 从现象分析到方法设计的逻辑链清晰
- 价值: ⭐⭐⭐⭐ 对 LVLM 部署加速有实际意义，plug-and-play 设计实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] SCOPE: Saliency-Coverage Oriented Token Pruning for Efficient Multimodal LLMs](scope_saliency-coverage_oriented_token_pruning_for_efficient_multimodel_llms.md)
- [\[NeurIPS 2025\] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin.md)
- [\[ICCV 2025\] METEOR: Multi-Encoder Collaborative Token Pruning for Efficient Vision Language Models](../../ICCV2025/multimodal_vlm/meteor_multi-encoder_collaborative_token_pruning_for_efficient_vision_language_m.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](../../ACL2026/multimodal_vlm/hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)
- [\[ECCV 2024\] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](../../ECCV2024/multimodal_vlm/ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
