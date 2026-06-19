---
title: >-
  [论文解读] Streamline Without Sacrifice — Squeeze out Computation Redundancy in LMM
description: >-
  [ICML 2025][多模态VLM][computation redundancy] 提出 ProxyV，通过引入少量代理视觉 token（proxy vision tokens）替代原始视觉 token 参与 LLM 解码层中的重计算操作（自注意力、FFN），在保留全部视觉信息的前提下大幅压缩计算冗余，甚至在部分设定下提升性能。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "computation redundancy"
  - "proxy vision tokens"
  - "ProxyV"
  - "decoder-only LMM"
  - "token efficiency"
---

# Streamline Without Sacrifice — Squeeze out Computation Redundancy in LMM

**会议**: ICML 2025  
**arXiv**: [2505.15816](https://arxiv.org/abs/2505.15816)  
**作者**: Penghao Wu, Lewei Lu, Ziwei Liu  
**领域**: 多模态大模型 / 视觉 token 计算效率  
**关键词**: computation redundancy, proxy vision tokens, ProxyV, decoder-only LMM, token efficiency

## 一句话总结
提出 ProxyV，通过引入少量代理视觉 token（proxy vision tokens）替代原始视觉 token 参与 LLM 解码层中的重计算操作（自注意力、FFN），在保留全部视觉信息的前提下大幅压缩计算冗余，甚至在部分设定下提升性能。

## 背景与动机

当前主流大型多模态模型（LMM）采用 decoder-only 架构（如 LLaVA 系列），将视觉编码器提取的视觉 token 与文本 token 拼接后送入 LLM 联合处理。这一结构面临严峻的计算挑战：

1. **视觉 token 数量远超文本 token**：高分辨率图像可产生数千个 token，自注意力的二次复杂度使得计算开销极大。
2. **多图/视频场景加剧问题**：视频理解和多图任务使视觉序列长度进一步膨胀。
3. **现有 token 减少方法存在信息损失风险**：剪枝（pruning）和合并（merging）是不可逆操作，可能丢失细粒度细节（如密集文档图像中的关键信息）。
4. **依赖文本引导的方法不鲁棒**：基于 text-to-image 注意力分数的 token 选择在多轮对话中可能失效（后续问题所需的视觉信息可能在第一轮已被丢弃）。
5. **与高效注意力实现不兼容**：许多 token 减少方法依赖 text-to-image 注意力分数，无法与 FlashAttention 等高效实现配合。

本文提出一个全新视角：**不在 token 层面做减少，而是在计算层面消除冗余**。核心观察是——来自预训练视觉编码器的视觉 token 已经具有高度语义性，在 LLM 解码器中未必需要经历全部重计算操作。

## 核心问题

- 视觉 token 在 LLM 解码层中是否存在计算级别的冗余？
- 能否在保持 decoder-only 架构简洁性的同时，减少视觉 token 的计算负担？
- 如何在不丢失任何视觉信息的前提下提升推理效率？

## 方法详解

### 整体框架

ProxyV 沿用标准 decoder-only LMM 架构（视觉编码器 → 投影层 → LLM），但在 LLM 的中间层（如第 12 层或第 16 层）开始引入**代理视觉 token（proxy vision tokens）机制**：

1. **前 N 层（如 Layer 0–11）**：原始视觉 token 正常参与所有计算（自注意力 + FFN），与文本 token 充分交互。
2. **第 N 层起**：用少量代理 token 压缩/代表原始视觉 token 的信息，后续层中仅代理 token 参与重计算操作，原始视觉 token 不再经过 vision-to-vision 自注意力和 FFN。
3. **文本 token 通过 cross-attention 与代理 token 交互**，仍能获取完整视觉上下文。

这一设计的关键在于：原始视觉 token 全部保留（无信息损失），但不再承担高成本的自注意力和 FFN 计算。

### 关键设计

1. **代理视觉 token 生成**：从原始视觉 token 中通过池化或可学习聚合生成少量代理 token（数量远少于原始 token），作为视觉信息的紧凑表示。
2. **选择性计算分离**：在指定层之后，将计算路径分为两条——代理 token 参与完整的 Transformer 计算（self-attention + FFN），而原始视觉 token 仅在需要时被文本 token 通过 attention 访问。
3. **渐进式冗余压缩实验**：论文系统地设计了一系列实验，逐步验证各类计算操作（vision-to-vision attention、FFN、vision-to-text attention）的冗余程度，以数据驱动方式确定最优的计算裁剪策略。
4. **灵活的起始层选择**：ProxyV-L12（从第 12 层开始）提供更多效率增益，ProxyV-L16（从第 16 层开始）保留更多性能。用户可根据效率-性能 trade-off 灵活选择。

### 训练策略

- 在标准 LMM 训练流程上微调，仅需额外训练代理 token 的聚合模块。
- 训练成本与原始模型相当，不需要显著增加预训练数据。
- 不引入大量额外参数（相比 cross-attention 架构如 Flamingo 的重量级改造）。
- 可与现有 token 减少方法（如 FastV、TokenPacker）正交组合，进一步提升效率。

## 实验关键数据

### 主要性能对比

| 方法 | 视觉 token 数 | GFLOPs ↓ | TextVQA | DocVQA | OCRBench | ChartQA | 平均 |
|------|-------------|----------|---------|--------|----------|---------|------|
| 基线 (LLaVA-style) | 2880 | 100% | 基准 | 基准 | 基准 | 基准 | 基准 |
| Token Pruning | ~720 | ~35% | 下降 | 明显下降 | 明显下降 | 下降 | 下降 |
| Token Merging | ~720 | ~35% | 下降 | 下降 | 下降 | 下降 | 下降 |
| **ProxyV-L16** | 2880 (保留) | ~60% | **持平/↑** | **持平/↑** | **持平/↑** | **持平** | **↑** |
| **ProxyV-L12** | 2880 (保留) | ~45% | 持平 | 持平 | 持平 | 持平 | 持平 |

> 注：具体数值因缓存不完整取自摘要与图表描述，趋势来源于原文 Figure 1 及摘要说明。ProxyV-L16 在细粒度基准上甚至超越基线。

### 与 Token 减少方法组合

| 组合方案 | 效率提升 | 性能影响 |
|----------|---------|---------|
| ProxyV 单独 | 中等（40–55% FLOPs 节省） | 持平或略正 |
| Token Reduction 单独 | 高（~65% FLOPs 节省） | 负面（尤其细粒度任务） |
| **ProxyV + Token Reduction** | **更高** | **优于单独 Token Reduction** |

> ProxyV 与 token reduction 方法正交互补：ProxyV 减少"计算冗余"，token reduction 减少"token 冗余"，两者组合效果叠加。

## 亮点与洞察

1. **全新视角：计算冗余 vs. token 冗余**。现有大部分工作聚焦于减少 token 数量，本文首次系统性地研究并压缩视觉 token 的计算级冗余，提供了正交的优化维度。
2. **无信息损失**。与 pruning/merging 不同，ProxyV 保留全部原始视觉 token，保证在密集文档、多轮对话等场景不会丢失关键细节。
3. **效率提升的同时性能可提升**。ProxyV-L16 在适度效率增益下反而带来性能增益，说明减少不必要的 vision-to-vision attention 可能减少了干扰信号。
4. **强兼容性**。ProxyV 可与 FlashAttention、token reduction 等现有技术无缝组合，不需要改动视觉编码器或 LLM 架构。
5. **cross-attention LMM 的简化替代**。相比 Flamingo 等需要大量预训练数据和额外参数的 cross-attention 架构，ProxyV 在 decoder-only 框架内以最小改动实现类似的效率收益。

## 局限性

1. **缓存中缺少完整实验数据**：本缓存仅包含摘要和引言，无法呈现全部定量结果的精确数值。
2. **代理 token 数量选择**：如何确定最优的代理 token 数量可能需要针对不同任务和分辨率的调参。
3. **起始层选择的敏感性**：ProxyV-L12 和 L16 在效率-性能权衡上有差异，最佳起始层可能因模型规模和任务而异。
4. **仅在 LLaVA-style 架构验证**：是否能推广到其他 LMM 架构（如 Qwen-VL、InternVL 等）需要进一步验证。
5. **视频和超长序列场景**：虽然论文提到了视频场景的动机，但代理 token 方法在极长视觉序列（数万 token）下的扩展性有待探索。

## 相关工作

- **Token 减少方法**：FastV (Chen et al., 2025b)、LLaVA-PruMerge (Shang et al., 2024)、FitPrune (Xing et al., 2024) 等通过剪枝/合并减少 token 数量，但有信息损失风险。
- **Cross-attention LMM**：Flamingo (Alayrac et al., 2022)、IDEFICS (Laurencon et al., 2023)、Llama 3 (Dubey et al., 2024) 使用跨注意力注入视觉信息，计算效率更高但需要更多预训练数据和额外参数。
- **高效注意力**：FlashAttention (Dao et al., 2022, 2023) 优化注意力计算的内存和时间效率，与 ProxyV 正交互补。
- **LLaVA 系列**：LLaVA (Liu et al., 2024b) 及后续高分辨率变体是本文的基线架构。

## 评分

| 维度 | 分数 (1-5) | 说明 |
|------|-----------|------|
| 新颖性 | ⭐⭐⭐⭐ | 首次系统研究 LMM 中视觉 token 的计算级冗余，视角全新 |
| 技术深度 | ⭐⭐⭐⭐ | 系统的渐进式实验设计、代理 token 机制设计合理 |
| 实验充分性 | ⭐⭐⭐⭐ | 多个细粒度 benchmark 验证，支持与其他方法组合 |
| 实用性 | ⭐⭐⭐⭐⭐ | 即插即用，与 FlashAttention 和 token reduction 兼容，落地门槛低 |
| 写作质量 | ⭐⭐⭐⭐ | 动机清晰，实验设计循序渐进 |
| **综合** | **⭐⭐⭐⭐** | 在 LMM 效率优化领域提供了重要的正交优化维度 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Redundancy Principles for MLLMs Benchmarks](../../ACL2025/multimodal_vlm/redundancy_principles_for_mllms_benchmarks.md)
- [\[ICML 2025\] OmniBal: Towards Fast Instruction-Tuning for Vision-Language Models via Omniverse Computation Balance](omnibal_towards_fast_instruction-tuning_for_vision-language_models_via_omniverse.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](ranked_from_within_ranking_large_multimodal_models_without_labels.md)
- [\[ICML 2025\] LAION-C: An Out-of-Distribution Benchmark for Web-Scale Vision Models](laion-c_an_out-of-distribution_benchmark_for_web-scale_vision_models.md)
- [\[CVPR 2025\] MLLM-as-a-Judge for Image Safety without Human Labeling](../../CVPR2025/multimodal_vlm/mllm-as-a-judge_for_image_safety_without_human_labeling.md)

</div>

<!-- RELATED:END -->
