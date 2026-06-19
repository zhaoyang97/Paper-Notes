---
title: >-
  [论文解读] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints
description: >-
  [NeurIPS 2025][多模态VLM][Native MLLM] 本文系统研究了在数据约束条件下原生多模态大语言模型(Native MLLM)的设计空间与缩放特性，发现视觉编码器与LLM之间存在正相关的最优缩放关系，并基于此提出了NaViL模型，仅用约6亿预训练图文对即可达到顶级MLLM的竞争性性能。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "Native MLLM"
  - "Scaling Law"
  - "Visual Encoder"
  - "Mixture-of-Experts"
  - "端到端训练"
---

# NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints

**会议**: NeurIPS 2025  
**arXiv**: [2510.08565](https://arxiv.org/abs/2510.08565)  
**代码**: [GitHub](https://github.com/OpenGVLab/NaViL)  
**领域**: Multimodal VLM  
**关键词**: Native MLLM, Scaling Law, Visual Encoder, Mixture-of-Experts, 端到端训练

## 一句话总结

本文系统研究了在数据约束条件下原生多模态大语言模型(Native MLLM)的设计空间与缩放特性，发现视觉编码器与LLM之间存在正相关的最优缩放关系，并基于此提出了NaViL模型，仅用约6亿预训练图文对即可达到顶级MLLM的竞争性性能。

## 研究背景与动机

**领域现状**：当前主流MLLM采用"组合式训练"（Compositional Training）范式——先独立预训练视觉编码器和LLM，再通过多模态训练进行融合。

**现有痛点**：组合式训练使得视觉和语言组件的联合缩放特性难以探索，且视觉-语言对齐受限于分离式训练的局限性。

**核心矛盾**：原生MLLM（端到端训练）展现了更好的缩放规律潜力，但现有研究主要在"无限资源"假设下评估，对数据和计算受限场景的实际可行性缺乏系统研究。

**本文目标**：在实际数据约束下，原生MLLM能否达到甚至超越顶级组合式MLLM的性能上限？

**切入角度**：系统探索原生MLLM的关键架构选择（LLM初始化、MoE、视觉编码器结构）和视觉-语言组件的联合缩放规律。

**核心idea**：视觉编码器的最优参数量随LLM参数量在对数尺度上线性增长，两者应联合缩放以获得最优性能。

## 方法详解

### 整体框架

NaViL 是一个端到端训练的原生 MLLM，由三部分组成：视觉编码器 $\mathcal{V}_{d,w}$、MLP 连接器 $\mathcal{C}$、以及 MoE 增强的 LLM。支持任意分辨率输入，使用视觉多尺度打包（Visual Multi-scale Packing）来提升推理性能。

### 关键设计

1. **LLM 初始化**：使用预训练 LLM（InternLM2-Base）初始化语言参数，而非从头训练。实验表明初始化模型的收敛速度优于从头训练超过 10 倍，且零样本图像描述性能显著领先。原因在于多模态训练数据的文本多样性和质量远不如纯语言预训练语料。

2. **模态专用 MoE**：在 LLM 的每一层引入模态特定的注意力专家和 FFN 专家，使用不同的投影矩阵 $W_Q^m, W_K^m, W_V^m, W_O^m$ 处理视觉和文本特征，但进行统一的全局注意力计算：
$$x_{i,m}^{l'} = x_{i,m}^{l-1} + \text{MHA-MMoE}(\text{RMSNorm}(x_{i,m}^{l-1}))$$
$$x_{i,m}^{l} = x_{i,m}^{l'} + \text{FFN-MMoE}(\text{RMSNorm}(x_{i,m}^{l'}))$$
MoE 使模型在不增加训练/推理代价的情况下，仅用 1/10 的数据即可达到相同的验证损失。

3. **视觉编码器架构优化**：在固定参数预算下探索最优的深度 $d$ 和宽度 $w$ 组合（参数量 $\mathcal{N} = 12 \times d \times w^2$）。实验发现极端的深度或宽度配置表现较差，中等配置在大规模数据下区别不大。

4. **视觉多尺度打包（Visual Multi-scale Packing）**：推理时将输入图像以下采样率 $\tau$ 持续下采样生成多尺度图像序列 $\{I_i\}_{i=0}^n$，分别由视觉编码器处理后拼接送入 LLM，使用特殊标记 `<end_of_scale>` 分隔不同尺度。

### 缩放特性发现

- **独立缩放 LLM**：验证损失随 LLM 参数量增加呈对数线性下降，符合传统语言缩放规律。
- **独立缩放视觉编码器**：性能收益递减——固定 LLM 时，增大视觉编码器到一定阈值后收益趋近于零，说明性能上限受 LLM 容量约束。
- **联合缩放**：最优视觉编码器大小的对数与 LLM 大小的对数呈线性关系，需联合缩放。这与组合式方法使用固定大小视觉编码器的做法形成对比。

### 训练策略

- **Stage 1 多模态生成预训练**：先用 5 亿图文对训练（3 亿网络采集 + 2 亿合成描述），仅训练视觉参数；再用 1.85 亿高质量数据解冻注意力层文本参数。
- **Stage 2 监督微调**：解冻全部参数，使用 6800 万高质量多模态数据微调。

## 实验关键数据

### 主实验

| 模型 | #激活参数 | Avg | MMVet | MMMU | MMB | MME | MathVista | OCRBench | CCB |
|------|-----------|-----|-------|------|-----|-----|-----------|----------|-----|
| InternVL-2.5-2B (组合式) | 2.2B | 67.0 | 60.8 | 43.6 | 74.7 | 2138 | 51.3 | 804 | 81.7 |
| Mono-InternVL (原生) | 1.8B | 56.4 | 40.1 | 33.7 | 65.5 | 1875 | 45.7 | 767 | 66.3 |
| **NaViL-2B** (原生) | **2.4B** | **67.1** | **78.3** | **41.8** | **71.2** | **1822** | **50.0** | **796** | **83.9** |

NaViL-2B 在多数指标上超越组合式对标模型 InternVL-2.5-2B，且大幅超越所有已有原生 MLLM。

### 消融实验

| 设计选择 | 效果 |
|---------|------|
| 有 vs 无 LLM 初始化 | 初始化版本收敛快 10x+，零样本描述性能显著领先 |
| 有 vs 无 MoE | MoE 版本仅用 1/10 数据达到相同损失 |
| 视觉编码器 d=3/6/12/24/48 | 极端配置表现差，中等配置差异小 |

### 关键发现

- 原生 MLLM 首次在 2B 参数量级达到与顶级组合式 MLLM 竞争的性能
- 视觉编码器最优大小随 LLM 大小呈对数线性增长
- MoE 架构对处理多模态异构数据至关重要

## 亮点与洞察

- **首个系统性研究**：在数据约束下对原生 MLLM 的设计空间和缩放特性进行了全面系统的探索
- **实用性强**：仅约 6 亿预训练数据就能训练出竞争力强的原生 MLLM
- **缩放规律新发现**：视觉编码器和 LLM 的最优缩放关系为原生 MLLM 设计提供了重要指导
- **模态专用 MoE 设计**：除 FFN 专家外还引入注意力专家，解决了模态间特征尺度差异问题

## 局限与展望

- 仅探索了图像模态，未扩展到视频、音频等其他模态
- 缩放规律的验证范围有限（LLM 最大 7B），更大规模是否仍成立有待验证
- 视觉编码器架构直接复用 LLM 的 Transformer 层，未探索更多视觉专用设计
- 多尺度打包增加了推理时的计算开销

## 相关工作与启发

- **Chameleon**：从头训练的原生 MLLM，性能远逊于 NaViL，说明 LLM 初始化至关重要
- **Mono-InternVL**：首个引入模态 MoE 的原生 MLLM，NaViL 在此基础上进一步引入注意力专家
- **组合式 MLLM**（InternVL、Qwen2VL）：采用固定大小视觉编码器的策略被证明是次优的

## 评分

- 新颖性: ⭐⭐⭐⭐ 系统性研究原生MLLM缩放规律是新颖贡献，但架构设计本身并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 14个基准测试，大量消融实验和缩放分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，分析系统
- 价值: ⭐⭐⭐⭐⭐ 为原生MLLM设计提供了重要的实践指导和理论洞察

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Scaling Laws for Native Multimodal Models](../../ICCV2025/multimodal_vlm/scaling_laws_for_native_multimodal_models.md)
- [\[NeurIPS 2025\] BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)
- [\[NeurIPS 2025\] The Narrow Gate: Localized Image-Text Communication in Native Multimodal Models](the_narrow_gate_localized_imagetext_communication_in_native.md)
- [\[NeurIPS 2025\] Evaluating Multimodal Large Language Models on Core Music Perception Tasks](evaluating_multimodal_large_language_models_on_core_music_perception_tasks.md)

</div>

<!-- RELATED:END -->
