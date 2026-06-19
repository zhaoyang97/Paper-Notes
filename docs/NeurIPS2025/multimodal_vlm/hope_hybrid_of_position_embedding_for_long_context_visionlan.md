---
title: >-
  [论文解读] HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models
description: >-
  [NeurIPS 2025][多模态VLM][旋转位置编码] 首次从理论上分析多模态RoPE的频率分配策略对长上下文VLM的影响，提出HoPE，将最低频率设为零用于时间建模以保证语义偏好性质，配合动态时间缩放机制，在长视频理解任务上提升8.35%、检索任务上提升22.23%。 领域现状 领域现状：VLM在长上下文场景下性能显…
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "旋转位置编码"
  - "长上下文"
  - "频率分配"
  - "视觉语言模型"
  - "视频理解"
---

# HoPE: Hybrid of Position Embedding for Long Context Vision-Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2505.20444](https://arxiv.org/abs/2505.20444)  
**代码**: [GitHub](https://github.com/hrlics/HoPE)  
**领域**: 多模态VLM / 位置编码  
**关键词**: 旋转位置编码, 长上下文, 频率分配, 视觉语言模型, 视频理解

## 一句话总结

首次从理论上分析多模态RoPE的频率分配策略对长上下文VLM的影响，提出HoPE，将最低频率设为零用于时间建模以保证语义偏好性质，配合动态时间缩放机制，在长视频理解任务上提升8.35%、检索任务上提升22.23%。

## 研究背景与动机

### 领域现状

**领域现状**：VLM在长上下文场景下性能显著下降，尤其是长视频任务中甚至难以完成目标计数和时间定位

### 核心矛盾

**核心矛盾**：RoPE在文本LLM中成功实现长度泛化，但直接应用1D RoPE无法捕获视频的时空结构

### 现有痛点

**现有痛点**：现有多模态RoPE扩展的局限：

### 解决思路

**解决思路**：M-RoPE（Qwen2-VL）：最高频率分配给时间维度，启发式设计缺乏理论分析

### 补充说明

**补充说明**：VideoRoPE：最低频率分配给时间维度，经验表现好但长距离下仍不可靠

### 补充说明

**补充说明**：固定和单向的时间缩放因子无法适应不同速度和信息密度的视频

### 补充说明

**补充说明**：核心问题：**不同频率分配策略如何影响长程语义建模？能否获得理论保证？**

## 方法详解

### 整体框架

HoPE包含两个组件：（1）混合频率分配（HFA）策略——高频率交错编码空间信息 $(x,y)$，最低频率设为零用于时间建模，保证语义偏好性质的上界；（2）动态时间缩放（DTS）机制——训练时随机选择缩放因子（含压缩和拉伸），推理时灵活适配不同长度。

### 关键设计

1. **混合频率分配（HFA）**:
    - 功能：将128维旋转编码分为96维空间编码和32维时间编码，时间维频率设为零（退化为NoPE）
    - 核心思路：利用 $\cos(0 \cdot \Delta t) = 1$ 恒成立，消除时间距离对注意力分数的负面影响
    - 设计动机：理论证明（Theorem 3.1）任何非零频率在足够长的上下文中都会破坏语义偏好性质，零频率提供最强保证
    - 关键定理：$\sum_{i \in i_t} 2\sigma^2 \cdot 1 \geq \sum_{i \in i_t} 2\sigma^2 \cos(\Delta t \cdot \theta_i)$，零频率方案优于任何其他频率选择

2. **动态时间缩放（DTS）**:
    - 功能：训练时从集合 $\Gamma = \{0.5, 0.75, 1, 1.25, 1.5\}$ 随机选择缩放因子
    - 核心思路：拉伸（$\gamma > 1$）保留空间细节，压缩（$\gamma < 1$）增强语义偏好，双向缩放让模型学习多尺度时间关系
    - 设计动机：真实视频速度各异，固定缩放无法适配；推理时可根据任务需求灵活选择

### 损失函数 / 训练策略

- 基于Qwen2-1.5B/7B骨干，使用LLaVA-Video-178k子集训练（约30k短视频 + 3k中长视频）
- 训练上下文长度8k，最大视频帧128帧，采样率2
- 学习率1e-5（2B）/ 2e-5（7B），cosine scheduler，batch size 128
- 训练约304 H800 GPU小时

## 实验关键数据

### 主实验（表格）

Qwen2-7B-Video模型，32k上下文长度：

| 方法 | MLVU | LongVideoBench | Video-MME |
|------|------|----------------|-----------|
| Vanilla RoPE | 61.03 | 51.29 | 57.99 |
| M-RoPE | 62.46 | 53.49 | 58.37 |
| VideoRoPE | 62.51 | 53.82 | 59.13 |
| **HoPE** | **63.85** | **55.34** | **59.44** |

长视频检索（V-NIAH）：HoPE相比最佳基线提升22.23%。

### 消融实验

- **3D结构**：从1D RoPE引入3D结构即可提升性能，验证Proposition 3.1
- **HFA策略**：基于3D结构进一步引入HFA，平均提升1.69
- **DTS机制**：在HFA基础上加DTS提供额外增益，增强对不同视频速度的鲁棒性
- **推理缩放因子选择**：检索任务偏好小因子（$\gamma=0.75$），理解任务偏好大因子（$\gamma=1.5$）

### 关键发现

- 64k外推时所有方法性能大幅下降，但HoPE最鲁棒（Video-MME: 27.34 vs Vanilla 26.13）
- 模型规模提升放大了HoPE的优势（2B到7B，LongVideoBench 32k增益从0.66到4.05）
- 检索与理解任务对缩放因子的偏好相反：检索需保持语义偏好、理解需保留空间细节

## 亮点与洞察

- 首次对多模态RoPE频率分配进行严格理论分析，而非纯经验对比
- "零频率=NoPE用于时间维"的发现简洁深刻——最低频率不够低，必须为零才能保证
- 语义偏好性质（Definition 3.1）提供了分析框架，可推广到更多位置编码设计
- 双向缩放是学习多尺度时间关系的自然方式，推理灵活性强

## 局限与展望

- 实验仅到7B规模，更大模型和更多训练数据可能进一步放大优势
- 64k外推性能仍大幅下降，极端长度泛化仍是开放问题
- 零频率策略完全放弃了时间维的显式编码，对需要精确时间定位的任务可能不利
- 未在decoder-only架构的因果注意力中讨论HoPE与隐式位置学习的交互

## 相关工作与启发

- 与LongRoPE、YaRN等LLM长度扩展工作互补，HoPE专注多模态扩展
- NoPE在decoder-only LLM中的成功为零频率策略提供了先验支持
- 动态缩放思想可推广到其他多模态输入（如音频、点云等）的位置编码设计

## 评分

- ⭐⭐⭐⭐⭐ — 理论分析深入，设计简洁有效，实验全面，对VLM长上下文领域有重要启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables](needleinatable_exploring_long-context_capability_of_large_language_models_toward.md)
- [\[NeurIPS 2025\] MMLongBench: Benchmarking Long-Context Vision-Language Models Effectively and Thoroughly](mmlongbench_benchmarking_longcontext_visionlanguage_models_e.md)
- [\[NeurIPS 2025\] Context Informs Pragmatic Interpretation in Vision-Language Models](context_informs_pragmatic_interpretation_in_vision-language_models.md)
- [\[NeurIPS 2025\] ExGra-Med: Extended Context Graph Alignment for Medical Vision-Language Models](exgra-med_extended_context_graph_alignment_for_medical_vision-language_models.md)
- [\[CVPR 2025\] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](../../CVPR2025/multimodal_vlm/identifying_and_mitigating_position_bias_of_multi-image_vision-language_models.md)

</div>

<!-- RELATED:END -->
