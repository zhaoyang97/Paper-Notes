---
title: >-
  [论文解读] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization
description: >-
  [NeurIPS 2025][视频理解][长视频理解] LongVPO提出两阶段DPO框架，Stage 1通过锚定短片段构造伪长视频偏好数据并引入anchor-only参考模型近似解决上下文长度不匹配问题，Stage 2通过递归字幕生成和多片段推理任务在真实长视频上自训练，仅用16K合成样本即超越大规模监督训练的长视频模型。
tags:
  - "NeurIPS 2025"
  - "视频理解"
  - "长视频理解"
  - "DPO"
  - "视觉语言模型"
  - "偏好优化"
  - "短到长迁移"
---

# LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2602.02341](https://arxiv.org/abs/2602.02341)  
**代码**: [GitHub](https://github.com/MCG-NJU/LongVPO)  
**领域**: LLM对齐  
**关键词**: 长视频理解, DPO, 视觉语言模型, 偏好优化, 短到长迁移

## 一句话总结

LongVPO提出两阶段DPO框架，Stage 1通过锚定短片段构造伪长视频偏好数据并引入anchor-only参考模型近似解决上下文长度不匹配问题，Stage 2通过递归字幕生成和多片段推理任务在真实长视频上自训练，仅用16K合成样本即超越大规模监督训练的长视频模型。

## 研究背景与动机

### 现有痛点

**现有痛点**：**领域现状**：短上下文VLM在长视频任务上表现下降的两个核心挑战：

1. **长视频标注稀缺**：高质量的长视频问答标注极其昂贵且覆盖不全
2. **位置偏差 (Lost-in-the-middle)**：短上下文VLM通过位置编码扩展处理长序列时，中间位置的内容容易被忽略

现有DPO方法（如LongVILA）假设参考模型本身支持长上下文推理，但短上下文VLM不满足这一假设。且依赖闭源模型生成偏好数据会引入外部偏差。

## 方法详解

### 整体框架

两阶段渐进式DPO训练：
- **Stage 1**：锚定短片段学习——用短视频SFT数据拼接合成伪长视频，用anchor-only近似解决参考模型退化
- **Stage 2**：长视频自训练对齐——在真实长视频上通过递归字幕和多片段推理构造偏好数据

### 关键设计

1. **锚定线索的短到长学习 (Stage 1)**:
    - 功能：将多个短片段拼接为伪长视频，围绕锚定片段生成偏好三元组 $(q_i, y_i^+, y_i^-)$
    - 核心思路：选择一个anchor短片段生成问答对作为preferred response，用非anchor片段的回答作为dispreferred response；随机化anchor位置来缓解位置偏差
    - 设计动机：模拟长视频中的"大海捞针"场景，训练模型在大量干扰中找到正确片段
    - 质量过滤：场景相似度过滤（DINOv2嵌入去除与anchor过于相似的干扰片段）+ 问题特异性过滤（LLM验证问题确实依赖anchor的特有信息）
    - **Anchor-only参考模型近似**：$\pi_{ref}(y|x_i) \approx \pi_{ref}(y|x_{i,anchor})$，只在anchor片段上评估参考模型，避免上下文长度不匹配导致的得分退化

2. **长视频偏好对齐自训练 (Stage 2)**:
    - 功能：在真实长视频上构造多片段推理的偏好数据进行DPO训练
    - 核心思路：递归字幕→LLM生成跨场景问答→场景引用追踪→构造部分/无关上下文的dispreferred回答
    - 设计动机：Stage 1使用拼接片段缺乏真实长视频的叙事连贯性，Stage 2用真实视频补充因果推理和事件链理解
    - Dispreferred生成策略：(a) 部分证据推理——只给相关场景的子集；(b) 无关场景幻觉——只给不相关场景

### 损失函数 / 训练策略

- 两阶段均使用DPO损失 + SFT辅助损失：$\mathcal{L} = \mathcal{L}_{DPO} + \alpha \cdot \frac{-\log \pi_\theta(y^+|x)}{|y^+|}$
- Stage 1的DPO损失使用anchor-only参考模型近似
- Stage 2的参考模型固定为Stage 1的checkpoint
- 总训练数据仅~16K样本

## 实验关键数据

### 主实验（表格）

| 模型 | 参数量 | LVBench | LongVideoBench | MLVU M-Avg | Video-MME (w/o sub) | MVBench |
|------|--------|---------|----------------|------------|---------------------|---------|
| LLaVA-Video-7B | 7B | - | 58.2 | 70.8 | 63.3 | 58.6 |
| Kangaroo-8B | 8B | - | 54.8 | 61.0 | 56.0 | 61.0 |
| LongVU-7B | 7B | - | - | 65.4 | 60.6 | 66.9 |
| **LongVPO-7B** | **7B** | **-** | **>58** | **>70** | **>63** | **维持** |

LongVPO在多个长视频benchmark上超越大规模监督训练的模型，同时在短视频MVBench上保持竞争力。

### 消融实验

- 仅Stage 1即可带来显著提升，Stage 2进一步增强多片段推理能力
- Anchor-only近似对Stage 1至关重要——不使用时DPO训练不稳定
- 场景相似度过滤和问题特异性过滤缺一不可
- 位置随机化有效缓解位置偏差

### 关键发现

- 短上下文VLM存在明显的"lost-in-the-middle"现象，中间位置内容检索能力显著下降
- 仅16K合成样本即可匹配或超越使用大规模标注数据训练的模型
- Anchor-only参考模型近似是关键创新，解决了DPO从短到长迁移的根本障碍

## 亮点与洞察

- **无需长视频标注**：完全从短视频数据出发，通过巧妙的数据合成实现长视频能力
- **数据效率极高**：16K样本 vs 其他方法的百万级数据
- **渐进式设计合理**：Stage 1建立基本的片段检索能力，Stage 2培养跨片段推理能力
- Anchor-only近似是优雅的工程解决方案，既降低了计算成本又解决了分布偏移

## 局限与展望

- Stage 2依赖递归字幕的质量，字幕错误会传播到偏好数据
- 未探索超过1小时的超长视频场景
- Self-generated的preferred response不一定完美，可能限制性能上限
- 拼接式伪长视频与真实长视频的分布差异难以完全弥合

## 相关工作与启发

- 延续了DPO从NLP到多模态的扩展趋势
- 与LongVILA、VideoChat-Flash等长视频方法的区别：不需要长视频标注
- Anchor-only近似思路可推广到其他需要处理上下文长度不匹配的DPO场景

## 评分

⭐⭐⭐⭐ — 方法设计优雅、数据效率出色，两阶段渐进式设计逻辑清晰，是长视频理解的有效范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Self-alignment of Large Video Language Models with Refined Regularized Preference Optimization](self-alignment_of_large_video_language_models_with_refined_regularized_preferenc.md)
- [\[AAAI 2026\] TSPO: Temporal Sampling Policy Optimization for Long-form Video Language Understanding](../../AAAI2026/video_understanding/tspo_temporal_sampling_policy_optimization_for_long-form_video_language_understa.md)
- [\[CVPR 2025\] T*: Re-thinking Temporal Search for Long-Form Video Understanding](../../CVPR2025/video_understanding/re-thinking_temporal_search_for_long-form_video_understanding.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)
- [\[NeurIPS 2025\] Unleashing Hour-Scale Video Training for Long Video-Language Understanding](unleashing_hour-scale_video_training_for_long_video-language_understanding.md)

</div>

<!-- RELATED:END -->
