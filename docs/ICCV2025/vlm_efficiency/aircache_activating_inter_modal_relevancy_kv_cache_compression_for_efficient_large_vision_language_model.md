---
title: >-
  [论文解读] AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference
description: >-
  [ICCV 2025][多模态VLM][KV cache compression] 提出AirCache，通过精英观测窗口（利用文本自注意力筛选关键文本token评估视觉token重要性）和自适应层间预算分配（基于重要性分数分布的强度和偏度），实现仅保留10%视觉KV缓存即可保持模型性能，解码延迟降低29%-66%。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "KV cache compression"
  - "视觉语言"
  - "elite observation window"
  - "adaptive budget allocation"
  - "注意力机制"
---

# AirCache: Activating Inter-modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference

**会议**: ICCV 2025  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: KV cache compression, vision-language model, elite observation window, adaptive budget allocation, attention analysis

## 一句话总结

提出AirCache，通过精英观测窗口（利用文本自注意力筛选关键文本token评估视觉token重要性）和自适应层间预算分配（基于重要性分数分布的强度和偏度），实现仅保留10%视觉KV缓存即可保持模型性能，解码延迟降低29%-66%。

## 研究背景与动机

大型视觉语言模型（LVLM）处理高分辨率图像、多图和视频时产生大量视觉token，加上长文本生成需求，KV缓存的线性增长导致巨大的内存和带宽压力。现有方法分两类：(1) Token剪枝——在prefill阶段减少token，但激进删除导致视觉信息严重丢失；(2) KV缓存压缩——在解码阶段剪枝缓存数据，对性能影响更小。然而现有KV缓存压缩方法存在两个问题：观测窗口选择不当（使用全部或连续局部文本token导致评估不一致）和均匀层间预算分配不优。

## 方法详解

### 整体框架

AirCache在prefill完成后介入KV缓存存储过程，包含两个组件：(1) 视觉token重要性评分——通过精英观测窗口筛选关键文本token，用跨模态注意力投票评估视觉token重要性；(2) 层间KV缓存预算分配——基于重要性分数分布的强度和偏度为不同层动态分配压缩预算。

### 关键设计

1. **精英观测窗口(Elite Observation Window)**: 利用文本token间的自注意力分数，以最后一个文本token为参考，筛选出注意力分数高于阈值α·max的关键文本token。用这些精选的关键文本token作为观测窗口，通过其与视觉token的跨模态注意力评估视觉token重要性。相比使用全部文本token或连续局部窗口，精英窗口内的token对同一视觉token的评价更一致，减少投票排名中的噪声。

2. **自适应层间预算分配**: 从两个维度量化不同层的KV缓存压缩预算：(a) 分布强度——某层分配给所有视觉token的注意力总和，反映该层对视觉信息的重视程度；(b) 分布偏度——重要性分数分布的头部效应，少数视觉token获得极高分数而多数token分数平庸。偏度越大表明压缩潜力越大。两者结合实现比均匀分配更优的层间预算分配。

3. **视觉token重要性的头部效应**: 实验发现视觉token重要性呈极强的头部分布——少数token极重要而绝大多数不重要。仅保留10%的重要视觉KV缓存，模型性能下降不到1%。不同层的视觉信息强度差异显著，早期层和晚期层侧重点不同。

### 损失函数 / 训练策略

无需训练的方法，直接在推理阶段的KV缓存管理中应用。兼容主流LVLM架构（LLaVA-OV、InternVL等）。

## 实验关键数据

### 主实验

| 方法 | 视觉KV保留率 | 平均性能 | 解码延迟降低 |
|------|-------------|---------|-------------|
| Full Cache | 100% | 基准 | - |
| H2O | 10% | 显著下降 | - |
| SnapKV | 10% | 中等下降 | - |
| **AirCache** | **10%** | **<1%下降** | **29%-66%** |

在多种LVLM和基准上，AirCache以10%保留率实现最接近全缓存的性能。

### 消融实验

- 精英窗口 vs 全文本窗口 vs 局部窗口：精英窗口一致性最强
- 自适应分配 vs 均匀分配：自适应显著更优
- 保留率从10%到50%：AirCache在低保留率下优势更明显
- 不同层的视觉注意力强度差异：验证了非均匀分配的必要性

### 关键发现

- 视觉token重要性呈强烈的头部效应，90%的KV缓存可安全移除
- 不同文本token对视觉token的评价差异大，需精心选择观测窗口
- 层间预算不应均匀分配——某些层对视觉信息更敏感

## 亮点与洞察

- 精英观测窗口的设计直觉清晰——不是所有文本token都适合评估视觉重要性
- 基于分布强度和偏度的双维度预算分配比启发式方法更有理论基础
- 10%保留率下性能下降<1%，证明视觉KV缓存冗余极大
- 与token剪枝正交，可联合使用进一步加速

## 局限与展望

- 精英窗口的阈值α需要调参
- 在极低保留率（<5%）下性能可能急剧下降
- 仅针对解码阶段优化，prefill阶段的开销未减少
- 未在视频理解等长序列场景中充分验证

## 相关工作与启发

- H2O、SnapKV是LLM领域的KV缓存压缩方法
- PyramidKV提出分层预算分配
- VL-Cache利用视觉token稀疏性，是最直接的LVLM KV缓存对比方法
- 方法可扩展到长视频和多图场景的高效推理

## 评分

- 新颖性: ⭐⭐⭐⭐ — 精英窗口和双维度预算分配设计新颖
- 技术深度: ⭐⭐⭐⭐ — 注意力分布分析深入
- 实验充分性: ⭐⭐⭐⭐ — 多模型、多基准、多保留率对比
- 写作质量: ⭐⭐⭐⭐ — 动机和可视化分析出色
- 实用价值: ⭐⭐⭐⭐⭐ — 无需训练、兼容性好、加速显著

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](scaling_inferencetime_search_with_vision_value_model_for_imp.md)
- [\[ICCV 2025\] Free-MoRef: Instantly Multiplexing Context Perception Capabilities of Video-MLLMs within Single Inference](free-moref_instantly_multiplexing_context_perception_capabilities_of_video-mllms.md)
- [\[ICCV 2025\] Vision-Language Models Can't See the Obvious](vision-language_models_cant_see_the_obvious.md)
- [\[ICCV 2025\] The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models](the_inter-intra_modal_measure_a_predictive_lens_on_fine-tuning_outcomes_in_visio.md)
- [\[ICCV 2025\] TAB: Transformer Attention Bottlenecks enable User Intervention and Debugging in Vision-Language Models](tab_transformer_attention_bottlenecks_enable_user_intervention_and_debugging_in_.md)

</div>

<!-- RELATED:END -->
