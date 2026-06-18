---
title: >-
  [论文解读] AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference
description: >-
  [ICCV 2025][多模态VLM][KV Cache压缩] 提出 AirCache，一种面向 LVLM 的 KV Cache 压缩方法，通过精英观察窗口（Elite Observation Window）评估视觉 token 重要性，结合基于重要性分数分布强度与偏度的自适应层级预算分配，在仅保留 10% 视觉 KV Cache 时性能损失不超过 1%，解码延迟降低 29%-66%。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "KV Cache压缩"
  - "大视觉语言模型"
  - "跨模态注意力"
  - "层级预算分配"
  - "推理加速"
---

# AirCache: Activating Inter-Modal Relevancy KV Cache Compression for Efficient Large Vision-Language Model Inference

**会议**: ICCV 2025  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: KV Cache压缩, 大视觉语言模型, 跨模态注意力, 层级预算分配, 推理加速

## 一句话总结

提出 AirCache，一种面向 LVLM 的 KV Cache 压缩方法，通过精英观察窗口（Elite Observation Window）评估视觉 token 重要性，结合基于重要性分数分布强度与偏度的自适应层级预算分配，在仅保留 10% 视觉 KV Cache 时性能损失不超过 1%，解码延迟降低 29%-66%。

## 研究背景与动机

### KV Cache 瓶颈
大视觉语言模型（LVLM）在推理时面临严重的 KV Cache 内存压力：
- **视觉 token 暴增**：高分辨率、多图像、视频序列等场景使视觉 token 数量呈指数增长
- **长上下文生成**：KV Cache 随序列长度线性增长，导致延迟和内存不可持续

### 两类加速方法的对比

| 类别 | 时机 | 优点 | 缺点 |
|------|------|------|------|
| Token Pruning | Prefill 阶段 | 同时减少后续层的计算量和 KV Cache | 在前向传播前丢弃 token，视觉信息不可逆丢失 |
| KV Cache Compression | Decoding 阶段 | token 已完成完整前向传播，注意力机制已区分重要性 | 不影响 prefill 速度 |

**为什么 KV Cache 压缩更优？** 因为所有 token 已经完成完整前向传播，causal attention 已经在 token 间建立了重要性差异。选择性删除某些 token 对模型性能影响极小。Token pruning 在 prefill 前就丢弃 token，导致视觉信息永久丢失。

### 现有 KV Cache 方法的不足

作者深入分析了现有方法在 LVLM 场景下的问题：

**观察窗口选择不当**：如 SnapKV 使用全部文本 token 或连续局部文本 token 作为评估视觉 token 重要性的窗口，但不同文本 token 对同一视觉 token 的评价差异极大，导致投票机制引入大量噪声

**均匀预算分配次优**：不同层对视觉 token 的关注程度差异显著，均匀分配预算不是最优策略

### 核心观察

通过实验分析（Figure 1），作者发现：
- 使用全部文本 token 评估视觉 token 重要性时，一致性差
- 使用最后 16 个文本 token 评估时，结果不稳定
- 使用最后 16 个视觉 token 评估时，缺乏文本指导
- 只有约 10% 的视觉 token 对最终结果有显著正向影响（head effect）

## 方法详解

### 整体框架

AirCache 在 prefill 阶段完成后一次性压缩 KV Cache，包含两个核心组件：

1. **Elite Observation Window**：选择关键文本 token 构建精英观察窗口，评估视觉 token 重要性
2. **Layer-wise Budget Allocation**：基于重要性分数分布的强度和偏度，自适应分配各层的压缩预算

方法在 prefill 完成后执行，兼容主流 LVLM 架构。

### 关键设计一：Elite Observation Window（精英观察窗口）

**动机**：不同文本 token 对视觉 token 重要性的评价差异巨大。如果观察窗口中的文本 token 对同一视觉 token 给出截然不同的评分，投票机制就会引入大量噪声。跨模态差异会进一步放大这种噪声。

**方法**：利用文本 token 之间的自注意力来筛选关键文本 token。

首先将输入 prompt 的隐状态重组为视觉和文本部分：
$$X = \text{Concat}(X_v, X_t) \in \mathbb{R}^{(N_v + N_t) \times D}$$

计算文本单模态注意力矩阵：
$$\text{Att} = \text{Softmax}\left(\frac{Q_t K_t^T}{\sqrt{D}}\right) \in \mathbb{R}^{N_t \times N_t}$$

以最后一个文本 token 为参考，筛选出获得高注意力分数的关键文本 token：
$$k = \{j | \text{Att}[N_t-1, j] \geq \alpha \cdot \max \text{Att}[N_t-1, :]\}$$

其中 $\alpha \in [0,1]$ 为相关性阈值（实验中设为 0.9）。

然后用这些关键文本 token 的 query 与视觉 token + 关键文本 token 的 key 计算跨模态注意力：
$$A_{vtk} = \text{Softmax}\left(\frac{Q_{tk} K_{vtk}^T}{\sqrt{D}}\right)$$

最终视觉 token 的重要性分数通过沿文本维度平均池化得到：
$$I_v = \frac{1}{N_{tk}} \sum_{j=0}^{N_{tk}-1} A_{vtk}[j, :N_v]$$

**为什么精英窗口更优？**
- 精英窗口内的文本 token 倾向于对同一视觉 token 给出更一致的评价
- 减少了噪声干扰，提升了投票排名的准确性
- 计算复杂度更低，因为关键文本 token 数量远小于全部文本 token

### 关键设计二：Layer-wise KV Cache Budget Allocation（层级预算分配）

作者发现不同层对视觉信息的关注存在显著差异，需要差异化分配压缩预算。从两个维度量化：

**维度一：分布强度（Strength）**
$$s_t = \sum_{i=0}^{N_v-1} I_v[i]$$

排除文本 token 间的注意力，仅聚合文本 token 对所有视觉 token 的注意力分数总和。越大说明该层越重视视觉信息，应分配更多预算。

**维度二：分布偏度（Skewness）**
$$s_k = \frac{N_v}{(N_v-1)(N_v-2)} \sum_{i=1}^{N_v} \left(\frac{I_v[i] - \mu_{I_v}}{\sigma_{I_v}}\right)^3$$

偏度衡量注意力分布的"头部效应"程度。偏度高说明少数视觉 token 获得了极高关注（头部效应明显），该层的注意力分配更精准、更有信息量，应分配更多预算保护这些头部 token。

**最终预算公式**：
$$\hat{r} = \frac{1}{2}(s'_t + s'_k) \cdot r$$

其中 $s'_t, s'_k$ 分别是跨层归一化后的强度和偏度，$r$ 是原始预算。

**为什么结合强度和偏度？** 
- 仅用强度：知道哪些层重要，但不知道注意力是否精准聚焦
- 仅用偏度：知道注意力是否聚焦，但不知道层对视觉信息的总体关注度
- 二者结合：既考虑层的视觉关注量，又考虑关注的精准度，更全面

### 损失函数 / 训练策略

**无需训练**。AirCache 在 prefill 完成后一次性执行：
1. 对每层完成 prefill 后保存完整 KV Cache
2. 所有层计算完毕后，统一计算各层预算并压缩
3. 压缩后的 KV Cache 用于后续 decoding

相关性阈值 $\alpha = 0.9$，实验在 8×A100-80G 上进行。

## 实验关键数据

### 主实验

**VQA 数据集（LLaVA-OV-7B）— 视觉 KV Cache 保留比例 vs 性能**：

| 方法 | ChatQA 10% | InfoVQA 10% | DocVQA 10% | TextVQA 10% | ChatQA 1% | DocVQA 1% |
|------|-----------|------------|-----------|-----------|----------|----------|
| Full | 80.3 | 66.1 | 87.0 | 76.0 | 80.3 | 87.0 |
| H2O | 77.4 | 59.2 | 74.2 | 70.1 | 71.0 | 55.3 |
| SnapKV | 79.3 | 64.2 | 84.4 | 73.4 | 72.9 | 64.1 |
| PrefixKV | 78.2 | 61.1 | 80.5 | 72.7 | 70.9 | 55.4 |
| **AirCache** | **79.9** | **65.7** | **85.5** | **75.3** | **76.4** | **73.2** |

保留 10% 时性能差距控制在约 1% 以内，保留 1% 时远超其他方法（ChatQA +3.5 vs SnapKV，DocVQA +9.1 vs SnapKV）。

**多模型泛化（保留 10%）**：

| 模型 | 方法 | ChatQA | InfoVQA | DocVQA | TextVQA |
|------|------|--------|---------|--------|---------|
| InternVL2-8B | SnapKV | 80.4 | 72.3 | 90.1 | 73.9 |
| InternVL2-8B | **AirCache** | **81.7** | **72.6** | **90.0** | **77.0** |
| Qwen2-VL-7B | SnapKV | 81.6 | 74.9 | 87.2 | 83.1 |
| Qwen2-VL-7B | **AirCache** | **82.3** | **75.2** | **92.9** | **83.4** |

**推理加速效果**：

| Batch Size | Prompt Length | 50% 解码延迟降低 | 10% 解码延迟降低 | 10% 吞吐提升 |
|------------|-------------|-----------------|-----------------|-------------|
| 8 | 2k | -19.0% | -29.3% | +41.6% |
| 8 | 32k | -38.7% | -65.7% | +192.1% |
| 16 | 16k | -37.7% | -65.3% | +188.3% |

输入越长，加速效果越显著。Batch=16, Prompt=16k 时，10% 保留率下吞吐提升 **188.3%**。

### 消融实验

**精英观察窗口 vs 其他窗口（LLaVA-OV-7B, 1% 保留率）**：

| 观察窗口 | ChatQA | InfoVQA | DocVQA | TextVQA |
|---------|--------|---------|--------|---------|
| 连续窗口(16) | 70.4 | 56.6 | 61.3 | 55.9 |
| 连续窗口(32) | 72.9 | 57.8 | 64.1 | 58.2 |
| 全部文本 token | 72.2 | 58.4 | 65.7 | 57.0 |
| 视觉窗口(32) | 68.8 | 55.1 | 59.2 | 53.7 |
| **精英窗口(Ours)** | **76.4** | **62.5** | **73.2** | **67.1** |

精英窗口大幅领先，纯视觉窗口最差（缺乏文本指导）。

**层级预算分配消融**：

| 分配策略 | ChatQA | InfoVQA | DocVQA | TextVQA |
|---------|--------|---------|--------|---------|
| 均匀分配 | 72.2 | 57.5 | 69.9 | 62.4 |
| 金字塔分配(PyramidKV) | 69.6 | 54.9 | 55.8 | 52.6 |
| 仅强度 $s_t$ | 74.2 | 59.8 | 71.1 | 64.9 |
| 仅偏度 $s_k$ | 74.7 | 61.4 | 71.9 | 63.6 |
| **强度+偏度(Ours)** | **76.4** | **62.5** | **73.2** | **67.1** |

PyramidKV 的金字塔分配在 LLM 中表现好但在 LVLM 中效果差 → 多模态模型有独特的层级特性。

**与 Token Pruning 方法对比（保留 1%）**：

| 方法 | ChatQA | MMBench | MME | Prefill延迟 | Decode延迟 |
|------|--------|---------|-----|-----------|-----------|
| FastV | 16.9 | 33.6 | 786 | 5.4s | 11.7s |
| IVTP | 22.5 | 36.2 | 849 | 5.8s | 12.6s |
| **AirCache** | **76.4** | **82.3** | **1585** | 9.8s | 11.8s |

Token pruning 在极端压缩比下性能崩溃，AirCache 仍保持良好性能。

### 关键发现

1. **10% 视觉 KV Cache 即可**：仅保留 10% 的视觉 token，平均性能损失不超过 1%，验证了视觉 token 重要性分布的严重"头部效应"
2. **跨模态关联是关键**：精英观察窗口利用文本→文本自注意力筛选关键文本 token，再用关键文本评估视觉 token 重要性，比直接用全部文本或视觉 token 更稳定准确
3. **层级差异不可忽视**：LVLM 各层对视觉信息的关注差异巨大，PyramidKV 等 LLM 中有效的策略在 LVLM 中反而有害
4. **KV Cache 压缩优于 Token Pruning**：prefill 阶段的跨模态交互将视觉信息聚合到关键文本 token 中，即使大量删除视觉 KV Cache，关键信息仍可通过文本 token 保留

## 亮点与洞察

1. **精英观察窗口是核心创新**：通过文本自注意力筛选"有代表性的"文本 token 来评估视觉 token，解决了观察窗口一致性差的核心问题
2. **双维度预算分配**：强度（关注量）+ 偏度（关注精准度）的组合比任何单一指标或启发式规则都更优
3. **对 LVLM 注意力机制的深入分析**：揭示了不同层的视觉 token 重要性分布特征差异，为未来研究提供了重要参考
4. **实际部署价值高**：10% 保留率 + 不到 1% 性能损失 + 最高 66% 解码延迟降低，适合大规模部署
5. **KV Cache 压缩 vs Token Pruning 的深度对比**：清晰展示了两类方法的根本差异——prefill 阶段的信息聚合使得 KV Cache 压缩天然优于 Token Pruning

## 局限与展望

1. **Prefill 阶段无加速**：AirCache 在 prefill 后执行压缩，不减少 prefill 时间（甚至略增约 5-12%）
2. **一次性压缩**：在 prefill 完成后统一压缩所有层的 KV Cache，不支持 decoding 过程中的动态调整
3. **超参数敏感性**：相关性阈值 $\alpha=0.9$ 可能需要针对不同任务调优
4. **长输出场景**：文中指出 VQA 数据集（需要较长输出）更能体现 KV Cache 压缩的效果，短输出任务（如选择题）受 KV Cache 影响有限
5. **与 token pruning 的结合**：两种方法并非互斥，结合 prefill 加速和 decoding 加速可能获得更全面的提升

## 相关工作与启发

- **SnapKV**：使用局部窗口策略评估 token 重要性，AirCache 的精英窗口是对其的重要改进
- **PyramidKV**：在 LLM 中使用金字塔预算分配，但在 LVLM 中失效 → 说明跨模态模型有独特特性
- **VL-Cache**：利用视觉 token 稀疏性做层级预算分配，AirCache 进一步引入偏度维度
- **核心启发**：LVLM 中的 KV Cache 压缩不能简单套用 LLM 的方法，跨模态差异需要专门处理。文本自注意力可以作为一个有效的桥梁来建立跨模态关联

## 评分

- 新颖性: ⭐⭐⭐⭐ （精英观察窗口和双维度预算分配设计新颖）
- 实验充分度: ⭐⭐⭐⭐⭐ （3个模型架构、多种保留率、丰富消融、延迟/吞吐实测）
- 写作质量: ⭐⭐⭐⭐⭐ （动机分析清晰，消融设计层层递进）
- 价值: ⭐⭐⭐⭐ （实用性强，但 prefill 无加速限制了整体价值）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] FOLDER: Accelerating Multi-modal Large Language Models with Enhanced Performance](folder_accelerating_multi-modal_large_language_models_with_enhanced_performance.md)
- [\[ICCV 2025\] The Inter-Intra Modal Measure: A Predictive Lens on Fine-Tuning Outcomes in Vision-Language Models](the_inter-intra_modal_measure_a_predictive_lens_on_fine-tuning_outcomes_in_visio.md)
- [\[ICCV 2025\] Scaling Inference-Time Search with Vision Value Model for Improved Visual Comprehension](scaling_inferencetime_search_with_vision_value_model_for_imp.md)
- [\[ACL 2025\] MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference](../../ACL2025/multimodal_vlm/madakv_adaptive_modality-perception_kv_cache_eviction_for_efficient_multimodal_l.md)
- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)

</div>

<!-- RELATED:END -->
