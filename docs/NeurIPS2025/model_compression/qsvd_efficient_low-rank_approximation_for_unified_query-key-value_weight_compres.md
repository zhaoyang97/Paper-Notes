---
title: >-
  [论文解读] QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression
description: >-
  [NeurIPS 2025 Spotlight][模型压缩][VLM压缩] 提出QSVD方法，通过对QKV联合权重矩阵的SVD分解共享下投影矩阵来减少KV缓存和计算开销，结合基于重要性评分的自适应秩分配和量化技术，在VLM上实现超过10%的精度提升且硬件成本更低。 视觉语言模型（VLM）在图像描述、视觉问答等任务中表现出色…
tags:
  - "NeurIPS 2025 Spotlight"
  - "模型压缩"
  - "VLM压缩"
  - "SVD"
  - "KV缓存"
  - "量化"
  - "低秩近似"
---

# QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.16292](https://arxiv.org/abs/2510.16292)  
**代码**: [https://github.com/SAI-Lab-NYU/QSVD](https://github.com/SAI-Lab-NYU/QSVD)  
**领域**: 多模态VLM / 模型压缩  
**关键词**: VLM压缩, SVD, KV缓存, 量化, 低秩近似

## 一句话总结

提出QSVD方法，通过对QKV联合权重矩阵的SVD分解共享下投影矩阵来减少KV缓存和计算开销，结合基于重要性评分的自适应秩分配和量化技术，在VLM上实现超过10%的精度提升且硬件成本更低。

## 研究背景与动机

视觉语言模型（VLM）在图像描述、视觉问答等任务中表现出色，但面临巨大的计算成本挑战——高维视觉和文本数据的联合处理要求大量计算，自回归token生成对显存带宽形成瓶颈。

**核心痛点**：

**KV缓存占用巨大**：Multi-Head Attention中Key和Value矩阵随序列长度线性增长，成为推理速度的主要瓶颈

**现有SVD压缩方案效率不足**：传统方法对Q、K、V矩阵分别做SVD，产生3组独立的下投影矩阵，参数和计算冗余

**量化与SVD的兼容性差**：SVD分解后的中间表示 $C_{qkv}$ 存在严重的通道级异常值，阻碍低精度量化

**核心矛盾**：如何在保持VLM精度的前提下，同时降低权重参数量、KV缓存大小和计算FLOPs？

**本文切入角度**：受DeepSeek-v3的Multi-Head Latent Attention启发，对QKV的**联合权重矩阵**做SVD，让Q、K、V共享一个下投影矩阵，从而只需缓存一份低维中间表示即可重构K和V。进一步结合自适应秩分配和兼容SVD的量化方案。

## 方法详解

### 整体框架

QSVD包含三个核心组件：(1) QKV联合SVD压缩 (2) 基于重要性评分的跨层秩分配 (3) 适配低秩VLM的后训练量化。

### 关键设计

1. **QKV联合SVD分解**：

    - 将 $W_Q, W_K, W_V \in \mathbb{R}^{E \times E}$ 拼接为 $W_{\text{concat}} \in \mathbb{R}^{E \times 3E}$
    - 对拼接矩阵做低秩SVD：$W_{\text{concat}} \approx W_r^d \times \Sigma_r \times W_r^u$
    - 分割为共享下投影 $W_{qkv}^d \in \mathbb{R}^{E \times r}$ 和三个独立上投影 $W_q^u, W_k^u, W_v^u \in \mathbb{R}^{r \times E}$
    - 推理时只需缓存 $C_{qkv} = X \cdot W_{qkv}^d$（大小 $r \times L$），K和V可从 $C_{qkv}$ 按需重构
    - 对比：传统分别SVD需要 $6rE$ 参数和 $2rL$ 缓存；QSVD仅需 $4rE$ 参数和 $rL$ 缓存

2. **基于重要性评分的跨层秩分配**：

    - 每个奇异值 $\sigma_i$ 被截断时对训练损失的影响通过一阶展开估计：$\Delta L_{\sigma_i} \approx \langle \Delta W_{\sigma_i}, G_W \rangle_F$
    - 重要性评分：$\hat{I}_{\sigma_i} = \frac{1}{N}\sum_{n=1}^N \sigma_i^2 [U^T G_W^{(n)} V]_{(i,i)}^2$
    - 关键优化：通过数学变换避免构造完整的 $\Delta W_{\sigma_i}$ 矩阵，内存从 $O(E^3)$ 降至 $O(E^2)$
    - 全局排序所有层的奇异值，保留top-k个最重要的，实现跨层最优秩分配

3. **兼容SVD的量化方案**：

    - SVD分解后 $C_{qkv} = X W_r^d \Sigma_r^\beta$ 存在严重通道级异常值（因 $\Sigma_r^\beta$ 的值域差异大）
    - 引入两个正交矩阵 $H_1, H_2$：$Y = (XH_1^\top)(H_1 W_{qkv}^d H_2^\top)(H_2 W_{qkv}^u)$
    - 核心创新：将 $\beta$ 作为可学习参数，在校准集上优化以最小化量化误差 $\min_\beta \sum_d \|Y_d - Y_d'\|^2$
    - $\beta$ 控制奇异值在上下投影之间的分配,直接影响 $C_{qkv}$ 的异常值分布

### 损失函数 / 训练策略

无需训练。所有操作为后训练压缩（PTQ），仅需256个校准样本（取自ScienceQA训练集）用于重要性评分计算和 $\beta$ 优化。

## 实验关键数据

### 主实验

**SVD压缩精度对比（FP16, ScienceQA-IMG）**

| 方法 | SmolVLM 2B (R2=37.5%) | LLaVA-Next 7B (R2=22.5%) | LLaVA-v1.5 13B (R2=22.5%) |
|------|----------------------|--------------------------|--------------------------|
| ASVD | 53.84% | 50.72% | 64.70% |
| SVD-LLM | 65.89% | 65.94% | 71.44% |
| **QSVD-noQ** | **83.78%** | **69.91%** | **71.79%** |
| FP16基线 | 84.53% | 69.51% | 71.78% |

**量化+SVD联合压缩对比（LLaVA-v1.5 7B）**

| 方法 | W8A8精度 | W8A4精度 | W4A4精度 | R2 |
|------|---------|---------|---------|-----|
| DuQuant | 66.53% | 57.36% | 52.56% | 50%/25%/25% |
| QVLM | 64.65% | 55.24% | 51.12% | 50%/25%/25% |
| QASVD | 52.95% | 41.92% | 12.61% | 50%/25%/25% |
| **QSVD** | **67.57%** | **65.61%** | **55.16%** | **18.75%/9.38%/9.38%** |

### 消融实验

| 配置 | ScienceQA | VizWiz | 说明 |
|------|-----------|--------|------|
| QSVD完整版（W8A4） | 65.61% | 52.18% | 基线 |
| 无$\beta$优化 | 精度显著下降 | 显著下降 | $\beta$调节异常值分布至关重要 |
| 均匀秩分配（替代重要性评分） | 精度下降 | 下降 | 跨层自适应秩分配有效 |
| 分别SVD（同等硬件成本） | 50.72% | 47.78% | 联合SVD显著优于分别SVD |

### 关键发现

- QSVD在SmolVLM 2B上R2=37.5%（即KV缓存降至原来37.5%）时精度仅从84.53%降至83.78%，几乎无损
- 在W8A4量化+SVD联合压缩下，QSVD以**仅9.38%的KV缓存**实现65.61%精度（ScienceQA），DuQuant在50%KV缓存下为57.36%
- QASVD（ASVD+QuaRot）在W4A4下完全崩溃（12.61%），QSVD保持55.16%——证明β优化对量化兼容性的关键作用
- 跨模型（2B到13B）一致性好：QSVD在所有评估的5个VLM上都显著优于基线
- KV缓存大小的减少直接转化为推理加速

## 亮点与洞察

- **QKV共享下投影是核心创新**：受DeepSeek MLA启发但将其作为后训练压缩技术，无需重新训练
- **重要性评分的计算效率优化**巧妙：利用 $\hat{I}_{\sigma_i} = \frac{1}{N}\sum \sigma_i^2 [U^T G_W V]_{(i,i)}^2$ 避免 $O(E^3)$ 内存
- **$\beta$ 参数的引入**解决了SVD+量化不兼容的关键问题——控制奇异值在上下投影间的分配比例直接影响中间激活的异常值分布
- 整体方案模块化程度高，SVD和量化可独立或联合使用

## 局限与展望

- 仅对self-attention层的QKV权重做SVD，未压缩FFN层（FFN通常占更多参数）
- 校准数据集依赖（使用ScienceQA训练样本），可能对其他任务不是最优
- 未评估在生成式任务（如图像描述）上的效果，仅评估分类/VQA
- 与LoRA等训练时压缩方法的对比缺失
- 实际GPU加速数据较少，主要报告理论FLOPs/缓存减少

## 相关工作与启发

- 与DeepSeek MLA的关系：MLA在训练时学习低秩投影，QSVD在推理时通过SVD实现类似效果，适用于已有模型
- 与Palu、ASVD等KV缓存压缩工作互补——QSVD在更低硬件成本下实现更高精度
- $\beta$ 优化的思路可推广到其他需要在低秩分解和量化间取得平衡的场景

## 评分

- 新颖性: ⭐⭐⭐⭐ QKV联合SVD + 可学习β + 重要性评分的组合新颖且有效
- 实验充分度: ⭐⭐⭐⭐ 5个VLM模型、3个数据集、多种量化配置，但缺少生成任务评估
- 写作质量: ⭐⭐⭐⭐ 公式推导清晰，效率分析详尽，图表直观
- 价值: ⭐⭐⭐⭐ 对VLM部署有直接实用价值，方法简洁高效

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SCOPE: Optimizing Key-Value Cache Compression in Long-context Generation](../../ACL2025/model_compression/scope_optimizing_key-value_cache_compression_in_long-context_generation.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](../../ICLR2026/model_compression/revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ICLR 2026\] FreqKV: Key-Value Compression in Frequency Domain for Context Window Extension](../../ICLR2026/model_compression/freqkv_key-value_compression_in_frequency_domain_for_context_window_extension.md)
- [\[NeurIPS 2025\] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)
- [\[NeurIPS 2025\] KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction](kvzip_query-agnostic_kv_cache_compression_with_context_reconstruction.md)

</div>

<!-- RELATED:END -->
