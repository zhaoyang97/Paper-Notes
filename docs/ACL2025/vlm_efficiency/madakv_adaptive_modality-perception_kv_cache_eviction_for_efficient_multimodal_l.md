---
title: >-
  [论文解读] MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference
description: >-
  [多模态VLM] 本文提出MadaKV，一种模态感知的KV缓存逐出策略，通过模态偏好自适应（MPA）和层级压缩补偿（HCC）两个组件，在保持多模态长上下文任务性能的同时，显著降低KV缓存内存占用（80-95%）和解码延迟（1.3-1.5倍加速）。 - KV缓存问题：自回归生成中，KV缓存存储所有历史token的Key和Val…
tags:
  - "多模态VLM"
---

# MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference

| 属性 | 内容 |
|------|------|
| 标题 | MadaKV: Adaptive Modality-Perception KV Cache Eviction for Efficient Multimodal Long-Context Inference |
| 会议 | ACL2025 |
| arXiv | [2506.15724](https://arxiv.org/abs/2506.15724) |
| 代码 | - |
| 领域 | Multimodal VLM / Efficient Inference |
| 关键词 | KV Cache Eviction, Multimodal LLM, Modality Preference, Long-Context, Inference Efficiency |

## 一句话总结

本文提出MadaKV，一种模态感知的KV缓存逐出策略，通过模态偏好自适应（MPA）和层级压缩补偿（HCC）两个组件，在保持多模态长上下文任务性能的同时，显著降低KV缓存内存占用（80-95%）和解码延迟（1.3-1.5倍加速）。

## 研究背景与动机

- **KV缓存问题**：自回归生成中，KV缓存存储所有历史token的Key和Value以避免重复计算，但随序列长度增加，内存开销急剧增长
- **现有方法的不足**：StreamingLLM、H2O、SnapKV等KV缓存逐出方法针对单模态（纯文本）设计，不感知模态信息，在多模态场景中效果欠佳
- **多模态特殊性**：
    - 不同模态的信息密度不同：文本token编码语义概念简洁，视觉token需要大量空间表示精细空间信息
    - 注意力头对不同模态有不同偏好（modality preference）
    - 不同任务中模态的重要性差异显著（如文本搜索更关注文本，图像检索更关注视觉）
- **LOOK-M的局限**：虽然针对多模态但经验性地固定模态优先级（优先逐出视觉token），忽视了模态重要性的动态变化

## 方法详解

### 整体框架

MadaKV是一个即插即用（plug-and-play）的模态自适应KV缓存压缩策略，包含两个核心组件：

### 1. 模态偏好自适应（Modality Preference Adaptation, MPA）

**模态偏好度量**：通过代理token（proxy tokens，取prompt末尾几个token）计算每个token的重要性：

$$\psi(i) = \sum_{j \in \mathcal{P}} \alpha_{j \to i}$$

然后分别对视觉token和文本token计算总偏好权重：

$$w_v = \sum_{i \in X_v} \psi(i), \quad w_t = \sum_{i \in X_t} \psi(i)$$

**模态预算分配**：根据偏好权重按比例分配每个注意力头中各模态的缓存预算：

$$\varphi_v^{l,h} = \frac{w_v}{w_v + w_t} \varphi^l, \quad \varphi_t^{l,h} = \frac{w_t}{w_v + w_t} \varphi^l$$

每个注意力头独立进行模态级别的KV缓存逐出，而非统一对待所有token。

### 2. 层级压缩补偿（Hierarchical Compression Compensation, HCC）

**动机**：不同层的注意力模式差异显著——浅层注意力分散、深层集中于少量token、最终层又趋于均匀。不同层需要不同的缓存预算。

**稀疏度计算**：对每个注意力头，找到能覆盖 $\theta$ 比例总重要性所需的最少token数：

$$k_v^{l,h} = \min\{|\mathcal{C}_v| \mid \sum_{i \in \mathcal{C}_v} \psi(i) \geq \theta w_v\}$$

**层间补偿**：当前层的预算补偿值为实际需求与分配预算之差的累积：

$$K^l = \sum_{h=1}^{H}(k_v^{l,h} + k_t^{l,h} - \varphi^l)$$

下一层预算根据补偿值调整：

$$\varphi^{l+1} = \varphi^l - \frac{K^l}{L - l}$$

正补偿值意味着当前层超支，后续层需分担削减；负值意味着省了预算可留给更需要的层。

### 关键观察

论文通过LLaVA-v1.5-7B在MileBench上的实验得到四个维度的观察：

| 维度 | 观察 |
|------|------|
| Token级别 | 仅20%的token即捕获约90%的注意力分数；文本token注意力集中，视觉token注意力分散 |
| 注意力头级别 | 不同头对模态有不同偏好，表现为分配给不同模态的注意力分数比例差异大 |
| 层级别 | 初始层注意力均匀分布，中间层集中，最终层又趋均匀——首尾层需保留更多KV |
| 任务级别 | 文本搜索任务中文本token更重要，图像检索中视觉token更重要 |

## 实验

### 主实验：MileBench上的性能比较

在LLaVA-v1.5-7B上（20%缓存预算），MadaKV平均准确率28.22%，接近Full Cache的28.59%：

| 方法 | TN | IEdit | MMCoQA | STD | ALFRED | 平均 |
|------|-----|-------|--------|-----|--------|------|
| Full Cache | 9.68 | 7.98 | 33.50 | 16.32 | 15.18 | 28.59 |
| StreamingLLM | 3.12 | 3.59 | 26.00 | 11.77 | 3.73 | 20.91 |
| H2O | 2.50 | 5.51 | 28.00 | 15.73 | 14.86 | 25.80 |
| SnapKV | 3.27 | 6.03 | 29.00 | 14.82 | 14.40 | 26.43 |
| LOOK-M | 3.34 | 6.51 | 29.50 | 15.79 | 13.96 | 26.47 |
| **MadaKV** | **9.38** | **6.97** | **31.00** | **15.85** | **15.06** | **28.22** |

在TN（Text Needle）任务上MadaKV优势最为突出（9.38 vs LOOK-M的3.34），因为该任务需要在大量视觉干扰中定位文本信息。

### Qwen2.5-VL-7B结果

MadaKV在Qwen2.5-VL-7B上同样有效，平均62.74% vs Full Cache的63.34%，仅下降0.6个百分点。

### 不同缓存预算影响

- 50%缓存预算时大多数方法接近Full Cache性能
- MadaKV在所有预算水平上一致优于基线
- TN任务中，MadaKV用20%预算的性能 = LOOK-M用60%预算的性能
- 缓存<10%时MadaKV优势尤为明显

### 效率分析

| 配置 | 解码延迟 | KV缓存显存 |
|------|---------|-----------|
| Full Cache | 27.85 ms/token | 1.63 GiB |
| MadaKV (20%) | 19.57 ms/token | 0.41 GiB |
| MadaKV (5%) | 17.16 ms/token | 0.16 GiB |

20%预算下解码速度提升1.42倍,显存减少75%。

### 消融实验

| MPA | HCC | TN | IEdit | ALFRED |
|-----|-----|----|-------|--------|
| ✘ | ✘ | 2.47 | 3.55 | 14.32 |
| ✔ | ✘ | 6.58 | 5.72 | 14.86 |
| ✘ | ✔ | 5.51 | 5.19 | 14.61 |
| ✔ | ✔ | **9.38** | **6.97** | **15.06** |

两个组件互补，MPA贡献更大（尤其在TN任务上提升4.11），HCC进一步提升3.07。

## 亮点与洞察

1. **模态感知的关键创新**：与现有方法对所有token一视同仁不同，MadaKV首次让KV缓存逐出"看到"模态信息，根据每个注意力头的模态偏好差异化处理
2. **即插即用设计**：不需要模型微调，可直接应用于任何基于Transformer的MLLM
3. **层间协调机制**：HCC通过跨层预算补偿实现全局优化，避免单层过度压缩导致的错误级联传播
4. **实验设计全面**：覆盖多种模型（LLaVA-v1.5-7B/13B、Qwen2.5-VL-7B）、多种任务类型、多种缓存预算级别
5. **代理token的巧妙使用**：选择prompt末尾token作为代理来评估token重要性，因为它们通常是任务相关问题

## 局限性

1. 仅在7B/13B规模模型上验证，未测试34B/70B大模型
2. 仅覆盖视觉和文本两种模态，未扩展到视频、音频等
3. 未在极长上下文（如100K+ tokens）场景测试
4. MPA中代理token的选择策略比较粗糙（简单取末尾），可能不是最优

## 相关工作

- **KV缓存优化**：量化方法（FP16→INT8）、逐出策略（StreamingLLM, H2O, SnapKV, PyramidKV）
- **多模态KV缓存**：LOOK-M优先逐出视觉token，FastGen和H2O的轻量分析
- **高效注意力**：稀疏注意力（Longformer）、低秩近似
- **MLLM推理优化**：激活检查点、offloading、动态内存管理

## 评分 ⭐⭐⭐⭐

**优点**：问题定义清晰，观察扎实（四个维度的注意力分析），方法直觉且有效，实验覆盖全面，effect明确（80-95%内存减少，1.3-1.5x加速）。

**不足**：核心思想相对直接（按模态注意力分数比例分配预算）；仅在MileBench一个基准上测试多模态长上下文能力，泛化性有待验证。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Physics Context Builders: A Modular Framework for Physical Reasoning in Vision-Language Models](../../ICCV2025/multimodal_vlm/physics_context_builders_a_modular_framework_for_physical_reasoning_in_vision-la.md)
- [\[ACL 2025\] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale](mammoth_vl_multimodal_reasoning.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](spatialmqa_mllm_spatial_relations.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[ACL 2025\] Multimodal Coreference Resolution for Chinese Social Media Dialogues: Dataset and Benchmark Approach](multimodal_coreference_resolution_for_chinese_social_media_dialogues_dataset_and.md)

</div>

<!-- RELATED:END -->
