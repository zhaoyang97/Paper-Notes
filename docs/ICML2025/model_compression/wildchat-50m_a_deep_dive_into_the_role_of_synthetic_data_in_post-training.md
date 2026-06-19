---
title: >-
  [论文解读] WildChat-50m: A Deep Dive Into the Role of Synthetic Data in Post-Training
description: >-
  [ICML 2025][模型压缩][合成数据] 构建迄今最大的公开聊天数据集 WildChat-50m（50+ 开源模型 × 100万+ 对话 = 1.25 亿条转录），系统研究不同数据生成模型（DGM）的合成数据质量，并设计 Re-Wild SFT 混合方案，仅用 Tulu-3 SFT 数据量的 40% 即在多项基准上超越其表现。
tags:
  - "ICML 2025"
  - "模型压缩"
  - "合成数据"
  - "SFT"
  - "数据生成模型"
  - "LLM后训练"
  - "数据混合"
---

# WildChat-50m: A Deep Dive Into the Role of Synthetic Data in Post-Training

**会议**: ICML 2025  
**arXiv**: [2501.18511](https://arxiv.org/abs/2501.18511)  
**代码**: [https://github.com/penfever/wildchat-50m](https://github.com/penfever/wildchat-50m)  
**领域**: 模型压缩/LLM后训练  
**关键词**: 合成数据, SFT, 数据生成模型, LLM后训练, 数据混合

## 一句话总结

构建迄今最大的公开聊天数据集 WildChat-50m（50+ 开源模型 × 100万+ 对话 = 1.25 亿条转录），系统研究不同数据生成模型（DGM）的合成数据质量，并设计 Re-Wild SFT 混合方案，仅用 Tulu-3 SFT 数据量的 40% 即在多项基准上超越其表现。

## 研究背景与动机

### 1. LLM 后训练的重要性

LLM 后训练（SFT、DPO、蒸馏）是解锁模型能力的关键步骤。OpenAI 的 test-time scaling 和 DeepSeek 的推理模型都依赖高质量合成数据。但开源社区在数据策划方面远落后于工业实验室。

### 2. 核心挑战

- 大规模公开合成数据集稀缺，阻碍了对数据生成模型质量的系统比较
- 不同 DGM 生成的回复质量差异有多大？如何选择最优 DGM？
- 现有 SFT 数据混合方案（如 Tulu-3）复杂且体量大，能否找到更高效替代？

### 3. 本文切入

用 50+ 开源模型在 WildChat-1M 的 100 万+ prompt 上生成回复，构建 1.25 亿条转录的超大规模数据集，支持系统化的 DGM 比较和 SFT 实验。

## 方法详解

### 整体框架

1. **数据收集**：在 12×8 H100 集群上用 VLLM 推理 50+ 模型，对 WildChat prompt 生成多轮回复
2. **数据分析**：比较各模型的吞吐效率、回复相似度、质量指标
3. **SFT 实验**：设计 Re-Wild 数据混合方案，用 Llama-3.1-8B-Base 做 SFT 并在多基准评测

### 关键设计

#### 1. 大规模多模型数据收集

- **规模**：54 个 DGM（19 个预训练模型 + 35 个微调变体），参数量 0.5B-104B
- **统一环境**：同一硬件（H100）+ 同一推理框架（VLLM），确保公平比较
- **总成本**：约 10000 H100-hours
- 最大模型用 FP8 量化推理，其余用 bfloat16

#### 2. DGM 质量分析

- **吞吐效率**：最快（Llama-2-7B: 37357 tok/s）比最慢（Qwen2.5-72B: 3163 tok/s）快 10 倍+
- **回复相似度**：不同 LLM 的回复异常相似——虽然预训练/后训练数据不完全重叠
- **质量排序**：通过下游 SFT 性能间接衡量 DGM 的合成数据质量

#### 3. Re-Wild 数据混合

| 数据源 | 数量 |
|--------|------|
| WildChat-Q72（Qwen2.5-72B 回复） | 246,750 |
| MMLU Auxiliary Train | 99,800 |
| Tulu 3 Persona Hub Algebra | 20,000 |
| **总计** | **~366K** |

仅 Tulu-3 SFT 数据量的 ~40%。设计原则：互补技能（聊天 + 知识 + 数学）。训练：AdamW, lr=2e-5, 1 epoch, cosine scheduler, 4×H100, ~5.5 小时。

## 实验关键数据

### 主实验：Re-Wild vs 基线

| 方法 | 数据量 | 基准表现趋势 | 说明 |
|------|--------|------------|------|
| Tulu-3 SFT（Allen AI） | ~900K | 基准 | 复杂多源混合 |
| **Re-Wild（本文）** | **~366K** | **超越 Tulu-3** | 40% 数据量 |
| L8B:Q72 单源 | 250K | 中高 | 仅聊天数据 |
| L8B:L8I 自蒸馏 | 250K | 中 | 小模型蒸馏效果有限 |

论文 Fig.1 展示 Re-Wild 在 9 个基准的加权平均上超越所有基线。

### DGM 选择消融

| DGM | SFT 后均分趋势 | 参数量 | 推理速度 |
|-----|---------------|--------|---------|
| Qwen2.5-72B-Instruct | 最高 | 72B | 3163 tok/s |
| Llama-3.3-70B | 次高 | 70B | 中 |
| Cohere-CRP-104B | 中高 | 104B | 中低 |
| Qwen2-7B-Instruct | 中 | 7B | 高 |
| Llama-2-7B-Chat | 低 | 7B | 37357 tok/s |

### 关键发现

- 大模型（72B+）作为 DGM 的合成数据质量显著优于小模型
- 不同 LLM 的回复惊人相似，但细微差异在下游 SFT 中被放大
- 简单的三源混合可以超过复杂多源方案——数据量不是越多越好
- Re-Wild 用 40% 数据量超过 Tulu-3

## 亮点与洞察

- **数据集规模史无前例**：比 WildChat-1M 大 50 倍+，首个支持系统化 DGM 比较的数据集
- **"少即是多"的数据策划**：精选 DGM 和互补数据源比堆量更重要
- **实用 DGM 选择指南**：为学术实验室提供了"用哪个模型生成训练数据"的量化参考
- **完全开源**：数据集、代码、SFT 方案全部公开

## 局限与展望

- 仅做 SFT 实验，DPO/RLHF 等偏好对齐阶段待探索
- 回复相似度的发现未深入分析差异来源
- 最大模型用 FP8 量化推理，量化对回复质量的影响未隔离
- 仅在 Llama-3.1-8B-Base 上验证 SFT

## 相关工作与启发

- **vs Tulu-3**：更复杂的多源混合但 Re-Wild 以 40% 数据量超越
- **vs DeepSeek 蒸馏**：工业级蒸馏不公开数据，本文提供开源替代
- **vs 数据策划研究**：首次在 50+ DGM 规模上系统比较合成数据质量

## 评分

- 新颖性: ⭐⭐⭐⭐ 数据集规模和系统化比较是核心贡献
- 实验充分度: ⭐⭐⭐⭐⭐ 50+ 模型 × 9 基准 × 多消融
- 写作质量: ⭐⭐⭐⭐ 结构清晰，分析深入
- 价值: ⭐⭐⭐⭐⭐ 对开源 LLM 后训练社区有重大实用价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[ICCV 2025\] StolenLoRA: Exploring LoRA Extraction Attacks via Synthetic Data](../../ICCV2025/model_compression/stolenlora_exploring_lora_extraction_attacks_via_synthetic_data.md)
- [\[ICML 2025\] Merge-Friendly Post-Training Quantization for Multi-Target Domain Adaptation](merge-friendly_post-training_quantization_for_multi-target_domain_adaptation.md)
- [\[ECCV 2024\] MetaAug: Meta-Data Augmentation for Post-Training Quantization](../../ECCV2024/model_compression/metaaug_meta-data_augmentation_for_post-training_quantization.md)
- [\[ICML 2025\] FGFP: A Fractional Gaussian Filter and Pruning for Deep Neural Networks Compression](fgfp_a_fractional_gaussian_filter_and_pruning_for_deep_neural_networks_compressi.md)

</div>

<!-- RELATED:END -->
